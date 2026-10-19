"""Dormant E3 cron delivery runtime.

This module is the narrow seam between the durable E2 outbox and a caller
supplied sender.  It deliberately does not discover channels, providers,
message buses, timers, or a default sender.  A destination is sent only after
the E2 claim has been durably marked ``sending``; the sender then receives a
detached immutable request and may perform arbitrary asynchronous work,
including reading the outbox through its own authority.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast, runtime_checkable

from nanobot.cron.delivery_store import (
    CronDeliveryOutboxStore,
    DeliveryClaim,
    DeliveryOutboxRecord,
    DeliverySendFence,
)
from nanobot.cron.occurrence_store import CronOccurrenceStore

_MAX_IDENTIFIER_BYTES = 512
_MAX_REASON_BYTES = 512
_MAX_RECEIPT_BYTES = 64 * 1024
_MAX_MESSAGE_BYTES = 256 * 1024
_MAX_BUNDLE_BYTES = 1_048_576
_DIGEST_HEX_LENGTH = 64


CRON_DELIVERY_RUNTIME_UNAVAILABLE_DIAGNOSTIC = (
    "cron delivery runtime is unavailable; inject an explicit CronDeliveryRuntime "
    "and sender substrate before using delivery operations"
)


class CronDeliveryRuntimeError(RuntimeError):
    """Base error raised by the dormant delivery runtime."""


class CronDeliveryRuntimeUnavailableError(CronDeliveryRuntimeError):
    """The runtime or its explicitly required authority is not configured."""


class DispatchRequestError(CronDeliveryRuntimeError, ValueError):
    """A sender request is not a strict detached text-only request."""


class InvalidSenderResultError(CronDeliveryRuntimeError, ValueError):
    """A sender returned an unclassified or malformed result."""


class PartialBundleError(CronDeliveryRuntimeError):
    """A sender signals that only part of a message bundle was processed."""

    def __init__(self, reason_code: str = "partial_bundle") -> None:
        self.reason_code = _bounded_text(reason_code, label="partial bundle reason")
        super().__init__(self.reason_code)


class CronDeliverySettlementError(CronDeliveryRuntimeError):
    """The sender outcome could not be durably recorded.

    ``classification`` is retained even when the sender raised an exception,
    so an operator can reconcile the exact attempt without invoking a sender
    again.  The original sender exception, when any, is available as
    ``original_error`` and is also preserved as the exception cause.
    """

    def __init__(
        self,
        message: str,
        *,
        claim: DeliveryClaim,
        classification: SenderResult,
        original_error: BaseException | None = None,
    ) -> None:
        self.claim = claim
        self.classification = classification
        self.result = classification
        self.sender_result = classification
        self.outcome = classification
        self.original_error = original_error
        self.send_error = original_error
        super().__init__(message)


class CronDeliveryReentrancyError(CronDeliveryRuntimeError):
    """A sender tried to dispatch the claim whose sender is currently running."""


def _bounded_text(value: object, *, label: str, maximum: int = _MAX_IDENTIFIER_BYTES) -> str:
    if not isinstance(value, str) or not value:
        raise DispatchRequestError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > maximum:
        raise DispatchRequestError(f"{label} exceeds the {maximum}-byte limit")
    return value


def _bounded_reason(value: object) -> str:
    return _bounded_text(value, label="reason_code", maximum=_MAX_REASON_BYTES)


def _assert_json_value(value: object, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DispatchRequestError(f"{label} contains a non-finite JSON number")
        return
    if isinstance(value, Mapping):
        value = cast(Mapping[object, object], value)
        for key, item in value.items():
            if not isinstance(key, str):
                raise DispatchRequestError(f"{label} contains a non-string JSON key")
            _assert_json_value(item, label=label)
        return
    if isinstance(value, (list, tuple)):
        value = cast(list[object] | tuple[object, ...], value)
        for item in value:
            _assert_json_value(item, label=label)
        return
    raise DispatchRequestError(f"{label} contains a non-JSON value")


def _detach_json(value: object, *, label: str) -> object:
    _assert_json_value(value, label=label)
    if isinstance(value, Mapping):
        # _assert_json_value recursively checked string keys on this path.
        value = cast(Mapping[str, object], value)
        return MappingProxyType(
            {key: _detach_json(item, label=label) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        value = cast(list[object] | tuple[object, ...], value)
        return _FrozenSequence(_detach_json(item, label=label) for item in value)
    return value


class _FrozenSequence(tuple[object, ...]):
    """Immutable JSON array retaining ordinary list/tuple equality."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(cast(list[object] | tuple[object, ...], other))
        return NotImplemented


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        value = cast(Mapping[object, object], value)
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        value = cast(tuple[object, ...], value)
        return [_plain_json(item) for item in value]
    if isinstance(value, list):
        value = cast(list[object], value)
        return [_plain_json(item) for item in value]
    return value


def _encoded_size(value: object, *, label: str) -> int:
    try:
        encoded = json.dumps(
            _plain_json(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise DispatchRequestError(f"{label} must contain finite JSON data") from exc
    return len(encoded)


def _freeze_receipt(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise InvalidSenderResultError("confirmed delivery requires a non-empty receipt object")
    value = cast(Mapping[object, object], value)
    try:
        detached = _detach_json(value, label="receipt")
    except DispatchRequestError as exc:
        raise InvalidSenderResultError(str(exc)) from exc
    if not isinstance(detached, Mapping):  # pragma: no cover - guarded above
        raise InvalidSenderResultError("confirmed delivery requires a receipt object")
    detached = cast(Mapping[str, object], detached)  # _detach_json checked all string keys.
    try:
        receipt_size = _encoded_size(detached, label="receipt")
    except DispatchRequestError as exc:
        raise InvalidSenderResultError(str(exc)) from exc
    if receipt_size > _MAX_RECEIPT_BYTES:
        raise InvalidSenderResultError(
            f"receipt exceeds the {_MAX_RECEIPT_BYTES}-byte limit"
        )
    return detached


@dataclass(frozen=True, slots=True, eq=False)
class DispatchMessage(Mapping[str, object]):
    """One detached text-only message in a dispatch bundle."""

    content: str
    media: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.content), str):
            raise DispatchRequestError("message content must be text")
        if len(self.content.encode("utf-8")) > _MAX_MESSAGE_BYTES:
            raise DispatchRequestError(
                f"message content exceeds the {_MAX_MESSAGE_BYTES}-byte limit"
            )
        if self.media not in ((), []):
            raise DispatchRequestError("delivery messages must not carry media")
        object.__setattr__(self, "media", ())

    def __getitem__(self, key: str) -> object:
        if key == "content":
            return self.content
        if key == "media":
            return []
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "content"
        yield "media"

    def __len__(self) -> int:
        return 2

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DispatchMessage):
            return self.content == other.content
        if isinstance(other, Mapping):
            return dict(self) == dict(cast(Mapping[object, object], other))
        return NotImplemented

    @classmethod
    def from_value(cls, value: object) -> DispatchMessage:
        if isinstance(value, cls):
            return cls(value.content, ())
        if not isinstance(value, Mapping) or set(cast(Mapping[object, object], value)) != {"content", "media"}:
            raise DispatchRequestError(
                "dispatch messages must have exactly content and media fields"
            )
        value = cast(Mapping[str, Any], value)  # Exact raw message wire fields checked above.
        if not isinstance(value["media"], (list, tuple)) or value["media"] != []:
            raise DispatchRequestError("delivery messages must not carry media")
        return cls(value["content"], ())


@dataclass(frozen=True, slots=True)
class DispatchRequest:
    """Immutable sender input derived from one durably sending E2 attempt."""

    outbox_id: str
    destination_id: str
    attempt: int
    action_id: str
    request_digest: str
    channel: str
    to: str
    messages: tuple[DispatchMessage, ...]

    if TYPE_CHECKING:
        # Construction accepts raw bundles; __post_init__ validates and narrows
        # them to the immutable concrete messages stored by this dataclass.
        def __init__(
            self,
            outbox_id: str,
            destination_id: str,
            attempt: int,
            action_id: str,
            request_digest: str,
            channel: str,
            to: str,
            messages: Sequence[object],
        ) -> None:
            ...

    def __post_init__(self) -> None:
        for name in ("outbox_id", "destination_id", "action_id", "request_digest"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != _DIGEST_HEX_LENGTH
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise DispatchRequestError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.attempt) is not int or self.attempt < 1:
            raise DispatchRequestError("attempt must be a positive integer")
        _bounded_text(self.channel, label="channel")
        _bounded_text(self.to, label="to")
        if isinstance(self.messages, (str, bytes)) or not isinstance(cast(object, self.messages), Sequence):
            raise DispatchRequestError("messages must be an ordered sequence")
        detached = tuple(DispatchMessage.from_value(message) for message in self.messages)
        if not detached:
            raise DispatchRequestError("messages must contain at least one text message")
        if _encoded_size(
            [{"content": item.content, "media": cast(list[object], [])} for item in detached], label="messages"
        ) > _MAX_BUNDLE_BYTES:
            raise DispatchRequestError(f"messages exceed the {_MAX_BUNDLE_BYTES}-byte limit")
        object.__setattr__(self, "messages", detached)

    @property
    def request_id(self) -> str:
        """Alias for the stable E2 request digest used by senders."""

        return self.request_digest

    @property
    def outbox(self) -> str:
        return self.outbox_id

    @property
    def destination(self) -> str:
        return self.destination_id

    @property
    def action(self) -> str:
        return self.action_id

    @property
    def message_bundle(self) -> tuple[DispatchMessage, ...]:
        return self.messages


@dataclass(frozen=True, slots=True)
class ConfirmedDelivered:
    """The sender has a strict durable receipt for the complete bundle."""

    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt", _freeze_receipt(self.receipt))


@dataclass(frozen=True, slots=True)
class ConfirmedNotSent:
    """The sender confirms that no message in the bundle was sent."""

    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _bounded_reason(self.reason_code))


@dataclass(frozen=True, slots=True)
class PermanentNotSent:
    """The sender permanently declined the complete bundle."""

    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _bounded_reason(self.reason_code))


@dataclass(frozen=True, slots=True)
class UnknownOutcome:
    """The sender outcome cannot establish whether an external effect occurred."""

    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _bounded_reason(self.reason_code))


@dataclass(frozen=True, slots=True)
class PartialBundleOutcome:
    """A sender reports partial bundle processing; the whole destination is uncertain."""

    reason_code: str = "partial_bundle"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _bounded_reason(self.reason_code))


SenderResult: TypeAlias = (
    ConfirmedDelivered | ConfirmedNotSent | PermanentNotSent | UnknownOutcome
)


@runtime_checkable
class Sender(Protocol):
    """The only external delivery capability accepted by E3."""

    async def send(self, request: DispatchRequest) -> SenderResult | PartialBundleOutcome:
        """Send one complete immutable request exactly once."""
        ...


DeliverySender = Sender


def _result_for_settlement(result: SenderResult) -> dict[str, object]:
    if type(result) is ConfirmedDelivered:
        return {"outcome": "confirmed_delivered", "receipt": _plain_json(result.receipt)}
    if type(result) is ConfirmedNotSent:
        return {"outcome": "confirmed_not_sent"}
    if type(result) is PermanentNotSent:
        return {"outcome": "permanent_not_sent"}
    if type(result) is UnknownOutcome:
        return {"outcome": "unknown"}
    raise InvalidSenderResultError("sender returned an unclassified result")


@dataclass(slots=True)
class _DispatchEntry:
    """One process/store/claim invocation retained for the claim lifetime."""

    claim: DeliveryClaim
    task: asyncio.Task[SenderResult] | None = None
    sender_task: asyncio.Future[object] | None = None
    reentrant_error: CronDeliveryReentrancyError | None = None


class _DispatchRegistry:
    """Synchronous registry shared by runtime adapters for one store object."""

    def __init__(self, store: CronDeliveryOutboxStore) -> None:
        self.store = store
        self.process_id = os.getpid()
        self.lock = threading.RLock()
        self.entries: dict[int, _DispatchEntry] = {}


_REGISTRY_ATTRIBUTE = "_nanobot_cron_delivery_runtime_registry_v1"
_REGISTRY_ATTACH_LOCK = threading.Lock()


def _registry_for(store: CronDeliveryOutboxStore) -> _DispatchRegistry:
    """Return the registry attached to this exact store authority.

    The E2 store deliberately owns claim capabilities, while E3 owns sender
    invocation.  Keeping this small registry on the exact store object joins
    separate runtime adapters without weakening E2's capability checks or
    introducing a process-global id map that could suffer id reuse.
    """

    registry = getattr(store, _REGISTRY_ATTRIBUTE, None)
    if (
        isinstance(registry, _DispatchRegistry)
        and registry.store is store
        and registry.process_id == os.getpid()
    ):
        return registry
    with _REGISTRY_ATTACH_LOCK:
        registry = getattr(store, _REGISTRY_ATTRIBUTE, None)
        if not (
            isinstance(registry, _DispatchRegistry)
            and registry.store is store
            and registry.process_id == os.getpid()
        ):
            registry = _DispatchRegistry(store)
            setattr(store, _REGISTRY_ATTRIBUTE, registry)
        return registry


@dataclass(slots=True)
class _InvocationState:
    """Signals that remain true even if the sender suppresses cancellation."""

    caller_cancelled: bool = False
    deadline_expired: bool = False
    caller_error: asyncio.CancelledError | None = None
    timeout_error: asyncio.TimeoutError | None = None


@dataclass(frozen=True, slots=True)
class _InvocationMarker:
    """Inherited task context identifying one active sender invocation."""

    store: CronDeliveryOutboxStore
    claim: DeliveryClaim
    entry: _DispatchEntry


_ACTIVE_INVOCATIONS: ContextVar[tuple[_InvocationMarker, ...]] = ContextVar(
    "nanobot_cron_delivery_active_invocations",
    default=(),
)


def _active_invocation_marker(
    store: CronDeliveryOutboxStore,
    claim: DeliveryClaim,
) -> _InvocationMarker | None:
    for marker in reversed(_ACTIVE_INVOCATIONS.get()):
        if marker.store is store and marker.claim is claim:
            return marker
    return None


def _validate_sender(sender: object) -> Any:
    method = getattr(sender, "send", None)
    if not callable(method):
        raise InvalidSenderResultError("sender must provide an async send(request) method")
    return method


def _validate_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if type(timeout) not in (int, float):
        raise ValueError("timeout must be a positive finite number")
    try:
        bounded = float(timeout)
    except (OverflowError, ValueError):
        raise ValueError("timeout must be a positive finite number") from None
    if not math.isfinite(bounded) or bounded <= 0:
        raise ValueError("timeout must be a positive finite number")
    return bounded


def _destination_for_claim(record: DeliveryOutboxRecord, claim: DeliveryClaim) -> Any:
    for destination in record.destinations:
        if destination.destination_id == claim.destination_id:
            if not destination.attempts:
                break
            attempt = destination.attempts[-1]
            if (
                attempt.attempt == claim.attempt
                and attempt.action_id == claim.action_id
                and attempt.request_digest == claim.request_digest
                and destination.status == "sending"
                and attempt.status == "sending"
            ):
                return destination
            break
    raise CronDeliveryRuntimeError("durably sending claim does not match its outbox record")


def _request_from_record(record: DeliveryOutboxRecord, claim: DeliveryClaim) -> DispatchRequest:
    destination = _destination_for_claim(record, claim)
    return DispatchRequest(
        outbox_id=claim.outbox_id,
        destination_id=claim.destination_id,
        attempt=claim.attempt,
        action_id=claim.action_id,
        request_digest=claim.request_digest,
        channel=destination.channel,
        to=destination.to,
        messages=record.messages,
    )


class CronDeliveryRuntime:
    """Use E2's store-scoped claims as the sole sender invocation authority."""

    def __init__(
        self,
        outbox_store: CronDeliveryOutboxStore,
        occurrence_store: CronOccurrenceStore | None = None,
        *,
        sender: Sender | None = None,
    ) -> None:
        if not isinstance(cast(object, outbox_store), CronDeliveryOutboxStore):
            raise TypeError("CronDeliveryRuntime requires a CronDeliveryOutboxStore")
        if occurrence_store is not None and not isinstance(cast(object, occurrence_store), CronOccurrenceStore):
            raise TypeError("occurrence_store must be a CronOccurrenceStore")
        if (
            occurrence_store is not None
            and occurrence_store.namespace != outbox_store.namespace
        ):
            raise ValueError("delivery and occurrence stores must share one namespace")
        if sender is not None:
            _validate_sender(sender)
        self.outbox_store = outbox_store
        self.store = outbox_store
        self.occurrence_store = occurrence_store
        self.sender = sender
        self._dispatch_registry = _registry_for(outbox_store)

    def _require_occurrence_store(self) -> CronOccurrenceStore:
        if self.occurrence_store is None:
            raise CronDeliveryRuntimeUnavailableError(
                f"{CRON_DELIVERY_RUNTIME_UNAVAILABLE_DIAGNOSTIC}; occurrence authority is absent"
            )
        return self.occurrence_store

    def prepare_completed(
        self,
        occurrence_id: str | CronOccurrenceStore,
        maybe_occurrence_id: str | None = None,
    ) -> DeliveryOutboxRecord:
        """Prepare one completed occurrence through the exact E1 authority."""
        occurrence_store = self._require_occurrence_store()
        if isinstance(occurrence_id, CronOccurrenceStore):
            if maybe_occurrence_id is None:
                raise TypeError("prepare_completed requires an occurrence_id")
            if occurrence_id is not occurrence_store:
                raise ValueError("prepare_completed requires the runtime occurrence authority")
            occurrence_key = maybe_occurrence_id
        else:
            if maybe_occurrence_id is not None:
                raise TypeError("prepare_completed accepts one occurrence_id")
            occurrence_key = occurrence_id
        return self.outbox_store.prepare_completed(occurrence_store, occurrence_key)

    def get(self, outbox_id: str) -> DeliveryOutboxRecord:
        return self.outbox_store.get(outbox_id)

    get_outbox = get

    def list(self) -> list[DeliveryOutboxRecord]:
        return self.outbox_store.list()

    list_outboxes = list

    def status(self, outbox_id: str) -> str:
        return self.get(outbox_id).status

    get_status = status

    def reconciliation_required(self) -> list[DeliveryOutboxRecord]:
        return self.outbox_store.reconciliation_required()

    get_reconciliation_required = reconciliation_required

    def resolve_after_reconciliation(
        self,
        outbox_id: str | DeliveryClaim,
        destination_id: str | None = None,
        *,
        expected_revision: int,
        action_id: str,
        request_digest: str,
        evidence_ref: str,
        predecessor_fenced_ref: str,
        resolution: str | Mapping[str, Any],
        receipt: Mapping[str, Any] | None = None,
        predecessor_attempt: int | None = None,
        predecessor_owner: str | None = None,
        attempt: int | None = None,
        owner: str | None = None,
    ) -> DeliveryOutboxRecord:
        """Forward explicit evidence to E2, which acquires the destination fence."""
        return self.outbox_store.resolve_after_reconciliation(
            outbox_id,
            destination_id,
            expected_revision=expected_revision,
            action_id=action_id,
            request_digest=request_digest,
            evidence_ref=evidence_ref,
            predecessor_fenced_ref=predecessor_fenced_ref,
            resolution=resolution,
            receipt=receipt,
            predecessor_attempt=predecessor_attempt,
            predecessor_owner=predecessor_owner,
            attempt=attempt,
            owner=owner,
        )

    resolve = resolve_after_reconciliation

    def claim(
        self,
        outbox_id: str,
        destination_id: str,
        expected_revision: int,
        owner: str | None = None,
    ) -> DeliveryClaim:
        return self.outbox_store.claim_destination(
            outbox_id,
            destination_id,
            expected_revision,
            owner=owner,
        )

    claim_destination = claim

    def _sender_for(self, sender: Sender | None) -> Sender:
        selected = sender if sender is not None else self.sender
        if selected is None:
            raise CronDeliveryRuntimeUnavailableError(
                f"{CRON_DELIVERY_RUNTIME_UNAVAILABLE_DIAGNOSTIC}; sender is absent"
            )
        _validate_sender(selected)
        return selected

    def _reentrancy_error(self, marker: _InvocationMarker) -> CronDeliveryReentrancyError:
        with self._dispatch_registry.lock:
            if marker.entry.reentrant_error is None:
                marker.entry.reentrant_error = CronDeliveryReentrancyError(
                    "sender cannot dispatch the claim whose invocation is currently running"
                )
            return marker.entry.reentrant_error

    def _reentrancy_error_for_claim(
        self,
        claim: DeliveryClaim,
    ) -> CronDeliveryReentrancyError | None:
        marker = _active_invocation_marker(self.outbox_store, claim)
        return None if marker is None else self._reentrancy_error(marker)

    def _reentrancy_error_for_ids(
        self,
        outbox_id: str,
        destination_id: str,
    ) -> CronDeliveryReentrancyError | None:
        for marker in reversed(_ACTIVE_INVOCATIONS.get()):
            if (
                marker.store is self.outbox_store
                and marker.claim.outbox_id == outbox_id
                and marker.claim.destination_id == destination_id
            ):
                return self._reentrancy_error(marker)
        return None

    def _settle(
        self,
        claim: DeliveryClaim,
        fence: DeliverySendFence,
        result: SenderResult,
        *,
        original_error: BaseException | None = None,
    ) -> SenderResult:
        try:
            self.outbox_store.settle(
                claim, _result_for_settlement(result), fence=fence
            )
        except BaseException as exc:
            settlement_error = CronDeliverySettlementError(
                "sender outcome could not be durably settled; reconciliation is required",
                claim=claim,
                classification=result,
                original_error=original_error,
            )
            raise settlement_error from exc
        return result

    @staticmethod
    def _exception_reason(error: BaseException) -> str:
        if isinstance(error, asyncio.TimeoutError):
            return "timeout"
        if isinstance(error, asyncio.CancelledError):
            return "cancelled"
        if isinstance(error, PartialBundleError):
            return error.reason_code
        if isinstance(error, InvalidSenderResultError):
            return "invalid_result"
        if isinstance(error, CronDeliveryReentrancyError):
            return "reentrant_dispatch"
        return "sender_exception"

    @staticmethod
    def _normalize_result(value: object) -> SenderResult:
        # Exact variant types are part of the sender boundary.  Rebuilding the
        # value also validates fields on instances manufactured through
        # object.__new__ or otherwise corrupted after construction.
        try:
            if type(value) is ConfirmedDelivered:
                return ConfirmedDelivered(value.receipt)  # type: ignore[attr-defined]
            if type(value) is ConfirmedNotSent:
                return ConfirmedNotSent(value.reason_code)  # type: ignore[attr-defined]
            if type(value) is PermanentNotSent:
                return PermanentNotSent(value.reason_code)  # type: ignore[attr-defined]
            if type(value) is UnknownOutcome:
                return UnknownOutcome(value.reason_code)  # type: ignore[attr-defined]
            if type(value) is PartialBundleOutcome:
                return UnknownOutcome(value.reason_code)  # type: ignore[attr-defined]
        except BaseException as exc:
            raise InvalidSenderResultError("sender returned a malformed result") from exc
        raise InvalidSenderResultError("sender returned an unclassified result")

    async def _cancel_and_join_sender(
        self,
        sender_task: asyncio.Future[object],
        state: _InvocationState,
    ) -> None:
        """Cancel a sender and wait through repeated caller cancellation.

        A transport is allowed to catch ``CancelledError`` and perform its own
        asynchronous cleanup.  The runtime cannot settle the durable claim
        until that task has actually stopped, so every cancellation received
        while joining is remembered and the join continues.
        """

        if not sender_task.done():
            sender_task.cancel()
        while not sender_task.done():
            try:
                await asyncio.shield(sender_task)
            except asyncio.CancelledError as exc:
                if sender_task.done():
                    break
                state.caller_cancelled = True
                if state.caller_error is None:
                    state.caller_error = exc
            except BaseException:
                # The sender task's own exception is consumed below.  A task
                # may report it through shield before the done flag is
                # observed, so continue until the task is definitely complete.
                if sender_task.done():
                    break
        if sender_task.cancelled():
            return
        try:
            sender_task.result()
        except BaseException:
            pass

    @staticmethod
    async def _await_sender(
        sender_task: asyncio.Future[object],
        timeout: float | None,
    ) -> object:
        # Shield keeps a caller/deadline cancellation from silently deciding
        # the sender task's result.  The caller below explicitly cancels and
        # joins that task after recording the independent signal.
        awaited = asyncio.shield(sender_task)
        if timeout is None:
            return await awaited
        return await asyncio.wait_for(awaited, timeout)

    async def _dispatch_sender(
        self,
        claim: DeliveryClaim,
        fence: DeliverySendFence,
        request: DispatchRequest,
        selected_sender: Sender,
        timeout: float | None,
        entry: _DispatchEntry,
    ) -> SenderResult:
        """Invoke and classify one sender task after the sending fence."""

        state = _InvocationState()
        sender_task: asyncio.Future[object] | None = None
        setup_error: BaseException | None = None
        marker = _InvocationMarker(self.outbox_store, claim, entry)
        marker_token = _ACTIVE_INVOCATIONS.set((*_ACTIVE_INVOCATIONS.get(), marker))
        try:
            try:
                method = _validate_sender(selected_sender)
                returned = method(request)
            except BaseException as exc:
                setup_error = exc
            else:
                if not inspect.isawaitable(returned):
                    setup_error = InvalidSenderResultError(
                        "sender.send(request) must return an awaitable"
                    )
                else:
                    try:
                        sender_task = asyncio.ensure_future(returned)
                    except BaseException as exc:
                        setup_error = exc
                        if inspect.iscoroutine(returned):
                            returned.close()
        finally:
            _ACTIVE_INVOCATIONS.reset(marker_token)
        if setup_error is not None:
            unknown = UnknownOutcome(self._exception_reason(setup_error))
            self._settle(claim, fence, unknown, original_error=setup_error)
            raise setup_error
        if sender_task is None:  # pragma: no cover - setup either creates a future or fails
            setup_error = InvalidSenderResultError("sender task could not be created")
            unknown = UnknownOutcome(self._exception_reason(setup_error))
            self._settle(claim, fence, unknown, original_error=setup_error)
            raise setup_error
        entry.sender_task = sender_task

        try:
            value = await self._await_sender(sender_task, timeout)
        except asyncio.TimeoutError as exc:
            state.deadline_expired = True
            state.timeout_error = exc
            await self._cancel_and_join_sender(sender_task, state)
            if state.caller_cancelled:
                cancellation = state.caller_error or asyncio.CancelledError()
                unknown = UnknownOutcome("cancelled")
                self._settle(claim, fence, unknown, original_error=cancellation)
                raise cancellation
            unknown = UnknownOutcome("timeout")
            self._settle(claim, fence, unknown, original_error=exc)
            raise
        except asyncio.CancelledError as exc:
            # A cancellation from a sender task that is already done is a
            # sender outcome.  Otherwise this task was cancelled by its
            # dispatch caller and that signal has to survive sender cleanup.
            sender_cancelled = sender_task.done() and sender_task.cancelled()
            if not sender_cancelled:
                state.caller_cancelled = True
                state.caller_error = exc
            await self._cancel_and_join_sender(sender_task, state)
            reason = "cancelled"
            unknown = UnknownOutcome(reason)
            # Every caller_cancelled assignment also records caller_error before an await.
            original = cast(asyncio.CancelledError, state.caller_error) if state.caller_cancelled else exc
            self._settle(claim, fence, unknown, original_error=original)
            raise original
        except BaseException as exc:
            # If cancellation arrived while the sender was cleaning up, the
            # cleanup helper records it and it takes precedence over any
            # sender result or exception.
            if state.caller_cancelled:
                cancellation = state.caller_error or asyncio.CancelledError()
                unknown = UnknownOutcome("cancelled")
                self._settle(claim, fence, unknown, original_error=cancellation)
                raise cancellation
            unknown = UnknownOutcome(self._exception_reason(exc))
            self._settle(claim, fence, unknown, original_error=exc)
            raise

        if state.caller_cancelled:
            cancellation = state.caller_error or asyncio.CancelledError()
            unknown = UnknownOutcome("cancelled")
            self._settle(claim, fence, unknown, original_error=cancellation)
            raise cancellation
        if state.deadline_expired:
            timeout_error = state.timeout_error or asyncio.TimeoutError()
            unknown = UnknownOutcome("timeout")
            self._settle(claim, fence, unknown, original_error=timeout_error)
            raise timeout_error
        if entry.reentrant_error is not None:
            error = entry.reentrant_error
            unknown = UnknownOutcome("reentrant_dispatch")
            self._settle(claim, fence, unknown, original_error=error)
            raise error
        try:
            result = self._normalize_result(value)
        except BaseException as exc:
            unknown = UnknownOutcome(self._exception_reason(exc))
            self._settle(claim, fence, unknown, original_error=exc)
            raise
        return self._settle(claim, fence, result)

    async def _dispatch_once(
        self,
        claim: DeliveryClaim,
        selected_sender: Sender,
        timeout: float | None,
        entry: _DispatchEntry,
    ) -> SenderResult:
        # This synchronous store call is the only path that establishes sender
        # authority.  It completes before request construction and before any
        # await that can enter sender/channel code.
        with self.outbox_store.acquire_send_fence(claim) as fence:
            sending_record = self.outbox_store.mark_sending(claim, fence)
            try:
                request = _request_from_record(sending_record, claim)
            except BaseException as exc:
                unknown = UnknownOutcome("invalid_request")
                return self._settle(claim, fence, unknown, original_error=exc)
            return await self._dispatch_sender(
                claim, fence, request, selected_sender, timeout, entry
            )

    async def dispatch(
        self,
        claim: DeliveryClaim | str,
        destination_id: str | object | None = None,
        expected_revision: int | None = None,
        *,
        owner: str | None = None,
        sender: Sender | None = None,
        timeout: float | None = None,
    ) -> SenderResult:
        """Dispatch one claim and durably classify every sender exit path.

        The normal form is ``await dispatch(claim, sender=sender)``.  For
        convenience, ``outbox_id, destination_id, expected_revision`` may be
        supplied and this method will claim exactly once before dispatching.
        A sender may also be passed as the second positional argument when a
        ``DeliveryClaim`` is used.
        """
        # A runtime object may itself be inherited across fork; refresh the
        # exact-store/PID binding before consulting any retained task entry.
        registry = _registry_for(self.outbox_store)
        if registry is not self._dispatch_registry:
            self._dispatch_registry = registry
        selected_sender: Sender | None = sender
        if isinstance(claim, DeliveryClaim):
            if selected_sender is None and destination_id is not None and not isinstance(
                destination_id, str
            ):
                selected_sender = destination_id  # type: ignore[assignment]
            if expected_revision is not None or (
                isinstance(destination_id, str) and destination_id != claim.destination_id
            ):
                raise TypeError("a DeliveryClaim cannot be combined with destination arguments")
            delivery_claim = claim
            reentrant_error = self._reentrancy_error_for_claim(delivery_claim)
            if reentrant_error is not None:
                raise reentrant_error
            selected = self._sender_for(selected_sender)
            bounded_timeout = _validate_timeout(timeout)
        else:
            if not isinstance(cast(object, claim), str) or not isinstance(destination_id, str):
                raise TypeError("dispatch requires a DeliveryClaim or outbox/destination IDs")
            if expected_revision is None:
                raise TypeError("dispatch requires expected_revision for an outbox ID")
            reentrant_error = self._reentrancy_error_for_ids(claim, destination_id)
            if reentrant_error is not None:
                raise reentrant_error
            # Validate all sender configuration before claiming.  A bad
            # sender or deadline must leave the durable destination pending.
            selected = self._sender_for(selected_sender)
            bounded_timeout = _validate_timeout(timeout)
            delivery_claim = self.claim(
                claim,
                destination_id,
                expected_revision,
                owner=owner,
            )
        key = id(delivery_claim)
        current = asyncio.current_task()
        with self._dispatch_registry.lock:
            entry = self._dispatch_registry.entries.get(key)
            if entry is not None and entry.claim is not delivery_claim:
                # The registry retains its claim reference, so this is only a
                # defensive id-reuse guard for externally modified objects.
                self._dispatch_registry.entries.pop(key, None)
                entry = None
            if entry is None:
                entry = _DispatchEntry(delivery_claim)
                task = asyncio.create_task(
                    self._dispatch_once(delivery_claim, selected, bounded_timeout, entry)
                )
                entry.task = task
                self._dispatch_registry.entries[key] = entry
            else:
                task = entry.task
                if task is None:  # pragma: no cover - registry entries are published atomically
                    raise CronDeliveryRuntimeError("delivery dispatch registry entry is incomplete")
                if current is task or current is entry.sender_task:
                    error = CronDeliveryReentrancyError(
                        "sender cannot dispatch the claim whose invocation is currently running"
                    )
                    entry.reentrant_error = error
                    raise error
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            # A caller cancellation must not abandon a durably sending claim.
            if not task.done():
                task.cancel()
            # Repeated cancellation of this waiter must not interrupt the
            # cleanup join.  The underlying dispatch task owns settlement;
            # the original caller cancellation remains the public result even
            # if durable settlement itself fails.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    if task.done():
                        break
            raise cancellation
        finally:
            if task.done():
                # Retrieve terminal exceptions so a caller that was cancelled
                # after completion cannot leave an unobserved task warning.
                try:
                    task.exception()
                except BaseException:
                    pass


# Compatibility aliases keep the package vocabulary discoverable without
# introducing another runtime or another durable delivery authority.
CronDeliveryAuthority = CronDeliveryRuntime
DeliveryRuntime = CronDeliveryRuntime
DeliveryRuntimeError = CronDeliveryRuntimeError
DeliveryRuntimeUnavailableError = CronDeliveryRuntimeUnavailableError
DeliveryReentrancyError = CronDeliveryReentrancyError
DeliverySettlementError = CronDeliverySettlementError
SenderResultError = InvalidSenderResultError


__all__ = [
    "CRON_DELIVERY_RUNTIME_UNAVAILABLE_DIAGNOSTIC",
    "ConfirmedDelivered",
    "ConfirmedNotSent",
    "CronDeliveryRuntime",
    "CronDeliveryAuthority",
    "CronDeliveryRuntimeError",
    "CronDeliveryRuntimeUnavailableError",
    "CronDeliveryReentrancyError",
    "CronDeliverySettlementError",
    "DeliveryRuntime",
    "DeliveryRuntimeError",
    "DeliveryRuntimeUnavailableError",
    "DeliveryReentrancyError",
    "DeliverySettlementError",
    "DeliverySender",
    "DispatchMessage",
    "DispatchRequest",
    "DispatchRequestError",
    "InvalidSenderResultError",
    "PartialBundleError",
    "PartialBundleOutcome",
    "PermanentNotSent",
    "Sender",
    "SenderResult",
    "SenderResultError",
    "UnknownOutcome",
]
