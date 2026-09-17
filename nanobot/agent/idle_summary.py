"""Stateless, storage-free I3c idle source-summary adapter.

This module is an integration seam between the dormant I2 checkpoint capture
and the dormant I3b runtime binding.  It accepts an already qualified,
single-transport-attempt capability and produces an in-memory checkpoint
proposal.  It does not construct providers, read or write sessions, publish a
checkpoint, use a conversation state, or retry a transport call.

I3c computes only from the detached I2 capture.  It has no direct or ad-hoc
filesystem, source, checkpoint, or artifact reads or writes; its I3b authority
checks call the binding registry, whose revalidation may reload authoritative
source state and retire stale authority.  Cancellation cleanup joins injected
callbacks even when that extends beyond the operation deadline; I4 must
qualify a finite cleanup bound.  Injected transport and dependency verifiers
must also be qualified to release retained request references after
settlement.

The public records in this module are deliberately explicit.  A caller must
choose every limit, and a transport must provide both a deterministic
finalized-request measurement and a receipt for exactly one concrete attempt.
Unknown or contradictory evidence is rejected closed.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

from nanobot.agent.idle_compaction import IdleAuxiliaryUsage
from nanobot.agent.idle_runtime import IdleRuntimeBinding, IdleRuntimeBindingRegistry
from nanobot.agent.memory import Consolidator
from nanobot.providers.base import LLMUsage
from nanobot.session.checkpoint_dependencies import (
    ConsumedSummaryChunk,
    SummaryDependencyProof,
    derive_summary_references,
)
from nanobot.session.context_checkpoint import validate_context_checkpoint
from nanobot.session.idle_checkpoint import (
    IdleCheckpointCapture,
    _validate_capture_attestation,
    project_idle_root_nodes,
)
from nanobot.session.idle_compaction import capture_idle_token
from nanobot.session.source import SourceValidationError, prefix_digest
from nanobot.session.source_compaction import SourceCompactionCandidate
from nanobot.session.source_projection import CheckpointNode, source_nodes
from nanobot.session.source_request import SourceTranscript
from nanobot.utils.helpers import truncate_text_to_tokens

if TYPE_CHECKING:
    from nanobot.session.checkpoint_dependencies import SummaryEvidence


MAX_CHECKPOINT_CHARS = 64_000
_SUMMARY_SYSTEM_ROLE = "system"
_SUMMARY_USER_ROLE = "user"


class IdleSummaryError(RuntimeError):
    """Base error for an idle summary admission or preflight rejection."""


class IdleSummaryBindingError(IdleSummaryError):
    """The I2 capture and I3b binding do not describe one authority."""


class IdleSummaryCapabilityError(IdleSummaryError):
    """The injected provider capability is missing or not qualified."""


class IdleSummaryBudgetError(IdleSummaryError):
    """A finalized request or aggregate reservation cannot fit its budget."""


class IdleSummaryDependencyError(IdleSummaryError):
    """Dependency disclosure could not be verified for the current capture."""


class IdleSummaryNoProgressError(IdleSummaryError):
    """The captured source has no advancing visible closed evidence."""


class IdleSummaryTimeoutError(IdleSummaryError):
    """The adapter deadline elapsed and its one attempt was joined."""


class _IdleSummaryAttemptCancelledError(IdleSummaryError):
    """The injected attempt cancelled itself after one accounted dispatch."""


class _IdleSummaryUnknownUsageError(IdleSummaryError):
    """The response did not provide independently valid exact usage."""


@dataclass(frozen=True, slots=True)
class IdleSummaryLimits:
    """All limits for one stateless idle-summary operation.

    There are no defaults: the coordinator must explicitly select every
    spend-bearing value.  ``safety_buffer`` is allowed to be zero, while every
    count/time/output/input ceiling is strictly positive.
    """

    max_calls: int
    max_total_tokens: int
    max_wall_seconds: float
    max_input_tokens: int
    max_output_tokens: int
    max_chunks: int
    safety_buffer: int

    def __post_init__(self) -> None:
        for name in (
            "max_calls",
            "max_total_tokens",
            "max_input_tokens",
            "max_output_tokens",
            "max_chunks",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_wall_seconds, bool)
            or not isinstance(self.max_wall_seconds, (int, float))
            or not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be a finite positive number")
        if type(self.safety_buffer) is not int or self.safety_buffer < 0:
            raise ValueError("safety_buffer must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class IdleSummaryRequestReceipt:
    """Exact measurement of the finalized request handed to one transport."""

    provider: str
    transport: str
    model: str
    input_tokens: int
    reserved_output_tokens: int
    safety_buffer: int
    model_window_tokens: int
    total_reservation_tokens: int
    request_digest: str

    def __post_init__(self) -> None:
        for name in ("provider", "transport", "model", "request_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "input_tokens",
            "reserved_output_tokens",
            "safety_buffer",
            "model_window_tokens",
            "total_reservation_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.model_window_tokens <= 0:
            raise ValueError("model_window_tokens must be positive")
        if self.total_reservation_tokens != (
            self.input_tokens + self.reserved_output_tokens + self.safety_buffer
        ):
            raise ValueError("request reservation does not match its components")
        if len(self.request_digest) != hashlib.sha256().digest_size * 2 or any(
            character not in "0123456789abcdef" for character in self.request_digest
        ):
            raise ValueError("request_digest must be a lowercase sha256 digest")


@dataclass(frozen=True, slots=True)
class _IdleSummaryMeasurementIdentity:
    """Detached routing identity for one request measurement authority."""

    provider_name: str
    transport: str
    model: str
    context_window_tokens: int
    runtime_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("provider_name", "transport", "model", "runtime_fingerprint"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"measurement identity {name} is invalid")
        if type(self.context_window_tokens) is not int or self.context_window_tokens <= 0:
            raise ValueError("measurement identity context window is invalid")


@dataclass(frozen=True, slots=True)
class _IdleSummaryMeasurementSnapshot:
    """Detached primitive measurement and sealed-key evidence.

    The injected measurer owns the receipt object it returns.  This snapshot
    copies every receipt field and every input to the invocation-local cache,
    so later callback mutation cannot change either comparison authority or a
    request reservation.
    """

    finalized_digest: str
    identity_provider: str
    identity_transport: str
    identity_model: str
    identity_context_window_tokens: int
    identity_runtime_fingerprint: str
    max_output_tokens: int
    measurement_safety_buffer: int
    provider: str
    transport: str
    model: str
    input_tokens: int
    reserved_output_tokens: int
    safety_buffer: int
    model_window_tokens: int
    total_reservation_tokens: int
    request_digest: str

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            self.finalized_digest,
            self.identity_provider,
            self.identity_transport,
            self.identity_model,
            self.identity_context_window_tokens,
            self.identity_runtime_fingerprint,
            self.max_output_tokens,
            self.measurement_safety_buffer,
        )

    def request_receipt(self) -> IdleSummaryRequestReceipt:
        """Reconstruct an exact receipt detached from callback ownership."""
        return IdleSummaryRequestReceipt(
            self.provider,
            self.transport,
            self.model,
            self.input_tokens,
            self.reserved_output_tokens,
            self.safety_buffer,
            self.model_window_tokens,
            self.total_reservation_tokens,
            self.request_digest,
        )


@dataclass(frozen=True, slots=True)
class IdleSummaryMeasuredUsage:
    """Exact token receipt for one completed concrete transport attempt."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.reasoning_tokens is not None and (
            type(self.reasoning_tokens) is not int or self.reasoning_tokens < 0
        ):
            raise ValueError("reasoning_tokens must be None or a non-negative integer")
        visible = self.input_tokens + self.output_tokens
        if self.total_tokens < visible:
            raise ValueError("total_tokens cannot be below visible tokens")
        if self.reasoning_tokens is not None and self.total_tokens != visible + self.reasoning_tokens:
            raise ValueError("reasoning_tokens does not match total_tokens")

    @property
    def auxiliary_usage(self) -> IdleAuxiliaryUsage:
        return IdleAuxiliaryUsage(calls=1, tokens=self.total_tokens)

    def __add__(self, other: IdleSummaryMeasuredUsage) -> IdleSummaryMeasuredUsage:
        if type(other) is not IdleSummaryMeasuredUsage:
            return NotImplemented
        reasoning: int | None
        if self.reasoning_tokens is None or other.reasoning_tokens is None:
            reasoning = None
        else:
            reasoning = self.reasoning_tokens + other.reasoning_tokens
        return IdleSummaryMeasuredUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.total_tokens + other.total_tokens,
            reasoning,
        )


class _IdleSummaryRequestReservation:
    """Request-owned, single-use state that carries no request payload."""

    __slots__ = ("claimed", "owner", "__weakref__")

    def __init__(self) -> None:
        self.claimed = False
        self.owner: weakref.ReferenceType[IdleSummaryRequest] | None = None

    def bind(self, request: IdleSummaryRequest) -> None:
        if self.owner is None:
            self.owner = weakref.ref(request)

    def __copy__(self) -> _IdleSummaryRequestReservation:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _IdleSummaryRequestReservation:
        return self


@dataclass(frozen=True, slots=True, weakref_slot=True)
class IdleSummaryRequest:
    """Immutable request supplied to one qualified transport attempt."""

    messages: tuple[dict[str, Any], ...]
    receipt: IdleSummaryRequestReceipt
    _reservation: _IdleSummaryRequestReservation = field(
        default_factory=_IdleSummaryRequestReservation,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.receipt) is not IdleSummaryRequestReceipt:
            raise ValueError("request receipt is invalid")
        if type(self._reservation) is not _IdleSummaryRequestReservation:
            raise ValueError("request reservation is invalid")
        self._reservation.bind(self)
        if type(self.messages) is not tuple or not self.messages:
            raise ValueError("summary request must contain messages")
        if any(type(message) is not dict for message in self.messages):
            raise ValueError("summary request messages must be dictionaries")


@dataclass(frozen=True, slots=True)
class IdleSummaryAttemptResponse:
    """Typed result from exactly one qualified transport attempt."""

    content: str | None
    usage: IdleSummaryMeasuredUsage | LLMUsage | None
    request: IdleSummaryRequestReceipt
    finish_reason: str = "stop"
    tool_calls: tuple[object, ...] = ()
    attempts: int = 1
    provider_state: object | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not IdleSummaryRequestReceipt:
            raise ValueError("attempt request receipt is invalid")
        if not isinstance(self.finish_reason, str):
            raise ValueError("attempt finish reason is invalid")
        if type(self.tool_calls) is not tuple:
            raise ValueError("attempt tool calls must be a tuple")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("attempt count is invalid")


class IdleSummaryRequestMeasurer(Protocol):
    """Trusted local finalized-request measurer.

    This protocol is intentionally separate from the transport capability.  A
    future integration must supply a local deterministic measurer; this module
    never obtains one from a provider factory or from a network-capable object.
    """

    provider_name: str
    transport: str
    model: str
    context_window_tokens: int
    runtime_fingerprint: str

    def measure_request(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        max_output_tokens: int,
        safety_buffer: int,
    ) -> IdleSummaryRequestReceipt: ...


class IdleSummaryAttemptCapability(Protocol):
    """Compatibility protocol for the raw adapter supplied to the registry.

    Raw adapters are not accepted by :class:`IdleSummaryAdapter`; they must be
    sealed by :class:`IdleSummaryTransportRegistry` first.  The registry proves
    one adapter invocation per request reservation.  Qualification of SDK
    retries, fallbacks, output limits, and final transport behavior remains an
    explicit I4 gate.
    """

    provider_name: str
    transport: str
    model: str
    context_window_tokens: int
    runtime_fingerprint: str

    def attempt_once(
        self,
        request: IdleSummaryRequest,
    ) -> Awaitable[IdleSummaryAttemptResponse]: ...


_TRANSPORT_SEAL = object()


class IdleSummaryTransportRegistry:
    """Issue process/store/binding-bound transport capabilities.

    This dormant registry is the only constructor for a qualified transport.
    It does not construct a provider, authorize production spend, or prove
    SDK-internal retry/fallback behavior.  A returned capability records the
    exact process, source-store object, binding, and one adapter reservation;
    copied, reconstructed, forked, or foreign capabilities fail closed.
    """

    def __init__(
        self,
        *,
        binding_registry: IdleRuntimeBindingRegistry,
        source_store: object,
    ) -> None:
        if type(binding_registry) is not IdleRuntimeBindingRegistry:
            raise IdleSummaryCapabilityError("exact idle runtime binding registry is required")
        if source_store is None:
            raise IdleSummaryCapabilityError("exact source store object is required")
        self._authority = object()
        self._pid = os.getpid()
        self._binding_registry = binding_registry
        self._source_store = source_store
        # Keep only weak request-owned reservation identity.  The registry must
        # reject copied/reused reservation state without retaining full prompts.
        self._issued_requests: dict[
            int, weakref.ReferenceType[_IdleSummaryRequestReservation]
        ] = {}

    @property
    def binding_registry(self) -> IdleRuntimeBindingRegistry:
        return self._binding_registry

    @property
    def source_store(self) -> object:
        return self._source_store

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            raise IdleSummaryCapabilityError("qualified transport crossed a process boundary")

    @staticmethod
    def _validate_raw(raw: object) -> IdleSummaryAttemptCapability:
        if raw is None:
            raise IdleSummaryCapabilityError("raw one-attempt adapter is incomplete")
        try:
            values = {
                name: getattr(raw, name)
                for name in ("provider_name", "transport", "model", "runtime_fingerprint")
            }
            window = getattr(raw, "context_window_tokens")
            attempt = getattr(raw, "attempt_once")
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            raise IdleSummaryCapabilityError("raw one-attempt adapter is invalid") from None
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise IdleSummaryCapabilityError("raw one-attempt adapter identity is invalid")
        if type(window) is not int or window <= 0 or not callable(attempt):
            raise IdleSummaryCapabilityError("raw one-attempt adapter is invalid")
        return cast(IdleSummaryAttemptCapability, raw)

    def issue(
        self,
        binding: IdleRuntimeBinding,
        raw: object,
    ) -> IdleSummaryQualifiedTransport:
        self._check_process()
        if type(binding) is not IdleRuntimeBinding or binding.source_store is not self._source_store:
            raise IdleSummaryCapabilityError("transport binding is foreign")
        try:
            self._binding_registry.require_current(binding, binding.session_key)
        except Exception:
            raise IdleSummaryCapabilityError("transport binding is stale") from None
        adapter = self._validate_raw(raw)
        return IdleSummaryQualifiedTransport(
            _TRANSPORT_SEAL,
            self,
            binding,
            adapter,
            os.getpid(),
        )

    qualify = issue

    def _validate_capability(self, capability: IdleSummaryQualifiedTransport) -> None:
        self._check_process()
        if (
            type(capability) is not IdleSummaryQualifiedTransport
            or capability._registry is not self
            or capability._authority is not self._authority
            or capability._pid != self._pid
            or capability._source_store is not self._source_store
        ):
            raise IdleSummaryCapabilityError("qualified transport capability is foreign")
        if (
            type(capability._binding) is not IdleRuntimeBinding
            or capability._binding.source_store is not self._source_store
        ):
            raise IdleSummaryCapabilityError("qualified transport source binding changed")
        try:
            self._binding_registry.require_current(
                capability._binding,
                capability._binding.session_key,
            )
        except Exception:
            raise IdleSummaryCapabilityError("qualified transport binding is stale") from None
        raw = self._validate_raw(capability._raw)
        if (
            raw.provider_name != capability.provider_name
            or raw.transport != capability.transport
            or raw.model != capability.model
            or raw.context_window_tokens != capability.context_window_tokens
            or raw.runtime_fingerprint != capability.runtime_fingerprint
        ):
            raise IdleSummaryCapabilityError("qualified transport identity changed")

    def _claim_request(
        self,
        capability: IdleSummaryQualifiedTransport,
        request: IdleSummaryRequest,
    ) -> None:
        self._validate_capability(capability)
        if type(request) is not IdleSummaryRequest:
            raise IdleSummaryCapabilityError("transport reservation request is invalid")
        try:
            reservation = request._reservation
            owner = reservation.owner() if reservation.owner is not None else None
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            raise IdleSummaryCapabilityError("transport reservation request is invalid") from None
        if type(reservation) is not _IdleSummaryRequestReservation:
            raise IdleSummaryCapabilityError("transport reservation request is invalid")
        if owner is not request:
            raise IdleSummaryCapabilityError("transport reservation request is copied or foreign")
        reservation_id = id(reservation)
        previous = self._issued_requests.get(reservation_id)
        if previous is not None:
            previous_reservation = previous()
            if previous_reservation is reservation:
                raise IdleSummaryCapabilityError("transport reservation was invoked more than once")
            if previous_reservation is not None:
                # An identity collision is not safe to resolve by content.
                raise IdleSummaryCapabilityError("transport reservation identity collided")
            self._issued_requests.pop(reservation_id, None)
        if reservation.claimed:
            raise IdleSummaryCapabilityError("transport reservation was invoked more than once")
        reservation.claimed = True

        def release(_reference: weakref.ReferenceType[_IdleSummaryRequestReservation]) -> None:
            current = self._issued_requests.get(reservation_id)
            if current is _reference or current is not None and current() is None:
                self._issued_requests.pop(reservation_id, None)

        self._issued_requests[reservation_id] = weakref.ref(reservation, release)


@dataclass(frozen=True, slots=True, init=False, repr=False)
class IdleSummaryQualifiedTransport:
    """Sealed transport issued by :class:`IdleSummaryTransportRegistry`."""

    _registry: IdleSummaryTransportRegistry
    _authority: object
    _binding: IdleRuntimeBinding
    _raw: IdleSummaryAttemptCapability
    _pid: int
    _source_store: object
    _provider_name: str
    _transport: str
    _model: str
    _context_window_tokens: int
    _runtime_fingerprint: str

    def __init__(
        self,
        seal: object,
        registry: IdleSummaryTransportRegistry,
        binding: IdleRuntimeBinding,
        raw: IdleSummaryAttemptCapability,
        pid: int,
    ) -> None:
        if seal is not _TRANSPORT_SEAL:
            raise IdleSummaryCapabilityError("qualified transport cannot be reconstructed")
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_authority", registry._authority)
        object.__setattr__(self, "_binding", binding)
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_pid", pid)
        object.__setattr__(self, "_source_store", registry.source_store)
        object.__setattr__(self, "_provider_name", raw.provider_name)
        object.__setattr__(self, "_transport", raw.transport)
        object.__setattr__(self, "_model", raw.model)
        object.__setattr__(self, "_context_window_tokens", raw.context_window_tokens)
        object.__setattr__(self, "_runtime_fingerprint", raw.runtime_fingerprint)

    def __repr__(self) -> str:
        return "<idle-summary-qualified-transport opaque>"

    def __reduce__(self) -> object:
        raise TypeError("qualified transports cannot be serialized")

    def __copy__(self) -> IdleSummaryQualifiedTransport:
        raise TypeError("qualified transports cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> IdleSummaryQualifiedTransport:
        raise TypeError("qualified transports cannot be copied")

    @property
    def binding_registry(self) -> IdleRuntimeBindingRegistry:
        return self._registry.binding_registry

    @property
    def source_store(self) -> object:
        return self._source_store

    @property
    def binding(self) -> IdleRuntimeBinding:
        return self._binding

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def transport(self) -> str:
        return self._transport

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    @property
    def runtime_fingerprint(self) -> str:
        return self._runtime_fingerprint

    def attempt_once(self, request: IdleSummaryRequest) -> Awaitable[IdleSummaryAttemptResponse]:
        self._admit_request(request)
        return self._invoke_admitted(request)

    def _admit_request(self, request: IdleSummaryRequest) -> None:
        """Validate and claim a request before dispatch accounting advances."""
        self._registry._claim_request(self, request)

    def _invoke_admitted(self, request: IdleSummaryRequest) -> Awaitable[IdleSummaryAttemptResponse]:
        """Invoke the raw adapter after the sealed admission has succeeded."""
        result = self._raw.attempt_once(request)
        if not inspect.isawaitable(result):
            raise IdleSummaryError("qualified transport did not return an awaitable")
        return cast(Awaitable[IdleSummaryAttemptResponse], result)


# Public aliases retain discoverable names for integration callers while the
# adapter itself accepts only the sealed capability above.
QualifiedIdleSummaryAttempt = IdleSummaryQualifiedTransport
IdleQualifiedSummaryProvider = IdleSummaryQualifiedTransport


@dataclass(frozen=True, slots=True)
class IdleDependencyVerification:
    """Narrow verifier result bound to this exact capture and policy state."""

    capture: IdleCheckpointCapture
    binding: IdleRuntimeBinding
    references: tuple[str, ...]
    policy_fingerprint: str
    artifact_namespace: object | None

    def __post_init__(self) -> None:
        if type(self.capture) is not IdleCheckpointCapture:
            raise ValueError("dependency capture is invalid")
        if type(self.binding) is not IdleRuntimeBinding:
            raise ValueError("dependency binding is invalid")
        if type(self.references) is not tuple or any(
            not isinstance(reference, str) or not reference for reference in self.references
        ):
            raise ValueError("dependency references are invalid")
        if self.references != tuple(sorted(set(self.references))):
            raise ValueError("dependency references must be sorted and unique")
        if not isinstance(self.policy_fingerprint, str) or not self.policy_fingerprint:
            raise ValueError("dependency policy fingerprint is invalid")


class IdleSummaryDependencyVerifier(Protocol):
    """Verify exact references without exposing a reader or artifact bytes."""

    def verify(
        self,
        *,
        capture: IdleCheckpointCapture,
        binding: IdleRuntimeBinding,
        references: tuple[str, ...],
    ) -> IdleDependencyVerification: ...


@dataclass(frozen=True, slots=True)
class IdleSummaryAttemptAccounting:
    """Auditable reservation and attempt accounting for one computation."""

    reserved_calls: int
    reserved_tokens: int
    dispatched_calls: int
    known_measured_tokens: int
    unknown_dispatched: int
    exact: bool

    def __post_init__(self) -> None:
        for name in (
            "reserved_calls",
            "reserved_tokens",
            "dispatched_calls",
            "known_measured_tokens",
            "unknown_dispatched",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.dispatched_calls > self.reserved_calls:
            raise ValueError("dispatched calls exceed reservations")
        if self.unknown_dispatched > self.dispatched_calls:
            raise ValueError("unknown attempts exceed dispatched calls")
        if type(self.exact) is not bool:
            raise ValueError("exact accounting flag must be boolean")
        if self.exact and not (
            self.dispatched_calls > 0
            and self.unknown_dispatched == 0
            and self.dispatched_calls == self.reserved_calls
        ):
            raise ValueError("exact accounting flag is contradictory")

    @property
    def unknown_dispatched_calls(self) -> int:
        return self.unknown_dispatched


@dataclass(slots=True)
class _IdleSummaryAccountingLedger:
    reserved_calls: int = 0
    reserved_tokens: int = 0
    dispatched_calls: int = 0
    known_measured_tokens: int = 0
    unknown_dispatched: int = 0
    known_attempts: int = 0
    accounted_attempts: int = 0
    finalized: bool = False

    def reserve(self, receipt: IdleSummaryRequestReceipt) -> None:
        self.reserved_calls += 1
        self.reserved_tokens += receipt.total_reservation_tokens

    def dispatched(self) -> None:
        self.dispatched_calls += 1

    def known(self, usage: IdleSummaryMeasuredUsage) -> None:
        if self.accounted_attempts >= self.dispatched_calls:
            raise IdleSummaryError("attempt usage was accounted without a dispatch")
        self.accounted_attempts += 1
        self.known_attempts += 1
        self.known_measured_tokens += usage.total_tokens

    def unknown(self) -> None:
        if self.accounted_attempts >= self.dispatched_calls:
            raise IdleSummaryError("attempt outcome was accounted more than once")
        self.accounted_attempts += 1
        self.unknown_dispatched += 1

    def finalize(self) -> None:
        """Account every dispatched attempt still lacking a terminal outcome."""
        if self.finalized:
            return
        while self.accounted_attempts < self.dispatched_calls:
            self.unknown()
        self.finalized = True

    def snapshot(self, *, terminal: bool = False) -> IdleSummaryAttemptAccounting:
        exact = terminal and (
            self.dispatched_calls > 0
            and self.unknown_dispatched == 0
            and self.dispatched_calls == self.reserved_calls
            and self.known_attempts == self.dispatched_calls
        )
        return IdleSummaryAttemptAccounting(
            self.reserved_calls,
            self.reserved_tokens,
            self.dispatched_calls,
            self.known_measured_tokens,
            self.unknown_dispatched,
            exact,
        )


@dataclass(frozen=True, slots=True)
class IdleSummaryProposal:
    """One unpersisted checkpoint proposal and its complete fold evidence."""

    checkpoint: dict[str, object]
    proof: SummaryDependencyProof
    usage: IdleAuxiliaryUsage
    measured_usage: IdleSummaryMeasuredUsage
    requests: tuple[IdleSummaryRequestReceipt, ...]
    provider: str
    model: str

    @property
    def request_receipts(self) -> tuple[IdleSummaryRequestReceipt, ...]:
        return self.requests


@dataclass(frozen=True, slots=True)
class IdleSummaryResult:
    """Terminal adapter result; failed attempts never carry a proposal."""

    proposal: IdleSummaryProposal | None
    status: str
    usage: IdleAuxiliaryUsage | None
    usage_exact: bool
    reserved_usage: IdleAuxiliaryUsage
    measured_usage: IdleSummaryMeasuredUsage | None
    requests: tuple[IdleSummaryRequestReceipt, ...]
    provider: str
    model: str
    accounting: IdleSummaryAttemptAccounting

    @property
    def candidate(self) -> dict[str, object] | None:
        return self.proposal.checkpoint if self.proposal is not None else None

    @property
    def proof(self) -> SummaryDependencyProof | None:
        return self.proposal.proof if self.proposal is not None else None


@dataclass(frozen=True, slots=True)
class _PreparedCall:
    boundary: int
    evidence: SummaryEvidence
    payload: tuple[dict[str, Any], ...]
    references: tuple[str, ...]
    receipt: IdleSummaryRequestReceipt | None = None


def _positive_deadline(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise IdleSummaryError("deadline must be a finite monotonic timestamp")
    return float(value)


def _read_measurer_identity(
    measurer: object,
    *,
    error_message: str,
) -> _IdleSummaryMeasurementIdentity:
    try:
        values = {
            name: getattr(measurer, name)
            for name in ("provider_name", "transport", "model", "runtime_fingerprint")
        }
        window = getattr(measurer, "context_window_tokens")
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("measurement identity contains an invalid string")
        if type(window) is not int or window <= 0:
            raise ValueError("measurement identity context window is invalid")
        return _IdleSummaryMeasurementIdentity(
            str(values["provider_name"]),
            str(values["transport"]),
            str(values["model"]),
            window,
            str(values["runtime_fingerprint"]),
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise IdleSummaryCapabilityError(error_message) from None


def _transport_identity(
    transport: IdleSummaryQualifiedTransport,
) -> _IdleSummaryMeasurementIdentity:
    try:
        return _IdleSummaryMeasurementIdentity(
            str(transport.provider_name),
            str(transport.transport),
            str(transport.model),
            transport.context_window_tokens,
            str(transport.runtime_fingerprint),
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise IdleSummaryBindingError("qualified transport identity is invalid") from None


def _binding_runtime_identity(binding: IdleRuntimeBinding) -> tuple[str, str, int]:
    try:
        runtime = binding.auxiliary_runtime
        provider_name = getattr(runtime.provider, "provider_name")
        model = runtime.model
        context_window_tokens = runtime.context_window_tokens
        if (
            not isinstance(provider_name, str)
            or not provider_name.strip()
            or not isinstance(model, str)
            or not model.strip()
            or type(context_window_tokens) is not int
            or context_window_tokens <= 0
        ):
            raise ValueError("binding runtime identity is invalid")
        return str(provider_name), str(model), context_window_tokens
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise IdleSummaryCapabilityError("binding runtime identity is invalid") from None


def _qualified_measurer(
    measurer: object,
) -> tuple[IdleSummaryRequestMeasurer, _IdleSummaryMeasurementIdentity]:
    if measurer is None:
        raise IdleSummaryCapabilityError("trusted local request measurer is required")
    identity = _read_measurer_identity(
        measurer,
        error_message="trusted local request measurer is invalid",
    )
    try:
        measure = getattr(measurer, "measure_request")
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise IdleSummaryCapabilityError("trusted local request measurer is invalid") from None
    if not callable(measure):
        raise IdleSummaryCapabilityError("measurer capability method is invalid")
    return cast(IdleSummaryRequestMeasurer, measurer), identity


def _qualified_transport(
    transport: object,
    *,
    binding_registry: IdleRuntimeBindingRegistry,
    source_store: object,
    binding: IdleRuntimeBinding | None = None,
) -> IdleSummaryQualifiedTransport:
    if type(transport) is not IdleSummaryQualifiedTransport:
        raise IdleSummaryCapabilityError("sealed qualified transport is required")
    capability = cast(IdleSummaryQualifiedTransport, transport)
    if (
        capability.binding_registry is not binding_registry
        or capability.source_store is not source_store
        or binding is not None
        and capability.binding is not binding
    ):
        raise IdleSummaryCapabilityError("qualified transport is foreign to this authority")
    capability._registry._validate_capability(capability)
    return capability


def _request_digest(messages: Sequence[Mapping[str, object]]) -> str:
    try:
        payload = json.dumps(
            list(messages), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise IdleSummaryCapabilityError("summary request cannot be measured") from None
    return hashlib.sha256(payload).hexdigest()


def _usage_from_response(value: object) -> IdleSummaryMeasuredUsage | None:
    if value is None:
        return None
    if type(value) is IdleSummaryMeasuredUsage:
        measured = cast(IdleSummaryMeasuredUsage, value)
        try:
            values = (
                measured.input_tokens,
                measured.output_tokens,
                measured.total_tokens,
                measured.reasoning_tokens,
            )
        except AttributeError:
            raise IdleSummaryError("summary attempt usage is invalid") from None
        if (
            any(type(item) is not int or item < 0 for item in values[:3])
            or measured.reasoning_tokens is not None
            and (type(measured.reasoning_tokens) is not int or measured.reasoning_tokens < 0)
        ):
            raise IdleSummaryError("summary attempt usage is invalid")
        try:
            return IdleSummaryMeasuredUsage(*values[:3], measured.reasoning_tokens)
        except (TypeError, ValueError):
            raise IdleSummaryError("summary attempt usage is invalid") from None
    if type(value) is not LLMUsage:
        raise IdleSummaryError("summary attempt usage is malformed")
    usage = cast(LLMUsage, value)
    try:
        raw_values = (
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.request_count,
        )
    except AttributeError:
        raise IdleSummaryError("summary attempt usage is invalid") from None
    if any(type(item) is not int or item < 0 for item in raw_values):
        raise IdleSummaryError("summary attempt usage is invalid")
    if usage.request_count != 1:
        raise IdleSummaryError("summary attempt usage hides multiple requests")
    reasoning = usage.total_tokens - usage.input_tokens - usage.output_tokens
    try:
        return IdleSummaryMeasuredUsage(
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            reasoning,
        )
    except (TypeError, ValueError):
        raise IdleSummaryError("summary attempt usage is invalid") from None


def _detach_receipt(receipt: object) -> IdleSummaryRequestReceipt:
    """Copy a callback-owned receipt into an independently validated value."""
    if type(receipt) is not IdleSummaryRequestReceipt:
        raise IdleSummaryBudgetError("provider request measurement is malformed")
    try:
        values = tuple(
            getattr(receipt, name)
            for name in (
                "provider",
                "transport",
                "model",
                "input_tokens",
                "reserved_output_tokens",
                "safety_buffer",
                "model_window_tokens",
                "total_reservation_tokens",
                "request_digest",
            )
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise IdleSummaryBudgetError("provider request measurement is inaccessible") from None
    if any(type(value) is not str for value in (*values[:3], values[8])) or any(
        type(value) is not int for value in values[3:8]
    ):
        raise IdleSummaryBudgetError("provider request measurement is malformed")
    try:
        return IdleSummaryRequestReceipt(*values)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise IdleSummaryBudgetError("provider request measurement is malformed") from None


def _validate_receipt(
    receipt: object,
    *,
    identity: _IdleSummaryMeasurementIdentity,
    limits: IdleSummaryLimits,
    messages: Sequence[Mapping[str, object]],
) -> IdleSummaryRequestReceipt:
    receipt = _detach_receipt(receipt)
    expected_digest = _request_digest(messages)
    if receipt.provider != identity.provider_name or receipt.transport != identity.transport:
        raise IdleSummaryBindingError("request provider identity changed")
    if receipt.model != identity.model or receipt.model_window_tokens != identity.context_window_tokens:
        raise IdleSummaryBindingError("request model identity changed")
    if receipt.reserved_output_tokens != limits.max_output_tokens:
        raise IdleSummaryBudgetError("request output reservation changed")
    if receipt.safety_buffer != limits.safety_buffer:
        raise IdleSummaryBudgetError("request safety buffer changed")
    if receipt.request_digest != expected_digest:
        raise IdleSummaryBudgetError("request receipt does not bind finalized payload")
    return receipt


def _extract_response_usage(
    response: object,
    *,
    request: IdleSummaryRequest,
    limits: IdleSummaryLimits,
) -> IdleSummaryMeasuredUsage:
    try:
        raw_usage = getattr(response, "usage")
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise _IdleSummaryUnknownUsageError("summary attempt usage is unknown") from None
    try:
        usage = _usage_from_response(raw_usage)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except IdleSummaryError as exc:
        raise _IdleSummaryUnknownUsageError(str(exc)) from None
    except BaseException:
        raise _IdleSummaryUnknownUsageError("summary attempt usage is malformed") from None
    if usage is None:
        raise _IdleSummaryUnknownUsageError("summary attempt usage is unknown")
    if usage.input_tokens != request.receipt.input_tokens:
        raise IdleSummaryError("summary usage does not match finalized request input")
    if usage.output_tokens > limits.max_output_tokens:
        raise IdleSummaryError("summary output usage exceeds its cap")
    if usage.total_tokens > request.receipt.total_reservation_tokens:
        raise IdleSummaryError("summary usage exceeds reserved request cost")
    if usage.total_tokens > limits.max_total_tokens:
        raise IdleSummaryError("summary usage exceeds aggregate token ceiling")
    return usage


def _validate_response_content(
    response: object,
    *,
    request: IdleSummaryRequest,
    transport: IdleSummaryQualifiedTransport,
    limits: IdleSummaryLimits,
) -> str:
    try:
        attempts = response.attempts
        provider_state = response.provider_state
        response_request = response.request
        finish_reason = response.finish_reason
        tool_calls = response.tool_calls
        summary = response.content
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise IdleSummaryError("summary attempt response is malformed") from None
    if type(response) is not IdleSummaryAttemptResponse:
        raise IdleSummaryError("summary attempt response is malformed")
    try:
        if attempts != 1 or provider_state is not None:
            raise IdleSummaryError("summary attempt did not remain one stateless transport")
        if response_request != request.receipt:
            raise IdleSummaryError("summary response request receipt changed")
        if finish_reason != "stop" or tool_calls:
            raise IdleSummaryError("summary response is truncated, errored, or tool-bearing")
        if (
            not isinstance(summary, str)
            or not summary.strip()
            or len(summary) > MAX_CHECKPOINT_CHARS
            or truncate_text_to_tokens(summary, limits.max_output_tokens) != summary
        ):
            raise IdleSummaryError("summary response is empty, oversized, or truncated")
        if transport.provider_name != request.receipt.provider or transport.model != request.receipt.model:
            raise IdleSummaryBindingError("summary response provider identity changed")
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except IdleSummaryError:
        raise
    except BaseException:
        raise IdleSummaryError("summary attempt response is malformed") from None
    return summary


def _checkpoint_from_capture(
    capture: IdleCheckpointCapture,
) -> tuple[SourceCompactionCandidate, CheckpointNode | None]:
    """Build the root-history candidate from only the detached I2 snapshot."""
    token = capture.token
    if not token.checkpoint_valid:
        raise IdleSummaryBindingError("captured checkpoint is invalid")
    if not token.pending_state.idle_eligible:
        raise IdleSummaryBindingError("captured source is pending or recovering")
    snapshot = capture.snapshot
    if snapshot.state.session_key != token.session_key:
        raise IdleSummaryBindingError("capture source identity changed")
    stored = snapshot.checkpoint
    previous: CheckpointNode | None = None
    previous_boundary = 0
    if stored is not None:
        if type(stored) is not dict:
            raise IdleSummaryBindingError("captured checkpoint is malformed")
        parent = stored.get("parent_checkpoint_id")
        source_revision = stored.get("source_revision")
        if type(source_revision) is not int or source_revision >= snapshot.state.revision:
            raise IdleSummaryBindingError("captured checkpoint has invalid provenance")
        try:
            # SourceSnapshot normally validates this before capture.  Re-run
            # exact schema, digest, and closed-boundary validation here so a
            # forged detached capture cannot reach a provider call.
            validated = validate_context_checkpoint(
                stored,
                snapshot.events,
                generation=snapshot.state.generation,
                source_revision=source_revision,
                parent_checkpoint_id=parent,
            )
            previous_boundary = validated["boundary"]
            previous = CheckpointNode(
                validated["checkpoint_id"],
                validated["summary"],
                tuple(validated["artifact_references"]),
            )
        except (KeyError, TypeError, ValueError, SourceValidationError):
            raise IdleSummaryBindingError("captured checkpoint is invalid") from None
        if validated != stored or stored.get("checkpoint_id") != snapshot.state.checkpoint_id:
            raise IdleSummaryBindingError("captured checkpoint identity changed")
    try:
        canonical = project_idle_root_nodes(capture, source_nodes(snapshot.events))
        covered = {event["_event_id"] for event in snapshot.events[:previous_boundary]}
        newest_user = next(
            (node.event_id for node in reversed(canonical) if node.message.get("role") == "user"),
            None,
        )
        transcript_nodes = [previous] if previous is not None else []
        transcript_nodes.extend(
            node
            for node in canonical
            if node.event_id not in covered or node.event_id == newest_user
        )
        transcript = SourceTranscript(tuple(transcript_nodes))
        eligible = frozenset(event["_event_id"] for event in snapshot.events)
        candidate = SourceCompactionCandidate.capture(
            snapshot,
            transcript,
            eligible_ids=eligible,
            accepted_boundary=previous_boundary,
            accepted_checkpoint=previous,
        )
    except (AttributeError, KeyError, RecursionError, StopIteration, SourceValidationError,
            TypeError, ValueError):
        raise IdleSummaryBindingError("captured root source projection is invalid") from None
    return candidate, previous


def _validate_capture_binding(
    capture: object,
    binding: object,
    identity: _IdleSummaryMeasurementIdentity,
    binding_registry: IdleRuntimeBindingRegistry,
) -> tuple[IdleCheckpointCapture, IdleRuntimeBinding]:
    if type(capture) is not IdleCheckpointCapture:
        raise IdleSummaryBindingError("exact I2 IdleCheckpointCapture is required")
    if type(binding) is not IdleRuntimeBinding:
        raise IdleSummaryBindingError("exact I3b IdleRuntimeBinding is required")
    if type(binding_registry) is not IdleRuntimeBindingRegistry:
        raise IdleSummaryBindingError("exact I3b binding registry is required")
    actual_capture = cast(IdleCheckpointCapture, capture)
    actual_binding = cast(IdleRuntimeBinding, binding)
    try:
        _validate_capture_attestation(actual_capture)
    except (AttributeError, RecursionError, SourceValidationError, TypeError, ValueError):
        raise IdleSummaryBindingError("capture attestation is invalid") from None
    try:
        binding_registry.require_current(actual_binding, actual_binding.session_key)
    except Exception:
        raise IdleSummaryBindingError("binding is not current authority") from None
    snapshot = actual_capture.snapshot
    token = actual_capture.token
    try:
        runtime_checkpoint_present = token.pending_state.runtime_checkpoint_present
        current_token = capture_idle_token(
            snapshot,
            runtime_checkpoint_present=runtime_checkpoint_present,
        )
    except (AttributeError, SourceValidationError, TypeError, ValueError):
        raise IdleSummaryBindingError("capture token cannot be revalidated") from None
    if current_token is None or current_token != token:
        raise IdleSummaryBindingError("capture token and detached source changed")
    if actual_binding.source_store is not actual_capture._source:
        raise IdleSummaryBindingError("capture and binding use different source stores")
    if (
        token.session_key != actual_binding.session_key
        or token.incarnation != actual_binding.incarnation
        or token.generation != actual_binding.generation
        or token.store_identity != actual_binding.store_identity
        or token.source_revision != snapshot.state.revision
        or token.source_revision != actual_binding.revision
        or token.checkpoint_id != snapshot.state.checkpoint_id
        or token.checkpoint_id != actual_binding.checkpoint_id
        or token.checkpoint_fingerprint != snapshot.state.checkpoint_fingerprint
        or token.checkpoint_fingerprint != actual_binding.checkpoint_fingerprint
        or snapshot.state.session_key != actual_binding.session_key
        or snapshot.state.incarnation != actual_binding.incarnation
        or snapshot.state.generation != actual_binding.generation
        or snapshot.state.store_identity != actual_binding.store_identity
        or snapshot.state.revision != actual_binding.revision
        or snapshot.state.checkpoint_id != actual_binding.checkpoint_id
        or snapshot.state.checkpoint_fingerprint != actual_binding.checkpoint_fingerprint
    ):
        raise IdleSummaryBindingError("capture and binding source identity changed")
    policy = actual_binding.policy_binding
    if policy is None or actual_capture.policy_fingerprint != policy.fingerprint:
        raise IdleSummaryBindingError("capture and binding policy fingerprint changed")
    if actual_capture.runtime_fingerprint != identity.runtime_fingerprint:
        raise IdleSummaryBindingError("capture and qualified runtime fingerprint changed")
    runtime = actual_binding.auxiliary_runtime
    provider_name = getattr(runtime.provider, "provider_name", None)
    if (
        identity.provider_name != provider_name
        or identity.model != runtime.model
        or identity.context_window_tokens != runtime.context_window_tokens
    ):
        raise IdleSummaryBindingError("qualified measurer does not match binding runtime")
    dependencies = actual_capture.dependencies
    if dependencies is not None and (
        dependencies._source is not actual_binding.source_store
        or dependencies._policy_fingerprint != policy.fingerprint
    ):
        raise IdleSummaryBindingError("capture dependency capability changed")
    return actual_capture, actual_binding


def _verify_dependencies(
    verifier: object | None,
    *,
    capture: IdleCheckpointCapture,
    binding: IdleRuntimeBinding,
    references: tuple[str, ...],
) -> None:
    if not references:
        return
    if verifier is None or not callable(getattr(verifier, "verify", None)):
        raise IdleSummaryDependencyError("dependent summary requires an injected verifier")
    try:
        result = verifier.verify(capture=capture, binding=binding, references=references)
    except Exception:
        raise IdleSummaryDependencyError("summary dependency verification failed") from None
    _validate_dependency_result(result, capture=capture, binding=binding, references=references)


def _validate_dependency_result(
    result: object,
    *,
    capture: IdleCheckpointCapture,
    binding: IdleRuntimeBinding,
    references: tuple[str, ...],
) -> None:
    if type(result) is not IdleDependencyVerification:
        raise IdleSummaryDependencyError("summary dependency verifier returned no bound result")
    if (
        result.capture is not capture
        or result.binding is not binding
        or result.references != references
        or result.policy_fingerprint != binding.policy_binding.fingerprint
        or result.artifact_namespace is not binding.artifact_namespace
    ):
        raise IdleSummaryDependencyError("summary dependency verification changed state")


async def _join_owned_task(task: asyncio.Task[object]) -> None:
    """Cancel and fully join an owned callback despite repeated cancellation.

    Joining can extend beyond the operation deadline.  The injected callback
    must release retained request references after settlement, and I4 must
    establish a finite cleanup bound before activation.
    """
    if not task.done():
        task.cancel()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            continue
        except BaseException:
            break
    try:
        await asyncio.shield(task)
    except BaseException:
        # Cleanup errors must never replace the original timeout/cancellation.
        pass


async def _verify_dependencies_async(
    verifier: object | None,
    *,
    capture: IdleCheckpointCapture,
    binding: IdleRuntimeBinding,
    references: tuple[str, ...],
    deadline: float,
    clock: Callable[[], float],
) -> None:
    if not references:
        return
    if verifier is None or not callable(getattr(verifier, "verify", None)):
        raise IdleSummaryDependencyError("dependent summary requires an injected verifier")

    async def invoke() -> object:
        result = verifier.verify(capture=capture, binding=binding, references=references)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    remaining = deadline - clock()
    if not math.isfinite(remaining) or remaining <= 0:
        raise IdleSummaryTimeoutError("summary deadline elapsed before dependency verification")
    task = asyncio.create_task(invoke())
    try:
        result = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
    except asyncio.TimeoutError:
        await _join_owned_task(task)
        raise IdleSummaryTimeoutError("summary dependency verification timed out") from None
    except asyncio.CancelledError:
        await _join_owned_task(task)
        raise
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        raise IdleSummaryDependencyError("summary dependency verification failed") from None
    _validate_dependency_result(result, capture=capture, binding=binding, references=references)


class IdleSummaryAdapter:
    """Fold detached root evidence through one sealed transport capability.

    The constructor accepts a trusted local request measurer separately from a
    process/store/binding-bound transport.  It returns only in-memory results;
    no provider, scanner, publisher, conversation state, retry, or fallback is
    constructed here.
    """

    def __init__(
        self,
        transport: IdleSummaryQualifiedTransport,
        limits: IdleSummaryLimits,
        *,
        request_measurer: IdleSummaryRequestMeasurer | None = None,
        binding_registry: IdleRuntimeBindingRegistry | None = None,
        dependency_verifier: IdleSummaryDependencyVerifier | None = None,
        clock: Callable[[], float] = monotonic,
        accounting_sink: Callable[[IdleSummaryAttemptAccounting], object] | None = None,
    ) -> None:
        if type(binding_registry) is not IdleRuntimeBindingRegistry:
            raise IdleSummaryCapabilityError("exact idle runtime binding registry is required")
        if type(limits) is not IdleSummaryLimits:
            raise IdleSummaryCapabilityError("explicit IdleSummaryLimits are required")
        if not callable(clock):
            raise IdleSummaryCapabilityError("monotonic clock is invalid")
        if accounting_sink is not None and not callable(accounting_sink):
            raise IdleSummaryCapabilityError("accounting sink is invalid")
        actual_registry = cast(IdleRuntimeBindingRegistry, binding_registry)
        self._binding_registry = actual_registry
        self._transport = _qualified_transport(
            transport,
            binding_registry=actual_registry,
            source_store=transport.source_store if type(transport) is IdleSummaryQualifiedTransport else object(),
        )
        self._measurer, self._measurement_identity = _qualified_measurer(request_measurer)
        if self._measurement_identity != _transport_identity(self._transport):
            raise IdleSummaryCapabilityError("measurer and transport identities disagree")
        binding_provider, binding_model, binding_window = _binding_runtime_identity(
            self._transport.binding
        )
        if (
            self._measurement_identity.provider_name != binding_provider
            or self._measurement_identity.model != binding_model
            or self._measurement_identity.context_window_tokens != binding_window
        ):
            raise IdleSummaryCapabilityError("measurer and binding identities disagree")
        self._limits = limits
        self._clock = clock
        self._dependency_verifier = dependency_verifier
        self._accounting_sink = accounting_sink
        self._last_accounting = IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)

    @property
    def provider(self) -> IdleSummaryQualifiedTransport:
        """Compatibility alias for the sealed transport capability."""
        return self._transport

    @property
    def transport(self) -> IdleSummaryQualifiedTransport:
        return self._transport

    @property
    def request_measurer(self) -> IdleSummaryRequestMeasurer:
        return self._measurer

    @property
    def binding_registry(self) -> IdleRuntimeBindingRegistry:
        return self._binding_registry

    @property
    def limits(self) -> IdleSummaryLimits:
        return self._limits

    @property
    def last_accounting(self) -> IdleSummaryAttemptAccounting:
        """Latest ledger snapshot, including a caller-cancelled computation."""
        return self._last_accounting

    @property
    def accounting(self) -> IdleSummaryAttemptAccounting:
        return self._last_accounting

    def _publish_accounting(
        self,
        ledger: _IdleSummaryAccountingLedger,
        *,
        terminal: bool = False,
    ) -> IdleSummaryAttemptAccounting:
        # Intermediate observations are explicitly non-terminal.  A caller
        # must not mistake the first fully measured attempt in a multi-chunk
        # computation for the final cost of the whole computation.
        snapshot = ledger.snapshot(terminal=terminal)
        self._last_accounting = snapshot
        if self._accounting_sink is not None:
            try:
                self._accounting_sink(snapshot)
            except BaseException:
                # Accounting observation cannot replace the operation outcome.
                pass
        return snapshot

    def _finalize_accounting(self, ledger: _IdleSummaryAccountingLedger) -> IdleSummaryAttemptAccounting:
        """Close one computation without replacing an original exception."""
        if ledger.finalized:
            return ledger.snapshot(terminal=True)
        ledger.finalize()
        return self._publish_accounting(ledger, terminal=True)

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise IdleSummaryError("monotonic clock returned an invalid value")
        return float(value)

    def _require_current(self, binding: IdleRuntimeBinding) -> None:
        """Reconfirm authority immediately before crossing a disclosure boundary."""
        try:
            self._binding_registry.require_current(binding, binding.session_key)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            raise IdleSummaryBindingError("binding is no longer current authority") from None

    def _require_transport_current(
        self,
        binding: IdleRuntimeBinding,
        source_store: object,
    ) -> None:
        """Revalidate the sealed live transport and its current binding."""
        try:
            _qualified_transport(
                self._transport,
                binding_registry=self._binding_registry,
                source_store=source_store,
                binding=binding,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            raise IdleSummaryBindingError("qualified transport authority changed") from None
        if _transport_identity(self._transport) != self._measurement_identity:
            raise IdleSummaryBindingError("qualified transport identity changed")

    def _require_measurer_identity(self) -> _IdleSummaryMeasurementIdentity:
        """Re-read the live measurer and compare it with frozen authority."""
        try:
            identity = _read_measurer_identity(
                self._measurer,
                error_message="request measurer identity is invalid",
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except IdleSummaryCapabilityError:
            raise IdleSummaryBindingError("request measurer identity is invalid") from None
        expected = self._measurement_identity
        if identity.provider_name != expected.provider_name:
            raise IdleSummaryBindingError("request provider identity changed")
        if identity.transport != expected.transport:
            raise IdleSummaryBindingError("request transport identity changed")
        if identity.model != expected.model:
            raise IdleSummaryBindingError("request model identity changed")
        if identity.context_window_tokens != expected.context_window_tokens:
            raise IdleSummaryBindingError("request context window identity changed")
        if identity.runtime_fingerprint != expected.runtime_fingerprint:
            raise IdleSummaryBindingError("request runtime fingerprint changed")
        return identity

    def _require_measurement_authority(
        self,
        binding: IdleRuntimeBinding,
        source_store: object,
    ) -> None:
        """Require both live measurement identity and sealed transport authority."""
        self._require_measurer_identity()
        self._require_transport_current(binding, source_store)

    def _deadline(self, requested: float) -> float:
        return min(requested, self._now() + float(self._limits.max_wall_seconds))

    def _prepare_payload(
        self,
        candidate: SourceCompactionCandidate,
        boundary: int,
        after: int,
        previous_summary: str | None,
        consumed: Sequence[ConsumedSummaryChunk],
    ) -> _PreparedCall:
        evidence = candidate.summary_evidence(boundary, after=after)
        if not evidence.nodes:
            raise IdleSummaryNoProgressError("summary boundary has no visible evidence")
        try:
            payload_list = Consolidator.source_summary_messages(
                [node.message for node in evidence.nodes], previous_summary
            )
            payload = tuple(deepcopy(payload_list))
            prospective = SummaryDependencyProof(
                candidate,
                tuple([*consumed, ConsumedSummaryChunk(evidence, previous_summary, "preflight")]),
            )
            references = derive_summary_references(prospective, candidate.snapshot)
        except (SourceValidationError, TypeError, ValueError):
            raise IdleSummaryError("canonical summary evidence is invalid") from None
        return _PreparedCall(boundary, evidence, payload, references)

    async def _preflight_indivisible_groups(
        self,
        candidate: SourceCompactionCandidate,
        boundaries: Sequence[int],
        *,
        previous_summary: str | None,
        capture: IdleCheckpointCapture,
        binding: IdleRuntimeBinding,
        deadline: float,
        measurement_cache: dict[tuple[object, ...], _IdleSummaryMeasurementSnapshot],
    ) -> None:
        """Reject impossible atomic groups before any dispatch.

        A cumulative boundary may be too large while an earlier closed prefix
        and its later remainder both fit.  The preflight therefore advances
        through adjacent closed groups and measures each group's canonical
        payload independently.  Exact measurement decides whether each atomic
        group can fit; no reservation or transport dispatch occurs here.
        """
        group_start = candidate.previous_boundary
        prior_summary = previous_summary
        # Dependency proofs are contiguous even though the future folded
        # summary is unavailable during this pass.  Keep a local placeholder
        # chain for reference derivation; it never enters a measured payload.
        proof_previous_summary = previous_summary
        proof_chunks: list[ConsumedSummaryChunk] = []
        for boundary in boundaries:
            if self._now() >= deadline:
                raise IdleSummaryTimeoutError("summary deadline elapsed during preflight")
            evidence = candidate.summary_evidence(boundary, after=group_start)
            if not evidence.nodes:
                raise IdleSummaryNoProgressError("summary boundary has no visible evidence")
            try:
                payload = tuple(
                    deepcopy(
                        Consolidator.source_summary_messages(
                            [node.message for node in evidence.nodes], prior_summary
                        )
                    )
                )
                proof_chunks.append(
                    ConsumedSummaryChunk(evidence, proof_previous_summary, "preflight")
                )
                references = derive_summary_references(
                    SummaryDependencyProof(candidate, tuple(proof_chunks)),
                    candidate.snapshot,
                )
            except (SourceValidationError, TypeError, ValueError):
                raise IdleSummaryError("canonical summary evidence is invalid") from None
            prepared = _PreparedCall(boundary, evidence, payload, references)
            self._require_current(binding)
            self._require_transport_current(binding, capture._source)
            await _verify_dependencies_async(
                self._dependency_verifier,
                capture=capture,
                binding=binding,
                references=prepared.references,
                deadline=deadline,
                clock=self._now,
            )
            self._require_current(binding)
            self._require_transport_current(binding, capture._source)
            prepared = self._measure(
                prepared,
                deadline,
                measurement_cache,
                binding=binding,
                source_store=capture._source,
            )
            self._require_current(binding)
            self._require_transport_current(binding, capture._source)
            receipt = cast(IdleSummaryRequestReceipt, prepared.receipt)
            if (
                receipt.input_tokens > self._limits.max_input_tokens
                or receipt.total_reservation_tokens > self._measurement_identity.context_window_tokens
                or receipt.total_reservation_tokens > self._limits.max_total_tokens
            ):
                raise IdleSummaryBudgetError("no indivisible source group fits the summary budget")
            group_start = boundary
            # A future folded summary is provider output and is therefore not
            # known during this pass.  Its exact payload is checked with the
            # injected measurer immediately before that call.
            prior_summary = None
            proof_previous_summary = "preflight"

    def _measure(
        self,
        prepared: _PreparedCall,
        deadline: float,
        measurement_cache: dict[tuple[object, ...], _IdleSummaryMeasurementSnapshot],
        *,
        binding: IdleRuntimeBinding | None = None,
        source_store: object | None = None,
    ) -> _PreparedCall:
        if self._now() >= deadline:
            raise IdleSummaryTimeoutError("summary deadline elapsed before measurement")
        authority_binding = self._transport.binding if binding is None else binding
        authority_source = self._transport.source_store if source_store is None else source_store
        try:
            finalized_digest = _request_digest(prepared.payload)
            isolated_payload = deepcopy(prepared.payload)
            isolated_digest = _request_digest(isolated_payload)
        except (IdleSummaryCapabilityError, RecursionError, TypeError, ValueError):
            raise IdleSummaryBudgetError("summary request cannot be measured") from None
        if isolated_digest != finalized_digest:
            raise IdleSummaryBudgetError("summary request copy changed before measurement")
        try:
            measure_request = getattr(self._measurer, "measure_request")
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            raise IdleSummaryBudgetError("finalized request measurement failed") from None
        if not callable(measure_request):
            raise IdleSummaryBudgetError("finalized request measurement failed")
        # This is the last synchronous boundary before evidence reaches the
        # injected callback.  Both authorities must still match the identity
        # frozen at adapter construction.
        self._require_measurement_authority(authority_binding, authority_source)
        try:
            raw_receipt = measure_request(
                isolated_payload,
                max_output_tokens=self._limits.max_output_tokens,
                safety_buffer=self._limits.safety_buffer,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            raise IdleSummaryBudgetError("finalized request measurement failed") from None
        # Re-read both authorities before inspecting or retaining callback
        # output.  A callback cannot authorize its own identity mutation.
        self._require_measurement_authority(authority_binding, authority_source)
        try:
            if (
                _request_digest(prepared.payload) != finalized_digest
                or _request_digest(isolated_payload) != isolated_digest
            ):
                raise IdleSummaryBudgetError("request measurer mutated finalized payload")
        except (IdleSummaryCapabilityError, RecursionError, TypeError, ValueError):
            raise IdleSummaryBudgetError("request measurer mutated finalized payload") from None
        # Measurement is a trusted synchronous boundary, so cancellation
        # cannot interrupt it.  Recheck the operation deadline before allowing
        # its receipt to become an admissible reservation.
        if self._now() >= deadline:
            raise IdleSummaryTimeoutError("summary deadline elapsed during measurement")
        try:
            receipt = _validate_receipt(
                raw_receipt,
                identity=self._measurement_identity,
                limits=self._limits,
                messages=prepared.payload,
            )
            measurement = _IdleSummaryMeasurementSnapshot(
                finalized_digest,
                self._measurement_identity.provider_name,
                self._measurement_identity.transport,
                self._measurement_identity.model,
                self._measurement_identity.context_window_tokens,
                self._measurement_identity.runtime_fingerprint,
                self._limits.max_output_tokens,
                self._limits.safety_buffer,
                receipt.provider,
                receipt.transport,
                receipt.model,
                receipt.input_tokens,
                receipt.reserved_output_tokens,
                receipt.safety_buffer,
                receipt.model_window_tokens,
                receipt.total_reservation_tokens,
                receipt.request_digest,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except (IdleSummaryBindingError, IdleSummaryBudgetError):
            raise
        except BaseException:
            raise IdleSummaryBudgetError("provider request measurement is invalid") from None
        previous = measurement_cache.get(measurement.cache_key)
        if previous is not None and previous != measurement:
            raise IdleSummaryBudgetError("repeated request measurement changed")
        measurement_cache[measurement.cache_key] = measurement
        if self._now() >= deadline:
            raise IdleSummaryTimeoutError("summary deadline elapsed validating measurement")
        return _PreparedCall(
            prepared.boundary,
            prepared.evidence,
            prepared.payload,
            prepared.references,
            measurement.request_receipt(),
        )

    @staticmethod
    async def _join_cancelled(task: asyncio.Task[object]) -> None:
        await _join_owned_task(task)

    async def _attempt(
        self,
        request: IdleSummaryRequest,
        deadline: float,
        on_started: Callable[[], object],
    ) -> IdleSummaryAttemptResponse:
        remaining = deadline - self._now()
        if not math.isfinite(remaining) or remaining <= 0:
            raise IdleSummaryTimeoutError("summary deadline elapsed before dispatch")

        async def invoke() -> IdleSummaryAttemptResponse:
            if deadline - self._now() <= 0:
                raise IdleSummaryTimeoutError("summary deadline elapsed before dispatch")
            self._transport._admit_request(request)
            if deadline - self._now() <= 0:
                raise IdleSummaryTimeoutError("summary deadline elapsed before dispatch")
            on_started()
            result = self._transport._invoke_admitted(request)
            return cast(IdleSummaryAttemptResponse, await result)

        task = asyncio.create_task(invoke())
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.TimeoutError:
            await _join_owned_task(task)
            request = None  # type: ignore[assignment]
            raise IdleSummaryTimeoutError("summary deadline elapsed") from None
        except asyncio.CancelledError:
            if task.done() and not asyncio.current_task().cancelling():
                request = None  # type: ignore[assignment]
                raise _IdleSummaryAttemptCancelledError("summary attempt cancelled") from None
            await _join_owned_task(task)
            request = None  # type: ignore[assignment]
            raise

    def _failure(
        self,
        status: str,
        *,
        requests: list[IdleSummaryRequestReceipt],
        ledger: _IdleSummaryAccountingLedger,
        measured: IdleSummaryMeasuredUsage | None,
    ) -> IdleSummaryResult:
        accounting = self._finalize_accounting(ledger)
        reserved = IdleAuxiliaryUsage(
            calls=accounting.reserved_calls,
            tokens=accounting.reserved_tokens,
        )
        usage = (
            IdleAuxiliaryUsage(
                calls=accounting.dispatched_calls,
                tokens=accounting.known_measured_tokens,
            )
            if accounting.exact
            else reserved
        )
        return IdleSummaryResult(
            None,
            status,
            usage,
            accounting.exact,
            reserved,
            measured,
            tuple(requests),
            self._transport.provider_name,
            self._transport.model,
            accounting,
        )

    def _mark_unknown_if_needed(
        self,
        ledger: _IdleSummaryAccountingLedger,
        dispatched_before: int,
    ) -> None:
        if ledger.dispatched_calls > dispatched_before:
            ledger.unknown()
            self._publish_accounting(ledger)

    async def summarize(
        self,
        capture: IdleCheckpointCapture,
        binding: IdleRuntimeBinding,
        *,
        deadline: float,
    ) -> IdleSummaryResult:
        """Run one summary computation and terminalize its accounting on every exit."""
        ledger = _IdleSummaryAccountingLedger()
        requests: list[IdleSummaryRequestReceipt] = []
        measured_total: IdleSummaryMeasuredUsage | None = None
        try:
            deadline_value = _positive_deadline(deadline)
            operation_deadline = self._deadline(deadline_value)
            return await self._summarize_impl(
                capture,
                binding,
                operation_deadline=operation_deadline,
                ledger=ledger,
                requests=requests,
                measured_total=measured_total,
            )
        finally:
            self._finalize_accounting(ledger)

    async def _summarize_impl(
        self,
        capture: IdleCheckpointCapture,
        binding: IdleRuntimeBinding,
        *,
        operation_deadline: float,
        ledger: _IdleSummaryAccountingLedger,
        requests: list[IdleSummaryRequestReceipt],
        measured_total: IdleSummaryMeasuredUsage | None,
    ) -> IdleSummaryResult:
        """Return one final proposal or a terminal failed-attempt result."""
        measurement_cache: dict[tuple[object, ...], _IdleSummaryMeasurementSnapshot] = {}
        try:
            actual_capture, actual_binding = _validate_capture_binding(
                capture,
                binding,
                self._measurement_identity,
                self._binding_registry,
            )
            _qualified_transport(
                self._transport,
                binding_registry=self._binding_registry,
                source_store=actual_capture._source,
                binding=actual_binding,
            )
            candidate, previous_checkpoint = _checkpoint_from_capture(actual_capture)
            previous_summary = previous_checkpoint.summary if previous_checkpoint is not None else None
            visible_boundaries: list[int] = []
            for boundary in candidate.boundaries:
                evidence = candidate.summary_evidence(boundary, after=candidate.previous_boundary)
                if evidence.nodes:
                    visible_boundaries.append(boundary)
            if not visible_boundaries:
                raise IdleSummaryNoProgressError("captured source has no advancing visible evidence")
            await self._preflight_indivisible_groups(
                candidate,
                visible_boundaries,
                previous_summary=previous_summary,
                capture=actual_capture,
                binding=actual_binding,
                deadline=operation_deadline,
                measurement_cache=measurement_cache,
            )
        except IdleSummaryTimeoutError:
            return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise

        cursor = candidate.previous_boundary
        summary = previous_summary
        consumed: list[ConsumedSummaryChunk] = []

        while (
            cursor < visible_boundaries[-1]
            and ledger.reserved_calls < self._limits.max_calls
            and len(consumed) < self._limits.max_chunks
        ):
            if self._now() >= operation_deadline:
                return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
            chosen: _PreparedCall | None = None
            for boundary in reversed(visible_boundaries):
                if boundary <= cursor:
                    continue
                self._require_current(actual_binding)
                self._require_transport_current(actual_binding, actual_capture._source)
                prepared = self._prepare_payload(
                    candidate,
                    boundary,
                    cursor,
                    summary,
                    consumed,
                )
                self._require_current(actual_binding)
                try:
                    await _verify_dependencies_async(
                        self._dependency_verifier,
                        capture=actual_capture,
                        binding=actual_binding,
                        references=prepared.references,
                        deadline=operation_deadline,
                        clock=self._now,
                    )
                except IdleSummaryTimeoutError:
                    return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
                except asyncio.CancelledError:
                    self._finalize_accounting(ledger)
                    raise
                self._require_current(actual_binding)
                self._require_transport_current(actual_binding, actual_capture._source)
                try:
                    prepared = self._measure(
                        prepared,
                        operation_deadline,
                        measurement_cache,
                        binding=actual_binding,
                        source_store=actual_capture._source,
                    )
                except IdleSummaryTimeoutError:
                    return self._failure(
                        "timeout", requests=requests, ledger=ledger, measured=measured_total
                    )
                self._require_current(actual_binding)
                self._require_transport_current(actual_binding, actual_capture._source)
                receipt = cast(IdleSummaryRequestReceipt, prepared.receipt)
                if prepared.receipt.input_tokens > self._limits.max_input_tokens:
                    continue
                if (
                    prepared.receipt.total_reservation_tokens
                    > self._measurement_identity.context_window_tokens
                ):
                    continue
                if prepared.receipt.total_reservation_tokens > (
                    self._limits.max_total_tokens - ledger.reserved_tokens
                ):
                    continue
                chosen = prepared
                break
            if chosen is None:
                if ledger.reserved_calls == 0:
                    raise IdleSummaryBudgetError("no indivisible source group fits the summary budget")
                return self._failure("no_progress", requests=requests, ledger=ledger, measured=measured_total)

            if self._now() >= operation_deadline:
                return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
            self._require_current(actual_binding)
            self._require_transport_current(actual_binding, actual_capture._source)
            receipt = cast(IdleSummaryRequestReceipt, chosen.receipt)
            request = IdleSummaryRequest(chosen.payload, receipt)
            ledger.reserve(receipt)
            requests.append(receipt)
            self._publish_accounting(ledger)
            if self._now() >= operation_deadline:
                request = None  # type: ignore[assignment]
                return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
            self._require_current(actual_binding)
            dispatched_before = ledger.dispatched_calls
            try:
                response = await self._attempt(
                    request,
                    operation_deadline,
                    lambda: (ledger.dispatched(), self._publish_accounting(ledger))[0],
                )
            except IdleSummaryTimeoutError:
                self._mark_unknown_if_needed(ledger, dispatched_before)
                request = None  # type: ignore[assignment]
                return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
            except _IdleSummaryAttemptCancelledError:
                self._mark_unknown_if_needed(ledger, dispatched_before)
                request = None  # type: ignore[assignment]
                return self._failure("cancelled", requests=requests, ledger=ledger, measured=measured_total)
            except asyncio.CancelledError:
                self._mark_unknown_if_needed(ledger, dispatched_before)
                request = None  # type: ignore[assignment]
                self._finalize_accounting(ledger)
                raise
            except BaseException as exc:
                self._mark_unknown_if_needed(ledger, dispatched_before)
                self._publish_accounting(ledger)
                request = None  # type: ignore[assignment]
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                return self._failure("failed", requests=requests, ledger=ledger, measured=measured_total)

            if self._now() >= operation_deadline:
                self._mark_unknown_if_needed(ledger, dispatched_before)
                request = None  # type: ignore[assignment]
                return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
            try:
                measured = _extract_response_usage(
                    response,
                    request=request,
                    limits=self._limits,
                )
            except IdleSummaryError:
                self._mark_unknown_if_needed(ledger, dispatched_before)
                request = None  # type: ignore[assignment]
                return self._failure("invalid_response", requests=requests, ledger=ledger, measured=measured_total)
            ledger.known(measured)
            self._publish_accounting(ledger)
            measured_total = measured if measured_total is None else measured_total + measured
            if measured_total.total_tokens > self._limits.max_total_tokens:
                return self._failure("budget_exceeded", requests=requests, ledger=ledger, measured=measured_total)
            try:
                self._require_transport_current(actual_binding, actual_capture._source)
            except IdleSummaryBindingError:
                request = None  # type: ignore[assignment]
                return self._failure("stale_binding", requests=requests, ledger=ledger, measured=measured_total)
            try:
                response_summary = _validate_response_content(
                    response,
                    request=request,
                    transport=self._transport,
                    limits=self._limits,
                )
            except IdleSummaryError:
                request = None  # type: ignore[assignment]
                return self._failure("invalid_response", requests=requests, ledger=ledger, measured=measured_total)
            try:
                self._require_transport_current(actual_binding, actual_capture._source)
            except IdleSummaryBindingError:
                request = None  # type: ignore[assignment]
                return self._failure("stale_binding", requests=requests, ledger=ledger, measured=measured_total)
            summary = response_summary
            request = None  # type: ignore[assignment]
            consumed.append(ConsumedSummaryChunk(chosen.evidence, previous_summary, summary))
            previous_summary = summary
            cursor = chosen.boundary

        if not consumed or summary is None or cursor == candidate.previous_boundary:
            if ledger.reserved_calls == 0:
                raise IdleSummaryNoProgressError("summary produced no progress")
            return self._failure("no_progress", requests=requests, ledger=ledger, measured=measured_total)

        if self._now() >= operation_deadline:
            return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
        proof = SummaryDependencyProof(candidate, tuple(consumed))
        try:
            references = derive_summary_references(proof, candidate.snapshot)
            checkpoint_node = CheckpointNode(uuid4().hex, summary, references)
            rebuilt = candidate.rebuild(cursor, checkpoint_node)
            captured = candidate.snapshot
            checkpoint: dict[str, object] = {
                "schema_version": 2 if references else 1,
                "digest_version": 1,
                "summary_format_version": 1,
                "session_generation": captured.state.generation,
                "source_revision": captured.state.revision,
                "checkpoint_id": checkpoint_node.checkpoint_id,
                "parent_checkpoint_id": captured.state.checkpoint_id,
                "producing_model": actual_binding.auxiliary_runtime.model,
                "producing_provider": actual_binding.auxiliary_runtime.provider.provider_name,
                "digest_algorithm": "sha256",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "artifact_references": list(references),
                "boundary": cursor,
                "boundary_event_id": captured.events[cursor - 1]["_event_id"],
                "prefix_digest": prefix_digest(captured.events, cursor),
                "auxiliary_usage": {
                    "calls": ledger.dispatched_calls,
                    "tokens": measured_total.total_tokens if measured_total is not None else 0,
                    "input_tokens": measured_total.input_tokens if measured_total is not None else 0,
                    "output_tokens": measured_total.output_tokens if measured_total is not None else 0,
                    "total_tokens": measured_total.total_tokens if measured_total is not None else 0,
                },
                "projection_scope": "root",
                "projection_version": 1,
            }
            if measured_total is not None and measured_total.reasoning_tokens is not None:
                checkpoint["auxiliary_usage"]["reasoning_tokens"] = measured_total.reasoning_tokens  # type: ignore[index]
            validated = validate_context_checkpoint(
                checkpoint,
                captured.events,
                generation=captured.state.generation,
                source_revision=captured.state.revision,
                parent_checkpoint_id=captured.state.checkpoint_id,
            )
            if rebuilt is None:
                raise IdleSummaryError("summary rebuild failed")
        except (SourceValidationError, KeyError, TypeError, ValueError):
            return self._failure("invalid_proposal", requests=requests, ledger=ledger, measured=measured_total)

        if self._now() >= operation_deadline:
            return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
        try:
            self._require_transport_current(actual_binding, actual_capture._source)
        except IdleSummaryBindingError:
            return self._failure("stale_binding", requests=requests, ledger=ledger, measured=measured_total)
        if self._now() >= operation_deadline:
            return self._failure("timeout", requests=requests, ledger=ledger, measured=measured_total)
        accounting = self._finalize_accounting(ledger)
        if not accounting.exact or measured_total is None:
            return self._failure("unknown_usage", requests=requests, ledger=ledger, measured=measured_total)
        try:
            self._require_transport_current(actual_binding, actual_capture._source)
        except IdleSummaryBindingError:
            return self._failure("stale_binding", requests=requests, ledger=ledger, measured=measured_total)
        usage = IdleAuxiliaryUsage(calls=accounting.dispatched_calls, tokens=measured_total.total_tokens)
        proposal = IdleSummaryProposal(
            cast(dict[str, object], validated),
            proof,
            usage,
            measured_total,
            tuple(requests),
            self._transport.provider_name,
            self._transport.model,
        )
        return IdleSummaryResult(
            proposal,
            "complete",
            usage,
            True,
            IdleAuxiliaryUsage(calls=accounting.reserved_calls, tokens=accounting.reserved_tokens),
            measured_total,
            tuple(requests),
            proposal.provider,
            proposal.model,
            accounting,
        )

    async def __call__(
        self,
        capture: IdleCheckpointCapture,
        binding: IdleRuntimeBinding,
        *,
        deadline: float,
    ) -> IdleSummaryResult:
        return await self.summarize(capture, binding, deadline=deadline)


IdleSummaryProviderCapability = IdleSummaryQualifiedTransport
IdleSummaryUsageReceipt = IdleSummaryMeasuredUsage
IdleSummaryRequestInfo = IdleSummaryRequestReceipt
IdleSummaryAttempt = IdleSummaryQualifiedTransport
IdleSummaryUsage = IdleSummaryMeasuredUsage
IdleSummaryResponse = IdleSummaryAttemptResponse


__all__ = [
    "IdleDependencyVerification",
    "IdleQualifiedSummaryProvider",
    "IdleSummaryAttemptAccounting",
    "IdleSummaryAdapter",
    "IdleSummaryAttempt",
    "IdleSummaryAttemptCapability",
    "IdleSummaryAttemptResponse",
    "IdleSummaryBindingError",
    "IdleSummaryBudgetError",
    "IdleSummaryCapabilityError",
    "IdleSummaryDependencyError",
    "IdleSummaryDependencyVerifier",
    "IdleSummaryError",
    "IdleSummaryLimits",
    "IdleSummaryMeasuredUsage",
    "IdleSummaryNoProgressError",
    "IdleSummaryProposal",
    "IdleSummaryProviderCapability",
    "IdleSummaryQualifiedTransport",
    "IdleSummaryRequestMeasurer",
    "IdleSummaryRequest",
    "IdleSummaryRequestInfo",
    "IdleSummaryRequestReceipt",
    "IdleSummaryResult",
    "IdleSummaryResponse",
    "IdleSummaryTimeoutError",
    "IdleSummaryTransportRegistry",
    "IdleSummaryUsage",
    "IdleSummaryUsageReceipt",
    "QualifiedIdleSummaryAttempt",
]
