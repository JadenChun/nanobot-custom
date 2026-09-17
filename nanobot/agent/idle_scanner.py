"""Dormant I3d idle-compaction composition.

I3d is an explicitly injected adapter.  It composes the bounded source
discovery cursor, the existing generic coordinator, I3b runtime bindings, the
lease-free I2 capture/publication seam, and the stateless I3c summary adapter.
Construction performs no traversal, provider resolution, task creation, or
source write.  Production activation, cadence, and numeric policy selection
remain outside this module.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Protocol, cast

from nanobot.agent.idle_compaction import (
    IdleAuxiliaryLimits,
    IdleAuxiliaryUsage,
    IdleCompactionCoordinator,
    IdleCompactionPolicy,
    IdleComputation,
    IdlePublicationConflict,
)
from nanobot.agent.idle_runtime import (
    IdleRuntimeBinding,
    IdleRuntimeBindingRegistry,
    IdleRuntimeLimits,
    IdleRuntimeStaleError,
)
from nanobot.agent.idle_summary import (
    IdleSummaryAdapter,
    IdleSummaryAttemptAccounting,
    IdleSummaryLimits,
    IdleSummaryMeasuredUsage,
    IdleSummaryProposal,
    IdleSummaryRequestReceipt,
    IdleSummaryResult,
)
from nanobot.config.schema import IdleCompactionConfig
from nanobot.session.checkpoint_dependencies import SummaryDependencyProof
from nanobot.session.context_checkpoint import validate_context_checkpoint
from nanobot.session.idle_checkpoint import (
    IdleCheckpointCapture,
    IdleCheckpointConflict,
    IdleCheckpointError,
    IdlePolicyBinding,
)
from nanobot.session.idle_compaction import (
    IdleCaptureToken,
)
from nanobot.session.source import SourceValidationError
from nanobot.session.source_discovery import (
    SourceDiscoveryCursor,
    SourceDiscoveryError,
    SourceDiscoveryLimits,
    SourceDiscoveryResult,
    SourceDiscoveryRootChanged,
)
from nanobot.session.source_root import SourceRootAuthority, SourceRootAuthorityError
from nanobot.session.source_store import SourceSessionStore


class IdleScannerError(RuntimeError):
    """Base error for the dormant I3d composition."""


class IdleScannerIneligibleError(IdleScannerError):
    """A source or blocker snapshot cannot be admitted for idle work."""


IdleScannerIneligible = IdleScannerIneligibleError


class IdleScannerPublicationConflict(IdlePublicationConflict):
    """The captured proposal lost an authoritative publication race."""


@dataclass(frozen=True, slots=True)
class IdleBlockerSnapshot:
    """Content-free synchronous blocker authority for one canonical key."""

    revision: int
    clear: bool

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("blocker revision must be a non-negative integer")
        if type(self.clear) is not bool:
            raise TypeError("blocker clear must be a boolean")


class IdleBlockerAuthority(Protocol):
    """Mandatory injected synchronous runtime blocker capability."""

    def snapshot(self, session_key: str) -> IdleBlockerSnapshot: ...


class InMemoryIdleBlockerAuthority:
    """Small process-local blocker authority useful for deterministic tests.

    Missing keys are intentionally unavailable and therefore fail closed.  A
    state change increments the revision even when the new state has the same
    ``clear`` value, so a caller cannot reuse an old capture after an
    admission transition.
    """

    def __init__(self) -> None:
        self._states: dict[str, IdleBlockerSnapshot] = {}

    def snapshot(self, session_key: str) -> IdleBlockerSnapshot:
        value = self._states.get(session_key)
        if type(value) is not IdleBlockerSnapshot:
            raise IdleScannerIneligible("blocker authority is unavailable")
        return value

    def set_clear(self, session_key: str, clear: bool) -> IdleBlockerSnapshot:
        if type(session_key) is not str or not session_key:
            raise TypeError("session_key must be a non-empty string")
        if type(clear) is not bool:
            raise TypeError("clear must be a boolean")
        current = self._states.get(session_key)
        revision = 0 if current is None else current.revision + 1
        value = IdleBlockerSnapshot(revision, clear)
        self._states[session_key] = value
        return value

    def clear(self, session_key: str) -> IdleBlockerSnapshot:
        return self.set_clear(session_key, True)

    def block(self, session_key: str) -> IdleBlockerSnapshot:
        return self.set_clear(session_key, False)


@dataclass(slots=True)
class _PublicationState:
    consumed: bool = False
    completed: bool = False


@dataclass(frozen=True, slots=True)
class IdleScannerCapture:
    """I3d capture passed opaquely through the generic coordinator."""

    checkpoint: IdleCheckpointCapture
    binding: IdleRuntimeBinding
    blocker: IdleBlockerSnapshot
    summary_adapter: object = field(repr=False, compare=False)
    root_authority: SourceRootAuthority = field(repr=False, compare=False)
    _publication: _PublicationState = field(repr=False, compare=False, default_factory=_PublicationState)

    @property
    def capture(self) -> IdleCheckpointCapture:
        return self.checkpoint

    @property
    def i2_capture(self) -> IdleCheckpointCapture:
        return self.checkpoint


@dataclass(frozen=True, slots=True)
class IdleScannerComputation:
    """Validated I3c terminal result retained until the one-shot publisher."""

    result: IdleSummaryResult

    @property
    def proposal(self) -> IdleSummaryProposal:
        proposal = self.result.proposal
        if type(proposal) is not IdleSummaryProposal:
            raise IdleScannerError("summary computation has no proposal")
        return proposal

    @property
    def candidate(self) -> dict[str, object]:
        return self.proposal.checkpoint


def _as_policy(value: object | None) -> tuple[IdleCompactionPolicy, bool]:
    """Copy explicit I3 limits into the coordinator's generic policy."""

    if value is None:
        return IdleCompactionPolicy(), False
    if isinstance(value, IdleCompactionPolicy):
        return value, value.enabled
    limits = IdleRuntimeLimits.from_config(value)
    policy = IdleCompactionPolicy(
        idle_ttl_seconds=limits.idle_ttl_seconds,
        queue_capacity=limits.queue_capacity,
        max_concurrent_jobs=limits.max_concurrent_jobs,
        max_auxiliary_calls=limits.max_auxiliary_calls,
        max_auxiliary_tokens=limits.max_auxiliary_tokens,
        max_auxiliary_wall_seconds=limits.max_auxiliary_wall_seconds,
        initial_backoff_seconds=limits.initial_backoff_seconds,
        max_backoff_seconds=limits.max_backoff_seconds,
    )
    return policy, bool(limits.enabled and policy.enabled)


def _utc_now(clock: object) -> datetime:
    if not callable(clock):
        raise IdleScannerIneligible("UTC clock is unavailable")
    try:
        value = clock()
    except Exception:
        raise IdleScannerIneligible("UTC clock is unavailable") from None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise IdleScannerIneligible("UTC clock is not aware")
    value = value.astimezone(timezone.utc)
    if not math.isfinite(value.timestamp()):
        raise IdleScannerIneligible("UTC clock is invalid")
    return value


def _elapsed_idle(token: IdleCaptureToken, *, ttl_seconds: float, clock: object) -> bool:
    if type(ttl_seconds) not in (int, float) or isinstance(ttl_seconds, bool):
        return False
    if not math.isfinite(float(ttl_seconds)) or ttl_seconds <= 0:
        return False
    try:
        activity = datetime.fromisoformat(token.last_user_activity_at)
    except (TypeError, ValueError):
        return False
    if activity.tzinfo is None:
        return False
    try:
        now = _utc_now(clock)
        activity = activity.astimezone(timezone.utc)
        elapsed = (now - activity).total_seconds()
    except (OverflowError, ValueError, IdleScannerIneligible):
        return False
    # A future timestamp, including a rollback of an injected clock, is never
    # interpreted as an already-idle session.
    return math.isfinite(elapsed) and elapsed >= float(ttl_seconds)


def _adapter_runtime_fingerprint(adapter: object) -> str:
    candidates: list[object] = []
    transport = getattr(adapter, "transport", None)
    if transport is not None:
        candidates.append(getattr(transport, "runtime_fingerprint", None))
    candidates.append(getattr(adapter, "runtime_fingerprint", None))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    raise IdleScannerIneligible("qualified I3c runtime fingerprint is unavailable")


def _adapter_binding(adapter: object) -> IdleRuntimeBinding | None:
    transport = getattr(adapter, "transport", None)
    value = getattr(transport, "binding", None) if transport is not None else None
    if type(value) is IdleRuntimeBinding:
        return value
    value = getattr(adapter, "binding", None)
    return value if type(value) is IdleRuntimeBinding else None


class IdleCompactionScanner:
    """Explicit one-shot I3d scanner and coordinator composition."""

    def __init__(
        self,
        *,
        source_store: SourceSessionStore,
        binding_registry: IdleRuntimeBindingRegistry | None = None,
        blocker_authority: IdleBlockerAuthority | object | None = None,
        summary_adapter: IdleSummaryAdapter | object | None = None,
        summary_adapter_factory: object | None = None,
        summary_factory: object | None = None,
        policy: IdleCompactionPolicy | IdleRuntimeLimits | IdleCompactionConfig | None = None,
        limits: IdleCompactionPolicy | IdleRuntimeLimits | IdleCompactionConfig | None = None,
        config: IdleCompactionPolicy | IdleRuntimeLimits | IdleCompactionConfig | None = None,
        discovery: SourceDiscoveryCursor | None = None,
        discovery_limits: SourceDiscoveryLimits | None = None,
        coordinator: IdleCompactionCoordinator[object, object] | None = None,
        utc_clock: object | None = None,
        clock: object | None = None,
        monotonic_clock: object | None = None,
    ) -> None:
        if type(source_store) is not SourceSessionStore:
            raise TypeError("source_store must be an exact SourceSessionStore")
        if binding_registry is not None and type(binding_registry) is not IdleRuntimeBindingRegistry:
            raise TypeError("binding_registry must be an exact IdleRuntimeBindingRegistry")
        if policy is not None and (limits is not None or config is not None):
            raise ValueError("choose one of policy, limits, or config")
        if limits is not None and config is not None:
            raise ValueError("choose one of limits or config")
        chosen_limits = policy if policy is not None else limits if limits is not None else config
        self._policy, self._enabled = _as_policy(chosen_limits)
        self._source = source_store
        self._registry = binding_registry
        self._blockers = blocker_authority
        factories = [value for value in (summary_adapter_factory, summary_factory) if value is not None]
        if len(factories) == 2 and factories[0] is not factories[1]:
            raise ValueError("summary_adapter_factory and summary_factory must agree")
        self._summary_adapter = summary_adapter
        self._summary_factory = factories[0] if factories else None
        if self._summary_adapter is not None and self._summary_factory is not None:
            raise ValueError("choose summary_adapter or summary_adapter_factory")
        if discovery is not None and type(discovery) is not SourceDiscoveryCursor:
            raise TypeError("discovery must be an exact SourceDiscoveryCursor")
        if discovery is not None and discovery._backend is not source_store._backend:
            raise ValueError("discovery and source_store must share the exact backend")
        self._root_authority: SourceRootAuthority | None = None
        if self._enabled:
            if discovery is not None:
                self._root_authority = discovery.root_authority
            else:
                self._root_authority = source_store.create_source_root_authority()
        if discovery is None and discovery_limits is not None:
            if self._root_authority is None:
                raise IdleScannerIneligible("source-root authority is unavailable")
            created = SourceDiscoveryCursor(
                source_store._backend,
                limits=discovery_limits,
                root_authority=self._root_authority,
            )
            if type(created) is not SourceDiscoveryCursor:
                raise TypeError("source store returned an invalid discovery cursor")
            discovery = created
        self._discovery = discovery
        chosen_clock = utc_clock if utc_clock is not None else clock
        self._utc_clock = chosen_clock if chosen_clock is not None else (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock if monotonic_clock is not None else monotonic
        self._closed = False
        self._state_lock = asyncio.Lock()
        self._scan_task: asyncio.Task[int] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._last_discovery: SourceDiscoveryResult | None = None

        if coordinator is not None and type(coordinator) is not IdleCompactionCoordinator:
            raise TypeError("coordinator must be an exact IdleCompactionCoordinator")
        if coordinator is not None:
            self._coordinator = coordinator
        else:
            self._coordinator = IdleCompactionCoordinator(
                policy=self._policy if self._enabled else IdleCompactionPolicy(),
                enumerate_sessions=self._enumerate_sessions,
                capture=self._capture,
                compute=self._compute,
                validate=self._validate,
                publish=self._publish,
            )

    @property
    def coordinator(self) -> IdleCompactionCoordinator[object, object]:
        return self._coordinator

    @property
    def discovery(self) -> SourceDiscoveryCursor | None:
        return self._discovery

    @property
    def root_authority(self) -> SourceRootAuthority | None:
        return self._root_authority

    @property
    def last_discovery(self) -> SourceDiscoveryResult | None:
        return self._last_discovery

    @property
    def closed(self) -> bool:
        return self._closed

    async def _enumerate_sessions(self) -> tuple[str, ...]:
        if not self._enabled or self._discovery is None:
            return ()
        try:
            result = await self._discovery.scan_once()
        except SourceDiscoveryRootChanged:
            self._last_discovery = None
            if self._registry is not None:
                try:
                    self._registry.invalidate()
                except Exception:
                    pass
            self._invalidate_root()
            return ()
        except SourceDiscoveryError:
            # A root replacement permanently invalidates the process-local
            # cursor.  Keep subsequent explicit scans inert until the owner
            # supplies a fresh cursor rather than traversing a new root.
            return ()
        self._last_discovery = result
        return result.keys

    def _invalidate_root(self) -> None:
        """Retire the cursor and descriptor after a root reachability loss."""

        if self._discovery is not None:
            self._discovery.close()
        if self._root_authority is not None:
            self._root_authority.close()

    def _read_blocker(self, key: str) -> IdleBlockerSnapshot | None:
        authority = self._blockers
        if authority is None:
            return None
        method = getattr(authority, "snapshot", None)
        if not callable(method):
            return None
        try:
            snapshot = method(key)
        except Exception:
            return None
        return snapshot if type(snapshot) is IdleBlockerSnapshot and snapshot.clear else None

    async def _resolve_adapter(self, binding: IdleRuntimeBinding) -> object:
        if self._summary_factory is None:
            adapter = self._summary_adapter
        else:
            try:
                adapter = self._summary_factory(binding)  # type: ignore[misc]
                if inspect.isawaitable(adapter):
                    adapter = await cast(Any, adapter)
            except Exception:
                raise IdleScannerIneligible("I3c adapter is unavailable") from None
        if type(adapter) is not IdleSummaryAdapter or not callable(getattr(adapter, "summarize", None)):
            raise IdleScannerIneligible("I3c adapter is unavailable")
        self._validate_adapter_binding(adapter, binding)
        return adapter

    def _validate_adapter_binding(
        self,
        adapter: object,
        binding: IdleRuntimeBinding,
    ) -> IdleSummaryLimits:
        if type(adapter) is not IdleSummaryAdapter:
            raise IdleScannerIneligible("I3c adapter is unavailable")
        limits = getattr(adapter, "limits", None)
        if type(limits) is not IdleSummaryLimits:
            raise IdleScannerIneligible("I3c limits are unavailable")
        coordinator_limits = self._policy.auxiliary_limits
        if (
            limits.max_calls > coordinator_limits.max_calls
            or limits.max_total_tokens > coordinator_limits.max_tokens
            or limits.max_wall_seconds > coordinator_limits.max_wall_seconds
        ):
            raise IdleScannerIneligible("I3c limits exceed coordinator ceilings")
        adapter_binding = _adapter_binding(adapter)
        transport = getattr(adapter, "transport", None)
        if (
            adapter_binding is not binding
            or getattr(transport, "binding", None) is not binding
            or getattr(transport, "source_store", None) is not self._source
            or getattr(adapter, "binding_registry", None) is not self._registry
        ):
            raise IdleScannerIneligible("I3c adapter authority changed")
        _adapter_runtime_fingerprint(adapter)
        return limits

    async def _capture(self, key: str) -> IdleScannerCapture | None:
        if (
            not self._enabled
            or self._registry is None
            or self._blockers is None
            or self._root_authority is None
            or type(key) is not str
            or not key
        ):
            return None
        try:
            self._root_authority.require_current(self._source._backend)
            binding = await self._registry.resolve_current_async(
                key,
                root_authority=self._root_authority,
            )
            adapter = await self._resolve_adapter(binding)
            policy = binding.policy_binding
            if type(policy) is not IdlePolicyBinding:
                return None
            runtime_fingerprint = _adapter_runtime_fingerprint(adapter)
            checkpoint = self._source.capture_idle_checkpoint(
                key,
                runtime_fingerprint=runtime_fingerprint,
                policy_fingerprint=policy.fingerprint,
                policy=policy,
                root_authority=self._root_authority,
            )
            if type(checkpoint) is not IdleCheckpointCapture:
                return None
            token = checkpoint.token
            if not _elapsed_idle(
                token,
                ttl_seconds=self._policy.idle_ttl_seconds,
                clock=self._utc_clock,
            ):
                return None
            blocker = self._read_blocker(key)
            if blocker is None:
                return None
            if not await self._registry.revalidate_async(
                binding,
                key,
                root_authority=self._root_authority,
            ):
                return None
            return IdleScannerCapture(
                checkpoint,
                binding,
                blocker,
                adapter,
                self._root_authority,
            )
        except asyncio.CancelledError:
            raise
        except SourceRootAuthorityError:
            self._invalidate_root()
            return None
        except Exception:
            try:
                if self._root_authority is not None:
                    self._root_authority.require_current(self._source._backend)
            except SourceRootAuthorityError:
                self._invalidate_root()
            return None

    async def _compute(
        self,
        key: str,
        captured: object,
        limits: IdleAuxiliaryLimits,
    ) -> IdleComputation[IdleScannerComputation]:
        if type(captured) is not IdleScannerCapture:
            raise IdleScannerError("invalid I3d capture")
        item = cast(IdleScannerCapture, captured)
        summarize = getattr(item.summary_adapter, "summarize", None)
        if not callable(summarize):
            raise IdleScannerError("I3c adapter is unavailable")
        if item.root_authority is not self._root_authority:
            raise IdleScannerIneligible("source-root authority changed")
        item.root_authority.require_current(self._source._backend)
        self._validate_adapter_binding(item.summary_adapter, item.binding)
        # I3c owns its request reservation and provider dispatch bounds, but
        # its configured ceilings must remain within the coordinator's
        # immutable limits.  Binding validation performed immediately above
        # rechecks those ceilings after capture and before any request work.
        deadline_value = self._monotonic_clock()
        if (
            isinstance(deadline_value, bool)
            or not isinstance(deadline_value, (int, float))
            or not math.isfinite(float(deadline_value))
        ):
            raise IdleScannerError("monotonic clock is invalid")
        deadline = float(deadline_value) + limits.max_wall_seconds
        result = summarize(item.checkpoint, item.binding, deadline=deadline)
        if not inspect.isawaitable(result):
            raise IdleScannerError("I3c adapter did not return an awaitable")
        result = await cast(Any, result)
        if type(result) is not IdleSummaryResult:
            raise IdleScannerError("I3c result is malformed")
        if result.status in {"timeout", "budget_exceeded", "unknown_usage"}:
            from nanobot.agent.idle_compaction import IdleAuxiliaryBudgetError

            raise IdleAuxiliaryBudgetError("I3c auxiliary budget was not met")
        if result.status != "complete" and result.status not in {"no_progress", "stale_binding"}:
            raise IdleScannerError("I3c computation failed")
        usage = result.usage if type(result.usage) is IdleAuxiliaryUsage else None
        return IdleComputation(
            IdleScannerComputation(result),
            usage,
        )

    @staticmethod
    def _proposal_boundary(capture: IdleCheckpointCapture) -> int:
        checkpoint = capture.snapshot.checkpoint
        if checkpoint is None:
            return 0
        boundary = checkpoint.get("boundary")
        return boundary if type(boundary) is int and boundary >= 0 else -1

    def _validate_result(
        self,
        captured: IdleScannerCapture,
        computation: IdleScannerComputation,
        limits: IdleAuxiliaryLimits,
    ) -> bool:
        result = computation.result
        proposal = result.proposal
        if (
            result.status != "complete"
            or type(proposal) is not IdleSummaryProposal
            or result.usage_exact is not True
            or type(result.usage) is not IdleAuxiliaryUsage
            or result.usage != proposal.usage
            or result.usage.calls <= 0
            or result.usage.calls > limits.max_calls
            or result.usage.tokens > limits.max_tokens
        ):
            return False
        accounting = result.accounting
        if (
            type(accounting) is not IdleSummaryAttemptAccounting
            or accounting.exact is not True
            or accounting.unknown_dispatched != 0
            or accounting.reserved_calls != accounting.dispatched_calls
            or accounting.dispatched_calls != len(result.requests)
            or accounting.dispatched_calls != result.usage.calls
            or accounting.reserved_calls > limits.max_calls
            or accounting.reserved_tokens > limits.max_tokens
            or accounting.known_measured_tokens != result.usage.tokens
            or type(result.reserved_usage) is not IdleAuxiliaryUsage
            or result.reserved_usage.calls != accounting.reserved_calls
            or result.reserved_usage.calls > limits.max_calls
            or result.reserved_usage.tokens > limits.max_tokens
            or result.reserved_usage.tokens < result.usage.tokens
            or type(result.measured_usage) is not IdleSummaryMeasuredUsage
            or type(proposal.measured_usage) is not IdleSummaryMeasuredUsage
            or result.measured_usage != proposal.measured_usage
            or type(proposal.usage) is not IdleAuxiliaryUsage
            or proposal.usage != result.usage
            or proposal.requests != result.requests
            or result.provider != proposal.provider
            or result.model != proposal.model
        ):
            return False
        if any(
            type(value) is not int or value < 0
            for value in (
                result.usage.calls,
                result.usage.tokens,
                result.reserved_usage.calls,
                result.reserved_usage.tokens,
                accounting.reserved_calls,
                accounting.reserved_tokens,
                accounting.dispatched_calls,
                accounting.known_measured_tokens,
                accounting.unknown_dispatched,
            )
        ):
            return False
        if (
            type(result.requests) is not tuple
            or any(type(receipt) is not IdleSummaryRequestReceipt for receipt in result.requests)
            or accounting.reserved_tokens
            != sum(receipt.total_reservation_tokens for receipt in result.requests)
            or result.measured_usage.total_tokens != result.usage.tokens
            or proposal.measured_usage.total_tokens != result.usage.tokens
        ):
            return False
        transport = getattr(captured.summary_adapter, "transport", None)
        expected_provider = getattr(transport, "provider_name", None)
        expected_transport = getattr(transport, "transport", None)
        expected_model = getattr(transport, "model", None)
        if (
            not isinstance(expected_provider, str)
            or not isinstance(expected_transport, str)
            or not isinstance(expected_model, str)
            or result.provider != expected_provider
            or result.model != expected_model
            or any(
                receipt.provider != expected_provider
                or receipt.transport != expected_transport
                or receipt.model != expected_model
                for receipt in result.requests
            )
        ):
            return False
        checkpoint = proposal.checkpoint
        if not isinstance(checkpoint, dict):
            return False
        try:
            validated = validate_context_checkpoint(
                checkpoint,
                captured.checkpoint.snapshot.events,
                generation=captured.checkpoint.token.generation,
                source_revision=captured.checkpoint.token.source_revision,
                parent_checkpoint_id=captured.checkpoint.token.checkpoint_id,
            )
        except (SourceValidationError, KeyError, TypeError, ValueError):
            return False
        if validated != checkpoint:
            return False
        boundary = checkpoint.get("boundary")
        prior_boundary = self._proposal_boundary(captured.checkpoint)
        if type(boundary) is not int or prior_boundary < 0 or boundary <= prior_boundary:
            return False
        proof = proposal.proof
        if type(proof) is not SummaryDependencyProof:
            return False
        if proof.candidate.snapshot != captured.checkpoint.snapshot or proof.candidate.stream_id is not None:
            return False
        if proof.candidate.previous_boundary != prior_boundary:
            return False
        if (
            not proof.chunks
            or proof.chunks[-1].evidence.boundary != boundary
            or proof.chunks[-1].summary != checkpoint.get("summary")
        ):
            return False
        try:
            from nanobot.session.checkpoint_dependencies import derive_summary_references

            if derive_summary_references(proof, captured.checkpoint.snapshot) != tuple(
                checkpoint.get("artifact_references", ())
            ):
                return False
        except Exception:
            return False
        producing_provider = checkpoint.get("producing_provider")
        producing_model = checkpoint.get("producing_model")
        runtime = captured.binding.auxiliary_runtime
        try:
            runtime_fingerprint = _adapter_runtime_fingerprint(captured.summary_adapter)
        except IdleScannerIneligibleError:
            return False
        if (
            runtime_fingerprint != captured.checkpoint.runtime_fingerprint
            or producing_provider != getattr(runtime.provider, "provider_name", None)
            or producing_model != runtime.model
            or proposal.provider != producing_provider
            or proposal.model != producing_model
        ):
            return False
        usage_map = checkpoint.get("auxiliary_usage")
        measured = proposal.measured_usage
        if not isinstance(usage_map, dict) or not isinstance(measured, object):
            return False
        expected_usage = {
            "calls": result.usage.calls,
            "tokens": measured.total_tokens,
            "input_tokens": measured.input_tokens,
            "output_tokens": measured.output_tokens,
            "total_tokens": measured.total_tokens,
        }
        allowed_usage_keys = set(expected_usage)
        if measured.reasoning_tokens is not None:
            allowed_usage_keys.add("reasoning_tokens")
        if set(usage_map) != allowed_usage_keys:
            return False
        if any(usage_map.get(name) != value for name, value in expected_usage.items()):
            return False
        if hasattr(measured, "reasoning_tokens") and measured.reasoning_tokens is not None:
            if usage_map.get("reasoning_tokens") != measured.reasoning_tokens:
                return False
        return True

    async def _validate(
        self,
        key: str,
        captured: object,
        candidate: object,
    ) -> bool:
        if type(captured) is not IdleScannerCapture or type(candidate) is not IdleScannerComputation:
            return False
        item = cast(IdleScannerCapture, captured)
        result = cast(IdleScannerComputation, candidate)
        try:
            limits = self._policy.auxiliary_limits
            if item.root_authority is not self._root_authority:
                return False
            item.root_authority.require_current(self._source._backend)
            if _adapter_binding(item.summary_adapter) is not item.binding:
                return False
            return self._validate_result(item, result, limits)
        except SourceRootAuthorityError:
            self._invalidate_root()
            return False
        except Exception:
            return False

    def _publish_sync(
        self,
        key: str,
        captured: IdleScannerCapture,
        computation: IdleScannerComputation,
        binding: IdleRuntimeBinding,
    ) -> object:
        if captured.binding is not binding or binding.session_key != key:
            raise IdleScannerPublicationConflict("idle binding changed")
        if _adapter_binding(captured.summary_adapter) is not binding:
            raise IdleScannerPublicationConflict("idle adapter binding changed")
        if captured.root_authority is not self._root_authority:
            raise IdleScannerPublicationConflict("idle source-root authority changed")
        try:
            captured.root_authority.require_current(self._source._backend)
        except SourceRootAuthorityError:
            self._invalidate_root()
            raise IdleScannerPublicationConflict("idle source-root pathname changed") from None
        try:
            self._validate_adapter_binding(captured.summary_adapter, binding)
        except IdleScannerIneligible:
            raise IdleScannerPublicationConflict("idle adapter authority changed") from None
        if captured._publication.consumed:
            raise IdleScannerPublicationConflict("idle publication was already consumed")
        if not _elapsed_idle(
            captured.checkpoint.token,
            ttl_seconds=self._policy.idle_ttl_seconds,
            clock=self._utc_clock,
        ):
            raise IdleScannerPublicationConflict("idle TTL is no longer current")
        blocker = self._read_blocker(key)
        if blocker is None or blocker != captured.blocker:
            raise IdleScannerPublicationConflict("runtime blocker state changed")
        if not self._validate_result(captured, computation, self._policy.auxiliary_limits):
            raise IdleScannerPublicationConflict("idle proposal is no longer admissible")
        # Consume the one-shot proposal before entering I2.  Mark completion
        # only after I2 returns normally; a write-then-raise must remain a
        # failed attempt even when the underlying replacement was durable.
        captured._publication.consumed = True
        policy = binding.policy_binding
        if type(policy) is not IdlePolicyBinding:
            raise IdleScannerPublicationConflict("idle policy binding is unavailable")
        proposal = computation.proposal
        result = self._source.publish_idle_checkpoint(
            key,
            captured.checkpoint,
            proposal.checkpoint,
            runtime_fingerprint=captured.checkpoint.runtime_fingerprint,
            policy_fingerprint=policy.fingerprint,
            policy=policy,
            proof=proposal.proof,
            root_authority=captured.root_authority,
        )
        # This boundary is intentionally after the synchronous I2 callback
        # returns and before control reaches the registry await.  It survives
        # a callback that cancels its own task immediately after the write.
        captured._publication.completed = True
        return result

    async def _publish(self, key: str, captured: object, computation: object) -> object:
        if (
            self._registry is None
            or type(captured) is not IdleScannerCapture
            or type(computation) is not IdleScannerComputation
        ):
            raise IdleScannerPublicationConflict("idle publication authority is unavailable")
        item = cast(IdleScannerCapture, captured)
        proposal = cast(IdleScannerComputation, computation)

        def callback(binding: IdleRuntimeBinding) -> object:
            return self._publish_sync(key, item, proposal, binding)

        try:
            return await self._registry.publish_if_current(
                item.binding,
                callback,
                session_key=key,
                root_authority=item.root_authority,
            )
        except (IdleRuntimeStaleError, IdleCheckpointConflict, IdleCheckpointError):
            try:
                if self._root_authority is not None:
                    self._root_authority.require_current(self._source._backend)
            except SourceRootAuthorityError:
                self._invalidate_root()
            raise IdleScannerPublicationConflict("idle publication lost authority") from None
        except asyncio.CancelledError:
            if item._publication.completed:
                # The durable callback returned normally; cancellation was
                # delivered only after the committed completion boundary.
                return None
            raise
        except Exception as exc:
            try:
                if self._root_authority is not None:
                    self._root_authority.require_current(self._source._backend)
            except SourceRootAuthorityError:
                self._invalidate_root()
                raise IdleScannerPublicationConflict("idle source-root pathname changed") from None
            # An execution/artifact lock conflict is content-free and should
            # be discarded without coordinator backoff.  Other errors retain
            # ordinary failure/backoff semantics.
            name = type(exc).__name__
            if name in {"ExecutionBusyError", "ArtifactStorageBusyError"}:
                raise IdleScannerPublicationConflict("idle publication is busy") from None
            raise

    async def scan_once(self) -> int:
        """Coalesce overlapping explicit scans into one bounded operation."""

        if self._closed or not self._enabled or self._discovery is None:
            return 0
        async with self._state_lock:
            if self._closed:
                return 0
            task = self._scan_task
            if task is None or task.done():
                task = asyncio.create_task(self._coordinator.scan_once())
                self._scan_task = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._state_lock:
                    if self._scan_task is task:
                        self._scan_task = None

    async def foreground_admitted(self, session_key: str) -> None:
        """Notify local idle work after the durable foreground activity write."""

        if self._closed:
            return
        if self._registry is not None:
            self._registry.invalidate(session_key=session_key)
        # Invalidate the I3b binding before cancelling local work.  This keeps
        # a publication already waiting at its gate from observing the old
        # authority, while still supporting an externally owned coordinator.
        await self._coordinator.foreground_admitted(session_key)

    async def _close_operation(self) -> None:
        self._closed = True
        task = self._scan_task
        if task is not None and not task.done():
            task.cancel()
        close = getattr(self._coordinator, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        if self._discovery is not None:
            await self._discovery.aclose()
        if self._root_authority is not None:
            self._root_authority.close()

    async def aclose(self) -> None:
        """Close discovery/coordinator once and join cleanup under cancellation."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_operation())
        task = self._close_task
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                continue
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        await task

    async def close(self) -> None:
        await self.aclose()


# Descriptive aliases used by integration callers.
IdleScanner = IdleCompactionScanner
IdleCompactionAdapter = IdleCompactionScanner
IdleRuntimeBlockerSnapshot = IdleBlockerSnapshot
IdleRuntimeBlockerAuthority = IdleBlockerAuthority


__all__ = [
    "IdleBlockerAuthority",
    "IdleBlockerSnapshot",
    "IdleCompactionAdapter",
    "IdleCompactionScanner",
    "IdleScanner",
    "IdleScannerCapture",
    "IdleScannerComputation",
    "IdleScannerError",
    "IdleScannerIneligible",
    "IdleScannerIneligibleError",
    "IdleScannerPublicationConflict",
    "IdleRuntimeBlockerAuthority",
    "IdleRuntimeBlockerSnapshot",
    "InMemoryIdleBlockerAuthority",
]
