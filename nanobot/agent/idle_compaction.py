"""Dormant scheduling substrate for governed idle compaction.

This module intentionally owns no session store, model runtime, summarizer, or
checkpoint publisher.  An eventual runtime integration injects callbacks that
capture a durable idle token, compute a candidate, and validate that candidate
against the current generation/activity state.  The coordinator only bounds
when that work may run and discards work when foreground activity wins.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from time import monotonic
from typing import Generic, TypeVar

CaptureT = TypeVar("CaptureT")
CandidateT = TypeVar("CandidateT")

SessionEnumerator = Callable[[], Awaitable[Iterable[str]]]
IdleCapture = Callable[[str], Awaitable[CaptureT | None]]
IdleCompute = Callable[
    [str, CaptureT, "IdleAuxiliaryLimits"], Awaitable["IdleComputation[CandidateT]"]
]
IdleValidate = Callable[[str, CaptureT, CandidateT], Awaitable[bool]]
IdlePublish = Callable[[str, CaptureT, CandidateT], Awaitable[object] | object]
Clock = Callable[[], float]


class IdleCompactionError(RuntimeError):
    """Base error for a rejected callback result."""


class IdleAuxiliaryBudgetError(IdleCompactionError):
    """The auxiliary candidate computation exceeded its isolated budget."""


class IdleUnknownUsageError(IdleAuxiliaryBudgetError):
    """The candidate computation finished without usage evidence."""


class IdleInvalidUsageError(IdleAuxiliaryBudgetError):
    """The candidate computation returned malformed usage evidence."""


class IdlePublicationConflictError(IdleCompactionError):
    """The candidate lost an authoritative publication race."""


IdlePublicationConflict = IdlePublicationConflictError


@dataclass(frozen=True, slots=True)
class IdleAuxiliaryLimits:
    """Immutable pre-dispatch limits a compute callback must enforce itself."""

    max_calls: int
    max_tokens: int
    max_wall_seconds: float

    def __post_init__(self) -> None:
        _optional_positive_int(self.max_calls, "max_calls")
        _optional_positive_int(self.max_tokens, "max_tokens")
        _optional_positive_number(self.max_wall_seconds, "max_wall_seconds")


@dataclass(frozen=True, slots=True)
class IdleAuxiliaryUsage:
    """Exact auxiliary-call cost reported by a callback-owned computation."""

    calls: int
    tokens: int

    def __post_init__(self) -> None:
        for name, value in (("calls", self.calls), ("tokens", self.tokens)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class IdleComputation(Generic[CandidateT]):
    """An unpersisted candidate and the cost of computing it.

    ``usage=None`` is deliberately a failure.  The coordinator must never
    treat a missing provider receipt as free auxiliary work.
    """

    candidate: CandidateT
    usage: IdleAuxiliaryUsage | None


def _finite_non_negative(value: float | int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _optional_positive_int(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{name} must be a positive integer or None")


def _optional_positive_number(value: float | int | None, name: str) -> None:
    if value is None:
        return
    _finite_non_negative(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive or None")


@dataclass(frozen=True, slots=True)
class IdleCompactionPolicy:
    """Explicit opt-in limits for the dormant coordinator.

    There are intentionally no enabled production defaults.  A policy becomes
    enabled only when its TTL and every queue, concurrency, auxiliary-cost and
    retry limit below are supplied.  A future configuration owner must choose
    actual values and construct this policy; this module supplies no factory.
    """

    idle_ttl_seconds: float = 0.0
    queue_capacity: int | None = None
    max_concurrent_jobs: int | None = None
    max_auxiliary_calls: int | None = None
    max_auxiliary_tokens: int | None = None
    max_auxiliary_wall_seconds: float | None = None
    initial_backoff_seconds: float | None = None
    max_backoff_seconds: float | None = None

    def __post_init__(self) -> None:
        _finite_non_negative(self.idle_ttl_seconds, "idle_ttl_seconds")
        _optional_positive_int(self.queue_capacity, "queue_capacity")
        _optional_positive_int(self.max_concurrent_jobs, "max_concurrent_jobs")
        _optional_positive_int(self.max_auxiliary_calls, "max_auxiliary_calls")
        _optional_positive_int(self.max_auxiliary_tokens, "max_auxiliary_tokens")
        _optional_positive_number(self.max_auxiliary_wall_seconds, "max_auxiliary_wall_seconds")
        _optional_positive_number(self.initial_backoff_seconds, "initial_backoff_seconds")
        _optional_positive_number(self.max_backoff_seconds, "max_backoff_seconds")
        if (
            self.initial_backoff_seconds is not None
            and self.max_backoff_seconds is not None
            and self.initial_backoff_seconds > self.max_backoff_seconds
        ):
            raise ValueError("initial_backoff_seconds cannot exceed max_backoff_seconds")

    @property
    def enabled(self) -> bool:
        return self.idle_ttl_seconds > 0 and all(
            value is not None
            for value in (
                self.queue_capacity,
                self.max_concurrent_jobs,
                self.max_auxiliary_calls,
                self.max_auxiliary_tokens,
                self.max_auxiliary_wall_seconds,
                self.initial_backoff_seconds,
                self.max_backoff_seconds,
            )
        )

    @property
    def auxiliary_limits(self) -> IdleAuxiliaryLimits:
        """Return the concrete limits required before auxiliary dispatch."""
        if not self.enabled:
            raise RuntimeError("Disabled idle policy has no auxiliary limits")
        calls = self.max_auxiliary_calls
        tokens = self.max_auxiliary_tokens
        wall = self.max_auxiliary_wall_seconds
        if calls is None or tokens is None or wall is None:
            raise RuntimeError("Enabled idle policy lacks auxiliary limits")
        return IdleAuxiliaryLimits(calls, tokens, wall)


class IdleJobOutcome(str, Enum):
    """Content-free terminal state for a single session's latest job."""

    INELIGIBLE = "ineligible"
    QUEUE_FULL = "queue_full"
    BACKING_OFF = "backing_off"
    SUCCEEDED = "succeeded"
    DISCARDED = "discarded"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IdleBackoff:
    """Monotonic retry state for one session."""

    failures: int
    retry_at: float


@dataclass(frozen=True, slots=True)
class _QueuedJob(Generic[CaptureT]):
    identifier: int
    session_key: str
    epoch: int
    capture: CaptureT


class _PublicationOutcome(str, Enum):
    """Internal result of the one-shot publication phase."""

    SUCCEEDED = "succeeded"
    DISCARDED = "discarded"
    CANCELLED = "cancelled"
    FAILED = "failed"


class IdleCompactionCoordinator(Generic[CaptureT, CandidateT]):
    """Bounded, production-unwired coordinator for callback-owned idle work.

    ``capture`` is responsible for TTL and all durable eligibility checks; it
    returns ``None`` for an ineligible session.  ``compute`` runs with no
    coordinator lock held and must only produce an in-memory candidate.
    ``validate`` receives the opaque capture token and must revalidate the
    generation/activity token (and any storage CAS) before reporting success.
    It is a validation callback, not a publisher: this coordinator never
    writes a checkpoint or calls a summarizer itself.
    """

    def __init__(
        self,
        *,
        policy: IdleCompactionPolicy,
        enumerate_sessions: SessionEnumerator,
        capture: IdleCapture[CaptureT],
        compute: IdleCompute[CaptureT, CandidateT],
        validate: IdleValidate[CaptureT, CandidateT],
        publish: IdlePublish[CaptureT, CandidateT] | None = None,
        publication: IdlePublish[CaptureT, CandidateT] | None = None,
        clock: Clock = monotonic,
    ) -> None:
        self._policy = policy
        self._enumerate_sessions = enumerate_sessions
        self._capture = capture
        self._compute = compute
        self._validate = validate
        if publish is not None and publication is not None and publish is not publication:
            raise ValueError("publish and publication must refer to the same callback")
        self._publish = publish if publish is not None else publication
        self._clock = clock
        # A disabled policy must never create worker tasks or enqueue an item.
        # The inert queue therefore needs no invented production capacity.
        capacity = policy.queue_capacity if policy.enabled else 0
        self._queue: asyncio.Queue[_QueuedJob[CaptureT]] = asyncio.Queue(maxsize=capacity)
        self._control_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._shutdown_task: asyncio.Task[None] | None = None
        self._next_identifier = 1
        self._epochs: dict[str, int] = {}
        self._current: dict[str, int] = {}
        self._backoff: dict[str, IdleBackoff] = {}
        self._outcomes: dict[str, IdleJobOutcome] = {}
        self._workers: set[asyncio.Task[None]] = set()
        self._scans: set[asyncio.Task[int]] = set()
        self._enumerations: set[asyncio.Task[Iterable[str]]] = set()
        self._capturing: dict[str, tuple[int, asyncio.Task[CaptureT | None]]] = {}
        self._active: dict[str, tuple[int, asyncio.Task[None]]] = {}
        self._computing: dict[
            str, tuple[int, asyncio.Task[IdleComputation[CandidateT]]]
        ] = {}
        self._cleanup: dict[str, tuple[int, asyncio.Task[None]]] = {}
        self._publishing: dict[str, tuple[int, asyncio.Task[object]]] = {}

    @property
    def worker_count(self) -> int:
        """Number of live worker tasks, for integration observability."""
        return sum(not task.done() for task in self._workers)

    @property
    def queue_depth(self) -> int:
        """Number of physical queue entries awaiting a worker."""
        return self._queue.qsize()

    @property
    def admission_count(self) -> int:
        """Return reservations across capture, queue, and active work."""
        return len(self._current)

    @property
    def pending_callback_count(self) -> int:
        """Return live callback/task ownership for deterministic shutdown tests."""
        tasks: list[asyncio.Task[object]] = []
        tasks.extend(self._scans)
        tasks.extend(self._enumerations)
        tasks.extend(task for _, task in self._capturing.values())
        tasks.extend(task for _, task in self._active.values())
        tasks.extend(task for _, task in self._computing.values())
        tasks.extend(task for _, task in self._cleanup.values())
        tasks.extend(task for _, task in self._publishing.values())
        return sum(not task.done() for task in tasks)

    def outcome_for(self, session_key: str) -> IdleJobOutcome | None:
        """Return the latest content-free outcome for ``session_key``."""
        return self._outcomes.get(session_key)

    def backoff_for(self, session_key: str) -> IdleBackoff | None:
        """Return the current monotonic retry state for ``session_key``."""
        return self._backoff.get(session_key)

    async def scan_once(self) -> int:
        """Capture and queue currently eligible sessions once.

        An integration-owned scanner decides when to invoke this method.  No
        session enumeration, capture, task creation, or callback is attempted
        unless every finite limit is explicitly present in an enabled policy.
        """
        if not self._policy.enabled or self._closed:
            return 0
        scan = asyncio.create_task(self._scan_once())
        async with self._control_lock:
            if self._closed:
                scan.cancel()
            else:
                self._scans.add(scan)
                scan.add_done_callback(self._scans.discard)
        try:
            return await scan
        except asyncio.CancelledError:
            # Close cancels the coordinator-owned scan task.  Do not turn a
            # routine stop into a cancellation of the integration scanner.
            if self._closed or not asyncio.current_task().cancelling():
                return 0
            raise

    async def _scan_once(self) -> int:
        await self._ensure_workers()
        if self._closed:
            return 0
        enumeration = asyncio.create_task(self._enumerate_sessions())
        async with self._control_lock:
            if self._closed:
                enumeration.cancel()
            else:
                self._enumerations.add(enumeration)
                enumeration.add_done_callback(self._enumerations.discard)
        try:
            session_keys = await enumeration
        except asyncio.CancelledError:
            if self._closed or not asyncio.current_task().cancelling():
                return 0
            raise
        admitted = 0
        for session_key in session_keys:
            if not isinstance(session_key, str) or not session_key:
                continue
            if await self._capture_and_enqueue(session_key):
                admitted += 1
        return admitted

    async def foreground_admitted(self, session_key: str) -> None:
        """Invalidate local idle work after an already-admitted foreground turn.

        This method intentionally does not write user-activity metadata.  The
        foreground session source owner performs that durable update before it
        calls this notification.  Here we only cancel/discard local work.
        """
        if not isinstance(session_key, str) or not session_key:
            return
        local_task: (
            asyncio.Task[None]
            | asyncio.Task[CaptureT | None]
            | asyncio.Task[IdleComputation[CandidateT]]
            | None
        ) = None
        async with self._control_lock:
            self._epochs[session_key] = self._epochs.get(session_key, 0) + 1
            identifier = self._current.get(session_key)
            if identifier is not None:
                self._outcomes[session_key] = IdleJobOutcome.CANCELLED
                active_entry = self._active.get(session_key)
                if active_entry is not None and active_entry[0] == identifier:
                    local_task = active_entry[1]
                capture_entry = self._capturing.get(session_key)
                if capture_entry is not None and capture_entry[0] == identifier:
                    local_task = capture_entry[1]
                compute_entry = self._computing.get(session_key)
                if compute_entry is not None and compute_entry[0] == identifier:
                    local_task = compute_entry[1]
                publish_entry = self._publishing.get(session_key)
                if publish_entry is not None and publish_entry[0] == identifier:
                    local_task = publish_entry[1]
        self._request_cancel(local_task)

    async def drain(self) -> None:
        """Wait until every queue entry admitted before the call has settled."""
        await self._queue.join()

    async def aclose(self) -> None:
        """Stop admission and await one shared, cancellation-safe shutdown."""
        async with self._close_lock:
            shutdown = self._shutdown_task
            if shutdown is None:
                # Set this before scheduling the shared task so no concurrent
                # scan can create another callback after close begins.
                self._closed = True
                shutdown = asyncio.create_task(self._shutdown())
                self._shutdown_task = shutdown
        await asyncio.shield(shutdown)

    async def _shutdown(self) -> None:
        """Cancel callback owners, await cleanup, then retire worker tasks."""
        async with self._control_lock:
            self._closed = True
            queued = self._clear_queue_unlocked()
            for job in queued:
                self._finish_cancelled_unlocked(job.session_key, job.identifier)
            scan_tasks = tuple(self._scans)
            enumeration_tasks = tuple(self._enumerations)
            capture_tasks = tuple(task for _, task in self._capturing.values())
            active_tasks = tuple(task for _, task in self._active.values())
            compute_tasks = tuple(task for _, task in self._computing.values())
            cleanup_tasks = tuple(task for _, task in self._cleanup.values())
            publishing_tasks = tuple(task for _, task in self._publishing.values())
            worker_tasks = tuple(self._workers)
        # A worker keeps serving while its nested job unwinds.  Cancelling it
        # first can detach the wait_for child and let close return too early.
        for task in (
            *scan_tasks,
            *enumeration_tasks,
            *capture_tasks,
            *active_tasks,
            *compute_tasks,
            *publishing_tasks,
        ):
            self._request_cancel(task)
        owned_tasks = (
            *scan_tasks,
            *enumeration_tasks,
            *capture_tasks,
            *active_tasks,
            *compute_tasks,
            *cleanup_tasks,
            *publishing_tasks,
        )
        if owned_tasks:
            await asyncio.gather(*owned_tasks, return_exceptions=True)
        for task in worker_tasks:
            self._request_cancel(task)
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        self._scans.clear()
        self._enumerations.clear()
        self._capturing.clear()
        self._active.clear()
        self._computing.clear()
        self._cleanup.clear()
        self._publishing.clear()
        self._workers.clear()

    async def _ensure_workers(self) -> None:
        if not self._policy.enabled:
            return
        async with self._control_lock:
            if self._closed:
                return
            # ``enabled`` proved these fields are concrete, while retaining the
            # runtime assertion protects future policy changes.
            workers = self._policy.max_concurrent_jobs
            if workers is None:
                raise RuntimeError("Enabled idle policy lacks worker concurrency")
            missing = max(0, workers - sum(not task.done() for task in self._workers))
            for _ in range(missing):
                task = asyncio.create_task(self._worker())
                self._workers.add(task)
                task.add_done_callback(self._workers.discard)

    async def _capture_and_enqueue(self, session_key: str) -> bool:
        """Reserve one session, then capture it without a coordinator lock."""
        async with self._control_lock:
            if self._closed or not self._policy.enabled:
                return False
            now = self._read_clock()
            retry = self._backoff.get(session_key)
            if retry is not None and now < retry.retry_at:
                self._outcomes[session_key] = IdleJobOutcome.BACKING_OFF
                return False
            if session_key in self._current:
                return False
            if len(self._current) >= self._admission_capacity():
                self._outcomes[session_key] = IdleJobOutcome.QUEUE_FULL
                return False
            identifier = self._next_identifier
            self._next_identifier += 1
            epoch = self._epochs.get(session_key, 0)
            self._current[session_key] = identifier
            capture_task = asyncio.create_task(self._capture(session_key))
            self._capturing[session_key] = (identifier, capture_task)
        try:
            capture = await capture_task
        except asyncio.CancelledError:
            await self._finish_cancelled(session_key, identifier)
            # Foreground admission and close cancel the callback task, while a
            # caller cancelling scan_once cancels this outer task as well.
            # Only the latter must propagate to that caller.
            if self._closed or not asyncio.current_task().cancelling():
                return False
            raise
        except Exception:
            await self._finish_failure(session_key, identifier, IdleJobOutcome.FAILED)
            return False
        finally:
            async with self._control_lock:
                active_capture = self._capturing.get(session_key)
                if active_capture is not None and active_capture[0] == identifier:
                    self._capturing.pop(session_key, None)
        if capture is None:
            await self._finish_ineligible(session_key, identifier)
            return False
        async with self._control_lock:
            if not self._is_current_unlocked(session_key, identifier, epoch):
                self._finish_cancelled_unlocked(session_key, identifier)
                return False
            job = _QueuedJob(identifier, session_key, epoch, capture)
        try:
            # Reservation already bounded the total of running, queued, and
            # capturing jobs.  Waiting here lets a worker claim an available
            # global slot instead of rejecting a simultaneous capture merely
            # because the physical queue has not had a scheduling turn yet.
            await self._queue.put(job)
        except asyncio.CancelledError:
            await self._finish_cancelled(session_key, identifier)
            if self._closed or not asyncio.current_task().cancelling():
                return False
            raise
        return True

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                job_task: asyncio.Task[None] | None = None
                async with self._control_lock:
                    if self._is_current_unlocked(job.session_key, job.identifier, job.epoch):
                        job_task = asyncio.create_task(self._run_job(job))
                        self._active[job.session_key] = (job.identifier, job_task)
                    else:
                        self._finish_cancelled_unlocked(job.session_key, job.identifier)
                if job_task is not None:
                    try:
                        await job_task
                    except asyncio.CancelledError:
                        # A foreground notification cancels the nested job,
                        # not its shared worker.  Consume that expected child
                        # cancellation so the worker can drain other sessions.
                        # A cancellation of the worker itself (shutdown) still
                        # propagates after the child has been cleaned up.
                        if asyncio.current_task().cancelling():
                            if not job_task.done():
                                job_task.cancel()
                            await asyncio.gather(job_task, return_exceptions=True)
                            await self._finish_cancelled(job.session_key, job.identifier)
                            raise
                    finally:
                        async with self._control_lock:
                            active = self._active.get(job.session_key)
                            if active is not None and active[0] == job.identifier:
                                self._active.pop(job.session_key, None)
            except asyncio.CancelledError:
                await self._finish_cancelled(job.session_key, job.identifier)
                raise
            finally:
                self._queue.task_done()

    async def _run_job(self, job: _QueuedJob[CaptureT]) -> None:
        """Compute and validate one candidate without holding coordinator locks."""
        started = self._read_clock()
        limits = self._policy.auxiliary_limits
        compute_task = asyncio.create_task(
            self._compute(job.session_key, job.capture, limits)
        )
        async with self._control_lock:
            if not self._is_current_unlocked(job.session_key, job.identifier, job.epoch):
                self._request_cancel(compute_task)
            else:
                self._computing[job.session_key] = (job.identifier, compute_task)
        # The first terminal cause wins.  In particular, once a timeout is
        # observed, a later close waits for its cleanup but does not rewrite
        # the accountable budget failure into a generic cancellation.
        terminal = IdleJobOutcome.CANCELLED
        try:
            # Shield leaves cancellation ownership with this coordinator.  On
            # timeout, foreground activity, and shutdown we explicitly cancel
            # and await the callback task before the job may settle.
            result = await asyncio.wait_for(
                asyncio.shield(compute_task), timeout=limits.max_wall_seconds
            )
            elapsed = self._read_clock() - started
            if not math.isfinite(elapsed) or elapsed < 0 or elapsed > limits.max_wall_seconds:
                raise IdleAuxiliaryBudgetError("Idle auxiliary wall-time ceiling exceeded")
            if type(result) is not IdleComputation:
                raise IdleCompactionError("Idle compute callback returned an invalid result")
            self._validate_usage(result.usage, limits)
            if not await self._is_current(job):
                terminal = IdleJobOutcome.CANCELLED
            else:
                # The validator owns the opaque generation/activity comparison
                # and storage CAS.  It remains outside coordinator locks too.
                valid = await self._validate(job.session_key, job.capture, result.candidate)
                if not isinstance(valid, bool):
                    raise IdleCompactionError("Idle validate callback returned a non-boolean result")
                if not await self._is_current(job):
                    terminal = IdleJobOutcome.CANCELLED
                elif valid:
                    if self._publish is None:
                        # Preserve the old generic-caller behavior: a
                        # successful pure validation is the terminal success
                        # when no explicit publication phase was supplied.
                        terminal = IdleJobOutcome.SUCCEEDED
                    elif not await self._is_current(job):
                        terminal = IdleJobOutcome.CANCELLED
                    else:
                        publication = await self._publish_once(job, result.candidate)
                        terminal = {
                            _PublicationOutcome.SUCCEEDED: IdleJobOutcome.SUCCEEDED,
                            _PublicationOutcome.DISCARDED: IdleJobOutcome.DISCARDED,
                            _PublicationOutcome.CANCELLED: IdleJobOutcome.CANCELLED,
                            _PublicationOutcome.FAILED: IdleJobOutcome.FAILED,
                        }[publication]
                else:
                    terminal = IdleJobOutcome.DISCARDED
        except asyncio.TimeoutError:
            terminal = IdleJobOutcome.BUDGET_EXCEEDED
        except IdleAuxiliaryBudgetError:
            terminal = IdleJobOutcome.BUDGET_EXCEEDED
        except asyncio.CancelledError:
            terminal = IdleJobOutcome.CANCELLED
        except Exception:
            terminal = IdleJobOutcome.FAILED
        finally:
            cleanup = self._begin_compute_cleanup(job, compute_task)
            await self._await_cleanup_resilient(cleanup)
            await self._settle_terminal_resilient(job, terminal)
            await self._forget_job_tasks_resilient(job)

    def _validate_usage(self, usage: IdleAuxiliaryUsage | None, limits: IdleAuxiliaryLimits) -> None:
        if usage is None:
            raise IdleUnknownUsageError("Idle auxiliary usage is unknown")
        if type(usage) is not IdleAuxiliaryUsage:
            raise IdleInvalidUsageError("Idle auxiliary usage has an invalid type")
        try:
            calls, tokens = usage.calls, usage.tokens
        except AttributeError:
            raise IdleInvalidUsageError("Idle auxiliary usage has missing counters") from None
        if (
            type(calls) is not int
            or calls < 0
            or type(tokens) is not int
            or tokens < 0
        ):
            raise IdleInvalidUsageError("Idle auxiliary usage has invalid counters")
        if calls > limits.max_calls or tokens > limits.max_tokens:
            raise IdleAuxiliaryBudgetError("Idle auxiliary call or token ceiling exceeded")

    def _begin_compute_cleanup(
        self, job: _QueuedJob[CaptureT], compute_task: asyncio.Task[IdleComputation[CandidateT]]
    ) -> asyncio.Task[None]:
        """Create exactly one uncancelled waiter for a compute callback."""
        existing = self._cleanup.get(job.session_key)
        if existing is not None and existing[0] == job.identifier:
            return existing[1]
        self._request_cancel(compute_task)
        cleanup = asyncio.create_task(self._wait_for_compute_cleanup(compute_task))
        self._cleanup[job.session_key] = (job.identifier, cleanup)
        return cleanup

    @staticmethod
    async def _wait_for_compute_cleanup(
        task: asyncio.Task[IdleComputation[CandidateT]],
    ) -> None:
        """Wait for the callback's cooperative cancellation handler to finish."""
        await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    async def _await_cleanup_resilient(cleanup: asyncio.Task[None]) -> None:
        """Await cleanup even if its owner receives cancellation again."""
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        await asyncio.gather(cleanup, return_exceptions=True)

    async def _settle_terminal_resilient(
        self, job: _QueuedJob[CaptureT], terminal: IdleJobOutcome
    ) -> None:
        """Persist the selected terminal outcome despite owner cancellation."""
        while True:
            try:
                if terminal is IdleJobOutcome.BUDGET_EXCEEDED:
                    await self._finish_failure(job.session_key, job.identifier, terminal)
                elif terminal is IdleJobOutcome.FAILED:
                    await self._finish_failure(job.session_key, job.identifier, terminal)
                elif terminal is IdleJobOutcome.SUCCEEDED:
                    await self._finish_success(job.session_key, job.identifier)
                elif terminal is IdleJobOutcome.DISCARDED:
                    await self._finish_discarded(job.session_key, job.identifier)
                else:
                    await self._finish_cancelled(job.session_key, job.identifier)
                return
            except asyncio.CancelledError:
                continue

    async def _publish_once(
        self, job: _QueuedJob[CaptureT], candidate: CandidateT
    ) -> _PublicationOutcome:
        """Run the explicit publisher once and join it across cancellation.

        The publisher is deliberately a separate task.  A synchronous source
        write inside a binding gate cannot be interrupted by asyncio; when a
        caller is cancelled immediately after that write, the completed task
        remains a committed success.  A publisher exception consumes this
        attempt as well and is never retried by this coordinator.
        """

        async def invoke() -> object:
            result = self._publish(job.session_key, job.capture, candidate)  # type: ignore[misc]
            if inspect.isawaitable(result):
                return await result
            return result

        publication_task = asyncio.create_task(invoke())
        async with self._control_lock:
            if self._is_current_unlocked(job.session_key, job.identifier, job.epoch):
                self._publishing[job.session_key] = (job.identifier, publication_task)
            else:
                self._request_cancel(publication_task)
        try:
            try:
                result = await asyncio.shield(publication_task)
            except asyncio.CancelledError:
                # Cancellation before a gate is acquired must leave zero
                # writes.  Cancellation after a synchronous callback returns
                # observes its result and preserves the durable success.
                if not publication_task.done():
                    self._request_cancel(publication_task)
                await self._await_publication_cleanup(publication_task)
                if publication_task.cancelled():
                    if self._publication_completed(job.capture):
                        return _PublicationOutcome.SUCCEEDED
                    return _PublicationOutcome.CANCELLED
                try:
                    result = publication_task.result()
                except IdlePublicationConflict:
                    return _PublicationOutcome.DISCARDED
                except BaseException:
                    return _PublicationOutcome.FAILED
            except IdlePublicationConflict:
                return _PublicationOutcome.DISCARDED
            except asyncio.CancelledError:
                return _PublicationOutcome.CANCELLED
            except BaseException:
                return _PublicationOutcome.FAILED
            # ``False`` is the one explicit content-free conflict result;
            # source snapshots and other successful receipts are accepted.
            return (
                _PublicationOutcome.DISCARDED
                if result is False
                else _PublicationOutcome.SUCCEEDED
            )
        finally:
            async with self._control_lock:
                current = self._publishing.get(job.session_key)
                if current is not None and current[0] == job.identifier:
                    self._publishing.pop(job.session_key, None)

    @staticmethod
    async def _await_publication_cleanup(task: asyncio.Task[object]) -> None:
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
            pass

    @staticmethod
    def _publication_completed(capture: object) -> bool:
        """Read an optional content-free completion boundary from an adapter.

        Generic publishers do not need this hook and retain their existing
        cancellation behavior.  The dormant scanner supplies only a sealed
        boolean boundary set after its synchronous I2 callback returns.
        """
        state = getattr(capture, "_publication", None)
        return getattr(state, "completed", False) is True

    async def _forget_job_tasks_resilient(self, job: _QueuedJob[CaptureT]) -> None:
        """Release callback ownership only after terminal settlement."""
        while True:
            try:
                async with self._control_lock:
                    computing = self._computing.get(job.session_key)
                    if computing is not None and computing[0] == job.identifier:
                        self._computing.pop(job.session_key, None)
                    cleanup = self._cleanup.get(job.session_key)
                    if cleanup is not None and cleanup[0] == job.identifier:
                        self._cleanup.pop(job.session_key, None)
                return
            except asyncio.CancelledError:
                continue

    @staticmethod
    def _request_cancel(task: asyncio.Task[object] | None) -> None:
        """Request cancellation once, preserving callback cleanup handlers."""
        if task is not None and not task.done() and not task.cancelling():
            task.cancel()

    def _admission_capacity(self) -> int:
        workers = self._policy.max_concurrent_jobs
        queue = self._policy.queue_capacity
        if workers is None or queue is None:
            raise RuntimeError("Enabled idle policy lacks admission capacity")
        return workers + queue

    async def _is_current(self, job: _QueuedJob[CaptureT]) -> bool:
        async with self._control_lock:
            return self._is_current_unlocked(job.session_key, job.identifier, job.epoch)

    def _is_current_unlocked(self, session_key: str, identifier: int, epoch: int) -> bool:
        return (
            not self._closed
            and self._current.get(session_key) == identifier
            and self._epochs.get(session_key, 0) == epoch
        )

    async def _finish_ineligible(self, session_key: str, identifier: int) -> None:
        async with self._control_lock:
            if self._current.get(session_key) == identifier:
                self._current.pop(session_key, None)
                self._outcomes[session_key] = IdleJobOutcome.INELIGIBLE

    async def _finish_success(self, session_key: str, identifier: int) -> None:
        async with self._control_lock:
            if self._current.get(session_key) == identifier:
                self._current.pop(session_key, None)
                self._backoff.pop(session_key, None)
                self._outcomes[session_key] = IdleJobOutcome.SUCCEEDED

    async def _finish_discarded(self, session_key: str, identifier: int) -> None:
        async with self._control_lock:
            if self._current.get(session_key) == identifier:
                self._current.pop(session_key, None)
                self._outcomes[session_key] = IdleJobOutcome.DISCARDED

    async def _finish_cancelled(self, session_key: str, identifier: int) -> None:
        async with self._control_lock:
            self._finish_cancelled_unlocked(session_key, identifier)

    def _finish_cancelled_unlocked(self, session_key: str, identifier: int) -> None:
        if self._current.get(session_key) == identifier:
            self._current.pop(session_key, None)
            self._outcomes[session_key] = IdleJobOutcome.CANCELLED

    async def _finish_failure(
        self, session_key: str, identifier: int, outcome: IdleJobOutcome
    ) -> None:
        async with self._control_lock:
            if self._current.get(session_key) != identifier:
                return
            self._current.pop(session_key, None)
            prior = self._backoff.get(session_key)
            failures = 1 if prior is None else prior.failures + 1
            retry_at = self._read_clock() + self._backoff_delay(failures)
            self._backoff[session_key] = IdleBackoff(failures=failures, retry_at=retry_at)
            self._outcomes[session_key] = outcome

    def _backoff_delay(self, failures: int) -> float:
        initial = self._policy.initial_backoff_seconds
        maximum = self._policy.max_backoff_seconds
        if initial is None or maximum is None:
            raise RuntimeError("Enabled idle policy lacks retry limits")
        delay = float(initial)
        for _ in range(failures - 1):
            delay = min(float(maximum), delay * 2)
        return delay

    def _read_clock(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise IdleCompactionError("Idle coordinator clock returned an invalid value")
        return float(value)

    def _clear_queue_unlocked(self) -> list[_QueuedJob[CaptureT]]:
        queued: list[_QueuedJob[CaptureT]] = []
        while True:
            try:
                queued.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return queued
            else:
                self._queue.task_done()
