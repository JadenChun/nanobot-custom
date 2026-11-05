"""Explicitly injected dormant cron graph sharing foreground execution authorities."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, NoReturn, SupportsIndex, cast

from loguru import logger

from nanobot.cron.canonical_execution import SilentCronAgent
from nanobot.cron.channel_sender import CronChannelSender, ExactChannelAuthority
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.composition import ClaimedTextCompositionResult, compose_claimed_text_occurrence
from nanobot.cron.definition_store import CronDefinition
from nanobot.cron.delivery_runtime import CronDeliveryRuntime
from nanobot.cron.delivery_store import CronDeliveryOutboxStore
from nanobot.cron.interruption_recovery import recover_interrupted_cron_sources
from nanobot.cron.job_hooks import CronJobHooks
from nanobot.cron.occurrence import CronOccurrenceRuntime, destination_policy
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceClaim,
    TriggeredOccurrenceIdentity,
)
from nanobot.cron.quota_recovery import CronQuotaRecovery
from nanobot.cron.schedule import ScheduleClock
from nanobot.cron.scheduler import Scheduler
from nanobot.cron.scheduler_store import SchedulerConflictError, SchedulerStore, TriggerInputs
from nanobot.cron.service import CronService
from nanobot.runtime_errors import RuntimeOwnedWorkError
from nanobot.session.manager import SessionManager

_TRIGGER_TOKEN = object()


class CronWorkFailedError(RuntimeError):
    """Admitted effects joined with an explicit failure; no replay is authorized."""


@dataclass(frozen=True)
class CronRuntimeInjection:
    """Offline-only explicit gate; no config or foreground caller enables it."""

    clock: ScheduleClock
    manual_authorizer: Callable[[str, str, str], bool]
    drain_timeout: float = 5.0


class SystemScheduleClock:
    def wall_time_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def monotonic_time(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@dataclass(frozen=True)
class GatewayCronOptions:
    """Production inputs supplied only by the normal gateway owner."""

    clock: ScheduleClock
    drain_timeout: float = 5.0
    poll_interval: float = 1.0

    @staticmethod
    def manual_authorizer(_job_id: str, _instance: str, _revision: str) -> bool:
        return False


class TriggerCapability:
    def __init__(self, token: object, runtime: CronRuntime, source: Literal["manual", "system"],
                 controller: str | None, request_id: str, definition: CronDefinition, clock_ms: int):
        if token is not _TRIGGER_TOKEN:
            raise TypeError("trigger capability requires explicit authorization")
        self.source: Literal["manual", "system"]
        self.runtime, self.source, self.controller = runtime, source, controller
        self.request_id, self.definition, self.clock_ms = request_id, definition, clock_ms
        self.pid = os.getpid()

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        raise TypeError("trigger capability cannot be serialized")


class InternalController:
    def __init__(self, token: object, runtime: CronRuntime, name: str):
        if token is not _TRIGGER_TOKEN:
            raise TypeError("controller requires explicit construction")
        self._runtime, self._name = runtime, name

    def request(self, request_id: str, job_id: str, instance: str, revision: str) -> TriggerCapability:
        return self._runtime.request_from_controller(self, self._name, request_id, job_id, instance, revision)


class CronRuntime:
    def __init__(self, service: CronService, agent: SilentCronAgent, sessions: SessionManager,
                 channels: ExactChannelAuthority, root: Path, runtime_identity: str,
                 injection: CronRuntimeInjection | GatewayCronOptions, *, job_hooks: CronJobHooks | None = None,
                 quota_recovery: CronQuotaRecovery | None = None):
        if service.occurrence_store is not None or service.on_job is not None:
            raise ValueError("dormant graph requires definition-only service without callbacks")
        self.service, self.agent, self.sessions, self.channels = service, agent, sessions, channels
        self.injection = injection
        self.job_hooks = job_hooks
        self.quota_recovery = quota_recovery
        self._quota_controller: InternalController | None = None
        self._job_locks: dict[str, asyncio.Lock] = {}
        if injection.drain_timeout <= 0:
            raise ValueError("drain timeout must be positive")
        namespace = service.action_store.namespace
        occurrences = CronOccurrenceStore(root / "occurrences", namespace=namespace)
        projector = CronCompletionProjector(occurrences, root / "runs")
        outbox = CronDeliveryOutboxStore(root / "outbox", namespace=namespace)
        delivery = CronDeliveryRuntime(outbox, occurrence_store=occurrences)
        service.occurrence_store = occurrences
        service.occurrence_runtime = CronOccurrenceRuntime(service.definition_store, occurrences)
        service.completion_projector = projector
        service.delivery_store, service.delivery_runtime = outbox, delivery
        self.sender = CronChannelSender(channels)
        self.store = SchedulerStore(root / "scheduler", namespace=namespace, runtime_identity=runtime_identity)
        self.scheduler = Scheduler(service.definition_store, service.occurrence_runtime,
                                   self.store, injection.clock, self._dispatch,
                                   refresh_committed_revisions=isinstance(injection, GatewayCronOptions))
        self._accepting = False
        self._tasks: set[asyncio.Task[ClaimedTextCompositionResult]] = set()
        self._task_errors: list[BaseException] = []
        self._capabilities: dict[int, tuple[TriggerCapability, tuple[CronRuntime, str, str | None, str, dict[str, object], int, int]]] = {}
        self._controllers: dict[int, tuple[InternalController, str, int]] = {}
        self._recovery_fences: dict[str, str] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._startup_task: asyncio.Task[object] | None = None
        self._timer_task: asyncio.Task[None] | None = None
        self._shutdown_mode: Literal["bounded", "graceful", "forced"] = "bounded"
        self._shutdown_changed = asyncio.Event()

    def _require_acquisition_allowed(self) -> None:
        if self._close_task is not None:
            raise SchedulerConflictError("closed cron runtime cannot reacquire")
        if self._shutdown_mode != "bounded" or self._shutdown_changed.is_set():
            raise SchedulerConflictError("stopping cron runtime cannot acquire")

    def _acquire_scheduler(self) -> None:
        self._require_acquisition_allowed()
        self.scheduler.acquire_and_reconcile()

    def _activate_after_recovery(self) -> None:
        if self.quota_recovery is not None:
            self.quota_recovery.recover(self.service)
        self._accepting = True
        if self.quota_recovery is not None:
            self._quota_controller = self.construct_internal_controller("quota-retry")

    def acquire(self) -> None:
        """Synchronous acquisition retained for existing offline callers."""
        self._acquire_scheduler()
        try:
            self._activate_after_recovery()
        except BaseException:
            self.stop_claiming()
            self.scheduler.release_lease()
            raise

    async def acquire_for_gateway(self) -> bool:
        """Acquire, reconcile interruption evidence, then activate gateway cron."""
        if not isinstance(self.injection, GatewayCronOptions):
            raise SchedulerConflictError("gateway acquisition requires production cron options")
        task = asyncio.current_task()
        if task is None:
            raise SchedulerConflictError("gateway acquisition requires an asyncio task")
        self._startup_task = cast(asyncio.Task[object], task)
        try:
            self._acquire_scheduler()
            try:
                occurrences = cast(CronOccurrenceStore, self.service.occurrence_store)
                self._recovery_fences = await recover_interrupted_cron_sources(
                    self.sessions,
                    self.service,
                    occurrences,
                )
                for job_id, reason in sorted(self._recovery_fences.items()):
                    logger.warning("Cron job {} fenced during interruption recovery: {}", job_id, reason)
                if (
                    self._close_task is not None
                    or self._shutdown_mode != "bounded"
                    or self._shutdown_changed.is_set()
                ):
                    self.stop_claiming()
                    self.scheduler.release_lease()
                    return False
                self._activate_after_recovery()
                return True
            except BaseException:
                self.stop_claiming()
                self.scheduler.release_lease()
                raise
        finally:
            if self._startup_task is task:
                self._startup_task = None

    def start_timer_task(self) -> None:
        """Separate explicit offline injection; acquisition itself starts no task."""
        if not self._accepting or self._timer_task is not None:
            raise SchedulerConflictError("timer task cannot start")
        self._timer_task = asyncio.create_task(self._run_timer())

    async def _run_timer(self) -> None:
        while self._accepting:
            await self.run_due_quota_retries()
            await self.scheduler.wake_once()
            interval = self.injection.poll_interval if isinstance(self.injection, GatewayCronOptions) else 0.05
            await self.injection.clock.sleep(interval)

    async def run_timer(self) -> None:
        """Run under the gateway's supervision; exceptions stop that owner."""
        self.start_timer_task()
        assert self._timer_task is not None
        await asyncio.shield(self._timer_task)

    def status(self) -> dict[str, object]:
        running = self._accepting and self._timer_task is not None and not self._timer_task.done()
        pending = self.quota_recovery.pending() if self.quota_recovery is not None else []
        return {"enabled": running, "state": "running" if running else "inactive",
                "jobs": len(self.service.list_jobs(include_disabled=True)),
                "active_executions": len(self._tasks),
                "quota_retries": [item.model_dump() for item in pending],
                "recovery_required": [
                    {"job_id": job_id, "reason": reason}
                    for job_id, reason in sorted(self._recovery_fences.items())
                ]}

    async def run_due_quota_retries(self) -> None:
        if self.quota_recovery is None or self._quota_controller is None:
            return
        for pending in self.quota_recovery.pending():
            if pending.state != "pending" or pending.retry_at_ms > self.injection.clock.wall_time_ms():
                continue
            try:
                capability = self._quota_controller.request("quota:" + pending.origin,
                    pending.job_id, pending.instance, pending.revision)
            except SchedulerConflictError:
                self.quota_recovery.mark(pending, "unknown")
                continue
            self.quota_recovery.mark(pending, "dispatching")
            await self.execute_now(capability)

    async def _dispatch(self, claim: OccurrenceClaim, pre_begin: Callable[[], object]) -> object:
        if not self._accepting:
            raise SchedulerConflictError("cron ingress fenced")
        if claim.identity.job_id in self._recovery_fences:
            occurrence_store = cast(CronOccurrenceStore, self.service.occurrence_store)
            with occurrence_store.acquire_execution_fence(claim) as fence:
                occurrence_store.begin_execution(claim, fence)
                return self.service.complete_occurrence(
                    claim,
                    {
                        "status": "skipped",
                        "reason": "execution_recovery_required",
                        "delivery": {"version": 1, "deliver": False, "messages": []},
                    },
                    run_id=claim.run_id,
                    fence=fence,
                )
        async def compose() -> ClaimedTextCompositionResult:
            async def admitted() -> ClaimedTextCompositionResult:
                return await compose_claimed_text_occurrence(
                    self.service, self.agent, claim, self.sender, visible_sessions=self.sessions,
                    pre_begin=pre_begin, job_hooks=self.job_hooks, quota_recovery=self.quota_recovery)

            if self.quota_recovery is None:
                return await admitted()
            lock = self._job_locks.setdefault(claim.identity.job_id, asyncio.Lock())
            async with lock:
                return await admitted()

        task = asyncio.create_task(compose())
        self._tasks.add(task)
        task.add_done_callback(self._completed_task)
        return await asyncio.shield(task)

    def _completed_task(self, task: asyncio.Task[ClaimedTextCompositionResult]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self._task_errors.append(cast(BaseException, task.exception()))

    def _definition(self, job_id: str, instance: str, revision: str) -> CronDefinition:
        if not self._accepting or self.service.occurrence_runtime is None:
            raise SchedulerConflictError("cron ingress fenced")
        self.scheduler.require_lifetime_lease()
        definition = self.service.occurrence_runtime.committed_definition(job_id)
        if not definition.definition["enabled"] or (instance, revision) != (definition.job_instance_id, definition.definition_revision):
            raise SchedulerConflictError("trigger definition token mismatch")
        return definition

    def _issue(self, source: Literal["manual", "system"], controller: str | None,
               request_id: str, job_id: str, instance: str, revision: str) -> TriggerCapability:
        if not isinstance(cast(object, request_id), str) or not request_id or len(request_id.encode()) > 512:
            raise ValueError("request identity invalid")
        definition = self._definition(job_id, instance, revision)
        capability = TriggerCapability(_TRIGGER_TOKEN, self, source, controller, request_id,
                                       definition, self.injection.clock.wall_time_ms())
        self._capabilities[id(capability)] = (capability, self._capability_binding(capability))
        return capability

    @staticmethod
    def _capability_binding(capability: TriggerCapability) -> tuple[CronRuntime, str, str | None, str, dict[str, object], int, int]:
        return (capability.runtime, capability.source, capability.controller, capability.request_id,
                capability.definition.to_wire(), capability.clock_ms, capability.pid)

    def authorize_manual(self, request_id: str, job_id: str, instance: str, revision: str) -> TriggerCapability:
        if not self._accepting or self.injection.manual_authorizer(job_id, instance, revision) is not True:
            raise PermissionError("manual execution is not authorized")
        return self._issue("manual", None, request_id, job_id, instance, revision)

    def request_from_controller(self, controller: InternalController, name: str,
                                request_id: str, job_id: str, instance: str,
                                revision: str) -> TriggerCapability:
        """Issue only after checking the original controller identity and PID."""
        binding = self._controllers.get(id(controller))
        if binding != (controller, name, os.getpid()):
            raise PermissionError("controller is foreign, copied or changed")
        return self._issue("system", name, request_id, job_id, instance, revision)

    def construct_internal_controller(self, name: str) -> InternalController:
        if not self._accepting or not isinstance(cast(object, name), str) or not name.strip() or len(name.encode()) > 512:
            raise ValueError("explicit named controller required")
        if name.lower() in {"dream", "heartbeat"}:
            raise ValueError("Dream and Heartbeat controllers are not admitted")
        controller = InternalController(_TRIGGER_TOKEN, self, name)
        self._controllers[id(controller)] = (controller, name, os.getpid())
        return controller

    async def execute_now(self, capability: TriggerCapability) -> object:
        issued = self._capabilities.get(id(capability))
        if issued is None or issued[0] is not capability or capability.pid != os.getpid() or issued[1] != self._capability_binding(capability):
            raise PermissionError("foreign or changed trigger capability")
        definition = self._definition(capability.definition.job_id, capability.definition.job_instance_id,
                                      capability.definition.definition_revision)
        inputs: TriggerInputs = {"job_id": definition.job_id, "job_instance_id": definition.job_instance_id,
                  "definition_revision": definition.definition_revision}
        request = self.store.allocate_request(source=capability.source, controller=capability.controller,
            request_id=capability.request_id, inputs=inputs, clock_ms=capability.clock_ms)
        identity = TriggeredOccurrenceIdentity(self.store.namespace, definition.job_id,
            definition.definition_revision, request["clock_ms"], capability.source,
            request["sequence"], capability.controller)
        # This graph installs both collaborators at construction; _definition
        # rechecks the runtime before this synchronous allocation sequence.
        claim = cast(CronOccurrenceStore, self.service.occurrence_store).claim_triggered(identity,
            job_snapshot=definition.to_wire(), destination_policy=destination_policy(definition),
            owner=cast(CronOccurrenceRuntime, self.service.occurrence_runtime).owner)
        return await self._dispatch(claim, lambda: self._definition(definition.job_id,
            definition.job_instance_id, definition.definition_revision))

    def stop_claiming(self) -> None:
        self._accepting = False
        self.scheduler.stop_claiming()

    def begin_graceful_shutdown(self) -> None:
        """Fence new work and join current effects without a timeout or cancellation."""
        if self._shutdown_mode != "forced":
            self._shutdown_mode = "graceful"
            self._shutdown_changed.clear()
        self.stop_claiming()

    def request_bounded_shutdown(self) -> None:
        """Replace a graceful wait with the ordinary bounded drain policy."""
        if self._shutdown_mode != "forced":
            self._shutdown_mode = "bounded"
        self._shutdown_changed.set()
        self.stop_claiming()

    def force_shutdown(self) -> None:
        """Cancel admitted cron work directly; resistant effects retain ownership."""
        self._shutdown_mode = "forced"
        self._shutdown_changed.set()
        self.stop_claiming()
        for task in tuple(self._tasks):
            if not task.done() and task.cancelling() == 0:
                task.cancel()

    async def _drain_bounded(
        self,
        tasks: set[asyncio.Task[ClaimedTextCompositionResult]],
        *,
        already_cancelled: bool = False,
    ) -> None:
        pending = {task for task in tasks if not task.done()}
        if not pending:
            return
        if already_cancelled:
            _done, pending = await asyncio.wait(pending, timeout=self.injection.drain_timeout)
            if pending:
                raise RuntimeOwnedWorkError("cron effects did not drain; resources and lease retained")
            return
        _done, pending = await asyncio.wait(pending, timeout=self.injection.drain_timeout)
        for task in pending:
            task.cancel()
        if pending:
            _done, pending = await asyncio.wait(pending, timeout=self.injection.drain_timeout)
        if pending:
            raise RuntimeOwnedWorkError("cron effects did not drain; resources and lease retained")

    async def _join_gracefully(
        self,
        tasks: set[asyncio.Task[ClaimedTextCompositionResult]],
    ) -> None:
        pending = {task for task in tasks if not task.done()}
        if not pending:
            return
        changed = asyncio.create_task(self._shutdown_changed.wait())
        try:
            while pending:
                done, _pending = await asyncio.wait(
                    (*pending, changed), return_when=asyncio.FIRST_COMPLETED,
                )
                pending.difference_update(done)
                if self._shutdown_mode != "graceful":
                    await self._drain_bounded(
                        pending, already_cancelled=self._shutdown_mode == "forced",
                    )
                    return
        finally:
            changed.cancel()
            await asyncio.gather(changed, return_exceptions=True)

    async def _close(self) -> None:
        self.stop_claiming()
        startup = self._startup_task
        if startup is not None and startup is not asyncio.current_task() and not startup.done():
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)
        if self._timer_task is not None:
            self._timer_task.cancel()
            await asyncio.gather(self._timer_task, return_exceptions=True)
        tasks = tuple(self._tasks)
        if tasks:
            owned = set(tasks)
            if self._shutdown_mode == "graceful":
                await self._join_gracefully(owned)
            else:
                await self._drain_bounded(
                    owned, already_cancelled=self._shutdown_mode == "forced",
                )
        self.scheduler.release_lease()
        if self._task_errors:
            errors = tuple(self._task_errors)
            self._task_errors.clear()
            failure = CronWorkFailedError("admitted cron work failed; durable states require inspection")
            cause: BaseException = errors[0]
            if len(errors) > 1:
                if all(isinstance(error, Exception) for error in errors):
                    cause = ExceptionGroup(
                        "admitted cron work failures",
                        [error for error in errors if isinstance(error, Exception)],
                    )
                else:
                    cause = BaseExceptionGroup("admitted cron work failures", list(errors))
            raise failure from cause

    async def aclose(self) -> None:
        self.stop_claiming()
        if self._close_task is not None and self._close_task.done() and isinstance(self._close_task.exception(), RuntimeOwnedWorkError):
            self._close_task = None  # Explicit retry after resistant work joins.
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError
