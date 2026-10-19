"""Explicitly injected dormant cron graph sharing foreground execution authorities."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, NoReturn, SupportsIndex, cast

from nanobot.cron.canonical_execution import SilentCronAgent
from nanobot.cron.channel_sender import CronChannelSender, ExactChannelAuthority
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.composition import ClaimedTextCompositionResult, compose_claimed_text_occurrence
from nanobot.cron.definition_store import CronDefinition
from nanobot.cron.delivery_runtime import CronDeliveryRuntime
from nanobot.cron.delivery_store import CronDeliveryOutboxStore
from nanobot.cron.occurrence import CronOccurrenceRuntime, destination_policy
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceClaim,
    TriggeredOccurrenceIdentity,
)
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
                 injection: CronRuntimeInjection):
        if service.occurrence_store is not None or service.on_job is not None:
            raise ValueError("dormant graph requires definition-only service without callbacks")
        self.service, self.agent, self.sessions, self.channels = service, agent, sessions, channels
        self.injection = injection
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
                                   self.store, injection.clock, self._dispatch)
        self._accepting = False
        self._tasks: set[asyncio.Task[ClaimedTextCompositionResult]] = set()
        self._task_errors: list[BaseException] = []
        self._capabilities: dict[int, tuple[TriggerCapability, tuple[CronRuntime, str, str | None, str, dict[str, object], int, int]]] = {}
        self._controllers: dict[int, tuple[InternalController, str, int]] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._timer_task: asyncio.Task[None] | None = None

    def acquire(self) -> None:
        if self._close_task is not None:
            raise SchedulerConflictError("closed cron runtime cannot reacquire")
        self.scheduler.acquire_and_reconcile()
        self._accepting = True

    def start_timer_task(self) -> None:
        """Separate explicit offline injection; acquisition itself starts no task."""
        if not self._accepting or self._timer_task is not None:
            raise SchedulerConflictError("timer task cannot start")
        self._timer_task = asyncio.create_task(self._run_timer())

    async def _run_timer(self) -> None:
        while self._accepting:
            await self.scheduler.wake_once()
            await self.injection.clock.sleep(0.05)

    async def _dispatch(self, claim: OccurrenceClaim, pre_begin: Callable[[], object]) -> object:
        if not self._accepting:
            raise SchedulerConflictError("cron ingress fenced")
        task = asyncio.create_task(compose_claimed_text_occurrence(
            self.service, self.agent, claim, self.sender, visible_sessions=self.sessions,
            pre_begin=pre_begin,
        ))
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

    async def _close(self) -> None:
        self.stop_claiming()
        if self._timer_task is not None:
            self._timer_task.cancel()
            await asyncio.gather(self._timer_task, return_exceptions=True)
        tasks = tuple(self._tasks)
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=self.injection.drain_timeout)
            for task in pending:
                task.cancel()
            if pending:
                _done, pending = await asyncio.wait(pending, timeout=self.injection.drain_timeout)
            if pending:
                raise RuntimeOwnedWorkError("cron effects did not drain; resources and lease retained")
        self.scheduler.release_lease()
        if self._task_errors:
            self._task_errors.clear()
            raise CronWorkFailedError("admitted cron work failed; durable states require inspection")

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
