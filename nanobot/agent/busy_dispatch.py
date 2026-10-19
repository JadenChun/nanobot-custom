"""Explicit owning inbox consumer; no channel authentication or runtime registration."""

from __future__ import annotations

import asyncio
import math
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace

from nanobot.agent.automation_turns import AutomationTurnError
from nanobot.agent.busy_control import ControlReport, ControlSnapshot
from nanobot.agent.busy_steering import BusySteering, SteeringMessage
from nanobot.agent.context import ContextBuilder
from nanobot.agent.turn import RunRecord, RunStatus, TurnContext, TurnResult
from nanobot.bus.events import InboundMessage
from nanobot.session.busy_inbox import InboxConflictError, InboxEntry, SessionBusyInbox
from nanobot.session.manager import Session


@dataclass
class BusyTurn:
    """One consumer-owned invocation, bound only after actual session admission."""

    consumer: BusyInboxConsumer = field(repr=False)
    run_id: str
    initial: InboxEntry | None = field(default=None, repr=False)
    automation: InboundMessage | None = field(default=None, repr=False)
    steering: BusySteering | None = field(default=None, init=False, repr=False)
    initial_messages: tuple[SteeringMessage, ...] = field(default=(), init=False, repr=False)
    stop_requested: bool = field(default=False, init=False)
    admission_closed: bool = field(default=False, init=False)
    cleanup_error: BaseException | None = field(default=None, init=False, repr=False)
    delivery_error: str | None = field(default=None, init=False)

    def check_stop(self) -> None:
        """Prevent fresh dispatch if an in-flight collaborator ate cancellation."""
        if self.stop_requested:
            raise asyncio.CancelledError

    def admit(self, context: TurnContext, builder: ContextBuilder) -> None:
        self.check_stop()
        snapshot = self.consumer.current_run
        if (self.steering is not None or snapshot is None or context.run_id != self.run_id
                or snapshot.run_id != self.run_id or context.session_key != self.consumer.inbox.session_key):
            raise ValueError("busy turn requires its actual admitted invocation")
        self.steering = BusySteering(
            self.consumer.inbox, context=context, observed_run=snapshot,
            active_scope=self.consumer.active_scope, context_builder=builder,
            current_run=lambda: self.consumer.current_run,
        )

    def persist_initial(self, session: Session) -> None:
        if self.initial is None or self.steering is None or self.initial_messages:
            raise ValueError("busy initial input must be recorded once after admission")
        messages = self.steering.record_initial(self.initial)
        if self.steering.context.source_writer is None:
            if self.initial.payload is None:
                raise ValueError("busy initial input has no payload")
            for payload in (self.initial.payload, *self.initial.clarifications):
                session.add_message("user", payload.content, _run_id=self.run_id,
                                    **({"media": list(payload.media)} if payload.media else {}))
            session.provider_state = None
            self.consumer.inbox.sessions.save(session, fsync=True)
        self.initial_messages = messages


@dataclass
class _AutomationWork:
    message: InboundMessage = field(repr=False)
    result: asyncio.Future[TurnResult] = field(repr=False)


@dataclass
class _LiveRun:
    turn: BusyTurn
    snapshot: ControlSnapshot
    task: asyncio.Task[object]
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_sent: bool = False

    def matches(self, observed: ControlSnapshot | None) -> bool:
        return (observed is not None and observed.run_id == self.snapshot.run_id
                and observed.generation == self.snapshot.generation)


class BusyInboxConsumer:
    """One task per explicitly registered owner/session, one real turn at a time.

    Human work lives in the ledger, not a second memory queue. Automation retains
    its existing occurrence owner and gets a separate finite in-process lane.
    Wakes never create an overflow runner. Unknown effects fence both lanes.
    """

    def __init__(self, inbox: SessionBusyInbox, *,
                 execute: Callable[[BusyTurn], Awaitable[TurnResult]],
                 track_task: Callable[[asyncio.Task[None]], None],
                 automation_scope: tuple[str, ...], human_burst: int = 3,
                 terminal_record: Callable[[str], RunRecord | None] | None = None):
        if type(human_burst) is not int or not 1 <= human_burst <= 16:
            raise ValueError("human burst must be between one and sixteen")
        self.inbox = inbox
        self._execute = execute
        self._track_task = track_task
        self._automation_scope = automation_scope
        self._human_burst = human_burst
        self._terminal_record = terminal_record
        self._human_turns = 0
        self._generation = 0
        self._automation: deque[_AutomationWork] = deque()
        self.task: asyncio.Task[None] | None = None
        self._active: _LiveRun | None = None
        self.active_scope: tuple[str, ...] = ()
        self.last_error: BaseException | None = None
        self._closed = False
        self._automation_uncertain = False
        self._delivery_wakers: list[Callable[[], None]] = []

    @staticmethod
    def _queued(entries: tuple[InboxEntry, ...]) -> int:
        return sum(row.state == "next-turn" or (
            row.state == "control" and row.intent == "stop-and-replace") for row in entries)

    @property
    def current_run(self) -> ControlSnapshot | None:
        active = self._active
        return replace(active.snapshot, queued=self._queued(self.inbox.entries())) if active else None

    def _terminal(self, run_id: str) -> RunRecord | None:
        record = self._terminal_record(run_id) if self._terminal_record is not None else None
        if record is None or record.status is RunStatus.RUNNING or record.completed_at is None:
            return None
        if (record.run_id != run_id or record.session_ref is None
                or record.session_ref.session_key != self.inbox.session_key):
            raise RuntimeError("control requires this session's actual terminal record")
        return record

    def _report(self, observed: ControlSnapshot | None, *, active_run: ControlSnapshot | None,
                recovery: bool = False) -> ControlReport:
        entries = self.inbox.entries()
        record = self._terminal(observed.run_id) if observed is not None else None
        return ControlReport(active_run=active_run, queued=self._queued(entries),
            automation_queued=len(self._automation),
            recovery_required=recovery or self._automation_uncertain
                or any(row.state == "uncertain" or (row.state == "claimed" and (
                    self._active is None or row.claimed_run_id != self._active.snapshot.run_id)) for row in entries),
            target_status=record.status if record else None)

    def status(self, observed_run: ControlSnapshot | None = None) -> ControlReport:
        """Read actual ownership, durable work and terminal facts without history."""
        return self._report(observed_run, active_run=self.current_run)

    def _entry(self, identity: str, revision: int | None = None) -> InboxEntry:
        row = next((entry for entry in self.inbox.entries() if entry.id == identity), None)
        if row is None or (revision is not None and (type(revision) is not int or row.revision != revision)):
            raise InboxConflictError("busy inbox revision changed")
        return row

    def _settle_inactive_controls(self) -> None:
        settled = False
        for row in self.inbox.entries():
            if row.state == "control" and (self._active is None or not self._active.matches(row.observed_run)):
                report = self._report(row.observed_run, active_run=self.current_run,
                                      recovery=row.control_started)
                self.inbox.finish_control(row.id, row.revision, report=report)
                settled = True
        if settled:
            self._wake_delivery_wakers()

    async def apply_control(self, identity: str, revision: int, *, wait_seconds: float = 3.0) -> InboxEntry:
        """Apply an already-routed control to its exact live owner, at most once.

        The bounded wait watches one slice's settlement event. It never cancels
        execution after waiting and never transfers authority to a newer slice.
        The consumer, not this possibly cancelled waiter, persists the outcome.
        """
        if (isinstance(wait_seconds, bool) or not math.isfinite(wait_seconds)
                or not 0 <= wait_seconds <= 30):
            raise ValueError("control wait must be between zero and thirty seconds")
        if self._closed:
            raise RuntimeError("inbox consumer is closed")
        row = self._entry(identity, revision)
        if row.state != "control":
            if row.control_report is not None:
                return row
            raise InboxConflictError("input is not a routed control")
        active = self._active
        if row.intent == "status" or active is None or not active.matches(row.observed_run):
            finished = self.inbox.finish_control(row.id, row.revision,
                report=self._report(row.observed_run, active_run=self.current_run,
                                    recovery=row.control_started))
            self._wake_delivery_wakers()
            return finished
        if active.task is asyncio.current_task():
            raise RuntimeError("a run cannot wait for its own stop settlement")
        stopping = replace(active.snapshot, state="stopping", queued=self._queued(self.inbox.entries()))
        if not row.control_started:
            row = self.inbox.begin_control(row.id, row.revision, current_run=active.snapshot,
                                          report=self._report(row.observed_run, active_run=stopping))
        # There is deliberately no await between exact-handle validation, the
        # durable intent and the one cancellation request. This task may later
        # drain younger slices; only the captured live slice owns this request.
        if not active.turn.stop_requested:
            active.snapshot = stopping
            active.turn.stop_requested = True
            if not active.turn.admission_closed and not active.task.done() and not active.task.cancelling():
                active.cancel_sent = active.task.cancel()
        try:
            await asyncio.wait_for(active.settled.wait(), timeout=wait_seconds)
        except asyncio.TimeoutError:
            if self._active is active and not active.settled.is_set():
                active.snapshot = replace(active.snapshot, state="blocked")
                for pending in self.inbox.entries():
                    if pending.state == "control" and pending.control_started and active.matches(pending.observed_run):
                        self.inbox.block_control(pending.id, pending.revision,
                                                report=self.status(pending.observed_run))
        self._settle_inactive_controls()
        return self._entry(identity)

    def recover(self) -> None:
        """Only an idle registered owner may reconcile its orphaned ledger."""
        if self.task is not None and not self.task.done():
            raise RuntimeError("cannot recover a live inbox consumer")
        if self._automation_uncertain:
            raise RuntimeError("automation requires its occurrence owner's recovery")
        self.inbox.reconcile_orphans(live_run_ids=frozenset())
        self._settle_inactive_controls()

    def add_delivery_waker(self, waker: Callable[[], None]) -> None:
        if not callable(waker):
            raise TypeError("delivery waker must be callable")
        if waker not in self._delivery_wakers:
            self._delivery_wakers.append(waker)

    def remove_delivery_waker(self, waker: Callable[[], None]) -> None:
        try:
            self._delivery_wakers.remove(waker)
        except ValueError:
            pass

    def _wake_delivery_wakers(self) -> None:
        for waker in tuple(self._delivery_wakers):
            try:
                waker()
            except Exception:
                # Delivery wake failures cannot change task or control settlement.
                continue

    def wake(self) -> None:
        if self._closed:
            raise RuntimeError("inbox consumer is closed")
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._consume())
            self._track_task(self.task)

    async def wait_idle(self) -> None:
        task = self.task
        if task is not None:
            await asyncio.shield(task)

    async def submit_automation(self, message: InboundMessage) -> TurnResult:
        if self._closed:
            raise AutomationTurnError("inbox consumer is closed")
        if message.session_key != self.inbox.session_key:
            raise ValueError("automation belongs to another inbox session")
        if len(self._automation) >= self.inbox.limits.max_pending:
            raise AutomationTurnError("inbox automation capacity reached")
        future: asyncio.Future[TurnResult] = asyncio.get_running_loop().create_future()
        # A cancelled waiter does not abandon already accepted execution. Retrieve
        # eventual errors even if its occurrence owner has stopped waiting.
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        self._automation.append(_AutomationWork(deepcopy(message), future))
        self.wake()
        return await asyncio.shield(future)

    def _fail_queued(self, reason: str) -> None:
        while self._automation:
            work = self._automation.popleft()
            if not work.result.done():
                work.result.set_exception(AutomationTurnError(reason))

    async def _consume(self) -> None:
        work: _AutomationWork | None = None
        try:
            while not self._closed:
                if self._automation_uncertain:
                    self._fail_queued("automation requires its occurrence owner's recovery")
                    return
                self._settle_inactive_controls()
                entries = self.inbox.entries()
                if any(row.state in {"claimed", "uncertain"} for row in entries):
                    self._fail_queued("inbox requires owning recovery before more execution")
                    return
                ready = any(row.state == "next-turn" and not any(
                    prior.state == "pending-routing" and prior.eligible_seq < row.eligible_seq
                    for prior in entries) for row in entries)
                run_id = uuid.uuid4().hex
                initial = None
                work = None
                if self._automation and (not ready or self._human_turns >= self._human_burst):
                    work = self._automation.popleft()
                    self.active_scope = self._automation_scope
                    self._human_turns = 0
                else:
                    initial = self.inbox.claim_next_turn(run_id, current_run=None)
                    if initial is None:
                        return
                    self.active_scope = tuple(sorted(initial.effective_scope))
                    self._human_turns = min(self._human_turns + 1, self._human_burst)
                self._generation += 1
                turn = BusyTurn(self, run_id, initial=initial, automation=work.message if work else None)
                task = asyncio.current_task()
                assert task is not None
                active = _LiveRun(turn, ControlSnapshot(run_id, self._generation, "running"), task)
                self._active = active
                result = None
                error = None
                record = None
                try:
                    result = await self._execute(turn)
                    record = result.record
                    if (result.run_id != run_id or record is None or record.run_id != run_id
                            or record.status is not result.status or record.completed_at is None
                            or record.session_ref is None or record.session_ref.session_key != self.inbox.session_key):
                        raise RuntimeError("inbox execution requires its persisted terminal run record")
                except BaseException as exc:
                    error = exc
                    if not (turn.stop_requested and isinstance(exc, asyncio.CancelledError)):
                        self.last_error = exc
                finally:
                    # Use current durable revisions, including partial source
                    # bindings, never a stale in-memory claim after a write fault.
                    if turn.stop_requested:
                        record = self._terminal(run_id)
                    if turn.cleanup_error is not None:
                        self.last_error = turn.cleanup_error
                    complete = (turn.cleanup_error is None and record is not None
                        and record.status in {RunStatus.COMPLETED, RunStatus.COMMAND}
                        and ((result is not None and error is None)
                             or (turn.stop_requested and isinstance(error, asyncio.CancelledError))))
                    owned = tuple(row for row in self.inbox.entries()
                                  if row.state == "claimed" and row.claimed_run_id == run_id)
                    # The durable dispatch marker proves whether this claim
                    # could have reached effects. Managed privacy intentionally
                    # omits terminal run records for unadmitted cancellation.
                    safe_cancel = (turn.stop_requested and isinstance(error, asyncio.CancelledError)
                        and turn.cleanup_error is None and (record is None or record.status is RunStatus.CANCELLED)
                        and initial is not None and all(not row.dispatch_started for row in owned))
                    for row in self.inbox.entries():
                        if row.state == "claimed" and row.claimed_run_id == run_id:
                            if safe_cancel:
                                self.inbox.finish(row.id, row.revision, "cancelled")
                            elif row.dispatch_started:
                                self.inbox.finish(row.id, row.revision, "completed" if complete else "uncertain")
                    if turn.automation is not None and not complete:
                        self._automation_uncertain = True
                    for row in self.inbox.entries():
                        if row.state == "control" and active.matches(row.observed_run):
                            self.inbox.finish_control(row.id, row.revision,
                                report=self._report(row.observed_run, active_run=None,
                                                    recovery=not (complete or safe_cancel)))
                    self._active = None
                    self.active_scope = ()
                    self.inbox.reconcile_orphans(live_run_ids=frozenset(), recover_notices=False)
                    self._wake_delivery_wakers()
                    active.settled.set()
                    if active.cancel_sent:
                        active.task.uncancel()
                if work is not None and not work.result.done():
                    if complete and result is None:
                        work.result.set_exception(AutomationTurnError("automation completed with delivery uncertainty"))
                    elif not complete or result is None:
                        work.result.set_exception(AutomationTurnError("automation execution did not complete"))
                    elif result.record is not None and result.record.metadata.get("delivery_error_type"):
                        work.result.set_exception(AutomationTurnError("automation completed with delivery uncertainty"))
                    else:
                        work.result.set_result(result)
                work = None
                if not (complete or safe_cancel):
                    self._fail_queued("inbox execution requires owning recovery")
                    return
                if active.task.cancelling():
                    self._fail_queued("inbox execution interrupted")
                    return
                # A local provider may finish without suspending. Give resolved
                # occurrence waiters and new control input a slice before more work.
                await asyncio.sleep(0)
        except BaseException as exc:
            self.last_error = exc
            if work is not None and not work.result.done():
                work.result.set_exception(AutomationTurnError("inbox execution interrupted"))
            self._fail_queued("inbox execution interrupted")
        finally:
            active = self._active
            if active is not None:
                active.settled.set()
            self._active = None
            self.active_scope = ()

    async def aclose(self, *, exclude: asyncio.Task[object] | None = None) -> None:
        self._closed = True
        self._fail_queued("inbox consumer shut down before execution")
        task = self.task
        if task is not None and task is not asyncio.current_task() and task is not exclude:
            if not task.done() and not task.cancelling():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
