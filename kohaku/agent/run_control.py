"""Shared lifecycle and control identity for one active session run."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

from loguru import logger

from kohaku.bus.outbound_events import RunLifecycleState, RunState

RunStateSnapshot = RunState
_RUN_STOP_FIELDS = frozenset({"chat_id", "source_incarnation", "turn_id", "run_id"})
_RUN_STOP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class RunStopPayload(TypedDict):
    chat_id: str
    source_incarnation: str
    turn_id: str
    run_id: str


@dataclass(frozen=True, slots=True)
class RunStopTarget:
    chat_id: str
    source_incarnation: str
    turn_id: str
    run_id: str


def parse_run_stop_payload(value: object) -> RunStopTarget:
    """Validate the exact identity required by the dedicated WebUI Stop action."""
    if not isinstance(value, dict):
        raise ValueError("run.stop payload must be an object")
    raw = cast(dict[object, object], value)
    keys = tuple(raw)
    if (
        getattr(raw, "duplicate_keys", ())
        or any(not isinstance(key, str) for key in keys)
        or frozenset(key for key in keys if isinstance(key, str)) != _RUN_STOP_FIELDS
    ):
        raise ValueError("run.stop payload fields do not match the contract")
    values: list[str] = []
    for name in ("chat_id", "source_incarnation", "turn_id", "run_id"):
        candidate = raw.get(name)
        if not isinstance(candidate, str) or _RUN_STOP_ID_RE.fullmatch(candidate) is None:
            raise ValueError("run.stop payload contains an invalid identity")
        values.append(candidate)
    return RunStopTarget(*values)


class RunControlAdapter(Protocol):
    """Harness boundary for control actions on one admitted run."""

    async def request_stop(self) -> None: ...

    async def enqueue_steer(self, prompt: str) -> bool: ...

    async def resolve_approval(self, approval_id: str, decision: object) -> bool: ...


class ExternalStopper(Protocol):
    async def stop(self, session_key: str) -> None: ...


@dataclass(slots=True)
class TaskRunControlAdapter:
    """Cancel one native root task through the shared adapter contract."""

    task: asyncio.Task[object]

    async def request_stop(self) -> None:
        if not self.task.done() and not self.task.cancelling():
            self.task.cancel()

    async def enqueue_steer(self, prompt: str) -> bool:
        del prompt
        return False

    async def resolve_approval(self, approval_id: str, decision: object) -> bool:
        del approval_id, decision
        return False


@dataclass(slots=True)
class ExternalTaskRunControlAdapter:
    """Use the external harness owner's cancellation path for one run."""

    runner: ExternalStopper
    session_key: str
    stop_task: asyncio.Task[None] | None = None

    async def request_stop(self) -> None:
        if self.stop_task is None or self.stop_task.done():
            self.stop_task = asyncio.create_task(self.runner.stop(self.session_key))
            self.stop_task.add_done_callback(self._consume_stop_result)
        # ExternalHarnessRunner marks its session as stopping before its first
        # await, then the existing loop joins the root under its Stop deadline.
        await asyncio.sleep(0)

    @staticmethod
    def _consume_stop_result(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def enqueue_steer(self, prompt: str) -> bool:
        del prompt
        return False

    async def resolve_approval(self, approval_id: str, decision: object) -> bool:
        del approval_id, decision
        return False


@dataclass(slots=True)
class ActiveRunHandle:
    run_id: str
    turn_id: str | None
    source_incarnation: str | None
    state: RunLifecycleState
    cancel_requested: bool
    active_task: asyncio.Task[object] | None
    workspace_scope: dict[str, object] | None
    started_at: float
    harness_kind: str
    steer_mailbox: asyncio.Queue[object] | None = None
    active_children: tuple[str, ...] = ()
    adapter: RunControlAdapter | None = None
    can_steer: bool = False


class SessionRunController:
    """Own the current control identity for each session key."""

    def __init__(self) -> None:
        self._active: dict[str, ActiveRunHandle] = {}

    def register(self, session_key: str, handle: ActiveRunHandle) -> None:
        current = self._active.get(session_key)
        if current is not None and current is not handle:
            task = current.active_task
            if task is not None and not task.done():
                raise RuntimeError("The session already has an active run.")
        self._active[session_key] = handle

    def unregister(self, session_key: str, handle: ActiveRunHandle) -> bool:
        if self._active.get(session_key) is not handle:
            return False
        self._active.pop(session_key, None)
        return True

    def active_run(self, session_key: str) -> ActiveRunHandle | None:
        handle = self._active.get(session_key)
        if handle is None:
            return None
        task = handle.active_task
        if task is not None and task.done():
            return None
        return handle

    def snapshot(self, session_key: str) -> RunStateSnapshot | None:
        handle = self.active_run(session_key)
        if handle is None:
            return None
        active = handle.state in {"running", "waiting_approval"}
        return {
            "state": handle.state,
            "turn_id": handle.turn_id,
            "run_id": handle.run_id,
            "source_incarnation": handle.source_incarnation,
            "started_at": handle.started_at,
            "can_stop": active and all((
                handle.source_incarnation,
                handle.turn_id,
                handle.run_id,
            )),
            "can_steer": active and handle.can_steer,
        }

    async def request_stop(
        self,
        session_key: str,
        *,
        expected_incarnation: str | None,
        expected_turn_id: str | None = None,
        expected_run_id: str | None = None,
        on_state_change: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        """Mark and stop the matching run without acting on a replacement."""
        handle = self.active_run(session_key)
        if (
            handle is None
            or handle.source_incarnation != expected_incarnation
            or (expected_turn_id is not None and handle.turn_id != expected_turn_id)
            or (expected_run_id is not None and handle.run_id != expected_run_id)
            or handle.state == "stopping"
        ):
            return False
        handle.cancel_requested = True
        handle.state = "stopping"
        if on_state_change is not None:
            await on_state_change()
        adapter = handle.adapter
        try:
            if adapter is not None:
                await adapter.request_stop()
            else:
                task = handle.active_task
                if task is not None and not task.done() and not task.cancelling():
                    task.cancel()
        except asyncio.CancelledError:
            # A repeated Stop or caller cancellation cannot leave the run live.
            task = handle.active_task
            if task is not None and not task.done() and not task.cancelling():
                task.cancel()
            raise
        except Exception as exc:
            logger.warning("Run Stop adapter failed for {}: {}", session_key, type(exc).__name__)
            task = handle.active_task
            if task is not None and not task.done() and not task.cancelling():
                task.cancel()
        return True

    def set_state(
        self,
        session_key: str,
        run_id: str,
        state: RunLifecycleState,
    ) -> RunStateSnapshot | None:
        handle = self.active_run(session_key)
        if handle is None or handle.run_id != run_id or handle.cancel_requested:
            return None
        handle.state = state
        return self.snapshot(session_key)

    def set_steer_mailbox(
        self,
        session_key: str,
        run_id: str,
        mailbox: asyncio.Queue[object] | None,
    ) -> None:
        handle = self.active_run(session_key)
        if handle is not None and handle.run_id == run_id:
            handle.steer_mailbox = mailbox

    def set_active_children(
        self,
        session_key: str,
        run_id: str,
        child_ids: tuple[str, ...],
    ) -> None:
        handle = self.active_run(session_key)
        if handle is not None and handle.run_id == run_id:
            handle.active_children = child_ids
