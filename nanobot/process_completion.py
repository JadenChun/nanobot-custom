"""Durable process completion to the existing assistant's session-turn queue."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nanobot.agent.automation_turns import AutomationTurnError
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.exec_session import ExecCompletion
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.security.workspace_access import WorkspaceAccessMode, build_workspace_scope
from nanobot.session.manager import SessionManager
from nanobot.session.source_store import SourceOriginMismatchError, SourceState
from nanobot.triggers.local_session_turns import LOCAL_TRIGGER_META


class ProcessAssistant(Protocol):
    sessions: SessionManager

    async def submit_local_trigger_turn(self, msg: InboundMessage) -> OutboundMessage | None: ...


class ProcessRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    state: Literal["waiting", "ready", "dispatching", "done", "cancelled", "unknown"] = "waiting"
    started_at_ms: int
    session_key: str
    channel: str
    chat_id: str
    instruction: str = Field(max_length=64000)
    instruction_ref: str | None = None
    command: str = Field(max_length=16000)
    cwd: str
    project_path: str
    access_mode: WorkspaceAccessMode
    reply_metadata: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    reason: str | None = None
    source_state: SourceState | None = None
    _stored_json: str | None = PrivateAttr(default=None)

    @classmethod
    def from_stored_json(cls, raw: str) -> ProcessRecord:
        record = cls.model_validate_json(raw)
        record._stored_json = raw
        return record

    @property
    def original_json(self) -> str:
        """Original journal bytes, excluded from persisted/public record fields."""
        if self._stored_json is None:
            raise ValueError("process transition requires its original stored bytes")
        return self._stored_json


class _PersistenceLane:
    """One bounded ordered SQLite writer for one process owner."""

    def __init__(self, path: Path, *, max_pending: int = 64) -> None:
        self.path = path
        self._queue: queue.Queue[tuple[Callable[[sqlite3.Connection], Any], concurrent.futures.Future[Any]] | None] = queue.Queue(maxsize=max_pending)
        self._closed = False
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="process-completion-journal",
            daemon=True,
        )
        self._thread.start()
        self.run(lambda db: db.execute("CREATE TABLE IF NOT EXISTS processes (id TEXT PRIMARY KEY, data TEXT NOT NULL)"))

    def run(self, operation: Callable[[sqlite3.Connection], Any]) -> Any:
        if self._startup_error is not None:
            raise RuntimeError("process completion journal could not open") from self._startup_error
        if self._closed:
            raise RuntimeError("process completion journal is closed")
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._queue.put((operation, future))
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError as exc:
            raise RuntimeError("process completion journal did not settle") from exc

    def _run(self) -> None:
        try:
            db = sqlite3.connect(self.path)
        except BaseException as exc:
            self._startup_error = exc
            self._closed = True
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    return
                if item is not None:
                    item[1].set_exception(exc)
            return
        try:
            db.execute("PRAGMA synchronous=FULL")
            while True:
                item = self._queue.get()
                if item is None:
                    return
                operation, future = item
                try:
                    with db:
                        result = operation(db)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
        finally:
            db.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("process completion journal did not stop")


class ProcessCompletionRuntime:
    """Journal first, then submit once through canonical assistant ownership.

    A crash with an active process or admitted continuation is unknown. It is
    retained for inspection and never treated as permission to replay effects.
    """

    def __init__(self, root: Path, assistant: ProcessAssistant):
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = root / "processes.sqlite3"
        self.assistant = assistant
        self._started = False
        self._closed = False
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lane = _PersistenceLane(self.path)
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._loop_thread_id = threading.get_ident() if self._loop is not None else None

    def _read(self, operation: Callable[[sqlite3.Connection], Any]) -> Any:
        try:
            return self._lane.run(operation)
        except RuntimeError:
            # Closed runtimes keep their durable read surface inspectable.
            db = sqlite3.connect(self.path)
            try:
                return operation(db)
            finally:
                db.close()

    def records(self) -> list[ProcessRecord]:
        rows = self._read(lambda db: db.execute("SELECT data FROM processes").fetchall())
        return [self._parse_record(row[0]) for row in rows]

    @staticmethod
    def _parse_record(raw: str) -> ProcessRecord:
        return ProcessRecord.from_stored_json(raw)

    def get(self, identity: str) -> ProcessRecord:
        row = self._read(
            lambda db: db.execute(
                "SELECT data FROM processes WHERE id=?", (identity,)
            ).fetchone()
        )
        if row is None:
            raise ValueError("unknown background process")
        record = self._parse_record(row[0])
        if record.id != identity:
            raise ValueError("process journal identity mismatch")
        return record

    def _save(self, record: ProcessRecord, *, expected: str | None) -> None:
        raw = record.model_dump_json()
        def operation(db: sqlite3.Connection) -> None:
            db.execute("BEGIN IMMEDIATE")
            if expected is None:
                if db.execute("SELECT count(*) FROM processes").fetchone()[0] >= 4096:
                    raise ValueError(
                        "process journal is full; review retained records before starting more work"
                    )
                db.execute(
                    "INSERT INTO processes(id,data) VALUES (?,?)",
                    (record.id, raw),
                )
            else:
                result = db.execute(
                    "UPDATE processes SET data=? WHERE id=? AND data=?",
                    (raw, record.id, expected),
                )
                if result.rowcount != 1:
                    raise ValueError("process continuation state changed")

        self._lane.run(operation)

    def _transition(self, record: ProcessRecord, state: Literal["waiting", "ready", "dispatching", "done", "cancelled", "unknown"], **updates: object) -> None:
        replacement = ProcessRecord.model_validate({**record.model_dump(), **updates, "state": state})
        self._save(replacement, expected=record.original_json)

    def started(self, session_id: str, command: str, cwd: str, origin: RequestContext) -> None:
        if self._closed or not self._started or not origin.session_key or origin.workspace_scope is None:
            raise ValueError("background completion requires a running gateway and a captured workspace scope")
        if not origin.persistent_session:
            raise ValueError("durable background continuation is unavailable for temporary conversations")
        if not isinstance(origin.source_state, SourceState) or origin.source_state.session_key != origin.session_key:
            raise ValueError("background completion requires an admitted canonical source")
        scope = origin.workspace_scope
        record = ProcessRecord(
            id=session_id, started_at_ms=int(time.time() * 1000), session_key=origin.session_key,
            channel=origin.channel, chat_id=origin.chat_id, command=command, cwd=cwd,
            project_path=str(scope.project_path), access_mode=scope.access_mode,
            instruction=origin.original_user_text or "", instruction_ref=origin.turn_id,
            reply_metadata={name: origin.metadata[name] for name in ("message_id", "message_thread_id", "thread_id")
                            if name in origin.metadata},
            source_state=origin.source_state,
        )
        self._save(record, expected=None)

    def inline_completed(self, session_id: str) -> None:
        record = self.get(session_id)
        self._transition(record, "done", reason="returned_to_foreground")

    def finished(self, session_id: str, result: ExecCompletion) -> None:
        record = self.get(session_id)
        if record.state != "waiting":
            return  # One event per process; duplicates never enqueue a turn.
        self._transition(record, "unknown" if self._closed else "ready", output=result.output,
                         exit_code=result.exit_code, timed_out=result.timed_out)
        if not self._closed:
            self._schedule(session_id)

    def cancelled(self, session_id: str, reason: str) -> None:
        record = self.get(session_id)
        if record.state == "waiting":
            self._transition(record, "cancelled", reason=reason)

    def result(self, session_id: str, owner_session_key: str | None, *,
               source_state: SourceState | None = None) -> ExecCompletion | None:
        try:
            record = self.get(session_id)
        except ValueError:
            return None
        if (record.session_key != owner_session_key or record.state == "waiting"
                or record.source_state is None or source_state is None
                or not record.source_state.same_origin(source_state)):
            return None
        current = self.assistant.sessions.read_existing_canonical_source(record.session_key)
        if current is None or not current[1].state.same_origin(source_state):
            return None
        state = "unknown" if record.state == "unknown" else "cancelled" if record.state == "cancelled" else "completed"
        return ExecCompletion(record.output, record.exit_code, record.timed_out, record.state == "cancelled", state)

    def start(self) -> None:
        if self._closed or self._started:
            raise RuntimeError("process continuation runtime cannot start twice")
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            self._loop_thread_id = threading.get_ident()
        for record in self.records():
            if record.state in ("waiting", "dispatching"):
                self._transition(record, "unknown", reason="runtime_interrupted")
        self._started = True
        for record in self.records():
            if record.state == "ready":
                self._schedule(record.id)

    def _schedule(self, identity: str) -> None:
        loop = self._loop
        if (
            loop is not None
            and loop.is_running()
            and self._loop_thread_id is not None
            and threading.get_ident() != self._loop_thread_id
        ):
            loop.call_soon_threadsafe(self._schedule_on_loop, identity)
            return
        self._schedule_on_loop(identity)

    def _schedule_on_loop(self, identity: str) -> None:
        if identity not in self._tasks:
            task = asyncio.create_task(self._continue(identity))
            self._tasks[identity] = task
            task.add_done_callback(lambda done: self._task_done(identity, done))

    def _task_done(self, identity: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(identity, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Background continuation {} requires inspection: {}", identity, type(task.exception()).__name__)

    async def _continue(self, identity: str) -> None:
        record = await asyncio.to_thread(self.get, identity)
        if record.state != "ready":
            return
        current = self.assistant.sessions.read_existing_canonical_source(record.session_key)
        if (record.source_state is None or current is None
                or not current[1].state.same_origin(record.source_state)):
            await asyncio.to_thread(
                self._transition,
                record,
                "cancelled",
                reason="source_changed" if record.source_state is not None else "source_origin_missing",
            )
            return
        await asyncio.to_thread(self._transition, record, "dispatching")
        try:
            metadata = {**record.reply_metadata, LOCAL_TRIGGER_META: {
                "trigger_id": f"process:{identity}", "trigger_name": "Background process completion",
                "delivery_id": f"process:{identity}", "persist_content": f"Background process {identity} completed.",
            }}
            details = record.model_dump(exclude={"state", "session_key", "channel", "chat_id", "reply_metadata", "source_state"})
            message = InboundMessage(
                channel=record.channel, chat_id=record.chat_id, sender_id="system:process",
                session_key_override=record.session_key, metadata=metadata, input_role="system",
                workspace_scope=build_workspace_scope(record.project_path, record.access_mode),
                source_state=record.source_state,
                content=("A background process from this conversation finished. Continue its original task "
                         "using the retained project and process directory. The conversation may now be on "
                         "another project. Inspect this result; do not rerun the completed command.\n" + json.dumps(details)),
            )
            await self.assistant.submit_local_trigger_turn(message)
            await asyncio.to_thread(
                self._transition,
                await asyncio.to_thread(self.get, identity),
                "done",
            )
        except BaseException as exc:
            current = await asyncio.to_thread(self.get, identity)
            if current.state == "dispatching":
                stale_origin = (isinstance(exc, SourceOriginMismatchError)
                                or isinstance(exc, AutomationTurnError)
                                and isinstance(exc.__cause__, SourceOriginMismatchError))
                await asyncio.to_thread(
                    self._transition,
                    current,
                    "cancelled" if stale_origin else "unknown",
                    reason="source_changed" if stale_origin else "continuation_interrupted",
                )
                if stale_origin:
                    return
            raise

    async def aclose(self) -> None:
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._lane.close()
