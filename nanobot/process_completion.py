"""Durable process completion to the existing assistant's session-turn queue."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Literal, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.exec_session import ExecCompletion
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.security.workspace_access import WorkspaceAccessMode, build_workspace_scope
from nanobot.session.manager import SessionManager
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
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS processes (id TEXT PRIMARY KEY, data TEXT NOT NULL)")

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        # Create privately before SQLite opens it; journal files inherit this mode.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        os.close(fd)
        db = sqlite3.connect(self.path)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def records(self) -> list[ProcessRecord]:
        with self._connect() as db:
            return [ProcessRecord.model_validate_json(row[0]) for row in db.execute("SELECT data FROM processes")]

    def get(self, identity: str) -> ProcessRecord:
        with self._connect() as db:
            row = db.execute("SELECT data FROM processes WHERE id=?", (identity,)).fetchone()
            if row is None:
                raise ValueError("unknown background process")
            record = ProcessRecord.model_validate_json(row[0])
            if record.id != identity:
                raise ValueError("process journal identity mismatch")
            return record

    def _save(self, record: ProcessRecord, *, expected: str | None) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if expected is None:
                if db.execute("SELECT count(*) FROM processes").fetchone()[0] >= 4096:
                    raise ValueError("process journal is full; review retained records before starting more work")
                db.execute("INSERT INTO processes(id,data) VALUES (?,?)", (record.id, record.model_dump_json()))
            else:
                result = db.execute("UPDATE processes SET data=? WHERE id=? AND data=?",
                                    (record.model_dump_json(), record.id, expected))
                if result.rowcount != 1:
                    raise ValueError("process continuation state changed")

    def _transition(self, record: ProcessRecord, state: Literal["waiting", "ready", "dispatching", "done", "cancelled", "unknown"], **updates: object) -> None:
        replacement = ProcessRecord.model_validate({**record.model_dump(), **updates, "state": state})
        self._save(replacement, expected=record.model_dump_json())

    def started(self, session_id: str, command: str, cwd: str, origin: RequestContext) -> None:
        if self._closed or not self._started or not origin.session_key or origin.workspace_scope is None:
            raise ValueError("background completion requires a running gateway and a captured workspace scope")
        if not origin.persistent_session:
            raise ValueError("durable background continuation is unavailable for temporary conversations")
        scope = origin.workspace_scope
        record = ProcessRecord(
            id=session_id, started_at_ms=int(time.time() * 1000), session_key=origin.session_key,
            channel=origin.channel, chat_id=origin.chat_id, command=command, cwd=cwd,
            project_path=str(scope.project_path), access_mode=scope.access_mode,
            instruction=origin.original_user_text or "", instruction_ref=origin.turn_id,
            reply_metadata={name: origin.metadata[name] for name in ("message_id", "message_thread_id", "thread_id")
                            if name in origin.metadata},
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

    def result(self, session_id: str, owner_session_key: str | None) -> ExecCompletion | None:
        try:
            record = self.get(session_id)
        except ValueError:
            return None
        if record.session_key != owner_session_key or record.state == "waiting":
            return None
        state = "unknown" if record.state == "unknown" else "cancelled" if record.state == "cancelled" else "completed"
        return ExecCompletion(record.output, record.exit_code, record.timed_out, record.state == "cancelled", state)

    def start(self) -> None:
        if self._closed or self._started:
            raise RuntimeError("process continuation runtime cannot start twice")
        for record in self.records():
            if record.state in ("waiting", "dispatching"):
                self._transition(record, "unknown", reason="runtime_interrupted")
        self._started = True
        for record in self.records():
            if record.state == "ready":
                self._schedule(record.id)

    def _schedule(self, identity: str) -> None:
        if identity not in self._tasks:
            task = asyncio.create_task(self._continue(identity))
            self._tasks[identity] = task
            task.add_done_callback(lambda done: self._task_done(identity, done))

    def _task_done(self, identity: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(identity, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Background continuation {} requires inspection: {}", identity, type(task.exception()).__name__)

    async def _continue(self, identity: str) -> None:
        record = self.get(identity)
        if record.state != "ready":
            return
        if self.assistant.sessions.read_session_snapshot(record.session_key) is None:
            self._transition(record, "cancelled", reason="owner_session_missing")
            return
        self._transition(record, "dispatching")
        try:
            metadata = {**record.reply_metadata, LOCAL_TRIGGER_META: {
                "trigger_id": f"process:{identity}", "trigger_name": "Background process completion",
                "delivery_id": f"process:{identity}", "persist_content": f"Background process {identity} completed.",
            }}
            details = record.model_dump(exclude={"state", "session_key", "channel", "chat_id", "reply_metadata"})
            message = InboundMessage(
                channel=record.channel, chat_id=record.chat_id, sender_id="system:process",
                session_key_override=record.session_key, metadata=metadata, input_role="system",
                workspace_scope=build_workspace_scope(record.project_path, record.access_mode),
                content=("A background process from this conversation finished. Continue its original task "
                         "using the retained project and process directory. The conversation may now be on "
                         "another project. Inspect this result; do not rerun the completed command.\n" + json.dumps(details)),
            )
            await self.assistant.submit_local_trigger_turn(message)
            self._transition(self.get(identity), "done")
        except BaseException:
            current = self.get(identity)
            if current.state == "dispatching":
                self._transition(current, "unknown", reason="continuation_interrupted")
            raise

    async def aclose(self) -> None:
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
