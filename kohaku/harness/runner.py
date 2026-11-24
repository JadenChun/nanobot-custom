"""Run an external turn under the session owner's source writer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from kohaku.agent.hook import AgentHookContext, AgentRunHookContext
from kohaku.agent.tools.execution import KnownToolResult
from kohaku.bus.outbound_events import FileEditEvent
from kohaku.harness.backend import (
    ExternalApprovalAction,
    ExternalApprovalRegistry,
    ExternalEvent,
    ExternalHarnessProcess,
    ExternalReasoning,
    ExternalSessionStarted,
    ExternalText,
    ExternalToolBatchReady,
    ExternalToolCall,
    ExternalToolIntent,
    ExternalToolResult,
    ExternalTurnOutcome,
    ExternalTurnRequest,
    HarnessCapabilityError,
    SessionHarnessBackend,
)

CLAUDE_TOOL_NAMES = {
    "Read": "read_file", "Write": "write_file", "Edit": "edit_file",
    "NotebookEdit": "edit_file", "Bash": "exec", "Glob": "find_files", "Grep": "grep",
}


def normalize_tool_call(call: ExternalToolCall) -> ExternalToolCall:
    arguments = dict(call.arguments)
    for old, new in (("file_path", "path"), ("notebook_path", "path"),
                     ("old_string", "old_text"), ("new_string", "new_text")):
        if old in arguments:
            arguments[new] = arguments.pop(old)
    return ExternalToolCall(call.id, CLAUDE_TOOL_NAMES.get(call.name, call.name), arguments)


@dataclass(slots=True)
class _Session:
    incarnation: str
    workspace: Path
    session_id: str | None
    process: ExternalHarnessProcess | None = None
    launch_key: tuple[object, ...] | None = None
    idle_timer: asyncio.TimerHandle | None = None
    closing: asyncio.Task[None] | None = None
    active: asyncio.Task[object] | None = None
    done: asyncio.Future[None] | None = None
    stopping: bool = False

    def close_process(self) -> asyncio.Task[None] | None:
        if self.idle_timer is not None:
            self.idle_timer.cancel()
            self.idle_timer = None
        if self.process is not None:
            self.closing = asyncio.create_task(self.process.close())
            self.process = None
        return self.closing

    async def wait_closed(self) -> None:
        if (closing := self.closing) is not None:
            try:
                await asyncio.shield(closing)
            finally:
                if closing.done() and self.closing is closing:
                    self.closing = None


class ExternalHarnessRunner:
    """Keep one SDK process per admitted session and close it when idle."""

    def __init__(
        self, backend: SessionHarnessBackend, *,
        approvals: ExternalApprovalRegistry | None = None, idle_timeout: float = 300,
    ) -> None:
        self._backend = backend
        self.approvals = approvals if approvals is not None else ExternalApprovalRegistry()
        self._idle_timeout = idle_timeout
        self._sessions: dict[str, _Session] = {}
        self._closed = False

    async def run(self, request: ExternalTurnRequest) -> ExternalTurnOutcome:
        if self._closed:
            raise RuntimeError("The external harness runner is closed.")
        writer = request.writer
        if writer.run_id != request.run_id or writer.owner.key != request.session_key:
            raise ValueError("The source writer belongs to a different run or session.")
        writer.owner.validate_dispatch()
        if not request.is_authorized():
            raise PermissionError("The session has no current permission to run.")
        if request.binding.backend != "external" or request.binding.harness is None:
            raise ValueError("The session requires an external harness binding.")
        workspace = request.workspace.resolve(strict=True)
        if not workspace.is_dir():
            raise ValueError("The harness workspace must be a directory.")
        if request.binding.locked_workspace is not None and (
            Path(request.binding.locked_workspace).resolve() != workspace
        ):
            raise ValueError("The harness workspace is locked.")
        binding = replace(request.binding, locked_workspace=str(workspace))
        request = replace(request, workspace=workspace, binding=binding)
        session = self._sessions.get(request.session_key)
        incarnation = writer.committed_state.incarnation
        if session is None:
            session = _Session(incarnation, workspace, binding.external_session_id)
            self._sessions[request.session_key] = session
        if session.active is not None:
            raise RuntimeError("The session already has an active external turn.")

        if session.incarnation == incarnation and session.workspace != workspace:
            raise ValueError("The harness workspace is locked.")

        task = asyncio.current_task()
        assert task is not None
        session.active = task
        session.stopping = False
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        session.done = done
        turn = _Turn(request)
        reusable = False

        async def approve(
            call: ExternalToolCall, reason: str, allow_save: bool,
        ) -> ExternalApprovalAction:
            try:
                return await self.approvals.request(request, normalize_tool_call(call), reason, allow_save)
            except asyncio.CancelledError:
                turn.record_unapproved_call(call.id)
                raise

        try:
            if session.incarnation != incarnation:
                session.close_process()
                session.incarnation = incarnation
                session.workspace = workspace
                session.session_id = binding.external_session_id
            if session.idle_timer is not None:
                session.idle_timer.cancel()
                session.idle_timer = None
            launch_key = (
                workspace, request.runtime, request.usage_source, request.policy,
                binding.harness, binding.runtime_generation,
            )
            if session.launch_key != launch_key:
                session.close_process()
            await session.wait_closed()
            if not request.is_authorized():
                raise PermissionError("The session has no current permission to run.")
            if session.process is None:
                fork = session.session_id is None and request.prior_external_session_id is not None
                session.process = await self._backend.start(
                    request, resume=session.session_id or request.prior_external_session_id,
                    fork_session=fork,
                )
                session.launch_key = launch_key
            turn.persist_binding(session.session_id)
            await request.hook.before_run(turn.run_context)
            await turn.consume(session, session.process.run(request, approve))
            outcome = await turn.finish(session.session_id)
            reusable = (
                request.retain_process and outcome.status == "completed" and session.session_id is not None
            )
            return outcome
        except asyncio.CancelledError:
            session.stopping = True
            turn.status = "cancelled"
            for call_id in self.approvals.pending_tool_calls(request.run_id):
                turn.record_unapproved_call(call_id)
            raise
        except Exception as exc:
            turn.status = "failed"
            turn.run_context.exception = exc
            if isinstance(exc, HarnessCapabilityError):
                turn.error = str(exc)
            raise
        finally:
            self.approvals.cancel_for_run(request.run_id)
            try:
                if not reusable:
                    session.close_process()
                    await session.wait_closed()
                if not turn.settled and not writer.failed:
                    await turn.finish(session.session_id)
            finally:
                session.active = None
                done.set_result(None)
                if not request.retain_process:
                    self._sessions.pop(request.session_key, None)
                if reusable and session.process is not None:
                    session.idle_timer = asyncio.get_running_loop().call_later(
                        self._idle_timeout, session.close_process,
                    )

    async def stop(self, session_key: str) -> None:
        self.approvals.cancel_for_session(session_key)
        session = self._sessions.get(session_key)
        if session is None:
            return
        if session.active is not None:
            if session.active is asyncio.current_task():
                raise RuntimeError("Stop must run outside the active turn.")
            if not session.stopping:
                session.stopping = True
                session.active.cancel()
            if session.done is not None:
                await asyncio.shield(session.done)
        else:
            session.close_process()
            await session.wait_closed()

    async def shutdown(self) -> None:
        self._closed = True
        await asyncio.gather(*(self.stop(key) for key in self._sessions))
        self._sessions.clear()


@dataclass(slots=True)
class _Turn:
    request: ExternalTurnRequest
    context: AgentHookContext = field(init=False)
    run_context: AgentRunHookContext = field(default_factory=lambda: AgentRunHookContext([]))
    text: list[str] = field(default_factory=list)
    segment: list[str] = field(default_factory=list)
    queued_calls: dict[str, ExternalToolCall] = field(default_factory=dict)
    active_calls: dict[str, tuple[int, ExternalToolCall]] = field(default_factory=dict)
    results: dict[str, ExternalToolResult] = field(default_factory=dict)
    edits: list[dict[str, object]] = field(default_factory=list)
    commands: list[tuple[str, str]] = field(default_factory=list)
    status: Literal["completed", "failed", "cancelled"] = "failed"
    error: str | None = None
    reasoning: bool = False
    settled: bool = False

    def __post_init__(self) -> None:
        self.context = AgentHookContext(1, [], session_key=self.request.session_key)

    def persist_binding(self, session_id: str | None) -> None:
        binding = replace(self.request.binding, external_session_id=session_id)
        writer = self.request.writer
        payload = binding.payload()
        if writer.owner.read_snapshot().header["metadata"].get("harness_binding") != payload:
            writer.store.update_metadata(self.request.session_key, {"harness_binding": payload})

    async def consume(self, session: _Session, events: AsyncGenerator[ExternalEvent, None]) -> None:
        async with aclosing(events):
            async for event in events:
                if isinstance(event, ExternalSessionStarted):
                    if event.session_id != session.session_id:
                        self.persist_binding(event.session_id)
                        session.session_id = event.session_id
                        if self.request.on_session_id is not None:
                            self.request.on_session_id(event.session_id)
                elif isinstance(event, ExternalText):
                    await self.add_text(event.text)
                elif isinstance(event, ExternalReasoning):
                    if event.text is None:
                        await self.end_reasoning()
                    else:
                        self.reasoning = True
                        self.context.streamed_reasoning = True
                        await self.request.hook.emit_reasoning(event.text)
                elif isinstance(event, ExternalToolIntent):
                    await self.tool_intent(event.call)
                elif isinstance(event, ExternalToolResult):
                    if event.call_id in self.active_calls or event.call_id in self.queued_calls:
                        self.results[event.call_id] = event
                        await self.flush_tools()
                elif isinstance(event, ExternalToolBatchReady):
                    if self.queued_calls and not self.active_calls:
                        await self.write_batch()
                else:
                    if event.final_text and not self.text:
                        await self.add_text(event.final_text)
                    self.error = event.error
                    if event.cancelled:
                        self.status = "cancelled"
                    elif event.error is None and not (self.active_calls or self.queued_calls):
                        self.status = "completed"
                    else:
                        self.error = event.error or "Claude Code did not report every tool result."
                    return
        self.error = "Claude Code closed the stream before the turn finished."

    async def add_text(self, text: str) -> None:
        await self.end_reasoning()
        self.text.append(text)
        self.segment.append(text)
        self.context.streamed_content = True
        await self.request.hook.on_stream(self.context, text)

    async def end_reasoning(self) -> None:
        if self.reasoning:
            self.reasoning = False
            await self.request.hook.emit_reasoning_end()

    async def tool_intent(self, intent: ExternalToolCall) -> None:
        await self.end_reasoning()
        call = normalize_tool_call(intent)
        self.queued_calls[call.id] = call
        if call.name not in self.run_context.tools_used:
            self.run_context.tools_used.append(call.name)
        await self.request.hook.on_provider_tool_event(self.context, {
            "phase": "start", "call_id": call.id, "name": call.name, "arguments": call.arguments,
        })

    async def write_batch(self) -> None:
        calls = list(self.queued_calls.values())
        self.active_calls = {call.id: (index, call) for index, call in enumerate(calls)}
        self.queued_calls.clear()
        entry = self.request.writer.begin_tool_batch({
            "role": "assistant",
            "content": "".join(self.segment) or None,
            "tool_calls": [
                {"id": call.id, "type": "function", "function": {
                    "name": call.name, "arguments": json.dumps(call.arguments),
                }}
                for call in calls
            ],
        })
        await self.end_segment(entry["_event_id"], resuming=True)

    def record_unapproved_call(self, call_id: str) -> None:
        if call_id in self.active_calls or call_id in self.queued_calls:
            self.results[call_id] = ExternalToolResult(
                call_id, "The turn stopped before approval.", is_error=True, dispatched=False,
            )

    async def flush_tools(self) -> None:
        # SourceTurnWriter accepts results for one batch at a time. Buffer SDK
        # results that arrive for a later batch while the current batch settles.
        while self.results:
            if not self.active_calls:
                await self.write_batch()
            ready = self.results.keys() & self.active_calls.keys()
            if not ready:
                return
            for call_id in tuple(self.results):
                if call_id not in ready:
                    continue
                result = self.results.pop(call_id)
                ordinal, call = self.active_calls.pop(call_id)
                status = "error" if result.is_error else "ok"
                command = call.arguments.get("command")
                if call.name == "exec" and isinstance(command, str):
                    self.commands.append((command, status if result.dispatched else "not_run"))
                await self.request.writer.record_tool_result(KnownToolResult(
                    ordinal, call.id, call.name, result.content, result.content,
                    {"status": status}, result.dispatched,
                ))
                phase = "error" if result.is_error else "end"
                await self.request.hook.on_provider_tool_event(self.context, {
                    "phase": phase, "call_id": call_id, "name": call.name,
                    "arguments": call.arguments,
                    "error" if result.is_error else "result": result.content,
                })
                if result.file_edit is not None:
                    self.edits.append(result.file_edit)
                    if self.request.events.accepts(FileEditEvent):
                        await self.request.events.emit(FileEditEvent(file_edit_events=[result.file_edit]))

    async def end_segment(self, event_id: str, *, resuming: bool) -> None:
        self.context.source_event_id = event_id
        if self.segment:
            self.segment.clear()
            await self.request.hook.on_stream_end(self.context, resuming=resuming)

    async def finish(self, session_id: str | None) -> ExternalTurnOutcome:
        self.settled = True
        hook = self.request.hook
        try:
            await self.end_reasoning()
            if self.queued_calls:
                # Leave unresolved intents for source recovery after interruption.
                await self.write_batch()
            await self.flush_tools()
            entry = self.request.writer.append_message({
                "role": "assistant", "content": "".join(self.segment) or self.error or "",
                "_external_outcome": self.status, "_external_error": self.error,
                "fileEdits": self.edits,
            })
            await self.end_segment(entry["_event_id"], resuming=False)
            content = "".join(self.text)
            self.run_context.final_content = content
            self.run_context.error = self.error
            self.run_context.stop_reason = self.status
            if self.status == "failed":
                await hook.on_error(self.run_context)
            else:
                await hook.after_run(self.run_context)
            return ExternalTurnOutcome(
                replace(self.request.binding, external_session_id=session_id),
                content, self.status, self.error,
                tuple(self.run_context.tools_used), tuple(self.edits), tuple(self.commands),
            )
        finally:
            await hook.on_finally(self.run_context)
