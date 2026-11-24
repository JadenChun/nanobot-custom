"""Exercise the runner using an admitted source writer and scripted backend."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from kohaku.agent.hook import AgentHook
from kohaku.agent.turn import ApprovalDecision
from kohaku.harness.backend import (
    ExternalApprovalRegistry,
    ExternalReasoning,
    ExternalSessionStarted,
    ExternalText,
    ExternalToolCall,
    ExternalToolIntent,
    ExternalToolPolicy,
    ExternalToolResult,
    ExternalTurnFinished,
    ExternalTurnRequest,
    SessionHarnessBinding,
)
from kohaku.harness.runner import ExternalHarnessRunner
from kohaku.providers.base import GenerationSettings
from kohaku.session.execution_owner import acquire_execution
from kohaku.session.manager import JsonlSessionStore, Session
from kohaku.session.turn_writer import SourceTurnWriter
from kohaku.utils.llm_runtime import LLMRuntime


class CaptureHook(AgentHook):
    def __init__(self):
        super().__init__()
        self.text = []
        self.reasoning = []

    async def on_stream(self, context, delta):
        self.text.append(delta)

    async def emit_reasoning(self, text):
        self.reasoning.append(text)

    async def emit_reasoning_end(self):
        self.reasoning.append(None)


class ScriptedProcess:
    def __init__(self, script, session_id):
        self.script = script
        self.session_id = session_id
        self.closed = asyncio.Event()

    async def run(self, request, approve):
        yield ExternalSessionStarted(self.session_id)
        async for event in self.script(request, approve):
            yield event

    async def close(self):
        self.closed.set()


class ScriptedBackend:
    def __init__(self, script):
        self.script = script
        self.starts = []
        self.processes = []

    async def start(self, request, *, resume, fork_session):
        self.starts.append((resume, fork_session, request.policy.mode))
        session_id = resume if resume and not fork_session else f"sdk-{len(self.starts)}"
        process = ScriptedProcess(self.script, session_id)
        self.processes.append(process)
        return process


@pytest.fixture
def store(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return JsonlSessionStore(workspace, sessions_root=tmp_path / "sessions")


@asynccontextmanager
async def request_for(store, run_id, key="test:external", **updates):
    if store.load(key) is None:
        session = Session(key, messages=[{"role": "user", "content": "Investigate."}])
        store.save(session)
        store.source.upgrade_source_session(
            key, expected_revision=session.revision, expected_generation=session.generation,
            expected_incarnation=session.incarnation,
        )
    async with acquire_execution(store, key, run_id=run_id) as owner:
        request = ExternalTurnRequest(
            session_key=key, run_id=run_id,
            binding=SessionHarnessBinding("external", "claude_sdk", None, None, 3),
            workspace=store.workspace, prompt="Continue.",
            runtime=LLMRuntime(None, "test-model", GenerationSettings(), 4096),
            hook=CaptureHook(), writer=SourceTurnWriter(store.source, owner, run_id=run_id),
            is_authorized=lambda: True,
        )
        yield replace(request, **updates)


async def test_streams_reasoning_and_records_normalized_tools_and_observed_edits(store):
    tools = ("Read", "Write", "Edit", "Bash", "Glob", "Grep")
    file_edit = {"path": "new.txt", "call_id": "1", "tool": "write_file"}
    async def script(request, approve):
        yield ExternalReasoning("thinking")
        yield ExternalText("Checking. ")
        for i, tool in enumerate(tools):
            arguments = {"file_path": "new.txt"} if i < 3 else {}
            yield ExternalToolIntent(ExternalToolCall(str(i), tool, arguments))
        for i in range(len(tools)):
            yield ExternalToolResult(str(i), "tool output", file_edit=file_edit if i == 1 else None)
        yield ExternalText("Done.")
        yield ExternalTurnFinished()
    runner = ExternalHarnessRunner(ScriptedBackend(script))
    try:
        async with request_for(store, "first") as request:
            outcome = await runner.run(request)
            events = request.writer.stream_events()
            assert request.hook.text == ["Checking. ", "Done."]
            assert request.hook.reasoning == ["thinking", None]
        assert outcome.content == "Checking. Done."
        assert outcome.status == "completed"
        assert "thinking" not in json.dumps(events)
        intents = [entry for entry in events if entry.get("tool_calls")][0]
        expected = ["read_file", "write_file", "edit_file", "exec", "find_files", "grep"]
        assert [call["function"]["name"] for call in intents["tool_calls"]] == expected
        assert json.loads(intents["tool_calls"][0]["function"]["arguments"]) == {"path": "new.txt"}
        assert [entry["name"] for entry in events if entry["role"] == "tool"] == expected
        assert events[-1]["fileEdits"] == [file_edit]
    finally:
        await runner.shutdown()


async def test_idle_mode_changes_and_fork_resume_the_current_session_id(store):
    async def script(request, approve):
        if request.run_id == "two":
            yield ExternalSessionStarted("current-id")
        yield ExternalText("Done.")
        yield ExternalTurnFinished()
    backend = ScriptedBackend(script)
    runner = ExternalHarnessRunner(backend, idle_timeout=0.05)
    observed = []
    try:
        for run in ("one", "two"):
            async with request_for(store, run, on_session_id=observed.append) as request:
                await runner.run(request)
        assert backend.starts == [(None, False, "sandbox")]
        assert observed == ["sdk-1", "current-id"]
        await asyncio.wait_for(backend.processes[0].closed.wait(), 3)
        async with request_for(store, "three") as request:
            await runner.run(request)
        async with request_for(store, "four", policy=ExternalToolPolicy(mode="full_access")) as request:
            await runner.run(request)
        assert backend.starts[1:] == [
            ("current-id", False, "sandbox"), ("current-id", False, "full_access"),
        ]
        assert backend.processes[1].closed.is_set()
        async with request_for(
            store, "fork", key="test:fork", prior_external_session_id="current-id",
            on_session_id=observed.append,
        ) as request:
            result = await runner.run(request)
        assert backend.starts[-1] == ("current-id", True, "sandbox")
        assert result.binding.external_session_id == observed[-1] != "current-id"
        snapshot = store.source.load_source_snapshot("test:fork")
        assert snapshot.header["metadata"]["harness_binding"]["external_session_id"] == observed[-1]
    finally:
        await runner.shutdown()


@pytest.mark.parametrize(
    ("command", "action", "allowed", "saved"),
    [
        ("pytest -q", "approve", True, []),
        ("pytest -q", "deny", False, []),
        ("pytest -q", "approve_and_save", True, ["pytest"]),
        ("pytest -q && echo done", "approve_and_save", False, []),
    ],
)
async def test_approval_is_bound_single_use_and_saves_only_eligible_commands(
    store, command, action, allowed, saved,
):
    persisted = []
    registry = ExternalApprovalRegistry(persisted.append)
    async def notify(approval):
        decision = ApprovalDecision(
            approval.approval_id, action, approval.session_key,
            run_id=approval.run_id, generation=approval.runtime_generation,
        )
        assert not registry.resolve(replace(decision, generation=0))
        assert not registry.resolve(replace(decision, run_id="other"))
        assert not registry.resolve(replace(decision, session_key="test:other"))
        assert not registry.resolve(decision, save_exclusion_command="other")
        if action == "approve_and_save" and not saved:
            assert approval.save_exclusion_command is None
            assert not registry.resolve(decision)
            decision = replace(decision, decision="deny")
        assert registry.resolve(decision)
        assert not registry.resolve(decision)
    async with request_for(store, "approval", on_approval=notify) as request:
        call = ExternalToolCall("call-1", "exec", {"command": command})
        assert await registry.request(request, call, "Run outside the sandbox?", True) == (action if allowed else "deny")
    assert persisted == saved


async def test_stop_closes_process_and_dismisses_pending_approval(store):
    approval_pending = asyncio.Future()
    async def notify(approval):
        approval_pending.set_result(approval)

    async def script(request, approve):
        await approve(ExternalToolCall("tool-1", "Bash", {"command": "pytest"}), "Allow?", True)
        yield ExternalTurnFinished()

    backend = ScriptedBackend(script)
    runner = ExternalHarnessRunner(backend)

    async def run():
        async with request_for(store, "stop", on_approval=notify) as request:
            return await runner.run(request)

    task = asyncio.create_task(run())
    try:
        pending = await asyncio.wait_for(approval_pending, 3)
        await runner.stop("test:external")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.processes[0].closed.is_set()
        assert not runner.approvals.resolve(ApprovalDecision(
            pending.approval_id, "approve", pending.session_key,
            run_id=pending.run_id, generation=pending.runtime_generation,
        ))
        events = store.source.load_source_snapshot("test:external").events
        assert events[-1]["_external_outcome"] == "cancelled"
    finally:
        await runner.shutdown()


@pytest.mark.parametrize("cancel", ["run", "session"])
async def test_approval_waiter_is_dismissed_by_its_owner(store, cancel):
    registry = ExternalApprovalRegistry()
    async def notify(approval):
        if cancel == "run":
            registry.cancel_for_run(approval.run_id)
        else:
            registry.cancel_for_session(approval.session_key)
    async with request_for(store, "cancel", on_approval=notify) as request:
        assert await registry.request(
            request, ExternalToolCall("tool", "exec", {"command": "pytest"}), "Allow?", True,
        ) == "deny"
