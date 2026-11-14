"""Real filesystem failures retain effect evidence and fence the source owner."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.apply_patch import ApplyPatchTool
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.filesystem import EditFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.config.schema import ToolsConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.execution_owner import ExecutionOwnershipError, acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import validate_execution_links
from nanobot.session.turn_writer import SourceTurnWriter
from tests.agent.runner_helpers import make_run_spec


@pytest.fixture
def backend(tmp_path):
    store = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    session = Session("test:effect")
    store.save(session)
    store.source.upgrade_source_session(session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)
    return store


@pytest.mark.parametrize("failure", ["record_write", "touch", "write_then_raise", "cancel"])
async def test_real_edit_failure_stops_repeated_provider_and_tail_effect(
    backend, monkeypatch, failure,
):
    target = backend.workspace / "target.txt"
    target.write_text("x")
    tail = backend.workspace / "tail.txt"
    context = ToolContext(config=ToolsConfig(), workspace=str(backend.workspace))
    tool = EditFileTool.create(context)
    tools = ToolRegistry()
    tools.register(tool)
    tools.register(WriteFileTool.create(context))

    def fail(*_args):
        if failure == "cancel":
            raise asyncio.CancelledError("cancelled after write")
        raise OSError("tracking failed")

    if failure in {"record_write", "cancel"}:
        monkeypatch.setattr(type(tool._file_states), "record_write", fail)
    elif failure == "touch":
        monkeypatch.setattr(tool, "_record_touch", fail)
    else:
        original = Path.write_bytes

        def write_then_raise(path, data):
            result = original(path, data)
            if path == target:
                raise OSError("write acknowledgement lost")
            return result

        monkeypatch.setattr(Path, "write_bytes", write_then_raise)

    response = LLMResponse(content="", tool_calls=[
        ToolCallRequest("edit", "edit_file", {"path": str(target), "old_text": "x",
                                             "new_text": "xx", "occurrence": 1}),
        ToolCallRequest("tail", "write_file", {"path": str(tail), "content": "tail"}),
    ])
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=response)
    caught = None
    async with acquire_execution(backend, "test:effect", run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        writer.append_message({"role": "user", "content": "edit once"})
        try:
            await AgentRunner().run(make_run_spec(provider, initial_messages=[
                {"role": "user", "content": "edit once"}], tools=tools, model="offline",
                max_iterations=2, max_tool_result_chars=1000, source_writer=writer,
                turn_context=TurnContext(execution_owner=owner), concurrent_tools=False))
        except BaseException as exc:
            caught = exc
        assert target.read_text() == "xx"
        assert not tail.exists()
        assert provider.chat_with_retry.await_count == 1
        assert caught is not None
        if failure == "cancel":
            assert isinstance(caught, asyncio.CancelledError)
        else:
            assert type(caught).__name__ == "FilesystemEffectReconciliationRequiredError"
        snapshot = owner.read_snapshot()
        result = snapshot.events[-1]
        assert result["role"] == "tool" and result["_dispatched"] is True
        assert "Analyze the error" not in str(result["content"])
        facts = result["_filesystem_effects"]
        assert facts[0]["target"] == str(target)
        assert facts[0]["execution_id"] == result["_execution_id"]
        assert facts[0]["reconciliation_required"] is True
        assert len(str(facts)) < 2000
        assert snapshot.header["metadata"]["execution_recovery_required"] is True
        assert result["_execution_id"] in validate_execution_links(snapshot.events)
    reopened = JsonlSessionStore(backend.workspace, sessions_root=backend.sessions_root)
    assert reopened.source.load_source_snapshot("test:effect").header["metadata"][
        "execution_recovery_required"] is True


async def test_write_entry_precedes_mkdir_failure_and_pre_effect_refusal_has_no_fact(
    tmp_path, monkeypatch,
):
    tool = WriteFileTool(workspace=tmp_path, allowed_dir=tmp_path)
    refused = await ToolRegistry.execute_prepared(tool, {"path": "../outside", "content": "x"})
    assert not getattr(refused, "filesystem_effects", ())
    original = Path.mkdir

    def mkdir_then_raise(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path == tmp_path / "new-parent":
            raise OSError("mkdir acknowledgement lost")

    monkeypatch.setattr(Path, "mkdir", mkdir_then_raise)
    result = await ToolRegistry.execute_prepared(tool, {
        "path": "new-parent/new.txt", "content": "bounded content"})
    assert isinstance(result, ToolOutcome)
    assert result.filesystem_effects[0].reconciliation_required
    assert result.filesystem_effects[0].disposition == "uncertain"
    assert (tmp_path / "new-parent").exists()
    assert not (tmp_path / "new-parent/new.txt").exists()


async def run_mutation(backend, tool, arguments, *, reject_result=False):
    """Exercise production dispatch, durable recording and the next-model boundary."""
    tools = ToolRegistry()
    tools.register(tool)
    if tool.name != "write_file":
        tools.register(WriteFileTool(workspace=backend.workspace))
    tail = backend.workspace / "tail.txt"
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="", tool_calls=[
        ToolCallRequest("effect", tool.name, arguments),
        ToolCallRequest("tail", "write_file", {"path": str(tail), "content": "tail"}),
    ]))
    async with acquire_execution(backend, "test:effect", run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        writer.append_message({"role": "user", "content": "mutate once"})
        original = type(backend.source).append_source_events

        def append(store, *args, **kwargs):
            if reject_result and any(event.get("role") == "tool" for event in args[2]):
                raise OSError("mandatory result publication failed")
            return original(store, *args, **kwargs)

        with pytest.MonkeyPatch.context() as patches:
            patches.setattr(type(backend.source), "append_source_events", append)
            with pytest.raises(BaseException) as caught:
                await AgentRunner().run(make_run_spec(provider, initial_messages=[
                    {"role": "user", "content": "mutate once"}], tools=tools, model="offline",
                    max_iterations=2, max_tool_result_chars=1000, source_writer=writer,
                    turn_context=TurnContext(execution_owner=owner), concurrent_tools=False))
        assert provider.chat_with_retry.await_count == 1
        assert not tail.exists()
        assert writer.failed
        return caught.value, owner.read_snapshot()


@pytest.mark.parametrize("failure", ["record_write", "write_then_raise", "rollback", "mkdir", "cancel"])
async def test_patch_failures_retain_evidence_and_stop_the_turn(backend, monkeypatch, failure):
    from nanobot.agent.tools.filesystem_effect import FilesystemEffectCancelled

    tool = ApplyPatchTool(workspace=backend.workspace)
    target = backend.workspace / "new-parent" / "target.txt" if failure == "mkdir" else backend.workspace / "target.txt"
    if failure != "mkdir":
        target.write_text("x")
    original_write, original_mkdir = Path.write_text, Path.mkdir

    def fail(*args):
        if failure == "cancel":
            raise asyncio.CancelledError("cancelled after patch")
        raise OSError("tracking failed")

    def write(path, *args, **kwargs):
        result = original_write(path, *args, **kwargs)
        if path == target:
            raise OSError("write acknowledgement lost")
        return result

    def mkdir(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        if path == target.parent:
            raise OSError("mkdir acknowledgement lost")
        return result

    if failure in {"record_write", "cancel"}:
        monkeypatch.setattr(type(tool._file_states), "record_write", fail)
    elif failure == "mkdir":
        monkeypatch.setattr(Path, "mkdir", mkdir)
    else:
        monkeypatch.setattr(Path, "write_text", write)
        if failure == "rollback":
            monkeypatch.setattr(Path, "write_bytes", fail)
    arguments = {"edits": [{"path": str(target), "action": "add", "new_text": "next"}]}
    error, snapshot = await run_mutation(backend, tool, arguments)
    if failure == "cancel":
        assert isinstance(error, FilesystemEffectCancelled)
    else:
        assert "Reconciliation" in type(error).__name__
    result = snapshot.events[-1]
    assert result["role"] == "tool"
    assert result["_filesystem_effects"][0]["target"] == str(target)
    assert result["_filesystem_effects"][0]["operation"] == "apply_patch"
    assert snapshot.header["metadata"]["execution_recovery_required"] is True


@pytest.mark.parametrize("failure", ["touch", "replace_then_raise", "mkdir", "cancel"])
async def test_binary_image_publication_failure_fences_the_source(backend, monkeypatch, failure):
    import nanobot.agent.tools.filesystem as filesystem
    from nanobot.agent.tools.image_generation import ImageGenerationTool, ImageGenerationToolConfig

    writer = WriteFileTool(workspace=backend.workspace)
    tool = ImageGenerationTool(workspace=backend.workspace,
        config=ImageGenerationToolConfig(), write_tool=writer,
        media_root=backend.workspace / "media")
    client = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
        images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jC1cAAAAASUVORK5CYII="], raw={})))
    monkeypatch.setattr(tool, "_provider_client", lambda: client)
    target = backend.workspace / "generated_images" / "out.png"
    original_replace, original_mkdir = filesystem.os.replace, Path.mkdir

    def fail(*args):
        if failure == "cancel":
            raise asyncio.CancelledError("cancelled after image publication")
        raise OSError("tracking failed")

    def replace(source, destination, **kwargs):
        original_replace(source, destination, **kwargs)
        if destination == target:
            raise OSError("replace acknowledgement lost")

    def mkdir(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        if path == target.parent:
            raise OSError("mkdir acknowledgement lost")
        return result

    if failure in {"touch", "cancel"}:
        monkeypatch.setattr(writer, "_record_touch", fail)
    elif failure == "mkdir":
        monkeypatch.setattr(Path, "mkdir", mkdir)
    else:
        monkeypatch.setattr(filesystem.os, "replace", replace)
    error, snapshot = await run_mutation(backend, tool, {"prompt": "one pixel", "output_path": "out.png"})
    assert isinstance(error, asyncio.CancelledError) if failure == "cancel" else "Reconciliation" in type(error).__name__
    fact = snapshot.events[-1]["_filesystem_effects"][0]
    assert fact["operation"] == "generate_image" and fact["target"] == str(target)
    assert fact["reconciliation_required"] is True


async def test_failed_mandatory_result_write_leaves_intent_fenced_after_reopen(backend, monkeypatch):
    tool = WriteFileTool(workspace=backend.workspace)
    monkeypatch.setattr(tool, "_record_touch", lambda *_: (_ for _ in ()).throw(OSError("touch failed")))
    target = backend.workspace / "target.txt"
    error, snapshot = await run_mutation(backend, tool,
        {"path": str(target), "content": "once"}, reject_result=True)
    assert isinstance(error, OSError)
    assert target.read_text() == "once"
    assert snapshot.events[-1]["role"] == "assistant"
    reopened = JsonlSessionStore(backend.workspace, sessions_root=backend.sessions_root)
    async with acquire_execution(reopened, "test:effect", run_id="reopen") as owner:
        assert owner.recovery_required
        with pytest.raises(ExecutionOwnershipError):
            owner.validate_dispatch()


async def test_plugin_cannot_replay_concrete_effect_facts(backend, monkeypatch):
    tool = WriteFileTool(workspace=backend.workspace)
    monkeypatch.setattr(tool, "_record_touch", lambda *_: (_ for _ in ()).throw(OSError("touch failed")))
    result = await ToolRegistry.execute_prepared(tool, {"path": "actual.txt", "content": "actual"})

    class Plugin(WriteFileTool):
        async def execute(self, **kwargs):
            return ToolOutcome(content="plugin output", filesystem_effects=result.filesystem_effects)

    replayed = await ToolRegistry.execute_prepared(Plugin(workspace=backend.workspace), {})
    assert not replayed.filesystem_effects
    assert replayed.content == "plugin output"


async def test_context_free_registry_execution_keeps_mutation_evidence(backend, monkeypatch):
    tool = WriteFileTool(workspace=backend.workspace)
    monkeypatch.setattr(tool, "_record_touch", lambda *_: (_ for _ in ()).throw(OSError("touch failed")))
    tools = ToolRegistry()
    tools.register(tool)
    result = await tools.execute("write_file", {"path": "actual.txt", "content": "actual"})
    assert isinstance(result, ToolOutcome)
    assert result.filesystem_effects[0].reconciliation_required
    assert "Analyze the error" not in result.content


def test_historical_mutation_error_without_evidence_remains_unresolved():
    events = [
        {"role": "assistant", "tool_calls": [{"id": "call", "function": {"name": "write_file"}}],
         "_execution_ids": ["execution"]},
        {"role": "tool", "tool_call_id": "call", "name": "write_file", "content": "write failed",
         "_execution_id": "execution", "_dispatched": True, "_outcome_status": "error"},
    ]
    assert validate_execution_links(events) == {"execution"}
    events[-1]["_filesystem_effects"] = []
    # An explicit empty list is minted only after concrete pre-effect refusal.
    assert not validate_execution_links(events)
    json.dumps(events)


async def test_pre_effect_refusal_can_recover_without_a_false_mutation_fence(backend):
    tools = ToolRegistry()
    tools.register(WriteFileTool(workspace=backend.workspace, allowed_dir=backend.workspace))
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[LLMResponse(content="", tool_calls=[
        ToolCallRequest("refused", "write_file", {"path": "../outside", "content": "denied"}),
    ]), LLMResponse(content="The write was refused.")])
    async with acquire_execution(backend, "test:effect", run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        writer.append_message({"role": "user", "content": "try a write"})
        await AgentRunner().run(make_run_spec(provider, initial_messages=[
            {"role": "user", "content": "try a write"}], tools=tools, model="offline",
            max_iterations=2, max_tool_result_chars=1000, source_writer=writer,
            turn_context=TurnContext(execution_owner=owner), concurrent_tools=False))
        result = next(event for event in owner.read_snapshot().events if event["role"] == "tool")
        assert result["_filesystem_effects"] == []
        assert not owner.recovery_required
        assert not validate_execution_links(owner.read_snapshot().events)
        assert provider.chat_with_retry.await_count == 2
    assert not backend.source.load_source_snapshot("test:effect").header["metadata"].get(
        "execution_recovery_required")
