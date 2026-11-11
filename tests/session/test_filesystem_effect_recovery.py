"""Real filesystem failures retain effect evidence and fence the source owner."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.filesystem import EditFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.config.schema import ToolsConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.execution_owner import acquire_execution
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
            assert type(caught).__name__ == "FilesystemEffectReconciliationRequired"
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
