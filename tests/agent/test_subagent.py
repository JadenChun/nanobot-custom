"""Foreground delegation invariants and dormant SubagentManager adapters."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager, SubagentStatus
from nanobot.agent.tools.file_state import FileStates, bind_file_states, reset_file_states
from nanobot.agent.tools.filesystem import FileToolsConfig, ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig
from nanobot.llm_usage.context import llm_usage_source
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest
from nanobot.security.workspace_access import build_workspace_scope
from nanobot.utils.llm_runtime import LLMRuntime


def _runtime(provider: LLMProvider) -> LLMRuntime:
    provider.generation = GenerationSettings()
    return LLMRuntime.capture(provider, "test", context_window_tokens=128_000)


@pytest.mark.asyncio
async def test_retired_subagent_tool_loader_rejects_unsupported_scope(tmp_path):
    """The upstream background factory must not restore retired tools."""
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    try:
        with pytest.raises(ValueError, match="Unsupported runtime tool scope"):
            sm._build_tools()
    finally:
        await sm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["planner", "reviewer", "explorer", "worker"])
async def test_foreground_tools_isolate_file_read_state(tmp_path, role):
    """An independent delegated context must see contents, not a parent's read stub."""
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    manager = ForegroundAgentManager(provider=provider, workspace=tmp_path, runtime=_runtime(provider))
    def registry(call_id):
        if role == "worker":
            return manager._worker_tools(call_id, manager._normalize_scopes(tmp_path, ["note.txt"]))
        return manager._tools_for_role(role)

    parent_state = FileStates()
    parent_read = ReadFileTool(workspace=tmp_path, file_states=parent_state)
    token = bind_file_states(parent_state)
    try:
        assert (await parent_read.execute(path="note.txt")).startswith("1| hello")
        first = registry("first")
        second = registry("second")
        first_read = first.get("read_file")
        second_read = second.get("read_file")
        assert first_read is not second_read
        first_result, second_result = await asyncio.gather(
            first_read.execute(path="note.txt"), second_read.execute(path="note.txt"),
        )
        assert first_result.startswith("1| hello")
        assert second_result.startswith("1| hello")
        assert "File unchanged" not in second_result
        assert first_read._file_states is not parent_state
        assert first_read._file_states is not second_read._file_states
        assert "File unchanged" in await first_read.execute(path="note.txt")
        assert "File unchanged" in await parent_read.execute(path="note.txt")
        if role == "worker":
            assert first.get("write_file")._file_states is first_read._file_states
            assert first.get("edit_file")._file_states is first_read._file_states
    finally:
        reset_file_states(token)
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["planner", "reviewer", "explorer", "worker"])
async def test_foreground_runner_receives_contents_already_read_by_parent(tmp_path, role):
    (tmp_path / "note.txt").write_text("delegated evidence\n", encoding="utf-8")
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="reading", tool_calls=[ToolCallRequest(
            id="read", name="read_file", arguments={"path": "note.txt"},
        )]),
        LLMResponse(content="inspection finished"),
    ])
    manager = ForegroundAgentManager(provider=provider, workspace=tmp_path, runtime=_runtime(provider))
    parent_state = FileStates()
    parent_read = ReadFileTool(workspace=tmp_path, file_states=parent_state)
    token = bind_file_states(parent_state)
    try:
        assert "delegated evidence" in await parent_read.execute(path="note.txt")
        tools = (manager._worker_tools("worker", manager._normalize_scopes(tmp_path, ["note.txt"]))
                 if role == "worker" else manager._tools_for_role(role))
        result = await manager._run_role(
            role=role, prompt="Inspect files", assignment="Read note.txt",
            tools=tools, max_iterations=3,
        )
        assert result.stop_reason == "completed"
        assert provider.chat_with_retry.await_count == 2
        messages = provider.chat_with_retry.await_args_list[1].kwargs["messages"]
        tool_rows = [row for row in messages if row["role"] == "tool"]
        assert len(tool_rows) == 1
        assert "delegated evidence" in str(tool_rows[0]["content"])
        assert "File unchanged" not in str(tool_rows[0]["content"])
        assert parent_read._file_states is parent_state
        assert "File unchanged" in await parent_read.execute(path="note.txt")
    finally:
        reset_file_states(token)
        await manager.close()


def test_retired_subagent_preserves_file_toggle_without_loading_tools(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
        tools_config=ToolsConfig(file=FileToolsConfig(enable=False)),
    )

    assert sm._subagent_tools_config().file.enable is False
    with pytest.raises(ValueError, match="Unsupported runtime tool scope"):
        sm._build_tools()


def test_subagent_prompt_keeps_agent_paths_for_selected_project(tmp_path):
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    global_skill = agent_workspace / "skills" / "global-custom" / "SKILL.md"
    project_skill = project / "skills" / "project-custom" / "SKILL.md"
    global_skill.parent.mkdir(parents=True)
    project_skill.parent.mkdir(parents=True)
    global_skill.write_text("---\ndescription: global skill\n---\nGlobal", encoding="utf-8")
    project_skill.write_text("---\ndescription: project skill\n---\nProject", encoding="utf-8")
    manager = SubagentManager(
        workspace=agent_workspace,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    prompt = manager._build_subagent_prompt(workspace=project)

    assert "one root and relative SKILL.md paths" in prompt
    assert "Join them when using `read_file`" in prompt
    assert str(project.resolve()) not in prompt
    assert f"Nanobot's agent workspace: {agent_workspace.resolve()}" in prompt
    assert f"History log: {agent_workspace.resolve() / 'memory' / 'history.jsonl'}" in prompt
    assert "global-custom" in prompt
    assert "project-custom" not in prompt


def test_subagent_prompt_uses_relative_paths_in_agent_workspace(tmp_path):
    skill = tmp_path / "skills" / "custom" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\ndescription: custom skill\n---\nCustom", encoding="utf-8")
    manager = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )

    prompt = manager._build_subagent_prompt()

    assert str(tmp_path.resolve()) not in prompt
    assert "History log: memory/history.jsonl" in prompt
    assert "### Workspace skills (`skills`)" in prompt


@pytest.mark.asyncio
async def test_subagent_keeps_project_runtime_scope_with_agent_owned_tools(tmp_path):
    agent_workspace = tmp_path / "agent"
    project = tmp_path / "project"
    agent_workspace.mkdir()
    project.mkdir()
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    manager = SubagentManager(
        workspace=agent_workspace,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    manager.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    manager._announce_result = AsyncMock()
    # Adapter forwarding only: production background tool loading stays retired.
    tools = ToolRegistry()
    tools.register(ReadFileTool(workspace=agent_workspace.resolve()))
    manager._build_tools = MagicMock(return_value=tools)
    status = SubagentStatus(
        task_id="t1",
        label="label",
        task_description="task",
        started_at=0.0,
    )

    await manager._run_subagent(
        "t1",
        "task",
        "label",
        {"channel": "websocket", "chat_id": "direct"},
        status,
        _runtime(provider),
        workspace_scope=build_workspace_scope(project, "restricted"),
    )

    spec = manager.runner.run.call_args.args[0]
    assert spec.workspace == project
    assert spec.tools.get("read_file")._workspace == agent_workspace.resolve()
    assert spec.tools is tools
    await manager.close()


@pytest.mark.asyncio
async def test_foreground_tool_error_stops_without_extra_model_call(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="reading",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="read_file",
                    arguments={"path": "missing.txt"},
                )
            ],
        ),
        LLMResponse(content="recovered without restarting", tool_calls=[]),
    ])
    manager = ForegroundAgentManager(provider=provider, workspace=tmp_path, runtime=_runtime(provider))
    try:
        result = await manager._run_role(
            role="planner", prompt="Inspect files", assignment="read a missing file",
            tools=manager._tools_for_role("planner"), max_iterations=3,
        )
        assert result.stop_reason == "tool_error"
        assert provider.chat_with_retry.await_count == 1
        assert any(event["status"] == "error" for event in result.tool_events)
        assert not (tmp_path / "missing.txt").exists()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_spawned_subagent_inherits_llm_usage_source(tmp_path):
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "test"
    sm = SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )
    sm.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="ok", messages=[], stop_reason="completed")
    )
    sm._announce_result = AsyncMock()
    # Preserve legacy usage-source forwarding without admitting background tools.
    sm._build_tools = MagicMock(side_effect=lambda **_kwargs: ToolRegistry())
    sm._build_subagent_prompt = MagicMock(return_value="isolated usage adapter test")

    with llm_usage_source("cron"):
        await sm.spawn(
            "automation task",
            session_key="websocket:bound-automation",
            runtime=_runtime(provider),
        )
    tasks = list(sm._running_tasks.values())
    await asyncio.gather(*tasks)

    spec = sm.runner.run.call_args.args[0]
    assert spec.llm_usage_source == "cron"
    await sm.close()
