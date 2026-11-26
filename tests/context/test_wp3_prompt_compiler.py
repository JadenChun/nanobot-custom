"""Acceptance cases from the isolated WP-3 requirements author."""

import json
import re
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mcp.types import Tool as MCPDefinition

from kohaku.agent.context import ContextBuilder
from kohaku.agent.loop import AgentLoop
from kohaku.agent.tools.filesystem import ReadFileTool
from kohaku.agent.tools.mcp import MCPToolWrapper
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import DeliveryTarget, TurnRequest, TurnSource
from kohaku.context.budget import TokenMeter
from kohaku.context.project_docs import discover_project_instructions
from kohaku.providers.anthropic_provider import AnthropicProvider
from kohaku.providers.base import LLMResponse, ToolCallRequest
from kohaku.providers.bedrock_provider import BedrockProvider
from kohaku.runtime_context import (
    RUNTIME_CONTEXT_INPUT_META,
    RuntimeContextBlock,
    webui_quote_runtime_context,
)
from kohaku.session.manager import Session
from kohaku.webui.session_access import session_mentions_runtime_context
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_engine import engine_config as engine_config
from tests.agent.test_migration_source_engine import upgrade


def _skill(workspace, name, body, *, always=False):
    path = workspace / "skills" / name
    path.mkdir(parents=True)
    path.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Fixture skill.\nalways: {str(always).lower()}\n---\n"
        + body,
        encoding="utf-8",
    )


def test_empty_workspace_prompt_budget_and_tiers(tmp_path):
    builder = ContextBuilder(tmp_path)
    empty = builder.compile_system_prompt()
    assert len(empty.full_system_text) <= 8_000
    assert empty.section_tokens["operating_principles"] <= 800
    (tmp_path / "AGENTS.md").write_text("Use the fixture workflow.", encoding="utf-8")
    compiled = builder.compile_system_prompt()
    assert len(compiled.system_blocks()) == 2
    assert "".join(block["text"] for block in compiled.system_blocks()) == compiled.full_system_text


def test_prompt_keeps_whole_policies_and_only_matching_repository_context(tmp_path):
    workspace, related, unrelated = (tmp_path / name for name in ("workspace", "related", "unrelated"))
    for path in (workspace, related, unrelated):
        (path / "memory").mkdir(parents=True)
    (workspace / "USER.md").write_text("User profile marker", encoding="utf-8")
    (workspace / "memory/MEMORY.md").write_text("Workspace memory marker", encoding="utf-8")
    (related / "CONTEXT.md").write_text("Selected repository marker", encoding="utf-8")
    (related / "memory/MEMORY.md").write_text("Repository memory marker", encoding="utf-8")
    (unrelated / "CONTEXT.md").write_text("Unrelated repository marker", encoding="utf-8")
    (related / "runs").mkdir()
    (related / "runs/current.json").write_text("Volatile run marker", encoding="utf-8")
    builder = ContextBuilder(workspace, context_paths=[related, unrelated])
    visible_tool = frozenset({"web_search"})
    first = builder.compile_system_prompt(workspace=related, tool_names=visible_tool).full_system_text
    for text in ("User profile marker", "Workspace memory marker", "Repository memory marker",
                 "Selected repository marker", "untrusted data", "sandbox and folder permissions"):
        assert text in first
    for text in ("Unrelated repository marker", "Volatile run marker", "`message`",
                 "`select_workspace`", "`repository_catalog`", "`web_fetch`"):
        assert text not in first
    private = builder.compile_system_prompt(
        workspace=related, tool_names=visible_tool, include_memory=False,
    ).full_system_text
    assert "Selected repository marker" in private
    assert all(text not in private for text in ("User profile marker", "Workspace memory marker",
                                                 "Repository memory marker"))


def test_compiler_caps_large_sections_and_mcp_instruction_total(tmp_path):
    long_text = "fixture content " * 4000
    (tmp_path / "SOUL.md").write_text(long_text, encoding="utf-8")
    (tmp_path / "USER.md").write_text(long_text, encoding="utf-8")
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory/MEMORY.md").write_text(long_text, encoding="utf-8")
    _skill(tmp_path, "always-testing", long_text, always=True)
    for index in range(120):
        _skill(tmp_path, f"search-fixture-{index}", "A normal on-demand skill.")
    tools = ToolRegistry()
    tools.register(ReadFileTool(workspace=tmp_path))
    for index in range(4):
        wrapper = MCPToolWrapper(
            SimpleNamespace(), f"server{index}",
            MCPDefinition(name="fixture", inputSchema={"type": "object"}),
        )
        wrapper.prompt_instructions = long_text
        tools.register(wrapper)
    compiled = ContextBuilder(tmp_path).compile_system_prompt(tools=tools)
    for name, cap in (("SOUL.md", 400), ("USER.md", 500), ("memory", 1500),
                      ("always_skills", 2000), ("skill_index", 1200)):
        assert compiled.section_tokens[name] <= cap
        assert f"[Truncated to fit {cap}-token budget" in compiled.full_system_text
    assert 0 < compiled.section_tokens["mcp_instructions"] <= 1500
    assert all(tokens <= 500 for name, tokens in compiled.section_tokens.items() if name.startswith("mcp:"))
    assert "use read_file on the source file" in compiled.full_system_text


def test_workspace_switch_refreshes_snapshot_and_prefers_memory_summary(tmp_path):
    home, first, second = (tmp_path / name for name in ("home", "first", "second"))
    for path in (home, first, second):
        path.mkdir()
    (home / "memory").mkdir()
    (home / "memory/MEMORY.md").write_text("Legacy memory marker", encoding="utf-8")
    (home / "memory/memory_summary.md").write_text("Summary v1 marker", encoding="utf-8")
    (first / "AGENTS.md").write_text("First project marker", encoding="utf-8")
    (second / "AGENTS.md").write_text("Second project marker", encoding="utf-8")

    builder = ContextBuilder(home)
    session = Session("cli:workspace")
    frozen = builder.compile_system_prompt(workspace=first, session=session)
    assert "Legacy memory marker" not in frozen.full_system_text
    assert "Summary v1 marker" in frozen.session_snapshot_text
    (home / "memory/memory_summary.md").write_text("Summary v2 marker", encoding="utf-8")
    assert builder.compile_system_prompt(workspace=first, session=session) == frozen
    switched = builder.compile_system_prompt(workspace=second, session=session)
    assert "Summary v2 marker" in switched.session_snapshot_text
    assert "Second project marker" in switched.session_snapshot_text
    assert "First project marker" not in switched.full_system_text


@pytest.mark.parametrize("git_root", [True, False])
def test_project_instruction_hierarchy_and_byte_limit(tmp_path, git_root):
    root = tmp_path
    if git_root:
        (root / ".git").mkdir()
    project = root / "a" / "b"
    project.mkdir(parents=True)
    (root / "AGENTS.md").write_text("root marker", encoding="utf-8")
    (root / "a/AGENTS.md").write_text("replaced marker", encoding="utf-8")
    (root / "a/AGENTS.override.md").write_text("override marker", encoding="utf-8")
    (project / "AGENTS.md").write_text("leaf marker café " * 400, encoding="utf-8")
    assembled = discover_project_instructions(root, project, max_bytes=200)
    assert len(assembled.encode("utf-8")) <= 200
    assert assembled.index("root marker") < assembled.index("override marker") < assembled.index("leaf marker")
    assert "a/AGENTS.override.md" in assembled and "a/b/AGENTS.md" in assembled
    assert "replaced marker" not in assembled


def test_project_instruction_validator_and_escaping_symlink(tmp_path):
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    (workspace / "a/b").mkdir(parents=True)
    outside.write_text("outside secret marker", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("root allowed", encoding="utf-8")
    (workspace / "a/AGENTS.md").write_text("policy denied marker", encoding="utf-8")
    try:
        (workspace / "a/b/AGENTS.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlink unavailable")

    def authorize(path):
        if path.parent == workspace / "a":
            raise PermissionError("denied")

    result = discover_project_instructions(workspace, workspace / "a/b", validator=authorize)
    assert "root allowed" in result
    assert "policy denied marker" not in result and "outside secret marker" not in result


@pytest.mark.parametrize("canonical", [False, True])
async def test_prompt_epoch_retention_and_public_history(engine_config, canonical):
    workspace = engine_config.workspace_path
    (workspace / "memory").mkdir()
    memory = workspace / "memory/MEMORY.md"
    instructions = workspace / "AGENTS.md"
    memory.write_text("memory V1 marker", encoding="utf-8")
    instructions.write_text("local rules V1 marker", encoding="utf-8")
    skill_body = "Complete explicit skill body. " * 680
    _skill(workspace, "fixture-skill", skill_body)
    quote = webui_quote_runtime_context({"_webui_quote": "A selected quote fixture."})
    mention = session_mentions_runtime_context([
        {"id": "demo", "name": "demo", "session_key": "cli:demo", "title": "A prior conversation"},
    ])
    local = RuntimeContextBlock("fixture_cli_app", "TURN-ONLY CAPABILITY FIXTURE")
    provider = ScriptedProvider([LLMResponse(content="done")] * 5)
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "cli:fixture"
    if canonical:
        upgrade(loop, key)
    try:
        first = await loop.execute_turn(TurnRequest(
            content="Inspect $fixture-skill", session_key=key,
            route=DeliveryTarget(channel="cli", chat_id="original-chat"),
            metadata={RUNTIME_CONTEXT_INPUT_META: [quote, mention, local]},
        ))
        assert first.error is None, first.error
        first_system = deepcopy(provider.calls[0][0][0])
        first_user = provider.calls[0][0][-1]["content"]
        assert first_user.count(skill_body.strip()) == 1
        assert "A selected quote fixture." in first_user and '"name":"demo"' in first_user
        assert "Current Time:" not in first_system["content"]
        assert re.search(r"Current Time: \d{4}-\d\d-\d\d \d\d:00", first_user)
        assert "Channel: cli" in first_user and "Chat ID: original-chat" in first_user

        memory.write_text("memory V2 marker", encoding="utf-8")
        instructions.write_text("local rules V2 marker", encoding="utf-8")
        result = await loop.execute_turn(TurnRequest(content="Next question", session_key=key))
        assert result.error is None, result.error
        second = provider.calls[1][0]
        assert json.dumps(first_system, ensure_ascii=False) == json.dumps(second[0], ensure_ascii=False)
        assert "memory V1 marker" in second[0]["content"] and "local rules V1 marker" in second[0]["content"]
        replay = next(row["content"] for row in second if row.get("role") == "user")
        assert "A selected quote fixture." in replay and "@demo" in replay
        assert replay.count("[Used skill: fixture-skill]") == 1
        assert "Complete explicit skill body." not in replay
        assert "TURN-ONLY CAPABILITY FIXTURE" not in replay and "original-chat" not in replay
        assert "Current Time:" not in replay
        visible = loop.sessions.get_or_create(key).get_history(include_runtime_context=False)
        assert visible[0]["content"] == "Inspect $fixture-skill"

        independent = await loop.execute_turn(TurnRequest(content="Independent", session_key="cli:new-fixture"))
        assert independent.error is None, independent.error
        assert "memory V2 marker" in provider.calls[2][0][0]["content"]
        assert "local rules V2 marker" in provider.calls[2][0][0]["content"]

        reset = await loop.execute_turn(TurnRequest(content="/new", session_key=key))
        assert reset.error is None, reset.error
        result = await loop.execute_turn(TurnRequest(content="Question after new", session_key=key))
        assert result.error is None, result.error
        assert "memory V2 marker" in provider.calls[-1][0][0]["content"]
        assert "local rules V2 marker" in provider.calls[-1][0][0]["content"]
    finally:
        await loop.aclose()


async def test_native_provider_payloads_keep_both_system_cache_seams(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Session instructions fixture.", encoding="utf-8")
    builder = ContextBuilder(tmp_path)
    compiled = builder.compile_system_prompt()
    messages = builder.build_messages([], "Hello")
    assert messages[0]["content"] == compiled.full_system_text
    bedrock_call = Mock(return_value={"output": {"message": {"content": [{"text": "ok"}]}}})
    bedrock = BedrockProvider(client=SimpleNamespace(converse=bedrock_call), region="fixture")
    await bedrock.chat(messages, model="amazon.nova-pro-v1:0")
    assert bedrock_call.call_args.kwargs["system"] == [
        {"text": compiled.system_blocks()[0]["text"]}, {"cachePoint": {"type": "default"}},
        {"text": compiled.system_blocks()[1]["text"]}, {"cachePoint": {"type": "default"}},
    ]

    anthropic = AnthropicProvider(api_key="fixture")
    anthropic_call = AsyncMock(return_value=SimpleNamespace(content=[], stop_reason="end_turn", usage=None))
    anthropic._client = SimpleNamespace(messages=SimpleNamespace(create=anthropic_call))
    await anthropic.chat(messages)
    assert anthropic_call.await_args.kwargs["system"] == [
        {**block, "cache_control": {"type": "ephemeral"}} for block in compiled.system_blocks()
    ]


async def test_goal_guidance_only_on_creation_turn_and_system_skill_is_literal(engine_config):
    _skill(engine_config.workspace_path, "fixture-skill", "ANNOUNCEMENT MUST NOT LOAD THIS")
    objective = "Bounded objective marker " * 42
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            "goal", "create_goal", {"objective": objective, "ui_summary": "fixture step"},
        )]),
        LLMResponse(content="Goal saved.", finish_reason="refusal"),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            "finish", "update_goal", {"action": "complete", "recap": "Verified the fixture."},
        )]),
        LLMResponse(content="Goal complete."),
        LLMResponse(content="Announcement read."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "cli:source"
    upgrade(loop, key)
    try:
        created = await loop.execute_turn(TurnRequest(content="/goal " + objective, session_key=key))
        assert created.error is None, created.error
        assert "Goal Runtime Guidance" in provider.calls[0][0][-1]["content"]
        following = await loop.execute_turn(TurnRequest(content="Keep going", session_key=key))
        assert following.error is None, following.error
        current_rows = provider.calls[2][0]
        assert "Goal Runtime Guidance" not in str(current_rows)
        content = current_rows[-1]["content"]
        active = content.split("Active Goal\n", 1)[1].split("\n[/Runtime Context]", 1)[0]
        assert "Goal ID:" in active and "Current step: fixture step" in active
        summary = next(line.removeprefix("Objective: ") for line in active.splitlines()
                       if line.startswith("Objective: "))
        assert summary.startswith("Bounded objective marker") and len(summary) <= 500

        done = await loop.execute_turn(TurnRequest(
            content="System announcement uses literal $fixture-skill", session_key=key,
            route=DeliveryTarget(channel="system", chat_id=key), source=TurnSource.SYSTEM_COMPAT,
            sender_id="subagent",
        ))
        assert done.error is None, done.error
        assert "ANNOUNCEMENT MUST NOT LOAD THIS" not in str(provider.calls[-1][0])
    finally:
        await loop.aclose()


def test_discover_project_instructions_tolerates_non_utf8_and_unreadable_files(tmp_path):
    from kohaku.context.project_docs import discover_project_instructions

    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    sub = root / "pkg"
    sub.mkdir()
    (root / "AGENTS.md").write_bytes(b"Valid root \xff\xfe bytes\n")
    broken = sub / "AGENTS.md"
    broken.symlink_to(sub / "missing-target.md")
    result = discover_project_instructions(tmp_path, sub)
    assert "Valid root" in result
