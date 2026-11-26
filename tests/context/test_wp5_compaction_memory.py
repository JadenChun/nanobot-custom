"""Public acceptance cases from the WP-5 compaction and memory contract."""

import asyncio
import json
from copy import deepcopy

import pytest

from kohaku.agent.loop import AgentLoop
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import TurnRequest
from kohaku.context.budget import TokenMeter
from kohaku.context.memory_pipeline import MemorySearchTool
from kohaku.context_repo.catalog import RepositoryCatalog, RepositoryEntry
from kohaku.context_repo.policy import ResourceAccessPolicy
from kohaku.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from kohaku.security.folder_grants import FolderGrantStore
from kohaku.session.manager import Session, SessionPolicy
from tests.agent.test_migration_engine import engine_config as engine_config
from tests.agent.test_migration_source_engine import upgrade

_HEADINGS = (
    "Active Objective & Constraints",
    "Files Inspected / Modified",
    "Key Decisions, Findings & Verified Facts",
    "Open Errors, Blockers & Unresolved Questions",
    "Next Concrete Steps & Artifact Handles",
)
_LESSON = "The auth callback uses a safe redirect after the state mismatch bug."
_SECRET = "fixture.secret.abc123"
_HANDBOOK = f"# Handbook\n\n## OAuth\nHANDBOOK_DETAIL_ONLY. {_LESSON}\n"
_INDEX = "<!-- kohaku-memory-summary: v1 -->\n\nOAuth → memory/MEMORY.md # OAuth.\n"


class OfflineMemoryProvider(LLMProvider):
    def __init__(self):
        super().__init__(provider_name="synthetic")
        self.calls = []

    def get_default_model(self):
        return "synthetic-model"

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append((deepcopy(messages), model, kwargs.get("reasoning_effort")))
        first = messages[0]["content"]
        if first.startswith("Summarize historical evidence"):
            return LLMResponse(content="\n\n".join(
                f"## {index}. {heading}\nHistorical fact {index}."
                for index, heading in enumerate(_HEADINGS, 1)
            ))
        if first.startswith("Extract reusable lessons"):
            return LLMResponse(content=("- " + _LESSON + "\n") * 100)
        if "## Pending rollout files" in str(messages):
            if any(message["role"] == "tool" for message in messages):
                return LLMResponse(content="Updated the OAuth memories.")
            return LLMResponse(content=None, tool_calls=[
                ToolCallRequest(id="index", name="write_file", arguments={
                    "path": "memory/memory_summary.md", "content": _INDEX + (" topic" * 1550),
                }),
                ToolCallRequest(id="handbook", name="write_file", arguments={
                    "path": "memory/MEMORY.md", "content": _HANDBOOK,
                }),
            ])
        return LLMResponse(content="Verified the previous finding.")


async def _turn(loop, key, text):
    result = await loop.execute_turn(TurnRequest(content=text, session_key=key))
    assert result.error is None, result.error
    return result.content


def _seed(loop, key):
    session = loop.sessions.get_or_create(key)
    session.add_message("user", f"{_LESSON} Authorization: Bearer {_SECRET}")
    session.add_message("assistant", "Confirmed the cause in auth.py.")
    session.add_message("assistant", "", tool_calls=[{
        "id": "audit-test", "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }])
    session.add_message("tool", "EXCLUDE_TOOL_LOG", tool_call_id="audit-test", name="read_file")
    session.add_message("user", "Keep the recent continuation.")
    session.add_message("assistant", "Ready to resume.")
    loop.sessions.save(session)
    return upgrade(loop, key)


async def _rollouts(workspace):
    state_path = workspace / "memory/state.json"
    async with asyncio.timeout(5):
        while not state_path.is_file():
            await asyncio.sleep(0.01)
    return json.loads(state_path.read_text(encoding="utf-8"))["pending_rollouts"]


@pytest.mark.parametrize("override", [None, "cheap-summary-model"])
async def test_canonical_compact_uses_structured_user_checkpoint(engine_config, override):
    engine_config.agents.defaults.compaction_model = override
    engine_config.agents.defaults.reasoning_effort = "high"
    provider = OfflineMemoryProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "cli:compact-wp5"
    try:
        backend = _seed(loop, key)
        before = backend.source.load_source_snapshot(key)
        assert await _turn(loop, key, "/compact") == "Session compacted."
        after = backend.source.load_source_snapshot(key)
        assert [event for event in after.events if not event.get("_command")] == before.events
        assert [event["role"] for event in after.events if event.get("_command")] == ["user"]
        assert after.checkpoint is not None
        await _turn(loop, key, "Resume the auth work.")
        requests = [call for call in provider.calls if call[0][0]["content"].startswith("Summarize")]
        assert [(model, effort) for _, model, effort in requests] == [
            (override or "synthetic-model", "none"),
        ]
        messages = next(messages for messages, _, _ in reversed(provider.calls)
                        if "Resume the auth work." in str(messages))
        checkpoint, = [row for row in messages if "<conversation_summary>" in row.get("content", "")]
        assert checkpoint["role"] == "user"
        block = checkpoint["content"].split("<conversation_summary>\n")[1].split("\n</conversation_summary>")[0]
        assert [line for line in block.splitlines() if line.startswith("## ")] == [
            f"## {index}. {heading}" for index, heading in enumerate(_HEADINGS, 1)
        ]
        assert "Keep the recent continuation." in str(messages)
        assert await _rollouts(engine_config.workspace_path)
    finally:
        await loop.aclose()


@pytest.mark.parametrize("trigger", ["/compact", "/new"])
async def test_extraction_redacts_and_processes_one_session_revision(engine_config, trigger):
    provider = OfflineMemoryProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "cli:extract-wp5"
    try:
        backend = _seed(loop, key)
        assert await _turn(loop, key, trigger)
        workspace = engine_config.workspace_path
        pending = await _rollouts(workspace)
        filename, = pending.values()
        summary = (workspace / "memory/rollout_summaries" / filename).read_text(encoding="utf-8")
        raw = (workspace / "memory/raw_memories.md").read_text(encoding="utf-8")
        captured, = [call for call in provider.calls if call[0][0]["content"].startswith("Extract")]
        assert captured[2] == "none"
        assert _LESSON in summary and _LESSON in raw and len(summary) <= 4000
        assert all(_SECRET not in text and "EXCLUDE_TOOL_LOG" not in text
                   for text in (str(captured), summary, raw))
        runtime = loop.compaction_runtime(loop.llm_runtime())
        if trigger == "/compact":
            session = backend.source.load_source_snapshot(key).to_session()
            rollout = loop.memory_pipeline.capture_rollout(session)
            assert rollout is not None
            await loop.memory_pipeline.extract_session_rollout(rollout, runtime)
        for private_session in (
            Session(key="cli:temporary", policy=SessionPolicy(persist=False)),
            Session(key="dream:internal"),
        ):
            private_session.add_message("user", _LESSON)
            loop.queue_memory_rollout(private_session, runtime)
        assert [call for call in provider.calls if call[0][0]["content"].startswith("Extract")] == [captured]
        assert await _rollouts(workspace) == pending
    finally:
        await loop.aclose()


async def test_manual_dream_builds_bounded_index_and_preserves_old_fallback(
    engine_config, monkeypatch,
):
    # The engine fixture forbids process execution; dulwich would otherwise shell out
    # to look for a (nonexistent) git hook while committing the Dream snapshot.
    monkeypatch.setattr("dulwich.hooks.ShellHook.execute", lambda self, *a, **k: None)
    workspace = engine_config.workspace_path
    provider = OfflineMemoryProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        handbook = workspace / "memory/MEMORY.md"
        handbook.write_text(_HANDBOOK, encoding="utf-8")
        await _turn(loop, "cli:legacy-prompt-wp5", "Use prior context.")
        assert "HANDBOOK_DETAIL_ONLY" in provider.calls[-1][0][0]["content"]
        _seed(loop, "cli:rollout-wp5")
        await _turn(loop, "cli:rollout-wp5", "/new")
        assert await _rollouts(workspace)
        assert await _turn(loop, "cli:dream-wp5", "/dream") == "Dreaming..."
        async def _final_dream_message():
            while True:
                message = await loop.bus.consume_outbound()
                if message.content.startswith("Dream"):
                    return message

        notification = await asyncio.wait_for(_final_dream_message(), timeout=6)
        assert "Dream consolidated 1 rollout(s)." in notification.content
        summary = (workspace / "memory/memory_summary.md").read_text(encoding="utf-8")
        assert summary.startswith("<!-- kohaku-memory-summary: v1 -->")
        assert TokenMeter.count_text(summary) <= 1500
        assert _LESSON in handbook.read_text(encoding="utf-8")
        assert not await _rollouts(workspace)
        await _turn(loop, "cli:v1-prompt-wp5", "Use earlier context.")
        system = provider.calls[-1][0][0]["content"]
        assert "OAuth → memory/MEMORY.md # OAuth." in system
        assert "HANDBOOK_DETAIL_ONLY" not in system
    finally:
        await loop.aclose()


async def test_memory_search_scopes_limits_and_revocation(engine_config, tmp_path):
    workspace = engine_config.workspace_path
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=OfflineMemoryProvider())
    try:
        handbook = workspace / "memory/MEMORY.md"
        handbook.write_text(_HANDBOOK, encoding="utf-8")
        rollouts = workspace / "memory/rollout_summaries"
        rollouts.mkdir()
        for number in range(24):
            (rollouts / f"rollout-{number}.md").write_text(_LESSON, encoding="utf-8")
        (rollouts / "irrelevant.md").write_text("Use a different approach.", encoding="utf-8")
        _seed(loop, "cli:search-wp5")
        linked = tmp_path / "knowledge"
        linked.mkdir()
        guide = linked / "guide.md"
        guide.write_text(_LESSON, encoding="utf-8")
        catalog = RepositoryCatalog.for_workspace(workspace, sessions_root=loop.sessions.sessions_dir)
        catalog.update(RepositoryEntry(name="guide", path=str(linked), kind="knowledge"))
        grants = FolderGrantStore(
            agent_home=workspace, instance_root=workspace.parent / "instance",
            path=workspace.parent / "instance" / "webui" / "folder-grants.json",
        )
        grants.upsert({"path": str(linked), "file_access": "read_only", "commands": False}, expected_revision=0)
        policy = ResourceAccessPolicy(workspace=workspace, restrict_to_workspace=True, folder_grants=grants.view)
        tool = MemorySearchTool(workspace, loop.sessions, policy)

        async def search(scope="all", **kwargs):
            return json.loads(await tool.execute("auth redirect bug", scope=scope, **kwargs))

        assert len(await search()) == 8
        for scope, source in (
            ("memory", str(handbook)), ("rollouts", str(rollouts / "rollout-0.md")),
            ("sessions", "cli:search-wp5"), ("knowledge", str(guide)),
        ):
            hits = await search(scope)
            assert hits and {row["scope"] for row in hits} == {scope}
            assert any(row["source"] == source for row in hits)
            assert all(len(row["snippet"]) <= 400 for row in hits)
        assert len(await search("rollouts", limit=-1)) == 1
        assert len(await search("rollouts", limit=99)) == 20
        grants.revoke(str(linked), expected_revision=1)
        assert await search("knowledge") == []
        assert all(row["source"] != str(guide) for row in await search())

        # Non-WebUI channels must not honor WebUI-only folder grants.
        grants.upsert({"path": str(linked), "file_access": "read_only", "commands": False}, expected_revision=2)
        non_webui_policy = ResourceAccessPolicy(
            workspace=workspace, restrict_to_workspace=True, folder_grants=grants.view,
            managed_webui_only=True, sandbox_default=True,
        )
        non_webui_tool = MemorySearchTool(workspace, loop.sessions, non_webui_policy)
        assert json.loads(await non_webui_tool.execute("auth redirect bug", scope="knowledge")) == []
    finally:
        await loop.aclose()


async def test_second_canonical_compact_and_idle_source_compaction(engine_config):
    from datetime import datetime, timedelta, timezone

    provider = OfflineMemoryProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "cli:idle-compact-wp5"
    try:
        backend = _seed(loop, key)
        assert await _turn(loop, key, "/compact") == "Session compacted."
        first_cp = backend.source.load_source_snapshot(key).checkpoint
        assert first_cp is not None
        # Second /compact advances to cover the remaining completed turn.
        assert await _turn(loop, key, "/compact") == "Session compacted."
        second_cp = backend.source.load_source_snapshot(key).checkpoint
        assert second_cp is not None and second_cp["boundary"] > first_cp["boundary"]

        # Exercise compact_idle_source directly with a future cutoff.
        key2 = "cli:idle-runner-wp5"
        _seed(loop, key2)
        cutoff = datetime.now(timezone.utc) + timedelta(minutes=5)
        assert await loop.compact_idle_source(key2, cutoff, min_messages=2) is True
    finally:
        await loop.aclose()
