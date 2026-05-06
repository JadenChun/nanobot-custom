"""Tests for the generation-review validation loop in SubagentManager."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.subagent import SubagentManager
from nanobot.agent.write_guard import WriteScope


def _make_response(content: str | None = None, tool_calls: list | None = None):
    """Create a mock LLM response."""
    resp = MagicMock()
    resp.content = content
    resp.has_tool_calls = bool(tool_calls)
    resp.tool_calls = tool_calls or []
    resp.reasoning_content = None
    resp.thinking_blocks = None
    resp.finish_reason = "stop"
    return resp


def _make_manager(**overrides) -> SubagentManager:
    """Create a SubagentManager with mocked dependencies."""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock()

    bus = MagicMock()
    bus.publish_inbound = AsyncMock()

    workspace = Path("/tmp/test-workspace")

    with patch("nanobot.agent.subagent.ReadFileTool"), \
         patch("nanobot.agent.subagent.WriteFileTool"), \
         patch("nanobot.agent.subagent.EditFileTool"), \
         patch("nanobot.agent.subagent.ListDirTool"), \
         patch("nanobot.agent.subagent.ExecTool"), \
         patch("nanobot.agent.subagent.WebSearchTool"), \
         patch("nanobot.agent.subagent.WebFetchTool"):
        mgr = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            **overrides,
        )
    return mgr


class TestParseReview:
    """Tests for _parse_review static method."""

    def test_valid_approved(self):
        result = SubagentManager._parse_review(
            'Some analysis...\n\n---REVIEW---\n'
            '{"approved": true, "confidence": 92, "issues": [], "feedback": "Looks great."}\n'
            '---END---'
        )
        assert result["approved"] is True
        assert result["confidence"] == 92
        assert result["issues"] == []
        assert result["feedback"] == "Looks great."

    def test_valid_rejected(self):
        result = SubagentManager._parse_review(
            'I found problems.\n\n---REVIEW---\n'
            '{"approved": false, "confidence": 35, "issues": ["Missing FAQ section", "No CTA"], '
            '"feedback": "Add an FAQ section and include a call-to-action."}\n'
            '---END---'
        )
        assert result["approved"] is False
        assert result["confidence"] == 35
        assert len(result["issues"]) == 2
        assert "FAQ" in result["feedback"]

    def test_malformed_json_falls_back_to_rejection(self):
        result = SubagentManager._parse_review(
            '---REVIEW---\nnot valid json\n---END---'
        )
        assert result["approved"] is False
        assert result["confidence"] == 0

    def test_missing_block_falls_back_to_rejection(self):
        result = SubagentManager._parse_review(
            "The output looks good but I forgot the format."
        )
        assert result["approved"] is False
        assert result["confidence"] == 0

    def test_empty_input(self):
        result = SubagentManager._parse_review("")
        assert result["approved"] is False
        assert result["confidence"] == 0


class TestRunAgentLoop:
    """Tests for the shared _run_agent_loop method."""

    @pytest.mark.asyncio
    async def test_returns_content_when_no_tool_calls(self):
        mgr = _make_manager()
        mgr.provider.chat_with_retry.return_value = _make_response(content="Done!")

        with patch("nanobot.agent.subagent.ReadFileTool"), \
             patch("nanobot.agent.subagent.ListDirTool"), \
             patch("nanobot.agent.subagent.ExecTool"), \
             patch("nanobot.agent.subagent.WebSearchTool"), \
             patch("nanobot.agent.subagent.WebFetchTool"):
            tools = mgr._build_review_tools()

        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "do something"},
        ]

        result, msgs = await mgr._run_agent_loop("t1", messages, tools, max_iterations=5)
        assert result == "Done!"
        mgr.provider.chat_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_respects_max_iterations(self):
        mgr = _make_manager()

        # Create a tool call that keeps looping
        tc = MagicMock()
        tc.name = "read_file"
        tc.arguments = {"path": "/tmp/test"}
        tc.id = "tc1"
        tc.to_openai_tool_call.return_value = {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}

        mgr.provider.chat_with_retry.return_value = _make_response(content="thinking", tool_calls=[tc])

        with patch("nanobot.agent.subagent.ReadFileTool"), \
             patch("nanobot.agent.subagent.ListDirTool"), \
             patch("nanobot.agent.subagent.ExecTool"), \
             patch("nanobot.agent.subagent.WebSearchTool"), \
             patch("nanobot.agent.subagent.WebFetchTool"):
            tools = mgr._build_review_tools()
            # Mock tool execution
            tools.execute = AsyncMock(return_value="file content")

        messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "go"}]
        result, _ = await mgr._run_agent_loop("t1", messages, tools, max_iterations=3)
        assert result is None  # Never got a non-tool-call response
        assert mgr.provider.chat_with_retry.call_count == 3


class TestReviewLoop:
    """Tests for the generation-review loop orchestration."""

    @pytest.mark.asyncio
    async def test_approved_on_first_round(self):
        mgr = _make_manager()

        # Review agent approves immediately
        review_response = _make_response(
            content=(
                'Looks good!\n\n---REVIEW---\n'
                '{"approved": true, "confidence": 90, "issues": [], "feedback": "All good."}\n'
                '---END---'
            )
        )
        mgr.provider.chat_with_retry.return_value = review_response

        with patch("nanobot.agent.subagent.ReadFileTool"), \
             patch("nanobot.agent.subagent.ListDirTool"), \
             patch("nanobot.agent.subagent.ExecTool"), \
             patch("nanobot.agent.subagent.WebSearchTool"), \
             patch("nanobot.agent.subagent.WebFetchTool"):
            gen_tools = mgr._build_generation_tools()

        gen_messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "write article"},
        ]

        with patch.object(mgr, "_build_review_prompt", return_value="review prompt"):
            result, meta = await mgr._run_review_loop(
                task_id="t1",
                goal="Write a good article",
                generation_result="Here is the article...",
                generation_messages=gen_messages,
                generation_tools=gen_tools,
            )

        assert result == "Here is the article..."
        assert meta["approved"] is True
        assert meta["confidence"] == 90

    @pytest.mark.asyncio
    async def test_rejected_then_approved_on_round_2(self):
        mgr = _make_manager()

        # Round 1: review rejects, Round 2 generation revises, Round 2 review approves
        reject_response = _make_response(
            content=(
                'Issues found.\n\n---REVIEW---\n'
                '{"approved": false, "confidence": 40, "issues": ["Missing FAQ"], '
                '"feedback": "Add FAQ section."}\n'
                '---END---'
            )
        )
        revised_response = _make_response(content="Here is the revised article with FAQ...")
        approve_response = _make_response(
            content=(
                'Much better.\n\n---REVIEW---\n'
                '{"approved": true, "confidence": 88, "issues": [], "feedback": "Good now."}\n'
                '---END---'
            )
        )

        mgr.provider.chat_with_retry.side_effect = [
            reject_response,   # review round 1
            revised_response,  # generation revision
            approve_response,  # review round 2
        ]

        with patch("nanobot.agent.subagent.ReadFileTool"), \
             patch("nanobot.agent.subagent.ListDirTool"), \
             patch("nanobot.agent.subagent.ExecTool"), \
             patch("nanobot.agent.subagent.WebSearchTool"), \
             patch("nanobot.agent.subagent.WebFetchTool"):
            gen_tools = mgr._build_generation_tools()

        gen_messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "write article"},
        ]

        with patch.object(mgr, "_build_review_prompt", return_value="review prompt"):
            result, meta = await mgr._run_review_loop(
                task_id="t1",
                goal="Write article with FAQ",
                generation_result="Initial article without FAQ...",
                generation_messages=gen_messages,
                generation_tools=gen_tools,
            )

        assert "revised" in result.lower() or "FAQ" in result
        assert meta["approved"] is True

    @pytest.mark.asyncio
    async def test_max_rounds_returns_best_effort(self):
        mgr = _make_manager()

        # All rounds reject
        reject_low = _make_response(
            content=(
                '---REVIEW---\n'
                '{"approved": false, "confidence": 30, "issues": ["bad"], "feedback": "Fix it."}\n'
                '---END---'
            )
        )
        reject_medium = _make_response(
            content=(
                '---REVIEW---\n'
                '{"approved": false, "confidence": 55, "issues": ["meh"], "feedback": "Still not great."}\n'
                '---END---'
            )
        )
        reject_low2 = _make_response(
            content=(
                '---REVIEW---\n'
                '{"approved": false, "confidence": 45, "issues": ["ok-ish"], "feedback": "Close but no."}\n'
                '---END---'
            )
        )
        gen_response = _make_response(content="Revised output")

        mgr.provider.chat_with_retry.side_effect = [
            reject_low,      # review round 1 (confidence 30)
            gen_response,     # generation revision
            reject_medium,    # review round 2 (confidence 55)
            gen_response,     # generation revision
            reject_low2,      # review round 3 (confidence 45) — max reached
        ]

        with patch("nanobot.agent.subagent.ReadFileTool"), \
             patch("nanobot.agent.subagent.ListDirTool"), \
             patch("nanobot.agent.subagent.ExecTool"), \
             patch("nanobot.agent.subagent.WebSearchTool"), \
             patch("nanobot.agent.subagent.WebFetchTool"):
            gen_tools = mgr._build_generation_tools()

        gen_messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "write something"},
        ]

        with patch.object(mgr, "_build_review_prompt", return_value="review prompt"):
            result, meta = await mgr._run_review_loop(
                task_id="t1",
                goal="Write something good",
                generation_result="Initial output",
                generation_messages=gen_messages,
                generation_tools=gen_tools,
                max_rounds=3,
            )

        # Should return the result from the round with highest confidence (55 at round 2)
        assert meta["approved"] is False


class TestSpawnWithReview:
    """Tests for spawn() with review=True integration."""

    @pytest.mark.asyncio
    async def test_spawn_without_review_default(self):
        mgr = _make_manager()

        # Generation agent responds directly
        mgr.provider.chat_with_retry.return_value = _make_response(content="Task done.")

        with patch.object(mgr, "_build_subagent_prompt", return_value="subagent prompt"), \
             patch.object(mgr, "_build_generation_tools") as mock_tools:
            mock_tools.return_value = MagicMock()
            mock_tools.return_value.get_definitions.return_value = []
            mock_tools.return_value.execute = AsyncMock(return_value="ok")

            result = await mgr.spawn(task="simple task", origin_channel="cli", origin_chat_id="direct")

        assert "started" in result
        # Wait for background task
        await asyncio.sleep(0.1)
        # Successful background tasks stay silent unless notify=true.
        mgr.bus.publish_inbound.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawn_with_review_passes_params(self):
        mgr = _make_manager()

        with patch.object(mgr, "_run_subagent", new_callable=AsyncMock) as mock_run:
            result = await mgr.spawn(
                task="write article",
                goal="SEO-optimized article about fitness",
                review=True,
            )

        assert "started" in result
        # Wait for background task to start
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_spawn_no_final_response_reports_error(self):
        mgr = _make_manager()
        mgr.provider.chat_with_retry.return_value = _make_response(content=None, tool_calls=None)

        with patch.object(mgr, "_build_subagent_prompt", return_value="subagent prompt"), \
             patch.object(mgr, "_build_generation_tools") as mock_tools:
            mock_tools.return_value = MagicMock()
            mock_tools.return_value.get_definitions.return_value = []
            mock_tools.return_value.execute = AsyncMock(return_value="ok")

            result = await mgr.spawn(task="silent task", origin_channel="cli", origin_chat_id="direct")

        assert "started" in result
        await asyncio.sleep(0.1)
        # Generic failures remain silent unless notify=true or a forced-failure rule applies.
        mgr.bus.publish_inbound.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawn_rejects_overlapping_write_scope(self):
        mgr = _make_manager()
        block = asyncio.Event()

        async def fake_run(_spec):
            await block.wait()
            return SimpleNamespace(final_content="done", messages=[], stop_reason="stop")

        mgr.runner.run = AsyncMock(side_effect=fake_run)

        with patch.object(mgr, "_build_subagent_prompt", return_value="subagent prompt"), \
             patch.object(mgr, "_build_generation_tools") as mock_tools:
            mock_tools.return_value = MagicMock()
            first = await mgr.spawn(
                task="write draft",
                write_scope=["outputs/seo/"],
                origin_channel="cli",
                origin_chat_id="direct",
                session_key="cli:test",
            )
            second = await mgr.spawn(
                task="write another draft",
                write_scope=["outputs/seo/article.md"],
                origin_channel="cli",
                origin_chat_id="direct",
                session_key="cli:test",
            )

            assert "started" in first
            assert second.startswith("Error:")
            assert "overlaps" in second

            block.set()
            await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_write_conflict_forces_failure_notification_even_without_notify(self):
        mgr = _make_manager()
        mgr._last_run_result = SimpleNamespace(
            stop_reason="tool_error",
            tool_events=[{
                "name": "edit_file",
                "status": "error",
                "detail": "Error: File is currently locked for editing by main:cli:test",
            }],
        )

        with patch.object(mgr, "_build_subagent_prompt", return_value="subagent prompt"), \
             patch.object(mgr, "_build_generation_tools") as mock_tools, \
             patch.object(mgr, "_run_agent_loop", new=AsyncMock(return_value=(None, []))):
            mock_tools.return_value = MagicMock()
            await mgr._run_subagent(
                "sub-1",
                "edit the file",
                "edit the file",
                {"channel": "test", "chat_id": "c1"},
                notify=False,
                write_scope=(WriteScope.from_raw(Path("/tmp/test-workspace"), "drafts/"),),
                session_key="test:c1",
            )

        mgr.bus.publish_inbound.assert_called_once()


class TestBuildReviewPrompt:
    """Tests for the review agent system prompt."""

    def test_contains_goal(self):
        mgr = _make_manager()
        with patch("nanobot.agent.context.ContextBuilder._build_runtime_context", return_value="Time: now"):
            prompt = mgr._build_review_prompt("Write a blog post about AI")
        assert "Write a blog post about AI" in prompt

    def test_contains_skeptical_instructions(self):
        mgr = _make_manager()
        with patch("nanobot.agent.context.ContextBuilder._build_runtime_context", return_value="Time: now"):
            prompt = mgr._build_review_prompt("Test goal")
        assert "skeptical" in prompt.lower()
        assert "MUST NOT modify" in prompt
        assert "---REVIEW---" in prompt


@pytest.mark.asyncio
async def test_spawn_pipeline_is_explicitly_rejected():
    mgr = _make_manager()

    result = await mgr.spawn_pipeline(tasks=[{"task": "do work"}], session_key="cli:test")

    assert result.startswith("Error:")
    assert "write_scope safety model" in result

    def test_contains_workspace(self):
        mgr = _make_manager()
        with patch("nanobot.agent.context.ContextBuilder._build_runtime_context", return_value="Time: now"):
            prompt = mgr._build_review_prompt("Test goal")
        assert "test-workspace" in prompt


class TestBuildReviewTools:
    """Tests for the review agent tool set."""

    def test_has_read_only_tools(self):
        mgr = _make_manager()
        with patch("nanobot.agent.subagent.ReadFileTool") as MockRead, \
             patch("nanobot.agent.subagent.ListDirTool") as MockList, \
             patch("nanobot.agent.subagent.ExecTool") as MockExec, \
             patch("nanobot.agent.subagent.WebSearchTool") as MockSearch, \
             patch("nanobot.agent.subagent.WebFetchTool") as MockFetch:
            tools = mgr._build_review_tools()

        # Should have registered 5 tools
        MockRead.assert_called_once()
        MockList.assert_called_once()
        MockExec.assert_called_once()
        MockSearch.assert_called_once()
        MockFetch.assert_called_once()

    def test_no_write_tools(self):
        mgr = _make_manager()
        with patch("nanobot.agent.subagent.ReadFileTool"), \
             patch("nanobot.agent.subagent.ListDirTool"), \
             patch("nanobot.agent.subagent.ExecTool"), \
             patch("nanobot.agent.subagent.WebSearchTool"), \
             patch("nanobot.agent.subagent.WebFetchTool"), \
             patch("nanobot.agent.subagent.WriteFileTool") as MockWrite, \
             patch("nanobot.agent.subagent.EditFileTool") as MockEdit:
            tools = mgr._build_review_tools()

        # WriteFileTool and EditFileTool should NOT be called by _build_review_tools
        MockWrite.assert_not_called()
        MockEdit.assert_not_called()
