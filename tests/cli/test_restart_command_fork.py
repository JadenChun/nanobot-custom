"""Tests for /restart slash command."""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.context import TranscriptInput
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.providers.base import GenerationSettings, LLMResponse, LLMUsage
from tests.agent import p6_admitted_fixtures
from tests.agent.test_loop_runner_integration import _execute_inbound, _run_context

admitted_async = p6_admitted_fixtures.admitted_async
_run_agent = AgentLoop.run


async def _make_loop(tmp_path, admitted_async):
    """Use an owned, effect-fenced runtime graph with legacy history fixtures."""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False

    def build(inputs):
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test-model", context_window_tokens=200_000,
            session_manager=inputs.sessions,
            tool_registry=inputs.tools, cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery, enable_legacy_auto_compact=False,
        )

    loop = await admitted_async(tmp_path, build, keys=("telegram:c1", "cli:test"))
    return loop, loop.bus


@pytest.fixture(autouse=True)
def _restart_environment():
    # Restart notices are process-global but must not leak into another test.
    with patch.dict(os.environ, {}, clear=False):
        yield


@pytest.fixture
def restart_tasks():
    scheduled = []

    async def fast_sleep(_delay):
        return None

    def capture(coro):
        task = asyncio.create_task(coro)
        scheduled.append(task)
        return task

    with patch("nanobot.command.builtin.asyncio", SimpleNamespace(
        sleep=fast_sleep, create_task=capture,
    )):
        yield scheduled
    assert all(task.done() for task in scheduled)


class TestRestartCommand:

    @pytest.mark.asyncio
    async def test_restart_sends_message_and_calls_execv(self, tmp_path, admitted_async, restart_tasks):
        from nanobot.command.builtin import cmd_restart
        from nanobot.command.router import CommandContext

        loop, bus = await _make_loop(tmp_path, admitted_async)
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/restart")
        loop.restart_mode = "exec"
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        with patch("nanobot.command.builtin.os.execv") as mock_execv:
            out = await cmd_restart(ctx)
            assert "Restarting" in out.content
            assert restart_tasks
            await asyncio.wait_for(asyncio.gather(*restart_tasks), timeout=1)
            mock_execv.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_intercepted_in_run_loop(self, tmp_path, admitted_async, restart_tasks):
        """Verify /restart is handled at the run-loop level, not inside _dispatch."""
        loop, bus = await _make_loop(tmp_path, admitted_async)
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/restart")

        loop.restart_mode = "exec"
        with patch.object(loop, "_dispatch", new_callable=AsyncMock) as mock_dispatch, \
             patch("nanobot.command.builtin.os.execv"):
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(_run_agent(loop))
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_dispatch.assert_not_called()
            assert "Restarting" in out.content
            assert restart_tasks
            await asyncio.wait_for(asyncio.gather(*restart_tasks), timeout=1)

    @pytest.mark.asyncio
    async def test_status_intercepted_in_run_loop(self, tmp_path, admitted_async):
        """Verify /status is handled at the run-loop level for immediate replies."""
        loop, bus = await _make_loop(tmp_path, admitted_async)
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")

        with patch.object(loop, "_dispatch", new_callable=AsyncMock) as mock_dispatch:
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(_run_agent(loop))
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_dispatch.assert_not_called()
            assert "nanobot" in out.content.lower() or "Model" in out.content

    @pytest.mark.asyncio
    async def test_run_propagates_external_cancellation(self, tmp_path, admitted_async):
        """External task cancellation should not be swallowed by the inbound wait loop."""
        loop, _bus = await _make_loop(tmp_path, admitted_async)

        run_task = asyncio.create_task(_run_agent(loop))
        await asyncio.sleep(0.1)
        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_help_includes_restart(self, tmp_path, admitted_async):
        loop, bus = await _make_loop(tmp_path, admitted_async)
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/help")

        response = await _execute_inbound(loop, msg)

        assert response is not None
        assert "/restart" in response.content
        assert "/status" in response.content
        assert response.metadata == {"render_as": "text"}

    @pytest.mark.asyncio
    async def test_status_reports_runtime_info(self, tmp_path, admitted_async):
        loop, _bus = await _make_loop(tmp_path, admitted_async)
        session = loop.sessions.get_or_create("telegram:c1")
        session.messages = [{"role": "user", "content": "hello"}] * 3
        loop._start_time = time.time() - 125
        session.metadata["_last_usage"] = LLMUsage.reported(input_tokens=0, output_tokens=0).to_dict()
        loop.consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(20500, "tiktoken")
        )

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")

        response = await _execute_inbound(loop, msg)

        assert response is not None
        assert "Model: test-model" in response.content
        assert "Tokens: 0 in / 0 out" in response.content
        assert "Context: 20k/200k (10% of input budget)" in response.content
        assert "Session: 3 messages" in response.content
        assert "Uptime: 2m 5s" in response.content
        assert response.metadata == {"render_as": "text"}

    @pytest.mark.asyncio
    async def test_run_agent_loop_estimates_usage_when_provider_omits_it(
        self, tmp_path, admitted_async, monkeypatch,
    ):
        # Missing usage is estimated per result, never stale global usage or fabricated zero.
        loop, _bus = await _make_loop(tmp_path, admitted_async)
        monkeypatch.setattr(
            "nanobot.agent.runner.estimate_prompt_tokens_chain",
            lambda *_args, **_kwargs: (123, "test"),
        )
        monkeypatch.setattr("nanobot.agent.runner.estimate_message_tokens", lambda _message: 7)
        loop.provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(content="first", usage=LLMUsage.reported(input_tokens=9, output_tokens=4)),
            LLMResponse(content="second", usage=None),
        ])

        first = await loop._run_agent_loop(
            TranscriptInput(history=[], current_message=None),
            runtime=loop.llm_runtime(), run_context=_run_context(loop),
        )
        second = await loop._run_agent_loop(
            TranscriptInput(history=[], current_message=None),
            runtime=loop.llm_runtime(), run_context=_run_context(loop),
        )

        assert first.usage == LLMUsage.reported(input_tokens=9, output_tokens=4)
        assert second.usage == LLMUsage.estimated(input_tokens=123, output_tokens=7)

    @pytest.mark.asyncio
    async def test_status_falls_back_to_last_usage_when_context_estimate_missing(self, tmp_path, admitted_async):
        loop, _bus = await _make_loop(tmp_path, admitted_async)
        session = loop.sessions.get_or_create("telegram:c1")
        session.messages = [{"role": "user", "content": "hello"}]
        session.metadata["_last_usage"] = LLMUsage.reported(input_tokens=1200, output_tokens=34).to_dict()
        loop.consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(0, "none")
        )

        response = await _execute_inbound(loop,
            InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")
        )

        assert response is not None
        assert "Tokens: 1200 in / 34 out" in response.content
        assert "Context: 1k/200k (0% of input budget)" in response.content

    @pytest.mark.asyncio
    async def test_process_direct_preserves_render_metadata(self, tmp_path, admitted_async):
        loop, _bus = await _make_loop(tmp_path, admitted_async)
        session = loop.sessions.get_or_create("telegram:c1")
        session.messages = []
        response = await loop.process_direct("/status", session_key="cli:test")

        assert response is not None
        assert response.metadata == {"render_as": "text"}

    @pytest.mark.asyncio
    async def test_status_usage_is_isolated_between_sessions(self, tmp_path, admitted_async):
        loop, _bus = await _make_loop(tmp_path, admitted_async)
        session = loop.sessions.get_or_create("telegram:c1")
        session.metadata["_last_usage"] = LLMUsage.reported(
            input_tokens=1200, output_tokens=34,
        ).to_dict()
        loop.consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(0, "none")
        )

        other = await loop.process_direct("/status", session_key="cli:test")
        original = await loop.process_direct("/status", session_key="telegram:c1")

        assert other is not None and original is not None
        assert "Tokens: 0 in / 0 out" in other.content
        assert "Tokens: 1200 in / 34 out" in original.content
        assert "_last_usage" not in loop.sessions.get_or_create("cli:test").metadata
