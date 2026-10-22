"""Test /new archival behavior."""

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.utils.prompt_templates import render_template
from tests.agent import p6_admitted_fixtures

admitted_async = p6_admitted_fixtures.admitted_async

_ARCHIVE_PROMPT = render_template("agent/consolidator_archive.md", strip=True)


class TestNewCommandArchival:
    """Test /new archival behavior with the structured archive flow."""

    @staticmethod
    async def _make_loop(tmp_path: Path, admitted_async):
        from nanobot.agent.loop import AgentLoop
        from nanobot.providers.base import GenerationSettings, LLMResponse

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.estimate_prompt_tokens.return_value = (10_000, "test")
        provider.generation = GenerationSettings(max_tokens=100)
        provider.supports_bound_conversation_state.return_value = False
        provider.can_resume_conversation_state.return_value = False

        def build(inputs):
            return AgentLoop(
                bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
                model="test-model", context_window_tokens=1,
                session_manager=inputs.sessions, tool_registry=inputs.tools,
                cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
                enable_legacy_auto_compact=False,
            )

        loop = await admitted_async(tmp_path, build, keys=("cli:test",))
        loop.provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(content="ok", tool_calls=[])
        )
        loop.tools.get_definitions = MagicMock(return_value=[])
        return loop

    @pytest.mark.asyncio
    async def test_new_clears_session_immediately_even_if_archive_fails(
        self,
        tmp_path: Path,
        admitted_async,
    ) -> None:
        """/new clears session immediately; archive is fire-and-forget."""
        from nanobot.bus.events import InboundMessage

        loop = await self._make_loop(tmp_path, admitted_async)
        session = loop.sessions.get_or_create("cli:test")
        for i in range(5):
            session.add_message("user", f"msg{i}")
            session.add_message("assistant", f"resp{i}")
        loop.sessions.save(session)

        call_count = 0
        expected_runtime = loop.llm_runtime()

        async def _failing_summarize(session, *, archive_end, runtime) -> None:
            nonlocal call_count
            assert runtime is expected_runtime
            assert session.key == "cli:test"
            assert archive_end == len(session.messages)
            call_count += 1

        loop.consolidator.archive_session = _failing_summarize  # type: ignore[method-assign]

        new_msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/new")
        response = await loop.process_direct(
            new_msg.content, session_key=new_msg.session_key, runtime=expected_runtime,
        )

        assert response is not None
        assert "new session started" in response.content.lower()

        session_after = loop.sessions.get_or_create("cli:test")
        assert len(session_after.messages) == 0

        await asyncio.wait_for(asyncio.gather(*loop._background_tasks), timeout=2)
        await loop.aclose()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_new_reuses_replay_prefix_and_archives_only_unarchived_messages(
        self,
        tmp_path: Path,
        admitted_async,
    ) -> None:
        from nanobot.bus.events import InboundMessage

        loop = await self._make_loop(tmp_path, admitted_async)
        loop.set_runtime_context_window(128_000)
        session = loop.sessions.get_or_create("cli:test")
        for i in range(5):
            session.add_message("user", f"msg{i}")
            session.add_message("assistant", f"resp{i}")
        session.last_archived = len(session.messages) - 2
        ordinary_history = session.get_history()
        assert [message["content"] for message in ordinary_history] == [
            "msg4",
            "resp4",
        ]
        loop.sessions.save(session)

        expected_runtime = loop.llm_runtime()
        scheduled: list[Coroutine[Any, Any, object]] = []
        loop.schedule_background = scheduled.append  # type: ignore[method-assign]

        new_msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/new")
        response = await loop.process_direct(
            new_msg.content, session_key=new_msg.session_key, runtime=expected_runtime,
        )

        assert response is not None
        assert "new session started" in response.content.lower()

        assert len(scheduled) == 1
        await scheduled[0]
        await loop.aclose()
        sent = loop.provider.chat_with_retry.call_args.kwargs["messages"]
        assert sent[1:-1] == ordinary_history
        assert sent[-1]["content"] == _ARCHIVE_PROMPT

    @pytest.mark.asyncio
    async def test_new_clears_session_and_responds(self, tmp_path: Path, admitted_async) -> None:
        from nanobot.bus.events import InboundMessage

        loop = await self._make_loop(tmp_path, admitted_async)
        session = loop.sessions.get_or_create("cli:test")
        for i in range(3):
            session.add_message("user", f"msg{i}")
            session.add_message("assistant", f"resp{i}")
        loop.sessions.save(session)
        expected_runtime = loop.llm_runtime()

        async def _ok_summarize(session, *, archive_end, runtime) -> str:
            assert runtime is expected_runtime
            assert session.key == "cli:test"
            assert archive_end == len(session.messages)
            return "Summary."

        loop.consolidator.archive_session = _ok_summarize  # type: ignore[method-assign]

        new_msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/new")
        response = await loop.process_direct(
            new_msg.content, session_key=new_msg.session_key, runtime=expected_runtime,
        )

        assert response is not None
        assert "new session started" in response.content.lower()
        assert loop.sessions.get_or_create("cli:test").messages == []
        await asyncio.wait_for(asyncio.gather(*loop._background_tasks), timeout=2)

    @pytest.mark.asyncio
    async def test_aclose_cancels_and_joins_background_archival(
        self, tmp_path: Path, admitted_async,
    ) -> None:
        """Shutdown waits for cancellation cleanup, not archive completion."""
        loop = await self._make_loop(tmp_path, admitted_async)
        session = loop.sessions.get_or_create("cli:test")
        session.add_message("user", "retain prior work")
        session.add_message("assistant", "prior answer")
        loop.sessions.save(session)
        started = asyncio.Event()
        cancelled = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleaned = asyncio.Event()
        expected_runtime = loop.llm_runtime()

        async def slow_archive(session, *, archive_end, runtime):
            assert runtime is expected_runtime
            assert session.key == "cli:test"
            assert archive_end == 2
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release_cleanup.wait()
                cleaned.set()
                raise

        loop.consolidator.archive_session = slow_archive
        response = await loop.process_direct(
            "/new", session_key="cli:test", runtime=expected_runtime,
        )
        assert response is not None and "new session started" in response.content.lower()
        assert loop.sessions.get_or_create("cli:test").messages == []
        await asyncio.wait_for(started.wait(), timeout=2)
        background = tuple(loop._background_tasks)
        assert len(background) == 1
        closing = asyncio.create_task(loop.aclose())
        try:
            await asyncio.wait_for(cancelled.wait(), timeout=2)
            assert not closing.done()
            assert not cleaned.is_set()
        finally:
            release_cleanup.set()
            await asyncio.wait_for(closing, timeout=2)
        assert cleaned.is_set()
        assert all(task.done() and task.cancelled() for task in background)
        assert not loop._background_tasks
        assert loop._closed
