"""Terminal disposition regressions for source-write failure and late success."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunner
from nanobot.agent.turn import TurnContext
from nanobot.providers.base import LLMResponse
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import SourceValidationError
from nanobot.session.turn_writer import SourceTurnWriter
from tests.agent.runner_helpers import make_run_spec

_MAX_TOOL_RESULT_CHARS = 1000


def _tools() -> MagicMock:
    tools = MagicMock()
    tools.get_definitions.return_value = []
    return tools


@pytest.mark.asyncio
async def test_mandatory_source_write_failure_fails_owning_turn(tmp_path):
    store = JsonlSessionStore(tmp_path)
    session = Session("test:writer")
    store.save(session)
    store.source.upgrade_source_session(
        session.key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content="done",
            finish_reason="stop",
            tool_calls=[],
            usage=None,
        )
    )
    tools = _tools()

    async with acquire_execution(store, "test:writer", run_id="run") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="run")
        writer.append_message({"role": "user", "content": "input"})

        async def injection_callback():
            writer.fail()
            raise SourceValidationError("forced source write failure")

        spec = make_run_spec(
            provider,
            model="test-model",
            max_iterations=4,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            tools=tools,
            initial_messages=[{"role": "user", "content": "input"}],
            source_writer=writer,
            injection_callback=injection_callback,
            turn_context=TurnContext(execution_owner=owner),
        )

        with pytest.raises(SourceValidationError, match="forced source write failure"):
            await AgentRunner().run(spec)

    assert provider.chat_with_retry.await_count == 1
    assert writer.failed is True


@pytest.mark.asyncio
async def test_late_success_clears_prior_model_error_disposition():
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                finish_reason="error",
                tool_calls=[],
                usage=None,
            ),
            LLMResponse(
                content="final success",
                finish_reason="stop",
                tool_calls=[],
                usage=None,
            ),
        ]
    )
    tools = _tools()
    injection_calls = 0

    async def injection_callback():
        nonlocal injection_calls
        injection_calls += 1
        if injection_calls == 1:
            return [{"role": "user", "content": "continue after error"}]
        return []

    result = await AgentRunner().run(
        make_run_spec(
            provider,
            model="test-model",
            max_iterations=4,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            tools=tools,
            initial_messages=[{"role": "user", "content": "start"}],
            injection_callback=injection_callback,
        )
    )

    assert result.stop_reason == "completed"
    assert result.final_content == "final success"
    assert result.error is None
    assert provider.chat_with_retry.await_count == 2
    injected = [
        message
        for message in result.messages
        if message.get("role") == "user"
        and message.get("content") == "continue after error"
    ]
    assert len(injected) == 1
