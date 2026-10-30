"""Recover a text-formatted tool attempt without persisting or executing it."""

from collections import deque

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnRequest
from nanobot.providers.base import LLMProvider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from tests.agent.runner_helpers import make_run_spec
from tests.agent.test_migration_engine import engine_config as _engine_config_factory

XML_TOOL_CALL = (
    "<tool_call>\n<function=memory_search>\n<parameter=query>\n"
    "earlier projects\n</parameter>\n</function>\n</tool_call>"
)


class _TextProvider(LLMProvider):
    def __init__(self, replies):
        super().__init__(provider_name="text-tool-test")
        self.replies = deque(replies)
        self.calls = []
        self.parser = OpenAICompatProvider(api_key="test", api_type="chat_completions")

    def get_default_model(self):
        return "test-model"

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools})
        assert self.replies, "unexpected extra model request"
        reply = self.replies.popleft()
        message = {"content": reply} if isinstance(reply, str) else reply
        return self.parser._parse({
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        })


@pytest.fixture
def engine_config(tmp_path, monkeypatch):
    return _engine_config_factory.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("corrected", ["Here is the project summary.", XML_TOOL_CALL])
async def test_text_tool_request_retries_once_and_preserves_usage(corrected):
    provider = _TextProvider([XML_TOOL_CALL, corrected])
    result = await AgentRunner().run(make_run_spec(
        provider, model="test-model", tools=ToolRegistry(), max_iterations=1,
        max_tool_result_chars=1000,
        initial_messages=[{"role": "user", "content": "check earlier projects"}],
    ))

    assert len(provider.calls) == 2
    assert provider.calls[1]["tools"] == []
    retry_note = provider.calls[1]["messages"][-1]["content"]
    assert "provided tool" in retry_note
    assert XML_TOOL_CALL not in str(provider.calls[1]["messages"])
    assert "<tool_call>" not in str(result.messages)
    assert "<tool_call>" not in result.final_content
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 20
    if corrected == XML_TOOL_CALL:
        assert result.stop_reason == "error"
    else:
        assert result.stop_reason == "completed"
        assert result.final_content == corrected


@pytest.mark.parametrize("corrected", ["Here is the project summary.", XML_TOOL_CALL])
async def test_source_session_never_records_raw_markup_or_runtime_retry_note(engine_config, corrected):
    provider = _TextProvider([XML_TOOL_CALL, corrected])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "cli:text-tool-source"
    try:
        session = loop.sessions.get_or_create(key)
        loop.sessions.save(session)
        backend = loop.sessions._jsonl_store
        backend.source.upgrade_source_session(
            key, expected_revision=session.revision, expected_generation=session.generation,
            expected_incarnation=session.incarnation,
        )
        loop.sessions.invalidate(key)

        result = await loop.execute_turn(TurnRequest(content="check earlier projects", session_key=key))

        assert len(provider.calls) == 2
        events = backend.source.load_source_snapshot(key).events
        assert "<tool_call>" not in str(events)
        assert "memory_search" not in str(events)
        assert provider.calls[1]["messages"][-1]["content"] not in str(events)
        assert "<tool_call>" not in result.content
        if corrected != XML_TOOL_CALL:
            assert result.content == corrected
            assert events[-1]["content"] == corrected
    finally:
        await loop.aclose()


async def test_corrective_retry_can_read_a_real_file_and_continue_a_source_session(engine_config):
    (engine_config.workspace_path / "project.txt").write_text("verified project details")
    provider = _TextProvider([XML_TOOL_CALL, {
        "content": None, "tool_calls": [{
            "id": "read-project", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"project.txt"}'},
        }],
    }, "Project checked."])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "cli:text-tool-source-read"
    try:
        session = loop.sessions.get_or_create(key)
        loop.sessions.save(session)
        backend = loop.sessions._jsonl_store
        backend.source.upgrade_source_session(
            key, expected_revision=session.revision, expected_generation=session.generation,
            expected_incarnation=session.incarnation,
        )
        loop.sessions.invalidate(key)

        result = await loop.execute_turn(TurnRequest(content="check the project file", session_key=key))

        assert result.error is None, result.error
        assert result.content == "Project checked."
        assert result.tools_used == ["read_file"]
        assert len(provider.calls) == 3
        assert any(tool["function"]["name"] == "read_file" for tool in provider.calls[1]["tools"])
        assert any(message.get("role") == "tool" and "verified project details" in message["content"]
                   for message in provider.calls[2]["messages"])
        events = backend.source.load_source_snapshot(key).events
        assert "<tool_call>" not in str(events)
        assert "memory_search" not in str(events)
        assert events[-1]["content"] == "Project checked."
    finally:
        await loop.aclose()
