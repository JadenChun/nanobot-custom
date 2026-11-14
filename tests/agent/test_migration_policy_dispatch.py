"""Policy and terminal outcomes must survive the upstream execution path."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.policy import DelegatedReadOnlyPolicy
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.providers.base import LLMResponse, ToolCallRequest
from tests.agent.runner_helpers import make_run_spec

pytest_plugins = ("tests.agent.test_migration_source_engine",)


class RecordingTool(Tool):
    def __init__(self, name, calls, outcome):
        self._name, self.calls, self.outcome = name, calls, outcome

    name = property(lambda self: self._name)
    description = property(lambda self: self._name)
    parameters = property(lambda self: {"type": "object", "properties": {}})

    async def execute(self, context=None):
        self.calls.append((self.name, context))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize("mode", ["policy", "terminal", "cancel"])
async def test_real_runner_stops_before_later_tools_and_model_calls(mode):
    calls = []
    context = TurnContext(run_id="test-run")
    tools = ToolRegistry()
    outcome = {"policy": "read", "terminal": ToolOutcome("blocked", "policy_blocked", {"role": "worker"}),
               "cancel": asyncio.CancelledError()}[mode]
    tools.register(RecordingTool("read_file", calls, outcome))
    tools.register(RecordingTool("write_file", calls, "written"))
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="r", name="read_file", arguments={}),
        ToolCallRequest(id="w", name="write_file", arguments={}),
    ]))
    inject = AsyncMock(return_value=[])
    spec = make_run_spec(provider, initial_messages=[{"role": "user", "content": "inspect"}],
                         tools=tools, model="test", max_iterations=1, max_tool_result_chars=10000,
                         concurrent_tools=True, turn_context=context,
                         tool_policy=DelegatedReadOnlyPolicy(role="reviewer") if mode == "policy" else None,
                         injection_callback=inject)
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await AgentRunner().run(spec)
    else:
        result = await AgentRunner().run(spec)
        assert result.stop_reason == "policy_blocked"
        assert result.policy_metadata["role"] == ("reviewer" if mode == "policy" else "worker")
        assert [m["tool_call_id"] for m in result.messages if m["role"] == "tool"] == ["r", "w"]
        assert "Skipped" in result.messages[-1]["content"]
    assert calls == ([] if mode == "policy" else [("read_file", context)])
    assert provider.chat_with_retry.await_count == 1
    inject.assert_not_awaited()


@pytest.mark.parametrize("outcome", [ToolOutcome("failed", "tool_error", {"reason": "fixture"}),
                                    ToolResult.error("Error: failed")])
async def test_tool_errors_preserve_fail_fast_semantics(outcome):
    calls = []
    tools = ToolRegistry()
    tools.register(RecordingTool("read_file", calls, outcome))
    tools.register(RecordingTool("write_file", calls, "written"))
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="r", name="read_file", arguments={}),
        ToolCallRequest(id="w", name="write_file", arguments={}),
    ]))
    result = await AgentRunner().run(make_run_spec(
        provider, initial_messages=[], tools=tools, model="test", max_iterations=1,
        max_tool_result_chars=10000, fail_on_tool_error=True, concurrent_tools=True))
    assert result.stop_reason == "tool_error"
    assert calls == [("read_file", None)]
    assert provider.chat_with_retry.await_count == 1
    assert "Skipped" in result.messages[-1]["content"]
    if isinstance(outcome, ToolOutcome):
        assert result.policy_metadata == {"reason": "fixture"}


async def test_delegation_captures_crawler_capacity_without_mutating_provider(engine_config, monkeypatch):
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.social_crawl import SocialCrawlTool
    from nanobot.agent.turn import TurnRequest
    from nanobot.providers.base import GenerationSettings
    from nanobot.providers.factory import ProviderSnapshot
    from tests.agent.test_migration_engine import ScriptedProvider
    from tests.agent.test_migration_source_engine import upgrade

    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    engine_config.agents.crawler.enabled = True
    engine_config.agents.crawler.provider = "openrouter"
    engine_config.agents.crawler.model = "crawler"
    engine_config.agents.crawler.max_tokens.input = 20000
    engine_config.agents.crawler.max_tokens.output = 2000

    crawler = ScriptedProvider([LLMResponse(content="done")])
    crawler.generation = GenerationSettings(max_tokens=3000, temperature=0.2)
    main_provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="crawl", name="crawl_research", arguments={"task": "Inspect a page."},
        )]),
        LLMResponse(content="parent final"),
    ])
    main_provider.generation = GenerationSettings(max_tokens=4000, temperature=0.7)

    import nanobot.extensions.crawler as crawler_module

    def snapshot(config, *, preset):
        return ProviderSnapshot(
            provider=crawler,
            model=preset.model,
            context_window_tokens=preset.context_window_tokens,
            signature=("crawler",),
            generation=crawler.generation,
        )

    monkeypatch.setattr(crawler_module, "build_provider_snapshot", snapshot)

    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=main_provider)
    captured = {}
    original = loop.execute_child_turn

    async def record_child_turn(request, **kwargs):
        captured["request"] = request
        return await original(request, **kwargs)

    loop.execute_child_turn = record_child_turn
    monkeypatch.setattr(SocialCrawlTool, "prepare", AsyncMock())
    monkeypatch.setattr(SocialCrawlTool, "cleanup", AsyncMock())
    upgrade(loop, "cli:source")
    try:
        await loop.execute_turn(TurnRequest(content="Crawl a page.", session_key="cli:source"))
        request = captured["request"]
        assert request.runtime.provider is crawler
        assert request.runtime.context_window_tokens == 20000
        assert request.runtime.generation.max_tokens == 2000
        assert request.runtime.generation.temperature == 0.2
        # The capacity overrides derive an immutable runtime. They never mutate
        # the crawler or main provider generation snapshots.
        assert crawler.generation.max_tokens == 3000
        assert main_provider.generation.max_tokens == 4000
    finally:
        await loop.aclose()
