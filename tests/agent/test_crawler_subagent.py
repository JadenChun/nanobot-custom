import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.foreground import build_foreground_dependencies
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.social_crawl import SocialCrawlTool
from nanobot.agent.turn import TurnRequest
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.providers.factory import ProviderSnapshot
from nanobot.runtime_context import RuntimeContextBlock
from nanobot.utils.llm_runtime import LLMRuntime
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

pytest_plugins = ("tests.agent.test_migration_source_engine",)


class _RecordingCrawlerProvider(ScriptedProvider):
    """Record the model and generation overrides the canonical runner supplies."""

    def __init__(self, responses):
        super().__init__(responses)
        self.chat_kwargs = []

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.chat_kwargs.append((model, dict(kwargs), tools))
        self.calls.append((messages, tools, model))
        return self.responses.popleft()


class _BlockingCrawlerProvider(ScriptedProvider):
    """A crawler provider that blocks so the test can cancel the child turn."""

    def __init__(self, started: asyncio.Event) -> None:
        super().__init__([])
        self.started = started

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append((messages, tools, model))
        self.started.set()
        await asyncio.Future()


def _manager(tmp_path, **kwargs) -> ForegroundAgentManager:
    provider = MagicMock()
    provider.get_default_model.return_value = "main-model"
    provider.generation = GenerationSettings(max_tokens=4096, temperature=0.1)
    return ForegroundAgentManager(
        provider=provider,
        workspace=tmp_path,
        **kwargs,
    )


def _enable_crawler(config, *, model="vendor/cheap-crawler-model", input_tokens=18000,
                    output_tokens=1600, iterations=12, reasoning="low"):
    config.agents.crawler.enabled = True
    config.agents.crawler.provider = "openrouter"
    config.agents.crawler.model = model
    config.agents.crawler.max_tokens.input = input_tokens
    config.agents.crawler.max_tokens.output = output_tokens
    config.agents.crawler.max_tool_iterations = iterations
    config.agents.crawler.reasoning_effort = reasoning
    return config


def _install_crawler_snapshot(monkeypatch, crawler_provider, *, use_provider_generation=False):
    import nanobot.extensions.crawler as crawler_module

    def snapshot(config, *, preset):
        generation = (
            crawler_provider.generation
            if use_provider_generation
            else preset.to_generation_settings()
        )
        return ProviderSnapshot(
            provider=crawler_provider,
            model=preset.model,
            context_window_tokens=preset.context_window_tokens,
            signature=("crawler",),
            generation=generation,
        )

    monkeypatch.setattr(crawler_module, "build_provider_snapshot", snapshot)


def _build_crawler_loop(engine_config, monkeypatch, *, parent_provider, crawler_provider,
                        use_provider_generation=False, capture_child=True):
    """Build a source-backed loop whose parent turn delegates to the crawler role."""
    _install_crawler_snapshot(
        monkeypatch,
        crawler_provider,
        use_provider_generation=use_provider_generation,
    )
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=parent_provider)
    captured = {}
    if capture_child:
        original = loop.execute_child_turn

        async def record_child_turn(request, **kwargs):
            captured["request"] = request
            return await original(request, **kwargs)

        loop.execute_child_turn = record_child_turn
    prepare = AsyncMock()
    cleanup = AsyncMock()
    monkeypatch.setattr(SocialCrawlTool, "prepare", prepare)
    monkeypatch.setattr(SocialCrawlTool, "cleanup", cleanup)
    upgrade(loop, "cli:source")
    return loop, captured, prepare, cleanup


def _parent_delegating_to_crawler(task, *, final="parent final"):
    return ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="crawl",
            name="crawl_research",
            arguments={"task": task},
        )]),
        LLMResponse(content=final),
    ])


def test_crawler_config_accepts_camel_case_limits() -> None:
    config = Config(
        agents={
            "crawler": {
                "enabled": True,
                "provider": "openrouter",
                "model": "vendor/cheap-crawler-model",
                "maxTokens": {"input": 18000, "output": 1600},
                "maxToolIterations": 12,
                "reasoningEffort": "low",
            }
        }
    )

    assert config.agents.crawler.enabled is True
    assert config.agents.crawler.provider == "openrouter"
    assert config.agents.crawler.model == "vendor/cheap-crawler-model"
    assert config.agents.crawler.max_tokens.input == 18000
    assert config.agents.crawler.max_tokens.output == 1600
    assert config.agents.crawler.max_tool_iterations == 12


def test_crawler_role_can_use_a_provider_separate_from_main_agent(tmp_path) -> None:
    crawler_provider = MagicMock()
    crawler_provider.generation = GenerationSettings(max_tokens=2048, temperature=0.1)
    manager = _manager(tmp_path, crawler_provider=crawler_provider)

    assert manager.crawler_provider is crawler_provider
    assert manager.provider is not crawler_provider
    assert manager.crawler_runner is not manager.runner


@pytest.mark.asyncio
async def test_main_loop_exposes_only_foreground_crawler_delegation(tmp_path, monkeypatch) -> None:
    from nanobot.agent.loop import AgentLoop

    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    provider = MagicMock()
    provider.get_default_model.return_value = "main-model"
    provider.generation = GenerationSettings(max_tokens=4096, temperature=0.1)
    config = Config(
        agents={
            "defaults": {"workspace": str(tmp_path), "model": "main-model"},
            "crawler": {
                "enabled": True,
                "provider": "openrouter",
                "model": "vendor/cheap-crawler-model",
            },
        },
        providers={"openrouter": {"apiKey": "crawler-key"}},
    )
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    loop = AgentLoop.from_config(config, bus=MessageBus(), tool_registry=ToolRegistry(),
                                 provider=provider)

    try:
        assert loop.tools.get("crawl_research") is not None
        assert loop.tools.get("spawn") is None
        assert loop.tools.get("spawn_crawler") is None
        assert loop.tools.get("spawn_pipeline") is None
    finally:
        # The constructed graph is owned by this loop in the real composition path.
        await loop.aclose()


def test_crawler_provider_is_created_from_its_own_provider_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    config = Config(
        agents={
            "defaults": {
                "workspace": str(tmp_path),
                "provider": "anthropic",
                "model": "anthropic/main-model",
            },
            "crawler": {
                "enabled": True,
                "provider": "openrouter",
                "model": "vendor/cheap-crawler-model",
            },
        },
        providers={
            "anthropic": {"apiKey": "main-key"},
            "openrouter": {"apiKey": "sk-or-crawler-key"},
        },
    )

    main_provider = MagicMock()
    main_provider.get_default_model.return_value = "anthropic/main-model"
    main_provider.generation = GenerationSettings(max_tokens=4096, temperature=0.1)
    runtime = LLMRuntime.capture(main_provider, "anthropic/main-model", context_window_tokens=120000)
    deps = build_foreground_dependencies(
        workspace=config.workspace_path,
        runtime=runtime,
        tools_config=config.tools,
        restrict_to_workspace=True,
        config=config,
    )
    crawler_provider = deps.foreground_manager.crawler_provider

    assert crawler_provider.default_model == "vendor/cheap-crawler-model"
    assert deps.foreground_manager.provider is main_provider
    assert crawler_provider is not main_provider


@pytest.mark.asyncio
async def test_foreground_crawler_uses_its_own_model_and_budget(engine_config, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    _enable_crawler(engine_config, model="vendor/cheap-crawler-model", input_tokens=17000,
                    output_tokens=1400, iterations=9, reasoning="low")
    crawler_provider = _RecordingCrawlerProvider([LLMResponse(content="finished")])
    parent = _parent_delegating_to_crawler("Inspect the supplied public page.")
    loop, captured, prepare, cleanup = _build_crawler_loop(
        engine_config, monkeypatch, parent_provider=parent, crawler_provider=crawler_provider,
    )
    ambient_calls = []

    async def ambient(request):
        ambient_calls.append(request)
        return [
            RuntimeContextBlock(
                source="private",
                content="AMBIENT PRIVATE CONTEXT",
            )
        ]

    loop._runtime_context_providers = [ambient]
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Crawl the supplied page.", session_key="cli:source",
        ))
        assert result.error is None, result.error
        request = captured["request"]
        assert request.runtime.provider is crawler_provider
        assert request.runtime.model == "vendor/cheap-crawler-model"
        assert request.runtime.context_window_tokens == 17000
        assert request.max_iterations == 9
        assert request.runtime.generation.max_tokens == 1400
        assert request.runtime.generation.reasoning_effort == "low"
        assert request.tools.tool_names == ["social_crawl"]
        assert request.tools.get("social_crawl").supports_parallel_calls is False
        prompt = request.instructions
        assert "compact rendered HTML" in prompt
        assert "manages the session internally" in prompt
        assert "viewport screenshot" in prompt
        assert "Never\nclose the browser yourself" in prompt
        # The crawler role used its own provider with the derived budget; the
        # main provider did not serve the child turn.
        assert len(crawler_provider.calls) == 1
        crawler_messages, _crawler_tools, _crawler_model = crawler_provider.calls[0]
        system_prompt = crawler_messages[0]["content"]
        assert "Crawler Role" in system_prompt
        assert str(engine_config.workspace_path) not in system_prompt
        assert "Context Repositories" not in system_prompt
        assert "# Memory" not in system_prompt
        assert "AMBIENT PRIVATE CONTEXT" not in str(crawler_messages)
        assert len(ambient_calls) == 1
        assert crawler_messages[-1]["content"] == "Inspect the supplied public page."
        model, kwargs, _tools = crawler_provider.chat_kwargs[0]
        assert model == "vendor/cheap-crawler-model"
        assert kwargs["max_tokens"] == 1400
        assert kwargs["reasoning_effort"] == "low"
        prepare.assert_awaited_once()
        cleanup.assert_awaited_once()
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_authenticated_crawler_prompt_is_authorized_read_only(engine_config, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    monkeypatch.setenv("CRAWL4AI_AUTH_PROFILE_ENABLED", "true")
    _enable_crawler(engine_config)
    crawler_provider = _RecordingCrawlerProvider([LLMResponse(content="finished")])
    parent = _parent_delegating_to_crawler("Inspect public comments visible after login.")
    loop, captured, _prepare, _cleanup = _build_crawler_loop(
        engine_config, monkeypatch, parent_provider=parent, crawler_provider=crawler_provider,
    )
    try:
        await loop.execute_turn(TurnRequest(
            content="Crawl logged-in comments.", session_key="cli:source",
        ))
        prompt = captured["request"].instructions
        assert "operator-prepared authenticated browser profile" in prompt
        assert "including non-public content visible to the signed-in account" in prompt
        assert "never request, expose, or enter them" in prompt
        assert "outside" in prompt
        assert "profile's authorized scope" in prompt
        assert "Do not open DMs" in prompt
        assert "Clicking is disabled" in prompt
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_foreground_crawler_propagates_cancellation_to_real_runner(engine_config, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    _enable_crawler(engine_config)
    started = asyncio.Event()
    crawler_provider = _BlockingCrawlerProvider(started)
    parent = _parent_delegating_to_crawler("Inspect one page.")
    loop, _captured, prepare, cleanup = _build_crawler_loop(
        engine_config, monkeypatch, parent_provider=parent, crawler_provider=crawler_provider,
        capture_child=False,
    )
    try:
        turn_task = asyncio.create_task(loop.execute_turn(TurnRequest(
            content="Crawl one page.", session_key="cli:source",
        )))
        await asyncio.wait_for(started.wait(), timeout=1)
        turn_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await turn_task
        prepare.assert_awaited_once()
        cleanup.assert_awaited_once()
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_crawler_tool_waits_for_foreground_result(engine_config, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    _enable_crawler(engine_config, model="crawler-model")
    crawler_provider = _RecordingCrawlerProvider([LLMResponse(content="findings")])
    parent = _parent_delegating_to_crawler("Inspect this public Instagram post")
    loop, captured, _prepare, _cleanup = _build_crawler_loop(
        engine_config, monkeypatch, parent_provider=parent, crawler_provider=crawler_provider,
    )
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Inspect a public post.", session_key="cli:source",
        ))
        # The parent turn awaited the crawler and received its findings, not a
        # background acknowledgement.
        assert result.content == "parent final"
        assert len(crawler_provider.calls) == 1
        tool_rows = [row for row in parent.calls[1][0] if row.get("role") == "tool"]
        assert [row["content"] for row in tool_rows] == ["findings"]
        # The crawler child used its own provider and settled in the same turn.
        assert captured["request"].runtime.provider is crawler_provider
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        children = tree.children_of("cli:source")
        assert [child.state for child in children] == ["completed"]
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_foreground_crawler_is_rejected_when_worker_is_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "false")
    manager = _manager(tmp_path)

    result = await manager.run_crawler(task="Inspect a page")

    assert result == "Error: crawler worker integration is disabled"
