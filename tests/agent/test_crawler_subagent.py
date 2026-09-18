import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.foreground import build_foreground_dependencies
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.crawler import CrawlResearchTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.social_crawl import SocialCrawlTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.utils.llm_runtime import LLMRuntime


class _BlockingProvider(LLMProvider):
    """Provider that lets the test observe cancellation at the real runner seam."""

    def __init__(self, started: asyncio.Event) -> None:
        super().__init__(provider_name="crawler-test")
        self.started = started
        self.generation = GenerationSettings(max_tokens=256, temperature=0.1)

    async def chat(self, messages, tools=None, model=None, **kwargs):
        _ = messages, tools, model, kwargs
        self.started.set()
        await asyncio.Future()

    def get_default_model(self) -> str:
        return "crawler-test-model"


def _manager(tmp_path, **kwargs) -> ForegroundAgentManager:
    provider = MagicMock()
    provider.get_default_model.return_value = "main-model"
    provider.generation = GenerationSettings(max_tokens=4096, temperature=0.1)
    return ForegroundAgentManager(
        provider=provider,
        workspace=tmp_path,
        **kwargs,
    )


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
async def test_foreground_crawler_uses_its_own_model_and_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    crawler_provider = MagicMock()
    crawler_provider.generation = GenerationSettings(max_tokens=2048, temperature=0.1)
    manager = _manager(
        tmp_path,
        crawler_provider=crawler_provider,
        crawler_model="openrouter/cheap-crawler-model",
        crawler_max_iterations=9,
        crawler_max_input_tokens=17000,
        crawler_max_output_tokens=1400,
        crawler_reasoning_effort="low",
    )
    manager.crawler_runner.run = AsyncMock(return_value=AgentRunResult(
        stop_reason="completed", final_content="finished", messages=[]
    ))
    monkeypatch.setattr(SocialCrawlTool, "prepare", AsyncMock())
    monkeypatch.setattr(SocialCrawlTool, "cleanup", AsyncMock())

    result = await manager.run_crawler(task="Inspect the supplied public page.")

    assert result == "finished"
    spec = manager.crawler_runner.run.await_args.args[0]
    assert spec.runtime.provider is crawler_provider
    assert spec.runtime.model == "openrouter/cheap-crawler-model"
    assert spec.runtime.context_window_tokens == 17000
    assert spec.max_iterations == 9
    assert spec.runtime.generation.max_tokens == 1400
    assert spec.runtime.generation.reasoning_effort == "low"
    assert spec.concurrent_tools is False
    assert spec.tools.tool_names == ["social_crawl"]
    assert "compact rendered HTML" in spec.initial_messages[0]["content"]
    assert "manages the session internally" in spec.initial_messages[0]["content"]
    assert "viewport screenshot" in spec.initial_messages[0]["content"]
    assert "Never\nclose the browser yourself" in spec.initial_messages[0]["content"]


@pytest.mark.asyncio
async def test_authenticated_crawler_prompt_is_authorized_read_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    monkeypatch.setenv("CRAWL4AI_AUTH_PROFILE_ENABLED", "true")
    manager = _manager(tmp_path)
    manager.crawler_runner.run = AsyncMock(
        return_value=MagicMock(stop_reason="completed", final_content="finished", tool_events=[])
    )
    monkeypatch.setattr(SocialCrawlTool, "prepare", AsyncMock())
    monkeypatch.setattr(SocialCrawlTool, "cleanup", AsyncMock())

    await manager.run_crawler(task="Inspect public comments visible after login.")

    prompt = manager.crawler_runner.run.await_args.args[0].initial_messages[0]["content"]
    assert "operator-prepared authenticated browser profile" in prompt
    assert "including non-public content visible to the signed-in account" in prompt
    assert "never request, expose, or enter them" in prompt
    assert "outside" in prompt
    assert "profile's authorized scope" in prompt
    assert "Do not open DMs" in prompt
    assert "Clicking is disabled" in prompt


@pytest.mark.asyncio
async def test_foreground_crawler_propagates_cancellation_to_real_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    started = asyncio.Event()
    crawler_provider = _BlockingProvider(started)
    manager = _manager(
        tmp_path,
        crawler_provider=crawler_provider,
        crawler_model="crawler-test-model",
    )
    prepare = AsyncMock()
    cleanup = AsyncMock()
    monkeypatch.setattr(SocialCrawlTool, "prepare", prepare)
    monkeypatch.setattr(SocialCrawlTool, "cleanup", cleanup)

    task = asyncio.create_task(manager.run_crawler(task="Inspect one page."))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    prepare.assert_awaited_once()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_crawler_tool_waits_for_foreground_result(tmp_path, monkeypatch) -> None:
    from nanobot.providers.base import LLMResponse
    from tests.agent.test_migration_engine import ScriptedProvider

    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    main_provider = ScriptedProvider([])
    crawler_provider = ScriptedProvider([LLMResponse(content="findings")])
    manager = ForegroundAgentManager(
        provider=main_provider,
        crawler_provider=crawler_provider,
        crawler_model="crawler-model",
        workspace=tmp_path,
    )
    monkeypatch.setattr(SocialCrawlTool, "prepare", AsyncMock())
    monkeypatch.setattr(SocialCrawlTool, "cleanup", AsyncMock())
    tool = CrawlResearchTool(manager)

    result = await tool.execute("Inspect this public Instagram post")

    assert result == "findings"
    assert len(main_provider.calls) == 0
    assert len(crawler_provider.calls) == 1
    assert getattr(manager, "_background_tasks", set()) == set()


@pytest.mark.asyncio
async def test_foreground_crawler_is_rejected_when_worker_is_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "false")
    manager = _manager(tmp_path)

    result = await manager.run_crawler(task="Inspect a page")

    assert result == "Error: crawler worker integration is disabled"
