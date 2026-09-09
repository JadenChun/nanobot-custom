"""Compose owned foreground capabilities without activating external resources."""
from __future__ import annotations

from pathlib import Path

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.tools.context import ForkToolDependencies
from nanobot.agent.tools.social_crawl import crawl_tools_enabled
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.config.schema import Config, ModelPresetConfig, ToolsConfig
from nanobot.context_repo import ContextRepoManager
from nanobot.providers.factory import build_provider_snapshot
from nanobot.utils.llm_runtime import LLMRuntime, runtime_from_provider_snapshot


def validate_foreground_dependencies(
    deps: ForkToolDependencies, *, workspace: Path, restrict_to_workspace: bool,
) -> None:
    """Reject split policy/lock ownership before registering powerful tools."""
    manager = deps.foreground_manager
    policy = deps.resource_policy
    if (policy is not manager.resource_policy
        or policy.context_manager is not manager.context_manager
        or deps.file_locks is not manager.file_lock_registry
        or policy.workspace != workspace.expanduser().resolve()
        or policy.restrict_to_workspace != restrict_to_workspace
        or manager.workspace.expanduser().resolve() != policy.workspace
        or manager.restrict_to_workspace != restrict_to_workspace):
        raise ValueError('Inconsistent foreground dependency policy, workspace or file locks')


def build_foreground_dependencies(
    *, workspace: Path, runtime: LLMRuntime, tools_config: ToolsConfig,
    restrict_to_workspace: bool, config: Config | None = None,
) -> ForkToolDependencies:
    """Build one shared context/policy/lock graph; never connect or run tools."""
    defaults = config.agents.defaults if config is not None else None
    context_manager = ContextRepoManager.from_config(
        context_paths=defaults.context_paths if defaults else None,
        context_repos=defaults.context_repos if defaults else None,
    )
    crawler = config.agents.crawler if config is not None else None
    crawler_enabled = bool(crawler and crawler.enabled and crawl_tools_enabled())
    crawler_runtime = None
    if crawler_enabled:
        assert config is not None and crawler is not None
        crawler_config = config.model_copy(deep=True)
        crawler_config.agents.defaults.fallback = []
        crawler_config.agents.defaults.fallback_models = []
        preset = ModelPresetConfig(
            model=crawler.model or config.resolve_preset().model,
            provider=crawler.provider or 'auto', temperature=0.1,
            max_tokens=crawler.max_tokens.output,
            context_window_tokens=crawler.max_tokens.input,
            reasoning_effort=crawler.reasoning_effort,
        )
        crawler_runtime = runtime_from_provider_snapshot(build_provider_snapshot(crawler_config, preset=preset))
    locks = FileLockRegistry()
    manager = ForegroundAgentManager(
        provider=runtime.provider, model=runtime.model, runtime=runtime,
        workspace=workspace, context_manager=context_manager,
        restrict_to_workspace=restrict_to_workspace, file_lock_registry=locks,
        web_search_config=tools_config.web.search, web_proxy=tools_config.web.proxy,
        agent_browser_config=tools_config.agent_browser,
        agent_device_config=tools_config.agent_device,
        exec_config=tools_config.exec, mcp_servers=tools_config.mcp_servers,
        crawler_runtime=crawler_runtime,
        crawler_max_iterations=crawler.max_tool_iterations if crawler else 20,
        crawler_max_input_tokens=crawler.max_tokens.input if crawler else 20000,
        crawler_max_output_tokens=crawler.max_tokens.output if crawler else 2000,
        crawler_reasoning_effort=crawler.reasoning_effort if crawler else 'low',
    )
    return ForkToolDependencies(manager, manager.resource_policy, locks, crawler_enabled=crawler_enabled)
