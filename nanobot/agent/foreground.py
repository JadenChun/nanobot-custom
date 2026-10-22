"""Compose owned foreground capabilities without activating external resources."""
from __future__ import annotations

from pathlib import Path

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.tools.context import ForkToolDependencies
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.config.schema import Config, ToolsConfig
from nanobot.context_repo import ContextRepoManager
from nanobot.extensions.crawler import crawler_settings
from nanobot.utils.llm_runtime import LLMRuntime


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
    optional = crawler_settings(config)
    locks = FileLockRegistry()
    manager = ForegroundAgentManager(
        provider=runtime.provider, model=runtime.model, runtime=runtime,
        workspace=workspace, context_manager=context_manager,
        restrict_to_workspace=restrict_to_workspace, file_lock_registry=locks,
        web_search_config=tools_config.web.search, web_proxy=tools_config.web.proxy,
        agent_browser_config=tools_config.agent_browser,
        agent_device_config=tools_config.agent_device,
        exec_config=tools_config.exec, mcp_servers=tools_config.mcp_servers,
        crawler_runtime=optional.runtime,
        crawler_max_iterations=optional.max_iterations,
        crawler_max_input_tokens=optional.max_input_tokens,
        crawler_max_output_tokens=optional.max_output_tokens,
        crawler_reasoning_effort=optional.reasoning_effort,
    )
    return ForkToolDependencies(manager, manager.resource_policy, locks, crawler_enabled=optional.enabled)
