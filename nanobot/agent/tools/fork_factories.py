"""Explicit foreground capabilities; importing a tool never enables it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

from nanobot.agent.tools.agent_browser import AgentBrowserTool
from nanobot.agent.tools.agent_device import AgentDeviceTool
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.crawler import CrawlResearchTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.delegation import DelegateTaskTool, PlanTaskTool, ReviewWorkTool
from nanobot.agent.tools.desktop_use import DesktopUseTool
from nanobot.agent.tools.exec_session import ExecSessionTool, ListExecSessionsTool
from nanobot.agent.tools.explore import ExploreTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.image_generation import ImageGenerationTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.social_crawl import crawl_tools_enabled
from nanobot.agent.tools.trend_vpn import (
    TrendVpnBrowserFetchTool,
    TrendVpnFetchTool,
    TrendVpnSessionCloseTool,
    TrendVpnSessionStartTool,
    vpn_tools_enabled,
)
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool


@dataclass(frozen=True)
class ToolFactory:
    name: str
    enabled: Callable[[], bool]
    create: Callable[[], Tool]


def construct_core_tools(ctx: ToolContext) -> list[Tool]:
    """Construct the complete set before mutating the destination registry."""
    deps = ctx.fork_dependencies
    if deps is None or cast(object, deps.foreground_manager) is None:
        raise ValueError("Foreground tool construction requires foreground_manager dependencies")
    if cast(object, deps.resource_policy) is None or cast(object, deps.file_locks) is None:
        raise ValueError("Foreground tool construction requires resource policy and file locks")
    entries: list[ToolFactory] = []

    def upstream(name: str, cls: type[Tool]) -> None:
        entries.append(ToolFactory(name, lambda: cls.enabled(ctx), lambda: cls.create(ctx)))

    for name, cls in (("read_file", ReadFileTool), ("write_file", WriteFileTool),
                      ("edit_file", EditFileTool), ("list_dir", ListDirTool),
                      ("exec", ExecTool), ("exec_session", ExecSessionTool),
                      ("list_exec_sessions", ListExecSessionsTool), ("web_search", WebSearchTool),
                      ("web_fetch", WebFetchTool), ("message", MessageTool),
                      ("cron", CronTool), ("generate_image", ImageGenerationTool)):
        upstream(name, cls)
    for name, cls, cfg in (("agent_browser", AgentBrowserTool, ctx.config.agent_browser),
                           ("agent_device", AgentDeviceTool, ctx.config.agent_device)):
        entries.append(ToolFactory(name, lambda cfg=cfg: cfg.enabled,
                                   lambda cls=cls, cfg=cfg: cls(
                                       package=cfg.package, timeout=cfg.timeout,
                                       max_output_chars=cfg.max_output_chars, working_dir=ctx.workspace)))
    desktop = ctx.config.desktop_use
    entries.append(ToolFactory("desktop_use", lambda: desktop.enabled, lambda: DesktopUseTool(
        screenshot_delay=desktop.screenshot_delay, typing_delay_ms=desktop.typing_delay_ms,
        scaling_enabled=desktop.scaling_enabled, max_output_chars=desktop.max_output_chars,
        working_dir=ctx.workspace, humanize_typing=desktop.humanize_typing)))
    for name, cls in (("plan_task", PlanTaskTool), ("delegate_task", DelegateTaskTool),
                      ("review_work", ReviewWorkTool)):
        entries.append(ToolFactory(name, lambda: True,
                                   lambda cls=cls: cls(deps.foreground_manager)))
    entries.append(ToolFactory("explore", lambda: True, lambda: ExploreTool(
        deps.foreground_manager, max_iterations=deps.explore_max_iterations)))
    entries.append(ToolFactory("crawl_research", lambda: deps.crawler_enabled and crawl_tools_enabled(),
                               lambda: CrawlResearchTool(deps.foreground_manager)))
    for name, cls in (("trend_vpn_session_start", TrendVpnSessionStartTool),
                      ("trend_vpn_fetch", TrendVpnFetchTool),
                      ("trend_vpn_browser_fetch", TrendVpnBrowserFetchTool),
                      ("trend_vpn_session_close", TrendVpnSessionCloseTool)):
        entries.append(ToolFactory(name, vpn_tools_enabled, cls))
    result: list[Tool] = []
    for entry in entries:
        if not entry.enabled():
            continue
        tool = entry.create()
        if tool.name != entry.name:
            raise ValueError(f"Tool factory name mismatch: expected {entry.name}")
        result.append(tool)
    return result
