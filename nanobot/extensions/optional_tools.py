"""Optional crawler and VPN registrations, preserving their existing opt-in gates."""

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.crawler import CrawlResearchTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.social_crawl import SocialCrawlTool, crawl_tools_enabled
from nanobot.agent.tools.trend_vpn import (
    TrendVpnBrowserFetchTool,
    TrendVpnFetchTool,
    TrendVpnSessionCloseTool,
    TrendVpnSessionStartTool,
    vpn_tools_enabled,
)


def construct_optional_tools(ctx: ToolContext) -> list[Tool]:
    deps = ctx.fork_dependencies
    if deps is None:
        raise ValueError("optional tools require the foreground capability owner")
    result: list[Tool] = []
    if deps.crawler_enabled and crawl_tools_enabled():
        result.append(CrawlResearchTool(deps.foreground_manager))
    if vpn_tools_enabled():
        result.extend((TrendVpnSessionStartTool(), TrendVpnFetchTool(),
                       TrendVpnBrowserFetchTool(), TrendVpnSessionCloseTool()))
    return result


def optional_role_tools(role: str) -> ToolRegistry | None:
    if role != "crawler":
        return None
    tools = ToolRegistry()
    tools.register(SocialCrawlTool())
    return tools
