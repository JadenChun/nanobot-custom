"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry

_READ_ONLY_PREFIXES = (
    "get_",
    "list_",
    "inspect_",
    "preview_",
    "read_",
    "search_",
    "find_",
    "fetch_",
    "status_",
)

_SERIAL_ONLY_PREFIXES = (
    "preview_",
)

_MUTATING_PREFIXES = (
    "create_",
    "open_",
    "add_",
    "update_",
    "set_",
    "remove_",
    "delete_",
    "trim_",
    "split_",
    "move_",
    "import_",
    "save_",
    "export_",
    "generate_",
    "cancel_",
    "retake_",
    "fill_",
)


@dataclass(frozen=True, slots=True)
class MCPToolMetadata:
    """Normalized MCP metadata used for policy decisions inside nanobot."""

    server_name: str
    original_name: str
    wrapped_name: str
    transport_type: str
    server_url: str | None
    server_command: str | None
    read_only_hint: bool | None
    destructive_hint: bool | None
    idempotent_hint: bool | None
    open_world_hint: bool | None
    trusted_for_heuristics: bool
    read_only: bool
    read_only_source: str
    supports_parallel_calls: bool


def _extract_nullable_branch(options: Any) -> tuple[dict[str, Any], bool] | None:
    """Return the single non-null branch for nullable unions."""
    if not isinstance(options, list):
        return None

    non_null: list[dict[str, Any]] = []
    saw_null = False
    for option in options:
        if not isinstance(option, dict):
            return None
        if option.get("type") == "null":
            saw_null = True
            continue
        non_null.append(option)

    if saw_null and len(non_null) == 1:
        return non_null[0], True
    return None


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    """Normalize only nullable JSON Schema patterns for tool definitions."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)

    raw_type = normalized.get("type")
    if isinstance(raw_type, list):
        non_null = [item for item in raw_type if item != "null"]
        if "null" in raw_type and len(non_null) == 1:
            normalized["type"] = non_null[0]
            normalized["nullable"] = True

    for key in ("oneOf", "anyOf"):
        nullable_branch = _extract_nullable_branch(normalized.get(key))
        if nullable_branch is not None:
            branch, _ = nullable_branch
            merged = {k: v for k, v in normalized.items() if k != key}
            merged.update(branch)
            normalized = merged
            normalized["nullable"] = True
            break

    if "properties" in normalized and isinstance(normalized["properties"], dict):
        normalized["properties"] = {
            name: _normalize_schema_for_openai(prop)
            if isinstance(prop, dict)
            else prop
            for name, prop in normalized["properties"].items()
        }

    if "items" in normalized and isinstance(normalized["items"], dict):
        normalized["items"] = _normalize_schema_for_openai(normalized["items"])

    if normalized.get("type") != "object":
        return normalized

    normalized.setdefault("properties", {})
    normalized.setdefault("required", [])
    return normalized


def _annotation_bool(annotations: Any, field_name: str) -> bool | None:
    value = getattr(annotations, field_name, None)
    return value if isinstance(value, bool) else None


def _is_trusted_local_mcp(transport_type: str, server_url: str | None) -> bool:
    if transport_type == "stdio":
        return True
    if not server_url:
        return False
    parsed = urlparse(server_url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _classify_read_only(
    tool_name: str,
    *,
    transport_type: str,
    server_url: str | None,
    annotations: Any,
) -> tuple[bool, str, bool]:
    read_only_hint = _annotation_bool(annotations, "readOnlyHint")
    if read_only_hint is not None:
        return read_only_hint, "annotation", _is_trusted_local_mcp(transport_type, server_url)

    trusted_for_heuristics = _is_trusted_local_mcp(transport_type, server_url)
    if trusted_for_heuristics:
        lowered_name = tool_name.lower()
        if lowered_name.startswith(_READ_ONLY_PREFIXES):
            return True, "heuristic", True
        if lowered_name.startswith(_MUTATING_PREFIXES):
            return False, "heuristic", True
        return False, "heuristic_default_mutating", True

    return False, "default_mutating", False


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as a nanobot Tool."""

    def __init__(
        self,
        session,
        server_name: str,
        tool_def,
        *,
        transport_type: str = "stdio",
        server_url: str | None = None,
        server_command: str | None = None,
        tool_timeout: int = 30,
    ):
        self._session = session
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        raw_schema = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._parameters = _normalize_schema_for_openai(raw_schema)
        self._tool_timeout = tool_timeout
        annotations = getattr(tool_def, "annotations", None)
        read_only, source, trusted_for_heuristics = _classify_read_only(
            tool_def.name,
            transport_type=transport_type,
            server_url=server_url,
            annotations=annotations,
        )
        self._mcp_metadata = MCPToolMetadata(
            server_name=server_name,
            original_name=tool_def.name,
            wrapped_name=self._name,
            transport_type=transport_type,
            server_url=server_url or None,
            server_command=server_command or None,
            read_only_hint=_annotation_bool(annotations, "readOnlyHint"),
            destructive_hint=_annotation_bool(annotations, "destructiveHint"),
            idempotent_hint=_annotation_bool(annotations, "idempotentHint"),
            open_world_hint=_annotation_bool(annotations, "openWorldHint"),
            trusted_for_heuristics=trusted_for_heuristics,
            read_only=read_only,
            read_only_source=source,
            supports_parallel_calls=not tool_def.name.lower().startswith(_SERIAL_ONLY_PREFIXES),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def mcp_metadata(self) -> MCPToolMetadata:
        return self._mcp_metadata

    @property
    def is_read_only(self) -> bool:
        return self._mcp_metadata.read_only

    @property
    def supports_parallel_calls(self) -> bool:
        return self._mcp_metadata.supports_parallel_calls

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=self._tool_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("MCP tool '{}' timed out after {}s", self._name, self._tool_timeout)
            return f"(MCP tool call timed out after {self._tool_timeout}s)"
        except asyncio.CancelledError:
            # MCP SDK's anyio cancel scopes can leak CancelledError on timeout/failure.
            # Re-raise only if our task was externally cancelled (e.g. /stop).
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
            return "(MCP tool call was cancelled)"
        except Exception as exc:
            logger.exception(
                "MCP tool '{}' failed: {}: {}",
                self._name,
                type(exc).__name__,
                exc,
            )
            return f"(MCP tool call failed: {type(exc).__name__})"

        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts) or "(no output)"


def is_read_only_mcp_tool(tool: Tool) -> bool:
    """Return whether a wrapped MCP tool is classified as read-only."""
    return isinstance(tool, MCPToolWrapper) and tool.is_read_only


async def connect_mcp_servers(
    mcp_servers: dict,
    registry: ToolRegistry,
    stack: AsyncExitStack,
    tool_filter: Callable[[MCPToolWrapper], bool] | None = None,
) -> None:
    """Connect to configured MCP servers and register their tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    for name, cfg in mcp_servers.items():
        try:
            transport_type = cfg.type
            if not transport_type:
                if cfg.command:
                    transport_type = "stdio"
                elif cfg.url:
                    # Convention: URLs ending with /sse use SSE transport; others use streamableHttp
                    transport_type = (
                        "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
                    )
                else:
                    logger.warning("MCP server '{}': no command or url configured, skipping", name)
                    continue

            if transport_type == "stdio":
                params = StdioServerParameters(
                    command=cfg.command, args=cfg.args, env=cfg.env or None
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif transport_type == "sse":
                def httpx_client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    merged_headers = {
                        "Accept": "application/json, text/event-stream",
                        **(cfg.headers or {}),
                        **(headers or {}),
                    }
                    return httpx.AsyncClient(
                        headers=merged_headers or None,
                        follow_redirects=True,
                        timeout=timeout,
                        auth=auth,
                    )

                read, write = await stack.enter_async_context(
                    sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
                )
            elif transport_type == "streamableHttp":
                # Always provide an explicit httpx client so MCP HTTP transport does not
                # inherit httpx's default 5s timeout and preempt the higher-level tool timeout.
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.headers or None,
                        follow_redirects=True,
                        timeout=None,
                    )
                )
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )
            else:
                logger.warning("MCP server '{}': unknown transport type '{}'", name, transport_type)
                continue

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools = await session.list_tools()
            enabled_tools = set(cfg.enabled_tools)
            allow_all_tools = "*" in enabled_tools
            registered_count = 0
            matched_enabled_tools: set[str] = set()
            available_raw_names = [tool_def.name for tool_def in tools.tools]
            available_wrapped_names = [f"mcp_{name}_{tool_def.name}" for tool_def in tools.tools]
            for tool_def in tools.tools:
                wrapped_name = f"mcp_{name}_{tool_def.name}"
                if (
                    not allow_all_tools
                    and tool_def.name not in enabled_tools
                    and wrapped_name not in enabled_tools
                ):
                    logger.debug(
                        "MCP: skipping tool '{}' from server '{}' (not in enabledTools)",
                        wrapped_name,
                        name,
                    )
                    continue
                wrapper = MCPToolWrapper(
                    session,
                    name,
                    tool_def,
                    transport_type=transport_type,
                    server_url=cfg.url or None,
                    server_command=cfg.command or None,
                    tool_timeout=cfg.tool_timeout,
                )
                if tool_filter is not None and not tool_filter(wrapper):
                    logger.debug(
                        "MCP: filtered out tool '{}' from server '{}' (read_only={}, source={})",
                        wrapper.name,
                        name,
                        wrapper.is_read_only,
                        wrapper.mcp_metadata.read_only_source,
                    )
                    continue
                registry.register(wrapper)
                logger.debug("MCP: registered tool '{}' from server '{}'", wrapper.name, name)
                registered_count += 1
                if enabled_tools:
                    if tool_def.name in enabled_tools:
                        matched_enabled_tools.add(tool_def.name)
                    if wrapped_name in enabled_tools:
                        matched_enabled_tools.add(wrapped_name)

            if enabled_tools and not allow_all_tools:
                unmatched_enabled_tools = sorted(enabled_tools - matched_enabled_tools)
                if unmatched_enabled_tools:
                    logger.warning(
                        "MCP server '{}': enabledTools entries not found: {}. Available raw names: {}. "
                        "Available wrapped names: {}",
                        name,
                        ", ".join(unmatched_enabled_tools),
                        ", ".join(available_raw_names) or "(none)",
                        ", ".join(available_wrapped_names) or "(none)",
                    )

            logger.info("MCP server '{}': connected, {} tools registered", name, registered_count)
        except asyncio.TimeoutError:
            logger.error(
                "MCP server '{}': connection timed out after {}s, skipping",
                name,
                cfg.tool_timeout,
            )
        except Exception as e:
            logger.error("MCP server '{}': failed to connect: {}", name, e)
