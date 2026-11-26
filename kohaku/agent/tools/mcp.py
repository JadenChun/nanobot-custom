"""MCP client and dynamic tool-provider lifecycle."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import re
import shutil
import urllib.parse
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeGuard, TypeVar, cast, overload
from urllib.parse import urlparse

import httpx
from loguru import logger

from kohaku.agent.tools.base import Tool, ToolResult, invalid_numeric_limit
from kohaku.agent.tools.progress import (
    ProgressThrottle,
    current_tool_progress_sink,
)
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import ToolOutcome
from kohaku.context.budget import TokenMeter
from kohaku.security.network import (
    PinnedDNSAsyncTransport,
    env_proxy_applies_to_url,
    httpx_env_proxy_mounts,
    resolve_url_target,
    validate_url_target,
)
from kohaku.utils.cancellation import task_is_cancelling

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Prompt, Resource
    from mcp.types import Tool as MCPToolDefinition

    from kohaku.agent.tools.mcp_oauth import MCPOAuthHandlers
    from kohaku.config.schema import Config, MCPServerConfig

# Transient connection errors that warrant a single retry.
# These typically happen when an MCP server restarts or a network
# connection is interrupted between calls.
_TRANSIENT_EXC_NAMES: frozenset[str] = frozenset((
    "ClosedResourceError",
    "BrokenResourceError",
    "EndOfStream",
    "BrokenPipeError",
    "ConnectionResetError",
    "ConnectionRefusedError",
    "ConnectionAbortedError",
    "ConnectionError",
))

_WINDOWS_SHELL_LAUNCHERS: frozenset[str] = frozenset(("npx", "npm", "pnpm", "yarn", "bunx"))

# Characters allowed in tool names by model providers (Anthropic, OpenAI, etc.).
# Replace anything outside [a-zA-Z0-9_-] with underscore and collapse runs.
_SANITIZE_RE = re.compile(r"_+")
_ReconnectCallback = Callable[[str, str, Tool], Awaitable[Tool | None]]
MCPServerLoader = Callable[[], Mapping[str, "MCPServerConfig"]]
MCPRuntimeStatus = Literal["connecting", "connected", "failed"]
_KOHAKU_MEDIA_CONTROL_TOOL = "cancel_kohaku_media_call"
_KOHAKU_MEDIA_CALL_META_KEY = "ai.nanobot.media.call_id"
_KOHAKU_MEDIA_SETTLE_STOP_REASON = "kohaku_media_settled"
_KOHAKU_MEDIA_CONTROL_TIMEOUT_SECONDS = 5.0
_MAX_MCP_PAGES = 20
_MAX_CONNECTIONS = 8
_MAX_DESCRIPTION_CHARS = 512
_MAX_SCHEMA_CHARS = 8192


class MCPConnection(Protocol):
    async def aclose(self) -> None: ...


async def _close_mcp_connection(name: str, connection: MCPConnection) -> None:
    try:
        await connection.aclose()
    except asyncio.CancelledError:
        if task_is_cancelling():
            raise
        logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
    except (RuntimeError, BaseExceptionGroup):
        logger.debug("MCP server '{}' cleanup error (can be ignored)", name)


async def close_mcp_connections(connections: Mapping[str, MCPConnection]) -> None:
    """Close owned connections, settling the group before propagating cancellation."""
    cancellation: asyncio.CancelledError | None = None
    for name, connection in connections.items():
        try:
            await _close_mcp_connection(name, connection)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
    if cancellation is not None:
        raise cancellation


class _OwnedMCPConnection:
    """Close an MCP transport from the task that originally opened it."""

    def __init__(self, owner: asyncio.Task[None], close_requested: asyncio.Event) -> None:
        self._owner = owner
        self._close_requested = close_requested

    async def aclose(self) -> None:
        self._close_requested.set()
        try:
            await asyncio.shield(self._owner)
        except asyncio.CancelledError:
            if not self._owner.cancelled():
                raise


def _is_malformed_mcp_progress_notification(message: Any) -> bool:
    payload = _mcp_jsonrpc_payload(message)
    if _payload_value(payload, "method") != "notifications/progress":
        return False

    params = _payload_value(payload, "params")
    return not _progress_params_have_token(params)


def _mcp_jsonrpc_payload(message: Any) -> Any:
    """Return the JSON-RPC payload across current and future MCP SDK shapes."""
    envelope = getattr(message, "message", message)
    return getattr(envelope, "root", None) or envelope


def _payload_value(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return cast(Mapping[str, Any], payload).get(key)
    return getattr(payload, key, None)


def _progress_params_have_token(params: Any) -> bool:
    if isinstance(params, Mapping):
        return "progressToken" in params
    return hasattr(params, "progressToken") or hasattr(params, "progress_token")


class _MalformedProgressNotificationFilter:
    def __init__(self, read_stream: Any, server_name: str) -> None:
        self._read_stream = read_stream
        self._server_name = server_name
        self._iterator: AsyncIterator[Any] | None = None

    async def __aenter__(self) -> "_MalformedProgressNotificationFilter":
        await self._read_stream.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return await self._read_stream.__aexit__(exc_type, exc, tb)

    def __aiter__(self) -> "_MalformedProgressNotificationFilter":
        self._iterator = self._read_stream.__aiter__()
        return self

    async def __anext__(self) -> Any:
        iterator = self._iterator
        if iterator is None:
            iterator = self._read_stream.__aiter__()
            self._iterator = iterator

        while True:
            message = await anext(iterator)
            if _is_malformed_mcp_progress_notification(message):
                logger.debug(
                    "MCP server '{}': dropped progress notification without progressToken",
                    self._server_name,
                )
                continue
            return message

    async def aclose(self) -> None:
        close = getattr(self._read_stream, "aclose", None)
        if close is not None:
            await close()


def _filter_malformed_mcp_progress_notifications(read_stream: Any, server_name: str) -> Any:
    if not all(hasattr(read_stream, name) for name in ("__aenter__", "__aexit__", "__aiter__")):
        return read_stream
    return _MalformedProgressNotificationFilter(read_stream, server_name)


def _sanitize_name(name: str) -> str:
    """Sanitize an MCP-derived name for model API compatibility."""
    return _SANITIZE_RE.sub("_", re.sub(r"[^a-zA-Z0-9_-]", "_", name))


_MAX_TOOL_NAME_LENGTH = 64
_HASH_LENGTH = 8


def _limit_tool_name(name: str, max_length: int = _MAX_TOOL_NAME_LENGTH) -> str:
    """Limit a tool name while keeping short names unchanged."""
    if len(name) <= max_length:
        return name

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    prefix_length = max_length - _HASH_LENGTH - 1
    return f"{name[:prefix_length]}_{digest}"


def _sanitize_mcp_tool_name(name: str) -> str:
    """Sanitize and limit an MCP-derived tool name."""
    return _limit_tool_name(_sanitize_name(name))


def _collision_name(name: str, original: str) -> str:
    suffix = "_" + hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
    return name[:_MAX_TOOL_NAME_LENGTH - len(suffix)] + suffix


def _is_transient(exc: BaseException) -> bool:
    """Check if an exception looks like a transient connection error."""
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES


def _is_transient_connection_failure(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        group = cast(BaseExceptionGroup[BaseException], exc)
        return bool(group.exceptions) and all(
            _is_transient_connection_failure(nested) for nested in group.exceptions
        )
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)) or _is_transient(exc)


def _log_mcp_connection_failure(name: str, exc: BaseException, hint: str = "") -> None:
    if _is_transient_connection_failure(exc):
        logger.warning("MCP server '{}': transient connection failure", name)
        logger.opt(exception=exc).debug(
            "MCP server '{}' transient connection failure details", name
        )
        return
    logger.opt(exception=exc).error("MCP server '{}': failed to connect: {}", name, hint)


def _is_session_terminated(exc: BaseException) -> bool:
    """Return True when the MCP SDK reports a dead client session."""
    if _is_transient(exc):
        return True
    messages = [str(exc)]
    error = getattr(exc, "error", None)
    if error is not None:
        messages.append(str(getattr(error, "message", "")))
    return any(
        marker in message.lower()
        for marker in ("session terminated", "connection closed")
        for message in messages
    )


async def _probe_http_url(url: str, timeout: float = 3.0) -> bool:
    """Quick TCP probe to check if an HTTP MCP server is reachable.

    Avoids entering ``streamable_http_client`` / ``sse_client`` when the port is
    closed — those transports use anyio task groups whose cleanup can raise
    ``RuntimeError`` / ``ExceptionGroup`` that escape the caller's try/except
    and crash the event loop.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if not port:
        port = 443 if parsed.scheme == "https" else 80
    ok, _, resolved_ips = resolve_url_target(url)
    if not ok:
        return False
    if env_proxy_applies_to_url(url):
        return True
    for target_host in resolved_ips or (host,):
        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target_host, port),
                timeout=timeout,
            )
            writer.close()
            with suppress(OSError, asyncio.TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), timeout=0.2)
            return True
        except (OSError, asyncio.TimeoutError):
            continue
    return False


def _redact_url(url: str) -> str:
    """Strip credentials and query/fragment before logging an MCP URL.

    Server URLs may embed secrets (``https://user:token@host/sse`` or a
    ``?token=`` query). Some deployments also put opaque tokens in the path, so
    log only the origin and a path placeholder.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        hostname = parts.hostname or ""
        netloc = f"[{hostname}]" if ":" in hostname else hostname
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        path = "/..." if parts.path and parts.path != "/" else parts.path
        return urllib.parse.urlunsplit((parts.scheme, netloc, path, "", ""))
    except Exception:
        return "<redacted-url>"


def _pinned_transport_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"transport": PinnedDNSAsyncTransport()}
    mounts = httpx_env_proxy_mounts()
    if mounts:
        kwargs["mounts"] = mounts
    return kwargs


async def _validate_mcp_request_url(request: httpx.Request) -> None:
    """Validate each outgoing MCP HTTP request, including redirect targets."""
    ok, error = validate_url_target(str(request.url))
    if not ok:
        raise httpx.RequestError(
            f"Blocked unsafe MCP URL {_redact_url(str(request.url))} ({error})",
            request=request,
        )


def _windows_command_basename(command: str) -> str:
    """Return the lowercase basename for a Windows command or path."""
    return command.replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()


def _normalize_windows_stdio_command(
    command: str,
    args: list[str] | None,
    env: dict[str, str] | None,
) -> tuple[str, list[str], dict[str, str] | None]:
    """Wrap Windows shell launchers so MCP stdio servers start reliably."""
    normalized_args = list(args or [])
    if os.name != "nt":
        return command, normalized_args, env

    basename = _windows_command_basename(command)
    if basename in {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        return command, normalized_args, env

    if basename.endswith((".exe", ".com")):
        return command, normalized_args, env

    resolved = shutil.which(command, path=(env or {}).get("PATH")) or command
    resolved_basename = _windows_command_basename(resolved)
    should_wrap = (
        basename in _WINDOWS_SHELL_LAUNCHERS
        or basename.endswith((".cmd", ".bat"))
        or resolved_basename.endswith((".cmd", ".bat"))
    )
    if not should_wrap:
        return command, normalized_args, env

    comspec = (env or {}).get("COMSPEC") or os.environ.get("COMSPEC") or "cmd.exe"
    return comspec, ["/d", "/c", command, *normalized_args], env


def _extract_nullable_branch(options: Any) -> tuple[dict[str, Any], bool] | None:
    """Return the single non-null branch for nullable unions."""
    if not isinstance(options, list):
        return None

    non_null: list[dict[str, Any]] = []
    saw_null = False
    for option in cast(list[object], options):
        if not isinstance(option, dict):
            return None
        option_schema = cast(dict[str, Any], option)
        if option_schema.get("type") == "null":
            saw_null = True
            continue
        non_null.append(option_schema)

    if saw_null and len(non_null) == 1:
        return non_null[0], True
    return None


def _resolve_local_schema_ref(root: dict[str, Any], ref: str) -> Any:
    """Resolve a local JSON Pointer without accepting remote references."""
    if not ref.startswith("#"):
        raise ValueError("not a local JSON Pointer")

    pointer = urllib.parse.unquote(ref[1:], errors="strict")
    if not pointer:
        return root
    if not pointer.startswith("/"):
        raise ValueError("not a local JSON Pointer")

    current: Any = root
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = cast(dict[str, Any], current)[part]
        elif isinstance(current, list):
            current = cast(list[Any], current)[int(part)]
        else:
            raise KeyError(part)
    return current


def _rewrite_local_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Hoist arbitrary local JSON-Pointer refs into provider-compatible ``$defs``."""
    rewritten_refs: dict[str, str] = {}
    generated_defs: dict[str, Any] = {}

    def rewrite(value: Any) -> Any:
        if isinstance(value, list):
            return [rewrite(item) for item in cast(list[Any], value)]
        if not isinstance(value, dict):
            return value

        rewritten = dict(cast(dict[str, Any], value))
        raw_ref = rewritten.get("$ref")
        ref = raw_ref if isinstance(raw_ref, str) else None
        is_rewritable_ref = False
        if ref is not None and not ref.startswith("#/$defs/"):
            try:
                pointer = urllib.parse.unquote(ref[1:], errors="strict")
            except (UnicodeDecodeError, ValueError):
                pass
            else:
                is_rewritable_ref = ref.startswith("#") and (
                    not pointer or pointer.startswith("/")
                )
        if is_rewritable_ref:
            assert ref is not None
            name = rewritten_refs.get(ref)
            if name is None:
                try:
                    target = _resolve_local_schema_ref(schema, ref)
                except (KeyError, IndexError, TypeError, UnicodeDecodeError, ValueError):
                    logger.warning("MCP tool schema contains an unresolved local $ref: {}", ref)
                else:
                    name = f"ref_{hashlib.sha256(ref.encode()).hexdigest()[:12]}"
                    existing_defs = schema.get("$defs")
                    while isinstance(existing_defs, dict) and name in existing_defs:
                        name += "_"
                    rewritten_refs[ref] = name
                    # Reserve the name before descending so recursive refs terminate.
                    generated_defs[name] = {}
                    generated_defs[name] = rewrite(target)
            if name is not None:
                rewritten["$ref"] = f"#/$defs/{name}"

        return {key: rewrite(item) for key, item in rewritten.items()}

    result = cast(dict[str, Any], rewrite(schema))
    if generated_defs:
        existing_defs = result.get("$defs")
        result["$defs"] = {
            **(existing_defs if isinstance(existing_defs, dict) else {}),
            **generated_defs,
        }
    return result


def _normalize_nullable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize nullable forms in structural subschemas only."""
    normalized = dict(schema)
    raw_type = normalized.get("type")
    if isinstance(raw_type, list):
        type_values = cast(list[Any], raw_type)
        non_null = [item for item in type_values if item != "null"]
        if "null" in type_values and len(non_null) == 1:
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

    properties = normalized.get("properties")
    if isinstance(properties, dict):
        property_schemas = cast(dict[str, Any], properties)
        normalized["properties"] = {
            name: (
                _normalize_nullable_schema(cast(dict[str, Any], prop))
                if isinstance(prop, dict)
                else prop
            )
            for name, prop in property_schemas.items()
        }
    items = normalized.get("items")
    if isinstance(items, dict):
        normalized["items"] = _normalize_nullable_schema(cast(dict[str, Any], items))
    definitions = normalized.get("$defs")
    if isinstance(definitions, dict):
        definition_schemas = cast(dict[str, Any], definitions)
        normalized["$defs"] = {
            name: _normalize_nullable_schema(cast(dict[str, Any], definition))
            if isinstance(definition, dict)
            else definition
            for name, definition in definition_schemas.items()
        }

    if normalized.get("type") == "object":
        normalized.setdefault("properties", {})
        normalized.setdefault("required", [])
    return normalized


_SUPPORTED_MCP_SCHEMA_KEYS = frozenset({
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "minItems", "maxItems",
    "nullable", "description", "title", "default", "$ref", "$schema",
    "definitions", "$defs", "anyOf", "oneOf", "allOf",
    "format", "pattern", "const", "examples",
})


def _validate_mcp_schema(schema: Any, *, path: str = "") -> str | None:
    """Validate the structure and supported keywords of an MCP input schema."""
    if not isinstance(schema, dict):
        return f"{path or 'schema'} must be an object"
    mapping = cast(dict[str, Any], schema)
    for key in mapping:
        if key not in _SUPPORTED_MCP_SCHEMA_KEYS:
            return f"{path or 'schema'} uses unsupported keyword {key!r}"
    if "$schema" in mapping:
        dialect = mapping["$schema"]
        if not isinstance(dialect, str) or not dialect.strip():
            return f"{path or 'schema'} requires a non-empty string '$schema'"
    for key in ("multipleOf", "exclusiveMinimum", "exclusiveMaximum"):
        error = invalid_numeric_limit(mapping, key, path=path)
        if error:
            return error
    raw_type = mapping.get("type")
    if raw_type is not None:
        types: list[Any] = (
            list(cast(list[Any], raw_type))
            if isinstance(raw_type, list) else [raw_type]
        )
        if any(not isinstance(item, str) or item not in {
            "string", "integer", "number", "boolean", "array", "object", "null",
        } for item in types):
            return f"{path or 'schema'} uses an unsupported type"
    for key, value in mapping.items():
        if key in {"properties", "definitions", "$defs"} and isinstance(value, dict):
            for name, child in cast(dict[str, Any], value).items():
                error = _validate_mcp_schema(child, path=f"{path}.{name}" if path else name)
                if error:
                    return error
        elif key == "items":
            error = _validate_mcp_schema(value, path=f"{path}[]" if path else "[]")
            if error:
                return error
        elif key == "additionalProperties" and isinstance(value, dict):
            error = _validate_mcp_schema(value, path=f"{path}.*" if path else "*")
            if error:
                return error
        elif key in {"anyOf", "oneOf", "allOf"} and isinstance(value, list):
            branches = cast(list[Any], value)
            for index, branch in enumerate(branches):
                error = _validate_mcp_schema(branch, path=f"{path}.{key}[{index}]")
                if error:
                    return error
    return None


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    """Normalize MCP JSON Schema patterns for tool definitions."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    schema_mapping = cast(dict[str, Any], schema)
    return _normalize_nullable_schema(_rewrite_local_schema_refs(schema_mapping))


class MCPGenerationLease:
    """Keep sampled and executing users of one generation alive through settlement."""

    def __init__(self) -> None:
        self._inflight = 0
        self._sampled = 0
        self._retired = False
        self._revoked = False
        self._closed = False
        self._closed_event = asyncio.Event()
        self._closer: Callable[[], Awaitable[None]] | None = None
        self._authorized: Callable[[], bool] | None = None
        self._close_task: asyncio.Task[None] | None = None

    def bind(self, closer: Callable[[], Awaitable[None]]) -> None:
        self._closer = closer

    def set_authorization(self, authorized: Callable[[], bool]) -> None:
        self._authorized = authorized

    def dispatch_error(self) -> str | None:
        if self._authorized is not None and not self._authorized():
            self.revoke()
        if self._revoked or self._closed:
            return "the bound connection generation is revoked or closed"
        return None

    def revoke(self) -> None:
        self._revoked = True

    def acquire_sampled(self) -> bool:
        if self._closed:
            return False
        self._sampled += 1
        return True

    def release_sampled(self) -> None:
        if self._sampled > 0:
            self._sampled -= 1
        self._close_when_idle()

    def begin(self) -> bool:
        if self.dispatch_error() is not None:
            return False
        self._inflight += 1
        return True

    def _close_when_idle(self) -> None:
        if self._retired and self._inflight == self._sampled == 0 and not self._closed:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())

    def end(self) -> None:
        if self._inflight > 0:
            self._inflight -= 1
        self._close_when_idle()

    async def retire(self) -> None:
        """Close now when idle; otherwise retain the generation through its last hold."""
        self._retired = True
        if self._inflight == self._sampled == 0:
            await self.close_once()

    async def close_once(self) -> None:
        self._close_when_idle()
        if self._close_task is not None:
            await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        try:
            closer = self._closer
            if closer is not None:
                await closer()
        finally:
            self._closed_event.set()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()


def _admitted_generation(execute_method: Any) -> Any:
    """Count one admitted call against the wrapper's published generation."""

    @functools.wraps(execute_method)
    async def wrapper(self: Any, **kwargs: Any) -> Any:
        lease = getattr(self, "_generation_lease", None)
        error = self.binding_error()
        if error is not None:
            return ToolResult.error(error)
        if isinstance(lease, MCPGenerationLease):
            if not lease.begin():
                return ToolResult.error("MCP call refused: the bound connection generation ended")
        try:
            return await execute_method(self, **kwargs)
        finally:
            if isinstance(lease, MCPGenerationLease):
                lease.end()

    return wrapper


class _MCPWrapperBase(Tool):
    """Common reconnect handling for wrappers bound to one MCP server session."""

    _plugin_discoverable = False
    _session: ClientSession
    _server_name: str
    _name: str
    _generation_lease: MCPGenerationLease | None = None
    _binding_ended: bool = False
    prompt_instructions: str | None = None

    def _set_mcp_connection(self, session: ClientSession, server_name: str) -> None:
        self._session = session
        self._server_name = server_name
        self._reconnect: _ReconnectCallback | None = None

    @property
    def server_name(self) -> str:
        """Configured MCP server this wrapper belongs to."""
        return self._server_name

    def set_reconnect_handler(self, reconnect: _ReconnectCallback) -> None:
        self._reconnect = reconnect

    def bind_generation(self, lease: MCPGenerationLease) -> None:
        """Bind this wrapper to its published connection generation."""
        self._generation_lease = lease
        self._binding_ended = False

    def binding_error(self) -> str | None:
        if self._binding_ended:
            return "MCP call refused: the bound connection generation ended"
        if self._generation_lease is not None:
            error = self._generation_lease.dispatch_error()
            if error is not None:
                return f"MCP call refused: {error}"
        return None

    @property
    def generation_lease(self) -> MCPGenerationLease | None:
        return self._generation_lease

    async def _refresh_session_after_termination(
        self,
        exc: BaseException,
        already_refreshed: bool,
        capability_kind: str,
    ) -> bool:
        if already_refreshed or not _is_session_terminated(exc) or self._reconnect is None:
            return False
        logger.warning(
            "MCP {} '{}' session terminated; reconnecting server '{}' for future calls",
            capability_kind,
            self._name,
            self._server_name,
        )
        # Publish a replacement generation for later model requests, but keep
        # this admitted wrapper on its original binding. A call must not be
        # retried by tool name against a different schema or session.
        await self._reconnect(self._server_name, self._name, self)
        self._binding_ended = True
        logger.warning(
            "MCP {} '{}' admitted generation ended; refusing retry on replacement",
            capability_kind,
            self._name,
        )
        return False


@dataclass(frozen=True, slots=True, kw_only=True)
class KohakuMediaStopSettlement(ToolOutcome):
    """Exact terminal proof from the configured Comfy video wrapper."""

    call_id: str
    disposition: Literal["stopped", "completed"]
    result_is_error: bool


def _kohaku_media_terminal_result(
    result: Any,
    call_id: str,
) -> tuple[Literal["stopped", "completed"], bool] | None:
    """Accept proof only when the original call reply matches its request ID."""
    structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, Mapping):
        return None
    content = cast(Mapping[str, Any], structured)
    if content.get("call_id") != call_id or content.get("stop_status") != "settled":
        return None
    disposition = content.get("outcome")
    if disposition not in ("stopped", "completed"):
        return None
    return disposition, getattr(result, "isError", False) is True


def is_kohaku_media_stop_settlement(
    result: Any,
    tool: Tool | None,
) -> TypeGuard[KohakuMediaStopSettlement]:
    """Accept the terminal type only from the exact eligible MCP wrapper."""
    return (
        type(tool) is MCPToolWrapper
        and tool.kohaku_media_stop_enabled
        and type(result) is KohakuMediaStopSettlement
        and result.stop_reason == _KOHAKU_MEDIA_SETTLE_STOP_REASON
    )


def mcp_generation_lease(tool: Tool) -> MCPGenerationLease | None:
    """Expose a concrete MCP wrapper's existing resource owner to request capture."""
    return tool.generation_lease if isinstance(tool, _MCPWrapperBase) else None


def mcp_generation_leases(tool: Tool) -> tuple[MCPGenerationLease, ...]:
    """Expose all generation leases owned by an MCP tool or meta-tool."""
    if isinstance(tool, _MCPWrapperBase):
        return (tool.generation_lease,) if tool.generation_lease is not None else ()
    if isinstance(tool, MCPMetaTool):
        seen: list[MCPGenerationLease] = []
        for server in tool.servers.values():
            if server.lease is not None and server.lease not in seen:
                seen.append(server.lease)
        return tuple(seen)
    return ()


def mcp_binding_error(tool: Tool) -> str | None:
    """Recheck the captured MCP wrapper without name-based session lookup."""
    return tool.binding_error() if isinstance(tool, _MCPWrapperBase) else None


def _image_block_data_url(block: Any, types: Any) -> str | None:
    """Return a base64 ``data:`` URL for an MCP image-bearing content block.

    Handles ``ImageContent`` directly and ``EmbeddedResource`` wrapping a binary
    blob with an ``image/*`` MIME type. Returns ``None`` for anything else.
    ``getattr`` guards keep this safe when the installed/faked ``mcp`` SDK does
    not expose a given type.
    """
    image_cls = getattr(types, "ImageContent", None)
    if image_cls is not None and isinstance(block, image_cls):
        mime = getattr(block, "mimeType", None) or "image/png"
        return f"data:{mime};base64,{block.data}"

    embedded_cls = getattr(types, "EmbeddedResource", None)
    blob_cls = getattr(types, "BlobResourceContents", None)
    if embedded_cls is not None and isinstance(block, embedded_cls):
        resource = getattr(block, "resource", None)
        if blob_cls is not None and isinstance(resource, blob_cls):
            blob_resource = cast(Any, resource)
            mime = getattr(blob_resource, "mimeType", None) or ""
            if isinstance(mime, str) and mime.startswith("image/"):
                return f"data:{mime};base64,{blob_resource.blob}"
    return None


def _mcp_media_tool_result(
    text_parts: list[str],
    artifacts: list[dict[str, Any]],
    *,
    reference_images: Sequence[str] = (),
) -> str:
    """Build the compact tool result for an MCP call that returned media.

    The base64 stays out of the model context entirely — only artifact paths and
    metadata are returned, so the result is small and the channel can deliver the
    saved file via the message tool.
    """
    payload: dict[str, Any] = {
        "artifacts": artifacts,
        "next_step": (
            "These media files were returned by an MCP tool and saved as local artifacts. "
            "Use the paths in 'artifacts' for the message tool's 'media' parameter. "
            "Use 'reference_images' to review the source images. To copy or inspect a saved "
            "file inside a Sandbox command, pass its exact path in the exec tool's "
            "'input_files' list and write only to an authorized workspace folder. "
            "Do not paste base64 or raw paths into your reply unless the user asks "
            "for debug details."
        ),
    }
    if reference_images:
        payload["reference_images"] = list(reference_images)
    text = "\n".join(part for part in text_parts if part)
    if text:
        payload["text"] = text
    return json.dumps(payload, ensure_ascii=False)


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
    trusted_for_heuristics = _is_trusted_local_mcp(transport_type, server_url)
    if read_only_hint is not None:
        # A remote annotation is a hint, not proof. An untrusted server cannot
        # authorize a read-only child by declaring its tool non-mutating.
        if read_only_hint and not trusted_for_heuristics:
            return False, "annotation_untrusted", False
        return read_only_hint, "annotation", trusted_for_heuristics

    if trusted_for_heuristics:
        lowered_name = tool_name.lower()
        if lowered_name.startswith(_READ_ONLY_PREFIXES):
            return True, "heuristic", True
        if lowered_name.startswith(_MUTATING_PREFIXES):
            return False, "heuristic", True
        return False, "heuristic_default_mutating", True

    return False, "default_mutating", False

def is_read_only_mcp_tool(tool: Tool) -> bool:
    """Return whether a wrapped MCP tool is classified as read-only."""
    return isinstance(tool, MCPMetaTool) or isinstance(tool, MCPToolWrapper) and tool.is_read_only

class MCPToolWrapper(_MCPWrapperBase):
    """Wraps a single MCP server tool as a nanobot Tool."""

    _plugin_discoverable = False
    _reconnected_tool: MCPToolWrapper | None = None

    def __init__(
        self,
        session: ClientSession,
        server_name: str,
        tool_def: MCPToolDefinition,
        tool_timeout: int = 30,
        *,
        transport_type: str = "stdio",
        server_url: str | None = None,
        server_command: str | None = None,
        media_output_roots: Sequence[Path] = (),
        generation: str = "",
        kohaku_media_cancel_available: bool = False,
        wrapped_name: str | None = None,
        normalized_schema: dict[str, Any] | None = None,
        defer_loading: bool = True,
    ):
        self._set_mcp_connection(session, server_name)
        self.generation = generation
        self._kohaku_media_stop_enabled = (
            server_name == "comfy"
            and transport_type == "stdio"
            and tool_def.name == "generate_video"
            and kohaku_media_cancel_available
        )
        self._media_output_roots = tuple(
            Path(root).expanduser() for root in media_output_roots
        )
        self._original_name = tool_def.name
        self._name = wrapped_name or _sanitize_mcp_tool_name(f"mcp_{server_name}_{tool_def.name}")
        self._description = (tool_def.description or tool_def.name)[:_MAX_DESCRIPTION_CHARS]
        self.exposure = "deferred" if defer_loading else "direct"
        self.namespace = f"mcp:{server_name.lower()}"
        raw_schema = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._parameters: dict[str, Any] = (
            normalized_schema if normalized_schema is not None
            else _normalize_schema_for_openai(raw_schema)
        )
        if len(json.dumps(self._parameters, ensure_ascii=False, separators=(",", ":"))) > _MAX_SCHEMA_CHARS:
            warning = " Input schema omitted because it exceeds the size limit."
            self._description = self._description[:_MAX_DESCRIPTION_CHARS - len(warning)] + warning
            self._parameters = {"type": "object", "properties": {}, "additionalProperties": True}
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

    def disambiguate_name(self) -> None:
        """Keep the original remote name while disambiguating the local schema."""
        self._name = _collision_name(self._name, f"{self.server_name}/{self._original_name}")
        self._mcp_metadata = replace(self._mcp_metadata, wrapped_name=self._name)

    def admit_reconnected_tool(self, replacement: Tool | None) -> None:
        """Link only an equivalent capability published by a transport reconnect."""
        if (
            isinstance(replacement, MCPToolWrapper)
            and self.mcp_metadata == replacement.mcp_metadata
            and self.exposure == replacement.exposure
            and self.parameters == replacement.parameters
            and self.description == replacement.description
        ):
            self._reconnected_tool = replacement

    def accepts_reconnected_tool(self, candidate: Tool) -> bool:
        successor = self._reconnected_tool
        while successor is not None:
            if successor is candidate:
                return True
            successor = successor._reconnected_tool
        return False

    @property
    def mcp_metadata(self) -> MCPToolMetadata:
        return self._mcp_metadata

    @property
    def is_read_only(self) -> bool:
        return self._mcp_metadata.read_only

    @property
    def kohaku_media_stop_enabled(self) -> bool:
        return self._kohaku_media_stop_enabled

    @property
    def read_only(self) -> bool:
        return self.is_read_only

    @property
    def supports_parallel_calls(self) -> bool:
        return self._mcp_metadata.supports_parallel_calls

    @property
    def exclusive(self) -> bool:
        return not self.supports_parallel_calls

    async def _attempt_media_cancel(self, call_id: str) -> Any | None:
        """Send one bounded control request without treating its ACK as proof."""
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(
                    _KOHAKU_MEDIA_CONTROL_TOOL,
                    arguments={"call_id": call_id},
                ),
                timeout=_KOHAKU_MEDIA_CONTROL_TIMEOUT_SECONDS,
            )
            structured = getattr(result, "structuredContent", None)
            if not isinstance(structured, Mapping):
                logger.debug("Comfy media cancel request for '{}' returned an invalid ACK", call_id)
            else:
                ack = cast(Mapping[str, Any], structured)
                if ack.get("call_id") != call_id or type(ack.get("accepted")) is not bool:
                    logger.debug("Comfy media cancel request for '{}' returned an invalid ACK", call_id)
                elif ack["accepted"] is False:
                    logger.debug("Comfy media cancel request for '{}' was not accepted", call_id)
            return result
        except Exception as exc:
            logger.debug(
                "Comfy media cancel request for '{}' did not settle: {}",
                call_id,
                type(exc).__name__,
            )
            return None

    async def _settle_cancelled_media_call(
        self,
        original_call: asyncio.Task[Any],
        call_id: str,
        cancellation: asyncio.CancelledError,
        *,
        arguments: Mapping[str, Any],
    ) -> KohakuMediaStopSettlement:
        """Wait for exact original-call proof after attempting the bound cancel RPC."""
        control = asyncio.create_task(self._attempt_media_cancel(call_id))
        pending = {original_call, control}
        while pending:
            try:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                if not task_is_cancelling():
                    raise
                continue
            pending.difference_update(cast(set[asyncio.Task[Any]], done))
        with suppress(asyncio.CancelledError, Exception):
            control.result()
        try:
            original_result = original_call.result()
        except (asyncio.CancelledError, Exception):
            raise cancellation
        terminal = _kohaku_media_terminal_result(original_result, call_id)
        if terminal is None:
            raise cancellation
        disposition, result_is_error = terminal
        return self._render_media_stop_settlement(
            original_result,
            arguments,
            call_id=call_id,
            disposition=disposition,
            result_is_error=result_is_error,
        )

    @staticmethod
    def _media_status_content(
        rendered: str,
        *,
        call_id: str,
        disposition: Literal["stopped", "completed"],
        result_is_error: bool,
    ) -> str:
        try:
            raw_payload = json.loads(rendered)
        except (ValueError, TypeError):
            raw_payload: Any = {"result": rendered}
        payload: dict[str, Any]
        if isinstance(raw_payload, dict):
            payload = cast(dict[str, Any], raw_payload)
        else:
            payload = {"result": raw_payload}
        payload["media_call"] = {
            "call_id": call_id,
            "stop_status": "settled",
            "outcome": disposition,
            "result_is_error": result_is_error,
            "summary": (
                "The video completed before Stop took effect. Its output files remain available."
                if disposition == "completed"
                else "ComfyUI confirmed the video stopped after cleanup and GPU restoration."
            ),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _render_media_stop_settlement(
        self,
        result: Any,
        arguments: Mapping[str, Any],
        *,
        call_id: str,
        disposition: Literal["stopped", "completed"],
        result_is_error: bool,
    ) -> KohakuMediaStopSettlement:
        original_content = getattr(result, "content", None)
        content: list[Any] = (
            cast(list[Any], original_content) if isinstance(original_content, list) else []
        )
        rendered, artifacts = self._render_call_result(content, arguments)
        from kohaku.utils.artifacts import media_artifact_ref

        status_content = self._media_status_content(
            rendered,
            call_id=call_id,
            disposition=disposition,
            result_is_error=result_is_error,
        )
        return KohakuMediaStopSettlement(
            content=status_content,
            stop_reason=_KOHAKU_MEDIA_SETTLE_STOP_REASON,
            media_artifacts=tuple(media_artifact_ref(artifact) for artifact in artifacts),
            call_id=call_id,
            disposition=disposition,
            result_is_error=result_is_error,
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

    def _mcp_progress_callback(self) -> Any:
        """Build a callback that forwards MCP progress to the live UI.

        The MCP SDK attaches ``_meta.progressToken`` only when a callback is
        present, so servers such as the ComfyUI harness start sending
        ``notifications/progress`` frames. Frames stay live-only: no transcript
        row is written for them.
        """
        sink = current_tool_progress_sink()
        if sink is None:
            return None
        throttle = ProgressThrottle()

        async def on_progress(
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            if not throttle.should_emit(total, message):
                return
            with suppress(Exception):
                await sink.emit(progress, total, message)

        return on_progress

    @_admitted_generation
    async def execute(self, **kwargs: Any) -> Any:
        retried_transient = False
        refreshed_session = False
        progress_callback = self._mcp_progress_callback()
        call_options: dict[str, Any] = (
            {"progress_callback": progress_callback}
            if progress_callback is not None
            else {}
        )
        while True:
            try:
                if self._kohaku_media_stop_enabled:
                    call_id = str(uuid.uuid4())
                    original_call = asyncio.create_task(
                        asyncio.wait_for(
                            self._session.call_tool(
                                self._original_name,
                                arguments=kwargs,
                                meta={_KOHAKU_MEDIA_CALL_META_KEY: call_id},
                                **call_options,
                            ),
                            timeout=self._tool_timeout,
                        )
                    )
                    try:
                        result = await asyncio.shield(original_call)
                    except asyncio.CancelledError as cancellation:
                        if not task_is_cancelling():
                            raise
                        return await self._settle_cancelled_media_call(
                            original_call,
                            call_id,
                            cancellation,
                            arguments=kwargs,
                        )
                else:
                    result = await asyncio.wait_for(
                        self._session.call_tool(
                            self._original_name,
                            arguments=kwargs,
                            **call_options,
                        ),
                        timeout=self._tool_timeout,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP tool '{}' timed out after {}s", self._name, self._tool_timeout
                )
                return ToolResult.error(
                    f"(MCP tool call timed out after {self._tool_timeout}s)"
                )
            except asyncio.CancelledError:
                # MCP SDK's anyio cancel scopes can leak CancelledError on timeout/failure.
                # Re-raise only if our task was externally cancelled (e.g. /stop).
                if task_is_cancelling():
                    raise
                logger.warning("MCP tool '{}' was cancelled by server/SDK", self._name)
                return ToolResult.error("(MCP tool call was cancelled)")
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "tool",
                ):
                    refreshed_session = True
                    continue
                if self._binding_ended:
                    return ToolResult.error(
                        "(MCP tool call refused: the bound connection generation ended)"
                    )
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP tool '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)  # Brief backoff before retry
                        continue
                    # Second transient failure — give up with retry-specific message
                    logger.exception(
                        "MCP tool '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return ToolResult.error(
                        f"(MCP tool call failed after retry: {type(exc).__name__})"
                    )
                logger.exception(
                    "MCP tool '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return ToolResult.error(
                    f"(MCP tool call failed: {type(exc).__name__})"
                )
            else:
                # Success — extract text and persist any image content as artifacts.
                try:
                    rendered, media_artifacts = self._render_call_result(result.content, kwargs)
                    if getattr(result, "isError", False):
                        return ToolResult.error(rendered)
                    if media_artifacts:
                        from kohaku.agent.turn import ToolOutcome
                        from kohaku.utils.artifacts import media_artifact_ref

                        return ToolOutcome(
                            content=rendered,
                            media_artifacts=tuple(
                                media_artifact_ref(artifact) for artifact in media_artifacts
                            ),
                        )
                    return rendered
                except Exception as exc:
                    logger.exception(
                        "MCP tool '{}' failed while rendering result: {}: {}",
                        self._name,
                        type(exc).__name__,
                        exc,
                    )
                    return ToolResult.error(
                        f"(MCP tool returned malformed content: {type(exc).__name__})"
                    )

    def _render_call_result(
        self,
        content: Any,
        arguments: Mapping[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Turn MCP content blocks into a tool result string.

        Text is concatenated as before. Image blocks are decoded and saved as
        local artifacts (mirroring the built-in image generation tool) so the
        model can deliver them via the message tool instead of trying to forward
        base64 — which would be truncated and bloat the context window.
        """
        from mcp import types

        text_parts: list[str] = []
        artifacts: list[dict[str, Any]] = []
        inline_images: list[dict[str, Any] | None] = []
        resource_link_type = getattr(types, "ResourceLink", None)
        for block in content:
            if isinstance(block, types.TextContent):
                text_parts.append(block.text)
                continue
            data_url = _image_block_data_url(block, types)
            if data_url is not None:
                stored = self._store_image_block(data_url, arguments)
                inline_images.append(stored)
                if stored is not None:
                    artifacts.append(stored)
                else:
                    text_parts.append("(MCP tool returned an image that could not be stored)")
                continue
            if resource_link_type is not None and isinstance(block, resource_link_type):
                stored = self._store_resource_link(block, arguments)
                if stored is not None:
                    artifacts.append(stored)
                    text_parts.append(
                        f"(MCP tool returned a {stored['kind']} file saved as an artifact)"
                    )
                else:
                    text_parts.append("(MCP tool returned a file that could not be staged)")
                continue
            text_parts.append(str(block))

        if artifacts:
            from kohaku.utils.artifacts import (
                candidate_artifact_count,
                parse_embedded_text_payload,
                select_candidate_artifacts,
            )

            payload = parse_embedded_text_payload("\n".join(text_parts))
            staged_videos = [
                artifact for artifact in artifacts if artifact.get("kind") == "video"
            ]
            review_images = [
                image["path"] for image in inline_images if image is not None
            ]
            if staged_videos and review_images:
                # Video tools return the clip as a resource link plus sampled
                # review frames. Keep the clip as the primary artifact and expose
                # the frames only as reference images so the assets library stays clean.
                return (
                    _mcp_media_tool_result(
                        text_parts,
                        staged_videos,
                        reference_images=review_images,
                    ),
                    staged_videos,
                )
            candidate_count = candidate_artifact_count(payload)
            if candidate_count is not None and candidate_count <= len(inline_images):
                # Each inline image has one slot. File links can repeat those images.
                candidates = select_candidate_artifacts(inline_images, payload)
                references = [
                    image["path"] for image in inline_images[:-candidate_count] if image is not None
                ]
                return _mcp_media_tool_result(
                    text_parts, candidates, reference_images=references,
                ), candidates
            candidates = select_candidate_artifacts(artifacts, payload)
            return _mcp_media_tool_result(text_parts, artifacts), candidates
        return ("\n".join(text_parts) or "(no output)"), []

    def _store_image_block(
        self, data_url: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Persist one image data URL as an artifact; return its metadata or None."""
        from kohaku.utils.artifacts import ArtifactError, store_generated_image_artifact

        try:
            return store_generated_image_artifact(
                data_url,
                prompt=str(arguments.get("prompt") or ""),
                model=str(arguments.get("model") or arguments.get("workflow") or ""),
                save_dir="generated",
                provider=f"mcp:{self._server_name}",
            )
        except (ArtifactError, OSError) as exc:
            logger.warning(
                "MCP tool '{}' returned an image that could not be stored: {}",
                self._name,
                exc,
            )
            return None

    def _store_resource_link(
        self, block: Any, arguments: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Stage a trusted local MCP resource file into the managed media store."""
        from kohaku.utils.artifacts import ArtifactError, store_generated_file_artifact

        raw_uri = getattr(block, "uri", None)
        # The MCP SDK models ``uri`` as ``pydantic.AnyUrl``, not ``str``.
        uri = str(raw_uri).strip() if raw_uri is not None else ""
        if not uri:
            logger.warning(
                "MCP tool '{}' returned a resource link without a URI", self._name
            )
            return None
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "file":
            logger.debug(
                "MCP tool '{}' returned non-file resource link scheme '{}'",
                self._name,
                parsed.scheme,
            )
            return None
        raw_path = urllib.parse.unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            raw_path = f"//{parsed.netloc}{raw_path}"
        mime = getattr(block, "mimeType", None)
        kind = "video" if isinstance(mime, str) and mime.startswith("video/") else "file"
        try:
            return store_generated_file_artifact(
                Path(raw_path),
                kind=kind,
                prompt=str(arguments.get("prompt") or ""),
                model=str(arguments.get("model") or arguments.get("workflow") or ""),
                provider=f"mcp:{self._server_name}",
                allowed_roots=self._media_output_roots,
                save_dir="generated",
            )
        except (ArtifactError, OSError) as exc:
            logger.warning(
                "MCP tool '{}' returned a file that could not be stored: {}",
                self._name,
                exc,
            )
            return None


class MCPResourceWrapper(_MCPWrapperBase):
    """Wraps an MCP resource URI as a read-only nanobot Tool."""

    _plugin_discoverable = False

    def __init__(
        self,
        session: ClientSession,
        server_name: str,
        resource_def: Resource,
        resource_timeout: int = 30,
    ):
        self._set_mcp_connection(session, server_name)
        self._uri = resource_def.uri
        self._name = _sanitize_mcp_tool_name(f"mcp_{server_name}_resource_{resource_def.name}")
        desc = resource_def.description or resource_def.name
        self._description = f"[MCP Resource] {desc}\nURI: {self._uri}"
        self._parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self._resource_timeout = resource_timeout

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
    def read_only(self) -> bool:
        return True

    @_admitted_generation
    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        retried_transient = False
        refreshed_session = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.read_resource(self._uri),
                    timeout=self._resource_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP resource '{}' timed out after {}s", self._name, self._resource_timeout
                )
                return ToolResult.error(f"(MCP resource read timed out after {self._resource_timeout}s)")
            except asyncio.CancelledError:
                if task_is_cancelling():
                    raise
                logger.warning("MCP resource '{}' was cancelled by server/SDK", self._name)
                return ToolResult.error("(MCP resource read was cancelled)")
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "resource",
                ):
                    refreshed_session = True
                    continue
                if self._binding_ended:
                    return ToolResult.error("(MCP resource read refused: the bound connection generation ended)")
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP resource '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.exception(
                        "MCP resource '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return ToolResult.error(f"(MCP resource read failed after retry: {type(exc).__name__})")
                logger.exception(
                    "MCP resource '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return ToolResult.error(f"(MCP resource read failed: {type(exc).__name__})")
            else:
                parts: list[str] = []
                for block in result.contents:
                    if isinstance(block, types.TextResourceContents):
                        parts.append(block.text)
                    elif isinstance(cast(object, block), types.BlobResourceContents):
                        parts.append(f"[Binary resource: {len(block.blob)} bytes]")
                    else:
                        parts.append(str(block))
                return "\n".join(parts) or "(no output)"


class MCPPromptWrapper(_MCPWrapperBase):
    """Wraps an MCP prompt as a read-only nanobot Tool."""

    _plugin_discoverable = False

    def __init__(
        self,
        session: ClientSession,
        server_name: str,
        prompt_def: Prompt,
        prompt_timeout: int = 30,
    ):
        self._set_mcp_connection(session, server_name)
        self._prompt_name = prompt_def.name
        self._name = _sanitize_mcp_tool_name(f"mcp_{server_name}_prompt_{prompt_def.name}")
        desc = prompt_def.description or prompt_def.name
        self._description = (
            f"[MCP Prompt] {desc}\n"
            "Returns a filled prompt template that can be used as a workflow guide."
        )
        self._prompt_timeout = prompt_timeout

        # Build parameters from prompt arguments
        properties: dict[str, Any] = {}
        required: list[str] = []
        for arg in prompt_def.arguments or []:
            prop: dict[str, Any] = {"type": "string"}
            if getattr(arg, "description", None):
                prop["description"] = arg.description
            properties[arg.name] = prop
            if arg.required:
                required.append(arg.name)
        self._parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

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
    def read_only(self) -> bool:
        return True

    @_admitted_generation
    async def execute(self, **kwargs: Any) -> str:
        from mcp import types
        from mcp.shared.exceptions import McpError

        retried_transient = False
        refreshed_session = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.get_prompt(self._prompt_name, arguments=kwargs),
                    timeout=self._prompt_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP prompt '{}' timed out after {}s", self._name, self._prompt_timeout
                )
                return ToolResult.error(f"(MCP prompt call timed out after {self._prompt_timeout}s)")
            except asyncio.CancelledError:
                if task_is_cancelling():
                    raise
                logger.warning("MCP prompt '{}' was cancelled by server/SDK", self._name)
                return ToolResult.error("(MCP prompt call was cancelled)")
            except McpError as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "prompt",
                ):
                    refreshed_session = True
                    continue
                if self._binding_ended:
                    return ToolResult.error("(MCP prompt call refused: the bound connection generation ended)")
                logger.exception(
                    "MCP prompt '{}' failed: code={} message={}",
                    self._name,
                    exc.error.code,
                    exc.error.message,
                )
                return ToolResult.error(f"(MCP prompt call failed: {exc.error.message} [code {exc.error.code}])")
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "prompt",
                ):
                    refreshed_session = True
                    continue
                if self._binding_ended:
                    return ToolResult.error("(MCP prompt call refused: the bound connection generation ended)")
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP prompt '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.exception(
                        "MCP prompt '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return ToolResult.error(f"(MCP prompt call failed after retry: {type(exc).__name__})")
                logger.exception(
                    "MCP prompt '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return ToolResult.error(f"(MCP prompt call failed: {type(exc).__name__})")
            else:
                parts: list[str] = []
                for message in result.messages:
                    content = message.content
                    if isinstance(content, types.TextContent):
                        parts.append(content.text)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, types.TextContent):
                                parts.append(block.text)
                            else:
                                parts.append(str(block))
                    else:
                        parts.append(str(content))
                return "\n".join(parts) or "(no output)"


_Page = TypeVar("_Page", covariant=True)
_Entry = TypeVar("_Entry")


class _MCPList(Protocol[_Page]):
    @overload
    def __call__(self) -> Awaitable[_Page]: ...

    @overload
    def __call__(self, cursor: str | None, /) -> Awaitable[_Page]: ...


async def _list_mcp_pages(
    request: _MCPList[_Page], entries: Callable[[_Page], list[_Entry]],
) -> list[_Entry]:
    found: list[_Entry] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(_MAX_MCP_PAGES):
        page = await request() if cursor is None else await request(cursor)
        found.extend(entries(page))
        next_cursor: object = getattr(page, "nextCursor", None)
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen:
            break
        cursor = next_cursor
        seen.add(cursor)
    return found


@dataclass
class _MCPExtraServer:
    name: str
    session: ClientSession
    timeout: int
    instructions: str | None
    resources: dict[str, Resource]
    prompts: dict[str, Prompt]
    lease: MCPGenerationLease | None = None
    reconnect: _ReconnectCallback | None = None


class MCPMetaTool(Tool):
    """Expose one shared MCP resource or prompt catalogue."""

    _plugin_discoverable = False
    exposure = "deferred"
    namespace = "mcp"
    search_keywords = ("MCP", "resource", "prompt", "list")

    def __init__(self, kind: Literal["resources", "prompts"], server: _MCPExtraServer) -> None:
        self.kind: Literal["resources", "prompts"] = kind
        self.servers = {server.name: server}

    def accepts_reconnected_tool(self, other: Tool) -> bool:
        return isinstance(other, MCPMetaTool) and other.kind == self.kind

    @property
    def name(self) -> str:
        return f"mcp_{self.kind}"

    @property
    def description(self) -> str:
        if self.kind == "resources":
            return "List or read the resources advertised by an MCP server."
        return "List or get the prompt templates advertised by an MCP server."

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        properties: dict[str, Any] = {"server": {"type": "string"}}
        if self.kind == "resources":
            properties["action"] = {"type": "string", "enum": ["list", "read"]}
            properties["uri"] = {"type": "string"}
        else:
            properties["action"] = {"type": "string", "enum": ["list", "get"]}
            properties["name"] = {"type": "string"}
            properties["arguments"] = {
                "type": "object", "additionalProperties": {"type": "string"},
            }
        return {"type": "object", "properties": properties, "required": ["action", "server"]}

    async def execute(
        self, action: str, server: str, uri: str | None = None,
        name: str | None = None, arguments: dict[str, str] | None = None,
    ) -> str:
        binding = self.servers.get(server)
        if binding is None:
            return ToolResult.error(f"MCP {self.kind} are unavailable for server {server!r}.")
        if binding.lease is not None and (reason := binding.lease.dispatch_error()):
            return ToolResult.error(f"MCP call refused: {reason}")
        if action == "list":
            rows: list[dict[str, object]]
            if self.kind == "resources":
                rows = [{"uri": uri, "name": item.name, "description": (item.description or "")[:512]}
                        for uri, item in binding.resources.items()]
            else:
                rows = [{"name": item.name, "description": (item.description or "")[:512],
                         "arguments": [{"name": arg.name, "required": bool(arg.required)}
                                       for arg in item.arguments or []]}
                        for item in binding.prompts.values()]
            return json.dumps(rows, ensure_ascii=False)
        wrapper: MCPPromptWrapper | MCPResourceWrapper
        params: dict[str, str] = {}
        if self.kind == "resources" and action == "read":
            resource = binding.resources.get(uri or "")
            if resource is None:
                return ToolResult.error("Supply an advertised resource URI.")
            wrapper = MCPResourceWrapper(binding.session, server, resource, binding.timeout)
        elif self.kind == "prompts" and action == "get":
            prompt = binding.prompts.get(name or "")
            if prompt is None:
                return ToolResult.error("Supply an advertised prompt name.")
            wrapper = MCPPromptWrapper(binding.session, server, prompt, binding.timeout)
            params = arguments or {}
            if issues := wrapper.validate_params(params):
                return ToolResult.error("Invalid MCP prompt arguments. " + " ".join(issues))
        else:
            return ToolResult.error(f"Unsupported action {action!r} for MCP {self.kind}.")
        if binding.lease is not None:
            wrapper.bind_generation(binding.lease)
        if binding.reconnect is not None:
            wrapper.set_reconnect_handler(binding.reconnect)
        try:
            return await wrapper.execute(**params)
        except asyncio.CancelledError:
            if task_is_cancelling():
                raise
            return ToolResult.error("MCP request was cancelled.")
        except Exception as exc:
            return ToolResult.error(f"MCP request failed: {type(exc).__name__}.")


def server_prompt_instructions(
    registry: ToolRegistry,
) -> dict[str, tuple[str, set[str]]]:
    """Group initialized server instructions by the admitted wrapper names."""
    servers: dict[str, tuple[str, set[str]]] = {}
    for tool in registry.iter_tools():
        if isinstance(tool, MCPMetaTool):
            for binding in tool.servers.values():
                if binding.instructions:
                    servers.setdefault(binding.name, (binding.instructions, set()))[1].add(tool.name)
        elif isinstance(tool, _MCPWrapperBase) and tool.prompt_instructions:
            _, names = servers.setdefault(
                tool.server_name, (tool.prompt_instructions, set()),
            )
            names.add(tool.name)
    return dict(sorted(servers.items()))


async def connect_mcp_servers(
    mcp_servers: dict[str, MCPServerConfig],
    registry: ToolRegistry,
    *,
    oauth_handlers: Mapping[str, MCPOAuthHandlers] | None = None,
    media_output_roots: Sequence[Path] = (),
    readiness_timeout: float = 10.0,
    data_dir: Path | None = None,
) -> dict[str, MCPConnection]:
    """Connect to configured MCP servers and register their tools, resources, prompts.

    Returns one connection handle per server.  Each handle keeps the task that
    entered the MCP SDK contexts alive so reconnect and shutdown can close
    AnyIO cancel scopes from their owning task.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    published_servers: set[str] = set()
    pending_publications: dict[str, list[Tool]] = {}

    async def open_single_server(
        name: str, cfg: MCPServerConfig, server_stack: AsyncExitStack
    ) -> bool:
        try:
            transport_type = cfg.type
            if not transport_type:
                if cfg.command:
                    transport_type = "stdio"
                elif cfg.url:
                    transport_type = (
                        "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
                    )
                else:
                    logger.warning("MCP server '{}': no command or url configured, skipping", name)
                    return False

            if transport_type in {"sse", "streamableHttp"}:
                ok, error = validate_url_target(cfg.url)
                if not ok:
                    logger.warning(
                        "MCP server '{}': blocked unsafe URL {} ({})",
                        name,
                        _redact_url(cfg.url),
                        error,
                    )
                    return False

            oauth_auth: httpx.Auth | None = None
            if cfg.auth == "oauth":
                if transport_type not in {"sse", "streamableHttp"}:
                    logger.warning(
                        "MCP server '{}': OAuth requires an SSE or Streamable HTTP transport",
                        name,
                    )
                    return False
                from kohaku.agent.tools.mcp_oauth import (
                    MCPAuthorizationRequiredError,
                    create_mcp_oauth_auth,
                )

                try:
                    oauth_auth = await create_mcp_oauth_auth(
                        name,
                        cfg.url,
                        (oauth_handlers or {}).get(name),
                        data_dir=data_dir,
                    )
                except MCPAuthorizationRequiredError:
                    logger.info("MCP server '{}': waiting for browser authorization", name)
                    return False

            if transport_type == "stdio":
                command, args, env = _normalize_windows_stdio_command(
                    cfg.command,
                    cfg.args,
                    cfg.env or None,
                )
                params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env,
                    cwd=cfg.cwd or None,
                )
                read, write = await server_stack.enter_async_context(stdio_client(params))
            elif transport_type == "sse":
                if not await _probe_http_url(cfg.url):
                    logger.warning("MCP server '{}': {} unreachable, skipping", name, _redact_url(cfg.url))
                    return False

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
                        event_hooks={"request": [_validate_mcp_request_url]},
                        follow_redirects=True,
                        timeout=timeout,
                        auth=auth,
                        **_pinned_transport_kwargs(),
                    )

                sse_kwargs: dict[str, Any] = {
                    "httpx_client_factory": httpx_client_factory,
                }
                if oauth_auth is not None:
                    sse_kwargs["auth"] = oauth_auth
                read, write = await server_stack.enter_async_context(
                    sse_client(cfg.url, **sse_kwargs)
                )
            elif transport_type == "streamableHttp":
                if not await _probe_http_url(cfg.url):
                    logger.warning("MCP server '{}': {} unreachable, skipping", name, _redact_url(cfg.url))
                    return False

                http_client_kwargs: dict[str, Any] = {
                    "headers": cfg.headers or None,
                    "event_hooks": {"request": [_validate_mcp_request_url]},
                    "follow_redirects": True,
                    "timeout": httpx.Timeout(30.0, connect=10.0),
                    **_pinned_transport_kwargs(),
                }
                if oauth_auth is not None:
                    http_client_kwargs["auth"] = oauth_auth
                http_client = await server_stack.enter_async_context(
                    httpx.AsyncClient(**http_client_kwargs)
                )
                read, write, _ = await server_stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )
            else:
                logger.warning("MCP server '{}': unknown transport type '{}'", name, transport_type)
                return False

            read = _filter_malformed_mcp_progress_notifications(read, name)
            session = await server_stack.enter_async_context(ClientSession(read, write))
            initialization = await session.initialize()
            raw_instructions: object = getattr(initialization, "instructions", None)
            instructions = (
                TokenMeter.truncate_text(raw_instructions.strip(), 500)
                if isinstance(raw_instructions, str) and raw_instructions.strip() else None
            )
            tool_defs = await _list_mcp_pages(session.list_tools, lambda page: page.tools)
            unique_tools = {tool.name: tool for tool in tool_defs}
            pending_tools: list[Tool] = []
            generation = hashlib.sha256(
                f"{name}:{cfg.type or ''}:{cfg.url or ''}:{cfg.command or ''}".encode("utf-8")
            ).hexdigest()[:16]
            enabled_tools = set(cfg.enabled_tools)
            allow_all_tools = "*" in enabled_tools
            registered_count = 0
            matched_enabled_tools: set[str] = set()
            available_raw_names = list(unique_tools)
            base_names = {
                raw: _sanitize_mcp_tool_name(f"mcp_{name}_{raw}") for raw in unique_tools
            }
            collisions = Counter(base_names.values())
            wrapped_names = {
                raw: _collision_name(base, f"{name}/{raw}") if collisions[base] > 1 else base
                for raw, base in base_names.items()
            }
            available_wrapped_names = list(wrapped_names.values())
            kohaku_media_cancel_available = (
                name == "comfy"
                and transport_type == "stdio"
                and _KOHAKU_MEDIA_CONTROL_TOOL in unique_tools
            )
            for raw, tool_def in unique_tools.items():
                wrapped_name = wrapped_names[raw]
                if tool_def.name == _KOHAKU_MEDIA_CONTROL_TOOL:
                    if enabled_tools and not allow_all_tools and (
                        tool_def.name in enabled_tools or wrapped_name in enabled_tools
                    ):
                        matched_enabled_tools.add(
                            tool_def.name if tool_def.name in enabled_tools else wrapped_name
                        )
                    logger.debug(
                        "MCP: keeping control tool '{}' private on server '{}'",
                        tool_def.name,
                        name,
                    )
                    continue
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
                normalized_schema = _normalize_schema_for_openai(tool_def.inputSchema)
                schema_error = _validate_mcp_schema(normalized_schema)
                if schema_error is not None:
                    logger.warning(
                        "MCP server '{}': refusing tool '{}' schema: {}",
                        name,
                        tool_def.name,
                        schema_error,
                    )
                    continue
                wrapper = MCPToolWrapper(
                    session,
                    name,
                    tool_def,
                    tool_timeout=cfg.tool_timeout,
                    transport_type=transport_type,
                    server_url=cfg.url,
                    server_command=cfg.command,
                    media_output_roots=media_output_roots,
                    generation=generation,
                    kohaku_media_cancel_available=kohaku_media_cancel_available,
                    wrapped_name=wrapped_name,
                    normalized_schema=normalized_schema,
                    defer_loading=cfg.defer_loading,
                )
                wrapper.prompt_instructions = instructions
                pending_tools.append(wrapper)
                registered_count += 1
                matched_enabled_tools.update(
                    enabled_tools.intersection({raw, wrapped_name})
                )

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

            # Only register resources and prompts when no tool restriction is
            # active.  enabledTools is a per-*tool* allowlist; resources and
            # prompts have no equivalent name filter, so they must be skipped
            # whenever the operator specified a tool subset.  An empty list
            # (deny-all) or a list of specific tool names both indicate that
            # the operator intended to restrict capabilities — registering
            # unrestricted resource/prompt wrappers would violate that intent.
            # The default ["*"] (allow-all) means no restriction was intended.
            if allow_all_tools:
                resources: list[Resource] = []
                prompts: list[Prompt] = []
                try:
                    resources = await _list_mcp_pages(session.list_resources, lambda page: page.resources)
                except Exception as e:
                    logger.debug(
                        "MCP server '{}': resources not supported or failed: {}", name, e
                    )

                try:
                    prompts = await _list_mcp_pages(session.list_prompts, lambda page: page.prompts)
                except Exception as e:
                    logger.debug(
                        "MCP server '{}': prompts not supported or failed: {}", name, e
                    )
                if resources or prompts:
                    extras = _MCPExtraServer(
                        name, session, cfg.tool_timeout, instructions,
                        {str(resource.uri): resource for resource in resources},
                        {prompt.name: prompt for prompt in prompts},
                    )
                    pending_tools.extend([MCPMetaTool("resources", extras), MCPMetaTool("prompts", extras)])
                    registered_count += 2
            else:
                logger.info(
                    "MCP server '{}': skipping resource/prompt registration "
                    "(enabledTools does not include '*' — only tools allowed)",
                    name,
                )

            pending_publications[name] = pending_tools
            logger.info(
                "MCP server '{}': connected, {} capabilities registered", name, registered_count
            )
            return True

        except Exception as e:
            hint = ""
            text = str(e).lower()
            if any(
                marker in text
                for marker in (
                    "parse error",
                    "invalid json",
                    "unexpected token",
                    "jsonrpc",
                    "content-length",
                )
            ):
                hint = (
                    " Hint: this looks like stdio protocol pollution. Make sure the MCP server writes "
                    "only JSON-RPC to stdout and sends logs/debug output to stderr instead."
                )
            _log_mcp_connection_failure(name, e, hint)
            return False

    async def connect_single_server(
        name: str, cfg: MCPServerConfig,
    ) -> tuple[str, MCPConnection | None]:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[bool] = loop.create_future()
        close_requested = asyncio.Event()

        async def own_connection() -> None:
            try:
                async with AsyncExitStack() as stack:
                    connected = await open_single_server(name, cfg, stack)
                    if not ready.done():
                        ready.set_result(connected)
                    if connected:
                        await close_requested.wait()
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
                raise

        owner = asyncio.create_task(own_connection(), name=f"mcp:{name}")
        connection = _OwnedMCPConnection(owner, close_requested)
        try:
            connected = await asyncio.wait_for(
                asyncio.shield(ready),
                timeout=readiness_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "MCP server '{}': readiness timed out after {}s",
                name,
                readiness_timeout,
            )
            close_requested.set()
            owner.cancel()
            with suppress(BaseException):
                await asyncio.shield(owner)
            return name, None
        except BaseException as exc:
            close_requested.set()
            owner.cancel()
            with suppress(BaseException):
                await asyncio.shield(owner)
            if isinstance(exc, asyncio.CancelledError) and not task_is_cancelling():
                logger.warning("MCP server '{}': connection cancelled by server/SDK", name)
                return name, None
            raise
        if not connected:
            await connection.aclose()
            return name, None
        return name, connection

    server_stacks: dict[str, MCPConnection] = {}
    slots = asyncio.Semaphore(_MAX_CONNECTIONS)

    async def connect_bounded(name: str, cfg: MCPServerConfig) -> None:
        async with slots:
            _, connection = await connect_single_server(name, cfg)
            if connection is not None:
                server_stacks[name] = connection

    try:
        results = await asyncio.gather(
            *(connect_bounded(name, cfg) for name, cfg in mcp_servers.items()),
            return_exceptions=True,
        )
        for name, result in zip(mcp_servers, results):
            if isinstance(result, Exception):
                _log_mcp_connection_failure(name, result)
            elif isinstance(result, BaseException):
                raise result
        # Use configuration-independent order for cross-server name collisions.
        for name in sorted(server_stacks):
            try:
                _publish_mcp_tools(registry, pending_publications[name])
            except Exception as exc:
                _log_mcp_connection_failure(name, exc)
                await _close_mcp_connection(name, server_stacks[name])
                server_stacks.pop(name)
            else:
                published_servers.add(name)
    except BaseException:
        # Callers can bound readiness/reload with a timeout. If cancellation
        # interrupts a later server, ownership of earlier connections has not
        # transferred yet, so roll the whole batch back before propagating it.
        for name in published_servers:
            _unregister_server_tools(registry, name)
        try:
            await close_mcp_connections(server_stacks)
        except BaseException as cleanup_exc:
            logger.debug("MCP batch rollback cleanup error (can be ignored): {}", cleanup_exc)
        raise

    return {name: server_stacks[name] for name in mcp_servers if name in server_stacks}


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return persisted session kwargs for MCP preset attachments."""
    mcp_presets = metadata.get("mcp_presets") if isinstance(metadata, Mapping) else None
    return {"mcp_presets": mcp_presets} if isinstance(mcp_presets, list) and mcp_presets else {}


def _configured_servers(config: Config) -> dict[str, MCPServerConfig]:
    from kohaku.agent.plugins import agent_plugin_mcp_servers

    return agent_plugin_mcp_servers(
        config.workspace_path,
        config.tools.mcp_servers,
    )


def _load_current_servers() -> dict[str, MCPServerConfig]:
    from kohaku.config.loader import load_config, resolve_config_env_vars

    return _configured_servers(resolve_config_env_vars(load_config()))


class MCPReadinessAbortedError(RuntimeError):
    """Bounded readiness outcome observed by waiters when the owner is cancelled."""


def _consume_readiness_gate_exception(gate: asyncio.Future[None]) -> None:
    """Retrieve a gate exception so an unawaited bounded outcome stays quiet."""
    if not gate.cancelled():
        gate.exception()


class MCPProvider:
    """Own configured MCP connections and their dynamic tool registrations."""

    def __init__(
        self,
        servers: Mapping[str, MCPServerConfig],
        registry: ToolRegistry,
        *,
        server_loader: MCPServerLoader | None = None,
        media_output_roots: Sequence[Path] = (),
        data_dir: Path | None = None,
    ) -> None:
        self._servers = dict(servers)
        self._data_dir = Path(data_dir) if data_dir is not None else None
        self._registry = registry
        self._server_loader = server_loader or _load_current_servers
        self._media_output_roots = tuple(media_output_roots)
        self._connections: dict[str, MCPConnection] = {}
        self._generation_leases: dict[str, MCPGenerationLease] = {}
        self._retired_generations: set[MCPGenerationLease] = set()
        self._runtime_statuses: dict[str, MCPRuntimeStatus] = {}
        self._lock = asyncio.Lock()
        self._closing = False
        self._connect_inflight: asyncio.Future[None] | None = None
        self._connect_waiters = 0

    @classmethod
    def from_config(
        cls,
        config: Config,
        registry: ToolRegistry,
        *,
        server_loader: MCPServerLoader | None = None,
    ) -> MCPProvider:
        return cls(
            _configured_servers(config),
            registry,
            server_loader=server_loader,
            media_output_roots=[Path(root) for root in config.tools.media_output_roots],
            data_dir=config.runtime_data_dir,
        )

    @property
    def configured_server_names(self) -> set[str]:
        return set(self._servers)

    @property
    def connected_server_names(self) -> set[str]:
        return set(self._connections)

    def runtime_status(self) -> dict[str, MCPRuntimeStatus]:
        """Return the latest connection-attempt result for configured servers."""
        return {
            name: status
            for name, status in self._runtime_statuses.items()
            if name in self._servers
        }

    def _set_runtime_status(
        self,
        server_names: Iterable[str],
        status: MCPRuntimeStatus,
    ) -> None:
        for name in server_names:
            self._runtime_statuses[name] = status

    def _record_connection_result(
        self,
        attempted: Iterable[str],
        connected: Iterable[str],
    ) -> None:
        attempted_names = set(attempted)
        connected_names = set(connected)
        self._set_runtime_status(connected_names, "connected")
        self._set_runtime_status(attempted_names - connected_names, "failed")

    async def connect(self) -> None:
        """Connect configured servers, sharing one initial readiness attempt.

        The first caller owns the connection task so transports that require a
        stable owning task keep working. Concurrent callers await that caller's
        completion gate and observe the same bounded result. Cancellation is
        caller-scoped: the owner's cancellation completes the gate with a bounded
        readiness failure instead of cancelling a waiter's own await.
        """
        inflight = self._connect_inflight
        if inflight is not None and not inflight.done():
            self._connect_waiters += 1
            try:
                await asyncio.shield(inflight)
            finally:
                self._connect_waiters -= 1
            return
        gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        gate.add_done_callback(_consume_readiness_gate_exception)
        self._connect_inflight = gate
        # Let concurrently scheduled callers join this attempt before the
        # owner starts work that may not suspend on a fast mock or cache hit.
        await asyncio.sleep(0)
        try:
            await self._connect_once()
        except asyncio.CancelledError:
            if not gate.done():
                if self._connect_waiters:
                    gate.set_exception(MCPReadinessAbortedError(
                        "MCP readiness was cancelled by its owning caller"
                    ))
                else:
                    gate.cancel()
            raise
        except BaseException as exc:
            if not gate.done():
                if self._connect_waiters:
                    gate.set_exception(exc)
                else:
                    gate.cancel()
            raise
        else:
            if not gate.done():
                if self._connect_waiters:
                    gate.set_result(None)
                else:
                    gate.cancel()
        finally:
            if self._connect_inflight is gate:
                self._connect_inflight = None

    async def _connect_once(self) -> None:
        async with self._lock:
            if self._closing:
                return
            configured_missing = {
                name: cfg
                for name, cfg in self._servers.items()
                if name not in self._connections
            }
            oauth_servers = {
                name: cfg
                for name, cfg in configured_missing.items()
                if cfg.auth == "oauth"
            }
            authorization_pending: set[str] = set()
            if oauth_servers:
                from kohaku.agent.tools.mcp_oauth import mcp_oauth_has_credentials

                authorization_pending = {
                    name
                    for name, cfg in oauth_servers.items()
                    if not mcp_oauth_has_credentials(name, cfg.url, data_dir=self._data_dir)
                }
            for name in authorization_pending:
                self._runtime_statuses.pop(name, None)
            missing_servers = {
                name: cfg
                for name, cfg in configured_missing.items()
                if name not in authorization_pending
            }
            if not missing_servers:
                return
            self._set_runtime_status(missing_servers, "connecting")
            try:
                candidate_registry = ToolRegistry()
                connected = await connect_mcp_servers(
                    missing_servers,
                    candidate_registry,
                    media_output_roots=self._media_output_roots,
                    data_dir=self._data_dir,
                )
                if self._closing:
                    await close_mcp_connections(connected)
                    return
                await self._publish_connections(connected, candidate_registry)
                self._record_connection_result(missing_servers, connected)
                self._attach_generation_leases(connected)
                self._attach_reconnect_handlers(connected)
                if connected:
                    logger.info("MCP connected servers: {}", sorted(connected))
                else:
                    logger.warning(
                        "No MCP servers connected successfully "
                        "(will retry on the next readiness check)"
                    )
            except asyncio.CancelledError:
                self._set_runtime_status(missing_servers, "failed")
                if task_is_cancelling():
                    raise
                logger.warning(
                    "MCP connection cancelled (will retry on the next readiness check)"
                )
            except BaseException as exc:
                self._set_runtime_status(missing_servers, "failed")
                logger.warning(
                    "Failed to connect MCP servers "
                    "(will retry on the next readiness check): {}",
                    exc,
                )

    async def reload(self) -> dict[str, Any]:
        """Reconcile live MCP connections with the current configuration."""
        async with self._lock:
            if self._closing:
                return self._closing_result()
            try:
                next_servers = dict(self._server_loader())
            except Exception as exc:
                logger.warning("MCP hot reload could not read config: {}", exc)
                return {
                    "ok": False,
                    "message": "Could not reload MCP config. Restart nanobot to pick up changes.",
                    "requires_restart": True,
                    "error": str(exc),
                }

            current_servers = dict(self._servers)
            current_names = set(current_servers)
            next_names = set(next_servers)
            from kohaku.agent.tools.mcp_oauth import mcp_oauth_has_credentials

            authorization_pending = {
                name
                for name, cfg in next_servers.items()
                if cfg.auth == "oauth" and not mcp_oauth_has_credentials(name, cfg.url, data_dir=self._data_dir)
            }
            removed = sorted(current_names - next_names)
            added = sorted(next_names - current_names)
            changed = sorted(
                name
                for name in current_names & next_names
                if _server_signature(current_servers[name])
                != _server_signature(next_servers[name])
            )

            tools_removed = 0
            for name in set(removed) | authorization_pending:
                tools_removed += _unregister_server_tools(self._registry, name)
                await self._close_server(name)

            for name in [*removed, *authorization_pending]:
                self._runtime_statuses.pop(name, None)

            self._servers = next_servers

            # Prepare replacement generations privately. Only a ready candidate
            # may retire the old healthy generation for that server.
            changed_failed: set[str] = set()
            changed_ready = set(changed) - authorization_pending
            if changed_ready:
                candidate_registry = ToolRegistry()
                self._set_runtime_status(changed_ready, "connecting")
                try:
                    candidates = await connect_mcp_servers(
                        {name: next_servers[name] for name in changed_ready},
                        candidate_registry,
                        media_output_roots=self._media_output_roots,
                        data_dir=self._data_dir,
                    )
                except BaseException:
                    self._set_runtime_status(changed_ready, "failed")
                    raise
                if self._closing:
                    await close_mcp_connections(candidates)
                    return self._closing_result()
                for name in changed_ready:
                    candidate = candidates.get(name)
                    if candidate is None:
                        changed_failed.add(name)
                        self._set_runtime_status({name}, "failed")
                        continue
                    tools_removed += _unregister_server_tools(self._registry, name)
                    await self._retire_server(name)
                    _publish_mcp_tools(self._registry, list(_server_tools(candidate_registry, name)))
                    self._connections[name] = candidate
                    # No await may expose these wrappers without their real
                    # generation; the next old server may suspend retirement.
                    self._attach_generation_leases({name})
                    self._attach_reconnect_handlers({name})
                    self._set_runtime_status({name}, "connected")

            retry_missing = sorted(
                name
                for name in next_names
                if name not in self._connections
                and name not in set(added) | set(changed)
                and name not in authorization_pending
            )
            to_connect_names = sorted(
                (set(added) | set(retry_missing)) - authorization_pending
            )
            to_connect = {name: next_servers[name] for name in to_connect_names}
            connected: dict[str, MCPConnection] = {}
            if to_connect:
                self._set_runtime_status(to_connect, "connecting")
                candidate_registry = ToolRegistry()
                try:
                    connected = await connect_mcp_servers(
                        to_connect,
                        candidate_registry,
                        media_output_roots=self._media_output_roots,
                        data_dir=self._data_dir,
                    )
                except BaseException:
                    self._set_runtime_status(to_connect, "failed")
                    raise
                if self._closing:
                    await close_mcp_connections(connected)
                    return self._closing_result()
                await self._publish_connections(connected, candidate_registry)
                self._record_connection_result(to_connect, connected)
                self._attach_generation_leases(connected)
                self._attach_reconnect_handlers(connected)

            failed = sorted(set(to_connect) - set(connected) | changed_failed)
            unchanged = not removed and not added and not changed and not retry_missing
            ok = not failed
            if failed:
                message = (
                    "MCP config reloaded, but some servers did not connect: "
                    + ", ".join(failed)
                )
            elif unchanged:
                message = "MCP config is already live."
            elif retry_missing and not added and not changed and not removed:
                message = "MCP connections refreshed without restarting nanobot."
            else:
                message = "MCP config reloaded without restarting nanobot."

            logger.info(
                "MCP hot reload: added={} changed={} removed={} retried={} "
                "connected={} failed={} tools_removed={}",
                added,
                changed,
                removed,
                retry_missing,
                sorted(connected),
                failed,
                tools_removed,
            )
            return {
                "ok": ok,
                "message": message,
                "added": added,
                "changed": changed,
                "removed": removed,
                "retried": retry_missing,
                "connected": sorted(self._connections),
                "configured": sorted(self._servers),
                "failed": failed,
                "tools_removed": tools_removed,
                "requires_restart": False,
            }

    @staticmethod
    def _closing_result() -> dict[str, Any]:
        return {
            "ok": False,
            "message": "MCP connections are shutting down.",
            "requires_restart": True,
        }

    def _attach_reconnect_handlers(self, server_names: Iterable[str]) -> None:
        async def reconnect(
            server_name: str,
            tool_name: str,
            stale_tool: Tool,
        ) -> Tool | None:
            return await self._refresh_terminated_server(
                server_name,
                tool_name,
                stale_tool,
            )

        for server_name in server_names:
            for tool_name in list(self._registry.tool_names):
                tool = self._registry.get(tool_name)
                if not _tool_belongs_to_server(tool, tool_name, server_name):
                    continue
                if isinstance(tool, _MCPWrapperBase):
                    tool.set_reconnect_handler(reconnect)
                elif isinstance(tool, MCPMetaTool):
                    tool.servers[server_name].reconnect = reconnect

    async def _refresh_terminated_server(
        self,
        server_name: str,
        tool_name: str,
        stale_tool: Tool,
    ) -> Tool | None:
        if self._closing:
            return None
        async with self._lock:
            if self._closing:
                return None
            cfg = self._servers.get(server_name)
            if cfg is None:
                logger.warning(
                    "MCP server '{}' session terminated but is no longer configured",
                    server_name,
                )
                return None

            current_tool = self._registry.get(tool_name)
            if (
                current_tool is not None
                and current_tool is not stale_tool
                and server_name in self._connections
            ):
                return current_tool

            logger.warning(
                "MCP server '{}' session terminated; refreshing connection",
                server_name,
            )
            previous = [
                tool for tool in _server_tools(self._registry, server_name)
                if isinstance(tool, MCPToolWrapper)
            ]
            _unregister_server_tools(self._registry, server_name)
            await self._close_server(server_name)

            self._set_runtime_status({server_name}, "connecting")
            candidate_registry = ToolRegistry()
            connected = await connect_mcp_servers(
                {server_name: cfg},
                candidate_registry,
                media_output_roots=self._media_output_roots,
                data_dir=self._data_dir,
            )
            if self._closing:
                await close_mcp_connections(connected)
                return None
            await self._publish_connections(connected, candidate_registry)
            self._record_connection_result({server_name}, connected)
            self._attach_generation_leases(connected)
            self._attach_reconnect_handlers(connected)
            if server_name not in connected:
                logger.warning(
                    "MCP server '{}' reconnect failed after session termination",
                    server_name,
                )
                return None
            for old_tool in previous:
                old_tool.admit_reconnected_tool(self._registry.get(old_tool.name))
            return self._registry.get(tool_name)

    async def _publish_connections(
        self, connections: Mapping[str, MCPConnection], candidates: ToolRegistry,
    ) -> None:
        """Publish only after readiness; attaching generation holds cannot suspend."""
        tools = [tool for tool in candidates.iter_tools()
                 if any(_tool_belongs_to_server(tool, tool.name, name) for name in connections)]
        try:
            _publish_mcp_tools(self._registry, tools)
        except BaseException:
            await close_mcp_connections(connections)
            raise
        self._connections.update(connections)

    def _attach_generation_leases(
        self,
        server_names: Iterable[str],
    ) -> None:
        """Bind newly published wrappers to one retire-on-last-call lease."""
        for server_name in server_names:
            connection = self._connections.get(server_name)
            if connection is None:
                continue
            wrappers = [tool for tool in self._registry.iter_tools()
                        if isinstance(tool, _MCPWrapperBase)
                        and _tool_belongs_to_server(tool, tool.name, server_name)]
            extras = [tool.servers[server_name] for tool in self._registry.iter_tools()
                      if isinstance(tool, MCPMetaTool) and server_name in tool.servers]
            if not wrappers and not extras:
                # Connections with no callable wrappers have no sampled/active
                # users. Preserve their existing same-task close path.
                continue
            lease = MCPGenerationLease()
            cfg = self._servers.get(server_name)
            if cfg is not None and cfg.auth == "oauth":
                def authorized(name: str = server_name, url: str = cfg.url) -> bool:
                    from kohaku.agent.tools.mcp_oauth import mcp_oauth_has_credentials

                    return mcp_oauth_has_credentials(name, url, data_dir=self._data_dir)

                lease.set_authorization(authorized)

            async def closer(
                name: str = server_name,
                conn: MCPConnection = connection,
                generation: MCPGenerationLease = lease,
            ) -> None:
                try:
                    await _close_mcp_connection(name, conn)
                finally:
                    self._retired_generations.discard(generation)

            lease.bind(closer)
            for tool in wrappers:
                tool.bind_generation(lease)
            for binding in extras:
                binding.lease = lease
            self._generation_leases[server_name] = lease

    async def _retire_server(self, server_name: str) -> None:
        """Retire a generation after its admitted calls settle."""
        connection = self._connections.pop(server_name, None)
        lease = self._generation_leases.pop(server_name, None)
        if connection is None:
            return
        if lease is None:
            await _close_mcp_connection(server_name, connection)
            return
        self._retired_generations.add(lease)
        await lease.retire()

    async def _close_server(self, server_name: str) -> None:
        lease = self._generation_leases.get(server_name)
        if lease is not None:
            lease.revoke()
        await self._retire_server(server_name)

    async def aclose(self) -> None:
        """Close every connection while excluding reconnect and hot reload."""
        self._closing = True
        # Fence admission before the lock or any resource cleanup can suspend.
        for lease in {*self._generation_leases.values(), *self._retired_generations}:
            lease.revoke()
        async with self._lock:
            self._runtime_statuses.clear()
            for name in self._servers:
                _unregister_server_tools(self._registry, name)
            cancellation: asyncio.CancelledError | None = None
            for name in list(self._connections):
                try:
                    await self._close_server(name)
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
            if cancellation is not None:
                raise cancellation
            # Retired generations remain owned after replacement/removal and
            # participate in shutdown even when no longer in _connections.
            for lease in tuple(self._retired_generations):
                await lease.wait_closed()


def _server_signature(cfg: Any) -> Any:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump(mode="json")
    return cfg


def _tool_prefix(server_name: str) -> str:
    return _sanitize_name(f"mcp_{server_name}_")


def _tool_belongs_to_server(tool: Tool | None, tool_name: str, server_name: str) -> bool:
    if isinstance(tool, MCPMetaTool):
        return server_name in tool.servers
    if isinstance(tool, _MCPWrapperBase):
        return getattr(tool, "_server_name", None) == server_name
    return tool_name.startswith(_tool_prefix(server_name))


def _server_tools(registry: ToolRegistry, server_name: str) -> Iterable[Tool]:
    for tool in registry.iter_tools():
        if isinstance(tool, MCPMetaTool):
            if binding := tool.servers.get(server_name):
                yield MCPMetaTool(tool.kind, binding)
        elif _tool_belongs_to_server(tool, tool.name, server_name):
            yield tool


def _publish_mcp_tools(registry: ToolRegistry, tools: list[Tool]) -> None:
    existing = {tool.name: tool for tool in registry.iter_tools()}
    merged: list[tuple[MCPMetaTool, MCPMetaTool]] = []
    additions: list[Tool] = []
    for tool in tools:
        present = existing.get(tool.name)
        if isinstance(present, MCPMetaTool) and isinstance(tool, MCPMetaTool):
            merged.append((present, tool))
            continue
        if (
            isinstance(present, MCPToolWrapper) and isinstance(tool, MCPToolWrapper)
            and present.server_name != tool.server_name
        ):
            tool.disambiguate_name()
        if tool.name in existing:
            raise ValueError(f"Duplicate tool registration: {tool.name}")
        existing[tool.name] = tool
        additions.append(tool)
    for tool in additions:
        registry.register(tool)
    for present, tool in merged:
        present.servers.update(tool.servers)


def _unregister_server_tools(registry: ToolRegistry, server_name: str) -> int:
    removed = 0
    for tool_name in list(registry.tool_names):
        tool = registry.get(tool_name)
        if _tool_belongs_to_server(tool, tool_name, server_name):
            if isinstance(tool, MCPMetaTool):
                tool.servers.pop(server_name, None)
                if tool.servers:
                    continue
            registry.unregister(tool_name)
            removed += 1
    return removed
