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
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from urllib.parse import urlparse

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.security.network import (
    PinnedDNSAsyncTransport,
    env_proxy_applies_to_url,
    httpx_env_proxy_mounts,
    resolve_url_target,
    validate_url_target,
)
from nanobot.utils.cancellation import task_is_cancelling

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Prompt, Resource
    from mcp.types import Tool as MCPToolDefinition

    from nanobot.agent.tools.mcp_oauth import MCPOAuthHandlers
    from nanobot.config.schema import Config, MCPServerConfig

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
    "enum", "minimum", "maximum", "minLength", "maxLength", "minItems",
    "maxItems", "nullable", "description", "title", "default", "$ref",
    "definitions", "anyOf",
})


def _is_null_schema_branch(branch: Any) -> bool:
    """Return whether one anyOf branch is the JSON null marker."""
    return isinstance(branch, dict) and cast(dict[str, Any], branch).get("type") == "null"


def _validate_mcp_schema(schema: Any, *, path: str = "") -> str | None:
    """Reject a schema whose advertised constraints the local validator cannot enforce."""
    if not isinstance(schema, dict):
        return f"{path or 'schema'} must be an object"
    mapping = cast(dict[str, Any], schema)
    for key in mapping:
        if key not in _SUPPORTED_MCP_SCHEMA_KEYS:
            return f"{path or 'schema'} uses unsupported keyword {key!r}"
    raw_type = mapping.get("type")
    if raw_type is not None:
        types: list[Any] = (
            list(cast(list[Any], raw_type))
            if isinstance(raw_type, list) else [raw_type]
        )
        if any(item not in {"string", "integer", "number", "boolean", "array", "object", "null"} for item in types):
            return f"{path or 'schema'} uses an unsupported type"
    for key, value in mapping.items():
        if key == "properties" and isinstance(value, dict):
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
        elif key == "anyOf" and isinstance(value, list):
            branches = cast(list[Any], value)
            non_null = [
                branch for branch in branches if not _is_null_schema_branch(branch)
            ]
            if len(non_null) > 1:
                return f"{path or 'schema'} uses an unsupported anyOf branch"
            for index, branch in enumerate(branches):
                error = _validate_mcp_schema(branch, path=f"{path}.anyOf[{index}]")
                if error:
                    return error
    return None


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    """Normalize MCP JSON Schema patterns for tool definitions."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    schema_mapping = cast(dict[str, Any], schema)
    return _normalize_nullable_schema(_rewrite_local_schema_refs(schema_mapping))


class _GenerationLease:
    """Keep one published MCP generation open until its admitted calls settle."""

    def __init__(self) -> None:
        self._inflight = 0
        self._retired = False
        self._closed = False
        self._closer: Callable[[], Awaitable[None]] | None = None
        self._guard = asyncio.Lock()

    def bind(self, closer: Callable[[], Awaitable[None]]) -> None:
        self._closer = closer

    def begin(self) -> None:
        self._inflight += 1

    def end(self) -> None:
        if self._inflight > 0:
            self._inflight -= 1
        if self._retired and self._inflight == 0 and not self._closed:
            asyncio.create_task(self.close_once())

    async def retire(self) -> None:
        """Close now when idle; otherwise close when the last admitted call ends."""
        self._retired = True
        if self._inflight == 0:
            await self.close_once()

    async def close_once(self) -> None:
        async with self._guard:
            if self._closed:
                return
            self._closed = True
        closer = self._closer
        if closer is not None:
            await closer()


def _admitted_generation(execute_method: Any) -> Any:
    """Count one admitted call against the wrapper's published generation."""

    @functools.wraps(execute_method)
    async def wrapper(self: Any, **kwargs: Any) -> Any:
        lease = getattr(self, "_generation_lease", None)
        if isinstance(lease, _GenerationLease):
            lease.begin()
        try:
            return await execute_method(self, **kwargs)
        finally:
            if isinstance(lease, _GenerationLease):
                lease.end()

    return wrapper


class _MCPWrapperBase(Tool):
    """Common reconnect handling for wrappers bound to one MCP server session."""

    _plugin_discoverable = False
    _session: ClientSession
    _server_name: str
    _name: str
    _generation_lease: _GenerationLease | None = None

    def _set_mcp_connection(self, session: ClientSession, server_name: str) -> None:
        self._session = session
        self._server_name = server_name
        self._reconnect: _ReconnectCallback | None = None

    def set_reconnect_handler(self, reconnect: _ReconnectCallback) -> None:
        self._reconnect = reconnect

    def bind_generation(self, lease: _GenerationLease) -> None:
        """Bind this wrapper to its published connection generation."""
        self._generation_lease = lease

    async def _refresh_session_after_termination(
        self,
        exc: BaseException,
        already_refreshed: bool,
        capability_kind: str,
    ) -> bool:
        if already_refreshed or not _is_session_terminated(exc) or self._reconnect is None:
            return False
        logger.warning(
            "MCP {} '{}' session terminated; reconnecting server '{}' before retry",
            capability_kind,
            self._name,
            self._server_name,
        )
        refreshed_tool = await self._reconnect(self._server_name, self._name, self)
        refreshed_session = getattr(refreshed_tool, "_session", None)
        if refreshed_session is None:
            logger.warning(
                "MCP {} '{}' could not refresh session for server '{}'",
                capability_kind,
                self._name,
                self._server_name,
            )
            return False
        self._session = refreshed_session
        return True


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


def _mcp_media_tool_result(text_parts: list[str], artifacts: list[dict[str, Any]]) -> str:
    """Build the compact tool result for an MCP call that returned media.

    The base64 stays out of the model context entirely — only artifact paths and
    metadata are returned, so the result is small and the channel can deliver the
    saved file via the message tool.
    """
    payload: dict[str, Any] = {
        "artifacts": artifacts,
        "next_step": (
            "These media files were returned by an MCP tool and saved as local artifacts. "
            "Call the message tool with the artifact 'path' values in the media "
            "parameter to deliver them to the user. To copy or inspect a saved "
            "file inside a Sandbox command, pass its exact path in the exec tool's "
            "'input_files' list and write only to an authorized workspace folder. "
            "Do not paste base64 or raw paths into your reply unless the user asks "
            "for debug details."
        ),
    }
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
    return isinstance(tool, MCPToolWrapper) and tool.is_read_only

class MCPToolWrapper(_MCPWrapperBase):
    """Wraps a single MCP server tool as a nanobot Tool."""

    _plugin_discoverable = False

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
    ):
        self._set_mcp_connection(session, server_name)
        self.generation = generation
        self._media_output_roots = tuple(
            Path(root).expanduser() for root in media_output_roots
        )
        self._original_name = tool_def.name
        self._name = _sanitize_mcp_tool_name(f"mcp_{server_name}_{tool_def.name}")
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
    def mcp_metadata(self) -> MCPToolMetadata:
        return self._mcp_metadata

    @property
    def is_read_only(self) -> bool:
        return self._mcp_metadata.read_only

    @property
    def read_only(self) -> bool:
        return self.is_read_only

    @property
    def supports_parallel_calls(self) -> bool:
        return self._mcp_metadata.supports_parallel_calls

    @property
    def exclusive(self) -> bool:
        return not self.supports_parallel_calls

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @_admitted_generation
    async def execute(self, **kwargs: Any) -> Any:
        retried_transient = False
        refreshed_session = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(self._original_name, arguments=kwargs),
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
                        from nanobot.agent.turn import ToolOutcome
                        from nanobot.utils.artifacts import media_artifact_ref

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
        resource_link_type = getattr(types, "ResourceLink", None)
        for block in content:
            if isinstance(block, types.TextContent):
                text_parts.append(block.text)
                continue
            data_url = _image_block_data_url(block, types)
            if data_url is not None:
                stored = self._store_image_block(data_url, arguments)
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
            from nanobot.utils.artifacts import (
                parse_embedded_text_payload,
                select_candidate_artifacts,
            )

            payload = parse_embedded_text_payload("\n".join(text_parts))
            candidates = select_candidate_artifacts(artifacts, payload)
            return _mcp_media_tool_result(text_parts, artifacts), candidates
        return ("\n".join(text_parts) or "(no output)"), []

    def _store_image_block(
        self, data_url: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Persist one image data URL as an artifact; return its metadata or None."""
        from nanobot.utils.artifacts import ArtifactError, store_generated_image_artifact

        try:
            return store_generated_image_artifact(
                data_url,
                prompt=str(arguments.get("prompt") or ""),
                model=str(arguments.get("model") or ""),
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
        from nanobot.utils.artifacts import ArtifactError, store_generated_file_artifact

        uri = getattr(block, "uri", None)
        if not isinstance(uri, str) or not uri:
            return None
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "file":
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
                model=str(arguments.get("model") or ""),
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
                return f"(MCP resource read timed out after {self._resource_timeout}s)"
            except asyncio.CancelledError:
                if task_is_cancelling():
                    raise
                logger.warning("MCP resource '{}' was cancelled by server/SDK", self._name)
                return "(MCP resource read was cancelled)"
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "resource",
                ):
                    refreshed_session = True
                    continue
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
                    return f"(MCP resource read failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP resource '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP resource read failed: {type(exc).__name__})"
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
                return f"(MCP prompt call timed out after {self._prompt_timeout}s)"
            except asyncio.CancelledError:
                if task_is_cancelling():
                    raise
                logger.warning("MCP prompt '{}' was cancelled by server/SDK", self._name)
                return "(MCP prompt call was cancelled)"
            except McpError as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "prompt",
                ):
                    refreshed_session = True
                    continue
                logger.exception(
                    "MCP prompt '{}' failed: code={} message={}",
                    self._name,
                    exc.error.code,
                    exc.error.message,
                )
                return f"(MCP prompt call failed: {exc.error.message} [code {exc.error.code}])"
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "prompt",
                ):
                    refreshed_session = True
                    continue
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
                    return f"(MCP prompt call failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP prompt '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP prompt call failed: {type(exc).__name__})"
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


async def connect_mcp_servers(
    mcp_servers: dict[str, MCPServerConfig],
    registry: ToolRegistry,
    *,
    oauth_handlers: Mapping[str, MCPOAuthHandlers] | None = None,
    media_output_roots: Sequence[Path] = (),
    readiness_timeout: float = 10.0,
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
                from nanobot.agent.tools.mcp_oauth import (
                    MCPAuthorizationRequiredError,
                    create_mcp_oauth_auth,
                )

                try:
                    oauth_auth = await create_mcp_oauth_auth(
                        name,
                        cfg.url,
                        (oauth_handlers or {}).get(name),
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
            await session.initialize()

            tools = await session.list_tools()
            pending_tools: list[Tool] = []
            generation = hashlib.sha256(
                f"{name}:{cfg.type or ''}:{cfg.url or ''}:{cfg.command or ''}".encode("utf-8")
            ).hexdigest()[:16]
            enabled_tools = set(cfg.enabled_tools)
            allow_all_tools = "*" in enabled_tools
            registered_count = 0
            matched_enabled_tools: set[str] = set()
            available_raw_names = [tool_def.name for tool_def in tools.tools]
            available_wrapped_names = [_sanitize_mcp_tool_name(f"mcp_{name}_{tool_def.name}") for tool_def in tools.tools]
            for tool_def in tools.tools:
                wrapped_name = _sanitize_mcp_tool_name(f"mcp_{name}_{tool_def.name}")
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
                schema_error = _validate_mcp_schema(tool_def.inputSchema)
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
                )
                pending_tools.append(wrapper)
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

            # Only register resources and prompts when no tool restriction is
            # active.  enabledTools is a per-*tool* allowlist; resources and
            # prompts have no equivalent name filter, so they must be skipped
            # whenever the operator specified a tool subset.  An empty list
            # (deny-all) or a list of specific tool names both indicate that
            # the operator intended to restrict capabilities — registering
            # unrestricted resource/prompt wrappers would violate that intent.
            # The default ["*"] (allow-all) means no restriction was intended.
            register_extras = allow_all_tools
            if register_extras:
                try:
                    resources_result = await session.list_resources()
                    for resource in resources_result.resources:
                        wrapper = MCPResourceWrapper(
                            session, name, resource, resource_timeout=cfg.tool_timeout
                        )
                        pending_tools.append(wrapper)
                        registered_count += 1
                        logger.debug(
                            "MCP: registered resource '{}' from server '{}'",
                            wrapper.name,
                            name,
                        )
                except Exception as e:
                    logger.debug(
                        "MCP server '{}': resources not supported or failed: {}", name, e
                    )

                try:
                    prompts_result = await session.list_prompts()
                    for prompt in prompts_result.prompts:
                        wrapper = MCPPromptWrapper(
                            session, name, prompt, prompt_timeout=cfg.tool_timeout
                        )
                        pending_tools.append(wrapper)
                        registered_count += 1
                        logger.debug(
                            "MCP: registered prompt '{}' from server '{}'",
                            wrapper.name,
                            name,
                        )
                except Exception as e:
                    logger.debug(
                        "MCP server '{}': prompts not supported or failed: {}", name, e
                    )
            else:
                logger.info(
                    "MCP server '{}': skipping resource/prompt registration "
                    "(enabledTools does not include '*' — only tools allowed)",
                    name,
                )

            if pending_tools:
                published: list[Tool] = []
                try:
                    for tool in pending_tools:
                        registry.register(tool)
                        published.append(tool)
                except BaseException:
                    for tool in published:
                        with suppress(Exception):
                            registry.unregister(tool.name)
                    raise
                logger.debug(
                    "MCP: published {} bindings for server '{}' generation {}",
                    len(published),
                    name,
                    generation,
                )
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
    attempted_names: list[str] = []

    try:
        for name, cfg in mcp_servers.items():
            attempted_names.append(name)
            try:
                result = await connect_single_server(name, cfg)
            except Exception as e:
                _log_mcp_connection_failure(name, e)
                continue
            if result[1] is not None:
                server_stacks[result[0]] = result[1]
    except BaseException:
        # Callers can bound readiness/reload with a timeout. If cancellation
        # interrupts a later server, ownership of earlier connections has not
        # transferred yet, so roll the whole batch back before propagating it.
        for name in attempted_names:
            _unregister_server_tools(registry, name)
        try:
            await close_mcp_connections(server_stacks)
        except BaseException as cleanup_exc:
            logger.debug("MCP batch rollback cleanup error (can be ignored): {}", cleanup_exc)
        raise

    return server_stacks


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return persisted session kwargs for MCP preset attachments."""
    mcp_presets = metadata.get("mcp_presets") if isinstance(metadata, Mapping) else None
    return {"mcp_presets": mcp_presets} if isinstance(mcp_presets, list) and mcp_presets else {}


def _configured_servers(config: Config) -> dict[str, MCPServerConfig]:
    from nanobot.agent.plugins import agent_plugin_mcp_servers

    return agent_plugin_mcp_servers(
        config.workspace_path,
        config.tools.mcp_servers,
    )


def _load_current_servers() -> dict[str, MCPServerConfig]:
    from nanobot.config.loader import load_config, resolve_config_env_vars

    return _configured_servers(resolve_config_env_vars(load_config()))


class MCPProvider:
    """Own configured MCP connections and their dynamic tool registrations."""

    def __init__(
        self,
        servers: Mapping[str, MCPServerConfig],
        registry: ToolRegistry,
        *,
        server_loader: MCPServerLoader | None = None,
        media_output_roots: Sequence[Path] = (),
    ) -> None:
        self._servers = dict(servers)
        self._registry = registry
        self._server_loader = server_loader or _load_current_servers
        self._media_output_roots = tuple(media_output_roots)
        self._connections: dict[str, MCPConnection] = {}
        self._generation_leases: dict[str, _GenerationLease] = {}
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
        completion gate and observe the same bounded result.
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
        self._connect_inflight = gate
        # Let concurrently scheduled callers join this attempt before the
        # owner starts work that may not suspend on a fast mock or cache hit.
        await asyncio.sleep(0)
        try:
            await self._connect_once()
        except asyncio.CancelledError:
            if not gate.done():
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
                from nanobot.agent.tools.mcp_oauth import mcp_oauth_has_credentials

                authorization_pending = {
                    name
                    for name, cfg in oauth_servers.items()
                    if not mcp_oauth_has_credentials(name, cfg.url)
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
                connected = await connect_mcp_servers(
                    missing_servers,
                    self._registry,
                    media_output_roots=self._media_output_roots,
                )
                if self._closing:
                    await close_mcp_connections(connected)
                    return
                self._connections.update(connected)
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
            from nanobot.agent.tools.mcp_oauth import mcp_oauth_has_credentials

            authorization_pending = {
                name
                for name, cfg in next_servers.items()
                if cfg.auth == "oauth" and not mcp_oauth_has_credentials(name, cfg.url)
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
            for name in removed:
                tools_removed += _unregister_server_tools(self._registry, name)
                await self._close_server(name)

            for name in [*removed, *authorization_pending]:
                self._runtime_statuses.pop(name, None)

            self._servers = next_servers

            # Prepare replacement generations privately. Only a ready candidate
            # may retire the old healthy generation for that server.
            changed_failed: set[str] = set()
            if changed:
                candidate_registry = ToolRegistry()
                self._set_runtime_status(changed, "connecting")
                try:
                    candidates = await connect_mcp_servers(
                        {name: next_servers[name] for name in changed},
                        candidate_registry,
                        media_output_roots=self._media_output_roots,
                    )
                except BaseException:
                    self._set_runtime_status(changed, "failed")
                    raise
                if self._closing:
                    await close_mcp_connections(candidates)
                    return self._closing_result()
                for name in changed:
                    candidate = candidates.get(name)
                    if candidate is None:
                        changed_failed.add(name)
                        self._set_runtime_status({name}, "failed")
                        continue
                    tools_removed += _unregister_server_tools(self._registry, name)
                    await self._retire_server(name)
                    for tool_name in list(candidate_registry.tool_names):
                        tool = candidate_registry.get(tool_name)
                        if tool is None:
                            continue
                        candidate_registry.unregister(tool_name)
                        if _tool_belongs_to_server(tool, tool_name, name):
                            self._registry.register(tool)
                    self._connections[name] = candidate
                    self._set_runtime_status({name}, "connected")
                published_changed = set(changed) - changed_failed
                self._attach_generation_leases(published_changed)
                self._attach_reconnect_handlers(published_changed)

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
                try:
                    connected = await connect_mcp_servers(
                        to_connect,
                        self._registry,
                        media_output_roots=self._media_output_roots,
                    )
                except BaseException:
                    self._set_runtime_status(to_connect, "failed")
                    raise
                if self._closing:
                    await close_mcp_connections(connected)
                    return self._closing_result()
                self._connections.update(connected)
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

    async def _refresh_terminated_server(
        self,
        server_name: str,
        tool_name: str,
        stale_tool: Tool,
    ) -> Tool | None:
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
            _unregister_server_tools(self._registry, server_name)
            await self._close_server(server_name)

            self._set_runtime_status({server_name}, "connecting")
            connected = await connect_mcp_servers(
                {server_name: cfg},
                self._registry,
                media_output_roots=self._media_output_roots,
            )
            if self._closing:
                await close_mcp_connections(connected)
                return None
            self._connections.update(connected)
            self._record_connection_result({server_name}, connected)
            self._attach_generation_leases(connected)
            self._attach_reconnect_handlers(connected)
            if server_name not in connected:
                logger.warning(
                    "MCP server '{}' reconnect failed after session termination",
                    server_name,
                )
                return None
            return self._registry.get(tool_name)

    def _attach_generation_leases(
        self,
        server_names: Iterable[str],
    ) -> None:
        """Bind newly published wrappers to one retire-on-last-call lease."""
        for server_name in server_names:
            connection = self._connections.get(server_name)
            if connection is None:
                continue
            lease = _GenerationLease()

            async def closer(
                name: str = server_name,
                conn: MCPConnection = connection,
            ) -> None:
                await _close_mcp_connection(name, conn)

            lease.bind(closer)
            for tool_name in list(self._registry.tool_names):
                tool = self._registry.get(tool_name)
                if not _tool_belongs_to_server(tool, tool_name, server_name):
                    continue
                if isinstance(tool, _MCPWrapperBase):
                    tool.bind_generation(lease)
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
        await lease.retire()

    async def _close_server(self, server_name: str) -> None:
        await self._retire_server(server_name)

    async def aclose(self) -> None:
        """Close every connection while excluding reconnect and hot reload."""
        self._closing = True
        async with self._lock:
            connections = dict(self._connections)
            self._connections.clear()
            self._generation_leases.clear()
            self._runtime_statuses.clear()
            for name in self._servers:
                _unregister_server_tools(self._registry, name)
            await close_mcp_connections(connections)


def _server_signature(cfg: Any) -> Any:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump(mode="json")
    return cfg


def _tool_prefix(server_name: str) -> str:
    return _sanitize_name(f"mcp_{server_name}_")


def _tool_belongs_to_server(tool: Tool | None, tool_name: str, server_name: str) -> bool:
    if isinstance(tool, _MCPWrapperBase):
        return getattr(tool, "_server_name", None) == server_name
    return tool_name.startswith(_tool_prefix(server_name))


def _unregister_server_tools(registry: ToolRegistry, server_name: str) -> int:
    removed = 0
    for tool_name in list(registry.tool_names):
        tool = registry.get(tool_name)
        if _tool_belongs_to_server(tool, tool_name, server_name):
            registry.unregister(tool_name)
            removed += 1
    return removed
