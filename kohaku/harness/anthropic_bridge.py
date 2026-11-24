"""Serve Anthropic Messages over an authenticated loopback connection."""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from aiohttp import web

from kohaku.llm_usage.context import (
    LLMUsageSource,
    bind_llm_usage_source,
    current_llm_usage_source,
    reset_llm_usage_source,
)
from kohaku.providers.anthropic_provider import AnthropicProvider
from kohaku.providers.base import LLMResponse, LLMUsage, ToolCallRequest, parse_tool_arguments
from kohaku.providers.request_budget import ProviderRequestRejected
from kohaku.utils.helpers import estimate_prompt_tokens

if TYPE_CHECKING:
    from kohaku.utils.llm_runtime import LLMRuntime


class _InvalidRequestError(ValueError):
    pass


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _InvalidRequestError(f"{field} must be an object.")
    # JSON objects have string keys.
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise _InvalidRequestError(f"{field} must be an array.")
    return cast(list[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise _InvalidRequestError(f"{field} must be a string.")
    return value


def _image(block: dict[str, object], *, native: bool) -> dict[str, object]:
    source = _object(block.get("source"), "image.source")
    if source.get("type") != "base64":
        raise _InvalidRequestError("Use base64 for image sources.")
    media_type = source.get("media_type")
    if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        raise _InvalidRequestError("The image media type is invalid.")
    data = _string(source.get("data"), "image.source.data")
    if native:
        return block
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{data}"},
    }


def _content_blocks(value: object, field: str) -> list[dict[str, object]]:
    return [_object(block, field) for block in _array(value, field)]


def _system_blocks(value: object) -> list[dict[str, object]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    blocks = _content_blocks(value, "system block")
    for block in blocks:
        if block.get("type") != "text":
            raise _InvalidRequestError("System blocks must contain text.")
        _string(block.get("text"), "system text")
    return blocks


def _chat_messages(body: dict[str, object], *, native: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    system_blocks = _system_blocks(system) if system is not None else []

    for item in _array(body.get("messages"), "messages"):
        message = _object(item, "message")
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            # Claude Code can send system instructions in the Messages array.
            system_blocks.extend(_system_blocks(content))
            continue
        if role not in ("user", "assistant"):
            raise _InvalidRequestError("Message roles must be user, assistant, or system.")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        parts: list[dict[str, object]] = []
        calls: list[dict[str, Any]] = []
        reasoning: list[str] = []
        for block in _content_blocks(content, "content block"):
            kind = block.get("type")
            if kind == "text":
                text = _string(block.get("text"), "text")
                parts.append(block if native else {"type": "text", "text": text})
            elif kind == "image" and role == "user":
                parts.append(_image(block, native=native))
            elif kind == "thinking" and role == "assistant":
                thinking = _string(block.get("thinking"), "thinking")
                reasoning.append(thinking)
                if native and block.get("signature"):
                    parts.append(block)
            elif kind == "redacted_thinking" and role == "assistant":
                if native:
                    parts.append(block)
            elif kind == "tool_use" and role == "assistant":
                call = ToolCallRequest(
                    id=_string(block.get("id"), "tool_use.id"),
                    name=_string(block.get("name"), "tool_use.name"),
                    arguments=_object(block.get("input"), "tool_use.input"),
                )
                if native:
                    parts.append(block)
                else:
                    calls.append(call.to_openai_tool_call())
            elif kind == "tool_result" and role == "user":
                call_id = _string(block.get("tool_use_id"), "tool_result.tool_use_id")
                result = block.get("content", "")
                if not isinstance(result, str):
                    result_parts: list[dict[str, object]] = []
                    for part in _content_blocks(result, "tool result block"):
                        if part.get("type") == "text":
                            text = _string(part.get("text"), "tool result text")
                            result_parts.append(part if native else {"type": "text", "text": text})
                        elif part.get("type") == "image":
                            result_parts.append(_image(part, native=native))
                        elif part.get("type") == "tool_reference":
                            name = _string(part.get("tool_name"), "tool reference name")
                            result_parts.append(
                                part if native else {"type": "text", "text": name}
                            )
                        else:
                            raise _InvalidRequestError("The tool result block type is unsupported.")
                    if not native and all(part.get("type") == "text" for part in result_parts):
                        result = "\n".join(str(part.get("text", "")) for part in result_parts)
                    else:
                        result = result_parts
                if native:
                    parts.append({**block, "content": result})
                else:
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
            else:
                raise _InvalidRequestError("The message content block type is unsupported.")

        if parts or calls or reasoning:
            translated: dict[str, Any] = {"role": role, "content": parts or None}
            if calls:
                translated["tool_calls"] = calls
            if reasoning:
                translated["reasoning_content"] = "\n".join(reasoning)
            messages.append(translated)

    if system_blocks:
        system_content = system_blocks if native else "\n".join(
            _string(block["text"], "system text") for block in system_blocks
        )
        messages.insert(0, {"role": "system", "content": system_content})
    return messages


def _chat_tools(value: object, *, native: bool) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    tools: list[dict[str, Any]] = []
    for tool in _content_blocks(value, "tool"):
        if tool.get("name") == "DeferredToolPlaceholder":
            continue
        if "input_schema" not in tool and isinstance(tool.get("type"), str):
            # Anthropic-hosted tools have no portable function schema.
            continue
        function = {
            "name": _string(tool.get("name"), "tool.name"),
            "parameters": _object(tool.get("input_schema"), "tool.input_schema"),
        }
        if "description" in tool:
            function["description"] = _string(tool["description"], "tool.description")
        entry: dict[str, Any] = {"type": "function", "function": function}
        if native and "cache_control" in tool:
            entry["cache_control"] = tool["cache_control"]
        tools.append(entry)
    return tools or None


def _tool_choice(value: object) -> str | dict[str, Any] | None:
    if value is None:
        return None
    choice = _object(value, "tool_choice")
    kind = choice.get("type")
    if kind == "tool":
        name = _string(choice.get("name"), "tool_choice.name")
        return {"type": "function", "function": {"name": name}}
    if kind == "any":
        return "required"
    if kind in ("auto", "none"):
        return str(kind)
    raise _InvalidRequestError("The tool_choice type is unsupported.")


def _reasoning_effort(body: dict[str, object], fallback: str | None, *, native: bool) -> str | None:
    thinking = body.get("thinking")
    settings = _object(thinking, "thinking") if thinking is not None else {}
    mode = settings.get("type")
    if mode == "disabled":
        return "none"
    output = body.get("output_config")
    effort = cast(dict[str, object], output).get("effort") if isinstance(output, dict) else None
    if native and mode == "adaptive":
        # The native provider exposes adaptive mode through reasoning_effort.
        return "adaptive"
    if isinstance(effort, str):
        return effort
    if mode == "enabled":
        budget = settings.get("budget_tokens")
        if isinstance(budget, int) and not isinstance(budget, bool):
            return "high" if budget >= 8192 else "medium" if budget >= 4096 else "low"
        return fallback or "medium"
    return (fallback or "medium") if mode == "adaptive" else fallback


@dataclass(frozen=True, slots=True)
class _Request:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    model: str
    max_tokens: int
    temperature: float
    reasoning_effort: str | None
    tool_choice: str | dict[str, Any] | None
    stop_sequences: tuple[str, ...]
    stream: bool

    @classmethod
    def parse(cls, body: dict[str, object], runtime: LLMRuntime, *, count: bool) -> _Request:
        native = isinstance(runtime.provider, AnthropicProvider)
        maximum = body.get("max_tokens", runtime.generation.max_tokens)
        temperature = body.get("temperature", runtime.generation.temperature)
        stream = body.get("stream", False)
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise _InvalidRequestError("max_tokens must be a positive integer.")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or ("temperature" in body and not 0 <= temperature <= 1)
        ):
            raise _InvalidRequestError("temperature must be between zero and one.")
        if not isinstance(stream, bool):
            raise _InvalidRequestError("stream must be a boolean.")

        sequences = tuple(
            _string(value, "stop sequence")
            for value in _array(body.get("stop_sequences", []), "stop_sequences")
        )
        if "" in sequences:
            raise _InvalidRequestError("Stop sequences must be nonempty.")
        choice = _tool_choice(body.get("tool_choice"))
        tools = _chat_tools(body.get("tools"), native=native and not count)
        if native and not count and choice == "none":
            # The native provider omits tool_choice="none" on the wire.
            tools = None
        return cls(
            messages=_chat_messages(body, native=native and not count),
            tools=tools,
            model=runtime.model,
            max_tokens=maximum,
            temperature=temperature,
            reasoning_effort=_reasoning_effort(body, runtime.generation.reasoning_effort, native=native),
            tool_choice=choice,
            stop_sequences=sequences,
            stream=stream,
        )

    def provider_args(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "tools": self.tools,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "tool_choice": self.tool_choice,
        }


class _StopSequences:
    """Apply stop sequences at the bridge because the provider API has no stop parameter."""

    def __init__(self, sequences: tuple[str, ...]) -> None:
        self._pattern = re.compile("|".join(map(re.escape, sequences))) if sequences else None
        self._sequences = sequences
        self._max_tail = max(map(len, sequences), default=1) - 1
        self._tail = ""
        self.matched: str | None = None

    def feed(self, text: str, *, final: bool = False) -> str:
        if self.matched is not None:
            return ""
        if self._pattern is None:
            return text
        text = self._tail + text
        self._tail = ""
        if match := self._pattern.search(text):
            self.matched = match[0]
            return text[:match.start()]
        if not final:
            for length in range(min(len(text), self._max_tail), 0, -1):
                tail = text[-length:]
                if any(seq.startswith(tail) for seq in self._sequences):
                    self._tail = tail
                    return text[:-length]
        return text


def _usage(usage: LLMUsage | None) -> dict[str, int]:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    result = {
        "input_tokens": max(
            0, usage.input_tokens - (usage.cache_read_tokens or 0) - (usage.cache_write_tokens or 0),
        ),
        "output_tokens": usage.output_tokens,
    }
    if usage.cache_read_tokens is not None:
        result["cache_read_input_tokens"] = usage.cache_read_tokens
    if usage.cache_write_tokens is not None:
        result["cache_creation_input_tokens"] = usage.cache_write_tokens
    return result


def _message(model: str, usage: LLMUsage | None = None) -> dict[str, Any]:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": _usage(usage),
    }


def _stop(response: LLMResponse, matched: str | None) -> dict[str, str | None]:
    if matched is not None:
        return {"stop_reason": "stop_sequence", "stop_sequence": matched}
    reason = {
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "tool_use": "tool_use",
        "stop_sequence": "stop_sequence",
        "content_filter": "refusal",
        "refusal": "refusal",
    }.get(response.finish_reason, "end_turn")
    if reason == "end_turn" and response.has_tool_calls:
        reason = "tool_use"
    return {"stop_reason": reason, "stop_sequence": None}


def _thinking(response: LLMResponse) -> list[dict[str, Any]]:
    if response.thinking_blocks:
        return response.thinking_blocks
    if response.reasoning_content:
        return [{"type": "thinking", "thinking": response.reasoning_content, "signature": ""}]
    return []


class _InvalidProviderOutputError(ValueError):
    pass


def _tool_blocks(response: LLMResponse) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for call in response.tool_calls:
        arguments = parse_tool_arguments(call.arguments)
        if not isinstance(arguments, dict) or not call.has_valid_name() or not call.id:
            raise _InvalidProviderOutputError("The model provider returned an invalid tool call.")
        blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": arguments})
    return blocks


def _error(kind: str, message: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": kind, "message": message}}


def _upstream_error(status: int | None, kind: str | None = None) -> tuple[int, dict[str, object]]:
    if kind == "context_length" or status in (400, 422):
        return 400, _error("invalid_request_error", "The model provider rejected the request.")
    if status == 413:
        return 413, _error("request_too_large", "The model request is too large.")
    if status == 429:
        return 429, _error("rate_limit_error", "The model provider rate limit was reached.")
    if status in (503, 529):
        return 529, _error("overloaded_error", "The model provider is unavailable.")
    if kind == "timeout" or status in (408, 504):
        return 504, _error("api_error", "The model provider did not respond in time.")
    return 502, _error("api_error", "The model provider request failed.")


class _MessageStream:
    def __init__(
        self, response: web.StreamResponse, stop: _StopSequences, *, native: bool,
    ) -> None:
        self._response = response
        self._stop = stop
        self._native = native
        self._index = -1
        self._open_kind: str | None = None
        self._saw_text = False
        self._saw_thinking = False
        self._tool_deltas: dict[int, list[str]] = {}
        self._tool_ids: dict[int, str] = {}

    async def event(self, kind: str, **fields: object) -> None:
        data = json.dumps({"type": kind, **fields})
        await self._response.write(f"event: {kind}\ndata: {data}\n\n".encode())

    async def _close_block(self) -> None:
        if self._open_kind is not None:
            await self.event("content_block_stop", index=self._index)
            self._open_kind = None

    async def _open_block(self, kind: str, **fields: object) -> None:
        await self._close_block()
        self._open_kind = kind
        self._index += 1
        await self.event(
            "content_block_start", index=self._index, content_block={"type": kind, **fields},
        )

    async def _delta(self, kind: str, **fields: object) -> None:
        await self.event("content_block_delta", index=self._index, delta={"type": kind, **fields})

    async def _text(self, text: str) -> None:
        if text:
            if self._open_kind != "text":
                await self._open_block("text", text="")
            await self._delta("text_delta", text=text)

    async def on_text(self, text: str) -> None:
        if text and not (self._native and self._saw_thinking):
            self._saw_text = True
            await self._text(self._stop.feed(text))

    async def on_thinking(self, text: str) -> None:
        if not text or self._stop.matched is not None:
            return
        self._saw_thinking = True
        if self._native:
            # The native provider returns signed thinking blocks at completion.
            return
        if self._open_kind != "thinking":
            await self._open_block("thinking", thinking="", signature="")
        await self._delta("thinking_delta", thinking=text)

    async def on_tool(self, delta: dict[str, Any]) -> None:
        index = delta.get("index", 0)
        if isinstance(index, int):
            call_id = delta.get("call_id")
            if isinstance(call_id, str) and call_id:
                self._tool_ids[index] = call_id
            pieces = self._tool_deltas.setdefault(index, [])
            arguments = delta.get("arguments_delta")
            if isinstance(arguments, str) and arguments:
                pieces.append(arguments)

    async def _emit_thinking(self, result: LLMResponse) -> None:
        blocks = _thinking(result)
        if not blocks:
            return
        for block in blocks:
            if block.get("type") != "thinking":
                await self._open_block(str(block["type"]), **{
                    key: value for key, value in block.items() if key != "type"
                })
                continue
            await self._open_block("thinking", thinking="", signature="")
            if text := block.get("thinking"):
                await self._delta("thinking_delta", thinking=text)
            if signature := block.get("signature"):
                await self._delta("signature_delta", signature=signature)
        await self._close_block()

    async def _emit_tools(self, blocks: list[dict[str, Any]]) -> None:
        # Providers can interleave parallel calls. Anthropic clients consume a
        # single block at a time and need the final ID and name at block start.
        indices = list(self._tool_deltas)
        by_id = {call_id: index for index, call_id in self._tool_ids.items()}
        for position, block in enumerate(blocks):
            call_id, arguments = block["id"], block["input"]
            await self._open_block("tool_use", id=call_id, name=block["name"], input={})
            stream_index = by_id.get(call_id, indices[position] if position < len(indices) else -1)
            pieces = self._tool_deltas.get(stream_index, [])
            try:
                complete: object = json.loads("".join(pieces)) if pieces else None
            except ValueError:
                complete = None
            if complete != arguments:
                pieces = [json.dumps(arguments)]
            for piece in pieces:
                await self._delta("input_json_delta", partial_json=piece)
        await self._close_block()

    async def finish(self, result: LLMResponse) -> None:
        if self._native or not self._saw_thinking:
            await self._emit_thinking(result)
        await self._text(self._stop.feed("" if self._saw_text else result.content or "", final=True))
        if self._stop.matched is None:
            await self._emit_tools(_tool_blocks(result))
        await self._close_block()
        await self.event("message_delta", delta=_stop(result, self._stop.matched), usage=_usage(result.usage))
        await self.event("message_stop")


class LoopbackAnthropicBridge:
    """Route Anthropic Messages requests to one bound Kohaku runtime."""

    def __init__(self, runtime: LLMRuntime, *, usage_source: LLMUsageSource | None = None) -> None:
        self._runtime = runtime
        self._usage_source: LLMUsageSource = (
            usage_source if usage_source is not None else current_llm_usage_source()
        )
        self._auth_token = secrets.token_urlsafe(32)
        self._base_url: str | None = None
        self._runner: web.AppRunner | None = None
        self._lifecycle = asyncio.Lock()
        self._active: set[asyncio.Task[Any]] = set()

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("Start the Anthropic bridge before you use its URL.")
        return self._base_url

    @property
    def auth_token(self) -> str:
        return self._auth_token

    async def start(self) -> None:
        async with self._lifecycle:
            if self._runner is not None:
                return
            app = web.Application(client_max_size=32 * 1024 * 1024)
            app.router.add_post("/v1/messages", self._handle)
            app.router.add_post("/v1/messages/count_tokens", self._handle)
            app.on_shutdown.append(self._cancel_requests)
            runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
            try:
                await runner.setup()
                await web.TCPSite(runner, "127.0.0.1", 0).start()
                address: object = runner.addresses[0]
                if not isinstance(address, tuple) or len(cast(tuple[object, ...], address)) != 2:
                    raise RuntimeError("The Anthropic bridge did not bind an IPv4 socket.")
                port = cast(tuple[object, ...], address)[1]
                if not isinstance(port, int) or port <= 0:
                    raise RuntimeError("The Anthropic bridge did not get a TCP port.")
            except BaseException:
                await runner.cleanup()
                raise
            self._runner = runner
            self._base_url = f"http://127.0.0.1:{port}"

    async def close(self) -> None:
        async with self._lifecycle:
            if self._runner is not None:
                await self._runner.cleanup()
                self._runner = None
                self._base_url = None

    async def __aenter__(self) -> LoopbackAnthropicBridge:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def _cancel_requests(self, _app: web.Application) -> None:
        active = tuple(self._active)
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)

    def _authorized(self, request: web.Request) -> bool:
        api_key = request.headers.get("x-api-key", "")
        scheme, _, bearer = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer":
            bearer = ""
        key = self._auth_token.encode()
        key_ok = hmac.compare_digest(api_key.encode(errors="surrogatepass"), key)
        bearer_ok = hmac.compare_digest(bearer.encode(errors="surrogatepass"), key)
        return key_ok or bearer_ok

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return web.json_response(
                _error("authentication_error", "The bridge token is missing or invalid."), status=401,
            )
        task = asyncio.current_task()
        if task is not None:
            self._active.add(task)
        try:
            return await self._dispatch(request)
        finally:
            if task is not None:
                self._active.discard(task)

    async def _dispatch(self, request: web.Request) -> web.StreamResponse:
        count = request.path == "/v1/messages/count_tokens"
        try:
            raw: object = await request.json()
            parsed = _Request.parse(_object(raw, "The request"), self._runtime, count=count)
        except web.HTTPRequestEntityTooLarge:
            return web.json_response(
                _error("request_too_large", "The request is too large."), status=413,
            )
        except _InvalidRequestError as exc:
            return web.json_response(_error("invalid_request_error", str(exc)), status=400)
        except ValueError:
            return web.json_response(_error("invalid_request_error", "The request must use JSON."), status=400)
        if count:
            tokens = estimate_prompt_tokens(parsed.messages, parsed.tools)
            return web.json_response({"input_tokens": tokens})

        usage_token = bind_llm_usage_source(self._usage_source)
        try:
            if parsed.stream:
                return await self._stream(request, parsed)
            return await self._complete(parsed)
        finally:
            reset_llm_usage_source(usage_token)

    async def _complete(self, parsed: _Request) -> web.Response:
        try:
            result = await self._runtime.provider.chat_with_retry(**parsed.provider_args())
            if result.finish_reason == "error":
                status, body = _upstream_error(result.error_status_code, result.error_kind)
                return web.json_response(body, status=status)
            stop = _StopSequences(parsed.stop_sequences)
            answer = stop.feed(result.content or "", final=True)
            content = _thinking(result)
            if answer:
                content = [*content, {"type": "text", "text": answer}]
            if stop.matched is None:
                content = [*content, *_tool_blocks(result)]
            return web.json_response({
                **_message(parsed.model, result.usage), **_stop(result, stop.matched), "content": content,
            })
        except ProviderRequestRejected:
            status, body = _upstream_error(400)
        except _InvalidProviderOutputError as exc:
            status, body = 502, _error("api_error", str(exc))
        except Exception:
            status, body = _upstream_error(None)
        return web.json_response(body, status=status)

    async def _stream(self, request: web.Request, parsed: _Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        })
        await response.prepare(request)
        stream = _MessageStream(
            response, _StopSequences(parsed.stop_sequences),
            native=isinstance(self._runtime.provider, AnthropicProvider),
        )
        await stream.event("message_start", message=_message(parsed.model))
        try:
            result = await self._runtime.provider.chat_stream_with_retry(
                **parsed.provider_args(),
                on_content_delta=stream.on_text,
                on_thinking_delta=stream.on_thinking,
                on_tool_call_delta=stream.on_tool,
            )
        except ConnectionError:
            transport = request.transport
            if transport is None or transport.is_closing():
                raise
            result = LLMResponse(content=None, finish_reason="error", error_kind="connection")
        except ProviderRequestRejected:
            result = LLMResponse(content=None, finish_reason="error", error_status_code=400)
        except Exception:
            result = LLMResponse(content=None, finish_reason="error")

        if result.finish_reason == "error":
            _, body = _upstream_error(result.error_status_code, result.error_kind)
            await stream.event("error", error=body["error"])
        else:
            try:
                await stream.finish(result)
            except _InvalidProviderOutputError as exc:
                await stream.event("error", error={"type": "api_error", "message": str(exc)})
        return response
