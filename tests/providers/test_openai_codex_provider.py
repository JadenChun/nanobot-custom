from __future__ import annotations

import json

import pytest

from nanobot.providers.openai_codex_provider import (
    _convert_messages,
    _normalize_codex_id,
)


def test_convert_messages_normalizes_long_tool_call_ids() -> None:
    raw_call = "call_" + "x" * 90
    raw_item = "item_" + "y" * 90
    combined = f"{raw_call}|{raw_item}"

    _, items = _convert_messages(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": combined,
                        "type": "function",
                        "function": {"name": "noop", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": combined, "name": "noop", "content": "ok"},
        ]
    )

    fn_call = items[1]
    fn_output = items[2]
    assert fn_call["type"] == "function_call"
    assert len(fn_call["call_id"]) <= 64
    assert len(fn_call["id"]) <= 64
    assert fn_output["type"] == "function_call_output"
    assert fn_output["call_id"] == fn_call["call_id"]


def test_normalize_codex_id_hashes_unsafe_values() -> None:
    normalized = _normalize_codex_id("call_bad+/value", "call_")

    assert normalized.startswith("call_")
    assert len(normalized) <= 64
    assert all(ch.isalnum() or ch in "_-" for ch in normalized)


@pytest.mark.asyncio
async def test_consume_sse_path_stores_safe_ids() -> None:
    from nanobot.providers.openai_codex_provider import _consume_sse

    raw_call = "call_" + "x" * 90
    raw_item = "item_" + "y" * 90
    added = {
        "type": "response.output_item.added",
        "item": {
            "type": "function_call",
            "call_id": raw_call,
            "id": raw_item,
            "name": "noop",
            "arguments": "{}",
        },
    }
    done = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": raw_call,
            "id": raw_item,
            "name": "noop",
            "arguments": "{}",
        },
    }
    completed = {"type": "response.completed", "response": {"status": "completed"}}
    lines = [
        f"data: {json.dumps(added)}",
        "",
        f"data: {json.dumps(done)}",
        "",
        f"data: {json.dumps(completed)}",
        "",
    ]

    class _Response:
        async def aiter_lines(self):
            for line in lines:
                yield line

    _, tool_calls, finish_reason, _image_calls = await _consume_sse(_Response())

    assert finish_reason == "stop"
    assert len(tool_calls) == 1
    call_id, item_id = tool_calls[0].id.split("|", 1)
    assert len(call_id) <= 64
    assert len(item_id) <= 64


@pytest.mark.asyncio
async def test_consume_sse_response_failed_is_retryable_transient() -> None:
    from nanobot.providers.openai_codex_provider import _CodexHTTPError, _consume_sse

    lines = ['data: {"type":"response.failed"}', ""]

    class _Response:
        async def aiter_lines(self):
            for line in lines:
                yield line

    with pytest.raises(_CodexHTTPError) as excinfo:
        await _consume_sse(_Response())

    assert excinfo.value.status_code == 502
    assert excinfo.value.retry_after is not None
    assert excinfo.value.retry_after > 0
    assert excinfo.value.error_code == "codex_response_failed"


@pytest.mark.asyncio
async def test_codex_transient_error_is_retried(monkeypatch) -> None:
    from nanobot.providers import openai_codex_provider as mod

    class _Token:
        account_id = "acct"
        access = "token"

    provider = mod.OpenAICodexProvider()
    calls = {"n": 0}

    async def flaky_request(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise mod._CodexHTTPError(
                "Codex response failed",
                status_code=502,
                retry_after=15.0,
                error_type="server_error",
                error_code="codex_response_failed",
            )
        return "recovered", [], "stop", []

    monkeypatch.setattr(mod, "_request_codex", flaky_request)
    monkeypatch.setattr(mod, "get_codex_token", lambda: _Token())

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)

    response = await provider.chat_with_retry([{"role": "user", "content": "hi"}])

    assert calls["n"] == 2
    assert response.finish_reason == "stop"
    assert response.content == "recovered"


@pytest.mark.asyncio
async def test_codex_transport_error_is_transient(monkeypatch) -> None:
    import httpx

    from nanobot.providers import openai_codex_provider as mod

    class _Token:
        account_id = "acct"
        access = "token"

    provider = mod.OpenAICodexProvider()

    async def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(mod, "_request_codex", boom)
    monkeypatch.setattr(mod, "get_codex_token", lambda: _Token())

    response = await provider.chat([{"role": "user", "content": "hi"}])

    assert response.finish_reason == "error"
    assert response.error_status_code == 503
    assert provider._is_transient_response(response) is True