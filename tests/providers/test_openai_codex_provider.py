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

    _, tool_calls, finish_reason = await _consume_sse(_Response())

    assert finish_reason == "stop"
    assert len(tool_calls) == 1
    call_id, item_id = tool_calls[0].id.split("|", 1)
    assert len(call_id) <= 64
    assert len(item_id) <= 64
