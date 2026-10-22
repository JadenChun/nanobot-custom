from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from nanobot.providers.openai_responses.converters import convert_messages, normalize_portable_id
from nanobot.providers.openai_responses.parsing import (
    ResponsesStreamCapture,
    consume_sse_with_reasoning,
    parse_response_output,
)
from nanobot.providers.openai_responses.state import build_responses_state, prepare_responses_input


def _call(raw: str, item: str = "fc_0") -> dict[str, object]:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": f"{raw}|{item}", "function": {"name": "noop", "arguments": {"text": "雪"}},
    }]}


def _safe(value: str) -> bool:
    return bool(value) and len(value) <= 64 and all(
        ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for ch in value
    )


@pytest.mark.parametrize("raw", ["call_" + "x" * 90, "bad+/value", "雪", "é"])
def test_portable_paired_ids_are_ascii_bounded_and_nonmutating(raw: str) -> None:
    messages = [{"role": "tool", "tool_call_id": f"{raw}|{'y' * 90}", "content": "ok"},
                _call(raw, "y" * 90), _call(raw, "y" * 90)]
    before = deepcopy(messages)
    _, items = convert_messages(messages)
    assert all(_safe(item["call_id"]) for item in items)
    assert items[0]["call_id"] == items[1]["call_id"] == items[2]["call_id"]
    assert all(_safe(item["id"]) for item in items[1:])
    assert items[1]["id"] != items[2]["id"]
    assert items == convert_messages(messages)[1]
    assert messages == before
    assert items[1]["arguments"] == '{"text": "雪"}'


@pytest.mark.parametrize("item", ["rs_same", "fc_0", "x" * 64])
def test_duplicate_item_ids_are_bounded(item: str) -> None:
    _, items = convert_messages([_call("call_1", item), _call("call_2", item),
                                 _call("call_3", item[:62] + "_2")])
    ids = [entry["id"] for entry in items]
    assert len(set(ids)) == 3
    assert all(_safe(value) for value in ids)
    assert ids[0] == item
    assert ids[1] == item[:62] + "_2"


def test_normalizer_preserves_safe_and_hashes_unsafe_strings() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        normalize_portable_id(123, "call_")
    with pytest.raises(ValueError, match="unsupported"):
        normalize_portable_id("unsafe+/", "bad_")
    for raw in ["call_ABC-123", "x" * 64]:
        assert normalize_portable_id(raw, "call_") == raw
    for raw in ["", "雪", "é", "bad+/", "x" * 65]:
        normalized = normalize_portable_id(raw, "call_")
        assert _safe(normalized)
        assert normalized.startswith("call_")
        assert normalized != raw
        assert normalized == normalize_portable_id(raw, "call_")


def test_generated_call_hash_cannot_steal_reserved_safe_raw_id(monkeypatch) -> None:
    candidate = normalize_portable_id("bad+/", "call_")
    original = normalize_portable_id
    monkeypatch.setattr(
        "nanobot.providers.openai_responses.converters.normalize_portable_id",
        lambda raw, prefix: candidate if raw in {"bad+/", "other+/"} else original(raw, prefix),
    )
    raws = ["bad+/", "other+/", candidate]
    messages = [{"role": "tool", "tool_call_id": raw, "content": "ok"} for raw in raws]
    messages.extend(_call(raw) for raw in raws)
    _, items = convert_messages(messages)
    ids = [entry["call_id"] for entry in items[:3]]
    assert len(set(ids)) == 3
    assert ids[2] == candidate
    assert all(_safe(value) for value in ids)
    assert ids == [entry["call_id"] for entry in items[3:]]


@pytest.mark.asyncio
async def test_sse_capture_state_and_pending_replay_keep_exact_native_ids() -> None:
    raw_call, raw_item = "call_+/雪" + "x" * 90, "item_+/雪" + "y" * 90
    native = {"type": "function_call", "call_id": raw_call, "id": raw_item,
              "name": "noop", "arguments": "{}"}
    events = [
        {"type": "response.output_item.added", "item": native},
        {"type": "response.output_item.done", "item": native},
        {"type": "response.completed", "response": {"status": "completed", "output": [native]}},
    ]
    before = deepcopy(events)

    class Response:
        async def aiter_lines(self):
            for event in events:
                yield "data: " + json.dumps(event)
                yield ""

    capture = ResponsesStreamCapture()
    _, calls, finish, _, _ = await consume_sse_with_reasoning(Response(), capture=capture)
    assert finish == "stop"
    assert len(calls) == 1
    assert calls[0].id == f"{raw_call}|{raw_item}"
    assert capture.completed is True
    assert capture.output_items == [native]
    state = build_responses_state(provider="openai:test", model="model", input_items=[],
                                  output_items=capture.output_items)
    state.pending_messages.append({"role": "tool", "tool_call_id": calls[0].id, "content": "ok"})
    state_before = deepcopy(state)
    _, items, replayed = prepare_responses_input([], state=state, provider="openai:test", model="model")
    assert replayed is True
    assert items[0] == native
    assert items[1]["call_id"] == items[0]["call_id"] == raw_call
    assert state == state_before
    assert events == before


@pytest.mark.parametrize("compatible", [True, False, None])
def test_sdk_native_replay_vs_portable_fallback(compatible: bool | None) -> None:
    raw_call, raw_item = "call_bad+/" + "雪" * 90, "item_bad+/" + "y" * 90
    native = {"type": "function_call", "call_id": raw_call, "id": raw_item,
              "name": "noop", "arguments": "{}"}
    response = SimpleNamespace(status="completed", output=[SimpleNamespace(**native)])
    parsed = parse_response_output(response, state_provider="openai:test", state_model="model",
                                   state_input_items=[])
    assert parsed.tool_calls[0].id == f"{raw_call}|{raw_item}"
    state = parsed.provider_state
    assert state is not None
    assert state.payload["items"] == [native]
    pending = {"role": "tool", "tool_call_id": parsed.tool_calls[0].id, "content": "ok"}
    state.pending_messages.append(pending)
    before = deepcopy(state)
    messages = [_call(raw_call, raw_item), pending]
    _, items, replayed = prepare_responses_input(
        messages, state=state if compatible is not None else None,
        provider="openai:test", model="model" if compatible else "other",
    )
    assert replayed is (compatible is True)
    if compatible:
        assert items[0] == native
        assert items[1]["call_id"] == items[0]["call_id"] == raw_call
    else:
        assert _safe(items[0]["id"])
        assert _safe(items[0]["call_id"])
        assert items[1]["call_id"] == items[0]["call_id"]
    assert state == before
