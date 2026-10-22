"""Tests for the Responses API code path (used by GitHub Copilot reasoning models)."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from nanobot.providers.base import LLMUsage
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.openai_responses.converters import convert_messages, convert_tools
from nanobot.providers.openai_responses.parsing import parse_response_output
from nanobot.providers.registry import find_by_name

# ---------------------------------------------------------------------------
# Static decision: when should we route through /v1/responses?
# ---------------------------------------------------------------------------


def _make_copilot_provider() -> OpenAICompatProvider:
    spec = find_by_name("github_copilot")
    assert spec is not None
    return OpenAICompatProvider(
        api_key="x",
        default_model="gpt-5.4",
        api_base=spec.default_api_base,
        spec=spec,
    )


def _make_openai_provider(api_type: str = "auto") -> OpenAICompatProvider:
    spec = find_by_name("openai")
    assert spec is not None
    return OpenAICompatProvider(
        api_key="x",
        default_model="gpt-4o",
        api_base=spec.default_api_base,
        spec=spec,
        api_type=api_type,
    )


def test_copilot_routes_reasoning_models_through_responses_api():
    p = _make_copilot_provider()
    assert p._should_use_responses_api("gpt-5.4", None) is True
    assert p._should_use_responses_api("github-copilot/gpt-5.4", None) is True
    assert p._should_use_responses_api("o3-mini", None) is True


def test_copilot_keeps_classic_models_on_chat_completions():
    p = _make_copilot_provider()
    assert p._should_use_responses_api("gpt-4o", None) is False
    assert p._should_use_responses_api("claude-sonnet-4.6", None) is False


def test_non_optin_provider_never_uses_responses_api():
    # The explicit chat_completions setting is the current opt-out, including
    # requests that would otherwise select Responses by model or effort.
    p = _make_openai_provider(api_type="chat_completions")
    assert p._should_use_responses_api("gpt-5.4", None) is False
    assert p._should_use_responses_api("o3-mini", "high") is False


def test_direct_openai_auto_routes_reasoning_models_through_responses_api():
    p = _make_openai_provider()
    assert p._should_use_responses_api("gpt-5.4", None) is True
    assert p._should_use_responses_api("o3-mini", None) is True


def test_direct_openai_auto_uses_explicit_effort_but_keeps_classic_default():
    p = _make_openai_provider()
    assert p._should_use_responses_api("gpt-4o", None) is False
    assert p._should_use_responses_api("gpt-4o", "high") is True


# ---------------------------------------------------------------------------
# Message → input translation
# ---------------------------------------------------------------------------


def test_messages_to_responses_input_collapses_system_into_instructions():
    messages = [
        {"role": "system", "content": "You are nanobot."},
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ]
    instructions, items = convert_messages(messages)
    # The shared codec deliberately selects the last system instruction.
    assert instructions == "Be concise."
    assert items == [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]


def test_messages_to_responses_input_translates_tool_roundtrip():
    messages = [
        {"role": "user", "content": "use the tool"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"q": "x"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "result-data"},
    ]
    instructions, items = convert_messages(messages)
    assert instructions == ""
    assert len(items) == 4
    assert items[0] == {
        "role": "user", "content": [{"type": "input_text", "text": "use the tool"}],
    }
    # assistant text becomes an output_text content item
    assert items[1] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "calling"}],
        "status": "completed",
        "id": "msg_1",
    }
    # tool call becomes a typed function_call item
    assert items[2] == {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_abc",
        "name": "lookup",
        "arguments": '{"q": "x"}',
    }
    # tool result becomes function_call_output
    assert items[3] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": "result-data",
    }


def test_messages_to_responses_input_serializes_dict_arguments():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": {"a": 1, "b": [2, 3]}},
                }
            ],
        }
    ]
    _, items = convert_messages(messages)
    assert items[0]["type"] == "function_call"
    assert items[0]["id"] == "fc_0"
    assert items[0]["call_id"] == "call_1"
    assert json.loads(items[0]["arguments"]) == {"a": 1, "b": [2, 3]}


# ---------------------------------------------------------------------------
# Tools translation
# ---------------------------------------------------------------------------


def test_tools_to_responses_flattens_function_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "look stuff up",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    out = convert_tools(tools)
    assert out == [
        {
            "type": "function",
            "name": "lookup",
            "description": "look stuff up",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_tools_to_responses_handles_empty():
    assert convert_tools([]) == []
    p = _make_copilot_provider()
    for tools in (None, []):
        body = p._build_responses_body(
            messages=[{"role": "user", "content": "hi"}], tools=tools,
            model="gpt-5.4", max_tokens=1000, temperature=0.5,
            reasoning_effort=None, tool_choice=None,
        )
        assert "tools" not in body
        assert "tool_choice" not in body


@pytest.mark.parametrize(("choice", "expected"), [
    ({"type": "function", "function": {"name": "lookup"}}, {"type": "function", "name": "lookup"}),
    ({"type": "function", "name": "lookup"}, {"type": "function", "name": "lookup"}),
    ("auto", "auto"),
    ("required", "required"),
    ("none", "none"),
    (None, "auto"),
], ids=["chat-named", "responses-named", "auto", "required", "none", "default"])
def test_tool_choice_to_responses_named_function(choice, expected):
    original = deepcopy(choice)
    body = _make_copilot_provider()._build_responses_body(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        model="gpt-5.4", max_tokens=1000, temperature=0.5,
        reasoning_effort=None, tool_choice=choice,
    )
    assert body["tool_choice"] == expected
    assert choice == original


# ---------------------------------------------------------------------------
# Full kwargs build
# ---------------------------------------------------------------------------


def test_build_responses_kwargs_includes_reasoning_effort_and_strips_prefix():
    p = _make_copilot_provider()
    kwargs = p._build_responses_body(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        tools=[{
            "type": "function",
            "function": {"name": "f", "description": "d", "parameters": {}},
        }],
        model="github-copilot/gpt-5.4",
        max_tokens=2048,
        temperature=0.5,
        reasoning_effort="high",
        tool_choice="auto",
    )
    assert kwargs["model"] == "gpt-5.4"  # strip_model_prefix=True
    assert kwargs["max_output_tokens"] == 2048
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["instructions"] == "sys"
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"] == [
        {"type": "function", "name": "f", "description": "d", "parameters": {}}
    ]
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert kwargs["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    assert kwargs["store"] is False
    assert kwargs["stream"] is False


def test_build_responses_kwargs_omits_reasoning_when_not_set():
    p = _make_copilot_provider()
    kwargs = p._build_responses_body(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model="gpt-5.4",
        max_tokens=1000,
        temperature=0.5,
        reasoning_effort=None,
        tool_choice=None,
    )
    assert "reasoning" not in kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_responses_extracts_text_and_usage():
    response = {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought process"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello world"}],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    parsed = parse_response_output(response)
    assert parsed.content == "hello world"
    assert parsed.reasoning_content == "thought process"
    assert parsed.finish_reason == "stop"
    assert parsed.usage == LLMUsage.reported(input_tokens=10, output_tokens=5, total_tokens=15)
    assert parsed.tool_calls == []


@pytest.mark.parametrize("item_id", [None, "fc_xyz"], ids=["fallback", "provided"])
def test_parse_responses_extracts_function_call(item_id):
    response = {
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "lookup",
                "arguments": '{"q": "weather"}',
            }
        ],
    }
    if item_id is not None:
        response["output"][0]["id"] = item_id
    parsed = parse_response_output(response)
    # Responses reports terminal status separately from callable output.
    assert parsed.finish_reason == "stop"
    assert parsed.has_tool_calls is True
    assert parsed.should_execute_tools is True
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.id == f"call_xyz|{item_id or 'fc_0'}"
    assert tc.name == "lookup"
    assert tc.arguments == {"q": "weather"}
    _, replay = convert_messages([
        {"role": "assistant", "content": None, "tool_calls": [tc.to_openai_tool_call()]},
        {"role": "tool", "tool_call_id": tc.id, "content": "result-data"},
    ])
    assert replay == [
        {"type": "function_call", "id": item_id or "fc_0", "call_id": "call_xyz",
         "name": "lookup", "arguments": '{"q": "weather"}'},
        {"type": "function_call_output", "call_id": "call_xyz", "output": "result-data"},
    ]


def test_parse_responses_handles_empty_arguments():
    response = {
        "status": "completed",
        "output": [
            {"type": "function_call", "call_id": "c1", "name": "noop", "arguments": ""}
        ],
    }
    parsed = parse_response_output(response)
    assert parsed.tool_calls[0].arguments == {}
    assert parsed.tool_calls[0].id == "c1|fc_0"
    assert parsed.tool_calls[0].name == "noop"


def test_parse_responses_marks_length_on_incomplete():
    response = {
        "status": "incomplete",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "partial"}]}
        ],
    }
    parsed = parse_response_output(response)
    assert parsed.finish_reason == "length"
    assert parsed.content == "partial"
