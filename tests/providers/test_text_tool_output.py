"""Unparsed tool envelopes must not become answers or streamed UI text."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.providers.base import LLMProvider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider

XML_TOOL_CALL = (
    "<tool_call>\n<function=memory_search>\n<parameter=query>\n"
    "earlier projects\n</parameter>\n</function>\n</tool_call>"
)


def _provider():
    return OpenAICompatProvider(api_key="test", api_type="chat_completions")


@pytest.mark.parametrize("shape", ["dict", "sdk", "string", "content", "chunks"])
def test_unparsed_tool_envelope_is_a_nontransient_error(shape):
    provider = _provider()
    if shape == "chunks":
        response = provider._parse_chunks([
            {"choices": [{"delta": {"content": XML_TOOL_CALL}, "finish_reason": "stop"}]},
        ])
    elif shape == "sdk":
        response = provider._parse(SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=XML_TOOL_CALL, tool_calls=[]),
                finish_reason="stop",
            )], usage=None,
        ))
    elif shape == "string":
        response = provider._parse(XML_TOOL_CALL)
    elif shape == "content":
        response = provider._parse({"content": XML_TOOL_CALL})
    else:
        response = provider._parse({"choices": [{
            "message": {"content": XML_TOOL_CALL}, "finish_reason": "stop",
        }], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 10

    assert response.finish_reason == "error"
    assert response.error_kind == "malformed_tool_output"
    assert response.error_should_retry is False
    assert not LLMProvider.is_transient_response(response)
    assert response.tool_calls == []
    assert "<tool_call>" not in response.content
    assert "memory_search" not in response.content


@pytest.mark.parametrize("content", [
    "  \n" + XML_TOOL_CALL,
    "<tool_call><function=read_file><parameter=path>README.md",
    '<tool_call>{"name":',
])
def test_truncated_and_whitespace_prefixed_attempts_are_rejected(content):
    response = _provider()._parse_chunks([content])
    assert response.finish_reason == "error"
    assert response.error_kind == "malformed_tool_output"
    assert response.tool_calls == []


@pytest.mark.parametrize("content", [
    "Here is the project summary.",
    "```xml\n" + XML_TOOL_CALL + "\n```",
    "The literal marker `<tool_call>` identifies a tool request.",
])
def test_prose_and_quoted_xml_are_preserved(content):
    response = _provider()._parse_chunks([content])
    assert response.content == content
    assert response.finish_reason == "stop"
    assert response.tool_calls == []


@pytest.mark.parametrize("finish_reason", ["refusal", "content_filter", "length"])
def test_tool_output_detection_preserves_terminal_reasons(finish_reason):
    response = _provider()._parse_chunks([{
        "choices": [{"delta": {"content": XML_TOOL_CALL}, "finish_reason": finish_reason}],
    }])
    assert response.finish_reason == finish_reason
    assert response.error_kind != "malformed_tool_output"


async def _stream_response(parts, on_content_delta):
    async def chunks():
        for part in parts:
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=part), finish_reason="stop",
                )], usage=None,
            )

    provider = _provider()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=chunks()),
    )))
    return await provider.chat_stream(
        messages=[{"role": "user", "content": "check the projects"}],
        on_content_delta=on_content_delta,
    )


async def test_stream_holds_split_tool_markup_before_it_reaches_the_ui():
    deltas = []

    async def collect(delta):
        deltas.append(delta)

    response = await _stream_response(list(XML_TOOL_CALL), collect)
    assert deltas == []
    assert response.error_kind == "malformed_tool_output"


async def test_stream_normalizes_json_tool_markup_without_leaking_it():
    deltas = []

    async def collect(delta):
        deltas.append(delta)

    response = await _stream_response([
        "<tool_", 'call>{"name":"read_file","arguments":{"path":"README.md"}}',
        "</tool_call>",
    ], collect)
    assert deltas == []
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}


@pytest.mark.parametrize("parts", [
    ["Hello", " there"], ["<", "3"], ["<"], [" ", "hello"],
    ["```xml\n", XML_TOOL_CALL, "\n```"],
])
async def test_stream_preserves_normal_text_and_literal_examples(parts):
    deltas = []

    async def collect(delta):
        deltas.append(delta)

    response = await _stream_response(parts, collect)
    assert "".join(deltas) == "".join(parts)
    assert response.content == "".join(parts)
    assert response.finish_reason == "stop"


async def test_ordinary_answers_keep_streaming_before_completion():
    deltas = []

    async def collect(delta):
        deltas.append(delta)

    async def chunks():
        for part in ["Hello", " there"]:
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=part), finish_reason="stop",
            )], usage=None)
            assert deltas[-1] == part

    provider = _provider()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=chunks()),
    )))
    response = await provider.chat_stream(messages=[], on_content_delta=collect)
    assert response.content == "Hello there"
