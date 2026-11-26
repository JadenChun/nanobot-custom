"""Exercise canonical tool result projections through the model turn boundary."""

import base64
import json
import re
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPDefinition

from kohaku.agent.loop import AgentLoop
from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.mcp import MCPToolWrapper
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.tools.web import WebFetchTool
from kohaku.agent.turn import TurnRequest
from kohaku.providers.anthropic_provider import AnthropicProvider
from kohaku.providers.base import LLMResponse, ToolCallRequest
from kohaku.providers.bedrock_provider import BedrockProvider
from tests.agent.test_finalized_budget_recovery import _png_bytes
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_engine import engine_config as engine_config
from tests.agent.test_migration_source_engine import upgrade


def _tool_call(call_id, name, arguments):
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


@pytest.mark.parametrize("use_default", [True, False])
async def test_canonical_mcp_and_web_fetch_preserve_complete_artifact_bytes(
    engine_config, monkeypatch, use_default,
):
    if not use_default:
        engine_config.artifacts.enabled = False
    texts = {"mcp": ("MCP data é\n" * 10_000)[:50_000],
             "web": ("Web data λ\n" * 10_000)[:50_000]}
    results = AsyncMock(return_value=CallToolResult(
        content=[TextContent(type="text", text=texts["mcp"])],
    ))
    mcp = MCPToolWrapper(
        SimpleNamespace(call_tool=results),
        "fixture",
        MCPDefinition(name="catalog", inputSchema={"type": "object", "properties": {}}),
    )
    fetch = AsyncMock(return_value=texts["web"])
    monkeypatch.setattr(WebFetchTool, "execute", fetch)

    class ArtifactProvider(ScriptedProvider):
        def __init__(self):
            super().__init__([])
            self.references = {}
            self.data = {"mcp": bytearray(), "web": bytearray()}
            self.done = set()
            self.read_calls = 0

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls.append((deepcopy(messages), tools, model))
            if len(self.calls) == 1:
                return LLMResponse(content=None, tool_calls=[
                    _tool_call("mcp", mcp.name, {}),
                    _tool_call("web", "web_fetch", {"url": "https://example.test/"}),
                ])
            if len(self.calls) == 2:
                assert any(row["function"]["name"] == "read_artifact" for row in tools) is use_default
                for row in messages:
                    if row.get("role") != "tool":
                        continue
                    call_id = row["tool_call_id"]
                    shown = row["content"]
                    reference = re.search(r'artifact:[0-9a-f]{32}', shown)
                    assert len(shown) < 2_000 and texts[call_id] not in shown
                    if not use_default:
                        assert reference is None
                        assert texts[call_id][:800] in shown and texts[call_id][-400:] in shown
                        continue
                    assert reference is not None
                    assert "read_artifact" in shown
                    self.references[call_id] = reference.group()
                if not use_default:
                    return LLMResponse(content="Verified both inline previews.")
                assert self.references.keys() == texts.keys()
            else:
                newest_results = [row for row in messages if row.get("role") == "tool"]
                for row in newest_results[-(len(self.data) - len(self.done)):]:
                    piece = json.loads(row["content"])
                    label = next(key for key, value in self.references.items()
                                 if value == piece["reference"])
                    assert piece["start_offset"] == len(self.data[label])
                    self.data[label].extend(piece["text"].encode("utf-8"))
                    assert piece["next_offset"] == len(self.data[label])
                    if piece["eof"]:
                        self.done.add(label)
                if self.done == texts.keys():
                    return LLMResponse(content="Verified both artifacts.")
            self.read_calls += 1
            return LLMResponse(content=None, tool_calls=[
                _tool_call(f"{label}-{self.read_calls}", "read_artifact", {
                    "reference": reference,
                    "offset": len(self.data[label]),
                    "max_bytes": 8192,
                })
                for label, reference in self.references.items() if label not in self.done
            ])

    provider = ArtifactProvider()
    tools = ToolRegistry()
    tools.register(mcp)
    engine_config.agents.defaults.max_tool_iterations = 20
    loop = AgentLoop.from_config(
        engine_config, provider=provider, tool_registry=tools, context_window_tokens=220_000,
    )
    upgrade(loop)
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Get both full results.", session_key="cli:source",
        ))
        assert result.error is None, result.error
        if use_default:
            assert result.content == "Verified both artifacts."
            assert {label: bytes(value) for label, value in provider.data.items()} == {
                label: value.encode("utf-8") for label, value in texts.items()
            }
        else:
            assert result.content == "Verified both inline previews."
    finally:
        await loop.aclose()


@pytest.mark.parametrize("oversized", [False, True])
async def test_second_tool_image_preserves_previous_request_message_bytes(engine_config, oversized):
    images = [
        "data:image/png;base64," + base64.b64encode(_png_bytes(side, side, 0)).decode("ascii")
        for side in (1, 2)
    ]

    class ImageFixture(Tool):
        name = "fixture_image"
        description = "Return one small fixture image."
        parameters = {
            "type": "object", "properties": {"index": {"type": "integer"}}, "required": ["index"],
        }

        async def execute(self, index):
            blocks = [
                {"type": "text", "text": f"Image {index}"},
                {"type": "image_url", "image_url": {"url": images[index]}},
            ]
            if oversized:
                blocks.extend(
                    {"type": "text", "text": prefix * 10_000}
                    for prefix in ("first block α\n", "second block β\n")
                )
            return blocks

    tools = ToolRegistry()
    tools.register(ImageFixture())
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[_tool_call("image-0", "fixture_image", {"index": 0})]),
        LLMResponse(content=None, tool_calls=[_tool_call("image-1", "fixture_image", {"index": 1})]),
        LLMResponse(content="Saw both images."),
    ])
    loop = AgentLoop.from_config(
        engine_config, provider=provider, tool_registry=tools, context_window_tokens=220_000,
    )
    upgrade(loop)
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Get the two images.", session_key="cli:source",
        ))
        assert result.error is None, result.error
        first, second = provider.calls[1][0], provider.calls[2][0]
        first_bytes = [json.dumps(row, ensure_ascii=False).encode("utf-8") for row in first]
        second_bytes = [json.dumps(row, ensure_ascii=False).encode("utf-8") for row in second]
        assert first_bytes == second_bytes[:len(first_bytes)]
        assert any(images[0].encode("utf-8") in row for row in first_bytes)
        assert any(images[1].encode("utf-8") in row for row in second_bytes[len(first_bytes):])
        if oversized:
            tool_message = next(row for row in first if row.get("tool_call_id") == "image-0")
            previews = [row["text"] for row in tool_message["content"][2:]]
            assert len(previews) == 2
            assert all(len(preview) < 2_000 for preview in previews)
            references = [re.search(r"artifact:[0-9a-f]{32}", preview) for preview in previews]
            assert all(references)
            assert references[0].group() != references[1].group()
    finally:
        await loop.aclose()


@pytest.mark.parametrize("backend", ["anthropic", "bedrock-nova", "bedrock-claude"])
async def test_finalized_native_backend_overflow_retries_compacted_source(engine_config, backend):
    requests = []

    def receive(kwargs):
        payload = json.dumps(kwargs, ensure_ascii=False)
        if "Summarize historical evidence" in payload:
            return "The older exchange discussed native request budgeting."
        requests.append(payload)
        if len(requests) == 1:
            error = RuntimeError("native request rejected")
            if backend == "anthropic":
                error.status_code = 400
                error.body = {"type": "error", "error": {
                    "type": "invalid_request_error", "message": "prompt is too long",
                }}
            else:
                error.response = {
                    "ResponseMetadata": {"HTTPStatusCode": 400},
                    "Error": {"Code": "ValidationException",
                              "Message": "Input is too long for requested model."},
                }
            raise error
        return "The latest question was preserved."

    if backend == "anthropic":
        model = "claude-sonnet-4-20250514"

        async def create(**kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=receive(kwargs))],
                stop_reason="end_turn", usage=None,
            )

        @asynccontextmanager
        async def stream(**kwargs):
            class Stream:
                async def __anext__(self):
                    raise StopAsyncIteration

                async def get_final_message(self):
                    return await create(**kwargs)

            yield Stream()

        provider = AnthropicProvider(api_key="fixture", default_model=model)
        provider._client = SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream))
    else:
        model = ("amazon.nova-pro-v1:0" if backend == "bedrock-nova"
                 else "global.anthropic.claude-sonnet-4-20250514-v1:0")

        def converse(**kwargs):
            return {"output": {"message": {"content": [{"text": receive(kwargs)}]}},
                    "stopReason": "end_turn"}

        def converse_stream(**kwargs):
            return {"stream": [
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": receive(kwargs)}}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]}

        provider = BedrockProvider(
            region="fixture", default_model=model,
            client=SimpleNamespace(converse=converse, converse_stream=converse_stream),
        )

    loop = AgentLoop.from_config(
        engine_config, provider=provider, tool_registry=ToolRegistry(), model=model,
        context_window_tokens=192_000,
    )
    session = loop.sessions.get_or_create("cli:source")
    session.add_message("user", " context" * 60_000)
    session.add_message("assistant", "I finished the old exchange.")
    upgrade(loop)
    try:
        result = await loop.execute_turn(TurnRequest(
            content="The latest user question.", session_key="cli:source",
        ))
        assert result.error is None, result.error
        assert result.content == "The latest question was preserved."
        model_requests = [r for r in requests if "Extract reusable lessons" not in r]
        assert len(model_requests) == 2
        assert len(model_requests[1]) < len(model_requests[0])
        assert len(model_requests[0]) > 240_000
        assert "The latest user question." in model_requests[1]
    finally:
        await loop.aclose()
