"""Separate a source-history seed's capacity from the later pressure proof."""

from __future__ import annotations

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.read_provenance import parse_read_receipt
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnRequest
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.utils import helpers
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_source_engine import upgrade

engine_config = engine_fixtures.engine_config


@pytest.mark.parametrize("capacity,overflow,expected_calls", [
    (64_000, True, 1), (128_000, False, 2),
])
async def test_large_seed_retains_the_hard_byte_budget_and_real_receipt(
    engine_config, monkeypatch, capacity, overflow, expected_calls,
):
    def unavailable():
        raise RuntimeError("deterministic offline encoding fallback")

    monkeypatch.setattr(helpers, "_get_token_encoding", unavailable)
    content = "covered retrieval result content for engine restart " * 800
    (engine_config.workspace_path / "file").write_text(content)
    provider = engine_fixtures.ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="read", name="read_file", arguments={"path": "file"},
        )]),
        LLMResponse(content="read complete"),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider,
                                 context_window_tokens=capacity)
    key = "cli:source"
    backend = upgrade(loop, key)
    try:
        result = await loop.execute_turn(TurnRequest(content="read file", session_key=key))
        if overflow:
            assert result.error is not None
            assert "local context budget" in result.error and "heuristic" in result.error
        else:
            assert result.error is None and result.content == "read complete"
        assert len(provider.calls) == expected_calls
        source = backend.source.load_source_snapshot(key)
        event = next(row for row in source.events if row.get("_source_origin"))
        receipt = parse_read_receipt(event["_source_origin"])
        assert receipt.source_access.relative_path == "file"
        assert receipt.matches(event["content"])
        assert content in event["content"]
    finally:
        await loop.aclose()
