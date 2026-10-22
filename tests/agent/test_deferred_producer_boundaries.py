"""Canonical-engine safety boundaries for generic receipt-free tool output."""

from __future__ import annotations

from copy import deepcopy

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome, TurnRequest
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.artifact import published_artifacts
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider

engine_config = engine_fixtures.engine_config


def _upgrade(loop: AgentLoop, key: str) -> object:
    session = loop.sessions.get_or_create(key)
    loop.sessions.save(session)
    backend = loop.sessions._jsonl_store
    backend.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    loop.sessions.invalidate(key)
    return backend


class _ReceiptFreeTool(Tool):
    """A local unqualified tool with no producer or retrieval receipt.

    The real registry, source writer, projection, and configured artifact
    factory still execute. This fixture does not represent or invoke a concrete
    filesystem, shell, web, session, delegation, plugin, or artifact adapter.
    """

    def __init__(self, name: str, result: str) -> None:
        self._name = name
        self.result = result
        self.executions = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Offline receipt-free tool fixture"

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **kwargs: object) -> ToolOutcome:
        assert not kwargs
        self.executions += 1
        return ToolOutcome(self.result, policy_metadata={"fixture": self.name})


class _ReceiptFreeProvider(ScriptedProvider):
    """Capture exact model messages without allowing an unplanned request."""

    def __init__(self, tool_name: str, *, final: str = "inline complete") -> None:
        super().__init__([
            LLMResponse(content=None, tool_calls=[
                ToolCallRequest(id="receipt-free-call", name=tool_name, arguments={}),
            ]),
            LLMResponse(content=final),
        ])
        self.tool_name = tool_name

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(tools), model))
        assert self.responses, "unexpected provider request"
        return self.responses.popleft()


class _OverflowProvider(_ReceiptFreeProvider):
    """Measure a receipt-free result as too large for every request/summary path."""

    def __init__(self, tool_name: str, marker: str) -> None:
        super().__init__(tool_name)
        self.marker = marker

    def estimate_prompt_tokens(self, messages, tools, model):
        if self.marker in str(messages):
            return 100_000, "receipt-free-overflow-fixture"
        return 100, "receipt-free-overflow-fixture"


def _mixed_utf8_result() -> str:
    return (
        "receipt-free fixture: plain text\n"
        "{\"status\": \"ok\", \"count\": 2}\n"
        "café U0001f642 \u6f22字\n"
        "- first item\n- second item"
    )


def _assert_no_artifact_publication(snapshot: object, *, workspace_id: str) -> None:
    events = snapshot.events
    assert not published_artifacts(snapshot, workspace_id=workspace_id)
    assert not any("_source_origin" in event or "_artifact_read" in event for event in events)


async def test_generic_receipt_free_tool_stays_canonical_inline_and_does_not_redispatch(
    engine_config,
) -> None:
    """A fitting receipt-free result is durable verbatim and remains inline."""
    engine_config.artifacts.enabled = True
    key = "cli:receipt-free"
    tool_name = "receipt_free_fixture"
    result_text = _mixed_utf8_result()
    adapter = _ReceiptFreeTool(tool_name, result_text)
    provider = _ReceiptFreeProvider(tool_name)
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    loop.tools.register(adapter)
    backend = _upgrade(loop, key)
    try:
        outcome = await loop.execute_turn(TurnRequest(content="run receipt-free fixture", session_key=key))

        assert outcome.error is None, outcome.error
        assert outcome.content == "inline complete"
        assert adapter.executions == 1
        assert len(provider.calls) == 2
        assert any(
            message.get("role") == "tool" and message.get("content") == result_text
            for message in provider.calls[1][0]
        )

        snapshot = backend.source.load_source_snapshot(key)
        canonical = next(event for event in snapshot.events if event.get("name") == tool_name)
        assert canonical["content"] == result_text
        assert canonical["content"].encode("utf-8") == result_text.encode("utf-8")
        # This test starts with no checkpoint; it only establishes that the
        # receipt-free tool did not create one.
        assert snapshot.checkpoint is None
        _assert_no_artifact_publication(snapshot, workspace_id=backend.sessions_dir.name)
    finally:
        await loop.aclose()

    restart_provider = ScriptedProvider([LLMResponse(content="restart complete")])
    restarted = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=restart_provider,
    )
    try:
        assert not restarted.tools.has(tool_name)
        outcome = await restarted.execute_turn(TurnRequest(content="continue", session_key=key))

        assert outcome.error is None, outcome.error
        assert outcome.content == "restart complete"
        assert adapter.executions == 1
        assert len(restart_provider.calls) == 1
        assert any(
            message.get("role") == "tool" and message.get("content") == result_text
            for message in restart_provider.calls[0][0]
        )
        _assert_no_artifact_publication(
            backend.source.load_source_snapshot(key),
            workspace_id=backend.sessions_dir.name,
        )
    finally:
        await restarted.aclose()


async def test_irreducibly_oversized_receipt_free_output_aborts_without_transport_or_redispatch(
    engine_config,
) -> None:
    """An indivisible receipt-free result fails closed after one real execution."""
    engine_config.artifacts.enabled = True
    key = "cli:receipt-free-overflow"
    tool_name = "receipt_free_overflow_fixture"
    marker = "receipt-free-overflow-marker"
    result_text = (marker + " caf\u00e9 \u6f22\u5b57 artifact: literal-only\n") * 1_500
    adapter = _ReceiptFreeTool(tool_name, result_text)
    provider = _OverflowProvider(tool_name, marker)
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=provider,
        context_window_tokens=12_000,
    )
    loop.tools.register(adapter)
    backend = _upgrade(loop, key)
    try:
        outcome = await loop.execute_turn(TurnRequest(content="run irreducible receipt-free output", session_key=key))

        assert outcome.error is not None and "context budget" in outcome.error
        assert adapter.executions == 1
        # One initial main request only: the indivisible group did not receive
        # an auxiliary summary request or a later main transport.
        assert len(provider.calls) == 1
        snapshot = backend.source.load_source_snapshot(key)
        canonical = next(event for event in snapshot.events if event.get("name") == tool_name)
        assert canonical["content"] == result_text
        assert canonical["content"].encode("utf-8") == result_text.encode("utf-8")
        assert snapshot.checkpoint is None
        _assert_no_artifact_publication(snapshot, workspace_id=backend.sessions_dir.name)
    finally:
        await loop.aclose()

    restart_provider = ScriptedProvider([LLMResponse(content="restart handles retained inline result")])
    restarted = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=restart_provider,
        context_window_tokens=64_000,
    )
    try:
        assert not restarted.tools.has(tool_name)
        outcome = await restarted.execute_turn(TurnRequest(content="resume", session_key=key))

        assert outcome.error is None, outcome.error
        assert outcome.content == "restart handles retained inline result"
        assert adapter.executions == 1
        assert len(restart_provider.calls) == 1
        assert any(
            message.get("role") == "tool" and message.get("content") == result_text
            for message in restart_provider.calls[0][0]
        )
        _assert_no_artifact_publication(
            backend.source.load_source_snapshot(key),
            workspace_id=backend.sessions_dir.name,
        )
    finally:
        await restarted.aclose()
