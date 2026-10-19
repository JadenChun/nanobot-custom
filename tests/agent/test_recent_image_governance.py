"""Image projection precedes budgets and retains canonical source ownership."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest

from nanobot.agent.context_governance import (
    ContextGovernanceConfig,
    ContextGovernor,
    ModelRequestState,
)
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnRequest
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.providers.conversation_state import ProviderConversationStateController
from nanobot.providers.image_history import recent_image_messages
from nanobot.session.source import SourceValidationError, validate_execution_links
from nanobot.session.source_projection import RuntimeNode, SourceNode
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_source_engine import upgrade
from tests.providers.test_recent_image_history import CapturingProvider, history, image, images

engine_config = engine_fixtures.engine_config


def state_for(messages, source):
    provider = CapturingProvider()
    ledger = SourceRequestLedger(SourceTranscript(tuple(
        SourceNode(str(index), message) for index, message in enumerate(messages))),
        initial_eligible_ids=frozenset()) if source else None
    return ModelRequestState(
        ContextGovernanceConfig(provider, "fixture", ToolRegistry(), None, "test:images", 16000,
                                context_window_tokens=4096, max_tokens=256, preserve_source=source),
        ProviderConversationStateController(provider=provider, model="fixture", messages=messages,
                                            session_id="test:images"), source=ledger,
    )


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("tools", [None, []])
async def test_budget_sees_only_selected_images_and_keeps_all_text(monkeypatch, source, tools):
    raw = history()
    before = deepcopy(raw)
    state = state_for(raw, source)
    estimates = []

    def estimate(provider, model, messages, definitions):
        estimates.append(deepcopy(messages))
        return (900 * len(images(messages)) + 50, "image fixture")

    monkeypatch.setattr("nanobot.agent.context_governance.estimate_prompt_tokens_chain", estimate)
    result, context = await ContextGovernor().prepare_request(
        state, deepcopy(raw), tool_definitions=tools, transcript=raw,
    )
    assert images(result) == [image("new-a"), image("new-b")]
    assert estimates and all(images(request) == images(result) for request in estimates)
    # Legacy preparation merges adjacent users, but no text is snipped.
    expected = recent_image_messages(before)
    if not source:
        expected = ContextGovernor._merge_adjacent_user_messages_for_model(expected)
    assert result == expected
    assert context.request_limits.total_tokens == 4096 and context.session_id == "test:images"
    assert context.conversation_state is None
    if source:
        assert state.source.recovery_transcript.messages() == before
        assert state.source_view.messages() == result == raw
        assert tuple(node.event_id for node in state.source_view.nodes) == ("0", "1", "2", "3")
    else:
        assert raw == before


async def test_request_only_notes_are_not_published_and_forged_views_still_fail():
    raw = history()
    state = state_for(raw, True)
    before = state.source.recovery_transcript
    note = RuntimeNode("retry", {"role": "user", "content": "request only"})
    view = state.source.transcript.with_runtime(note)
    governor = ContextGovernor()
    result, _ = await governor.prepare_request(state, view.messages(), transcript=raw,
                                              source_view=view, tool_definitions=[])
    assert state.source_view.nodes[-1] == note
    assert len(state.source.transcript.nodes) == 4 and len(result) == 5
    assert state.source.recovery_transcript == before
    assert state.source.eligible_ids == frozenset()
    epoch = state.source.projection_epoch
    ticket = state.source.issue(state.source_view)
    with pytest.raises(SourceValidationError, match="outstanding"):
        governor.project_source_images(state, state.source_view, raw)
    state.source.discard(ticket)
    await governor.prepare_request(state, state.source_view.messages(), transcript=raw,
                                   source_view=state.source_view, tool_definitions=[])
    assert state.source.projection_epoch == epoch
    forged = SourceTranscript((replace(state.source.transcript.nodes[0], message={"role": "user", "content": "forged"}),
                               *state.source.transcript.nodes[1:]))
    with pytest.raises(SourceValidationError, match="active projection"):
        await governor.prepare_request(state, forged.messages(), source_view=forged, tool_definitions=[])
    assert state.source.recovery_transcript == before


async def test_legacy_window_precedes_user_merge_and_invalidates_native_estimate(monkeypatch):
    raw = [history()[0], history()[2], history()[3]]
    state = state_for(raw, False)
    invalidated = Mock(wraps=state.conversation.invalidate)
    monkeypatch.setattr(state.conversation, "invalidate", invalidated)
    result, context = await ContextGovernor().prepare_request(state, raw, transcript=raw, tool_definitions=None)
    assert images(result) == [image("new-a"), image("new-b")]
    assert invalidated.called and context.conversation_state is None
    assert len(images(raw)) == 4


def tool_images(label, inline):
    blocks = [image(label + "-a"), image(label + "-b")]
    if not inline:
        for index, block in enumerate(blocks):
            block["image_url"]["url"] = f"https://images.invalid/{label}-{index}.png"
    return blocks


class CaptureImages(Tool):
    name = "fixture_images"
    description = "Return two fixture attachments."
    parameters = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}

    def __init__(self, inline):
        self.inline = inline

    async def execute(self, label, **kwargs):
        return tool_images(label, self.inline)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("finalization", [False, True])
async def test_source_runner_advances_image_window_and_preserves_durable_tool_events(engine_config, inline, finalization):
    if finalization:
        engine_config.agents.defaults.max_tool_iterations = 2
    provider = CapturingProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest("one", "fixture_images", {"label": "old"})]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest("two", "fixture_images", {"label": "new"})]),
        LLMResponse(content="compared"), LLMResponse(content="follow-up"),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    loop.tools.register(CaptureImages(inline))
    backend = upgrade(loop)
    try:
        result = await loop.execute_turn(TurnRequest(content="capture twice", session_key="cli:source"))
        assert result.error is None and result.content == "compared"
        assert len(provider.calls) == 3
        assert images(provider.calls[1][0]) == tool_images("old", inline)
        assert images(provider.calls[2][0]) == tool_images("new", inline)
        events = backend.source.load_source_snapshot("cli:source").events
        assert not validate_execution_links(events)
        # Existing persistence never saves inline data images. Public references
        # remain durable and must not be replaced by the model-window notice.
        assert len(images(events)) == (0 if inline else 4)
        tools = [event for event in events if event["role"] == "tool"]
        assert [row["tool_call_id"] for row in tools] == ["one", "two"]
        persisted_old = ([{"type": "text", "text": "[Image data unavailable in persisted source.]"}] * 2
                         if inline else tool_images("old", False))
        assert tools[0]["content"] == persisted_old
        followup = await loop.execute_turn(TurnRequest(content="text follow-up", session_key="cli:source"))
        assert followup.error is None and followup.content == "follow-up"
        assert images(provider.calls[-1][0]) == ([] if inline else tool_images("new", False))
        assert all(call[3] is not None and call[3].fits for call in provider.calls)
        assert recent_image_messages(provider.calls[-1][0]) == provider.calls[-1][0]
        assert backend.source.load_source_snapshot("cli:source").events[:len(events)] == events
    finally:
        await loop.aclose()
