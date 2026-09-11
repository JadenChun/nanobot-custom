"""Tests for provider-owned conversation-state lifecycle coordination."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
    ProviderConversationState,
    ProviderConversationTransportReceipt,
    ToolCallRequest,
)
from nanobot.providers.conversation_state import (
    ConversationRequestBinding,
    ProviderConversationStateController,
    allows_conversation_message_merge,
)


def _provider(*, resumable: bool = True, compact: bool = False) -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.can_resume_conversation_state.return_value = resumable
    provider.supports_native_compaction.return_value = compact
    provider.supports_bound_conversation_state.return_value = True
    provider.conversation_state_transport_receipt.return_value = (
        ProviderConversationTransportReceipt(
            provider="fixture",
            account="account-digest",
            endpoint="https://example.test/v1",
            protocol="fixture.responses",
            model="gpt-5.6",
        )
    )
    return provider


def _binding(**changes: object) -> ConversationRequestBinding:
    values: dict[str, object] = {
        "session_namespace": "webui:session",
        "authority_namespace": "foreground",
        "execution_owner": "owner-a",
        "session_generation": 1,
        "projection_epoch": 1,
        "runtime_prefix": {"runtime": "current"},
        "policy_prefix": {"policy": "current"},
        "tools": [],
        "tool_choice": {"mode": "auto"},
        "generation_settings": GenerationSettings(temperature=0.2, max_tokens=100),
    }
    values.update(changes)
    return ConversationRequestBinding(**values)


def _state(label: str, *, pending: list[dict] | None = None) -> ProviderConversationState:
    return ProviderConversationState(
        kind="openai_responses",
        provider="openai:test",
        model="gpt-5.6",
        version=1,
        payload={
            "items": [{"type": "reasoning", "encrypted_content": label}],
            "context_tokens": 42,
        },
        pending_messages=pending or [],
    )


def test_controller_replays_only_messages_after_provider_output() -> None:
    provider = _provider()
    binding = _binding()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "run a tool"},
    ]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
    )
    state = _state("first")

    controller.prepare_request(messages, context_window_tokens=200_000, binding=binding)
    response = LLMResponse(content=None, provider_state=state)
    invocation = controller.observe_response(response, messages)
    assert invocation is not None
    assert controller.settle_response(invocation, messages)
    assert allows_conversation_message_merge(messages[-1]) is False

    messages.append(controller.project_response_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function"}],
        },
        response,
    ))
    tool_message = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool result",
    }
    messages.append(tool_message)

    provider_context = controller.prepare_request(
        messages,
        context_window_tokens=200_000,
        binding=binding,
    )

    assert provider_context is not None
    assert provider_context.conversation_state is not None
    assert provider_context.conversation_state.payload == state.payload
    assert provider_context.conversation_state.pending_messages == [tool_message]
    assert controller.checkpoint(messages).pending_messages == [tool_message]


def test_controller_rejects_same_length_edit_to_accepted_prefix() -> None:
    """A count boundary must not turn an edited prefix into a resumed delta."""
    provider = _provider()
    accepted = [{"role": "user", "content": "original instruction"}]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
        state=_state("saved"),
    )

    context = controller.prepare_request(
        [{"role": "user", "content": "edited instruction"}],
        context_window_tokens=200_000,
        binding=_binding(),
    )

    assert context is not None
    assert context.conversation_state is None


def test_bound_state_requires_settlement_and_reuses_only_append_delta() -> None:
    provider = _provider()
    accepted = [{"role": "user", "content": "private original instruction"}]
    binding = ConversationRequestBinding(
        session_namespace="webui:private-session",
        authority_namespace="foreground",
        execution_owner="owner-a",
        session_generation=4,
        projection_epoch=8,
        runtime_prefix={"runtime": "current"},
        policy_prefix={"policy": "tools-scoped"},
        tools=[{"type": "function", "name": "read_file"}],
        tool_choice="auto",
        generation_settings={"temperature": 0.2, "max_tokens": 100},
    )
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
        session_id="webui:private-session",
    )
    controller.prepare_request(
        accepted,
        context_window_tokens=200_000,
        binding=binding,
    )
    invocation = controller.observe_response(
        LLMResponse(content="done", provider_state=_state("candidate")),
        accepted,
    )

    assert invocation is not None
    before_settlement = controller.prepare_request(
        [*accepted, {"role": "user", "content": "next"}],
        context_window_tokens=200_000,
        binding=binding,
    )
    assert before_settlement is not None
    assert before_settlement.conversation_state is None
    assert not controller.settle_response(invocation, accepted)

    # The abandoned dispatch above invalidated the pending candidate. Stage a
    # new one and settle it after the caller's durable-record boundary.
    controller.prepare_request(
        accepted,
        context_window_tokens=200_000,
        binding=binding,
    )
    invocation = controller.observe_response(
        LLMResponse(content="done", provider_state=_state("candidate")),
        accepted,
    )
    assert invocation is not None
    assert controller.settle_response(invocation, accepted)

    resumed = controller.prepare_request(
        [*accepted, {"role": "user", "content": "next"}],
        context_window_tokens=200_000,
        binding=binding,
    )
    assert resumed is not None
    assert resumed.conversation_state is not None
    assert resumed.conversation_state.pending_messages == [{
        "role": "user",
        "content": "next",
    }]
    state = controller.finish(accepted)
    assert state is not None
    private_record = state.to_private_record()
    assert private_record["conversation_binding"] is not None
    assert "private original instruction" not in str(private_record)
    assert "foreground" not in str(private_record)


@pytest.mark.parametrize(
    "change_binding",
    [
        lambda value: replace(value, policy_prefix={"policy": "changed"}),
        lambda value: replace(value, runtime_prefix={"runtime": "changed"}),
        lambda value: replace(value, tools=[{"type": "function", "name": "write_file"}]),
        lambda value: replace(value, tool_choice={"type": "required"}),
        lambda value: replace(value, generation_settings={"temperature": 0.9}),
        lambda value: replace(value, execution_owner="owner-b"),
        lambda value: replace(value, session_generation=2),
        lambda value: replace(value, projection_epoch=3),
    ],
)
def test_bound_state_rejects_changed_request_authority_or_policy(
    change_binding,
) -> None:
    provider = _provider()
    accepted = [{"role": "user", "content": "original"}]
    original = _binding(
        session_namespace="webui:session",
        authority_namespace="foreground",
        policy_prefix={"policy": "original"},
        runtime_prefix={"runtime": "original"},
        tools=[{"type": "function", "name": "read_file"}],
        tool_choice="auto",
        generation_settings={"temperature": 0.2},
        execution_owner="owner-a",
        session_generation=1,
        projection_epoch=2,
    )
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
    )
    controller.prepare_request(accepted, context_window_tokens=200_000, binding=original)
    invocation = controller.observe_response(
        LLMResponse(content="done", provider_state=_state("candidate")),
        accepted,
    )
    assert invocation is not None
    assert controller.settle_response(invocation, accepted)
    state = controller.finish(accepted)
    assert state is not None
    changed_binding = change_binding(original)

    reloaded = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
        state=state,
    )
    context = reloaded.prepare_request(
        [*accepted, {"role": "user", "content": "append"}],
        context_window_tokens=200_000,
        binding=changed_binding,
    )

    assert context is not None
    assert context.conversation_state is None


def test_bound_state_rejects_transport_receipt_change_and_unknown_context() -> None:
    provider = _provider()
    accepted = [{"role": "user", "content": "original"}]
    binding = _binding()
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
    )
    controller.prepare_request(accepted, context_window_tokens=200_000, binding=binding)
    invocation = controller.observe_response(
        LLMResponse(content="done", provider_state=_state("candidate")),
        accepted,
    )
    assert invocation is not None
    assert controller.settle_response(invocation, accepted)
    state = controller.finish(accepted)
    assert state is not None

    provider.conversation_state_transport_receipt.return_value = (
        ProviderConversationTransportReceipt(
            provider="fixture",
            account="rotated-account-digest",
            endpoint="https://example.test/v1",
            protocol="fixture.responses",
            model="gpt-5.6",
        )
    )
    reloaded = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
        state=state,
    )
    context = reloaded.prepare_request(
        [*accepted, {"role": "user", "content": "append"}],
        context_window_tokens=200_000,
        binding=binding,
    )
    assert context is not None
    assert context.conversation_state is None

    no_context = ProviderConversationStateController(
        provider=_provider(),
        model="gpt-5.6",
        messages=accepted,
        state=ProviderConversationState(
            kind="openai_responses",
            provider="openai:test",
            model="gpt-5.6",
            version=1,
            payload={"items": []},
        ),
    )
    unknown = no_context.prepare_request(
        accepted,
        context_window_tokens=200_000,
        binding=_binding(),
    )
    assert unknown is not None
    assert unknown.conversation_state is None


def test_persisted_bound_state_rejects_same_length_prefix_edit() -> None:
    provider = _provider()
    accepted = [{"role": "user", "content": "original"}]
    binding = _binding(policy_prefix={"policy": "stable"})
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
    )
    controller.prepare_request(accepted, context_window_tokens=200_000, binding=binding)
    invocation = controller.observe_response(
        LLMResponse(content="done", provider_state=_state("candidate")),
        accepted,
    )
    assert invocation is not None
    assert controller.settle_response(invocation, accepted)
    state = controller.finish(accepted)
    assert state is not None

    reloaded = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=[{"role": "user", "content": "modified"}],
        state=state,
    )
    context = reloaded.prepare_request(
        [{"role": "user", "content": "modified"}],
        context_window_tokens=200_000,
        binding=binding,
    )
    assert context is not None
    assert context.conversation_state is None


def test_late_or_duplicate_settlement_cannot_restore_discarded_state() -> None:
    provider = _provider()
    messages = [{"role": "user", "content": "original"}]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
    )
    controller.prepare_request(messages, context_window_tokens=200_000, binding=_binding())
    invocation = controller.observe_response(
        LLMResponse(content="done", provider_state=_state("candidate")),
        messages,
    )
    assert invocation is not None
    assert controller.discard_response(invocation)
    assert not controller.settle_response(invocation, messages)
    assert controller.finish(messages) is None


def test_unsettled_tool_response_forces_full_stateless_rebuild() -> None:
    """An unrecorded tool call cannot be omitted behind an older handle."""
    provider = _provider()
    binding = _binding()
    messages = [{"role": "user", "content": "start"}]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
    )
    controller.prepare_request(messages, context_window_tokens=200_000, binding=binding)
    first = LLMResponse(content=None, provider_state=_state("first"))
    first_ticket = controller.observe_response(first, messages)
    assert first_ticket is not None
    assert controller.settle_response(first_ticket, messages)
    messages.extend([
        controller.project_response_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            first,
        ),
        {"role": "tool", "tool_call_id": "call_1", "content": "first result"},
    ])
    resumed = controller.prepare_request(
        messages,
        context_window_tokens=200_000,
        binding=binding,
    )
    assert resumed is not None
    assert resumed.conversation_state is not None

    second = LLMResponse(content=None, provider_state=_state("second"))
    second_ticket = controller.observe_response(second, messages)
    assert second_ticket is not None
    messages.extend([
        controller.project_response_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_2", "type": "function"}],
            },
            second,
        ),
        {"role": "tool", "tool_call_id": "call_2", "content": "second result"},
    ])

    rebuilt = controller.prepare_request(
        messages,
        context_window_tokens=200_000,
        binding=binding,
    )

    assert rebuilt is not None
    assert rebuilt.conversation_state is None
    assert messages[-2]["tool_calls"][0]["id"] == "call_2"
    assert controller.finish(messages) is None


def test_private_metadata_forgery_cannot_fall_back_to_legacy_resume() -> None:
    provider = _provider()
    binding = _binding()
    accepted = [{"role": "user", "content": "original"}]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
    )
    controller.prepare_request(accepted, context_window_tokens=200_000, binding=binding)
    ticket = controller.observe_response(
        LLMResponse(content="done", provider_state=_state("candidate")),
        accepted,
    )
    assert ticket is not None
    assert controller.settle_response(ticket, accepted)
    state = controller.finish(accepted)
    assert state is not None
    record = state.to_private_record()

    malformed = deepcopy(record)
    malformed["conversation_binding"] = {"version": 1}
    malformed_state = ProviderConversationState.from_private_record(malformed)
    assert malformed_state is not None
    malformed_controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
        state=malformed_state,
    )
    malformed_context = malformed_controller.prepare_request(
        accepted,
        context_window_tokens=200_000,
        binding=binding,
    )
    assert malformed_context is not None
    assert malformed_context.conversation_state is None

    partial = deepcopy(record)
    partial.pop("transport_receipt")
    partial_state = ProviderConversationState.from_private_record(partial)
    assert partial_state is not None
    partial_controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
        state=partial_state,
    )
    partial_context = partial_controller.prepare_request(
        accepted,
        context_window_tokens=200_000,
        binding=binding,
    )
    assert partial_context is not None
    assert partial_context.conversation_state is None

    legacy = deepcopy(record)
    legacy.pop("conversation_binding")
    legacy.pop("transport_receipt")
    legacy_state = ProviderConversationState.from_private_record(legacy)
    assert legacy_state is not None
    legacy_controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
        state=legacy_state,
    )
    legacy_context = legacy_controller.prepare_request(
        accepted,
        context_window_tokens=200_000,
        binding=binding,
    )
    assert legacy_context is not None
    assert legacy_context.conversation_state is None


def test_binding_canonicalizes_real_generation_settings_and_rejects_opaque_values() -> None:
    provider = _provider()
    accepted = [{"role": "user", "content": "original"}]
    binding = _binding(generation_settings=GenerationSettings(
        temperature=0.25,
        max_tokens=256,
        reasoning_effort="low",
    ))
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
    )
    controller.prepare_request(accepted, context_window_tokens=200_000, binding=binding)
    ticket = controller.observe_response(
        LLMResponse(content="done", provider_state=_state("candidate")),
        accepted,
    )
    assert ticket is not None

    opaque = _binding(generation_settings=object())
    opaque_controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=accepted,
    )
    opaque_controller.prepare_request(
        accepted,
        context_window_tokens=200_000,
        binding=opaque,
    )
    assert opaque_controller.observe_response(
        LLMResponse(content="done", provider_state=_state("opaque")),
        accepted,
    ) is None


def test_underspecified_or_generic_provider_paths_are_stateless() -> None:
    messages = [{"role": "user", "content": "original"}]
    provider = _provider()
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
    )
    no_binding = controller.prepare_request(messages, context_window_tokens=200_000)
    assert no_binding is not None
    assert no_binding.conversation_state is None
    assert controller.observe_response(
        LLMResponse(content="done", provider_state=_state("candidate")),
        messages,
    ) is None
    assert controller.finish(messages) is None

    estimated = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
        state=_state("saved"),
    ).estimate_request_context_tokens(messages)
    assert estimated is None

    generic = _provider()
    generic.supports_bound_conversation_state.return_value = False
    generic_controller = ProviderConversationStateController(
        provider=generic,
        model="gpt-5.6",
        messages=messages,
        state=_state("saved"),
    )
    generic_context = generic_controller.prepare_request(
        messages,
        context_window_tokens=200_000,
        binding=_binding(),
    )
    assert generic_context is not None
    assert generic_context.conversation_state is None


def test_controller_uses_governed_messages_for_provider_state_delta() -> None:
    provider = _provider()
    binding = _binding()
    messages = [
        {"role": "user", "content": "run a tool"},
    ]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
    )
    state = _state("first")

    controller.prepare_request(messages, context_window_tokens=200_000, binding=binding)
    response = LLMResponse(content=None, provider_state=state)
    invocation = controller.observe_response(response, messages)
    assert invocation is not None
    assert controller.settle_response(invocation, messages)
    messages.extend([
        controller.project_response_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            response,
        ),
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "raw oversized result",
        },
    ])
    governed_messages = [
        messages[0],
        messages[1],
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "compacted result",
        },
    ]

    provider_context = controller.prepare_request(
        messages,
        context_window_tokens=200_000,
        model_messages=governed_messages,
        binding=binding,
    )

    assert provider_context is not None
    assert provider_context.conversation_state is not None
    assert provider_context.conversation_state.pending_messages == [{
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "compacted result",
    }]
    assert controller.checkpoint(messages).pending_messages[-1]["content"] == (
        "raw oversized result"
    )
    governed_checkpoint = controller.checkpoint(
        messages,
        model_messages=governed_messages,
    )
    assert governed_checkpoint is not None
    assert governed_checkpoint.pending_messages[-1]["content"] == "compacted result"


def test_controller_estimates_active_state_plus_pending_delta(monkeypatch) -> None:
    provider = _provider()
    binding = _binding()
    current_message = {"role": "user", "content": "new delta"}
    state = ProviderConversationState(
        kind="openai_responses",
        provider="openai:test",
        model="gpt-5.6",
        version=1,
        payload={
            "items": [{"type": "reasoning", "encrypted_content": "opaque"}],
            "context_tokens": 450,
        },
        pending_messages=[current_message],
    )
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=[current_message],
    )
    controller.prepare_request(
        [current_message],
        context_window_tokens=200_000,
        binding=binding,
    )
    ticket = controller.observe_response(
        LLMResponse(content="done", provider_state=state),
        [current_message],
    )
    assert ticket is not None
    assert controller.settle_response(ticket, [current_message])
    seen = {}

    def estimate(_provider, _model, messages, tools):
        seen["messages"] = messages
        seen["tools"] = tools
        return 100, "test-counter"

    monkeypatch.setattr(
        "nanobot.providers.conversation_state.estimate_prompt_tokens_chain",
        estimate,
    )

    tokens = controller.estimate_request_context_tokens(
        [current_message],
        model_messages=[current_message],
        tool_definitions=[{"type": "web_search"}],
        binding=binding,
    )

    assert tokens == 550
    assert seen == {
        "messages": [{"role": "user", "content": "new delta"}],
        "tools": [{"type": "web_search"}],
    }


@pytest.mark.parametrize("mismatch", ["policy", "prefix", "receipt"])
def test_context_estimate_rejects_the_same_binding_mismatches_as_dispatch(
    monkeypatch,
    mismatch: str,
) -> None:
    accepted = [{"role": "user", "content": "original"}]
    binding = _binding()

    def settled_controller() -> tuple[MagicMock, ProviderConversationStateController]:
        provider = _provider()
        controller = ProviderConversationStateController(
            provider=provider,
            model="gpt-5.6",
            messages=accepted,
        )
        controller.prepare_request(
            accepted,
            context_window_tokens=200_000,
            binding=binding,
        )
        ticket = controller.observe_response(
            LLMResponse(content="done", provider_state=_state("candidate")),
            accepted,
        )
        assert ticket is not None
        assert controller.settle_response(ticket, accepted)
        return provider, controller

    provider, controller = settled_controller()
    messages = accepted
    estimate_binding = binding
    if mismatch == "policy":
        estimate_binding = replace(binding, policy_prefix={"policy": "changed"})
    elif mismatch == "prefix":
        messages = [{"role": "user", "content": "modified"}]
    else:
        provider.conversation_state_transport_receipt.return_value = (
            ProviderConversationTransportReceipt(
                provider="fixture",
                account="rotated-account-digest",
                endpoint="https://example.test/v1",
                protocol="fixture.responses",
                model="gpt-5.6",
            )
        )
    monkeypatch.setattr(
        "nanobot.providers.conversation_state.estimate_prompt_tokens_chain",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not estimate mismatched state")),
    )

    assert controller.estimate_request_context_tokens(
        messages,
        model_messages=messages,
        binding=estimate_binding,
    ) is None
    assert controller.finish(messages) is None


def test_transient_response_invalidates_state() -> None:
    provider = _provider()
    binding = _binding()
    current_message = {"role": "user", "content": "continue"}
    supplemental = {"role": "user", "content": "internal finalization retry"}
    messages = [{"role": "system", "content": "system"}, current_message]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
        state=_state("saved", pending=[
            {"role": "tool", "content": "prior"},
            current_message,
        ]),
    )

    provider_context = controller.prepare_request(
        messages,
        context_window_tokens=200_000,
        supplemental_messages=[supplemental],
        binding=binding,
    )
    assert provider_context is not None
    assert provider_context.conversation_state is None

    controller.observe_response(
        LLMResponse(
            content="temporary failure",
            finish_reason="error",
            error_kind="timeout",
        ),
        messages,
    )
    assert controller.finish(messages) is None


def test_non_retryable_response_discards_saved_state() -> None:
    provider = _provider()
    binding = _binding()
    messages = [{"role": "user", "content": "continue"}]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
        state=_state("saved"),
    )

    controller.prepare_request(messages, context_window_tokens=200_000, binding=binding)
    controller.observe_response(
        LLMResponse(
            content="invalid request",
            finish_reason="error",
            error_status_code=400,
            error_should_retry=False,
        ),
        messages,
    )

    assert controller.finish(messages) is None


@pytest.mark.parametrize(
    ("finish_reason", "exposes_tool_call"),
    [
        ("length", False),
        ("length", True),
        ("refusal", True),
        ("content_filter", True),
    ],
)
def test_terminal_response_discards_candidate_state(
    finish_reason: str,
    exposes_tool_call: bool,
) -> None:
    provider = _provider()
    binding = _binding()
    messages = [{"role": "user", "content": "continue"}]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
        state=_state("saved"),
    )

    controller.prepare_request(messages, context_window_tokens=200_000, binding=binding)
    candidate = ProviderConversationState(
        kind="openai_responses",
        provider="openai:test",
        model="gpt-5.6",
        version=1,
        payload={
            "items": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec",
                "arguments": "{}",
            }],
        },
    )
    response = LLMResponse(
        content="terminal response",
        tool_calls=(
            [ToolCallRequest(id="call_1", name="exec", arguments={})]
            if exposes_tool_call
            else []
        ),
        finish_reason=finish_reason,
        provider_state=candidate,
    )
    assert response.has_tool_calls is exposes_tool_call
    assert response.should_execute_tools is False

    controller.observe_response(response, messages)

    assert controller.finish(messages) is None


def test_independent_request_exposes_context_without_capability_check() -> None:
    provider = _provider(compact=False)
    messages = [{"role": "user", "content": "hello"}]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
        state=_state("saved"),
    )

    provider_context = controller.independent_request_context(
        context_window_tokens=200_000,
    )
    assert provider_context is not None
    assert provider_context.conversation_state is None
    assert provider_context.context_window_tokens == 200_000
    provider.supports_native_compaction.assert_not_called()


def test_independent_request_derives_hard_limits_from_runtime_capacity() -> None:
    provider = _provider(compact=False)
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=[{"role": "user", "content": "hello"}],
    )

    provider_context = controller.independent_request_context(
        context_window_tokens=200_000,
        completion_reserve_tokens=4_096,
    )

    assert provider_context is not None
    assert provider_context.request_limits is not None
    assert provider_context.request_limits.total_tokens == 200_000
    assert provider_context.request_limits.completion_reserve_tokens == 4_096


def test_independent_request_exposes_session_id_without_token_budget() -> None:
    provider = _provider(compact=False)
    messages = [{"role": "user", "content": "hello"}]
    controller = ProviderConversationStateController(
        provider=provider,
        model="gpt-5.6",
        messages=messages,
        session_id="webui:cache-test",
    )

    provider_context = controller.prepare_request(
        messages,
        context_window_tokens=None,
    )

    assert provider_context == ProviderCallContext(
        session_id="webui:cache-test",
    )
