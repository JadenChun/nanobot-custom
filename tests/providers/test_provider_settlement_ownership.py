"""Controller-issued settlement tests, not native-provider qualification."""

from __future__ import annotations

import json
import pickle
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.providers import conversation_state as state_module
from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderConversationState,
    ProviderConversationTransportReceipt,
)
from nanobot.providers.conversation_state import (
    ConversationRequestBinding,
    ProviderConversationStateController,
    ProviderStateSettlement,
)


def _issued():
    # Same explicit unit-test transport declarations as the original owning
    # module. Only the controller creates the sealed handle; no private
    # capability, synthetic settlement, live provider or credential is used.
    receipt = ProviderConversationTransportReceipt(
        provider="fixture", account="account-digest", endpoint="https://example.test/v1",
        protocol="fixture.responses", model="fixture-model",
    )
    provider = MagicMock(spec=LLMProvider)
    provider.can_resume_conversation_state.return_value = True
    provider.supports_bound_conversation_state.return_value = True
    provider.conversation_state_transport_receipt.return_value = receipt
    messages = [{"role": "user", "content": "unit request"}]
    binding = ConversationRequestBinding(
        session_namespace="unit:session", authority_namespace="foreground",
        execution_owner="owner-a", session_generation=1, projection_epoch=1,
        runtime_prefix={"runtime": "current"}, policy_prefix={"policy": "current"},
        tools=[], tool_choice="auto", generation_settings={"max_tokens": 100},
    )
    controller = ProviderConversationStateController(
        provider=provider, model="fixture-model", messages=messages,
    )
    controller.prepare_request(messages, context_window_tokens=8192, binding=binding)
    response = LLMResponse(
        content="unit response",
        provider_state=ProviderConversationState(
            kind="openai_responses", provider="openai:test", model="fixture-model", version=1,
            payload={"items": [{"type": "reasoning", "encrypted_content": "unit"}], "context_tokens": 42},
            pending_messages=[], transport_receipt=receipt, continuation_metadata_present=True,
        ),
    )
    handle = controller.observe_response(response, messages)
    assert type(handle) is ProviderStateSettlement
    candidate = handle.candidate_for_persistence()
    assert type(candidate) is ProviderConversationState
    return controller, handle, candidate, messages


@pytest.mark.parametrize("identity", ["exact", "equal-clone", "absent"])
def test_candidate_observation_is_boolean_identity_only(identity: str) -> None:
    _, handle, candidate, _ = _issued()
    observed = candidate if identity == "exact" else deepcopy(candidate) if identity == "equal-clone" else None
    assert handle.matches_candidate(observed) is (identity == "exact")
    assert handle.candidate_for_persistence() is candidate


@pytest.mark.parametrize("delta", [0, -1, 1])
def test_invocation_observation_does_not_return_the_token(delta: int) -> None:
    controller, handle, candidate, _ = _issued()
    assert handle.matches_invocation(controller._provisional_invocation + delta) is (delta == 0)
    assert handle.candidate_for_persistence() is candidate


@pytest.mark.parametrize("change", ["unchanged", "clone", "payload", "wrong-type", "missing-digest"])
def test_candidate_validation_retains_exact_identity_type_and_digest(change: str) -> None:
    _, handle, candidate, _ = _issued()
    checked = candidate
    if change == "clone":
        checked = deepcopy(candidate)
    elif change == "payload":
        candidate.payload["context_tokens"] += 1
    elif change == "wrong-type":
        checked = {"payload": "not a state"}
    elif change == "missing-digest":
        handle._candidate_digest = None
    assert handle.is_unchanged_candidate(checked) is (change == "unchanged")


def test_exact_candidate_validation_keeps_repeated_digest_reads_and_live_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, handle, candidate, _ = _issued()
    original_read = ProviderStateSettlement.__getattribute__
    original_digest = state_module._digest
    trace: list[str] = []

    def read(self, name):
        if self is handle and name in {"_candidate", "_candidate_digest"}:
            trace.append(name)
        return original_read(self, name)

    def digest(value):
        trace.append("digest")
        return original_digest(value)

    monkeypatch.setattr(ProviderStateSettlement, "__getattribute__", read)
    monkeypatch.setattr(state_module, "_digest", digest)
    assert handle.is_unchanged_candidate(candidate) is True
    assert trace == ["_candidate", "_candidate_digest", "digest", "_candidate_digest"]


def test_identity_mismatch_short_circuits_before_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    _, handle, candidate, _ = _issued()

    def forbidden_digest(_value):
        pytest.fail("identity mismatch reached digest")

    monkeypatch.setattr(state_module, "_digest", forbidden_digest)
    assert handle.is_unchanged_candidate(deepcopy(candidate)) is False


def test_current_owner_operation_keeps_candidate_then_invocation_read_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, handle, _, _ = _issued()
    original_read = ProviderStateSettlement.__getattribute__
    trace: list[str] = []

    def read(self, name):
        if self is handle and name in {"_candidate", "_token"}:
            trace.append(name)
        return original_read(self, name)

    monkeypatch.setattr(ProviderStateSettlement, "__getattribute__", read)
    assert controller.is_current_settlement(handle) is True
    assert trace == ["_candidate", "_token"]


@pytest.mark.parametrize("operation", ["current", "commit", "discard"])
def test_public_controller_operation_keeps_live_private_dispatch(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    controller, handle, candidate, _ = _issued()
    private = {
        "current": "_settlement_is_current", "commit": "_commit_settlement", "discard": "_discard_settlement",
    }[operation]
    original = getattr(controller, private)
    calls: list[tuple[object, ...]] = []

    def observe(*arguments):
        calls.append(arguments)
        return original(*arguments)

    monkeypatch.setattr(controller, private, observe)
    if operation == "current":
        assert controller.is_current_settlement(handle) is True
        assert calls == [(handle,)]
    elif operation == "commit":
        assert controller.commit_provider_settlement(handle, candidate) is True
        assert calls == [(handle, candidate)]
        assert handle.candidate_for_persistence() is None
    else:
        assert controller.discard_provider_settlement(handle) is True
        assert calls == [(handle,)]
        assert handle.candidate_for_persistence() is None


def test_revocation_is_terminal_and_cannot_be_committed_through_controller() -> None:
    controller, handle, candidate, messages = _issued()
    assert handle.is_revoked() is False
    assert handle.revoke() is None
    assert handle.is_revoked() is True
    assert controller.is_current_settlement(handle) is False
    assert handle.candidate_for_persistence() is None
    assert not handle.commit_persisted(candidate)
    assert not controller.commit_provider_settlement(handle, candidate)
    assert not controller.discard_provider_settlement(handle)
    assert controller.checkpoint(messages) is None
    assert handle.revoke() is None


def test_revocation_keeps_the_original_close_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    _, handle, _, _ = _issued()
    original = ProviderStateSettlement._close
    calls: list[object] = []

    def close(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(ProviderStateSettlement, "_close", close)
    assert handle.revoke() is None
    assert calls == [handle]
    assert handle.is_revoked() is True


@pytest.mark.parametrize("operation", ["commit", "discard"])
def test_foreign_controller_cannot_adopt_or_discard_its_newer_candidate(operation: str) -> None:
    _, foreign, foreign_candidate, _ = _issued()
    controller, current, current_candidate, _ = _issued()
    assert controller.is_current_settlement(foreign) is False
    if operation == "commit":
        assert not controller.commit_provider_settlement(foreign, foreign_candidate)
    else:
        assert not controller.discard_provider_settlement(foreign)
    assert foreign.candidate_for_persistence() is None
    assert current.candidate_for_persistence() is current_candidate
    assert controller.is_current_settlement(current) is True


def test_exact_persisted_candidate_is_one_shot_and_restores_no_late_handle(tmp_path: Path) -> None:
    controller, handle, candidate, messages = _issued()
    record = candidate.to_private_record()
    path = tmp_path / "unit-provider-state.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert controller.commit_provider_settlement(handle, candidate) is True
    assert handle.is_revoked() is True
    assert controller.checkpoint(messages) is not None
    assert not controller.commit_provider_settlement(handle, candidate)
    assert not handle.commit_persisted(candidate)
    assert not handle.discard()


def test_mutated_exact_candidate_is_not_adopted_by_owner_operation() -> None:
    controller, handle, candidate, messages = _issued()
    candidate.payload["context_tokens"] += 1
    assert controller.commit_provider_settlement(handle, candidate) is False
    assert handle.is_revoked() is True
    assert controller.checkpoint(messages) is None


@pytest.mark.parametrize("copy", ["pickle", "deepcopy"])
def test_typed_observations_do_not_make_the_handle_copyable(copy: str) -> None:
    _, handle, candidate, _ = _issued()
    assert handle.matches_candidate(candidate) is True
    with pytest.raises(TypeError, match="cannot be serialized"):
        if copy == "pickle":
            pickle.dumps(handle)
        else:
            deepcopy(handle)
