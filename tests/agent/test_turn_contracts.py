from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import (
    ApprovalGrant,
    DeliveryTarget,
    HistoryMode,
    RunRecord,
    RunStatus,
    ScheduledTurnLink,
    SessionRunRef,
    ToolOutcome,
    TraceMode,
    TurnContext,
    TurnRequest,
    TurnResult,
    TurnSource,
)
from nanobot.session.manager import Session


def test_turn_request_is_frozen_and_enums_are_typed() -> None:
    request = TurnRequest(
        prompt="hello",
        source=TurnSource.DIRECT,
        history_mode=HistoryMode.FRESH,
        trace_mode=TraceMode.SANITIZED,
    )

    assert request.source.value == "direct"
    assert request.history_mode.value == "fresh"
    assert request.trace_mode.value == "sanitized"
    assert RunStatus.RUNNING.value == "running"
    with pytest.raises(FrozenInstanceError):
        request.prompt = "changed"


def test_turn_context_is_mutable_per_run_state() -> None:
    context = TurnContext(request=TurnRequest(prompt="hello"), run_id="run-1")
    context.iteration = 2
    context.cancelled = True

    assert context.iteration == 2
    assert context.cancelled is True


class _ContextTool(Tool):
    @property
    def name(self) -> str:
        return "context_tool"

    @property
    def description(self) -> str:
        return "test context plumbing"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, context: TurnContext | None = None, **_: Any) -> Any:
        return ToolOutcome(
            content={"run_id": context.run_id if context is not None else ""},
            stop_reason="done",
            policy_metadata={"allowed": True},
        )


@pytest.mark.asyncio
async def test_registry_context_path_returns_tool_outcome_and_legacy_path_stays_bare() -> None:
    registry = ToolRegistry()
    registry.register(_ContextTool())
    context = TurnContext(request=TurnRequest(prompt="hello"), run_id="run-2")

    contextual = await registry.execute("context_tool", {}, context=context)
    legacy = await registry.execute("context_tool", {})

    assert isinstance(contextual, ToolOutcome)
    assert contextual.content == {"run_id": "run-2"}
    assert contextual.stop_reason == "done"
    assert legacy == {"run_id": ""}


def test_session_history_omits_internal_metadata_recursively() -> None:
    session = Session(key="direct:test")
    session.messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello", "_run_id": "run-1"}],
            "_run_id": "run-1",
            "_internal": "secret",
        }
    )
    session.messages.append(
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"id": "call-1", "_run_id": "run-1"}],
            "_run_id": "run-1",
        }
    )

    history = session.get_history()

    assert history == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "call-1"}]},
    ]
    assert all("_run_id" not in repr(message) for message in history)


def test_approval_grant_is_available_as_a_typed_value() -> None:
    grant = ApprovalGrant(approved=True, grant_id="approval-1")
    assert grant.approved is True
    assert grant.grant_id == "approval-1"


@pytest.mark.parametrize("source", [TurnSource.SDK, "sdk"])
def test_turn_request_retains_runtime_enum_and_sequence_normalization(source) -> None:
    request = TurnRequest(
        content="hello",
        source=source,
        history_mode="fresh",
        trace_mode="sanitized",
        media=["image.png"],
        hooks=["hook"],
        tool_names=["read_file"],
    )

    assert request.source is TurnSource.SDK
    assert request.history_mode is HistoryMode.FRESH
    assert request.trace_mode is TraceMode.SANITIZED
    assert request.media == ("image.png",)
    assert request.hooks == ("hook",)
    assert request.tool_names == ("read_file",)
    assert request.message == request.prompt == request.content == "hello"


@pytest.mark.parametrize("reference_field", ["detail_ref", "session_ref"])
def test_run_record_retains_checked_mapping_reference_normalization(reference_field) -> None:
    record = RunRecord.from_dict({
        "run_id": "run-compat",
        "status": "completed",
        "source": "sdk",
        reference_field: {"session_key": 42, "run_id": 7, 99: "ignored"},
        "tools_used": ["read_file"],
    })

    reference = SessionRunRef(session_key="42", run_id="7")
    assert record.detail_ref == reference
    assert record.session_ref == reference
    assert record.session_key == "42"
    assert record.status is RunStatus.COMPLETED
    assert record.source is TurnSource.SDK
    assert record.tools_used == ("read_file",)
    assert record.to_dict()["detail_ref"] == {"session_key": "42", "run_id": "7"}


@pytest.mark.parametrize("payload", [None, [], {}, {"run_id": ""}])
def test_run_record_retains_nonmapping_and_missing_identity_refusal(payload) -> None:
    with pytest.raises(ValueError, match="run record must include a run_id"):
        RunRecord.from_dict(payload)


def test_scheduled_link_and_turn_result_retain_sequence_normalization() -> None:
    destination = DeliveryTarget(channel="telegram", chat_id="42")
    link = ScheduledTurnLink(job_id="job", additional_destinations=[destination])
    result = TurnResult(
        status="completed",
        tools_used=("read_file",),
        messages=({"role": "assistant", "content": "done"},),
    )

    assert link.additional_destinations == (destination,)
    assert link.schedule_id == "job"
    assert result.status is RunStatus.COMPLETED
    assert result.tools_used == ["read_file"]
    assert result.messages == [{"role": "assistant", "content": "done"}]
