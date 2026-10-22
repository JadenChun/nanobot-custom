from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.turn import RunStatus, TurnResult
from nanobot.channels.delivery_contract import ChannelTextPartReceipt, ChannelTextReceipt
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.callback_store import CronCallbackStore
from nanobot.cron.canonical_execution import build_claimed_fresh_request
from nanobot.cron.channel_sender import CronChannelSender
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.composition import (
    CronCompositionPreflightError,
    compose_claimed_text_occurrence,
)
from nanobot.cron.definition_store import CronDefinitionAction, CronDefinitionStore
from nanobot.cron.delivery_runtime import CronDeliveryRuntime, UnknownOutcome
from nanobot.cron.delivery_store import CronDeliveryOutboxStore
from nanobot.cron.occurrence_store import CronOccurrenceStore, OccurrenceExecutionStartedError
from nanobot.cron.service import CronService, _job_definition
from nanobot.cron.types import CronDestination, CronSchedule


def _claimed(tmp_path: Path, channels=("telegram", "telegram"), *, deliver=True, payload_overrides=None, timezone=None):
    root = tmp_path / "cron"
    namespace = "cron:composition-tests:" + tmp_path.name
    definitions = CronDefinitionStore(CronActionStore(root, namespace=namespace))
    occurrences = CronOccurrenceStore(root / "occurrences", namespace=namespace)
    service = CronService(
        root / "jobs.json",
        definition_store=definitions,
        occurrence_store=occurrences,
        completion_projector=CronCompletionProjector(occurrences, root / "runs"),
        delivery_runtime=CronDeliveryRuntime(
            CronDeliveryOutboxStore(root / "outbox", namespace=namespace),
            occurrence_store=occurrences,
        ),
    )
    job = service.add_job(
        "report", (CronSchedule(kind="cron", expr="0 9 * * *", tz=timezone) if timezone
                   else CronSchedule(kind="every", every_ms=60_000)), "compute",
        deliver=deliver,
        channel=channels[0] if deliver else None,
        to="A" if deliver else None,
        additional_destinations=(
            [CronDestination(channels[1], "B")] if deliver and len(channels) > 1 else None
        ),
    )
    job.payload.execution_mode = "legacy_fresh"
    job.payload.session_key = f"cron:{job.id}"
    job.payload.origin_channel = channels[0] if deliver else "cli"
    job.payload.origin_chat_id = "A" if deliver else "direct"
    for name, value in (payload_overrides or {}).items():
        setattr(job.payload, name, value)
    service._definition_action(CronDefinitionAction.update(
        _job_definition(job, job_instance_id=job.job_instance_id),
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    ))
    job = service.get_job(job.id)
    assert job is not None
    claim = service.claim_scheduled_occurrence(
        job.id, 1_700_000_000_000, owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    return service, claim


class ReceiptChannel:
    def __init__(self, calls, *, malformed_target=False):
        self.calls = calls
        self.malformed_target = malformed_target

    async def send_text_with_receipt(self, target, messages, *, action_id, request_digest):
        self.calls.append((target, tuple(messages)))
        return ChannelTextReceipt(
            channel="telegram",
            target="wrong" if self.malformed_target else target,
            action_id=action_id,
            request_digest=request_digest,
            parts=(ChannelTextPartReceipt(0, 0, 0, len(self.calls), int(target == "B") + 1),),
        )


async def test_frozen_topic_is_preserved_only_for_primary_destination(tmp_path):
    service, claim = _claimed(tmp_path, payload_overrides={"channel_meta": {"message_thread_id": 42}})
    observed = []
    sent = []

    class TopicChannel(ReceiptChannel):
        async def send_thread_text_with_receipt(self, target, messages, *, message_thread_id, action_id, request_digest):
            observed.append((target, message_thread_id))
            return await self.send_text_with_receipt(target, messages, action_id=action_id, request_digest=request_digest)

    sender = CronChannelSender(lambda _name: TopicChannel(sent))
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(run_id=claim.run_id, content="topic result")
    result = await compose_claimed_text_occurrence(service, agent, claim, sender)
    assert observed == [("A", 42)]
    assert [target for target, _ in sent] == ["A", "B"]
    assert len(result.deliveries) == 2
    request = build_claimed_fresh_request(claim)
    assert request.route.thread_id == "42"
    assert request.scheduled_link.visible_session_key == "telegram:A:topic:42"


@pytest.mark.parametrize(("field", "error_type", "message"), [
    ("claim", TypeError, "exact occurrence claim"),
    ("sender", TypeError, "exact receipt-capable channel sender"),
    ("visible_sessions", TypeError, "SessionManager"),
    ("callback_authority", TypeError, "CronCallbackStore"),
    ("delivery_owner", CronCompositionPreflightError, "non-empty"),
])
async def test_composition_retains_defensive_injected_value_rejection_before_effect(
    tmp_path, field, error_type, message,
):
    service, claim = _claimed(tmp_path)
    agent = AsyncMock()
    calls = []
    sender = CronChannelSender(lambda _name: ReceiptChannel(calls))
    values = {"claim": claim, "sender": sender, "visible_sessions": None,
              "callback_authority": None, "delivery_owner": "test-owner"}
    values[field] = object()

    with pytest.raises(error_type, match=message):
        await compose_claimed_text_occurrence(service, agent, **values)

    agent.execute_silent_cron_turn.assert_not_awaited()
    assert calls == [] and service.list_deliveries() == []
    with service.occurrence_store.acquire_execution_fence(claim) as fence:
        service.occurrence_store.begin_execution(claim, fence)


async def test_preflight_rejects_all_destinations_before_begin_or_effect(tmp_path: Path) -> None:
    service, claim = _claimed(tmp_path, ("telegram", "discord"))
    agent = AsyncMock()
    calls = []
    sender = CronChannelSender(lambda _name: ReceiptChannel(calls))
    callback = AsyncMock()
    callbacks = CronCallbackStore(tmp_path / "callbacks", callback)
    with pytest.raises(CronCompositionPreflightError, match="Telegram"):
        await compose_claimed_text_occurrence(
            service, agent, claim, sender, callback_authority=callbacks,
        )
    agent.execute_silent_cron_turn.assert_not_awaited()
    callback.assert_not_awaited()
    assert calls == []
    assert service.occurrence_store is not None
    with service.occurrence_store.acquire_execution_fence(claim) as fence:
        service.occurrence_store.begin_execution(claim, fence)


async def test_one_turn_callback_failure_and_frozen_receipt_delivery_order(tmp_path: Path) -> None:
    service, claim = _claimed(tmp_path)
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(
        run_id=claim.run_id, content="finished",
    )
    calls = []
    sender = CronChannelSender(lambda name: ReceiptChannel(calls) if name == "telegram" else None)
    callback = AsyncMock(side_effect=RuntimeError("callback failed"))
    callbacks = CronCallbackStore(tmp_path / "callbacks", callback)
    result = await compose_claimed_text_occurrence(
        service, agent, claim, sender, callback_authority=callbacks,
    )
    assert result.callback is not None and result.callback.state == "failed"
    assert calls == [("A", ("finished",)), ("B", ("finished",))]
    assert result.outbox is not None
    assert service.delivery_status(result.outbox.outbox_id) == "delivered"
    agent.execute_silent_cron_turn.assert_awaited_once()
    with pytest.raises(OccurrenceExecutionStartedError):
        await compose_claimed_text_occurrence(service, agent, claim, sender)
    agent.execute_silent_cron_turn.assert_awaited_once()


@pytest.mark.parametrize("status", [RunStatus.APPROVAL_REQUIRED, RunStatus.POLICY_BLOCKED])
async def test_unattended_policy_terminal_is_skipped_without_outbox_or_send(
    tmp_path: Path, status: RunStatus,
) -> None:
    service, claim = _claimed(tmp_path / status.value)
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(
        run_id=claim.run_id, status=status, content="must not persist",
    )
    calls = []
    result = await compose_claimed_text_occurrence(
        service, agent, claim, CronChannelSender(lambda _name: ReceiptChannel(calls)),
    )
    assert result.completion.status == "skipped"
    assert result.completion.completion["run_status"] == status.value
    assert "must not persist" not in str(result.completion.completion)
    assert result.outbox is None and result.deliveries == () and calls == []
    assert service.list_deliveries() == []


async def test_dynamic_unsupported_result_is_not_completed_sent_or_replayed(tmp_path: Path) -> None:
    service, claim = _claimed(tmp_path)
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(
        run_id=claim.run_id, status=RunStatus.ERROR, error="dynamic failure",
    )
    calls = []
    sender = CronChannelSender(lambda _name: ReceiptChannel(calls))
    with pytest.raises(ValueError, match="completed"):
        await compose_claimed_text_occurrence(service, agent, claim, sender)
    assert service.occurrence_store is not None
    assert service.occurrence_store.get(claim.occurrence_id).state == "claimed"
    assert calls == []
    with pytest.raises(OccurrenceExecutionStartedError):
        await compose_claimed_text_occurrence(service, agent, claim, sender)
    agent.execute_silent_cron_turn.assert_awaited_once()


async def test_ambiguous_receipt_is_reconciliation_required_and_never_resent(tmp_path: Path) -> None:
    service, claim = _claimed(tmp_path, ("telegram",))
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(run_id=claim.run_id, content="done")
    calls = []
    sender = CronChannelSender(lambda _name: ReceiptChannel(calls, malformed_target=True))
    result = await compose_claimed_text_occurrence(service, agent, claim, sender)
    assert result.outbox is not None
    assert result.deliveries == (UnknownOutcome("exception:ValueError"),)
    assert service.delivery_status(result.outbox.outbox_id) == "reconciliation-required"
    with pytest.raises(OccurrenceExecutionStartedError):
        await compose_claimed_text_occurrence(service, agent, claim, sender)
    assert calls == [("A", ("done",))]
