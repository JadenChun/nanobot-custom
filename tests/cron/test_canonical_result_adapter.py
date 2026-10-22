"""Dormant canonical turn-result to durable cron completion integration."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.agent.turn import RunStatus, TurnResult
from nanobot.bus.events import OutboundMessage
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.canonical_result import (
    UnsupportedCronResultError,
    complete_text_cron_result,
)
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.definition_store import CronDefinitionStore
from nanobot.cron.delivery_runtime import ConfirmedDelivered, CronDeliveryRuntime
from nanobot.cron.delivery_store import CronDeliveryOutboxStore
from nanobot.cron.occurrence_store import CronOccurrenceStore
from nanobot.cron.service import CronService
from nanobot.cron.types import CronDestination, CronSchedule


def _claimed_service(tmp_path: Path, *, deliver: bool = True):
    root = tmp_path / "cron"
    namespace = "cron:canonical-result-tests"
    definitions = CronDefinitionStore(CronActionStore(root, namespace=namespace))
    occurrences = CronOccurrenceStore(root / "occurrences", namespace=namespace)
    projector = CronCompletionProjector(occurrences, root / "runs")
    outbox = CronDeliveryOutboxStore(root / "outbox", namespace=namespace)
    service = CronService(
        root / "jobs.json", definition_store=definitions,
        occurrence_store=occurrences, completion_projector=projector,
        delivery_runtime=CronDeliveryRuntime(outbox, occurrence_store=occurrences),
    )
    job = service.add_job(
        "daily", CronSchedule(kind="every", every_ms=60_000), "report",
        deliver=deliver, channel="telegram" if deliver else None,
        to="primary" if deliver else None,
        additional_destinations=[CronDestination("discord", "secondary")] if deliver else None,
    )
    claim = service.claim_scheduled_occurrence(
        job.id, 1_700_000_000_000, owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    return service, claim


def _settle(service: CronService, claim, result: TurnResult):
    async def execute():
        assert service.occurrence_store is not None
        with service.occurrence_store.acquire_execution_fence(claim) as fence:
            service.occurrence_store.begin_execution(claim, fence)
            return complete_text_cron_result(service, claim, result, fence)

    return asyncio.run(execute())


def test_canonical_text_result_prepares_exact_durable_fanout_without_sending(tmp_path: Path) -> None:
    service, claim = _claimed_service(tmp_path)
    projection, outbox = _settle(
        service, claim, TurnResult(run_id=claim.run_id, content="finished report"),
    )

    assert projection.run_id == claim.run_id
    assert projection.completion["result"] == "finished report"
    assert projection.completion["delivery"] == {
        "version": 1, "deliver": True,
        "messages": [{"content": "finished report", "media": []}],
    }
    assert [(item.channel, item.to, item.status) for item in outbox.destinations] == [
        ("telegram", "primary", "pending"), ("discord", "secondary", "pending"),
    ]
    assert service.occurrence_store.get_completed_durable(claim.occurrence_id).completion == projection.completion
    assert service.prepare_completed_delivery(claim.occurrence_id) == outbox


def test_no_delivery_is_durable_no_work(tmp_path: Path) -> None:
    service, claim = _claimed_service(tmp_path, deliver=False)
    projection, outbox = _settle(
        service, claim, TurnResult(run_id=claim.run_id, content="private result"),
    )
    assert projection.completion["delivery"]["deliver"] is False
    assert outbox.destinations == ()
    assert service.delivery_status(outbox.outbox_id) == "no_work"


@pytest.mark.parametrize("result", [
    TurnResult(run_id="wrong", content="report"),
    TurnResult(content="report"),
    TurnResult(run_id="placeholder", status=RunStatus.APPROVAL_REQUIRED, content="report"),
    TurnResult(run_id="placeholder", content=""),
    TurnResult(run_id="placeholder", content="x" * (256 * 1024 + 1)),
    TurnResult(run_id="placeholder", content="report", sent_messages=(
        OutboundMessage("telegram", "primary", "already sent"),
    )),
])
def test_unsupported_or_unproven_result_leaves_occurrence_claimed(
    tmp_path: Path, result: TurnResult,
) -> None:
    service, claim = _claimed_service(tmp_path)
    if result.run_id == "placeholder":
        result = replace(result, run_id=claim.run_id)
    with pytest.raises(UnsupportedCronResultError):
        _settle(service, claim, result)
    assert service.occurrence_store.get(claim.occurrence_id).state == "claimed"
    assert service.list_deliveries() == []


def test_empty_canonical_ledger_does_not_consult_stale_tool_sends(tmp_path: Path) -> None:
    service, claim = _claimed_service(tmp_path)
    result = TurnResult(run_id=claim.run_id, content="report", sent_messages=())
    _, outbox = _settle(service, claim, result)
    assert outbox.messages == ({"content": "report", "media": []},)


def test_outbox_preparation_failure_recovers_without_recomputing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, claim = _claimed_service(tmp_path)
    original = service.prepare_completed_delivery
    monkeypatch.setattr(service, "prepare_completed_delivery", lambda _id: (_ for _ in ()).throw(
        OSError("outbox temporarily unavailable"),
    ))
    with pytest.raises(OSError, match="outbox temporarily unavailable"):
        _settle(
            service, claim, TurnResult(run_id=claim.run_id, content="computed once"),
        )
    persisted = service.occurrence_store.get_completed_durable(claim.occurrence_id)
    assert persisted.completion is not None
    assert persisted.completion["result"] == "computed once"
    monkeypatch.setattr(service, "prepare_completed_delivery", original)
    assert service.project_occurrence_completion(claim.occurrence_id).run_id == claim.run_id
    outbox = service.prepare_completed_delivery(claim.occurrence_id)
    assert outbox.messages == ({"content": "computed once", "media": []},)


@pytest.mark.asyncio
async def test_prepared_result_dispatches_only_under_explicit_fake_sender_claim(tmp_path: Path) -> None:
    service, claim = _claimed_service(tmp_path)
    assert service.occurrence_store is not None
    with service.occurrence_store.acquire_execution_fence(claim) as fence:
        service.occurrence_store.begin_execution(claim, fence)
        _, outbox = complete_text_cron_result(
            service, claim, TurnResult(run_id=claim.run_id, content="report"), fence,
        )
    requests = []

    class FakeSender:
        async def send(self, request):
            requests.append(request)
            return ConfirmedDelivered({"fake_receipt": request.destination_id})

    assert requests == []
    destination = outbox.destinations[0]
    delivery_claim = service.claim_delivery(
        outbox.outbox_id, destination.destination_id, outbox.record_revision, owner="test",
    )
    await service.dispatch_delivery(delivery_claim, sender=FakeSender())
    assert [(request.channel, request.to, request.messages[0].content) for request in requests] == [
        ("telegram", "primary", "report"),
    ]
    assert service.delivery_status(outbox.outbox_id) == "partial"
    current = service.get_delivery(outbox.outbox_id)
    second = current.destinations[1]
    second_claim = service.claim_delivery(
        current.outbox_id, second.destination_id, current.record_revision, owner="test",
    )
    await service.dispatch_delivery(second_claim, sender=FakeSender())
    assert [(request.channel, request.to) for request in requests] == [
        ("telegram", "primary"), ("discord", "secondary"),
    ]
    assert service.delivery_status(outbox.outbox_id) == "delivered"
