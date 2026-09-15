"""Focused E3 acceptance tests for the dormant cron delivery runtime."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest

from nanobot.cron.action_store import CronActionStore
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.definition_store import CronDefinitionAction, CronDefinitionStore
from nanobot.cron.delivery_runtime import (
    ConfirmedDelivered,
    ConfirmedNotSent,
    CronDeliveryReentrancyError,
    CronDeliveryRuntime,
    CronDeliveryRuntimeUnavailableError,
    CronDeliverySettlementError,
    DispatchRequest,
    InvalidSenderResultError,
    PartialBundleError,
    PartialBundleOutcome,
    PermanentNotSent,
    UnknownOutcome,
)
from nanobot.cron.delivery_store import (
    CronDeliveryOutboxStore,
    DeliveryOutboxOwnershipError,
    DeliveryOutboxPersistenceError,
)
from nanobot.cron.occurrence_store import CronOccurrenceStore
from nanobot.cron.service import CronService, _job_definition
from nanobot.cron.types import CronDestination, CronSchedule

NAMESPACE = "cron:package-e3-tests"


class FakeSender:
    def __init__(self, result: object = UnknownOutcome("test")) -> None:
        self.result = result
        self.requests: list[DispatchRequest] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.read_during_send = None

    async def send(self, request: DispatchRequest) -> object:
        self.requests.append(request)
        self.started.set()
        if self.read_during_send is not None:
            self.read_during_send(request)
        await self.release.wait()
        return self.result


class CancellationSuppressingSender:
    def __init__(self) -> None:
        self.requests: list[DispatchRequest] = []
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def send(self, request: DispatchRequest) -> object:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
            return ConfirmedDelivered({"id": "late"})
        raise AssertionError("sender wait unexpectedly completed")


def _service(
    tmp_path: Path,
    *,
    destinations: bool = False,
    inject_runtime: bool = True,
) -> tuple[CronService, CronDeliveryRuntime | None]:
    root = tmp_path / "cron"
    definition_store = CronDefinitionStore(CronActionStore(root, namespace=NAMESPACE))
    occurrence_store = CronOccurrenceStore(root / "occurrences", namespace=NAMESPACE)
    completion_projector = CronCompletionProjector(occurrence_store, root / "runs")
    outbox = CronDeliveryOutboxStore(root / "outbox", namespace=NAMESPACE)
    runtime = CronDeliveryRuntime(outbox, occurrence_store=occurrence_store)
    service = CronService(
        root / "jobs.json",
        definition_store=definition_store,
        occurrence_store=occurrence_store,
        completion_projector=completion_projector,
        delivery_runtime=runtime if inject_runtime else None,
    )
    return service, runtime if inject_runtime else None


def _prepared(
    tmp_path: Path,
    *,
    destinations: bool = False,
) -> tuple[CronService, CronDeliveryRuntime, object]:
    service, runtime = _service(tmp_path, destinations=destinations)
    assert runtime is not None
    additional = [{"channel": "web", "to": "room-2"}] if destinations else []
    job = service.add_job(
        "delivery",
        CronSchedule(kind="every", every_ms=60_000),
        "computed",
        deliver=True,
        channel="cli",
        to="room-1",
    )
    if destinations:
        job.payload.additional_destinations = [
            CronDestination(channel=item["channel"], to=item["to"]) for item in additional
        ]
        service._definition_action(  # type: ignore[attr-defined]
            CronDefinitionAction.update(
                _job_definition(job, job_instance_id=job.job_instance_id),
                expected_job_instance_id=job.job_instance_id,
                expected_definition_revision=job.definition_revision,
            )
        )
        job = service.get_job(job.id)
        assert job is not None
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    service.complete_occurrence(
        claim,
        {
            "status": "ok",
            "result": "computed",
            "delivery": {
                "version": 1,
                "deliver": True,
                "messages": [{"content": "hello", "media": []}],
            },
        },
    )
    record = runtime.prepare_completed(claim.occurrence_id)
    return service, runtime, record


def _first_claim(runtime: CronDeliveryRuntime, record: object):
    destination = record.destinations[0]
    return runtime.claim(record.outbox_id, destination.destination_id, record.record_revision, owner="sender")


def _claim_destination(runtime: CronDeliveryRuntime, record: object, index: int):
    destination = record.destinations[index]
    return runtime.claim(record.outbox_id, destination.destination_id, record.record_revision, owner="sender")


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (ConfirmedDelivered({"provider_id": "receipt-1"}), "delivered"),
        (ConfirmedNotSent("rate_limited"), "pending"),
        (PermanentNotSent("blocked"), "failed"),
        (UnknownOutcome("provider_unknown"), "reconciliation-required"),
    ],
)
async def test_classified_results_settle_without_reinvocation(
    tmp_path: Path,
    result: object,
    status: str,
) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = FakeSender(result)
    sender.release.set()

    returned = await runtime.dispatch(claim, sender=sender)

    assert returned == result
    assert len(sender.requests) == 1
    assert sender.requests[0].channel == "cli"
    assert sender.requests[0].to == "room-1"
    assert sender.requests[0].messages[0].content == "hello"
    assert runtime.status(record.outbox_id) == status


async def test_sender_can_read_outbox_while_awaiting_without_lock_held(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    observed: list[str] = []

    def read_during_send(request: DispatchRequest) -> None:
        observed.append(runtime.get(request.outbox_id).status)

    sender = FakeSender(ConfirmedDelivered({"id": "receipt"}))
    sender.read_during_send = read_during_send
    sender.release.set()
    await runtime.dispatch(claim, sender=sender)

    assert observed == ["reconciliation-required"]


@pytest.mark.parametrize("kind", ["timeout", "exception", "invalid", "partial"])
async def test_failure_paths_fence_destination_as_unknown_before_propagation(
    tmp_path: Path,
    kind: str,
) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)

    class FailingSender(FakeSender):
        async def send(self, request: DispatchRequest) -> object:
            self.requests.append(request)
            self.started.set()
            if kind == "timeout":
                await asyncio.Event().wait()
            if kind == "exception":
                raise RuntimeError("transport failed")
            if kind == "invalid":
                return {"outcome": "delivered"}
            raise PartialBundleError()

    sender = FailingSender()
    with pytest.raises((asyncio.TimeoutError, RuntimeError, InvalidSenderResultError, PartialBundleError)):
        await runtime.dispatch(claim, sender=sender, timeout=0.01 if kind == "timeout" else None)

    assert len(sender.requests) == 1
    current = runtime.get(record.outbox_id)
    assert current.reconciliation_required
    assert current.destinations[0].attempts[-1].status == "unknown"


async def test_cancellation_settles_unknown_before_reraising(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = FakeSender()
    task = asyncio.create_task(runtime.dispatch(claim, sender=sender))
    await sender.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(sender.requests) == 1
    assert runtime.get(record.outbox_id).status == "reconciliation-required"


async def test_same_claim_concurrent_dispatch_invokes_sender_once(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = FakeSender(ConfirmedNotSent("retryable"))
    first = asyncio.create_task(runtime.dispatch(claim, sender=sender))
    second = asyncio.create_task(runtime.dispatch(claim, sender=sender))
    await sender.started.wait()
    sender.release.set()

    assert await first == ConfirmedNotSent("retryable")
    assert await second == ConfirmedNotSent("retryable")
    assert len(sender.requests) == 1


async def test_same_claim_across_runtime_adapters_invokes_sender_once(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    second_runtime = CronDeliveryRuntime(
        runtime.outbox_store,
        occurrence_store=runtime.occurrence_store,
    )
    first_sender = FakeSender(ConfirmedNotSent("retryable"))
    second_sender = FakeSender(ConfirmedDelivered({"id": "unexpected"}))
    first = asyncio.create_task(runtime.dispatch(claim, sender=first_sender))
    await first_sender.started.wait()
    second = asyncio.create_task(second_runtime.dispatch(claim, sender=second_sender))
    await asyncio.sleep(0)
    first_sender.release.set()

    assert await first == ConfirmedNotSent("retryable")
    assert await second == ConfirmedNotSent("retryable")
    assert len(first_sender.requests) == 1
    assert second_sender.requests == []


async def test_repeated_cancellation_waits_for_sender_cleanup(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = CancellationSuppressingSender()
    dispatch_task = asyncio.create_task(runtime.dispatch(claim, sender=sender))
    await sender.started.wait()

    dispatch_task.cancel()
    await sender.cancelled.wait()
    dispatch_task.cancel()
    await asyncio.sleep(0)
    assert not dispatch_task.done()

    sender.release.set()
    with pytest.raises(asyncio.CancelledError):
        await dispatch_task
    assert runtime.status(record.outbox_id) == "reconciliation-required"


async def test_sender_suppressed_cancellation_cannot_confirm_delivery(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = CancellationSuppressingSender()
    dispatch_task = asyncio.create_task(runtime.dispatch(claim, sender=sender))
    await sender.started.wait()
    dispatch_task.cancel()
    await sender.cancelled.wait()
    sender.release.set()

    with pytest.raises(asyncio.CancelledError):
        await dispatch_task
    assert len(sender.requests) == 1
    assert runtime.status(record.outbox_id) == "reconciliation-required"
    assert runtime.get(record.outbox_id).destinations[0].attempts[-1].status == "unknown"


async def test_sender_suppressed_timeout_cannot_confirm_delivery(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = CancellationSuppressingSender()
    dispatch_task = asyncio.create_task(runtime.dispatch(claim, sender=sender, timeout=0.01))
    await sender.started.wait()
    await sender.cancelled.wait()
    assert not dispatch_task.done()
    sender.release.set()

    with pytest.raises(asyncio.TimeoutError):
        await dispatch_task
    assert len(sender.requests) == 1
    assert runtime.status(record.outbox_id) == "reconciliation-required"
    assert runtime.get(record.outbox_id).destinations[0].attempts[-1].status == "unknown"


async def test_mark_sending_failure_invokes_no_sender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = FakeSender(ConfirmedDelivered({"id": "unused"}))

    def fail_mark(_claim: object) -> object:
        raise DeliveryOutboxPersistenceError("sending fence failed")

    monkeypatch.setattr(runtime.outbox_store, "mark_sending", fail_mark)
    with pytest.raises(DeliveryOutboxPersistenceError):
        await runtime.dispatch(claim, sender=sender)
    assert sender.requests == []


async def test_settlement_failure_does_not_reinvoke_and_reports_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = FakeSender(ConfirmedNotSent("retryable"))
    sender.release.set()

    def fail_settle(*_args: object, **_kwargs: object) -> object:
        raise DeliveryOutboxPersistenceError("settlement disk failure")

    monkeypatch.setattr(runtime.outbox_store, "settle", fail_settle)
    with pytest.raises(CronDeliverySettlementError) as caught:
        await runtime.dispatch(claim, sender=sender)
    assert caught.value.classification == ConfirmedNotSent("retryable")
    assert len(sender.requests) == 1
    assert runtime.reconciliation_required()[0].outbox_id == record.outbox_id

    with pytest.raises(CronDeliverySettlementError):
        await runtime.dispatch(claim, sender=sender)
    assert len(sender.requests) == 1


async def test_cancellation_wins_over_settlement_failure_without_resend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = FakeSender()

    def fail_settle(*_args: object, **_kwargs: object) -> object:
        raise DeliveryOutboxPersistenceError("settlement disk failure")

    monkeypatch.setattr(runtime.outbox_store, "settle", fail_settle)
    dispatch_task = asyncio.create_task(runtime.dispatch(claim, sender=sender))
    await sender.started.wait()
    dispatch_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await dispatch_task
    assert len(sender.requests) == 1
    assert runtime.status(record.outbox_id) == "reconciliation-required"

    with pytest.raises(CronDeliverySettlementError):
        await runtime.dispatch(claim, sender=FakeSender(ConfirmedDelivered({"id": "retry"})))
    assert len(sender.requests) == 1


async def test_sender_base_exception_fences_unknown_and_propagates(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)

    class SenderFailure(BaseException):
        pass

    class BaseExceptionSender:
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def send(self, request: DispatchRequest) -> object:
            self.requests.append(request)
            raise SenderFailure("fatal transport")

    sender = BaseExceptionSender()
    with pytest.raises(SenderFailure):
        await runtime.dispatch(claim, sender=sender)
    assert len(sender.requests) == 1
    assert runtime.get(record.outbox_id).destinations[0].attempts[-1].status == "unknown"


async def test_malformed_result_subclass_fences_unknown(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)

    class MalformedDelivered(ConfirmedDelivered):
        pass

    sender = FakeSender(MalformedDelivered({"id": "receipt"}))
    sender.release.set()
    with pytest.raises(InvalidSenderResultError):
        await runtime.dispatch(claim, sender=sender)
    assert len(sender.requests) == 1
    assert runtime.get(record.outbox_id).destinations[0].attempts[-1].status == "unknown"


async def test_same_claim_sender_reentrancy_fails_without_deadlock(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)

    class ReentrantSender:
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def send(self, request: DispatchRequest) -> object:
            self.requests.append(request)
            await runtime.dispatch(claim, sender=self)
            raise AssertionError("reentrant dispatch unexpectedly returned")

    sender = ReentrantSender()
    with pytest.raises(CronDeliveryReentrancyError):
        await asyncio.wait_for(runtime.dispatch(claim, sender=sender), timeout=1)
    assert len(sender.requests) == 1
    assert runtime.get(record.outbox_id).destinations[0].attempts[-1].status == "unknown"


async def test_direct_child_same_claim_reentrancy_fails_without_deadlock(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)

    class DirectChildSender:
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def send(self, request: DispatchRequest) -> object:
            self.requests.append(request)
            child = asyncio.create_task(runtime.dispatch(claim, sender=self))
            return await child

    sender = DirectChildSender()
    with pytest.raises(CronDeliveryReentrancyError):
        await asyncio.wait_for(runtime.dispatch(claim, sender=sender), timeout=1)
    assert len(sender.requests) == 1
    assert runtime.get(record.outbox_id).destinations[0].attempts[-1].status == "unknown"


async def test_nested_child_same_claim_reentrancy_fails_without_deadlock(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)

    class NestedChildSender:
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def nested_child(self) -> object:
            child = asyncio.create_task(runtime.dispatch(claim, sender=self))
            return await child

        async def send(self, request: DispatchRequest) -> object:
            self.requests.append(request)
            helper = asyncio.create_task(self.nested_child())
            return await helper

    sender = NestedChildSender()
    with pytest.raises(CronDeliveryReentrancyError):
        await asyncio.wait_for(runtime.dispatch(claim, sender=sender), timeout=1)
    assert len(sender.requests) == 1
    assert runtime.get(record.outbox_id).destinations[0].attempts[-1].status == "unknown"


async def test_copied_store_registry_isolated_and_original_dispatch_survives(
    tmp_path: Path,
) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    original_sender = FakeSender(ConfirmedDelivered({"id": "original"}))
    original_task = asyncio.create_task(runtime.dispatch(claim, sender=original_sender))
    await original_sender.started.wait()

    copied_store = copy.copy(runtime.outbox_store)
    copied_runtime = CronDeliveryRuntime(
        copied_store,
        occurrence_store=runtime.occurrence_store,
    )
    copied_sender = FakeSender(ConfirmedDelivered({"id": "copied"}))
    copied_sender.release.set()
    assert copied_runtime._dispatch_registry is not runtime._dispatch_registry
    with pytest.raises(DeliveryOutboxOwnershipError):
        await copied_runtime.dispatch(claim, sender=copied_sender)
    assert copied_sender.requests == []

    original_sender.release.set()
    assert await original_task == ConfirmedDelivered({"id": "original"})
    assert len(original_sender.requests) == 1
    assert runtime.status(record.outbox_id) == "delivered"


@pytest.mark.parametrize(
    ("sender", "timeout"),
    [
        (object(), None),
        (FakeSender(ConfirmedDelivered({"id": "unused"})), 0),
        (FakeSender(ConfirmedDelivered({"id": "unused"})), float("nan")),
    ],
)
async def test_id_overload_preflights_sender_and_timeout_before_claim(
    tmp_path: Path,
    sender: object,
    timeout: float | None,
) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    destination = record.destinations[0]

    with pytest.raises((InvalidSenderResultError, ValueError)):
        await runtime.dispatch(
            record.outbox_id,
            destination.destination_id,
            record.record_revision,
            sender=sender,  # type: ignore[arg-type]
            timeout=timeout,
        )
    current = runtime.get(record.outbox_id)
    assert current.record_revision == record.record_revision
    assert current.destinations[0].status == "pending"


async def test_id_overload_claims_then_dispatches_valid_configuration(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    destination = record.destinations[0]
    sender = FakeSender(ConfirmedDelivered({"id": "valid"}))
    sender.release.set()

    result = await runtime.dispatch(
        record.outbox_id,
        destination.destination_id,
        record.record_revision,
        sender=sender,
    )

    assert result == ConfirmedDelivered({"id": "valid"})
    assert len(sender.requests) == 1
    assert runtime.status(record.outbox_id) == "delivered"


def test_request_and_sender_results_are_strict_and_detached() -> None:
    receipt = {"provider": {"id": "r-1"}}
    result = ConfirmedDelivered(receipt)
    receipt["provider"] = {"id": "mutated"}
    assert result.receipt["provider"] == {"id": "r-1"}
    with pytest.raises(ValueError):
        DispatchRequest(
            "a" * 64,
            "b" * 64,
            1,
            "c" * 64,
            "d" * 64,
            "cli",
            "target",
            ({"content": "text", "media": ["image"]},),
        )


async def test_partial_bundle_result_makes_whole_destination_unknown(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    sender = FakeSender(PartialBundleOutcome())
    sender.release.set()

    returned = await runtime.dispatch(claim, sender=sender)

    assert returned == UnknownOutcome("partial_bundle")
    assert runtime.status(record.outbox_id) == "reconciliation-required"


async def test_pending_destination_can_retry_without_recomputation(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path)
    claim = _first_claim(runtime, record)
    first_sender = FakeSender(ConfirmedNotSent("temporary"))
    first_sender.release.set()
    await runtime.dispatch(claim, sender=first_sender)
    pending = runtime.get(record.outbox_id)
    retry_claim = _first_claim(runtime, pending)
    second_sender = FakeSender(ConfirmedDelivered({"id": "second"}))
    second_sender.release.set()
    await runtime.dispatch(retry_claim, sender=second_sender)

    assert len(first_sender.requests) == 1
    assert len(second_sender.requests) == 1
    assert runtime.status(record.outbox_id) == "delivered"


async def test_two_destinations_retry_only_the_missing_destination(tmp_path: Path) -> None:
    _service_obj, runtime, record = _prepared(tmp_path, destinations=True)
    assert len(record.destinations) == 2
    first_claim = _claim_destination(runtime, record, 0)
    first_sender = FakeSender(ConfirmedDelivered({"id": "first"}))
    first_sender.release.set()
    await runtime.dispatch(first_claim, sender=first_sender)

    pending = runtime.get(record.outbox_id)
    second_claim = _claim_destination(runtime, pending, 1)
    second_sender = FakeSender(ConfirmedNotSent("temporary"))
    second_sender.release.set()
    await runtime.dispatch(second_claim, sender=second_sender)

    retry_record = runtime.get(record.outbox_id)
    retry_claim = _claim_destination(runtime, retry_record, 1)
    retry_sender = FakeSender(ConfirmedDelivered({"id": "second"}))
    retry_sender.release.set()
    await runtime.dispatch(retry_claim, sender=retry_sender)

    settled = runtime.get(record.outbox_id)
    assert len(first_sender.requests) == 1
    assert len(second_sender.requests) == 1
    assert len(retry_sender.requests) == 1
    assert len(settled.destinations[0].attempts) == 1
    assert len(settled.destinations[1].attempts) == 2
    assert runtime.status(record.outbox_id) == "delivered"


def test_management_service_has_no_delivery_side_effect_and_explicit_injection_works(
    tmp_path: Path,
) -> None:
    root = tmp_path / "management"
    service = CronService(root / "jobs.json")
    assert service.delivery_runtime is None
    assert not (root / "outbox").exists()
    with pytest.raises(CronDeliveryRuntimeUnavailableError):
        service.list_deliveries()

    injected_service, runtime = _service(tmp_path / "injected")
    assert runtime is not None
    assert injected_service.delivery_runtime is runtime
    assert injected_service.delivery_store is runtime.outbox_store


def test_runtime_rejects_mismatched_occurrence_authority_namespace(tmp_path: Path) -> None:
    outbox = CronDeliveryOutboxStore(tmp_path / "outbox", namespace=NAMESPACE)
    occurrence = CronOccurrenceStore(tmp_path / "occurrences", namespace="cron:other")
    with pytest.raises(ValueError, match="share one namespace"):
        CronDeliveryRuntime(outbox, occurrence_store=occurrence)


def test_service_rejects_mismatched_delivery_occurrence_authority(tmp_path: Path) -> None:
    root = tmp_path / "authority-mismatch"
    definitions = CronDefinitionStore(CronActionStore(root / "actions", namespace=NAMESPACE))
    service_occurrence = CronOccurrenceStore(root / "service-occurrences", namespace=NAMESPACE)
    runtime_occurrence = CronOccurrenceStore(root / "runtime-occurrences", namespace=NAMESPACE)
    projector = CronCompletionProjector(service_occurrence, root / "runs")
    outbox = CronDeliveryOutboxStore(root / "outbox", namespace=NAMESPACE)
    runtime = CronDeliveryRuntime(outbox, occurrence_store=runtime_occurrence)

    with pytest.raises(ValueError, match="service occurrence authority"):
        CronService(
            root / "jobs.json",
            definition_store=definitions,
            occurrence_store=service_occurrence,
            completion_projector=projector,
            delivery_runtime=runtime,
        )


def test_management_only_service_creates_no_delivery_or_occurrence_files(tmp_path: Path) -> None:
    root = tmp_path / "definition-only"
    service = CronService(root / "jobs.json")
    assert service.delivery_runtime is None
    assert service.occurrence_store is None
    assert not (root / "outbox").exists()
    assert not (root / "occurrences").exists()


def test_runtime_has_no_channel_provider_or_bus_wiring() -> None:
    source = Path(__file__).parents[2].joinpath("nanobot", "cron", "delivery_runtime.py").read_text()
    assert "nanobot.channels" not in source
    assert "nanobot.providers" not in source
    assert "nanobot.bus" not in source
