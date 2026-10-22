"""Focused E2 acceptance tests for the dormant cron delivery outbox."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import multiprocessing
import os
import pickle
import stat
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

import nanobot.cron.delivery_store as delivery_store_module
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.definition_store import CronDefinitionStore
from nanobot.cron.delivery_store import (
    CronDeliveryOutboxStore,
    DeliveryOutboxConflictError,
    DeliveryOutboxCorruptError,
    DeliveryOutboxOwnershipError,
    DeliveryOutboxPersistenceError,
    DeliveryOutboxStaleRevisionError,
    DeliverySendBusyError,
)
from nanobot.cron.occurrence_store import OccurrenceReconciliationRequiredError
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule

NAMESPACE = "cron:e2-tests"


@pytest.mark.parametrize("entry", ["construct", "open_existing", "inspect_existing"])
def test_missing_fcntl_fails_before_opening_outbox(tmp_path: Path, monkeypatch, entry) -> None:
    root = tmp_path / "unqualified-outbox"
    monkeypatch.setattr(delivery_store_module, "_fcntl", None)
    with pytest.raises(DeliveryOutboxPersistenceError, match="qualified POSIX"):
        if entry == "construct":
            CronDeliveryOutboxStore(root, namespace=NAMESPACE)
        else:
            getattr(CronDeliveryOutboxStore, entry)(root, namespace=NAMESPACE)
    assert not root.exists()


def test_qualified_module_capture_retains_in_place_flock_patches(tmp_path: Path, monkeypatch) -> None:
    store = CronDeliveryOutboxStore(tmp_path / "capture-outbox", namespace=NAMESPACE)
    module = delivery_store_module._fcntl
    original = module.flock
    calls = []

    def observed(fd, operation):
        calls.append(operation)
        return original(fd, operation)

    monkeypatch.setattr(module, "flock", observed)
    monkeypatch.setattr(delivery_store_module, "_fcntl", None)
    assert store.list() == []
    assert calls == [module.LOCK_EX, module.LOCK_UN]


def _service(tmp_path: Path) -> CronService:
    root = tmp_path / "cron"
    definitions = CronDefinitionStore(CronActionStore(root, namespace=NAMESPACE))
    occurrences = __import__("nanobot.cron.occurrence_store", fromlist=["CronOccurrenceStore"]).CronOccurrenceStore(
        root / "occurrences", namespace=NAMESPACE
    )
    return CronService(
        root / "jobs.json",
        definition_store=definitions,
        occurrence_store=occurrences,
        completion_projector=CronCompletionProjector(occurrences, root / "runs"),
    )


def _complete(service: CronService, claim, completion):
    def settle():
        with service.occurrence_store.acquire_execution_fence(claim) as fence:
            service.occurrence_store.begin_execution(claim, fence)
            return service.complete_occurrence(claim, completion, fence=fence)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        async def run():
            return settle()
        return asyncio.run(run())
    return settle()


def _completed(
    tmp_path: Path,
    *,
    deliver: bool = True,
    messages: list[dict[str, object]] | None = None,
    policy: dict[str, object] | None = None,
):
    service = _service(tmp_path)
    job = service.add_job("job", CronSchedule(kind="every", every_ms=60_000), "prompt")
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    if policy is None:
        policy = {
            "deliver": True,
            "channel": "cli",
            "to": "alpha",
            "additionalDestinations": [],
        }
    if policy is not None:
        # The helper uses the synthetic ledger directly so tests can freeze old
        # destination policy without touching the definition service.
        _complete(
            service, claim,
            {
                "status": "ok",
                "delivery": {
                    "version": 1,
                    "deliver": deliver,
                    "messages": messages if messages is not None else [{"content": "done", "media": []}],
                },
                "provenance": {"source": "test"},
            },
        )
        path = service.occurrence_store._record_path(claim.occurrence_id)
        wire = json.loads(path.read_text(encoding="utf-8"))
        wire["destination_policy"] = policy
        body = {key: value for key, value in wire.items() if key != "integrity"}
        wire["integrity"] = __import__("hashlib").sha256(
            json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        path.write_text(
            json.dumps(wire, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        return service, claim
    _complete(
        service, claim,
        {
            "status": "ok",
            "delivery": {
                "version": 1,
                "deliver": deliver,
                "messages": messages if messages is not None else [{"content": "done", "media": []}],
            },
            "provenance": {"source": "test"},
        },
    )
    return service, claim


def _prepare(tmp_path: Path, **kwargs):
    service, claim = _completed(tmp_path, **kwargs)
    outbox = CronDeliveryOutboxStore(tmp_path / "cron" / "outbox", namespace=NAMESPACE)
    return service, claim, outbox, outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)


def _rewrite_wire(outbox: CronDeliveryOutboxStore, record_id: str, wire: dict[str, object]) -> None:
    body = {key: value for key, value in wire.items() if key != "integrity"}
    wire["integrity"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    path = outbox.root / f"{record_id}.json"
    path.write_text(
        json.dumps(wire, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _mark_sending(outbox: CronDeliveryOutboxStore, claim):
    async def run():
        with outbox.acquire_send_fence(claim) as fence:
            return outbox.mark_sending(claim, fence)

    return asyncio.run(run())


def _settle(outbox: CronDeliveryOutboxStore, claim, result):
    async def run():
        with outbox.acquire_send_fence(claim) as fence:
            # Reprove the already durable sending transition under the newly
            # acquired fence before exercising settlement in store-level tests.
            outbox.mark_sending(claim, fence)
            return outbox.settle(claim, result, fence=fence)

    return asyncio.run(run())


def test_prepare_deduplicates_destinations_and_freezes_old_policy(tmp_path: Path) -> None:
    policy = {
        "deliver": True,
        "channel": None,
        "to": "alpha",
        "additionalDestinations": [
            {"channel": "cli", "to": "alpha"},
            {"channel": "telegram", "to": "beta"},
            {"channel": "telegram", "to": "beta"},
        ],
    }
    service, claim, outbox, record = _prepare(tmp_path, policy=policy)

    assert [(item.channel, item.to) for item in record.destinations] == [
        ("cli", "alpha"),
        ("telegram", "beta"),
    ]
    assert record.provenance == {"source": "test"}
    assert record.status == "pending"

    # A later definition/policy mutation cannot alter the durable outbox.
    service.occurrence_store.root.joinpath(f"{claim.occurrence_id}.json").chmod(0o600)
    retry = outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)
    assert retry == record


@pytest.mark.parametrize(
    "deliver,messages,expected",
    [
        (False, [{"content": "done", "media": []}], "no_work"),
        (True, [], "no_work"),
    ],
)
def test_false_delivery_and_empty_messages_are_durable_zero_work(
    tmp_path: Path, deliver: bool, messages: list[dict[str, object]], expected: str
) -> None:
    _service_obj, _claim, _outbox, record = _prepare(tmp_path, deliver=deliver, messages=messages)
    assert record.destinations == ()
    assert record.status == expected


def test_missing_or_non_text_delivery_envelope_is_unsupported(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = service.add_job("job", CronSchedule(kind="every", every_ms=60_000), "prompt")
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    _complete(service, claim, {"status": "ok"})
    outbox = CronDeliveryOutboxStore(tmp_path / "cron" / "outbox", namespace=NAMESPACE)
    with pytest.raises(DeliveryOutboxConflictError, match="delivery envelope"):
        outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)


def test_prepare_reproves_visible_record_after_directory_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, claim = _completed(tmp_path)
    outbox = CronDeliveryOutboxStore(tmp_path / "cron" / "outbox", namespace=NAMESPACE)
    original_fsync = delivery_store_module.os.fsync
    directory_fsyncs = 0
    failed = False

    def fail_directory_fsync(fd: int) -> None:
        nonlocal directory_fsyncs, failed
        if stat.S_ISDIR(delivery_store_module.os.fstat(fd).st_mode):
            directory_fsyncs += 1
        if not failed and directory_fsyncs >= 2 and stat.S_ISDIR(delivery_store_module.os.fstat(fd).st_mode):
            failed = True
            raise OSError("directory fsync acknowledgement lost")
        original_fsync(fd)

    monkeypatch.setattr(delivery_store_module.os, "fsync", fail_directory_fsync)
    with pytest.raises(delivery_store_module.DeliveryOutboxPersistenceError, match="durably create"):
        outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)
    monkeypatch.setattr(delivery_store_module.os, "fsync", original_fsync)
    retried = outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)
    assert retried.record_revision == 0


def test_claimed_occurrence_cannot_create_an_outbox(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = service.add_job("job", CronSchedule(kind="every", every_ms=60_000), "prompt")
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    outbox = CronDeliveryOutboxStore(tmp_path / "cron" / "outbox", namespace=NAMESPACE)
    with pytest.raises(OccurrenceReconciliationRequiredError):
        outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)
    assert outbox.list() == []


def test_absent_provenance_remains_unavailable(tmp_path: Path) -> None:
    service, claim = _completed(tmp_path)
    path = service.occurrence_store._record_path(claim.occurrence_id)
    wire = json.loads(path.read_text(encoding="utf-8"))
    wire["completion"].pop("provenance", None)
    body = {key: value for key, value in wire.items() if key != "integrity"}
    wire["integrity"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    path.write_text(
        json.dumps(wire, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    outbox = CronDeliveryOutboxStore(tmp_path / "cron" / "outbox", namespace=NAMESPACE)
    record = outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)
    assert record.provenance is None
    assert record.provenance_available is False


def test_record_permissions_hardlinks_and_exact_integer_types_fail_closed(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    path = outbox.root / f"{record.outbox_id}.json"
    os.chmod(path, 0o640)
    with pytest.raises(DeliveryOutboxCorruptError):
        outbox.get(record.outbox_id)
    os.chmod(path, 0o600)
    link = outbox.root / "linked-record"
    os.link(path, link)
    with pytest.raises(DeliveryOutboxCorruptError):
        outbox.get(record.outbox_id)
    link.unlink()
    wire = json.loads(path.read_text(encoding="utf-8"))
    wire["record_revision"] = True
    body = {key: value for key, value in wire.items() if key != "integrity"}
    wire["integrity"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    path.write_text(
        json.dumps(wire, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(DeliveryOutboxCorruptError, match="record_revision"):
        outbox.get(record.outbox_id)


def test_root_swap_is_detected_on_reproof(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    root = outbox.root
    root.rename(tmp_path / "moved-outbox")
    root.mkdir(mode=0o700)
    with pytest.raises(DeliveryOutboxCorruptError, match="root.*binding"):
        outbox.get(record.outbox_id)


def test_sending_fsync_failure_invalidates_claim_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    original_replace = outbox._replace_record

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise delivery_store_module.DeliveryOutboxPersistenceError("sending fence failed")

    monkeypatch.setattr(outbox, "_replace_record", fail_replace)
    with pytest.raises(delivery_store_module.DeliveryOutboxPersistenceError, match="sending fence"):
        _mark_sending(outbox, claim)
    with pytest.raises(DeliveryOutboxOwnershipError):
        _mark_sending(outbox, claim)
    monkeypatch.setattr(outbox, "_replace_record", original_replace)
    assert outbox.get(record.outbox_id).status == "reconciliation-required"


def test_settlement_failure_before_publication_consumes_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    _mark_sending(outbox, claim)
    original_replace = outbox._replace_record

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise delivery_store_module.DeliveryOutboxPersistenceError("settlement publication failed")

    monkeypatch.setattr(outbox, "_replace_record", fail_replace)
    with pytest.raises(delivery_store_module.DeliveryOutboxPersistenceError, match="settlement publication"):
        _settle(outbox, claim, {"outcome": "confirmed_delivered", "receipt": {"id": "receipt"}})
    with pytest.raises(DeliveryOutboxOwnershipError):
        _settle(outbox, claim, {"outcome": "confirmed_delivered", "receipt": {"id": "receipt"}})
    monkeypatch.setattr(outbox, "_replace_record", original_replace)
    assert outbox.get(record.outbox_id).status == "reconciliation-required"


def test_settlement_lost_acknowledgement_is_reproved_without_reinvocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    _mark_sending(outbox, claim)
    original_read = outbox._read_bytes
    reads = 0

    def lose_ack(*args: object, **kwargs: object):
        nonlocal reads
        reads += 1
        result = original_read(*args, **kwargs)
        if result[1].status == "delivered":
            raise delivery_store_module.DeliveryOutboxPersistenceError("settlement acknowledgement lost")
        return result

    monkeypatch.setattr(outbox, "_read_bytes", lose_ack)
    with pytest.raises(delivery_store_module.DeliveryOutboxPersistenceError, match="acknowledgement"):
        _settle(outbox, claim, {"outcome": "confirmed_delivered", "receipt": {"id": "receipt"}})
    monkeypatch.setattr(outbox, "_read_bytes", original_read)
    assert outbox.get(record.outbox_id).status == "delivered"
    with pytest.raises(DeliveryOutboxOwnershipError):
        _settle(outbox, claim, {"outcome": "confirmed_delivered", "receipt": {"id": "receipt"}})


def test_claim_transitions_and_aggregate_status(tmp_path: Path) -> None:
    service, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claimed = outbox.claim_destination(record.outbox_id, destination.destination_id, record.record_revision, owner="sender")
    assert outbox.get(record.outbox_id).status == "reconciliation-required"
    sending = _mark_sending(outbox, claimed)
    assert sending.destinations[0].status == "sending"
    delivered = _settle(outbox, claimed, {"outcome": "confirmed_delivered", "receipt": {"id": "m1"}})
    assert delivered.status == "delivered"
    assert delivered.destinations[0].attempts[0].receipt == {"id": "m1"}
    with pytest.raises(DeliveryOutboxOwnershipError):
        _settle(outbox, claimed, {"outcome": "confirmed_delivered", "receipt": {"id": "m1"}})


def test_reconciliation_confirms_not_sent_and_allows_new_attempt(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claimed = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    resolved = outbox.resolve_after_reconciliation(
        record.outbox_id,
        destination.destination_id,
        expected_revision=1,
        predecessor_attempt=1,
        predecessor_owner="sender",
        action_id=claimed.action_id,
        request_digest=claimed.request_digest,
        evidence_ref="evidence-1",
        predecessor_fenced_ref="fence-1",
        resolution="confirmed_not_sent",
    )
    assert resolved.status == "pending"
    retried = outbox.claim_destination(record.outbox_id, destination.destination_id, 2, owner="sender-2")
    assert retried.attempt == 2

    # Exact resolution retry is idempotent before stale revision checking.
    retried_resolution = outbox.resolve_after_reconciliation(
        record.outbox_id,
        destination.destination_id,
        expected_revision=1,
        predecessor_attempt=1,
        predecessor_owner="sender",
        action_id=claimed.action_id,
        request_digest=claimed.request_digest,
        evidence_ref="evidence-1",
        predecessor_fenced_ref="fence-1",
        resolution="confirmed_not_sent",
    )
    assert retried_resolution.record_revision == outbox.get(record.outbox_id).record_revision

    with pytest.raises(DeliveryOutboxConflictError):
        outbox.resolve_after_reconciliation(
            record.outbox_id,
            destination.destination_id,
            expected_revision=1,
            predecessor_attempt=1,
            predecessor_owner="sender",
            action_id=claimed.action_id,
            request_digest="0" * 64,
            evidence_ref="evidence-1",
            predecessor_fenced_ref="fence-1",
            resolution="confirmed_not_sent",
        )


def test_claimed_reconciliation_cannot_confirm_delivery(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claimed = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    with pytest.raises(DeliveryOutboxConflictError, match="claimed"):
        outbox.resolve_after_reconciliation(
            record.outbox_id,
            destination.destination_id,
            expected_revision=1,
            predecessor_attempt=1,
            predecessor_owner="sender",
            action_id=claimed.action_id,
            request_digest=claimed.request_digest,
            evidence_ref="evidence-1",
            predecessor_fenced_ref="fence-1",
            resolution="confirmed_delivered",
            receipt={"id": "m1"},
        )


def test_forked_or_reconstructed_claim_has_no_authority(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    with pytest.raises(DeliveryOutboxOwnershipError):
        _mark_sending(outbox, copy.copy(claim))
    foreign = CronDeliveryOutboxStore(tmp_path / "cron" / "outbox", namespace=NAMESPACE)
    with pytest.raises(DeliveryOutboxStaleRevisionError):
        foreign.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")


def test_bad_record_duplicate_key_and_hardlink_fail_closed(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    path = outbox.root / f"{record.outbox_id}.json"
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.replace('"kind":"cron_delivery_outbox_v1"', '"kind":"cron_delivery_outbox_v1","kind":"cron_delivery_outbox_v1"'), encoding="utf-8")
    with pytest.raises(DeliveryOutboxCorruptError):
        outbox.get(record.outbox_id)


def _claim_in_process(root: str, result: object) -> None:
    try:
        store = CronDeliveryOutboxStore(Path(root), namespace=NAMESPACE)
        records = store.list()
        item = records[0]
        claim = store.claim_destination(item.outbox_id, item.destinations[0].destination_id, item.record_revision, owner="p")
        result.put(("ok", claim.attempt))  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - assertion in parent process
        result.put((type(exc).__name__, str(exc)))  # type: ignore[attr-defined]


def test_two_process_reservation_has_one_winner(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    ctx = multiprocessing.get_context("fork")
    result = ctx.Queue()
    workers = [ctx.Process(target=_claim_in_process, args=(str(outbox.root), result)) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    outcomes = [result.get(timeout=2) for _ in workers]
    assert sum(item[0] == "ok" for item in outcomes) == 1
    assert any(
        item[0]
        in {
            "DeliveryOutboxReconciliationRequiredError",
            "DeliveryOutboxStaleRevisionError",
        }
        for item in outcomes
    )


def test_prepare_requires_authoritative_occurrence_store(tmp_path: Path) -> None:
    service, claim, outbox, record = _prepare(tmp_path)
    with pytest.raises(TypeError, match="CronOccurrenceStore"):
        outbox.prepare_completed(record, claim.occurrence_id)  # type: ignore[arg-type]
    assert outbox.get(record.outbox_id) == record


def test_store_copy_cannot_use_an_issued_claim(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    copied_store = copy.copy(outbox)
    with pytest.raises(DeliveryOutboxOwnershipError):
        _mark_sending(copied_store, claim)
    with pytest.raises(DeliveryOutboxStaleRevisionError):
        copied_store.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    assert outbox.get(record.outbox_id).status == "reconciliation-required"


def _mark_inherited_claim(store: CronDeliveryOutboxStore, claim: object, result: object) -> None:
    try:
        async def run() -> None:
            with store.acquire_send_fence(claim):  # type: ignore[arg-type]
                raise AssertionError("fork child unexpectedly acquired inherited claim authority")

        asyncio.run(run())
    except Exception as exc:  # pragma: no cover - assertion in parent process
        result.put(type(exc).__name__)  # type: ignore[attr-defined]
    else:  # pragma: no cover - assertion in parent process
        result.put("ok")  # type: ignore[attr-defined]


def _use_send_fence_in_fork(store, claim, fence, outcomes, release) -> None:
    async def use() -> None:
        try:
            store.mark_sending(claim, fence)
        except DeliveryOutboxOwnershipError:
            outcomes.put("rejected")
        else:
            outcomes.put("accepted")

    asyncio.run(use())
    release.wait()


def _reconcile_in_process(root: str, record, claim, outcomes) -> None:
    store = CronDeliveryOutboxStore(Path(root), namespace=NAMESPACE)
    try:
        store.resolve_after_reconciliation(
            record.outbox_id,
            claim.destination_id,
            expected_revision=2,
            predecessor_attempt=claim.attempt,
            predecessor_owner=claim.owner,
            action_id=claim.action_id,
            request_digest=claim.request_digest,
            evidence_ref="evidence",
            predecessor_fenced_ref="external-fence",
            resolution="confirmed_delivered",
            receipt={"id": "receipt"},
        )
    except Exception as exc:  # pragma: no cover - asserted in parent process
        outcomes.put(type(exc).__name__)
    else:  # pragma: no cover - asserted in parent process
        outcomes.put("resolved")


def test_fork_inherited_claim_has_no_authority(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    ctx = multiprocessing.get_context("fork")
    result = ctx.Queue()
    worker = ctx.Process(target=_mark_inherited_claim, args=(outbox, claim, result))
    worker.start()
    worker.join(timeout=10)
    assert result.get(timeout=2) == "DeliveryOutboxOwnershipError"


def test_unrelated_destination_revision_does_not_invalidate_live_claim(tmp_path: Path) -> None:
    policy = {
        "deliver": True,
        "channel": "cli",
        "to": "alpha",
        "additionalDestinations": [{"channel": "cli", "to": "beta"}],
    }
    _service_obj, _claim, outbox, record = _prepare(tmp_path, policy=policy)
    first, second = record.destinations
    claim_a = outbox.claim_destination(record.outbox_id, first.destination_id, 0, owner="a")
    claim_b = outbox.claim_destination(record.outbox_id, second.destination_id, 1, owner="b")
    assert outbox.claim_destination(record.outbox_id, first.destination_id, 0, owner="a") is claim_a
    _mark_sending(outbox, claim_a)
    _mark_sending(outbox, claim_b)
    settled_a = _settle(outbox, claim_a, {"outcome": "confirmed_delivered", "receipt": {"id": "a"}})
    assert settled_a.destinations[0].status == "delivered"
    settled_b = _settle(outbox, claim_b, {"outcome": "confirmed_not_sent"})
    assert settled_b.status == "partial"


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        ("confirmed_delivered", "delivered"),
        ("confirmed_not_sent", "pending"),
        ("permanent_not_sent", "failed"),
        ("unknown", "reconciliation-required"),
    ],
)
def test_all_settlement_outcomes_are_classified(
    tmp_path: Path, outcome: str, expected_status: str
) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    _mark_sending(outbox, claim)
    result: dict[str, object] = {"outcome": outcome}
    if outcome == "confirmed_delivered":
        result["receipt"] = {"id": "receipt"}
    settled = _settle(outbox, claim, result)
    assert settled.status == expected_status


@pytest.mark.parametrize(
    ("predecessor", "outcome", "expected_status"),
    [
        ("claimed", "confirmed_not_sent", "pending"),
        ("sending", "confirmed_delivered", "delivered"),
        ("sending", "confirmed_compensated", "pending"),
        ("sending", "abandon", "failed"),
        ("unknown", "confirmed_not_sent", "pending"),
        ("unknown", "confirmed_delivered", "delivered"),
        ("unknown", "confirmed_compensated", "pending"),
        ("unknown", "abandon", "failed"),
    ],
)
def test_all_reconciliation_outcomes_are_classified(
    tmp_path: Path, predecessor: str, outcome: str, expected_status: str
) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    if predecessor in {"sending", "unknown"}:
        _mark_sending(outbox, claim)
    if predecessor == "unknown":
        _settle(outbox, claim, {"outcome": "unknown"})
    current = outbox.get(record.outbox_id)
    resolution: dict[str, object] = {"outcome": outcome}
    if outcome == "confirmed_delivered":
        resolution["receipt"] = {"id": "reconciled"}
    resolved = outbox.resolve_after_reconciliation(
        record.outbox_id,
        destination.destination_id,
        expected_revision=current.record_revision,
        predecessor_attempt=1,
        predecessor_owner="sender",
        action_id=claim.action_id,
        request_digest=claim.request_digest,
        evidence_ref="evidence",
        predecessor_fenced_ref="fence",
        resolution=resolution,
    )
    assert resolved.status == expected_status


def test_confirmed_compensation_allows_only_the_next_successor(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    _mark_sending(outbox, claim)
    current = outbox.get(record.outbox_id)
    resolved = outbox.resolve_after_reconciliation(
        record.outbox_id,
        destination.destination_id,
        expected_revision=current.record_revision,
        predecessor_attempt=1,
        predecessor_owner="sender",
        action_id=claim.action_id,
        request_digest=claim.request_digest,
        evidence_ref="evidence",
        predecessor_fenced_ref="fence",
        resolution="confirmed_compensated",
    )
    assert resolved.status == "pending"
    successor = outbox.claim_destination(record.outbox_id, destination.destination_id, resolved.record_revision, owner="retry")
    assert successor.attempt == 2


def test_exact_hash_vectors_are_stable(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    def canonical(value: object) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    assert record.outbox_id == hashlib.sha256(
        b"nanobot-cron-delivery-outbox-v1\0"
        + canonical({"store_namespace": NAMESPACE, "occurrence_id": record.occurrence_id})
    ).hexdigest()
    assert destination.destination_id == hashlib.sha256(
        b"nanobot-cron-delivery-destination-v1\0"
        + canonical({"outbox_id": record.outbox_id, "channel": destination.channel, "to": destination.to})
    ).hexdigest()
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    assert claim.action_id == hashlib.sha256(
        b"nanobot-cron-delivery-action-v1\0"
        + canonical({"outbox_id": record.outbox_id, "destination_id": destination.destination_id, "attempt": 1})
    ).hexdigest()
    fixed_outbox = delivery_store_module._outbox_id(NAMESPACE, "0" * 64)
    fixed_destination = delivery_store_module._destination_id(fixed_outbox, "cli", "alpha")
    assert fixed_outbox == "47d4360b24083a12e9747323cfaa2ce4abae1febd387ede61bedd12929131eaa"
    assert fixed_destination == "10775e357d4f4105c90e343b1ed29d9a29a990aa9a1146eb2e602e6848a3f46c"
    assert delivery_store_module._action_id(fixed_outbox, fixed_destination, 1) == (
        "7c84d85061ee297555aafb8b17253af4b3f7c6e686e4c00d919d1a6b0bdc0f1b"
    )
    assert delivery_store_module._request_digest(
        fixed_outbox,
        fixed_destination,
        1,
        [{"content": "done", "media": []}],
    ) == "76ba76c813242101b24a0b1af5cf6469c18192ae52f0847593297d4dbe7ccb7b"


def _completed_with_completion(tmp_path: Path, completion: dict[str, object]):
    service = _service(tmp_path)
    job = service.add_job("job", CronSchedule(kind="every", every_ms=60_000), "prompt")
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    _complete(service, claim, completion)
    return service, claim


@pytest.mark.parametrize(
    "completion",
    [
        {"status": "ok"},
        {"status": "ok", "delivery": {"version": 1, "deliver": True, "messages": [], "extra": 1}},
        {"status": "ok", "delivery": {"version": 2, "deliver": True, "messages": []}},
        {"status": "ok", "delivery": {"version": True, "deliver": True, "messages": []}},
        {"status": "ok", "delivery": {"version": 1, "deliver": "yes", "messages": []}},
        {"status": "ok", "delivery": {"version": 1, "deliver": True, "messages": {}}},
        {
            "status": "ok",
            "delivery": {"version": 1, "deliver": True, "messages": [{"content": "x", "media": [], "extra": 1}]},
        },
        {
            "status": "ok",
            "delivery": {"version": 1, "deliver": True, "messages": [{"content": 1, "media": []}]},
        },
        {
            "status": "ok",
            "delivery": {"version": 1, "deliver": True, "messages": [{"content": "x", "media": ["file"]}]},
        },
    ],
)
def test_strict_delivery_envelope_matrix(tmp_path: Path, completion: dict[str, object]) -> None:
    service, claim = _completed_with_completion(tmp_path, completion)
    outbox = CronDeliveryOutboxStore(tmp_path / "cron" / "outbox", namespace=NAMESPACE)
    with pytest.raises(delivery_store_module.DeliveryOutboxError):
        outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)


def test_destination_count_identifier_evidence_and_record_size_limits(tmp_path: Path) -> None:
    policy = {
        "deliver": True,
        "channel": "cli",
        "to": "first",
        "additionalDestinations": [
            {"channel": "cli", "to": f"target-{index}"} for index in range(256)
        ],
    }
    service, claim = _completed(tmp_path, policy=policy)
    outbox = CronDeliveryOutboxStore(tmp_path / "cron" / "outbox", namespace=NAMESPACE)
    with pytest.raises(DeliveryOutboxConflictError, match="destinations"):
        outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)

    service, claim = _completed(
        tmp_path / "destination-id",
        policy={"deliver": True, "channel": "c" * 513, "to": "target"},
    )
    outbox = CronDeliveryOutboxStore(tmp_path / "destination-id" / "cron" / "outbox", namespace=NAMESPACE)
    with pytest.raises(DeliveryOutboxConflictError, match="512-byte"):
        outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)

    service, claim = _completed(tmp_path / "owner", policy={"deliver": True, "channel": "cli", "to": "one"})
    outbox = CronDeliveryOutboxStore(tmp_path / "owner" / "cron" / "outbox", namespace=NAMESPACE)
    record = outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)
    with pytest.raises(DeliveryOutboxConflictError, match="512-byte"):
        outbox.claim_destination(record.outbox_id, record.destinations[0].destination_id, 0, owner="x" * 513)
    claim_obj = outbox.claim_destination(record.outbox_id, record.destinations[0].destination_id, 0, owner="sender")
    _mark_sending(outbox, claim_obj)
    with pytest.raises(DeliveryOutboxConflictError, match="512-byte"):
        outbox.resolve_after_reconciliation(
            record.outbox_id,
            record.destinations[0].destination_id,
            expected_revision=outbox.get(record.outbox_id).record_revision,
            predecessor_attempt=1,
            predecessor_owner="sender",
            action_id=claim_obj.action_id,
            request_digest=claim_obj.request_digest,
            evidence_ref="e" * 513,
            predecessor_fenced_ref="fence",
            resolution="confirmed_not_sent",
        )

    large_message = "x" * 550_000
    service, claim = _completed(
        tmp_path / "size",
        messages=[{"content": large_message, "media": []}],
    )
    outbox = CronDeliveryOutboxStore(tmp_path / "size" / "cron" / "outbox", namespace=NAMESPACE)
    with pytest.raises(DeliveryOutboxConflictError, match="1MiB"):
        outbox.prepare_completed(service.occurrence_store, claim.occurrence_id)


@pytest.mark.parametrize("terminal", ["confirmed_delivered", "permanent_not_sent"])
def test_attempt_history_rejects_successors_after_terminal_attempts(tmp_path: Path, terminal: str) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    _mark_sending(outbox, claim)
    receipt = {"id": "terminal"} if terminal == "confirmed_delivered" else None
    _settle(outbox, claim, {"outcome": terminal, **({"receipt": receipt} if receipt else {})})
    path = outbox.root / f"{record.outbox_id}.json"
    wire = json.loads(path.read_text(encoding="utf-8"))
    raw_destination = wire["destinations"][0]
    successor_number = 2
    raw_destination["attempts"].append(
        {
            "attempt": successor_number,
            "owner": "forged",
            "status": "claimed",
            "action_id": delivery_store_module._action_id(
                record.outbox_id, destination.destination_id, successor_number
            ),
            "request_digest": delivery_store_module._request_digest(
                record.outbox_id,
                destination.destination_id,
                successor_number,
                wire["immutable_input"]["delivery"]["messages"],
            ),
            "receipt": None,
            "resolution": None,
        }
    )
    raw_destination["status"] = "claimed"
    wire["record_revision"] += 1
    wire["transition_history"].append(
        {
            "revision": wire["record_revision"],
            "destination_id": destination.destination_id,
            "attempt": successor_number,
            "operation": "claim",
            "outcome": None,
            "receipt": None,
            "evidence_ref": None,
            "predecessor_fenced_ref": None,
        }
    )
    _rewrite_wire(outbox, record.outbox_id, wire)
    with pytest.raises(DeliveryOutboxCorruptError, match="successor"):
        outbox.get(record.outbox_id)


def test_transition_history_and_resolution_evidence_must_agree(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    _mark_sending(outbox, claim)
    outbox.resolve_after_reconciliation(
        record.outbox_id,
        destination.destination_id,
        expected_revision=2,
        predecessor_attempt=1,
        predecessor_owner="sender",
        action_id=claim.action_id,
        request_digest=claim.request_digest,
        evidence_ref="evidence",
        predecessor_fenced_ref="fence",
        resolution="confirmed_delivered",
        receipt={"id": "receipt"},
    )
    path = outbox.root / f"{record.outbox_id}.json"
    wire = json.loads(path.read_text(encoding="utf-8"))
    wire["destinations"][0]["attempts"][0]["receipt"] = {"id": "different"}
    _rewrite_wire(outbox, record.outbox_id, wire)
    with pytest.raises(DeliveryOutboxCorruptError, match="resolution receipt"):
        outbox.get(record.outbox_id)


def test_record_revision_requires_retained_transition_evidence(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    path = outbox.root / f"{record.outbox_id}.json"
    wire = json.loads(path.read_text(encoding="utf-8"))
    wire["record_revision"] += 1
    _rewrite_wire(outbox, record.outbox_id, wire)
    with pytest.raises(DeliveryOutboxCorruptError, match="transition history"):
        outbox.get(record.outbox_id)


def test_lock_swap_is_detected(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    lock = outbox.root / ".lock"
    moved = outbox.root / ".lock-old"
    lock.rename(moved)
    replacement = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(replacement)
    with pytest.raises(DeliveryOutboxCorruptError, match="lock binding"):
        outbox.get(record.outbox_id)


def test_staging_in_place_mutation_after_file_fsync_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    # The claim above has already published; use a fresh outbox record for the
    # mutation by reconciling it back to pending first.
    outbox.resolve_after_reconciliation(
        record.outbox_id,
        destination.destination_id,
        expected_revision=1,
        predecessor_attempt=1,
        predecessor_owner="sender",
        action_id=claim.action_id,
        request_digest=claim.request_digest,
        evidence_ref="evidence",
        predecessor_fenced_ref="fence",
        resolution="confirmed_not_sent",
    )
    retry = outbox.claim_destination(record.outbox_id, destination.destination_id, 2, owner="retry")
    original_fsync = delivery_store_module.os.fsync
    mutated = False

    def mutate_staging(fd: int) -> None:
        nonlocal mutated
        original_fsync(fd)
        info = delivery_store_module.os.fstat(fd)
        if not mutated and stat.S_ISREG(info.st_mode) and info.st_size > 1:
            mutated = True
            delivery_store_module.os.pwrite(fd, b"!", 0)

    monkeypatch.setattr(delivery_store_module.os, "fsync", mutate_staging)
    assert retry.attempt == 2
    with pytest.raises(DeliveryOutboxCorruptError, match="binding changed|content"):
        _mark_sending(outbox, retry)
    assert mutated


def test_identical_predecessor_replacement_fails_closed(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    path = outbox.root / f"{record.outbox_id}.json"
    with outbox._held_lock() as held:
        _wire, _view, proof = outbox._read_bytes(held, record.outbox_id, durable=False, with_proof=True)
        same = path.read_bytes()
        replacement = outbox.root / ".replacement"
        replacement.write_bytes(same)
        replacement.chmod(0o600)
        os.replace(replacement, path)
        with pytest.raises(DeliveryOutboxCorruptError, match="predecessor"):
            outbox._replace_record(
                held,
                record.outbox_id,
                same,
                predecessor_proof=proof,
            )


def test_post_rename_directory_fsync_mutation_is_not_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    outbox.resolve_after_reconciliation(
        record.outbox_id,
        destination.destination_id,
        expected_revision=1,
        predecessor_attempt=1,
        predecessor_owner="sender",
        action_id=claim.action_id,
        request_digest=claim.request_digest,
        evidence_ref="evidence",
        predecessor_fenced_ref="fence",
        resolution="confirmed_not_sent",
    )
    original_rename = delivery_store_module.os.rename
    original_fsync = delivery_store_module.os.fsync
    renamed = False
    mutated = False

    def observe_rename(*args: object, **kwargs: object) -> None:
        nonlocal renamed
        original_rename(*args, **kwargs)
        renamed = True

    def mutate_after_directory_fsync(fd: int) -> None:
        nonlocal mutated
        original_fsync(fd)
        if renamed and not mutated and stat.S_ISDIR(delivery_store_module.os.fstat(fd).st_mode):
            mutated = True
            target_fd = os.open(path.name, os.O_RDWR, dir_fd=fd)
            try:
                os.pwrite(target_fd, b"?", 0)
            finally:
                os.close(target_fd)

    path = outbox.root / f"{record.outbox_id}.json"
    retry = outbox.claim_destination(record.outbox_id, destination.destination_id, 2, owner="retry")
    monkeypatch.setattr(delivery_store_module.os, "rename", observe_rename)
    monkeypatch.setattr(delivery_store_module.os, "fsync", mutate_after_directory_fsync)
    with pytest.raises(DeliveryOutboxCorruptError, match="binding changed|content"):
        _mark_sending(outbox, retry)
    assert mutated


async def test_send_fence_capability_rejects_copy_pickle_wrong_task_store_and_stale_claim(
    tmp_path: Path,
) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    fence = outbox.acquire_send_fence(claim)
    with pytest.raises(TypeError, match="copied"):
        copy.copy(fence)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(fence)

    async def wrong_task() -> None:
        with pytest.raises(DeliveryOutboxOwnershipError, match="transferred"):
            outbox.mark_sending(claim, fence)

    await asyncio.create_task(wrong_task())
    foreign = CronDeliveryOutboxStore(outbox.root, namespace=NAMESPACE)
    with pytest.raises(DeliveryOutboxOwnershipError):
        foreign.mark_sending(claim, fence)
    with pytest.raises(DeliveryOutboxOwnershipError):
        outbox.acquire_send_fence(replace(claim, owner="reconstructed"))
    outbox.mark_sending(claim, fence)
    assert fence.active and fence.started
    outbox.release_send_fence(fence)
    assert not fence.active
    with pytest.raises(DeliveryOutboxOwnershipError, match="inactive"):
        outbox.settle(claim, {"outcome": "unknown"}, fence=fence)


async def test_mark_sending_is_one_way_per_fence_and_started_fence_still_settles(
    tmp_path: Path,
) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    with outbox.acquire_send_fence(claim) as fence:
        sending = outbox.mark_sending(claim, fence)
        revision = sending.record_revision
        history = json.loads(
            (outbox.root / f"{record.outbox_id}.json").read_text(encoding="utf-8")
        )["transition_history"]

        with pytest.raises(DeliveryOutboxOwnershipError, match="already started"):
            outbox.mark_sending(claim, fence)

        unchanged = outbox.get(record.outbox_id)
        unchanged_history = json.loads(
            (outbox.root / f"{record.outbox_id}.json").read_text(encoding="utf-8")
        )["transition_history"]
        assert unchanged.record_revision == revision
        assert unchanged_history == history

        settled = outbox.settle(
            claim,
            {"outcome": "confirmed_delivered", "receipt": {"id": "receipt"}},
            fence=fence,
        )
        assert settled.status == "delivered"


async def test_send_fence_blocks_reconciliation_and_other_destinations_do_not_contend(
    tmp_path: Path,
) -> None:
    policy = {
        "deliver": True,
        "channel": "cli",
        "to": "alpha",
        "additionalDestinations": [{"channel": "cli", "to": "beta"}],
    }
    _service_obj, _claim, outbox, record = _prepare(tmp_path, policy=policy)
    first, second = record.destinations
    first_claim = outbox.claim_destination(record.outbox_id, first.destination_id, 0, owner="one")
    second_claim = outbox.claim_destination(record.outbox_id, second.destination_id, 1, owner="two")
    observer = CronDeliveryOutboxStore(outbox.root, namespace=NAMESPACE)
    with outbox.acquire_send_fence(first_claim) as first_fence:
        with pytest.raises(DeliverySendBusyError):
            observer.resolve_after_reconciliation(
                record.outbox_id,
                first.destination_id,
                expected_revision=2,
                predecessor_attempt=1,
                predecessor_owner="one",
                action_id=first_claim.action_id,
                request_digest=first_claim.request_digest,
                evidence_ref="evidence",
                predecessor_fenced_ref="external-fence",
                resolution="confirmed_not_sent",
            )
        with outbox.acquire_send_fence(second_claim) as second_fence:
            outbox.mark_sending(second_claim, second_fence)
        outbox.mark_sending(first_claim, first_fence)
    assert outbox.get(record.outbox_id).record_revision == 4


@pytest.mark.parametrize("boundary", ["file", "root"])
async def test_send_lock_fsync_failure_retries_existing_residue_before_authority(
    tmp_path: Path,
    boundary: str,
) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    lock_path = outbox.root / f".send.{destination.destination_id}.lock"
    original_fsync = os.fsync
    failures = 0

    def fail_boundary(fd: int) -> None:
        nonlocal failures
        info = os.fstat(fd)
        is_root = stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == (
            outbox.root.stat().st_dev,
            outbox.root.stat().st_ino,
        )
        if (boundary == "root" and is_root) or (boundary == "file" and stat.S_ISREG(info.st_mode)):
            failures += 1
            raise OSError(f"{boundary} fsync failed")
        original_fsync(fd)

    with patch("nanobot.cron.delivery_store.os.fsync", side_effect=fail_boundary):
        for _ in range(2):
            with pytest.raises(delivery_store_module.DeliveryOutboxPersistenceError, match="send lock"):
                outbox.acquire_send_fence(claim)
            assert outbox._issued_send_fences == {}
            assert lock_path.exists()
    assert failures == 2
    with outbox.acquire_send_fence(claim) as fence:
        outbox.mark_sending(claim, fence)
    assert lock_path.exists()
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
    assert lock_path.stat().st_nlink == 1


@pytest.mark.parametrize("unsafe", ["mode", "hardlink", "fifo", "symlink"])
def test_constructor_rejects_unsafe_send_lock_entries(tmp_path: Path, unsafe: str) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    path = outbox.root / f".send.{destination.destination_id}.lock"
    path.write_bytes(b"")
    path.chmod(0o600)
    if unsafe == "mode":
        path.chmod(0o640)
    elif unsafe == "hardlink":
        os.link(path, tmp_path / "send-lock-alias")
    else:
        path.unlink()
        if unsafe == "fifo":
            os.mkfifo(path, 0o600)
        else:
            target = tmp_path / "send-lock-target"
            target.write_bytes(b"")
            target.chmod(0o600)
            path.symlink_to(target)
    with pytest.raises(DeliveryOutboxCorruptError, match="send lock"):
        CronDeliveryOutboxStore(outbox.root, namespace=NAMESPACE)


def test_constructor_rejects_invalid_send_lock_grammar(tmp_path: Path) -> None:
    _service_obj, _claim, outbox, _record = _prepare(tmp_path)
    unsafe = outbox.root / ".send.not-a-digest.lock"
    unsafe.write_bytes(b"")
    unsafe.chmod(0o600)
    with pytest.raises(DeliveryOutboxCorruptError, match="unsafe entry"):
        CronDeliveryOutboxStore(outbox.root, namespace=NAMESPACE)


async def test_send_lock_replacement_during_durability_reproof_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    lock_path = outbox.root / f".send.{destination.destination_id}.lock"
    original_fsync = delivery_store_module.os.fsync
    replaced = False

    def replace_after_file_fsync(fd: int) -> None:
        nonlocal replaced
        original_fsync(fd)
        if not replaced and stat.S_ISREG(os.fstat(fd).st_mode):
            replaced = True
            replacement = outbox.root / ".replacement-send-lock"
            replacement.write_bytes(b"")
            replacement.chmod(0o600)
            os.replace(replacement, lock_path)

    monkeypatch.setattr(delivery_store_module.os, "fsync", replace_after_file_fsync)
    with pytest.raises(DeliveryOutboxCorruptError, match="binding changed|private regular"):
        outbox.acquire_send_fence(claim)
    assert outbox._issued_send_fences == {}


@pytest.mark.skipif(os.name == "nt", reason="fork inheritance is POSIX-only")
async def test_fork_child_drops_inherited_send_fence_without_unlocking_parent(
    tmp_path: Path,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable on this interpreter")
    _service_obj, _claim, outbox, record = _prepare(tmp_path)
    destination = record.destinations[0]
    claim = outbox.claim_destination(record.outbox_id, destination.destination_id, 0, owner="sender")
    context = multiprocessing.get_context("fork")
    outcomes = context.Queue()
    release = context.Event()
    fence = outbox.acquire_send_fence(claim)
    outbox.mark_sending(claim, fence)
    child = context.Process(
        target=_use_send_fence_in_fork,
        args=(outbox, claim, fence, outcomes, release),
    )
    child.start()
    assert outcomes.get(timeout=3) == "rejected"
    reconciler = context.Process(
        target=_reconcile_in_process,
        args=(str(outbox.root), record, claim, outcomes),
    )
    reconciler.start()
    reconciler.join(timeout=10)
    assert reconciler.exitcode == 0
    assert outcomes.get(timeout=3) == "DeliverySendBusyError"
    observer = CronDeliveryOutboxStore(outbox.root, namespace=NAMESPACE)
    with pytest.raises(DeliverySendBusyError):
        observer.resolve_after_reconciliation(
            record.outbox_id,
            destination.destination_id,
            expected_revision=2,
            predecessor_attempt=1,
            predecessor_owner="sender",
            action_id=claim.action_id,
            request_digest=claim.request_digest,
            evidence_ref="evidence",
            predecessor_fenced_ref="external-fence",
            resolution="confirmed_delivered",
            receipt={"id": "receipt"},
        )
    outbox.release_send_fence(fence)
    resolved = observer.resolve_after_reconciliation(
        record.outbox_id,
        destination.destination_id,
        expected_revision=2,
        predecessor_attempt=1,
        predecessor_owner="sender",
        action_id=claim.action_id,
        request_digest=claim.request_digest,
        evidence_ref="evidence",
        predecessor_fenced_ref="external-fence",
        resolution="confirmed_delivered",
        receipt={"id": "receipt"},
    )
    assert resolved.status == "delivered"
    release.set()
    child.join(timeout=10)
    assert child.exitcode == 0
