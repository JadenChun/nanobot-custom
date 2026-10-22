"""Package E1 acceptance tests for canonical completion provenance."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import nanobot.cron.completion as completion_module
import nanobot.cron.occurrence_store as occurrence_store_module
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.completion import (
    CronCompletionConflictError,
    CronCompletionPersistenceError,
    CronCompletionProjector,
)
from nanobot.cron.definition_store import CronDefinitionStore
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceCorruptError,
    OccurrenceReconciliationRequiredError,
)
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule

NAMESPACE = "cron:package-e1-tests"


def test_projector_captures_admitted_flock_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    module = completion_module._fcntl
    assert module is not None
    calls: list[int] = []
    original = module.flock

    def flock(fd: int, operation: int) -> None:
        calls.append(operation)
        original(fd, operation)

    monkeypatch.setattr(module, "flock", flock)
    # Probe the captured collaborator, not a supported global-rebinding API.
    monkeypatch.setattr(completion_module, "_fcntl", None)
    with service.completion_projector._held_lock() as held:
        assert held.lock_fd >= 0
    assert calls == [module.LOCK_EX, module.LOCK_UN]


def test_projector_missing_flock_refuses_before_root_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CronOccurrenceStore(tmp_path / "occurrences", namespace=NAMESPACE)
    root = tmp_path / "runs"
    monkeypatch.setattr(completion_module, "_fcntl", None)
    with pytest.raises(CronCompletionPersistenceError, match="POSIX flock"):
        CronCompletionProjector(store, root)
    assert not root.exists()


def _service(tmp_path: Path) -> CronService:
    root = tmp_path / "cron"
    definition_store = CronDefinitionStore(CronActionStore(root, namespace=NAMESPACE))
    occurrence_store = CronOccurrenceStore(root / "occurrences", namespace=NAMESPACE)
    completion_projector = CronCompletionProjector(occurrence_store, root / "runs")
    return CronService(
        root / "jobs.json",
        definition_store=definition_store,
        occurrence_store=occurrence_store,
        completion_projector=completion_projector,
    )


def _claim(service: CronService):
    job = service.add_job(
        "daily",
        CronSchedule(kind="every", every_ms=60_000),
        "send report",
    )
    return service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )


def _complete(service: CronService, claim, completion, **kwargs):
    async def execute():
        with service.occurrence_store.acquire_execution_fence(claim) as fence:
            service.occurrence_store.begin_execution(claim, fence)
            return service.completion_projector.complete(
                claim, completion, fence=fence, **kwargs
            )

    return asyncio.run(execute())


def _complete_raw(service: CronService, claim, completion):
    async def execute():
        with service.occurrence_store.acquire_execution_fence(claim) as fence:
            service.occurrence_store.begin_execution(claim, fence)
            return service.occurrence_store.complete(claim, completion, fence)

    return asyncio.run(execute())


def test_provenance_is_canonical_before_projection_and_survives_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    projector = service.completion_projector
    provenance = {"session_key": "telegram:42", "source": "timer"}
    monkeypatch.setattr(
        projector,
        "_replace_projection",
        lambda *_args: (_ for _ in ()).throw(OSError("projection disk")),
    )

    with pytest.raises(OSError, match="projection disk"):
        _complete(service, claim, {"status": "ok", "result": "computed"}, provenance=provenance)

    durable = service.occurrence_store.get_completed_durable(claim.occurrence_id)
    assert durable.completion is not None
    assert durable.completion["provenance"] == provenance

    restarted_store = CronOccurrenceStore(tmp_path / "cron" / "occurrences", namespace=NAMESPACE)
    restarted_projector = CronCompletionProjector(restarted_store, tmp_path / "cron" / "runs")
    recovered = restarted_projector.project_completed(claim.occurrence_id)
    assert recovered.provenance == provenance
    assert recovered.completion["provenance"] == provenance


def test_provenance_is_strict_and_reserved_conflicts_fail_before_completion(tmp_path: Path) -> None:
    service = _service(tmp_path)
    claim = _claim(service)

    with pytest.raises(CronCompletionConflictError, match="provenance"):
        service.completion_projector.complete(
            claim,
            {"status": "ok", "provenance": {"source": "completion"}},
            provenance={"source": "argument"},
            fence=None,  # type: ignore[arg-type]
        )
    assert service.occurrence_store.get(claim.occurrence_id).state == "claimed"

    with pytest.raises(CronCompletionConflictError, match="provenance"):
        service.completion_projector.complete(
            claim, {"status": "ok"}, provenance=["invalid"], fence=None  # type: ignore[arg-type]
        )
    with pytest.raises(CronCompletionConflictError, match="finite"):
        service.completion_projector.complete(
            claim,
            {"status": "ok"},
            provenance={"invalid": float("nan")},
            fence=None,  # type: ignore[arg-type]
        )
    with pytest.raises(CronCompletionConflictError, match="provenance"):
        service.completion_projector.complete(
            claim,
            {"status": "ok"},
            provenance={"generation": 1.0},
            fence=None,  # type: ignore[arg-type]
        )


def test_historical_completion_without_provenance_remains_empty_and_unfabricated(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    _complete_raw(service, claim, {"status": "ok", "result": "legacy"})

    record = service.occurrence_store.get_completed_durable(claim.occurrence_id)
    assert record.completion is not None
    assert "provenance" not in record.completion
    projection = service.completion_projector.project_completed(claim.occurrence_id)
    assert projection.provenance == {}

    with pytest.raises(CronCompletionConflictError, match="provenance"):
        service.completion_projector.project_completed(
            claim.occurrence_id,
            provenance={"source": "fabricated"},
        )


def test_completed_durable_read_reproves_record_and_directory_and_rejects_claimed_or_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    with pytest.raises(ValueError, match="occurrence_id"):
        service.occurrence_store.get_completed_durable(1)  # type: ignore[arg-type]
    with pytest.raises(OccurrenceReconciliationRequiredError, match="claimed"):
        service.occurrence_store.get_completed_durable(claim.occurrence_id)

    _complete_raw(service, claim, {"status": "ok", "result": "computed"})
    fsynced: list[int] = []
    original_fsync = occurrence_store_module.os.fsync

    def record_fsync(fd: int) -> None:
        fsynced.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(occurrence_store_module.os, "fsync", record_fsync)
    durable = service.occurrence_store.get_completed_durable(claim.occurrence_id)
    assert durable.state == "completed"
    assert len(fsynced) >= 2

    record_path = service.occurrence_store.root / f"{claim.occurrence_id}.json"
    record_path.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(OccurrenceCorruptError):
        service.occurrence_store.get_completed_durable(claim.occurrence_id)


def test_completed_durable_read_rejects_live_root_swap(tmp_path: Path) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    _complete_raw(service, claim, {"status": "ok", "result": "computed"})
    root = service.occurrence_store.root
    root.rename(tmp_path / "moved-occurrences")
    root.mkdir(mode=0o700)

    with pytest.raises(OccurrenceCorruptError, match="root.*binding"):
        service.occurrence_store.get_completed_durable(claim.occurrence_id)


def test_completed_durable_read_rejects_record_replacement_after_file_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    _complete_raw(service, claim, {"status": "ok", "result": "computed"})
    record_path = service.occurrence_store.root / f"{claim.occurrence_id}.json"
    record_identity = (record_path.stat().st_dev, record_path.stat().st_ino)
    moved_path = tmp_path / "moved-record.json"
    original_fsync = occurrence_store_module.os.fsync
    replaced = False

    def replace_after_record_fsync(fd: int) -> None:
        nonlocal replaced
        original_fsync(fd)
        if not replaced and (occurrence_store_module.os.fstat(fd).st_dev, occurrence_store_module.os.fstat(fd).st_ino) == record_identity:
            replaced = True
            payload = record_path.read_bytes()
            record_path.rename(moved_path)
            record_path.write_bytes(payload)
            record_path.chmod(0o600)

    monkeypatch.setattr(occurrence_store_module.os, "fsync", replace_after_record_fsync)
    with pytest.raises(OccurrenceCorruptError, match="record binding"):
        service.occurrence_store.get_completed_durable(claim.occurrence_id)


def test_completed_durable_read_rejects_in_place_mutation_after_file_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    _complete_raw(service, claim, {"status": "ok", "result": "computed"})
    record_path = service.occurrence_store.root / f"{claim.occurrence_id}.json"
    record_identity = (record_path.stat().st_dev, record_path.stat().st_ino)
    original_fsync = occurrence_store_module.os.fsync
    mutated = False

    def mutate_after_record_fsync(fd: int) -> None:
        nonlocal mutated
        original_fsync(fd)
        if not mutated and (occurrence_store_module.os.fstat(fd).st_dev, occurrence_store_module.os.fstat(fd).st_ino) == record_identity:
            mutated = True
            with record_path.open("r+b") as record_file:
                record_file.seek(0)
                record_file.write(b" ")

    monkeypatch.setattr(occurrence_store_module.os, "fsync", mutate_after_record_fsync)
    with pytest.raises(OccurrenceCorruptError, match="record"):
        service.occurrence_store.get_completed_durable(claim.occurrence_id)


def test_completed_durable_read_rejects_root_swap_after_file_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    _complete_raw(service, claim, {"status": "ok", "result": "computed"})
    root = service.occurrence_store.root
    original_fsync = occurrence_store_module.os.fsync
    swapped = False

    def swap_root_after_file_fsync(fd: int) -> None:
        nonlocal swapped
        original_fsync(fd)
        if not swapped:
            swapped = True
            root.rename(tmp_path / "moved-occurrences-during-fsync")
            root.mkdir(mode=0o700)

    monkeypatch.setattr(occurrence_store_module.os, "fsync", swap_root_after_file_fsync)
    with pytest.raises(OccurrenceCorruptError, match="root.*binding"):
        service.occurrence_store.get_completed_durable(claim.occurrence_id)


def _write_legacy_projection(service: CronService, claim: object, provenance: dict[str, object]) -> Path:
    projector = service.completion_projector
    record = service.occurrence_store.get_completed_durable(claim.occurrence_id)
    projection_claim = claim
    completion = projector._validate_completion(
        record.completion,
        claim=projection_claim,
        run_id=claim.run_id,
    )
    data = projector._wire_projection(
        claim=projection_claim,
        completion=completion,
        provenance=provenance,
    )
    with projector._held_lock() as held:
        projector._replace_projection(claim.run_id, data, held)
    return projector.projection_path(claim.run_id)


def test_legacy_projection_provenance_is_preserved_only_without_explicit_retry(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    _complete_raw(service, claim, {"status": "ok", "result": "legacy"})
    historical = {"source": "historical-d"}
    path = _write_legacy_projection(service, claim, historical)
    before = path.read_bytes()

    recovered = service.completion_projector.project_completed(claim.occurrence_id)
    assert recovered.provenance == historical
    assert path.read_bytes() == before

    with pytest.raises(CronCompletionConflictError, match="provenance"):
        service.completion_projector.project_completed(
            claim.occurrence_id,
            provenance=historical,
        )
    with pytest.raises(CronCompletionConflictError, match="canonical content"):
        service.completion_projector.project_completed(
            claim.occurrence_id,
            provenance={},
        )
    assert path.read_bytes() == before


def test_projection_rejects_canonical_provenance_bool_integer_mismatch(tmp_path: Path) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    projection = _complete(
        service,
        claim,
        {"status": "ok", "result": "computed"},
        provenance={"flag": True},
    )
    path = service.completion_projector.projection_path(projection.run_id)
    wire = json.loads(path.read_text(encoding="utf-8"))
    wire["provenance"] = {"flag": 1}
    wire["integrity"] = completion_module._integrity(
        {key: value for key, value in wire.items() if key != "integrity"}
    )
    path.write_text(
        completion_module._canonical_json(wire, label="test projection"),
        encoding="utf-8",
    )

    with pytest.raises(CronCompletionConflictError, match="provenance"):
        service.completion_projector.project_completed(claim.occurrence_id)


def test_projection_read_rejects_record_replacement_after_file_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    projection = _complete(service, claim, {"status": "ok", "result": "computed"})
    path = service.completion_projector.projection_path(projection.run_id)
    projection_identity = (path.stat().st_dev, path.stat().st_ino)
    moved_path = tmp_path / "moved-projection.json"
    original_fsync = completion_module.os.fsync
    replaced = False

    def replace_after_projection_fsync(fd: int) -> None:
        nonlocal replaced
        original_fsync(fd)
        if not replaced and (completion_module.os.fstat(fd).st_dev, completion_module.os.fstat(fd).st_ino) == projection_identity:
            replaced = True
            payload = path.read_bytes()
            path.rename(moved_path)
            path.write_bytes(payload)
            path.chmod(0o600)

    monkeypatch.setattr(completion_module.os, "fsync", replace_after_projection_fsync)
    with pytest.raises(CronCompletionConflictError, match="projection binding"):
        service.completion_projector.project_completed(claim.occurrence_id)


def test_projection_read_rejects_in_place_mutation_after_file_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    projection = _complete(service, claim, {"status": "ok", "result": "computed"})
    path = service.completion_projector.projection_path(projection.run_id)
    projection_identity = (path.stat().st_dev, path.stat().st_ino)
    original_fsync = completion_module.os.fsync
    mutated = False

    def mutate_after_projection_fsync(fd: int) -> None:
        nonlocal mutated
        original_fsync(fd)
        if not mutated and (completion_module.os.fstat(fd).st_dev, completion_module.os.fstat(fd).st_ino) == projection_identity:
            mutated = True
            with path.open("r+b") as projection_file:
                projection_file.seek(0)
                projection_file.write(b" ")

    monkeypatch.setattr(completion_module.os, "fsync", mutate_after_projection_fsync)
    with pytest.raises(CronCompletionConflictError, match="projection"):
        service.completion_projector.project_completed(claim.occurrence_id)


def test_projection_read_rejects_root_swap_after_file_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    claim = _claim(service)
    projection = _complete(service, claim, {"status": "ok", "result": "computed"})
    path = service.completion_projector.projection_path(projection.run_id)
    projection_identity = (path.stat().st_dev, path.stat().st_ino)
    root = service.completion_projector.root
    original_fsync = completion_module.os.fsync
    swapped = False

    def swap_root_after_projection_fsync(fd: int) -> None:
        nonlocal swapped
        original_fsync(fd)
        if not swapped and (completion_module.os.fstat(fd).st_dev, completion_module.os.fstat(fd).st_ino) == projection_identity:
            swapped = True
            root.rename(tmp_path / "moved-runs-during-fsync")
            root.mkdir(mode=0o700)

    monkeypatch.setattr(completion_module.os, "fsync", swap_root_after_projection_fsync)
    with pytest.raises(CronCompletionConflictError, match="root.*binding"):
        service.completion_projector.project_completed(claim.occurrence_id)
