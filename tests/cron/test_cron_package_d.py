"""Package D acceptance tests for occurrence authority and completion projection."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nanobot.cron.action_store import CronActionStore
from nanobot.cron.completion import (
    CronCompletionConflictError,
    CronCompletionIncompleteError,
    CronCompletionProjector,
)
from nanobot.cron.definition_store import CronDefinitionStore
from nanobot.cron.occurrence import (
    CronOccurrenceDefinitionConflictError,
    CronOccurrenceRuntimeUnavailableError,
)
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceReconciliationRequiredError,
)
from nanobot.cron.service import CRON_DEFINITION_ONLY_DIAGNOSTIC, CronService
from nanobot.cron.types import CronSchedule

NAMESPACE = "cron:package-d-tests"


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


def _job(service: CronService):
    return service.add_job(
        "daily",
        CronSchedule(kind="every", every_ms=60_000),
        "send report",
    )


def _complete(service: CronService, claim, completion, **kwargs):
    async def execute():
        with service.occurrence_store.acquire_execution_fence(claim) as fence:
            service.occurrence_store.begin_execution(claim, fence)
            return service.complete_occurrence(claim, completion, fence=fence, **kwargs)

    return asyncio.run(execute())


def test_definition_bound_claim_retry_is_exact_and_identity_is_frozen(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = _job(service)

    first = service.claim_timer_occurrence(
        job.id,
        1_700_000_000_000,
        owner="timer-worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    retry = service.claim_manual_occurrence(
        job.id,
        1_700_000_000_000,
        owner="timer-worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )

    assert retry is first
    assert first.identity.job_id == job.id
    assert first.identity.job_revision == job.definition_revision
    assert first.identity.scheduled_fire_time_ms == 1_700_000_000_000
    assert first.job_snapshot["definitionRevision"] == job.definition_revision
    assert first.destination_policy["deliver"] is False


def test_timer_manual_and_system_attempts_share_one_durable_claim(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = _job(service)
    outcomes: list[object] = []

    def attempt(source: str, owner: str) -> None:
        try:
            outcomes.append(
                service.occurrence_runtime.claim(
                    job_id=job.id,
                    scheduled_fire_time_ms=1_700_000_000_000,
                    source=source,
                    owner=owner,
                    expected_definition_revision=job.definition_revision,
                    expected_job_instance_id=job.job_instance_id,
                )
            )
        except OccurrenceReconciliationRequiredError as exc:
            outcomes.append(exc)

    threads = [
        (source, f"worker-{source}") for source in ("timer", "manual", "system")
    ]
    for source, owner in threads:
        attempt(source, owner)

    assert sum(not isinstance(item, Exception) for item in outcomes) == 1
    assert sum(isinstance(item, OccurrenceReconciliationRequiredError) for item in outcomes) == 2


def test_definition_change_and_delete_reject_stale_occurrence_admission(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = _job(service)
    stale_revision = job.definition_revision
    changed = service.update_job(
        job.id,
        message="changed",
        job_instance_id=job.job_instance_id,
        definition_revision=job.definition_revision,
    )

    with pytest.raises(CronOccurrenceDefinitionConflictError, match="revision"):
        service.claim_scheduled_occurrence(
            job.id,
            1_700_000_000_000,
            owner="stale-worker",
            expected_job_instance_id=job.job_instance_id,
            expected_definition_revision=stale_revision,
        )

    assert changed.definition_revision != stale_revision
    assert service.remove_job(
        job.id,
        job_instance_id=changed.job_instance_id,
        definition_revision=changed.definition_revision,
    ) == "removed"
    with pytest.raises(CronOccurrenceDefinitionConflictError, match="does not exist"):
        service.claim_scheduled_occurrence(
            job.id,
            1_700_000_060_000,
            owner="deleted-worker",
        )


async def test_completion_is_bound_to_run_identity_and_idempotent(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = _job(service)
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )

    with service.occurrence_store.acquire_execution_fence(claim) as fence:
        service.occurrence_store.begin_execution(claim, fence)
        first = service.complete_occurrence(
            claim,
            {"status": "ok", "result": "computed"},
            provenance={"session_key": "telegram:42", "source": "timer"},
            fence=fence,
        )
        duplicate = service.complete_occurrence(
            claim,
            {"result": "computed", "status": "ok"},
            provenance={"source": "timer", "session_key": "telegram:42"},
            fence=fence,
        )
    assert duplicate == first
    assert first.occurrence_id == claim.occurrence_id
    assert first.run_id == claim.run_id
    assert first.delivery_state == "fenced"
    assert first.completion["occurrence_id"] == claim.occurrence_id

    with pytest.raises(CronCompletionConflictError, match="run identity"):
        service.completion_projector.project_completed(
            claim.occurrence_id,
            run_id="forged-run",
        )


def test_completion_projection_failure_is_recoverable_and_claimed_is_never_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    job = _job(service)
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    projector = service.completion_projector
    original_replace = projector._replace_projection
    monkeypatch.setattr(projector, "_replace_projection", lambda *_args: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(OSError, match="disk"):
        _complete(service, claim, {"status": "ok", "result": "computed"})
    assert service.occurrence_store.get(claim.occurrence_id).state == "completed"

    monkeypatch.setattr(projector, "_replace_projection", original_replace)
    recovered = projector.project_completed(claim.occurrence_id, run_id=claim.run_id)
    assert recovered.completion["status"] == "ok"

    second_claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_060_000,
        owner="worker-2",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    with pytest.raises(CronCompletionIncompleteError, match="claimed"):
        projector.project_completed(second_claim.occurrence_id, run_id=second_claim.run_id)


def test_restart_reprojects_completed_and_exposes_claimed_state(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = _job(service)
    completed_claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    _complete(
        service,
        completed_claim,
        {"status": "ok", "result": "computed"},
        provenance={"session_key": "telegram:42"},
    )
    claimed = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_060_000,
        owner="worker-2",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )

    restarted_store = CronOccurrenceStore(tmp_path / "cron" / "occurrences", namespace=NAMESPACE)
    restarted_projector = CronCompletionProjector(
        restarted_store,
        tmp_path / "cron" / "runs",
    )
    recovered = restarted_projector.project_completed(completed_claim.occurrence_id)
    assert recovered.completion["result"] == "computed"
    assert [record.occurrence_id for record in restarted_store.reconciliation_required()] == [
        claimed.occurrence_id
    ]
    with pytest.raises(CronCompletionIncompleteError, match="claimed"):
        restarted_projector.project_completed(claimed.occurrence_id)


def test_service_execution_and_delivery_remain_fenced(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = _job(service)

    assert service.admission_diagnostic(job) == CRON_DEFINITION_ONLY_DIAGNOSTIC
    assert service._timer_task is None


def test_management_only_service_does_not_create_d_runtime_authorities(tmp_path: Path) -> None:
    root = tmp_path / "cron"
    service = CronService(root / "jobs.json")

    assert service.occurrence_store is None
    assert service.occurrence_runtime is None
    assert service.completion_projector is None
    assert not (root / "occurrences").exists()
    assert not (root / "runs").exists()
    with pytest.raises(CronOccurrenceRuntimeUnavailableError, match="execution is fenced"):
        service.claim_occurrence("missing", 1_700_000_000_000)


def test_admit_rechecks_exact_definition_after_final_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    job = _job(service)
    runtime = service.occurrence_runtime
    assert runtime is not None
    original_replay = runtime._committed_definition
    calls = 0

    def replay_with_post_claim_change(job_id: str):
        nonlocal calls
        calls += 1
        definition = original_replay(job_id)
        if calls == 3:
            service.update_job(
                job.id,
                message="changed after claim",
                job_instance_id=job.job_instance_id,
                definition_revision=job.definition_revision,
            )
            definition = original_replay(job_id)
        return definition

    monkeypatch.setattr(runtime, "_committed_definition", replay_with_post_claim_change)
    with pytest.raises(CronOccurrenceDefinitionConflictError, match="changed after"):
        runtime.admit(
            job_id=job.id,
            scheduled_fire_time_ms=1_700_000_000_000,
            source="timer",
            owner="worker",
            expected_job_instance_id=job.job_instance_id,
            expected_definition_revision=job.definition_revision,
        )


def test_completion_rejects_symlinked_ancestor_and_live_root_swap(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = _job(service)
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(CronCompletionConflictError, match="private directory"):
        CronCompletionProjector(service.occurrence_store, link / "runs")

    root = service.completion_projector.root
    root.rename(tmp_path / "old-runs")
    root.mkdir(mode=0o700)
    with pytest.raises(CronCompletionConflictError, match="root binding"):
        _complete(service, claim, {"status": "ok", "result": "computed"})


def test_completion_rejects_root_swap_after_lock_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    job = _job(service)
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    projector = service.completion_projector
    root = projector.root
    moved_root = tmp_path / "moved-runs"
    original_replace = projector._replace_projection

    def replace_after_swap(run_id: str, data: bytes, held: object) -> None:
        root.rename(moved_root)
        root.mkdir(mode=0o700)
        original_replace(run_id, data, held)

    monkeypatch.setattr(projector, "_replace_projection", replace_after_swap)
    with pytest.raises(CronCompletionConflictError, match="root.*binding"):
        _complete(service, claim, {"status": "ok", "result": "computed"})
    assert not (root / projector._projection_name(claim.run_id)).exists()


def test_oversized_valid_projection_is_rejected_before_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    job = _job(service)
    claim = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    projection = _complete(service, claim, {"status": "ok", "result": "computed"})
    path = service.completion_projector.projection_path(projection.run_id)
    payload = path.read_bytes()
    path.write_bytes(payload + b" " * 32)
    monkeypatch.setattr("nanobot.cron.completion._MAX_PROJECTION_BYTES", len(payload) + 1)
    with pytest.raises(CronCompletionConflictError, match="size limit"):
        service.completion_projector.project_completed(claim.occurrence_id, run_id=claim.run_id)
