"""C3 definition-only CronService integration coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.cron.action_store import ActionStoreStaleGenerationError, CronActionStore
from nanobot.cron.definition_store import (
    CronDefinitionStore,
    DefinitionStoreConflictError,
)
from nanobot.cron.service import (
    CRON_DEFINITION_ONLY_DIAGNOSTIC,
    CronDefinitionConflictError,
    CronDefinitionServiceError,
    CronDefinitionTokenRequiredError,
    CronService,
)
from nanobot.cron.types import CronSchedule

NAMESPACE = "cron:c3-service-tests"


def _service(tmp_path: Path, callback=None) -> CronService:
    root = tmp_path / "cron"
    definition_store = CronDefinitionStore(
        CronActionStore(root, namespace=NAMESPACE),
    )
    return CronService(
        root / "jobs.json",
        on_job=callback,
        definition_store=definition_store,
    )


def _legacy_job(job_id: str, name: str) -> dict[str, object]:
    return {
        "id": job_id,
        "name": name,
        "enabled": True,
        "schedule": {
            "kind": "every",
            "atMs": None,
            "everyMs": 60_000,
            "expr": None,
            "tz": None,
        },
        "payload": {
            "kind": "agent_turn",
            "message": "send report",
            "deliver": False,
            "channel": None,
            "to": None,
            "additionalDestinations": [],
            "channelMeta": {},
            "sessionKey": None,
            "executionMode": "session_bound",
            "originChannel": None,
            "originChatId": None,
            "originMetadata": {},
        },
        "state": {
            "nextRunAtMs": 100,
            "lastRunAtMs": 90,
            "lastStatus": "ok",
            "lastError": None,
            "lastRunStatus": "succeeded",
            "runHistory": [],
        },
        "createdAtMs": 1,
        "updatedAtMs": 2,
        "deleteAfterRun": False,
        "jobInstanceId": None,
        "definitionRevision": None,
    }


def _write_private(path: Path, value: bytes) -> None:
    path.write_bytes(value)
    os.chmod(path, 0o600)


def test_management_uses_typed_definitions_without_legacy_files(tmp_path: Path) -> None:
    service = _service(tmp_path)

    job = service.add_job(
        "daily",
        CronSchedule(kind="every", every_ms=60_000),
        "send report",
    )

    assert job.job_instance_id
    assert job.definition_revision
    assert service.get_job(job.id) == job
    assert service.store_path == tmp_path / "cron" / "jobs.json"
    assert not service.store_path.exists()
    assert not (service.store_path.parent / "action.jsonl").exists()

    raw = service.definition_store.action_store.replay()
    assert raw.candidate is not None
    assert raw.candidate["kind"] == "cron_definition_candidate_v1"
    assert raw.pending_actions == ()
    assert "state" not in raw.candidate["definitions"][0]
    assert "createdAtMs" not in raw.candidate["definitions"][0]


def test_update_and_delete_fail_closed_without_exact_tokens(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = service.add_job(
        "daily",
        CronSchedule(kind="every", every_ms=60_000),
        "send report",
    )

    with pytest.raises(CronDefinitionTokenRequiredError, match="job_instance_id"):
        service.update_job(job.id, name="changed")
    with pytest.raises(CronDefinitionTokenRequiredError, match="definition_revision"):
        service.remove_job(job.id, job_instance_id=job.job_instance_id)

    changed = service.update_job(
        job.id,
        name="changed",
        job_instance_id=job.job_instance_id,
        definition_revision=job.definition_revision,
    )
    assert changed.name == "changed"
    assert changed.job_instance_id == job.job_instance_id
    assert changed.definition_revision != job.definition_revision

    assert (
        service.remove_job(
            job.id,
            job_instance_id=changed.job_instance_id,
            definition_revision=changed.definition_revision,
        )
        == "removed"
    )
    assert service.list_jobs(include_disabled=True) == []


def test_stale_revision_is_rejected_by_definition_authority(tmp_path: Path) -> None:
    first = _service(tmp_path)
    second = _service(tmp_path)
    original = first.add_job(
        "daily",
        CronSchedule(kind="every", every_ms=60_000),
        "send report",
    )
    changed = second.update_job(
        original.id,
        name="changed elsewhere",
        job_instance_id=original.job_instance_id,
        definition_revision=original.definition_revision,
    )

    with pytest.raises(DefinitionStoreConflictError, match="revision"):
        first.update_job(
            original.id,
            name="stale update",
            job_instance_id=original.job_instance_id,
            definition_revision=original.definition_revision,
        )
    assert first.get_job(original.id).name == changed.name
    assert first.definition_store.action_store.replay().pending_actions == ()


def test_checkpoint_stale_after_receipt_does_not_append_a_second_action(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    original_checkpoint = service.definition_store.checkpoint

    def reject_durable_checkpoint(replay):
        if replay.pending_actions:
            raise ActionStoreStaleGenerationError("simulated stale checkpoint")
        return original_checkpoint(replay)

    monkeypatch.setattr(service.definition_store, "checkpoint", reject_durable_checkpoint)
    with pytest.raises(CronDefinitionConflictError, match="durably acknowledged"):
        service.add_job(
            "daily",
            CronSchedule(kind="every", every_ms=60_000),
            "send report",
        )

    raw = service.definition_store.action_store.replay()
    assert len(raw.receipts) == 1
    assert len(raw.pending_actions) == 1


@pytest.mark.asyncio
async def test_start_and_all_execution_entrypoints_remain_fenced(tmp_path: Path) -> None:
    callback = AsyncMock()
    service = _service(tmp_path, callback)
    service.add_job(
        "daily",
        CronSchedule(kind="every", every_ms=60_000),
        "send report",
    )

    await service.start()
    assert service._timer_task is None
    jobs = service.list_jobs(include_disabled=True)
    assert jobs
    assert all(service.admission_diagnostic(job) == CRON_DEFINITION_ONLY_DIAGNOSTIC for job in jobs)
    assert await service.run_job(jobs[0].id, force=True) is False
    await service._on_timer()
    callback.assert_not_awaited()
    assert service._get_next_wake_ms() is None


def test_legacy_jobs_snapshot_is_never_a_definition_fallback(tmp_path: Path) -> None:
    root = tmp_path / "cron"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    (root / "jobs.json").write_text('{"version": 1, "jobs": []}', encoding="utf-8")

    service = CronService(root / "jobs.json")

    with pytest.raises(CronDefinitionServiceError, match="explicit stopped-writer conversion"):
        service.list_jobs(include_disabled=True)
    assert (root / "jobs.json").read_text(encoding="utf-8") == '{"version": 1, "jobs": []}'


def test_service_materializes_converted_operations_once(tmp_path: Path) -> None:
    root = tmp_path / "cron"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    job = _legacy_job("daily", "Daily snapshot")
    _write_private(
        root / "action.jsonl",
        (json.dumps({"action": "add", "params": job}, separators=(",", ":")) + "\n").encode(),
    )
    _write_private(
        root / "jobs.json",
        json.dumps({"version": 1, "jobs": [job]}, separators=(",", ":")).encode(),
    )
    CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    definition_store = CronDefinitionStore(CronActionStore(root, namespace=NAMESPACE))
    service = CronService(root / "jobs.json", definition_store=definition_store)

    first = service.list_jobs(include_disabled=True)
    raw = definition_store.action_store.replay()
    second = service.list_jobs(include_disabled=True)
    settled = definition_store.action_store.replay()

    assert [item.name for item in first] == ["Daily snapshot"]
    assert second == first
    assert raw.candidate is not None
    assert raw.candidate["kind"] == "cron_definition_candidate_v1"
    assert "operations" not in raw.candidate
    assert settled.candidate == raw.candidate
    assert "state" not in json.dumps(settled.candidate, sort_keys=True)
