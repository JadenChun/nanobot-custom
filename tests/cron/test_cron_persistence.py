"""Persistence tests for ``nanobot.cron.service.CronService``.

These tests target the specific failure mode where a corrupt or partially
written ``jobs.json`` would silently turn into an empty job list on the next
start, deleting every scheduled job.  See ``fix(cron): atomic write for
jobs.json + don't silently overwrite corrupt store``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob, CronPayload, CronSchedule
from tests.cron import p6_compat_fixtures as cases
from tests.cron.stage3_fixtures import offline_denial  # noqa: F401


def _seeded_store(tmp_path: Path) -> tuple[CronService, Path]:
    """Build a service with one persisted job on disk and return both the
    service and the resolved store path.  Adds the job via the action log
    (the path used when the service is not running) and then triggers a
    merge so ``jobs.json`` is written, mirroring the persisted on-disk
    state seen in production."""
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path)
    service.add_job(
        name="Daily Loving Message",
        schedule=CronSchedule(kind="cron", expr="0 10 * * *", tz="Asia/Kuwait"),
        message="hello",
    )
    # add_job appended to action.jsonl; flush to jobs.json by toggling
    # ``_running`` long enough for ``_merge_action`` to do its rewrite.
    service._running = True
    try:
        service._load_store()
    finally:
        service._running = False
    assert store_path.exists()
    return service, store_path


def _corrupt_store(tmp_path: Path) -> Path:
    store_path = tmp_path / "cron" / "jobs.json"
    store_path.parent.mkdir(parents=True, mode=0o700)
    store_path.write_text("{not valid json", encoding="utf-8")
    store_path.chmod(0o600)
    return store_path


def _assert_single_corrupt_backup(store_path: Path) -> None:
    assert not store_path.exists()
    backups = list(store_path.parent.glob("jobs.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{not valid json"


def _system_job(job_id: str = "dream") -> CronJob:
    return CronJob(
        id=job_id,
        name="Dream",
        schedule=CronSchedule(kind="cron", expr="0 */2 * * *", tz="UTC"),
        payload=CronPayload(kind="system_event"),
    )


def test_save_store_is_atomic(tmp_path: Path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.typed_roundtrip(tmp_path)


def test_save_store_failure_does_not_corrupt_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.failed_checkpoint(tmp_path, monkeypatch)


def test_atomic_write_ignores_unsupported_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.failed_checkpoint(tmp_path, monkeypatch, directory=True)


def test_load_jobs_preserves_corrupt_store_and_returns_none(
    tmp_path: Path,
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    path = _corrupt_store(tmp_path)
    before = path.read_bytes()
    service = CronService(path)
    with pytest.raises(RuntimeError, match='legacy jobs.json is not a cron definition authority'):
        service._load_jobs()
    with pytest.raises(cases.CronDefinitionServiceError, match='explicit stopped-writer conversion'):
        service.list_jobs()
    assert path.read_bytes() == before and not list(path.parent.glob('*.corrupt-*'))
    root, result, _store = cases.convert(tmp_path / 'explicit', raw=before)
    assert result.snapshot_status == 'unavailable' and (root / 'jobs.json').read_bytes() == before


def test_start_refuses_to_overwrite_corrupt_store(tmp_path: Path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    import asyncio
    path = _corrupt_store(tmp_path)
    before = path.read_bytes()
    service = CronService(path)
    with pytest.raises(cases.CronDefinitionServiceError, match='explicit stopped-writer conversion'):
        asyncio.run(service.start())
    assert not service._running and service._timer_task is None
    assert path.read_bytes() == before
    _root, result, _store = cases.convert(tmp_path / 'explicit', raw=before)
    assert result.snapshot_status == 'unavailable'


def test_load_store_falls_back_to_in_memory_on_corruption_after_start(
    tmp_path: Path,
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.corrupt_authority(tmp_path, converted=True)


@pytest.mark.parametrize(
    ("api_name", "call"),
    [
        ("list_jobs", lambda service: service.list_jobs()),
        ("get_job", lambda service: service.get_job("missing")),
        ("status", lambda service: service.status()),
        ("remove_job", lambda service: service.remove_job("missing")),
        ("enable_job", lambda service: service.enable_job("missing", enabled=False)),
        ("update_job", lambda service: service.update_job("missing", name="new name")),
        ("register_system_job", lambda service: service.register_system_job(_system_job())),
        ("remove_system_job", lambda service: service.remove_system_job("heartbeat")),
    ],
)
def test_public_apis_raise_clear_error_for_unavailable_corrupt_store(
    tmp_path: Path,
    api_name: str,
    call: Callable[[CronService], object],
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    path = _corrupt_store(tmp_path)
    before = path.read_bytes()
    service = CronService(path)
    with pytest.raises(cases.CronDefinitionServiceError, match='explicit stopped-writer conversion'):
        call(service)
    assert api_name and path.read_bytes() == before
    assert not (path.parent / 'action.jsonl').exists() and not list(path.parent.glob('*.corrupt-*'))


@pytest.mark.asyncio
async def test_run_job_raises_clear_error_and_restores_running_state_for_corrupt_store(
    tmp_path: Path,
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    path = _corrupt_store(tmp_path)
    before = path.read_bytes()
    service = CronService(path)
    assert await service.run_job('missing') is False
    assert not service._running and service._timer_task is None
    assert path.read_bytes() == before and not service._run_records_dir.exists()
    with pytest.raises(cases.CronDefinitionServiceError):
        service.list_jobs()
    await cases.completion(tmp_path / 'healthy')


@pytest.mark.asyncio
async def test_run_job_preserves_running_state_when_corrupt_store_unavailable(
    tmp_path: Path,
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    path = _corrupt_store(tmp_path)
    before = path.read_bytes()
    service = CronService(path)
    service._running = True
    try:
        assert await service.run_job('missing') is False
        assert service._running and service._timer_task is None
        assert path.read_bytes() == before
    finally:
        service.stop()
    await cases.completion(tmp_path / 'healthy')


def test_running_add_job_raises_clear_error_for_unavailable_corrupt_store(
    tmp_path: Path,
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    path = _corrupt_store(tmp_path)
    before = path.read_bytes()
    service = CronService(path)
    service._running = True
    try:
        with pytest.raises(cases.CronDefinitionServiceError, match='explicit stopped-writer conversion'):
            service.add_job('running', CronSchedule(kind='every', every_ms=60000), 'hello')
        assert service._running and service._timer_task is None
        assert path.read_bytes() == before and not (path.parent / 'action.jsonl').exists()
    finally:
        service.stop()


def test_stopped_add_job_still_appends_action_without_loading_corrupt_store(
    tmp_path: Path,
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    path = _corrupt_store(tmp_path)
    before = path.read_bytes()
    service = CronService(path)
    with pytest.raises(cases.CronDefinitionServiceError, match='explicit stopped-writer conversion'):
        service.add_job('stopped', CronSchedule(kind='every', every_ms=60000), 'hello')
    assert not service._running and path.read_bytes() == before
    assert not (path.parent / 'action.jsonl').exists()
    cases.typed_roundtrip(tmp_path / 'healthy')


def test_public_api_uses_in_memory_snapshot_when_disk_becomes_corrupt(
    tmp_path: Path,
) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.corrupt_authority(tmp_path)


def test_full_round_trip_survives_repeated_save_load(tmp_path: Path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.typed_roundtrip(tmp_path)
