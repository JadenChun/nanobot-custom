import asyncio

import pytest

from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from tests.cron import p6_compat_fixtures as cases
from tests.cron.stage3_fixtures import offline_denial  # noqa: F401


def test_add_job_rejects_unknown_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    with pytest.raises(ValueError, match="unknown timezone 'America/Vancovuer'"):
        service.add_job(
            name="tz typo",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancovuer"),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_add_job_accepts_valid_timezone(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service = CronService(tmp_path / 'jobs.json')
    job = service.add_job('tz ok', CronSchedule(kind='cron', expr='0 9 * * *', tz='America/Vancouver'), 'hello')
    assert job.schedule.tz == 'America/Vancouver' and job.state.next_run_at_ms is None
    commit = service.definition_store.replay().candidate.scheduling_commits[0]
    assert cases.next_fire(commit['schedule'], commit['commit_clock_ms'], commit['commit_clock_ms']) > commit['commit_clock_ms']
    assert CronService(service.store_path).get_job(job.id).schedule == job.schedule


def test_additional_destinations_are_persisted_and_loaded(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.route_roundtrip(tmp_path)


def test_legacy_job_without_additional_destinations_still_loads(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    job = cases.legacy_job()
    job['payload'].pop('additionalDestinations')
    root, result, store = cases.convert(tmp_path, job)
    assert result.snapshot_converted
    _service, jobs = cases.converted_view(root, store)
    assert jobs[0].payload.additional_destinations == []
    assert [destination.to for destination in jobs[0].payload.delivery_destinations()] == ['42']


@pytest.mark.asyncio
async def test_execute_job_records_run_history(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.completion(tmp_path / 'current')


@pytest.mark.asyncio
async def test_execute_job_records_canonical_run_id_when_callback_returns_result(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.completion(tmp_path / 'valid')
    await cases.completion(tmp_path / 'forged', forge=True)


def test_legacy_run_history_without_run_id_loads_as_none(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.route_roundtrip(tmp_path, legacy=True, anonymous=True)


@pytest.mark.asyncio
async def test_run_history_records_errors(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.completion(tmp_path / 'represented', status='error', error='boom')
    await cases.compute_failure(tmp_path / 'computation')


@pytest.mark.asyncio
async def test_run_history_trimmed_to_max(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.ring(tmp_path / 'current')


@pytest.mark.asyncio
async def test_run_history_persisted_to_disk(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.completion(tmp_path / 'current')


@pytest.mark.asyncio
async def test_running_service_honors_external_disable(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy', running=True)
    await cases.external_scheduler(tmp_path / 'current', disable=True)


def test_start_reloads_recomputes_and_persists_as_one_locked_rmw(tmp_path, monkeypatch) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    asyncio.run(cases.definition_fence(tmp_path / 'legacy', running=True))
    cases.concurrent_cas(tmp_path / 'current')
