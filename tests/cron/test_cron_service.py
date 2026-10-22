import asyncio
import json
import time

import pytest

from nanobot.cron.service import CronJobSkippedError, CronService
from nanobot.cron.types import CronJob, CronPayload, CronSchedule
from nanobot.runtime_context import RUNTIME_CONTEXT_INPUT_META
from tests.cron import p6_compat_fixtures as cases
from tests.cron.stage3_fixtures import offline_denial  # noqa: F401


async def _wait_until(predicate, *, timeout: float = 1.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    assert predicate()


def _bound_chat(chat_id: str = "chat-1") -> dict[str, str]:
    return {
        "session_key": f"websocket:{chat_id}",
        "origin_channel": "websocket",
        "origin_chat_id": chat_id,
    }


def test_load_jobs_accepts_snake_case_schedule_and_run_history(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.alias_conversion(tmp_path)


def test_cron_job_from_dict_rejects_malformed_run_history() -> None:
    with pytest.raises(TypeError):
        CronJob.from_dict(
            {
                "id": "j1",
                "name": "t",
                "state": {"run_history": [None]},
            }
        )


def test_load_jobs_coerces_string_schedule_and_state_ms(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.scalar_siblings(tmp_path)


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


@pytest.mark.parametrize("expr", [None, "", "   "])
def test_add_job_rejects_missing_cron_expression(tmp_path, expr: str | None) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    with pytest.raises(ValueError, match="requires a non-empty 'expr'"):
        service.add_job(
            name="missing expression",
            schedule=CronSchedule(kind="cron", expr=expr),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_add_job_rejects_invalid_cron_expression_before_persisting(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    with pytest.raises(ValueError, match="invalid cron expression"):
        service.add_job(
            name="bad expression",
            schedule=CronSchedule(kind="cron", expr="not a cron expression"),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_write_run_record_uses_cron_runs_dir(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service = CronService(tmp_path / 'legacy' / 'jobs.json')
    with pytest.raises(CronJobSkippedError):
        service.write_run_record('job:1', {'status': 'queued'})
    assert not service._run_records_dir.exists()
    asyncio.run(cases.completion(tmp_path / 'current'))


@pytest.mark.asyncio
async def test_unbound_agent_jobs_are_disabled_on_add(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path)
    job = CronService(tmp_path / 'jobs.json').list_jobs(include_disabled=True)[0]
    assert job.enabled and job.state.last_error is None


def test_unbound_agent_jobs_are_disabled_on_load(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.unbound_conversion(tmp_path)


def test_add_job_migrates_legacy_delivery_context(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.route_roundtrip(tmp_path)


def test_load_store_migrates_legacy_delivery_context(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.route_roundtrip(tmp_path, legacy=True)


def test_load_store_disables_malformed_legacy_payload(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.unbound_conversion(tmp_path, malformed=True)


def test_list_bound_agent_jobs_includes_migrated_legacy_delivery_payloads(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service = CronService(tmp_path / 'jobs.json')
    schedule = CronSchedule(kind='every', every_ms=60000)
    one = service.add_job('bound', schedule, 'hello', **_bound_chat())
    two = service.add_job('legacy route', schedule, 'hello', deliver=True, channel='websocket', to='chat-1', session_key='websocket:chat-1')
    service.add_job('other', schedule, 'other', **_bound_chat('other'))
    assert {job.id for job in service.list_bound_cron_jobs_for_session('websocket:chat-1')} == {one.id, two.id}
    disabled = service.enable_job(two.id, False, **cases.tokens(two))
    assert not disabled.enabled
    assert {job.id for job in service.list_bound_cron_jobs_for_session('websocket:chat-1', include_disabled=False)} == {one.id}
    assert {job.id for job in service.list_bound_cron_jobs_for_session('websocket:chat-1')} == {one.id, two.id}
    assert service.get_job(two.id).payload.channel == 'websocket'


def test_add_job_preserves_origin_delivery_context(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.route_roundtrip(tmp_path)


@pytest.mark.asyncio
async def test_start_heals_runtime_context_from_pending_external_add(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    job = cases.legacy_job()
    job['payload']['originMetadata'][RUNTIME_CONTEXT_INPUT_META] = [{'source': 'webui_quote', 'content': 'historical'}]
    actions = (json.dumps({'action': 'add', 'params': job}) + '\n').encode()
    root, result, store = cases.convert(tmp_path, job, actions=actions)
    assert result.journal_converted and not (root / 'action.jsonl').exists()
    service, jobs = cases.converted_view(root, store)
    assert jobs[0].payload.origin_metadata == job['payload']['originMetadata']
    before = cases.candidate(service)
    await service.start()
    service.stop()
    assert cases.candidate(service) == before and service._timer_task is None


@pytest.mark.asyncio
async def test_channel_meta_and_session_key_survive_store_reload(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service, _job = cases.route_roundtrip(tmp_path)
    before = cases.candidate(service)
    await service.start()
    service.stop()
    assert cases.candidate(service) == before


@pytest.mark.asyncio
async def test_execute_job_records_run_history(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.completion(tmp_path / 'current')


@pytest.mark.asyncio
async def test_run_history_records_errors(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.completion(tmp_path / 'represented', status='error', error='boom')
    await cases.compute_failure(tmp_path / 'computation')


@pytest.mark.asyncio
async def test_run_history_records_skipped_jobs(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.completion(tmp_path / 'current', status='skipped', error='missing session binding')


@pytest.mark.asyncio
async def test_run_history_records_job_cancellation(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.compute_failure(tmp_path / 'current', cancel=True)


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
async def test_run_job_disabled_does_not_flip_running_state(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service = CronService(tmp_path / 'legacy' / 'jobs.json')
    job = service.add_job('disabled', CronSchedule(kind='every', every_ms=100), 'hello')
    service.enable_job(job.id, False, **cases.tokens(job))
    assert await service.run_job(job.id) is False and not service._running
    assert service.list_jobs() == [] and not service.list_jobs(include_disabled=True)[0].enabled
    await cases.external_scheduler(tmp_path / 'current', disable=True)


@pytest.mark.asyncio
async def test_run_job_preserves_running_service_state(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy', running=True)
    await cases.completion(tmp_path / 'current')


@pytest.mark.asyncio
async def test_manual_run_persists_completion_when_callback_lists_jobs(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.reentrant(tmp_path / 'current')


@pytest.mark.asyncio
async def test_overlapping_manual_runs_preserve_stopped_service_state(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.overlap(tmp_path / 'current')


@pytest.mark.asyncio
async def test_manual_run_does_not_restart_service_stopped_during_execution(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy', running=True)
    await cases.overlap(tmp_path / 'current', closing=True)


@pytest.mark.asyncio
async def test_running_service_honors_external_disable(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy', running=True)
    await cases.external_scheduler(tmp_path / 'current', disable=True)


def test_remove_job_refuses_system_jobs(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service = CronService(tmp_path / 'jobs.json')
    job = service.register_system_job(CronJob(id='dream', name='dream', schedule=CronSchedule(kind='every', every_ms=60000), payload=CronPayload(kind='system_event')))
    before = cases.candidate(service)
    with pytest.raises(cases.CronDefinitionTokenRequiredError):
        service.remove_job(job.id)
    assert service.remove_job(job.id, **cases.tokens(job)) == 'protected'
    assert cases.candidate(service) == before and service.get_job(job.id) == job


def test_remove_system_job_retires_persisted_system_job(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service = CronService(tmp_path / 'jobs.json')
    job = service.register_system_job(CronJob(id='heartbeat', name='heartbeat', schedule=CronSchedule(kind='every', every_ms=60000), payload=CronPayload(kind='system_event')))
    before = cases.candidate(service)
    assert service.remove_system_job(job.id) is False and cases.candidate(service) == before
    assert service.remove_system_job(job.id, **cases.tokens(job)) is True
    assert CronService(service.store_path).get_job(job.id) is None
    assert service.remove_system_job(job.id) is False
    other = service.register_system_job(CronJob(id='dream', name='dream', schedule=CronSchedule(kind='every', every_ms=60000), payload=CronPayload(kind='system_event')))
    assert service.remove_job(other.id, **cases.tokens(other)) == 'protected'


def test_remove_system_job_without_store_file(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path)

    assert service.remove_system_job("heartbeat") is False
    assert not store_path.exists()


@pytest.mark.asyncio
async def test_start_server_not_jobs(tmp_path):
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy', running=True)
    await cases.external_scheduler(tmp_path / 'current')


@pytest.mark.asyncio
async def test_subsecond_job_not_delayed_to_one_second(tmp_path):
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.reentrant(tmp_path / 'current', scheduled=True)


@pytest.mark.asyncio
async def test_running_service_picks_up_external_add(tmp_path):
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy', running=True)
    await cases.external_scheduler(tmp_path / 'current')


@pytest.mark.asyncio
async def test_add_job_during_jobs_exec(tmp_path):
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.reentrant(tmp_path / 'current', add=True)


@pytest.mark.asyncio
async def test_external_update_preserves_run_history_records(tmp_path):
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.history_mutation(tmp_path)


def test_stale_instance_remove_preserves_external_add(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.stale_remove(tmp_path)


# ── timer race regression tests ──


@pytest.mark.asyncio
async def test_save_store_failure_retries_without_replaying_job(tmp_path, monkeypatch):
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.projection_retry(tmp_path / 'current', monkeypatch)


@pytest.mark.asyncio
async def test_timer_execution_is_not_rolled_back_by_list_jobs_reload(tmp_path):
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.reentrant(tmp_path / 'current', scheduled=True)


# ── update_job tests ──


def test_update_job_changes_name(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.changed_definition(tmp_path, 'name')


def test_update_job_changes_schedule(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.changed_definition(tmp_path, 'schedule')


def test_update_job_changes_message(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.changed_definition(tmp_path, 'message')


def test_update_job_changes_cron_expression(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.changed_definition(tmp_path, 'cron')


def test_update_job_not_found(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    result = service.update_job("nonexistent", name="x")
    assert result == "not_found"


def test_update_job_rejects_system_job(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service = CronService(tmp_path / 'jobs.json')
    job = service.register_system_job(CronJob(id='dream', name='dream', schedule=CronSchedule(kind='every', every_ms=60000), payload=CronPayload(kind='system_event')))
    before = cases.candidate(service)
    with pytest.raises(cases.CronDefinitionTokenRequiredError):
        service.update_job(job.id)
    assert service.update_job(job.id, **cases.tokens(job)) == 'protected'
    assert cases.candidate(service) == before and service.get_job(job.id) == job


def test_update_job_validates_schedule(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service = CronService(tmp_path / 'jobs.json')
    job = service.add_job('validate', CronSchedule(kind='every', every_ms=60000), 'hello')
    before = cases.candidate(service)
    with pytest.raises(ValueError, match='unknown timezone'):
        service.update_job(job.id, schedule=CronSchedule(kind='cron', expr='0 9 * * *', tz='Bad/Zone'), **cases.tokens(job))
    assert cases.candidate(service) == before


@pytest.mark.asyncio
async def test_update_job_preserves_run_history(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.history_mutation(tmp_path)


def test_update_job_offline_writes_action(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    service, old, new = cases.changed_definition(tmp_path, 'name')
    assert not service._running and service.get_job(old.id).name == new.name
    assert service.action_store.replay().pending_actions == ()
    assert not (service.store_path.parent / 'action.jsonl').exists()


def test_update_job_migrates_legacy_delivery_target(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.route_roundtrip(tmp_path, update=True)


@pytest.mark.asyncio
async def test_list_jobs_during_on_job_does_not_cause_stale_reload(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    await cases.definition_fence(tmp_path / 'legacy')
    await cases.reentrant(tmp_path / 'current', scheduled=True, two=True)


def test_load_jobs_accepts_null_run_history_ms(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.scalar_siblings(tmp_path, null=True)


def test_load_jobs_skips_null_run_history_elements(tmp_path) -> None:
    """Retained invariant at its current owner; see the accepted P6 mapping."""
    cases.history_elements(tmp_path)
