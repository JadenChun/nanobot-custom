from dataclasses import asdict

import pytest

from nanobot.cron.types import CronDestination, CronJob


def test_staging_fields_survive_store_and_action_decoding():
    data = {'id': 'job', 'name': 'report', 'enabled': False,
            'schedule': {'kind': 'cron', 'expr': '0 9 * * *', 'tz': 'Asia/Kuala_Lumpur'},
            'payload': {'deliver': True, 'channel': 'telegram', 'to': '42',
                        'additionalDestinations': [{'channel': 'telegram', 'to': '43'},
                                                   {'channel': 'telegram', 'to': '42'}]},
            'state': {'nextRunAtMs': 123, 'lastRunStatus': 'completed',
                      'runHistory': [{'runAtMs': 100, 'status': 'ok', 'runId': 'run',
                                      'runStatus': 'completed', 'occurrenceId': 'occurrence'}]}}
    job = CronJob.from_store_dict(data)
    raw = asdict(job)
    copy = CronJob.from_dict(raw)
    assert isinstance(raw["payload"], dict)
    assert isinstance(raw["state"]["run_history"][0], dict)
    assert copy == job
    assert copy.payload.delivery_destinations() == [CronDestination('telegram', '42'), CronDestination('telegram', '43')]
    assert copy.state.run_history[0].occurrence_id == 'occurrence'
    assert copy.state.last_run_status == 'completed'
    assert copy.state.next_run_at_ms == 123 and not copy.enabled
    assert copy.schedule.tz == 'Asia/Kuala_Lumpur'
    copy.payload.deliver = False
    assert copy.payload.delivery_destinations() == []


@pytest.mark.parametrize("execution_mode", ["session_bound", "legacy_fresh"])
def test_service_serializer_preserves_identifiers_and_destinations(tmp_path, execution_mode):
    import json

    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJobState, CronPayload, CronRunRecord, CronSchedule, CronStore
    job = CronJob(id='job', name='fixture', schedule=CronSchedule(kind='every', every_ms=1000),
                  payload=CronPayload(kind='system_event', deliver=False, execution_mode=execution_mode,
                                      additional_destinations=[CronDestination('telegram', '42')]),
                  state=CronJobState(last_run_status='completed', run_history=[
                      CronRunRecord(run_at_ms=1, status='ok', run_id='run',
                                    run_status='completed', occurrence_id='occurrence')]))
    service = CronService(tmp_path / 'jobs.json')
    service._store = CronStore(jobs=[job])
    service._save_store()
    stored = json.loads(service.store_path.read_text())['jobs'][0]
    assert CronJob.from_store_dict(stored) == job


def test_staging_preparation_is_idempotent_and_preserves_delivery_and_state():
    from nanobot.cron.migration import prepare_staging_job
    data = {'id': 'job', 'name': 'fixture', 'enabled': False, 'createdAtMs': 7, 'updatedAtMs': 9,
            'schedule': {'kind': 'every', 'everyMs': 1000},
            'payload': {'deliver': False, 'additionalDestinations': [{'channel': 'telegram', 'to': '42'}]},
            'state': {'nextRunAtMs': 123, 'lastError': 'historical error'}}
    original = CronJob.from_store_dict(data)
    once = prepare_staging_job(original)
    twice = prepare_staging_job(once.job)
    assert once == twice and once.diagnostic is None
    assert once.execution_mode == 'legacy_fresh'
    assert once.job.payload.session_key == 'cron:job'
    assert (once.job.payload.origin_channel, once.job.payload.origin_chat_id) == ('cli', 'direct')
    assert once.job.state == original.state
    assert once.job.updated_at_ms == 9 and not once.job.enabled
    assert not once.job.payload.deliver
    assert original.payload.session_key is None


def test_partial_binding_is_preserved_with_admission_diagnostic():
    from nanobot.cron.migration import prepare_staging_job
    from nanobot.cron.types import CronPayload
    job = CronJob(id='job', name='fixture', payload=CronPayload(session_key='telegram:42', channel='telegram', to='43'))
    result = prepare_staging_job(job)
    assert result.job == job
    assert result.execution_mode == 'unresolved' and result.diagnostic


def test_complete_binding_does_not_infer_history_mode_from_session_name():
    from nanobot.cron.migration import prepare_staging_job
    from nanobot.cron.types import CronPayload
    job = CronJob(id='job', name='fixture', payload=CronPayload(
        session_key='cron:job', origin_channel='telegram', origin_chat_id='42'))
    result = prepare_staging_job(job)
    assert result.job == job and result.execution_mode == 'session_bound'


def test_prepared_history_mode_survives_store_and_action_roundtrip():
    from nanobot.cron.migration import prepare_staging_job
    original = CronJob(id='job', name='fixture')
    prepared = prepare_staging_job(original)
    decoded = CronJob.from_dict(asdict(prepared.job))
    assert decoded.payload.execution_mode == 'legacy_fresh'
    assert prepare_staging_job(decoded) == prepared


def test_conflicting_complete_binding_is_preserved_and_nested_data_is_detached():
    from nanobot.cron.migration import prepare_staging_job
    from nanobot.cron.types import CronJobState, CronPayload, CronRunRecord
    job = CronJob(id='job', name='fixture', payload=CronPayload(
        channel='telegram', to='43', session_key='telegram:42',
        origin_channel='telegram', origin_chat_id='42', origin_metadata={'nested': {'value': 1}}),
        state=CronJobState(run_history=[CronRunRecord(run_at_ms=1, status='ok')]))
    result = prepare_staging_job(job)
    assert result.diagnostic and result.job == job
    result.job.payload.origin_metadata['nested']['value'] = 2
    result.job.state.run_history[0].status = 'error'
    assert job.payload.origin_metadata['nested']['value'] == 1
    assert job.state.run_history[0].status == 'ok'


def test_unknown_execution_mode_requires_reconciliation():
    from nanobot.cron.migration import prepare_staging_job
    job = CronJob(id='job', name='fixture')
    job.payload.execution_mode = 'unsupported'
    result = prepare_staging_job(job)
    assert result.job == job and result.diagnostic
