"""Migration must retain unresolved schedule state while fencing execution."""
import json
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import AsyncMock

import pytest

from nanobot.cron.service import CronService
from nanobot.cron.types import (
    CronDestination,
    CronJob,
    CronJobState,
    CronPayload,
    CronSchedule,
    CronStore,
)


def fixture_job(payload):
    return CronJob(id='fixture', name='fixture', payload=payload,
                   schedule=CronSchedule(kind='at', at_ms=1), delete_after_run=True,
                   created_at_ms=7, updated_at_ms=9,
                   state=CronJobState(next_run_at_ms=1, last_error='historical'))


@pytest.mark.parametrize('payload', [CronPayload(), CronPayload(session_key='partial'),
    CronPayload(session_key='s', origin_channel='telegram', origin_chat_id='42',
                channel='telegram', to='43')])
async def test_inadmissible_jobs_keep_state_and_never_execute(tmp_path, payload):
    original = fixture_job(payload)
    path = tmp_path / 'jobs.json'
    path.write_text(json.dumps({'jobs': [asdict(original)]}))
    callback = AsyncMock()
    service = CronService(path, on_job=callback)
    job = service.get_job(original.id)
    assert job.state == original.state and job.updated_at_ms == 9 and job.enabled
    assert service.admission_diagnostic(job)
    before = deepcopy(job)
    service._recompute_next_runs()
    assert job == before
    assert service._get_next_wake_ms() is None
    assert await service.run_job(job.id, force=True) is False
    await service._on_timer()
    after = service.get_job(job.id)
    assert after == before
    callback.assert_not_awaited()
    assert not service._run_records_dir.exists()


async def test_complete_modern_job_still_executes(tmp_path):
    callback = AsyncMock(return_value='done')
    service = CronService(tmp_path / 'jobs.json', on_job=callback)
    job = fixture_job(CronPayload(session_key='s', origin_channel='telegram', origin_chat_id='42'))
    service._store = CronStore(jobs=[job])
    service._save_store()
    assert service.admission_diagnostic(job) is None
    assert await service.run_job(job.id, force=True)
    callback.assert_awaited_once()
    assert service.get_job(job.id) is None


@pytest.mark.parametrize('changes', [
    {'execution_mode': 'legacy_fresh'}, {'execution_mode': 'unsupported'},
    {'deliver': True}, {'channel': 'telegram', 'to': '42'},
    {'channel_meta': {'topic': 42}},
    {'additional_destinations': [CronDestination('telegram', '43')]},
])
async def test_complete_binding_does_not_bypass_execution_admission(tmp_path, changes):
    payload = CronPayload(session_key='s', origin_channel='telegram', origin_chat_id='42')
    for key, value in changes.items():
        setattr(payload, key, value)
    job = fixture_job(payload)
    before = deepcopy(job)
    callback = AsyncMock()
    service = CronService(tmp_path / 'jobs.json', on_job=callback)
    await service._execute_job(job)
    assert service.admission_diagnostic(job)
    assert job == before
    callback.assert_not_awaited()


async def test_mixed_due_jobs_do_not_starve_system_jobs(tmp_path):
    blocked = fixture_job(CronPayload(session_key='partial'))
    valid = fixture_job(CronPayload(kind='system_event'))
    valid.id = 'system'
    before = deepcopy(blocked)
    callback = AsyncMock()
    service = CronService(tmp_path / 'jobs.json', on_job=callback)
    service._store = CronStore(jobs=[blocked, valid])
    service._save_store()
    await service._on_timer()
    callback.assert_awaited_once()
    assert callback.await_args.args[0].id == 'system'
    assert service.get_job(blocked.id) == before
    assert service._get_next_wake_ms() is None


async def test_action_update_and_enable_preserve_admission_fence(tmp_path):
    service = CronService(tmp_path / 'jobs.json', on_job=AsyncMock())
    job = service.add_job('legacy', CronSchedule(kind='every', every_ms=1000), 'hello')
    assert service.get_job(job.id).payload.execution_mode == 'legacy_fresh'
    service.update_job(job.id, message='updated')
    service.enable_job(job.id, False)
    service.enable_job(job.id, True)
    loaded = service.get_job(job.id)
    assert loaded.enabled and loaded.payload.message == 'updated'
    assert loaded.payload.execution_mode == 'legacy_fresh'
    assert service.admission_diagnostic(loaded)
    assert not await service.run_job(job.id, force=True)
    service.on_job.assert_not_awaited()


async def test_bound_runner_rejects_legacy_before_any_side_effect(tmp_path):
    from nanobot.cron.bound_runner import run_bound_cron_job
    job = fixture_job(CronPayload())
    with pytest.raises(ValueError, match='adapter is not installed'):
        await run_bound_cron_job(job, agent=None, cron=None)


async def test_webui_rejects_fenced_run_with_diagnostic(tmp_path):
    from types import SimpleNamespace

    from websockets.datastructures import Headers
    from websockets.http11 import Request

    from nanobot.webui.ws_http import GatewayHTTPHandler
    service = CronService(tmp_path / 'jobs.json', on_job=AsyncMock())
    job = service.add_job('legacy', CronSchedule(kind='every', every_ms=1000), 'hello')
    handler = SimpleNamespace(check_api_token=lambda request: True,
                              cron_service=service, local_trigger_store=None)
    response = await GatewayHTTPHandler._handle_webui_automation_action(
        handler, Request(f'/api/automations/run?id={job.id}', Headers()), 'run')
    assert response.status_code == 409
    assert b'adapter is not installed' in response.body
    service.on_job.assert_not_awaited()


@pytest.mark.parametrize('name', ['dream', 'heartbeat'])
async def test_gateway_callback_checks_admission_before_named_internal_jobs(tmp_path, name):
    # Extract the real nested callback without starting the guarded gateway.
    import ast
    from pathlib import Path
    from typing import Any

    from nanobot.cron.service import CronJobSkippedError
    source = Path(__file__).parents[2] / 'nanobot/cli/gateway_runtime.py'
    tree = ast.parse(source.read_text())
    callback = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == 'on_cron_job')
    module = ast.Module(body=[callback], type_ignores=[])
    service = CronService(tmp_path / 'jobs.json')
    namespace = {'CronJob': CronJob, 'Any': Any, 'cron': service,
                 'CronJobSkippedError': CronJobSkippedError}
    exec(compile(module, str(source), 'exec'), namespace)
    job = fixture_job(CronPayload())
    job.name = name
    # No agent/model/delivery objects exist: touching one before admission fails.
    with pytest.raises(CronJobSkippedError, match='adapter is not installed'):
        await namespace['on_cron_job'](job)
