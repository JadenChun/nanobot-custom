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
)
from tests.cron import p6_compat_fixtures as cases
from tests.cron.stage3_fixtures import offline_denial  # noqa: F401


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
    before = deepcopy(original)
    from nanobot.cron.migration import prepare_staging_job
    prepared_result = prepare_staging_job(original)
    prepared = prepared_result.job
    assert original == before
    assert prepared.state == original.state and prepared.updated_at_ms == 9 and prepared.enabled
    assert prepared.created_at_ms == 7 and prepared.delete_after_run
    assert prepared.state.next_run_at_ms == 1 and prepared.state.last_error == 'historical'
    if payload.session_key == 'partial':
        assert prepared_result.execution_mode == 'unresolved'
        assert prepared_result.diagnostic == 'partial explicit cron binding requires reconciliation'
        assert prepared.payload == payload
    elif payload.to == '43':
        assert prepared_result.execution_mode == 'unresolved'
        assert prepared_result.diagnostic == 'legacy route conflicts with explicit cron origin'
        assert prepared.payload == payload
    else:
        assert prepared_result.diagnostic is None and prepared_result.execution_mode == 'legacy_fresh'
        assert prepared.payload.session_key == 'cron:fixture'
        assert prepared.payload.origin_channel == 'cli' and prepared.payload.origin_chat_id == 'direct'
    callback = AsyncMock()
    raw = json.dumps({'jobs': [asdict(original)]}).encode()
    path = tmp_path / 'ordinary' / 'jobs.json'
    path.parent.mkdir(mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)
    service = CronService(path, on_job=callback)
    with pytest.raises(cases.CronDefinitionServiceError, match='explicit stopped-writer conversion'):
        service.get_job(original.id)
    assert path.read_bytes() == raw
    root, result, store = cases.convert(tmp_path / 'explicit', raw=raw)
    assert result.snapshot_converted
    loaded = cases.converted_view(root, store)[1][0]
    assert loaded.enabled and loaded.state.run_history == [] and loaded.updated_at_ms == 0
    assert loaded.state.next_run_at_ms is None and loaded.state.last_error is None
    assert await service.run_job(original.id, force=True) is False
    await service._on_timer()
    callback.assert_not_awaited()
    assert path.read_bytes() == raw and not service._run_records_dir.exists()


async def test_complete_modern_job_still_executes(tmp_path):
    original = fixture_job(CronPayload(session_key='s', origin_channel='telegram', origin_chat_id='42'))
    raw = json.dumps({'jobs': [asdict(original)]}).encode()
    root, result, store = cases.convert(tmp_path / 'default', raw=raw)
    assert result.snapshot_converted
    callback = AsyncMock(return_value='done')
    service = CronService(root / 'jobs.json', action_store=store, on_job=callback)
    job = service.get_job(original.id)
    assert job.payload == original.payload and job.delete_after_run and job.enabled
    assert job.state.run_history == [] and job.state.next_run_at_ms is None
    before = cases.candidate(service)
    try:
        await service.start()
        assert await service.run_job(job.id, force=True) is False
        await service._on_timer()
        with pytest.raises(cases.CronJobSkippedError, match='adapter is not installed'):
            await service._execute_job(job)
        service._save_store()
        assert service._timer_task is None and service._get_next_wake_ms() is None
        callback.assert_not_awaited()
        assert cases.candidate(service) == before and service.get_job(job.id) == job
        assert CronService(service.store_path, action_store=store).get_job(job.id) == job
        assert (root / 'jobs.json').read_bytes() == raw
        assert not service._run_records_dir.exists() and service.occurrence_store is None
    finally:
        service.stop()
    await cases.completion(tmp_path / 'supported')
    await cases.unsupported_execution(tmp_path / 'unsupported')


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
    with pytest.raises(cases.CronJobSkippedError, match='adapter is not installed'):
        await service._execute_job(job)
    assert job == before and service.admission_diagnostic(job)
    callback.assert_not_awaited()
    await cases.completion(tmp_path / 'positive')


async def test_mixed_due_jobs_do_not_starve_system_jobs(tmp_path):
    service = CronService(tmp_path / 'default' / 'jobs.json', on_job=AsyncMock())
    blocked = service.add_job('blocked', CronSchedule(kind='every', every_ms=100), 'hello', session_key='partial')
    system = service.register_system_job(CronJob(id='system', name='system', schedule=CronSchedule(kind='every', every_ms=100), payload=CronPayload(kind='system_event')))
    before = cases.candidate(service)
    await service._on_timer()
    assert await service.run_job(blocked.id, force=True) is False
    assert await service.run_job(system.id, force=True) is False
    assert cases.candidate(service) == before
    service.on_job.assert_not_awaited()
    await cases.reentrant(tmp_path / 'healthy', scheduled=True, two=True)


async def test_action_update_and_enable_preserve_admission_fence(tmp_path):
    service = CronService(tmp_path / 'jobs.json', on_job=AsyncMock())
    job = service.add_job('legacy', CronSchedule(kind='every', every_ms=1000), 'hello')
    mode = job.payload.execution_mode
    before = cases.candidate(service)
    with pytest.raises(cases.CronDefinitionTokenRequiredError):
        service.update_job(job.id, message='denied')
    assert cases.candidate(service) == before
    current = service.update_job(job.id, message='updated', **cases.tokens(job))
    assert current is not None
    before = cases.candidate(service)
    with pytest.raises(cases.DefinitionStoreConflictError):
        service.enable_job(job.id, False, **cases.tokens(job))
    assert cases.candidate(service) == before
    current = service.enable_job(job.id, False, **cases.tokens(current))
    current = service.enable_job(job.id, True, **cases.tokens(current))
    assert current.enabled and current.payload.message == 'updated'
    assert current.payload.execution_mode == mode and service.admission_diagnostic(current)
    assert await service.run_job(job.id, force=True) is False
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
    # Verify the accepted default-composition boundary without starting the gateway.
    import ast
    import asyncio
    from pathlib import Path

    from tests.migration.test_p5_stage3_composition import composition
    source = Path(__file__).parents[2] / 'nanobot/cli/gateway_runtime.py'
    tree = ast.parse(source.read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'on_cron_job' for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'register_system_job' for node in ast.walk(tree))
    graph, _item, _clock, calls, channels = cases.runtime(tmp_path / 'injected')
    try:
        with pytest.raises(ValueError):
            graph.construct_internal_controller(name)
        assert calls == [] and channels.calls == []
    finally:
        await cases.finish(graph)
    base = tmp_path / 'default'
    base.mkdir()
    value, _item, _clock, events, channels = composition(base)
    try:
        handle = await asyncio.wait_for(value.acquire(), 2)
        assert handle.cron_runtime is None and handle.cron.on_job is None
        assert handle.cron.occurrence_store is None
        assert not (base / 'runtime-data' / 'cron-runtime').exists()
        assert events == ['channels-acquire'] and channels.calls == []
    finally:
        await asyncio.wait_for(value.aclose(), 2)
    assert not any(path.name == 'callbacks' for path in base.rglob('*'))
