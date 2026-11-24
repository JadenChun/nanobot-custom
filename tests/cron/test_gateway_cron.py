"""Explicit gateway scheduling admission and lifetime ownership, without networking."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from kohaku.agent.turn import TurnResult
from kohaku.config.schema import GatewayCronConfig
from kohaku.cron.gateway import GatewayCron
from kohaku.cron.service import CronService
from kohaku.cron.types import CronSchedule
from kohaku.session.manager import SessionManager
from tests.cron.stage3_fixtures import FakeChannels, definitions


def _gateway(tmp_path, *, now=1000):
    store, item, clock = definitions(tmp_path)
    clock.now = now
    service = CronService(tmp_path / "definitions" / "jobs.json", definition_store=store)
    calls = []

    async def compute(request):
        calls.append(request)
        return TurnResult(run_id=request.run_id, content="result")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    channels = FakeChannels()
    channels.receipt_transport_ready = False
    gateway = GatewayCron(GatewayCronConfig(enabled=True), service,
                          SimpleNamespace(execute_silent_cron_turn=compute),
                          SessionManager(workspace, sessions_root=tmp_path / "sessions"),
                          channels, workspace, tmp_path / "instance", clock=clock)
    return gateway, clock, calls, channels, item


async def _until(condition):
    async with asyncio.timeout(3):
        while not condition():
            await asyncio.sleep(0.001)


async def test_waits_for_authenticated_transport_then_runs_and_drains_once(tmp_path):
    gateway, clock, calls, channels, _item = _gateway(tmp_path)
    owner = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.01)
    assert gateway.status()["state"] == "waiting_for_channels"
    assert gateway.runtime._accepting is False
    assert not calls and not channels.calls
    channels.receipt_transport_ready = True
    await _until(lambda: gateway.runtime._accepting)
    assert gateway.service.status()["enabled"] is True
    clock.now = 1100
    await _until(lambda: channels.calls == ["A", "B"])
    await gateway.aclose()
    await asyncio.gather(owner, return_exceptions=True)
    assert len(calls) == 1 and gateway.status()["enabled"] is False
    # A new lease reconciles completed work instead of replaying its effects.
    gateway.runtime.scheduler.store.acquire_lease().release()


async def test_new_job_is_fresh_and_does_not_drop_already_due_work(tmp_path):
    gateway, clock, calls, channels, _item = _gateway(tmp_path)
    gateway.runtime.acquire()
    job = gateway.service.add_job("new", CronSchedule(kind="every", every_ms=100),
        "new work", deliver=True, channel="telegram", to="C", session_key="telegram:C",
        origin_metadata={"message_thread_id": 42})
    assert job.payload.execution_mode == "legacy_fresh"
    assert job.payload.session_key == f"cron:{job.id}"
    assert job.payload.channel_meta == {"message_thread_id": 42}
    # Keep this test's fake channel topic-free; the topic receipt path has its
    # own transport tests. This second new job proves both cursors can fire.
    gateway.service.add_job("other", CronSchedule(kind="every", every_ms=100),
        "another", deliver=True, channel="telegram", to="D")
    clock.now = 1100
    await gateway.runtime.scheduler.wake_once()
    await gateway.runtime.scheduler.wake_once()
    await gateway.runtime.scheduler.wake_once()
    assert len(calls) == 3
    assert channels.calls.count("A") == 1 and channels.calls.count("D") == 1
    await gateway.aclose()


async def test_startup_classifies_missed_fires_without_replaying(tmp_path):
    gateway, clock, calls, channels, _item = _gateway(tmp_path, now=1500)
    gateway.runtime.acquire()
    await gateway.runtime.scheduler.wake_once()
    assert calls == channels.calls == []
    clock.now = 1600
    await gateway.runtime.scheduler.wake_once()
    assert len(calls) == 1
    await gateway.aclose()


async def test_timer_error_reaches_gateway_owner_and_releases_lease(tmp_path, monkeypatch):
    gateway, _clock, _calls, channels, _item = _gateway(tmp_path)
    channels.receipt_transport_ready = True

    async def failed():
        raise OSError("injected timer failure")

    monkeypatch.setattr(gateway.runtime.scheduler, "wake_once", failed)
    with pytest.raises(OSError, match="timer failure"):
        await gateway.run()
    assert gateway.status()["state"] == "failed"
    await gateway.aclose()
    gateway.runtime.scheduler.store.acquire_lease().release()


def test_unmigrated_jobs_refused_before_runtime_admission(tmp_path):
    service = CronService(tmp_path / "definitions" / "jobs.json")
    service.add_job("unbound", CronSchedule(kind="every", every_ms=100), "work")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="fresh-turn migration"):
        GatewayCron(GatewayCronConfig(enabled=True), service, SimpleNamespace(),
            SessionManager(workspace, sessions_root=tmp_path / "sessions"), FakeChannels(),
            workspace, tmp_path / "instance")
    assert not (tmp_path / "instance" / "cron-runtime").exists()


async def test_run_now_and_timer_use_the_same_job_preset(tmp_path):
    gateway, clock, calls, channels, item = _gateway(tmp_path)
    job = gateway.service.update_job(
        item.job_id, model_preset="reports", job_instance_id=item.job_instance_id,
        definition_revision=item.definition_revision,
    )
    channels.receipt_transport_ready = True
    owner = asyncio.create_task(gateway.run())
    try:
        await _until(lambda: gateway.status()["state"] == "running")
        assert await gateway.service.run_job(job.id, request_id="user-run-now")
        await _until(lambda: len(channels.calls) == 2)
        clock.now = 1100
        await _until(lambda: len(channels.calls) == 4)
        assert [request.cron_model_preset for request in calls] == ["reports", "reports"]
        assert len({request.run_id for request in calls}) == 2
    finally:
        await gateway.aclose()
        with suppress(asyncio.CancelledError):
            await owner
