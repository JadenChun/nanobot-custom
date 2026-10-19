import asyncio

import pytest

from nanobot.agent.turn import TurnResult
from nanobot.cron.runtime import RuntimeOwnedWorkError
from tests.cron.stage3_fixtures import offline_denial  # noqa: F401
from tests.migration.test_p5_stage3_composition import composition


async def test_repeated_caller_cancellation_does_not_abandon_drain(tmp_path):
    begun, finish = asyncio.Event(), asyncio.Event()
    events = []

    async def compute(request):
        events.append("begin")
        begun.set()
        await finish.wait()
        events.append("effect-finish")
        return TurnResult(run_id=request.run_id, content="done")

    value, item, clock, events, channels = composition(tmp_path, injection=True, events=events, compute=compute)
    handle = await value.acquire()
    graph = handle.cron_runtime
    graph.injection = type(graph.injection)(clock, lambda *_: True, 1)
    clock.now = 1100
    wake = asyncio.create_task(graph.scheduler.wake_once())
    await begun.wait()
    close = asyncio.create_task(value.aclose())
    await asyncio.sleep(0)
    close.cancel()
    await asyncio.sleep(0)
    close.cancel()
    assert not graph._accepting and "channels-close" not in events
    assert graph.scheduler._lease is not None
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await close
    await wake
    assert events.index("effect-finish") < events.index("channels-close")
    assert graph.scheduler._lease is None


async def test_resistant_effect_retains_resources_and_explicit_close_retry(tmp_path):
    begun, finish = asyncio.Event(), asyncio.Event()
    events = []

    async def compute(request):
        begun.set()
        while not finish.is_set():
            try:
                await finish.wait()
            except asyncio.CancelledError:
                continue
        return TurnResult(run_id=request.run_id, content="done")

    value, item, clock, events, channels = composition(tmp_path, injection=True, events=events, compute=compute)
    handle = await value.acquire()
    clock.now = 1100
    wake = asyncio.create_task(handle.cron_runtime.scheduler.wake_once())
    await begun.wait()
    with pytest.raises(ExceptionGroup) as failure:
        await value.aclose()
    assert isinstance(failure.value.exceptions[0], RuntimeOwnedWorkError)
    assert "channels-close" not in events and handle.cron_runtime.scheduler._lease is not None
    finish.set()
    await wake
    await value.aclose()
    assert "channels-close" in events and handle.cron_runtime.scheduler._lease is None


async def test_prebegin_lease_loss_fences_compute_and_manual_allocation(tmp_path, monkeypatch):
    from nanobot.cron.scheduler_store import SchedulerConflictError

    value, item, clock, events, channels = composition(tmp_path, injection=True)
    handle = await value.acquire()
    graph = handle.cron_runtime
    clock.now = 1100
    real = graph._dispatch

    async def lose(claim, prebegin):
        graph.scheduler._lease._active = False
        return await real(claim, prebegin)

    monkeypatch.setattr(graph.scheduler, "_dispatch", lose)
    with pytest.raises(SchedulerConflictError):
        await graph.scheduler.wake_once()
    with pytest.raises(SchedulerConflictError):
        graph.authorize_manual("now", item.job_id, item.job_instance_id, item.definition_revision)
    assert "compute" not in events and channels.calls == [] and graph.store.read()["requests"] == []
    graph.scheduler._lease._active = True
    # Owned task failure is explicit, but joined work permits shared closure.
    with pytest.raises(ExceptionGroup):
        await value.aclose()
    assert "channels-close" in events


async def test_partial_graph_acquisition_unwinds_lease_before_channels(tmp_path, monkeypatch):
    from nanobot.cron.runtime import CronRuntime

    value, item, clock, events, channels = composition(tmp_path, injection=True)
    original = CronRuntime.acquire
    graphs = []

    def fail_after_acquisition(graph):
        original(graph)
        graphs.append(graph)
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(CronRuntime, "acquire", fail_after_acquisition)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        await value.acquire()
    assert len(graphs) == 1 and graphs[0].scheduler._lease is None
    assert not graphs[0]._accepting and channels.calls == []
    assert events == ["channels-acquire", "channels-close", "agent-close", "mcp-close"]
