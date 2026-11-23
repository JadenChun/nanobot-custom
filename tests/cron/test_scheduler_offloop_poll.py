"""A01-F03: the durable scheduler poll must not stall the event loop.

Requirements case U1 (req-a38-cron/CASES.md).  The review finding named
``Scheduler.wake_once``: the lease proof, definition replay, fsync state read,
whole-ledger inspection and retained-history reproof all ran synchronously
before the first await on the shared loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kohaku.agent.turn import TurnResult
from kohaku.cron.action_store import CronActionStore
from kohaku.cron.definition_store import (
    CronDefinition,
    CronDefinitionAction,
    CronDefinitionStore,
)
from kohaku.cron.runtime import CronRuntime, CronRuntimeInjection
from kohaku.cron.service import CronService
from kohaku.session.manager import SessionManager
from tests.cron.stage3_fixtures import (  # noqa: F401
    FakeChannels,
    FakeClock,
    offline_denial,
    runtime,
)

PHASE_SECONDS = 0.3
LOOP_BOUND_SECONDS = 0.100
FIRES = (1100, 1200, 1300)


def _job(job_id: str, instance: str, fire_ms: int) -> CronDefinition:
    return CronDefinition.from_definition(
        {
            "id": job_id,
            "name": job_id,
            "enabled": True,
            "schedule": {
                "kind": "at",
                "atMs": fire_ms,
                "everyMs": None,
                "expr": None,
                "tz": None,
            },
            "payload": {
                "kind": "agent_turn",
                "message": "private compute",
                "deliver": True,
                "channel": "telegram",
                "to": "A",
                "additionalDestinations": [{"channel": "telegram", "to": "B"}],
                "channelMeta": {},
                "sessionKey": f"cron:{job_id}",
                "executionMode": "legacy_fresh",
                "originChannel": "telegram",
                "originChatId": "A",
                "originMetadata": {},
            },
            "deleteAfterRun": False,
        },
        job_instance_id=instance,
    )


def _three_job_graph(tmp_path: Path):
    """Build three published one-shot jobs with distinct next-fire times."""
    clock = FakeClock(now=1000)
    definitions = CronDefinitionStore(
        CronActionStore(tmp_path / "definitions", namespace="stage3"),
        commit_clock_ms=clock.wall_time_ms,
    )
    for index, fire in enumerate(FIRES):
        job_id = f"job-{index}"
        definitions.append(
            action_id=f"create-{index}",
            action=CronDefinitionAction.create(_job(job_id, chr(97 + index) * 64, fire)),
        )
    definitions.checkpoint(definitions.replay())
    service = CronService(tmp_path / "definitions" / "jobs.json", definition_store=definitions)
    calls: list[str] = []

    async def compute(request):
        calls.append(request.run_id)
        return TurnResult(run_id=request.run_id, content="private result")

    agent = SimpleNamespace(execute_silent_cron_turn=compute)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    sessions = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    channels = FakeChannels()
    graph = CronRuntime(
        service,
        agent,
        sessions,
        channels,
        tmp_path / "runtime",
        "copied-root",
        CronRuntimeInjection(clock, lambda *_: True, 0.1),
    )
    graph.acquire()
    return graph, clock, calls, channels


async def _summary(graph, calls: list[str], channels, fired: list[bool]) -> dict[str, object]:
    records = graph.service.occurrence_store.inspect_records()
    state = graph.store.read()
    return {
        "fired": fired,
        "call_count": len(calls),
        "channel_targets": list(channels.calls),
        "occurrences": sorted(
            (
                record.state,
                getattr(record.identity, "scheduled_fire_time_ms", None),
            )
            for record in records
        ),
        "classifications": [
            (item["kind"], item["key"], item["revision"]) for item in state["classifications"]
        ],
        "cursor_next_fire": [item["next_fire"] for item in state["cursors"]],
        "retained_classifications": len(
            graph.store.retained_evidence(state)["classifications"]
        ),
    }


async def _reference(tmp_path: Path) -> tuple[dict[str, object], list[bool]]:
    graph, clock, calls, channels = _three_job_graph(tmp_path)
    fired: list[bool] = []
    for fire in (*FIRES, FIRES[-1]):
        clock.now = fire
        fired.append(await graph.scheduler.wake_once() is not None)
    summary = await _summary(graph, calls, channels, fired)
    await graph.aclose()
    return summary, fired


async def _measured(
    tmp_path: Path, monkeypatch
) -> tuple[dict[str, object], float, list[float]]:
    graph, clock, calls, channels = _three_job_graph(tmp_path)
    original_read = graph.store.read
    original_inspect = graph.service.occurrence_store.inspect_records

    def slow_read():
        time.sleep(PHASE_SECONDS)
        return original_read()

    def slow_inspect():
        time.sleep(PHASE_SECONDS)
        return original_inspect()

    monkeypatch.setattr(graph.store, "read", slow_read)
    monkeypatch.setattr(graph.service.occurrence_store, "inspect_records", slow_inspect)

    ticks: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks.append(time.monotonic())

    probe = asyncio.create_task(ticker())
    await asyncio.sleep(0.02)
    fired: list[bool] = []
    clock.now = FIRES[0]
    start = time.monotonic()
    fired.append(await graph.scheduler.wake_once() is not None)
    end = time.monotonic()
    tick_snapshot = list(ticks)
    for fire in (*FIRES[1:], FIRES[-1]):
        clock.now = fire
        fired.append(await graph.scheduler.wake_once() is not None)
    stop.set()
    await probe
    duration = end - start
    gaps: list[float] = []
    previous = start
    for stamp in tick_snapshot:
        if start <= stamp <= end:
            gaps.append(stamp - previous)
            previous = stamp
    gaps.append(end - previous)
    summary = await _summary(graph, calls, channels, fired)
    await graph.aclose()
    return summary, duration, gaps


@pytest.mark.asyncio
async def test_wake_keeps_loop_live_and_preserves_result(tmp_path, monkeypatch) -> None:
    reference, reference_fired = await _reference(tmp_path / "reference")
    measured, duration, gaps = await _measured(tmp_path / "measured", monkeypatch)

    # The work really ran, and both phases stayed on the worker thread.
    assert duration >= 2 * PHASE_SECONDS
    assert len(gaps) >= 2
    assert max(gaps) <= LOOP_BOUND_SECONDS, f"loop stalled for {max(gaps):.3f}s"
    # Ordering, identity, occurrence settlement and retained proof are unchanged.
    assert reference_fired == [True, True, True, False]
    assert measured == reference
    assert measured["call_count"] == 3
    assert measured["channel_targets"] == ["A", "B"] * 3
    assert [fire for _state, fire in measured["occurrences"]] == list(FIRES)
    assert measured["retained_classifications"] >= 1


@pytest.mark.asyncio
async def test_cancelled_wake_waits_for_the_worker_before_releasing(tmp_path, monkeypatch) -> None:
    graph, _item, clock, calls, channels = runtime(tmp_path)
    original_read = graph.store.read
    release = threading.Event()

    def gated_read():
        release.wait(5.0)
        return original_read()

    monkeypatch.setattr(graph.store, "read", gated_read)
    clock.now = 1100
    wake = asyncio.create_task(graph.scheduler.wake_once())
    await asyncio.sleep(0.05)
    wake.cancel()
    await asyncio.sleep(0.05)
    assert not wake.done(), "cancellation must not abandon the durable read"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await wake
    assert calls == []
    await graph.aclose()
