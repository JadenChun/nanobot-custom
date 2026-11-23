"""Journal reads and startup recovery must not stall the event loop.

Requirements cases U1 to U3 from
``a3-evidence/a3-reconciliation-2026-11-14/resume2/req-a38-repairs/CASES.md``.
The common observation rule applies: a 5 ms ticker records timestamps, and the
maximum inter-tick gap during the measured interval must stay at or below
100 ms. The running product path for retained results is covered separately.

The single journal lane is held with one slow operation so the measured call
really waits behind queued journal work.
"""

import asyncio
import json
import re
import sqlite3
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.tools.context import request_context
from nanobot.agent.tools.exec_session import ExecCompletion
from nanobot.process_completion import ProcessCompletionRuntime
from tests.tools.test_exec_session_tools import _python_command
from tests.tools.test_process_completion import settled, setup

LOOP_GAP_BOUND_S = 0.1
TICK_S = 0.005
LANE_HOLD_S = 0.2


def _slow_journal_operation(db: sqlite3.Connection) -> None:
    db.execute("SELECT 1").fetchall()
    time.sleep(LANE_HOLD_S)


def _raw_records(path: Path) -> dict[str, str]:
    with sqlite3.connect(path) as db:
        return {identity: raw for identity, raw in db.execute("SELECT id, data FROM processes")}


def _max_gap(ticks: list[float], start: float, end: float) -> float:
    window = [tick for tick in ticks if start <= tick <= end]
    assert window, "the ticker recorded no run during the measured interval"
    points = [start, *window, end]
    return max(later - earlier for earlier, later in zip(points, points[1:]))


@asynccontextmanager
async def _ticker():
    ticks: list[float] = []

    async def run() -> None:
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(TICK_S)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.02)
    try:
        yield ticks
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def _hold_lane(runtime: ProcessCompletionRuntime) -> "asyncio.Task[Any]":
    return asyncio.create_task(asyncio.to_thread(runtime._lane.run, _slow_journal_operation))


async def _await_lane_pending(runtime: ProcessCompletionRuntime) -> None:
    async with asyncio.timeout(2):
        while runtime._lane._queue.qsize() < 1:
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_u1_ordered_journal_write_does_not_stall_loop(tmp_path):
    _project, _assistant, runtime, manager, _tool, origin = setup(tmp_path)
    try:
        runtime.started("u1-base", "base command", str(origin.workspace), origin)
        runtime.inline_completed("u1-base")
        before = _raw_records(runtime.path)
        async with _ticker() as ticks:
            holder = _hold_lane(runtime)
            await asyncio.sleep(0.02)
            queued_at = time.monotonic()
            writer = asyncio.create_task(asyncio.to_thread(
                runtime.started, "u1-write", "queued command", str(origin.workspace), origin))
            await _await_lane_pending(runtime)
            await asyncio.sleep(0.02)
            await writer
            completed_at = time.monotonic()
            await holder
        assert _max_gap(ticks, queued_at, completed_at) <= LOOP_GAP_BOUND_S
        after = _raw_records(runtime.path)
        assert set(after) == {"u1-base", "u1-write"}
        assert json.loads(after["u1-write"])["id"] == "u1-write"
        assert after["u1-base"] == before["u1-base"]
        assert runtime.get("u1-write").state == "waiting"
    finally:
        await manager.close_all()
        await runtime.aclose()


@pytest.mark.asyncio
async def test_u2_startup_recovery_does_not_stall_loop(tmp_path):
    _project, assistant, runtime, manager, _tool, origin = setup(tmp_path)
    runtime.started("interrupted", "interrupted command", str(origin.workspace), origin)
    runtime.started("rescheduled", "rescheduled command", str(origin.workspace), origin)
    runtime.finished("rescheduled", ExecCompletion("rescheduled output", 0))
    await runtime.aclose()
    before = _raw_records(runtime.path)
    reopened = ProcessCompletionRuntime(tmp_path / "runtime", assistant)
    try:
        assert json.loads(before["rescheduled"])["state"] == "ready"
        async with _ticker() as ticks:
            holder = _hold_lane(reopened)
            await asyncio.sleep(0.02)
            started_at = time.monotonic()
            await reopened.astart()
            recovered_at = time.monotonic()
            await holder
        assert _max_gap(ticks, started_at, recovered_at) <= LOOP_GAP_BOUND_S
        after = _raw_records(reopened.path)
        assert set(after) == {"interrupted", "rescheduled"}
        recovered = json.loads(after["interrupted"])
        original = json.loads(before["interrupted"])
        assert recovered["id"] == "interrupted"
        assert recovered["state"] == "unknown"
        assert recovered["reason"] == "runtime_interrupted"
        assert recovered == {**original, "state": "unknown", "reason": "runtime_interrupted"}
        assert after["rescheduled"] == before["rescheduled"]
        assert reopened.get("interrupted").state == "unknown"
        await settled(reopened, "rescheduled", "done")
        assert len(assistant.messages) == 1
        assert assistant.messages[0].session_key == origin.session_key
    finally:
        await reopened.aclose()
        await manager.close_all()


@pytest.mark.asyncio
async def test_u3_retained_result_read_does_not_stall_loop(tmp_path):
    _project, assistant, runtime, manager, _tool, origin = setup(tmp_path)
    try:
        runtime.started("completed-job", "completed command", str(origin.workspace), origin)
        runtime.finished("completed-job", ExecCompletion("retained output", 3))
        await settled(runtime, "completed-job", "done")
        before = _raw_records(runtime.path)
        async with _ticker() as ticks:
            holder = _hold_lane(runtime)
            await asyncio.sleep(0.02)
            queued_at = time.monotonic()
            writer = asyncio.create_task(asyncio.to_thread(
                runtime.started, "u3-queued", "queued command", str(origin.workspace), origin))
            await _await_lane_pending(runtime)
            read = asyncio.create_task(
                runtime.aresult("completed-job", origin.session_key, source_state=origin.source_state))
            await writer
            retained = await read
            returned_at = time.monotonic()
            await holder
        assert retained is not None
        assert retained.output == "retained output" and retained.exit_code == 3
        assert retained.state == "completed" and not retained.terminated
        assert _max_gap(ticks, queued_at, returned_at) <= LOOP_GAP_BOUND_S
        after = _raw_records(runtime.path)
        assert after["completed-job"] == before["completed-job"]
        assert "u3-queued" in after
    finally:
        await manager.close_all()
        await runtime.aclose()


@pytest.mark.asyncio
async def test_exec_session_retained_result_poll_does_not_stall_loop(tmp_path):
    """The running product path: a poll for a settled job reads off the loop."""
    project, _assistant, runtime, manager, tool, origin = setup(tmp_path)
    try:
        script = "from pathlib import Path; import time; "
        script += "\nwhile not Path('release').exists(): time.sleep(.01)\nprint('finished')"
        with request_context(origin):
            initial = await tool.execute(command=_python_command(script), notify_on_exit=True,
                                         yield_time_ms=0)
        identity = re.search(r"session_id: ([a-f0-9]+)", initial).group(1)
        (project / "release").touch()
        await settled(runtime, identity, "done")
        async with _ticker() as ticks:
            holder = _hold_lane(runtime)
            await asyncio.sleep(0.02)
            polled_at = time.monotonic()
            with request_context(origin):
                poll = await manager.write(
                    session_id=identity,
                    chars=None,
                    close_stdin=False,
                    terminate=False,
                    yield_time_ms=0,
                    max_output_chars=2000,
                    owner_session_key=origin.session_key,
                )
            returned_at = time.monotonic()
            await holder
        assert poll.done and poll.completion_state == "completed"
        assert poll.exit_code == 0 and "finished" in poll.output
        assert _max_gap(ticks, polled_at, returned_at) <= LOOP_GAP_BOUND_S
    finally:
        await manager.close_all()
        await runtime.aclose()
