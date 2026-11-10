"""Regression coverage for inline child stop and control-generation checks."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.turn import TurnContext
from nanobot.session.manager import SessionManager

PARENT = "websocket:parent"


def manager_for(tmp_path: Path):
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    sessions = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    manager = ForegroundAgentManager(
        provider=provider,
        workspace=tmp_path,
        sessions=sessions,
    )
    parent = sessions.get_or_create(PARENT)
    sessions.save(parent)
    return manager, sessions


class GateRunner:
    def __init__(self, *, gate: asyncio.Event | None = None, text: str = "child result"):
        self.gate = gate
        self.text = text
        self.calls: list[str | None] = []
        self.cancelled = False
        self.finished = asyncio.Event()

    async def run(self, spec):
        key = spec.turn_context.session_key if spec.turn_context else None
        self.calls.append(key)
        if self.gate is not None:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        else:
            await asyncio.sleep(0.15)
        self.finished.set()
        return AgentRunResult(final_content=self.text, messages=[])


class TwoGateRunner:
    def __init__(self, gates):
        self.gates = gates
        self.calls = 0
        self.cancelled: list[bool] = []

    async def run(self, spec):
        index = self.calls
        self.calls += 1
        self.cancelled.append(False)
        try:
            await self.gates[index].wait()
        except asyncio.CancelledError:
            self.cancelled[index] = True
            raise
        return AgentRunResult(final_content=f"result {index}", messages=[])


async def _wait_for(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    return predicate()


def _active_children(manager, count: int):
    children = manager._delegation_tree.children_of(PARENT)
    if len(children) >= count and all(child.state == "active" for child in children[:count]):
        return children
    return None


async def _settle(task: asyncio.Task) -> None:
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except asyncio.CancelledError:
        pass


async def _release_gate_after(gate: asyncio.Event, *, delay: float) -> None:
    await asyncio.sleep(delay)
    gate.set()


async def test_foreground_stop_cancels_the_runner_and_settles_the_child(tmp_path):
    manager, _sessions = manager_for(tmp_path)
    gate = asyncio.Event()
    runner = GateRunner(gate=gate)
    manager.runner = runner
    ctx = TurnContext(session_key=PARENT, run_id="turn-1")

    task = asyncio.create_task(
        manager.run_worker(
            contract="Do the bounded work with acceptance evidence and validation.",
            write_scope=["out.txt"],
            turn_context=ctx,
        )
    )

    children = await _wait_for(lambda: _active_children(manager, 1))
    assert children is not None
    child = children[0]

    stopped = await manager.control_child(
        parent_key=PARENT,
        child_id=child.child_id,
        action="stop",
    )

    assert stopped["ok"] is True
    assert stopped["state"] == "stopped"
    assert runner.cancelled is True

    await _settle(task)

    final = await _wait_for(
        lambda: manager._delegation_tree.child(child.child_id)
        if manager._delegation_tree.child(child.child_id).state == "stopped"
        else None
    )
    assert final is not None
    assert final.state == "stopped"


async def test_stopping_one_sibling_leaves_the_other_running(tmp_path):
    manager, _sessions = manager_for(tmp_path)
    gate_a = asyncio.Event()
    gate_b = asyncio.Event()
    runner = TwoGateRunner([gate_a, gate_b])
    manager.runner = runner
    ctx = TurnContext(session_key=PARENT, run_id="turn-1")

    task_a = asyncio.create_task(
        manager.run_worker(
            contract="first sibling contract with acceptance evidence.",
            write_scope=["out.txt"],
            turn_context=ctx,
        )
    )
    first = await _wait_for(lambda: _active_children(manager, 1))
    assert first is not None

    task_b = asyncio.create_task(
        manager.run_worker(
            contract="second sibling contract with acceptance evidence.",
            write_scope=["out.txt"],
            turn_context=ctx,
        )
    )
    both = await _wait_for(lambda: _active_children(manager, 2))
    assert both is not None
    child_a, child_b = both[0], both[1]

    stopped = await manager.control_child(
        parent_key=PARENT,
        child_id=child_a.child_id,
        action="stop",
    )
    assert stopped["ok"] is True
    await _settle(task_a)

    cancelled = await _wait_for(lambda: runner.cancelled[0] if runner.cancelled and runner.cancelled[0] else None)
    assert cancelled is True
    assert runner.cancelled[1] is False
    assert manager._delegation_tree.child(child_b.child_id).state == "active"

    gate_b.set()
    await asyncio.wait_for(task_b, timeout=5.0)

    completed = await _wait_for(
        lambda: manager._delegation_tree.child(child_b.child_id)
        if manager._delegation_tree.child(child_b.child_id).state == "completed"
        else None
    )
    assert completed is not None
    assert completed.state == "completed"


async def test_control_generation_is_checked(tmp_path):
    manager, _sessions = manager_for(tmp_path)
    gate = asyncio.Event()
    runner = GateRunner(gate=gate)
    manager.runner = runner
    ctx = TurnContext(session_key=PARENT, run_id="turn-1")

    await manager.run_worker(
        contract="Do the bounded work with acceptance evidence and validation.",
        write_scope=["out.txt"],
        turn_context=ctx,
        background=True,
    )
    child = manager._delegation_tree.children_of(PARENT)[0]
    assert child.state == "active"

    stale = await manager.control_child(
        parent_key=PARENT,
        child_id=child.child_id,
        action="wait",
        runtime_generation=99,
    )
    assert stale["ok"] is False
    assert stale["reason"] == "stale_generation"

    matching_release = asyncio.create_task(_release_gate_after(gate, delay=0.05))
    matching = await asyncio.wait_for(
        manager.control_child(
            parent_key=PARENT,
            child_id=child.child_id,
            action="wait",
            runtime_generation=child.runtime_generation,
        ),
        timeout=5.0,
    )
    await matching_release
    assert matching["ok"] is True

    await asyncio.wait_for(runner.finished.wait(), timeout=2.0)
    await _wait_for(
        lambda: manager._delegation_tree.child(child.child_id).state == "completed"
    )
