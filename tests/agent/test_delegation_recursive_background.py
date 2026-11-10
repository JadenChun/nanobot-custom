"""Regression coverage for recursive delegation and background child continuity."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.turn import TurnContext
from nanobot.session.manager import SessionManager

PARENT = "websocket:parent"
EXPLORE_OK = (
    "---EXPLORE---\n"
    '{"summary":"nested","findings":["f"],"references":[],'
    '"open_questions":[],"searched_areas":[],"partial":false}\n'
    "---END---"
)


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


class NestedRunner:
    def __init__(self, manager):
        self.manager = manager
        self.keys: list[str | None] = []

    async def run(self, spec):
        key = spec.turn_context.session_key if spec.turn_context else None
        self.keys.append(key)
        if len(self.keys) == 1:
            nested = await self.manager.run_explore(
                task="nested read-only task",
                thoroughness="quick",
                max_iterations=2,
                turn_context=spec.turn_context,
            )
            return AgentRunResult(final_content=f"worker saw: {nested['summary']}", messages=[])
        return AgentRunResult(final_content=EXPLORE_OK, messages=[])


def _messages(sessions: SessionManager, key: str) -> list[dict]:
    return sessions.get_or_create(key).messages


def _text(message: dict) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else str(content)


def _roles(messages: list[dict]) -> set[str | None]:
    return {message.get("role") for message in messages}


async def _wait_for(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    return predicate()


async def test_recursive_child_and_grandchild_have_separate_identities_and_histories(tmp_path):
    manager, sessions = manager_for(tmp_path)
    runner = NestedRunner(manager)
    manager.runner = runner
    ctx = TurnContext(session_key=PARENT, run_id="turn-1")

    await manager.run_worker(
        contract="Do the bounded work with acceptance evidence and validation.",
        write_scope=["out.txt"],
        turn_context=ctx,
    )

    children = manager._delegation_tree.children_of(PARENT)
    assert len(children) == 1
    child = children[0]
    assert child.depth == 1
    assert child.state == "completed"

    grandchildren = manager._delegation_tree.children_of(child.session_key)
    assert len(grandchildren) == 1
    grandchild = grandchildren[0]
    assert grandchild.depth == 2
    assert grandchild.parent_key == child.session_key
    assert grandchild.state == "completed"

    child_messages = _messages(sessions, child.session_key)
    grandchild_messages = _messages(sessions, grandchild.session_key)
    assert {"user", "assistant"} <= _roles(child_messages)
    assert {"user", "assistant"} <= _roles(grandchild_messages)
    assert any("nested read-only task" in _text(message) for message in grandchild_messages)

    assert child.child_id != grandchild.child_id
    assert child.session_key != grandchild.session_key
    assert runner.keys[0] == child.session_key


async def test_third_delegation_level_is_refused_with_no_child(tmp_path):
    manager, sessions = manager_for(tmp_path)
    runner = NestedRunner(manager)
    manager.runner = runner
    ctx = TurnContext(session_key=PARENT, run_id="turn-1")

    await manager.run_worker(
        contract="Do the bounded work with acceptance evidence and validation.",
        write_scope=["out.txt"],
        turn_context=ctx,
    )

    child = manager._delegation_tree.children_of(PARENT)[0]
    grandchild = manager._delegation_tree.children_of(child.session_key)[0]
    third_ctx = TurnContext(session_key=grandchild.session_key, run_id="turn-3")
    manager.runner = GateRunner(text=EXPLORE_OK)

    refused = await manager.run_explore(
        task="too deep",
        thoroughness="quick",
        max_iterations=2,
        turn_context=third_ctx,
    )

    if isinstance(refused, dict):
        assert refused["partial"] is True
        summary = str(refused["summary"]).lower()
    else:
        summary = str(refused).lower()
    assert "limit" in summary or "refus" in summary
    assert manager._delegation_tree.children_of(grandchild.session_key) == ()


async def test_background_child_continues_after_the_call_and_uses_independent_history(tmp_path):
    manager, sessions = manager_for(tmp_path)
    gate = asyncio.Event()
    runner = GateRunner(gate=gate)
    manager.runner = runner
    ctx = TurnContext(session_key=PARENT, run_id="turn-1")

    started = time.monotonic()
    result = await manager.run_worker(
        contract="Do the bounded work with acceptance evidence and validation.",
        write_scope=["out.txt"],
        turn_context=ctx,
        background=True,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    children = manager._delegation_tree.children_of(PARENT)
    assert len(children) == 1
    child = children[0]
    assert "background" in str(result).lower()
    assert child.child_id in str(result)
    assert child.state == "active"

    duplicate = await manager.run_worker(
        contract="Do the bounded work with acceptance evidence and validation.",
        write_scope=["out.txt"],
        turn_context=ctx,
        background=True,
    )
    assert child.child_id in str(duplicate)
    assert "already active" in str(duplicate).lower()
    started_call = await _wait_for(lambda: len(runner.calls) == 1)
    assert started_call is True
    assert len(runner.calls) == 1

    gate.set()
    await asyncio.wait_for(runner.finished.wait(), timeout=2.0)
    completed = await _wait_for(
        lambda: manager._delegation_tree.child(child.child_id)
        if manager._delegation_tree.child(child.child_id).state == "completed"
        else None
    )
    assert completed is not None
    assert completed.state == "completed"

    messages = _messages(sessions, child.session_key)
    assert {"user", "assistant"} <= _roles(messages)
    assistant = next(message for message in messages if message.get("role") == "assistant")
    assert "child result" in _text(assistant)
