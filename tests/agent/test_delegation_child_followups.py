"""Canonical follow-up behavior for delegated children."""

import asyncio
import re
from collections import deque
from copy import deepcopy

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnRequest
from nanobot.bus.events import InboundMessage
from nanobot.command.builtin import cmd_stop
from nanobot.command.router import CommandContext
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.source import validate_execution_links
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

pytest_plugins = ("tests.agent.test_migration_source_engine",)


def _tool_call(name: str, arguments: dict[str, object], call_id: str) -> LLMResponse:
    return LLMResponse(content=None, tool_calls=[ToolCallRequest(
        id=call_id,
        name=name,
        arguments=arguments,
    )])


def _tool_names(tools: object) -> set[str]:
    if not isinstance(tools, list):
        return set()
    return {
        str(function["name"])
        for item in tools
        if isinstance(item, dict)
        and isinstance(item.get("function"), dict)
        and isinstance((function := item["function"]).get("name"), str)
    }


class _FollowupProvider(ScriptedProvider):
    """Separate parent and child scripts, with optional offline child barriers."""

    def __init__(
        self,
        parent_responses: list[LLMResponse],
        child_responses: list[LLMResponse] | None = None,
        *,
        gate_first_child: bool = False,
        gate_child_call: int | None = None,
        resist_child_cancellation: bool = False,
    ) -> None:
        super().__init__(parent_responses)
        self.child_responses = deque(child_responses or [])
        self.parent_calls: list[tuple[list[dict[str, object]], object, str | None]] = []
        self.child_calls: list[tuple[list[dict[str, object]], object, str | None]] = []
        self.child_started = asyncio.Event()
        self.child_release = asyncio.Event()
        self.child_cancel_received = asyncio.Event()
        self.child_has_both_followups = asyncio.Event()
        self.gate_first_child = gate_first_child
        self.gate_child_call = gate_child_call or (1 if gate_first_child else None)
        self.resist_child_cancellation = resist_child_cancellation
        self.active_child_requests = 0
        self.max_active_child_requests = 0

    async def chat(self, messages, tools=None, model=None, **kwargs):
        detached_messages = deepcopy(messages)
        call = (detached_messages, tools, model)
        self.calls.append(call)
        names = _tool_names(tools)
        child_request = "delegate_task" not in names and "followup_child" not in names
        if not child_request:
            self.parent_calls.append(call)
            assert self.responses, "unexpected extra parent provider request"
            return self.responses.popleft()

        self.child_calls.append(call)
        self.active_child_requests += 1
        self.max_active_child_requests = max(
            self.max_active_child_requests,
            self.active_child_requests,
        )
        try:
            if self._messages_contain(detached_messages, "Follow-up A") and self._messages_contain(
                detached_messages, "Follow-up B",
            ):
                self.child_has_both_followups.set()
            if len(self.child_calls) == self.gate_child_call:
                self.child_started.set()
                try:
                    await self.child_release.wait()
                except asyncio.CancelledError:
                    if not self.resist_child_cancellation:
                        raise
                    self.child_cancel_received.set()
                    await self.child_release.wait()
            if self.child_responses:
                return self.child_responses.popleft()
            return LLMResponse(content="The child applied its queued follow-ups.")
        finally:
            self.active_child_requests -= 1

    @staticmethod
    def _messages_contain(messages: list[dict[str, object]], text: str) -> bool:
        return any(text in str(message.get("content", "")) for message in messages)


def _make_loop(engine_config, provider: _FollowupProvider, parent_key: str) -> AgentLoop:
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=provider,
    )
    upgrade(loop, parent_key)
    return loop


def _source_snapshot(loop: AgentLoop, session_key: str):
    backend = loop.sessions.source_backend(session_key)
    assert backend is not None
    return backend.source.load_source_snapshot(session_key)


def _pending_followups(loop: AgentLoop, child_id: str) -> list[dict[str, object]]:
    child = loop._fork_dependencies.foreground_manager._delegation_tree.child(child_id)
    assert child is not None
    snapshot = _source_snapshot(loop, child.session_key)
    value = snapshot.header["metadata"].get("source_pending_followups", [])
    assert isinstance(value, list)
    return value


def _child_id_from_delegated_result(
    loop: AgentLoop,
    parent_key: str,
    *,
    tool_call_id: str = "delegate",
) -> str:
    snapshot = _source_snapshot(loop, parent_key)
    event = next(
        event for event in snapshot.events
        if event.get("role") == "tool" and event.get("tool_call_id") == tool_call_id
    )
    match = re.search(r"Child\s+([A-Za-z0-9_-]+)\s+started in background", str(event["content"]))
    assert match is not None, event
    return match.group(1)


async def _start_background_child(loop: AgentLoop, parent_key: str) -> str:
    result = await loop.execute_turn(TurnRequest(
        content="Start the delegated fixture task in the background.",
        session_key=parent_key,
    ))
    assert result.error is None, result.error
    return _child_id_from_delegated_result(loop, parent_key)


async def _parent_followup(
    loop: AgentLoop,
    provider: _FollowupProvider,
    *,
    parent_key: str,
    child_id: str,
    content: str,
    call_id: str,
) -> str:
    provider.responses.extend([
        _tool_call("followup_child", {"child_id": child_id, "content": content}, call_id),
        LLMResponse(content="The original chat received the follow-up disposition."),
    ])
    result = await loop.execute_turn(TurnRequest(
        content=f"Submit {content} to the delegated child.",
        session_key=parent_key,
    ))
    assert result.error is None, result.error
    assert "followup_child" in result.tools_used
    event = next(
        event for event in _source_snapshot(loop, parent_key).events
        if event.get("role") == "tool" and event.get("tool_call_id") == call_id
    )
    return str(event["content"])


async def _wait_for_child_completion(loop: AgentLoop, child_id: str) -> None:
    tree = loop._fork_dependencies.foreground_manager._delegation_tree
    assert tree is not None
    child = await asyncio.wait_for(tree.wait_child(child_id, timeout=5), timeout=6)
    assert child is not None
    assert child.state == "completed", child.metadata()


async def test_followup_cancel_before_dispatch_releases_claim(engine_config, monkeypatch):
    parent_key = "cli:followup-cancel-claim"
    provider = _FollowupProvider(
        [
            _tool_call(
                "delegate_task",
                {
                    "contract": "Complete the bounded offline fixture task.",
                    "write_scope": ["fixture.txt"],
                    "background": True,
                },
                "delegate",
            ),
            LLMResponse(content="The delegated child is running."),
        ],
        child_responses=[LLMResponse(content="child final")],
    )
    loop = _make_loop(engine_config, provider, parent_key)
    try:
        child_id = await _start_background_child(loop, parent_key)
        await _wait_for_child_completion(loop, child_id)
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        assert tree is not None

        async def cancel_activate(_child_id: str):
            raise asyncio.CancelledError()

        monkeypatch.setattr(tree, "activate", cancel_activate)
        provider.responses.extend([
            _tool_call(
                "followup_child",
                {"child_id": child_id, "content": "Continue once."},
                "followup",
            ),
            LLMResponse(content="The follow-up was interrupted."),
        ])
        with pytest.raises(asyncio.CancelledError):
            await loop.execute_turn(TurnRequest(
                content="Continue the delegated child.",
                session_key=parent_key,
            ))
        child = tree.child(child_id)
        assert child is not None
        assert child.state == "stopped"
        assert child.holds_capacity is False
    finally:
        await loop.aclose()


async def test_busy_child_followups_are_durable_ordered_and_retry_safe(engine_config, monkeypatch):
    parent_key = "cli:followup-busy"
    provider = _FollowupProvider(
        [
            _tool_call("delegate_task", {
                "contract": "Complete the bounded offline fixture task.",
                "write_scope": ["fixture.txt"],
                "background": True,
            }, "delegate"),
            LLMResponse(content="The delegated child is running."),
        ],
        gate_first_child=True,
    )
    loop = _make_loop(engine_config, provider, parent_key)
    manager = loop._fork_dependencies.foreground_manager
    original_followup = getattr(manager, "followup_child", None)
    retries: dict[str, object] = {}
    try:
        child_id = await _start_background_child(loop, parent_key)
        await asyncio.wait_for(provider.child_started.wait(), timeout=3)
        child = manager._delegation_tree.child(child_id)
        assert child is not None and child.current_run_id is not None
        first_run_id = child.current_run_id

        if original_followup is not None:
            async def retry_same_dispatch(*, child_id: str, content: str, turn_context=None):
                disposition = await original_followup(
                    child_id=child_id,
                    content=content,
                    turn_context=turn_context,
                )
                if content == "Follow-up A" and "retry" not in retries:
                    after_first = deepcopy(_pending_followups(loop, child_id))
                    retry = await original_followup(
                        child_id=child_id,
                        content=content,
                        turn_context=turn_context,
                    )
                    conflict = await original_followup(
                        child_id=child_id,
                        content="Changed content under the same intent.",
                        turn_context=turn_context,
                    )
                    after_conflict = deepcopy(_pending_followups(loop, child_id))
                    retries.update({
                        "first": disposition,
                        "retry": retry,
                        "conflict": conflict,
                        "after_first": after_first,
                        "after_conflict": after_conflict,
                    })
                return disposition

            monkeypatch.setattr(manager, "followup_child", retry_same_dispatch)
        result_a = await _parent_followup(
            loop, provider, parent_key=parent_key, child_id=child_id,
            content="Follow-up A", call_id="followup-a",
        )
        assert result_a == "Follow-up queued."
        assert retries["first"] == retries["retry"] == "Follow-up queued."
        assert str(retries["conflict"]).startswith("Follow-up refused:")
        assert retries["after_first"] == retries["after_conflict"]
        assert [row["content"] for row in retries["after_first"]] == ["Follow-up A"]

        if original_followup is not None:
            async def capture_b_before_ack(*, child_id: str, content: str, turn_context=None):
                disposition = await original_followup(
                    child_id=child_id,
                    content=content,
                    turn_context=turn_context,
                )
                retries["b_before_ack"] = deepcopy(_pending_followups(loop, child_id))
                return disposition

            monkeypatch.setattr(manager, "followup_child", capture_b_before_ack)
        result_b = await _parent_followup(
            loop, provider, parent_key=parent_key, child_id=child_id,
            content="Follow-up B", call_id="followup-b",
        )
        assert result_b == "Follow-up queued."
        durable = _pending_followups(loop, child_id)
        assert [row["content"] for row in durable] == ["Follow-up A", "Follow-up B"]
        if original_followup is not None:
            assert [row["content"] for row in retries["b_before_ack"]] == [
                "Follow-up A", "Follow-up B",
            ]
        assert len({row["id"] for row in durable}) == 2
        assert len(provider.child_calls) == 1
        assert manager._delegation_tree.child(child_id).current_run_id == first_run_id

        provider.child_release.set()
        await asyncio.wait_for(provider.child_has_both_followups.wait(), timeout=5)
        await _wait_for_child_completion(loop, child_id)
        child = manager._delegation_tree.child(child_id)
        child_snapshot = _source_snapshot(loop, child.session_key)
        inputs = [
            str(event.get("content", ""))
            for event in child_snapshot.events
            if event.get("role") == "user"
        ]
        assert inputs.count("Follow-up A") == 1
        assert inputs.count("Follow-up B") == 1
        assert inputs.index("Follow-up A") < inputs.index("Follow-up B")
        assert _pending_followups(loop, child_id) == []
        assert provider.max_active_child_requests == 1
    finally:
        provider.child_release.set()
        await loop.aclose()


async def test_idle_child_followup_uses_existing_canonical_history(engine_config):
    parent_key = "cli:followup-idle"
    provider = _FollowupProvider(
        [
            _tool_call("delegate_task", {
                "contract": "Complete the bounded offline fixture task.",
                "write_scope": ["fixture.txt"],
                "background": True,
            }, "delegate"),
            LLMResponse(content="The delegated child is running."),
        ],
        child_responses=[
            LLMResponse(content="The original child result."),
            LLMResponse(content="The child applied the new follow-up."),
        ],
    )
    loop = _make_loop(engine_config, provider, parent_key)
    try:
        child_id = await _start_background_child(loop, parent_key)
        await _wait_for_child_completion(loop, child_id)
        manager = loop._fork_dependencies.foreground_manager
        tree = manager._delegation_tree
        assert tree is not None
        child = tree.child(child_id)
        assert child is not None and child.state == "completed"
        old_run_id = child.current_run_id
        old_run_sequence = child.run_sequence
        original_snapshot = _source_snapshot(loop, child.session_key)
        original_incarnation = original_snapshot.state.incarnation
        original_store_identity = original_snapshot.state.store_identity

        disposition = await _parent_followup(
            loop, provider, parent_key=parent_key, child_id=child_id,
            content="One idle follow-up.", call_id="idle-followup",
        )
        assert disposition == "Follow-up applied."
        await _wait_for_child_completion(loop, child_id)
        current = tree.child(child_id)
        assert current is not None and current.current_run_id != old_run_id
        assert current.run_sequence == old_run_sequence + 1
        updated_snapshot = _source_snapshot(loop, child.session_key)
        assert updated_snapshot.state.incarnation == original_incarnation
        assert updated_snapshot.state.store_identity == original_store_identity
        assert updated_snapshot.events[:len(original_snapshot.events)] == original_snapshot.events
        inputs = [
            str(event.get("content", ""))
            for event in updated_snapshot.events
            if event.get("role") == "user"
        ]
        assert inputs.count("One idle follow-up.") == 1
        assert len(provider.child_calls) == 2
        assert provider._messages_contain(provider.child_calls[-1][0], "original child")
        assert provider._messages_contain(provider.child_calls[-1][0], "One idle follow-up.")
        assert provider.max_active_child_requests == 1
    finally:
        await loop.aclose()


async def test_followups_submitted_at_child_settlement_remain_ordered(engine_config, monkeypatch):
    parent_key = "cli:followup-settlement"
    provider = _FollowupProvider(
        [
            _tool_call("delegate_task", {
                "contract": "Complete the bounded offline fixture task.",
                "write_scope": ["fixture.txt"],
                "background": True,
            }, "delegate"),
            LLMResponse(content="The delegated child is running."),
        ],
        child_responses=[LLMResponse(content="The child reached settlement.")],
    )
    loop = _make_loop(engine_config, provider, parent_key)
    manager = loop._fork_dependencies.foreground_manager
    original_finalize = manager._finalize_admission
    settling = asyncio.Event()
    first_submission_started = asyncio.Event()
    release_settlement = asyncio.Event()
    worker_finalizations = 0
    followup_tasks: list[asyncio.Task[str]] = []
    try:
        async def hold_settlement(admission, result, **kwargs):
            nonlocal worker_finalizations
            if kwargs.get("role") == "worker":
                worker_finalizations += 1
                if worker_finalizations == 2:
                    settling.set()
                    await release_settlement.wait()
            return await original_finalize(admission, result, **kwargs)

        monkeypatch.setattr(manager, "_finalize_admission", hold_settlement)
        child_id = await _start_background_child(loop, parent_key)
        await _wait_for_child_completion(loop, child_id)
        tree = manager._delegation_tree
        assert tree is not None
        child = tree.child(child_id)
        assert child is not None and child.state == "completed"
        original_run_id = child.current_run_id
        original_snapshot = _source_snapshot(loop, child.session_key)

        idle_disposition = await _parent_followup(
            loop, provider, parent_key=parent_key, child_id=child_id,
            content="Idle follow-up.", call_id="idle-before-settlement",
        )
        assert idle_disposition == "Follow-up applied."
        await asyncio.wait_for(settling.wait(), timeout=3)
        child = tree.child(child_id)
        assert child is not None and child.current_run_id != original_run_id
        assert child.run_sequence == 2
        assert len(provider.child_calls) == 2
        assert provider.active_child_requests == 0

        original_followup = manager.followup_child

        async def signal_first_submission(*, child_id: str, content: str, turn_context=None):
            if content == "Follow-up A":
                first_submission_started.set()
            return await original_followup(
                child_id=child_id,
                content=content,
                turn_context=turn_context,
            )

        monkeypatch.setattr(manager, "followup_child", signal_first_submission)
        first_task = asyncio.create_task(_parent_followup(
            loop, provider, parent_key=parent_key, child_id=child_id,
            content="Follow-up A", call_id="settlement-followup-a",
        ))
        followup_tasks.append(first_task)
        await asyncio.wait_for(first_submission_started.wait(), timeout=3)

        release_settlement.set()
        first = await asyncio.wait_for(first_task, timeout=5)
        second = await _parent_followup(
            loop, provider, parent_key=parent_key, child_id=child_id,
            content="Follow-up B", call_id="settlement-followup-b",
        )
        assert first in {"Follow-up queued.", "Follow-up applied."}
        assert second in {"Follow-up queued.", "Follow-up applied."}

        await _wait_for_child_completion(loop, child_id)
        child = tree.child(child_id)
        snapshot = _source_snapshot(loop, child.session_key)
        inputs = [
            str(event.get("content", ""))
            for event in snapshot.events
            if event.get("role") == "user"
        ]
        assert inputs.count("Follow-up A") == 1
        assert inputs.count("Follow-up B") == 1
        assert inputs.index("Follow-up A") < inputs.index("Follow-up B")
        assert inputs.count("Idle follow-up.") == 1
        assert snapshot.events[:len(original_snapshot.events)] == original_snapshot.events
        assert snapshot.state.incarnation == original_snapshot.state.incarnation
        assert snapshot.state.store_identity == original_snapshot.state.store_identity
        assert _pending_followups(loop, child_id) == []
        assert provider.max_active_child_requests == 1
        assert provider.active_child_requests == 0
    finally:
        release_settlement.set()
        provider.child_release.set()
        for task in followup_tasks:
            if not task.done():
                await asyncio.gather(task, return_exceptions=True)
        await loop.aclose()


async def test_unresolved_child_effect_blocks_the_next_provider_request(engine_config):
    parent_key = "cli:followup-unresolved-effect"
    blocked_target = engine_config.workspace_path / "blocked-target"
    blocked_target.mkdir()
    provider = _FollowupProvider(
        [
            _tool_call("delegate_task", {
                "contract": "Write the requested fixture file.",
                "write_scope": ["blocked-target"],
                "background": True,
            }, "delegate"),
            LLMResponse(content="The parent received the child result."),
        ],
        child_responses=[_tool_call("write_file", {
            "path": "blocked-target",
            "content": "The target is a directory.",
        }, "unresolved-write")],
    )
    loop = _make_loop(engine_config, provider, parent_key)
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Delegate the file write.",
            session_key=parent_key,
        ))
        assert result.error is None, result.error
        child_id = _child_id_from_delegated_result(loop, parent_key)
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        assert tree is not None
        await asyncio.wait_for(tree.wait_child(child_id, timeout=5), timeout=6)
        child = tree.child(child_id)
        assert child is not None
        child_events = _source_snapshot(loop, child.session_key).events
        assert len(provider.child_calls) == 1
        assert validate_execution_links(child_events)
        assert any(
            event.get("tool_call_id") == "unresolved-write"
            and any(
                fact.get("reconciliation_required") is True
                for fact in event.get("_filesystem_effects", [])
            )
            for event in child_events
            if isinstance(event.get("_filesystem_effects"), list)
        )
    finally:
        await loop.aclose()


async def test_original_stop_discards_queued_child_input_before_release(engine_config):
    parent_key = "cli:followup-stop"
    (engine_config.workspace_path / "fixture.txt").write_text("offline fixture contents")
    provider = _FollowupProvider(
        [
            _tool_call("delegate_task", {
                "contract": "Complete the bounded offline fixture task.",
                "write_scope": ["fixture.txt"],
                "background": True,
            }, "delegate"),
            LLMResponse(content="The delegated child is running."),
        ],
        child_responses=[
            _tool_call("read_file", {"path": "fixture.txt"}, "read-before-stop"),
            _tool_call("read_file", {"path": "fixture.txt"}, "read-after-stop"),
        ],
        gate_child_call=2,
        resist_child_cancellation=True,
    )
    loop = _make_loop(engine_config, provider, parent_key)
    child_id = None
    stop_task = None
    try:
        child_id = await _start_background_child(loop, parent_key)
        await asyncio.wait_for(provider.child_started.wait(), timeout=3)
        child = loop._fork_dependencies.foreground_manager._delegation_tree.child(child_id)
        assert child is not None and child.current_run_id is not None
        assert len(provider.child_calls) == 2
        original_run_id = child.current_run_id

        disposition = await _parent_followup(
            loop, provider, parent_key=parent_key, child_id=child_id,
            content="Follow-up A", call_id="stop-followup",
        )
        assert disposition == "Follow-up queued."
        assert [row["content"] for row in _pending_followups(loop, child_id)] == ["Follow-up A"]

        stop_task = asyncio.create_task(cmd_stop(CommandContext(
            msg=InboundMessage(
                channel="cli",
                chat_id="followup-stop",
                sender_id="user",
                content="/stop",
            ),
            session=None,
            key=parent_key,
            raw="/stop",
            loop=loop,
        )))
        await asyncio.wait_for(provider.child_cancel_received.wait(), timeout=3)
        provider.child_release.set()
        await asyncio.wait_for(stop_task, timeout=5)

        current = loop._fork_dependencies.foreground_manager._delegation_tree.child(child_id)
        assert current is not None
        assert current.current_run_id == original_run_id
        assert len(provider.child_calls) == 2
        assert _pending_followups(loop, child_id) == []
        inputs = [
            str(event.get("content", ""))
            for event in _source_snapshot(loop, child.session_key).events
            if event.get("role") == "user"
        ]
        assert "Follow-up A" not in inputs
    finally:
        provider.child_release.set()
        if stop_task is not None and not stop_task.done():
            await asyncio.gather(stop_task, return_exceptions=True)
        await loop.aclose()


async def test_idle_followup_refuses_after_scoped_write_link_retarget(engine_config):
    parent_key = "cli:followup-changed-target"
    workspace = engine_config.workspace_path
    inside_a = workspace / "inside-a.txt"
    inside_b = workspace / "inside-b.txt"
    link = workspace / "workspace-note"
    inside_a.write_text("A0")
    inside_b.write_text("B0")
    link.symlink_to(inside_a.name)

    provider = _FollowupProvider(
        [
            _tool_call("delegate_task", {
                "contract": "Inspect the admitted workspace note and finish the task.",
                "write_scope": ["workspace-note"],
                "background": True,
            }, "delegate"),
            LLMResponse(content="The original chat received the worker result."),
        ],
        child_responses=[
            LLMResponse(content="The worker completed with the admitted link target."),
            _tool_call("write_file", {
                "path": "workspace-note",
                "content": "B1",
            }, "write-retargeted-link"),
        ],
    )
    loop = _make_loop(engine_config, provider, parent_key)
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Delegate the scoped workspace note task.",
            session_key=parent_key,
        ))
        assert result.error is None, result.error
        child_id = _child_id_from_delegated_result(loop, parent_key)
        await _wait_for_child_completion(loop, child_id)
        manager = loop._fork_dependencies.foreground_manager
        tree = manager._delegation_tree
        assert tree is not None
        child = tree.child(child_id)
        assert child is not None and child.current_run_id is not None
        original_run_id = child.current_run_id
        original_run_sequence = child.run_sequence
        original_child_calls = len(provider.child_calls)
        assert original_child_calls == 1

        link.unlink()
        link.symlink_to(inside_b.name)

        disposition = await _parent_followup(
            loop, provider, parent_key=parent_key, child_id=child_id,
            content="Replace the link target's text with B1.",
            call_id="changed-target-followup",
        )
        assert disposition.startswith("Follow-up refused:")
        current = tree.child(child_id)
        assert current is not None
        assert current.current_run_id == original_run_id
        assert current.run_sequence == original_run_sequence
        assert len(provider.child_calls) == original_child_calls
        assert inside_a.read_text() == "A0"
        assert inside_b.read_text() == "B0"
    finally:
        await loop.aclose()


async def test_stopped_retained_child_input_is_skipped_after_restart(engine_config, monkeypatch):
    parent_key = "cli:followup-stop-retained"
    provider = _FollowupProvider(
        [
            _tool_call("delegate_task", {
                "contract": "Complete the bounded offline fixture task.",
                "write_scope": ["fixture.txt"],
                "background": True,
            }, "delegate"),
            LLMResponse(content="The delegated child is running."),
        ],
        child_responses=[LLMResponse(content="The original child result.")],
    )
    loop = _make_loop(engine_config, provider, parent_key)
    manager = loop._fork_dependencies.foreground_manager
    original_runner_run = loop.runner.run
    runner_result_ready = asyncio.Event()
    release_runner_result = asyncio.Event()
    held_runner_results = 0
    restarted = None
    stop_task = None
    try:
        async def hold_terminal_runner_result(spec):
            nonlocal held_runner_results
            result = await original_runner_run(spec)
            if (
                held_runner_results == 0
                and isinstance(spec.session_key, str)
                and spec.session_key.startswith("delegated:")
            ):
                held_runner_results += 1
                runner_result_ready.set()
                await release_runner_result.wait()
            return result

        monkeypatch.setattr(loop.runner, "run", hold_terminal_runner_result)
        child_id = await _start_background_child(loop, parent_key)
        await asyncio.wait_for(runner_result_ready.wait(), timeout=3)
        tree = manager._delegation_tree
        assert tree is not None
        child = tree.child(child_id)
        assert child is not None and child.current_run_id is not None
        original_run_id = child.current_run_id

        queued = await _parent_followup(
            loop, provider, parent_key=parent_key, child_id=child_id,
            content="stopped input", call_id="retained-stop-followup",
        )
        assert queued == "Follow-up queued."
        assert [row["content"] for row in _pending_followups(loop, child_id)] == [
            "stopped input",
        ]

        release_runner_result.set()
        await _wait_for_child_completion(loop, child_id)
        settled = tree.child(child_id)
        assert settled is not None and settled.current_run_id == original_run_id
        prior_snapshot = _source_snapshot(loop, child.session_key)
        assert [row["content"] for row in _pending_followups(loop, child_id)] == [
            "stopped input",
        ]
        prior_inputs = [
            str(event.get("content", ""))
            for event in prior_snapshot.events
            if event.get("role") == "user"
        ]
        assert "stopped input" not in prior_inputs

        stop_task = asyncio.create_task(cmd_stop(CommandContext(
            msg=InboundMessage(
                channel="cli",
                chat_id="followup-stop-retained",
                sender_id="user",
                content="/stop",
            ),
            session=None,
            key=parent_key,
            raw="/stop",
            loop=loop,
        )))
        await asyncio.wait_for(stop_task, timeout=5)
        stopped = tree.child(child_id)
        assert stopped is not None and stopped.current_run_id == original_run_id
        assert _source_snapshot(loop, child.session_key).events == prior_snapshot.events
        assert [row["content"] for row in _pending_followups(loop, child_id)] == [
            "stopped input",
        ]

        await loop.aclose()
        restarted = AgentLoop.from_config(
            engine_config,
            tool_registry=ToolRegistry(),
            provider=provider,
        )
        restarted_manager = restarted._fork_dependencies.foreground_manager
        restarted_tree = restarted_manager._delegation_tree
        assert restarted_tree is not None
        restored = restarted_tree.child(child_id)
        assert restored is not None and restored.current_run_id == original_run_id
        assert len(provider.child_calls) == 1
        assert [row["content"] for row in _pending_followups(restarted, child_id)] == [
            "stopped input",
        ]

        fresh = await _parent_followup(
            restarted, provider, parent_key=parent_key, child_id=child_id,
            content="fresh input", call_id="fresh-after-stop-followup",
        )
        assert fresh == "Follow-up applied."
        await _wait_for_child_completion(restarted, child_id)

        current = restarted_tree.child(child_id)
        assert current is not None
        assert current.current_run_id != original_run_id
        assert current.run_sequence == restored.run_sequence + 1
        snapshot = _source_snapshot(restarted, child.session_key)
        assert snapshot.state.incarnation == prior_snapshot.state.incarnation
        assert snapshot.state.store_identity == prior_snapshot.state.store_identity
        assert snapshot.events[:len(prior_snapshot.events)] == prior_snapshot.events
        inputs = [
            str(event.get("content", ""))
            for event in snapshot.events
            if event.get("role") == "user"
        ]
        assert inputs.count("fresh input") == 1
        assert "stopped input" not in inputs
        assert len(provider.child_calls) == 2
        assert provider._messages_contain(provider.child_calls[-1][0], "fresh input")
        assert not provider._messages_contain(provider.child_calls[-1][0], "stopped input")
        assert provider.max_active_child_requests == 1
        assert _pending_followups(restarted, child_id) == []
    finally:
        release_runner_result.set()
        provider.child_release.set()
        if stop_task is not None and not stop_task.done():
            await asyncio.gather(stop_task, return_exceptions=True)
        if restarted is not None:
            await restarted.aclose()
        else:
            await loop.aclose()


async def test_stale_child_and_parent_references_refuse_without_mutation(engine_config, monkeypatch):
    parent_key = "cli:followup-stale"
    provider = _FollowupProvider(
        [
            _tool_call("delegate_task", {
                "contract": "Complete the first offline fixture task.",
                "write_scope": ["fixture.txt"],
                "background": True,
            }, "delegate"),
            LLMResponse(content="The first child completed."),
            _tool_call("followup_child", {
                "child_id": "stale-child-placeholder",
                "content": "A stale child input.",
            }, "stale-child-followup"),
            LLMResponse(content="The original chat received the refusal."),
            _tool_call("delegate_task", {
                "contract": "Complete a replacement offline fixture task.",
                "write_scope": ["fixture.txt"],
                "background": True,
            }, "delegate-replacement"),
            LLMResponse(content="The replacement child completed."),
            _tool_call("followup_child", {
                "child_id": "stale-parent-placeholder",
                "content": "Follow-up for the replacement child.",
            }, "capture-parent-dispatch"),
            LLMResponse(content="The replacement child received the follow-up."),
        ],
        child_responses=[
            LLMResponse(content="The first child result."),
            LLMResponse(content="The replacement child result."),
            LLMResponse(content="The replacement child applied its follow-up."),
        ],
    )
    loop = _make_loop(engine_config, provider, parent_key)
    manager = loop._fork_dependencies.foreground_manager
    original_followup = getattr(manager, "followup_child", None)
    captured: list[tuple[object, object]] = []
    try:
        stale_child_id = await _start_background_child(loop, parent_key)
        await _wait_for_child_completion(loop, stale_child_id)
        stale_child = manager._delegation_tree.child(stale_child_id)
        assert stale_child is not None
        old_child_run_id = stale_child.current_run_id

        assert loop.sessions.delete_session(stale_child.session_key) is True
        upgrade(loop, stale_child.session_key)
        replacement_child_snapshot = _source_snapshot(loop, stale_child.session_key)
        assert replacement_child_snapshot.state.incarnation != stale_child.source_state.incarnation

        provider.responses[0] = _tool_call("followup_child", {
            "child_id": stale_child_id,
            "content": "A stale child input.",
        }, "stale-child-followup")
        refusal = await loop.execute_turn(TurnRequest(
            content="Try the retained child reference.",
            session_key=parent_key,
        ))
        assert refusal.error is None, refusal.error
        stale_tool_event = next(
            event for event in _source_snapshot(loop, parent_key).events
            if event.get("role") == "tool" and event.get("tool_call_id") == "stale-child-followup"
        )
        assert str(stale_tool_event["content"]).startswith("Follow-up refused:")
        after_stale_child = _source_snapshot(loop, stale_child.session_key)
        assert after_stale_child.events == replacement_child_snapshot.events
        assert after_stale_child.header["metadata"].get("source_pending_followups", []) == []
        assert manager._delegation_tree.child(stale_child_id).current_run_id == old_child_run_id
        assert len(provider.child_calls) == 1

        replacement_result = await loop.execute_turn(TurnRequest(
            content="Start the replacement child.",
            session_key=parent_key,
        ))
        assert replacement_result.error is None, replacement_result.error
        current_child_id = _child_id_from_delegated_result(
            loop,
            parent_key,
            tool_call_id="delegate-replacement",
        )
        await _wait_for_child_completion(loop, current_child_id)
        current_child = manager._delegation_tree.child(current_child_id)
        assert current_child is not None

        async def capture_parent_binding(*, child_id: str, content: str, turn_context=None):
            binding = turn_context.active_tool_intent_binding if turn_context is not None else None
            captured.append((turn_context, binding))
            return await original_followup(
                child_id=child_id,
                content=content,
                turn_context=turn_context,
            )

        if original_followup is not None:
            monkeypatch.setattr(manager, "followup_child", capture_parent_binding)
        provider.responses[0] = _tool_call("followup_child", {
            "child_id": current_child_id,
            "content": "Follow-up for the replacement child.",
        }, "capture-parent-dispatch")
        valid = await loop.execute_turn(TurnRequest(
            content="Send one follow-up to the replacement child.",
            session_key=parent_key,
        ))
        assert valid.error is None, valid.error
        valid_tool_event = next(
            event for event in _source_snapshot(loop, parent_key).events
            if event.get("role") == "tool" and event.get("tool_call_id") == "capture-parent-dispatch"
        )
        assert valid_tool_event["content"] == "Follow-up applied."
        assert len(provider.child_calls) == 3
        assert captured and captured[0][0] is not None and captured[0][1] is not None
        valid_child_snapshot = _source_snapshot(loop, current_child.session_key)
        valid_child_run_id = manager._delegation_tree.child(current_child_id).current_run_id
        parent_replacement = loop.sessions.delete_session(parent_key)
        assert parent_replacement is True
        upgrade(loop, parent_key)
        replacement_parent_snapshot = _source_snapshot(loop, parent_key)

        context, binding = captured[0]
        context.active_tool_intent_binding = binding
        tool = loop.tools.get("followup_child")
        assert tool is not None
        stale_parent_result = await tool.execute_with_context(
            context,
            child_id=current_child_id,
            content="Follow-up for the replacement child.",
        )
        assert str(stale_parent_result).startswith("Follow-up refused:")
        assert _source_snapshot(loop, current_child.session_key).events == valid_child_snapshot.events
        assert _pending_followups(loop, current_child_id) == []
        assert manager._delegation_tree.child(current_child_id).current_run_id == valid_child_run_id
        assert _source_snapshot(loop, parent_key).events == replacement_parent_snapshot.events
        assert len(provider.child_calls) == 3
    finally:
        await loop.aclose()
