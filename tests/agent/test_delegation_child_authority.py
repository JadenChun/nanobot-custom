"""Canonical delegated-child authority regressions."""

import asyncio

from nanobot.agent.child_sessions import ChildAdmission, ChildRefusal
from nanobot.agent.loop import AgentLoop, _TurnExecutionOptions
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import (
    ApprovalDecision,
    DeliveryTarget,
    RunStatus,
    TurnRequest,
    TurnSource,
)
from nanobot.bus.events import InboundMessage
from nanobot.command.builtin import cmd_stop
from nanobot.command.router import CommandContext
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.security.workspace_access import current_workspace_focus
from nanobot.session.pending_followups import mint_pending_followup_id
from nanobot.session.source import validate_execution_links
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

pytest_plugins = ("tests.agent.test_migration_source_engine",)


async def test_scoped_worker_write_uses_child_risky_policy(engine_config):
    target = engine_config.workspace_path / "protected.txt"
    original = "".join(f"protected line {index}\n" for index in range(20))
    target.write_text(original)
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="delegate-write",
            name="delegate_task",
            arguments={
                "contract": "Replace protected.txt with the supplied content.",
                "write_scope": ["protected.txt"],
            },
        )]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="write-protected",
            name="write_file",
            arguments={"path": "protected.txt", "content": "unauthorized replacement"},
        )]),
        LLMResponse(content="The parent task is complete."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    parent_backend = upgrade(loop)
    parent_key = "cli:source"
    state = parent_backend.source.load_source_snapshot(parent_key).state
    reference = loop.sessions.desktop_permissions.activate_once(
        session_key=parent_key,
        source_binding={
            "store_identity": state.store_identity,
            "incarnation": state.incarnation,
            "generation": state.generation,
        },
        task_turn_id="parent-task",
        task_input_event_id="parent-input",
        path=engine_config.workspace_path,
        file_access="read_write",
        commands=False,
    )
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Delegate the bounded write.", session_key=parent_key,
        ), _options=_TurnExecutionOptions(workspace_task_reference=reference))

        assert result.error is None, result.error
        assert target.read_text() == original
        child, = loop._fork_dependencies.foreground_manager._delegation_tree.children_of(
            "cli:source",
        )
        assert child.current_run_id is not None
        child_run = loop.run_store.load(child.current_run_id)
        assert child_run is not None
        assert child_run.status is RunStatus.APPROVAL_REQUIRED
        child_backend = loop.sessions.source_backend(child.session_key)
        assert child_backend is not None
        child_events = child_backend.source.load_source_snapshot(child.session_key).events
        parent_events = parent_backend.source.load_source_snapshot("cli:source").events
        assert not validate_execution_links(child_events)
        assert any(
            row.get("tool_call_id") == "write-protected" for row in child_events
        )
        assert not any(row.get("tool_call_id") == "write-protected" for row in parent_events)
        assert child.state == "waiting_approval", (
            child.state,
            child.outcome,
            child.current_run_id,
            child.blocked_run_id,
            child.pending_approval_id,
            child.metadata(),
            loop.approval_admission._pending,
        )
        assert child.holds_capacity is True
        assert child.pending_approval_id is not None
        assert child.blocked_run_id == child.current_run_id
        manager = loop._fork_dependencies.foreground_manager
        assert child.child_id in manager._suspended_children
        assert child.child_id in manager._child_task_references
        assert loop.sessions.desktop_permissions.is_current_task_reference(reference)
    finally:
        await loop.aclose()


async def test_original_stop_traverses_settled_child_to_resistant_grandchild(engine_config):
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]))
    parent_key = "cli:source"
    upgrade(loop, parent_key)
    manager = loop._fork_dependencies.foreground_manager
    tree = manager._delegation_tree
    assert tree is not None
    parent = loop.sessions.get_or_create(parent_key)

    settled_child = await tree.reserve(
        parent=parent,
        parent_key=parent_key,
        assignment="finish before the descendant",
        attempt_id="attempt-settled-ancestor",
    )
    assert isinstance(settled_child, ChildAdmission)
    child_parent = loop.sessions.get_or_create(settled_child.session_key)
    background_child = await tree.reserve(
        parent=child_parent,
        parent_key=settled_child.session_key,
        assignment="continue in background",
        attempt_id="attempt-background-descendant",
    )
    assert isinstance(background_child, ChildAdmission)
    completed_ancestor = await tree.complete(
        settled_child.child_id,
        outcome="ordinary child response completed",
        state="completed",
    )
    assert completed_ancestor is not None
    assert completed_ancestor.holds_capacity is False
    assert await tree.activate(background_child.child_id) is not None

    started = asyncio.Event()
    cancellation_received = asyncio.Event()
    release = asyncio.Event()

    async def resistant_background_owner():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellation_received.set()
            await release.wait()

    owner_task = asyncio.create_task(resistant_background_owner())
    await tree.attach_task(background_child.child_id, owner_task)
    await started.wait()

    capacity_parents = []
    for index in range(3):
        capacity_key = f"cli:stop-capacity-{index}"
        upgrade(loop, capacity_key)
        capacity_parent = loop.sessions.get_or_create(capacity_key)
        capacity_child = await tree.reserve(
            parent=capacity_parent,
            parent_key=capacity_key,
            assignment=f"hold gateway slot {index}",
            attempt_id=f"attempt-stop-capacity-{index}",
        )
        assert isinstance(capacity_child, ChildAdmission)
        capacity_parents.append(capacity_parent)

    message = InboundMessage(
        channel="cli",
        chat_id="source",
        sender_id="user",
        content="/stop",
    )
    stop_task = asyncio.create_task(cmd_stop(CommandContext(
        msg=message,
        session=None,
        key=parent_key,
        raw="/stop",
        loop=loop,
    )))
    try:
        await asyncio.wait_for(cancellation_received.wait(), timeout=2)

        stopping_descendant = tree.child(background_child.child_id)
        assert stopping_descendant is not None
        assert stopping_descendant.state == "stopping"
        assert stopping_descendant.holds_capacity is True
        assert tree.is_stopping(background_child.child_id)
        refused_while_held = await tree.reserve(
            parent=capacity_parents[0],
            parent_key=capacity_parents[0].key,
            assignment="wait for the stopped owner to settle",
            attempt_id="attempt-capacity-still-held",
        )
        assert isinstance(refused_while_held, ChildRefusal)
        assert refused_while_held.reason == "gateway_child_limit"

        release.set()
        await asyncio.wait_for(stop_task, timeout=2)
        stopped_descendant = tree.child(background_child.child_id)
        assert stopped_descendant is not None
        assert stopped_descendant.state == "stopped"
        assert stopped_descendant.holds_capacity is False
        assert tree.is_stopping(background_child.child_id) is False
        still_completed = tree.child(settled_child.child_id)
        assert still_completed is not None
        assert still_completed.state == "completed"
        assert still_completed.holds_capacity is False

        admitted_after_settlement = await tree.reserve(
            parent=capacity_parents[0],
            parent_key=capacity_parents[0].key,
            assignment="use the released gateway slot",
            attempt_id="attempt-capacity-released",
        )
        assert isinstance(admitted_after_settlement, ChildAdmission)
    finally:
        release.set()
        if not stop_task.done():
            await asyncio.gather(stop_task, return_exceptions=True)
        await loop.aclose()


async def test_original_stop_keeps_root_fence_until_resistant_root_settles(engine_config):
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]))
    parent_key = "cli:source"
    upgrade(loop, parent_key)
    manager = loop._fork_dependencies.foreground_manager
    tree = manager._delegation_tree
    assert tree is not None
    parent = loop.sessions.get_or_create(parent_key)
    root_started = asyncio.Event()
    cancellation_received = asyncio.Event()
    release = asyncio.Event()
    statuses = []

    class StatusPublisher:
        async def run_status_changed(self, message, session_key, status):
            statuses.append(status)

    loop.runtime_event_publisher = StatusPublisher()

    async def resistant_root_owner():
        root_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellation_received.set()
            await release.wait()

    root_task = asyncio.create_task(resistant_root_owner())
    loop._track_active_task(parent_key, root_task)
    await root_started.wait()
    message = InboundMessage(
        channel="cli",
        chat_id="source",
        sender_id="user",
        content="/stop",
    )
    stop_task = asyncio.create_task(cmd_stop(CommandContext(
        msg=message,
        session=None,
        key=parent_key,
        raw="/stop",
        loop=loop,
    )))
    try:
        await asyncio.wait_for(cancellation_received.wait(), timeout=2)
        assert statuses == ["stopping"]
        refused_while_root_held = await tree.reserve(
            parent=parent,
            parent_key=parent_key,
            assignment="do not launch while the stopped root still runs",
            attempt_id="attempt-root-stop-fence",
        )
        assert isinstance(refused_while_root_held, ChildRefusal)
        assert refused_while_root_held.reason == "session_stopping"

        release.set()
        await asyncio.wait_for(stop_task, timeout=2)
        assert root_task.done()
        assert statuses == ["stopping", "interrupted"]
        admitted_after_settlement = await tree.reserve(
            parent=parent,
            parent_key=parent_key,
            assignment="launch after the original root settles",
            attempt_id="attempt-after-root-stop",
        )
        assert isinstance(admitted_after_settlement, ChildAdmission)
    finally:
        release.set()
        if not stop_task.done():
            await asyncio.gather(stop_task, return_exceptions=True)
        await loop.aclose()


async def test_original_stop_fences_new_followup_until_root_settlement(engine_config):
    class ResistantProvider(ScriptedProvider):
        def __init__(self):
            super().__init__([])
            self.started = asyncio.Event()
            self.cancel_received = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls.append((messages, tools, model))
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancel_received.set()
                await self.release.wait()
            return LLMResponse(content="The original turn settled.")

    provider = ResistantProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    parent_key = "cli:source"
    upgrade(loop, parent_key)
    turn_task = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="Keep the original root active.",
        session_key=parent_key,
        route=DeliveryTarget(channel="cli", chat_id="source"),
    )))
    loop._track_active_task(parent_key, turn_task)
    stop_task = None
    message = InboundMessage(
        channel="cli",
        chat_id="source",
        sender_id="user",
        content="/stop",
    )
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=2)
        stop_task = asyncio.create_task(cmd_stop(CommandContext(
            msg=message,
            session=None,
            key=parent_key,
            raw="/stop",
            loop=loop,
        )))
        await asyncio.wait_for(provider.cancel_received.wait(), timeout=2)
        followup = InboundMessage(
            channel="websocket",
            chat_id="source",
            sender_id="user",
            content="This follow-up must not enter after Stop.",
        )
        outcome = await loop._admit_pending_activity(
            parent_key,
            followup,
            followup_id=mint_pending_followup_id(followup),
        )
        assert outcome.followup_pending is False

        provider.release.set()
        await asyncio.wait_for(stop_task, timeout=2)
        await asyncio.wait_for(asyncio.gather(turn_task, return_exceptions=True), timeout=2)
    finally:
        provider.release.set()
        if stop_task is not None and not stop_task.done():
            await asyncio.gather(stop_task, return_exceptions=True)
        await loop.aclose()


async def test_rejected_original_stop_leaves_followup_admission_open(engine_config):
    class WaitingProvider(ScriptedProvider):
        def __init__(self):
            super().__init__([])
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls.append((messages, tools, model))
            self.started.set()
            await self.release.wait()
            return LLMResponse(content="The current root continues.")

    provider = WaitingProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    parent_key = "cli:source"
    upgrade(loop, parent_key)
    turn_task = asyncio.create_task(loop.execute_turn(TurnRequest(
        content="Keep the current root open.",
        session_key=parent_key,
        route=DeliveryTarget(channel="cli", chat_id="source"),
    )))
    loop._track_active_task(parent_key, turn_task)
    newer_owner_release = asyncio.Event()
    newer_owner = asyncio.create_task(newer_owner_release.wait())
    try:
        await asyncio.wait_for(provider.started.wait(), timeout=2)
        stop_admission = loop._capture_original_chat_stop(parent_key, None)
        assert stop_admission is not None
        loop._active_tasks[parent_key].add(newer_owner)

        cancelled = await loop._cancel_active_tasks(
            parent_key,
            _stop_admission=stop_admission,
        )
        assert cancelled == 0
        assert not turn_task.cancelling()
        assert not newer_owner.cancelling()

        followup = InboundMessage(
            channel="websocket",
            chat_id="source",
            sender_id="user",
            content="This follow-up remains admissible after stale Stop.",
        )
        outcome = await loop._admit_pending_activity(
            parent_key,
            followup,
            followup_id=mint_pending_followup_id(followup),
        )
        assert outcome.followup_pending is True

        provider.release.set()
        newer_owner_release.set()
        await asyncio.gather(turn_task, newer_owner, return_exceptions=True)
    finally:
        provider.release.set()
        newer_owner_release.set()
        if not turn_task.done():
            await asyncio.gather(turn_task, return_exceptions=True)
        if not newer_owner.done():
            await asyncio.gather(newer_owner, return_exceptions=True)
        await loop.aclose()


async def test_old_original_stop_cannot_cancel_replacement_lifetime(engine_config):
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]))
    parent_key = "cli:source"
    upgrade(loop, parent_key)
    manager = loop._fork_dependencies.foreground_manager
    tree = manager._delegation_tree
    assert tree is not None
    stop_entered = asyncio.Event()
    continue_stop = asyncio.Event()
    statuses = []

    class StopBarrierPublisher:
        async def run_status_changed(self, message, session_key, status):
            statuses.append(status)
            if status == "stopping":
                stop_entered.set()
                await continue_stop.wait()

    loop.runtime_event_publisher = StopBarrierPublisher()
    old_owner_release = asyncio.Event()
    old_owner = asyncio.create_task(old_owner_release.wait())
    loop._track_active_task(parent_key, old_owner)
    message = InboundMessage(
        channel="cli",
        chat_id="source",
        sender_id="user",
        content="/stop",
    )
    stop_task = asyncio.create_task(cmd_stop(CommandContext(
        msg=message,
        session=None,
        key=parent_key,
        raw="/stop",
        loop=loop,
    )))
    replacement_owner_release = asyncio.Event()
    replacement_owner = None
    try:
        await asyncio.wait_for(stop_entered.wait(), timeout=2)
        assert loop.sessions.delete_session(parent_key) is True
        replacement = loop.sessions.get_or_create(parent_key)
        loop.sessions.save(replacement)
        upgrade(loop, parent_key)
        replacement = loop.sessions.get_or_create(parent_key)
        replacement_owner = asyncio.create_task(replacement_owner_release.wait())
        loop._track_active_task(parent_key, replacement_owner)

        continue_stop.set()
        result = await asyncio.wait_for(stop_task, timeout=2)
        assert result.content == "No active task to stop."
        assert statuses == ["stopping"]
        assert old_owner.cancelled() is False
        assert replacement_owner.cancelled() is False
        admitted = await tree.reserve(
            parent=replacement,
            parent_key=parent_key,
            assignment="replacement source owns new work",
            attempt_id="attempt-new-source-after-old-stop",
        )
        assert isinstance(admitted, ChildAdmission)
    finally:
        continue_stop.set()
        old_owner_release.set()
        replacement_owner_release.set()
        if not stop_task.done():
            await asyncio.gather(stop_task, return_exceptions=True)
        await asyncio.gather(old_owner, return_exceptions=True)
        if replacement_owner is not None:
            await asyncio.gather(replacement_owner, return_exceptions=True)
        await loop.aclose()


async def test_child_approval_resumes_same_source_with_new_run(engine_config):
    target = engine_config.workspace_path / "protected.txt"
    original = "".join(f"protected line {index}\n" for index in range(20))
    replacement = "approved replacement\n"
    target.write_text(original)
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="delegate-write",
            name="delegate_task",
            arguments={
                "contract": "Replace protected.txt with the supplied content.",
                "write_scope": ["protected.txt"],
            },
        )]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="write-protected",
            name="write_file",
            arguments={"path": "protected.txt", "content": replacement},
        )]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="write-protected-again",
            name="write_file",
            arguments={"path": "protected.txt", "content": replacement},
        )]),
        LLMResponse(content="The approved replacement is complete."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    parent_key = "websocket:approval-child"
    upgrade(loop, parent_key)
    route = DeliveryTarget(channel="websocket", chat_id="approval-child")
    try:
        parent_result = await loop.execute_turn(TurnRequest(
            content="Delegate the bounded write.",
            session_key=parent_key,
            route=route,
        ))
        assert parent_result.error is None, parent_result.error
        assert target.read_text() == original
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        child, = tree.children_of(parent_key)
        assert child.state == "waiting_approval", (
            child.state,
            child.outcome,
            child.current_run_id,
            child.blocked_run_id,
            child.pending_approval_id,
        )
        assert child.holds_capacity is True
        blocked_run_id = child.current_run_id
        assert blocked_run_id is not None
        assert child.pending_approval_id is not None
        pending = loop.approval_admission._pending[child.pending_approval_id]
        assert pending.session_key == child.session_key
        assert pending.run_id == blocked_run_id
        decision = ApprovalDecision(
            approval_id=pending.approval_id,
            decision="approve",
            session_key=pending.session_key,
            session_incarnation=pending.session_incarnation,
            run_id=pending.run_id,
            turn_id=pending.turn_id,
            effect_id=pending.effect_id,
            generation=pending.generation,
        )

        resumed = await loop.execute_turn(TurnRequest(
            content="approve",
            session_key=parent_key,
            route=route,
            approval_decision=decision,
        ))

        assert resumed.error is None, resumed.error
        assert resumed.status is RunStatus.COMPLETED
        assert resumed.run_id is not None and resumed.run_id != blocked_run_id
        assert target.read_text() == replacement
        settled = tree.child(child.child_id)
        assert settled is not None
        assert settled.state == "completed"
        assert settled.holds_capacity is False
        assert settled.session_key == child.session_key
        assert settled.current_run_id == resumed.run_id
        assert settled.run_sequence == child.run_sequence + 1
        blocked_record = loop.run_store.load(blocked_run_id)
        resumed_record = loop.run_store.load(resumed.run_id)
        assert blocked_record is not None and blocked_record.status is RunStatus.APPROVAL_REQUIRED
        assert resumed_record is not None and resumed_record.status is RunStatus.COMPLETED
        manager = loop._fork_dependencies.foreground_manager
        assert child.child_id not in manager._suspended_children
        assert child.child_id not in manager._child_task_references
        assert pending.approval_id not in loop.approval_admission._pending
        assert len(provider.calls) == 4
    finally:
        await loop.aclose()


async def test_sibling_child_approvals_remain_independent(engine_config):
    targets = {
        name: engine_config.workspace_path / name
        for name in ("first.txt", "second.txt")
    }
    originals = {
        name: "".join(f"protected line {index}\n" for index in range(20))
        for name in targets
    }
    replacements = {name: f"approved {name}\n" for name in targets}
    for name, path in targets.items():
        path.write_text(originals[name])

    class SiblingProvider(ScriptedProvider):
        def __init__(self):
            super().__init__([])
            self.parent_calls = 0
            self.child_calls = 0

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls.append((messages, tools, model))
            is_child = any("Worker Role" in str(message.get("content")) for message in messages)
            if not is_child:
                self.parent_calls += 1
                name = "first.txt" if self.parent_calls == 1 else "second.txt"
                return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id=f"delegate-{name}",
                    name="delegate_task",
                    arguments={
                        "contract": f"Replace {name} with the approved content.",
                        "write_scope": [name],
                    },
                )])
            self.child_calls += 1
            if self.child_calls in {1, 2, 3, 5}:
                name = {
                    1: "first.txt",
                    2: "second.txt",
                    3: "first.txt",
                    5: "second.txt",
                }[self.child_calls]
                return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id=f"write-{self.child_calls}",
                    name="write_file",
                    arguments={"path": name, "content": replacements[name]},
                )])
            return LLMResponse(content="Approved child write completed.")

    provider = SiblingProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    parent_key = "websocket:sibling-approvals"
    upgrade(loop, parent_key)
    route = DeliveryTarget(channel="websocket", chat_id="sibling-approvals")
    try:
        first_result = await loop.execute_turn(TurnRequest(
            content="Start the first worker.", session_key=parent_key, route=route,
        ))
        second_result = await loop.execute_turn(TurnRequest(
            content="Start the second worker.", session_key=parent_key, route=route,
        ))
        assert first_result.error is None and second_result.error is None

        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        first, second = tree.children_of(parent_key)
        assert first.state == second.state == "waiting_approval"
        assert first.holds_capacity and second.holds_capacity
        first_pending = loop.approval_admission._pending[first.pending_approval_id]
        second_pending = loop.approval_admission._pending[second.pending_approval_id]
        assert first_pending.approval_id != second_pending.approval_id
        assert first_pending.session_key == first.session_key
        assert second_pending.session_key == second.session_key
        assert first_pending.approval_id in loop.approval_admission._aliases[
            "websocket:sibling-approvals"
        ]
        assert second_pending.approval_id in loop.approval_admission._aliases[
            "websocket:sibling-approvals"
        ]

        first_decision = ApprovalDecision(
            approval_id=first_pending.approval_id,
            decision="approve",
            run_id=first_pending.run_id,
            effect_id=first_pending.effect_id,
        )
        first_run_id = first.current_run_id
        first_resumed = await loop.execute_turn(TurnRequest(
            content="Approve only the first child.",
            session_key=parent_key,
            route=route,
            approval_decision=first_decision,
        ))
        assert first_resumed.error is None
        assert first_resumed.run_id != first_run_id
        assert targets["first.txt"].read_text() == replacements["first.txt"]
        assert targets["second.txt"].read_text() == originals["second.txt"]
        assert first_pending.approval_id not in loop.approval_admission._pending
        assert second_pending.approval_id in loop.approval_admission._pending
        before_duplicate = len(provider.calls)
        duplicate = await loop.execute_turn(TurnRequest(
            content="Repeat the first decision.",
            session_key=parent_key,
            route=route,
            approval_decision=first_decision,
        ))
        assert duplicate.stop_reason == "approval_stale"
        assert len(provider.calls) == before_duplicate
        assert second_pending.approval_id in loop.approval_admission._pending

        second_decision = ApprovalDecision(
            approval_id=second_pending.approval_id,
            decision="approve",
            run_id=second_pending.run_id,
            effect_id=second_pending.effect_id,
        )
        second_resumed = await loop.execute_turn(TurnRequest(
            content="Approve the second child.",
            session_key=parent_key,
            route=route,
            approval_decision=second_decision,
        ))
        assert second_resumed.error is None
        assert second_resumed.run_id != second.current_run_id
        assert targets["second.txt"].read_text() == replacements["second.txt"]
        assert all(child.state == "completed" and not child.holds_capacity for child in (
            tree.child(first.child_id),
            tree.child(second.child_id),
        ))
    finally:
        await loop.aclose()


async def test_background_child_retains_parent_task_authority_until_settled(engine_config):
    class BarrierProvider(ScriptedProvider):
        def __init__(self):
            super().__init__([
                LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id="delegate-background",
                    name="delegate_task",
                    arguments={
                        "contract": "Inspect the supplied task context and report when released.",
                        "write_scope": ["unused-output.txt"],
                        "background": True,
                    },
                )]),
                LLMResponse(content="Parent response completed."),
            ])
            self.child_started = asyncio.Event()
            self.child_release = asyncio.Event()
            self.child_task_reference_is_live = False

        async def chat(self, messages, tools=None, model=None, **kwargs):
            if any("Worker Role" in str(message.get("content")) for message in messages):
                focus = current_workspace_focus()
                assert focus is not None
                assert focus.task_reference is reference
                assert focus.permission_owner is loop.sessions.desktop_permissions
                self.child_task_reference_is_live = focus.permission_owner.is_current_task_reference(
                    reference,
                )
                self.calls.append((messages, tools, model))
                self.child_started.set()
                await self.child_release.wait()
                return LLMResponse(content="Background child completed.")
            return await super().chat(messages, tools=tools, model=model, **kwargs)

    provider = BarrierProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "websocket:permission-child"
    parent_backend = upgrade(loop, key)
    state = parent_backend.source.load_source_snapshot(key).state
    source_binding = {
        "store_identity": state.store_identity,
        "incarnation": state.incarnation,
        "generation": state.generation,
    }
    reference = loop.sessions.desktop_permissions.activate_once(
        session_key=key,
        source_binding=source_binding,
        task_turn_id="parent-task",
        task_input_event_id="parent-input",
        path=engine_config.workspace_path,
        file_access="read_write",
        commands=False,
    )
    try:
        result = await loop.execute_turn(
            TurnRequest(
                content="Start a background child.",
                source=TurnSource.GATEWAY,
                session_key=key,
                route=DeliveryTarget(channel="websocket", chat_id="permission-child"),
            ),
            _options=_TurnExecutionOptions(workspace_task_reference=reference),
        )

        assert result.error is None, result.error
        assert result.content == "Parent response completed."
        await asyncio.wait_for(provider.child_started.wait(), timeout=2)
        assert provider.child_task_reference_is_live
        assert loop.sessions.desktop_permissions.is_current_task_reference(reference)

        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        child, = tree.children_of(key)
        assert child.state == "active"
        provider.child_release.set()
        settled = await tree.wait_child(child.child_id, timeout=2)
        assert settled is not None and settled.state == "completed"
        assert not loop.sessions.desktop_permissions.is_current_task_reference(reference)
    finally:
        provider.child_release.set()
        await loop.aclose()
