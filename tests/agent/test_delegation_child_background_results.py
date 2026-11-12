"""Durable delivery of background child results (A3-5 AC5)."""

import asyncio
from unittest.mock import AsyncMock

from nanobot.agent.child_sessions import DelegationTree
from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.turn import RunStatus
from nanobot.providers.base import LLMResponse
from nanobot.session.pending_followups import decode_pending_followups
from nanobot.session.source_store import SourceSessionStore
from tests.agent.test_delegation_child_followups import (
    _FollowupProvider,
    _make_loop,
    _start_background_child,
    _tool_call,
    _wait_for_child_completion,
)

pytest_plugins = ("tests.agent.test_migration_source_engine",)


def _background_provider(*, gate: bool = False) -> _FollowupProvider:
    return _FollowupProvider(
        [
            _tool_call(
                "delegate_task",
                {
                    "contract": "Finish the bounded offline fixture task.",
                    "write_scope": ["fixture.txt"],
                    "background": True,
                },
                "delegate",
            ),
            LLMResponse(content="The child is running."),
        ],
        child_responses=[LLMResponse(content="child final result")],
        gate_first_child=gate,
    )


async def _start_without_delivery(engine_config, parent_key, *, gate: bool = False):
    provider = _background_provider(gate=gate)
    loop = _make_loop(engine_config, provider, parent_key)
    manager = loop._fork_dependencies.foreground_manager
    original_reconcile = manager.reconcile_background_results
    manager.reconcile_background_results = AsyncMock(return_value=(0, 0, 0))
    child_id = await _start_background_child(loop, parent_key)
    if not gate:
        await _wait_for_child_completion(loop, child_id)
    return loop, manager, provider, child_id, original_reconcile


async def _replace_parent_source(loop, parent_key):
    loop.sessions.delete_session(parent_key)
    replacement = loop.sessions.get_or_create(parent_key)
    loop.sessions.save(replacement)
    backend = loop.sessions._jsonl_store
    backend.source.upgrade_source_session(
        parent_key,
        expected_revision=replacement.revision,
        expected_generation=replacement.generation,
        expected_incarnation=replacement.incarnation,
    )
    loop.sessions.invalidate(parent_key)
    return backend


async def _wait_admitted(tree, child_id):
    for _ in range(200):
        child = tree.child(child_id)
        if child is not None and child.background_result is not None:
            if child.background_result.state == "admitted":
                return child.background_result
        await asyncio.sleep(0.01)
    raise AssertionError("background result was not admitted")


async def _collect(target, message):
    target.append(message)


async def test_background_result_is_admitted_once(engine_config):
    parent_key = "websocket:ac5-probe"
    loop = _make_loop(engine_config, _background_provider(), parent_key)
    try:
        child_id = await _start_background_child(loop, parent_key)
        await _wait_for_child_completion(loop, child_id)
        manager = loop._fork_dependencies.foreground_manager
        tree = manager._delegation_tree
        assert tree is not None
        child = tree.child(child_id)
        assert child is not None and child.holds_capacity is False
        record = await _wait_admitted(tree, child_id)
        assert record.result == "child final result"
        assert record.journal_id == record.outbox_id

        backend = loop.sessions.source_backend(parent_key)
        assert backend is not None
        snapshot = backend.source.load_source_snapshot(parent_key)
        rows = snapshot.header["metadata"].get("source_pending_followups", [])
        assert len(rows) == 1, rows
        assert rows[0]["background_result"]["outbox_id"] == record.outbox_id

        second = await manager.reconcile_background_results()
        assert second == (0, 0, 0), second
        assert len(
            snapshot.header["metadata"].get("source_pending_followups", [])
        ) == 1

        messages = decode_pending_followups(rows, parent_key)
        assert len(messages) == 1
        message = messages[0]
        assert message.channel == "websocket"
        assert message.chat_id == "ac5-probe"
        assert "child final result" in message.content
        assert message.session_key_override == parent_key
    finally:
        await loop.aclose()


async def test_background_result_preserves_cli_route(engine_config):
    parent_key = "cli:ac5-cli"
    loop, manager, _provider, child_id, original = await _start_without_delivery(
        engine_config, parent_key,
    )
    try:
        result = await original()
        assert result == (1, 0, 0), result
        tree = manager._delegation_tree
        assert tree is not None
        child = tree.child(child_id)
        assert child is not None and child.background_result is not None
        assert child.background_result.route_channel == "cli"
        assert child.background_result.route_chat_id == "ac5-cli"
        backend = loop.sessions.source_backend(parent_key)
        assert backend is not None
        snapshot = backend.source.load_source_snapshot(parent_key)
        rows = snapshot.header["metadata"].get("source_pending_followups", [])
        assert len(rows) == 1
        messages = decode_pending_followups(rows, parent_key)
        assert len(messages) == 1
        assert messages[0].channel == "cli"
        assert messages[0].chat_id == "ac5-cli"
    finally:
        await loop.aclose()


async def test_shared_tree_reconciles_result_once(engine_config):
    parent_key = "websocket:ac5-shared"
    loop, manager, _provider, child_id, original = await _start_without_delivery(
        engine_config, parent_key,
    )
    try:
        tree = manager._delegation_tree
        assert tree is not None
        second = ForegroundAgentManager(
            provider=manager.provider,
            workspace=loop.workspace,
            sessions=loop.sessions,
            delegation_tree=tree,
        )
        published = []
        manager.set_background_result_sink(
            lambda message: _collect(published, message),
        )
        second.set_background_result_sink(
            lambda message: _collect(published, message),
        )
        results = await asyncio.gather(
            original(),
            second.reconcile_background_results(),
        )
        assert sum(result[0] for result in results) == 1, results
        assert len(published) == 1
        backend = loop.sessions.source_backend(parent_key)
        assert backend is not None
        snapshot = backend.source.load_source_snapshot(parent_key)
        rows = snapshot.header["metadata"].get("source_pending_followups", [])
        assert len(rows) == 1
        await second.close()
    finally:
        await loop.aclose()


async def test_background_result_survives_restart_and_admits_once(engine_config):
    parent_key = "websocket:ac5-restart"
    loop, _manager, _provider, child_id, _original = await _start_without_delivery(
        engine_config, parent_key,
    )
    try:
        tree = DelegationTree(loop.sessions)
        restarted = ForegroundAgentManager(
            provider=loop._fork_dependencies.foreground_manager.provider,
            workspace=loop.workspace,
            sessions=loop.sessions,
            delegation_tree=tree,
        )
        published = []
        restarted.set_background_result_sink(
            lambda message: _collect(published, message),
        )
        try:
            first = await restarted.reconcile_background_results()
            assert first == (1, 0, 0), first
            child = tree.child(child_id)
            assert child is not None and child.background_result is not None
            assert child.background_result.state == "admitted"
            assert len(published) == 1
            assert "child final result" in published[0].content
            backend = loop.sessions.source_backend(parent_key)
            assert backend is not None
            snapshot = backend.source.load_source_snapshot(parent_key)
            rows = snapshot.header["metadata"].get("source_pending_followups", [])
            assert len(rows) == 1
            second = await restarted.reconcile_background_results()
            assert second == (0, 0, 0), second
            assert len(
                snapshot.header["metadata"]["source_pending_followups"]
            ) == 1
        finally:
            await restarted.close()
    finally:
        await loop.aclose()


async def test_background_result_for_replaced_parent_is_stale(engine_config):
    parent_key = "websocket:ac5-stale"
    loop, manager, _provider, child_id, original = await _start_without_delivery(
        engine_config, parent_key,
    )
    try:
        tree = manager._delegation_tree
        assert tree is not None
        before = tree.child(child_id)
        assert before is not None and before.current_run_id is not None
        child_run_id = before.current_run_id
        backend = await _replace_parent_source(loop, parent_key)
        replacement_state = backend.source.load_source_snapshot(parent_key).state
        result = await original()
        assert result == (0, 1, 0), result
        child = tree.child(child_id)
        assert child is not None and child.background_result is not None
        assert child.background_result.state == "stale"
        assert child.background_result.stale_reason == "parent_source_changed"
        assert child.current_run_id == child_run_id
        run = loop.run_store.load(child_run_id)
        assert run is not None and run.status is RunStatus.COMPLETED
        snapshot = backend.source.load_source_snapshot(parent_key)
        assert snapshot.state.same_origin(replacement_state)
        assert "source_pending_followups" not in snapshot.header["metadata"]
    finally:
        await loop.aclose()


async def test_background_result_for_deleted_parent_is_stale(engine_config):
    parent_key = "websocket:ac5-deleted"
    loop, manager, _provider, child_id, original = await _start_without_delivery(
        engine_config, parent_key,
    )
    try:
        loop.sessions.delete_session(parent_key)
        result = await original()
        assert result == (0, 1, 0), result
        tree = manager._delegation_tree
        assert tree is not None
        child = tree.child(child_id)
        assert child is not None and child.background_result is not None
        assert child.background_result.state == "stale"
        assert child.background_result.stale_reason == "parent_unavailable"
    finally:
        await loop.aclose()


async def test_parent_replacement_before_child_finishes_marks_result_stale(engine_config):
    parent_key = "websocket:ac5-stale-running"
    loop, _manager, provider, child_id, _original = await _start_without_delivery(
        engine_config, parent_key, gate=True,
    )
    try:
        await asyncio.wait_for(provider.child_started.wait(), timeout=3)
        await _replace_parent_source(loop, parent_key)
        provider.child_release.set()
        await _wait_for_child_completion(loop, child_id)
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        assert tree is not None
        child = tree.child(child_id)
        assert child is not None and child.background_result is not None
        assert child.background_result.state == "stale"
        assert child.background_result.stale_reason == "parent_source_changed"
    finally:
        provider.child_release.set()
        await loop.aclose()


async def test_missing_result_is_rebuilt_from_completed_run(engine_config):
    parent_key = "websocket:ac5-recover"
    provider = _background_provider()
    loop = _make_loop(engine_config, provider, parent_key)
    try:
        manager = loop._fork_dependencies.foreground_manager
        manager.record_background_result = AsyncMock(return_value=True)
        manager.reconcile_background_results = AsyncMock(return_value=(0, 0, 0))
        child_id = await _start_background_child(loop, parent_key)
        await _wait_for_child_completion(loop, child_id)
        tree = manager._delegation_tree
        assert tree is not None
        child = tree.child(child_id)
        assert child is not None
        assert child.background_delivery is not None
        assert child.background_result is None

        restarted_tree = DelegationTree(loop.sessions)
        restarted = ForegroundAgentManager(
            provider=manager.provider,
            workspace=loop.workspace,
            sessions=loop.sessions,
            delegation_tree=restarted_tree,
        )
        try:
            recovered = await restarted.recover_missing_background_results(
                loop.run_store,
            )
            assert recovered == 1
            child = restarted_tree.child(child_id)
            assert child is not None and child.background_result is not None
            assert child.background_result.result == "child final result"
            assert child.background_result.state == "pending"
        finally:
            await restarted.close()
    finally:
        await loop.aclose()


async def test_transient_parent_journal_failure_stays_pending(engine_config, monkeypatch):
    parent_key = "websocket:ac5-transient"
    loop, manager, _provider, child_id, original = await _start_without_delivery(
        engine_config, parent_key,
    )
    tree = manager._delegation_tree
    assert tree is not None
    calls = 0
    admit = SourceSessionStore.admit_pending_followup

    def fail_once(self, key, owner, record):
        nonlocal calls
        if calls == 0:
            calls += 1
            raise OSError("temporary parent journal failure")
        return admit(self, key, owner, record)

    monkeypatch.setattr(SourceSessionStore, "admit_pending_followup", fail_once)
    try:
        first = await original()
        assert first == (0, 0, 1), first
        child = tree.child(child_id)
        assert child is not None and child.background_result is not None
        assert child.background_result.state == "pending"

        second = await original()
        assert second == (1, 0, 0), second
        child = tree.child(child_id)
        assert child is not None and child.background_result is not None
        assert child.background_result.state == "admitted"
    finally:
        await loop.aclose()


async def test_background_result_consumes_into_parent_turn_once(engine_config):
    parent_key = "websocket:ac5-consume"
    provider = _FollowupProvider(
        [
            _tool_call(
                "delegate_task",
                {
                    "contract": "Finish the bounded offline fixture task.",
                    "write_scope": ["fixture.txt"],
                    "background": True,
                },
                "delegate",
            ),
            LLMResponse(content="The child is running."),
            LLMResponse(content="The parent received the child result."),
        ],
        child_responses=[LLMResponse(content="child final result")],
    )
    loop = _make_loop(engine_config, provider, parent_key)
    try:
        child_id = await _start_background_child(loop, parent_key)
        await _wait_for_child_completion(loop, child_id)
        manager = loop._fork_dependencies.foreground_manager
        tree = manager._delegation_tree
        assert tree is not None
        record = await _wait_admitted(tree, child_id)
        outbox_id = record.outbox_id

        run_task = asyncio.create_task(loop.run())
        try:
            async def consumed():
                backend = loop.sessions.source_backend(parent_key)
                assert backend is not None
                snapshot = backend.source.load_source_snapshot(parent_key)
                return any(
                    event.get("_pending_followup_id") == outbox_id
                    for event in snapshot.events
                )

            for _ in range(200):
                if await consumed():
                    break
                await asyncio.sleep(0.01)
            assert await consumed()
            backend = loop.sessions.source_backend(parent_key)
            assert backend is not None
            snapshot = backend.source.load_source_snapshot(parent_key)
            assert "source_pending_followups" not in snapshot.header["metadata"]
            event = next(
                event for event in snapshot.events
                if event.get("_pending_followup_id") == outbox_id
            )
            assert event.get("_background_result") is not None
            assert "Background child" in str(event.get("content"))
            assert "child final result" in str(event.get("content"))
        finally:
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
    finally:
        await loop.aclose()
