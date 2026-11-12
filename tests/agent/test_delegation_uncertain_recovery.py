"""Explicit owner-evidence settlement for uncertain delegated children."""


from nanobot.agent.child_sessions import ChildAdmission, DelegationTree
from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import (
    RunRecord,
    RunStatus,
    SessionRunRef,
    TurnSource,
)
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.turn_writer import SourceTurnWriter
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

pytest_plugins = ("tests.agent.test_migration_source_engine",)


async def _uncertain_child(engine_config, *, complete: bool = True):
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
    )
    upgrade(loop, "cli:source")
    tree = loop._fork_dependencies.foreground_manager._delegation_tree
    assert tree is not None
    parent = loop.sessions.get_or_create("cli:source")
    admission = await tree.reserve(
        parent=parent,
        parent_key="cli:source",
        assignment="settle the uncertain child",
        attempt_id="parent-run",
    )
    assert isinstance(admission, ChildAdmission)
    await tree.activate(admission.child_id)
    await tree.register_run(admission.child_id, "child-run")
    backend = loop.sessions.source_backend(admission.session_key)
    assert backend is not None
    async with acquire_execution(
        backend,
        admission.session_key,
        run_id="child-run",
    ) as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="child-run")
        writer.append_message({"role": "user", "content": "assignment"})
        if complete:
            writer.append_message({"role": "assistant", "content": "result"})
        stream_id = writer.stream_id
    loop.run_store.save(RunRecord(
        run_id="child-run",
        status=RunStatus.COMPLETED,
        source=TurnSource.DIRECT,
        session_ref=SessionRunRef(
            session_key=admission.session_key,
            run_id="child-run",
        ),
        stop_reason="completed",
        completed_at="2026-11-12T00:00:00+00:00",
        metadata={"source_stream_id": stream_id},
    ))
    restarted_tree = DelegationTree(loop.sessions)
    restored = restarted_tree.child(admission.child_id)
    assert restored is not None and restored.state == "uncertain"
    manager = ForegroundAgentManager(
        provider=loop._fork_dependencies.foreground_manager.provider,
        workspace=loop.workspace,
        sessions=loop.sessions,
        delegation_tree=restarted_tree,
    )
    return loop, manager, restarted_tree, admission


async def test_completed_run_settles_uncertain_child(engine_config):
    loop, manager, tree, admission = await _uncertain_child(engine_config)
    try:
        settled = await manager.settle_uncertain_child(
            admission.child_id,
            run_store=loop.run_store,
        )
        assert settled is not None
        assert settled.state == "completed"
        assert settled.outcome == "recovered_completed"
        assert settled.holds_capacity is False
        assert settled.terminal is True
    finally:
        await manager.close()
        await loop.aclose()


async def test_mismatched_run_evidence_keeps_uncertain(engine_config):
    loop, manager, tree, admission = await _uncertain_child(engine_config)
    try:
        loop.run_store.save(RunRecord(
            run_id="child-run",
            status=RunStatus.COMPLETED,
            source=TurnSource.DIRECT,
            session_ref=SessionRunRef(
                session_key="delegated:another-child",
                run_id="child-run",
            ),
            stop_reason="completed",
            completed_at="2026-11-12T00:00:00+00:00",
        ))
        settled = await manager.settle_uncertain_child(
            admission.child_id,
            run_store=loop.run_store,
        )
        assert settled is None
        current = tree.child(admission.child_id)
        assert current is not None and current.state == "uncertain"
    finally:
        await manager.close()
        await loop.aclose()


async def test_missing_terminal_response_keeps_uncertain(engine_config):
    loop, manager, tree, admission = await _uncertain_child(engine_config, complete=False)
    try:
        settled = await manager.settle_uncertain_child(
            admission.child_id,
            run_store=loop.run_store,
        )
        assert settled is None
        current = tree.child(admission.child_id)
        assert current is not None
        assert current.state == "uncertain"
        assert current.holds_capacity is True
    finally:
        await manager.close()
        await loop.aclose()
