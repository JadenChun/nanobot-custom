from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import Session, SessionManager, SessionStorageError
from nanobot.session.runtime_authority import DurableSessionAuthority, OfflineSessionOperator
from nanobot.session.session_mode import SessionModeError
from nanobot.session.source import SourceValidationError


def test_bootstrap_then_convert_preserves_rollback_and_commits_canonical(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"
    legacy = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy")
    session.messages.append({"role": "user", "content": "hello"})
    legacy.save(session, fsync=True)
    original = legacy._get_session_path(session.key).read_bytes()

    async def exercise() -> None:
        with pytest.raises(SessionModeError, match="explicit offline bootstrap"):
            await DurableSessionAuthority(workspace, sessions_root=root).acquire()
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        assert manager.resolve_session_mode(session.key).mode == "legacy"
        manager.convert_legacy_session(session.key, offline_copy=True)
        assert manager.resolve_session_mode(session.key).mode == "canonical"
        rollback = list((manager.sessions_dir / ".session-mode" / "rollback").rglob("legacy.jsonl"))
        assert len(rollback) == 1
        assert rollback[0].read_bytes() == original
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_managed_upgrade_and_restore_are_always_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "hello"}])
    unmanaged.save(session, fsync=True)

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        snapshot = manager.read_session_snapshot(session.key)
        assert snapshot is not None
        with pytest.raises(SessionModeError, match="managed session namespace"):
            manager._jsonl_store.source.upgrade_source_session(
                session.key,
                expected_revision=snapshot.revision,
                expected_generation=snapshot.generation,
                expected_incarnation=snapshot.incarnation,
            )
        with pytest.raises(SessionModeError, match="managed session namespace"):
            manager._jsonl_store.restore_to_workspace()
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_tampered_frozen_reset_plan_fails_before_source_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=root).acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:canonical")
        async with acquire_execution(
            manager._jsonl_store,
            "cli:canonical",
            run_id="reset-test",
        ) as owner:
            await owner.quiesce()
            snapshot = owner.read_snapshot()
            plan = manager._jsonl_store.source.plan_source_reset(
                snapshot,
                [],
                updated_at="2026-09-30T00:00:00+00:00",
            )
            before = manager._get_session_path("cli:canonical").read_bytes()
            tampered = replace(plan, target_bytes=plan.target_bytes + b" ")
            with pytest.raises(SourceValidationError, match="Invalid reset plan"):
                manager._jsonl_store.source.reset_source_session(
                    "cli:canonical",
                    snapshot.state,
                    [],
                    owner=owner,
                    plan=tampered,
                )
            assert manager._get_session_path("cli:canonical").read_bytes() == before

            rows = [json.loads(line) for line in plan.target_bytes.splitlines()]
            rows[0]["metadata"]["forged"] = True
            forged_bytes = b"".join(
                (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                for row in rows
            )
            forged = replace(plan, target_bytes=forged_bytes)
            with pytest.raises(SourceValidationError, match="Invalid reset plan"):
                manager._jsonl_store.source.reset_source_session(
                    "cli:canonical",
                    snapshot.state,
                    [],
                    owner=owner,
                    plan=forged,
                )
            assert manager._get_session_path("cli:canonical").read_bytes() == before
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_manager_reset_session_uses_frozen_p4_reset_and_updates_binding(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=root).acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:canonical")
        before = manager.resolve_session_mode("cli:canonical")
        async with acquire_execution(
            manager._jsonl_store,
            "cli:canonical",
            run_id="manager-reset",
        ) as owner:
            await owner.quiesce()
            reset = manager.reset_session("cli:canonical", owner=owner)
            assert reset is not None
            assert reset.messages == []
            assert reset.generation == 1
        after = manager.resolve_session_mode("cli:canonical")
        assert after.generation == 1
        assert after.record_revision == before.record_revision + 1
        assert after.binding != before.binding
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_parse_legacy_bytes_is_strict_and_side_effect_free(tmp_path: Path) -> None:
    session = Session("cli:legacy", messages=[{"role": "user", "content": "hello"}])
    backend = SessionManager(tmp_path)._jsonl_store
    data = backend.render_legacy_bytes(session)
    before = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    parsed = backend.parse_legacy_bytes(
        data,
        expected_key=session.key,
    )
    assert parsed.key == session.key
    assert parsed.messages == session.messages
    after = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    duplicate = data.replace(b'{"_type": "metadata",', b'{"_type": "metadata", "key": "x",', 1)
    with pytest.raises(SessionStorageError):
        backend.parse_legacy_bytes(
            duplicate,
            expected_key=session.key,
        )
    final = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert final == before


@pytest.mark.parametrize(
    "conflict",
    [
        "runtime_checkpoint",
        "pending_user_turn",
        "pending_user_followups",
        "execution_lease",
        "execution_recovery_required",
        "artifact_conflict",
        "source_conflict",
        "receipt_recovery_required",
        "checkpoint_sidecar",
    ],
)
def test_conversion_admission_rejects_recovery_and_execution_conflicts(
    tmp_path: Path,
    conflict: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "exact"}])
    if conflict != "checkpoint_sidecar":
        session.metadata[conflict] = {"pending": True}
    unmanaged.save(session, fsync=True)
    if conflict in {"execution_lease", "execution_recovery_required"}:
        source = unmanaged._get_session_path(session.key)
        lines = source.read_bytes().splitlines(keepends=True)
        header = json.loads(lines[0])
        header["metadata"][conflict] = {"pending": True}
        lines[0] = (json.dumps(header, ensure_ascii=False) + "\n").encode()
        source.write_bytes(b"".join(lines))

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        if conflict == "checkpoint_sidecar":
            checkpoint_session = lease.manager.read_session_snapshot(session.key)
            checkpoint_session.metadata["runtime_checkpoint"] = {"pending": True}
            lease.manager._jsonl_store.save_runtime_checkpoint(checkpoint_session)
        before = lease.manager._get_session_path(session.key).read_bytes()
        with pytest.raises(SessionModeError, match="offline recovery"):
            lease.manager.convert_legacy_session(session.key, offline_copy=True)
        assert lease.manager.resolve_session_mode(session.key).mode == "legacy"
        assert lease.manager._get_session_path(session.key).read_bytes() == before
        assert not list(
            (lease.manager.sessions_dir / ".session-mode" / "rollback").glob("**/legacy.jsonl")
        )
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())
