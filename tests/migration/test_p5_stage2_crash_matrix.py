from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import DurableSessionAuthority, OfflineSessionOperator
from nanobot.session.session_mode import SessionModeError


@pytest.mark.parametrize("phase", ["master_prepared", "witness_preparing", "records_committed", "registry_active"])
@pytest.mark.parametrize("change", ["none", "source", "sidecar", "sidecar_absent"])
def test_bootstrap_pair_resume_requires_exact_bytes(tmp_path: Path, phase: str, change: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "exact"}])
    unmanaged.save(session, fsync=True)
    session.metadata["runtime_checkpoint"] = {"pending": True}
    unmanaged._jsonl_store.save_runtime_checkpoint(session)
    source = unmanaged._get_session_path(session.key)
    sidecar = unmanaged._jsonl_store.get_runtime_checkpoint_path(session.key)
    original = source.read_bytes(), sidecar.read_bytes()

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        with pytest.raises(RuntimeError, match="injected"):
            first.manager._mode_resolver.bootstrap(offline_copy=True, fault_after=phase)
        await first.close()
        if change == "source":
            source.write_bytes(source.read_bytes().replace(b"exact", b"other"))
        elif change == "sidecar":
            value = json.loads(sidecar.read_bytes())
            value["checkpoint"] = {"changed": True}
            sidecar.write_bytes(json.dumps(value).encode())
        elif change == "sidecar_absent":
            sidecar.unlink()
        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        if change == "none":
            assert second.manager.bootstrap_existing_sessions(offline_copy=True) == 1
            assert (source.read_bytes(), sidecar.read_bytes()) == original
        else:
            before = source.read_bytes(), sidecar.read_bytes() if sidecar.exists() else None
            with pytest.raises(SessionModeError):
                second.manager.bootstrap_existing_sessions(offline_copy=True)
            assert (source.read_bytes(), sidecar.read_bytes() if sidecar.exists() else None) == before
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["prepared", "source_installed", "sidecar_removed", "target_installed", "mode_record_published", "mode_committed"])
@pytest.mark.parametrize("change", ["none", "source", "sidecar"])
def test_legacy_reset_pair_fault_matrix(tmp_path: Path, phase: str, change: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "exact"}])
    unmanaged.save(session, fsync=True)
    session.metadata["runtime_checkpoint"] = {"pending": True}
    unmanaged._jsonl_store.save_runtime_checkpoint(session)
    source = unmanaged._get_session_path(session.key)
    sidecar = unmanaged._jsonl_store.get_runtime_checkpoint_path(session.key)
    original = source.read_bytes(), sidecar.read_bytes()

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        first.manager.bootstrap_existing_sessions(offline_copy=True)
        with pytest.raises(RuntimeError, match="injected"):
            first.manager._mode_resolver.reset(session.key, [], owner=None, fault_after=phase)
        await first.close()
        if change == "source":
            source.write_bytes(source.read_bytes() + b" ")
        elif change == "sidecar":
            sidecar.write_bytes(b'{"changed":true}\n')
            sidecar.chmod(0o600)
        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        if change == "none":
            resolved = second.manager.recover_session_mode(session.key)
            assert resolved.generation == int(phase != "prepared")
            if phase == "prepared":
                assert (source.read_bytes(), sidecar.read_bytes()) == original
            else:
                assert not sidecar.exists()
                assert second.manager.read_session_snapshot(session.key).messages == []
            assert second.manager.recover_session_mode(session.key) == resolved
        else:
            before = source.read_bytes(), sidecar.read_bytes() if sidecar.exists() else None
            with pytest.raises(SessionModeError):
                second.manager.recover_session_mode(session.key)
            assert (source.read_bytes(), sidecar.read_bytes() if sidecar.exists() else None) == before
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["prepared", "staged", "source_installed", "target_installed", "mode_record_published", "mode_committed"])
@pytest.mark.parametrize("change", ["none", "source", "record", "sidecar", "target_sidecar", "stage", "target", "target_record"])
def test_fork_exact_source_fault_matrix(tmp_path: Path, phase: str, change: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        first.manager.bootstrap_existing_sessions(offline_copy=True)
        first.manager.admit_persistent_write("cli:source")
        resolver = first.manager._mode_resolver
        source_path = first.manager._get_session_path("cli:source")
        source = source_path.read_bytes()
        record = resolver.storage.read(resolver._record_name("cli:source"))
        with pytest.raises(RuntimeError, match="injected"):
            resolver.fork("cli:source", "cli:target", 0, fault_after=phase)
        prepared = next(
            resolver._journal_records(op)[0] for op in resolver.storage.list(".session-mode/journals")
            if resolver._journal_records(op)[0]["operation"] == "fork"
        )
        assert prepared["source_record"].encode() == record
        assert prepared["source_record_digest"] == hashlib.sha256(record).hexdigest()
        assert prepared["source_input_digest"] == hashlib.sha256(source).hexdigest()
        assert prepared["source_input_size"] == len(source)
        assert prepared["source_binding"]["session_key"] == "cli:source"
        assert prepared["sidecar_bytes"] is None
        if change == "source":
            resolver.storage.write(f"{resolver.backend.storage_key('cli:source')}.jsonl", source + b" ")
        elif change == "record":
            resolver.storage.write(resolver._record_name("cli:source"), record + b" ")
        elif change in {"sidecar", "target_sidecar"}:
            key = "cli:source" if change == "sidecar" else "cli:target"
            resolver.storage.write(f"{resolver.backend.storage_key(key)}.checkpoint.json", b"{}\n")
        elif change == "stage":
            resolver.storage.write(f".session-mode/{prepared['staging_name']}", b"changed\n")
        elif change == "target":
            resolver.storage.write(f"{resolver.backend.storage_key('cli:target')}.jsonl", b"changed\n")
        elif change == "target_record":
            resolver.storage.write(resolver._record_name("cli:target"), b"{}\n")
        await first.close()
        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        if change == "none":
            resolved = second.manager.recover_session_mode("cli:target")
            assert resolved.mode == ("canonical" if phase in {"mode_record_published", "mode_committed"} else None)
            assert source_path.read_bytes() == source
            assert second.manager.recover_session_mode("cli:target") == resolved
        else:
            with pytest.raises(SessionModeError):
                second.manager.recover_session_mode("cli:target")
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["create", "convert", "fork", "export"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_prepared_precedes_every_operation_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, when: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    if operation == "convert":
        unmanaged = SessionManager(workspace, sessions_root=root)
        unmanaged.save(Session("cli:source", messages=[]), fsync=True)

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = first.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        if operation in {"fork", "export"}:
            manager.admit_persistent_write("cli:source")
        resolver = manager._mode_resolver
        append = resolver._append_journal

        def fault(op: str, **fields: object) -> None:
            if fields.get("phase") == "prepared" and fields.get("operation") == operation:
                assert not resolver.storage.exists(f".session-mode/staging/{op}")
                assert not resolver.storage.exists(f".session-mode/exports/{op}")
                assert not list((manager.sessions_dir / ".session-mode" / "rollback").glob("**/legacy.jsonl"))
                if when == "after":
                    append(op, **fields)
                raise RuntimeError("prepared-boundary fault")
            append(op, **fields)

        monkeypatch.setattr(resolver, "_append_journal", fault)
        calls = {
            "create": lambda: resolver.ensure_canonical("cli:target"),
            "convert": lambda: resolver.convert("cli:source", offline_copy=True),
            "fork": lambda: resolver.fork("cli:source", "cli:target", 0),
            "export": lambda: resolver.begin_export("cli:source", tmp_path / "exported", {"manifest.json": b"{}\n"}),
        }
        with pytest.raises(RuntimeError, match="prepared-boundary"):
            calls[operation]()
        assert not list((manager.sessions_dir / ".session-mode" / "staging").glob("**/*.jsonl"))
        assert not list((manager.sessions_dir / ".session-mode" / "exports").glob("**/*.*"))
        assert not list((manager.sessions_dir / ".session-mode" / "rollback").glob("**/legacy.jsonl"))
        await first.close()
        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        key = "cli:target" if operation in {"create", "fork"} else "cli:source"
        result = second.manager.recover_session_mode(key)
        assert result.mode == ("legacy" if operation == "convert" else "canonical" if operation == "export" else None)
        assert not (tmp_path / "exported").exists()
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["create", "convert", "reset", "fork"])
def test_canonical_producer_rechecks_sidecar_before_record_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    if operation == "convert":
        unmanaged = SessionManager(workspace, sessions_root=root)
        unmanaged.save(Session("cli:source"), fsync=True)

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = first.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        if operation in {"fork", "reset"}:
            manager.admit_persistent_write("cli:source")
        resolver = manager._mode_resolver
        key = "cli:target" if operation in {"create", "fork"} else "cli:source"
        record_name = resolver._record_name(key)
        predecessor = resolver.storage.read(record_name) if resolver.storage.exists(record_name) else None
        append = resolver._append_journal

        def introduce(op: str, **fields: object) -> None:
            append(op, **fields)
            if fields.get("phase") == "target_installed" and fields.get("operation") == operation:
                resolver.storage.write(f"{resolver.backend.storage_key(key)}.checkpoint.json", b"{}\n")

        monkeypatch.setattr(resolver, "_append_journal", introduce)
        calls = {
            "create": lambda: resolver.ensure_canonical(key),
            "convert": lambda: resolver.convert(key, offline_copy=True),
            "reset": lambda: resolver.reset(key, [], owner=None),
            "fork": lambda: resolver.fork("cli:source", key, 0),
        }
        with pytest.raises(SessionModeError):
            calls[operation]()
        assert (resolver.storage.read(record_name) if resolver.storage.exists(record_name) else None) == predecessor
        await first.close()
        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        with pytest.raises(SessionModeError):
            second.manager.recover_session_mode(key)
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["prepared", "target_installed"])
def test_conversion_precommit_crash_restores_exact_legacy(tmp_path: Path, phase: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "exact"}])
    unmanaged.save(session, fsync=True)
    original = unmanaged._get_session_path(session.key).read_bytes()

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        with pytest.raises(RuntimeError, match="injected"):
            manager._mode_resolver.convert(session.key, offline_copy=True, fault_after=phase)
        with pytest.raises(RuntimeError, match="offline recovery"):
            manager.resolve_session_mode(session.key)
        assert manager.recover_session_mode(session.key).mode == "legacy"
        assert manager._get_session_path(session.key).read_bytes() == original
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_mode_record_publication_crash_completes_canonical_forward(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "exact"}])
    unmanaged.save(session, fsync=True)

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        with pytest.raises(RuntimeError, match="injected"):
            manager._mode_resolver.convert(
                session.key,
                offline_copy=True,
                fault_after="mode_record_published",
            )
        with pytest.raises(RuntimeError, match="offline recovery"):
            manager.resolve_session_mode(session.key)
        assert manager.recover_session_mode(session.key).mode == "canonical"
        assert manager.source_backend(session.key) is manager._jsonl_store
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("operation", "phase", "expected_mode"),
    [
        ("create", "prepared", None),
        ("create", "target_installed", None),
        ("create", "mode_record_published", "canonical"),
        ("convert", "prepared", "legacy"),
        ("convert", "target_installed", "legacy"),
        ("convert", "mode_record_published", "canonical"),
    ],
)
def test_crash_recovery_uses_fresh_operator(
    tmp_path: Path,
    operation: str,
    phase: str,
    expected_mode: str | None,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"

    if operation == "convert":
        unmanaged = SessionManager(workspace, sessions_root=root)
        unmanaged.save(
            Session("cli:item", messages=[{"role": "user", "content": "exact"}]),
            fsync=True,
        )

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        if operation == "convert":
            first.manager.bootstrap_existing_sessions(offline_copy=True)
            invoke = lambda: first.manager._mode_resolver.convert(  # noqa: E731
                "cli:item", offline_copy=True, fault_after=phase
            )
        else:
            first.manager.bootstrap_existing_sessions(offline_copy=True)
            invoke = lambda: first.manager._mode_resolver.ensure_canonical(  # noqa: E731
                "cli:item", fault_after=phase
            )
        with pytest.raises(RuntimeError, match="injected"):
            invoke()
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        resolved = second.manager.recover_session_mode("cli:item")
        assert resolved.mode == expected_mode
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("phase", "expected_generation"),
    [
        ("prepared", 0),
        ("target_installed", 1),
        ("mode_record_published", 1),
    ],
)
def test_canonical_reset_crash_recovery_uses_fresh_operator(
    tmp_path: Path,
    phase: str,
    expected_generation: int,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"

    async def exercise() -> None:
        first = await DurableSessionAuthority(workspace, sessions_root=root).acquire()
        first.manager.admit_persistent_write("cli:item")
        owner_context = acquire_execution(
            first.manager._jsonl_store,
            "cli:item",
            run_id=f"reset-{phase}",
        )
        owner = await owner_context.__aenter__()
        try:
            await owner.quiesce()
            with pytest.raises(RuntimeError, match="injected"):
                first.manager._mode_resolver.reset(
                    "cli:item",
                    [],
                    owner=owner,
                    fault_after=phase,
                )
        finally:
            owner._active = False
            owner._lock.release()
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        resolved = second.manager.recover_session_mode("cli:item")
        assert resolved.mode == "canonical"
        assert resolved.generation == expected_generation
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


def test_canonical_reset_prepared_rejects_graceful_owner_mutation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"

    async def exercise() -> None:
        first = await DurableSessionAuthority(workspace, sessions_root=root).acquire()
        first.manager.admit_persistent_write("cli:item")
        async with acquire_execution(
            first.manager._jsonl_store,
            "cli:item",
            run_id="reset-graceful",
        ) as owner:
            await owner.quiesce()
            with pytest.raises(RuntimeError, match="injected"):
                first.manager._mode_resolver.reset(
                    "cli:item", [], owner=owner, fault_after="prepared"
                )
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        with pytest.raises(SessionModeError, match="offline recovery"):
            second.manager.recover_session_mode("cli:item")
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "phase",
    [
        "master_prepared",
        "registry_preparing",
        "witness_preparing",
        "records_committed",
        "target_installed",
        "registry_active",
        "mode_committed",
    ],
)
def test_bootstrap_paired_witness_resume_from_fresh_operator(
    tmp_path: Path,
    phase: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "exact"}])
    unmanaged.save(session, fsync=True)
    original = unmanaged._get_session_path(session.key).read_bytes()

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        with pytest.raises(RuntimeError, match="injected"):
            first.manager._mode_resolver.bootstrap(
                offline_copy=True,
                fault_after=phase,
            )
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        assert second.manager.bootstrap_existing_sessions(offline_copy=True) == 1
        assert second.manager.resolve_session_mode(session.key).mode == "legacy"
        assert second.manager._get_session_path(session.key).read_bytes() == original
        journals = second.manager._mode_resolver.storage.list(".session-mode/journals")
        master = [
            second.manager._mode_resolver._journal_records(operation_id)
            for operation_id in journals
            if second.manager._mode_resolver._journal_records(operation_id)[0]["source_key"]
            is None
        ]
        assert len(master) == 1
        assert [row["phase"] for row in master[0]] == [
            "prepared",
            "target_installed",
            "mode_committed",
            "complete",
        ]
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


def test_runtime_rejects_active_witnesses_with_incomplete_bootstrap_master(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"

    async def exercise() -> None:
        operator = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        with pytest.raises(RuntimeError, match="injected"):
            operator.manager._mode_resolver.bootstrap(
                offline_copy=True,
                fault_after="mode_committed",
            )
        assert operator.close is not None
        await operator.close()
        with pytest.raises(SessionModeError, match="offline recovery"):
            await DurableSessionAuthority(workspace, sessions_root=root).acquire()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("phase", "expected_generation", "expect_original"),
    [
        ("prepared", 0, True),
        ("target_installed", 1, False),
        ("mode_record_published", 1, False),
    ],
)
def test_legacy_reset_crash_recovery_from_fresh_operator(
    tmp_path: Path,
    phase: str,
    expected_generation: int,
    expect_original: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "exact"}])
    unmanaged.save(session, fsync=True)
    original = unmanaged._get_session_path(session.key).read_bytes()

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        first.manager.bootstrap_existing_sessions(offline_copy=True)
        with pytest.raises(RuntimeError, match="injected"):
            first.manager._mode_resolver.reset(
                session.key,
                [],
                owner=None,
                fault_after=phase,
            )
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        resolved = second.manager.recover_session_mode(session.key)
        assert resolved.mode == "legacy"
        assert resolved.generation == expected_generation
        current = second.manager._get_session_path(session.key).read_bytes()
        assert (current == original) is expect_original
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("phase", "expected_state", "source_exists"),
    [
        ("prepared", "active", True),
        ("mode_record_published", "deleted", False),
        ("mode_committed", "deleted", False),
        ("source_deleted", "deleted", False),
    ],
)
def test_delete_crash_recovery_from_fresh_operator(
    tmp_path: Path,
    phase: str,
    expected_state: str,
    source_exists: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "exact"}])
    unmanaged.save(session, fsync=True)

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        first.manager.bootstrap_existing_sessions(offline_copy=True)
        with pytest.raises(RuntimeError, match="injected"):
            first.manager._mode_resolver.delete(session.key, fault_after=phase)
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        resolved = second.manager.recover_session_mode(session.key)
        assert resolved.state == expected_state
        assert second.manager._get_session_path(session.key).exists() is source_exists
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["prepared", "mode_record_published"])
def test_delete_recovery_rejects_changed_bound_sidecar(tmp_path: Path, phase: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    session = Session("cli:legacy", messages=[{"role": "user", "content": "exact"}])
    unmanaged.save(session, fsync=True)

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        first.manager.bootstrap_existing_sessions(offline_copy=True)
        storage_key = first.manager._jsonl_store.storage_key(session.key)
        sidecar_name = f"{storage_key}.checkpoint.json"
        checkpoint_session = first.manager.read_session_snapshot(session.key)
        checkpoint_session.metadata["runtime_checkpoint"] = {"bound": True}
        first.manager._jsonl_store.save_runtime_checkpoint(checkpoint_session)
        with pytest.raises(RuntimeError, match="injected"):
            first.manager._mode_resolver.delete(session.key, fault_after=phase)
        first.manager._mode_resolver.storage.write(sidecar_name, b'{"changed":true}\n')
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        with pytest.raises(SessionModeError, match="offline recovery"):
            second.manager.recover_session_mode(session.key)
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


def test_recreate_binds_and_advances_prior_tombstone(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        lease.manager.admit_persistent_write("cli:item")
        assert lease.manager._mode_resolver.delete("cli:item") is True
        tombstone = lease.manager.resolve_session_mode("cli:item")
        recreated = lease.manager.admit_persistent_write("cli:item")
        assert recreated.record_revision == tombstone.record_revision + 1
        assert recreated.incarnation != tombstone.incarnation
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("phase", "expected_mode"),
    [
        ("prepared", None),
        ("target_installed", None),
        ("mode_record_published", "canonical"),
    ],
)
def test_fork_crash_recovery_from_fresh_operator(
    tmp_path: Path,
    phase: str,
    expected_mode: str | None,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        first.manager.bootstrap_existing_sessions(offline_copy=True)
        first.manager.admit_persistent_write("cli:source")
        with pytest.raises(RuntimeError, match="injected"):
            first.manager._mode_resolver.fork(
                "cli:source",
                "cli:target",
                0,
                fault_after=phase,
            )
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        resolved = second.manager.recover_session_mode("cli:target")
        assert resolved.mode == expected_mode
        assert second.manager.resolve_session_mode("cli:source").mode == "canonical"
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["prepared", "installed", "published"])
def test_export_crash_recovery_proves_destination(
    tmp_path: Path,
    phase: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    destination = tmp_path / "exported"

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        first.manager.bootstrap_existing_sessions(offline_copy=True)
        first.manager.admit_persistent_write("cli:source")
        tx = first.manager._mode_resolver.begin_export(
            "cli:source",
            destination,
            {"manifest.json": b"{}\n", "canonical-source.jsonl": b"source\n"},
        )
        if phase == "installed":
            with pytest.raises(RuntimeError, match="injected"):
                tx.mark_published(fault_after_install=True)
            with pytest.raises(SessionModeError, match="offline recovery"):
                tx.abort()
        elif phase == "published":
            tx.mark_published()
        tx._parent_pin.close()
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        resolved = second.manager.recover_session_mode("cli:source")
        assert resolved.mode == "canonical"
        if phase == "prepared":
            assert not destination.exists()
        else:
            assert (destination / "manifest.json").read_bytes() == b"{}\n"
            assert (destination / "canonical-source.jsonl").read_bytes() == b"source\n"
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


def test_export_partial_stage_never_appears_at_final_destination(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    destination = tmp_path / "exported"

    async def exercise() -> None:
        first = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        first.manager.bootstrap_existing_sessions(offline_copy=True)
        first.manager.admit_persistent_write("cli:source")
        tx = first.manager._mode_resolver.begin_export(
            "cli:source",
            destination,
            {"manifest.json": b"{}\n", "canonical-source.jsonl": b"source\n"},
        )
        with pytest.raises(RuntimeError, match="injected"):
            tx.mark_published(fault_after_stage_file=True)
        assert not destination.exists()
        assert (tmp_path / tx.publication_stage_name).is_dir()
        tx.abort()
        assert not destination.exists()
        assert first.close is not None
        await first.close()

        second = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        assert second.manager.recover_session_mode("cli:source").mode == "canonical"
        assert not destination.exists()
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


def test_export_complete_revalidates_changed_destination(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    destination = tmp_path / "exported"

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        lease.manager.admit_persistent_write("cli:source")
        tx = lease.manager._mode_resolver.begin_export(
            "cli:source",
            destination,
            {"manifest.json": b"{}\n", "canonical-source.jsonl": b"source\n"},
        )
        tx.mark_published()
        (destination / "manifest.json").write_bytes(b"changed\n")
        with pytest.raises(SessionModeError, match="offline recovery"):
            tx.complete()
        tx._parent_pin.close()
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_export_published_phase_cannot_be_rechained_to_aborted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    destination = tmp_path / "exported"

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        lease.manager.admit_persistent_write("cli:source")
        tx = lease.manager._mode_resolver.begin_export(
            "cli:source",
            destination,
            {"manifest.json": b"{}\n", "canonical-source.jsonl": b"source\n"},
        )
        tx.mark_published()
        journal = lease.manager._mode_resolver._journal_records(tx.operation_id)
        published_raw = (
            json.dumps(journal[-1], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        aborted = dict(journal[-1])
        aborted["sequence"] = 3
        aborted["phase"] = "aborted"
        aborted["previous_record_digest"] = hashlib.sha256(published_raw).hexdigest()
        aborted_raw = (
            json.dumps(aborted, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        lease.manager._mode_resolver.storage.write(
            f".session-mode/journals/{tx.operation_id}/0003-aborted.json",
            aborted_raw,
            replace=False,
        )
        with pytest.raises(SessionModeError, match="offline recovery"):
            lease.manager.resolve_session_mode("cli:source")
        tx._parent_pin.close()
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())
