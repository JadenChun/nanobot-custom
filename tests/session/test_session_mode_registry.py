from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import nanobot.session.mode_storage as mode_storage
from nanobot.session.execution_owner import execution_lock_guard
from nanobot.session.manager import Session, SessionManager, SessionStore
from nanobot.session.runtime_authority import DurableSessionAuthority, OfflineSessionOperator
from nanobot.session.session_mode import SessionModeError


def _legacy_with_checkpoint(tmp_path: Path) -> tuple[Path, Path, SessionManager, Session, bytes]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    manager = SessionManager(workspace, sessions_root=root)
    session = Session("cli:private", messages=[{"role": "user", "content": "private-body"}])
    manager.save(session, fsync=True)
    session.metadata["runtime_checkpoint"] = {"pending": "private-checkpoint"}
    manager._jsonl_store.save_runtime_checkpoint(session)
    return workspace, root, manager, session, manager._jsonl_store.get_runtime_checkpoint_path(session.key).read_bytes()


@pytest.mark.parametrize("mutation", [
    "unknown", "missing", "version_bool", "version_old", "count_bool", "revision_bool",
    "generation_bool", "key", "timestamp", "count", "revision", "generation", "incarnation",
    "checkpoint", "provider_invalid", "provider_unknown", "duplicate", "nan", "orphan", "canonical",
])
def test_paired_inventory_rejects_invalid_sidecars_without_mutation(tmp_path: Path, mutation: str) -> None:
    workspace, root, unmanaged, session, raw = _legacy_with_checkpoint(tmp_path)
    sidecar = unmanaged._jsonl_store.get_runtime_checkpoint_path(session.key)
    value = json.loads(raw)
    if mutation == "unknown":
        value["unknown"] = "private"
    elif mutation == "missing":
        value.pop("provider_state")
    elif mutation == "version_bool":
        value["version"] = True
    elif mutation == "version_old":
        value["version"] = 1
    elif mutation.endswith("_bool"):
        value[{"count_bool": "base_message_count", "revision_bool": "base_revision", "generation_bool": "base_generation"}[mutation]] = True
    elif mutation in {"key", "timestamp", "incarnation"}:
        value[{"key": "session_key", "timestamp": "base_updated_at", "incarnation": "base_incarnation"}[mutation]] = "different"
    elif mutation in {"count", "revision", "generation"}:
        value[{"count": "base_message_count", "revision": "base_revision", "generation": "base_generation"}[mutation]] += 1
    elif mutation == "checkpoint":
        value["checkpoint"] = []
    elif mutation == "provider_invalid":
        value["provider_state"] = {"private": True}
    elif mutation == "provider_unknown":
        value["provider_state"] = {"kind": "fixture", "provider": "fixture", "model": "m", "version": 1, "payload": {}, "pending_messages": [], "unknown": True}
    elif mutation == "orphan":
        unmanaged._get_session_path(session.key).unlink()
    elif mutation == "canonical":
        sidecar.unlink()
        unmanaged._jsonl_store.source.upgrade_source_session(
            session.key, expected_revision=session.revision,
            expected_generation=session.generation, expected_incarnation=session.incarnation,
        )
    if mutation == "duplicate":
        raw = raw.replace(b"{", b'{"version":2,', 1)
    elif mutation == "nan":
        raw = raw.replace(b'"checkpoint":{', b'"checkpoint":{"x":NaN,', 1)
    else:
        raw = json.dumps(value).encode()
    sidecar.write_bytes(raw)
    sidecar.chmod(0o600)
    before = (sidecar.read_bytes(), sidecar.stat().st_ino, sidecar.stat().st_mtime_ns)

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        with pytest.raises(SessionModeError) as error:
            lease.manager.bootstrap_existing_sessions(offline_copy=True)
        assert str(error.value) == "session mode evidence is invalid; offline recovery is required"
        assert (sidecar.read_bytes(), sidecar.stat().st_ino, sidecar.stat().st_mtime_ns) == before
        assert not (lease.manager.sessions_dir / ".session-mode").exists()
        await lease.close()

    asyncio.run(exercise())


def test_bootstrap_journals_exact_pair_and_managed_reads_are_pure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import base64

    workspace, root, unmanaged, session, sidecar_bytes = _legacy_with_checkpoint(tmp_path)
    sidecar = unmanaged._jsonl_store.get_runtime_checkpoint_path(session.key)
    source = unmanaged._get_session_path(session.key)
    # mtime must not decide whether valid evidence is applied.
    os.utime(sidecar, ns=(1, 1))
    before = {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) for path in (source, sidecar)}

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        resolver = manager._mode_resolver
        rows = [resolver._journal_records(op)[0] for op in resolver.storage.list(".session-mode/journals")]
        master = next(row for row in rows if row["source_key"] is None)
        item = next(row for row in rows if row["source_key"] == session.key)
        with manager._jsonl_store.locked_session_files():
            inventory = resolver._source_inventory()[0]
        proof = {"items": [{
            "key": session.key, "source_name": inventory.source_name,
            "source_digest": hashlib.sha256(source.read_bytes()).hexdigest(), "source_size": source.stat().st_size,
            "sidecar_name": inventory.sidecar_name, "sidecar_state": "present",
            "sidecar_digest": hashlib.sha256(sidecar_bytes).hexdigest(), "sidecar_size": len(sidecar_bytes),
        }]}
        encoded = (json.dumps(proof, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        assert master["input_digest"] == hashlib.sha256(encoded).hexdigest() and master["input_size"] == len(encoded)
        assert item["input_digest"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert item["sidecar_digest"] == hashlib.sha256(sidecar_bytes).hexdigest()
        assert base64.b64decode(item["sidecar_bytes"]) == sidecar_bytes
        monkeypatch.setattr(manager._jsonl_store, "_overlay_runtime_checkpoint_unlocked", lambda *_args: pytest.fail("managed compatibility cleanup"))
        for _ in range(2):
            loaded = manager.read_session_snapshot(session.key)
            assert loaded.metadata["runtime_checkpoint"] == {"pending": "private-checkpoint"}
            assert len(manager.list_sessions()) == 1
        assert all((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) == state for path, state in before.items())
        assert manager.upsert_session_metadata(session.key, {"title": "durable"}, removals=("nothing",))
        assert manager.read_session_snapshot(session.key).metadata["runtime_checkpoint"] == {"pending": "private-checkpoint"}
        await lease.close()

    asyncio.run(exercise())


def _authority(tmp_path: Path) -> DurableSessionAuthority:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return DurableSessionAuthority(workspace, sessions_root=tmp_path / "state" / "sessions")


def test_first_write_creates_canonical_record_but_read_does_not(tmp_path: Path) -> None:
    async def exercise() -> None:
        lease = await _authority(tmp_path).acquire()
        manager = lease.manager
        assert manager.read_session_snapshot("cli:one") is None
        assert not (manager.sessions_dir / ".session-mode").exists()
        manager.admit_persistent_write("cli:one")
        resolved = manager.resolve_session_mode("cli:one")
        assert resolved.mode == "canonical"
        assert manager.source_backend("cli:one") is manager._jsonl_store
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_runtime_lease_is_nonblocking(tmp_path: Path) -> None:
    async def exercise() -> None:
        first = await _authority(tmp_path).acquire()
        with pytest.raises(SessionModeError, match="session authority is busy"):
            await _authority(tmp_path).acquire()
        assert first.close is not None
        await first.close()

    asyncio.run(exercise())


def test_offline_copy_rebinds_workspace_claim_through_pinned_root(tmp_path: Path) -> None:
    original_workspace = tmp_path / "original-workspace"
    original_workspace.mkdir()
    original_root = tmp_path / "original-root"
    SessionManager(original_workspace, sessions_root=original_root)
    copied_workspace = tmp_path / "copied-workspace"
    copied_root = tmp_path / "copied-root"
    shutil.copytree(original_workspace, copied_workspace)
    shutil.copytree(original_root, copied_root)
    workspace_id = (copied_workspace / ".nanobot" / "workspace-id").read_text().strip()
    claim = copied_root / workspace_id / ".workspace"
    assert Path(claim.read_text().strip()) == original_workspace

    async def exercise() -> None:
        lease = await OfflineSessionOperator(
            copied_workspace,
            copied_sessions_root=copied_root,
        ).acquire()
        assert Path(claim.read_text().strip()) == copied_workspace
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_empty_acquire_close_reacquire_remains_absent_and_unregistered(tmp_path: Path) -> None:
    async def exercise() -> None:
        first = await _authority(tmp_path).acquire()
        assert first.manager.read_session_snapshot("cli:absent") is None
        assert first.close is not None
        await first.close()

        second = await _authority(tmp_path).acquire()
        assert second.manager.read_session_snapshot("cli:absent") is None
        assert not (second.manager.sessions_dir / ".session-mode").exists()
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


def test_runtime_rejects_legacy_workspace_before_constructor_migration(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    legacy = workspace / "sessions"
    legacy.mkdir(parents=True)
    source = legacy / "cli_legacy.jsonl"
    source.write_text('{"_type":"metadata","key":"cli:legacy"}\n', encoding="utf-8")
    root = tmp_path / "state" / "sessions"

    async def exercise() -> None:
        with pytest.raises(SessionModeError, match="offline bootstrap"):
            await DurableSessionAuthority(workspace, sessions_root=root).acquire()

    asyncio.run(exercise())
    assert source.is_file()
    assert not list(root.glob("[!.]*"))


def test_unprivileged_managers_before_and_after_registration_cannot_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"
    early = SessionManager(workspace, sessions_root=root)

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        lease.manager.bootstrap_existing_sessions(offline_copy=True)

        with pytest.raises(SessionModeError, match="managed session namespace"):
            early.save(Session("cli:early"), fsync=True)
        with pytest.raises(SessionModeError, match="managed session namespace"):
            SessionManager(workspace, sessions_root=root)

        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_preconstructed_custom_manager_cannot_write_after_registration(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"
    custom = MagicMock(spec=SessionStore)
    early = SessionManager(workspace, sessions_root=root, store=custom)

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        with pytest.raises(SessionModeError, match="managed session namespace"):
            early.save(Session("cli:custom"), fsync=True)
        custom.save.assert_not_called()
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_unknown_mode_entry_and_replaced_record_directory_fail_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        manager.admit_persistent_write("cli:item")
        records = manager.sessions_dir / ".session-mode" / "records"
        detached = records.with_name("records-detached")
        records.rename(detached)
        records.mkdir(mode=0o700)
        for source in detached.iterdir():
            shutil.copyfile(source, records / source.name)
            (records / source.name).chmod(0o600)
        with pytest.raises(SessionModeError):
            manager.resolve_session_mode("cli:item")
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_unknown_journal_entry_is_not_filtered(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        journals = manager.sessions_dir / ".session-mode" / "journals"
        unexpected = journals / "unexpected"
        unexpected.write_text("x", encoding="utf-8")
        unexpected.chmod(0o600)
        with pytest.raises(SessionModeError):
            manager.resolve_session_mode("cli:item")
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_preconstructed_source_root_and_execution_lock_cannot_mutate_claimed_namespace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"
    early = SessionManager(workspace, sessions_root=root)
    source_root = early._jsonl_store.source.create_source_root_authority()

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        actions = (
            lambda: source_root.write_text_atomic("bypass.jsonl", "x\n"),
            lambda: source_root.open_directory("bypass-dir", create=True),
            lambda: source_root.lock_file("bypass.lock", blocking=False),
            lambda: source_root.unlink_relative("missing"),
            lambda: source_root.open_regular(".workspace", flags=os.O_RDWR),
        )
        for action in actions:
            with pytest.raises(SessionModeError, match="managed session namespace"):
                action()
        with pytest.raises(SessionModeError, match="managed session namespace"):
            with execution_lock_guard(early._jsonl_store, "cli:item"):
                pass
        assert not (early.sessions_dir / "bypass.jsonl").exists()
        assert not (early.sessions_dir / "bypass-dir").exists()
        assert not (early.sessions_dir / "bypass.lock").exists()
        assert lease.close is not None
        await lease.close()

    try:
        asyncio.run(exercise())
    finally:
        source_root.close()


def test_unsupported_nofollow_fails_before_sessions_root_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "missing" / "sessions"
    monkeypatch.setattr(mode_storage.os, "O_NOFOLLOW", None)

    async def exercise() -> None:
        with pytest.raises(SessionModeError, match="unavailable"):
            await DurableSessionAuthority(workspace, sessions_root=root).acquire()

    asyncio.run(exercise())
    assert not root.parent.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "registry_bool_version",
        "record_bool_generation",
        "record_bool_source_version",
        "journal_bool_version",
        "journal_untyped_source",
        "journal_unaware_timestamp",
        "journal_duplicate_key",
    ],
)
def test_strict_mode_schema_rejects_ambiguous_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"

    def canonical(value: dict[str, object]) -> bytes:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        manager.admit_persistent_write("cli:item")
        mode_root = manager.sessions_dir / ".session-mode"
        if mutation == "registry_bool_version":
            registry_path = mode_root / "registry.json"
            witness_path = workspace / ".nanobot" / "session-mode-binding.json"
            registry = json.loads(registry_path.read_bytes())
            registry["schema_version"] = True
            registry_path.write_bytes(canonical(registry))
            witness_path.write_bytes(canonical(registry))
        elif mutation in {"record_bool_generation", "record_bool_source_version"}:
            record_path = next((mode_root / "records").glob("*.json"))
            record = json.loads(record_path.read_bytes())
            if mutation == "record_bool_generation":
                record["generation"] = True
            else:
                record["source_schema_version"] = True
            record_path.write_bytes(canonical(record))
        else:
            journal_path = next(
                path
                for path in (mode_root / "journals").glob("*/0004-complete.json")
                if json.loads(path.read_bytes())["target_key"] == "cli:item"
            )
            journal = json.loads(journal_path.read_bytes())
            if mutation == "journal_bool_version":
                journal["schema_version"] = True
                journal_path.write_bytes(canonical(journal))
            elif mutation == "journal_untyped_source":
                journal["source_key"] = 7
                journal_path.write_bytes(canonical(journal))
            elif mutation == "journal_unaware_timestamp":
                journal["timestamp"] = "2026-10-01T00:00:00"
                journal_path.write_bytes(canonical(journal))
            else:
                raw = canonical(journal)
                journal_path.write_bytes(raw.replace(b"{", b'{"schema_version":1,', 1))
        with pytest.raises(SessionModeError, match="offline recovery"):
            manager.resolve_session_mode("cli:item")
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_deleted_record_requires_exact_committed_journal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        manager.admit_persistent_write("cli:item")
        assert manager._mode_resolver.delete("cli:item") is True
        record_path = next((manager.sessions_dir / ".session-mode" / "records").glob("*.json"))
        record = json.loads(record_path.read_bytes())
        record["committed_operation_id"] = "0" * 32
        record_path.write_bytes(
            (
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
        )
        with pytest.raises(SessionModeError, match="offline recovery"):
            manager.resolve_session_mode("cli:item")
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_unsafe_existing_ancestor_rejects_before_descendant_creation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    root = unsafe / "new" / "sessions"

    async def exercise() -> None:
        with pytest.raises(SessionModeError):
            await DurableSessionAuthority(workspace, sessions_root=root).acquire()

    asyncio.run(exercise())
    assert not (unsafe / "new").exists()


def test_initial_first_write_acquires_migration_before_session_files(tmp_path: Path) -> None:
    events: list[str] = []
    held: list[str] = []

    class TraceLock:
        def __init__(self, name: str) -> None:
            self.name = name

        @property
        def is_locked(self) -> bool:
            return self.name in held

        def __enter__(self) -> None:
            if self.name == "files":
                assert "migration" in held
            events.append(f"enter:{self.name}")
            held.append(self.name)

        def __exit__(self, *_args: object) -> None:
            assert held.pop() == self.name
            events.append(f"exit:{self.name}")

    async def exercise() -> None:
        lease = await _authority(tmp_path).acquire()
        lease.manager._jsonl_store._migration_lock = TraceLock("migration")
        lease.manager._jsonl_store._session_files_lock = TraceLock("files")
        lease.manager.admit_persistent_write("cli:item")
        assert events[:2] == ["enter:migration", "enter:files"]
        assert events[-2:] == ["exit:files", "exit:migration"]
        assert events.count("enter:migration") == 1
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("mutation", ["registry_id", "bool_generation"])
def test_rechained_create_journal_requires_registry_and_typed_binding(
    tmp_path: Path,
    mutation: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "state" / "sessions"

    def canonical(value: dict[str, object]) -> bytes:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        manager.admit_persistent_write("cli:item")
        journals = manager.sessions_dir / ".session-mode" / "journals"
        journal_dir = next(
            directory
            for directory in journals.iterdir()
            if json.loads(next(directory.glob("0001-*.json")).read_bytes()).get("operation")
            == "create"
        )
        previous: str | None = None
        for path in sorted(journal_dir.glob("*.json")):
            row = json.loads(path.read_bytes())
            row["previous_record_digest"] = previous
            if mutation == "registry_id":
                row["registry_id"] = "a" * 32
            else:
                row["target_binding"]["generation"] = False
            raw = canonical(row)
            path.write_bytes(raw)
            previous = hashlib.sha256(raw).hexdigest()
        with pytest.raises(SessionModeError, match="offline recovery"):
            manager.resolve_session_mode("cli:item")
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())
