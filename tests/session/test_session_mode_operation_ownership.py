"""Real-owner regression controls for mode locks and export phase operations."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import pytest

from nanobot.session import execution_owner
from nanobot.session.execution_owner import ExecutionBusyError, acquire_execution
from nanobot.session.runtime_authority import OfflineSessionOperator, SessionAuthorityLease
from nanobot.session.session_mode import SessionModeError

KEY = "cli:operation-ownership"


@asynccontextmanager
async def _managed(tmp_path: Path) -> AsyncIterator[SessionAuthorityLease]:
    tmp_path.mkdir(exist_ok=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    lease = await OfflineSessionOperator(
        workspace, copied_sessions_root=tmp_path / "sessions"
    ).acquire()
    try:
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        lease.manager.admit_persistent_write(KEY)
        yield lease
    finally:
        await lease.close()


def _bytes(manager) -> dict[Path, bytes]:
    return {
        path: path.read_bytes()
        for path in manager.sessions_dir.rglob("*")
        if path.is_file()
    }


async def test_files_lock_observation_never_acquires_or_returns_the_lock(tmp_path: Path) -> None:
    async with _managed(tmp_path) as lease:
        backend = lease.manager._jsonl_store
        assert backend.session_files_locked is False
        with backend.locked_session_files():
            assert backend.session_files_locked is True
            with backend.locked_session_files():
                assert backend.session_files_locked is True
            assert backend.session_files_locked is True
        assert backend.session_files_locked is False


@pytest.mark.parametrize("fail", [False, True])
async def test_migration_scope_releases_the_real_lock(tmp_path: Path, fail: bool) -> None:
    async with _managed(tmp_path) as lease:
        backend = lease.manager._jsonl_store
        assert not backend._migration_lock.is_locked
        if fail:
            with pytest.raises(RuntimeError, match="operation failed"):
                with backend.locked_namespace_migration() as value:
                    assert value is None
                    assert backend._migration_lock.is_locked
                    raise RuntimeError("operation failed")
        else:
            with backend.locked_namespace_migration() as value:
                assert value is None
                assert backend._migration_lock.is_locked
        assert not backend._migration_lock.is_locked
        assert not backend.session_files_locked


@pytest.mark.parametrize("keyword", [False, True])
async def test_owner_scope_keeps_migration_key_execution_files_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keyword: bool
) -> None:
    async with _managed(tmp_path) as lease:
        backend = lease.manager._jsonl_store
        resolver = lease.manager._mode_resolver
        trace: list[str] = []
        migration = backend._migration_lock
        files = backend.locked_session_files
        execution = execution_owner.execution_lock_guard

        class TracedMigration:
            def __enter__(self):
                trace.append("migration:enter")
                return migration.__enter__()

            def __exit__(self, *arguments):
                try:
                    return migration.__exit__(*arguments)
                finally:
                    trace.append("migration:exit")

        class TracedKeywords(dict):
            def get(self, name, default=None):
                trace.append("key")
                assert migration.is_locked
                return super().get(name, default)

        @contextmanager
        def traced_execution(store, key):
            assert store is backend and key == KEY and migration.is_locked
            trace.append("execution:enter")
            with execution(store, key):
                try:
                    yield
                finally:
                    trace.append("execution:exit")

        @contextmanager
        def traced_files():
            assert migration.is_locked
            trace.append("files:enter")
            with files() as directory:
                try:
                    yield directory
                finally:
                    trace.append("files:exit")

        monkeypatch.setattr(backend, "_migration_lock", TracedMigration())
        monkeypatch.setattr(backend, "locked_session_files", traced_files)
        monkeypatch.setattr(execution_owner, "execution_lock_guard", traced_execution)
        with resolver.locked_mode_operation(
            migration=True,
            execution=True,
            arguments=() if keyword else (KEY,),
            keywords=TracedKeywords(key=KEY) if keyword else {},
        ) as value:
            assert value is None and backend.session_files_locked
            trace.append("body")
        expected = ["migration:enter"] + (["key"] if keyword else []) + [
            "execution:enter", "files:enter", "body", "files:exit",
            "execution:exit", "migration:exit",
        ]
        assert trace == expected
        assert not migration.is_locked and not backend.session_files_locked
        assert not hasattr(resolver._guard_local, "scope")


@pytest.mark.parametrize("key", [None, "", 1])
async def test_invalid_key_rejects_and_unwinds_migration(tmp_path: Path, key: object) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        backend = manager._jsonl_store
        before = _bytes(manager)
        with pytest.raises(SessionModeError):
            with manager._mode_resolver.locked_mode_operation(
                migration=True, execution=True, arguments=(key,), keywords={}
            ):
                pytest.fail("invalid key entered operation")
        assert _bytes(manager) == before
        assert not backend._migration_lock.is_locked and not backend.session_files_locked


@pytest.mark.parametrize("transaction", [False, True])
async def test_operation_scope_rejects_files_first(tmp_path: Path, transaction: bool) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        owner = tx if transaction else resolver
        before = _bytes(manager)
        with manager._jsonl_store.locked_session_files():
            with pytest.raises(SessionModeError):
                with owner.locked_mode_operation(
                    migration=True, execution=True, arguments=(KEY,), keywords={}
                ):
                    pytest.fail("files-first scope entered")
        assert _bytes(manager) == before
        tx.abort()


@pytest.mark.parametrize("transaction", [False, True])
async def test_scope_failure_releases_real_execution_and_files_locks(
    tmp_path: Path, transaction: bool
) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        owner = tx if transaction else resolver
        with pytest.raises(RuntimeError, match="operation failed"):
            with owner.locked_mode_operation(
                migration=False, execution=True, arguments=(KEY,), keywords={}
            ):
                raise RuntimeError("operation failed")
        assert not manager._jsonl_store.session_files_locked
        assert not hasattr(resolver._guard_local, "scope")
        with execution_owner.execution_lock_guard(manager._jsonl_store, KEY):
            pass
        tx.abort()


async def test_export_scope_uses_transaction_key_not_call_arguments(tmp_path: Path) -> None:
    async with _managed(tmp_path) as lease:
        resolver = lease.manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        with tx.locked_mode_operation(
            migration=False, execution=True, arguments=(None,), keywords={"key": ""}
        ):
            assert resolver._guard_local.scope[1] == (KEY,)
            assert lease.manager._jsonl_store.session_files_locked
        tx.abort()


async def test_transaction_owner_predicate_returns_only_identity_observation(tmp_path: Path) -> None:
    async with _managed(tmp_path) as first, _managed(tmp_path / "second") as second:
        resolver = first.manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        assert tx.belongs_to_resolver(resolver) is True
        assert tx.belongs_to_resolver(second.manager._mode_resolver) is False
        tx.abort()


@pytest.mark.parametrize("status", ["exact", "absent", "conflict"])
async def test_stage_proof_uses_actual_pinned_parent_and_exact_files(
    tmp_path: Path, status: str
) -> None:
    async with _managed(tmp_path) as lease:
        resolver = lease.manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        stage = tmp_path / "probe"
        if status != "absent":
            stage.mkdir(mode=0o700)
            payload = stage / "manifest.json"
            payload.write_bytes(b"{}\n" if status == "exact" else b"xx\n")
            payload.chmod(0o600)
        if status == "exact":
            assert resolver.require_export_stage(
                tx._parent_pin, stage.name, tx.binding["files"]
            ) is None
        else:
            with pytest.raises(SessionModeError):
                resolver.require_export_stage(tx._parent_pin, stage.name, tx.binding["files"])
        tx.abort()


async def test_stage_operation_keeps_the_live_owner_reader_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _managed(tmp_path) as lease:
        resolver = lease.manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        stage = tmp_path / "probe"
        stage.mkdir(mode=0o700)
        payload = stage / "manifest.json"
        payload.write_bytes(b"{}\n")
        payload.chmod(0o600)
        calls: list[str] = []

        def rejecting_reader(parent, name, files):
            assert parent is tx._parent_pin and files is tx.binding["files"]
            calls.append(name)
            return "conflict"

        with monkeypatch.context() as patch:
            patch.setattr(resolver, "_export_directory_status", rejecting_reader)
            with pytest.raises(SessionModeError):
                resolver.require_export_stage(tx._parent_pin, stage.name, tx.binding["files"])
        assert calls == [stage.name]
        tx.abort()


async def test_direct_phase_operation_holds_guards_and_retains_exact_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        append = resolver._append_journal
        trace: list[str] = []

        def guarded_append(*arguments, **keywords):
            assert manager._jsonl_store.session_files_locked
            assert resolver._guard_local.scope[1] == (KEY,)
            trace.append(keywords["phase"])
            return append(*arguments, **keywords)

        monkeypatch.setattr(resolver, "_append_journal", guarded_append)
        with tx.locked_mode_operation(
            migration=False, execution=True, arguments=(), keywords={}
        ):
            assert resolver.record_export_phase(tx, phase="aborted") is None
        rows = resolver._journal_records(tx.operation_id)
        assert trace == ["aborted"]
        assert [row["phase"] for row in rows] == ["prepared", "aborted"]
        for name in ("source_key", "target_key", "target_binding", "staging_name", "input_digest", "output_digest"):
            assert rows[0][name] == rows[1][name]
        before = _bytes(manager)
        with tx.locked_mode_operation(
            migration=False, execution=True, arguments=(), keywords={}
        ):
            with pytest.raises(SessionModeError):
                resolver.record_export_phase(tx, phase="aborted")
        assert _bytes(manager) == before
        assert not manager._jsonl_store.session_files_locked
        assert not hasattr(resolver._guard_local, "scope")
        assert resolver.recover(KEY).mode == "canonical"
        tx._parent_pin.close()


async def test_phase_operation_rejects_without_the_real_transaction_scope(tmp_path: Path) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        before = _bytes(manager)
        with pytest.raises(SessionModeError):
            resolver.record_export_phase(tx, phase="aborted")
        assert _bytes(manager) == before
        tx.abort()


@pytest.mark.parametrize("scope", ["files-only", "other-key", "foreign-task", "foreign-thread"])
async def test_wrong_call_scope_rejects_before_any_export_or_journal_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        manager.admit_persistent_write("cli:other")
        before = _bytes(manager)

        def forbidden_read(*_arguments):
            pytest.fail("unadmitted export/journal read")

        def reject():
            with pytest.raises(SessionModeError):
                resolver.record_export_phase(tx, phase="aborted")

        async def foreign_task():
            reject()

        with monkeypatch.context() as patch:
            patch.setattr(resolver, "_export_destination_status", forbidden_read)
            patch.setattr(resolver, "_journal_records", forbidden_read)
            if scope == "files-only":
                with manager._jsonl_store.locked_session_files():
                    reject()
            elif scope == "other-key":
                with resolver.locked_mode_operation(
                    migration=False, execution=True, arguments=("cli:other",), keywords={}
                ):
                    reject()
            else:
                with tx.locked_mode_operation(
                    migration=False, execution=True, arguments=(), keywords={}
                ):
                    if scope == "foreign-task":
                        await asyncio.create_task(foreign_task())
                    else:
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            pool.submit(reject).result(timeout=2)
        assert _bytes(manager) == before
        tx.abort()


async def test_publication_phase_cannot_claim_an_absent_destination(tmp_path: Path) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        before = _bytes(manager)
        with tx.locked_mode_operation(
            migration=False, execution=True, arguments=(), keywords={}
        ):
            with pytest.raises(SessionModeError):
                resolver.record_export_phase(tx, phase="published")
        assert _bytes(manager) == before
        tx.abort()


async def test_complete_phase_uses_real_published_destination_and_journal(tmp_path: Path) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        tx.mark_published()
        with tx.locked_mode_operation(
            migration=False, execution=True, arguments=(), keywords={}
        ):
            assert resolver.record_export_phase(tx, phase="complete") is None
        assert [row["phase"] for row in resolver._journal_records(tx.operation_id)] == [
            "prepared", "published", "complete",
        ]
        assert (tmp_path / "output" / "manifest.json").read_bytes() == b"{}\n"
        assert resolver.recover(KEY).mode == "canonical"
        tx._parent_pin.close()


async def test_invalid_phase_cannot_write_journal_evidence(tmp_path: Path) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        before = _bytes(manager)
        with tx.locked_mode_operation(
            migration=False, execution=True, arguments=(), keywords={}
        ):
            with pytest.raises(SessionModeError):
                resolver.record_export_phase(tx, phase="prepared")
        assert _bytes(manager) == before
        tx.abort()


@pytest.mark.parametrize("mismatch", ["key", "binding", "staging", "operation"])
async def test_public_phase_operation_cannot_rebind_a_different_journal(
    tmp_path: Path, mismatch: str
) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        original_key, original_binding, original_id = tx.key, tx.binding, tx.operation_id
        if mismatch == "key":
            manager.admit_persistent_write("cli:other")
            tx.key = "cli:other"
        elif mismatch == "binding":
            tx.binding = dict(tx.binding, session={})
        elif mismatch == "staging":
            tx.operation_id = "a" * 32
        else:
            record = json.loads(resolver.storage.read(resolver._record_name(KEY)))
            tx.operation_id = record["committed_operation_id"]
        before = _bytes(manager)
        with tx.locked_mode_operation(
            migration=False, execution=True, arguments=(), keywords={}
        ):
            with pytest.raises(SessionModeError):
                resolver.record_export_phase(tx, phase="aborted")
        assert _bytes(manager) == before
        tx.key, tx.binding, tx.operation_id = original_key, original_binding, original_id
        tx.abort()


async def test_foreign_transaction_rejects_before_touching_either_namespace(tmp_path: Path) -> None:
    async with _managed(tmp_path) as first, _managed(tmp_path / "second") as second:
        resolver = first.manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        before = _bytes(first.manager), _bytes(second.manager)
        with tx.locked_mode_operation(
            migration=False, execution=True, arguments=(), keywords={}
        ):
            with pytest.raises(SessionModeError):
                second.manager._mode_resolver.record_export_phase(tx, phase="aborted")
        assert (_bytes(first.manager), _bytes(second.manager)) == before
        tx.abort()


async def test_completion_requires_real_destination_before_journal_write(tmp_path: Path) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        before = _bytes(manager)
        with tx.locked_mode_operation(
            migration=False, execution=True, arguments=(), keywords={}
        ):
            with pytest.raises(SessionModeError):
                resolver.record_export_phase(tx, phase="complete")
        assert _bytes(manager) == before
        tx.abort()


async def test_direct_phase_operation_cannot_bypass_a_real_execution_owner(tmp_path: Path) -> None:
    async with _managed(tmp_path) as lease:
        manager = lease.manager
        resolver = manager._mode_resolver
        tx = resolver.begin_export(KEY, tmp_path / "output", {"manifest.json": b"{}\n"})
        async with acquire_execution(manager._jsonl_store, KEY, run_id="owner-control"):
            before = _bytes(manager)
            with pytest.raises(ExecutionBusyError):
                with tx.locked_mode_operation(
                    migration=False, execution=True, arguments=(), keywords={}
                ):
                    resolver.record_export_phase(tx, phase="aborted")
            assert _bytes(manager) == before
        tx.abort()
