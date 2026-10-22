from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest
from loguru import logger

from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import (
    SESSION_MODEL_PRESET_METADATA_KEY,
    Session,
    SessionManager,
)
from nanobot.session.runtime_authority import DurableSessionAuthority, OfflineSessionOperator
from nanobot.session.session_mode import SessionModeError
from nanobot.session.source import new_source_event


def test_manager_clear_absent_deleted_and_active_canonical(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        first = manager.reset_session("cli:item")
        assert first is not None and first.generation == 0 and first.messages == []
        assert manager.resolve_session_mode("cli:item").record_revision == 1
        cached = manager.get_or_create("cli:item")
        cleared = manager.reset_session("cli:item")
        assert cleared is not None and cleared.generation == 1
        assert manager.get_or_create("cli:item") is not cached
        assert manager.delete_session("cli:item")
        recreated = manager.reset_session("cli:item")
        assert recreated is not None and recreated.generation == 0
        assert recreated.incarnation != first.incarnation
        await lease.close()

    asyncio.run(exercise())


def test_unmanaged_absent_clear_persists_empty_and_invalid_managed_fork_is_noncreating(tmp_path: Path) -> None:
    unmanaged_workspace = tmp_path / "unmanaged"
    unmanaged_workspace.mkdir()
    unmanaged = SessionManager(unmanaged_workspace, sessions_root=tmp_path / "unmanaged-sessions")
    empty = unmanaged.reset_session("cli:absent")
    assert empty is not None and empty.messages == [] and empty.generation == 0
    assert unmanaged.read_session_snapshot("cli:absent").messages == []

    workspace = tmp_path / "managed"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:source")
        before = {p: p.read_bytes() for p in manager.sessions_dir.rglob("*") if p.is_file()}
        assert manager.fork_session_before_user_index("cli:source", "cli:target", 1) is None
        assert manager.resolve_session_mode("cli:target").state == "absent"
        assert all(path.read_bytes() == raw for path, raw in before.items())
        assert not manager._mode_resolver.storage.exists(manager._mode_resolver._record_name("cli:target"))
        await lease.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["create", "convert", "reset", "delete", "fork", "bootstrap", "recover", "export", "snapshot"])
def test_direct_lifecycle_rejects_files_first(tmp_path: Path, operation: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        manager.admit_persistent_write("cli:source")
        resolver = manager._mode_resolver
        before = {p: p.read_bytes() for p in manager.sessions_dir.rglob("*") if p.is_file()}
        calls = {
            "create": lambda: resolver.ensure_canonical("cli:target"),
            "convert": lambda: resolver.convert("cli:source", offline_copy=True),
            "reset": lambda: resolver.reset("cli:source", [], owner=None),
            "delete": lambda: resolver.delete("cli:source"),
            "fork": lambda: resolver.fork("cli:source", "cli:target", 0),
            "bootstrap": lambda: resolver.bootstrap(offline_copy=True),
            "recover": lambda: resolver.recover("cli:source"),
            "export": lambda: resolver.begin_export("cli:source", tmp_path / "out", {"manifest.json": b"{}\n"}),
            "snapshot": lambda: resolver.locked_snapshot("cli:source").__enter__(),
        }
        with manager._jsonl_store.locked_session_files():
            with pytest.raises(SessionModeError):
                calls[operation]()
        assert all(path.read_bytes() == raw for path, raw in before.items())
        await lease.close()

    asyncio.run(exercise())


def test_guard_reuse_is_subset_and_task_bound_and_clears_after_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        manager.admit_persistent_write("cli:source")
        resolver = manager._mode_resolver
        with resolver.locked_snapshot("cli:source"):
            with resolver.locked_snapshot("cli:source"):
                pass
            with pytest.raises(SessionModeError):
                with resolver.locked_snapshot("cli:other"):
                    pass

            async def other_task() -> None:
                with pytest.raises(SessionModeError):
                    with resolver.locked_snapshot("cli:source"):
                        pass

            await asyncio.create_task(other_task())
            tx = resolver.begin_export("cli:source", tmp_path / "out", {"manifest.json": b"{}\n"})
            tx.abort()
        assert not hasattr(resolver._guard_local, "scope")
        with pytest.raises(RuntimeError, match="test-failure"):
            with resolver.locked_snapshot("cli:source"):
                raise RuntimeError("test-failure")
        assert not hasattr(resolver._guard_local, "scope")
        with resolver.locked_snapshot("cli:source"):
            pass
        await lease.close()

    asyncio.run(exercise())


def test_keyword_lifecycle_calls_and_guard_thread_isolation(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=tmp_path / "sessions").acquire()
        resolver = lease.manager._mode_resolver
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        assert resolver.ensure_canonical(key="cli:source").mode == "canonical"
        with resolver.locked_snapshot(key="cli:source"):
            tx = resolver.begin_export(key="cli:source", destination=tmp_path / "out", files={"manifest.json": b"{}\n"})
            tx.abort()
            def different_thread() -> None:
                from nanobot.session.execution_owner import ExecutionBusyError

                with pytest.raises(ExecutionBusyError):
                    with resolver.locked_snapshot("cli:source"):
                        pass
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(different_thread).result(timeout=2)
        assert resolver.recover(key="cli:source").mode == "canonical"
        await lease.close()

    asyncio.run(exercise())


def test_recovery_hint_is_revalidated_before_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        with pytest.raises(RuntimeError, match="injected"):
            manager._mode_resolver.ensure_canonical("cli:target", fault_after="prepared")
        resolver = manager._mode_resolver
        hint = resolver._recovery_hint
        calls = 0

        def changing(key: str):
            nonlocal calls
            calls += 1
            keys, raw = hint(key)
            return (keys, raw) if calls == 1 else (keys, raw + b" ")

        before = {p: p.read_bytes() for p in manager.sessions_dir.rglob("*") if p.is_file()}
        monkeypatch.setattr(resolver, "_recovery_hint", changing)
        with pytest.raises(SessionModeError):
            resolver.recover("cli:target")
        assert all(path.read_bytes() == raw for path, raw in before.items())
        assert not hasattr(resolver._guard_local, "scope")
        await lease.close()

    asyncio.run(exercise())


def test_snapshot_foreign_task_exit_rejects_before_postcondition_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nanobot.session.execution_owner import execution_lock_guard

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:source")
        resolver = manager._mode_resolver
        snapshot = resolver.locked_snapshot("cli:source")
        snapshot.__enter__()

        async def foreign_exit() -> None:
            with monkeypatch.context() as patch:
                patch.setattr(resolver.storage, "read", lambda *_args, **_kwargs: pytest.fail("foreign postcondition read"))
                with pytest.raises(SessionModeError) as error:
                    snapshot.__exit__(None, None, None)
                assert str(error.value) == "session mode evidence is invalid; offline recovery is required"

        await asyncio.create_task(foreign_exit())
        assert not hasattr(resolver._guard_local, "scope")
        assert not manager._jsonl_store._session_files_lock.is_locked
        with execution_lock_guard(manager._jsonl_store, "cli:source"):
            pass
        with resolver.locked_snapshot("cli:source"):
            pass
        await lease.close()

    asyncio.run(exercise())


def test_snapshot_closed_lifecycle_rejects_reuse_and_exit_before_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:source")
        resolver = manager._mode_resolver
        snapshot = resolver.locked_snapshot("cli:source")
        snapshot.__enter__()
        resolver.close()
        with monkeypatch.context() as patch:
            patch.setattr(resolver.storage, "read", lambda *_args, **_kwargs: pytest.fail("closed postcondition read"))
            with pytest.raises(SessionModeError, match="^session authority is closed$"):
                with resolver.locked_snapshot("cli:source"):
                    pass
            with pytest.raises(SessionModeError, match="^session authority is closed$"):
                snapshot.__exit__(None, None, None)
        assert not hasattr(resolver._guard_local, "scope")
        assert not manager._jsonl_store._session_files_lock.is_locked
        await lease.close()

    asyncio.run(exercise())


def test_partial_multikey_guard_failure_releases_acquired_prefix(tmp_path: Path) -> None:
    from nanobot.session.execution_owner import ExecutionBusyError, execution_lock_guard

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        resolver = manager._mode_resolver
        keys = tuple(sorted(("cli:a", "cli:z"), key=manager._jsonl_store.storage_key))
        with execution_lock_guard(manager._jsonl_store, keys[1]):
            with pytest.raises(ExecutionBusyError):
                with resolver._execution_guards(keys):
                    pytest.fail("partially acquired guards admitted")
            assert not hasattr(resolver._guard_local, "scope")
            with execution_lock_guard(manager._jsonl_store, keys[0]):
                pass
        with resolver._execution_guards(keys):
            pass
        assert not hasattr(resolver._guard_local, "scope")
        await lease.close()

    asyncio.run(exercise())


def test_copied_context_after_snapshot_exit_acquires_fresh_os_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager
    from contextvars import copy_context

    import nanobot.session.execution_owner as execution_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:source")
        resolver = manager._mode_resolver
        with resolver.locked_snapshot("cli:source"):
            copied = copy_context()
        assert not hasattr(resolver._guard_local, "scope")
        guard = execution_module.execution_lock_guard
        attempts: list[str] = []

        @contextmanager
        def trace(backend: object, key: str):
            attempts.append(key)
            with guard(backend, key):
                yield

        monkeypatch.setattr(execution_module, "execution_lock_guard", trace)

        async def probe(*, busy: bool) -> None:
            if busy:
                with pytest.raises(execution_module.ExecutionBusyError):
                    with resolver.locked_snapshot("cli:source"):
                        pass
            else:
                with resolver.locked_snapshot("cli:source"):
                    pass

        with guard(manager._jsonl_store, "cli:source"):
            await asyncio.create_task(probe(busy=True), context=copied)
        await asyncio.create_task(probe(busy=False), context=copied)
        assert attempts == ["cli:source", "cli:source"]
        assert not hasattr(resolver._guard_local, "scope")
        await lease.close()

    asyncio.run(exercise())


def test_reciprocal_forks_take_one_sorted_guard_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    import nanobot.session.execution_owner as execution_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    order: list[str] = []
    guard = execution_module.execution_lock_guard

    @contextmanager
    def trace(backend: object, key: str):
        order.append(key)
        with guard(backend, key):
            yield

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        for key in ("cli:a", "cli:z"):
            manager.admit_persistent_write(key)
        monkeypatch.setattr(execution_module, "execution_lock_guard", trace)
        for source, target in (("cli:a", "cli:z"), ("cli:z", "cli:a")):
            with pytest.raises(SessionModeError):
                manager._mode_resolver.fork(source, target, 0)
        expected = sorted(("cli:a", "cli:z"), key=manager._jsonl_store.storage_key)
        assert order == expected + expected
        await lease.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["reset", "delete", "fork"])
@pytest.mark.parametrize("evidence", ["unresolved", "recovery", "lease"])
def test_ownerless_canonical_mutations_reject_execution_evidence(tmp_path: Path, operation: str, evidence: str) -> None:
    import json

    from nanobot.session.execution_owner import ExecutionOwnershipError

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:source")
        cached = manager.get_or_create("cli:source")
        path = manager._get_session_path("cli:source")
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]["revision"] += 1
        if evidence == "unresolved":
            event = new_source_event({"role": "assistant", "content": "", "tool_calls": [{"id": "call", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]}, run_id="run")
            event["_execution_ids"] = ["execution"]
            rows.append(event)
        else:
            rows[0]["metadata"]["execution_recovery_required" if evidence == "recovery" else "execution_lease"] = True if evidence == "recovery" else {"lease_id": "abandoned"}
        manager._mode_resolver.storage.write(f"{manager._jsonl_store.storage_key('cli:source')}.jsonl", b"".join((json.dumps(row) + "\n").encode() for row in rows))
        before = {p: p.read_bytes() for p in manager.sessions_dir.rglob("*") if p.is_file()}
        calls = {"reset": lambda: manager.reset_session("cli:source"), "delete": lambda: manager.delete_session("cli:source"), "fork": lambda: manager.fork_session_before_user_index("cli:source", "cli:target", 0)}
        with pytest.raises(ExecutionOwnershipError):
            calls[operation]()
        assert all(path.read_bytes() == raw for path, raw in before.items())
        assert manager._cache.get("cli:source") is cached
        await lease.close()

    asyncio.run(exercise())


def test_manager_metadata_upsert_validates_removals_and_creation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        assert not manager.upsert_session_metadata("cli:item", {"title": "draft"})
        assert not (manager.sessions_dir / ".session-mode").exists()
        with pytest.raises((TypeError, ValueError)):
            manager.upsert_session_metadata("cli:item", {"bad": float("nan")}, create_if_absent=True)
        assert not (manager.sessions_dir / ".session-mode").exists()
        updates = {"title": "old", "nested": {"x": 1}, "generation": 999}
        assert manager.upsert_session_metadata("cli:item", updates, create_if_absent=True)
        updates["nested"]["x"] = 2
        cached = manager.get_or_create("cli:item")
        binding = manager.resolve_session_mode("cli:item").binding
        assert manager.upsert_session_metadata(
            "cli:item", {"title": "new", "execution_lease": {}},
            removals=("nested", "source_schema_version"), fsync=True,
        )
        loaded = manager.get_or_create("cli:item")
        assert loaded is not cached and loaded.metadata["title"] == "new"
        assert "nested" not in loaded.metadata and "generation" not in loaded.metadata
        assert "execution_lease" not in loaded.metadata
        assert loaded.metadata["source_schema_version"] == 1
        assert manager.resolve_session_mode("cli:item").binding == binding
        await lease.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["reset", "delete", "fork", "direct_reset", "direct_delete", "direct_fork"])
def test_real_owner_contention_preserves_bytes_and_cache(tmp_path: Path, operation: str) -> None:
    from nanobot.session.execution_owner import ExecutionBusyError

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:item")
        cached = manager.get_or_create("cli:item")
        async with acquire_execution(manager._jsonl_store, "cli:item", run_id="held"):
            before = {p: p.read_bytes() for p in manager.sessions_dir.rglob("*") if p.is_file()}
            calls = {
                "reset": lambda: manager.reset_session("cli:item"),
                "delete": lambda: manager.delete_session("cli:item"),
                "fork": lambda: manager.fork_session_before_user_index("cli:item", "cli:target", 0),
                "direct_reset": lambda: manager._mode_resolver.reset("cli:item", [], owner=None),
                "direct_delete": lambda: manager._mode_resolver.delete("cli:item"),
                "direct_fork": lambda: manager._mode_resolver.fork("cli:item", "cli:target", 0),
            }
            with pytest.raises(ExecutionBusyError):
                calls[operation]()
            assert all(path.read_bytes() == raw for path, raw in before.items())
            assert manager._cache.get("cli:item") is cached
            assert not manager._get_session_path("cli:target").exists()
        await lease.close()

    asyncio.run(exercise())


def test_managed_delete_commits_tombstone_before_recreation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(
            workspace,
            sessions_root=tmp_path / "sessions",
        ).acquire()
        manager = lease.manager
        first = manager.admit_persistent_write("cli:delete")
        assert manager.delete_session("cli:delete")
        assert manager.read_session_snapshot("cli:delete") is None
        second = manager.admit_persistent_write("cli:delete")
        assert first.incarnation != second.incarnation
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_canonical_metadata_revision_keeps_mode_binding_and_cache_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(
            workspace, sessions_root=tmp_path / "sessions"
        ).acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:item")
        cached = manager.get_or_create("cli:item")
        binding = manager.resolve_session_mode("cli:item").binding
        record = next((manager.sessions_dir / ".session-mode" / "records").glob("*.json"))
        before = record.read_bytes()
        assert manager.update_session_metadata("cli:item", {"title": "changed"})
        assert manager.resolve_session_mode("cli:item").binding == binding
        assert record.read_bytes() == before
        refreshed = manager.get_or_create("cli:item")
        assert refreshed is not cached
        assert refreshed.metadata["title"] == "changed"
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_ephemeral_session_is_detached_from_existing_durable_cache(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    durable = manager.get_or_create("cli:item")
    durable.messages.append({"role": "user", "content": "durable"})
    transient = manager.get_or_create_transient("cli:item")
    transient.messages.append({"role": "user", "content": "ephemeral"})
    assert transient is not durable
    assert not transient.policy.persist
    assert manager.get_cached("cli:item") is durable
    assert durable.messages == [{"role": "user", "content": "durable"}]


def test_canonical_fork_preserves_exact_event_prefix_and_drops_identity_metadata(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(
            workspace, sessions_root=tmp_path / "sessions"
        ).acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:source")
        snapshot = manager._jsonl_store.source.load_source_snapshot("cli:source")
        user = new_source_event({"role": "user", "content": "one"}, run_id="run-1")
        user.update(_stream_id="root")
        call = {
            "id": "read-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"one.txt"}'},
        }
        intent = new_source_event(
            {"role": "assistant", "content": "", "tool_calls": [call]},
            run_id="run-1",
        )
        intent.update(
            _execution_ids=["execution-read-1"],
            _stream_id="child",
            _parent_stream_id="root",
        )
        result = new_source_event(
            {
                "role": "tool",
                "name": "read_file",
                "tool_call_id": "read-1",
                "content": "inline tool evidence",
            },
            run_id="run-1",
        )
        result.update(
            _execution_id="execution-read-1",
            _stream_id="child",
            _parent_stream_id="root",
        )
        answer = new_source_event(
            {"role": "assistant", "content": "answer"}, run_id="run-1"
        )
        answer.update(_stream_id="child", _parent_stream_id="root")
        next_user = new_source_event(
            {"role": "user", "content": "two"}, run_id="run-2"
        )
        next_user.update(_stream_id="next")
        events = [user, intent, result, answer, next_user]
        committed = manager._jsonl_store.source.append_source_events(
            "cli:source", snapshot.state, events
        )
        manager._jsonl_store.source.update_metadata(
            "cli:source",
            {
                "portable": {"keep": True},
                "artifact_manifest": {"drop": True},
                "context_checkpoint": {"drop": True},
            },
        )
        target = manager.fork_session_before_user_index("cli:source", "cli:fork", 1)
        assert target is not None
        forked = manager._jsonl_store.source.load_source_snapshot("cli:fork")
        assert forked.events == committed.events[:4]
        assert forked.events[1]["_execution_ids"] == ["execution-read-1"]
        assert forked.events[2]["_execution_id"] == "execution-read-1"
        assert forked.events[2]["content"] == "inline tool evidence"
        assert forked.events[1]["_stream_id"] == "child"
        assert forked.events[1]["_parent_stream_id"] == "root"
        assert forked.state.incarnation != committed.state.incarnation
        assert forked.header["metadata"]["portable"] == {"keep": True}
        assert "artifact_manifest" not in forked.header["metadata"]
        assert "context_checkpoint" not in forked.header["metadata"]
        assert target.provider_state is None
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_legacy_provider_state_survives_manager_read_while_canonical_has_none(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    legacy = Session("cli:legacy", messages=[{"role": "user", "content": "one"}])
    from nanobot.providers.base import ProviderConversationState

    legacy.provider_state = ProviderConversationState(
        kind="fixture",
        provider="fixture",
        model="fixture-model",
        version=1,
        payload={"id": "x"},
    )
    unmanaged.save(legacy, fsync=True)

    async def exercise() -> None:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        loaded = lease.manager.get_or_create("cli:legacy")
        assert loaded.provider_state == legacy.provider_state
        lease.manager.admit_persistent_write("cli:canonical")
        assert lease.manager.get_or_create("cli:canonical").provider_state is None
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_rename_prevalidates_all_modes_before_first_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(
            workspace, sessions_root=tmp_path / "sessions"
        ).acquire()
        manager = lease.manager
        for key in ("cli:a", "cli:b"):
            manager.admit_persistent_write(key)
            manager.update_session_metadata(
                key, {SESSION_MODEL_PRESET_METADATA_KEY: "old"}
            )
        before = deepcopy(manager.read_session_metadata("cli:a"))
        record = manager._mode_resolver._record_name("cli:b")
        raw = manager._mode_resolver.storage.read(record)
        manager._mode_resolver.storage.write(record, raw + b" ")
        with pytest.raises(SessionModeError, match="offline recovery"):
            manager.rename_model_preset("old", "new")
        assert manager.read_session_metadata("cli:a") == before
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_rename_reloads_cache_after_mode_binding_changes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(
            workspace, sessions_root=tmp_path / "sessions"
        ).acquire()
        manager = lease.manager
        manager.admit_persistent_write("cli:item")
        manager.update_session_metadata(
            "cli:item", {SESSION_MODEL_PRESET_METADATA_KEY: "old"}
        )
        stale = manager.get_or_create("cli:item")
        before_binding = manager.resolve_session_mode("cli:item").binding
        async with acquire_execution(
            manager._jsonl_store, "cli:item", run_id="reset"
        ) as owner:
            await owner.quiesce()
            manager._mode_resolver.reset("cli:item", [], owner=owner)
        manager._jsonl_store.source.update_metadata(
            "cli:item", {SESSION_MODEL_PRESET_METADATA_KEY: "current"}
        )
        assert manager.resolve_session_mode("cli:item").binding != before_binding

        assert manager.rename_model_preset("old", "new") == 0
        assert stale.metadata[SESSION_MODEL_PRESET_METADATA_KEY] == "old"
        metadata = manager.read_session_metadata("cli:item")
        assert metadata is not None
        assert metadata["metadata"][SESSION_MODEL_PRESET_METADATA_KEY] == "current"
        assert manager.get_cached("cli:item") is not stale
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_rename_rejects_missing_backing_before_first_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(
            workspace, sessions_root=tmp_path / "sessions"
        ).acquire()
        manager = lease.manager
        for key in ("cli:a", "cli:b"):
            manager.admit_persistent_write(key)
            manager.update_session_metadata(
                key, {SESSION_MODEL_PRESET_METADATA_KEY: "old"}
            )
            manager.invalidate(key)
        first_before = manager._get_session_path("cli:a").read_bytes()
        manager._get_session_path("cli:b").unlink()

        with pytest.raises(SessionModeError):
            manager.rename_model_preset("old", "new")
        assert manager._get_session_path("cli:a").read_bytes() == first_before
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_managed_flush_failure_diagnostic_is_content_free(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise() -> None:
        lease = await DurableSessionAuthority(
            workspace, sessions_root=tmp_path / "sessions"
        ).acquire()
        manager = lease.manager
        key = "cli:P5_PRIVATE_FLUSH_SENTINEL"
        content = "P5_PRIVATE_CONTENT_SENTINEL"
        manager.admit_persistent_write(key)
        cached = manager.get_or_create(key)
        cached.messages.append({"role": "user", "content": content})
        source_path = manager._get_session_path(key)
        source_before = source_path.read_bytes()
        record_name = manager._mode_resolver._record_name(key)
        mode_before = manager._mode_resolver.storage.read(record_name)
        logs: list[str] = []
        sink = logger.add(logs.append, level="WARNING", format="{message}")
        try:
            assert manager.flush_all() == 0
        finally:
            logger.remove(sink)

        rendered = "\n".join(str(message) for message in logs)
        assert "Failed to flush managed session" in rendered
        for private in (key, content, str(source_path), str(workspace), str(tmp_path)):
            assert private not in rendered
        assert source_path.read_bytes() == source_before
        assert manager._mode_resolver.storage.read(record_name) == mode_before
        assert manager._cache.get(key) is cached
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())
