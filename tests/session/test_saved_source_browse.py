"""Browse-only metadata must come from the exact validated or committed source."""

import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import pytest

from kohaku.session.mode_storage import MODE_JSON_LIMIT
from kohaku.session.runtime_authority import DurableSessionAuthority
from kohaku.session.session_mode import SessionModeError
from kohaku.session.source import new_source_event
from kohaku.session.source_store import SourceSessionStore

KEY = "websocket:saved-source"


@asynccontextmanager
async def _managed(root: Path):
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    lease = await DurableSessionAuthority(workspace, sessions_root=root / "sessions").acquire()
    try:
        yield lease.manager
    finally:
        assert lease.close is not None
        await lease.close()


def _seed(manager, prompts: int = 24) -> None:
    assert manager.upsert_session_metadata(KEY, {"webui": True}, create_if_absent=True)
    source = manager._jsonl_store.source
    current = source.load_source_snapshot(KEY).state
    source.append_source_events(KEY, current, [
        new_source_event({"role": "user", "content": f"prompt\u2028{i}"}, run_id="test")
        for i in range(prompts)
    ])


@contextmanager
def _forbid_source_body(manager, monkeypatch):
    path = manager._get_session_path(KEY).stat()
    read = os.read

    def check_read(fd, length):
        target = os.fstat(fd)
        assert (target.st_dev, target.st_ino) != (path.st_dev, path.st_ino)
        return read(fd, length)

    def unexpected(*args, **kwargs):
        pytest.fail("browse loaded or parsed the source body")

    with monkeypatch.context() as patch:
        patch.setattr(os, "read", check_read)
        patch.setattr(SourceSessionStore, "_read_unlocked", unexpected)
        yield


@pytest.mark.parametrize("prompts", [24, 240])
async def test_browse_uses_acquisition_and_writer_evidence_without_a_body_read(
    tmp_path, monkeypatch, prompts,
):
    async with _managed(tmp_path) as manager:
        _seed(manager, prompts)
        with _forbid_source_body(manager, monkeypatch):
            initial = manager.read_saved_source_metadata(KEY)
        assert initial is not None and initial.webui

    async with _managed(tmp_path) as manager:
        with _forbid_source_body(manager, monkeypatch):
            assert manager.read_saved_source_metadata(KEY) == initial
        path = manager._get_session_path(KEY)
        inode = path.stat().st_ino
        assert manager.upsert_session_metadata(KEY, {
            "title": "A longer title",
            "workspace_scope": {"project_path": str(tmp_path / "workspace")},
        })
        assert path.stat().st_ino != inode
        # A different journal operation invalidates the unrelated mode cache.
        manager.admit_persistent_write("cli:unrelated")
        with _forbid_source_body(manager, monkeypatch):
            latest = manager.read_saved_source_metadata(KEY)
            assert latest is not None
            assert latest.state.same_origin(initial.state)
            assert latest.state.revision == initial.state.revision + 1
            assert latest.workspace_scope == {"project_path": str(tmp_path / "workspace")}
            latest.workspace_scope["project_path"] = "caller mutated a detached value"
            assert manager.read_saved_source_metadata(KEY).workspace_scope != latest.workspace_scope


@pytest.mark.parametrize("change", ["replace", "same_size_edit", "sidecar", "mode_record"])
async def test_browse_does_not_adopt_changed_or_inconsistent_evidence(tmp_path, monkeypatch, change):
    async with _managed(tmp_path) as manager:
        _seed(manager)
        assert manager.read_saved_source_metadata(KEY) is not None
        path = manager._get_session_path(KEY)
        if change == "replace":
            replacement = path.with_suffix(".tmp")
            replacement.write_bytes(path.read_bytes())
            replacement.chmod(0o600)
            replacement.replace(path)
        elif change == "same_size_edit":
            before = path.stat()
            path.write_bytes(path.read_bytes().replace(b'"role": "user"', b'"role": "nope"'))
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif change == "sidecar":
            path.with_suffix(".checkpoint.json").write_text("{}")
        else:
            resolver = manager._mode_resolver
            assert resolver is not None
            resolver.storage.write(resolver._record_name(KEY), b"{}")
        with _forbid_source_body(manager, monkeypatch), pytest.raises(SessionModeError):
            manager.read_saved_source_metadata(KEY)
        if change == "same_size_edit":
            with pytest.raises(SessionModeError):
                manager.resolve_session_mode(KEY)


async def test_browse_refuses_unfinished_reset_even_before_the_source_changes(tmp_path, monkeypatch):
    async with _managed(tmp_path) as manager:
        _seed(manager)
        assert manager.read_saved_source_metadata(KEY) is not None
        path = manager._get_session_path(KEY)
        inode = path.stat().st_ino
        resolver = manager._mode_resolver
        assert resolver is not None
        with pytest.raises(RuntimeError, match="injected session-mode fault"):
            resolver.reset(KEY, [], fault_after="prepared")
        assert path.stat().st_ino == inode
        with _forbid_source_body(manager, monkeypatch), pytest.raises(SessionModeError):
            manager.read_saved_source_metadata(KEY)


async def test_post_replace_failure_does_not_publish_a_browse_proof(tmp_path, monkeypatch):
    async with _managed(tmp_path) as manager:
        _seed(manager)
        old = manager.read_saved_source_metadata(KEY)
        assert old is not None
        backend = manager._jsonl_store
        inode = manager._get_session_path(KEY).stat().st_ino

        def fail_fsync(path):
            raise OSError("injected directory fsync failure")

        with monkeypatch.context() as patch, pytest.raises(OSError, match="injected"):
            patch.setattr(type(backend), "_fsync_directory", staticmethod(fail_fsync))
            backend.source.update_metadata(KEY, {"title": "new title"})
        assert manager._get_session_path(KEY).stat().st_ino != inode
        with _forbid_source_body(manager, monkeypatch), pytest.raises(SessionModeError):
            manager.read_saved_source_metadata(KEY)
        assert manager.resolve_session_mode(KEY).state == "active"
        with _forbid_source_body(manager, monkeypatch):
            checked = manager.read_saved_source_metadata(KEY)
        assert checked is not None and checked.state.revision == old.state.revision + 1


async def test_writer_does_not_stamp_a_mutation_during_directory_sync(tmp_path, monkeypatch):
    async with _managed(tmp_path) as manager:
        _seed(manager)
        backend = manager._jsonl_store
        path = manager._get_session_path(KEY)
        sync_directory = backend._fsync_directory

        def change_after_replace(directory):
            before = path.stat()
            path.write_bytes(path.read_bytes().replace(b'"role": "user"', b'"role": "nope"'))
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            sync_directory(directory)

        with monkeypatch.context() as patch:
            patch.setattr(type(backend), "_fsync_directory", staticmethod(change_after_replace))
            backend.source.update_metadata(KEY, {"title": "committed"})
        with _forbid_source_body(manager, monkeypatch), pytest.raises(SessionModeError):
            manager.read_saved_source_metadata(KEY)


async def test_writer_proof_obeys_the_existing_header_byte_bound(tmp_path, monkeypatch):
    async with _managed(tmp_path) as manager:
        _seed(manager)
        # UTF-8 exceeds the header bound even though the character count fits.
        manager._jsonl_store.source.update_metadata(KEY, {"title": "é" * (MODE_JSON_LIMIT // 2)})
        with _forbid_source_body(manager, monkeypatch), pytest.raises(SessionModeError):
            manager.read_saved_source_metadata(KEY)
        with pytest.raises(SessionModeError):
            manager.resolve_session_mode(KEY)


async def test_descriptor_relative_writer_stamps_the_exact_replacement(tmp_path, monkeypatch):
    async with _managed(tmp_path) as manager:
        _seed(manager)
        source = manager._jsonl_store.source
        root = source.create_source_root_authority()
        try:
            with source.execution_transaction(KEY, root_authority=root):
                header, events = source._read_unlocked(KEY, root_authority=root)
                header["revision"] += 1
                committed = source._write(header, events, root_authority=root)
            with _forbid_source_body(manager, monkeypatch):
                assert manager.read_saved_source_metadata(KEY).state == committed.state
        finally:
            root.close()


async def test_reset_and_recreation_have_distinct_browse_origins(tmp_path, monkeypatch):
    async with _managed(tmp_path) as manager:
        _seed(manager)
        before = manager.read_saved_source_metadata(KEY)
        assert before is not None
        assert manager.reset_session(KEY) is not None
        with _forbid_source_body(manager, monkeypatch):
            after_reset = manager.read_saved_source_metadata(KEY)
        assert after_reset is not None
        assert not after_reset.state.same_origin(before.state)

        assert manager.delete_session(KEY)
        assert manager.read_saved_source_metadata(KEY) is None
        assert manager.upsert_session_metadata(KEY, {"webui": True}, create_if_absent=True)
        with _forbid_source_body(manager, monkeypatch):
            recreated = manager.read_saved_source_metadata(KEY)
        assert recreated is not None
        assert not recreated.state.same_origin(before.state)
        assert not recreated.state.same_origin(after_reset.state)
