"""Checked execution admission preserves real locks, readers and cleanup hooks."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest

from nanobot.session.execution_owner import (
    ExecutionOwner,
    ExecutionOwnershipError,
    acquire_execution,
    execution_lock_guard,
    execution_transaction,
)
from nanobot.session.manager import (
    JsonlSessionStore,
    Session,
    SessionStorageError,
    SessionWriteConflict,
)
from nanobot.session.source_root import SourceRootAuthority
from nanobot.session.source_store import SourceSessionStore


@pytest.fixture
def backend(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def seed(backend, key="test:admission", *, legacy=False):
    session = Session(key, messages=[{"role": "user", "content": "input"}])
    backend.save(session)
    if not legacy:
        backend.source.upgrade_source_session(
            key, expected_revision=session.revision, expected_generation=session.generation,
            expected_incarnation=session.incarnation,
        )
    return key


@pytest.mark.parametrize("destructive", [False, True])
async def test_owner_transaction_keeps_lock_and_live_hook_order(backend, monkeypatch, destructive):
    key = seed(backend)
    async with acquire_execution(backend, key, run_id="order") as owner:
        if destructive:
            await owner.quiesce()
        events = []
        mutex, files = owner._mutex, backend.locked_session_files
        read, validate, drain = (
            SourceSessionStore.load_source_snapshot, owner._validate_unlocked, owner._require_drained,
        )

        @contextmanager
        def observed_mutex():
            with mutex:
                events.append("mutex-enter")
                try:
                    yield
                finally:
                    events.append("mutex-exit")

        @contextmanager
        def observed_files():
            with files():
                events.append("files-enter")
                try:
                    yield
                finally:
                    events.append("files-exit")

        def observed_read(source, *args, **kwargs):
            events.append("read")
            return read(source, *args, **kwargs)

        def observed_validate(*args, **kwargs):
            events.append("validate")
            return validate(*args, **kwargs)

        def observed_drain():
            events.append("drain")
            return drain()

        before = backend.get_session_path(key).read_bytes()
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_mutex", observed_mutex())
            patch.setattr(backend, "locked_session_files", observed_files)
            patch.setattr(SourceSessionStore, "load_source_snapshot", observed_read)
            patch.setattr(owner, "_validate_unlocked", observed_validate)
            patch.setattr(owner, "_require_drained", observed_drain)
            with execution_transaction(backend, key, owner, destructive=destructive):
                events.append("body")
        assert events == [
            "mutex-enter", *(["drain"] if destructive else []), "files-enter", "read",
            "files-enter", "files-exit", "validate", "body", "files-exit", "mutex-exit",
        ]
        assert backend.get_session_path(key).read_bytes() == before


def test_outer_type_fence_runs_before_alleged_owner_attribute_lookup(backend):
    key = seed(backend)
    touched = []

    class AllegedOwner:
        def __getattribute__(self, name):
            touched.append(name)
            raise AssertionError("an invalid owner must not be inspected")

    with pytest.raises(ExecutionOwnershipError):
        with execution_transaction(backend, key, AllegedOwner()):
            pytest.fail("invalid owner admitted")
    assert touched == []


def test_owner_method_also_refuses_subclasses(backend):
    key = seed(backend)

    class DerivedOwner(ExecutionOwner):
        pass

    alleged = object.__new__(DerivedOwner)
    with pytest.raises(ExecutionOwnershipError):
        with ExecutionOwner.transaction(alleged, backend, key):
            pytest.fail("subclass admitted")


async def test_wrong_key_and_expired_owner_refuse_before_storage(backend, monkeypatch):
    key = seed(backend)

    @contextmanager
    def forbidden_storage():
        pytest.fail("invalid owner reached storage")
        yield

    async with acquire_execution(backend, key, run_id="key") as owner:
        with monkeypatch.context() as patch:
            patch.setattr(backend, "locked_session_files", forbidden_storage)
            with pytest.raises(ExecutionOwnershipError):
                with owner.transaction(backend, "test:wrong"):
                    pytest.fail("wrong key admitted")
            with pytest.raises(ExecutionOwnershipError):
                with owner.transaction(backend, key, destructive=True):
                    pytest.fail("undrained owner admitted")
    with monkeypatch.context() as patch:
        patch.setattr(backend, "locked_session_files", forbidden_storage)
        with pytest.raises(ExecutionOwnershipError):
            with owner.transaction(backend, key):
                pytest.fail("expired owner admitted")


async def test_transaction_keeps_registered_task_fence(backend):
    key = seed(backend)
    admitted = []
    async with acquire_execution(backend, key, run_id="tasks") as owner:
        async def unregistered():
            with pytest.raises(ExecutionOwnershipError):
                with owner.transaction(backend, key):
                    pytest.fail("unregistered child admitted")

        async def registered():
            with owner.transaction(backend, key):
                admitted.append("child")

        await asyncio.create_task(unregistered())
        task = asyncio.create_task(registered())
        owner.track_task(task)
        await task
        with owner.transaction(backend, key):
            admitted.append("root")
    assert admitted == ["child", "root"]


async def test_transaction_keeps_backend_and_lease_validation(backend, tmp_path):
    key = seed(backend)
    other = JsonlSessionStore(tmp_path / "other", sessions_root=tmp_path / "other-private")
    seed(other, key)
    async with acquire_execution(backend, key, run_id="lease") as owner:
        with pytest.raises(ExecutionOwnershipError):
            with owner.transaction(other, key):
                pytest.fail("other backend admitted")
        snapshot = owner.read_snapshot()
        token = snapshot.header["metadata"]["execution_lease"]["token"]
        snapshot.header["metadata"]["execution_lease"]["token"] = "changed"
        backend.source._write(snapshot.header, snapshot.events)
        try:
            with pytest.raises(ExecutionOwnershipError):
                with owner.transaction(backend, key):
                    pytest.fail("changed lease admitted")
        finally:
            snapshot.header["metadata"]["execution_lease"]["token"] = token
            backend.source._write(snapshot.header, snapshot.events)


@contextmanager
def read_scope(backend, key, root):
    if root is None:
        with execution_lock_guard(backend, key), backend.locked_session_files():
            yield
    else:
        with root.operation(backend):
            with root.lock_nested_file(
                ".execution-locks", f"{backend.storage_key(key)}.lock", blocking=False,
            ), root.lock_file(".session-files.lock", blocking=True):
                yield


@pytest.mark.parametrize("use_root", [False, True])
@pytest.mark.parametrize("marker", [None, "execution_lease", "execution_recovery_required"])
def test_metadata_admission_is_read_only_and_presence_based(backend, use_root, marker):
    key = seed(backend)
    if marker is not None:
        snapshot = backend.source.load_source_snapshot(key)
        snapshot.header["metadata"][marker] = False
        backend.source._write(snapshot.header, snapshot.events)
    before = backend.get_session_path(key).read_bytes()
    root = SourceRootAuthority.issue(backend) if use_root else None
    try:
        with read_scope(backend, key, root):
            if marker is None:
                assert backend.source.assert_no_execution_metadata(key, root_authority=root) is None
            else:
                with pytest.raises(ExecutionOwnershipError):
                    backend.source.assert_no_execution_metadata(key, root_authority=root)
        assert backend.get_session_path(key).read_bytes() == before
    finally:
        if root is not None:
            root.close()


@pytest.mark.parametrize("mode", ["source", "legacy", "root"])
def test_reader_kwargs_and_metadata_lookup_timing_are_exact(backend, monkeypatch, mode):
    key = seed(backend, legacy=mode == "legacy")
    root = SourceRootAuthority.issue(backend) if mode == "root" else None
    calls, lookups = [], []
    read = SourceSessionStore._read_unlocked

    class ObservedHeader(dict):
        def __getitem__(self, name):
            if name == "metadata":
                lookups.append(name)
            return super().__getitem__(name)

    def observed_read(source, *args, **kwargs):
        calls.append((args, kwargs))
        header, events = read(source, *args, **kwargs)
        return ObservedHeader(header), events

    try:
        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, "_read_unlocked", observed_read)
            with read_scope(backend, key, root):
                assert backend.source.assert_no_execution_metadata(
                    key, allow_legacy=mode == "legacy", root_authority=root,
                ) is None
        expected = {"root_authority": root} if mode == "root" else {"require_source": mode != "legacy"}
        assert calls == [((key,), expected)]
        assert lookups == ["metadata"] * (1 if mode == "root" else 2)
        calls.clear()
        lookups.clear()
        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, "_read_unlocked", observed_read)
            with execution_transaction(
                backend, key, allow_legacy=mode == "legacy", root_authority=root,
            ):
                pass
        assert calls == [((key,), expected)]
        assert lookups == ["metadata"] * (1 if mode == "root" else 2)
    finally:
        if root is not None:
            root.close()


def test_legacy_format_remains_explicit(backend):
    key = seed(backend, legacy=True)
    with read_scope(backend, key, None):
        with pytest.raises(SessionStorageError):
            backend.source.assert_no_execution_metadata(key)
        assert backend.source.assert_no_execution_metadata(key, allow_legacy=True) is None


@pytest.mark.parametrize("use_root", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_read_fault_identity_and_missing_file_translation(backend, monkeypatch, use_root, missing):
    key = seed(backend)
    root = SourceRootAuthority.issue(backend) if use_root else None
    failure = FileNotFoundError("missing") if missing else OSError("read-fault")
    before = backend.get_session_path(key).read_bytes()

    def fail_read(*_args, **_kwargs):
        raise failure

    try:
        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, "_read_unlocked", fail_read)
            with read_scope(backend, key, root):
                with pytest.raises(SessionWriteConflict if missing else OSError) as captured:
                    backend.source.assert_no_execution_metadata(key, root_authority=root)
        if missing:
            assert type(captured.value) is SessionWriteConflict
            assert captured.value.__cause__ is None and captured.value.__suppress_context__
        else:
            assert captured.value is failure
        assert backend.get_session_path(key).read_bytes() == before
    finally:
        if root is not None:
            root.close()


async def test_wrong_or_missing_revocation_capability_cannot_expire_running_owner(backend):
    key = seed(backend)
    async with acquire_execution(backend, key, run_id="capability") as owner:
        finish = asyncio.Event()
        child = asyncio.create_task(finish.wait())
        owner.track_task(child)
        try:
            for capability in (None, object()):
                with pytest.raises(ExecutionOwnershipError):
                    owner.revoke_acquisition(capability)
                owner.validate_dispatch()
                assert not child.done()
            with pytest.raises(TypeError):
                owner.revoke_acquisition()
            owner.validate_dispatch()
        finally:
            finish.set()
            await child


async def test_acquisition_finally_revokes_before_release_after_close_fault(backend, monkeypatch):
    key = seed(backend)
    failure = OSError("close-fault")
    events = []
    with monkeypatch.context() as patch:
        with pytest.raises(OSError) as captured:
            async with acquire_execution(backend, key, run_id="fault") as owner:
                revoke, release = owner.revoke_acquisition, owner._lock.release

                async def fail_close():
                    raise failure

                def observed_revoke(capability):
                    assert owner._active and owner._lock.is_locked
                    events.append("revoke")
                    revoke(capability)

                def observed_release(*args, **kwargs):
                    assert not owner._active
                    events.append("release")
                    return release(*args, **kwargs)

                patch.setattr(owner, "aclose", fail_close)
                patch.setattr(owner, "revoke_acquisition", observed_revoke)
                patch.setattr(owner._lock, "release", observed_release)
        assert captured.value is failure
        assert events == ["revoke", "release"]
    with pytest.raises(ExecutionOwnershipError):
        owner.validate_dispatch()
    async with acquire_execution(backend, key, run_id="recovery") as replacement:
        assert replacement.recovery_required
