"""Destructive writes retire real owners without shortening execution lock lifetime."""

from pathlib import Path

import pytest

from nanobot.session.execution_owner import (
    ExecutionBusyError,
    ExecutionOwner,
    ExecutionOwnershipError,
    acquire_execution,
    execution_lock_guard,
)
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source_store import SourceSessionStore
from tests.session.test_execution_owner import seed


@pytest.fixture
def backend(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def retirement_command():
    command = getattr(ExecutionOwner, "retire_after_destructive_write", None)
    assert callable(command), "owner-local destructive retirement command is missing"
    return command


def mutate(backend, snapshot, owner, operation):
    if operation == "reset":
        return backend.source.reset_source_session(
            snapshot.state.session_key, snapshot.state, [], owner=owner,
        )
    return backend.source.delete_source_session(
        snapshot.state.session_key, snapshot.state, owner=owner,
    )


def require_lock_retained(backend, owner):
    assert owner._lock.is_locked
    with pytest.raises(ExecutionBusyError), execution_lock_guard(backend, owner.key):
        pytest.fail("retirement released the execution lock")


@pytest.mark.parametrize("operation", ["reset", "delete"])
async def test_success_uses_nonvirtual_command_and_late_original_hook(
    backend, monkeypatch, operation,
):
    retirement_command()
    initial = seed(backend)
    path = backend.get_session_path(initial.state.session_key)
    trace = []
    hook = ExecutionOwner._invalidate_after_destructive_write
    fsync = type(backend)._fsync_directory
    unlink = Path.unlink

    async with acquire_execution(backend, initial.state.session_key, run_id="success") as owner:
        snapshot = owner.read_snapshot()
        await owner.quiesce()

        def retire():
            assert owner._active
            require_lock_retained(backend, owner)
            if operation == "reset":
                committed = backend.source.load_source_snapshot(owner.key)
                assert committed.state.generation == snapshot.state.generation + 1
                assert committed.events == []
                assert "execution_lease" not in committed.header["metadata"]
            else:
                assert not path.exists()
            trace.append("retire")
            hook(owner)

        def directory(directory_path):
            trace.append("directory")
            return fsync(directory_path)

        def remove(actual_path, *args, **kwargs):
            if actual_path == path:
                trace.append("unlink")
            return unlink(actual_path, *args, **kwargs)

        def forbidden_override():
            pytest.fail("an instance override redirected source retirement")

        with monkeypatch.context() as patch:
            patch.setattr(owner, "retire_after_destructive_write", forbidden_override)
            patch.setattr(owner, "_invalidate_after_destructive_write", retire)
            patch.setattr(type(backend), "_fsync_directory", staticmethod(directory))
            patch.setattr(Path, "unlink", remove)
            result = mutate(backend, snapshot, owner, operation)
        assert trace == (["directory", "retire"] if operation == "reset"
                         else ["unlink", "retire", "directory"])
        assert owner._active is False
        require_lock_retained(backend, owner)
        with pytest.raises(ExecutionOwnershipError if operation == "reset" else FileNotFoundError):
            owner.validate_dispatch()
        if operation == "reset":
            assert result.state.generation == snapshot.state.generation + 1
        else:
            assert result is None
        await owner.aclose()
        require_lock_retained(backend, owner)
    with execution_lock_guard(backend, owner.key):
        pass


@pytest.mark.parametrize("operation", ["reset", "delete"])
async def test_prewrite_failure_does_not_retire_or_modify_source(backend, monkeypatch, operation):
    retirement_command()
    initial = seed(backend)
    path = backend.get_session_path(initial.state.session_key)
    fault = OSError("destructive write refused")
    retired = []
    unlink = Path.unlink

    async with acquire_execution(backend, initial.state.session_key, run_id="prewrite") as owner:
        snapshot = owner.read_snapshot()
        before = path.read_bytes()
        await owner.quiesce()

        def fail_write(*_args, **_kwargs):
            raise fault

        def fail_unlink(actual_path, *args, **kwargs):
            if actual_path == path:
                raise fault
            return unlink(actual_path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_invalidate_after_destructive_write", lambda: retired.append(True))
            if operation == "reset":
                patch.setattr(SourceSessionStore, "_write", fail_write)
            else:
                patch.setattr(Path, "unlink", fail_unlink)
            with pytest.raises(OSError) as caught:
                mutate(backend, snapshot, owner, operation)
        assert caught.value is fault
        assert retired == [] and owner._active
        assert path.read_bytes() == before
        assert owner.read_snapshot().state == snapshot.state
        require_lock_retained(backend, owner)
    with execution_lock_guard(backend, initial.state.session_key):
        pass


@pytest.mark.parametrize("operation", ["reset", "delete"])
async def test_directory_failure_preserves_distinct_retirement_timing(
    backend, monkeypatch, operation,
):
    retirement_command()
    initial = seed(backend)
    path = backend.get_session_path(initial.state.session_key)
    fault = OSError("directory durability failed")
    retired = []
    hook = ExecutionOwner._invalidate_after_destructive_write

    with pytest.raises(OSError) as caught:
        async with acquire_execution(backend, initial.state.session_key, run_id="durability") as owner:
            snapshot = owner.read_snapshot()
            await owner.quiesce()

            def retire():
                retired.append(True)
                hook(owner)

            def fail_directory(_path):
                raise fault

            with monkeypatch.context() as patch:
                patch.setattr(owner, "_invalidate_after_destructive_write", retire)
                patch.setattr(type(backend), "_fsync_directory", staticmethod(fail_directory))
                try:
                    mutate(backend, snapshot, owner, operation)
                except OSError as failure:
                    assert failure is fault
                    assert owner._active is (operation == "reset")
                    require_lock_retained(backend, owner)
                    raise
    assert caught.value is fault
    assert retired == ([] if operation == "reset" else [True])
    assert owner._active is False
    if operation == "reset":
        committed = backend.source.load_source_snapshot(owner.key)
        assert committed.state.generation == snapshot.state.generation + 1
        assert committed.events == []
        assert "execution_lease" not in committed.header["metadata"]
        assert fault.__notes__ == ["Execution ownership cleanup also failed."]
    else:
        assert not path.exists()
        assert not getattr(fault, "__notes__", [])
    with execution_lock_guard(backend, owner.key):
        pass


@pytest.mark.parametrize("operation", ["reset", "delete"])
@pytest.mark.parametrize("phase", ["before", "after"])
async def test_original_retirement_fault_keeps_first_error_and_lock(
    backend, monkeypatch, operation, phase,
):
    retirement_command()
    initial = seed(backend)
    path = backend.get_session_path(initial.state.session_key)
    fault = RuntimeError("retirement hook failed")
    hook = ExecutionOwner._invalidate_after_destructive_write
    trace = []

    with pytest.raises(RuntimeError) as caught:
        async with acquire_execution(backend, initial.state.session_key, run_id="hook") as owner:
            snapshot = owner.read_snapshot()
            await owner.quiesce()

            def fail_hook():
                trace.append("hook")
                require_lock_retained(backend, owner)
                if phase == "after":
                    hook(owner)
                raise fault

            with monkeypatch.context() as patch:
                patch.setattr(owner, "_invalidate_after_destructive_write", fail_hook)
                try:
                    mutate(backend, snapshot, owner, operation)
                except RuntimeError as failure:
                    assert failure is fault
                    assert owner._active is (phase == "before")
                    require_lock_retained(backend, owner)
                    raise
    assert caught.value is fault and trace == ["hook"]
    assert owner._active is False
    if operation == "reset":
        assert backend.source.load_source_snapshot(owner.key).events == []
    else:
        assert not path.exists()
    assert getattr(fault, "__notes__", []) == (
        ["Execution ownership cleanup also failed."] if phase == "before" else []
    )
    with execution_lock_guard(backend, owner.key):
        pass


async def test_direct_command_is_late_revocation_only_and_idempotent(backend, monkeypatch):
    command = retirement_command()
    initial = seed(backend)
    hook = ExecutionOwner._invalidate_after_destructive_write
    trace = []
    async with acquire_execution(backend, initial.state.session_key, run_id="direct") as owner:
        path = backend.get_session_path(owner.key)
        before = path.read_bytes()

        def first():
            trace.append("first")
            hook(owner)

        def second():
            trace.append("second")
            hook(owner)

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_invalidate_after_destructive_write", first)
            assert command(owner) is None
            patch.setattr(owner, "_invalidate_after_destructive_write", second)
            assert command(owner) is None
        assert trace == ["first", "second"]
        assert path.read_bytes() == before
        assert owner._active is False
        require_lock_retained(backend, owner)
        await owner.aclose()
        require_lock_retained(backend, owner)
        with pytest.raises(ExecutionOwnershipError):
            owner.validate_dispatch()
    with execution_lock_guard(backend, owner.key):
        pass
    async with acquire_execution(backend, owner.key, run_id="recover") as replacement:
        assert replacement.recovery_required
        with pytest.raises(ExecutionOwnershipError):
            replacement.validate_dispatch()
        await replacement.quiesce()
        current = replacement.read_snapshot()
        mutate(backend, current, replacement, "reset")


@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_ownerless_mutation_never_selects_retirement(backend, monkeypatch, operation):
    retirement_command()
    initial = seed(backend)

    def forbidden(_owner):
        pytest.fail("ownerless mutation selected retirement")

    monkeypatch.setattr(ExecutionOwner, "retire_after_destructive_write", forbidden)
    result = mutate(backend, initial, None, operation)
    if operation == "reset":
        assert result.events == []
    else:
        assert result is None
