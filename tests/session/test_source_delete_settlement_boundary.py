"""Post-delete backend settlement preserves original hooks and observation order."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.session.execution_owner import (
    ExecutionBusyError,
    ExecutionOwner,
    acquire_execution,
    execution_lock_guard,
)
from nanobot.session.manager import JsonlSessionStore, SessionWriteConflict
from nanobot.session.source_store import SourceSessionStore
from tests.session.test_execution_owner import seed


@pytest.fixture(autouse=True)
def _require_owner_operation() -> None:
    assert callable(getattr(JsonlSessionStore, "settle_source_deletion", None)), (
        "source-delete backend settlement operation is missing"
    )


class _ProbeBackend(JsonlSessionStore):
    def __init__(self, workspace: Path, sessions_root: Path, trace: list[str]) -> None:
        self.armed = False
        self.trace = trace
        self.lookup_fault: RuntimeError | None = None
        self.directory_fault: RuntimeError | None = None
        super().__init__(workspace, sessions_root=sessions_root)

    def __getattribute__(self, name: str) -> object:
        if name in {"_fsync_directory", "sessions_dir"} and self.armed:
            if name == "_fsync_directory":
                self.trace.append("lookup")
                if self.lookup_fault is not None:
                    raise self.lookup_fault
            else:
                self.trace.append("directory")
                if self.directory_fault is not None:
                    raise self.directory_fault
        return super().__getattribute__(name)


@pytest.mark.parametrize("failure", [None, "observe1", "lookup", "observe2", "directory", "fsync"])
def test_private_hook_precedes_second_backend_and_namespace_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    trace: list[str] = []
    first = _ProbeBackend(tmp_path / "first-workspace", tmp_path / "first-private", trace)
    second = _ProbeBackend(tmp_path / "second-workspace", tmp_path / "second-private", trace)
    second_directory = second.sessions_dir
    fault = RuntimeError("synthetic settlement fault")
    observed = 0

    def sync(directory: Path) -> None:
        trace.append("fsync")
        assert directory == second_directory
        if failure == "fsync":
            raise fault

    def forbidden_later_hook(_directory: Path) -> None:
        pytest.fail("namespace observation redirected the already selected private hook")

    def observe() -> JsonlSessionStore:
        nonlocal observed
        observed += 1
        phase = f"observe{observed}"
        trace.append(phase)
        if failure == phase:
            raise fault
        if observed == 1:
            return first
        assert observed == 2
        object.__setattr__(first, "_fsync_directory", forbidden_later_hook)
        return second

    monkeypatch.setattr(first, "_fsync_directory", sync)
    first.lookup_fault = fault if failure == "lookup" else None
    second.directory_fault = fault if failure == "directory" else None
    first.armed = second.armed = True

    if failure is None:
        assert JsonlSessionStore.settle_source_deletion(None, observe) is None
    else:
        with pytest.raises(RuntimeError) as caught:
            JsonlSessionStore.settle_source_deletion(None, observe)
        assert caught.value is fault

    order = ["observe1", "lookup", "observe2", "directory", "fsync"]
    assert trace == (order if failure is None else order[:order.index(failure) + 1])


class _ObservedSource(SourceSessionStore):
    def __init__(self, backend: JsonlSessionStore, trace: list[str]) -> None:
        self.deleted = False
        self.trace = trace
        super().__init__(backend)

    def __getattribute__(self, name: str) -> object:
        if name == "_backend" and self.deleted:
            self.trace.append("backend")
        return super().__getattribute__(name)


@pytest.mark.parametrize("with_owner", [False, True])
async def test_real_delete_uses_class_command_after_unlink_before_late_backend_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_owner: bool,
) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    initial = seed(backend)
    path = backend.get_session_path(initial.state.session_key)
    trace: list[str] = []
    source = _ObservedSource(backend, trace)
    command = JsonlSessionStore.settle_source_deletion
    real_unlink = Path.unlink
    real_sync = backend._fsync_directory
    original_retirement = ExecutionOwner._invalidate_after_destructive_write
    expected_owner: ExecutionOwner | None = None

    def unlink(current: Path, *args: object, **kwargs: object) -> None:
        real_unlink(current, *args, **kwargs)
        if current == path:
            trace.append("unlink")
            source.deleted = True

    def retire() -> None:
        assert expected_owner is not None and expected_owner._lock.is_locked
        assert not path.exists()
        trace.append("retire")
        original_retirement(expected_owner)

    def settle(owner: ExecutionOwner | None, observe: Callable[[], JsonlSessionStore]) -> None:
        assert owner is expected_owner and backend.session_files_locked
        assert source.deleted and not path.exists()
        trace.append("settle")
        command(owner, observe)

    def sync(directory: Path) -> None:
        assert backend.session_files_locked and not path.exists()
        if expected_owner is not None:
            assert not expected_owner._active and expected_owner._lock.is_locked
            with pytest.raises(ExecutionBusyError), execution_lock_guard(backend, expected_owner.key):
                pytest.fail("settlement released the execution lock")
        trace.append("fsync")
        real_sync(directory)

    def forbidden_instance_command(*_args: object) -> None:
        pytest.fail("instance override redirected backend settlement")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", unlink)
        patch.setattr(JsonlSessionStore, "settle_source_deletion", staticmethod(settle))
        patch.setattr(backend, "settle_source_deletion", forbidden_instance_command)
        patch.setattr(backend, "_fsync_directory", sync)
        if with_owner:
            async with acquire_execution(backend, initial.state.session_key, run_id="settlement") as owner:
                expected_owner = owner
                snapshot = owner.read_snapshot()
                await owner.quiesce()
                patch.setattr(owner, "_invalidate_after_destructive_write", retire)
                assert source.delete_source_session(owner.key, snapshot.state, owner=owner) is None
                assert not owner._active and owner._lock.is_locked
        else:
            assert source.delete_source_session(initial.state.session_key, initial.state) is None

    assert trace == ["unlink", "settle", *(["retire"] if with_owner else []), "backend", "backend", "fsync"]
    assert not path.exists()


@pytest.mark.parametrize("failure", ["stale", "unlink"])
def test_predelete_failure_never_invokes_backend_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    snapshot = seed(backend)
    path = backend.get_session_path(snapshot.state.session_key)
    original = path.read_bytes()
    fault = OSError("synthetic unlink failure")
    real_unlink = Path.unlink

    def forbidden(*_args: object) -> None:
        pytest.fail("predelete failure invoked settlement")

    def unlink(current: Path, *args: object, **kwargs: object) -> None:
        if current == path:
            raise fault
        real_unlink(current, *args, **kwargs)

    monkeypatch.setattr(JsonlSessionStore, "settle_source_deletion", staticmethod(forbidden))
    expected = snapshot.state
    if failure == "stale":
        expected = replace(expected, revision=expected.revision + 1)
        with pytest.raises(SessionWriteConflict):
            backend.source.delete_source_session(expected.session_key, expected)
    else:
        monkeypatch.setattr(Path, "unlink", unlink)
        with pytest.raises(OSError) as caught:
            backend.source.delete_source_session(expected.session_key, expected)
        assert caught.value is fault
    assert path.read_bytes() == original
