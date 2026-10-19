"""Namespace ownership retains real lease admission and late failure ordering."""

import os
from pathlib import Path

import pytest

from nanobot.session import mode_storage
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.mode_storage import LifetimeWriterLease, SessionModeError


def file_bytes(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_constructor_preflight_delegates_to_relative_storage_owner(tmp_path, monkeypatch):
    original = LifetimeWriterLease.validate_store_construction
    observed = []

    def validate(workspace, root, access):
        observed.append((workspace, root, access))
        return original(workspace, root, access)

    workspace, root = tmp_path / "workspace", tmp_path / "sessions"
    with monkeypatch.context() as patch:
        patch.setattr(LifetimeWriterLease, "validate_store_construction", staticmethod(validate))
        mode_storage.preflight_store_construction(workspace, root, None)
    assert observed == [(workspace, root, None)]
    assert list(tmp_path.iterdir()) == []


def test_write_guard_delegates_nonvirtually_to_backend_owner(tmp_path, monkeypatch):
    original = JsonlSessionStore.require_namespace_write_admission
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    observed = []
    before = file_bytes(tmp_path)

    def require(value, *, select_namespace_claimed, mode_error):
        observed.append(value)
        return original(value, select_namespace_claimed=select_namespace_claimed, mode_error=mode_error)

    def forbidden_override(**kwargs):
        raise AssertionError("instance override must not replace write admission")

    with monkeypatch.context() as patch:
        patch.setattr(JsonlSessionStore, "require_namespace_write_admission", require)
        patch.setattr(backend, "require_namespace_write_admission", forbidden_override)
        mode_storage.require_namespace_write_access(backend)
    assert observed == [backend]
    assert file_bytes(tmp_path) == before


@pytest.mark.parametrize("fault_after", ["witness", "marker"])
def test_preflight_helper_faults_remain_late_and_close_pins(tmp_path, monkeypatch, fault_after):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    original_pin = mode_storage.PinnedDirectoryPath
    original_stat = LifetimeWriterLease._relative_stat
    original_read = LifetimeWriterLease._relative_read
    failure = RuntimeError("controlled late preflight helper failure")
    pinned = []
    before = file_bytes(tmp_path)

    def pin(path):
        value = original_pin(path)
        pinned.append(value.fd)
        return value

    def reject(*args):
        raise failure

    with monkeypatch.context() as patch:
        def after_witness(value, parts):
            result = original_stat(value, parts)
            if parts == (".nanobot", "session-mode-binding.json"):
                patch.setattr(LifetimeWriterLease, "_relative_read", staticmethod(reject))
            return result

        def after_marker(value, parts):
            result = original_read(value, parts)
            patch.setattr(LifetimeWriterLease, "_relative_stat", staticmethod(reject))
            return result

        patch.setattr(mode_storage, "PinnedDirectoryPath", pin)
        if fault_after == "witness":
            patch.setattr(LifetimeWriterLease, "_relative_stat", staticmethod(after_witness))
        else:
            patch.setattr(LifetimeWriterLease, "_relative_read", staticmethod(after_marker))
        with pytest.raises(RuntimeError) as caught:
            mode_storage.preflight_store_construction(backend.workspace, backend.sessions_root, None)
    assert caught.value is failure
    assert len(pinned) == (1 if fault_after == "witness" else 2)
    for fd in pinned:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert file_bytes(tmp_path) == before


def test_guard_selects_namespace_helper_before_workspace_argument(tmp_path, monkeypatch):
    def forbidden_helper(*args):
        raise AssertionError("helper was selected after workspace observation")

    with monkeypatch.context() as patch:
        class ObservingStore(JsonlSessionStore):
            def __getattribute__(self, name):
                value = super().__getattribute__(name)
                fields = object.__getattribute__(self, "__dict__")
                if name == "workspace" and fields.get("observe_namespace_selection", False):
                    patch.setattr(mode_storage, "namespace_claimed", forbidden_helper)
                return value

        backend = ObservingStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
        before = file_bytes(tmp_path)
        backend.observe_namespace_selection = True
        mode_storage.require_namespace_write_access(backend)
        assert file_bytes(tmp_path) == before


def test_guard_uses_access_selected_before_workspace_id_observation(tmp_path):
    workspace, root = tmp_path / "workspace", tmp_path / "sessions"
    workspace.mkdir()
    issuer = LifetimeWriterLease(workspace, root)
    issuer.acquire()
    try:
        access = issuer.issue_access(operator=False)
        backend = JsonlSessionStore(workspace, sessions_root=root, _managed_access=access)
        original = backend.sessions_dir

        class ChangingPath(type(original)):
            @property
            def name(self):
                backend._managed_namespace_access = None
                return super().name

        before = file_bytes(tmp_path)
        backend.sessions_dir = ChangingPath(original)
        try:
            mode_storage.require_namespace_write_access(backend)
        finally:
            backend.sessions_dir = original
            backend._managed_namespace_access = access
        assert file_bytes(tmp_path) == before
    finally:
        issuer.close()


def test_guard_selects_error_factory_after_claim_check(tmp_path, monkeypatch):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    witness = backend.workspace / ".nanobot" / "session-mode-binding.json"
    witness.write_bytes(b"{}\n")
    original = mode_storage.namespace_claimed
    failure = RuntimeError("controlled late write admission failure")
    before = file_bytes(tmp_path)

    with monkeypatch.context() as patch:
        def claimed(*args):
            result = original(*args)
            assert result is True
            patch.setattr(mode_storage, "_mode_error", lambda: failure)
            return result

        patch.setattr(mode_storage, "namespace_claimed", claimed)
        with pytest.raises(RuntimeError) as caught:
            mode_storage.require_namespace_write_access(backend)
    assert caught.value is failure
    assert file_bytes(tmp_path) == before


@pytest.mark.parametrize("claimed", [False, True])
def test_genuine_access_is_required_even_without_registry_claim(tmp_path, claimed):
    workspace, root = tmp_path / "workspace", tmp_path / "sessions"
    workspace.mkdir()
    issuer = LifetimeWriterLease(workspace, root)
    issuer.acquire()
    try:
        access = issuer.issue_access(operator=False)
        backend = JsonlSessionStore(workspace, sessions_root=root, _managed_access=access)
        if claimed:
            (backend.sessions_dir / ".session-mode").mkdir(mode=0o700)
        assert mode_storage.namespace_claimed(workspace, root, backend.sessions_dir.name) is claimed
        before = file_bytes(tmp_path)
        mode_storage.require_namespace_write_access(backend)
        access.revoke()
        with pytest.raises(SessionModeError):
            mode_storage.require_namespace_write_access(backend)
        with pytest.raises(SessionModeError):
            backend.save(Session("cli:denied"))
        assert file_bytes(tmp_path) == before
        assert not backend.get_session_path("cli:denied").exists()
    finally:
        issuer.close()
