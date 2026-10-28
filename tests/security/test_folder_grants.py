"""Persistent local grants are strict, revisioned, and fail closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.security.folder_grants import (
    FolderGrant,
    FolderGrantStore,
    FolderGrantsUnavailableError,
    FolderGrantUpsertResult,
    GrantRevisionConflictError,
    NativeProcessesActiveError,
    NativeProcessesUnknownError,
)


def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FolderGrantStore:
    webui = tmp_path / "instance" / "webui"
    monkeypatch.setattr("nanobot.security.folder_grants.get_webui_dir", lambda: webui)
    return FolderGrantStore(agent_home=tmp_path / "agent", instance_root=tmp_path / "instance")


def test_store_starts_legacy_and_activation_survives_empty_grants(tmp_path, monkeypatch):
    grant_store = _store(tmp_path, monkeypatch)

    assert grant_store.snapshot().enabled is False
    assert grant_store.snapshot().revision == 0

    snapshot = grant_store.revoke("/missing", expected_revision=0)

    assert snapshot.enabled is True
    assert snapshot.revision == 0
    assert snapshot.grants == ()
    assert json.loads(grant_store.path.read_text()) == {
        "version": 1,
        "enabled": True,
        "revision": 0,
        "grants": [],
    }


def test_upsert_canonicalizes_and_idempotence_keeps_revision(tmp_path, monkeypatch):
    grant_store = _store(tmp_path, monkeypatch)
    root = tmp_path / "projects" / "one"
    root.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    first = grant_store.upsert(
        {"path": str(alias), "file_access": "read_write", "commands": True},
        expected_revision=0,
    )
    repeated = grant_store.upsert(
        {"path": str(root), "file_access": "read_write", "commands": True},
        expected_revision=1,
    )

    assert first.revision == repeated.revision == 1
    assert first.grants == (FolderGrant(path=str(root.resolve()), file_access="read_write", commands=True),)


def test_upsert_result_keeps_validated_spelling_and_actual_noop_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    project = home / "project"
    nested = home / "nested"
    project.mkdir(parents=True)
    nested.mkdir()
    alias = tmp_path / "project-alias"
    alias.symlink_to(project, target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))
    store = _store(tmp_path, monkeypatch)
    expected = {"file_access": "read_only", "commands": False}

    initial = store.upsert({"path": str(project), **expected}, expected_revision=0)
    submitted_paths = (
        str(alias),
        "~/project",
        str(home / "nested" / ".." / "project"),
        str(project),
    )

    for submitted_path in submitted_paths:
        operation = store.upsert_with_result(
            {"path": submitted_path, **expected},
            expected_revision=1,
        )

        assert isinstance(operation, FolderGrantUpsertResult)
        assert operation.submitted_path == submitted_path
        assert operation.grant.path == str(project.resolve())
        assert operation.grant.file_access == "read_only"
        assert operation.grant.commands is False
        assert operation.snapshot == initial

    # The legacy API still returns only the committed snapshot.
    assert store.upsert({"path": str(alias), **expected}, expected_revision=1) == initial


def test_ui_input_expands_home_before_persisting_canonical_root(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "project"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    grant_store = _store(tmp_path, monkeypatch)

    snapshot = grant_store.upsert(
        {"path": "~/project", "file_access": "read_only", "commands": False},
        expected_revision=0,
    )

    assert snapshot.grants[0].path == str(project.resolve())
    assert "~/project" not in grant_store.path.read_text(encoding="utf-8")


def test_stale_noop_and_revoke_missing_are_revision_checked(tmp_path, monkeypatch):
    grant_store = _store(tmp_path, monkeypatch)
    root = tmp_path / "project"
    root.mkdir()
    grant_store.upsert(
        {"path": str(root), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )

    with pytest.raises(GrantRevisionConflictError):
        grant_store.revoke(str(tmp_path / "missing"), expected_revision=0)
    unchanged = grant_store.revoke(str(tmp_path / "missing"), expected_revision=1)

    assert unchanged.revision == 1


@pytest.mark.parametrize(
    "value",
    [
        {"path": "relative", "file_access": "read_only", "commands": False},
        {"path": "/tmp", "file_access": "admin", "commands": False},
        {"path": "/tmp", "file_access": "read_only", "commands": 1},
        {"path": "/tmp", "file_access": "read_only", "commands": False, "read": True},
    ],
)
def test_invalid_grant_input_is_rejected(tmp_path, monkeypatch, value):
    grant_store = _store(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        grant_store.upsert(value, expected_revision=0)


def test_store_rejects_private_runtime_overlap_and_noncanonical_identity(tmp_path, monkeypatch):
    grant_store = _store(tmp_path, monkeypatch)
    private_child = tmp_path / "instance" / "credentials"
    private_child.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "private-alias"
    alias.symlink_to(private_child, target_is_directory=True)

    with pytest.raises(ValueError, match="private"):
        grant_store.upsert(
            {"path": str(tmp_path / "instance"), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
    with pytest.raises(ValueError, match="private"):
        grant_store.upsert(
            {"path": str(alias), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )

    grant_store.upsert(
        {"path": str(outside), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    outside.rename(tmp_path / "moved")
    (tmp_path / "outside").symlink_to(tmp_path / "moved", target_is_directory=True)
    assert grant_store.effective_grant_for(tmp_path / "outside" / "file.txt") is None


def test_store_restart_loads_snapshot_and_external_replacement_fails_closed(tmp_path, monkeypatch):
    grant_store = _store(tmp_path, monkeypatch)
    root = tmp_path / "project"
    root.mkdir()
    grant_store.upsert(
        {"path": str(root), "file_access": "read_write", "commands": False},
        expected_revision=0,
    )
    restarted = _store(tmp_path, monkeypatch)
    assert restarted.snapshot().enabled is True
    assert restarted.snapshot().revision == 1

    restarted.path.write_text("{}", encoding="utf-8")
    with pytest.raises(FolderGrantsUnavailableError):
        restarted.snapshot()


def test_store_deletion_after_activation_is_fail_closed_after_restart(tmp_path, monkeypatch):
    grant_store = _store(tmp_path, monkeypatch)
    root = tmp_path / "project"
    root.mkdir()
    grant_store.upsert(
        {"path": str(root), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    grant_store.path.unlink()

    restarted = _store(tmp_path, monkeypatch)

    assert restarted.view.is_managed is True
    with pytest.raises(FolderGrantsUnavailableError):
        restarted.snapshot()


def test_first_activation_waits_for_owned_native_process_reservation(tmp_path, monkeypatch):
    grant_store = _store(tmp_path, monkeypatch)
    root = tmp_path / "project"
    root.mkdir()
    reservation = grant_store.view.try_begin_native_process()
    assert reservation is not None

    with pytest.raises(NativeProcessesActiveError, match="native_processes_active"):
        grant_store.upsert(
            {"path": str(root), "file_access": "read_write", "commands": True},
            expected_revision=0,
        )

    assert reservation.cancel_if_unlaunched() is True
    snapshot = grant_store.upsert(
        {"path": str(root), "file_access": "read_write", "commands": True},
        expected_revision=0,
    )
    assert snapshot.enabled is True
    assert snapshot.revision == 1


def test_unproven_native_admission_stays_unknown_after_restart(tmp_path, monkeypatch):
    grant_store = _store(tmp_path, monkeypatch)
    root = tmp_path / "project"
    root.mkdir()
    reservation = grant_store.view.try_begin_native_process()
    assert reservation is not None
    reservation.mark_unproven()

    with pytest.raises(NativeProcessesUnknownError, match="native_processes_unknown"):
        grant_store.upsert(
            {"path": str(root), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )

    restarted = _store(tmp_path, monkeypatch)
    with pytest.raises(NativeProcessesUnknownError, match="native_processes_unknown"):
        restarted.upsert(
            {"path": str(root), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )


@pytest.mark.parametrize("path", ["/proc", "/sys", "/dev", "/run", "/"])
def test_command_grant_cannot_cover_host_control_trees(path):
    with pytest.raises(ValueError, match="cannot cover"):
        FolderGrant.model_validate(
            {"path": path, "file_access": "read_only", "commands": True}
        )


def test_workspace_factory_shares_committed_view_and_writer(tmp_path, monkeypatch):
    webui = tmp_path / "instance" / "webui"
    monkeypatch.setattr("nanobot.security.folder_grants.get_webui_dir", lambda: webui)
    home = tmp_path / "agent"
    root = tmp_path / "project"
    home.mkdir()
    root.mkdir()

    first = FolderGrantStore.for_workspace(home)
    second = FolderGrantStore.for_workspace(home)
    first.upsert(
        {"path": str(root), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )

    assert first is second
    assert second.snapshot().revision == 1
    assert second.effective_grant_for(root / "file.txt") is not None


def test_missing_grant_root_remains_metadata_but_grants_no_access(tmp_path, monkeypatch):
    grant_store = _store(tmp_path, monkeypatch)
    root = tmp_path / "project"
    root.mkdir()
    grant_store.upsert(
        {"path": str(root), "file_access": "read_write", "commands": False},
        expected_revision=0,
    )
    root.rmdir()

    assert grant_store.snapshot().grants[0].path == str(root.resolve())
    assert grant_store.effective_grant_for(root / "file.txt") is None
