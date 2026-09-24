from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nanobot.session.manager import JsonlSessionStore
from nanobot.session.recovery_locator import (
    RecoveryLocatorError,
    locate_recovery_candidate,
)

WORKSPACE_ID = "a" * 32


def _layout(tmp_path: Path, key: str = "telegram:chat-1") -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_dir = workspace / ".nanobot"
    state_dir.mkdir()
    (state_dir / "workspace-id").write_text(f"{WORKSPACE_ID}\n", encoding="ascii")

    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    namespace = sessions_root / WORKSPACE_ID
    namespace.mkdir()
    (namespace / ".workspace").write_text(
        f"{workspace.resolve()}\n", encoding="utf-8"
    )
    session_path = namespace / f"{JsonlSessionStore.storage_key(key)}.jsonl"
    session_path.write_text("existing session bytes\n", encoding="utf-8")
    return workspace, sessions_root, session_path


def _locate(workspace: Path, sessions_root: Path, key: str = "telegram:chat-1"):
    return locate_recovery_candidate(workspace, sessions_root, key)


def _expect_rejected(workspace: Path, sessions_root: Path, key: str = "telegram:chat-1"):
    with pytest.raises(RecoveryLocatorError) as error:
        _locate(workspace, sessions_root, key)
    assert str(workspace) not in str(error.value)
    assert str(sessions_root) not in str(error.value)
    assert key not in str(error.value)


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes | None, int, int, int, int, int, int]]:
    result: dict[str, tuple[bytes | None, int, int, int, int, int, int]] = {}
    for path in (root, *root.rglob("*")):
        info = path.lstat()
        payload = (
            path.read_bytes()
            if path.name in {"workspace-id", ".workspace"}
            and path.is_file()
            and not path.is_symlink()
            else None
        )
        result[str(path.relative_to(root))] = (
            payload,
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    return result


def test_locates_existing_canonical_session_without_constructing_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, expected_path = _layout(tmp_path)
    before = _tree_snapshot(tmp_path)

    def forbidden_constructor(*args, **kwargs) -> None:
        raise AssertionError("recovery locator must not construct JsonlSessionStore")

    monkeypatch.setattr(JsonlSessionStore, "__init__", forbidden_constructor)
    candidate = _locate(workspace, sessions_root)

    assert candidate.workspace_id == WORKSPACE_ID
    assert candidate.session_key == "telegram:chat-1"
    assert candidate.session_path == expected_path
    assert candidate.namespace_path == expected_path.parent
    assert candidate.requires_authority_revalidation is True
    with pytest.raises(FrozenInstanceError):
        candidate.workspace_id = "b" * 32
    assert _tree_snapshot(tmp_path) == before


def test_selects_canonical_file_when_a_lossy_collision_also_exists(tmp_path: Path) -> None:
    key = "a:b"
    workspace, sessions_root, canonical_path = _layout(tmp_path, key)
    lossy_path = canonical_path.parent / "a_b.jsonl"
    lossy_path.write_text("legacy lossy bytes\n", encoding="utf-8")

    candidate = _locate(workspace, sessions_root, key)

    assert candidate.session_path == canonical_path
    assert candidate.session_path != lossy_path


def test_rejects_a_lossy_only_session_file(tmp_path: Path) -> None:
    key = "a:b"
    workspace, sessions_root, canonical_path = _layout(tmp_path, key)
    canonical_path.unlink()
    (canonical_path.parent / "a_b.jsonl").write_text("legacy bytes\n", encoding="utf-8")

    _expect_rejected(workspace, sessions_root, key)


@pytest.mark.parametrize("missing", ["workspace-id", "namespace", "namespace-marker", "session"])
def test_rejects_missing_required_entries(tmp_path: Path, missing: str) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    if missing == "workspace-id":
        (workspace / ".nanobot" / "workspace-id").unlink()
    elif missing == "namespace":
        import shutil

        shutil.rmtree(session_path.parent)
    elif missing == "namespace-marker":
        (session_path.parent / ".workspace").unlink()
    else:
        session_path.unlink()

    _expect_rejected(workspace, sessions_root)


def test_rejects_invalid_and_oversized_markers(tmp_path: Path) -> None:
    workspace, sessions_root, _ = _layout(tmp_path)
    workspace_id_path = workspace / ".nanobot" / "workspace-id"
    workspace_id_path.write_bytes(b"A" * 80)
    _expect_rejected(workspace, sessions_root)


def test_rejects_copied_workspace_at_another_canonical_path(tmp_path: Path) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    copied_workspace = tmp_path / "copied-workspace"
    shutil.copytree(workspace, copied_workspace)

    _expect_rejected(copied_workspace, sessions_root)


def test_rejects_competing_namespace_marker_for_same_workspace(tmp_path: Path) -> None:
    workspace, sessions_root, _ = _layout(tmp_path)
    competing = sessions_root / ("b" * 32)
    competing.mkdir()
    (competing / ".workspace").write_text(
        f"{workspace.resolve()}\n", encoding="utf-8"
    )

    _expect_rejected(workspace, sessions_root)


def test_rejects_symlinked_path_components(tmp_path: Path) -> None:
    workspace, sessions_root, _ = _layout(tmp_path)
    state_dir = workspace / ".nanobot"
    saved_state = workspace / ".nanobot.saved"
    state_dir.rename(saved_state)
    state_dir.symlink_to(saved_state, target_is_directory=True)

    _expect_rejected(workspace, sessions_root)


def test_rejects_symlinked_namespace_or_session(tmp_path: Path) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    saved_session = session_path.with_suffix(".saved")
    session_path.rename(saved_session)
    session_path.symlink_to(saved_session)

    _expect_rejected(workspace, sessions_root)


def test_rejects_symlinked_namespace_marker(tmp_path: Path) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    marker = session_path.parent / ".workspace"
    saved_marker = session_path.parent / ".workspace.saved"
    marker.rename(saved_marker)
    marker.symlink_to(saved_marker)

    _expect_rejected(workspace, sessions_root)


def test_rejects_symlinked_namespace_directory(tmp_path: Path) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    namespace = session_path.parent
    saved_namespace = sessions_root / "saved-namespace"
    namespace.rename(saved_namespace)
    namespace.symlink_to(saved_namespace, target_is_directory=True)

    _expect_rejected(workspace, sessions_root)


def test_rejects_symlinked_external_root(tmp_path: Path) -> None:
    workspace, sessions_root, _ = _layout(tmp_path)
    linked_root = tmp_path / "sessions-link"
    linked_root.symlink_to(sessions_root, target_is_directory=True)

    _expect_rejected(workspace, linked_root)


def test_rejects_oversized_namespace_binding_marker(tmp_path: Path) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    (session_path.parent / ".workspace").write_bytes(b"x" * 20_000)

    _expect_rejected(workspace, sessions_root)


@pytest.mark.parametrize("target", ["workspace", "sessions-root", "state", "namespace"])
def test_rejects_world_writable_controlled_directories(tmp_path: Path, target: str) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    controlled_directory = {
        "workspace": workspace,
        "sessions-root": sessions_root,
        "state": workspace / ".nanobot",
        "namespace": session_path.parent,
    }[target]
    controlled_directory.chmod(0o777)

    _expect_rejected(workspace, sessions_root)


@pytest.mark.parametrize("target", ["workspace-id", "namespace-marker", "session"])
def test_rejects_group_or_other_writable_files(tmp_path: Path, target: str) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    controlled_file = {
        "workspace-id": workspace / ".nanobot" / "workspace-id",
        "namespace-marker": session_path.parent / ".workspace",
        "session": session_path,
    }[target]
    controlled_file.chmod(0o666)

    _expect_rejected(workspace, sessions_root)


def test_accepts_legacy_0755_controlled_directories(tmp_path: Path) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    for path in (workspace, sessions_root, workspace / ".nanobot", session_path.parent):
        path.chmod(0o755)

    candidate = _locate(workspace, sessions_root)

    assert candidate.session_path == session_path


def test_rejects_foreign_owned_marker_from_stat_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, _ = _layout(tmp_path)
    import nanobot.session.recovery_locator as locator

    original_stat = os.stat

    class ForeignOwnerStat:
        def __init__(self, original: os.stat_result) -> None:
            self._original = original

        def __getattr__(self, name: str):
            if name == "st_uid":
                return os.geteuid() + 1
            return getattr(self._original, name)

    def inject_foreign_owner(path, **kwargs):
        result = original_stat(path, **kwargs)
        if path == "workspace-id":
            return ForeignOwnerStat(result)
        return result

    monkeypatch.setattr(locator, "_require_descriptor_support", lambda: None)
    monkeypatch.setattr(locator.os, "stat", inject_foreign_owner)

    _expect_rejected(workspace, sessions_root)


def test_rejects_source_mode_change_during_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    import nanobot.session.recovery_locator as locator

    original_check = locator._check_file_entry

    def change_mode_before_recheck(parent_fd: int, name: str, expected: os.stat_result) -> None:
        if name == session_path.name:
            session_path.chmod(0o666)
        original_check(parent_fd, name, expected)

    monkeypatch.setattr(locator, "_check_file_entry", change_mode_before_recheck)

    _expect_rejected(workspace, sessions_root)


def test_rejects_namespace_mode_change_during_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    import nanobot.session.recovery_locator as locator

    original_check = locator._check_directory_entry
    changed = False

    def change_mode_after_open(parent_fd: int, name: str, expected: os.stat_result) -> None:
        nonlocal changed
        original_check(parent_fd, name, expected)
        if name == WORKSPACE_ID and not changed:
            changed = True
            session_path.parent.chmod(0o777)

    monkeypatch.setattr(locator, "_check_directory_entry", change_mode_after_open)

    _expect_rejected(workspace, sessions_root)


def test_closes_all_descriptors_after_injected_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, _ = _layout(tmp_path)
    import nanobot.session.recovery_locator as locator

    opened: list[int] = []
    closed: list[int] = []
    original_open = os.open
    original_close = os.close

    def track_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def track_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    def injected_failure(parent_fd: int, name: str, expected: os.stat_result) -> None:
        raise RecoveryLocatorError()

    monkeypatch.setattr(locator, "_require_descriptor_support", lambda: None)
    monkeypatch.setattr(locator.os, "open", track_open)
    monkeypatch.setattr(locator.os, "close", track_close)
    monkeypatch.setattr(locator, "_check_file_entry", injected_failure)

    with pytest.raises(RecoveryLocatorError):
        _locate(workspace, sessions_root)

    assert Counter(opened) == Counter(closed)


def test_rejects_namespace_path_replacement_during_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    import nanobot.session.recovery_locator as locator

    original = locator._check_directory_entry
    changed = False

    def replace_after_first_open(parent_fd: int, name: str, expected: os.stat_result) -> None:
        nonlocal changed
        original(parent_fd, name, expected)
        if name == WORKSPACE_ID and not changed:
            changed = True
            moved = sessions_root / "moved-namespace"
            os.rename(sessions_root / WORKSPACE_ID, moved)
            replacement = sessions_root / WORKSPACE_ID
            replacement.mkdir()
            (replacement / ".workspace").write_text(
                f"{workspace.resolve()}\n", encoding="utf-8"
            )
            (replacement / session_path.name).write_text("replacement\n", encoding="utf-8")

    monkeypatch.setattr(locator, "_check_directory_entry", replace_after_first_open)
    _expect_rejected(workspace, sessions_root)


def test_rejects_missing_external_sessions_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_dir = workspace / ".nanobot"
    state_dir.mkdir()
    (state_dir / "workspace-id").write_text(f"{WORKSPACE_ID}\n", encoding="ascii")

    _expect_rejected(workspace, tmp_path / "missing-sessions")


def test_never_opens_the_session_source_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, sessions_root, session_path = _layout(tmp_path)
    import nanobot.session.recovery_locator as locator

    original_open = os.open
    monkeypatch.setattr(locator, "_require_descriptor_support", lambda: None)

    def guarded_open(path, *args, **kwargs):
        if path == session_path.name and kwargs.get("dir_fd") is not None:
            raise AssertionError("recovery locator opened the session source")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(locator.os, "open", guarded_open)

    candidate = _locate(workspace, sessions_root)

    assert candidate.session_path == session_path


def test_rejects_blank_or_non_string_session_keys(tmp_path: Path) -> None:
    workspace, sessions_root, _ = _layout(tmp_path)
    for key in ("", " \t\n", None, 123):
        with pytest.raises(RecoveryLocatorError):
            locate_recovery_candidate(workspace, sessions_root, key)  # type: ignore[arg-type]
