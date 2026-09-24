"""Read-only locator for an existing canonical session recovery candidate.

The returned candidate is a detached path and identity snapshot only. Every
subsequent source access must pass through a checked authority that revalidates
the workspace, namespace, marker, and session entry bindings.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

_ERROR_MESSAGE = "session recovery candidate is unavailable"
_WORKSPACE_ID_RE = re.compile(rb"[0-9a-f]{32}(?:\n)?\Z")
_NAMESPACE_RE = re.compile(r"[0-9a-f]{32}\Z")
_WORKSPACE_MARKER_MAX_BYTES = 4096
_MAX_PATH_COMPONENTS = 128

# Snapshots intentionally omit atime because a descriptor read may update it.
_FileIdentity = tuple[int, int, int, int, int, int, int, int, int]
_DirectoryIdentity = tuple[int, int, int, int, int]


class RecoveryLocatorError(RuntimeError):
    """A recovery candidate could not be established safely."""

    def __init__(self) -> None:
        super().__init__(_ERROR_MESSAGE)


@dataclass(frozen=True, slots=True)
class RecoveryCandidate:
    """Detached candidate; its paths require later checked-authority validation."""

    workspace_path: Path
    sessions_root: Path
    namespace_path: Path
    session_path: Path
    workspace_id: str
    session_key: str
    workspace_identity: _DirectoryIdentity
    sessions_root_identity: _DirectoryIdentity
    namespace_identity: _DirectoryIdentity
    workspace_marker_identity: _FileIdentity
    namespace_marker_identity: _FileIdentity
    session_file_identity: _FileIdentity

    @property
    def requires_authority_revalidation(self) -> bool:
        return True


def _fail() -> None:
    raise RecoveryLocatorError() from None


def _require_descriptor_support() -> None:
    required_flags = ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    if any(not isinstance(getattr(os, flag, None), int) for flag in required_flags):
        _fail()
    if os.open not in getattr(os, "supports_dir_fd", set()):
        _fail()
    if os.stat not in getattr(os, "supports_dir_fd", set()):
        _fail()
    if os.stat not in getattr(os, "supports_follow_symlinks", set()):
        _fail()
    if os.listdir not in getattr(os, "supports_fd", set()):
        _fail()
    if not callable(getattr(os, "geteuid", None)):
        _fail()


def _path_input(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        _fail()
    if ".." in value.parts or len(value.parts) > _MAX_PATH_COMPONENTS + 1:
        _fail()
    return value


def _open_flags(*, directory: bool) -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= os.O_DIRECTORY
    else:
        flags |= os.O_NONBLOCK
    return flags


def _directory_identity(value: os.stat_result) -> _DirectoryIdentity:
    if not stat.S_ISDIR(value.st_mode) or value.st_ino <= 0:
        _fail()
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
    )


def _require_controlled_directory(value: os.stat_result) -> _DirectoryIdentity:
    identity = _directory_identity(value)
    permissions = stat.S_IMODE(value.st_mode)
    if (
        value.st_uid != os.geteuid()
        or permissions & 0o700 != 0o700
        or permissions & (0o022 | 0o7000)
    ):
        _fail()
    return identity


def _require_controlled_file(value: os.stat_result) -> _FileIdentity:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or value.st_uid != os.geteuid()
        or not stat.S_IMODE(value.st_mode) & 0o400
        or stat.S_IMODE(value.st_mode) & (0o022 | 0o7000)
    ):
        _fail()
    return _file_identity(value)


def _file_identity(value: os.stat_result) -> _FileIdentity:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_uid,
        value.st_gid,
    )


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return _file_identity(left) == _file_identity(right)


def _check_directory_entry(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _require_controlled_directory(current) != _require_controlled_directory(expected):
        _fail()


def _check_file_entry(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    _require_controlled_file(current)
    _require_controlled_file(expected)
    if not _same_file_state(current, expected):
        _fail()


def _open_absolute_directory(path: Path, owned_fds: list[int]) -> tuple[int, os.stat_result]:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or len(path.parts) > _MAX_PATH_COMPONENTS + 1
    ):
        _fail()
    current_fd = os.open(path.anchor, _open_flags(directory=True))
    owned_fds.append(current_fd)
    current_info = os.fstat(current_fd)
    _directory_identity(current_info)

    for component in path.parts[1:]:
        if component in {"", ".", ".."}:
            _fail()
        child_fd = os.open(component, _open_flags(directory=True), dir_fd=current_fd)
        owned_fds.append(child_fd)
        child_info = os.fstat(child_fd)
        child_identity = _directory_identity(child_info)
        entry_info = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
        if _directory_identity(entry_info) != child_identity:
            _fail()
        current_fd = child_fd
        current_info = child_info

    return current_fd, current_info


def _open_child_directory(
    parent_fd: int,
    name: str,
    owned_fds: list[int],
) -> tuple[int, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    _require_controlled_directory(before)
    child_fd = os.open(name, _open_flags(directory=True), dir_fd=parent_fd)
    owned_fds.append(child_fd)
    opened = os.fstat(child_fd)
    if _require_controlled_directory(opened) != _require_controlled_directory(before):
        _fail()
    _check_directory_entry(parent_fd, name, opened)
    return child_fd, opened


def _read_bounded_marker(
    parent_fd: int,
    name: str,
    max_bytes: int,
    owned_fds: list[int],
) -> tuple[bytes, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    _require_controlled_file(before)
    if before.st_size > max_bytes:
        _fail()

    descriptor = os.open(name, _open_flags(directory=False), dir_fd=parent_fd)
    owned_fds.append(descriptor)
    try:
        opened = os.fstat(descriptor)
        _require_controlled_file(opened)
        if not _same_file_state(before, opened):
            _fail()

        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(payload) > max_bytes or len(payload) != opened.st_size:
            _fail()
        if not _same_file_state(opened, after):
            _fail()
        _check_file_entry(parent_fd, name, opened)
        return payload, opened
    finally:
        owned_fds.remove(descriptor)
        try:
            os.close(descriptor)
        except OSError:
            pass


def _decode_workspace_id(payload: bytes) -> str:
    if len(payload) > 33 or _WORKSPACE_ID_RE.fullmatch(payload) is None:
        _fail()
    return payload.rstrip(b"\n").decode("ascii")


def _valid_workspace_binding(payload: bytes) -> bool:
    if not payload or len(payload) > _WORKSPACE_MARKER_MAX_BYTES or not payload.endswith(b"\n"):
        return False
    try:
        path_text = payload[:-1].decode("utf-8")
    except UnicodeDecodeError:
        return False
    candidate = Path(path_text)
    return (
        candidate.is_absolute()
        and ".." not in candidate.parts
        and str(candidate) == path_text
    )


def _check_absolute_binding(
    path: Path,
    expected: os.stat_result,
    owned_fds: list[int],
) -> None:
    descriptor, current = _open_absolute_directory(path, owned_fds)
    del descriptor
    if _require_controlled_directory(current) != _require_controlled_directory(expected):
        _fail()


def _check_namespace_marker_observations(
    root_fd: int,
    observations: list[
        tuple[str, _DirectoryIdentity, bytes | None, _FileIdentity | None]
    ],
    owned_fds: list[int],
) -> None:
    for name, expected_directory, expected_payload, expected_marker in observations:
        namespace_fd, namespace_info = _open_child_directory(root_fd, name, owned_fds)
        try:
            if _require_controlled_directory(namespace_info) != expected_directory:
                _fail()
            try:
                payload, marker_info = _read_bounded_marker(
                    namespace_fd,
                    ".workspace",
                    _WORKSPACE_MARKER_MAX_BYTES,
                    owned_fds,
                )
            except FileNotFoundError:
                if expected_payload is not None or expected_marker is not None:
                    _fail()
            else:
                if (
                    expected_payload is None
                    or expected_marker is None
                    or payload != expected_payload
                    or _file_identity(marker_info) != expected_marker
                ):
                    _fail()
        finally:
            try:
                os.close(namespace_fd)
            except OSError:
                pass
            owned_fds.remove(namespace_fd)


def _close_owned(fds: list[int]) -> None:
    for descriptor in reversed(fds):
        try:
            os.close(descriptor)
        except OSError:
            pass


def locate_recovery_candidate(
    workspace: Path,
    external_sessions_root: Path,
    session_key: str,
) -> RecoveryCandidate:
    """Locate an existing canonical JSONL entry without reading or changing it.

    Workspace and namespace markers are read through no-follow descriptors.
    The JSONL entry itself is only statted.  Unsupported descriptor guarantees
    fail closed here and have no effect on ordinary session-store construction.

    A copied workspace moved to another canonical path fails its workspace
    marker binding. A byte-for-byte copy installed at the same canonical path
    cannot be authenticated by this locator; the later checked authority must
    revalidate the binding and content before using the candidate.
    """
    if not isinstance(session_key, str) or not session_key.strip():
        _fail()

    owned_fds: list[int] = []
    try:
        _require_descriptor_support()
        workspace_input = _path_input(workspace)
        sessions_root = _path_input(external_sessions_root)
        # Workspace paths may arrive through platform aliases such as /tmp;
        # bind markers to the strict, canonical workspace path used by storage.
        workspace_path = workspace_input.resolve(strict=True)
        workspace_fd, workspace_info = _open_absolute_directory(workspace_path, owned_fds)
        workspace_identity = _require_controlled_directory(workspace_info)
        root_fd, root_info = _open_absolute_directory(sessions_root, owned_fds)
        root_identity = _require_controlled_directory(root_info)

        state_fd, state_info = _open_child_directory(workspace_fd, ".nanobot", owned_fds)
        workspace_id_bytes, workspace_marker_info = _read_bounded_marker(
            state_fd, "workspace-id", 33, owned_fds
        )
        workspace_id = _decode_workspace_id(workspace_id_bytes)
        expected_binding = f"{workspace_path}\n".encode("utf-8")

        if len(expected_binding) > _WORKSPACE_MARKER_MAX_BYTES:
            _fail()
        root_before_scan = os.fstat(root_fd)
        namespace_names = os.listdir(root_fd)
        if workspace_id not in namespace_names:
            _fail()

        target_namespace_fd: int | None = None
        target_namespace_info: os.stat_result | None = None
        target_namespace_marker_info: os.stat_result | None = None
        matching_bindings = 0
        marker_observations: list[
            tuple[str, _DirectoryIdentity, bytes | None, _FileIdentity | None]
        ] = []

        for name in sorted(namespace_names):
            if _NAMESPACE_RE.fullmatch(name) is None:
                continue
            namespace_fd, namespace_info = _open_child_directory(root_fd, name, owned_fds)
            try:
                try:
                    marker_entry = os.stat(".workspace", dir_fd=namespace_fd, follow_symlinks=False)
                except FileNotFoundError:
                    if name == workspace_id:
                        _fail()
                    marker_observations.append(
                        (name, _require_controlled_directory(namespace_info), None, None)
                    )
                    continue
                if not stat.S_ISREG(marker_entry.st_mode):
                    _fail()
                marker_bytes, marker_info = _read_bounded_marker(
                    namespace_fd,
                    ".workspace",
                    _WORKSPACE_MARKER_MAX_BYTES,
                    owned_fds,
                )
                if not _valid_workspace_binding(marker_bytes):
                    _fail()
                marker_observations.append(
                    (
                        name,
                        _require_controlled_directory(namespace_info),
                        marker_bytes,
                        _file_identity(marker_info),
                    )
                )
                if marker_bytes == expected_binding:
                    matching_bindings += 1
                    if name != workspace_id:
                        _fail()
                if name == workspace_id:
                    if marker_bytes != expected_binding:
                        _fail()
                    target_namespace_fd = namespace_fd
                    target_namespace_info = namespace_info
                    target_namespace_marker_info = marker_info
                    namespace_fd = -1
            finally:
                if namespace_fd >= 0:
                    try:
                        os.close(namespace_fd)
                    except OSError:
                        pass
                    owned_fds.remove(namespace_fd)

        if (
            matching_bindings != 1
            or target_namespace_fd is None
            or target_namespace_info is None
            or target_namespace_marker_info is None
        ):
            _fail()

        # Resolve the storage encoding through the established static codec,
        # without constructing a store or consulting a lossy legacy filename.
        from nanobot.session.manager import JsonlSessionStore

        stem = JsonlSessionStore.storage_key(session_key)
        decoded = JsonlSessionStore.decode_storage_key(stem)
        if decoded != session_key or JsonlSessionStore.storage_key(decoded) != stem:
            _fail()
        session_name = f"{stem}.jsonl"
        session_info = os.stat(
            session_name,
            dir_fd=target_namespace_fd,
            follow_symlinks=False,
        )
        _require_controlled_file(session_info)
        namespace_identity = _require_controlled_directory(target_namespace_info)
        session_identity = _file_identity(session_info)

        # Recheck each path binding and observed leaf before detaching identities.
        _check_file_entry(state_fd, "workspace-id", workspace_marker_info)
        _check_directory_entry(workspace_fd, ".nanobot", state_info)
        _check_file_entry(target_namespace_fd, ".workspace", target_namespace_marker_info)
        _check_file_entry(target_namespace_fd, session_name, session_info)
        _check_directory_entry(root_fd, workspace_id, target_namespace_info)
        _check_namespace_marker_observations(root_fd, marker_observations, owned_fds)
        if _require_controlled_directory(os.fstat(root_fd)) != root_identity:
            _fail()
        if _file_identity(os.fstat(root_fd)) != _file_identity(root_before_scan):
            _fail()
        _check_absolute_binding(workspace_path, workspace_info, owned_fds)
        _check_absolute_binding(sessions_root, root_info, owned_fds)

        return RecoveryCandidate(
            workspace_path=workspace_path,
            sessions_root=sessions_root,
            namespace_path=sessions_root / workspace_id,
            session_path=sessions_root / workspace_id / session_name,
            workspace_id=workspace_id,
            session_key=session_key,
            workspace_identity=workspace_identity,
            sessions_root_identity=root_identity,
            namespace_identity=namespace_identity,
            workspace_marker_identity=_file_identity(workspace_marker_info),
            namespace_marker_identity=_file_identity(target_namespace_marker_info),
            session_file_identity=session_identity,
        )
    except RecoveryLocatorError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RuntimeError, NotImplementedError):
        raise RecoveryLocatorError() from None
    finally:
        _close_owned(owned_fds)
