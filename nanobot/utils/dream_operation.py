"""Private durable state and no-follow file operations for Dream recovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Generator, Literal, cast
from uuid import uuid4

_OPERATION_FILENAME = "nanobot-dream-operation.json"
_OPERATION_MAX_BYTES = 4 * 1024 * 1024
_OPERATION_MAX_FILES = 10_000
_FILE_MAX_BYTES = 64 * 1024 * 1024
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}\Z")
_OPERATION_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_FIXED_PATHS = frozenset(("SOUL.md", "USER.md", "memory/MEMORY.md", "memory/.dream_cursor"))
_MISSING = object()


class DreamOperationStorageError(RuntimeError):
    """Content-free error for an unsafe or incomplete Dream storage operation."""


@dataclass(frozen=True, slots=True)
class DreamFileState:
    """Digest, mode and size for a regular file; None represents absence."""

    sha256: str
    mode: int
    size: int

    def __post_init__(self) -> None:
        if (
            not _DIGEST_RE.fullmatch(self.sha256)
            or type(self.mode) is not int
            or not 0 <= self.mode <= 0o777
            or type(self.size) is not int
            or not 0 <= self.size <= _FILE_MAX_BYTES
        ):
            raise DreamOperationStorageError("Dream file evidence is invalid")


@dataclass(frozen=True, slots=True)
class DreamFileChange:
    """One admitted path and the states allowed before and after settlement."""

    path: str
    before: DreamFileState | None
    target: DreamFileState | None


@dataclass(frozen=True, slots=True)
class DreamOperationRecord:
    """Versioned operation proof. It intentionally contains no file contents."""

    operation_id: str
    kind: Literal["restore", "run"]
    phase: Literal["restoring", "running", "settling"]
    workspace_path: str
    workspace_device: int
    workspace_inode: int
    starting_head: str
    target_commit: str | None
    target_tree: str
    cursor_before: int
    cursor_after: int
    completion: Literal["completed", "incomplete"] | None
    commit_message_sha256: str | None
    changes: tuple[DreamFileChange, ...]
    settlement_sha: str | None = None


def new_operation_id() -> str:
    """Return an unpredictable, content-free operation identifier."""

    return uuid4().hex


def workspace_identity(workspace: Path) -> tuple[str, int, int]:
    """Resolve and identify the workspace directory without lending a handle."""

    try:
        root = workspace.resolve(strict=True)
        info = root.stat()
    except OSError as exc:
        raise DreamOperationStorageError("Dream workspace is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise DreamOperationStorageError("Dream workspace is not a directory")
    return str(root), info.st_dev, info.st_ino


@contextmanager
def dream_workspace_lock(workspace: Path) -> Generator[None, None, None]:
    """Take the existing durable workspace-memory lock for one short transition."""

    from nanobot.agent.memory_recovery_storage import (
        MemoryStorageError,
        open_workspace_memory_lock,
    )

    try:
        with open_workspace_memory_lock(workspace):
            yield
    except MemoryStorageError as exc:
        raise DreamOperationStorageError("Dream workspace ownership is unavailable") from exc


def read_operation(workspace: Path) -> DreamOperationRecord | None:
    """Read and strictly validate the single durable Dream operation record."""

    from nanobot.attachments._media import MediaRetentionError, object_json, read_bytes

    path = _operation_path(workspace)
    try:
        payload = read_bytes(path, limit=_OPERATION_MAX_BYTES)
    except FileNotFoundError:
        return None
    except (OSError, MediaRetentionError) as exc:
        raise DreamOperationStorageError("Dream operation evidence is unsafe") from exc
    try:
        raw = object_json(payload)
        return _parse_operation(raw, workspace)
    except (ValueError, TypeError, KeyError, DreamOperationStorageError) as exc:
        raise DreamOperationStorageError("Dream operation evidence is invalid") from exc


def write_operation(workspace: Path, record: DreamOperationRecord) -> None:
    """Durably publish one bounded versioned operation record."""

    from nanobot.attachments._media import MediaRetentionError, atomic_write_bytes

    _validate_operation(record, workspace)
    payload = json.dumps(
        _operation_to_json(record), ensure_ascii=False, sort_keys=True, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _OPERATION_MAX_BYTES:
        raise DreamOperationStorageError("Dream operation evidence is too large")
    try:
        path = _operation_path(workspace)
        atomic_write_bytes(path, payload)
    except (OSError, MediaRetentionError) as exc:
        raise DreamOperationStorageError("Dream operation evidence could not be published") from exc


def clear_operation(workspace: Path) -> None:
    """Durably remove the operation record after complete settlement proof."""

    from nanobot.attachments._media import (
        MediaRetentionError,
        generation,
        unlink_generation,
    )

    path = _operation_path(workspace)
    try:
        parent = _directory_fd(path.parent)
        try:
            try:
                info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return
        finally:
            os.close(parent)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise DreamOperationStorageError("Dream operation evidence is unsafe")
        unlink_generation(path, generation(info))
    except (OSError, MediaRetentionError) as exc:
        raise DreamOperationStorageError("Dream operation evidence could not be cleared") from exc


def read_admitted_file(workspace: Path, relative_path: str) -> tuple[bytes, DreamFileState] | None:
    """Read one regular admitted file without following any path component."""

    _validate_admitted_path(relative_path)
    try:
        parent, identities = _open_parent(workspace, relative_path, create=False)
    except FileNotFoundError:
        return None
    try:
        _verify_parent(workspace, relative_path, identities)
        return _read_file_at(parent, PurePosixPath(relative_path).name)
    except (OSError, ValueError) as exc:
        raise DreamOperationStorageError("Dream admitted path is unsafe") from exc
    finally:
        os.close(parent)


def scan_admitted_worktree(workspace: Path) -> dict[str, tuple[bytes, DreamFileState]]:
    """Snapshot the admitted fixed files and regular workspace skill descendants."""

    result: dict[str, tuple[bytes, DreamFileState]] = {}
    for path in sorted(_FIXED_PATHS):
        item = read_admitted_file(workspace, path)
        if item is not None:
            result[path] = item

    try:
        root_fd = _directory_fd(workspace.resolve(strict=True))
        try:
            try:
                skills_fd = os.open(
                    "skills", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd,
                )
            except FileNotFoundError:
                return result
            except OSError as exc:
                raise DreamOperationStorageError("Dream skills root is unsafe") from exc
            try:
                _walk_skills(workspace, skills_fd, "skills", result)
            finally:
                os.close(skills_fd)
        finally:
            os.close(root_fd)
    except (OSError, ValueError) as exc:
        raise DreamOperationStorageError("Dream workspace tree is unsafe") from exc
    if len(result) > _OPERATION_MAX_FILES:
        raise DreamOperationStorageError("Dream admitted snapshot is too large")
    return result


def atomic_publish_admitted_file(
    workspace: Path,
    relative_path: str,
    data: bytes,
    mode: int,
    *,
    allowed_current: tuple[DreamFileState | None, ...],
) -> DreamFileState:
    """Atomically publish exact bytes after revalidating bound parent descriptors."""

    _validate_admitted_path(relative_path)
    if len(data) > _FILE_MAX_BYTES or mode not in (0o644, 0o755):
        raise DreamOperationStorageError("Dream target file is unsupported")
    target = DreamFileState(hashlib.sha256(data).hexdigest(), mode, len(data))
    try:
        parent, identities = _open_parent(workspace, relative_path, create=True)
        name = PurePosixPath(relative_path).name
        temp = f".dream-{uuid4().hex}.tmp"
        try:
            current = _read_file_at(parent, name)
            current_state = current[1] if current is not None else None
            if current_state not in allowed_current:
                raise DreamOperationStorageError("Dream admitted path changed during restore")
            descriptor = os.open(
                temp,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            try:
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short Dream file publication")
                    view = view[written:]
                os.fchmod(descriptor, mode)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _verify_parent(workspace, relative_path, identities)
            current = _read_file_at(parent, name)
            current_state = current[1] if current is not None else None
            if current_state not in allowed_current:
                raise DreamOperationStorageError("Dream admitted path changed during restore")
            os.replace(temp, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            _verify_parent(workspace, relative_path, identities)
            published = _read_file_at(parent, name)
            if published is None or published[1] != target or published[0] != data:
                raise DreamOperationStorageError("Dream admitted file readback failed")
            return target
        finally:
            try:
                os.unlink(temp, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)
    except DreamOperationStorageError:
        raise
    except (OSError, ValueError) as exc:
        raise DreamOperationStorageError("Dream admitted file publication failed") from exc


def atomic_delete_admitted_file(
    workspace: Path,
    relative_path: str,
    *,
    allowed_current: tuple[DreamFileState | None, ...],
) -> None:
    """Delete one admitted regular file and fsync its bound parent directory."""

    _validate_admitted_path(relative_path)
    try:
        parent, identities = _open_parent(workspace, relative_path, create=False)
    except FileNotFoundError:
        return
    try:
        name = PurePosixPath(relative_path).name
        current = _read_file_at(parent, name)
        current_state = current[1] if current is not None else None
        if current_state is None:
            if None not in allowed_current:
                raise DreamOperationStorageError("Dream admitted path changed during restore")
            return
        if current_state not in allowed_current:
            raise DreamOperationStorageError("Dream admitted path changed during restore")
        _verify_parent(workspace, relative_path, identities)
        current = _read_file_at(parent, name)
        current_state = current[1] if current is not None else None
        if current_state not in allowed_current:
            raise DreamOperationStorageError("Dream admitted path changed during restore")
        os.unlink(name, dir_fd=parent)
        os.fsync(parent)
        _verify_parent(workspace, relative_path, identities)
    except DreamOperationStorageError:
        raise
    except (OSError, ValueError) as exc:
        raise DreamOperationStorageError("Dream admitted file deletion failed") from exc
    finally:
        os.close(parent)
    _remove_empty_skill_parents(workspace, relative_path)


def ensure_admitted_parent(workspace: Path, relative_path: str) -> None:
    """Create missing admitted parent directories without following aliases."""

    _validate_admitted_path(relative_path)
    parent, _identities = _open_parent(workspace, relative_path, create=True)
    os.close(parent)


def validate_admitted_parent(workspace: Path, relative_path: str) -> None:
    """Refuse a parent that is currently an alias or a non-directory obstruction."""

    _validate_admitted_path(relative_path)
    try:
        parent, _identities = _open_parent(workspace, relative_path, create=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DreamOperationStorageError("Dream admitted parent is unsafe") from exc
    os.close(parent)


def state_for_bytes(data: bytes, mode: int) -> DreamFileState:
    """Build bounded immutable evidence for one regular file payload."""

    if len(data) > _FILE_MAX_BYTES:
        raise DreamOperationStorageError("Dream admitted file exceeds its limit")
    return DreamFileState(hashlib.sha256(data).hexdigest(), mode, len(data))


def _operation_path(workspace: Path) -> Path:
    root = workspace.resolve(strict=True)
    return root / ".git" / _OPERATION_FILENAME


def _validate_admitted_path(path: str) -> None:
    if type(path) is not str or not path or "\\" in path or "\x00" in path:
        raise DreamOperationStorageError("Dream path is invalid")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != path
        or not (path in _FIXED_PATHS or path.startswith("skills/") and len(pure.parts) > 1)
    ):
        raise DreamOperationStorageError("Dream path is outside the admitted snapshot")


def _directory_fd(path: Path) -> int:
    from nanobot.attachments._media import directory_fd

    return directory_fd(path)


def _open_parent(
    workspace: Path,
    relative_path: str,
    *,
    create: bool,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    _validate_admitted_path(relative_path)
    root_path = workspace.resolve(strict=True)
    root_fd = _directory_fd(root_path)
    root_info = os.fstat(root_fd)
    identities: list[tuple[int, int]] = [(root_info.st_dev, root_info.st_ino)]
    descriptor = root_fd
    try:
        for component in PurePosixPath(relative_path).parts[:-1]:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            info = os.fstat(child)
            identities.append((info.st_dev, info.st_ino))
            if descriptor != root_fd:
                os.close(descriptor)
            descriptor = child
        if descriptor == root_fd:
            return os.dup(root_fd), tuple(identities)
        return descriptor, tuple(identities)
    except BaseException:
        if descriptor != root_fd:
            os.close(descriptor)
        raise
    finally:
        os.close(root_fd)


def _verify_parent(
    workspace: Path,
    relative_path: str,
    identities: tuple[tuple[int, int], ...],
) -> None:
    parent, current = _open_parent(workspace, relative_path, create=False)
    try:
        if current != identities:
            raise DreamOperationStorageError("Dream admitted parent changed during restore")
    finally:
        os.close(parent)


def _read_file_at(parent: int, name: str) -> tuple[bytes, DreamFileState] | None:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
    except FileNotFoundError:
        return None
    before = os.fstat(descriptor)
    try:
        if not stat.S_ISREG(before.st_mode) or before.st_size > _FILE_MAX_BYTES:
            raise DreamOperationStorageError("Dream admitted entry is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, _FILE_MAX_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _FILE_MAX_BYTES:
                raise DreamOperationStorageError("Dream admitted file exceeds its limit")
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if _generation(before) != _generation(after) or _generation(before) != _generation(current):
            raise DreamOperationStorageError("Dream admitted file changed while reading")
        data = b"".join(chunks)
        state = DreamFileState(hashlib.sha256(data).hexdigest(), stat.S_IMODE(before.st_mode), len(data))
        return data, state
    finally:
        os.close(descriptor)


def _generation(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
    )


def _walk_skills(
    workspace: Path,
    directory: int,
    prefix: str,
    result: dict[str, tuple[bytes, DreamFileState]],
) -> None:
    with os.scandir(directory) as entries:
        for entry in entries:
            path = f"{prefix}/{entry.name}"
            _validate_admitted_path(path)
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                try:
                    _walk_skills(workspace, child, path, result)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise DreamOperationStorageError("Dream skills contain an unsupported entry")
            item = read_admitted_file(workspace, path)
            if item is None:
                raise DreamOperationStorageError("Dream skill changed during snapshot")
            result[path] = item
            if len(result) > _OPERATION_MAX_FILES:
                raise DreamOperationStorageError("Dream admitted snapshot is too large")


def _remove_empty_skill_parents(workspace: Path, relative_path: str) -> None:
    parts = PurePosixPath(relative_path).parts
    if not parts or parts[0] != "skills":
        return
    for index in range(len(parts) - 1, 1, -1):
        child_name = parts[index - 1]
        parent_path = "/".join(parts[:index - 1])
        try:
            parent_fd = _directory_fd(workspace.resolve(strict=True) / parent_path)
        except OSError:
            return
        try:
            child_fd = os.open(
                child_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                if os.listdir(child_fd):
                    return
            finally:
                os.close(child_fd)
            os.rmdir(child_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileNotFoundError:
            continue
        except OSError as exc:
            if exc.errno in (39, 66):  # ENOTEMPTY across supported POSIX hosts.
                return
            raise DreamOperationStorageError("Dream skill parent cleanup failed") from exc
        finally:
            os.close(parent_fd)


def _state_to_json(state: DreamFileState | None) -> object:
    if state is None:
        return None
    return {"sha256": state.sha256, "mode": state.mode, "size": state.size}


def _operation_to_json(record: DreamOperationRecord) -> dict[str, object]:
    return {
        "version": 1,
        "operation_id": record.operation_id,
        "kind": record.kind,
        "phase": record.phase,
        "workspace_path": record.workspace_path,
        "workspace_device": record.workspace_device,
        "workspace_inode": record.workspace_inode,
        "starting_head": record.starting_head,
        "target_commit": record.target_commit,
        "target_tree": record.target_tree,
        "cursor_before": record.cursor_before,
        "cursor_after": record.cursor_after,
        "completion": record.completion,
        "commit_message_sha256": record.commit_message_sha256,
        "changes": [
            {"path": change.path, "before": _state_to_json(change.before), "target": _state_to_json(change.target)}
            for change in record.changes
        ],
        "settlement_sha": record.settlement_sha,
    }


def _parse_state(value: object) -> DreamFileState | None:
    if value is None:
        return None
    fields = _object_fields(value, {"sha256", "mode", "size"}, "invalid Dream file state")
    digest = fields["sha256"]
    mode = fields["mode"]
    size = fields["size"]
    if type(digest) is not str or type(mode) is not int or type(size) is not int:
        raise DreamOperationStorageError("invalid Dream file state")
    return DreamFileState(digest, mode, size)


def _object_fields(value: object, expected: set[str], error: str) -> dict[str, object]:
    """Validate an untrusted JSON object before exposing typed fields."""

    if type(value) is not dict:
        raise DreamOperationStorageError(error)
    # The exact container check plus per-key validation below establishes a
    # string-keyed mapping; values remain object until each owning parser checks them.
    raw = cast(dict[object, object], value)
    fields: dict[str, object] = {}
    for key in raw:
        if type(key) is not str:
            raise DreamOperationStorageError(error)
        fields[key] = raw[key]
    if set(fields) != expected:
        raise DreamOperationStorageError(error)
    return fields


def _object_list(value: object, *, maximum: int) -> list[object]:
    """Validate a JSON array and return values for per-item parsing."""

    if type(value) is not list:
        raise DreamOperationStorageError("invalid Dream operation paths")
    items = cast(list[object], value)
    if len(items) > maximum:
        raise DreamOperationStorageError("invalid Dream operation paths")
    return items


def _parse_operation(raw: dict[str, object], workspace: Path) -> DreamOperationRecord:
    expected = {
        "version", "operation_id", "kind", "phase", "workspace_path", "workspace_device",
        "workspace_inode", "starting_head", "target_commit", "target_tree", "cursor_before",
        "cursor_after", "completion", "commit_message_sha256", "changes", "settlement_sha",
    }
    if set(raw) != expected or type(raw.get("version")) is not int or raw["version"] != 1:
        raise DreamOperationStorageError("invalid Dream operation record")

    operation_id = _required_string(raw, "operation_id")
    kind = _required_string(raw, "kind")
    phase = _required_string(raw, "phase")
    workspace_path = _required_string(raw, "workspace_path")
    starting_head = _required_string(raw, "starting_head")
    target_tree = _required_string(raw, "target_tree")
    device = _required_integer(raw, "workspace_device")
    inode = _required_integer(raw, "workspace_inode")
    cursor_before = _required_integer(raw, "cursor_before")
    cursor_after = _required_integer(raw, "cursor_after")
    target_commit_value = raw.get("target_commit")
    target_commit = target_commit_value if type(target_commit_value) is str else None
    settlement_value = raw.get("settlement_sha")
    settlement_sha = settlement_value if type(settlement_value) is str else None
    completion_value = raw.get("completion")
    if completion_value not in (None, "completed", "incomplete"):
        raise DreamOperationStorageError("invalid Dream operation classification")
    completion = completion_value if completion_value in ("completed", "incomplete") else None
    message_value = raw.get("commit_message_sha256")
    commit_message_sha256 = message_value if type(message_value) is str else None
    changes_raw = _object_list(raw.get("changes"), maximum=_OPERATION_MAX_FILES)
    changes: list[DreamFileChange] = []
    for item in changes_raw:
        fields = _object_fields(item, {"path", "before", "target"}, "invalid Dream operation path")
        path = fields["path"]
        if type(path) is not str:
            raise DreamOperationStorageError("invalid Dream operation path")
        _validate_admitted_path(path)
        changes.append(DreamFileChange(path, _parse_state(fields["before"]), _parse_state(fields["target"])))
    if len({change.path for change in changes}) != len(changes):
        raise DreamOperationStorageError("duplicate Dream operation path")

    if kind not in ("restore", "run") or phase not in ("restoring", "running", "settling"):
        raise DreamOperationStorageError("invalid Dream operation phase")
    if type(target_commit_value) not in (str, type(None)) or type(settlement_value) not in (str, type(None)):
        raise DreamOperationStorageError("invalid Dream operation SHA")
    if type(completion_value) not in (str, type(None)) or type(message_value) not in (str, type(None)):
        raise DreamOperationStorageError("invalid Dream operation classification")
    record = DreamOperationRecord(
        operation_id=operation_id,
        kind="restore" if kind == "restore" else "run",
        phase="restoring" if phase == "restoring" else "running" if phase == "running" else "settling",
        workspace_path=workspace_path,
        workspace_device=device,
        workspace_inode=inode,
        starting_head=starting_head,
        target_commit=target_commit,
        target_tree=target_tree,
        cursor_before=cursor_before,
        cursor_after=cursor_after,
        completion="completed" if completion == "completed" else "incomplete" if completion == "incomplete" else None,
        commit_message_sha256=commit_message_sha256,
        changes=tuple(changes),
        settlement_sha=settlement_sha,
    )
    _validate_operation(record, workspace)
    return record


def _validate_operation(record: DreamOperationRecord, workspace: Path) -> None:
    path, device, inode = workspace_identity(workspace)
    if (
        not _OPERATION_ID_RE.fullmatch(record.operation_id)
        or record.workspace_path != path
        or record.workspace_device != device
        or record.workspace_inode != inode
        or not _OBJECT_ID_RE.fullmatch(record.starting_head)
        or not _OBJECT_ID_RE.fullmatch(record.target_tree)
        or record.target_commit is not None and not _OBJECT_ID_RE.fullmatch(record.target_commit)
        or record.settlement_sha is not None and not _OBJECT_ID_RE.fullmatch(record.settlement_sha)
        or record.commit_message_sha256 is not None
        and not _DIGEST_RE.fullmatch(record.commit_message_sha256)
        or type(record.cursor_before) is not int or record.cursor_before < 0
        or type(record.cursor_after) is not int or record.cursor_after < 0
        or record.kind == "restore" and (
            record.phase != "restoring" or record.target_commit is None
            or record.completion is not None or record.commit_message_sha256 is not None
        )
        or record.kind == "run" and (
            record.target_commit is not None
            or record.phase == "running" and (
                record.completion is not None or record.commit_message_sha256 is not None
            )
            or record.phase == "settling" and (
                record.completion is None or record.commit_message_sha256 is None
            )
        )
        or record.phase == "running" and record.kind != "run"
        or len(record.changes) > _OPERATION_MAX_FILES
        or tuple(change.path for change in record.changes) != tuple(sorted(change.path for change in record.changes))
    ):
        raise DreamOperationStorageError("Dream operation proof does not match its workspace")
    for change in record.changes:
        _validate_admitted_path(change.path)
        if change.before is not None:
            DreamFileState(change.before.sha256, change.before.mode, change.before.size)
        if change.target is not None:
            DreamFileState(change.target.sha256, change.target.mode, change.target.size)


def _required_string(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if type(result) is not str:
        raise DreamOperationStorageError("invalid Dream operation string")
    return result


def _required_integer(value: dict[str, object], key: str) -> int:
    result = value.get(key)
    if type(result) is not int:
        raise DreamOperationStorageError("invalid Dream operation integer")
    return result
