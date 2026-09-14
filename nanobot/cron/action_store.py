"""Private, durable action journal for the staged cron migration.

This module deliberately knows nothing about cron jobs.  It durably accepts
opaque JSON action requests, returns idempotency receipts, and lets a later
adapter checkpoint an equally opaque candidate.  In particular it must not
interpret ``add``, ``update``, or ``del`` requests from the old service.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Mapping

try:  # Keep this module importable on Windows; construction fails closed.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - covered through the platform gate
    _fcntl = None


_SCHEMA_VERSION = 2
_MAX_NAMESPACE_BYTES = 512
_MAX_ACTION_ID_BYTES = 512
_MAX_REQUEST_BYTES = 262_144
_MAX_RECORD_BYTES = 1_048_576
_MAX_JOURNAL_BYTES = 8_388_608
_MAX_RECEIPTS = 4_096
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_ACTION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,511}")
_TEMP_ENTRY_RE = re.compile(r"\.(?:snapshot|journal)\.[0-9a-f]{32}\.tmp")


class CronActionStoreError(RuntimeError):
    """Base error for the detached durable action journal."""


class ActionStorePersistenceError(CronActionStoreError):
    """A requested durability boundary could not be established."""


class ActionStoreCorruptError(CronActionStoreError):
    """An owned on-disk object is malformed, unsafe, or ambiguous."""


class ActionStoreConflictError(CronActionStoreError):
    """An action identity or exact checkpoint retry has conflicting content."""


class ActionStoreStaleGenerationError(CronActionStoreError):
    """A checkpoint was based on a state that is no longer current."""


class ActionStoreConversionRequiredError(CronActionStoreError):
    """A pre-v2 action journal was found and must be converted offline."""


@dataclass(frozen=True)
class ActionStoreGeneration:
    """The exact compare-and-swap token returned by :meth:`replay`."""

    journal_generation: int
    snapshot_generation: int
    watermark: int
    sequence: int


@dataclass(frozen=True)
class ActionReceipt:
    """Durable idempotency evidence for one canonical action request."""

    store_namespace: str
    action_id: str
    request_digest: str
    sequence: int
    journal_generation: int


@dataclass(frozen=True)
class ActionRequest:
    """A detached journal action which a later adapter may interpret."""

    receipt: ActionReceipt
    request: dict[str, Any]


@dataclass(frozen=True)
class ActionStoreReplay:
    """A detached generic checkpoint candidate plus unapplied actions.

    ``candidate`` is the caller-provided mapping in the last checkpoint, or
    ``None`` before the first checkpoint.  ``pending_actions`` is ordered and
    contains every action after the candidate watermark.  No cron-service
    operation is applied in this module.
    """

    store_namespace: str
    generation: ActionStoreGeneration
    candidate: dict[str, Any] | None
    pending_actions: tuple[ActionRequest, ...]
    receipts: tuple[ActionReceipt, ...]


@dataclass(frozen=True)
class _Snapshot:
    journal_generation: int
    snapshot_generation: int
    watermark: int
    prior_generation: ActionStoreGeneration
    candidate: dict[str, Any]
    receipts: tuple[ActionReceipt, ...]


@dataclass(frozen=True)
class _Journal:
    generation: int
    base_sequence: int
    actions: tuple[ActionRequest, ...]


@dataclass(frozen=True)
class _State:
    replay: ActionStoreReplay
    journal: _Journal
    snapshot: _Snapshot | None
    cleanup_required: bool


@dataclass(frozen=True)
class _HeldStore:
    store: CronActionStore
    root_fd: int
    lock_fd: int
    root_identity: tuple[int, int]
    lock_identity: tuple[int, int]

    def assert_bound(self) -> None:
        self.store._assert_binding(self)


def _inode(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _private(info: os.stat_result, *, directory: bool, links: int = 1) -> None:
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or (not directory and info.st_nlink != links)
    ):
        raise ActionStoreCorruptError("action store object is not private or has an unexpected type")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(data: bytes, *, label: str) -> object:
    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=_no_duplicate_object, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ActionStoreCorruptError(f"{label} is not strict JSON") from exc


def _canonical_json(value: object, *, error: type[CronActionStoreError] = ActionStoreConflictError) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError, RecursionError) as exc:
        raise error("action store values must be finite JSON data") from exc


def _copy_json_mapping(
    value: Mapping[str, Any], *, label: str, error: type[CronActionStoreError] = ActionStoreConflictError
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{label} must be a JSON object")
    encoded = _canonical_json(dict(value), error=error).encode("utf-8")
    if len(encoded) > _MAX_REQUEST_BYTES:
        raise error(f"{label} exceeds the { _MAX_REQUEST_BYTES } byte limit")
    parsed = _strict_json_loads(encoded, label=label)
    if not isinstance(parsed, dict):  # defensive; mappings encode to objects
        raise error(f"{label} must be a JSON object")
    return parsed


def _digest_request(request: Mapping[str, Any]) -> str:
    canonical = _canonical_json(dict(request)).encode("utf-8")
    return hashlib.sha256(b"nanobot-cron-action-v2\0" + canonical).hexdigest()


class CronActionStore:
    """POSIX-qualified version-2 journal with opaque checkpoint candidates.

    ``root`` must be a dedicated, absolute private directory.  The store does
    not read or repair ``jobs.json``; a corrupt legacy jobs snapshot therefore
    cannot prevent a new action from being durably queued.  A legacy
    ``action.jsonl`` or non-v2 ``journal.jsonl`` is rejected with an explicit
    conversion-required error rather than being interpreted.
    """

    def __init__(self, root: Path, *, namespace: str):
        self._require_qualified_platform()
        root = Path(root)
        if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
            raise ValueError("action store root must be an absolute non-root path without '..'")
        self.root = root
        self.namespace = self._validate_namespace(namespace)
        self._root_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None
        with self._root_fd(create=True) as root_fd:
            self._root_identity = _inode(os.fstat(root_fd))
        with self._held_lock() as held:
            self._detect_legacy(held)
            try:
                self._read_journal(held)
            except FileNotFoundError:
                self._create_journal(held, generation=1, base_sequence=0)
            self._verify_journal_durable(held)
            if self._read_snapshot(held) is not None:
                self._verify_snapshot_durable(held)
            self._read_state(held)

    @staticmethod
    def _require_qualified_platform() -> None:
        required_calls = {os.open, os.mkdir, os.stat, os.unlink, os.rename}
        if (
            os.name != "posix"
            or _fcntl is None
            or not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
            or not required_calls <= os.supports_dir_fd
        ):
            raise ActionStorePersistenceError(
                "private cron action store requires qualified POSIX descriptor and flock support"
            )

    @staticmethod
    def _validate_namespace(namespace: str) -> str:
        if not isinstance(namespace, str) or not namespace or len(namespace.encode("utf-8")) > _MAX_NAMESPACE_BYTES:
            raise ValueError(f"namespace must be a non-empty UTF-8 string no longer than {_MAX_NAMESPACE_BYTES} bytes")
        return namespace

    @staticmethod
    def _validate_action_id(action_id: str) -> str:
        if not isinstance(action_id, str) or not _ACTION_ID_RE.fullmatch(action_id):
            raise ValueError("action_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,511}")
        return action_id

    @staticmethod
    def _validate_generation(generation: ActionStoreGeneration) -> ActionStoreGeneration:
        if (
            type(generation.journal_generation) is not int
            or type(generation.snapshot_generation) is not int
            or type(generation.watermark) is not int
            or type(generation.sequence) is not int
            or generation.journal_generation < 1
            or generation.snapshot_generation < 0
            or generation.watermark < 0
            or generation.sequence < generation.watermark
        ):
            raise ValueError("expected_generation contains invalid integer fields")
        return generation

    @staticmethod
    def _dir_flags() -> int:
        return os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_NONBLOCK

    @contextmanager
    def _root_fd(self, *, create: bool = False) -> Generator[int, None, None]:
        fd: int | None = None
        try:
            fd = os.open(self.root.anchor, self._dir_flags())
            for part in self.root.parts[1:]:
                child: int | None = None
                created = False
                try:
                    try:
                        child = os.open(part, self._dir_flags(), dir_fd=fd)
                    except FileNotFoundError:
                        if not create:
                            raise ActionStorePersistenceError("private action store root is unavailable") from None
                        try:
                            os.mkdir(part, 0o700, dir_fd=fd)
                        except FileExistsError:
                            pass
                        else:
                            created = True
                        child = os.open(part, self._dir_flags(), dir_fd=fd)
                    if created:
                        os.fchmod(child, 0o700)
                    child_info = os.fstat(child)
                    entry = os.stat(part, dir_fd=fd, follow_symlinks=False)
                    if not stat.S_ISDIR(child_info.st_mode) or _inode(child_info) != _inode(entry):
                        raise ActionStoreCorruptError("action store root component binding changed")
                    if create:
                        os.fsync(child)
                        os.fsync(fd)
                        if _inode(os.stat(part, dir_fd=fd, follow_symlinks=False)) != _inode(child_info):
                            raise ActionStoreCorruptError("action store root component binding changed")
                    os.close(fd)
                    fd, child = child, None
                finally:
                    if child is not None:
                        os.close(child)
            info = os.fstat(fd)
            _private(info, directory=True)
            if self._root_identity is not None and _inode(info) != self._root_identity:
                raise ActionStoreCorruptError("action store root binding changed")
            yield fd
        except CronActionStoreError:
            raise
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ActionStoreCorruptError("action store root must be a private directory, never a link") from exc
            raise ActionStorePersistenceError("private action store root is unavailable") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _open_lock(self, root_fd: int) -> int:
        flags = os.O_RDWR | _O_NOFOLLOW | _O_NONBLOCK
        lock_fd: int | None = None
        created = False
        try:
            for _ in range(8):
                try:
                    lock_fd = os.open(".lock", flags, dir_fd=root_fd)
                    break
                except FileNotFoundError:
                    try:
                        lock_fd = os.open(".lock", flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=root_fd)
                    except FileExistsError:
                        continue
                    created = True
                    break
            if lock_fd is None:
                raise ActionStorePersistenceError("action store lock could not be established")
            if created:
                os.fchmod(lock_fd, 0o600)
                os.fsync(lock_fd)
                os.fsync(root_fd)
            _private(os.fstat(lock_fd), directory=False)
            current = _inode(os.fstat(lock_fd))
            if self._lock_identity is not None and current != self._lock_identity:
                raise ActionStoreCorruptError("action store lock binding changed")
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX)
            return lock_fd
        except CronActionStoreError:
            if lock_fd is not None:
                os.close(lock_fd)
            raise
        except OSError as exc:
            if lock_fd is not None:
                os.close(lock_fd)
            if exc.errno == errno.ELOOP:
                raise ActionStoreCorruptError("action store lock must not be a symlink") from exc
            raise ActionStorePersistenceError("could not acquire action store lock") from exc

    def _assert_binding(self, held: _HeldStore) -> None:
        root_info = os.fstat(held.root_fd)
        _private(root_info, directory=True)
        if _inode(root_info) != held.root_identity or _inode(root_info) != self._root_identity:
            raise ActionStoreCorruptError("action store root binding changed")
        # A held descriptor alone cannot detect a rename followed by a new
        # private directory at the configured path. Re-walk the absolute path
        # before every acknowledgement/durability boundary and require it to
        # resolve to the descriptor that carries this operation's lock.
        with self._root_fd() as current_root_fd:
            current_info = os.fstat(current_root_fd)
            if _inode(current_info) != held.root_identity:
                raise ActionStoreCorruptError("configured action store root was replaced")
        lock_info = os.fstat(held.lock_fd)
        _private(lock_info, directory=False)
        entry = os.stat(".lock", dir_fd=held.root_fd, follow_symlinks=False)
        _private(entry, directory=False)
        if (
            _inode(lock_info) != held.lock_identity
            or _inode(entry) != held.lock_identity
            or self._lock_identity != held.lock_identity
        ):
            raise ActionStoreCorruptError("action store lock binding changed")

    def _assert_file_binding(self, held: _HeldStore, name: str, fd: int) -> os.stat_result:
        """Bind an owned descriptor to its current entry across durability boundaries."""
        held.assert_bound()
        info = os.fstat(fd)
        _private(info, directory=False)
        entry = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
        _private(entry, directory=False)
        if _inode(info) != _inode(entry):
            raise ActionStoreCorruptError(f"{name} binding changed during durable operation")
        return info

    @contextmanager
    def _held_lock(self) -> Generator[_HeldStore, None, None]:
        with self._root_fd() as root_fd:
            lock_fd = self._open_lock(root_fd)
            try:
                lock_identity = _inode(os.fstat(lock_fd))
                if self._lock_identity is None:
                    self._lock_identity = lock_identity
                held = _HeldStore(self, root_fd, lock_fd, _inode(os.fstat(root_fd)), lock_identity)
                held.assert_bound()
                yield held
                held.assert_bound()
            finally:
                try:
                    _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    def _detect_legacy(self, held: _HeldStore) -> None:
        try:
            legacy = os.stat("action.jsonl", dir_fd=held.root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ActionStorePersistenceError("could not inspect legacy action journal") from exc
        if stat.S_ISLNK(legacy.st_mode):
            raise ActionStoreCorruptError("legacy action journal must not be a symlink")
        raise ActionStoreConversionRequiredError(
            "legacy action.jsonl detected; offline conversion is required before v2 action-store use"
        )

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("short action store write")
            offset += written

    def _read_private_bytes(self, held: _HeldStore, name: str, *, maximum: int) -> bytes:
        held.assert_bound()
        fd: int | None = None
        try:
            fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=held.root_fd)
            before = self._assert_file_binding(held, name, fd)
            if before.st_size < 0 or before.st_size > maximum:
                raise ActionStoreCorruptError(f"{name} has an unsafe binding or size")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    raise ActionStoreCorruptError(f"{name} was truncated during read")
                chunks.append(chunk)
                remaining -= len(chunk)
            after = self._assert_file_binding(held, name, fd)
            if _inode(before) != _inode(after) or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ):
                raise ActionStoreCorruptError(f"{name} changed while being read")
            return b"".join(chunks)
        except FileNotFoundError:
            raise
        except CronActionStoreError:
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ActionStoreCorruptError(f"{name} must not be a symlink") from exc
            raise ActionStorePersistenceError(f"could not read {name}") from exc
        finally:
            if fd is not None:
                os.close(fd)

    @staticmethod
    def _as_int(value: object, *, label: str, minimum: int = 0) -> int:
        if type(value) is not int or value < minimum:
            raise ActionStoreCorruptError(f"{label} must be an integer of at least {minimum}")
        return value

    def _header_wire(self, *, generation: int, base_sequence: int) -> bytes:
        return (
            _canonical_json(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "kind": "journal_header",
                    "store_namespace": self.namespace,
                    "journal_generation": generation,
                    "base_sequence": base_sequence,
                },
                error=ActionStoreCorruptError,
            ).encode("utf-8")
            + b"\n"
        )

    def _validate_header(self, value: object) -> tuple[int, int]:
        fields = {"schema_version", "kind", "store_namespace", "journal_generation", "base_sequence"}
        if not isinstance(value, dict):
            raise ActionStoreConversionRequiredError("journal is not a version-2 action-store journal")
        version = value.get("schema_version")
        if type(version) is not int:
            raise ActionStoreCorruptError("journal schema version must be an exact integer")
        if version != _SCHEMA_VERSION or value.get("kind") != "journal_header":
            raise ActionStoreConversionRequiredError("journal conversion to version 2 is required")
        if set(value) != fields:
            raise ActionStoreCorruptError("journal header has an invalid schema")
        if value["store_namespace"] != self.namespace:
            raise ActionStoreCorruptError("journal belongs to a different action-store namespace")
        return (
            self._as_int(value["journal_generation"], label="journal generation", minimum=1),
            self._as_int(value["base_sequence"], label="journal base sequence"),
        )

    def _validate_action(self, value: object, *, generation: int, sequence: int) -> ActionRequest:
        fields = {
            "schema_version", "kind", "store_namespace", "journal_generation", "sequence", "action_id",
            "request_digest", "request",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ActionStoreCorruptError("journal action has an invalid schema")
        version = self._as_int(value["schema_version"], label="journal action schema version", minimum=1)
        if version != _SCHEMA_VERSION or value["kind"] != "action":
            raise ActionStoreCorruptError("journal action has an invalid version or kind")
        action_generation = self._as_int(value["journal_generation"], label="action journal generation", minimum=1)
        if value["store_namespace"] != self.namespace or action_generation != generation:
            raise ActionStoreCorruptError("journal action has a foreign namespace or generation")
        if self._as_int(value["sequence"], label="action sequence", minimum=1) != sequence:
            raise ActionStoreCorruptError("journal action sequence is not contiguous")
        action_id = value["action_id"]
        try:
            self._validate_action_id(action_id)
        except ValueError as exc:
            raise ActionStoreCorruptError("journal action has an invalid action id") from exc
        digest = value["request_digest"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ActionStoreCorruptError("journal action has an invalid request digest")
        try:
            request = _copy_json_mapping(value["request"], label="journal action request", error=ActionStoreCorruptError)
        except TypeError as exc:  # Mapping runtime protocol can reject unusual objects
            raise ActionStoreCorruptError("journal action request is invalid") from exc
        if _digest_request(request) != digest:
            raise ActionStoreCorruptError("journal action request digest does not match")
        return ActionRequest(ActionReceipt(self.namespace, action_id, digest, sequence, generation), request)

    def _read_journal(self, held: _HeldStore) -> _Journal:
        data = self._read_private_bytes(held, "journal.jsonl", maximum=_MAX_JOURNAL_BYTES)
        if not data or not data.endswith(b"\n"):
            raise ActionStoreCorruptError("journal must be non-empty and newline terminated")
        lines = data.splitlines()
        if not lines or any(not line for line in lines):
            raise ActionStoreCorruptError("journal contains an empty record")
        header = _strict_json_loads(lines[0], label="journal header")
        generation, base = self._validate_header(header)
        actions: list[ActionRequest] = []
        seen: set[str] = set()
        for offset, line in enumerate(lines[1:], start=1):
            if len(line) > _MAX_RECORD_BYTES:
                raise ActionStoreCorruptError("journal action exceeds the record size limit")
            action = self._validate_action(line and _strict_json_loads(line, label="journal action"), generation=generation,
                                           sequence=base + offset)
            if action.receipt.action_id in seen:
                raise ActionStoreCorruptError("journal has a duplicate action id")
            seen.add(action.receipt.action_id)
            actions.append(action)
        return _Journal(generation, base, tuple(actions))

    def _create_journal(self, held: _HeldStore, *, generation: int, base_sequence: int) -> None:
        self._write_new_file(held, "journal.jsonl", self._header_wire(generation=generation, base_sequence=base_sequence))

    def _write_new_file(self, held: _HeldStore, name: str, data: bytes) -> None:
        held.assert_bound()
        fd: int | None = None
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_NONBLOCK, 0o600, dir_fd=held.root_fd)
            os.fchmod(fd, 0o600)
            self._assert_file_binding(held, name, fd)
            self._write_all(fd, data)
            self._assert_file_binding(held, name, fd)
            os.fsync(fd)
            self._assert_file_binding(held, name, fd)
            os.fsync(held.root_fd)
            self._assert_file_binding(held, name, fd)
        except CronActionStoreError:
            raise
        except OSError as exc:
            raise ActionStorePersistenceError(f"could not durably create {name}") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _append_journal(self, held: _HeldStore, action: ActionRequest) -> None:
        wire = _canonical_json(
            {
                "schema_version": _SCHEMA_VERSION,
                "kind": "action",
                "store_namespace": self.namespace,
                "journal_generation": action.receipt.journal_generation,
                "sequence": action.receipt.sequence,
                "action_id": action.receipt.action_id,
                "request_digest": action.receipt.request_digest,
                "request": _copy_json_mapping(action.request, label="action request"),
            }
        ).encode("utf-8") + b"\n"
        if len(wire) > _MAX_RECORD_BYTES:
            raise ActionStoreConflictError("encoded action exceeds the private record size limit")
        held.assert_bound()
        fd: int | None = None
        try:
            fd = os.open("journal.jsonl", os.O_WRONLY | os.O_APPEND | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=held.root_fd)
            info = self._assert_file_binding(held, "journal.jsonl", fd)
            if info.st_size + len(wire) > _MAX_JOURNAL_BYTES:
                raise ActionStoreConflictError("encoded journal exceeds the private journal size limit")
            self._write_all(fd, wire)
            self._assert_file_binding(held, "journal.jsonl", fd)
            os.fsync(fd)
            self._assert_file_binding(held, "journal.jsonl", fd)
            os.fsync(held.root_fd)
            self._assert_file_binding(held, "journal.jsonl", fd)
        except CronActionStoreError:
            raise
        except OSError as exc:
            raise ActionStorePersistenceError("could not durably append action journal") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _verify_journal_durable(self, held: _HeldStore) -> None:
        fd: int | None = None
        try:
            fd = os.open("journal.jsonl", os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=held.root_fd)
            self._assert_file_binding(held, "journal.jsonl", fd)
            os.fsync(fd)
            self._assert_file_binding(held, "journal.jsonl", fd)
            os.fsync(held.root_fd)
            self._assert_file_binding(held, "journal.jsonl", fd)
        except CronActionStoreError:
            raise
        except OSError as exc:
            raise ActionStorePersistenceError("could not verify journal durability") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _read_snapshot(self, held: _HeldStore) -> _Snapshot | None:
        try:
            data = self._read_private_bytes(held, "snapshot.json", maximum=_MAX_RECORD_BYTES)
        except FileNotFoundError:
            return None
        value = _strict_json_loads(data, label="action snapshot")
        fields = {
            "schema_version", "kind", "store_namespace", "journal_generation", "snapshot_generation",
            "base_sequence", "watermark", "prior_generation", "candidate", "receipts",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ActionStoreCorruptError("action snapshot has an invalid schema")
        version = self._as_int(value["schema_version"], label="snapshot schema version", minimum=1)
        if version != _SCHEMA_VERSION or value["kind"] != "snapshot":
            raise ActionStoreConversionRequiredError("snapshot conversion to version 2 is required")
        if value["store_namespace"] != self.namespace:
            raise ActionStoreCorruptError("snapshot belongs to a different action-store namespace")
        journal_generation = self._as_int(value["journal_generation"], label="snapshot journal generation", minimum=2)
        snapshot_generation = self._as_int(value["snapshot_generation"], label="snapshot generation", minimum=1)
        base = self._as_int(value["base_sequence"], label="snapshot base sequence")
        watermark = self._as_int(value["watermark"], label="snapshot watermark")
        if base != watermark:
            raise ActionStoreCorruptError("snapshot base sequence must equal its watermark")
        prior = value["prior_generation"]
        prior_fields = {"journal_generation", "snapshot_generation", "watermark", "sequence"}
        if not isinstance(prior, dict) or set(prior) != prior_fields:
            raise ActionStoreCorruptError("snapshot prior generation has an invalid schema")
        prior_generation = ActionStoreGeneration(
            self._as_int(prior["journal_generation"], label="prior journal generation", minimum=1),
            self._as_int(prior["snapshot_generation"], label="prior snapshot generation"),
            self._as_int(prior["watermark"], label="prior watermark"),
            self._as_int(prior["sequence"], label="prior sequence"),
        )
        if (
            journal_generation != prior_generation.journal_generation + 1
            or snapshot_generation != prior_generation.snapshot_generation + 1
            or watermark != prior_generation.sequence
            or prior_generation.sequence < prior_generation.watermark
        ):
            raise ActionStoreCorruptError("snapshot generations or watermark are not monotonic")
        try:
            candidate = _copy_json_mapping(value["candidate"], label="snapshot candidate", error=ActionStoreCorruptError)
        except TypeError as exc:
            raise ActionStoreCorruptError("snapshot candidate is invalid") from exc
        raw_receipts = value["receipts"]
        if not isinstance(raw_receipts, list) or len(raw_receipts) > _MAX_RECEIPTS or len(raw_receipts) != watermark:
            raise ActionStoreCorruptError("snapshot has an invalid receipt count")
        receipts: list[ActionReceipt] = []
        seen: set[str] = set()
        previous_receipt_generation = 1
        for sequence, raw in enumerate(raw_receipts, start=1):
            receipt_fields = {"action_id", "request_digest", "sequence", "journal_generation"}
            if not isinstance(raw, dict) or set(raw) != receipt_fields:
                raise ActionStoreCorruptError("snapshot receipt has an invalid schema")
            if self._as_int(raw["sequence"], label="snapshot receipt sequence", minimum=1) != sequence:
                raise ActionStoreCorruptError("snapshot receipts are not contiguous")
            action_id = raw["action_id"]
            try:
                self._validate_action_id(action_id)
            except ValueError as exc:
                raise ActionStoreCorruptError("snapshot receipt has an invalid action id") from exc
            digest = raw["request_digest"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ActionStoreCorruptError("snapshot receipt has an invalid request digest")
            receipt_generation = self._as_int(
                raw["journal_generation"], label="snapshot receipt journal generation", minimum=1
            )
            if receipt_generation > prior_generation.journal_generation:
                raise ActionStoreCorruptError("snapshot receipt generation is later than its checkpoint journal")
            if receipt_generation < previous_receipt_generation:
                raise ActionStoreCorruptError("snapshot receipt generations are not chronological")
            if sequence <= prior_generation.watermark and receipt_generation >= prior_generation.journal_generation:
                raise ActionStoreCorruptError("snapshot receipt predates its journal generation")
            if sequence > prior_generation.watermark and receipt_generation != prior_generation.journal_generation:
                raise ActionStoreCorruptError("snapshot receipt conflicts with its compaction boundary")
            if action_id in seen:
                raise ActionStoreCorruptError("snapshot has a duplicate action id")
            seen.add(action_id)
            receipts.append(ActionReceipt(self.namespace, action_id, digest, sequence, receipt_generation))
            previous_receipt_generation = receipt_generation
        return _Snapshot(journal_generation, snapshot_generation, watermark, prior_generation, candidate, tuple(receipts))

    @staticmethod
    def _same_receipt(left: ActionReceipt, right: ActionReceipt) -> bool:
        return (
            left.action_id == right.action_id
            and left.request_digest == right.request_digest
            and left.sequence == right.sequence
            and left.journal_generation == right.journal_generation
        )

    def _read_state(self, held: _HeldStore) -> _State:
        journal = self._read_journal(held)
        snapshot = self._read_snapshot(held)
        if snapshot is None:
            if journal.base_sequence != 0:
                raise ActionStoreCorruptError("journal has a nonzero base but no snapshot")
            receipts = tuple(action.receipt for action in journal.actions)
            replay = ActionStoreReplay(
                self.namespace,
                ActionStoreGeneration(
                    journal.generation, 0, 0, journal.actions[-1].receipt.sequence if journal.actions else 0
                ),
                None,
                tuple(self._copy_action(action) for action in journal.actions),
                tuple(self._copy_receipt(receipt) for receipt in receipts),
            )
            return _State(replay, journal, None, False)

        if journal.generation == snapshot.journal_generation:
            if journal.base_sequence != snapshot.watermark:
                raise ActionStoreCorruptError("journal base does not match its snapshot watermark")
            cleanup_required = False
        elif journal.generation == snapshot.journal_generation - 1:
            # A snapshot became durable before the journal cleanup.  The old
            # journal is accepted only if it proves exactly the same prefix;
            # anything else is an ambiguous crash and is fenced.
            if journal.base_sequence != snapshot.prior_generation.watermark:
                raise ActionStoreCorruptError("pre-cleanup journal base conflicts with snapshot")
            if journal.actions and journal.actions[-1].receipt.sequence != snapshot.watermark:
                raise ActionStoreCorruptError("pre-cleanup journal has an ambiguous tail")
            if not journal.actions and journal.base_sequence != snapshot.watermark:
                raise ActionStoreCorruptError("pre-cleanup journal is missing snapshot actions")
            cleanup_required = True
        else:
            raise ActionStoreCorruptError("snapshot and journal generations do not coordinate")

        receipt_by_sequence = {receipt.sequence: receipt for receipt in snapshot.receipts}
        pending: list[ActionRequest] = []
        seen_ids = {receipt.action_id for receipt in snapshot.receipts}
        for action in journal.actions:
            if action.receipt.sequence <= snapshot.watermark:
                expected = receipt_by_sequence.get(action.receipt.sequence)
                if expected is None or not self._same_receipt(action.receipt, expected):
                    raise ActionStoreCorruptError("journal prefix conflicts with snapshot receipts")
                continue
            if cleanup_required:
                raise ActionStoreCorruptError("pre-cleanup journal has an ambiguous action after the watermark")
            if action.receipt.action_id in seen_ids:
                raise ActionStoreCorruptError("journal action duplicates a checkpoint receipt")
            seen_ids.add(action.receipt.action_id)
            pending.append(action)
        all_receipts = tuple(snapshot.receipts) + tuple(action.receipt for action in pending)
        if len(all_receipts) > _MAX_RECEIPTS:
            raise ActionStoreCorruptError("action receipt limit exceeded; an offline export is required")
        effective_generation = snapshot.journal_generation
        current_sequence = pending[-1].receipt.sequence if pending else snapshot.watermark
        replay = ActionStoreReplay(
            self.namespace,
            ActionStoreGeneration(
                effective_generation, snapshot.snapshot_generation, snapshot.watermark, current_sequence
            ),
            _copy_json_mapping(snapshot.candidate, label="snapshot candidate"),
            tuple(self._copy_action(action) for action in pending),
            tuple(self._copy_receipt(receipt) for receipt in all_receipts),
        )
        return _State(replay, journal, snapshot, cleanup_required)

    @staticmethod
    def _copy_receipt(receipt: ActionReceipt) -> ActionReceipt:
        return ActionReceipt(
            receipt.store_namespace, receipt.action_id, receipt.request_digest, receipt.sequence, receipt.journal_generation
        )

    @classmethod
    def _copy_action(cls, action: ActionRequest) -> ActionRequest:
        return ActionRequest(cls._copy_receipt(action.receipt), _copy_json_mapping(action.request, label="action request"))

    def _snapshot_wire(self, state: _State, candidate: Mapping[str, Any]) -> bytes:
        generation = state.replay.generation
        receipts = state.replay.receipts
        if len(receipts) > _MAX_RECEIPTS:
            raise ActionStoreConflictError("action receipt limit exceeded; an offline export is required")
        body = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "snapshot",
            "store_namespace": self.namespace,
            "journal_generation": generation.journal_generation + 1,
            "snapshot_generation": generation.snapshot_generation + 1,
            "base_sequence": receipts[-1].sequence if receipts else 0,
            "watermark": receipts[-1].sequence if receipts else 0,
            "prior_generation": {
                "journal_generation": generation.journal_generation,
                "snapshot_generation": generation.snapshot_generation,
                "watermark": generation.watermark,
                "sequence": generation.sequence,
            },
            "candidate": _copy_json_mapping(candidate, label="checkpoint candidate"),
            "receipts": [
                {
                    "action_id": receipt.action_id,
                    "request_digest": receipt.request_digest,
                    "sequence": receipt.sequence,
                    "journal_generation": receipt.journal_generation,
                }
                for receipt in receipts
            ],
        }
        encoded = _canonical_json(body).encode("utf-8")
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ActionStoreConflictError("encoded snapshot exceeds the private record size limit")
        return encoded

    def _replace_file(self, held: _HeldStore, *, target: str, data: bytes, prefix: str) -> None:
        temporary = f".{prefix}.{uuid.uuid4().hex}.tmp"
        fd: int | None = None
        published = False
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_NONBLOCK, 0o600,
                         dir_fd=held.root_fd)
            os.fchmod(fd, 0o600)
            self._assert_file_binding(held, temporary, fd)
            self._write_all(fd, data)
            self._assert_file_binding(held, temporary, fd)
            os.fsync(fd)
            self._assert_file_binding(held, temporary, fd)
            held.assert_bound()
            os.rename(temporary, target, src_dir_fd=held.root_fd, dst_dir_fd=held.root_fd)
            published = True
            # Keep the fsynced staging descriptor open across rename. Opening
            # the destination anew would accept a same-owner/mode replacement
            # inserted by a rename wrapper between publication and fsync.
            self._assert_file_binding(held, target, fd)
            os.fsync(held.root_fd)
            self._assert_file_binding(held, target, fd)
        except CronActionStoreError:
            raise
        except OSError as exc:
            raise ActionStorePersistenceError(f"could not durably replace {target}") from exc
        finally:
            if fd is not None:
                os.close(fd)
            if not published:
                try:
                    os.unlink(temporary, dir_fd=held.root_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def _verify_snapshot_durable(self, held: _HeldStore) -> None:
        fd: int | None = None
        try:
            fd = os.open("snapshot.json", os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=held.root_fd)
            self._assert_file_binding(held, "snapshot.json", fd)
            os.fsync(fd)
            self._assert_file_binding(held, "snapshot.json", fd)
            os.fsync(held.root_fd)
            self._assert_file_binding(held, "snapshot.json", fd)
        except CronActionStoreError:
            raise
        except OSError as exc:
            raise ActionStorePersistenceError("could not verify snapshot durability") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _finish_cleanup(self, held: _HeldStore, state: _State) -> None:
        if not state.cleanup_required or state.snapshot is None:
            return
        # Re-prove the already-published snapshot before deleting the only
        # replayable journal prefix.  This is the crash ambiguity fence.
        self._verify_snapshot_durable(held)
        self._replace_file(
            held,
            target="journal.jsonl",
            data=self._header_wire(generation=state.snapshot.journal_generation, base_sequence=state.snapshot.watermark),
            prefix="journal",
        )

    def replay(self) -> ActionStoreReplay:
        """Return a detached generic candidate and its ordered pending actions."""
        with self._held_lock() as held:
            state = self._read_state(held)
            self._verify_journal_durable(held)
            if state.snapshot is not None:
                self._verify_snapshot_durable(held)
            return self._copy_replay(state.replay)

    @classmethod
    def _copy_replay(cls, replay: ActionStoreReplay) -> ActionStoreReplay:
        return ActionStoreReplay(
            replay.store_namespace,
            ActionStoreGeneration(
                replay.generation.journal_generation,
                replay.generation.snapshot_generation,
                replay.generation.watermark,
                replay.generation.sequence,
            ),
            None if replay.candidate is None else _copy_json_mapping(replay.candidate, label="snapshot candidate"),
            tuple(cls._copy_action(action) for action in replay.pending_actions),
            tuple(cls._copy_receipt(receipt) for receipt in replay.receipts),
        )

    def append(self, *, action_id: str, request: Mapping[str, Any]) -> ActionReceipt:
        """Durably append one opaque request or return its exact old receipt."""
        action_id = self._validate_action_id(action_id)
        canonical_request = _copy_json_mapping(request, label="action request")
        digest = _digest_request(canonical_request)
        with self._held_lock() as held:
            state = self._read_state(held)
            self._verify_journal_durable(held)
            if state.cleanup_required:
                self._finish_cleanup(held, state)
                state = self._read_state(held)
            for receipt in state.replay.receipts:
                if receipt.action_id != action_id:
                    continue
                if receipt.request_digest != digest:
                    raise ActionStoreConflictError("action_id already has a different canonical request")
                # A caller may be retrying after an fsync failure.  Re-prove
                # the relevant durable object before acknowledging success.
                if receipt.sequence <= state.replay.generation.watermark:
                    self._verify_snapshot_durable(held)
                else:
                    self._verify_journal_durable(held)
                return self._copy_receipt(receipt)
            if len(state.replay.receipts) >= _MAX_RECEIPTS:
                raise ActionStoreConflictError("action receipt limit exceeded; an offline export is required")
            sequence = state.replay.receipts[-1].sequence + 1 if state.replay.receipts else 1
            receipt = ActionReceipt(self.namespace, action_id, digest, sequence, state.journal.generation)
            self._append_journal(held, ActionRequest(receipt, canonical_request))
            return self._copy_receipt(receipt)

    def checkpoint(self, *, expected_generation: ActionStoreGeneration, candidate: Mapping[str, Any]) -> ActionStoreReplay:
        """CAS-checkpoint an opaque candidate, then compact the journal.

        The same call is safe to retry after a filesystem failure: a published
        matching snapshot records its prior generation and is settled before
        success is returned.  A different candidate for that prior generation
        is a conflict rather than an accidental overwrite.
        """
        if not isinstance(expected_generation, ActionStoreGeneration):
            raise ValueError("expected_generation must be an ActionStoreGeneration returned by replay()")
        expected_generation = self._validate_generation(expected_generation)
        canonical_candidate = _copy_json_mapping(candidate, label="checkpoint candidate")
        with self._held_lock() as held:
            state = self._read_state(held)
            self._verify_journal_durable(held)
            if state.snapshot is not None:
                self._verify_snapshot_durable(held)
            if state.replay.generation != expected_generation:
                snapshot = state.snapshot
                if (
                    snapshot is not None
                    and snapshot.prior_generation == expected_generation
                    and not state.replay.pending_actions
                    and state.replay.generation.watermark == snapshot.watermark
                ):
                    if _canonical_json(snapshot.candidate) != _canonical_json(canonical_candidate):
                        raise ActionStoreConflictError("checkpoint generation already has a different candidate")
                    if state.cleanup_required:
                        self._finish_cleanup(held, state)
                    else:
                        # The caller may be retrying the directory fsync after
                        # the replacement was visible.  Do not acknowledge the
                        # old checkpoint until both durable objects are proven.
                        self._verify_snapshot_durable(held)
                        self._verify_journal_durable(held)
                    return self._copy_replay(self._read_state(held).replay)
                raise ActionStoreStaleGenerationError("checkpoint generation is stale")
            if state.cleanup_required:
                self._finish_cleanup(held, state)
                state = self._read_state(held)
                if state.replay.generation != expected_generation:
                    raise ActionStoreStaleGenerationError("checkpoint generation changed during recovery")
            snapshot_data = self._snapshot_wire(state, canonical_candidate)
            # This call fsyncs the snapshot file, replaces it, and fsyncs the
            # directory before _replace_file touches journal.jsonl.
            self._replace_file(held, target="snapshot.json", data=snapshot_data, prefix="snapshot")
            snapshot = self._read_snapshot(held)
            if snapshot is None:  # pragma: no cover - defensive after successful replace
                raise ActionStorePersistenceError("published action snapshot disappeared")
            self._replace_file(
                held,
                target="journal.jsonl",
                data=self._header_wire(generation=snapshot.journal_generation, base_sequence=snapshot.watermark),
                prefix="journal",
            )
            return self._copy_replay(self._read_state(held).replay)
