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
from typing import Any, Generator, Mapping, Never, TypedDict, cast

from nanobot.cron.types import CronJob, CronStore

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
_LEGACY_CONVERSION_VERSION = 1
_LEGACY_BACKUP_DIR = "legacy-backups"
_LEGACY_MANIFEST = "conversion.json"
_LEGACY_COMPLETION = "conversion-complete.json"
_LEGACY_CANDIDATE_EVIDENCE_KEY = "legacy_conversion_evidence"
_LEGACY_CANDIDATE_EVIDENCE_VERSION = 1
_MISSING = object()


class _ConversionAction(TypedDict):
    status: str
    source_digest: str | None
    backup_digest: str | None
    operations: list[Any]


class _ConversionJobs(TypedDict):
    status: str
    source_digest: str | None
    backup_digest: str | None


class _ConversionManifest(TypedDict):
    schema_version: int
    kind: str
    store_namespace: str
    conversion_identity: str
    identity_action_digest: str | None
    identity_jobs_digest: str | None
    action: _ConversionAction
    jobs: _ConversionJobs


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


class ActionStoreLegacyConversionError(CronActionStoreError):
    """A stopped-writer legacy source is malformed or unsafe to convert."""


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
class LegacyConversionResult:
    """Detached result from the stopped-writer legacy conversion boundary."""

    conversion_identity: str
    action_count: int
    journal_converted: bool
    snapshot_converted: bool
    snapshot_status: str
    action_backup_digest: str | None
    jobs_backup_digest: str | None


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
        self.store.require_held_store_binding(self)


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


def _reject_constant(value: str) -> Never:
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


def decode_scheduler_state_json(data: bytes) -> object:
    return _strict_json_loads(data, label="scheduler state")


def decode_pinned_scheduler_state_json(data: bytes) -> object:
    return _strict_json_loads(data, label="pinned scheduler state")


def _canonical_json(value: object, *, error: type[CronActionStoreError] = ActionStoreConflictError) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError, RecursionError) as exc:
        raise error("action store values must be finite JSON data") from exc


def _copy_json_mapping(
    value: object, *, label: str, error: type[CronActionStoreError] = ActionStoreConflictError
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{label} must be a JSON object")
    # Mapping keys remain untrusted until JSON encoding has accepted them.
    encoded = _canonical_json(dict(cast(Mapping[object, object], value)), error=error).encode("utf-8")
    if len(encoded) > _MAX_REQUEST_BYTES:
        raise error(f"{label} exceeds the { _MAX_REQUEST_BYTES } byte limit")
    parsed = _strict_json_loads(encoded, label=label)
    if not isinstance(parsed, dict):  # defensive; mappings encode to objects
        raise error(f"{label} must be a JSON object")
    # The JSON decoder constructs string-keyed objects; values are opaque JSON.
    return cast(dict[str, Any], parsed)


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

    def __init__(self, root: Path, *, namespace: str, _allow_legacy_source: bool = False):
        self._flock = self._require_qualified_platform()
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
            if not _allow_legacy_source:
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
    def _require_qualified_platform():
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
        # Infer the concrete stdlib module type, which has no annotation spelling.
        return _fcntl

    @staticmethod
    def _validate_namespace(namespace: object) -> str:
        if not isinstance(namespace, str) or not namespace or len(namespace.encode("utf-8")) > _MAX_NAMESPACE_BYTES:
            raise ValueError(f"namespace must be a non-empty UTF-8 string no longer than {_MAX_NAMESPACE_BYTES} bytes")
        return namespace

    @staticmethod
    def _validate_action_id(action_id: object) -> str:
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
            self._flock.flock(lock_fd, self._flock.LOCK_EX)
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

    def require_held_store_binding(self, held: _HeldStore) -> None:
        """Reprove the existing root and lock checks for an already held operation."""
        return self._assert_binding(held)

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
                    self._flock.flock(lock_fd, self._flock.LOCK_UN)
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
        try:
            manifest = self._read_conversion_manifest(held)
        except FileNotFoundError:
            manifest = None
        try:
            completion = self._read_conversion_completion(held)
        except FileNotFoundError:
            completion = None
        if manifest is not None:
            action = manifest["action"]
            jobs = manifest["jobs"]
            if action["status"] not in {"absent", "converted"} or jobs["status"] == "pending":
                raise ActionStoreCorruptError("legacy conversion is incomplete; only the explicit converter may resume")
            raise ActionStoreCorruptError("legacy action journal reappeared after conversion; mixed writers are fenced")
        if completion is not None or self._has_legacy_backup_evidence(held):
            raise ActionStoreCorruptError("legacy action journal reappeared after conversion; mixed writers are fenced")
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
        # This private parser receives decoded JSON; field values are checked below.
        value = cast(dict[str, Any], value)
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
        if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
            raise ActionStoreCorruptError("journal action has an invalid schema")
        # Exact field-set validation proves string keys, not the opaque values.
        value = cast(dict[str, Any], value)
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
        if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
            raise ActionStoreCorruptError("action snapshot has an invalid schema")
        # Exact field-set validation proves string keys; nested values are checked below.
        value = cast(dict[str, Any], value)
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
        if not isinstance(prior, dict) or set(cast(dict[object, object], prior)) != prior_fields:
            raise ActionStoreCorruptError("snapshot prior generation has an invalid schema")
        prior = cast(dict[str, Any], prior)  # Exact string-key field set checked above.
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
        if not isinstance(raw_receipts, list):
            raise ActionStoreCorruptError("snapshot has an invalid receipt count")
        raw_receipts = cast(list[object], raw_receipts)  # Elements are validated individually.
        if len(raw_receipts) > _MAX_RECEIPTS or len(raw_receipts) != watermark:
            raise ActionStoreCorruptError("snapshot has an invalid receipt count")
        receipts: list[ActionReceipt] = []
        seen: set[str] = set()
        previous_receipt_generation = 1
        for sequence, raw in enumerate(raw_receipts, start=1):
            receipt_fields = {"action_id", "request_digest", "sequence", "journal_generation"}
            if not isinstance(raw, dict) or set(cast(dict[object, object], raw)) != receipt_fields:
                raise ActionStoreCorruptError("snapshot receipt has an invalid schema")
            raw = cast(dict[str, Any], raw)  # Exact string-key field set checked above.
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

    # The following stopped-writer boundary is deliberately here, rather than
    # in CronService.  It must establish source/backups/receipts before later
    # packages give any converted action job semantics or execution authority.
    @classmethod
    def convert_legacy(cls, root: Path, *, namespace: str) -> LegacyConversionResult:
        """Convert stopped legacy ``action.jsonl``/``jobs.json`` sources.

        This is an explicit, local migration operation.  It is not called by
        startup and it never writes service state or enables execution.
        """
        store = cls(root, namespace=namespace, _allow_legacy_source=True)
        return store.convert_legacy_sources()

    @staticmethod
    def _legacy_digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _conversion_identity(self, action_digest: str | None, jobs_digest: str | None) -> str:
        action_token = action_digest if action_digest is not None else "no-legacy-actions"
        jobs_token = jobs_digest if jobs_digest is not None else "no-legacy-jobs"
        return hashlib.sha256(
            b"nanobot-cron-legacy-conversion-v1\0"
            + self.namespace.encode("utf-8")
            + b"\0"
            + action_token.encode("ascii")
            + b"\0"
            + jobs_token.encode("ascii")
        ).hexdigest()

    @staticmethod
    def _digest_value(value: object) -> str:
        return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _require_digest(value: object, *, label: str, nullable: bool = False) -> str | None:
        if nullable and value is None:
            return None
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ActionStoreCorruptError(f"{label} must be a SHA-256 digest")
        return value

    def _read_legacy_source(self, held: _HeldStore, name: str) -> bytes | None:
        try:
            return self._read_private_bytes(held, name, maximum=_MAX_JOURNAL_BYTES)
        except FileNotFoundError:
            return None

    @contextmanager
    def _legacy_backup_fd(self, held: _HeldStore) -> Generator[int, None, None]:
        fd: int | None = None
        created = False
        try:
            try:
                fd = os.open(_LEGACY_BACKUP_DIR, self._dir_flags(), dir_fd=held.root_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(_LEGACY_BACKUP_DIR, 0o700, dir_fd=held.root_fd)
                except FileExistsError:
                    pass
                else:
                    created = True
                fd = os.open(_LEGACY_BACKUP_DIR, self._dir_flags(), dir_fd=held.root_fd)
            self._assert_backup_dir_binding(held, fd)
            if created:
                os.fchmod(fd, 0o700)
                self._assert_backup_dir_binding(held, fd)
                os.fsync(fd)
                self._assert_backup_dir_binding(held, fd)
                os.fsync(held.root_fd)
                self._assert_backup_dir_binding(held, fd)
            yield fd
        except CronActionStoreError:
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ActionStoreCorruptError("legacy backup directory must not be a symlink") from exc
            raise ActionStorePersistenceError("could not establish legacy backup directory") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _assert_backup_dir_binding(self, held: _HeldStore, backup_fd: int) -> os.stat_result:
        """Bind the held backup directory to the configured root entry."""
        held.assert_bound()
        info = os.fstat(backup_fd)
        entry = os.stat(_LEGACY_BACKUP_DIR, dir_fd=held.root_fd, follow_symlinks=False)
        _private(info, directory=True)
        _private(entry, directory=True)
        if _inode(info) != _inode(entry):
            raise ActionStoreCorruptError("legacy backup directory binding changed")
        return info

    def _assert_backup_file_binding(
        self, held: _HeldStore, backup_fd: int, name: str, fd: int
    ) -> os.stat_result:
        """Bind a backup descriptor to its entry and enclosing held directory."""
        self._assert_backup_dir_binding(held, backup_fd)
        info = os.fstat(fd)
        entry = os.stat(name, dir_fd=backup_fd, follow_symlinks=False)
        _private(info, directory=False)
        _private(entry, directory=False)
        if _inode(info) != _inode(entry):
            raise ActionStoreCorruptError("legacy backup file binding changed")
        return info

    def _backup_legacy_bytes(self, held: _HeldStore, data: bytes) -> str:
        """Publish a content-addressed backup without ever writing its final name in place."""
        digest = self._legacy_digest(data)
        name = digest + ".bin"
        with self._legacy_backup_fd(held) as backup_fd:
            try:
                os.stat(name, dir_fd=backup_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ActionStorePersistenceError("could not inspect legacy conversion backup") from exc
            else:
                existing = self._verify_legacy_backup(held, digest)
                if existing != data:
                    raise ActionStoreCorruptError("content-addressed legacy backup conflicts with source bytes")
                return digest

            fd: int | None = None
            temporary = f".legacy-backup.{uuid.uuid4().hex}.tmp"
            try:
                fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_NONBLOCK,
                    0o600,
                    dir_fd=backup_fd,
                )
                os.fchmod(fd, 0o600)
                self._assert_backup_file_binding(held, backup_fd, temporary, fd)
                self._write_all(fd, data)
                self._assert_backup_file_binding(held, backup_fd, temporary, fd)
                os.fsync(fd)
                self._assert_backup_file_binding(held, backup_fd, temporary, fd)
                self._assert_backup_dir_binding(held, backup_fd)
                os.rename(temporary, name, src_dir_fd=backup_fd, dst_dir_fd=backup_fd)
                # Keep the fsynced staging descriptor open so a same-owner
                # replacement at the final name is never acknowledged.
                self._assert_backup_file_binding(held, backup_fd, name, fd)
                os.fsync(backup_fd)
                self._assert_backup_file_binding(held, backup_fd, name, fd)
                return digest
            except CronActionStoreError:
                raise
            except OSError as exc:
                raise ActionStorePersistenceError("could not durably preserve legacy source bytes") from exc
            finally:
                if fd is not None:
                    os.close(fd)

    def _verify_legacy_backup(self, held: _HeldStore, digest: str) -> bytes:
        """Read, fsync, and bind a recorded immutable backup before trusting it."""
        name = digest + ".bin"
        with self._legacy_backup_fd(held) as backup_fd:
            fd: int | None = None
            try:
                try:
                    fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=backup_fd)
                except FileNotFoundError as exc:
                    raise ActionStoreCorruptError("legacy conversion backup disappeared") from exc
                info = self._assert_backup_file_binding(held, backup_fd, name, fd)
                if info.st_size > _MAX_JOURNAL_BYTES:
                    raise ActionStoreCorruptError("legacy conversion backup has an unsafe binding or size")
                chunks: list[bytes] = []
                remaining = info.st_size
                while remaining:
                    chunk = os.read(fd, remaining)
                    if not chunk:
                        raise ActionStoreCorruptError("legacy conversion backup was truncated during read")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                after = self._assert_backup_file_binding(held, backup_fd, name, fd)
                if _inode(after) != _inode(info) or after.st_size != info.st_size:
                    raise ActionStoreCorruptError("legacy conversion backup binding changed during read")
                data = b"".join(chunks)
                if self._legacy_digest(data) != digest:
                    raise ActionStoreCorruptError("legacy conversion backup digest does not match its name")
                os.fsync(fd)
                self._assert_backup_file_binding(held, backup_fd, name, fd)
                os.fsync(backup_fd)
                self._assert_backup_file_binding(held, backup_fd, name, fd)
                return data
            except CronActionStoreError:
                raise
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ActionStoreCorruptError("legacy conversion backup must not be a symlink") from exc
                raise ActionStorePersistenceError("could not read legacy conversion backup") from exc
            finally:
                if fd is not None:
                    os.close(fd)

    def _read_conversion_manifest(self, held: _HeldStore) -> _ConversionManifest:
        data = self._read_private_bytes(held, _LEGACY_MANIFEST, maximum=_MAX_RECORD_BYTES)
        value = _strict_json_loads(data, label="legacy conversion manifest")
        fields = {
            "schema_version", "kind", "store_namespace", "conversion_identity", "identity_action_digest",
            "identity_jobs_digest", "action", "jobs",
        }
        if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
            raise ActionStoreCorruptError("legacy conversion manifest has an invalid schema")
        # Exact field-set validation proves string keys; the schema follows below.
        value = cast(dict[str, Any], value)
        if type(value["schema_version"]) is not int or value["schema_version"] != _LEGACY_CONVERSION_VERSION:
            raise ActionStoreCorruptError("legacy conversion manifest has an invalid version")
        if value["kind"] != "legacy_cron_conversion" or value["store_namespace"] != self.namespace:
            raise ActionStoreCorruptError("legacy conversion manifest belongs to another store")
        self._require_digest(value["conversion_identity"], label="conversion identity")
        self._require_digest(value["identity_action_digest"], label="identity action source", nullable=True)
        self._require_digest(value["identity_jobs_digest"], label="identity jobs source", nullable=True)
        action = value["action"]
        jobs = value["jobs"]
        action_fields = {"status", "source_digest", "backup_digest", "operations"}
        jobs_fields = {"status", "source_digest", "backup_digest"}
        if not isinstance(action, dict) or set(cast(dict[object, object], action)) != action_fields or not isinstance(jobs, dict) or set(cast(dict[object, object], jobs)) != jobs_fields:
            raise ActionStoreCorruptError("legacy conversion manifest has invalid source state")
        action = cast(dict[str, Any], action)  # Exact source field sets checked above.
        jobs = cast(dict[str, Any], jobs)
        if action["status"] not in {"absent", "pending", "removing", "converted"}:
            raise ActionStoreCorruptError("legacy action conversion status is invalid")
        if jobs["status"] not in {"absent", "unavailable", "pending", "converted"}:
            raise ActionStoreCorruptError("legacy jobs conversion status is invalid")
        action_source = self._require_digest(action["source_digest"], label="legacy action source", nullable=True)
        action_backup = self._require_digest(action["backup_digest"], label="legacy action backup", nullable=True)
        jobs_source = self._require_digest(jobs["source_digest"], label="legacy jobs source", nullable=True)
        jobs_backup = self._require_digest(jobs["backup_digest"], label="legacy jobs backup", nullable=True)
        if not isinstance(action["operations"], list):
            raise ActionStoreCorruptError("legacy action operations are invalid")
        if action["status"] == "absent" and (
            action["source_digest"] is not None or action_backup is not None or action["operations"]
        ):
            raise ActionStoreCorruptError("legacy absent action state contains source data")
        if action["status"] in {"pending", "removing", "converted"} and (
            action["source_digest"] is None or action_backup is None
        ):
            raise ActionStoreCorruptError("legacy action conversion state lacks an immutable backup")
        if action_backup is not None and action_backup != action_source:
            raise ActionStoreCorruptError("legacy action source and backup digests do not agree")
        if jobs["status"] == "absent" and (jobs["source_digest"] is not None or jobs_backup is not None):
            raise ActionStoreCorruptError("legacy absent jobs state contains source data")
        if jobs["status"] in {"unavailable", "converted"} and (
            jobs["source_digest"] is None or jobs_backup is None
        ):
            raise ActionStoreCorruptError("legacy jobs conversion state lacks an immutable backup")
        if jobs_backup is not None and jobs_backup != jobs_source:
            raise ActionStoreCorruptError("legacy jobs source and backup digests do not agree")
        if jobs["status"] == "pending" and jobs["source_digest"] is None and jobs_backup is not None:
            raise ActionStoreCorruptError("legacy pending jobs state has an orphan backup")
        if jobs["status"] == "pending" and jobs["source_digest"] is not None and jobs_backup is None:
            raise ActionStoreCorruptError("legacy pending jobs state lacks an immutable backup")
        if action_backup is not None:
            self._verify_legacy_backup(held, action_backup)
        if jobs_backup is not None:
            self._verify_legacy_backup(held, jobs_backup)
        # All manifest fields and nested source shapes were validated on this path.
        return cast(_ConversionManifest, _copy_json_mapping(value, label="legacy conversion manifest", error=ActionStoreCorruptError))

    def _copy_legacy_conversion_evidence(self, value: object, *, label: str) -> dict[str, Any]:
        evidence = _copy_json_mapping(value, label=label, error=ActionStoreCorruptError)
        fields = {
            "schema_version",
            "conversion_identity",
            "identity_action_digest",
            "identity_jobs_digest",
            "action_backup_digest",
            "jobs_backup_digest",
        }
        if set(evidence) != fields or type(evidence["schema_version"]) is not int:
            raise ActionStoreCorruptError(f"{label} has an invalid schema")
        if evidence["schema_version"] != _LEGACY_CANDIDATE_EVIDENCE_VERSION:
            raise ActionStoreCorruptError(f"{label} has an invalid version")
        self._require_digest(evidence["conversion_identity"], label=f"{label} conversion identity")
        for name in fields - {"schema_version", "conversion_identity"}:
            self._require_digest(evidence[name], label=f"{label} {name}", nullable=True)
        return evidence

    def _read_conversion_completion(self, held: _HeldStore) -> dict[str, Any]:
        data = self._read_private_bytes(held, _LEGACY_COMPLETION, maximum=_MAX_RECORD_BYTES)
        value = _strict_json_loads(data, label="legacy conversion completion provenance")
        fields = {
            "schema_version",
            "kind",
            "store_namespace",
            "action_status",
            "jobs_status",
            "conversion_evidence",
        }
        if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
            raise ActionStoreCorruptError("legacy conversion completion provenance has an invalid schema")
        # Exact field-set validation proves string keys; field values follow below.
        value = cast(dict[str, Any], value)
        if type(value["schema_version"]) is not int or value["schema_version"] != _LEGACY_CONVERSION_VERSION:
            raise ActionStoreCorruptError("legacy conversion completion provenance has an invalid version")
        if value["kind"] != "legacy_cron_conversion_completion" or value["store_namespace"] != self.namespace:
            raise ActionStoreCorruptError("legacy conversion completion provenance belongs to another store")
        if value["action_status"] not in {"absent", "converted"}:
            raise ActionStoreCorruptError("legacy conversion completion provenance has an incomplete action status")
        if value["jobs_status"] not in {"absent", "unavailable", "converted"}:
            raise ActionStoreCorruptError("legacy conversion completion provenance has an incomplete jobs status")
        evidence = self._copy_legacy_conversion_evidence(
            value["conversion_evidence"], label="legacy conversion completion provenance evidence"
        )
        result = _copy_json_mapping(value, label="legacy conversion completion provenance", error=ActionStoreCorruptError)
        result["conversion_evidence"] = evidence
        return result

    def _completion_provenance(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        action = _copy_json_mapping(manifest["action"], label="legacy action manifest", error=ActionStoreCorruptError)
        jobs = _copy_json_mapping(manifest["jobs"], label="legacy jobs manifest", error=ActionStoreCorruptError)
        if action["status"] not in {"absent", "converted"} or jobs["status"] not in {
            "absent", "unavailable", "converted"
        }:
            raise ActionStoreCorruptError("legacy conversion is incomplete; completion provenance cannot be written")
        return {
            "schema_version": _LEGACY_CONVERSION_VERSION,
            "kind": "legacy_cron_conversion_completion",
            "store_namespace": self.namespace,
            "action_status": action["status"],
            "jobs_status": jobs["status"],
            "conversion_evidence": self._legacy_conversion_evidence(manifest),
        }

    def _write_conversion_completion(
        self, held: _HeldStore, manifest: Mapping[str, Any], *, current_jobs_digest: str | None
    ) -> None:
        """Publish terminal conversion provenance only after re-proving its evidence."""
        action_bytes = self._read_legacy_source(held, "action.jsonl")
        if action_bytes is not None:
            raise ActionStoreCorruptError("legacy action journal reappeared before completion provenance")
        jobs_bytes = self._read_legacy_source(held, "jobs.json")
        jobs_digest = None if jobs_bytes is None else self._legacy_digest(jobs_bytes)
        if jobs_digest != current_jobs_digest:
            raise ActionStoreCorruptError("legacy jobs snapshot changed before completion provenance")
        identity = self._verified_conversion_identity(held, manifest, current_jobs_digest=jobs_digest)
        operations = self._reconstruct_manifest_operations(held, manifest, conversion_identity=identity)
        self._require_legacy_operation_receipts(held, operations)
        provenance = self._completion_provenance(manifest)
        try:
            existing = self._read_conversion_completion(held)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if _canonical_json(existing) != _canonical_json(provenance):
                raise ActionStoreCorruptError("legacy conversion completion provenance does not match the manifest")
            return
        encoded = _canonical_json(provenance).encode("utf-8")
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ActionStoreLegacyConversionError("legacy conversion completion provenance exceeds the private record limit")
        self._replace_file(held, target=_LEGACY_COMPLETION, data=encoded, prefix="conversion-complete")

    def _remove_conversion_completion(self, held: _HeldStore) -> None:
        """Durably retire prior terminal provenance before explicit recovery resumes."""
        fd: int | None = None
        try:
            fd = os.open(_LEGACY_COMPLETION, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=held.root_fd)
            self._assert_file_binding(held, _LEGACY_COMPLETION, fd)
            os.unlink(_LEGACY_COMPLETION, dir_fd=held.root_fd)
            held.assert_bound()
            os.fsync(held.root_fd)
            held.assert_bound()
        except FileNotFoundError:
            return
        except CronActionStoreError:
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ActionStoreCorruptError("legacy conversion completion provenance must not be a symlink") from exc
            raise ActionStorePersistenceError("could not retire legacy conversion completion provenance") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _has_legacy_backup_evidence(self, held: _HeldStore) -> bool:
        """Detect final content-addressed backups without treating temp files as evidence."""
        fd: int | None = None
        try:
            fd = os.open(_LEGACY_BACKUP_DIR, self._dir_flags(), dir_fd=held.root_fd)
            self._assert_backup_dir_binding(held, fd)
            return any(re.fullmatch(r"[0-9a-f]{64}\.bin", name) is not None for name in os.listdir(fd))
        except FileNotFoundError:
            return False
        except CronActionStoreError:
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ActionStoreCorruptError("legacy backup directory must not be a symlink") from exc
            raise ActionStorePersistenceError("could not inspect legacy conversion backups") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _write_conversion_manifest(self, held: _HeldStore, manifest: Mapping[str, Any]) -> None:
        data = _canonical_json(_copy_json_mapping(manifest, label="legacy conversion manifest")).encode("utf-8")
        if len(data) > _MAX_RECORD_BYTES:
            raise ActionStoreLegacyConversionError("legacy conversion manifest exceeds the private record limit")
        self._replace_file(held, target=_LEGACY_MANIFEST, data=data, prefix="conversion")

    def _parse_legacy_actions(self, data: bytes) -> list[tuple[str, dict[str, Any], int]]:
        if data and not data.endswith(b"\n"):
            raise ActionStoreLegacyConversionError("legacy action journal must be newline terminated")
        parsed: list[tuple[str, dict[str, Any], int]] = []
        for ordinal, line in enumerate(data.splitlines(), start=1):
            if not line:
                raise ActionStoreLegacyConversionError("legacy action journal contains an empty line")
            try:
                value = _strict_json_loads(line, label="legacy action")
            except ActionStoreCorruptError as exc:
                raise ActionStoreLegacyConversionError("legacy action journal is malformed") from exc
            if not isinstance(value, dict) or set(cast(dict[object, object], value)) != {"action", "params"}:
                raise ActionStoreLegacyConversionError("legacy action has an invalid schema")
            value = cast(dict[str, Any], value)  # Exact string-key field set checked above.
            operation = value["action"]
            if operation not in {"add", "update", "del"} or not isinstance(value["params"], dict):
                raise ActionStoreLegacyConversionError("legacy action is not an accepted add/update/del operation")
            parsed.append((operation, _copy_json_mapping(cast(dict[object, object], value["params"]), label="legacy action params"), ordinal))
        return parsed

    @staticmethod
    def _legacy_field(data: Mapping[object, object], camel: str, snake: str, *, default: object = _MISSING) -> object:
        has_camel = camel in data
        has_snake = snake in data
        if has_camel and has_snake:
            raise ActionStoreLegacyConversionError(f"legacy cron record has both {camel} and {snake}")
        if has_camel:
            return data[camel]
        if has_snake:
            return data[snake]
        if default is _MISSING:
            raise ActionStoreLegacyConversionError(f"legacy cron record is missing {camel}")
        return default

    @staticmethod
    def _legacy_known_fields(data: Mapping[object, object], fields: set[str], *, label: str) -> None:
        unknown = set(data) - fields
        if unknown:
            raise ActionStoreLegacyConversionError(f"{label} has unknown fields")

    @staticmethod
    def _legacy_int(value: object, *, label: str, nullable: bool = False, minimum: int = 0) -> int | None:
        if nullable and value is None:
            return None
        if type(value) is not int or value < minimum:
            raise ActionStoreLegacyConversionError(f"{label} must be an exact integer")
        return value

    @staticmethod
    def _legacy_bool(value: object, *, label: str) -> bool:
        if type(value) is not bool:
            raise ActionStoreLegacyConversionError(f"{label} must be a boolean")
        return value

    @staticmethod
    def _legacy_string(value: object, *, label: str, nullable: bool = False, empty: bool = False) -> str | None:
        if nullable and value is None:
            return None
        if not isinstance(value, str) or (not empty and not value):
            raise ActionStoreLegacyConversionError(f"{label} must be a non-empty string")
        return value

    @staticmethod
    def _legacy_mapping(value: object, *, label: str) -> dict[object, object]:
        if not isinstance(value, dict):
            raise ActionStoreLegacyConversionError(f"{label} must be an object")
        return cast(dict[object, object], value)

    def _validate_legacy_schedule(self, value: object) -> None:
        data = self._legacy_mapping(value, label="legacy cron schedule")
        self._legacy_known_fields(data, {"kind", "atMs", "at_ms", "everyMs", "every_ms", "expr", "tz"}, label="legacy cron schedule")
        kind = data.get("kind")
        if kind not in {"at", "every", "cron"}:
            raise ActionStoreLegacyConversionError("legacy cron schedule has an unknown kind")
        at_ms = self._legacy_field(data, "atMs", "at_ms", default=None)
        every_ms = self._legacy_field(data, "everyMs", "every_ms", default=None)
        self._legacy_int(at_ms, label="legacy cron atMs", nullable=True)
        self._legacy_int(every_ms, label="legacy cron everyMs", nullable=True, minimum=1)
        expr = data.get("expr")
        tz = data.get("tz")
        self._legacy_string(expr, label="legacy cron expr", nullable=True)
        self._legacy_string(tz, label="legacy cron timezone", nullable=True)
        if kind == "at" and at_ms is None:
            raise ActionStoreLegacyConversionError("legacy at schedule is missing atMs")
        if kind == "every" and every_ms is None:
            raise ActionStoreLegacyConversionError("legacy every schedule is missing everyMs")
        if kind == "cron" and not isinstance(expr, str):
            raise ActionStoreLegacyConversionError("legacy cron schedule is missing expr")

    def _validate_legacy_payload(self, value: object) -> None:
        data = self._legacy_mapping(value, label="legacy cron payload")
        self._legacy_known_fields(
            data,
            {
                "kind", "message", "deliver", "channel", "to", "additionalDestinations", "additional_destinations",
                "channelMeta", "channel_meta", "sessionKey", "session_key", "executionMode", "execution_mode",
                "originChannel", "origin_channel", "originChatId", "origin_chat_id", "originMetadata", "origin_metadata",
            },
            label="legacy cron payload",
        )
        kind = data.get("kind", "agent_turn")
        if kind not in {"agent_turn", "system_event"}:
            raise ActionStoreLegacyConversionError("legacy cron payload has an unknown kind")
        self._legacy_string(data.get("message", ""), label="legacy cron message", empty=True)
        self._legacy_bool(data.get("deliver", False), label="legacy cron delivery flag")
        for field in ("channel", "to"):
            self._legacy_string(data.get(field), label=f"legacy cron {field}", nullable=True)
        destinations = self._legacy_field(data, "additionalDestinations", "additional_destinations", default=[])
        if not isinstance(destinations, list):
            raise ActionStoreLegacyConversionError("legacy cron destinations must be a list")
        for destination in cast(list[object], destinations):
            item = self._legacy_mapping(destination, label="legacy cron destination")
            self._legacy_known_fields(item, {"channel", "to"}, label="legacy cron destination")
            self._legacy_string(item.get("channel"), label="legacy cron destination channel")
            self._legacy_string(item.get("to"), label="legacy cron destination to")
        for camel, snake, label in (
            ("channelMeta", "channel_meta", "legacy cron channel metadata"),
            ("originMetadata", "origin_metadata", "legacy cron origin metadata"),
        ):
            metadata = self._legacy_field(data, camel, snake, default={})
            self._legacy_mapping(metadata, label=label)
        for camel, snake, label in (
            ("sessionKey", "session_key", "legacy cron session key"),
            ("originChannel", "origin_channel", "legacy cron origin channel"),
            ("originChatId", "origin_chat_id", "legacy cron origin chat id"),
        ):
            self._legacy_string(self._legacy_field(data, camel, snake, default=None), label=label, nullable=True)
        mode = self._legacy_field(data, "executionMode", "execution_mode", default="session_bound")
        if mode not in {"session_bound", "legacy_fresh"}:
            raise ActionStoreLegacyConversionError("legacy cron execution mode is invalid")

    def _validate_legacy_state(self, value: object) -> None:
        data = self._legacy_mapping(value, label="legacy cron state")
        self._legacy_known_fields(
            data,
            {"nextRunAtMs", "next_run_at_ms", "lastRunAtMs", "last_run_at_ms", "lastStatus", "last_status", "lastError", "last_error", "lastRunStatus", "last_run_status", "runHistory", "run_history"},
            label="legacy cron state",
        )
        for camel, snake, label in (
            ("nextRunAtMs", "next_run_at_ms", "legacy cron next run"),
            ("lastRunAtMs", "last_run_at_ms", "legacy cron last run"),
        ):
            self._legacy_int(self._legacy_field(data, camel, snake, default=None), label=label, nullable=True)
        last_status = self._legacy_field(data, "lastStatus", "last_status", default=None)
        if last_status is not None and last_status not in {"ok", "error", "skipped"}:
            raise ActionStoreLegacyConversionError("legacy cron last status is invalid")
        last_run_status = self._legacy_field(data, "lastRunStatus", "last_run_status", default=None)
        if last_run_status is not None and last_run_status not in {
            "completed", "succeeded", "failed", "pending", "running", "cancelled", "skipped", "error"
        }:
            raise ActionStoreLegacyConversionError("legacy cron last run status is invalid")
        self._legacy_string(self._legacy_field(data, "lastError", "last_error", default=None), label="legacy cron last error", nullable=True, empty=True)
        history = self._legacy_field(data, "runHistory", "run_history", default=[])
        if not isinstance(history, list):
            raise ActionStoreLegacyConversionError("legacy cron run history must be a list")
        for record in cast(list[object], history):
            item = self._legacy_mapping(record, label="legacy cron run history record")
            self._legacy_known_fields(
                item,
                {"runAtMs", "run_at_ms", "status", "durationMs", "duration_ms", "error", "runId", "run_id", "runStatus", "run_status", "occurrenceId", "occurrence_id"},
                label="legacy cron run history record",
            )
            self._legacy_int(self._legacy_field(item, "runAtMs", "run_at_ms"), label="legacy cron run timestamp")
            if item.get("status") not in {"ok", "error", "skipped"}:
                raise ActionStoreLegacyConversionError("legacy cron run status is invalid")
            self._legacy_int(self._legacy_field(item, "durationMs", "duration_ms", default=0), label="legacy cron run duration")
            self._legacy_string(item.get("error"), label="legacy cron run error", nullable=True, empty=True)
            self._legacy_string(self._legacy_field(item, "runId", "run_id", default=None), label="legacy cron run id", nullable=True)
            run_status = self._legacy_field(item, "runStatus", "run_status", default=None)
            if run_status is not None and run_status not in {"completed", "succeeded", "failed", "pending", "running", "cancelled", "skipped", "error"}:
                raise ActionStoreLegacyConversionError("legacy cron run outcome is invalid")
            self._legacy_string(self._legacy_field(item, "occurrenceId", "occurrence_id", default=None), label="legacy cron occurrence id", nullable=True)

    def _strict_legacy_job(self, value: object) -> CronJob:
        data = self._legacy_mapping(value, label="legacy cron job")
        self._legacy_known_fields(
            data,
            {"id", "name", "enabled", "schedule", "payload", "state", "createdAtMs", "created_at_ms", "updatedAtMs", "updated_at_ms", "deleteAfterRun", "delete_after_run", "jobInstanceId", "job_instance_id", "definitionRevision", "definition_revision"},
            label="legacy cron job",
        )
        self._legacy_string(data.get("id"), label="legacy cron job id")
        self._legacy_string(data.get("name"), label="legacy cron job name")
        self._legacy_bool(data.get("enabled", True), label="legacy cron enabled")
        self._validate_legacy_schedule(data.get("schedule"))
        payload = data.get("payload", {})
        state = data.get("state", {})
        self._validate_legacy_payload(payload)
        self._validate_legacy_state(state)
        for camel, snake, label in (
            ("createdAtMs", "created_at_ms", "legacy cron creation timestamp"),
            ("updatedAtMs", "updated_at_ms", "legacy cron update timestamp"),
        ):
            self._legacy_int(self._legacy_field(data, camel, snake, default=0), label=label)
        self._legacy_bool(self._legacy_field(data, "deleteAfterRun", "delete_after_run", default=False), label="legacy cron delete-after-run")
        self._legacy_string(self._legacy_field(data, "jobInstanceId", "job_instance_id", default=None), label="legacy cron job instance id", nullable=True)
        self._legacy_string(self._legacy_field(data, "definitionRevision", "definition_revision", default=None), label="legacy cron definition revision", nullable=True)
        try:
            return CronJob.from_store_dict(_copy_json_mapping(data, label="legacy cron job"))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ActionStoreLegacyConversionError("legacy cron job cannot be decoded") from exc

    def _strict_legacy_store(self, value: object) -> CronStore:
        data = self._legacy_mapping(value, label="legacy jobs snapshot")
        self._legacy_known_fields(data, {"version", "jobs"}, label="legacy jobs snapshot")
        self._legacy_int(data.get("version", 1), label="legacy jobs version", minimum=1)
        jobs = data.get("jobs")
        if not isinstance(jobs, list):
            raise ActionStoreLegacyConversionError("legacy jobs snapshot must contain a jobs list")
        # _legacy_int above proved the version before constructing the legacy record.
        return CronStore(version=cast(int, data.get("version", 1)), jobs=[self._strict_legacy_job(job) for job in cast(list[object], jobs)])

    def _canonical_legacy_job(self, params: Mapping[str, Any], conversion_identity: str) -> dict[str, Any]:
        try:
            job = self._strict_legacy_job(params)
        except ActionStoreLegacyConversionError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ActionStoreLegacyConversionError("legacy add/update does not contain a valid cron job") from exc
        if not isinstance(cast(object, job.id), str) or not job.id:
            raise ActionStoreLegacyConversionError("legacy cron job has an invalid id")
        canonical = job.to_store_dict()
        instance = hashlib.sha256(
            b"nanobot-cron-job-instance-v1\0" + conversion_identity.encode("ascii") + b"\0" + job.id.encode("utf-8")
        ).hexdigest()
        canonical["jobInstanceId"] = instance
        definition = {
            key: value
            for key, value in canonical.items()
            if key not in {"state", "createdAtMs", "updatedAtMs", "definitionRevision"}
        }
        canonical["definitionRevision"] = self._digest_value(definition)
        return _copy_json_mapping(canonical, label="canonical legacy cron job")

    def _legacy_operations(
        self, parsed: list[tuple[str, dict[str, Any], int]], *, conversion_identity: str, source_digest: str
    ) -> list[dict[str, Any]]:
        operations: list[dict[str, Any]] = []
        for operation, params, ordinal in parsed:
            action_id = hashlib.sha256(
                b"nanobot-cron-legacy-action-v1\0"
                + conversion_identity.encode("ascii")
                + b"\0"
                + source_digest.encode("ascii")
                + b"\0"
                + str(ordinal).encode("ascii")
            ).hexdigest()
            source = {
                "conversion_identity": conversion_identity,
                "source_digest": source_digest,
                "line_ordinal": ordinal,
            }
            if operation in {"add", "update"}:
                job = self._canonical_legacy_job(params, conversion_identity)
                request: dict[str, Any] = {
                    "kind": "legacy_cron_action_v1",
                    "operation": "upsert",
                    "legacy_operation": operation,
                    "job_id": job["id"],
                    "job": job,
                    # Legacy update records often copied stale runtime history.
                    # Package C must merge definition changes without allowing
                    # that copy to overwrite the newer jobs snapshot runtime.
                    "runtime_state_policy": "snapshot_wins",
                    "source": source,
                }
            else:
                job_id = params.get("job_id")
                if not isinstance(job_id, str) or not job_id or set(params) != {"job_id"}:
                    raise ActionStoreLegacyConversionError("legacy del must contain only a non-empty job_id")
                request = {
                    "kind": "legacy_cron_action_v1",
                    "operation": "delete",
                    "legacy_operation": operation,
                    "job_id": job_id,
                    "source": source,
                }
            operations.append({"action_id": action_id, "request": _copy_json_mapping(request, label="legacy request")})
        return operations

    def _verified_conversion_identity(
        self, held: _HeldStore, manifest: Mapping[str, Any], *, current_jobs_digest: str | None
    ) -> str:
        """Recompute the identity from immutable sources, never manifest authority."""
        action = _copy_json_mapping(manifest.get("action"), label="legacy action manifest", error=ActionStoreCorruptError)
        jobs = _copy_json_mapping(manifest.get("jobs"), label="legacy jobs manifest", error=ActionStoreCorruptError)
        action_digest = manifest.get("identity_action_digest")
        if action_digest is not None:
            if action.get("source_digest") != action_digest or not isinstance(action_digest, str) or self._legacy_digest(self._verify_legacy_backup(held, action_digest)) != action_digest:
                raise ActionStoreCorruptError("legacy action backup cannot establish conversion identity")
        jobs_digest = manifest.get("identity_jobs_digest")
        if jobs_digest is not None:
            if not isinstance(jobs_digest, str):
                raise ActionStoreCorruptError("legacy jobs source cannot establish conversion identity")
            if jobs.get("source_digest") == jobs_digest:
                if self._legacy_digest(self._verify_legacy_backup(held, jobs_digest)) != jobs_digest:
                    raise ActionStoreCorruptError("legacy jobs backup cannot establish conversion identity")
            elif jobs.get("source_digest") is None:
                if current_jobs_digest != jobs_digest:
                    raise ActionStoreCorruptError("legacy jobs source changed before identity backup")
            elif self._legacy_digest(self._verify_legacy_backup(held, jobs_digest)) != jobs_digest:
                raise ActionStoreCorruptError("legacy original jobs backup cannot establish conversion identity")
        identity = self._conversion_identity(action_digest, jobs_digest)
        if manifest.get("conversion_identity") != identity:
            raise ActionStoreCorruptError("legacy conversion identity does not match verified source digests")
        return identity

    def _reconstruct_manifest_operations(
        self, held: _HeldStore, manifest: Mapping[str, Any], *, conversion_identity: str
    ) -> list[dict[str, Any]]:
        """Derive actions from immutable bytes; the manifest is evidence, never authority."""
        action = _copy_json_mapping(manifest.get("action"), label="legacy action manifest", error=ActionStoreCorruptError)
        source_digest = action.get("source_digest")
        backup_digest = action.get("backup_digest")
        if source_digest is None:
            expected: list[dict[str, Any]] = []
        else:
            if not isinstance(source_digest, str) or backup_digest != source_digest:
                raise ActionStoreCorruptError("legacy action manifest does not bind its backup")
            data = self._verify_legacy_backup(held, source_digest)
            if self._legacy_digest(data) != source_digest:
                raise ActionStoreCorruptError("legacy action backup digest changed during reconstruction")
            try:
                parsed = self._parse_legacy_actions(data)
                expected = self._legacy_operations(
                    parsed,
                    conversion_identity=conversion_identity,
                    source_digest=source_digest,
                )
            except (KeyError, TypeError, ValueError, ActionStoreLegacyConversionError) as exc:
                raise ActionStoreCorruptError("legacy action backup cannot reconstruct conversion operations") from exc
        if _canonical_json(action.get("operations")) != _canonical_json(expected):
            raise ActionStoreCorruptError("legacy conversion manifest operations do not match the immutable backup")
        return expected

    def _require_legacy_operation_receipts(self, held: _HeldStore, operations: list[dict[str, Any]]) -> None:
        """Require durable, ordered v2 evidence before a legacy source can retire."""
        state = self._read_state(held)
        self._verify_journal_durable(held)
        if state.snapshot is not None:
            self._verify_snapshot_durable(held)
        by_id = {receipt.action_id: receipt for receipt in state.replay.receipts}
        prior_sequence: int | None = None
        generation: int | None = None
        for operation in operations:
            action_id = self._validate_action_id(operation.get("action_id"))
            request = _copy_json_mapping(operation.get("request"), label="reconstructed legacy action")
            receipt = by_id.get(action_id)
            if receipt is None:
                raise ActionStoreCorruptError("legacy conversion is missing a durable action receipt")
            if receipt.request_digest != _digest_request(request) or receipt.store_namespace != self.namespace:
                raise ActionStoreCorruptError("legacy conversion action receipt conflicts with reconstructed request")
            if prior_sequence is not None and receipt.sequence != prior_sequence + 1:
                raise ActionStoreCorruptError("legacy conversion action receipts are not sequence-compatible")
            if generation is not None and receipt.journal_generation != generation:
                raise ActionStoreCorruptError("legacy conversion action receipts span journal generations")
            prior_sequence = receipt.sequence
            generation = receipt.journal_generation

    def _append_legacy_operations(self, held: _HeldStore, operations: list[dict[str, Any]]) -> None:
        state = self._read_state(held)
        if state.cleanup_required:
            self._finish_cleanup(held, state)
            state = self._read_state(held)
        for operation in operations:
            action_id = self._validate_action_id(operation.get("action_id"))
            request = _copy_json_mapping(operation.get("request"), label="legacy converted request")
            digest = _digest_request(request)
            existing = next((item for item in state.replay.receipts if item.action_id == action_id), None)
            if existing is not None:
                if existing.request_digest != digest:
                    raise ActionStoreConflictError("legacy action id conflicts with an existing canonical request")
                continue
            sequence = state.replay.receipts[-1].sequence + 1 if state.replay.receipts else 1
            receipt = ActionReceipt(self.namespace, action_id, digest, sequence, state.journal.generation)
            self._append_journal(held, ActionRequest(receipt, request))
            state = self._read_state(held)

    def _legacy_conversion_evidence(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Return the small record Package C must carry across candidate changes.

        This record deliberately binds source and backup evidence only.  It
        neither describes cron operations nor authorizes applying them.
        """
        action = _copy_json_mapping(manifest.get("action"), label="legacy action manifest", error=ActionStoreCorruptError)
        jobs = _copy_json_mapping(manifest.get("jobs"), label="legacy jobs manifest", error=ActionStoreCorruptError)
        return {
            "schema_version": _LEGACY_CANDIDATE_EVIDENCE_VERSION,
            "conversion_identity": manifest["conversion_identity"],
            "identity_action_digest": manifest["identity_action_digest"],
            "identity_jobs_digest": manifest["identity_jobs_digest"],
            "action_backup_digest": action["backup_digest"],
            "jobs_backup_digest": jobs["backup_digest"],
        }

    def _require_conversion_candidate_evidence(
        self, candidate: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> None:
        """Verify only the stable conversion seam, leaving candidate semantics opaque."""
        if _LEGACY_CANDIDATE_EVIDENCE_KEY in candidate:
            carried = self._copy_legacy_conversion_evidence(
                candidate[_LEGACY_CANDIDATE_EVIDENCE_KEY],
                label="legacy conversion candidate evidence",
            )
            if _canonical_json(carried) != _canonical_json(expected):
                raise ActionStoreCorruptError("legacy conversion candidate evidence does not match the manifest")
            return

        # Package B emitted this representation before the narrow evidence
        # record existed.  It remains readable only when its two existing
        # source-proof fields derive the same evidence.  Package C must carry
        # the record above once it changes the candidate kind or shape.
        if (
            candidate.get("kind") == "legacy_cron_snapshot_candidate"
            and type(candidate.get("schema_version")) is int
            and candidate["schema_version"] == _LEGACY_CONVERSION_VERSION
            and candidate.get("conversion_identity") == expected["conversion_identity"]
            and candidate.get("jobs_source_digest") == expected["jobs_backup_digest"]
        ):
            return
        raise ActionStoreCorruptError("legacy conversion candidate lacks matching evidence")

    def _verify_completed_conversion_evidence(
        self, held: _HeldStore, replay: ActionStoreReplay
    ) -> dict[str, Any] | None:
        """Fence all conversion state from normal action-store operations."""
        candidate_marks_conversion = replay.candidate is not None and (
            _LEGACY_CANDIDATE_EVIDENCE_KEY in replay.candidate
            or replay.candidate.get("kind") == "legacy_cron_snapshot_candidate"
        )
        legacy_receipt_evidence = any(
            action.request.get("kind") == "legacy_cron_action_v1" for action in replay.pending_actions
        )
        action_bytes = self._read_legacy_source(held, "action.jsonl")
        try:
            completion = self._read_conversion_completion(held)
        except FileNotFoundError:
            completion = None
        try:
            manifest = self._read_conversion_manifest(held)
        except FileNotFoundError:
            if action_bytes is not None:
                if completion is not None or candidate_marks_conversion or legacy_receipt_evidence or self._has_legacy_backup_evidence(held):
                    raise ActionStoreCorruptError(
                        "legacy action journal reappeared after conversion; mixed writers are fenced"
                    )
                raise ActionStoreConversionRequiredError(
                    "legacy action.jsonl detected; offline conversion is required before v2 action-store use"
                )
            if completion is not None or candidate_marks_conversion or legacy_receipt_evidence or self._has_legacy_backup_evidence(held):
                raise ActionStoreCorruptError("legacy conversion provenance is missing its manifest")
            return None

        action = _copy_json_mapping(manifest["action"], label="legacy action manifest", error=ActionStoreCorruptError)
        jobs = _copy_json_mapping(manifest["jobs"], label="legacy jobs manifest", error=ActionStoreCorruptError)
        if action_bytes is not None:
            if action["status"] not in {"absent", "converted"} or jobs["status"] == "pending":
                raise ActionStoreCorruptError("legacy conversion is incomplete; only the explicit converter may resume")
            raise ActionStoreCorruptError("legacy action journal reappeared after conversion; mixed writers are fenced")
        if action["status"] not in {"absent", "converted"} or jobs["status"] == "pending":
            raise ActionStoreCorruptError("legacy conversion is incomplete; only the explicit converter may resume")
        if completion is None:
            raise ActionStoreCorruptError("legacy conversion completion provenance is missing")

        jobs_bytes = self._read_legacy_source(held, "jobs.json")
        jobs_digest = None if jobs_bytes is None else self._legacy_digest(jobs_bytes)
        identity = self._verified_conversion_identity(
            held, manifest, current_jobs_digest=jobs_digest
        )
        operations = self._reconstruct_manifest_operations(held, manifest, conversion_identity=identity)
        self._require_legacy_operation_receipts(held, operations)

        jobs_status = jobs["status"]
        if jobs_status == "absent":
            if jobs_bytes is not None:
                raise ActionStoreCorruptError("legacy jobs snapshot reappeared after conversion")
        elif jobs_status in {"unavailable", "converted"}:
            if jobs_bytes is None:
                raise ActionStoreCorruptError("legacy jobs snapshot disappeared after conversion")
            if jobs_digest != jobs["source_digest"]:
                raise ActionStoreCorruptError("legacy jobs snapshot changed after conversion; mixed writers are fenced")
        else:  # _read_conversion_manifest already bounds the possible values.
            raise ActionStoreCorruptError("legacy jobs conversion status is invalid")

        evidence = self._legacy_conversion_evidence(manifest)
        expected_completion = self._completion_provenance(manifest)
        if _canonical_json(completion) != _canonical_json(expected_completion):
            raise ActionStoreCorruptError("legacy conversion completion provenance does not match the manifest")
        if jobs_status == "converted":
            if replay.candidate is None:
                raise ActionStoreCorruptError("legacy converted jobs snapshot is missing its candidate")
            self._require_conversion_candidate_evidence(replay.candidate, evidence)
        elif replay.candidate is not None:
            # A later adapter may checkpoint a representation after an
            # unavailable/absent source only if it carries the same evidence.
            self._require_conversion_candidate_evidence(replay.candidate, evidence)
        return evidence

    def _legacy_jobs_candidate(
        self,
        *,
        conversion_identity: str,
        jobs_digest: str,
        jobs: CronStore,
        operations: list[dict[str, Any]],
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        ids: set[str] = set()
        canonical_jobs: list[dict[str, Any]] = []
        for job in jobs.jobs:
            canonical = self._canonical_legacy_job(job.to_store_dict(), conversion_identity)
            if canonical["id"] in ids:
                raise ActionStoreLegacyConversionError("legacy jobs snapshot has duplicate job ids")
            ids.add(canonical["id"])
            canonical_jobs.append(canonical)
        return _copy_json_mapping(
            {
                "schema_version": 1,
                "kind": "legacy_cron_snapshot_candidate",
                "conversion_identity": conversion_identity,
                "jobs_source_digest": jobs_digest,
                _LEGACY_CANDIDATE_EVIDENCE_KEY: _copy_json_mapping(
                    evidence,
                    label="legacy conversion candidate evidence",
                ),
                "jobs": canonical_jobs,
                "operations": operations,
                "runtime_state_policy": "snapshot_wins",
            },
            label="legacy jobs snapshot candidate",
        )

    def _checkpoint_legacy_snapshot_candidate(self, candidate: Mapping[str, Any]) -> None:
        """Checkpoint the converter's pending candidate without opening public use.

        The converter is the sole allowed resumer while its manifest is
        incomplete.  This keeps normal replay/append/checkpoint calls fenced
        without weakening their public boundary.
        """
        canonical_candidate = _copy_json_mapping(candidate, label="legacy jobs snapshot candidate")
        with self._held_lock() as held:
            manifest = self._read_conversion_manifest(held)
            action = manifest["action"]
            jobs = manifest["jobs"]
            if action["status"] not in {"absent", "converted"} or jobs["status"] != "pending":
                raise ActionStoreCorruptError("legacy conversion is not ready for its snapshot checkpoint")
            state = self._read_state(held)
            self._verify_journal_durable(held)
            if state.snapshot is not None:
                self._verify_snapshot_durable(held)
            if state.cleanup_required:
                self._finish_cleanup(held, state)
                state = self._read_state(held)
            if state.replay.candidate is not None:
                if _canonical_json(state.replay.candidate) != _canonical_json(canonical_candidate):
                    raise ActionStoreConflictError("existing v2 checkpoint conflicts with legacy snapshot conversion")
                return
            snapshot_data = self._snapshot_wire(state, canonical_candidate)
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

    def convert_legacy_sources(self) -> LegacyConversionResult:
        """Run the one-way stopped-writer conversion without service wiring."""
        jobs_for_snapshot: CronStore | None = None
        snapshot_candidate: dict[str, Any] | None = None
        manifest: _ConversionManifest | None
        with self._held_lock() as held:
            try:
                manifest = self._read_conversion_manifest(held)
            except FileNotFoundError:
                manifest = None
            try:
                completion = self._read_conversion_completion(held)
            except FileNotFoundError:
                completion = None
            action_bytes = self._read_legacy_source(held, "action.jsonl")
            action_digest = None if action_bytes is None else self._legacy_digest(action_bytes)
            parsed_actions = [] if action_bytes is None else self._parse_legacy_actions(action_bytes)
            initial_jobs_bytes = self._read_legacy_source(held, "jobs.json")
            initial_jobs_digest = None if initial_jobs_bytes is None else self._legacy_digest(initial_jobs_bytes)

            if manifest is None:
                if completion is not None:
                    raise ActionStoreCorruptError("legacy conversion completion provenance is missing its manifest")
                identity = self._conversion_identity(action_digest, initial_jobs_digest)
                operations = [] if action_digest is None else self._legacy_operations(
                    parsed_actions, conversion_identity=identity, source_digest=action_digest
                )
                manifest = {
                    "schema_version": _LEGACY_CONVERSION_VERSION,
                    "kind": "legacy_cron_conversion",
                    "store_namespace": self.namespace,
                    "conversion_identity": identity,
                    "identity_action_digest": action_digest,
                    "identity_jobs_digest": initial_jobs_digest,
                    "action": {
                        "status": "absent" if action_digest is None else "pending",
                        "source_digest": action_digest,
                        "backup_digest": None,
                        "operations": operations,
                    },
                    # ``pending`` means that this is the first conversion
                    # pass and jobs.json has not yet been inspected.  It is
                    # distinct from ``absent``, which fences a jobs.json
                    # created after an already-completed conversion.
                    "jobs": {"status": "pending", "source_digest": None, "backup_digest": None},
                }
            else:
                if completion is not None and (
                    manifest["action"]["status"] not in {"absent", "converted"}
                    or manifest["jobs"]["status"] not in {"absent", "unavailable", "converted"}
                ):
                    raise ActionStoreCorruptError("legacy conversion completion provenance conflicts with an incomplete manifest")
                identity = self._verified_conversion_identity(
                    held, manifest, current_jobs_digest=initial_jobs_digest
                )
                operations = self._reconstruct_manifest_operations(
                    held, manifest, conversion_identity=identity
                )
                if completion is not None and _canonical_json(completion) != _canonical_json(
                    self._completion_provenance(manifest)
                ):
                    raise ActionStoreCorruptError("legacy conversion completion provenance does not match the manifest")
                action_state = manifest["action"]
                if action_bytes is not None:
                    if action_state["status"] == "converted":
                        raise ActionStoreCorruptError("legacy action journal reappeared after conversion")
                    if action_digest != action_state["source_digest"]:
                        raise ActionStoreCorruptError("legacy action journal changed while conversion was pending")
                elif action_state["status"] == "pending":
                    raise ActionStoreCorruptError("legacy action journal disappeared while conversion was pending")

            action_state = manifest["action"]
            if action_bytes is not None and action_state["status"] == "pending":
                action_state["backup_digest"] = self._backup_legacy_bytes(held, action_bytes)
                self._write_conversion_manifest(held, manifest)
                self._append_legacy_operations(held, operations)
                self._require_legacy_operation_receipts(held, operations)
                # Re-read the source through the held descriptor before its
                # destructive transition; a stopped writer is required.
                current = self._read_legacy_source(held, "action.jsonl")
                if current != action_bytes:
                    raise ActionStoreCorruptError("legacy action journal changed during conversion")
                # Make source retirement resumable.  A failure after the
                # unlink but before the manifest update must not look like a
                # disappearing mixed writer on retry, while a source that
                # reappears after a completed retirement remains corruption.
                action_state["status"] = "removing"
                self._write_conversion_manifest(held, manifest)

            if action_state["status"] == "removing":
                self._require_legacy_operation_receipts(held, operations)
                if action_bytes is not None:
                    current = self._read_legacy_source(held, "action.jsonl")
                    if current != action_bytes:
                        raise ActionStoreCorruptError("legacy action journal changed during conversion")
                    os.unlink("action.jsonl", dir_fd=held.root_fd)
                    held.assert_bound()
                    os.fsync(held.root_fd)
                    held.assert_bound()
                action_state["status"] = "converted"
                self._write_conversion_manifest(held, manifest)

            if action_state["status"] == "converted":
                self._require_legacy_operation_receipts(held, operations)

            jobs_bytes = self._read_legacy_source(held, "jobs.json")
            jobs_state = manifest["jobs"]
            jobs_digest = None if jobs_bytes is None else self._legacy_digest(jobs_bytes)
            if manifest["jobs"]["source_digest"] is None and jobs_digest != initial_jobs_digest:
                raise ActionStoreCorruptError("legacy jobs snapshot changed during conversion preflight")
            if jobs_state["status"] in {"converted", "pending", "absent"} and jobs_state["source_digest"] is not None:
                if jobs_digest != jobs_state["source_digest"]:
                    raise ActionStoreCorruptError("legacy jobs snapshot changed after conversion; mixed writers are fenced")
            if jobs_bytes is not None and jobs_state["status"] == "absent":
                raise ActionStoreCorruptError("legacy jobs snapshot reappeared after conversion")

            if jobs_bytes is not None and jobs_state["status"] != "converted":
                if jobs_state["status"] == "unavailable" and jobs_digest != jobs_state["source_digest"]:
                    # A restored stopped-writer snapshot starts a new explicit
                    # recovery pass.  Retire the prior terminal provenance
                    # before marking this manifest pending, so public callers
                    # remain fenced if recovery is interrupted.
                    self._remove_conversion_completion(held)
                    completion = None
                jobs_state["source_digest"] = jobs_digest
                jobs_state["backup_digest"] = self._backup_legacy_bytes(held, jobs_bytes)
                try:
                    raw_jobs = _strict_json_loads(jobs_bytes, label="legacy jobs snapshot")
                    if not isinstance(raw_jobs, dict):
                        raise ValueError("jobs snapshot must be an object")
                    jobs_for_snapshot = self._strict_legacy_store(cast(dict[object, object], raw_jobs))
                except (ActionStoreCorruptError, ActionStoreLegacyConversionError, AttributeError, KeyError, TypeError, ValueError):
                    jobs_state["status"] = "unavailable"
                    self._write_conversion_manifest(held, manifest)
                    self._write_conversion_completion(
                        held, manifest, current_jobs_digest=jobs_digest
                    )
                    return LegacyConversionResult(
                        identity,
                        len(operations),
                        action_state["status"] in {"absent", "converted"},
                        False,
                        "unavailable",
                        action_state["backup_digest"],
                        jobs_state["backup_digest"],
                    )
                jobs_state["status"] = "pending"
                snapshot_candidate = self._legacy_jobs_candidate(
                    conversion_identity=identity,
                    # This branch has jobs_bytes; its digest was computed above.
                    jobs_digest=cast(str, jobs_digest),
                    jobs=jobs_for_snapshot,
                    operations=operations,
                    evidence=self._legacy_conversion_evidence(manifest),
                )
                self._write_conversion_manifest(held, manifest)
            elif jobs_state["status"] == "converted":
                self._write_conversion_completion(held, manifest, current_jobs_digest=jobs_digest)
                return LegacyConversionResult(
                    identity,
                    len(operations),
                    action_state["status"] in {"absent", "converted"},
                    True,
                    "converted",
                    action_state["backup_digest"],
                    jobs_state["backup_digest"],
                )
            elif jobs_bytes is None:
                jobs_state["status"] = "absent"
                self._write_conversion_manifest(held, manifest)
                self._write_conversion_completion(held, manifest, current_jobs_digest=None)
                return LegacyConversionResult(
                    identity,
                    len(operations),
                    action_state["status"] in {"absent", "converted"},
                    False,
                    "absent",
                    action_state["backup_digest"],
                    jobs_state["backup_digest"],
                )

        if snapshot_candidate is None:
            raise ActionStoreLegacyConversionError("legacy jobs snapshot conversion did not produce a candidate")
        self._checkpoint_legacy_snapshot_candidate(snapshot_candidate)
        with self._held_lock() as held:
            manifest = self._read_conversion_manifest(held)
            current_jobs = self._read_legacy_source(held, "jobs.json")
            identity = self._verified_conversion_identity(
                held,
                manifest,
                current_jobs_digest=self._legacy_digest(current_jobs) if current_jobs is not None else None,
            )
            operations = self._reconstruct_manifest_operations(held, manifest, conversion_identity=identity)
            self._require_legacy_operation_receipts(held, operations)
            if current_jobs is None or self._legacy_digest(current_jobs) != manifest["jobs"]["source_digest"]:
                raise ActionStoreCorruptError("legacy jobs snapshot changed during checkpoint conversion")
            manifest["jobs"]["status"] = "converted"
            self._write_conversion_manifest(held, manifest)
            self._write_conversion_completion(
                held,
                manifest,
                current_jobs_digest=self._legacy_digest(current_jobs),
            )
            return LegacyConversionResult(
                identity,
                len(operations),
                manifest["action"]["status"] in {"absent", "converted"},
                True,
                "converted",
                manifest["action"]["backup_digest"],
                manifest["jobs"]["backup_digest"],
            )

    def replay(self) -> ActionStoreReplay:
        """Return a detached generic candidate and its ordered pending actions."""
        with self._held_lock() as held:
            state = self._read_state(held)
            self._verify_completed_conversion_evidence(held, state.replay)
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

    def append(
        self,
        *,
        action_id: str,
        request: Mapping[str, Any],
        expected_generation: ActionStoreGeneration | None = None,
    ) -> ActionReceipt:
        """Durably append one opaque request or return its exact old receipt."""
        action_id = self._validate_action_id(action_id)
        canonical_request = _copy_json_mapping(request, label="action request")
        digest = _digest_request(canonical_request)
        with self._held_lock() as held:
            state = self._read_state(held)
            self._verify_completed_conversion_evidence(held, state.replay)
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
            if expected_generation is not None:
                if not isinstance(cast(object, expected_generation), ActionStoreGeneration):
                    raise ValueError("expected_generation must be an ActionStoreGeneration returned by replay()")
                expected_generation = self._validate_generation(expected_generation)
                if state.replay.generation != expected_generation:
                    raise ActionStoreStaleGenerationError("append generation is stale")
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
        if not isinstance(cast(object, expected_generation), ActionStoreGeneration):
            raise ValueError("expected_generation must be an ActionStoreGeneration returned by replay()")
        expected_generation = self._validate_generation(expected_generation)
        canonical_candidate = _copy_json_mapping(candidate, label="checkpoint candidate")
        with self._held_lock() as held:
            state = self._read_state(held)
            conversion_evidence = self._verify_completed_conversion_evidence(held, state.replay)
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
            if conversion_evidence is not None:
                self._require_conversion_candidate_evidence(canonical_candidate, conversion_evidence)
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
