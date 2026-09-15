"""Private, synthetic-only durable ledger for scheduled cron occurrences.

This module is intentionally not wired to cron execution yet. A claim is an
in-process capability: it is issued only after durable exclusive creation and
can complete only in the issuing process. Opening another ledger instance is
read-only for claims; it reports them for reconciliation instead of revoking a
live owner's authority.
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
from typing import Any, Generator, Literal, Mapping

try:  # Keep importable on Windows; construction then fails closed below.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised through the qualified gate
    _fcntl = None


_SCHEMA_VERSION = 1
_MAX_RECORD_BYTES = 1_048_576
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_OCCURRENCE_ID_RE = re.compile(r"[0-9a-f]{64}")
_TEMP_ENTRY_RE = re.compile(r"\.[0-9a-f]{64}\.[0-9a-f]{32}\.tmp")
_RECONCILIATION_RESOLUTIONS = frozenset({"confirmed_not_started", "confirmed_compensated"})


class OccurrenceStoreError(RuntimeError):
    """Base error for the private occurrence ledger."""


class OccurrencePersistenceError(OccurrenceStoreError):
    """A filesystem operation did not durably establish the requested state."""


class OccurrenceCorruptError(OccurrenceStoreError):
    """An on-disk object is malformed, unsafe, or conflicts with its binding."""


class OccurrenceConflictError(OccurrenceStoreError):
    """A caller reused an occurrence identity with changed admitted input."""


class OccurrenceReconciliationRequiredError(OccurrenceStoreError):
    """An existing durable claim cannot execute automatically."""


class OccurrenceOwnershipError(OccurrenceStoreError):
    """The supplied owner/generation or local claim issuance is no longer valid."""


class OccurrenceAlreadyCompletedError(OccurrenceStoreError):
    """The occurrence already has a canonical durable completion."""


class OccurrenceNotFoundError(OccurrenceStoreError):
    """The requested occurrence does not exist in this namespace."""


@dataclass(frozen=True)
class ScheduledOccurrenceIdentity:
    """The complete collision-resistant identity for a synthetic scheduled fire."""

    store_namespace: str
    job_id: str
    job_revision: str | int
    scheduled_fire_time_ms: int

    def as_dict(self) -> dict[str, object]:
        return {
            "store_namespace": self.store_namespace,
            "job_id": self.job_id,
            "job_revision": self.job_revision,
            "scheduled_fire_time_ms": self.scheduled_fire_time_ms,
        }


@dataclass(frozen=True)
class OccurrenceClaim:
    """An exact process-local authority to compute one scheduled occurrence."""

    occurrence_id: str
    identity: ScheduledOccurrenceIdentity
    owner: str
    generation: int
    job_snapshot: dict[str, Any]
    destination_policy: dict[str, Any]

    @property
    def run_id(self) -> str:
        """Return the stable run identity for this occurrence generation."""
        return f"cron-occurrence-{self.occurrence_id}-{self.generation}"


@dataclass(frozen=True)
class OccurrenceRecord:
    """A detached durable view; it never grants computation or delivery authority."""

    occurrence_id: str
    identity: ScheduledOccurrenceIdentity
    state: Literal["claimed", "completed"]
    owner: str
    generation: int
    job_snapshot: dict[str, Any]
    destination_policy: dict[str, Any]
    completion: dict[str, Any] | None
    reconciliation: dict[str, Any] | None

    @property
    def reconciliation_required(self) -> bool:
        return self.state == "claimed"


@dataclass(frozen=True)
class _RootBinding:
    """Held no-follow descriptor chain for the configured root pathname."""

    store: CronOccurrenceStore
    fds: tuple[int, ...]
    parts: tuple[str, ...]
    identities: tuple[tuple[int, int], ...]

    @property
    def root_fd(self) -> int:
        return self.fds[-1]

    def assert_bound(self) -> None:
        self.store._assert_path_binding(self)

    def close(self) -> None:
        for fd in reversed(self.fds):
            try:
                os.close(fd)
            except OSError:
                pass


@dataclass(frozen=True)
class _HeldLedger:
    """Directory and lock descriptors that remain bound for one operation."""

    store: CronOccurrenceStore
    root_binding: _RootBinding
    lock_fd: int
    root_identity: tuple[int, int]
    lock_identity: tuple[int, int]

    @property
    def root_fd(self) -> int:
        return self.root_binding.root_fd

    def assert_bound(self) -> None:
        self.store._assert_binding(self)


def new_occurrence_owner() -> str:
    """Return an opaque owner token suitable for one runtime process/turn."""
    return uuid.uuid4().hex


def _inode(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _private(info: os.stat_result, *, directory: bool, links: int = 1) -> None:
    """Reject non-private, foreign, linked, or wrong-type filesystem objects."""
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or (not directory and info.st_nlink != links)
    ):
        raise OccurrenceCorruptError("occurrence ledger object is not private or has an unexpected type")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError, RecursionError) as exc:
        raise OccurrenceConflictError("ledger values must be finite JSON data") from exc


def _detached_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OccurrenceConflictError(f"{label} must be a JSON object")
    detached = json.loads(_canonical_json(dict(value)))
    if not isinstance(detached, dict):
        raise OccurrenceConflictError(f"{label} must be a JSON object")
    return detached


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(dict(value)))


def _occurrence_id(identity: ScheduledOccurrenceIdentity) -> str:
    payload = _canonical_json(identity.as_dict()).encode("utf-8")
    return hashlib.sha256(b"nanobot-cron-occurrence-v1\0" + payload).hexdigest()


def _integrity(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(body)).encode("utf-8")).hexdigest()


class CronOccurrenceStore:
    """POSIX-qualified filesystem ledger for synthetic scheduled fires only.

    The configured root must be an absolute, non-root directory. The ledger
    walks from the filesystem anchor with no-follow directory descriptors and
    binds the final root inode for this instance. Existing roots and locks are
    validated, never chmod-repaired. Ancestors are traversed by held file
    descriptor but are not owned by this ledger; deployment selects a trusted
    ancestor path.
    """

    def __init__(self, root: Path, *, namespace: str):
        self._require_qualified_platform()
        root = Path(root)
        if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
            raise ValueError("occurrence root must be an absolute non-root path without '..'")
        self.root = root
        self.namespace = self._validate_namespace(namespace)
        self._root_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._issued_claims: dict[int, tuple[OccurrenceClaim, int]] = {}
        with self._root_fd(create=True) as root_fd:
            self._root_identity = _inode(os.fstat(root_fd))
        self._validate_existing_records()

    @staticmethod
    def _require_qualified_platform() -> None:
        required_calls = {os.open, os.mkdir, os.stat, os.unlink, os.rename}
        if (
            os.name != "posix"
            or _fcntl is None
            or not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
            or not required_calls <= os.supports_dir_fd
        ):
            raise OccurrencePersistenceError(
                "private occurrence ledger requires qualified POSIX descriptor and flock support"
            )

    @staticmethod
    def _validate_namespace(namespace: str) -> str:
        if not isinstance(namespace, str) or not namespace or len(namespace) > 512:
            raise ValueError("namespace must be a non-empty string no longer than 512 characters")
        return namespace

    @staticmethod
    def _validate_identity(
        *, namespace: str, job_id: str, job_revision: str | int, scheduled_fire_time_ms: int
    ) -> ScheduledOccurrenceIdentity:
        if not isinstance(job_id, str) or not job_id or len(job_id) > 4096:
            raise ValueError("job_id must be a non-empty string no longer than 4096 characters")
        if isinstance(job_revision, bool) or not isinstance(job_revision, (str, int)):
            raise ValueError("job_revision must be a string or integer")
        if isinstance(job_revision, str) and not job_revision:
            raise ValueError("job_revision must not be empty")
        if isinstance(scheduled_fire_time_ms, bool) or not isinstance(scheduled_fire_time_ms, int):
            raise ValueError("scheduled_fire_time_ms must be an integer")
        return ScheduledOccurrenceIdentity(namespace, job_id, job_revision, scheduled_fire_time_ms)

    @staticmethod
    def _validate_owner(owner: str) -> str:
        if not isinstance(owner, str) or not owner or len(owner) > 512:
            raise ValueError("owner must be a non-empty string no longer than 512 characters")
        return owner

    @staticmethod
    def _dir_flags() -> int:
        return os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_NONBLOCK

    def _open_root_binding(self, *, create: bool = False) -> _RootBinding:
        """Open the configured root and retain every no-follow path descriptor."""
        fds: list[int] = []
        try:
            try:
                current_fd = os.open(self.root.anchor, self._dir_flags())
                fds.append(current_fd)
                for part in self.root.parts[1:]:
                    child: int | None = None
                    created = False
                    try:
                        try:
                            child = os.open(part, self._dir_flags(), dir_fd=current_fd)
                        except FileNotFoundError:
                            if not create:
                                raise OccurrencePersistenceError("private occurrence root is unavailable") from None
                            try:
                                os.mkdir(part, 0o700, dir_fd=current_fd)
                            except FileExistsError:
                                pass
                            else:
                                created = True
                            child = os.open(part, self._dir_flags(), dir_fd=current_fd)
                        child_info = os.fstat(child)
                        parent_binding = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                        if not stat.S_ISDIR(child_info.st_mode) or _inode(parent_binding) != _inode(child_info):
                            raise OccurrenceCorruptError("occurrence root component binding changed")
                        # Constructor admission must prove every visible component
                        # and its parent entry durable, including pre-existing
                        # components. Otherwise a failed parent fsync can be
                        # silently bypassed by a later constructor.
                        if create:
                            if created:
                                os.fchmod(child, 0o700)
                            os.fsync(child)
                            os.fsync(current_fd)
                            rebound = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                            if _inode(rebound) != _inode(child_info):
                                raise OccurrenceCorruptError("occurrence root component binding changed")
                        elif created:
                            os.fchmod(child, 0o700)
                        fds.append(child)
                        current_fd = child
                        child = None
                    finally:
                        if child is not None:
                            try:
                                os.close(child)
                            except OSError:
                                pass
                info = os.fstat(current_fd)
                _private(info, directory=True)
                if self._root_identity is not None and _inode(info) != self._root_identity:
                    raise OccurrenceCorruptError("occurrence root binding changed")
                return _RootBinding(
                    self,
                    tuple(fds),
                    tuple(self.root.parts[1:]),
                    tuple(_inode(os.fstat(fd)) for fd in fds),
                )
            except OccurrenceStoreError:
                raise
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise OccurrenceCorruptError("occurrence root must be a private directory, never a link") from exc
                raise OccurrencePersistenceError("private occurrence root is unavailable") from exc
        except BaseException:
            for fd in reversed(fds):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    @contextmanager
    def _root_binding(self, *, create: bool = False) -> Generator[_RootBinding, None, None]:
        """Hold the configured root descriptor chain for one operation."""
        binding = self._open_root_binding(create=create)
        try:
            yield binding
        finally:
            binding.close()

    @contextmanager
    def _root_fd(self, *, create: bool = False) -> Generator[int, None, None]:
        """Open the configured root through held no-follow descriptors."""
        with self._root_binding(create=create) as binding:
            yield binding.root_fd

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
                        lock_fd = os.open(
                            ".lock", flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=root_fd
                        )
                    except FileExistsError:
                        continue
                    created = True
                    break
            if lock_fd is None:
                raise OccurrencePersistenceError("occurrence lock could not be established")
            if created:
                os.fchmod(lock_fd, 0o600)
                os.fsync(lock_fd)
                os.fsync(root_fd)
            _private(os.fstat(lock_fd), directory=False)
            current = _inode(os.fstat(lock_fd))
            if self._lock_identity is not None and current != self._lock_identity:
                raise OccurrenceCorruptError("occurrence lock binding changed")
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX)
            return lock_fd
        except OccurrenceStoreError:
            if lock_fd is not None:
                os.close(lock_fd)
            raise
        except OSError as exc:
            if lock_fd is not None:
                os.close(lock_fd)
            if exc.errno == errno.ELOOP:
                raise OccurrenceCorruptError("occurrence lock must not be a symlink") from exc
            raise OccurrencePersistenceError("could not acquire occurrence ledger lock") from exc

    def _assert_path_binding(self, binding: _RootBinding) -> None:
        try:
            if len(binding.fds) != len(self.root.parts) or len(binding.parts) != len(self.root.parts) - 1:
                raise OccurrenceCorruptError("occurrence root path binding changed")
            for fd, expected in zip(binding.fds, binding.identities):
                if _inode(os.fstat(fd)) != expected:
                    raise OccurrenceCorruptError("occurrence root path binding changed")
            for parent_fd, child_fd, part, expected in zip(
                binding.fds,
                binding.fds[1:],
                binding.parts,
                binding.identities[1:],
            ):
                child_info = os.fstat(child_fd)
                if not stat.S_ISDIR(child_info.st_mode) or _inode(child_info) != expected:
                    raise OccurrenceCorruptError("occurrence root path binding changed")
                entry = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if _inode(entry) != expected:
                    raise OccurrenceCorruptError("occurrence root path binding changed")
            root_info = os.fstat(binding.root_fd)
            _private(root_info, directory=True)
            if _inode(root_info) != binding.identities[-1] or (
                self._root_identity is not None and _inode(root_info) != self._root_identity
            ):
                raise OccurrenceCorruptError("occurrence root binding changed")
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENOENT}:
                raise OccurrenceCorruptError("occurrence root path binding changed") from exc
            raise OccurrenceCorruptError("occurrence root path binding could not be verified") from exc

    def _assert_binding(self, held: _HeldLedger) -> None:
        held.root_binding.assert_bound()
        root_info = os.fstat(held.root_fd)
        _private(root_info, directory=True)
        if _inode(root_info) != held.root_identity or _inode(root_info) != self._root_identity:
            raise OccurrenceCorruptError("occurrence root binding changed")
        lock_info = os.fstat(held.lock_fd)
        _private(lock_info, directory=False)
        path_lock = os.stat(".lock", dir_fd=held.root_fd, follow_symlinks=False)
        _private(path_lock, directory=False)
        if (
            _inode(lock_info) != held.lock_identity
            or _inode(path_lock) != held.lock_identity
            or self._lock_identity != held.lock_identity
        ):
            raise OccurrenceCorruptError("occurrence lock binding changed")

    @contextmanager
    def _held_lock(self) -> Generator[_HeldLedger, None, None]:
        with self._root_binding() as root_binding:
            root_fd = root_binding.root_fd
            lock_fd = self._open_lock(root_fd)
            try:
                lock_identity = _inode(os.fstat(lock_fd))
                if self._lock_identity is None:
                    self._lock_identity = lock_identity
                held = _HeldLedger(
                    self,
                    root_binding,
                    lock_fd,
                    _inode(os.fstat(root_fd)),
                    lock_identity,
                )
                held.assert_bound()
                yield held
                held.assert_bound()
            finally:
                try:
                    _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    @staticmethod
    def _record_name(occurrence_id: str) -> str:
        if not isinstance(occurrence_id, str) or _OCCURRENCE_ID_RE.fullmatch(occurrence_id) is None:
            raise ValueError("occurrence_id must be a SHA-256 hexadecimal digest")
        return occurrence_id + ".json"

    def _record_path(self, occurrence_id: str) -> Path:
        """Compatibility inspection helper; filesystem operations use ``dir_fd``."""
        return self.root / self._record_name(occurrence_id)

    @staticmethod
    def _check_record_size(data: bytes) -> None:
        if len(data) > _MAX_RECORD_BYTES:
            raise OccurrenceConflictError("encoded record exceeds the private ledger size limit")

    @staticmethod
    def _record_signature(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        """Return metadata whose change invalidates a durable record read."""
        return (
            info.st_mode,
            info.st_uid,
            info.st_gid,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def _assert_record_binding(
        self,
        held: _HeldLedger,
        name: str,
        fd: int,
        *,
        expected_inode: tuple[int, int],
        expected_signature: tuple[int, int, int, int, int, int, int],
    ) -> os.stat_result:
        descriptor = os.fstat(fd)
        _private(descriptor, directory=False)
        entry = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
        _private(entry, directory=False)
        if (
            _inode(descriptor) != expected_inode
            or _inode(entry) != expected_inode
            or self._record_signature(descriptor) != expected_signature
            or self._record_signature(entry) != expected_signature
        ):
            raise OccurrenceCorruptError("occurrence record binding changed")
        return descriptor

    def _read_record(
        self, held: _HeldLedger, occurrence_id: str, *, durable: bool = False
    ) -> OccurrenceRecord:
        held.assert_bound()
        name = self._record_name(occurrence_id)
        fd: int | None = None
        try:
            fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=held.root_fd)
            before = os.fstat(fd)
            _private(before, directory=False)
            entry = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            _private(entry, directory=False)
            before_inode = _inode(before)
            before_signature = self._record_signature(before)
            if _inode(entry) != before_inode or self._record_signature(entry) != before_signature:
                raise OccurrenceCorruptError("occurrence record binding changed")
            if before.st_size <= 0 or before.st_size > _MAX_RECORD_BYTES:
                raise OccurrenceCorruptError(f"occurrence {occurrence_id} has an unsafe size")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    raise OccurrenceCorruptError(f"occurrence {occurrence_id} was truncated during read")
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(fd)
            if _inode(after) != before_inode or self._record_signature(after) != before_signature:
                raise OccurrenceCorruptError("occurrence record changed while being read")
            entry_after = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            _private(entry_after, directory=False)
            if _inode(entry_after) != before_inode or self._record_signature(entry_after) != before_signature:
                raise OccurrenceCorruptError("occurrence record binding changed")
            try:
                parsed = json.loads(b"".join(chunks).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OccurrenceCorruptError(f"occurrence {occurrence_id} is not valid JSON") from exc
            record = self._validate_record(parsed, occurrence_id)
            if durable:
                os.fsync(fd)
                held.assert_bound()
                self._assert_record_binding(
                    held,
                    name,
                    fd,
                    expected_inode=before_inode,
                    expected_signature=before_signature,
                )
                os.fsync(held.root_fd)
                held.assert_bound()
                self._assert_record_binding(
                    held,
                    name,
                    fd,
                    expected_inode=before_inode,
                    expected_signature=before_signature,
                )
            return record
        except FileNotFoundError as exc:
            raise OccurrenceNotFoundError(f"occurrence {occurrence_id} does not exist") from exc
        except OccurrenceStoreError:
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise OccurrenceCorruptError(f"occurrence {occurrence_id} must not be a symlink") from exc
            raise OccurrencePersistenceError(f"could not read or fsync occurrence {occurrence_id}") from exc
        finally:
            if fd is not None:
                os.close(fd)

    @staticmethod
    def _identity_from_wire(value: object) -> ScheduledOccurrenceIdentity:
        expected = {"store_namespace", "job_id", "job_revision", "scheduled_fire_time_ms"}
        if not isinstance(value, dict) or set(value) != expected:
            raise OccurrenceCorruptError("occurrence identity has an invalid schema")
        try:
            return CronOccurrenceStore._validate_identity(
                namespace=value["store_namespace"],
                job_id=value["job_id"],
                job_revision=value["job_revision"],
                scheduled_fire_time_ms=value["scheduled_fire_time_ms"],
            )
        except (KeyError, ValueError) as exc:
            raise OccurrenceCorruptError("occurrence identity is invalid") from exc

    @staticmethod
    def _validate_record(value: object, occurrence_id: str) -> OccurrenceRecord:
        fields = {
            "schema_version", "occurrence_id", "identity", "state", "owner", "generation",
            "job_snapshot", "destination_policy", "completion", "reconciliation", "integrity",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise OccurrenceCorruptError(f"occurrence {occurrence_id} has an invalid schema")
        if value["schema_version"] != _SCHEMA_VERSION or value["occurrence_id"] != occurrence_id:
            raise OccurrenceCorruptError(f"occurrence {occurrence_id} has an invalid version or identifier")
        body = {key: item for key, item in value.items() if key != "integrity"}
        if not isinstance(value["integrity"], str) or value["integrity"] != _integrity(body):
            raise OccurrenceCorruptError(f"occurrence {occurrence_id} failed its integrity check")
        identity = CronOccurrenceStore._identity_from_wire(value["identity"])
        if _occurrence_id(identity) != occurrence_id:
            raise OccurrenceCorruptError(f"occurrence {occurrence_id} conflicts with its identity")
        state = value["state"]
        owner, generation = value["owner"], value["generation"]
        if state not in {"claimed", "completed"}:
            raise OccurrenceCorruptError(f"occurrence {occurrence_id} has an invalid state")
        if not isinstance(owner, str) or not owner or type(generation) is not int or generation < 1:
            raise OccurrenceCorruptError(f"occurrence {occurrence_id} has invalid ownership")
        try:
            snapshot = _detached_mapping(value["job_snapshot"], label="job snapshot")
            policy = _detached_mapping(value["destination_policy"], label="destination policy")
            completion = (
                None if value["completion"] is None
                else _detached_mapping(value["completion"], label="completion")
            )
            reconciliation = (
                None if value["reconciliation"] is None
                else _detached_mapping(value["reconciliation"], label="reconciliation")
            )
        except OccurrenceConflictError as exc:
            raise OccurrenceCorruptError(f"occurrence {occurrence_id} has invalid JSON content") from exc
        if (state == "completed") != (completion is not None):
            raise OccurrenceCorruptError(f"occurrence {occurrence_id} has invalid completion state")
        return OccurrenceRecord(
            occurrence_id, identity, state, owner, generation, snapshot, policy, completion, reconciliation
        )

    @staticmethod
    def _wire_record(
        *,
        occurrence_id: str,
        identity: ScheduledOccurrenceIdentity,
        state: Literal["claimed", "completed"],
        owner: str,
        generation: int,
        job_snapshot: Mapping[str, Any],
        destination_policy: Mapping[str, Any],
        completion: Mapping[str, Any] | None,
        reconciliation: Mapping[str, Any] | None,
    ) -> bytes:
        body: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "occurrence_id": occurrence_id,
            "identity": identity.as_dict(),
            "state": state,
            "owner": owner,
            "generation": generation,
            "job_snapshot": _copy_mapping(job_snapshot),
            "destination_policy": _copy_mapping(destination_policy),
            "completion": None if completion is None else _copy_mapping(completion),
            "reconciliation": None if reconciliation is None else _copy_mapping(reconciliation),
        }
        body["integrity"] = _integrity(body)
        encoded = _canonical_json(body).encode("utf-8")
        CronOccurrenceStore._check_record_size(encoded)
        return encoded

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("short occurrence ledger write")
            offset += written

    def _create_exclusive(self, held: _HeldLedger, occurrence_id: str, data: bytes) -> None:
        self._check_record_size(data)
        held.assert_bound()
        fd: int | None = None
        try:
            fd = os.open(
                self._record_name(occurrence_id),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_NONBLOCK,
                0o600,
                dir_fd=held.root_fd,
            )
            os.fchmod(fd, 0o600)
            _private(os.fstat(fd), directory=False)
            self._write_all(fd, data)
            os.fsync(fd)
            held.assert_bound()
            os.fsync(held.root_fd)
            held.assert_bound()
        except FileExistsError:
            raise
        except OccurrenceStoreError:
            raise
        except OSError as exc:
            raise OccurrencePersistenceError(f"could not durably claim occurrence {occurrence_id}") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _replace_record(self, held: _HeldLedger, occurrence_id: str, data: bytes) -> None:
        self._check_record_size(data)
        held.assert_bound()
        temporary = f".{occurrence_id}.{uuid.uuid4().hex}.tmp"
        fd: int | None = None
        published = False
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_NONBLOCK,
                0o600,
                dir_fd=held.root_fd,
            )
            os.fchmod(fd, 0o600)
            _private(os.fstat(fd), directory=False)
            self._write_all(fd, data)
            os.fsync(fd)
            os.close(fd)
            fd = None
            held.assert_bound()
            os.rename(
                temporary,
                self._record_name(occurrence_id),
                src_dir_fd=held.root_fd,
                dst_dir_fd=held.root_fd,
            )
            published = True
            held.assert_bound()
            os.fsync(held.root_fd)
            held.assert_bound()
        except OccurrenceStoreError:
            raise
        except OSError as exc:
            raise OccurrencePersistenceError(f"could not durably replace occurrence {occurrence_id}") from exc
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

    def _assert_namespace(self, record: OccurrenceRecord) -> None:
        if record.identity.store_namespace != self.namespace:
            raise OccurrenceConflictError("occurrence belongs to a different ledger namespace")

    def _validate_existing_records(self) -> None:
        """Validate final records without mutating an outstanding live claim."""
        with self._held_lock() as held:
            try:
                entries = os.listdir(held.root_fd)
            except OSError as exc:
                raise OccurrencePersistenceError("could not list occurrence ledger") from exc
            for name in entries:
                if name == ".lock":
                    continue
                if _TEMP_ENTRY_RE.fullmatch(name):
                    _private(os.stat(name, dir_fd=held.root_fd, follow_symlinks=False), directory=False)
                    continue
                if not name.endswith(".json") or not _OCCURRENCE_ID_RE.fullmatch(name[:-5]):
                    raise OccurrenceCorruptError(f"unsafe entry in occurrence ledger: {name}")
                self._read_record(held, name[:-5])

    @staticmethod
    def _assert_frozen_inputs(
        record: OccurrenceRecord,
        snapshot: Mapping[str, Any],
        destination_policy: Mapping[str, Any],
    ) -> None:
        if _canonical_json(record.job_snapshot) != _canonical_json(dict(snapshot)):
            raise OccurrenceConflictError("occurrence identity conflicts with its admitted job snapshot")
        if _canonical_json(record.destination_policy) != _canonical_json(dict(destination_policy)):
            raise OccurrenceConflictError("occurrence identity conflicts with its admitted destination policy")

    def _issued_claim_for_record(self, record: OccurrenceRecord) -> OccurrenceClaim | None:
        """Return the process-local capability for an exact retry, if present."""
        for issued, process_id in self._issued_claims.values():
            if (
                process_id == os.getpid()
                and issued.occurrence_id == record.occurrence_id
                and issued.owner == record.owner
                and issued.generation == record.generation
            ):
                return issued
        return None

    def _issue_claim(self, record: OccurrenceRecord) -> OccurrenceClaim:
        claim = OccurrenceClaim(
            record.occurrence_id,
            record.identity,
            record.owner,
            record.generation,
            _copy_mapping(record.job_snapshot),
            _copy_mapping(record.destination_policy),
        )
        self._issued_claims[id(claim)] = (claim, os.getpid())
        return claim

    def _assert_issued_claim(self, claim: object) -> OccurrenceClaim:
        if not isinstance(claim, OccurrenceClaim):
            raise OccurrenceOwnershipError("completion requires an issued occurrence claim")
        issued = self._issued_claims.get(id(claim))
        if issued is None or issued[0] is not claim or issued[1] != os.getpid():
            raise OccurrenceOwnershipError("claim was not issued by this store in this process")
        return claim

    def claim_scheduled(
        self,
        *,
        job_id: str,
        job_revision: str | int,
        scheduled_fire_time_ms: int,
        job_snapshot: Mapping[str, Any],
        destination_policy: Mapping[str, Any],
        owner: str,
    ) -> OccurrenceClaim:
        """Exclusively and durably claim a synthetic scheduled fire before effects."""
        identity = self._validate_identity(
            namespace=self.namespace,
            job_id=job_id,
            job_revision=job_revision,
            scheduled_fire_time_ms=scheduled_fire_time_ms,
        )
        snapshot = _detached_mapping(job_snapshot, label="job snapshot")
        policy = _detached_mapping(destination_policy, label="destination policy")
        owner = self._validate_owner(owner)
        occurrence_id = _occurrence_id(identity)
        record_data = self._wire_record(
            occurrence_id=occurrence_id,
            identity=identity,
            state="claimed",
            owner=owner,
            generation=1,
            job_snapshot=snapshot,
            destination_policy=policy,
            completion=None,
            reconciliation=None,
        )
        with self._held_lock() as held:
            try:
                existing = self._read_record(held, occurrence_id)
            except OccurrenceNotFoundError:
                try:
                    self._create_exclusive(held, occurrence_id, record_data)
                except FileExistsError:
                    existing = self._read_record(held, occurrence_id)
                else:
                    return self._issue_claim(
                        OccurrenceRecord(
                            occurrence_id, identity, "claimed", owner, 1, snapshot, policy, None, None
                        )
                    )
            self._assert_namespace(existing)
            if existing.identity != identity:
                raise OccurrenceCorruptError("occurrence digest collides with a different identity")
            self._assert_frozen_inputs(existing, snapshot, policy)
            if existing.state == "completed":
                raise OccurrenceAlreadyCompletedError("occurrence already has a durable canonical completion")
            retry = self._issued_claim_for_record(existing)
            if retry is not None and retry.owner == owner:
                # A caller which lost the acknowledgement after exclusive
                # creation may retry the exact identity in the same process.
                # Returning the originally issued object preserves the
                # process-local capability without allowing another owner to
                # execute the occurrence.
                return retry
            raise OccurrenceReconciliationRequiredError(
                "occurrence already has a durable claim; reconcile it before any retry"
            )

    def get(self, occurrence_id: str) -> OccurrenceRecord:
        """Read a detached record without granting computation or delivery authority."""
        with self._held_lock() as held:
            record = self._read_record(held, occurrence_id)
            self._assert_namespace(record)
            return record

    def get_completed_durable(self, occurrence_id: str) -> OccurrenceRecord:
        """Read one completed record only after reproving record and directory durability.

        The returned record is detached evidence.  It never grants a claim,
        computation authority, or delivery authority, and a claimed record is
        explicitly left for reconciliation.
        """
        with self._held_lock() as held:
            record = self._read_record(held, occurrence_id, durable=True)
            self._assert_namespace(record)
            if record.state != "completed" or record.completion is None:
                raise OccurrenceReconciliationRequiredError(
                    "occurrence is still claimed; reconcile it before treating it as completed"
                )
            return record

    def reconciliation_required(self) -> list[OccurrenceRecord]:
        """List claimed occurrences which a newly opened runtime must reconcile."""
        with self._held_lock() as held:
            records: list[OccurrenceRecord] = []
            try:
                names = sorted(os.listdir(held.root_fd))
            except OSError as exc:
                raise OccurrencePersistenceError("could not list occurrence ledger") from exc
            for name in names:
                if name == ".lock":
                    continue
                if _TEMP_ENTRY_RE.fullmatch(name):
                    _private(os.stat(name, dir_fd=held.root_fd, follow_symlinks=False), directory=False)
                    continue
                if not name.endswith(".json") or not _OCCURRENCE_ID_RE.fullmatch(name[:-5]):
                    raise OccurrenceCorruptError(f"unsafe entry in occurrence ledger: {name}")
                record = self._read_record(held, name[:-5])
                if record.identity.store_namespace == self.namespace and record.reconciliation_required:
                    records.append(record)
            return records

    def reclaim_after_reconciliation(
        self,
        occurrence_id: str,
        *,
        owner: str,
        expected_owner: str,
        expected_generation: int,
        resolution: Literal["confirmed_not_started", "confirmed_compensated"],
    ) -> OccurrenceClaim:
        """Issue a successor only after an externally fenced, explicit reconciliation."""
        owner = self._validate_owner(owner)
        if not isinstance(expected_owner, str) or not expected_owner:
            raise ValueError("expected_owner must be a non-empty string")
        if type(expected_generation) is not int:
            raise ValueError("expected_generation must be an integer")
        if resolution not in _RECONCILIATION_RESOLUTIONS:
            raise ValueError("resolution must be an explicit approved reconciliation result")
        with self._held_lock() as held:
            record = self._read_record(held, occurrence_id)
            self._assert_namespace(record)
            if record.state == "completed":
                raise OccurrenceAlreadyCompletedError("a completed occurrence cannot be reclaimed")
            if record.owner != expected_owner or record.generation != expected_generation:
                raise OccurrenceOwnershipError("reconciliation predecessor owner or generation is stale")
            reconciliation = {
                "resolution": resolution,
                "replaced_owner": record.owner,
                "replaced_generation": record.generation,
            }
            data = self._wire_record(
                occurrence_id=record.occurrence_id,
                identity=record.identity,
                state="claimed",
                owner=owner,
                generation=record.generation + 1,
                job_snapshot=record.job_snapshot,
                destination_policy=record.destination_policy,
                completion=None,
                reconciliation=reconciliation,
            )
            self._replace_record(held, occurrence_id, data)
            return self._issue_claim(
                OccurrenceRecord(
                    record.occurrence_id,
                    record.identity,
                    "claimed",
                    owner,
                    record.generation + 1,
                    record.job_snapshot,
                    record.destination_policy,
                    None,
                    reconciliation,
                )
            )

    def complete(self, claim: OccurrenceClaim, completion: Mapping[str, Any]) -> OccurrenceRecord:
        """Persist canonical computation output before any later delivery adapter runs."""
        claim = self._assert_issued_claim(claim)
        canonical_completion = _detached_mapping(completion, label="completion")
        with self._held_lock() as held:
            record = self._read_record(held, claim.occurrence_id)
            self._assert_namespace(record)
            if record.identity != claim.identity:
                raise OccurrenceCorruptError("claim identity does not match its durable record")
            if record.owner != claim.owner or record.generation != claim.generation:
                raise OccurrenceOwnershipError("claim owner or generation is stale")
            self._assert_frozen_inputs(record, claim.job_snapshot, claim.destination_policy)
            if record.state == "completed":
                if _canonical_json(record.completion) != _canonical_json(canonical_completion):
                    raise OccurrenceConflictError("occurrence already has a different canonical completion")
                # A visible completion after a failed directory fsync is not
                # delivery authority. Retry verifies and fsyncs file + dir.
                verified = self._read_record(held, claim.occurrence_id, durable=True)
                if (
                    verified.state != "completed"
                    or verified.owner != claim.owner
                    or verified.generation != claim.generation
                    or _canonical_json(verified.completion) != _canonical_json(canonical_completion)
                ):
                    raise OccurrenceCorruptError("completed occurrence changed during durable retry")
                return verified
            data = self._wire_record(
                occurrence_id=record.occurrence_id,
                identity=record.identity,
                state="completed",
                owner=record.owner,
                generation=record.generation,
                job_snapshot=record.job_snapshot,
                destination_policy=record.destination_policy,
                completion=canonical_completion,
                reconciliation=record.reconciliation,
            )
            self._replace_record(held, record.occurrence_id, data)
            return OccurrenceRecord(
                record.occurrence_id,
                record.identity,
                "completed",
                record.owner,
                record.generation,
                _copy_mapping(record.job_snapshot),
                _copy_mapping(record.destination_policy),
                _copy_mapping(canonical_completion),
                None if record.reconciliation is None else _copy_mapping(record.reconciliation),
            )
