"""Private, synthetic-only durable ledger for scheduled cron occurrences.

This module is intentionally not wired to the production cron scheduler. A
claim is an in-process capability: it is issued only after durable exclusive
creation and can complete only in the issuing process. The dormant one-shot
computation start gate consumes that capability locally; opening another ledger
instance reports unfinished claims for reconciliation instead of revoking a
live owner's authority.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import re
import stat
import threading
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Literal, Mapping, NoReturn, SupportsIndex, cast

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
_EXECUTION_LOCK_RE = re.compile(r"\.execution\.([0-9a-f]{64})\.lock")
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


class OccurrenceExecutionStartedError(OccurrenceReconciliationRequiredError):
    """This exact process-local claim has already admitted a computation."""


class OccurrenceExecutionBusyError(OccurrenceReconciliationRequiredError):
    """The exact occurrence is fenced by a live execution or reconciliation."""


class OccurrenceOwnershipError(OccurrenceStoreError):
    """The supplied owner/generation or local claim issuance is no longer valid."""


class OccurrenceAlreadyCompletedError(OccurrenceStoreError):
    """The occurrence already has a canonical durable completion."""


class OccurrenceNotFoundError(OccurrenceStoreError):
    """The requested occurrence does not exist in this namespace."""


_FENCE_CONSTRUCTION_TOKEN = object()
_LIVE_FENCES: weakref.WeakSet[OccurrenceExecutionFence] = weakref.WeakSet()


@dataclass(slots=True)
class _FenceLifecycle:
    active: bool = True
    started: bool = False


class OccurrenceExecutionFence:
    """Non-serializable, task-bound authority for one claimed generation."""

    __slots__ = (
        "_store", "_claim", "_occurrence_id", "_owner", "_generation",
        "_frozen_snapshot", "_frozen_policy", "_pid", "_task", "_lifecycle",
        "_root_binding", "_lock_fd", "_lock_inode", "__weakref__",
    )

    def __init__(
        self,
        token: object,
        *,
        store: CronOccurrenceStore,
        claim: OccurrenceClaim,
        task: asyncio.Task[object],
        root_binding: _RootBinding,
        lock_fd: int,
        lock_inode: tuple[int, int],
        lifecycle: _FenceLifecycle | None = None,
    ) -> None:
        if token is not _FENCE_CONSTRUCTION_TOKEN:
            raise TypeError("occurrence execution fences are issued only by their store")
        self._store = store
        self._claim = claim
        self._occurrence_id = claim.occurrence_id
        self._owner = claim.owner
        self._generation = claim.generation
        self._frozen_snapshot = _canonical_json(claim.job_snapshot)
        self._frozen_policy = _canonical_json(claim.destination_policy)
        self._pid = os.getpid()
        self._task = task
        self._lifecycle = _FenceLifecycle() if lifecycle is None else lifecycle
        self._root_binding = root_binding
        self._lock_fd = lock_fd
        self._lock_inode = lock_inode

    @property
    def active(self) -> bool:
        return self._lifecycle.active

    @property
    def started(self) -> bool:
        return self._lifecycle.started

    def __enter__(self) -> OccurrenceExecutionFence:
        self._store.assert_execution_fence_valid(self, require_started=False)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._store.release_execution_fence(self)

    def __copy__(self) -> NoReturn:
        raise TypeError("occurrence execution fences cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("occurrence execution fences cannot be copied")

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        raise TypeError("occurrence execution fences cannot be serialized")

    def issued_state_mismatch(
        self, *, store: CronOccurrenceStore, issued: _IssuedFence, claim: OccurrenceClaim | None
    ) -> bool:
        return (
            not self._lifecycle.active
            or self._store is not store
            or self._claim is not issued.claim
            or self._pid != issued.pid
            or self._task is not issued.task
            or self._root_binding is not issued.root_binding
            or self._lock_fd != issued.lock_fd
            or self._lock_inode != issued.lock_inode
            or self._occurrence_id != issued.claim.occurrence_id
            or self._owner != issued.claim.owner
            or self._generation != issued.claim.generation
            or self._frozen_snapshot != issued.frozen_snapshot
            or self._frozen_policy != issued.frozen_policy
            or self._lifecycle.started != issued.started
            or (claim is not None and issued.claim is not claim)
            or _canonical_json(issued.claim.job_snapshot) != issued.frozen_snapshot
            or _canonical_json(issued.claim.destination_policy) != issued.frozen_policy
            or self._lifecycle is not issued.lifecycle
        )

    def close_inherited_after_fork(self) -> None:
        if self._pid == os.getpid():
            return
        self._after_fork_child()

    def _after_fork_child(self) -> None:
        """Close inherited descriptions without unlocking the parent's open file description."""
        if not self._lifecycle.active:
            return
        self._lifecycle.active = False
        try:
            os.close(self._lock_fd)
        except OSError:
            pass
        self._root_binding.close()


def _close_inherited_execution_fences() -> None:
    for fence in tuple(_LIVE_FENCES):
        fence.close_inherited_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_inherited_execution_fences)


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
class TriggeredOccurrenceIdentity(ScheduledOccurrenceIdentity):
    """Disjoint execute-now identity; timer v1 remains byte-for-byte unchanged."""

    source: Literal["manual", "system"]
    sequence: int
    controller: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {**super().as_dict(), "trigger_version": 2, "source": self.source,
                "sequence": self.sequence, "controller": self.controller}


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
        self.store.assert_root_binding(self)

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
        self.store.assert_ledger_binding(self)


@dataclass
class _IssuedFence:
    """Store-private authoritative state for an issued execution fence."""

    fence: OccurrenceExecutionFence
    claim: OccurrenceClaim
    pid: int
    task: asyncio.Task[object]
    root_binding: _RootBinding
    lock_fd: int
    lock_inode: tuple[int, int]
    frozen_snapshot: str
    frozen_policy: str
    lifecycle: _FenceLifecycle
    started: bool = False


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


def _detached_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OccurrenceConflictError(f"{label} must be a JSON object")
    detached = json.loads(_canonical_json(dict(cast(Mapping[object, object], value))))
    if not isinstance(detached, dict):
        raise OccurrenceConflictError(f"{label} must be a JSON object")
    # JSON decoding constructs string-keyed objects with opaque JSON values.
    return cast(dict[str, Any], detached)


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(dict(value)))


def _occurrence_id(identity: ScheduledOccurrenceIdentity) -> str:
    payload = _canonical_json(identity.as_dict()).encode("utf-8")
    domain = b"nanobot-cron-trigger-v2\0" if isinstance(identity, TriggeredOccurrenceIdentity) else b"nanobot-cron-occurrence-v1\0"
    return hashlib.sha256(domain + payload).hexdigest()


def _integrity(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(body)).encode("utf-8")).hexdigest()


def scheduler_bridge_identity_conflicts(identity: ScheduledOccurrenceIdentity, namespace: str,
                                       bridge: Mapping[str, object]) -> bool:
    return identity.store_namespace != namespace or _occurrence_id(identity) != bridge["occurrence_id"]


class CronOccurrenceStore:
    """POSIX-qualified filesystem ledger for synthetic scheduled fires only.

    The configured root must be an absolute, non-root directory. The ledger
    walks from the filesystem anchor with no-follow directory descriptors and
    binds the final root inode for this instance. Existing roots and locks are
    validated, never chmod-repaired. Ancestors are traversed by held file
    descriptor but are not owned by this ledger; deployment selects a trusted
    ancestor path.
    """

    @staticmethod
    def acquire_scheduler_permanent_lock(permit: object, fd: int) -> None:
        from nanobot.cron.scheduler_store import require_scheduler_owner_permit

        require_scheduler_owner_permit(permit)
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def release_scheduler_permanent_lock(permit: object, fd: int) -> None:
        from nanobot.cron.scheduler_store import require_scheduler_owner_permit

        require_scheduler_owner_permit(permit)
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)

    @staticmethod
    def validate_scheduler_namespace(namespace: object) -> str:
        return CronOccurrenceStore._validate_namespace(namespace)

    @staticmethod
    def admit_existing_inspection(root: Path, *, namespace: str) -> tuple[Path, str]:
        CronOccurrenceStore._require_qualified_platform()
        if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
            raise ValueError("explicit lexical absolute non-root path required")
        return root, CronOccurrenceStore._validate_namespace(namespace)

    @staticmethod
    def inspect_existing_evidence(
        root: Path, *, namespace: str, max_records: Callable[[], int]
    ) -> dict[str, object]:
        root, namespace = CronOccurrenceStore.admit_existing_inspection(root, namespace=namespace)
        reader = CronOccurrenceStore.__new__(CronOccurrenceStore)
        reader.root, reader.namespace = root, namespace
        reader._root_identity = reader._lock_identity = None
        records: list[dict[str, object]] = []
        with reader._root_binding() as binding:
            names = sorted(os.listdir(binding.root_fd))
            if len(names) > max_records() * 2 + 1:
                raise OccurrenceCorruptError("inspection record bound exceeded")
            for name in names:
                if name == ".lock":
                    _private(os.stat(name, dir_fd=binding.root_fd, follow_symlinks=False), directory=False)
                    continue
                if _EXECUTION_LOCK_RE.fullmatch(name):
                    reader._validate_execution_lock_entry(binding, name)
                    continue
                if not name.endswith(".json") or not _OCCURRENCE_ID_RE.fullmatch(name[:-5]):
                    raise OccurrenceCorruptError("unsafe occurrence inspection entry")
                record = reader._read_record(binding, name[:-5])
                reader._assert_namespace(record)
                records.append({"occurrence_id": record.occurrence_id, "state": record.state,
                    "owner": record.owner, "generation": record.generation,
                    "identity": record.identity.as_dict()})
            binding.assert_bound()
        return {"classification": "occurrence_evidence", "records": records}

    @staticmethod
    def validate_reconciliation_request(
        *, external_evidence: str, resolution: str, expected_generation: int,
        expected_owner: str, successor_owner: str
    ) -> Literal["confirmed_not_started", "confirmed_compensated"]:
        if external_evidence != "EXTERNAL-EVIDENCE-VERIFIED" or resolution not in {"confirmed_not_started", "confirmed_compensated"}:
            raise ValueError("explicit external evidence and supported resolution required")
        supported_resolution: Literal["confirmed_not_started", "confirmed_compensated"] = (
            "confirmed_not_started" if resolution == "confirmed_not_started" else "confirmed_compensated")
        if type(expected_generation) is not int or expected_generation < 1:
            raise ValueError("exact predecessor generation required")
        for owner in (expected_owner, successor_owner):
            CronOccurrenceStore._validate_owner(owner)
        return supported_resolution

    def __init__(self, root: Path, *, namespace: str):
        self._flock = self._require_qualified_platform()
        root = Path(root)
        if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
            raise ValueError("occurrence root must be an absolute non-root path without '..'")
        self.root = root
        self.namespace = self._validate_namespace(namespace)
        self._root_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._issued_claims: dict[int, tuple[OccurrenceClaim, int]] = {}
        self._execution_guard = threading.Lock()
        self._started_executions: set[int] = set()
        self._issued_fences: dict[int, _IssuedFence] = {}
        with self._root_fd(create=True) as root_fd:
            self._root_identity = _inode(os.fstat(root_fd))
        self._validate_existing_records()

    @staticmethod
    def _require_qualified_platform():
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
        # Infer the concrete stdlib module type, which has no annotation spelling.
        return _fcntl

    @staticmethod
    def _validate_namespace(namespace: object) -> str:
        if not isinstance(namespace, str) or not namespace or len(namespace) > 512:
            raise ValueError("namespace must be a non-empty string no longer than 512 characters")
        return namespace

    @staticmethod
    def _validate_identity(
        *, namespace: str, job_id: object, job_revision: object, scheduled_fire_time_ms: object
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
    def _validate_owner(owner: object) -> str:
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
            self._flock.flock(lock_fd, self._flock.LOCK_EX)
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

    def assert_root_binding(self, binding: _RootBinding) -> None:
        self._assert_path_binding(binding)

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

    def assert_ledger_binding(self, held: _HeldLedger) -> None:
        self._assert_binding(held)

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
                    self._flock.flock(lock_fd, self._flock.LOCK_UN)
                finally:
                    os.close(lock_fd)

    @staticmethod
    def _record_name(occurrence_id: str) -> str:
        if not isinstance(cast(object, occurrence_id), str) or _OCCURRENCE_ID_RE.fullmatch(occurrence_id) is None:
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
        held: _HeldLedger | _RootBinding,
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
        self, held: _HeldLedger | _RootBinding, occurrence_id: str, *, durable: bool = False
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
    def parse_persisted_identity(value: object) -> ScheduledOccurrenceIdentity:
        return CronOccurrenceStore._identity_from_wire(value)

    @staticmethod
    def _identity_from_wire(value: object) -> ScheduledOccurrenceIdentity:
        expected = {"store_namespace", "job_id", "job_revision", "scheduled_fire_time_ms"}
        triggered = isinstance(value, dict) and "trigger_version" in value
        if triggered:
            expected |= {"trigger_version", "source", "sequence", "controller"}
        if not isinstance(value, dict) or set(cast(dict[object, object], value)) != expected:
            raise OccurrenceCorruptError("occurrence identity has an invalid schema")
        value = cast(dict[str, Any], value)  # Exact string-key field set checked above.
        try:
            timer = CronOccurrenceStore._validate_identity(
                namespace=value["store_namespace"],
                job_id=value["job_id"],
                job_revision=value["job_revision"],
                scheduled_fire_time_ms=value["scheduled_fire_time_ms"],
            )
            if not triggered:
                return timer
            source, sequence, controller = value["source"], value["sequence"], value["controller"]
            if type(value["trigger_version"]) is not int or value["trigger_version"] != 2 or source not in {"manual", "system"} or type(sequence) is not int or sequence < 1:
                raise ValueError("invalid trigger identity")
            if (source == "manual" and controller is not None) or (source == "system" and (not isinstance(controller, str) or not controller.strip() or len(controller.encode()) > 512)):
                raise ValueError("invalid trigger controller")
            return TriggeredOccurrenceIdentity(timer.store_namespace, timer.job_id, timer.job_revision, timer.scheduled_fire_time_ms, source, sequence, controller)
        except (KeyError, ValueError) as exc:
            raise OccurrenceCorruptError("occurrence identity is invalid") from exc

    @staticmethod
    def _validate_record(value: object, occurrence_id: str) -> OccurrenceRecord:
        fields = {
            "schema_version", "occurrence_id", "identity", "state", "owner", "generation",
            "job_snapshot", "destination_policy", "completion", "reconciliation", "integrity",
        }
        if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
            raise OccurrenceCorruptError(f"occurrence {occurrence_id} has an invalid schema")
        value = cast(dict[str, Any], value)  # Exact string-key field set checked above.
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
                execution_match = _EXECUTION_LOCK_RE.fullmatch(name)
                if execution_match is not None:
                    self._validate_execution_lock_entry(held.root_binding, name)
                    continue
                if _TEMP_ENTRY_RE.fullmatch(name):
                    _private(os.stat(name, dir_fd=held.root_fd, follow_symlinks=False), directory=False)
                    continue
                if not name.endswith(".json") or not _OCCURRENCE_ID_RE.fullmatch(name[:-5]):
                    raise OccurrenceCorruptError(f"unsafe entry in occurrence ledger: {name}")
                self._read_record(held, name[:-5])

    @staticmethod
    def _execution_lock_name(occurrence_id: str) -> str:
        if not isinstance(cast(object, occurrence_id), str) or _OCCURRENCE_ID_RE.fullmatch(occurrence_id) is None:
            raise ValueError("occurrence_id must be a SHA-256 hexadecimal digest")
        return f".execution.{occurrence_id}.lock"

    @staticmethod
    def _private_execution_lock(info: os.stat_result) -> None:
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
        ):
            raise OccurrenceCorruptError(
                "occurrence execution lock is not a private regular owner file with exact mode 0600"
            )

    def _assert_execution_lock_binding(
        self,
        root_binding: _RootBinding,
        name: str,
        lock_fd: int,
        lock_inode: tuple[int, int],
    ) -> None:
        root_binding.assert_bound()
        try:
            descriptor = os.fstat(lock_fd)
            entry = os.stat(name, dir_fd=root_binding.root_fd, follow_symlinks=False)
        except OSError as exc:
            raise OccurrenceCorruptError("occurrence execution lock binding could not be verified") from exc
        self._private_execution_lock(descriptor)
        self._private_execution_lock(entry)
        if _inode(descriptor) != lock_inode or _inode(entry) != lock_inode:
            raise OccurrenceCorruptError("occurrence execution lock binding changed")

    def _validate_execution_lock_entry(self, root_binding: _RootBinding, name: str) -> None:
        if _EXECUTION_LOCK_RE.fullmatch(name) is None:
            raise OccurrenceCorruptError(f"unsafe entry in occurrence ledger: {name}")
        fd: int | None = None
        try:
            fd = os.open(
                name,
                os.O_RDWR | _O_NOFOLLOW | _O_NONBLOCK,
                dir_fd=root_binding.root_fd,
            )
            lock_inode = _inode(os.fstat(fd))
            self._assert_execution_lock_binding(root_binding, name, fd, lock_inode)
        except OccurrenceStoreError:
            raise
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENXIO}:
                raise OccurrenceCorruptError(
                    "occurrence execution lock must be a private regular file, never a link or FIFO"
                ) from exc
            raise OccurrencePersistenceError("could not validate occurrence execution lock") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _acquire_execution_lock(
        self, occurrence_id: str
    ) -> tuple[_RootBinding, int, tuple[int, int]]:
        """Acquire one persistent occurrence lock without ever taking the ledger lock."""
        name = self._execution_lock_name(occurrence_id)
        root_binding = self._open_root_binding()
        lock_fd: int | None = None
        created = False
        locked = False
        granted = False
        try:
            for _ in range(8):
                try:
                    lock_fd = os.open(
                        name,
                        os.O_RDWR | _O_NOFOLLOW | _O_NONBLOCK,
                        dir_fd=root_binding.root_fd,
                    )
                    break
                except FileNotFoundError:
                    try:
                        lock_fd = os.open(
                            name,
                            os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_NONBLOCK,
                            0o600,
                            dir_fd=root_binding.root_fd,
                        )
                    except FileExistsError:
                        continue
                    created = True
                    break
            if lock_fd is None:
                raise OccurrencePersistenceError("occurrence execution lock could not be established")
            if created:
                os.fchmod(lock_fd, 0o600)
            lock_inode = _inode(os.fstat(lock_fd))
            self._assert_execution_lock_binding(root_binding, name, lock_fd, lock_inode)
            # A failed first create can leave the permanent lock pathname
            # visible without acknowledging either durability boundary. Every
            # acquisition therefore reproves file and root durability, not
            # only the call which won O_EXCL creation.
            os.fsync(lock_fd)
            self._assert_execution_lock_binding(root_binding, name, lock_fd, lock_inode)
            os.fsync(root_binding.root_fd)
            self._assert_execution_lock_binding(root_binding, name, lock_fd, lock_inode)
            try:
                self._flock.flock(lock_fd, self._flock.LOCK_EX | self._flock.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                    raise OccurrenceExecutionBusyError(
                        f"occurrence {occurrence_id} has a live execution or reconciliation fence"
                    ) from exc
                raise
            locked = True
            self._assert_execution_lock_binding(root_binding, name, lock_fd, lock_inode)
            granted = True
            return root_binding, lock_fd, lock_inode
        except OccurrenceStoreError:
            raise
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENXIO}:
                raise OccurrenceCorruptError(
                    "occurrence execution lock must be a private regular file, never a link or FIFO"
                ) from exc
            raise OccurrencePersistenceError("could not acquire occurrence execution lock") from exc
        finally:
            if not granted and lock_fd is not None:
                if locked:
                    try:
                        self._flock.flock(lock_fd, self._flock.LOCK_UN)
                    except OSError:
                        pass
                os.close(lock_fd)
            if not granted:
                root_binding.close()

    @contextmanager
    def _held_execution_lock(
        self, occurrence_id: str
    ) -> Generator[tuple[_RootBinding, int, tuple[int, int]], None, None]:
        root_binding, lock_fd, lock_inode = self._acquire_execution_lock(occurrence_id)
        error: BaseException | None = None
        try:
            yield root_binding, lock_fd, lock_inode
            self._assert_execution_lock_binding(
                root_binding,
                self._execution_lock_name(occurrence_id),
                lock_fd,
                lock_inode,
            )
        except BaseException as exc:
            error = exc
        try:
            self._flock.flock(lock_fd, self._flock.LOCK_UN)
        except OSError as exc:
            if error is None:
                error = OccurrencePersistenceError("could not release occurrence execution lock")
                error.__cause__ = exc
        finally:
            try:
                os.close(lock_fd)
            finally:
                root_binding.close()
        if error is not None:
            raise error

    def acquire_execution_fence(self, claim: OccurrenceClaim) -> OccurrenceExecutionFence:
        """Issue a non-transferable task capability while holding the occurrence lock."""
        claim = self._assert_issued_claim(claim)
        try:
            task = asyncio.current_task()
        except RuntimeError as exc:
            raise OccurrenceOwnershipError(
                "an occurrence execution fence requires an active asyncio task"
            ) from exc
        if task is None:
            raise OccurrenceOwnershipError("an occurrence execution fence requires an active asyncio task")
        root_binding, lock_fd, lock_inode = self._acquire_execution_lock(claim.occurrence_id)
        fence: OccurrenceExecutionFence | None = None
        try:
            lifecycle = _FenceLifecycle()
            fence = OccurrenceExecutionFence(
                _FENCE_CONSTRUCTION_TOKEN,
                store=self,
                claim=claim,
                task=task,
                root_binding=root_binding,
                lock_fd=lock_fd,
                lock_inode=lock_inode,
                lifecycle=lifecycle,
            )
            issued = _IssuedFence(
                fence=fence,
                claim=claim,
                pid=os.getpid(),
                task=task,
                root_binding=root_binding,
                lock_fd=lock_fd,
                lock_inode=lock_inode,
                frozen_snapshot=_canonical_json(claim.job_snapshot),
                frozen_policy=_canonical_json(claim.destination_policy),
                lifecycle=lifecycle,
            )
            with self._execution_guard:
                self._issued_fences[id(fence)] = issued
                _LIVE_FENCES.add(fence)
            return fence
        except BaseException:
            try:
                self._flock.flock(lock_fd, self._flock.LOCK_UN)
            finally:
                os.close(lock_fd)
                root_binding.close()
            raise

    def assert_execution_fence_valid(self, fence: object, *, require_started: bool) -> None:
        self._assert_execution_fence(fence, require_started=require_started)

    def _assert_execution_fence(
        self,
        fence: object,
        *,
        claim: OccurrenceClaim | None = None,
        require_started: bool,
    ) -> _IssuedFence:
        if type(fence) is not OccurrenceExecutionFence:
            raise OccurrenceOwnershipError("operation requires an exact issued occurrence execution fence")
        issued = self._issued_fences.get(id(fence))
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if (
            issued is None
            or issued.fence is not fence
            or issued.pid != os.getpid()
            or issued.task is not task
            or fence.issued_state_mismatch(store=self, issued=issued, claim=claim)
        ):
            raise OccurrenceOwnershipError("occurrence execution fence is foreign, stale, inactive, or transferred")
        if require_started and not issued.started:
            raise OccurrenceOwnershipError("occurrence execution fence has not started computation")
        self._assert_execution_lock_binding(
            issued.root_binding,
            self._execution_lock_name(issued.claim.occurrence_id),
            issued.lock_fd,
            issued.lock_inode,
        )
        return issued

    def release_execution_fence(self, fence: OccurrenceExecutionFence) -> None:
        """Revalidate and release an exact task-owned fence; persistent files remain."""
        with self._execution_guard:
            issued = self._assert_execution_fence(fence, require_started=False)
            del self._issued_fences[id(fence)]
            _LIVE_FENCES.discard(fence)
            issued.lifecycle.active = False
        error: BaseException | None = None
        try:
            self._assert_execution_lock_binding(
                issued.root_binding,
                self._execution_lock_name(issued.claim.occurrence_id),
                issued.lock_fd,
                issued.lock_inode,
            )
        except BaseException as exc:
            error = exc
        try:
            self._flock.flock(issued.lock_fd, self._flock.LOCK_UN)
        except OSError as exc:
            if error is None:
                error = OccurrencePersistenceError("could not release occurrence execution lock")
                error.__cause__ = exc
        finally:
            try:
                os.close(issued.lock_fd)
            finally:
                issued.root_binding.close()
        if error is not None:
            raise error

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

    def begin_execution(self, claim: OccurrenceClaim, fence: OccurrenceExecutionFence) -> None:
        """Consume one issued claim's computation start after durable reproof.

        A failed or cancelled model call cannot use this process-local claim a
        second time. Until canonical completion, the durable claimed record
        remains reconciliation-required; an external process/turn fence is still
        required before any production reconciliation or scheduler activation.
        """
        claim = self._assert_issued_claim(claim)
        with self._execution_guard:
            issued_fence = self._assert_execution_fence(
                fence, claim=claim, require_started=False
            )
            if id(claim) in self._started_executions:
                raise OccurrenceExecutionStartedError("occurrence computation already started")
            with self._held_lock() as held:
                record = self._read_record(held, claim.occurrence_id, durable=True)
                self._assert_namespace(record)
                if record.identity != claim.identity:
                    raise OccurrenceCorruptError("claim identity does not match its durable record")
                if record.owner != claim.owner or record.generation != claim.generation:
                    raise OccurrenceOwnershipError("claim owner or generation is stale")
                self._assert_frozen_inputs(record, claim.job_snapshot, claim.destination_policy)
                if record.state != "claimed":
                    raise OccurrenceAlreadyCompletedError("occurrence is already completed")
            self._started_executions.add(id(claim))
            issued_fence.started = True
            issued_fence.lifecycle.started = True

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
        return self._claim_identity(identity, job_snapshot=job_snapshot, destination_policy=destination_policy, owner=owner)

    def claim_triggered(
        self, identity: TriggeredOccurrenceIdentity, *, job_snapshot: Mapping[str, Any],
        destination_policy: Mapping[str, Any], owner: str,
    ) -> OccurrenceClaim:
        """Low-level structural seam; authenticated adapters own production admission."""
        if type(identity) is not TriggeredOccurrenceIdentity:
            raise ValueError("trigger requires its exact versioned identity")
        validated = self._identity_from_wire(identity.as_dict())
        if validated.store_namespace != self.namespace:
            raise ValueError("trigger namespace mismatch")
        return self._claim_identity(validated, job_snapshot=job_snapshot, destination_policy=destination_policy, owner=owner)

    def _claim_identity(
        self, identity: ScheduledOccurrenceIdentity, *, job_snapshot: Mapping[str, Any],
        destination_policy: Mapping[str, Any], owner: str,
    ) -> OccurrenceClaim:
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
                if _EXECUTION_LOCK_RE.fullmatch(name):
                    self._validate_execution_lock_entry(held.root_binding, name)
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

    def inspect_records(self) -> list[OccurrenceRecord]:
        """Read every exact timer/trigger record for scheduler crash inspection."""
        with self._held_lock() as held:
            records: list[OccurrenceRecord] = []
            for name in sorted(os.listdir(held.root_fd)):
                if name == ".lock":
                    continue
                if _EXECUTION_LOCK_RE.fullmatch(name):
                    self._validate_execution_lock_entry(held.root_binding, name)
                    continue
                if _TEMP_ENTRY_RE.fullmatch(name):
                    raise OccurrenceCorruptError("scheduler cannot admit occurrence staging residue")
                if not name.endswith(".json") or not _OCCURRENCE_ID_RE.fullmatch(name[:-5]):
                    raise OccurrenceCorruptError("unsafe occurrence entry")
                record = self._read_record(held, name[:-5], durable=True)
                self._assert_namespace(record)
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
        if not isinstance(cast(object, expected_owner), str) or not expected_owner:
            raise ValueError("expected_owner must be a non-empty string")
        if type(expected_generation) is not int:
            raise ValueError("expected_generation must be an integer")
        if resolution not in _RECONCILIATION_RESOLUTIONS:
            raise ValueError("resolution must be an explicit approved reconciliation result")
        with self._held_execution_lock(occurrence_id):
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

    def complete(
        self,
        claim: OccurrenceClaim,
        completion: Mapping[str, Any],
        fence: OccurrenceExecutionFence,
    ) -> OccurrenceRecord:
        """Persist canonical computation output before any later delivery adapter runs."""
        claim = self._assert_issued_claim(claim)
        canonical_completion = _detached_mapping(completion, label="completion")
        self._assert_execution_fence(fence, claim=claim, require_started=True)
        with self._held_lock() as held:
            self._assert_execution_fence(fence, claim=claim, require_started=True)
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
                self._assert_execution_fence(fence, claim=claim, require_started=True)
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
            self._assert_execution_fence(fence, claim=claim, require_started=True)
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
