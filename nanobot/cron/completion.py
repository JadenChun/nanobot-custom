"""Durable completion and history projection for admitted cron occurrences.

The occurrence ledger remains the computation authority.  This module first
persists a canonical completion through that ledger, then projects the exact
occurrence/run identity into a separate private run-record directory.  The
projection never updates a definition candidate and its delivery state is
explicitly fenced for Package E.
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

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - construction fails closed on Windows
    _fcntl = None

from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceClaim,
    OccurrenceNotFoundError,
    OccurrenceReconciliationRequiredError,
    OccurrenceRecord,
)

_SCHEMA_VERSION = 1
_PROJECTION_KIND = "cron_run_projection_v1"
_MAX_PROJECTION_BYTES = 1_048_576
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,511}")
_STATUS_VALUES = frozenset({"ok", "error", "skipped"})
_PROVENANCE_KEY = "provenance"
_MISSING = object()
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class CronCompletionError(RuntimeError):
    """Base error for completion projection."""


class CronCompletionConflictError(CronCompletionError):
    """A completion or projection conflicts with its exact run identity."""


class CronCompletionIncompleteError(CronCompletionError):
    """A claimed occurrence has no durable canonical completion yet."""


class CronCompletionPersistenceError(CronCompletionError):
    """A projection durability boundary did not complete."""


@dataclass(frozen=True)
class CronCompletionProjection:
    """Detached durable history projection; delivery remains fenced."""

    run_id: str
    occurrence_id: str
    job_id: str
    job_revision: str | int
    generation: int
    status: str
    completion: dict[str, Any]
    provenance: dict[str, Any]
    delivery_state: str = "fenced"


@dataclass(frozen=True)
class _RootBinding:
    """Held no-follow descriptor chain for the configured root pathname."""

    store: CronCompletionProjector
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
class _HeldProjection:
    """Descriptor-bound root and lock for one projection operation."""

    store: CronCompletionProjector
    root_binding: _RootBinding
    lock_fd: int
    root_identity: tuple[int, int]
    lock_identity: tuple[int, int]

    @property
    def root_fd(self) -> int:
        return self.root_binding.root_fd

    def assert_bound(self) -> None:
        self.store._assert_binding(self)


def _canonical_json(value: object, *, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise CronCompletionConflictError(f"{label} must contain finite JSON data") from exc


def _copy_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CronCompletionConflictError(f"{label} must be a JSON object")
    try:
        copied = json.loads(_canonical_json(dict(value), label=label))
    except (TypeError, ValueError, RecursionError, json.JSONDecodeError) as exc:
        raise CronCompletionConflictError(f"{label} must be a JSON object") from exc
    if not isinstance(copied, dict):  # pragma: no cover - mapping encodes as object
        raise CronCompletionConflictError(f"{label} must be a JSON object")
    return copied


def _strict_json_object(value: object, *, label: str) -> dict[str, Any]:
    """Copy a finite JSON object and reject non-string keys at every depth."""
    if not isinstance(value, Mapping):
        raise CronCompletionConflictError(f"{label} must be a JSON object")

    def check_keys(item: object) -> None:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise CronCompletionConflictError(f"{label} must be a JSON object")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)

    check_keys(value)
    return _copy_mapping(value, label=label)


def _integrity(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(body), label="projection").encode("utf-8")).hexdigest()


def _exact_equal(actual: object, expected: object) -> bool:
    """Compare identity values without accepting bool/number coercion."""
    return type(actual) is type(expected) and actual == expected


def run_id_for_claim(claim: OccurrenceClaim) -> str:
    """Derive one stable run identity for one occurrence generation."""
    if not isinstance(claim, OccurrenceClaim):
        raise CronCompletionConflictError("completion requires an occurrence claim")
    return claim.run_id


class CronCompletionProjector:
    """Persist canonical occurrence completion and a detached run projection."""

    def __init__(self, occurrence_store: CronOccurrenceStore, root: Path):
        if not isinstance(occurrence_store, CronOccurrenceStore):
            raise TypeError("occurrence_store must be a CronOccurrenceStore")
        root = Path(root)
        if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
            raise ValueError("completion root must be an absolute non-root path without '..'")
        if (
            _fcntl is None
            or os.name != "posix"
            or not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
            or not {os.open, os.mkdir, os.stat, os.rename, os.unlink} <= os.supports_dir_fd
        ):
            raise CronCompletionPersistenceError("completion projection requires qualified POSIX flock support")
        self.occurrence_store = occurrence_store
        self.root = root
        self._root_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None
        with self._root_fd(create=True) as root_fd:
            self._root_identity = self._inode(os.fstat(root_fd))

    @staticmethod
    def _inode(info: os.stat_result) -> tuple[int, int]:
        return info.st_dev, info.st_ino

    @staticmethod
    def _private(info: os.stat_result, *, directory: bool, links: int = 1) -> None:
        expected = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not expected(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or (not directory and info.st_nlink != links)
        ):
            raise CronCompletionConflictError(
                "completion projection object is not private or has an unexpected type"
            )

    @staticmethod
    def _dir_flags() -> int:
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK

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
                                raise CronCompletionPersistenceError("completion projection root is unavailable") from None
                            try:
                                os.mkdir(part, 0o700, dir_fd=current_fd)
                            except FileExistsError:
                                pass
                            else:
                                created = True
                            child = os.open(part, self._dir_flags(), dir_fd=current_fd)
                        child_info = os.fstat(child)
                        parent_entry = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                        if not stat.S_ISDIR(child_info.st_mode) or self._inode(parent_entry) != self._inode(child_info):
                            raise CronCompletionConflictError("completion root component binding changed")
                        if created:
                            os.fchmod(child, 0o700)
                        if create:
                            os.fsync(child)
                            os.fsync(current_fd)
                            rebound = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                            if self._inode(rebound) != self._inode(child_info):
                                raise CronCompletionConflictError("completion root component binding changed")
                        fds.append(child)
                        current_fd = child
                        child = None
                    finally:
                        if child is not None:
                            os.close(child)
                info = os.fstat(current_fd)
                self._private(info, directory=True)
                if self._root_identity is not None and self._inode(info) != self._root_identity:
                    raise CronCompletionConflictError("completion root binding changed")
                return _RootBinding(
                    self,
                    tuple(fds),
                    tuple(self.root.parts[1:]),
                    tuple(self._inode(os.fstat(fd)) for fd in fds),
                )
            except CronCompletionError:
                raise
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise CronCompletionConflictError("completion root must be a private directory") from exc
                raise CronCompletionPersistenceError("could not establish completion projection root") from exc
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
                        lock_fd = os.open(".lock", flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=root_fd)
                    except FileExistsError:
                        continue
                    created = True
                    break
            if lock_fd is None:
                raise CronCompletionPersistenceError("completion projection lock could not be established")
            if created:
                os.fchmod(lock_fd, 0o600)
                os.fsync(lock_fd)
                os.fsync(root_fd)
            info = os.fstat(lock_fd)
            self._private(info, directory=False)
            if self._lock_identity is not None and self._inode(info) != self._lock_identity:
                raise CronCompletionConflictError("completion projection lock binding changed")
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX)
            return lock_fd
        except CronCompletionError:
            if lock_fd is not None:
                os.close(lock_fd)
            raise
        except OSError as exc:
            if lock_fd is not None:
                os.close(lock_fd)
            if exc.errno == errno.ELOOP:
                raise CronCompletionConflictError("completion projection lock must not be a symlink") from exc
            raise CronCompletionPersistenceError("could not acquire completion projection lock") from exc

    def _assert_path_binding(self, binding: _RootBinding) -> None:
        try:
            if len(binding.fds) != len(self.root.parts) or len(binding.parts) != len(self.root.parts) - 1:
                raise CronCompletionConflictError("completion root path binding changed")
            for fd, expected in zip(binding.fds, binding.identities):
                if self._inode(os.fstat(fd)) != expected:
                    raise CronCompletionConflictError("completion root path binding changed")
            for parent_fd, child_fd, part, expected in zip(
                binding.fds,
                binding.fds[1:],
                binding.parts,
                binding.identities[1:],
            ):
                child_info = os.fstat(child_fd)
                if not stat.S_ISDIR(child_info.st_mode) or self._inode(child_info) != expected:
                    raise CronCompletionConflictError("completion root path binding changed")
                entry = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if self._inode(entry) != expected:
                    raise CronCompletionConflictError("completion root path binding changed")
            root_info = os.fstat(binding.root_fd)
            self._private(root_info, directory=True)
            if self._inode(root_info) != binding.identities[-1] or (
                self._root_identity is not None and self._inode(root_info) != self._root_identity
            ):
                raise CronCompletionConflictError("completion root binding changed")
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENOENT}:
                raise CronCompletionConflictError("completion root path binding changed")
            raise CronCompletionConflictError("completion root path binding could not be verified") from exc

    def _assert_binding(self, held: _HeldProjection) -> None:
        held.root_binding.assert_bound()
        root_info = os.fstat(held.root_fd)
        self._private(root_info, directory=True)
        if self._inode(root_info) != held.root_identity or self._inode(root_info) != self._root_identity:
            raise CronCompletionConflictError("completion projection root binding changed")
        lock_info = os.fstat(held.lock_fd)
        self._private(lock_info, directory=False)
        entry = os.stat(".lock", dir_fd=held.root_fd, follow_symlinks=False)
        if self._inode(lock_info) != held.lock_identity or self._inode(entry) != held.lock_identity:
            raise CronCompletionConflictError("completion projection lock binding changed")

    @contextmanager
    def _held_lock(self) -> Generator[_HeldProjection, None, None]:
        with self._root_binding() as root_binding:
            root_fd = root_binding.root_fd
            lock_fd = self._open_lock(root_fd)
            try:
                lock_identity = self._inode(os.fstat(lock_fd))
                if self._lock_identity is None:
                    self._lock_identity = lock_identity
                held = _HeldProjection(
                    self,
                    root_binding,
                    lock_fd,
                    self._inode(os.fstat(root_fd)),
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
    def _validate_run_id(run_id: str) -> str:
        if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
            raise CronCompletionConflictError("run_id is invalid")
        return run_id

    @staticmethod
    def _projection_name(run_id: str) -> str:
        digest = hashlib.sha256(
            b"nanobot-cron-run-projection-v1\0" + run_id.encode("utf-8")
        ).hexdigest()
        return f"{digest}.json"

    def projection_path(self, run_id: str) -> Path:
        """Return the digest-keyed path used for one run projection."""
        return self.root / self._projection_name(self._validate_run_id(run_id))

    @staticmethod
    def _validate_completion(
        value: Mapping[str, Any],
        *,
        claim: OccurrenceClaim,
        run_id: str,
    ) -> dict[str, Any]:
        completion = _copy_mapping(value, label="cron completion")
        status = completion.get("status")
        if status not in _STATUS_VALUES:
            raise CronCompletionConflictError("completion status must be ok, error, or skipped")
        expected = {
            "run_id": run_id,
            "occurrence_id": claim.occurrence_id,
            "job_id": claim.identity.job_id,
            "job_revision": claim.identity.job_revision,
            "generation": claim.generation,
        }
        for key, expected_value in expected.items():
            if key in completion and not _exact_equal(completion[key], expected_value):
                raise CronCompletionConflictError(f"completion conflicts with exact {key}")
            completion[key] = expected_value
        if _PROVENANCE_KEY in completion:
            completion[_PROVENANCE_KEY] = CronCompletionProjector._validate_provenance(
                completion[_PROVENANCE_KEY],
                claim=claim,
                run_id=run_id,
            )
        return completion

    @staticmethod
    def _validate_provenance(
        value: object,
        *,
        claim: OccurrenceClaim,
        run_id: str,
    ) -> dict[str, Any]:
        provenance = _strict_json_object(value, label="cron completion provenance")
        expected = {
            "run_id": run_id,
            "occurrence_id": claim.occurrence_id,
            "job_id": claim.identity.job_id,
            "job_revision": claim.identity.job_revision,
            "generation": claim.generation,
        }
        for key, expected_value in expected.items():
            if key in provenance and not _exact_equal(provenance[key], expected_value):
                raise CronCompletionConflictError(f"provenance conflicts with exact {key}")
        return provenance

    @staticmethod
    def _wire_projection(
        *,
        claim: OccurrenceClaim,
        completion: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> bytes:
        run_id = run_id_for_claim(claim)
        body: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "kind": _PROJECTION_KIND,
            "run_id": run_id,
            "occurrence_id": claim.occurrence_id,
            "job_id": claim.identity.job_id,
            "job_revision": claim.identity.job_revision,
            "generation": claim.generation,
            "status": completion["status"],
            "completion": _copy_mapping(completion, label="cron completion"),
            "provenance": _strict_json_object(provenance, label="cron completion provenance"),
            "delivery_state": "fenced",
        }
        body["integrity"] = _integrity(body)
        encoded = _canonical_json(body, label="cron completion projection").encode("utf-8")
        if len(encoded) > _MAX_PROJECTION_BYTES:
            raise CronCompletionConflictError("cron completion projection exceeds its size limit")
        return encoded

    @staticmethod
    def _projection_from_wire(value: object, *, run_id: str) -> CronCompletionProjection:
        if not isinstance(value, dict):
            raise CronCompletionConflictError("cron completion projection is not an object")
        fields = {
            "schema_version",
            "kind",
            "run_id",
            "occurrence_id",
            "job_id",
            "job_revision",
            "generation",
            "status",
            "completion",
            "provenance",
            "delivery_state",
            "integrity",
        }
        if set(value) != fields:
            raise CronCompletionConflictError("cron completion projection has an invalid schema")
        body = {key: item for key, item in value.items() if key != "integrity"}
        if value["schema_version"] != _SCHEMA_VERSION or value["kind"] != _PROJECTION_KIND:
            raise CronCompletionConflictError("cron completion projection has an invalid version or kind")
        if value["run_id"] != run_id or value["delivery_state"] != "fenced":
            raise CronCompletionConflictError("cron completion projection conflicts with its run identity")
        if not isinstance(value["integrity"], str) or value["integrity"] != _integrity(body):
            raise CronCompletionConflictError("cron completion projection failed its integrity check")
        if value["status"] not in _STATUS_VALUES:
            raise CronCompletionConflictError("cron completion projection has an invalid status")
        if type(value["generation"]) is not int or value["generation"] < 1:
            raise CronCompletionConflictError("cron completion projection has an invalid generation")
        completion = _copy_mapping(value["completion"], label="cron completion")
        provenance = _strict_json_object(value["provenance"], label="cron completion provenance")
        if (
            not _exact_equal(completion.get("run_id"), value["run_id"])
            or not _exact_equal(completion.get("occurrence_id"), value["occurrence_id"])
            or not _exact_equal(completion.get("job_id"), value["job_id"])
            or not _exact_equal(completion.get("job_revision"), value["job_revision"])
            or not _exact_equal(completion.get("generation"), value["generation"])
            or not _exact_equal(completion.get("status"), value["status"])
        ):
            raise CronCompletionConflictError("completion projection identity is not bound to its completion")
        if _PROVENANCE_KEY in completion and _canonical_json(
            _strict_json_object(completion[_PROVENANCE_KEY], label="cron completion provenance"),
            label="cron completion provenance",
        ) != _canonical_json(provenance, label="cron completion provenance"):
            raise CronCompletionConflictError(
                "completion projection provenance is not bound to its completion"
            )
        return CronCompletionProjection(
            run_id=value["run_id"],
            occurrence_id=value["occurrence_id"],
            job_id=value["job_id"],
            job_revision=value["job_revision"],
            generation=value["generation"],
            status=value["status"],
            completion=completion,
            provenance=provenance,
        )

    @staticmethod
    def _projection_matches(
        left: CronCompletionProjection,
        right: CronCompletionProjection,
        *,
        include_provenance: bool,
    ) -> bool:
        body_left: dict[str, Any] = {
            "run_id": left.run_id,
            "occurrence_id": left.occurrence_id,
            "job_id": left.job_id,
            "job_revision": left.job_revision,
            "generation": left.generation,
            "status": left.status,
            "completion": left.completion,
        }
        body_right: dict[str, Any] = {
            "run_id": right.run_id,
            "occurrence_id": right.occurrence_id,
            "job_id": right.job_id,
            "job_revision": right.job_revision,
            "generation": right.generation,
            "status": right.status,
            "completion": right.completion,
        }
        if include_provenance:
            body_left["provenance"] = left.provenance
            body_right["provenance"] = right.provenance
        return _canonical_json(body_left, label="cron completion projection") == _canonical_json(
            body_right,
            label="cron completion projection",
        )

    def _read_projection(
        self,
        run_id: str,
        *,
        durable: bool = False,
        held: _HeldProjection | None = None,
    ) -> CronCompletionProjection:
        if held is None:
            with self._held_lock() as owned:
                return self._read_projection(run_id, durable=durable, held=owned)
        held.assert_bound()
        name = self._projection_name(self._validate_run_id(run_id))
        fd: int | None = None
        try:
            fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=held.root_fd)
            before = os.fstat(fd)
            self._private(before, directory=False)
            before_inode = self._inode(before)
            before_signature = self._projection_signature(before)
            if before.st_size <= 0:
                raise CronCompletionConflictError("cron completion projection is empty")
            if before.st_size > _MAX_PROJECTION_BYTES:
                raise CronCompletionConflictError("cron completion projection exceeds its size limit")
            entry = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            self._private(entry, directory=False)
            if self._inode(entry) != before_inode or self._projection_signature(entry) != before_signature:
                raise CronCompletionConflictError("cron completion projection binding changed")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    raise CronCompletionConflictError("cron completion projection was truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            after = os.fstat(fd)
            if (
                self._inode(after) != before_inode
                or self._projection_signature(after) != before_signature
            ):
                raise CronCompletionConflictError("cron completion projection changed while being read")
            entry_after = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            self._private(entry_after, directory=False)
            if self._inode(entry_after) != before_inode or self._projection_signature(entry_after) != before_signature:
                raise CronCompletionConflictError("cron completion projection binding changed")
            try:
                parsed = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CronCompletionConflictError("cron completion projection is not valid JSON") from exc
            projection = self._projection_from_wire(parsed, run_id=run_id)
            if durable:
                os.fsync(fd)
                held.assert_bound()
                self._assert_projection_binding(
                    held,
                    name,
                    fd,
                    expected_inode=before_inode,
                    expected_signature=before_signature,
                )
                os.fsync(held.root_fd)
                held.assert_bound()
                self._assert_projection_binding(
                    held,
                    name,
                    fd,
                    expected_inode=before_inode,
                    expected_signature=before_signature,
                )
            return projection
        except FileNotFoundError as exc:
            raise OccurrenceNotFoundError(f"run projection {run_id} does not exist") from exc
        except (CronCompletionError, OccurrenceNotFoundError):
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise CronCompletionConflictError("cron completion projection must not be a symlink") from exc
            raise CronCompletionPersistenceError("could not read completion projection") from exc
        finally:
            if fd is not None:
                os.close(fd)

    @staticmethod
    def _projection_signature(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        """Return metadata whose change invalidates a durable projection read."""
        return (
            info.st_mode,
            info.st_uid,
            info.st_gid,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def _assert_projection_binding(
        self,
        held: _HeldProjection,
        name: str,
        fd: int,
        *,
        expected_inode: tuple[int, int],
        expected_signature: tuple[int, int, int, int, int, int, int],
    ) -> os.stat_result:
        descriptor = os.fstat(fd)
        self._private(descriptor, directory=False)
        entry = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
        self._private(entry, directory=False)
        if (
            self._inode(descriptor) != expected_inode
            or self._inode(entry) != expected_inode
            or self._projection_signature(descriptor) != expected_signature
            or self._projection_signature(entry) != expected_signature
        ):
            raise CronCompletionConflictError("cron completion projection binding changed")
        return descriptor

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            count = os.write(fd, data[offset:])
            if count <= 0:
                raise OSError("short completion projection write")
            offset += count

    def _replace_projection(
        self,
        run_id: str,
        data: bytes,
        held: _HeldProjection | None = None,
    ) -> None:
        if held is None:
            with self._held_lock() as owned:
                self._replace_projection(run_id, data, owned)
            return
        held.assert_bound()
        name = self._projection_name(self._validate_run_id(run_id))
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
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
            self._private(os.fstat(fd), directory=False)
            self._write_all(fd, data)
            os.fsync(fd)
            os.rename(temporary, name, src_dir_fd=held.root_fd, dst_dir_fd=held.root_fd)
            published = True
            staged_inode = self._inode(os.fstat(fd))
            destination = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            if self._inode(destination) != staged_inode:
                raise CronCompletionConflictError("completion projection destination binding changed")
            held.assert_bound()
            os.fsync(held.root_fd)
            held.assert_bound()
            destination = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            if self._inode(destination) != staged_inode:
                raise CronCompletionConflictError("completion projection destination binding changed")
        except CronCompletionError:
            raise
        except OSError as exc:
            raise CronCompletionPersistenceError("could not durably replace completion projection") from exc
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

    @staticmethod
    def _record_for_claim(
        record: OccurrenceRecord,
        claim: OccurrenceClaim,
    ) -> None:
        if (
            record.occurrence_id != claim.occurrence_id
            or record.identity != claim.identity
            or record.owner != claim.owner
            or record.generation != claim.generation
        ):
            raise CronCompletionConflictError("completion claim does not match exact occurrence/run identity")

    def project_completed(
        self,
        claim_or_occurrence_id: OccurrenceClaim | str,
        *,
        run_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> CronCompletionProjection:
        """Project an already completed occurrence, including after restart."""
        claim = claim_or_occurrence_id if isinstance(claim_or_occurrence_id, OccurrenceClaim) else None
        occurrence_id = claim.occurrence_id if claim is not None else claim_or_occurrence_id
        if not isinstance(occurrence_id, str) or not occurrence_id:
            raise CronCompletionConflictError("occurrence_id is invalid")
        try:
            record = self.occurrence_store.get_completed_durable(occurrence_id)
        except OccurrenceReconciliationRequiredError as exc:
            raise CronCompletionIncompleteError(
                "occurrence is still claimed; crash state is not success"
            ) from exc
        if claim is not None:
            self._record_for_claim(record, claim)
            expected_run_id = run_id_for_claim(claim)
        else:
            expected_run_id = f"cron-occurrence-{record.occurrence_id}-{record.generation}"
        if run_id is not None and self._validate_run_id(run_id) != expected_run_id:
            raise CronCompletionConflictError("completion conflicts with exact run identity")
        run_id = expected_run_id
        if record.state != "completed" or record.completion is None:
            raise CronCompletionIncompleteError("occurrence is still claimed; crash state is not success")
        completion = _copy_mapping(record.completion, label="cron completion")
        projection_claim = claim or OccurrenceClaim(
            record.occurrence_id,
            record.identity,
            record.owner,
            record.generation,
            record.job_snapshot,
            record.destination_policy,
        )
        completion = self._validate_completion(
            completion,
            claim=projection_claim,
            run_id=run_id,
        )
        canonical_provenance = (
            self._validate_provenance(
                completion[_PROVENANCE_KEY],
                claim=projection_claim,
                run_id=run_id,
            )
            if _PROVENANCE_KEY in completion
            else {}
        )
        if provenance is None:
            detached_provenance = canonical_provenance
        else:
            detached_provenance = self._validate_provenance(
                provenance,
                claim=projection_claim,
                run_id=run_id,
            )
            if _canonical_json(detached_provenance, label="cron completion provenance") != _canonical_json(
                canonical_provenance,
                label="cron completion provenance",
            ):
                raise CronCompletionConflictError(
                    "completion provenance conflicts with canonical completion"
                )
        data = self._wire_projection(
            claim=projection_claim,
            completion=completion,
            provenance=detached_provenance,
        )
        with self._held_lock() as held:
            try:
                existing = self._read_projection(run_id, held=held)
            except OccurrenceNotFoundError:
                self._replace_projection(run_id, data, held)
                return self._projection_from_wire(json.loads(data.decode("utf-8")), run_id=run_id)
            candidate = self._projection_from_wire(json.loads(data.decode("utf-8")), run_id=run_id)
            if _PROVENANCE_KEY in completion or provenance is not None:
                if not self._projection_matches(existing, candidate, include_provenance=True):
                    raise CronCompletionConflictError("run projection already has different canonical content")
            else:
                # A legacy D projection may carry historical provenance that
                # was never part of the completed occurrence. Preserve that
                # strictly validated evidence when the caller omitted an
                # explicit provenance value; it must not become E2 input.
                if not self._projection_matches(existing, candidate, include_provenance=False):
                    raise CronCompletionConflictError("run projection already has different canonical content")
            # A visible projection after a failed directory fsync is not an
            # acknowledgement.  Reprove file and directory durability.
            return self._read_projection(run_id, durable=True, held=held)

    def complete(
        self,
        claim: OccurrenceClaim,
        completion: Mapping[str, Any] | None = None,
        *,
        status: str | None = None,
        result: Any = _MISSING,
        error: str | None | object = _MISSING,
        run_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> CronCompletionProjection:
        """Persist canonical completion, then project it into run history."""
        if not isinstance(claim, OccurrenceClaim):
            raise CronCompletionConflictError("completion requires an occurrence claim")
        expected_run_id = run_id_for_claim(claim)
        if run_id is not None and self._validate_run_id(run_id) != expected_run_id:
            raise CronCompletionConflictError("completion conflicts with exact run identity")
        values = {} if completion is None else _copy_mapping(completion, label="cron completion")
        if status is not None:
            if "status" in values and values["status"] != status:
                raise CronCompletionConflictError("completion status conflicts with supplied status")
            values["status"] = status
        if result is not _MISSING:
            values["result"] = result
        if error is not _MISSING:
            values["error"] = error
        if _PROVENANCE_KEY in values:
            values[_PROVENANCE_KEY] = self._validate_provenance(
                values[_PROVENANCE_KEY],
                claim=claim,
                run_id=expected_run_id,
            )
        if provenance is not None:
            supplied_provenance = self._validate_provenance(
                provenance,
                claim=claim,
                run_id=expected_run_id,
            )
            if _PROVENANCE_KEY in values and _canonical_json(
                values[_PROVENANCE_KEY], label="cron completion provenance"
            ) != _canonical_json(supplied_provenance, label="cron completion provenance"):
                raise CronCompletionConflictError(
                    "completion provenance conflicts with reserved provenance"
                )
            values[_PROVENANCE_KEY] = supplied_provenance
        values.setdefault("status", "ok")
        canonical = self._validate_completion(values, claim=claim, run_id=expected_run_id)
        # The ledger is the source of computation truth.  If this write is
        # durable but projection fails, project_completed can safely resume;
        # a claimed record can never be projected as success.
        self.occurrence_store.complete(claim, canonical)
        return self.project_completed(claim, run_id=expected_run_id, provenance=provenance)

    complete_occurrence = complete


# A short name is useful to callers that describe the object as a run store.
CronCompletionStore = CronCompletionProjector


__all__ = [
    "CronCompletionConflictError",
    "CronCompletionError",
    "CronCompletionIncompleteError",
    "CronCompletionPersistenceError",
    "CronCompletionProjector",
    "CronCompletionProjection",
    "CronCompletionStore",
    "run_id_for_claim",
]
