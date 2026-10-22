"""Private durable observations for the dormant claimed-cron composition.

The observer receives identities and terminal classifications only.  It never
receives model content, tool arguments, delivery messages, or an authority
capable of resuming computation or delivery.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import stat
import uuid
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, Literal, TypeAlias, cast

try:
    import fcntl
except ImportError:  # pragma: no cover - construction fails closed on Windows
    fcntl = None  # type: ignore[assignment]

CallbackKind: TypeAlias = Literal["result"]
CallbackState: TypeAlias = Literal["pending", "invoking", "succeeded", "failed", "observed", "abandoned"]
Callback: TypeAlias = Callable[["CronCallbackObservation"], object | Awaitable[object]]
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,511}")


class CronCallbackError(RuntimeError):
    """Base callback-ledger error."""


class CronCallbackConflictError(CronCallbackError):
    """A callback identity or durable transition conflicts."""


class CronCallbackReconciliationRequiredError(CronCallbackError):
    """An invocation may have happened and cannot be retried automatically."""


@dataclass(frozen=True, slots=True)
class CronCallbackObservation:
    occurrence_id: str
    run_id: str
    job_id: str
    job_revision: str | int
    generation: int
    kind: CallbackKind
    result_status: str

    def __post_init__(self) -> None:
        for name in ("occurrence_id", "run_id", "job_id", "result_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
                raise CronCallbackConflictError(f"callback {name} is invalid")
        if self.kind != "result":
            raise CronCallbackConflictError("callback kind is invalid")
        if type(self.job_revision) not in (str, int) or isinstance(self.job_revision, bool):
            raise CronCallbackConflictError("callback job revision is invalid")
        if type(self.generation) is not int or self.generation < 1:
            raise CronCallbackConflictError("callback generation is invalid")


@dataclass(frozen=True, slots=True)
class CronCallbackRecord:
    callback_id: str
    observation: CronCallbackObservation
    state: CallbackState
    failure_code: str | None = None


@dataclass(frozen=True)
class _RootBinding:
    store: "CronCallbackStore"
    fds: tuple[int, ...]
    parts: tuple[str, ...]
    identities: tuple[tuple[int, int], ...]

    @property
    def root_fd(self) -> int:
        return self.fds[-1]

    def assert_bound(self) -> None:
        self.store.require_callback_root_binding(self)

    def close(self) -> None:
        for fd in reversed(self.fds):
            try:
                os.close(fd)
            except OSError:
                pass


class CronCallbackStore:
    """One durable, non-retryable callback record per occurrence and kind."""

    def __init__(self, root: Path, observer: Callback) -> None:
        if not callable(observer):
            raise TypeError("callback observer must be callable")
        root = Path(root)
        if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
            raise ValueError("callback root must be an absolute non-root path without '..'")
        if os.name != "posix" or fcntl is None:
            raise CronCallbackError("callback storage requires POSIX flock support")
        self.root = root
        self._observer = observer
        self._path_identities: tuple[tuple[int, int], ...] | None = None
        self._lock_identity: tuple[int, int] | None = None
        with self._root_binding(create=True) as binding:
            self._path_identities = binding.identities
            lock_fd = self._open_lock(binding, create=True)
            try:
                self._lock_identity = self._inode(os.fstat(lock_fd))
            finally:
                os.close(lock_fd)

    @staticmethod
    def _assert_private(info: os.stat_result, *, directory: bool) -> None:
        expected = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not expected(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or (not directory and info.st_nlink != 1)
        ):
            raise CronCallbackConflictError("callback storage must be private and owner-controlled")

    @staticmethod
    def callback_id(observation: CronCallbackObservation) -> str:
        return hashlib.sha256(
            b"nanobot-cron-callback-v1\0"
            + observation.occurrence_id.encode()
            + b"\0"
            + observation.kind.encode()
        ).hexdigest()

    @staticmethod
    def _inode(info: os.stat_result) -> tuple[int, int]:
        return info.st_dev, info.st_ino

    @staticmethod
    def _dir_flags() -> int:
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)

    def _open_root_binding(self, *, create: bool) -> _RootBinding:
        fds: list[int] = []
        try:
            current_fd = os.open(self.root.anchor, self._dir_flags())
            fds.append(current_fd)
            for part in self.root.parts[1:]:
                child = -1
                created = False
                try:
                    try:
                        child = os.open(part, self._dir_flags(), dir_fd=current_fd)
                    except FileNotFoundError:
                        if not create:
                            raise CronCallbackConflictError(
                                "callback root path binding is unavailable"
                            ) from None
                        try:
                            os.mkdir(part, 0o700, dir_fd=current_fd)
                        except FileExistsError:
                            pass
                        else:
                            created = True
                        child = os.open(part, self._dir_flags(), dir_fd=current_fd)
                    child_info = os.fstat(child)
                    entry_info = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                    if (
                        not stat.S_ISDIR(child_info.st_mode)
                        or self._inode(child_info) != self._inode(entry_info)
                    ):
                        raise CronCallbackConflictError(
                            "callback root component binding changed"
                        )
                    if created:
                        os.fchmod(child, 0o700)
                        os.fsync(child)
                        os.fsync(current_fd)
                    fds.append(child)
                    current_fd = child
                    child = -1
                finally:
                    if child >= 0:
                        os.close(child)
            self._assert_private(os.fstat(current_fd), directory=True)
            identities = tuple(self._inode(os.fstat(fd)) for fd in fds)
            if self._path_identities is not None and identities != self._path_identities:
                raise CronCallbackConflictError("callback root path binding changed")
            return _RootBinding(self, tuple(fds), tuple(self.root.parts[1:]), identities)
        except CronCallbackError:
            for fd in reversed(fds):
                os.close(fd)
            raise
        except OSError as exc:
            for fd in reversed(fds):
                os.close(fd)
            raise CronCallbackConflictError(
                "callback root must use a stable no-follow directory chain"
            ) from exc

    @contextmanager
    def _root_binding(self, *, create: bool = False) -> Generator[_RootBinding, None, None]:
        binding = self._open_root_binding(create=create)
        try:
            yield binding
        finally:
            binding.close()

    def require_callback_root_binding(self, binding: _RootBinding) -> None:
        """Reprove the existing callback root checks without invoking an observer."""
        return self._assert_root_binding(binding)

    def _assert_root_binding(self, binding: _RootBinding) -> None:
        try:
            if self._path_identities is not None and binding.identities != self._path_identities:
                raise CronCallbackConflictError("callback root path binding changed")
            for fd, identity in zip(binding.fds, binding.identities):
                if self._inode(os.fstat(fd)) != identity:
                    raise CronCallbackConflictError("callback root path binding changed")
            for parent_fd, child_fd, part, identity in zip(
                binding.fds, binding.fds[1:], binding.parts, binding.identities[1:]
            ):
                entry = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(entry.st_mode)
                    or self._inode(entry) != identity
                    or self._inode(os.fstat(child_fd)) != identity
                ):
                    raise CronCallbackConflictError("callback root path binding changed")
            self._assert_private(os.fstat(binding.root_fd), directory=True)
        except CronCallbackError:
            raise
        except OSError as exc:
            raise CronCallbackConflictError("callback root path binding changed") from exc

    @staticmethod
    def _record_name(callback_id: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", callback_id) is None:
            raise CronCallbackConflictError("callback ID is invalid")
        return f"{callback_id}.json"

    def _open_lock(self, binding: _RootBinding, *, create: bool = False) -> int:
        flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        fd = -1
        created = False
        try:
            try:
                fd = os.open(".lock", flags, dir_fd=binding.root_fd)
            except FileNotFoundError:
                if not create:
                    raise CronCallbackConflictError("callback lock is unavailable") from None
                fd = os.open(
                    ".lock", flags | os.O_CREAT | os.O_EXCL, 0o600,
                    dir_fd=binding.root_fd,
                )
                created = True
            if created:
                os.fchmod(fd, 0o600)
                os.fsync(fd)
                os.fsync(binding.root_fd)
            self._assert_private(os.fstat(fd), directory=False)
            identity = self._inode(os.fstat(fd))
            if self._lock_identity is not None and identity != self._lock_identity:
                raise CronCallbackConflictError("callback lock binding changed")
            entry = os.stat(".lock", dir_fd=binding.root_fd, follow_symlinks=False)
            if self._inode(entry) != identity:
                raise CronCallbackConflictError("callback lock binding changed")
            return fd
        except CronCallbackError:
            if fd >= 0:
                os.close(fd)
            raise
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            raise CronCallbackConflictError(
                "callback lock must be a stable private regular file"
            ) from exc

    def _assert_lock_binding(self, binding: _RootBinding, lock_fd: int) -> None:
        binding.assert_bound()
        try:
            info = os.fstat(lock_fd)
            entry = os.stat(".lock", dir_fd=binding.root_fd, follow_symlinks=False)
        except OSError as exc:
            raise CronCallbackConflictError("callback lock binding changed") from exc
        self._assert_private(info, directory=False)
        identity = self._inode(info)
        if identity != self._lock_identity or self._inode(entry) != identity:
            raise CronCallbackConflictError("callback lock binding changed")

    @contextmanager
    def _locked(self) -> Generator[_RootBinding, None, None]:
        lock_api = fcntl
        if lock_api is None:
            raise CronCallbackError("callback storage requires POSIX flock support")
        with self._root_binding() as binding:
            lock_fd = self._open_lock(binding)
            try:
                lock_api.flock(lock_fd, lock_api.LOCK_EX)
                self._assert_lock_binding(binding, lock_fd)
                yield binding
                self._assert_lock_binding(binding, lock_fd)
            finally:
                try:
                    lock_api.flock(lock_fd, lock_api.LOCK_UN)
                finally:
                    os.close(lock_fd)

    def _encode(self, record: CronCallbackRecord) -> bytes:
        value = {
            "schema_version": 1,
            "kind": "cron_callback_record_v1",
            "callback_id": record.callback_id,
            "observation": asdict(record.observation),
            "state": record.state,
            "failure_code": record.failure_code,
        }
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                          separators=(",", ":")).encode()

    def _decode(self, data: bytes, callback_id: str) -> CronCallbackRecord:
        try:
            value = json.loads(data)
            if set(value) != {"schema_version", "kind", "callback_id", "observation", "state", "failure_code"}:
                raise ValueError
            observation = CronCallbackObservation(**value["observation"])
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CronCallbackConflictError("callback record is corrupt") from exc
        if value["schema_version"] != 1 or value["kind"] != "cron_callback_record_v1":
            raise CronCallbackConflictError("callback record schema is invalid")
        states = {"pending", "invoking", "succeeded", "failed", "observed", "abandoned"}
        if value["callback_id"] != callback_id or value["state"] not in states:
            raise CronCallbackConflictError("callback record identity or state is invalid")
        failure = value["failure_code"]
        if failure is not None and failure != "callback_failed":
            raise CronCallbackConflictError("callback failure evidence is invalid")
        return CronCallbackRecord(callback_id, observation, value["state"], failure)

    def _read(self, binding: _RootBinding, callback_id: str) -> CronCallbackRecord:
        name = self._record_name(callback_id)
        fd = -1
        try:
            binding.assert_bound()
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=binding.root_fd)
            except FileNotFoundError:
                # Only absence proved before a descriptor is acquired can
                # authorize first publication. Disappearance after this point
                # is a conflicting mutation, never an absent record.
                raise KeyError(callback_id) from None
            info = os.fstat(fd)
            self._assert_private(info, directory=False)
            entry = os.stat(name, dir_fd=binding.root_fd, follow_symlinks=False)
            self._assert_private(entry, directory=False)
            opened_identity = self._inode(info)
            if self._inode(entry) != opened_identity:
                raise CronCallbackConflictError("callback record binding changed")
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > 65536:
                    raise CronCallbackConflictError("callback record exceeds its size limit")
                chunks.append(chunk)
            binding.assert_bound()
            final_info = os.fstat(fd)
            entry = os.stat(name, dir_fd=binding.root_fd, follow_symlinks=False)
            self._assert_private(final_info, directory=False)
            self._assert_private(entry, directory=False)
            if (
                self._inode(final_info) != opened_identity
                or self._inode(entry) != opened_identity
            ):
                raise CronCallbackConflictError("callback record binding changed")
            return self._decode(b"".join(chunks), callback_id)
        except KeyError:
            raise
        except CronCallbackError:
            raise
        except OSError as exc:
            raise CronCallbackConflictError("callback record is not a safe private file") from exc
        finally:
            if fd >= 0:
                os.close(fd)

    def _write(self, binding: _RootBinding, record: CronCallbackRecord) -> None:
        name = self._record_name(record.callback_id)
        temporary = f".{record.callback_id}.{uuid.uuid4().hex}.tmp"
        binding.assert_bound()
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=binding.root_fd,
        )
        published = False
        try:
            data = self._encode(record)
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
            staged_identity = self._inode(os.fstat(fd))
            os.rename(
                temporary, name,
                src_dir_fd=binding.root_fd,
                dst_dir_fd=binding.root_fd,
            )
            published = True
            entry = os.stat(name, dir_fd=binding.root_fd, follow_symlinks=False)
            if self._inode(entry) != staged_identity:
                raise CronCallbackConflictError("callback record destination binding changed")
            binding.assert_bound()
            os.fsync(binding.root_fd)
            binding.assert_bound()
            entry = os.stat(name, dir_fd=binding.root_fd, follow_symlinks=False)
            if self._inode(entry) != staged_identity:
                raise CronCallbackConflictError("callback record destination binding changed")
        finally:
            os.close(fd)
            if not published:
                try:
                    os.unlink(temporary, dir_fd=binding.root_fd)
                except FileNotFoundError:
                    pass

    def publish(self, observation: CronCallbackObservation) -> CronCallbackRecord:
        if not isinstance(cast(object, observation), CronCallbackObservation):
            raise TypeError("callback publication requires a typed observation")
        callback_id = self.callback_id(observation)
        candidate = CronCallbackRecord(callback_id, observation, "pending")
        with self._locked() as binding:
            try:
                existing = self._read(binding, callback_id)
            except KeyError:
                self._write(binding, candidate)
                return candidate
            if existing.observation != observation:
                raise CronCallbackConflictError("callback identity has conflicting observation")
            return existing

    def get(self, callback_id: str) -> CronCallbackRecord:
        with self._locked() as binding:
            return self._read(binding, callback_id)

    def _transition(self, callback_id: str, expected: set[str], state: CallbackState,
                    failure_code: str | None = None) -> CronCallbackRecord:
        with self._locked() as binding:
            record = self._read(binding, callback_id)
            if record.state not in expected:
                raise CronCallbackReconciliationRequiredError(
                    f"callback in state {record.state} cannot transition automatically"
                )
            updated = CronCallbackRecord(callback_id, record.observation, state, failure_code)
            self._write(binding, updated)
            return updated

    def _assert_authority(self) -> None:
        """Reprove the bound root and lock immediately before external code."""
        with self._locked():
            pass

    async def invoke_once(self, observation: CronCallbackObservation) -> CronCallbackRecord:
        record = self.publish(observation)
        if record.state != "pending":
            raise CronCallbackReconciliationRequiredError(
                f"callback in state {record.state} cannot be invoked automatically"
            )
        self._transition(record.callback_id, {"pending"}, "invoking")
        self._assert_authority()
        try:
            outcome = self._observer(observation)
            if inspect.isawaitable(outcome):
                await outcome
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._transition(record.callback_id, {"invoking"}, "failed", "callback_failed")
        return self._transition(record.callback_id, {"invoking"}, "succeeded")

    def reconcile(self, callback_id: str, *, resolution: Literal["observed", "abandoned"]) -> CronCallbackRecord:
        if resolution not in {"observed", "abandoned"}:
            raise ValueError("callback resolution must be observed or abandoned")
        return self._transition(callback_id, {"invoking", "failed"}, resolution)


__all__ = [
    "Callback", "CronCallbackConflictError", "CronCallbackError",
    "CronCallbackObservation", "CronCallbackRecord",
    "CronCallbackReconciliationRequiredError", "CronCallbackStore",
]
