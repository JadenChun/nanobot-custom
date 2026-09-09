"""Dormant cross-process source execution ownership and mutation fencing.

The OS lock is retained across effects; directory locks protect only short storage
transactions. There is deliberately no expiry or forced takeover of a live owner.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from typing import TYPE_CHECKING, AsyncIterator, Iterator

from filelock import FileLock, Timeout

from nanobot.session.manager import SessionWriteConflict
from nanobot.session.source import validate_execution_links

if TYPE_CHECKING:
    from nanobot.session.manager import JsonlSessionStore
    from nanobot.session.source_store import SourceSnapshot

_CAPABILITY = object()


class ExecutionBusyError(SessionWriteConflict):
    """Another execution retains the per-session OS lock."""


class ExecutionOwnershipError(SessionWriteConflict):
    """The supplied capability no longer owns this source execution."""


def _execution_lock(backend: JsonlSessionStore, key: str) -> FileLock:
    directory = backend.sessions_dir / ".execution-locks"
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    # Files stay in place even after session deletion; unlinking splits lock inodes.
    return FileLock(str(directory / f"{backend.storage_key(key)}.lock"), timeout=0, mode=0o600, thread_local=False)


class ExecutionOwner:
    """Nonserializable capability; only acquisition can create an active owner."""

    def __init__(self, capability: object, backend: JsonlSessionStore, key: str,
                 lock: FileLock, snapshot: SourceSnapshot, token: str, recovery_required: bool):
        if capability is not _CAPABILITY:
            raise ExecutionOwnershipError()
        self._backend = backend
        self.key = key
        self._lock = lock
        self._token = token
        self._incarnation = snapshot.state.incarnation
        self._generation = snapshot.state.generation
        self._store_identity = snapshot.state.store_identity
        self._root_task = asyncio.current_task()
        self._pid = os.getpid()
        self._mutex = threading.RLock()
        self._active = True
        self._closing = False
        self._children: set[asyncio.Task[object]] = set()
        self.recovery_required = recovery_required
        self.unresolved_executions = validate_execution_links(snapshot.events)

    def __reduce__(self):
        raise TypeError("Execution ownership cannot be serialized")

    def _require_registered_task(self) -> None:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            raise ExecutionOwnershipError() from None
        if os.getpid() != self._pid or (task is not self._root_task and task not in self._children):
            raise ExecutionOwnershipError()

    def _validate_unlocked(self, backend: JsonlSessionStore, snapshot: SourceSnapshot,
                           *, dispatch: bool = False) -> None:
        self._require_registered_task()
        lease = snapshot.header["metadata"].get("execution_lease")
        if (
            not self._active or not self._lock.is_locked
            or backend.sessions_dir != self._backend.sessions_dir
            or snapshot.state.store_identity != self._store_identity
            or snapshot.state.session_key != self.key
            or snapshot.state.incarnation != self._incarnation
            or snapshot.state.generation != self._generation
            or not isinstance(lease, dict) or lease.get("token") != self._token
            or lease.get("incarnation") != self._incarnation
            or type(lease.get("generation")) is not int
            or lease.get("generation") != self._generation
            or type(lease.get("schema_version")) is not int
            or lease.get("schema_version") != 1
        ):
            raise ExecutionOwnershipError()
        if dispatch and (self._closing or self.recovery_required or lease.get("recovery_required") is not False):
            raise ExecutionOwnershipError()

    def read_snapshot(self) -> SourceSnapshot:
        with self._mutex, self._backend.locked_session_files():
            snapshot = self._backend.source.load_source_snapshot(self.key)
            self._validate_unlocked(self._backend, snapshot)
            return snapshot

    def validate_dispatch(self) -> None:
        """Check authority immediately before an effect while retaining the OS lock."""
        with self._mutex, self._backend.locked_session_files():
            snapshot = self._backend.source.load_source_snapshot(self.key)
            self._validate_unlocked(self._backend, snapshot, dispatch=True)

    def track_task(self, task: asyncio.Task[object]) -> None:
        """Retain ownership until a registered child capable of effects has finished."""
        self._require_registered_task()
        if not self._active or self._closing or task is self._root_task or task.get_loop() is not asyncio.get_running_loop():
            raise ExecutionOwnershipError()
        self._children.add(task)
        task.add_done_callback(self._children.discard)

    async def quiesce(self) -> None:
        """Root-only barrier before release, reset, or delete."""
        if os.getpid() != self._pid or asyncio.current_task() is not self._root_task:
            raise ExecutionOwnershipError()
        self._closing = True
        interrupted = False
        while pending := [task for task in self._children if not task.done()]:
            joined = asyncio.gather(*pending, return_exceptions=True)
            while not joined.done():
                try:
                    await asyncio.shield(joined)
                except asyncio.CancelledError:
                    interrupted = True
        if interrupted:
            raise asyncio.CancelledError

    def _require_drained(self) -> None:
        if not self._closing or any(not task.done() for task in self._children):
            raise ExecutionOwnershipError()
        if os.getpid() != self._pid or asyncio.current_task() is not self._root_task:
            raise ExecutionOwnershipError()

    async def aclose(self) -> None:
        if os.getpid() != self._pid or asyncio.current_task() is not self._root_task:
            raise ExecutionOwnershipError()
        if not self._active:
            return
        interrupted = False
        try:
            try:
                await self.quiesce()
            except asyncio.CancelledError:
                interrupted = True
            with self._mutex, self._backend.locked_session_files():
                snapshot = self._backend.source.load_source_snapshot(self.key)
                self._validate_unlocked(self._backend, snapshot)
                header = snapshot.header
                retained_lease = deepcopy(header["metadata"]["execution_lease"])
                if self.recovery_required or validate_execution_links(snapshot.events):
                    header["metadata"]["execution_recovery_required"] = True
                header["metadata"].pop("execution_lease", None)
                header["revision"] += 1
                try:
                    self._backend.source._write(header, snapshot.events)
                except BaseException as failure:
                    # A replacement may have happened before fsync failed. Keep
                    # abandoned-owner evidence even in that complete-new-file case.
                    try:
                        current = self._backend.source.load_source_snapshot(self.key)
                        if current.state.incarnation == self._incarnation and current.state.generation == self._generation:
                            retained_lease["recovery_required"] = True
                            current.header["metadata"]["execution_lease"] = retained_lease
                            current.header["metadata"]["execution_recovery_required"] = True
                            current.header["revision"] += 1
                            self._backend.source._write(current.header, current.events)
                    except BaseException:
                        failure.add_note("Persisting execution recovery metadata also failed.")
                    raise
        finally:
            self._active = False
            self._lock.release()
        if interrupted:
            raise asyncio.CancelledError

    def _invalidate_after_destructive_write(self) -> None:
        self._active = False
        # The surrounding mutation context still holds the OS lock until it exits.


@asynccontextmanager
async def acquire_execution(backend: JsonlSessionStore, key: str, *, run_id: str) -> AsyncIterator[ExecutionOwner]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ExecutionOwnershipError()
    lock = _execution_lock(backend, key)
    owner = None
    try:
        while True:
            try:
                lock.acquire(timeout=0)
                break
            except Timeout:
                await asyncio.sleep(0.05)
        with backend.locked_session_files():
            snapshot = backend.source.load_source_snapshot(key)
            metadata = snapshot.header["metadata"]
            recovery_required = (
                "execution_lease" in metadata
                or "execution_recovery_required" in metadata
                or bool(validate_execution_links(snapshot.events))
                or backend.get_runtime_checkpoint_path(key).exists()
                or backend.get_runtime_checkpoint_path(key).is_symlink()
            )
            token = uuid.uuid4().hex
            metadata["execution_lease"] = {
                "schema_version": 1, "token": token, "run_id": run_id,
                "incarnation": snapshot.state.incarnation, "generation": snapshot.state.generation,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "recovery_required": recovery_required,
            }
            snapshot.header["revision"] += 1
            committed = backend.source._write(snapshot.header, snapshot.events)
            owner = ExecutionOwner(_CAPABILITY, backend, key, lock, committed, token, recovery_required)
        try:
            yield owner
        except BaseException as primary:
            try:
                await owner.aclose()
            except BaseException:
                primary.add_note("Execution ownership cleanup also failed.")
            raise
        else:
            await owner.aclose()
    finally:
        if owner is not None:
            owner._active = False
        lock.release()


@contextmanager
def execution_transaction(backend: JsonlSessionStore, key: str,
                          owner: ExecutionOwner | None = None, *, destructive: bool = False,
                          allow_legacy: bool = False) -> Iterator[None]:
    """Always acquire execution authority before entering the directory lock."""
    if owner is not None:
        if type(owner) is not ExecutionOwner:
            raise ExecutionOwnershipError()
        with owner._mutex:
            if not owner._active or not owner._lock.is_locked or owner.key != key:
                raise ExecutionOwnershipError()
            if destructive:
                owner._require_drained()
            with backend.locked_session_files():
                snapshot = backend.source.load_source_snapshot(key)
                owner._validate_unlocked(backend, snapshot)
                yield
        return
    with execution_lock_guard(backend, key), backend.locked_session_files():
        try:
            header, _ = backend.source._read_unlocked(key, require_source=not allow_legacy)
        except FileNotFoundError:
            raise SessionWriteConflict() from None
        if "execution_lease" in header["metadata"] or "execution_recovery_required" in header["metadata"]:
            raise ExecutionOwnershipError()
        yield


@contextmanager
def execution_lock_guard(backend: JsonlSessionStore, key: str) -> Iterator[None]:
    """Acquire only the OS lock; callers must recheck format under storage lock."""
    lock = _execution_lock(backend, key)
    try:
        try:
            lock.acquire(timeout=0)
        except Timeout:
            raise ExecutionBusyError() from None
        yield
    finally:
        lock.release()
