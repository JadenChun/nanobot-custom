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
from typing import TYPE_CHECKING, AsyncGenerator, Generator, NoReturn, cast

from filelock import FileLock, Timeout

from nanobot.session.manager import SessionWriteConflict
from nanobot.session.source import validate_execution_links
from nanobot.session.source_store import SourceSessionStore

if TYPE_CHECKING:
    from nanobot.session.manager import JsonlSessionStore
    from nanobot.session.source_root import SourceRootAuthority
    from nanobot.session.source_store import SourceSnapshot
    from nanobot.workspace_access_recovery import WorkspaceAccessSourceBinding

_CAPABILITY = object()


class ExecutionBusyError(SessionWriteConflict):
    """Another execution retains the per-session OS lock."""


class ExecutionOwnershipError(SessionWriteConflict):
    """The supplied capability no longer owns this source execution."""


def _execution_lock(backend: JsonlSessionStore, key: str) -> FileLock:
    from nanobot.session.mode_storage import require_namespace_write_access

    require_namespace_write_access(backend)
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

    def __reduce__(self) -> NoReturn:
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
            or not isinstance(lease, dict) or cast(dict[object, object], lease).get("token") != self._token
            or cast(dict[object, object], lease).get("incarnation") != self._incarnation
            or type(cast(dict[object, object], lease).get("generation")) is not int
            or cast(dict[object, object], lease).get("generation") != self._generation
            or type(cast(dict[object, object], lease).get("schema_version")) is not int
            or cast(dict[object, object], lease).get("schema_version") != 1
        ):
            raise ExecutionOwnershipError()
        if dispatch and (self._closing or self.recovery_required or cast(dict[object, object], lease).get("recovery_required") is not False):
            raise ExecutionOwnershipError()

    @contextmanager
    def transaction(
        self, backend: JsonlSessionStore, key: str, *, destructive: bool = False,
    ) -> Generator[None, None, None]:
        """Validate this owner under its mutex before entering the storage lock."""
        if type(self) is not ExecutionOwner:
            raise ExecutionOwnershipError()
        with self._mutex:
            if not self._active or not self._lock.is_locked or self.key != key:
                raise ExecutionOwnershipError()
            if destructive:
                self._require_drained()
            with backend.locked_session_files():
                snapshot = backend.source.load_source_snapshot(key)
                self._validate_unlocked(backend, snapshot)
                yield

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

    @contextmanager
    def closing_transaction(self, backend: JsonlSessionStore) -> Generator[SourceSnapshot, None, None]:
        """Derive close data under the real owner locks after root quiescence."""
        if type(self) is not ExecutionOwner:
            raise ExecutionOwnershipError()
        with self._mutex, backend.locked_session_files():
            snapshot = backend.source.load_source_snapshot(self.key)
            self._validate_unlocked(backend, snapshot)
            self._require_drained()
            yield snapshot

    @staticmethod
    def copy_execution_lease(lease: dict[str, object]) -> dict[str, object]:
        """Detach validated lease data for close-time recovery, not authority."""
        return deepcopy(lease)

    def matches_current_execution_lifetime(self, snapshot: SourceSnapshot) -> bool:
        """Compare only close-recovery identity data with the original short circuit."""
        return (
            snapshot.state.incarnation == self._incarnation
            and snapshot.state.generation == self._generation
        )

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
            from nanobot.session.source_store import SourceSessionStore

            SourceSessionStore.settle_execution_close(self._backend, self)
        finally:
            self._active = False
            self._lock.release()
        if interrupted:
            raise asyncio.CancelledError

    def revoke_acquisition(self, capability: object) -> None:
        """Acquisition-finally fence; it does not release the retained OS lock."""
        if capability is not _CAPABILITY:
            raise ExecutionOwnershipError()
        self._active = False

    def retire_after_destructive_write(self) -> None:
        """Revoke dispatch after reset/delete without releasing the retained OS lock."""
        self._invalidate_after_destructive_write()

    def _invalidate_after_destructive_write(self) -> None:
        self._active = False
        # The surrounding mutation context still holds the OS lock until it exits.


@asynccontextmanager
async def acquire_execution(
    backend: JsonlSessionStore,
    key: str,
    *,
    run_id: str,
    expected_source: WorkspaceAccessSourceBinding | None = None,
) -> AsyncGenerator[ExecutionOwner, None]:
    if not isinstance(cast(object, run_id), str) or not run_id.strip():
        raise ExecutionOwnershipError()
    if expected_source is not None:
        from nanobot.workspace_access_recovery import parse_workspace_access_source_binding

        try:
            expected_source = parse_workspace_access_source_binding(expected_source)
        except (TypeError, ValueError):
            raise ExecutionOwnershipError() from None
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
            if expected_source is not None and (
                snapshot.state.session_key != key
                or expected_source != {
                    "store_identity": snapshot.state.store_identity,
                    "incarnation": snapshot.state.incarnation,
                    "generation": snapshot.state.generation,
                }
            ):
                raise SessionWriteConflict()
            metadata = snapshot.header["metadata"]
            recovery_required = (
                "execution_lease" in metadata
                or "execution_recovery_required" in metadata
                or bool(validate_execution_links(snapshot.events))
                or backend.get_runtime_checkpoint_path(key).exists()
                or backend.get_runtime_checkpoint_path(key).is_symlink()
            )
            token = uuid.uuid4().hex
            committed = SourceSessionStore.publish_execution_lease(
                backend, snapshot, metadata, token=token, run_id=run_id,
                recovery_required=recovery_required,
                select_acquired_at=lambda: datetime.now(timezone.utc).isoformat(),
            )
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
            owner.revoke_acquisition(_CAPABILITY)
        lock.release()


@contextmanager
def execution_transaction(backend: JsonlSessionStore, key: str,
                          owner: ExecutionOwner | None = None, *, destructive: bool = False,
                          allow_legacy: bool = False,
                          root_authority: SourceRootAuthority | None = None) -> Generator[None, None, None]:
    """Always acquire execution authority before entering the directory lock."""
    if root_authority is not None:
        if owner is not None or destructive or allow_legacy:
            raise ExecutionOwnershipError()
        with root_authority.operation(backend):
            with _descriptor_execution_lock_guard(backend, key, root_authority), root_authority.lock_file(
                ".session-files.lock", blocking=True
            ):
                backend.source.assert_no_execution_metadata(key, root_authority=root_authority)
                yield
        return
    if owner is not None:
        if type(owner) is not ExecutionOwner:
            raise ExecutionOwnershipError()
        with owner.transaction(backend, key, destructive=destructive):
            yield
        return
    with execution_lock_guard(backend, key), backend.locked_session_files():
        backend.source.assert_no_execution_metadata(key, allow_legacy=allow_legacy)
        yield


@contextmanager
def execution_lock_guard(backend: JsonlSessionStore, key: str) -> Generator[None, None, None]:
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


@contextmanager
def _descriptor_execution_lock_guard(
    backend: JsonlSessionStore,
    key: str,
    root_authority: SourceRootAuthority,
) -> Generator[None, None, None]:
    """Acquire the same flock-compatible execution lock relative to the root."""

    try:
        with root_authority.lock_nested_file(
            ".execution-locks",
            f"{backend.storage_key(key)}.lock",
            blocking=False,
        ):
            yield
    except BlockingIOError:
        raise ExecutionBusyError() from None
