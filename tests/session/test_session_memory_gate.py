"""Session construction cooperates with workspace memory recovery locking."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery import MemoryReceiptRecoveryRequiredError
from nanobot.session import manager as session_manager_module
from nanobot.session.manager import JsonlSessionStore


def _storage_capable() -> bool:
    if os.name != "posix":
        return False
    if any(
        not isinstance(getattr(os, name, None), int)
        for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    ):
        return False
    if not all(
        function in getattr(os, "supports_dir_fd", set())
        for function in (os.open, os.stat, os.mkdir, os.link)
    ):
        return False
    if os.link not in getattr(os, "supports_follow_symlinks", set()):
        return False
    return (
        storage._fcntl is not None
        and isinstance(getattr(storage._fcntl, "LOCK_NB", None), int)
    )


def _fork_context() -> Any:
    if not _storage_capable():
        pytest.skip("workspace memory locks require POSIX no-follow dir-fd/flock")
    try:
        return multiprocessing.get_context("fork")
    except (AttributeError, ValueError):
        pytest.skip("deterministic process contention requires fork")


def _receipt_bytes() -> bytes:
    memory_output = "# Memory\nreceipt publication test\n"
    payload = {
        "version": 1,
        "chunk_digest": "a" * 64,
        "history_marker_id": "marker-1",
        "history_entry": "[2026-09-18 12:00] recovered summary",
        "memory_output": memory_output,
        "base_memory_digest": "b" * 64,
        "output_memory_digest": hashlib.sha256(memory_output.encode("utf-8")).hexdigest(),
        "reference_only": False,
        "created_at": "2026-09-18T12:00:00+00:00",
        "session_key": "cli:receipt",
        "start_offset": 0,
        "end_offset": 2,
        "session_generation": 4,
    }
    return json.dumps(payload).encode("utf-8")


def _session_bytes(key: str, content: str, updated_at: str) -> bytes:
    metadata = {
        "_type": "metadata",
        "key": key,
        "created_at": "2026-09-17T00:00:00",
        "updated_at": updated_at,
        "metadata": {},
        "last_consolidated": 0,
    }
    return (
        json.dumps(metadata, separators=(",", ":"))
        + "\n"
        + json.dumps({"role": "user", "content": content}, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _snapshot(path: Path) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns


def _seed_migration_pair(
    workspace: Path, sessions_root: Path
) -> tuple[Path, bytes, tuple[int, ...], Path, bytes, tuple[int, ...]]:
    workspace.mkdir(mode=0o700)
    legacy_dir = workspace / "sessions"
    legacy_dir.mkdir()
    key = "telegram:memory-gate"
    name = f"{JsonlSessionStore.storage_key(key)}.jsonl"
    source = legacy_dir / name
    source_bytes = _session_bytes(key, "legacy source", "2026-09-19T00:00:00")
    source.write_bytes(source_bytes)

    workspace_id = "a" * 32
    state_dir = workspace / ".nanobot"
    state_dir.mkdir()
    (state_dir / "workspace-id").write_text(f"{workspace_id}\n", encoding="utf-8")
    sessions_root.mkdir()
    namespace = sessions_root / workspace_id
    namespace.mkdir()
    marker = namespace / ".workspace"
    marker.write_text(f"{workspace.resolve()}\n", encoding="utf-8")
    destination = namespace / name
    destination_bytes = _session_bytes(
        key, "canonical destination", "2026-09-18T00:00:00"
    )
    destination.write_bytes(destination_bytes)
    return (
        source,
        source_bytes,
        _snapshot(source),
        destination,
        destination_bytes,
        _snapshot(destination),
    )


def _install_flock_probe(report: Any) -> None:
    """Report whether the next exclusive flock is contended, without sleeping."""
    assert storage._fcntl is not None
    fcntl = storage._fcntl
    real_flock = fcntl.flock
    first_exclusive = True

    def probing_flock(fd: int, operation: int) -> None:
        nonlocal first_exclusive
        if first_exclusive and operation == fcntl.LOCK_EX:
            first_exclusive = False
            try:
                real_flock(fd, operation | fcntl.LOCK_NB)
            except BlockingIOError:
                report.put("contended")
            else:
                real_flock(fd, fcntl.LOCK_UN)
                report.put("free")
        real_flock(fd, operation)

    fcntl.flock = probing_flock


def _construct_store_process(
    workspace: str,
    sessions_root: str,
    result: Any,
    *,
    flock_report: Any | None = None,
    migration_entered: Any | None = None,
    continue_migration: Any | None = None,
) -> None:
    if flock_report is not None:
        _install_flock_probe(flock_report)
    if migration_entered is not None and continue_migration is not None:
        migrate = JsonlSessionStore._migrate_from_workspace

        def pause_migration(self: JsonlSessionStore, source_workspace: Path) -> None:
            migration_entered.set()
            if not continue_migration.wait(timeout=10):
                raise TimeoutError("test did not release paused migration")
            migrate(self, source_workspace)

        JsonlSessionStore._migrate_from_workspace = pause_migration
    try:
        JsonlSessionStore(Path(workspace), sessions_root=Path(sessions_root))
    except BaseException as exc:
        result.put(("error", type(exc).__name__, str(exc)))
    else:
        result.put(("ok",))


def _hold_authority_and_publish(
    workspace: str,
    authority_ready: Any,
    publish_receipt: Any,
    receipt_published: Any,
    release_authority: Any,
    result: Any,
) -> None:
    try:
        with storage.WorkspaceMemoryAuthority(Path(workspace)):
            authority_ready.set()
            if not publish_receipt.wait(timeout=10):
                raise TimeoutError("test did not request receipt publication")
            receipt = Path(workspace) / "memory" / ".consolidation.pending.json"
            receipt.write_bytes(_receipt_bytes())
            receipt.chmod(0o600)
            receipt_published.set()
            if not release_authority.wait(timeout=10):
                raise TimeoutError("test did not release memory authority")
    except BaseException as exc:
        result.put(("error", type(exc).__name__, str(exc)))
    else:
        result.put(("ok",))


def _enter_authority_process(
    workspace: str,
    flock_report: Any,
    authority_entered: Any,
    result: Any,
) -> None:
    _install_flock_probe(flock_report)
    try:
        with storage.WorkspaceMemoryAuthority(Path(workspace)):
            authority_entered.set()
        result.put(("ok",))
    except BaseException as exc:
        result.put(("error", type(exc).__name__, str(exc)))


def _finish_processes(*processes: Any) -> None:
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_constructor_waits_for_authority_then_rechecks_published_receipt(
    tmp_path: Path,
) -> None:
    context = _fork_context()
    workspace = tmp_path / "workspace"
    sessions_root = tmp_path / "private-sessions"
    source, source_bytes, source_stat, destination, destination_bytes, destination_stat = (
        _seed_migration_pair(workspace, sessions_root)
    )
    namespace = destination.parent
    workspace_id_marker = workspace / ".nanobot" / "workspace-id"
    workspace_id_bytes = workspace_id_marker.read_bytes()
    workspace_id_stat = _snapshot(workspace_id_marker)
    marker = namespace / ".workspace"
    marker_bytes = marker.read_bytes()
    marker_stat = _snapshot(marker)

    authority_ready = context.Event()
    publish_receipt = context.Event()
    receipt_published = context.Event()
    release_authority = context.Event()
    authority_result = context.Queue()
    constructor_report = context.Queue()
    constructor_result = context.Queue()
    authority = context.Process(
        target=_hold_authority_and_publish,
        args=(
            str(workspace),
            authority_ready,
            publish_receipt,
            receipt_published,
            release_authority,
            authority_result,
        ),
    )
    constructor = context.Process(
        target=_construct_store_process,
        args=(str(workspace), str(sessions_root), constructor_result),
        kwargs={"flock_report": constructor_report},
    )
    authority.start()
    constructor_started = False
    try:
        assert authority_ready.wait(timeout=10)
        constructor.start()
        constructor_started = True
        assert constructor_report.get(timeout=10) == "contended"
        assert workspace_id_marker.read_bytes() == workspace_id_bytes
        assert _snapshot(workspace_id_marker) == workspace_id_stat
        assert not (sessions_root / ".workspace-migration.lock").exists()
        assert not (namespace / ".session-files.lock").exists()
        assert source.read_bytes() == source_bytes
        assert _snapshot(source) == source_stat
        assert destination.read_bytes() == destination_bytes
        assert _snapshot(destination) == destination_stat

        publish_receipt.set()
        assert receipt_published.wait(timeout=10)
    finally:
        release_authority.set()
        _finish_processes(authority, constructor)

    assert constructor_started
    assert constructor_result.get(timeout=2) == (
        "error",
        MemoryReceiptRecoveryRequiredError.__name__,
        "memory recovery required",
    )
    assert authority_result.get(timeout=2) == ("ok",)
    assert (workspace / "memory" / ".consolidation.pending.json").read_bytes() == _receipt_bytes()
    assert source.read_bytes() == source_bytes
    assert _snapshot(source) == source_stat
    assert destination.read_bytes() == destination_bytes
    assert _snapshot(destination) == destination_stat
    assert workspace_id_marker.read_bytes() == workspace_id_bytes
    assert _snapshot(workspace_id_marker) == workspace_id_stat
    assert marker.read_bytes() == marker_bytes
    assert _snapshot(marker) == marker_stat
    assert not (namespace / ".migration-conflicts").exists()


def test_authority_waits_until_constructor_migration_finishes(tmp_path: Path) -> None:
    context = _fork_context()
    workspace = tmp_path / "workspace"
    sessions_root = tmp_path / "private-sessions"
    source, source_bytes, _, destination, _, _ = _seed_migration_pair(workspace, sessions_root)
    migration_entered = context.Event()
    continue_migration = context.Event()
    constructor_result = context.Queue()
    authority_report = context.Queue()
    authority_entered = context.Event()
    authority_result = context.Queue()
    constructor = context.Process(
        target=_construct_store_process,
        args=(str(workspace), str(sessions_root), constructor_result),
        kwargs={
            "migration_entered": migration_entered,
            "continue_migration": continue_migration,
        },
    )
    authority = context.Process(
        target=_enter_authority_process,
        args=(str(workspace), authority_report, authority_entered, authority_result),
    )
    constructor.start()
    authority_started = False
    try:
        assert migration_entered.wait(timeout=10)
        authority.start()
        authority_started = True
        assert authority_report.get(timeout=10) == "contended"
        assert not authority_entered.is_set()
        assert source.exists()
        assert source.read_bytes() == source_bytes
        assert not (sessions_root / ("a" * 32) / ".migration-conflicts").exists()
    finally:
        continue_migration.set()
        _finish_processes(constructor, authority)

    assert authority_started
    assert constructor_result.get(timeout=2) == ("ok",)
    assert authority_result.get(timeout=2) == ("ok",)
    assert authority_entered.is_set()
    assert not source.exists()
    assert destination.read_bytes() == source_bytes


def test_receipt_published_after_preflight_blocks_before_session_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    (workspace / "memory").mkdir()
    legacy_dir = workspace / "sessions"
    legacy_dir.mkdir()
    key = "telegram:receipt-arrival"
    source = legacy_dir / f"{JsonlSessionStore.storage_key(key)}.jsonl"
    source_bytes = _session_bytes(key, "legacy source", "2026-09-19T00:00:00")
    source.write_bytes(source_bytes)
    source_stat = _snapshot(source)
    sessions_root = tmp_path / "private-sessions"
    receipt = workspace / "memory" / ".consolidation.pending.json"
    real_preflight = session_manager_module.require_memory_receipt_recovery
    calls = 0

    def publish_after_preflight(path: Path) -> None:
        nonlocal calls
        calls += 1
        real_preflight(path)
        if calls == 1:
            receipt.write_bytes(_receipt_bytes())
            receipt.chmod(0o600)

    monkeypatch.setattr(
        session_manager_module,
        "require_memory_receipt_recovery",
        publish_after_preflight,
    )

    with pytest.raises(MemoryReceiptRecoveryRequiredError) as failure:
        JsonlSessionStore(workspace, sessions_root=sessions_root)

    assert failure.value.args == ("memory recovery required",)
    assert calls == 2
    assert not sessions_root.exists()
    assert not (workspace / ".nanobot").exists()
    assert source.read_bytes() == source_bytes
    assert _snapshot(source) == source_stat
    assert receipt.read_bytes() == _receipt_bytes()


def test_constructor_accepts_existing_ordinary_0755_memory_layout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    memory = workspace / "memory"
    memory.mkdir(mode=0o755)
    memory.chmod(0o755)
    marker = memory / "existing.txt"
    marker.write_bytes(b"ordinary workspace memory")
    memory_snapshot = _snapshot(memory)
    marker_snapshot = _snapshot(marker)

    JsonlSessionStore(workspace, sessions_root=tmp_path / "private-sessions")

    assert _snapshot(memory) == memory_snapshot
    assert _snapshot(marker) == marker_snapshot
    assert marker.read_bytes() == b"ordinary workspace memory"
    assert not (memory / ".recovery").exists()


def test_namespace_setup_failure_releases_memory_gate_and_keeps_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _storage_capable():
        pytest.skip("workspace memory locks require POSIX no-follow dir-fd/flock")
    workspace = tmp_path / "workspace"
    sessions_root = tmp_path / "private-sessions"
    workspace.mkdir(mode=0o700)
    legacy_dir = workspace / "sessions"
    legacy_dir.mkdir()
    key = "telegram:setup-failure"
    source = legacy_dir / f"{JsonlSessionStore.storage_key(key)}.jsonl"
    source_bytes = _session_bytes(key, "legacy remains", "2026-09-19T00:00:00")
    source.write_bytes(source_bytes)
    source_stat = _snapshot(source)

    def fail_workspace_id_setup(workspace: Path, root: Path) -> str:
        raise OSError("simulated workspace marker setup failure")

    monkeypatch.setattr(
        JsonlSessionStore,
        "_load_or_create_workspace_id",
        staticmethod(fail_workspace_id_setup),
    )

    with pytest.raises(OSError, match="simulated workspace marker setup failure"):
        JsonlSessionStore(workspace, sessions_root=sessions_root)

    assert source.read_bytes() == source_bytes
    assert _snapshot(source) == source_stat
    assert not (workspace / ".nanobot" / "workspace-id").exists()
    assert not any(sessions_root.glob("*/.workspace"))

    assert storage._fcntl is not None
    lock_fd = os.open(workspace / ".memory-recovery.lock", os.O_RDWR | os.O_NOFOLLOW)
    try:
        storage._fcntl.flock(lock_fd, storage._fcntl.LOCK_EX | storage._fcntl.LOCK_NB)
        storage._fcntl.flock(lock_fd, storage._fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def test_constructor_fails_closed_when_posix_gate_capability_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _storage_capable():
        pytest.skip("POSIX memory lock capability injection requires dir-fd/flock support")
    assert storage._fcntl is not None
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    sessions_root = tmp_path / "private-sessions"

    monkeypatch.setattr(storage._fcntl, "LOCK_NB", None)

    with pytest.raises(storage.MemoryStorageUnavailableError):
        JsonlSessionStore(workspace, sessions_root=sessions_root)

    assert not sessions_root.exists()
    assert not (workspace / ".nanobot" / "workspace-id").exists()
    assert not (workspace / ".memory-recovery.lock").exists()
