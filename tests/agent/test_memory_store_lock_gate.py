"""Concurrency tests for MemoryStore's workspace memory lock gate."""

import hashlib
import json
import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_recovery import MemoryReceiptRecoveryRequiredError
from nanobot.agent.memory_recovery_storage import WorkspaceMemoryAuthority


def _posix_gate_supported() -> bool:
    try:
        storage._require_lock_gate_capabilities()
    except storage.MemoryStorageUnavailableError:
        return False
    return True


def _receipt_bytes() -> bytes:
    memory_output = "recovered memory"
    payload = {
        "version": 1,
        "chunk_digest": hashlib.sha256(b"chunk").hexdigest(),
        "history_marker_id": "marker",
        "history_entry": "entry",
        "memory_output": memory_output,
        "base_memory_digest": hashlib.sha256(b"base memory").hexdigest(),
        "output_memory_digest": hashlib.sha256(memory_output.encode()).hexdigest(),
        "reference_only": False,
        "created_at": "2026-01-01T00:00:00Z",
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def _publish_receipt(memory_dir: Path) -> None:
    staging = memory_dir / ".receipt.staging"
    staging.write_bytes(_receipt_bytes())
    staging.chmod(0o600)
    staging.replace(memory_dir / ".consolidation.pending.json")


def _memory_store_worker(
    workspace: str,
    gate_requested: Any,
    continue_gate: Any,
    result: Any,
) -> None:
    from nanobot.agent import memory
    from nanobot.agent.memory_recovery_storage import open_workspace_memory_lock

    def observed_gate(path: Path | str) -> object:
        gate_requested.set()
        if not continue_gate.wait(10):
            raise TimeoutError("test did not release memory gate handshake")
        return open_workspace_memory_lock(path)

    memory.open_workspace_memory_lock = observed_gate
    try:
        MemoryStore(Path(workspace))
    except Exception as exc:
        result.put(("error", f"{type(exc).__name__}: {exc}"))
    else:
        result.put(("ok", "constructed"))


def _paused_migration_worker(
    workspace: str,
    migration_paused: Any,
    continue_migration: Any,
    result: Any,
) -> None:
    from nanobot.agent.memory import MemoryStore

    migrate = MemoryStore._maybe_migrate_legacy_history

    def pause_migration(self: MemoryStore) -> None:
        migration_paused.set()
        if not continue_migration.wait(10):
            raise TimeoutError("test did not release legacy migration")
        migrate(self)

    MemoryStore._maybe_migrate_legacy_history = pause_migration
    try:
        MemoryStore(Path(workspace))
    except Exception as exc:
        result.put(("error", f"{type(exc).__name__}: {exc}"))
    else:
        result.put(("ok", "constructed"))


def _authority_worker(
    workspace: str,
    authority_attempting: Any,
    authority_entered: Any,
    result: Any,
) -> None:
    from nanobot.agent.memory_recovery_storage import WorkspaceMemoryAuthority

    authority = WorkspaceMemoryAuthority(Path(workspace))
    authority_attempting.set()
    try:
        with authority:
            authority_entered.set()
            result.put(("ok", "entered"))
    except Exception as exc:
        result.put(("error", f"{type(exc).__name__}: {exc}"))


def _receive(result: Any) -> tuple[str, str]:
    try:
        return result.get(timeout=10)
    except Empty:
        pytest.fail("worker did not report a result")


def _finish_process(process: multiprocessing.Process) -> None:
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    assert process.exitcode == 0


def _cleanup_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)


def _make_private_memory_dir(workspace: Path) -> Path:
    memory_dir = workspace / "memory"
    memory_dir.mkdir()
    memory_dir.chmod(0o700)
    return memory_dir


@pytest.mark.skipif(
    not _posix_gate_supported(),
    reason="workspace memory lock gate requires POSIX descriptor/flock support",
)
def test_memory_store_rechecks_receipt_after_waiting_for_authority_lock(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    memory_dir = _make_private_memory_dir(tmp_path)
    legacy = memory_dir / "HISTORY.md"
    legacy.write_text("[2026-01-01 00:00] legacy history\n", encoding="utf-8")
    gate_requested = context.Event()
    continue_gate = context.Event()
    result = context.Queue()
    process = context.Process(
        target=_memory_store_worker,
        args=(str(tmp_path), gate_requested, continue_gate, result),
    )

    try:
        with WorkspaceMemoryAuthority(tmp_path):
            process.start()
            assert gate_requested.wait(10), "constructor never reached the memory gate"
            _publish_receipt(memory_dir)
            # The worker passed its read-only entry preflight and is held before
            # gate entry until after the receipt is durably published.
            continue_gate.set()

        assert _receive(result) == (
            "error",
            "MemoryReceiptRecoveryRequiredError: memory recovery required",
        )
        _finish_process(process)
        assert legacy.read_text(encoding="utf-8") == "[2026-01-01 00:00] legacy history\n"
        assert not (memory_dir / "history.jsonl").exists()
        assert not (memory_dir / ".cursor").exists()
        assert not (memory_dir / ".dream_cursor").exists()
    finally:
        continue_gate.set()
        _cleanup_process(process)
        result.close()


@pytest.mark.skipif(
    not _posix_gate_supported(),
    reason="workspace memory lock gate requires POSIX descriptor/flock support",
)
def test_authority_waits_until_legacy_history_migration_finishes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    memory_dir = _make_private_memory_dir(tmp_path)
    legacy = memory_dir / "HISTORY.md"
    legacy.write_text("[2026-01-01 00:00] legacy history\n", encoding="utf-8")
    migration_paused = context.Event()
    continue_migration = context.Event()
    authority_attempting = context.Event()
    authority_entered = context.Event()
    memory_result = context.Queue()
    authority_result = context.Queue()
    memory_process = context.Process(
        target=_paused_migration_worker,
        args=(str(tmp_path), migration_paused, continue_migration, memory_result),
    )
    authority_process = context.Process(
        target=_authority_worker,
        args=(str(tmp_path), authority_attempting, authority_entered, authority_result),
    )

    try:
        memory_process.start()
        assert migration_paused.wait(10), "legacy migration was not reached"
        authority_process.start()
        assert authority_attempting.wait(10), "authority did not attempt lock entry"
        assert not authority_entered.wait(0.25), (
            "authority entered while legacy history migration held the gate"
        )

        continue_migration.set()
        assert _receive(memory_result) == ("ok", "constructed")
        assert _receive(authority_result) == ("ok", "entered")
        assert authority_entered.is_set()
        _finish_process(memory_process)
        _finish_process(authority_process)
        assert not legacy.exists()
        assert (memory_dir / "HISTORY.md.bak").read_text(encoding="utf-8") == (
            "[2026-01-01 00:00] legacy history\n"
        )
        assert (memory_dir / "history.jsonl").is_file()
    finally:
        continue_migration.set()
        _cleanup_process(memory_process)
        _cleanup_process(authority_process)
        memory_result.close()
        authority_result.close()


@pytest.mark.skipif(
    not _posix_gate_supported(),
    reason="workspace memory lock gate requires POSIX descriptor/flock support",
)
def test_existing_memory_directory_mode_is_preserved(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(mode=0o755)
    memory_dir.chmod(0o755)

    store = MemoryStore(tmp_path)

    assert store.memory_dir == memory_dir
    assert memory_dir.stat().st_mode & 0o777 == 0o755


@pytest.mark.skipif(
    not _posix_gate_supported(),
    reason="workspace memory lock gate requires POSIX descriptor/flock support",
)
def test_memory_gate_is_released_when_git_store_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent import memory

    def fail_git_store(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected GitStore failure")

    monkeypatch.setattr(memory, "GitStore", fail_git_store)

    with pytest.raises(RuntimeError, match="injected GitStore failure"):
        MemoryStore(tmp_path)

    (tmp_path / "memory").chmod(0o700)
    with WorkspaceMemoryAuthority(tmp_path):
        pass


@pytest.mark.parametrize("receipt_arrives", [False, True])
def test_windows_noop_gate_preserves_both_receipt_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_arrives: bool,
) -> None:
    from nanobot.agent import memory

    monkeypatch.setattr(memory, "_is_windows", lambda: True)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    legacy = memory_dir / "HISTORY.md"
    legacy.write_text("[2026-01-01 00:00] legacy history\n", encoding="utf-8")
    real_preflight = memory.require_memory_receipt_recovery
    preflight_calls = 0

    def record_preflight(path: Path) -> None:
        nonlocal preflight_calls
        preflight_calls += 1
        real_preflight(path)
        if receipt_arrives and preflight_calls == 1:
            _publish_receipt(path.parent)

    def unexpected_posix_gate(path: Path | str) -> object:
        pytest.fail("Windows constructor attempted to acquire the POSIX memory gate")

    monkeypatch.setattr(memory, "require_memory_receipt_recovery", record_preflight)
    monkeypatch.setattr(memory, "open_workspace_memory_lock", unexpected_posix_gate)

    if receipt_arrives:
        with pytest.raises(MemoryReceiptRecoveryRequiredError) as failure:
            MemoryStore(tmp_path)
        assert str(failure.value) == "memory recovery required"
        assert legacy.read_text(encoding="utf-8") == "[2026-01-01 00:00] legacy history\n"
        assert not (memory_dir / "history.jsonl").exists()
    else:
        store = MemoryStore(tmp_path)
        assert store.memory_dir == memory_dir
        assert not legacy.exists()
        assert (memory_dir / "history.jsonl").is_file()

    assert preflight_calls == 2


def test_unavailable_posix_gate_is_not_silently_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent import memory

    monkeypatch.setattr(memory, "_is_windows", lambda: False)

    def unavailable_gate(path: Path | str) -> object:
        raise storage.MemoryStorageUnavailableError()

    monkeypatch.setattr(memory, "open_workspace_memory_lock", unavailable_gate)

    with pytest.raises(storage.MemoryStorageUnavailableError):
        MemoryStore(tmp_path)

    assert tmp_path.is_dir()
    assert not (tmp_path / "memory").exists()
