"""Crash and hostile-layout tests for the private receipt quarantine index."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction_terminal as terminal
from nanobot.session.checked_source import open_checked_recovery_session
from nanobot.session.recovery_locator import locate_recovery_candidate
from tests.agent.test_memory_recovery_transaction_prepare import _SESSION_KEY, _WORKSPACE_ID
from tests.agent.test_memory_recovery_transaction_terminal import (
    _reconcile,
    _setup,
    _transaction_chain,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor protocol")
RECEIPT = b'{"unrecognized":"retain exactly", "x": 1}\n'
DIGEST = hashlib.sha256(RECEIPT).hexdigest()


def test_r2_quarantine_second_comparison_preserves_changing_owner_order():
    first = object.__new__(storage.WorkspaceMemoryAuthority)
    second = object.__new__(storage.WorkspaceMemoryAuthority)
    trace = []
    store = object.__new__(quarantine.ReceiptQuarantineStore)
    store._gate = None
    store._authority = first

    class Operation(storage.WorkspaceMemoryOperation):
        @property
        def _authority(self):
            trace.append("operation-owner")
            store._authority = second
            return second

        def require_current(self):
            trace.append("current")

    store._operation = object.__new__(Operation)
    store._require_lock()
    assert trace == ["operation-owner", "current"]
    assert store._authority is second
    # The excluded eager-store-read predicate observes first, not second.
    store._authority = first
    owner_before_operation_read = store._authority
    assert store._operation._authority is not owner_before_operation_read


@pytest.mark.parametrize("fault", [RuntimeError("owner"), KeyboardInterrupt("owner"),
                                   storage.MemoryStorageError("closed")])
@pytest.mark.parametrize("point", ["owner", "current"])
def test_r2_quarantine_first_comparison_fault_order(point, fault, tmp_path):
    _, binding = _layout(tmp_path)
    authority = object.__new__(storage.WorkspaceMemoryAuthority)
    trace = []

    class Operation(storage.WorkspaceMemoryOperation):
        @property
        def _authority(self):
            trace.append("owner")
            if point == "owner":
                raise fault
            return authority

        def require_current(self):
            trace.append("current")
            raise fault

    operation = object.__new__(Operation)
    if point == "current" and isinstance(fault, storage.MemoryStorageError):
        with pytest.raises(quarantine.QuarantineError):
            quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding)
    else:
        with pytest.raises(type(fault)) as raised:
            quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding)
        assert raised.value is fault
    assert trace == (["owner"] if point == "owner" else ["owner", "current"])
    trace.clear()
    with pytest.raises(quarantine.QuarantineError):
        quarantine.ReceiptQuarantineStore.from_operation(object(), operation, binding)
    assert not trace


def _layout(tmp_path: Path) -> tuple[Path, quarantine.QuarantineBinding]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = workspace / "memory"
    memory.mkdir(mode=0o700)
    (memory / ".consolidation.pending.json").write_bytes(RECEIPT)
    (memory / ".consolidation.pending.json").chmod(0o600)
    sessions = tmp_path / "sessions"
    sessions.mkdir(mode=0o700)
    return workspace, quarantine.QuarantineBinding(
        workspace, sessions, "a" * 32, "b" * 64, "c" * 64, DIGEST, len(RECEIPT)
    )


def _publish_v1_committed(store: quarantine.ReceiptQuarantineStore) -> None:
    """Model Q1 structural residue, which must never satisfy the v2 barrier."""
    records, digests = store._prefix(2)
    store._publish("committed.json", store._record(
        "committed", digests[1], records[0]["receipt_stat"], records[1]["copy_stat"],
    ))


def _real_layout(tmp_path: Path) -> tuple[Path, Path, Path, quarantine.QuarantineBinding]:
    workspace, sessions, _, receipt, _ = _setup(tmp_path)
    _reconcile(workspace, sessions)
    raw = receipt.read_bytes()
    _, records = _transaction_chain(workspace)
    binding = quarantine.QuarantineBinding(
        workspace, sessions, _WORKSPACE_ID, records[0].identity, records[-1].digest,
        hashlib.sha256(raw).hexdigest(), len(raw),
    )
    return workspace, sessions, receipt, binding


def _seal_real_receipt(workspace: Path, sessions: Path, receipt: Path,
                       binding: quarantine.QuarantineBinding) -> str:
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        raw = receipt.read_bytes()
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
            store.prepare(raw)
            store.copy_receipt()
            receipt.unlink()
            absence = store.prove_durable_absence()
            with (quarantine.open_validated_quarantine_copy(
                authority, operation, binding, require_phase="copy-ready", require_pending_absent=True,
            ) as copy, open_checked_recovery_session(operation, candidate) as checked):
                proof = terminal.prove_terminal_complete_under_locks(
                    authority, operation, checked, _WORKSPACE_ID, copy,
                )
                store.commit(absence, proof)
                return proof.semantic_seal_digest


def test_real_terminal_seal_parity_and_durable_barrier(tmp_path: Path) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    seal = _seal_real_receipt(workspace, sessions, receipt, binding)
    state = quarantine.inspect_receipt_quarantine(workspace)
    assert state.committed_identities == state.sealed_identities == (binding.transaction_identity,)
    record = (workspace / "memory" / ".recovery" / "receipt-quarantine"
              / binding.transaction_identity / "committed.json")
    value = json.loads(record.read_bytes())
    assert value["version"] == 2 and value["semantic_seal_digest"] == seal
    assert "unrecognized" not in record.read_text()
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        with quarantine.open_validated_quarantine_copy(
            authority, operation, binding, require_phase="committed", require_pending_absent=True,
        ) as copy:
            assert copy.semantic_seal_digest == seal
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        durable = quarantine.establish_durable_quarantine_seals(gate)
    assert durable.sealed_identities == (binding.transaction_identity,)
    assert durable.semantic_seal_digests == (seal,)


def test_targeted_existing_binding_is_bootstrap_only_and_never_creates(tmp_path: Path) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
            store.prepare(receipt.read_bytes())
            store.copy_receipt()
    receipt.unlink()
    with storage.open_existing_workspace_memory(workspace) as authority:
        with authority.operation() as operation:
            observed = quarantine.read_existing_copy_ready_binding(
                authority, operation, sessions, binding.transaction_identity,
            )
            assert observed == binding
            with quarantine.open_validated_quarantine_copy(
                authority, operation, observed, require_phase="copy-ready",
                require_pending_absent=True,
            ) as copy:
                assert copy.raw_bytes
            with pytest.raises(quarantine.QuarantineError):
                quarantine.read_existing_copy_ready_binding(
                    authority, operation, sessions, "a" * 64,
                )
    assert not (workspace / "memory" / ".recovery" / "receipt-quarantine" / ("a" * 64)).exists()


@pytest.mark.parametrize("damage", ["uppercase", "extra", "wrong-chain", "wrong-version"])
def test_strict_v2_record_rejects_malformed_seal_or_chain(tmp_path: Path, damage: str) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    record = (workspace / "memory" / ".recovery" / "receipt-quarantine"
              / binding.transaction_identity / "committed.json")
    value = json.loads(record.read_bytes())
    if damage == "uppercase":
        value["semantic_seal_digest"] = "A" * 64
    elif damage == "extra":
        value["duplicate_claim"] = "secret"
    elif damage == "wrong-chain":
        value["predecessor_digest"] = "0" * 64
    else:
        value["version"] = 1
    record.write_bytes(quarantine._canonical(value))
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        with pytest.raises(quarantine.QuarantineError) as error:
            quarantine.establish_durable_quarantine_seals(gate)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("boundary", ["copy", "record", "identity", "index", "recovery", "memory", "workspace"])
def test_barrier_fsync_failure_is_content_free_and_retriable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    original = os.fsync
    index = workspace / "memory" / ".recovery" / "receipt-quarantine"
    target = {
        "copy": index / binding.transaction_identity / "receipt.bin",
        "record": index / binding.transaction_identity / "committed.json",
        "identity": index / binding.transaction_identity,
        "index": index,
        "recovery": workspace / "memory" / ".recovery",
        "memory": workspace / "memory",
        "workspace": workspace,
    }[boundary]

    def injected(fd: int) -> None:
        if os.fstat(fd).st_ino == target.stat().st_ino:
            raise OSError("injected secret fsync")
        original(fd)

    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        monkeypatch.setattr(os, "fsync", injected)
        with pytest.raises(quarantine.QuarantineError) as error:
            quarantine.establish_durable_quarantine_seals(gate)
        assert "secret" not in str(error.value)
        monkeypatch.undo()
        assert quarantine.establish_durable_quarantine_seals(gate).sealed_identities == (
            binding.transaction_identity,
        )


def test_v1_structural_committed_is_not_durable_seal(tmp_path: Path) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
            store.copy_receipt()
            (workspace / "memory" / ".consolidation.pending.json").unlink()
            _publish_v1_committed(store)
    state = quarantine.inspect_receipt_quarantine(workspace)
    assert state.committed_identities == (binding.transaction_identity,)
    assert state.sealed_identities == ()
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        with quarantine.open_validated_quarantine_copy(
            authority, operation, binding, require_phase="committed", require_pending_absent=True,
        ) as copy:
            assert copy.semantic_seal_digest is None
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        with pytest.raises(quarantine.QuarantineError):
            quarantine.establish_durable_quarantine_seals(gate)


def test_commit_rejects_forged_cross_operation_stale_and_live_pending(tmp_path: Path) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    raw = receipt.read_bytes()
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
            store.prepare(raw)
            store.copy_receipt()
            receipt.unlink()
            absence = store.prove_durable_absence()
            with (quarantine.open_validated_quarantine_copy(
                authority, operation, binding, require_phase="copy-ready", require_pending_absent=True,
            ) as copy, open_checked_recovery_session(operation, candidate) as checked):
                proof = terminal.prove_terminal_complete_under_locks(
                    authority, operation, checked, _WORKSPACE_ID, copy,
                )
                for forged in (object(), object.__new__(terminal.TerminalCompleteProof)):
                    with pytest.raises(quarantine.QuarantineError) as error:
                        store.commit(absence, forged)  # type: ignore[arg-type]
                    assert str(error.value) == str(quarantine.QuarantineError())
                with authority.operation() as other_operation:
                    with quarantine.ReceiptQuarantineStore.from_operation(
                        authority, other_operation, binding,
                    ) as other_store:
                        other_absence = other_store.prove_durable_absence()
                        with pytest.raises(quarantine.QuarantineError):
                            other_store.commit(other_absence, proof)
                receipt.write_bytes(raw)
                receipt.chmod(0o600)
                with pytest.raises(quarantine.QuarantineError):
                    store.commit(absence, proof)
                receipt.unlink()
                with pytest.raises(quarantine.QuarantineError):
                    store.commit(absence, proof)  # transient pending invalidated the fsynced absence
                # Reproof is current only until the checked session/target drifts.
                absence = store.prove_durable_absence()
                memory = workspace / "memory"
                target = memory / "MEMORY.md"
                target.write_bytes(target.read_bytes() + b"changed\n")
                with pytest.raises(quarantine.QuarantineError):
                    store.commit(absence, proof)
    assert not (workspace / "memory" / ".recovery" / "receipt-quarantine"
                / binding.transaction_identity / "committed.json").exists()


@pytest.mark.parametrize("boundary", ["create", "write", "file-fsync", "parent-fsync"])
def test_sealed_commit_publication_fault_blocks_until_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    raw = receipt.read_bytes()
    original_open, original_write, original_fsync = os.open, os.write, os.fsync
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
            store.prepare(raw)
            store.copy_receipt()
            receipt.unlink()
            absence = store.prove_durable_absence()
            with (quarantine.open_validated_quarantine_copy(
                authority, operation, binding, require_phase="copy-ready", require_pending_absent=True,
            ) as copy, open_checked_recovery_session(operation, candidate) as checked):
                proof = terminal.prove_terminal_complete_under_locks(
                    authority, operation, checked, _WORKSPACE_ID, copy,
                )
                target_fd: int | None = None

                def intercept_open(path: str, *args: object, **kwargs: object) -> int:
                    nonlocal target_fd
                    if path == "committed.json" and boundary == "create":
                        raise OSError("injected secret create")
                    fd = original_open(path, *args, **kwargs)
                    if path == "committed.json" and len(args) > 0 and args[0] & os.O_CREAT:
                        target_fd = fd
                    return fd

                def intercept_write(fd: int, data: bytes) -> int:
                    if fd == target_fd and boundary == "write":
                        raise OSError("injected secret write")
                    return original_write(fd, data)

                def intercept_fsync(fd: int) -> None:
                    if (target_fd is not None and
                            ((fd == target_fd and boundary == "file-fsync")
                             or (fd == store._identity_fd and boundary == "parent-fsync"))):
                        raise OSError("injected secret fsync")
                    original_fsync(fd)

                original_recheck = terminal.TerminalCompleteProof.require_current

                def recheck_then_inject(self: terminal.TerminalCompleteProof) -> None:
                    original_recheck(self)
                    monkeypatch.setattr(os, "open", intercept_open)
                    monkeypatch.setattr(os, "write", intercept_write)
                    monkeypatch.setattr(os, "fsync", intercept_fsync)

                monkeypatch.setattr(terminal.TerminalCompleteProof, "require_current", recheck_then_inject)
                with pytest.raises(quarantine.QuarantineError) as error:
                    store.commit(absence, proof)
                assert "secret" not in str(error.value)
                if boundary in {"file-fsync", "parent-fsync"}:
                    assert target_fd is not None, "publication did not reach exclusive create"
                monkeypatch.undo()
    record = (workspace / "memory" / ".recovery" / "receipt-quarantine"
              / binding.transaction_identity / "committed.json")
    if boundary in {"file-fsync", "parent-fsync"}:
        assert json.loads(record.read_bytes())["semantic_seal_digest"] == proof.semantic_seal_digest
        with storage.open_existing_workspace_memory_lock(workspace) as gate:
            assert quarantine.establish_durable_quarantine_seals(gate).sealed_identities == (
                binding.transaction_identity,
            )
    else:
        with storage.open_existing_workspace_memory_lock(workspace) as gate:
            with pytest.raises(quarantine.QuarantineError):
                quarantine.establish_durable_quarantine_seals(gate)


@pytest.mark.parametrize("attack", ["copy-bytes", "copy-symlink", "record-missing", "second-incomplete", "pending"])
def test_barrier_rejects_any_unsafe_identity_or_pending(
    tmp_path: Path, attack: str,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    index = workspace / "memory" / ".recovery" / "receipt-quarantine"
    identity = index / binding.transaction_identity
    if attack == "copy-bytes":
        (identity / "receipt.bin").write_bytes(b"tampered")
    elif attack == "copy-symlink":
        (identity / "receipt.bin").unlink()
        (identity / "receipt.bin").symlink_to(tmp_path / "other")
    elif attack == "record-missing":
        (identity / "committed.json").unlink()
    elif attack == "second-incomplete":
        (index / ("f" * 64)).mkdir(mode=0o700)
    else:
        receipt.write_bytes(b"pending")
        receipt.chmod(0o600)
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        with pytest.raises(quarantine.QuarantineError):
            quarantine.establish_durable_quarantine_seals(gate)


def test_barrier_detects_in_call_identity_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    index = workspace / "memory" / ".recovery" / "receipt-quarantine"
    identity = index / binding.transaction_identity
    original = quarantine._read
    swapped = False

    def swap(fd: int, name: str, limit: int):
        nonlocal swapped
        result = original(fd, name, limit)
        if name == "receipt.bin" and not swapped:
            swapped = True
            identity.rename(index / ("e" * 64))
            identity.mkdir(mode=0o700)
        return result

    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        monkeypatch.setattr(quarantine, "_read", swap)
        with pytest.raises(quarantine.QuarantineError):
            quarantine.establish_durable_quarantine_seals(gate)
    assert swapped


def test_clean_absence_does_not_write(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert quarantine.inspect_receipt_quarantine(missing).clean
    assert not missing.exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert quarantine.inspect_receipt_quarantine(workspace).clean
    assert list(workspace.iterdir()) == []


def test_existing_only_barrier_refuses_unsealed_and_does_not_create(tmp_path: Path) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
            store.copy_receipt()
            (workspace / "memory" / ".consolidation.pending.json").unlink()
            with pytest.raises(quarantine.QuarantineError):
                store.commit(store.prove_durable_absence(), object())
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        with pytest.raises(quarantine.QuarantineError):
            quarantine.establish_durable_quarantine_seals(gate)
    missing = tmp_path / "missing"
    with pytest.raises(storage.MemoryStorageError):
        with storage.open_existing_workspace_memory_lock(missing):
            pass
    assert not missing.exists()


def test_empty_existing_barrier_does_not_create_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with storage.open_workspace_memory_lock(workspace):
        pass  # Establish only the existing lock; no recovery layout.
    original_open = os.open

    def no_create(path: str, flags: int, *args: object, **kwargs: object) -> int:
        assert not flags & (os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_RDWR)
        return original_open(path, flags, *args, **kwargs)

    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        monkeypatch.setattr(os, "open", no_create)
        monkeypatch.setattr(os, "mkdir", lambda *a, **kw: pytest.fail("unexpected mkdir"))
        monkeypatch.setattr(os, "write", lambda *a, **kw: pytest.fail("unexpected write"))
        assert quarantine.establish_durable_quarantine_seals(gate).sealed_identities == ()
        monkeypatch.undo()
    assert sorted(p.name for p in workspace.iterdir()) == [".memory-recovery.lock"]


def test_pending_and_empty_or_unknown_index_are_never_clean(tmp_path: Path) -> None:
    workspace, _ = _layout(tmp_path)
    assert not quarantine.inspect_receipt_quarantine(workspace).clean
    index = workspace / "memory" / ".recovery" / "receipt-quarantine"
    index.mkdir(parents=True, mode=0o700)
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)  # unsafe 0755 ancestor
    (workspace / "memory" / ".recovery").chmod(0o700)
    assert quarantine.inspect_receipt_quarantine(workspace).incomplete
    (index / "unknown").write_bytes(b"x")
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)


def test_sequence_and_durable_absence_proof(tmp_path: Path) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
            assert quarantine.inspect_receipt_quarantine(workspace).incomplete
            with pytest.raises(quarantine.QuarantineError):
                store.commit(True, object())  # type: ignore[arg-type]
            store.copy_receipt()
            assert quarantine.inspect_receipt_quarantine(workspace).incomplete
            with pytest.raises(quarantine.QuarantineError):
                store.prove_durable_absence()
            (workspace / "memory" / ".consolidation.pending.json").unlink()
            store.prove_durable_absence()
            _publish_v1_committed(store)
    state = quarantine.inspect_receipt_quarantine(workspace)
    assert not state.clean and not state.incomplete
    assert state.committed_identities == ("b" * 64,)
    scope = workspace / "memory" / ".recovery" / "receipt-quarantine" / ("b" * 64)
    assert (scope / "receipt.bin").read_bytes() == RECEIPT
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in scope.iterdir())


@pytest.mark.parametrize("phase", ["copy-ready", "committed"])
def test_authority_store_and_typed_reader_recheck_exact_evidence(
    tmp_path: Path, phase: str,
) -> None:
    workspace, binding = _layout(tmp_path)
    pending = workspace / "memory" / ".consolidation.pending.json"
    original = quarantine._stat_tuple(pending.stat())
    with storage.open_workspace_memory(workspace) as authority:
        with authority.operation() as operation:
            with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
                store.prepare(RECEIPT)
                store.copy_receipt()
                pending.unlink()
                if phase == "committed":
                    _publish_v1_committed(store)
            with quarantine.open_validated_quarantine_copy(
                authority, operation, binding, require_phase=phase, require_pending_absent=True,
            ) as copy:
                assert copy.raw_bytes == RECEIPT
                assert copy.original_stat == original
                assert copy.copy_stat != original
                assert copy.phase == phase and copy.binding == binding
                copy.require_current()
                pending.write_bytes(RECEIPT)
                pending.chmod(0o600)
                with pytest.raises(quarantine.QuarantineError):
                    copy.require_current()
                pending.unlink()
            with pytest.raises(quarantine.QuarantineError):
                copy.require_current()
            with pytest.raises(quarantine.QuarantineError):
                copy.__enter__()


def test_authority_composition_rejects_foreign_closed_and_cross_thread(tmp_path: Path) -> None:
    workspace, binding = _layout(tmp_path)
    (tmp_path / "other").mkdir()
    with storage.open_workspace_memory(workspace) as authority:
        with authority.operation() as operation:
            with storage.open_workspace_memory(tmp_path / "other") as foreign:
                with pytest.raises(quarantine.QuarantineError):
                    quarantine.ReceiptQuarantineStore.from_operation(foreign, operation, binding)
            with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
                errors: list[Exception] = []

                def check() -> None:
                    try:
                        store.prepare(RECEIPT)
                    except Exception as error:
                        errors.append(error)

                worker = threading.Thread(target=check)
                worker.start()
                worker.join()
                assert len(errors) == 1 and isinstance(errors[0], quarantine.QuarantineError)
                store.prepare(RECEIPT)
                store.copy_receipt()
            with quarantine.open_validated_quarantine_copy(
                authority, operation, binding, require_phase="copy-ready",
            ) as copy:
                errors.clear()
                worker = threading.Thread(target=lambda: _check_cross_thread(copy, errors))
                worker.start()
                worker.join()
                assert len(errors) == 1 and isinstance(errors[0], quarantine.QuarantineError)
        with pytest.raises(quarantine.QuarantineError):
            quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding)


def _check_cross_thread(copy: quarantine.ValidatedQuarantineCopy, errors: list[Exception]) -> None:
    try:
        copy.require_current()
    except Exception as error:
        errors.append(error)


@pytest.mark.parametrize("boundary", ["identity", "sessions"])
def test_typed_reader_rejects_ancestor_replacement(tmp_path: Path, boundary: str) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory(workspace) as authority:
        with authority.operation() as operation:
            with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
                store.prepare(RECEIPT)
                store.copy_receipt()
            with quarantine.open_validated_quarantine_copy(
                authority, operation, binding, require_phase="copy-ready",
            ) as copy:
                path = (workspace / "memory" / ".recovery" / "receipt-quarantine" / ("b" * 64)
                        if boundary == "identity" else binding.sessions_root)
                path.rename(path.with_name(path.name + "-old"))
                path.mkdir(mode=0o700)
                with pytest.raises(quarantine.QuarantineError):
                    copy.require_current()


def test_typed_reader_has_no_publication_or_durability_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory(workspace) as authority:
        with authority.operation() as operation:
            with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
                store.prepare(RECEIPT)
                store.copy_receipt()
            index = workspace / "memory" / ".recovery" / "receipt-quarantine"
            before = sorted(p.name for p in index.iterdir())

            def forbidden(*_args: object, **_kwargs: object) -> None:
                pytest.fail("targeted reader attempted a write or fsync")

            with monkeypatch.context() as patch:
                patch.setattr(quarantine.os, "fsync", forbidden)
                patch.setattr(quarantine.os, "mkdir", forbidden)
                with quarantine.open_validated_quarantine_copy(
                    authority, operation, binding, require_phase="copy-ready",
                ) as copy:
                    assert copy.raw_bytes == RECEIPT
                    with pytest.raises(quarantine.QuarantineError):
                        copy.prepare(RECEIPT)
            assert sorted(p.name for p in index.iterdir()) == before


@pytest.mark.parametrize("name", ["receipt.bin", "copy-ready.json"])
def test_typed_reader_fails_on_copy_record_and_path_swaps(tmp_path: Path, name: str) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory(workspace) as authority:
        with authority.operation() as operation:
            with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
                store.prepare(RECEIPT)
                store.copy_receipt()
            scope = workspace / "memory" / ".recovery" / "receipt-quarantine" / ("b" * 64)
            with quarantine.open_validated_quarantine_copy(
                authority, operation, binding, require_phase="copy-ready",
            ) as copy:
                path = scope / name
                raw = path.read_bytes()
                path.unlink()
                path.write_bytes(raw)
                path.chmod(0o600)
                with pytest.raises(quarantine.QuarantineError):
                    copy.require_current()
            # A fresh reader cannot admit a replaced copy (recorded copy stat).
            # Canonical records do not self-record their own inode, so an
            # identical record replaced *before* a fresh lease is admissible.
            if name == "receipt.bin":
                with pytest.raises(quarantine.QuarantineError):
                    with quarantine.open_validated_quarantine_copy(
                        authority, operation, binding, require_phase="copy-ready",
                    ):
                        pass


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "mode", "bytes", "unknown"])
def test_committed_corruption_fails_closed(tmp_path: Path, attack: str) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
            store.copy_receipt()
            (workspace / "memory" / ".consolidation.pending.json").unlink()
            _publish_v1_committed(store)
    scope = workspace / "memory" / ".recovery" / "receipt-quarantine" / ("b" * 64)
    copy = scope / "receipt.bin"
    if attack == "symlink":
        copy.unlink()
        copy.symlink_to(tmp_path / "other")
    elif attack == "hardlink":
        os.link(copy, scope / "other")
    elif attack == "mode":
        copy.chmod(0o644)
    elif attack == "bytes":
        copy.write_bytes(b"changed")
    else:
        (scope / "other").write_bytes(b"x")
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)


@pytest.mark.parametrize("boundary", ["copy-file", "copy-parent", "absence-parent"])
def test_interrupted_durability_never_claims_next_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
            original = os.fsync
            index = workspace / "memory" / ".recovery" / "receipt-quarantine" / ("b" * 64)

            def fail_selected(fd: int) -> None:
                info = os.fstat(fd)
                if ((boundary == "copy-file" and stat.S_ISREG(info.st_mode)
                     and info.st_size == len(RECEIPT))
                    or (boundary == "copy-parent" and stat.S_ISDIR(info.st_mode)
                        and info.st_ino == index.stat().st_ino)
                    or (boundary == "absence-parent" and stat.S_ISDIR(info.st_mode)
                        and info.st_ino == (workspace / "memory").stat().st_ino)):
                    raise OSError("injected private path/bytes must not escape")
                original(fd)

            if boundary == "absence-parent":
                store.copy_receipt()
                (workspace / "memory" / ".consolidation.pending.json").unlink()
            monkeypatch.setattr(os, "fsync", fail_selected)
            with pytest.raises(quarantine.QuarantineError) as error:
                if boundary == "absence-parent":
                    store.prove_durable_absence()
                else:
                    store.copy_receipt()
            assert "injected" not in str(error.value)
            assert not (index / "committed.json").exists()
            if boundary != "absence-parent":
                assert not (index / "copy-ready.json").exists()
    assert quarantine.inspect_receipt_quarantine(workspace).incomplete


@pytest.mark.parametrize("damage", ["duplicate", "noncanonical", "wrong-predecessor", "wrong-workspace", "nan"])
def test_strict_record_parser_rejects_corruption(tmp_path: Path, damage: str) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
    record = (workspace / "memory" / ".recovery" / "receipt-quarantine" / ("b" * 64)
              / "prepared.json")
    original = record.read_bytes()
    if damage == "duplicate":
        altered = original.replace(b'"version":1', b'"version":1,"version":1')
    elif damage == "noncanonical":
        altered = b" " + original
    elif damage == "wrong-predecessor":
        altered = original.replace(b'"predecessor_digest":null', b'"predecessor_digest":"' + b"0" * 64 + b'"')
    elif damage == "wrong-workspace":
        altered = original.replace(str(workspace).encode(), str(tmp_path).encode())
    else:
        altered = original.replace(b'"version":1', b'"version":NaN')
    assert altered != original
    record.write_bytes(altered)
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)


@pytest.mark.parametrize(("field", "invalid"), [
    ("version", True), ("phase", []), ("predecessor_digest", "unexpected"),
    ("workspace", 7), ("sessions_root", []), ("workspace_id", None),
    ("transaction_identity", 7), ("complete_digest", False), ("receipt_digest", []),
    ("workspace_stat", [1, 2, 3, 4, True]), ("sessions_root_stat", [1, 2]),
    ("receipt_size", True), ("receipt_stat", [0] * 8 + ["bad"]),
    ("copy_name", "receipt.bin"), ("copy_stat", [0] * 9),
])
def test_record_decoder_rejects_invalid_field_types(tmp_path: Path, field: str, invalid: object) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
    record = (workspace / "memory" / ".recovery" / "receipt-quarantine"
              / binding.transaction_identity / "prepared.json")
    value = json.loads(record.read_bytes())
    value[field] = invalid
    record.write_bytes(quarantine._canonical(value))
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)
    assert (workspace / "memory" / ".consolidation.pending.json").read_bytes() == RECEIPT


def test_multiple_committed_identities_remain_discoverable(tmp_path: Path) -> None:
    workspace, binding = _layout(tmp_path)
    for tx in ("b" * 64, "d" * 64):
        pending = workspace / "memory" / ".consolidation.pending.json"
        if not pending.exists():
            pending.write_bytes(RECEIPT)
            pending.chmod(0o600)
        with storage.open_workspace_memory_lock(workspace) as gate:
            with quarantine.ReceiptQuarantineStore(
                gate, replace(binding, transaction_identity=tx)
            ) as store:
                store.prepare(RECEIPT)
                store.copy_receipt()
                pending.unlink()
                _publish_v1_committed(store)
    assert quarantine.inspect_receipt_quarantine(workspace).committed_identities == (
        "b" * 64, "d" * 64,
    )
    assert len(quarantine.inspect_receipt_quarantine(workspace, expected=binding).committed_identities) == 2


def test_present_evidence_without_descriptor_support_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _ = _layout(tmp_path)
    monkeypatch.setattr(quarantine, "_support", quarantine._fail)
    assert quarantine.inspect_receipt_quarantine(tmp_path / "absent").clean
    empty = tmp_path / "empty"
    (empty / "memory").mkdir(parents=True, mode=0o700)
    assert quarantine.inspect_receipt_quarantine(empty).clean
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)


def test_streamed_index_discovers_committed_beyond_256_with_incomplete_residue(
    tmp_path: Path,
) -> None:
    workspace, binding = _layout(tmp_path)
    index = workspace / "memory" / ".recovery" / "receipt-quarantine"
    for identity in ("b" * 64, "d" * 64):
        pending = workspace / "memory" / ".consolidation.pending.json"
        if not pending.exists():
            pending.write_bytes(RECEIPT)
            pending.chmod(0o600)
        with storage.open_workspace_memory_lock(workspace) as gate:
            with quarantine.ReceiptQuarantineStore(
                gate, replace(binding, transaction_identity=identity)
            ) as store:
                store.prepare(RECEIPT)
                store.copy_receipt()
                pending.unlink()
                _publish_v1_committed(store)
    for number in range(257):
        (index / f"{number:064x}").mkdir(mode=0o700)
    state = quarantine.inspect_receipt_quarantine(workspace)
    assert state.incomplete and not state.clean
    assert state.committed_identities == ("b" * 64, "d" * 64)
    assert len(list(index.iterdir())) == 259


@pytest.mark.parametrize("location", ["unknown-index", "unknown-identity", "orphan-copy"])
def test_unknown_or_orphan_symlink_fails_with_fixed_error(tmp_path: Path, location: str) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            if location == "unknown-identity":
                store.prepare(RECEIPT)
    index = workspace / "memory" / ".recovery" / "receipt-quarantine"
    identity = index / ("b" * 64)
    if location == "unknown-index":
        (index / "unexpected").symlink_to(tmp_path / "unsafe")
    elif location == "unknown-identity":
        (identity / "unexpected").symlink_to(tmp_path / "unsafe")
    else:
        (identity / "receipt.bin").symlink_to(tmp_path / "unsafe")
    with pytest.raises(quarantine.QuarantineError) as error:
        quarantine.inspect_receipt_quarantine(workspace)
    assert str(error.value) == str(quarantine.QuarantineError())


@pytest.mark.parametrize("transition", ["memory", "index", "pending-clean", "pending-committed", "index-stat"])
def test_inspection_observes_in_call_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = workspace / "memory"
    if transition != "memory":
        memory.mkdir(mode=0o700)
    if transition in {"index", "pending-committed", "index-stat"}:
        (memory / ".recovery").mkdir(mode=0o700)
    if transition in {"pending-committed", "index-stat"}:
        (memory / ".consolidation.pending.json").write_bytes(RECEIPT)
        (memory / ".consolidation.pending.json").chmod(0o600)
        sessions = tmp_path / "sessions"
        sessions.mkdir(mode=0o700)
        binding = quarantine.QuarantineBinding(
            workspace, sessions, "a" * 32, "b" * 64, "c" * 64, DIGEST, len(RECEIPT)
        )
        with storage.open_workspace_memory_lock(workspace) as gate:
            with quarantine.ReceiptQuarantineStore(gate, binding) as store:
                store.prepare(RECEIPT)
                store.copy_receipt()
                (memory / ".consolidation.pending.json").unlink()
                _publish_v1_committed(store)
    original = quarantine._assert_inspection_bindings
    injected = False

    def inject(*args: object, **kwargs: object) -> None:
        nonlocal injected
        if not injected:
            injected = True
            if transition == "memory":
                memory.mkdir(mode=0o700)
            elif transition == "index":
                (memory / ".recovery" / "receipt-quarantine").mkdir(mode=0o700)
            elif transition in {"pending-clean", "pending-committed"}:
                (memory / ".consolidation.pending.json").write_bytes(RECEIPT)
                (memory / ".consolidation.pending.json").chmod(0o600)
            else:
                (memory / ".recovery" / "receipt-quarantine" / "unexpected").mkdir(mode=0o700)
        original(*args, **kwargs)

    monkeypatch.setattr(quarantine, "_assert_inspection_bindings", inject)
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)
    assert injected


def test_absent_workspace_created_during_inspection_is_not_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "new-workspace"
    original = quarantine._open_absolute
    first = True

    def appear_after_absence(path: Path) -> int | None:
        nonlocal first
        if path == workspace and first:
            first = False
            workspace.mkdir(mode=0o700)
            return None
        return original(path)

    monkeypatch.setattr(quarantine, "_open_absolute", appear_after_absence)
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)
    assert not first


@pytest.mark.parametrize("boundary", ["create", "write", "reread", "replace"])
def test_copy_install_boundaries_leave_visible_unclaimable_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
            index = workspace / "memory" / ".recovery" / "receipt-quarantine" / ("b" * 64)
            original_open, original_write, original_read = os.open, os.write, quarantine._read

            def fail_create(name: str, *args: object, **kwargs: object) -> int:
                if name == "receipt.bin":
                    raise OSError("private injected create")
                return original_open(name, *args, **kwargs)

            def fail_write(fd: int, data: bytes) -> int:
                if stat.S_ISREG(os.fstat(fd).st_mode) and len(data) == len(RECEIPT):
                    raise OSError("private injected write")
                return original_write(fd, data)

            def fail_reread(fd: int, name: str, limit: int):
                if name == "receipt.bin":
                    if boundary == "replace":
                        (index / name).unlink()
                        (index / name).symlink_to(tmp_path / "unsafe")
                        return original_read(fd, name, limit)
                    return None
                return original_read(fd, name, limit)

            if boundary == "create":
                monkeypatch.setattr(os, "open", fail_create)
            elif boundary == "write":
                monkeypatch.setattr(os, "write", fail_write)
            else:
                monkeypatch.setattr(quarantine, "_read", fail_reread)
            with pytest.raises(quarantine.QuarantineError) as error:
                store.copy_receipt()
            assert "injected" not in str(error.value)
            assert (index / "prepared.json").exists()
            assert not (index / "copy-ready.json").exists()
            assert not (index / "committed.json").exists()
    monkeypatch.undo()
    if boundary in {"write", "replace"}:
        with pytest.raises(quarantine.QuarantineError):
            quarantine.inspect_receipt_quarantine(workspace)
    else:
        assert quarantine.inspect_receipt_quarantine(workspace).incomplete


@pytest.mark.parametrize("field", ["workspace_id", "sessions_root_stat", "receipt_digest", "copy_stat"])
def test_canonical_but_wrong_binding_fails_closed(tmp_path: Path, field: str) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
            store.copy_receipt()
    scope = workspace / "memory" / ".recovery" / "receipt-quarantine" / ("b" * 64)
    record = scope / ("copy-ready.json" if field == "copy_stat" else "prepared.json")
    value = json.loads(record.read_bytes())
    if field == "workspace_id":
        value[field] = "f" * 32
    elif field == "sessions_root_stat":
        value[field][1] += 1
    elif field == "receipt_digest":
        value[field] = "f" * 64
    else:
        value[field][1] += 1
    record.write_bytes(quarantine._canonical(value))
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)


def test_inspection_detects_identity_directory_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(RECEIPT)
            store.copy_receipt()
            (workspace / "memory" / ".consolidation.pending.json").unlink()
            _publish_v1_committed(store)
    index = workspace / "memory" / ".recovery" / "receipt-quarantine"
    original = quarantine._read
    swapped = False

    def swap_after_read(fd: int, name: str, limit: int):
        nonlocal swapped
        result = original(fd, name, limit)
        if name == "receipt.bin" and not swapped:
            swapped = True
            (index / ("b" * 64)).rename(index / ("e" * 64))
            (index / ("b" * 64)).mkdir(mode=0o700)
        return result

    monkeypatch.setattr(quarantine, "_read", swap_after_read)
    with pytest.raises(quarantine.QuarantineError):
        quarantine.inspect_receipt_quarantine(workspace)
    assert swapped


@pytest.mark.parametrize("phase", ["prepared", "copy-ready", "committed"])
@pytest.mark.parametrize("boundary", ["create", "write", "file-fsync", "parent-fsync"])
def test_record_publication_interruption_retains_private_blocking_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, boundary: str,
) -> None:
    workspace, binding = _layout(tmp_path)
    receipt = workspace / "memory" / ".consolidation.pending.json"
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            if phase != "prepared":
                store.prepare(RECEIPT)
            if phase == "committed":
                store.copy_receipt()
                receipt.unlink()
                store.prove_durable_absence()
            identity = workspace / "memory" / ".recovery" / "receipt-quarantine" / ("b" * 64)
            name = f"{phase}.json"
            target_fd: int | None = None
            original_open, original_write, original_fsync = os.open, os.write, os.fsync

            def intercept_open(path: str, *args: object, **kwargs: object) -> int:
                nonlocal target_fd
                if path == name and boundary == "create":
                    raise OSError("injected record create")
                descriptor = original_open(path, *args, **kwargs)
                if path == name:
                    target_fd = descriptor
                return descriptor

            def intercept_write(fd: int, data: bytes) -> int:
                if fd == target_fd and boundary == "write":
                    raise OSError("injected record write")
                return original_write(fd, data)

            def intercept_fsync(fd: int) -> None:
                if (target_fd is not None and
                    ((fd == target_fd and boundary == "file-fsync") or
                     (fd == store._identity_fd and boundary == "parent-fsync"))):
                    raise OSError("injected record fsync")
                original_fsync(fd)

            monkeypatch.setattr(os, "open", intercept_open)
            monkeypatch.setattr(os, "write", intercept_write)
            monkeypatch.setattr(os, "fsync", intercept_fsync)
            with pytest.raises(quarantine.QuarantineError) as error:
                if phase == "prepared":
                    store.prepare(RECEIPT)
                elif phase == "copy-ready":
                    store.copy_receipt()
                else:
                    if phase == "committed":
                        _publish_v1_committed(store)
            assert "injected" not in str(error.value)
            assert (identity / "receipt.bin").exists() == (phase == "committed" or phase == "copy-ready")
            if (identity / name).exists():
                assert stat.S_IMODE((identity / name).stat().st_mode) == 0o600
            assert stat.S_IMODE(identity.stat().st_mode) == 0o700
    monkeypatch.undo()
    try:
        state = quarantine.inspect_receipt_quarantine(workspace)
    except quarantine.QuarantineError:
        pass  # A partial direct-exclusive record is fixed-error blocking residue.
    else:
        assert not state.clean
        if phase != "committed" or boundary in {"create", "write"}:
            assert state.incomplete
        # A fully written committed record after fsync failure remains only
        # structural evidence; there is no startup admission in this module.
    if phase == "committed":
        assert (identity / "receipt.bin").read_bytes() == RECEIPT
