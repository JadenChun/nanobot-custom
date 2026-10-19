"""Quarantine metadata predicates never replace real lock/lease admission."""

from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from tests.agent.test_memory_recovery_quarantine_storage import _layout


def file_bytes(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("owner_type", [storage.WorkspaceMemoryLockGate, storage.WorkspaceMemoryAuthority])
@pytest.mark.parametrize("existing_only", [False, True])
def test_existing_layout_metadata_is_pure_and_does_not_open_a_closed_owner(
    tmp_path, owner_type, existing_only,
):
    workspace, _ = _layout(tmp_path)
    owner = owner_type(workspace, existing_only=existing_only)
    before = file_bytes(tmp_path)
    assert owner_type.uses_existing_layout_only(owner) is existing_only
    assert owner._workspace_fd is None
    assert file_bytes(tmp_path) == before
    if isinstance(owner, storage.WorkspaceMemoryLockGate):
        with pytest.raises(storage.MemoryStorageError):
            owner.require_current()
    else:
        with pytest.raises(storage.MemoryStorageError):
            with owner.operation():
                raise AssertionError("closed owner must not admit an operation")
    assert file_bytes(tmp_path) == before


def test_seal_barrier_observes_gate_policy_nonvirtually_before_current(tmp_path, monkeypatch):
    original = storage.WorkspaceMemoryLockGate.uses_existing_layout_only
    workspace, _ = _layout(tmp_path)
    trace = []

    def policy(value):
        trace.append("policy")
        return original(value)

    def forbidden(*args):
        raise AssertionError("policy rejection must precede current-lock checks")

    with storage.open_workspace_memory_lock(workspace) as gate:
        before = file_bytes(tmp_path)
        with monkeypatch.context() as patch:
            patch.setattr(storage.WorkspaceMemoryLockGate, "uses_existing_layout_only", policy)
            patch.setattr(gate, "uses_existing_layout_only", forbidden)
            patch.setattr(gate, "require_current", forbidden)
            with pytest.raises(quarantine.QuarantineError):
                quarantine.establish_durable_quarantine_seals(gate)
        assert trace == ["policy"]
        assert file_bytes(tmp_path) == before


def test_existing_quarantine_requires_authority_policy_before_factory(tmp_path, monkeypatch):
    original = storage.WorkspaceMemoryAuthority.uses_existing_layout_only
    workspace, binding = _layout(tmp_path)
    trace = []

    def policy(value):
        trace.append("policy")
        return original(value)

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected policy must not enter operation/store factory")

    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        before = file_bytes(tmp_path)
        with monkeypatch.context() as patch:
            patch.setattr(storage.WorkspaceMemoryAuthority, "uses_existing_layout_only", policy)
            patch.setattr(authority, "uses_existing_layout_only", forbidden)
            patch.setattr(operation, "require_current", forbidden)
            patch.setattr(quarantine.ReceiptQuarantineStore, "from_operation", classmethod(forbidden))
            with pytest.raises(quarantine.QuarantineError):
                quarantine.ReceiptQuarantineStore.from_existing_operation(authority, operation, binding)
        assert trace == ["policy"]
        assert file_bytes(tmp_path) == before


def test_operation_issuer_is_observed_before_comparison_callback(tmp_path, monkeypatch):
    compare = storage.WorkspaceMemoryOperation.matches_authority_observation
    workspace, _ = _layout(tmp_path)
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    other = storage.WorkspaceMemoryAuthority(other_workspace)
    with storage.open_workspace_memory(workspace) as authority:
        with authority.operation() as operation:
            selected = [other]
            trace = []

            def issuer(value):
                assert value is operation
                trace.append("issuer")
                selected[0] = authority
                return authority

            def observe():
                trace.append("comparison")
                return selected[0]

            before = file_bytes(tmp_path)
            with monkeypatch.context() as patch:
                patch.setattr(storage.WorkspaceMemoryOperation, "_authority", property(issuer), raising=False)
                assert compare(operation, observe)
            assert trace == ["issuer", "comparison"]
            assert file_bytes(tmp_path) == before
        assert compare(operation, lambda: authority)
        assert not compare(operation, lambda: other)
        with pytest.raises(storage.MemoryStorageError):
            operation.require_current()


def test_store_lock_check_preserves_changing_owner_with_genuine_operation(tmp_path, monkeypatch):
    workspace, binding = _layout(tmp_path)
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    other = storage.WorkspaceMemoryAuthority(other_workspace)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        store = quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding)
        store._authority = other
        original_current = operation.require_current
        trace = []

        def issuer(value):
            assert value is operation
            trace.append("issuer")
            store._authority = authority
            return authority

        def current():
            trace.append("current")
            original_current()

        def forbidden(*args):
            raise AssertionError("instance predicate override must not replace issuer comparison")

        before = file_bytes(tmp_path)
        with monkeypatch.context() as patch:
            patch.setattr(storage.WorkspaceMemoryOperation, "_authority", property(issuer), raising=False)
            patch.setattr(operation, "require_current", current)
            patch.setattr(operation, "matches_authority_observation", forbidden, raising=False)
            store._require_lock()
        assert trace[:2] == ["issuer", "current"]
        assert store._authority is authority
        assert file_bytes(tmp_path) == before


def test_foreign_authority_rejected_before_current_with_real_issued_operation(tmp_path, monkeypatch):
    workspace, binding = _layout(tmp_path)
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    other = storage.WorkspaceMemoryAuthority(other_workspace)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        store = quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding)
        store._authority = other
        before = file_bytes(tmp_path)

        def forbidden():
            raise AssertionError("foreign issuer must be rejected before current-lock checks")

        with monkeypatch.context() as patch:
            patch.setattr(operation, "require_current", forbidden)
            with pytest.raises(quarantine.QuarantineError):
                store._require_lock()
        assert file_bytes(tmp_path) == before


def test_closed_operation_identity_metadata_cannot_admit_a_quarantine_store(tmp_path):
    workspace, binding = _layout(tmp_path)
    with storage.open_workspace_memory(workspace) as authority:
        with authority.operation() as operation:
            store = quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding)
        before = file_bytes(tmp_path)
        with pytest.raises(quarantine.QuarantineError) as caught:
            store._require_lock()
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
        assert file_bytes(tmp_path) == before
