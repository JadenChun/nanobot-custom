"""Shared receipt-binding checks retain their distinct policies and fault seams."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_conversion_apply as application
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction_prepare as preparation

_RECEIPT = ".consolidation.pending.json"
_BINDINGS = (
    (application, "require_conversion_receipt_binding", "_bound_receipt",
     application.ConversionApplyError, "receipt_snapshot"),
    (preparation, "require_transaction_receipt_binding", "_receipt_bound",
     preparation.MemoryRecoveryTransactionPreparationError, "issued"),
)


def _write_receipt(workspace: Path, content: bytes = b"receipt fixture") -> Path:
    receipt = workspace / "memory" / _RECEIPT
    receipt.write_bytes(content)
    receipt.chmod(0o600)
    return receipt


@pytest.mark.parametrize("module,public,private,error_type,snapshot_name", _BINDINGS)
def test_supported_receipt_binding_signatures_and_exports(
    module, public, private, error_type, snapshot_name,
) -> None:
    operation = getattr(module, public)
    signature = inspect.signature(operation)
    assert tuple(signature.parameters) == ("authority", snapshot_name)
    assert all(parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               and parameter.default is inspect.Parameter.empty
               for parameter in signature.parameters.values())
    assert signature.return_annotation == "None"
    assert public in module.__all__


@pytest.mark.parametrize("module,public,private,error_type,snapshot_name", _BINDINGS)
def test_dynamic_forwarding_uses_real_bound_authorities(
    tmp_path, monkeypatch, module, public, private, error_type, snapshot_name,
) -> None:
    operation = getattr(module, public)
    original = getattr(module, private)
    calls = []

    def first(authority, snapshot):
        calls.append(("first", authority, snapshot))
        return original(authority, snapshot)

    def second(authority, snapshot):
        calls.append(("second", authority, snapshot))
        return original(authority, snapshot)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        _write_receipt(tmp_path)
        snapshot = authority.snapshot_target(_RECEIPT)
        monkeypatch.setattr(module, private, first)
        assert operation(authority, snapshot) is None
        monkeypatch.setattr(module, private, second)
        assert operation(authority=authority, **{snapshot_name: snapshot}) is None
        assert calls == [("first", authority, snapshot), ("second", authority, snapshot)]
        assert authority.snapshot_target(_RECEIPT).stat_identity == snapshot.stat_identity


@pytest.mark.parametrize("module,public,private,error_type,snapshot_name", _BINDINGS)
@pytest.mark.parametrize("kind", [OSError, RuntimeError, KeyboardInterrupt])
def test_forwarding_preserves_private_fault_identity(
    tmp_path, monkeypatch, module, public, private, error_type, snapshot_name, kind,
) -> None:
    operation = getattr(module, public)
    fault = kind("binding probe")
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        _write_receipt(tmp_path)
        snapshot = authority.snapshot_target(_RECEIPT)

        def fail(actual_authority, actual_snapshot):
            assert actual_authority is authority and actual_snapshot is snapshot
            raise fault

        monkeypatch.setattr(module, private, fail)
        with pytest.raises(kind) as raised:
            operation(authority, snapshot)
        assert raised.value is fault


@pytest.mark.parametrize("module,public,private,error_type,snapshot_name", _BINDINGS)
def test_forwarding_preserves_owner_refusal_identity(
    tmp_path, monkeypatch, module, public, private, error_type, snapshot_name,
) -> None:
    operation = getattr(module, public)
    fault = error_type()
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        _write_receipt(tmp_path)
        snapshot = authority.snapshot_target(_RECEIPT)

        def fail(actual_authority, actual_snapshot):
            assert actual_authority is authority and actual_snapshot is snapshot
            raise fault

        monkeypatch.setattr(module, private, fail)
        with pytest.raises(error_type) as raised:
            operation(authority, snapshot)
        assert raised.value is fault


@pytest.mark.parametrize("module,public,private,error_type,snapshot_name", _BINDINGS)
def test_real_current_receipt_binding_does_not_mutate_files(
    tmp_path, module, public, private, error_type, snapshot_name,
) -> None:
    operation = getattr(module, public)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        receipt = _write_receipt(tmp_path)
        snapshot = authority.snapshot_target(_RECEIPT)
        before = receipt.read_bytes(), receipt.lstat()
        assert operation(authority, snapshot) is None
        assert (receipt.read_bytes(), receipt.lstat()) == before


@pytest.mark.parametrize("module,public,private,error_type,snapshot_name", _BINDINGS)
@pytest.mark.parametrize("change", ["content", "inode", "absence"])
def test_real_receipt_binding_refuses_changed_receipts(
    tmp_path, module, public, private, error_type, snapshot_name, change,
) -> None:
    operation = getattr(module, public)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        receipt = _write_receipt(tmp_path)
        snapshot = authority.snapshot_target(_RECEIPT)
        if change == "content":
            receipt.write_bytes(b"changed receipt")
        elif change == "inode":
            replacement = tmp_path / "memory" / "replacement"
            replacement.write_bytes(receipt.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, receipt)
        else:
            receipt.unlink()
        with pytest.raises(error_type) as raised:
            operation(authority, snapshot)
        assert raised.value.__cause__ is None


@pytest.mark.parametrize("module,public,private,error_type,snapshot_name", _BINDINGS)
def test_real_closed_authority_is_still_refused(
    tmp_path, module, public, private, error_type, snapshot_name,
) -> None:
    operation = getattr(module, public)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        _write_receipt(tmp_path)
        snapshot = authority.snapshot_target(_RECEIPT)
    with pytest.raises(storage.MemoryStorageError) as raised:
        operation(authority, snapshot)
    assert raised.value.code == "closed"


@pytest.mark.parametrize("invalid", [None, "receipt", object()])
def test_conversion_binding_retains_noncopy_nonsnapshot_refusal(tmp_path, invalid) -> None:
    operation = getattr(application, "require_conversion_receipt_binding")
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(application.ConversionApplyError):
            operation(authority, invalid)


def test_receipt_binding_variants_keep_distinct_absence_policies(tmp_path: Path) -> None:
    conversion_binding = getattr(application, "require_conversion_receipt_binding")
    transaction_binding = getattr(preparation, "require_transaction_receipt_binding")
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        absent = authority.snapshot_target(_RECEIPT)
        assert absent.content is None and absent.stat_identity is None
        # A conversion binding comparison is not a receipt parsing/authorization grant.
        assert conversion_binding(authority, absent) is None
        with pytest.raises(preparation.MemoryRecoveryTransactionPreparationError):
            transaction_binding(authority, absent)
        assert not (tmp_path / "memory" / _RECEIPT).exists()
