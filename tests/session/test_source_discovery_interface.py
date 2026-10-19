"""Discovery uses qualified owner operations, never raw root/backend getters."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from nanobot.session import source_root as root_module
from nanobot.session import source_store
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source import SourceValidationError
from nanobot.session.source_discovery import (
    SourceDiscoveryCursor,
    SourceDiscoveryLimits,
    SourceDiscoveryRootChangedError,
)
from nanobot.session.source_root import SourceRootClosedError
from tests.session.test_source_discovery import _source_session


@pytest.fixture
def _source_authority(tmp_path: Path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    _source_session(backend, "source:one")
    authority = backend.source.create_source_root_authority()
    try:
        yield backend, authority
    finally:
        authority.close()


def test_backend_identity_operation_has_explicit_signature(_source_authority) -> None:
    backend, authority = _source_authority
    operation = getattr(authority, "matches_backend")
    signature = inspect.signature(operation)
    assert tuple(signature.parameters) == ("backend",)
    parameter = signature.parameters["backend"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty
    assert parameter.annotation == "JsonlSessionStore" and signature.return_annotation == "bool"


def test_backend_identity_operation_requires_the_exact_object(_source_authority) -> None:
    backend, authority = _source_authority
    operation = getattr(authority, "matches_backend")
    other = JsonlSessionStore(backend.workspace, sessions_root=backend.sessions_dir.parent)
    assert other is not backend and other.sessions_dir == backend.sessions_dir
    assert operation(backend) is True and operation(backend=other) is False
    authority.close()
    # Identity is deliberately not an admission/currentness assertion or getter.
    assert operation(backend) is True


def test_iterator_operation_has_explicit_signature(_source_authority) -> None:
    backend, authority = _source_authority
    operation = getattr(authority, "open_entry_iterator")
    signature = inspect.signature(operation)
    assert not signature.parameters
    assert signature.return_annotation == "_Iterator[os.DirEntry[str]]"


def test_iterator_operation_returns_the_real_descriptor_iterator(
    _source_authority, monkeypatch,
) -> None:
    backend, authority = _source_authority
    operation = getattr(authority, "open_entry_iterator")
    expected = {path.name for path in backend.sessions_dir.iterdir()}
    original = root_module.os.scandir
    calls, created = [], []

    def observed(descriptor):
        calls.append(descriptor)
        result = original(descriptor)
        created.append(result)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(root_module.os, "scandir", observed)
        iterator = operation()
    try:
        assert iterator is created[0] and len(calls) == 1 and calls[0] == authority._fd
        entries = list(iterator)
        assert {entry.name for entry in entries} == expected
        assert all(isinstance(entry, root_module.os.DirEntry) for entry in entries)
    finally:
        iterator.close()


@pytest.mark.parametrize("fault_kind", [OSError, ValueError, KeyboardInterrupt])
def test_iterator_operation_preserves_exact_scandir_faults(
    _source_authority, monkeypatch, fault_kind,
) -> None:
    backend, authority = _source_authority
    operation = getattr(authority, "open_entry_iterator")
    fault = fault_kind("discovery scan probe")
    calls = []

    def fail(descriptor):
        calls.append(descriptor)
        raise fault

    with monkeypatch.context() as patch:
        patch.setattr(root_module.os, "scandir", fail)
        with pytest.raises(fault_kind) as raised:
            operation()
    assert raised.value is fault and calls == [authority._fd]


def test_iterator_operation_rechecks_closed_authority_before_scanning(
    _source_authority, monkeypatch,
) -> None:
    backend, authority = _source_authority
    operation = getattr(authority, "open_entry_iterator")
    authority.close()

    def forbidden(*args, **kwargs):
        raise AssertionError("closed authority must fail before scandir")

    with monkeypatch.context() as patch:
        patch.setattr(root_module.os, "scandir", forbidden)
        with pytest.raises(SourceRootClosedError):
            operation()


@pytest.mark.asyncio
async def test_cursor_translates_owner_closing_after_the_path_precheck(
    _source_authority, monkeypatch,
) -> None:
    backend, authority = _source_authority
    cursor = SourceDiscoveryCursor(
        backend, limits=SourceDiscoveryLimits(20, 1, 500, 128), root_authority=authority,
    )
    original = cursor._check_root

    def close_after_check():
        original()
        authority.close()

    monkeypatch.setattr(cursor, "_check_root", close_after_check)
    try:
        with pytest.raises(SourceDiscoveryRootChangedError):
            cursor._open_iterator()
    finally:
        await cursor.aclose()


def test_source_json_callback_has_explicit_signature() -> None:
    operation = getattr(source_store, "reject_duplicate_source_keys")
    signature = inspect.signature(operation)
    assert tuple(signature.parameters) == ("pairs",)
    parameter = signature.parameters["pairs"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty
    assert parameter.annotation == "list[tuple[str, object]]"
    assert signature.return_annotation == "dict[str, object]"
    assert not hasattr(source_store, "__all__")


@pytest.mark.parametrize("raw", ['{"key":1,"key":2}', '{"metadata":{"key":1,"key":2}}'])
def test_source_json_callback_refuses_actual_nested_duplicate_keys(raw) -> None:
    operation = getattr(source_store, "reject_duplicate_source_keys")
    with pytest.raises(SourceValidationError, match="Duplicate source object key"):
        json.loads(raw, object_pairs_hook=operation)


def test_source_json_callback_dynamically_forwards_real_values(monkeypatch) -> None:
    operation = getattr(source_store, "reject_duplicate_source_keys")
    original = source_store._object_pairs
    pairs = [("name", "José"), ("metadata", {"seen": False})]
    calls, results = [], []

    def first(actual):
        assert actual is pairs
        calls.append("first")
        result = original(actual)
        results.append(result)
        return result

    def second(actual):
        assert actual is pairs
        calls.append("second")
        result = original(actual)
        results.append(result)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(source_store, "_object_pairs", first)
        a = operation(pairs)
        patch.setattr(source_store, "_object_pairs", second)
        b = operation(pairs=pairs)
    assert a is results[0] and b is results[1]
    assert a == b == {"name": "José", "metadata": {"seen": False}}
    assert a["metadata"] is b["metadata"] is pairs[1][1]
    assert calls == ["first", "second"]


@pytest.mark.parametrize("fault_kind", [OSError, ValueError, KeyboardInterrupt])
def test_source_json_callback_preserves_exact_faults(monkeypatch, fault_kind) -> None:
    operation = getattr(source_store, "reject_duplicate_source_keys")
    pairs = [("key", "value")]
    fault = fault_kind("JSON callback probe")
    calls = []

    def fail(actual):
        calls.append(actual)
        raise fault

    with monkeypatch.context() as patch:
        patch.setattr(source_store, "_object_pairs", fail)
        with pytest.raises(fault_kind) as raised:
            operation(pairs)
    assert raised.value is fault and len(calls) == 1 and calls[0] is pairs
