"""Native generator and closed-map preservation, without runtime activation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from nanobot.agent import memory_recovery_startup_admission as admission
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.tools import producer_adapters
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.search import GrepTool
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source_root import SourceRootAuthorityError, SourceRootClosedError
from tests.tools.test_producer_adapters import _receipt


def _backend(tmp_path: Path) -> JsonlSessionStore:
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")


def test_source_operation_yields_exact_authority_and_defers_close(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    authority = backend.source.create_source_root_authority()
    try:
        with authority.operation(backend) as yielded:
            assert yielded is authority
            yielded.require_current(backend)
            authority.close()
            assert not authority.closed
            with pytest.raises(SourceRootClosedError):
                authority.require_current(backend)
        assert authority.closed
    finally:
        authority.close()


def test_nested_source_operation_keeps_descriptor_until_outer_exit(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    authority = backend.source.create_source_root_authority()
    try:
        with authority.operation() as outer:
            with authority.operation(backend) as inner:
                assert outer is inner is authority
                authority.close()
                assert not authority.closed
            assert not authority.closed
        assert authority.closed
    finally:
        authority.close()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyError, asyncio.CancelledError])
def test_source_operation_propagates_body_exception_and_finishes_close(tmp_path, error_type):
    backend = _backend(tmp_path)
    authority = backend.source.create_source_root_authority()
    error = error_type("synthetic body refusal")
    try:
        with pytest.raises(error_type) as caught:
            with authority.operation() as yielded:
                assert yielded is authority
                authority.close()
                assert not authority.closed
                raise error
        assert caught.value is error
        assert authority.closed
    finally:
        authority.close()


def test_source_operation_rejects_distinct_backend_before_yield(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    other = _backend(tmp_path)
    authority = backend.source.create_source_root_authority()
    entered = []
    try:
        with pytest.raises(SourceRootAuthorityError, match="backend changed"):
            with authority.operation(other):
                entered.append(True)
        assert entered == []
        assert not authority.closed
        with authority.operation(backend) as yielded:
            assert yielded is authority
        authority.close()
        assert authority.closed
    finally:
        authority.close()


def test_startup_context_yields_real_gate_and_releases_it(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    sessions = tmp_path / "sessions"
    assert not workspace.exists()
    with admission.admitted_startup_mutation(workspace, sessions_root=sessions) as gate:
        assert gate is not None
        gate.require_current()
    with pytest.raises(storage.MemoryStorageError) as closed:
        gate.require_current()
    assert closed.value.code == "closed"
    with storage.open_existing_workspace_memory_lock(workspace) as reopened:
        reopened.require_current()
    assert not (workspace / "memory" / ".consolidation.pending.json").exists()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyError, asyncio.CancelledError])
def test_startup_context_preserves_body_error_and_releases_lock(tmp_path, error_type):
    workspace = tmp_path / "workspace"
    sessions = tmp_path / "sessions"
    error = error_type("synthetic body refusal")
    with pytest.raises(error_type) as caught:
        with admission.admitted_startup_mutation(workspace, sessions_root=sessions) as gate:
            assert gate is not None
            gate.require_current()
            raise error
    assert caught.value is error
    with pytest.raises(storage.MemoryStorageError) as closed:
        gate.require_current()
    assert closed.value.code == "closed"
    with storage.open_existing_workspace_memory_lock(workspace) as reopened:
        reopened.require_current()


def test_startup_context_pending_receipt_refuses_before_yield(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    memory = workspace / "memory"
    memory.mkdir(parents=True)
    pending = memory / ".consolidation.pending.json"
    pending.write_bytes(b"synthetic pending receipt")
    entered = []
    with pytest.raises(admission.StartupReceiptAdmissionError) as caught:
        with admission.admitted_startup_mutation(workspace, sessions_root=tmp_path / "sessions"):
            entered.append(True)
    assert entered == []
    assert str(caught.value) == "memory recovery required"
    assert caught.value.__cause__ is None
    assert pending.read_bytes() == b"synthetic pending receipt"


def test_startup_context_unsupported_clean_branch_yields_none_without_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only this module's branch selector changes, not native descriptor operations.
    monkeypatch.setattr(admission, "os", SimpleNamespace(name="unsupported"))
    workspace = tmp_path / "absent"
    with admission.admitted_startup_mutation(workspace, sessions_root=tmp_path / "sessions") as gate:
        assert gate is None
        assert not workspace.exists()
    assert not workspace.exists()


@pytest.mark.parametrize(
    "tool_type, producer",
    [(ReadFileTool, "builtin_read_file_text"), (GrepTool, "builtin_grep_file_text")],
)
@pytest.mark.parametrize("content", ["", "plain text", "Unicode: \N{PILE OF POO}"])
def test_closed_map_accepts_only_matching_exact_builtin_receipt(tmp_path, tool_type, producer, content):
    tool = tool_type(workspace=tmp_path)
    receipt = _receipt(content, producer=producer)
    assert producer_adapters.accepted_producer_receipt(tool, content, receipt)
    other = "builtin_grep_file_text" if producer == "builtin_read_file_text" else "builtin_read_file_text"
    assert not producer_adapters.accepted_producer_receipt(tool, content, _receipt(content, producer=other))
    assert not producer_adapters.accepted_producer_receipt(tool, content + "changed", receipt)


@pytest.mark.parametrize("tool_type", [ReadFileTool, GrepTool])
def test_closed_map_rejects_builtin_subclasses(tmp_path, tool_type):
    subclass = type("UnissuedSubclass", (tool_type,), {})
    tool = subclass(workspace=tmp_path)
    assert not producer_adapters.accepted_producer_receipt(tool, "text", _receipt("text"))


def test_closed_map_miss_does_not_read_plugin_like_metadata() -> None:
    class Lookalike:
        def __getattribute__(self, name: str):
            raise AssertionError("plugin metadata must not be read")

    assert not producer_adapters.accepted_producer_receipt(Lookalike(), "text", _receipt("text"))


def test_closed_map_remains_immutable_and_exact() -> None:
    registry = producer_adapters._PRODUCER_ADAPTERS
    assert type(registry) is MappingProxyType
    assert set(registry) == {ReadFileTool, GrepTool}
    with pytest.raises(TypeError):
        registry[Tool] = registry[ReadFileTool]
    assert set(registry) == {ReadFileTool, GrepTool}
