"""Native generator and closed-map preservation, without runtime activation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import MappingProxyType

import pytest

from kohaku.agent.tools import producer_adapters
from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.filesystem import ReadFileTool
from kohaku.agent.tools.search import GrepTool
from kohaku.session.manager import JsonlSessionStore
from kohaku.session.source_root import SourceRootAuthorityError, SourceRootClosedError
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
