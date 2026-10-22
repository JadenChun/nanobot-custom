"""Closed built-in producer receipt registry tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.producer_adapters import accepted_producer_receipt
from nanobot.agent.tools.read_provenance import (
    FilesystemReadProducer,
    FilesystemReadReceipt,
    parse_read_receipt,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome
from nanobot.session.artifact import FilesystemArtifactOrigin


def _receipt(content: str, *, producer: FilesystemReadProducer = "builtin_read_file_text") -> FilesystemReadReceipt:
    encoded = content.encode("utf-8")
    return FilesystemReadReceipt(
        FilesystemArtifactOrigin("filesystem", "workspace", None, "evidence.txt"),
        "a" * 64,
        len(encoded),
        hashlib.sha256(encoded).hexdigest(),
        producer=producer,
    )


class _CustomTool(Tool):
    def __init__(self, outcome: ToolOutcome, *, name: str = "read_file") -> None:
        self._outcome = outcome
        self._name = name
        self.producer_id = "builtin_read_file_text"
        self.trusted = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "test-only producer lookalike"

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: object) -> ToolOutcome:
        return self._outcome


class _ReadFileSubclass(ReadFileTool):
    async def execute(self, **kwargs: object) -> ToolOutcome:
        return _outcome("subclass result")


def _outcome(content: str, *, receipt: FilesystemReadReceipt | None = None,
             stop_reason: str | None = None, policy_metadata: dict[str, object] | None = None,
             ) -> ToolOutcome:
    return ToolOutcome(
        content=content,
        stop_reason=stop_reason,
        policy_metadata={} if policy_metadata is None else policy_metadata,
        producer_receipt=receipt if receipt is not None else _receipt(content),
    )


@pytest.mark.asyncio
async def test_exact_read_file_is_the_only_initial_registered_producer(tmp_path):
    content = "plain Unicode: \N{PILE OF POO}"
    tool = ReadFileTool(workspace=tmp_path)
    outcome = _outcome(content)
    tool.execute = _CustomTool(outcome).execute  # type: ignore[method-assign]

    accepted = await ToolRegistry.execute_prepared(tool, {})

    assert accepted.producer_receipt == outcome.producer_receipt
    assert accepted_producer_receipt(tool, content, outcome.producer_receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        _CustomTool(_outcome("fake name"), name="read_file"),
        _CustomTool(_outcome("custom plugin"), name="custom_read_file"),
    ],
)
async def test_name_or_plugin_like_tools_cannot_claim_builtin_producer(tool):
    original = await tool.execute()

    accepted = await ToolRegistry.execute_prepared(tool, {})

    assert accepted.content == original.content
    assert accepted.producer_receipt is None
    assert not accepted_producer_receipt(tool, original.content, original.producer_receipt)


@pytest.mark.asyncio
async def test_read_file_subclass_cannot_claim_builtin_producer(tmp_path):
    tool = _ReadFileSubclass(workspace=tmp_path)

    accepted = await ToolRegistry.execute_prepared(tool, {})

    assert accepted.content == "subclass result"
    assert accepted.producer_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt, content",
    [
        (_receipt("wrong producer", producer="builtin_grep_file_text"), "wrong producer"),
        (replace(_receipt("malformed"), root_binding="malformed"), "malformed"),
        (_receipt("original"), "altered"),
        (_receipt("\N{GRINNING FACE}"), "\N{SMILING FACE WITH OPEN MOUTH}"),
    ],
)
async def test_invalid_producer_evidence_drops_only_the_producer_authority(tmp_path, receipt, content):
    tool = ReadFileTool(workspace=tmp_path)
    original = _outcome(content, receipt=receipt, stop_reason="stop", policy_metadata={"keep": "value"})
    tool.execute = _CustomTool(original).execute  # type: ignore[method-assign]

    accepted = await ToolRegistry.execute_prepared(tool, {})

    assert accepted.content == content
    assert accepted.producer_receipt is None
    assert accepted.stop_reason == "stop"
    assert accepted.policy_metadata == {"keep": "value"}


@pytest.mark.asyncio
async def test_tool_errors_strip_producer_authority_and_keep_error_metadata(tmp_path):
    content = ToolResult.error("Error: expected tool failure")
    outcome = ToolOutcome(content, policy_metadata={"reason": "expected"}, producer_receipt=_receipt(str(content)))
    tool = ReadFileTool(workspace=tmp_path)
    tool.execute = _CustomTool(outcome).execute  # type: ignore[method-assign]

    accepted = await ToolRegistry.execute_prepared(tool, {})

    assert accepted.content == content
    assert accepted.stop_reason == "tool_error"
    assert accepted.policy_metadata == {"reason": "expected"}
    assert accepted.producer_receipt is None


def test_receipt_parser_accepts_the_reserved_future_producer_without_changing_wire_shape():
    receipt = _receipt("future", producer="builtin_grep_file_text")

    raw = receipt.to_dict()

    assert parse_read_receipt(raw) == receipt
    assert raw == {
        "source_access": {
            "kind": "filesystem",
            "access_class": "workspace",
            "repository": None,
            "relative_path": "evidence.txt",
        },
        "root_binding": "a" * 64,
        "byte_length": len("future".encode("utf-8")),
        "sha256": hashlib.sha256(b"future").hexdigest(),
        "schema_version": 1,
        "producer": "builtin_grep_file_text",
        "block_index": None,
    }
