"""Closed, audited producer receipt adapters.

Only this module decides which concrete built-ins may attach producer evidence.
Tool names, plugin metadata, and subclass relationships are intentionally not
inputs to that decision: those are mutable extension points, while evidence
requires a fixed implementation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.read_provenance import (
    FilesystemReadProducer,
    FilesystemReadReceipt,
    parse_read_receipt,
)
from nanobot.agent.tools.search import GrepTool


@dataclass(frozen=True, slots=True)
class _ProducerAdapter:
    """One reviewed implementation/receipt combination.

    The descriptor stays private so adding a producer requires changing this
    module and its tests rather than registering an arbitrary tool at runtime.
    """

    producer_id: FilesystemReadProducer
    receipt_type: type[FilesystemReadReceipt]

    def accepts(self, content: object, receipt: object) -> bool:
        if type(content) is not str or type(receipt) is not self.receipt_type:
            return False
        try:
            parsed = parse_read_receipt(receipt.to_dict())
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            parsed == receipt
            and parsed.producer == self.producer_id
            and parsed.matches(content)
        )


# This is deliberately private and immutable.  New producers require an
# explicit adapter audit; no tool property, class name, plugin registration,
# or runtime flag can add an entry.
_PRODUCER_ADAPTERS: Final = MappingProxyType({
    ReadFileTool: _ProducerAdapter("builtin_read_file_text", FilesystemReadReceipt),
    GrepTool: _ProducerAdapter("builtin_grep_file_text", FilesystemReadReceipt),
})


def accepted_producer_receipt(tool: Tool, content: object, receipt: object) -> bool:
    """Return whether a concrete built-in produced valid scalar text evidence."""
    adapter = _PRODUCER_ADAPTERS.get(type(tool))
    return adapter is not None and adapter.accepts(content, receipt)
