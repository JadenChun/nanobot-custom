"""Child-only selection of child-produced artifacts for a trusted parent handoff."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.turn import ToolOutcome
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_grants import ArtifactHandoffCollector, ArtifactHandoffReceipt
from nanobot.session.execution_owner import ExecutionOwnershipError
from nanobot.session.source import SourceValidationError

if TYPE_CHECKING:
    from abc import ABCMeta

    class _ConstantToolMetadata(ABCMeta):
        # Model class-level value lookup, not runtime property descriptors.
        @property
        def name(cls) -> str:
            ...

        @property
        def description(cls) -> str:
            ...

        @property
        def parameters(cls) -> dict[str, object]:
            ...

        @property
        def read_only(cls) -> bool:
            ...
else:
    # Keep the existing runtime metaclass, abstractness and MRO unchanged.
    _ConstantToolMetadata = type(Tool)


class ArtifactHandoffTool(Tool, metaclass=_ConstantToolMetadata):
    """Select current-child evidence for an eventual, verified parent handoff."""

    _plugin_discoverable = False
    # Preserve runtime class constants while declaring the Tool instance contract.
    if TYPE_CHECKING:
        @property
        def name(self) -> str:
            ...

        @property
        def description(self) -> str:
            ...

        @property
        def read_only(self) -> bool:
            ...

        @property
        def parameters(self) -> dict[str, object]:
            ...
    else:
        name = "handoff_artifact"
        description = (
            "Select up to 16 artifacts produced by this child run for the parent after this child "
            "completes successfully. References received from the parent cannot be handed back."
        )
        read_only = True
        parameters = {
            "type": "object",
            "properties": {
                "artifact_references": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^artifact:[0-9a-f]{32}$"},
                    "minItems": 1,
                    "maxItems": 16,
                    "uniqueItems": True,
                },
            },
            "required": ["artifact_references"],
            "additionalProperties": False,
        }

    def __init__(self, collector: ArtifactHandoffCollector) -> None:
        self.collector = collector

    async def execute(self, **kwargs: Any) -> str:
        return ToolResult.error("Error: Artifact handoff requires admitted child execution")

    async def execute_with_context(
        self,
        context: Any,
        artifact_references: list[str],
        **kwargs: Any,
    ) -> ToolOutcome:
        try:
            receipt = self.collector.select(artifact_references)
            return ToolOutcome(
                content=json.dumps(
                    {"accepted": artifact_references}, ensure_ascii=False, separators=(",", ":")
                ),
                handoff_receipt=receipt,
            )
        except (ArtifactValidationError, ExecutionOwnershipError, SourceValidationError, OSError):
            return ToolOutcome(
                content=ToolResult.error("Error: Artifact handoff is unavailable or access was denied"),
                stop_reason="tool_error",
            )

    def accepts_receipt(self, receipt: object) -> bool:
        return isinstance(receipt, ArtifactHandoffReceipt) and receipt.belongs_to(self.collector)


__all__ = ["ArtifactHandoffTool"]
