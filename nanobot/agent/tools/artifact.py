"""Explicitly admitted artifact retrieval; never auto-enable a private reader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.turn import ToolOutcome
from nanobot.session.artifact import ArtifactValidationError, parse_artifact_reference
from nanobot.session.artifact_reader import ArtifactReader, ArtifactReadResult
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


@dataclass(frozen=True)
class ArtifactRetrievalReceipt:
    reference: str
    start_offset: int
    next_offset: int
    sha256: str

    def matches(self, text: object) -> bool:
        return isinstance(text, str) and hashlib.sha256(text.encode('utf-8')).hexdigest() == self.sha256

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': 1, **asdict(self)}


def parse_retrieval_receipt(raw: object) -> ArtifactRetrievalReceipt:
    if not isinstance(raw, dict):
        raise ArtifactValidationError('Invalid artifact retrieval receipt')
    raw = cast(dict[object, object], raw)
    if (set(raw) != {'schema_version', 'reference', 'start_offset',
            'next_offset', 'sha256'} or type(raw['schema_version']) is not int
            or raw['schema_version'] != 1 or type(raw['start_offset']) is not int
            or type(raw['next_offset']) is not int or not 0 <= raw['start_offset'] <= raw['next_offset']
            or not isinstance(raw['sha256'], str) or len(raw['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in raw['sha256'])):
        raise ArtifactValidationError('Invalid artifact retrieval receipt')
    parse_artifact_reference(raw['reference'])
    return ArtifactRetrievalReceipt(cast(str, raw['reference']), raw['start_offset'], raw['next_offset'], raw['sha256'])


def bounded_artifact_response(result: ArtifactReadResult) -> tuple[str, ArtifactReadResult]:
    """Cap the complete UTF-8 JSON envelope, including escaping and metadata."""
    maximum = 16_384
    original = result
    low, high = 0, len(original.text)
    while low < high:
        count = (low + high + 1) // 2
        text = original.text[:count]
        candidate = ArtifactReadResult(original.reference, text, original.start_offset,
            original.start_offset + len(text.encode('utf-8')), original.eof and count == len(original.text))
        payload = json.dumps(asdict(candidate), ensure_ascii=False, separators=(',', ':'))
        if len(payload.encode('utf-8')) <= maximum:
            low = count
        else:
            high = count - 1
    text = original.text[:low]
    result = ArtifactReadResult(original.reference, text, original.start_offset,
        original.start_offset + len(text.encode('utf-8')), original.eof and low == len(original.text))
    payload = json.dumps(asdict(result), ensure_ascii=False, separators=(',', ':'))
    if len(payload.encode('utf-8')) > maximum:
        raise ArtifactValidationError('Artifact response exceeds output budget')
    return payload, result


class ReadArtifactTool(Tool, metaclass=_ConstantToolMetadata):
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
        name = 'read_artifact'
        description = ('Read a stored text artifact by opaque reference. Byte offsets skip initial partial '
                       'UTF-8 characters; trailing partial characters remain for the next page. '
                       'Use next_offset to continue. max_bytes must be 4–16384. '
                       'The complete JSON response is capped at 16384 UTF-8 bytes.')
        read_only = True
        parameters = {'type': 'object', 'properties': {
            'reference': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0},
            'max_bytes': {'type': 'integer', 'minimum': 4, 'maximum': 16384}},
            'required': ['reference'], 'additionalProperties': False}

    def __init__(self, reader: ArtifactReader):
        self.reader = reader

    async def execute(self, **kwargs: Any) -> str:
        return ToolResult.error('Error: Artifact retrieval requires admitted execution')

    async def execute_with_context(self, context: Any, reference: str, offset: int = 0,
                                   max_bytes: int = 8192, **kwargs: Any) -> ToolOutcome:
        try:
            result = self.reader.read(reference, offset=offset, max_bytes=max_bytes)
            payload, result = bounded_artifact_response(result)
            receipt = ArtifactRetrievalReceipt(result.reference, result.start_offset,
                result.next_offset, hashlib.sha256(payload.encode('utf-8')).hexdigest())
            return ToolOutcome(content=payload, retrieval_receipt=receipt)
        except (OSError, SourceValidationError, ExecutionOwnershipError, UnicodeError):
            return ToolOutcome(content=ToolResult.error('Error: Artifact is unavailable or access was denied'),
                               stop_reason='tool_error')
