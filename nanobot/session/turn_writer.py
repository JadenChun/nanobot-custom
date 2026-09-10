"""Trusted per-invocation source streams sharing one admitted execution owner."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.turn import ToolOutcome
from nanobot.session.execution_owner import ExecutionOwner
from nanobot.session.source import SourceValidationError, canonical_events, new_source_event
from nanobot.session.source_store import SourceSessionStore, SourceSnapshot


def _persisted_content(value: Any) -> Any:
    """Retain text evidence without newly persisting inline images or reasoning."""
    if not isinstance(value, list):
        return value
    retained = []
    for block in value:
        if isinstance(block, dict):
            if block.get('type') in {'thinking', 'reasoning', 'redacted_thinking', 'reasoning_content'}:
                continue
            image = block.get('image_url')
            if (block.get('type') == 'image_url' and isinstance(image, dict)
                    and isinstance(image.get('url'), str) and image['url'].startswith('data:image/')):
                retained.append({'type': 'text', 'text': '[Image data unavailable in persisted source.]'})
                continue
        retained.append(block)
    return retained


@dataclass
class _Writes:
    snapshot: SourceSnapshot
    failed: bool = False
    pending: dict[str, Any] | None = None
    committed: list[dict[str, Any]] = field(default_factory=list)


class SourceTurnWriter:
    """Allocate identities before writes; never replay a failed write or effect."""

    def __init__(self, store: SourceSessionStore, owner: ExecutionOwner, *, run_id: str,
                 parent_stream_id: str | None = None, writes: _Writes | None = None):
        self.store, self.owner, self.run_id = store, owner, run_id
        self.stream_id = uuid.uuid4().hex
        self.parent_stream_id = parent_stream_id
        self._writes = writes or _Writes(owner.read_snapshot())
        self._batch_callback: Callable[[KnownToolResult], Awaitable[str]] | None = None

    def fail(self) -> None:
        """Stop every stream after a mandatory runner or recording failure."""
        self._writes.failed = True

    def child_stream(self) -> SourceTurnWriter:
        return SourceTurnWriter(self.store, self.owner, run_id=self.run_id,
                                parent_stream_id=self.stream_id, writes=self._writes)

    @property
    def committed_events(self) -> list[dict[str, Any]]:
        return deepcopy([e for e in self._writes.committed if e.get('_stream_id') == self.stream_id])

    def _append(self, payload: dict[str, Any], *, execution_ids: list[str] | None = None,
                execution_id: str | None = None, source_origin: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._writes.failed:
            raise SourceValidationError('Source turn recording failed')
        try:
            self.owner.validate_dispatch()
            clean = deepcopy(payload)
            for key in ('reasoning_content', 'thinking_blocks', '_meta', '_stream_id', '_parent_stream_id', '_source_origin'):
                clean.pop(key, None)
            if 'content' in clean:
                clean['content'] = _persisted_content(clean['content'])
            event = new_source_event(clean, run_id=self.run_id)
            if source_origin is not None:
                event['_source_origin'] = deepcopy(source_origin)
            event['_stream_id'] = self.stream_id
            if self.parent_stream_id is not None:
                event['_parent_stream_id'] = self.parent_stream_id
            if execution_ids is not None:
                event['_execution_ids'] = execution_ids
            if execution_id is not None:
                event['_execution_id'] = execution_id
            self._writes.pending = event
            current = self.owner.read_snapshot()
            # Metadata may advance independently. Source changes must belong to
            # this invocation's shared coordinator, including its child streams.
            if current.events != self._writes.snapshot.events:
                raise SourceValidationError('Source turn changed outside admitted writer')
            committed = self.store.append_source_events(self.owner.key, current.state, [event], owner=self.owner)
        except BaseException:
            self._writes.failed = True
            raise
        self._writes.snapshot = committed
        self._writes.committed.append(event)
        self._writes.pending = None
        return deepcopy(event)

    def append_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get('role') not in {'user', 'assistant'} or 'tool_calls' in payload:
            self._writes.failed = True
            raise SourceValidationError('Invalid source turn message')
        return self._append(payload)

    def begin_tool_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        calls = payload.get('tool_calls')
        if payload.get('role') != 'assistant' or not isinstance(calls, list) or not calls:
            self._writes.failed = True
            raise SourceValidationError('Invalid source tool batch')
        ids = [uuid.uuid4().hex for _ in calls]
        event = self._append(payload, execution_ids=ids)
        declared = deepcopy(calls)
        completed: set[int] = set()

        async def record(value: KnownToolResult) -> str:
            try:
                index = value.ordinal
                if (type(index) is not int or not 0 <= index < len(declared) or index in completed
                        or value.tool_call_id != declared[index].get('id')
                        or value.name != declared[index].get('function', {}).get('name')):
                    self._writes.failed = True
                    raise SourceValidationError('Invalid source result binding')
                raw = value.raw_result.content if isinstance(value.raw_result, ToolOutcome) else value.raw_result
                if isinstance(raw, str):
                    raw = str(raw)
                canonical_events([{'content': raw}])
                origin = None
                if value.producer_receipt is not None:
                    from nanobot.agent.tools.read_provenance import parse_read_receipt

                    try:
                        receipt = parse_read_receipt(value.producer_receipt.to_dict())
                    except (ValueError, AttributeError, TypeError):
                        receipt = None
                    if receipt is not None and value.dispatched and value.event.get('status') == 'ok' and receipt.matches(raw):
                        origin = receipt.to_dict()
                    else:
                        logger.debug('Ignoring unmatched filesystem producer receipt')
                committed = self._append({'role': 'tool' , 'tool_call_id': value.tool_call_id, 'name': value.name,
                              'content': raw, '_dispatched': value.dispatched,
                              '_outcome_status': value.event.get('status', '')}, execution_id=ids[index],
                              source_origin=origin)
                completed.add(index)
                return committed['_event_id']
            except BaseException:
                self._writes.failed = True
                raise

        self._batch_callback = record
        return event

    async def record_tool_result(self, value: KnownToolResult) -> None:
        await self.record_tool_result_event(value)

    async def record_tool_result_event(self, value: KnownToolResult) -> str:
        """Return the exact durable identity without changing callback compatibility."""
        if self._batch_callback is None:
            self._writes.failed = True
            raise SourceValidationError('Source result has no committed intent')
        return await self._batch_callback(value)

    def project_history(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Detached root history with provenance, independent of archival cursors."""
        return self.project_events(self.owner.read_snapshot().events)

    @staticmethod
    def project_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        from nanobot.session.source_projection import project_events

        return project_events(events)
