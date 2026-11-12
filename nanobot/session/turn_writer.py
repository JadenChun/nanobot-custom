"""Trusted per-invocation source streams sharing one admitted execution owner."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from loguru import logger

from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.tools.filesystem_effect import FilesystemEffectFact
from nanobot.agent.turn import ToolOutcome
from nanobot.session.execution_owner import ExecutionOwner
from nanobot.session.pending_followups import PendingFollowupRecord
from nanobot.session.source import (
    SourceValidationError,
    canonical_events,
    new_source_event,
)
from nanobot.session.source_store import SourceSessionStore, SourceSnapshot


def _intent_arguments_digest(value: object) -> str:
    """Hash normalized JSON arguments without using provider call IDs as authority."""
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, UnicodeError, RecursionError):
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_tool_result_identity(value: object) -> tuple[object, object]:
    """Return the legacy result identity for a source-validated tool call."""
    if not isinstance(value, dict):
        raise AttributeError(f"'{type(value).__name__}' object has no attribute 'get'")
    # The source event was canonically validated before the writer called this helper.
    call = cast(dict[str, object], value)
    function_value = call.get("function", {})
    if not isinstance(function_value, dict):
        raise AttributeError(
            f"'{type(function_value).__name__}' object has no attribute 'get'"
        )
    function = cast(dict[str, object], function_value)
    return call.get("id"), function.get("name")


def _persisted_content(value: Any) -> Any:
    """Retain text evidence without newly persisting inline images or reasoning."""
    if not isinstance(value, list):
        return value
    retained: list[object] = []
    for block in cast(list[object], value):
        if isinstance(block, dict):
            if cast(dict[object, object], block).get('type') in {'thinking', 'reasoning', 'redacted_thinking', 'reasoning_content'}:
                continue
            image = cast(dict[object, object], block).get('image_url')
            if (cast(dict[object, object], block).get('type') == 'image_url' and isinstance(image, dict)
                    and isinstance(cast(dict[object, object], image).get('url'), str)
                    and cast(str, cast(dict[object, object], image)['url']).startswith('data:image/')):
                retained.append({'type': 'text', 'text': '[Image data unavailable in persisted source.]'})
                continue
        retained.append(cast(object, block))
    return retained


@dataclass
class _Writes:
    snapshot: SourceSnapshot
    failed: bool = False
    pending: dict[str, Any] | None = None
    committed: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SourceInjectionBinding:
    """Writer-minted proof joining a projection to one committed source event."""

    event_id: str
    followup_id: str | None
    message: dict[str, object]


@dataclass(frozen=True, slots=True)
class SourceToolIntentBinding:
    """Writer-minted proof for one ordinal in a durable assistant tool batch."""

    event_id: str
    execution_id: str
    ordinal: int
    run_id: str
    stream_id: str
    tool_name: str
    tool_call_id: str
    arguments_digest: str


class SourceTurnWriter:
    """Allocate identities before writes; never replay a failed write or effect."""

    def __init__(self, store: SourceSessionStore, owner: ExecutionOwner, *, run_id: str,
                 parent_stream_id: str | None = None, writes: _Writes | None = None):
        self.store, self.owner, self.run_id = store, owner, run_id
        self.stream_id = uuid.uuid4().hex
        self.parent_stream_id = parent_stream_id
        self._writes = writes or _Writes(owner.read_snapshot())
        self._injection_bindings: dict[int, tuple[SourceInjectionBinding, bytes, bytes]] = {}
        self._tool_intent_bindings: dict[
            int, tuple[SourceToolIntentBinding, bytes]
        ] = {}
        self._tool_intents_by_event: dict[
            str, tuple[SourceToolIntentBinding | None, ...]
        ] = {}
        self._batch_callback: Callable[[KnownToolResult], Awaitable[str]] | None = None

    def fail(self) -> None:
        """Stop every stream after a mandatory runner or recording failure."""
        self._writes.failed = True

    @property
    def failed(self) -> bool:
        """Whether this shared source invocation has been revoked or failed."""
        return self._writes.failed

    def stream_events(self) -> list[dict[str, Any]]:
        """Return the current stream's durable events under the admitted owner."""
        self.owner.validate_dispatch()
        return [
            deepcopy(event)
            for event in self.owner.read_snapshot().events
            if event.get('_stream_id') == self.stream_id
        ]

    def child_stream(self) -> SourceTurnWriter:
        return SourceTurnWriter(self.store, self.owner, run_id=self.run_id,
                                parent_stream_id=self.stream_id, writes=self._writes)

    @property
    def committed_events(self) -> list[dict[str, Any]]:
        return deepcopy([e for e in self._writes.committed if e.get('_stream_id') == self.stream_id])

    def _append(self, payload: dict[str, object], *, execution_ids: list[str] | None = None,
                execution_id: str | None = None, source_origin: dict[str, Any] | None = None,
                 artifact_read: dict[str, Any] | None = None,
                 filesystem_effects: tuple[FilesystemEffectFact, ...] | None = None) -> dict[str, Any]:
        if self._writes.failed:
            raise SourceValidationError('Source turn recording failed')
        try:
            self.owner.validate_dispatch()
            clean = deepcopy(payload)
            for key in ('reasoning_content', 'thinking_blocks', '_meta', '_stream_id', '_parent_stream_id', '_source_origin', '_artifact_read', '_busy_input_id', '_pending_followup_id', '_child_followup', '_background_result', '_filesystem_effects'):
                clean.pop(key, None)
            if 'content' in clean:
                clean['content'] = _persisted_content(clean['content'])
            event = new_source_event(clean, run_id=self.run_id)
            if source_origin is not None:
                event['_source_origin'] = deepcopy(source_origin)
            if artifact_read is not None:
                event['_artifact_read'] = deepcopy(artifact_read)
            event['_stream_id'] = self.stream_id
            if self.parent_stream_id is not None:
                event['_parent_stream_id'] = self.parent_stream_id
            if execution_ids is not None:
                event['_execution_ids'] = execution_ids
            if execution_id is not None:
                event['_execution_id'] = execution_id
                if filesystem_effects is not None:
                    event['_filesystem_effects'] = [fact.to_dict(execution_id) for fact in filesystem_effects]
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

    def consume_pending_followup(
        self,
        record: PendingFollowupRecord,
        projection: dict[str, object],
    ) -> SourceInjectionBinding:
        """Atomically commit original input and return its invocation-bound projection."""
        if self.failed or self.parent_stream_id is not None or type(record) is not PendingFollowupRecord:
            raise SourceValidationError("Invalid source follow-up binding")
        try:
            canonical_events([projection])
            self.owner.validate_dispatch()
            current = self.owner.read_snapshot()
            if current.events != self._writes.snapshot.events:
                raise SourceValidationError("Source turn changed outside admitted writer")
            payload: dict[str, object] = {"role": "user", "content": record.content}
            if record.media:
                payload["media"] = list(record.media)
            event = new_source_event(payload, run_id=self.run_id)
            event["_pending_followup_id"] = record.identifier
            if record.child_followup is not None:
                event["_child_followup"] = record.child_followup.to_dict()
            if record.background_result is not None:
                event["_background_result"] = record.background_result.to_dict()
            event["_stream_id"] = self.stream_id
            self._writes.pending = event
            committed_snapshot, committed_event = self.store.consume_pending_followup(
                self.owner.key,
                self.owner,
                record,
                event,
            )
        except BaseException:
            self._writes.failed = True
            raise
        self._writes.snapshot = committed_snapshot
        if not any(item.get("_event_id") == committed_event.get("_event_id")
                   for item in self._writes.committed):
            self._writes.committed.append(committed_event)
        self._writes.pending = None
        return self._bind_injection(committed_event, projection, record.identifier)

    def append_injection(
        self, original: dict[str, object], projection: dict[str, object],
    ) -> SourceInjectionBinding:
        """Bind an internal callback's projection to its newly committed input."""
        canonical_events([projection])
        return self._bind_injection(self.append_message(original), projection, None)

    def _bind_injection(
        self, event: dict[str, object], projection: dict[str, object], followup_id: str | None,
    ) -> SourceInjectionBinding:
        binding = SourceInjectionBinding(
            event_id=str(event["_event_id"]),
            followup_id=followup_id,
            message=deepcopy(projection),
        )
        self._injection_bindings[id(binding)] = (
            binding,
            canonical_events([event]),
            canonical_events([binding.message]),
        )
        return binding

    def validate_injection_binding(self, binding: object) -> SourceInjectionBinding:
        """Reject forged, changed, cross-writer, or no-longer-owned bindings."""
        saved = self._injection_bindings.get(id(binding))
        if (
            type(binding) is not SourceInjectionBinding
            or saved is None
            or saved[0] is not binding
        ):
            raise SourceValidationError("Invalid source user injection binding")
        self.owner.validate_dispatch()
        snapshot = self.owner.read_snapshot()
        events = [event for event in snapshot.events if event.get("_event_id") == binding.event_id]
        if (
            len(events) != 1
            or events[0].get("_pending_followup_id") != binding.followup_id
            or events[0].get("_stream_id") != self.stream_id
            or canonical_events([events[0]]) != saved[1]
            or canonical_events([binding.message]) != saved[2]
        ):
            raise SourceValidationError("Source user injection binding changed")
        return binding

    def begin_tool_batch(self, payload: dict[str, object]) -> dict[str, Any]:
        calls_value = payload.get('tool_calls')
        if (payload.get('role') != 'assistant' or not isinstance(calls_value, list)
                or not calls_value):
            self._writes.failed = True
            raise SourceValidationError('Invalid source tool batch')
        calls: list[object] = cast(list[object], calls_value)
        ids = [uuid.uuid4().hex for _ in calls]
        event = self._append(payload, execution_ids=ids)
        declared: list[object] = deepcopy(calls)
        bindings: list[SourceToolIntentBinding | None] = []
        for ordinal, (call_value, execution_id) in enumerate(zip(declared, ids)):
            if not isinstance(call_value, dict):
                bindings.append(None)
                continue
            # _append validated the full payload as canonical JSON before this view.
            call = cast(dict[str, object], call_value)
            function_value = call.get("function")
            if not isinstance(function_value, dict):
                bindings.append(None)
                continue
            function = cast(dict[str, object], function_value)
            tool_name = function.get("name")
            tool_call_id = call.get("id")
            if not isinstance(tool_name, str) or not tool_name or not isinstance(tool_call_id, str):
                bindings.append(None)
                continue
            if (
                tool_name not in {"followup_child", "delegate_task"}
                or self.parent_stream_id is not None
            ):
                bindings.append(None)
                continue
            arguments: object = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (TypeError, ValueError):
                    pass
            digest = _intent_arguments_digest(arguments)
            binding = SourceToolIntentBinding(
                event_id=str(event["_event_id"]),
                execution_id=execution_id,
                ordinal=ordinal,
                run_id=self.run_id,
                stream_id=self.stream_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments_digest=digest,
            )
            bindings.append(binding)
        if any(binding is not None for binding in bindings):
            durable_event = canonical_events([event])
            for binding in bindings:
                if binding is not None:
                    self._tool_intent_bindings[id(binding)] = (binding, durable_event)
        self._tool_intents_by_event[str(event["_event_id"])] = tuple(bindings)
        completed: set[int] = set()

        async def record(value: KnownToolResult) -> str:
            try:
                index = value.ordinal
                if type(index) is not int or not 0 <= index < len(declared) or index in completed:
                    self._writes.failed = True
                    raise SourceValidationError('Invalid source result binding')
                tool_call_id, tool_name = _source_tool_result_identity(declared[index])
                if value.tool_call_id != tool_call_id or value.name != tool_name:
                    self._writes.failed = True
                    raise SourceValidationError('Invalid source result binding')
                raw = value.raw_result.content if isinstance(value.raw_result, ToolOutcome) else value.raw_result
                if isinstance(raw, str):
                    raw = str(raw)
                canonical_events([{'content': raw}])
                origin = None
                if value.producer_receipt is not None:
                    from nanobot.agent.tools.read_provenance import (
                        canonical_tool_name_for_producer,
                        parse_read_receipt,
                    )

                    try:
                        receipt = parse_read_receipt(value.producer_receipt.to_dict())
                    except (ValueError, AttributeError, TypeError):
                        receipt = None
                    if (receipt is not None and value.dispatched and value.event.get('status') == 'ok'
                            and canonical_tool_name_for_producer(receipt.producer) == value.name
                            and receipt.matches(raw)):
                        origin = receipt.to_dict()
                    else:
                        logger.debug('Ignoring unmatched filesystem producer receipt')
                retrieval = None
                if value.retrieval_receipt is not None:
                    from nanobot.agent.tools.artifact import (
                        ArtifactRetrievalReceipt,
                        parse_retrieval_receipt,
                    )

                    candidate = value.retrieval_receipt
                    if (type(candidate) is ArtifactRetrievalReceipt and value.dispatched
                            and value.event.get('status') == 'ok' and candidate.matches(raw)):
                        try:
                            retrieval = parse_retrieval_receipt(candidate.to_dict()).to_dict()
                        except ValueError:
                            logger.debug('Ignoring invalid artifact retrieval receipt')
                tool_event: dict[str, Any] = {
                    'role': 'tool',
                    'tool_call_id': value.tool_call_id,
                    'name': value.name,
                    'content': raw,
                    '_dispatched': value.dispatched,
                    '_outcome_status': value.event.get('status', ''),
                }
                preparation_error_kind = value.event.get('_prepare_error_kind')
                if not value.dispatched and preparation_error_kind in {'unknown_tool', 'withheld_tool'}:
                    tool_event['_prepare_error_kind'] = preparation_error_kind
                committed = self._append(tool_event, execution_id=ids[index],
                                         source_origin=origin, artifact_read=retrieval,
                                         filesystem_effects=value.filesystem_effects)
                completed.add(index)
                if any(fact.reconciliation_required for fact in value.filesystem_effects or ()):
                    self.owner.recovery_required = True
                    self._writes.failed = True
                return committed['_event_id']
            except BaseException:
                self._writes.failed = True
                raise

        self._batch_callback = record
        return event

    def bind_tool_intents(
        self,
        event_id: str,
    ) -> tuple[SourceToolIntentBinding | None, ...] | None:
        """Return writer-minted identities for child dispatch tool calls."""
        if self.failed:
            raise SourceValidationError("Tool intent is unavailable after source failure")
        bindings = self._tool_intents_by_event.get(event_id)
        if bindings is None or not any(binding is not None for binding in bindings):
            return None
        return bindings

    def owns_tool_intent(self, binding: object) -> bool:
        """Confirm this writer minted the exact binding for the current stream."""
        saved = self._tool_intent_bindings.get(id(binding))
        if (
            type(binding) is not SourceToolIntentBinding
            or saved is None
            or saved[0] is not binding
            or binding.run_id != self.run_id
            or binding.stream_id != self.stream_id
        ):
            return False
        self.owner.validate_dispatch()
        return True

    def validate_tool_intent_binding(
        self,
        binding: object,
        *,
        arguments: object,
    ) -> SourceToolIntentBinding:
        """Reject copied, stale, unrelated, or changed tool intent evidence."""
        saved = self._tool_intent_bindings.get(id(binding))
        if (
            type(binding) is not SourceToolIntentBinding
            or saved is None
            or saved[0] is not binding
            or binding.run_id != self.run_id
            or binding.stream_id != self.stream_id
            or _intent_arguments_digest(arguments) != binding.arguments_digest
        ):
            raise SourceValidationError("Invalid source tool intent binding")
        self.owner.validate_dispatch()
        snapshot = self.owner.read_snapshot()
        events = [event for event in snapshot.events if event.get("_event_id") == binding.event_id]
        if len(events) != 1:
            raise SourceValidationError("Source tool intent is no longer current")
        event = events[0]
        event_bytes = canonical_events([event])
        calls_value = event.get("tool_calls")
        identities_value = event.get("_execution_ids")
        if (
            event.get("role") != "assistant"
            or event.get("_run_id") != binding.run_id
            or event.get("_stream_id") != binding.stream_id
            or not isinstance(calls_value, list)
            or not isinstance(identities_value, list)
            or event_bytes != saved[1]
        ):
            raise SourceValidationError("Source tool intent binding changed")
        calls: list[object] = cast(list[object], calls_value)
        identities: list[object] = cast(list[object], identities_value)
        if (
            binding.ordinal >= len(calls)
            or binding.ordinal >= len(identities)
            or identities[binding.ordinal] != binding.execution_id
        ):
            raise SourceValidationError("Source tool intent binding changed")
        call_value = calls[binding.ordinal]
        if not isinstance(call_value, dict):
            raise SourceValidationError("Source tool intent call is invalid")
        # event_bytes validated the recursive JSON object-key invariant above.
        call = cast(dict[str, object], call_value)
        function_value = call.get("function")
        if not isinstance(function_value, dict):
            raise SourceValidationError("Source tool intent function is invalid")
        function = cast(dict[str, object], function_value)
        stored_arguments: object = function.get("arguments")
        if isinstance(stored_arguments, str):
            try:
                stored_arguments = json.loads(stored_arguments)
            except (TypeError, ValueError):
                pass
        if (
            call.get("id") != binding.tool_call_id
            or function.get("name") != binding.tool_name
            or _intent_arguments_digest(stored_arguments) != binding.arguments_digest
        ):
            raise SourceValidationError("Source tool intent identity changed")
        return binding

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
