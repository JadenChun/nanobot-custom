"""H1 safe-step adapter; no independent runner, transport or completion authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from nanobot.agent.busy_control import ControlSnapshot
from nanobot.agent.context import ContextBuilder
from nanobot.agent.turn import TurnContext
from nanobot.session.busy_inbox import InboxEntry, InboxPayload, SessionBusyInbox
from nanobot.session.execution_owner import ExecutionOwner
from nanobot.session.source import SourceValidationError
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.utils.document import reference_non_image_attachments


@dataclass(frozen=True, slots=True)
class SteeringMessage:
    message: dict[str, object] = field(repr=False)
    source_event_id: str | None


def _binding_id(inbox: SessionBusyInbox, ingress_id: str, index: int) -> str:
    return hashlib.sha256(json.dumps([inbox.owner_id, inbox.journal_key, ingress_id, index]).encode()).hexdigest()


def reconcile_busy_source(inbox: SessionBusyInbox, owner: ExecutionOwner) -> None:
    """Recover real source evidence, not permission to retry external effects.

    The real owner validates the inbox's backend under its existing transaction.
    This remains usable during source recovery, when effect dispatch is forbidden.
    Complete or partial input bindings retain their uncertain/claimed disposition.
    """
    backend = inbox.sessions.source_backend(inbox.session_key)
    if backend is None:
        raise ValueError("busy source reconciliation requires a canonical session")
    with owner.transaction(backend, inbox.session_key):
        events = owner.read_snapshot().events
        for row in inbox.entries():
            if row.state not in {"claimed", "uncertain"} or not row.dispatch_started or row.payload is None:
                continue
            identities: list[str] = []
            stream: str | None = None
            missing = False
            for index, payload in enumerate((row.payload, *row.clarifications)):
                token = _binding_id(inbox, row.id, index)
                matching = [event for event in events if event.get("_busy_input_id") == token]
                if not matching:
                    missing = True
                    continue
                event = matching[0]
                event_id, stream_id = event.get("_event_id"), event.get("_stream_id")
                if (missing or len(matching) != 1 or not isinstance(event_id, str) or not event_id
                        or not isinstance(stream_id, str) or not stream_id
                        or event.get("_parent_stream_id") is not None or event.get("_run_id") != row.claimed_run_id
                        or event.get("role") != "user" or event.get("content") != payload.content
                        or event.get("media", []) != list(payload.media)
                        or (stream is not None and stream != stream_id)):
                    raise SourceValidationError("Conflicting busy input recovery evidence")
                stream = stream_id
                identities.append(event_id)
            if tuple(identities[:len(row.source_event_ids)]) != row.source_event_ids:
                raise SourceValidationError("Busy input source manifest lost durable evidence")
            if identities:
                inbox.recover_source_binding(row.id, row.revision, tuple(identities))


class BusySteering:
    """Claim only the named active invocation's authorized next-step lane.

    A dispatch marker is committed before rendering/source work. Mandatory write
    failures propagate, leaving conservative orphan recovery. Draining does not
    finish the ingress: the owning turn must settle it after persistence/effects.
    The cap counts whole ingresses, so clarification messages are never sliced.
    """

    def __init__(self, inbox: SessionBusyInbox, *, context: TurnContext,
                 observed_run: ControlSnapshot, active_scope: tuple[str, ...],
                 context_builder: ContextBuilder,
                 current_run: Callable[[], ControlSnapshot | None]):
        self.inbox = inbox
        self.context = context
        self.observed_run = observed_run
        self.active_scope = active_scope
        self.context_builder = context_builder
        self.current_run = current_run
        self._writer = context.source_writer
        self._policy = context.policy
        self._claims: list[InboxEntry] = []
        self.validate_context(context, self._writer)

    @property
    def claims(self) -> tuple[InboxEntry, ...]:
        return tuple(self._claims)

    def validate_context(self, context: TurnContext | None, writer: SourceTurnWriter | None) -> None:
        if (context is None or context is not self.context or context.run_id != self.observed_run.run_id
                or self.observed_run.state != "running" or context.session_key != self.inbox.session_key
                or context.policy is not self._policy or writer is not self._writer
                or context.source_writer is not writer):
            raise ValueError("busy steering requires the same admitted run context")
        if writer is not None and (writer.owner is not context.execution_owner
                or writer.owner.key != self.inbox.session_key or writer.run_id != context.run_id
                or writer.parent_stream_id is not None):
            raise ValueError("busy steering requires the same root source owner")
        backend = self.inbox.sessions.source_backend(self.inbox.session_key)
        if writer is not None:
            if backend is None:
                raise ValueError("busy steering source owner requires the canonical inbox backend")
            with writer.owner.transaction(backend, self.inbox.session_key):
                pass
        elif backend is not None:
            raise ValueError("canonical busy steering requires its root source writer")

    def _message(self, payload: InboxPayload) -> dict[str, object]:
        text, images = reference_non_image_attachments(payload.content, list(payload.media))
        return {"role": "user", "content": self.context_builder.build_user_content(text, images or None)}

    def record_initial(self, row: InboxEntry) -> tuple[SteeringMessage, ...]:
        """Record a next-turn claim through the same source boundary as a steer."""
        self.validate_context(self.context, self.context.source_writer)
        if (self._claims or row.state != "claimed" or row.dispatch_started
                or row.claimed_run_id != self.context.run_id
                or row.effective_scope != frozenset(self.active_scope)):
            raise ValueError("initial busy input requires the owning undispatched claim")
        return self._record_claim(row)

    def _record_claim(self, row: InboxEntry) -> tuple[SteeringMessage, ...]:
        claim_index = len(self._claims)
        self._claims.append(row)
        row = self.inbox.begin_dispatch(row.id, row.revision)
        self._claims[claim_index] = row
        if row.payload is None:
            raise SourceValidationError("Busy input has no original payload")
        messages: list[SteeringMessage] = []
        source_ids: list[str] = []
        writer = self._writer
        for index, payload in enumerate((row.payload, *row.clarifications)):
            message = self._message(payload)
            event_id = None
            if writer is not None:
                binding_id = _binding_id(self.inbox, row.id, index)
                event = writer.append_busy_input(payload.content, payload.media, binding_id=binding_id)
                event_id = event.get("_event_id")
                if (not isinstance(event_id, str) or not event_id
                        or not any(saved.get("_event_id") == event_id and saved.get("_busy_input_id") == binding_id
                                   and saved.get("role") == "user" and saved.get("content") == payload.content
                                   and saved.get("media", []) == list(payload.media)
                                   for saved in writer.stream_events())):
                    raise SourceValidationError("Busy input requires an actual owned source event")
                source_ids.append(event_id)
            messages.append(SteeringMessage(message, event_id))
        if writer is not None:
            row = self.inbox.bind_source(row.id, row.revision, tuple(source_ids))
        self._claims[claim_index] = row
        return tuple(messages)

    def drain(self, *, limit: int) -> tuple[SteeringMessage, ...]:
        if type(limit) is not int or not 1 <= limit <= 3:
            raise ValueError("busy steering drain limit must be between one and three")
        self.validate_context(self.context, self.context.source_writer)
        current = self.current_run()
        if (self.context.cancelled or current is None or current.state != "running"
                or current.run_id != self.observed_run.run_id
                or current.generation != self.observed_run.generation):
            return ()
        writer = self._writer
        if writer is not None:
            writer.owner.validate_dispatch()
            if writer.failed:
                raise SourceValidationError("Source turn recording failed")
        messages: list[SteeringMessage] = []
        for _ in range(limit):
            row = self.inbox.claim_steer(current, active_scope=self.active_scope)
            if row is None:
                break
            messages.extend(self._record_claim(row))
        return tuple(messages)
