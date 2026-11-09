"""In-process approval admission shared by canonical turn entrypoints.

Snapshot before selecting the session lock; consume only after acquiring that
lock. The short map lock never spans model, tool, session, or delivery work.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass

from nanobot.agent.turn import ApprovalDecision


@dataclass(frozen=True, slots=True)
class PendingApproval:
    summary: str
    created_at: float
    session_key: str
    approval_id: str = ""
    session_incarnation: str | None = None
    run_id: str | None = None
    turn_id: str | None = None
    effect_id: str | None = None
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class ApprovalSnapshot:
    alias_key: str
    pending: PendingApproval | None
    action: str
    effective_key: str
    decision: ApprovalDecision | None = None
    typed: bool = False
    strict: bool = False


class ApprovalAdmission:
    """Preserve staging's alias identity checks across session-lock admission."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._pending: dict[str, PendingApproval] = {}

    async def snapshot(self, key: str, content: str) -> ApprovalSnapshot:
        async with self._guard:
            pending = self._pending.get(key)
            if pending is None:
                return ApprovalSnapshot(key, None, 'none', key)
            raw = content.strip()
            if re.fullmatch(r"(yes|y|approve|approved|go ahead|continue|do it|run it)\s*[.!]?", raw, re.I):
                action, effective_key = 'affirmative', pending.session_key
            elif re.fullmatch(r"(no|n|cancel|stop|don't|do not)\s*[.!]?", raw, re.I):
                action, effective_key = 'negative', key
            elif raw.startswith('/'):
                action, effective_key = 'command', key
            else:
                action, effective_key = 'other', key
            return ApprovalSnapshot(key, pending, action, effective_key)

    async def snapshot_decision(
        self,
        key: str,
        decision: ApprovalDecision | None,
    ) -> ApprovalSnapshot:
        """Resolve one exact typed decision under the admission lock."""
        async with self._guard:
            pending = self._pending.get(key)
            if pending is None:
                return ApprovalSnapshot(
                    key, None, 'none', key, decision=decision,
                    typed=decision is not None, strict=True,
                )
            if decision is None:
                return ApprovalSnapshot(key, pending, 'other', key, strict=True)
            if not self._decision_matches(pending, decision):
                return ApprovalSnapshot(
                    key, pending, 'stale', key, decision=decision, typed=True,
                    strict=True,
                )
            action = 'affirmative' if decision.decision == 'approve' else 'negative'
            effective_key = pending.session_key if action == 'affirmative' else key
            return ApprovalSnapshot(
                key, pending, action, effective_key, decision=decision, typed=True,
                strict=True,
            )

    async def consume(self, snapshot: ApprovalSnapshot) -> str:
        """Revalidate under the selected session lock and consume by identity.

        A typed decision must still match the stored request. Unrelated text
        and slash commands do not consume the request.
        """
        if snapshot.pending is None or snapshot.action == 'command':
            return snapshot.action
        if snapshot.strict and not snapshot.typed:
            # The typed loop never grants or consumes from text.
            return 'other'
        if not snapshot.typed:
            # Legacy text remains available only to direct callers and tests;
            # the loop uses snapshot_decision and never grants from text.
            async with self._guard:
                if self._pending.get(snapshot.alias_key) is not snapshot.pending:
                    return 'stale'
                self._remove_identity(snapshot.pending)
                return snapshot.action
        async with self._guard:
            if self._pending.get(snapshot.alias_key) is not snapshot.pending:
                return 'stale'
            if snapshot.typed and (
                snapshot.decision is None
                or not self._decision_matches(snapshot.pending, snapshot.decision)
            ):
                return 'stale'
            self._remove_identity(snapshot.pending)
            return snapshot.action

    @staticmethod
    def _decision_matches(
        pending: PendingApproval,
        decision: ApprovalDecision,
    ) -> bool:
        if decision.approval_id != pending.approval_id:
            return False
        for expected, actual in (
            (pending.session_incarnation, decision.session_incarnation),
            (pending.run_id, decision.run_id),
            (pending.turn_id, decision.turn_id),
            (pending.effect_id, decision.effect_id),
            (pending.generation, decision.generation),
        ):
            if expected is not None and actual is not None and expected != actual:
                return False
        if decision.session_key is not None and (
            decision.session_key not in {pending.session_key, pending.session_key.split(':', 1)[-1]}
        ):
            return False
        return True

    def _remove_identity(self, pending: PendingApproval) -> bool:
        removed = False
        for key, value in list(self._pending.items()):
            if value is pending:
                del self._pending[key]
                removed = True
        return removed

    async def store(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        summary: str,
        approval_id: str | None = None,
        session_incarnation: str | None = None,
        run_id: str | None = None,
        turn_id: str | None = None,
        effect_id: str | None = None,
        generation: int | None = None,
    ) -> PendingApproval:
        async with self._guard:
            pending = PendingApproval(
                summary,
                time.time(),
                session_key,
                approval_id=approval_id or uuid.uuid4().hex,
                session_incarnation=session_incarnation,
                run_id=run_id,
                turn_id=turn_id,
                effect_id=effect_id,
                generation=generation,
            )
            self._pending[session_key] = pending
            self._pending[f'{channel}:{chat_id}'] = pending
            return pending

    async def clear(self, key: str, pending: PendingApproval | None = None) -> bool:
        async with self._guard:
            candidate = pending or self._pending.get(key)
            if candidate is None or self._pending.get(key) is not candidate:
                return False
            return self._remove_identity(candidate)
