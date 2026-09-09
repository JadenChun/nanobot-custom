"""In-process approval admission shared by canonical turn entrypoints.

Snapshot before selecting the session lock; consume only after acquiring that
lock. The short map lock never spans model, tool, session, or delivery work.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PendingApproval:
    summary: str
    created_at: float
    session_key: str


@dataclass(frozen=True, slots=True)
class ApprovalSnapshot:
    alias_key: str
    pending: PendingApproval | None
    action: str
    effective_key: str


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

    async def consume(self, snapshot: ApprovalSnapshot) -> str:
        """Revalidate under the selected session lock and consume by identity.

        Only affirmative grants approval. Negative/stale callers must terminate
        without model calls. Unrelated text consumes the old request without
        granting; slash commands leave it pending, matching staging.
        """
        if snapshot.pending is None or snapshot.action == 'command':
            return snapshot.action
        async with self._guard:
            if self._pending.get(snapshot.alias_key) is not snapshot.pending:
                return 'stale'
            self._remove_identity(snapshot.pending)
            return snapshot.action

    def _remove_identity(self, pending: PendingApproval) -> bool:
        removed = False
        for key, value in list(self._pending.items()):
            if value is pending:
                del self._pending[key]
                removed = True
        return removed

    async def store(self, *, session_key: str, channel: str, chat_id: str,
                    summary: str) -> PendingApproval:
        async with self._guard:
            pending = PendingApproval(summary, time.time(), session_key)
            self._pending[session_key] = pending
            self._pending[f'{channel}:{chat_id}'] = pending
            return pending

    async def clear(self, key: str, pending: PendingApproval | None = None) -> bool:
        async with self._guard:
            candidate = pending or self._pending.get(key)
            if candidate is None or self._pending.get(key) is not candidate:
                return False
            return self._remove_identity(candidate)
