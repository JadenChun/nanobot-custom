"""In-process approval admission shared by canonical turn entrypoints.

Snapshot before selecting the session lock; consume only after acquiring that
lock. The short map lock never spans model, tool, session, or delivery work.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from nanobot.agent.turn import ApprovalDecision, TurnContext

if TYPE_CHECKING:
    from nanobot.session.source_store import SourceState

APPROVAL_TTL_SECONDS = 15 * 60


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
    expires_at: float = 0.0
    source_state: SourceState | None = None


@dataclass(frozen=True, slots=True)
class ApprovalSnapshot:
    alias_key: str
    pending: PendingApproval | None
    action: str
    effective_key: str
    decision: ApprovalDecision | None = None
    typed: bool = False
    strict: bool = False


_AUTHORITY_SEAL = object()


class EffectApprovalAuthority:
    """Sealed, volatile authority for one concrete effect attempt in one run.

    Policy replacements share this object. Availability checks never reserve or
    refill it; the synchronous dispatch check consumes it without an await.
    """

    def __init__(self, seal: object, effect_id: str,
                 check: Callable[[str, TurnContext | None, bool], bool]):
        if seal is not _AUTHORITY_SEAL:
            raise TypeError("Effect authority can only be issued by approval admission")
        self._effect_id = effect_id
        self._check = check

    def __reduce__(self) -> NoReturn:
        raise TypeError("Effect approval authority cannot be serialized")

    @property
    def effect_id(self) -> str:
        return self._effect_id

    def permits(self, effect_id: str) -> bool:
        return self._check(effect_id, None, False)

    def consume(self, effect_id: str, context: TurnContext | None) -> bool:
        return self._check(effect_id, context, True)


class ApprovalAdmission:
    """Index exact requests and restrict every lookup to an authorized alias."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._pending: dict[str, PendingApproval] = {}
        self._aliases: dict[str, dict[str, PendingApproval]] = {}
        self._grant_guard = threading.RLock()
        self._grants: dict[str, tuple[PendingApproval, TurnContext, EffectApprovalAuthority]] = {}

    def _issue(self, pending: PendingApproval, context: TurnContext,
               source_is_current: Callable[[], bool] | None) -> EffectApprovalAuthority:
        used = False
        run_id = context.run_id

        def check(effect_id: str, dispatched_context: TurnContext | None, spend: bool) -> bool:
            nonlocal used
            with self._grant_guard:
                held = self._grants.get(pending.approval_id)
                if (used or held is None or held[2] is not authority
                        or effect_id != pending.effect_id or pending.expires_at <= time.time()):
                    return False
            # Owner/source checks finish before the short authority lock. No
            # await separates them from the atomic one-use decision below.
            if (
                context.run_id != run_id or not self._context_matches(pending, context)
                or (spend and dispatched_context is not context)
                or (source_is_current is not None and not source_is_current())
            ):
                return False
            with self._grant_guard:
                held = self._grants.get(pending.approval_id)
                if (
                    used or held is None or held[2] is not authority
                    or effect_id != pending.effect_id or pending.expires_at <= time.time()
                ):
                    return False
                if spend:
                    used = True
                return True

        authority = EffectApprovalAuthority(_AUTHORITY_SEAL, pending.effect_id or "", check)
        with self._grant_guard:
            self._grants[pending.approval_id] = (pending, context, authority)
        return authority

    def _latest(self, key: str) -> PendingApproval | None:
        candidates = self._aliases.get(key, {})
        for pending in list(candidates.values())[::-1]:
            if pending.expires_at and pending.expires_at <= time.time():
                self._remove_identity(pending)
            else:
                return pending
        return None

    def _owns_alias(self, key: str, pending: PendingApproval) -> bool:
        return (
            self._pending.get(pending.approval_id) is pending
            and self._aliases.get(key, {}).get(pending.approval_id) is pending
        )

    async def snapshot(self, key: str, content: str) -> ApprovalSnapshot:
        """Legacy reply classification; this path never issues effect authority."""
        async with self._guard:
            pending = self._latest(key)
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
        """Resolve an exact ID only through its original-chat/session aliases."""
        async with self._guard:
            latest = self._latest(key)
            pending = self._pending.get(decision.approval_id) if decision else latest
            if pending is not None and pending.expires_at and pending.expires_at <= time.time():
                self._remove_identity(pending)
                pending = None
            if decision is None:
                return ApprovalSnapshot(key, pending, 'other' if pending else 'none', key, strict=True)
            if pending is None or not self._owns_alias(key, pending):
                return ApprovalSnapshot(
                    key, latest, 'stale', key, decision=decision, typed=True, strict=True,
                )
            if not self._decision_matches(pending, decision):
                return ApprovalSnapshot(
                    key, pending, 'stale', key, decision=decision, typed=True, strict=True,
                )
            action = 'affirmative' if decision.decision == 'approve' else 'negative'
            return ApprovalSnapshot(
                key, pending, action, pending.session_key, decision=decision, typed=True,
                strict=True,
            )

    @staticmethod
    def _context_matches(pending: PendingApproval, context: TurnContext) -> bool:
        if context.cancelled or not context.run_id or context.session_key != pending.session_key:
            return False
        owner = context.execution_owner
        if pending.source_state is not None:
            if owner is None or not owner.read_snapshot().state.same_origin(pending.source_state):
                return False
        elif owner is not None:
            state = owner.read_snapshot().state
            if (pending.session_incarnation is not None
                    and pending.session_incarnation != state.incarnation):
                return False
            if pending.generation is not None and pending.generation != state.generation:
                return False
        return pending.run_id != context.run_id

    async def consume(self, snapshot: ApprovalSnapshot, *, context: TurnContext | None = None,
                      source_is_current: Callable[[], bool] | None = None) -> str:
        """Consume under the selected owner admission; install authority privately.

        The loop supplies its admitted context and live source/run check. The
        child owner can use the same seam with its own narrowed live check.
        Legacy direct callers retain classification but receive no authority.
        """
        if snapshot.pending is None or snapshot.action in {'command', 'stale'}:
            return snapshot.action
        if snapshot.strict and not snapshot.typed:
            return 'other'
        # The caller already holds the selected execution/session admission.
        # Keep source I/O out of the map lock; identity is checked again below.
        current = context is None or (
            self._context_matches(snapshot.pending, context)
            and (source_is_current is None or source_is_current())
        )
        async with self._guard:
            pending = snapshot.pending
            if not self._owns_alias(snapshot.alias_key, pending):
                return 'stale'
            if pending.expires_at and pending.expires_at <= time.time():
                self._remove_identity(pending)
                return 'stale'
            if snapshot.typed and (
                snapshot.decision is None or not self._decision_matches(pending, snapshot.decision)
            ):
                return 'stale'
            if not current:
                self._remove_identity(pending)
                return 'stale'
            self._remove_identity(pending)
            if snapshot.typed and snapshot.action == 'affirmative' and context is not None:
                context.policy.approval_granted = True
                if pending.effect_id and re.fullmatch(r'[0-9a-f]{64}', pending.effect_id):
                    context.policy.effect_authority = self._issue(pending, context, source_is_current)
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

    async def revoke(self, session_key: str) -> int:
        """Revoke pending requests and all unspent grants for one source."""
        async with self._guard:
            victims = [pending for pending in self._pending.values()
                       if pending.session_key == session_key]
            removed = sum(1 for pending in victims if self._remove_identity(pending))
            with self._grant_guard:
                for approval_id, (pending, _context, _authority) in list(self._grants.items()):
                    if pending.session_key == session_key:
                        del self._grants[approval_id]
            return removed

    def retire(self, context: TurnContext) -> None:
        """End an invocation's volatile authority, including unused grants."""
        with self._grant_guard:
            for approval_id, (_pending, granted_context, _authority) in list(self._grants.items()):
                if granted_context is context:
                    del self._grants[approval_id]

    def _remove_identity(self, pending: PendingApproval) -> bool:
        if self._pending.get(pending.approval_id) is not pending:
            return False
        del self._pending[pending.approval_id]
        for key, candidates in list(self._aliases.items()):
            if candidates.get(pending.approval_id) is pending:
                del candidates[pending.approval_id]
            if not candidates:
                del self._aliases[key]
        return True

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
        source_state: SourceState | None = None,
    ) -> PendingApproval:
        async with self._guard:
            identifier = approval_id or uuid.uuid4().hex
            if identifier in self._pending or identifier in self._grants:
                raise ValueError("approval ID is already in use")
            # One source invocation supersedes its own earlier request. Other
            # sources sharing the visible chat keep independently indexed IDs.
            for previous in list(self._pending.values()):
                if previous.session_key == session_key:
                    self._remove_identity(previous)
            created_at = time.time()
            pending = PendingApproval(
                summary,
                created_at,
                session_key,
                approval_id=identifier,
                session_incarnation=session_incarnation,
                run_id=run_id,
                turn_id=turn_id,
                effect_id=effect_id,
                generation=generation,
                expires_at=created_at + APPROVAL_TTL_SECONDS,
                source_state=source_state,
            )
            self._pending[identifier] = pending
            for alias in {session_key, f'{channel}:{chat_id}'}:
                self._aliases.setdefault(alias, {})[identifier] = pending
            return pending

    async def clear(self, key: str, pending: PendingApproval | None = None) -> bool:
        async with self._guard:
            candidate = pending or self._latest(key)
            if candidate is None or not self._owns_alias(key, candidate):
                return False
            return self._remove_identity(candidate)
