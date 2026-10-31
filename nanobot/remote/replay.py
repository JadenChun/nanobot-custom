"""Immutable mobile operation replay cache (M12).

The cache is keyed only by ``(installation_id, gateway_generation, request_id)``.
Access tokens, socket epochs and serialized submits never become operation
identity. A cache miss after an uncertain send is ``outcome_unknown`` with zero
new effects.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from nanobot.remote.protocol import (
    MutationDecision,
    MutationEnvelope,
    MutationMode,
    MutationOutcome,
    OperationKey,
)

Authorize = Callable[[], Awaitable[bool]]
Effect = Callable[[MutationEnvelope], Awaitable[dict[str, Any]]]


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayRecord:
    key: OperationKey
    principal: str
    action: str
    digest: str
    result: dict[str, Any]
    effect_count: int
    created_at: float
    expires_at: float


class OperationReplayCache:
    def __init__(
        self,
        *,
        max_entries: int = 512,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_entries < 1 or max_entries > 8192:
            raise ReplayError("max_entries is outside the accepted bound")
        if ttl_seconds <= 0 or ttl_seconds > 3600:
            raise ReplayError("ttl_seconds is outside the accepted bound")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._lock = asyncio.Lock()
        self._records: dict[OperationKey, ReplayRecord] = {}
        self._key_locks: dict[OperationKey, asyncio.Lock] = {}

    async def execute(
        self,
        envelope: MutationEnvelope,
        *,
        current_generation: int,
        current_epoch: int,
        authorize: Authorize,
        effect: Effect,
    ) -> MutationOutcome:
        if not await authorize():
            return _denied(envelope, 403)
        key = OperationKey(
            installation_id=envelope.installation_id,
            generation=envelope.gateway_generation,
            request_id=envelope.request_id,
        )
        if envelope.gateway_generation != current_generation:
            return _denied(envelope, 409)
        if envelope.mode is MutationMode.SUBMIT and envelope.epoch != current_epoch:
            return _denied(envelope, 409)

        async with self._lock:
            self._expire_locked()
            record = self._records.get(key)
            if record is not None:
                return self._join(envelope, record)
            if envelope.mode is MutationMode.REPLAY_ONLY:
                return MutationOutcome(
                    decision=MutationDecision.OUTCOME_UNKNOWN,
                    request_id=envelope.request_id,
                    effect_count=0,
                    http_status=409,
                )
            key_lock = self._key_locks.setdefault(key, asyncio.Lock())

        async with key_lock:
            async with self._lock:
                self._expire_locked()
                record = self._records.get(key)
                if record is not None:
                    return self._join(envelope, record)
            try:
                result = await effect(envelope)
            except Exception:
                async with self._lock:
                    self._key_locks.pop(key, None)
                raise
            now = self.clock()
            record = ReplayRecord(
                key=key,
                principal=envelope.principal,
                action=envelope.action,
                digest=envelope.digest,
                result=result,
                effect_count=1,
                created_at=now,
                expires_at=now + self.ttl_seconds,
            )
            async with self._lock:
                self._evict_locked()
                self._records[key] = record
                self._key_locks.pop(key, None)
            if not await authorize():
                # The effect is durable but this delivery is no longer authorized.
                return _denied(envelope, 403)
            return MutationOutcome(
                decision=MutationDecision.APPLIED,
                request_id=envelope.request_id,
                effect_id=envelope.request_id,
                effect_count=1,
                http_status=200,
            )

    def _join(self, envelope: MutationEnvelope, record: ReplayRecord) -> MutationOutcome:
        if (
            record.principal != envelope.principal
            or record.action != envelope.action
            or record.digest != envelope.digest
        ):
            return _denied(envelope, 409)
        return MutationOutcome(
            decision=MutationDecision.REPLAYED,
            request_id=envelope.request_id,
            effect_id=envelope.request_id,
            effect_count=1,
            http_status=200,
        )

    def _expire_locked(self) -> None:
        now = self.clock()
        for key, record in list(self._records.items()):
            if now >= record.expires_at:
                self._records.pop(key, None)

    def _evict_locked(self) -> None:
        while len(self._records) >= self.max_entries:
            oldest = min(self._records, key=lambda key: self._records[key].created_at)
            self._records.pop(oldest, None)


def _denied(envelope: MutationEnvelope, status: int) -> MutationOutcome:
    return MutationOutcome(
        decision=MutationDecision.DENIED,
        request_id=envelope.request_id,
        effect_count=0,
        http_status=status,
    )
