"""Bounded whole-job Codex recovery, separate from provider transport retries."""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Awaitable, Callable, Generator, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from nanobot.agent.turn import TurnResult
from nanobot.cron.definition_store import CronDefinition
from nanobot.cron.occurrence_store import (
    OccurrenceClaim,
    OccurrenceRecord,
    TriggeredOccurrenceIdentity,
)
from nanobot.cron.schedule import ScheduleClock
from nanobot.cron.scheduler_store import SchedulerStore
from nanobot.cron.service import CronService


def codex_retry_at(usage: object, *, now_ms: int, retries: int) -> int | None:
    """Three primary-window retries, reset + two minutes, at most 24h away."""
    if retries >= 3 or not isinstance(usage, dict):
        return None
    data = cast(dict[str, object], usage)
    if data.get("rate_limit_reached_type") == "secondary":
        return None
    raw = data.get("rate_limit")
    if not isinstance(raw, dict):
        return None
    limits = cast(dict[str, object], raw)
    secondary = limits.get("secondary_window")
    if isinstance(secondary, dict):
        used = cast(dict[str, object], secondary).get("used_percent", 0)
        if type(used) not in (int, float) or not isinstance(used, (int, float)) or not math.isfinite(used) or used >= 100 or used < 0:
            return None
    primary = limits.get("primary_window")
    if not isinstance(primary, dict):
        return None
    reset = cast(dict[str, object], primary).get("reset_at")
    if type(reset) is not int or reset <= 0:
        return None
    due = reset * 1000 + 120_000
    if due - now_ms > 86_400_000:
        return None
    return due if due > now_ms else now_ms + 120_000


async def _codex_usage() -> object:
    from nanobot.providers.codex_auth import get_codex_usage

    return await asyncio.to_thread(get_codex_usage)


class QuotaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    epoch: int = Field(ge=0)
    retry_at_ms: int | None = Field(default=None, ge=0)
    retry_count: int = Field(ge=0, le=3)


class PendingQuotaRetry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    job_id: str
    instance: str
    revision: str
    origin: str
    retry_at_ms: int
    state: Literal["pending", "dispatching", "unknown"] = "pending"


class _QuotaState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    epoch: int = 0
    retry_count: int = Field(default=0, ge=0, le=3)
    pending: PendingQuotaRetry | None = None


class CronQuotaRecovery:
    def __init__(self, root: Path, clock: ScheduleClock, *, is_codex: Callable[[], bool],
                 usage: Callable[[], Awaitable[object]] = _codex_usage):
        self.root, self.clock, self.is_codex, self.usage = root, clock, is_codex, usage

    @contextmanager
    def _db(self) -> Generator[sqlite3.Connection]:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self.root / "quota.sqlite3"
        if path.is_symlink() or self.root.is_symlink():
            raise ValueError("quota state must not be a symlink")
        connection = sqlite3.connect(path, timeout=5)
        try:
            if os.name != "nt":
                path.chmod(0o600)
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS jobs (instance TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _state(connection: sqlite3.Connection, instance: str) -> _QuotaState:
        row: object = connection.execute("SELECT value FROM jobs WHERE instance=?", (instance,)).fetchone()
        if row is None:
            return _QuotaState()
        if not isinstance(row, tuple):
            raise ValueError("invalid quota state row")
        values = cast(tuple[object, ...], row)
        if len(values) != 1 or not isinstance(values[0], str):
            raise ValueError("invalid quota state payload")
        return _QuotaState.model_validate_json(values[0])

    @staticmethod
    def _save(connection: sqlite3.Connection, instance: str, state: _QuotaState) -> None:
        connection.execute("INSERT INTO jobs(instance,value) VALUES(?,?) ON CONFLICT(instance) DO UPDATE SET value=excluded.value",
                           (instance, state.model_dump_json()))

    def state(self, instance: str) -> _QuotaState:
        with self._db() as connection:
            return self._state(connection, instance)

    def import_legacy(self, pending: PendingQuotaRetry | None, *, instance: str,
                      retry_count: int) -> None:
        """Seed the old budget once; old pending work remains unknown for review."""
        if type(retry_count) is not int or retry_count < 0:
            raise ValueError("invalid legacy retry budget")
        if pending is not None and (pending.instance != instance or pending.state != "unknown"):
            raise ValueError("legacy retry must be bound and quarantined")
        with self._db() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE instance=?", (instance,)).fetchone() is not None:
                return
            self._save(connection, instance, _QuotaState(retry_count=min(retry_count, 3), pending=pending))

    def reconcile_pending(self, pending: PendingQuotaRetry, *, scheduler: SchedulerStore,
                          resolution: Literal["abandon", "confirmed_not_started"],
                          external_evidence: str) -> None:
        """Explicit stopped-writer review, never a model/tool replay decision."""
        if scheduler.root.absolute() != (self.root.parent / "scheduler").absolute():
            raise ValueError("quota reconciliation requires its exact scheduler")
        if not external_evidence.strip() or len(external_evidence.encode()) > 4096:
            raise ValueError("bounded external reconciliation evidence is required")
        if resolution not in {"abandon", "confirmed_not_started"}:
            raise ValueError("unsupported quota reconciliation")
        with scheduler.stopped_writer() as lease, self._db() as connection:
            state = self._state(connection, pending.instance)
            if state.pending != pending or pending.state != "unknown":
                raise ValueError("quota reconciliation identity changed or is not unknown")
            state.pending = None if resolution == "abandon" else pending.model_copy(update={"state": "pending"})
            connection.execute("CREATE TABLE IF NOT EXISTS reviews (origin TEXT, resolution TEXT, evidence TEXT, clock_ms INTEGER)")
            connection.execute("INSERT INTO reviews VALUES(?,?,?,?)", (pending.origin, resolution,
                               external_evidence, self.clock.wall_time_ms()))
            self._save(connection, pending.instance, state)
            lease.reprove()

    def defer(self, claim: OccurrenceClaim) -> bool:
        definition = CronDefinition.decode_wire(claim.job_snapshot)
        pending = self.state(definition.job_instance_id).pending
        if pending is None:
            return False
        if pending.revision != definition.definition_revision:
            return False
        return not (isinstance(claim.identity, TriggeredOccurrenceIdentity)
                    and claim.identity.controller == "quota-retry"
                    and pending.state == "dispatching"
                    and pending.revision == definition.definition_revision)

    async def event(self, claim: OccurrenceClaim, result: TurnResult) -> QuotaEvent:
        definition = CronDefinition.decode_wire(claim.job_snapshot)
        state = self.state(definition.job_instance_id)
        due = None
        error = (result.error or "").casefold()
        # A whole-job retry is safe only when no tool or explicit send ran.
        # Uncertain execution, verification and transport failure never qualify.
        if (self.is_codex() and result.error and ("rate limit" in error or "rate_limit" in error)
                and not result.tools_used and not result.sent_messages and state.retry_count < 3):
            try:
                due = codex_retry_at(await self.usage(), now_ms=self.clock.wall_time_ms(), retries=state.retry_count)
            except Exception:
                due = None
        return QuotaEvent(epoch=state.epoch, retry_at_ms=due,
                          retry_count=state.retry_count + int(due is not None))

    def settle(self, record: OccurrenceRecord, *, delivered: bool) -> None:
        completion = record.completion
        if completion is None:
            return
        metadata: object = completion.get("runtime_metadata")
        if not isinstance(metadata, dict):
            return
        event_value = cast(dict[str, object], metadata).get("quota_recovery")
        if event_value is None:
            return
        event = QuotaEvent.model_validate(event_value)
        definition = CronDefinition.decode_wire(record.job_snapshot)
        with self._db() as connection:
            state = self._state(connection, definition.job_instance_id)
            if event.epoch < state.epoch:
                return
            if event.epoch != state.epoch:
                raise ValueError("quota completion sequence has a gap")
            state.epoch += 1
            state.retry_count = 0 if delivered else event.retry_count
            state.pending = None
            if event.retry_at_ms is not None:
                if delivered:
                    raise ValueError("a delivered result cannot also schedule quota recovery")
                state.pending = PendingQuotaRetry(job_id=definition.job_id, instance=definition.job_instance_id,
                    revision=definition.definition_revision, origin=record.occurrence_id, retry_at_ms=event.retry_at_ms)
            self._save(connection, definition.job_instance_id, state)

    def pending(self) -> list[PendingQuotaRetry]:
        with self._db() as connection:
            # sqlite3 returns a list; validate each dynamic row at this edge.
            rows = connection.execute("SELECT instance FROM jobs ORDER BY instance").fetchall()
            result: list[PendingQuotaRetry] = []
            for raw in cast(list[object], rows):
                if not isinstance(raw, tuple):
                    raise ValueError("invalid quota state identity")
                row = cast(tuple[object, ...], raw)
                if len(row) != 1 or not isinstance(row[0], str):
                    raise ValueError("invalid quota state identity")
                state = self._state(connection, row[0])
                if state.pending is not None:
                    result.append(state.pending)
            return result

    def mark(self, pending: PendingQuotaRetry, state: Literal["dispatching", "unknown"]) -> None:
        with self._db() as connection:
            current = self._state(connection, pending.instance)
            if current.pending is None or current.pending != pending:
                raise ValueError("quota retry changed before dispatch")
            current.pending.state = state
            self._save(connection, pending.instance, current)

    def recover(self, service: CronService) -> None:
        for pending in self.pending():
            if pending.state == "dispatching":
                self.mark(pending, "unknown")
        assert service.occurrence_store is not None
        records = service.occurrence_store.inspect_records()

        def epoch(record: OccurrenceRecord) -> int:
            value: object = (record.completion or {}).get("runtime_metadata")
            if isinstance(value, dict):
                event = cast(dict[str, object], value).get("quota_recovery")
                if event is not None:
                    return QuotaEvent.model_validate(event).epoch
            return -1

        outboxes = service.list_deliveries()
        for record in sorted(records, key=epoch):
            delivered = any(outbox.occurrence_id == record.occurrence_id
                            and service.delivery_status(outbox.outbox_id) == "delivered" for outbox in outboxes)
            self.settle(record, delivered=delivered)
