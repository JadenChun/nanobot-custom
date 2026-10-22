"""Dormant timer owner: exact missed classification, cursor commit, then one shot."""

from __future__ import annotations

from copy import deepcopy
from typing import Awaitable, Callable, Literal

from nanobot.cron.definition_store import CronDefinition, CronDefinitionStore, ScheduleCommit
from nanobot.cron.occurrence import CronOccurrenceRuntime
from nanobot.cron.occurrence_store import (
    OccurrenceClaim,
    OccurrenceRecord,
    TriggeredOccurrenceIdentity,
)
from nanobot.cron.schedule import ScheduleClock, is_fire, next_fire
from nanobot.cron.scheduler_store import (
    ObservedOccurrence,
    ScheduleCursor,
    SchedulerConflictError,
    SchedulerLease,
    SchedulerState,
    SchedulerStore,
    verify_occurrence_evidence,
)


def _key(definition: CronDefinition) -> str:
    return definition.job_instance_id + ":" + definition.definition_revision


def _exclusion(record: OccurrenceRecord) -> ObservedOccurrence:
    return {"occurrence_id": record.occurrence_id, "fire": record.identity.scheduled_fire_time_ms,
            "owner": record.owner, "generation": record.generation, "state": record.state}


class Scheduler:
    def __init__(self, definitions: CronDefinitionStore, occurrences: CronOccurrenceRuntime,
                 store: SchedulerStore, clock: ScheduleClock,
                 dispatch: Callable[[OccurrenceClaim, Callable[[], None]], Awaitable[object]],
                 *, refresh_committed_revisions: bool = False):
        if definitions.action_store.namespace != store.namespace or occurrences.definition_store is not definitions:
            raise ValueError("scheduler graph authorities must be exact and share namespace")
        self.definitions, self.occurrences, self.store, self.clock = definitions, occurrences, store, clock
        self._dispatch = dispatch
        self._refresh_committed_revisions = refresh_committed_revisions
        self._lease: SchedulerLease | None = None
        self._accepting = False

    def _committed(self) -> tuple[tuple[CronDefinition, ScheduleCommit], ...]:
        replay = self.definitions.replay()
        if replay.pending_actions:
            raise SchedulerConflictError("definition checkpoint pending")
        result: list[tuple[CronDefinition, ScheduleCommit]] = []
        for definition in replay.candidate.definitions:
            if not definition.definition["enabled"]:
                continue
            entry = next((item for item in replay.candidate.scheduling_commits
                          if item["job_instance_id"] == definition.job_instance_id
                          and item["definition_revision"] == definition.definition_revision), None)
            if entry is None or entry["schedule"] != definition.definition["schedule"]:
                raise SchedulerConflictError("definition scheduling evidence missing or conflicting")
            next_fire(entry["schedule"], entry["commit_clock_ms"], entry["commit_clock_ms"])
            result.append((definition, entry))
        return tuple(result)

    @staticmethod
    def _prove_records(state: SchedulerState, records: list[OccurrenceRecord]) -> None:
        verify_occurrence_evidence(state, records)

    @staticmethod
    def _advance(state: SchedulerState, definition: CronDefinition, commit: ScheduleCommit, lower: int,
                 upper: int, kind: Literal["missed", "claimed", "observed"], exclusions: list[ObservedOccurrence]) -> None:
        key = _key(definition)
        old = next((item for item in state["cursors"] if item["key"] == key), None)
        revision = 1 if old is None else old["revision"] + 1
        cursor: ScheduleCursor = {"key": key, "job_id": definition.job_id, "commit": commit,
                  "revision": revision, "boundary": upper,
                  "next_fire": next_fire(commit["schedule"], commit["commit_clock_ms"], upper)}
        if old is not None:
            state["cursors"][state["cursors"].index(old)] = cursor
        else:
            state["cursors"].append(cursor)
        state["classifications"].append({"key": key, "revision": revision,
            "predecessor_revision": revision - 1, "lower": lower, "upper": upper,
            "kind": kind, "exclusions": exclusions})

    def acquire_and_reconcile(self) -> None:
        self._lease = self.store.acquire_lease()
        self._accepting = False
        try:
            snapshot = self.clock.wall_time_ms()
            if type(snapshot) is not int or snapshot < 0:
                raise SchedulerConflictError("invalid acquisition clock")
            committed = self._committed()
            state = self.store.read_for_acquisition(self._lease)
            records = self.occurrences.occurrence_store.inspect_records()
            self._prove_records(state, records)
            successor = deepcopy(state)
            for definition, commit in committed:
                current = next((item for item in state["cursors"] if item["key"] == _key(definition)), None)
                lower = current["boundary"] if current else commit["commit_clock_ms"]
                if current is None and commit["schedule"]["kind"] == "at":
                    lower = min(lower, commit["schedule"]["atMs"] - 1)
                upper = max(snapshot, lower)
                exclusions = [_exclusion(record) for record in records
                    if not isinstance(record.identity, TriggeredOccurrenceIdentity)
                    and record.identity.job_id == definition.job_id
                    and record.identity.job_revision == definition.definition_revision
                    and record.job_snapshot.get("jobInstanceId") == definition.job_instance_id
                    and lower < record.identity.scheduled_fire_time_ms <= upper
                    and is_fire(commit["schedule"], commit["commit_clock_ms"], record.identity.scheduled_fire_time_ms)]
                if current is None or upper > lower:
                    self._advance(successor, definition, commit, lower, upper, "missed", exclusions)
            if successor != state:
                successor["generation"] += 1
                self._lease.reprove()
                self.store.compare_and_swap(state, successor)
            self._lease.reprove()
            self._accepting = True
        except BaseException:
            self._lease.release()
            self._lease = None
            raise

    def _reprove(self, definition: CronDefinition) -> None:
        if not self._accepting or self._lease is None:
            raise SchedulerConflictError("scheduler ingress fenced")
        self._lease.reprove()
        if not any(item == definition for item, _entry in self._committed()):
            raise SchedulerConflictError("scheduler definition changed before begin")

    async def wake_once(self) -> object | None:
        if not self._accepting or self._lease is None:
            raise SchedulerConflictError("scheduler ingress fenced")
        self._lease.reprove()
        committed = self._committed()
        state = self.store.read()
        records = self.occurrences.occurrence_store.inspect_records()
        self._prove_records(state, records)
        # New definitions committed while this lease is held begin at their
        # exact commit boundary. Existing cursors retain all pending due work.
        # Acquisition separately classifies downtime as missed; no old claim
        # or delivered occurrence is ever replayed here.
        successor = deepcopy(state)
        for definition, commit in committed:
            if any(item["key"] == _key(definition) for item in state["cursors"]):
                continue
            if not self._refresh_committed_revisions:
                raise SchedulerConflictError("new revision requires scheduler reacquisition")
            lower = commit["commit_clock_ms"]
            if commit["schedule"]["kind"] == "at":
                lower = min(lower, commit["schedule"]["atMs"] - 1)
            self._advance(successor, definition, commit, lower, lower, "missed", [])
        if successor != state:
            successor["generation"] += 1
            self._lease.reprove()
            self.store.compare_and_swap(state, successor)
            state = successor
        candidates: list[tuple[int, str, CronDefinition, ScheduleCommit, ScheduleCursor]] = []
        for definition, commit in committed:
            cursor = next((item for item in state["cursors"] if item["key"] == _key(definition)), None)
            if cursor is None:
                raise SchedulerConflictError("committed revision has no scheduler cursor")
            fire = cursor["next_fire"]
            if fire is not None and fire <= self.clock.wall_time_ms():
                candidates.append((fire, definition.job_id, definition, commit, cursor))
        if not candidates:
            return None
        fire, _job_id, definition, commit, cursor = min(candidates, key=lambda item: item[:2])
        self._reprove(definition)
        existing = next((record for record in records if not isinstance(record.identity, TriggeredOccurrenceIdentity)
            and record.identity.job_id == definition.job_id and record.identity.job_revision == definition.definition_revision
            and record.identity.scheduled_fire_time_ms == fire), None)
        claim = None
        if existing is None:
            claim = self.occurrences.claim_timer(definition.job_id, fire,
                expected_job_instance_id=definition.job_instance_id,
                expected_definition_revision=definition.definition_revision)
            existing = self.occurrences.occurrence_store.get(claim.occurrence_id)
        successor = deepcopy(state)
        self._advance(successor, definition, commit, cursor["boundary"], fire,
                      "claimed" if claim else "observed", [_exclusion(existing)])
        successor["generation"] += 1
        self._lease.reprove()
        self.store.compare_and_swap(state, successor)
        if claim is not None:
            return await self._dispatch(claim, lambda: self._reprove(definition))
        return None

    def stop_claiming(self) -> None:
        self._accepting = False

    def require_lifetime_lease(self) -> None:
        """Reprove the current lifetime lease without exposing its capability."""
        if self._lease is None:
            raise SchedulerConflictError("cron lifetime lease unavailable")
        self._lease.reprove()

    def release_lease(self) -> None:
        if self._lease is not None:
            try:
                self._lease.release()
            finally:
                if not self._lease.is_active:
                    self._lease = None

    async def aclose(self) -> None:
        # The owning runtime drains all admitted work before calling this.
        self.stop_claiming()
        self.release_lease()
