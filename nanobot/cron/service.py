"""Definition-only cron management service during the staged migration.

``CronActionStore`` and ``CronDefinitionStore`` are the only definition
authority in this module.  The legacy ``jobs.json`` and ``action.jsonl``
paths are retained as constructor compatibility metadata, but are never read
or written by the service.  Runtime state, history, occurrence claims,
completion projection, and delivery belong to later migration packages.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import EllipsisType
from typing import Any, Callable, Coroutine, Literal

from loguru import logger

from nanobot.cron.action_store import (
    ActionStoreStaleGenerationError,
    CronActionStore,
)
from nanobot.cron.definition_store import (
    CronDefinition,
    CronDefinitionAction,
    CronDefinitionStore,
    CronDefinitionStoreReplay,
    DefinitionStoreConflictError,
)
from nanobot.cron.types import (
    CronJob,
    CronJobState,
    CronPayload,
    CronSchedule,
    CronStore,
)

CRON_DEFINITION_ONLY_DIAGNOSTIC = (
    "cron definition-only execution adapter is not installed; execution is fenced "
    "until Packages D/E install runtime, occurrence, and delivery authorities"
)


class CronJobSkippedError(Exception):
    """Raised by cron callbacks when a job was intentionally skipped."""


class CronDefinitionServiceError(RuntimeError):
    """Base error for definition-only service management."""


class CronDefinitionTokenRequiredError(CronDefinitionServiceError):
    """A mutation did not provide the exact definition concurrency tokens."""


class CronDefinitionConflictError(CronDefinitionServiceError):
    """A durable action was acknowledged but could not be checkpointed."""


def _now_ms() -> int:
    """Return wall-clock milliseconds for compatibility with callers."""
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """Compute a next run for legacy callers without granting scheduling authority."""
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None
    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        return now_ms + schedule.every_ms
    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter

            base_dt = datetime.fromtimestamp(
                now_ms / 1000,
                tz=ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo,
            )
            return int(croniter(schedule.expr, base_dt).get_next(datetime).timestamp() * 1000)
        except Exception:
            return None
    return None


def _validate_schedule_for_add(schedule: CronSchedule) -> None:
    """Validate schedule syntax before it enters the typed definition store."""
    if schedule.tz and schedule.kind != "cron":
        raise ValueError("tz can only be used with cron schedules")
    if schedule.kind == "cron":
        if not schedule.expr or not schedule.expr.strip():
            raise ValueError("cron schedule requires a non-empty 'expr'")
        try:
            from croniter import croniter

            croniter(schedule.expr)
        except Exception as exc:
            raise ValueError(f"invalid cron expression '{schedule.expr}': {exc}") from None
        if schedule.tz:
            try:
                from zoneinfo import ZoneInfo

                ZoneInfo(schedule.tz)
            except Exception:
                raise ValueError(f"unknown timezone '{schedule.tz}'") from None


def _normalize_agent_turn_job(_job: CronJob) -> bool:
    """Compatibility shim; definition reads must not mutate execution fields."""
    return False


def _definition_revision_action_id(action: CronDefinitionAction) -> str:
    """Derive an idempotent action identity from the exact typed request."""
    encoded = json.dumps(
        action.to_wire(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "cron-definition-" + hashlib.sha256(encoded).hexdigest()


def _job_definition(job: CronJob, *, job_instance_id: str) -> CronDefinition:
    """Project a public job shape into C2's strict definition shape."""
    payload = job.payload
    definition: dict[str, Any] = {
        "id": job.id,
        "name": job.name,
        "enabled": job.enabled,
        "schedule": {
            "kind": job.schedule.kind,
            "atMs": job.schedule.at_ms,
            "everyMs": job.schedule.every_ms,
            "expr": job.schedule.expr,
            "tz": job.schedule.tz,
        },
        "payload": {
            "kind": payload.kind,
            "message": payload.message,
            "deliver": payload.deliver,
            "channel": payload.channel,
            "to": payload.to,
            "additionalDestinations": [
                {"channel": destination.channel, "to": destination.to}
                for destination in payload.additional_destinations
            ],
            "channelMeta": deepcopy(payload.channel_meta),
            "sessionKey": payload.session_key,
            "executionMode": payload.execution_mode,
            "originChannel": payload.origin_channel,
            "originChatId": payload.origin_chat_id,
            "originMetadata": deepcopy(payload.origin_metadata),
        },
        "deleteAfterRun": job.delete_after_run,
    }
    return CronDefinition.from_definition(definition, job_instance_id=job_instance_id)


def _public_job(definition: CronDefinition) -> CronJob:
    """Project a definition into the compatibility ``CronJob`` API.

    The state and timestamp fields intentionally remain empty: C2 owns
    definitions only, so filling them from a definition would imply ownership
    over runtime/history state that has not been migrated yet.
    """
    wire = definition.to_wire()
    return CronJob(
        id=wire["id"],
        name=wire["name"],
        enabled=wire["enabled"],
        schedule=CronSchedule.from_store_dict(wire["schedule"]),
        payload=CronPayload.from_store_dict(wire["payload"]),
        state=CronJobState(),
        created_at_ms=0,
        updated_at_ms=0,
        delete_after_run=wire["deleteAfterRun"],
        job_instance_id=definition.job_instance_id,
        definition_revision=definition.definition_revision,
    )


class CronService:
    """Manage typed cron definitions while all execution remains fenced."""

    _MAX_CAS_RETRIES = 3

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
        max_sleep_ms: int = 300_000,
        *,
        definition_store: CronDefinitionStore | None = None,
        action_store: CronActionStore | None = None,
        namespace: str | None = None,
    ):
        """Create a service over the C2 definition authority.

        ``store_path`` remains the historical ``jobs.json`` location for
        callers that still provide it.  Unless a definition store is injected,
        the parent directory is used as the dedicated v2 action-store root;
        the path itself is never opened.  Conversion is intentionally absent
        from construction and must have completed through the stopped-writer
        converter before converted definitions are visible.
        """
        path = Path(store_path).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if definition_store is not None and action_store is not None:
            raise TypeError("provide definition_store or action_store, not both")
        if definition_store is None:
            if action_store is None:
                action_root = path.parent
                store_namespace = namespace or f"cron:{action_root}"
                action_store = CronActionStore(action_root, namespace=store_namespace)
            definition_store = CronDefinitionStore(action_store)
        elif not isinstance(definition_store, CronDefinitionStore):
            raise TypeError("definition_store must be a CronDefinitionStore")

        self.store_path = path
        self.definition_store = definition_store
        self.on_job = on_job
        self.max_sleep_ms = max_sleep_ms
        self._store: CronStore | None = None
        self._timer_task: asyncio.Task[None] | None = None
        self._running = False
        self._active_executions = 0
        self._store_dirty = False
        # Kept as an observable compatibility path.  No writer in this class
        # opens it; the action store lock is the sole mutation boundary.
        self._run_records_dir = path.parent / "runs"

    @property
    def action_store(self) -> CronActionStore:
        """Return the injected C1/C2 durability authority."""
        return self.definition_store.action_store

    def _should_persist_store(self) -> bool:
        """Compatibility answer; runtime snapshots are not service-owned."""
        return False

    def admission_diagnostic(self, _job: CronJob) -> str:
        """Return the stable execution fence for every job kind."""
        return CRON_DEFINITION_ONLY_DIAGNOSTIC

    @property
    def migration_diagnostic(self) -> str:
        """Expose the same stable fence for callers without a job object."""
        return CRON_DEFINITION_ONLY_DIAGNOSTIC

    def _prepare_agent_binding(self, _job: CronJob) -> bool:
        """Do not rewrite definition payloads while runtime adapters are absent."""
        return False

    def _prepare_store_agent_bindings(self) -> bool:
        """Do not mutate loaded definitions during management reads."""
        return False

    def _replay_and_settle(self) -> CronDefinitionStoreReplay:
        """Replay C2 and compact converted/pending operations once.

        C2 materializes legacy embedded operations in replay.  The first
        management read checkpoints that typed result, replacing the legacy
        representation with the C2 candidate so the operations are not
        interpreted again on later reads.
        """
        raw = self.definition_store.action_store.replay()
        legacy_snapshot = self.store_path.exists()
        if legacy_snapshot:
            candidate = raw.candidate
            converted_candidate = (
                candidate is not None
                and candidate.get("kind") == "legacy_cron_snapshot_candidate"
            )
            carried_evidence = (
                candidate is not None
                and candidate.get("kind") == "cron_definition_candidate_v1"
                and "legacy_conversion_evidence" in candidate
            )
            if not converted_candidate and not carried_evidence:
                raise CronDefinitionServiceError(
                    "legacy jobs.json detected; explicit stopped-writer conversion is required"
                )
        replay = self.definition_store.replay()
        raw_kind = raw.candidate.get("kind") if raw.candidate is not None else None
        if raw_kind == "legacy_cron_snapshot_candidate" or replay.pending_actions:
            replay = self.definition_store.checkpoint(replay)
        return replay

    def _load_store(self, *, reload_during_execution: bool = False) -> CronStore:
        """Load only typed definitions into detached public projections."""
        del reload_during_execution
        replay = self._replay_and_settle()
        jobs = [_public_job(definition) for definition in replay.candidate.definitions]
        self._store = CronStore(version=2, jobs=jobs)
        self._store_dirty = False
        return self._store

    def _require_store(self, *, reload_during_execution: bool = False) -> CronStore:
        return self._load_store(reload_during_execution=reload_during_execution)

    def _load_jobs(self) -> None:
        """Reject the former jobs.json read seam instead of falling back."""
        raise RuntimeError(
            "legacy jobs.json is not a cron definition authority; use CronDefinitionStore.replay()"
        )

    def _save_store(self) -> None:
        """Reject runtime snapshot writes; C2 stores definitions only."""
        self._store_dirty = False

    def write_run_record(self, _run_id: str, _record: dict[str, Any]) -> None:
        """Fence completion projection until a later runtime package owns it."""
        raise CronJobSkippedError(CRON_DEFINITION_ONLY_DIAGNOSTIC)

    async def start(self) -> None:
        """Start management availability without arming a timer or running jobs."""
        self._running = False
        self._arm_timer()
        self._load_store()
        self._running = True
        self._timer_task = None
        logger.info(
            "Cron definition service started with {} definitions; execution fenced",
            len(self._store.jobs),
        )

    def stop(self) -> None:
        """Stop management availability and cancel any compatibility timer."""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    def _recompute_next_runs(self) -> None:
        """Do not project automatic scheduling into definition state."""
        return

    def _get_next_wake_ms(self) -> None:
        """No wake exists while the timer authority is migration-fenced."""
        return None

    def _arm_timer(self) -> None:
        """Keep the timer unarmed until Packages D/E install execution authority."""
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    async def _on_timer(self) -> None:
        """Ignore timer ticks without selecting, executing, or projecting jobs."""
        return

    async def _execute_job(self, _job: CronJob) -> None:
        """Fence direct occurrence execution before any callback or side effect."""
        raise CronJobSkippedError(CRON_DEFINITION_ONLY_DIAGNOSTIC)

    def _definition_action(self, action: CronDefinitionAction) -> CronDefinitionStoreReplay:
        """Append and settle one typed action with a pre-ack CAS retry window."""
        action_id = _definition_revision_action_id(action)
        replay = self._replay_and_settle()
        for _attempt in range(self._MAX_CAS_RETRIES):
            self._ensure_action_current(action, action_id=action_id, replay=replay)
            try:
                self.definition_store.append(
                    action_id=action_id,
                    action=action,
                    expected_generation=replay.generation,
                )
            except ActionStoreStaleGenerationError:
                # No receipt was acknowledged.  Refresh the generation and
                # retry the same exact request; never append again after a
                # durable receipt has been returned.
                replay = self._replay_and_settle()
                continue
            break
        else:
            raise CronDefinitionConflictError(
                "cron definition changed before its durable action receipt was acknowledged"
            )

        settled = self.definition_store.replay()
        if settled.pending_actions:
            try:
                settled = self.definition_store.checkpoint(settled)
            except ActionStoreStaleGenerationError as exc:
                raise CronDefinitionConflictError(
                    "cron definition action was durably acknowledged but its checkpoint became stale; retry management read"
                ) from exc
        self._store = CronStore(
            version=2,
            jobs=[_public_job(definition) for definition in settled.candidate.definitions],
        )
        return settled

    @classmethod
    def _ensure_action_current(
        cls,
        action: CronDefinitionAction,
        *,
        action_id: str,
        replay: CronDefinitionStoreReplay,
    ) -> None:
        """Reject a stale definition before it can poison the durable journal."""
        if action_id in {receipt.action_id for receipt in replay.receipts}:
            return
        current = cls._find_definition(replay, action.job_id)
        if action.operation == "create":
            if current is not None:
                raise DefinitionStoreConflictError("cron definition already exists")
            return
        if current is None:
            raise DefinitionStoreConflictError("cron definition does not exist")
        if current.job_instance_id != action.expected_job_instance_id:
            raise DefinitionStoreConflictError("cron definition job instance does not match")
        if current.definition_revision != action.expected_definition_revision:
            raise DefinitionStoreConflictError("cron definition revision does not match")

    @staticmethod
    def _find_definition(replay: CronDefinitionStoreReplay, job_id: str) -> CronDefinition | None:
        return next(
            (definition for definition in replay.candidate.definitions if definition.job_id == job_id),
            None,
        )

    @staticmethod
    def _require_tokens(
        *,
        job: CronDefinition,
        job_instance_id: str | None,
        definition_revision: str | None,
    ) -> None:
        if not job_instance_id:
            raise CronDefinitionTokenRequiredError(
                "job_instance_id is required for cron definition mutation"
            )
        if not definition_revision:
            raise CronDefinitionTokenRequiredError(
                "definition_revision is required for cron definition mutation"
            )
        if job_instance_id != job.job_instance_id:
            raise DefinitionStoreConflictError("cron definition job instance does not match")
        if definition_revision != job.definition_revision:
            raise DefinitionStoreConflictError("cron definition revision does not match")

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """List detached public projections of typed definitions."""
        store = self._require_store()
        jobs = store.jobs if include_disabled else [job for job in store.jobs if job.enabled]
        return sorted(jobs, key=lambda job: job.id)

    def list_bound_cron_jobs_for_session(
        self,
        session_key: str,
        *,
        include_disabled: bool = True,
    ) -> list[CronJob]:
        """Return typed definitions whose payload carries the session key."""
        return [
            job
            for job in self.list_jobs(include_disabled=include_disabled)
            if job.payload.session_key == session_key
        ]

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
        channel_meta: dict[str, Any] | None = None,
        session_key: str | None = None,
        origin_channel: str | None = None,
        origin_chat_id: str | None = None,
        origin_metadata: dict[str, Any] | None = None,
    ) -> CronJob:
        """Create and checkpoint one typed definition."""
        _validate_schedule_for_add(schedule)
        existing = self._require_store().jobs
        job_id = uuid.uuid4().hex[:8]
        while any(job.id == job_id for job in existing):
            job_id = uuid.uuid4().hex[:8]
        job = CronJob(
            id=job_id,
            name=name,
            enabled=True,
            schedule=deepcopy(schedule),
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
                channel_meta=deepcopy(channel_meta or {}),
                session_key=session_key,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
                origin_metadata=deepcopy(origin_metadata or {}),
            ),
            state=CronJobState(),
            delete_after_run=delete_after_run,
        )
        instance = _stable_job_instance_id(self.action_store.namespace, job.id)
        action = CronDefinitionAction.create(_job_definition(job, job_instance_id=instance))
        settled = self._definition_action(action)
        result = self._find_definition(settled, job.id)
        if result is None:  # pragma: no cover - checkpoint/replay invariant
            raise CronDefinitionServiceError("created cron definition disappeared during checkpoint")
        logger.info("Cron: added definition '{}' ({})", name, job.id)
        return _public_job(result)

    def register_system_job(self, job: CronJob) -> CronJob:
        """Register a system definition without granting it execution authority.

        Existing definitions are left untouched when the legacy startup caller
        does not provide exact tokens.  This preserves the old idempotent
        registration shape while preventing an implicit compare-and-swap bypass.
        """
        if job.payload.kind != "system_event":
            raise ValueError("system registration requires a system_event payload")
        replay = self._replay_and_settle()
        current = self._find_definition(replay, job.id)
        if current is None:
            instance = job.job_instance_id or _stable_job_instance_id(self.action_store.namespace, job.id)
            action = CronDefinitionAction.create(_job_definition(job, job_instance_id=instance))
        else:
            if not job.job_instance_id or not job.definition_revision:
                return _public_job(current)
            self._require_tokens(
                job=current,
                job_instance_id=job.job_instance_id,
                definition_revision=job.definition_revision,
            )
            action = CronDefinitionAction.update(
                _job_definition(job, job_instance_id=current.job_instance_id),
                expected_job_instance_id=current.job_instance_id,
                expected_definition_revision=current.definition_revision,
            )
        settled = self._definition_action(action)
        result = self._find_definition(settled, job.id)
        if result is None:  # pragma: no cover - checkpoint/replay invariant
            raise CronDefinitionServiceError("registered system definition disappeared during checkpoint")
        return _public_job(result)

    def remove_system_job(
        self,
        job_id: str,
        *,
        job_instance_id: str | None = None,
        definition_revision: str | None = None,
    ) -> bool:
        """Remove a system definition only with exact tokens.

        The historical startup reconciler has no token parameters, so it is a
        safe no-op until a later caller supplies the C2 tokens.
        """
        replay = self._replay_and_settle()
        current = self._find_definition(replay, job_id)
        if current is None or current.definition["payload"]["kind"] != "system_event":
            return False
        if not job_instance_id or not definition_revision:
            return False
        self._require_tokens(
            job=current,
            job_instance_id=job_instance_id,
            definition_revision=definition_revision,
        )
        action = CronDefinitionAction.delete(
            job_id,
            expected_job_instance_id=current.job_instance_id,
            expected_definition_revision=current.definition_revision,
        )
        self._definition_action(action)
        return True

    def remove_job(
        self,
        job_id: str,
        *,
        job_instance_id: str | None = None,
        definition_revision: str | None = None,
        expected_job_instance_id: str | None = None,
        expected_definition_revision: str | None = None,
    ) -> Literal["removed", "protected", "not_found"]:
        """Delete a definition with the exact instance and revision tokens."""
        if job_instance_id is None:
            job_instance_id = expected_job_instance_id
        if definition_revision is None:
            definition_revision = expected_definition_revision
        replay = self._replay_and_settle()
        current = self._find_definition(replay, job_id)
        if current is None:
            return "not_found"
        self._require_tokens(
            job=current,
            job_instance_id=job_instance_id,
            definition_revision=definition_revision,
        )
        if current.definition["payload"]["kind"] == "system_event":
            logger.info("Cron: refused to remove protected system definition {}", job_id)
            return "protected"
        self._definition_action(
            CronDefinitionAction.delete(
                job_id,
                expected_job_instance_id=current.job_instance_id,
                expected_definition_revision=current.definition_revision,
            )
        )
        logger.info("Cron: removed definition {}", job_id)
        return "removed"

    def enable_job(
        self,
        job_id: str,
        enabled: bool = True,
        *,
        job_instance_id: str | None = None,
        definition_revision: str | None = None,
        expected_job_instance_id: str | None = None,
        expected_definition_revision: str | None = None,
    ) -> CronJob | None:
        """Change only the definition enabled flag with exact CAS tokens."""
        if job_instance_id is None:
            job_instance_id = expected_job_instance_id
        if definition_revision is None:
            definition_revision = expected_definition_revision
        replay = self._replay_and_settle()
        current = self._find_definition(replay, job_id)
        if current is None:
            return None
        self._require_tokens(
            job=current,
            job_instance_id=job_instance_id,
            definition_revision=definition_revision,
        )
        if current.definition["payload"]["kind"] == "system_event":
            return None
        changed = deepcopy(current.definition)
        changed["enabled"] = enabled
        action = CronDefinitionAction.update(
            CronDefinition.from_definition(changed, job_instance_id=current.job_instance_id),
            expected_job_instance_id=current.job_instance_id,
            expected_definition_revision=current.definition_revision,
        )
        settled = self._definition_action(action)
        result = self._find_definition(settled, job_id)
        return None if result is None else _public_job(result)

    def update_job(
        self,
        job_id: str,
        *,
        name: str | None = None,
        schedule: CronSchedule | None = None,
        message: str | None = None,
        deliver: bool | None = None,
        channel: str | None | EllipsisType = ...,
        to: str | None | EllipsisType = ...,
        delete_after_run: bool | None = None,
        job_instance_id: str | None = None,
        definition_revision: str | None = None,
        expected_job_instance_id: str | None = None,
        expected_definition_revision: str | None = None,
    ) -> CronJob | Literal["not_found", "protected"]:
        """Update definition fields with exact C2 instance/revision tokens."""
        if job_instance_id is None:
            job_instance_id = expected_job_instance_id
        if definition_revision is None:
            definition_revision = expected_definition_revision
        replay = self._replay_and_settle()
        current = self._find_definition(replay, job_id)
        if current is None:
            return "not_found"
        self._require_tokens(
            job=current,
            job_instance_id=job_instance_id,
            definition_revision=definition_revision,
        )
        if current.definition["payload"]["kind"] == "system_event":
            return "protected"
        changed = deepcopy(current.definition)
        if schedule is not None:
            _validate_schedule_for_add(schedule)
            changed["schedule"] = {
                "kind": schedule.kind,
                "atMs": schedule.at_ms,
                "everyMs": schedule.every_ms,
                "expr": schedule.expr,
                "tz": schedule.tz,
            }
        if name is not None:
            changed["name"] = name
        payload = changed["payload"]
        if message is not None:
            payload["message"] = message
        if deliver is not None:
            payload["deliver"] = deliver
        if channel is not ...:
            payload["channel"] = channel
        if to is not ...:
            payload["to"] = to
        if delete_after_run is not None:
            changed["deleteAfterRun"] = delete_after_run
        action = CronDefinitionAction.update(
            CronDefinition.from_definition(changed, job_instance_id=current.job_instance_id),
            expected_job_instance_id=current.job_instance_id,
            expected_definition_revision=current.definition_revision,
        )
        settled = self._definition_action(action)
        result = self._find_definition(settled, job_id)
        if result is None:  # pragma: no cover - checkpoint/replay invariant
            raise CronDefinitionServiceError("updated cron definition disappeared during checkpoint")
        logger.info("Cron: updated definition '{}' ({})", result.definition["name"], job_id)
        return _public_job(result)

    async def run_job(self, _job_id: str, force: bool = False) -> bool:
        """Fence manual/forced execution without invoking a callback."""
        del force
        return False

    def get_job(self, job_id: str) -> CronJob | None:
        """Return one detached typed definition projection."""
        return next((job for job in self._require_store().jobs if job.id == job_id), None)

    def status(self) -> dict[str, object]:
        """Return management status while exposing the execution fence."""
        store = self._require_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": None,
            "migration_diagnostic": CRON_DEFINITION_ONLY_DIAGNOSTIC,
        }


def _stable_job_instance_id(namespace: str, job_id: str) -> str:
    """Derive a stable identity for a newly created definition."""
    return hashlib.sha256(
        b"nanobot-cron-job-instance-v1\0"
        + namespace.encode("utf-8")
        + b"\0"
        + job_id.encode("utf-8")
    ).hexdigest()


__all__ = [
    "CRON_DEFINITION_ONLY_DIAGNOSTIC",
    "CronDefinitionConflictError",
    "CronDefinitionServiceError",
    "CronDefinitionTokenRequiredError",
    "CronJobSkippedError",
    "CronService",
]
