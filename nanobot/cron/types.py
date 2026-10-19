"""Cron types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast, overload

from nanobot.utils.dict_keys import get_camel_snake


@overload
def _store_int(value: Any, default: Literal[None]) -> int | None: ...


@overload
def _store_int(value: Any, default: int = 0) -> int: ...


def _store_int(value: Any, default: int | None = 0) -> int | None:
    """Coerce JSON numerics to int; treat null/blank like a missing key."""
    if value is None or value == "":
        return default
    return int(value)


@dataclass
class CronSchedule:
    """Schedule definition for a cron job."""
    kind: Literal["at", "every", "cron"]
    # For "at": timestamp in ms
    at_ms: int | None = None
    # For "every": interval in ms
    every_ms: int | None = None
    # For "cron": cron expression (e.g. "0 9 * * *")
    expr: str | None = None
    # Timezone for cron expressions
    tz: str | None = None

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronSchedule:
        return cls(
            kind=data["kind"],
            at_ms=_store_int(get_camel_snake(data, "atMs", "at_ms"), None),
            every_ms=_store_int(get_camel_snake(data, "everyMs", "every_ms"), None),
            expr=data.get("expr"),
            tz=data.get("tz"),
        )


@dataclass(frozen=True)
class CronDestination:
    """One channel/chat destination for a cron result."""

    channel: str
    to: str


@dataclass
class CronPayload:
    """What to do when the job runs."""
    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    # Legacy delivery fields used by pre-session-bound cron jobs.
    deliver: bool = False
    channel: str | None = None  # e.g. "whatsapp"
    to: str | None = None  # e.g. phone number
    additional_destinations: list[CronDestination] = field(default_factory=list)
    channel_meta: dict[str, Any] = field(default_factory=dict)
    session_key: str | None = None  # original session key for correct session recording
    execution_mode: Literal["session_bound", "legacy_fresh"] = "session_bound"
    origin_channel: str | None = None
    origin_chat_id: str | None = None
    origin_metadata: dict[str, Any] = field(default_factory=dict)

    def delivery_destinations(self) -> list[CronDestination]:
        """Return de-duplicated primary and additional result destinations."""
        if not self.deliver:
            return []

        destinations: list[CronDestination] = []
        if self.to:
            destinations.append(CronDestination(channel=self.channel or "cli", to=str(self.to)))
        destinations.extend(self.additional_destinations)

        unique: list[CronDestination] = []
        seen: set[tuple[str, str]] = set()
        for destination in destinations:
            key = (destination.channel, destination.to)
            if not destination.channel or not destination.to or key in seen:
                continue
            seen.add(key)
            unique.append(destination)
        return unique


    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronPayload:
        return cls(
            kind=data.get("kind", "agent_turn"),
            message=data.get("message", ""),
            deliver=data.get("deliver", False),
            channel=data.get("channel"),
            to=data.get("to"),
            additional_destinations=[
                entry if isinstance(entry, CronDestination) else CronDestination(**entry)
                for entry in (get_camel_snake(data, "additionalDestinations", "additional_destinations", cast(list[Any], [])) or cast(list[Any], []))
            ],
            channel_meta=dict(
                get_camel_snake(data, "channelMeta", "channel_meta", {}) or {}
            ),
            session_key=get_camel_snake(data, "sessionKey", "session_key"),
            execution_mode=get_camel_snake(data, "executionMode", "execution_mode", "session_bound"),
            origin_channel=get_camel_snake(data, "originChannel", "origin_channel"),
            origin_chat_id=get_camel_snake(data, "originChatId", "origin_chat_id"),
            origin_metadata=dict(
                get_camel_snake(data, "originMetadata", "origin_metadata", {}) or {}
            ),
        )


@dataclass
class CronRunRecord:
    """A single execution record for a cron job."""
    run_at_ms: int
    status: Literal["ok", "error", "skipped"]
    duration_ms: int = 0
    error: str | None = None
    run_id: str | None = None
    run_status: str | None = None
    occurrence_id: str | None = None

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronRunRecord:
        return cls(
            run_at_ms=_store_int(get_camel_snake(data, "runAtMs", "run_at_ms", 0)),
            status=data["status"],
            duration_ms=_store_int(get_camel_snake(data, "durationMs", "duration_ms", 0)),
            error=data.get("error"),
            run_id=get_camel_snake(data, "runId", "run_id"),
            run_status=get_camel_snake(data, "runStatus", "run_status"),
            occurrence_id=get_camel_snake(data, "occurrenceId", "occurrence_id"),
        )


@dataclass
class CronJobState:
    """Runtime state of a job."""
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None
    last_run_status: str | None = None
    run_history: list[CronRunRecord] = field(default_factory=list)

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronJobState:
        history = cast(object, get_camel_snake(data, "runHistory", "run_history", []) or [])
        if not isinstance(history, list):
            raise ValueError("cron run history must be a list")
        parsed_history: list[CronRunRecord] = []
        for record in cast(list[object], history):
            if isinstance(record, CronRunRecord):
                parsed_history.append(record)
            elif isinstance(record, dict):
                parsed_history.append(CronRunRecord.from_store_dict(cast(dict[Any, Any], record)))
            else:
                raise ValueError("cron run history contains an invalid record")
        return cls(
            next_run_at_ms=_store_int(
                get_camel_snake(data, "nextRunAtMs", "next_run_at_ms"), None
            ),
            last_run_at_ms=_store_int(
                get_camel_snake(data, "lastRunAtMs", "last_run_at_ms"), None
            ),
            last_status=get_camel_snake(data, "lastStatus", "last_status"),
            last_error=get_camel_snake(data, "lastError", "last_error"),
            last_run_status=get_camel_snake(data, "lastRunStatus", "last_run_status"),
            run_history=parsed_history,
        )


@dataclass
class CronJob:
    """A scheduled job."""
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False
    # Conversion-only stable identities. They are optional until the stopped
    # writer converter has established them; ordinary service construction is
    # deliberately unchanged in this package.
    job_instance_id: str | None = None
    definition_revision: str | None = None

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]) -> CronJob:
        kwargs = dict(kwargs)
        state_kwargs = dict(cast(dict[str, Any], kwargs.get("state", {})))
        state_kwargs["run_history"] = [
            record
            if isinstance(record, CronRunRecord)
            else CronRunRecord(**cast(dict[str, Any], record))
            for record in cast(list[object], state_kwargs.get("run_history", []))
        ]
        kwargs["schedule"] = CronSchedule(
            **cast(dict[str, Any], kwargs.get("schedule", {"kind": "every"}))
        )
        kwargs["payload"] = CronPayload.from_store_dict(cast(dict[str, Any], kwargs.get("payload", {})))
        kwargs["state"] = CronJobState(**state_kwargs)
        return cls(**cast(Any, kwargs))

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronJob:
        """Load a job from jobs.json (camelCase with snake_case fallbacks)."""
        return cls(
            id=data["id"],
            name=data["name"],
            enabled=data.get("enabled", True),
            schedule=CronSchedule.from_store_dict(data["schedule"]),
            payload=CronPayload.from_store_dict(data.get("payload") or {}),
            state=CronJobState.from_store_dict(data.get("state") or {}),
            created_at_ms=_store_int(get_camel_snake(data, "createdAtMs", "created_at_ms", 0)),
            updated_at_ms=_store_int(get_camel_snake(data, "updatedAtMs", "updated_at_ms", 0)),
            delete_after_run=bool(
                get_camel_snake(data, "deleteAfterRun", "delete_after_run", False)
            ),
            job_instance_id=get_camel_snake(data, "jobInstanceId", "job_instance_id"),
            definition_revision=get_camel_snake(data, "definitionRevision", "definition_revision"),
        )

    def to_store_dict(self) -> dict[str, Any]:
        """Return a detached canonical jobs.json-compatible representation."""
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "schedule": {
                "kind": self.schedule.kind,
                "atMs": self.schedule.at_ms,
                "everyMs": self.schedule.every_ms,
                "expr": self.schedule.expr,
                "tz": self.schedule.tz,
            },
            "payload": {
                "kind": self.payload.kind,
                "message": self.payload.message,
                "deliver": self.payload.deliver,
                "channel": self.payload.channel,
                "to": self.payload.to,
                "additionalDestinations": [
                    {"channel": destination.channel, "to": destination.to}
                    for destination in self.payload.additional_destinations
                ],
                "channelMeta": dict(self.payload.channel_meta),
                "sessionKey": self.payload.session_key,
                "executionMode": self.payload.execution_mode,
                "originChannel": self.payload.origin_channel,
                "originChatId": self.payload.origin_chat_id,
                "originMetadata": dict(self.payload.origin_metadata),
            },
            "state": {
                "nextRunAtMs": self.state.next_run_at_ms,
                "lastRunAtMs": self.state.last_run_at_ms,
                "lastStatus": self.state.last_status,
                "lastError": self.state.last_error,
                "lastRunStatus": self.state.last_run_status,
                "runHistory": [
                    {
                        "runAtMs": record.run_at_ms,
                        "status": record.status,
                        "durationMs": record.duration_ms,
                        "error": record.error,
                        "runId": record.run_id,
                        "runStatus": record.run_status,
                        "occurrenceId": record.occurrence_id,
                    }
                    for record in self.state.run_history
                ],
            },
            "createdAtMs": self.created_at_ms,
            "updatedAtMs": self.updated_at_ms,
            "deleteAfterRun": self.delete_after_run,
            "jobInstanceId": self.job_instance_id,
            "definitionRevision": self.definition_revision,
        }


@dataclass
class CronStore:
    """Persistent store for cron jobs."""
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)

    @classmethod
    def from_store_dict(cls, data: dict[str, Any]) -> CronStore:
        """Decode a detached jobs.json object without service-side mutation."""
        if not isinstance(cast(object, data), dict) or "jobs" not in data or not isinstance(data["jobs"], list):
            raise ValueError("legacy jobs store must be an object with a jobs list")
        version = _store_int(data.get("version", 1))
        jobs: list[CronJob] = []
        for item in cast(list[object], data["jobs"]):
            if not isinstance(item, dict):
                raise ValueError("legacy jobs store contains an invalid job")
            jobs.append(CronJob.from_store_dict(cast(dict[Any, Any], item)))
        return cls(version=version, jobs=jobs)

    def to_store_dict(self) -> dict[str, Any]:
        return {"version": self.version, "jobs": [job.to_store_dict() for job in self.jobs]}
