"""Cron tool for scheduling reminders and tasks."""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import json
from collections.abc import Mapping
from contextvars import ContextVar, Token
from datetime import datetime
from typing import Any, cast

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import ToolContext, current_request_context
from nanobot.agent.tools.schema import (
    ArraySchema,
    IntegerSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.agent.turn import ToolOutcome, TurnContext, TurnSource
from nanobot.cron.definition_store import DefinitionStoreConflictError
from nanobot.cron.service import (
    CronDefinitionConflictError,
    CronDefinitionTokenRequiredError,
    CronService,
)
from nanobot.cron.types import CronDestination, CronJob, CronJobState, CronSchedule
from nanobot.runtime_context import RUNTIME_CONTEXT_INPUT_META
from nanobot.session.keys import UNIFIED_SESSION_KEY

_MAX_DESTINATIONS = 256
_MAX_IDENTIFIER_BYTES = 512


def _snapshot_origin_metadata(
    metadata: Mapping[str, object] | Mapping[object, object],
) -> dict[str, object]:
    """Detach complete finite-JSON fields, excluding live runtime context."""
    snapshot: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or key == RUNTIME_CONTEXT_INPUT_META:
            continue
        try:
            encoded = json.dumps(value, allow_nan=False, sort_keys=True)
            snapshot[key] = json.loads(encoded)
        except (TypeError, ValueError, RecursionError):
            continue
    return snapshot

_ADDITIONAL_DESTINATION = ObjectSchema(
    properties={
        "channel": StringSchema(
            "Destination channel name.",
            min_length=1,
            max_length=_MAX_IDENTIFIER_BYTES,
        ),
        "to": StringSchema(
            "Destination recipient identifier.",
            min_length=1,
            max_length=_MAX_IDENTIFIER_BYTES,
        ),
    },
    required=["channel", "to"],
    additional_properties=False,
)

_CRON_PARAMETERS = tool_parameters_schema(
    action=StringSchema("Action to perform", enum=["add", "list", "remove"]),
    name=StringSchema(
        "Optional short human-readable label for the job "
        "(e.g., 'weather-monitor', 'daily-standup'). Defaults to first 30 chars of message."
    ),
    message=StringSchema(
        "REQUIRED when action='add'. Instruction for the agent to execute when the job triggers "
        "(e.g., 'Send a reminder to WeChat: xxx' or 'Check system status and report'). "
        "Not used for action='list' or action='remove'."
    ),
    every_seconds=IntegerSchema(description="Interval in seconds (for recurring tasks)"),
    cron_expr=StringSchema("Cron expression like '0 9 * * *' (for scheduled tasks)"),
    tz=StringSchema(
        "Optional IANA timezone for cron expressions (e.g. 'America/Vancouver'). "
        "When omitted with cron_expr, the tool's default timezone applies."
    ),
    at=StringSchema(
        "ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00'). "
        "Naive values use the tool's default timezone."
    ),
    additional_destinations=ArraySchema(
        _ADDITIONAL_DESTINATION,
        description=(
            "Optional additional result destinations for action='add'. Each item is exactly "
            "{channel,to}; the current chat remains the primary destination."
        ),
        max_items=_MAX_DESTINATIONS,
    ),
    job_id=StringSchema("REQUIRED when action='remove'. Job ID to remove (obtain via action='list')."),
    required=["action"],
    description=(
        "Action-specific parameters: add requires a non-empty message plus one schedule "
        "(every_seconds, cron_expr, or at); remove requires job_id; list only needs action. "
        "Per-action requirements are enforced at runtime (see field descriptions) so the "
        "top-level schema stays compatible with providers (e.g. OpenAI Codex/Responses) that "
        "reject oneOf/anyOf/allOf/enum/not at the root of function parameters."
    ),
)


@tool_parameters(_CRON_PARAMETERS)
class CronTool(Tool):
    """Tool to schedule reminders and recurring tasks."""

    def __init__(self, cron_service: CronService, default_timezone: str = "UTC"):
        self._cron = cron_service
        self._default_timezone = default_timezone
        self._in_cron_context: ContextVar[bool] = ContextVar("cron_in_context", default=False)

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.cron_service is not None

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        cron_service = ctx.cron_service
        if cron_service is None:
            raise RuntimeError("CronTool requires an initialized cron service")
        return cls(cron_service=cron_service, default_timezone=ctx.timezone)

    @staticmethod
    def _request_route() -> tuple[str, str, str, dict[str, Any]]:
        """Return routing from the authoritative request snapshot."""
        ctx = current_request_context()
        if ctx is None:
            return "", "", "", {}
        raw_key = f"{ctx.channel}:{ctx.chat_id}" if ctx.channel and ctx.chat_id else ""
        session_key = (
            raw_key if ctx.session_key == UNIFIED_SESSION_KEY else (ctx.session_key or "")
        )
        return session_key, ctx.channel or "", ctx.chat_id or "", _snapshot_origin_metadata(ctx.metadata or {})

    @staticmethod
    def _turn_route(context: TurnContext) -> tuple[str, str, str, dict[str, Any]]:
        """Return the detached route admitted with an immutable turn request."""
        request = context.request
        if request is None:
            return "", "", "", {}
        route = request.route or request.delivery_target
        if route is None:
            return "", "", "", _snapshot_origin_metadata(request.metadata or {})
        channel = route.channel or ""
        chat_id = route.chat_id or route.to or route.recipient or ""
        raw_key = f"{channel}:{chat_id}" if channel and chat_id else ""
        session_key = (
            raw_key if request.session_key == UNIFIED_SESSION_KEY else (request.session_key or "")
        )
        return session_key, channel, chat_id, _snapshot_origin_metadata(request.metadata or {})

    @staticmethod
    def _additional_destinations(
        value: list[dict[str, Any]] | None,
        *,
        primary_channel: str,
        primary_to: str,
    ) -> tuple[list[CronDestination], ToolResult | None]:
        """Validate and trim tool destinations without changing their order."""
        if value is None:
            return [], None
        if type(value) is not list:
            return [], ToolResult.error("Error: additional_destinations must be an array")
        if len(value) > _MAX_DESTINATIONS:
            return [], ToolResult.error(
                f"Error: additional_destinations cannot contain more than {_MAX_DESTINATIONS} items"
            )

        destinations: list[CronDestination] = []
        for index, item in enumerate(value):
            if type(item) is not dict:
                return [], ToolResult.error(
                    f"Error: additional_destinations[{index}] must be an object"
                )
            if set(item) != {"channel", "to"}:
                return [], ToolResult.error(
                    f"Error: additional_destinations[{index}] must contain exactly channel and to"
                )
            channel = item["channel"]
            target = item["to"]
            if type(channel) is not str or type(target) is not str:
                return [], ToolResult.error(
                    f"Error: additional_destinations[{index}] channel and to must be strings"
                )
            channel = channel.strip()
            target = target.strip()
            if not channel or not target:
                return [], ToolResult.error(
                    "Error: each additional destination requires channel and to"
                )
            if len(channel.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
                return [], ToolResult.error(
                    f"Error: additional_destinations[{index}] channel exceeds the "
                    f"{_MAX_IDENTIFIER_BYTES}-byte limit"
                )
            if len(target.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
                return [], ToolResult.error(
                    f"Error: additional_destinations[{index}] to exceeds the "
                    f"{_MAX_IDENTIFIER_BYTES}-byte limit"
                )
            destinations.append(CronDestination(channel=channel, to=target))

        unique = {(primary_channel, primary_to)}
        unique.update((item.channel, item.to) for item in destinations)
        if len(unique) > _MAX_DESTINATIONS:
            return [], ToolResult.error(
                f"Error: cron jobs cannot contain more than {_MAX_DESTINATIONS} unique destinations"
            )
        return destinations, None

    @staticmethod
    def _validate_primary_route(channel: object, target: object) -> ToolResult | None:
        """Reject an unauthenticated or oversized primary delivery route."""
        if type(channel) is not str or type(target) is not str:
            return ToolResult.error(
                "Error: scheduled cron jobs must be created from an authenticated chat session"
            )
        if not channel.strip() or not target.strip():
            return ToolResult.error(
                "Error: scheduled cron jobs must be created from an authenticated chat session"
            )
        if len(channel.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
            return ToolResult.error(
                f"Error: primary channel exceeds the {_MAX_IDENTIFIER_BYTES}-byte limit"
            )
        if len(target.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
            return ToolResult.error(
                f"Error: primary to exceeds the {_MAX_IDENTIFIER_BYTES}-byte limit"
            )
        return None

    def set_cron_context(self, active: bool) -> Token[bool]:
        """Mark whether the tool is executing inside a cron job callback."""
        return self._in_cron_context.set(active)

    def reset_cron_context(self, token: Token[bool]) -> None:
        """Restore previous cron context."""
        self._in_cron_context.reset(token)

    @staticmethod
    def _validate_timezone(tz: str) -> str | None:
        from zoneinfo import ZoneInfo

        try:
            ZoneInfo(tz)
        except (KeyError, Exception):
            return ToolResult.error(f"Error: unknown timezone '{tz}'")
        return None

    def _display_timezone(self, schedule: CronSchedule) -> str:
        """Pick the most human-meaningful timezone for display."""
        return schedule.tz or self._default_timezone

    @staticmethod
    def _format_timestamp(ms: int, tz_name: str) -> str:
        from zoneinfo import ZoneInfo

        dt = datetime.fromtimestamp(ms / 1000, tz=ZoneInfo(tz_name))
        return f"{dt.isoformat()} ({tz_name})"

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return (
            "Schedule reminders and recurring tasks. Actions: add, list, remove. "
            f"If tz is omitted, cron expressions and naive ISO times default to {self._default_timezone}."
        )

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        errors = super().validate_params(params)
        action = params.get("action")
        if action == "add" and not str(params.get("message") or "").strip():
            errors.append("message is required when action='add'")
        if action == "remove" and not str(params.get("job_id") or "").strip():
            errors.append("job_id is required when action='remove'")
        additional = params.get("additional_destinations")
        if isinstance(additional, list):
            for index, item in enumerate(cast(list[object], additional)):
                if type(item) is dict and set(cast(dict[object, object], item)) == {"channel", "to"}:
                    for field in ("channel", "to"):
                        value = cast(dict[object, object], item)[field]
                        if type(value) is str and not value.strip():
                            errors.append(
                                f"additional_destinations[{index}] {field} must be nonblank"
                            )
                        elif type(value) is str and len(value.strip().encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
                            errors.append(
                                f"additional_destinations[{index}] {field} exceeds the "
                                f"{_MAX_IDENTIFIER_BYTES}-byte limit"
                            )
        return errors

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Keep destination values strict while retaining legacy scalar casts."""
        raw_destinations = params.get("additional_destinations")
        casted = super().cast_params(params)
        if "additional_destinations" in params:
            casted["additional_destinations"] = raw_destinations
        return casted

    async def execute(
        self,
        action: str,
        name: str | None = None,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        additional_destinations: list[dict[str, Any]] | None = None,
    ) -> str:
        if action == "add":
            if self._in_cron_context.get():
                return ToolResult.error("Error: cannot schedule new jobs from within a cron job execution")
            return self._add_job(
                name,
                message,
                every_seconds,
                cron_expr,
                tz,
                at,
                additional_destinations=additional_destinations,
                route=self._request_route(),
            )
        elif action == "list":
            return self._list_jobs()
        elif action == "remove":
            return self._remove_job(job_id)
        return f"Unknown action: {action}"

    async def execute_with_context(self, context: TurnContext, **kwargs: Any) -> ToolOutcome:
        """Execute using the immutable turn request route.

        The explicit context route is authoritative for a contextual call.  A
        cron-origin turn is rejected here as well as through the legacy
        ContextVar guard so nested scheduling cannot bypass the policy when no
        ContextVar was installed.
        """
        action = kwargs.get("action")
        if action == "add":
            request = context.request
            if self._in_cron_context.get() or (
                request is not None and request.source is TurnSource.CRON
            ):
                return ToolOutcome(
                    content=ToolResult.error(
                        "Error: cannot schedule new jobs from within a cron job execution"
                    )
                )
            result = self._add_job(
                kwargs.get("name"),
                kwargs.get("message", ""),
                kwargs.get("every_seconds"),
                kwargs.get("cron_expr"),
                kwargs.get("tz"),
                kwargs.get("at"),
                additional_destinations=kwargs.get("additional_destinations"),
                route=self._turn_route(context),
            )
            return ToolOutcome(content=result)
        if action == "list":
            return ToolOutcome(content=self._list_jobs())
        if action == "remove":
            return ToolOutcome(content=self._remove_job(kwargs.get("job_id")))
        return ToolOutcome(content=f"Unknown action: {action}")

    def _add_job(
        self,
        name: str | None,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
        *,
        additional_destinations: list[dict[str, Any]] | None = None,
        route: tuple[str, str, str, dict[str, Any]] | None = None,
    ) -> str:
        if not message:
            return ToolResult.error(
                "Error: cron action='add' requires a non-empty 'message' parameter "
                "describing what to do when the job triggers "
                "(e.g. the reminder text). Retry including message=\"...\"."
            )
        session_key, origin_channel, origin_chat_id, origin_metadata = (
            route if route is not None else self._request_route()
        )
        if not session_key:
            return ToolResult.error("Error: scheduled cron jobs must be created from a chat session")
        if not origin_channel or not origin_chat_id:
            return ToolResult.error("Error: scheduled cron jobs must be created from a chat session")
        route_error = self._validate_primary_route(origin_channel, origin_chat_id)
        if route_error is not None:
            return route_error
        destinations, destination_error = self._additional_destinations(
            additional_destinations,
            primary_channel=origin_channel,
            primary_to=origin_chat_id,
        )
        if destination_error is not None:
            return destination_error
        if tz and not cron_expr:
            return ToolResult.error("Error: tz can only be used with cron_expr")
        if tz:
            if err := self._validate_timezone(tz):
                return err

        # Build schedule
        delete_after = False
        if every_seconds:
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            effective_tz = tz or self._default_timezone
            if err := self._validate_timezone(effective_tz):
                return err
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=effective_tz)
        elif at:
            from zoneinfo import ZoneInfo

            try:
                dt = datetime.fromisoformat(at)
            except ValueError:
                return ToolResult.error(f"Error: invalid ISO datetime format '{at}'. Expected format: YYYY-MM-DDTHH:MM:SS")
            if dt.tzinfo is None:
                if err := self._validate_timezone(self._default_timezone):
                    return err
                dt = dt.replace(tzinfo=ZoneInfo(self._default_timezone))
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True
        else:
            return ToolResult.error("Error: either every_seconds, cron_expr, or at is required")

        job = self._cron.add_job(
            name=name or message[:30],
            schedule=schedule,
            message=message,
            deliver=True,
            channel=origin_channel,
            to=origin_chat_id,
            additional_destinations=destinations,
            delete_after_run=delete_after,
            session_key=session_key,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            origin_metadata=origin_metadata,
        )
        return f"Created job '{job.name}' (id: {job.id})"

    def _format_timing(self, schedule: CronSchedule) -> str:
        """Format schedule as a human-readable timing string."""
        if schedule.kind == "cron":
            tz = f" ({schedule.tz})" if schedule.tz else ""
            return f"cron: {schedule.expr}{tz}"
        if schedule.kind == "every" and schedule.every_ms:
            ms = schedule.every_ms
            if ms % 3_600_000 == 0:
                return f"every {ms // 3_600_000}h"
            if ms % 60_000 == 0:
                return f"every {ms // 60_000}m"
            if ms % 1000 == 0:
                return f"every {ms // 1000}s"
            return f"every {ms}ms"
        if schedule.kind == "at" and schedule.at_ms:
            return f"at {self._format_timestamp(schedule.at_ms, self._display_timezone(schedule))}"
        return schedule.kind

    def _format_state(self, state: CronJobState, schedule: CronSchedule) -> list[str]:
        """Format job run state as display lines."""
        lines: list[str] = []
        display_tz = self._display_timezone(schedule)
        if state.last_run_at_ms:
            info = (
                f"  Last run: {self._format_timestamp(state.last_run_at_ms, display_tz)}"
                f" — {state.last_status or 'unknown'}"
            )
            if state.last_error:
                info += f" ({state.last_error})"
            lines.append(info)
        if state.next_run_at_ms:
            lines.append(f"  Next run: {self._format_timestamp(state.next_run_at_ms, display_tz)}")
        return lines

    @staticmethod
    def _system_job_purpose(job: CronJob) -> str:
        if job.name == "dream":
            return "Dream memory consolidation for long-term memory."
        return "System-managed internal job."

    def _list_jobs(self) -> str:
        jobs = self._cron.list_jobs()
        if not jobs:
            return "No scheduled jobs."
        lines: list[str] = []
        for j in jobs:
            timing = self._format_timing(j.schedule)
            parts = [f"- {j.name} (id: {j.id}, {timing})"]
            destinations = j.payload.delivery_destinations()
            if destinations:
                parts.append("  Delivery: " + ", ".join(
                    f"{destination.channel}:{destination.to}" for destination in destinations
                ))
            if j.payload.kind == "system_event":
                parts.append(f"  Purpose: {self._system_job_purpose(j)}")
                parts.append("  Protected: visible for inspection, but cannot be removed.")
            parts.extend(self._format_state(j.state, j.schedule))
            lines.append("\n".join(parts))
        return "Scheduled jobs:\n" + "\n".join(lines)

    def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return ToolResult.error("Error: job_id is required for remove")
        job = self._cron.get_job(job_id)
        if job is None:
            return f"Job {job_id} not found"
        try:
            result = self._cron.remove_job(
                job_id,
                job_instance_id=job.job_instance_id,
                definition_revision=job.definition_revision,
            )
        except (
            CronDefinitionTokenRequiredError,
            DefinitionStoreConflictError,
            CronDefinitionConflictError,
        ) as exc:
            return ToolResult.error(
                f"Error: could not remove job {job_id}: cron definition conflict ({exc}). "
                "Refresh the job list and retry."
            )
        if result == "removed":
            return f"Removed job {job_id}"
        if result == "protected":
            if job.name == "dream":
                return (
                    "Cannot remove job `dream`.\n"
                    "This is a system-managed Dream memory consolidation job for long-term memory.\n"
                    "It remains visible so you can inspect it, but it cannot be removed."
                )
            return (
                f"Cannot remove job `{job_id}`.\n"
                "This is a protected system-managed cron job."
            )
        return f"Job {job_id} not found"
