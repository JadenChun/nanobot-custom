"""Dormant, recoverable visible-chat reference for a durably completed cron turn.

This does not send a message or make an unfinished occurrence look successful.
It is invoked explicitly with the completed occurrence ID, including on retry
after a session write failed. The cron detail session remains the full trace.
"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from nanobot.cron.canonical_execution import build_claimed_fresh_request
from nanobot.cron.completion import CronCompletionConflictError
from nanobot.cron.occurrence_store import OccurrenceClaim
from nanobot.cron.service import CronService
from nanobot.session.manager import Session, SessionManager, SessionWriteConflict


def link_completed_text_result(
    service: CronService, sessions: SessionManager, occurrence_id: str,
) -> str:
    """Link one text completion to its frozen origin chat without rerunning it."""
    if service.occurrence_store is None or service.completion_projector is None:
        raise ValueError("visible cron linking requires durable occurrence and completion authorities")
    if not isinstance(cast(object, sessions), SessionManager):
        raise TypeError("visible cron linking requires a session manager")
    projection = service.project_occurrence_completion(occurrence_id)
    record = service.occurrence_store.get_completed_durable(occurrence_id)
    claim = OccurrenceClaim(
        record.occurrence_id, record.identity, record.owner, record.generation,
        record.job_snapshot, record.destination_policy,
    )
    request = build_claimed_fresh_request(claim)  # Validate frozen definition and route, not an execution grant.
    completion = projection.completion
    result = completion.get("result")
    delivery = completion.get("delivery")
    if (
        set(completion) != {
            "status", "result", "delivery", "run_id", "occurrence_id",
            "job_id", "job_revision", "generation",
        }
        or completion.get("status") != "ok" or not isinstance(result, str) or not result.strip()
        or not isinstance(delivery, dict) or set(cast(dict[object, object], delivery)) != {"version", "deliver", "messages"}
        or type(cast(dict[object, object], delivery).get("deliver")) is not bool
        or cast(dict[object, object], delivery).get("deliver") != claim.destination_policy.get("deliver")
        or type(cast(dict[object, object], delivery).get("version")) is not int or delivery["version"] != 1
        or cast(dict[object, object], delivery).get("messages") != [{"content": result, "media": []}]
        or len(result.encode("utf-8")) > 256 * 1024
    ):
        raise CronCompletionConflictError("visible link requires a canonical text-only completion")
    link = request.scheduled_link
    if link is None or not link.visible_session_key:
        raise CronCompletionConflictError("completed cron turn has no frozen visible route")
    for attempt in range(3):
        if not sessions.get_or_create(link.visible_session_key).policy.persist:
            raise CronCompletionConflictError("visible cron link requires a durable chat session")
        # Absent managed routes must be created outside the files lock: the
        # existing mode authority owns migration/creation lock ordering.
        if sessions.read_session_metadata(link.visible_session_key) is None:
            sessions.upsert_session_metadata(link.visible_session_key, {}, create_if_absent=True, fsync=True)
        try:
            with sessions.locked_session_files():
                metadata = sessions.read_session_metadata(link.visible_session_key)
                if metadata is None:
                    raise SessionWriteConflict()
                # Read/merge/publication share the manager's reentrant files
                # lock. Never modify cached history or metadata before commit.
                visible = Session(key=link.visible_session_key, metadata=deepcopy(metadata["metadata"]))
                visible.add_scheduled_run(
                    job_id=record.identity.job_id, run_id=projection.run_id,
                    instruction=request.content, result=result,
                    detail_session_key=request.session_key or f"cron:{record.identity.job_id}",
                    occurrence_id=record.occurrence_id,
                )
                if not sessions.upsert_session_metadata(link.visible_session_key,
                    {Session.SCHEDULED_RING_KEY: visible.recent_scheduled_runs()}, fsync=True):
                    raise SessionWriteConflict()
            break
        except SessionWriteConflict:
            sessions.invalidate(link.visible_session_key)
            if attempt == 2:
                raise
    return link.visible_session_key
