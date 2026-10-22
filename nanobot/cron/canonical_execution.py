"""Dormant, one-shot legacy-fresh cron computation from an exact occurrence claim.

This is callable only with explicitly injected authorities. Nothing in the
scheduler, gateway, CLI, cron tool, or migration admission invokes it yet.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Protocol, cast

from nanobot.agent.turn import (
    DeliveryTarget,
    HistoryMode,
    ScheduledTurnLink,
    TurnRequest,
    TurnResult,
    TurnSource,
)
from nanobot.cron.canonical_result import complete_text_cron_result
from nanobot.cron.completion import CronCompletionProjection
from nanobot.cron.definition_store import CronDefinition
from nanobot.cron.delivery_store import DeliveryOutboxRecord
from nanobot.cron.occurrence_store import OccurrenceClaim
from nanobot.cron.service import CronService
from nanobot.cron.types import CronPayload, cron_thread_id
from nanobot.session.manager import SessionManager


class SilentCronAgent(Protocol):
    async def execute_silent_cron_turn(self, request: TurnRequest) -> TurnResult: ...


def _request_for_claim(claim: OccurrenceClaim) -> TurnRequest:
    definition = CronDefinition.decode_wire(claim.job_snapshot, label="claimed cron definition")
    if (
        definition.job_id != claim.identity.job_id
        or definition.definition_revision != claim.identity.job_revision
        or not definition.definition["enabled"]
    ):
        raise ValueError("cron occurrence conflicts with its frozen definition identity")
    payload = CronPayload.from_store_dict(definition.definition["payload"])
    channel, chat_id = payload.channel or "cli", str(payload.to or "direct")
    if (
        payload.kind != "agent_turn"
        or payload.execution_mode != "legacy_fresh"
        or payload.session_key != f"cron:{definition.job_id}"
        or payload.origin_channel != channel or payload.origin_chat_id != chat_id
        or not payload.message.strip()
    ):
        raise ValueError("claimed cron job is not a complete legacy_fresh agent turn")
    frozen = claim.destination_policy
    for key in ("deliver", "channel", "to", "additionalDestinations", "channelMeta",
                "sessionKey", "originChannel", "originChatId", "originMetadata"):
        if frozen.get(key) != definition.definition["payload"][key]:
            raise ValueError("claimed cron destination policy conflicts with frozen definition")
    destinations = payload.delivery_destinations()
    thread_id = cron_thread_id(payload.channel_meta)
    visible_key = f"{channel}:{chat_id}" + (f":topic:{thread_id}" if thread_id is not None else "")
    instruction = payload.message
    if payload.planning_mode == "on":
        instruction = "Plan this task before taking actions.\n\n" + instruction
    elif payload.planning_mode == "off":
        instruction = "Complete this task directly; do not run a separate planning stage.\n\n" + instruction
    return TurnRequest(
        content=instruction,
        source=TurnSource.CRON,
        session_key=f"cron:{definition.job_id}",
        route=DeliveryTarget(channel=channel, chat_id=chat_id, thread_id=str(thread_id) if thread_id is not None else None),
        run_id=claim.run_id,
        history_mode=HistoryMode.FRESH,
        scheduled_link=ScheduledTurnLink(
            job_id=definition.job_id,
            job_name=definition.definition["name"],
            instruction=instruction,
            visible_session_key=visible_key,
            additional_destinations=tuple(
                DeliveryTarget(channel=destination.channel, chat_id=destination.to)
                for destination in destinations
                if (destination.channel, destination.to) != (channel, chat_id)
            ),
            occurrence_id=claim.occurrence_id,
        ),
        metadata={"cron_job_id": definition.job_id, "cron_occurrence_id": claim.occurrence_id,
                  **({"message_thread_id": thread_id} if thread_id is not None else {})},
    )


def build_claimed_fresh_request(claim: OccurrenceClaim) -> TurnRequest:
    """Validate a frozen route and build a request without consuming its claim."""
    return _request_for_claim(claim)


async def execute_claimed_fresh_text(
    service: CronService,
    agent: SilentCronAgent,
    claim: OccurrenceClaim,
    *,
    visible_sessions: SessionManager | None = None,
) -> tuple[TurnResult, CronCompletionProjection, DeliveryOutboxRecord]:
    """Compute once, complete, and prepare the outbox without sending anything.

    The one-shot start gate is consumed before awaiting the model, even if
    cancellation or an unsupported result leaves the ledger claimed. A caller
    must reconcile that claim; it must never replay the turn automatically.
    """
    if service.occurrence_store is None or service.completion_projector is None:
        raise ValueError("claimed cron execution requires occurrence and completion authorities")
    if service.delivery_runtime is None:
        raise ValueError("claimed cron execution requires an explicit delivery outbox authority")
    if not isinstance(cast(object, claim), OccurrenceClaim):
        raise TypeError("claimed cron execution requires an exact occurrence claim")
    if not callable(getattr(agent, "execute_silent_cron_turn", None)):
        raise TypeError("claimed cron execution requires a silent canonical agent")
    request = _request_for_claim(claim)
    with service.occurrence_store.acquire_execution_fence(claim) as fence:
        service.occurrence_store.begin_execution(claim, fence)
        result = await agent.execute_silent_cron_turn(request)
        projection, outbox = complete_text_cron_result(service, claim, result, fence)
        if visible_sessions is not None:
            # Import at the adapter edge: the standalone recovery linker also
            # validates frozen turns using the request builder above.
            from nanobot.cron.visible_link import link_completed_text_result

            link_completed_text_result(service, visible_sessions, claim.occurrence_id)
        return result, projection, outbox
