"""Dormant, text-only canonical turn-result settlement for claimed cron work.

This adapter never invokes the agent, schedules a job, links visible chat, or
sends to a channel. A caller must separately own the exact occurrence claim
and suppress ordinary turn publication and message-tool sends before running
the model. Unsupported results remain claimed for explicit reconciliation.
"""

from __future__ import annotations

from typing import Mapping, cast

from nanobot.agent.turn import RunStatus, TurnResult
from nanobot.cron.completion import CronCompletionProjection
from nanobot.cron.delivery_store import DeliveryOutboxRecord
from nanobot.cron.occurrence_store import OccurrenceClaim, OccurrenceExecutionFence
from nanobot.cron.service import CronService


class UnsupportedCronResultError(ValueError):
    """The canonical result cannot be safely represented by the text-only outbox."""


def complete_text_cron_result(
    service: CronService,
    claim: OccurrenceClaim,
    result: TurnResult,
    fence: OccurrenceExecutionFence,
    *,
    runtime_metadata: Mapping[str, object] | None = None,
) -> tuple[CronCompletionProjection, DeliveryOutboxRecord]:
    """Durably settle one exact completed turn and prepare, but never dispatch, its outbox.

    The occurrence claim freezes the destination policy. An empty canonical
    ``sent_messages`` ledger is authoritative; an already-sent explicit message
    cannot be rolled into this uniform text-only fanout without risking a
    duplicate or a lost channel-specific result. A failure after completion
    must be recovered by projecting that occurrence and preparing its outbox,
    never by rerunning the agent.
    """
    if not isinstance(cast(object, claim), OccurrenceClaim) or not isinstance(cast(object, result), TurnResult):
        raise TypeError("cron result settlement requires a claim and canonical TurnResult")
    if result.run_id != claim.run_id:
        raise UnsupportedCronResultError("cron result must prove the exact occurrence run ID")
    if result.status is not RunStatus.COMPLETED or result.error:
        raise UnsupportedCronResultError("only completed, error-free cron results are supported")
    if result.sent_messages:
        raise UnsupportedCronResultError("explicit message sends require a separate delivery adapter")
    if not isinstance(result.content, str) or not result.content.strip():
        raise UnsupportedCronResultError("cron result must have non-empty text")
    if len(result.content.encode("utf-8")) > 256 * 1024:
        raise UnsupportedCronResultError("cron result exceeds the text-only sender limit")
    deliver = claim.destination_policy.get("deliver")
    if type(deliver) is not bool:
        raise UnsupportedCronResultError("claim has no exact boolean delivery policy")

    projection = service.complete_occurrence(
        claim,
        {
            "status": "ok",
            "result": result.content,
            **({"runtime_metadata": dict(runtime_metadata)} if runtime_metadata is not None else {}),
            "delivery": {
                "version": 1,
                "deliver": deliver,
                "messages": [{"content": result.content, "media": []}],
            },
        },
        run_id=claim.run_id,
        fence=fence,
    )
    outbox = service.prepare_completed_delivery(claim.occurrence_id)
    return projection, outbox
