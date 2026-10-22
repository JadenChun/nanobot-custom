"""Dormant composition for one already-claimed canonical text occurrence.

This module constructs no runtime authority.  Every capability is supplied by
the caller, and no scheduler, gateway, bus, channel manager, or provider path
imports this entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

from nanobot.agent.turn import RunStatus, TurnResult
from nanobot.cron.callback_store import (
    CronCallbackObservation,
    CronCallbackRecord,
    CronCallbackStore,
)
from nanobot.cron.canonical_execution import SilentCronAgent, build_claimed_fresh_request
from nanobot.cron.canonical_result import complete_text_cron_result
from nanobot.cron.channel_sender import CronChannelSender
from nanobot.cron.completion import CronCompletionProjection
from nanobot.cron.definition_store import CronDefinition
from nanobot.cron.delivery_runtime import SenderResult, UnknownOutcome
from nanobot.cron.delivery_store import DeliveryOutboxRecord
from nanobot.cron.job_hooks import CronHookOutcome, CronJobHooks
from nanobot.cron.occurrence_store import OccurrenceClaim
from nanobot.cron.quota_recovery import CronQuotaRecovery
from nanobot.cron.service import CronService
from nanobot.cron.types import CronPayload, cron_thread_id
from nanobot.session.manager import SessionManager


class CronCompositionPreflightError(ValueError):
    """Frozen work is outside the deliberately narrow P3 composition."""


@dataclass(frozen=True, slots=True)
class ClaimedTextCompositionResult:
    turn: TurnResult
    completion: CronCompletionProjection
    outbox: DeliveryOutboxRecord | None
    callback: CronCallbackRecord | None
    deliveries: tuple[SenderResult, ...]
    hooks: CronHookOutcome | None = None


def _preflight(
    service: CronService,
    agent: SilentCronAgent,
    claim: OccurrenceClaim,
    sender: CronChannelSender,
    visible_sessions: SessionManager | None,
    callback_authority: CronCallbackStore | None,
    job_hooks: CronJobHooks | None,
) -> None:
    if service.occurrence_store is None or service.completion_projector is None:
        raise CronCompositionPreflightError(
            "composition requires explicit occurrence and completion authorities"
        )
    if service.delivery_runtime is None:
        raise CronCompositionPreflightError("composition requires an explicit delivery authority")
    if not isinstance(cast(object, claim), OccurrenceClaim):
        raise TypeError("composition requires an exact occurrence claim")
    if not callable(getattr(agent, "execute_silent_cron_turn", None)):
        raise TypeError("composition requires an explicitly injected silent agent")
    if not isinstance(cast(object, sender), CronChannelSender):
        raise TypeError("composition requires an exact receipt-capable channel sender")
    if visible_sessions is not None and not isinstance(cast(object, visible_sessions), SessionManager):
        raise TypeError("visible session authority must be a SessionManager")
    if callback_authority is not None and not isinstance(cast(object, callback_authority), CronCallbackStore):
        raise TypeError("callback authority must be a CronCallbackStore")

    # This validates the exact definition/claim identity and complete frozen
    # route before begin_execution can consume the one-shot computation grant.
    build_claimed_fresh_request(claim)
    definition = CronDefinition.decode_wire(claim.job_snapshot, label="claimed cron definition")
    payload = CronPayload.from_store_dict(definition.definition["payload"])
    if (payload.verifiers or payload.post_delivery_command or payload.alert_to) and job_hooks is None:
        raise CronCompositionPreflightError("configured job contracts require an explicit hook adapter")
    try:
        cron_thread_id(payload.channel_meta)
    except ValueError as exc:
        raise CronCompositionPreflightError(str(exc)) from exc
    destinations = payload.delivery_destinations()
    if payload.deliver and not destinations:
        raise CronCompositionPreflightError("delivery requires at least one exact destination")
    if any(destination.channel != "telegram" for destination in destinations):
        raise CronCompositionPreflightError(
            "only exact Telegram receipt destinations are supported"
        )


def _observation(
    claim: OccurrenceClaim,
    projection: CronCompletionProjection,
) -> CronCallbackObservation:
    return CronCallbackObservation(
        occurrence_id=claim.occurrence_id,
        run_id=projection.run_id,
        job_id=claim.identity.job_id,
        job_revision=claim.identity.job_revision,
        generation=claim.generation,
        kind="result",
        result_status=projection.status,
    )


async def compose_claimed_text_occurrence(
    service: CronService,
    agent: SilentCronAgent,
    claim: OccurrenceClaim,
    sender: CronChannelSender,
    *,
    visible_sessions: SessionManager | None = None,
    callback_authority: CronCallbackStore | None = None,
    delivery_owner: str = "cron-composition",
    pre_begin: Callable[[], object] | None = None,
    job_hooks: CronJobHooks | None = None,
    quota_recovery: CronQuotaRecovery | None = None,
) -> ClaimedTextCompositionResult:
    """Compute and settle one exact claim, then callback and dispatch once.

    Once execution begins, every unsupported, cancelled, or failed path is a
    reconciliation boundary.  This API has no recovery/resume mode and never
    calls the model or sender a second time.
    """
    _preflight(service, agent, claim, sender, visible_sessions, callback_authority, job_hooks)
    if not isinstance(cast(object, delivery_owner), str) or not delivery_owner.strip():
        raise CronCompositionPreflightError("delivery owner must be non-empty")
    request = build_claimed_fresh_request(claim)
    assert service.occurrence_store is not None

    with service.occurrence_store.acquire_execution_fence(claim) as fence:
        if pre_begin is not None:
            pre_begin()
        service.occurrence_store.begin_execution(claim, fence)
        if quota_recovery is not None and quota_recovery.defer(claim):
            turn = TurnResult(run_id=claim.run_id, status=RunStatus.POLICY_BLOCKED,
                              error="waiting_for_quota_recovery")
            projection = service.complete_occurrence(claim, {
                "status": "skipped", "reason": "waiting_for_quota_recovery",
                "delivery": {"version": 1, "deliver": False, "messages": []}},
                run_id=claim.run_id, fence=fence)
            return ClaimedTextCompositionResult(turn, projection, None, None, ())
        turn = await agent.execute_silent_cron_turn(request)
        quota_event = await quota_recovery.event(claim, turn) if quota_recovery is not None else None
        runtime_metadata: dict[str, object] = ({"quota_recovery": quota_event.model_dump()}
                                              if quota_event is not None else {})
        blocked_reason = None
        if job_hooks is not None:
            if turn.run_id != claim.run_id or turn.sent_messages:
                raise CronCompositionPreflightError("job hooks require the exact silent result")
            if turn.status is RunStatus.COMPLETED and not turn.error:
                verified, reason = await job_hooks.verify(claim, turn)
                if not verified:
                    blocked_reason = reason
            else:
                blocked_reason = turn.status.value
        if quota_event is not None and quota_event.retry_at_ms is not None:
            blocked_reason = "quota_retry_scheduled"

        if blocked_reason is not None and turn.status not in {RunStatus.APPROVAL_REQUIRED, RunStatus.POLICY_BLOCKED}:
            projection = service.complete_occurrence(
                claim, {"status": "error", "run_status": turn.status.value,
                        "error": blocked_reason,
                        "runtime_metadata": runtime_metadata,
                        "delivery": {"version": 1, "deliver": False, "messages": []}},
                run_id=claim.run_id, fence=fence,
            )
            outbox = None
        elif turn.status in {RunStatus.APPROVAL_REQUIRED, RunStatus.POLICY_BLOCKED}:
            if turn.sent_messages or turn.run_id != claim.run_id:
                raise CronCompositionPreflightError(
                    "blocked cron result conflicts with its exact silent execution"
                )
            projection = service.complete_occurrence(
                claim,
                {
                    "status": "skipped",
                    "run_status": turn.status.value,
                    "runtime_metadata": runtime_metadata,
                    "delivery": {"version": 1, "deliver": False, "messages": []},
                },
                run_id=claim.run_id,
                fence=fence,
            )
            outbox = None
        else:
            # Dynamic unsupported results deliberately raise only after the
            # one-shot start is consumed, leaving the occurrence claimed for
            # reconciliation and preventing computation replay.
            projection, outbox = complete_text_cron_result(service, claim, turn, fence,
                runtime_metadata=runtime_metadata if quota_recovery is not None else None)
            if visible_sessions is not None:
                from nanobot.cron.visible_link import link_completed_text_result

                link_completed_text_result(service, visible_sessions, claim.occurrence_id)

        callback_record = None
        if callback_authority is not None:
            # invoke_once absorbs ordinary callback failure into durable
            # ``failed`` state. Cancellation intentionally leaves ``invoking``.
            callback_record = await callback_authority.invoke_once(
                _observation(claim, projection)
            )

        outcomes: list[SenderResult] = []
        if outbox is not None:
            for frozen in outbox.destinations:
                current = service.get_delivery(outbox.outbox_id)
                current_destination = next(
                    item for item in current.destinations
                    if item.destination_id == frozen.destination_id
                )
                delivery_claim = service.claim_delivery(
                    current.outbox_id,
                    current_destination.destination_id,
                    current.record_revision,
                    owner=delivery_owner,
                )
                try:
                    outcomes.append(await service.dispatch_delivery(delivery_claim, sender=sender))
                except Exception as exc:
                    # E3 has already durably classified an exception as
                    # uncertain.  Surface that content-free classification
                    # while continuing frozen-order fanout; this claim can
                    # never be dispatched again without reconciliation.
                    outcomes.append(UnknownOutcome(f"exception:{type(exc).__name__}"))

        if quota_recovery is not None:
            quota_recovery.settle(service.occurrence_store.get_completed_durable(claim.occurrence_id),
                delivered=outbox is not None and service.delivery_status(outbox.outbox_id) == "delivered")
        hooks = await job_hooks.after_delivery(claim, projection, tuple(outcomes), sender,
            blocked_reason=blocked_reason,
            deferred_retry_at_ms=quota_event.retry_at_ms if quota_event is not None else None) if job_hooks is not None else None
        return ClaimedTextCompositionResult(
            turn=turn,
            completion=projection,
            outbox=outbox,
            callback=callback_record,
            deliveries=tuple(outcomes),
            hooks=hooks,
        )


__all__ = [
    "ClaimedTextCompositionResult",
    "CronCompositionPreflightError",
    "compose_claimed_text_occurrence",
]
