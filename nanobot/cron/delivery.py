"""Build de-duplicated outbound deliveries for cron results."""

from collections.abc import Iterable

from loguru import logger

from nanobot.bus.events import DeliveryResult, OutboundMessage
from nanobot.cron.types import CronDestination


def build_result_messages(
    content: str,
    destinations: Iterable[CronDestination],
    sent_messages: Iterable[OutboundMessage] = (),
) -> list[OutboundMessage]:
    """Build final-result messages that were not already sent explicitly."""
    sent = {
        (message.channel, message.chat_id, message.content, tuple(message.media))
        for message in sent_messages
    }
    return [
        OutboundMessage(channel=destination.channel, chat_id=destination.to, content=content)
        for destination in destinations
        if (destination.channel, destination.to, content, ()) not in sent
    ]



def is_client_deliverable(verification_verdict, *, skip_verification: bool) -> bool:
    """Whether a client-facing scheduled result may be delivered to the group.

    FAIL CLOSED: for ANY client-facing job, group delivery requires an explicit
    PASS verification verdict.  ``skip_verification`` does not bypass the gate;
    it only controls whether the framework's internal verifier runs.  A
    missing/FAIL/PARTIAL verdict never delivers to the group.
    """
    return verification_verdict == "PASS"



def classify_delivery_results(results: Iterable[DeliveryResult]) -> str:
    """Aggregate per-destination transport results into one outcome.

    Returns "success" when every required destination was confirmed (or there
    were none), "failed" when any confirmed transport failure occurred, and
    "unknown" when nothing failed but at least one result was unconfirmed.
    """
    statuses = [r.status for r in results]
    if not statuses:
        return "success"
    if any(s == "failed" for s in statuses):
        return "failed"
    if any(s == "unknown" for s in statuses):
        return "unknown"
    return "success"


async def send_owner_alert(
    bus,
    *,
    channel: str | None,
    to: str | None,
    job_name: str,
    failure_stage: str,
    reason: str,
) -> DeliveryResult | None:
    """Send ONE concise owner operational-alert DM via the existing transport.

    Returns ``None`` when no owner alert destination is configured (no-op).
    The alert is sent with the delivery-acknowledged transport so its own
    transport result can be logged; a failed/unknown owner alert is only
    logged and NEVER re-alerts (no recursion).
    """
    destination = CronDestination(channel=channel or "telegram", to=str(to)) if to else None
    if destination is None:
        return None
    msg = OutboundMessage(
        channel=destination.channel,
        chat_id=destination.to,
        content=f"\u26a0\ufe0f {job_name} {failure_stage}\n\n{reason}",
        metadata={"_owner_alert": True},
    )
    result = await bus.publish_outbound_and_wait(msg)
    if result.status == "success":
        logger.info("Owner alert delivered: {} — {}", job_name, failure_stage)
    elif result.status == "failed":
        logger.error(
            "Owner alert transport FAILED: {} — {} ({})",
            job_name, failure_stage, result.error,
        )
    else:
        logger.warning(
            "Owner alert confirmation UNKNOWN: {} — {} ({})",
            job_name, failure_stage, result.error,
        )
    return result


def build_explicit_fanout_messages(
    destinations: list[CronDestination],
    sent_messages: Iterable[OutboundMessage],
) -> list[OutboundMessage]:
    """Copy explicit primary-destination messages to missing destinations."""
    if len(destinations) < 2:
        return []

    messages = list(sent_messages)
    primary = destinations[0]
    primary_messages = [
        message
        for message in messages
        if message.channel == primary.channel and message.chat_id == primary.to
    ]
    sent = {
        (message.channel, message.chat_id, message.content, tuple(message.media))
        for message in messages
    }

    copies: list[OutboundMessage] = []
    for message in primary_messages:
        for destination in destinations[1:]:
            signature = (
                destination.channel,
                destination.to,
                message.content,
                tuple(message.media),
            )
            if signature in sent:
                continue
            sent.add(signature)
            copies.append(OutboundMessage(
                channel=destination.channel,
                chat_id=destination.to,
                content=message.content,
                media=list(message.media),
            ))
    return copies
