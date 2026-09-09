"""Build de-duplicated outbound deliveries for cron results."""

from collections.abc import Iterable

from nanobot.bus.events import OutboundMessage
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
