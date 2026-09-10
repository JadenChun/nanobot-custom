"""Async message queue for decoupled channel-agent communication."""

import asyncio

from nanobot.bus.events import DeliveryResult, InboundMessage, OutboundMessage

# Bounded wait for a delivery acknowledgement.  Covers the dispatcher's outer
# retry/backoff (send_max_retries attempts, 1s + 2s backoff) plus the channel's
# own bounded send retries, with headroom for a short queue backlog before the
# dispatcher picks the message up.  Prevents a caller hanging forever if the
# dispatcher is stopped or an acknowledgement is never resolved.
DELIVERY_ACK_TIMEOUT_S = 30.0


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels (enqueue only)."""
        await self.outbound.put(msg)

    async def publish_outbound_and_wait(
        self,
        msg: OutboundMessage,
        *,
        timeout: float = DELIVERY_ACK_TIMEOUT_S,
    ) -> DeliveryResult:
        """Enqueue an outbound message and await its terminal delivery result.

        ``publish_outbound`` only enqueues.  This helper attaches an internal
        acknowledgement future, enqueues, and waits for the channel-manager
        dispatcher to resolve it after the FINAL send attempt (post-retry).  A
        bounded ``timeout`` yields a failure result instead of hanging if the
        acknowledgement is never resolved (dispatcher stopped, unexpected bug).
        """
        loop = asyncio.get_running_loop()
        ack: asyncio.Future[DeliveryResult] = loop.create_future()
        msg.delivery_ack = ack
        await self.publish_outbound(msg)
        try:
            return await asyncio.wait_for(ack, timeout=timeout)
        except asyncio.TimeoutError:
            return DeliveryResult(
                success=False, error="delivery acknowledgement timed out"
            )

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
