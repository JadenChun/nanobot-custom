"""Async message queue for decoupled channel-agent communication."""

import asyncio

from loguru import logger

from nanobot.bus.events import DeliveryResult, InboundMessage, OutboundMessage

# Bounded wait for a delivery acknowledgement, derived from the REAL retry
# envelope in this repo:
#   * One HTTP request uses httpx *phase* timeouts (not additive for the caller,
#     but the per-request pathological ceiling is connect(30)+read(30)+write(5,
#     default)+pool(5) = 70s; a single stalled phase is typically ~30s).
#   * TelegramChannel._call_with_retry: 3 attempts (TimedOut only), 0.5s+1.0s
#     backoff.  _send_text also falls back from HTML to plain text, i.e. up to
#     6 HTTP attempts per channel.send().
#   * ChannelManager._send_with_retry: send_max_retries (default 3) attempts,
#     1s+2s backoff.
#   => up to 18 HTTP attempts.  Realistic network-failure worst case is
#      3 * (3*30 + 1.5) + 3 ~= 277.5s; the pathological ceiling is ~1272s.
# 300s covers the realistic retry envelope so normal transient failures resolve
# within the wait, while remaining bounded so a cron run cannot stall forever.
# It is intentionally shorter than the pathological ceiling: if it elapses the
# result is UNKNOWN (see DeliveryResult.status), never assumed failed.
# Overridable per call and via channels.delivery_ack_timeout.
DELIVERY_ACK_TIMEOUT_S = 300.0


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    """

    def __init__(self, *, outbound_ack_timeout: float | None = None):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        # Configured default wait for delivery acknowledgements (channels
        # .delivery_ack_timeout).  ``None`` falls back to the module default.
        self._outbound_ack_timeout = outbound_ack_timeout

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
        timeout: float | None = None,
    ) -> DeliveryResult:
        """Enqueue an outbound message and await its terminal delivery result.

        ``publish_outbound`` only enqueues.  This helper attaches an internal
        acknowledgement future, enqueues, and waits for the channel-manager
        dispatcher to resolve it after the FINAL send attempt (post-retry).  A
        bounded ``timeout`` yields a failure result instead of hanging if the
        acknowledgement is never resolved (dispatcher stopped, unexpected bug).
        """
        effective_timeout = (
            timeout if timeout is not None
            else (self._outbound_ack_timeout or DELIVERY_ACK_TIMEOUT_S)
        )
        loop = asyncio.get_running_loop()
        ack: asyncio.Future[DeliveryResult] = loop.create_future()
        msg.delivery_ack = ack
        await self.publish_outbound(msg)
        # ``asyncio.wait_for`` CANCELS ``ack`` on timeout.  That is safe: the
        # dispatcher's ``_resolve_ack`` checks ``ack.done()`` and becomes a
        # no-op for a cancelled future, so a late successful send neither
        # crashes the dispatcher nor revives the result.  The queued message is
        # unaffected and may still be delivered.
        try:
            return await asyncio.wait_for(ack, timeout=effective_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Delivery acknowledgement timed out after {}s for {}:{}; "
                "downstream send result is UNKNOWN",
                effective_timeout, msg.channel, msg.chat_id,
            )
            # UNKNOWN, not FAILED: the queued send may still complete late.
            return DeliveryResult(
                status="unknown", error="delivery acknowledgement timed out"
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
