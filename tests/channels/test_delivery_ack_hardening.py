"""Hardening tests for the delivery acknowledgement layer."""

import asyncio
import dataclasses

import pytest

import nanobot.channels.manager as manager_mod
from nanobot.bus.events import DeliveryResult, OutboundMessage
from nanobot.bus.queue import DELIVERY_ACK_TIMEOUT_S, MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import ChannelsConfig, Config


class MockChannel(BaseChannel):
    name = "mock"
    display_name = "Mock"

    def __init__(self, config, bus, *, fail_times: int = 0):
        super().__init__(config, bus)
        self.send_calls = 0
        self.fail_times = fail_times

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send(self, msg):
        self.send_calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("send boom")


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(manager_mod, "_SEND_RETRY_DELAYS", (0, 0, 0))


@pytest.fixture
def bus():
    return MessageBus()


@pytest.fixture
def manager(bus):
    return ChannelManager(Config(), bus)


async def _dispatcher(manager):
    task = asyncio.create_task(manager._dispatch_outbound())
    await asyncio.sleep(0)
    return task


# 1. runtime ack cannot enter the dataclass serialization path
def test_delivery_ack_is_not_a_dataclass_field():
    msg = OutboundMessage(channel="mock", chat_id="c", content="hi")
    # Not a declared field...
    assert "delivery_ack" not in {f.name for f in dataclasses.fields(msg)}
    # ...and therefore absent from asdict()/astuple().
    assert "delivery_ack" not in dataclasses.asdict(msg)
    assert len(dataclasses.astuple(msg)) == len(dataclasses.fields(msg))
    # It still exists as a runtime attribute.
    assert msg.delivery_ack is None


# 2 + 4. timeout behavior explicit, safe, and NOT a success
@pytest.mark.asyncio
async def test_timeout_is_explicit_failure_not_success(bus):
    # No dispatcher running -> acknowledgement never resolves.
    result = await bus.publish_outbound_and_wait(
        OutboundMessage(channel="mock", chat_id="c", content="hi"), timeout=0.05
    )
    assert isinstance(result, DeliveryResult)
    assert result.success is False
    assert "timed out" in (result.error or "")


# 3. late dispatcher activity after timeout must not crash
@pytest.mark.asyncio
async def test_late_send_after_timeout_does_not_crash(bus, manager):
    chan = MockChannel({}, bus)
    manager.channels["mock"] = chan
    msg = OutboundMessage(channel="mock", chat_id="c", content="late")
    # Waiter times out before any dispatcher exists.
    result = await bus.publish_outbound_and_wait(msg, timeout=0.05)
    assert result.success is False
    assert msg.delivery_ack.done()  # cancelled by wait_for

    # Now a dispatcher picks up the still-queued message and sends it.
    task = await _dispatcher(manager)
    try:
        await asyncio.sleep(0.05)
        assert chan.send_calls == 1          # client still received it
        assert not task.done()               # dispatcher survived
        # Resolving a cancelled ack is a safe no-op.
        manager._resolve_ack(msg, DeliveryResult(success=True, error="late"))
    finally:
        task.cancel()


# 5. configured default accommodates the real retry envelope
def test_default_timeout_covers_retry_envelope():
    # Telegram read/connect timeout is 30s; manager retries 3 attempts -> the
    # default must exceed a single attempt and a couple of retries.
    assert DELIVERY_ACK_TIMEOUT_S >= 90.0
    # ...but must stay bounded so a cron run cannot stall indefinitely.
    assert DELIVERY_ACK_TIMEOUT_S <= 600.0
    assert ChannelsConfig().delivery_ack_timeout == DELIVERY_ACK_TIMEOUT_S


# 6. final-message ack is not prematurely completed by progress handling
@pytest.mark.asyncio
async def test_progress_message_does_not_complete_final_ack(bus):
    # Cron runs with task_update_mode="result" (progress suppressed).
    cfg = Config()
    cfg.channels.task_update_mode = "result"
    manager = ChannelManager(cfg, bus)
    chan = MockChannel({}, bus)
    manager.channels["mock"] = chan
    task = await _dispatcher(manager)
    try:
        # A progress message (filtered from delivery in result mode).
        await bus.publish_outbound(OutboundMessage(
            channel="mock", chat_id="c", content="working",
            metadata={"_progress": True},
        ))
        # The final client result carries the ack.
        final = OutboundMessage(channel="mock", chat_id="c", content="final")
        result = await bus.publish_outbound_and_wait(final, timeout=2.0)
        assert result.success is True
        # Only the final message was actually sent; progress was filtered.
        assert chan.send_calls == 1
        assert final.delivery_ack.result().success is True
    finally:
        task.cancel()