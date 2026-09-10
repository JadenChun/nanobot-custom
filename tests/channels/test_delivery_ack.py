"""Tests for the delivery acknowledgement layer.

Covers MessageBus.publish_outbound_and_wait + ChannelManager ack resolution
using the real queue/manager with mocked channels.
"""

import asyncio

import pytest

import nanobot.channels.manager as manager_mod
from nanobot.bus.events import DeliveryResult, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


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
    # Make retry backoff instant for deterministic, fast tests.
    monkeypatch.setattr(manager_mod, "_SEND_RETRY_DELAYS", (0, 0, 0))


@pytest.fixture
def bus():
    return MessageBus()


@pytest.fixture
def manager(bus):
    return ChannelManager(Config(), bus)


async def _with_dispatcher(manager):
    task = asyncio.create_task(manager._dispatch_outbound())
    await asyncio.sleep(0)  # let it start
    return task


@pytest.mark.asyncio
async def test_publish_outbound_is_enqueue_only(bus, manager):
    """publish_outbound must enqueue and return without any channel send."""
    chan = MockChannel({}, bus)
    manager.channels["mock"] = chan
    msg = OutboundMessage(channel="mock", chat_id="c", content="hi")

    await bus.publish_outbound(msg)

    assert bus.outbound.qsize() == 1
    assert chan.send_calls == 0
    assert msg.delivery_ack is None  # fire-and-forget default preserved


@pytest.mark.asyncio
async def test_ack_success_first_attempt(bus, manager):
    chan = MockChannel({}, bus)
    manager.channels["mock"] = chan
    task = await _with_dispatcher(manager)
    try:
        result = await bus.publish_outbound_and_wait(
            OutboundMessage(channel="mock", chat_id="c", content="hi"), timeout=2.0
        )
        assert isinstance(result, DeliveryResult)
        assert result.success is True
        assert result.error is None
        assert chan.send_calls == 1
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_ack_success_after_retry(bus, manager):
    chan = MockChannel({}, bus, fail_times=1)  # first attempt fails
    manager.channels["mock"] = chan
    task = await _with_dispatcher(manager)
    try:
        result = await bus.publish_outbound_and_wait(
            OutboundMessage(channel="mock", chat_id="c", content="hi"), timeout=2.0
        )
        assert result.success is True
        assert chan.send_calls == 2  # failed once, then succeeded
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_ack_failure_all_retries(bus, manager):
    chan = MockChannel({}, bus, fail_times=99)
    manager.channels["mock"] = chan
    task = await _with_dispatcher(manager)
    try:
        result = await bus.publish_outbound_and_wait(
            OutboundMessage(channel="mock", chat_id="c", content="hi"), timeout=2.0
        )
        assert result.success is False
        assert "RuntimeError" in (result.error or "")
        # Default send_max_retries == 3.
        assert chan.send_calls == 3
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_ack_unknown_channel_fails(bus, manager):
    task = await _with_dispatcher(manager)
    try:
        result = await bus.publish_outbound_and_wait(
            OutboundMessage(channel="nope", chat_id="c", content="hi"), timeout=2.0
        )
        assert result.success is False
        assert "unknown channel" in (result.error or "")
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_non_ack_message_failure_is_swallowed(bus, manager):
    chan = MockChannel({}, bus, fail_times=99)
    manager.channels["mock"] = chan
    task = await _with_dispatcher(manager)
    try:
        # No ack attached: legacy fire-and-forget must not raise or kill the loop.
        await bus.publish_outbound(
            OutboundMessage(channel="mock", chat_id="c", content="hi")
        )
        await asyncio.sleep(0.05)
        assert chan.send_calls == 3
        assert not task.done()  # dispatcher still alive
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_ack_resolved_exactly_once(bus, manager):
    chan = MockChannel({}, bus)
    manager.channels["mock"] = chan
    task = await _with_dispatcher(manager)
    msg = OutboundMessage(channel="mock", chat_id="c", content="hi")
    try:
        result = await bus.publish_outbound_and_wait(msg, timeout=2.0)
        assert result.success is True
        # A second resolve attempt must be a no-op (ack already done).
        manager._resolve_ack(msg, DeliveryResult(status="failed", error="late"))
        assert msg.delivery_ack.result().success is True
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_ack_timeout_returns_failure_and_dispatcher_still_usable(bus, manager):
    """No dispatcher running -> bounded timeout; then a later dispatcher works."""
    timed_out = await bus.publish_outbound_and_wait(
        OutboundMessage(channel="mock", chat_id="c", content="hi"), timeout=0.05
    )
    assert timed_out.success is False
    assert "timed out" in (timed_out.error or "")

    # Dispatcher started afterwards must still deliver fresh acked messages.
    chan = MockChannel({}, bus)
    manager.channels["mock"] = chan
    task = await _with_dispatcher(manager)
    try:
        ok = await bus.publish_outbound_and_wait(
            OutboundMessage(channel="mock", chat_id="c", content="again"), timeout=2.0
        )
        assert ok.success is True
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_two_acked_messages_get_independent_results(bus, manager):
    ok_chan = MockChannel({}, bus)
    bad_chan = MockChannel({}, bus, fail_times=99)
    manager.channels["ok"] = ok_chan
    manager.channels["bad"] = bad_chan
    task = await _with_dispatcher(manager)
    try:
        good = asyncio.create_task(bus.publish_outbound_and_wait(
            OutboundMessage(channel="ok", chat_id="c", content="good"), timeout=2.0
        ))
        bad = asyncio.create_task(bus.publish_outbound_and_wait(
            OutboundMessage(channel="bad", chat_id="c", content="bad"), timeout=2.0
        ))
        r_good, r_bad = await asyncio.gather(good, bad)
        assert r_good.success is True
        assert r_bad.success is False
    finally:
        task.cancel()