"""Owner operational-alert tests for cron delivery (mocked channels only)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import nanobot.channels.manager as manager_mod
from nanobot.bus.events import DeliveryResult, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.cli.commands import _run_cron_job
from nanobot.config.schema import Config
from nanobot.cron.delivery import classify_delivery_results, send_owner_alert
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob, CronPayload, CronSchedule

GROUP = "-5340461568"
OWNER = "6344587670"


class RecordingChannel(BaseChannel):
    name = "telegram"
    display_name = "Telegram"

    def __init__(self, config, bus, *, fail_chats=()):
        super().__init__(config, bus)
        self.sent = []  # list[(chat_id, content)]
        self.fail_chats = set(fail_chats)

    async def start(self): pass
    async def stop(self): pass

    async def send(self, msg):
        if msg.chat_id in self.fail_chats:
            raise RuntimeError("send boom")
        self.sent.append((msg.chat_id, msg.content))


class _Sessions:
    def __init__(self): self.saved = []
    def get_or_create(self, key):
        return SimpleNamespace(retain_recent_legal_suffix=lambda n: None)
    def save(self, s): self.saved.append(s)


class _Tools:
    def get(self, name): return None  # no cron tool, no message tool


class _Agent:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc
        self.tools = _Tools()
        self.sessions = _Sessions()
        self.failures = []
    async def process_direct(self, *a, **k):
        if self._exc:
            raise self._exc
        return self._resp
    def record_task_failure(self, **k):
        self.failures.append(k)


def _resp(verdict="PASS", content="IDEA"):
    return OutboundMessage(channel="telegram", chat_id=GROUP, content=content,
                           metadata={"_verification": verdict})


def _job(**over):
    payload = dict(message="m", deliver=True, channel="telegram", to=GROUP,
                   skip_verification=False, alert_channel="telegram", alert_to=OWNER)
    payload.update(over)
    return CronJob(id="j1", name="Daily Content Idea",
                   schedule=CronSchedule(kind="every", every_ms=60_000),
                   payload=CronPayload(**payload))


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(manager_mod, "_SEND_RETRY_DELAYS", (0, 0, 0))


def _manager(bus, *, fail_chats=()):
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus, fail_chats=fail_chats)
    mgr.channels["telegram"] = chan
    return mgr, chan


async def _run(bus, mgr, job, agent):
    task = asyncio.create_task(mgr._dispatch_outbound())
    await asyncio.sleep(0)
    try:
        await _run_cron_job(agent, bus, job)
        await asyncio.sleep(0.05)
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_pass_group_success_owner_silent():
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr, chan = _manager(bus)
    await _run(bus, mgr, _job(), _Agent(resp=_resp("PASS")))
    assert [c for c, _ in chan.sent] == [GROUP]
    assert len([c for c, _ in chan.sent if c == OWNER]) == 0


@pytest.mark.asyncio
async def test_generation_exception_group_zero_owner_once():
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr, chan = _manager(bus)
    agent = _Agent(exc=RuntimeError("gen boom"))
    with pytest.raises(RuntimeError):
        await _run(bus, mgr, _job(), agent)
    owners = [x for x in chan.sent if x[0] == OWNER]
    assert len([c for c, _ in chan.sent if c == GROUP]) == 0
    assert len(owners) == 1
    assert len(agent.failures) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["FAIL", "PARTIAL", "MISSING"])
async def test_verification_failure_group_zero_owner_once(verdict):
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr, chan = _manager(bus)
    r = OutboundMessage(channel="telegram", chat_id=GROUP, content="IDEA",
                        metadata={} if verdict == "MISSING" else {"_verification": verdict})
    agent = _Agent(resp=r)
    await _run(bus, mgr, _job(), agent)
    assert len([c for c, _ in chan.sent if c == GROUP]) == 0
    assert len([c for c, _ in chan.sent if c == OWNER]) == 1
    assert len(agent.failures) == 1


@pytest.mark.asyncio
async def test_group_delivery_failed_owner_once():
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr, chan = _manager(bus, fail_chats={GROUP})
    agent = _Agent(resp=_resp("PASS"))
    await _run(bus, mgr, _job(), agent)
    owners = [x for x in chan.sent if x[0] == OWNER]
    assert len(owners) == 1
    assert "Delivery Failed" in owners[0][1]
    assert len(agent.failures) == 1


@pytest.mark.asyncio
async def test_group_delivery_unknown_owner_once_unconfirmed_wording():
    # No dispatcher: acknowledgement times out -> UNKNOWN.
    bus = MessageBus(outbound_ack_timeout=0.05)
    agent = _Agent(resp=_resp("PASS"))
    await _run_cron_job(agent, bus, _job())
    # drain enqueued messages (never sent; no dispatcher)
    queued = []
    while bus.outbound.qsize():
        queued.append(await bus.consume_outbound())
    to_group = [m for m in queued if m.chat_id == GROUP]
    to_owner = [m for m in queued if m.chat_id == OWNER]
    assert len(to_group) == 1
    assert len(to_owner) == 1
    assert "Unconfirmed" in to_owner[0].content
    assert "Delivery Failed" not in to_owner[0].content
    assert len(agent.failures) == 1


@pytest.mark.asyncio
async def test_owner_alert_transport_failure_no_recursion():
    # Owner channel fails; only one owner send is attempted (no recursion).
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr, chan = _manager(bus, fail_chats={GROUP, OWNER})
    agent = _Agent(resp=_resp("PASS"))
    await _run(bus, mgr, _job(), agent)
    # owner channel failed, so nothing recorded; ensure only one attempt total
    assert len(chan.sent) == 0
    assert len(agent.failures) == 1


@pytest.mark.asyncio
async def test_multiple_required_destinations_one_failure_owner_once():
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr, chan = _manager(bus, fail_chats={"-2"})
    from nanobot.cron.types import CronDestination
    job = _job(additional_destinations=[CronDestination("telegram", "-2")])
    agent = _Agent(resp=_resp("PASS"))
    await _run(bus, mgr, job, agent)
    assert len([c for c, _ in chan.sent if c == OWNER]) == 1
    assert len(agent.failures) == 1


@pytest.mark.asyncio
async def test_meta_style_job_silent():
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr, chan = _manager(bus)
    # Meta: deliver=false, no alert destination -> nothing anywhere.
    job = _job(deliver=False, skip_verification=True, alert_to=None)
    agent = _Agent(resp=_resp("PASS"))
    await _run(bus, mgr, job, agent)
    assert chan.sent == []


def test_classify_delivery_results():
    assert classify_delivery_results([]) == "success"
    assert classify_delivery_results([DeliveryResult("success")]) == "success"
    assert classify_delivery_results([DeliveryResult("success"), DeliveryResult("unknown")]) == "unknown"
    assert classify_delivery_results([DeliveryResult("unknown"), DeliveryResult("failed")]) == "failed"


@pytest.mark.asyncio
async def test_send_owner_alert_no_target_is_noop():
    bus = MessageBus()
    assert await send_owner_alert(bus, channel="telegram", to=None,
                                  job_name="J", failure_stage="Failed", reason="r") is None


def test_cronpayload_alert_fields_round_trip(tmp_path: Path):
    store = tmp_path / "cron" / "jobs.json"
    svc = CronService(store)
    job = svc.add_job(name="Daily Content Idea", schedule=CronSchedule(kind="every", every_ms=60_000),
                      message="m", deliver=True, channel="telegram", to=GROUP,
                      alert_channel="telegram", alert_to=OWNER)
    loaded = CronService(store).get_job(job.id)
    assert loaded is not None
    assert loaded.payload.alert_channel == "telegram"
    assert loaded.payload.alert_to == OWNER
    dest = loaded.payload.alert_destination()
    assert dest is not None and dest.to == OWNER and dest.channel == "telegram"


def test_cronpayload_alert_defaults_backward_compatible(tmp_path: Path):
    store = tmp_path / "cron" / "jobs.json"
    svc = CronService(store)
    job = svc.add_job(name="X", schedule=CronSchedule(kind="every", every_ms=60_000),
                      message="m", deliver=False, channel="telegram", to=GROUP)
    loaded = CronService(store).get_job(job.id)
    assert loaded.payload.alert_channel is None
    assert loaded.payload.alert_to is None
    assert loaded.payload.alert_destination() is None