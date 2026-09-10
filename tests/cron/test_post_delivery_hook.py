"""Post-delivery bookkeeping hook tests (mocked channels; real subprocesses)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import nanobot.channels.manager as manager_mod
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.cli.commands import _run_cron_job
from nanobot.config.schema import Config
from nanobot.cron.delivery import run_post_delivery_command
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob, CronPayload, CronSchedule

GROUP = "-5340461568"
OWNER = "6344587670"

_RECORDER = (
    "import json,sys;"
    "open(sys.argv[1],'a',encoding='utf-8').write(json.dumps(sys.argv[2:])+chr(10))"
)


class RecordingChannel(BaseChannel):
    name = "telegram"
    display_name = "Telegram"

    def __init__(self, config, bus, *, fail_chats=()):
        super().__init__(config, bus)
        self.sent = []
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
    def get(self, name): return None


class _Agent:
    def __init__(self, resp=None):
        self._resp = resp
        self.tools = _Tools()
        self.sessions = _Sessions()
        self.failures = []

    async def process_direct(self, *a, **k): return self._resp
    def record_task_failure(self, **k): self.failures.append(k)


def _resp(verdict="PASS", content="IDEA"):
    return OutboundMessage(channel="telegram", chat_id=GROUP, content=content,
                           metadata={"_verification": verdict})


def _job(record_path: Path | None = None, **over):
    payload = dict(message="m", deliver=True, channel="telegram", to=GROUP,
                   skip_verification=False, alert_channel="telegram", alert_to=OWNER)
    if record_path is not None:
        payload["post_delivery_command"] = [
            sys.executable, "-c", _RECORDER, str(record_path), "{ack_status}", "{date}",
        ]
    payload.update(over)
    return CronJob(id="j1", name="Daily Content Idea",
                   schedule=CronSchedule(kind="every", every_ms=60_000),
                   payload=CronPayload(**payload))


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(manager_mod, "_SEND_RETRY_DELAYS", (0, 0, 0))


async def _run(bus, mgr, job, agent):
    task = asyncio.create_task(mgr._dispatch_outbound())
    await asyncio.sleep(0)
    try:
        await _run_cron_job(agent, bus, job)
        await asyncio.sleep(0.05)
    finally:
        task.cancel()


def _recorded(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_ack_success_reports_success(tmp_path):
    rec = tmp_path / "ack.jsonl"
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    await _run(bus, mgr, _job(rec), _Agent(_resp("PASS")))
    assert [c for c, _ in chan.sent if c == GROUP]
    assert [r[0] for r in _recorded(rec)] == ["success"]


@pytest.mark.asyncio
async def test_ack_failed_reports_failed(tmp_path):
    rec = tmp_path / "ack.jsonl"
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus, fail_chats={GROUP})
    mgr.channels["telegram"] = chan
    await _run(bus, mgr, _job(rec), _Agent(_resp("PASS")))
    assert [r[0] for r in _recorded(rec)] == ["failed"]


@pytest.mark.asyncio
async def test_blocked_verification_reports_blocked(tmp_path):
    rec = tmp_path / "ack.jsonl"
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    await _run(bus, mgr, _job(rec), _Agent(_resp("FAIL")))
    assert not [c for c, _ in chan.sent if c == GROUP]
    assert [r[0] for r in _recorded(rec)] == ["blocked"]


@pytest.mark.asyncio
async def test_date_template_is_the_run_date(tmp_path):
    rec = tmp_path / "ack.jsonl"
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    await _run(bus, mgr, _job(rec), _Agent(_resp("PASS")))
    date = _recorded(rec)[0][1]
    assert len(date) == 10 and date[4] == "-" and date[7] == "-"


@pytest.mark.asyncio
async def test_no_command_means_no_hook(tmp_path):
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    # Must not raise even though no post_delivery_command is configured.
    await _run(bus, mgr, _job(), _Agent(_resp("PASS")))
    assert [c for c, _ in chan.sent if c == GROUP]


@pytest.mark.asyncio
async def test_missing_command_does_not_break_delivery(tmp_path):
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    job = _job(post_delivery_command=[str(tmp_path / "does-not-exist"), "{ack_status}"])
    await _run(bus, mgr, job, _Agent(_resp("PASS")))
    assert [c for c, _ in chan.sent if c == GROUP]


@pytest.mark.asyncio
async def test_failing_command_does_not_break_delivery(tmp_path):
    rec = tmp_path / "ack.jsonl"
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    job = _job(post_delivery_command=[
        sys.executable, "-c", "import sys; sys.exit(3)", "{ack_status}",
    ])
    await _run(bus, mgr, job, _Agent(_resp("PASS")))
    assert [c for c, _ in chan.sent if c == GROUP]
    assert not rec.exists()


@pytest.mark.asyncio
async def test_run_post_delivery_command_reports_nonzero():
    result = await run_post_delivery_command(
        [sys.executable, "-c", "import sys; sys.exit(4)"], date="2026-09-11",
        ack_status="success", timeout=10,
    )
    assert result["status"] == "failed"
    assert result["ack_status"] == "success"


@pytest.mark.asyncio
async def test_run_post_delivery_command_reports_ok():
    result = await run_post_delivery_command(
        [sys.executable, "-c", "print(1)"], date="2026-09-11",
        ack_status="unknown", timeout=10,
    )
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_run_post_delivery_command_missing_binary():
    result = await run_post_delivery_command(
        ["/nonexistent/binary", "{ack_status}"], date="2026-09-11",
        ack_status="failed", timeout=10,
    )
    assert result["status"] == "unknown"
    assert result["error"]


def test_post_delivery_command_is_persisted_and_loaded(tmp_path):
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path)
    job = service.add_job(
        name="hooked",
        schedule=CronSchedule(kind="cron", expr="0 8 * * *", tz="Asia/Kuala_Lumpur"),
        message="m",
        deliver=True,
        channel="telegram",
        to=GROUP,
        post_delivery_command=["python", "mark.py", "{ack_status}"],
    )
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    assert raw["jobs"][0]["payload"]["post_delivery_command"] == [
        "python", "mark.py", "{ack_status}"
    ]
    loaded = CronService(store_path).get_job(job.id)
    assert loaded is not None
    assert loaded.payload.post_delivery_command == ["python", "mark.py", "{ack_status}"]


def test_legacy_job_without_post_delivery_command_loads(tmp_path):
    store_path = tmp_path / "cron" / "jobs.json"
    service = CronService(store_path)
    job = service.add_job(
        name="legacy",
        schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="Asia/Kuala_Lumpur"),
        message="m",
    )
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    for j in raw["jobs"]:
        j["payload"].pop("post_delivery_command", None)
    store_path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = CronService(store_path).get_job(job.id)
    assert loaded is not None
    assert loaded.payload.post_delivery_command is None

# --------------------------------------------------------------------------
# Owner alerting when bookkeeping fails (the client message is already sent)
# --------------------------------------------------------------------------

_FAILER = "import sys; sys.exit(3)"


def _owner_msgs(chan):
    return [content for chat, content in chan.sent if chat == OWNER]


def _group_msgs(chan):
    return [content for chat, content in chan.sent if chat == GROUP]


@pytest.mark.asyncio
async def test_hook_failure_after_success_alerts_owner_once(tmp_path):
    """ACK success + recording failure: delivered, owner alerted exactly once."""
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    job = _job(post_delivery_command=[sys.executable, "-c", _FAILER, "{ack_status}"])

    await _run(bus, mgr, job, _Agent(_resp("PASS")))

    # The client message is still delivered - bookkeeping never undoes it.
    assert len(_group_msgs(chan)) == 1
    owners = _owner_msgs(chan)
    assert len(owners) == 1
    assert "recording failed" in owners[0]
    assert "Rotation state may require attention" in owners[0]


@pytest.mark.asyncio
async def test_hook_success_does_not_alert_owner(tmp_path):
    rec = tmp_path / "ack.jsonl"
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan

    await _run(bus, mgr, _job(rec), _Agent(_resp("PASS")))

    assert len(_group_msgs(chan)) == 1
    assert _owner_msgs(chan) == []


@pytest.mark.asyncio
async def test_hook_failure_after_failed_ack_does_not_double_alert(tmp_path):
    """A failed delivery already alerted; the recording failure must not add one."""
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus, fail_chats={GROUP})
    mgr.channels["telegram"] = chan
    job = _job(post_delivery_command=[sys.executable, "-c", _FAILER, "{ack_status}"])

    await _run(bus, mgr, job, _Agent(_resp("PASS")))

    assert _group_msgs(chan) == []
    assert len(_owner_msgs(chan)) == 1


@pytest.mark.asyncio
async def test_hook_failure_after_blocked_does_not_double_alert(tmp_path):
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    job = _job(post_delivery_command=[sys.executable, "-c", _FAILER, "{ack_status}"])

    await _run(bus, mgr, job, _Agent(_resp("FAIL")))

    assert _group_msgs(chan) == []
    assert len(_owner_msgs(chan)) == 1


@pytest.mark.asyncio
async def test_hook_failure_with_dead_owner_channel_does_not_recurse(tmp_path):
    """If the owner alert itself cannot be sent, nothing recurses."""
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus, fail_chats={OWNER})
    mgr.channels["telegram"] = chan
    job = _job(post_delivery_command=[sys.executable, "-c", _FAILER, "{ack_status}"])

    await _run(bus, mgr, job, _Agent(_resp("PASS")))

    assert len(_group_msgs(chan)) == 1
    assert _owner_msgs(chan) == []


@pytest.mark.asyncio
async def test_hook_failure_alert_is_sent_to_the_configured_owner(tmp_path):
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    job = _job(
        alert_to="9999",
        post_delivery_command=[sys.executable, "-c", _FAILER, "{ack_status}"],
    )

    await _run(bus, mgr, job, _Agent(_resp("PASS")))

    assert [chat for chat, _ in chan.sent if chat == "9999"]
    assert [content for chat, content in chan.sent if chat == "9999"][0].count("recording failed") == 1
