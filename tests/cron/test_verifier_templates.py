"""Run-scoped verifier template tests: {date} and {response_file}."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

import nanobot.channels.manager as manager_mod
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.cli.commands import _run_cron_job
from nanobot.config.schema import Config
from nanobot.cron.delivery import _render_verifier_argv, run_external_verifiers
from nanobot.cron.types import CronJob, CronPayload, CronSchedule, CronVerifier

GROUP = "-5340461568"
OWNER = "6344587670"


def test_render_substitutes_date_and_repo_root():
    out = _render_verifier_argv(
        ["{repo_root}/agent-workspace/outputs/research/{date}-daily-trend-research.md"],
        date="2026-09-10", repo_root="/repo",
    )
    assert out == ["/repo/agent-workspace/outputs/research/2026-09-10-daily-trend-research.md"]


@pytest.mark.asyncio
async def test_response_file_template_receives_exact_content(tmp_path):
    code = (
        "import sys,json;"
        "t=open(sys.argv[1],encoding='utf-8').read();"
        "print(json.dumps({'verified': t.strip()=='CLIENT COPY'}))"
    )
    v = CronVerifier(name="content", argv=(sys.executable, "-c", code, "{response_file}"), timeout=10)
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = _Chan({}, bus)
    mgr.channels["telegram"] = chan
    task = asyncio.create_task(mgr._dispatch_outbound())
    await asyncio.sleep(0)
    try:
        await _run_cron_job(_Agent(_resp("PASS")), bus, _job([v]))
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
    assert len([c for c, _ in chan.sent if c == GROUP]) == 1


@pytest.mark.asyncio
async def test_missing_response_template_arg_blocks(tmp_path):
    # A verifier requiring {response_file} is validated against the real run text.
    code = (
        "import sys,json;"
        "t=open(sys.argv[1],encoding='utf-8').read();"
        "print(json.dumps({'verified': 'DIFFERENT' in t}))"
    )
    v = CronVerifier(name="content", argv=(sys.executable, "-c", code, "{response_file}"), timeout=10)
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = _Chan({}, bus)
    mgr.channels["telegram"] = chan
    task = asyncio.create_task(mgr._dispatch_outbound())
    await asyncio.sleep(0)
    try:
        await _run_cron_job(_Agent(_resp("PASS")), bus, _job([v]))
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
    assert len([c for c, _ in chan.sent if c == GROUP]) == 0
    assert len([c for c, _ in chan.sent if c == OWNER]) == 1


class _Chan(BaseChannel):
    name = "telegram"
    display_name = "Telegram"

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.sent = []

    async def start(self): pass
    async def stop(self): pass

    async def send(self, msg): self.sent.append((msg.chat_id, msg.content))


class _Sessions:
    def get_or_create(self, key): return SimpleNamespace(retain_recent_legal_suffix=lambda n: None)
    def save(self, s): pass


class _Agent:
    def __init__(self, resp):
        self._resp = resp
        self.tools = SimpleNamespace(get=lambda name: None)
        self.sessions = _Sessions()
        self.failures = []

    async def process_direct(self, *a, **k): return self._resp
    def record_task_failure(self, **k): self.failures.append(k)


def _resp(verdict="PASS"):
    from nanobot.bus.events import OutboundMessage
    return OutboundMessage(channel="telegram", chat_id=GROUP, content="CLIENT COPY",
                           metadata={"_verification": verdict})


def _job(verifiers):
    return CronJob(id="j1", name="Daily Trend Pulse",
                   schedule=CronSchedule(kind="cron", expr="0 7 * * *", tz="Asia/Kuala_Lumpur"),
                   payload=CronPayload(message="m", deliver=True, channel="telegram", to=GROUP,
                                       skip_verification=False, alert_channel="telegram",
                                       alert_to=OWNER, verifiers=verifiers))


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(manager_mod, "_SEND_RETRY_DELAYS", (0, 0, 0))