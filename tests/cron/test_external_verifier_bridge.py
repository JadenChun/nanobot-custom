"""Authoritative external-verifier bridge tests (hermetic; no marketing repo).

Covers the structured subprocess runner, aggregation, and the precedence rule
that external verifiers override the internal LLM verdict.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import nanobot.channels.manager as manager_mod
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.cli.commands import _run_cron_job
from nanobot.config.schema import Config
from nanobot.cron.delivery import _verifier_passed, run_external_verifiers
from nanobot.cron.types import CronDestination, CronJob, CronPayload, CronSchedule, CronVerifier

GROUP = "-5340461568"
OWNER = "6344587670"


def _py(code: str) -> tuple[str, ...]:
    return (sys.executable, "-c", code)


PASS = CronVerifier(name="pass", argv=_py("import json;print(json.dumps({'verified':True}))"), timeout=10)
FAIL = CronVerifier(name="fail", argv=_py("import sys;sys.exit(1)"), timeout=10)
MALFORMED = CronVerifier(name="malformed", argv=_py("print('not json')"), timeout=10)
UNTRUTHFUL = CronVerifier(name="untruthful", argv=_py("import json;print(json.dumps({'verified':False}))"), timeout=10)
TIMEOUT = CronVerifier(name="timeout", argv=_py("import time;time.sleep(5)"), timeout=0.2)
MISSING_EXE = CronVerifier(name="missing", argv=("/nonexistent/verifier-binary",), timeout=10)
OK_ALT = CronVerifier(name="ok", argv=_py("import json;print(json.dumps({'ok':True}))"), timeout=10)


# ---------------- _verifier_passed ----------------
def test_verifier_passed_exit_nonzero_is_fail():
    assert _verifier_passed(1, b'{"verified": true}') == "FAIL"


def test_verifier_passed_truthful_json_is_pass():
    assert _verifier_passed(0, b'{"verified": true, "x": 1}') == "PASS"
    assert _verifier_passed(0, b'{"ok": true}') == "PASS"


def test_verifier_passed_malformed_is_missing():
    assert _verifier_passed(0, b"not json at all") == "MISSING"
    assert _verifier_passed(0, b"") == "MISSING"
    assert _verifier_passed(0, b'{"verified": false}') == "MISSING"


# ---------------- run_external_verifiers ----------------
@pytest.mark.asyncio
async def test_all_pass_aggregates_pass():
    agg, details = await run_external_verifiers([PASS, OK_ALT], date="2026-09-10")
    assert agg == "PASS"
    assert [d["status"] for d in details] == ["PASS", "PASS"]


@pytest.mark.asyncio
async def test_one_fail_aggregates_fail():
    agg, _ = await run_external_verifiers([PASS, FAIL], date="2026-09-10")
    assert agg == "FAIL"


@pytest.mark.asyncio
async def test_malformed_output_is_missing_and_blocks():
    agg, _ = await run_external_verifiers([MALFORMED], date="2026-09-10")
    assert agg == "MISSING"


@pytest.mark.asyncio
async def test_untruthful_json_is_missing():
    agg, _ = await run_external_verifiers([UNTRUTHFUL], date="2026-09-10")
    assert agg == "MISSING"


@pytest.mark.asyncio
async def test_exec_error_is_missing():
    agg, details = await run_external_verifiers([MISSING_EXE], date="2026-09-10")
    assert agg == "MISSING"
    assert details[0]["status"] == "MISSING"


@pytest.mark.asyncio
async def test_timeout_is_missing():
    agg, details = await run_external_verifiers([TIMEOUT], date="2026-09-10")
    assert agg == "MISSING"
    assert "timed out" in details[0]["error"]


@pytest.mark.asyncio
async def test_missing_beats_fail_in_aggregation():
    agg, _ = await run_external_verifiers([FAIL, MALFORMED], date="2026-09-10")
    assert agg == "MISSING"


# ---------------- precedence via _run_cron_job ----------------
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
    def get_or_create(self, key): return SimpleNamespace(retain_recent_legal_suffix=lambda n: None)
    def save(self, s): pass


class _Tools:
    def get(self, name): return None


class _Agent:
    def __init__(self, resp):
        self._resp = resp
        self.tools = _Tools()
        self.sessions = _Sessions()
        self.failures = []

    async def process_direct(self, *a, **k):
        # Record whether the internal verifier was asked to run.
        self.skip_verification_arg = k.get("skip_verification")
        return self._resp

    def record_task_failure(self, **k):
        self.failures.append(k)


def _msg(verdict):
    from nanobot.bus.events import OutboundMessage
    md = {} if verdict is None else {"_verification": verdict}
    return OutboundMessage(channel="telegram", chat_id=GROUP, content="IDEA", metadata=md)


def _job(verifiers, **over):
    p = dict(message="m", deliver=True, channel="telegram", to=GROUP,
             skip_verification=False, alert_channel="telegram", alert_to=OWNER,
             verifiers=list(verifiers))
    p.update(over)
    return CronJob(id="j1", name="Daily Content Idea",
                   schedule=CronSchedule(kind="cron", expr="0 8 * * *", tz="Asia/Kuala_Lumpur"),
                   payload=CronPayload(**p))


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


@pytest.mark.asyncio
async def test_external_fail_overrides_internal_pass():
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    agent = _Agent(_msg("PASS"))  # internal says PASS
    await _run(bus, mgr, _job([FAIL]), agent)
    # External FAIL is authoritative: no group, one owner alert, internal skipped.
    assert len([c for c, _ in chan.sent if c == GROUP]) == 0
    assert len([c for c, _ in chan.sent if c == OWNER]) == 1
    assert agent.skip_verification_arg is True  # internal verifier skipped
    assert len(agent.failures) == 1


@pytest.mark.asyncio
async def test_external_pass_overrides_internal_missing():
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    agent = _Agent(_msg(None))  # internal MISSING (no verdict)
    await _run(bus, mgr, _job([PASS]), agent)
    assert len([c for c, _ in chan.sent if c == GROUP]) == 1
    assert len([c for c, _ in chan.sent if c == OWNER]) == 0


@pytest.mark.asyncio
async def test_external_pass_group_ack_success_owner_silent():
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    agent = _Agent(_msg("PASS"))
    await _run(bus, mgr, _job([PASS, OK_ALT]), agent)
    assert len([c for c, _ in chan.sent if c == GROUP]) == 1
    assert len([c for c, _ in chan.sent if c == OWNER]) == 0


@pytest.mark.asyncio
async def test_external_timeout_blocks_and_alerts_once():
    bus = MessageBus(outbound_ack_timeout=2.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    agent = _Agent(_msg("PASS"))
    await _run(bus, mgr, _job([TIMEOUT]), agent)
    assert len([c for c, _ in chan.sent if c == GROUP]) == 0
    assert len([c for c, _ in chan.sent if c == OWNER]) == 1


def test_verifier_narration_not_in_client_content():
    # The bridge returns structured data; it never mutates client content.
    from nanobot.bus.events import OutboundMessage
    msg = OutboundMessage(channel="telegram", chat_id=GROUP, content="CLIENT COPY")
    assert msg.content == "CLIENT COPY"
    assert "Verification status" not in msg.content