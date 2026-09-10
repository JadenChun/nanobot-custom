"""Real marketing-verifier bridge integration tests.

These execute the ACTUAL ``tools/verify_*.py`` scripts from the marketing
context repo against deterministic offline fixtures.  They are skipped when the
marketing repo is not present on this host.
"""

from __future__ import annotations

import asyncio
import json
import os
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
from nanobot.cron.delivery import run_external_verifiers
from nanobot.cron.types import CronJob, CronPayload, CronSchedule, CronVerifier

MARKETING = Path(os.environ.get("MARKETING_CONTEXT_DIR", "/opt/marketing-agent/client-marketing-assistance"))
TOOLS = MARKETING / "tools"
GROUP = "-5340461568"
OWNER = "6344587670"

pytestmark = pytest.mark.skipif(
    not (TOOLS / "verify_trend_report.py").exists(),
    reason="marketing context repo not present",
)

PY = sys.executable


def tool(name: str) -> str:
    return str(TOOLS / name)


def verifier(name: str, *args: str, timeout: float = 60.0) -> CronVerifier:
    return CronVerifier(name=name, argv=(PY, *args), timeout=timeout)


# ---------------------------------------------------------------- fixtures
TREND_COLLECTION = {
    "version": 1,
    "gate_status": "complete",
    "sources": [
        {"source_type": "reddit", "attempted": True, "completed": True,
         "command": "web_fetch https://old.reddit.com/r/CatsMY/new/ --json",
         "status": "blocked", "signal_count": 0},
    ],
    "listening_checks": {
        "brand_terms": ["EGOCAT", "Ego Cat", "Egocat", "@egocatmalaysia"],
        "topics": ["multi-cat feeding budgets"],
        "sources": [{"name": "CatsMY public listing",
                     "url": "https://old.reddit.com/r/CatsMY/new/",
                     "status": "limited", "checked_at": "2026-07-17T09:00:00+08:00"}],
    },
}

TREND_REPORT_VALID = """# Ego Cat Daily Trend Research

## Executive takeaway
Two practical content opportunities are worth testing today.

## New / fresh opportunities
### MAKE
1. Multi-cat feeding budgets — **Pillar: Education** — https://old.reddit.com/r/CatsMY/new/

### CONSIDER
None today.

### SKIP
None today.

## Still valid from previous research
Rescue storytelling remains useful background.

## Skipped repeat
No unchanged repeat was promoted today.

## Sources checked
| What we looked at | What was useful | What was limited | Links |
|---|---|---|---|
| Reddit communities | Owner questions | Some pages were unavailable | https://old.reddit.com/r/CatsMY/new/ |

## Brand and owner pulse
No current EGOCAT mention was found in the bounded public checks. Pet owners
were discussing multi-cat feeding budgets on
https://old.reddit.com/r/CatsMY/new/.

## Novelty check
Fresh angles reviewed.
"""

TREND_REPORT_INVALID = "# Ego Cat Daily Trend Research\n\n## Executive takeaway\nNothing.\n"

WEEKLY_TELEGRAM_VALID = """📊 Weekly Performance Review — 2026-08-03 to 2026-08-10

🏆 Best Performer
A practical owner Reel — 1,200 views, 400 reach, 8 saves and 6 shares after two days.
What happened: It drew specific owner replies and useful participation.
Why it may have worked: The clear situation may have made the value easy to understand and share.
What to take from it: A clear situation can make participation easier, but one post is not proof of a rule.

📉 Weakest Performer
Generic teaser — 300 views and no saves after five days.
What happened: It received light reactions but no durable interaction.
Why it may be struggling: The teaser may not have offered an immediate payoff.
What to try differently: Try opening with the useful payoff directly.

🧠 What We Learned This Week
• Specific contribution prompts may encourage owners to provide more useful responses.
• Attention and meaningful participation should be evaluated separately.

Confidence:
Early signals, not proven rules yet.

🎯 What To Try Next
1. Give Community posts a specific useful contribution role rather than a broad reaction prompt.
2. Build immediate value into announcement content instead of relying only on anticipation.
"""

WEEKLY_TELEGRAM_INVALID = "📊 Weekly Performance Review\n\nNo data collected this week.\n"

IDEA_PILLAR_VALID = """# EGOCAT Daily Content Idea — 21 August 2026

## Decision

- **POV / pillar:** Awareness
- **Approved category:** Premium Hair & Skin dry cat food
- **Product role:** supporting
- **Primary objective:** Help Malaysian cat owners notice EGOCAT's local-cat positioning and remember the brand's responsible-care point of view.
- **Content mechanism:** Identity recognition through a specific local-cat care moment.
- **Payoff:** Owners can save the reminder and remember a more generous way to see local cats.
- **Product removal test:** Pass — the positioning reminder still has meaning without the pack.
- **Repeat-control:** Fresh execution; it uses a care reminder rather than a role-personification joke.

## Idea

**Title:** *Local Cats Deserve the Main Frame*

Show EGOCAT as the brand that celebrates local mixed-breed cats and makes responsible everyday care easier to recognise.

**Hook:** "Local cat pun deserve care yang orang nampak dan ingat."

**Shoot:**
1. Show a local mixed-breed cat and identify EGOCAT's local-cat positioning.
2. Show the approved dry-food category beside a simple, factual care reminder.
3. Close with the message: "Local cats deserve thoughtful care."

**CTA:** Save this reminder for the next mealtime setup.
**Why:** Viewers should notice EGOCAT's local-cat positioning and remember it as a responsible advocate, not just a funny cat account.
"""

IDEA_PILLAR_INVALID = """# EGOCAT Daily Content Idea — 21 August 2026

## Decision

- **POV / pillar:** Awareness
- **Approved category:** Premium Hair & Skin dry cat food; pack shown only as a factual mealtime prop
- **Objective:** Build local-cat pride and test shares plus a low-friction title-choice comment

## Idea

**Title:** *Local Cat, Full Ego — Main-Character Mealtime*

Give a local mixed-breed cat a playful "main character" introduction at mealtime.

**CTA:** Comment A = CEO rumah, B = Director of drama.
"""

IDEA_TELEGRAM_VALID = """💡 Today's Content Idea — Education
Product: Premium Hair & Skin dry cat food
Idea: Make a short checklist showing how owners can make feeding routines easier to follow.
Format: Post — Carousel
Hook: "Kalau waktu makan selalu jadi kecoh, cuba tiga langkah ini."
Structure:
1. Cover: the messy routine vs the simple reset.
2. Main: portion and prep setup.
3. Close: save the routine checklist.
CTA: Save this routine and share it with another cat owner.
Why: It turns a familiar owner problem into a useful, product-relevant teaching moment.
Inspiration: [Trending owner routine](https://www.tiktok.com/@example/video/123)
"""

IDEA_TELEGRAM_INVALID = "💡 Today's Content Idea\n\nNothing structured yet.\n"


def write(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


# ---------------------------------------------------- real verifier proofs
@pytest.mark.asyncio
async def test_trend_real_pass(tmp_path):
    r = write(tmp_path, "report.md", TREND_REPORT_VALID)
    c = write(tmp_path, "collection.json", json.dumps(TREND_COLLECTION))
    agg, details = await run_external_verifiers(
        [verifier("trend", tool("verify_trend_report.py"), "--report", r, "--collection", c, "--json")],
        date="2026-09-10",
    )
    assert agg == "PASS", details


@pytest.mark.asyncio
async def test_trend_real_fail(tmp_path):
    r = write(tmp_path, "report.md", TREND_REPORT_INVALID)
    c = write(tmp_path, "collection.json", json.dumps(TREND_COLLECTION))
    agg, _ = await run_external_verifiers(
        [verifier("trend", tool("verify_trend_report.py"), "--report", r, "--collection", c, "--json")],
        date="2026-09-10",
    )
    assert agg == "FAIL"


@pytest.mark.asyncio
async def test_weekly_real_pass(tmp_path):
    r = write(tmp_path, "weekly.md", WEEKLY_TELEGRAM_VALID)
    agg, details = await run_external_verifiers(
        [verifier("weekly", tool("verify_weekly_review.py"), "--report", r, "--telegram", "--json")],
        date="2026-09-10",
    )
    assert agg == "PASS", details


@pytest.mark.asyncio
async def test_weekly_real_fail(tmp_path):
    r = write(tmp_path, "weekly.md", WEEKLY_TELEGRAM_INVALID)
    agg, _ = await run_external_verifiers(
        [verifier("weekly", tool("verify_weekly_review.py"), "--report", r, "--telegram", "--json")],
        date="2026-09-10",
    )
    assert agg == "FAIL"


@pytest.mark.asyncio
async def test_idea_real_pass(tmp_path):
    idea = write(tmp_path, "idea.md", IDEA_PILLAR_VALID)
    tel = write(tmp_path, "idea-tg.md", IDEA_TELEGRAM_VALID)
    agg, details = await run_external_verifiers(
        [
            verifier("idea_pillar", tool("verify_idea_pillar.py"), "--input", idea,
                     "--expected-pillar", "Awareness", "--json"),
            verifier("idea_telegram", tool("verify_telegram_output.py"), "--kind", "idea",
                     "--input", tel, "--json"),
        ],
        date="2026-09-10",
    )
    assert agg == "PASS", details


@pytest.mark.asyncio
async def test_idea_bad_pillar_fails(tmp_path):
    idea = write(tmp_path, "idea.md", IDEA_PILLAR_INVALID)
    tel = write(tmp_path, "idea-tg.md", IDEA_TELEGRAM_VALID)
    agg, _ = await run_external_verifiers(
        [
            verifier("idea_pillar", tool("verify_idea_pillar.py"), "--input", idea,
                     "--expected-pillar", "Awareness", "--json"),
            verifier("idea_telegram", tool("verify_telegram_output.py"), "--kind", "idea",
                     "--input", tel, "--json"),
        ],
        date="2026-09-10",
    )
    assert agg == "FAIL"


@pytest.mark.asyncio
async def test_idea_bad_telegram_fails(tmp_path):
    idea = write(tmp_path, "idea.md", IDEA_PILLAR_VALID)
    tel = write(tmp_path, "idea-tg.md", IDEA_TELEGRAM_INVALID)
    agg, _ = await run_external_verifiers(
        [
            verifier("idea_pillar", tool("verify_idea_pillar.py"), "--input", idea,
                     "--expected-pillar", "Awareness", "--json"),
            verifier("idea_telegram", tool("verify_telegram_output.py"), "--kind", "idea",
                     "--input", tel, "--json"),
        ],
        date="2026-09-10",
    )
    assert agg == "FAIL"


# ---------------------------------------- end-to-end through _run_cron_job
class RecordingChannel(BaseChannel):
    name = "telegram"
    display_name = "Telegram"

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.sent = []

    async def start(self): pass
    async def stop(self): pass

    async def send(self, msg):
        self.sent.append((msg.chat_id, msg.content))


class _Sessions:
    def get_or_create(self, key): return SimpleNamespace(retain_recent_legal_suffix=lambda n: None)
    def save(self, s): pass


class _Agent:
    def __init__(self, resp):
        self._resp = resp
        self.tools = SimpleNamespace(get=lambda name: None)
        self.sessions = _Sessions()
        self.failures = []

    async def process_direct(self, *a, **k):
        self.skip_verification_arg = k.get("skip_verification")
        return self._resp

    def record_task_failure(self, **k):
        self.failures.append(k)


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


async def _run(bus, mgr, job, agent):
    task = asyncio.create_task(mgr._dispatch_outbound())
    await asyncio.sleep(0)
    try:
        await _run_cron_job(agent, bus, job)
        await asyncio.sleep(0.05)
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_e2e_trend_valid_group_eligible_owner_silent(tmp_path):
    r = write(tmp_path, "report.md", TREND_REPORT_VALID)
    c = write(tmp_path, "collection.json", json.dumps(TREND_COLLECTION))
    v = [verifier("trend", tool("verify_trend_report.py"), "--report", r, "--collection", c, "--json")]
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    await _run(bus, mgr, _job(v), _Agent(_resp("PASS")))
    assert len([c for c, _ in chan.sent if c == GROUP]) == 1
    assert len([c for c, _ in chan.sent if c == OWNER]) == 0


@pytest.mark.asyncio
async def test_e2e_trend_invalid_group_zero_owner_once(tmp_path):
    r = write(tmp_path, "report.md", TREND_REPORT_INVALID)
    c = write(tmp_path, "collection.json", json.dumps(TREND_COLLECTION))
    v = [verifier("trend", tool("verify_trend_report.py"), "--report", r, "--collection", c, "--json")]
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    await _run(bus, mgr, _job(v), _Agent(_resp("PASS")))
    assert len([c for c, _ in chan.sent if c == GROUP]) == 0
    assert len([c for c, _ in chan.sent if c == OWNER]) == 1


@pytest.mark.asyncio
async def test_e2e_verifier_narration_not_in_client_content(tmp_path):
    r = write(tmp_path, "report.md", TREND_REPORT_VALID)
    c = write(tmp_path, "collection.json", json.dumps(TREND_COLLECTION))
    v = [verifier("trend", tool("verify_trend_report.py"), "--report", r, "--collection", c, "--json")]
    bus = MessageBus(outbound_ack_timeout=5.0)
    mgr = ChannelManager(Config(), bus)
    chan = RecordingChannel({}, bus)
    mgr.channels["telegram"] = chan
    await _run(bus, mgr, _job(v), _Agent(_resp("PASS")))
    group_texts = [t for cc, t in chan.sent if cc == GROUP]
    assert group_texts
    for text in group_texts:
        assert "Verification status" not in text
        assert "gate narration" not in text
        assert "trend report verified" not in text.lower()