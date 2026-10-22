"""Configured external checks gate exact client text; hooks never resend it."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.turn import TurnResult
from nanobot.cron.channel_sender import CronChannelSender
from nanobot.cron.composition import CronCompositionPreflightError, compose_claimed_text_occurrence
from nanobot.cron.job_hooks import CronJobHooks, run_adapter
from nanobot.cron.types import CronVerifier
from tests.cron.test_cron_composition import ReceiptChannel, _claimed


def _setup(tmp_path, *, verified=True, post_fails=False, alert=False):
    code = ("import json,sys; from pathlib import Path; "
            f"print(json.dumps({{'verified': {verified!r} and Path(sys.argv[1]).read_text() == 'Exact client text' "
            "and sys.argv[2] == '2023-11-15' and sys.argv[3] == 'READY'}))")
    (tmp_path / "status-2023-11-15.md").write_text("**Internal status:** READY")
    post = ("import sys; from pathlib import Path; "
            "Path('ack.txt').write_text('|'.join(sys.argv[1:])); "
            f"sys.exit({1 if post_fails else 0})")
    service, claim = _claimed(tmp_path, channels=("telegram",), timezone="Asia/Kuala_Lumpur", payload_overrides={
        "verifiers": [CronVerifier("evidence", (sys.executable, "-c", code, "{response_file}", "{date}", "{status}"),
                                   cwd=str(tmp_path), status_file="status-{date}.md")],
        "skip_verification": True,  # Never bypass an authoritative external check.
        "alert_to": "owner" if alert else None,
        "post_delivery_command": [sys.executable, "-c", post, "{ack_status}", "{date}", "{repo_root}"],
    })
    agent = SimpleNamespace(execute_silent_cron_turn=AsyncMock(return_value=TurnResult(
        run_id=claim.run_id, content="Exact client text")))
    calls = []
    sender = CronChannelSender(lambda _name: ReceiptChannel(calls))
    hooks = CronJobHooks(tmp_path / "hooks", default_cwd=tmp_path)
    return service, claim, agent, sender, calls, hooks


@pytest.mark.asyncio
async def test_verifier_checks_this_exact_response_and_bookkeeping_gets_confirmed_outcome(tmp_path):
    service, claim, agent, sender, calls, hooks = _setup(tmp_path, alert=True)
    result = await compose_claimed_text_occurrence(service, agent, claim, sender, job_hooks=hooks)
    assert calls == [("A", ("Exact client text",))]
    assert result.completion.status == "ok"
    assert result.hooks.ack_status == "success"
    assert result.hooks.bookkeeping.state == "succeeded" and result.hooks.owner_alert is None
    assert (tmp_path / "ack.txt").read_text() == f"success|2023-11-15|{tmp_path}"
    assert not list((tmp_path / "hooks").glob("response-*"))


@pytest.mark.asyncio
async def test_failed_verification_blocks_client_and_alerts_only_owner(tmp_path):
    service, claim, agent, sender, calls, hooks = _setup(tmp_path, verified=False, alert=True)
    result = await compose_claimed_text_occurrence(service, agent, claim, sender, job_hooks=hooks)
    assert result.completion.status == "error" and result.outbox is None
    assert [target for target, _ in calls] == ["owner"]
    assert result.hooks.ack_status == "blocked" and result.hooks.owner_alert.state == "succeeded"
    assert (tmp_path / "ack.txt").read_text().startswith("blocked|")
    assert "Exact client text" not in str(calls)


@pytest.mark.asyncio
async def test_bookkeeping_failure_after_success_does_not_resend_client_result(tmp_path):
    service, claim, agent, sender, calls, hooks = _setup(tmp_path, post_fails=True, alert=True)
    result = await compose_claimed_text_occurrence(service, agent, claim, sender, job_hooks=hooks)
    assert result.completion.status == "ok" and result.hooks.bookkeeping.state == "failed"
    assert [target for target, _ in calls] == ["A", "owner"]
    again = await hooks.after_delivery(claim, result.completion, result.deliveries, sender)
    assert again.bookkeeping.state == "failed" and again.owner_alert.state == "succeeded"
    assert [target for target, _ in calls] == ["A", "owner"]
    assert agent.execute_silent_cron_turn.await_count == 1


@pytest.mark.asyncio
async def test_unknown_client_and_failed_owner_delivery_do_not_recurse_or_replay(tmp_path):
    service, claim, agent, _sender, calls, hooks = _setup(tmp_path, alert=True)

    class FailingChannel:
        async def send_text_with_receipt(self, target, *_args, **_kwargs):
            calls.append(target)
            raise ConnectionError("ambiguous send")

    sender = CronChannelSender(lambda _name: FailingChannel())
    result = await compose_claimed_text_occurrence(service, agent, claim, sender, job_hooks=hooks)
    assert result.hooks.ack_status == "unknown"
    assert result.hooks.owner_alert.state == "failed" and calls == ["A", "owner"]
    assert (tmp_path / "ack.txt").read_text().startswith("unknown|")


@pytest.mark.asyncio
async def test_missing_adapter_fails_before_consuming_execution(tmp_path):
    service, claim, agent, sender, calls, _hooks = _setup(tmp_path)
    with pytest.raises(CronCompositionPreflightError, match="hook adapter"):
        await compose_claimed_text_occurrence(service, agent, claim, sender)
    agent.execute_silent_cron_turn.assert_not_awaited()
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("script,timeout,reason", [
    ("import time; time.sleep(30)", .03, "timeout"),
    ("print('x' * 70000)", 2, "output_limit"),
    ("import sys; sys.exit(7)", 2, "nonzero_exit"),
])
async def test_adapter_execution_is_bounded_and_reports_only_failure_codes(tmp_path, script, timeout, reason):
    outcome = await run_adapter([sys.executable, "-c", script], cwd=tmp_path, timeout=timeout)
    assert outcome.reason == reason and outcome.returncode != 0


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [{"ok": "true"}, {"verified": 1}, {}, {"ok": False}])
async def test_nontruthful_json_verdict_never_passes(tmp_path, verdict):
    verifier = CronVerifier("bad", (sys.executable, "-c", f"print({json.dumps(verdict)!r})"))
    _service, claim = _claimed(tmp_path, payload_overrides={"verifiers": [verifier]})
    passed, _reason = await CronJobHooks(tmp_path / "hooks", default_cwd=tmp_path).verify(
        claim, TurnResult(run_id=claim.run_id, content="client text"))
    assert not passed
