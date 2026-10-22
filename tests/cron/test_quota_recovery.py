"""Whole-job quota recovery retains a durable intent and never replays effects."""

from unittest.mock import AsyncMock

import pytest

from nanobot.agent.turn import RunStatus, TurnResult
from nanobot.cron.quota_recovery import CronQuotaRecovery, codex_retry_at
from tests.cron.test_gateway_cron import _gateway


def _usage(reset=2, *, weekly=0, reached=None):
    return {"rate_limit": {"primary_window": {"reset_at": reset},
                           "secondary_window": {"used_percent": weekly}},
            "rate_limit_reached_type": reached}


@pytest.mark.parametrize("usage,retries,expected", [
    (_usage(), 0, 122000), (_usage(), 2, 122000), (_usage(), 3, None),
    (_usage(1), 0, 121000), (_usage(100000), 0, None),
    (_usage(weekly=100), 0, None), (_usage(reached="secondary"), 0, None),
    ({}, 0, None), (None, 0, None), (_usage(reset=True), 0, None),
    (_usage(weekly=float("nan")), 0, None),
])
def test_retry_window_contract(usage, retries, expected):
    assert codex_retry_at(usage, now_ms=3000, retries=retries) == expected


def test_expired_buffer_uses_two_minutes_from_now():
    assert codex_retry_at(_usage(1), now_ms=125000, retries=0) == 245000


async def test_retry_survives_restart_and_success_resets_budget(tmp_path):
    gateway, clock, calls, channels, item = _gateway(tmp_path)
    quota = gateway.runtime.quota_recovery
    quota.is_codex = lambda: True
    quota.usage = AsyncMock(return_value=_usage())

    async def limited(request):
        calls.append(request)
        return TurnResult(run_id=request.run_id, status=RunStatus.ERROR, error="Codex rate limit")

    gateway.runtime.agent.execute_silent_cron_turn = limited
    gateway.runtime.acquire()
    clock.now = 1100
    result = await gateway.runtime.scheduler.wake_once()
    assert result.completion.status == "error" and result.outbox is None
    assert not channels.calls and len(calls) == 1
    state = quota.state(item.job_instance_id)
    assert state.retry_count == 1 and state.pending.retry_at_ms == 122000
    # A periodic fire while waiting does not run the model or spend the budget.
    clock.now = 1200
    await gateway.runtime.scheduler.wake_once()
    assert len(calls) == 1 and quota.state(item.job_instance_id).retry_count == 1
    # Simulate the completion-to-retry-store crash seam: durable completion
    # alone is sufficient to reconstruct the intent, idempotently.
    with quota._db() as connection:
        connection.execute("DELETE FROM jobs")
    quota.recover(gateway.service)
    quota.recover(gateway.service)
    assert quota.state(item.job_instance_id).retry_count == 1

    async def success(request):
        calls.append(request)
        return TurnResult(run_id=request.run_id, content="completed after reset")

    gateway.runtime.agent.execute_silent_cron_turn = success
    clock.now = 122000
    await gateway.runtime.run_due_quota_retries()
    assert len(calls) == 2 and channels.calls == ["A", "B"]
    assert quota.state(item.job_instance_id).retry_count == 0
    assert quota.pending() == []
    quota.recover(gateway.service)
    assert quota.state(item.job_instance_id).retry_count == 0
    await gateway.aclose()


@pytest.mark.parametrize("reason", ["weekly", "effects", "provider", "unreadable"])
async def test_unsafe_or_unsupported_failures_never_schedule_whole_job_retry(tmp_path, reason):
    gateway, clock, calls, _channels, _item = _gateway(tmp_path)
    quota = gateway.runtime.quota_recovery
    quota.is_codex = lambda: reason != "provider"
    quota.usage = AsyncMock(return_value=_usage(weekly=100 if reason == "weekly" else 0),
                            side_effect=OSError("quota unavailable") if reason == "unreadable" else None)

    async def limited(request):
        calls.append(request)
        return TurnResult(run_id=request.run_id, status=RunStatus.ERROR,
                          error="Codex rate limit", tools_used=["write_file"] if reason == "effects" else [])

    gateway.runtime.agent.execute_silent_cron_turn = limited
    gateway.runtime.acquire()
    clock.now = 1100
    await gateway.runtime.scheduler.wake_once()
    assert quota.pending() == [] and len(calls) == 1
    if reason in {"effects", "provider"}:
        quota.usage.assert_not_awaited()
    await gateway.aclose()


async def test_restart_never_blindly_repeats_an_uncertain_retry_dispatch(tmp_path):
    gateway, clock, _calls, _channels, item = _gateway(tmp_path)
    quota = gateway.runtime.quota_recovery
    quota.is_codex = lambda: True
    quota.usage = AsyncMock(return_value=_usage())

    async def limited(request):
        return TurnResult(run_id=request.run_id, status=RunStatus.ERROR, error="rate_limit")

    gateway.runtime.agent.execute_silent_cron_turn = limited
    gateway.runtime.acquire()
    clock.now = 1100
    await gateway.runtime.scheduler.wake_once()
    quota.mark(quota.pending()[0], "dispatching")
    restarted = CronQuotaRecovery(quota.root, clock, is_codex=lambda: True)
    restarted.recover(gateway.service)
    assert restarted.state(item.job_instance_id).pending.state == "unknown"
    gateway.runtime.quota_recovery = restarted
    clock.now = 122000
    await gateway.runtime.run_due_quota_retries()
    assert restarted.state(item.job_instance_id).pending.state == "unknown"
    await gateway.aclose()


async def test_three_reschedules_exhaust_budget_without_multiplying_provider_calls(tmp_path):
    gateway, clock, calls, _channels, item = _gateway(tmp_path)
    quota = gateway.runtime.quota_recovery
    quota.is_codex = lambda: True
    quota.usage = AsyncMock(return_value=_usage())

    async def limited(request):
        calls.append(request)
        return TurnResult(run_id=request.run_id, status=RunStatus.ERROR, error="Codex rate limit")

    gateway.runtime.agent.execute_silent_cron_turn = limited
    gateway.runtime.acquire()
    clock.now = 1100
    await gateway.runtime.scheduler.wake_once()
    for count in (1, 2, 3):
        assert quota.state(item.job_instance_id).retry_count == count
        clock.now = quota.pending()[0].retry_at_ms
        await gateway.runtime.run_due_quota_retries()
    assert len(calls) == 4
    assert quota.usage.await_count == 3
    assert quota.state(item.job_instance_id).retry_count == 3
    assert quota.pending() == []
    await gateway.aclose()
