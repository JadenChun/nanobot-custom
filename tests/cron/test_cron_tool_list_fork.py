"""Tests for CronTool._list_jobs() output formatting."""

from datetime import datetime, timezone

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.cron import CronTool
from nanobot.cron.service import CronService
from nanobot.cron.types import CronDestination, CronJobState, CronSchedule
from tests.cron import p6_compat_fixtures as cases
from tests.cron.stage3_fixtures import offline_denial  # noqa: F401


def _make_tool(tmp_path) -> CronTool:
    service = CronService(tmp_path / "cron" / "jobs.json")
    return CronTool(service)


def _make_tool_with_tz(tmp_path, tz: str) -> CronTool:
    service = CronService(tmp_path / "cron" / "jobs.json")
    return CronTool(service, default_timezone=tz)


# -- _format_timing tests --


def test_format_timing_cron_with_tz(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    s = CronSchedule(kind="cron", expr="0 9 * * 1-5", tz="America/Denver")
    assert tool._format_timing(s) == "cron: 0 9 * * 1-5 (America/Denver)"


def test_format_timing_cron_without_tz(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    s = CronSchedule(kind="cron", expr="*/5 * * * *")
    assert tool._format_timing(s) == "cron: */5 * * * *"


def test_format_timing_every_hours(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    s = CronSchedule(kind="every", every_ms=7_200_000)
    assert tool._format_timing(s) == "every 2h"


def test_format_timing_every_minutes(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    s = CronSchedule(kind="every", every_ms=1_800_000)
    assert tool._format_timing(s) == "every 30m"


def test_format_timing_every_seconds(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    s = CronSchedule(kind="every", every_ms=30_000)
    assert tool._format_timing(s) == "every 30s"


def test_format_timing_every_non_minute_seconds(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    s = CronSchedule(kind="every", every_ms=90_000)
    assert tool._format_timing(s) == "every 90s"


def test_format_timing_every_milliseconds(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    s = CronSchedule(kind="every", every_ms=200)
    assert tool._format_timing(s) == "every 200ms"


def test_format_timing_at(tmp_path) -> None:
    tool = _make_tool_with_tz(tmp_path, "Asia/Shanghai")
    s = CronSchedule(kind="at", at_ms=1773684000000)
    result = tool._format_timing(s)
    assert "Asia/Shanghai" in result
    assert result.startswith("at 2026-")


def test_format_timing_fallback(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    s = CronSchedule(kind="every")  # no every_ms
    assert tool._format_timing(s) == "every"


# -- _format_state tests --


def test_format_state_empty(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    state = CronJobState()
    assert tool._format_state(state, CronSchedule(kind="every")) == []


def test_format_state_last_run_ok(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    state = CronJobState(last_run_at_ms=1773673200000, last_status="ok")
    lines = tool._format_state(state, CronSchedule(kind="cron", expr="0 9 * * *", tz="UTC"))
    assert len(lines) == 1
    assert "Last run:" in lines[0]
    assert "ok" in lines[0]


def test_format_state_last_run_with_error(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    state = CronJobState(last_run_at_ms=1773673200000, last_status="error", last_error="timeout")
    lines = tool._format_state(state, CronSchedule(kind="cron", expr="0 9 * * *", tz="UTC"))
    assert len(lines) == 1
    assert "error" in lines[0]
    assert "timeout" in lines[0]


def test_format_state_next_run_only(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    state = CronJobState(next_run_at_ms=1773684000000)
    lines = tool._format_state(state, CronSchedule(kind="cron", expr="0 9 * * *", tz="UTC"))
    assert len(lines) == 1
    assert "Next run:" in lines[0]


def test_format_state_both(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    state = CronJobState(
        last_run_at_ms=1773673200000, last_status="ok", next_run_at_ms=1773684000000
    )
    lines = tool._format_state(state, CronSchedule(kind="cron", expr="0 9 * * *", tz="UTC"))
    assert len(lines) == 2
    assert "Last run:" in lines[0]
    assert "Next run:" in lines[1]


def test_format_state_unknown_status(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    state = CronJobState(last_run_at_ms=1773673200000, last_status=None)
    lines = tool._format_state(state, CronSchedule(kind="cron", expr="0 9 * * *", tz="UTC"))
    assert "unknown" in lines[0]


# -- _list_jobs integration tests --


def test_list_empty(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    assert tool._list_jobs() == "No scheduled jobs."


def test_list_cron_job_shows_expression_and_timezone(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    tool._cron.add_job(
        name="Morning scan",
        schedule=CronSchedule(kind="cron", expr="0 9 * * 1-5", tz="America/Denver"),
        message="scan",
    )
    result = tool._list_jobs()
    assert "cron: 0 9 * * 1-5 (America/Denver)" in result


def test_list_every_job_shows_human_interval(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    tool._cron.add_job(
        name="Frequent check",
        schedule=CronSchedule(kind="every", every_ms=1_800_000),
        message="check",
    )
    result = tool._list_jobs()
    assert "every 30m" in result


def test_list_every_job_hours(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    tool._cron.add_job(
        name="Hourly check",
        schedule=CronSchedule(kind="every", every_ms=7_200_000),
        message="check",
    )
    result = tool._list_jobs()
    assert "every 2h" in result


def test_list_every_job_seconds(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    tool._cron.add_job(
        name="Fast check",
        schedule=CronSchedule(kind="every", every_ms=30_000),
        message="check",
    )
    result = tool._list_jobs()
    assert "every 30s" in result


def test_list_every_job_non_minute_seconds(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    tool._cron.add_job(
        name="Ninety-second check",
        schedule=CronSchedule(kind="every", every_ms=90_000),
        message="check",
    )
    result = tool._list_jobs()
    assert "every 90s" in result


def test_list_every_job_milliseconds(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    tool._cron.add_job(
        name="Sub-second check",
        schedule=CronSchedule(kind="every", every_ms=200),
        message="check",
    )
    result = tool._list_jobs()
    assert "every 200ms" in result


def test_list_at_job_shows_iso_timestamp(tmp_path) -> None:
    tool = _make_tool_with_tz(tmp_path, "Asia/Shanghai")
    tool._cron.add_job(
        name="One-shot",
        schedule=CronSchedule(kind="at", at_ms=1773684000000),
        message="fire",
    )
    result = tool._list_jobs()
    assert "at 2026-" in result
    assert "Asia/Shanghai" in result


def test_list_shows_last_run_state(tmp_path) -> None:
    cases.definition_display(_make_tool(tmp_path / 'definition'), field='last')


def test_list_shows_error_message(tmp_path) -> None:
    cases.definition_display(_make_tool(tmp_path / 'definition'), field='error')


def test_list_shows_next_run(tmp_path) -> None:
    cases.definition_display(_make_tool(tmp_path / 'definition'), field='next')


async def test_add_cron_job_defaults_to_tool_timezone(tmp_path) -> None:
    tool = _make_tool_with_tz(tmp_path, "Asia/Shanghai")
    with request_context(RequestContext(channel="telegram", chat_id="chat-1", session_key="telegram:chat-1")):
        result = await tool.execute(action="add", message="Morning standup", cron_expr="0 8 * * *")

    assert result.startswith("Created job")
    job = CronService(tool._cron.store_path).list_jobs()[0]
    assert job.schedule.tz == "Asia/Shanghai"


async def test_add_cron_job_accepts_additional_destinations(tmp_path) -> None:
    tool = _make_tool_with_tz(tmp_path, "Asia/Kuala_Lumpur")
    with request_context(RequestContext(channel="telegram", chat_id="6344587670", session_key="telegram:6344587670")):
        result = await tool.execute(
            action="add", message="Daily report", cron_expr="0 8 * * *",
            additional_destinations=[{"channel": "telegram", "to": "-1001234567890"}],
        )

    assert result.startswith("Created job")
    job = CronService(tool._cron.store_path).list_jobs()[0]
    assert job.payload.delivery_destinations() == [
        CronDestination(channel="telegram", to="6344587670"),
        CronDestination(channel="telegram", to="-1001234567890"),
    ]
    assert "telegram:6344587670, telegram:-1001234567890" in await tool.execute(action="list")


async def test_add_cron_job_rejects_invalid_additional_destination(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    before = tool._cron.action_store.replay()
    with request_context(RequestContext(channel="telegram", chat_id="6344587670", session_key="telegram:6344587670")):
        result = await tool.execute(
            action="add", message="Daily report", every_seconds=60,
            additional_destinations=[{"channel": "telegram", "to": ""}],
        )

    assert tool._cron.action_store.replay() == before
    assert CronService(tool._cron.store_path).list_jobs() == []
    assert result == "Error: each additional destination requires channel and to"


@pytest.mark.parametrize("destination", [{"channel": "", "to": "42"}, {"channel": "  ", "to": "42"}, {"channel": "telegram", "to": " \t"}])
async def test_blank_destination_diagnostic_has_no_definition_effect(tmp_path, destination) -> None:
    tool = _make_tool(tmp_path)
    before = tool._cron.action_store.replay()
    with request_context(RequestContext(channel="telegram", chat_id="42", session_key="telegram:42")):
        result = await tool.execute(action="add", message="test", every_seconds=60, additional_destinations=[destination])
    assert result == "Error: each additional destination requires channel and to"
    assert tool._cron.action_store.replay() == before
    assert CronService(tool._cron.store_path).list_jobs() == []


async def test_list_delivery_display_preserves_order_and_omits_silent_jobs(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    job = tool._cron.add_job(
        name="ordered", schedule=CronSchedule(kind="every", every_ms=60000), message="test",
        deliver=True, channel="telegram", to="42",
        additional_destinations=[CronDestination("slack", "C2"), CronDestination("telegram", "42"), CronDestination("telegram", "43")],
        session_key="telegram:42", origin_channel="telegram", origin_chat_id="42",
    )
    before = tool._cron.action_store.replay()
    # A mutated detached projection cannot control the list's real definition replay.
    job.payload.additional_destinations.reverse()
    result = await tool.execute(action="list")
    assert "  Delivery: telegram:42, slack:C2, telegram:43" in result
    assert tool._cron.action_store.replay() == before
    silent = _make_tool(tmp_path / "silent")
    silent._cron.add_job(
        name="silent", schedule=CronSchedule(kind="every", every_ms=60000), message="test",
        deliver=False, additional_destinations=[CronDestination("telegram", "42")],
        session_key="telegram:42", origin_channel="telegram", origin_chat_id="42",
    )
    assert "Delivery:" not in await silent.execute(action="list")
    empty = _make_tool(tmp_path / "empty")
    assert await empty.execute(action="list") == "No scheduled jobs."


async def test_add_at_job_uses_default_timezone_for_naive_datetime(tmp_path) -> None:
    tool = _make_tool_with_tz(tmp_path, "Asia/Shanghai")
    with request_context(RequestContext(channel="telegram", chat_id="chat-1", session_key="telegram:chat-1")):
        result = await tool.execute(action="add", message="Morning reminder", at="2026-03-25T08:00:00")

    assert result.startswith("Created job")
    job = CronService(tool._cron.store_path).list_jobs()[0]
    expected = int(datetime(2026, 3, 25, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert job.schedule.at_ms == expected


def test_list_excludes_disabled_jobs(tmp_path) -> None:
    tool = _make_tool(tmp_path)
    job = tool._cron.add_job('Disabled job', CronSchedule(kind='every', every_ms=60000), 'test', session_key='websocket:chat-1', origin_channel='websocket', origin_chat_id='chat-1')
    disabled = tool._cron.enable_job(job.id, enabled=False, **cases.tokens(job))
    assert disabled is not None and not disabled.enabled
    assert tool._list_jobs() == 'No scheduled jobs.'
    assert 'Disabled job' not in tool._list_jobs()
    assert [item.id for item in tool._cron.list_jobs(include_disabled=True)] == [job.id]
