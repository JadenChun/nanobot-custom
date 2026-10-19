from datetime import UTC, datetime

import pytest

from nanobot.cron.schedule import ScheduleError, is_fire, next_fire
from tests.cron.stage3_fixtures import offline_denial  # noqa: F401


def ms(value):
    return int(datetime.fromisoformat(value).astimezone(UTC).timestamp() * 1000)


def cron(expr, tz):
    return {"kind": "cron", "expr": expr, "tz": tz}


def test_exact_anchored_interval_and_one_shot_boundaries():
    schedule = {"kind": "every", "everyMs": 100}
    assert next_fire(schedule, 1234, 1234) == 1334
    assert next_fire(schedule, 1234, 1400) == 1434
    assert next_fire(schedule, 1234, 1200) == 1334
    assert is_fire(schedule, 1234, 1434)
    assert not is_fire(schedule, 1234, 1400)
    assert next_fire({"kind": "at", "atMs": 10}, 0, 10) is None
    assert next_fire({"kind": "at", "atMs": 10}, 0, 9) == 10


@pytest.mark.parametrize("tz", [None, "", "not/a/timezone"])
def test_explicit_iana_required(tz):
    with pytest.raises(ScheduleError):
        next_fire(cron("0 1 * * *", tz), 0, 1000)


def test_gap_skipped_and_both_fold_instants_are_distinct():
    schedule = cron("30 2 * * *", "America/New_York")
    assert next_fire(schedule, 0, ms("2026-03-08T06:00:00+00:00")) == ms("2026-03-09T06:30:00+00:00")
    schedule = cron("30 1 * * *", "America/New_York")
    first = next_fire(schedule, 0, ms("2026-11-01T04:00:00+00:00"))
    assert first == ms("2026-11-01T05:30:00+00:00")
    assert next_fire(schedule, 0, first) == ms("2026-11-01T06:30:00+00:00")


def test_non_hour_lord_howe_fold():
    schedule = cron("45 1 * * *", "Australia/Lord_Howe")
    first = next_fire(schedule, 0, ms("2026-04-04T13:00:00+00:00"))
    assert first == ms("2026-04-04T14:45:00+00:00")
    assert next_fire(schedule, 0, first) == ms("2026-04-04T15:15:00+00:00")


def test_six_field_seconds_and_named_months_preserved():
    assert next_fire(cron("* * * * * */15", "UTC"), 0, 1000) == 15000
    assert next_fire(cron("0 0 1 JAN *", "UTC"), 0, ms("2026-02-01T00:00:00+00:00")) == ms("2027-01-01T00:00:00+00:00")
