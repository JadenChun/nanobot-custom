"""Pure schedule arithmetic over immutable C2 commit evidence; no wall-clock inference."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Mapping, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

MAX_CRON_CANDIDATES = 400_000
SCHEDULE_FORMAT = 1


class ScheduleError(ValueError):
    """Missing or unsupported immutable scheduling input."""


class ScheduleClock(Protocol):
    def wall_time_ms(self) -> int: ...
    def monotonic_time(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


def schedule_digest(schedule: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(schedule), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(b"nanobot-cron-schedule-v1\0" + encoded.encode()).hexdigest()


def next_fire(schedule: Mapping[str, object], anchor_ms: int, after_ms: int) -> int | None:
    """First scheduled UTC millisecond strictly after the supplied boundary.

    Croniter enumerates *naive local* candidates. Both folds are round-trip
    validated separately, avoiding its timezone arithmetic around DST gaps.
    A two-day lookbehind covers all IANA UTC-offset discontinuities. The bounded
    search fails closed rather than returning an approximate fire.
    """
    if type(anchor_ms) is not int or type(after_ms) is not int:
        raise ScheduleError("schedule clocks must be exact integers")
    kind = schedule.get("kind")
    if kind == "every":
        interval = schedule.get("everyMs")
        if type(interval) is not int or interval < 1:
            raise ScheduleError("invalid interval")
        return anchor_ms + max(1, (after_ms - anchor_ms) // interval + 1) * interval
    if kind == "at":
        at = schedule.get("atMs")
        if type(at) is not int or at < 0:
            raise ScheduleError("invalid one-shot timestamp")
        return at if at > after_ms else None
    if kind != "cron":
        raise ScheduleError("unknown schedule kind")
    timezone, expression = schedule.get("tz"), schedule.get("expr")
    if not isinstance(timezone, str) or not timezone or not isinstance(expression, str):
        raise ScheduleError("cron requires an explicit IANA timezone and expression")
    try:
        zone = ZoneInfo(timezone)
        local = datetime.fromtimestamp(after_ms / 1000, UTC).astimezone(zone).replace(tzinfo=None)
        iterator = croniter(expression, local - timedelta(days=2))
        best: int | None = None
        previous = None
        for _ in range(MAX_CRON_CANDIDATES):
            candidate = iterator.get_next(datetime)
            if previous is not None and candidate <= previous:
                raise ScheduleError("non-increasing local cron candidate")
            previous = candidate
            for fold in (0, 1):
                aware = candidate.replace(tzinfo=zone, fold=fold)
                utc = aware.astimezone(UTC)
                restored = utc.astimezone(zone)
                if restored.replace(tzinfo=None) != candidate or restored.fold != fold:
                    continue
                instant = int(utc.timestamp()) * 1000 + utc.microsecond // 1000
                if instant > after_ms and (best is None or instant < best):
                    best = instant
            if best is not None:
                # Every future local candidate is at most 24h from UTC. Once
                # beyond best+2d, no later local candidate can beat this fire.
                limit = datetime.fromtimestamp(best / 1000, UTC).replace(tzinfo=None)
                if candidate > limit + timedelta(days=2):
                    return best
        raise ScheduleError("cron candidate bound exceeded")
    except (ZoneInfoNotFoundError, OverflowError, ValueError) as exc:
        raise ScheduleError("invalid or unrepresentable cron schedule") from exc


def is_fire(schedule: Mapping[str, object], anchor_ms: int, instant_ms: int) -> bool:
    return next_fire(schedule, anchor_ms, instant_ms - 1) == instant_ms
