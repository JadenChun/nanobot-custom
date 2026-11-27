"""Host checks keep a success for good and retry a failure after a cooldown."""

from __future__ import annotations

import subprocess
import threading
import types
from collections.abc import Iterator

import pytest
from loguru import logger

from kohaku.agent.tools import executor_capabilities as capabilities
from kohaku.agent.tools import sandbox
from kohaku.agent.tools.executor_capabilities import CachedHostCheck
from kohaku.security import darwin_proc


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Check:
    def __init__(self, *results: bool) -> None:
        self.results = list(results)
        self.runs = 0

    def __call__(self) -> bool:
        self.runs += 1
        return self.results.pop(0)


def test_a_failure_is_retried_after_the_cooldown_and_not_before() -> None:
    clock, check = _Clock(), _Check(False, False, True)
    cached = CachedHostCheck(check, cooldown=300, clock=clock)

    assert cached() is False
    clock.now += 299.9
    assert cached() is False
    assert check.runs == 1
    clock.now += 0.1
    assert cached() is False
    assert check.runs == 2
    # The second failure starts a new cooldown.
    clock.now += 299.9
    assert cached() is False
    assert check.runs == 2
    clock.now += 0.1
    assert cached() is True
    assert check.runs == 3


def test_the_cooldown_starts_when_the_failed_attempt_ends() -> None:
    clock = _Clock()
    runs: list[float] = []

    def slow_failure() -> bool:
        runs.append(clock.now)
        clock.now += 60
        return False

    cached = CachedHostCheck(slow_failure, cooldown=300, clock=clock)
    cached()
    clock.now += 299
    cached()

    assert len(runs) == 1


def test_a_success_is_never_rerun_until_cleared() -> None:
    clock, check = _Clock(), _Check(True, False)
    cached = CachedHostCheck(check, cooldown=300, clock=clock)

    assert cached() is True
    clock.now += 1_000_000
    assert cached() is True
    assert check.runs == 1
    cached.cache_clear()
    assert cached() is False
    assert check.runs == 2


def test_a_check_that_raises_is_not_remembered() -> None:
    attempts: list[int] = []

    def check() -> bool:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("interrupted")
        return True

    cached = CachedHostCheck(check, clock=_Clock())
    with pytest.raises(RuntimeError, match="interrupted"):
        cached()

    assert cached() is True


@pytest.mark.parametrize("outcome", [True, False])
def test_concurrent_callers_run_the_check_once(outcome: bool) -> None:
    entered, release = threading.Event(), threading.Event()
    runs: list[int] = []
    results: list[bool] = []

    def check() -> bool:
        runs.append(1)
        entered.set()
        release.wait(timeout=5)
        return outcome

    cached = CachedHostCheck(check, clock=_Clock())
    threads = [
        threading.Thread(target=lambda: results.append(cached()), daemon=True) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=5)
    # Give the other callers time to arrive while the first is still checking.
    threads[-1].join(timeout=0.2)
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert len(runs) == 1
    assert results == [outcome] * 8


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Clock]:
    """Drive the module-level checks with a fake monotonic clock."""
    fake = _Clock()
    monkeypatch.setattr(capabilities, "time", types.SimpleNamespace(monotonic=fake))
    capabilities._macos_self_test_passed.cache_clear()
    sandbox._seatbelt_profile_probe.cache_clear()
    yield fake
    capabilities._macos_self_test_passed.cache_clear()
    sandbox._seatbelt_profile_probe.cache_clear()


def test_the_self_test_recovers_and_warns_once_per_failed_attempt(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock,
) -> None:
    results = [
        darwin_proc.SelfTestResult(False, ("busy",)),
        darwin_proc.SelfTestResult(False, ("still busy",)),
        darwin_proc.SelfTestResult(True, ()),
    ]
    monkeypatch.setattr(darwin_proc, "seatbelt_self_test", lambda: results.pop(0))
    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        assert capabilities._macos_self_test_passed() is False
        assert capabilities._macos_self_test_passed() is False
        assert len(warnings) == 1
        clock.now += 300
        assert capabilities._macos_self_test_passed() is False
        assert capabilities._macos_self_test_passed() is False
        assert len(warnings) == 2
        clock.now += 300
        assert capabilities._macos_self_test_passed() is True
        assert capabilities._macos_self_test_passed() is True
    finally:
        logger.remove(sink)

    assert len(warnings) == 2
    assert results == []


def test_the_profile_probe_recovers_from_a_timeout(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock,
) -> None:
    runs: list[int] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        runs.append(1)
        if len(runs) == 1:
            raise subprocess.TimeoutExpired(command, 10)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sandbox, "subprocess", types.SimpleNamespace(
        run=run, DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError,
    ))

    assert sandbox._seatbelt_profile_probe() is False
    assert sandbox._seatbelt_profile_probe() is False
    assert len(runs) == 1
    clock.now += 300
    assert sandbox._seatbelt_profile_probe() is True
    clock.now += 1_000_000
    assert sandbox._seatbelt_profile_probe() is True
    assert len(runs) == 2
