"""Opt-in gate for tests that send real signals to real processes (A3-2M, WP0).

The macOS launch-domain engine kills processes. A mistaken membership test once
killed unrelated user processes on a developer machine, so any test that sends
a real signal is marked darwin_host and runs only on macOS with an explicit
opt-in, which the CI macOS job sets. Fake-table unit tests carry no marker and
run everywhere.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import pytest

DESTRUCTIVE_ENV = "KOHAKU_DARWIN_DESTRUCTIVE_TESTS"


_Test = TypeVar("_Test", bound=Callable[..., Any])


def native_supervisor_marks() -> list[pytest.MarkDecorator]:
    """Marks for tests that drive the real native supervisor.

    Linux runs them whenever the supervisor is available. macOS runs them only on
    an opted-in host, because they kill real processes, with the Seatbelt switch
    enabled by the native_supervisor_host fixture.
    """
    if sys.platform == "darwin":
        return [pytest.mark.darwin_host, pytest.mark.usefixtures("native_supervisor_host")]
    return [pytest.mark.skipif(
        sys.platform != "linux",
        reason="requires a native whole-tree supervisor (Linux or macOS)",
    )]


def requires_native_supervisor(test: _Test) -> _Test:
    for mark in native_supervisor_marks():
        test = mark(test)
    return test


def managed_confinement_marks() -> list[pytest.MarkDecorator]:
    """Marks for tests that run a command inside the real managed sandbox.

    Linux runs them whenever bubblewrap is usable. macOS confinement also ends
    each command's process tree with real signals, so it follows the same opt-in
    as the other native supervisor tests.
    """
    if sys.platform == "darwin":
        return native_supervisor_marks()
    from kohaku.agent.tools.sandbox import managed_bwrap_available

    return [pytest.mark.skipif(not managed_bwrap_available(), reason="managed bwrap unavailable")]


def requires_managed_confinement(test: _Test) -> _Test:
    for mark in managed_confinement_marks():
        test = mark(test)
    return test


def darwin_host_skip_reason(platform: str, environ: Mapping[str, str]) -> str | None:
    """Return why darwin_host tests must be skipped, or None when they may run."""
    if platform != "darwin":
        return "darwin_host tests run only on macOS"
    if environ.get(DESTRUCTIVE_ENV) != "1":
        return (
            f"set {DESTRUCTIVE_ENV}=1 on a disposable macOS host (the CI job does) "
            "to run tests that signal real processes"
        )
    return None
