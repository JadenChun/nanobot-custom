"""Opt-in gate for tests that send real signals to real processes (A3-2M, WP0).

The macOS launch-domain engine kills processes. A mistaken membership test once
killed unrelated user processes on a developer machine, so any test that sends
a real signal is marked darwin_host and runs only on macOS with an explicit
opt-in, which the CI macOS job sets. Fake-table unit tests carry no marker and
run everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping

DESTRUCTIVE_ENV = "KOHAKU_DARWIN_DESTRUCTIVE_TESTS"


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
