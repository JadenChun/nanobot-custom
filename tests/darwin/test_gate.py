"""The darwin_host opt-in gate (A3-2M, WP0)."""

from __future__ import annotations

import os
import sys

import pytest

from tests.darwin.gate import DESTRUCTIVE_ENV, darwin_host_skip_reason


@pytest.mark.parametrize("platform", ["linux", "win32", "freebsd"])
def test_other_platforms_always_skip(platform) -> None:
    assert darwin_host_skip_reason(platform, {DESTRUCTIVE_ENV: "1"}) is not None


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", "2"])
def test_macos_needs_the_exact_opt_in(value) -> None:
    environ = {} if value is None else {DESTRUCTIVE_ENV: value}

    reason = darwin_host_skip_reason("darwin", environ)

    assert reason is not None
    assert DESTRUCTIVE_ENV in reason


def test_macos_with_opt_in_runs() -> None:
    assert darwin_host_skip_reason("darwin", {DESTRUCTIVE_ENV: "1"}) is None


@pytest.mark.darwin_host
def test_marked_tests_reach_the_body_only_on_an_opted_in_macos_host() -> None:
    # Skipped everywhere else, so executing this body proves the gate opened.
    assert sys.platform == "darwin"
    assert os.environ[DESTRUCTIVE_ENV] == "1"
