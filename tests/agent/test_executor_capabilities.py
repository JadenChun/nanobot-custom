"""Executor capability seam (A3-2M, WP1)."""

from __future__ import annotations

import dataclasses

import pytest

from kohaku.agent.tools import executor_capabilities as caps
from kohaku.agent.tools import native_process


def _platform(monkeypatch, value: str) -> None:
    monkeypatch.setattr(caps, "current_platform", lambda: value)


def test_linux_reports_bwrap_and_subreaper(monkeypatch) -> None:
    _platform(monkeypatch, "linux")
    monkeypatch.setattr("kohaku.agent.tools.sandbox.managed_bwrap_available", lambda: True)

    assert caps.executor_capabilities() == caps.ExecutorCapabilities(
        platform="linux", confinement="bwrap", tree_ownership="linux_subreaper",
    )
    assert caps.managed_confinement_available() is True
    assert caps.tree_ownership_available() is True


def test_linux_without_bwrap_keeps_tree_ownership(monkeypatch) -> None:
    _platform(monkeypatch, "linux")
    monkeypatch.setattr("kohaku.agent.tools.sandbox.managed_bwrap_available", lambda: False)

    snapshot = caps.executor_capabilities()

    assert snapshot.confinement is None
    assert snapshot.tree_ownership == "linux_subreaper"


@pytest.mark.parametrize("platform", ["darwin", "other"])
def test_unsupported_platforms_fail_closed_even_if_bwrap_is_patched(
    monkeypatch, platform,
) -> None:
    _platform(monkeypatch, platform)
    monkeypatch.setattr("kohaku.agent.tools.sandbox.managed_bwrap_available", lambda: True)

    snapshot = caps.executor_capabilities()

    assert snapshot == caps.ExecutorCapabilities(
        platform=platform, confinement=None, tree_ownership=None,
    )
    assert caps.managed_confinement_available() is False
    assert caps.tree_ownership_available() is False


@pytest.mark.parametrize("platform", ["linux", "darwin", "other"])
def test_native_supervisor_wrapper_follows_the_seam(monkeypatch, platform) -> None:
    _platform(monkeypatch, platform)

    assert native_process.native_supervisor_available() is caps.tree_ownership_available()
    assert native_process.native_supervisor_available() is (platform == "linux")


def test_snapshot_is_immutable() -> None:
    snapshot = caps.ExecutorCapabilities("linux", None, None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.confinement = "bwrap"  # type: ignore[misc]
