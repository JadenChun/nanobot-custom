"""The macOS executor switch gates tree ownership and fails closed (D-M5)."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from kohaku.agent.tools import executor_capabilities as capabilities
from kohaku.agent.tools import native_process, sandbox
from kohaku.config.loader import load_config
from kohaku.security import darwin_proc
from kohaku.security.desktop_permissions import WorkspacePermissionUnavailableError
from tests.agent.test_executor_capability_parity import BINDING, CHAT, _build, _commands


class _SelfTest:
    def __init__(self) -> None:
        self.ok = True
        self.runs: list[bool] = []

    def __call__(self) -> darwin_proc.SelfTestResult:
        self.runs.append(self.ok)
        return darwin_proc.SelfTestResult(self.ok, () if self.ok else ("broken",))


@pytest.fixture
def self_test(monkeypatch: pytest.MonkeyPatch) -> Iterator[_SelfTest]:
    """Control the self-test result and record how often it ran."""
    control = _SelfTest()
    monkeypatch.setattr(darwin_proc, "seatbelt_self_test", control)
    capabilities._macos_self_test_passed.cache_clear()
    yield control
    capabilities._macos_self_test_passed.cache_clear()


def _platform(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    monkeypatch.setattr(capabilities, "current_platform", lambda: platform)


def test_switch_off_never_runs_the_self_test(monkeypatch, self_test) -> None:
    _platform(monkeypatch, "darwin")

    assert capabilities.tree_ownership() is None
    assert self_test.runs == []
    assert native_process._supervisor_module().endswith("native_process_supervisor")


def test_switch_on_with_a_passing_self_test_enables_ownership_once(
    monkeypatch, self_test,
) -> None:
    _platform(monkeypatch, "darwin")
    capabilities.configure_macos_seatbelt(True)

    assert capabilities.tree_ownership() == "darwin_seatbelt"
    assert capabilities.tree_ownership() == "darwin_seatbelt"
    assert self_test.runs == [True]
    assert native_process._supervisor_module().endswith("native_process_supervisor_darwin")


def test_a_failing_self_test_fails_closed(monkeypatch, self_test) -> None:
    _platform(monkeypatch, "darwin")
    self_test.ok = False
    capabilities.configure_macos_seatbelt(True)

    assert capabilities.tree_ownership() is None
    assert capabilities.confinement() is None


@pytest.mark.parametrize("platform", ["linux", "other"])
def test_switch_changes_nothing_off_macos(monkeypatch, self_test, platform: str) -> None:
    _platform(monkeypatch, platform)
    before = capabilities.tree_ownership()
    capabilities.configure_macos_seatbelt(True)

    assert capabilities.tree_ownership() == before
    assert self_test.runs == []


def _macos(monkeypatch: pytest.MonkeyPatch, *, profile_ok: bool) -> None:
    """Report macOS everywhere a projection asks, with a chosen profile probe."""
    _platform(monkeypatch, "darwin")
    monkeypatch.setattr(sandbox, "current_platform", lambda: "darwin")
    monkeypatch.setattr(sandbox, "_seatbelt_profile_probe", lambda: profile_ok)
    monkeypatch.setattr(sandbox, "managed_bwrap_available", lambda: False)
    monkeypatch.setattr("kohaku.session.workspace_focus._linux_host", lambda: False)


@pytest.mark.parametrize(
    ("switch", "profile_ok", "full_expected", "once_expected"),
    [
        (False, True, (False, "none", "supervisor_unavailable"),
         (False, "none", "supervisor_unavailable")),
        (True, True, (True, "native", None), (True, "seatbelt", None)),
        # Ownership alone opens Full access. Once also needs confinement.
        (True, False, (True, "native", None), (False, "none", "sandbox_unavailable")),
    ],
)
def test_macos_projections_follow_the_switch(
    tmp_path: Path, monkeypatch, self_test, switch: bool, profile_ok: bool,
    full_expected: tuple[bool, str, str | None],
    once_expected: tuple[bool, str, str | None],
) -> None:
    _macos(monkeypatch, profile_ok=profile_ok)
    capabilities.configure_macos_seatbelt(switch)
    owner, sessions, _home = _build(tmp_path)
    enable = functools.partial(
        sessions.desktop_permissions.enable_full,
        session_key=CHAT, source_binding=dict(BINDING), persistent=True, existing=True,
        local=True, idle=True, configured_sandbox=False,
        supervisor_available=native_process.native_supervisor_available(),
        stores_healthy=True,
    )
    project = tmp_path / "project"
    project.mkdir()
    if not switch:
        with pytest.raises(WorkspacePermissionUnavailableError):
            enable()
    else:
        enable()
        full = owner.effective_access(
            project,
            permission_reference=sessions.desktop_permissions.full_reference(
                session_key=CHAT, source_binding=dict(BINDING), request_turn_id="turn-1",
            ),
        )
        assert (full["source"], _commands(full)) == ("full", full_expected)

    root = tmp_path / "task-root"
    root.mkdir()
    once = owner.effective_access(
        root,
        permission_reference=sessions.desktop_permissions.activate_once(
            session_key=CHAT, source_binding=dict(BINDING), task_turn_id="turn-once",
            task_input_event_id="input-once", path=root, file_access="read_write", commands=True,
        ),
    )
    assert (once["source"], _commands(once)) == ("task", once_expected)


@pytest.mark.parametrize(
    ("tools", "enabled"),
    [({}, False), ({"exec": {"macosSeatbelt": True}}, True)],
)
def test_loading_a_config_applies_the_switch(
    tmp_path: Path, monkeypatch, self_test, tools: dict[str, object], enabled: bool,
) -> None:
    _platform(monkeypatch, "darwin")
    capabilities.configure_macos_seatbelt(not enabled)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tools": tools}), encoding="utf-8")

    load_config(path)

    assert capabilities.macos_seatbelt_ready() is enabled


def test_loading_without_process_policy_leaves_the_switch_alone(
    tmp_path: Path, monkeypatch, self_test,
) -> None:
    _platform(monkeypatch, "darwin")
    capabilities.configure_macos_seatbelt(True)
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    load_config(path, apply_network_policy=False)

    assert capabilities.macos_seatbelt_ready() is True


@pytest.mark.parametrize(
    ("switch", "profile_ok", "expected"),
    [
        (False, True, (False, "none", "unsupported_platform")),
        (True, True, (True, "seatbelt", None)),
        (True, False, (False, "none", "unsupported_platform")),
    ],
)
def test_macos_sandbox_commands_follow_the_switch(
    tmp_path: Path, monkeypatch, self_test, switch: bool, profile_ok: bool,
    expected: tuple[bool, str, str | None],
) -> None:
    _macos(monkeypatch, profile_ok=profile_ok)
    capabilities.configure_macos_seatbelt(switch)
    owner, _sessions, home = _build(tmp_path)

    access = owner.effective_access(home)

    assert access["policy"] == "managed"
    assert _commands(access) == expected
