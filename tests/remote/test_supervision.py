from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nanobot.config.schema import RemoteAccessConfig
from nanobot.remote.supervision import (
    UNIT_PREFIX,
    ServicePlan,
    SupervisionError,
    SystemdUserAdapter,
    build_service_plan,
)


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="active\n", stderr="")


def test_plan_is_exact_and_owned(tmp_path: Path) -> None:
    plan = build_service_plan(
        RemoteAccessConfig(enabled=True, state_dir=str(tmp_path), socket_path=str(tmp_path / "owner.sock")),
        python_executable="/usr/bin/python3",
        unit_dir=tmp_path / "units",
    )
    assert plan.unit_name.startswith(UNIT_PREFIX)
    assert plan.unit_path.name == plan.unit_name
    assert "/usr/bin/python3" in plan.unit_text()
    assert "Restart=on-failure" in plan.unit_text()


def test_install_remove_are_dry_by_default_and_preserve_other_units(tmp_path: Path) -> None:
    runner = FakeRunner()
    adapter = SystemdUserAdapter(runner=runner)
    plan = build_service_plan(
        RemoteAccessConfig(enabled=True, state_dir=str(tmp_path), socket_path=str(tmp_path / "owner.sock")),
        unit_dir=tmp_path / "units",
    )
    preview = adapter.install(plan)
    assert preview["applied"] is False
    assert not plan.unit_path.exists()
    result = adapter.install(plan, apply=True)
    assert result["applied"] is True
    assert plan.unit_path.exists()
    assert ["systemctl", "--user", "enable", "--now", plan.unit_name] in runner.commands
    removed = adapter.remove(plan, apply=True)
    assert removed["applied"] is True
    assert not plan.unit_path.exists()
    assert ["systemctl", "--user", "disable", "--now", plan.unit_name] in runner.commands
    assert all("daemon-reload" in command or "enable" in command or "disable" in command for command in runner.commands)


def test_unowned_unit_is_rejected(tmp_path: Path) -> None:
    adapter = SystemdUserAdapter(runner=FakeRunner())
    plan = build_service_plan(
        RemoteAccessConfig(enabled=True, state_dir=str(tmp_path), socket_path=str(tmp_path / "owner.sock")),
        unit_dir=tmp_path / "units",
    )
    bad = ServicePlan(
        unit_name="evil.service",
        unit_path=tmp_path / "evil.service",
        command=plan.command,
        state_dir=plan.state_dir,
        socket_path=plan.socket_path,
        description="evil",
    )
    with pytest.raises(SupervisionError):
        adapter.install(bad)
