from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kohaku.config.schema import RemoteAccessConfig
from kohaku.remote.supervision import (
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
        RemoteAccessConfig(
            enabled=True,
            state_dir=str(tmp_path),
            socket_path=str(tmp_path / "owner.sock"),
            listener_port=18124,
        ),
        python_executable="/usr/bin/python3",
        unit_dir=tmp_path / "units",
    )
    assert plan.unit_name.startswith(UNIT_PREFIX)
    assert plan.unit_path.name == plan.unit_name
    assert "/usr/bin/python3" in plan.unit_text()
    assert "Restart=on-failure" in plan.unit_text()
    assert "--listener-port" in plan.unit_text()
    assert "18124" in plan.unit_text()


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


def test_supervision_cli_plan_carries_listener_port(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kohaku.remote.supervision",
            "--state-dir",
            str(tmp_path),
            "--socket",
            str(tmp_path / "owner.sock"),
            "--unit-dir",
            str(tmp_path / "units"),
            "--python",
            "/usr/bin/python3",
            "--listener-port",
            "18124",
            "--json",
            "plan",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "--listener-port" in payload["unit_text"]
    assert "18124" in payload["unit_text"]


class StepRunner:
    """Fake runner that returns a chosen return code for one systemctl step."""

    def __init__(
        self,
        *,
        fail_step: str | None = None,
        fail_returncode: int = 1,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.fail_step = fail_step
        self.fail_returncode = fail_returncode
        self.stdout = stdout
        self.stderr = stderr
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if self.fail_step is not None and self.fail_step in command:
            return subprocess.CompletedProcess(
                command, self.fail_returncode, stdout=self.stdout, stderr="step failed"
            )
        return subprocess.CompletedProcess(command, 0, stdout=self.stdout, stderr=self.stderr)


def _plan(tmp_path: Path) -> ServicePlan:
    return build_service_plan(
        RemoteAccessConfig(
            enabled=True,
            state_dir=str(tmp_path),
            socket_path=str(tmp_path / "owner.sock"),
        ),
        unit_dir=tmp_path / "units",
    )


def test_install_applied_on_success(tmp_path: Path) -> None:
    runner = StepRunner(stdout="active\n")
    adapter = SystemdUserAdapter(runner=runner)
    plan = _plan(tmp_path)

    result = adapter.install(plan, apply=True)

    assert result["applied"] is True
    assert plan.unit_path.exists()
    assert any("daemon-reload" in command for command in runner.commands)
    assert ["systemctl", "--user", "enable", "--now", plan.unit_name] in runner.commands


def test_install_not_applied_when_enable_fails(tmp_path: Path) -> None:
    runner = StepRunner(fail_step="enable")
    adapter = SystemdUserAdapter(runner=runner)
    plan = _plan(tmp_path)

    result = adapter.install(plan, apply=True)

    assert result["applied"] is not True
    assert result["failed_step"] == "enable"
    assert result["returncode"] == 1
    assert result["stderr"] == "step failed"
    assert ["systemctl", "--user", "enable", "--now", plan.unit_name] in runner.commands


def test_remove_keeps_unit_when_disable_fails(tmp_path: Path) -> None:
    runner = StepRunner(fail_step="disable")
    adapter = SystemdUserAdapter(runner=runner)
    plan = _plan(tmp_path)
    plan.unit_path.parent.mkdir(parents=True, exist_ok=True)
    plan.unit_path.write_text(plan.unit_text(), encoding="utf-8")

    result = adapter.remove(plan, apply=True)

    assert plan.unit_path.exists()
    assert result["applied"] is not True
    assert result["failed_step"] == "disable"
    assert result["returncode"] == 1
    assert ["systemctl", "--user", "disable", "--now", plan.unit_name] in runner.commands


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (0, "active\n", "active"),
        (3, "inactive\n", "inactive"),
        (3, "", "inactive"),
        (4, "", "not-found"),
        (1, "", "unknown"),
    ],
)
def test_status_reflects_command_result(
    tmp_path: Path, returncode: int, stdout: str, expected: str
) -> None:
    adapter = SystemdUserAdapter(
        runner=StepRunner(fail_step="is-active", fail_returncode=returncode, stdout=stdout)
    )
    plan = _plan(tmp_path)

    assert adapter.status(plan) == expected
