"""Opt-in user-service supervision for the independent owner (M19)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from nanobot.config.schema import RemoteAccessConfig

UNIT_PREFIX = "nanobot-remote-access-"


class SupervisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServicePlan:
    unit_name: str
    unit_path: Path
    command: tuple[str, ...]
    state_dir: Path
    socket_path: Path
    description: str

    def unit_text(self) -> str:
        exec_start = " ".join(_quote(part) for part in self.command)
        return "\n".join(
            [
                "[Unit]",
                f"Description={self.description}",
                "After=network-online.target",
                "",
                "[Service]",
                "Type=simple",
                f"ExecStart={exec_start}",
                "Restart=on-failure",
                "RestartSec=2",
                "NoNewPrivileges=true",
                "PrivateTmp=true",
                "Environment=PYTHONUNBUFFERED=1",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )


def build_service_plan(
    config: RemoteAccessConfig,
    *,
    python_executable: str | None = None,
    unit_dir: Path | None = None,
) -> ServicePlan:
    state_dir = Path(config.state_dir).expanduser() if config.state_dir else None
    if state_dir is None:
        from nanobot.config.paths import get_data_dir

        state_dir = get_data_dir() / "remote-access"
    socket_path = Path(config.socket_path).expanduser() if config.socket_path else state_dir / "owner.sock"
    digest = hashlib.sha256(str(socket_path).encode()).hexdigest()[:12]
    unit_name = f"{UNIT_PREFIX}{digest}.service"
    target_dir = unit_dir or (Path.home() / ".config" / "systemd" / "user")
    command = [
        python_executable or sys.executable,
        "-m",
        "nanobot.remote.owner",
        "--state-dir",
        str(state_dir),
        "--socket",
        str(socket_path),
    ]
    if config.listener_port is not None:
        command.extend(["--listener-port", str(config.listener_port)])
    return ServicePlan(
        unit_name=unit_name,
        unit_path=target_dir / unit_name,
        command=tuple(command),
        state_dir=state_dir,
        socket_path=socket_path,
        description=f"Nanobot remote-access owner ({digest})",
    )


class SystemdUserAdapter:
    def __init__(
        self,
        *,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._runner = runner or _run

    def install(self, plan: ServicePlan, *, apply: bool = False) -> dict[str, object]:
        self._validate(plan)
        text = plan.unit_text()
        result: dict[str, object] = {
            "unit": plan.unit_name,
            "path": str(plan.unit_path),
            "unit_text": text,
            "commands": [
                ["systemctl", "--user", "daemon-reload"],
                ["systemctl", "--user", "enable", "--now", plan.unit_name],
            ],
            "applied": apply,
        }
        if not apply:
            return result
        plan.unit_path.parent.mkdir(parents=True, exist_ok=True)
        plan.unit_path.write_text(text, encoding="utf-8")
        self._runner(["systemctl", "--user", "daemon-reload"])
        self._runner(["systemctl", "--user", "enable", "--now", plan.unit_name])
        return result

    def remove(self, plan: ServicePlan, *, apply: bool = False) -> dict[str, object]:
        self._validate(plan)
        result: dict[str, object] = {
            "unit": plan.unit_name,
            "path": str(plan.unit_path),
            "commands": [
                ["systemctl", "--user", "disable", "--now", plan.unit_name],
                ["systemctl", "--user", "daemon-reload"],
            ],
            "applied": apply,
        }
        if not apply:
            return result
        self._runner(["systemctl", "--user", "disable", "--now", plan.unit_name])
        if plan.unit_path.exists():
            plan.unit_path.unlink()
        self._runner(["systemctl", "--user", "daemon-reload"])
        return result

    def status(self, plan: ServicePlan) -> str:
        self._validate(plan)
        result = self._runner(["systemctl", "--user", "is-active", plan.unit_name])
        return (result.stdout or "").strip() or "unknown"

    def _validate(self, plan: ServicePlan) -> None:
        if not plan.unit_name.startswith(UNIT_PREFIX) or not plan.unit_name.endswith(".service"):
            raise SupervisionError("refusing a unit outside the owned remote-access prefix")
        if plan.unit_path.name != plan.unit_name:
            raise SupervisionError("unit path does not match the owned unit name")
        if plan.socket_path.parent != plan.state_dir:
            raise SupervisionError("socket must remain inside the selected owner state directory")


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--unit-dir", type=Path)
    parser.add_argument("--listener-port", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "install", "remove", "status"):
        sub.add_parser(name)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = RemoteAccessConfig(
        enabled=True,
        state_dir=str(args.state_dir),
        socket_path=str(args.socket),
        listener_port=args.listener_port,
    )
    plan = build_service_plan(
        config,
        python_executable=args.python,
        unit_dir=args.unit_dir,
    )
    adapter = SystemdUserAdapter()
    if args.command == "plan":
        result: object = {
            "unit": plan.unit_name,
            "path": str(plan.unit_path),
            "unit_text": plan.unit_text(),
        }
    elif args.command == "install":
        result = adapter.install(plan, apply=args.apply)
    elif args.command == "remove":
        result = adapter.remove(plan, apply=args.apply)
    else:
        result = {"unit": plan.unit_name, "status": adapter.status(plan)}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
