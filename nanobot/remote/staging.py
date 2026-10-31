"""Staged loopback owner/listener and dry-run private-entry planning (M09)."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nanobot.remote.client import OwnerRequestError, OwnerUnavailableError, RemoteAccessClient
from nanobot.remote.http import RemoteHTTPListener
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.protocol import ContractError, validate_origin

PREFIX = "/remote/v1"


@dataclass
class ServePlan:
    origin: str
    listener_port: int
    prefix: str
    prior: dict[str, Any]
    desired_command: list[str]
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ready: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TailscaleServeAdapter:
    """Read-only planning plus an explicit, separately opted-in apply command."""

    def __init__(self, tailscale_bin: str = "tailscale") -> None:
        self.tailscale_bin = tailscale_bin

    def status_json(self) -> dict[str, Any]:
        return self._run_json(["status", "--json"])

    def serve_status_json(self) -> dict[str, Any]:
        return self._run_json(["serve", "status", "--json"])

    def funnel_status_json(self) -> dict[str, Any]:
        try:
            return self._run_json(["funnel", "status", "--json"])
        except (OSError, RuntimeError, json.JSONDecodeError):
            return {}

    def _run_json(self, args: list[str]) -> dict[str, Any]:
        result = subprocess.run(
            [self.tailscale_bin, *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip() or "tailscale command failed")
        value = json.loads(result.stdout or "{}")
        if not isinstance(value, dict):
            raise RuntimeError("tailscale returned a non-object JSON value")
        return value

    def plan(self, *, origin: str, listener_port: int, prefix: str = PREFIX) -> ServePlan:
        status = self.status_json()
        serve = self.serve_status_json()
        funnel = self.funnel_status_json()
        prior = {
            "backend_state": status.get("BackendState"),
            "dns_name": str((status.get("Self") or {}).get("DNSName", "")),
            "serve": serve,
            "funnel": funnel,
        }
        conflicts: list[str] = []
        warnings: list[str] = []
        ready = True
        if status.get("BackendState") != "Running":
            ready = False
            warnings.append("tailscale backend is not running")
        self_info = status.get("Self") or {}
        dns_name = str(self_info.get("DNSName", "")).rstrip(".")
        expected_origin = f"https://{dns_name}" if dns_name else ""
        if not dns_name:
            ready = False
            warnings.append("tailscale self DNS name is unavailable")
        try:
            normalized_origin = validate_origin(origin)
        except ContractError as exc:
            normalized_origin = origin.rstrip("/")
            ready = False
            warnings.append(f"origin is not an accepted HTTPS origin: {exc}")
        if normalized_origin != expected_origin:
            ready = False
            warnings.append(f"origin must equal the tailnet origin {expected_origin!r}")
        if "insecure" in normalized_origin or "funnel" in normalized_origin:
            ready = False
            warnings.append("insecure or Funnel origins are not accepted")
        if not 1024 <= listener_port <= 65535:
            ready = False
            warnings.append("listener port must be an unprivileged loopback TCP port")
        if funnel:
            ready = False
            warnings.append("Funnel is active; public ingress is not accepted")
        if prefix in json.dumps(serve, sort_keys=True):
            ready = False
            conflicts.append(f"serve already has the owned prefix {prefix}")
        if serve and not conflicts:
            warnings.append("existing serve entries are preserved; teardown must be reviewed")
        command = [
            self.tailscale_bin,
            "serve",
            "--bg",
            "--https=443",
            f"--set-path={prefix}",
            f"http://127.0.0.1:{listener_port}",
        ]
        return ServePlan(
            origin=origin.rstrip("/"),
            listener_port=listener_port,
            prefix=prefix,
            prior=prior,
            desired_command=command,
            conflicts=conflicts,
            warnings=warnings,
            ready=ready,
        )

    def apply(self, plan: ServePlan, *, allow_live: bool = False) -> None:
        if not allow_live:
            raise RuntimeError("live private-entry changes require explicit local opt-in")
        if not plan.ready:
            raise RuntimeError("serve plan is not ready")
        subprocess.run(plan.desired_command, check=True, timeout=30)


def start_owner_and_listener(
    *,
    state_dir: Path,
    socket_path: Path,
    listener_port: int = 0,
) -> tuple[RemoteAccessOwner, RemoteHTTPListener]:
    owner = RemoteAccessOwner(state_dir, socket_path)
    owner.start()
    listener = RemoteHTTPListener(owner.pairing, host="127.0.0.1", port=listener_port)
    listener.start()
    return owner, listener


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="read-only private-entry plan; no mapping is changed")
    plan.add_argument("--origin", required=True)
    plan.add_argument("--port", type=int, required=True)
    plan.add_argument("--prefix", default=PREFIX)
    plan.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="read owner status over the private socket")
    status.add_argument("--socket", type=Path, required=True)

    approve = sub.add_parser("approve", help="approve one pending pairing request locally")
    approve.add_argument("--socket", type=Path, required=True)
    approve.add_argument("--admin-token-file", type=Path, required=True)
    approve.add_argument("--pairing-id", required=True)
    approve.add_argument("--code", required=True)
    approve.add_argument("--scope", action="append", default=[])

    start = sub.add_parser("start", help="run the loopback owner/listener in the foreground")
    start.add_argument("--state-dir", type=Path, required=True)
    start.add_argument("--socket", type=Path, required=True)
    start.add_argument("--port", type=int, default=0)
    start.add_argument("--ready-file", type=Path, default=None)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "plan":
        plan = TailscaleServeAdapter().plan(
            origin=args.origin, listener_port=args.port, prefix=args.prefix
        )
        if args.json:
            print(json.dumps(plan.as_dict(), sort_keys=True))
        else:
            print(f"ready={plan.ready}")
            print(f"prefix={plan.prefix} origin={plan.origin} listener={plan.listener_port}")
            for conflict in plan.conflicts:
                print(f"conflict: {conflict}")
            for warning in plan.warnings:
                print(f"warning: {warning}")
            print("command:", " ".join(plan.desired_command))
        return 0 if plan.ready else 1
    if args.command == "status":
        try:
            print(json.dumps(RemoteAccessClient(args.socket).status(), indent=2, sort_keys=True))
        except (OwnerUnavailableError, OwnerRequestError) as exc:
            print(f"owner unavailable: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "approve":
        token = args.admin_token_file.read_text(encoding="ascii").strip()
        client = RemoteAccessClient(args.socket, admin_token=token)
        try:
            result = client.approve_pairing(
                pairing_id=args.pairing_id,
                comparison_code=args.code,
                scopes=args.scope or None,
            )
        except (OwnerUnavailableError, OwnerRequestError) as exc:
            print(f"approval failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "start":
        owner, listener = start_owner_and_listener(
            state_dir=args.state_dir,
            socket_path=args.socket,
            listener_port=args.port,
        )
        owner.store.admin_token()
        ready = {
            "owner_socket": str(owner.socket_path),
            "listener": f"http://127.0.0.1:{listener.bound_port}",
            "admin_token_file": str(owner.store.admin_token_path),
        }
        if args.ready_file:
            args.ready_file.write_text(json.dumps(ready) + "\n", encoding="utf-8")
        print("REMOTE_STAGING_READY " + json.dumps(ready), flush=True)
        stop_event = threading.Event()

        def _stop(_signum: int, _frame: object) -> None:
            stop_event.set()

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        try:
            while not stop_event.wait(0.5):
                pass
        finally:
            listener.stop()
            owner.stop()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
