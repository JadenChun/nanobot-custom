#!/usr/bin/env python3
"""Qualify an isolated local release switch and rollback with real processes.

The accepted deployment is the local desktop-managed gateway. This harness uses
the installed interpreters and the real gateway readiness contract. It never
touches the live release venv or state: each target gets an isolated config
directory and ports.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _write_config(source: Path, target: Path, *, gateway_port: int) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    gateway = data.setdefault("gateway", {})
    gateway["host"] = "127.0.0.1"
    gateway["port"] = gateway_port
    gateway["cron"] = {**gateway.get("cron", {}), "enabled": False}
    agents = data.setdefault("agents", {})
    defaults = agents.setdefault("defaults", {})
    defaults["workspace"] = str(target.parent / "workspace")
    defaults["contextRepos"] = []
    defaults["fallback"] = []
    for name, channel in data.get("channels", {}).items():
        if isinstance(channel, dict) and name != "websocket":
            channel["enabled"] = False
    websocket = data.setdefault("channels", {}).setdefault("websocket", {})
    websocket["enabled"] = True
    websocket["host"] = "127.0.0.1"
    websocket["port"] = gateway_port + 1
    remote = data.get("remoteAccess")
    if isinstance(remote, dict):
        remote["enabled"] = False
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / "workspace").mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data), encoding="utf-8")


def _health(port: int, *, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=timeout
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return payload if isinstance(payload, dict) else None


def _wait_ready(process: subprocess.Popen[bytes], port: int, *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"gateway exited with {process.returncode} before readiness")
        payload = _health(port)
        if payload is not None and payload.get("ready") is True:
            return payload
        time.sleep(0.25)
    raise RuntimeError("gateway did not become ready")


def _start(python: Path, config: Path, port: int, log: Path) -> subprocess.Popen[bytes]:
    handle = open(log, "wb")
    home = log.parent / "home"
    runtime = log.parent / "runtime"
    home.mkdir(parents=True, exist_ok=True)
    for name in ("config", "cache", "data", "state"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(runtime / "config"),
        "XDG_CACHE_HOME": str(runtime / "cache"),
        "XDG_DATA_HOME": str(runtime / "data"),
        "XDG_STATE_HOME": str(runtime / "state"),
        "PYTHONPATH": "",
    }
    process = subprocess.Popen(
        [
            str(python),
            "-m",
            "nanobot",
            "gateway",
            "--foreground",
            "--port",
            str(port),
            "--config",
            str(config),
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=log.parent,
        env=env,
        start_new_session=True,
    )
    process._qualification_log = handle  # type: ignore[attr-defined]
    return process


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    with __import__("contextlib").suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with __import__("contextlib").suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    handle = getattr(process, "_qualification_log", None)
    if handle is not None:
        handle.close()


def _run_failure(python: Path, config: Path, port: int, log: Path) -> int:
    """Run a target expected to fail; return its nonzero exit code."""
    process = _start(python, config, port, log)
    try:
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return -1
    finally:
        _stop(process)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18871)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    previous = Path(os.path.abspath(args.previous_python))
    candidate = Path(os.path.abspath(args.candidate_python))
    source = Path(os.path.abspath(args.config))
    for label, python in (("previous", previous), ("candidate", candidate)):
        if not python.is_file():
            raise SystemExit(f"{label} interpreter is missing: {python}")

    work = Path(tempfile.mkdtemp(prefix="nanobot-local-rollback-"))
    previous_config = work / "previous" / "config.json"
    candidate_config = work / "candidate" / "config.json"
    missing_config = work / "candidate-failing" / "missing.json"
    _write_config(source, previous_config, gateway_port=args.port)
    _write_config(source, candidate_config, gateway_port=args.port)

    receipt: dict[str, Any] = {
        "previous_python": str(previous),
        "candidate_python": str(candidate),
        "port": args.port,
        "work": str(work),
    }
    previous_process: subprocess.Popen[bytes] | None = None
    candidate_process: subprocess.Popen[bytes] | None = None
    try:
        # 1. Previous target is ready.
        previous_process = _start(
            previous, previous_config, args.port, work / "previous-gateway.log"
        )
        receipt["previous_ready"] = _wait_ready(
            previous_process, args.port, timeout=args.timeout
        )

        # 2. Pre-switch failure: the candidate cannot load its config, so the
        # live target must stay ready.
        exit_code = _run_failure(
            candidate,
            missing_config,
            args.port,
            work / "candidate-failing.log",
        )
        receipt["pre_switch_failure_exit"] = exit_code
        receipt["pre_switch_live_ready"] = _health(args.port)
        if (
            exit_code == 0
            or receipt["pre_switch_live_ready"] is None
            or receipt["pre_switch_live_ready"].get("ready") is not True
        ):
            raise RuntimeError("pre-switch failure disturbed the live target")

        # 3. Switch to the candidate and verify the live target.
        _stop(previous_process)
        previous_process = None
        candidate_process = _start(
            candidate, candidate_config, args.port, work / "candidate-gateway.log"
        )
        receipt["candidate_ready"] = _wait_ready(
            candidate_process, args.port, timeout=args.timeout
        )

        # 4. Post-switch failure: the candidate stops, then roll back.
        _stop(candidate_process)
        candidate_process = None
        exit_code = _run_failure(
            candidate,
            missing_config,
            args.port,
            work / "post-switch-failure.log",
        )
        receipt["post_switch_failure_exit"] = exit_code
        previous_process = _start(
            previous, previous_config, args.port, work / "rollback-gateway.log"
        )
        receipt["rollback_ready"] = _wait_ready(
            previous_process, args.port, timeout=args.timeout
        )
        receipt["result"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        receipt["result"] = "FAIL"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _stop(candidate_process)
        _stop(previous_process)
        if args.receipt is not None:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        if receipt.get("result") == "PASS":
            shutil.rmtree(work, ignore_errors=True)
        else:
            receipt["work"] = str(work)

    print(json.dumps(receipt, indent=2))
    return 0 if receipt.get("result") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
