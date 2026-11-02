"""Read-only Tailscale/Serve readiness for the desktop pairing UI."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, cast


def _run_json(command: list[str]) -> dict[str, Any] | None:
    if not shutil.which(command[0]):
        return None
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        value: object = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def tailscale_status() -> dict[str, Any]:
    """Return bounded readiness data; never mutate Tailscale or Serve state."""

    binary = shutil.which("tailscale")
    if binary is None:
        return {
            "available": False,
            "connected": False,
            "serve_mapped": False,
            "dns_name": "",
            "origin": "",
            "error": "tailscale CLI is not installed",
        }
    status = _run_json([binary, "status", "--json"])
    if status is None:
        return {
            "available": True,
            "connected": False,
            "serve_mapped": False,
            "dns_name": "",
            "origin": "",
            "error": "Tailscale is not running or not signed in",
        }
    self_info: object = status.get("Self")
    self_object: dict[str, Any] = (
        cast(dict[str, Any], self_info) if isinstance(self_info, dict) else {}
    )
    dns_name: object = self_object.get("DNSName")
    dns_text = dns_name.strip(".") if isinstance(dns_name, str) else ""
    backend_state = status.get("BackendState")
    connected = isinstance(backend_state, str) and backend_state == "Running"
    serve = _run_json([binary, "serve", "status", "--json"]) or {}
    mapped = "/mobile/v1" in json.dumps(serve, sort_keys=True)
    origin = f"https://{dns_text}" if dns_text else ""
    return {
        "available": True,
        "connected": connected,
        "serve_mapped": mapped,
        "dns_name": dns_text,
        "origin": origin,
        "error": "" if connected and mapped else (
            "Tailscale Serve is missing /mobile/v1" if connected else "Tailscale is not connected"
        ),
    }
