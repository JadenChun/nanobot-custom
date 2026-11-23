"""Read-only Tailscale/Serve readiness for the desktop pairing UI."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, cast
from urllib.parse import urlparse


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


def _serve_target(
    serve: dict[str, Any],
    *,
    dns_name: str,
    expected_port: int | None,
) -> str | None:
    """Return the exact /mobile/v1 proxy target, if it matches the host."""
    web = serve.get("Web")
    if not isinstance(web, dict):
        return None
    web_map = cast(dict[str, Any], web)
    for key, entry in web_map.items():
        if not key.startswith(f"{dns_name}:"):
            continue
        if not isinstance(entry, dict):
            continue
        entry_map = cast(dict[str, Any], entry)
        handlers = entry_map.get("Handlers")
        if not isinstance(handlers, dict):
            continue
        handler_map = cast(dict[str, Any], handlers)
        handler = handler_map.get("/mobile/v1")
        if not isinstance(handler, dict):
            continue
        handler_value = cast(dict[str, Any], handler)
        proxy = handler_value.get("Proxy")
        if not isinstance(proxy, str) or not proxy:
            continue
        if expected_port is not None:
            parsed = urlparse(proxy)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
                or parsed.port != expected_port
            ):
                continue
        return proxy
    return None


def tailscale_status(expected_port: int | None = None) -> dict[str, Any]:
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
    target = _serve_target(serve, dns_name=dns_text, expected_port=expected_port)
    mapped = target is not None
    origin = f"https://{dns_text}" if dns_text else ""
    return {
        "available": True,
        "connected": connected,
        "serve_mapped": mapped,
        "serve_target": target or "",
        "dns_name": dns_text,
        "origin": origin,
        "error": "" if connected and mapped else (
            "Tailscale Serve is missing the exact /mobile/v1 target"
            if connected else "Tailscale is not connected"
        ),
    }
