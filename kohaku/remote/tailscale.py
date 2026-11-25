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
) -> tuple[str, str]:
    """Return the proxy target and its readiness issue."""
    web = serve.get("Web")
    if not isinstance(web, dict):
        return "", "serve_missing"
    web_map = cast(dict[str, Any], web)
    issue = "serve_missing"
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
        if "/mobile/v1" not in handler_map:
            continue
        if issue == "serve_missing":
            issue = "serve_wrong_port"
        handler = handler_map["/mobile/v1"]
        if not isinstance(handler, dict):
            continue
        handler_value = cast(dict[str, Any], handler)
        proxy = handler_value.get("Proxy")
        if not isinstance(proxy, str) or not proxy:
            continue
        try:
            parsed = urlparse(proxy)
            port = parsed.port
        except ValueError:
            continue
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or (expected_port is not None and port != expected_port)
        ):
            continue
        if parsed.path.rstrip("/") != "/mobile/v1":
            issue = "serve_path_stripped"
            continue
        return proxy, ""
    return "", issue


def tailscale_status(expected_port: int | None = None) -> dict[str, Any]:
    """Return bounded readiness data; never mutate Tailscale or Serve state."""

    port = expected_port if expected_port is not None else 18790
    result: dict[str, Any] = {
        "available": False,
        "connected": False,
        "serve_mapped": False,
        "serve_target": "",
        "dns_name": "",
        "origin": "",
        "expected_port": expected_port,
        "recommended_command": (
            "tailscale serve --bg --https=443 --set-path=/mobile/v1 "
            f"http://127.0.0.1:{port}/mobile/v1"
        ),
    }
    binary = shutil.which("tailscale")
    if binary is None:
        return {**result, "issue": "cli_missing", "error": "The Tailscale CLI is not installed."}
    status = _run_json([binary, "status", "--json"])
    not_running = "Tailscale is not running or not signed in."
    if status is None:
        return {**result, "available": True, "issue": "not_running", "error": not_running}
    self_info: object = status.get("Self")
    self_object: dict[str, Any] = (
        cast(dict[str, Any], self_info) if isinstance(self_info, dict) else {}
    )
    dns_name: object = self_object.get("DNSName")
    dns_text = dns_name.strip(".") if isinstance(dns_name, str) else ""
    backend_state = status.get("BackendState")
    connected = isinstance(backend_state, str) and backend_state == "Running"
    serve = _run_json([binary, "serve", "status", "--json"]) or {}
    target, issue = _serve_target(serve, dns_name=dns_text, expected_port=expected_port)
    if not connected:
        issue = "not_running"
    port_hint = f" on port {expected_port}" if expected_port is not None else ""
    errors = {
        "": "",
        "not_running": not_running,
        "serve_missing": "Tailscale Serve is missing the exact /mobile/v1 target on this host.",
        "serve_wrong_port": f"The /mobile/v1 Serve target must use loopback HTTP{port_hint}.",
        "serve_path_stripped": (
            "Tailscale Serve strips /mobile/v1 when the proxy URL omits that path."
        ),
    }
    return {
        **result,
        "available": True,
        "connected": connected,
        "serve_mapped": bool(target),
        "serve_target": target,
        "dns_name": dns_text,
        "origin": f"https://{dns_text}" if dns_text else "",
        "issue": issue,
        "error": errors[issue],
    }
