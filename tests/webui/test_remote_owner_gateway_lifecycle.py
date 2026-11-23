"""Exercise the Remote Access setting through an isolated command-line gateway."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import httpx
import pytest
import websockets

from tests.webui.test_gateway_webui_smoke import (
    _free_port,
    _get_json,
    _recv_until,
    _stop_gateway,
    _wait_for_bootstrap,
    _write_smoke_config,
)


class _Instance(NamedTuple):
    config: Path
    state_dir: Path
    web_url: str
    remote_port: int

    @property
    def remote_url(self) -> str:
        return f"http://127.0.0.1:{self.remote_port}"


def _instance(tmp_path: Path, *, enabled: bool) -> _Instance:
    config = tmp_path / "config.json"
    state_dir = tmp_path / "remote-access"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ws_port, gateway_port, remote_port = (_free_port() for _ in range(3))
    _write_smoke_config(
        config, workspace=workspace, ws_port=ws_port, gateway_port=gateway_port
    )
    payload = json.loads(config.read_text())
    payload["remoteAccess"] = {
        "enabled": enabled, "stateDir": str(state_dir), "listenerPort": remote_port,
    }
    config.write_text(json.dumps(payload))
    return _Instance(config, state_dir, f"http://127.0.0.1:{ws_port}", remote_port)


@contextmanager
def _running_gateway(instance: _Instance) -> Iterator[dict]:
    log_path = instance.config.with_name("gateway.log")
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable, "-m", "kohaku", "gateway",
                "--foreground", "--config", str(instance.config),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        yield _wait_for_bootstrap(instance.web_url, process, log_path)
    finally:
        _stop_gateway(process)


def _status(instance: _Instance, bootstrap: dict) -> dict:
    return _get_json(
        f"{instance.web_url}/api/settings/remote-access", token=bootstrap["api_token"]
    )


def _assert_ready(instance: _Instance, bootstrap: dict) -> str:
    status = _status(instance, bootstrap)
    assert status["enabled"] is True
    assert status["owner_available"] is True
    assert status["installation_id"]
    assert _status_code(f"{instance.remote_url}/remote/v1/status") == 401
    assert _status_code(f"{instance.web_url}/mobile/v1/bootstrap") == 401
    return status["installation_id"]


def _status_code(url: str) -> int:
    return httpx.get(url, timeout=3, trust_env=False).status_code


def _assert_remote_port_closed(instance: _Instance) -> None:
    with pytest.raises(ConnectionRefusedError), socket.create_connection(
        ("127.0.0.1", instance.remote_port), timeout=2
    ):
        pass


async def _set_remote_access(bootstrap: dict, enabled: bool) -> dict:
    url = f'{bootstrap["ws_url"]}?token={bootstrap["token"]}&client_id=remote-lifecycle'
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            "type": "webui_request",
            "request_id": f"remote-lifecycle-enable-{enabled}",
            "action": "settings.remote_access.enable",
            "payload": {"enabled": enabled},
        }))
        return await _recv_until(ws, "webui_response")


def test_enabled_gateway_starts_owner_and_preserves_installation(tmp_path: Path) -> None:
    instance = _instance(tmp_path, enabled=True)
    with _running_gateway(instance) as bootstrap:
        installation_id = _assert_ready(instance, bootstrap)
    _assert_remote_port_closed(instance)

    with _running_gateway(instance) as bootstrap:
        assert _assert_ready(instance, bootstrap) == installation_id
    _assert_remote_port_closed(instance)


async def test_settings_start_and_stop_the_remote_listener(tmp_path: Path) -> None:
    instance = _instance(tmp_path, enabled=False)
    with _running_gateway(instance) as bootstrap:
        assert _status(instance, bootstrap)["enabled"] is False
        assert _status_code(f"{instance.web_url}/mobile/v1/bootstrap") == 404
        _assert_remote_port_closed(instance)
        assert (await _set_remote_access(bootstrap, True))["ok"] is True
        _assert_ready(instance, bootstrap)

    with _running_gateway(instance) as bootstrap:
        _assert_ready(instance, bootstrap)
        assert (await _set_remote_access(bootstrap, False))["ok"] is True
        assert _status(instance, bootstrap)["enabled"] is False
        assert _status_code(f"{instance.web_url}/mobile/v1/bootstrap") == 404
        _assert_remote_port_closed(instance)

    with _running_gateway(instance) as bootstrap:
        assert _status(instance, bootstrap)["enabled"] is False
        assert _status_code(f"{instance.web_url}/mobile/v1/bootstrap") == 404
        _assert_remote_port_closed(instance)


def test_gateway_reuses_an_independent_remote_owner(tmp_path: Path) -> None:
    instance = _instance(tmp_path, enabled=True)
    owner = subprocess.Popen([
        sys.executable, "-m", "kohaku.remote.owner",
        "--state-dir", str(instance.state_dir), "--listener-port", str(instance.remote_port),
    ])
    try:
        url = f"{instance.remote_url}/remote/v1/status"
        deadline = time.monotonic() + 8
        while True:
            try:
                assert _status_code(url) == 401
                break
            except httpx.ConnectError:
                if time.monotonic() >= deadline or owner.poll() is not None:
                    raise
                time.sleep(0.1)
        with _running_gateway(instance) as bootstrap:
            _assert_ready(instance, bootstrap)
        assert _status_code(url) == 401
    finally:
        _stop_gateway(owner)


async def test_settings_keep_remote_access_disabled_if_port_is_occupied(tmp_path: Path) -> None:
    instance = _instance(tmp_path, enabled=False)
    with _running_gateway(instance) as bootstrap, socket.socket() as occupied:
        occupied.bind(("127.0.0.1", instance.remote_port))
        occupied.listen()
        assert (await _set_remote_access(bootstrap, True))["ok"] is False
        assert _status(instance, bootstrap)["enabled"] is False
        assert _status_code(f"{instance.web_url}/mobile/v1/bootstrap") == 404
