"""Test-only foreground gateway with exact local effects, never public activation.

The normal CLI remains migration-blocked. This subprocess fixture exercises its
real foreground composition under the same explicit positive-test admission as
other P6 fixtures, while retaining receipt checks and durable session leases.
"""

from __future__ import annotations

import errno
import shutil
import socket
import sys
from pathlib import Path

import pytest

from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.cli.gateway_runtime import _run_gateway
from nanobot.config.loader import load_config, set_config_path
from nanobot.providers.unconfigured_provider import UnconfiguredProvider


def main() -> None:
    path = Path(sys.argv[1])
    config = load_config(path)
    set_config_path(path)
    assert config.runtime_data_dir == path.parent
    assert config.workspace_path == path.parent / "workspace"
    assert config.agents.defaults.model == "custom/smoke-model"
    assert config.agents.defaults.dream.enabled is False
    assert config.gateway.heartbeat.enabled is False
    assert not config.tools.mcp_servers
    websocket = WebSocketConfig.model_validate(config.channels.websocket)
    assert websocket.host == "127.0.0.1"
    assert config.gateway.host == "127.0.0.1"
    endpoints = {("127.0.0.1", websocket.port), ("127.0.0.1", config.gateway.port)}

    def audit(event, args):
        if event in {"socket.bind", "socket.connect"}:
            stream, address = args
            if stream.family != socket.AF_INET or address not in endpoints:
                raise PermissionError(errno.EPERM, "smoke fixture network effect denied")
        if event == "subprocess.Popen":
            executable, arguments, _cwd, _env = args
            shells = {"/bin/sh", "/bin/bash", shutil.which("bash")}
            if executable not in shells or list(arguments) not in (
                [executable, "-c", "printf shell-ok"],
                [executable, "-l", "-c", "printf shell-ok"],
            ):
                raise PermissionError(errno.EPERM, "smoke fixture subprocess denied")
        if event in {"os.system", "os.exec", "os.posix_spawn"}:
            raise PermissionError(errno.EPERM, "smoke fixture process effect denied")

    async def no_model_call(*_args, **_kwargs):
        raise AssertionError("gateway smoke must not request a model response")

    sys.addaudithook(audit)
    with pytest.MonkeyPatch.context() as patches:
        patches.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
        patches.setattr(UnconfiguredProvider, "chat", no_model_call)
        # Memory Git history is independent of transport/recovery; do not run
        # its optional hook subprocesses in this exact-effect smoke fixture.
        patches.setattr("nanobot.utils.gitstore.GitStore.init", lambda self: False)
        _run_gateway(
            config,
            webui_static_dist=False,
            health_server_enabled=False,
            unconfigured_provider_error="test-only provider; model calls forbidden",
        )


if __name__ == "__main__":
    main()
