"""Test-only foreground gateway with exact local effects, never public activation.

The normal CLI remains migration-blocked. This subprocess fixture exercises its
real foreground composition under the same explicit positive-test admission as
other P6 fixtures, while retaining receipt checks and durable session leases.
"""

from __future__ import annotations

import errno
import os
import shutil
import socket
import sys
from pathlib import Path
from stat import S_ISFIFO

import pytest

from nanobot.agent.tools.sandbox import managed_bwrap_available
from nanobot.agent.tools.shell import ExecTool
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

    # Preflight the real Sandbox before installing the exact-effect audit hook.
    # Only its fixed printf command is admitted below; model calls stay forbidden.
    managed_bwrap_available()
    allowed_shell_commands = {"printf shell-ok"}

    def is_expected_native_supervisor(executable: object, arguments: object) -> bool:
        if executable != sys.executable or not isinstance(arguments, (list, tuple)):
            return False
        argv = list(arguments)
        if len(argv) != 12 or argv[:4] != [
            sys.executable,
            "-I",
            "-m",
            "nanobot.agent.tools.native_process_supervisor",
        ]:
            return False
        expected_flags = ["--gate-fd", "--status-fd", "--control-fd", "--config-fd"]
        descriptors: list[int] = []
        for index, flag in enumerate(expected_flags):
            flag_index = 4 + index * 2
            if argv[flag_index] != flag:
                return False
            value = argv[flag_index + 1]
            if not isinstance(value, str) or not value.isdecimal():
                return False
            descriptor = int(value)
            try:
                if not S_ISFIFO(os.fstat(descriptor).st_mode):
                    return False
            except OSError:
                return False
            descriptors.append(descriptor)
        return len(set(descriptors)) == len(expected_flags)

    def audit(event, args):
        if event in {"socket.bind", "socket.connect"}:
            stream, address = args
            if stream.family != socket.AF_INET or address not in endpoints:
                raise PermissionError(errno.EPERM, "smoke fixture network effect denied")
        if event == "subprocess.Popen":
            executable, arguments, _cwd, _env = args
            shells = {"/bin/sh", "/bin/bash", shutil.which("bash")}
            shell_command = (
                isinstance(arguments, (list, tuple))
                and len(arguments) in {3, 4}
                and list(arguments[:-1]) in ([executable, "-c"], [executable, "-l", "-c"])
                and arguments[-1] in allowed_shell_commands
            )
            if executable in shells and shell_command:
                return
            if is_expected_native_supervisor(executable, arguments):
                return
            raise PermissionError(errno.EPERM, "smoke fixture subprocess denied")
        if event in {"os.system", "os.exec", "os.posix_spawn"}:
            raise PermissionError(errno.EPERM, "smoke fixture process effect denied")

    async def no_model_call(*_args, **_kwargs):
        raise AssertionError("gateway smoke must not request a model response")

    sys.addaudithook(audit)
    with pytest.MonkeyPatch.context() as patches:
        execute = ExecTool.execute

        async def execute_expected_smoke_command(self, *args, **kwargs):
            command = kwargs.get("command") or kwargs.get("cmd")
            if command != "printf shell-ok":
                raise PermissionError(errno.EPERM, "smoke fixture command denied")
            prepared = self._prepare_command(command, login=kwargs.get("login"), shell=kwargs.get("shell"))
            if not isinstance(prepared, str) and hasattr(prepared, "command"):
                allowed_shell_commands.add(prepared.command)
            return await execute(self, *args, **kwargs)

        patches.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
        patches.setattr(UnconfiguredProvider, "chat", no_model_call)
        patches.setattr(ExecTool, "execute", execute_expected_smoke_command)
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
