"""Focused S2 checks for headless gateway and classic-agent startup compatibility."""

from __future__ import annotations

import io
import plistlib
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from loguru import logger
from rich.console import Console
from typer.testing import CliRunner

from nanobot.cli import log_control
from nanobot.cli.gateway import create_gateway_app
from nanobot.config.schema import Config
from nanobot.gateway import GatewayRuntimePaths, GatewayStartOptions, GatewayStatus, RuntimeResult
from nanobot.gateway.runtime import GatewayRuntime, build_gateway_command
from nanobot.gateway.service import (
    GatewayServiceInstaller,
    GatewayServiceOptions,
    GatewayServiceResult,
)
from nanobot.process_runtime import ProcessStartOptions


def test_agent_and_gateway_help_accept_compatibility_log_file() -> None:
    """The entry point parses both restored flags before any runtime admission."""
    root = Path(__file__).resolve().parents[2]

    for command in ("agent", "gateway"):
        result = subprocess.run(
            [sys.executable, "-m", "nanobot", command, "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "--log-file" in result.stdout


def test_file_only_agent_runtime_logging_does_not_write_to_stderr(tmp_path: Path) -> None:
    """A file-only classic-agent run keeps runtime output away from the terminal."""
    root = Path(__file__).resolve().parents[2]
    log_path = tmp_path / "agent.log"
    script = "\n".join(
        (
            "from loguru import logger",
            "from nanobot.cli.log_control import configure_agent_runtime_logs",
            (
                "configure_agent_runtime_logs(show_console=False, "
                f"log_file={str(log_path)!r})"
            ),
            "logger.info('file-only-runtime-log')",
            "logger.complete()",
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "file-only-runtime-log" not in result.stderr
    assert "file-only-runtime-log" in log_path.read_text(encoding="utf-8")


def test_cli_log_reconfiguration_replaces_only_owned_console_sink(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Agent and gateway logging leave application-owned Loguru sinks intact."""
    foreign_output = io.StringIO()
    cli_output = io.StringIO()
    foreign_sink_id = logger.add(foreign_output, format="{message}")
    cli_sink_id = logger.add(cli_output, format="{message}")
    original_console_sink_id = log_control._cli_console_sink_id
    original_file_sink_id = log_control._compatibility_file_sink_id
    log_control.register_cli_console_sink(cli_sink_id)
    log_control._compatibility_file_sink_id = None

    app = typer.Typer()
    app.add_typer(
        create_gateway_app(
            console=Console(file=io.StringIO()),
            log_handler_id=cli_sink_id,
            load_runtime_config=lambda _config, _workspace: Config(),
            run_gateway=lambda _config, **_kwargs: None,
        ),
        name="gateway",
    )
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)

    try:
        log_control.configure_agent_runtime_logs(
            show_console=False,
            log_file=str(tmp_path / "agent.log"),
        )

        first_result = CliRunner().invoke(app, ["gateway", "--verbose"])
        assert first_result.exit_code == 0, first_result.output
        first_gateway_sink_id = log_control._cli_console_sink_id

        second_result = CliRunner().invoke(app, ["gateway", "--verbose"])
        assert second_result.exit_code == 0, second_result.output
        assert log_control._cli_console_sink_id != first_gateway_sink_id
        assert first_gateway_sink_id is not None
        try:
            logger.remove(first_gateway_sink_id)
        except ValueError:
            pass
        else:
            raise AssertionError("the replaced CLI console sink was still registered")

        logger.info("foreign-sink-survives")
        assert "foreign-sink-survives" in foreign_output.getvalue()
        assert cli_output.getvalue() == ""
    finally:
        for sink_id in (
            log_control._cli_console_sink_id,
            log_control._compatibility_file_sink_id,
            foreign_sink_id,
        ):
            if sink_id is None:
                continue
            try:
                logger.remove(sink_id)
            except ValueError:
                pass
        log_control._cli_console_sink_id = original_console_sink_id
        log_control._compatibility_file_sink_id = original_file_sink_id


def test_compatibility_log_file_replaces_its_owned_sink(tmp_path: Path) -> None:
    """Selecting a new compatibility file does not retain the prior file sink."""
    foreign_output = io.StringIO()
    foreign_sink_id = logger.add(foreign_output, format="{message}")
    original_file_sink_id = log_control._compatibility_file_sink_id
    log_control._compatibility_file_sink_id = None
    first_path = tmp_path / "first.log"
    second_path = tmp_path / "second.log"

    try:
        first_sink_id = log_control.add_compatibility_log_file(str(first_path))
        logger.info("first-file-record")
        logger.complete()

        second_sink_id = log_control.add_compatibility_log_file(str(second_path))
        logger.info("second-file-record")
        logger.complete()

        assert first_sink_id != second_sink_id
        assert "first-file-record" in first_path.read_text(encoding="utf-8")
        assert "second-file-record" not in first_path.read_text(encoding="utf-8")
        assert "second-file-record" in second_path.read_text(encoding="utf-8")
        assert "first-file-record" in foreign_output.getvalue()
        assert "second-file-record" in foreign_output.getvalue()
        try:
            logger.remove(first_sink_id)
        except ValueError:
            pass
        else:
            raise AssertionError("the replaced compatibility file sink was still registered")
    finally:
        for sink_id in (log_control._compatibility_file_sink_id, foreign_sink_id):
            if sink_id is None:
                continue
            try:
                logger.remove(sink_id)
            except ValueError:
                pass
        log_control._compatibility_file_sink_id = original_file_sink_id


def test_agent_log_configuration_without_a_file_clears_owned_file_sink(tmp_path: Path) -> None:
    """A later agent invocation cannot keep logging to a previous file."""
    foreign_output = io.StringIO()
    foreign_sink_id = logger.add(foreign_output, format="{message}")
    original_console_sink_id = log_control._cli_console_sink_id
    original_file_sink_id = log_control._compatibility_file_sink_id
    log_control._cli_console_sink_id = None
    log_control._compatibility_file_sink_id = None
    log_path = tmp_path / "agent.log"

    try:
        log_control.add_compatibility_log_file(str(log_path))
        logger.info("file-a-record")
        logger.complete()

        log_control.configure_agent_runtime_logs(show_console=True, log_file=None)
        assert log_control._compatibility_file_sink_id is None
        logger.info("after-file-a")
        logger.complete()

        contents = log_path.read_text(encoding="utf-8")
        assert "file-a-record" in contents
        assert "after-file-a" not in contents
        assert "after-file-a" in foreign_output.getvalue()
    finally:
        for sink_id in (
            log_control._cli_console_sink_id,
            log_control._compatibility_file_sink_id,
            foreign_sink_id,
        ):
            if sink_id is None:
                continue
            try:
                logger.remove(sink_id)
            except ValueError:
                pass
        log_control._cli_console_sink_id = original_console_sink_id
        log_control._compatibility_file_sink_id = original_file_sink_id


@pytest.mark.parametrize("fails", [False, True])
def test_foreground_gateway_releases_compatibility_file_sink_on_completion(
    monkeypatch,
    tmp_path: Path,
    fails: bool,
) -> None:
    """A foreground gateway releases its selected file after any exit path."""
    app = typer.Typer()
    log_path = tmp_path / "gateway.log"
    original_file_sink_id = log_control._compatibility_file_sink_id
    log_control._compatibility_file_sink_id = None

    def run_gateway(_config: Config, **_kwargs: object) -> None:
        logger.info("gateway-runtime-record")
        if fails:
            raise RuntimeError("gateway runtime failed")

    app.add_typer(
        create_gateway_app(
            console=Console(file=io.StringIO()),
            log_handler_id=0,
            load_runtime_config=lambda _config, _workspace: Config(),
            run_gateway=run_gateway,
        ),
        name="gateway",
    )
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)

    try:
        result = CliRunner().invoke(app, ["gateway", "--log-file", str(log_path)])

        assert result.exit_code == (1 if fails else 0)
        assert log_control._compatibility_file_sink_id is None
        logger.info("after-gateway-completion")
        logger.complete()

        contents = log_path.read_text(encoding="utf-8")
        assert "gateway-runtime-record" in contents
        assert "after-gateway-completion" not in contents
    finally:
        for sink_id in (log_control._compatibility_file_sink_id,):
            if sink_id is None:
                continue
            try:
                logger.remove(sink_id)
            except ValueError:
                pass
        log_control._compatibility_file_sink_id = original_file_sink_id


def test_gateway_log_file_is_carried_by_detached_and_service_commands(tmp_path: Path) -> None:
    """The same selected path survives child, systemd, and launchd rendering."""
    log_path = tmp_path / "gateway log.log"
    options = GatewayStartOptions(
        port=18790,
        verbose=True,
        workspace=str(tmp_path / "workspace with spaces"),
        config_path=str(tmp_path / "config.json"),
        log_file=str(log_path),
    )
    command = build_gateway_command("/venv/bin/python", options)

    assert command[-2:] == ["--log-file", str(log_path)]

    systemd = GatewayServiceInstaller(platform_name="Linux", home=tmp_path).install(
        GatewayServiceOptions(start=options, python_executable="/venv/bin/python"),
        dry_run=True,
    )
    assert systemd.content is not None
    exec_start = next(
        line.removeprefix("ExecStart=")
        for line in systemd.content.splitlines()
        if line.startswith("ExecStart=")
    )
    assert shlex.split(exec_start) == command

    launchd = GatewayServiceInstaller(platform_name="Darwin", home=tmp_path).install(
        GatewayServiceOptions(start=options, python_executable="/venv/bin/python"),
        dry_run=True,
    )
    assert launchd.content is not None
    payload = plistlib.loads(launchd.content.encode("utf-8"))
    assert payload["ProgramArguments"] == command


def test_gateway_runtime_accepts_legacy_process_start_options(tmp_path: Path) -> None:
    """Existing callers can still render gateway commands from shared options."""
    options = ProcessStartOptions(port=18790, verbose=True)
    expected = [
        "/python",
        "-m",
        "nanobot",
        "gateway",
        "--foreground",
        "--port",
        "18790",
        "--verbose",
    ]

    assert build_gateway_command("/python", options) == expected
    runtime = GatewayRuntime(
        paths=GatewayRuntimePaths.for_instance(data_dir=tmp_path),
        python_executable="/python",
    )
    assert runtime._build_child_command(options) == expected


def test_gateway_commands_propagate_log_path_without_parent_file_sink(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Detached, restart, service, and foreground forms use the selected path."""
    runner = CliRunner()
    app = typer.Typer()
    paths = GatewayRuntimePaths.for_instance(data_dir=tmp_path)
    status = GatewayStatus(
        running=True,
        pid=123,
        state_path=paths.state_path,
        log_path=paths.log_path,
        port=18790,
        reason="running",
        launch_mode="background",
    )
    seen: dict[str, object] = {}

    class Runtime:
        def __init__(self) -> None:
            self.paths = paths

        def start_background(self, options: GatewayStartOptions) -> RuntimeResult:
            seen["background"] = options
            return RuntimeResult(True, "gateway_started_background", status)

        def restart(self, options: GatewayStartOptions, *, timeout_s: int) -> RuntimeResult:
            seen["restart"] = (options, timeout_s)
            return RuntimeResult(True, "gateway_started_background", status)

        def status(self) -> GatewayStatus:
            return status

    class Service:
        def install(self, options: GatewayServiceOptions, *, dry_run: bool) -> GatewayServiceResult:
            seen["service"] = (options, dry_run)
            return GatewayServiceResult(True, "service_install_dry_run", "systemd", None)

    def run_gateway(_config: Config, **kwargs: object) -> None:
        seen["foreground"] = kwargs

    add_log_file = Mock(return_value=1)
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr(
        "nanobot.cli.log_control.add_compatibility_log_file",
        add_log_file,
    )
    app.add_typer(
        create_gateway_app(
            console=Console(),
            log_handler_id=0,
            load_runtime_config=lambda _config, _workspace: Config(),
            run_gateway=run_gateway,
            runtime_factory=lambda **_kwargs: Runtime(),
            service_factory=Service,
        ),
        name="gateway",
    )

    log_path = str(tmp_path / "selected.log")
    assert runner.invoke(app, ["gateway", "--background", "--log-file", log_path]).exit_code == 0
    assert isinstance(seen["background"], GatewayStartOptions)
    assert seen["background"].log_file == log_path
    add_log_file.assert_not_called()

    assert runner.invoke(app, ["gateway", "restart", "--log-file", log_path]).exit_code == 0
    restart_options, restart_timeout = seen["restart"]
    assert isinstance(restart_options, GatewayStartOptions)
    assert restart_options.log_file == log_path
    assert restart_timeout == 20

    assert runner.invoke(
        app,
        ["gateway", "install-service", "--dry-run", "--log-file", log_path],
    ).exit_code == 0
    service_options, dry_run = seen["service"]
    assert isinstance(service_options, GatewayServiceOptions)
    assert service_options.start.log_file == log_path
    assert dry_run is True

    assert runner.invoke(app, ["gateway", "--log-file", log_path]).exit_code == 0
    foreground = seen["foreground"]
    assert isinstance(foreground, dict)
    assert foreground["port"] is None
    assert foreground["webui_bundle_mode"] == "warn"
    assert foreground["unconfigured_provider_error"] is None
    assert foreground["log_file"] == log_path
    add_log_file.assert_called_once_with(log_path)
