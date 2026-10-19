import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nanobot.cli import entry
from nanobot.cli.commands import app


@pytest.mark.parametrize("args", [["--workspace", "./project"], ["-mhello"], ["-m", "hello"]])
def test_agent_options_require_an_explicit_agent_subcommand(args) -> None:
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_root_metadata_and_subcommands_keep_the_root_cli() -> None:
    for args in (
        ["--help"],
        ["--version"],
        ["gateway", "--help"],
        ["webui", "--help"],
    ):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output


def test_root_shell_completion_keeps_root_subcommands() -> None:
    env = os.environ.copy()
    env.update(
        {
            "_NANOBOT_COMPLETE": "complete_bash",
            "COMP_WORDS": "nanobot ",
            "COMP_CWORD": "1",
        }
    )
    script = (
        "import sys; "
        "from nanobot.cli.entry import main; "
        "sys.argv = ['nanobot']; "
        "main()"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert {"agent", "gateway", "webui"} <= set(result.stdout.splitlines())
    assert "not supported" not in result.stderr


@pytest.mark.parametrize("args", [[], ["agent", "-m", "hello"], ["gateway"], ["webui"]])
def test_entry_dispatches_unchanged_arguments_after_console_setup(monkeypatch, args) -> None:
    from nanobot.cli import commands

    calls = []
    monkeypatch.setattr(entry.sys, "argv", ["nanobot", *args])
    monkeypatch.setattr(entry, "_configure_windows_console", lambda: calls.append("console"))
    monkeypatch.setattr(commands, "app", lambda: calls.append(list(entry.sys.argv)))
    entry.main()
    assert calls == ["console", ["nanobot", *args]]


def test_bare_invocation_does_not_discover_desktop_or_start_runtime(monkeypatch) -> None:
    from nanobot.cli import desktop_target

    monkeypatch.setattr(entry.sys, "argv", ["nanobot"])
    monkeypatch.setattr(desktop_target, "dispatch_bare_desktop_target",
                        lambda *_args: pytest.fail("desktop discovery must not start"))
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready",
                        lambda: pytest.fail("bare help must not request runtime admission"))
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 0


@pytest.mark.parametrize("args", [
    [], ["--session", "websocket:chat"], ["--theme=light"], ["--classic"],
    ["--no-tui"], ["-m", "hello"], ["-mhello"], ["--message=hello"],
])
def test_explicit_agent_options_reach_real_session_preflight(tmp_path, args) -> None:
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "agents": {"defaults": {
            "workspace": str(workspace),
            "model": "ollama/llama3.2",
            "provider": "ollama",
            "sessionTtlMinutes": 15,
        }},
        "providers": {"ollama": {"apiBase": "http://localhost:11434/v1"}},
    }), encoding="utf-8")
    config_bytes = config_path.read_bytes()

    result = CliRunner().invoke(app, ["agent", *args, "--config", str(config_path)])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert result.exception.code == 1
    assert "idle" in result.output.lower()
    assert "No such option" not in result.output
    assert config_path.read_bytes() == config_bytes
    assert not workspace.exists()
