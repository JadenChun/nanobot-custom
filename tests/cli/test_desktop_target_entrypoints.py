"""Classic launchers do not implicitly attach to an optional desktop runtime."""

import pytest
from typer.testing import CliRunner

from nanobot.cli import commands, desktop_target, entry


@pytest.fixture
def no_desktop_discovery(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("classic entrypoint must not discover, prompt, or attach to desktop")

    monkeypatch.setattr(desktop_target, "_interactive_shell", lambda: True)
    monkeypatch.setattr(desktop_target, "discover_desktop_target", forbidden)
    monkeypatch.setattr(desktop_target, "dispatch_bare_desktop_target", forbidden)
    monkeypatch.setattr(desktop_target, "_choose_target", forbidden)


@pytest.mark.usefixtures("no_desktop_discovery")
@pytest.mark.parametrize("launcher", ["modern", "retained"])
@pytest.mark.parametrize("args", [[], ["--help"], ["--version"], ["agent", "--help"],
                                  ["webui", "--help"], ["gateway", "--help"]])
def test_help_and_metadata_never_start_runtime(monkeypatch, args, launcher):
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready",
                        lambda: pytest.fail("help must not request runtime admission"))
    if launcher == "retained":
        result = CliRunner().invoke(commands.app, args)
        assert result.exit_code == 0, result.output
    else:
        monkeypatch.setattr(entry.sys, "argv", ["nanobot", *args])
        with pytest.raises(SystemExit) as exc:
            entry.main()
        assert exc.value.code == 0


@pytest.mark.usefixtures("no_desktop_discovery")
@pytest.mark.parametrize("launcher", ["modern", "retained"])
@pytest.mark.parametrize("args", [["webui"], ["webui", "--no-open"],
                                  ["webui", "--port", "8765"], ["agent"],
                                  ["agent", "--classic"], ["agent", "-m", "hello"]])
def test_execution_loads_config_before_desktop_discovery(
    monkeypatch, tmp_path, capsys, args, launcher,
):
    config_path = tmp_path / "invalid.json"
    config_path.write_text("{broken", encoding="utf-8")
    original_config = config_path.read_bytes()
    invocation_args = [*args, "--config", str(config_path)]

    if launcher == "retained":
        result = CliRunner().invoke(commands.app, invocation_args)
        assert result.exit_code == 1, result.output
        output = result.output
    else:
        monkeypatch.setattr(entry.sys, "argv", ["nanobot", *invocation_args])
        with pytest.raises(SystemExit) as exc:
            entry.main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err

    assert "Invalid configuration" in output
    assert "Traceback" not in output
    assert config_path.read_bytes() == original_config
