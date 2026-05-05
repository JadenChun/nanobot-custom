import json

from nanobot.config.loader import load_config, save_config


def test_load_config_migrates_legacy_int_max_tokens(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "maxTokens": 1234,
                        "memoryWindow": 42,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.agents.defaults.max_tokens.input == 120000
    assert config.agents.defaults.max_tokens.output == 1234
    assert not hasattr(config.agents.defaults, "memory_window")


def test_save_config_writes_only_max_tokens_object(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "maxTokens": 2222,
                        "maxInputTokens": 100000,
                        "contextWindowTokens": 65536,
                        "memoryWindow": 30,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    save_config(config, config_path)
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = saved["agents"]["defaults"]

    assert defaults["maxTokens"] == {"input": 100000, "output": 2222}
    assert "contextWindowTokens" not in defaults
    assert "maxInputTokens" not in defaults
    assert "memoryWindow" not in defaults


def test_load_config_maps_context_window_tokens_to_max_tokens_input(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "maxTokens": {"output": 3333},
                        "contextWindowTokens": 77777,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.agents.defaults.max_tokens.input == 77777
    assert config.agents.defaults.max_tokens.output == 3333


def test_onboard_does_not_crash_with_legacy_memory_window(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    workspace = tmp_path / "workspace"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "maxTokens": 3333,
                        "memoryWindow": 50,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: config_path)
    monkeypatch.setattr("nanobot.cli.commands.get_workspace_path", lambda _workspace=None: workspace)

    from typer.testing import CliRunner
    from nanobot.cli.commands import app
    runner = CliRunner()
    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0


def test_save_config_preserves_provider_api_keys_list(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "gemini": {
                        "apiKeys": ["gem-1", "gem-2"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    save_config(config, config_path)
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert saved["providers"]["gemini"]["apiKeys"] == ["gem-1", "gem-2"]


    def test_load_config_accepts_context_repos_strings_and_objects(tmp_path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "agents": {
                        "defaults": {
                            "contextRepos": [
                                "~/context/plain",
                                {
                                    "path": "~/context/managed",
                                    "readOnly": True,
                                    "autoSync": False,
                                    "credentialProfile": "pota",
                                },
                            ]
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        config = load_config(config_path)
        context_repos = config.agents.defaults.context_repos

        assert context_repos[0] == "~/context/plain"
        assert context_repos[1].path == "~/context/managed"
        assert context_repos[1].read_only is True
        assert context_repos[1].auto_sync is False
        assert context_repos[1].credential_profile == "pota"


def test_onboard_refresh_backfills_missing_channel_fields(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    config_path = tmp_path / "config.json"
    workspace = tmp_path / "workspace"
    config_path.write_text(
        json.dumps(
            {
                "channels": {
                    "qq": {
                        "enabled": False,
                        "appId": "",
                        "secret": "",
                        "allowFrom": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: config_path)
    monkeypatch.setattr("nanobot.cli.commands.get_workspace_path", lambda _workspace=None: workspace)
    monkeypatch.setattr(
        "nanobot.channels.registry.discover_all",
        lambda: {
            "qq": SimpleNamespace(
                default_config=lambda: {
                    "enabled": False,
                    "appId": "",
                    "secret": "",
                    "allowFrom": [],
                    "msgFormat": "plain",
                }
            )
        },
    )

    from typer.testing import CliRunner
    from nanobot.cli.commands import app
    runner = CliRunner()
    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["channels"]["qq"]["msgFormat"] == "plain"


def test_load_config_backfills_ltx_desktop_mcp_timeout(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tools": {
                    "mcpServers": {
                        "ltx-desktop": {
                            "url": "http://127.0.0.1:8765/mcp",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.tools.mcp_servers["ltx-desktop"].tool_timeout == 120


def test_load_config_preserves_explicit_ltx_desktop_mcp_timeout(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tools": {
                    "mcpServers": {
                        "ltx-desktop": {
                            "url": "http://127.0.0.1:8765/mcp",
                            "toolTimeout": 45,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.tools.mcp_servers["ltx-desktop"].tool_timeout == 45
