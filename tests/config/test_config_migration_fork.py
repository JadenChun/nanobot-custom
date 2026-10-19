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

    assert config.agents.defaults.context_window_tokens == 120000
    assert config.agents.defaults.max_tokens == 1234
    assert not hasattr(config.agents.defaults, "memory_window")


def test_save_config_writes_upstream_token_fields_losslessly(tmp_path) -> None:
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

    assert defaults["maxTokens"] == 2222
    assert defaults["contextWindowTokens"] == 100000
    assert "maxInputTokens" not in defaults
    assert "memoryWindow" not in defaults
    reloaded = load_config(config_path)
    assert reloaded.agents.defaults.max_tokens == 2222
    assert reloaded.agents.defaults.context_window_tokens == 100000


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
    assert config.agents.defaults.context_window_tokens == 77777
    assert config.agents.defaults.max_tokens == 3333


def test_onboard_does_not_crash_with_legacy_memory_window(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.cli.commands._onboard_plugins", lambda _path: None)
    # Temporary-config integration only; production activation stays guarded.
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
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

    from typer.testing import CliRunner

    from nanobot.cli.commands import app
    runner = CliRunner()
    result = runner.invoke(app, ["onboard", "--workspace", str(workspace)], input="n\n")

    assert result.exit_code == 0
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["agents"]["defaults"]["maxTokens"] == 3333
    assert saved["agents"]["defaults"]["workspace"] == str(workspace)


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
                                "targetRepoPaths": {
                                    "pota_website": "~/context/website",
                                },
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
    assert context_repos[1].target_repo_paths == {"pota_website": "~/context/website"}


def test_save_config_omits_empty_legacy_context_paths(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "contextPaths": [],
                        "contextRepos": ["~/context/managed"],
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
    assert "contextPaths" not in defaults
    assert defaults["contextRepos"] == ["~/context/managed"]


def test_onboard_refresh_backfills_missing_channel_fields(tmp_path, monkeypatch) -> None:
    # Temporary-config integration only; production activation stays guarded.
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    from nanobot.channels.plugin import load_channel_package

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
    monkeypatch.setattr(
        "nanobot.channels.registry.discover_plugins",
        lambda: {"qq": load_channel_package("qq")},
    )

    from typer.testing import CliRunner

    from nanobot.cli.commands import app
    runner = CliRunner()
    result = runner.invoke(app, ["onboard", "--workspace", str(workspace)], input="n\n")

    assert result.exit_code == 0
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["channels"]["qq"]["msgFormat"] == "plain"
    assert saved["agents"]["defaults"]["workspace"] == str(workspace)


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


def test_checked_wire_validators_preserve_objects_and_non_string_keys() -> None:
    from types import MappingProxyType

    import pytest

    from nanobot.config.schema import (
        AgentDefaults,
        ChannelsConfig,
        IdleCompactionConfig,
        ToolsConfig,
    )

    marker = object()
    for validator in (
        ChannelsConfig.preserve_fork_activation_defaults,
        IdleCompactionConfig.reject_alias_duplicates,
        AgentDefaults.reject_idle_compaction_alias_duplicates,
        ToolsConfig._select_image_configuration,
    ):
        assert validator(marker) is marker

    wire = {marker: marker, "websocket": {marker: marker, "enabled": True}, "sendProgress": []}
    copied = ChannelsConfig.preserve_fork_activation_defaults(wire)
    assert copied is not wire
    assert copied[marker] is marker
    assert copied["websocket"] == {marker: marker, "enabled": True}
    assert copied["taskUpdateMode"] == "result"
    assert "taskUpdateMode" not in wire
    explicit = {"task_update_mode": "verbose", "sendProgress": False}
    assert "taskUpdateMode" not in ChannelsConfig.preserve_fork_activation_defaults(explicit)

    mapping = MappingProxyType({marker: marker})
    idle = IdleCompactionConfig.reject_alias_duplicates(mapping)
    assert idle == {marker: marker}
    assert idle is not mapping
    assert AgentDefaults.reject_idle_compaction_alias_duplicates(mapping) is mapping
    with pytest.raises(ValueError, match="Specify only one"):
        IdleCompactionConfig.reject_alias_duplicates(
            MappingProxyType({"idle_ttl_seconds": 0, "idleTtlSeconds": 0})
        )
    with pytest.raises(ValueError, match="Specify only one"):
        AgentDefaults.reject_idle_compaction_alias_duplicates(
            MappingProxyType({"idleCompaction": {}, "idle_compaction": {}})
        )

    image_wire = {marker: marker}
    selected = ToolsConfig._select_image_configuration(image_wire)
    assert selected == {marker: marker, "image_generation": {"configuration_source": "legacy"}}
    assert image_wire == {marker: marker}
    for alias in ("image_generation", "imageGeneration"):
        configured = {alias: {"configuration_source": "explicit"}}
        assert ToolsConfig._select_image_configuration(configured) is configured


def test_artifact_root_checked_edge_keeps_direct_call_errors_and_lexical_path(tmp_path) -> None:
    import inspect

    import pytest

    from nanobot.config.schema import ArtifactsConfig

    assert list(inspect.signature(ArtifactsConfig._validate_root).parameters) == ["value"]
    assert ArtifactsConfig._validate_root(None) is None
    for invalid in (object(), 42, "", "relative", "/"):
        with pytest.raises(ValueError, match="absolute"):
            ArtifactsConfig._validate_root(invalid)
    lexical = str(tmp_path / "missing" / ".." / "artifacts")
    assert ArtifactsConfig._validate_root(lexical) == str(tmp_path / "artifacts")
