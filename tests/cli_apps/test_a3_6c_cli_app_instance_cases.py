"""Requirement case U1: two explicit instances keep separate CLI-app state."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from nanobot.agent import plugins as agent_plugins
from nanobot.apps.cli.service import CliAppManager
from nanobot.config.loader import save_config, set_config_path
from nanobot.config.schema import Config
from nanobot.webui.cli_apps_api import cli_apps_action, cli_apps_payload


def _config(root: Path) -> Path:
    workspace = root / "workspace"
    workspace.mkdir()
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    path = root / "config.json"
    save_config(config, path)
    return path


def _seed_catalog(root: Path, apps: list[dict[str, object]]) -> None:
    data_dir = root / "cli-apps"
    data_dir.mkdir(parents=True, exist_ok=True)
    for source in ("harness", "public", "extensions"):
        (data_dir / f"{source}_registry_cache.json").write_text(
            json.dumps({"_cached_at": time.time(), "data": {"clis": apps}}),
            encoding="utf-8",
        )


def _snapshot(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _installed(config_path: Path) -> list[str]:
    payload = asyncio.run(cli_apps_payload(installed_only=True, config_path=config_path))
    return sorted(str(app["name"]) for app in payload["apps"])


def _app(name: str) -> dict[str, object]:
    return {
        "name": name,
        "display_name": name,
        "description": f"{name} fixture",
        "package_manager": "bundled",
        "entry_point": "sh",
    }


def test_two_explicit_instances_keep_separate_cli_app_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    set_config_path(ambient / "config.json")
    root_a = tmp_path / "rootA"
    root_b = tmp_path / "rootB"
    root_a.mkdir()
    root_b.mkdir()
    config_a = _config(root_a)
    config_b = _config(root_b)
    _seed_catalog(root_a, [_app("appA")])
    _seed_catalog(root_b, [_app("appB")])

    monkeypatch.setattr(CliAppManager, "install_skill", lambda self, app: root_a)
    monkeypatch.setattr(agent_plugins, "set_agent_plugin_enabled", lambda *args, **kwargs: None)

    root_b_before = _snapshot(root_b)
    ambient_before = _snapshot(ambient)

    cli_apps_action("install", {"name": ["appA"]}, config_path=config_a)
    assert _snapshot(root_b) == root_b_before
    assert _snapshot(ambient) == ambient_before

    cli_apps_action("install", {"name": ["appB"]}, config_path=config_b)

    assert "appA" in _installed(config_a)
    assert "appB" not in _installed(config_a)
    assert "appB" in _installed(config_b)
    assert "appA" not in _installed(config_b)

    installed_a = json.loads((root_a / "cli-apps" / "installed.json").read_text(encoding="utf-8"))
    assert "appA" in installed_a["apps"]
    installed_b = json.loads((root_b / "cli-apps" / "installed.json").read_text(encoding="utf-8"))
    assert "appA" not in installed_b["apps"]
