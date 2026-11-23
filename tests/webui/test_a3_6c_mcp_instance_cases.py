"""Requirement case U2: two explicit instances keep separate MCP runtime state."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp.shared.auth import OAuthToken

from nanobot.agent.tools.mcp_oauth import (
    MCPOAuthStorage,
    mcp_oauth_has_credentials,
)
from nanobot.config.loader import load_config, set_config_path
from nanobot.webui.mcp_presets_api import mcp_presets_action

_SERVER_URL = "https://mcp.example.com/mcp"


def _config(root: Path) -> Path:
    workspace = root / "workspace"
    workspace.mkdir()
    path = root / "config.json"
    path.write_text(
        json.dumps({"agents": {"defaults": {"workspace": str(workspace)}}}),
        encoding="utf-8",
    )
    return path


def test_two_explicit_instances_keep_separate_mcp_state(tmp_path: Path) -> None:
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    set_config_path(ambient / "config.json")
    root_a = tmp_path / "rootA"
    root_b = tmp_path / "rootB"
    root_a.mkdir()
    root_b.mkdir()
    config_a = _config(root_a)
    config_b = _config(root_b)

    mcp_presets_action("enable", {"name": ["playwright"]}, config_path=config_a)
    mcp_presets_action("enable", {"name": ["playwright"]}, config_path=config_b)

    cwd_a = Path(str(load_config(config_a).tools.mcp_servers["playwright"].cwd))
    cwd_b = Path(str(load_config(config_b).tools.mcp_servers["playwright"].cwd))
    assert cwd_a == root_a / "mcp" / "playwright"
    assert cwd_b == root_b / "mcp" / "playwright"
    assert cwd_a.is_dir() and cwd_b.is_dir()
    assert not (ambient / "mcp").exists()

    storage_a = MCPOAuthStorage("shared-name", _SERVER_URL, data_dir=root_a)
    storage_b = MCPOAuthStorage("shared-name", _SERVER_URL, data_dir=root_b)
    asyncio.run(storage_a.set_tokens(OAuthToken(access_token="tokenA")))
    asyncio.run(storage_b.set_tokens(OAuthToken(access_token="tokenB")))

    credentials_a = (root_a / "auth" / "mcp.json").read_text(encoding="utf-8")
    credentials_b = (root_b / "auth" / "mcp.json").read_text(encoding="utf-8")
    assert "tokenA" in credentials_a
    assert "tokenB" not in credentials_a
    assert "tokenB" in credentials_b
    assert "tokenA" not in credentials_b
    assert not (ambient / "auth" / "mcp.json").exists()

    assert mcp_oauth_has_credentials("shared-name", _SERVER_URL, data_dir=root_a)
    assert mcp_oauth_has_credentials("shared-name", _SERVER_URL, data_dir=root_b)
    assert not mcp_oauth_has_credentials("shared-name", _SERVER_URL, data_dir=ambient)

    restarted_a = MCPOAuthStorage("shared-name", _SERVER_URL, data_dir=root_a)
    restored = asyncio.run(restarted_a.get_tokens())
    assert restored is not None and restored.access_token == "tokenA"
