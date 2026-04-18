"""Configuration loading utilities."""

import json
from pathlib import Path

import pydantic
from loguru import logger

from nanobot.config.schema import Config

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return Path.home() / ".nanobot" / "config.json"


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
            data = _migrate_config(data)
            return Config.model_validate(data)
        except (json.JSONDecodeError, ValueError, pydantic.ValidationError) as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            logger.warning("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json", by_alias=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Migrate legacy token fields into maxTokens { input, output }.
    agents = data.get("agents", {})
    defaults = agents.get("defaults", {})
    mt = defaults.get("maxTokens")
    legacy_input = defaults.get("maxInputTokens")
    if legacy_input is None:
        legacy_input = defaults.get("contextWindowTokens")

    if isinstance(mt, int):
        defaults["maxTokens"] = {
            "input": legacy_input if isinstance(legacy_input, int) else 120000,
            "output": mt,
        }
    elif isinstance(mt, dict):
        if "input" not in mt and isinstance(legacy_input, int):
            mt["input"] = legacy_input
        mt.setdefault("input", 120000)
        mt.setdefault("output", 4096)
    elif isinstance(legacy_input, int):
        defaults["maxTokens"] = {"input": legacy_input, "output": 4096}

    defaults.pop("maxInputTokens", None)
    defaults.pop("contextWindowTokens", None)

    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")

    # Backfill tool_timeout=120 for ltx-desktop MCP server (default 30s is too short).
    mcp_servers = tools.get("mcpServers", {})
    ltx = mcp_servers.get("ltx-desktop")
    if isinstance(ltx, dict) and "toolTimeout" not in ltx:
        ltx["toolTimeout"] = 120

    return data
