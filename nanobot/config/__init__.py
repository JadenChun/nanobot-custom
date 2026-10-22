"""Configuration exports without loading schemas during path-only imports."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.config.errors import ConfigIssue, ConfigLoadError
    from nanobot.config.loader import get_config_path, load_config
    from nanobot.config.paths import (
        get_cli_history_path,
        get_context_locks_dir,
        get_credentials_dir,
        get_cron_dir,
        get_data_dir,
        get_legacy_sessions_dir,
        get_logs_dir,
        get_media_dir,
        get_runtime_subdir,
        get_webui_dir,
        get_workspace_path,
        is_default_workspace,
    )
    from nanobot.config.schema import Config


_LAZY_EXPORTS = {
    "ConfigIssue": ".errors",
    "ConfigLoadError": ".errors",
    "get_config_path": ".loader",
    "load_config": ".loader",
    "get_cli_history_path": ".paths",
    "get_context_locks_dir": ".paths",
    "get_cron_dir": ".paths",
    "get_credentials_dir": ".paths",
    "get_data_dir": ".paths",
    "get_legacy_sessions_dir": ".paths",
    "get_logs_dir": ".paths",
    "get_media_dir": ".paths",
    "get_runtime_subdir": ".paths",
    "get_webui_dir": ".paths",
    "get_workspace_path": ".paths",
    "is_default_workspace": ".paths",
    "Config": ".schema",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "Config",
    "ConfigIssue",
    "ConfigLoadError",
    "load_config",
    "get_config_path",
    "get_data_dir",
    "get_runtime_subdir",
    "get_media_dir",
    "get_cron_dir",
    "get_credentials_dir",
    "get_context_locks_dir",
    "get_logs_dir",
    "get_webui_dir",
    "get_workspace_path",
    "is_default_workspace",
    "get_cli_history_path",
    "get_legacy_sessions_dir",
]
