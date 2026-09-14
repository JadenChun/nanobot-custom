"""Runtime log visibility controls shared by CLI commands."""

import sys
from typing import Any

from loguru import logger

__all__ = [
    "_set_nanobot_logs",
    "add_compatibility_log_file",
    "clear_compatibility_log_file",
    "configure_agent_runtime_logs",
    "register_cli_console_sink",
    "replace_cli_console_sink",
]

_COMPATIBILITY_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)

_cli_console_sink_id: int | None = None
_compatibility_file_sink_id: int | None = None
_DEFAULT_CONSOLE_SINK_ID = 0


def _set_nanobot_logs(enabled: bool) -> None:
    if enabled:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")


def _remove_sink(sink_id: int | None) -> None:
    """Remove a known CLI sink that may already have been replaced."""
    if sink_id is None:
        return
    try:
        logger.remove(sink_id)
    except ValueError:
        pass


def register_cli_console_sink(sink_id: int) -> None:
    """Record the console sink created by the CLI entry point."""
    global _cli_console_sink_id
    _cli_console_sink_id = sink_id


def replace_cli_console_sink(sink: Any, **kwargs: Any) -> int:
    """Replace only the console sink owned by the CLI."""
    global _cli_console_sink_id
    _remove_sink(
        _cli_console_sink_id
        if _cli_console_sink_id is not None
        else _DEFAULT_CONSOLE_SINK_ID
    )
    _cli_console_sink_id = logger.add(sink, **kwargs)
    return _cli_console_sink_id


def _remove_cli_console_sink() -> None:
    global _cli_console_sink_id
    _remove_sink(
        _cli_console_sink_id
        if _cli_console_sink_id is not None
        else _DEFAULT_CONSOLE_SINK_ID
    )
    _cli_console_sink_id = None


def add_compatibility_log_file(path: str) -> int:
    """Keep the foreground fork's file logging/rotation contract."""
    global _compatibility_file_sink_id
    logger.enable("nanobot")
    _remove_sink(_compatibility_file_sink_id)
    _compatibility_file_sink_id = logger.add(
        path,
        rotation="20 MB",
        retention="7 days",
        compression="gz",
        format=_COMPATIBILITY_LOG_FORMAT,
        encoding="utf-8",
        enqueue=True,
    )
    return _compatibility_file_sink_id


def clear_compatibility_log_file() -> None:
    """Remove the compatibility file sink owned by the current CLI invocation."""
    global _compatibility_file_sink_id
    _remove_sink(_compatibility_file_sink_id)
    _compatibility_file_sink_id = None


def configure_agent_runtime_logs(*, show_console: bool, log_file: str | None) -> None:
    """Configure classic-agent runtime logs without corrupting terminal input."""
    if log_file is None:
        clear_compatibility_log_file()
    if not show_console and not log_file:
        _set_nanobot_logs(False)
        return

    # The regular CLI sink writes directly to stderr. Remove only that sink
    # for a file-only agent so prompt_toolkit's terminal remains reserved for
    # chat while application-owned sinks keep receiving records.
    if show_console:
        replace_cli_console_sink(
            lambda message: sys.stderr.write(str(message)),
            format=_COMPATIBILITY_LOG_FORMAT,
        )
    else:
        _remove_cli_console_sink()
    if log_file:
        add_compatibility_log_file(log_file)
    else:
        logger.enable("nanobot")
