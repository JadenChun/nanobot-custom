"""Runtime log visibility controls shared by CLI commands."""

from loguru import logger

__all__ = ["_set_nanobot_logs"]


def _set_nanobot_logs(enabled: bool) -> None:
    if enabled:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")


def add_compatibility_log_file(path: str) -> int:
    """Keep the foreground fork's file logging/rotation contract."""
    from loguru import logger
    logger.enable("nanobot")
    return logger.add(path, rotation="20 MB", retention="7 days", compression="gz", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}", encoding="utf-8", enqueue=True)
