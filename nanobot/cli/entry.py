"""Console entrypoint for the classic, explicitly selected CLI commands."""

from __future__ import annotations

import os
import sys
from contextlib import suppress

from nanobot.cli.process_identity import set_cli_process_identity as set_cli_process_identity


def _configure_windows_console() -> None:
    if sys.platform != "win32" or sys.stdout.encoding == "utf-8":
        return
    os.environ["PYTHONIOENCODING"] = "utf-8"
    with suppress(Exception):
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    """Preserve classic fork dispatch without desktop activation."""
    _configure_windows_console()
    from nanobot.cli.commands import app
    app()
