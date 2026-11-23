"""Private stdin/stdout entry point for the native desktop host."""

from __future__ import annotations

import sys

from kohaku.desktop.adapter import DesktopAdapter
from kohaku.desktop.protocol import serve


def main() -> int:
    adapter = DesktopAdapter()
    try:
        serve(sys.stdin.buffer, sys.stdout.buffer, adapter.handle)
    except BrokenPipeError:
        return 0
    finally:
        try:
            adapter.close()
        except Exception:
            sys.stderr.write("Kohaku Desktop adapter: cleanup_failed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
