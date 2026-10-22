"""Standalone content-free copied-root preflight; no normal CLI startup imports."""
import argparse
import json
from typing import NoReturn

from nanobot._migration_preflight import CHECKS, Check, inspect_copy, reduce_checks


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("invalid arguments")


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(add_help=False, allow_abbrev=False)
    for name in ("copy-root", "config", "workspace", "sessions-root", "runtime-data-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stopped-writer-copy", action="store_true", required=True)
    try:
        options = parser.parse_args(argv)
        report = inspect_copy(**vars(options))
    except (ValueError, TypeError, OSError):
        report = reduce_checks(tuple(Check(name, "FAIL", "INVALID") for name in CHECKS))
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["result"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
