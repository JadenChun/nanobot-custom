"""Import a consistent legacy cron copy without touching its source or running jobs."""

import argparse
from pathlib import Path

from nanobot.cron.import_copy import import_legacy_cron_copy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--timezone", help="Explicit timezone for old cron schedules without one")
    args = parser.parse_args()
    report = import_legacy_cron_copy(args.source, args.destination, timezone=args.timezone)
    # Avoid printing private prompts, destinations, results or original state.
    print(f"Imported {len(report.definitions)} jobs. Receipt: {args.destination / 'migration-receipt.json'}")
    if report.pending_retries_require_review:
        print(f"{len(report.pending_retries_require_review)} jobs retain legacy retry state for review; no retry was activated.")


if __name__ == "__main__":
    main()
