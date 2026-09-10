#!/usr/bin/env python3
"""Idempotently set owner operational-alert destinations on client cron jobs.

Version-controlled migration for the runtime cron store
(``/opt/marketing-agent/.nanobot/workspace/cron/jobs.json``).  It adds/normalizes
ONLY the owner alert destination:

    alert_channel = "telegram"
    alert_to      = "6344587670"

for the three client-facing jobs (Daily trend research, Weekly performance
review, Daily Content Idea).  It never alters schedules, the client group `to`,
`deliver`, `skip_verification`, Meta analytics, or disabled jobs.

Idempotent: re-running makes no further change.  Use --dry-run to preview.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

DEFAULT_STORE = Path("/opt/marketing-agent/.nanobot/workspace/cron/jobs.json")
OWNER_ALERT_CHANNEL = "telegram"
OWNER_ALERT_TO = "6344587670"
TARGET_JOB_NAMES = {
    "Daily trend research",
    "Weekly performance review",
    "Daily Content Idea",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def migrate(data: dict, *, dry_run: bool) -> list[dict]:
    changes: list[dict] = []
    for job in data.get("jobs", []):
        if job.get("name") not in TARGET_JOB_NAMES:
            continue
        payload = job.setdefault("payload", {})
        before = (payload.get("alert_channel"), payload.get("alert_to"))
        after = (OWNER_ALERT_CHANNEL, OWNER_ALERT_TO)
        if before != after:
            changes.append({
                "id": job.get("id"),
                "name": job.get("name"),
                "alert_channel": {"from": before[0], "to": after[0]},
                "alert_to": {"from": before[1], "to": after[1]},
            })
            if not dry_run:
                payload["alert_channel"] = OWNER_ALERT_CHANNEL
                payload["alert_to"] = OWNER_ALERT_TO
    return changes


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store: Path = args.store
    if not store.exists():
        print(json.dumps({"ok": False, "error": f"store not found: {store}"}) if args.json
              else f"store not found: {store}")
        return 1
    if not args.dry_run:
        backup = store.with_suffix(store.suffix + ".bak-alert-migration")
        if not backup.exists():
            shutil.copy2(store, backup)
    data = json.loads(store.read_text(encoding="utf-8"))
    changes = migrate(data, dry_run=args.dry_run)
    if changes and not args.dry_run:
        store.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {"ok": True, "dry_run": args.dry_run, "changed": changes}
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json
          else f"{'Would change' if args.dry_run else 'Changed'} {len(changes)} job(s): "
               + ", ".join(c["name"] for c in changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())