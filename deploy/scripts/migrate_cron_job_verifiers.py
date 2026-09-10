#!/usr/bin/env python3
"""Idempotently attach AUTHORITATIVE external verifiers to the client cron jobs.

Version-controlled migration for the runtime cron store
(``/opt/marketing-agent/.nanobot/workspace/cron/jobs.json``).

It adds/normalizes ONLY ``payload.verifiers`` for the three client-facing jobs:

  * Daily trend research
  * Weekly performance review
  * Daily Content Idea (telegram-shape verifier only; see PILLAR NOTE below)

It never alters schedules, ``to`` (client group), ``deliver``,
``skip_verification``, ``alert_channel``/``alert_to``, the Meta analytics job, or
the disabled legacy weekly job.

Verifier contract (see ``nanobot/cron/delivery.py``): each verifier is executed
directly (no shell) and must exit 0 AND print a JSON object with
``"verified": true`` or ``"ok": true``.  Anything else is non-PASS and blocks
client delivery.  Templates: ``{date}`` (run date, job tz), ``{repo_root}``
(verifier cwd) and ``{response_file}`` (this run's exact client response).

PILLAR NOTE: ``verify_idea_pillar.py`` requires ``--expected-pillar`` (the
rotation-assigned pillar).  That assignment is prompt-driven and has no
deterministic source until the delivery-status/rotation resolver lands, so it is
deliberately NOT wired here.  Until then the Daily Content Idea job keeps its
existing internal-verifier gate for pillar assignment while the telegram-shape
verifier becomes authoritative for the client-format contract.

Dry-run by default.  Pass ``--apply`` to write (a ``.bak`` backup is made).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_STORE = Path("/opt/marketing-agent/.nanobot/workspace/cron/jobs.json")
REPO_ROOT = "/opt/marketing-agent/client-marketing-assistance"
PYTHON = f"{REPO_ROOT}/.venv/bin/python"

_T = "{repo_root}/agent-workspace/outputs/research/{date}-daily-trend-research.md"
_TC = "{repo_root}/agent-workspace/runs/{date}-daily-trend-research/collection.json"
_WR = "{repo_root}/agent-workspace/outputs/reports/{date}-weekly-performance-review.md"
_WP = "{repo_root}/agent-workspace/state/weekly-performance-review-input.json"

VERIFIERS_BY_JOB: dict[str, list[dict]] = {
    "Daily trend research": [
        {
            "name": "trend_report",
            "argv": [
                PYTHON, "{repo_root}/tools/verify_trend_report.py",
                "--report", _T, "--collection", _TC, "--json",
            ],
            "cwd": REPO_ROOT,
            "timeout": 180,
        },
        {
            "name": "trend_telegram",
            "argv": [
                PYTHON, "{repo_root}/tools/verify_telegram_output.py",
                "--kind", "trend", "--input", "{response_file}",
                "--research-status", "{status}", "--json",
            ],
            "cwd": REPO_ROOT,
            "timeout": 120,
            # The internal research status is declared in the run's internal report.
            "status_file": _T,
            "status_regex": r"(?im)^\*\*Internal status:\*\*\s*([A-Za-z_]+)",
        },
    ],
    "Weekly performance review": [
        {
            "name": "weekly_report",
            "argv": [
                PYTHON, "{repo_root}/tools/verify_weekly_review.py",
                "--report", _WR, "--packet", _WP, "--json",
            ],
            "cwd": REPO_ROOT,
            "timeout": 180,
        },
        {
            "name": "weekly_telegram",
            "argv": [
                PYTHON, "{repo_root}/tools/verify_weekly_review.py",
                "--report", "{response_file}", "--telegram", "--json",
            ],
            "cwd": REPO_ROOT,
            "timeout": 120,
        },
    ],
    "Daily Content Idea": [
        {
            "name": "idea_telegram",
            "argv": [
                PYTHON, "{repo_root}/tools/verify_telegram_output.py",
                "--kind", "idea", "--input", "{response_file}", "--json",
            ],
            "cwd": REPO_ROOT,
            "timeout": 120,
        },
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--apply", action="store_true",
                        help="write changes (default is dry-run)")
    parser.add_argument("--json", action="store_true")
    return parser


def migrate(data: dict, *, apply: bool) -> list[dict]:
    changes: list[dict] = []
    for job in data.get("jobs", []):
        name = job.get("name")
        if name not in VERIFIERS_BY_JOB:
            continue
        payload = job.setdefault("payload", {})
        before = payload.get("verifiers")
        after = VERIFIERS_BY_JOB[name]
        if before != after:
            changes.append({
                "id": job.get("id"),
                "name": name,
                "verifier_names_before": [v.get("name") for v in (before or [])],
                "verifier_names_after": [v.get("name") for v in after],
            })
            if apply:
                payload["verifiers"] = after
    return changes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = json.loads(args.store.read_text(encoding="utf-8"))
    changes = migrate(data, apply=args.apply)

    if args.apply and changes:
        backup = args.store.with_suffix(args.store.suffix + ".bak")
        shutil.copy2(args.store, backup)
        args.store.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if args.json:
        print(json.dumps({"applied": args.apply, "changes": changes}, indent=2, ensure_ascii=False))
    else:
        mode = "APPLIED" if args.apply else "DRY-RUN"
        print(f"[{mode}] {len(changes)} job(s) would change")
        for c in changes:
            print(f"  - {c['name']}: {c['verifier_names_before']} -> {c['verifier_names_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())