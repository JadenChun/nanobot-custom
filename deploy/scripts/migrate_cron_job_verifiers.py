#!/usr/bin/env python3
"""Idempotently attach AUTHORITATIVE external verifiers to the client cron jobs.

Version-controlled migration for the runtime cron store
(``/opt/marketing-agent/.nanobot/workspace/cron/jobs.json``).

It adds/normalizes ``payload.verifiers`` for the three client-facing jobs, plus
the optional ``payload.post_delivery_command`` for the Daily Content Idea:

  * Daily trend research      -> trend_report, trend_telegram
  * Weekly performance review -> weekly_report, weekly_telegram
  * Daily Content Idea        -> daily_content_idea (composite) + post-delivery hook

It never alters schedules, ``to`` (client group), ``deliver``,
``skip_verification``, ``alert_channel``/``alert_to``, the Meta analytics job, or
the disabled legacy weekly job.

Verifier contract (see ``nanobot/cron/delivery.py``): each verifier is executed
directly (no shell) and must exit 0 AND print a JSON object with
``"verified": true`` or ``"ok": true``.  Anything else is non-PASS and blocks
client delivery.  Templates: ``{date}`` (run date, job tz), ``{repo_root}``
(verifier cwd) and ``{response_file}`` (this run's exact client response).

COMPOSITE DAILY IDEA VERIFIER: ``tools/verify_daily_content_idea.py`` resolves
the rotation-assigned pillar itself from the delivered-only history (marketing
context - a normal Rev3 idea that is BOTH delivered AND verification-pass), then
checks the pillar fit and the Telegram client contract.  It records the verdict
as ``verification_status`` on the EXACT idea record (never inferred later), and
stamps the run-state handshake carrying that exact ``idea_id``.  The pillar is
NEVER computed by Nanobot and never injected as a constant.

POST-DELIVERY HOOK: ``tools/mark_delivery_status.py`` maps the client-delivery
ACK outcome (``{ack_status}``) to the marketing record's TRANSPORT
``delivery_status``, addressing the exact record via the run-state handshake
(falling back to an unambiguous date match, refusing on ambiguity).  It carries
the marketing business logic; Nanobot only invokes it.  A failed recording after
a successful delivery raises one owner alert and never a resend.

GENERATION PILLAR: the Daily Content Idea prompt gets an idempotent, sentinel-
delimited instruction to source its pillar from ``tools/next_daily_pillar.py``
(the delivered-only resolver) instead of inferring it from idea history, and to
record through ``tools/record_daily_idea.py`` so the record id is pinned to the
scheduled date (making date -> record exactly 1:1 and same-day reruns
idempotent).

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
_ID = "{repo_root}/agent-workspace/outputs/ideas/{date}-daily-content-idea.md"

#: Optional marketing-side post-delivery bookkeeping.  Nanobot reports the
#: client-delivery ACK outcome; marketing maps it to its own delivery status.
POST_DELIVERY_BY_JOB: dict[str, list[str]] = {
    "Daily Content Idea": [
        PYTHON, "{repo_root}/tools/mark_delivery_status.py",
        "--date", "{date}", "--ack", "{ack_status}", "--json",
    ],
}

#: Sentinel for idempotent insertion of the assigned-pillar instruction.
PILLAR_SENTINEL = "ASSIGNED PILLAR (soft default)"
PILLAR_INSTRUCTION = f"""{PILLAR_SENTINEL}
FIRST, before any ideation, run exactly:
python3 tools/next_daily_pillar.py --json
Use the returned assigned_pillar as this run's DEFAULT pillar. Only a successfully verified and delivered idea consumes a rotation slot. The assigned pillar is a strong default, NOT an absolute rule: you MAY override it when today's trend research contains a genuinely strong opportunity that does not fit the assigned pillar. An override requires ALL of: the trend is the idea's primary evidence (source_mode trend_led); the chosen pillar is one of the five rotation pillars and genuinely fits the trend; the internal note records BOTH an "Assigned pillar:" line and a "Pillar override:" line naming the chosen pillar and the reason; and the client-facing title declares the chosen pillar. Do not override for a weak or marginal fit. The rotation continues from the pillar actually delivered, so a skipped pillar returns on the next cycle.
END ASSIGNED PILLAR

"""

#: Jobs whose generation prompt must source the pillar from the resolver.
PILLAR_INSTRUCTION_BY_JOB = {"Daily Content Idea"}

#: The Daily Idea must be recorded through the deterministic-id wrapper so a
#: same-day retry/rerun updates one record instead of creating a second one.
RECORD_TOOL_OLD = "tools/record_idea.py"
RECORD_TOOL_NEW = "tools/record_daily_idea.py"

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
            # The internal research status is the validated, structured field
            # of the run's collection bundle. Resolving it here (not from a
            # free-form report header the agent may omit or reformat) keeps the
            # framework from crashing on an unresolvable status.
            "status_file": _TC,
            "status_regex": r'"research_status"\s*:\s*"([A-Za-z_]+)"',
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
            # Composite marketing verifier: resolves the rotation-assigned pillar
            # from the delivered-only history, then checks BOTH the pillar fit
            # (internal report) and the Telegram client contract (this run's
            # response).  Nanobot never computes --expected-pillar itself.
            "name": "daily_content_idea",
            "argv": [
                PYTHON, "{repo_root}/tools/verify_daily_content_idea.py",
                "--input", "{response_file}",
                "--idea-report", _ID,
                "--date", "{date}",
                "--json",
            ],
            "cwd": REPO_ROOT,
            "timeout": 180,
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
        change: dict = {"id": job.get("id"), "name": name}

        before = payload.get("verifiers")
        after = VERIFIERS_BY_JOB[name]
        if before != after:
            change["verifier_names_before"] = [v.get("name") for v in (before or [])]
            change["verifier_names_after"] = [v.get("name") for v in after]
            if apply:
                payload["verifiers"] = after

        if name in POST_DELIVERY_BY_JOB:
            before_cmd = payload.get("post_delivery_command")
            after_cmd = POST_DELIVERY_BY_JOB[name]
            if before_cmd != after_cmd:
                change["post_delivery_command_before"] = before_cmd
                change["post_delivery_command_after"] = after_cmd
                if apply:
                    payload["post_delivery_command"] = after_cmd

        if name in PILLAR_INSTRUCTION_BY_JOB:
            message = str(payload.get("message") or "")
            updated = message

            # Pin the record id to the scheduled date so a same-day retry/rerun
            # UPDATES the same record instead of creating a second one, which is
            # what makes an exact date -> record correlation safe.
            if RECORD_TOOL_OLD in updated and RECORD_TOOL_NEW not in updated:
                updated = updated.replace(RECORD_TOOL_OLD, RECORD_TOOL_NEW)
                change["record_tool"] = f"{RECORD_TOOL_OLD} -> {RECORD_TOOL_NEW}"

            if PILLAR_SENTINEL not in updated:
                anchor = "PILLAR ROTATION (soft default):"
                if anchor in updated:
                    updated = updated.replace(anchor, PILLAR_INSTRUCTION + anchor, 1)
                else:
                    updated = PILLAR_INSTRUCTION + updated
                change["pillar_instruction"] = "insert"

            if apply and updated != message:
                payload["message"] = updated

        if len(change) > 2:
            changes.append(change)
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
            print(f"  - {c['name']}")
            if "verifier_names_after" in c:
                print(f"      verifiers: {c.get('verifier_names_before')} -> {c['verifier_names_after']}")
            if "post_delivery_command_after" in c:
                print(f"      post_delivery_command -> {c['post_delivery_command_after']}")
            if "record_tool" in c:
                print(f"      record_tool: {c['record_tool']}")
            if "pillar_instruction" in c:
                print(f"      pillar_instruction: {c['pillar_instruction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())