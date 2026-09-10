"""Tests for the {status} template (verifier input resolved from a run file)."""

from __future__ import annotations

import pytest

from nanobot.cron.delivery import _resolve_status, run_external_verifiers
from nanobot.cron.types import CronVerifier

REPORT = """# Daily Trend Research

**Internal status:** limited_signal

Body text.
"""


def test_resolve_status_reads_internal_status(tmp_path):
    report = tmp_path / "2026-09-10-daily-trend-research.md"
    report.write_text(REPORT, encoding="utf-8")
    v = CronVerifier(
        name="trend_telegram",
        argv=("echo", "{status}"),
        status_file=str(report),
    )
    assert _resolve_status(v, date="2026-09-10", repo_root="") == "limited_signal"


def test_resolve_status_missing_file_is_empty(tmp_path):
    v = CronVerifier(name="v", argv=("echo",), status_file=str(tmp_path / "nope.md"))
    assert _resolve_status(v, date="2026-09-10", repo_root="") == ""


def test_resolve_status_no_status_file_is_empty():
    v = CronVerifier(name="v", argv=("echo",))
    assert _resolve_status(v, date="2026-09-10", repo_root="") == ""


def test_render_status_template_is_substituted(tmp_path):
    report = tmp_path / "r.md"
    report.write_text(REPORT, encoding="utf-8")
    v = CronVerifier(name="v", argv=("echo", "{status}"), status_file=str(report))
    out = _render_helper(v, tmp_path)
    assert out == ["echo", "limited_signal"]


def _render_helper(v, tmp_path):
    from nanobot.cron.delivery import _render_verifier_argv
    return _render_verifier_argv(
        v.argv,
        date="2026-09-10",
        repo_root=str(tmp_path),
        status=_resolve_status(v, date="2026-09-10", repo_root=str(tmp_path)),
    )


@pytest.mark.asyncio
async def test_status_template_flows_into_real_process(tmp_path):
    # A verifier that only passes when the resolved status matches.
    import json
    import sys
    report = tmp_path / "r.md"
    report.write_text(REPORT, encoding="utf-8")
    code = (
        "import sys,json;"
        "print(json.dumps({'verified': sys.argv[1]=='limited_signal'}))"
    )
    v = CronVerifier(
        name="v",
        argv=(sys.executable, "-c", code, "{status}"),
        status_file=str(report),
        timeout=10,
    )
    agg, _ = await run_external_verifiers([v], date="2026-09-10", default_cwd=str(tmp_path))
    assert agg == "PASS"


@pytest.mark.asyncio
async def test_unresolved_status_fails_closed(tmp_path):
    import sys
    code = "import sys,json;print(json.dumps({'verified': bool(sys.argv[1])}))"
    v = CronVerifier(
        name="v",
        argv=(sys.executable, "-c", code, "{status}"),
        status_file=str(tmp_path / "missing.md"),
        timeout=10,
    )
    agg, _ = await run_external_verifiers([v], date="2026-09-10", default_cwd=str(tmp_path))
    assert agg == "MISSING"