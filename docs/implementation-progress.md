# Implementation progress: mobile remote and session context lifecycle

Branch: `feat/mobile-remote-context-lifecycle`
Worktree: `/home/jaden/Documents/Utilities/nanobot-custom-worktrees/mobile-remote-context-lifecycle`
Base: `a90191a41b025884e76ce029b98a5d5349171f2e`

This log records package status, acceptance evidence, commands and exits, and
remaining blockers. Private logs and runtime data stay outside the repository.

## Environment

| Item | Value |
| --- | --- |
| Prepared by | repo-local `.venv` (uv 0.13.7), Python 3.14.8 |
| Setup | `python3 -m venv .venv`, `.venv/bin/pip install -U uv`, `.venv/bin/uv sync --all-extras --dev`, `.venv/bin/uv run --no-sync python -m scripts.install_channel_dependencies --all-channels` |
| Import check | `.venv/bin/python -c "import nanobot; print(nanobot.__file__)"` -> worktree `nanobot/__init__.py` |
| Not used | original checkout editable virtualenv, live services, credentials |

## Package status

| Package | Status | Notes |
| --- | --- | --- |
| P1 provider route and request preparation | complete (focused checks) | typed route metadata, wrapper preview, auxiliary primary binding |
| P2 ordinary context lifecycle / recovery | in progress | follows P1 |
| P3 authorized media inputs | pending | independent of P1/P2 |
| Mobile A socket correctness | pending | |
| Mobile B delivery authorization | pending | |
| Mobile C owner wiring / foreground recovery | pending | after A/B |
| Mobile D pairing continuity | pending | after C |
| Mobile E request authority/backend | pending | after D |
| Mobile F composer settings/presentation | pending | after E/C |
| Mobile G standard feature dispatcher | pending | separate change |
| Integrated qualification / M38 | pending | external resources may remain blocked |

## Acceptance evidence

### P1

Outcomes:
- Wrapper capability/accounting no longer disappears because fallbacks exist. The next
  main attempt (including a circuit-open primary) is prepared against the leaf that
  dispatch will actually use, with that leaf's model, window, reserve and policy.
- Auxiliary source-summary preparation stays on the captured primary and reports
  `purpose=source_summary`; wrapper `supports_source_summary` delegates to the primary.
- `PreparedRequestBudgetPreview.route` records content-free route facts and an opaque
  token; the governor accepts only routes consistent with their measurement.
- Auxiliary output caps are bounded by the selected route's declared generation and
  capacity instead of the superseded fixed 2,048 ceiling.

Focused checks:
- `.venv/bin/ruff check` on changed provider/governor files -> PASS.
- `.venv/bin/python -m pytest tests/providers -q` -> 1918 passed, 3 skipped.
- `.venv/bin/python -m pytest tests/agent/test_source_pressure_compaction.py tests/agent/test_finalized_budget_recovery.py -q` -> 59 passed.
- New authored cases: `.venv/bin/python -m pytest tests/providers/test_route_preview_contract.py -q` -> 6 passed.

Requirements-only author:
- Session `ses_e5ee14ff8ffe9xq73nEMpavR2E`; artifact
  `/tmp/opencode/p1-requirements-tests/test_route_preview_contract.py` (outside repo).
- Cases R1-R5 were adapted to the real public provider boundary without changing
  scenarios or expected outcomes. The author noted missing constructor/chat details;
  the adapter only supplies those concrete boundaries.

Type evidence:
- Frozen before report under identical dependencies:
  `/tmp/opencode/nanobot-p6-baseline-e76086b1` at `e76086b1`, full unchanged
  `basedpyright --outputjson` -> 0 errors / 528 files, warnings 0.
  Report `/tmp/opencode/p6-before-e76086b1.json`.
- Final after-report and `scripts/check_type_regressions.py` comparison pending the
  integrated checkpoint.

Known limitation:
- Route selection is recorded for planning/accounting; dispatch independently
  re-evaluates the same next-attempt decision and reprices the selected leaf. A
  cross-check rejecting a stale selection is not yet enforced by a typed dispatch
  exception; final per-leaf guards still reject over-budget bodies.

## Commits

| Commit | Scope |
| --- | --- |
| e76086b1 | handoff documentation (pre-existing) |
