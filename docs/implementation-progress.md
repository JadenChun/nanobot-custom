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
| P2 ordinary context lifecycle / recovery | focused complete; secondary replanning partial | preventive budgeting wired; bounded replanning not expanded |
| P3 authorized media inputs | focused complete | exact read-only input snapshots; foreground only |
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

### P2

Outcomes:
- Pressure and admission now use the route-finalized prepared request (P1 wrapper
  preview) rather than a generic legacy estimate whenever the route supports it.
- Summary generation allowance resolves from the captured summary runtime's
  generation setting, clamped by declared model output capability, route capacity
  and the operation budget. The accepted `SourceCompactionConfig` override is
  `summaryGenerationTokens` / `summary_generation_tokens`.
- Retained summary text is separately bounded by the rebuilt main request's
  available space and is the publication/truncation boundary. A completed `stop`
  response over that retained budget is discarded, not published.
- Compaction reserves capacity for the summary envelope, useful generation,
  estimate safety and bounded next-step growth; proactive compaction can trigger
  before the old 90%-of-budget ratio when that reserve would otherwise be consumed.
- Trigger/target ratios and the generation override are typed deployment settings
  with camelCase and snake_case spellings and validation (target < trigger).

Focused checks:
- `.venv/bin/ruff check` on changed context/config/provider files -> PASS.
- `.venv/bin/python -m pytest tests/agent/test_source_pressure_compaction.py tests/agent/test_source_compaction_publication.py tests/agent/test_source_dependency_compaction.py tests/session/test_source_compaction.py tests/agent/test_finalized_budget_recovery.py -q` -> 119 passed.
- `.venv/bin/python -m pytest tests/agent/test_memory_budget_hardening.py -q` -> 17 passed.
- `.venv/bin/python -m pytest tests/config tests/agent/test_context_governance.py tests/agent/test_compaction_proof_boundary.py tests/agent/test_defensive_boundary_views.py tests/agent/test_loop_consolidation_tokens.py -q` -> passed after the fixture updates below.
- New authored config cases: `.venv/bin/python -m pytest tests/config/test_source_compaction_policy.py -q` -> 16 passed.

Conflicting cases updated intentionally with the accepted plan:
- `tests/agent/test_source_pressure_compaction.py` now expects the summary
  allowance from the captured generation instead of a fixed 2,048.
- `tests/agent/test_memory_budget_hardening.py` / `tests/agent/p6_memory_contract_fixtures.py`
  pin the synthetic fixture's runtime generation to 2,048 so its exact-boundary
  arithmetic remains deterministic under a model-derived budget.

Requirements-only author:
- Session `ses_e5ed3a802ffe1dYcK02mlHNPk7`; artifact
  `/tmp/opencode/p2-requirements-tests/test_source_compaction_config.py` (outside repo),
  integrated unchanged as `tests/config/test_source_compaction_policy.py`.

Known limitations:
- Exceptional summary-output truncation fails honestly within the existing call
  ceiling and preserves source/checkpoint; a bounded smaller-chunk retry strategy
  was not added. Preventive continuity is the validated path.
- No live provider canary or physical long-session journey has been run; these
  remain qualification items, not claims.

### P3

Outcomes:
- `exec` accepts an additive optional `input_files` list of exact paths. Omission
  preserves the existing shell confinement.
- Each path resolves through the same `read_file` policy, must be an approved
  regular file, and is copied into a bounded server-owned per-invocation snapshot
  (16 files / 64 MiB) before launch. Symlinks, private runtime storage and changed
  sources fail closed. Writes to the mounted paths are read-only.
- Managed bubblewrap restores only those snapshot files at their original paths,
  after private instance masking. Unlisted siblings/parent directories stay
  masked, and the whole media directory is never mounted.
- Foreground-only: `yield_time_ms`/`notify_on_exit` with `input_files` is rejected
  because snapshot lifetime cannot be guaranteed.
- MCP image results and generated-image results advertise the exact-input contract;
  saved replay images now say no bytes are present and allow an authorized reread.

Focused checks:
- `.venv/bin/ruff check` on changed tool/provider files -> PASS.
- `.venv/bin/python -m pytest tests/tools/test_sandbox.py tests/tools/test_filesystem_tools.py tests/tools/test_mcp_tool.py tests/utils/test_artifacts.py tests/agent/test_recent_image_governance.py tests/agent/test_managed_exec.py tests/providers/test_recent_image_history.py -q` -> 199 passed.
- Real managed-bwrap smoke: `test_exec_input_files_exposes_only_exact_approved_files`
  copies bytes from a media path, denies an unlisted sibling, denies private
  storage and a symlink, verifies read-only mounting, and checks the advertised
  schema bound.

Requirements-only author:
- Session `ses_e5ecc4b29ffeTf6rSA3Fxrsdkq`; case specification artifact
  `/tmp/opencode/p3-requirements-tests/input_files_cases.md` (outside repo),
  integrated into the focused managed-exec test.

Known limitations:
- Session/background shell commands cannot use `input_files` in this change.
- No cross-turn automatic historical image injection or binary-retention migration
  was added; explicit reread remains the supported path.

## Commits

| Commit | Scope |
| --- | --- |
| e76086b1 | handoff documentation (pre-existing) |
