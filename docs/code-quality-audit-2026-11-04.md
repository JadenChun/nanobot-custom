# Code quality audit — 2026-11-04

Status: audit charter and internal size baseline. This is **not** a code-quality
verdict, a refactor plan, or a release record. No production code was changed and
no full analysis has been run yet.

Scope: this fork's codebase, not upstream parity. The previous
`docs/upstream-divergence-audit-2026-11-04.md` was removed because "how far are we
from upstream?" is not a useful quality question for a heavily customized fork.
Only its internal size measurements are retained in Appendix A.

## Measurement refs

| Item | Ref |
|---|---|
| Code baseline | `staging` at `90a6e2fc` (2026-11-04) |
| Previous doc baseline | `ea4d6a65` / `d7217cb0` |
| Upstream | Not used as a quality criterion |

## Why this audit exists

The goal is a codebase that is **minimal, efficient, clean, and durable**:

- minimal in concepts and invariants, not merely in lines;
- efficient in startup, memory, I/O, and event-loop behavior;
- clean in boundaries, ownership, error handling, and naming;
- durable in the sense that a future maintainer can change one thing without
  understanding the whole tree.

Design patterns and "best practice" are means, not goals. A pattern is justified
only when it removes coupling, protects a real boundary, or makes a required
failure mode explicit. A new abstraction must remove more complexity than it adds.

## Operating principles

1. **One owner per invariant.** Session state, run state, cron schedules, memory
   recovery, provider selection, and tool registration each need one authoritative
   owner. Parallel ledgers are defects even when they currently agree.
2. **Delete before abstract. Merge before split.** Most cleanup should remove or
   collapse code, not add a layer.
3. **Boundaries before patterns.** Domain code must not depend on delivery
   surfaces. Adapters at the edge may depend inward; the core must not depend on
   `webui`, `channels`, `desktop`, or `mobile`.
4. **Measure before optimize.** Performance work starts from a profile, not a
   guess. Correctness and recovery work starts from a named failure mode.
5. **Behavior preservation.** Every audit-driven change must be small, reversible,
   and covered by a test of observable behavior. No drive-by rewrites.
6. **No line-count target.** LOC is a heatmap. A 400-line module with one clear
   invariant is better than four 100-line modules that share mutable state.
7. **No speculative generality.** A config flag, adapter, or interface needs a
   current caller and a real boundary. "We might need it later" is not enough.
8. **Ratchet, don't big-bang.** Baseline existing debt, fail on new violations,
   and reduce the baseline in focused batches.

## What "clean" means here

A module is in good shape when all of these are true:

- its owned invariant is named and documented;
- it has one reason to change;
- its dependencies point in the intended direction;
- its public surface is small and typed;
- errors are explicit and mapped consistently;
- writes are atomic and retries are idempotent where required;
- it can be tested without constructing the whole runtime;
- it can be deleted or replaced without editing unrelated modules;
- it does not block the event loop or perform redundant I/O;
- its tests assert behavior, not private call order.

## Audit dimensions

| Dimension | What must be established |
|---|---|
| **Boundaries** | Intended import direction, no cycles, no domain-to-delivery dependencies, adapters isolated at edges. |
| **Ownership** | One source of truth per invariant; no parallel stores, caches, or recovery ledgers. |
| **Complexity** | Function and module complexity is justified by requirements; large files are cohesive, not mixed responsibilities. |
| **Duplication** | No copy-pasted state machines, provider wrappers, channel boilerplate, or recovery logic. |
| **Minimality** | Every abstraction has a current caller or a hard boundary; dead code and compatibility shims are removable. |
| **State and durability** | Atomic writes, explicit transaction boundaries, idempotent retries, recovery tied to a named failure mode. |
| **Async and errors** | No blocking calls in `asyncio`, no unbounded queues, no swallowed exceptions, cancellation and cleanup are explicit. |
| **Performance** | Startup, memory, per-turn overhead, streaming, and tool I/O are measured; no optimization without evidence. |
| **Tests** | Behavior-focused, minimal fixtures, no implementation-coupled assertions, skips are justified, risky modules covered. |
| **Security** | Tool/workspace/credential boundaries are explicit; untrusted input paths are identified; approval and audit semantics are consistent. |
| **Docs and config** | Documentation matches current behavior; unused options and stale compatibility paths are removed. |

## Severity and disposition

| Severity | Meaning |
|---|---|
| **P0** | Correctness, security, data-loss, or recovery defect. Fix before other cleanup. |
| **P1** | Duplicated invariant, unsafe ownership boundary, or complexity hotspot that will cause defects or block change. |
| **P2** | Cleanup: dead code, naming, small duplication, stale config/docs. Batch these. |

Each finding gets one disposition:

- **Keep** — justified as-is; record why.
- **Simplify** — same behavior, less code or fewer concepts.
- **Merge** — two owners/layers become one.
- **Delete** — no current caller or requirement.
- **Split** — cohesive pieces separated, without adding indirection.
- **Rewrite** — last resort; requires a behavioral contract and tests first.

## Phase 0 — Inventory and heatmap (read-only)

Produce a module inventory before judging anything:

- LOC and file count per package, including test/production split;
- fan-in and fan-out per module;
- import cycles and layer violations;
- function complexity distribution;
- duplicate-code clusters;
- dead-code candidates;
- test coverage and slowest tests;
- churn over the last 6–12 months;
- risk score = complexity × churn × blast radius.

Suggested commands (install audit tools only if already available or explicitly
approved; do not add dependencies as part of this phase):

```bash
.venv/bin/ruff check nanobot --select C901,PLR0911,PLR0912,PLR0913,PLR0915 --statistics
.venv/bin/radon cc -s -a nanobot/agent nanobot/session nanobot/cron
.venv/bin/radon mi -s nanobot/agent nanobot/session nanobot/cron
.venv/bin/vulture nanobot --min-confidence 80
.venv/bin/pylint --disable=all --enable=duplicate-code nanobot/agent nanobot/session nanobot/cron
.venv/bin/pytest --cov=nanobot --cov-report=term-missing --durations=25 -q
```

For architecture direction, use an import-graph tool (`import-linter`, `tach`, or
`pydeps`) if available; otherwise generate the graph with a small read-only
script and record the exceptions.

## Phase 1 — Boundaries and ownership

Questions:

1. What is the intended dependency direction? Write it down before fixing
   violations. A reasonable starting point: `agent`/`session`/`providers` must
   not import `webui`, `channels`, `desktop`, or `mobile`; delivery adapters may
   import inward.
2. Which modules mutate the same invariant? List every writer for session state,
   run state, cron state, memory state, and provider selection.
3. Are there parallel ledgers, caches, or recovery records that can disagree?
4. Does any module reach into another module's private helpers or lock order?
5. Can each module be deleted without touching unrelated code? If not, document
   the coupling and decide whether it is required.

The first target is not "make imports pretty"; it is **remove duplicate ownership**.

## Phase 2 — Complexity, duplication, and abstraction

Questions:

1. Which functions exceed a cognitive-complexity threshold, and are they complex
   because the domain is complex or because responsibilities are mixed?
2. Which files exceed ~800–1,000 lines, and are they cohesive?
3. Which duplicate clusters represent parallel implementations of the same
   invariant (providers, fallbacks, stores, recovery, channel setup)?
4. Which abstractions have exactly one implementation and no boundary
   requirement?
5. Which generic layers exist only to pass data through? Collapse them.

Start with a small set of thresholds and calibrate after the baseline. Do not
mass-format or mass-rename; churn hides real changes and makes review harder.

## Phase 3 — State, durability, and async correctness

This is likely the highest-risk area in this codebase.

Questions:

1. For each `*_store.py`, `*_recovery*.py`, and `*_state.py`: what invariant does
   it own, what failure mode requires it, and what test proves that failure mode?
2. Are writes atomic? Are partial writes recoverable? Is retry idempotent?
3. Is there one transaction boundary per external effect, or multiple layers
   that can each claim success?
4. Are session/run/cron/memory identities stable across restart and retry?
5. Does any `asyncio` path call blocking I/O, hold a lock across an `await`, or
   leave a task uncancelled?
6. Are queues bounded? Are backpressure and timeout behavior explicit?
7. Are errors typed and mapped consistently, or converted to strings and lost?

Specific clusters to inspect first:

- `agent/memory_recovery_*` — 24 files, 15,569 lines;
- `cron/*_store.py` and delivery/reconciliation layers;
- `session/manager.py` and the session source/store/recovery split;
- `remote/` versus `webui/` remote paths;
- `desktop/` adapter versus gateway runtime;
- provider fallback/pooled/legacy wrappers.

These are **hypotheses**, not findings. Each cluster needs an invariant map before
any code is removed.

## Phase 4 — Performance and resources

Measure, then optimize:

- gateway startup time and import cost;
- RSS/PSS at idle and under a representative turn;
- event-loop blocking and long tasks;
- per-turn serialization/deserialization and redundant I/O;
- token streaming overhead;
- tool subprocess and file-scanning cost;
- memory growth across sessions and recoveries.

Use `py-spy`, `scalene`, or `cProfile` where appropriate, and keep before/after
measurements with the same workload. No performance change without a profile and
a repeatable comparison.

## Phase 5 — Tests, docs, and configuration

- Separate behavior tests from implementation tests; remove tests that assert
  private call order or duplicate fixtures.
- Identify skips and conditional tests that hide untested risk.
- Check coverage of the modules identified in Phases 1–3, not just global coverage.
- Find config keys, docs, and CLI flags with no current reader.
- Check that migration/type-debt policy is followed; do not refresh a baseline to
  absorb new debt.
- Remove stale compatibility shims only after confirming no supported caller.

## Initial hotspots (hypotheses)

From the size baseline, the first inspection order is:

1. `agent/` — 63,066 lines; `agent/loop.py` ~4,876; `memory_recovery_*` 24 files.
2. `session/` — 25,696 lines; `session/manager.py` ~3,428.
3. `cron/` — 17,306 lines; multiple stores and reconciliation layers.
4. `providers/` — 23,864 lines; fallback/pooled/legacy wrapper overlap.
5. `channels/` — 65,910 lines; shared setup/validation/delivery duplication.
6. `webui/` and `remote/` — overlapping remote/session paths.
7. `desktop/` — small Python adapter, but a large Rust shell and process boundary.

Do not start with the largest file. Start with the invariant that has the most
writers and the highest blast radius.

## Findings format

Each finding should be recorded as:

```text
ID: F-001
Severity: P0 | P1 | P2
Disposition: Keep | Simplify | Merge | Delete | Split | Rewrite
Location: path:line
Invariant: what state/behavior is owned here
Evidence: complexity, duplication, import graph, test gap, or profile
Risk: what can go wrong if unchanged
Minimal fix: the smallest change that restores one owner/boundary
Tests: behavior tests that must pass
Rollback: how to revert the change safely
```

## Deliverables

1. **Module heatmap** — production/test LOC, files, fan-in/out, complexity,
   coverage, churn, risk score.
2. **Findings register** — P0/P1/P2 with evidence and minimal fixes.
3. **Delete/merge list** — dead code, duplicate abstractions, unnecessary layers.
4. **Target architecture note** — dependency direction, ownership map, public
   boundaries. Short; not a diagram exercise.
5. **Guardrails** — automated checks that prevent regression.
6. **Phased remediation plan** — small batches, each with tests and a rollback.

The audit is complete when every module has a disposition and the guardrails are
in place. A list of findings alone is not completion.

## Guardrails

- Enforce intended import direction in CI; fail on new cycles or core-to-delivery
  imports.
- Ratchet complexity and file-size thresholds against the recorded baseline; fail
  on new violations, not on existing debt.
- Fail on new duplicate-code clusters above a calibrated threshold.
- Require a named invariant and a failure-mode test for any new `*_store.py`,
  `*_recovery*.py`, or persistence layer.
- Require coverage of changed behavior, not a global percentage target.
- Keep the P6 type-debt policy: no baseline refresh to hide new signatures.
- Measure gateway startup and per-turn overhead when changing shared paths.

## Non-goals

- Upstream behavioral parity or code-shape parity.
- A full rewrite, a language migration, or a framework replacement.
- Reducing total line count for its own sake.
- Formatting-only churn, mass renames, or pattern cargo cult.
- Multi-tenant bot profiles, recursive agent promotion, or platform governance.
- New abstractions without a current caller and a real boundary.

## Appendix A — internal size baseline

Measured on `staging` at `90a6e2fc`. Counts include blank lines, comments, and
docstrings; they are not SLOC. Python tests under `nanobot/channels/*/tests` are
included in the package total and shown separately where noted.

| Area | Lines | Files |
|---|---:|---:|
| Python package | 268,423 | 594 |
| — non-test | 230,458 | 526 |
| — test (channel-local) | 38,965 | 68 |
| Python root tests | 277,875 | 761 |
| Python scripts | 1,439 | 5 |
| TS/JS (webui/mobile/tui) | 128,426 | 329 |
| — non-test | 72,879 | 216 |
| — test | 55,547 | 113 |
| Rust (Tauri source) | 3,224 | 7 |
| **Non-test total (Python + TS/JS + Rust)** | **306,561** | — |
| **Test total (Python + TS/JS)** | **372,387** | ~942 |

Top Python directories:

| Directory | Lines |
|---|---:|
| `channels` | 65,910 |
| `agent` | 63,066 |
| `session` | 25,696 |
| `webui` | 24,775 |
| `providers` | 23,864 |
| `cron` | 17,306 |
| `cli` | 9,768 |
| `remote` | 6,679 |
| `utils` | 6,134 |
| `security` | 4,645 |
| `config` | 2,115 |
| `gateway` | 1,975 |
| `context_repo` | 1,776 |
| `apps` | 1,622 |
| `skills` | 1,437 |
| `command` | 1,318 |
| `triggers` | 986 |
| `crawler` | 944 |
| `desktop` | 795 |
| `llm_usage` | 754 |
| `bus` | 740 |
| `sdk` | 719 |
| `api` | 550 |
| `pairing` | 403 |
| `audio` | 319 |
| `extensions` | 165 |

`agent/memory_recovery_*`: 24 files, 15,569 lines.

These numbers are for prioritization only. A smaller line count is not the goal,
and a large directory is not automatically defective.

## Appendix B — reproduction

```bash
# Production/test split
find nanobot -name '*.py' -not -path '*/tests/*' -print0 | xargs -0 cat | wc -l
find tests -name '*.py' -print0 | xargs -0 cat | wc -l

# Per-directory totals
for d in nanobot/*/; do
  printf '%-20s %s\n' "$(basename "$d")" \
    "$(find "$d" -name '*.py' -print0 | xargs -0 cat | wc -l)"
done | sort -k2 -nr

# Frontend and Rust source
find webui/src mobile/src tui/src -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.mjs' | xargs cat | wc -l
find desktop/src-tauri -path '*/target' -prune -o -name '*.rs' -print0 | xargs -0 cat | wc -l
```

For future comparisons, pin the measured commit and record the exact commands,
tool versions, and environment. Do not compare line counts across different
scopes and call it quality.
