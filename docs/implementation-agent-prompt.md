# Coding-agent handoff: mobile remote and session context lifecycle

Implement the accepted plans step by step. This authorizes implementation and local commits;
the preparation commit contains documentation only. Continue between packages without routine
approval questions. Complete independent work before reporting an external qualification blocker.

## Workspace and accepted scope

- Worktree: `/home/jaden/Documents/Utilities/nanobot-custom-worktrees/mobile-remote-context-lifecycle`
- Branch: `feat/mobile-remote-context-lifecycle`
- Fetched staging base: `a90191a41b025884e76ce029b98a5d5349171f2e` (2026-11-03).
- Read [session-recovery-architecture.md](session-recovery-architecture.md) and
  [mobile-remote-request-plan.md](mobile-remote-request-plan.md) in full. These are the accepted
  specifications; the sequence below coordinates their packages, not a replacement specification.
- Read `/home/jaden/.agents/AGENTS.md`, repository `AGENTS.md`, `.agent/{design,security,gotchas}.md`,
  `CONTRIBUTING.md` and applicable migration/release instructions. Follow the current
  `orchestrate-development` skill where applicable. Historical notes do not revive deferred graphs.

Use existing provider, governor, source/checkpoint, permission, native and UI owners. Make the
smallest correct changes; avoid incidental refactors, dependencies, new frameworks and test
infrastructure. Provider-specific semantics belong in adapters and declared capabilities, not
model-name branches in the governor. Reference DSH/OpenCode for complete lifecycle patterns;
do not infer equivalent production failures from their last-resort guards or copy their constants.

The primary context direction is consistent prepared-request accounting for the actual route,
reserved compaction capacity, model-derived summary budgets and verified prefix reuse when
compatible. Ordinary long-session continuity comes before secondary emergency recovery. The old
fixed 2,048 summary ceiling and 8:1 compression assumption are superseded. Keep durable source,
checkpoint publication proofs, current permissions and temporary-chat privacy intact.

## Execution discipline

1. Confirm this worktree/branch/base and inspect existing changes. You are not alone; preserve
   other work and use explicit, sequential ownership for overlapping files. Do not reset,
   overwrite or import unrelated repair branches. Reuse completed work/evidence when resuming.
2. Create `docs/implementation-progress.md` with package status, acceptance evidence, author
   artifact/session IDs, commit IDs, exact checks/exits and genuine remaining blockers. Update
   it at each accepted checkpoint; keep private logs/data outside the repository.
3. Prepare an isolated dependency environment before freezing baseline evidence. This new
   worktree has no prepared virtualenv or Android toolchain. Do not reuse the original checkout's
   editable virtualenv: it may import the original source. Confirm imports point here. Inspect
   installed tool availability and use the documented bootstrap procedures; do not alter live
   desktop services, shared model workloads or credentials to make tooling available.
4. For every Python repair, follow `docs/migration/p6-error-attribution.md`: freeze before/after
   full-source checker reports under identical dependencies, retain source/environment hashes
   and use `scripts/check_type_regressions.py` to require zero new diagnostic signatures across
   all files. Preserve immutable before-source evidence. Do not refresh a baseline to absorb
   errors. Historical passing counts are not qualification of this environment.
5. Per package, identify the smallest existing focused checks first. Add/change cases only for
   a concrete missing outcome. Before implementation, have a fresh `REQUIREMENTS_ONLY_TEST_AUTHOR`
   session author necessary cases using requirements, use cases and public interfaces only.
   Give it no parent history, repository/path/access, implementation plan/code, existing tests,
   diffs, logs or analysis. Use the shared skill's `references/requirements-only-tests.md`.
   Retain its scratch artifact/session ID. Integrate mechanically; send scenario/expectation
   corrections back to a requirements-only session. Never derive assertions from private logic.
6. Implement one bounded package, inspect its actual diff, run affected checks and applicable
   gates, update progress, then make a focused local commit. Reuse passing evidence for
   unchanged inputs. Do not run broad suites or add reviewers at every checkpoint. Never run
   `ruff format`. Any mandatory integrated release review remains required at its actual gate.
7. Resolve routine engineering details without reopening settled product choices. If repository
   reality prevents an agreed outcome, document the concrete conflict and missing requirement;
   do not silently reduce scope, weaken safeguards or claim completion.

## Implementation sequence

**Step 1 — Session P1: provider route and request preparation.** Carry the selected route's
capabilities, generation semantics and accounting through wrappers; keep primary-only auxiliary
preparation distinct from main failover. Support configured aliases and optional exact counting.
Reprepare/reprice after route changes; retain each leaf's final guard. Extend existing seams only.

**Step 2 — Session P2: ordinary context lifecycle, then exceptional recovery.** First align
pressure/admission measurements, reserve summary capacity, resolve summary generation versus
retained-text budgets, and restore legal prefix preparation. Validate normal long-session
continuation across representative capacities before adding bounded replanning/chunk recovery.
Preserve current operation deadlines/call ceiling, closed evidence coverage, dependency proofs
and atomic publication. Update conflicting compaction guidance/cases with this implementation.

**Step 3 — Session P3: authorized media inputs.** Make advertised generated files usable in the
managed sandbox through exact read-only invocation inputs under existing read policy. Preserve
private storage masking and honest saved-image availability. No new media authority, automatic
historical image injection or binary-retention migration.

**Step 4 — Mobile A, then B.** Correct socket identity/terminal lifecycle/UTF-8 handling, then
enforce fresh delivery authorization after awaited private reads. Preserve useful public-boundary
coverage while relocating the test-only helper. Keep these as independently reviewable commits.

**Step 5 — Mobile C, then D.** Wire owner operations and serialized native refresh; implement
bounded foreground connection probing and canonical persistent-history reconciliation. Then fix
pending pairing restoration and desktop-local invitation renewal. Preserve the temporary-ended
experience; never auto-replay uncertain submissions. Observe native ABI/build requirements.

**Step 6 — Mobile E, then F.** Implement request-scoped workspace/permission admission before
the UI consumes it. Preserve four states, immutable submission/queue snapshots, Full consent,
ephemeral authority and active-run context. Then polish consistent attachment/workspace/mode
controls and next-message editing. Verify touch, keyboard, accessibility and desktop layout.
Coordinate optional session P4 status/error presentation here; change it only for an unmet
observable requirement, with one owner for shared WebUI files.

**Step 7 — Mobile G as a separate change.** Converge supported mobile feature transport onto
the standard dispatcher with explicit compatibility handling. Preserve operation contracts,
authorization and replay semantics. This is required for full parity; do not silently omit it
or make it a prerequisite for the earlier urgent fixes.

**Step 8 — Integrated qualification and release readiness.** Run remaining mandatory integrated
checks, relevant type/lint/build gates and realistic continuity/reconnect/media journeys. Reuse
existing harnesses and disposable instances, with sanitized baseline/candidate inputs. Record
actual usage/latency/cache evidence where supported; do not invent performance thresholds/results.
Follow the plans for native, emulator, physical-phone and exact signed-candidate evidence.
Complete the required M38 independent integrated release review once the candidate is ready.
Unavailable phone/signing resources leave named qualification items pending while independent
implementation continues. Do not equate a bounded repair pass with P6 closure or full release.

## Boundaries and completion report

Do not modify the live session, installed gateway/owner/app, Tailscale setup, credentials or
model services. Do not install over the daily mobile app, run destructive custody checks on it,
merge to staging, push or publish. M39 installation remains a separate concrete user opt-in.
Prepare reviewable artifacts and rollback inputs before any later installation request.

Continue through all implementable packages. Report completion only for outcomes actually
achieved; optional follow-ups stay separate from required work. At the end provide package and
acceptance status, commits/files changed, exact checks/exits, qualification artifacts, remaining
requirements and external blockers. Keep the branch clean after each checkpoint and leave all
user data and unrelated work intact.
