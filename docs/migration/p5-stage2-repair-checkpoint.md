# P5 Stage 2 repair checkpoint

Status as of 2026-10-01: **repair review CORRECT; Stage 1 cutoff restoration PASS;
Stage 2 is not accepted**. This
checkpoint records the repair fan-in after the rejected
[first attempt](p5-stage2-checkpoint.md). It does not overwrite that historical
checkpoint, replace the accepted
[implementation contract](p5-stage2-implementation-contract.md) or
[repair contract](p5-stage2-repair-contract.md), issue Stage 2 PASS, or authorize
Stage 3, P5 closure, D9, activation, live-state work, dependencies, background
work, deployment, or Windows qualification.

## Repair result and accepted dependencies

Packages A and B are accepted as repair dependencies, not as Stage 2 closure.
C-R replaced the rejected export implementation, removed the old private export
path, and made the unapproved five-field journal binding fail closed. D implements
the copied-state artifact operator with real execution ownership confined to its
private scratch source. E implemented the copied-state CLI adapters and proposed
default wiring. The falsey-injection defect from the historical attempt is fixed:
explicit authority selection uses `is not None`, so this is no longer an open
finding.

The fresh integrated review returned **CORRECT**, not PASS. It found the overall
architecture usable, but AC2, AC5, and AC6 still require correction:

- **AC2:** bootstrap and list do not yet account for checkpoint sidecars in their
  inventory and exact journal bindings. Legacy reset can also leave a stale
  checkpoint.
- **AC5:** delete and fork do not yet hold the required execution-before-storage
  lock, and fork does not yet journal exact source evidence.
- **AC6:** SDK clear/restore/ingest and WebUI title/scope paths can still apply
  legacy save behavior to canonical sessions instead of routing through the
  manager's canonical authority. The WebUI scope save is part of the same broken
  consumer boundary.

The reviewer found the remaining acceptance criteria covered, except that AC12 is
necessarily qualified by the environment and test limitations below. These
findings require a fresh bounded integrity replan. They are not a third correction
round for exhausted package A under another name.

Saved managed import is not an approved Stage 2 operation and should reject early as
unsupported unless a later contract explicitly admits it. A detached
`save=False` session must remain detached: it must change neither manager cache nor
durable state. Metadata changes continue through `SessionManager`; consumers must
not add local canonical/legacy persistence decisions.

## Material canonical SDK clear gate

A positive synchronous canonical SDK `clear()` conflicts with the accepted
task-bound asynchronous `ExecutionOwner`. No product decision is made in this
checkpoint. Two designs remain **unapproved** and require an explicit user choice:

1. Authorize a manager-owned out-of-turn reset using the existing
   `execution_transaction`, a frozen reset plan, nonblocking busy rejection, and
   recovery-required rejection. This preserves the synchronous SDK API and adds no
   second owner abstraction.
2. Add an asynchronous SDK reset that acquires the real execution owner, while the
   synchronous `clear()` remains legacy-only and rejects canonical sessions.

Read-only architecture analysis recommends the first design, but that recommendation
is not approval and does not resolve the gate. Do not infer either answer from the
current implementation or passing tests. Until this choice and the AC2/AC5/AC6
replan are accepted, the canonical SDK-clear path cannot support a Stage 2 PASS.

## Safety cutoff and rollback state

Package E safety-only correction 2 restored the actual
`Stage1UnavailableSessionAuthority` default required by the repair contract's
rollback boundary. Explicit injected authority still wins through `is not None`;
`DurableSessionAuthority` is now explicit-only. The production migration guards are
unchanged. The rollback touched only `nanobot/runtime_composition.py`,
`tests/migration/test_p5_stage1_composition.py`, and
`tests/migration/test_p5_stage2_composition.py`.

The worker and root validations below verify the restored cutoff. A final narrow
independent review returned PASS for this safety-only rollback: the actual default
is `Stage1UnavailableSessionAuthority`, authority acquisition remains before
bus/tools/MCP/cron/agent/channels work, explicit injection is intact, and guards did
not change. Root independently accepted that rollback PASS only. It is not a Stage 2
PASS or activation authorization.

## Validation evidence

All selections were offline and overlap; their counts must not be summed.

| Selection | Exact outcome |
| --- | --- |
| Contract Section 10 Stage 2 selection, pre-rollback post-flush diagnostic | 174 passed in 8.00s |
| Contract Section 10 Stage 2 selection, root post-rollback rerun | 174 passed in 8.77s; exit 0 |
| Complete `tests/session` selection before rollback; session implementation was untouched by rollback | 934 passed in 20.02s |
| Stage 1 startup/composition selection before rollback | 80 passed in 21.52s |
| Contract Section 10 Stage 1 matrix, root post-rollback rerun | 80 passed in 18.62s; exit 0 |
| Adapter engine/SDK/API selection | 84 passed in 9.09s |
| E worker focused rollback selection | 25 passed |
| Root scoped Ruff on the three rollback files, exact compileall, and `git diff --check` | Passed; exit 0 |

The broader scoped Ruff/compileall/diff validation before rollback also passed. The
broad contract Ruff selection still reports 69 unchanged baseline errors in eight
unrelated files and is not green evidence. BasedPyright and native Windows remain
unqualified. The real unpatched CLI cannot start in this environment because
`setproctitle` is missing; the CLI operator tests patch only the process-title
setter, not Stage 2 behavior. No dependency was installed to hide that gap.

## Next boundary

Ask the user to choose the canonical SDK-clear contract, then write a fresh bounded
integrity repair contract for AC2, AC5, and AC6. Re-run the exact integrated
selections and fresh independent review after that repair. Stage 2 remains stopped
at the Stage 1 cutoff; Stage 3 must not start, and root alone may eventually issue
Stage 2 PASS.
