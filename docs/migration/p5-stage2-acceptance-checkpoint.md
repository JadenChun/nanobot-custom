# P5 Stage 2 offline acceptance checkpoint

Status as of 2026-10-01: **root PASS for guarded offline Stage 2**. This supersedes
Stage 2-open claims in the historical [attempt](p5-stage2-checkpoint.md) and
[repair](p5-stage2-repair-checkpoint.md) checkpoints without modifying either.
The [implementation contract](p5-stage2-implementation-contract.md),
[integrity/compatibility contract](p5-stage2-integrity-contract.md), accepted
[Stage 1](p5-stage1-checkpoint.md) and [P4](p4-closure-checkpoint.md) remain the
behavioral authorities. Production is still guarded and inactive.

Workspace: `/Users/jaden/Documents/github/nanobot-harness-candidate`, branch
`codex/harness-migration-candidate`. The protected fork was not edited. Unrelated
dirty work was preserved; no commit, installation, live-state/config/credential
operation, deployment or activation occurred.

## Delivered behavior

Exact paired source/checkpoint inventory and immutable journal evidence cover
bootstrap, reset, delete, fork and recovery. Managed validation is pure; canonical
sources reject checkpoint sidecars. Prepared proof precedes staging. Lifecycle
operations use execution-before-files locks, including direct resolver calls;
reuse checks lifecycle, PID, thread, task and held-key subsets before postcondition
reads. Fork source/target locks are sorted and source provenance is exact.

SDK clear uses manager-owned reset. Saved managed imports reject early;
`save=False` imports remain detached. Titles and workspace metadata use manager
authority. Unmanaged metadata publication preserves cached identity/local history,
updates committed metadata/base only after success, and supports later saves.
New ordinary sessions receive a committed cache snapshot.

Regular/new sessions and temporary contexts are preserved. WebUI explicitly
retains a temporary Session in the existing manager cache. Engine admission
selects that exact identity and nonpersistence before canonical creation, retaining
cross-turn history in memory with existing memory/tool/log restrictions. Eviction,
ownership, discard/disconnect, active cancellation, queued admission rejection and
late transcript suppression are covered. IDs never determine persistence.
Managed default transient construction and explicit SDK ephemeral execution remain
detached, including at a live temporary key. No mode commands, context-builder or
memory redesign, additional authority, or approval-map change was included.

The actual definition-only default is `DurableSessionAuthority`, constructed from
captured preflight workspace/session roots. Explicit falsey injection wins through
`is not None`; definition stays read-only. Historical cutoff tests now explicitly
inject `Stage1UnavailableSessionAuthority`.

## Root validation

All selections were offline with temporary fixtures and the pinned Python/Ruff
paths. Counts overlap and must not be summed. Exact commands are in implementation
Section 10 and integrity Sections 6/8.

| Selection | Outcome |
| --- | --- |
| Complete Section 10 Stage 2 selection, restored default | 335 passed in 37.82s |
| Complete `tests/session` | 1043 passed in 50.59s |
| Section 10 Stage 1 startup matrix | 80 passed in 38.43s |
| Package III composition/engine/CLI | 56 passed in 4.92s |
| Compatibility/consumers/engine/fork/workspace | 93 passed in 7.04s |
| Three unchanged websocket regressions | 3 passed in 1.45s |
| Existing title selection | 8 passed, 70 deselected in 0.20s |
| Frozen export | 19 passed in 6.64s |
| Frozen export/crash | 27 passed, 126 deselected in 8.41s |
| Frozen artifact operator/rematerialization/factory/store | 75 passed in 12.53s |
| Final critical Stage 2 rerun | 335 passed in 17.97s |
| Final critical compatibility plus unchanged regressions | 96 passed in 4.89s |
| Changed-file Ruff, exact compileall and diff checks | Passed |

The unchanged six-module compatibility diagnostic is **108 failed, 267 passed**
in 13.50s. Failed-node comparison adds nothing against the saved 114-failure
baseline, and removes the six owned title failures. Remaining historical guarded
constructors, removed `_make_provider` expectation and unrelated websocket settings
mismatch are not green evidence or authorization to weaken guards/fixtures.

Exact broad Ruff reports **69 pre-existing findings**; JSON fingerprint comparison
has zero additions/removals. BasedPyright is unavailable, native CLI lacks
`setproctitle`, and native Windows remains unqualified. CLI operator tests retain
only their existing process-title-setter patch. No dependency was installed.

## Fresh review and root acceptance

A fresh read-only **GPT-6.1 Sol Medium** reviewer inspected the actual integrated
203435-character repair diff, contracts, call paths, crash matrices and raw results.
Its independent operations/temporary/engine/consumer/composition selection passed
**120 tests in 8.11s**. Recommendation: **PASS**, no material finding or missing
regression within the bounded repair. Root independently inspected the changed
boundaries, ran the above checks, verified scope/hashes and issued final PASS.

| Criterion | Accepted evidence |
| --- | --- |
| AC1 | Read-only definition, captured roots, falsey injection, first canonical write and unchanged guards. |
| AC2 | Paired inventory, pure strict overlay, exact bootstrap/resume and complete read-only listing. |
| AC3 | Lifetime lease, nonblocking contention and direct resolver task-bound execution-before-files locking. |
| AC4 | Record-based mixed modes, legacy provider state and accepted canonical execution ownership. |
| AC5 | Fresh-operator crash matrices, exact predecessor/target/source/sidecar proof, prepared-before-stage and forward-only committed recovery. |
| AC6 | Manager reset/fork/metadata, SDK clear/imports, title/scope consumers and postcommit cache compatibility. |
| AC7 | Frozen lossless/compatibility exporter and exact export/crash gates. |
| AC8 | Frozen opt-in artifact quotas and blob-before-reference publication. |
| AC9 | Exact copied-state rematerialization, unchanged input and private scratch execution bookkeeping. |
| AC10 | TTL/idle preflight rejection and dormant background/retention/rematerialization construction. |
| AC11 | Content-free managed validation and changed consumer errors/diagnostics. |
| AC12 | Exact offline selections pass; diagnostic, lint and environmental non-passes remain qualified above. |

Compatibility C1-C6 are covered by unchanged regression nodes, real managed/
unmanaged lifecycle fixtures, two actual engine turns, deterministic disposal
race, post-retention corrupt evidence, detached ephemeral tests, cache publication
failure/retry/save and frozen dependency gates. All nodes in this completion round
used fresh contexts and explicit GPT-6.1 Sol, never GPT-5.6 Sol. Fresh integrated
review also covers the earlier capacity-based analyst reuse. No correction round
was needed for the new temporary package or Package III.

Full pre/post file hashes show no unrelated implementation edit. Frozen exporter/
artifact files, `execution_owner.py`, `migration_guard.py`, historical Stage 2
checkpoints, Stage 1 checkpoint and P4 closure are unchanged. Raw local snapshots/
evidence are retained under `/tmp/nanobot-stage2-compat.BBWZmr` and
`/tmp/nanobot-stage2-review.MTl8pi`; contract commands are reproducible.

## Qualified boundaries and next stage

Cooperating locks do not establish old-binary exclusion; stopped-writer copied-state
qualification remains required. Unmanaged absent metadata upsert retains its
pre-existing two-commit behavior: metadata failure can leave an empty durable
baseline, but publishes no cache/base changes; retry and later save are covered.

**Stage 3 is ready as the next separately authorized functional stage; it has not
started.** This does not close P5 or authorize D9, live bootstrap/conversion,
retention/removal, scheduler/Dream/Heartbeat/callback/idle activation, installation,
P6/native/Windows qualification, P7 canary, deployment or live traffic.
