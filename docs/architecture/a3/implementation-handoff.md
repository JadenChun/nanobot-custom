# A3 implementation correction handoff

## Continuation checkpoint: artifact handoff wired (2026-11-12)

Resume from `751ea75c` on `codex/staging-audit-implementation`. Staging remains
`f1a6bb85`. This checkpoint supersedes the artifact status and the known-red list in
the stop record below. The plan stays frozen as `A3-PLAN-2026-11-11`.

Completed in this checkpoint:

- `CanonicalChildArtifactInput` now passes through `ChildTurnRequest`, `TurnContext`,
  and `AgentRunSpec`.
- An awaited worker receives a collector even when it has no parent artifact
  references. This permits child-to-parent handoff for worker output.
- The runner prepares the receipt while the child owner is live. `execute_child_turn`
  seals it only after the exact completed run and healthy closed source.
- `TurnResult` and `AgentRunResult` carry the receipt to the durable parent tool
  result. Adoption stays one-shot.
- Settlement accepts a second `SourceSessionStore` wrapper over the same backend.
  It retains the admitted writer binding for later acceptance checks.
- `nanobot/extensions/crawler.py` no longer passes the removed `runner` argument.
- The crawler keeps its role-owned `social_crawl` tool. All other role tools remain
  narrowed by the live parent binding.
- The retired inline delegation tests now use canonical `AgentLoop` turns,
  source-backed trees, or recorded `ChildTurnRequest` values. Two cases were retired
  with named canonical coverage.

Verification on the Linux worktree with Python 3.14.10, BasedPyright 1.40.2, and
pytest 9.2.0:

| Check | Result |
| --- | --- |
| Migrated delegation suite plus canonical child and artifact suites | **240 passed**. |
| Artifact grant, handoff, reader, origin tests | **118 passed**. |
| Canonical child authority, controls, sessions, capacity, events, config | **46 passed**. |
| Child lifecycle and follow-up backend set | **68 passed**. |
| Full Ruff over `nanobot tests conftest.py` | **Passed**. |
| Whitespace check | **Passed**. |
| Configured strict check (`tool.basedpyright`, `nanobot`) | **0 errors / 545 files**, exit 0. |

The configured strict result is not the old full-source count in the stop record.
Reproduce it with the recorded environment before treating P6 as closed.

### Next work

1. Continue AC4 ordered follow-up from the existing canonical journal and source owner.
2. Implement AC5 durable background result delivery with restart reconciliation.
3. Finish AC8 projection, hydration, and canonical history access.
4. Then implement the client work and the remaining release gates.

No staging merge, deployment, or release acceptance is claimed.

## Latest handoff: stopped with unfinished backend work (2026-11-12)

**User-requested stop; combined backend and A3 remain unaccepted.** This entry supersedes
the earlier paused checkpoint and the later external `root-resume-state.md` instruction
to finish the backend first. The next agent must continue the current unfinished child
and artifact work before starting clients. The implementation graph is closed as
`STOPPED_USER_REQUEST`, without an acceptance verdict. Implementation owners have stopped;
no implementation, test, gateway or UI stage remains active.

The plan remains frozen as **`A3-PLAN-2026-11-11`**. Use the
[design](design.md), [product contract](v1-product-contract.md),
[implementation plan](implementation-plan.md), and the canonical-child/client specifications
later in this document. Architecture and product choices have not reopened. Future coding
harness extension points remain required; SuperAgent coordinates normal sessions and does
not redefine ordinary subagents.

### Git checkpoint and scope

Continue from the **WIP pause commit containing this entry** on
`codex/staging-audit-implementation`. Its parent is
`48684630ef1876f64531f70ffa0312bab4d1a063`. Staging was verified locally and remotely at
`f1a6bb85588ba044fc2ca50e3e9f41eb2f13b6f9` and remains separate. This is a preservation
checkpoint for all task-owned source, tests and documentation, including incomplete changes;
it is not an accepted backend delivery or authorization to merge into staging.
Verified author/committer: `Jaden <changherng.jc@gmail.com>`.

The saved changes include the previously accepted authority/MCP/media/recovery foundations,
the Dream correction, provisional canonical-child changes, and unfinished separate-owner
artifact adaptation. Existing CI evidence-step ordering and attachment migration-retirement
documentation belong to those foundations. No dependency, lockfile or checker-setting change
was made for this handoff. Raw QA files and ignored build/cache files remain outside Git.
The record below is sufficient to resume from the branch without the earlier chat; local
QA paths provide additional raw evidence when available.

### Latest frozen evidence

QA root:
`/private/var/folders/9f/pzdn55jj0gj7yrqrqx405lk80000gn/T/nanobot-a3-completion-20261112-iqd9sf_h`.
Paths below are relative to that root. Candidate `handoff-paused-20261112-final` freezes
**2,714 files**. All frozen bytes matched the live worktree before handoff-document edits.
Python **3.13.2**, **183 unchanged distributions**, BasedPyright **1.40.1**; interpreter:
`/Users/jaden/Documents/github/nanobot/.venv/bin/python`.

| Check on the latest frozen combined source | Result |
| --- | --- |
| Python syntax | **1,468 files parsed**, no syntax failures; this does not qualify runtime behavior. |
| Whitespace | `git diff --check` passed. |
| Full Ruff | **FAIL: 21 findings**, exit 1: I001 × 7, F401 × 8, F811 × 6. |
| Full-source strict | **FAIL: 1,078 errors / 545 files**, exit 1. |
| Global signature comparison against accepted 962 baseline | **FAIL: 116 introduced, zero resolved**, across all files. |
| Comparison against earlier paused source (1,010) | **98 introduced, 30 resolved**; this is diagnostic history, not a replacement baseline. |
| Combined tests, installed artifact and fresh review | No new broad test suite, artifact qualification or fresh integrated review on this final combined checkpoint. |

The 116 introduced signatures are distributed as follows; do not omit unchanged files
from comparison merely because the current repair does not edit them.

| File | Introduced signatures |
| --- | ---: |
| `nanobot/agent/child_sessions.py` | 8 |
| `nanobot/agent/delegation.py` | 14 |
| `nanobot/agent/turn.py` | 1 |
| `nanobot/extensions/crawler.py` | 3 |
| `nanobot/session/artifact_grants.py` | 88 |
| `nanobot/session/artifact_reader.py` | 2 |

Ruff findings are in `delegation.py` (7), `loop.py` (1), `turn.py` (2),
`artifact_grants.py` (2), `test_delegated_child_events.py` (3), and
`test_delegation_child_authority.py` (6). No repair was attempted during the handoff.
Receipts: `handoff-paused-20261112-final-{context,hashes,check-summary,syntax}.json`,
`handoff-paused-20261112-final-{strict,lint}-receipt.json`, and
`handoff-paused-20261112-final-{integration,prior-pause}-comparison.json`.
The matching patch and Git status are retained alongside them.

The immutable 962-error report is
`/private/tmp/nanobot-a3-resume-20261112-hovb2rb1/integration-strict.json`, with its
`integration/` source in the same directory, originating from integration
`685bee9279c0bbb962ef78344428afeaba121066`. Follow
[the full-source regression rule](../../migration/p6-error-attribution.md) and
`scripts/check_type_regressions.py`. Never absorb the 116 signatures into a refreshed
baseline. On a different host, reproduce baseline and candidate under one recorded
dependency environment; historical totals alone are insufficient.

### Completed work and evidence limits

**Foundations:** `foundations-corrected-2` retains its bounded **CLOSED_PASS**:
13,849 tests passed, 41 skipped, six subtests; full Ruff/diff passed; strict 962 / 544
files with zero introduced signatures. Acceptance covers AUTH-01, IR-MCP-01 and
DR-F01/02/03/04/05/06 only. Preserve those owners and assertions. It does not accept the
later child/artifact source, close P6, or qualify the release.

**Dream:** correction round 1 is finished and independently verified **in isolation**.
This round changed `nanobot/agent/memory.py`, `nanobot/utils/dream_operation.py`,
`tests/agent/test_git_store.py`, and `tests/agent/test_memory_store.py`. All 19 introduced
Dream diagnostics were repaired. New regressions cover directory obstruction and an
actual issued `WriteFileTool` racing restore across distinct owners; the existing stale
registry test remains. Root candidate `completion-dream-r1` passed **218 tests**, full
Ruff/diff and unchanged-dependency global comparison: strict **962 / 545 files**, zero
introduced, with 19 resolved against its correction input. This supersedes the older
216-test/19-error Dream status below. Combined acceptance and fresh review remain pending.
See `current-completion/dream-root-precheck.json`, `completion-dream-r1-*`, and
`current-completion/dream/{checkpoint.md,commands.txt,source-hashes.json}`.

### Canonical child backend: current partial state

The actual canonical child session/run/history path is present. Exact original-chat
approval resumes the same child source with a new run. Production 20-line overwrite,
sibling approval isolation and duplicate refusal have focused positive evidence. Separate
limits now default to depth 2, two immediate children, four descendants per root, and four
children gateway-wide via `agents.defaults.delegation_tree`. Malformed recovery scans fence
new admissions. Child Stop uses one bounded `asyncio.wait` deadline, retaining resistant
task handles and uncertain capacity; awaited child tasks register under the parent
ExecutionOwner while background children remain independent.

| Acceptance criterion | Remaining work at stop |
| --- | --- |
| AC1 canonical source/run/history | Partial: custom-store refusal, adverse persistence and owner-close proofs remain. The nested planner test now verifies separate child history, permitting only delegated bootstrap while retaining the parent legacy-save prohibition. |
| AC2 authority/private holds | Partial: complete live revocation, disabled-capability and captured-MCP lifetime barriers; preserve exact-effect admission and private parent task holds. |
| AC3 original-chat exact approval | Partial: positive same-child/new-run and sibling cases pass; denial, stale source/ancestry, revocation, Stop and duplicate-dispatch races remain. |
| AC4 ordered same-child follow-up | Open: no child control entrypoint is wired. Reuse `_SourceActivityAdmission.submit`, owner-registered `admit_pending_activity`, and the existing `PendingFollowupRecord`/writer admission-consumption journal. |
| AC5 background result delivery | Open: implement durable completion outbox to the exact captured parent source lifetime and route, with restart/consume/stale-parent deduplication. |
| AC6 finite budgets and recovery | Partial: three-root 2+2+1 capacity regression now passes. Valid uncertain records still need explicit owner-evidence settlement; list/read/persistence faults and shared gateway tree composition across AgentLoop instances need proofs. |
| AC7 original-chat subtree Stop | Partial: root cancellation/join still precedes `stop_children`; fence subtree admission first. Durable stopping projection, control epoch/source-lifetime guard and resistant root-owner quiescence remain unfinished. |
| AC8 typed projection and history | Partial/open: basic `DelegatedChildEvent` remains; complete source/run/state ordering, capacity/canStop/owned-work/epoch, durable-before-event delivery, reconnect hydration and canonical full-history accessor. |
| AC9 artifact handoff | API adaptation is unfinished and not integrated into actual child writer/runner settlement; see below. |
| AC10 regression and acceptance | Incomplete: production barriers, closest broad suite, full lint, global zero-new type ratchet and fresh combined review remain. |

Child owner report: `QA/current-completion/children/worker-checkpoint.md`; ownership is
released and its commands have exited. Focused evidence includes earlier **21 passing**
cases, then **12 capacity/history cases**, **12 child Stop cases**, and the final **one-case
parent task-registration probe**. Counts overlap and must not be summed. Root independently
passed the exact approval probe on `completion-child-approval-interim`; that combined command
also failed the then-stale planner fixture. The worker subsequently corrected that fixture
and passed it, but root has not replayed the final combined package. Earlier failing runs
remain recorded; no claim is made that every prior failure is now resolved.

The worker used the fixed interpreter with `PYTHONDONTWRITEBYTECODE=1`, exact candidate
`PYTHONPATH`, explicit `-c <source>/pyproject.toml` and short unique `/tmp` basetemp paths.
Useful continuation targets are `tests/agent/test_delegation_child_authority.py`,
`tests/agent/test_delegation_child_controls.py`, `tests/agent/test_delegation_child_sessions.py`,
`tests/agent/test_delegation_gateway_capacity.py`, `tests/agent/test_delegated_child_events.py`,
`tests/agent/test_migration_source_engine.py::test_source_nested_planner_stream_does_not_leak_into_root_history`,
and `tests/config/test_delegation_tree_config.py`. These are starting points for the next
agent, not tests run during this documentation handoff. The child checkpoint records exact
commands/exits; its pytest output was in tool results rather than separate raw log files.

### Separate-owner artifacts: unfinished API and integration seam

Current WIP modifies only `nanobot/session/artifact_grants.py` and
`nanobot/session/artifact_reader.py` in this package. No completed artifact regression
evidence is available; these modules introduce **90** type signatures in the frozen source.
Do not treat the API as accepted or the child runtime integration as present.
The final artifact worker report was unavailable when its report-only turn was stopped;
the status here comes from the frozen source and root receipts, without inferred test
completion. Artifact write ownership is revoked and must be reassigned on continuation.

The proposed API is:

```python
binding = ArtifactDirectChildBinding.capture(admission, parent_reader, child_writer)
grants = ArtifactGrantSet.issue_downward(
    parent_reader, child_writer, refs, child_binding=binding
)
collector = ArtifactHandoffCollector(child_reader, child_binding=binding)
receipt = collector.prepare_after_completion(stop_reason=stop_reason)
# Only after exact canonical run/source settlement, if a receipt was prepared:
if receipt is not None:
    receipt.seal_after_settlement(run_store=run_store, source_store=source_store)
```

Capture the actual live parent/child writers and durable direct-child relation, exact
source/run/stream identities and private invocation nonce. A typed `ChildAdmission` alone
is not live authority. Preserve private BLOB → SOURCE lock order and actual
`ReadFileTool.verify_artifact_evidence` checks. Prepare the candidate before child owner
close; carry it in a typed internal `TurnResult` field; seal in
`AgentLoop.execute_child_turn` after `execute_turn` returns and the exact RunRecord is
COMPLETED with a healthy closed source: no execution lease, recovery marker or unresolved
execution links. Never validate an expired child owner as live. Upward acceptance is once
only after the durable parent tool result. Approval resume cannot reuse the old expired
parent reader/grants. Keep the explicit background-plus-artifact refusal.

`runner.py` still relies on the inline `parent_stream_id` artifact path while the canonical
child now owns a root writer of its own. No binding capture, typed candidate propagation,
post-settlement seal or real positive/negative child integration test is wired. Give the
artifact modules one owner and runtime/factory integration another owner only with an
explicit API handoff; integrate against actual writers without mutual waiting. Relevant
existing tests are `tests/session/test_artifact_{grants,handoff_boundary,reader}.py` and
`tests/tools/test_artifact_origin_boundary.py`. Preserve inline same-owner behavior and
sibling/transitive/forged/replaced-source/private-location/range/replay refusals.

### Resume order and outstanding acceptance

1. Sync and inspect the implementation branch, verify identity/refs, preserve this WIP,
   and read this entry plus the canonical-child/client specifications below. The user's
   implementation preference remains fresh-context **Luna Max workers through the
   orchestrator**, with bounded ownership and independent evidence-based acceptance.
2. Finish the unfinished backend: AC1/2/3 adverse paths, ordered input/outbox, recovery,
   root Stop, typed projection/hydration/history, and artifact API plus runtime integration.
   Fix the recorded lint/type regressions without weakening tests or accepted foundations.
3. Freeze the combined source under the same dependencies; run affected child, artifact,
   Dream and command suites, relevant broader regression coverage, full Ruff/diff and the
   unchanged full-source global ratchet against the immutable baseline. Obtain one fresh
   bounded Sol Max integrated review, then root PASS/CORRECT/REJECT. Dream isolation alone
   cannot close the backend package.
4. Only after backend protocol settlement, implement WebUI/TUI original-chat activity,
   sibling approval, Stop, usable background composer and optional read-only detail/return
   preserving parent draft/scroll/focus. Android uses the same responsive UI contract;
   browser/PTY interaction evidence remains pending.
5. Qualify the matching final installed macOS artifact and stopped-copy/restart behavior.
   The earlier macOS wheel PASS applies only to `children-before` (wheel SHA-256
   `d2e393faed74c31217bed8266d83570e19121a662c098fa997a25d2ac0e9df93`), not these changed
   bytes. Linux/Android evidence retains its prior source scope. Whole-tree supervisor
   qualification, full A3/P6, release and staging merge remain open.

External continuation contracts: `current-completion/{child-contract,dream-contract,
artifact-contract}.md` and original `contracts/{canonical-children,child-spec,child-client,
dream-correction}.md`. `current-completion/graph.json` and `acceptance-ledger.json` record
the stop. Their earlier complete-first instructions are superseded. Resume unfinished work
only when the user starts the next task; no automatic continuation is requested.

## Earlier paused checkpoint (2026-11-12; superseded)

This historical record predates the latest child, artifact and Dream changes. Its counts,
uncommitted-Git state and resume instructions do not describe the checkpoint above.

**User-requested pause.** Both implementation owners are released; no implementation, test, gateway or UI stage is running. A3 remains
unfinished and unaccepted; resume only when asked. Frozen `A3-PLAN-2026-11-11`, architecture and future coding-harness/session-based
SuperAgent direction are unchanged.

### Frozen candidate

On `codex/staging-audit-implementation`, HEAD is `48684630ef1876f64531f70ffa0312bab4d1a063` and staging is
`f1a6bb85588ba044fc2ca50e3e9f41eb2f13b6f9`; live remote refs match. Index is empty; corrections are local/uncommitted, with no commit,
push or merge. Continue from this worktree or saved snapshot: remote HEAD alone lacks WIP.

Verified identity: `Jaden <changherng.jc@gmail.com>`.

`paused-current` freezes 2,712 files. Source/test/other-doc bytes matched frozen hashes before this handoff-only edit. Environment:
Python 3.13.2, 183 distributions, BasedPyright 1.40.1; dependencies and checker settings unchanged. Root receipts/hashes are under
`/private/var/folders/9f/pzdn55jj0gj7yrqrqx405lk80000gn/T/nanobot-a3-completion-20261112-iqd9sf_h/`.

### Root evidence and accepted foundation

Current critical replay passed **24 tests** (child-authority, permission-owner and five immutable child/Dream probes). No full 14k
broad rerun occurred on the incomplete combined child package. Full Ruff fails with **14 child production/test findings**: I001 (4),
F401 (7), F811 (3).

Full-source strict exits 1 at **1,010 diagnostics/545 files**, vs baseline 962; global all-file ratchet fails at **48 introduced, zero
resolved** (19 Dream, 29 child). Do not refresh the baseline or compare edited files alone. Strict/comparator receipts and Ruff reports
are in the paused-current QA bundle.

Frozen `foundations-corrected-2` remains **CLOSED_PASS** (13,849 passed, 41 skipped, six subtests; full Ruff/diff pass; strict 962/544,
zero introduced/resolved vs immutable `685bee92`). Strict debt remains; P6 is incomplete. This does not accept paused-current or A3.
AUTH-01, IR-MCP-01 and DR-F01/02/03/04/05/06 remain frozen dependencies.

Root macOS PASS applies only to immutable `children-before` wheel (SHA-256
`d2e393faed74c31217bed8266d83570e19121a662c098fa997a25d2ac0e9df93`): 544-file source/bundle parity, installed CLI/process checks, 36
stopped-copy/restart/revocation checks, unpatched foreground gateway and static/model routes, offline SSE, and exact two-message
history after clean restart passed. Requalify the matching final artifact/stopped-copy restart. Whole-tree supervisor was unavailable;
Full refused authority. No Once/Full whole-tree, physical Android, changed-APK/device equivalence, signing, deployment or release
claim. Linux/Android evidence retains its prior source scope.

### Canonical child correction — partial

Canonical AgentLoop child path/source/run and risky-write blocking are present, with private parent Once-task reference holds. The
green 24-test replay does not establish AC1–AC10; AC2 is partial. Live revocation, disabled-capability and captured-MCP lifetime
barriers remain open.

AC3 is scaffolding only: `ChildApprovalResume` and tree/manager helpers are not wired to `AgentLoop.execute_turn` or initial
settlement. Exact-parent typed approval, same-child/new-run resume, denial/stale/duplicate handling and retained `waiting_approval`
capacity are unproven; blocked settlement may fail/release instead of waiting. Wire and validate this path before acceptance.

Earlier focused child report: **41 passed/10 failed**, not rerun at pause. Worker attribution (nine direct-tree fixtures missing
canonical parent owner; one synthetic event-path mismatch) is unaccepted; the resumed worker lacked the exact command. Keep failures
visible. AC4–AC10 remain: ordered follow-up; durable exact-parent result outbox; global/root/depth limits and uncertain capacity;
bounded subtree Stop; durable typed projection/hydration/full history; sealed cross-owner artifact handoff; integrated
fault/barrier/regression coverage.

### Dream correction — worker complete, root acceptance pending

Isolated seven-file Dream overlay passed **216 tests** (212 focused plus four immutable root probes), full Ruff and source/dependency
parity. Strict is **981 diagnostics/545 files** vs 962: 19 introduced, zero resolved; ratchet fails. Combined replay also passed
Dream's four root probes. Repair 19 signatures and add the directory-obstruction and real write-vs-restore barrier regressions. Actual
`cmd_dream` with production MemoryStore/GitStore persistence errors passed via offline `process_direct` adapter; live
AgentLoop/provider cancellation and power-loss behavior remain unqualified.

### Clients and resume order

WebUI/TUI work has not started: original-chat task/result/sibling approval/Stop, background composer, read-only detail/return
preserving draft/scroll/focus, typed TUI approval and shared responsive Android UI remain. Browser/PTY evidence is pending; offline
provider fixture is prepared and wire-proven only.

Resume from this worktree after checking refs and paused-current hashes; preserve WIP and accepted foundations. Continue fresh-context
Luna Max owners: wire child AC2/AC3; address known failures, 14 Ruff findings and 29 type signatures; repair Dream's 19 signatures/two
gaps. Finish backend AC4–AC10; freeze combined source and run affected broad/command suites, full Ruff/diff and unchanged-dependency
global zero-new strict. Then implement clients and verify browser/PTY; qualify matching installed macOS artifact/stopped-copy restart;
obtain fresh bounded Sol Max integrated review and root PASS/CORRECT/REJECT. No A3, merge-readiness or release acceptance is claimed.

Resume evidence: `paused-current-{context,hashes,package-inventory}.json`; `evidence/canonical-children/pause/pause-checkpoint.md`;
`dream-correction/root-pause-checkpoint.json`. `contracts/dream-resume-correction.md` is prepared but not dispatched; original
`contracts/canonical-children.md`, `child-spec.md` and `child-client.md` remain authoritative.

## Authority, MCP and media foundation checkpoint (2026-11-12)

The prerequisite correction graph is closed with **PASS** under frozen
`A3-PLAN-2026-11-11`. This checkpoint supersedes earlier pending statuses for
**AUTH-01, IR-MCP-01 and DR-F02/03/04/06**. The implementation branch remains
`codex/staging-audit-implementation` at `48684630e`; these latest corrections are
local and uncommitted. Repo-local Git identity is `Jaden <changherng.jc@gmail.com>`.

- Exact approval now issues sealed, volatile authority for one concrete dispatch
  attempt. Repeated/concurrent effects cannot reuse it; failure does not refill it.
  Independent sibling approval IDs retain their original-chat addresses. Child
  integration must reuse this admission owner rather than invent another registry.
- Each model request captures immutable concrete tool/schema bindings. Independent
  child/background work borrows narrowed holds through settlement. Reload/revocation
  never redirects an old request to a replacement; every visible replacement owns
  its exact generation before the next await. Dream filters preserve live admission
  without retaining their local issuing registry indefinitely.
- Media retention shares strict owner-declared inventory and publication/deletion
  fencing. Invalid or incomplete evidence blocks every destruction branch; exact
  durable pending custody protects unpublished bytes. Canonical source publication
  participates through both `JsonlSessionStore.persist_source_records` branches.
  Valid retained display references protect bytes, real orphans can reclaim, and
  trusted workspace moves retain inode-bound declarations. The first-release route
  retires the advertised attachment migration instead of adding legacy rewriting.

Root validated an immutable 2,710-file candidate under the unchanged Python 3.13.2
environment with 183 distributions: **13,849 passed, 41 skipped**, six subtests and
59 warnings; full Ruff and whitespace checks passed. Full-source BasedPyright
reported **962 diagnostics across 544 files**, with **zero introduced or resolved
diagnostic signatures** against immutable integration `685bee9279c0bbb962ef78344428afeaba121066`.
This passes the bounded repair ratchet; strict checking remains nonzero and P6 is
not complete. The recorded broad command covers the affected backend, CLI/utils
and WebSocket suites, rather than every optional channel or device matrix.

Independent review found four defects, all corrected and rechecked. Its 280-test
replay and final 13-node gateway recheck passed. Additional compatibility fixes
preserve existing assertions, including stale approval with no foreground activity,
history budgeting, temporary-chat privacy and signed historical-media reads. The
last correction restores the test directory accessor's real creation contract;
it changes no production behavior or test assertions.

Continue with **AUTH-02 and CHILD-01..06**, then **UI-01/02**, using the canonical
child and client specifications below. Source identity, mutation uncertainty,
sampled MCP ownership, exact approval and media publication are accepted dependencies
and must survive that work. Backend projection/control schema precedes client edits.
The user selected **GPT-6 Luna Max workers for subsequent implementation and
corrections** to conserve quota; keep evidence-based independent Sol Max review.
Two later fixture corrections have already used Luna Max successfully.

No final child/client, macOS selected-mode/artifact, changed Android-artifact or
release qualification is claimed. The overall correction task remains active;
staging merge and Git publication are separate actions. Current raw snapshots,
contracts, commands, hashes, final review and root acceptance remain outside Git at
`/private/var/folders/9f/pzdn55jj0gj7yrqrqx405lk80000gn/T/nanobot-a3-completion-20261112-iqd9sf_h`,
with `root-foundation-final-verdict.json` and `foundation-review/recheck-2/review.md`
as the foundation acceptance records.

## Recovery foundation checkpoint (2026-11-12)

**DR-F01 and DR-F05 have a bounded implementation PASS. Full A3 acceptance remains
open.** The accepted `A3-PLAN-2026-11-11` is unchanged. At this checkpoint the recovery
changes are local and uncommitted on `codex/staging-audit-implementation`, above
`48684630ef1876f64531f70ffa0312bab4d1a063`. Staging remains `f1a6bb85`; no merge or
publication was performed. Repo-local Git identity is `Jaden <changherng.jc@gmail.com>`.

- **DR-F01.** Concrete first-party write/edit/patch and binary-publication paths retain
  bounded mutation evidence, including mkdir, cancellation and tracking failures.
  Registry/result copies preserve trusted facts; plugins cannot supply them. Mandatory
  result persistence atomically fences unresolved effects before the reconciliation
  abort. Tests prove no tail mutation or next model request, and fencing after reopen.
  Historical mutation errors without evidence remain uncertain. No readback proof is
  claimed and no automatic replay or general reconciliation bypass was added.
- **DR-F05.** Typed source state comes from the admitted execution owner and follows
  delayed process work. Store/key/incarnation/generation are rechecked under final
  acquisition locks before lease publication, without source creation or fallback.
  Retained-result access checks the same origin; journal CAS compares original stored
  bytes. Missing/stale origins refuse, originless records remain inspectable, and an
  interrupted admitted continuation remains unknown. Revision advances remain valid.
- **Integration corrections.** Admitted body/cleanup errors preserve their original
  identity. Bus completion has one notifier; only a typed source rejection before the
  bus scope receives early settlement. A valid origin can also require an existing
  session. Definition-only SDK close, mandatory-close failure assertions and CI strict
  evidence-step ordering now follow their existing contracts.

### Verification

All final checks used one frozen candidate, with 2,705 input-file hashes and the same
183-distribution Python 3.13.2 environment. No dependencies or checker settings changed.

| Boundary | Final evidence |
| --- | --- |
| Affected backend suite | **11,468 passed, 32 skipped, 55 warnings, 6 subtests passed** across agent/session/tools/migration/cron/triggers/WebUI/remote/context/scripts and SDK streaming. This is an affected-suite run, not full release qualification. |
| Correction regressions | 70 focused tests passed. Independent review replayed 23 tests and found no remaining defects in the bounded correction. Existing lifecycle assertions were preserved; new regressions fail against frozen pre-correction production. |
| Lint and source integrity | Full `ruff check nanobot tests conftest.py` and `git diff --check` passed. Final source/test bytes match the frozen tested candidate; the correction changed only its four assigned files. |
| Full-source type comparison | Immutable integration `685bee92` to final: **962 to 962 diagnostics, zero introduced signatures**, 542 candidate files. Original WIP to final: **968 to 962**, six resolved and zero introduced. Strict checking still exits 1 in this environment; this is not strict-zero/P6 completion. |

The initial broad run had nine failures: six notification ownership regressions and
three fixtures needing the new source/admission condition. Those failures are repaired
and covered by the final passing run. The independent review also caught admitted-error
misclassification and the valid existing-session carrier conflict; both are corrected.
The recovery review/correction graph is closed with **PASS** after one correction round.

Raw reports, exact commands/environment, immutable source snapshots, hashes, red/green
logs and `root-verdict.json` are retained outside Git at
`/private/tmp/nanobot-a3-resume-20261112-hovb2rb1`. The final snapshot is
`accepted-candidate`; use `final-broad-receipt.json`, `final-strict-receipt.json` and
`final-integration-comparison.json` for the final evidence rather than earlier runs.

### Continue from here

Next implement **IR-MCP-01** using the sampled binding specification below. Then finish
media retention/publication **DR-F02/03/04/06**, followed by canonical child authority
and the original-chat WebUI/TUI behavior. These packages have not been accepted by this
recovery review. Physical Android, final macOS selected-mode qualification and the other
release gates below remain open. Keep implementation separate from staging.

## Mobile correction checkpoint (2026-11-12)

Resume from `codex/staging-audit-implementation` in the implementation worktree.
The mobile source is in `8153953f40046499e66ff01324b111b049363bd3`.
The correction follows the accepted `A3-PLAN-2026-11-11` on staging `f1a6bb85`.
[Implementation plan](implementation-plan.md)

- **PA-01.** Mobile commands require current scopes and the owner-bound Host and Origin.
  Queued WebSocket delivery repeats the access check. Session deletion requires owner scope and `chat.write`.
- **PA-02 and PA-03.** Cancel uses the durable native pairing fence. The bootstrap joins early handshake results.
  The native broker restricts credential operations to native callers. The bridge exposes typed public status fields.
- **PA-07 and PA-G01.** The server and Android use the fixed v2 proof corpus.
  The loopback data-server test checks one-use recovery, mutation replay and revocation.

### Verification

| Boundary | Evidence |
| --- | --- |
| Python | 247 server compatibility tests and 213 WebSocket channel tests passed. The final import and type edits passed their affected 44 and 4 tests. Mobile-slice Ruff passed. |
| WebUI and Go | 37 WebUI mobile tests passed. Go broker tests, Go vet and the shared WebUI build passed. |
| Android | 20 host unit tests passed. Instrumented Kotlin compiled. The unsigned APK contains the corrected bootstrap and both native ABIs. Its manifest declares minSdk 24. |
| Strict types | The mobile snapshot applies the mobile changes to `a8b3abf5`. It reports zero diagnostics across 541 files against frozen integration `685bee92`. The combined branch comparison resolved seven diagnostics and introduced zero across 542 files. |

At the mobile checkpoint, the combined branch still reported six strict diagnostics and
two Ruff findings in recovery files; the recovery checkpoint above resolves those findings.
The Linux API 36 emulator crashed before boot. Android device execution remains pending.
Full A3 acceptance remains open.

Raw logs, dependency inventories, source hashes and the accepted test cases are under
`/home/jaden/Documents/Utilities/nanobot-custom-worktrees/a3-evidence/mobile-correction-2026-11-12-c0f12f00`.

### Next stage

The recovery checkpoint above completes the next foundation package described here.
Continue with its remaining work order and the specifications below.

The November 11 pause record below describes `c0f12f00` and its evidence.

## November 11 pause record

**Paused by the user on 2026-11-11. This is an incomplete implementation checkpoint,
not a merge-ready or release-qualified result.** Resume only when asked. All previous
worker write ownership has been released; no validation or agent jobs remain running.

Status and failure descriptions in this historical record are superseded by the
November 12 checkpoints above. Its accepted requirements and remaining specifications
still apply; do not repeat already completed mobile or recovery corrections.

This record is self-contained for an agent continuing from the implementation branch.
The architectural direction is settled. Continue the corrections below; do not reopen
the plan merely because the implementation is incomplete.

## Branches, authority and scope

| Item | Identity / instruction |
| --- | --- |
| Implementation branch | `codex/staging-audit-implementation` |
| Original implementation reviewed | `0518de5f5013b77cbc20d2c5b456e973f33c68f5` |
| Frozen staging baseline | `f1a6bb85588ba044fc2ca50e3e9f41eb2f13b6f9` |
| Local integration of staging into implementation | `685bee9279c0bbb962ef78344428afeaba121066` |
| Verified lifecycle correction | `a8b3abf5f32760dd65d0865cfc3023d29be9afa2` |
| Incomplete mobile/recovery changes | The WIP commit adding this handoff; inspect its diff against the lifecycle commit |
| Git author and committer | `Jaden <changherng.jc@gmail.com>`; verify repo-local identity before further commits |
| Existing implementation PR | [#27](https://github.com/JadenChun/nanobot-custom/pull/27); do not merge it |

The user authorized corrections on the implementation branch, then explicitly requested
a pause, Git synchronization and this handoff. Staging remains unchanged. Do not merge
the implementation into staging, deploy, sign packages or operate physical devices as
part of resuming ordinary correction work. A later publication request should be
handled in its stated scope.

Read [implementation-plan.md](implementation-plan.md),
[v1-product-contract.md](v1-product-contract.md), [decisions.md](decisions.md),
[design.md](design.md), [reference-design.md](reference-design.md),
[future-harness-extension.md](future-harness-extension.md) and
[plan-review.md](plan-review.md). **A3-PLAN-2026-11-11 is authoritative.** The archived
implementation packets contain superseded requirements; apply the current plan's
explicit decisions. Existing frozen architecture records and the historical 136-finding
audit ledger were not edited to imply implementation acceptance.

Key settled decisions:

- Linux and macOS are executors; Android is a remote client. Linux/Android prior
  qualification is user-confirmed and retains its original source/artifact scope.
  macOS qualification remains. Windows is outside this selected matrix.
- This is the first release. No legacy migration, forced re-pairing, historical executable
  reader or two-APK upgrade requirement should be added.
- Normal subagents use canonical sessions/runs. The original chat mediates follow-up,
  approval and Stop. Awaited work is the default; explicit background work is allowed.
  Optional child detail is read-only, with return to the original chat.
- Full/Once consent is volatile. History, restart and saved metadata never restore it.
  Uncertain effects retain logical ownership/capacity until reconciliation.
- Future coding-harness extension must remain possible. Future SuperAgent coordinates
  normal sessions, independently of subagent behavior; task associations consume no
  child depth/capacity. Do not add a new router or all-provider semaphore now.

## What has actually been verified

| Work | Evidence and limits |
| --- | --- |
| Staging integration | Eight conflicts and one test adaptation resolved; latest staging behavior retained. Worker: 470 backend passes, 15 host/native skips, 60 frontend passes, scoped lint/type checks. Coordinator independently replayed 101 relevant backend tests. Frozen strict comparison: 962 to 962 diagnostics, 541 files, zero introduced signatures. Integration only. |
| PA-05 / PA-06 cleanup | SDK callers join one shielded close attempt. Failure fences admission and permits retry; direct SDK and composition retain exact unfinished closers, skip settled resources and preserve failures. Coordinator: 109 lifecycle tests passed. Lifecycle-only frozen full-source comparison: 962 to 962 diagnostics, 541 files, zero introduced signatures. Saved separately in the lifecycle commit. |
| Mobile scope boundary, intermediate snapshot | New production owner/IPC/edge/router/HTTP regression: 10 failures to 11 passes; coordinator independently replayed all 11. Later edge/proof encoding edits were **not** retested. Not final mobile acceptance. |
| Proof format, intermediate snapshot | Python fixed-corpus plus existing proof tests: 47 passed. Initial red was a missing decoder API at collection. Kotlin consumption remains incomplete/unverified. |
| Pairing restart, intermediate snapshot | Approved-but-unissued restart regression: 1 failure / 5 passes to 6 passes. Broader restart/store compatibility checks remain. |
| Shared TypeScript mobile runtime | Correlation/disposal suite: 4 failures / 13 passes to 17 passes. Native/bootstrap correlation remains unfinished. |
| Go broker baseline | Before native corrections, actual darwin/arm64 `GOTOOLCHAIN=go1.27.3 go test -mod=readonly ./internal/broker` passed in `mobile/native-network`. No Go source edits. |
| TUI baseline | Before child-client edits, pinned Bun 1.3.13: type check passed; 211 tests passed, 0 failed, across 22 files. No TUI correction has started. |

**These are snapshot-specific results. The final WIP tree has not passed an integrated
test suite, strict diagnostic comparison, frontend build/lint, native build or fresh
independent review.** Strict checking already has debt; a bounded repair PASS is not P6
or release completion.

## Immediate WIP hazards

1. Android production compilation currently fails in
   `mobile/android/app/src/main/java/ai/nanobot/mobile/network/CustodyStore.kt:577`:
   unresolved `OsConstants.O_DIRECTORY`. No native tests ran green at the latest snapshot.
   The local log named `native-green.log` is a **failed** prospective run, not a pass.
2. `mobile/mobile-bootstrap.js` is unchanged. Its Cancel handler does not invoke the new
   native `pairCancel` fence; its early handshake-result correlation is also unchanged.
3. New `java.util.Base64` use in `DeviceProofCanonicalizer.kt` needs review against
   Android minSdk 24. Do not raise minSdk or relax the proof encoding to avoid the issue.
4. Filesystem recovery is partial and has no post-edit green run. The new fact acceptance
   code accesses `.name` on an `object`, which likely needs a proper concrete type guard.
   Patch/binary/notebook mutation coverage and trust/failure-path checks remain.
5. DR-F05 process origin identity has not been implemented. Child/approval, sampled MCP,
   media retention and client corrections have not started.

Do not fix these while merely reviewing or syncing this paused checkpoint. A continuation
request authorizes the next implementation work.

## Progress and ready work order

The review established 23 outstanding defects before these corrections: 17 P1 and 6 P2.
PA-04 is a separate qualification/capability gate; PA-08 was already repaired by the
staging integration. Do not count either as an outstanding newly reproduced defect.

| Findings | Status at pause | Next package |
| --- | --- | --- |
| PA-05, PA-06 | Verified bounded correction | Preserve the separate lifecycle commit |
| PA-01, PA-02, PA-03, PA-07; prerequisite PA-G01 | Partial mobile/native/server WIP | Finish mobile, then run final compatibility/ratchet checks |
| DR-F01 | Partial effect/source WIP; no green validation | Finish recovery foundation |
| DR-F05 | Not implemented | Same recovery foundation / shared admission owner |
| IR-MCP-01 | Not implemented | Sampled registry/MCP binding ownership |
| DR-F02, DR-F03, DR-F04, DR-F06 | Not implemented | Media retention/publication and migration retirement |
| AUTH-01, AUTH-02; CHILD-01 through CHILD-06 | Not implemented | Canonical child execution and authority |
| UI-01, UI-02; source-only LIMIT-01 through LIMIT-03 | Not implemented | Original-chat WebUI/TUI projections and navigation |

Finish the mobile and recovery foundations first. Then sampled MCP and media ownership
can proceed on disjoint files after shared source ownership is released. Canonical child
work depends on source identity, effect fencing, MCP borrowing and safe source publication.
Client work follows a settled child projection/control interface. Finish with integrated
checks and one fresh independent review of the combined changed behavior.

Use one writer for overlapping source/runner/registry/turn/context/gateway files.
Previously active workers are stopped; do not assume an old worker still owns a file.
If using the orchestration skill, route this resumed task afresh rather than reviving a
closed graph implicitly. Keep the core small and use existing owners.

## Mobile correction specification

Current changed areas are `nanobot/remote/{auth,device_proof,edge,ipc,operations,pairing}.py`,
the scope/origin gates in `nanobot/webui/{gateway_endpoint,inbound_commands,ws_http}.py`,
`webui/src/lib/mobile-runtime.ts`, related tests, and the Android network/custody classes.
New Android classes are `NativeRequestCorrelator.kt` and `PendingAttemptFence.kt` with
host test files. `app/build.gradle` adds test-only `org.json:json:20240303`.

- **PA-01:** Enforce required action scope and owner-bound Host/Origin at the actual
  production shared dispatcher. A media.read-only grant cannot hydrate chat. chat.read
  or media.read cannot mutate sidebar/session state. Require current grant checks at
  admission, before effects and before private delivery; unknown operations refuse.
  Preserve the current operation registry and newly added HTTP mobile revalidator.
- **PA-02:** Cancel must durably fence the native pairing attempt before queued collect,
  poll or token writes can finish, including when the operation executor is blocked.
  Compare attempt generation atomically with custody writes; QR/manual restart must
  share the fence. Persistence failures are explicit. Wire the existing JS Cancel UI
  to this native boundary; token absence alone is not cancellation proof.
- **PA-03:** TS/native must reserve or join early completions before a returned request
  ID can be registered. Bounds include launch count, pending/tombstone count and bytes.
  Keep native-only credential-response classification through timeout/cancel and late
  terminal delivery. Unmatched/late raw credential bodies never reach WebView. Finish
  bootstrap handshake correlation as well as the shared runtime and Kotlin correlator.
- **PA-07:** Persist non-bearer approved-but-unissued device binding/expiry. Restart
  requires fresh proof. Never persist/replay an issued bearer response. Cancel, expiry
  and revocation must remain effective across restart and lost delivery.
- **PA-G01:** The v2 schema/corpus was fixed before detailed proof work. Python and Kotlin
  must consume the same independent expected bytes/signatures/rejections, not derive
  expected data from the implementation being tested. Pin P256/ECDSA-SHA256, SPKI/DER,
  exact typed sorted compact UTF-8 JSON, strict unpadded base64url, valid scalar Unicode,
  and signed-long positive integer bounds. Existing padded-base64 test fixtures need
  alignment; never relax production verification to preserve obsolete fixture encoding.

Fixed artifacts, unchanged at pause:

| File | Git blob | SHA-256 |
| --- | --- | --- |
| `mobile/contracts/device-proof-v2-schema.json` | `a13c2461d0762e33f1660f74edfa313f2eba6ca7` | `8afc76b8f2b75f3c23e0c19154734fd82c11a78ad3b116614c492a0700f0959c` |
| `mobile/contracts/device-proof-v2-vectors.json` | `e55a0a97c0a2c127a66daa09fb1cce0b2e71105c` | `270d3d2715258bb161c97eba520e445068f8fed6f06c01bc57076e4e18d4abfd` |

Finish full Kotlin verification/rejection/replay corpus consumption and the compatibility
suites; test actual native producer behavior, not only a disconnected canonicalizer.
Host JVM tests do not establish physical Android Keystore or device qualification.

## Recovery foundation specification

### DR-F01: preserve mutation uncertainty and stop execution

Observed trigger: a real edit changes bytes, then file-state/touch tracking raises.
Generic retry feedback lets a repeating offline provider edit again (`x` becomes `xxx`
instead of `xx`), while source incorrectly appears settled. This is not unconditional
runner replay.

The partial WIP adds `tools/filesystem_effect.py`, write/edit instrumentation, private
`ToolOutcome`/`KnownToolResult` propagation, source fields and an atomic recovery marker.
It must still be finished and tested:

- Only concrete first-party mutation tools can mint immutable bounded effect facts:
  operation, resolved target, disposition, reconciliation flag, intended/observed hashes
  and byte counts. Never duplicate whole file contents or accept plugin/text authority.
- Mark effect entry before the first mutation, including mkdir. A write that raises
  after entry is uncertain unless permitted readback proves the result. Tracking
  failure retains an applied/uncertain fact. Instrument `apply_patch.py` too; attempted
  rollback is not proof that partial effects vanished. Trace other concrete mutations,
  including binary/notebook writes, before claiming full coverage.
- Preserve facts through every registry envelope normalization and recorded-value copy.
  SourceTurnWriter strips reserved fields from untrusted payloads, binds facts to the
  execution ID and stores them alongside the unchanged raw result. Atomic source
  publication sets recovery-required; matching call/result links do not settle an
  unresolved fact.
- After mandatory result persistence, set `owner.recovery_required` and raise a dedicated
  reconciliation abort. Stop the remaining batch/runner: no retry hint, next model
  request or normal synthetic final source append. Undispatched tail calls may remain
  conservatively unresolved. Distinguish cancellation, policy denial and mandatory
  persistence failure. A prior durable intent fences a failed result write; never retry
  the effect or add a general revoked-write bypass.
- Reopen keeps evidence inspectable and effect/model admission fenced. Existing mutation
  error rows without enough evidence cannot silently become “never started.”

New `tests/session/test_filesystem_effect_recovery.py` red run had five failures, including
the `x -> xxx` behavior and missing mkdir evidence. Two fixture-attribute failures were
corrected afterward but not rerun. `ast.parse` of the nine owned files passed at pause;
that proves syntax only. Next test write-then-raise, record_write/touch failure, pre-effect
denial, cancellation, mandatory result-write failure, tail-call refusal and reopen.

### DR-F05: bind delayed work at final canonical admission

Use existing `SourceState` with typed `RequestContext.source_state` and
`TurnRequest.expected_source_state`, propagated by trusted inbound/deferred/local-turn
construction. Populate only from the admitted ExecutionOwner snapshot. Persist process
origin `(store_identity, session_key, incarnation, generation)`; revision/checkpoint
advances do not invalidate an origin.

Compare all four fields inside `acquire_execution(validate_snapshot=...)` under execution
and namespace locks, **before publishing the lease**. A condition requires an existing
canonical source: no create, alternate namespace or discarded precondition fallback.
Preserve independent workspace/Full permission checks. Delete/recreate/reset barriers
must produce zero replacement-source writes and zero provider calls.

Stale completion remains inspectable with an explicit non-executable disposition; notify
the automation waiter so it neither hangs nor records success. Interrupted dispatching
remains unknown and is never replayed. Process JSON CAS must compare original stored
bytes: adding Pydantic defaults changes serialization and otherwise breaks old rows.
Originless rows remain readable/exportable but cannot continue; never infer their origin
from today's session. Retained-result retrieval must compare origin too, not just key.

## Sampled MCP binding specification: IR-MCP-01

Observed trigger: runner captures an old schema, awaits the provider, MCP reload publishes
a same-name replacement, then execution dispatches that replacement. Fix through current
ToolRegistry, runner and generation lease owners.

- `ToolRegistry.capture_request_tools()` should return owned `CapturedToolBindings` with
  detached definitions, exact concrete captured registry/registrations, generation pins,
  live dispatch/revocation guards and idempotent cancellation-safe release. Capture and
  pin synchronously before preparation/provider awaits; preserve owning-loop atomicity.
- Transfer capture with `RecordedResponse`; preparation, risk target resolution and
  dispatch use it. Logical prepared/fallback/stream attempts use the same definitions.
  Release after response/result/source settlement, including every non-tool/error/cancel
  exit. Binding identity is independent of tool name and survives filtering.
- Exact authorized old binding or explicit refusal is acceptable. Replacement-by-name
  is not. Removal, disable, lost authorization and shutdown revoke immediately despite
  pins; failed replacement preserves the authorized old registration.
- Generation ownership includes sampled and active holds, tracks retired generations
  through shutdown and closes exactly once after the final hold. Preserve refusal to
  retry a terminated wrapper on a replacement.
- Provide `ToolRegistry.borrow_filtered(names) -> CapturedToolBindings` with independent
  narrowed holds before the parent capture releases. A background child needs ownership,
  not raw wrapper references. Borrowing cannot restore disabled/revoked tools.

Use real runner/provider barriers for reload, failed reload, removal/OAuth revocation,
revocation after preparation, filtered/child borrowing, cancellation and shutdown.
Replacement execution count must stay zero and no holds may leak.

## Media retention/publication specification: DR-F02/03/04/06

Observed defects: malformed/missing reference evidence is treated as complete; quota
deletion bypasses incomplete evidence; generated GC sees only WebUI and deletes media
referenced by canonical Telegram source; migration dry-run creates blobs/sidecars.

Use current AttachmentStore/lifecycle and source/WebUI storage owners, not a new service:

- One private `MediaReferenceEvidence` carries hashes, paths, completeness, reasons and
  declared root identities. Evidence alone is not deletion authority: require a live
  nonserializable AttachmentStore fence. Caller `evidence_complete=True` cannot authorize.
- Runtime owners declare actual shared-media roots: every channel/child source and
  workspace namespace; nested followups/reference metadata; retained source, staging,
  rollback/interruption archives; WebUI transcripts/segments/manifests/display/thread
  records; pending uploads/publications/temporary-cleanup retries; other supported
  reference-bearing formats. Custom/pre-contract/unknown coverage refuses destruction.
- Validate exact canonical bytes with the strict source parser; no repairing listings,
  truncated/tolerant reader or regex-as-validation. Then extract known attachment/hash/
  key/media/artifact references, including JSON producer results and platform paths.
  Missing roots, unreadable/traversal/parse/nonobject/version errors, unsupported reference
  forms, incomplete pagination or exhausted budget make the entire inventory incomplete.
- Qualify all evidence before any retention/orphan/quota/blob/sidecar deletion. Every
  branch shares one no-deletion gate and reports reasons/unmet quota.
- Persist per-publication pending pins before commit/adoption/generated bytes are exposed.
  Identical hashes need independent tokens. Release/transfer only the exact token after
  durable source/display publication proof. Failure/uncertain cancellation retains it
  across grace/restart; TTL and volatile UploadRegistry absence are not proof. Keep
  read-only path-inspection helpers pure.
- Lock order: existing leading migration/private-artifact/execution locks, ExecutionOwner
  mutex, namespace files locks in stable order, then media publication/deletion fence.
  Never media -> source, and never await network/model/queue while holding media fence.
  Maintenance revalidates declarations/identities and inventories/deletes under those
  protections; bounded busy/incomplete admission refuses. Publication validates new
  references under the same order: missing asset after deletion must refuse publication.
  WebUI publication/rotation and temporary shared-blob cleanup participate.
- Validate containment/generation without following replacement symlinks. Durable bounded
  deletion intent precedes unlink; blob failure preserves sidecar, sidecar failure retains
  retry ownership. Reopen retries require fresh complete evidence and same asset generation.
  Preview creates no intent.
- **DR-F06 selected route:** retire advertised `attachments migrate` and its mutating
  lifecycle implementation, updating only current attachment docs/tests. Optional retained
  callable explicitly refuses without mutation. Do not build unnecessary legacy rewriting.
  Whole-tree paths/modes/hashes must prove dry-run/declined/refused work writes nothing.

Tests must cover canonical-only/raw-artifact/nested-followup/display/pending-upload roots,
corruption across all branches, quota/grace, both publication/deletion race orders,
namespace insertion, separate processes and partial unlink/reopen retry.

## Canonical children and authority specification

Findings: one exact approval dispatches an identical risky effect twice (AUTH-01);
a scoped child overwrites a real 20-line file without approval (AUTH-02); children have
only summary history and parent-owned tool events (CHILD-01); three roots can run six
children despite gateway cap four (CHILD-02); uncertainty frees slots (CHILD-03); ordered
same-child followup and durable background parent completion are missing (CHILD-04/05);
governing AGENTS is absent (CHILD-06).

- DelegationTree owns ancestry/attempts/reservations; NativeChildBackend is a shallow task
  handle. Actual execution uses `AgentLoop.execute_turn -> _source_execution /
  acquire_execution -> SourceTurnWriter -> AgentRunner -> RunStore`. Never duplicate this
  owner path, use parent `child_stream` as child history, or append final-only exchanges
  after a canonical run. Source/RunRecord must exist before provider/tool dispatch.
- Bind a trusted typed internal delegation adapter with child source/key, new run ID,
  expected SourceState, role tools/runtime/instructions/limits, hooks and narrowed live
  authority. Inbound metadata cannot manufacture it. Normal ContextBuilder supplies
  AGENTS/bootstrap independently of parent conversation, including nested and idle runs.
  Unsupported nonpersistent/custom-store durable delegation refuses before persistence.
- Intersect parent admitted disabled/current capabilities with child role. Read-only
  full-batch denial remains. Worker scoped tools/FileLockRegistry and canonical risky
  policy both apply, resolving targets through the actual child/captured registry.
  Do not inherit parent reusable approval sets or establish a separate read-only MCP
  connection generation. Child pins span background settlement and preserve revocation.
- Root ordinary task-ended retirement must defer private volatile permission retirement
  while descendants still own that task. Stop/replacement/revocation fences immediately;
  final settled descendant retires the deferred hold. Never persist/revive consent.
- ApprovalAdmission indexes exact opaque approval ID and authorized original-chat alias,
  supports sibling requests, and issues sealed volatile single-use exact-effect authority.
  Consume atomically at actual per-call dispatch after preparation/target/live checks;
  duplicate/concurrent calls cannot share it and failure never replenishes it. Child
  approval-required closes that invocation with APPROVAL_REQUIRED, retains waiting capacity
  and projects to original chat without a second parent approval. A valid typed decision
  resumes the same child with a new run, bound to the blocked run/source/current authority,
  transferring the held reservation once. Never hold parent lock awaiting a human.
- Internal parent-mediated send/followup/wait/cancel use a stable admitted tool-intent ID.
  Existing PendingFollowupRecord journal owns ordered busy input before acknowledgment;
  SourceInjectionBinding applies at a safe request boundary and only after model-request
  acknowledgment. Duplicates reconcile, conflicts refuse. Terminal/idle races preserve one
  serialized run or durable next input. Unresolved effects block continuation.
- Awaited result is durable parent tool output only. Background completion uses a durable
  child outbox with stable ID, exact captured parent source lifetime and original route,
  then the parent canonical pending journal. Dedup `_pending_followup_id` before running
  another provider request. Replaced/reset/deleted parent retains result/stale-delivery
  evidence and is never recreated. Restart may reconcile delivery but never replay child.
- Finite configurable defaults: root depth 0/max depth 2; two immediate child runs;
  four root descendants; four gateway-wide child runs, excluding primary turns. Separate
  `holds_capacity` from task/live/terminal flags. Reserved, joining, waiting approval,
  stopping and uncertain work all count. Persist before exposing/releasing transitions;
  unknown retains reservations across restore until explicit owner reconciliation.
  Malformed unresolved scans fail closed; idle/settled history consumes no slot.
- Original-chat Stop fences subtree admission first, through idle ancestors, invalidates
  pending approval/input/launch, then cancels and joins. Use bounded `asyncio.wait` with
  retained resistant handles/observers; `wait_for(gather)` can overrun under suppressed
  cancellation. Unproven effects stay uncertain and keep logical capacity even if physical
  OS/execution lease closes after quiescence. Preserve target-turn guard; children-only
  Stop also binds source lifetime/control epoch and expected root presence or absence so
  a delayed Stop cannot cancel unrelated later work.
- Child metadata/projection needs exact parent/root/child source lifetimes and run refs,
  monotonic run/state sequence, capacity/settlement/owned-work facts, task/result summary,
  actionable approval and durable completion disposition. Persist before event delivery.
- Existing summary-only children remain unchanged readable/exportable inspect-only.
  Do not synthesize tool provenance/RunRecords/consent. Settled summaries use no capacity;
  unresolved ones stay fenced. Executable followup requires newly admitted canonical child,
  linked to retained historical context, not bulk migration.

Artifact handoff is a required dependency, not optional cleanup. Existing grants/receipts
assume the same ExecutionOwner. Adapt narrowly to sealed origin SourceState/stream/
descriptor and exact target child writer/run plus private direct-child nonce. Preserve
reader evidence verification, private-blob -> source lock order, current child read policy
and both source lifetimes. Never rewrite provenance or revive expired/restarted authority.
Seal upward receipt only after successful child source/owner settlement; accept once after
durable parent tool result. Keep awaited downward/upward handoff and existing explicit
background+artifact_refs refusal. Preserve sibling/transitive/forged/replaced/replay/range
refusals; do not create a general cross-session artifact API.

Acceptance needs real factory/loop/offline providers and barriers: canonical source/run
before provider, tool rows only child, governing marker nested/idle, disabled/live revoked
tools, real risky overwrite approval, one dispatch per grant, sibling approvals, three-root
gateway cap, uncertainty across restore, ordered busy/idle followup, background delivery
once/stale parent, launch/approval/resistant Stop, persistence failures and artifact positive
and forgery/replay cases. Do not infer settlement merely from exception/task completion.

## Client projection and behavior specification

UI-01: Stop disappears after parent reply and cancellation lacks projection. UI-02:
disconnected completion/hydration loses canonical child state and TUI history drops rows.
Additional source gaps: writable child detail/no return, absent TUI typed approval, and
activity without meaningful task/results.

- Persist child activity with existing `turn_activity` even without subscribers; hydration
  emits current canonical snapshots independently of goal-status handling. WebUI/TUI
  compare source lifetime, then run_sequence and state_sequence. Identical duplicate is
  idempotent; conflicting duplicate refuses; old run cannot overwrite newer state.
  Disconnection means unavailable, not inferred completion.
- Separate root streaming from `hasOwnedWork/canStop`: composer usable after parent reply,
  background task label/status/result visible, original-chat Stop remains and is bound
  to server-owned source/control epoch. Historical UI facts never authorize live control.
- Multiple child approval requests stay actionable in original chat while root may be
  active. TUI `MessageOptions`/send must carry typed ApprovalDecision; text is not consent.
- Optional child detail uses full canonical history, explicit read-only state and parent
  return; suppress composer/permission and independent lifecycle controls. Preserve parent
  draft/scroll/focus on keyboard/touch return. Do not require public `/child steer/wait`
  management. Use meaningful sibling labels/results without hover/color dependence.
- Android shared UI has the same behavior and touch return. Preserve mobile scope/origin
  checks; any new read/control operation must be explicitly registered with required scope,
  not admitted by unknown action name.

Relevant owners are `webui/src/App.tsx`, `hooks/useNanobotStream.ts`, `lib/types.ts`,
`lib/activity-timeline.ts`, `components/MessageBubble.tsx`, `components/thread/*`,
`tui/src/{app,protocol,transcript}.ts` and existing Python gateway/turn-delivery adapters.
Settle backend event/control schema before client edits. Add deterministic reducer/protocol/
navigation tests, then actual browser/host interaction proof; source/DOM assertions alone
do not establish visual behavior.

## Reproducible validation and environment notes

Obey `AGENTS.md`, `.agent/{design,security,gotchas}.md`, `CONTRIBUTING.md` and
`docs/migration/p6-error-attribution.md`. **Do not run `ruff format`.** Preserve unrelated
work and current staging repairs, including opaque asset cursors, temporary-chat authority,
goal tools, local triggers, clipboard/skills behavior, cancellation and Darwin IPC.

The existing local Python interpreter is
`/Users/jaden/Documents/github/nanobot/.venv/bin/python` (Python 3.13.2), with 183 frozen
distributions, BasedPyright 1.40.1, pytest 9.1.1, cryptography 50.0.1 and Ruff 0.16.7.
Optional `discord`/`olm` are absent: this environment is not full optional-channel CI.
No Python dependencies were changed. Use `PYTHONDONTWRITEBYTECODE=1`, explicit
`PYTHONPATH` pointing at the exact candidate/snapshot, pytest `-c <source>/pyproject.toml`,
and short unique `/tmp` basetemp paths (long macOS Unix-socket paths cause false failures).

Useful focused commands from the repository root, using that interpreter or an equivalent
properly recorded environment:

```sh
python -m pytest -c pyproject.toml --basetemp=/tmp/nb-a3-mobile -q \
  tests/remote/test_production_scope_boundary.py \
  tests/remote/test_device_proof_corpus.py tests/remote/test_device_proof.py \
  tests/remote/test_pairing_persistence.py
python -m pytest -c pyproject.toml --basetemp=/tmp/nb-a3-effects -q \
  tests/session/test_filesystem_effect_recovery.py
python -m pytest -c pyproject.toml --basetemp=/tmp/nb-a3-origin -q \
  tests/agent/test_process_completion_scope.py tests/tools/test_process_completion.py
python -m pytest -c pyproject.toml --basetemp=/tmp/nb-a3-mcp -q \
  tests/agent/test_mcp_generation_retention.py tests/agent/test_mcp_binding_refusal.py \
  tests/agent/test_managed_mcp_access.py
python -m pytest -c pyproject.toml --basetemp=/tmp/nb-a3-media -q \
  tests/attachments tests/webui/test_media_maintenance.py
python -m pytest -c pyproject.toml --basetemp=/tmp/nb-a3-source -q \
  tests/session/test_source_turn_writer.py tests/session/test_source_store.py \
  tests/session/test_execution_owner.py tests/session/test_artifact_grants.py
```

These are starting points, not a replacement for new acceptance tests, affected child/
approval/gateway suites or final broader checks. Run meaningful red tests before behavior
changes where practical, then green and relevant existing tests; do not weaken assertions.

For Python repairs, freeze exact full-source before/after snapshots under the **same**
dependency environment, retain hashes/commands/reports and run:

```sh
python -m basedpyright --outputjson --pythonpath /path/to/python > after.json
python scripts/check_type_regressions.py before.json after.json \
  --baseline-root /absolute/frozen/before --current-root /absolute/frozen/after
```

`before.json` and `after.json` are positional arguments. Enforce zero newly introduced
signatures across all files; a lower total alone does not pass. Current correction baseline
is immutable integration commit `685bee9279c0bbb962ef78344428afeaba121066`, not this WIP.
Do not refresh a baseline to absorb WIP errors. On another environment, reproduce that
immutable source and the candidate under the same recorded dependencies. Bounded repair
PASS never closes P6 while unchanged strict checking is nonzero or required full tests fail.

Frontend/tool setup already performed locally with lockfiles unchanged:

- `webui`: `bun install --frozen-lockfile` with available Bun 1.3.10; focused runtime
  command is `bun run test --run src/tests/mobile-runtime.test.ts`.
- `tui`: pinned Bun 1.3.13. Use
  `npm exec --yes --package bun@1.3.13 -- bun install --frozen-lockfile`, then the same
  prefix for `bun run check` and `bun test`.
- `mobile`: lock requires Bun 1.4.2; Bun 1.3.10 cannot parse it. Use
  `npm exec --yes --package bun@1.4.2 -- bun install --frozen-lockfile`, then the same
  prefix for `bun run build` and `bun run sync:android` when ignored Capacitor inputs
  need regeneration. Verify tracked generated files/locks remain intentional.
- `mobile/native-network`: `GOTOOLCHAIN=go1.27.3 go test -mod=readonly ./internal/broker`.
  Automatic pinned Go download works on this darwin/arm64 host.

Android SDK is `/Users/jaden/Library/Android/sdk`. Default JDK 17 and installed JDK 23 do
not satisfy the Capacitor plugin's exact Java 21 selector. A temporary host Temurin
21.0.13+9 is at `/tmp/nanobot-mobile-jdk21/jdk-21.0.13+9/Contents/Home`. It was downloaded
from Adoptium's mac/aarch64 API; archive SHA-256 is
`bf036b35b071bfb2f54f2c0af8ee8d88afa95ee6eb3dce02c0f94cfc2524d04a`.
If absent, recreate an appropriate host toolchain; do not change the pinned Linux release
tuple in `mobile/toolchain.lock.json` to fit the host.

The last native command, from repository root, was:

```sh
JAVA_HOME=/tmp/nanobot-mobile-jdk21/jdk-21.0.13+9/Contents/Home \
ANDROID_HOME=/Users/jaden/Library/Android/sdk \
mobile/android/gradlew -p mobile/android :app:testDebugUnitTest --console=plain
```

It failed production compilation as described above. Gradle 8.14.3 and the declared
NDK 29.0.14206865 were downloaded. Generated Capacitor/plugin/assets/build directories
and caches are ignored; they are not Git delivery. No signer inputs or device operations
were used. Task-owned idle Gradle/Kotlin daemons were stopped on pause.

## Remaining acceptance and release limits

PA-08 Darwin IPC is already repaired by integration; the actual positive Unix-domain
socket proof passed. Do not regress it or report it as an outstanding candidate defect.

PA-04 is an honest macOS selected-mode qualification/capability gate: ordinary POSIX
execution works, while owner-backed Once/Full whole-tree proof is unavailable and correctly
refuses. Do not flip a platform flag, equate a process-group result with whole-tree proof,
or invent a Linux confinement port to label macOS qualified. Qualify selected modes and
installed artifacts, or retain the explicit refusal/gap.

Installed/current-state rollback, interrupted Dream restoration, complete asset pagination,
matching source/bundle/toolchain artifacts and broader release qualification remain named
gates. They were not established as additional live defects by the bounded correction
analysis. Prior Linux/Android evidence does not automatically qualify changed candidate
bytes. No full release, physical Android or final macOS qualification is claimed here.

After all corrections, independently inspect the combined diff, replay critical production
tests, enforce the frozen full-source ratchet, run relevant broader backend/frontend/native
checks, and obtain a fresh bounded independent review because security, persistence,
concurrency and protocol boundaries changed. Fix findings through their actual owners.
Only then issue an implementation acceptance verdict; staging merge remains a separate step.

## Local evidence retained outside Git

The handoff above does not require these temporary directories, but raw reports, command
receipts, source snapshots, accepted work contracts and recovery backups remain available
on the current workstation:

- Correction QA:
  `/var/folders/9f/pzdn55jj0gj7yrqrqx405lk80000gn/T/nanobot-a3-corrections-20261111-iidnl2o5`
- Original immutable-candidate review:
  `/var/folders/9f/pzdn55jj0gj7yrqrqx405lk80000gn/T/nanobot-a3-implementation-review-20261111-i_7i4455`

Correction QA contains `context.json`, `graph.json`, `progress.json`, `contracts/`,
`evidence/`, `integrated-base/`, `lifecycle-after/` and `pause-backup/`. In particular,
`contracts/child-spec.md` and `contracts/recovery-spec.md` retain the detailed read-only
safety mappings. `pause-backup` preserves the original dirty tree, index state and hashes
before checkpoint commits. Keep raw QA and ignored build/toolchain artifacts outside Git;
do not commit a stack of logs or silently turn this status into architectural approval.
