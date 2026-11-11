# A3 implementation correction handoff

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

The combined branch still reports six strict diagnostics and two Ruff findings in recovery files.
The Linux API 36 emulator crashed before boot. Android device execution remains pending.
Full A3 acceptance remains open.

Raw logs, dependency inventories, source hashes and the accepted test cases are under
`/home/jaden/Documents/Utilities/nanobot-custom-worktrees/a3-evidence/mobile-correction-2026-11-12-c0f12f00`.

### Next stage

Finish the recovery foundation in the implementation worktree.
Start DR-F01 with `tests/session/test_filesystem_effect_recovery.py`.
Prove that an applied or uncertain mutation blocks another model request and survives reopening.
Then implement DR-F05 at final source admission. [Recovery foundation specification](#recovery-foundation-specification)

The November 11 pause record below describes `c0f12f00` and its evidence.

## November 11 pause record

**Paused by the user on 2026-11-11. This is an incomplete implementation checkpoint,
not a merge-ready or release-qualified result.** Resume only when asked. All previous
worker write ownership has been released; no validation or agent jobs remain running.

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
