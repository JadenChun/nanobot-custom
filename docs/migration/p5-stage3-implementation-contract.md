# P5 Stage 3 implementation contract

Status: **root PASS for guarded offline Stage 3 implementation** (2026-10-01),
recorded in the [acceptance checkpoint](p5-stage3-acceptance-checkpoint.md).
P5 closure and all activation gates remain open. Stage 4 is next, requires separate
scope/authorization, and has not started.

### Accepted correction round 1 addendum (2026-10-01)

Root authorized the design below for two independent-review findings, accepted one
bounded correction round after root validation and independent narrow recheck, and
issued guarded offline implementation PASS. Every activation gate remains open.
This addendum does not change D6/D7 or any P3 execution, completion, outbox, receipt
or reconciliation state machine.

- **Observed reconciliation continuity:** strict scheduler v1 remains readable with
  no bridge fields and reads publish nothing. Only explicit stopped-writer bridge
  publication upgrades the scheduler to strict v2. At most 4,096 append-only bridges
  fit inside the existing 1 MiB scheduler-state bound, without pruning any cursor,
  classification, exclusion or request. Each exact-field bridge retains the complete
  occurrence identity/ID, frozen snapshot-and-policy digest, predecessor and successor
  owner/exact generation, approved resolution, explicit external-evidence attestation,
  and scheduler publication generation. Joins are contiguous, immutable and nonforking.
  Admission reproves immutable inputs and every required bridge through the current
  occurrence's exact final reconciliation attestation. Completed observations never
  become claimed. Bridges grant observation continuity only, never execution or send.
- **Operator ordering and lost acknowledgements:** under the permanent stopped-writer
  lease, durable predecessor/history/request reproof and capacity checks precede the
  unchanged internally execution-fenced occurrence reclaim. Durable exact successor
  reproof precedes scheduler bridge CAS; acknowledgement follows both durable reproofs.
  An explicit exact retry recognizes a published successor and only appends a missing
  bridge or reproves an existing one; it never reclaims twice. A next reconciliation
  cannot overwrite an unretained prior attestation. An occurrence-only publication
  fences startup until the same explicit operation repairs its bridge. Unsafe residue
  or binding/content conflict stays failclosed and is preserved, not repaired/deleted.
- **Lease-epoch refresh:** the public acquisition read may adopt an operator's changed
  scheduler inode only after acquisition/reproof of the permanent lease and proof of
  a reconciliation-only extension of the incumbent pin. Cursors/classifications/
  requests and existing bridge prefix are identical, and contiguous appended bridge
  publication generations account for the entire generation increase. Ordinary and
  continuously-held-lease reads remain pinned; identical-content replacement, root
  swap, history rewrite and unexplained generation changes reject without cache reset.
- **Exact trigger clocks:** an existing authenticated request must match both exact
  definition inputs and its admitted clock snapshot. Exact integer clock validation
  precedes comparison, so bool/float equality cannot borrow a retained request. A new
  manual or named-system authorization at a different clock rejects before claim or
  effects. The original process-local pre-effect capability may retry its original
  clock/sequence; restart allocation, begun/completed and other P3 fences stay intact.

Focused correction tests use the unchanged offline denial fixtures and copied fake
stores. They cover same/fresh scheduler resume strictly after acquisition, retained
multiple edges, malformed chains/bindings, the two-authority crash/ack windows,
explicit repair-before-next-reconcile, busy lease/execution fences, refresh attacks,
pre-mutation bounds/forensic residue, and manual/named-system changed-clock rejection
plus unchanged original-capability retry. Root's completed review and acceptance
evidence are in the linked checkpoint. BasedPyright remains unavailable and
unqualified, with no installation authorized.

This is the bounded implementation specification for Stage 3 of the accepted
[P5 integration/activation plan](p5-integration-activation-plan.md). Root accepted
this architecture as conforming to D6/D7; no new product-policy gate is open.
Stage 2 has [guarded offline PASS](p5-stage2-acceptance-checkpoint.md). The
[P3 closure](p3-closure-checkpoint.md), definition, occurrence, completion,
composition, outbox, receipt and reconciliation contracts remain authoritative
except for the explicit additive scheduling/trigger structural extensions below.
Implementation is accepted only within the guarded offline scope of the checkpoint;
P5 closure and all activation gates remain open.

Workspace: `/Users/jaden/Documents/github/nanobot-harness-candidate` only.
`/Users/jaden/Documents/github/nanobot` is protected. Preserve all existing dirty
work. No commit, installation, dependency installation, deployment, live
state/config/credentials, provider/channel network operation or native Windows
qualification is authorized.

## 1. Objective and accepted decisions

Build dormant production occurrence/completion/outbox/sender composition,
durable schedule cursor and POSIX lifetime lease, restart/shutdown machinery,
authenticated manual and explicit internal-controller adapters, and offline
operator inspection/reconciliation surfaces. The timer feature gate remains
disabled and is admitted only by explicit offline test injection.

The accepted decisions are:

- One gateway owns timer claims under a continuously held POSIX lease. No other
  foreground entrypoint becomes a timer owner.
- On acquisition/reacquisition, unclaimed fires at or before one acquisition-clock
  snapshot become durably missed. Existing claims are observed, not missed or
  replayed. Resume strictly after that snapshot without automatic catch-up.
- While ownership is continuous, delayed wakes claim the exact scheduled due fire
  once. Progress derives from scheduled timestamps, not completion timestamps.
- Interval anchors belong to definition-revision commit evidence. Cron expressions
  require explicit IANA timezones. Old missing evidence is not inferred.
- Manual execution is one authenticated execute-now request, with actual current
  clock and durable unique sequence. System execution requires an explicitly
  constructed named internal controller; it is not a generic agent-turn alias.
- Reuse accepted one-shot computation, bounded canonical text, exact Telegram
  receipt delivery, frozen-order fanout and explicit uncertain-state reconciliation.
- Default construction remains definition-only. No production callback observer,
  Dream/Heartbeat controller or legacy gateway callback registration exists.

Non-goals: armed production timer; live execution/send; non-Telegram/rich/media
support; stored or implicit human cron approvals; automatic stale-claim reclaim,
retry, reconciliation or completed-work delivery replay; callbacks/retention
enablement; Dream/Heartbeat execution; idle/rematerializer construction; live
normalization; guard change; deployment/install/commit; P6/P7/H1/P8 or Windows claim.

## 2. Minimal ownership graph and APIs

Use one cohesive implementation owner. Scheduling persistence, additive identity
parsing and shared composition are coupled; no second worker edits cron formats.
Keep the agent loop/runner unchanged. Suggested focused modules are
`nanobot/cron/schedule.py`, `scheduler_store.py`, `scheduler.py`,
`runtime.py` and `runtime_inspection.py`; combine modules where this removes
unnecessary indirection without mixing durable authorities.

The optional graph uses the existing definition service/store, exact foreground
agent, exact SessionManager and exact ChannelManager already acquired by
RuntimeComposition. It constructs one occurrence store, completion projector,
delivery outbox/runtime and CronChannelSender, and uses
`compose_claimed_text_occurrence` for each exact admitted claim. No additional
runner, session manager, bus, ordinary-delivery path or channel manager is allowed.

Provide narrow typed seams with equivalent responsibilities:

- `ScheduleClock.wall_time_ms()` and `monotonic_time()` plus fakeable sleep.
- Pure next-fire/range-membership calculation from immutable schedule evidence.
- Scheduler-store exact CAS transitions and sealed lifetime-lease capability.
- `Scheduler.acquire_and_reconcile()`, `wake_once()`, `stop_claiming()` and
  cancellation-resistant `aclose()`; starting the task is separate from defining
  or acquiring the default graph.
- Manual authorization/controller factories and adapter methods consuming sealed
  capabilities, not a caller-supplied source enum.
- Explicit-root offline inspection and exact predecessor reconciliation forwarding.

Names may follow local convention. Do not create a general scheduling framework,
plugin abstraction, background retry service or new execution engine.

## 3. Atomic C2 revision-commit scheduling evidence

Add a strict `cron_definition_candidate_v2` scheduling-evidence branch to C2;
continue reading v1 and legacy-conversion candidates for management. Definition
objects and their canonical digest inputs remain byte-for-byte unchanged. Do not
add anchors, timestamps, cursors or runtime fields to `CronDefinition.to_wire()`.
Existing typed definition actions/concurrency tokens and C1 wire/conversion remain
unchanged. Carry `legacy_conversion_evidence` exactly into either candidate version
and update C3's converted-candidate recognition without weakening its legacy fence.

Each retained scheduling-commit entry binds at least:

- store namespace, job instance and definition revision;
- exact immutable schedule encoding/digest and scheduling-format version;
- immutable commit-clock milliseconds and commit kind (`revision` or explicit
  stopped-writer `normalization`);
- originating typed mutation receipt identity, request digest, sequence and journal
  generation when available; absence for legacy normalization is explicitly labelled,
  never filled with invented mutation provenance;
- the exact predecessor ActionStoreGeneration and target checkpoint generation.

Target journal/snapshot generations are predecessor values plus one; target
watermark/base sequence is the predecessor latest visible sequence. Validate exact
integers, receipt ordering/membership, candidate applied-action IDs and generation
relationships. Retained historical entry generations must not exceed the current
checkpoint. The C1 snapshot is the durable publication authority for both the C2
definition candidate and its scheduling evidence.

C2 checkpoint constructs new entries only from its own revalidated replay and the
typed actions it actually commits. Choose the commit-clock snapshot immediately
before the exact checkpoint transaction; it is the transaction's logical commit
clock, not a claim about the wall time at final directory-fsync acknowledgement.
Never accept caller-provided anchor dictionaries or infer time from `CronJob`,
projection, file mtime or scheduler startup. A pending replay does not fabricate
executable commit entries. Runtime rejects pending definition actions as before.

Definition and scheduling entries are in the same candidate passed to the existing
C1 CAS checkpoint. No independently authoritative sidecar is permitted. A stale
checkpoint grants nothing. Retain the exact prepared candidate for identical
lost-ack retry; do not regenerate its timestamp. If publication is visible after an
error, C1 replay/checkpoint durability reproof must recover that exact evidence.
C1 replay already reproves snapshot and journal durability: no C1 format change or
weaker read path is necessary. Reject forged/mutated replay scheduling entries;
checkpoint must rederive authoritative definition transitions and allowed new
metadata, rather than trusting a caller-replaced candidate carrying an owner token.

Because definitionRevision is a content digest, retain immutable first-commit
evidence keyed by `(jobInstanceId, definitionRevision)`. No-op/checkpoint retries
preserve that anchor. A→B→A reuses A's retained anchor and scheduling history; it
does not reset a cursor or recreate consumed occurrence identities. Deletion does
not prune historical evidence. Bounds fail closed and never drop idempotency proof.

V1/missing-evidence definitions stay manageable, but scheduler admission must reject
them. Provide explicit stopped-writer normalization with exact instance/revision
tokens, no active lease and no unresolved relevant occurrence state. Missing timezone
must be supplied explicitly by a typed token-bearing definition update. Establish a
new, labelled normalization commit-clock in a C2 checkpoint; never claim to recover
an unavailable original timestamp. An already retained anchor cannot be overwritten.
Construction, reads and restart do not silently normalize/recreate missing evidence.

## 4. Durable scheduling state and missed evidence

The scheduler store owns no definition, computation, completion or delivery authority.
Its strict versioned state binds namespace, captured workspace/runtime-data identity,
definition instance/revision/commit evidence, cursor revision, last classified
boundary, next fire, and retained classifications with exact predecessor evidence.

Follow P3 private POSIX persistence: absolute non-root lexical paths; trusted
ancestors; held no-follow descriptor chains; root mode 0700; single-link owned
regular files mode 0600; exclusive staging; file/directory fsync; atomic replace;
pre/post binding/content reproof; strict finite JSON/exact integers/known fields;
bounded reads, records and retained evidence. Unsafe roots/files, residue,
replacement, hardlinks or symlinks fail closed and are not repaired/chmod-fixed.
Specify constants and test bounds. No evidence deletion or automatic pruning.

After acquiring the lifetime lease, capture one wall-clock snapshot and reprove
definition commits. Inspect existing occurrence records before any missed
classification. Existing claimed/completed timer IDs in the elapsed interval are
explicit observed exclusions, never declared missed, completed, reclaimed or run.

Missed ranges are allowed only as exact set representations: immutable
schedule/anchor/format evidence, precise lower and upper UTC millisecond boundaries,
and exact observed-occurrence exclusions. Their semantics must answer whether a
particular scheduled fire is missed without approximate counts or guessed history.
Exclude every existing occurrence in the represented window. Keep bounded evidence;
if exact representation exceeds a bound, fail startup without computation instead
of truncating it. Do not label a scheduler range a P3 completion or occurrence claim.

Publishing the missed/observed classifications and advancing the cursor to the first
fire strictly after acquisition time is one atomic scheduler-store CAS transition.
No cursor advancement may lose its predecessor classification. An independent
legacy low-level claim intersecting committed missed evidence is a conflict, never
permission to silently relabel or run it. Only the new sealed production adapter
may dispatch timer work.

For an exact due fire under continuously held lease:

1. Reprove lease/root and exact current definition/evidence.
2. Create the durable P3 occurrence claim for that scheduled timestamp.
3. Atomically publish scheduler claim evidence and the successor cursor.
4. Reprove lease/admission immediately before consuming `begin_execution`.
5. Delegate once to accepted claimed-text composition.

Do not hold short scheduler/definition/ledger locks across model/channel awaits.
Lease/root reproof plus begin admission must be synchronous with respect to task
interleaving; the worker may add a narrow optional pre-begin check to composition,
defaulting to existing P3 behavior.

Crash ordering is mandatory:

| Crash boundary | Restart behavior |
| --- | --- |
| Before occurrence claim | Elapsed unclaimed fire becomes missed. |
| Claim durable, cursor not committed | Discover exact claim; record observed claim and advance without computation. |
| Cursor committed, computation not begun | Preserve claimed reconciliation state; do not reclaim/run. |
| Computation begun or completion ambiguous | P3 one-shot/reconciliation authority; no replay. |
| Completed/projection/outbox partial | Inspect durable state; no automatic projection/delivery replay. |
| Sending/settlement ambiguous | Existing E2/E3 explicit reconciliation; no resend. |

A cursor claiming an absent or conflicting occurrence is corruption. A free lease
or execution lock is not evidence that detached effects did not occur.

## 5. Next-fire and clock rules

- `every`: first fire is immutable anchor plus interval. Successors use integer
  arithmetic from scheduled fires, never execution/completion/current wake time.
- `at`: one exact timestamp; equality with acquisition snapshot is already elapsed
  and therefore missed when unclaimed.
- `cron`: require a valid explicit IANA timezone. Return strictly increasing UTC
  millisecond instants. Validate local wall-time candidates by timezone round trip;
  nonexistent spring-forward times are skipped and distinct valid fall-back instants
  remain distinct. Include non-hour DST transitions in acceptance. Do not rely on
  legacy `_compute_next_run`, machine timezone, silent exception fallback or a
  non-increasing croniter result. Preserve valid existing expression support.
- Acquisition/reacquisition resumes strictly after the captured snapshot. Under
  uninterrupted ownership, a delayed wake processes the exact next due fire once;
  successive due fires retain their scheduled identities, never a synthetic now fire.
- Wall-clock reversal cannot move a committed cursor backward or duplicate a fire.
  Monotonic time controls sleeps/drain deadlines; wall time controls schedule epochs.

## 6. Truthful manual/system trigger identities

Preserve the four-field P3 timer identity, hash domain, serialized bytes and generic
low-level `claim` ABI. Preserve tests where legacy source strings share one identity.
Those low-level source strings remain provenance-only and are not production
authorization capabilities.

Add a distinct versioned manual/system identity branch containing source and durable
unique sequence in addition to the unchanged job/revision/current-clock inputs.
Its hash domain must be disjoint from timer v1. Never fabricate milliseconds,
definition revisions, job IDs or namespaces to create uniqueness. Parser validation
must reject impossible source/sequence/version combinations. Exact request retries
retain the same identity; different requests at equal clock times do not collide
with each other or a timer occurrence.

The scheduler store durably allocates sequence and exact authenticated request-ID
binding before claim creation. Same request ID with changed canonical inputs is a
conflict. A sequence allocated before a crash is retained but not execution authority;
startup never finishes it automatically. An exact process-local retry may reprove
its original pre-effect capability, but a begun request cannot execute again.

Manual APIs consume a sealed capability issued only after explicit authorization;
reject before sequence allocation when authorization fails. Bind the request to
exact definition tokens and admitted current-clock snapshot. System APIs consume a
separately constructed named-controller capability and controller-specific sequence;
there is no generic `source=system` privilege, agent-turn alias, Dream or Heartbeat
factory. Production callers cannot select a timer identity or impersonate its lease.
Existing cron-tool management and legacy `run_job` remain fenced by default.

Approved structural integration edge: `delivery_store.py` currently validates an
exact four-field occurrence identity. Its owning worker may add strict acceptance
and deterministic immutable-input/outbox-ID derivation for this versioned
manual/system branch. Preserve timer parsing/bytes/IDs exactly. Add round-trip,
malformed-branch, timer/manual/system separation and preparation-provenance tests.
This is an explicitly authorized narrow structural edit, not permission to change
E2 transitions, receipt binding, send/reconciliation fences or retry policy.
Completion/identity forwarding may receive equally narrow structural support if
required; no new completion states or fabricated canonical provenance are allowed.

## 7. Composition, lease, shutdown and callbacks

Capture a lexical absolute runtime-data root in read-only preflight, defaulted from
the config-instance runtime-data directory without creating it. Bind optional cron
roots to captured config/workspace identity, not later global config lookup or mtime.
Use a dedicated private runtime-data cron subtree keyed by a domain-separated
workspace/definition-namespace digest; preserve the current definition root and
namespace. Optional authorities share that exact namespace and never overlap
operator evidence roots. Reject links/root swaps rather than resolving them away.

The lease is one permanent private POSIX file, nonblocking `flock`, validated before
and after acquisition/use/release. It is held continuously from reconciliation
through admitted-work drain. It is never unlinked or treated as a time-expiring lease.
Fork children close inherited descriptors without `LOCK_UN`; unsupported platforms
fail closed. No native Windows backend/fallback is implemented or claimed.

The injected runtime graph starts no provider/send during construction/reconciliation.
Admit tasks only after successful graph/channel acquisition. Register optional
ownership transactionally so partial acquisition unwinds reverse exactly once.
Do not replace Stage 1 acquisition/rollback with an unowned background task.

Shutdown or lease loss fences new admissions immediately. Reprove timer ownership
before claim and immediately before begin. Started work retains existing occurrence
and destination effect grants/fences; it is drained or cancelled/joined under the
owning runtime. Stop timer/manual/controller ingress first, then drain/fence admitted
work before channels/stores close and before lease release. Repeated caller
cancellation must not abandon owned cleanup tasks. Bounded drain timeout/error
remains explicit, preserves claimed/sending/uncertain state and never claims cleanup
success or releases ownership while resistant effect tasks still use resources.

Use `compose_claimed_text_occurrence` unchanged except narrow trigger identity,
pre-begin fencing or proven lifecycle corrections. Preflight the complete frozen
destination set before execution. Keep exact one silent canonical turn, durable
completion-before-outbox/send, empty authoritative explicit-send ledger and exact
Telegram text receipt transport. Fanout retains frozen order and does not recompute.

Production passes no callback authority and constructs no callback directory/task.
Injection-only callback tests preserve durable completion/outbox/delivery state
across callback failure/cancellation. Failure cannot replay computation, erase pending
work or overwrite settled sends. Cancellation may retain accepted invoking/pending
states for explicit inspection; it does not authorize automatic delivery replay.
No production reference/registration/fallback to legacy `on_cron_job` may remain.

## 8. Operator plane

Reuse existing offline `cron-delivery` evidence/plan/apply commands and semantics.
Add explicit-root existing-only scheduler/occurrence inspection and exact predecessor
reconciliation forwarding. Do not discover production paths, load config credentials,
construct channels/providers, claim work, run computations, dispatch delivery or arm
timers. Inspection creates no roots/locks/artifacts, performs no fsync/repair and
returns bounded content-free evidence, including exact IDs only in the operator plane.

Occurrence reconciliation requires exact owner/generation plus explicit external
evidence and delegates to the existing internally execution-fenced store API. It
never infers outcome from a free lock and never computes a successor. Normalization
and reconciliation are explicit copied/offline stopped-writer operations in this
stage. Routine logs/metrics contain fixed classification enums, not paths, IDs,
targets, prompts, outputs, receipts, raw exceptions or credentials.

## 9. Acceptance criteria

- **AC1 — commit evidence:** strict C2 v2 definition/evidence atomic publication;
  receipt/generation binding; identical lost-ack retry; stale/forged replay rejection;
  no-op/A→B→A retained anchor/history; v1 reads and explicit-only normalization.
- **AC2 — schedules:** every/at/cron boundaries, explicit timezone, increasing UTC
  fires, DST gap/fold/non-hour transitions and clock reversal match Section 5.
- **AC3 — startup missed:** exact acquisition snapshot, durable bounded missed ranges
  excluding existing claims, cursor crash proofs and first strictly future fire;
  missed work invokes provider/tools/sender zero times.
- **AC4 — continuous due:** delayed uninterrupted wake claims exact due identities
  once, advances from scheduled timestamp and does not fabricate now/catch-up fires.
- **AC5 — lease/binding:** two-owner/two-process contention yields one timer claimant;
  fork/root/lease/record replacement and durability failures fail closed; no TTL,
  takeover, lock deletion or unsupported-platform fallback.
- **AC6 — trigger authorization:** unauthorized requests allocate/claim nothing;
  equal-clock authenticated requests have unique durable sequences; exact retries
  do not duplicate; manual/system cannot impersonate timer; v1 ABI/hash preserved;
  E2 branch parsing/IDs/provenance are exact.
- **AC7 — effects/nonreplay:** exact computation/delivery grants call fake provider
  and fake Telegram sender once; frozen fanout order; unsupported/dynamic failures,
  approval-required/policy-blocked, started/uncertain records remain non-replay;
  callback failure/cancellation cannot erase or override settled delivery evidence.
- **AC8 — lifecycle:** stop-claiming precedes effect drain/channel close/lease release;
  pre-begin lease-loss fences computation; repeated cancellation, resistant tasks,
  partial acquisition and cleanup errors preserve ownership and explicit failure.
- **AC9 — operator:** existing-only inspection changes no filesystem bytes/objects;
  explicit-root exact reconciliation uses existing fences and no effects; no guessed
  metadata, automatic claim finishing or live-path discovery.
- **AC10 — dormancy/shared graph:** default define/acquire/close constructs no optional
  directories, leases, callbacks, timer tasks, Dream/Heartbeat jobs or background
  provider/send; injected graph reuses exact foreground/channel/session identities;
  no legacy production callback reference or fallback.
- **AC11 — regression/scope:** current P3 ABI, definitions, timer IDs and receipt/send
  semantics remain green; frozen boundaries/hashes preserved; historical fixtures
  are classified, not weakened/restored to force green. The three mixed historical
  modules below retain their fixtures and must add zero failed nodes over the
  exact 16-node baseline; they are not required-green counts.
- **AC12 — evidence:** required new tests and the explicitly labelled current-contract
  green selections below pass under copied/offline denial fixtures; lint/compile/diff
  evidence and fresh GPT-6.1 review are recorded. Separately labelled historical
  diagnostics are classified, not required to pass, and remain outside green counts;
  environment/platform failures remain qualified.

## 10. Required test files and exact validation

New files are required deliverables (equivalent focused organization requires root
approval before implementation):

- `tests/cron/test_schedule.py`
- `tests/cron/test_schedule_commit.py`
- `tests/cron/test_scheduler_store.py`
- `tests/cron/test_scheduler.py`
- `tests/cron/test_runtime_adapters.py`
- `tests/migration/test_p5_stage3_composition.py`
- `tests/migration/test_p5_stage3_shutdown.py`
- `tests/cli/test_cron_runtime_inspection.py`

Use fake clock/provider/Telegram receipt channel, temporary copied stores, exact
crash injection before/after commit/claim/cursor/begin/completion/callback/send/settle,
two local POSIX processes only where needed for lease tests, root/lock/content attacks
and explicit denial of network and unauthorized process effects. An environment
variable alone is not sufficient denial evidence. POSIX worker-process fixtures are
the narrow permitted exception to subprocess denial; no external application runs.

Run each command independently; overlapping counts must not be summed. The first
four pytest commands below are required current-contract green selections. Do not
run unbounded historical timer selections. Do not install tools/dependencies.

```bash
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/cron/test_schedule.py \
  tests/cron/test_schedule_commit.py \
  tests/cron/test_scheduler_store.py \
  tests/cron/test_scheduler.py \
  tests/cron/test_runtime_adapters.py \
  tests/migration/test_p5_stage3_composition.py \
  tests/migration/test_p5_stage3_shutdown.py \
  tests/cli/test_cron_runtime_inspection.py --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/cron/test_occurrence_store.py \
  tests/cron/test_cron_package_d.py \
  tests/cron/test_cron_package_e.py \
  tests/cron/test_canonical_execution.py \
  tests/cron/test_canonical_result_adapter.py \
  tests/cron/test_cron_composition.py \
  tests/cron/test_cron_callback_store.py \
  tests/cron/test_cron_channel_sender.py \
  tests/cron/test_cron_delivery_runtime.py \
  tests/cron/test_cron_delivery_store.py \
  tests/cron/test_delivery_reconciliation.py \
  tests/cron/test_visible_completion_link.py \
  tests/cli/test_cron_delivery_reconciliation.py \
  tests/agent/test_cron_silent_turn.py --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/cron/test_action_store.py \
  tests/cron/test_action_store_conversion.py \
  tests/cron/test_definition_store.py \
  tests/cron/test_cron_definition_service.py \
  tests/cron/test_cron_tool_schema_contract.py --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/migration/test_p5_stage1_composition.py \
  tests/migration/test_p5_stage1_shutdown.py \
  tests/migration/test_p5_stage1_adapter_matrix.py \
  tests/migration/test_p5_stage2_composition.py \
  tests/cli/test_startup_admission_agent_gateway.py \
  tests/cli/test_startup_admission_onboard_webui_api.py --tb=short

/Users/jaden/.pyenv/versions/3.13.1/bin/ruff check \
  nanobot/cron nanobot/runtime_composition.py nanobot/cli/cron_delivery.py \
  nanobot/cli/gateway_runtime.py nanobot/cli/commands.py \
  tests/cron/test_schedule.py tests/cron/test_schedule_commit.py \
  tests/cron/test_scheduler_store.py tests/cron/test_scheduler.py \
  tests/cron/test_runtime_adapters.py \
  tests/migration/test_p5_stage3_composition.py \
  tests/migration/test_p5_stage3_shutdown.py \
  tests/cli/test_cron_runtime_inspection.py

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
  -m compileall -q nanobot/cron nanobot/runtime_composition.py \
  nanobot/cli/cron_delivery.py nanobot/cli/gateway_runtime.py nanobot/cli/commands.py

git diff --check
```

The following exact selection is a **historical diagnostic, not a green acceptance
gate**. Root already observed historical failures in persistence/admission and
constructor-era gateway fixtures. Reuse captured evidence where available. If a
fresh run is needed to attribute a changed boundary, run only that bounded
selection, with an external wall-clock limit of 60 seconds and interrupt/join the
test process on expiry. Record partial results and timeout honestly. Do not install
a timeout plugin, silently skip failing nodes, restore removed APIs or rewrite old
fixtures to turn this command green. No full-cron or historical timer-wait rerun is
required. Any finite node omission must name the exact node, baseline rationale and
root approval; it never changes current-contract green gates.

```bash
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/cron/test_cron_persistence.py \
  tests/cron/test_migration_cron_admission.py \
  tests/cli/test_gateway_runtime.py --tb=short
```

The following finite mixed modules are also **historical diagnostics, not green
acceptance gates**. Run independently of current primitive selections. Root's
pre-implementation combined action/definition/service/schema/tool-list selection
reported **14 failed, 275 passed in 7.34s**; adding the migration-fields module to
the separate primitive selection reported **2 failed, 228 passed in 6.67s**.
The exact total baseline is **16 failed nodes**: the fourteen tool-list failures
retained by root and
`test_migration_cron_fields.py::test_service_serializer_preserves_identifiers_and_destinations[session_bound]`
and `[legacy_fresh]`. They assert removed runtime-state/context/token interfaces,
route-clearing policy or legacy `jobs.json` publication. Keep fixtures unchanged;
zero additional failed nodes is required, but these counts never enter green counts.

```bash
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/cron/test_migration_cron_fields.py \
  tests/cron/test_cron_tool_list.py \
  tests/cron/test_cron_tool_list_fork.py --tb=short
```

Extend lint/compile arguments with any other explicitly allowed changed CLI/test
paths. Separate exact changed-file Ruff from broad diagnostic Ruff; pre-existing
findings are not implementation failures or permission for unrelated cleanups.
No `ruff format`. Run BasedPyright only if already available; otherwise record it
unqualified. Native Windows, CLI/environment dependencies and release remain P6.

Baseline qualifications supplied by root:

- Accepted P3 composition/execution/result/sender/callback selection: **54 passed
  in 2.42s**, offline; this does not substitute for Stage 3 acceptance.
- Full `tests/cron` currently aborts collection because
  `test_cron_mutation_callers.py` imports removed
  `_register_system_job_with_current_tokens`. Do not restore the legacy helper.
- A diagnostic excluding that module was interrupted after **167.48s**, with
  **40 failed, 402 passed** reported; legacy timer waits make it unsuitable as an
  unbounded acceptance command. This is partial diagnostic evidence, not green.
- Baseline tree hashes (1,824 files) and cron/runtime snapshots are retained at
  `/tmp/nanobot-stage3.dZZOBF`. Preserve unrelated work and classify historical
  removed-ABI/destructive-execution fixtures by exact node and accepted conflict.

## 11. Allowed files, frozen boundaries and review protocol

Allowed: focused new cron modules; narrow C2/C3 edits to `definition_store.py` and
`service.py`; additive identity/admission changes to `occurrence_store.py` and
`occurrence.py`; the explicit E2 identity parser/immutable ID branch in
`delivery_store.py`; narrow composition pre-begin/lifecycle wiring; shared
`runtime_composition.py`; gateway lifecycle integration; explicit operator CLI module
and command registration; corresponding cron/CLI/migration tests; this contract and
a new Stage 3 checkpoint. A narrow completion serialization branch is allowed only
if identity forwarding requires it, with unchanged state/provenance semantics.

Frozen: C1 action-store wire/converter; definition wire/digest/typed mutation requests;
timer v1 IDs; occurrence execution/reconciliation state machines except additive
trigger parsing; E2/E3 transitions, receipts, destination fences and reconciliation;
callback store/policy; Telegram receipt transport; canonical text/result/approval
policy; session/source/checkpoint/artifact/recovery/export authorities; agent
loop/runner; migration guard; deploy/install/release files; historical checkpoints.
Do not edit broad conftest/global denial behavior or historical fixtures to hide
non-passes. Seek root approval before crossing a frozen boundary or expanding scope.

Root owns graph sequencing and is read-only for implementation/task-file edits after
dispatch. One cohesive GPT-6.1 Sol worker owns this contract's complete package and
must preserve others' dirty changes. Worker reports exact files, command outcomes,
crash evidence, frozen hashes and remaining qualified limits before acceptance.
Use fresh GPT-6.1 Sol integrated review of the actual joined diff and raw evidence;
do not use GPT-5.6 or claim prior review as this stage's acceptance. At most two
bounded correction rounds under this specification; a persistent material failure
returns to root planning rather than widening authority or endlessly patching tests.

Rollback is disabling/removing the injected scheduler facade while retaining the
definition-only service and all durable scheduling/occurrence/outbox evidence.
Never delete uncertain claims/outboxes or reset anchors/cursors to simulate rollback.
Global guard and timer/system/callback/Dream/Heartbeat gates remain closed throughout.
