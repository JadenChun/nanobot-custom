# P5 integration and activation plan

Status (2026-10-02): **D1–D8 and guarded offline Stages1–4 accepted;
root technical PASS and fresh integrated PASS support P5 DONE / full Stage5
closure ACCEPTED / READY_TO_BEGIN_P6; activation NOT_READY/NOT_AUTHORIZED**.
[Current closure](p5-closure-checkpoint.md) and
[permanent compatibility evidence](p5-compatibility-closure.md) reconcile all116
source cases/628 identities:613 COVERED/15 precise prior accepted changes, zero gaps,
with exact raw commands/outcomes and historical audit retained. Root final publication
PASS accepts AC1–AC5/all nine statements, not writer self-PASS. The original audit's62/5/561
and21-pass/1-fail are historical, not current closure. P6 NOT_STARTED, D9
NOT_AUTHORIZED; P7/H1/P8/native/install/type qualification not delivered.
Later not-started/current-state sequencing below is retained pre-implementation
history, superseded by this status only; accepted policy and operational body,
Stage5 objective and §§6–9 gates stay unchanged.
This plan depends on the accepted [P3 adapter closure](p3-closure-checkpoint.md)
and [P4 durable context/session closure](p4-closure-checkpoint.md), and supersedes
P5 sequencing implied by the chronological
[canonical-turn work log](p5-canonical-turn-work-in-progress.md). It does not
supersede those documents' accepted contracts or evidence.

The Stage 1 evidence now refers to the typed `define()`/async `acquire()` lifecycle
and fail-closed session authority in [the Stage 1 checkpoint](p5-stage1-checkpoint.md),
not the rejected deferred-proxy/synchronous-build implementation. D9 remains a
separate user gate after Stage 2 and the remaining P5/P6 evidence.

The final Stage 1 correction evidence also fixes the lifecycle boundary rather
than widening activation: terminal acquisition failure is replayed identically,
channel acquisition and startup roll back all provisional ownership, restart
notice and channel cleanup tasks are owned/joined, and cleanup errors propagate
after reverse exactly-once attempts. Classic CLI and API mutations occur only
after successful acquisition, with default-cutoff sentinels covering session-store
construction/migration, providers, plugins, listeners, and tasks. Exact-one turn
evidence is attributed only to real adapter paths: classic CLI orchestration, SDK
run/stream, bus, API, and representative Telegram headless ingress. Gateway has no
separate turn adapter beyond shared channel/bus ingress.

## 1. Current-state verdict

The candidate is **partially integrated but not activation-ready**. Ordinary direct,
bus, API, SDK, and representative Telegram paths largely converge on
`AgentLoop.execute_turn` and `AgentRunner`, including one lock owner and bounded
offline actual-engine evidence. Production entrypoints and external tool loading
remain protected by `MigrationIncompleteError`; many positive integration fixtures
reach the candidate only through test-local guard patches.

Guarded offline Stages 1-3 now accept the shared composition/lifecycle, durable
session policy and dormant cron machinery. The remaining work is the separately
authorized Stage 4 integrated matrix and later closure/release/activation evidence,
not another turn loop. The following bullets retain the **pre-implementation
baseline**, superseded where applicable by the Stage 1-3 checkpoints; they are not
current claims that accepted session or scheduler implementation is missing:

- strict source/checkpoint behavior is selected only when persisted
  `source_schema_version` metadata already exists; an absent session and an
  unconverted session still select legacy JSONL;
- provider conversation state is persisted only on the ordinary legacy path and is
  deliberately excluded from source, fresh, and ephemeral execution;
- artifact projection depends on source sessions, and `ArtifactRematerializer` has
  no production caller;
- dormant I1/I2/I3 idle compaction has no production construction, but the older
  `AutoCompact` can run when `session_ttl_minutes` is nonzero;
- acquired CLI and gateway graphs create a definition-only `CronService`. The
  historical gateway Dream/Heartbeat/cron callback execution branch has been
  removed. Nothing calls the accepted
  `compose_claimed_text_occurrence` from a scheduler, gateway, or CLI;
- cron lacks an accepted next-fire cursor, missed-run policy, scheduler lease,
  startup reconciliation, shutdown contract, and production construction of its
  occurrence/delivery/callback authorities;
- omitted `gateway.heartbeat.enabled` now resolves false; explicit configuration
  remains representable but inactive;
- channel discovery preserves explicitly configured/enabled legacy Python
  entry points through lazy adaptation and reserves packaged names; and
- the representative offline P5 selection recorded by the parent plan is not green:
  one DOCX attachment case lacks `python-docx` in that environment. BasedPyright and
  native Windows remain unqualified.

No count in a work log is promoted here into integrated acceptance, and overlapping
suites must not be summed.

## 2. Non-negotiable invariants

1. Nanobot remains the base. A fork behavior is retained unless a verified
   equivalent replacement is explicitly accepted.
2. Channels are transports for one owner-bound assistant, not independent agents or
   memories. H1 busy-input steering and its durable inbox remain later scope.
3. `execute_turn`/`AgentRunner` is the sole computation owner for a turn. No adapter,
   scheduler, recovery path, or callback may repeat reduction, model transport, tool
   effects, callbacks, or result delivery.
4. Exactly one durable mode owns a session at a time. A legacy JSONL projection and a
   canonical source log may coexist for export/rollback, but never as concurrent
   writers or peers selected by freshness.
5. Corrupt, partial, converting, unsupported, or conflicting state fails before
   model/tool/channel work. Recovery and startup admission remain content-free and
   fail-closed.
6. Unattended cron receives no implicit approval and cannot consume stored human
   approvals. Its accepted result remains bounded canonical text; initial delivery
   remains exact Telegram receipt transport; callbacks occur only after durable
   lifecycle/result state; uncertain computation or sending is never replayed
   automatically.
7. Every external effect has one authority and one durable predecessor. A stale
   occurrence or delivery claim is reconciled, never “retried” by recomputing.
8. Canonical source is authoritative after conversion. Checkpoints, provider state,
   artifact projections, and legacy exports are derived state and cannot select or
   rewrite the source by being newer.
9. I1/I2/I3 idle work, artifact rematerialization, timers, callbacks, live delivery,
   live-state conversion, and shadow measurement that invokes a provider remain off
   unless their named gate is accepted.
10. Guard removal, scheduler activation, deployment, live traffic, P6 release/native
    qualification, P7 canary/promotion, H1, and optional P8 are separate authorities.

## 3. Decision ledger

### 3.1 Already accepted; implementations must not reopen these decisions

| ID | Accepted decision | Consequence for P5 |
| --- | --- | --- |
| A1 | One canonical turn owner and no duplicate external effects. | Integrate entrypoints by delegation to one composition object; do not add another runner or delivery workflow. |
| A2 | P3 unattended-cron policy and exact Telegram receipt delivery. | Production cron composition must use the accepted occurrence, completion, outbox, sender, and callback contracts without widening result/transport support. |
| A3 | P4 fail-closed recovery, canonical source/checkpoint governance, and provider-state boundaries. | P5 may select and compose these authorities, not weaken them or infer recovery. |
| A4 | Artifact rematerialization and I1/I2/I3 are dormant/operator-only or injection-only. | They receive explicit interfaces and negative startup tests, not automatic callers. |
| A5 | The global guard stays closed during implementation and acceptance. | All stages are testable by injected composition or test-local guard patches; none makes the branch runnable. |
| A6 | `MigrationIncompleteError` removal is a final isolated user-authorized gate. | It is not part of an implementation worker's default scope or any refactor below. |

### 3.2 Accepted P5 implementation policy (D1-D8)

The user accepted D1-D8 as the P5 implementation-policy gate before Stage 1.
They determine configuration, public compatibility, persistent-mode admission,
and the production composition ABI. D9 remains unaccepted and is a later
activation gate; it is not inferred from D1-D8 acceptance.

| ID | Recommended answer and rationale | Compatibility impact | Acceptance point |
| --- | --- | --- | --- |
| D1 — session rollout | After the guard is eventually removed, create absent persistent sessions as canonical-source sessions. Keep already valid legacy sessions legacy until an explicit stopped-writer conversion. Select mode from a versioned, durable mode record bound to the session incarnation/generation; never from “which file is newest.” Fail closed on partial/unsupported/corrupt/conflicting state. | New sessions no longer persist provider conversation handles initially. Existing sessions continue unchanged during mixed mode. Tools that inspect JSONL must use an export/projection API. | Accept before Stage 1; approve each live conversion batch separately in P7. |
| D2 — conversion and rollback | Use a per-session conversion journal: `prepared` snapshot/export → canonical source install and verification → mode-record commit → legacy becomes read-only rollback material. Preserve a lossless canonical export plus a compatibility JSONL export. Downgrade is allowed only through an offline exporter that proves the target format can represent the state; otherwise it fails without changing mode. Mixed mode lasts through P7 canary and at least one accepted rollback-capable release, expressed as milestones rather than dates. | Downgrades may require the old binary to read the compatibility export; canonical-only features block lossy downgrade. Conversion is not transparent while a writer runs. | Accept before Stage 1; accept retention/removal of rollback material after P7 separately. |
| D3 — idle compaction | Treat nonzero legacy `session_ttl_minutes`/`idleCompactAfterMinutes` as an activation incompatibility: report it at preflight and require explicit disablement. Do not map it silently to I1/I2/I3. Keep both legacy AutoCompact and new idle construction inactive at first activation. | Operators relying on automatic legacy compaction must disable it or remain on staging; `/dream` remains a separate explicit command. | Accept before Stage 1; any future idle policy needs a separate product/operations gate. |
| D4 — artifacts | Keep artifacts opt-in. Default the root under the config instance's runtime-data directory, require an explicit absolute root otherwise, enforce existing per-blob/per-namespace quotas, and perform no automatic retention deletion during P5/P7. Preserve inline source evidence for the rollback window; publish references only after durable blob install. Add an operator-only “inspect/rematerialize one exact reference into copied/offline state” surface; never rematerialize on reads/startup. | Enabling artifacts can turn oversized qualified tool output into private references and may consume disk until an operator-managed retention policy is later accepted. Disabling artifacts preserves inline behavior and fails pressure rather than losing evidence. | Accept before Stage 1, including whether current quota defaults are acceptable; production retention deletion remains a later gate. |
| D5 — activation seam | Introduce one `RuntimeComposition` (name may follow repository convention) built once per process after config parsing and startup admission. It owns shared session, foreground, channel, cron-definition, and optional durable authorities; entrypoints receive narrow facades from it. Construction itself performs no model call, timer arm, send, conversion, or rematerialization. Keep every production entrypoint's global guard before construction until D9. | Constructor injection tests change to use the same composition factory. No public entrypoint becomes runnable during P5. | Accept before Stage 1. |
| D6 — cron semantics | One gateway process is the only timer owner, protected by a single-owner process/file lease and a durable schedule cursor. While one lease is continuously held, a delayed wake claims its exact due fire once; on startup or lease reacquisition, every unclaimed fire at or before the acquired-clock snapshot is durably classified `missed` without provider/send work. An `every` schedule has an immutable anchor at definition-revision commit and advances from scheduled fire time, not completion; cron expressions require an explicit IANA timezone and also advance from the last scheduled fire. An `at` definition has one fire and becomes missed if already elapsed at reconciliation. Scheduling resumes at the first fire strictly after the reconciliation clock—no automatic catch-up. `manual` means one authenticated execute-now request with its own current-clock occurrence and unique sequence; `system` is reserved for explicitly constructed internal controllers, never a generic agent-turn alias. A stale/started/uncertain claim requires operator reconciliation and is never reclaimed for computation. | Users who expected catch-up must manually run a new occurrence. Existing cron definitions without timezone/anchor require stopped-writer normalization before scheduler admission. Multiple gateways cannot share timer ownership. Native Windows lease behavior remains P6 work. | Accept before Stage 1 as part of D1-D8; timer enablement is still a separate later gate. |
| D7 — Dream, Heartbeat, callbacks | Do not schedule Dream or Heartbeat in initial P5/P7 foreground activation. Change effective compatibility migration so omitted Heartbeat becomes explicitly disabled before guard removal; do not let the schema's current true default spend provider tokens. Keep manual `/dream`; remove/isolate the unreachable legacy callback rather than reconnecting it. Initial cron production composition has no callback observer. A later callback option may use the accepted content-free store at a private runtime-data root, with no automatic record deletion during canary; it remains POSIX-only and startup fails closed when configured elsewhere. | Existing implicit Heartbeat/Dream schedules stop; users must explicitly accept a later system-job policy. Callback consumers are unavailable initially. | Accept before Stage 1 for safe config migration; accept any later Dream, Heartbeat, callback, retention, or non-POSIX backend independently before enablement. |
| D8 — external legacy channels | Preserve installed legacy channel compatibility through explicit, lazy entry-point adaptation, not global auto-load. Enumerate names without importing code; load only a configured/enabled unknown channel; built-in package names are reserved; duplicate names fail preflight. Legacy class entry points get a narrow descriptor adapter, default disabled, and the same dependency/config/authorization checks as packaged channels. | An installed plugin is no longer activated merely by installation. A plugin must be named in config and may need to update to the descriptor contract for management features. | Accept before Stage 1 because this is a public extension contract and code-loading boundary. |
| D9 — activation order | Complete P5 with the guard intact; complete P6 artifact/rehearsal and obtain its verdict; then authorize one isolated guard-removal change for foreground/direct/API/SDK/channel paths while every background timer/system job/rematerializer remains independently disabled. P7 first exercises a single-consumer canary. Cron timer activation is a second explicit P7 gate after foreground stability; Dream/Heartbeat/callbacks remain later gates. | Activation is intentionally incremental; “runtime enabled” does not mean “all automation enabled.” | User acceptance is required after P5 and P6 evidence, immediately before the isolated guard-removal change, and again before timer activation. |

Rejected alternatives:

- **Convert every session at startup:** rejected because startup would mutate live state,
  broaden rollback, and turn one corrupt session into fleet-wide risk.
- **Choose canonical versus legacy by file presence/mtime:** rejected because two writers
  or a partial conversion could silently fork history.
- **Persist provider handles in both modes now:** rejected because P4 accepted only
  ordinary legacy settlement; expanding it couples source conversion to opaque
  provider replay semantics.
- **Reactivate gateway's legacy `on_cron_job`:** rejected because it bypasses accepted
  occurrence/outbox/callback ownership and can duplicate computation or delivery.
- **Catch up missed schedules automatically:** rejected because provider/tool/send
  effects would occur after an unobserved downtime and can burst on restart.
- **Auto-load all installed legacy channel entry points:** rejected because discovery
  would execute third-party code and could widen enabled transports unexpectedly.
- **Use telemetry shadow turns:** rejected because a shadow model/tool/send path would
  violate single execution and create privacy/cost effects.

## 4. Precise architecture and contracts

### 4.1 Single runtime composition authority

The process has one composition root. CLI agent, gateway, API server, SDK facade, and
channel manager may request a foreground facade from it, but may not independently
construct `SessionManager`, foreground dependencies, `AgentLoop`, channel delivery,
or cron execution authorities. The gateway alone may additionally request a cron
definition facade and, behind a separately disabled timer gate, a scheduler facade.

Construction order is fixed:

1. parse and normalize config without writing it;
2. run global migration guard (while it exists), startup receipt admission, and
   configuration incompatibility checks;
3. bind canonical workspace/runtime-data identities and create one `SessionManager`;
4. construct foreground dependencies, `AgentLoop`, and one `TurnDeliveryFactory`;
5. construct channel descriptors/managers against that same bus/session/delivery
   graph; and
6. construct definition-only cron management. Occurrence, completion, outbox,
   sender, callback, and scheduler authorities are constructed only when their
   explicit feature gates are admitted.

Startup is transactional at the component level: a failure closes already-created
owned resources in reverse order and starts no listener, timer, provider request, or
send. Shutdown first stops ingress and timer claiming, then drains/fences admitted
turns and deliveries, then closes channels, agent-owned resources, and stores. A
second entrypoint cannot “helpfully” start an already-owned component.

### 4.2 Session-mode state machine

For each persistent session, the admitted mode is one of `absent`, `legacy`,
`converting`, `canonical`, `exporting`, `quarantined`, or `unsupported`.

- `absent` creates canonical source state only after a real admitted first write.
- `legacy` reads/writes only the existing JSONL authority. Provider state retains
  the P4 ordinary-legacy behavior.
- `converting` and `exporting` reject turns and mutations. Their journal identifies
  source snapshot, target generation, digests, and the one permitted recovery step.
- `canonical` uses source execution ownership and checkpoints; JSONL/export files
  are read-only projections. Provider conversation state remains absent.
- `quarantined` and `unsupported` admit only the existing explicit operator recovery
  or export inspection surfaces, never a model or external effect.

The durable mode record is generation-bound and written only after target durability
is verified. If a crash happens before its commit, legacy remains authoritative and
the staged target is ignored/quarantined. If it happens after commit, canonical is
authoritative and recovery may finish read-only projection/export publication; it
must not roll back ownership based on an older file. Conversion accepts no active
session lock owner and no unresolved recovery/occurrence state.

Reset, delete, model-preset change, and API/session listing must dispatch through the
same mode resolver. Missing or corrupt mode evidence must not fall back to legacy.
Export bundles contain the canonical source/checkpoint metadata needed for future
re-import plus a clearly labelled compatibility JSONL when representable. Export is
never an authority switch.

### 4.3 Artifact policy

Artifact writes remain source-session-only and restricted to the reviewed producers.
The blob install is durable before the source event publishes its reference. Quota
failure leaves bounded inline evidence or fails the turn according to the accepted
producer contract; it never publishes a dangling reference. Artifact content,
references, and paths are excluded from metrics and callback records.

No background garbage collector is introduced in P5. An operator report may list
aggregate reachable/unreachable byte counts without content or user text. Any later
deletion requires a source/checkpoint reachability proof, rollback-window closure,
and its own accepted retention policy. Rematerialization accepts an exact source
identity, snapshot/generation, and artifact reference, writes only to copied/offline
state during P5, and never invokes tools or a provider.

### 4.4 Cron production composition

Cron is divided into non-overlapping owners:

| Owner | Sole responsibility |
| --- | --- |
| Definition service | Typed definition mutation/replay; no runtime state or callback. |
| Scheduler | Lease, durable next-fire cursor, fakeable clock, due/missed classification, and exact occurrence claim creation. |
| Occurrence/completion authority | One-shot computation grant and terminal result state. |
| Canonical executor | One call to `execute_silent_cron_turn` for the exact claim. |
| Delivery outbox/runtime | Frozen destinations, one sender invocation per durable claim, receipt settlement, and uncertain-send reconciliation. |
| Callback authority | Optional content-free post-result observation; never computation/delivery authority. |

The initial adapter calls the accepted `compose_claimed_text_occurrence` for an exact
claim. The legacy gateway callback is removed or moved to a test-only historical
adapter with no production registration; it must not coexist as fallback. Definition
management remains usable with the timer disabled. Manual execution creates a new
occurrence through the scheduler/claim adapter—it never invokes `on_job` or calls the
agent directly. System jobs require a dedicated adapter and explicit enablement;
Dream and Heartbeat have none initially.

On restart, the scheduler acquires its lease before reading/updating the cursor. It
records missed classifications without computing, enumerates claimed/started and
outbox `claimed/sending/uncertain` records for content-free operator diagnostics, and
does not resolve them automatically. On shutdown, it stops claiming first, awaits
pre-effect work, and preserves already-started/uncertain classifications if the
bounded drain cannot settle. Lease loss fences further claims immediately.

### 4.5 Observability, privacy, and shadow limits

Allowed metrics are bounded counters/gauges and coarse latency histograms for:
admission result, session mode, conversion phase, canonical/legacy turn result,
scheduler lease state, due/missed/claimed occurrence state, outbox state, callback
state, channel adapter result, and startup/shutdown failures. Labels may contain only
fixed enums and component names; no session/job/run/message IDs, user/channel IDs,
paths, prompts, model output, tool arguments/results, artifact references, receipts,
or credentials. Logs follow the same rule for routine paths; operator inspection
uses explicit local commands and exact IDs supplied by the operator.

“Shadow” means pure projection/validation against copied state or already-loaded
immutable snapshots. It performs no provider/model request, tool execution, callback,
channel send, persistence mutation, conversion, rematerialization, or timer claim.
Mismatch reports contain classifications and digests, not content. Cardinality and
retention must be bounded by the existing operational telemetry backend; if no such
backend is configured, local aggregate logs are sufficient and no new exporter is
added in P5.

## 5. Functional implementation stages

Each stage includes implementation and validation. Stages are sequential because
they transfer shared composition and persistence ownership. A fresh worker receives
only the named stage after its dependencies and decision gates are accepted.

### Stage 1 — composition root and closed activation seams

**Closure checkpoint.** Stage 1 is accepted for its guarded offline composition
scope in [the Stage 1 checkpoint](p5-stage1-checkpoint.md), following independent
review and orchestrator verification. Owned instance-bound shutdown tasks replace
recyclable object IDs; shield-owned acquisition/startup rollback survives repeated
cancellation and stops channels before joining resistant start finalizers.
Graceful shutdown and genuine owner cancellation remain distinct. The final Stage 1
selection passed 89 tests; startup/API/SDK compatibility passed 50. Full channels
reported 443 passed and 12 missing-`setproctitle` failures; CLI subprocess
qualification is also blocked by missing `click`. These dependency/platform/release
qualifications remain visible and are not treated as green. Stage 2 may now use the
accepted composition seam. The rejected first attempt remains recorded in its
[historical checkpoint](p5-stage2-checkpoint.md). The later repair fan-in and CORRECT
review are recorded separately in the [repair checkpoint](p5-stage2-repair-checkpoint.md):
the falsey-authority injection defect is fixed and the Stage 1 cutoff restoration
passed independent safety review, but AC2/AC5/AC6 and a material canonical SDK-clear
decision remain open.
P5 closure and D9 remain separate
gates.

**Objective.** Create the single process composition authority, route every
foreground entrypoint through its injected facade, add read-only session-mode
preflight, normalize omitted Heartbeat to disabled for the candidate, and implement
explicit lazy legacy-channel entry-point adaptation. Keep the global guard and all
background authorities closed.

**Observable behavior.** Direct, bus, API, SDK, CLI/gateway construction, and
packaged/external channels use the same session/agent/delivery identities in offline
tests. Constructing or closing a partial graph causes no provider, subprocess,
network, listener, timer, store conversion, or send. An enabled duplicate/invalid
external channel fails before runtime import/activation. Production invocations still
raise `MigrationIncompleteError` before composition.

**Dependencies.** Accepted D1-D8; P3/P4 closure; existing startup admission.

**Ownership boundaries.** Integration owner: runtime composition module,
`agent/loop.py`, public SDK facade, API/gateway/CLI construction, config schema/loader,
and lifecycle tests. Channel owner: registry/manager and entry-point adapter tests.
Only the integration owner edits shared loop/config/entrypoint files.

**Non-goals.** No session conversion/default switch, artifact runtime policy, cron
occurrence authority, timer, legacy callback, rematerializer, idle runtime, guard
change, live plugin, or network test.

The existing top-level local-trigger enqueue command is preserved as a durable
queue-writing surface. Stage 1 disables only the gateway local-trigger queue runner;
it does not delete or relabel the enqueue command, and no enqueued item is executed
by an alternate Stage 1 authority.

**Acceptance criteria.** Identity assertions prove one graph; every entrypoint calls
one `execute_turn`; negative startup closes in reverse order; default/omitted
Heartbeat cannot register work; external plugins are lazy/config-explicit and cannot
shadow built-ins; all production guard-denial tests remain green.

**Validation discovery.** Start with `tests/migration/test_p1_foundation.py`,
`tests/migration/test_headless_startup_compatibility.py`, `tests/cli/test_commands.py`,
`tests/cli/test_commands_fork.py`, `tests/agent/test_migration_api_engine.py`, SDK/API
facade suites discoverable with `rg -l 'Nanobot|chat/completions|execute_turn' tests`,
and channel discovery/manager/headless suites under `tests/channels/`. Add a
no-network/no-process composition matrix and startup-negative cases. Run scoped Ruff,
Python compile checks, and `git diff --check`; BasedPyright is required when available,
otherwise its absence is recorded rather than treated as a pass.

**Correction/rollback boundary.** Revert the composition facade and adapter routing
as one package; the unchanged guard prevents production exposure. Do not retain a
half-migrated entrypoint.

**Sizing.** This is large enough to eliminate duplicate construction across all
foreground surfaces, yet bounded because it creates no persistent state or scheduler.

### Stage 2 — canonical session rollout, conversion/export, and artifact policy

**Acceptance status (2026-10-01).** Stage 2 has root **PASS for guarded offline
implementation**, with fresh integrated review and exact validation recorded in the
[acceptance checkpoint](p5-stage2-acceptance-checkpoint.md). The first prototype and
earlier repair stop remain historical in the unchanged
[attempt](p5-stage2-checkpoint.md) and [repair](p5-stage2-repair-checkpoint.md)
checkpoints. The accepted integrity and compatibility completion resolves
AC2/AC5/AC6 and the synchronous SDK-clear choice, preserves regular and temporary
contexts, and restores the definition-only durable default from captured preflight
roots. Production guards remain closed. Stage 3's subsequent guarded offline
acceptance is recorded below; it supersedes its former not-started status without
changing Stage 2's historical checkpoints.

**Objective.** Implement the durable mode resolver/state machine, canonical default
for new persistent sessions, stopped-writer conversion/export/downgrade validation,
legacy TTL incompatibility, and opt-in artifact/rematerialization operator surfaces.

**Observable behavior.** Copied-state fixtures can run mixed legacy and canonical
sessions concurrently without cross-writing. New persistent sessions become
canonical; valid legacy sessions remain legacy. Corrupt, partial, unsupported, or
dual-authority conflicts fail before provider/tools. Crash injection at every journal,
source install, mode commit, export, and artifact publication boundary produces one
deterministic authority after restart. Nonzero legacy TTL blocks candidate preflight.
Rematerialization is exact, offline/operator-only, and absent from startup/reads.

**Dependencies.** Stage 1 and accepted D1-D4.

**Ownership boundaries.** Session owner: mode record, resolver, conversion/export,
source-store integration, and copied-state fixtures. Context/artifact owner: configured
factory, quota/report, exact rematerialization command. Integration owner alone wires
`AgentLoop` admission and config. Provider owner verifies that provider state remains
legacy-only.

**Non-goals.** No live-state conversion, automatic bulk migration/deletion/retention,
provider-state expansion, idle construction, timer, guard change, or Windows claim.

**Acceptance criteria.** All state transitions and crash points are deterministic;
conversion is idempotent before commit and never rolls authority back after commit;
lossy downgrade rejects; canonical execution has no provider handle; reset/delete/
listing/model selection use the same resolver; quota failure has no dangling source
reference; raw inline/export evidence remains available for rollback; legacy
AutoCompact cannot run in candidate mode.

**Validation discovery.** Extend session/source/checkpoint/artifact suites under
`tests/session/`, P4 recovery/startup suites under `tests/agent/`, config migration
suites under `tests/config/`, and real-engine session fixtures discoverable via
`rg -l 'source_backend|source_schema_version|provider_state|Artifact' tests`. Use
copied directories only, fake provider, denied network/process, fake filesystem crash
points, and fresh-process reloads. Include corrupt header/version, symlink/root swap,
quota, unsupported export, and startup-negative cases.

**Correction/rollback boundary.** Before mode commit, delete/quarantine only the
staged target and continue with the untouched legacy snapshot. After mode commit,
canonical remains authoritative; rollback is via verified export, never file rename or
mtime selection. The production guard remains the fleet-wide rollback fence.

**Sizing.** This is one cohesive persistent-data package with a clear mode-commit
boundary; cron and runtime activation remain outside it.

### Stage 3 — dormant production cron composition and lifecycle

**Acceptance status (2026-10-01).** Root issued **PASS for guarded offline Stage 3
implementation**, after actual diff/code review, exact required gates, scope/hash
audit and independent narrow PASS, recorded in the
[acceptance checkpoint](p5-stage3-acceptance-checkpoint.md). The
[implementation contract](p5-stage3-implementation-contract.md) includes the
root-approved observation-bridge and exact-trigger-clock correction design. Initial
independent review found two defects; one bounded correction round resolved them.
No production timer, Dream, Heartbeat, callback, guard removal, deployment or Windows
qualification is authorized. This does not close P5 or D9.

**Objective.** Build production occurrence/completion/outbox/sender composition,
scheduler cursor/lease/restart/shutdown machinery, manual/system claim adapters, and
operator reconciliation surfaces while keeping the timer feature gate disabled.
Delete or isolate the unreachable legacy gateway callback.

**Observable behavior.** With a fake clock, an injected scheduler gate can classify
future, due, and missed fires; only a current lease holder creates a claim. The fake
provider and fake Telegram channel are each invoked once under their exact computation
and delivery claims; callbacks (when injected only in tests) follow durable completion,
and crashes leave explicit non-replay states. Gateway/CLI production construction
remains definition-only unless the disabled scheduler gate is explicitly injected in
tests. Dream and Heartbeat create no definition or provider call.

**Dependencies.** Stages 1-2; accepted D6-D7; P3 cron contracts.

**Ownership boundaries.** Cron owner: scheduler/cursor/lease, production composition,
manual/system adapters, restart/reconciliation, callback construction policy, and cron
tests. Integration owner: gateway lifecycle and removal of the old callback only.
Channel owner: exact Telegram sender adapter. No other owner edits cron store formats.

**Non-goals.** No armed production timer, non-Telegram delivery, rich/media result,
automatic stale-claim reconciliation, automatic retry, Dream/Heartbeat execution,
callback enablement/retention deletion, live send, guard change, or Windows lease.

**Acceptance criteria.** Next-fire rules match D6 across timezone/DST cases; missed
fires make zero provider/tool/send calls; competing owners yield one claimant; manual
and system sources cannot impersonate timer claims; started computation and uncertain
sends never replay; fanout continues in frozen order without recomputation; callback
failure/cancellation cannot replay or suppress settled delivery state; shutdown and
lease loss fence new claims; no production reference to legacy `on_cron_job` remains.

**Validation discovery.** Extend `tests/cron/` occurrence, completion, outbox,
delivery, callback, composition, and reconciliation suites plus CLI/gateway lifecycle
tests. Required matrices use fake clock (timezone/DST/restart), fake provider, fake
Telegram sender, copied cron stores, two-owner lease contention, crash points before/
after claim/begin/completion/callback/send/settle, cancellation-resistant cleanup, and
network/process denial. Legacy cron tests are classified per Section 6 before use.

**Correction/rollback boundary.** Disable/remove the injected scheduler facade and
retain definition-only service plus durable ledgers for inspection. Never roll back by
deleting uncertain claims/outboxes. The timer gate and global guard remain closed.

**Sizing.** This is the complete automation ownership graph, but its disabled gate and
Telegram-text boundary keep review and side effects bounded.

### Stage 4 — integrated offline acceptance, shadow projection, and recovery drills

**Current status (2026-10-01).** Guarded offline Stage 4 **ROOT PASS**, recorded in
the [acceptance checkpoint](p5-stage4-acceptance-checkpoint.md). Stage 5 closure/
preflight/P6 handoff is next, requires separate scope/authorization, and is **not
started**. This acceptance does not authorize P5 closure, activation or any later gate.

**Objective.** Validate the assembled foreground and dormant cron graphs against
copied state, classify stale/conflicting tests, and add content-free operational
diagnostics and shadow projections.

**Observable behavior.** A clean offline process starts through test-local guard
injection, serves fake direct/API/SDK/Telegram inputs through one runner, mixes legacy
and canonical sessions, restarts after injected crashes, and closes with no leaked
tasks. Dormant cron can be injected for fake-clock tests but is absent in the default
graph. Shadow comparison reads immutable/copies only and reports bounded mismatch
classes. Startup rejects all known unsafe/corrupt states before any external effect.

**Dependencies.** Stages 1-3.

**Ownership boundaries.** Integration owner owns the matrix and composition fixes;
session, cron, channel, provider, and release owners contribute isolated fixtures and
may correct only their established boundaries. A fresh reviewer inspects the joined
diff and raw evidence.

**Non-goals.** No live provider/channel/credential/state, deployment, wheel/service
rehearsal, guard removal, timer activation, Windows claim, H1, or P8.

**Acceptance criteria.** Fake direct/API/SDK/Telegram all call the same runner once;
same-session serialization and different-session concurrency hold across both modes;
crash/restart never duplicates tool/send/callback effects; startup-negative matrix is
effect-free; privacy-label tests reject identifiers/content; default startup has zero
background provider calls; all owned resources close; every selected failure is
classified rather than hidden.

**Validation discovery.** Build a manifest from the focused suites named in Stages
1-3 plus P3/P4 closure selections discoverable from their checkpoint documents. Run
each selection independently and report it independently. Add OS-level no-network
and no-process denial where loopback fixtures do not require local binds; use fake
transports otherwise. Exercise copied-state conversion/export/restart, fake provider,
fake channel, fake clock, crash points, startup negatives, and default-no-background
checks. Resolve the DOCX dependency as P6 packaging evidence or mark the exact case
environment-blocked; do not inject an undeclared dependency and call P5 green.

**Correction/rollback boundary.** Correct the owning stage and rerun its focused suite
before the integrated matrix. If a failure shows two authorities or nondeterministic
persistent ownership, reject the plan premise and return to planning; do not patch the
fixture. Guard and feature gates remain the rollback.

**Sizing.** This joins all P5 behaviors into one reviewable acceptance package without
crossing into release or live operations.

### Stage 5 — P5 closure artifact and activation-ready handoff

**Objective.** Produce a binary closure checkpoint, configuration/migration preflight
report, rollback/export runbook, P6 handoff manifest, and isolated proposed guard-
removal diff scope—without removing the guard or enabling any background authority.

**Observable behavior.** One offline command/report can say `READY` or `NOT_READY`
from fixed checks: config compatibility, state-mode inventory, unresolved recovery,
artifact policy, channel plugin conflicts, cron ledger/lease/timer state, dependency
availability, and disabled Dream/Heartbeat/callback/idle/rematerializer gates. Reports
are content-free and make no mutation. P6 receives exact artifact and rehearsal
inputs; P7 receives explicit single-consumer and rollback prerequisites.

**Dependencies.** Stage 4 PASS and fresh integrated review.

**Ownership boundaries.** Integration owner writes the P5 checkpoint/preflight;
release owner consumes (not silently changes) it in P6; operator owns later live
inventory/conversion/canary decisions. The guard file is read-only in this stage.

**Non-goals.** No commit/deploy, live preflight, state mutation, guard change, timer,
system jobs, callback, retention deletion, P6/P7 qualification, H1, or P8.

**Acceptance criteria.** Section 7's P5 definition is demonstrably true; the report
fails closed on every open gate; rollback identifies the authoritative mode at each
step; the guard-removal scope is isolated and contains no unrelated runtime behavior;
P6/P7 owners can execute their gates without reconstructing P5 history.

**Validation discovery.** Run the Stage 4 manifest, preflight unit/CLI fixtures under
`tests/migration/` and `tests/cli/`, documentation link validation, scoped Ruff/
compile/type checks, `git diff --check`, and a fresh diff review. Inspect the built
artifact only in P6; native Windows remains P6 evidence.

**Correction/rollback boundary.** A `NOT_READY` result leaves the guard and every
feature gate closed. Correct the owning earlier stage; never edit the report to waive
a failed check.

**Sizing.** This is substantial operational handoff and evidence consolidation, while
remaining independently reviewable because it changes no execution policy.

## 6. Test-evidence classification

Tests are classified, not deleted, renamed “obsolete,” or rewritten merely to turn a
selection green:

- **Current contract:** exercises an accepted invariant through a supported public or
  composition seam; it is required acceptance evidence.
- **Compatibility port required:** asserts a preserved behavior but calls a removed
  private ABI, import path, or constructor. Port only the fixture seam and retain its
  behavioral assertion.
- **Historical conflicting contract:** asserts behavior explicitly rejected by an
  accepted P3/P4/P5 decision, such as legacy jobs.json execution, destructive retry,
  or bypass of sender-first authorization. Record the exact conflict and keep it out
  of pass counts until a reviewed deletion/archival decision.
- **Environment/release blocked:** behavior is still required but a declared optional
  dependency, platform, or packaging artifact is absent. Record the raw failure; P6
  owns dependency/native qualification.
- **Guard-negative:** intentionally proves the real production guard. It must never be
  patched in that test.
- **Candidate-positive:** may use a narrowly scoped test-local guard injection and
  fake authorities, but must also prove no network/process/live credential use.

The existing legacy cron failures, removed private runner/context ABIs, old Telegram
package exports, DOCX dependency failure, and guard-gated injection fixtures must be
placed in these categories by node ID with owner and rationale. Counts remain per
command; overlapping selections are never summed.

## 7. Binary P5 definition of done

P5 is **DONE** only if all statements below are true with the global guard still
active; otherwise it is **NOT DONE**:

1. One production composition root supplies all foreground entrypoints and there is
   one turn/lock/delivery owner per request.
2. Session-mode admission, canonical-new default, mixed legacy/canonical operation,
   stopped-writer conversion, crash recovery, export, and downgrade rejection pass on
   copied state; no live state was touched.
3. Legacy AutoCompact, new idle work, rematerialization, callbacks, Dream, Heartbeat,
   and timer execution are all absent by default and proven absent.
4. Artifact opt-in/root/quota/rollback/operator contracts pass without automatic
   retention deletion.
5. Dormant cron production composition has one scheduler owner, deterministic fire/
   missed semantics, exact claim/execution/delivery ownership, restart/shutdown and
   reconciliation behavior, and no legacy callback path.
6. Packaged and explicitly configured external channels enter the same manager/turn
   path without shadowing or eager third-party code load.
7. Fake-clock, fake-provider, fake-channel, copied-state, crash-point,
   startup-negative, cancellation, privacy, and no-network/no-process matrices pass;
   every non-pass is classified and no required P5 case remains blocked.
8. Fresh independent review returns PASS; scoped lint/compile/type evidence is
   recorded, with unavailable tools explicitly unqualified.
9. P6 handoff and P7 rollback/canary prerequisites are complete, while P6, P7, H1,
   P8, native Windows, release packaging, deployment, and live traffic are not
   falsely claimed.

## 8. Separate guard-removal authorization checklist

Guard removal is not P5 implementation work. The user may authorize its isolated
change only after all of the following are recorded:

- P5 is DONE under Section 7 and D1-D8 are marked accepted;
- P6 has produced an immutable artifact/dependency inventory and passed copied-state,
  rollback, service-start, no-external-effect, and target-platform rehearsal;
- the exact artifact/commit, config instance, workspace, runtime-data root, rollback
  artifact, single gateway consumer, and operator are identified;
- live preflight reports no unresolved recovery/quarantine/conversion, plugin
  collision, nonzero legacy TTL, active scheduler lease, uncertain cron computation or
  send, and no omitted-true Heartbeat configuration;
- backups/exports are verified readable without changing authority;
- foreground feature flags permit only the intended direct/API/SDK/channel surfaces;
  timer, Dream, Heartbeat, callbacks, idle, automatic rematerialization, and retention
  deletion remain disabled;
- monitoring and stop/rollback thresholds are accepted, including content-free
  diagnostics and provider/channel spend limits;
- the proposed diff changes only the guard boundary and necessary explicit activation
  declaration; guard-negative coverage is replaced with equally explicit
  activation/preflight coverage, never silently deleted; and
- the user explicitly authorizes that exact guard-removal change. A later, separate
  user authorization is required before the single gateway timer lease is enabled.

After guard removal, P7 owns the single-consumer foreground canary, observation,
rollback/export drill, and promotion. H1 remains required before claiming the intended
busy conversational harness; P8 remains optional. No P7 success implicitly enables
cron system jobs, callbacks, idle compaction, artifact retention deletion, or a
non-POSIX callback backend.

## 9. Failure and recovery summary

| Failure | Required behavior | Recovery authority |
| --- | --- | --- |
| Composition/startup failure | Close partial resources; zero listener/timer/provider/send. | Fix config/dependency/state, then restart. |
| Session mode conflict/corruption | Quarantine/fail before turn execution. | Existing explicit recovery plus copied-state inspection; no freshness heuristic. |
| Conversion crash before mode commit | Legacy remains authoritative; staged canonical state is ignored or quarantined. | Resume/abort the journal offline. |
| Conversion crash after mode commit | Canonical remains authoritative; finish projections only. | Canonical recovery/export tooling. |
| Artifact install/reference failure | No dangling published reference; preserve inline evidence or fail closed. | Exact operator inspection/rematerialization on copied state. |
| Scheduler lease loss/restart miss | No new claim; elapsed fires become missed without computation. | New owner acquires lease and advances durable cursor. |
| Started/uncertain cron computation | Never replay. | Operator reconciles terminal evidence. |
| Claimed/sending/uncertain delivery | Never resend automatically. | Accepted delivery reconciliation with exact predecessor. |
| Callback invoking/failed | Never reinvoke automatically; delivery state remains independent. | Content-free operator reconciliation; optional callback remains disabled initially. |
| Shutdown timeout/cancellation | Stop ingress/claims; preserve durable uncertain state; no “successful” cleanup fiction. | Restart admission and explicit reconciliation. |

This plan deliberately prefers explicit disabled seams and copied-state proof over a
second implementation path. The final activation act is therefore small, auditable,
and unable by itself to arm cron or other background spend.
