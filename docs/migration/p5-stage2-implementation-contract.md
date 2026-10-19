# P5 Stage 2 implementation contract

Status as of 2026-10-01: **accepted specification; guarded offline Stage 2 PASS**,
recorded in the [acceptance checkpoint](p5-stage2-acceptance-checkpoint.md).
Production remains inactive. The following prior evidence is historical; the
[integrity completion contract](p5-stage2-integrity-contract.md) supplies its
accepted decisions and repairs. The historical
[attempt checkpoint](p5-stage2-checkpoint.md) records the rejected first attempt.
The historical [repair checkpoint](p5-stage2-repair-checkpoint.md) records accepted
repair dependencies, the fixed falsey-injection defect, the remaining AC2/AC5/AC6
findings, the unapproved canonical SDK-clear choice, and the independently accepted
restoration of the Stage 1 cutoff. This contract implements only Stage 2 of
[the accepted P5 plan](p5-integration-activation-plan.md),
using [the accepted Stage 1 composition](p5-stage1-checkpoint.md) and preserving
[P4 closure](p4-closure-checkpoint.md). D1-D4 are accepted. D9, live-state work,
guard removal, deployment, background work, and Windows qualification are not
authorized.

## 1. Objective and current evidence

Deliver one durable session authority which routes each persistent key through a
versioned mode record, creates an absent session as canonical only at its first
admitted persistent write, keeps bootstrapped legacy sessions legacy, and supports
stopped-writer bootstrap, conversion, export, reset, delete, fork, and artifact
inspection/rematerialization on copied state. All production guards remain closed.

The implementation must preserve these observed facts:

- `RuntimeComposition.define()` is synchronous/read-only; `acquire()` is the only
  construction boundary. Its current default `Stage1UnavailableSessionAuthority`
  fails before `SessionManager`, store, provider, tool, plugin, or listener work.
- `SessionManager.__init__` eagerly constructs `JsonlSessionStore`, even with a
  custom store. `source_backend()` currently infers source mode from
  `source_schema_version`; it must no longer be the managed-runtime authority.
- `SourceSessionStore.upgrade_source_session()` replaces the active JSONL in place.
  Preserve that public compatibility API outside a managed namespace; do not use it
  as the managed authority switch.
- `SourceSessionStore`, `SourceRootAuthority`, `ExecutionOwner`, source CAS,
  checkpoints, and recovery markers are accepted P4 substrate. Reuse them; do not
  add another source log or execution-owner abstraction.
- Legacy turns have only process-local per-key asyncio serialization. Source turns
  also have a persistent per-key execution lock. Consequently conversion cannot
  infer legacy quiescence from the source lock alone.
- `ConfiguredArtifactReaderFactory` is already opt-in, quota-bound, and wired by
  `AgentLoop.from_config`; `ArtifactRematerializer` is dormant and injection-only.
- `sessions restore-workspace` directly constructs a manager and copies JSONL. It
  must not remain a canonical downgrade bypass in a managed namespace.
- The offline session baseline is 862 passing tests. The 45 failures in the noted
  legacy loop selection are guard-denied historical constructors; Stage 2 must not
  remove the guard or rewrite those fixtures merely to make them green.

## 2. Decisions and rejected alternatives

### 2.1 Accepted implementation decisions

1. There is one `DurableSessionAuthority` definition. Construction is read-only;
   acquisition obtains a lifetime, nonblocking cooperating-writer lease before it
   constructs exactly one `SessionManager`. Its close callback closes the manager's
   authority and releases the lease after the agent graph closes.
2. `RuntimeComposition.define()` defaults to that definition-only authority, built
   from its already captured workspace/session-root/preflight facts. An explicitly
   injected authority still wins. Tests which require the old cutoff inject
   `Stage1UnavailableSessionAuthority` explicitly. Entrypoint guard calls remain
   before `define()` and are not edited or patched in guard-negative tests.
3. Managed default-store access uses a `SessionModeResolver` owned by the manager.
   The existing `JsonlSessionStore` and `SourceSessionStore` remain the two storage
   implementations selected by it. Direct/custom-store compatibility outside a
   managed namespace remains legacy behavior.
4. Existing state is never implicitly bootstrapped during startup. An explicit
   stopped-writer bootstrap adopts valid existing legacy and canonical-v1 files.
   An empty, unregistered namespace is initialized only inside the first admitted
   persistent write. Reads, lists of a genuinely empty namespace, definition, and
   preflight create no session or registry state.
5. A lifetime writer lease serializes cooperating Stage 2 runtimes and offline
   operators. It does **not** prove that an older binary is stopped because older
   binaries do not know this lock. Bootstrap/conversion/export therefore remain
   copied-state/offline operator actions with an explicit old-writer-stop
   prerequisite.
6. Any generation/incarnation or authority-changing operation is journaled. The
   current mode record, not file presence, mtime, or an export, selects authority.
7. Artifacts retain the existing factory, producer allowlist, source binding,
   quotas, inline fallback, and blob-before-reference ordering. Stage 2 adds only a
   content-free report and an exact copied-state rematerialization adapter.

### 2.2 Rejected alternatives

- Mode from `source_schema_version`, file presence, mtime, or newest export: permits
  dual writers or silent rollback after a partial conversion.
- Convert/bootstrap all sessions during acquisition: mutates startup and makes one
  bad session fleet-wide admission risk.
- A second canonical store: duplicates accepted P4 CAS, execution, checkpoint, and
  artifact rules.
- Treat the per-source execution lock as proof that legacy writers stopped: it is
  invisible to an already running legacy turn or old binary.
- In-place upgrade followed by a best-effort mode write: a crash can leave no
  deterministic precommit legacy authority.
- Write both legacy and canonical histories: violates one-writer ownership.
- Automatic artifact rematerialization/retention: violates D4 and the accepted P4
  dormant boundary.

## 3. Durable layout and record contracts

All names below are relative to the selected session namespace. Files are private,
regular, single-link, no-follow, bounded JSON; writes use exclusive staging, file
fsync, atomic replace where replacement is allowed, and parent-directory fsync.
Unknown fields or versions fail closed. JSON rejects duplicate keys, booleans as
integers, NaN, and noncanonical encodings.

```text
.session-mode/
  registry.json
  records/<storage-key>.json
  journals/<operation-id>/<sequence>-<phase>.json
  staging/<operation-id>/...
  rollback/<storage-key>/<operation-id>/legacy.jsonl
  exports/<operation-id>/...                 # only staging; final output is explicit
```

The second registry witness is
`<workspace>/.nanobot/session-mode-binding.json`. The writer lock is a persistent
file under the session root keyed by SHA-256 of the canonical workspace path, so it
can be acquired before `JsonlSessionStore` creates/claims a namespace. It is never
unlinked. Implement it with the repository's checked, no-follow ownership/mode
rules. Binding the resulting namespace and `workspace_id` is part of acquisition.

`registry.json` and the workspace witness contain exactly:

```json
{
  "schema_version": 1,
  "registry_id": "32 lowercase hex",
  "workspace_id": "existing 32 lowercase hex workspace identity",
  "namespace": "workspace_id",
  "state": "preparing|active",
  "bootstrap_operation_id": "32 lowercase hex or null"
}
```

Both witnesses must agree. Neither witness plus no `.session-mode` residue means
genuinely unregistered state. Exactly one witness, differing fields, `preparing`
without its valid journal, or registry residue without witnesses is partial state
and rejects runtime. Once an active registry exists, a JSONL without a mode record
is lost mode evidence, never unrecorded legacy. Deletion of both witnesses and all
registry evidence is information-theoretically indistinguishable from a never-
bootstrapped namespace; same-UID/manual destruction is outside the cooperating-
writer guarantee and must be stated in the operator warning.

An active `records/<storage-key>.json` contains:

```json
{
  "schema_version": 1,
  "registry_id": "...",
  "record_revision": 1,
  "session_key": "exact unsanitized key",
  "storage_key": "existing reversible base64url storage key",
  "state": "active|deleted|quarantined",
  "mode": "legacy|canonical|null",
  "source_schema_version": "null or 1",
  "incarnation": "exact session incarnation",
  "generation": 0,
  "transition_revision": 1,
  "transition_digest": "lowercase SHA-256 of the JSONL installed by the committed operation or null",
  "committed_operation_id": "32 lowercase hex",
  "committed_at": "timezone-aware ISO 8601"
}
```

`mode` is null only for `deleted`/`quarantined`. An active record must match the
parsed header's key, incarnation, generation, and source schema. It must also point
to a valid journal whose `mode_committed` target binding, revision, and digest equal
the record's transition fields. `transition_digest` is commit evidence for that
authority/generation transition, not a checksum that ordinary turns must keep
current. If the live header revision equals `transition_revision`, its exact bytes
must still match `transition_digest`; a lower revision is always a conflict. If the
live revision is greater, the digest is not compared: the selected legacy or strict
source parser, existing revision/CAS grammar, and unchanged incarnation/generation
govern those later ordinary writes. An incomplete journal always takes precedence
and applies Section 7's exact predecessor/target digest rules before this steady-
state rule. Unsupported record or source versions classify `unsupported`;
corrupt/partial evidence classifies `quarantined`; neither admits turns or ordinary
mutation.

Each operation journal is an append-only predecessor-digest chain. Every record
contains `schema_version`, `registry_id`, `operation_id`, `sequence`, `phase`,
`operation` (`bootstrap|create|convert|reset|delete|fork|export`), source/target
keys, prior mode-record digest/binding, target binding, exact input/output
digests/sizes and relative staging/rollback names, a timezone-aware timestamp, and
`previous_record_digest`. Valid phases are `prepared`, `target_installed`,
`mode_committed`, `published`, `complete`, or `aborted`; an operation uses only the
applicable ordered subset. Records are create-only. Missing sequence, unknown phase,
bad predecessor, ambiguous temp residue, or a fact that matches neither predecessor
nor target fails closed for explicit recovery.

Routine metrics/logs expose only fixed operation/mode/phase/result enums. They must
not include keys, paths, IDs, digests, prompts, events, references, or credentials.

## 4. Admission, locking, and compatibility

Lock order is fixed: lifetime writer lease; artifact transaction when applicable;
per-session execution lock; `.session-files.lock`. Bootstrap also uses the existing
workspace migration lock before the session-files lock. Do not acquire these in the
opposite order. Mode/journal publication occurs while the applicable storage lock is
held. No lock is removed on cleanup.

The lifetime lease is nonblocking at runtime acquisition: contention fails before
bus/provider/tools/channels. Operator bootstrap/conversion/export also fails busy;
it never waits for a possibly live runtime. Before those operators proceed they
require an explicit offline/copy assertion and print/return the qualification that
old binaries do not honor the lease. They must reject the configured live namespace
when invoked in copied-state mode.

At turn admission, after the effective key is known and its existing asyncio lock is
held, but before approval consumption, run-store mutation, provider construction/use,
tool dispatch, or delivery effects:

- ephemeral/fresh nonpersistent work does not create a mode record;
- an active legacy record enters the existing legacy path;
- an active canonical record loads through strict source validation and then uses
  the existing `ExecutionOwner`/`SourceTurnWriter` path;
- an absent key in an active registry runs journaled canonical creation, commits an
  empty canonical-v1 source and then its mode record, and only then admits the user
  write; and
- converting/exporting, journal residue, missing record, corrupt, unsupported, or
  conflicting evidence fails content-free before the listed effects.

If the whole namespace is empty and unregistered, first-write creation first commits
the paired active registry, then follows canonical creation. A read must return
absent without doing this. Canonical sessions never load or persist provider
conversation state. Ordinary legacy sessions retain the accepted P4 provider-state
behavior. Provider code is not changed except for focused assertions of that rule.

All managed manager entrypoints call the resolver before consulting cached session
objects. Cache entries bind `(registry_id, record_revision, mode, incarnation,
generation)`; every journal commit, conversion, reset, delete, fork target, or
bootstrap invalidates affected entries. `record_revision` changes only when the
mode record itself changes. An ordinary legacy append/metadata write or canonical
source append, checkpoint write, or execution-owner lease acquire/release does not
rewrite the mode record or invalidate it merely because the JSONL revision/digest
advanced; existing `Session`/source revision CAS continues to govern the cached
content. The resolver rechecks stable mode/incarnation/generation bindings without
using mtime or requiring the transition digest to equal later bytes. Direct
`SessionManager`, `JsonlSessionStore`, `upgrade_source_session`, and custom stores
preserve existing test/embedder behavior only when no active managed registry claims
that namespace. When a registry claims it, unmanaged mutation and
`restore_to_workspace()` reject with a fixed error; they cannot bypass the resolver.
Reads may use a narrow resolved facade, not raw format inference.

## 5. State transitions and operations

### 5.1 Bootstrap and creation

Bootstrap is explicit and stopped-writer only. Under the lease it enumerates bounded
canonical filenames, rejects symlinks/unknown/conflicting entries, strictly parses
every header, validates canonical-v1 source files, and creates legacy/canonical mode
records from exact identities and digests. It publishes `preparing`, all records and
the workspace witness, then `active`. A crash before `active` leaves runtime blocked;
rerunning the same bootstrap operation may publish only missing, byte-identical
steps. It never converts, repairs, or loads provider/tool code.

First-write creation writes `prepared`, installs and verifies an empty canonical
source with a new incarnation/generation zero, writes `target_installed`, commits the
canonical mode record, then `mode_committed`/`complete`. Before mode commit, restart
quarantines/removes only the exact staged/installed target and returns to absent.
After commit, canonical remains authoritative and recovery only finishes the journal.

### 5.2 Legacy-to-canonical conversion

Conversion requires an active legacy record, no runtime checkpoint, pending turn or
follow-up, unresolved recovery/receipt, active execution evidence, or artifact/source
conflict. It snapshots exact legacy bytes to the rollback path, fsyncs and verifies
them, records `prepared`, converts through the existing strict legacy-to-source
transform, stages and verifies canonical-v1 bytes, installs them at the active
JSONL name, and records `target_installed`. Only then does it atomically replace the
mode record with canonical and record `mode_committed`. The rollback file becomes
immutable retained evidence; no Stage 2 path deletes it.

Before mode commit, legacy is authoritative. Recovery restores the exact verified
rollback snapshot to the active name when the canonical target was installed,
quarantines the target residue, records `aborted`, and leaves the legacy mode record
unchanged. Before installation it may simply abort identical staging. After mode
commit, canonical is authoritative forever; recovery validates/finishes journal and
exports only, never restores legacy due to an older file. Retry is exact/idempotent;
ambiguous bytes fail closed. The compatibility `upgrade_source_session()` remains
available only outside managed namespaces.

### 5.3 Reset, delete, fork, metadata, list

- Reset journals prior/target bindings. Legacy uses its conflict-safe replacement;
  canonical uses `reset_source_session`. After the data generation advances, commit
  the matching mode record. A crash in between is resolved only from the journal;
  it is never interpreted by mtime. Artifact bytes remain; source reset clears the
  manifest as already accepted.
- Delete commits a `deleted` tombstone first, then removes the exact prior active
  JSONL and sidecar and records completion. After tombstone commit the key is
  logically absent even if cleanup remains. A later first write gets a new
  incarnation and canonical mode. Artifacts and rollback material are retained.
- Fork resolves the source mode and requires an absent/deleted target. It preserves
  the current public-prefix behavior, removes volatile metadata, and creates a
  canonical target via a create/fork journal. Canonical source prefixes retain valid
  source linkages; legacy prefixes pass the strict upgrade transform. An ambiguous
  legacy tool history fails without a target or mode record.
- Per-session model selection and metadata update dispatch to the strict source
  metadata API for canonical and existing metadata transaction for legacy. Bulk
  preset rename preserves existing best-effort rollback semantics but resolves and
  validates every key before its first write; converting/exporting/conflict aborts
  before mutation.
- Listing enumerates registry records and backing files as one bounded consistency
  check, then uses strict source projection for canonical and the ordinary reader
  for legacy. It performs no repair/bootstrap. One missing/extra/conflicting record
  fails the operation. Deleted tombstones are not listed.

The SDK/WebUI delete, fork, model selection, reset, direct reads, and session-handle
paths continue to call `SessionManager`; do not add consumer-local mode checks.

### 5.4 Export and downgrade proof

Export is stopped-writer and journaled as `exporting`; turns/mutations reject while
its incomplete journal exists. A canonical lossless bundle contains exact
`canonical-source.jsonl` bytes and a canonical manifest binding registry/mode,
incarnation/generation, source/checkpoint schema, artifact references, file digest
and size. Inline source evidence remains present; private blobs are not silently
claimed portable. The final bundle is installed only at an explicit absolute,
nonexistent operator destination after every file and the directory are verified
and synced. Export never changes mode.

A separately labelled `compatibility-history.jsonl` may be included only when a
pure converter proves that the target legacy grammar can represent every public
event, tool call/result pairing, portable durable metadata value, archive boundary,
and inline artifact evidence. It must parse the produced JSONL with the legacy
reader and compare the normalized public transcript/metadata to the canonical
input. Active/recovery-required or unresolved executions, source-only evidence with
no inline representation, unsupported roles/metadata, or any unequal round trip are
`unsupported`; compatibility output and downgrade fail, the journal aborts, and the
mode record is byte-identical. The manifest labels this file `compatibility`, never
`canonical` or `lossless`.

`sessions restore-workspace` retains old behavior only for an unmanaged legacy
namespace. A managed namespace directs the operator to the verified exporter and
cannot copy a canonical active file as a downgrade.

## 6. Artifact operator boundary

Do not change the default, quotas, producer set, inline fallback, or runtime root:
enabled artifacts without an explicit root use the captured config instance's
`runtime_data_dir / "context-artifacts"`; explicit roots remain absolute and pass
the existing overlap/symlink validation. A blob is durable before a descriptor is
published. Quota/install/publication failure publishes no dangling reference and
retains bounded inline evidence or fails the already accepted turn boundary.

Add one service with thin `sessions artifacts inspect` and `sessions artifacts
rematerialize` CLI adapters. Both require explicit copied sessions root and absolute
copied artifact root, reject identity with the configured live roots, acquire the
copy's writer lease, resolve one exact canonical record, and perform no provider,
tool execution, network, startup registration, or source write. Inspect reports only
fixed status plus aggregate reachable/unreachable blob counts and bytes. It does not
return content, paths, session/reference identifiers, or delete anything.
Rematerialize additionally requires exact key, incarnation, generation and opaque
reference supplied by the operator. It must not acquire an `ExecutionOwner` against
the operator's input copy: `acquire_execution()` and owner close legitimately write
lease/revision bookkeeping, so doing that would violate P4's read-only source
contract even when the artifact operation itself is read-only.

Instead, while the copied input is locked, validate and read its exact canonical
bytes and mode binding, then install those bytes into a fresh private scratch
workspace/session namespace with the same `workspace_id`, key, incarnation, and
generation. The scratch namespace is never registered as a managed authority and is
never used for a turn. Acquire the real existing `ExecutionOwner` only there,
construct an invocation-local reader through the existing configured policy, and
call the existing `ArtifactRematerializer` against the scratch snapshot while its
owner is active. The destination `PrivateArtifactStore` is the explicitly supplied
copied artifact root and retains the original workspace/session/incarnation/
generation namespace binding. Thus P4 authorization is neither bypassed nor faked;
only disposable scratch source receives execution-lease revisions. Before reporting
success, re-read the locked input copy and require its full source bytes and mode
record to equal the pre-scratch values, and verify the destination blob through the
existing store. A scratch construction/owner/rematerialization failure leaves the
input and its mode record unchanged and publishes no source reference. No call site
is added to startup, reads, loop, projection, checkpoint admission, or automatic
recovery.

## 7. Crash/restart verdicts

| Last durable fact | Restart authority and action |
| --- | --- |
| Bootstrap absent; no witnesses/residue | Genuine unregistered namespace; runtime admits only empty reads, existing data requires explicit bootstrap. |
| Bootstrap preparing/partial | No runtime authority; exact operator resume only. |
| Create target installed, no mode commit | No session authority; remove/quarantine exact target and return to absent. |
| Conversion prepared/staged | Legacy; discard/quarantine exact stage and abort. |
| Conversion target installed, no mode commit | Restore verified legacy snapshot, quarantine target, retain legacy record. |
| Canonical mode commit durable | Canonical; never roll back, finish journal/projection only. |
| Reset data installed, old mode binding | Reject; journal validates target and commits new binding, otherwise quarantine. |
| Delete tombstone committed, file remains | Deleted; finish exact cleanup, never read the file. |
| Fork/create target installed, no target mode | Source unchanged; remove/quarantine target and return target to absent. |
| Export staged/published, no completion | Session mode unchanged; validate and finish exact publication or abort residue. |
| Artifact blob installed, no source reference | Blob unreachable and retained; no reference fabricated. |
| Source reference committed | Reference authoritative; missing blob is operator-rematerializable only from exact inline source. |
| Unknown/corrupt journal or record | Quarantined/unsupported; zero turn/provider/tool action. |

No recovery path retries a provider, tool, delivery, conversion transform on changed
input, or artifact publication.

## 8. One cohesive implementation package

Use one `development_worker_high` package because the journal, resolver, manager,
composition default, and operations share an irreversible commit boundary. Splitting
them would create a period in which consumers can bypass incomplete authority.

**Workspace:** `/Users/jaden/Documents/github/nanobot-harness-candidate`

**May edit:** new session mode/authority/conversion/export/artifact-operator modules;
`nanobot/session/manager.py`, `source_store.py`, `runtime_authority.py`;
`nanobot/runtime_composition.py`; the narrow admission seam in
`nanobot/agent/loop.py`; session operator commands in `nanobot/cli/commands.py`;
necessary session exports; and focused new/updated tests under `tests/session/`,
`tests/migration/`, `tests/agent/`, `tests/cli/`, and `tests/providers/`.

**Must not edit:** migration guards or their ordering; provider implementations;
P4 source/checkpoint/execution/artifact semantics except the explicitly named reuse
seams; cron, channels, idle runtime, Dream, Heartbeat, callbacks, timers, recovery
receipt policy, unrelated tests, configuration defaults, dependencies, or existing
migration checkpoints. Preserve all unrelated dirty work.

**Red-green sequence:**

1. Add failing layout/parser/resolver tests, paired-witness/bootstrap cases, and
   acquisition contention/default-cutoff tests.
2. Add failing first-write and mixed-mode actual-engine tests with fake provider,
   denied network/process, and explicit test-local guard injection.
3. Add table-driven crash tests at every journal phase, each using a fresh copied
   directory and fresh manager process/reload.
4. Add failing operation/export/downgrade and unmanaged-bypass tests.
5. Add failing artifact root/quota/publication/report/exact-rematerialization tests.
6. Implement the smallest authority/resolver/coordinator, then the narrow
   composition/admission/CLI adapters. Refactor only after focused green.

## 9. Acceptance criteria

- **AC1:** `define()` and reads are byte/inode/mtime stable. First admitted
  persistent write creates exactly one canonical session/mode; ephemeral work does
  not. Guarded production still fails at the original guard before composition.
- **AC2:** Explicit bootstrap adopts valid copied legacy/canonical fixtures without
  changing their JSONL bytes. No implicit bootstrap occurs. Partial/lost/conflicting
  witness or record evidence rejects before runtime effects.
- **AC3:** Two cooperating authorities yield one lifetime writer; conversion while a
  runtime owns it fails busy. Documentation/tests explicitly do not claim old-binary
  exclusion from this lock.
- **AC4:** Mixed legacy/canonical keys route only through their records. Legacy keeps
  provider state; canonical has none. Source execution retains P4 owner/recovery,
  checkpoint, and artifact semantics.
- **AC5:** Every injected crash/reload has exactly the verdict in Section 7. Before
  conversion mode commit legacy bytes/record remain authoritative; after commit
  canonical never rolls back. Retrying exact recovery is idempotent.
- **AC6:** Reset, delete, fork, list, direct read, model selection/rename, SDK/WebUI
  consumers, and cache invalidation use the resolver. Raw upgrade/restore/unmanaged
  mutation cannot bypass an active managed registry; custom/unmanaged compatibility
  outside it remains green.
- **AC7:** Lossless export verifies exact canonical bytes and metadata. Compatibility
  output is visibly labelled and produced only after successful round-trip proof.
  Unsupported/lossy downgrade changes neither mode nor active bytes.
- **AC8:** Artifact disabled mode creates no storage. Enabled roots/quotas retain
  current behavior; publication never precedes durable blob install. Failure leaves
  no dangling reference and preserves inline source evidence.
- **AC9:** Artifact inspection is aggregate/content-free. Exact rematerialization
  operates only on explicit copied roots and exact identity/state/reference, changes
  no input source or mode-record byte, confines real execution-owner bookkeeping to
  a private scratch clone, and has no automatic caller or retention deletion.
- **AC10:** Nonzero legacy TTL and enabled idle runtime remain preflight errors;
  legacy `AutoCompact`, I1/I2/I3, timer, Dream, Heartbeat, callback, retention, and
  rematerializer startup construction remain absent.
- **AC11:** Routine errors/metrics contain no session/user/path/content/reference/
  digest/credential labels. Malformed input failures are content-free.
- **AC12:** The focused and full-session offline selections pass; dependency-blocked,
  BasedPyright, and native Windows evidence is reported honestly, never promoted.

## 10. Exact validation selection

Use no live config/state/credentials/network, install no dependency, and operate only
on temporary copied fixtures:

```bash
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/session/test_session_mode_registry.py \
  tests/session/test_session_mode_conversion.py \
  tests/session/test_session_mode_export.py \
  tests/session/test_session_mode_operations.py \
  tests/session/test_artifact_operator.py \
  tests/migration/test_p5_stage2_composition.py \
  tests/migration/test_p5_stage2_crash_matrix.py \
  tests/agent/test_p5_stage2_engine.py \
  tests/cli/test_p5_stage2_sessions.py \
  tests/providers/test_conversation_state.py --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/session --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/migration/test_p5_stage1_composition.py \
  tests/migration/test_p5_stage1_shutdown.py \
  tests/migration/test_p5_stage1_adapter_matrix.py \
  tests/cli/test_startup_admission_agent_gateway.py \
  tests/cli/test_startup_admission_onboard_webui_api.py --tb=short

/Users/jaden/.pyenv/versions/3.13.1/bin/ruff check \
  nanobot/session nanobot/runtime_composition.py nanobot/agent/loop.py \
  nanobot/cli/commands.py tests/session tests/migration tests/agent tests/cli tests/providers

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
  -m compileall -q nanobot/session nanobot/runtime_composition.py nanobot/agent/loop.py \
  nanobot/cli/commands.py

git diff --check
```

Test filenames listed as new are required deliverables. Do not run `ruff format`.
BasedPyright and native Windows remain unqualified in this environment. The worker
must return exact command outcomes, crash-point coverage by phase, files changed,
AC evidence, classified non-passes, and residual risks. Stage 2 is not accepted
until an independent integrated review inspects the diff and returns PASS.

## 11. Rollback, observability, and remaining gates

The global guard is the fleet rollback fence. Before a mode commit, recovery returns
to the verified predecessor. After a canonical mode commit, rollback is a verified
export into an older binary's separate copied state, never a mode-record edit,
rename, or mtime choice. Disable the Stage 2 default authority as one composition
package if implementation acceptance fails; never retain a partially routed manager.

No material D1-D4 product gate remains for this package. These later gates remain
explicit and do not block offline implementation: approval of each live bootstrap/
conversion batch; retention/removal of rollback or artifact material; any automatic
retention/rematerialization or idle policy; D9 guard removal; P6 packaging/native
qualification (including Windows); P7 canary/promotion; and all cron/background
activation. A Stage 2 PASS authorizes none of them.
