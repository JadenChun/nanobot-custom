# P5 Stage 2 integrity completion contract

Status as of 2026-10-01: **guarded offline Stage 2 PASS**; see the final
[acceptance checkpoint](p5-stage2-acceptance-checkpoint.md). Production remains
inactive. This contract closes the AC2/AC5/AC6 integrity findings recorded in the
[repair checkpoint](p5-stage2-repair-checkpoint.md). It supplements, and does not
replace, Sections 1-7 or AC1-AC12 of the
[Stage 2 implementation contract](p5-stage2-implementation-contract.md), the
[repair contract](p5-stage2-repair-contract.md), or the accepted
[P4 closure](p4-closure-checkpoint.md).

The implementation workspace is exclusively
`/Users/jaden/Documents/github/nanobot-harness-candidate` on branch
`codex/harness-migration-candidate`. Preserve all unrelated dirty work. This
contract authorizes no commit, install, live root/config/credential access,
network, deployment, guard change, background activation, Stage 3 work, or Windows
qualification.

## 1. Objective, evidence, and accepted decisions

Finish the offline Stage 2 authority by making the source/checkpoint inventory,
lifecycle journals, execution locking, SDK session operations, and WebUI metadata
consumers obey one persistent authority. Restore the definition-only durable
default only after root accepts the integrity and consumer packages as ready
dependencies. Package III then restores the guarded default for the exact fan-in
and review. Any final fan-in/review/root rejection restores the Stage 1 cutoff; a
root Stage 2 PASS is the only final acceptance.

The following defects are confirmed implementation evidence, not questions for a
worker to rediscover:

- `SessionModeResolver._source_names()` accepts checkpoint basenames but returns
  only JSONL names. Bootstrap journals only JSONL bytes, and `list_modes()` compares
  only records with JSONL files. An orphan sidecar and a canonical JSONL plus
  sidecar can therefore escape the durable inventory.
- Bootstrap master and per-record `prepared` rows omit exact sidecar evidence.
  Managed legacy reads reach `JsonlSessionStore._overlay_runtime_checkpoint_unlocked()`,
  which uses mtime, unlinks stale/malformed evidence, and logs the key and exception.
  That is incompatible with exact, content-free managed recovery.
- Legacy reset does not bind or remove an existing sidecar. Delete has exact
  sidecar fields but enters only the files lock. Fork reads/builds before the
  resolver files lock and journals no exact source record, binding, bytes, or
  sidecar proof.
- Delete and fork omit the execution-before-files guard. A live canonical owner can
  otherwise rewrite after a tombstone or race the fork snapshot.
- `SessionClient.ingest()`, `restore()`, and `clear()` mutate a cached `Session`
  then call legacy `save()`. Managed canonical save creates an absent canonical
  record and then rejects replacement. WebUI generated/fork titles and persisted
  workspace scope use the same invalid path.
- The actual default is intentionally rolled back to
  `Stage1UnavailableSessionAuthority`. Explicit injection already uses
  `is not None`; the Stage 2, session, and Stage 1 selections recorded in the
  checkpoint are green but do not test the defects above.

### Accepted product and architecture decisions

1. **Synchronous SDK clear is retained.** A manager-owned, out-of-turn canonical
   reset uses the existing `execution_transaction`, one frozen `SourceResetPlan`,
   nonblocking busy rejection, and recovery-required rejection. There is no async
   replacement API and no second execution-owner abstraction.
2. In-turn canonical `/new` retains its current real `ExecutionOwner`. It quiesces
   that owner, uses the same manager/resolver reset protocol, and invalidates the
   owner only after the destructive source write.
3. A saved import into **any managed namespace**, including a bootstrapped legacy
   key, is unsupported and fails early with one fixed content-free error. Stage 2
   adds no persistent import/migration format. Unmanaged saved ingest/restore stays
   compatible.
4. `save=False` SDK ingest/restore is detached: it constructs and returns a new
   snapshot without resolving/creating authority, changing manager cache, or
   writing state. This applies to managed and unmanaged runtimes.
5. An SDK clear of an absent or deleted managed key preserves the public “persist
   an empty session” behavior by admitting one empty canonical generation-zero
   creation. It does not create and immediately reset a second generation. An
   active key performs the journaled reset.
6. The existing accepted canonical-fork rule is unchanged: exact public event
   prefix and linkages are retained; `artifact_manifest` and
   `context_checkpoint` are dropped; descriptors/blobs are not copied or rebound.
7. C-R export and D artifact-operator implementation are frozen dependencies.
   Their product behavior and files do not reopen. The shared journal parser may
   gain exact nullable evidence fields; existing export calls receive canonical
   null defaults without changes to `session_export.py`.
8. The lifetime writer lease remains the lifecycle prerequisite. Per-key lock
   order is fixed below and applies inside resolver public methods, so direct
   resolver calls cannot bypass it.
9. All other accepted decisions, non-goals, later rollout gates, and AC1-AC12 remain
   in force. No material product or operational gate remains for offline dispatch.

### Rejected alternatives

- Do not add async `clear()`, a second owner/capability, a second reset
  implementation, or consumer-local mode checks.
- Do not infer a sidecar from mtime, delete/repair it on a managed read, ignore an
  orphan, or allow a canonical source with a legacy sidecar.
- Do not acquire files before execution, rely on manager-only locking, or leave
  resolver direct calls unguarded.
- Do not prebuild a fork outside its continuous lock, rerun a transform on changed
  input, or treat target bytes alone as source provenance.
- Do not make managed ingest/restore a hidden conversion or add a new persistent
  import operation.
- Do not restore the durable default before both implementation packages and final
  focused activation tests are accepted.

## 2. Exact inventory and journal contract

### 2.1 Paired source inventory

Introduce one internal immutable inventory value per storage key, assembled under
the pinned namespace and `.session-files.lock`:

```text
key, storage_key,
source_name, exact source bytes, SHA-256, size,
sidecar_name, sidecar_state (absent|present),
exact sidecar bytes/digest/size when present
```

Enumeration is bounded and returns sorted pairs. A basename must round-trip through
the existing reversible storage-key codec. The only legal pairs are:

- legacy JSONL plus no sidecar; or
- legacy JSONL plus one strictly valid version-2 checkpoint sidecar; or
- canonical-v1 JSONL plus no sidecar.

A sidecar without its JSONL, a canonical JSONL with a sidecar, duplicate/colliding
stems, an unsafe entry, or an unknown namespace entry fails content-free. There is
never an implicit repair.

The strict sidecar parser is pure and consumes exact source bytes plus exact
sidecar bytes. It requires the current version and exact fields
`version`, `session_key`, `base_updated_at`, `base_message_count`,
`base_revision`, `base_generation`, `base_incarnation`, `checkpoint`, and
`provider_state`; it validates each binding against the parsed legacy session and
validates provider state through the existing private-record parser. Unknown or
missing fields, booleans as integers, malformed values, or a mismatch fail. It
does not inspect mtime, touch the filesystem, mutate a cached session, log a key or
path, or suppress an error.

Managed bootstrap, list, load, reset, delete, fork, and recovery use this paired
inventory/parser. Unmanaged legacy load retains its current compatibility cleanup
behavior. Split the managed pure path from
`JsonlSessionStore._overlay_runtime_checkpoint_unlocked()`; do not silently change
unmanaged behavior.

### 2.2 Bootstrap and list proof

The master bootstrap inventory bytes canonically encode, for every sorted key, the
source name/digest/size and explicit sidecar state plus digest/size when present.
The master `prepared.input_digest/input_size` bind those exact inventory bytes.
Every per-record bootstrap `prepared` row binds the exact source bytes through
`input_digest/input_size` and the exact sidecar bytes through the existing
`sidecar_*` fields; all-null sidecar fields mean **verified absence under the same
lock**, never “not inspected.” Resume rereads and byte-compares both parts before
publishing a missing record or active witnesses.

`list_modes()` enumerates records, JSONL names, and checkpoint names in one bounded
consistency pass. It requires one active record for every legal pair and no live
pair for a deleted record. It strictly parses the current source and, for legacy,
the current optional sidecar. Ordinary valid legacy sidecar advancement is checked
against the current JSONL rather than compared to old bootstrap bytes. Listing is
read-only and never starts recovery or removes evidence.

### 2.3 Operation-specific journal evidence

Keep journal schema version 1 because no Stage 2 format has shipped, but expand the
exact field set with nullable `source_record`, `source_record_digest`,
`source_binding`, `source_input_digest`, and `source_input_size`. `_append_journal()`
emits all fields on every row and copies the immutable first-row values. Parsers
reject missing fields and require nulls for operations that do not use them.

- `bootstrap`, `reset`, and `delete` continue to use
  `input_digest/input_size` for the affected active JSONL and `sidecar_*` for its
  exact present bytes or verified absence.
- `fork` uses `source_record*` and `source_binding` for the exact active source mode
  record, `source_input_*` for the exact source JSONL, and `sidecar_*` for the exact
  source sidecar or verified absence. Existing `prior_record*` remains target
  predecessor/tombstone evidence; `output_*` remains the frozen target JSONL.
- Export/create/convert fields not applicable to their accepted protocols are null.
  C-R continues to use only the public export transaction.

Every canonical producer/consumer boundary proves sidecar absence under the same
lock: create proves source and sidecar absent before preparing and before install;
convert proves the legacy sidecar absent before freezing canonical output; canonical
reset proves sidecar absence before planning and before mode commit; fork proves a
canonical source has no sidecar and the target sidecar is absent before prepare,
install, and record commit; export proves its canonical snapshot has no sidecar.
Recovery rejects a sidecar at any canonical source/target. A legacy fork may bind a
strictly valid source sidecar as source evidence, but its canonical target still
requires verified sidecar absence.

No operation-owned rollback/staging entry may precede its durable `prepared` row.
Create, convert, fork, and export first compute immutable bytes and names in memory,
append `prepared` with their exact digests/sizes/names, and only then create the
bound rollback/stage entries. A fault before `prepared` therefore leaves no
operation residue. A fault after `prepared` may leave only names and bytes exactly
bound by that row; recovery removes, resumes, or quarantines only from that proof.
Reset has no stage and writes only its frozen target after `prepared`. This ordering
is implemented in the shared resolver transaction, including `begin_export()`;
`session_export.py` remains unchanged.

The parser validates embedded record text against its digest and binding. A fork
row is invalid unless the source record text produces `source_binding`, the key is
the row's `source_key`, and the source data/sidecar fields are complete. Recovery
never fills missing proof from current files.

## 3. Locking, transaction, and recovery invariants

The universal order is:

```text
lifetime lifecycle lease
  -> sorted per-session execution lock(s), or current owner mutex/held lock
    -> .session-files.lock
```

Artifact transactions retain their already accepted position between lifetime and
execution locks when applicable. Bootstrap alone also retains the migration lock
before `.session-files.lock`. No journal, source, sidecar, record, cache, or
consumer mutation may begin before the applicable locks are held.

### 3.1 Reset

`SessionModeResolver.reset()` is no longer wrapped by a files-lock-first decorator.
It enters exactly one existing `execution_transaction` across resolution, pure
sidecar validation, frozen plan creation, `prepared`, data install, exact sidecar
cleanup, mode-record commit, and journal completion:

- with the supplied current `ExecutionOwner` and `destructive=True` for in-turn
  canonical reset; or
- ownerless, nonblocking, `allow_legacy=True`, and `destructive=True` for manager
  out-of-turn reset.

The owner mutex is entered before `.session-files.lock`. Busy lock, live lease,
abandoned lease, `execution_recovery_required`, unresolved execution, or canonical
runtime-checkpoint/sidecar evidence rejects an **ownerless out-of-turn canonical**
reset before a journal or cache change. A legacy reset instead accepts an exact
strictly valid legacy sidecar and binds/removes it through the cleanup protocol
below. A current-owner in-turn `/new` retains the accepted P4 behavior: the caller
first drains and quiesces its real owner, then the reset may clear that owner's
unresolved/recovery state through the frozen destructive plan. It is not rejected
merely because its own valid lease or P4 recovery evidence exists.

Avoid nested lock acquisition explicitly. Refactor
`SourceSessionStore.reset_source_session()` into its unchanged public transaction
wrapper plus one private `_reset_source_session_unlocked(...)` apply function. The
private function revalidates the expected state and exact `SourceResetPlan`, writes
the frozen target once, and invalidates a supplied real owner; it never acquires an
execution or files lock. The resolver calls that private function only while its
single transaction is active. This is a reuse seam, not a new owner abstraction.

Before `prepared`, reset captures exact predecessor source, record, and sidecar
evidence and freezes target bytes/timestamp/state. Installation order is target
source, exact sidecar unlink plus directory durability, then `target_installed`.
Recovery rules are deterministic:

- exact predecessor source plus exact predecessor sidecar: abort before commit;
- exact target source plus exact predecessor sidecar: remove only that verified
  sidecar and continue;
- exact target source plus verified sidecar absence: continue;
- exact predecessor source plus unexpected sidecar absence, changed sidecar, or
  source matching neither predecessor nor target: quarantine/fail closed;
- exact target mode record: complete forward and never regenerate target bytes.

The same rules clear stale legacy sidecars. No recovery path reruns the reset
transform or chooses a timestamp again.

### 3.2 Delete

`SessionModeResolver.delete()` enters one ownerless
`execution_transaction(..., allow_legacy=True, destructive=True)` before the files
lock and retains it through exact inventory, `prepared`, tombstone replace,
`mode_committed`, source/sidecar unlink, directory durability, and `complete`.
Canonical busy/recovery evidence rejects before `prepared`. The tombstone remains
the logical commit point. Recovery after it verifies the exact journaled source and
sidecar before removing either; changed or missing-before-authorized-cleanup
evidence fails closed. An active owner can no longer write after tombstone because
the same persistent execution-lock inode is held until cleanup finishes.

### 3.3 Fork

`SessionModeResolver.fork()` owns source selection and target construction; the
manager passes `source_key`, `target_key`, and `before_user_index`, not prebuilt
bytes. The resolver acquires distinct source and target execution locks in sorted
storage-key order with the existing nonblocking `execution_lock_guard`, then one
`.session-files.lock`. It rechecks source active state, target absent/deleted state,
execution/recovery evidence, source record, exact source JSONL, and optional strict
legacy sidecar under those locks.

It computes the approved prefix and frozen canonical target once from that exact
snapshot. It writes `prepared` with complete source and target predecessor proof
before staging or installing the target, stages/verifies the frozen bytes, then
installs target, commits its record, and completes while retaining both execution
locks. Immediately before target install it byte-compares the source record,
source JSONL, and source sidecar state with `prepared`. Recovery requires the same
source proof and uses only the journal-owned frozen stage/target; it never rebuilds
from newer source bytes. Any mismatch quarantines/fails closed. The source remains
unchanged.

This two-key ordering is mandatory for direct resolver calls and prevents reciprocal
fork deadlock. It does not claim exclusion of old binaries that ignore Stage 2
locks; the accepted stopped-writer qualification remains unchanged for offline
operations.

## 4. Manager and consumer contracts

### 4.1 Manager-owned APIs

Package I implements these complete seams; Package II may call them but may not
inspect mode or journals:

1. `SessionManager.reset_session(key, *, owner=None, replacement_events=None)` owns
   all clear policy. For absent/deleted managed state it calls
   `ensure_canonical()` once and returns the empty generation-zero session. For an
   active managed key it delegates to resolver reset. It invalidates/reloads cache
   only after commit; busy/recovery/failure leaves cache and file-state consumers
   unchanged. Unmanaged behavior remains compatible.
2. `SessionManager.fork_session_before_user_index()` delegates snapshot, prefix,
   transform, and target installation to resolver for managed state. It never loads
   or constructs managed fork bytes before the resolver transaction.
3. Add `SessionManager.upsert_session_metadata(key, updates, *, removals=(),
   create_if_absent=False, fsync=False) -> bool`. It validates/filter-copies input,
   resolves before cache access, atomically sets/removes metadata through strict
   source metadata mutation for canonical or conflict-safe legacy mutation, and
   invalidates/refreshes cache only after commit. With `create_if_absent=True`, an
   absent/deleted managed key is admitted once as empty canonical before metadata
   mutation; otherwise absence returns `False`. Reserved execution/source fields
   cannot be set or removed. Source-store metadata mutation may be extended only
   narrowly to accept validated removals under its existing storage transaction.
4. Managed reads call the pure exact sidecar parser. They never invoke the
   mtime/unlink/log behavior. Routine failures remain fixed and content-free.

### 4.2 SDK

In `nanobot/sdk/clients.py`:

- `ingest(..., save=True)` and `restore(..., save=True)` check
  `SessionManager.has_managed_authority` immediately after validating the explicit
  key and raise one fixed `RuntimeError("managed session import is unsupported")`
  before `get_or_create`, message iteration, cache access, or storage mutation.
- The same calls with `save=False` build a detached `Session` locally. They do not
  call manager get/create/save or authority resolution. Input validation remains
  unchanged and the returned snapshot reflects only the detached import.
- `clear()` calls `SessionManager.reset_session()` directly. Only after successful
  commit does it discard loop file-read state and return the committed empty
  snapshot. It performs no local `Session.clear()` or `save()` and exposes the
  existing fixed busy/recovery exception without content.
- Unmanaged saved ingest/restore/clear and all read/list/export APIs remain
  compatible.

### 4.3 WebUI

- `maybe_generate_webui_title()` reads routed/target metadata without creating or
  caching a target and persists cleaning/generated titles only through
  `upsert_session_metadata()`. Add one explicit private/default-false adapter
  argument such as `allow_projected_target_create=False`; ordinary direct helper
  calls always pass/retain false and therefore use `create_if_absent=False`.
- `maybe_generate_webui_title_after_turn()` is the only caller allowed to set that
  adapter true. It does so only when the inbound metadata has `webui=True`, the
  origin `websocket:<chat_id>` differs from the routed/unified key, and a
  noncreating `read_session_metadata(routed_key)` proves the accepted turn's routed
  session is already durable. After a provider returns a usable title, the helper
  performs one `upsert_session_metadata(origin_key,
  {WEBUI_TITLE_METADATA_KEY: value},
  create_if_absent=True)`. This preserves accepted unified-session per-chat title
  projection without treating a draft or temporary chat as persistent.
- No target is created when the routed durable proof is absent, a title already
  exists, the input has no useful user text, provider generation fails, or the
  returned title is unusable. Pure title reads remain noncreating. There is no new
  memory, channel, event, or autonomous persistence policy.
- `create_webui_chat_fork()` persists an optional requested title through the same
  manager seam after the journaled fork. Fork rollback on later transcript/title
  failure retains existing delete semantics, now protected by resolver locking.
- `WebUIWorkspaceResolver.persist_scope()` uses one manager metadata upsert for
  `webui=True` plus `workspace_scope`. It sets `create_if_absent=True` only after an
  accepted persistent message; draft/new-chat staging remains memory-only and
  temporary chats remain nonpersistent. It does not create/get/mutate/save a
  cached session locally.
- In-turn workspace-scope capture continues through the existing turn/source
  writer. Package II does not add a second persistence path.
- Title diagnostics are fixed and content-free. The failure and unusable-output
  branches log only stable messages/enums; they do not log a session key, chat ID,
  exception/traceback, provider `finish_reason`, generated content, or path.

## 5. Work graph, ownership, and correction limits

The user explicitly requested all subagents use GPT-6.1 Sol, never GPT-5.6 Sol;
use native explicit GPT-6.1 Sol routing at High for integrity core, Medium for
consumers/wiring, and read-only Medium for fresh review, preserving the contracts
and correction limits below.

Root authorized the bounded synchronous resolver guard reuse seam after read-only
concurrency analysis: temporary thread-local scopes bind exact PID, thread, current
asyncio task, and immutable sorted held keys; only same-context subset reuse is
legal, and fresh/expanding execution acquisition under a files lock fails closed.
Bootstrap/recovery discover read-only lock hints and revalidate them under sorted
execution guards then the files lock. Existing exporter/artifact callers and
`execution_owner.py` remain frozen. Root also authorized only replacing the existing
conversion checkpoint-admission fixture's malformed sidecar with a real version-2
checkpoint, preserving its rejection assertion.

No recursive delegation. At most two disjoint write workers may run, but the
critical path below is sequential because Package II consumes Package I APIs and
default restoration waits for both. Every new package receives at most **two**
bundled correction rounds; do not revive exhausted A or E.

### I — integrity core (High)

High is required because exact persistent evidence, two-key concurrency, crash
recovery, and the accepted P4 owner boundary meet at one irreversible commit.

May edit only:

- `nanobot/session/session_mode.py`: paired inventory, exact journal fields/parser,
  bootstrap/list/recovery, and resolver reset/delete/fork transaction bodies;
- `nanobot/session/manager.py`: managed pure overlay routing, reset/metadata/fork
  seams and cache publication;
- `nanobot/session/source_store.py`: only the private already-locked reset apply
  factor and validated metadata removals needed by the manager seam;
- `tests/session/test_session_mode_registry.py`;
- `tests/session/test_session_mode_conversion.py` only for shared journal-parser
  regression;
- `tests/session/test_session_mode_operations.py`; and
- `tests/migration/test_p5_stage2_crash_matrix.py`.

It must not edit `execution_owner.py`; the existing `execution_transaction` and
`execution_lock_guard` are sufficient. If repository evidence disproves that, stop
and report the exact contradiction rather than add an owner abstraction or widen
ownership. It must not edit SDK, WebUI, composition, CLI, loop/command wiring,
exporter, artifact operator, guards, or unrelated tests.

Red-green order:

1. Orphan/canonical sidecar, every canonical producer's absence check, and strict
   valid/invalid sidecar inventory tests.
2. Bootstrap master/per-record exact sidecar proof and resume/reload faults.
3. Reset faults for every source/sidecar/record phase, including stale legacy
   sidecar and unchanged-input-only recovery.
4. Delete owner contention and every tombstone/cleanup reload phase.
5. Fork source-owner contention, reciprocal lock-order test, exact source
   record/data/sidecar journal proof, source-change rejection, and every stage/
   install/record reload phase.
6. Create/convert/fork/export faults immediately before and after `prepared` prove
   no unbound rollback/stage residue exists.
7. Manager absent clear, active out-of-turn canonical clear, busy/recovery rejection,
   current-owner reset, metadata upsert, cache publication, and direct-resolver
   lock-bypass probes.

Ready gate: every managed lifecycle operation is execution-before-files even when
called directly; all prepared rows contain exact required evidence or verified
absence; every fault reloads through a fresh resolver/manager and reaches exactly
one specified verdict; no changed-input transform is retried.

### II — SDK and WebUI consumers (Medium, after I)

May edit only:

- `nanobot/sdk/clients.py`;
- `nanobot/session/webui_turns.py`;
- `nanobot/webui/forking.py`;
- `nanobot/webui/workspaces.py`;
- new `tests/session/test_p5_stage2_consumers.py` as the isolated Stage 2 consumer
  acceptance module;
- only the existing WebUI-title cases in `tests/agent/test_loop_save_turn.py`;
- `tests/webui/test_forking.py`;
- `tests/utils/test_webui_workspaces.py`.

It consumes only Package I manager APIs. It must not edit manager/resolver/source,
loop/command in-turn reset wiring, composition/defaults, CLI, exporter, artifact
operator, guards, protocol behavior, historical facade/constructor fixtures, or
unrelated tests. In particular, it does not edit `tests/test_nanobot_facade.py`,
`tests/test_nanobot_facade_fork.py`, or websocket settings/constructor expectations
to make the broad baseline greener.

Red-green coverage includes detached `save=False` cache/authority byte stability;
early saved-managed ingest/restore rejection for absent, legacy, and canonical
keys; unmanaged compatibility; managed SDK clear positive absent/active cases plus
busy and recovery rejection; generated and fork-title canonical persistence; and
workspace-scope draft, first accepted message, update, and temporary-chat cases.
Title coverage includes an absent per-chat target after a durable unified turn
creating exactly one metadata-only target, and negative cases proving no target on
draft/temporary, failed provider, unusable title, existing title, or missing routed
durability proof. It also asserts title failure diagnostics contain none of the
key, exception text, finish reason, or generated content.
The new consumer module constructs real managed/unmanaged managers and
`SessionClient`/WebUI helpers through accepted injection seams with a fake
loop/transport/runtime. It must not patch or bypass migration guards and must not
call historical `Nanobot.from_config()`/`AgentLoop` constructors that are
intentionally denied at the current migration boundary.

The owned existing title cases in `tests/agent/test_loop_save_turn.py` may replace
their `_make_full_loop` setup only with narrow direct fixtures containing a real
`SessionManager`, fake provider, model, and the title helper. Do not globally patch
constructors or migration guards, and do not alter unrelated loop expectations.

Package II's deterministic ready gate is:

1. all cases in `tests/session/test_p5_stage2_consumers.py` pass;
2. all of `tests/webui/test_forking.py` and
   `tests/utils/test_webui_workspaces.py` pass; and
3. the existing title cases selected from `tests/agent/test_loop_save_turn.py` pass.

The broad six-module compatibility run is diagnostic comparison, not the ready
gate. Before I edits it produced **114 failed, 261 passed**: most failures are
unchanged `MigrationIncompleteError` constructor denials; the set also includes a
removed `_make_provider` expectation and one unrelated websocket settings
8192-versus-4096 mismatch. Rerun the identical command after II, compare failed node
IDs with root's saved baseline, and report the complete classified result. No new
unrelated failure is allowed, but the 114 baseline failures are neither promoted to
green nor repaired by weakening guards or rewriting historical fixtures.

### III — guarded definition-only default restoration (Medium, after I and II ready acceptance)

May edit only `nanobot/runtime_composition.py`, narrowly necessary default/captured-
preflight assertions in `tests/migration/test_p5_stage1_composition.py`, and
`tests/migration/test_p5_stage2_composition.py`. This package may be folded into
the Package II worker only after II is accepted and with an explicit sequential
ownership transfer; it is never concurrent consumer work.

`RuntimeComposition.define()` remains synchronous and read-only. When no authority
is explicitly supplied, it constructs `DurableSessionAuthority` from the already
captured `report.workspace` and `report.sessions_root`; it does not reread config or
touch the namespace. `session_authority is not None` remains the selection rule.
Tests that require the historical Stage 1 cutoff inject
`Stage1UnavailableSessionAuthority` explicitly. Guard-negative tests and guard
order are not edited. Acquisition remains the first constructor/mutation boundary,
and authority close remains after agent graph close.

The sequencing is exact: root first accepts I and II as ready dependencies while
the actual default is still `Stage1UnavailableSessionAuthority`; III then restores
the definition-only durable default; only then do the complete Section 10 fan-in,
fresh integrated review, and root AC1-AC12 verification run against that default.
If any of those final gates fails, restore the Stage 1 cutoff as the bounded rollback
before further correction. III does not imply Stage 2 PASS by itself.

## 6. Acceptance mapping and verification

| Existing criterion | Required new evidence |
| --- | --- |
| AC1 | III proves definition byte/inode/mtime stability, captured preflight, explicit falsey injection, first-write canonical behavior, and unchanged guard order. |
| AC2 | I proves paired bootstrap inventory, exact sidecar journal evidence, orphan/canonical-sidecar rejection, pure managed sidecar validation, and read-only complete listing. |
| AC3 | I owner-contention tests prove reset/delete/fork are nonblocking busy and direct resolver calls cannot bypass the persistent execution lock. Old-binary exclusion is still not claimed. |
| AC4 | I/II prove mixed modes keep strict routing and provider-state behavior; no new owner/source store is introduced. |
| AC5 | I table-drives every reset/delete/fork phase and reload, exact predecessor/target/sidecar verdict, idempotent completion, reciprocal fork ordering, and no transform retry on changed input. |
| AC6 | I manager seams and II SDK/WebUI tests prove clear, fork title, generated title, workspace scope, reads, cache invalidation, and unmanaged compatibility use the resolver/metadata authority. |
| AC7 | Re-run frozen C-R export and crash tests; no exporter file changes. |
| AC8-AC9 | Re-run frozen artifact operator/rematerialization/factory/store tests; no D file changes. |
| AC10 | III and the Stage 1 matrix prove TTL/idle preflight and dormant background/rematerialization boundaries remain closed. |
| AC11 | I malformed sidecar/journal and II consumer-error tests assert fixed content-free messages and no key/path/digest/content labels. |
| AC12 | Run the exact Section 10 selections and report broad Ruff baseline, BasedPyright, native CLI dependency, and Windows qualifications honestly. |

Package commands, all offline and network-denied:

```bash
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/session/test_session_mode_registry.py \
  tests/session/test_session_mode_conversion.py \
  tests/session/test_session_mode_operations.py \
  tests/migration/test_p5_stage2_crash_matrix.py --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/session/test_p5_stage2_consumers.py \
  tests/webui/test_forking.py tests/utils/test_webui_workspaces.py --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/agent/test_loop_save_turn.py \
  -k 'webui_title or projected_title' --tb=short

# Required diagnostic comparison only; expected baseline is 114 failed, 261 passed.
# Compare failed node IDs with root's saved pre-I output; do not treat this as green.
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/test_nanobot_facade.py tests/test_nanobot_facade_fork.py \
  tests/agent/test_loop_save_turn.py tests/webui/test_forking.py \
  tests/utils/test_webui_workspaces.py \
  nanobot/channels/websocket/tests/test_websocket_channel.py --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/migration/test_p5_stage1_composition.py \
  tests/migration/test_p5_stage2_composition.py \
  tests/agent/test_p5_stage2_engine.py \
  tests/cli/test_p5_stage2_sessions.py --tb=short
```

Each package also runs scoped Ruff on only its owned production/tests, Python 3.13
`compileall` on owned production files, and `git diff --check`. Do not run
`ruff format`.

After I and II ready acceptance and the III guarded-default change, rerun
**unchanged** all commands from implementation-contract Section
10: the ten-module Stage 2 selection, all `tests/session`, the five-module Stage 1
startup matrix, the broad scoped Ruff selection, compileall, and `git diff --check`.
Also rerun the frozen C-R focused/export-crash selection and D artifact selection
from the repair contract. The previously reported 69 broad Ruff findings in eight
unrelated test files are baseline, not green and not in package ownership.
BasedPyright and native Windows remain unqualified. The unpatched native CLI's
missing `setproctitle` remains an environmental limitation; tests may keep their
existing title-setter patch but may not claim native CLI qualification.

After fan-in, one fresh `development_analysis_medium` reviewer receives the
implementation contract, repair contract/checkpoint, this contract, raw diff, all
package/fan-in results, and fault-phase matrix. It returns PASS/CORRECT/REJECT for
the integrated change. Root then independently reads the diff, reruns critical
selections, maps AC1-AC12 to evidence, and alone may accept Stage 2.

## 7. Failure, rollback, observability, and remaining gates

Any package contradiction or need to edit outside its ownership is a stop condition,
not permission to improvise. A failed I or II remains behind the actual Stage 1
default. A failed III restores `Stage1UnavailableSessionAuthority` as one narrow
composition change and reruns the safety-cutoff selection; it never leaves a
partially routed default.

Before a mode-record commit, recovery uses only exact journaled predecessor bytes.
After canonical or tombstone commit, it completes forward. A fork never changes its
source. Changed source, target, sidecar, journal, or stage evidence quarantines/fails
closed; no provider, tool, delivery, reset/fork transform, or publication is
repeated. Lock files are persistent and never unlinked.

Routine logs and metrics expose only fixed operation/mode/phase/result enums. They
must not contain session keys, paths, IDs, digests, prompts, events, sidecar values,
references, receipts, exception text, or credentials. Pure validation returns
fixed content-free failures.

No offline implementation gate remains. These later gates remain outside Stage 2:
approval of any live bootstrap/conversion batch; retention/removal of rollback or
artifact material; automatic retention/rematerialization/idle policy; D9 guard
removal; dependency installation; P6 packaging/native/Windows qualification; P7
canary/promotion; deployment; and all cron/background activation. A Stage 2 PASS
authorizes none of them.

## 8. Accepted session-compatibility completion addendum (2026-10-01)

The user approved one bounded completion round after the consumer compatibility
diagnostic introduced three websocket regressions. Stage 2 is still not PASS.
This addendum preserves the completed integrity and consumer work; it does not
reopen the migration design or erase either historical rejected checkpoint.

### Evidence and accepted decisions

- Root reproduced unchanged `test_temporary_chat_is_transient_and_discarded`,
  `test_disconnect_discards_temporary_chat`, and
  `test_temporary_looking_id_does_not_define_session_policy`: **3 failed**.
  Existing managed engine selection remains **8 passed**.
- The accepted-message handler unconditionally persists workspace scope even
  when server-owned temporary policy exists. The new metadata upsert thereby
  creates durable state and invalidates the temporary cache.
- Unmanaged metadata upsert invalidates ordinary cached sessions instead of
  publishing committed metadata as the existing update seam does.
- Managed transient construction is intentionally detached, but WebUI temporary
  policy and engine required-session admission use the manager cache. Engine
  execution admission additionally selects persistence before restoration reads
  SessionPolicy. A channel-only fix therefore cannot preserve managed temporary
  behavior.
- Read-only GPT-6.1 Sol High analysis established the minimum seam: explicitly
  retain a WebUI temporary Session in the existing manager cache, validate its
  resolver binding, select that exact identity under the existing per-key
  admission lock before source admission, and hand it to restoration. Existing
  policy-driven history, memory/tool restrictions, content hiding, weak overflow
  cache, cancellation and late-event tombstones are reused unchanged.
- Regular/new sessions and temporary sessions remain supported. IDs never define
  persistence policy. Explicit SDK ephemeral execution and default managed
  transient construction remain detached even at an existing persistent or live
  temporary key. No material product gate remains for this bounded repair.

### Compatibility write package and ownership

Use one GPT-6.1 Sol High worker, isolated context, no recursive delegation.
High is warranted by the unusually subtle temporary privacy/admission handoff,
where channel acceptance and engine execution must agree before durable effects.
One writer owns this entire cross-boundary compatibility seam; no parallel edits.
The permitted production changes are exclusively:

1. `nanobot/session/manager.py`: explicit optional transient retention using the
   existing cache/binding machinery; validate authority and reject persistent
   key/cache-policy collisions before retention; retain detached managed default;
   publish committed unmanaged upsert metadata/removals and `_base_metadata`
   without dropping cached local history or publishing before commit. Managed
   durable upsert invalidation remains unchanged. No resolver/source changes.
2. `nanobot/webui/temporary_chats.py`: explicitly request temporary retention at
   creation. Preserve ownership, restrictions, disposal, media and tombstones.
3. `nanobot/webui/inbound_commands.py`: persist post-dispatch scope only when
   `temporary_policy is None`. No new protocol or commands.
4. `nanobot/agent/loop.py`: under `_admit_after_bus_cleanup`, recheck required
   session existence, select the live nonpersistent cached Session unless explicit
   SDK ephemeral takes precedence, and compute effective ephemeral before
   `_source_execution`. Pass the exact selected Session into `_process_turn` and
   its initial TurnContext. Keep existing source validation, serialization,
   persistent admission, owner, recovery and policy behavior. No runner changes.

Tests may add cases only in `tests/session/test_p5_stage2_consumers.py`,
`tests/agent/test_p5_stage2_engine.py` using its existing admitted synthetic
engine fixture without modifying guard patches, and a new isolated
`tests/session/test_p5_stage2_temporary_sessions.py` for real managed/unmanaged
manager and channel lifecycle coverage. The existing three websocket regressions
and all historical facade/constructor/settings expectations remain unchanged.
No test weakening, global guard patch, fixture rewrite, broad cleanup, or
`ruff format`. Preserve all unrelated dirty changes.

The unmanaged upsert publication is the final available correction of Package I
metadata cache behavior (I already used one correction); this addendum does not
restart its core-integrity budget. Caller policy repair is Package II's first
correction. Newly authorized temporary compatibility has at most two bundled
correction rounds. Exhaustion or required writes beyond this seam is a stop,
not authorization to expand or revive exhausted packages A/E.

### Checkable compatibility acceptance

- C1: All three unchanged websocket regressions pass, and an ordinary temporary-
  looking ID remains persistent. Unmanaged upsert preserves cached identity,
  local history, committed metadata/removals/base, and failure-before-commit state.
- C2: Real managed temporary creation and two engine turns preserve one live
  Session identity and replay first-turn history; no temporary JSONL, sidecar,
  registry record/journal, transcript, title/workspace metadata, or run record is
  created. Memory exclusion, disabled tools, content hiding and in-memory model
  selection remain intact.
- C3: Strongly retained active temporary Session survives bounded cache eviction;
  unauthorized owners, prohibited commands, workspace updates and recovery attach
  remain rejected. Discard/disconnect clean media/cache, cancel through existing
  runtime control, reject late/queued required-session admission, and prevent late
  stream/completion transcript persistence. Include a deterministic disposal race.
- C4: Existing explicit SDK ephemeral absent/canonical cases stay unchanged and
  green; explicit ephemeral at a live temporary key is detached and does not alter
  its history. Default managed transient creation remains detached.
- C5: Invalid/conflicting managed evidence and persistent/cache-policy collisions
  reject activation/admission before approval consumption, model/tool calls or
  durable effects, with content-free errors. Existing resolver checks are not
  bypassed by cached policy.
- C6: Existing Package I, consumer/title, frozen export/artifact gates pass. The
  identical six-module diagnostic has no added failed node relative to the saved
  114-failure baseline; unchanged constructor denials and unrelated settings remain
  diagnostic, not green evidence. Scoped lint, compileall and diff checks pass.

Red-green order: reproduce the three unchanged failures, add the managed lifecycle
and admission regressions before production edits, then implement the minimum seam
and run the complete compatibility selection. Worker returns exact changed files,
red/green command outcomes, C1-C6 evidence and any limitation. Commands use
`NANOBOT_DISABLE_NETWORK=1` and the pinned Python/Ruff paths in Section 6.
Root independently inspects the pre-repair delta and runs:

```bash
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/session/test_p5_stage2_temporary_sessions.py \
  tests/session/test_p5_stage2_consumers.py tests/agent/test_p5_stage2_engine.py \
  tests/webui/test_forking.py tests/utils/test_webui_workspaces.py --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  nanobot/channels/websocket/tests/test_websocket_channel.py::test_temporary_chat_is_transient_and_discarded \
  nanobot/channels/websocket/tests/test_websocket_channel.py::test_disconnect_discards_temporary_chat \
  nanobot/channels/websocket/tests/test_websocket_channel.py::test_temporary_looking_id_does_not_define_session_policy --tb=short
```

After compatibility ready acceptance, Package III is dispatched under its exact
Section 5 ownership; no consumer work runs concurrently. Complete unchanged
Section 10 fan-in, frozen dependencies, fresh GPT-6.1 Sol Medium integrated review,
and root AC1-AC12 verdict then finish Stage 2. Any final rejection restores the
Stage 1 cutoff before further corrections. No Stage 3 begins in this round.

Non-goals remain new mode-switch commands, per-request context builder or memory
redesign, temporary approval-map behavior changes (already process-local), new
session authorities, guards, live state/config/credentials, commits, dependency
installation, deployment, background/scheduler activation, and Windows qualification.

### Root compatibility dependency acceptance

Root accepted C1-C6 as **READY PASS**, not Stage 2 PASS, with the actual Stage 1
cutoff still installed. Independent reruns: compatibility 93 passed in 7.04s;
unchanged websocket regressions 3 passed in 1.45s; integrity 236 passed in 18.38s;
frozen export/crash 27 passed (126 deselected) in 4.19s; frozen artifact 75 passed
in 5.78s; titles 8 passed (70 deselected) in 0.20s. The exact diagnostic is
108 failed, 267 passed in 13.50s: no added failed nodes against the original 114,
and the six owned title failures removed. This diagnostic is not green evidence.
Root scoped Ruff, compileall and diff checks passed; pre-repair full file hashes
show only the seven owned compatibility files plus this contract changed.
Package III may now begin with the original Section 5 ownership and rollback gate.
