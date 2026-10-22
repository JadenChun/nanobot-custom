# P5 Stage 2 repair contract

Status as of 2026-10-01: **guarded offline Stage 2 PASS** in the
[acceptance checkpoint](p5-stage2-acceptance-checkpoint.md). Production remains
inactive. The [integrity completion contract](p5-stage2-integrity-contract.md)
supplies the accepted superseding decisions; the following prior repair evidence
is retained as history. This document
partitions the accepted [Stage 2 specification](p5-stage2-implementation-contract.md) after the
rejected [first attempt](p5-stage2-checkpoint.md). The historical
[repair checkpoint](p5-stage2-repair-checkpoint.md) records the accepted dependency packages,
remaining AC2/AC5/AC6 findings, material unapproved canonical SDK-clear choice, qualified
validation evidence, and accepted safety-cutoff restoration. It does not replace Sections 1-7
or AC1-AC12 of that specification. [P4 closure](p4-closure-checkpoint.md), the
[Stage 1 checkpoint](p5-stage1-checkpoint.md), and P5 D1-D8 in the
[activation plan](p5-integration-activation-plan.md) remain authoritative. At that repair stop,
the default was restored to `Stage1UnavailableSessionAuthority`; durable authority stayed
explicit-only until the final integrity/compatibility acceptance. Guards, live state,
configuration, credentials, dependencies,
background work, activation, and Windows qualification are out of scope.

## 1. Accepted decisions and gates

The repair retains one durable mode authority, canonical-on-first-persistent-write, explicit
stopped-writer bootstrap/conversion/export, transition-only digests, permanent canonical
authority after commit, P4 source/execution/artifact ownership, and copied-state-only artifact
operators. No startup path adopts or migrates nonempty unregistered state.

Rejected alternatives remain those in the accepted specification. In particular, do not infer
mode from headers/files/mtime, dual-write, use `upgrade_source_session()` in a managed namespace,
construct a second source store, acquire an owner on the copied artifact input, or make
rematerialization automatic.

**Fork rule accepted by the user on 2026-09-30:** `ArtifactDescriptor` and
`bound_artifact_text()` bind a descriptor to source `session_key`, `incarnation`, and
`generation`, so a canonical fork does not copy or rebind descriptors or blobs. The target
preserves the full canonical event prefix, including exact event/execution/stream linkages and
inline tool-result evidence, but drops `artifact_manifest` and `context_checkpoint`. Original
source blobs and metadata remain untouched. Package B implements and tests exactly this rule.

Reference pattern only: official [app-server documentation](https://learn.chatgpt.com/docs/app-server)
says `thread/fork` copies stored history to a new thread ID; it establishes no nanobot artifact semantics.
DeepSeek's pinned [`buildForkSeed`](https://github.com/deepseek-ai/deepseek-harness/blob/477b4f420553e8a52c2fbccc464d7561b239c443/packages/core/session/src/fork.ts)
copies the exact event prefix, marks the inherited cut/open tail, and preserves the original log and valid event-based compaction surfaces in `fork.spec.ts`. Dropping nanobot's identity-bound header manifest/checkpoint is the approved adaptation, not an exact clone; add no synthetic closure policy.
No material product or operational gate remains before offline implementation dispatch. Later
live-operation, retention, activation, deployment, and platform gates remain out of scope as
listed in the accepted specification; they do not block package A.

Prerequisite evidence, not Stage 2 acceptance: explicit authority selection now uses `is not None`, with 80 Stage 1 startup/composition tests passing; the existing session/provider baseline is 905 passing. Stage 2 still requires this contract's complete fan-in, validation, independent review, and root PASS.

## 2. Frozen shared interfaces and invariants

Package A owns these interfaces; later packages consume them without private journal calls.

1. Add an opaque, process-local `ManagedNamespaceAccess`, issued only after the lifetime writer
   lease is held. `SessionManager`/`JsonlSessionStore` receive it through a private construction
   argument; the backend retains it for accepted P4 source writes. It is nonserializable,
   process-bound, namespace-inode-bound, and revoked before lease release.
2. Before `JsonlSessionStore.__init__` can create/claim/migrate anything, authority performs a
   pinned, no-follow read of workspace identity, target namespace, both registry witnesses, and
   legacy `<workspace>/sessions`. A runtime rejects any nonempty unregistered namespace or
   legacy workspace directory. Managed runtime/operator construction disables
   `_migrate_from_workspace`; bootstrap adopts only files already in the explicit copied session
   namespace.
3. Every default-store mutation rechecks registry claim while its storage lock is held. Active,
   preparing, partial, or residual `.session-mode` evidence rejects unless the backend has the
   matching live capability. This covers save/checkpoint/repair/metadata/delete/source writes and
   managers constructed before registration. A new direct/custom `SessionManager` rejects a
   claimed namespace before constructor mutation. Managed `restore_to_workspace()` and
   `upgrade_source_session()` always reject even with the capability; their unmanaged behavior is
   unchanged.
4. Implement one descriptor-relative storage primitive (by extending the accepted
   `SourceRootAuthority` pattern or a smaller sibling) for session-mode directories, witnesses,
   records, journals, staging, rollback, source installation, and locks. It pins exact directory
   identities and performs checked `open/stat/mkdir/replace/unlink/fsync` relative to descriptors.
   Files are same-UID regular single-link `0600`; directories are same-UID no-follow `0700`;
   locks are never unlinked. Unsupported POSIX primitives fail closed. JSON is capped at the
   existing 1 MiB limit; enumeration and operation-record counts use named hard limits and fail
   content-free on overflow.
5. `SessionModeResolver.resolve()` is read-only. Its stable binding is
   `(registry_id, record_revision, mode, incarnation, generation)`. Manager cache entries carry
   that binding and every managed public entrypoint resolves before returning a cached object or
   touching storage. Binding change invalidates the entry. Ordinary legacy/source appends,
   checkpoint writes, metadata writes, and execution leases do not change the mode record or
   transition digest.
6. Resolver exposes high-level `bootstrap`, `ensure_canonical`, `convert`, `recover`, `reset`,
   `delete`, `fork`, `list`, and an export transaction API. Only it writes records/journals. C gets
   a public `SessionExportTransaction` with an operation-owned stage, exact destination binding,
   `mark_published()`, `complete()`, and pre-publication `abort()`; D needs only resolved binding
   and exact locked snapshot APIs. `JsonlSessionStore.parse_legacy_bytes()` is the strict,
   side-effect-free compatibility reader used by C.
7. Add `SessionManager.reset_session(...)` as the sole `/new` persistence seam. Legacy reset uses
   `Session.clear()` plus the existing conflict-safe save. Canonical reset requires the current
   `ExecutionOwner` and calls the existing `reset_source_session`; factor its deterministic
   transform so a frozen `SourceResetPlan` (including timestamp and exact target bytes/state) can
   be journaled before installation. Do not bypass P4 generation advance, metadata clearing, or
   owner invalidation. `agent/loop.py` and `command/builtin.py` delegate to this manager API.
8. Lock order is: copy/runtime lifetime lease; artifact transaction when applicable; per-session
   execution lock; `.session-files.lock`. Bootstrap additionally takes the existing workspace
   migration lock before `.session-files.lock`. No code may enter these in reverse order.

Copied-state operators preserve the persisted logical `workspace_id` and rebind only the copy's
`.workspace` path claim under the copy lease; they never invoke normal copied-workspace ID
rotation. Runtime keeps the existing move/copy isolation behavior. Scratch rematerialization uses
the same logical ID in a private, unregistered, disposable namespace; it is never a runtime or
turn authority.

## 3. Strict record and crash protocol

Use the exact schemas and transition-digest rule in specification Section 3. Parsers require exact
field sets, canonical UTF-8 JSON, duplicate-key rejection, real integers (not booleans), aware
timestamps, valid storage-key round trip, operation-owned relative names, and exact legal phase
sequences. Unknown entries, gaps, duplicate phases, bad digest chains, unsafe residue, or bytes
matching neither predecessor nor target quarantine the operation.

Legal sequences are: bootstrap/create/convert/reset/fork
`prepared -> target_installed -> mode_committed -> complete` (or `aborted` before record commit);
delete `prepared -> mode_committed -> complete`; export
`prepared -> published -> complete` (or `aborted` before publication). Record publication is the
authority commit; a crash after the exact target record replace but before its `mode_committed`
journal row completes forward. Every `prepared` row binds exact prior record bytes, prior active
and sidecar bytes (or absence), planned target record bytes, target data digest/size, and immutable
staging/rollback names before the first authoritative replace.

Paired-witness bootstrap is resumable, not heuristic:

1. Under the lease and locks, strictly enumerate/parse one immutable input inventory. Create the
   master bootstrap `prepared` journal before either witness.
2. Publish byte-identical `preparing` registry then workspace witness. For each sorted session,
   create a per-record bootstrap journal, publish the planned record, then commit/complete that
   journal. A retry may create only missing byte-identical facts.
3. After all input bytes and records revalidate, append master `target_installed`, replace both
   witnesses with the same planned `active` bytes, then append master `mode_committed` and
   `complete`. A one-witness or active/preparing mismatch blocks runtime but the same operator can
   resume from the master journal.
4. Empty first-write registration uses the same master protocol with an empty inventory, then a
   separate create journal. Reads never start it.

Operation recovery is exact:

- create/fork before record commit removes or quarantines only the verified target; an exact
  committed record completes forward;
- conversion before record commit restores the verified immutable legacy rollback snapshot; an
  exact canonical record completes forward and never rolls back;
- reset with predecessor bytes aborts, with exact planned output commits the new binding, and with
  neither quarantines; canonical output is produced only by the frozen P4 reset plan;
- delete becomes logically deleted at tombstone replace and recovery only verifies/finishes exact
  source/sidecar cleanup;
- export before publication may remove exact operation staging and abort; an exact installed
  nonexistent-destination bundle completes `published/complete`, while any preexisting or changed
  destination fails closed; and
- recovery never repeats a transform, provider/tool/delivery effect, or artifact publication on
  changed input. Runtime resolution reports recovery-required; only the offline operator mutates
  incomplete evidence.

## 4. Work graph and ownership

All workers preserve unrelated dirty work, use test-first changes, and stop on contract/repository
conflict. No worker changes guards/defaults before E, installs dependencies, uses live roots, or
receives more than two substantial packages. At most two disjoint write workers run concurrently.

### A — storage, registry, transition core (High; first package of shared owner)

Own: `session_mode.py`, `runtime_authority.py`, `manager.py`, `source_store.py`,
`source_root.py`, narrowly necessary `execution_owner.py`, new mode-storage modules, and
`test_session_mode_registry.py`, `test_session_mode_conversion.py`,
`test_p5_stage2_crash_matrix.py`. Implement Sections 2-3: lease/copy identity, constructor and raw
writer fence, strict grammar, paired bootstrap, create/convert/recovery, frozen reset plan, and
public later-package APIs. Do not integrate consumers or default composition.

Ready acceptance: unregistered nonempty state is rejected before constructor migration; paired
witness resume and every create/convert/reset crash window reload from a fresh manager; raw,
custom, upgrade, and restore bypass probes fail in a claimed namespace while unmanaged cases pass;
transition digests tolerate later same-identity revisions.

### B — manager routing and real engine (High; second/final package of shared owner)

Depends on A and retains A's shared-file ownership. May additionally edit `agent/loop.py`,
`command/builtin.py`, `runtime_composition.py`, `test_session_mode_operations.py`,
`test_p5_stage2_composition.py`, `test_p5_stage2_engine.py`, and focused provider-state tests.
Implement binding-aware cache/read/list, canonical/legacy reset, tombstone delete, metadata/model
selection and prevalidated best-effort rename, fork under the user-approved artifact rule, and
actual engine/admission integration. Keep composition default unavailable.

Ready acceptance: all managed SDK/WebUI/manager paths resolve first; first persistent engine write
is canonical before approvals/provider/tools, ephemeral work is absent; mixed modes preserve
legacy provider state and canonical no-state/P4 owner behavior; reset goes through the manager and
P4 method; delete/list/cache/reload crash cases match Section 3.

### C — export and legacy round-trip proof (Medium; independent after A API freeze)

Own only `session_export.py`, optional new export-local modules, and
`test_session_mode_export.py`. Use `SessionExportTransaction`; do not edit shared authority/store
files. Compatibility output must parse through `parse_legacy_bytes()` and compare normalized
public transcript, tool pairing, portable durable metadata, archive boundary, and inline artifact
evidence. Unsupported source-only evidence aborts byte-identically. Test crashes before/after
stage install, destination publication, journal publication, and completion.

### D — copied artifact operator (Medium; parallel with C after A API freeze)

Own new `artifact_operator.py`, narrowly necessary `artifact_store.py`,
`artifact_rematerialization.py`/`artifact_factory.py` API extensions, and
`test_artifact_operator.py`. Inspect is aggregate/content-free. For rematerialization: lock and
snapshot input; acquire destination artifact transaction; build exact private scratch source;
acquire the real `ExecutionOwner` only on scratch; create the existing configured reader and call
`ArtifactRematerializer` with the already-held blob transaction; close owner; verify destination;
re-read and byte-compare copied source and mode record; remove scratch. Failure publishes no source
reference and never mutates input. Disabled artifacts create nothing.

### E — thin operators, default, and fan-in wiring (Medium; only after A-D)

Own `cli/commands.py`, final `runtime_composition.py` edits, necessary session exports,
`test_p5_stage2_sessions.py`, and final composition/engine test wiring. Add thin copied-root
bootstrap/convert/recover/export/artifact commands with explicit offline qualification and live-root
identity rejection. Switch `define()` from the Stage 1 cutoff to definition-only
`DurableSessionAuthority` using captured preflight paths; explicit injected authority still wins.
Do not edit guard order. Acquisition remains the first constructor/mutation boundary and releases
the session authority last after the graph closes.

## 5. Verification, review, rollout, and rollback

Each package runs its owned tests plus scoped Ruff, Python 3.13 `compileall`, and
`git diff --check`. Final fan-in must run the exact three pytest selections, Ruff selection, and
compileall command from specification Section 10. Focused package commands are:

```bash
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q tests/session/test_session_mode_registry.py tests/session/test_session_mode_conversion.py tests/migration/test_p5_stage2_crash_matrix.py --tb=short  # A
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q tests/session/test_session_mode_operations.py tests/migration/test_p5_stage2_composition.py tests/agent/test_p5_stage2_engine.py tests/providers/test_conversation_state.py --tb=short  # B
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q tests/session/test_session_mode_export.py --tb=short  # C
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q tests/session/test_artifact_operator.py tests/session/test_artifact_rematerialization.py tests/session/test_artifact_factory.py tests/session/test_artifact_store.py --tb=short  # D
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q tests/cli/test_p5_stage2_sessions.py tests/migration/test_p5_stage2_composition.py tests/agent/test_p5_stage2_engine.py --tb=short  # E
```

The nine required Stage 2 modules are the five session modules (the four
`test_session_mode_{registry,conversion,export,operations}.py` modules and
`test_artifact_operator.py`), two migration modules, the agent engine module, and the CLI sessions
module; `tests/providers/test_conversation_state.py` remains required compatibility evidence.
No missing/dependency-blocked/Windows/BasedPyright result is promoted to green.

After complete fan-in, one fresh `development_analysis_medium` reviewer receives this contract,
the accepted specification, raw diff, and test results and returns PASS/CORRECT/REJECT. Each
package gets at most two bundled correction rounds by its original owner. Root independently reads
the diff, reruns critical selections, verifies AC1-AC12, and alone issues Stage 2 PASS.

All intermediate defaults fail closed; no partial authority is promoted. Rollback before mode
commit uses only verified predecessor evidence; after canonical commit it is verified export into
separate copied state, never a mode edit. If fan-in fails, restore the Stage 1 default as one
composition change and retain no partially routed default. Routine logs/metrics expose only fixed
operation/mode/phase/result enums—never keys, paths, IDs, digests, content, references, receipts,
or credentials. No live bootstrap/conversion, retention, D9, deployment, background activation,
or platform qualification is authorized.

## 6. First dispatch

Package A has been dispatched to one `development_worker_high` with sole ownership of the shared
manager/source boundary. The stronger worker is required because constructor-before-migration fencing and the
record/journal commit windows cross persistent authority and existing P4 lock ownership. Its
completion report must map tests to the A acceptance bullets, enumerate crash points and exact
commands, and report any repository contradiction instead of weakening the contract.

## 7. Post-A export-boundary repair addendum

This addendum narrows sequencing and ownership after package A completed. It does not reopen or
replace Sections 1-6 above, specification Sections 1-7, or AC1-AC12. A's owner reported 77 owned
tests and 939 session/provider tests passing, with scoped Ruff and compileall green; root
independently reran the 77 owned tests successfully. That evidence freezes A as a dependency. It
is not Stage 2 PASS, and A's two correction rounds remain exhausted.

### 7.1 Evidence and decision

The retained `session_export.py` is the rejected first-attempt prototype, not accepted substrate.
It calls resolver-private `_append_journal`, imports the export-only `_atomic_write` and
`_fsync_dir` helpers, reads live files outside `locked_snapshot()`, publishes with replacing
`os.replace`, and never proves compatibility output by parsing it through
`parse_legacy_bytes()`. Its one export test therefore establishes neither AC7 nor the crash
contract. A broad green suite containing that prototype cannot be used as export acceptance.

The current shared parser also accepts two export journal bindings: the accepted destination/file
plan produced by `begin_export(key, destination, files)`, and an old five-field object containing
only `registry_id`, `record_revision`, `mode`, `incarnation`, and `generation`. The latter binds no
destination and no exact bundle names, digests, or sizes. No compatibility or migration policy
approves that persistent shape, so it must be rejected rather than carried as a legacy reader.

Adopt one replacement package, **C-R**, at the existing export boundary. This is a new package
with its own correction allowance, not a third A correction. Reject these alternatives:

- accepting the old five-field journal as backward compatibility, because no released or accepted
  Stage 2 format exists and it weakens the exact prepared-export invariant;
- making the broad session suite C-R's ready gate, because it can pass while the rejected exporter
  remains installed;
- sending the parser change back to A, because A's two correction rounds are exhausted; and
- adding another transaction, journal version, fallback parser, or downgrade policy. The accepted
  public transaction and strict legacy reader are sufficient.

### 7.2 C-R contract and temporary ownership transfer

Dispatch one `development_worker_high` for C-R. High is required because the replacement joins a
persistent journal grammar, exclusive destination publication, and a proof of downgrade fidelity
after the earlier Medium prototype was rejected. C-R owns only:

- all of `nanobot/session/session_export.py`;
- all of `tests/session/test_session_mode_export.py`; and
- in `nanobot/session/session_mode.py`, exactly deletion of `_fsync_dir`, deletion of
  `_atomic_write`, and deletion of the old five-field export arm inside
  `_valid_journal_binding()`.

No other symbol in `session_mode.py` transfers. C-R may not edit `begin_export()`,
`SessionExportTransaction`, `locked_snapshot()`, `_append_journal()`, storage/manager/source files,
the A tests, defaults, guards, CLI, dependencies, or any package B/D/E file. It may add one
export-local module only if keeping the pure normalizer in `session_export.py` would materially
reduce readability. A and B must not write `session_mode.py` while C-R is active. After root accepts
C-R, ownership of the whole shared manager/mode boundary transfers to B for its already specified
package; C-R receives no later shared-file corrections.

C-R must replace the prototype rather than adapt its private path:

1. Require the existing offline-operator authority and an absolute nonexistent destination. Enter
   `locked_snapshot(key)`, require its resolved binding to be canonical, and strictly parse the
   returned source bytes. Build every bundle byte in memory from that exact snapshot. Do not read
   the active JSONL through a path or a second live-store load.
2. The lossless bundle contains the exact snapshot bytes as `canonical-source.jsonl` and the
   specification Section 5.4 manifest. The manifest binds the canonical classification, complete
   session binding, source/checkpoint schema, artifact references, and the canonical source file's
   exact digest and size. The transaction plan, not a self-referential manifest checksum, binds the
   exact digest and size of every bundle file. Private blobs are not labelled portable.
3. When compatibility is requested, create a tentative legacy candidate from the strict source
   parse using the existing legacy grammar. Preserve every normalized public event, exact
   assistant-call/tool-result pairing, every portable durable metadata value, the exact archive
   boundary, and inline artifact evidence. Source ownership/linkage fields and source-only metadata
   are not reclassified as legacy data. Active or recovery-required execution, an unresolved tool
   tail, ambiguous provider call pairing, unsupported metadata/role, or source-only artifact
   evidence without its inline representation is unsupported.
4. Call `begin_export(key, destination, files)` exactly once with the complete immutable file map.
   Thus the durable `prepared` row always contains the destination identity and the exact name,
   digest, and size of every proposed bundle file. Re-read the transaction-owned staged files;
   parse `compatibility-history.jsonl` through `parse_legacy_bytes(expected_key=key)` and compare
   its normalized public transcript, tool pairing, portable metadata, archive boundary, and inline
   evidence with the strict canonical input. A raw `json.loads` comparison is not proof.
5. On a proof or validation failure after `prepared`, call only the transaction's pre-publication
   `abort()` and return the fixed unsupported error. The active source and mode record must remain
   byte-identical and the final destination absent. A crash leaves the accepted recovery protocol
   to `recover`; code must not fabricate an abort after publication.
6. On success call only `mark_published()` and then `complete()`. Do not import/call
   `_append_journal`, `_atomic_write`, `_fsync_dir`, resolver path fields, or `os.replace`. The
   transaction remains the sole staging, no-overwrite publication, verification, and phase owner.

The five-field export binding must fail closed through normal journal parsing/resolution. Do not
add a version discriminator or grandfather test for it. Removing the two export-only helpers is
part of deleting the rejected C implementation surface; if another accepted caller is discovered,
C-R stops and reports the contradiction rather than expanding ownership.

### 7.3 C-R ready gate

`test_session_mode_export.py` must independently cover: exact lossless bytes and manifest binding;
a real strict-reader compatibility round trip with tool pairing, portable metadata, nonzero archive
boundary, and inline artifact evidence; unsupported source-only evidence with byte-identical source
and record plus no final destination; rejection of the old five-field journal binding; transaction
use without private helper calls; destination preexistence/no-overwrite; and crashes before
publication, after destination install, after `published`, and before `complete`. Existing export
cases in the crash matrix remain dependency evidence and are not moved into C-R ownership.

The C-R worker runs, in order:

```bash
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/session/test_session_mode_export.py --tb=short

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/session/test_session_mode_export.py \
  tests/migration/test_p5_stage2_crash_matrix.py -k export --tb=short

/Users/jaden/.pyenv/versions/3.13.1/bin/ruff check \
  nanobot/session/session_export.py nanobot/session/session_mode.py \
  tests/session/test_session_mode_export.py tests/migration/test_p5_stage2_crash_matrix.py

NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
  -m compileall -q nanobot/session/session_export.py nanobot/session/session_mode.py

git diff --check
```

Root then inspects the actual diff and journal rows, reruns both pytest commands, and reruns A's
focused command from Section 5 to prove that the narrow parser deletion did not regress the frozen
foundation. C-R is ready only if all three pytest selections pass and the diff contains no private
export path. A broad suite is deferred until fan-in and cannot substitute for any of these checks.
C-R gets at most two bundled correction rounds. A failure requiring any shared change beyond the
three transferred symbols is `REJECT`: do not spend a third A correction or silently widen C-R.

### 7.4 Revised sequencing and final acceptance

The ready graph is now:

1. A is frozen and complete as dependency evidence.
2. Run C-R. D may run concurrently because its write set is disjoint, preserving the maximum of
   two concurrent write workers. B waits because it will regain the shared `session_mode.py`
   boundary only after C-R passes.
3. After C-R root acceptance, run B (and D if not already complete). Then run E only after B, C-R,
   and D are each accepted. No partial package changes the Stage 1 default.
4. Run the complete Section 10 fan-in selections unchanged. The required export module is the C-R
   result; no count from the rejected prototype is carried forward.
5. Give one fresh `development_analysis_medium` reviewer the accepted specification, this full
   repair contract, raw diff, package and fan-in results, and the old-binding rejection evidence.
   The reviewer returns PASS/CORRECT/REJECT. Root independently verifies AC1-AC12 and alone may
   issue Stage 2 PASS.

Failure and rollback remain unchanged: before publication an exact transaction abort/recovery may
remove only operation-owned staging; after exclusive destination installation recovery verifies
that exact destination and completes forward; export never changes session mode; a conflicting or
changed destination fails closed. Routine diagnostics remain fixed enums only and contain no keys,
paths, IDs, digests, content, references, receipts, or credentials. The default remains
`Stage1UnavailableSessionAuthority` until E, and this addendum authorizes no activation, live-state
operation, retention, guard change, install, commit, deployment, background work, or platform
qualification.

## 8. Repair review checkpoint and safe stop

Packages A and B are accepted as dependencies. C-R replaced the rejected exporter and removed
the old five-field binding; D implemented the copied-state scratch-owner artifact operator; E
implemented the operator/default wiring. Explicit authority selection now uses `is not None`, so
the historical falsey-injection finding is closed. The current integrated review nevertheless
returned **CORRECT**, not PASS, for AC2 bootstrap/list sidecar and journal evidence, AC5
delete/fork locking and exact source evidence, and AC6 canonical SDK/WebUI manager routing. The
complete evidence and exact qualified validation results are recorded in the
[repair checkpoint](p5-stage2-repair-checkpoint.md).

This is a safe-stop boundary, not another A correction or Stage 2 closure. E safety-only
correction 2 restored the actual Stage 1 unavailable default under this contract's rollback rule;
focused worker tests, exact root reruns, and independent narrow review accepted that rollback as
PASS. A fresh bounded integrity replan must follow. It also requires an explicit user decision
between a manager-owned synchronous out-of-turn canonical reset and an async canonical SDK reset
with legacy-only synchronous `clear()`; neither design is approved by this contract update.
