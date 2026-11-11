# A3-3 implementation packet — Durable session runtime

**Status.** Ready for packet review. Dispatch after A3-1. A3-4 needs the accepted results of A3-2 and A3-3.
[A3-1 handoff](A3-1-authority-foundations.md) · [A3-2 process contract](A3-2-runtime-process-ownership.md) · [Session source review](../module-review-2026-11-06/reports/H02.md)

## Objective and boundary

Keep a conversation addressable by the same durable identity across restart.
Use `SessionManager`, the canonical source store, and the existing writer lease.
Keep one history writer for each session. Keep run, source, session, and permission identities distinct.
Add only the runtime-facing session facts that the existing session owner or the next cutover consumes.
Keep child launch and client navigation for A3-5.

## 1. Bind runtime metadata to the existing session

Approve the metadata key, version, writer, and older-reader behavior before any new field is persisted.
Include fixtures for an existing conversation and a stale parent incarnation in that contract.

Extend the existing session metadata read and write path under the current ownership check.
Use the existing instance namespace, session key, and incarnation to identify the session.
For a parent reference, retain the parent's identity and incarnation. A reused key cannot take ownership of an old child.
Keep the selected profile, native or external execution kind, delegation depth, optional backend binding, runtime generation, and recovery disposition.
Mint topology and execution facts at the admitted server boundary.
Keep credentials, live handles, and arbitrary client metadata outside those fields.
Use the existing atomic metadata update and revision checks.

An old conversation without runtime metadata opens as a main, default-profile native session.
Apply those defaults when reading it. Persist new facts when the current owner commits a change.
Keep existing keys, transcript content, artifacts, and legacy mode handling.
If a runtime reference is malformed or stale, keep the history readable and refuse its use for execution until the owner reconciles it.

**Edit boundary.** Runtime-facing metadata in `nanobot/session/manager.py` and the existing canonical metadata path in `nanobot/session/source_store.py`.
Use `nanobot/session/runtime_authority.py` and `nanobot/session/execution_owner.py` for acquisition and generation checks.
Update `nanobot/runtime_composition.py` only to expose the existing session owner through the runtime handle.
Do not introduce a parallel history store or duplicate session registry.

**AC3.1.** Open a pre-stage conversation, restart, and continue it through the normal SDK or gateway path.
The same session identity and history survive. No second session or duplicate event appears.
An admitted child fixture retains its parent, profile, depth, and execution kind across restart.
An old child reference cannot resolve to a new parent with the same key.

## 2. Reconcile execution state after restart

Require the current owner to reconcile prior running state before admitting new effects.
Preserve completed history and recorded receipts.
Project an interrupted legacy run as interrupted. Keep unknown effects fenced until existing recovery proves their disposition.
Resume a backend continuation only when its existing provider or backend binding proves that the exact state is resumable.
Use the current recovery and explicit operator flows for ambiguous state.
Avoid replaying an old child, approval, tool intent, or remote operation on startup.

**Edit boundary.** `nanobot/session/recovery.py`, the runtime authority boundary, and the existing run projection in `nanobot/agent/run_store.py` if it must expose the reconciled result.
Preserve checked reads, source receipts, offline export, and mode conversion.
On platforms where the writer lease cannot be proved, keep the managed writer closed.
Qualification for a supported platform needs an exclusive lease with the same failure guarantees.

**AC3.2.** Reopen a fixture with a completed turn, an interrupted run, and an uncertain effect.
The completed turn remains readable. The interrupted run stays terminal and the uncertain effect is not replayed.
A stale lease or generation cannot mutate the reopened session.
An unqualified platform does not acquire a managed writer.

## 3. Validate SDK ingest before publication — R06, E02-F01

In `SessionClient.ingest`, validate the complete detached input before updating cached or persisted session state.
Apply metadata and messages through the existing save boundary under the current ownership rules.
Preserve the refusal for managed persistent import.
On malformed input or a proved pre-commit failure, keep the visible session at its last committed state.
If the save may have committed, reconcile through the existing owner before publishing or retrying.

**Edit boundary.** `nanobot/sdk/clients.py` and the existing session save adapter if required.
Keep the SDK request and error contract.

**AC3.3.** Submit valid metadata and a valid message followed by an invalid message.
The SDK rejects the request. A later flush or restart exposes neither a partial message prefix nor the rejected metadata.
Valid unmanaged ingest and managed refusal still work.

## Verification and rollback

Start with `tests/session/test_source_store.py`, `tests/session/test_source_persistence_boundary.py`, `tests/session/test_execution_owner.py`, `tests/session/test_session_mode_conversion.py`, and `tests/agent/test_migration_sdk.py`.
Use a pre-stage session fixture and a current session fixture for restart checks.
Verify the rollback reader against new metadata before enabling writes in a shared instance.
Keep an export or snapshot from the pre-stage writer and record its restore procedure.
Do not reopen the same data root with two writer versions.
Return the session reference, recovery dispositions, and lease guarantees for A3-4 and A3-5.
