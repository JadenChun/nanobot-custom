# A3-8 implementation packet — Durability, retention, and scheduler

**Status.** Ready for packet review. Dispatch after the accepted A3-6 and A3-7 changes.
[Capability packet](A3-6-capability-readiness.md) · [Mobile packet](A3-7-mobile-pairing.md) · [Durability findings](../module-review-2026-11-06/integrated/I02-durability.md)

## Objective and boundary

Keep history, execution evidence, and references readable as they grow.
Bound work before accepting it. Apply retention only through the owner of the durable data.
Keep canonical history, WebUI display transcripts, cron state, and media under their existing owners.
Use A3-4 outcome semantics for applied, partial, and uncertain mutations.

## 1. Preserve source and transcript proof at capacity — R07, R12

First reject an append that would make a managed source unreadable. Check the actual serialized limit before acknowledgement or effects.
Keep a supported read or export path for existing oversize sources.
Separate the current mode proof from retained terminal journal evidence.
Choose a verified current index or a stopped-owner epoch before changing the writer format.
Record the read and rollback contracts. Keep explicit references to retained archives.
Preserve pending transitions and the original proof chain.
If proof or index publication fails, retain the last valid generation and refuse incompatible writes.
Keep archived history reachable. Record any future segmentation and archive format before the writer starts using it.

Give WebUI transcript rotation one active generation or committed watermark.
After restart, read each event from the committed segment or active tail once.
Use a reversible or collision-resistant transcript identity. Check the old mapping before migrating existing files.
Quarantine ambiguous old filename collisions for review. Never merge their histories automatically.
Apply the same read budget to the no-query transcript path and paginated reads.

**Edit boundary.** `nanobot/session/source_store.py`, `nanobot/session/session_mode.py`, and `nanobot/session/mode_storage.py` for managed source and proof.
Use `nanobot/webui/transcript.py` and its owning read route for display history.
Keep A3-3 identity and storage authority. Do not weaken proof checks to improve latency.

**AC8.1.** A boundary append either remains readable or fails before it is accepted.
The journal can cross its current live limit while preserving retained proof.
A crash between segment and tail publication leaves each turn visible once.
Distinct chat IDs never read or delete each other's transcript. Older exports remain verifiable.

## 2. Preserve cron and continuation durability — R07, R10, R19

Move blocking persistence behind an ordered, bounded lane for each owner.
Keep the process-continuation journal and cron stores separate.
Commit intent before dispatch. Keep the scheduler and delivery lease until the queued write and owned work settle.
Retain an unknown outcome when a crash occurs after dispatch and before settlement.
Requeue only when the owner can prove that the effect did not occur.
Add scheduler epoch rollover under the existing stopped-owner fence. Verify the predecessor chain before resuming timers.
Evict completed in-memory delivery entries after waiter settlement. Preserve durable receipts.

Keep cron administration within the authorized instance owner.
Keep the local trigger consumer inactive unless its durable delivery identity and approval provenance are separately accepted.
If activation is in scope, bind each delivery ID to one canonical settlement and preserve unknown outcomes.

**Edit boundary.** `nanobot/process_completion.py`, `nanobot/cron/runtime.py`, `nanobot/cron/scheduler_store.py`, and `nanobot/cron/delivery_runtime.py`.
Use `nanobot/triggers/local_runner.py` only for an accepted activation slice.
Keep the current channel sender receipt contract.

**AC8.2.** Inject crashes before dispatch and after dispatch but before settlement.
Known unstarted work can resume. Unknown work stays inspectable without a second effect.
Rollover preserves cursor and occurrence order. Cancel and shutdown wait for owned writes.
After completed delivery waiters settle, the live registry stays bounded and durable receipts remain readable.

## 3. Require complete reference evidence before deletion — R05, R12

Verify the canonical session roots, display copies, pending uploads, and other retained references before proposing blob deletion.
If a root or record is missing, corrupt, or unreadable, report incomplete evidence and retain the affected bytes.
Parse drive and UNC paths where the data format allows them. Check the result against the instance media root.
Keep migration dry-run read-only. Reclaim a sidecar only after the owner proves the matching bytes are unreferenced.
Return complete or partial deletion from the owning session and display cleanup paths.
Retain failed Temporary Chat cleanup until the owner can retry or quarantine it.
Retire tombstones only after late events and active references are fenced.

Keep Dream's current mutation authority and audit scope in agreement.
Include every admitted skill change in the same diff, commit, and restore owner, or refuse that change before it writes.
Advance the cursor only after the existing owner can prove settlement.

**Edit boundary.** `nanobot/attachments/lifecycle.py`, `nanobot/webui/media_maintenance.py`, `nanobot/webui/temporary_chats.py`, and the existing session deletion and transcript cleanup paths.
Keep Dream mutation and restore changes in `nanobot/agent/memory.py` and its existing run path.

**AC8.3.** Missing or corrupt reference evidence prevents prune. A dry-run leaves the source, blobs, and sidecars unchanged.
A partial delete reports what remains and preserves retry ownership. A retained reference blocks blob and sidecar reclamation.
A Dream run that changes only a skill records the change or rejects it before writing.
An interrupted Dream run keeps its cursor and restores the admitted paths.

## 4. Bound parser input and expose a complete asset index — R13

Page the generated asset library beyond the current session cap.
Use a stable continuation that retains the filter and ordering. Report truncation when the bounded index cannot continue.
Keep session and cursor checks from A3-6.
Check image count and input bytes before reading full reference images. Carry the encoded-byte budget to provider dispatch.
Enforce PDF expansion limits before materializing page content. Use a bounded decoder path or refuse unsupported streams.
If the configured audio duration is a server limit, measure it with a bounded parser before provider dispatch.
Record the chosen duration contract before dispatching this slice.

**Edit boundary.** `nanobot/webui/generated_asset_index.py`, `nanobot/webui/generated_assets.py`, the existing route, and `webui/src/hooks/useGeneratedAssetLibrary.ts`.
Use `nanobot/agent/tools/image_generation.py`, `nanobot/providers/image_generation.py`, `nanobot/utils/document.py`, and `nanobot/audio/transcription.py` for their own budgets.

**AC8.4.** An asset from a later session can be reached with a continuation. Repeated pages expose each matching asset once.
Oversize references or expanded documents fail before the unbounded allocation or provider request.
Audio above an enforced duration limit cannot pass by declaring a shorter duration.

## Verification and rollback

Use `tests/session/test_source_store.py`, `tests/session/test_session_mode_operations.py`, `tests/utils/test_webui_transcript.py`, and `tests/cron/test_scheduler_store.py`.
Use `tests/cron/test_cron_delivery_runtime.py`, `tests/tools/test_process_completion.py`, `tests/attachments/test_attachment_lifecycle.py`, and `tests/webui/test_media_maintenance.py`.
Use `tests/agent/test_p6_dream_compact_boundaries.py`, `tests/webui/test_generated_assets.py`, `tests/tools/test_image_generation_tool.py`, and `tests/test_document_parsing.py`.
Verify recovery on copied stores before enabling any writer or destructive action on retained data.
Retain before and after source generations, archive checksums, and restore evidence.
If the old reader cannot open a new format, use the verified export or rollback snapshot. Preserve the newer evidence for reconciliation.
