# A3-4 implementation packet — Client and control contracts

**Status.** Ready for packet review. Dispatch after A3-2 and A3-3 pass.
[Process contract](A3-2-runtime-process-ownership.md) · [Session contract](A3-3-durable-session-runtime.md) · [Client findings](../module-review-2026-11-06/integrated/I03-client-capabilities.md)

## Objective and boundary

Make admission, approval, Stop, and reconnect agree on the current session and turn.
Use the existing WebUI router, `NanobotClient`, shared React app, and TUI client.
Keep the current composer and queue interaction. Expose the result of each control through its owning request.
Keep child topology for A3-5 and the device protocol cutover for A3-7.

## 1. Replace text-based approval — R01, R02

Replace the text-derived authority in `ApprovalAdmission` with a typed decision.
The request contains an opaque approval ID, the owning session incarnation, run, effect identity, and current generation.
Bind the effect to the admitted arguments and target. Store the details on the server.
Accept a request ID and an explicit approve or deny decision from an authorized control.
Resolve the session, current permission, and exact pending request under the existing admission lock.
Consume an approval once and apply it only to the bound effect.
If the target, scope, generation, or authority changed, require a fresh request.
If the continuation requests a different effect, apply ordinary policy to that effect.

The WebUI approval card sends the typed decision. Give the TUI and supported command surfaces an explicit ID-based control.
Ordinary conversation text, tool output, and automation cannot make the decision.
Switch each supported surface with the server cutover. An old client can read the conversation, but its text cannot grant approval.
If a surface lacks an authenticated control, keep risky execution blocked on that surface.

Use the A3-1 target resolver and the existing path and scope gates.
Use the bound instance root for local folder grants.
Apply the same exact-effect requirement to native desktop input and nested agent actions.
Keep automatic Git synchronization in its owner. The presence of a Git word in a user command supplies no approval.

**Edit boundary.** `nanobot/agent/turn_admission.py`, the approval branch in `nanobot/agent/loop.py`, and the approval and dispatch checks in `nanobot/agent/policy.py`.
Use `nanobot/webui/inbound_commands.py` and `nanobot/webui/workspaces.py` for authenticated controls and grant projection.
Use `webui/src/components/thread/ApprovalRequestCard.tsx`, `webui/src/components/thread/ThreadShell.tsx`, `webui/src/lib/nanobot-client.ts`, and the matching TUI control.
Keep the folder-grant and temporary-access lifetimes separate from effect approval.

**AC4.1.** Two clients race to approve one effect. Only one valid decision can authorize the original effect.
A denied, expired, stale, or revoked decision starts no effect.
Chat text such as “yes” does not approve. A supported typed decision still continues the authorized action.

## 2. Retain queued input until admission and settle Stop — R06, R14

Reuse the existing request settings and `turn_id` where they identify the same attempt.
Keep `message_accepted` compatible as a route-level event. The current handler can emit it before the session owner commits input.
Define an owner receipt for durable admission before changing the queue contract.
Tie submission to one stable attempt ID, session incarnation, and intended active turn when steering.
Bind the attempt to its submitted payload. A repeated ID with different input must fail.
Check the target again at admission. Reject a steer if the target turn changed.
Queue selection accepts text, attachments, or both. Use the same rule for manual and automatic send.
Peek the next eligible item. Commit dequeue only after the owner confirms admission of that item.
If transport fails or the response is lost, keep the item with its attempt ID.
Reconcile against the same attempt after an authorized reconnect.
If the owner cannot confirm admission or rejection, keep the item unresolved and prevent automatic resubmission.
Record temporary-session and older-client behavior in the receipt contract before the cutover.

Send Stop for the admitted turn. Project working, stopping, and interrupted from the owner.
Keep stopping while process or source settlement is pending. Reconnect reads the same state.
Treat ordinary chat Stop aliases as requests routed to the same owner where supported.
Avoid a new client state framework.

**Edit boundary.** Turn reservation and projection in `nanobot/webui/inbound_commands.py` and `nanobot/session/webui_turns.py`.
Use the existing canonical admission owner for the receipt.
Record its exact write set in the accepted contract. Keep the WebUI transcript out of execution authority.
Use the existing Stop handler in `nanobot/command/builtin.py`.
Own queue state in `webui/src/components/thread/ThreadComposer.tsx`, `webui/src/hooks/useNanobotStream.ts`, `tui/src/app.ts`, and `tui/src/prompt-queue.ts`.
Keep existing transport envelopes where they carry the required identity. Add only the missing status or lookup fields.

**AC4.2.** Two sends that race for one idle session admit one turn. The second receives a keyed admission result without starting a concurrent turn.
A stale steer does not join another turn. An attachment-only first item sends before the later text item.
A route-level event without an owner commit leaves the item in place.
A failed local send leaves the item in place. A disconnect after owner acceptance reconciles to one admitted prompt.
A reconnect without proof does not create a second prompt.

**AC4.3.** Stop enters stopping until the run and its required effects settle.
A disconnect and reconnect during Stop preserves the same turn and state.
Repeated Stop calls do not stop a later turn or cause a second settlement.

## 3. Separate mutation outcome from acknowledgement — R06, R14, R16

Keep the existing bounded request-ID ledger and its source binding.
Record the authoritative effect before attempting response projection or a client notification.
Reuse the existing result receipt when the effect is known applied. Report projection or refresh failure separately.
If the owner cannot prove the effect, retain the uncertain disposition and require readback.
For known applied or partial effects, invalidate the affected projection even if the actor's response fails.
Read prior replay records through the current compatibility path. A completed legacy response alone proves no new authorization or effect.
Apply the first cutover to the control and workspace mutations in this packet. Leave storage cascade semantics with A3-8 and model reload semantics with A3-6.

Keep a failed sidebar save visibly pending or unsaved until bounded retry or authoritative readback.
Fence async client commits by the captured session and request generation.
Apply the fence to TUI context, WebUI channel setup and skill detail. Apply completion metadata only within the originating turn.
Keep transcript-only search and history reads on one authorized contract.
For a permitted transcript-only row, return the corresponding read-only display history. It cannot serve as a canonical execution source.
Parse request flags as their declared type at the edge.

**Edit boundary.** `nanobot/webui/replay_store.py`, the response and invalidation paths in `nanobot/webui/inbound_commands.py`, and the owning WebUI history and settings routes.
Use `webui/src/hooks/useSidebarState.ts`, `webui/src/lib/thread-event-projection.ts`, `webui/src/components/settings/channels/ChannelSetupPanel.tsx`, and `webui/src/components/settings/SkillsCatalogSettings.tsx`.
Keep mobile route and credential changes in A3-7.

**AC4.4.** Simulate a committed control mutation with a lost reply. Readback shows the applied result without a second effect.
A pre-effect refusal and an ambiguous timeout retain different outcomes.
An unsuccessful sidebar save remains visible. An earlier session's async completion cannot alter the current view.
An empty turn leaves prior assistant completion metadata intact.

## Verification and rollback

Start with `tests/agent/test_migration_approval_admission.py`, `tests/webui/test_permission_v2_api.py`, `tests/webui/test_durable_replay.py`, `tests/webui/test_webui_mutation_replay_auth.py`, and `nanobot/channels/websocket/tests/test_websocket_channel.py`.
Use `webui/src/tests/thread-composer.test.tsx`, `webui/src/tests/nanobot-client.test.ts`, `webui/src/tests/useSidebarState.test.tsx`, `tui/src/app.test.ts`, and `tui/src/prompt-queue.test.ts`.
Exercise a two-client session against the real gateway dispatcher using fixture effects.
Record which clients support each versioned control before cutting over.
Drain outstanding controls before a rollback. If rollback would restore approval by chat text, keep risky controls disabled until the matching server and client are restored.
Return the admission, control, and outcome contract to A3-5.
