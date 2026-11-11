# A3-5 implementation packet — Durable delegation cutover

**Status.** Ready for packet review. Dispatch after the A3-4 control contract passes.
[Session contract](A3-3-durable-session-runtime.md) · [Control contract](A3-4-client-control-contracts.md) · [Delegation evidence](../architecture-review-2026-11-06/delegation-context.md)

## Objective and boundary

Give each new delegated child a real session with independent execution and history.
Replace the two live delegation paths with one owner at the server and client cutover.
Run native children through the normal Nanobot runner.
Keep existing conversations and historical delegation display. Use durable child sessions for new work.

Before dispatch, record the supported profile policies and finite depth and live-child limits.
Retain the current per-role and per-turn limits where they describe the same resource.
Select the new tree limits explicitly. Keep a child from increasing its parent's authority or capacity.

## 1. Admit one durable child under the current session

Reserve capacity and allocate the child identity before launch.
Commit the parent reference, profile, narrowed policy facts, and accepted assignment through the A3-3 session owner.
Project the child row from the committed child reference. Keep one durable owner of the ancestry edge.
Bind one spawn attempt to its admitted parent and input identity. Recover a repeated attempt by the same identity.
If a child never started, release the reservation. If execution is uncertain, keep its durable record for reconciliation.
Use the same owner for foreground wait and background execution.
Let a background child continue after normal parent completion under its admitted scope and live revocation checks.

**Edit boundary.** Rework the active role path in `nanobot/agent/delegation.py` and its construction in `nanobot/agent/foreground.py`.
Use the A3-3 session authority, A3-2 process owner, and existing runner.
Route `nanobot/agent/tools/delegation.py`, `nanobot/agent/tools/explore.py`, and supported spawn callers through the same child owner.
Replace the live `SubagentManager` wiring in `nanobot/agent/loop.py` and tool construction in the same release.
An approved compatibility wrapper may translate an old call to the current owner. It cannot maintain a second topology.

**AC5.1.** Two siblings and one nested child have separate durable identities and independent histories.
The parent can join one child or complete while an admitted background child continues.
A repeated spawn request returns the same child. Depth or capacity refusal creates no active child or leaked reservation.

## 2. Narrow authority and own cancellation

Derive the child policy from the intersection of live parent authority, role policy, and the admitted assignment.
Use the selected workspace and the effective paths from A3-1. Keep private roots and ungranted history outside the assignment.
Carry only the governing instructions and inputs required by the task.
Apply disabled skill state and the existing skill read policy to each child.
Treat remote MCP `readOnlyHint` as a hint. Require trusted operation evidence for a read-only child.
Classify crawler expansion that clicks a page as a mutation. Refuse it under a read-only promise.
Keep A3-4 approval attached to the child's actual effect and live authority.

Fence child admission before stopping a subtree.
Stop one child and its descendants by their admitted identities. Keep siblings independent.
Keep the parent or child stopping until required effects and descendants settle.
On restart, use A3-3 reconciliation. An unresolved launch or effect remains uncertain.

**Edit boundary.** Child policy in `nanobot/agent/delegation.py` and `nanobot/agent/write_guard.py`.
Use the existing stop and ownership paths in `nanobot/agent/loop.py` and `nanobot/session/execution_owner.py`.
Keep MCP operation classification in `nanobot/agent/tools/mcp.py` and crawler action classification in `nanobot/crawler/worker.py`.

**AC5.2.** A child cannot widen its tool, file, approval, or network scope. A stale or revoked parent permission cannot authorize an effect.
Concurrent sibling writes respect admitted scopes. An untrusted read-only annotation cannot authorize a mutating call.

**AC5.3.** Stopping a child interrupts its descendants and leaves siblings running.
Stopping the parent stops the admitted active subtree. A restart preserves history and fences uncertain effects.
A prelaunch backend refusal and a settled child failure release their slots.
An uncertain launch stays owned and visible until reconciliation.

## 3. Bind the backend and client to the child session

Use a narrow backend interface owned by delegation. Define start, resume, send, steer, interrupt, wait, and close.
Use one runtime binding for the current child generation. Route native work to the normal runner.
Require backend proof before resume. Report unsupported external execution through the existing admission response.
This stage supplies the interface for future external backends.

Project typed child references and outcomes in the parent activity row.
Open the referenced session through existing authorized session reads.
Use a side view in the desktop workbench and a full-screen child view on mobile.
Keep the child row as the navigation entry point. Keep the existing sidebar scope.
Read legacy announcements for historical display only. Do not infer child identity or completion from text or old process-local task IDs.

**Edit boundary.** The backend contract and adapters live with `nanobot/agent/delegation.py`.
Use the current session read and event owners in `nanobot/session/webui_turns.py` and `nanobot/webui/inbound_commands.py`.
Use `webui/src/components/thread/ThreadShell.tsx`, `webui/src/lib/nanobot-client.ts`, and shared thread activity rendering.
Update `tui/src/protocol.ts` and `tui/src/app.ts` where they project or control the new typed events.

**AC5.4.** A parent activity row opens the same authorized child in desktop and mobile layouts.
Wait, steer, and Stop target the child's current generation. An unrelated client cannot open or control it.
Historical announcements remain readable without creating or controlling a child.
The active runtime works when the old announcement and injected-result text paths are absent.

## Verification and rollback

Start with `tests/agent/test_migration_foreground.py`, `tests/agent/test_migration_run_details.py`, `tests/agent/test_subagent_lifecycle.py`, and `tests/session/test_execution_owner.py`.
Verify the normal runner and the SDK or gateway entry points. Use real parent and child session stores with fixture providers.
Use `webui/src/tests/thread-shell.test.tsx`, `webui/src/tests/nanobot-client.test.ts`, and `tui/src/app.test.ts` for projection and control coverage.
Verify a native child, an unavailable external backend, and the backend lifetime contract with a fixture adapter.
Cut over creation, stop, read, and client projection together. Preserve old history without manufacturing child sessions.
For rollback, stop new admission and settle the active tree. Retain all committed child history and export it if the rollback reader cannot open it.
Keep the old runtime from adopting an unresolved child or rewriting its history.
