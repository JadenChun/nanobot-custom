# OpenCode Handoff Plan — Kohaku Chat Page Refactor (Remaining Packages)

> **Target model:** DeepSeek v4.1 Flash (or equivalent fast coding agent) via OpenCode  
> **Branch:** `codex/chat-page-work-plan`  
> **Authority docs:**
> - [gui-session-history-design.md](./gui-session-history-design.md)
> - [gui-session-history-work-plan.md](./gui-session-history-work-plan.md)
> - [AGENTS.md](../../AGENTS.md)

## Rules Before You Start (READ CAREFULLY)

1. **Work on top of the committed baseline on `codex/chat-page-work-plan`.** Do NOT amend existing commits, do NOT merge `staging` or `codex/staging-audit-implementation` (`17f1b6cf`), and do NOT push to remote.
2. **One commit per phase.** Make a separate, clean commit after completing and verifying each phase (Phase A, Phase B, Phase C) so a reviewer can audit each phase independently via `git diff`.
3. **Never weaken checks or tests.** Do not delete failing assertions, do not add `# type: ignore` / `eslint-disable` to hide regressions, and do not refresh the type-check baseline.
4. **Do NOT touch completed modules unless explicitly listed in a phase:**
   - Do NOT edit `nanobot/webui/transcript_index.py`, `webui/src/lib/composer-drafts.ts`, `webui/src/lib/context-usage-retention.ts`, or `mobile/native-network/internal/broker/policy.go`.
   - Do NOT retire or delete the WebUI transcript writer (Package 5 cutover is explicitly deferred until after human/Neon review of Phase C).
5. **Local toolchain paths on this machine:**
   - Python: `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/basedpyright`, `.venv/bin/python`
   - WebUI: `/home/jaden/.bun/bin/bun` (run inside `webui/`)
   - Frozen `basedpyright` baseline JSON:
     `/home/jaden/.gemini/antigravity/brain/af8c4d14-99ea-496a-9de3-dce8452f59b5/scratch/basedpyright-before.json`

### Required Verification Commands (Run Before Every Commit)

- **For any Python change (`nanobot/` or `tests/`):**
  ```bash
  .venv/bin/ruff check nanobot/
  .venv/bin/basedpyright --outputjson > /tmp/basedpyright-opencode.json
  .venv/bin/python scripts/check_type_regressions.py \
    /home/jaden/.gemini/antigravity/brain/af8c4d14-99ea-496a-9de3-dce8452f59b5/scratch/basedpyright-before.json \
    /tmp/basedpyright-opencode.json \
    --baseline-root "$PWD" --current-root "$PWD"
  git diff --check
  ```
  `check_type_regressions.py` **MUST** output `"verdict": "PASS_NO_NEW_DIAGNOSTICS"` and `"introduced": 0`.
- **For any WebUI change (`webui/`):**
  ```bash
  (cd webui && /home/jaden/.bun/bin/bun run lint && /home/jaden/.bun/bin/bun run build && /home/jaden/.bun/bin/bun run test -- --maxWorkers=4 --minWorkers=1)
  git diff --check
  ```
  All test files must pass with zero lint warnings.

---

## Phase A — Package 4B: WebUI Next-Turn Settings & Lifting the Active-Turn Folder Lock

**Goal:** Wire the already-completed Gate G3 backend next-turn preferences into the WebUI so users can select next-turn working folder, model preset, and reasoning effort immediately—even while a turn is running—without locking the folder picker, disturbing active Stop/Steer/recovery, or triggering a history reload.

### Already Implemented Backend Contract (Do NOT re-implement)
- `GET /api/sessions/{key}/webui-thread?view=controls` (`nanobot/webui/ws_http.py:2145-2180`) already returns:
  - `workspace_scope` (confirmed active scope)
  - `next_workspace_scope` (pending next-turn scope, when set)
  - `next_turn_preferences` (`{ revision, workspace_scope?, model_preset?, reasoning_effort?, permission_mode? }`)
  - `next_model_preset`, `next_reasoning_effort`
  - `model_preset`, `reasoning_effort` (confirmed active model/effort)
- Session attach (`nanobot/webui/session_projection.py:77-89`) already emits `next_model_preset`, `next_reasoning_effort`, `next_workspace_scope`, `next_turn_revision`.
- WebSocket `set_workspace_scope` (`nanobot/webui/inbound_commands.py:889-915`) already accepts `next_turn: true` (or `next_workspace_scope` in the payload) while a chat is running, persists via `set_next_turn_scope`, and replies with `{ workspace_scope, next_workspace_scope, next_turn_revision, source_token }` without mutating active turn A.
- Model/effort selection (`loop.set_session_model_preset`) already routes to `next_turn_preferences` when a turn is active and promotes on next new-turn admission.

### Exact Files to Edit
1. `webui/src/lib/types.ts`
   - Add `next_workspace_scope?: WorkspaceScopePayload | null`, `next_model_preset?: string | null`, `next_reasoning_effort?: string | null`, `next_turn_revision?: number | null`, `next_turn_preferences?: { revision: number; workspace_scope?: WorkspaceScopePayload; model_preset?: string; reasoning_effort?: string; permission_mode?: string } | null` to `WebuiThreadPersistedPayload` / `WebuiThreadControlsPayload` and session attach/update event types.
2. `webui/src/lib/nanobot-client.ts` (and `webui/src/lib/api.ts` if needed)
   - Extend the workspace scope mutation helper (`setWorkspaceScope` / `set_workspace_scope` envelope) with an optional `{ nextTurn?: boolean }` flag that sends `next_turn: true` and returns the correlated `{ workspace_scope, next_workspace_scope, next_turn_revision, source_token }` acknowledgement.
   - Expose `next_workspace_scope`, `next_model_preset`, `next_reasoning_effort`, `next_turn_revision` from `session_updated` and attach frames without dropping existing fields.
3. `webui/src/hooks/useSessions.ts`
   - Store `nextWorkspaceScope`, `nextModelPreset`, `nextReasoningEffort`, `nextTurnRevision` from `view=controls` (and reset them on session switch / `historyAccessDenied` / source replacement). Never fetch `view=history` just to update settings.
4. `webui/src/components/thread/WorkspaceControls.tsx` & `webui/src/components/thread/ThreadComposer.tsx` & `webui/src/components/thread/ThreadShell.tsx`
   - **Lift the active-turn folder-picker lock** when the backend supports `next_turn` (keep the picker enabled during an active run; display the selected next folder immediately as the next-turn choice, distinct from the active turn's folder if different).
   - When `turnActive` is true and the user chooses folder B:
     - Send `set_workspace_scope` with `next_turn: true`.
     - Do **NOT** clear or invalidate the active turn's `workspaceAccessRequest` (folder recovery card), do **NOT** move focus out of the editor, and do **NOT** alter Stop or Steer targeting for active turn A.
     - Show control-local pending/error feedback only when delayed or rejected; on rejection, restore the confirmed display and preserve the composer draft (never silently fall back to folder A).
   - **Ordering new-turn admission against in-flight setting changes:**
     - Track per-chat in-flight next-turn setting promises (`workspace`, `model/effort`, `permission` mode).
     - When the user presses **Send** for a new turn (or when automatic queue drain triggers after natural completion) while a next-turn setting acknowledgement is still in flight:
       - Wait for the dependent setting acknowledgement to settle for the matching chat incarnation before admitting the new turn.
       - If the setting acknowledgement **fails/rejects**, abort the send (keep the draft intact or keep the queued row ready) and show the rejection—**never** silently send under the old folder/model/mode!
       - **Important:** Stop and Steer MUST NOT wait on next-turn setting acknowledgements; they address active turn A immediately.
     - Consume setting responses only if they match the current chat incarnation and latest local operation sequence (ignore stale out-of-order replies).

### Phase A Tests Required (in `webui/src/tests/`)
Add focused tests in `webui/src/tests/thread-composer.test.tsx`, `webui/src/tests/thread-shell.test.tsx`, and/or a new `webui/src/tests/next-turn-settings.test.tsx`:
1. Start turn A in folder A -> folder picker stays enabled -> select folder B while turn A is running -> sends `next_turn: true`, displays B as next folder, keeps editor focus, leaves turn A's Stop/Steer and folder-recovery request untouched.
2. Hold folder B or model B acknowledgement -> press Send immediately -> Send waits for B's ack and uses B on success; if B is rejected, Send does NOT fire under A and the draft is preserved intact.
3. Same ordering for automatic queue drain on natural completion (queued row stays ready while setting ack is pending, drains once settled, refuses on rejection).
4. Out-of-order stale setting reply (e.g., A completes or older request resolves late) cannot overwrite a newer selection for B.
5. No `view=history` fetch is triggered by changing model, effort, folder, or permission mode.

### Phase A Exit Check
Run full WebUI lint + build + test suite + `git diff --check`. Commit with message: `feat(webui): wire Package 4B next-turn settings and lift active-turn folder lock`.

---

## Phase B — Boundary 1 Follow-Up: Revisit Attach Coalescing & Cold-Start Audit

**Goal:** Eliminate redundant attach hydrations when revisiting a chat on the same live WebSocket connection + same source incarnation, without weakening Stop, notifications, recovery, or uncertain-send proof; audit cold-start ordering in `nanobot/cli/gateway_runtime.py`.

### Exact Files to Inspect & Edit
1. `webui/src/lib/nanobot-client.ts` & `webui/src/hooks/useNanobotStream.ts`
   - Currently simultaneous listeners share their first attach, but switching A -> B -> A on the same still-open WebSocket connection can re-run full attach hydration.
   - Track attached chat state keyed by `(socketGeneration, chatId, sourceIncarnation, sourceGeneration)`.
   - On a warm revisit over the **same live open socket** with no reconnect gap and no unresolved input-attempt proof:
     - Reuse the active subscription/projection instead of sending a duplicate full attach or replaying stale recovery/Stop frames.
     - Always perform a fresh attach after a socket reconnect, after source replacement/reset, or when `hasUnsettledRun(chatId)` / explicit input-attempt proof is pending.
     - Never drop notification subscriptions for active or explicitly followed chats.
2. `nanobot/cli/gateway_runtime.py` (audit first; edit ONLY if clearly safe & bounded)
   - Inspect lines 570–685 where `await handle.mcp.connect()` and `await recovery.scan()` run before `channels.start_all()`.
   - Write a focused test in `tests/cli/` or `tests/webui/` measuring/qualifying whether `recovery.scan()` scans settled session bodies (it should use bounded metadata for settled sessions and only run full canonical recovery for unresolved sessions).
   - Do **NOT** weaken startup admission fences or allow new turn input to race past recovery. If splitting read-ready vs execution-ready requires risky lifecycle changes, document the exact findings in `docs/architecture/gui-session-history-work-plan.md` instead of forcing an unsafe refactor.

### Phase B Tests Required
- `webui/src/tests/nanobot-client.test.ts` (and `useNanobotStream.test.tsx`):
  - A -> B -> A revisit on the same live open socket does not send a redundant attach or replay stale Stop/recovery, while keeping active-chat notifications alive.
  - Reconnect gap or unresolved input-attempt proof still performs full attach + exact canonical admission reconciliation.
- Python test verifying `recovery.scan()` does not load canonical bodies of already-settled sessions (only unresolved ones).

### Phase B Exit Check
Run both Python checks (`ruff`, `basedpyright` regression check, owning `pytest`) AND WebUI checks (`lint`, `build`, `test`) + `git diff --check`. Commit with message: `perf(chat): coalesce same-socket revisit attach and bound settled startup scan`.

---

## Phase C — Gate G4 Parity & Rollback Audit (Tests + Inventory ONLY — NO Writer Cutover)

**Goal:** Produce deterministic test evidence for Gate G4 comparing the canonical source vs. WebUI display transcript across all required display facts, so Neon + user can review and approve (or reject) the Package 5 writer cutover safely.

**CRITICAL:** Do **NOT** disable, delete, or change the WebUI transcript writer in this phase! Only add parity tests and record exact findings in `docs/architecture/gui-session-history-work-plan.md`.

### Exact Deliverables
1. Create `tests/webui/test_g4_display_parity.py` testing real sessions across:
   - Prompt & answer text (including `!shell`, `/command`, Steer multi-prompt turns, cross-session messages)
   - Bounded tool traces, `fileEdits`, and approval outcomes
   - Permitted reasoning display vs. non-persisted raw/private reasoning
   - Context compaction notices and completion `contextUsage` / `roundUsages`
   - Fork boundary markers and delegated parent/child lineage rows
   - Incomplete tails & recovery state after simulated interruption/restart
   - Media/attachment references and cleanup on reset/delete
   - Rollback compatibility: old pre-cutover segmented transcripts remain readable alongside canonical-backed sessions
2. For each category, assert what canonical source already preserves vs. what currently exists **only** in the WebUI display transcript (clearly marking any display-only gap with a documented finding rather than faking parity).
3. Append a `### Gate G4 parity audit results` section to `docs/architecture/gui-session-history-work-plan.md` summarizing the exact matrix (pass vs. gap per category) and exact pytest output.

### Phase C Exit Check
Run `.venv/bin/pytest tests/webui/test_g4_display_parity.py -q`, `.venv/bin/ruff check nanobot/`, `basedpyright` regression check (`introduced: 0`), and `git diff --check`. Commit with message: `test(webui): add Gate G4 display-fact parity and rollback audit suite`.

---

## Handoff Prompt to Paste Into OpenCode

> Read `docs/architecture/opencode-handoff-plan.md` and `AGENTS.md` in full before editing anything. Execute **Phase A**, **Phase B**, and **Phase C** in order, making one verified git commit per phase on `codex/chat-page-work-plan`. Follow every guardrail strictly: do not touch completed G2/Package 2/Package 3/Package 4A files unless listed, do not retire the WebUI transcript writer in Phase C, and run all required lint, build, pytest, and `scripts/check_type_regressions.py` checks (`PASS_NO_NEW_DIAGNOSTICS`, `introduced: 0`) before every commit.
