# OpenCode Overnight Handoff Plan: Gate G4 -> Package 5 -> Package 6 (Linux Qualification)

> **Branch:** `codex/chat-page-work-plan` (base commit: `3b4012a0`)
> **Model target:** DeepSeek on OpenCode (overnight autonomous run — best-effort through all feasible remaining stages)
> **Execution rule:** Execute **Stage 1 -> Stage 2 -> Stage 3** strictly in order. Make **one clean, independently verified local git commit per stage**. Do **NOT** start Stage N+1 until Stage N passes every verification gate and is committed. Do **NOT** run `git push`.

---

## 1. Context & Authority Documents

Read these files first before editing anything:
1. `AGENTS.md` (repo conventions, linting, strict type regression rules)
2. `docs/architecture/gui-session-history-design.md` — especially §"Final durable ownership" (lines 104–106), Gate **G4** (line 119), Packages **5–6** (lines 129–130), and Durability (line 140)
3. `docs/architecture/gui-session-history-work-plan.md` — especially `## Remaining implementation and acceptance` (lines 425–443)

### Already Completed & Verified on `codex/chat-page-work-plan`
- **G1, G2, Package 2 (server), Package 3 (WebUI), Package 4A (WebUI drafts/focus/usage), Gate G3 (backend)** (`c63a88a1`)
- **Phase A / Package 4B (WebUI next-turn settings + active-turn folder lock lift)** (`72eeeb5d` + review cleanup `2725983d`)
- **Phase B / Boundary 1 follow-up (warm revisit attach coalescing + bounded settled startup scan)** (`4cf01820`)

### Explicitly Blocked / Hardware-Unavailable (Do NOT Attempt)
- **Go mobile broker** (`mobile/native-network/internal/broker/policy.go`) and `17f1b6cf` reconciliation — `go` is not installed on this machine.
- **macOS Codex Desktop side-by-side** and **paired Android device smoke** — requires macOS/Android hardware.
- **Staging branch merge** — keep `codex/chat-page-work-plan` separate from `staging` / `codex/staging-audit-implementation`.

---

## 2. Global Guardrails (Apply to Every Stage)

1. **Local toolchain only (no `uv`, no `go`, no package installs):**
   - Python tests: `.venv/bin/pytest`
   - Python lint: `.venv/bin/ruff`
   - Python type check: `.venv/bin/basedpyright`
   - Type regression check: `.venv/bin/python scripts/check_type_regressions.py`
   - WebUI (only when `webui/` is touched): `/home/jaden/.bun/bin/bun` inside `webui/`
2. **Capture a frozen `basedpyright` baseline BEFORE Stage 1:**
   ```bash
   .venv/bin/basedpyright --outputjson > /tmp/overnight-basedpyright-before.json || true
   ```
   Every stage that touches Python MUST pass `scripts/check_type_regressions.py --before /tmp/overnight-basedpyright-before.json --after /tmp/overnight-basedpyright-after.json` with **`PASS_NO_NEW_DIAGNOSTICS` (`introduced: 0`)**.
3. **Never touch unrelated untracked files:**
   - Do NOT stage, commit, modify, or delete `images/rebrand/` or `docs/architecture/opencode-handoff-comfy-video-assets.md`.
4. **Never leave a half-broken commit:**
   - If time runs out or an unresolvable blocker is hit mid-stage, revert any uncommitted partial edits for that stage (`git checkout -- ...`), record the exact stopping point in `docs/architecture/gui-session-history-work-plan.md` if appropriate, and stop cleanly so all prior committed stages stay green.
5. **Do NOT push (`git push`).** Leave all overnight commits local on `codex/chat-page-work-plan` for review tomorrow.

---

## 3. Code Map (Avoid Blind Repo Scans)

- **Canonical Source Owners:**
  - `nanobot/session/turn_writer.py` — `SourceTurnWriter` (`admit_input`, `append_message`, `append_injection`, `begin_tool_batch`, `project_history`, `project_events`)
  - `nanobot/session/source_store.py` — canonical JSONL source store, `SourceState` (`store_identity`, `incarnation`, `generation`)
  - `nanobot/session/manager.py` — `SessionManager` (reset `/new`, delete, fork, `replace_header_atomic`, `read_startup_header_metadata`)
  - `nanobot/session/next_turn_preferences.py` — detached origin-bound `.session-prefs/` store
- **Display Transcript & Index Owners:**
  - `nanobot/webui/transcript.py` — `WebUITranscriptRecorder`, `validate_retained_transcript_record` (451), `validate_retained_webui_thread` (533), `_recover_interrupted_rotation` (934), `write_session_messages_as_transcript` (2049), `delete_webui_transcript_result` (2130), `_session_backfill_turns` (2395), `_recover_incomplete_turns` (2616), `_repair_unfinished_tail` (2680), `tool_trace_lines_from_events` (2806), `_bounded_tool_event` (2859), `_filter_covered_file_edit_tool_events` (2993), `_media_from_signed_urls` (3071), `replay_transcript_to_ui_messages` (3090), `fork_boundary_message_count` (4074), `has_pending_tool_calls` (4083), `has_unfinished_transcript_tail` (4134), `completed_turn_ids` (4148), `build_webui_thread_response` (4280)
  - `nanobot/webui/transcript_index.py` — `delete_transcript_index` (385), `rebuild_transcript_index` (833), `on_transcript_record_appended` (1052), `on_transcript_rotated` (1125), `on_transcript_rewritten` (1252), `read_preindex_latest_tail` (1522), `read_history_window` (1816), `read_prompt_summary_range` (2022)
  - `nanobot/webui/ws_http.py` & `nanobot/webui/workspaces.py` — live HTTP/WS consumers of thread history & `/turns`
- **Existing Test Pattern References:**
  - `tests/webui/test_transcript_index.py`
  - `tests/webui/test_managed_display_origin.py`
  - `tests/webui/test_child_thread_route.py`
  - `tests/session/test_next_turn_preferences.py`
  - `tests/session/test_recovery_startup_scan.py`

---

## 4. Stage 1 — Phase C: Gate G4 Display Parity & Rollback Audit

### Rules for Stage 1
- **Zero production code edits in Stage 1.** Only create `tests/webui/test_g4_display_parity.py` and update `docs/architecture/gui-session-history-work-plan.md`.
- Be 100% honest about which display facts cannot be reconstructed from canonical source alone — that list directly drives Stage 2.

### Deliverable 1A: `tests/webui/test_g4_display_parity.py`
Use `tmp_path` isolation and cover all **9 Gate G4 categories**:
1. **Prompt, answer, turn identity, & admission**: compare canonical `SourceTurnWriter` vs. `WebUITranscriptRecorder` / `replay_transcript_to_ui_messages` / `build_webui_thread_response` (`turnId`, `promptId`, multi-prompt Steer turns; display row is NOT proof of canonical admission).
2. **Permitted reasoning detail & redaction**: permitted reasoning survives replay + `validate_retained_webui_thread`; raw/private provider reasoning is never persisted or leaked.
3. **Bounded tool traces, approvals, & file-edit diffs**: `toolEvents`, `toolTrace`, approval outcomes, `fileEdits`, `_filter_covered_file_edit_tool_events`, `_bounded_tool_event`.
4. **Context compaction markers & turn usage**: compaction markers and `_sanitize_turn_usage` survive replay, segmentation, and index rebuild.
5. **Fork & parent/child delegated lineage**: `fork_boundary_message_count`, `_delegated_child_lifetime`, stale parent lifetime isolation.
6. **Incomplete tails, interruption, & crash/restart recoverability**: `has_unfinished_transcript_tail`, `has_pending_tool_calls`, `_recover_incomplete_turns`, `_repair_unfinished_tail`, and `_recover_interrupted_rotation` (`.rotation-intent.json` left on disk).
7. **Media, attachments, signed URLs, & generated-asset references**: `rewrite_local_markdown_images`, `_media_from_signed_urls`, no unauthorized raw host path leakage.
8. **Lifecycle cleanup**: `delete_webui_transcript_result` + `delete_transcript_index` remove all segments/manifest/index; source replacement / incarnation bump invalidates prior-lifetime cursors and rejects stale display rows.
9. **Reversible read compatibility & rollback proof**:
   - Index delete + rebuild (`delete_transcript_index` -> `rebuild_transcript_index`) produces identical `UIMessage` rows, `promptId`s, cursors, and `read_prompt_summary_range` outputs across `latest`/`before`/`after`/`around`.
   - Legacy pre-index JSONL (no index dir) is served cleanly by `read_preindex_latest_tail` + `build_webui_thread_response` with no bulk migration.
   - Rollback safety: deleting/ignoring the index still leaves committed JSONL + segment manifest directly readable by `replay_transcript_to_ui_messages`.

### Deliverable 1B: Update `docs/architecture/gui-session-history-work-plan.md`
- Update the Package 4B / Boundary 1 status lines so they reflect `72eeeb5d`, `4cf01820`, and `2725983d`.
- Replace `### G4 inventory started; not cutover evidence` with:
  1. **Field-by-Field Provenance Matrix** (all 9 categories: `Display Fact | Canonical Owner | Display Owner | Reconstructible from Canonical Alone? | Package 5 Action`)
  2. **Live Consumer Inventory** (every caller of `transcript.py` and `transcript_index.py` found via grep across `nanobot/`)
  3. **Rollback & Pre-Index Proof Summary**
  4. **Gate G4 Verdict** (`G4 PASSED WITH EXPLICIT DISTINCT DISPLAY-FACT LIST` or `G4 OPEN: <blockers>`)

### Stage 1 Verification & Commit
```bash
.venv/bin/pytest tests/webui/test_g4_display_parity.py tests/webui/test_transcript_index.py tests/webui/test_managed_display_origin.py tests/webui/test_child_thread_route.py -v
.venv/bin/ruff check nanobot/ tests/webui/test_g4_display_parity.py
.venv/bin/basedpyright --outputjson > /tmp/overnight-basedpyright-after.json || true
.venv/bin/python scripts/check_type_regressions.py --before /tmp/overnight-basedpyright-before.json --after /tmp/overnight-basedpyright-after.json
git diff --check
git add tests/webui/test_g4_display_parity.py docs/architecture/gui-session-history-work-plan.md
git commit -m "test(webui): complete Gate G4 display parity suite and rollback audit"
```

---

## 5. Stage 2 — Package 5: Transcript Ownership & Reversible Writer Cutover

> **Precondition:** Proceed to Stage 2 **ONLY** if Stage 1 committed cleanly AND the Gate G4 verdict confirms every live consumer and distinct display fact is accounted for. If Gate G4 finds an unresolvable blocker that cannot be safely addressed without breaking invariants, document it and skip to Stage 3.

### Architectural Requirements (from `gui-session-history-design.md` §104–106 & §129)
1. **Retire duplicate full-history transcript writes for the new format** without losing any required display fact:
   - Canonical session (`SourceSessionStore` / `SourceTurnWriter`) remains the sole durable store for model/execution history (prompt text, assistant answer text, turn/prompt identity, admission).
   - Persist **only distinct, permitted display facts** (exactly those identified in Stage 1's G4 matrix — e.g., bounded `fileEdits`, approval outcomes, bounded `toolEvents`/`toolTrace`, permitted reasoning summary, compaction markers, turn usage, delegated child display summaries, authorized media/asset refs, fork markers) bound to `(store_identity, incarnation, generation)` and `(turn_id, prompt_id)`.
   - Never persist raw internal reasoning, private tool payloads, or full model context in the GUI display projection.
2. **Zero bulk migration & 100% reversible read compatibility**:
   - Existing pre-cutover full transcripts (both segmented and legacy pre-index) MUST remain readable without migration.
   - Mixed sessions (older turns in pre-cutover format + newer turns in consolidated format) MUST replay seamlessly in `build_webui_thread_response`, `read_history_window`, and `read_prompt_summary_range`.
   - Deleting the transcript index and rebuilding it MUST reproduce identical `UIMessage` projections, `promptId`s, and cursors for legacy, consolidated, and mixed sessions.
   - Keep `GET /api/sessions/{key}/webui-thread?view=history` and `/turns` bounded (0 full canonical body loads for settled indexed reads; seekable range reads only).
3. **Preserve all G1/G2/G3 invariants**:
   - Never weaken managed-origin checks (`store_identity`, `incarnation`, `generation`), rotation recovery, deletion cleanup (`delete_webui_transcript_result` + `delete_transcript_index`), or active/next-turn settings isolation.

### Deliverable 2: Implementation + `tests/webui/test_package5_transcript_consolidation.py`
- Implement the consolidated display-fact writer/reader path cleanly and minimally.
- Add `tests/webui/test_package5_transcript_consolidation.py` proving:
  1. New turns no longer duplicate full canonical prompt/answer text in the display store (verify on-disk bytes/content directly).
  2. All distinct display facts from the G4 matrix survive live turn -> reload -> restart -> index delete & rebuild.
  3. Pre-cutover segmented and pre-index transcripts still read identically with zero migration.
  4. Mixed old + new turns page cleanly across `latest`, `before`, `after`, `around`, and `/turns`.
  5. Reset (`/new`), delete, incarnation replacement, and interrupted rotation/recovery behave identically and fail closed on stale origins.
- Update `docs/architecture/gui-session-history-work-plan.md` with the Package 5 checkpoint and evidence.

### Stage 2 Verification & Commit
```bash
.venv/bin/pytest tests/webui/ tests/session/ tests/security/test_workspace_access_recovery.py -q
.venv/bin/ruff check nanobot/ tests/webui/test_package5_transcript_consolidation.py
.venv/bin/basedpyright --outputjson > /tmp/overnight-basedpyright-after.json || true
.venv/bin/python scripts/check_type_regressions.py --before /tmp/overnight-basedpyright-before.json --after /tmp/overnight-basedpyright-after.json
git diff --check
```
Stage only Stage 2's modified/new files and commit:
```bash
git commit -m "feat(webui): consolidate display-fact transcript writes with reversible reader"
```

---

## 6. Stage 3 — Package 6 (Linux-Executable Qualification Suite)

> **Goal:** Execute all automated Linux-runnable qualification checks and record an exact sign-off ledger so only macOS/Android/Go hardware checks remain.

### Deliverable 3A: Backend Scale, Cold-Start, & Durability Qualification
Create `tests/webui/test_package6_qualification.py` verifying:
1. **Large-history bounded reads (200+ turns, multi-segment)**:
   - `build_webui_thread_response` (`latest`, `before`, `after`, `around`) and `/turns` (`limit=100`) succeed with **0 canonical full-body loads** and bounded byte-range reads.
2. **Cold-start settled recovery**:
   - `RecoveryCoordinator.scan` over a mix of large settled sessions and one journal-marked unsettled session loads 0 canonical bodies for settled sessions while fully recovering the unsettled one.
3. **End-to-end durability drill**:
   - Multi-turn write -> rotation -> interrupted rotation recovery -> index delete & rebuild -> restart -> identical `UIMessage` + `promptId` + pagination cursors.
4. **End-to-end active/next-turn isolation**:
   - Active turn A in folder A + next-turn folder B + Steer on A + natural turn A completion -> turn B admitted under B with `Once` grants retired and A never overwriting B.

### Deliverable 3B: WebUI Integrated Chat-Page Qualification
Create `webui/src/tests/package6-qualification.test.tsx` verifying the integrated user journey:
1. First paint of saved history + prompt rail visibility on a short latest window of a long chat.
2. Jump to unloaded older prompt -> fetch -> mount -> highlight -> live-tail isolation while viewing older window -> Go to latest.
3. Draft preservation across `A -> B -> A` switch and simulated relaunch (with incarnation invalidation).
4. Active-turn next-folder change while turn A runs + immediate Steer bypass + held-ack new-turn Send waiting for ack.

### Deliverable 3C: Broad Suite Run & Work-Plan Qualification Ledger
- Run full WebUI gates:
  ```bash
  (cd webui && /home/jaden/.bun/bin/bun run lint && /home/jaden/.bun/bin/bun run build && /home/jaden/.bun/bin/bun run test -- --maxWorkers=4 --minWorkers=1)
  ```
- Run broad Python tests:
  ```bash
  .venv/bin/pytest tests/webui/ tests/session/ tests/security/test_workspace_access_recovery.py tests/remote/ -q
  .venv/bin/ruff check nanobot/ tests/webui/test_package6_qualification.py
  .venv/bin/basedpyright --outputjson > /tmp/overnight-basedpyright-after.json || true
  .venv/bin/python scripts/check_type_regressions.py --before /tmp/overnight-basedpyright-before.json --after /tmp/overnight-basedpyright-after.json
  git diff --check
  ```
- Update `docs/architecture/gui-session-history-work-plan.md` with:
  - Package 6 Linux qualification pass counts and timings.
  - Explicit remaining external-environment checklist: (1) macOS Codex Desktop visual/native comparison, (2) paired Android device smoke, (3) Go mobile broker `mobile/native-network/internal/broker/policy.go` + `17f1b6cf` reconciliation.
- Commit cleanly:
  ```bash
  git commit -m "test(chat-page): add Package 6 Linux qualification suite and update work plan"
  ```

---

## 7. Copy-Paste Overnight Prompt for OpenCode

> Read `docs/architecture/opencode-handoff-phase-c-g4.md`, `AGENTS.md`, `docs/architecture/gui-session-history-design.md`, and `docs/architecture/gui-session-history-work-plan.md` in full before making any edits. Execute the overnight plan on `codex/chat-page-work-plan` strictly in stage order (**Stage 1: Gate G4 Audit -> Stage 2: Package 5 Transcript Consolidation -> Stage 3: Package 6 Linux Qualification**) at best effort for your available time:
> 1. Before any edits, capture the frozen `basedpyright` baseline at `HEAD` into `/tmp/overnight-basedpyright-before.json`.
> 2. Complete **Stage 1 (Gate G4)** first with **zero production code edits**, verify all gates (`pytest`, `ruff`, `git diff --check`, `scripts/check_type_regressions.py` -> `PASS_NO_NEW_DIAGNOSTICS`, `introduced: 0`), and make one clean local commit.
> 3. Only after Stage 1 is committed and green, proceed to **Stage 2 (Package 5)** if Stage 1's G4 verdict has no unresolvable blockers. Keep full backwards read compatibility (no bulk migration), verify all Python suites + `ruff` + `basedpyright` (`introduced: 0`), and make one clean local commit.
> 4. Only after Stage 2 is committed (or cleanly skipped with documented G4 blockers), proceed to **Stage 3 (Package 6 Linux qualification)**, run both Python and WebUI (`bun run lint`, `bun run build`, `bun run test -- --maxWorkers=4 --minWorkers=1`) full checks, update `docs/architecture/gui-session-history-work-plan.md`, and make one clean local commit.
> 5. Never touch `images/rebrand/` or `docs/architecture/opencode-handoff-comfy-video-assets.md`, never leave a broken half-commit (revert incomplete work if you hit a limit), and do **NOT** run `git push`.
