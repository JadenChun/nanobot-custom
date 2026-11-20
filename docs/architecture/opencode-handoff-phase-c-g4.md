# OpenCode Handoff Plan: Phase C — Gate G4 Display Parity & Rollback Audit

> **Branch:** `codex/chat-page-work-plan` (base commit: `2725983d`)
> **Model target:** DeepSeek on OpenCode (overnight autonomous run)
> **Scope:** **Gate G4 ONLY** — field-by-field provenance & live-consumer inventory, automated display-vs-canonical parity & rollback test suite, and Gate G4 audit record in `docs/architecture/gui-session-history-work-plan.md`.
> **Explicitly OUT OF SCOPE:** Package 5 transcript writer cutover, modifying `WebUITranscriptRecorder` ownership, touching `webui/` frontend code, Go mobile broker, and Package 6 bundled-app qualification.

---

## 1. Context & Authority Documents

Read these files first before writing any code:
1. `AGENTS.md` (repo conventions, linting, strict type regression rules)
2. `docs/architecture/gui-session-history-design.md` — specifically:
   - §"Final durable ownership" (lines 104–106)
   - Gate **G4** row in the gates table (line 119)
   - Durability requirements (line 140)
3. `docs/architecture/gui-session-history-work-plan.md` — specifically:
   - `### G4 inventory started; not cutover evidence` (lines 436–443)
   - `## Remaining implementation and acceptance` (lines 425–435)

### Current Branch State (`2725983d`)
Already completed, reviewed, verified, and pushed on `codex/chat-page-work-plan`:
- **G1, G2, Package 2 (server), Package 3 (WebUI), Package 4A (WebUI drafts/focus/usage), Gate G3 (backend)** (`c63a88a1`)
- **Phase A / Package 4B (WebUI next-turn settings + active-turn folder lock lift)** (`72eeeb5d` + review cleanup `2725983d`)
- **Phase B / Boundary 1 follow-up (warm revisit attach coalescing + bounded settled startup scan)** (`4cf01820`)

---

## 2. Hard Guardrails (Read Carefully)

1. **Zero production code changes in `nanobot/` or `webui/`.**
   Gate G4 is an **audit and proof gate**, not a writer cutover.
   - Do **NOT** modify `nanobot/webui/transcript.py`, `nanobot/webui/transcript_index.py`, `nanobot/session/turn_writer.py`, or any other file under `nanobot/` or `webui/`.
   - If you discover a genuine parity gap or a display fact that cannot be reconstructed from canonical source alone, **document it honestly** in the G4 audit table and assert the observed behavior in `tests/webui/test_g4_display_parity.py`. Finding a real gap is a *successful* G4 audit outcome; hiding it or prematurely refactoring production writers is a failure.
2. **Only two files should be created/modified:**
   - **CREATE:** `tests/webui/test_g4_display_parity.py`
   - **EDIT:** `docs/architecture/gui-session-history-work-plan.md`
3. **Never touch unrelated untracked files:**
   - Do NOT stage, commit, or delete `images/rebrand/` or `docs/architecture/opencode-handoff-comfy-video-assets.md`.
4. **Local toolchain (no `uv`, no `go`, no package installs):**
   - Python tests: `.venv/bin/pytest`
   - Python lint: `.venv/bin/ruff`
   - Python type check: `.venv/bin/basedpyright`
   - Regression comparator: `.venv/bin/python scripts/check_type_regressions.py`
5. **One clean git commit on `codex/chat-page-work-plan`** when all verification commands pass. Do **NOT** push (`git push`); leave the commit local for human review tomorrow.

---

## 3. Code Map (Where Everything Lives)

Use this map instead of scanning the entire repository:

### Canonical Source Owners
- `nanobot/session/turn_writer.py` — `SourceTurnWriter` (`admit_input`, `append_message`, `append_injection`, `begin_tool_batch`, `project_history`, `project_events`)
- `nanobot/session/source_store.py` — canonical JSONL source store, `SourceState` (`store_identity`, `incarnation`, `generation`)
- `nanobot/session/manager.py` — `SessionManager` (reset `/new`, delete, fork, header metadata)

### Display Transcript & Index Owners
- `nanobot/webui/transcript.py` —
  - `WebUITranscriptRecorder` (display writer)
  - `validate_retained_transcript_record` (line 451), `validate_retained_webui_thread` (line 533)
  - `replay_transcript_to_ui_messages` (line 3090)
  - `build_webui_thread_response` (line 4280)
  - `write_session_messages_as_transcript` (line 2049), `_session_backfill_turns` (line 2395)
  - `_recover_incomplete_turns` (line 2616), `_repair_unfinished_tail` (line 2680), `_recover_interrupted_rotation` (line 934)
  - `fork_boundary_message_count` (line 4074), `has_pending_tool_calls` (line 4083), `has_unfinished_transcript_tail` (line 4134), `completed_turn_ids` (line 4148)
  - `delete_webui_transcript_result` (line 2130)
  - `_sanitize_turn_usage` (line 102), `rewrite_local_markdown_images` (line 116), `_media_from_signed_urls` (line 3071)
  - `tool_trace_lines_from_events` (line 2806), `_bounded_tool_event` (line 2859), `_filter_covered_file_edit_tool_events` (line 2993), `_delegated_child_lifetime` (line 423)
- `nanobot/webui/transcript_index.py` —
  - `rebuild_transcript_index` (line 833), `delete_transcript_index` (line 385)
  - `on_transcript_record_appended` (line 1052), `on_transcript_rotated` (line 1125), `on_transcript_rewritten` (line 1252)
  - `read_preindex_latest_tail` (line 1522), `read_history_window` (line 1816), `read_prompt_summary_range` (line 2022)
- `nanobot/webui/ws_http.py` & `nanobot/webui/workspaces.py` — live HTTP/WS consumers of `build_webui_thread_response` and `/turns`

### Existing Tests to Reuse as Pattern References
Read these two test files to match fixture setup, monkeypatching of data dirs, and helper styles:
- `tests/webui/test_transcript_index.py`
- `tests/webui/test_managed_display_origin.py`
- `tests/webui/test_child_thread_route.py` (for delegated child lineage fixtures if needed)

---

## 4. Deliverable 1: `tests/webui/test_g4_display_parity.py`

Build a focused, fast, deterministic pytest suite (target runtime < 15s) covering **all 9 Gate G4 verification categories**. Every test must use `tmp_path` isolation (no writes to real `~/.nanobot`).

### Required Test Matrix (9 Categories)

1. **Prompt, Answer, Turn Identity, & Admission**
   - Compare canonical `SourceTurnWriter` committed turns vs. `WebUITranscriptRecorder` + `replay_transcript_to_ui_messages` + `build_webui_thread_response`.
   - Verify `turnId`, `promptId`, user/assistant text, and multi-prompt Steer turns (multiple user prompts inside one `turnId` preserve distinct `promptId` values in both replay and `read_prompt_summary_range`).
   - Verify a display-only row does **not** constitute canonical admission (canonical `SourceState` / `SourceTurnWriter` remains the sole admission proof).

2. **Permitted Reasoning Detail & Redaction**
   - Verify permitted reasoning summary/preview survives replay and `validate_retained_webui_thread`.
   - Verify private/raw internal reasoning or raw provider payload fields are not leaked into the retained UI projection.
   - Document in test docstring/assertions whether permitted reasoning text is reconstructible from canonical source or is a distinct display fact.

3. **Bounded Tool Traces, Approvals, & File-Edit Diffs**
   - Exercise `toolEvents`, `toolTrace`, approval outcomes, and `fileEdits` through `replay_transcript_to_ui_messages` and `build_webui_thread_response`.
   - Verify covered file-edit tool events are deduplicated (`_filter_covered_file_edit_tool_events`) and oversized tool values are bounded (`_bounded_tool_event`).
   - Assert explicitly which display-shaping facts (e.g. formatted `fileEdits` diffs, approval UI badges, bounded `toolEvents`) exist in display replay vs. canonical `SourceTurnWriter` events.

4. **Context Compaction Markers & Turn Usage**
   - Exercise compaction markers and per-turn token usage (`_sanitize_turn_usage`).
   - Verify both survive replay, segmentation, and index rebuild without corruption or negative/non-int leaks.

5. **Fork & Parent/Child Delegated Lineage**
   - Exercise a forked session (`fork_boundary_message_count`) and delegated child events (`_delegated_child_lifetime` / child run metadata).
   - Verify fork boundaries remain exact across replay and index windows, and stale/reset parent lifetimes do not leak child updates.

6. **Incomplete Tails, Interruption, & Crash/Restart Recoverability**
   - Simulate an unfinished turn (prompt + pending tool call or missing final assistant row): verify `has_unfinished_transcript_tail` and `has_pending_tool_calls`, and verify `_recover_incomplete_turns` / `_repair_unfinished_tail` reconciles cleanly against canonical session messages without duplicating turns.
   - Simulate an interrupted segment rotation (`.rotation-intent.json` left on disk): verify `_recover_interrupted_rotation` restores consistent committed state and `rebuild_transcript_index` recovers exact turn/prompt counts.

7. **Media, Attachments, Signed URLs, & Generated-Asset References**
   - Exercise user attachments and assistant generated assets / local markdown image links (`rewrite_local_markdown_images`, `_media_from_signed_urls`).
   - Verify media references survive replay and window paging without leaking unauthorized raw host paths.

8. **Lifecycle Cleanup (Reset `/new`, Delete, Source Replacement)**
   - Verify `delete_webui_transcript_result` removes active transcript, rotated segments, manifest, and index (`delete_transcript_index`), leaving no stale readable history.
   - Verify source replacement / incarnation bump (`SourceState` origin change) invalidates prior-lifetime cursors/locators and rejects stale prior-lifetime display rows.

9. **Reversible Read Compatibility & Rollback Proof**
   - **Index delete & rebuild parity:** Write a multi-turn, multi-segment transcript -> capture `build_webui_thread_response` (`latest`, `before`, `after`, `around`) and `read_prompt_summary_range` -> delete the entire index directory (`delete_transcript_index`) -> rebuild (`rebuild_transcript_index`) -> assert byte-for-byte / field-for-field identical `UIMessage` outputs, `promptId`s, cursors, and prompt summaries.
   - **Pre-index legacy compatibility:** Construct a legacy/pre-index JSONL transcript (no index directory present) -> assert `read_preindex_latest_tail` and `build_webui_thread_response` serve the latest window cleanly without requiring bulk migration.
   - **Rollback safety:** Verify that committed display JSONL + segment manifest remain directly readable by `replay_transcript_to_ui_messages` even if the index directory is deleted or ignored (proving an older reader or post-rollback boot still reads committed history intact).

---

## 5. Deliverable 2: Update `docs/architecture/gui-session-history-work-plan.md`

1. **Add a brief Phase A cleanup note** under the existing checkpoints (or update the Package 4A note at lines 376–378) so the doc accurately reflects that Package 4B (`72eeeb5d` + `2725983d`) and Boundary 1 follow-up (`4cf01820`) are complete and verified.
2. **Replace `### G4 inventory started; not cutover evidence` (lines 436–443)** with a complete **Gate G4 Provenance, Live-Consumer, & Rollback Audit** section containing:
   - **Table 1 — Field-by-Field Provenance Matrix** (all 9 categories):
     Columns: `Display Fact | Canonical Source Owner | Display Transcript Owner | Reconstructible from Canonical Alone? (Yes / Partial / No) | Package 5 Requirement (Drop duplicate vs. Persist distinct display fact)`
   - **Table 2 — Live Consumer Inventory**:
     Every live reader/caller of `nanobot/webui/transcript.py` and `nanobot/webui/transcript_index.py` (found via grep across `nanobot/`), noting file, symbol, and what it reads.
   - **Rollback & Pre-Index Compatibility Verdict**:
     Explicit statement of what `tests/webui/test_g4_display_parity.py` proved (index delete/rebuild equivalence, pre-index legacy read, interrupted rotation recovery, lifecycle cleanup).
   - **Gate G4 Decision**:
     Clear, evidence-backed verdict for Package 5 — either:
     - `G4 PASSED (with explicit distinct display-fact list)` listing the exact facts Package 5 must retain as distinct display facts vs. facts it can drop from duplicate full-history writes, **OR**
     - `G4 OPEN (blockers found)` listing any fact or live consumer that currently lacks a canonical or distinct-fact replacement.
   - **Verification Log**:
     Exact commands run, pytest pass counts, ruff result, and `basedpyright` regression check (`introduced: 0`).

---

## 6. Verification Checklist (Must Pass Before Commit)

Run every command below from the repo root `/home/jaden/Documents/Utilities/nanobot-custom`:

### Step 0 — Capture frozen `basedpyright` baseline BEFORE editing
```bash
.venv/bin/basedpyright --outputjson > /tmp/g4-basedpyright-before.json || true
```

### Step 1 — Run the new Gate G4 suite + existing WebUI transcript/origin suites
```bash
.venv/bin/pytest \
  tests/webui/test_g4_display_parity.py \
  tests/webui/test_transcript_index.py \
  tests/webui/test_managed_display_origin.py \
  tests/webui/test_child_thread_route.py \
  -v
```
All tests must pass (0 failures, 0 errors).

### Step 2 — Lint & whitespace check
```bash
.venv/bin/ruff check nanobot/ tests/webui/test_g4_display_parity.py
git diff --check
```

### Step 3 — Strict type regression check (must report `PASS_NO_NEW_DIAGNOSTICS`, `introduced: 0`)
```bash
.venv/bin/basedpyright --outputjson > /tmp/g4-basedpyright-after.json || true
.venv/bin/python scripts/check_type_regressions.py \
  --before /tmp/g4-basedpyright-before.json \
  --after /tmp/g4-basedpyright-after.json
```

### Step 4 — Verify git scope & commit
```bash
git status --short
```
Confirm ONLY `tests/webui/test_g4_display_parity.py` and `docs/architecture/gui-session-history-work-plan.md` are modified/new (plus the two untouched untracked paths `images/rebrand/` and `docs/architecture/opencode-handoff-comfy-video-assets.md`).

Stage ONLY those two files and commit:
```bash
git add tests/webui/test_g4_display_parity.py docs/architecture/gui-session-history-work-plan.md
git commit -m "test(webui): complete Gate G4 display parity suite and rollback audit"
```
Do **NOT** run `git push`.

---

## 7. Starter Prompt for OpenCode (Copy-Paste)

> Read `docs/architecture/opencode-handoff-phase-c-g4.md`, `AGENTS.md`, `docs/architecture/gui-session-history-design.md`, and `docs/architecture/gui-session-history-work-plan.md` in full before making any edits. Execute **Phase C (Gate G4 — Display Parity & Rollback Audit)** strictly within the guardrails:
> 1. Capture the `basedpyright` baseline JSON at `HEAD` (`2725983d`) before editing.
> 2. Do **NOT** modify any production code under `nanobot/` or `webui/`, and do **NOT** perform Package 5 writer cutover.
> 3. Create `tests/webui/test_g4_display_parity.py` covering all 9 Gate G4 parity, lifecycle, crash/restart, pre-index, and rollback categories using `tmp_path` isolation.
> 4. Grep all live consumers of `nanobot/webui/transcript.py` and `nanobot/webui/transcript_index.py` across `nanobot/` and record the complete Gate G4 Field-by-Field Provenance Matrix, Live Consumer Inventory, Rollback Proof, and Package 5 Readiness Verdict in `docs/architecture/gui-session-history-work-plan.md`.
> 5. Run all required `pytest`, `ruff`, `git diff --check`, and `scripts/check_type_regressions.py` checks (`PASS_NO_NEW_DIAGNOSTICS`, `introduced: 0`), then make one clean commit (`test(webui): complete Gate G4 display parity suite and rollback audit`) on `codex/chat-page-work-plan` without pushing and without touching `images/rebrand/` or `docs/architecture/opencode-handoff-comfy-video-assets.md`.
