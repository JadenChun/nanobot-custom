# Subagent display and response format implementation plan

Status as of 2026-11-27: P0, C, E, A, and B are implemented on
`codex/subagent-response-display`. D and F have not started.

Reference session that reproduces every issue:
parent `websocket:85f41d61-7609-46ac-af28-815123da98f5` and child
`delegated:64aeb66da18249f3bc6813df20cbd5bd` (claude_sdk explore child, 124 tool calls, result
"Status: partial (loop guard)"), stored under
`~/.nanobot/sessions/c8bce12efbae6f54fbef0366e1fec4b8/`.

## Product decisions

1. **The subagent appears where it was started.** A subagent row sits in the parent turn's activity
   stream, right after the tool call that spawned it. It is never appended to the end of
   the chat, and never shown below the final answer or a later turn.
2. **A subagent is a step, not a message.** While running it shows as a live step inside the activity
   group. After the turn it collapses with the other steps into one compact row: icon, short
   title, harness badge, status, duration, and step count. "Open history" opens the full transcript.
3. **Child history looks like the main stream.** The child history overlay uses the same
   step descriptions as the parent ("Read `foo.ts`", "Ran `rg …`"). It never shows
   "Completed step <tool output>".
4. **Partial results are not shown as Completed.** A loop-guard or partial result gets a
   warning state.
5. **Links work everywhere.** The model writes PRs, issues and commits as full markdown links. The desktop
   app opens safe http(s) links in the system browser on macOS and Linux.
6. **Response format guides, it does not restrict.** Answers match the depth of the question.
   Detailed explanations stay detailed when the user asks for them (P0).
7. **The final answer is visually distinct.** Interim progress notes inside a turn render as muted
   activity text, and only the final reply gets the answer bubble.

## Confirmed decisions and separate work

| Question | Default in this plan | Affects |
|---|---|---|
| `label` parameter on explore/delegate tools | Optional. Fallback is the first sentence of the assignment with workspace-relative paths, at most 60 characters | B |
| How to represent "partial" | `outcome: "partial"` (the lifecycle state stays `completed`). The UI maps it to a warning state | B |
| Moving `~/.nanobot` to `~/.kohaku` | Out of scope. It is a separate migration, because `~/.nanobot` is the live data folder | none |

## Dependency order and file ownership

| Wave | Package | Can run in parallel with | Owns (exclusive while running) |
|---|---|---|---|
| done | P0 Prompt response format | n/a | `kohaku/templates/agent/tool_contract.md`, `_snippets/format_hint.md`, `tests/agent/test_context_prompt_cache.py` |
| 1 | C Child history projection | E | `kohaku/webui/ws_http.py` (child history only), `webui/src/components/thread/activity/trace-activity-model.ts`, `AgentActivityCluster.tsx` (traceLines only), `DelegatedChildHistoryOverlay.tsx`, matching tests |
| 1 | E Desktop external links | C | `desktop/src-tauri/src/**` |
| 2 | A Inline anchoring | none | `kohaku/bus/outbound_events.py`, `kohaku/bus/outbound_wire.py`, `kohaku/agent/child_sessions.py`, `kohaku/agent/delegation.py`, `kohaku/webui/transcript.py`, `webui/src/lib/types.ts`, `delegated-child-projection.ts`, `activity-timeline.ts`, `ThreadShell.tsx`, `AgentActivityCluster.tsx`, `MessageBubble.tsx` |
| 3 | B Compact card, label, partial | none | `kohaku/agent/tools/explore.py`, `kohaku/agent/tools/delegation.py`, `kohaku/agent/child_sessions.py`, `kohaku/agent/delegation.py`, `MessageBubble.tsx`, `SubagentStatus.tsx`, `DelegatedChildHistoryOverlay.tsx` (title only) |
| 4 | D Session panel subagent list | F | `SessionInfoPopover.tsx`, `ThreadInspector.tsx`, a new `SubagentList.tsx`, tests |
| 4 | F Progress notes and subagent report shape | D | `webui/src/lib/activity-timeline.ts`, `MessageBubble.tsx`, `kohaku/templates/agent/subagent_system.md` |

Packages that share a file never run in the same wave. Each worker is told that it is not alone
in the codebase. It must not revert edits it did not make, and it must not run
`git add`, `git commit`, `git reset`, `git checkout` or `git stash`.

## Shared validation commands

```bash
# Python
ruff check kohaku/ tests/
pytest <package test paths> -q
# Strict types: zero new diagnostic signatures (docs/migration/p6-error-attribution.md)
uv run --no-sync basedpyright --outputjson > /tmp/<pkg>-after.json   # compare with baseline
uv run --no-sync python scripts/check_type_regressions.py <before.json> <after.json> \
  --baseline-root "$PWD" --current-root "$PWD"
# WebUI
cd webui && bun run test && bun run build
# Desktop
cd desktop/src-tauri && cargo check && cargo test
```

Each worker captures the basedpyright "before" report before editing any Python file.

---

## P0. Prompt response format (implemented)

- `tool_contract.md`: "Communication" is replaced by "Response Format". It covers leading with the outcome,
  matching depth to the ask ("Aim for readability, not brevity"), keeping answers easy to scan,
  skipping tool narration, putting questions at the end, writing full markdown links for PRs, issues and commits, and writing files as
  `[name](/abs/path:line)`.
- `format_hint.md`: new `websocket` branch saying the app renders GFM and that links and paths are clickable.
- Tests: 113 passing, ruff clean. The fresh-workspace prompt is 7,259 characters, under the 8,000 cap.
- The workspace `SOUL.md` ("I am nanobot") is a separate, user-owned follow-up.

---

## C. Child history projection matches the main stream

**Objective.** Tool rows in the child history overlay show the same step descriptions as the parent
stream, never "Completed step <output>". The overlay pins the assignment and shows a real duration.

**Evidence.**
- The row builder in `_child_history_response` (`kohaku/webui/ws_http.py` around 4901–5120, tool events
  around 4860–4897) emits tool rows as `kind:"trace"` with the tool **output** as `content` and no `traces`.
- `traceLines()` (`AgentActivityCluster.tsx` ~487) falls back to `content`, and
  `describeTraceLine` (`activity/trace-activity-model.ts` ~71) renders it as "Completed step …".
- The main stream builds trace text with `_format_tool_call_trace` (`kohaku/webui/transcript.py` ~3248).
- The assignment is the first child row and is hidden behind "Load earlier activity" (page limit 120).
  "Worked" has no duration.

**Acceptance criteria.**
1. Every child-history tool row has `traces` built with `_format_tool_call_trace` from the matching
   assistant `tool_calls` entry (matched by tool_call_id). Tool output is kept only as result detail, never as the
   trace line. If possible, reuse the same projection helper the parent transcript uses instead of a parallel copy.
2. Frontend safety net: when a trace row has `toolEvents` but no `traces`, trace lines come from the start events
   (`formatToolCallTrace`), never from `content`.
3. The overlay always shows the assignment at the top as a collapsible block, collapsed by default and
   showing its first line, whatever page is loaded. The response includes the assignment
   separately from the paged rows (for example `assignment` field), or the client gets it from the
   first page once and caches it.
4. "Worked" shows the duration from the first to the last child timestamp (for example "Worked 6m 12s").
5. Reference-session replay test: a fixture shaped like the 124-tool claude_sdk child produces rows
   whose trace lines start with a verb such as Read, Ran or Searched. No row starts with "Completed step".

**Test-first.** Add failing tests first:
- `tests/webui/test_delegated_child_replay.py` (or a new `test_child_history_traces.py`): history rows carry traces
  and not output-as-trace. Cover both native and claude_sdk child event shapes.
- `webui/src/tests/trace-activity-model.test.ts`: the fallback uses toolEvents.
- An overlay test (new `webui/src/tests/delegated-child-history-overlay.test.tsx`) for the pinned assignment and duration.

**Validation.** `pytest tests/webui -q -k "child or delegated"`, ruff, basedpyright regression check,
`cd webui && bun run test && bun run build`.

**Completion report.** Files changed, tests added and their results, before/after screenshots or a description of one
reference-session row, any deviation from the criteria.

---

## E. Desktop opens external links (macOS)

**Objective.** On macOS and Linux, clicking an http(s) link in the WebUI opens the system browser.

**Evidence.**
- `desktop/src-tauri/src/native.rs`: `on_new_window` denies everything (~739). `on_navigation` blocks off-site
  navigation (~700). The external-open path (~668–688, `open_external_url` ~718) is
  `#[cfg(target_os = "linux")]` only. `tauri_plugin_opener` is registered in `main.rs:151`. Window
  setup is in `controller.rs` ~947.
- The WebUI renders links correctly with `target="_blank"` (`MarkdownTextRenderer.tsx` ~631–680).

**Acceptance criteria.**
1. On macOS, a new-window request or an off-site navigation to an `http`/`https` URL calls the opener
   (`app.opener().open_url` or the existing `open_external_url`, no longer Linux-only), and the webview
   does not navigate.
2. Other schemes (`file:`, `javascript:`, `data:`, custom) are still denied. The scheme check is one shared
   pure function with unit tests.
3. In-app navigation (gateway origin) is unchanged.
4. The Linux behavior is unchanged.

**Test-first.** A Rust unit test for the URL classification function (allowed external, in-app, denied) before
wiring it in.

**Validation.** `cd desktop/src-tauri && cargo check && cargo test`. Manual check: run the desktop app on macOS,
click a `[#1](https://github.com/...)` link, and confirm it opens in the default browser.

**Completion report.** Diff summary, test output, the manual check result (or a note that it was not done).

---

## A. Subagent anchored inline in its turn

**Objective.** A subagent row renders in the activity stream right after its spawning tool call,
both live and after reload. It never moves to the end of the chat.

**Evidence.**
- `mergeDelegatedChildRows` appends child rows at the end (`webui/src/lib/delegated-child-projection.ts`
  ~611, ~629). Seen live: a card landed below a later turn's answer.
- `projectOrderedTurn` treats `delegated_child` as a hard section break (`activity-timeline.ts` ~229).
- The payload already has `turn_id`, `parent_run_id`. The parent `run_id` equals the child's `attempt_id`.
- The spawning tool call id is available when the child is reserved:
  `turn_context.active_tool_intent_binding.tool_call_id` (`kohaku/agent/turn.py` ~642,
  `SourceToolIntentBinding` in `kohaku/session/turn_writer.py` ~96). `delegation.py` ~1787 calls
  `tree.reserve(...)` (`child_sessions.py` ~1456), which has no call-id parameter today.

**Acceptance criteria.**
1. `reserve()` accepts an optional `spawn_call_id`, persists it in the child record (backward-compatible
   read: missing means `None`), and `delegation.py` passes the active binding's `tool_call_id`.
2. `DelegatedChildEvent`, `outbound_wire.py`, the transcript projection (`kohaku/webui/transcript.py`
   ~424–486 and ~4068–4175) and `webui/src/lib/types.ts` (~138) carry `spawn_call_id`.
3. `mergeDelegatedChildRows` places a child row after the row that holds the matching tool call
   (by `spawn_call_id`). Fallback order: (a) the last row of the same `turn_id` / `parent_run_id` before
   that turn's final answer; (b) append, only when nothing in the transcript matches.
4. `projectOrderedTurn` no longer splits the activity group at a delegated child. The child is a step
   inside the group, live (expanded status) and settled (collapsed with the other steps).
5. Hydration (`ThreadShell.tsx` ~2626–2700) and live updates give the same order. A reload
   does not move a child.
6. Old sessions without `spawn_call_id` render with fallback (a) and do not crash.

**Test-first.**
- Python: a reserve/persist/read round trip with and without `spawn_call_id`. Wire serialization includes the field.
  Transcript replay emits it.
- TS (`webui/src/tests/delegated-child-projection.test.ts`): inserts after the spawning call; turn fallback
  stays before that turn's final answer and before later turns; the legacy append path; live and hydrated order
  are the same.
- TS (activity timeline): a child inside one turn yields a single activity group.

**Validation.** `pytest tests/agent tests/bus tests/webui -q -k "child or delegat or outbound or transcript"`,
ruff, basedpyright regression check, `cd webui && bun run test && bun run build`.

**Completion report.** Files, tests, a description of reference-session ordering after the change,
any persisted-schema notes.

---

## B. Compact subagent card, short label, partial outcome

**Objective.** The settled subagent row is one line: icon, short title, harness badge, status, duration and
steps. Partial results show a warning state. The result summary keeps its structure.

**Evidence.**
- `DelegatedChildRow` in `webui/src/components/MessageBubble.tsx` (~93–133). Status line in
  `thread/SubagentStatus.tsx`.
- `task_label` = the first 160 characters of the assignment (`child_sessions.py` ~1482).
- `result_summary` = `_display_preview(result, 1000)`, which collapses newlines (~299–302, ~1051). For
  claude_sdk children it includes early narration, not only the final message.
- `explore.py` ~109–113 returns "partial (loop guard)" (from `delegation.py` ~3833–3893), but the card
  says Completed.

**Acceptance criteria.**
1. The explore and delegate tools accept an optional `label` (at most 60 characters, documented in the schema as "short
   title shown in the UI"). It is stored in the child record and sent as `task_label`. When it is missing,
   `task_label` is the first sentence of the assignment, with absolute workspace paths made relative, at most 60 characters.
   The full assignment stays available to the history overlay.
2. `result_summary` comes from the child's final assistant message only. Newlines are kept and the length
   is capped at 1000 characters.
3. A loop-guard or partial result sets `outcome="partial"` (the lifecycle state stays `completed`). The
   frontend shows a warning icon and "Partial" in place of "Completed". Existing outcomes are unchanged.
4. The settled card has a single row in the layout above. The summary is behind a disclosure ("Show result",
   rendered as markdown). Running state shows the live status line from `SubagentStatus`.
5. The history overlay header uses the short title.

**Test-first.** Python tests for label default and override, the fallback derivation, final-message summary
with newlines kept, and partial outcome mapping. TS component tests for the compact row, the partial
warning, and the summary disclosure.

**Validation.** `pytest tests/agent -q -k "explore or delegat or child"`, ruff, basedpyright regression
check, `cd webui && bun run test && bun run build`.

**Completion report.** Files, tests, a screenshot or description of the reference-session card.

---

## D. Subagents in the session panel

**Objective.** The Session details panel lists the subagents in the session, with navigation.

**Evidence.** `SessionInfoPopover.tsx` / `ThreadInspector.tsx` show only "Automations 0".

**Acceptance criteria.**
1. A "Subagents (n)" section lists each child with: short title, harness badge, status (including Partial),
   duration and step count, newest first.
2. "Jump to" scrolls the thread to the child's inline row (uses A's anchor) and briefly highlights it.
3. "Open history" opens `DelegatedChildHistoryOverlay`.
4. The empty state is hidden or says "No subagents". The section reuses existing projection data and adds no
   new endpoint unless one is needed.

**Test-first.** Extend `webui/src/tests/session-info-popover.test.tsx`: list rendering, status mapping,
and that the jump and open callbacks are called.

**Validation.** `cd webui && bun run test && bun run build`.

---

## F. Progress notes and subagent report shape

**Objective.** Interim assistant text inside a turn reads as activity, not as an answer. Subagent results
come back in a short, consistent shape.

**Evidence.** Interim notes ("I'll look for a browser tool…") render as full answer bubbles with
a timestamp and copy button (`projectOrderedTurn` / `isAssistantAnswer` in `activity-timeline.ts`,
`MessageBubble.tsx`).

**Acceptance criteria.**
1. Assistant text that is followed by more tool activity in the same turn renders as muted text inside
   the activity group, with no timestamp or copy button. Only the turn's final assistant message gets the answer
   bubble. Live streaming text is shown as the answer until a tool call follows it. Then it moves into
   the activity group with no layout jump beyond the restyle.
2. A turn whose only text is final is unchanged.
3. `kohaku/templates/agent/subagent_system.md` asks for a report shape: first line `Status: done|partial|blocked`,
   then findings with `[file](path:line)` references, then gaps or next steps. It is phrased as a default shape,
   with no length limit when detail is needed.

**Test-first.** Activity-timeline tests: interim text gets the activity type, the final text gets the answer type, and
streaming then a tool call reclassifies. A template test checks that the report shape is present.

**Validation.** `cd webui && bun run test && bun run build`,
`pytest tests/agent/test_context_prompt_cache.py -q` (plus any subagent template test), ruff.

---

## Acceptance by the orchestrator

For each package:
1. Read the full diff and confirm that only owned files changed.
2. Re-run that package's validation commands.
3. Check against the reference session (reload the WebUI) for A, B, C and D.
4. Record PASS or the findings in this file's status table below.

| Package | Status | Notes |
|---|---|---|
| P0 | PASS | 113 tests pass; committed in `a813dbf01`. |
| C | PASS 2026-11-27 | Child history rows carry `traces` from `replay_transcript_to_ui_messages`, calls matched by `tool_call_id` across pages; response adds `assignment` + `duration_ms`; overlay pins the assignment. Reference child: 124/124 rows matched ("Read file …", "Searched files"). Orchestrator re-run: 42 py + 41 TS tests pass; basedpyright 962→962, 0 new. Pre-existing unrelated failures: `test_claude_child_approval_and_stop_survive_parent_reconnect[approve\|deny]` (HTTP 401 on unchanged code too). |
| E | PASS 2026-11-27 | `native.rs` only: `classify_link` (Allow / OpenExternal / Deny), macOS on_navigation + on_new_window route safe http(s) to the system browser; chat URLs carrying `bootstrapSecret` never leave the webview. cargo test 21/21. Manual macOS click test pending. MACOS.md updated. |
| A | PASS 2026-11-27 | Optional `spawn_call_id` in child record, event, wire, transcript and TS types (old records read as none). Child renders right after its spawning tool row inside the activity group; legacy records land after their turn's last work row, before the final answer. Reference session: child moved from below a later turn's answer into its own turn. Orchestrator re-run: 47 py + 39 TS tests pass, ruff clean; basedpyright 962→962. Extra approved edits: `loop.py` (`_publish_delegated_child`), `ws_http.py` (`_hydrated_child_payload`), `ThreadMessages.tsx` (2 props), `tests/webui/test_managed_display_origin.py` stub. |
| B | PASS 2026-11-27 | Short titles, partial status, run metrics, and a separate Markdown result disclosure. See "Package B verification" below. |
| D | not started | |
| F | not started | |

## Package B verification

- Optional `label` (60 characters) and a first-sentence, workspace-relative fallback are persisted at admission.
  The row and history header also shorten old labels without rewriting stored data.
- `outcome="partial"` keeps the lifecycle state `completed` and remains eligible for follow-up. Iteration-limit
  failures remain failures. The result preview keeps the final reply's Markdown structure and is shown on demand.
- Settled run duration and tool-result count are optional display facts, carried through live events, replay, and
  hydration. Zero is valid; live and old records without totals show no metrics. A new run cannot inherit totals
  from an earlier one. The history action and Markdown result disclosure are separate controls.
- The 35 package-specific Python tests are green; 78 tests passed in the focused Python rerun. The full WebUI
  suite passed 2,002 tests and the production build succeeded. Ruff passed. The full-source BasedPyright diagnostic
  comparison against `38e587efb` found 962 → 962 diagnostics across 547 files, with zero new signatures.
- The broader Python selection passed 373 tests and skipped 5. Its only two failures, the gateway reconnect
  tests for mobile approve/deny, fail with HTTP 401 on macOS. Both were reproduced from an unchanged export
  of `38e587efb`. They were neither skipped in code nor fixed by B.
- The reference child was checked read-only: its stored legacy outcome is still "Completed" and it has no
  persisted run totals. Its 160-character stored title projects to 60 characters. New partial rows and the
  Markdown disclosure were verified with WebUI component tests; no live desktop app check was run.

Next: D and F (wave 4), as separate work from B.

## Follow-ups found during review

- **E1 (desktop OAuth popups):** under the desktop shell `window.open` now opens the system browser but
  still returns `null`. `openMcpOAuthPopup` (`webui/src/components/settings/system/createSystemSettingsActions.ts`)
  then reports "popup blocked", and its Continue link can open a second tab. Treat a `null` return
  inside the desktop shell as "opened externally" (or expose a `desktop.open_url` command and call it
  directly). Provider OAuth in `SettingsPage.tsx` already ignores the return value.
- **E2:** Linux build of `native.rs` not compiled in review (only the aarch64-apple-darwin target was available).
- **C1 (grouping parity):** child history keeps each tool result as its own step; the main stream merges
  consecutive calls ("Searched files · 3 searches"). Needs segment ids on child rows.
- **C2:** `activity-message-model.ts` still falls back to `content` when merging trace rows (only affects
  rows from older servers without `traces`).
- **C3:** child duration falls back to the child record's `created_at`/`updated_at` when events have no
  timestamps (the reference session has none). A multi-run child shows its whole lifetime.
- **C4:** the "Assignment" overlay label has no locale entries yet.
- **A1:** `spawn_call_id` is missing (turn fallback used) for non-streaming turns with role tools that have no
  durable intent (e.g. `explore`), likely including nested non-streaming child turns.
- The permission/tool-dispatch workstream was moved to `codex/remove-managed-tool-allowlist`; it is not
  included in this feature branch.

