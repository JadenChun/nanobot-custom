# Session-load simplification: work plan status

Codex-shaped: bounded server read + renderer-side cache. No server-side cache.

## Baseline

| Suite | Result |
|---|---|
| `tests/webui` | 739 passed, 1 pre-existing failure (`test_child_approval_ingress::test_streaming_sibling_controls_do_not_touch_parent_and_stop_remains_responsive`) |
| Web UI (`vitest`) | 1764 passed |

The single Python failure fails identically with these changes stashed, so it is
pre-existing and not a regression signal.

## WI-1 — Backward bounded read: DONE

`nanobot/webui/transcript.py`

- `_group_transcript_turn_lines` groups raw lines into turns **without parsing**.
  Safe because `_record_json_line` writes compact separators, so an unescaped
  `"turn_end"` can only be a structural key, never string content.
- `_grouped_turn_stats` derives turn/user counts from those groups; the previous
  full parse of the active chunk is gone.
- `_chunk_turn` parses only the turns a page selects.
- `_active_chunk_groups` reads the file once per page and shares the groups.
- Removed the superseded `_scan_transcript_turn_stats`.

**Correctness:** A/B against the pre-change module, 4 sessions × 3 page sizes
(40 / 160 / unlimited) = 12 comparisons, **0 payload mismatches**.

**Performance:** neutral in practice (12→17 ms small session, 281→322 ms largest).
The profile shows the remaining cost is `replay_transcript_to_ui_messages`
(235 ms of 279 ms), dominated by `json.dumps` inside `_bounded_tool_event`, not by
the file read. WI-1 is structural groundwork, not a visible win on its own.

## WI-2 — Gate the delegated-children fetch: DONE

`webui/src/hooks/useSessions.ts`

Fetch runs only when the rendered page contains a `delegated_child` row.
Verified from `ws_http.py:1938` that the thread payload already carries
`stop_target`, so the child page is a fallback and Stop is unaffected.

**Tests:** `useSessions` 46 (one new: childless page issues no request);
full Web UI 1764 passing. Deployed bundle verified identical in repo and venv.

## WI-3 — Turn totals and whole-session index: DONE (revived on a real need)

Originally rejected because `total_known_message_count` had no consumer. It has
one now: a navigator that lists every turn beside the chat needs the whole
session, not the rendered page.

**Server** (`nanobot/webui/transcript.py`, `nanobot/webui/ws_http.py`):

- `build_webui_thread_response` page gained three fields, all free — they come
  from the chunk refs already computed for paging: `first_turn_ordinal`,
  `loaded_turn_count`, `total_turn_count`.
- `session_turn_index()` returns every turn as `{ordinal, label, created_at,
  cursor}`. It parses **only user rows** and groups raw lines, so no record
  replay. The per-turn `cursor` is the page cursor whose window ends at that
  turn, so a jump needs no knowledge of the cursor encoding.
- `GET /api/sessions/<key>/turns?limit=&before=` serves it, paged backwards.

Measured cost of the index on the largest staging session: user-row parse over
groups, no replay. The earlier 20–835 ms figure was for an exact *message*
total by replaying all history; the turn total costs nothing.

**Client:**

- `SessionTurnRail` (`webui/src/components/thread/SessionTurnRail.tsx`) — every
  turn placed by position in the session; the loaded window is emphasised.
  Mounted only while the rendered window covers less than the whole session, so
  `PromptRail` keeps owning the on-screen turns.
- `useSessionHistory` gained `turnIndex`, `totalTurnCount`, `loadedTurnRange`,
  `loadTurnIndex()`, `loadTurnAt(cursor)`.
- `fetchSessionTurns` in `lib/api.ts`; types in `lib/types.ts`.

**Tests:** `tests/webui/test_session_turn_index.py` (6 — totals, labels, paging,
per-turn cursor jump, HTTP route, invalid query); `session-turn-rail.test.tsx`
(3). Python `tests/webui` **745 passed**; Web UI **1768 passed**.

## WI-4 — Persist replay identities: NOT NEEDED

Measured on all four real sessions: **0 records** lack
`_WEBUI_REPLAY_IDENTITY_KEY` on a page, and `_ensure_replay_identities` costs
0.0 ms. The SHA-256 path only applies to legacy rows that no longer occur.

## WI-5 — Single migration check: NOT NEEDED

`_migrate_legacy_transcript` runs 7× per page build at 0.034 ms per call
(3.4 ms per 100 calls) — 0.24 ms total. Not worth the churn.

## WI-6 — Renderer page cache: DONE

Implemented the Codex property that matters for a revisit — **the renderer holds
the loaded window, so returning to a session paints without a round trip** —
without re-keying every row.

Codex's `entitiesByKey` + `islands` exists because its storage unit is a turn and
the same turn arrives from three places (page load, scroll-back, stream). Our UI
rows are not turn-shaped: `compaction`, `delegated_child` and fork markers are
not part of a turn, and `useSessions` already merges pages through
`mergeLatestHistory`. Re-keying would add a normalisation layer over rows that
have no turn identity, which is structure without a reader.

What was added instead (`webui/src/lib/session-page-cache.ts`):

- a bounded (8-entry, LRU) session-keyed store of the **settled** window:
  `messages`, `beforeCursor`, `hasMoreBefore`, `userMessageOffset`,
  `completedTurnIds`, `forkBoundaryMessageCount`, `version`;
- a revisit seeds state from it, so the transcript paints immediately while the
  network refresh still runs;
- the seed sets `lineage: 0` deliberately, so the refresh is still treated as the
  initial page for that mount and the merge keeps any scroll-back prefix;
- the cache is written from a settled-state effect (not mid-fetch), so a merged
  prefix is part of what a revisit paints, and loading/error states are skipped;
- only committed pages are cached. The live turn stays owned by the stream hook,
  so a stale in-flight snapshot can never win after a reconnect.

**Tests:** `useSessions` 47 (new: "paints a previously loaded session from the
renderer cache before the revisit fetch settles"); full Web UI **1765 passing**.
The cache is module-level, so `src/tests/setup.ts` resets it per test.

## WI-7 — Repair scoping: DONE

Not dead code. Verified across 5 sessions: `bde81382` **is** repaired correctly
(needs backfill → False after injection); `48f5d7fa` was **unsatisfiable**
(turn 0 is a legacy continuation with no prompt row, so it stayed True forever).

**Fix:** `_needs_user_event_backfill(lines, *, skip_leading_turn=False)` skips the
leading continuation turn, and the caller passes `skip_leading_turn=True` when
`page["has_more_before"]` is set (the page did not start at turn 0).

**Verified effect (loader call counts):**

| Session | Loader calls before | after |
|---|---|---|
| `48f5d7fa` (unsatisfiable) | 1 per load (full session read) | **0** |
| `bde81382` (genuinely repairable) | 1 | **1** (repair preserved) |

**Correctness:** 0 payload mismatches of 12 A/B comparisons.
**Tests:** `tests/webui` 739 passed, 1 pre-existing failure.

## WI-8 — Remove read-path manifest write: NOT TRIGGERED

`_rebuild_segment_manifest` never fired during reads on any of 4 sessions
(0 rebuilds). The concern is real in principle (a GET can rewrite segments) but
there is no observed occurrence to fix.

## WI-9 — Single storage: DEFERRED

See `session-storage-duplication.md`. 241 MB display data vs 5.4 MB canonical in
the staging workspace; 12 production modules; 3,790-line module. Own project.

## Rule followed throughout

Every change is verified by A/B payload hash against the pre-change module, not
by timing alone. A mismatch means the change is wrong, regardless of speed.

## Final verification (2026-11-20 03:14)

| Check | Result |
|---|---|
| `tests/webui` (Python) | 739 passed, 1 pre-existing failure (`test_child_approval_ingress::test_streaming_sibling_controls_do_not_touch_parent_and_stop_remains_responsive`) |
| Web UI (`vitest`) | **1765 passed**, 110 files |
| Payload A/B vs pre-change module | **0 mismatches of 12** (4 sessions × 3 page sizes) |
| `ruff` | clean |
| Deployed venv module | `skip_leading_turn` present; page builds (52 msgs) |
| Deployed bundle = repo bundle | `index-B8vmfCR3.js` in both |
| Full-session loader calls for `48f5d7fa` | **0** (was 1 per load) |
| `delegated-children` requests since restart | **0** (session has no child rows) |
| Live page totals | `first_turn_ordinal=23`, `loaded_turn_count=7`, `total_turn_count=30` |
| Live `/turns` index | 30 turns, first label read, every entry carries a cursor |
| Deployed bundle = repo bundle | `index-HgHCnNgH.js` in both |

## Incidents recorded

1. **Stray stash pop.** A `git stash push` failed with `could not write index`
   and the following `git stash pop` applied an unrelated pre-existing stash
   (`docs/mobile-remote-request-plan`), leaving four unmerged paths
   (`docs/README.md`, `docs/nanobot-remaining-work.md`,
   `nanobot/webui/file_preview.py`, `nanobot/webui/ws_http.py`). Recovered by
   restoring those four to `HEAD`; the stash entry is still in `git stash list`,
   nothing was lost, and no conflict markers remain in `nanobot/` or `webui/src/`.
   Lesson: never run `git stash pop` without `--index`-free, path-scoped
   verification that the top entry is the one just pushed.

2. **Reverted micro-optimisation.** A length-only JSON encoder added to
   `_bounded_tool_event` produced 8 wrong values out of 16 cases. It was caught
   by the A/B payload hash check and reverted. `_bounded_tool_event` remains the
   largest single cost of a page projection (≈207 ms of 279 ms on the largest
   session) and is a candidate for a separate, carefully verified change.

3. **Untracked scratch harness.** `webui/src/composer-preview.tsx`,
   `webui/preview.html` and `webui/preview-frame.html` are standalone preview
   files created during this session. They are not referenced by the app, but
   `composer-preview.tsx` fails the repo-wide `tsc` check. Build and tests are
   unaffected; decide whether to keep or remove them.
