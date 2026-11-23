# Kohaku chat page: implementation work plan

**2026-11-20 · `codex/chat-page-work-plan`** · Based on
`a895ac9ba7840c9012570c3706bef6b55a671a43` from
`origin/codex/staging-audit-implementation`.
The [design and release checks](gui-session-history-design.md) remain the authority.
This branch does not include the separate macOS terminal changes.

## Resume from this branch

This is a **handoff checkpoint**, not a merge or install candidate. On another
device, fetch and check out `origin/codex/chat-page-work-plan` (or fast-forward
its local tracking branch). Keep this branch separate from staging.

- **Start at G2, not the GUI rail.** Finish the writer audit below, then
  prove one committed range/index projection at the existing display commit
  boundary. Compare the authenticated latest read on real short and long
  unsegmented and rotated histories, with and without an index. A missing
  proof or oversized recent record is not an empty chat. Only after that,
  wire bounded history and exact prompt windows into the gateway and renderer.
- **Reconcile the mobile registry before qualifying prompt navigation.**
  `origin/codex/staging-audit-implementation` advanced to `17f1b6cfc` with
  `/turns` admission on the mobile bridge. That commit is not in this
  checkpoint; review it against the new prompt-index contract before
  integration. The split history and controls routes are already covered
  by the paired transport's existing `chat.read` checks.
- **Reproduce on the new device.** Use the repo's strict dependency setup
  in [CONTRIBUTING.md](../../CONTRIBUTING.md), then run:

  ```sh
  uv run --no-sync pytest tests/session/test_saved_source_browse.py \
    tests/webui/test_managed_display_origin.py tests/webui/test_session_turn_index.py \
    tests/webui/test_input_attempt_readback.py tests/webui/test_permission_v2_api.py \
    --basetemp="$(mktemp -d /tmp/khp.XXXXXX)"
  (cd webui && bun install --frozen-lockfile && bun run test --maxWorkers=4 --minWorkers=1)
  (cd mobile/native-network && go test ./internal/broker)
  ```

  The tests and summarized evidence are in Git. The development Mac's
  dependency copy, raw logs, and disposable probes below are local-only;
  they are not prerequisites to resuming work.

## Current checkpoint: split reads and composer action ownership

**Partial implementation; not ready to merge or install as the completed
refactor.** The managed source owner has a browse-only metadata read, and
new managed turns can bind display records to their producing source. Neither
is wired into the history page yet: its reader still needs a committed
coverage witness. In the running-gateway and renderer paths, a saved
page no longer requires the permission, folder request, or Stop projection.
On the original source, an injected permission-projection failure returned
HTTP 503 on desktop and 502 on the mobile adapter instead of the two saved
rows. The candidate serves those rows even when the permission reader fails
or the Stop reader would fail if invoked. The same endpoint refuses a
deleted managed session even if its display transcript remains on disk.

- `GET /api/sessions/{key}/webui-thread?view=history` validates the read
  principal, session source or existing legacy read path, and child lineage
  before replaying saved rows. Persisted workspace context resolves relative
  media paths but does not restore Full access; the existing media policy
  still authorizes each signed path. The response does not project
  permissions, folder requests, or Stop. The existing turn-tail hints and
  committed completion IDs remain for fenced reconciliation; they do not
  authorize execution or prove admission. The unqualified combined response
  remains for the TUI and existing callers.
- `?view=controls` uses the same operation and existing live owners, without
  replaying the display transcript. The GUI publishes the history window
  without awaiting it, uses it for explicit permission/folder-request
  readback and Stop refresh, and keeps the existing socket live events.
  Missing controls do not provide a legacy Stop fallback on the split path.
  A transport failure on refresh leaves the same client's settled rows visible.
  An authenticated read refusal clears the rows and cache. An in-flight prompt
  index request cannot restore labels after a switch or refusal. Both views
  pass through the existing mobile operation and its `chat.read` admission and
  delivery checks.
- Simultaneous renderer listeners now share their first socket attach for a
  chat. A canonical attach reuses its validated source for descriptive
  handshake metadata instead of loading it separately. Tests cover ordinary
  attach and exact input-attempt proof: each has one source read and one
  remaining metadata read for the live hydrator, not two metadata reads.
  Reconnect, explicit proof, Stop projection, and mobile delivery checks
  remain in place. Revisits can still reattach.
- The selected chat's editor can accept a draft before the sidebar and live
  controls settle. Submission and Stop have separate guards. The shared model
  catalog comes from the app, not a second chat-page fetch. A send that never
  hands an attempt to the transport leaves the draft intact. Editing or
  sending one queue row preserves the other rows and a separate draft.
  Empty Enter no longer steers;
  the deliberate Steer action requires the exact active turn. Queue drain
  requires a live natural completion, not an idle snapshot, Stop, or replay.
  Late camera and dictation results cannot enter a different selected draft.
  These are action/ownership fixes, not end-to-end draft restore or next-turn
  settings.
- Page and turn-index reads now use structural transcript events, including
  across real segment rotation. A reply that says `turn_end` no longer changes
  the turn count or page boundaries. No storage format, writer, or startup
  ordering changed.

### G2 preflight: why the latest window is not bounded yet

An isolated managed gateway with current, pre-index history returned four
visible rows for every fixture. Counts are for the authenticated
`view=history&limit=4&direction=latest` request, not elapsed times.

| Fixture | Settled turns | Canonical events loaded | Display records decoded | Segment entries decoded across manifest reads |
| --- | ---: | ---: | ---: | ---: |
| Single active file | 24 | 48 | 72 | 0 |
| Single active file | 240 | 480 | 720 | 0 |
| Rotated (5 segments) | 24 | 48 | 12 | 10 |
| Rotated (59 segments) | 240 | 480 | 12 | 118 |

The managed resolver also read the full canonical file twice before the
snapshot load. Rotation bounds display decoding in this fixture, not
canonical validation or manifest work. A title-only edit replaced the
canonical file and shifted the first event offset by 46 bytes while leaving
the source incarnation, generation, and event IDs unchanged. Public prompt
locators must therefore outlive ordinary appends and metadata edits; physical
ranges must be checked against the exact published file/segment version.

**G2 remains open.** Prove a bounded browse-authority read at the existing
session owner and a committed, rebuildable range/index projection before
changing the page contract. Do not treat a transcript, directory entry, or
unverified first line as read authority. G2 also needs a safe bounded tail
path for pre-index chats and interrupted-publication/replacement tests. The
existing scan is not accepted as a first-open solution merely because the
controls are independent.

### G2 checkpoint: shared read guard and bounded-tail probe

The desktop prompt-summary route used only the API principal and display
file. In a managed gateway it returned labels for an absent session, a
deleted session, a non-WebUI source, and an active record whose source file
was missing. It now uses the existing saved-thread source check before
reading labels. The summary read stays off-thread and uses the existing
sessions-file lock across that check and the display read. Valid summaries
still work without consulting live permission or Stop state. The legacy
combined thread response and the mobile operation registry are unchanged.
This fixes the missing check; it does not make the check bounded.

An isolated probe reused evidence from a **prior full validation** of each
exact canonical file. It rechecked the managed namespace, mode record,
journal, and private file identity. With that evidence available, it read no
canonical body on the browse. Across both 24- and 240-turn fixtures, the
no-index display tail read **2,048 bytes and decoded seven records** to
reproduce four visible rows. The rotated fixture used at most **576 bytes**
of the current v3 manifest to find the active file; it did not derive totals
or read segment entries. Faults immediately before and after real manifest
publication selected the old and new active file respectively, without
repairing the files. An oversized recent record produced an explicit
unavailable result, not an empty page. A metadata-only source replacement
rejected the old file proof while preserving logical origin.

### G2 checkpoint: managed browse source proof

The [session owner](../../nanobot/session/session_mode.py) now reads detached
history metadata from proof of the exact private source file. The existing
acquisition inventory stamps that proof when it fully validates the registered
source. Both existing atomic source writers stamp the file they installed,
not the file that happens to be at its path later; they return the stamp
only after directory sync, and the owner publishes it after media custody
settles. No extra source scan, persistent field, or write path was added.

The browse read rechecks the pinned namespace, mode record, journal
(including unfinished reset/delete), absence of a canonical sidecar, and the
current private file version. A missing or changed proof is unavailable,
not an ordinary mode-cache hit or a reason to trust the current header.
Execution and recovery retain full validation. On 24- and 240-prompt
fixtures, first browse after acquisition and immediately after a metadata
rewrite reads **zero canonical body bytes**. The inode changes on metadata
rewrite without changing the logical origin; reset and delete/recreation
change the origin. The mode check still enumerates the bounded live journal
set. This is source-owner evidence, not a first-open gateway or startup
latency claim.

**Why the browse proof is not enough.** In the prior gateway probe a
deleted session with a retained display file returned 404,
but recreating that key returned the four **old** rows, even with a freshly
validated canonical source. A valid filename, current source, or byte range
cannot authorize those rows. Ingress can also write a user display row
before the first canonical admission, and a delayed output can arrive after
replacement. Use the originating turn's lifetime at publication, not the
current key lookup at delivery time. Verify the first-admission, fork, and
reset/recovery cases before adopting any unbound prefix. The existing ingress
source token can identify an already-bound turn; it is not proof that every
writer used that lifetime or that any display row was admitted.

### G2 checkpoint: managed display publication

- An ordinary managed WebUI input now archives from the existing canonical
  admission receipt, with the exact committed prompt event ID. A canonical
  append failure leaves no user display row. The normal answer and completion
  carry the admitted turn's origin; a delayed receipt or answer cannot append
  to the display history of a reset or recreated chat. A title rewrite does
  not invalidate the origin, and a qualified read after reopening the source
  still excludes retained rows from its previous lifetime.
- The real stream path closes and archives the previous answer segment
  before publishing a committed Steer receipt. The runner still consumes
  guidance at the same canonical boundary. A failed guidance display write
  marks the active turn's completion incomplete, not a new turn for the
  guidance request.
- Managed forks use the committed fork as fallback when the parent's display
  prefix is unbound, cannot match prompt positions, or has a marked or
  interrupted display tail. Proven parent display rows and the fork marker
  are published under the fork's own origin. A fork no longer loses a
  committed answer just because the matching prompt had already been archived.

**G2 still open.** These checks prove provenance for selected writers, not
complete display coverage, legacy compatibility, or a bounded history read.
The existing history and prompt-summary endpoints still use the compatibility
reader; the retained-file read after key recreation is not resolved by
stamping new rows alone. Audit the remaining publishers (including inline
commands, delegated activity, and recovery), then publish coverage in the
one rebuildable index at the existing commit boundary. Do not add a second
content writer or adopt an unbound display file after reset/recreation. The
probe's manifest suffix check is not a production commit witness. Prove a
bounded pre-index latest read and exact older/newer/around windows; a bounded
tail alone cannot infer global counts or direct prompt locations.

The next writer audit has three known gaps. [Inline commands and shell
input](../../nanobot/webui/inbound_commands.py) can still archive at ingress.
[Cross-session input](../../nanobot/session/webui_turns.py) archives on
`UserInputAccepted`, which the [agent loop](../../nanobot/agent/loop.py)
publishes before it dispatches or queues the input, not after a canonical
append. [Delegated activity](../../nanobot/webui/outbound_projection.py)
already has a typed root lifetime but is archived without passing a display
origin to the recorder. Classify those records and interrupted/recovery
output before using a covered range as a complete view. No reader, index,
or legacy-adoption change was made at this checkpoint.

### G2 closure & Package 2 gateway checkpoint

- **Writer audit closed:**
  - Managed WebUI ingress ([inbound_commands.py](../../nanobot/webui/inbound_commands.py))
    no longer archives `!shell` or `/command` rows at ingress. Idle managed
    commands commit through `_dispatch_source_command` in
    [loop.py](../../nanobot/agent/loop.py), passing canonical `source_event_id`
    into `input_committed` (preserving `!cmd` display text in
    `CommittedUserInput`) and carrying `WEBUI_DISPLAY_SOURCE_KEY` on the
    assistant reply and `turn_end`. `/new` and busy inline commands stay
    live-only and no longer append unbound rows.
  - Managed cross-session inputs no longer archive on `UserInputAccepted`
    ([webui_turns.py](../../nanobot/session/webui_turns.py)). Both new-turn
    commits (`TurnDelivery.input_committed`) and mid-turn injections
    (`_publish_injection_committed`, ordered after open stream segments) now
    archive after canonical append via `UserInputEvent.input` with
    `_webui_source_origin` and `_webui_source_prompt_id`, and `turn_end`
    preserves `WEBUI_DISPLAY_SOURCE_KEY`.
  - Delegated activity ([loop.py](../../nanobot/agent/loop.py),
    [outbound_projection.py](../../nanobot/webui/outbound_projection.py)) now
    binds archival to the typed root lifetime (`root_source` / root fields on
    `DelegatedChildEvent`) and drops delayed updates after root reset/delete.
  - `WebUITranscriptRecorder.append` ([transcript.py](../../nanobot/webui/transcript.py))
    now refuses to write unbound rows for non-legacy sessions under managed
    authority; `TurnDeliveryFactory.unrouted` also binds `admitted_source` on
    early failures.
- **Committed range/index projection & bounded pre-index tail:**
  - Added [transcript_index.py](../../nanobot/webui/transcript_index.py):
    rebuildable `head.json` + 64-turn paged directory (`pages/NNNNNN.json`)
    bound to exact `(origin, active_file, st_ino, st_dev, active_covered_bytes,
    manifest stat)`. Maintained O(1) at append, O(active_file) at rotation,
    O(delta) on interrupted-index catchup, and deleted on transcript deletion.
  - Serves `latest`, `before`, `after`, and exact `around(prompt_locator)`
    windows via coalesced byte-range reads with line-boundary and origin
    verification, plus O(log P) `read_prompt_summary_range` without opening
    JSONL chunks. Pre-index histories use `read_preindex_latest_tail` (bounded
    reverse tail read, returning explicit `oversized_tail` instead of empty
    when a recent record exceeds the tail budget, and rejecting retained
    prior-lifetime rows after reset/recreation).
- **Gateway & edge wiring (Package 2 server side):**
  - `GET /api/sessions/{key}/webui-thread?view=history` and
    `GET /api/sessions/{key}/turns` ([ws_http.py](../../nanobot/webui/ws_http.py),
    [workspaces.py](../../nanobot/webui/workspaces.py)) now use
    `read_saved_source_metadata` (0 canonical body loads) and the bounded index,
    supporting `direction=latest|before|after|around`, `after`, `around`, and
    `prompt_id`, returning HTTP 409 on stale cursors/locators and per-prompt
    `prompts` summaries alongside backwards-compatible `turns`.
  - [edge.py](../../nanobot/remote/edge.py) admits `after`, `around`, and
    `prompt_id` on the existing `chat.read` thread route. Go native broker
    allowlist (`mobile/native-network/internal/broker/policy.go`) and
    reconciliation of `17f1b6cf` (`/turns` mobile registry) remain open until
    tested with Go.

### Package 3 checkpoint: viewport, prompt rail, navigator, and reconnect

- **Exact prompt identity & wire contract:**
  - [transcript.py](../../nanobot/webui/transcript.py) now emits `promptId` on
    replayed user `UIMessage` rows matching `transcript_index._prompt_id_for_record`
    (canonical `_webui_source_prompt_id` or `legacy:{turn_ordinal}:{record_index}`),
    and validates `promptId` in retained display threads.
  - [types.ts](../../webui/src/lib/types.ts), [api.ts](../../webui/src/lib/api.ts),
    and [session-page-cache.ts](../../webui/src/lib/session-page-cache.ts)
    now carry `promptId`, `SessionPromptSummary`, `PendingPromptReveal`,
    `after_cursor`/`has_more_after`/`total_prompt_count`/`coverage`/`target_prompt_id`,
    forward `direction=before|after|around`, `after`, `around`, and `prompt_id`,
    and surface HTTP `409` (`stale_history_cursor`) and `404` (`prompt_not_found`)
    explicitly while preserving `null` for plain missing-session 404s.
- **History hook (`useSessionHistory`) & live-tail isolation:**
  - [useSessions.ts](../../webui/src/hooks/useSessions.ts) fixes the prior bug
    that always returned `turnIndex: []` and `totalTurnCount: 0`, switches
    `loadTurnIndex` to a single bounded page (`limit=100`) with on-demand
    `loadMorePromptIndex`, adds `loadNewer` (`direction=after`) with boundary
    deduplication, adds `revealPrompt` (`direction=around`) keeping the old view
    intact while fetching and cancelling superseded reveals, adds `goToLatest`,
    and isolates background refreshes while reading an older window
    (`hasMoreAfter === true`): same-lifetime updates set `tailActivityPending`
    without stealing scroll, while source incarnation/generation replacement
    invalidates stale windows, cursors, and prompt summaries.
- **Prompt rail, navigator drawer, and viewport:**
  - [PromptRail.tsx](../../webui/src/components/thread/PromptRail.tsx),
    [PromptNavigator.tsx](../../webui/src/components/thread/PromptNavigator.tsx),
    [promptNavigation.ts](../../webui/src/components/thread/promptNavigation.ts),
    [ThreadMessages.tsx](../../webui/src/components/thread/ThreadMessages.tsx),
    [ThreadViewport.tsx](../../webui/src/components/thread/ThreadViewport.tsx),
    and [ThreadShell.tsx](../../webui/src/components/thread/ThreadShell.tsx)
    keep the rail visible on short latest windows of long chats, key unloaded
    markers by stable prompt identity, bucket dense ranges (`>= 30`) into
    bounded DOM markers while exposing every individual prompt (and on-demand
    older summary loading) in `PromptNavigator`, execute fetch → mount → scroll
    + temporary highlight (`data-prompt-highlighted="true"`) to the exact
    `data-prompt-key` row (including multiple user prompts inside one Steer
    turn), surface non-destructive reveal error banners with Retry / Go to
    latest, trigger forward paging near the bottom of older windows, and keep
    live streaming isolated at the tail until the reader returns via Go to
    latest or sends a new message.
- **Verification:**
  - Python: `basedpyright --outputjson` + `scripts/check_type_regressions.py`
    -> `PASS_NO_NEW_DIAGNOSTICS` (`764 -> 764`, `introduced: 0`);
    `ruff check nanobot/` -> `All checks passed!`; `git diff --check` clean;
    `pytest tests/webui/test_transcript_index.py` -> **12 passed**.
  - WebUI: `bun run lint` (`eslint --max-warnings 0`) -> pass;
    `bun run build` (`tsc -p tsconfig.build.json && vite build`) -> pass;
    `bun run test -- --maxWorkers=4 --minWorkers=1` -> **110 test files passed,
    1,792 tests passed**, including the new integrated `ThreadShell` reveal +
    live-tail isolation test.

### Package 4A (Composer drafts/focus/usage) & Gate G3 (Backend bounded settings + next-turn split) checkpoint (`c63a88a1`)

- **Package 4A — Composer drafts, focus/caret, first-send handoff, and window-independent context usage:**
  - Added [composer-drafts.ts](../../webui/src/lib/composer-drafts.ts) and
    [context-usage-retention.ts](../../webui/src/lib/context-usage-retention.ts),
    wired through [ThreadComposer.tsx](../../webui/src/components/thread/ThreadComposer.tsx),
    [ThreadShell.tsx](../../webui/src/components/thread/ThreadShell.tsx),
    [useSessions.ts](../../webui/src/hooks/useSessions.ts), and
    [App.tsx](../../webui/src/App.tsx).
  - Preserves chat-scoped drafts (`text`, quotes, mentions, ready attachments,
    caret/selection, focus) synchronously across switches (`A -> B -> A`) and
    persists bounded device-local saved-chat drafts across relaunch (keyed by
    hashed auth namespace + session key + `incarnation:generation`, keeping
    temporary and new-chat drafts ephemeral, showing an explicit reattach notice
    for unrecoverable relaunch attachments, and invalidating on delete, `/new`
    generation bump, incarnation replacement, or revoked read access).
  - Preserves caret/focus on switch without stealing focus during background
    updates or popping touch keyboards, binds first-send handoff strictly to the
    created chat, and retains the latest qualified tail context usage while
    viewing an older history window (`hasMoreAfter === true`).
- **Gate G3 & Backend Next-Turn Preferences:**
  - Added [next_turn_preferences.py](../../nanobot/session/next_turn_preferences.py)
    storing detached, origin-bound `(store_identity, incarnation, generation)`
    next-turn choices (`workspace_scope`, `model_preset`, `reasoning_effort`,
    `permission_mode`) under `.session-prefs/` with atomic write + fsync, plus
    header-only atomic replacement (`replace_header_atomic`) in
    [manager.py](../../nanobot/session/manager.py) and
    [source_store.py](../../nanobot/session/source_store.py) so routine settings
    reads and writes are O(1) bounded independently of settled canonical history.
  - Separated next-turn preferences from the admitted active turn scope across
    [loop.py](../../nanobot/agent/loop.py),
    [workspaces.py](../../nanobot/webui/workspaces.py),
    [workspace_focus.py](../../nanobot/session/workspace_focus.py),
    [workspace_access_recovery.py](../../nanobot/session/workspace_access_recovery.py),
    [session_projection.py](../../nanobot/webui/session_projection.py), and
    [ws_http.py](../../nanobot/webui/ws_http.py): selecting folder/model B
    during run A leaves A's scope, Steer, Stop, recovery request, and `Once`
    grants intact; A's completion never overwrites B; next-turn admission
    retires `Once` grants, validates B without fallback to A, and promotes
    authorized B into confirmed session metadata.
  - Note: Package 4B frontend UI wiring (lifting the active-turn folder-picker
    lock and displaying/ordering next-turn choices in `ThreadComposer`/
    `WorkspaceControls`) is now unblocked and ready next.
- **Verification:**
  - WebUI: `bun run lint` (`--max-warnings 0`) -> pass; `bun run build` -> pass;
    `bun run test -- --maxWorkers=4 --minWorkers=1` -> **112 test files passed,
    1,813 tests passed** (including new `thread-composer-drafts.test.tsx` and
    `context-usage-retention.test.ts`).
  - Python: `basedpyright --outputjson` + `scripts/check_type_regressions.py`
    -> `PASS_NO_NEW_DIAGNOSTICS` (`764 -> 764`, `introduced: 0`);
    `ruff check nanobot/` -> `All checks passed!`; `git diff --check` clean;
    `pytest tests/session/ tests/security/test_workspace_access_recovery.py tests/webui/test_transcript_index.py tests/webui/test_managed_display_origin.py tests/webui/test_child_thread_route.py tests/webui/test_permission_v2_api.py -q` -> **2,115 passed** (including all 11 new tests in `tests/session/test_next_turn_preferences.py`).

### Boundary 1 follow-up: warm revisit coalescing and bounded settled startup scan (`4cf01820`)

- **Package 4B (Composer next-turn settings)** is committed separately at
  `72eeeb5d` with review cleanup `2725983d`: next-turn model, effort, and folder
  choices go through one correlated chat-scoped operation and apply immediately
  without locking the picker or editor during a run. The running turn keeps its
  admitted scope; only a new turn consumes the next-turn choice and retires
  `Once` grants.
- **Client attach reuse (same live socket):**
  - `NanobotClient.attach` now reuses an existing live subscription for
    `(socket, chat)` when the socket is still open and the source incarnation is
    unchanged. A -> B -> A on one connection no longer sends a duplicate full
    attach or replays the previous recovery/Stop projection.
  - A fresh attach is still forced for: a socket reconnect (attached map is
    cleared on close), an observed source replacement via
    `invalidateAttachment` (wired from `useSessions` when a lifetime change is
    detected), an unsettled run (`hasUnsettledRun`), or unsettled input-attempt
    proof. Notification subscriptions and per-chat listeners are untouched, so
    live delivery for active or followed chats continues.
- **Bounded settled startup scan:**
  - `RecoveryCoordinator.scan` now classifies each candidate from
    `SessionManager.read_startup_header_metadata`, which reads only the intact
    header of the default JSONL store and never a canonical source body.
    Unresolved sessions still go through `get_or_create` +
    `load_source_snapshot`, preserving the owner-fenced journal/checkpoint path.
  - `JsonlSessionStore.read_bounded_header_metadata` implements the bounded
    header read; a non-default store keeps its previous read path so unknown
    backends are never guessed.
  - `tests/session/test_recovery_startup_scan.py` proves a settled canonical
    session loads no body at startup and that a journal-marked session still
    performs full owner-validated recovery. No startup admission fence was
    weakened and new input still cannot race past recovery.
- **Verification:**
  - WebUI: `bun run lint` (`--max-warnings 0`) pass; `bun run build` pass;
    `bun run test -- --maxWorkers=4 --minWorkers=1` -> **113 test files, 1,832
    tests passed** (including new client warm-revisit/fresh-attach cases and the
    stream revisit test).
  - Python: `ruff check nanobot/` -> `All checks passed!`;
    `basedpyright` + `scripts/check_type_regressions.py` ->
    `PASS_NO_NEW_DIAGNOSTICS` (`764 -> 764`, `introduced: 0`);
    `pytest` on the recovery/follow-up/checked-read suites -> **89 passed**.

## Remaining implementation and acceptance

| Boundary | Work and exit check |
| --- | --- |
| **1 · Attach and readiness follow-up** | Session projection, gateway and client transport. Coalesce revisits on the same live socket/source without replaying stale recovery or Stop state, losing notification subscriptions, or skipping exact input-attempt proof. Separately qualify genuine cold startup: MCP connection and recovery precede channel startup in [gateway_runtime.py](../../nanobot/cli/gateway_runtime.py). Split readiness only if cold-start evidence warrants it. |
| **2 · Storage windows and prompt summaries** | Storage and gateway readers, after **G2**. Use real segmented and pre-index fixtures to bind ranges to committed generations. Prove bounded latest, older, newer, and exact-around reads and a bounded pre-index first open; reject obsolete ranges after replacement. Keep the writer unchanged. |
| **3 · Viewport and prompt rail** | History hook, viewport and mobile navigator, after 2. Prove unloaded prompt selection reaches the exact mounted row, both paging directions work without gaps, and navigation/reconnect cannot replace a newer selection or steal the reader's anchor. |
| **4 · Composer and next-turn settings** | Finish draft restoration on switch and relaunch, focus/caret, first-send payload handoff, and usage independent of the selected window. Keep edits to shared shell files in sequence with 3. Before changing settings, settle **G3**: bounded metadata readback/commit at the existing owner, instant selection, and exact new-turn admission without changing the active scope or granting authority on timeout. The active-workspace lock remains until that boundary is proven. |
| **5 · Transcript consolidation** | **Complete.** Bounded canonical event-offset reader (`canonical_event_index.py`) + reversible display-fact writer cutover (`_webui_text_ref` + `prompt_preview`) landed with 0 canonical body loads, zero bulk migration, and full mixed/pre-index read compatibility. See Package 5 checkpoint below. |
| **6 · Bundled apps** | Compare first open, prompt navigation and composer in Kohaku and Codex Desktop on macOS; smoke Linux and one paired Android device. Repeat affected checks after writer cutover. **Linux-executable qualification is complete** (see the Package 6 ledger). macOS, Android, and Go checks remain. |

### Phase C Gate G4: display parity and rollback audit

**Gate G4 result: G4 PASSED WITH EXPLICIT DISTINCT DISPLAY-FACT LIST.**
New write evidence is `tests/webui/test_g4_display_parity.py` (22 checks). No
production code changed in this stage. The audit compares the canonical
[source turn writer](../../nanobot/session/turn_writer.py) with
[WebUI transcript](../../nanobot/webui/transcript.py) replay,
`build_webui_thread_response`, and [transcript_index.py](../../nanobot/webui/transcript_index.py).
Package 5 can retire the duplicate full-history write, but it must keep every
fact marked "display-only" below. It must not reconstruct those facts from the
canonical source.

#### Field-by-field provenance matrix

"Canonical alone?" answers one question: can the chat page rebuild this fact
from the canonical session store without the display transcript? "Replay" means
`replay_transcript_to_ui_messages` plus `build_webui_thread_response`.

| # | Display fact | Canonical owner | Display owner | Reconstructible from canonical alone? | Package 5 action |
| --- | --- | --- | --- | --- | --- |
| 1 | Prompt text, answer text, turn identity, prompt identity, admission | `SourceTurnWriter.append_message` / `admit_input`; `_event_id` is the durable prompt identity | `user`, `stream_end`, `message` rows; `_webui_source_prompt_id`; `turn_id`; `completed_turn_ids` | **Partly.** Text and prompt identity are canonical. A settled **turn boundary** (`turn_end`), `turn_id`, and the completed-turn list are display facts. | Keep display prompt/answer rows bound to `(store_identity, incarnation, generation, turn_id, prompt_id)`. Keep the settled `turn_end` marker. Never treat a display row as admission proof. |
| 2 | Permitted reasoning detail; raw/private reasoning redaction | `_persisted_content` strips `thinking`/`redaction`/`reasoning_content`; `_append` drops `reasoning_content`/`thinking_blocks`/`_meta` | `reasoning_delta` / `reasoning_end` / `message(kind=reasoning)`; replayed `reasoning` field | **No.** Canonical source keeps no reasoning text on purpose. | Keep the permitted display reasoning summary only. Never persist raw provider reasoning. |
| 3a | Bounded tool traces (`traces`, `toolEvents`) | Assistant `tool_calls` survive canonically, but no per-call start/end result payload, phase, or error text is canonical | `message(kind=tool_hint/progress)` with `tool_events`; `_bounded_tool_event`; `tool_trace_lines_from_events` | **No.** The canonical tool event holds only role, name, call id, arguments, and result content. It has no `phase`, no redacted oversized preview, and no merged start/end ordering. | Keep bounded `toolEvents` and `traces`. Cap inline payloads and oversized fields at the existing limits. |
| 3b | Approval outcome | `TurnEndEvent.outcome`, delegated `pending_approval_id`/`state` | `turn_end.outcome`; `delegated_child.pending_approval_id`; replayed `delegatedChild` | **Partly.** Canonical `TurnEndEvent` outcome exists, but the refusal/waiting display row and its label are display-only. | Keep the delegated approval row and its `pending_approval_id`, `can_stop`, `owned_work`, `state`, `outcome`. |
| 3c | File-edit diffs (`fileEdits`) | Canonical filesystem effect facts (`operation`, `target`, `disposition`, hashes) are execution evidence, not per-edit added/removed display lines | `file_edit` rows; `upsert_file_edits`; `_filter_covered_file_edit_tool_events` | **No.** The canonical store has no `+/-` display summary and no call/path grouping for the page. | Keep `fileEdits` and the covered-tool-hint filtering. Do not merge file-edit facts into a canonical copy. |
| 4 | Context-compaction markers and turn usage | Canonical retains no compaction marker and no turn token usage; compaction is a runtime notification | `context_compaction` rows; `turn_end` `usage`, `round_usages`, `context_window_tokens`, `context_usage`; `_sanitize_turn_usage` | **No.** These are derived display facts and transport observations. | Keep them as bounded display facts. Sanitize usage on write and replay. |
| 5 | Fork and parent/child delegated lineage | Fork boundary comes from the canonical fork operation; child lifetime evidence is canonical metadata and events | `fork_marker`; `fork_boundary_message_count`; `delegated_child`; `_delegated_child_lifetime` | **Partly.** The fork exists canonically, but the display cut position and the collapsed one-row child lifecycle are display facts. | Keep `fork_marker` and the delegated child row. Keep the stale-lifetime and stale-sequence rejection. |
| 6 | Incomplete tails, interruption, and crash/restart recovery | Canonical answer text is durable; the incomplete tail marker is a display fact | `transcript_incomplete`; `_recover_incomplete_turns`; `_repair_unfinished_tail`; `_recover_interrupted_rotation` | **Partly.** Recovery can restore answer text from canon. The pending/incomplete display state is display-only. | Keep `transcript_incomplete` and the rotation intent file. Rotation recovery must stay replayable after any writer change. |
| 7 | Media, attachments, signed URLs, generated assets | Canonical user `media` paths survive; signed URLs are process-local | `user.media_paths`/`attachments`; `stream_end.media`; `_media_from_signed_urls`; `rewrite_local_markdown_images` | **No for URLs.** Canonical keeps paths, not signed URLs or asset display. | Keep authorized references and re-sign on read. Never persist a raw host path or a stale signed URL as durable truth. |
| 8 | Retention, deletion, and incarnation replacement | Canonical reset/delete and incarnation/generation are the authority | `delete_webui_transcript_result`; `delete_transcript_index`; origin stamping in `_record_origin` | **Yes for authority.** The lifecycle decision is canonical. | Bind every display fact to the origin. Delete index and segments on delete. Reject prior-lifetime rows and cursors. |
| 9 | Read compatibility and rollback | Canonical history is unchanged | Indexed windows, `read_preindex_latest_tail`, direct JSONL/segment read | **Yes.** Durable JSONL plus the segment manifest replay without the index. | Keep zero bulk migration. Keep the bounded pre-index tail and index rebuild. |

#### Live consumer inventory

Every non-test importer of [transcript.py](../../nanobot/webui/transcript.py) or
[transcript_index.py](../../nanobot/webui/transcript_index.py) under `nanobot/`,
found by import-graph scan, with the behavior each one depends on:

| Consumer | Imports | Depends on | G4 status |
| --- | --- | --- | --- |
| [ws_http.py](../../nanobot/webui/ws_http.py) | `HistoryWindowError`, `WEBUI_TRANSCRIPT_SCHEMA_VERSION`, `build_webui_thread_response`, `delete_webui_transcript_result`, `read_transcript_lines`, `session_turn_index`, `webui_transcript_signature` | Bounded history/turns windows, split delete result, prompt-summary ranges, display rows for file-preview references, generated-asset signature | Covered. Display facts needed: prompt/answer, turn identity, traces, compaction, usage, child rows, media refs. |
| [session_access.py](../../nanobot/webui/session_access.py) | `build_webui_thread_response`, `normalize_session_mentions_metadata` | The combined thread read for session search/tools | Covered. Uses the same replay projection. |
| [forking.py](../../nanobot/webui/forking.py) | `append_fork_marker`, `delete_webui_transcript`, `fork_transcript_before_user_index`, `write_managed_fork_transcript`, `write_session_messages_as_transcript` | Fork cut position and display prefix copy | Covered. Needs `fork_marker` and prompt-position identity. |
| [gateway_services.py](../../nanobot/webui/gateway_services.py) | `WebUITranscriptRecorder` | The managed display writer | Covered. Package 5 changes only what this writer persists, not the recorder's admission binding. |
| [generated_asset_index.py](../../nanobot/webui/generated_asset_index.py) | `media_publication`, `read_transcript_lines`, `webui_transcript_signature` | Asset references and media custody during writes | Covered. Needs `stream_end.media` / `media_paths` and the write signature. |
| [media_gateway.py](../../nanobot/webui/media_gateway.py) | `rewrite_local_markdown_images` | Signed media rewrite for replay | Covered. Paths stay canonical; URLs stay read-time. |
| [thread_disk.py](../../nanobot/webui/thread_disk.py) | `delete_webui_transcript` | Legacy JSON cleanup | Covered. No display fact. |
| [attachments/_inventory.py](../../nanobot/attachments/_inventory.py) | `validate_retained_transcript_record`, `validate_retained_webui_thread` | Fail-closed retention of display records | Covered. Package 5 must keep these validators satisfied. |
| [session/webui_turns.py](../../nanobot/session/webui_turns.py) | `append_session_message_input` | Cross-session display input after admission | Covered. Needs `session_message` display metadata. |
| [session/recovery.py](../../nanobot/session/recovery.py) | `has_unfinished_transcript_tail` | Startup recovery for an interrupted display tail | Covered. Needs `transcript_incomplete` / unfinished-tail detection. |
| [channels/websocket/runtime.py](../../nanobot/channels/websocket/runtime.py) | `WEBUI_TRANSCRIPT_INCOMPLETE_KEY` | Live incomplete marker | Covered. Needs the incomplete key. |

No live consumer reads a canonical body to rebuild a display fact. Each one
either reads a display fact directly or reads canonical state for authority.

#### Rollback and pre-index proof summary

All proofs run against `tests/webui/test_g4_display_parity.py` with `tmp_path`:

- **Index delete and rebuild:** for a 12-turn chat with a two-prompt Steer turn,
  `delete_transcript_index` then `rebuild_transcript_index` reproduces identical
  `UIMessage` rows, `promptId`s, `before`/`after`/`around` cursors, prompt
  summaries, and `/turns` output.
- **Pre-index chat:** `read_preindex_latest_tail` and
  `build_webui_thread_response` serve a bounded `tail_only` window with explicit
  `has_more_before`, no total-count claim, and no index creation. There is no
  bulk migration.
- **Rollback without the index:** deleting only the index still replays the
  committed JSONL, segment files, and manifest. A rotated 7-turn fixture serves
  the bounded tail, then the rebuild restores the identical full projection.
- **Canonical-only chat:** a canonical turn with no display rows produces an
  explicit empty display read. The canonical text is never substituted for a
  display projection.
- **Live turn:** one real managed turn proves the display prompt row appears
  only after canonical admission, carries the committed canonical event id, and
  survives a restart plus index rebuild. A forced canonical append failure
  creates no display row.

#### Gate G4 verdict

**G4 PASSED WITH EXPLICIT DISTINCT DISPLAY-FACT LIST.**

The following are distinct display facts that Package 5 must keep durable under
`(store_identity, incarnation, generation)` and `(turn_id, prompt_id)`. They
cannot be rebuilt from the canonical source:

1. Settled turn boundaries and `turn_id` / completed-turn markers.
2. Permitted reasoning summary text.
3. Bounded `toolEvents` and `traces` (phase, redacted previews, merged order).
4. Approval display rows and delegated-child lifecycle rows.
5. `fileEdits` diffs and covered-tool-hint filtering.
6. Context-compaction markers and sanitized turn usage / context usage.
7. Fork markers and fork message cut positions.
8. `transcript_incomplete` markers and interrupted-rotation intent.
9. Authorized media references and generated-asset references.

The following are not display facts and must not be duplicated: raw or private
reasoning, full canonical model context, private tool payloads beyond the
bounded preview, and stale signed URLs. No live consumer lacks a replacement.
No blocker requires keeping the duplicate full write: Package 5 may proceed.

### Phase C Gate G4 verification

```sh
.venv/bin/pytest tests/webui/test_g4_display_parity.py tests/webui/test_transcript_index.py \
  tests/webui/test_managed_display_origin.py tests/webui/test_child_thread_route.py -q
.venv/bin/ruff check nanobot/ tests/webui/test_g4_display_parity.py
.venv/bin/basedpyright --outputjson > /tmp/overnight-basedpyright-after.json || true
.venv/bin/python scripts/check_type_regressions.py \
  --before /tmp/overnight-basedpyright-before.json \
  --after /tmp/overnight-basedpyright-after.json
git diff --check
```

Result: **79 passed**; Ruff `All checks passed!`; type regression check
`PASS_NO_NEW_DIAGNOSTICS` (`introduced: 0`); `git diff --check` clean. No
production code changed.

### Package 5: bounded canonical event-offset reader & reversible writer cutover

**Status: COMPLETE.** All three unblock conditions and cutover invariants are
implemented and verified with zero regressions.

**macOS storage check (2026-11-23).** Staging `8034271c9` reproduced three
failures because the [header replacement path](../../nanobot/session/manager.py)
called `os.copy_file_range` before taking its buffered fallback. With the fallback
used when the function is unavailable, both existing storage suites passed
(26 tests on macOS arm64, Python 3.14.2). An isolated browser run sent a message,
received a synthetic reply, and reopened the saved chat without an error row.
This verifies the affected flow, not the entire macOS release.

- **Bounded canonical event-offset index ([canonical_event_index.py](../../nanobot/session/canonical_event_index.py)):**
  - Stores a rebuildable per-event byte-offset map under
    `<sessions_dir>/.event-index/<storage_key>.json` (registered in
    `mode_storage.py`, `session_mode.py`, and `attachments/_inventory.py`),
    storing offsets relative to `body_offset` and bound to both
    `(store_identity, session_key, incarnation, generation)` and the physical
    `PrivateFileVersion` (device, inode, size, mtime_ns, uid, gid, mode, links,
    ctime_ns).
  - Maintained atomically on `JsonlSessionStore.persist_source_records`, updated
    in O(1) on header-only `replace_header_atomic` without rescanning events,
    deleted on `delete_source_session` and `SessionManager.delete_session`, and
    rebuilt lazily via streaming `os.pread` scan if missing or stale (without
    ever invoking `load_source_snapshot`).
  - `SessionManager.read_canonical_events_by_ids` /
    `SourceSessionStore.read_canonical_events_by_ids` open the canonical file,
    verify `PrivateFileVersion` and bounded line-0 header origin fail-closed,
    and read only the requested event lines via `os.pread` (**0 canonical body
    loads**).
- **Reversible consolidated display-fact writer & reader ([transcript.py](../../nanobot/webui/transcript.py), [ws_http.py](../../nanobot/webui/ws_http.py), [session_access.py](../../nanobot/webui/session_access.py), [runner.py](../../nanobot/agent/runner.py), [progress_hook.py](../../nanobot/agent/progress_hook.py), [channels/manager.py](../../nanobot/channels/manager.py)):**
  - For managed sessions, `WebUITranscriptRecorder.append` verifies via O(1)
    `read_canonical_events_by_ids` that a committed `"user"` prompt or
    non-merged `"stream_end"`/`"message"` answer matches the canonical event's
    content; when identical, it stores `_webui_text_ref` (plus bounded ≤120-char
    `prompt_preview` for `/turns` and `PromptRail`) and omits the duplicated
    full `"text"` body from the display store. When display copy intentionally
    differs (e.g. `!cmd` vs `/__shell cmd`), inline `"text"` is preserved
    losslessly.
  - All distinct G4 display facts (settled `turn_end`, permitted reasoning,
    bounded `toolEvents`/`traces`, approvals, `fileEdits`, compaction, usage,
    delegated child rows, fork markers, `transcript_incomplete`, authorized
    media refs) remain durable in the display store.
  - `build_webui_thread_response` hydrates `_webui_text_ref` rows in the
    selected bounded window via `canonical_events_loader` before replay;
    `write_managed_fork_transcript` materializes consolidated parent prefixes
    under the new fork origin; pre-cutover inline-text and pre-index transcripts
    continue reading unchanged with zero bulk migration.
  - Also registered rebuildable `<stem>.index` directories in
    [attachments/_inventory.py](../../nanobot/attachments/_inventory.py) so media
    retention maintenance stays complete alongside the transcript read index.
- **Verification:**
  - `tests/webui/test_package5_transcript_consolidation.py` -> **4 passed**
    (including 24- and 240-turn rotated mixed/consolidated HTTP windows with 0
    canonical body loads and dual-index delete/rebuild).
  - Combined focused suite (`test_package5_transcript_consolidation.py`,
    `test_g4_display_parity.py`, `test_package6_qualification.py`,
    `test_managed_display_origin.py`, `test_transcript_index.py`,
    `test_child_thread_route.py`, `test_recovery_startup_scan.py`,
    `test_next_turn_preferences.py`, `tests/attachments/`) -> **286 passed**.
  - `ruff check nanobot/ tests/webui/test_package5_transcript_consolidation.py`
    -> `All checks passed!`.
  - `basedpyright` + `scripts/check_type_regressions.py` ->
    `PASS_NO_NEW_DIAGNOSTICS` (`764 -> 764`, `introduced: 0`).
  - `git diff --check` -> clean.

### Package 6: Linux-executable qualification

**Status: Linux-runnable qualification complete. Hardware-only checks remain
external.** No production code changed; this stage adds one Python suite, one
WebUI suite, and this ledger.

#### What ran and passed

| Check | Command | Result |
| --- | --- | --- |
| New Python qualification | `.venv/bin/pytest tests/webui/test_package6_qualification.py -q` | **6 passed** |
| New WebUI qualification | `bun run test -- src/tests/package6-qualification.test.tsx` | **5 passed** |
| Broad Python suites | `.venv/bin/pytest tests/webui/ tests/session/ tests/security/test_workspace_access_recovery.py tests/remote/ -q` | **3,076 passed, 1 failed** (pre-existing, see below) |
| Python lint | `.venv/bin/ruff check nanobot/ tests/webui/test_package6_qualification.py` | `All checks passed!` |
| Type regression | `basedpyright` + `scripts/check_type_regressions.py` | `PASS_NO_NEW_DIAGNOSTICS`, `764 -> 764`, `introduced: 0` |
| WebUI lint | `bun run lint` (`--max-warnings 0`) | pass |
| WebUI build | `bun run build` | pass (`✓ built in 5.08s`) |
| WebUI tests | `bun run test -- --maxWorkers=4 --minWorkers=1` | **113 files, 1,838 tests passed** |
| Whitespace | `git diff --check` | clean |

The broad Python run's one failure is
`tests/webui/test_child_approval_ingress.py::test_streaming_sibling_controls_do_not_touch_parent_and_stop_remains_responsive`.
It fails identically on a pristine `HEAD` checkout in a separate worktree, and it
matches the failure this plan already discloses below. It is not caused by this
stage.

#### Python qualification coverage (`tests/webui/test_package6_qualification.py`)

1. **Large-history bounded reads.** Over 220 turns, with and without real
   rotation (≥20 segments), `latest`/`before`/`after`/`around` and `/turns`
   decode ≤66 records, read ≤2 index pages, decode 0 manifest entries, and load
   **0 canonical bodies**. The HTTP route pages backward to prompt 0 with no gap
   or duplicate and still loads 0 canonical bodies.
2. **Cold-start settled recovery.** Three settled 40-turn chats plus one
   journal-marked session: `RecoveryCoordinator.scan` loads no canonical body for
   the settled chats and fully recovers the unsettled one.
3. **End-to-end durability drill.** Multi-turn write → real rotation →
   interrupted rotation recovery (intent file on disk) → index delete and
   rebuild → restart. Rows, prompt ids, pagination cursors, and totals are
   identical, and deletion clears every artifact.
4. **Active/next-turn isolation.** Turn A keeps folder A; the next-turn choice
   holds folder B; a header write during A's completion cannot overwrite B; a new
   turn consumes exactly B; reset bumps generation and orphans the choice.

#### WebUI qualification coverage (`webui/src/tests/package6-qualification.test.tsx`)

1. First paint of saved history plus a visible prompt rail on a short latest
   window of a long chat.
2. Jump to an unloaded Steer prompt, mount and highlight the exact row, forward
   paging without stealing the reader's position, and Go to latest.
3. Draft preservation across A → B → A, a simulated relaunch, and incarnation
   invalidation.
4. Next-folder picker enabled during a run, the held acknowledgement path, and
   no history fetch or model run from a folder change.

#### Remaining external-environment checklist

These cannot run on this Linux host and are explicitly outstanding:

1. **macOS Codex Desktop visual/native comparison.** First open, true cold
   launch, editor ordering, control placement and focus, keyboard/IME behavior,
   and pending/error feedback, compared side by side with the bundled app.
2. **Paired Android device smoke.** The shared chat path and equivalent touch
   behavior on one paired device.
3. **Go mobile broker.** `mobile/native-network/internal/broker/policy.go`
   allowlist for `after`/`around`/`prompt_id`, plus reconciliation of
   `17f1b6cf` (`/turns` mobile registry). `go` is not installed on this host.

## Reproduction and checks

Use the frozen dependency copy and logs under
`/private/tmp/kohaku-chat-g1.bjyRBp` on the development Mac. The baseline and
candidate strict reports used the same interpreter, checker, dependency hashes
and checker config, with imports from this checkout.

- Gateway follow-up: **299 passed** across thread, session projection, input
  attempt, Stop, permission, child lineage, media, remote operation, and
  WebSocket route/reconnect tests (`continue-focused-gateway.log`). Use a
  short pytest base such as `--basetemp="$(mktemp -d /tmp/khp.XXXXXX)"` on
  macOS. The broader pre-follow-up run was **1,474 passed, 1 failed**. The
  child-approval Stop timing assertion also failed with the same pending Stop
  response on a clean export of `a895ac9ba`. It remains a disclosed
  validation failure, not a waived check.
- Renderer: **1,785 passed in 110 suites** with
  `bun run test --maxWorkers=4 --minWorkers=1` (`continue-webui-full.log`).
  The ThreadShell check paints saved rows before the sidebar, withholds
  controls, keeps Stop disabled when controls are unavailable, preserves the
  rows through transport failure, and withdraws them on explicit refusal.
- Native broker: `go test ./internal/broker` in `mobile/native-network`
  passed. This is not a paired-device smoke test.
- Static: unchanged full BasedPyright **0 → 0** diagnostics across 549 files,
  with frozen environment and source manifests in `resume-after` and
  `continue-after`. `scripts/check_type_regressions.py` reported zero new
  signatures. Ruff, `bun run build`, `bun run lint`, and
  `git diff --check` passed.
- The measured G2 fixture and raw counts are in `continue_g2_probe.py` and
  `continue-g2-preflight.json`. Run the probe against the checkout with
  `PYTHONPATH="$PWD"` and the same interpreter; it uses disposable histories.
- Prompt-summary guard: four expected refusals returned 200 before the fix
  (`g2-index-authority-red.log`); **251 focused gateway tests** passed on the
  candidate (`g2-turn-index-focused.log`). Full strict passed **0 → 0** across
  549 files, with the same dependency and environment hashes as
  `continue-after`; the comparator added no signatures. Ruff and
  `git diff --check` passed. The isolated bounded-tail and lifetime probes
  are in `g2_browse_storage_probe.py` and `g2-browse-storage-proof.json`;
  their results are not production latency or release evidence.
- Managed browse proof: **45 focused tests** passed, including real
  reacquisition, both writers, unchanged file identity across an unfinished
  reset, and fail-closed replacement. An intermediate candidate incorrectly
  accepted a same-size edit during directory sync and a UTF-8 header above
  the existing read limit; both regressions now pass
  (`g2-source-receipt-red.log`, `g2-source-proof-focused.log`).
  **2,292 passed** across session, media custody, and affected gateway routes
  (`g2-source-proof-owning.log`). The wider run had **2,316 passed, 1 failed**.
  The same unrelated-source
  cron recovery assertion failed on a clean export of `a895ac9ba`
  (`g2-source-proof-integrated.log`, `g2-source-proof-cron-baseline.log`).
  Unchanged full strict passed **0 → 0** across 549 files against
  `g2-turn-index-after`, with identical checker config, command, environment
  and dependency hashes. The diagnostic comparator, Ruff, and
  `git diff --check` passed.

- Managed display publication: the original gateway probes reproduced the
  premature prompt and delayed output (`g2-display-origin-red.log`). The
  actual channel send path also reproduced reversed Steer/reply order and
  an answer lost from a fork after either a display write failure or
  interrupted delivery (`g2-display-finish-real-fork-red.log`). On the
  candidate, **137 focused tests passed**, including real streamed delivery,
  failed guidance publication, canonical append failure, reset/recreation,
  reopening the source, and managed forks (`g2-display-finish-final-focused.log`).
  The broader gateway/session/runner run was **3,612 passed, 2 failed**.
  Both failures reproduce at `a895ac9ba`: the sibling-approval Stop check
  receives pending instead of stopped, and a source-engine policy fixture
  expects approval where the shell deny filter rejects first
  (`g2-display-finish-integrated.log`, `g2-display-finish-baseline.log`).
  They remain validation failures, not waived checks. Full strict checking
  passed **0 → 0** across 549 files; `g2-display-finish-before` and
  `g2-display-finish-final` retain the source hashes and identical dependency,
  environment, checker config and command manifests. The diagnostic
  comparator, Ruff, and `git diff --check` passed.

- Handoff: independently checked the source proof (**150** tests), split
  routes and attach (**231**), and managed display writes (**114**) at
  their respective commit snapshots. The combined focused Python run on
  the final source tree was **481 passed**. WebUI: **1,785 passed** across
  110 suites, build and lint passed. The native broker suite passed.
  Full strict on the same source and frozen dependencies reported **0
  diagnostics** across 549 files. `ruff check nanobot/` and
  `git diff --check` passed. An extra Ruff run over the whole test tree
  was not clean (including fixture re-exports); that is not the repository's
  application lint check. The previously disclosed broad-run failures and
  bundled-app checks above remain open.

No full local Python suite, native app parity, or release qualification is
claimed.
