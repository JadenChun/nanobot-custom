# Generated Assets Gallery — Implementation Plan

**Status:** Implemented (Milestones A, B, and Phase 3)

**Goal:** Add a session-header button that opens a read-only gallery of every image or video generated
in that session by the ComfyUI MCP tools, grouped in tool-call sequence, with alternative flat
"all assets" and cross-session views.

**Scope:** WebUI + WebUI backend. No change to agent prompts, Comfy workflow execution, or what the
model sees in tool results.

**Non-goals for the first milestone:** Changing the agent critique loop, auto-delivering candidates to
the chat, cross-session asset browsing, video playback, mobile support.

---

## 1. Current state

### What already works

- **Every Comfy candidate is persisted.** MCP image blocks are converted into artifacts by
  `nanobot/agent/tools/mcp.py::_render_call_result` and stored by
  `nanobot/utils/artifacts.py::store_generated_image_artifact`.
  - Images: `~/.nanobot-staging/media/generated/<date>/img_*.png`
  - Metadata sidecar: matching `img_*.json` (prompt, model, provider, source images, created_at)
- **WebUI can serve images.** `nanobot/webui/media_api.py` signs paths under the media root as
  `/api/media/<sig>/<payload>`, and `sign_or_stage_media_path()` can stage allowed external files.
- **Tool-call order is persisted.** WebUI transcript records contain `turn_id`, `tool_events[]` with
  `call_id`, `name`, `arguments`, `phase`, and the full `result`. `read_transcript_lines()` reads all
  rotated segments, not only the active one.
- **The header has an action slot.** `ThreadHeader` accepts `actions`, `promptNavigatorAction`, and
  `sessionInfoAction`; `SessionInfoPopover` is the established header-action pattern.
- **Lightbox exists.** `webui/src/components/ImageLightbox.tsx` and `AttachmentTile` can render
  thumbnail grids and full-size images.
- **Session-scoped route pattern exists.** `nanobot/webui/ws_http.py::_dispatch_session_routes` handles
  `/api/sessions/{key}/webui-thread`, `/file-preview`, `/automations`, etc.
- **Session-reference authorization exists.** `nanobot/webui/file_preview.py::file_preview_reference`
  already limits previews to paths referenced by the session transcript.

### What is missing

1. **No session/turn/tool-call linkage in artifact metadata.** The sidecar knows the prompt and model,
   but not which chat, turn, or tool call produced the file.
2. **No asset query API.** The WebUI would have to parse tool-result strings.
3. **No gallery surface.** Image-generation activity is intentionally one line; tool result images are
   not rendered.
4. **Video files are not captured.** Comfy returns video as an MCP `resource_link`, while nanobot only
   converts image content blocks into artifacts.
5. **No lifecycle policy.** Session deletion, temporary chats, forks, orphan cleanup, and retention are
   undefined for generated assets.
6. **No mobile bridge support** for a new session route.

### Key constraint

The WebUI transcript is a display log. It is good enough for the first gallery and for backfilling old
sessions, but it should not become the permanent asset database. Tool-result strings are
tool-specific, can be large, and are normalized/offloaded in other code paths.

---

## 2. Target architecture

```
Comfy MCP / image tool
        │
        ▼
Generated media artifact
  bytes + sidecar metadata
        │
        ▼
ToolOutcome.media_artifacts
        │
        ▼
Persisted tool event
  session, turn, call, sequence, tool name, artifact refs
        │
        ▼
Optional session projection/index
        │
        ▼
GET /api/sessions/{key}/generated-assets
        │
        ▼
Header button → Generated assets sheet
```

### 2.1 Artifact record

Keep the existing image file + JSON sidecar. Extend the sidecar with:

```json
{
  "id": "img_c772c1de09e4",
  "kind": "image",
  "mime": "image/png",
  "path": "/home/jaden/.nanobot-staging/media/generated/2026-11-03/img_c772c1de09e4.png",
  "name": "img_c772c1de09e4.png",
  "bytes": 1842231,
  "sha256": "…",
  "prompt": "…",
  "model": "…",
  "provider": "mcp:comfy",
  "source_images": [],
  "created_at": "2026-11-03T14:39:12+08:00"
}
```

The sidecar remains human-inspectable and is the source of truth for file metadata. Session/turn/call
provenance belongs to the event reference, not the media file, so forked sessions can reference the
same immutable bytes.

### 2.2 Event reference

Add a structured `media_artifacts` list to `ToolOutcome` and persist it on the tool event:

```json
{
  "version": 1,
  "phase": "end",
  "call_id": "…",
  "name": "mcp_comfy_generate_image",
  "arguments": { "prompt": "…", "workflow": "qwen_image_2_1" },
  "result": "…unchanged model-visible text…",
  "media_artifacts": [
    {
      "id": "img_c772c1de09e4",
      "kind": "image",
      "mime": "image/png",
      "path": "/home/jaden/.nanobot-staging/media/generated/2026-11-03/img_c772c1de09e4.png",
      "name": "img_c772c1de09e4.png",
      "bytes": 1842231,
      "sha256": "…"
    }
  ]
}
```

The model-visible `result` stays exactly as it is today, including the existing instruction to use the
`message` tool for delivery.

### 2.3 API

```
GET /api/sessions/{key}/generated-assets
    ?group=call|flat
    &kind=image|video
    &cursor=…
    &limit=…
```

```json
{
  "schemaVersion": 1,
  "sessionKey": "websocket:…",
  "total": 14,
  "groups": [
    {
      "id": "turn:…/call:…",
      "turnId": "…",
      "toolCallId": "…",
      "toolName": "mcp_comfy_generate_image",
      "sequence": 3,
      "startedAt": 1793438338161,
      "arguments": { "prompt": "…", "workflow": "qwen_image_2_1", "candidates": 4 },
      "artifacts": [
        {
          "id": "img_c772c1de09e4",
          "kind": "image",
          "mime": "image/png",
          "name": "img_c772c1de09e4.png",
          "url": "/api/media/<sig>/<payload>",
          "createdAt": 1793438339000
        }
      ]
    }
  ],
  "next": null
}
```

Rules:

- Sign URLs just in time; never persist signed URLs.
- Only return artifacts referenced by this session’s transcript/event stream.
- Never accept an arbitrary path from the browser.
- Skip missing files instead of failing the whole response.
- Keep the response schema stable so the backend can move from transcript parsing to structured refs
  without changing the UI.

### 2.4 UI

Insert a button in the session header, next to the session-info and prompt-navigator actions.

```
[sidebar] [title]                         [assets] [info] [prompts] [temp] [theme]
```

Opening it shows a right-side sheet:

- Header: “Generated assets”, total count, refresh action.
- View switch: **By tool call** (default) / **All assets**.
- By tool call: groups in call order, each with sequence number, tool label, time, prompt excerpt,
  candidate count, thumbnail grid, and candidate numbers.
- All assets: one flat chronological grid.
- Clicking a thumbnail opens `ImageLightbox` with the group’s images.
- Metadata expander: prompt, workflow, seed, model, created_at.
- Optional marker: “Delivered” for artifacts that appear in an assistant message’s media list.
- Loading, empty, error/retry states.
- Refetch on `turn_end` while the panel is open.

Recommended component: `webui/src/components/ui/sheet.tsx` with a wider `className`, not a small
popover, because the panel must show a thumbnail grid.

---

## 3. Phase 1 — Read-only gallery from current transcript

**Outcome:** A working header button and gallery for existing and new sessions, with no agent-loop
changes. The backend parser is a compatibility layer behind the final API contract.

### Backend tasks

- [x] Add `nanobot/webui/generated_assets.py`.
  - `build_generated_asset_groups(records, session_key)` pure function.
  - Walk all transcript records in order.
  - For each `tool_events` entry with `phase == "end"` and a Comfy tool name:
    - Prefer structured `media_artifacts` if present (Phase 2).
    - Fallback: parse `result` JSON for `artifacts[]` (`id`, `path`, `mime`, `prompt`, `created_at`).
  - Group by `(turn_id, call_id)`; preserve first-seen order; assign `sequence`.
  - Deduplicate by artifact id/path.
  - Skip paths outside `get_media_dir()`, missing files, and non-image MIME types for the first
    milestone.
  - Read sidecar JSON when present for prompt/model/provider/bytes.
- [x] Add route in `nanobot/webui/ws_http.py::_dispatch_session_routes`:
  - `^/api/sessions/([^/]+)/generated-assets$`
  - Check API token, decode session key, require a WebSocket session key.
  - Read transcript lines in a worker thread.
  - Sign each artifact path via the existing media gateway.
  - Return the stable response schema.
- [x] Add security tests:
  - A path referenced by another session is not returned.
  - A path outside the media root is ignored.
  - Missing file does not fail the request.
  - Unauthenticated request is rejected.
- [x] Add the route to the mobile bridge only when mobile support is in scope.

### Frontend tasks

- [x] Add types in `webui/src/lib/types.ts`:
  - `GeneratedAsset`
  - `GeneratedAssetGroup`
  - `SessionGeneratedAssetsPayload`
- [x] Add `fetchSessionGeneratedAssets()` in `webui/src/lib/api.ts`.
- [x] Add `useSessionGeneratedAssets()` in `webui/src/hooks/`:
  - Fetch on open.
  - Abort on close/session change.
  - Expose `refresh()`.
  - Support a refresh key driven by `turn_end` / `displayMessages` completion.
- [x] Add `webui/src/components/thread/GeneratedAssetsSheet.tsx`.
  - Uses `Sheet`, `Button`, `SegmentedControl`, thumbnail grid, `ImageLightbox`.
  - Two view modes.
  - Group and flat layouts.
  - Empty/error/loading states.
- [x] Add `webui/src/components/thread/GeneratedAssetsButton.tsx`.
  - Header icon button.
  - `aria-label`, tooltip, optional count badge.
  - Owns sheet open state.
- [x] Add `generatedAssetsAction?: ReactNode` to `ThreadHeader` and render it before
  `sessionInfoAction`.
- [x] In `ThreadShell`, build the action from `historyKey` + `token` and pass it to `ThreadHeader`.
- [x] Add i18n keys under `thread.generatedAssets.*` in `webui/src/i18n/locales/*/common.json`.
  - The locale parity test (`webui/src/tests/i18n.test.tsx`) requires every locale to match the
    English key shape.
- [x] Add tests:
  - Button opens and closes the sheet.
  - Groups render in tool-call order.
  - Flat view renders all artifacts.
  - Empty state.
  - API error + retry.
  - Thumbnail click opens the lightbox.
  - Session change closes/refetches.

### Acceptance criteria

- For a session that has Comfy candidates, the header button opens a gallery showing every candidate
  stored under the media root for that session.
- Groups are ordered by the persisted tool-call sequence, not filename or wall-clock guesswork.
- The flat view shows the same assets in chronological order.
- Existing sessions that predate structured refs work through the result-JSON fallback.
- No prompt, tool-result text, or agent behavior changes.
- `null`/missing artifacts never produce a 500.

---

## 4. Phase 2 — Durable structured references

**Outcome:** New sessions carry explicit artifact references in their event stream, so the gallery
stops depending on Comfy JSON parsing. The Phase 1 parser remains as a fallback for old sessions.

### Backend tasks

- [x] Add `GeneratedAssetRef` (or equivalent) to `nanobot/utils/artifacts.py`.
- [x] Extend `store_generated_image_artifact()`:
  - Add `kind`, `bytes`, `sha256`, `name`.
  - Keep existing keys for compatibility.
- [x] Extend `ToolOutcome` in `nanobot/agent/turn.py`:
  - `media_artifacts: tuple[Mapping[str, Any], ...] = ()`.
  - Default empty; no effect on existing tools.
- [x] Preserve the field everywhere `ToolOutcome` is copied/normalized:
  - `nanobot/agent/tools/execution.py::_copy_recorded_value`
  - `nanobot/agent/tools/registry.py` normalization branches
  - any `replace(outcome, …)` paths that rebuild an outcome
- [x] Change `KnownToolResult.event` and related event typing from `dict[str, str]` to a permissive
  event payload type.
- [x] In `_execute_tool_call`, attach `media_artifacts` to the success event.
- [x] In `nanobot/utils/progress_events.py`:
  - Start payload includes `media_artifacts: []`.
  - Finish payload copies artifacts from the event or outcome.
- [x] In `nanobot/agent/tools/mcp.py`:
  - `_render_call_result()` returns text plus artifact metadata.
  - `MCPToolWrapper.execute()` returns `ToolOutcome(content=…, media_artifacts=…)` when artifacts
    exist; otherwise keep returning the string.
  - Do not alter the model-visible text.
- [x] Keep the built-in `ImageGenerationTool` unchanged in this phase to avoid breaking its direct
  `execute()` tests; add it in a follow-up if the same gallery should cover OpenRouter/Codex images.
- [x] Update the WebUI transcript/event types and the gallery API to prefer `media_artifacts`.
- [x] Backfill behavior: if a session has no structured refs, use the Phase 1 parser.

### Tests

- [x] MCP image result produces text + structured refs.
- [x] `ToolOutcome` refs survive registry execution, `_copy_recorded_value`, and result recording.
- [x] Progress start/finish payloads preserve refs.
- [x] Transcript replay preserves refs and grouping.
- [x] Existing agent tests that assert MCP tool result text are unchanged.
- [x] Gallery prefers structured refs over parsing the result string.

### Acceptance criteria

- A new Comfy session shows the gallery with no result-string parsing.
- A pre-existing session still shows the gallery through the fallback.
- Tool results shown to the model are byte-for-byte compatible with the current behavior.
- Dropping/altering Comfy’s JSON `artifacts` field no longer breaks new-session galleries.

---

## 5. Phase 3 — Projection, video, lifecycle, mobile

These are follow-ups and should not block the first gallery.

### Projection/index

- [x] Add a rebuildable per-session artifact projection:
  - `webui/<session-stem>.generated-assets.json`.
- [x] Keep the event refs as the source of truth; the index is derived from a cheap transcript
  signature and rebuilt when the transcript changes.
- [x] Add pagination/cursor support to the API and the gallery `Load more` action.

### Video and file assets

- [x] Capture MCP `resource_link` outputs (for example generated videos) into the media store.
- [x] Introduce a safe staging helper for trusted local MCP output paths (`tools.mediaOutputRoots`).
- [x] Add video thumbnails and a video-capable lightbox to the gallery; file artifacts are indexed too.

### Lifecycle

- [x] Decide and document: session delete keeps bytes and removes transcript references.
- [x] Hide the gallery for temporary chats.
- [x] Add orphan detection/GC for sidecars with no event reference.
- [x] Add retention/quota settings (`tools.generatedMediaRetentionDays`,
  `tools.generatedMediaOrphanGraceDays`, `tools.generatedMediaMaxBytes`) and the
  `nanobot media-gc` command.
- [x] Session forks copy artifact references through the copied transcript events; a regression test
  covers the gallery after a fork.

### Mobile

- [x] Add the route to `nanobot/remote/contracts/operations-v1.json`.
- [x] Add the edge/protocol route and the `webui/src/lib/http.ts` mobile path handling.
- [x] Reuse the existing signed media proxy for images and videos.
- [x] Add the cross-session `/api/generated-assets` library endpoint and the gallery's
  **All sessions** view.

---

## 6. Testing strategy

### Backend

- Unit tests for transcript extraction, ordering, dedupe, missing files, and fallback parsing.
- Route tests for authentication, session validation, response schema, and URL signing.
- MCP wrapper tests for text compatibility and structured refs.
- Progress-event tests for start/finish payloads.
- Existing WebUI thread/fork tests must continue to pass.

### Frontend

- Component tests with mocked API responses.
- Header action test: button appears for a session with a key and opens the sheet.
- Grouped/flat view tests.
- Empty/error/loading tests.
- Lightbox test.
- Locale parity test for the new keys.

### Manual verification

1. Generate a small Comfy batch in a WebUI session.
2. Open the header gallery while the turn is still running and after it completes.
3. Confirm all candidates appear, including ones the agent did not deliver.
4. Confirm grouping follows the tool-call order.
5. Confirm the flat view matches.
6. Refresh the page and confirm the gallery rehydrates from the transcript.
7. Fork the session and confirm the copied transcript shows the parent’s earlier assets.
8. Verify a non-media session shows an empty state, not an error.

---

## 7. Security and privacy

- The API must derive artifacts from the session’s persisted references only; never accept a `path`
  query parameter from the client.
- Reject paths outside the media root for Phase 1. Any future external staging must use an explicit
  allowlist and the existing managed read policy.
- Signed URLs are short-lived and relative to the server; do not store them in sidecars or
  transcripts.
- Remote users see only their own session key’s assets.
- Artifact prompts and metadata can contain sensitive content; do not expose them through
  unauthenticated routes or cross-session queries.
- Temporary chats need an explicit policy before their assets are exposed.

---

## 8. Backward compatibility and migration

- No migration is required for Phase 1: it reads existing transcript tool events.
- Phase 2 is additive. Old events without `media_artifacts` use the fallback parser.
- Old sidecars remain valid; new fields are optional on read.
- The API schema is versioned. The UI must tolerate missing `arguments`, missing sidecars, and skipped
  files.
- Do not rewrite historical transcripts or sidecars as part of this work.

---

## 9. Open decisions and recommended defaults

| Decision | Recommendation |
|---|---|
| Images only or images + video first | Both implemented; video capture requires `tools.mediaOutputRoots` |
| Button always visible or only when assets exist | Always visible for non-temporary sessions; badge count after load |
| Session delete | Keep bytes, remove transcript references; `nanobot media-gc` handles expiry/orphans |
| Forked sessions | Show parent assets through copied event references |
| Temporary chats | Hide gallery in Phase 1; define lifecycle in Phase 3 |
| Delivered/selected marker | Add if cheap by matching assistant media paths; not required |
| Mobile | Read-only routes allowlisted and contract inventory updated; the WebUI sheet is used on mobile |
| Per-session index | Implemented with signature invalidation, pagination, and `Load more` |

---

## 10. Recommended delivery sequence

1. **Milestone A — visible feature:** Phase 1 backend parser + API + header button + sheet + tests.
2. **Milestone B — durable refs:** Phase 2 structured refs + fallback compatibility + tests.
3. **Milestone C — hardening:** Phase 3 video/lifecycle/mobile/index as needed.

Milestone A is deliberately useful on its own, but its backend is a compatibility adapter. Milestone B
is where the architecture becomes stable.

---

## 11. Implementation notes — Milestone A

Implemented on `feat/generated-assets-gallery`:

- `nanobot/webui/generated_assets.py` projects asset groups from persisted transcript tool events,
  validates paths against the media root, enriches metadata from sidecars, and signs URLs.
- `GET /api/sessions/{key}/generated-assets` is registered in `nanobot/webui/ws_http.py`.
- `GeneratedAssetsSheet` adds the header button and the grouped/flat gallery views.
- The mobile API allowlist includes the new read-only route.
- Tests: `tests/webui/test_generated_assets.py` and
  `webui/src/tests/generated-assets-sheet.test.tsx`.

One implementation detail worth preserving: Comfy MCP batches place critic/reference image blocks
before the generated candidates. The projection therefore keeps only the trailing
`candidate_count` images from each tool result, instead of showing reference images as candidates.

### Milestone B — structured references

Implemented:

- `ToolOutcome.media_artifacts` carries minimal durable asset references
  (`id`, `kind`, `mime`, `name`, `path`, `bytes`, `sha256`).
- `MCPToolWrapper` returns a `ToolOutcome` for image-bearing results and keeps the model-visible text
  unchanged.
- `_copy_recorded_value`, registry reconstruction paths, and tool-error/stop paths preserve the new
  field.
- Start/finish tool progress payloads carry `media_artifacts`.
- Built-in OpenRouter/Codex image generation returns a string-compatible `ToolResult` with
  `media_artifacts`, which the registry promotes to `ToolOutcome`.
- The gallery prefers structured references and keeps the result-JSON parser for older sessions.
- Artifact sidecars now include `kind`, `name`, `bytes`, and `sha256`.
- Contract copies and the mobile desktop-parity inventory include the new read-only route.
- Per-session `generated-assets.json` projection caches raw groups keyed by a transcript signature;
  the API supports `limit`/`cursor` pagination and the gallery has a `Load more` action.

### Completed hardening

- **Video/resource-link capture:** trusted local MCP files are staged only under configured
  `tools.mediaOutputRoots`.
- **Orphan GC/retention:** `nanobot media-gc` removes expired or unreferenced generated media and
  enforces the configured byte quota without deleting referenced assets.
- **Cross-session asset library:** exposed through `/api/generated-assets` and the gallery's
  **All sessions** view.
