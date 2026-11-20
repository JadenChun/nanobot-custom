# Deferred design note: single session storage (W9)

**Status:** deferred, not started. Recorded so the context survives the
session-load simplification work (WI-1 … WI-8).

**One-line summary:** the WebUI keeps a second, derived copy of every
conversation. Removing that copy is the largest simplification available in the
session-load path, but it is a persistence migration, not a load optimisation.

## What exists today

Two independent stores hold the same conversation:

| Store | Path | Role |
|---|---|---|
| Canonical source | `~/.nanobot-staging/sessions/<workspace-id>/*.jsonl` | append-only record of every model-visible fact; drives agent history, replay, forks |
| WebUI transcript | `~/.nanobot-staging/webui/*.jsonl` + `*.segments/` | display projection written on every turn by `WebUITranscriptRecorder` |

The WebUI projection is **derived** from the canonical log, but it is persisted
and maintained separately, with its own rotation, segmentation, manifest, and
legacy migration.

## Measured cost (staging workspace, 2026-11-20)

| Store | Size |
|---|---|
| `sessions/` (canonical, 6 files) | **5.4 MB** |
| `webui/` (display, 5 sessions + 19 segment files) | **241 MB** |

Per-conversation duplication, one representative chat:

| File | Size |
|---|---|
| `sessions/.../d2Vic29ja2V0OjQ4ZjVkN2Zh….jsonl` | 1.4 MB |
| `webui/d2Vic29ja2V0OjQ4ZjVkN2Zh….jsonl` | 1.7 MB |

Worst offenders are segmented histories: one session holds 103 MB of segments,
another 85 MB, with individual segment files at 53 MB and 36 MB.

Note the ratio: the display copy is roughly **45× the canonical data** in this
workspace, because segmentation keeps both rotated segments and the live tail.

## Why the duplication exists

The canonical source stores raw events; the WebUI needs `UIMessage`-shaped rows
(traces, file edits, compaction rows, delegated-child rows, fork markers) plus
replay identities and stable sign-in URLs for media. Rather than project on read,
the earlier design materialised the projection to disk so reads stayed simple.

That decision is what the current load work has to compensate for.

## Machinery that exists only to serve the display copy

All of this lives in `nanobot/webui/transcript.py` (**3,790 lines**):

| Constant / function | Purpose |
|---|---|
| `_ACTIVE_TRANSCRIPT_ROTATE_BYTES = 2 MB` | rotate the live file |
| `_TARGET_ACTIVE_TRANSCRIPT_BYTES = 1 MB` | target live-tail size |
| `_TRANSCRIPT_SEGMENT_MANIFEST_VERSION = 3` (+ legacy v2) | manifest schema |
| `_TRANSCRIPT_ROTATION_INTENT_VERSION`, `rotation.json` | crash-safe rotation intent |
| `_TRANSCRIPT_SEGMENT_RE` (`^\d{6}\.jsonl$`) | segment naming |
| `_rebuild_segment_manifest` | re-parses and **rewrites** segments (a read-path write — see WI-8) |
| `_migrate_legacy_transcript` | called from 6 sites per request chain (WI-5) |
| `_ensure_replay_identities` | SHA-256 identity per record per read (WI-4) |
| `_needs_user_event_backfill` + 5 repair functions | ~183 lines; repair legacy turns at read time (WI-7) |

If the display copy were removed, the rotation/segmentation/manifest layer would
have no reason to exist.

## Production consumers that would need rework (12 modules)

| Module | Uses |
|---|---|
| `nanobot/webui/ws_http.py` | thread payload, permissions scan, assets, file preview (4 call sites) |
| `nanobot/webui/forking.py` | fork boundaries |
| `nanobot/webui/generated_asset_index.py` | `read_transcript_lines`, `webui_transcript_signature`, `media_publication` |
| `nanobot/webui/session_access.py` | mobile thread payload |
| `nanobot/webui/gateway_services.py` | `WebUITranscriptRecorder` — writes on every turn |
| `nanobot/attachments/_inventory.py` | media retention reconciliation |
| `nanobot/session/webui_turns.py` | `append_session_message_input` |
| `nanobot/session/recovery.py` | `has_unfinished_transcript_tail` |
| `nanobot/channels/websocket/runtime.py` | `WEBUI_TRANSCRIPT_INCOMPLETE_KEY` |
| `nanobot/webui/thread_disk.py` | `delete_webui_transcript` |
| `nanobot/webui/media_gateway.py` | `rewrite_local_markdown_images` |
| `nanobot/webui/session_access.py` | thread projection for remote clients |

## Why it is not part of the load plan

1. **It changes what is persisted.** Rotation, fork, recovery, and media
   retention all treat the WebUI transcript as a durable artifact. Removing it
   is a migration with a data-loss surface.
2. **Blast radius.** 12 production modules and 3,790 lines in one module; the
   change is not attributable to a load regression if something breaks.
3. **The load fixes should make it unnecessary.** WI-1 makes page cost O(page)
   instead of O(file); WI-6 makes revisits a client-side map hit. Neither needs
   the display copy to be removed in order to be fast.
4. **Existing sessions must migrate.** 241 MB of display data would need a
   one-time conversion or discard, with a decision on whether history replayed
   from the canonical log renders identically.

## If it is ever picked up, the shape of that project

- Confirm the canonical log alone can reproduce every display row the WebUI
  needs (traces, file edits, compaction, delegated children, fork markers).
- Land WI-1…WI-8 first, so the projection is cheap and page-scoped before the
  storage layer moves.
- Treat it as a migration: staging workspace first, with a rollback that keeps
  the display copy readable until verified.
- Prove equivalence on a real long session (the 103 MB segmented one is the
  useful stress case, not a short chat).
- Only then delete rotation, segmentation, and the manifest layer.

## Revisit trigger

Re-measure after WI-1 and WI-6. If first load is page-bounded and revisits are
in-memory, the duplication is a **disk-usage** problem rather than a latency
problem, and should be prioritised on that basis instead.
