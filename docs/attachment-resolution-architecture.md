# Attachment resolution architecture

Status: Core pipeline implemented 2026-11-05. Compatibility fields remain for
old clients and legacy sessions; see the implementation plan for the removal
gates.
Last updated: 2026-11-05.

## 1. Incident

A mobile client uploaded a bedroom photo and asked “Make her sit on the bed”.
The gateway stored the image correctly at
`~/.nanobot-staging/media/websocket/c813a906ea5c.jpg`, but the model called the
Comfy MCP edit tool with `source_image_path=/tmp/bed-photo.png`. That path never
existed, so the tool returned `ENOENT` and the turn ended with the raw stack
trace.

The image was not lost in transport. The phone sent it, the remote edge
forwarded it, and the gateway persisted it. The model never received the
resolved path and invented one.

## 2. Principle

**The server owns media storage and context assembly. Clients are transport
adapters.**

A desktop WebUI, a mobile WebUI, the HTTP API, and every chat channel may acquire
bytes differently, but after the server ingress boundary they must produce the
same canonical attachment record. The agent then receives one uniform input and
the LLM receives one server-assembled projection. No client chooses a storage
path, a reference format, or how media enters the prompt.

This keeps the agent architecture solid: the agent loop, provider adapters, and
tools do not learn about client types. They consume server-owned attachments.

## 3. Current flow

```
mobile WebUI / desktop WebUI
  └─ message envelope { media: [{ data_url, name? }] }
       └─ websocket channel / mobile edge
            └─ store_inbound_attachments()
                 └─ save_base64_data_url()
                      └─ <data>/media/<channel>/<uuid>.<ext>
            └─ InboundMessage.media = [local absolute paths]
                 └─ agent turn
                      ├─ context.build_user_content()
                      │    └─ image bytes as base64 + _meta.path
                      ├─ provider._sanitize_empty_content()
                      │    └─ strips _meta
                      └─ persistence
                           ├─ user text + media paths
                           └─ history replay adds "[image: path]"
```

## 4. Defect

Current-turn image blocks carried the path only in the internal `_meta` field.
`_sanitize_empty_content()` strips `_meta` before dispatch, so the model saw the
pixels but not the path. Non-image attachments already received a textual
`[Attachment: path]` reference, and history replay already synthesized
`[image: path]` from the persisted `media` field. The current turn was the only
place where the path disappeared.

This is a harness context-assembly defect, not a mobile-client or
remote-transport defect. The mobile client is host-neutral by design; the
gateway owns the file and the tools run beside it.

## 5. Phase 0 fix (implemented)

`ContextBuilder.build_user_content()` now adds a separate text block with the
same `[image: path]` reference that history replay uses. The user text block is
left byte-identical, so persistence and existing runtime-tag guarantees are
unchanged. The persisted `media` field remains the source of truth for replay.

`_sanitize_persisted_blocks()` skips the synthetic placeholder when a text block
already carries the same `[image: path]` reference, preventing duplicates in
multimodal history. `content_with_media_breadcrumbs()` is idempotent for the
same reason.

This fixes image-edit MCP calls, video first-frame paths, image-only turns, and
duplicate replay references. It is a last-mile context fix, not the target
pipeline.

## 6. Current fragmentation

The server does not yet have one media path. Each edge owns part of the job:

| Source | Acquisition | Storage | Context contribution |
|---|---|---|---|
| WebUI / mobile WS | inline `data_url` | `webui/attachment_ingress.py` → `media/websocket/<uuid>.<ext>` | context builder + history breadcrumbs |
| Mobile `attachment.write` | JSON `data_base64` | same ingress helper, but returns `{"paths": [...]}` to the client | none until the client sends a message |
| HTTP API | JSON data URL or multipart | `api/server.py` → `media/api/<uuid>_<name>` | passes paths |
| Telegram | platform download | `telegram/runtime.py` → `media/telegram/<file_unique_id>` | channel appends `[image: path]` |
| Discord | platform download | `discord/runtime.py` → `media/discord/<attachment.id>_<name>` | channel appends `[attachment: ...]` markers |
| Other channels | platform download | per-channel directory and naming | per-channel markers |

Consequences:

- Five storage/naming policies and no shared metadata.
- Context references assembled in the channel, the context builder, and replay.
- Mobile `attachment.write` exposes gateway paths to the client.
- Rejected turns can leave staged files behind because storage happens before
  workspace/admission checks.
- No dedupe, no reference-aware retention, and no single place to enforce media
  policy.
- A fix for one client does not automatically hold for another.

## 7. Target architecture

```text
Desktop WebUI ─┐
Mobile WebUI  ─┼─ transport adapter (bytes / data URL / stream) ─┐
HTTP API      ─┤                                                  │
Chat channels ─┘                                                  ▼
                                              AttachmentIngress (one)
                                                validate against source policy
                                                hash + atomic store
                                                create AttachmentRecord
                                                stage / bind to session+turn
                                                              │
                                        InboundMessage.attachments = [record]
                                                              │
                                              AttachmentContext (one)
                                                record -> validated local path
                                                image -> vision block + path ref
                                                non-image -> [Attachment: path]
                                                              │
                                                          LLM / tools
                                                              │
                                            AttachmentProjection (one)
                                                record -> signed media URL (UI)
```

Components:

- **AttachmentInput** — bytes/stream plus name, MIME, source, and source policy.
  Transport-specific decoding stays in the adapter.
- **AttachmentIngress** — validates count, size, and MIME against the source
  policy; hashes; stores atomically; returns an `AttachmentRecord`.
- **AttachmentStore** — content-addressed files under
  `media/attachments/<hash[:2]>/<hash><ext>`, with a sidecar metadata record
  (original name, MIME, bytes, source, created time).
- **AttachmentRecord** — the persisted identity: `id`, `sha256`, `mime`,
  `bytes`, `name`, `source`, `created_at`, optional legacy `storage_key`.
- **AttachmentRef** — runtime view of a record plus its resolved local `path`.
  The absolute path is never persisted.
- **AttachmentBinder** — stages uploads and binds them to an admitted turn or
  session; cleans up staged files when the turn is rejected.
- **AttachmentContext** — the only component that emits `[image: ...]` and
  `[Attachment: ...]` text, for both the current turn and replay.
- **AttachmentProjection** — maps records to signed WebUI media URLs. Clients
  never see filesystem paths.
- **AttachmentLifecycle** — reference-aware retention, temporary-chat cleanup,
  dedupe, and quota.

## 8. Contracts

### 8.1 Client to server

Legacy inline form, kept for external/API and old-client compatibility:

```json
{ "media": [{ "data_url": "data:image/jpeg;base64,...", "name": "photo.jpg" }] }
```

Upload-first form (the implemented desktop/mobile path):

```text
WebSocket attachment frame:
  { "type": "attachment", "request_id": "...", "chat_id": "...",
    "data_url": "data:image/jpeg;base64,...", "name": "photo.jpg" }
  -> { "event": "attachment_stored", "request_id": "...", "id": "att_...",
       "name": "photo.jpg", "mime": "image/jpeg", "bytes": 913092 }

message { "media": [{ "id": "att_..." }] }
```

The native mobile broker chunks large WebSocket frames; the server accepts
bounded uploads up to 24 MiB. A resumable native stream is a follow-up, not a
correctness requirement for the current client.

Neither form accepts a client-supplied filesystem path. Both normalize to the
same `AttachmentRecord`.

### 8.2 Server internal

- `InboundMessage.attachments: list[AttachmentRecord]`. Keep the existing
  `media` path list only as a migration field; remove it after backfill.
- Transcript events store `AttachmentRecord` objects, not absolute paths.
  Legacy events with `media` paths are backfilled into records.
- The model projection contains vision blocks plus server-generated path
  references. The model does not need to know which client uploaded the file.
- The UI projection contains signed URLs built from records. No paths.

### 8.3 `attachment.write`

Return `{id, name, mime, bytes}`, never `{"paths": [...]}`. The client binds the
ID to a later message; the gateway validates that the ID belongs to the same
installation and session. This removes the host-path leak and makes mobile and
desktop uploads identical at the server boundary.

## 9. Client parity rules

- Desktop and mobile use the same message envelope and the same attachment
  service. There is no mobile-specific media branch in the agent or context
  builder.
- The remote edge authenticates and forwards; it does not store media and does
  not define a path contract.
- A new client can be added by implementing only transport acquisition and the
  common attachment envelope.
- A client may display a local preview before upload, but that preview is not
  the server reference.

## 10. Context assembly rules

- `AttachmentContext` is the only source of `[image: ...]` and
  `[Attachment: ...]` text. Remove channel-level breadcrumbs and the separate
  replay breadcrumb implementation.
- Current turn and history replay use the same function. A replay of an
  attachment turn must produce the same model-visible reference as the original
  turn.
- Images produce both a vision block and a path reference; path-aware tools need
  an exact file. Non-images produce an attachment path reference only.
- References are deduplicated. If an attachment is missing or expired, emit a
  typed unavailable marker and log; never emit a stale path.
- The model does not receive model-visible IDs or need a resolver while tools
  run on the gateway. The server resolves refs to paths during context assembly.

## 11. Storage, lifecycle, and security

- Content-addressed files: `media/attachments/<sha256[:2]>/<sha256><ext>`.
  Identical bytes dedupe; the extension is derived from the validated MIME.
- Sidecar metadata records the original name, MIME, bytes, source, and creation
  time. The ref ID is stable and independent of the current path.
- Bind after admission: stage the upload, then commit the binding only when the
  turn is admitted. Rejected turns delete staged files.
- Retention walks session refs and the manifest; a hash file referenced by any
  live session is retained. Temporary chats purge on disconnect or discard.
- Resolve only refs present in the session manifest. Reject cross-session and
  cross-installation reuse for remote ingress.
- Resolve paths under the media root and apply the existing managed-folder
  policy before handing a path to a tool.
- Keep `media.read` signing and native transport authorization separate from
  model-side resolution.
- Do not put base64 or raw paths into thread projections. The mobile streaming
  plan already requires references plus signed URLs for rendered media.

## 12. Migration stages

All stages are part of the committed plan. Compatibility fields are transitional
and are removed at the final gate, not kept indefinitely.

| Stage | Change | Client impact |
|---|---|---|
| 0 | Current-turn path reference; idempotent replay (implemented) | None |
| 1 | Extract `AttachmentIngress` / `AttachmentStore`; WebUI and mobile edge use it; stage/commit admission | None |
| 2 | Move HTTP API and channel downloads to the same service; remove channel-level context markers | None |
| 3 | Add upload-first IDs; `attachment.write` returns IDs; message accepts `{id}`; native streaming for large mobile files | Desktop/mobile update |
| 4 | Store records in transcripts; UI projection signs from records; migrate legacy paths | UI-only |
| 5 | Reference-aware retention and GC; remove legacy path fields and shims | None |

Stage 1 is the key architectural step: it creates the one server boundary that
all later clients share. The complete plan is in
[attachment resolution implementation plan](attachment-resolution-implementation-plan.md).

Implementation note: the central adoption hook in `AgentLoop._restore_turn`
copies every adapter media path under the media root through
`AttachmentIngress`, so channel adapters share canonical storage without
per-channel rewrites. `nanobot attachments migrate` rewrites legacy `media`
events into `attachments` records for sessions that predate record storage.

## 13. Acceptance checks

- Desktop and mobile uploads of the same image produce the same stored hash,
  ref shape, model-visible content, and UI projection.
- Only one ingress service writes media; no channel or transport writes directly
  to a media directory.
- A rejected or unadmitted turn leaves no orphaned staged files.
- An image-edit tool receives a real, readable `source_image_path` on the first
  turn, from either client.
- History replay produces exactly one reference per attachment and no duplicate
  markers.
- A missing file produces a typed unavailable result, not a stale path.
- Upload-first transfers a large attachment without a whole-body base64 WS
  frame; inline remains compatible.
- Security: cross-session ID, revoked installation, path traversal, and managed
  folder denial all fail closed.
- Retention does not delete a hash file still referenced by a live session.

## 14. Non-goals and open questions

- Model-visible attachment IDs and a resolver tool are not part of this target.
  The server resolves records to local paths during context assembly while tools
  run on the gateway. Revisit only if tools execute off-gateway or path exposure
  becomes unacceptable.
- Cross-session dedupe versus per-session copies: the target uses a global
  content-addressed store with reference-aware retention.
- Upload-first transport: bounded 24 MiB WebSocket uploads are implemented and
  the native broker chunks frames. A resumable native stream for larger files
  remains a follow-up.
- Desktop upload parity: use the same `/api/attachments` contract for desktop
  and mobile rather than a second dialect.
- Outbound media (agent-generated images, `message` tool attachments, channel
  sends) remains on its existing pipeline; this plan covers inbound attachments
  and the context handed to the LLM.
