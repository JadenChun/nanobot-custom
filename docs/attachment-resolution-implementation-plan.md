# Attachment resolution implementation plan

Status: Core implementation and Android emulator native Tailscale upload
verification delivered 2026-11-05. Resumable streaming remains the only optional
follow-up.
Related design: [attachment resolution architecture](attachment-resolution-architecture.md).

## Implementation status (2026-11-05)

| Workstream | State |
|---|---|
| W1 canonical store/ingress | Implemented: `nanobot/attachments/`, content-addressed store, staging, records, policy |
| W2 context formatter | Implemented: `AttachmentContext`, resolver, current turn/replay/memory use it |
| W3 HTTP API | Implemented: API JSON/multipart uploads use `AttachmentIngress` |
| W3 channels | Implemented centrally: every adapter path under the media root is copied through `AttachmentIngress` into the canonical store during `_restore_turn` |
| W4 upload-first server | Implemented: WS `attachment` frame returns IDs; `message.media:[{id}]` binds IDs with installation/session checks |
| W4 WebUI client | Implemented for all attachments: upload-first with inline fallback when the upload frame fails |
| W5 transcript records | Implemented: new user events persist records and omit `media`; replay/UI derive paths from records; `attachments migrate` rewrites legacy events |
| W6 retention | Implemented: reference hash scan, `prune_unreferenced`, `nanobot attachments prune`, and `nanobot attachments migrate` |
| Native large upload | Bounded chunked WS transport supports 24 MiB; resumable streaming remains an optional enhancement |
| Media-path removal | Applied on staging: no session JSONL contains `media`; new user events are records-only; `media_paths_for_message` no longer reads legacy `media`. `InboundMessage`/`TurnRequest` store only `attachments`; `media` is a constructor argument and compatibility property backed by a transient private compatibility field, never persisted canonical data |
| Channel markers | Successful path and basename markers are stripped during canonical adoption; `AttachmentContext` owns the model projection |
| Android emulator verification | PASS 2026-11-05: emulator 37.2.13 + API 36 x86_64 boots with `-gpu angle_indirect`; `qualify-emulator.sh --native-tests` finished 48 tests and reported `M01_RESULT=PASS`. Native Tailscale path: the emulator joined as `nanobot-android` (100.114.89.37), `PairingSeedInstrumentedTest` passed, owner pairing was approved, and the real app sent an `attachment` frame plus `message` with `media:[{id}]` over the embedded tsnet transport. The gateway stored the canonical blob; the session event has exactly one `attachments` record and no `media`; the model inspected the file |

The sections below remain the design and sequencing reference.

## Verification evidence (2026-11-05)

- `pytest tests -q`: **15,856 passed, 23 skipped, 6 subtests passed**.
- Final dataclass check: `InboundMessage` and `TurnRequest` expose `attachments` and no `media` dataclass field; `media` is a compatibility property/constructor argument only.
- Post-dedupe focused suites: `tests/attachments`, attachment/session/WebSocket/API
  and remote tests **310 passed**; `tests/webui` **502 passed**.
- `bun run test` in `webui/`: **1,443 passed** across 93 files.
- `bunx tsc -p tsconfig.build.json --noEmit`: clean.
- `ruff check` over the attachment service, agent context/loop, API, WebUI,
  remote composition, CLI, and new tests: clean.
- `nanobot attachments --help` registers `prune` and `migrate`.
- Live staging migration: gateway stopped, `attachments migrate --yes` migrated
  **1 event in 1 file**; a follow-up dry run reports **no legacy media events**.
  The original file is retained under
  `~/.nanobot-staging/attachment-migration-backups/`; the rewritten session is
  mode 0600 and the gateway health endpoint returned
  `{"status":"ok","process":"alive","ready":true,"websocket":"running"}`.
- Live staging `attachments prune --dry-run` reports **no unreferenced blobs**.
- WebSocket tests cover inline upload, upload-by-ID binding, dedupe, channel
  adoption, and temporary-chat cleanup with content-addressed storage.
- Android emulator verification (2026-11-05):
  - Emulator 37.2.13, API 36 `google_apis` x86_64, AVD `nanobot-m01`, KVM.
  - Boots reliably with `-gpu angle_indirect`; `auto`, `host`, `swiftshader`,
    `swiftshader_indirect`, `off` and `guest` segfault during display setup on
    this Fedora 44 host.
  - `qualify-emulator.sh --native-tests --gpu angle_indirect` finished
    **48 tests** and reported
    `M01_RESULT=PASS native probe executed on emulator-5590`.
  - The WebUI bundle was rebuilt into `nanobot/web/dist`; Chrome on the
    emulator loaded the staging WebUI through
    `adb reverse tcp:8767 tcp:8767`.
  - Attaching a 70-byte PNG produced an `attachment` frame followed by a
    `message` frame with `media: [{"id": "att_5a49..."}]`. The gateway stored
    `media/attachments/cb/cbd5dcb8...png` (bytes match, sidecar present) and
    persisted a user event with **one** `attachments` record and no `media`.
  - The model read the canonical content-addressed file and answered.
  - Native Tailscale path: the emulator joined the tailnet as its own node
    `nanobot-android` / `100.114.89.37`. `PairingSeedInstrumentedTest` passed
    with `OK (1 test)`; owner pairing was approved with scopes
    `owner, chat.read, chat.write, media.read, attachment.write`.
  - Attaching a 69-byte PNG through the real app produced an `attachment`
    frame followed by a `message` frame with
    `media: [{"id": "att_6a96..."}]`, captured at the Capacitor native
    transport boundary. The gateway stored
    `media/attachments/ac/aca25774...png` (bytes match, sidecar present) and
    persisted exactly one `attachments` record with no `media`; the model
    inspected the canonical file.
  - Evidence: `mobile/evidence/20261105-145500-native-upload/`.
  - `go test ./...` in `mobile/native-network` passes the broker package.

Known follow-ups are optional and do not block the active desktop/mobile path:
resumable native streaming beyond the 24 MiB bounded transport.

## 1. Final state

When this plan is complete, one server-owned pipeline handles every inbound
attachment regardless of client:

```text
Desktop WebUI ─┐
Mobile WebUI  ─┼─ transport adapter (bytes / stream / data URL) ─┐
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

End-state contracts:

- **One committed store.** All new inbound attachments are content-addressed
  under `<media_root>/attachments/<sha256[:2]>/<sha256><ext>`.
- **One record type.** `AttachmentRecord` is persisted in the transcript:
  `id`, `sha256`, `mime`, `bytes`, `name`, `source`, `created_at`.
  A migration-only `storage_key` is removed at the W6 gate.
- **One ingress.** WebUI, mobile, HTTP API, and every chat channel call
  `AttachmentIngress`; none writes a canonical media file directly.
- **One context formatter.** `AttachmentContext` is the only code that emits
  `[image: ...]` and `[Attachment: ...]`. Current turn, replay, pending
  injections, and memory formatting all use it.
- **One projection.** `AttachmentProjection` maps records to signed WebUI URLs;
  clients never receive filesystem paths.
- **Transcripts store records, not paths.** New user events carry
  `attachments: [record]`; the `media` path list is removed after migration.
- **Clients are host-neutral.** Official desktop and mobile clients use
  upload-first IDs. Inline `data_url` remains only as an external/API and legacy
  client compatibility input.
- **Retention is reference-aware.** A content-addressed blob is deleted only
  when no live session record references it.
- **Model-visible IDs are not used.** The server resolves records to local paths
  during context assembly because tools run on the gateway. This is the final
  design, not a temporary shortcut.

Removed in the final state:

- direct `media_dir` writes from API and channel ingress;
- channel-specific `[image: path]` / attachment markers for successful ingress;
- `webui/attachment_ingress.py` compatibility shim;
- `media: list[str]` on `InboundMessage`, `TurnRequest`, and transcript events;
- `reference_non_image_attachments` and `content_with_media_breadcrumbs` as
  separate formatting implementations;
- `{"paths": [...]}` responses from `attachment.write`.

## 2. Non-goals

- Thumbnails or image derivatives.
- Encryption-at-rest changes.
- Offline attachment persistence or client-side outboxes.
- Model-visible attachment IDs or a resolver tool.
- Replacing platform-specific outbound attachment sending.

## 3. Baseline

Phase 0 is implemented: the current turn gets `[image: path]`, persistence does
not duplicate it, and replay is idempotent. This plan preserves that behavior
while moving the source of truth into the attachment service.

Deployment scope: WebSocket and the local HTTP API are enabled; no chat channel
is currently enabled. Channel migration is still part of the complete plan and
is sequenced when a channel is enabled or prioritized.

## 4. Workstreams

| Workstream | Result | Depends on |
|---|---|---|
| W1 Canonical service | `nanobot/attachments/` store/ingress, staging, records | Phase 0 |
| W2 Context service | one formatter and resolver for current turn, replay, memory | W1 |
| W3 API and channels | API and all inbound channels use W1/W2 | W1, W2 |
| W4 Upload-first | server-issued attachment IDs; desktop/mobile upload paths | W1 |
| W5 Transcript records | sessions persist records; projection signs URLs | W1, W2 |
| W6 Lifecycle | manifests, retention/GC, legacy migration, removal gates | W3–W5 |

## 5. W1 — canonical attachment service

### 5.1 Package

Create `nanobot/attachments/`:

| File | Responsibility |
|---|---|
| `models.py` | `AttachmentRecord`, `StagedAttachment`, `AttachmentBatchResult`, rejection type |
| `policy.py` | `AttachmentIngressLimits` and per-source policy helpers |
| `store.py` | content-addressed store, stage/commit/discard/resolve, sidecar metadata |
| `ingress.py` | one validation/normalization implementation |
| `binder.py` | stage/bind admission and rollback |
| `context.py` | one model-reference formatter |
| `projection.py` | record -> signed media URL adapter |
| `lifecycle.py` | manifests, retention, migration helpers |
| `__init__.py` | public exports |

Proposed records:

```python
@dataclass(frozen=True, slots=True)
class AttachmentRecord:
    id: str                  # "att_" + uuid4().hex; stable per upload
    sha256: str
    mime: str
    bytes: int
    name: str                # sanitized display name, capped at 128 chars
    source: str              # websocket, api, telegram, ...
    created_at: float
    storage_key: str = ""    # migration only; empty => content-addressed

@dataclass(frozen=True, slots=True)
class AttachmentRef:
    record: AttachmentRecord
    path: Path | None       # resolved at runtime; never persisted as an absolute path

@dataclass(frozen=True, slots=True)
class StagedAttachment:
    token: str
    temp_path: Path
    mime: str
    name: str | None
    bytes: int
    sha256: str
```

Records are serialized with a small `to_dict`/`from_dict` pair and a
`schema_version: 1` in the transcript/manifest envelope. Unknown fields are
ignored on read so a newer writer does not break an older reader.

Store layout:

```text
<media_root>/attachments/<sha256[:2]>/<sha256><ext>
<media_root>/attachments/<sha256[:2]>/<sha256>.json
<media_root>/.staging/<token>
```

- New writes use the content hash. Images, audio, and video are hash-only;
  documents may append a safe-name suffix for display until W5 moves names to
  records, after which the store is hash-only.
- The sidecar records content facts only (`sha256`, MIME, bytes, created time),
  not upload names or sources; those live in `AttachmentRecord`.
- Writes are atomic (temp + replace + fsync, then sidecar). The attachments root
  is mode 0700; resolve rejects symlinks that escape the media root.
- `resolve(record)` validates existence and containment under the media root.
  Legacy `storage_key` paths are resolved relative to the media root and are
  read-only; no new writes use them.
- Images are validated by magic bytes in addition to the declared MIME. SVG
  remains rejected for inbound user uploads to avoid the embedded-script
  surface.
- `AttachmentRecord.id` is random per upload; `sha256` is the dedupe key. Two
  uploads of identical bytes share a blob but keep distinct record IDs and
  source metadata.

Add `get_attachments_dir()` to `nanobot/config/paths.py`, returning
`get_media_dir() / "attachments"`. The store root, the resolver default, and
the WebUI signer all use the media root (`get_media_dir(None)`), so signed URLs
continue to work without copying files into a channel directory.

### 5.2 Policy

```python
@dataclass(frozen=True, slots=True)
class AttachmentPolicy:
    limits: AttachmentIngressLimits
    allowed_mime: frozenset[str]
    max_videos: int = 1
    max_video_bytes: int = 20 * 1024 * 1024
```

Per-source policies:

| Source | Policy |
|---|---|
| WebUI/mobile WS | existing 4 files, 6 MiB/file, 24 MiB total, image/video/document allowlist |
| HTTP API | existing 10 MiB/file, image/document allowlist |
| Telegram/Discord/etc. | current per-channel limits; audio allowed where the channel supports it |

The policy is data; the validation code is shared.

### 5.3 Ingress interfaces

```python
class AttachmentIngress:
    def accept_data_url(self, item: AttachmentInput, *, source: str, policy: AttachmentPolicy, session_key: str) -> StagedAttachment
    def accept_bytes(self, data: bytes, *, name: str | None, mime: str | None, source: str, policy: AttachmentPolicy, session_key: str) -> StagedAttachment
    def accept_path(self, path: Path, *, name: str | None, mime: str | None, source: str, policy: AttachmentPolicy, session_key: str) -> StagedAttachment
    def commit(self, staged: StagedAttachment) -> AttachmentRecord
    def discard(self, staged: StagedAttachment) -> None

    def accept_batch(self, items: Sequence[AttachmentInput], *, source: str, policy: AttachmentPolicy, session_key: str) -> AttachmentBatchResult
    def commit_batch(self, staged: Sequence[StagedAttachment]) -> tuple[AttachmentRecord, ...]
    def discard_batch(self, staged: Sequence[StagedAttachment]) -> None
```

`AttachmentPolicy` carries count, per-file, total, MIME allowlist, and video
limits. The existing `AttachmentIngressLimits` becomes a field of it so the
WebUI policy is unchanged.

### 5.4 Wiring

- `WebUIMediaGateway` builds the service from its media directory.
- `nanobot/webui/inbound_commands.py` uses `accept_batch` and keeps paths for
  the current transcript/UI until W5.
- `nanobot/remote/composition.py` uses the same service.
- `nanobot/webui/attachment_ingress.py` becomes a compatibility shim and is
  removed in W6.
- `webui/ingress_policy.py` imports the shared limits from
  `nanobot.attachments.policy`.

### 5.5 Staging and admission

- `accept_batch` stages files; it does not create a committed record.
- After the message is admitted, `commit_batch` moves files into the store and
  writes records.
- Any rejection, scope failure, or dispatch failure calls `discard_batch`.
- Temporary chats register committed records and purge them on disconnect or
  discard.
- A staging TTL sweeper removes abandoned staging files.

### 5.6 Acceptance

- Only the service writes committed attachment files.
- WebUI and mobile edge use the same ingress function.
- Batch rejection leaves no staged files.
- Hash dedupe works.
- Existing attachment limits and tests remain green.

## 6. W2 — context service

### 6.1 Resolver

```python
class AttachmentResolver(Protocol):
    def resolve(self, record: AttachmentRecord) -> Path | None: ...
```

`ContextBuilder` receives an optional resolver:

```python
ContextBuilder(..., attachment_resolver: AttachmentResolver | None = None)
```

The default resolver uses `get_attachments_dir()` and the media root, so
existing constructors and tests keep working. Tests can inject a fake or a
temporary store. The resolver enforces media-root containment and returns
`None` for missing or legacy-unavailable files.

`AttachmentContext` consumes the resolver and never accepts a client path.

### 6.2 Formatter

```python
class AttachmentContext:
    def user_content(self, text: str, records: Sequence[AttachmentRecord]) -> str | list[dict]
    def append_references(self, text: str, records: Sequence[AttachmentRecord]) -> str
```

Rules:

- Image record + readable path -> vision block + `[image: path]`.
- Non-image record + readable path -> `[Attachment: path]`.
- Missing file -> typed unavailable marker and a log entry; no stale path.
- References are deduplicated by record ID and hash.
- Current turn, pending injections, replay, and memory formatting use the same
  formatter.
- The formatter never emits a path outside the media root and never emits a
  client-supplied path.

### 6.3 Use sites

- `ContextBuilder.build_user_content`
- `reference_non_image_attachments` (temporary wrapper, removed in W6)
- `content_with_media_breadcrumbs` (temporary wrapper, removed in W6)
- `MemoryStore._format_messages`
- `session/source_projection.py` and `session/manager.py`

During W2–W5, keep a path-based `build_user_content` compatibility wrapper for
legacy callers and tests. It creates legacy records with a `storage_key` and
delegates to `AttachmentContext`; the wrapper is removed at the W6 gate.

Context compaction may drop old attachment turns from the prompt, but it must
never convert a record into a stale path or fabricate availability. Session
history remains the source of truth.

### 6.4 Acceptance

- Exactly one code path emits attachment references.
- Current turn and replay produce identical model content for the same record.
- Missing-file behavior is tested.
- A resolver failure never falls back to a stale path.

## 7. W3 — HTTP API and channel migration

### 7.1 HTTP API

Replace direct writes in `nanobot/api/server.py`:

- JSON `data_url` -> `accept_data_url`.
- Multipart file parts -> `accept_bytes` / `accept_path`.
- Preserve the 10 MB API per-file limit as a policy.
- Keep `(text, media_paths)` until W5.
- Keep remote-image-URL rejection.
- Tests: `tests/test_api_attachment.py`, API/WebUI hash parity.

### 7.2 Channels

Each channel keeps acquisition, auth, retries, platform limits, and failure
markers. Only canonical storage and successful context markers move to W1/W2.

| Channel | Acquisition stays | Migration |
|---|---|---|
| Telegram | `_download_message_media` | pass bytes/path to ingress; remove `[image: path]` marker; keep transcription |
| Discord | `_download_attachments` | pass path; remove attachment markers |
| Slack | `_download_slack_file` | pass path |
| WhatsApp | `_download_media` | pass path |
| Signal | `_assemble_inbound_content` | pass path |
| Mattermost | `_download_file` | pass path |
| Feishu | `_download_and_save_media` | pass path |
| WeCom / Weixin | channel download helpers | pass path |
| NapCat / QQ | image/download helpers | pass path |
| DingTalk | file download | pass path |
| Email | MIME extraction | pass bytes |
| Matrix | media download | pass path |

Completion gate: no inbound channel writes a canonical media file or formats a
successful attachment reference. Outbound attachment sending remains
channel-specific.

### 7.3 Acceptance

- API and channel uploads produce the same record shape as WebUI.
- No channel-level successful `[image:]` markers remain.
- Failure markers remain user-visible.
- Per-channel tests pass.

## 8. W4 — upload-first transport

### 8.1 Server contract

```text
POST /api/attachments
  multipart/form-data or raw body
  -> { "id": "att_...", "name": "...", "mime": "...", "bytes": 913092 }

POST /mobile/v1/attachment
  chunked or bounded body
  -> { "id": "att_...", "name": "...", "mime": "...", "bytes": 913092 }

message { "media": [{ "id": "att_..." }] }
```

Route ownership:

- Desktop WebUI HTTP: `nanobot/webui/ws_http.py` / `nanobot/webui/gateway_endpoint.py`,
  using a streaming multipart or raw-body parser. Do not add a second HTTP
  server.
- Mobile: the existing WS `type: attachment` frame remains the bounded path;
  add an HTTP/chunked route for large files. The native broker currently limits
  an HTTP body to 1 MiB and a WS frame to 6 MiB; align the operation contract's
  `max_request_bytes` (currently 256 KiB) with the chosen path.

Rules:

- Uploads are staged and bound to the installation/session on message send.
- Upload IDs are scoped to the uploading installation. The first message send
  binds the ID to that session; retries of the same turn are idempotent, and
  reuse in the same session is allowed. Cross-session and cross-installation
  reuse is rejected.
- Unbound uploads expire by TTL; progress refreshes the TTL.
- Foreign or expired IDs are rejected.
- Inline `data_url` remains accepted for external API compatibility and old
  clients.
- `attachment.write` returns records/IDs, never paths.
- Upload responses include only `{id, name, mime, bytes}`; no path.
- `POST /api/attachments` uses the same WebUI authentication as other HTTP
  routes. `POST /mobile/v1/attachment` requires `attachment.write` and the
  current installation identity.

### 8.2 Desktop client

- `useAttachedImages` uploads first; local blob preview remains client-side.
- `nanobot-client.sendMessage` sends `{id}`.
- `ThreadComposer` shows upload progress and retry.
- Inline fallback only for compatibility.

### 8.3 Mobile client and native transport

- Shared WebUI uses the same upload-first code.
- Small files can use the existing bounded native HTTP request.
- Large files use a new native streaming upload:
  - broker methods `upload_open`, `upload_chunk`, `upload_commit`,
    `upload_cancel`, or a loopback upload proxy;
  - bounded chunks, progress events, cancellation, and timeout;
  - no whole-body base64 in JS;
  - server writes an atomic temp file and returns an ID.
- Align the mobile operation contract: the current `attachment.write`
  `max_request_bytes` (256 KiB) and the WS attachment limit (6 MiB) disagree;
  choose one policy and test it.
- Protocol version bump; old clients keep inline data URLs.

### 8.4 Acceptance

- Desktop and mobile uploads of the same image produce the same record.
- A 20 MiB video uploads without a whole-body base64 WS frame.
- A message with a foreign/expired ID is rejected.
- Inline fallback still works.

## 9. W5 — transcript records and projection

### 9.1 Transcript

New user events:

```json
{
  "role": "user",
  "content": "edit this",
  "attachments": [
    {
      "id": "att_...",
      "sha256": "...",
      "mime": "image/jpeg",
      "bytes": 913092,
      "name": "photo.jpg",
      "source": "websocket",
      "created_at": 1793803728.4
    }
  ]
}
```

- `InboundMessage.attachments: tuple[AttachmentRecord, ...]`.
- `TurnRequest.attachments` replaces `media`.
- The transcript no longer stores absolute paths.
- During transition, readers accept `media` paths and lazily create legacy
  records; writers emit records only.

### 9.2 Projection

- `webui/transcript.py` and `source_projection.py` read records.
- `AttachmentProjection` signs the resolved path and returns
  `{kind, url, name, mime, bytes}`; no path in the UI payload.
- The signed URL format can remain `/api/media/<sig>/<payload>`.
- During the client transition, keep emitting `media_urls` for older clients
  and add the record-backed fields alongside it; remove the compatibility field
  at the W6 gate.

### 9.3 Legacy migration

- On session read, events with `media` and no `attachments` get transient legacy
  records; on the next save, records are persisted.
- A maintenance command (`nanobot attachments migrate`) processes all sessions
  with dry-run, per-session atomic rewrite, backup, and a resumable journal.
- Missing legacy files get an unavailable record; no path is invented.
- After migration, `media` fields and legacy readers are removed.

### 9.4 Acceptance

- New transcripts contain records, not paths.
- Replay, fork, and search work with records.
- UI displays correct names/MIME/size from records.
- Legacy sessions migrate idempotently.
- Missing files produce typed unavailable behavior.

## 10. W6 — lifecycle, retention, and removal

### 10.1 Manifests

- Per-session manifest:
  `sessions/<workspace_id>/<storage_key>.attachments.jsonl`.
- Appended on commit; rebuildable from transcript events.
- Temporary chats have temporary manifests purged on discard/disconnect.

### 10.2 Retention

- Global content-addressed store; blobs shared across sessions.
- GC scans all live manifests under a store lock.
- Delete a blob only when no manifest references its hash, no active turn holds
  a lease, and it is older than a grace period.
- Delete abandoned staging files by TTL.
- Missing/corrupt manifest: rebuild from transcript or skip deletion.
- Optional quota per session/store; report bytes.
- Temporary-chat cleanup removes its manifest; the shared blob survives if
  another session still references it.

### 10.3 Removal gates

Remove legacy code only after:

- all sessions pass the migration command, including legacy `storage_key`
  canonicalization;
- all official clients send upload IDs;
- an old-client compatibility window has elapsed;
- projection and replay use records only.

Then remove:

- `media` fields and path lists;
- `storage_key` and legacy path readers;
- `webui/attachment_ingress.py` shim;
- direct `save_base64_data_url` callers;
- channel successful image markers;
- `reference_non_image_attachments` / `content_with_media_breadcrumbs` wrappers.

### 10.4 Acceptance

- Deleting a session does not delete a blob another session references.
- Temporary chat cleanup purges its records.
- Abandoned staging files are removed.
- Legacy fields are gone after the removal gate.

## 11. Test matrix

| Area | Tests |
|---|---|
| Store | hash, dedupe, atomic write, containment, missing file |
| Ingress | MIME/size/count/total limits, batch rollback, stage/commit/discard |
| Context | current turn, replay, pending injection, memory, idempotence, missing file |
| WebUI/mobile | same envelope -> same record; upload -> message -> model path |
| API | data URL, multipart, 10 MB limit, hash parity |
| Channels | per-channel acquisition + shared storage; no successful markers |
| Transcript | records persisted; fork/replay; legacy migration; missing file |
| Projection | signed URL, name/MIME/bytes, no path leakage |
| Lifecycle | session delete, shared blob retention, temporary chat purge, staging TTL |
| Security | cross-session ID, revoked installation, path traversal, managed folder denial |
| Performance | 20 MiB upload without whole-body base64; large session replay |

## 12. Rollout and rollback

1. W1 + W2 behind internal wiring; no protocol change.
2. W3 API first, then channels as enabled.
3. W4 server accepts IDs; official clients switch with inline fallback.
4. W5 dual-read/dual-write transcripts; then migration.
5. W6 GC and removal after migration evidence.

Rollback:

- W1/W2: keep the shim and old path call sites until the new service is stable.
- W3: revert per-channel changes independently.
- W4: clients fall back to inline data URLs; server keeps both.
- W5: stop writing records; readers still accept paths.
- W6: GC and removal only after backups and migration verification.

## 13. Open decisions

1. **Manifest storage:** per-session JSONL + scan (recommended, matches session
   storage) or a SQLite refcount index.
2. **Legacy migration:** lazy records only, or a required one-time canonical
   copy/rewrite. Recommended: lazy records for correctness, maintenance command
   for canonicalization before removal.
3. **Native upload:** chunked broker methods vs loopback upload proxy. Recommend
   chunked broker methods for explicit bounds and progress.
4. **Dedupe:** global store (recommended) with reference-aware GC.
5. **Channel scope/order:** migrate per enabled channel; define the completion
   gate as all supported inbound channels using W1.
6. **Official client upload threshold:** always upload-first (recommended) or
   inline below a small size with upload-first above it.

## 14. Definition of complete

- Desktop and mobile uploads of the same image produce the same record and the
  same model-visible content.
- Every inbound channel uses `AttachmentIngress`; no channel writes canonical
  media or formats successful attachment references.
- Sessions persist records, not paths.
- Official clients use upload-first IDs; inline remains only for external/legacy
  compatibility.
- Retention is reference-aware and temporary chats purge cleanly.
- Legacy migration is complete and the temporary shims/fields are removed.
- The full test matrix passes, including the mobile physical-device upload path.
