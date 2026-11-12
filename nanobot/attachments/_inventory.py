"""Exact, bounded, nonrepairing inventory of the declared media-reference formats."""

from __future__ import annotations

import base64
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlsplit

from nanobot.attachments._media import (
    MediaCustody,
    MediaReferenceEvidence,
    MediaRetentionError,
    TemporaryMediaCustody,
    directory_fd,
    generation,
    identity,
    object_json,
    read_bytes,
    unique_json,
)

MAX_FILES = 100_000
MAX_BYTES = 512 * 1024 * 1024
MAX_NODES = 1_000_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_BREADCRUMB = re.compile(r"\[(?:image|file|video|audio|attachment):\s*([^\]\n]+)\]")
_MARKDOWN = re.compile(r"!\[[^\]\n]*\]\((?:<([^>\n]+)>|([^\s)]+))\)")
_RECOGNIZED = re.compile(r'"(?:attachments|media|media_artifacts|artifacts|storage_key|source_images|sha256|artifact_manifest)"\s*:')
_PATH_KEYS = {"path", "storage_key", "media_path", "media_paths", "media", "source_images", "reference_images"}
_MEDIA_LISTS = {"attachments", "media_artifacts", "artifacts", "media_publications", "source_images", "reference_images", "media", "media_paths"}


@dataclass
class InventoryBudget:
    files: set[Path] = field(default_factory=set)
    sizes: dict[Path, int] = field(default_factory=dict)
    bytes: int = 0

    def include(self, paths: list[Path]) -> list[Path]:
        self.files.update(paths)
        if len(self.files) > MAX_FILES:
            raise MediaRetentionError("media evidence file budget exhausted")
        return paths

    def read(self, path: Path) -> bytes:
        self.include([path])
        data = read_bytes(path)
        previous = self.sizes.get(path, 0)
        self.bytes += max(0, len(data) - previous)
        self.sizes[path] = max(previous, len(data))
        if self.bytes > MAX_BYTES:
            raise MediaRetentionError("media evidence byte budget exhausted")
        return data


@dataclass
class References:
    root: Path
    hashes: set[str] = field(default_factory=set)
    paths: set[Path] = field(default_factory=set)
    publications: set[str] = field(default_factory=set)
    artifacts: set[str] = field(default_factory=set)
    nodes: int = 0
    budget: InventoryBudget = field(default_factory=InventoryBudget)

    def read(self, path: Path) -> bytes:
        return self.budget.read(path)

    def files(self, path: Path) -> list[Path]:
        return self.budget.include(walk_files(path))

    def path(self, value: str, *, relative: bool = False) -> None:
        if not value or "\x00" in value:
            raise MediaRetentionError("invalid media reference path")
        text = value.replace("\\", "/")
        candidate = Path(text)
        stem = candidate.name.split(".", 1)[0]
        if _HASH.fullmatch(stem):
            self.hashes.add(stem)
        if ".." in candidate.parts:
            raise MediaRetentionError("media reference traversal")
        foreign = re.match(r"^[A-Za-z]:/", text) is not None or text.startswith("//")
        if relative and not candidate.is_absolute() and not foreign:
            candidate = self.root / candidate
        if candidate.is_absolute() and candidate.is_relative_to(self.root):
            self.paths.add(candidate)
        # A Windows absolute path still supplies a canonical hash identity.
        elif foreign and not _HASH.fullmatch(stem):
            raise MediaRetentionError("unsupported foreign media path")

    def scan(self, value: object, *, key: str = "", depth: int = 0) -> None:
        self.nodes += 1
        if self.nodes > MAX_NODES or depth > 80:
            raise MediaRetentionError("media reference traversal budget exhausted")
        if isinstance(value, dict):
            raw_row = cast(dict[object, object], value)
            if any(not isinstance(name, str) for name in raw_row):
                raise MediaRetentionError("invalid media reference key")
            row = cast(dict[str, object], raw_row)
            if key == "artifact_manifest":
                # The source owner validates manifest membership and full text.
                if set(row) != {"schema_version", "descriptors"} or row.get("schema_version") != 1:
                    raise MediaRetentionError("invalid private artifact manifest")
            if key in {"attachments", "media_artifacts", "artifacts", "media_publications"}:
                if "version" in row and (type(row["version"]) is not int or row["version"] != 1):
                    raise MediaRetentionError("unsupported media record version")
                if not any(name in row for name in ("path", "storage_key", "sha256", "url")):
                    raise MediaRetentionError("unknown media reference representation")
                if key == "attachments" and "sha256" in row and "mime" in row and not row.get("storage_key"):
                    from nanobot.utils.media_decode import mime_extension

                    sha, mime = row["sha256"], row["mime"]
                    if not isinstance(sha, str) or not _HASH.fullmatch(sha) or not isinstance(mime, str):
                        raise MediaRetentionError("invalid attachment storage identity")
                    self.path(str(self.root / "attachments" / sha[:2] / f"{sha}{mime_extension(mime)}"))
            for name, item in row.items():
                if (key == "target" and set(row) == {"kind", "path"}
                        and row["kind"] == "unknown" and name == "path" and item is None):
                    # FolderAccessUnknownTarget is a typed non-media owner
                    # record. A null path in an actual media record still fails.
                    continue
                if (key in {"edits", "fileEdits"} and name == "path" and item == ""
                        and row.get("pending") is True and row.get("phase") == "start"
                        and row.get("status") == "editing"):
                    # The display owner emits this known placeholder before
                    # an edit has acquired its eventual workspace path.
                    continue
                if name in _MEDIA_LISTS and not isinstance(item, list):
                    raise MediaRetentionError("invalid media reference collection")
                if name in {"id", "artifact_id"} and isinstance(item, str):
                    self.publications.add(item)
                if name == "sha256":
                    if not isinstance(item, str) or not _HASH.fullmatch(item):
                        raise MediaRetentionError("invalid media reference hash")
                    self.hashes.add(item)
                if key == "display_message" and name in {"images", "media"}:
                    # Retained UIMessage media has object rows, unlike source
                    # and transcript media path lists. Its owner validated it.
                    self.scan(cast(object, item), key="display_media", depth=depth + 1)
                    continue
                self.scan(row[name], key=name, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in cast(list[object] | tuple[object, ...], value):
                if key in {"media", "media_paths", "source_images", "reference_images"} and not isinstance(item, str):
                    raise MediaRetentionError("invalid media path collection")
                if key in {"attachments", "media_artifacts", "artifacts", "media_publications"} and not isinstance(item, dict):
                    raise MediaRetentionError("invalid media record collection")
                self.scan(cast(object, item), key=key, depth=depth + 1)
        elif isinstance(value, str):
            if key in _PATH_KEYS:
                self.path(value, relative=key == "storage_key")
            elif key == "url" and value.startswith("/api/media/"):
                url = urlsplit(value)
                parts = url.path.split("/")
                if len(parts) != 5:
                    raise MediaRetentionError("invalid signed media reference")
                try:
                    decoded = base64.urlsafe_b64decode(parts[-1] + "=" * (-len(parts[-1]) % 4)).decode()
                except (ValueError, UnicodeError):
                    raise MediaRetentionError("invalid signed media reference") from None
                self.path(decoded, relative=True)
                publications = parse_qs(url.query).get("publication", [])
                if len(publications) > 1:
                    raise MediaRetentionError("ambiguous media publication identity")
                if publications:
                    if not re.fullmatch(r"media_[0-9a-f]{32}", publications[0]):
                        raise MediaRetentionError("invalid media publication identity")
                    self.publications.add(publications[0])
            if value.startswith("artifact:"):
                from nanobot.session.artifact import parse_artifact_reference

                self.artifacts.add(parse_artifact_reference(value))
            if key in {"content", "result", "text", "summary", "prior_record", "target_record", "sidecar_bytes", "source_record"}:
                text = value.strip()
                if text.startswith(("{", "[")):
                    try:
                        embedded = unique_json(text)
                    except (ValueError, OSError):
                        if _RECOGNIZED.search(text):
                            raise MediaRetentionError("malformed recognized media producer result") from None
                    else:
                        self.scan(embedded, depth=depth + 1)
                for match in _BREADCRUMB.finditer(value):
                    self.path(match.group(1).strip())
                for match in _MARKDOWN.finditer(value):
                    target = match.group(1) or match.group(2)
                    if target.startswith("/api/media/"):
                        self.scan(target, key="url", depth=depth + 1)
                    elif not target.startswith(("http:", "https:", "data:")):
                        self.path(target)
                # Plain source text can itself be an exact managed path.
                if text.startswith(str(self.root) + "/"):
                    self.path(text)
        elif key in _MEDIA_LISTS or key in {"path", "storage_key", "sha256"}:
            raise MediaRetentionError("unsupported media reference value")


def retained_markdown_media(text: str) -> list[dict[str, str]]:
    """Keep exact public rewrite identities for the actual display writer."""
    records: list[dict[str, str]] = []
    for match in _MARKDOWN.finditer(text):
        target = match.group(1) or match.group(2)
        if target.startswith("/api/media/"):
            records.append({"url": target})
            if len(records) > 1024:
                raise MediaRetentionError("markdown media publication budget exhausted")
    return records


def rows(data: bytes) -> list[dict[str, object]]:
    if not data:
        return []
    text = data.decode("utf-8")
    if not text or not text.endswith("\n"):
        raise MediaRetentionError("incomplete media evidence JSONL")
    result: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            raise MediaRetentionError("empty media evidence row")
        result.append(object_json(line))
    return result


def walk_files(root: Path) -> list[Path]:
    found: list[Path] = []

    def visit(path: Path, depth: int) -> None:
        if depth > 16:
            raise MediaRetentionError("media evidence directory depth exhausted")
        descriptor = directory_fd(path)
        try:
            names = sorted(os.listdir(descriptor))
            if len(names) + len(found) > MAX_FILES:
                raise MediaRetentionError("media evidence file budget exhausted")
            for name in names:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                child = path / name
                if stat.S_ISDIR(info.st_mode):
                    visit(child, depth + 1)
                elif stat.S_ISREG(info.st_mode):
                    found.append(child)
                else:
                    raise MediaRetentionError("nonregular media evidence or traversal")
        finally:
            os.close(descriptor)

    visit(root, 0)
    return found


def canonical_references(data: bytes, *, expected_key: str | None, workspace_id: str,
                         references: References) -> None:
    from nanobot.session.artifact import bound_artifact_text, published_artifacts
    from nanobot.session.pending_followups import parse_pending_followup_records
    from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY
    from nanobot.session.source_store import (
        SourceSnapshot,
        SourceState,
        validate_source_session_bytes,
    )

    header, events = validate_source_session_bytes(data, expected_key=expected_key)
    # Validate reserved nested followups independently of generic reference descent.
    metadata = header["metadata"]
    if SOURCE_PENDING_FOLLOWUPS_METADATA_KEY in metadata:
        parse_pending_followup_records(metadata[SOURCE_PENDING_FOLLOWUPS_METADATA_KEY])
    state = SourceState("retention-inspection", header["key"], header["incarnation"],
                        header["revision"], header["generation"], None, "")
    snapshot = SourceSnapshot(state, header, events, None)
    descriptors = published_artifacts(snapshot, workspace_id=workspace_id)
    known_artifacts = {descriptor.artifact_id for descriptor in descriptors}
    prior_artifacts = set(references.artifacts)
    references.scan(header)
    references.scan(events)
    for descriptor in descriptors:
        # This is the full raw canonical tool text, never a retrieval preview.
        references.scan(bound_artifact_text(descriptor, snapshot, workspace_id=workspace_id).decode(), key="content")
    if references.artifacts - prior_artifacts - known_artifacts:
        raise MediaRetentionError("unbound private artifact reference")


def _memory(path: Path, references: References) -> None:
    # Only the known journal is classified; arbitrary workspace files are not scanned.
    if (path / ".recovery").exists() or (path / ".consolidation.pending.json").exists():
        raise MediaRetentionError("unsupported retained memory recovery evidence")
    for item in path.iterdir():
        if item.name.startswith("HISTORY.md"):
            raise MediaRetentionError("unsupported retained memory backup evidence")
    history = path / "history.jsonl"
    if not history.exists() and not history.is_symlink():
        return
    previous = -1
    for row in rows(references.read(history)):
        if (not {"cursor", "timestamp", "content"} <= set(row)
                or set(row) - {"cursor", "timestamp", "content", "session_key", "source"}
                or type(row["cursor"]) is not int or row["cursor"] <= previous
                or not isinstance(row["timestamp"], str) or not isinstance(row["content"], str)
                or any(not isinstance(row[name], str) for name in ("session_key", "source") if name in row)):
            raise MediaRetentionError("invalid retained memory journal")
        previous = row["cursor"]
        references.scan(row)


def inspect_webui(path: Path, references: References) -> None:
    from nanobot.webui.transcript import (
        validate_retained_transcript_record,
        validate_retained_webui_thread,
    )

    files = references.files(path)
    for item in files:
        relative = item.relative_to(path)
        if relative == Path("..native-process-admissions.json.lock"):
            # NativeAdmissionLedger uses this exact empty FileLock alongside
            # its ledger. Ledger data itself still needs supported evidence.
            if references.read(item):
                raise MediaRetentionError("invalid native admission lock evidence")
        elif item.suffix == ".jsonl":
            for row in rows(references.read(item)):
                validate_retained_transcript_record(row)
                references.scan(row)
        elif item.name == "manifest.json" and item.parent.name.endswith(".segments"):
            data = object_json(references.read(item))
            if (set(data) != {"version", "segments", "active"} or type(data["version"]) is not int
                    or data["version"] != 3 or not isinstance(data["segments"], list)):
                raise MediaRetentionError("invalid display segment manifest")
            seen: set[str] = set()
            for raw in cast(list[object], data["segments"]):
                if not isinstance(raw, dict):
                    raise MediaRetentionError("invalid display segment entry")
                row = cast(dict[str, object], raw)
                segment = row.get("id")
                if (set(row) != {"id", "bytes", "turn_count", "user_count"}
                        or not isinstance(segment, str) or not re.fullmatch(r"\d{6}", segment)
                        or segment in seen or any(type(row[name]) is not int or cast(int, row[name]) < 0
                                                 for name in ("bytes", "turn_count", "user_count"))):
                    raise MediaRetentionError("invalid display segment entry")
                seen.add(segment)
                raw_segment = references.read(item.parent / f"{segment}.jsonl")
                if len(raw_segment) != row["bytes"]:
                    raise MediaRetentionError("display segment size mismatch")
            active = data["active"]
            if active is not None:
                stem = item.parent.name.removesuffix(".segments")
                if (not isinstance(active, str) or not re.fullmatch(re.escape(stem) + r"\.active\.[0-9a-f]{16}\.jsonl", active)):
                    raise MediaRetentionError("display manifest active traversal")
                references.read(path / active)
        elif item.name.endswith(".generated-assets.json"):
            data = object_json(references.read(item))
            if (set(data) != {"version", "sessionKey", "signature", "groups"}
                    or type(data["version"]) is not int or data["version"] != 1
                    or not isinstance(data["sessionKey"], str) or not isinstance(data["signature"], list)
                    or not isinstance(data["groups"], list)):
                raise MediaRetentionError("invalid generated display index")
            for raw in cast(list[object], data["groups"]):
                if not isinstance(raw, dict) or not isinstance(cast(dict[str, object], raw).get("artifacts"), list):
                    raise MediaRetentionError("invalid generated display group")
            references.scan(data)
        elif item.suffix == ".json" and len(relative.parts) == 1:
            data = object_json(references.read(item))
            if item.name in {"sidebar-state.json", "workspace-state.json"}:
                # These known owners store no media; references anywhere still protect bytes.
                if any(key in data for key in _MEDIA_LISTS | {"path", "storage_key"}):
                    raise MediaRetentionError("unsupported media in UI control record")
                references.scan(data)
            elif isinstance(data.get("messages"), list):
                validate_retained_webui_thread(data)
                for row in cast(list[dict[str, object]], data["messages"]):
                    references.scan(row, key="display_message")
                references.scan({name: value for name, value in data.items() if name != "messages"})
            else:
                raise MediaRetentionError("unknown retained display record")
        elif relative.parts[0] == "session-index" and item.suffix == ".json":
            data = object_json(references.read(item))
            if data.get("version") != 8:
                raise MediaRetentionError("unsupported session index version")
            references.scan(data)
        else:
            raise MediaRetentionError("unknown display evidence file")


def inventory(custody: MediaCustody, registry: dict[str, object]) -> MediaReferenceEvidence:
    references = References(custody.root)
    reasons: list[str] = []
    identities: list[tuple[str, tuple[int, int]]] = []
    declarations = cast(list[dict[str, object]], registry["roots"])
    declared_namespaces: dict[Path, set[Path]] = {}
    pending: list[tuple[str, Path]] = []
    temporary: list[TemporaryMediaCustody] = []
    assets: dict[Path, tuple[int, ...]] = {}
    try:
        # All custody records participate in the same aggregate evidence budget.
        for path in references.files(custody.state):
            relative = path.relative_to(custody.state)
            if relative == Path("publication.lock"):
                continue
            if relative == Path("roots.json") or (
                len(relative.parts) == 2 and relative.parts[0] in {"pending", "deletions"}
                and re.fullmatch(r"[0-9a-f]{32}\.json", relative.name)
            ):
                references.read(path)
            else:
                raise MediaRetentionError("unknown media retention state")
        # Existing undeclared standard owners are pre-contract coverage, not
        # evidence of absence. Custom roots must be declared by their owners.
        standard_sessions = custody.root.parent / "sessions"
        if (standard_sessions.exists() or standard_sessions.is_symlink()) and not any(
            row["kind"] == "sessions" and row["sessions_root"] == str(standard_sessions)
            for row in declarations
        ):
            raise MediaRetentionError("undeclared sessions reference root")
        for kind, selected in (("webui", custody.root.parent / "webui"),
                               ("temporary", custody.root.parent / "temporary-chat-cleanup")):
            if (selected.exists() or selected.is_symlink()) and not any(
                row["kind"] == kind and row["path"] == str(selected) for row in declarations
            ):
                raise MediaRetentionError(f"undeclared {kind} reference root")
        for declaration in declarations:
            root = Path(cast(str, declaration["path"]))
            descriptor = directory_fd(root)
            try:
                actual = identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)
            if list(actual) != declaration["identity"]:
                raise MediaRetentionError("declared media root identity changed")
            identities.append((str(root), actual))
            kind = declaration["kind"]
            if kind == "unsupported_sessions":
                raise MediaRetentionError("selected custom session backend has unsupported retained media evidence")
            if kind == "sessions":
                from nanobot.session.manager import JsonlSessionStore
                from nanobot.session.session_mode import (
                    SessionModeError,
                    inspect_retained_media_records,
                )

                session_root = Path(cast(str, declaration["sessions_root"]))
                workspace = Path(cast(str, declaration["workspace"]))
                declared_namespaces.setdefault(session_root, set()).add(root)
                descriptor = directory_fd(session_root)
                try:
                    identities.append((str(session_root), identity(os.fstat(descriptor))))
                finally:
                    os.close(descriptor)
                descriptor = directory_fd(workspace)
                try:
                    workspace_identity = identity(os.fstat(descriptor))
                    if list(workspace_identity) != declaration["workspace_identity"]:
                        raise MediaRetentionError("declared workspace identity changed")
                    identities.append((str(workspace), workspace_identity))
                finally:
                    os.close(descriptor)
                if references.read(root / ".workspace").decode().strip() != str(workspace):
                    raise MediaRetentionError("session workspace binding mismatch")
                files = references.files(root)
                mode_files: dict[str, bytes] = {}
                for path in files:
                    relative = path.relative_to(root)
                    if relative.parts[0] == ".session-mode":
                        mode_files[str(relative)] = references.read(path)
                        continue
                    if path.name in {".workspace", ".session-files.lock"}:
                        continue
                    if relative.parts[0] == ".execution-locks" and path.suffix == ".lock":
                        continue
                    if len(relative.parts) == 1 and path.suffix == ".jsonl":
                        expected_key = JsonlSessionStore.session_key_from_path(path)
                        if expected_key is None:
                            raise MediaRetentionError("unknown canonical session filename")
                        data = references.read(path)
                        canonical_references(data, expected_key=expected_key, workspace_id=root.name, references=references)
                    else:
                        raise MediaRetentionError("unsupported canonical owner record")
                try:
                    retained = inspect_retained_media_records(root, workspace, mode_files)
                except SessionModeError as exc:
                    raise MediaRetentionError(str(exc)) from exc
                for format_name, key, data in retained:
                    if format_name == "canonical":
                        canonical_references(data, expected_key=key, workspace_id=root.name, references=references)
                    elif format_name == "legacy":
                        JsonlSessionStore.parse_legacy_bytes(data, expected_key=key)
                        references.scan(rows(data))  # Validate/extract RAW rows; no legacy projection.
                    else:
                        references.scan(unique_json(data))
                memory = workspace / "memory"
                if memory.exists() or memory.is_symlink():
                    descriptor = directory_fd(memory)
                    try:
                        identities.append((str(memory), identity(os.fstat(descriptor))))
                    finally:
                        os.close(descriptor)
                    _memory(memory, references)
            elif kind == "memory":
                _memory(root, references)
            elif kind == "webui":
                inspect_webui(root, references)
            elif kind == "temporary":
                for path in references.files(root):
                    if path.name != "retained.json":
                        raise MediaRetentionError("unknown temporary cleanup evidence")
                    data = object_json(references.read(path))
                    if (set(data) != {"version", "chats"} or type(data["version"]) is not int
                            or data["version"] != 1 or not isinstance(data["chats"], dict)):
                        raise MediaRetentionError("unsupported temporary cleanup version")
                    for key, entries in cast(dict[str, object], data["chats"]).items():
                        if not key or not isinstance(entries, list):
                            raise MediaRetentionError("invalid temporary cleanup custody")
                        seen: set[str] = set()
                        for entry_value in cast(list[object], entries):
                            if not isinstance(entry_value, dict):
                                raise MediaRetentionError("invalid temporary cleanup entry")
                            entry = cast(dict[str, object], entry_value)
                            raw_path, tokens, asset_generation = entry.get("path"), entry.get("tokens"), entry.get("generation")
                            if (set(entry) != {"path", "tokens", "generation"} or not isinstance(raw_path, str)
                                    or not Path(raw_path).is_absolute() or ".." in Path(raw_path).parts
                                    or not Path(raw_path).is_relative_to(custody.root) or raw_path in seen
                                    or not isinstance(tokens, list) or not tokens
                                    or any(not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token) for token in cast(list[object], tokens))
                                    or not isinstance(asset_generation, list) or len(cast(list[object], asset_generation)) != 5
                                    or any(type(value) is not int for value in cast(list[object], asset_generation))):
                                raise MediaRetentionError("invalid generation-bound temporary custody")
                            seen.add(raw_path)
                            temporary.append(TemporaryMediaCustody(key, Path(raw_path), tuple(cast(list[str], tokens)), tuple(cast(list[int], asset_generation))))
        for session_root, declared in declared_namespaces.items():
            for path in session_root.iterdir():
                if path.name == ".workspace-migration.lock" or path.name.startswith(".session-writer-") and path.suffix == ".lock":
                    if not stat.S_ISREG(path.lstat().st_mode):
                        raise MediaRetentionError("invalid namespace allocation lock")
                    continue
                if path not in declared:
                    raise MediaRetentionError("undeclared session namespace")
        for row in custody.records("pending"):
            pending.append((cast(str, row["token"]), Path(cast(str, row["path"]))))
            # Pending pins are separate from independent live references, so
            # only an exact owner can consume one without hiding other pins.
        custody.records("deletions")  # Validate intents, but do not pin themselves forever.
        attachments = custody.root / "attachments"
        if attachments.exists() or attachments.is_symlink():
            for path in references.files(attachments):
                if path.suffix == ".json":
                    row = object_json(references.read(path))
                    if (set(row) != {"version", "sha256", "mime", "bytes", "created_at"}
                            or type(row["version"]) is not int or row["version"] != 1
                            or not isinstance(row["sha256"], str) or row["sha256"] != path.stem
                            or not _HASH.fullmatch(row["sha256"]) or not isinstance(row["mime"], str)
                            or type(row["bytes"]) is not int or row["bytes"] < 0
                            or type(row["created_at"]) not in {int, float}):
                        raise MediaRetentionError("invalid attachment sidecar")
                    assets[path] = generation(path.lstat())
                else:
                    if not _HASH.fullmatch(path.stem) or path.parent.name != path.stem[:2]:
                        raise MediaRetentionError("unknown attachment blob representation")
                    assets[path] = generation(path.lstat())
        staging = custody.root / ".staging"
        if staging.exists() or staging.is_symlink():
            for path in references.files(staging):
                if not re.fullmatch(r"[0-9a-f]{32}\.[A-Za-z0-9]+", path.name):
                    raise MediaRetentionError("unknown attachment staging record")
        generated_roots = {Path(cast(str, row["path"])) for row in declarations if row["kind"] == "generated"}
        standard_generated = custody.root / "generated"
        if standard_generated.exists() or standard_generated.is_symlink():
            generated_roots.add(standard_generated)
        for generated in sorted(generated_roots):
            for sidecar in references.files(generated):
                if sidecar.suffix != ".json":
                    assets[sidecar] = generation(sidecar.lstat())
                    continue
                row = object_json(references.read(sidecar))
                required = {"id", "kind", "name", "path", "mime", "bytes", "sha256", "prompt", "model", "provider", "source_images", "created_at"}
                if (set(row) != required or row.get("kind") not in {"image", "video", "file"}
                        or not isinstance(row["path"], str) or Path(row["path"]).with_suffix(".json") != sidecar
                        or type(row["bytes"]) is not int or row["bytes"] < 0
                        or not isinstance(row["sha256"], str) or not _HASH.fullmatch(row["sha256"])
                        or not isinstance(row["source_images"], list)
                        or any(not isinstance(row[name], str) for name in ("id", "name", "mime", "prompt", "model", "provider", "created_at"))):
                    raise MediaRetentionError("invalid generated media sidecar")
                references.scan(cast(list[object], row["source_images"]), key="source_images")
                assets[sidecar] = generation(sidecar.lstat())
        for claim in temporary:
            if claim.path.exists() or claim.path.is_symlink():
                info = claim.path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise MediaRetentionError("invalid temporary media generation")
                assets[claim.path] = generation(info)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        reasons.append(f"{type(exc).__name__}: {exc}")
    return MediaReferenceEvidence(frozenset(references.hashes), frozenset(references.paths),
                                  not reasons, tuple(reasons), tuple(identities),
                                  cast(int, registry["generation"]), tuple(pending), tuple(temporary),
                                  tuple(assets.items()))
