"""Extract durable rollout lessons and retrieve workspace memory with FTS5."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal, cast

from kohaku.agent.tools.base import Tool, ToolExposure, ToolResult
from kohaku.context.budget import TokenMeter
from kohaku.context_repo.catalog import RepositoryCatalog
from kohaku.runtime_context import public_history_message
from kohaku.security.workspace_access import WORKSPACE_SCOPE_METADATA_KEY
from kohaku.session.history_visibility import is_hidden_history_message
from kohaku.utils.prompt_templates import render_template
from kohaku.utils.secret_scan import PLAINTEXT_SECRET_PATTERNS
from kohaku.utils.workspace_lock import open_workspace_memory_lock

if TYPE_CHECKING:
    from kohaku.agent.memory import MemoryStore
    from kohaku.context_repo.policy import ResourceAccessPolicy
    from kohaku.session.manager import Session, SessionManager
    from kohaku.utils.llm_runtime import LLMRuntime

MEMORY_SUMMARY_MARKER = "<!-- kohaku-memory-summary: v1 -->"
_ROLLOUT_MARKER = re.compile(r"^<!-- rollout: ([a-f0-9]{24}) -->$", re.MULTILINE)
_PRIVATE_PREFIXES = ("dream:", "delegated:", "child:", "websocket:temp", "ephemeral:")
_ASSIGNED_SECRET = re.compile(
    r"""(?i)\b(api[_-]?key|password|passwd|token|secret|credential)\s*["']?\s*[:=]\s*["']?([^\s"'\x60,;]+)"""
)
_KEY_BLOCK = re.compile(r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?-----END [^-\n]*PRIVATE KEY-----", re.S)
_SCOPE = Literal["all", "memory", "rollouts", "sessions", "knowledge"]


def redact_memory_text(text: str) -> str:
    """Remove recognizable credentials before model dispatch, storage, or indexing."""
    text = _KEY_BLOCK.sub("[redacted private key]", text)
    for pattern in PLAINTEXT_SECRET_PATTERNS:
        text = pattern.sub(lambda match: (
            match.group(1) + "[redacted]" if match.lastindex == 2 else "[redacted]"
        ), text)
    return _ASSIGNED_SECRET.sub(
        lambda match: match[0] if match.group(2) == "[redacted]"
        else f"{match.group(1)}=[redacted]", text,
    )


def atomic_memory_text(path: Path, text: str) -> None:
    """Publish UTF-8 text durably through a temporary file in the same directory."""
    if path.is_symlink():
        raise PermissionError("Memory files must use their original paths.")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if os.name != "nt":
            mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
            os.chmod(temporary, mode)
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _read_text(path: Path, maximum: int = 256_000) -> str:
    with path.open("r", encoding="utf-8") as stream:
        return stream.read(maximum)


def _visible_text(session: Session) -> list[str]:
    rows: list[str] = []
    for event in session.messages:
        role = event.get("role")
        if role not in ("user", "assistant") or is_hidden_history_message(event) or event.get("_command"):
            continue
        if event.get("_parent_stream_id"):
            continue
        message = public_history_message(event)
        body: object = message.get("content")
        if isinstance(body, str) and body.strip():
            rows.append(f"{role}: {redact_memory_text(body)}")
    return rows


@dataclass
class MemoryState:
    processed_revisions: dict[str, int] = field(default_factory=dict)
    processed_digests: dict[str, str] = field(default_factory=dict)
    pending_rollouts: dict[str, str] = field(default_factory=dict)
    last_phase2_at: str | None = None


@dataclass(frozen=True)
class Rollout:
    key: str
    stamp: str
    revision: int
    path: str
    evidence: str

    @property
    def digest(self) -> str:
        return sha256(self.evidence.encode()).hexdigest()


class MemoryPipeline:
    """Own rollout watermarks and the handoff to the fenced Dream operation."""

    def __init__(self, store: MemoryStore, *, threshold: int = 5, max_rollouts: int = 200) -> None:
        self.store = store
        self.workspace = store.workspace
        self.root = store.memory_dir
        self.state_path = self.root / "state.json"
        self.raw_path = self.root / "raw_memories.md"
        self.rollout_path = self.root / "rollout_summaries"
        self.threshold = threshold
        self.max_rollouts = max_rollouts
        self._extract = asyncio.Lock()
        self._consolidate = asyncio.Lock()

    def _validate_storage(self) -> None:
        if (
            self.root.resolve() != self.root
            or any(path.is_symlink() for path in (self.state_path, self.raw_path, self.rollout_path))
        ):
            raise PermissionError("Memory storage must remain in the workspace.")

    def _state(self) -> MemoryState:
        self._validate_storage()
        if not self.state_path.exists():
            return MemoryState()
        data: object = json.loads(_read_text(self.state_path, 2_000_000))
        if not isinstance(data, dict):
            raise ValueError("Memory watermarks are invalid.")
        row = cast(dict[object, object], data)
        revisions: dict[str, int] = {}
        pending: dict[str, str] = {}
        raw_revisions = row.get("processed_revisions")
        if isinstance(raw_revisions, dict):
            for key, value in cast(dict[object, object], raw_revisions).items():
                if isinstance(key, str) and type(value) is int and value >= 0:
                    revisions[key] = value
        raw_pending = row.get("pending_rollouts")
        if isinstance(raw_pending, dict):
            for key, value in cast(dict[object, object], raw_pending).items():
                if isinstance(key, str) and isinstance(value, str) and re.fullmatch(
                    r"[0-9]{8}-[a-z0-9-]+\.md", value
                ):
                    pending[key] = value
        digests: dict[str, str] = {}
        raw_digests = row.get("processed_digests")
        if isinstance(raw_digests, dict):
            for key, value in cast(dict[object, object], raw_digests).items():
                if isinstance(key, str) and isinstance(value, str):
                    digests[key] = value
        at = row.get("last_phase2_at")
        return MemoryState(revisions, digests, pending, at if isinstance(at, str) else None)

    def _save(self, state: MemoryState) -> None:
        atomic_memory_text(self.state_path, json.dumps(asdict(state), ensure_ascii=False, sort_keys=True))

    @staticmethod
    def capture_rollout(session: Session) -> Rollout | None:
        if not session.policy.persist or session.key.startswith(_PRIVATE_PREFIXES):
            return None
        evidence = _visible_text(session)
        if not evidence or not any(row.startswith("user:") for row in evidence):
            return None
        stamp = sha256(f"{session.key}/{session.incarnation}/{session.generation}".encode()).hexdigest()[:16]
        slug = re.sub(r"[^a-z0-9]+", "-", session.key.lower()).strip("-")[:48] or "session"
        key = sha256(f"{stamp}/{session.revision}".encode()).hexdigest()[:24]
        path = f"{session.updated_at:%Y%m%d}-{slug}-{key[:10]}.md"
        return Rollout(key, stamp, session.revision, path, "\n\n".join(evidence)[-56_000:])

    def _eligible(self, rollout: Rollout) -> bool:
        with open_workspace_memory_lock(self.workspace):
            state = self._state()
            # Same evidence at a later revision (e.g. after /compact) is not new evidence.
            if state.processed_digests.get(rollout.stamp) == rollout.digest:
                return False
            return state.processed_revisions.get(rollout.stamp, -1) < rollout.revision

    async def extract_session_rollout(self, rollout: Rollout, runtime: LLMRuntime) -> int:
        """Write at most one extraction per session revision and return the queue size."""
        async with self._extract:
            if not await asyncio.to_thread(self._eligible, rollout):
                return 0
            prompt = render_template("agent/memory_stage1_extract.md", strip=True)
            evidence = TokenMeter.truncate_text(
                rollout.evidence, max(1, min(12_000, runtime.context_window_tokens // 2)),
            )
            async with asyncio.timeout(120):
                response = await runtime.provider.chat_once(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": evidence},
                    ],
                    model=runtime.model,
                    max_tokens=min(1200, runtime.generation.max_tokens),
                    reasoning_effort="none",
                )
            if response.finish_reason != "stop" or response.has_tool_calls or not response.content:
                return 0
            summary = redact_memory_text(response.content).strip()[:4000]
            return await asyncio.to_thread(self._record, rollout, summary)

    def _record(self, rollout: Rollout, summary: str) -> int:
        with open_workspace_memory_lock(self.workspace):
            state = self._state()
            if state.processed_revisions.get(rollout.stamp, -1) >= rollout.revision:
                return len(state.pending_rollouts)
            state.processed_revisions[rollout.stamp] = rollout.revision
            state.processed_digests[rollout.stamp] = rollout.digest
            if not summary or summary == "NO_REUSABLE_MEMORY":
                self._save(state)
                return len(state.pending_rollouts)
            self.rollout_path.mkdir(exist_ok=True)
            atomic_memory_text(self.rollout_path / rollout.path, summary)
            prior = self.raw_path.read_text(encoding="utf-8") if self.raw_path.is_file() else ""
            marker = f"<!-- rollout: {rollout.key} -->"
            if marker not in prior:
                lines = [line.strip() for line in summary.splitlines() if line.strip()]
                bullets = "\n".join(f"- {line.lstrip('- ').strip()}" for line in lines[:30])
                atomic_memory_text(self.raw_path, f"{prior.rstrip()}\n\n{marker}\n{bullets}\n".lstrip())
            state.pending_rollouts[rollout.key] = rollout.path
            self._save(state)
            return len(state.pending_rollouts)

    def _prepare(self) -> tuple[dict[str, str], str, str] | None:
        with open_workspace_memory_lock(self.workspace):
            pending = self._state().pending_rollouts
            if not pending:
                return None
            raw = _read_text(self.raw_path) if self.raw_path.is_file() else ""
        files = "\n".join(f"- memory/rollout_summaries/{name}" for name in pending.values())
        prompt = (
            self.store.default_dream_prompt()
            + "\n\n## Pending rollout files\n" + files
            + "\n\n## Extracted evidence\n" + TokenMeter.truncate_text(redact_memory_text(raw), 8000)
        )
        # Rollout consolidation has no history range to advance; versioning is
        # initialized lazily so /dream works on a workspace never synced from templates.
        self.store.git.init()
        cursor = self.store.get_last_dream_cursor()
        operation = self.store.reserve_dream_run(cursor, cursor, allow_empty_batch=True)
        return pending, prompt, operation

    async def consolidate_memories(
        self, run: Callable[[str, str], Awaitable[object | None]],
    ) -> str | None:
        """Consolidate pending rollouts using the workspace's existing Dream fence."""
        async with self._consolidate:
            # Reserve the operation before the first cancellable await.
            batch = self._prepare()
            if batch is None:
                return None
            pending, prompt, operation = batch
            complete = False
            try:
                response = await run(prompt, operation)
                complete = self.store.dream_run_completed(response)
                if complete:
                    complete = await asyncio.to_thread(self.store.bound_phase2_outputs, operation)
            finally:
                diff = self.store.dream_content_diff()
                self.store.finish_dream_run(
                    operation, completed=complete,
                    commit_message=self.store.build_dream_commit_message("dream: rollout consolidation", diff),
                )
            if not complete:
                return "Dream did not complete. Pending rollouts are available for retry."
            await asyncio.to_thread(self._finish, pending)
            return f"Dream consolidated {len(pending)} rollout(s)." + (f"\n\n{diff}" if diff else "")

    def _finish(self, processed: Mapping[str, str]) -> None:
        with open_workspace_memory_lock(self.workspace):
            state = self._state()
            # Persist progress before removing evidence. A crash can leave old raw
            # blocks, but it cannot lose unprocessed rollouts.
            for key in processed:
                state.pending_rollouts.pop(key, None)
            state.last_phase2_at = datetime.now(timezone.utc).isoformat()
            self._save(state)
            raw = self.raw_path.read_text(encoding="utf-8") if self.raw_path.is_file() else ""
            markers = list(_ROLLOUT_MARKER.finditer(raw))
            remaining = [
                raw[marker.start():markers[index + 1].start() if index + 1 < len(markers) else len(raw)]
                for index, marker in enumerate(markers)
                if marker[1] in state.pending_rollouts
            ]
            atomic_memory_text(self.raw_path, "".join(remaining))
            retained = sorted(self.rollout_path.glob("*.md"), reverse=True)
            for path in retained[self.max_rollouts:]:
                if path.name not in state.pending_rollouts.values():
                    path.unlink()


@dataclass(frozen=True)
class _SearchSource:
    scope: str
    key: str
    signature: str
    read: Callable[[], str]


class MemorySearchTool(Tool):
    """Search durable memory and authorized sources by relevance."""

    exposure = ToolExposure.DIRECT

    def __init__(self, workspace: Path, sessions: SessionManager, policy: ResourceAccessPolicy):
        self.workspace = workspace.resolve()
        self.sessions = sessions
        self.policy = policy
        self.catalog = RepositoryCatalog.for_workspace(workspace, sessions_root=sessions.sessions_dir)

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return "Search memory, rollouts, past sessions, and linked knowledge."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object", "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "scope": {"type": "string", "enum": ["all", "memory", "rollouts", "sessions", "knowledge"]},
                "limit": {"type": "integer"},
            },
        }

    @property
    def read_only(self) -> bool:
        return True

    def _allowed(self, path: Path) -> bool:
        from kohaku.security.folder_grants import FolderGrantsUnavailableError

        try:
            self.policy.validate_path(path, "read")
        except (OSError, ValueError, RuntimeError):
            return False
        if self.policy.restrict_to_workspace:
            resolved = path.resolve(strict=False)
            allowed_roots = (self.workspace.resolve(strict=False), *self.policy.context_manager.read_roots())
            if not any(resolved.is_relative_to(root.resolve(strict=False)) for root in allowed_roots):
                view = self.policy.folder_grants
                if not self.policy.folder_grants_allowed or view is None:
                    return False
                try:
                    if view.effective_grant_for(resolved) is None:
                        return False
                except FolderGrantsUnavailableError:
                    return False
        return True

    def _file(self, scope: str, path: Path, root: Path) -> _SearchSource | None:
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not self._allowed(resolved) or not resolved.is_file():
                return None
            info = resolved.stat()
        except OSError:
            return None
        return _SearchSource(scope, str(resolved), f"{info.st_mtime_ns}:{info.st_size}",
                             lambda: _read_text(resolved) if self._allowed(resolved) else "")

    def _session_text(self, key: str) -> str:
        session = self.sessions.read_session_snapshot(key)
        if session is None:
            return ""
        rows = _visible_text(session)
        backend = self.sessions.source_backend(key)
        if backend is not None:
            checkpoint = backend.source.load_source_snapshot(key).checkpoint
            if checkpoint is not None and isinstance(summary := checkpoint.get("summary"), str):
                rows.append(redact_memory_text(summary))
        return "\n\n".join(rows)[-256_000:]

    def _sources(self, scope: _SCOPE) -> Iterator[_SearchSource]:
        memory = self.workspace / "memory"
        if scope in ("all", "memory") and (source := self._file(
            "memory", memory / "MEMORY.md", self.workspace,
        )):
            yield source
        if scope in ("all", "rollouts"):
            for path in (memory / "rollout_summaries").glob("*.md"):
                if source := self._file("rollouts", path, self.workspace):
                    yield source
        if scope in ("all", "knowledge"):
            for entry in self.catalog.entries():
                if entry.kind != "knowledge":
                    continue
                root = Path(entry.path).resolve()
                if self._allowed(root):
                    for path in root.rglob("*.md"):
                        if not any(part.startswith(".") for part in path.relative_to(root).parts):
                            if source := self._file("knowledge", path, root):
                                yield source
        if scope in ("all", "sessions"):
            for row in self.sessions.list_sessions():
                key: object = row.get("key")
                if not isinstance(key, str) or key.startswith(_PRIVATE_PREFIXES):
                    continue
                meta = self.sessions.read_session_metadata(key)
                if meta is None:
                    continue
                metadata: object = meta.get("metadata")
                project = (
                    cast(dict[object, object], metadata).get(WORKSPACE_SCOPE_METADATA_KEY)
                    if isinstance(metadata, dict) else None
                )
                if isinstance(project, dict):
                    path = cast(dict[object, object], project).get("project_path")
                    if isinstance(path, str) and not self._allowed(Path(path)):
                        continue
                yield _SearchSource(
                    "sessions", key, str(row.get("updated_at") or ""),
                    lambda key=key: self._session_text(key),
                )

    def _search(self, query: str | None, scope: _SCOPE, limit: int) -> str:
        root = self.workspace / "memory"
        index = root / "memory_index.sqlite"
        if root.is_symlink() or index.is_symlink():
            return ToolResult.error("Memory index path is unavailable.")
        root.mkdir(exist_ok=True)
        sources = {(source.scope, source.key): source for source in self._sources(scope)}
        with closing(sqlite3.connect(index, timeout=10)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS signatures (scope, source, stamp, PRIMARY KEY(scope, source))")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS content USING fts5(scope UNINDEXED, source UNINDEXED, body)")
            indexed: dict[tuple[str, str], str] = {
                (kind, key): stamp for kind, key, stamp in db.execute("SELECT * FROM signatures")
            }
            for kind, key in indexed.keys() - sources.keys():
                if scope in ("all", kind):
                    db.execute("DELETE FROM content WHERE scope=? AND source=?", (kind, key))
                    db.execute("DELETE FROM signatures WHERE scope=? AND source=?", (kind, key))
            for (kind, key), source in sources.items():
                if indexed.get((kind, key)) == source.signature:
                    continue
                text = redact_memory_text(source.read())
                sections = re.split(r"(?m)(?=^#{2,3} )", text)
                chunks = [part[i:i + 1800] for part in sections for i in range(0, len(part), 1600)]
                db.execute("DELETE FROM content WHERE scope=? AND source=?", (kind, key))
                db.executemany("INSERT INTO content VALUES(?,?,?)",
                               [(kind, key, chunk) for chunk in chunks if chunk.strip()])
                db.execute("INSERT OR REPLACE INTO signatures VALUES(?,?,?)", (kind, key, source.signature))
            if query is None:
                return "[]"
            rows = db.execute(
                "SELECT scope, source, snippet(content,2,'','',' … ',45) FROM content "
                "WHERE content MATCH ? AND (?='all' OR scope=?) ORDER BY bm25(content), source",
                (query, scope, scope),
            )
            hits: list[dict[str, str]] = []
            for kind, key, fragment in rows:
                if (kind, key) not in sources:
                    continue
                hit = {"scope": kind, "source": key, "snippet": fragment[:400]}
                candidate = json.dumps([*hits, hit], ensure_ascii=False)
                if TokenMeter.count_text(candidate) > 2000:
                    break
                hits.append(hit)
                if len(hits) == limit:
                    break
        return json.dumps(hits, ensure_ascii=False)

    async def refresh_dream_index(self) -> None:
        """Refresh the handbook and rollouts after Dream publishes them."""
        for scope in ("memory", "rollouts"):
            await asyncio.to_thread(self._search, None, scope, 0)

    async def execute(self, query: str, scope: _SCOPE = "all", limit: int = 8) -> str:
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold())))[:30]
        if not terms:
            return ToolResult.error("Enter a memory search query.")
        match = " AND ".join(f'"{term}"' for term in terms)
        try:
            return await asyncio.to_thread(self._search, match, scope, max(1, min(20, limit)))
        except (OSError, ValueError, sqlite3.Error):
            return ToolResult.error("Memory search could not read the index.")
