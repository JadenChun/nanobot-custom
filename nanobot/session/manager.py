"""Session management for conversation history."""

import base64
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import uuid
from collections import OrderedDict
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Generator,
    Literal,
    Protocol,
    TypedDict,
    cast,
)
from weakref import WeakValueDictionary

from filelock import FileLock
from loguru import logger

from nanobot.agent.memory_recovery import MEMORY_RECEIPT_FILENAME, require_memory_receipt_recovery
from nanobot.config.paths import get_default_sessions_root, get_legacy_sessions_dir
from nanobot.providers.base import ProviderConversationState
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_HISTORY_META,
    public_history_message,
)
from nanobot.security.desktop_permissions import (
    DesktopPermissionOwner,
    WorkspacePermissionConflictError,
)
from nanobot.session.history_visibility import HIDDEN_HISTORY_META, is_hidden_history_message
from nanobot.session.model_selection import SESSION_MODEL_PRESET_METADATA_KEY
from nanobot.session.persistence_metadata import (
    BUSY_INBOX_METADATA_KEY,
    DURABLE_METADATA_KEYS,
    accepted_metadata_updates,
    is_busy_recovery_key,
)
from nanobot.session.summary import SUMMARY_CONTINUATION_TEXT
from nanobot.utils.helpers import (
    content_with_media_breadcrumbs,
    ensure_dir,
    estimate_message_tokens,
    find_legal_message_start,
    recent_message_start_index,
    safe_filename,
    strip_think,
)
from nanobot.utils.subagent_channel_display import scrub_subagent_announce_body
from nanobot.workspace_access_recovery import (
    WorkspaceAccessRecord,
    WorkspaceAccessSourceBinding,
    WorkspaceScopeMetadata,
    parse_workspace_access_record,
)

if TYPE_CHECKING:
    from nanobot.session.artifact_operator import CopiedArtifactReadOperations
    from nanobot.session.execution_owner import ExecutionOwner
    from nanobot.session.mode_storage import ManagedNamespaceAccess
    from nanobot.session.session_export import CanonicalExportOperations
    from nanobot.session.session_mode import ResolvedSessionMode, SessionModeResolver
    from nanobot.session.source_root import SourceRootAuthority
    from nanobot.session.source_store import SourceSessionStore, SourceSnapshot


SESSION_CACHE_MAX_SIZE = 128
_MESSAGE_TIME_PREFIX_RE = re.compile(r"^\[Message Time: [^\]]+\]\n?")
_LOCAL_IMAGE_BREADCRUMB_RE = re.compile(r"^\[image: (?:/|~)[^\]]+\]\s*$")
_TOOL_CALL_ECHO_RE = re.compile(r'^\s*(?:generate_image|message)\([^)]*\)\s*$')
_SESSION_PREVIEW_MAX_CHARS = 120
_SESSION_LIST_PREVIEW_MAX_RECORDS = 200
_SESSION_LIST_PREVIEW_MAX_CHARS = 1_000_000
_SESSION_DATA_ERRORS = (ValueError, TypeError, AttributeError, KeyError)
_RUNTIME_CHECKPOINT_DATA_ERRORS = (OSError, *_SESSION_DATA_ERRORS)
_PROVIDER_STATE_RECORD_TYPE = "provider_state"
_PROVIDER_STATE_RECORD_PREFIX_RE = re.compile(
    r'^\s*\{\s*"_type"\s*:\s*"provider_state"\s*(?:,|\})'
)
_RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
_RUNTIME_CHECKPOINT_VERSION = 2
_RESERVED_DURABLE_METADATA = DURABLE_METADATA_KEYS
_RUNTIME_CHECKPOINT_SUFFIX = ".checkpoint.json"
_FORK_VOLATILE_METADATA_KEYS = {
    BUSY_INBOX_METADATA_KEY,
    "workspace_access_request",
    "goal_state",
    "pending_user_turn",
    "pending_user_followups",
    "runtime_checkpoint",
    "session_handle",
    "webui_recovery",
    "thread_goal",
    "title",
    "title_user_edited",
}
_WORKSPACE_STATE_DIR = ".nanobot"
_WORKSPACE_ID_FILE = "workspace-id"
_WORKSPACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SESSION_MIGRATION_LOCK_TIMEOUT_SECONDS = 30
_SESSION_FILES_LOCK_FILENAME = ".session-files.lock"
_COPY_CHUNK_SIZE = 1024 * 1024


def _json_object(value: object) -> dict[str, Any]:
    """Narrow a decoded JSON object while preserving its original values."""
    if not isinstance(value, dict):
        raise ValueError("session records must be JSON objects")
    return cast(dict[str, Any], value)


def _archive_offset(data: dict[str, Any]) -> int:
    """Read the Memory archive watermark across the field-name migration."""
    for key in ("last_archived", "last_consolidated"):
        offset = cast(object, data.get(key))
        if isinstance(offset, int) and not isinstance(offset, bool):
            return offset
    return 0


# TODO(0.3.2): Remove the write_stdin replay migration after 0.3.1.
def _migrate_legacy_exec_arguments(container: dict[str, Any]) -> bool:
    raw_arguments = cast(object, container.get("arguments"))
    encoded = isinstance(raw_arguments, str)
    if encoded:
        try:
            decoded: object = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return False
    else:
        decoded = raw_arguments
    if not isinstance(decoded, dict):
        return False

    arguments = cast(dict[str, Any], decoded)
    changed = False
    if "chars" in arguments:
        if "input" not in arguments:
            arguments["input"] = arguments["chars"]
        arguments.pop("chars")
        changed = True

    wait_key = (
        "wait_timeout_ms"
        if arguments.get("wait_for") or arguments.get("until_exit")
        else "yield_time_ms"
    )
    if "timeout_ms" not in arguments and wait_key in arguments:
        arguments["timeout_ms"] = arguments[wait_key]
    for key in ("yield_time_ms", "wait_timeout_ms", "max_output_chars", "max_output_tokens"):
        if key in arguments:
            arguments.pop(key)
            changed = True

    if changed:
        container["arguments"] = (
            json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            if encoded
            else arguments
        )
    return changed


def _migrate_legacy_exec_tool_call(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    tool_call = cast(dict[str, Any], value)
    function_value = cast(object, tool_call.get("function"))
    function = (
        cast(dict[str, Any], function_value)
        if isinstance(function_value, dict)
        else tool_call
    )
    name = function.get("name")
    if name not in {"write_stdin", "exec_session"}:
        return False

    changed = name == "write_stdin"
    if changed:
        function["name"] = "exec_session"
    return _migrate_legacy_exec_arguments(function) or changed


def _migrate_legacy_exec_message(message: dict[str, Any]) -> bool:
    changed = False
    if message.get("name") == "write_stdin":
        message["name"] = "exec_session"
        changed = True
    tool_calls = cast(object, message.get("tool_calls"))
    if isinstance(tool_calls, list):
        for tool_call in cast(list[object], tool_calls):
            changed = _migrate_legacy_exec_tool_call(tool_call) or changed
    return changed


def _migrate_legacy_exec_session_records(
    messages: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> bool:
    changed = False
    for message in messages:
        changed = _migrate_legacy_exec_message(message) or changed

    checkpoint_value = cast(object, metadata.get(_RUNTIME_CHECKPOINT_KEY))
    if not isinstance(checkpoint_value, dict):
        return changed
    checkpoint = cast(dict[str, Any], checkpoint_value)
    assistant = cast(object, checkpoint.get("assistant_message"))
    if isinstance(assistant, dict):
        changed = _migrate_legacy_exec_message(cast(dict[str, Any], assistant)) or changed
    pending = cast(object, checkpoint.get("pending_tool_calls"))
    if isinstance(pending, list):
        for tool_call in cast(list[object], pending):
            changed = _migrate_legacy_exec_tool_call(tool_call) or changed
    completed = cast(object, checkpoint.get("completed_tool_results"))
    if isinstance(completed, list):
        for result in cast(list[object], completed):
            if isinstance(result, dict):
                result_data = cast(dict[str, Any], result)
                if result_data.get("name") == "write_stdin":
                    result_data["name"] = "exec_session"
                    changed = True
    return changed


def _is_provider_state_record_line(line: str) -> bool:
    """Recognize the canonical private record without decoding its opaque payload."""
    return _PROVIDER_STATE_RECORD_PREFIX_RE.match(line) is not None


def _sanitize_assistant_replay_text(content: str) -> str:
    """Remove internal replay artifacts that the model may have copied before.

    These strings are useful as runtime/session metadata, but when they appear
    in assistant examples they become demonstrations for the model to repeat.
    """
    content = _MESSAGE_TIME_PREFIX_RE.sub("", content, count=1)
    lines = [
        line
        for line in content.splitlines()
        if not _LOCAL_IMAGE_BREADCRUMB_RE.match(line)
        and not _TOOL_CALL_ECHO_RE.match(line)
    ]
    return "\n".join(lines).strip()


def _text_preview(content: object) -> str:
    """Return compact display text for session lists."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in cast(list[object], content):
            if isinstance(block, dict):
                block_data = cast(dict[object, object], block)
                if block_data.get("type") != "text":
                    continue
                value = block_data.get("text")
                if isinstance(value, str):
                    parts.append(value)
        text = " ".join(parts)
    else:
        return ""
    text = _sanitize_assistant_replay_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _SESSION_PREVIEW_MAX_CHARS:
        text = text[: _SESSION_PREVIEW_MAX_CHARS - 1].rstrip() + "…"
    return text


def _message_preview_text(message: dict[str, Any]) -> str:
    """Session list preview text; subagent inject blobs are shortened for display."""
    message = public_history_message(message)
    content = cast(object, message.get("content"))
    if message.get("injected_event") == "subagent_result" and isinstance(content, str):
        content = scrub_subagent_announce_body(content)
    return _text_preview(content)


def _metadata_title(metadata: object) -> str:
    if not isinstance(metadata, dict):
        return ""
    metadata_data = cast(dict[object, object], metadata)
    title = metadata_data.get("title")
    if not isinstance(title, str):
        return ""
    if metadata_data.get("title_user_edited") is True:
        return title
    return strip_think(title)


@dataclass(frozen=True)
class SessionPolicy:
    """Runtime rules that do not belong in durable session data."""

    persist: bool = True
    log_content: bool = True
    disabled_tools: frozenset[str] = frozenset()


@dataclass
class Session:
    """A conversation session."""

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Keep the legacy storage name while persisted sessions and SDK callers migrate.
    last_consolidated: int = 0
    provider_state: ProviderConversationState | None = field(default=None, repr=False)
    policy: SessionPolicy = field(default_factory=SessionPolicy, repr=False, compare=False)

    revision: int = 0
    generation: int = 0
    incarnation: str = field(default_factory=lambda: secrets.token_hex(16))
    _base_messages: list[dict[str, Any]] | None = field(default=None, repr=False, compare=False)
    _base_metadata: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _base_provider_state: ProviderConversationState | None = field(default=None, repr=False, compare=False)
    _base_revision: int = field(default=0, repr=False, compare=False)
    _base_generation: int = field(default=0, repr=False, compare=False)
    _base_incarnation: str | None = field(default=None, repr=False, compare=False)
    _replacement_pending: bool = field(default=False, repr=False, compare=False)
    _projection_edit_pending: bool = field(default=False, repr=False, compare=False)
    _append_ids: dict[tuple[int, int], tuple[dict[str, Any], str]] = field(default_factory=dict, repr=False, compare=False)

    def _remember_committed_base(self) -> None:
        self._base_messages = deepcopy(self.messages)
        self._base_metadata = deepcopy(self.metadata)
        self._base_provider_state = deepcopy(self.provider_state)
        self._base_revision = self.revision
        self._base_generation = self.generation
        self._base_incarnation = self.incarnation
        self._replacement_pending = False
        self._projection_edit_pending = False
        self._append_ids.clear()

    def record_committed_base(self) -> None:
        self._remember_committed_base()

    def prepare_legacy_commit(self, disk: "Session | None") -> "Session":
        base = self._base_messages
        if disk is None:
            if base is not None:
                raise SessionWriteConflict()
        elif (
            base is None
            or disk.generation != self._base_generation
            or disk.incarnation != self._base_incarnation
        ):
            raise SessionWriteConflict()
        if self.metadata.get("source_schema_version") is not None or (
            disk is not None and disk.metadata.get("source_schema_version") is not None
        ):
            raise SessionWriteConflict()
        candidate = deepcopy(self)
        if self._replacement_pending or self._projection_edit_pending:
            if self._replacement_pending and disk is not None:
                from nanobot.session.busy_inbox import require_busy_inbox_disposition

                require_busy_inbox_disposition(disk.metadata)
            if disk is not None and disk.revision != self._base_revision:
                raise SessionWriteConflict()
            candidate.generation = (disk.generation if disk else 0) + int(self._replacement_pending)
        else:
            base = base or []
            if self.messages[:len(base)] != base or len(self.messages) < len(base):
                raise SessionWriteConflict()
            if disk is not None and disk.messages[:len(base)] != base:
                raise SessionWriteConflict()
            candidate.messages = deepcopy(disk.messages if disk else [])
            by_id = {row.get("_event_id"): row for row in candidate.messages if row.get("_event_id")}
            for position, row in enumerate(self.messages[len(base):], start=len(base)):
                # Keep identity stable across a failed atomic-write retry without
                # mutating the caller's event or committed revision bookkeeping.
                append_key = (position, id(row))
                pending = self._append_ids.get(append_key)
                if pending is None or pending[0] is not row:
                    pending = (row, uuid.uuid4().hex)
                    self._append_ids[append_key] = pending
                identity = pending[1]
                addition = deepcopy(row)
                addition["_event_id"] = identity
                if identity in by_id:
                    if by_id[identity] != addition:
                        raise SessionWriteConflict()
                else:
                    candidate.messages.append(addition)
                    by_id[identity] = addition
            candidate.generation = disk.generation if disk else 0
        if disk is not None:
            candidate.metadata = _merge_session_metadata(self, disk)
            candidate.incarnation = disk.incarnation
            candidate.created_at = disk.created_at
            current_checkpoint = self.metadata.get(_RUNTIME_CHECKPOINT_KEY)
            base_checkpoint = self._base_metadata.get(_RUNTIME_CHECKPOINT_KEY)
            disk_checkpoint = disk.metadata.get(_RUNTIME_CHECKPOINT_KEY)
            if not self._replacement_pending and current_checkpoint != base_checkpoint:
                if disk_checkpoint not in (base_checkpoint, current_checkpoint):
                    raise SessionWriteConflict()
                if disk.messages != (self._base_messages or []) and current_checkpoint != disk_checkpoint:
                    raise SessionWriteConflict()
            if (
                not self._replacement_pending
                and (
                    disk.messages != (self._base_messages or [])
                    or disk_checkpoint != base_checkpoint
                )
                and len(candidate.messages) > len(disk.messages)
                and candidate.metadata.get(_RUNTIME_CHECKPOINT_KEY) is not None
            ):
                # Never discard dispatch evidence to make two branches merge.
                raise SessionWriteConflict()
            candidate.updated_at = max(self.updated_at, disk.updated_at, key=lambda value: value.timestamp())
            if not self._replacement_pending:
                if self.provider_state == self._base_provider_state:
                    candidate.provider_state = deepcopy(disk.provider_state)
                elif disk.provider_state not in (self._base_provider_state, self.provider_state):
                    raise SessionWriteConflict()
                if disk.messages != (self._base_messages or []):
                    if (
                        self.provider_state != self._base_provider_state
                        and self.provider_state != disk.provider_state
                    ):
                        raise SessionWriteConflict()
                    if len(candidate.messages) > len(disk.messages):
                        # Independent transcript branches cannot share a provider handle.
                        candidate.provider_state = None
            if not self._replacement_pending and not self._projection_edit_pending:
                candidate.last_consolidated = max(disk.last_consolidated, self.last_consolidated)
        else:
            candidate.metadata = {
                key: deepcopy(value) for key, value in self.metadata.items()
                if key not in _RESERVED_DURABLE_METADATA
            }
        if self._replacement_pending:
            for key in ("context_checkpoint", "runtime_checkpoint", "_last_summary"):
                candidate.metadata.pop(key, None)
            candidate.provider_state = None
        candidate.revision = (disk.revision if disk else 0) + 1
        return candidate

    def publish_legacy_commit(self, candidate: "Session") -> None:
        self.messages = candidate.messages
        self.metadata = candidate.metadata
        self.revision = candidate.revision
        self.generation = candidate.generation
        self.incarnation = candidate.incarnation
        self.created_at = candidate.created_at
        self.updated_at = candidate.updated_at
        self.last_consolidated = candidate.last_consolidated
        self.provider_state = candidate.provider_state
        self._remember_committed_base()

    def require_checkpoint_base(self, disk: "Session | None") -> "Session":
        if disk is None or disk.metadata.get("source_schema_version") is not None or (
            disk.revision != self._base_revision
            or disk.generation != self._base_generation
            or disk.incarnation != self._base_incarnation
        ):
            raise SessionWriteConflict()
        return disk

    def prepare_checkpoint_write(self, disk: "Session") -> object:
        checkpoint = self.metadata.get(_RUNTIME_CHECKPOINT_KEY)
        if disk.metadata.get(_RUNTIME_CHECKPOINT_KEY) not in (
            self._base_metadata.get(_RUNTIME_CHECKPOINT_KEY), checkpoint,
        ) or disk.provider_state not in (self._base_provider_state, self.provider_state):
            raise SessionWriteConflict()
        return checkpoint

    def publish_checkpoint_write(self, checkpoint: object) -> None:
        self._base_metadata[_RUNTIME_CHECKPOINT_KEY] = deepcopy(checkpoint)
        self._base_provider_state = deepcopy(self.provider_state)

    def publish_committed_metadata_update(self, accepted: dict[str, Any]) -> None:
        self.metadata.update(accepted)
        # These changes already committed. Do not replay them as local
        # modifications over a later writer when this cached object saves.
        self._base_metadata.update(deepcopy(accepted))

    def publish_committed_metadata_upsert(
        self,
        accepted: dict[str, Any],
        accepted_removals: Collection[str],
        load_committed: Callable[[str], "Session | None"],
    ) -> bool:
        if self._base_messages is None:
            # Bind a fresh cached caller to the newly committed empty
            # baseline without treating its local history as committed.
            # This also handles retry after a creation's metadata failure.
            committed = load_committed(self.key)
            if committed is None:
                return False
            self.revision = self._base_revision = committed.revision
            self.generation = self._base_generation = committed.generation
            self.incarnation = self._base_incarnation = committed.incarnation
            self._base_messages = deepcopy(committed.messages)
            self._base_metadata = deepcopy(committed.metadata)
            self._base_provider_state = deepcopy(committed.provider_state)
        self.metadata.update(deepcopy(accepted))
        self._base_metadata.update(deepcopy(accepted))
        for name in accepted_removals:
            self.metadata.pop(name, None)
            self._base_metadata.pop(name, None)
        return True

    def merge_commit_metadata(self, disk: "Session") -> dict[str, Any]:
        merged = deepcopy(disk.metadata)
        base = self._base_metadata
        for key in base.keys() | self.metadata.keys():
            if key in _RESERVED_DURABLE_METADATA:
                continue
            if key not in self.metadata:
                if key in base and disk.metadata.get(key) == base[key]:
                    merged.pop(key, None)
            elif key not in base or self.metadata[key] != base[key]:
                merged[key] = deepcopy(self.metadata[key])
        key = Session.SCHEDULED_RING_KEY
        if self.metadata.get(key) != base.get(key):
            ring = disk.recent_scheduled_runs()
            known = {item.get("run_id") for item in ring}
            for item in self.recent_scheduled_runs():
                if item.get("run_id") not in known:
                    ring.append(deepcopy(item))
                    known.add(item.get("run_id"))
            merged[key] = ring[-Session.SCHEDULED_RING_LIMIT:]
        return merged

    def replace_messages(self, messages: list[dict[str, Any]]) -> None:
        """Request a revision-checked transcript replacement on the next save."""
        self.messages = deepcopy(messages)
        self._replacement_pending = True
        self.last_consolidated = 0
        self.provider_state = None
        for key in ("context_checkpoint", "runtime_checkpoint", "_last_summary"):
            self.metadata.pop(key, None)
        self.updated_at = datetime.now()

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.metadata), dict):
            self.metadata = {}
        if not isinstance(cast(object, self.provider_state), ProviderConversationState):
            self.provider_state = None
        # An out-of-range offset (corrupt metadata) would hide all history; reset it.
        last_consolidated = cast(object, self.last_consolidated)
        if (
            isinstance(last_consolidated, bool)
            or not isinstance(last_consolidated, int)
            or not 0 <= last_consolidated <= len(self.messages)
        ):
            self.last_consolidated = 0

    @property
    def last_archived(self) -> int:
        """End of the latest committed Memory checkpoint."""
        return self.last_consolidated

    @last_archived.setter
    def last_archived(self, value: int) -> None:
        self.last_consolidated = value

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def commit_summary_checkpoint(
        self,
        summary: str,
        *,
        insert_at: int | None = None,
        last_active: datetime | None = None,
    ) -> None:
        """Replace replay before a hidden boundary while preserving the transcript."""
        self._projection_edit_pending = True
        boundary = len(self.messages) if insert_at is None else insert_at
        self.messages.insert(boundary, {
            "role": "user",
            "content": SUMMARY_CONTINUATION_TEXT,
            HIDDEN_HISTORY_META: True,
            "timestamp": datetime.now().isoformat(),
        })
        self.metadata["_last_summary"] = {
            "text": summary,
            "last_active": (last_active or self.updated_at).isoformat(),
        }
        self.last_archived = boundary

    SCHEDULED_RING_KEY = "_recent_scheduled_runs"
    SCHEDULED_RING_LIMIT = 20
    SCHEDULED_EXCERPT_CHARS = 240
    SCHEDULED_PROJECTION_CHARS = 6000

    def add_run_message(self, run_id: str, role: str, content: str, **kwargs: Any) -> None:
        """Add a detail message tagged for a specific run."""
        self.add_message(role, content, _run_id=run_id, **kwargs)

    def get_run_messages(self, run_id: str) -> list[dict[str, Any]]:
        """Return persisted detail messages belonging to ``run_id``.

        Unlike :meth:`get_history`, this is a trace-facing API and therefore
        retains the internal tag and any other persisted detail fields.
        """
        matching = [deepcopy(message) for message in self.messages if message.get("_run_id") == run_id]
        if matching:
            return matching
        # Compatibility for scheduled links stored in the bounded metadata
        # ring.  The full detail remains in the cron execution session; this
        # synthetic view keeps older trace readers source-compatible without
        # appending a second instruction/result pair to visible history.
        for item in self.recent_scheduled_runs():
            if item.get("run_id") == run_id:
                return [
                    {"role": "user", "content": item.get("instruction", ""), "_run_id": run_id},
                    {"role": "assistant", "content": item.get("result", ""), "_run_id": run_id},
                ]
        return []

    def recent_scheduled_runs(self) -> list[dict[object, Any]]:
        """Return the bounded scheduled-awareness metadata ring."""
        value = self.metadata.get(self.SCHEDULED_RING_KEY, [])
        if not isinstance(value, list):
            return []
        # Entries written before approval metadata was introduced remain
        # readable; callers and the projection use ``False`` for the missing
        # field without rewriting the legacy metadata.
        # Legacy metadata may have arbitrary keys; the guards establish only
        # the outer list/dict containers, not a canonical scheduled-record shape.
        return [dict(cast(dict[object, object], item)) for item in cast(list[object], value)
                if isinstance(item, dict)][-self.SCHEDULED_RING_LIMIT:]

    def add_scheduled_run(
        self, *, job_id: str, run_id: str, instruction: str, result: str,
        detail_session_key: str, occurrence_id: str | None = None,
        status: str = "completed", approval_granted: bool = False,
        stop_reason: str | None = None,
    ) -> None:
        """Keep bounded scheduled awareness; never mirror the full turn into chat."""
        def excerpt(text: str) -> str:
            text = text.strip()
            if len(text) <= self.SCHEDULED_EXCERPT_CHARS:
                return text
            return text[:self.SCHEDULED_EXCERPT_CHARS - 3].rstrip() + "..."

        entry: dict[object, object] = {
            "job_id": job_id,
            "run_id": run_id,
            "detail_ref": {"session_key": detail_session_key, "run_id": run_id},
            "instruction": excerpt(instruction),
            "result": excerpt(result),
            "status": status,
            "run_status": status,
            "approval_granted": approval_granted,
            **({"occurrence_id": occurrence_id} if occurrence_id else {}),
            **({"stop_reason": stop_reason} if stop_reason else {}),
        }
        ring = self.recent_scheduled_runs()
        for prior in ring:
            if prior.get("run_id") == run_id:
                if any(prior.get(key) != value for key, value in entry.items()):
                    raise SessionWriteConflict()
                return
        entry["timestamp"] = datetime.now().isoformat()
        ring.append(entry)
        self.metadata[self.SCHEDULED_RING_KEY] = ring[-self.SCHEDULED_RING_LIMIT:]
        self.updated_at = datetime.now()

    def get_model_history(
        self, max_messages: int = 0, *, extend_to_user: bool = False,
    ) -> list[dict[str, Any]]:
        """Project scheduled references into model context, not visible messages."""
        history = self.get_history(max_messages, extend_to_user=extend_to_user)
        # Pre-migration chat files may contain full scheduled wrapper pairs.
        # Retain those on disk for trace readers, but avoid teaching the model
        # the same job twice alongside its bounded metadata reference.
        collapsed: list[dict[str, Any]] = []
        index = 0
        while index < len(history):
            current = history[index]
            if (
                current.get("role") == "user"
                and isinstance(current.get("content"), str)
                and current["content"].startswith((
                    "[Scheduled task from cron:", "[Scheduled task approval from cron:",
                ))
                and index + 1 < len(history)
                and history[index + 1].get("role") == "assistant"
            ):
                index += 2
                continue
            collapsed.append(current)
            index += 1
        ring = self.recent_scheduled_runs()
        if not ring:
            return collapsed
        lines = [
            "[Recent Scheduled Runs]",
            "Recent scheduled activity is metadata only; retrieve full detail by run reference.",
            "Completed means the computation finished; it does not prove channel delivery.",
        ]
        for item in reversed(ring):
            ref = item.get("detail_ref") or cast(dict[object, object], {})
            lines.append(
                f"- job={item.get('job_id', '')} run={item.get('run_id', '')} "
                f"status={item.get('status', '')} "
                f"approval_granted={str(bool(item.get('approval_granted', False))).lower()} "
                f"ref={ref.get('session_key', '')}/{ref.get('run_id', '')}"
            )
            if item.get("stop_reason"):
                lines.append(f"  stop_reason: {item['stop_reason']}")
            lines.append(f"  instruction: {item.get('instruction', '')}")
            lines.append(f"  result: {item.get('result', '')}")
        projection = "\n".join(lines)[:self.SCHEDULED_PROJECTION_CHARS]
        return [{"role": "user", "content": projection}] + collapsed

    @staticmethod
    def _strip_internal_metadata(value: Any) -> Any:
        """Remove internal keys recursively from model-facing message data."""
        if isinstance(value, dict):
            return {
                key: Session._strip_internal_metadata(item)
                for key, item in cast(dict[object, object], value).items()
                if not str(key).startswith("_")
            }
        if isinstance(value, list):
            return [Session._strip_internal_metadata(item) for item in cast(list[object], value)]
        if isinstance(value, tuple):
            return tuple(Session._strip_internal_metadata(item) for item in cast(tuple[object, ...], value))
        return value

    messages_for_run = get_run_messages

    def get_history(
        self,
        max_messages: int = 0,
        *,
        max_tokens: int = 0,
        extend_to_user: bool = False,
        include_runtime_context: bool = True,
    ) -> list[dict[str, Any]]:
        """Return recent replayable messages for LLM input.

        A committed summary checkpoint replaces its old prefix with the stored
        summary and resumes replay at a hidden continuation marker. A positive
        ``max_messages`` applies an additional caller-owned count limit.
        """
        replayable = self.messages[self.last_archived:]
        if max_messages <= 0:
            start_idx = 0
        else:
            start_idx = recent_message_start_index(
                replayable,
                max_messages,
                extend_to_user=extend_to_user,
            )
        sliced = replayable[start_idx:]

        # Avoid starting mid-turn when possible, except for proactive
        # assistant deliveries that the user may be replying to.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                start = i
                if i > 0 and sliced[i - 1].get("_channel_delivery"):
                    start = i - 1
                sliced = sliced[start:]
                break

        # Drop orphan tool results at the front.
        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            if message.get("_command"):
                continue
            has_persisted_runtime_context = isinstance(
                message.get(RUNTIME_CONTEXT_HISTORY_META),
                dict,
            )
            if not include_runtime_context:
                message = public_history_message(message)
            content = message.get("content", "")
            role = message.get("role")
            if role == "assistant" and isinstance(content, str):
                content = _sanitize_assistant_replay_text(content)
            # Synthesize an ``[image: path]`` breadcrumb from the persisted
            # ``media`` kwarg so LLM replay still sees *something* where the
            # image used to be. Without this, an image-only user turn
            # replays as an empty user message — the assistant's reply then
            # looks like it's responding to nothing.
            content = content_with_media_breadcrumbs(
                role,
                content,
                message.get("media"),
            )
            cli_apps = cast(object, message.get("cli_apps"))
            if (
                include_runtime_context
                and not has_persisted_runtime_context
                and role == "user"
                and isinstance(cli_apps, list)
                and cli_apps
                and isinstance(content, str)
            ):
                cli_lines: list[str] = []
                for item in cast(list[object], cli_apps[:8]):
                    if not isinstance(item, dict):
                        continue
                    item_data = cast(dict[object, object], item)
                    name = str(item_data.get("name") or "").strip().lower()
                    if not name:
                        continue
                    entry_point = (
                        str(item_data.get("entry_point") or "unknown").strip() or "unknown"
                    )
                    cli_lines.append(
                        f"[CLI App Attachment: @{name}; tool=run_cli_app; entry_point={entry_point}; "
                        f"skill=skills/cli-app-{name}/SKILL.md]"
                    )
                if cli_lines:
                    breadcrumbs = "\n".join(cli_lines)
                    content = f"{content}\n{breadcrumbs}" if content else breadcrumbs
            if role == "assistant" and isinstance(content, str) and not content.strip():
                if not any(key in message for key in ("tool_calls", "reasoning_content", "thinking_blocks")):
                    continue
            entry: dict[str, Any] = {"role": message["role"], "content": content}
            for key in ("tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks"):
                if key in message:
                    entry[key] = message[key]
            out.append(self._strip_internal_metadata(entry))

        if max_tokens > 0 and out:
            kept: list[dict[str, Any]] = []
            used = 0
            for message in reversed(out):
                tokens = estimate_message_tokens(message)
                if kept and used + tokens > max_tokens:
                    break
                kept.append(message)
                used += tokens
            kept.reverse()

            # Keep history aligned to the first visible user turn.
            first_user = next((i for i, m in enumerate(kept) if m.get("role") == "user"), None)
            if first_user is not None:
                kept = kept[first_user:]
            else:
                # Tight token budgets can otherwise leave assistant-only tails.
                # If a user turn exists in the unsliced output, recover the
                # nearest one even if it slightly exceeds the token budget.
                recovered_user = next(
                    (i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "user"),
                    None,
                )
                if recovered_user is not None:
                    kept = out[recovered_user:]

            # And keep a legal tool-call boundary at the front.
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
            out = kept
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.replace_messages([])

class SessionPayload(TypedDict):
    key: str
    created_at: str | None
    updated_at: str | None
    metadata: dict[str, Any]
    messages: list[dict[str, Any]]


class SessionMetadataPayload(TypedDict):
    key: str
    created_at: str | None
    updated_at: str | None
    metadata: dict[str, Any]


class SessionInfo(TypedDict):
    key: str
    created_at: str
    updated_at: str
    title: str
    preview: str
    path: str


@dataclass(frozen=True)
class _SessionFileSnapshot:
    digest: str
    size: int
    mtime_ns: int
    updated_at: float
    device: int
    inode: int


@dataclass(frozen=True)
class SessionRestoreResult:
    restored: int
    unchanged: int
    conflicts: tuple[Path, ...]


class SessionStore(Protocol):
    def load(self, key: str) -> Session | None: ...

    def save(self, session: Session, *, fsync: bool = False) -> None: ...

    def delete(self, key: str) -> bool: ...

    def read(self, key: str) -> SessionPayload | None: ...

    def read_metadata(self, key: str) -> SessionMetadataPayload | None: ...

    def update_metadata(
        self,
        key: str,
        updates: dict[str, Any],
        *,
        fsync: bool = False,
    ) -> bool: ...

    def list_sessions(self) -> list[SessionInfo]: ...


class JsonlSessionStore:
    """JSONL implementation of session persistence."""

    def __init__(
        self,
        workspace: Path,
        *,
        sessions_root: Path | None = None,
        _managed_access: "ManagedNamespaceAccess | None" = None,
    ):
        from nanobot.agent.memory_recovery_startup_admission import (
            admitted_startup_mutation,
            require_startup_receipt_admission,
        )

        canonical_workspace = Path(workspace).expanduser().resolve(strict=False)
        root = (Path(sessions_root).expanduser().resolve(strict=False)
                if sessions_root is not None else get_default_sessions_root())
        from nanobot.session.mode_storage import preflight_store_construction

        preflight_store_construction(canonical_workspace, root, _managed_access)
        receipt_path = canonical_workspace / "memory" / MEMORY_RECEIPT_FILENAME
        require_startup_receipt_admission(canonical_workspace, sessions_root=root)
        if root == canonical_workspace or root.is_relative_to(canonical_workspace):
            raise RuntimeError(
                "session storage must be outside the agent workspace; "
                "move --config outside --workspace or choose a nested workspace directory"
            )
        with admitted_startup_mutation(canonical_workspace, sessions_root=root) as memory_gate:
            if memory_gate is not None:
                memory_gate.require_current()
            require_memory_receipt_recovery(receipt_path)
            ensure_dir(root)
            with suppress(OSError):
                os.chmod(root, 0o700)
            self.workspace = canonical_workspace
            self.sessions_root = root
            self._managed_namespace_access = _managed_access
            self._migration_lock = FileLock(
                str(root / ".workspace-migration.lock"),
                timeout=_SESSION_MIGRATION_LOCK_TIMEOUT_SECONDS,
            )
            with self._migration_lock:
                if memory_gate is not None:
                    memory_gate.require_current()
                require_memory_receipt_recovery(receipt_path)
                workspace_id = self._load_or_create_workspace_id(canonical_workspace, root)
                workspace_id = self._claim_workspace_namespace(
                    root,
                    canonical_workspace,
                    workspace_id,
                )
                self.sessions_dir = ensure_dir(root / workspace_id)
                if _managed_access is not None:
                    _managed_access.bind(self)
                self.legacy_sessions_dir = get_legacy_sessions_dir()
                self._session_files_lock = FileLock(
                    str(self.sessions_dir / _SESSION_FILES_LOCK_FILENAME)
                )
                with self._session_files_lock:
                    if memory_gate is not None:
                        memory_gate.require_current()
                    require_memory_receipt_recovery(receipt_path)
                    if _managed_access is None:
                        self._migrate_from_workspace(canonical_workspace)

    @property
    def source(self) -> "SourceSessionStore":
        from nanobot.session.source_store import SourceSessionStore

        return SourceSessionStore(self)

    @contextmanager
    def locked_session_files(self) -> Generator[Path, None, None]:
        """Guard direct access to canonical session files in this directory."""
        with self._session_files_lock:
            from nanobot.session.mode_storage import require_namespace_write_access

            require_namespace_write_access(self)
            yield self.sessions_dir

    def require_namespace_write_admission(
        self, *, select_namespace_claimed: Callable[[], Callable[[Path, Path, str | None], bool]],
        mode_error: Callable[[], Exception],
    ) -> None:
        """Check this backend's existing access without exposing or replacing it.

        Late helpers retain the mode-storage entry point's lookup order. This
        operation does not acquire locks or admit a constructor/source write.
        """
        access = self._managed_namespace_access
        workspace_id = self.sessions_dir.name
        if select_namespace_claimed()(self.workspace, self.sessions_root, workspace_id):
            if access is None:
                raise mode_error()
            access.require(self)
        elif access is not None:
            access.require(self)

    @property
    def session_files_locked(self) -> bool:
        """Observe this caller's files-lock state without exposing or acquiring it."""
        return self._session_files_lock.is_locked

    @contextmanager
    def locked_namespace_migration(self) -> Generator[None, None, None]:
        """Hold the existing namespace migration lock without handing it to callers."""
        with self._migration_lock:
            yield

    def require_unmanaged_namespace(self) -> None:
        """Reject compatibility operations once any durable mode claim exists."""
        from nanobot.session.mode_storage import ModeStorageError, namespace_claimed

        if namespace_claimed(self.workspace, self.sessions_root, self.sessions_dir.name):
            raise ModeStorageError("managed session namespace denies compatibility operation")

    @contextmanager
    def cancelled_record_guard(self, key: str) -> Generator[bool, None, None]:
        """Fence header-only unmanaged eligibility through an insert-only run write."""
        # Cancellation must not synchronously wait for a cooperating converter.
        with self._session_files_lock.acquire(timeout=0), self.locked_session_files():
            self.require_unmanaged_namespace()
            allowed = self._managed_namespace_access is None
            path = self.get_session_path(key)
            try:
                path.lstat()
            except FileNotFoundError:
                # A sidecar alone is ambiguous evidence, not an ordinary absence.
                checkpoint = self.get_runtime_checkpoint_path(key)
                allowed = allowed and not (checkpoint.exists() or checkpoint.is_symlink())
            else:
                try:
                    header = self._format_header(path)
                    allowed = (allowed and header.get("key") == key
                               and "source_schema_version" not in header.get("metadata", {}))
                except SessionStorageError:
                    allowed = False
            lossy = self.get_legacy_lossy_path(key)
            if lossy != path and (lossy.exists() or lossy.is_symlink()):
                allowed = False
            yield allowed

    @staticmethod
    def settle_source_deletion(
        owner: "ExecutionOwner | None", observe_backend: Callable[[], "JsonlSessionStore"],
    ) -> None:
        """Retire an already-deleted source owner, then settle namespace durability."""
        if owner is not None:
            from nanobot.session.execution_owner import ExecutionOwner

            ExecutionOwner.retire_after_destructive_write(owner)
        observe_backend()._fsync_directory(observe_backend().sessions_dir)

    @staticmethod
    def create_copied_artifact_scratch(
        root: Path,
        *,
        workspace_id: str,
        key: str,
        source_bytes: bytes,
        construct_backend: Callable[[Path, Path], "JsonlSessionStore"],
        mismatch_error: Callable[[], Exception],
    ) -> "JsonlSessionStore":
        """Install an exact copied source only within a newly allocated scratch namespace."""
        workspace = root / "workspace"
        sessions_root = root / "sessions"
        workspace.mkdir(mode=0o700)
        state = workspace / ".nanobot"
        state.mkdir(mode=0o700)
        marker = state / "workspace-id"
        marker.write_text(workspace_id + "\n", encoding="ascii")
        marker.chmod(0o600)
        sessions_root.mkdir(mode=0o700)
        namespace = sessions_root / workspace_id
        namespace.mkdir(mode=0o700)
        claim = namespace / ".workspace"
        claim.write_text(str(workspace) + "\n", encoding="utf-8")
        claim.chmod(0o600)
        backend = construct_backend(workspace, sessions_root)
        backend._write_text_atomic(backend.get_session_path(key), source_bytes.decode("utf-8"))
        if backend.get_session_path(key).read_bytes() != source_bytes:
            raise mismatch_error()
        return backend

    @staticmethod
    def persist_source_records(
        source: "SourceSessionStore",
        header: dict[str, object],
        events: list[dict[str, object]],
        *,
        root_authority: "SourceRootAuthority | None" = None,
        observe_backend: Callable[[], "JsonlSessionStore"],
    ) -> "SourceSnapshot":
        """Prepare actual source records before either existing namespace commit path."""
        from nanobot.session.source_store import SourceSessionStore

        prepared = SourceSessionStore.prepare_source_persistence(source, header, events)
        if root_authority is None:
            observe_backend()._write_text_atomic(
                observe_backend().get_session_path(prepared.read_key()), prepared.text,
            )
        else:
            from nanobot.session.mode_storage import require_namespace_write_access

            require_namespace_write_access(observe_backend())
            root_authority.require_current(observe_backend())
            root_authority.write_text_atomic(
                f"{observe_backend().storage_key(prepared.read_key())}.jsonl",
                prepared.text,
            )
        return prepared.snapshot

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        with suppress(PermissionError, NotImplementedError):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
            finally:
                os.close(fd)

    @classmethod
    def _write_text_atomic(cls, path: Path, content: str, *, mode: int = 0o600) -> None:
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        try:
            with open(tmp, "x", encoding="utf-8") as handle:
                os.chmod(tmp, mode)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            cls._fsync_directory(path.parent)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def _read_workspace_id(cls, marker: Path) -> str:
        if marker.is_symlink():
            raise RuntimeError(f"workspace identity marker must not be a symlink: {marker}")
        value = marker.read_text(encoding="utf-8").strip()
        if not _WORKSPACE_ID_RE.fullmatch(value):
            raise RuntimeError(
                f"workspace identity marker is invalid: {marker}; "
                "restore its original 32-character identifier before starting nanobot"
            )
        return value

    @staticmethod
    def _workspace_id_path(workspace: Path) -> Path:
        state_dir = workspace / _WORKSPACE_STATE_DIR
        if state_dir.is_symlink():
            raise RuntimeError(f"workspace state directory must not be a symlink: {state_dir}")
        ensure_dir(state_dir)
        return state_dir / _WORKSPACE_ID_FILE

    @classmethod
    def _find_workspace_namespace(cls, workspace: Path, root: Path) -> str | None:
        """Recover an identity marker removed by cleanup at the same workspace path."""
        matches: list[str] = []
        for sessions_dir in root.iterdir():
            if (
                not _WORKSPACE_ID_RE.fullmatch(sessions_dir.name)
                or sessions_dir.is_symlink()
                or not sessions_dir.is_dir()
            ):
                continue
            marker = sessions_dir / ".workspace"
            if marker.is_symlink() or not marker.is_file():
                continue
            try:
                recorded = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
                recorded = recorded.resolve(strict=False)
                same_workspace = recorded == workspace or (
                    recorded.exists() and recorded.samefile(workspace)
                )
            except (OSError, UnicodeError, ValueError):
                continue
            if same_workspace:
                matches.append(sessions_dir.name)
        if len(matches) > 1:
            raise RuntimeError(
                f"multiple session namespaces claim workspace {workspace}; "
                "remove the stale namespace marker before starting nanobot"
            )
        return matches[0] if matches else None

    @classmethod
    def _load_or_create_workspace_id(cls, workspace: Path, root: Path) -> str:
        marker = cls._workspace_id_path(workspace)
        if marker.exists() or marker.is_symlink():
            return cls._read_workspace_id(marker)

        recovered = cls._find_workspace_namespace(workspace, root)
        if recovered is not None:
            cls._write_text_atomic(marker, f"{recovered}\n")
            return recovered

        workspace_id = secrets.token_hex(16)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(marker, flags, 0o600)
        except FileExistsError:
            return cls._read_workspace_id(marker)
        try:
            payload = f"{workspace_id}\n".encode("ascii")
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        except BaseException:
            with suppress(OSError):
                marker.unlink()
            raise
        finally:
            os.close(fd)
        cls._fsync_directory(marker.parent)
        return workspace_id

    @classmethod
    def _replace_workspace_id(cls, workspace: Path, workspace_id: str) -> None:
        cls._write_text_atomic(cls._workspace_id_path(workspace), f"{workspace_id}\n")

    @classmethod
    def _write_workspace_marker(cls, sessions_dir: Path, workspace: Path) -> None:
        cls._write_text_atomic(sessions_dir / ".workspace", f"{workspace}\n")

    @classmethod
    def _claim_workspace_namespace(
        cls,
        root: Path,
        workspace: Path,
        workspace_id: str,
    ) -> str:
        """Bind a stable workspace ID, rotating copied live workspaces apart."""
        for _attempt in range(3):
            sessions_dir = root / workspace_id
            marker = sessions_dir / ".workspace"
            if sessions_dir.is_symlink():
                raise RuntimeError(f"session namespace must not be a symlink: {sessions_dir}")
            if not sessions_dir.exists():
                ensure_dir(sessions_dir)
                cls._write_workspace_marker(sessions_dir, workspace)
                return workspace_id
            if marker.is_symlink():
                raise RuntimeError(f"session workspace marker must not be a symlink: {marker}")
            if not marker.exists():
                if any(sessions_dir.iterdir()):
                    raise RuntimeError(
                        f"session namespace has data but no workspace marker: {sessions_dir}"
                    )
                cls._write_workspace_marker(sessions_dir, workspace)
                return workspace_id

            recorded_text = marker.read_text(encoding="utf-8").strip()
            if not recorded_text:
                raise RuntimeError(f"session workspace marker is empty: {marker}")
            recorded = Path(recorded_text).expanduser().resolve(strict=False)
            if recorded == workspace:
                return workspace_id
            try:
                same_workspace = recorded.exists() and recorded.samefile(workspace)
            except OSError:
                same_workspace = False
            if same_workspace:
                cls._write_workspace_marker(sessions_dir, workspace)
                return workspace_id
            if not recorded.exists():
                # The identity marker travelled with a renamed or moved workspace.
                cls._write_workspace_marker(sessions_dir, workspace)
                return workspace_id

            # Both paths exist and are different: this is a copy, not a move.
            workspace_id = secrets.token_hex(16)
            cls._replace_workspace_id(workspace, workspace_id)

        raise RuntimeError(f"could not allocate an isolated session namespace for {workspace}")

    @staticmethod
    def _session_file_snapshot(path: Path) -> _SessionFileSnapshot | None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError:
            return None
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                return None
            digest = hashlib.sha256()
            saw_record = False
            updated_at: float | None = None
            with os.fdopen(fd, "rb", closefd=False) as handle:
                for raw_line in handle:
                    digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    value: object = json.loads(raw_line.decode("utf-8"))
                    data = _json_object(value)
                    saw_record = True
                    if data.get("_type") == "metadata":
                        raw_updated_at = cast(object, data.get("updated_at"))
                        if isinstance(raw_updated_at, str) and raw_updated_at:
                            updated_at = datetime.fromisoformat(raw_updated_at).timestamp()
            after = os.fstat(fd)
            if (
                not saw_record
                or before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                return None
            return _SessionFileSnapshot(
                digest=digest.hexdigest(),
                size=after.st_size,
                mtime_ns=after.st_mtime_ns,
                updated_at=(updated_at if updated_at is not None else after.st_mtime_ns / 1e9),
                device=after.st_dev,
                inode=after.st_ino,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            return None
        finally:
            os.close(fd)

    @classmethod
    def _prepare_copy(
        cls,
        src: Path,
        dst_dir: Path,
        snapshot: _SessionFileSnapshot,
    ) -> Path:
        tmp = dst_dir / f".{src.name}.{secrets.token_hex(8)}.tmp"
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        src_fd = os.open(src, flags)
        try:
            before = os.fstat(src_fd)
            if (
                before.st_dev != snapshot.device
                or before.st_ino != snapshot.inode
                or before.st_size != snapshot.size
                or before.st_mtime_ns != snapshot.mtime_ns
            ):
                raise OSError("session source changed before migration")
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(src_fd, "rb", closefd=False) as source, open(tmp, "xb") as target:
                os.chmod(tmp, 0o600)
                while chunk := source.read(_COPY_CHUNK_SIZE):
                    digest.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            after = os.fstat(src_fd)
            if (
                digest.hexdigest() != snapshot.digest
                or size != snapshot.size
                or after.st_dev != snapshot.device
                or after.st_ino != snapshot.inode
                or after.st_size != snapshot.size
                or after.st_mtime_ns != snapshot.mtime_ns
            ):
                raise OSError("session source changed during migration")
            return tmp
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            os.close(src_fd)

    @classmethod
    def _install_snapshot(
        cls,
        src: Path,
        dst: Path,
        snapshot: _SessionFileSnapshot,
    ) -> None:
        tmp = cls._prepare_copy(src, dst.parent, snapshot)
        try:
            os.replace(tmp, dst)
            cls._fsync_directory(dst.parent)
            installed = cls._session_file_snapshot(dst)
            if installed is None or installed.digest != snapshot.digest:
                raise OSError(f"session migration verification failed: {dst}")
        finally:
            tmp.unlink(missing_ok=True)

    def _archive_conflict(
        self,
        src: Path,
        snapshot: _SessionFileSnapshot,
        label: str,
    ) -> Path:
        conflict_dir = ensure_dir(self.sessions_dir / ".migration-conflicts")
        conflict = conflict_dir / (
            f"{src.stem}.{label}.{snapshot.digest[:12]}.{secrets.token_hex(4)}.jsonl"
        )
        self._install_snapshot(src, conflict, snapshot)
        return conflict

    @classmethod
    def _remove_migrated_source(
        cls,
        src: Path,
        snapshot: _SessionFileSnapshot,
    ) -> bool:
        try:
            current = src.stat(follow_symlinks=False)
            if (
                current.st_dev != snapshot.device
                or current.st_ino != snapshot.inode
                or current.st_size != snapshot.size
                or current.st_mtime_ns != snapshot.mtime_ns
            ):
                return False
            src.unlink()
            cls._fsync_directory(src.parent)
            return True
        except OSError:
            return False

    def _migrate_from_workspace(self, workspace: Path) -> None:
        """Durably copy legacy sessions out of the workspace, then remove the source."""
        old_dir = workspace / "sessions"
        if old_dir.is_symlink() or not old_dir.is_dir():
            if old_dir.is_symlink():
                logger.warning("Skipping symlinked legacy sessions directory: {}", old_dir)
            return
        for src in old_dir.glob("*.jsonl"):
            if src.is_symlink() or not src.is_file():
                logger.warning("Skipping unsafe legacy session file: {}", src)
                continue
            dst = self.sessions_dir / src.name
            source_snapshot = self._session_file_snapshot(src)
            if source_snapshot is None:
                logger.warning("Skipping invalid or changing legacy session file: {}", src)
                continue
            try:
                destination_snapshot = self._session_file_snapshot(dst) if dst.exists() else None
                if dst.exists() and destination_snapshot is None:
                    logger.warning(
                        "Keeping legacy session because destination is invalid: {}",
                        dst,
                    )
                    continue

                if destination_snapshot is None:
                    self._install_snapshot(src, dst, source_snapshot)
                elif destination_snapshot.digest == source_snapshot.digest:
                    pass
                elif source_snapshot.updated_at > destination_snapshot.updated_at:
                    archived = self._archive_conflict(dst, destination_snapshot, "destination")
                    self._install_snapshot(src, dst, source_snapshot)
                    logger.warning("Archived older session migration conflict at {}", archived)
                else:
                    archived = self._archive_conflict(src, source_snapshot, "workspace")
                    logger.warning("Archived older session migration conflict at {}", archived)

                installed = self._session_file_snapshot(dst)
                if installed is None:
                    raise OSError(f"session migration destination is unreadable: {dst}")
                selected_digest = (
                    source_snapshot.digest
                    if destination_snapshot is None
                    or source_snapshot.updated_at > destination_snapshot.updated_at
                    else destination_snapshot.digest
                )
                if installed.digest != selected_digest:
                    raise OSError(f"session migration selected unexpected data: {dst}")
                if not self._remove_migrated_source(src, source_snapshot):
                    logger.warning(
                        "Session migrated but legacy source changed or could not be removed: {}",
                        src,
                    )
            except OSError as exc:
                logger.warning("Failed to migrate session {}: {}", src, exc)

    def restore_to_workspace(self) -> SessionRestoreResult:
        """Copy canonical sessions back for an explicit downgrade or rollback."""
        self.require_unmanaged_namespace()
        restored = 0
        unchanged = 0
        conflicts: list[Path] = []
        old_dir = self.workspace / "sessions"
        if old_dir.is_symlink():
            raise RuntimeError(f"refusing to restore into symlinked sessions directory: {old_dir}")
        ensure_dir(old_dir)

        with self._migration_lock, self._session_files_lock:
            for src in self.sessions_dir.glob("*.jsonl"):
                if self.session_key_from_path(src) is None:
                    continue
                source_snapshot = self._session_file_snapshot(src)
                if source_snapshot is None:
                    conflicts.append(src)
                    continue
                dst = old_dir / src.name
                if dst.exists():
                    destination_snapshot = self._session_file_snapshot(dst)
                    if (
                        destination_snapshot is not None
                        and destination_snapshot.digest == source_snapshot.digest
                    ):
                        unchanged += 1
                    else:
                        conflicts.append(dst)
                    continue
                self._install_snapshot(src, dst, source_snapshot)
                restored += 1
        return SessionRestoreResult(
            restored=restored,
            unchanged=unchanged,
            conflicts=tuple(conflicts),
        )

    @staticmethod
    def safe_key(key: str) -> str:
        return safe_filename(key.replace(":", "_"))

    @staticmethod
    def storage_key(key: str) -> str:
        return base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")

    @staticmethod
    def decode_storage_key(stem: str) -> str | None:
        try:
            padding = 4 - len(stem) % 4
            if padding != 4:
                stem += "=" * padding
            return base64.urlsafe_b64decode(stem).decode("utf-8")
        except _SESSION_DATA_ERRORS:
            return None

    @classmethod
    def session_key_from_path(cls, path: Path) -> str | None:
        key = cls.decode_storage_key(path.stem)
        if key is None or cls.storage_key(key) != path.stem:
            return None
        return key

    def get_session_path(self, key: str) -> Path:
        return self.sessions_dir / f"{self.storage_key(key)}.jsonl"

    def get_runtime_checkpoint_path(self, key: str) -> Path:
        return self.sessions_dir / f"{self.storage_key(key)}{_RUNTIME_CHECKPOINT_SUFFIX}"

    def get_legacy_lossy_path(self, key: str) -> Path:
        return self.sessions_dir / f"{safe_filename(key.replace(':', '_'))}.jsonl"

    def get_legacy_session_path(self, key: str) -> Path:
        return self.legacy_sessions_dir / f"{self.safe_key(key)}.jsonl"

    def load(self, key: str) -> Session | None:
        with self.locked_session_files():
            return self._load_unlocked(key)

    @staticmethod
    def parse_legacy_bytes(data: bytes, *, expected_key: str | None = None) -> Session:
        """Parse the actual legacy JSONL grammar without filesystem side effects."""

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for name, value in pairs:
                if name in result:
                    raise SessionStorageError()
                result[name] = value
            return result

        try:
            text = data.decode("utf-8")
            if text.encode("utf-8") != data:
                raise SessionStorageError()
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            provider_state: ProviderConversationState | None = None
            created_at: datetime | None = None
            updated_at: datetime | None = None
            last_consolidated = 0
            revision = generation = 0
            incarnation = "legacy"
            session_key: str | None = None
            metadata_seen = False
            for raw_line in text.splitlines():
                if not raw_line.strip():
                    continue
                value: object = json.loads(
                    raw_line,
                    object_pairs_hook=unique,
                    parse_constant=lambda _value: (_ for _ in ()).throw(SessionStorageError()),
                )
                row = _json_object(value)
                record_type = row.get("_type")
                if record_type == "metadata":
                    if metadata_seen or messages or provider_state is not None:
                        raise SessionStorageError()
                    metadata_seen = True
                    key_value = row.get("key")
                    if not isinstance(key_value, str) or not key_value:
                        raise SessionStorageError()
                    session_key = key_value
                    if expected_key is not None and session_key != expected_key:
                        raise SessionStorageError()
                    revision = _storage_counter(row.get("revision", 0))
                    generation = _storage_counter(row.get("generation", 0))
                    incarnation_value = row.get("incarnation", "legacy")
                    if not isinstance(incarnation_value, str) or not incarnation_value:
                        raise SessionStorageError()
                    incarnation = incarnation_value
                    metadata_value = row.get("metadata", {})
                    if not isinstance(metadata_value, dict) or "source_schema_version" in metadata_value:
                        raise SessionStorageError()
                    metadata = deepcopy(cast(dict[str, Any], metadata_value))
                    created_value = row.get("created_at")
                    updated_value = row.get("updated_at")
                    created_at = (
                        datetime.fromisoformat(created_value)
                        if isinstance(created_value, str) and created_value
                        else None
                    )
                    updated_at = (
                        datetime.fromisoformat(updated_value)
                        if isinstance(updated_value, str) and updated_value
                        else None
                    )
                    last_consolidated = _archive_offset(row)
                elif record_type == _PROVIDER_STATE_RECORD_TYPE:
                    if not metadata_seen or provider_state is not None:
                        raise SessionStorageError()
                    provider_state = ProviderConversationState.from_private_record(row.get("state"))
                    if provider_state is None:
                        raise SessionStorageError()
                else:
                    if not metadata_seen:
                        raise SessionStorageError()
                    messages.append(deepcopy(row))
            if not metadata_seen or session_key is None:
                raise SessionStorageError()
            session = Session(
                key=session_key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
                provider_state=provider_state,
                revision=revision,
                generation=generation,
                incarnation=incarnation,
            )
            if _migrate_legacy_exec_session_records(session.messages, session.metadata):
                session.provider_state = None
            Session.record_committed_base(session)
            return session
        except (SessionStorageError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
            raise SessionStorageError() from None

    @staticmethod
    def overlay_managed_checkpoint_bytes(session: Session, data: bytes) -> Session:
        """Apply one exact managed legacy checkpoint without filesystem side effects."""

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for name, value in pairs:
                if name in result:
                    raise SessionStorageError()
                result[name] = value
            return result

        expected = {
            "version",
            "session_key",
            "base_updated_at",
            "base_message_count",
            "base_revision",
            "base_generation",
            "base_incarnation",
            "checkpoint",
            "provider_state",
        }
        try:
            value: object = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=unique,
                parse_constant=lambda _value: (_ for _ in ()).throw(SessionStorageError()),
            )
            raw = _json_object(value)
            if set(raw) != expected:
                raise SessionStorageError()
            if (
                type(raw.get("version")) is not int
                or raw["version"] != _RUNTIME_CHECKPOINT_VERSION
                or raw.get("session_key") != session.key
                or raw.get("base_updated_at") != session.updated_at.isoformat()
                or type(raw.get("base_message_count")) is not int
                or raw["base_message_count"] != len(session.messages)
                or type(raw.get("base_revision")) is not int
                or raw["base_revision"] != session.revision
                or type(raw.get("base_generation")) is not int
                or raw["base_generation"] != session.generation
                or raw.get("base_incarnation") != session.incarnation
                or not isinstance(raw.get("checkpoint"), dict)
            ):
                raise SessionStorageError()
            provider_record = raw.get("provider_state")
            if provider_record is not None and (
                not isinstance(provider_record, dict)
                or set(cast(dict[object, object], provider_record)) not in (
                    {"kind", "provider", "model", "version", "payload", "pending_messages"},
                    {"kind", "provider", "model", "version", "payload", "pending_messages",
                     "conversation_binding", "transport_receipt"},
                )
            ):
                raise SessionStorageError()
            provider_state = (
                None
                if provider_record is None
                else ProviderConversationState.from_private_record(cast(object, provider_record))
            )
            if provider_record is not None and provider_state is None:
                raise SessionStorageError()
            detached = deepcopy(session)
            detached.metadata[_RUNTIME_CHECKPOINT_KEY] = deepcopy(raw["checkpoint"])
            detached.provider_state = provider_state
            Session.record_committed_base(detached)
            return detached
        except (SessionStorageError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
            raise SessionStorageError() from None

    @staticmethod
    def _format_header(path: Path) -> dict[str, Any]:
        """Identify format only from an intact, unambiguous first metadata row."""
        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError("Duplicate session header key")
            return result

        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                raise SessionStorageError()
            with path.open(encoding="utf-8") as handle:
                first = next((line for line in handle if line.strip()), "")
            header = _json_object(json.loads(first, object_pairs_hook=unique))
            if header.get("_type") != "metadata" or not isinstance(header.get("metadata", {}), dict):
                raise SessionStorageError()
            return header
        except (*_SESSION_DATA_ERRORS, UnicodeError, RecursionError):
            raise SessionStorageError() from None

    @classmethod
    def _has_legacy_header(cls, path: Path) -> bool:
        try:
            return "source_schema_version" not in cls._format_header(path).get("metadata", {})
        except SessionStorageError:
            return False

    def _load_unlocked(self, key: str, *, strict: bool = False) -> Session | None:
        path = self.get_session_path(key)
        if not path.exists():
            return None
        self._format_header(path)

        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: datetime | None = None
            updated_at: datetime | None = None
            last_consolidated = 0
            provider_state: ProviderConversationState | None = None
            revision = generation = 0
            incarnation = "legacy"
            metadata_seen = False

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    raw_data: object = json.loads(line)
                    data = _json_object(raw_data)

                    record_type = data.get("_type")
                    if record_type == "metadata":
                        if strict and (metadata_seen or messages or provider_state is not None):
                            raise SessionStorageError()
                        metadata_seen = True
                        revision = _storage_counter(data.get("revision", 0))
                        generation = _storage_counter(data.get("generation", 0))
                        incarnation = data.get("incarnation", "legacy")
                        if not isinstance(incarnation, str) or not incarnation:
                            raise SessionStorageError()

                        metadata_value = cast(object, data.get("metadata", {}))
                        metadata = (
                            cast(dict[str, Any], metadata_value)
                            if isinstance(metadata_value, dict)
                            else {}
                        )
                        if "source_schema_version" in metadata:
                            return self.source.load_source_snapshot(key).to_session()
                        created_at_value = cast(object, data.get("created_at"))
                        updated_at_value = cast(object, data.get("updated_at"))
                        created_at = (
                            datetime.fromisoformat(created_at_value)
                            if isinstance(created_at_value, str) and created_at_value
                            else None
                        )
                        updated_at = (
                            datetime.fromisoformat(updated_at_value)
                            if isinstance(updated_at_value, str) and updated_at_value
                            else None
                        )
                        last_consolidated = _archive_offset(data)
                    elif record_type == _PROVIDER_STATE_RECORD_TYPE:
                        provider_state = ProviderConversationState.from_private_record(
                            data.get("state")
                        )
                    else:
                        messages.append(data)

            if not metadata_seen:
                raise SessionStorageError()
            session = Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
                provider_state=provider_state,
                revision=revision, generation=generation, incarnation=incarnation,
            )
            if not strict:
                self._overlay_checkpoint_unlocked(session, path)
            if _migrate_legacy_exec_session_records(session.messages, session.metadata):
                session.provider_state = None
            Session.record_committed_base(session)
            return session
        except _SESSION_DATA_ERRORS as e:
            if strict or not self._has_legacy_header(path):
                raise SessionStorageError() from None
            logger.warning("Failed to load session {}: {}", key, e)
            repaired = self._repair_unlocked(key)
            if repaired is not None:
                logger.info(
                    "Recovered session {} from corrupt file ({} messages)",
                    key,
                    len(repaired.messages),
                )
            return repaired

    def repair(self, key: str, *, path: Path | None = None) -> Session | None:
        with self.locked_session_files():
            return self._repair_unlocked(key, path=path)

    def _repair_unlocked(self, key: str, *, path: Path | None = None) -> Session | None:
        if path is None:
            path = self.get_session_path(key)
        if not path.exists():
            return None

        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: datetime | None = None
            updated_at: datetime | None = None
            last_consolidated = 0
            provider_state: ProviderConversationState | None = None
            skipped = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_data: object = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if not isinstance(raw_data, dict):
                        skipped += 1
                        continue
                    data = cast(dict[str, Any], raw_data)

                    record_type = data.get("_type")
                    if record_type == "metadata":
                        metadata_value = cast(object, data.get("metadata", {}))
                        metadata = (
                            cast(dict[str, Any], metadata_value)
                            if isinstance(metadata_value, dict)
                            else {}
                        )
                        if "source_schema_version" in metadata:
                            return self.source.load_source_snapshot(key).to_session()
                        created_at_value = cast(object, data.get("created_at"))
                        if isinstance(created_at_value, str) and created_at_value:
                            with suppress(ValueError):
                                created_at = datetime.fromisoformat(created_at_value)
                        updated_at_value = cast(object, data.get("updated_at"))
                        if isinstance(updated_at_value, str) and updated_at_value:
                            with suppress(ValueError):
                                updated_at = datetime.fromisoformat(updated_at_value)
                        last_consolidated = _archive_offset(data)
                    elif record_type == _PROVIDER_STATE_RECORD_TYPE:
                        candidate = ProviderConversationState.from_private_record(
                            data.get("state")
                        )
                        if candidate is None:
                            skipped += 1
                        else:
                            provider_state = candidate
                    else:
                        messages.append(data)

            if skipped:
                logger.warning("Skipped {} corrupt lines in session {}", skipped, key)

            if not messages and not metadata and provider_state is None:
                return None

            session = Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
                provider_state=provider_state,
            )
            self._overlay_checkpoint_unlocked(session, path)
            if _migrate_legacy_exec_session_records(session.messages, session.metadata):
                session.provider_state = None
            return session
        except _SESSION_DATA_ERRORS as e:
            logger.warning("Repair failed for session {}: {}", key, e)
            return None

    @staticmethod
    def session_payload(session: Session) -> SessionPayload:
        return {
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "messages": session.messages,
        }

    def save(self, session: Session, *, fsync: bool = False) -> None:
        with self.locked_session_files():
            self._save_checked_unlocked(session, fsync=fsync)

    def _save_checked_unlocked(self, session: Session, *, fsync: bool = False) -> None:
        disk = self._load_unlocked(session.key, strict=True)
        if disk is not None:
            self._overlay_checkpoint_unlocked(disk, self.get_session_path(session.key))
        candidate = Session.prepare_legacy_commit(session, disk)
        self._save_unlocked(candidate, fsync=fsync)
        # Publish only after the complete write (including requested fsync) succeeds.
        Session.publish_legacy_commit(session, candidate)

    def save_runtime_checkpoint(self, session: Session) -> None:
        """Atomically persist only the volatile in-flight turn state.

        A checkpoint is written several times during a tool-heavy turn. Keeping it
        beside the append history avoids copying the full transcript at each safe
        recovery boundary.
        """
        with self.locked_session_files():
            path = self.get_session_path(session.key)
            if not path.exists():
                # A user turn normally creates the session first. Internal callers
                # may checkpoint a fresh session, so establish the durable base once.
                self._save_checked_unlocked(session)
                return

            disk = self._load_unlocked(session.key, strict=True)
            disk = Session.require_checkpoint_base(session, disk)
            self._overlay_checkpoint_unlocked(disk, path)
            checkpoint = Session.prepare_checkpoint_write(session, disk)
            if not isinstance(checkpoint, dict):
                # Removal must also clear a checkpoint materialized in the main
                # file by an unrelated metadata write, retaining current handles.
                self._save_checked_unlocked(session)
                return

            payload: dict[str, Any] = {
                "version": _RUNTIME_CHECKPOINT_VERSION,
                "session_key": session.key,
                "base_updated_at": session.updated_at.isoformat(),
                "base_message_count": len(session.messages),
                "base_revision": disk.revision,
                "base_generation": disk.generation,
                "base_incarnation": disk.incarnation,
                "checkpoint": checkpoint,
                "provider_state": (
                    session.provider_state.to_private_record()
                    if session.provider_state is not None
                    else None
                ),
            }
            target = self.get_runtime_checkpoint_path(session.key)
            tmp = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
            try:
                with open(tmp, "x", encoding="utf-8") as handle:
                    os.chmod(tmp, 0o600)
                    json.dump(
                        payload,
                        handle,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                os.replace(tmp, target)
                Session.publish_checkpoint_write(session, cast(object, checkpoint))
            finally:
                tmp.unlink(missing_ok=True)

    def _overlay_checkpoint_unlocked(self, session: Session, main_path: Path) -> None:
        if self._managed_namespace_access is None:
            self._overlay_runtime_checkpoint_unlocked(session, main_path)
            return
        checkpoint_path = self.get_runtime_checkpoint_path(session.key)
        if checkpoint_path.exists():
            if "source_schema_version" in session.metadata:
                raise SessionStorageError()
            overlaid = self.overlay_managed_checkpoint_bytes(session, checkpoint_path.read_bytes())
            session.metadata = overlaid.metadata
            session.provider_state = overlaid.provider_state
            Session.record_committed_base(session)

    def _overlay_runtime_checkpoint_unlocked(self, session: Session, main_path: Path) -> None:
        if "source_schema_version" in session.metadata:
            return
        checkpoint_path = self.get_runtime_checkpoint_path(session.key)
        try:
            checkpoint_stat = checkpoint_path.lstat()
            if not stat.S_ISREG(checkpoint_stat.st_mode):
                logger.warning(
                    "Ignoring non-regular runtime checkpoint for session {}",
                    session.key,
                )
                return
            # A complete session save supersedes an older sidecar. This comparison
            # closes the small crash window between replacing the JSONL and unlinking
            # its previous checkpoint.
            if main_path.stat().st_mtime_ns > checkpoint_stat.st_mtime_ns:
                checkpoint_path.unlink(missing_ok=True)
                return
            raw = _json_object(json.loads(checkpoint_path.read_text(encoding="utf-8")))
            if (
                raw.get("version") != _RUNTIME_CHECKPOINT_VERSION
                or raw.get("session_key") != session.key
                or raw.get("base_updated_at") != session.updated_at.isoformat()
                or raw.get("base_message_count") != len(session.messages)
                or raw.get("base_revision") != session.revision
                or raw.get("base_generation") != session.generation
                or raw.get("base_incarnation") != session.incarnation
                or not isinstance(raw.get("checkpoint"), dict)
            ):
                checkpoint_path.unlink(missing_ok=True)
                return
            provider_record = raw.get("provider_state")
            provider_state = (
                None
                if provider_record is None
                else ProviderConversationState.from_private_record(provider_record)
            )
            if provider_record is not None and provider_state is None:
                raise ValueError("invalid checkpoint provider state")
            session.metadata[_RUNTIME_CHECKPOINT_KEY] = cast(
                dict[str, Any], raw["checkpoint"]
            )
            session.provider_state = provider_state
        except FileNotFoundError:
            return
        except _RUNTIME_CHECKPOINT_DATA_ERRORS as exc:
            logger.warning(
                "Ignoring invalid runtime checkpoint for session {}: {}",
                session.key,
                exc,
            )
            # Atomic writes mean a malformed target cannot become valid later.
            # Remove it once so future loads do not repeatedly parse and log it.
            with suppress(OSError):
                if checkpoint_path.is_file() and not checkpoint_path.is_symlink():
                    checkpoint_path.unlink()

    def _save_unlocked(self, session: Session, *, fsync: bool = False) -> None:
        path = self.get_session_path(session.key)
        tmp_path = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")

        try:
            with open(tmp_path, "xb") as f:
                os.chmod(tmp_path, 0o600)
                f.write(self.render_legacy_bytes(session))
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())

            os.replace(tmp_path, path)

            # The full record now contains the authoritative checkpoint state (or
            # its removal), so an older volatile overlay is no longer needed.
            self.get_runtime_checkpoint_path(session.key).unlink(missing_ok=True)

            if fsync:
                with suppress(PermissionError):
                    fd = os.open(str(path.parent), os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    except OSError as exc:
                        if exc.errno != errno.EINVAL:
                            raise
                    finally:
                        os.close(fd)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def render_legacy_bytes(session: Session) -> bytes:
        """Render the exact legacy grammar without touching storage."""

        metadata_line = {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "revision": session.revision,
            "generation": session.generation,
            "incarnation": session.incarnation,
            "last_archived": session.last_archived,
            "last_consolidated": session.last_consolidated,
        }
        rows: list[dict[str, Any]] = [metadata_line]
        if session.provider_state is not None:
            rows.append(
                {
                    "_type": _PROVIDER_STATE_RECORD_TYPE,
                    "state": session.provider_state.to_private_record(),
                }
            )
        rows.extend(session.messages)
        try:
            return b"".join(
                (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                for row in rows
            )
        except (TypeError, ValueError, UnicodeError):
            raise SessionStorageError() from None

    def commit_busy_inbox_metadata(
        self, key: str, expected: object, replacement: dict[str, object],
    ) -> None:
        """CAS only the reserved inbox record, retaining history and recovery."""
        with self.locked_session_files():
            session = self._load_unlocked(key, strict=True)
            if session is None:
                raise SessionWriteConflict()
            if "source_schema_version" in session.metadata:
                self.source.commit_busy_inbox_metadata(key, expected, replacement)
                return
            self._overlay_checkpoint_unlocked(session, self.get_session_path(key))
            if session.metadata.get(BUSY_INBOX_METADATA_KEY) != expected:
                raise SessionWriteConflict()
            session.metadata[BUSY_INBOX_METADATA_KEY] = deepcopy(replacement)
            session.revision += 1
            self._save_unlocked(session, fsync=True)

    def update_metadata(
        self,
        key: str,
        updates: dict[str, Any],
        *,
        removals: Collection[str] = (),
        fsync: bool = False,
    ) -> bool:
        """Advance metadata revision, retaining the authoritative recovery overlay."""
        with self.locked_session_files():
            session = self._load_unlocked(key, strict=True)
            if session is None:
                return False
            if "source_schema_version" in session.metadata:
                self.source.update_metadata(key, updates, removals=tuple(removals))
                return True
            self._overlay_checkpoint_unlocked(session, self.get_session_path(key))
            accepted = {
                name: deepcopy(value) for name, value in updates.items()
                if name not in _RESERVED_DURABLE_METADATA
            }
            accepted_removals = {
                name for name in removals
                if isinstance(cast(object, name), str) and name not in _RESERVED_DURABLE_METADATA
            }
            session.metadata.update(accepted)
            for name in accepted_removals:
                session.metadata.pop(name, None)
            session.revision += 1
            self._save_unlocked(session, fsync=fsync)
            return True

    def delete(self, key: str) -> bool:
        from nanobot.session.execution_owner import execution_lock_guard

        # Recheck format after taking execution ownership: upgrade can race a delete.
        with execution_lock_guard(self, key), self.locked_session_files():
            path = self.get_session_path(key)
            if path.exists() and "source_schema_version" in self._format_header(path).get("metadata", {}):
                from nanobot.session.source_store import SourceSessionStore

                SourceSessionStore.delete_unowned_source(self, key)
                return True
            return self._delete_unlocked(key)

    def _delete_unlocked(self, key: str) -> bool:
        from nanobot.session.busy_inbox import require_busy_inbox_disposition

        session = self._load_unlocked(key, strict=True)
        if session is not None:
            require_busy_inbox_disposition(session.metadata)
        paths = [
            self.get_session_path(key),
            self.get_runtime_checkpoint_path(key),
            self.get_legacy_lossy_path(key),
            self.get_legacy_session_path(key),
        ]
        deleted = False
        for path in paths:
            if not path.exists():
                continue
            try:
                path.unlink()
                deleted = True
            except OSError as e:
                logger.warning("Failed to delete session file {}: {}", path, e)
        return deleted

    def read(self, key: str) -> SessionPayload | None:
        with self.locked_session_files():
            return self._read_unlocked(key)

    def _read_unlocked(self, key: str) -> SessionPayload | None:
        path = self.get_session_path(key)
        if not path.exists():
            return None
        self._format_header(path)
        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: str | None = None
            updated_at: str | None = None
            stored_key: str | None = None
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw_data: object = json.loads(line)
                    data = _json_object(raw_data)
                    record_type = data.get("_type")
                    if record_type == "metadata":
                        metadata_value = cast(object, data.get("metadata", {}))
                        metadata = (
                            cast(dict[str, Any], metadata_value)
                            if isinstance(metadata_value, dict)
                            else {}
                        )
                        if "source_schema_version" in metadata:
                            return self.session_payload(self.source.load_source_snapshot(key).to_session())
                        created_at_value = cast(object, data.get("created_at"))
                        updated_at_value = cast(object, data.get("updated_at"))
                        stored_key_value = cast(object, data.get("key"))
                        created_at = (
                            created_at_value if isinstance(created_at_value, str) else None
                        )
                        updated_at = (
                            updated_at_value if isinstance(updated_at_value, str) else None
                        )
                        stored_key = (
                            stored_key_value if isinstance(stored_key_value, str) else None
                        )
                    elif record_type == _PROVIDER_STATE_RECORD_TYPE:
                        continue
                    else:
                        messages.append(data)
            _migrate_legacy_exec_session_records(messages, metadata)
            return {
                "key": stored_key or key,
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": metadata,
                "messages": messages,
            }
        except _SESSION_DATA_ERRORS as e:
            if not self._has_legacy_header(path):
                raise SessionStorageError() from None
            logger.warning("Failed to read session {}: {}", key, e)
            repaired = self._repair_unlocked(key, path=path)
            if repaired is not None:
                logger.info("Recovered read-only session view {} from corrupt file", key)
                return self.session_payload(repaired)
            return None

    def read_metadata(self, key: str) -> SessionMetadataPayload | None:
        with self.locked_session_files():
            return self._read_metadata_unlocked(key)

    def _read_metadata_unlocked(self, key: str) -> SessionMetadataPayload | None:
        path = self.get_session_path(key)
        if not path.exists():
            return None
        self._format_header(path)
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw_data: object = json.loads(line)
                    data = _json_object(raw_data)
                    if data.get("_type") != "metadata":
                        return None
                    metadata_value = cast(object, data.get("metadata", {}))
                    if isinstance(metadata_value, dict) and "source_schema_version" in metadata_value:
                        source = self.source.load_source_snapshot(key).to_session()
                        return {
                            "key": source.key, "created_at": source.created_at.isoformat(),
                            "updated_at": source.updated_at.isoformat(), "metadata": source.metadata,
                        }
                    key_value = cast(object, data.get("key"))
                    created_at_value = cast(object, data.get("created_at"))
                    updated_at_value = cast(object, data.get("updated_at"))
                    return {
                        "key": key_value if isinstance(key_value, str) and key_value else key,
                        "created_at": (
                            created_at_value if isinstance(created_at_value, str) else None
                        ),
                        "updated_at": (
                            updated_at_value if isinstance(updated_at_value, str) else None
                        ),
                        "metadata": (
                            cast(dict[str, Any], metadata_value)
                            if isinstance(metadata_value, dict)
                            else {}
                        ),
                    }
            return None
        except _SESSION_DATA_ERRORS as e:
            if not self._has_legacy_header(path):
                raise SessionStorageError() from None
            logger.warning("Failed to read session metadata {}: {}", key, e)
            repaired = self._repair_unlocked(key, path=path)
            if repaired is not None:
                logger.info("Recovered read-only session metadata {} from corrupt file", key)
                return {
                    "key": repaired.key,
                    "created_at": repaired.created_at.isoformat(),
                    "updated_at": repaired.updated_at.isoformat(),
                    "metadata": repaired.metadata,
                }
            return None

    def list_sessions(self) -> list[SessionInfo]:
        with self.locked_session_files():
            return self._list_sessions_unlocked()

    def _list_sessions_unlocked(self) -> list[SessionInfo]:
        sessions: list[SessionInfo] = []

        for path in self.sessions_dir.glob("*.jsonl"):
            storage_key = self.session_key_from_path(path)
            if storage_key is None:
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        raw_data: object = json.loads(first_line)
                        data = _json_object(raw_data)
                        if data.get("_type") == "metadata":
                            key_value = cast(object, data.get("key"))
                            key = (
                                key_value
                                if isinstance(key_value, str) and key_value
                                else storage_key
                            )
                            metadata = cast(object, data.get("metadata", {}))
                            title = _metadata_title(metadata)
                            preview = ""
                            fallback_preview = ""
                            scanned_records = 0
                            scanned_chars = 0
                            for line in f:
                                if not line.strip():
                                    continue
                                if _is_provider_state_record_line(line):
                                    continue
                                scanned_records += 1
                                scanned_chars += len(line)
                                if (
                                    scanned_records > _SESSION_LIST_PREVIEW_MAX_RECORDS
                                    or scanned_chars > _SESSION_LIST_PREVIEW_MAX_CHARS
                                ):
                                    break
                                raw_item: object = json.loads(line)
                                item = _json_object(raw_item)
                                if item.get("_type") in {
                                    "metadata",
                                    _PROVIDER_STATE_RECORD_TYPE,
                                }:
                                    continue
                                text = _message_preview_text(item)
                                if not text:
                                    continue
                                if item.get("role") == "user":
                                    preview = text
                                    break
                                if not fallback_preview and item.get("role") == "assistant":
                                    fallback_preview = text
                            preview = preview or fallback_preview
                            fallback_time = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                            created_at = cast(object, data.get("created_at"))
                            updated_at = cast(object, data.get("updated_at"))
                            sessions.append(
                                {
                                    "key": key,
                                    "created_at": (
                                        created_at
                                        if isinstance(created_at, str) and created_at
                                        else fallback_time
                                    ),
                                    "updated_at": (
                                        updated_at
                                        if isinstance(updated_at, str) and updated_at
                                        else fallback_time
                                    ),
                                    "title": title,
                                    "preview": preview,
                                    "path": str(path),
                                }
                            )
            except FileNotFoundError:
                continue
            except _SESSION_DATA_ERRORS:
                if not self._has_legacy_header(path):
                    logger.warning("Ignoring session listing with unidentifiable storage header")
                    continue
                repaired = self._repair_unlocked(storage_key, path=path)
                if repaired is not None:
                    sessions.append(
                        {
                            "key": repaired.key,
                            "created_at": repaired.created_at.isoformat(),
                            "updated_at": repaired.updated_at.isoformat(),
                            "title": _metadata_title(repaired.metadata),
                            "preview": next(
                                (
                                    text
                                    for msg in repaired.messages
                                    if (text := _message_preview_text(msg))
                                ),
                                "",
                            ),
                            "path": str(path),
                        }
                    )
                continue
        return sorted(sessions, key=lambda item: item["updated_at"], reverse=True)


class SessionManager:
    """Manage session identity, caching, retention, and persistence."""

    def __init__(
        self,
        workspace: Path,
        *,
        store: SessionStore | None = None,
        sessions_root: Path | None = None,
        _managed_access: "ManagedNamespaceAccess | None" = None,
    ):
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self.sessions_root = (
            Path(sessions_root).expanduser().resolve(strict=False)
            if sessions_root is not None
            else get_default_sessions_root()
        )
        self._custom_store = store
        self._jsonl_store = JsonlSessionStore(
            workspace,
            sessions_root=self.sessions_root,
            _managed_access=_managed_access,
        )
        self._cache: OrderedDict[str, Session] = OrderedDict()
        # Gateway controllers and loop wrappers share this one volatile owner.
        self.desktop_permissions = DesktopPermissionOwner()
        self._mode_bindings: dict[str, tuple[str, int, str, str, int] | None] = {}
        # Preserve identity for sessions held by active callers without retaining idle ones.
        self._overflow_cache: WeakValueDictionary[str, Session] = WeakValueDictionary()
        self._max_cached_sessions = SESSION_CACHE_MAX_SIZE
        self._delete_observer: Callable[[str], None] | None = None
        self._permission_lifecycle_observer: Callable[[
            str, Literal["session_reset", "session_deleted"],
        ], None] | None = None
        self._mode_resolver: SessionModeResolver | None = None
        self._mode_operator = False
        self._cancellation_privacy_lock = Lock()
        self._cancellation_receipts: dict[int, tuple[object, str, bool]] = {}

    def _cancellation_policy_allows(self, key: str) -> bool:
        """Inspect only published policy; caller holds the privacy mutex."""
        cached = self._cache.get(key) or self._overflow_cache.get(key)
        return (self._custom_store is None and self._mode_resolver is None
                and (cached is None or cached.policy.persist))

    @contextmanager
    def track_unadmitted_cancellation(
        self, key: str, *, eligible: bool = True,
    ) -> Generator[object, None, None]:
        """Track just this outstanding call; any transient publication revokes it."""
        receipt = object()
        with self._cancellation_privacy_lock:
            self._cancellation_receipts[id(receipt)] = (
                receipt, key, eligible and self._cancellation_policy_allows(key),
            )
        try:
            yield receipt
        finally:
            with self._cancellation_privacy_lock:
                self._cancellation_receipts.pop(id(receipt), None)

    def record_unadmitted_cancellation(
        self, receipt: object, insert: Callable[[], bool],
    ) -> bool:
        """Authorize an exact live receipt through synchronous insert completion."""
        with self._cancellation_privacy_lock:
            registration = self._cancellation_receipts.get(id(receipt))
            if registration is None or registration[0] is not receipt or not registration[2]:
                return False
            key = registration[1]
        # _remember may run under the files lock: files BEFORE privacy is vital.
        # Neither registration nor publication waits for storage under privacy.
        with self._jsonl_store.cancelled_record_guard(key) as allowed:
            with self._cancellation_privacy_lock:
                if (not allowed or self._cancellation_receipts.get(id(receipt)) != registration
                        or not self._cancellation_policy_allows(key)):
                    return False
                return insert()

    def _install_mode_resolver(
        self,
        resolver: "SessionModeResolver",
        *,
        operator: bool,
    ) -> None:
        """Internal authority hook; direct constructors remain unmanaged."""
        if self._custom_store is not None or self._mode_resolver is not None:
            raise RuntimeError("managed session resolver cannot be installed")
        with self._cancellation_privacy_lock:
            self._mode_resolver = resolver
            self._mode_operator = operator
            for identity, (receipt, key, _) in self._cancellation_receipts.items():
                self._cancellation_receipts[identity] = (receipt, key, False)

    def compose_session_modes(
        self,
        select_constructor: Callable[[], type["SessionModeResolver"]],
        retain_close: Callable[[Callable[[], None]], None],
        *,
        operator: bool,
        verify_registered_modes: bool,
    ) -> None:
        """Compose the owned backend; hand back cleanup, not storage authority."""
        resolver = select_constructor()(self._jsonl_store)

        def close() -> None:
            resolver.close()

        # The outer writer owner must retain cleanup before either fallible step.
        retain_close(close)
        self._install_mode_resolver(resolver, operator=operator)
        if verify_registered_modes:
            resolver.list_modes()

    def copied_artifact_read_operations(self) -> "CopiedArtifactReadOperations | None":
        """Hand off only late reads from the once-selected mode resolver."""
        resolver = self._mode_resolver
        if resolver is None:
            return None
        from nanobot.session.artifact_operator import CopiedArtifactReadOperations

        return CopiedArtifactReadOperations(
            locked_snapshot=lambda key: resolver.locked_snapshot(key),
            read_workspace_id=lambda: resolver.workspace_id,
            read_workspace=lambda: resolver.workspace,
            read_namespace=lambda: resolver.namespace,
        )

    def canonical_export_operations(self) -> "CanonicalExportOperations":
        """Retain only export commands, never the owned resolver or backend."""
        from nanobot.session import session_export

        resolver = self._mode_resolver
        if resolver is None or not self._mode_operator:
            raise session_export.SessionModeError("offline session operator authority is required")
        return session_export.CanonicalExportOperations(
            locked_snapshot=lambda key: resolver.locked_snapshot(key),
            begin_export=lambda key, target, files: resolver.begin_export(key, target, files),
            select_legacy_parser=lambda: self._jsonl_store.parse_legacy_bytes,
        )

    def resolve_session_mode(self, key: str) -> "ResolvedSessionMode":
        if self._mode_resolver is None:
            raise RuntimeError("session namespace is unmanaged")
        return self._mode_resolver.resolve(key)

    @property
    def has_managed_authority(self) -> bool:
        return self._mode_resolver is not None

    def admit_persistent_write(self, key: str) -> "ResolvedSessionMode":
        """Commit canonical authority before an admitted persistent turn writes."""
        if self._mode_resolver is None:
            raise RuntimeError("session namespace is unmanaged")
        return self._mode_resolver.ensure_canonical(key)

    def bootstrap_existing_sessions(self, *, offline_copy: bool) -> int:
        if self._mode_resolver is None or not self._mode_operator:
            raise RuntimeError("offline session operator authority is required")
        result = self._mode_resolver.bootstrap(offline_copy=offline_copy)
        self._cache.clear()
        self._overflow_cache.clear()
        return result

    def convert_legacy_session(self, key: str, *, offline_copy: bool) -> "ResolvedSessionMode":
        if self._mode_resolver is None or not self._mode_operator:
            raise RuntimeError("offline session operator authority is required")
        result = self._mode_resolver.convert(key, offline_copy=offline_copy)
        self.invalidate(key)
        return result

    def recover_session_mode(self, key: str) -> "ResolvedSessionMode":
        if self._mode_resolver is None or not self._mode_operator:
            raise RuntimeError("offline session operator authority is required")
        result = self._mode_resolver.recover(key)
        self.invalidate(key)
        return result

    def reset_session(
        self,
        key: str,
        *,
        owner: object | None = None,
        replacement_events: list[dict[str, Any]] | None = None,
    ) -> Session | None:
        """Sole persistence seam for clearing one managed or unmanaged session."""

        events = replacement_events or []
        self._retire_permissions_for_lifecycle(key, "session_reset")
        if self._mode_resolver is not None:
            resolved = self._mode_resolver.resolve(key)
            if resolved.state in {"absent", "deleted"}:
                self._mode_resolver.ensure_canonical(key)
                self.invalidate(key)
                return self._jsonl_store.source.load_source_snapshot(key).to_session()
            if resolved.state != "active":
                raise RuntimeError("session mode evidence is invalid; offline recovery is required")
            if resolved.mode == "canonical":
                self._mode_resolver.reset(
                    key,
                    cast(list[dict[str, object]], events),
                    owner=owner,
                )
                self.invalidate(key)
                return self._jsonl_store.source.load_source_snapshot(key).to_session()
            self._mode_resolver.reset(
                key,
                cast(list[dict[str, object]], events),
                owner=None,
            )
            self.invalidate(key)
            return self._jsonl_store.load(key)
        source_backend = self.source_backend(key)
        if source_backend is not None:
            snapshot = source_backend.source.load_source_snapshot(key)
            reset = source_backend.source.reset_source_session(
                key,
                snapshot.state,
                events,
                owner=cast("ExecutionOwner | None", owner),
            )
            self.invalidate(key)
            return reset.to_session()
        session = self._cached(key) or self._load(key)
        if session is None:
            session = Session(key=key, messages=deepcopy(events))
            self.save(session, fsync=True)
            return session
        session.replace_messages(events)
        self.save(session, fsync=True)
        return session

    @property
    def _store(self) -> SessionStore:
        return self._custom_store if self._custom_store is not None else self._jsonl_store

    @property
    def sessions_dir(self) -> Path:
        return self._jsonl_store.sessions_dir

    @property
    def legacy_sessions_dir(self) -> Path:
        return self._jsonl_store.legacy_sessions_dir

    def _remember(self, session: Session) -> None:
        """Keep recent sessions strongly cached without duplicating live objects."""
        with self._cancellation_privacy_lock:
            if not session.policy.persist:
                for identity, (receipt, key, eligible) in self._cancellation_receipts.items():
                    if key == session.key and eligible:
                        self._cancellation_receipts[identity] = (receipt, key, False)
            self._overflow_cache.pop(session.key, None)
            self._cache[session.key] = session
            self._cache.move_to_end(session.key)
            while len(self._cache) > self._max_cached_sessions:
                key, evicted = self._cache.popitem(last=False)
                self._overflow_cache[key] = evicted

        if self._mode_resolver is not None:
            self._mode_bindings[session.key] = self._mode_resolver.resolve(session.key).binding

    def _cached(self, key: str) -> Session | None:
        if self._mode_resolver is not None:
            binding = self._mode_resolver.resolve(key).binding
            if key in self._mode_bindings and self._mode_bindings[key] != binding:
                self.invalidate(key)
        session = self._cache.get(key)
        if session is not None:
            self._cache.move_to_end(key)
            return session

        session = self._overflow_cache.get(key)
        if session is not None:
            self._remember(session)
        return session

    def get_cached(self, key: str) -> Session | None:
        """Return a cached session without creating or loading one from disk."""
        return self._cached(key)

    def set_delete_observer(self, observer: Callable[[str], None]) -> None:
        """Observe explicit session deletion for process-local state cleanup."""
        self._delete_observer = observer

    def set_permission_lifecycle_observer(
        self,
        observer: Callable[[str, Literal["session_reset", "session_deleted"]], None],
    ) -> None:
        """Compose permission fencing ahead of destructive source lifecycle work."""
        self._permission_lifecycle_observer = observer

    def _retire_permissions_for_lifecycle(
        self,
        key: str,
        reason: Literal["session_reset", "session_deleted"],
    ) -> None:
        observer = self._permission_lifecycle_observer
        if observer is None:
            return
        try:
            observer(key, reason)
        except WorkspacePermissionConflictError:
            raise SessionWriteConflict() from None

    @staticmethod
    def safe_key(key: str) -> str:
        """Public helper used by HTTP handlers to map an arbitrary key to a stable filename stem."""
        return JsonlSessionStore.safe_key(key)

    @staticmethod
    def _storage_key(key: str) -> str:
        """Collision-resistant encoding for internal session storage filenames."""
        return JsonlSessionStore.storage_key(key)

    @staticmethod
    def _decode_storage_key(stem: str) -> str | None:
        """Reverse _storage_key(): decode a base64url (no-padding) stem back to the original key."""
        return JsonlSessionStore.decode_storage_key(stem)

    @staticmethod
    def decode_storage_key(stem: str) -> str | None:
        """Public decoder for components that inspect canonical session filenames."""
        return SessionManager._decode_storage_key(stem)

    @classmethod
    def _session_key_from_path(cls, path: Path) -> str | None:
        """Decode a session key only from a canonical collision-resistant filename."""
        return JsonlSessionStore.session_key_from_path(path)

    def _get_session_path(self, key: str) -> Path:
        """Get the collision-resistant workspace path for a session."""
        return self._jsonl_store.get_session_path(key)

    def _get_runtime_checkpoint_path(self, key: str) -> Path:
        """Get the private in-flight checkpoint path for a session."""
        return self._jsonl_store.get_runtime_checkpoint_path(key)

    def _get_legacy_lossy_path(self, key: str) -> Path:
        """Previous workspace session path using lossy ':' to '_' replacement."""
        return self._jsonl_store.get_legacy_lossy_path(key)

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.nanobot/sessions/)."""
        return self._jsonl_store.get_legacy_session_path(key)

    @contextmanager
    def locked_session_files(self) -> Generator[Path, None, None]:
        """Guard exceptional direct access to canonical JSONL files."""
        with self._jsonl_store.locked_session_files() as sessions_dir:
            yield sessions_dir

    def source_backend(self, key: str) -> JsonlSessionStore | None:
        """Select strict execution only for explicitly converted default-store sessions."""
        if self._store is not self._jsonl_store:
            return None
        if self._mode_resolver is not None:
            resolved = self._mode_resolver.resolve(key)
            return self._jsonl_store if resolved.state == "active" and resolved.mode == "canonical" else None
        payload = self._jsonl_store.read_metadata(key)
        if payload is not None and "source_schema_version" in payload.get("metadata", {}):
            return self._jsonl_store
        return None

    def read_existing_canonical_source(
        self,
        key: str,
        *,
        expected_binding: WorkspaceAccessSourceBinding | None = None,
    ) -> "tuple[JsonlSessionStore, SourceSnapshot] | None":
        """Read one active canonical source without creating or caching a Session."""
        cached = self.get_cached(key)
        if cached is not None and not cached.policy.persist:
            return None
        backend = self.source_backend(key)
        if backend is None:
            return None
        try:
            snapshot = backend.source.load_source_snapshot(key)
        except FileNotFoundError:
            return None
        if snapshot.state.session_key != key:
            return None
        if expected_binding is not None:
            from nanobot.workspace_access_recovery import parse_workspace_access_source_binding

            try:
                checked = parse_workspace_access_source_binding(expected_binding)
            except (TypeError, ValueError):
                raise SessionWriteConflict() from None
            actual = {
                "store_identity": snapshot.state.store_identity,
                "incarnation": snapshot.state.incarnation,
                "generation": snapshot.state.generation,
            }
            if checked != actual:
                raise SessionWriteConflict()
        return backend, snapshot

    def invalidate_workspace_access_request(self, key: str) -> bool:
        """Persist supersession for accepted ordinary input on an active source."""
        if self._store is not self._jsonl_store:
            return False
        cached = self.get_cached(key)
        if cached is not None and not cached.policy.persist:
            return False
        backend = self.source_backend(key)
        if backend is None:
            return False
        with self.locked_session_files():
            if self._mode_resolver is not None:
                resolved = self._mode_resolver.resolve(key)
                if resolved.state != "active" or resolved.mode != "canonical":
                    return False
            changed = backend.source.invalidate_workspace_access_request(key)
            if changed:
                self.invalidate(key)
            return changed

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        session = self._cached(key)
        if session is not None:
            return session

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._remember(session)
        return session

    def get_or_create_transient(
        self,
        key: str,
        *,
        disabled_tools: Collection[str] = (),
        retain: bool = False,
    ) -> Session:
        """Return non-persistent history, optionally activating a live temporary session."""
        policy = SessionPolicy(
            persist=False,
            log_content=False,
            disabled_tools=frozenset(disabled_tools),
        )
        if retain:
            if self._mode_resolver is not None:
                resolved = self._mode_resolver.resolve(key)
                if resolved.state not in {"absent", "deleted"}:
                    raise RuntimeError("temporary session activation conflicts with session policy")
            elif self._load(key) is not None:
                raise RuntimeError("temporary session activation conflicts with session policy")
            session = self._cached(key)
            if session is not None:
                if session.policy != policy:
                    raise RuntimeError("temporary session activation conflicts with session policy")
                return session
            session = Session(key=key, policy=policy)
            self._remember(session)
            return session
        if self._mode_resolver is not None:
            return Session(key=key, policy=policy)
        session = self.get_cached(key)
        if session is None:
            session = Session(key=key, policy=policy)
            self._remember(session)
        elif session.policy != policy:
            return Session(key=key, policy=policy)
        return session

    def _load(self, key: str) -> Session | None:
        if self._mode_resolver is not None:
            resolved = self._mode_resolver.resolve(key)
            if resolved.state != "active":
                return None
        return self._store.load(key)

    def _repair(self, key: str, *, path: Path | None = None) -> Session | None:
        """Attempt to recover a session from a corrupt JSONL file."""
        return self._jsonl_store.repair(key, path=path)

    def save(self, session: Session, *, fsync: bool = False) -> None:
        """Persist a session and retain it in the cache."""
        if not session.policy.persist:
            return

        if self._mode_resolver is not None:
            resolved = self._mode_resolver.resolve(session.key)
            if resolved.state == "absent":
                resolved = self._mode_resolver.ensure_canonical(session.key)
            if resolved.mode == "canonical":
                raise RuntimeError("canonical session writes require source authority")

        if self._store is not self._jsonl_store:
            with self._jsonl_store.locked_session_files():
                pass
        self._store.save(session, fsync=fsync)
        self._remember(session)

    def save_runtime_checkpoint(self, session: Session) -> None:
        """Persist volatile recovery state without rewriting long history."""
        if not session.policy.persist:
            return
        if self._mode_resolver is not None:
            resolved = self._mode_resolver.resolve(session.key)
            if resolved.mode == "canonical":
                raise RuntimeError("canonical checkpoints require source authority")
        if self._store is self._jsonl_store:
            self._jsonl_store.save_runtime_checkpoint(session)
            self._remember(session)
            return
        # Third-party stores keep their existing all-or-nothing semantics until
        # they opt into a dedicated checkpoint primitive.
        with self._jsonl_store.locked_session_files():
            pass
        self.save(session)

    def rename_model_preset(self, old_name: str, new_name: str) -> int:
        """Rename a session-scoped model preset across durable and live sessions."""
        if old_name == new_name:
            return 0

        if self._mode_resolver is not None:
            self._mode_resolver.list_modes()
        cached = dict(self._overflow_cache.items())
        cached.update(self._cache)
        keys = set(cached)
        keys.update(item["key"] for item in self._store.list_sessions())

        candidates: list[tuple[str, Session, str | None]] = []
        for key in sorted(keys):
            mode: str | None = None
            if self._mode_resolver is not None:
                resolved = self._mode_resolver.resolve(key)
                if resolved.state != "active":
                    continue
                mode = resolved.mode
            session = self._cached(key) or self._load(key)
            if (
                session is not None
                and session.metadata.get(SESSION_MODEL_PRESET_METADATA_KEY) == old_name
            ):
                candidates.append((key, session, mode))

        changed: list[tuple[str, Session, str | None]] = []
        try:
            for key, session, mode in candidates:
                session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = new_name
                changed.append((key, session, mode))
                if mode == "canonical":
                    self.update_session_metadata(
                        key,
                        {SESSION_MODEL_PRESET_METADATA_KEY: new_name},
                        fsync=True,
                    )
                elif session.policy.persist:
                    self.save(session, fsync=True)
                else:
                    self._remember(session)
        except BaseException:
            for key, session, mode in reversed(changed):
                session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = old_name
                try:
                    if mode == "canonical":
                        self.update_session_metadata(
                            key,
                            {SESSION_MODEL_PRESET_METADATA_KEY: old_name},
                            fsync=True,
                        )
                    elif session.policy.persist:
                        self.save(session, fsync=True)
                    else:
                        self._remember(session)
                except Exception:
                    logger.exception(
                        "Failed to roll back managed model preset rename",
                    )
            raise
        return len(changed)

    def flush_all(self) -> int:
        """Re-save every cached session with fsync for durable shutdown.

        Returns the number of sessions flushed.  Errors on individual
        sessions are logged but do not prevent other sessions from being
        flushed.
        """
        flushed = 0
        cached = dict(self._overflow_cache.items())
        cached.update(self._cache)
        for key, session in cached.items():
            try:
                self.save(session, fsync=True)
                flushed += 1
            except Exception:
                if self._mode_resolver is not None:
                    logger.warning("Failed to flush managed session")
                else:
                    logger.warning("Failed to flush session {}", key, exc_info=True)
        return flushed

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)
        self._overflow_cache.pop(key, None)
        self._mode_bindings.pop(key, None)

    def delete_session(self, key: str) -> bool:
        """Delete a persisted session and invalidate its cache entry."""
        self._retire_permissions_for_lifecycle(key, "session_deleted")
        if self._mode_resolver is not None:
            deleted = self._mode_resolver.delete(key)
            if deleted:
                self.invalidate(key)
            if self._delete_observer is not None:
                self._delete_observer(key)
            return deleted
        if self._store is not self._jsonl_store:
            with self._jsonl_store.locked_session_files():
                pass
        deleted = self._store.delete(key)
        self.invalidate(key)
        if self._delete_observer is not None:
            self._delete_observer(key)
        return deleted

    def restore_sessions_to_workspace(self) -> SessionRestoreResult:
        """Restore session files to the pre-relocation path for an explicit rollback."""
        if self._mode_resolver is not None and self._mode_resolver.is_registered():
            raise RuntimeError("managed sessions require verified export")
        return self._jsonl_store.restore_to_workspace()

    def fork_session_before_user_index(
        self,
        source_key: str,
        target_key: str,
        before_user_index: int,
    ) -> Session | None:
        """Create *target_key* from *source_key* before a global user-message index.

        ``before_user_index`` is zero-based over user messages in the full session:
        ``0`` means "before the first user message", ``1`` means "before the
        second user message", and so on. A value equal to the total user-message
        count copies the full session prefix. WebUI assistant-reply forks pass
        the next user index so the selected completed assistant turn is included.
        """
        if before_user_index < 0:
            return None
        if self._mode_resolver is not None:
            source_resolved = self._mode_resolver.resolve(source_key)
            target_resolved = self._mode_resolver.resolve(target_key)
            if (
                source_resolved.state != "active"
                or target_resolved.state not in {"absent", "deleted"}
            ):
                return None
            if self._mode_resolver.fork(source_key, target_key, before_user_index) is None:
                return None
            self.invalidate(target_key)
            created = self._jsonl_store.source.load_source_snapshot(target_key).to_session()
            self._remember(created)
            return created
        source = self._cached(source_key) or self._load(source_key)
        if source is None:
            return None

        source_messages = source.messages
        copied: list[dict[str, Any]] = []
        user_index = 0
        found_target = False
        for message in source_messages:
            if message.get("role") == "user" and not is_hidden_history_message(message):
                if user_index == before_user_index:
                    found_target = True
                    break
                user_index += 1
            copied.append(public_history_message(message))
        if user_index == before_user_index:
            found_target = True
        if not found_target:
            return None

        metadata = deepcopy(source.metadata)
        for key in _FORK_VOLATILE_METADATA_KEYS | {
            "artifact_manifest",
            "context_checkpoint",
        }:
            metadata.pop(key, None)

        source_archive = source.last_archived
        last_consolidated = min(source_archive, len(copied))
        if source_archive > len(copied):
            metadata.pop("_last_summary", None)
            last_consolidated = 0

        now = datetime.now()
        target = Session(
            key=target_key,
            messages=copied,
            created_at=now,
            updated_at=now,
            metadata=metadata,
            last_consolidated=last_consolidated,
        )
        self.save(target, fsync=True)
        return target

    def read_session_file(self, key: str) -> dict[str, Any] | None:
        """Read a session without populating the cache."""
        if self._mode_resolver is not None and self._mode_resolver.resolve(key).state != "active":
            return None
        return cast(dict[str, Any] | None, self._store.read(key))

    def read_session_snapshot(self, key: str) -> Session | None:
        """Load a detached session snapshot without populating the runtime cache."""
        if self._mode_resolver is not None and self._mode_resolver.resolve(key).state != "active":
            return None
        return self._store.load(key)

    def read_session_metadata(self, key: str) -> dict[str, Any] | None:
        """Read session metadata without loading the transcript."""
        if self._mode_resolver is not None and self._mode_resolver.resolve(key).state != "active":
            return None
        return cast(dict[str, Any] | None, self._store.read_metadata(key))

    def update_session_metadata(
        self,
        key: str,
        updates: dict[str, Any],
        *,
        fsync: bool = False,
    ) -> bool:
        """Atomically update metadata without replacing session history."""
        if self._mode_resolver is not None:
            return self.upsert_session_metadata(key, updates, fsync=fsync)
        if self._store is not self._jsonl_store:
            with self._jsonl_store.locked_session_files():
                pass
        updated = self._store.update_metadata(key, updates, fsync=fsync)
        if updated and (session := self.get_cached(key)) is not None:
            is_source = "source_schema_version" in session.metadata
            if self._store is self._jsonl_store:
                persisted = self._store.read_metadata(key)
                if persisted is None:
                    self.invalidate(key)
                    return updated
                is_source = "source_schema_version" in persisted["metadata"]
                if is_source and "source_schema_version" not in session.metadata:
                    self.invalidate(key)
                    return updated
            accepted = accepted_metadata_updates(updates, source=is_source)
            Session.publish_committed_metadata_update(session, accepted)
        return updated

    def prepare_busy_inbox_metadata(self, key: str) -> None:
        """Admit storage before the inbox takes its short files transaction.

        Existing sessions do not reacquire execution ownership: input must remain
        admissible while their task owns the execution lock. Creation follows the
        established migration -> execution -> files ordering instead.
        """
        if self._store is not self._jsonl_store:
            raise RuntimeError("busy inbox requires the admitted JSONL authority")
        cached = self.get_cached(key)
        if cached is not None and not cached.policy.persist:
            raise RuntimeError("temporary inbox requires a disjoint recovery journal")
        if self.read_session_metadata(key) is None:
            self.upsert_session_metadata(key, {}, create_if_absent=True, fsync=True)

    def commit_busy_inbox_metadata(
        self, key: str, expected: object, replacement: dict[str, object],
    ) -> None:
        """Dedicated inbox boundary under the admitted namespace's storage lock."""
        if self._store is not self._jsonl_store:
            raise RuntimeError("busy inbox requires the admitted JSONL authority")
        cached = self.get_cached(key)
        if cached is not None and not cached.policy.persist:
            raise RuntimeError("temporary inbox requires a disjoint recovery journal")
        with self.locked_session_files():
            if self._mode_resolver is not None and self._mode_resolver.resolve(key).state != "active":
                raise SessionWriteConflict()
            self._jsonl_store.commit_busy_inbox_metadata(key, expected, replacement)
            self.invalidate(key)

    def commit_workspace_access_request(
        self,
        key: str,
        *,
        expected_binding: WorkspaceAccessSourceBinding,
        expected_record: WorkspaceAccessRecord | None,
        replacement_record: WorkspaceAccessRecord | None,
        expected_latest_input_event_id: str | None = None,
        expected_workspace_scope: WorkspaceScopeMetadata | None = None,
        replacement_workspace_scope: WorkspaceScopeMetadata | None = None,
    ) -> WorkspaceAccessSourceBinding:
        """Commit one recovery field against the current persistent source owner."""
        if self._store is not self._jsonl_store:
            raise SessionWriteConflict()
        if expected_record is not None:
            expected_record = parse_workspace_access_record(expected_record)
        if replacement_record is not None:
            replacement_record = parse_workspace_access_record(replacement_record)
        cached = self.get_cached(key)
        if cached is not None and not cached.policy.persist:
            raise SessionWriteConflict()
        with self.locked_session_files():
            if self._mode_resolver is not None:
                resolved = self._mode_resolver.resolve(key)
                if resolved.state != "active" or resolved.mode != "canonical":
                    raise SessionWriteConflict()
            snapshot = self._jsonl_store.source.commit_workspace_access_request(
                key,
                expected_binding=expected_binding,
                expected_record=expected_record,
                replacement_record=replacement_record,
                expected_latest_input_event_id=expected_latest_input_event_id,
                expected_workspace_scope=expected_workspace_scope,
                replacement_workspace_scope=replacement_workspace_scope,
            )
            self.invalidate(key)
        return {
            "store_identity": snapshot.state.store_identity,
            "incarnation": snapshot.state.incarnation,
            "generation": snapshot.state.generation,
        }

    def upsert_session_metadata(
        self,
        key: str,
        updates: dict[str, Any],
        *,
        removals: Collection[str] = (),
        create_if_absent: bool = False,
        fsync: bool = False,
    ) -> bool:
        """Set/remove portable metadata, publishing cache changes only after commit."""
        if not isinstance(cast(object, updates), dict) or any(not isinstance(cast(object, name), str) for name in updates):
            raise TypeError("invalid session metadata")
        if isinstance(removals, (str, bytes)) or any(not isinstance(cast(object, name), str) for name in removals):
            raise TypeError("invalid session metadata")
        accepted = accepted_metadata_updates(updates, source=True)
        accepted_removals = tuple(
            name for name in removals if name not in _RESERVED_DURABLE_METADATA
            and name not in {"runtime_checkpoint", "pending_user_turn", "pending_user_followups", "webui_recovery"}
        )
        try:
            json.dumps(accepted, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise ValueError("invalid session metadata") from None
        if self._mode_resolver is not None:
            resolved = self._mode_resolver.resolve(key)
            if resolved.state in {"absent", "deleted"}:
                if not create_if_absent:
                    return False
                resolved = self._mode_resolver.ensure_canonical(key)
            if resolved.state != "active":
                raise RuntimeError("session mode evidence is invalid; offline recovery is required")
            if resolved.mode == "canonical":
                self._jsonl_store.source.update_metadata(key, accepted, removals=accepted_removals)
                updated = True
            else:
                updated = self._jsonl_store.update_metadata(
                    key, accepted, removals=accepted_removals, fsync=fsync
                )
            if updated:
                self.invalidate(key)
            return updated
        if self._store is self._jsonl_store:
            created = False
            if create_if_absent and self.read_session_snapshot(key) is None:
                # Creation commits an empty durable baseline, not the cached
                # caller's unsaved history. Publish only the metadata below.
                self._jsonl_store.save(Session(key), fsync=fsync)
                created = True
            updated = self._jsonl_store.update_metadata(
                key, accepted, removals=accepted_removals, fsync=fsync
            )
            if not updated:
                return False
            session = self.get_cached(key)
            if session is None:
                if created and (committed := self._jsonl_store.load(key)) is not None:
                    self._remember(committed)
                return True
            persisted = self._jsonl_store.read_metadata(key)
            if persisted is None or (
                "source_schema_version" in persisted["metadata"]
                and "source_schema_version" not in session.metadata
            ):
                self.invalidate(key)
                return True
            if not Session.publish_committed_metadata_upsert(
                session, accepted, accepted_removals,
                lambda session_key: self._jsonl_store.load(session_key),
            ):
                self.invalidate(key)
                return True
            return updated
        return self.update_session_metadata(key, accepted, fsync=fsync)

    def list_sessions(self) -> list[dict[str, Any]]:
        if self._mode_resolver is not None:
            self._mode_resolver.list_modes()
        items = cast(list[dict[str, Any]], self._store.list_sessions())
        return [item for item in items if not is_busy_recovery_key(item.get("key"))]


class SessionWriteConflict(Exception):  # noqa: N818 - public conflict contract
    """Raised when a stale session attempts a replacement write.

    The exception text is intentionally fixed and content-free: session
    messages may contain secrets and must never be copied into an error.
    """

    DEFAULT_MESSAGE = "Session changed concurrently; reload and retry."

    def __init__(self) -> None:
        super().__init__(self.DEFAULT_MESSAGE)


class SessionStorageError(Exception):
    """Existing evidence could not be read safely; never include its content."""

    def __init__(self) -> None:
        super().__init__("Session storage is invalid; recovery is required.")


def _storage_counter(value: object) -> int:
    if type(value) is not int or value < 0:
        raise SessionStorageError()
    return value


def _merge_session_metadata(session: Session, disk: Session) -> dict[str, Any]:
    """Apply only caller changes to the disk snapshot, preserving reserved state."""
    return Session.merge_commit_metadata(session, disk)
