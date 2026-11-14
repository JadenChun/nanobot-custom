"""Connection-owned Temporary Chat behavior for the WebUI."""

from __future__ import annotations

import hashlib
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from nanobot.attachments._media import (
    MediaFence,
    atomic_write,
    configured_custody,
    generation,
    object_json,
    read_bytes,
)
from nanobot.attachments.models import AttachmentRecord
from nanobot.attachments.store import AttachmentStore
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_SESSION_DISCARD,
    InboundMessage,
)
from nanobot.bus.queue import MessageBus
from nanobot.config.paths import get_runtime_subdir
from nanobot.security.desktop_permissions import PermissionSourceBinding
from nanobot.security.workspace_access import WorkspaceScope
from nanobot.session.manager import Session, SessionManager
from nanobot.webui.workspaces import WebUIWorkspaceController, webui_workspace_state_path

_TEMPORARY_CHAT_DISABLED_TOOLS = frozenset({
    "create_goal",
    "update_goal",
    "spawn",
    "cron",
    "select_workspace",
})
_TEMPORARY_CHAT_COMMANDS = frozenset({"/model", "/stop"})


class TemporaryChatError(ValueError):
    """A stable WebUI protocol error for a Temporary Chat operation."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class TemporaryChatMessagePolicy:
    """Server-owned message rules for one active Temporary Chat."""

    session_key: str
    workspace_scope: WorkspaceScope
    source_binding: PermissionSourceBinding
    require_existing_session: bool = True
    hydrate_transcript: bool = False
    persist_transcript: bool = False


class WebUITemporaryChats:
    """Own Temporary Chat creation, policy, attachments, and disposal."""

    def __init__(
        self,
        *,
        bus: MessageBus,
        session_manager: SessionManager | None,
        workspaces: WebUIWorkspaceController,
        logger: Any,
        channel_name: str = "websocket",
    ) -> None:
        self._bus = bus
        self._sessions = session_manager
        self._workspaces = workspaces
        self._logger = logger
        self._channel_name = channel_name
        self._owners: dict[str, object] = {}
        self._cleanup_retry_store = (
            get_runtime_subdir("temporary-chat-cleanup") / "retained.json"
        )
        custody = configured_custody()
        self._media_store = AttachmentStore(custody.root / "attachments", media_root=custody.root)
        # The gateway's workspace/permission owners can create this display
        # root before a first transcript. Declare the actual selected path so
        # exclusive temporary uploads can still be safely disposed.
        self._media_store.custody.declare("webui", webui_workspace_state_path().parent.resolve(strict=True))
        self._media_store.custody.declare("temporary", self._cleanup_retry_store.parent.resolve(strict=True))
        self._cleanup_entries: dict[str, list[dict[str, object]]] = {}
        self._cleared_cleanup_chats: set[str] = set()
        self._cleanup_retries = self._load_cleanup_retries()
        self._owner_chat_ids: dict[object, set[str]] = {}
        # Keep active sessions alive if the bounded manager cache evicts them
        # between WebUI turns. SessionPolicy remains the authority below.
        self._active_sessions: dict[str, Session] = {}
        # Retain policy-derived tombstones until shutdown so late outbound
        # events cannot create a durable transcript after a chat is discarded.
        self._known_transient_chat_ids: set[str] = set()
        self._media_paths: dict[str, set[str]] = {}
        self._incarnations: dict[str, str] = {}
        self._access_modes: dict[str, Literal["restricted", "full"]] = {}

    def _session_key(self, chat_id: str) -> str:
        return f"{self._channel_name}:{chat_id}"

    def _load_cleanup_retries(self) -> dict[str, set[str]]:
        try:
            raw = object_json(read_bytes(self._cleanup_retry_store))
        except FileNotFoundError:
            return {}
        if (set(raw) != {"version", "chats"} or type(raw["version"]) is not int
                or raw["version"] != 1 or not isinstance(raw["chats"], dict)):
            raise TemporaryChatError("temporary_cleanup_recovery_required")
        retained: dict[str, set[str]] = {}
        for raw_chat_id, raw_entries in cast(dict[object, object], raw["chats"]).items():
            if not isinstance(raw_chat_id, str) or not raw_chat_id or not isinstance(raw_entries, list):
                raise TemporaryChatError("temporary_cleanup_recovery_required")
            kept: set[str] = set()
            entries: list[dict[str, object]] = []
            for value in cast(list[object], raw_entries):
                if not isinstance(value, dict):
                    raise TemporaryChatError("temporary_cleanup_recovery_required")
                entry = cast(dict[str, object], value)
                path, tokens, asset_generation = entry.get("path"), entry.get("tokens"), entry.get("generation")
                if (set(entry) != {"path", "tokens", "generation"} or not isinstance(path, str)
                        or not Path(path).is_absolute() or ".." in Path(path).parts
                        or not Path(path).is_relative_to(self._media_store.media_root)
                        or not isinstance(tokens, list) or not tokens
                        or any(not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token) for token in cast(list[object], tokens))
                        or not isinstance(asset_generation, list) or len(cast(list[object], asset_generation)) != 5
                        or any(type(item) is not int for item in cast(list[object], asset_generation))):
                    raise TemporaryChatError("temporary_cleanup_recovery_required")
                kept.add(path)
                entries.append(entry)
            if kept:
                retained[raw_chat_id] = kept
                self._cleanup_entries[raw_chat_id] = entries
        return retained

    def _save_cleanup_retries(self, gate: MediaFence | None = None) -> None:
        path = self._cleanup_retry_store
        custody = self._media_store.custody
        with custody.fence(create=True) if gate is None else nullcontext(gate) as active_gate:
            active_gate.require(custody)
            payload: dict[str, object]
            try:
                payload = object_json(read_bytes(path))
            except FileNotFoundError:
                payload = {"version": 1, "chats": {}}
            if set(payload) != {"version", "chats"} or payload["version"] != 1 or not isinstance(payload["chats"], dict):
                raise TemporaryChatError("temporary_cleanup_recovery_required")
            chats = cast(dict[str, object], payload["chats"])
            chats.update(self._cleanup_entries)
            for chat_id in self._cleared_cleanup_chats:
                chats.pop(chat_id, None)
            atomic_write(path, payload)
            self._cleared_cleanup_chats.clear()

    def _cached_session_is_transient(self, chat_id: str) -> bool:
        if self._sessions is None:
            return False
        session = self._sessions.get_cached(self._session_key(chat_id))
        return session is not None and not session.policy.persist

    def create(self, owner: object, *, trusted_webui: bool) -> str:
        # A newly created chat id has no retained cleanup from an older chat.
        """Create a server-identified chat owned by one authenticated WebUI connection."""
        if not trusted_webui:
            raise TemporaryChatError("access_denied")
        if self._sessions is None:
            raise TemporaryChatError("temporary_chat_unavailable")

        chat_id = str(uuid.uuid4())
        self._cleanup_retries.pop(chat_id, None)
        session = self._sessions.get_or_create_transient(
            self._session_key(chat_id),
            disabled_tools=_TEMPORARY_CHAT_DISABLED_TOOLS,
            retain=True,
        )
        if session.policy.persist:
            raise RuntimeError("Temporary Chat must use a non-persistent session policy")
        self._owners[chat_id] = owner
        self._owner_chat_ids.setdefault(owner, set()).add(chat_id)
        self._active_sessions[chat_id] = session
        self._known_transient_chat_ids.add(chat_id)
        self._incarnations[chat_id] = str(uuid.uuid4())
        self._access_modes[chat_id] = "restricted"
        return chat_id

    def message_policy(
        self,
        owner: object,
        chat_id: str,
        content: str,
    ) -> TemporaryChatMessagePolicy | None:
        """Return Temporary Chat rules, or ``None`` for an ordinary chat."""
        if not self._cached_session_is_transient(chat_id):
            if chat_id in self._known_transient_chat_ids:
                raise TemporaryChatError("temporary_chat_unavailable")
            return None
        if self._owners.get(chat_id) is not owner or self._sessions is None:
            raise TemporaryChatError("temporary_chat_unavailable")

        session = self._sessions.get_cached(self._session_key(chat_id))
        if session is None:
            raise TemporaryChatError("temporary_chat_unavailable")

        command = content.strip().split(maxsplit=1)[0].lower() if content.strip() else ""
        if command.startswith("/") and command not in _TEMPORARY_CHAT_COMMANDS:
            raise TemporaryChatError("temporary_chat_command_rejected")

        mode = self._access_modes.get(chat_id, "restricted")
        binding = self.binding(owner, chat_id)
        if binding is None:
            raise TemporaryChatError("temporary_chat_unavailable")
        return TemporaryChatMessagePolicy(
            session_key=self._session_key(chat_id),
            workspace_scope=self._workspaces.ephemeral_scope(mode),
            source_binding=binding,
        )

    def binding(
        self,
        owner: object | None,
        chat_id: str,
    ) -> PermissionSourceBinding | None:
        """Return the private ephemeral source binding for its active owner.

        ``owner=None`` is reserved for the server-side permission controller and
        is never reachable from a client field.
        """
        if owner is not None and self._owners.get(chat_id) is not owner:
            return None
        incarnation = self._incarnations.get(chat_id)
        if incarnation is None:
            return None
        return {
            "store_identity": f"temporary:{chat_id}",
            "incarnation": incarnation,
            "generation": 1,
        }

    def access_mode(self, chat_id: str) -> Literal["restricted", "full"]:
        return self._access_modes.get(chat_id, "restricted")

    def set_access_mode(self, chat_id: str, mode: Literal["restricted", "full"]) -> None:
        if chat_id not in self._incarnations:
            raise TemporaryChatError("temporary_chat_unavailable")
        if mode not in {"restricted", "full"}:
            raise TemporaryChatError("invalid_request_settings")
        self._access_modes[chat_id] = mode

    def validate_attach(self, chat_id: str) -> None:
        """Reject attempts to recover a non-persistent session."""
        if not self._cached_session_is_transient(chat_id):
            if chat_id in self._known_transient_chat_ids:
                raise TemporaryChatError("temporary_chat_unavailable")
            return
        raise TemporaryChatError("temporary_chat_unavailable")

    def validate_workspace_update(self, chat_id: str) -> None:
        """Prevent non-persistent sessions from acquiring durable workspace state."""
        if self._cached_session_is_transient(chat_id):
            raise TemporaryChatError("temporary_chat_workspace_rejected")
        if chat_id in self._known_transient_chat_ids:
            raise TemporaryChatError("temporary_chat_unavailable")

    def register_media(self, owner: object, chat_id: str, paths: list[str],
                       *, records: tuple[AttachmentRecord, ...] = ()) -> None:
        if not paths:
            return
        if self._owners.get(chat_id) is not owner:
            raise TemporaryChatError("temporary_chat_unavailable")
        store = self._media_store
        with store.custody.fence(create=True) as gate:
            pending = store.custody.records("pending")
            owned_ids = {record.id for record in records}
            entries = self._cleanup_entries.setdefault(chat_id, [])
            existing_paths = {str(entry["path"]) for entry in entries}
            for raw_path in paths:
                path = Path(raw_path).expanduser().absolute()
                if not path.is_relative_to(store.media_root) or ".." in path.parts:
                    raise TemporaryChatError("unsupported_temporary_media")
                data = read_bytes(path)
                tokens = [str(row["token"]) for row in pending if row["publication_id"] in owned_ids and row["path"] == str(path)]
                if str(path) in existing_paths:
                    entry = next(entry for entry in entries if entry["path"] == str(path))
                    entry["tokens"] = sorted(set(cast(list[str], entry["tokens"])) | set(tokens))
                    continue
                if not tokens:
                    tokens = [store.custody.pin(gate, publication_id=f"temporary:{chat_id}:{uuid.uuid4().hex}",
                                                 path=path, sha256=hashlib.sha256(data).hexdigest())]
                entries.append({"path": str(path), "tokens": tokens, "generation": list(generation(path.lstat()))})
            self._save_cleanup_retries(gate)
            self._media_paths.setdefault(chat_id, set()).update(paths)

    def chat_ids_for_owner(self, owner: object) -> tuple[str, ...]:
        return tuple(self._owner_chat_ids.get(owner, ()))

    def owns(self, owner: object, chat_id: str) -> bool:
        return self._owners.get(chat_id) is owner

    def should_persist_transcript(self, chat_id: str) -> bool:
        """Apply the session policy and retain it for late events after disposal."""
        return (
            not self._cached_session_is_transient(chat_id)
            and chat_id not in self._known_transient_chat_ids
        )

    def _discard_media(self, chat_id: str) -> set[str]:
        paths = self._media_paths.pop(chat_id, set()) | self._cleanup_retries.get(chat_id, set())
        removed: set[Path]
        try:
            removed = self._media_store.custody.cleanup_temporary(chat_id)
        except (OSError, ValueError, TypeError):
            removed = set()
        failed = paths - {str(path) for path in removed}
        if failed:
            self._cleanup_retries[chat_id] = failed
        else:
            self._cleanup_retries.pop(chat_id, None)
            self._cleanup_entries.pop(chat_id, None)
            self._cleared_cleanup_chats.add(chat_id)
        if failed:
            self._cleanup_entries[chat_id] = [entry for entry in self._cleanup_entries.get(chat_id, []) if entry["path"] in failed]
        self._save_cleanup_retries()
        return failed

    def pending_cleanup(self, chat_id: str) -> tuple[str, ...]:
        """Return retained temporary-chat cleanup paths awaiting a retry."""
        return tuple(sorted(self._cleanup_retries.get(chat_id, ())))

    def retry_cleanup(self, chat_id: str) -> tuple[str, ...]:
        """Retry owned temporary-chat cleanup; return any still-retained paths."""
        failed = self._discard_media(chat_id)
        return tuple(sorted(failed))

    def _forget_owner(self, owner: object, chat_id: str) -> None:
        self._owners.pop(chat_id, None)
        self._incarnations.pop(chat_id, None)
        self._access_modes.pop(chat_id, None)
        chat_ids = self._owner_chat_ids.get(owner)
        if chat_ids is None:
            return
        chat_ids.discard(chat_id)
        if not chat_ids:
            self._owner_chat_ids.pop(owner, None)

    async def discard(self, owner: object, chat_id: str) -> None:
        """Forget one owned chat and cancel any active work through the message bus."""
        if (
            not self._cached_session_is_transient(chat_id)
            or self._owners.get(chat_id) is not owner
        ):
            raise TemporaryChatError("temporary_chat_unavailable")

        session_key = self._session_key(chat_id)
        binding = self.binding(None, chat_id)
        if binding is not None:
            try:
                self._workspaces.discard_temporary_permissions(chat_id, binding)
            except Exception:
                self._logger.warning(
                    "failed to retire a temporary Full permission",
                )
        self._forget_owner(owner, chat_id)
        self._active_sessions.pop(chat_id, None)
        self._discard_media(chat_id)
        if self._sessions is not None:
            self._sessions.invalidate(session_key)
        await self._bus.publish_inbound(
            InboundMessage(
                channel=self._channel_name,
                sender_id="webui",
                chat_id=chat_id,
                content="",
                metadata={
                    INBOUND_META_RUNTIME_CONTROL: RUNTIME_CONTROL_SESSION_DISCARD,
                },
                session_key_override=session_key,
            )
        )

    def close(self) -> None:
        """Release process-local resources during gateway shutdown."""
        for chat_id in tuple(self._owners):
            self._discard_media(chat_id)
            if self._sessions is not None:
                self._sessions.invalidate(self._session_key(chat_id))
        for chat_id in tuple(self._cleanup_retries):
            remaining = self.retry_cleanup(chat_id)
            if remaining:
                self._logger.warning(
                    "temporary cleanup retained {} path(s) for {}",
                    len(remaining),
                    chat_id,
                )
        self._owners.clear()
        self._owner_chat_ids.clear()
        self._active_sessions.clear()
        self._known_transient_chat_ids.clear()
