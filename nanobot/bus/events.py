"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from nanobot.attachments.models import AttachmentRecord

if TYPE_CHECKING:
    from nanobot.events import AgentEvent
    from nanobot.security.workspace_access import WorkspaceScope

# Optional ``OutboundMessage.metadata`` key for structured, channel-agnostic UI
# payloads. Value is JSON-serializable with at least ``kind``; rich clients may
# render it and other channels may ignore unknown keys.
OUTBOUND_META_AGENT_UI = "_agent_ui"

# Internal-only inbound metadata minted by trusted transports and runtime
# services. Never accept these keys verbatim from an untrusted client.
INBOUND_META_RUNTIME_CONTROL = "_runtime_control"
INBOUND_META_USER_SHELL = "_user_shell"
RUNTIME_CONTROL_ACK = "_ack"
RUNTIME_CONTROL_IMAGE_GENERATION_RELOAD = "image_generation_reload"
RUNTIME_CONTROL_SESSION_DISCARD = "session_discard"


@dataclass(init=False)
class InboundMessage:
    """Message received from a chat channel."""

    channel: str  # telegram, discord, slack, whatsapp
    sender_id: str  # User identifier
    chat_id: str  # Chat/channel identifier
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    attachments: tuple[AttachmentRecord, ...] = ()  # Canonical server-owned records
    _media_paths: tuple[str, ...] = field(default=(), repr=False, compare=False, init=False)
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data
    session_key_override: str | None = None  # Optional override for thread-scoped sessions
    require_existing_session: bool = False
    input_role: Literal["user", "system"] | None = None
    # Trusted local jobs may retain a turn scope without changing the chat's
    # selected project. This capability is never decoded from wire metadata.
    workspace_scope: "WorkspaceScope | None" = field(default=None, repr=False)

    def __init__(
        self,
        channel: str,
        sender_id: str,
        chat_id: str,
        content: str,
        timestamp: datetime | None = None,
        media: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key_override: str | None = None,
        require_existing_session: bool = False,
        input_role: Literal["user", "system"] | None = None,
        workspace_scope: "WorkspaceScope | None" = None,
        attachments: tuple[AttachmentRecord, ...] = (),
    ) -> None:
        self.channel = channel
        self.sender_id = sender_id
        self.chat_id = chat_id
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.attachments = tuple(attachments)
        self._media_paths = tuple(str(path) for path in (media or ()) if path)
        self.metadata = dict(metadata or {})
        self.session_key_override = session_key_override
        self.require_existing_session = require_existing_session
        self.input_role = input_role
        self.workspace_scope = workspace_scope
        if media:
            self._adopt_media(media)
        if self.workspace_scope is not None:
            from nanobot.security.workspace_access import WorkspaceScope

            if not isinstance(cast(object, self.workspace_scope), WorkspaceScope):
                raise TypeError("workspace scope requires a trusted local WorkspaceScope")

    def _adopt_media(self, media: list[str] | tuple[str, ...]) -> None:
        from nanobot.attachments.adoption import adopt_media_paths

        result = adopt_media_paths(
            list(self._media_paths),
            source=self.channel,
            existing=self.attachments,
        )
        if result.records:
            self.attachments = (*self.attachments, *result.records)
        content = self.content
        for original, canonical in result.replacements:
            content = content.replace(original, canonical)
            for marker in (
                "image", "file", "video", "animation", "voice", "audio", "Attachment",
            ):
                content = content.replace(f"[{marker}: {canonical}]", "")
        for path in self._media_paths:
            name = Path(path).name
            for marker in (
                "image", "file", "video", "animation", "voice", "audio", "attachment",
            ):
                content = content.replace(f"[{marker}: {name}]", "")
            content = content.replace(f"[attachment: {name} — saved to {path}]", "")
        if content != self.content:
            self.content = content

    @property
    def media(self) -> list[str]:
        """Compatibility view of adapter acquisition paths.

        Canonical consumers must read ``attachments``. The paths are retained
        only so adapters and older tests can inspect what they handed in.
        """
        return list(self._media_paths)

    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"

    @property
    def is_user_input(self) -> bool:
        """Whether this message should enter the conversation as user input."""
        if self.input_role is not None:
            return self.input_role == "user"
        return self.channel != "system"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel.

    ``event`` carries internal runtime/UI semantics. ``metadata`` is reserved
    for channel routing context (``message_id``, thread ids, etc.) and optional
    ``OUTBOUND_META_AGENT_UI`` blobs for rich clients.
    """

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    buttons: list[list[str]] = field(default_factory=list)
    event: "AgentEvent | None" = None
