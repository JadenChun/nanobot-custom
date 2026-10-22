"""Telegram channel implementation using python-telegram-bot."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
import unicodedata
import weakref
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
)
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    MessageEntity,
    ReactionTypeEmoji,
    ReplyParameters,
    Update,
    User,
)
from telegram.error import BadRequest, InvalidToken, NetworkError, TimedOut
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.request import BaseRequest, HTTPXRequest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import ProgressEvent
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.delivery_contract import (
    ChannelDeliveryPreflightError,
    ChannelDeliveryRejectedError,
    ChannelTextPartReceipt,
    ChannelTextReceipt,
    PartialChannelTextDeliveryError,
)
from nanobot.channels.routed_delivery import (
    RoutedAttachment,
    RoutedDeliveryError,
    RoutedDeliveryReceipt,
    RoutedDeliveryRequest,
    RoutedPartKind,
    RoutedPartReceipt,
)
from nanobot.command.builtin import build_help_text
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base
from nanobot.security.network import validate_url_target
from nanobot.session.busy_inbox import InboxConflictError, TelegramAlbumMember
from nanobot.utils.helpers import split_message
from nanobot.utils.logging_bridge import redirect_lib_logging

if TYPE_CHECKING:
    from nanobot.channels.telegram.busy_inbox import TelegramBusyInbox

TELEGRAM_MAX_MESSAGE_LEN = 4000  # Telegram message character limit
# Telegram's actual API limit is 4096; we split raw markdown at 4000 as a
# safety margin for mid-stream edits (plain text).  On stream end, we split
# raw markdown into chunks whose rendered HTML fits Telegram's true 4096-char
# boundary so the final rendered message never overflows.
TELEGRAM_HTML_MAX_LEN = 4096
TELEGRAM_REPLY_CONTEXT_MAX_LEN = TELEGRAM_MAX_MESSAGE_LEN  # Max length for reply context in user message

# python-telegram-bot exposes a six-parameter Application generic. Nanobot
# doesn't customize its context/data/job-queue types, so keep that SDK boundary
# explicit rather than allowing unspecialized generics to spread Unknown.
TelegramApplication: TypeAlias = Application[Any, Any, Any, Any, Any, Any]
_T = TypeVar("_T")

# A healthy getUpdates long poll completes every ~10s even with no traffic;
# PTB retries timeouts silently, so stalls must be detected here.
POLL_STALE_SECONDS = 120.0
POLL_WATCH_INTERVAL = 1.0
RESTART_BACKOFF_INITIAL_SECONDS = 5.0
RESTART_BACKOFF_MAX_SECONDS = 300.0
# How long a send waits out a rebuild; short because ChannelManager dispatches
# every channel from one serial loop.
APP_RESTART_SEND_WAIT_SECONDS = 2.0


class _LivenessTrackedRequest(BaseRequest):
    """Wrap the getUpdates request pool, reporting each completed round trip."""

    __slots__ = ("inner", "_on_round_trip")

    def __init__(self, inner: BaseRequest, on_round_trip: Callable[[], None]) -> None:
        super().__init__()
        self.inner = inner
        self._on_round_trip = on_round_trip

    @property
    def read_timeout(self) -> float | None:
        return self.inner.read_timeout

    async def initialize(self) -> None:
        await self.inner.initialize()

    async def shutdown(self) -> None:
        await self.inner.shutdown()

    async def do_request(self, *args: Any, **kwargs: Any) -> tuple[int, bytes]:
        result = await self.inner.do_request(*args, **kwargs)
        self._on_round_trip()
        return result


def _split_telegram_markdown(content: str, max_len: int) -> list[str]:
    """Split raw Telegram Markdown without leaving fenced code blocks unbalanced."""
    if not content:
        return []
    content = content.lstrip()
    if not content:
        return []
    if len(content) <= max_len:
        return [content]

    def fence_line(fence_pos: int) -> str:
        line_end = content.find("\n", fence_pos)
        if line_end < 0:
            return content[fence_pos:]
        return content[fence_pos:line_end]

    def split_inside_fenced_code_block(pos: int) -> tuple[bool, int, str]:
        if content[:pos].count("```") % 2 == 0:
            return False, -1, ""
        opening = content.rfind("```", 0, pos)
        if opening < 0:
            return True, -1, "```"
        return True, opening, fence_line(opening)

    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break

        cut = content[:max_len]
        pos = cut.rfind("\n")
        if pos <= 0:
            pos = cut.rfind(" ")
        if pos <= 0:
            pos = max_len

        inside_code, opening, fence = split_inside_fenced_code_block(pos)
        if inside_code:
            if opening > 0:
                pos = opening
            else:
                closing = "\n```"
                min_code_pos = len(fence)
                if content.startswith(fence + "\n"):
                    min_code_pos += 1
                # When the only break in range is the opening fence newline,
                # cutting there re-emits the same fence and never advances.
                if pos < min_code_pos:
                    if min_code_pos + len(closing) >= max_len:
                        chunks.append(content[:max_len])
                        content = content[max_len:].lstrip()
                        continue
                    budget = max_len - len(closing)
                    recut = content[:budget]
                    adjusted = recut.rfind("\n", min_code_pos)
                    if adjusted < min_code_pos:
                        adjusted = recut.rfind(" ", min_code_pos)
                    pos = adjusted if adjusted > min_code_pos else budget
                elif pos + len(closing) > max_len:
                    budget = max_len - len(closing)
                    if budget <= min_code_pos:
                        chunks.append(content[:max_len])
                        content = content[max_len:].lstrip()
                        continue
                    recut = content[:budget]
                    adjusted = recut.rfind("\n", min_code_pos)
                    if adjusted < min_code_pos:
                        adjusted = recut.rfind(" ", min_code_pos)
                    pos = adjusted if adjusted > min_code_pos else budget
                if pos <= min_code_pos:
                    chunks.append(content[:max_len])
                    content = content[max_len:].lstrip()
                    continue
                chunks.append(content[:pos] + closing)
                remainder = content[pos:]
                if remainder.startswith("\n"):
                    remainder = remainder[1:]
                content = f"{fence}\n{remainder}"
                continue

        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


def _escape_telegram_html(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tool_hint_to_telegram_blockquote(text: str) -> str:
    """Render tool hints as an expandable blockquote (collapsed by default)."""
    return f"<blockquote expandable>{_escape_telegram_html(text)}</blockquote>" if text else ""


def _strip_md(s: str) -> str:
    """Strip markdown inline formatting from text."""
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'__(.+?)__', r'\1', s)
    s = re.sub(r'~~(.+?)~~', r'\1', s)
    s = re.sub(r'`([^`]+)`', r'\1', s)
    return s.strip()


def _strip_md_block(text: str) -> str:
    """Strip block-level and inline markdown for readable plain-text preview.

    Used during streaming mid-edits so users see clean text instead of raw
    markdown syntax while the response is still being generated.
    """
    # Code blocks -> just the code
    text = re.sub(r'```(?:[^\n]*\n)?([\s\S]*?)```', r'\1', text)
    # Headers -> plain text
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r'^>\s*(.*)$', r'\1', text, flags=re.MULTILINE)
    # Bold / italic / strikethrough
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Bullet lists
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)
    # Numbered lists (normalize spacing)
    text = re.sub(r'^(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)
    return text


def _render_table_box(table_lines: list[str]) -> str:
    """Convert markdown pipe-table to compact aligned text for <pre> display."""

    def dw(s: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)

    rows: list[list[str]] = []
    has_sep = False
    for line in table_lines:
        cells = [_strip_md(c) for c in line.strip().strip('|').split('|')]
        if all(re.match(r'^:?-+:?$', c) for c in cells if c):
            has_sep = True
            continue
        rows.append(cells)
    if not rows or not has_sep:
        return '\n'.join(table_lines)

    ncols = max(len(r) for r in rows)
    for r in rows:
        r.extend([''] * (ncols - len(r)))
    widths = [max(dw(r[c]) for r in rows) for c in range(ncols)]

    def dr(cells: list[str]) -> str:
        return '  '.join(f'{c}{" " * (w - dw(c))}' for c, w in zip(cells, widths))

    out = [dr(rows[0])]
    out.append('  '.join('─' * w for w in widths))
    for row in rows[1:]:
        out.append(dr(row))
    return '\n'.join(out)


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""

    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []
    def save_code_block(m: re.Match[str]) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r'```(?:[^\n]*\n)?([\s\S]*?)```', save_code_block, text)

    # 1.5. Convert markdown tables to box-drawing (reuse code_block placeholders)
    lines = text.split('\n')
    rebuilt: list[str] = []
    li = 0
    while li < len(lines):
        if re.match(r'^\s*\|.+\|', lines[li]):
            tbl: list[str] = []
            while li < len(lines) and re.match(r'^\s*\|.+\|', lines[li]):
                tbl.append(lines[li])
                li += 1
            box = _render_table_box(tbl)
            if box != '\n'.join(tbl):
                code_blocks.append(box)
                rebuilt.append(f"\x00CB{len(code_blocks) - 1}\x00")
            else:
                rebuilt.extend(tbl)
        else:
            rebuilt.append(lines[li])
            li += 1
    text = '\n'.join(rebuilt)

    # 2. Extract and protect inline code
    inline_codes: list[str] = []
    def save_inline_code(m: re.Match[str]) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r'`([^`]+)`', save_inline_code, text)

    # 3. Headers # Title -> <b>Title</b> (preserve visual hierarchy)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'⟪B⟫\1⟪/B⟫', text, flags=re.MULTILINE)

    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r'^>\s*(.*)$', r'\1', text, flags=re.MULTILINE)

    # 5. Escape HTML special characters
    text = _escape_telegram_html(text)

    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # 7. Bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])', r'<i>\1</i>', text)

    # 9. Strikethrough ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)

    # 10. Bullet lists - item -> • item
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)

    # 10.5. Numbered lists  1. item -> 1. item (keep number, normalize indent)
    text = re.sub(r'^(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)

    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = _escape_telegram_html(code)
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")

    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = _escape_telegram_html(code)
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")

    # 13. Restore header bold markers (inserted in step 3, after HTML escaping)
    text = text.replace('⟪B⟫', '<b>').replace('⟪/B⟫', '</b>')

    return text


def _split_telegram_markdown_html_chunks(
    content: str, max_html_len: int,
) -> list[tuple[str, str]]:
    """Return raw Markdown and rendered HTML chunk pairs within Telegram's limit."""
    chunks: list[tuple[str, str]] = []
    pending = _split_telegram_markdown(content, TELEGRAM_MAX_MESSAGE_LEN)
    while pending:
        chunk = pending.pop(0)
        html = _markdown_to_telegram_html(chunk)
        if len(html) <= max_html_len:
            chunks.append((chunk, html))
            continue

        # Markdown can expand when rendered as HTML (tags/entities). Re-split
        # the raw markdown with a smaller budget instead of slicing HTML tags.
        next_limit = max(1, int(len(chunk) * max_html_len / len(html)) - 8)
        next_limit = min(next_limit, len(chunk) - 1)
        if next_limit <= 0:
            raise ValueError("A rendered Telegram HTML token exceeds the message limit")
        parts = _split_telegram_markdown(chunk, next_limit)
        if len(parts) == 1 and parts[0] == chunk:
            raise ValueError("Unable to split Telegram Markdown within the HTML limit")
        pending = parts + pending
    return chunks


def _split_telegram_markdown_html(content: str, max_html_len: int) -> list[str]:
    """Split raw Telegram Markdown and return HTML chunks within Telegram's limit."""
    return [html for _, html in _split_telegram_markdown_html_chunks(content, max_html_len)]


_SEND_MAX_RETRIES = 3
_SEND_RETRY_BASE_DELAY = 0.5  # seconds, doubled each retry
_STREAM_EDIT_INTERVAL_DEFAULT = 0.6  # min seconds between edit_message_text calls


@dataclass
class _StreamBuf:
    """Per-chat streaming accumulator for progressive message editing."""
    text: str = ""
    message_id: int | None = None
    last_edit: float = 0.0
    stream_id: str | None = None
    accepted_delivery_id: str | None = None
    pending_chunks: list[tuple[str, str]] | None = None
    pending_cursor: int = 0
    pending_thread_kwargs: dict[str, int] | None = None


@dataclass
class _QueuedTelegramUpdate:
    """Telegram update staged for per-session ordered processing."""

    kind: Literal["command", "message"]
    update: Update
    context: ContextTypes.DEFAULT_TYPE
    sort_key: tuple[int, int]


class TelegramConfig(Base):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""
    mode: Literal["polling", "webhook"] = "polling"
    allow_from: list[str] = Field(default_factory=list)
    allow_chats: list[str] = Field(default_factory=list)
    fast_commands: dict[str, list[str]] = Field(default_factory=dict)
    fast_command_workdir: str = ""
    fast_command_timeout: int = 120
    proxy: str | None = None
    reply_to_message: bool = False
    react_emoji: str = "👀"
    group_policy: Literal["open", "mention"] = "mention"
    connection_pool_size: int = 32
    pool_timeout: float = 5.0
    streaming: bool = True
    # Enable inline keyboard buttons in Telegram messages.
    inline_keyboards: bool = False
    # Opt in to Bot API 10.1 sendRichMessage for richer markdown rendering.
    rich_messages: bool = False
    stream_edit_interval: float = Field(default=_STREAM_EDIT_INTERVAL_DEFAULT, ge=0.1)
    webhook_url: str = ""
    webhook_listen_host: str = "127.0.0.1"
    webhook_listen_port: int = Field(default=8081, ge=1, le=65535)
    webhook_path: str = "/telegram"
    webhook_secret_token: str = ""
    webhook_max_connections: int = Field(default=4, ge=1, le=100)

    @field_validator("webhook_path")
    @classmethod
    def webhook_path_must_start_with_slash(cls, value: str) -> str:
        value = value.strip() or "/telegram"
        if not value.startswith("/"):
            raise ValueError('webhook_path must start with "/"')
        return value

    @model_validator(mode="after")
    def validate_webhook_config(self) -> "TelegramConfig":
        if self.mode != "webhook":
            return self

        url = self.webhook_url.strip()
        if not url:
            raise ValueError("webhook_url is required when Telegram mode is webhook")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("webhook_url must be a public HTTPS URL")
        secret = self.webhook_secret_token.strip()
        if not secret:
            raise ValueError("webhook_secret_token is required when Telegram mode is webhook")
        if len(secret) > 256 or re.match(r"^[A-Za-z0-9_-]+$", secret) is None:
            raise ValueError(
                "webhook_secret_token must be 1-256 characters using only A-Z, a-z, 0-9, _ and -"
            )
        return self


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling or webhook mode.

    Long polling is the default. Webhook mode requires a public HTTPS URL and a
    Telegram secret token.
    """

    name = "telegram"
    display_name = "Telegram"

    # Commands registered with Telegram's command menu
    BOT_COMMANDS: list[BotCommand] = [
        BotCommand("start", "Start the bot"),
        BotCommand("new", "Start a new conversation"),
        BotCommand("stop", "Stop the current task"),
        BotCommand("restart", "Restart the bot"),
        BotCommand("status", "Show bot status"),
        BotCommand("quota", "Show Codex OAuth quota"),
        BotCommand("history", "Show recent conversation messages"),
        BotCommand("goal", "Start a sustained objective (long-running task)"),
        BotCommand("trigger", "Create a named local trigger"),
        BotCommand("pairing", "Manage DM pairing (approve/deny/list)"),
        BotCommand("model", "Switch runtime model preset"),
        BotCommand("skill", "List enabled skills"),
        BotCommand("dream", "Run Dream memory consolidation now"),
        BotCommand("dream_log", "Show the latest Dream memory change"),
        BotCommand("dream_restore", "Restore Dream memory to an earlier version"),
        BotCommand("dream_prompt", "Tell Dream how to organize memory"),
        BotCommand("help", "Show available commands"),
    ]

    # Regex for slash commands routed to AgentLoop via ``_forward_command``.
    # Hyphenated ``dream-*`` commands stay on a separate handler (below).
    TELEGRAM_BUS_SLASH_COMMAND_RE = re.compile(
        r"^/(?:new|stop|restart|status|quota|dream|history|goal|trigger|pairing|model|skill)(?:@\w+)?(?:\s+.*)?$"
    )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return TelegramConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = TelegramConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self._app: TelegramApplication | None = None
        self._stream_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._typing_tasks: dict[str, asyncio.Task[None]] = {}  # chat_id -> typing loop task
        self._media_group_buffers: dict[str, dict[str, Any]] = {}
        self._media_group_tasks: dict[str, asyncio.Task[None]] = {}
        self._message_threads: dict[tuple[str, int], int] = {}
        self._bot_user_id: int | None = None
        self._bot_username: str | None = None
        self._stream_bufs: dict[str, _StreamBuf] = {}  # chat_id -> streaming state
        self._inbound_buffers: dict[str, list[_QueuedTelegramUpdate]] = {}
        self._inbound_workers: dict[str, asyncio.Task[None]] = {}
        self._rich_send_disabled: bool = False  # Latch off if Bot API < 10.1
        self._last_poll_ok: float = 0.0  # monotonic time of last getUpdates round trip
        self._app_ready = asyncio.Event()  # cleared while the app is being rebuilt
        self._teardown_lock = asyncio.Lock()
        # Receipt-bearing cron bundles use a separate per-target lock for their
        # complete provider attempt. Ordinary sends intentionally remain outside
        # this lock and can therefore still interleave with this dormant path.
        self._receipt_target_locks: dict[tuple[int, int | None] | int, asyncio.Lock] = {}
        self._busy_inbox: TelegramBusyInbox | None = None
        self._authenticated_startup: Callable[[TelegramChannel], Awaitable[None]] | None = None

    def _require_app(self) -> TelegramApplication:
        if self._app is None:
            raise RuntimeError("Telegram application is not started")
        return self._app

    @property
    def authenticated_account_id(self) -> str | None:
        return str(self._bot_user_id) if type(self._bot_user_id) is int and self._bot_user_id > 0 else None

    def install_authenticated_startup(self, callback: Callable[[TelegramChannel], Awaitable[None]]) -> None:
        """Install a local owner before receiving any provider updates."""
        if self._running or self._authenticated_startup is not None:
            raise RuntimeError("Telegram startup owner must be installed before channel start")
        self._authenticated_startup = callback

    def remove_authenticated_startup(self, callback: Callable[[TelegramChannel], Awaitable[None]]) -> None:
        if self._authenticated_startup == callback:
            self._authenticated_startup = None

    def register_busy_inbox(self, edge: TelegramBusyInbox) -> None:
        """Bind one AgentLoop-owned edge after this bot's identity is known."""
        if self._busy_inbox is not None:
            raise RuntimeError("Telegram channel already has a registered busy inbox")
        if (not edge.matches_channel(self) or self._bot_user_id is None
                or str(self._bot_user_id) != edge.account_id):
            raise ValueError("Telegram busy inbox must match this authenticated bot account")
        self._busy_inbox = edge
        edge.wake_notices()

    def unregister_busy_inbox(self, edge: TelegramBusyInbox) -> None:
        if self._busy_inbox is edge:
            self._busy_inbox = None

    def is_chat_allowed(self, chat_id: str | int, chat_type: str) -> bool:
        """Group restrictions supplement sender authorization; private chats pass."""
        allowed = self.config.allow_chats
        return chat_type == "private" or not allowed or "*" in allowed or str(chat_id) in allowed

    def is_allowed(self, sender_id: str) -> bool:
        """Preserve Telegram's legacy id|username allowlist matching."""
        allow_list = self.config.allow_from
        if not allow_list:
            return False
        if "*" in allow_list or str(sender_id) in allow_list:
            return True

        sender_str = str(sender_id)
        if sender_str.count("|") != 1:
            return False

        sid, username = sender_str.split("|", 1)
        if not sid.isdigit() or not username:
            return False

        return sid in allow_list or username in allow_list

    def _is_update_allowed(self, message: Message, user: User) -> bool:
        return (self.is_allowed(self._sender_id(user))
                and self.is_chat_allowed(message.chat_id, message.chat.type))

    def busy_account_matches(self, account_id: str) -> bool:
        """Expose the current bot identity check to the registered H1 edge."""
        return self._bot_user_id is not None and str(self._bot_user_id) == account_id

    def owns_busy_inbox(self, edge: object) -> bool:
        """Tell the H1 edge whether it is still the channel's active owner."""
        return self._busy_inbox is edge

    @staticmethod
    def busy_sender_id(user: object) -> str | None:
        """Parse sender identity from an untrusted H1 transport boundary."""
        user_id = getattr(user, "id", None)
        username = getattr(user, "username", None)
        if (type(user_id) is not int or user_id <= 0
                or username is not None and not isinstance(username, str)):
            return None
        return f"{user_id}|{username}" if username else str(user_id)

    def busy_update_allowed(self, message: object, user: object) -> bool:
        """Apply the channel's current sender and chat policy to an H1 update."""
        sender_id = self.busy_sender_id(user)
        chat_id = getattr(message, "chat_id", None)
        chat_type = getattr(getattr(message, "chat", None), "type", None)
        if sender_id is None or type(chat_id) is not int or not isinstance(chat_type, str):
            return False
        return self.is_allowed(sender_id) and self.is_chat_allowed(chat_id, chat_type)

    @staticmethod
    def _normalize_telegram_command(content: str) -> str:
        """Map Telegram-safe command aliases back to canonical nanobot commands."""
        if not content.startswith("/"):
            return content
        if content == "/dream_log" or content.startswith("/dream_log "):
            return content.replace("/dream_log", "/dream-log", 1)
        if content == "/dream_restore" or content.startswith("/dream_restore "):
            return content.replace("/dream_restore", "/dream-restore", 1)
        if content == "/dream_prompt" or content.startswith("/dream_prompt "):
            return content.replace("/dream_prompt", "/dream-prompt", 1)
        return content

    async def start(self) -> None:
        """Start the Telegram bot, rebuilding the app whenever polling stalls."""
        if not self.config.token:
            self.logger.error("bot token not configured")
            return

        redirect_lib_logging("telegram")
        redirect_lib_logging("httpx", level="WARNING")

        self._running = True
        backoff = RESTART_BACKOFF_INITIAL_SECONDS
        while self._running:
            try:
                await self._start_app()
            except InvalidToken:
                # A config error, not a blip: fail the channel. The scrubbed
                # re-raise keeps PTB's token-bearing message out of the log.
                await self._teardown_app()
                self._running = False
                self.logger.error("bot token rejected by Telegram")
                raise RuntimeError("Telegram bot token was rejected by the server") from None
            except Exception as e:
                await self._teardown_app()
                if not self._running:
                    break
                if not self._is_transient_startup_error(e):
                    # Never heals on its own: fail instead of retrying forever
                    # while ChannelManager keeps reporting the channel running.
                    self._running = False
                    self.logger.error("startup failed: {}", self._format_telegram_error(e))
                    raise
                self.logger.error(
                    "startup failed: {}; retrying in {:.0f}s",
                    self._format_telegram_error(e),
                    backoff,
                )
                await self._idle(backoff)
                backoff = min(backoff * 2, RESTART_BACKOFF_MAX_SECONDS)
                continue

            backoff = RESTART_BACKOFF_INITIAL_SECONDS
            if not self._running:
                # stop() ran while _start_app() was mid-flight and tore down the
                # previous (possibly None) app; this one would leak otherwise.
                await self._teardown_app()
                break
            stalled = await self._watch_polling()
            if not stalled or not self._running:
                break
            self.logger.warning(
                "polling stalled: no getUpdates round trip for {:.0f}s; "
                "rebuilding connection pools and restarting",
                time.monotonic() - self._last_poll_ok,
            )
            await self._teardown_app()

    async def _start_app(self) -> None:
        """Build, initialize and start the Telegram application."""
        proxy = self.config.proxy or None

        # Separate pools so long-polling (getUpdates) never starves outbound sends.
        api_request = HTTPXRequest(
            connection_pool_size=self.config.connection_pool_size,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )
        poll_request = HTTPXRequest(
            connection_pool_size=4,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )
        builder = (
            Application.builder()
            .token(self.config.token)
            .request(api_request)
            .get_updates_request(_LivenessTrackedRequest(poll_request, self._note_poll_ok))
        )
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)

        # Add command handlers (using Regex to support @username suffixes before bot initialization)
        self._app.add_handler(MessageHandler(filters.Regex(r"^/start(?:@\w+)?$"), self._on_start))
        self._app.add_handler(
            MessageHandler(
                filters.Regex(TelegramChannel.TELEGRAM_BUS_SLASH_COMMAND_RE),
                self._forward_command,
            )
        )
        self._app.add_handler(
            MessageHandler(
                filters.Regex(
                    r"^/(dream-log|dream_log|dream-restore|dream_restore|dream-prompt|dream_prompt)(?:@\w+)?(?:\s+.*)?$"
                ),
                self._forward_command,
            )
        )
        self._app.add_handler(MessageHandler(filters.Regex(r"^/help(?:@\w+)?$"), self._on_help))

        if self.config.fast_commands:
            names = [re.escape(name.lstrip("/")) for name in self.config.fast_commands]
            self._app.add_handler(MessageHandler(
                filters.Regex(re.compile(r"^/(?:" + "|".join(names) + r")(?:@\w+)?(?:\s.*)?$", re.IGNORECASE)),
                self._forward_command,
            ))

        # Add message handler for text, photos, video, voice, documents, and locations
        self._app.add_handler(
            MessageHandler(
                (filters.TEXT | filters.PHOTO | filters.VIDEO | filters.VIDEO_NOTE
                 | filters.ANIMATION | filters.VOICE | filters.AUDIO
                 | filters.Document.ALL | filters.LOCATION)
                & ~filters.COMMAND,
                self._on_message
            )
        )

        # Conditionally register inline keyboard callback handler
        if self.config.inline_keyboards:
            self._app.add_handler(CallbackQueryHandler(self._on_callback_query))
            allowed_updates = ["message", "callback_query"]
            self.logger.debug("inline keyboards enabled")
        else:
            allowed_updates = ["message"]

        if self.config.mode == "webhook":
            self.logger.info("Starting bot (webhook mode)...")
        else:
            self.logger.info("Starting bot (polling mode)...")

        # Initialize and start receiving updates
        await self._app.initialize()
        await self._app.start()

        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        self._bot_user_id = getattr(bot_info, "id", None)
        self._bot_username = getattr(bot_info, "username", None)
        self.logger.info("bot @{} connected", bot_info.username)
        if self._authenticated_startup is not None:
            await self._authenticated_startup(self)

        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            self.logger.debug("bot commands registered")
        except Exception as e:
            self.logger.warning("Failed to register bot commands: {}", e)

        if self.config.mode == "webhook":
            # ``url_path`` is the local HTTP route. ``webhook_url`` is the
            # public HTTPS URL Telegram calls; reverse proxies may rewrite it.
            await cast(Any, self._app.updater).start_webhook(
                listen=self.config.webhook_listen_host,
                port=self.config.webhook_listen_port,
                url_path=self.config.webhook_path.lstrip("/"),
                webhook_url=self.config.webhook_url.strip(),
                allowed_updates=allowed_updates,
                drop_pending_updates=False,
                secret_token=self.config.webhook_secret_token.strip(),
                max_connections=self.config.webhook_max_connections,
            )
        else:
            self._last_poll_ok = time.monotonic()
            await cast(Any, self._app.updater).start_polling(
                allowed_updates=allowed_updates,
                drop_pending_updates=False,  # Process pending messages on startup
                error_callback=self._on_polling_error,
            )

        self._app_ready.set()

    @staticmethod
    def _is_transient_startup_error(exc: Exception) -> bool:
        """Report whether a startup failure is worth retrying.

        HTTPXRequest wraps every httpx failure into NetworkError/TimedOut, so
        anything else is terminal: a bad proxy raises ValueError, an already
        bound webhook port raises OSError.
        """
        return isinstance(exc, NetworkError | TimedOut | asyncio.TimeoutError)

    async def _wait_for_app(self) -> TelegramApplication | None:
        """Return the live app, briefly waiting out an in-flight rebuild.

        Returning quietly while ``start()`` rebuilds would let the manager count
        the message as delivered, so raise once the wait runs out. None means the
        channel is stopped: nothing left to deliver.
        """
        if self._app_ready.is_set() and self._app is not None:
            return self._app
        if not self._running:
            return None
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._app_ready.wait(), APP_RESTART_SEND_WAIT_SECONDS)
        if not self._app_ready.is_set() or self._app is None:
            raise RuntimeError("Telegram application is restarting; message not delivered")
        return self._app

    def _note_poll_ok(self) -> None:
        # HTTP error statuses count too: the watchdog detects transport stalls,
        # not logical failures.
        self._last_poll_ok = time.monotonic()

    async def _watch_polling(self) -> bool:
        """Idle until stop(); in polling mode, return True when getUpdates goes stale."""
        watch = self.config.mode != "webhook"
        while self._running:
            await asyncio.sleep(POLL_WATCH_INTERVAL)
            if watch and time.monotonic() - self._last_poll_ok > POLL_STALE_SECONDS:
                return True
        return False

    async def _idle(self, seconds: float) -> None:
        """Sleep in short steps so stop() stays responsive."""
        deadline = time.monotonic() + seconds
        while self._running and time.monotonic() < deadline:
            await asyncio.sleep(POLL_WATCH_INTERVAL)

    async def _teardown_app(self) -> None:
        """Shut down the application, tolerating partially started state."""
        async with self._teardown_lock:
            app, self._app = self._app, None
            self._app_ready.clear()
            if not app:
                return
            for step in (cast(Any, app.updater).stop, app.stop, app.shutdown):
                try:
                    await step()
                except Exception as e:
                    self.logger.debug("teardown step failed: {}", e)
            # Application.shutdown() skips the HTTPX pools unless initialize()
            # finished, so a failed startup leaks one per retry. This is idempotent.
            try:
                await app.bot.shutdown()
            except Exception as e:
                self.logger.debug("bot shutdown failed: {}", e)

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False

        # Cancel all typing indicators
        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)

        for task in self._media_group_tasks.values():
            task.cancel()
        self._media_group_tasks.clear()
        self._media_group_buffers.clear()

        for task in self._inbound_workers.values():
            task.cancel()
        self._inbound_workers.clear()
        self._inbound_buffers.clear()

        if self._app:
            self.logger.info("Stopping bot...")
        # Join an in-flight supervisor teardown before ChannelManager cancels
        # start(), otherwise cancellation can strand the old HTTPX pools.
        await self._teardown_app()

    @staticmethod
    def _get_media_type(path: str) -> Literal["photo", "video", "voice", "audio", "document"]:
        """Guess media type from file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("jpg", "jpeg", "png", "gif", "webp"):
            return "photo"
        if ext in ("mp4", "mov", "avi", "mkv", "webm", "3gp"):
            return "video"
        if ext == "ogg":
            return "voice"
        if ext in ("mp3", "m4a", "wav", "aac"):
            return "audio"
        return "document"

    @staticmethod
    def _is_remote_media_url(path: str) -> bool:
        return path.startswith(("http://", "https://"))

    def busy_is_remote_media_url(self, path: str) -> bool:
        return self._is_remote_media_url(path)

    def busy_media_type(
        self, path: str,
    ) -> Literal["photo", "video", "voice", "audio", "document"]:
        return self._get_media_type(path)

    @staticmethod
    def _split_routed_text(content: str) -> tuple[str, ...]:
        """Return the exact Markdown chunks used by receipt-bearing sends."""
        return tuple(
            markdown
            for markdown, _html in _split_telegram_markdown_html_chunks(
                content, TELEGRAM_HTML_MAX_LEN,
            )
        )

    def busy_split_routed_text(self, content: str) -> tuple[str, ...]:
        return self._split_routed_text(content)

    @staticmethod
    def _is_rich_capability_error(exc: Exception) -> bool:
        """True when the error indicates sendRichMessage is unavailable."""
        err = str(exc).lower()
        return (
            "method not found" in err
            or "unknown method" in err
            or "bad request: invalid parameter" in err
        )

    async def _try_send_rich(
        self,
        chat_id: int,
        content: str,
        reply_params: ReplyParameters | dict[str, int | bool] | None = None,
        thread_kwargs: dict[str, int] | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> bool:
        """Attempt sendRichMessage (Bot API 10.1). Returns True on success."""
        if not self._app:
            return False

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "rich_message": {
                "markdown": content,
            },
        }
        if reply_params is not None:
            # sendRichMessage uses reply_parameters (object), not reply_to_message_id.
            if hasattr(reply_params, "message_id"):
                payload["reply_parameters"] = {
                    "message_id": cast(ReplyParameters, reply_params).message_id,
                    "allow_sending_without_reply": True,
                }
            else:
                payload["reply_parameters"] = reply_params
        if thread_kwargs:
            payload.update({
                k: v
                for k, v in thread_kwargs.items()
                if v is not None  # pyright: ignore[reportUnnecessaryComparison]
            })
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        try:
            await self._call_with_retry(
                self._app.bot.do_api_request,
                "sendRichMessage",
                api_kwargs=payload,
            )
            return True
        except BadRequest as exc:
            if self._is_rich_capability_error(exc):
                self.logger.debug("sendRichMessage not available, disabling")
                self._rich_send_disabled = True
            else:
                self.logger.debug("sendRichMessage rejected: {}", exc)
            return False
        except Exception as exc:
            err_str = str(exc).lower()
            is_timeout = "timed out" in err_str or isinstance(exc, TimedOut)
            if is_timeout:
                self.logger.debug("sendRichMessage timeout, falling back to legacy path")
                return False
            self.logger.debug("sendRichMessage failed: {}", exc)
            return False

    async def _try_edit_rich(self, chat_id: int, message_id: int, content: str) -> bool:
        """Upgrade an existing message to rich in place via editMessageText (Bot API 10.1).

        Editing in place keeps the message identity, so the streaming preview is
        upgraded without the delete-and-resend pattern that caused flickering and
        dropped line breaks (issue #4470).

        Returns True when the rich edit is in place (including the ambiguous
        "message is not modified" retry outcome after a response timeout).
        Returns False only when the legacy HTML path should take over:
        capability errors (server older than Bot API 10.1, which also trip the
        rich latch) and content-shaped BadRequest rejections. Transport,
        rate-limit, and unexpected errors propagate so the final-edit retry
        contract is preserved — ChannelManager retries the buffered send
        instead of an immediate legacy edit doubling connection demand.
        """
        if not self._app:
            return False

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "rich_message": {
                "markdown": content,
            },
        }
        try:
            await self._call_with_retry(
                self._app.bot.do_api_request,
                "editMessageText",
                api_kwargs=payload,
            )
            return True
        except BadRequest as exc:
            if self._is_not_modified_error(exc):
                # Ambiguous success: the rich edit was applied server-side but
                # its response timed out, so the retry hit "message is not
                # modified". Treat it as done rather than letting the legacy
                # edit overwrite the already-successful rich result.
                self.logger.debug("Rich stream edit already applied for {}", chat_id)
                return True
            # Before Bot API 10.1, editMessageText ignores rich_message and
            # reports the absent text argument instead.
            pre_rich_edit_server = (
                bool(content)
                and str(exc).strip().lower() == "message text is empty"
            )
            if self._is_rich_capability_error(exc) or pre_rich_edit_server:
                self.logger.debug("editMessageText rich_message not available, disabling")
                self._rich_send_disabled = True
                return False
            # Content-shaped rejections (invalid markdown, unsupported media in
            # the rich payload, …) fall back to the legacy HTML edit.
            self.logger.debug("editMessageText rich_message rejected: {}", exc)
            return False
        except Exception:
            # Transport, rate-limit, and unexpected errors propagate so the
            # final-edit retry contract stays intact: ChannelManager retries
            # the buffered send instead of this handler doubling connection
            # demand with an immediate legacy edit.
            raise

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram."""
        app = await self._wait_for_app()
        if app is None:
            self.logger.warning("bot not running")
            return

        progress_event = msg.event if isinstance(msg.event, ProgressEvent) else None

        # Only stop typing indicator and remove reaction for final responses
        if progress_event is None:
            self._stop_typing(msg.chat_id)
            if reply_to_message_id := msg.metadata.get("message_id"):
                with suppress(ValueError):
                    await self._remove_reaction(msg.chat_id, int(reply_to_message_id))

        try:
            chat_id = int(msg.chat_id)
        except ValueError:
            self.logger.exception("Invalid chat_id: {}", msg.chat_id)
            return
        reply_to_message_id = msg.metadata.get("message_id")
        message_thread_id = msg.metadata.get("message_thread_id")
        if message_thread_id is None and reply_to_message_id is not None:
            message_thread_id = self._message_threads.get((msg.chat_id, reply_to_message_id))
        thread_kwargs: dict[str, int] = {}
        if message_thread_id is not None:
            thread_kwargs["message_thread_id"] = message_thread_id

        reply_params = None
        if self.config.reply_to_message:
            if reply_to_message_id:
                reply_params = ReplyParameters(
                    message_id=reply_to_message_id,
                    allow_sending_without_reply=True
                )

        caption = None
        if (msg.media and msg.content and msg.content != "[empty message]"
                and len(msg.content.encode("utf-16-le")) // 2 <= 1024
                and not getattr(msg, "buttons", None) and not progress_event):
            caption = _markdown_to_telegram_html(msg.content)
        if msg.media and not msg.content:
            caption = "Attached file."
        caption_sent = False
        # Send media files
        for media_path in (msg.media or []):
            try:
                media_type = self._get_media_type(media_path)
                sender = {
                    "photo": app.bot.send_photo,
                    "video": app.bot.send_video,
                    "voice": app.bot.send_voice,
                    "audio": app.bot.send_audio,
                }.get(media_type, app.bot.send_document)
                param = {
                    "photo": "photo",
                    "video": "video",
                    "voice": "voice",
                    "audio": "audio",
                }.get(media_type, "document")
                extra: dict[str, Any] = {}
                if caption is not None and not caption_sent:
                    extra.update(caption=caption, parse_mode="HTML")
                if media_type == "video":
                    extra["supports_streaming"] = True

                # Telegram Bot API accepts HTTP(S) URLs directly for media params.
                if self._is_remote_media_url(media_path):
                    ok, error = validate_url_target(media_path)
                    if not ok:
                        raise ValueError(f"unsafe media URL: {error}")
                    await self._call_with_retry(
                        sender,
                        chat_id=chat_id,
                        **{param: media_path},
                        reply_parameters=reply_params,
                        **thread_kwargs,
                        **extra,
                    )
                    caption_sent = caption_sent or "caption" in extra
                    continue

                media_bytes = Path(media_path).read_bytes()
                filename = Path(media_path).name
                send_kwargs = {param: media_bytes, "filename": filename}
                await self._call_with_retry(
                    sender,
                    chat_id=chat_id,
                    reply_parameters=reply_params,
                    **thread_kwargs,
                    **extra,
                    **send_kwargs,
                )
                caption_sent = caption_sent or "caption" in extra
            except Exception:
                filename = media_path.rsplit("/", 1)[-1]
                self.logger.exception("Failed to send media {}", media_path)
                await app.bot.send_message(
                    chat_id=chat_id,
                    text=f"[Failed to send: {filename}]",
                    reply_parameters=reply_params,
                    **thread_kwargs,
                )

        # Send text content
        if msg.content and msg.content != "[empty message]" and not caption_sent:
            render_as_blockquote = bool(progress_event and progress_event.tool_hint)
            buttons = cast(list[list[str]], getattr(msg, "buttons", None) or [])
            reply_markup = self._build_keyboard(buttons) if buttons else None
            text = msg.content
            # Fallback: no native keyboard → splice labels into the message so the choices survive.
            if buttons and reply_markup is None:
                text = f"{text}\n\n{self._buttons_as_text(buttons)}"

            # Bot API 10.1 rich fast-path: send raw markdown via sendRichMessage.
            # All non-blockquote content tries rich first; _rich_send_disabled
            # latches off permanently if the server doesn't support it.
            if (
                not render_as_blockquote
                and self.config.rich_messages
                and not getattr(self, "_rich_send_disabled", False)
            ):
                rich_ok = await self._try_send_rich(
                    chat_id, text, reply_params, thread_kwargs, reply_markup,
                )
                if rich_ok:
                    return

            chunks = _split_telegram_markdown(text, TELEGRAM_MAX_MESSAGE_LEN)
            for i, chunk in enumerate(chunks):
                is_last = (i == len(chunks) - 1)
                await self._send_text(
                    chat_id, chunk, reply_params, thread_kwargs,
                    render_as_blockquote=render_as_blockquote,
                    reply_markup=reply_markup if is_last else None,
                )

    @staticmethod
    def _is_html_parse_rejection(exc: BadRequest) -> bool:
        """Recognize only Telegram's definite, pre-creation entity parse errors."""

        message = str(exc).strip().lower()
        return message.startswith("can't parse entities:") or message.startswith(
            "can't parse entity:"
        )

    @staticmethod
    def _receipt_part(
        response: object,
        *,
        index: int,
        message_index: int,
        part_index: int,
        expected_chat_id: int,
    ) -> ChannelTextPartReceipt:
        """Validate the dynamic Telegram acknowledgement before retaining evidence."""

        message_id = getattr(response, "message_id", None)
        chat = getattr(response, "chat", None)
        provider_chat_id = getattr(chat, "id", None)
        if type(message_id) is not int or message_id <= 0:
            raise ValueError("Telegram acknowledgement has an invalid message id")
        if type(provider_chat_id) is not int or provider_chat_id != expected_chat_id:
            raise ValueError("Telegram acknowledgement has an invalid provider chat id")
        timestamp_value = getattr(response, "date", None)
        provider_timestamp: str | None = None
        if timestamp_value is not None:
            if not hasattr(timestamp_value, "isoformat"):
                raise ValueError("Telegram acknowledgement has an invalid provider timestamp")
            provider_timestamp = timestamp_value.isoformat()
        return ChannelTextPartReceipt(
            index=index,
            message_index=message_index,
            part_index=part_index,
            provider_message_id=message_id,
            provider_chat_id=provider_chat_id,
            provider_timestamp=provider_timestamp,
        )

    async def send_text_with_receipt(
        self,
        target: str,
        messages: Sequence[str],
        *,
        action_id: str,
        request_digest: str,
    ) -> ChannelTextReceipt:
        return await self._send_text_bundle_with_receipt(target, messages, action_id=action_id,
                                                        request_digest=request_digest)

    @property
    def receipt_transport_ready(self) -> bool:
        return self._app is not None and self._app_ready.is_set()

    async def send_thread_text_with_receipt(
        self, target: str, messages: Sequence[str], *, message_thread_id: int,
        action_id: str, request_digest: str,
    ) -> ChannelTextReceipt:
        if type(message_thread_id) is not int or message_thread_id <= 0:
            raise ChannelDeliveryPreflightError("invalid_thread")
        return await self._send_text_bundle_with_receipt(target, messages, message_thread_id=message_thread_id,
                                                        action_id=action_id, request_digest=request_digest)

    async def _send_text_bundle_with_receipt(
        self, target: str, messages: Sequence[str], *, action_id: str, request_digest: str,
        message_thread_id: int | None = None,
    ) -> ChannelTextReceipt:
        """Send one text bundle with no hidden retries and strict acknowledgements.

        The process-local lock prevents two receipt-path bundles for the same
        target from interleaving. It cannot serialize ordinary channel sends or
        sends made by another process.
        """

        app = self._app
        if app is None or not self._app_ready.is_set():
            raise ChannelDeliveryPreflightError("telegram_app_unavailable")
        if not isinstance(cast(object, target), str) or not target or target.strip() != target:
            raise ChannelDeliveryPreflightError("invalid_target")
        try:
            chat_id = int(target)
        except ValueError as exc:
            raise ChannelDeliveryPreflightError("invalid_target") from exc
        if chat_id == 0 or str(chat_id) != target:
            raise ChannelDeliveryPreflightError("invalid_target")
        if isinstance(messages, (str, bytes)) or not isinstance(cast(object, messages), Sequence) or not messages:
            raise ChannelDeliveryPreflightError("invalid_message_bundle")

        rendered: list[tuple[int, int, str, str]] = []
        for message_index, message in enumerate(messages):
            if not isinstance(cast(object, message), str):
                raise ChannelDeliveryPreflightError("invalid_message_bundle")
            try:
                chunks = _split_telegram_markdown_html_chunks(
                    message, TELEGRAM_HTML_MAX_LEN
                )
            except ValueError as exc:
                raise ChannelDeliveryPreflightError("invalid_message_bundle") from exc
            if not chunks:
                raise ChannelDeliveryPreflightError("invalid_message_bundle")
            rendered.extend(
                (message_index, part_index, markdown, html)
                for part_index, (markdown, html) in enumerate(chunks)
            )

        # Prove the request binding and worst-case part structure are valid and
        # bounded before the first irreversible provider call.
        try:
            ChannelTextReceipt(
                channel=self.name,
                target=target,
                action_id=action_id,
                request_digest=request_digest,
                parts=tuple(
                    ChannelTextPartReceipt(
                        index=index,
                        message_index=message_index,
                        part_index=part_index,
                        provider_message_id=9_223_372_036_854_775_807 - index,
                        provider_chat_id=chat_id,
                        provider_timestamp="x" * 64,
                    )
                    for index, (message_index, part_index, _markdown, _html) in enumerate(
                        rendered
                    )
                ),
            )
        except ValueError as exc:
            raise ChannelDeliveryPreflightError("invalid_receipt_request") from exc

        lock = self._receipt_target_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._receipt_target_locks[chat_id] = lock
        async with lock:
            receipts: list[ChannelTextPartReceipt] = []
            thread_options = {"message_thread_id": message_thread_id} if message_thread_id is not None else {}
            for message_index, part_index, markdown, html in rendered:
                try:
                    try:
                        response = await app.bot.send_message(
                            chat_id=chat_id, text=html, parse_mode="HTML", **thread_options,
                        )
                    except BadRequest as exc:
                        if not self._is_html_parse_rejection(exc):
                            raise ChannelDeliveryRejectedError("provider_rejected") from exc
                        try:
                            response = await app.bot.send_message(chat_id=chat_id, text=markdown, **thread_options)
                        except BadRequest as plain_exc:
                            raise ChannelDeliveryRejectedError(
                                "provider_rejected"
                            ) from plain_exc
                    if message_thread_id is not None and response.message_thread_id != message_thread_id:
                        raise RuntimeError("telegram_thread_receipt_mismatch")
                    receipt_part = self._receipt_part(
                        response,
                        index=len(receipts),
                        message_index=message_index,
                        part_index=part_index,
                        expected_chat_id=chat_id,
                    )
                except BaseException as exc:
                    if receipts:
                        prefix = ChannelTextReceipt(
                            channel=self.name,
                            target=target,
                            action_id=action_id,
                            request_digest=request_digest,
                            parts=tuple(receipts),
                        )
                        raise PartialChannelTextDeliveryError(prefix) from exc
                    raise
                receipts.append(receipt_part)

        return ChannelTextReceipt(
            channel=self.name,
            target=target,
            action_id=action_id,
            request_digest=request_digest,
            parts=tuple(receipts),
        )

    async def send_routed_with_receipt(
        self, request: RoutedDeliveryRequest,
    ) -> RoutedDeliveryReceipt:
        """Preflight and send one immutable H1 bundle to its exact Telegram route."""
        from nanobot.channels.routed_delivery import validate_routed_receipt

        edge = self._busy_inbox
        if edge is None or not edge.authorizes_address(request.address):
            raise RoutedDeliveryError("definitely-unsent", "telegram_route_not_authorized")
        app = self._app
        if app is None or not self._app_ready.is_set():
            raise RoutedDeliveryError("definitely-unsent", "telegram_app_unavailable")
        address = request.address
        if address.channel != self.name or str(self._bot_user_id) != address.account_id:
            raise RoutedDeliveryError("definitely-unsent", "telegram_account_mismatch")
        try:
            chat_id = int(address.chat_id)
            thread_id = int(address.thread_id) if address.thread_id is not None else None
            reply_id = int(address.reply_to_message_id) if address.reply_to_message_id is not None else None
        except ValueError as exc:
            raise RoutedDeliveryError("definitely-unsent", "telegram_route_invalid") from exc
        if (str(chat_id) != address.chat_id or chat_id == 0
                or (address.thread_id is not None and str(thread_id) != address.thread_id)
                or (address.reply_to_message_id is not None and str(reply_id) != address.reply_to_message_id)):
            raise RoutedDeliveryError("definitely-unsent", "telegram_route_invalid")

        prepared_media: list[tuple[RoutedAttachment, bytes | str]] = []
        for attachment in request.bundle.attachments:
            if attachment.reference_digest is not None:
                if (not self._is_remote_media_url(attachment.source)
                        or hashlib.sha256(attachment.source.encode("utf-8")).hexdigest()
                        != attachment.reference_digest):
                    raise RoutedDeliveryError("definitely-unsent", "telegram_media_reference_changed")
                ok, _reason = validate_url_target(attachment.source)
                if not ok:
                    raise RoutedDeliveryError("definitely-unsent", "telegram_media_url_rejected")
                prepared_media.append((attachment, attachment.source))
                continue
            if attachment.content_digest is None or self._is_remote_media_url(attachment.source):
                raise RoutedDeliveryError("definitely-unsent", "telegram_media_binding_invalid")
            try:
                media_bytes = Path(attachment.source).read_bytes()
            except (OSError, ValueError) as exc:
                raise RoutedDeliveryError("definitely-unsent", "telegram_media_unavailable") from exc
            if (len(media_bytes) > 50 * 1024 * 1024
                    or hashlib.sha256(media_bytes).hexdigest() != attachment.content_digest):
                raise RoutedDeliveryError("definitely-unsent", "telegram_media_changed")
            prepared_media.append((attachment, media_bytes))

        prepared_text: list[tuple[int, str, str]] = []
        for part_index, text in enumerate(request.bundle.emitted_text_parts):
            try:
                chunks = _split_telegram_markdown_html_chunks(text, TELEGRAM_HTML_MAX_LEN)
            except ValueError as exc:
                raise RoutedDeliveryError("definitely-unsent", "telegram_text_invalid") from exc
            if len(chunks) != 1 or chunks[0][0] != text:
                raise RoutedDeliveryError("definitely-unsent", "telegram_text_empty")
            markdown, html = chunks[0]
            prepared_text.append((part_index, markdown, html))
        if not prepared_media and not prepared_text:
            raise RoutedDeliveryError("definitely-unsent", "telegram_bundle_empty")

        reply_params = (
            ReplyParameters(message_id=reply_id)
            if reply_id is not None else None
        )
        target_lock_key = (chat_id, thread_id)
        lock = self._receipt_target_locks.get(target_lock_key)
        if lock is None:
            lock = asyncio.Lock()
            self._receipt_target_locks[target_lock_key] = lock

        receipts: list[RoutedPartReceipt] = []

        def receipt_prefix() -> RoutedDeliveryReceipt | None:
            if not receipts:
                return None
            return RoutedDeliveryReceipt(
                address=address,
                action_id=request.action_id,
                bundle_digest=request.bundle_digest,
                parts=tuple(receipts),
            )

        def authorize_before_send() -> None:
            if (self._busy_inbox is not edge or self._app is not app
                    or not self._app_ready.is_set() or not edge.authorizes_address(address)):
                disposition = "uncertain" if receipts else "definitely-unsent"
                raise RoutedDeliveryError(
                    disposition, "telegram_route_revoked", receipt=receipt_prefix(),
                )

        def make_receipt(
            response: Message, *, message_index: int, part_index: int, kind: RoutedPartKind,
        ) -> RoutedPartReceipt:
            message_id = response.message_id
            response_chat = response.chat.id
            response_thread = response.message_thread_id
            response_reply = (
                response.reply_to_message.message_id
                if response.reply_to_message is not None else None
            )
            if type(message_id) is not int or message_id <= 0 or type(response_chat) is not int:
                raise ValueError("Telegram acknowledgement has incomplete message identity")
            if response_chat != chat_id or response_thread != thread_id or response_reply != reply_id:
                raise ValueError("Telegram acknowledgement has incomplete exact-route evidence")
            return RoutedPartReceipt(
                index=len(receipts),
                message_index=message_index,
                part_index=part_index,
                kind=kind,
                provider_message_id=message_id,
                provider_chat_id=response_chat,
                provider_thread_id=response_thread,
                provider_reply_to_message_id=response_reply,
            )

        try:
            await lock.acquire()
        except asyncio.CancelledError as exc:
            raise RoutedDeliveryError(
                "definitely-unsent", "telegram_send_cancelled_before_provider",
            ) from exc
        try:
            for index, (attachment, source) in enumerate(prepared_media):
                authorize_before_send()
                try:
                    filename = attachment.filename if isinstance(source, bytes) else None
                    if attachment.media_type == "photo":
                        response = await app.bot.send_photo(
                            chat_id=chat_id, photo=source, filename=filename,
                            reply_parameters=reply_params, message_thread_id=thread_id,
                        )
                    elif attachment.media_type == "video":
                        response = await app.bot.send_video(
                            chat_id=chat_id, video=source, filename=filename,
                            supports_streaming=True, reply_parameters=reply_params,
                            message_thread_id=thread_id,
                        )
                    elif attachment.media_type == "voice":
                        response = await app.bot.send_voice(
                            chat_id=chat_id, voice=source, filename=filename,
                            reply_parameters=reply_params, message_thread_id=thread_id,
                        )
                    elif attachment.media_type == "audio":
                        response = await app.bot.send_audio(
                            chat_id=chat_id, audio=source, filename=filename,
                            reply_parameters=reply_params, message_thread_id=thread_id,
                        )
                    else:
                        response = await app.bot.send_document(
                            chat_id=chat_id, document=source, filename=filename,
                            reply_parameters=reply_params, message_thread_id=thread_id,
                        )
                    receipts.append(make_receipt(
                        response, message_index=index, part_index=0, kind="media",
                    ))
                except RoutedDeliveryError:
                    raise
                except BadRequest as exc:
                    disposition = "uncertain" if receipts else "definitely-unsent"
                    raise RoutedDeliveryError(
                        disposition, "telegram_media_rejected", receipt=receipt_prefix(),
                    ) from exc
                except BaseException as exc:
                    raise RoutedDeliveryError(
                        "uncertain", "telegram_media_send_uncertain", receipt=receipt_prefix(),
                    ) from exc

            text_index = len(prepared_media)
            for part_index, markdown, html in prepared_text:
                authorize_before_send()
                try:
                    response = await app.bot.send_message(
                        chat_id=chat_id, text=html, parse_mode="HTML",
                        reply_parameters=reply_params, message_thread_id=thread_id,
                    )
                except BadRequest as exc:
                    if not self._is_html_parse_rejection(exc):
                        disposition = "uncertain" if receipts else "definitely-unsent"
                        raise RoutedDeliveryError(
                            disposition, "telegram_text_rejected", receipt=receipt_prefix(),
                        ) from exc
                    authorize_before_send()
                    try:
                        response = await app.bot.send_message(
                            chat_id=chat_id, text=markdown,
                            reply_parameters=reply_params, message_thread_id=thread_id,
                        )
                    except BadRequest as plain_exc:
                        disposition = "uncertain" if receipts else "definitely-unsent"
                        raise RoutedDeliveryError(
                            disposition, "telegram_text_rejected", receipt=receipt_prefix(),
                        ) from plain_exc
                    except BaseException as plain_exc:
                        raise RoutedDeliveryError(
                            "uncertain", "telegram_text_send_uncertain", receipt=receipt_prefix(),
                        ) from plain_exc
                except RoutedDeliveryError:
                    raise
                except BaseException as exc:
                    raise RoutedDeliveryError(
                        "uncertain", "telegram_text_send_uncertain", receipt=receipt_prefix(),
                    ) from exc
                try:
                    receipts.append(make_receipt(
                        response, message_index=text_index,
                        part_index=part_index, kind="text",
                    ))
                except ValueError as exc:
                    raise RoutedDeliveryError(
                        "uncertain", "telegram_receipt_invalid", receipt=receipt_prefix(),
                    ) from exc
        finally:
            lock.release()

        receipt = RoutedDeliveryReceipt(
            address=address,
            action_id=request.action_id,
            bundle_digest=request.bundle_digest,
            parts=tuple(receipts),
        )
        try:
            validate_routed_receipt(request, receipt)
        except ValueError as exc:
            raise RoutedDeliveryError("uncertain", "telegram_receipt_invalid", receipt=receipt) from exc
        return receipt

    async def _call_with_retry(
        self,
        fn: Callable[..., Awaitable[_T]],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Call an async Telegram API function with retry on pool/network timeout and RetryAfter."""
        from telegram.error import RetryAfter

        for attempt in range(1, _SEND_MAX_RETRIES + 1):
            try:
                return await fn(*args, **kwargs)
            except TimedOut:
                if attempt == _SEND_MAX_RETRIES:
                    raise
                delay = _SEND_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.logger.warning(
                    "timeout (attempt {}/{}), retrying in {:.1f}s",
                    attempt, _SEND_MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            except RetryAfter as e:
                if attempt == _SEND_MAX_RETRIES:
                    raise
                retry_after = e.retry_after
                delay = (
                    retry_after.total_seconds()
                    if isinstance(retry_after, timedelta)
                    else float(retry_after)
                )
                self.logger.warning(
                    "Flood Control (attempt {}/{}), retrying in {:.1f}s",
                    attempt, _SEND_MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Telegram retry loop exited unexpectedly")

    async def _send_text(
        self,
        chat_id: int,
        text: str,
        reply_params: ReplyParameters | None = None,
        thread_kwargs: dict[str, int] | None = None,
        render_as_blockquote: bool = False,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        """Send a plain text message with HTML fallback."""
        app = self._require_app()
        try:
            html = _tool_hint_to_telegram_blockquote(text) if render_as_blockquote else _markdown_to_telegram_html(text)
            await self._call_with_retry(
                app.bot.send_message,
                chat_id=chat_id, text=html, parse_mode="HTML",
                reply_parameters=reply_params,
                reply_markup=reply_markup,
                **(thread_kwargs or {}),
            )
        except BadRequest as e:
            self.logger.warning("HTML parse failed, falling back to plain text: {}", e)
            try:
                await self._call_with_retry(
                    app.bot.send_message,
                    chat_id=chat_id,
                    text=text,
                    reply_parameters=reply_params,
                    reply_markup=reply_markup,
                    **(thread_kwargs or {}),
                )
            except Exception:
                self.logger.exception("Error sending message")
                raise

    @staticmethod
    def _is_not_modified_error(exc: Exception) -> bool:
        return isinstance(exc, BadRequest) and "message is not modified" in str(exc).lower()

    async def send_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
        stream_end: bool = False,
        resuming: bool = False,
        merge_next: bool = False,
    ) -> None:
        lock = self._stream_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._stream_locks[chat_id] = lock
        async with lock:
            await self._send_delta_locked(chat_id, delta, metadata, stream_id=stream_id,
                                          stream_end=stream_end, resuming=resuming, merge_next=merge_next)

    async def _send_delta_locked(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
        stream_end: bool = False,
        resuming: bool = False,
        merge_next: bool = False,
    ) -> None:
        """Progressive message editing: send on first delta, edit on subsequent ones."""
        app = await self._wait_for_app()
        if app is None:
            return
        meta = metadata or {}
        int_chat_id = int(chat_id)

        pending = self._stream_bufs.get(chat_id)
        if (pending is not None and pending.pending_chunks is not None
                and (stream_id is None or pending.stream_id is None or pending.stream_id == stream_id)):
            thread = meta.get("message_thread_id")
            await self._flush_stream_overflow(int_chat_id, pending,
                                             {"message_thread_id": thread} if thread is not None else {})

        if stream_end and merge_next:
            if not delta:
                return
            stream_end = False
        if stream_end:
            buf = self._stream_bufs.get(chat_id)
            if not buf or not buf.message_id or not buf.text:
                return
            if stream_id is not None and buf.stream_id is not None and buf.stream_id != stream_id:
                return
            if not resuming:
                self._stop_typing(chat_id)
            if reply_to_message_id := meta.get("message_id"):
                with suppress(ValueError):
                    await self._remove_reaction(chat_id, int(reply_to_message_id))
            thread_kwargs: dict[str, int] = {}
            if message_thread_id := meta.get("message_thread_id"):
                thread_kwargs["message_thread_id"] = message_thread_id
            raw_text = buf.text

            # Try upgrading the streaming preview to rich in place (Bot API 10.1:
            # editMessageText gained a rich_message parameter). Editing in place
            # keeps the message identity, so there is no delete-and-resend and
            # none of the flickering / dropped line breaks from issue #4470.
            # The previous branch here was unreachable: it was guarded by
            # ``not buf.message_id`` after an early return had already ensured
            # ``buf.message_id`` is set (issue #5516).
            if self.config.rich_messages and not getattr(self, "_rich_send_disabled", False):
                rich_ok = await self._try_edit_rich(int_chat_id, buf.message_id, raw_text)
                if rich_ok:
                    self._stream_bufs.pop(chat_id, None)
                    return

            # Legacy path: edit existing streaming message with HTML
            html_chunks = _split_telegram_markdown_html(raw_text, TELEGRAM_HTML_MAX_LEN)
            primary_html = html_chunks[0]
            extra_html_chunks = html_chunks[1:]
            try:
                await self._call_with_retry(
                    app.bot.edit_message_text,
                    chat_id=int_chat_id, message_id=buf.message_id,
                    text=primary_html, parse_mode="HTML",
                )
            except BadRequest as e:
                # Only fall back to plain text on actual HTML parse/format errors.
                # Network errors (TimedOut, NetworkError) should propagate immediately
                # to avoid doubling connection demand during pool exhaustion.
                if self._is_not_modified_error(e):
                    self.logger.debug("Final stream edit already applied for {}", chat_id)
                    self._stream_bufs.pop(chat_id, None)
                    return
                self.logger.debug("Final stream edit failed (HTML), trying plain: {}", e)
                # Fall back to raw markdown (not HTML) so users don't see raw tags.
                primary_plain = split_message(raw_text, TELEGRAM_MAX_MESSAGE_LEN)[0] if len(raw_text) > TELEGRAM_MAX_MESSAGE_LEN else raw_text
                try:
                    await self._call_with_retry(
                        app.bot.edit_message_text,
                        chat_id=int_chat_id, message_id=buf.message_id,
                        text=primary_plain,
                    )
                except Exception as e2:
                    if self._is_not_modified_error(e2):
                        self.logger.debug("Final stream plain edit already applied for {}", chat_id)
                    else:
                        self.logger.warning("Final stream edit failed: {}", e2)
                        raise  # Let ChannelManager handle retry
            for extra_html_chunk in extra_html_chunks:
                try:
                    await self._call_with_retry(
                        app.bot.send_message,
                        chat_id=int_chat_id, text=extra_html_chunk,
                        parse_mode="HTML",
                        **thread_kwargs,
                    )
                except Exception:
                    # Fall back to _send_text which handles HTML→plain gracefully.
                    await self._send_text(int_chat_id, extra_html_chunk)
            self._stream_bufs.pop(chat_id, None)
            return

        buf = self._stream_bufs.get(chat_id)
        if buf is None or (stream_id is not None and buf.stream_id is not None and buf.stream_id != stream_id):
            buf = _StreamBuf(stream_id=stream_id)
            self._stream_bufs[chat_id] = buf
        elif buf.stream_id is None:
            buf.stream_id = stream_id
        delivery_id = meta.get("_delivery_id")
        if delivery_id is None or delivery_id != buf.accepted_delivery_id:
            buf.text += delta
            buf.accepted_delivery_id = delivery_id

        if not buf.text.strip():
            return

        now = time.monotonic()
        stream_thread_kwargs: dict[str, int] = {}
        if message_thread_id := meta.get("message_thread_id"):
            stream_thread_kwargs["message_thread_id"] = message_thread_id
        if buf.message_id is None:
            initial = _split_telegram_markdown(buf.text, TELEGRAM_MAX_MESSAGE_LEN)[0]
            preview = _strip_md_block(initial)
            try:
                sent = await self._call_with_retry(
                    app.bot.send_message,
                    chat_id=int_chat_id, text=preview,
                    **stream_thread_kwargs,
                )
                buf.message_id = sent.message_id
                buf.last_edit = now
            except Exception as e:
                self.logger.warning("Stream initial send failed: {}", e)
                raise  # Let ChannelManager handle retry
            if len(buf.text) > TELEGRAM_MAX_MESSAGE_LEN:
                await self._flush_stream_overflow(int_chat_id, buf, stream_thread_kwargs)
        elif (now - buf.last_edit) >= self.config.stream_edit_interval:
            if len(buf.text) > TELEGRAM_MAX_MESSAGE_LEN:
                await self._flush_stream_overflow(int_chat_id, buf, stream_thread_kwargs)
                buf.last_edit = now
                return
            preview = _strip_md_block(buf.text)
            try:
                await self._call_with_retry(
                    app.bot.edit_message_text,
                    chat_id=int_chat_id, message_id=buf.message_id,
                    text=preview,
                )
                buf.last_edit = now
            except Exception as e:
                if self._is_not_modified_error(e):
                    buf.last_edit = now
                    return
                self.logger.warning("Stream edit failed: {}", e)
                raise  # Let ChannelManager handle retry

    async def _flush_stream_overflow(
        self,
        chat_id: int,
        buf: "_StreamBuf",
        thread_kwargs: dict[str, int],
    ) -> None:
        """Split an oversized stream buffer mid-flight.

        Edits the current stream message with the first chunk, sends any
        intermediate chunks as standalone messages, then opens a new message
        for the tail so subsequent deltas continue streaming into it.
        """
        if buf.pending_chunks is None:
            chunks = _split_telegram_markdown_html_chunks(buf.text, TELEGRAM_HTML_MAX_LEN)
            if len(chunks) <= 1:
                return
            buf.pending_chunks = chunks
            buf.pending_cursor = 0
            buf.pending_thread_kwargs = dict(thread_kwargs)
        thread_kwargs = buf.pending_thread_kwargs or {}
        chunks = buf.pending_chunks
        app = self._require_app()
        first_markdown, first_html = chunks[0]
        if buf.pending_cursor == 0:
            try:
                await self._call_with_retry(
                    app.bot.edit_message_text,
                    chat_id=chat_id, message_id=buf.message_id,
                    text=first_html,
                    parse_mode="HTML",
                )
            except BadRequest as e:
                if not self._is_not_modified_error(e):
                    self.logger.warning(
                        "Stream overflow HTML edit failed, falling back to plain text: {}", e
                    )
                    try:
                        await self._call_with_retry(
                            app.bot.edit_message_text,
                            chat_id=chat_id, message_id=buf.message_id,
                            text=first_markdown,
                        )
                    except Exception as plain_error:
                        if not self._is_not_modified_error(plain_error):
                            self.logger.warning("Stream overflow plain edit failed: {}", plain_error)
                            raise
            except Exception as e:
                self.logger.warning("Stream overflow edit failed: {}", e)
                raise

            buf.pending_cursor = 1

        async def send_chunk(markdown: str, html: str) -> Any:
            try:
                return await self._call_with_retry(
                    app.bot.send_message,
                    chat_id=chat_id, text=html, parse_mode="HTML", **thread_kwargs,
                )
            except BadRequest as e:
                self.logger.warning(
                    "Stream overflow HTML send failed, falling back to plain text: {}", e
                )
                return await self._call_with_retry(
                    app.bot.send_message,
                    chat_id=chat_id, text=markdown, **thread_kwargs,
                )

        while buf.pending_cursor < len(chunks):
            markdown, html = chunks[buf.pending_cursor]
            sent = await send_chunk(markdown, html)
            buf.pending_cursor += 1
            if buf.pending_cursor == len(chunks):
                buf.message_id = sent.message_id
                buf.text = markdown
        buf.pending_chunks = None
        buf.pending_thread_kwargs = None
        buf.pending_cursor = 0

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return

        user = update.effective_user
        if not self._is_update_allowed(update.message, user):
            return
        await update.message.reply_text(
            f"👋 Hi {user.first_name}! I'm nanobot.\n\n"
            "Send me a message and I'll respond!\n"
            "Type /help to see available commands."
        )

    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command for allowed users only."""
        if not update.message or not update.effective_user:
            return
        user = update.effective_user
        if not self._is_update_allowed(update.message, user):
            return
        await update.message.reply_text(build_help_text())

    @staticmethod
    def _sender_id(user: User) -> str:
        """Build sender_id with username for allowlist matching."""
        sid = str(user.id)
        return f"{sid}|{user.username}" if user.username else sid

    async def _send_pairing_code_if_private(
        self, sender_id: str, message: Message, user: User
    ) -> None:
        if message.chat.type != "private":
            return
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(message.chat_id),
            content="",
            metadata=self._build_message_metadata(message, user),
            is_dm=True,
        )

    @staticmethod
    def _derive_topic_session_key(message: Message) -> str | None:
        """Derive topic-scoped session key for Telegram chats with threads."""
        message_thread_id = getattr(message, "message_thread_id", None)
        if message_thread_id is None:
            return None
        return f"telegram:{message.chat_id}:topic:{message_thread_id}"

    @staticmethod
    def _build_message_metadata(message: Message, user: User) -> dict[str, Any]:
        """Build common Telegram inbound metadata payload."""
        reply_to = getattr(message, "reply_to_message", None)
        return {
            "message_id": message.message_id,
            "user_id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "is_group": message.chat.type != "private",
            "message_thread_id": getattr(message, "message_thread_id", None),
            "is_forum": bool(getattr(message.chat, "is_forum", False)),
            "reply_to_message_id": getattr(reply_to, "message_id", None) if reply_to else None,
        }

    async def _extract_reply_context(self, message: Message) -> str | None:
        """Extract text from the message being replied to, if any."""
        reply = getattr(message, "reply_to_message", None)
        if not reply:
            return None
        text = getattr(reply, "text", None) or getattr(reply, "caption", None) or ""
        if len(text) > TELEGRAM_REPLY_CONTEXT_MAX_LEN:
            text = text[:TELEGRAM_REPLY_CONTEXT_MAX_LEN] + "..."

        if not text:
            return None

        bot_id, _ = await self._ensure_bot_identity()
        reply_user = getattr(reply, "from_user", None)

        if bot_id and reply_user and getattr(reply_user, "id", None) == bot_id:
            return f"[Reply to bot: {text}]"
        elif reply_user and getattr(reply_user, "username", None):
            return f"[Reply to @{reply_user.username}: {text}]"
        elif reply_user and getattr(reply_user, "first_name", None):
            return f"[Reply to {reply_user.first_name}: {text}]"
        else:
            return f"[Reply to: {text}]"

    async def _download_message_media(
        self, msg: Message, *, add_failure_content: bool = False
    ) -> tuple[list[str], list[str]]:
        """Download media from a message (current or reply). Returns (media_paths, content_parts)."""
        media_file = None
        media_type = None
        if getattr(msg, "photo", None):
            media_file = msg.photo[-1]
            media_type = "image"
        elif getattr(msg, "voice", None):
            media_file = msg.voice
            media_type = "voice"
        elif getattr(msg, "audio", None):
            media_file = msg.audio
            media_type = "audio"
        elif getattr(msg, "document", None):
            media_file = msg.document
            media_type = "file"
        elif getattr(msg, "video", None):
            media_file = msg.video
            media_type = "video"
        elif getattr(msg, "video_note", None):
            media_file = msg.video_note
            media_type = "video"
        elif getattr(msg, "animation", None):
            media_file = msg.animation
            media_type = "animation"
        if not media_file or not self._app:
            return [], []
        try:
            file = await self._app.bot.get_file(media_file.file_id)
            ext = self._get_extension(
                cast(str, media_type),
                getattr(media_file, "mime_type", None),
                getattr(media_file, "file_name", None),
            )
            media_dir = get_media_dir("telegram")
            unique_id = getattr(media_file, "file_unique_id", media_file.file_id)
            file_path = media_dir / f"{unique_id}{ext}"
            await file.download_to_drive(str(file_path))
            path_str = str(file_path)
            if media_type in ("voice", "audio"):
                transcription = await self.transcribe_audio(file_path)
                if transcription:
                    self.logger.info("Transcribed {}: {}...", media_type, transcription[:50])
                    return [path_str], [f"[transcription: {transcription}]"]
                return [path_str], [f"[{media_type}: {path_str}]"]
            return [path_str], [f"[{media_type}: {path_str}]"]
        except Exception as e:
            self.logger.warning("Failed to download message media: {}", e)
            if add_failure_content:
                return [], [f"[{media_type}: download failed]"]
            return [], []

    async def _ensure_bot_identity(self) -> tuple[int | None, str | None]:
        """Load bot identity once and reuse it for mention/reply checks."""
        if self._bot_user_id is not None or self._bot_username is not None:
            return self._bot_user_id, self._bot_username
        if not self._app:
            return None, None
        bot_info = await self._app.bot.get_me()
        self._bot_user_id = getattr(bot_info, "id", None)
        self._bot_username = getattr(bot_info, "username", None)
        return self._bot_user_id, self._bot_username

    @staticmethod
    def _has_mention_entity(
        text: str,
        entities: list[MessageEntity] | None,
        bot_username: str,
        bot_id: int | None,
    ) -> bool:
        """Check Telegram mention entities against the bot username."""
        handle = f"@{bot_username}".lower()
        for entity in entities or []:
            entity_type = getattr(entity, "type", None)
            if entity_type == "text_mention":
                user = getattr(entity, "user", None)
                if user is not None and bot_id is not None and getattr(user, "id", None) == bot_id:
                    return True
                continue
            if entity_type != "mention":
                continue
            offset = getattr(entity, "offset", None)
            length = getattr(entity, "length", None)
            if offset is None or length is None:
                continue
            if text[offset : offset + length].lower() == handle:
                return True
        return handle in text.lower()

    async def _is_group_message_for_bot(self, message: Message) -> bool:
        """Allow group messages when policy is open, @mentioned, or replying to the bot."""
        if message.chat.type == "private" or self.config.group_policy == "open":
            return True

        bot_id, bot_username = await self._ensure_bot_identity()
        if bot_username:
            text = message.text or ""
            caption = message.caption or ""
            if self._has_mention_entity(
                text,
                getattr(message, "entities", None),
                bot_username,
                bot_id,
            ):
                return True
            if self._has_mention_entity(
                caption,
                getattr(message, "caption_entities", None),
                bot_username,
                bot_id,
            ):
                return True

        reply_user = getattr(getattr(message, "reply_to_message", None), "from_user", None)
        return bool(bot_id and reply_user and reply_user.id == bot_id)

    def _remember_thread_context(self, message: Message) -> None:
        """Cache Telegram thread context by chat/message id for follow-up replies."""
        message_thread_id = getattr(message, "message_thread_id", None)
        if message_thread_id is None:
            return
        key = (str(message.chat_id), message.message_id)
        self._message_threads[key] = message_thread_id
        if len(self._message_threads) > 1000:
            self._message_threads.pop(next(iter(self._message_threads)))

    @staticmethod
    def _queue_key_for_message(message: Message) -> str:
        """Return the final nanobot session key used for ordered Telegram ingress."""
        return TelegramChannel._derive_topic_session_key(message) or f"telegram:{message.chat_id}"

    @staticmethod
    def _sort_key_for_update(update: Update) -> tuple[int, int]:
        """Sort by chat message id first, then Telegram update id."""
        message = getattr(update, "message", None)
        message_id = int(getattr(message, "message_id", 0) or 0)
        update_id = int(getattr(update, "update_id", 0) or 0)
        return (message_id, update_id)

    def _enqueue_ordered_update(
        self,
        *,
        kind: Literal["command", "message"],
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Stage a Telegram update behind a short per-session reorder window."""
        message = update.message
        if message is None:
            return
        key = self._queue_key_for_message(message)
        self._inbound_buffers.setdefault(key, []).append(
            _QueuedTelegramUpdate(
                kind=kind,
                update=update,
                context=context,
                sort_key=self._sort_key_for_update(update),
            )
        )
        if key not in self._inbound_workers:
            self._inbound_workers[key] = asyncio.create_task(
                self._drain_ordered_updates(key)
            )

    async def _drain_ordered_updates(self, key: str) -> None:
        """Drain one Telegram session buffer in stable message order."""
        try:
            while self._running:
                await asyncio.sleep(0.2)
                batch = self._inbound_buffers.get(key, [])
                if not batch:
                    break
                self._inbound_buffers[key] = []
                batch.sort(key=lambda item: item.sort_key)
                for item in batch:
                    try:
                        if item.kind == "command":
                            await self._process_forward_command(item.update, item.context)
                        else:
                            await self._process_message_update(item.update, item.context)
                    except Exception as e:
                        self.logger.warning(
                            "Telegram queued update handling failed for {}: {}",
                            key,
                            e,
                        )
            if not self._inbound_buffers.get(key):
                self._inbound_buffers.pop(key, None)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning("Telegram ordered update worker failed for {}: {}", key, e)
        finally:
            if not self._inbound_buffers.get(key):
                self._inbound_workers.pop(key, None)

    async def _try_fast_command(self, message: Message) -> bool:
        """Execute operator-configured argv only; message arguments are never passed."""
        text = (message.text or "").strip()
        head = text.split(maxsplit=1)[0] if text else ""
        name = head.split("@", 1)[0].lower()
        argv = self.config.fast_commands.get(name)
        if not argv:
            return False
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=self.config.fast_command_workdir or None,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.config.fast_command_timeout)
        except BaseException:
            from nanobot.agent.tools.shell import ExecTool
            cleanup = asyncio.create_task(ExecTool.kill_process_tree(process))
            # Cleanup has bounded waits internally. Repeated cancellation of
            # this handler must not abandon its owned process or descendants.
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            raise
        output = stdout.decode(errors="replace").strip()
        error = stderr.decode(errors="replace").strip()
        if process.returncode:
            output = error or output or f"{name} failed with exit code {process.returncode}."
        output = output or f"{name} completed."
        reply = ReplyParameters(message_id=message.message_id, allow_sending_without_reply=True) if self.config.reply_to_message else None
        thread = getattr(message, "message_thread_id", None)
        for chunk in split_message(output, TELEGRAM_MAX_MESSAGE_LEN):
            await self._send_text(int(message.chat_id), chunk, reply,
                                  {"message_thread_id": thread} if thread is not None else {})
        return True

    async def _forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward slash commands to the bus for unified handling in AgentLoop."""
        if not update.message or not update.effective_user:
            return
        if not self._is_update_allowed(update.message, update.effective_user):
            return
        if not self._running:
            await self._process_forward_command(update, context)
            return
        self._enqueue_ordered_update(kind="command", update=update, context=context)

    async def _process_forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process a queued slash command."""
        message = update.message
        user = update.effective_user
        if message is None or user is None:
            return
        sender_id = self._sender_id(user)
        if not self._is_update_allowed(message, user):
            return
        self._remember_thread_context(message)

        # Strip @bot_username suffix if present
        content = message.text or ""
        if content.startswith("/") and "@" in content:
            cmd_part, *rest = content.split(" ", 1)
            cmd_part = cmd_part.split("@")[0]
            content = f"{cmd_part} {rest[0]}" if rest else cmd_part
        content = self._normalize_telegram_command(content)

        edge = self._busy_inbox
        if edge is not None:
            if not await self._is_group_message_for_bot(message):
                return
            if await edge.handle_update(message, user, content=content, direct=True):
                return

        if await self._try_fast_command(message):
            return

        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(message.chat_id),
            content=content,
            metadata=self._build_message_metadata(message, user),
            session_key=self._derive_topic_session_key(message),
            is_dm=message.chat.type == "private",
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return
        if update.message.chat.type != "private":
            self.logger.info(
                "Telegram group update received: chat_id={} chat_type={} sender_user_id={}",
                update.message.chat_id, update.message.chat.type, update.effective_user.id,
            )
        if not self._is_update_allowed(update.message, update.effective_user):
            return
        if not self._running:
            await self._process_message_update(update, context)
            return
        self._enqueue_ordered_update(kind="message", update=update, context=context)

    async def _process_message_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process a queued Telegram message update."""

        message = update.message
        user = update.effective_user
        if message is None or user is None:
            return
        chat_id = message.chat_id
        sender_id = self._sender_id(user)
        if not self._is_update_allowed(message, user):
            return
        self._remember_thread_context(message)

        direct = message.chat.type == "private" or await self._is_group_message_for_bot(message)
        if not direct:
            return

        # Build content from text and/or media
        content_parts: list[str] = []
        media_paths: list[str] = []

        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)

        # Location content
        if message.location:
            lat = message.location.latitude
            lon = message.location.longitude
            content_parts.append(f"[location: {lat}, {lon}]")

        # Download current message media
        current_media_paths, current_media_parts = await self._download_message_media(
            message, add_failure_content=True
        )
        media_paths.extend(current_media_paths)
        content_parts.extend(current_media_parts)
        if current_media_paths:
            self.logger.debug("Downloaded message media to {}", current_media_paths[0])

        # Reply context: text and/or media from the replied-to message
        reply = getattr(message, "reply_to_message", None)
        if reply is not None:
            reply_ctx = await self._extract_reply_context(message)
            reply_media, reply_media_parts = await self._download_message_media(reply)
            if reply_media:
                media_paths = reply_media + media_paths
                self.logger.debug("Attached replied-to media: {}", reply_media[0])
            tag = reply_ctx or (f"[Reply to: {reply_media_parts[0]}]" if reply_media_parts else None)
            if tag:
                content_parts.insert(0, tag)
        content = "\n".join(content_parts) if content_parts else "[empty message]"

        self.logger.debug("message from {}: {}...", sender_id, content[:50])

        str_chat_id = str(chat_id)
        metadata = self._build_message_metadata(message, user)
        session_key = self._derive_topic_session_key(message)

        # Telegram media groups: buffer briefly, forward as one aggregated turn.
        if media_group_id := getattr(message, "media_group_id", None):
            key = f"{str_chat_id}:{media_group_id}"
            if key not in self._media_group_buffers:
                self._media_group_buffers[key] = {
                    "sender_id": sender_id, "chat_id": str_chat_id,
                    "chat_type": message.chat.type,
                    "contents": [], "media": [],
                    "members": [], "member_messages": {},
                    "metadata": metadata,
                    "session_key": session_key,
                    "media_group_id": str(media_group_id),
                    "direct": direct,
                }
                self._start_typing(str_chat_id)
                self._media_group_buffers[key]["typing_task"] = self._typing_tasks.get(str_chat_id)
                await self._add_reaction(str_chat_id, message.message_id, self.config.react_emoji)
            buf = self._media_group_buffers[key]
            member = TelegramAlbumMember.create(
                str(message.message_id), content, tuple(media_paths),
            )
            if self._busy_inbox is not None:
                previous = next(
                    (value for value in buf["members"] if value.message_id == member.message_id),
                    None,
                )
                if previous is not None:
                    if previous.digest != member.digest:
                        raise InboxConflictError("album member identity payload changed")
                    return
            if content and content != "[empty message]":
                buf["contents"].append(content)
            buf["media"].extend(media_paths)
            buf["members"].append(member)
            buf["member_messages"][member.message_id] = (message, user)
            if key not in self._media_group_tasks:
                self._media_group_tasks[key] = asyncio.create_task(self._flush_media_group(key))
            return

        # Start typing indicator before processing
        self._start_typing(str_chat_id)
        await self._add_reaction(str_chat_id, message.message_id, self.config.react_emoji)

        edge = self._busy_inbox
        if edge is not None and await edge.handle_update(
            message, user, content=content, media=tuple(media_paths), direct=direct,
        ):
            return

        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str_chat_id,
            content=content,
            media=media_paths,
            metadata=metadata,
            session_key=session_key,
        )

    async def _flush_media_group(self, key: str) -> None:
        """Wait briefly, then forward buffered media-group as one turn."""
        try:
            await asyncio.sleep(0.6)
            if not (buf := self._media_group_buffers.pop(key, None)):
                return
            if not self.is_allowed(buf["sender_id"]) or not self.is_chat_allowed(
                buf["chat_id"], buf.get("chat_type", "group")
            ):
                task = buf.get("typing_task")
                if task is not None:
                    if self._typing_tasks.get(buf["chat_id"]) is task:
                        self._typing_tasks.pop(buf["chat_id"], None)
                    if not task.done():
                        task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                return
            content = "\n".join(buf["contents"]) or "[empty message]"
            edge = self._busy_inbox
            members = tuple(buf["members"])
            member_messages = buf["member_messages"]
            if edge is not None and members:
                members = tuple(sorted(members, key=lambda member: int(member.message_id)))
                content = "\n".join(member.content for member in members if member.content) or "[empty message]"
                media = tuple(path for member in members for path in member.media)
                first_id = members[0].message_id
                message, user = member_messages[first_id]
                if await edge.handle_update(
                    message,
                    user,
                    content=content,
                    media=media,
                    album_id=buf["media_group_id"],
                    album_members=members,
                    direct=buf["direct"],
                    transport_id=f"album:{buf['chat_id']}:{buf['media_group_id']}",
                ):
                    return
            await self._handle_message(
                sender_id=buf["sender_id"], chat_id=buf["chat_id"],
                content=content, media=list(dict.fromkeys(buf["media"])),
                metadata=buf["metadata"],
                session_key=buf.get("session_key"),
            )
        finally:
            self._media_group_tasks.pop(key, None)

    def _start_typing(self, chat_id: str) -> None:
        """Start sending 'typing...' indicator for a chat."""
        # Cancel any existing typing task for this chat
        self._stop_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))

    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def _add_reaction(self, chat_id: str, message_id: int, emoji: str) -> None:
        """Add emoji reaction to a message (best-effort, non-blocking)."""
        if not self._app or not emoji:
            return
        try:
            await self._app.bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception as e:
            self.logger.debug("reaction failed: {}", e)

    async def _remove_reaction(self, chat_id: str, message_id: int) -> None:
        """Remove emoji reaction from a message (best-effort, non-blocking)."""
        if not self._app:
            return
        try:
            await self._app.bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=message_id,
                reaction=[],
            )
        except Exception as e:
            self.logger.debug("reaction removal failed: {}", e)

    async def _typing_loop(self, chat_id: str) -> None:
        """Repeatedly send 'typing' action until cancelled."""
        try:
            with suppress(asyncio.CancelledError):
                while self._app:
                    await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                    await asyncio.sleep(4)
        except Exception as e:
            self.logger.debug("Typing indicator stopped for {}: {}", chat_id, e)

    @staticmethod
    def _format_telegram_error(exc: Exception | None) -> str:
        """Return a short, readable error summary for logs."""
        if exc is None:
            return "None"
        text = str(exc).strip()
        if text:
            return text
        if exc.__cause__ is not None:
            cause = exc.__cause__
            cause_text = str(cause).strip()
            if cause_text:
                return f"{exc.__class__.__name__} ({cause_text})"
            return f"{exc.__class__.__name__} ({cause.__class__.__name__})"
        return exc.__class__.__name__

    def _on_polling_error(self, exc: Exception) -> None:
        """Keep long-polling network failures to a single readable line."""
        summary = self._format_telegram_error(exc)
        if isinstance(exc, (NetworkError, TimedOut)):
            self.logger.warning("polling network issue: {}", summary)
        else:
            self.logger.error("polling error: {}", summary)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log polling / handler errors instead of silently swallowing them."""
        summary = self._format_telegram_error(context.error)

        if isinstance(context.error, (NetworkError, TimedOut)):
            self.logger.warning("network issue: {}", summary)
        else:
            self.logger.error("error: {}", summary)

    def _get_extension(
        self,
        media_type: str,
        mime_type: str | None,
        filename: str | None = None,
    ) -> str:
        """Get file extension based on media type or original filename."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                "image/webp": ".webp",
                "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
                "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
                "video/x-matroska": ".mkv", "video/3gpp": ".3gp",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]

        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "video": ".mp4", "file": ""}
        if ext := type_map.get(media_type, ""):
            return ext

        if filename:
            return "".join(Path(filename).suffixes)

        return ""

    def _build_keyboard(self, buttons: list[list[str]]) -> InlineKeyboardMarkup | None:
        """Build inline keyboard markup if inline_keyboards is enabled."""
        if not buttons or not self.config.inline_keyboards:
            return None
        keyboard = [
            [InlineKeyboardButton(label, callback_data=self._safe_callback_data(label)) for label in row]
            for row in buttons
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def _safe_callback_data(label: str) -> str:
        # Telegram caps callback_data at 64 bytes UTF-8; truncate at a char boundary so the keyboard still sends.
        encoded = label.encode("utf-8")
        if len(encoded) <= 64:
            return label
        return encoded[:64].decode("utf-8", errors="ignore")

    @staticmethod
    def _buttons_as_text(buttons: list[list[str]]) -> str:
        # Buttons are semantic options; when we can't render a keyboard, the user still needs to see them.
        return "\n".join(" ".join(f"[{label}]" for label in row) for row in buttons if row)

    async def _on_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline keyboard button clicks (callback queries)."""
        if not update.callback_query or not update.effective_user:
            return
        query = update.callback_query
        user = update.effective_user
        query_message = query.message
        chat_id = query_message.chat.id if query_message else None
        sender_id = self._sender_id(user)
        if query_message is None or not chat_id:
            self.logger.warning("Callback query without chat_id")
            return
        if not self.is_allowed(sender_id):
            return
        if not self.is_chat_allowed(chat_id, query_message.chat.type):
            return
        button_label = query.data or ""
        await query.answer()
        if isinstance(query_message, Message):
            with suppress(Exception):
                await query_message.edit_reply_markup(reply_markup=None)
        self.logger.debug("Inline button tap from {}: {}", sender_id, button_label)
        self._start_typing(str(chat_id))
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(chat_id),
            content=button_label,
            metadata={
                "callback_query_id": query.id,
                "button_label": button_label,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_callback": True,
            },
        )
