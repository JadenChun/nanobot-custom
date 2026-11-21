"""Live progress channel from a running tool to the user-visible activity UI.

Some tools can report incremental progress while they run: an MCP server sends
``notifications/progress`` frames, and the ComfyUI harness uses them for GPU
handoff and generation phases. The sink below carries those frames from the
tool call to the progress hook without touching persisted history.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

ToolProgressPublish = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ToolProgressSink:
    """Publish endpoint bound to the tool call that is running now."""

    call_id: str
    name: str
    publish: ToolProgressPublish

    async def emit(
        self,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        payload: dict[str, Any] = {
            "version": 1,
            "phase": "progress",
            "call_id": self.call_id,
            "name": self.name,
            "progress": progress,
            "total": total,
            "message": message or "",
        }
        await self.publish(payload)


_CURRENT_SINK: ContextVar[ToolProgressSink | None] = ContextVar(
    "nanobot_tool_progress_sink",
    default=None,
)


@contextmanager
def bind_tool_progress_sink(sink: ToolProgressSink | None) -> Generator[None, None, None]:
    """Scope a sink to one tool execution."""
    token = _CURRENT_SINK.set(sink)
    try:
        yield
    finally:
        _CURRENT_SINK.reset(token)


def current_tool_progress_sink() -> ToolProgressSink | None:
    return _CURRENT_SINK.get()


class ProgressThrottle:
    """Coalesce rapid progress bursts without hiding real phase changes.

    A changed message always passes. Repeated updates of the same phase are
    limited to one frame per interval.
    """

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self._interval = interval_seconds
        self._last_emit = 0.0
        self._last_message: str | None = None
        self._last_total: float | None = None

    def should_emit(
        self,
        total: float | None,
        message: str | None,
        *,
        now: float | None = None,
    ) -> bool:
        current = time.monotonic() if now is None else now
        message_changed = (message or "") != (self._last_message or "")
        total_changed = total != self._last_total
        if message_changed or total_changed or current - self._last_emit >= self._interval:
            self._last_emit = current
            self._last_message = message
            self._last_total = total
            return True
        return False

    def reset(self) -> None:
        self._last_emit = 0.0
        self._last_message = None
        self._last_total = None
