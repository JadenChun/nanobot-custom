"""Logical prompt-budget measurement and portable file locks.

The configured ``maxTokens.input`` value is the total logical context budget,
including the completion reserve.  This module is deliberately independent of
the agent/session layers so provider boundaries, delegated roles, and memory
consolidation can all apply the same rule.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from kohaku.utils.helpers import estimate_prompt_tokens_chain

DEFAULT_PROMPT_SAFETY_BUFFER = 1024


class PortableFileLock:
    """Small stdlib-only advisory lock usable by CLI and gateway processes."""

    # ``AsyncPortableFileLock`` may release the lock from a different worker
    # thread than the one which acquired it.  A plain lock is deliberately
    # used here: unlike ``RLock``, its ownership is not tied to a thread.
    _thread_locks: dict[str, threading.Lock] = {}
    _thread_guard = threading.Lock()

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        self._fd: int | None = None
        with self._thread_guard:
            self._local_lock = self._thread_locks.setdefault(self.path, threading.Lock())

    def acquire(self) -> None:
        self._local_lock.acquire()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(self._fd)
            self._fd = None
            self._local_lock.release()
            raise

    def release(self) -> None:
        fd = self._fd
        self._fd = None
        try:
            if fd is not None:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        finally:
            self._local_lock.release()

    def __enter__(self) -> "PortableFileLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()


@contextmanager
def portable_file_lock(path: str | os.PathLike[str]):
    """Context-manager spelling for synchronous persistence paths."""
    lock = PortableFileLock(path)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


class AsyncPortableFileLock:
    """Async wrapper that avoids blocking the event loop while waiting."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._lock = PortableFileLock(path)
        self._acquire_task: Any | None = None
        self._acquired = False

    async def __aenter__(self) -> "AsyncPortableFileLock":
        import asyncio

        # ``to_thread`` cancellation does not cancel the underlying blocking
        # acquire.  Shield it and arrange for a late successful acquire to be
        # released, otherwise cancelling a waiter can leak a process lock.
        task = asyncio.create_task(asyncio.to_thread(self._lock.acquire))
        self._acquire_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            def _release_late(done: asyncio.Future[None]) -> None:
                if done.cancelled() or done.exception() is not None:
                    return
                try:
                    self._lock.release()
                except Exception:
                    pass

            task.add_done_callback(_release_late)
            raise
        self._acquired = True
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        import asyncio

        if not self._acquired:
            return
        self._acquired = False
        await asyncio.to_thread(self._lock.release)


@dataclass(frozen=True, slots=True)
class PromptBudget:
    """Total logical context and the reserve held back for a completion."""

    total_tokens: int
    completion_reserve: int = 0
    safety_buffer: int = DEFAULT_PROMPT_SAFETY_BUFFER

    @property
    def prompt_limit(self) -> int:
        """Maximum logical input tokens allowed before provider transport."""
        return max(
            0,
            int(self.total_tokens)
            - max(0, int(self.completion_reserve))
            - max(0, int(self.safety_buffer)),
        )

    # Compatibility spellings used by older context-budget callers.
    @property
    def effective_prompt_limit(self) -> int:
        return self.prompt_limit

    @property
    def available_budget(self) -> int:
        return self.prompt_limit

    @property
    def context_window_tokens(self) -> int:
        return self.total_tokens

    @property
    def max_tokens(self) -> int:
        return self.total_tokens

    @classmethod
    def from_generation(
        cls,
        generation: Any,
        *,
        total_tokens: int | None = None,
        completion_reserve: int | None = None,
        safety_buffer: int = DEFAULT_PROMPT_SAFETY_BUFFER,
    ) -> "PromptBudget | None":
        """Build a budget from provider generation settings when configured."""
        total = total_tokens
        if total is None:
            total = getattr(generation, "context_window_tokens", 0)
        if not isinstance(total, (int, float)) or int(total) <= 0:
            return None
        reserve = completion_reserve
        if reserve is None:
            reserve = getattr(generation, "max_tokens", 0)
        return cls(
            total_tokens=int(total),
            completion_reserve=max(0, int(reserve or 0)),
            safety_buffer=safety_buffer,
        )


@dataclass(frozen=True, slots=True)
class PromptMeasurement:
    """A measured prompt and the budget against which it was checked."""

    prompt_tokens: int
    source: str
    limit: int | None = None
    total_tokens: int | None = None
    completion_reserve: int = 0
    safety_buffer: int = DEFAULT_PROMPT_SAFETY_BUFFER

    @property
    def fits(self) -> bool:
        return self.limit is None or self.prompt_tokens <= self.limit

    @property
    def over_by(self) -> int:
        if self.limit is None:
            return 0
        return max(0, self.prompt_tokens - self.limit)


class PromptBudgetExceeded(Exception):  # noqa: N818 - public contract name
    """Raised locally when a prompt cannot fit; never includes prompt content."""

    DEFAULT_MESSAGE = "Prompt exceeds the configured context budget."

    def __init__(self, measurement: PromptMeasurement | None = None) -> None:
        self.measurement = measurement
        # Keep exception text deterministic and content-free.  In particular,
        # do not include the offending user message or serialized tool schema.
        super().__init__(self.DEFAULT_MESSAGE)


def _fallback_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    """Conservative deterministic fallback if a tokenizer is unavailable."""
    try:
        payload = json.dumps(
            {"messages": messages, "tools": tools or []},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return max(1, (len(payload) + 3) // 4 + len(messages) * 4)
    except Exception:
        return max(1, sum(len(str(message)) for message in messages) // 4)


def measure_prompt(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    budget: PromptBudget | None = None,
) -> PromptMeasurement:
    """Measure a fully rendered message/tool payload without mutating it."""
    measured, source = estimate_prompt_tokens_chain(provider, model, messages, tools)
    if not isinstance(cast(object, measured), int) or measured <= 0:
        measured = _fallback_prompt_tokens(messages, tools)
        source = "fallback"
    return PromptMeasurement(
        prompt_tokens=int(measured),
        source=str(source),
        limit=budget.prompt_limit if budget is not None else None,
        total_tokens=budget.total_tokens if budget is not None else None,
        completion_reserve=budget.completion_reserve if budget is not None else 0,
        safety_buffer=budget.safety_buffer if budget is not None else DEFAULT_PROMPT_SAFETY_BUFFER,
    )


# Descriptive aliases make the boundary helper easy to discover from tests and
# integrations while keeping one implementation.
measure_prompt_tokens = measure_prompt


def assert_prompt_fits(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    budget: PromptBudget | None = None,
) -> PromptMeasurement:
    """Measure and raise :class:`PromptBudgetExceeded` when over the limit."""
    measurement = measure_prompt(provider, model, messages, tools, budget)
    if budget is not None and not measurement.fits:
        raise PromptBudgetExceeded(measurement)
    return measurement


assert_prompt_budget = assert_prompt_fits
