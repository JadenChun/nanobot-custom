"""Session ownership boundary for staged runtime composition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Protocol, runtime_checkable

from nanobot.session.manager import SessionManager
from nanobot.session.mode_storage import LifetimeWriterLease
from nanobot.session.session_mode import SessionModeError, SessionModeResolver


class SessionAuthorityUnavailableError(RuntimeError):
    """No durable session authority is installed for the Stage 1 runtime."""


SessionCloseCallback = Callable[[], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class SessionAuthorityLease:
    """A concrete manager plus an optional authority-owned close operation."""

    manager: SessionManager
    close: SessionCloseCallback | None = None

    @property
    def owned(self) -> bool:
        return self.close is not None


@runtime_checkable
class SessionAuthority(Protocol):
    """Acquire one already-resolved session manager without mode selection."""

    async def acquire(self) -> SessionAuthorityLease: ...


@dataclass(frozen=True, slots=True)
class BorrowedSessionAuthority:
    """Test/embedder authority for a caller-owned, non-mutating manager."""

    manager: SessionManager

    async def acquire(self) -> SessionAuthorityLease:
        return SessionAuthorityLease(self.manager)


class Stage1UnavailableSessionAuthority:
    """Fail-closed authority used until Stage 2 installs durable resolution."""

    async def acquire(self) -> SessionAuthorityLease:
        raise SessionAuthorityUnavailableError(
            "session authority is unavailable during P5 Stage 1"
        )


class DurableSessionAuthority:
    """Definition-only Stage 2 authority for one cooperating runtime writer."""

    def __init__(self, workspace: Path, *, sessions_root: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self.sessions_root = Path(sessions_root).expanduser().resolve(strict=False)
        self._gate = asyncio.Lock()
        self._lease: SessionAuthorityLease | None = None

    async def acquire(self) -> SessionAuthorityLease:
        async with self._gate:
            if self._lease is not None:
                return self._lease
            writer = LifetimeWriterLease(self.workspace, self.sessions_root)
            writer.acquire()
            close_resolver: Callable[[], None] | None = None

            def retain_close(action: Callable[[], None]) -> None:
                nonlocal close_resolver
                close_resolver = action

            try:
                access = writer.issue_access(operator=False)
                manager = SessionManager(
                    self.workspace,
                    sessions_root=self.sessions_root,
                    _managed_access=access,
                )
                # Existing data without durable mode evidence is never adopted at startup.
                SessionManager.compose_session_modes(
                    manager, lambda: SessionModeResolver, retain_close,
                    operator=False, verify_registered_modes=True,
                )
            except BaseException:
                if close_resolver is not None:
                    close_resolver()
                writer.close()
                raise
            retained_close = close_resolver
            assert retained_close is not None
            closed = False

            async def close() -> None:
                nonlocal closed
                if closed:
                    return
                closed = True
                retained_close()
                writer.close()

            self._lease = SessionAuthorityLease(manager, close)
            return self._lease


class OfflineSessionOperator:
    """Explicit copied-state owner for bootstrap/conversion/export operations."""

    def __init__(
        self,
        copied_workspace: Path,
        *,
        copied_sessions_root: Path,
        live_workspace: Path | None = None,
        live_sessions_root: Path | None = None,
    ) -> None:
        self.workspace = Path(copied_workspace).expanduser().resolve(strict=False)
        self.sessions_root = Path(copied_sessions_root).expanduser().resolve(strict=False)
        for copied, live in (
            (self.workspace, live_workspace),
            (self.sessions_root, live_sessions_root),
        ):
            if live is not None:
                live_path = Path(live).expanduser().resolve(strict=False)
                if copied == live_path or (
                    copied.exists() and live_path.exists() and copied.samefile(live_path)
                ):
                    raise SessionModeError("copied-state roots must not identify live state")

    async def acquire(self) -> SessionAuthorityLease:
        writer = LifetimeWriterLease(self.workspace, self.sessions_root)
        writer.acquire()
        close_resolver: Callable[[], None] | None = None

        def retain_close(action: Callable[[], None]) -> None:
            nonlocal close_resolver
            close_resolver = action

        try:
            writer.rebind_copied_workspace_claim()
            access = writer.issue_access(operator=True)
            manager = SessionManager(
                self.workspace,
                sessions_root=self.sessions_root,
                _managed_access=access,
            )
            SessionManager.compose_session_modes(
                manager, lambda: SessionModeResolver, retain_close,
                operator=True, verify_registered_modes=False,
            )
        except BaseException:
            if close_resolver is not None:
                close_resolver()
            writer.close()
            raise
        retained_close = close_resolver
        assert retained_close is not None
        closed = False

        async def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            retained_close()
            writer.close()

        return SessionAuthorityLease(manager, close)


__all__ = [
    "BorrowedSessionAuthority",
    "DurableSessionAuthority",
    "OfflineSessionOperator",
    "SessionAuthority",
    "SessionAuthorityLease",
    "SessionAuthorityUnavailableError",
    "Stage1UnavailableSessionAuthority",
]
