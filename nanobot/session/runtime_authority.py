"""Session ownership boundary for staged runtime composition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Protocol, cast, runtime_checkable

from nanobot.session.manager import SessionManager
from nanobot.session.mode_storage import LifetimeWriterLease
from nanobot.session.session_mode import SessionModeError, SessionModeResolver

RUNTIME_SESSION_METADATA_KEY = "_runtime_session"
RUNTIME_SESSION_METADATA_VERSION = 1


class RuntimeSessionMetadataError(ValueError):
    """A runtime session reference is malformed or no longer current."""


@dataclass(frozen=True, slots=True)
class RuntimeSessionMetadata:
    """Versioned, content-free runtime facts attached to one session."""

    profile: str = "default"
    execution_kind: Literal["native", "external"] = "native"
    delegation_depth: int = 0
    parent_key: str | None = None
    parent_incarnation: str | None = None
    backend_binding: str | None = None
    runtime_generation: int = 0
    recovery_disposition: Literal[
        "none", "interrupted", "completed", "uncertain",
    ] = "none"

    def __post_init__(self) -> None:
        if type(self.profile) is not str or not self.profile.strip():
            raise RuntimeSessionMetadataError("runtime profile must be non-empty")
        if self.execution_kind not in {"native", "external"}:
            raise RuntimeSessionMetadataError("runtime execution kind is invalid")
        if type(self.delegation_depth) is not int or self.delegation_depth < 0:
            raise RuntimeSessionMetadataError("runtime delegation depth is invalid")
        if (self.parent_key is None) != (self.parent_incarnation is None):
            raise RuntimeSessionMetadataError("runtime parent reference is incomplete")
        for value in (self.parent_key, self.parent_incarnation, self.backend_binding):
            if value is not None and (type(value) is not str or not value.strip()):
                raise RuntimeSessionMetadataError("runtime reference is invalid")
        if type(self.runtime_generation) is not int or self.runtime_generation < 0:
            raise RuntimeSessionMetadataError("runtime generation is invalid")
        if self.recovery_disposition not in {
            "none", "interrupted", "completed", "uncertain",
        }:
            raise RuntimeSessionMetadataError("runtime recovery disposition is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "version": RUNTIME_SESSION_METADATA_VERSION,
            "profile": self.profile,
            "execution_kind": self.execution_kind,
            "delegation_depth": self.delegation_depth,
            "parent_key": self.parent_key,
            "parent_incarnation": self.parent_incarnation,
            "backend_binding": self.backend_binding,
            "runtime_generation": self.runtime_generation,
            "recovery_disposition": self.recovery_disposition,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSessionState:
    """Parsed runtime facts plus whether they may be used for execution."""

    metadata: RuntimeSessionMetadata
    present: bool
    executable: bool
    issue: str | None = None


def parse_runtime_session_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    parent_incarnation: str | None = None,
) -> RuntimeSessionState:
    """Parse old, current, malformed, or stale runtime session metadata."""
    raw = metadata.get(RUNTIME_SESSION_METADATA_KEY) if metadata is not None else None
    if raw is None:
        return RuntimeSessionState(RuntimeSessionMetadata(), False, True)
    if not isinstance(raw, Mapping):
        return RuntimeSessionState(
            RuntimeSessionMetadata(), True, False, "malformed",
        )
    data = dict(cast(Mapping[str, Any], raw))
    if data.get("version") != RUNTIME_SESSION_METADATA_VERSION:
        return RuntimeSessionState(
            RuntimeSessionMetadata(), True, False, "unsupported_version",
        )
    try:
        parsed = RuntimeSessionMetadata(
            profile=cast(str, data.get("profile", "default")),
            execution_kind=cast(
                Literal["native", "external"],
                data.get("execution_kind", "native"),
            ),
            delegation_depth=cast(int, data.get("delegation_depth", 0)),
            parent_key=cast(str | None, data.get("parent_key")),
            parent_incarnation=cast(str | None, data.get("parent_incarnation")),
            backend_binding=cast(str | None, data.get("backend_binding")),
            runtime_generation=cast(int, data.get("runtime_generation", 0)),
            recovery_disposition=cast(
                Literal["none", "interrupted", "completed", "uncertain"],
                data.get("recovery_disposition", "none"),
            ),
        )
    except (TypeError, ValueError):
        return RuntimeSessionState(
            RuntimeSessionMetadata(), True, False, "malformed",
        )
    if parsed.parent_key is not None and parent_incarnation != parsed.parent_incarnation:
        return RuntimeSessionState(parsed, True, False, "stale_parent")
    return RuntimeSessionState(parsed, True, True)


def read_runtime_session(
    manager: SessionManager,
    key: str,
) -> RuntimeSessionState:
    """Read runtime facts without making an unreadable history unusable."""
    session = manager.read_session_snapshot(key)
    if session is None:
        return RuntimeSessionState(RuntimeSessionMetadata(), False, True)
    raw = session.metadata.get(RUNTIME_SESSION_METADATA_KEY)
    parent_incarnation = None
    if isinstance(raw, Mapping):
        parent_key = cast(Mapping[str, Any], raw).get("parent_key")
        if isinstance(parent_key, str) and parent_key:
            parent = manager.read_session_snapshot(parent_key)
            if parent is not None:
                parent_incarnation = parent.incarnation
    return parse_runtime_session_metadata(
        session.metadata,
        parent_incarnation=parent_incarnation,
    )


def write_runtime_session(
    manager: SessionManager,
    key: str,
    metadata: RuntimeSessionMetadata,
    *,
    create_if_absent: bool = False,
) -> bool:
    """Persist runtime facts through the existing metadata owner."""
    if metadata.parent_key is not None:
        parent = manager.read_session_snapshot(metadata.parent_key)
        if parent is None or parent.incarnation != metadata.parent_incarnation:
            raise RuntimeSessionMetadataError("runtime parent reference is stale")
    return manager.upsert_session_metadata(
        key,
        {RUNTIME_SESSION_METADATA_KEY: metadata.payload()},
        create_if_absent=create_if_absent,
    )


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
    "RUNTIME_SESSION_METADATA_KEY",
    "RUNTIME_SESSION_METADATA_VERSION",
    "RuntimeSessionMetadata",
    "RuntimeSessionMetadataError",
    "RuntimeSessionState",
    "SessionAuthority",
    "SessionAuthorityLease",
    "SessionAuthorityUnavailableError",
    "Stage1UnavailableSessionAuthority",
    "parse_runtime_session_metadata",
    "read_runtime_session",
    "write_runtime_session",
]
