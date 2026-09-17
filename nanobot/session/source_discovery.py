"""Bounded, read-only discovery of converted source sessions.

The ordinary session listing path intentionally reads previews and may repair
legacy files.  Idle admission needs a much smaller authority boundary: walk
the exact converted-session directory, inspect only a bounded first header
line, and return canonical keys.  This module owns that boundary and never
loads a complete session, writes a file, repairs a header, or populates a
cache.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.session.source import SourceValidationError
from nanobot.session.source_root import SourceRootAuthority, SourceRootAuthorityError

if TYPE_CHECKING:
    from nanobot.session.manager import JsonlSessionStore


class SourceDiscoveryError(RuntimeError):
    """Base error for a source discovery cursor."""


class SourceDiscoveryClosedError(SourceDiscoveryError):
    """The cursor was closed and cannot be scanned again."""


class SourceDiscoveryRootChangedError(SourceDiscoveryError):
    """The pinned source directory no longer identifies the same directory."""


SourceDiscoveryRootChanged = SourceDiscoveryRootChangedError


@dataclass(frozen=True, slots=True)
class SourceDiscoveryLimits:
    """Positive physical and header limits for one discovery cursor."""

    max_entries_inspected: int
    max_candidates_returned: int
    max_header_bytes: int
    max_decoded_key_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "max_entries_inspected",
            "max_candidates_returned",
            "max_header_bytes",
            "max_decoded_key_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    # Short aliases make the explicit bounds convenient for callers without
    # introducing optional or production-selected defaults.
    @property
    def entries_inspected(self) -> int:
        return self.max_entries_inspected

    @property
    def candidates_returned(self) -> int:
        return self.max_candidates_returned

    @property
    def header_bytes(self) -> int:
        return self.max_header_bytes

    @property
    def decoded_key_bytes(self) -> int:
        return self.max_decoded_key_bytes


@dataclass(frozen=True, slots=True)
class SourceDiscoveryCounters:
    """Content-free counters for one physical traversal slice."""

    inspected: int
    returned: int
    rejected: int
    exhausted: bool

    def __post_init__(self) -> None:
        for name in ("inspected", "returned", "rejected"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.exhausted) is not bool:
            raise TypeError("exhausted must be a boolean")
        if self.returned + self.rejected > self.inspected:
            raise ValueError("discovery counters exceed inspected entries")

    @property
    def entries_inspected(self) -> int:
        return self.inspected

    @property
    def candidates_returned(self) -> int:
        return self.returned

    @property
    def invalid(self) -> int:
        return self.rejected


@dataclass(frozen=True, slots=True)
class SourceDiscoveryResult:
    """Canonical keys and content-free counters from one scan slice."""

    keys: tuple[str, ...]
    counters: SourceDiscoveryCounters

    def __post_init__(self) -> None:
        if type(self.keys) is not tuple or any(type(key) is not str for key in self.keys):
            raise TypeError("keys must be a tuple of strings")
        if type(self.counters) is not SourceDiscoveryCounters:
            raise TypeError("counters must be SourceDiscoveryCounters")

    @property
    def candidate_keys(self) -> tuple[str, ...]:
        return self.keys

    @property
    def session_keys(self) -> tuple[str, ...]:
        return self.keys

    @property
    def inspected(self) -> int:
        return self.counters.inspected

    @property
    def returned(self) -> int:
        return self.counters.returned

    @property
    def rejected(self) -> int:
        return self.counters.rejected

    @property
    def exhausted(self) -> bool:
        return self.counters.exhausted


def _directory_identity(path: Path) -> tuple[int, int, int]:
    """Return a stable identity while refusing a symlink root."""

    try:
        raw = path.lstat()
    except OSError as exc:
        raise SourceDiscoveryRootChanged("source directory is unavailable") from exc
    if stat.S_ISLNK(raw.st_mode) or not stat.S_ISDIR(raw.st_mode):
        raise SourceDiscoveryRootChanged("source directory is not a regular directory")
    return raw.st_dev, raw.st_ino, raw.st_mode & stat.S_IFMT(raw.st_mode)


def _strict_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return value


class SourceDiscoveryCursor:
    """Process-local cursor over one exact ``JsonlSessionStore.sessions_dir``.

    Construction pins only the directory identity and creates no iterator.
    The iterator is opened on the first explicit scan and is closed whenever a
    traversal is exhausted, the root changes, or :meth:`close` is called.
    """

    def __init__(
        self,
        backend: JsonlSessionStore,
        *,
        limits: SourceDiscoveryLimits,
        root_authority: SourceRootAuthority | None = None,
        own_root_authority: bool = False,
    ) -> None:
        if type(limits) is not SourceDiscoveryLimits:
            raise TypeError("limits must be SourceDiscoveryLimits")
        sessions_dir = getattr(backend, "sessions_dir", None)
        if not isinstance(sessions_dir, Path):
            raise TypeError("backend must expose an exact sessions_dir")
        self._backend = backend
        issued_authority = False
        if root_authority is None:
            root_authority = SourceRootAuthority.issue_for_backend(backend)
            self._owns_authority = True
            issued_authority = True
        else:
            if root_authority._backend is not backend:
                raise TypeError("root_authority must belong to the exact backend")
            self._owns_authority = own_root_authority
        try:
            self._root_authority = root_authority
            # Keep the lexical root so a symlink swap cannot be silently resolved
            # to a different directory.  JsonlSessionStore normally supplies an
            # already canonical directory, but discovery owns this extra check.
            self._root = sessions_dir.expanduser().absolute()
            self._root_identity = _directory_identity(self._root)
            self._limits = limits
            self._iterator: Iterator[os.DirEntry[str]] | None = None
            self._pending_key: str | None = None
            self._closed = False
            self._resources_closed = False
            self._scan_lock = asyncio.Lock()
        except BaseException:
            # A cursor that issued its own authority must release it even if
            # root qualification fails before the instance is usable. Borrowed
            # authorities remain owned by their caller.
            if issued_authority:
                root_authority.close()
            raise

    @property
    def limits(self) -> SourceDiscoveryLimits:
        return self._limits

    @property
    def root(self) -> Path:
        return self._root

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def cursor_open(self) -> bool:
        return self._iterator is not None

    @property
    def root_authority(self) -> SourceRootAuthority:
        return self._root_authority

    def _close_iterator(self) -> None:
        iterator = self._iterator
        self._iterator = None
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def _invalidate_root(self, message: str, *, cause: BaseException | None = None) -> None:
        self._closed = True
        self._release_resources()
        if cause is None:
            raise SourceDiscoveryRootChanged(message)
        raise SourceDiscoveryRootChanged(message) from cause

    def _release_resources(self) -> None:
        if self._resources_closed:
            return
        self._resources_closed = True
        self._close_iterator()
        self._pending_key = None
        if self._owns_authority:
            self._root_authority.close()

    def _check_root(self) -> None:
        try:
            self._root_authority.require_current(self._backend)
        except SourceRootAuthorityError as exc:
            self._invalidate_root("source directory identity changed", cause=exc)

    def _open_iterator(self) -> Iterator[os.DirEntry[str]]:
        self._check_root()
        try:
            return os.scandir(self._root_authority._fd)
        except OSError as exc:
            self._invalidate_root("source directory is unavailable", cause=exc)

    def _read_candidate(self, entry: os.DirEntry[str]) -> str | None:
        """Validate one physical entry without loading its session history."""

        # lstat through DirEntry is intentionally required: follow_symlinks is
        # false for both the type check and the identity check.
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError:
            return None
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISREG(entry_stat.st_mode):
            return None
        try:
            name = entry.name
            if not name.endswith(".jsonl") or name == ".jsonl":
                return None
            # The source store's helper is deliberately used for both decode
            # and the canonical filename round-trip.
            path = Path(name)
            key = self._backend.session_key_from_path(path)
            if key is None or self._backend.storage_key(key) != name[:-6]:
                return None
            if len(key.encode("utf-8", "strict")) > self._limits.max_decoded_key_bytes:
                return None
            # Open the exact regular entry without following a replacement
            # symlink, then compare its identity with the lstat result.  This
            # keeps the containment/type check meaningful across the small
            # lstat-to-read race while still reading only the bounded header.
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            descriptor = self._root_authority.open_regular(name, flags=flags)
            try:
                opened_stat = os.fstat(descriptor)
                if (
                    opened_stat.st_dev != entry_stat.st_dev
                    or opened_stat.st_ino != entry_stat.st_ino
                    or not stat.S_ISREG(opened_stat.st_mode)
                ):
                    return None
                raw_buffer = bytearray()
                while len(raw_buffer) <= self._limits.max_header_bytes:
                    chunk = os.read(descriptor, 1)
                    if not chunk:
                        break
                    raw_buffer.extend(chunk)
                    if chunk == b"\n":
                        break
                raw = bytes(raw_buffer)
            finally:
                os.close(descriptor)
            newline = raw.find(b"\n")
            if newline < 0 or newline > self._limits.max_header_bytes:
                return None
            line = raw[:newline]
            if len(line) > self._limits.max_header_bytes:
                return None
            try:
                decoded = line.decode("utf-8", "strict")
                from nanobot.session.source_store import _object_pairs

                value = json.loads(decoded, object_pairs_hook=_object_pairs)
            except (UnicodeError, ValueError, TypeError, RecursionError):
                return None
            header = _strict_object(value)
            if header is None or header.get("_type") != "metadata":
                return None
            if header.get("key") != key:
                return None
            metadata = header.get("metadata")
            if not isinstance(metadata, dict) or type(metadata.get("source_schema_version")) is not int or metadata.get("source_schema_version") != 1:
                return None
            return key
        except (
            OSError,
            UnicodeError,
            ValueError,
            TypeError,
            RecursionError,
            SourceValidationError,
            SourceRootAuthorityError,
        ):
            return None

    def _scan_slice(self) -> SourceDiscoveryResult:
        if self._closed:
            raise SourceDiscoveryClosedError("source discovery cursor is closed")
        self._check_root()
        if self._iterator is None:
            self._iterator = self._open_iterator()
        keys: list[str] = []
        inspected = 0
        rejected = 0
        exhausted = False
        try:
            if self._pending_key is not None:
                keys.append(self._pending_key)
                self._pending_key = None
            while (
                inspected < self._limits.max_entries_inspected
                and len(keys) < self._limits.max_candidates_returned
            ):
                try:
                    entry = next(self._iterator)
                except StopIteration:
                    exhausted = True
                    self._close_iterator()
                    break
                except OSError as exc:
                    self._invalidate_root("source traversal failed", cause=exc)
                inspected += 1
                key = self._read_candidate(entry)
                if key is None:
                    rejected += 1
                    continue
                keys.append(key)
        finally:
            # Detect replacement between slices and invalidate the live cursor.
            # A mutation race is content-free; callers can retry from a fresh
            # traversal after observing the root-change outcome.
            try:
                self._check_root()
            except SourceDiscoveryRootChanged:
                raise
        return SourceDiscoveryResult(
            tuple(keys),
            SourceDiscoveryCounters(inspected, len(keys), rejected, exhausted),
        )

    async def scan_once(self) -> SourceDiscoveryResult:
        """Inspect one bounded traversal slice, serializing overlapping calls."""

        async with self._scan_lock:
            return self._scan_slice()

    async def discover_once(self) -> SourceDiscoveryResult:
        """Compatibility spelling for :meth:`scan_once`."""

        return await self.scan_once()

    def close(self) -> None:
        self._closed = True
        self._release_resources()

    async def aclose(self) -> None:
        async with self._scan_lock:
            self.close()


# Names used by integration callers and tests remain deliberately descriptive;
# they all identify the same exact cursor implementation.
BoundedSourceDiscovery = SourceDiscoveryCursor
SourceSessionDiscovery = SourceDiscoveryCursor
SourceDiscovery = SourceDiscoveryCursor


__all__ = [
    "BoundedSourceDiscovery",
    "SourceDiscovery",
    "SourceDiscoveryClosedError",
    "SourceDiscoveryCounters",
    "SourceDiscoveryError",
    "SourceDiscoveryLimits",
    "SourceDiscoveryResult",
    "SourceDiscoveryRootChanged",
    "SourceDiscoveryRootChangedError",
    "SourceDiscoveryCursor",
    "SourceSessionDiscovery",
]
