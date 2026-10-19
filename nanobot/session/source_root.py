"""Pinned descriptor authority for the converted-session directory.

The ordinary session APIs retain their pathname behavior.  The dormant I3d
path uses :class:`SourceRootAuthority` instead: it pins the exact directory
inode, performs all source and lock operations relative to that descriptor,
and fails closed when the POSIX primitives needed for that guarantee are not
available.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator as _Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, NoReturn, cast

if TYPE_CHECKING:
    from nanobot.session.manager import JsonlSessionStore

try:  # pragma: no cover - import availability is platform dependent.
    import fcntl
except ImportError:  # pragma: no cover - exercised by platform qualification.
    fcntl = None  # type: ignore[assignment]


class SourceRootAuthorityError(RuntimeError):
    """The pinned source-root capability cannot be used safely."""


class SourceRootUnavailableError(SourceRootAuthorityError):
    """The required descriptor-relative POSIX operations are unavailable."""


class SourceRootClosedError(SourceRootAuthorityError):
    """The pinned source-root descriptor has been closed."""


def _supports_dir_fd(function: object) -> bool:
    return function in getattr(os, "supports_dir_fd", set[object]())


def _require_platform() -> None:
    required_flags = ("O_NOFOLLOW", "O_DIRECTORY")
    if any(not isinstance(getattr(os, name, None), int) for name in required_flags):
        raise SourceRootUnavailableError("descriptor-relative no-follow directory access is unavailable")
    if fcntl is None or not all(
        isinstance(getattr(fcntl, name, None), int)
        for name in ("LOCK_EX", "LOCK_NB", "LOCK_UN")
    ) or not callable(getattr(fcntl, "flock", None)):
        raise SourceRootUnavailableError("flock is unavailable")
    if not _supports_dir_fd(os.open) or not _supports_dir_fd(os.stat):
        raise SourceRootUnavailableError("descriptor-relative open/stat is unavailable")
    if not _supports_dir_fd(os.unlink) or not _supports_dir_fd(os.mkdir):
        raise SourceRootUnavailableError("descriptor-relative cleanup is unavailable")
    # ``os.replace`` is an alias with a separate membership entry on some
    # Python builds; rename is the underlying dir-fd primitive it requires.
    if not _supports_dir_fd(os.rename):
        raise SourceRootUnavailableError("descriptor-relative replacement is unavailable")
    try:
        os.replace("source", "destination", src_dir_fd=-1, dst_dir_fd=-1)
    except (FileNotFoundError, NotADirectoryError, OSError):
        # The call reached the platform and failed only because the sentinels
        # are not real entries.  A TypeError means the keyword contract is
        # absent and must be treated as unsupported.
        pass
    except TypeError as exc:
        raise SourceRootUnavailableError(
            "descriptor-relative replacement is unavailable"
        ) from exc
    if not callable(getattr(os, "fsync", None)):
        raise SourceRootUnavailableError("directory fsync is unavailable")


def _identity_from_stat(value: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISDIR(value.st_mode):
        raise SourceRootUnavailableError("source root is not a directory")
    return value.st_dev, value.st_ino


class _DescriptorLock:
    """One flock-compatible descriptor lock with deterministic cleanup."""

    def __init__(self, fd: int, *, blocking: bool) -> None:
        self._fd = fd
        self._blocking = blocking
        self._held = False

    def acquire(self, *, blocking: bool) -> None:
        if self._held:
            raise SourceRootAuthorityError("descriptor lock is already held")
        if fcntl is None:
            raise SourceRootUnavailableError("flock is unavailable")
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(self._fd, flags)
        except OSError as exc:
            if not blocking and exc.errno in {
                errno.EACCES,
                errno.EAGAIN,
                getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
            }:
                raise BlockingIOError(errno.EAGAIN, "descriptor lock is busy") from None
            raise
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        try:
            if fcntl is None:
                raise SourceRootUnavailableError("flock is unavailable")
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            self._held = False
            os.close(self._fd)

    def __enter__(self) -> _DescriptorLock:
        try:
            self.acquire(blocking=self._blocking)
        except BaseException:
            os.close(self._fd)
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


class SourceRootAuthority:
    """Opaque process-local authority for one exact source directory.

    The descriptor and identity are private.  The capability is bound to the
    exact backend object and process, and cannot be copied or serialized.
    """

    def __init__(
        self,
        capability: object,
        backend: JsonlSessionStore,
        descriptor: int,
        identity: tuple[int, int],
    ) -> None:
        if capability is not _CAPABILITY:
            raise SourceRootAuthorityError("invalid source-root capability")
        self._backend = backend
        self._fd = descriptor
        self._identity = identity
        self._pid = os.getpid()
        self._closed = False
        self._closing = False
        self._active = 0

    @classmethod
    def issue(cls, backend: JsonlSessionStore) -> SourceRootAuthority:
        _require_platform()
        if type(backend) is not _backend_type():
            raise SourceRootAuthorityError("invalid source backend")
        path = backend.sessions_dir
        descriptor: int | None = None
        try:
            initial = path.lstat()
            if stat.S_ISLNK(initial.st_mode) or not stat.S_ISDIR(initial.st_mode):
                raise SourceRootUnavailableError("source root is not a regular directory")
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            cloexec = getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(path, flags | cloexec)
            opened = os.fstat(descriptor)
            identity = _identity_from_stat(opened)
            if (initial.st_dev, initial.st_ino) != identity:
                raise SourceRootUnavailableError("source root changed during pinning")
        except SourceRootAuthorityError:
            if descriptor is not None:
                os.close(descriptor)
            raise
        except (OSError, ValueError) as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise SourceRootUnavailableError("source root cannot be pinned") from exc
        return cls(_CAPABILITY, backend, descriptor, identity)

    @classmethod
    def issue_for_backend(cls, backend: JsonlSessionStore) -> SourceRootAuthority:
        """Internal compatibility factory for direct discovery callers."""

        return cls.issue(backend)

    @property
    def closed(self) -> bool:
        return self._closed

    def __reduce__(self) -> NoReturn:
        raise TypeError("source-root authority cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("source-root authority cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("source-root authority cannot be copied")

    def matches_backend(self, backend: JsonlSessionStore) -> bool:
        """Compare exact backend identity without returning or admitting that backend."""
        return self._backend is backend

    def open_entry_iterator(self) -> _Iterator[os.DirEntry[str]]:
        """Open a real directory iterator after reproving the existing capability checks."""
        self._require_backend(self._backend)
        return os.scandir(self._fd)

    def _require_backend(self, backend: JsonlSessionStore) -> None:
        if backend is not self._backend:
            raise SourceRootAuthorityError("source-root backend changed")
        if os.getpid() != self._pid:
            raise SourceRootAuthorityError("source-root authority crossed a process boundary")
        if self._closed or self._closing:
            raise SourceRootClosedError("source-root authority is closed")
        try:
            current = os.fstat(self._fd)
        except OSError as exc:
            raise SourceRootClosedError("source-root descriptor is unavailable") from exc
        if _identity_from_stat(current) != self._identity:
            raise SourceRootAuthorityError("source-root descriptor identity changed")

    def require_current(self, backend: JsonlSessionStore | None = None) -> None:
        """Validate capability identity and current pathname reachability."""

        actual = self._backend if backend is None else backend
        self._require_backend(actual)
        try:
            current = actual.sessions_dir.lstat()
        except OSError as exc:
            raise SourceRootAuthorityError("source-root pathname is unavailable") from exc
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
            raise SourceRootAuthorityError("source-root pathname is unavailable")
        if (current.st_dev, current.st_ino) != self._identity:
            raise SourceRootAuthorityError("source-root pathname identifies another directory")

    validate = require_current

    @contextmanager
    def operation(self, backend: JsonlSessionStore | None = None) -> Generator[SourceRootAuthority, None, None]:
        self.require_current(backend)
        self._active += 1
        try:
            yield self
        finally:
            self._active -= 1
            if self._closing and self._active == 0:
                self._close_descriptor()

    def _close_descriptor(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._fd)
        except OSError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        if self._active == 0:
            self._close_descriptor()

    def _open_relative(self, name: str, flags: int, *, mode: int = 0o600) -> int:
        self._require_backend(self._backend)
        if not isinstance(cast(object, name), str) or not name or name.startswith("/"):
            raise SourceRootAuthorityError("invalid relative source name")
        try:
            descriptor = os.open(
                name,
                flags | os.O_NOFOLLOW,
                mode,
                dir_fd=self._fd,
            )
        except TypeError as exc:
            raise SourceRootUnavailableError("descriptor-relative open is unavailable") from exc
        return descriptor

    def open_regular(self, name: str, *, flags: int = os.O_RDONLY) -> int:
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            from nanobot.session.mode_storage import require_namespace_write_access

            require_namespace_write_access(self._backend)
        descriptor = self._open_relative(name, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SourceRootAuthorityError("source entry is not a regular file")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def open_directory(self, name: str, *, create: bool = False) -> int:
        self._require_backend(self._backend)
        if create:
            from nanobot.session.mode_storage import require_namespace_write_access

            require_namespace_write_access(self._backend)
            try:
                os.mkdir(name, 0o700, dir_fd=self._fd)
            except FileExistsError:
                pass
        descriptor = self._open_relative(name, os.O_RDONLY | os.O_DIRECTORY)
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise SourceRootAuthorityError("source entry is not a directory")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def lock_file(self, name: str, *, blocking: bool) -> _DescriptorLock:
        from nanobot.session.mode_storage import require_namespace_write_access

        require_namespace_write_access(self._backend)
        descriptor = self._open_relative(name, os.O_RDWR | os.O_CREAT, mode=0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SourceRootAuthorityError("source lock is not a regular file")
            return _DescriptorLock(descriptor, blocking=blocking)
        except BaseException:
            os.close(descriptor)
            raise

    def lock_nested_file(self, directory: str, name: str, *, blocking: bool) -> _DescriptorLock:
        from nanobot.session.mode_storage import require_namespace_write_access

        require_namespace_write_access(self._backend)
        parent = self.open_directory(directory, create=True)
        try:
            descriptor = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
        finally:
            os.close(parent)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SourceRootAuthorityError("source lock is not a regular file")
            return _DescriptorLock(descriptor, blocking=blocking)
        except BaseException:
            os.close(descriptor)
            raise

    def stat_relative(self, name: str) -> os.stat_result | None:
        try:
            value = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except TypeError as exc:
            raise SourceRootUnavailableError("descriptor-relative stat is unavailable") from exc
        return value

    def unlink_relative(self, name: str) -> None:
        from nanobot.session.mode_storage import require_namespace_write_access

        require_namespace_write_access(self._backend)
        try:
            os.unlink(name, dir_fd=self._fd)
        except TypeError as exc:
            raise SourceRootUnavailableError("descriptor-relative cleanup is unavailable") from exc

    def fsync_directory(self) -> None:
        self._require_backend(self._backend)
        from nanobot.session.mode_storage import require_namespace_write_access

        require_namespace_write_access(self._backend)
        os.fsync(self._fd)

    def write_text_atomic(self, name: str, text: str, *, mode: int = 0o600) -> None:
        self._require_backend(self._backend)
        from nanobot.session.mode_storage import require_namespace_write_access

        require_namespace_write_access(self._backend)
        if not isinstance(cast(object, text), str) or not name or "/" in name:
            raise SourceRootAuthorityError("invalid source replacement")
        temporary = f".{name}.{os.urandom(8).hex()}.tmp"
        descriptor: int | None = None
        try:
            descriptor = self._open_relative(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                mode=mode,
            )
            payload = text.encode("utf-8")
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, name, src_dir_fd=self._fd, dst_dir_fd=self._fd)
            self.fsync_directory()
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=self._fd)
            except FileNotFoundError:
                pass
            except OSError:
                # The replacement already succeeded; preserve that result.
                if not self._closed:
                    raise


def _backend_type() -> type[JsonlSessionStore]:
    from nanobot.session.manager import JsonlSessionStore

    return JsonlSessionStore


_CAPABILITY = object()


__all__ = [
    "SourceRootAuthority",
    "SourceRootAuthorityError",
    "SourceRootClosedError",
    "SourceRootUnavailableError",
]
