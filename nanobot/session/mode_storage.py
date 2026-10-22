"""Private storage and capability primitives for durable session-mode authority."""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

if TYPE_CHECKING:
    from nanobot.session.manager import JsonlSessionStore

try:  # pragma: no cover - availability is platform dependent.
    import fcntl
except ImportError:  # pragma: no cover - exercised by platform qualification.
    fcntl = None  # type: ignore[assignment]


_ACCESS_TOKEN = object()
_WORKSPACE_ID_NAME = "workspace-id"
_WORKSPACE_ID_LENGTH = 32
MODE_JSON_LIMIT = 1_048_576
MODE_ENUM_LIMIT = 4096


class SessionModeError(RuntimeError):
    """A content-free session-mode storage boundary failure."""


ModeStorageError = SessionModeError


def _mode_error(message: str = "managed session namespace requires durable authority") -> SessionModeError:
    return SessionModeError(message)


def _hex32(value: str) -> bool:
    return len(value) == _WORKSPACE_ID_LENGTH and all(ch in "0123456789abcdef" for ch in value)


def _require_descriptor_platform() -> None:
    if any(not isinstance(getattr(os, name, None), int) for name in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise _mode_error("session mode storage is unavailable")
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
        raise _mode_error("session mode storage is unavailable")
    if os.mkdir not in os.supports_dir_fd or os.unlink not in os.supports_dir_fd:
        raise _mode_error("session mode storage is unavailable")


def _ensure_private_directory_path(path: Path) -> None:
    """Create missing absolute components only through pinned parent descriptors."""

    _require_descriptor_platform()
    descriptor = os.open(
        "/",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        for component in path.parts[1:]:
            parent_value = os.fstat(descriptor)
            parent_owner = parent_value.st_uid
            parent_mode = parent_value.st_mode
            parent_is_sticky_root = parent_owner == 0 and bool(parent_mode & stat.S_ISVTX)
            if (
                parent_owner not in {os.geteuid(), 0}
                or (parent_mode & 0o022 and not parent_is_sticky_root)
            ):
                raise _mode_error()
            created = False
            try:
                before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    created = True
                except FileExistsError:
                    pass
                before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            child = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            opened = os.fstat(child)
            if _directory_identity(opened) != _directory_identity(before):
                os.close(child)
                raise _mode_error()
            owner, mode = opened.st_uid, opened.st_mode
            sticky_root = owner == 0 and bool(mode & stat.S_ISVTX)
            if owner not in {os.geteuid(), 0} or (mode & 0o022 and not sticky_root):
                os.close(child)
                raise _mode_error()
            if created and (opened.st_uid != os.geteuid() or opened.st_mode & 0o777 != 0o700):
                os.close(child)
                raise _mode_error()
            os.close(descriptor)
            descriptor = child
    except (OSError, TypeError, NotImplementedError):
        raise _mode_error("session mode storage is unavailable") from None
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class _DirectoryBaseline:
    fd: int
    parent_fd: int | None
    name: str | None
    identity: tuple[int, int, int, int]


def _directory_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    if not stat.S_ISDIR(value.st_mode):
        raise _mode_error()
    return value.st_dev, value.st_ino, value.st_uid, value.st_mode


class PinnedDirectoryPath:
    """Retain every no-follow ancestor descriptor and parent edge."""

    def __init__(self, path: Path) -> None:
        _require_descriptor_platform()
        if not path.is_absolute():
            raise _mode_error()
        self.path = path
        self._pid = os.getpid()
        self._closed = False
        self._baselines: list[_DirectoryBaseline] = []
        try:
            root = os.open(
                "/",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            )
            self._baselines.append(
                _DirectoryBaseline(root, None, None, _directory_identity(os.fstat(root)))
            )
            parent = root
            for component in path.parts[1:]:
                before = os.stat(component, dir_fd=parent, follow_symlinks=False)
                child = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent,
                )
                identity = _directory_identity(os.fstat(child))
                if identity != _directory_identity(before):
                    os.close(child)
                    raise _mode_error()
                after = os.stat(component, dir_fd=parent, follow_symlinks=False)
                if _directory_identity(after) != identity:
                    os.close(child)
                    raise _mode_error()
                self._baselines.append(
                    _DirectoryBaseline(child, parent, component, identity)
                )
                parent = child
            self._validate_policy()
            self.validate()
        except BaseException:
            self.close()
            raise

    @property
    def fd(self) -> int:
        self.validate()
        return self._baselines[-1].fd

    def _validate_policy(self) -> None:
        effective_uid = os.geteuid()
        for index, baseline in enumerate(self._baselines):
            owner, mode = baseline.identity[2], baseline.identity[3]
            leaf = index == len(self._baselines) - 1
            sticky_root = owner == 0 and bool(mode & stat.S_ISVTX)
            if owner not in {effective_uid, 0}:
                raise _mode_error()
            if mode & 0o022 and not sticky_root:
                raise _mode_error()
            if leaf and owner != effective_uid:
                raise _mode_error()

    def validate(self) -> None:
        if self._closed or os.getpid() != self._pid or not self._baselines:
            raise _mode_error()
        for baseline in self._baselines:
            if _directory_identity(os.fstat(baseline.fd)) != baseline.identity:
                raise _mode_error()
            if baseline.parent_fd is not None and baseline.name is not None:
                edge = os.stat(
                    baseline.name,
                    dir_fd=baseline.parent_fd,
                    follow_symlinks=False,
                )
                if _directory_identity(edge) != baseline.identity:
                    raise _mode_error()
        current = self.path.stat(follow_symlinks=False)
        if _directory_identity(current) != self._baselines[-1].identity:
            raise _mode_error()

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for baseline in reversed(self._baselines):
            try:
                os.close(baseline.fd)
            except OSError:
                pass
        self._baselines.clear()


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise _mode_error() from None
    return True


def _read_private_text(path: Path, *, maximum: int = 4096) -> str:
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or opened.st_mode & 0o022
            or opened.st_size > maximum
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise _mode_error()
        raw = os.read(descriptor, maximum + 1)
        if len(raw) != opened.st_size:
            raise _mode_error()
        return raw.decode("utf-8")
    except (OSError, UnicodeError):
        raise _mode_error() from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def workspace_id_if_present(workspace: Path) -> str | None:
    marker = workspace / ".nanobot" / _WORKSPACE_ID_NAME
    if not _entry_exists(marker):
        return None
    value = _read_private_text(marker).strip()
    if not _hex32(value):
        raise _mode_error()
    return value


def namespace_claimed(workspace: Path, sessions_root: Path, workspace_id: str | None) -> bool:
    """Return true for active, preparing, partial, or residual claim evidence."""

    witness = workspace / ".nanobot" / "session-mode-binding.json"
    if _entry_exists(witness):
        return True
    if workspace_id is None:
        return False
    mode_root = sessions_root / workspace_id / ".session-mode"
    return _entry_exists(mode_root)


class ManagedNamespaceAccess:
    """Opaque, revocable process-local access to one leased session namespace."""

    __slots__ = (
        "_backend",
        "_identity",
        "_issuer",
        "_operator",
        "_pid",
        "_revoked",
        "_root",
        "_workspace",
    )

    def __init__(
        self,
        token: object,
        issuer: object,
        workspace: Path,
        root: Path,
        *,
        operator: bool,
    ) -> None:
        if token is not _ACCESS_TOKEN:
            raise _mode_error()
        self._issuer = issuer
        self._workspace = workspace
        self._root = root
        self._operator = operator
        self._pid = os.getpid()
        self._backend: JsonlSessionStore | None = None
        self._identity: tuple[int, int] | None = None
        self._revoked = False

    def __reduce__(self) -> NoReturn:
        raise TypeError("managed namespace access cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("managed namespace access cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("managed namespace access cannot be copied")

    @property
    def operator(self) -> bool:
        self._require_live()
        return self._operator

    def _require_live(self) -> None:
        if self._revoked or os.getpid() != self._pid:
            raise _mode_error()
        held = getattr(self._issuer, "access_is_live", None)
        if not callable(held) or held(self) is not True:
            raise _mode_error()
        validate = getattr(self._issuer, "validate_paths", None)
        if not callable(validate):
            raise _mode_error()
        validate()

    def preflight(self, workspace: Path, root: Path) -> None:
        self._require_live()
        if workspace != self._workspace or root != self._root:
            raise _mode_error()
        preflight = getattr(self._issuer, "preflight_access", None)
        if not callable(preflight):
            raise _mode_error()
        preflight(operator=self._operator)

    def bind(self, backend: JsonlSessionStore) -> None:
        self._require_live()
        if self._backend is not None:
            raise _mode_error()
        if backend.workspace != self._workspace or backend.sessions_root != self._root:
            raise _mode_error()
        try:
            value = backend.sessions_dir.stat(follow_symlinks=False)
        except OSError:
            raise _mode_error() from None
        if (
            not stat.S_ISDIR(value.st_mode)
            or value.st_uid != os.geteuid()
            or value.st_mode & 0o022
        ):
            raise _mode_error()
        self._backend = backend
        self._identity = (value.st_dev, value.st_ino)

    def require(self, backend: JsonlSessionStore) -> None:
        self._require_live()
        if self._backend is not backend or self._identity is None:
            raise _mode_error()
        try:
            current = backend.sessions_dir.stat(follow_symlinks=False)
        except OSError:
            raise _mode_error() from None
        if (current.st_dev, current.st_ino) != self._identity or not stat.S_ISDIR(current.st_mode):
            raise _mode_error()

    def revoke(self) -> None:
        self._revoked = True
        self._backend = None
        self._identity = None


class LifetimeWriterLease:
    """Persistent nonblocking cooperating-writer lease and access issuer."""

    def __init__(self, workspace: Path, sessions_root: Path) -> None:
        self.workspace = workspace
        self.sessions_root = sessions_root
        self._descriptor: int | None = None
        self._access: ManagedNamespaceAccess | None = None
        self._pid = os.getpid()
        self._workspace_pin: PinnedDirectoryPath | None = None
        self._root_pin: PinnedDirectoryPath | None = None

    def acquire(self) -> None:
        if fcntl is None or not callable(getattr(fcntl, "flock", None)):
            raise _mode_error("session authority lease is unavailable")
        if self._descriptor is not None:
            raise _mode_error()
        descriptor = -1
        try:
            self._workspace_pin = PinnedDirectoryPath(self.workspace)
            _ensure_private_directory_path(self.sessions_root)
            self._root_pin = PinnedDirectoryPath(self.sessions_root)
            identity = hashlib.sha256(os.fsencode(self.workspace)).hexdigest()
            name = f".session-writer-{identity}.lock"
            descriptor = os.open(
                name,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=self._root_pin.fd,
            )
            opened = os.fstat(descriptor)
            named = os.stat(name, dir_fd=self._root_pin.fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
                or opened.st_mode & 0o777 != 0o600
                or opened.st_size != 0
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise _mode_error("session authority lease is unsafe")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.validate_paths()
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            if self._workspace_pin is not None:
                self._workspace_pin.close()
                self._workspace_pin = None
            if self._root_pin is not None:
                self._root_pin.close()
                self._root_pin = None
            if exc.errno in {errno.EACCES, errno.EAGAIN, getattr(errno, "EWOULDBLOCK", errno.EAGAIN)}:
                raise _mode_error("session authority is busy") from None
            raise _mode_error("session authority lease is unavailable") from None
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            if self._workspace_pin is not None:
                self._workspace_pin.close()
                self._workspace_pin = None
            if self._root_pin is not None:
                self._root_pin.close()
                self._root_pin = None
            raise
        self._descriptor = descriptor

    def validate_paths(self) -> None:
        if self._workspace_pin is None or self._root_pin is None:
            raise _mode_error()
        self._workspace_pin.validate()
        self._root_pin.validate()

    @staticmethod
    def _relative_stat(pin: PinnedDirectoryPath, parts: tuple[str, ...]) -> os.stat_result | None:
        pin.validate()
        descriptor = os.dup(pin.fd)
        try:
            for component in parts[:-1]:
                try:
                    child = os.open(
                        component,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=descriptor,
                    )
                except FileNotFoundError:
                    return None
                os.close(descriptor)
                descriptor = child
            try:
                return os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return None
        finally:
            os.close(descriptor)

    @staticmethod
    def _relative_read(pin: PinnedDirectoryPath, parts: tuple[str, ...]) -> str | None:
        if LifetimeWriterLease._relative_stat(pin, parts) is None:
            return None
        descriptor = os.dup(pin.fd)
        file_fd = -1
        try:
            for component in parts[:-1]:
                child = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
            file_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            value = os.fstat(file_fd)
            named = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(value.st_mode)
                or value.st_uid != os.geteuid()
                or value.st_nlink != 1
                or value.st_mode & 0o022
                or value.st_size > 4096
                or (value.st_dev, value.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise _mode_error()
            raw = os.read(file_fd, 4097)
            if len(raw) != value.st_size:
                raise _mode_error()
            pin.validate()
            return raw.decode("utf-8")
        except (OSError, UnicodeError):
            raise _mode_error() from None
        finally:
            if file_fd >= 0:
                os.close(file_fd)
            os.close(descriptor)

    @staticmethod
    def _relative_replace_private(
        pin: PinnedDirectoryPath,
        parts: tuple[str, ...],
        data: bytes,
    ) -> None:
        pin.validate()
        descriptor = os.dup(pin.fd)
        temporary = f".{parts[-1]}.{secrets.token_hex(8)}.tmp"
        file_fd = -1
        try:
            for component in parts[:-1]:
                child = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
            file_fd = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=descriptor,
            )
            os.fchmod(file_fd, 0o600)
            view = memoryview(data)
            while view:
                written = os.write(file_fd, view)
                if written <= 0:
                    raise OSError
                view = view[written:]
            os.fsync(file_fd)
            os.close(file_fd)
            file_fd = -1
            pin.validate()
            os.replace(temporary, parts[-1], src_dir_fd=descriptor, dst_dir_fd=descriptor)
            os.fsync(descriptor)
            pin.validate()
        except (OSError, TypeError, NotImplementedError):
            raise _mode_error() from None
        finally:
            if file_fd >= 0:
                os.close(file_fd)
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            os.close(descriptor)

    @staticmethod
    def _relative_directory_nonempty(
        pin: PinnedDirectoryPath,
        parts: tuple[str, ...],
        *,
        namespace_scaffolding: bool = False,
    ) -> bool:
        value = LifetimeWriterLease._relative_stat(pin, parts)
        if value is None:
            return False
        if not stat.S_ISDIR(value.st_mode):
            return True
        descriptor = os.dup(pin.fd)
        try:
            for component in parts:
                child = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
            names = os.listdir(descriptor)
            if not namespace_scaffolding:
                return bool(names)
            allowed = {".workspace", ".session-files.lock", ".execution-locks"}
            return any(name not in allowed for name in names)
        except OSError:
            raise _mode_error() from None
        finally:
            os.close(descriptor)

    def preflight_access(self, *, operator: bool) -> None:
        self.validate_paths()
        assert self._workspace_pin is not None and self._root_pin is not None
        if self._relative_directory_nonempty(self._workspace_pin, ("sessions",)):
            raise _mode_error("session namespace requires explicit offline bootstrap")
        marker = self._relative_read(self._workspace_pin, (".nanobot", "workspace-id"))
        if marker is None:
            return
        workspace_id = marker.strip()
        if not _hex32(workspace_id):
            raise _mode_error()
        witness = self._relative_stat(
            self._workspace_pin, (".nanobot", "session-mode-binding.json")
        )
        residue = self._relative_stat(
            self._root_pin, (workspace_id, ".session-mode")
        )
        if not operator and witness is None and residue is None and self._relative_directory_nonempty(
            self._root_pin,
            (workspace_id,),
            namespace_scaffolding=True,
        ):
            raise _mode_error("session namespace requires explicit offline bootstrap")

    def rebind_copied_workspace_claim(self) -> None:
        """Rebind only an explicitly supplied copied namespace through pinned descriptors."""

        self.validate_paths()
        assert self._workspace_pin is not None and self._root_pin is not None
        marker = self._relative_read(self._workspace_pin, (".nanobot", "workspace-id"))
        if marker is None:
            return
        workspace_id = marker.strip()
        if not _hex32(workspace_id):
            raise _mode_error()
        claim_parts = (workspace_id, ".workspace")
        claim = self._relative_read(self._root_pin, claim_parts)
        if claim is None:
            return
        try:
            recorded = Path(claim.strip()).expanduser().resolve(strict=False)
        except (OSError, RuntimeError):
            raise _mode_error() from None
        if recorded == self.workspace:
            return
        self._relative_replace_private(
            self._root_pin,
            claim_parts,
            f"{self.workspace}\n".encode("utf-8"),
        )

    def issue_access(self, *, operator: bool) -> ManagedNamespaceAccess:
        if self._descriptor is None or self._access is not None or os.getpid() != self._pid:
            raise _mode_error()
        access = ManagedNamespaceAccess(
            _ACCESS_TOKEN,
            self,
            self.workspace,
            self.sessions_root,
            operator=operator,
        )
        self._access = access
        return access

    def access_is_live(self, access: ManagedNamespaceAccess) -> bool:
        return (
            access is self._access
            and self._descriptor is not None
            and os.getpid() == self._pid
        )

    def close(self) -> None:
        access, self._access = self._access, None
        if access is not None:
            access.revoke()
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        if self._workspace_pin is not None:
            self._workspace_pin.close()
            self._workspace_pin = None
        if self._root_pin is not None:
            self._root_pin.close()
            self._root_pin = None

    @staticmethod
    def validate_store_construction(
        workspace: Path,
        sessions_root: Path,
        access: ManagedNamespaceAccess | None,
    ) -> None:
        """Fence constructor mutation using this owner's private relative checks."""
        if access is not None:
            access.preflight(workspace, sessions_root)
            return
        if not workspace.exists():
            return
        workspace_pin = PinnedDirectoryPath(workspace)
        root_pin: PinnedDirectoryPath | None = None
        try:
            witness = LifetimeWriterLease._relative_stat(
                workspace_pin, (".nanobot", "session-mode-binding.json")
            )
            if witness is not None:
                raise _mode_error()
            marker = LifetimeWriterLease._relative_read(
                workspace_pin, (".nanobot", "workspace-id")
            )
            if marker is None or not sessions_root.exists():
                return
            workspace_id = marker.strip()
            if not _hex32(workspace_id):
                raise _mode_error()
            root_pin = PinnedDirectoryPath(sessions_root)
            if LifetimeWriterLease._relative_stat(root_pin, (workspace_id, ".session-mode")) is not None:
                raise _mode_error()
        finally:
            workspace_pin.close()
            if root_pin is not None:
                root_pin.close()


def preflight_store_construction(
    workspace: Path,
    sessions_root: Path,
    access: ManagedNamespaceAccess | None,
) -> None:
    """Fence constructor mutation before workspace identity or namespace claims."""
    LifetimeWriterLease.validate_store_construction(workspace, sessions_root, access)


def require_namespace_write_access(backend: JsonlSessionStore) -> None:
    """Reject every unprivileged write when any registry evidence claims the namespace."""
    from nanobot.session.manager import JsonlSessionStore

    JsonlSessionStore.require_namespace_write_admission(
        backend, select_namespace_claimed=lambda: namespace_claimed,
        mode_error=lambda: _mode_error(),
    )


class SessionModeStorage:
    """Pinned descriptor-relative storage for all session-mode evidence."""

    def __init__(self, backend: JsonlSessionStore) -> None:
        required = ("O_NOFOLLOW", "O_DIRECTORY")
        if any(not isinstance(getattr(os, name, None), int) for name in required):
            raise _mode_error("session mode storage is unavailable")
        if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
            raise _mode_error("session mode storage is unavailable")
        self._backend = backend
        self._pid = os.getpid()
        self._closed = False
        self._namespace_pin = PinnedDirectoryPath(backend.sessions_dir)
        self._workspace_pin = PinnedDirectoryPath(backend.workspace / ".nanobot")
        self._namespace_fd = self._namespace_pin.fd
        self._workspace_fd = self._workspace_pin.fd
        self._namespace_identity = self._identity(self._namespace_fd)
        self._workspace_identity = self._identity(self._workspace_fd)
        self._directory_identities: dict[tuple[bool, tuple[str, ...]], tuple[int, int]] = {}

    @staticmethod
    def _identity(descriptor: int) -> tuple[int, int]:
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.geteuid() or value.st_mode & 0o022:
            raise _mode_error()
        return value.st_dev, value.st_ino

    @staticmethod
    def _open_absolute_directory(path: Path, *, private: bool) -> int:
        descriptor: int | None = None
        try:
            before = path.stat(follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise _mode_error()
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            )
            opened = os.fstat(descriptor)
            if (
                (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or opened.st_uid != os.geteuid()
                or opened.st_mode & 0o022
                or (private and opened.st_mode & 0o777 != 0o700)
            ):
                raise _mode_error()
            return descriptor
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._namespace_pin.close()
        self._workspace_pin.close()

    def _require_current(self) -> None:
        if self._closed or os.getpid() != self._pid:
            raise _mode_error()
        self._namespace_pin.validate()
        self._workspace_pin.validate()
        if self._identity(self._namespace_fd) != self._namespace_identity:
            raise _mode_error()
        if self._identity(self._workspace_fd) != self._workspace_identity:
            raise _mode_error()
        for path, expected in (
            (self._backend.sessions_dir, self._namespace_identity),
            (self._backend.workspace / ".nanobot", self._workspace_identity),
        ):
            try:
                current = path.stat(follow_symlinks=False)
            except OSError:
                raise _mode_error() from None
            if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != expected:
                raise _mode_error()

    @staticmethod
    def _parts(name: str) -> tuple[str, ...]:
        if not isinstance(cast(object, name), str) or not name or name.startswith("/"):
            raise _mode_error()
        parts = tuple(name.split("/"))
        if any(not part or part in {".", ".."} or len(part.encode()) > 255 for part in parts):
            raise _mode_error()
        return parts

    def _parent(
        self,
        name: str,
        *,
        witness: bool = False,
        create: bool = False,
    ) -> tuple[int, str]:
        self._require_current()
        parts = self._parts(name)
        descriptor = os.dup(self._workspace_fd if witness else self._namespace_fd)
        try:
            traversed: list[str] = []
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=descriptor)
                        os.fsync(descriptor)
                    except FileExistsError:
                        pass
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                value = os.fstat(child)
                if (
                    not stat.S_ISDIR(value.st_mode)
                    or value.st_uid != os.geteuid()
                    or value.st_mode & 0o777 != 0o700
                ):
                    os.close(child)
                    raise _mode_error()
                traversed.append(part)
                identity = (value.st_dev, value.st_ino)
                identity_key = (witness, tuple(traversed))
                known = self._directory_identities.setdefault(identity_key, identity)
                if known != identity:
                    os.close(child)
                    raise _mode_error()
                os.close(descriptor)
                descriptor = child
            return descriptor, parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    def _verify_parent(self, name: str, descriptor: int, *, witness: bool = False) -> None:
        current, _ = self._parent(name, witness=witness)
        try:
            if self._identity(current) != self._identity(descriptor):
                raise _mode_error()
        finally:
            os.close(current)

    def exists(self, name: str, *, witness: bool = False) -> bool:
        try:
            parent, leaf = self._parent(name, witness=witness)
        except FileNotFoundError:
            return False
        try:
            try:
                os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return False
            return True
        finally:
            os.close(parent)

    def read(self, name: str, *, witness: bool = False, limit: int = MODE_JSON_LIMIT) -> bytes:
        parent, leaf = self._parent(name, witness=witness)
        descriptor = -1
        try:
            before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            descriptor = os.open(
                leaf,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
                or opened.st_mode & 0o777 != 0o600
                or opened.st_size > limit
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise _mode_error()
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65536))
                if not chunk:
                    raise _mode_error()
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if (after.st_dev, after.st_ino, after.st_size) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ):
                raise _mode_error()
            self._verify_parent(name, parent, witness=witness)
            return b"".join(chunks)
        except OSError:
            raise _mode_error() from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def write(
        self,
        name: str,
        data: bytes,
        *,
        witness: bool = False,
        replace: bool = True,
        limit: int = MODE_JSON_LIMIT,
    ) -> None:
        if not isinstance(cast(object, data), bytes) or len(data) > limit:
            raise _mode_error()
        parent, leaf = self._parent(name, witness=witness, create=True)
        temporary = f".{leaf}.{secrets.token_hex(8)}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent,
            )
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
            value = os.fstat(descriptor)
            named = os.stat(temporary, dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISREG(value.st_mode)
                or value.st_uid != os.geteuid()
                or value.st_nlink != 1
                or value.st_mode & 0o777 != 0o600
                or (value.st_dev, value.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise _mode_error()
            if replace:
                self._verify_parent(name, parent, witness=witness)
                os.replace(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            else:
                try:
                    os.link(
                        temporary,
                        leaf,
                        src_dir_fd=parent,
                        dst_dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    raise _mode_error() from None
                os.unlink(temporary, dir_fd=parent)
            os.fsync(parent)
            self._verify_parent(name, parent, witness=witness)
        except OSError:
            raise _mode_error() from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            os.close(parent)

    def unlink(self, name: str) -> None:
        parent, leaf = self._parent(name)
        try:
            os.unlink(leaf, dir_fd=parent)
            os.fsync(parent)
        except FileNotFoundError:
            return
        except OSError:
            raise _mode_error() from None
        finally:
            os.close(parent)

    def list(self, name: str, *, suffix: str | None = None) -> tuple[str, ...]:
        try:
            parent, leaf = self._parent(name)
        except FileNotFoundError:
            return ()
        directory = -1
        try:
            directory = os.open(
                leaf,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            value = os.fstat(directory)
            if value.st_uid != os.geteuid() or value.st_mode & 0o777 != 0o700:
                raise _mode_error()
            names = os.listdir(directory)
            if len(names) > MODE_ENUM_LIMIT:
                raise _mode_error()
            if suffix is not None:
                if any(not item.endswith(suffix) for item in names):
                    raise _mode_error()
            return tuple(sorted(names))
        except FileNotFoundError:
            return ()
        except OSError:
            raise _mode_error() from None
        finally:
            if directory >= 0:
                os.close(directory)
            os.close(parent)

    def list_namespace(self) -> tuple[str, ...]:
        self._require_current()
        descriptor = -1
        try:
            descriptor = os.open(
                ".",
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self._namespace_fd,
            )
            if self._identity(descriptor) != self._namespace_identity:
                raise _mode_error()
            names = os.listdir(descriptor)
            if len(names) > MODE_ENUM_LIMIT:
                raise _mode_error()
            self._require_current()
            return tuple(sorted(names))
        except OSError:
            raise _mode_error() from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def install_session_file(self, staged_name: str, target_name: str) -> None:
        source_parent, source_leaf = self._parent(staged_name)
        try:
            self._require_current()
            self._verify_parent(staged_name, source_parent)
            os.replace(
                source_leaf,
                target_name,
                src_dir_fd=source_parent,
                dst_dir_fd=self._namespace_fd,
            )
            os.fsync(self._namespace_fd)
            self._require_current()
        except OSError:
            raise _mode_error() from None
        finally:
            os.close(source_parent)



__all__ = [
    "LifetimeWriterLease",
    "ManagedNamespaceAccess",
    "ModeStorageError",
    "SessionModeError",
    "SessionModeStorage",
    "namespace_claimed",
    "preflight_store_construction",
    "require_namespace_write_access",
    "workspace_id_if_present",
]
