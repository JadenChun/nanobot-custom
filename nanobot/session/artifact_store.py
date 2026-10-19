"""Private immutable artifact bytes; no descriptor publication or read grants.

POSIX directory-relative no-follow operations are required. Unsupported platforms
fail closed until an equivalent implementation is qualified. Transactions retain
an advisory store lock; callers acquire source-session transactions after it.
"""

from __future__ import annotations

import codecs
import errno
import hashlib
import os
import re
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Iterable, Literal, cast

from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    parse_artifact_descriptor,
    parse_artifact_reference,
)

MAX_RANGE_BYTES = 16_384
DEFAULT_MAX_BLOB_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_NAMESPACE_BYTES = 1024 * 1024 * 1024
MAX_INSPECTION_ENTRIES = 10_000
_CAPABILITY = object()


class ArtifactStorageError(OSError):
    """Content-free storage failure; no reference may be published on failure."""


class ArtifactStorageBusyError(ArtifactStorageError):
    """The private artifact lock is held by another transaction."""


class ArtifactInstallError(ArtifactStorageError):
    """Content-free failure while installing one missing immutable blob."""


class ArtifactInstallCorruptionError(ArtifactInstallError):
    """An existing blob or crash residue does not match its descriptor."""


class ArtifactInstallLimitError(ArtifactInstallError):
    """A missing blob cannot be installed within the configured limits."""


class ArtifactInstallIOError(ArtifactInstallError):
    """A missing blob was not installed because storage I/O failed."""


class ArtifactInstallUncertainError(ArtifactInstallError):
    """Installation reached the immutable name but durability is uncertain."""


@dataclass(frozen=True)
class ArtifactBlobInstallResult:
    status: Literal['installed', 'already_present']


@dataclass(frozen=True)
class ArtifactBlobInspection:
    """Content-free aggregate for one exact artifact namespace."""

    reachable_blobs: int
    reachable_bytes: int
    unreachable_blobs: int
    unreachable_bytes: int
    missing_blobs: int
    missing_bytes: int


def _private(info: os.stat_result, *, directory: bool, links: int = 1) -> None:
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (not expected(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or (not directory and info.st_nlink != links)):
        raise ArtifactStorageError('Artifact storage object is not private or has an unexpected type')


def _flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK


class PrivateArtifactStore:
    """A private namespace selected by trusted runtime state, never model input."""

    def __init__(self, root: Path, *, workspace_id: str, session_key: str,
                 incarnation: str, generation: int,
                 max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
                 max_namespace_bytes: int = DEFAULT_MAX_NAMESPACE_BYTES,
                 location_validator: Callable[[], None] | None = None):
        if (os.name != 'posix' or not all(hasattr(os, flag) for flag in ('O_DIRECTORY', 'O_NOFOLLOW'))
                or not {os.open, os.mkdir, os.stat, os.unlink, os.link} <= os.supports_dir_fd
                or os.link not in os.supports_follow_symlinks):
            raise ArtifactStorageError('Safe artifact filesystem operations are unavailable')
        root = Path(root)
        if not root.is_absolute() or root == Path(root.anchor) or '..' in root.parts:
            raise ArtifactValidationError('Artifact root must be an absolute non-root directory')
        parse_artifact_reference('artifact:' + workspace_id)
        if (not isinstance(cast(object, session_key), str) or not session_key.strip()
                or not isinstance(cast(object, incarnation), str) or not incarnation.strip()
                or type(generation) is not int or generation < 0
                or type(max_blob_bytes) is not int or max_blob_bytes <= 0
                or type(max_namespace_bytes) is not int or max_namespace_bytes < max_blob_bytes):
            raise ArtifactValidationError('Invalid artifact store namespace or capacity')
        self.root = root
        self.workspace_id, self.session_key = workspace_id, session_key
        self.incarnation, self.generation = incarnation, generation
        self.max_blob_bytes = max_blob_bytes
        self.max_namespace_bytes = max_namespace_bytes
        self._location_validator = location_validator
        # Hash only filename components. The persisted workspace UUID remains
        # the workspace identity; full source bindings stay in the descriptor.
        self._parts = (workspace_id, hashlib.sha256(session_key.encode()).hexdigest(),
                       hashlib.sha256(incarnation.encode()).hexdigest(), str(generation))
        self._root_identity: tuple[int, int] | None = None
        with self._root_fd(create=True) as fd:
            info = os.fstat(fd)
            self._root_identity = (info.st_dev, info.st_ino)

    @contextmanager
    def _root_fd(self, *, create: bool = False) -> Generator[int, None, None]:
        """Walk from the filesystem anchor without resolving or following links."""
        fd = None
        try:
            if self._location_validator is not None:
                self._location_validator()
            fd = os.open(self.root.anchor, _flags())
            parts = self.root.parts[1:]
            for index, part in enumerate(parts):
                if create and index == len(parts) - 1:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    os.fsync(fd)
                child = os.open(part, _flags(), dir_fd=fd)
                os.close(fd)
                fd = child
            info = os.fstat(fd)
            _private(info, directory=True)
            if self._root_identity is not None and self._root_identity != (info.st_dev, info.st_ino):
                raise ArtifactStorageError('Artifact root identity changed')
            yield fd
        except ArtifactStorageError:
            raise
        except OSError:
            raise ArtifactStorageError('Private artifact root is unavailable') from None
        finally:
            if fd is not None:
                os.close(fd)

    @contextmanager
    def transaction(self, *, blocking: bool = True) -> Generator[ArtifactBlobTransaction, None, None]:
        """Hold the store lock before a caller's source-session CAS transaction.

        Existing callers wait for the lock by default. A caller that is
        already a best-effort admission path can pass ``blocking=False`` to
        fail immediately when another process owns the private namespace.
        """
        import fcntl

        if type(blocking) is not bool:
            raise ArtifactStorageError('Artifact transaction blocking mode must be boolean')
        opened: list[int] = []
        lock_fd = None
        transaction = None
        try:
            with self._root_fd() as root_fd:
                parent = root_fd
                for part in self._parts:
                    try:
                        os.mkdir(part, 0o700, dir_fd=parent)
                    except FileExistsError:
                        pass
                    os.fsync(parent)
                    child = os.open(part, _flags(), dir_fd=parent)
                    opened.append(child)
                    _private(os.fstat(child), directory=True)
                    parent = child
                # macOS may report ENOENT for concurrent O_CREAT|O_NOFOLLOW
                # opens. Separate lookup from atomic exclusive creation.
                for _ in range(8):
                    try:
                        lock_fd = os.open('.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                        break
                    except FileNotFoundError:
                        try:
                            lock_fd = os.open('.lock', os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                                              | os.O_NONBLOCK, 0o600, dir_fd=parent)
                            break
                        except FileExistsError:
                            continue
                if lock_fd is None:
                    raise ArtifactStorageError('Artifact lock could not be established')
                _private(os.fstat(lock_fd), directory=False)
                os.fsync(lock_fd)
                os.fsync(parent)
                lock_mode = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
                try:
                    fcntl.flock(lock_fd, lock_mode)
                except OSError as exc:
                    if not blocking and exc.errno in (errno.EACCES, errno.EAGAIN):
                        raise ArtifactStorageBusyError('Private artifact transaction is busy') from None
                    raise
                transaction = ArtifactBlobTransaction(_CAPABILITY, self, parent, lock_fd)
                yield transaction
        except ArtifactStorageError:
            raise
        except OSError:
            raise ArtifactStorageError('Private artifact transaction failed') from None
        finally:
            if transaction is not None:
                ArtifactBlobTransaction.end(transaction)
            if lock_fd is not None:
                os.close(lock_fd)
            for fd in reversed(opened):
                os.close(fd)

    def assert_namespace_binding(
        self, *, select_store: Callable[[], PrivateArtifactStore],
        binding_changed: Callable[[Callable[[], os.stat_result], os.stat_result], bool],
    ) -> None:
        """Reject a changed namespace; never expose descriptors or admit blob access.

        The transaction selects its store again after the root hook, and compares
        live identities after its inode hook. Lazy stat results keep that order
        without moving raw directory descriptors across the ownership boundary.
        """
        opened: list[int] = []
        try:
            with self._root_fd() as root:
                parent = root
                for part in select_store()._parts:
                    child = os.open(part, _flags(), dir_fd=parent)
                    opened.append(child)
                    _private(os.fstat(child), directory=True)
                    parent = child
                lock = os.stat('.lock', dir_fd=parent, follow_symlinks=False)
                _private(lock, directory=False)
                if binding_changed(lambda: os.fstat(parent), lock):
                    raise ArtifactStorageError('Artifact namespace or lock binding changed')
        finally:
            for fd in reversed(opened):
                os.close(fd)


class ArtifactBlobTransaction:
    """Internal integrity-checked byte operations; callers still need read policy."""

    def __init__(self, capability: object, store: PrivateArtifactStore, directory_fd: int, lock_fd: int):
        if capability is not _CAPABILITY:
            raise ArtifactStorageError('Artifact transaction requires a store capability')
        self._store, self._fd, self._active = store, directory_fd, True
        self._directory_identity = self._inode(os.fstat(directory_fd))
        self._lock_identity = self._inode(os.fstat(lock_fd))

    @staticmethod
    def _inode(info: os.stat_result) -> tuple[int, int]:
        return info.st_dev, info.st_ino

    def matches_store(self, store: PrivateArtifactStore | None) -> bool:
        """Observe exact configured identity only, without admitting a byte operation."""
        return self._store is store

    def end(self) -> None:
        """Irreversibly revoke this transaction before its store releases the lock."""
        self._active = False

    def assert_reachable(self) -> None:
        """Recheck namespace/lock binding immediately before descriptor publication."""
        if not self._active:
            raise ArtifactStorageError('Artifact transaction has ended')
        PrivateArtifactStore.assert_namespace_binding(
            self._store, select_store=lambda: self._store,
            binding_changed=lambda directory, lock: (
                self._inode(directory()) != self._directory_identity
                or self._inode(lock) != self._lock_identity
            ),
        )

    def _descriptor(self, descriptor: ArtifactDescriptor) -> ArtifactDescriptor:
        if not self._active:
            raise ArtifactStorageError('Artifact transaction has ended')
        self.assert_reachable()
        descriptor = parse_artifact_descriptor(descriptor.to_dict())
        store = self._store
        if ((descriptor.workspace_id, descriptor.session_key, descriptor.incarnation, descriptor.generation)
                != (store.workspace_id, store.session_key, store.incarnation, store.generation)):
            raise ArtifactValidationError('Artifact belongs to another store namespace')
        return descriptor

    def verify_durable(self, descriptor: ArtifactDescriptor, *, store: PrivateArtifactStore) -> None:
        """Verify bytes under the exact runtime-configured store binding."""
        if self._store is not store:
            raise ArtifactStorageError('Artifact transaction uses another configured store')
        self._verified_range(descriptor, offset=0, limit=0, durable=True)

    def _namespace_bytes(self) -> int:
        """Count only trusted blob/staging entries while the store lock is held."""
        if not self._active:
            raise ArtifactStorageError('Artifact transaction has ended')
        total = 0
        for name in os.listdir(self._fd):
            info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
            if name == '.lock':
                _private(info, directory=False)
                continue
            if re.fullmatch(r'[0-9a-f]{32}\.blob', name):
                _private(info, directory=False)
            elif re.fullmatch(r'\.tmp-[0-9a-f]{32}', name):
                _private(info, directory=False, links=info.st_nlink)
                if info.st_nlink not in {1, 2}:
                    raise ArtifactStorageError('Artifact staging entry has unexpected links')
            else:
                raise ArtifactStorageError('Artifact namespace has an unexpected entry')
            total += info.st_size
        return total

    def inspect(self, descriptors: Iterable[ArtifactDescriptor]) -> ArtifactBlobInspection:
        """Classify immutable blobs without returning names, references, paths, or content."""
        expected: dict[str, ArtifactDescriptor] = {}
        for index, candidate in enumerate(descriptors):
            if index >= MAX_INSPECTION_ENTRIES:
                raise ArtifactStorageError('Artifact inspection exceeds configured bounds')
            descriptor = self._descriptor(candidate)
            if descriptor.artifact_id in expected:
                raise ArtifactStorageError('Artifact inspection evidence is invalid')
            expected[descriptor.artifact_id] = descriptor

        reachable_blobs = reachable_bytes = 0
        unreachable_blobs = unreachable_bytes = 0
        present: set[str] = set()
        try:
            for index, name in enumerate(os.listdir(self._fd)):
                if index >= MAX_INSPECTION_ENTRIES:
                    raise ArtifactStorageError('Artifact inspection exceeds configured bounds')
                info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
                if name == '.lock':
                    _private(info, directory=False)
                    continue
                match = re.fullmatch(r'([0-9a-f]{32})\.blob', name)
                if match is None:
                    raise ArtifactStorageError('Artifact namespace has an unexpected entry')
                _private(info, directory=False)
                identity = match.group(1)
                descriptor = expected.get(identity)
                if descriptor is None:
                    unreachable_blobs += 1
                    unreachable_bytes += info.st_size
                    continue
                self._verified_range(descriptor, offset=0, limit=0, durable=True)
                present.add(identity)
                reachable_blobs += 1
                reachable_bytes += info.st_size
        except ArtifactStorageError:
            raise
        except OSError:
            raise ArtifactStorageError('Artifact inspection failed') from None
        missing = [descriptor for identity, descriptor in expected.items() if identity not in present]
        return ArtifactBlobInspection(
            reachable_blobs,
            reachable_bytes,
            unreachable_blobs,
            unreachable_bytes,
            len(missing),
            sum(descriptor.byte_length for descriptor in missing),
        )

    def put(self, descriptor: ArtifactDescriptor, content: bytes) -> None:
        """Install complete immutable bytes, or verify an identical existing blob."""
        descriptor = self._descriptor(descriptor)
        if (not isinstance(cast(object, content), bytes) or len(content) != descriptor.byte_length
                or hashlib.sha256(content).hexdigest() != descriptor.sha256):
            raise ArtifactValidationError('Artifact bytes do not match descriptor integrity')
        try:
            content.decode('utf-8')
        except UnicodeError:
            raise ArtifactValidationError('Artifact bytes are not UTF-8 text') from None
        name = descriptor.artifact_id + '.blob'
        try:
            os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            self._recover_linked_install(descriptor)
            # A crashed install may retain the one permitted staging hard link.
            # Recover it before quota accounting, which intentionally rejects
            # all remaining staging entries as untrusted retained usage.
            self._namespace_bytes()
            self._verified_range(descriptor, offset=0, limit=0, durable=True)
            return
        if descriptor.byte_length > self._store.max_blob_bytes:
            raise ArtifactStorageError('Artifact exceeds configured blob capacity')
        if self._namespace_bytes() + len(content) > self._store.max_namespace_bytes:
            raise ArtifactStorageError('Artifact namespace quota exceeded')
        temporary = '.tmp-' + secrets.token_hex(16)
        fd = None
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
                         0o600, dir_fd=self._fd)
            _private(os.fstat(fd), directory=False)
            remaining = memoryview(content)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise ArtifactStorageError('Artifact write did not advance')
                remaining = remaining[written:]
            os.fsync(fd)
            try:
                os.link(temporary, name, src_dir_fd=self._fd, dst_dir_fd=self._fd, follow_symlinks=False)
            except FileExistsError:
                # A separately installed blob must match exactly; never overwrite.
                self._verified_range(descriptor, offset=0, limit=0, durable=True)
        finally:
            if fd is not None:
                os.close(fd)
                os.unlink(temporary, dir_fd=self._fd)
        os.fsync(self._fd)
        self.assert_reachable()

    def install_missing(
        self, descriptor: ArtifactDescriptor, content: bytes
    ) -> ArtifactBlobInstallResult:
        """Install only absent content and report immutable reuse distinctly.

        This is the narrow recovery seam.  It never overwrites, removes, or
        quarantines an existing blob.  Failures after the immutable name is
        linked are reported as uncertain so an exact retry can verify/reuse it.
        """
        descriptor = self._descriptor(descriptor)
        if (not isinstance(cast(object, content), bytes) or len(content) != descriptor.byte_length
                or hashlib.sha256(content).hexdigest() != descriptor.sha256):
            raise ArtifactValidationError('Artifact bytes do not match descriptor integrity')
        try:
            content.decode('utf-8')
        except UnicodeError:
            raise ArtifactValidationError('Artifact bytes are not UTF-8 text') from None
        name = descriptor.artifact_id + '.blob'
        try:
            os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError:
            raise ArtifactInstallIOError('Artifact installation I/O failed') from None
        else:
            try:
                self._recover_linked_install(descriptor)
                self._namespace_bytes()
                self._verified_range(descriptor, offset=0, limit=0, durable=True)
            except ArtifactStorageError:
                raise ArtifactInstallCorruptionError('Existing artifact is corrupt') from None
            except OSError:
                raise ArtifactInstallUncertainError(
                    'Artifact installation durability is uncertain'
                ) from None
            return ArtifactBlobInstallResult('already_present')

        if descriptor.byte_length > self._store.max_blob_bytes:
            raise ArtifactInstallLimitError('Artifact installation exceeds configured limits')
        recovered = self._recover_unlinked_install(descriptor, content)
        if recovered:
            return ArtifactBlobInstallResult('installed')
        try:
            if self._namespace_bytes() + len(content) > self._store.max_namespace_bytes:
                raise ArtifactInstallLimitError('Artifact installation exceeds configured limits')
        except ArtifactInstallLimitError:
            raise
        except (ArtifactStorageError, OSError):
            raise ArtifactInstallIOError('Artifact installation I/O failed') from None

        temporary = '.tmp-' + secrets.token_hex(16)
        fd = None
        linked = False
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                         | os.O_NONBLOCK, 0o600, dir_fd=self._fd)
            _private(os.fstat(fd), directory=False)
            remaining = memoryview(content)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError
                remaining = remaining[written:]
            os.fsync(fd)
            os.link(temporary, name, src_dir_fd=self._fd, dst_dir_fd=self._fd,
                    follow_symlinks=False)
            linked = True
            os.close(fd)
            fd = None
            os.unlink(temporary, dir_fd=self._fd)
            os.fsync(self._fd)
            self._verified_range(descriptor, offset=0, limit=0, durable=True)
            self.assert_reachable()
            return ArtifactBlobInstallResult('installed')
        except BaseException as exc:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if not linked:
                try:
                    os.unlink(temporary, dir_fd=self._fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            if not isinstance(exc, Exception):
                raise
            if linked:
                raise ArtifactInstallUncertainError(
                    'Artifact installation durability is uncertain'
                ) from None
            if isinstance(exc, ArtifactInstallError):
                raise
            raise ArtifactInstallIOError('Artifact installation I/O failed') from None

    def _recover_unlinked_install(self, descriptor: ArtifactDescriptor, content: bytes) -> bool:
        """Adopt one exact pre-link crash residue without touching other stages."""
        try:
            self._namespace_bytes()
            stages = [
                entry for entry in os.listdir(self._fd)
                if re.fullmatch(r'\.tmp-[0-9a-f]{32}', entry) is not None
            ]
        except ArtifactStorageError:
            raise ArtifactInstallCorruptionError('Artifact staging evidence is corrupt') from None
        except OSError:
            raise ArtifactInstallIOError('Artifact installation I/O failed') from None
        if not stages:
            return False
        if len(stages) != 1:
            raise ArtifactInstallCorruptionError('Artifact staging evidence is ambiguous')

        stage = stages[0]
        fd = None
        try:
            fd = os.open(stage, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._fd)
            before = os.fstat(fd)
            _private(before, directory=False)
            if before.st_size != descriptor.byte_length:
                raise ArtifactStorageError('Artifact staging length mismatch')
            digest, position = hashlib.sha256(), 0
            decoder = codecs.getincrementaldecoder('utf-8')()
            while chunk := os.read(fd, 65_536):
                if position + len(chunk) > descriptor.byte_length:
                    raise ArtifactStorageError('Artifact staging grew during read')
                if chunk != content[position:position + len(chunk)]:
                    raise ArtifactStorageError('Artifact staging content mismatch')
                digest.update(chunk)
                decoder.decode(chunk)
                position += len(chunk)
            decoder.decode(b'', final=True)
            after = os.fstat(fd)
            _private(after, directory=False)
            if (
                position != descriptor.byte_length
                or digest.hexdigest() != descriptor.sha256
                or (before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise ArtifactStorageError('Artifact staging integrity verification failed')
            os.fsync(fd)
            os.close(fd)
            fd = None
        except ArtifactStorageError:
            raise ArtifactInstallCorruptionError('Artifact staging evidence is corrupt') from None
        except UnicodeError:
            raise ArtifactInstallCorruptionError('Artifact staging evidence is corrupt') from None
        except OSError:
            raise ArtifactInstallIOError('Artifact installation I/O failed') from None
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

        name = descriptor.artifact_id + '.blob'
        linked = False
        try:
            os.link(stage, name, src_dir_fd=self._fd, dst_dir_fd=self._fd,
                    follow_symlinks=False)
            linked = True
            expected_inode = self._inode(before)
            if any(
                self._inode(os.stat(entry, dir_fd=self._fd, follow_symlinks=False))
                != expected_inode
                for entry in (stage, name)
            ):
                raise ArtifactStorageError('Artifact staging binding changed')
            os.unlink(stage, dir_fd=self._fd)
            os.fsync(self._fd)
            self._verified_range(descriptor, offset=0, limit=0, durable=True)
            self.assert_reachable()
            return True
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            if linked:
                raise ArtifactInstallUncertainError(
                    'Artifact installation durability is uncertain'
                ) from None
            if isinstance(exc, ArtifactInstallError):
                raise
            if isinstance(exc, ArtifactStorageError):
                raise ArtifactInstallCorruptionError(
                    'Artifact staging evidence is corrupt'
                ) from None
            raise ArtifactInstallIOError('Artifact installation I/O failed') from None

    def _recover_linked_install(self, descriptor: ArtifactDescriptor) -> None:
        """Recover only a verified blob's single matching internal staging link."""
        name = descriptor.artifact_id + '.blob'
        info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        if info.st_nlink != 2:
            return
        _private(info, directory=False, links=2)
        matches: list[str] = []
        for entry in os.listdir(self._fd):
            if re.fullmatch(r'\.tmp-[0-9a-f]{32}', entry) is None:
                continue
            candidate = os.stat(entry, dir_fd=self._fd, follow_symlinks=False)
            if self._inode(candidate) == self._inode(info):
                _private(candidate, directory=False, links=2)
                matches.append(entry)
        if len(matches) != 1:
            raise ArtifactStorageError('Artifact has unexpected hard links')
        self._verified_range(descriptor, offset=0, limit=0, durable=True, links=2)
        # Names are runtime-owned under the store lock. Do not delete an entry
        # that changed after verification, and never follow a temporary alias.
        if any(self._inode(os.stat(entry, dir_fd=self._fd, follow_symlinks=False)) != self._inode(info)
               for entry in (name, matches[0])):
            raise ArtifactStorageError('Artifact staging binding changed')
        os.unlink(matches[0], dir_fd=self._fd)
        os.fsync(self._fd)

    def read_range(self, descriptor: ArtifactDescriptor, *, offset: int = 0,
                   limit: int = MAX_RANGE_BYTES) -> bytes:
        """Return raw UTF-8 byte offsets after verifying the entire immutable blob.

        A range may split a code point. The eventual text tool must apply its
        documented boundary policy; this internal API does not decode ranges.
        """
        return self._verified_range(descriptor, offset=offset, limit=limit, durable=False)

    def _verified_range(self, descriptor: ArtifactDescriptor, *, offset: int, limit: int,
                        durable: bool, links: int = 1) -> bytes:
        descriptor = self._descriptor(descriptor)
        if (type(offset) is not int or offset < 0 or type(limit) is not int
                or not 0 <= limit <= MAX_RANGE_BYTES):
            raise ArtifactValidationError('Invalid bounded artifact byte range')
        fd = os.open(descriptor.artifact_id + '.blob', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=self._fd)
        try:
            before = os.fstat(fd)
            _private(before, directory=False, links=links)
            if before.st_size != descriptor.byte_length:
                raise ArtifactStorageError('Artifact byte length mismatch')
            digest, position, result = hashlib.sha256(), 0, bytearray()
            decoder = codecs.getincrementaldecoder('utf-8')()
            while chunk := os.read(fd, 65_536):
                if position + len(chunk) > descriptor.byte_length:
                    raise ArtifactStorageError('Artifact grew during read')
                digest.update(chunk)
                decoder.decode(chunk)
                start, end = max(offset - position, 0), min(offset + limit - position, len(chunk))
                if end > start:
                    result.extend(chunk[start:end])
                position += len(chunk)
            decoder.decode(b'', final=True)
            after = os.fstat(fd)
            _private(after, directory=False, links=links)
            if (position != descriptor.byte_length or digest.hexdigest() != descriptor.sha256
                    or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise ArtifactStorageError('Artifact integrity verification failed')
            if durable:
                os.fsync(fd)
                os.fsync(self._fd)
            self.assert_reachable()
            return bytes(result)
        except UnicodeError:
            raise ArtifactStorageError('Artifact is not UTF-8 text') from None
        finally:
            os.close(fd)
