"""Built-in text read evidence, separate from model-visible tool content."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

if TYPE_CHECKING:
    from nanobot.context_repo.policy import ResourceAccessPolicy
    from nanobot.session.artifact import FilesystemArtifactOrigin


FilesystemReadProducer = Literal['builtin_read_file_text', 'builtin_grep_file_text']
_FILESYSTEM_READ_PRODUCERS: Final[frozenset[FilesystemReadProducer]] = frozenset({
    'builtin_read_file_text',
    'builtin_grep_file_text',
})


def root_binding(root: Path) -> str:
    """Prevent relative locators being rebased onto another project/repository."""
    return hashlib.sha256(str(root.expanduser().resolve()).encode()).hexdigest()


@dataclass(frozen=True)
class FilesystemReadReceipt:
    source_access: FilesystemArtifactOrigin
    root_binding: str
    byte_length: int
    sha256: str
    schema_version: int = 1
    producer: FilesystemReadProducer = 'builtin_read_file_text'
    block_index: None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def matches(self, content: object) -> bool:
        if not isinstance(content, str):
            return False
        try:
            raw = content.encode('utf-8')
        except UnicodeError:
            return False
        return len(raw) == self.byte_length and hashlib.sha256(raw).hexdigest() == self.sha256


def parse_read_receipt(raw: object) -> FilesystemReadReceipt:
    from nanobot.session.artifact import ArtifactValidationError, _origin

    if (not isinstance(raw, dict) or set(raw) != {'source_access', 'root_binding', 'byte_length',
            'sha256', 'schema_version', 'producer', 'block_index'}
            or type(raw['schema_version']) is not int or raw['schema_version'] != 1
            or not isinstance(raw['producer'], str)
            or raw['producer'] not in _FILESYSTEM_READ_PRODUCERS
            or raw['block_index'] is not None
            or type(raw['byte_length']) is not int or raw['byte_length'] < 0
            or any(not isinstance(raw[key], str) or len(raw[key]) != 64
                   or any(c not in '0123456789abcdef' for c in raw[key])
                   for key in ('sha256', 'root_binding'))):
        raise ArtifactValidationError('Invalid filesystem producer receipt')
    return FilesystemReadReceipt(_origin(raw['source_access']), raw['root_binding'],
                                 raw['byte_length'], raw['sha256'], producer=raw['producer'])


def locate_read(path: Path, workspace: Path | None, policy: ResourceAccessPolicy | None
                ) -> tuple[FilesystemArtifactOrigin, str] | None:
    """Select policy precedence without inferring grants from touched paths."""
    from nanobot.session.artifact import FilesystemArtifactOrigin

    if policy is not None:
        policy.validate_path(path, 'read')
        manager = policy.context_manager
        target = manager.find_target_repo_for_path(path)
        repo = manager.find_repo_for_path(path)
        if target and (repo is None or len(str(target.path or '')) >= len(str(repo.path))):
            if target.path is None:
                return None
            root = target.path.resolve()
            # A target name is only unique within its owning context repository.
            identity = json.dumps([target.context_repo_name, target.name], separators=(',', ':'))
            origin = FilesystemArtifactOrigin('filesystem', 'target_repository', identity,
                                              path.relative_to(root).as_posix())
            return origin, root_binding(root)
        if repo:
            root = repo.path.resolve()
            return (FilesystemArtifactOrigin('filesystem', 'context_repository', repo.name,
                                              path.relative_to(root).as_posix()), root_binding(root))
    if workspace is not None:
        root = workspace.expanduser().resolve()
        if path.is_relative_to(root):
            return (FilesystemArtifactOrigin('filesystem', 'workspace', None,
                                              path.relative_to(root).as_posix()), root_binding(root))
    return None


def _stamp(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextmanager
def qualified_text_bytes(path: Path, *, max_bytes: int
                         ) -> Iterator[tuple[bytes, Callable[[], None]] | None]:
    """Hold a no-follow descriptor chain through rendering and receipt creation.

    Unsupported systems return no qualification. Supported systems reject changed
    objects; callers must not issue receipts from ordinary fallback reads.
    """
    if (os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW')
            or not hasattr(os, 'O_DIRECTORY') or os.open not in os.supports_dir_fd):
        yield None
        return
    fds = []
    try:
        fds.append(os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        for part in path.parts[1:-1]:
            fds.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=fds[-1]))
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fds[-1])
        fds.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise OSError('Unqualified text source')
        chunks = []
        count = 0
        while chunk := os.read(fd, min(65536, max_bytes + 1 - count)):
            chunks.append(chunk)
            count += len(chunk)
            if count > max_bytes:
                raise OSError('Text source exceeded read bound')
        def validate() -> None:
            if _stamp(os.fstat(fd)) != _stamp(before):
                raise OSError('Text source changed during read')
            # Rewalk the absolute name without following links. Every parent and
            # the final leaf must still identify the held objects.
            parent = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for index, part in enumerate(path.parts[1:]):
                    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                    if index < len(path.parts) - 2:
                        flags |= os.O_DIRECTORY
                    child = os.open(part, flags, dir_fd=parent)
                    os.close(parent)
                    parent = child
                    if _stamp(os.fstat(child))[:2] != _stamp(os.fstat(fds[index + 1]))[:2]:
                        raise OSError('Text source namespace changed')
                if _stamp(os.fstat(parent)) != _stamp(before):
                    raise OSError('Text source changed during read')
            finally:
                os.close(parent)
        validate()
        yield b''.join(chunks), validate
    finally:
        for fd in reversed(fds):
            os.close(fd)



def revalidate_read_receipt(receipt: FilesystemReadReceipt, *, workspace: Path | None,
                            policy: ResourceAccessPolicy | None,
                            resolve_read: Callable[[str], Path]) -> None:
    """Reconstruct only the recorded namespace, then apply current tool policy."""
    from nanobot.session.artifact import ArtifactValidationError

    receipt = parse_read_receipt(receipt.to_dict())
    origin = receipt.source_access
    roots = []
    if origin.access_class == 'workspace':
        if workspace is not None:
            roots = [workspace.expanduser().resolve()]
    elif policy is not None:
        manager = policy.context_manager
        if origin.access_class == 'context_repository':
            roots = [repo.path.resolve() for repo in manager.repos if repo.name == origin.repository]
        else:
            roots = [target.path.resolve() for target in manager.target_repos()
                     if target.path is not None and json.dumps(
                         [target.context_repo_name, target.name], separators=(',', ':')) == origin.repository]
    if len(roots) != 1 or root_binding(roots[0]) != receipt.root_binding:
        raise ArtifactValidationError('Artifact source namespace is unavailable')
    candidate = roots[0].joinpath(*origin.relative_path.split('/'))
    try:
        resolved = resolve_read(str(candidate))
        current = locate_read(resolved, workspace, policy)
    except (OSError, ValueError, RuntimeError):
        raise ArtifactValidationError('Artifact source access is unavailable') from None
    if current != (origin, receipt.root_binding):
        raise ArtifactValidationError('Artifact source access changed')
