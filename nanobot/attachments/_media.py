"""Private storage custody, OS fencing and generation-bound media deletion.

Source/files locks are always acquired by the owning writer before this module.
Maintenance obtains the leading memory/migration locks and every declared files
lock before entering the same media fence. No application callback runs here.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import uuid
from collections.abc import Generator, Iterable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

_LIMIT = 64 * 1024 * 1024
_MAX_RECORDS = 100_000
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class MediaRetentionError(OSError):
    """Media custody or reference coverage cannot be established."""


def media_custody_available() -> bool:
    """The qualified executors provide descriptor-relative no-follow I/O."""
    return os.name == "posix" and hasattr(os, "O_NOFOLLOW")


def require_media_custody() -> None:
    if not media_custody_available():
        raise MediaRetentionError("managed media publication and retention unavailable on this host")


def identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def generation(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (*identity(info), info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def directory_fd(path: Path) -> int:
    """Open every component without following aliases replaced by symlinks."""
    require_media_custody()
    if not path.is_absolute() or ".." in path.parts:
        raise MediaRetentionError("invalid media storage path")
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in path.parts[1:]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_bytes(path: Path, *, limit: int = _LIMIT) -> bytes:
    parent = directory_fd(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise MediaRetentionError("invalid or oversized media evidence")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                data = handle.read(limit + 1)
            if len(data) > limit or generation(before) != generation(os.fstat(descriptor)):
                raise MediaRetentionError("media evidence changed while reading")
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if generation(current) != generation(before):
                raise MediaRetentionError("media evidence was replaced")
            return data
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def unique_json(data: bytes | str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in items:
            if name in result:
                raise MediaRetentionError("duplicate media evidence key")
            result[name] = value
        return result

    def invalid(_value: str) -> NoReturn:
        raise MediaRetentionError("nonfinite media evidence value")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def object_json(data: bytes | str) -> dict[str, object]:
    value = unique_json(data)
    if not isinstance(value, dict):
        raise MediaRetentionError("nonobject media evidence")
    return cast(dict[str, object], value)


def fsync_directory(path: Path) -> None:
    descriptor = directory_fd(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    atomic_write_bytes(path, payload)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = directory_fd(path.parent)
    temp = f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temp, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
    finally:
        try:
            os.unlink(temp, dir_fd=parent)
        except FileNotFoundError:
            pass
        os.close(parent)


def unlink_generation(path: Path, expected: tuple[int, ...]) -> None:
    parent = directory_fd(path.parent)
    try:
        current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or generation(current) != expected:
            raise MediaRetentionError("media asset generation changed")
        os.unlink(path.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


class _OSLock:
    def __init__(self, path: Path, *, create: bool = False, blocking: bool = False) -> None:
        self.path, self.create, self.blocking = path, create, blocking
        self.fd: int | None = None
        self.binding: tuple[int, int] | None = None

    def __enter__(self) -> _OSLock:
        import fcntl

        parent = directory_fd(self.path.parent)
        try:
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            if self.create:
                flags |= os.O_CREAT
            descriptor = os.open(self.path.name, flags, 0o600, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode) or os.fstat(descriptor).st_nlink != 1:
                raise MediaRetentionError("invalid media lock")
            fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB))
            self.fd, self.binding = descriptor, identity(os.fstat(descriptor))
            self.require_current()
            return self
        except BaseException:
            self.fd = None
            os.close(descriptor)
            raise

    def require_current(self) -> None:
        parent = directory_fd(self.path.parent)
        try:
            info = os.stat(self.path.name, dir_fd=parent, follow_symlinks=False)
            if self.fd is None or not stat.S_ISREG(info.st_mode) or identity(info) != self.binding:
                raise MediaRetentionError("media lock was replaced")
            if identity(os.fstat(self.fd)) != self.binding:
                raise MediaRetentionError("media lock descriptor changed")
        finally:
            os.close(parent)

    def __exit__(self, *_exc: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@dataclass(frozen=True, slots=True)
class TemporaryMediaCustody:
    chat_id: str
    path: Path
    tokens: tuple[str, ...]
    generation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MediaReferenceEvidence:
    hashes: frozenset[str]
    paths: frozenset[Path]
    complete: bool
    reasons: tuple[str, ...]
    roots: tuple[tuple[str, tuple[int, int]], ...]
    declaration_generation: int
    pending: tuple[tuple[str, Path], ...] = ()
    temporary: tuple[TemporaryMediaCustody, ...] = ()
    assets: tuple[tuple[Path, tuple[int, ...]], ...] = ()


class MediaFence:
    def __init__(self, custody: MediaCustody, lock: _OSLock) -> None:
        self.custody, self.lock = custody, lock
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.active = True
        self.evidence: MediaReferenceEvidence | None = None
        self.root_identity = identity(custody.root.lstat())
        self.state_identity = identity(custody.state.lstat())
        self.leading: list[_OSLock] = []
        self.cleanup: TemporaryMediaCustody | None = None
        self.declaration_bytes: bytes | None = None
        self.namespace_lists: dict[Path, tuple[str, ...]] = {}

    def require(self, custody: MediaCustody) -> None:
        if (custody is not self.custody or not self.active or self.pid != os.getpid()
                or self.thread != threading.get_ident() or not custody.owns_fence(self)):
            raise MediaRetentionError("invalid live media fence")
        descriptor = directory_fd(custody.root)
        try:
            if identity(os.fstat(descriptor)) != self.root_identity:
                raise MediaRetentionError("media root was replaced")
        finally:
            os.close(descriptor)
        descriptor = directory_fd(custody.state)
        try:
            if identity(os.fstat(descriptor)) != self.state_identity:
                raise MediaRetentionError("media custody directory was replaced")
        finally:
            os.close(descriptor)
        self.lock.require_current()
        for lock in self.leading:
            lock.require_current()
        for path, expected_names in self.namespace_lists.items():
            descriptor = directory_fd(path)
            try:
                if tuple(sorted(os.listdir(descriptor))) != expected_names:
                    raise MediaRetentionError("session namespace inventory changed")
            finally:
                os.close(descriptor)
        if self.declaration_bytes is not None and read_bytes(custody.registry) != self.declaration_bytes:
            raise MediaRetentionError("media declarations changed during maintenance")
        if self.evidence is not None:
            for path, expected in self.evidence.roots:
                descriptor = directory_fd(Path(path))
                try:
                    if identity(os.fstat(descriptor)) != expected:
                        raise MediaRetentionError("declared root changed during maintenance")
                finally:
                    os.close(descriptor)

    def __copy__(self) -> NoReturn:
        raise TypeError("media fence cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("media fence cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("media fence cannot be serialized")


class MediaCustody:
    """The store's private durable media state; construction is read-only."""

    def __init__(self, media_root: Path) -> None:
        self.root = media_root
        self.state = media_root / ".retention"
        self.registry = self.state / "roots.json"
        self._live_fences: set[MediaFence] = set()
        self._maintenance_evidence: dict[MediaFence, MediaReferenceEvidence] = {}

    def owns_fence(self, gate: MediaFence) -> bool:
        return gate in self._live_fences

    @contextmanager
    def fence(self, *, create: bool = False, blocking: bool = True) -> Generator[MediaFence, None, None]:
        require_media_custody()
        if create:
            self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        with _LOCKS_GUARD:
            mutex = _LOCKS.setdefault(str(self.root), threading.RLock())
        acquired = mutex.acquire(blocking=blocking)
        if not acquired:
            raise MediaRetentionError("media fence busy")
        try:
            with _OSLock(self.state / "publication.lock", create=create, blocking=blocking) as lock:
                gate = MediaFence(self, lock)
                self._live_fences.add(gate)
                try:
                    yield gate
                finally:
                    gate.active = False
                    self._maintenance_evidence.pop(gate, None)
                    self._live_fences.discard(gate)
        finally:
            mutex.release()

    def declarations(self) -> tuple[dict[str, object], bytes]:
        raw = read_bytes(self.registry)
        value = object_json(raw)
        if (set(value) != {"version", "generation", "roots"} or type(value["version"]) is not int
                or value["version"] != 1 or type(value["generation"]) is not int
                or value["generation"] < 1 or not isinstance(value["roots"], list)
                or not value["roots"] or len(cast(list[object], value["roots"])) > 1024):
            raise MediaRetentionError("invalid media root declaration")
        seen: set[tuple[str, str]] = set()
        for item in cast(list[object], value["roots"]):
            if not isinstance(item, dict):
                raise MediaRetentionError("invalid media root declaration")
            row = cast(dict[str, object], item)
            kind, path = row.get("kind"), row.get("path")
            expected = {"kind", "path", "identity"}
            if kind == "sessions":
                expected |= {"sessions_root", "workspace", "workspace_identity"}
            if (kind not in {"sessions", "webui", "memory", "temporary", "generated", "unsupported_sessions"}
                    or set(row) != expected or not isinstance(path, str)
                    or not Path(path).is_absolute() or ".." in Path(path).parts):
                raise MediaRetentionError("unsupported media root declaration")
            if kind == "generated" and not Path(path).is_relative_to(self.root):
                raise MediaRetentionError("generated media declaration escapes store")
            for name in ("identity", "workspace_identity") if kind == "sessions" else ("identity",):
                pair = row.get(name)
                if not isinstance(pair, list) or len(cast(list[object], pair)) != 2 or any(type(v) is not int for v in cast(list[object], pair)):
                    raise MediaRetentionError("invalid declared root identity")
            key = str(kind), path
            if key in seen:
                raise MediaRetentionError("duplicate media root declaration")
            seen.add(key)
            if kind == "sessions":
                for name in ("sessions_root", "workspace"):
                    entry = row.get(name)
                    if not isinstance(entry, str) or not Path(entry).is_absolute() or ".." in Path(entry).parts:
                        raise MediaRetentionError("invalid session media declaration")
                if Path(path).parent != Path(cast(str, row["sessions_root"])):
                    raise MediaRetentionError("invalid namespace declaration")
        return value, raw

    def declare(self, kind: str, path: Path, *, sessions_root: Path | None = None,
                workspace: Path | None = None) -> None:
        # Plain text/session startup remains available outside the qualified
        # executors. A skipped declaration never certifies media coverage.
        if not media_custody_available():
            return
        descriptor = directory_fd(path)
        try:
            row: dict[str, object] = {"kind": kind, "path": str(path), "identity": list(identity(os.fstat(descriptor)))}
        finally:
            os.close(descriptor)
        if kind == "sessions":
            if sessions_root is None or workspace is None:
                raise MediaRetentionError("session declaration needs its owning workspace")
            descriptor = directory_fd(workspace)
            try:
                workspace_identity = list(identity(os.fstat(descriptor)))
            finally:
                os.close(descriptor)
            row.update(sessions_root=str(sessions_root), workspace=str(workspace),
                       workspace_identity=workspace_identity)
        with self.fence(create=True) as gate:
            registry: dict[str, object]
            try:
                registry, _raw = self.declarations()
            except FileNotFoundError:
                registry = {"version": 1, "generation": 1, "roots": []}
            roots = cast(list[object], registry["roots"])
            for existing in roots:
                if not isinstance(existing, dict):
                    raise MediaRetentionError("invalid media declaration")
                declared = cast(dict[str, object], existing)
                if declared.get("kind") == kind and declared.get("path") == str(path):
                    if existing != row:
                        roots = self._relocated_session_roots(roots, declared, row)
                        registry["roots"] = roots
                        registry["generation"] = cast(int, registry["generation"]) + 1
                        gate.require(self)
                        atomic_write(self.registry, registry)
                    return
            gate.require(self)
            roots.append(row)
            registry["generation"] = cast(int, registry["generation"]) + 1
            atomic_write(self.registry, registry)

    @staticmethod
    def _relocated_session_roots(roots: list[object], prior: dict[str, object],
                                 current: dict[str, object]) -> list[object]:
        """Follow only the source owner's committed, inode-preserving move.

        JsonlSessionStore calls declaration after claiming its namespace under
        memory/migration/files locks. Its new binding and travelled identity
        marker must agree; path changes cannot refresh any recorded identity.
        """
        if (prior.get("kind") != "sessions" or current.get("kind") != "sessions"
                or {name: value for name, value in prior.items() if name != "workspace"}
                != {name: value for name, value in current.items() if name != "workspace"}):
            raise MediaRetentionError("declared media root was replaced")
        old_workspace = Path(cast(str, prior["workspace"]))
        workspace = Path(cast(str, current["workspace"]))
        namespace = Path(cast(str, current["path"]))
        if old_workspace == workspace:
            raise MediaRetentionError("declared media root was replaced")
        try:
            old_workspace.lstat()
        except FileNotFoundError:
            pass
        else:
            descriptor = directory_fd(old_workspace)
            try:
                if list(identity(os.fstat(descriptor))) != prior["workspace_identity"]:
                    raise MediaRetentionError("previous workspace path was replaced")
            finally:
                os.close(descriptor)
        for selected, expected in ((namespace, current["identity"]),
                                   (workspace, current["workspace_identity"])):
            descriptor = directory_fd(selected)
            try:
                if list(identity(os.fstat(descriptor))) != expected:
                    raise MediaRetentionError("relocated media root identity changed")
            finally:
                os.close(descriptor)
        if (read_bytes(namespace / ".workspace", limit=4096).decode().strip() != str(workspace)
                or read_bytes(workspace / ".nanobot" / "workspace-id", limit=128).decode().strip() != namespace.name):
            raise MediaRetentionError("relocated workspace owner binding mismatch")
        relocated: list[object] = []
        for raw in roots:
            declared = cast(dict[str, object], raw)
            if declared is prior:
                relocated.append(current)
            elif declared["kind"] == "memory" and declared["path"] == str(old_workspace / "memory"):
                memory = workspace / "memory"
                descriptor = directory_fd(memory)
                try:
                    if list(identity(os.fstat(descriptor))) != declared["identity"]:
                        raise MediaRetentionError("relocated memory root was replaced")
                finally:
                    os.close(descriptor)
                relocated.append({**declared, "path": str(memory)})
            else:
                relocated.append(declared)
        keys = [(row["kind"], row["path"]) for row in cast(list[dict[str, object]], relocated)]
        if len(set(keys)) != len(keys):
            raise MediaRetentionError("duplicate relocated media root")
        return relocated

    def pin(self, gate: MediaFence, *, publication_id: str, path: Path, sha256: str,
            token: str | None = None) -> str:
        gate.require(self)
        if not publication_id or not path.is_relative_to(self.root):
            raise MediaRetentionError("invalid managed publication")
        pending = self.records("pending")
        for row in pending:
            if row["publication_id"] == publication_id:
                if row["path"] != str(path) or row["sha256"] != sha256:
                    raise MediaRetentionError("publication custody conflict")
                return cast(str, row["token"])
        if len(pending) >= _MAX_RECORDS:
            raise MediaRetentionError("pending media custody capacity exhausted")
        token = token or uuid.uuid4().hex
        atomic_write(self.state / "pending" / f"{token}.json", {
            "version": 1, "token": token, "publication_id": publication_id,
            "path": str(path), "sha256": sha256,
        })
        return token

    def records(self, name: str) -> list[dict[str, object]]:
        directory = self.state / name
        if not directory.exists() and not directory.is_symlink():
            return []
        descriptor = directory_fd(directory)
        try:
            names = os.listdir(descriptor)
        finally:
            os.close(descriptor)
        if len(names) > _MAX_RECORDS:
            raise MediaRetentionError("media custody budget exhausted")
        rows: list[dict[str, object]] = []
        for filename in sorted(names):
            if not filename.endswith(".json"):
                raise MediaRetentionError("unknown media custody residue")
            row = object_json(read_bytes(directory / filename, limit=8192))
            expected = ({"version", "token", "publication_id", "path", "sha256"} if name == "pending"
                        else {"version", "token", "path", "blob_generation", "sidecar", "sidecar_generation"})
            token = row.get("token")
            if (set(row) != expected or type(row.get("version")) is not int or row["version"] != 1
                    or not isinstance(token, str) or len(token) != 32
                    or any(c not in "0123456789abcdef" for c in token) or filename != f"{token}.json"):
                raise MediaRetentionError("invalid media custody record")
            path = row.get("path")
            if (not isinstance(path, str) or not Path(path).is_absolute() or ".." in Path(path).parts
                    or not Path(path).is_relative_to(self.root)):
                raise MediaRetentionError("media custody path escapes store")
            if name == "pending":
                sha = row.get("sha256")
                if (not isinstance(row.get("publication_id"), str) or not row["publication_id"]
                        or not isinstance(sha, str) or len(sha) != 64
                        or any(c not in "0123456789abcdef" for c in sha)):
                    raise MediaRetentionError("invalid pending media identity")
            else:
                sidecar = row.get("sidecar")
                if not isinstance(sidecar, str) or Path(sidecar) != Path(path).with_suffix(".json"):
                    raise MediaRetentionError("invalid deletion sidecar")
                for key in ("blob_generation", "sidecar_generation"):
                    value = row[key]
                    if value is None and key == "sidecar_generation":
                        continue
                    if not isinstance(value, list) or len(cast(list[object], value)) != 5 or any(type(v) is not int for v in cast(list[object], value)):
                        raise MediaRetentionError("invalid deletion generation")
            rows.append(row)
        return rows

    def acknowledge(self, gate: MediaFence, tokens: Iterable[str]) -> None:
        gate.require(self)
        for token in tokens:
            path = self.state / "pending" / f"{token}.json"
            unlink_generation(path, generation(path.lstat()))

    @contextmanager
    def maintenance(self) -> Generator[tuple[MediaFence, MediaReferenceEvidence], None, None]:
        from nanobot.attachments._inventory import inventory

        require_media_custody()
        registry, raw = self.declarations()
        roots = cast(list[dict[str, object]], registry["roots"])
        workspaces = sorted({Path(cast(str, row["workspace"])) for row in roots if row["kind"] == "sessions"}
                            | {Path(cast(str, row["path"])).parent for row in roots if row["kind"] == "memory"})
        session_roots = sorted({Path(cast(str, row["sessions_root"])) for row in roots if row["kind"] == "sessions"})
        with ExitStack() as stack:
            leading: list[_OSLock] = []
            # These are the existing leading locks, not a second recovery protocol.
            for workspace in workspaces:
                leading.append(stack.enter_context(_OSLock(workspace / ".memory-recovery.lock")))
            for root in session_roots:
                leading.append(stack.enter_context(_OSLock(root / ".workspace-migration.lock")))
            for row in sorted(roots, key=lambda row: str(row["path"])):
                if row["kind"] == "sessions":
                    leading.append(stack.enter_context(_OSLock(Path(cast(str, row["path"])) / ".session-files.lock")))
            gate = stack.enter_context(self.fence(blocking=False))
            gate.leading = leading
            if read_bytes(self.registry) != raw:
                raise MediaRetentionError("media root declarations changed")
            gate.declaration_bytes = raw
            for root in session_roots:
                descriptor = directory_fd(root)
                try:
                    gate.namespace_lists[root] = tuple(sorted(os.listdir(descriptor)))
                finally:
                    os.close(descriptor)
            evidence = inventory(self, registry)
            gate.evidence = evidence
            self._maintenance_evidence[gate] = evidence
            gate.require(self)
            yield gate, evidence

    def delete(self, gate: MediaFence, evidence: MediaReferenceEvidence, path: Path) -> bool:
        """One gate for blob and sidecar deletion, including a reopened intent."""
        gate.require(self)
        if (gate.evidence is not evidence or self._maintenance_evidence.get(gate) is not evidence
                or not evidence.complete):
            raise MediaRetentionError("fresh complete media evidence required")
        if path in evidence.paths or path.stem in evidence.hashes:
            return False
        consuming = gate.cleanup
        consumed_tokens = consuming.tokens if consuming is not None and consuming.path == path else ()
        if any(pinned == path and token not in consumed_tokens for token, pinned in evidence.pending):
            return False
        if any(row.path == path and row is not consuming for row in evidence.temporary):
            return False
        if not path.is_relative_to(self.root) or path.suffix == ".json":
            raise MediaRetentionError("invalid media deletion target")
        rows = self.records("deletions")
        matching = [row for row in rows if row["path"] == str(path)]
        if len(matching) > 1:
            raise MediaRetentionError("duplicate deletion custody")
        sidecar = path.with_suffix(".json")
        if matching:
            intent = matching[0]
        else:
            info = path.lstat()
            expected = dict(evidence.assets).get(path)
            if expected is None or generation(info) != expected:
                raise MediaRetentionError("media asset changed after inventory")
            if not stat.S_ISREG(info.st_mode):
                raise MediaRetentionError("invalid media blob")
            sidecar_info = sidecar.lstat() if sidecar.exists() or sidecar.is_symlink() else None
            if sidecar_info is not None and not stat.S_ISREG(sidecar_info.st_mode):
                raise MediaRetentionError("invalid media sidecar")
            expected_sidecar = dict(evidence.assets).get(sidecar)
            if (generation(sidecar_info) if sidecar_info is not None else None) != expected_sidecar:
                raise MediaRetentionError("media sidecar changed after inventory")
            intent = {"version": 1, "token": uuid.uuid4().hex, "path": str(path),
                      "blob_generation": list(generation(info)), "sidecar": str(sidecar),
                      "sidecar_generation": list(generation(sidecar_info)) if sidecar_info is not None else None}
            atomic_write(self.state / "deletions" / f"{intent['token']}.json", intent)
        gate.require(self)
        blob_generation = tuple(cast(list[int], intent["blob_generation"]))
        try:
            current = path.lstat()
        except FileNotFoundError:
            current = None
        if current is not None:
            if generation(current) != blob_generation:
                # Retire only this stale retry. A later fresh retention pass
                # may assess a genuine orphan in the replacement generation.
                stale = self.state / "deletions" / f"{intent['token']}.json"
                unlink_generation(stale, generation(stale.lstat()))
                return False
            unlink_generation(path, blob_generation)
        side_generation = intent["sidecar_generation"]
        if side_generation is not None:
            try:
                unlink_generation(sidecar, tuple(cast(list[int], side_generation)))
            except FileNotFoundError:
                pass
        intent_path = self.state / "deletions" / f"{intent['token']}.json"
        unlink_generation(intent_path, generation(intent_path.lstat()))
        return True

    def cleanup_temporary(self, chat_id: str) -> set[Path]:
        """Consume only custody read from the exact durable temporary owner.

        A chat name or caller path is not deletion authority. The live fence,
        complete inventory, exact pending tokens and original generation are
        checked together; unrelated pins and newly published references win.
        """
        removed: set[Path] = set()
        with self.maintenance() as (gate, evidence):
            if not evidence.complete:
                return removed
            pending = {str(row["token"]): row for row in self.records("pending")}
            for claim in evidence.temporary:
                if claim.chat_id != chat_id:
                    continue
                path = claim.path
                if path.exists() or path.is_symlink():
                    if generation(path.lstat()) != claim.generation:
                        continue
                if any(token in pending and pending[token]["path"] != str(path) for token in claim.tokens):
                    raise MediaRetentionError("temporary custody token mismatch")
                gate.cleanup = claim
                try:
                    if not path.exists() and not path.is_symlink() and not path.with_suffix(".json").exists():
                        gate.require(self)
                        self.acknowledge(gate, [token for token in claim.tokens if token in pending])
                        removed.add(path)
                        continue
                    if self.delete(gate, evidence, path):
                        self.acknowledge(gate, [token for token in claim.tokens if token in pending])
                        removed.add(path)
                except OSError:
                    continue
                finally:
                    gate.cleanup = None
        return removed


def configured_custody() -> MediaCustody:
    from nanobot.config.paths import inspect_media_dir

    return MediaCustody(inspect_media_dir())
