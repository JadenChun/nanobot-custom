"""Bounded server directory browser for owner workspace selection (M37).

The browser exposes configured roots, canonical relative directory listings and
single-use resolve tokens. It never returns file contents, a generic filesystem
API or authority derived from a listing. A resolve token only authorizes a
later fresh grant-commit validation in the permission owner; it is not itself a
filesystem grant.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, cast

from nanobot.config.schema import DirectoryBrowserConfig

MAX_ROOTS = 32
MAX_ENTRIES = 100
MAX_SCANNED = 1_000
MAX_RESPONSE_BYTES = 64 * 1024
MAX_PATH_BYTES = 4_096
MAX_DEPTH = 64
MAX_LIMIT = 100
DEFAULT_LIMIT = 50
DEADLINE_SECONDS = 2.0
TOKEN_TTL_SECONDS = 300.0
MAX_RESOLVE_RECORDS = 256
MAX_CURSOR_RECORDS = 256


class DirectoryBrowserError(RuntimeError):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class _Root:
    id: str
    label: str
    path: Path


@dataclass(frozen=True)
class _ResolveRecord:
    token: str
    installation_id: str
    root_id: str
    relative_path: str
    canonical: Path
    policy_revision: str
    directory_revision: str
    canonical_identity: tuple[int, int]
    expires_at: float
    consumed: bool = False


class WorkspaceDirectoryBrowser:
    def __init__(
        self,
        config_provider: Callable[[], DirectoryBrowserConfig],
        *,
        clock: Callable[[], float] = time.monotonic,
        secret: bytes | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._clock = clock
        self._secret = secret or secrets.token_bytes(32)
        self._resolves: dict[str, _ResolveRecord] = {}
        self._cursors: dict[str, dict[str, Any]] = {}

    # -- public API --------------------------------------------------------

    def roots(self) -> dict[str, Any]:
        roots, policy_revision = self._roots()
        return {
            "roots": [
                {
                    "id": root.id,
                    "label": root.label,
                    "display_path": str(root.path),
                    "revision": self._directory_revision(root.id, "", policy_revision),
                }
                for root in roots
            ],
            "policy_revision": policy_revision,
            "empty": not roots,
        }

    def list(
        self,
        *,
        root_id: str,
        relative_path: str = "",
        cursor: str | None = None,
        limit: int = DEFAULT_LIMIT,
        installation_id: str = "local",
    ) -> dict[str, Any]:
        started = self._clock()
        if isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
            raise DirectoryBrowserError(400, "invalid_limit")
        root, policy_revision = self._require_root(root_id)
        relative = self._relative_for_root(root, relative_path)
        canonical = self._contained(root, relative)
        directory_revision = self._directory_revision(root.id, str(relative), policy_revision)
        current_identity = self._identity(canonical)
        if current_identity is None:
            raise DirectoryBrowserError(409, "directory_changed")

        offset = 0
        if cursor:
            record = self._take_cursor(
                cursor,
                installation_id=installation_id,
                root_id=root.id,
                relative_path=str(relative),
                policy_revision=policy_revision,
                directory_identity=current_identity,
                directory_snapshot=self._directory_snapshot(canonical),
            )
            offset = cast(int, record["offset"])

        try:
            scanned = 0
            index = 0
            entries: list[dict[str, Any]] = []
            more = False
            with os.scandir(canonical) as iterator:
                for dirent in iterator:
                    scanned += 1
                    if scanned > MAX_SCANNED:
                        raise DirectoryBrowserError(413, "scan_too_large")
                    if self._clock() - started > DEADLINE_SECONDS:
                        raise DirectoryBrowserError(503, "directory_deadline")
                    if not self._is_directory(dirent):
                        continue
                    if index < offset:
                        index += 1
                        continue
                    if len(entries) >= limit:
                        more = True
                        break
                    entry = self._entry(root, relative, dirent, installation_id)
                    if entry:
                        entries.append(entry)
                        index += 1
            payload = {
                "root_id": root.id,
                "relative_path": str(relative),
                "entries": entries,
                "next_cursor": (
                    self._mint_cursor(
                        installation_id=installation_id,
                        root_id=root.id,
                        relative_path=str(relative),
                        policy_revision=policy_revision,
                        offset=offset + len(entries),
                        directory_snapshot=self._directory_snapshot(canonical),
                    )
                    if more
                    else None
                ),
                "policy_revision": policy_revision,
                "directory_revision": directory_revision,
            }
        except PermissionError as exc:
            raise DirectoryBrowserError(403, "directory_inaccessible") from exc
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise DirectoryBrowserError(404, "directory_not_found") from exc
        except OSError as exc:
            raise DirectoryBrowserError(503, "directory_unavailable") from exc
        self._check_size(payload)
        return payload

    def resolve(
        self,
        *,
        root_id: str,
        relative_path: str = "",
        policy_revision: str,
        installation_id: str = "local",
    ) -> dict[str, Any]:
        root, current_policy = self._require_root(root_id)
        if policy_revision != current_policy:
            raise DirectoryBrowserError(409, "policy_changed")
        relative = self._relative_for_root(root, relative_path)
        canonical = self._contained(root, relative, require_directory=True)
        canonical_identity = self._identity(canonical)
        if canonical_identity is None:
            raise DirectoryBrowserError(409, "directory_changed")
        directory_revision = self._directory_revision(root.id, str(relative), current_policy)
        token = secrets.token_urlsafe(32)
        now = self._clock()
        self._resolves[token] = _ResolveRecord(
            token=token,
            installation_id=installation_id,
            root_id=root.id,
            relative_path=str(relative),
            canonical=canonical,
            policy_revision=current_policy,
            directory_revision=directory_revision,
            canonical_identity=canonical_identity,
            expires_at=now + TOKEN_TTL_SECONDS,
        )
        self._trim_resolves()
        payload = {
            "root_id": root.id,
            "relative_path": str(relative),
            "display_path": str(canonical),
            "directory_id": self._directory_id(root.id, str(relative), canonical, installation_id),
            "directory_revision": directory_revision,
            "policy_revision": current_policy,
            "expires_at": now + TOKEN_TTL_SECONDS,
            "commit_token": token,
        }
        self._check_size(payload)
        return payload

    def consume_commit_token(
        self,
        token: str,
        *,
        installation_id: str,
        root_id: str,
        relative_path: str,
        extent: str,
    ) -> dict[str, Any]:
        """Validate a fresh grant commit; listing/resolve alone grants nothing."""
        if extent not in {"read", "write"}:
            raise DirectoryBrowserError(400, "invalid_extent")
        record = self._resolves.get(token)
        if record is None:
            raise DirectoryBrowserError(409, "resolve_missing")
        if record.consumed:
            raise DirectoryBrowserError(409, "resolve_reused")
        if self._clock() >= record.expires_at:
            self._resolves.pop(token, None)
            raise DirectoryBrowserError(409, "resolve_expired")
        if record.installation_id != installation_id:
            raise DirectoryBrowserError(403, "resolve_other_device")
        if record.root_id != root_id:
            raise DirectoryBrowserError(409, "resolve_root_changed")
        root, current_policy = self._require_root(root_id)
        if current_policy != record.policy_revision:
            raise DirectoryBrowserError(409, "policy_changed")
        relative = self._normalize_relative(relative_path)
        if str(relative) != record.relative_path:
            raise DirectoryBrowserError(409, "resolve_path_changed")
        canonical = self._contained(root, relative, require_directory=True)
        if canonical != record.canonical:
            raise DirectoryBrowserError(409, "path_replaced")
        if self._directory_revision(root.id, str(relative), current_policy) != record.directory_revision:
            raise DirectoryBrowserError(409, "directory_changed")
        if self._identity(canonical) != record.canonical_identity:
            raise DirectoryBrowserError(409, "path_replaced")
        self._resolves[token] = _ResolveRecord(
            token=record.token,
            installation_id=record.installation_id,
            root_id=record.root_id,
            relative_path=record.relative_path,
            canonical=record.canonical,
            policy_revision=record.policy_revision,
            directory_revision=record.directory_revision,
            canonical_identity=record.canonical_identity,
            expires_at=record.expires_at,
            consumed=True,
        )
        return {
            "root_id": root.id,
            "relative_path": record.relative_path,
            "canonical_path": str(canonical),
            "directory_id": self._directory_id(root.id, record.relative_path, canonical, installation_id),
            "policy_revision": current_policy,
            "directory_revision": record.directory_revision,
            "extent": extent,
        }

    # -- internals ---------------------------------------------------------

    def _roots(self) -> tuple[list[_Root], str]:
        config = self._config_provider()
        roots: list[_Root] = []
        for raw in config.roots[:MAX_ROOTS]:
            path = Path(raw.path).expanduser()
            try:
                canonical = path.resolve(strict=True)
            except OSError:
                continue
            if not canonical.is_dir():
                continue
            roots.append(_Root(id=raw.id, label=raw.label, path=canonical))
        digest = hashlib.sha256(
            "\x00".join(f"{root.id}\x1f{root.path}" for root in roots).encode("utf-8")
        ).hexdigest()[:16]
        return roots, digest

    def _require_root(self, root_id: str) -> tuple[_Root, str]:
        if not root_id:
            raise DirectoryBrowserError(400, "invalid_root")
        roots, policy_revision = self._roots()
        for root in roots:
            if root.id == root_id:
                return root, policy_revision
        raise DirectoryBrowserError(404, "root_not_found")

    def _relative_for_root(self, root: _Root, relative_path: object) -> PurePosixPath:
        if isinstance(relative_path, str) and relative_path.startswith("/"):
            if len(relative_path.encode("utf-8")) > MAX_PATH_BYTES or "\x00" in relative_path:
                raise DirectoryBrowserError(400, "invalid_path")
            candidate = Path(relative_path).resolve(strict=False)
            if candidate != root.path and root.path not in candidate.parents:
                raise DirectoryBrowserError(403, "outside_root")
            try:
                relative = candidate.relative_to(root.path)
            except ValueError as exc:
                raise DirectoryBrowserError(403, "outside_root") from exc
            return PurePosixPath(*relative.parts)
        return self._normalize_relative(relative_path)

    def _normalize_relative(self, relative_path: object) -> PurePosixPath:
        if relative_path is None:
            return PurePosixPath()
        if not isinstance(relative_path, str):
            raise DirectoryBrowserError(400, "invalid_path")
        if len(relative_path.encode("utf-8")) > MAX_PATH_BYTES or "\x00" in relative_path:
            raise DirectoryBrowserError(400, "invalid_path")
        if relative_path.startswith("/") or "\\" in relative_path:
            raise DirectoryBrowserError(400, "invalid_path")
        relative = PurePosixPath(relative_path)
        if relative.is_absolute():
            raise DirectoryBrowserError(400, "invalid_path")
        if len(relative.parts) > MAX_DEPTH:
            raise DirectoryBrowserError(400, "path_too_deep")
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise DirectoryBrowserError(400, "invalid_path")
        return relative

    def _contained(
        self,
        root: _Root,
        relative: PurePosixPath,
        *,
        require_directory: bool = False,
    ) -> Path:
        candidate = root.path / relative
        try:
            canonical = candidate.resolve(strict=require_directory or candidate.exists())
        except OSError as exc:
            raise DirectoryBrowserError(404, "directory_not_found") from exc
        if canonical != root.path and root.path not in canonical.parents:
            raise DirectoryBrowserError(403, "outside_root")
        if require_directory and not canonical.is_dir():
            raise DirectoryBrowserError(404, "directory_not_found")
        return canonical

    def _entry(
        self,
        root: _Root,
        relative: PurePosixPath,
        dirent: os.DirEntry[str],
        installation_id: str,
    ) -> dict[str, Any]:
        entry_relative = relative / dirent.name
        symlink = dirent.is_symlink()
        accessible = True
        try:
            if not symlink:
                accessible = dirent.is_dir()
                if not accessible:
                    return {}
            else:
                target = (root.path / entry_relative).resolve(strict=True)
                accessible = target.is_dir() and (target == root.path or root.path in target.parents)
        except OSError:
            accessible = False
        return {
            "name": dirent.name[:255],
            "directory_id": self._directory_id(
                root.id,
                str(entry_relative),
                root.path / entry_relative,
                installation_id,
            ),
            "symlink": symlink,
            "accessible": accessible,
        }

    @staticmethod
    def _directory_snapshot(path: Path) -> str:
        names: list[str] = []
        try:
            with os.scandir(path) as iterator:
                for index, dirent in enumerate(iterator):
                    if index >= MAX_SCANNED:
                        break
                    names.append(dirent.name)
        except OSError:
            return "unavailable"
        return hashlib.sha256("\x00".join(sorted(names)).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _identity(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return (stat.st_dev, stat.st_ino)

    @staticmethod
    def _is_directory(dirent: os.DirEntry[str]) -> bool:
        try:
            return dirent.is_dir(follow_symlinks=False) or dirent.is_symlink()
        except OSError:
            return False

    def _directory_revision(self, root_id: str, relative_path: str, policy_revision: str) -> str:
        return hashlib.sha256(
            f"{root_id}\x1f{relative_path}\x1f{policy_revision}".encode("utf-8")
        ).hexdigest()[:16]

    def _directory_id(
        self,
        root_id: str,
        relative_path: str,
        canonical: Path,
        installation_id: str,
    ) -> str:
        return hmac.new(
            self._secret,
            f"{installation_id}\x1f{root_id}\x1f{relative_path}\x1f{canonical}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:24]

    def _mint_cursor(
        self,
        *,
        installation_id: str,
        root_id: str,
        relative_path: str,
        policy_revision: str,
        offset: int,
        directory_snapshot: str,
    ) -> str:
        token = secrets.token_urlsafe(24)
        now = self._clock()
        self._cursors[token] = {
            "installation_id": installation_id,
            "root_id": root_id,
            "relative_path": relative_path,
            "policy_revision": policy_revision,
            "offset": offset,
            "directory_identity": self._identity(self._root_path(root_id, relative_path)),
            "directory_snapshot": directory_snapshot,
            "expires_at": now + TOKEN_TTL_SECONDS,
        }
        self._trim_cursors()
        return token

    def _take_cursor(
        self,
        cursor: str,
        *,
        installation_id: str,
        root_id: str,
        relative_path: str,
        policy_revision: str,
        directory_identity: tuple[int, int],
        directory_snapshot: str,
    ) -> dict[str, Any]:
        record = self._cursors.pop(cursor, None)
        if record is None:
            raise DirectoryBrowserError(409, "cursor_invalid")
        if self._clock() >= record["expires_at"]:
            raise DirectoryBrowserError(409, "cursor_expired")
        if (
            record["installation_id"] != installation_id
            or record["root_id"] != root_id
            or record["relative_path"] != relative_path
            or record["policy_revision"] != policy_revision
            or record["directory_identity"] != directory_identity
            or record["directory_snapshot"] != directory_snapshot
        ):
            raise DirectoryBrowserError(409, "cursor_mismatch")
        return record

    def _root_path(self, root_id: str, relative_path: str) -> Path:
        root, _policy = self._require_root(root_id)
        return root.path / relative_path

    def _trim_resolves(self) -> None:
        now = self._clock()
        for token, record in list(self._resolves.items()):
            if now >= record.expires_at:
                self._resolves.pop(token, None)
        while len(self._resolves) > MAX_RESOLVE_RECORDS:
            oldest = min(self._resolves, key=lambda key: self._resolves[key].expires_at)
            self._resolves.pop(oldest, None)

    def _trim_cursors(self) -> None:
        now = self._clock()
        for token, record in list(self._cursors.items()):
            if now >= record["expires_at"]:
                self._cursors.pop(token, None)
        while len(self._cursors) > MAX_CURSOR_RECORDS:
            oldest = min(self._cursors, key=lambda key: self._cursors[key]["expires_at"])
            self._cursors.pop(oldest, None)

    @staticmethod
    def _check_size(payload: dict[str, Any]) -> None:
        import json

        if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise DirectoryBrowserError(413, "response_too_large")
