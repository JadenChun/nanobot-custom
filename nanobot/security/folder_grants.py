"""Trusted, instance-local folder grants for managed WebUI access."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal, Self, cast

from filelock import FileLock, Timeout
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from nanobot.config.paths import (
    get_config_path,
    get_default_sessions_root,
    get_webui_dir,
)
from nanobot.security.native_admission import (
    NativeAdmissionActiveError,
    NativeAdmissionLedger,
    NativeAdmissionUnknownError,
    NativeOwnerTag,
    NativeProcessReservation,
)

if TYPE_CHECKING:
    from nanobot.security.desktop_permissions import (
        DesktopPermissionOwner,
        WorkspacePermissionReference,
    )

FolderFileAccess = Literal["read_only", "read_write"]

_MAX_FOLDER_GRANTS = 64
_MAX_GRANT_STORE_BYTES = 128 * 1024


class FolderGrant(BaseModel):
    """One canonical folder capability selected by the local user."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str = Field(max_length=2048)
    file_access: FolderFileAccess
    commands: StrictBool

    @field_validator("path")
    @classmethod
    def valid_path_text(cls, value: str) -> str:
        if not value or "\0" in value or not Path(value).is_absolute():
            raise ValueError("folder path must be absolute and contain no NUL characters")
        return value

    @model_validator(mode="after")
    def command_grant_is_not_system_control(self) -> Self:
        if self.commands and _overlaps_system_control(Path(self.path)):
            raise ValueError("command grants cannot cover /proc, /sys, /dev, or /run")
        return self


class FolderGrantInput(BaseModel):
    """Untrusted UI input before ``~`` expansion and filesystem resolution."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str = Field(max_length=2048)
    file_access: FolderFileAccess
    commands: StrictBool

    @field_validator("path")
    @classmethod
    def valid_path_text(cls, value: str) -> str:
        if not value or "\0" in value:
            raise ValueError("folder path must be nonempty and contain no NUL characters")
        return value


class _GrantStoreData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    version: Literal[1]
    enabled: StrictBool
    revision: StrictInt = Field(ge=0)
    grants: tuple[FolderGrant, ...] = Field(max_length=_MAX_FOLDER_GRANTS)

    @field_validator("enabled")
    @classmethod
    def must_be_activated(cls, value: bool) -> bool:
        if not value:
            raise ValueError("an activated folder grant store must remain enabled")
        return value

    @field_validator("grants")
    @classmethod
    def unique_roots(cls, value: tuple[FolderGrant, ...]) -> tuple[FolderGrant, ...]:
        paths = [grant.path for grant in value]
        if len(paths) != len(set(paths)):
            raise ValueError("folder grant roots must be unique")
        return tuple(sorted(value, key=lambda grant: grant.path))


class FolderGrantError(ValueError):
    """A local folder grant request could not be applied."""


class GrantRevisionConflictError(FolderGrantError):
    """The UI tried to update a grant snapshot that is no longer current."""


class FolderGrantsUnavailableError(FolderGrantError):
    """The activated grant authority is invalid or has changed unexpectedly."""


class NativeProcessesActiveError(FolderGrantError):
    """An unrestricted owned command is still running during first activation."""


class NativeProcessesUnknownError(FolderGrantError):
    """Native process admission is uncertain after losing same-boot proof."""


@dataclass(frozen=True, slots=True)
class FolderGrantSnapshot:
    enabled: bool
    revision: int
    grants: tuple[FolderGrant, ...]


@dataclass(frozen=True, slots=True)
class FolderGrantUpsertResult:
    """The exact validated grant and snapshot committed by one upsert."""

    submitted_path: str
    grant: FolderGrant
    snapshot: FolderGrantSnapshot


@dataclass(frozen=True, slots=True)
class _Fingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    digest: str


class FolderGrantView:
    """Read-only live view shared by tools; it exposes no persistence writer."""

    def __init__(
        self,
        path: Path,
        activation_path: Path,
        snapshot: FolderGrantSnapshot,
        fingerprint: _Fingerprint | None,
        activation_fingerprint: _Fingerprint | None,
        *,
        unavailable: bool = False,
        agent_home: Path,
        private_roots: tuple[Path, ...] = (),
        instance_root: Path,
    ) -> None:
        self._path = path
        self._activation_path = activation_path
        self._snapshot = snapshot
        self._fingerprint = fingerprint
        self._activation_fingerprint = activation_fingerprint
        self._unavailable = unavailable
        self._agent_home = agent_home
        self._private_roots = private_roots
        self._instance_root = instance_root
        self._lock = threading.RLock()
        self._native_admission = NativeAdmissionLedger(
            path.with_name(".native-process-admissions.json")
        )

    @property
    def is_managed(self) -> bool:
        """Whether managed access was activated or its store became suspect."""
        with self._lock:
            return (
                self._snapshot.enabled
                or self._unavailable
                or self._path.exists()
                or self._activation_path.exists()
            )

    def snapshot(self) -> FolderGrantSnapshot:
        with self._lock:
            if self._unavailable:
                raise FolderGrantsUnavailableError("folder grants are unavailable")
            try:
                current = _fingerprint_file(self._path)
                activation = _fingerprint_activation(self._activation_path)
            except (OSError, ValueError) as exc:
                self._unavailable = True
                raise FolderGrantsUnavailableError("folder grants are unavailable") from exc
            if (
                current != self._fingerprint
                or activation != self._activation_fingerprint
                or (self._snapshot.enabled and activation is None)
            ):
                self._unavailable = True
                raise FolderGrantsUnavailableError("folder grant store changed unexpectedly")
            return self._snapshot

    @property
    def sandbox_private_roots(self) -> tuple[Path, ...]:
        """Private runtime roots required to mask these paths in managed commands."""
        with self._lock:
            return self._private_roots

    @property
    def sandbox_instance_root(self) -> Path:
        """The instance boundary masked by managed command launches."""
        return self._instance_root

    def effective_grant_for(self, path: Path) -> FolderGrant | None:
        """Return the most-specific live grant covering the resolved target."""
        snapshot = self.snapshot()
        if not snapshot.enabled:
            return None
        resolved = path.expanduser().resolve(strict=False)
        matching: list[FolderGrant] = []
        for grant in snapshot.grants:
            root = Path(grant.path)
            if _root_identity_is_live(root) and _is_under(resolved, root):
                matching.append(grant)
        return max(matching, key=lambda grant: len(Path(grant.path).parts), default=None)

    def is_private_path(self, path: Path) -> bool:
        """Hide authoritative grant storage and its adjacent lock/temp files."""
        resolved = path.expanduser().resolve(strict=False)
        with self._lock:
            if any(_is_under(resolved, root) for root in self._private_roots):
                return True
            return _is_under(resolved, self._instance_root) and not _is_under(
                resolved,
                self._agent_home,
            )

    def overlaps_private_root(self, path: Path) -> bool:
        """Return whether a grant root overlaps any private runtime location."""
        resolved = path.expanduser().resolve(strict=False)
        with self._lock:
            return any(_overlaps(resolved, root) for root in self._private_roots) or _overlaps(
                resolved,
                self._instance_root,
            )

    def try_begin_native_process(self) -> NativeProcessReservation | None:
        """Durably reserve legacy launch admission before a launch gate opens."""
        try:
            return self._native_admission.begin(managed=lambda: self.is_managed)
        except (OSError, ValueError) as exc:
            self._unavailable = True
            raise FolderGrantsUnavailableError("native process admission is unavailable") from exc

    def try_begin_owned_native_process(
        self,
        permission_owner: DesktopPermissionOwner,
        permission_reference: WorkspacePermissionReference,
    ) -> NativeProcessReservation | None:
        """Reserve a launch under a current private Once/Full owner."""
        owner_tag = permission_owner.native_owner_tag(permission_reference)
        if owner_tag is None:
            return None
        reservations: list[NativeProcessReservation] = []

        def reserve() -> None:
            reservations.append(self._native_admission.begin_owned(
                owner_tag,
                permission_owner=permission_owner,
                permission_reference=permission_reference,
            ))

        try:
            if not permission_owner.with_current_reference(permission_reference, reserve):
                return None
            return reservations[0]
        except (OSError, ValueError) as exc:
            self._unavailable = True
            raise FolderGrantsUnavailableError("native process admission is unavailable") from exc

    def owned_native_process_counts(
        self,
        permission_owner: DesktopPermissionOwner,
        permission_reference: WorkspacePermissionReference,
    ) -> tuple[int, int]:
        owner_tag = permission_owner.native_owner_tag(permission_reference)
        if owner_tag is None:
            return 0, 0
        try:
            return self._native_admission.owner_counts(owner_tag)
        except (OSError, ValueError) as exc:
            self._unavailable = True
            raise FolderGrantsUnavailableError("native process admission is unavailable") from exc

    def tagged_native_process_counts(self, owner_tag: object) -> tuple[int, int]:
        try:
            return self._native_admission.owner_counts(cast(NativeOwnerTag, owner_tag))
        except (OSError, TypeError, ValueError) as exc:
            self._unavailable = True
            raise FolderGrantsUnavailableError("native process admission is unavailable") from exc

    def source_native_process_counts(
        self,
        *,
        session_key: str,
        source_binding: object,
    ) -> tuple[int, int]:
        try:
            binding = cast(dict[str, object], source_binding)
            return self._native_admission.source_counts(
                session_key=session_key,
                store_identity=cast(str, binding["store_identity"]),
                incarnation=cast(str, binding["incarnation"]),
                generation=cast(int, binding["generation"]),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            self._unavailable = True
            raise FolderGrantsUnavailableError("native process admission is unavailable") from exc

    def session_native_process_counts(
        self,
        *,
        session_key: str,
        kind: str | None = None,
    ) -> tuple[int, int]:
        """Account for owned trees across source generations after reset/restart."""
        if kind not in {None, "task_once", "chat_full"}:
            raise ValueError("invalid native permission owner kind")
        try:
            return self._native_admission.session_owner_counts(
                session_key=session_key,
                kind=cast("Literal['task_once', 'chat_full'] | None", kind),
            )
        except (OSError, TypeError, ValueError) as exc:
            self._unavailable = True
            raise FolderGrantsUnavailableError("native process admission is unavailable") from exc

    def end_native_process(self, reservation: NativeProcessReservation) -> None:
        """Keep opaque or unproven host work as a durable UNKNOWN barrier."""
        reservation.mark_unproven()

    @contextmanager
    def activation_guard(self) -> Generator[None, None, None]:
        """Serialize grant mutations with admission of unrestricted processes."""
        try:
            with self._native_admission.activation_guard(managed=lambda: self.is_managed):
                yield
        except NativeAdmissionUnknownError as exc:
            raise NativeProcessesUnknownError("native_processes_unknown") from exc
        except NativeAdmissionActiveError as exc:
            raise NativeProcessesActiveError("native_processes_active") from exc
        except (OSError, ValueError) as exc:
            self._unavailable = True
            raise FolderGrantsUnavailableError("native process admission is unavailable") from exc

    def store_fingerprint_matches(self, fingerprint: _Fingerprint | None) -> bool:
        """Check the writer's observed file identity without exposing it."""
        with self._lock:
            return fingerprint == self._fingerprint

    def publish_committed_snapshot(
        self,
        snapshot: FolderGrantSnapshot,
        fingerprint: _Fingerprint,
        activation_fingerprint: _Fingerprint,
    ) -> None:
        """Update the reader only after the store replacement is durable."""
        with self._lock:
            self._snapshot = snapshot
            self._fingerprint = fingerprint
            self._activation_fingerprint = activation_fingerprint
            self._unavailable = False


class FolderGrantStore:
    """Durable local writer plus the read-only snapshot consumed by tools."""

    def __init__(
        self,
        *,
        agent_home: Path,
        instance_root: Path | None = None,
        sessions_root: Path | None = None,
        path: Path | None = None,
    ) -> None:
        self.agent_home = agent_home.expanduser().resolve(strict=False)
        self.path = path or (get_webui_dir() / "folder-grants.json")
        self.lock_path = self.path.with_name("folder-grants.lock")
        self.activation_path = self.path.with_name(".folder-grants-active")
        instance = (instance_root or get_webui_dir().parent).expanduser().resolve(strict=False)
        sessions = (sessions_root or get_default_sessions_root()).expanduser().resolve(strict=False)
        config_path = get_config_path().expanduser().resolve(strict=False)
        webui_root = self.path.parent.expanduser().resolve(strict=False)
        from nanobot.context_repo.catalog import RepositoryCatalog

        catalog_root = RepositoryCatalog.for_workspace(
            self.agent_home,
            sessions_root=sessions,
        ).root.expanduser().resolve(strict=False)
        private_roots = tuple(
            _unique_paths(
                (
                    sessions,
                    instance / "credentials",
                    catalog_root,
                    webui_root,
                    config_path,
                )
            )
        )
        self.view = self._load_view(
            private_roots,
            instance_root=instance,
        )

    @classmethod
    def for_workspace(
        cls,
        workspace: Path,
        *,
        sessions_root: Path | None = None,
        path: Path | None = None,
    ) -> Self:
        """Return the shared instance authority using agent-home identity."""
        resolved_home = workspace.expanduser().resolve(strict=False)
        sessions = (sessions_root or get_default_sessions_root()).expanduser().resolve(strict=False)
        authority_path = path or (get_webui_dir() / "folder-grants.json")
        key = (
            str(authority_path.expanduser().resolve(strict=False)),
            str(resolved_home),
            str(sessions),
        )
        with _SHARED_STORES_LOCK:
            existing = _SHARED_STORES.get(key)
            if existing is not None:
                if not isinstance(existing, cls):
                    raise TypeError("folder grant store factory returned an incompatible owner")
                return existing
            store = cls(
                agent_home=resolved_home,
                sessions_root=sessions,
                path=authority_path,
            )
            _SHARED_STORES[key] = store
            return store

    def _load_view(
        self,
        private_roots: tuple[Path, ...],
        *,
        instance_root: Path,
    ) -> FolderGrantView:
        if not self.path.exists() and not self.activation_path.exists():
            return FolderGrantView(
                self.path,
                self.activation_path,
                FolderGrantSnapshot(False, 0, ()),
                None,
                None,
                agent_home=self.agent_home,
                private_roots=private_roots,
                instance_root=instance_root,
            )
        try:
            fingerprint = _fingerprint_file(self.path)
            activation_fingerprint = _fingerprint_activation(self.activation_path)
            if fingerprint is None or activation_fingerprint is None:
                raise ValueError("activated folder grant store is incomplete")
            data = _parse_store(self.path.read_bytes())
        except (OSError, ValueError):
            return FolderGrantView(
                self.path,
                self.activation_path,
                FolderGrantSnapshot(True, 0, ()),
                None,
                None,
                unavailable=True,
                agent_home=self.agent_home,
                private_roots=private_roots,
                instance_root=instance_root,
            )
        snapshot = FolderGrantSnapshot(data.enabled, data.revision, data.grants)
        return FolderGrantView(
            self.path,
            self.activation_path,
            snapshot,
            fingerprint,
            activation_fingerprint,
            agent_home=self.agent_home,
            private_roots=private_roots,
            instance_root=instance_root,
        )

    def snapshot(self) -> FolderGrantSnapshot:
        return self.view.snapshot()

    def upsert(
        self,
        raw: object,
        *,
        expected_revision: int,
    ) -> FolderGrantSnapshot:
        return self.upsert_with_result(raw, expected_revision=expected_revision).snapshot

    def upsert_with_result(
        self,
        raw: object,
        *,
        expected_revision: int,
    ) -> FolderGrantUpsertResult:
        revision = _validate_revision(expected_revision)
        submitted = self._validated_input(raw)
        grant = self._canonicalize_validated_input(submitted)
        snapshot = self._mutate(revision, upsert=grant)
        return FolderGrantUpsertResult(
            submitted_path=submitted.path,
            grant=grant,
            snapshot=snapshot,
        )

    def revoke(self, raw_path: object, *, expected_revision: int) -> FolderGrantSnapshot:
        revision = _validate_revision(expected_revision)
        path = _canonical_revoke_path(raw_path)
        return self._mutate(revision, revoke_path=path)

    def effective_grant_for(self, path: Path) -> FolderGrant | None:
        return self.view.effective_grant_for(path)

    @staticmethod
    def _validated_input(raw: object) -> FolderGrantInput:
        try:
            return FolderGrantInput.model_validate(raw)
        except Exception as exc:
            raise FolderGrantError(f"invalid folder grant: {exc}") from exc

    def _canonicalize_validated_input(self, requested: FolderGrantInput) -> FolderGrant:
        path = Path(requested.path).expanduser()
        if not path.is_absolute():
            raise FolderGrantError("folder path must be absolute")
        try:
            root = path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise FolderGrantError("folder path must be an existing directory") from exc
        if not root.is_dir():
            raise FolderGrantError("folder path must be an existing directory")
        canonical = str(root)
        if len(canonical) > 2048:
            raise FolderGrantError("canonical folder path is too long")
        if self.view.overlaps_private_root(root):
            raise FolderGrantError("folder grants cannot overlap private runtime storage")
        return FolderGrant.model_validate(
            {
                "path": canonical,
                "file_access": requested.file_access,
                "commands": requested.commands,
            }
        )

    def _mutate(
        self,
        expected_revision: int,
        *,
        upsert: FolderGrant | None = None,
        revoke_path: str | None = None,
    ) -> FolderGrantSnapshot:
        try:
            current = self.view.snapshot()
        except FolderGrantsUnavailableError:
            raise
        if current.revision != expected_revision:
            raise GrantRevisionConflictError("grants_changed")

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            with self.view.activation_guard():
                with FileLock(str(self.lock_path), timeout=5):
                    os.chmod(self.lock_path, 0o600)
                    latest_fingerprint = _fingerprint_file(self.path)
                    if not self.view.store_fingerprint_matches(latest_fingerprint):
                        raise FolderGrantsUnavailableError("folder grant store changed unexpectedly")
                    latest = self.view.snapshot()
                    if latest.revision != expected_revision:
                        raise GrantRevisionConflictError("grants_changed")

                    grants = {grant.path: grant for grant in latest.grants}
                    if upsert is not None:
                        grants[upsert.path] = upsert
                    elif revoke_path is not None:
                        grants.pop(revoke_path, None)
                    ordered = tuple(sorted(grants.values(), key=lambda grant: grant.path))
                    changed = ordered != latest.grants
                    activated = not latest.enabled
                    if not changed and not activated:
                        return latest
                    revision = latest.revision + (1 if changed else 0)
                    data = _GrantStoreData(
                        version=1,
                        enabled=True,
                        revision=revision,
                        grants=ordered,
                    )
                    if activated:
                        _atomic_write_activation(self.activation_path)
                    _atomic_write(self.path, data)
                    fingerprint = _fingerprint_file(self.path)
                    activation_fingerprint = _fingerprint_activation(self.activation_path)
                    if fingerprint is None or activation_fingerprint is None:
                        raise FolderGrantsUnavailableError("folder grants are unavailable")
                    snapshot = FolderGrantSnapshot(True, revision, ordered)
                    self.view.publish_committed_snapshot(snapshot, fingerprint, activation_fingerprint)
                    return snapshot
        except Timeout as exc:
            raise FolderGrantsUnavailableError("folder grant store is busy") from exc
        except (OSError, ValueError) as exc:
            if isinstance(exc, FolderGrantError):
                raise
            raise FolderGrantsUnavailableError("folder grants are unavailable") from exc


def _parse_store(raw: bytes) -> _GrantStoreData:
    if len(raw) > _MAX_GRANT_STORE_BYTES:
        raise ValueError("folder grant store is too large")
    return _GrantStoreData.model_validate_json(raw)


def _fingerprint_file(path: Path) -> _Fingerprint | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("folder grant store must be a regular file")
    if info.st_size > _MAX_GRANT_STORE_BYTES:
        raise ValueError("folder grant store is too large")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("folder grant store changed while opening")
        raw = stream.read(_MAX_GRANT_STORE_BYTES + 1)
    if len(raw) > _MAX_GRANT_STORE_BYTES:
        raise ValueError("folder grant store is too large")
    return _Fingerprint(
        device=opened.st_dev,
        inode=opened.st_ino,
        size=opened.st_size,
        modified_ns=opened.st_mtime_ns,
        digest=hashlib.sha256(raw).hexdigest(),
    )


def _fingerprint_activation(path: Path) -> _Fingerprint | None:
    fingerprint = _fingerprint_file(path)
    if fingerprint is None:
        return None
    if path.read_bytes() != b"nanobot-managed-folder-access-v1\n":
        raise ValueError("folder activation marker is invalid")
    return fingerprint


def _atomic_write_activation(path: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".folder-grants-active-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(b"nanobot-managed-folder-access-v1\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write(path: Path, data: _GrantStoreData) -> None:
    encoded = (data.model_dump_json(indent=2) + "\n").encode("utf-8")
    if len(encoded) > _MAX_GRANT_STORE_BYTES:
        raise FolderGrantError("folder grant store is too large")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".folder-grants-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_revision(value: object) -> int:
    if type(value) is not int or value < 0:
        raise FolderGrantError("expected_revision must be a non-negative integer")
    return value


def _canonical_revoke_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\0" in value or len(value) > 2048:
        raise FolderGrantError("path must be an absolute folder path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FolderGrantError("path must be absolute")
    return os.path.abspath(path)


def _root_identity_is_live(root: Path) -> bool:
    try:
        return root.is_dir() and root.resolve(strict=True) == root
    except (OSError, RuntimeError, ValueError):
        return False


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _overlaps(left: Path, right: Path) -> bool:
    return _is_under(left, right) or _is_under(right, left)


def _overlaps_system_control(path: Path) -> bool:
    return any(_overlaps(path, Path(root)) for root in ("/proc", "/sys", "/dev", "/run"))


def _unique_paths(paths: tuple[Path, ...]) -> list[Path]:
    return list(dict.fromkeys(path.expanduser().resolve(strict=False) for path in paths))


_SHARED_STORES: dict[tuple[str, str, str], FolderGrantStore] = {}
_SHARED_STORES_LOCK = threading.RLock()
