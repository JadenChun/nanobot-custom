"""Durable remote-access owner state (M07).

The store is independent of the gateway and agent. It holds only device
identity hashes, scopes, revocation state and owned private-entry metadata.
Plaintext refresh credentials are never persisted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

from nanobot.remote.protocol import REFRESH_IDLE_SECONDS, ContractError, validate_scopes

try:  # POSIX selected-OS boundary; Windows tests use the thread lock only.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

STATE_VERSION = 1
MAX_STATE_BYTES = 1024 * 1024
MAX_DEVICES = 256
MAX_OWNED_ENTRIES = 64
MAX_AUDIT_ENTRIES = 1000


class StoreError(RuntimeError):
    """Base class for durable store failures."""


class StoreVersionError(StoreError):
    """The state file declares an unsupported schema version."""


class StoreCorruptError(StoreError):
    """The state file cannot be parsed safely."""


class StoreUnavailableError(StoreError):
    """The owner state cannot be used without a fresh setup."""


@dataclass(frozen=True)
class RemoteDevice:
    installation_id: str
    label: str
    scopes: tuple[str, ...]
    created_at: float
    last_seen_at: float
    refresh_hash: str
    revoked_at: float | None = None

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None


@dataclass
class RemoteAccessState:
    version: int = STATE_VERSION
    installation_id: str = ""
    revision: int = 1
    enabled: bool = False
    devices: dict[str, RemoteDevice] = field(default_factory=dict)
    owned_entries: list[dict[str, Any]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class RemoteAccessStore:
    """Atomic, mode-restricted state under ``<instance-root>/remote-access``."""

    def __init__(
        self, root: Path | str | None = None, *, clock: Callable[[], float] = time.time
    ) -> None:
        if root is None:
            from nanobot.config.paths import get_data_dir

            root = get_data_dir() / "remote-access"
        self.root = Path(root).expanduser()
        self.state_path = self.root / "state.json"
        self.lock_path = self.root / "state.lock"
        self.admin_token_path = self.root / "admin.token"
        self.thread_lock = threading.RLock()
        self.clock = clock

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def _locked(self) -> "_StoreLock":
        return _StoreLock(self)

    def load(self) -> RemoteAccessState | None:
        if self.state_path.is_symlink():
            raise StoreCorruptError("state file must not be a symlink")
        if not self.state_path.exists():
            return None
        if self.state_path.stat().st_size > MAX_STATE_BYTES:
            raise StoreCorruptError("state file exceeds the bounded size")
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreCorruptError("state file is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise StoreCorruptError("state root must be an object")
        raw_dict = cast(dict[str, Any], raw)
        version = raw_dict.get("version")
        if version != STATE_VERSION:
            raise StoreVersionError(f"unsupported remote-access state version: {version!r}")
        return self._from_dict(raw_dict)

    def load_or_create(self) -> RemoteAccessState:
        with self._locked():
            state = self.load()
            if state is None:
                state = RemoteAccessState(installation_id=str(uuid.uuid4()))
                self._save_locked(state)
            return state

    def save(self, state: RemoteAccessState) -> None:
        with self._locked():
            self._save_locked(state)

    def _save_locked(self, state: RemoteAccessState) -> None:
        if state.version != STATE_VERSION:
            raise StoreVersionError(f"unsupported remote-access state version: {state.version}")
        if state.revision < 1:
            raise StoreError("revision must be positive")
        if len(state.devices) > MAX_DEVICES or len(state.owned_entries) > MAX_OWNED_ENTRIES:
            raise StoreError("state exceeds the bounded device/entry count")
        state.updated_at = self.clock()
        payload = json.dumps(self._to_dict(state), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        self.ensure_root()
        if self.state_path.is_symlink():
            raise StoreError("state file must not be a symlink")
        fd, temp_name = tempfile.mkstemp(prefix=".state-", dir=self.root)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.state_path)
            os.chmod(self.state_path, 0o600)
            self._sync_dir()
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _sync_dir(self) -> None:
        handle = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)

    def set_enabled(self, enabled: bool) -> RemoteAccessState:
        with self._locked():
            state = self._require_state()
            state.enabled = bool(enabled)
            state.revision += 1
            self._save_locked(state)
            return state

    def add_device(
        self,
        *,
        installation_id: str,
        label: str,
        scopes: tuple[str, ...],
        refresh_token: str,
    ) -> RemoteDevice:
        with self._locked():
            state = self._require_state()
            if installation_id in state.devices:
                raise StoreError("installation already exists")
            if len(state.devices) >= MAX_DEVICES:
                raise StoreError("device limit reached")
            try:
                validate_scopes(scopes)
            except ContractError as exc:
                raise StoreError(str(exc)) from exc
            device = RemoteDevice(
                installation_id=installation_id,
                label=label[:64],
                scopes=tuple(scopes),
                created_at=self.clock(),
                last_seen_at=self.clock(),
                refresh_hash=hash_refresh_token(refresh_token),
            )
            state.devices[installation_id] = device
            state.revision += 1
            self._save_locked(state)
            return device

    def get_device(self, installation_id: str) -> RemoteDevice | None:
        state = self.load()
        return state.devices.get(installation_id) if state else None

    def update_scopes(self, installation_id: str, scopes: tuple[str, ...]) -> RemoteDevice:
        with self._locked():
            state = self._require_state()
            device = state.devices.get(installation_id)
            if device is None or device.revoked:
                raise StoreError("device is not active")
            try:
                validate_scopes(scopes)
            except ContractError as exc:
                raise StoreError(str(exc)) from exc
            updated = RemoteDevice(
                installation_id=device.installation_id,
                label=device.label,
                scopes=tuple(scopes),
                created_at=device.created_at,
                last_seen_at=device.last_seen_at,
                refresh_hash=device.refresh_hash,
                revoked_at=device.revoked_at,
            )
            state.devices[installation_id] = updated
            state.revision += 1
            self._save_locked(state)
            return updated

    def rename_device(self, installation_id: str, label: str) -> RemoteDevice:
        with self._locked():
            state = self._require_state()
            device = state.devices.get(installation_id)
            if device is None or device.revoked:
                raise StoreError("device is not active")
            updated = RemoteDevice(
                installation_id=device.installation_id,
                label=label[:64],
                scopes=device.scopes,
                created_at=device.created_at,
                last_seen_at=device.last_seen_at,
                refresh_hash=device.refresh_hash,
                revoked_at=device.revoked_at,
            )
            state.devices[installation_id] = updated
            state.revision += 1
            self._save_locked(state)
            return updated

    def rotate_refresh(self, installation_id: str, refresh_token: str) -> RemoteDevice:
        with self._locked():
            state = self._require_state()
            device = state.devices.get(installation_id)
            if device is None or device.revoked:
                raise StoreError("device is not active")
            updated = RemoteDevice(
                installation_id=device.installation_id,
                label=device.label,
                scopes=device.scopes,
                created_at=device.created_at,
                last_seen_at=self.clock(),
                refresh_hash=hash_refresh_token(refresh_token),
                revoked_at=device.revoked_at,
            )
            state.devices[installation_id] = updated
            state.revision += 1
            self._save_locked(state)
            return updated

    def revoke_device(self, installation_id: str) -> RemoteDevice:
        with self._locked():
            state = self._require_state()
            device = state.devices.get(installation_id)
            if device is None:
                raise StoreError("device is unknown")
            updated = RemoteDevice(
                installation_id=device.installation_id,
                label=device.label,
                scopes=device.scopes,
                created_at=device.created_at,
                last_seen_at=device.last_seen_at,
                refresh_hash=device.refresh_hash,
                revoked_at=device.revoked_at or self.clock(),
            )
            state.devices[installation_id] = updated
            state.revision += 1
            self._save_locked(state)
            return updated

    def validate_refresh(self, installation_id: str, refresh_token: str) -> RemoteDevice | None:
        state = self.load()
        if state is None or not state.enabled:
            return None
        device = state.devices.get(installation_id)
        if device is None or device.revoked:
            return None
        now = self.clock()
        if now - device.last_seen_at > REFRESH_IDLE_SECONDS:
            return None
        if not hmac.compare_digest(device.refresh_hash, hash_refresh_token(refresh_token)):
            return None
        return device

    def admin_token(self) -> str:
        self.ensure_root()
        if self.admin_token_path.is_symlink():
            raise StoreError("admin token must not be a symlink")
        if self.admin_token_path.exists():
            token = self.admin_token_path.read_text(encoding="ascii").strip()
            if token:
                return token
        token = secrets.token_urlsafe(32)
        fd, temp_name = tempfile.mkstemp(prefix=".admin-", dir=self.root)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(token + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.admin_token_path)
            os.chmod(self.admin_token_path, 0o600)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return token

    def check_admin_token(self, token: str | None) -> bool:
        expected = self.admin_token()
        return bool(token) and hmac.compare_digest(expected, token or "")

    def owned_entries(self, entries: list[dict[str, Any]]) -> RemoteAccessState:
        with self._locked():
            state = self._require_state()
            if len(entries) > MAX_OWNED_ENTRIES:
                raise StoreError("owned-entry limit reached")
            cleaned: list[dict[str, Any]] = []
            for entry in entries:
                kind = entry.get("kind")
                value = entry.get("value")
                if not isinstance(kind, str) or not isinstance(value, str):
                    raise StoreError("owned entries require string kind/value")
                if len(kind) > 32 or len(value) > 512:
                    raise StoreError("owned entry is too large")
                cleaned.append({"kind": kind, "value": value, "added_at": time.time()})
            state.owned_entries = cleaned
            state.revision += 1
            self._save_locked(state)
            return state

    def record_audit(
        self,
        *,
        installation_id: str,
        operation: str,
        decision: str,
        request_id: str | None = None,
        actor_installation_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a bounded, sanitized device-operation audit record.

        Never stores credentials, payloads or native references. Audit writes
        do not change the authority revision, so they cannot invalidate an
        already-issued admission.
        """
        with self._locked():
            state = self._require_state()
            entry = {
                "at": self.clock(),
                "installation_id": _bounded_text(installation_id, 128),
                "operation": _bounded_text(operation, 64),
                "decision": _bounded_text(decision, 32),
                "request_id": _bounded_text(request_id, 128) if request_id else None,
                "actor_installation_id": (
                    _bounded_text(actor_installation_id, 128) if actor_installation_id else None
                ),
            }
            state.audit = [*state.audit[-MAX_AUDIT_ENTRIES + 1 :], entry]
            self._save_locked(state)
            return entry

    def snapshot(self) -> dict[str, Any]:
        state = self.load_or_create()
        return {
            "version": state.version,
            "installation_id": state.installation_id,
            "revision": state.revision,
            "enabled": state.enabled,
            "devices": [
                {
                    "installation_id": device.installation_id,
                    "label": device.label,
                    "scopes": list(device.scopes),
                    "status": "revoked" if device.revoked else "active",
                    "created_at": device.created_at,
                    "last_seen_at": device.last_seen_at,
                }
                for device in state.devices.values()
            ],
            "owned_entries": list(state.owned_entries),
            "audit": list(state.audit[-500:]),
        }

    def _require_state(self) -> RemoteAccessState:
        state = self.load()
        if state is None:
            raise StoreUnavailableError("remote-access state is not initialized")
        return state

    @staticmethod
    def _to_dict(state: RemoteAccessState) -> dict[str, Any]:
        return {
            "version": state.version,
            "installation_id": state.installation_id,
            "revision": state.revision,
            "enabled": state.enabled,
            "updated_at": state.updated_at,
            "devices": {
                installation_id: {
                    **asdict(device),
                    "scopes": list(device.scopes),
                }
                for installation_id, device in state.devices.items()
            },
            "owned_entries": list(state.owned_entries),
            "audit": list(state.audit[-500:]),
        }

    @staticmethod
    def _from_dict(raw: dict[str, Any]) -> RemoteAccessState:
        installation_id = raw.get("installation_id")
        revision = raw.get("revision")
        if not isinstance(installation_id, str) or not installation_id:
            raise StoreCorruptError("installation_id is missing")
        if not isinstance(revision, int) or revision < 1:
            raise StoreCorruptError("revision is invalid")
        raw_devices = raw.get("devices", {})
        if not isinstance(raw_devices, dict):
            raise StoreCorruptError("devices are invalid")
        devices_raw = cast(dict[str, Any], raw_devices)
        if len(devices_raw) > MAX_DEVICES:
            raise StoreCorruptError("devices are invalid")
        devices: dict[str, RemoteDevice] = {}
        for key, raw_value in devices_raw.items():
            if not isinstance(raw_value, dict):
                raise StoreCorruptError("device record is invalid")
            value = cast(dict[str, Any], raw_value)
            try:
                scopes = tuple(value.get("scopes", ()))
                validate_scopes(scopes)
                device = RemoteDevice(
                    installation_id=str(value["installation_id"]),
                    label=str(value["label"])[:64],
                    scopes=scopes,
                    created_at=float(value["created_at"]),
                    last_seen_at=float(value["last_seen_at"]),
                    refresh_hash=str(value["refresh_hash"]),
                    revoked_at=(
                        float(value["revoked_at"])
                        if value.get("revoked_at") is not None
                        else None
                    ),
                )
            except (KeyError, TypeError, ValueError, ContractError) as exc:
                raise StoreCorruptError("device record is malformed") from exc
            if device.installation_id != key:
                raise StoreCorruptError("device key does not match its record")
            devices[key] = device
        raw_entries = raw.get("owned_entries", [])
        if not isinstance(raw_entries, list):
            raise StoreCorruptError("owned entries are invalid")
        entries_raw = cast(list[Any], raw_entries)
        if len(entries_raw) > MAX_OWNED_ENTRIES:
            raise StoreCorruptError("owned entries are invalid")
        entries: list[dict[str, Any]] = []
        for raw_entry in entries_raw:
            if not isinstance(raw_entry, dict):
                raise StoreCorruptError("owned entry is invalid")
            entry = cast(dict[str, Any], raw_entry)
            entries.append(
                {
                    "kind": str(entry.get("kind", ""))[:32],
                    "value": str(entry.get("value", ""))[:512],
                    "added_at": float(entry.get("added_at", 0)),
                }
            )
        raw_audit = raw.get("audit", [])
        if not isinstance(raw_audit, list):
            raise StoreCorruptError("audit records are invalid")
        audit: list[dict[str, Any]] = []
        for item in cast(list[Any], raw_audit[-MAX_AUDIT_ENTRIES:]):
            if not isinstance(item, dict):
                continue
            record = cast(dict[str, Any], item)
            at = record.get("at")
            if not isinstance(at, (int, float)):
                continue
            audit.append(
                {
                    "at": float(at),
                    "installation_id": _bounded_text(record.get("installation_id"), 128),
                    "operation": _bounded_text(record.get("operation"), 64),
                    "decision": _bounded_text(record.get("decision"), 32),
                    "request_id": (
                        _bounded_text(record.get("request_id"), 128)
                        if isinstance(record.get("request_id"), str)
                        else None
                    ),
                    "actor_installation_id": (
                        _bounded_text(record.get("actor_installation_id"), 128)
                        if isinstance(record.get("actor_installation_id"), str)
                        else None
                    ),
                }
            )
        updated_at = raw.get("updated_at", time.time())
        return RemoteAccessState(
            version=STATE_VERSION,
            installation_id=installation_id,
            revision=revision,
            enabled=bool(raw.get("enabled", False)),
            devices=devices,
            owned_entries=entries,
            audit=audit,
            updated_at=float(updated_at) if isinstance(updated_at, (int, float)) else time.time(),
        )


class _StoreLock:
    def __init__(self, store: RemoteAccessStore) -> None:
        self.store = store
        self.fd: int | None = None

    def __enter__(self) -> None:
        self.store.ensure_root()
        self.store.thread_lock.acquire()
        if fcntl is not None:
            self.fd = os.open(self.store.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(self.fd, fcntl.LOCK_EX)

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self.fd is not None and fcntl is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
        self.store.thread_lock.release()


def _bounded_text(value: object, limit: int) -> str:
    text = str(value or "")
    text = "".join(char for char in text if char.isprintable())
    return text[:limit]
