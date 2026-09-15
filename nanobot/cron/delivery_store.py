"""Dormant, durable cron delivery outbox authority.

Package E2 records the delivery work implied by one completed occurrence.  It
does not know how to send a message and deliberately has no channel, provider,
bus, scheduler, or service integration.  A later runtime may use the
process-local claim returned by this module as the sole authority to invoke a
sender after the corresponding state has been durably fenced.

The store follows the private POSIX descriptor discipline used by the
occurrence ledger.  Every final object is a single-link ``0600`` regular file
under a private ``0700`` root, all mutations are descriptor-relative atomic
replacements, and file/root bindings are checked across every durability
boundary.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Literal

from nanobot.cron.occurrence_store import CronOccurrenceStore, OccurrenceRecord

try:  # Keep imports usable on Windows; construction fails closed below.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - qualified through the platform gate
    _fcntl = None


_SCHEMA_VERSION = 1
_KIND = "cron_delivery_outbox_v1"
_MAX_RECORD_BYTES = 1_048_576
_MAX_DESTINATIONS = 256
_MAX_ATTEMPTS = 256
_MAX_IDENTIFIER_BYTES = 512
_MAX_EVIDENCE_BYTES = 512
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_TEMP_ENTRY_RE = re.compile(r"\.[0-9a-f]{64}\.[0-9a-f]{32}\.tmp")
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)

_DESTINATION_STATES = frozenset(
    {"pending", "claimed", "sending", "uncertain", "delivered", "failed"}
)
_ATTEMPT_STATES = frozenset(
    {
        "claimed",
        "sending",
        "confirmed_delivered",
        "confirmed_not_sent",
        "permanent_not_sent",
        "unknown",
        "confirmed_compensated",
        "abandon",
    }
)
_RESOLUTION_STATES = frozenset(
    {"confirmed_delivered", "confirmed_not_sent", "confirmed_compensated", "abandon"}
)
_OUTCOME_STATES = frozenset(
    {"confirmed_delivered", "confirmed_not_sent", "permanent_not_sent", "unknown"}
)
_TRANSITION_OPERATIONS = frozenset({"claim", "mark_sending", "settle", "reconcile"})


class DeliveryOutboxError(RuntimeError):
    """Base error for the private cron delivery outbox."""


class DeliveryOutboxPersistenceError(DeliveryOutboxError):
    """A filesystem operation could not establish the requested durability."""


class DeliveryOutboxCorruptError(DeliveryOutboxError):
    """An owned on-disk object is malformed, unsafe, or ambiguously replaced."""


class DeliveryOutboxConflictError(DeliveryOutboxError):
    """An immutable input, action identity, or transition conflicts."""


class DeliveryOutboxStaleRevisionError(DeliveryOutboxConflictError):
    """A transition was based on a record revision that is no longer current."""


class DeliveryOutboxOwnershipError(DeliveryOutboxError):
    """A process-local claim is reconstructed, stale, foreign, or consumed."""


class DeliveryOutboxReconciliationRequiredError(DeliveryOutboxError):
    """A destination is already claimed or uncertain and cannot auto-execute."""


class DeliveryOutboxNotFoundError(DeliveryOutboxError):
    """The requested outbox record does not exist."""


class DeliveryOutboxUnsupportedError(DeliveryOutboxConflictError):
    """The completed occurrence uses a delivery shape outside E2's scope."""


@dataclass(frozen=True)
class DeliveryAttempt:
    """Detached history for one durable destination attempt."""

    attempt: int
    owner: str
    status: str
    action_id: str
    request_digest: str
    receipt: dict[str, Any] | None
    resolution: dict[str, Any] | None

    @property
    def attempt_number(self) -> int:
        return self.attempt


@dataclass(frozen=True)
class DeliveryDestination:
    """Detached frozen destination and its ordered attempt history."""

    destination_id: str
    channel: str
    to: str
    status: str
    attempts: tuple[DeliveryAttempt, ...]

    @property
    def target(self) -> tuple[str, str]:
        return self.channel, self.to


@dataclass(frozen=True)
class DeliveryOutboxRecord:
    """Detached durable outbox view; it never grants sender authority."""

    store_namespace: str
    outbox_id: str
    occurrence_id: str
    run_id: str
    immutable_input: dict[str, Any]
    destinations: tuple[DeliveryDestination, ...]
    record_revision: int

    @property
    def revision(self) -> int:
        return self.record_revision

    @property
    def status(self) -> str:
        if not self.destinations:
            return "no_work"
        states = {destination.status for destination in self.destinations}
        if states & {"claimed", "sending", "uncertain"}:
            return "reconciliation-required"
        if states == {"delivered"}:
            return "delivered"
        if states == {"failed"}:
            return "failed"
        if states == {"pending"}:
            return "pending"
        return "partial"

    @property
    def aggregate_status(self) -> str:
        return self.status

    @property
    def reconciliation_required(self) -> bool:
        return self.status == "reconciliation-required"

    @property
    def provenance(self) -> dict[str, Any] | None:
        if not self.immutable_input["provenance_available"]:
            return None
        return _copy_json_mapping(self.immutable_input["provenance"], label="provenance")

    @property
    def provenance_available(self) -> bool:
        return self.immutable_input["provenance_available"]

    @property
    def completion(self) -> dict[str, Any]:
        return _copy_json_mapping(self.immutable_input["completion"], label="completion")

    @property
    def delivery(self) -> dict[str, Any]:
        return _copy_json_mapping(self.immutable_input["delivery"], label="delivery")

    @property
    def destination_policy(self) -> dict[str, Any]:
        return _copy_json_mapping(self.immutable_input["destination_policy"], label="destination policy")

    @property
    def occurrence_identity(self) -> dict[str, Any]:
        return _copy_json_mapping(self.immutable_input["occurrence_identity"], label="occurrence identity")

    @property
    def job_snapshot(self) -> dict[str, Any]:
        return _copy_json_mapping(self.immutable_input["job_snapshot"], label="job snapshot")

    @property
    def messages(self) -> tuple[dict[str, Any], ...]:
        return tuple(_copy_json_mapping(message, label="delivery message") for message in self.delivery["messages"])


@dataclass(frozen=True)
class DeliveryClaim:
    """Sealed process-local authority for one reserved destination attempt."""

    outbox_id: str
    destination_id: str
    attempt: int
    owner: str
    expected_revision: int
    record_revision: int
    action_id: str
    request_digest: str

    @property
    def attempt_number(self) -> int:
        return self.attempt


@dataclass(frozen=True)
class _RootBinding:
    store: CronDeliveryOutboxStore
    fds: tuple[int, ...]
    parts: tuple[str, ...]
    identities: tuple[tuple[int, int], ...]

    @property
    def root_fd(self) -> int:
        return self.fds[-1]

    def assert_bound(self) -> None:
        self.store._assert_path_binding(self)

    def close(self) -> None:
        for fd in reversed(self.fds):
            try:
                os.close(fd)
            except OSError:
                pass


@dataclass(frozen=True)
class _HeldOutbox:
    store: CronDeliveryOutboxStore
    root_binding: _RootBinding
    lock_fd: int
    root_identity: tuple[int, int]
    lock_identity: tuple[int, int]

    @property
    def root_fd(self) -> int:
        return self.root_binding.root_fd

    def assert_bound(self) -> None:
        self.store._assert_binding(self)


@dataclass(frozen=True)
class _RecordProof:
    inode: tuple[int, int]
    signature: tuple[int, int, int, int, int, int, int]
    content: tuple[int, str]


@dataclass
class _IssuedClaim:
    claim: DeliveryClaim
    process_id: int
    store_identity: int
    phase: Literal["claimed", "sending"]
    current_revision: int


def _inode(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _private(info: os.stat_result, *, directory: bool, links: int = 1) -> None:
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or (not directory and info.st_nlink != links)
    ):
        raise DeliveryOutboxCorruptError(
            "delivery outbox object is not private or has an unexpected type"
        )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(data: bytes, *, label: str) -> object:
    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=_no_duplicate_object, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise DeliveryOutboxCorruptError(f"{label} is not strict JSON") from exc


def _assert_json_value(value: object, *, error: type[DeliveryOutboxError], label: str) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            raise error(f"{label} contains a non-finite JSON number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise error(f"{label} contains a non-string JSON key")
            _assert_json_value(item, error=error, label=label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_value(item, error=error, label=label)
        return
    raise error(f"{label} contains a non-JSON value")


def _canonical_json(
    value: object, *, error: type[DeliveryOutboxError] = DeliveryOutboxConflictError, label: str = "value"
) -> str:
    _assert_json_value(value, error=error, label=label)
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError, RecursionError) as exc:
        raise error(f"{label} must contain finite JSON data") from exc


def _copy_json_value(
    value: object, *, label: str, error: type[DeliveryOutboxError] = DeliveryOutboxConflictError
) -> object:
    encoded = _canonical_json(value, error=error, label=label).encode("utf-8")
    try:
        copied = _strict_json_loads(encoded, label=label)
    except DeliveryOutboxCorruptError as exc:
        raise error(f"{label} is not canonical finite JSON") from exc
    return copied


def _copy_json_mapping(
    value: Mapping[str, Any], *, label: str, error: type[DeliveryOutboxError] = DeliveryOutboxConflictError
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{label} must be a JSON object")
    copied = _copy_json_value(value, label=label, error=error)
    if not isinstance(copied, dict):  # pragma: no cover - mappings encode as objects
        raise error(f"{label} must be a JSON object")
    return copied


def _text(value: object, *, label: str, maximum: int = _MAX_IDENTIFIER_BYTES) -> str:
    if not isinstance(value, str) or not value:
        raise DeliveryOutboxConflictError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > maximum:
        raise DeliveryOutboxConflictError(f"{label} exceeds the {maximum}-byte limit")
    return value


def _text_corrupt(value: object, *, label: str, maximum: int = _MAX_IDENTIFIER_BYTES) -> str:
    if not isinstance(value, str) or not value:
        raise DeliveryOutboxCorruptError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > maximum:
        raise DeliveryOutboxCorruptError(f"{label} exceeds the {maximum}-byte limit")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0, corrupt: bool = False) -> int:
    error = DeliveryOutboxCorruptError if corrupt else DeliveryOutboxConflictError
    if type(value) is not int or value < minimum:
        raise error(f"{label} must be an integer at least {minimum}")
    return value


def _digest(value: object, *, label: str, corrupt: bool = False) -> str:
    error = DeliveryOutboxCorruptError if corrupt else DeliveryOutboxConflictError
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise error(f"{label} must be a lowercase SHA-256 digest")
    return value


def _content_signature(data: bytes) -> tuple[int, str]:
    """Return a bounded, deterministic signature for one private file body."""

    return len(data), hashlib.sha256(data).hexdigest()


def _integrity(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(body), error=DeliveryOutboxCorruptError, label="outbox record").encode("utf-8")).hexdigest()


def _outbox_id(namespace: str, occurrence_id: str) -> str:
    payload = _canonical_json({"store_namespace": namespace, "occurrence_id": occurrence_id}).encode("utf-8")
    return hashlib.sha256(b"nanobot-cron-delivery-outbox-v1\0" + payload).hexdigest()


def _destination_id(outbox_id: str, channel: str, to: str) -> str:
    payload = _canonical_json({"outbox_id": outbox_id, "channel": channel, "to": to}).encode("utf-8")
    return hashlib.sha256(b"nanobot-cron-delivery-destination-v1\0" + payload).hexdigest()


def _action_id(outbox_id: str, destination_id: str, attempt: int) -> str:
    payload = _canonical_json(
        {"outbox_id": outbox_id, "destination_id": destination_id, "attempt": attempt}
    ).encode("utf-8")
    return hashlib.sha256(b"nanobot-cron-delivery-action-v1\0" + payload).hexdigest()


def _request_digest(
    outbox_id: str, destination_id: str, attempt: int, messages: list[dict[str, Any]]
) -> str:
    payload = _canonical_json(
        {
            "outbox_id": outbox_id,
            "destination_id": destination_id,
            "attempt": attempt,
            "messages": messages,
        }
    ).encode("utf-8")
    return hashlib.sha256(b"nanobot-cron-delivery-request-v1\0" + payload).hexdigest()


def _validate_delivery_envelope(
    value: object, *, error: type[DeliveryOutboxError]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"version", "deliver", "messages"}:
        raise error("delivery envelope must have exactly version, deliver, and messages fields")
    if type(value["version"]) is not int or value["version"] != 1:
        raise error("delivery envelope version must be integer 1")
    if type(value["deliver"]) is not bool:
        raise error("delivery envelope deliver must be a boolean")
    messages = value["messages"]
    if not isinstance(messages, list):
        raise error("delivery envelope messages must be a list")
    copied_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"content", "media"}:
            raise error("delivery messages must have exactly content and media fields")
        if not isinstance(message["content"], str):
            raise error("delivery message content must be text")
        if not isinstance(message["media"], list) or message["media"] != []:
            raise DeliveryOutboxUnsupportedError("delivery messages must have empty media")
        copied_messages.append(
            _copy_json_mapping(message, label="delivery message", error=error)
        )
    copied = _copy_json_mapping(value, label="delivery envelope", error=error)
    copied["messages"] = copied_messages
    return copied


def _validate_occurrence_identity(
    value: object, *, namespace: str, error: type[DeliveryOutboxError]
) -> dict[str, Any]:
    expected = {"store_namespace", "job_id", "job_revision", "scheduled_fire_time_ms"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise error("occurrence identity has an invalid schema")
    if value["store_namespace"] != namespace:
        raise error("occurrence belongs to a different outbox namespace")
    if not isinstance(value["job_id"], str) or not value["job_id"]:
        raise error("occurrence identity job_id must be a non-empty string")
    if isinstance(value["job_revision"], bool) or not isinstance(value["job_revision"], (str, int)):
        raise error("occurrence identity job_revision has an invalid type")
    if isinstance(value["job_revision"], str) and not value["job_revision"]:
        raise error("occurrence identity job_revision must not be empty")
    if type(value["scheduled_fire_time_ms"]) is not int:
        raise error("occurrence identity scheduled_fire_time_ms must be an integer")
    return _copy_json_mapping(value, label="occurrence identity", error=error)


def _derive_destinations(
    *,
    policy: Mapping[str, Any],
    deliver: bool,
    messages: list[dict[str, Any]],
    outbox_id: str,
    error: type[DeliveryOutboxError],
) -> list[dict[str, Any]]:
    """Apply the established CronPayload ordering and exact-string dedupe."""
    if not deliver or not messages:
        return []
    primary_to = policy.get("to")
    primary_channel = policy.get("channel")
    candidates: list[tuple[object, object]] = []
    if primary_to:
        candidates.append(("cli" if not primary_channel else primary_channel, primary_to))
    additional = policy.get("additionalDestinations", policy.get("additional_destinations", []))
    if additional is None:
        additional = []
    if not isinstance(additional, list):
        raise error("destination policy additionalDestinations must be a list")
    for item in additional:
        if not isinstance(item, Mapping):
            raise error("destination policy contains an invalid additional destination")
        candidates.append((item.get("channel"), item.get("to")))

    destinations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for channel_value, to_value in candidates:
        if not channel_value or not to_value:
            continue
        if not isinstance(channel_value, str) or not isinstance(to_value, str):
            raise error("destination policy channel and to must be strings")
        if len(channel_value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
            raise error("destination channel exceeds the 512-byte limit")
        if len(to_value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
            raise error("destination to exceeds the 512-byte limit")
        pair = (channel_value, to_value)
        if pair in seen:
            continue
        seen.add(pair)
        destinations.append(
            {
                "destination_id": _destination_id(outbox_id, channel_value, to_value),
                "channel": channel_value,
                "to": to_value,
                "status": "pending",
                "attempts": [],
            }
        )
    if len(destinations) > _MAX_DESTINATIONS:
        raise error(f"outbox has more than {_MAX_DESTINATIONS} destinations")
    return destinations


def _destination_state(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "pending"
    status = attempts[-1]["status"]
    if status == "claimed":
        return "claimed"
    if status == "sending":
        return "sending"
    if status == "unknown":
        return "uncertain"
    if status == "confirmed_delivered":
        return "delivered"
    if status == "confirmed_not_sent":
        return "pending"
    if status in {"permanent_not_sent", "abandon"}:
        return "failed"
    if status == "confirmed_compensated":
        return "pending"
    raise DeliveryOutboxCorruptError("attempt has an impossible state")


class CronDeliveryOutboxStore:
    """Private one-record-per-completed-occurrence delivery outbox."""

    def __init__(self, root: Path, *, namespace: str):
        self._require_qualified_platform()
        root = Path(root)
        if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
            raise ValueError("delivery outbox root must be an absolute non-root path without '..'")
        self.root = root
        if not isinstance(namespace, str) or not namespace or len(namespace.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
            raise ValueError("namespace must be a non-empty string no longer than 512 UTF-8 bytes")
        self.namespace = namespace
        self._root_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._default_owner = uuid.uuid4().hex
        self._issued_claims: dict[int, _IssuedClaim] = {}
        with self._root_fd(create=True) as root_fd:
            self._root_identity = _inode(os.fstat(root_fd))
        self._validate_existing_records()

    @staticmethod
    def _require_qualified_platform() -> None:
        required_calls = {os.open, os.mkdir, os.stat, os.unlink, os.rename}
        if (
            os.name != "posix"
            or _fcntl is None
            or not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
            or not required_calls <= os.supports_dir_fd
        ):
            raise DeliveryOutboxPersistenceError(
                "private cron delivery outbox requires qualified POSIX descriptor and flock support"
            )

    @staticmethod
    def _dir_flags() -> int:
        return os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_NONBLOCK

    def _open_root_binding(self, *, create: bool = False) -> _RootBinding:
        fds: list[int] = []
        try:
            current_fd = os.open(self.root.anchor, self._dir_flags())
            fds.append(current_fd)
            for part in self.root.parts[1:]:
                child: int | None = None
                created = False
                try:
                    try:
                        child = os.open(part, self._dir_flags(), dir_fd=current_fd)
                    except FileNotFoundError:
                        if not create:
                            raise DeliveryOutboxPersistenceError("private delivery outbox root is unavailable") from None
                        try:
                            os.mkdir(part, 0o700, dir_fd=current_fd)
                        except FileExistsError:
                            pass
                        else:
                            created = True
                        child = os.open(part, self._dir_flags(), dir_fd=current_fd)
                    child_info = os.fstat(child)
                    entry = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                    if not stat.S_ISDIR(child_info.st_mode) or _inode(child_info) != _inode(entry):
                        raise DeliveryOutboxCorruptError("delivery outbox root component binding changed")
                    if create:
                        if created:
                            os.fchmod(child, 0o700)
                        os.fsync(child)
                        os.fsync(current_fd)
                        rebound = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                        if _inode(rebound) != _inode(child_info):
                            raise DeliveryOutboxCorruptError("delivery outbox root component binding changed")
                    elif created:
                        os.fchmod(child, 0o700)
                    fds.append(child)
                    current_fd = child
                    child = None
                finally:
                    if child is not None:
                        try:
                            os.close(child)
                        except OSError:
                            pass
            info = os.fstat(current_fd)
            _private(info, directory=True)
            if self._root_identity is not None and _inode(info) != self._root_identity:
                raise DeliveryOutboxCorruptError("delivery outbox root binding changed")
            return _RootBinding(
                self,
                tuple(fds),
                tuple(self.root.parts[1:]),
                tuple(_inode(os.fstat(fd)) for fd in fds),
            )
        except DeliveryOutboxError:
            for fd in reversed(fds):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        except OSError as exc:
            for fd in reversed(fds):
                try:
                    os.close(fd)
                except OSError:
                    pass
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise DeliveryOutboxCorruptError(
                    "delivery outbox root must be a private directory, never a link"
                ) from exc
            raise DeliveryOutboxPersistenceError("private delivery outbox root is unavailable") from exc

    @contextmanager
    def _root_binding(self, *, create: bool = False) -> Generator[_RootBinding, None, None]:
        binding = self._open_root_binding(create=create)
        try:
            yield binding
        finally:
            binding.close()

    @contextmanager
    def _root_fd(self, *, create: bool = False) -> Generator[int, None, None]:
        with self._root_binding(create=create) as binding:
            yield binding.root_fd

    def _open_lock(self, root_fd: int) -> int:
        flags = os.O_RDWR | _O_NOFOLLOW | _O_NONBLOCK
        lock_fd: int | None = None
        created = False
        try:
            for _ in range(8):
                try:
                    lock_fd = os.open(".lock", flags, dir_fd=root_fd)
                    break
                except FileNotFoundError:
                    try:
                        lock_fd = os.open(".lock", flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=root_fd)
                    except FileExistsError:
                        continue
                    created = True
                    break
            if lock_fd is None:
                raise DeliveryOutboxPersistenceError("delivery outbox lock could not be established")
            if created:
                os.fchmod(lock_fd, 0o600)
                os.fsync(lock_fd)
                os.fsync(root_fd)
            _private(os.fstat(lock_fd), directory=False)
            current = _inode(os.fstat(lock_fd))
            if self._lock_identity is not None and current != self._lock_identity:
                raise DeliveryOutboxCorruptError("delivery outbox lock binding changed")
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX)
            return lock_fd
        except DeliveryOutboxError:
            if lock_fd is not None:
                os.close(lock_fd)
            raise
        except OSError as exc:
            if lock_fd is not None:
                os.close(lock_fd)
            if exc.errno == errno.ELOOP:
                raise DeliveryOutboxCorruptError("delivery outbox lock must not be a symlink") from exc
            raise DeliveryOutboxPersistenceError("could not acquire delivery outbox lock") from exc

    def _assert_path_binding(self, binding: _RootBinding) -> None:
        try:
            if len(binding.fds) != len(self.root.parts) or len(binding.parts) != len(self.root.parts) - 1:
                raise DeliveryOutboxCorruptError("delivery outbox root path binding changed")
            for fd, expected in zip(binding.fds, binding.identities):
                if _inode(os.fstat(fd)) != expected:
                    raise DeliveryOutboxCorruptError("delivery outbox root path binding changed")
            for parent_fd, child_fd, part, expected in zip(
                binding.fds, binding.fds[1:], binding.parts, binding.identities[1:]
            ):
                child_info = os.fstat(child_fd)
                if not stat.S_ISDIR(child_info.st_mode) or _inode(child_info) != expected:
                    raise DeliveryOutboxCorruptError("delivery outbox root path binding changed")
                entry = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if _inode(entry) != expected:
                    raise DeliveryOutboxCorruptError("delivery outbox root path binding changed")
            root_info = os.fstat(binding.root_fd)
            _private(root_info, directory=True)
            if _inode(root_info) != binding.identities[-1] or (
                self._root_identity is not None and _inode(root_info) != self._root_identity
            ):
                raise DeliveryOutboxCorruptError("delivery outbox root binding changed")
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENOENT}:
                raise DeliveryOutboxCorruptError("delivery outbox root path binding changed") from exc
            raise DeliveryOutboxCorruptError("delivery outbox root path binding could not be verified") from exc

    def _assert_binding(self, held: _HeldOutbox) -> None:
        held.root_binding.assert_bound()
        root_info = os.fstat(held.root_fd)
        _private(root_info, directory=True)
        if _inode(root_info) != held.root_identity or _inode(root_info) != self._root_identity:
            raise DeliveryOutboxCorruptError("delivery outbox root binding changed")
        lock_info = os.fstat(held.lock_fd)
        _private(lock_info, directory=False)
        path_lock = os.stat(".lock", dir_fd=held.root_fd, follow_symlinks=False)
        _private(path_lock, directory=False)
        if (
            _inode(lock_info) != held.lock_identity
            or _inode(path_lock) != held.lock_identity
            or self._lock_identity != held.lock_identity
        ):
            raise DeliveryOutboxCorruptError("delivery outbox lock binding changed")

    @contextmanager
    def _held_lock(self) -> Generator[_HeldOutbox, None, None]:
        with self._root_binding() as root_binding:
            lock_fd = self._open_lock(root_binding.root_fd)
            try:
                lock_identity = _inode(os.fstat(lock_fd))
                if self._lock_identity is None:
                    self._lock_identity = lock_identity
                held = _HeldOutbox(
                    self,
                    root_binding,
                    lock_fd,
                    _inode(os.fstat(root_binding.root_fd)),
                    lock_identity,
                )
                held.assert_bound()
                yield held
                held.assert_bound()
            finally:
                try:
                    _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    @staticmethod
    def _record_name(outbox_id: str) -> str:
        if not isinstance(outbox_id, str) or _HASH_RE.fullmatch(outbox_id) is None:
            raise ValueError("outbox_id must be a SHA-256 hexadecimal digest")
        return outbox_id + ".json"

    def _record_path(self, outbox_id: str) -> Path:
        return self.root / self._record_name(outbox_id)

    @staticmethod
    def _record_signature(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            info.st_mode,
            info.st_uid,
            info.st_gid,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    @staticmethod
    def _read_fd_content(fd: int, *, links: int = 1) -> bytes:
        """Read one bounded descriptor body without trusting its pathname."""

        before = os.fstat(fd)
        _private(before, directory=False, links=links)
        if before.st_size <= 0 or before.st_size > _MAX_RECORD_BYTES:
            raise DeliveryOutboxCorruptError("delivery outbox record has an unsafe size")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(fd, before.st_size - offset, offset)
            if not chunk:
                raise DeliveryOutboxCorruptError("delivery outbox record was truncated during read")
            chunks.append(chunk)
            offset += len(chunk)
        after = os.fstat(fd)
        if (
            _inode(after) != _inode(before)
            or CronDeliveryOutboxStore._record_signature(after)
            != CronDeliveryOutboxStore._record_signature(before)
        ):
            raise DeliveryOutboxCorruptError("delivery outbox record changed while being read")
        return b"".join(chunks)

    def _assert_record_binding(
        self,
        held: _HeldOutbox,
        name: str,
        fd: int,
        *,
        expected_inode: tuple[int, int],
        expected_signature: tuple[int, int, int, int, int, int, int],
        expected_content: tuple[int, str] | None = None,
    ) -> os.stat_result:
        descriptor = os.fstat(fd)
        _private(descriptor, directory=False)
        entry = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
        _private(entry, directory=False)
        if (
            _inode(descriptor) != expected_inode
            or _inode(entry) != expected_inode
            or self._record_signature(descriptor) != expected_signature
            or self._record_signature(entry) != expected_signature
        ):
            raise DeliveryOutboxCorruptError("delivery outbox record binding changed")
        if expected_content is not None and _content_signature(self._read_fd_content(fd)) != expected_content:
            raise DeliveryOutboxCorruptError("delivery outbox record content changed")
        return descriptor

    def _read_bytes(
        self, held: _HeldOutbox, outbox_id: str, *, durable: bool, with_proof: bool = False
    ) -> tuple[dict[str, Any], DeliveryOutboxRecord] | tuple[dict[str, Any], DeliveryOutboxRecord, _RecordProof]:
        held.assert_bound()
        name = self._record_name(outbox_id)
        fd: int | None = None
        try:
            fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=held.root_fd)
            before = os.fstat(fd)
            _private(before, directory=False)
            entry = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            _private(entry, directory=False)
            before_inode = _inode(before)
            before_signature = self._record_signature(before)
            if _inode(entry) != before_inode or self._record_signature(entry) != before_signature:
                raise DeliveryOutboxCorruptError("delivery outbox record binding changed")
            if before.st_size <= 0 or before.st_size > _MAX_RECORD_BYTES:
                raise DeliveryOutboxCorruptError("delivery outbox record has an unsafe size")
            remaining = before.st_size
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    raise DeliveryOutboxCorruptError("delivery outbox record was truncated during read")
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(fd)
            if _inode(after) != before_inode or self._record_signature(after) != before_signature:
                raise DeliveryOutboxCorruptError("delivery outbox record changed while being read")
            entry_after = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            _private(entry_after, directory=False)
            if _inode(entry_after) != before_inode or self._record_signature(entry_after) != before_signature:
                raise DeliveryOutboxCorruptError("delivery outbox record binding changed")
            raw_bytes = b"".join(chunks)
            parsed = _strict_json_loads(raw_bytes, label="delivery outbox record")
            try:
                canonical_bytes = _canonical_json(
                    parsed, error=DeliveryOutboxCorruptError, label="delivery outbox record"
                ).encode("utf-8")
            except DeliveryOutboxError:
                raise
            if raw_bytes != canonical_bytes:
                raise DeliveryOutboxCorruptError("delivery outbox record is not canonical JSON")
            wire = self._validate_wire(parsed, outbox_id)
            record = self._record_from_wire(wire)
            record_proof = _RecordProof(
                before_inode,
                before_signature,
                _content_signature(raw_bytes),
            )
            if durable:
                os.fsync(fd)
                held.assert_bound()
                self._assert_record_binding(
                    held,
                    name,
                    fd,
                    expected_inode=before_inode,
                    expected_signature=before_signature,
                )
                os.fsync(held.root_fd)
                held.assert_bound()
                self._assert_record_binding(
                    held,
                    name,
                    fd,
                    expected_inode=before_inode,
                    expected_signature=before_signature,
                )
            if with_proof:
                return wire, record, record_proof
            return wire, record
        except FileNotFoundError as exc:
            raise DeliveryOutboxNotFoundError(f"outbox {outbox_id} does not exist") from exc
        except DeliveryOutboxError:
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise DeliveryOutboxCorruptError("delivery outbox record must not be a symlink") from exc
            raise DeliveryOutboxPersistenceError(f"could not read or fsync delivery outbox {outbox_id}") from exc
        finally:
            if fd is not None:
                os.close(fd)

    @staticmethod
    def _validate_resolution(
        value: object,
        *,
        attempt: Mapping[str, Any],
        corrupt: bool = True,
    ) -> dict[str, Any]:
        error = DeliveryOutboxCorruptError if corrupt else DeliveryOutboxConflictError
        expected = {
            "action_id",
            "request_digest",
            "predecessor_attempt",
            "predecessor_owner",
            "evidence_ref",
            "predecessor_fenced_ref",
            "outcome",
            "receipt",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise error("resolution has an invalid schema")
        if value["action_id"] != attempt["action_id"]:
            raise error("resolution action_id does not match its predecessor attempt")
        if value["request_digest"] != attempt["request_digest"]:
            raise error("resolution request digest does not match its predecessor attempt")
        if type(value["predecessor_attempt"]) is not int or value["predecessor_attempt"] != attempt["attempt"]:
            raise error("resolution predecessor attempt does not match its history")
        if value["predecessor_owner"] != attempt["owner"]:
            raise error("resolution predecessor owner does not match its history")
        _text_corrupt(value["action_id"], label="resolution action_id") if corrupt else _text(value["action_id"], label="resolution action_id")
        _digest(value["request_digest"], label="resolution request digest", corrupt=corrupt)
        ref_fn = _text_corrupt if corrupt else _text
        ref_fn(value["evidence_ref"], label="resolution evidence_ref", maximum=_MAX_EVIDENCE_BYTES)
        ref_fn(
            value["predecessor_fenced_ref"],
            label="resolution predecessor_fenced_ref",
            maximum=_MAX_EVIDENCE_BYTES,
        )
        if not isinstance(value["outcome"], str) or value["outcome"] not in _RESOLUTION_STATES:
            raise error("resolution outcome is invalid")
        receipt = value["receipt"]
        if value["outcome"] == "confirmed_delivered":
            if not isinstance(receipt, Mapping) or not receipt:
                raise error("confirmed delivery resolution requires a strict receipt object")
        elif receipt is not None:
            raise error("non-delivery resolution cannot carry a receipt")
        copied = _copy_json_mapping(value, label="resolution", error=error)
        if copied["receipt"] is not None:
            copied["receipt"] = _copy_json_mapping(copied["receipt"], label="resolution receipt", error=error)
        if _canonical_json(copied["receipt"], error=error) != _canonical_json(
            attempt["receipt"], error=error
        ):
            raise error("resolution receipt does not agree with its attempt")
        return copied

    def _validate_attempt(
        self,
        value: object,
        *,
        outbox_id: str,
        destination_id: str,
        destination_channel: str,
        destination_to: str,
        messages: list[dict[str, Any]],
        expected_number: int,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise DeliveryOutboxCorruptError("delivery attempt must be an object")
        expected = {"attempt", "owner", "status", "action_id", "request_digest", "receipt", "resolution"}
        if set(value) != expected:
            raise DeliveryOutboxCorruptError("delivery attempt has an invalid schema")
        if type(value["attempt"]) is not int or value["attempt"] != expected_number:
            raise DeliveryOutboxCorruptError("delivery attempts must be contiguous integers")
        if value["attempt"] < 1 or value["attempt"] > _MAX_ATTEMPTS:
            raise DeliveryOutboxCorruptError("delivery attempt exceeds the per-destination limit")
        owner = _text_corrupt(value["owner"], label="delivery attempt owner")
        status = value["status"]
        if not isinstance(status, str) or status not in _ATTEMPT_STATES:
            raise DeliveryOutboxCorruptError("delivery attempt has an invalid state")
        action_id = _digest(value["action_id"], label="delivery attempt action_id", corrupt=True)
        expected_action = _action_id(outbox_id, destination_id, expected_number)
        if action_id != expected_action:
            raise DeliveryOutboxCorruptError("delivery attempt action_id is not deterministic")
        request_digest = _digest(value["request_digest"], label="delivery request digest", corrupt=True)
        expected_digest = _request_digest(outbox_id, destination_id, expected_number, messages)
        if request_digest != expected_digest:
            raise DeliveryOutboxCorruptError("delivery request digest is not canonical")
        receipt = value["receipt"]
        if receipt is not None:
            if not isinstance(receipt, Mapping) or not receipt:
                raise DeliveryOutboxCorruptError("delivery receipt must be a non-empty JSON object")
            receipt = _copy_json_mapping(receipt, label="delivery receipt", error=DeliveryOutboxCorruptError)
        if status == "confirmed_delivered" and receipt is None:
            raise DeliveryOutboxCorruptError("confirmed delivery requires a receipt")
        if status != "confirmed_delivered" and receipt is not None:
            raise DeliveryOutboxCorruptError("only confirmed delivery may retain a receipt")
        resolution = value["resolution"]
        if resolution is not None:
            resolution = self._validate_resolution(resolution, attempt=value, corrupt=True)
            if status != resolution["outcome"]:
                raise DeliveryOutboxCorruptError("resolution outcome does not match attempt state")
        if status in {"claimed", "sending", "unknown"} and resolution is not None:
            raise DeliveryOutboxCorruptError("in-flight attempt cannot retain a resolution")
        if status == "confirmed_compensated" and resolution is None:
            raise DeliveryOutboxCorruptError("confirmed compensation requires a reconciliation resolution")
        if status == "abandon" and resolution is None:
            raise DeliveryOutboxCorruptError("abandon requires a reconciliation resolution")
        copied = _copy_json_mapping(value, label="delivery attempt", error=DeliveryOutboxCorruptError)
        copied["owner"] = owner
        copied["receipt"] = receipt
        copied["resolution"] = resolution
        return copied

    def _validate_transition_history(
        self,
        value: object,
        *,
        destinations: list[dict[str, Any]],
        record_revision: int,
    ) -> None:
        """Replay retained transitions to prove revision and attempt history."""

        if not isinstance(value, list) or len(value) != record_revision:
            raise DeliveryOutboxCorruptError("delivery transition history does not match record_revision")
        destination_by_id = {item["destination_id"]: item for item in destinations}
        replay: dict[str, list[dict[str, Any]]] = {destination_id: [] for destination_id in destination_by_id}
        states = {destination_id: "pending" for destination_id in destination_by_id}
        resolutions: dict[tuple[str, int], tuple[str, dict[str, Any] | None, str, str] | None] = {}

        for expected_revision, raw_event in enumerate(value, start=1):
            if not isinstance(raw_event, Mapping):
                raise DeliveryOutboxCorruptError("delivery transition must be an object")
            expected_fields = {
                "revision",
                "destination_id",
                "attempt",
                "operation",
                "outcome",
                "receipt",
                "evidence_ref",
                "predecessor_fenced_ref",
            }
            if set(raw_event) != expected_fields:
                raise DeliveryOutboxCorruptError("delivery transition has an invalid schema")
            if type(raw_event["revision"]) is not int or raw_event["revision"] != expected_revision:
                raise DeliveryOutboxCorruptError("delivery transition revisions must be contiguous")
            destination_id = _digest(raw_event["destination_id"], label="transition destination_id", corrupt=True)
            if destination_id not in destination_by_id:
                raise DeliveryOutboxCorruptError("delivery transition names an unknown destination")
            attempt_number = raw_event["attempt"]
            if type(attempt_number) is not int or attempt_number < 1 or attempt_number > _MAX_ATTEMPTS:
                raise DeliveryOutboxCorruptError("delivery transition attempt is invalid")
            operation = raw_event["operation"]
            if not isinstance(operation, str) or operation not in _TRANSITION_OPERATIONS:
                raise DeliveryOutboxCorruptError("delivery transition operation is invalid")
            outcome = raw_event["outcome"]
            receipt = raw_event["receipt"]
            evidence_ref = raw_event["evidence_ref"]
            predecessor_fenced_ref = raw_event["predecessor_fenced_ref"]
            if operation in {"claim", "mark_sending"}:
                if outcome is not None or receipt is not None or evidence_ref is not None or predecessor_fenced_ref is not None:
                    raise DeliveryOutboxCorruptError("non-settlement transition carries outcome evidence")
            elif operation == "settle":
                if not isinstance(outcome, str) or outcome not in _OUTCOME_STATES:
                    raise DeliveryOutboxCorruptError("settlement transition outcome is invalid")
                if evidence_ref is not None or predecessor_fenced_ref is not None:
                    raise DeliveryOutboxCorruptError("settlement transition carries reconciliation evidence")
            else:
                if not isinstance(outcome, str) or outcome not in _RESOLUTION_STATES:
                    raise DeliveryOutboxCorruptError("reconciliation transition outcome is invalid")
                _text_corrupt(evidence_ref, label="transition evidence_ref", maximum=_MAX_EVIDENCE_BYTES)
                _text_corrupt(
                    predecessor_fenced_ref,
                    label="transition predecessor_fenced_ref",
                    maximum=_MAX_EVIDENCE_BYTES,
                )
            if outcome == "confirmed_delivered":
                if not isinstance(receipt, Mapping) or not receipt:
                    raise DeliveryOutboxCorruptError("confirmed delivery transition requires a receipt")
                receipt = _copy_json_mapping(receipt, label="transition receipt", error=DeliveryOutboxCorruptError)
            elif receipt is not None:
                raise DeliveryOutboxCorruptError("non-delivery transition cannot carry a receipt")

            history = replay[destination_id]
            state = states[destination_id]
            if operation == "claim":
                if state != "pending" or attempt_number != len(history) + 1:
                    raise DeliveryOutboxCorruptError("delivery transition claims an impossible successor attempt")
                history.append({"status": "claimed", "receipt": None})
                states[destination_id] = "claimed"
            elif operation == "mark_sending":
                if not history or attempt_number != len(history) or state != "claimed":
                    raise DeliveryOutboxCorruptError("delivery transition marks an impossible attempt")
                history[-1]["status"] = "sending"
                states[destination_id] = "sending"
            elif operation == "settle":
                if not history or attempt_number != len(history) or state != "sending":
                    raise DeliveryOutboxCorruptError("delivery transition settles an impossible attempt")
                history[-1]["status"] = outcome
                history[-1]["receipt"] = receipt
                states[destination_id] = {
                    "confirmed_delivered": "delivered",
                    "confirmed_not_sent": "pending",
                    "permanent_not_sent": "failed",
                    "unknown": "uncertain",
                }[outcome]
            else:
                if not history or attempt_number != len(history) or state not in {"claimed", "sending", "uncertain"}:
                    raise DeliveryOutboxCorruptError("delivery transition resolves an impossible attempt")
                if state == "claimed" and outcome != "confirmed_not_sent":
                    raise DeliveryOutboxCorruptError("claimed reconciliation must confirm not sent")
                history[-1]["status"] = outcome
                history[-1]["receipt"] = receipt
                resolutions[(destination_id, attempt_number)] = (
                    outcome,
                    receipt,
                    evidence_ref,
                    predecessor_fenced_ref,
                )
                states[destination_id] = {
                    "confirmed_delivered": "delivered",
                    "confirmed_not_sent": "pending",
                    "confirmed_compensated": "pending",
                    "abandon": "failed",
                }[outcome]

        for destination in destinations:
            destination_id = destination["destination_id"]
            attempts = destination["attempts"]
            replayed = replay[destination_id]
            if len(replayed) != len(attempts):
                raise DeliveryOutboxCorruptError("delivery transition history omits an attempt")
            for number, (replayed_attempt, attempt) in enumerate(zip(replayed, attempts), start=1):
                if (
                    replayed_attempt["status"] != attempt["status"]
                    or _canonical_json(replayed_attempt["receipt"], error=DeliveryOutboxCorruptError)
                    != _canonical_json(attempt["receipt"], error=DeliveryOutboxCorruptError)
                ):
                    raise DeliveryOutboxCorruptError("delivery transition history disagrees with attempt history")
                retained = attempt["resolution"]
                resolution_marker = resolutions.get((destination_id, number))
                if resolution_marker is None:
                    if retained is not None:
                        raise DeliveryOutboxCorruptError("attempt resolution is not retained in transition history")
                else:
                    if retained is None:
                        raise DeliveryOutboxCorruptError("reconciliation resolution is missing from attempt history")
                    outcome_value, receipt_value, evidence_value, fenced_value = resolution_marker
                    if (
                        retained["outcome"] != outcome_value
                        or _canonical_json(retained["receipt"], error=DeliveryOutboxCorruptError)
                        != _canonical_json(receipt_value, error=DeliveryOutboxCorruptError)
                        or retained["evidence_ref"] != evidence_value
                        or retained["predecessor_fenced_ref"] != fenced_value
                    ):
                        raise DeliveryOutboxCorruptError("reconciliation evidence disagrees with transition history")
            if states[destination_id] != destination["status"]:
                raise DeliveryOutboxCorruptError("delivery transition history disagrees with destination state")

    def _validate_wire(self, value: object, outbox_id: str) -> dict[str, Any]:
        fields = {
            "schema_version",
            "kind",
            "store_namespace",
            "outbox_id",
            "immutable_input",
            "destinations",
            "record_revision",
            "transition_history",
            "integrity",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise DeliveryOutboxCorruptError("delivery outbox record has an invalid schema")
        body = {key: item for key, item in value.items() if key != "integrity"}
        if type(value["schema_version"]) is not int or value["schema_version"] != _SCHEMA_VERSION:
            raise DeliveryOutboxCorruptError("delivery outbox record has an invalid schema version")
        if value["kind"] != _KIND or value["outbox_id"] != outbox_id:
            raise DeliveryOutboxCorruptError("delivery outbox record has an invalid kind or identifier")
        _text_corrupt(value["store_namespace"], label="delivery outbox namespace")
        if value["store_namespace"] != self.namespace:
            raise DeliveryOutboxCorruptError("delivery outbox belongs to a different namespace")
        _digest(value["outbox_id"], label="outbox_id", corrupt=True)
        if not isinstance(value["integrity"], str) or value["integrity"] != _integrity(body):
            raise DeliveryOutboxCorruptError("delivery outbox record failed its integrity check")
        revision = _exact_int(value["record_revision"], label="record_revision", corrupt=True)

        immutable_value = value["immutable_input"]
        immutable_fields = {
            "occurrence_id",
            "run_id",
            "occurrence_identity",
            "occurrence_owner",
            "occurrence_generation",
            "job_snapshot",
            "destination_policy",
            "completion",
            "delivery",
            "provenance_available",
        }
        if isinstance(immutable_value, Mapping) and immutable_value.get("provenance_available") is True:
            immutable_fields = immutable_fields | {"provenance"}
        if not isinstance(immutable_value, dict) or set(immutable_value) != immutable_fields:
            raise DeliveryOutboxCorruptError("delivery outbox immutable input has an invalid schema")
        occurrence_id = _digest(immutable_value["occurrence_id"], label="occurrence_id", corrupt=True)
        if _outbox_id(self.namespace, occurrence_id) != outbox_id:
            raise DeliveryOutboxCorruptError("outbox_id does not match occurrence identity")
        run_id = _text_corrupt(immutable_value["run_id"], label="run_id")
        generation = _exact_int(
            immutable_value["occurrence_generation"], label="occurrence generation", minimum=1, corrupt=True
        )
        if run_id != f"cron-occurrence-{occurrence_id}-{generation}":
            raise DeliveryOutboxCorruptError("run_id does not match occurrence identity")
        _validate_occurrence_identity(
            immutable_value["occurrence_identity"], namespace=self.namespace, error=DeliveryOutboxCorruptError
        )
        identity = immutable_value["occurrence_identity"]
        identity_payload = _canonical_json(identity, error=DeliveryOutboxCorruptError).encode("utf-8")
        expected_occurrence_id = hashlib.sha256(
            b"nanobot-cron-occurrence-v1\0" + identity_payload
        ).hexdigest()
        if expected_occurrence_id != occurrence_id:
            raise DeliveryOutboxCorruptError("occurrence_id does not match occurrence identity")
        _text_corrupt(immutable_value["occurrence_owner"], label="occurrence owner")
        job_snapshot = _copy_json_mapping(
            immutable_value["job_snapshot"], label="job snapshot", error=DeliveryOutboxCorruptError
        )
        policy = _copy_json_mapping(
            immutable_value["destination_policy"], label="destination policy", error=DeliveryOutboxCorruptError
        )
        completion = _copy_json_mapping(
            immutable_value["completion"], label="completion", error=DeliveryOutboxCorruptError
        )
        delivery = _validate_delivery_envelope(
            immutable_value["delivery"], error=DeliveryOutboxCorruptError
        )
        if "delivery" not in completion or _canonical_json(completion["delivery"], error=DeliveryOutboxCorruptError) != _canonical_json(delivery, error=DeliveryOutboxCorruptError):
            raise DeliveryOutboxCorruptError("immutable completion and delivery envelope differ")
        if type(immutable_value["provenance_available"]) is not bool:
            raise DeliveryOutboxCorruptError("provenance_available must be a boolean")
        provenance_available = immutable_value["provenance_available"]
        if provenance_available:
            provenance = _copy_json_mapping(
                immutable_value["provenance"], label="provenance", error=DeliveryOutboxCorruptError
            )
            if "provenance" not in completion or _canonical_json(completion["provenance"], error=DeliveryOutboxCorruptError) != _canonical_json(provenance, error=DeliveryOutboxCorruptError):
                raise DeliveryOutboxCorruptError("immutable provenance differs from completion")
        elif "provenance" in immutable_value or "provenance" in completion:
            raise DeliveryOutboxCorruptError("unavailable provenance must remain absent")
        messages = delivery["messages"]
        raw_destinations = value["destinations"]
        if not isinstance(raw_destinations, list) or len(raw_destinations) > _MAX_DESTINATIONS:
            raise DeliveryOutboxCorruptError("delivery outbox destination count exceeds its limit")
        destinations: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        for raw_destination in raw_destinations:
            if not isinstance(raw_destination, Mapping):
                raise DeliveryOutboxCorruptError("delivery destination must be an object")
            destination_fields = {"destination_id", "channel", "to", "status", "attempts"}
            if set(raw_destination) != destination_fields:
                raise DeliveryOutboxCorruptError("delivery destination has an invalid schema")
            destination_id = _digest(raw_destination["destination_id"], label="destination_id", corrupt=True)
            channel = _text_corrupt(raw_destination["channel"], label="destination channel")
            to = _text_corrupt(raw_destination["to"], label="destination to")
            if destination_id != _destination_id(outbox_id, channel, to):
                raise DeliveryOutboxCorruptError("destination_id is not deterministic")
            if destination_id in seen_ids or (channel, to) in seen_pairs:
                raise DeliveryOutboxCorruptError("delivery outbox contains duplicate destinations")
            seen_ids.add(destination_id)
            seen_pairs.add((channel, to))
            attempts_value = raw_destination["attempts"]
            if not isinstance(attempts_value, list) or len(attempts_value) > _MAX_ATTEMPTS:
                raise DeliveryOutboxCorruptError("delivery destination attempt count exceeds its limit")
            attempts = [
                self._validate_attempt(
                    attempt,
                    outbox_id=outbox_id,
                    destination_id=destination_id,
                    destination_channel=channel,
                    destination_to=to,
                    messages=messages,
                    expected_number=index,
                )
                for index, attempt in enumerate(attempts_value, start=1)
            ]
            if any(attempt["status"] in {"claimed", "sending", "unknown"} for attempt in attempts[:-1]):
                raise DeliveryOutboxCorruptError("delivery attempt history has a non-terminal predecessor")
            status = raw_destination["status"]
            if not isinstance(status, str) or status not in _DESTINATION_STATES:
                raise DeliveryOutboxCorruptError("delivery destination has an invalid state")
            if status != _destination_state(attempts):
                raise DeliveryOutboxCorruptError("delivery destination state is impossible for its history")
            destination_wire = _copy_json_mapping(
                raw_destination, label="delivery destination", error=DeliveryOutboxCorruptError
            )
            destination_wire["attempts"] = attempts
            destinations.append(destination_wire)
        self._validate_transition_history(
            value["transition_history"],
            destinations=destinations,
            record_revision=revision,
        )
        expected_destinations = _derive_destinations(
            policy=policy,
            deliver=delivery["deliver"],
            messages=messages,
            outbox_id=outbox_id,
            error=DeliveryOutboxCorruptError,
        )
        if [
            (item["destination_id"], item["channel"], item["to"])
            for item in destinations
        ] != [
            (item["destination_id"], item["channel"], item["to"])
            for item in expected_destinations
        ]:
            raise DeliveryOutboxCorruptError("delivery destinations do not match frozen destination policy")
        immutable = _copy_json_mapping(
            immutable_value, label="immutable input", error=DeliveryOutboxCorruptError
        )
        immutable["job_snapshot"] = job_snapshot
        immutable["destination_policy"] = policy
        immutable["completion"] = completion
        immutable["delivery"] = delivery
        if provenance_available:
            immutable["provenance"] = _copy_json_mapping(
                immutable["provenance"], label="provenance", error=DeliveryOutboxCorruptError
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": _KIND,
            "store_namespace": self.namespace,
            "outbox_id": outbox_id,
            "immutable_input": immutable,
            "destinations": destinations,
            "record_revision": revision,
            "transition_history": _copy_json_value(
                value["transition_history"],
                label="transition history",
                error=DeliveryOutboxCorruptError,
            ),
        }

    @staticmethod
    def _record_from_wire(wire: Mapping[str, Any]) -> DeliveryOutboxRecord:
        immutable = _copy_json_mapping(wire["immutable_input"], label="immutable input")
        destinations: list[DeliveryDestination] = []
        for raw_destination in wire["destinations"]:
            attempts: list[DeliveryAttempt] = []
            for raw_attempt in raw_destination["attempts"]:
                receipt = (
                    None
                    if raw_attempt["receipt"] is None
                    else _copy_json_mapping(raw_attempt["receipt"], label="delivery receipt")
                )
                resolution = (
                    None
                    if raw_attempt["resolution"] is None
                    else _copy_json_mapping(raw_attempt["resolution"], label="resolution")
                )
                attempts.append(
                    DeliveryAttempt(
                        raw_attempt["attempt"],
                        raw_attempt["owner"],
                        raw_attempt["status"],
                        raw_attempt["action_id"],
                        raw_attempt["request_digest"],
                        receipt,
                        resolution,
                    )
                )
            destinations.append(
                DeliveryDestination(
                    raw_destination["destination_id"],
                    raw_destination["channel"],
                    raw_destination["to"],
                    raw_destination["status"],
                    tuple(attempts),
                )
            )
        return DeliveryOutboxRecord(
            wire["store_namespace"],
            wire["outbox_id"],
            immutable["occurrence_id"],
            immutable["run_id"],
            immutable,
            tuple(destinations),
            wire["record_revision"],
        )

    @staticmethod
    def _wire_bytes(body: Mapping[str, Any]) -> bytes:
        clean = {key: value for key, value in body.items() if key != "integrity"}
        encoded_body = dict(clean)
        encoded_body["integrity"] = _integrity(encoded_body)
        encoded = _canonical_json(encoded_body).encode("utf-8")
        if len(encoded) > _MAX_RECORD_BYTES:
            raise DeliveryOutboxConflictError("encoded delivery outbox record exceeds the 1MiB limit")
        return encoded

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("short delivery outbox write")
            offset += written

    def _create_exclusive(self, held: _HeldOutbox, outbox_id: str, data: bytes) -> None:
        if len(data) > _MAX_RECORD_BYTES:
            raise DeliveryOutboxConflictError("encoded delivery outbox record exceeds the 1MiB limit")
        held.assert_bound()
        fd: int | None = None
        name = self._record_name(outbox_id)
        intended_content = _content_signature(data)
        try:
            fd = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_NONBLOCK,
                0o600,
                dir_fd=held.root_fd,
            )
            os.fchmod(fd, 0o600)
            initial = os.fstat(fd)
            self._assert_record_binding(
                held,
                name,
                fd,
                expected_inode=_inode(initial),
                expected_signature=self._record_signature(initial),
            )
            self._write_all(fd, data)
            completed = os.fstat(fd)
            completed_signature = self._record_signature(completed)
            self._assert_record_binding(
                held,
                name,
                fd,
                expected_inode=_inode(completed),
                expected_signature=completed_signature,
                expected_content=intended_content,
            )
            os.fsync(fd)
            held.assert_bound()
            self._assert_record_binding(
                held,
                name,
                fd,
                expected_inode=_inode(completed),
                expected_signature=completed_signature,
                expected_content=intended_content,
            )
            os.fsync(held.root_fd)
            held.assert_bound()
            self._assert_record_binding(
                held,
                name,
                fd,
                expected_inode=_inode(completed),
                expected_signature=completed_signature,
                expected_content=intended_content,
            )
        except FileExistsError:
            raise
        except DeliveryOutboxError:
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise DeliveryOutboxCorruptError("delivery outbox record must not be a symlink") from exc
            raise DeliveryOutboxPersistenceError(f"could not durably create outbox {outbox_id}") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def _replace_record(
        self,
        held: _HeldOutbox,
        outbox_id: str,
        data: bytes,
        *,
        predecessor_proof: _RecordProof | None = None,
    ) -> None:
        if len(data) > _MAX_RECORD_BYTES:
            raise DeliveryOutboxConflictError("encoded delivery outbox record exceeds the 1MiB limit")
        held.assert_bound()
        temporary = f".{outbox_id}.{uuid.uuid4().hex}.tmp"
        fd: int | None = None
        predecessor_fd: int | None = None
        published = False
        name = self._record_name(outbox_id)
        intended_content = _content_signature(data)
        try:
            # Keep the predecessor descriptor, inode, metadata, and body bound
            # for the entire staging interval.  An attacker replacing the
            # target with an identical private file must still fail closed.
            predecessor_fd = os.open(
                name,
                os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK,
                dir_fd=held.root_fd,
            )
            predecessor = os.fstat(predecessor_fd)
            _private(predecessor, directory=False)
            predecessor_entry = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            _private(predecessor_entry, directory=False)
            predecessor_inode = _inode(predecessor)
            predecessor_signature = self._record_signature(predecessor)
            predecessor_content = _content_signature(self._read_fd_content(predecessor_fd))
            if (
                _inode(predecessor_entry) != predecessor_inode
                or self._record_signature(predecessor_entry) != predecessor_signature
            ):
                raise DeliveryOutboxCorruptError("delivery outbox predecessor binding changed")
            if predecessor_proof is not None and (
                predecessor_inode != predecessor_proof.inode
                or predecessor_signature != predecessor_proof.signature
                or predecessor_content != predecessor_proof.content
            ):
                raise DeliveryOutboxCorruptError("delivery outbox predecessor was replaced before staging")
            fd = os.open(
                temporary,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_NONBLOCK,
                0o600,
                dir_fd=held.root_fd,
            )
            os.fchmod(fd, 0o600)
            temp_before = os.fstat(fd)
            self._assert_record_binding(
                held,
                temporary,
                fd,
                expected_inode=_inode(temp_before),
                expected_signature=self._record_signature(temp_before),
            )
            self._write_all(fd, data)
            staged = os.fstat(fd)
            staged_inode = _inode(staged)
            staged_signature = self._record_signature(staged)
            self._assert_record_binding(
                held,
                temporary,
                fd,
                expected_inode=staged_inode,
                expected_signature=staged_signature,
                expected_content=intended_content,
            )
            os.fsync(fd)
            self._assert_record_binding(
                held,
                temporary,
                fd,
                expected_inode=staged_inode,
                expected_signature=staged_signature,
                expected_content=intended_content,
            )
            held.assert_bound()
            # Replacement is only valid for an already-owned final record;
            # an unlink, symlink, or type swap in the target path is a fence,
            # never an invitation to recreate the destination implicitly.
            self._assert_record_binding(
                held,
                name,
                predecessor_fd,
                expected_inode=predecessor_inode,
                expected_signature=predecessor_signature,
                expected_content=predecessor_content,
            )
            os.rename(
                temporary,
                name,
                src_dir_fd=held.root_fd,
                dst_dir_fd=held.root_fd,
            )
            published = True
            # Recheck the predecessor descriptor after rename as well.  A
            # concurrent in-place mutation in an interposed rename hook must
            # not be acknowledged merely because the final path now points at
            # the intended staging inode.
            predecessor_after = os.fstat(predecessor_fd)
            _private(predecessor_after, directory=False, links=0)
            if (
                _inode(predecessor_after) != predecessor_inode
                or _content_signature(self._read_fd_content(predecessor_fd, links=0)) != predecessor_content
            ):
                raise DeliveryOutboxCorruptError("delivery outbox predecessor changed during replacement")
            # The staging descriptor remains open across rename and therefore
            # proves the exact inode and intended content that receive
            # acknowledgement.  POSIX may update ctime when the directory
            # entry is renamed, so capture the post-rename signature only
            # after reproving the inode and content.
            renamed_info = os.fstat(fd)
            _private(renamed_info, directory=False)
            if _inode(renamed_info) == predecessor_inode:
                raise DeliveryOutboxCorruptError("delivery outbox replacement reused its predecessor inode")
            renamed_signature = self._record_signature(renamed_info)
            renamed_entry = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
            _private(renamed_entry, directory=False)
            if _inode(renamed_info) != _inode(renamed_entry):
                raise DeliveryOutboxCorruptError("delivery outbox record binding changed after rename")
            self._assert_record_binding(
                held,
                name,
                fd,
                expected_inode=_inode(renamed_info),
                expected_signature=renamed_signature,
                expected_content=intended_content,
            )
            os.fsync(held.root_fd)
            held.assert_bound()
            self._assert_record_binding(
                held,
                name,
                fd,
                expected_inode=_inode(renamed_info),
                expected_signature=renamed_signature,
                expected_content=intended_content,
            )
        except FileNotFoundError as exc:
            raise DeliveryOutboxCorruptError("delivery outbox predecessor is unavailable") from exc
        except DeliveryOutboxError:
            raise
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENOENT}:
                raise DeliveryOutboxCorruptError("delivery outbox record replacement was unsafe") from exc
            raise DeliveryOutboxPersistenceError(f"could not durably replace outbox {outbox_id}") from exc
        finally:
            if fd is not None:
                os.close(fd)
            if predecessor_fd is not None:
                os.close(predecessor_fd)
            if not published:
                try:
                    os.unlink(temporary, dir_fd=held.root_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def _validate_existing_records(self) -> None:
        with self._held_lock() as held:
            try:
                entries = os.listdir(held.root_fd)
            except OSError as exc:
                raise DeliveryOutboxPersistenceError("could not list delivery outbox") from exc
            for name in entries:
                if name == ".lock":
                    continue
                if _TEMP_ENTRY_RE.fullmatch(name):
                    try:
                        info = os.stat(name, dir_fd=held.root_fd, follow_symlinks=False)
                    except OSError as exc:
                        raise DeliveryOutboxCorruptError("delivery outbox temporary entry is unsafe") from exc
                    _private(info, directory=False)
                    continue
                if not name.endswith(".json") or _HASH_RE.fullmatch(name[:-5]) is None:
                    raise DeliveryOutboxCorruptError(f"unsafe entry in delivery outbox: {name}")
                self._read_bytes(held, name[:-5], durable=False)

    def _completed_input(
        self, completed: OccurrenceRecord
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        if not isinstance(completed, OccurrenceRecord):
            raise TypeError("prepare_completed accepts only a durable OccurrenceRecord")
        if completed.state != "completed" or completed.completion is None:
            raise DeliveryOutboxReconciliationRequiredError("occurrence is not a completed durable record")
        if completed.identity.store_namespace != self.namespace:
            raise DeliveryOutboxConflictError("completed occurrence belongs to a different namespace")
        completion = _copy_json_mapping(completed.completion, label="completed occurrence completion")
        if "delivery" not in completion:
            raise DeliveryOutboxUnsupportedError("completed occurrence has no explicit delivery envelope")
        delivery = _validate_delivery_envelope(completion["delivery"], error=DeliveryOutboxConflictError)
        provenance_available = "provenance" in completion
        if provenance_available:
            provenance = _copy_json_mapping(completion["provenance"], label="completion provenance")
        occurrence_id = _digest(completed.occurrence_id, label="occurrence_id")
        outbox_id = _outbox_id(self.namespace, occurrence_id)
        policy = _copy_json_mapping(completed.destination_policy, label="destination policy")
        immutable: dict[str, Any] = {
            "occurrence_id": occurrence_id,
            "run_id": f"cron-occurrence-{occurrence_id}-{completed.generation}",
            "occurrence_identity": _copy_json_mapping(completed.identity.as_dict(), label="occurrence identity"),
            "occurrence_owner": _text(completed.owner, label="occurrence owner"),
            "occurrence_generation": _exact_int(completed.generation, label="occurrence generation", minimum=1),
            "job_snapshot": _copy_json_mapping(completed.job_snapshot, label="job snapshot"),
            "destination_policy": policy,
            "completion": completion,
            "delivery": delivery,
            "provenance_available": provenance_available,
        }
        if provenance_available:
            immutable["provenance"] = provenance
        destinations = _derive_destinations(
            policy=policy,
            deliver=delivery["deliver"],
            messages=delivery["messages"],
            outbox_id=outbox_id,
            error=DeliveryOutboxConflictError,
        )
        body: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "kind": _KIND,
            "store_namespace": self.namespace,
            "outbox_id": outbox_id,
            "immutable_input": immutable,
            "destinations": destinations,
            "record_revision": 0,
            "transition_history": [],
        }
        encoded = self._wire_bytes(body)
        parsed = _strict_json_loads(encoded, label="new delivery outbox record")
        wire = self._validate_wire(parsed, outbox_id)
        return outbox_id, wire, destinations

    def prepare_completed(
        self,
        occurrence_store: CronOccurrenceStore,
        occurrence_id: str,
    ) -> DeliveryOutboxRecord:
        """Durably prepare one completed occurrence without invoking delivery.

        The authoritative ledger read happens immediately before candidate
        derivation and before this store acquires its own lock, so the two
        stores are never nested.  A detached occurrence record is never
        accepted as preparation authority.
        """
        if not isinstance(occurrence_store, CronOccurrenceStore):
            raise TypeError("prepare_completed requires a CronOccurrenceStore")
        if not isinstance(occurrence_id, str):
            raise TypeError("prepare_completed requires an occurrence_id string")
        completed = occurrence_store.get_completed_durable(occurrence_id)
        outbox_id, expected_wire, _destinations = self._completed_input(completed)
        expected_body = {key: value for key, value in expected_wire.items() if key != "integrity"}
        expected_immutable = expected_body["immutable_input"]
        with self._held_lock() as held:
            try:
                existing_wire, existing = self._read_bytes(held, outbox_id, durable=True)
            except DeliveryOutboxNotFoundError:
                data = self._wire_bytes(expected_body)
                try:
                    self._create_exclusive(held, outbox_id, data)
                except FileExistsError:
                    existing_wire, existing = self._read_bytes(held, outbox_id, durable=True)
                else:
                    return self._record_from_wire(expected_wire)
            if _canonical_json(existing_wire["immutable_input"], error=DeliveryOutboxConflictError) != _canonical_json(expected_immutable, error=DeliveryOutboxConflictError):
                raise DeliveryOutboxConflictError("completed occurrence immutable input conflicts with existing outbox")
            return existing

    def get(self, outbox_id: str) -> DeliveryOutboxRecord:
        """Return a detached, durably reproved outbox view."""
        with self._held_lock() as held:
            _wire, record = self._read_bytes(held, outbox_id, durable=True)
            return record

    read = get

    def list(self) -> list[DeliveryOutboxRecord]:
        """Return all retained outboxes in filename order."""
        with self._held_lock() as held:
            try:
                names = sorted(os.listdir(held.root_fd))
            except OSError as exc:
                raise DeliveryOutboxPersistenceError("could not list delivery outbox") from exc
            records: list[DeliveryOutboxRecord] = []
            for name in names:
                if name == ".lock" or _TEMP_ENTRY_RE.fullmatch(name):
                    continue
                if not name.endswith(".json") or _HASH_RE.fullmatch(name[:-5]) is None:
                    raise DeliveryOutboxCorruptError(f"unsafe entry in delivery outbox: {name}")
                _wire, record = self._read_bytes(held, name[:-5], durable=True)
                records.append(record)
            return records

    def reconciliation_required(self) -> list[DeliveryOutboxRecord]:
        """Return outboxes with a claimed, sending, or uncertain destination."""
        return [record for record in self.list() if record.reconciliation_required]

    get_reconciliation_required = reconciliation_required

    @staticmethod
    def _find_destination(wire: Mapping[str, Any], destination_id: str) -> tuple[int, dict[str, Any]]:
        for index, destination in enumerate(wire["destinations"]):
            if destination["destination_id"] == destination_id:
                return index, destination
        raise DeliveryOutboxNotFoundError(f"destination {destination_id} does not exist")

    @staticmethod
    def _find_attempt(destination: Mapping[str, Any], attempt_number: int) -> dict[str, Any]:
        attempts = destination["attempts"]
        if not attempts or attempts[-1]["attempt"] != attempt_number:
            raise DeliveryOutboxOwnershipError("claim does not identify the current destination attempt")
        return attempts[-1]

    @staticmethod
    def _copy_wire(wire: Mapping[str, Any]) -> dict[str, Any]:
        return _copy_json_mapping(wire, label="delivery outbox transition")

    def _issued_for_reservation(
        self, *, outbox_id: str, destination_id: str, expected_revision: int, owner: str
    ) -> _IssuedClaim | None:
        for issued in self._issued_claims.values():
            claim = issued.claim
            if (
                issued.process_id == os.getpid()
                and issued.store_identity == id(self)
                and issued.phase == "claimed"
                and claim.outbox_id == outbox_id
                and claim.destination_id == destination_id
                and claim.expected_revision == expected_revision
                and claim.owner == owner
            ):
                return issued
        return None

    def _assert_claim(self, claim: object) -> _IssuedClaim:
        if not isinstance(claim, DeliveryClaim):
            raise DeliveryOutboxOwnershipError("operation requires an issued delivery claim")
        issued = self._issued_claims.get(id(claim))
        if (
            issued is None
            or issued.claim is not claim
            or issued.process_id != os.getpid()
            or issued.store_identity != id(self)
        ):
            raise DeliveryOutboxOwnershipError("claim was not issued by this store object in this process")
        return issued

    @staticmethod
    def _bump_revision(
        wire: dict[str, Any],
        *,
        destination_id: str,
        attempt: int,
        operation: str,
        outcome: str | None = None,
        receipt: Mapping[str, Any] | None = None,
        evidence_ref: str | None = None,
        predecessor_fenced_ref: str | None = None,
    ) -> None:
        current = _exact_int(wire["record_revision"], label="record_revision", corrupt=True)
        history = wire.get("transition_history")
        if not isinstance(history, list) or len(history) != current:
            raise DeliveryOutboxCorruptError("delivery transition history is inconsistent")
        if operation not in _TRANSITION_OPERATIONS:
            raise DeliveryOutboxConflictError("delivery transition operation is invalid")
        history.append(
            {
                "revision": current + 1,
                "destination_id": destination_id,
                "attempt": attempt,
                "operation": operation,
                "outcome": outcome,
                "receipt": None if receipt is None else _copy_json_mapping(receipt, label="transition receipt"),
                "evidence_ref": evidence_ref,
                "predecessor_fenced_ref": predecessor_fenced_ref,
            }
        )
        wire["record_revision"] = current + 1

    def claim_destination(
        self,
        outbox_id: str,
        destination_id: str,
        expected_revision: int,
        owner: str | None = None,
    ) -> DeliveryClaim:
        """Durably reserve one pending destination and return a sealed claim."""
        _digest(outbox_id, label="outbox_id")
        _digest(destination_id, label="destination_id")
        expected_revision = _exact_int(expected_revision, label="expected_revision")
        owner = self._default_owner if owner is None else _text(owner, label="claim owner")
        with self._held_lock() as held:
            wire, record, predecessor_proof = self._read_bytes(
                held, outbox_id, durable=False, with_proof=True
            )
            if record.record_revision != expected_revision:
                issued = self._issued_for_reservation(
                    outbox_id=outbox_id,
                    destination_id=destination_id,
                    expected_revision=expected_revision,
                    owner=owner,
                )
                if issued is not None:
                    index, destination = self._find_destination(wire, destination_id)
                    if (
                        destination["status"] == "claimed"
                        and destination["attempts"]
                        and destination["attempts"][-1]["attempt"] == issued.claim.attempt
                        and destination["attempts"][-1]["owner"] == owner
                    ):
                        # Reprove a visible record before returning the exact
                        # same process-local object after an acknowledgement
                        # boundary was interrupted.
                        self._read_bytes(held, outbox_id, durable=True)
                        return issued.claim
                raise DeliveryOutboxStaleRevisionError("delivery outbox revision is stale")
            _index, destination = self._find_destination(wire, destination_id)
            if destination["status"] != "pending":
                if destination["status"] in {"claimed", "sending", "uncertain"}:
                    raise DeliveryOutboxReconciliationRequiredError(
                        "destination is already claimed or uncertain; reconcile it before another attempt"
                    )
                raise DeliveryOutboxConflictError("destination is not pending")
            attempt_number = len(destination["attempts"]) + 1
            if attempt_number > _MAX_ATTEMPTS:
                raise DeliveryOutboxConflictError("destination attempt limit exceeded")
            action_id = _action_id(outbox_id, destination_id, attempt_number)
            messages = wire["immutable_input"]["delivery"]["messages"]
            request_digest = _request_digest(outbox_id, destination_id, attempt_number, messages)
            destination["attempts"].append(
                {
                    "attempt": attempt_number,
                    "owner": owner,
                    "status": "claimed",
                    "action_id": action_id,
                    "request_digest": request_digest,
                    "receipt": None,
                    "resolution": None,
                }
            )
            destination["status"] = "claimed"
            self._bump_revision(
                wire,
                destination_id=destination_id,
                attempt=attempt_number,
                operation="claim",
            )
            encoded = self._wire_bytes(wire)
            self._replace_record(
                held,
                outbox_id,
                encoded,
                predecessor_proof=predecessor_proof,
            )
            claim = DeliveryClaim(
                outbox_id,
                destination_id,
                attempt_number,
                owner,
                expected_revision,
                wire["record_revision"],
                action_id,
                request_digest,
            )
            self._issued_claims[id(claim)] = _IssuedClaim(
                claim, os.getpid(), id(self), "claimed", claim.record_revision
            )
            return claim

    def mark_sending(self, claim: DeliveryClaim) -> DeliveryOutboxRecord:
        """Durably fence a claim as sending while retaining its settlement capability."""
        issued = self._assert_claim(claim)
        try:
            with self._held_lock() as held:
                wire, record, predecessor_proof = self._read_bytes(
                    held, claim.outbox_id, durable=False, with_proof=True
                )
                _index, destination = self._find_destination(wire, claim.destination_id)
                attempt = self._find_attempt(destination, claim.attempt)
                if attempt["owner"] != claim.owner:
                    raise DeliveryOutboxOwnershipError("claim owner is stale")
                if attempt["status"] == "sending" and destination["status"] == "sending":
                    issued.phase = "sending"
                    return record
                if attempt["status"] != "claimed" or destination["status"] != "claimed":
                    raise DeliveryOutboxOwnershipError("claim is no longer sendable")
                attempt["status"] = "sending"
                destination["status"] = "sending"
                self._bump_revision(
                    wire,
                    destination_id=claim.destination_id,
                    attempt=claim.attempt,
                    operation="mark_sending",
                )
                self._replace_record(
                    held,
                    claim.outbox_id,
                    self._wire_bytes(wire),
                    predecessor_proof=predecessor_proof,
                )
                issued.phase = "sending"
                issued.current_revision = wire["record_revision"]
                _new_wire, new_record = self._read_bytes(held, claim.outbox_id, durable=False)
                return new_record
        except Exception:
            # A failed sending fence must never leave a caller with send
            # authority that was not durably established.
            self._issued_claims.pop(id(claim), None)
            raise

    @staticmethod
    def _parse_settlement(
        result: object,
        *,
        outcome: str | None,
        receipt: Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None]:
        if result is not None and outcome is not None:
            raise DeliveryOutboxConflictError("settlement result and keyword outcome cannot be combined")
        if result is None:
            result = {"outcome": outcome, "receipt": receipt}
        if isinstance(result, str):
            outcome_value = result if outcome is None else outcome
            receipt_value = receipt
        elif isinstance(result, Mapping):
            if receipt is not None:
                raise DeliveryOutboxConflictError("settlement object and keyword receipt cannot be combined")
            if set(result) not in ({"outcome"}, {"outcome", "receipt"}):
                raise DeliveryOutboxConflictError("settlement must have outcome and optional receipt fields")
            outcome_value = result["outcome"]
            receipt_value = result.get("receipt")
        else:
            raise DeliveryOutboxConflictError("settlement must be a string or exact outcome object")
        if not isinstance(outcome_value, str) or outcome_value not in _OUTCOME_STATES:
            raise DeliveryOutboxConflictError("settlement outcome is invalid")
        if outcome_value == "confirmed_delivered":
            if not isinstance(receipt_value, Mapping) or not receipt_value:
                raise DeliveryOutboxConflictError("confirmed delivery requires a strict receipt object")
            return outcome_value, _copy_json_mapping(receipt_value, label="delivery receipt")
        if receipt_value is not None:
            raise DeliveryOutboxConflictError("only confirmed delivery may carry a receipt")
        return outcome_value, None

    def settle(
        self,
        claim: DeliveryClaim,
        result: object = None,
        *,
        outcome: str | None = None,
        receipt: Mapping[str, Any] | None = None,
    ) -> DeliveryOutboxRecord:
        """Durably settle one sending claim; the capability is then consumed."""
        issued = self._assert_claim(claim)
        outcome_value, receipt_value = self._parse_settlement(result, outcome=outcome, receipt=receipt)
        try:
            with self._held_lock() as held:
                wire, record, predecessor_proof = self._read_bytes(
                    held, claim.outbox_id, durable=False, with_proof=True
                )
                _index, destination = self._find_destination(wire, claim.destination_id)
                attempt = self._find_attempt(destination, claim.attempt)
                if (
                    attempt["owner"] != claim.owner
                    or attempt["action_id"] != claim.action_id
                    or attempt["request_digest"] != claim.request_digest
                ):
                    raise DeliveryOutboxOwnershipError("claim predecessor is stale")
                if issued.phase != "sending" or attempt["status"] != "sending":
                    raise DeliveryOutboxOwnershipError("claim has not been durably marked sending")
                attempt["status"] = outcome_value
                attempt["receipt"] = receipt_value
                destination["status"] = _destination_state(destination["attempts"])
                self._bump_revision(
                    wire,
                    destination_id=claim.destination_id,
                    attempt=claim.attempt,
                    operation="settle",
                    outcome=outcome_value,
                    receipt=receipt_value,
                )
                self._replace_record(
                    held,
                    claim.outbox_id,
                    self._wire_bytes(wire),
                    predecessor_proof=predecessor_proof,
                )
                self._issued_claims.pop(id(claim), None)
                _new_wire, settled = self._read_bytes(held, claim.outbox_id, durable=False)
                return settled
        except Exception:
            self._issued_claims.pop(id(claim), None)
            raise

    @staticmethod
    def _parse_resolution_input(
        resolution: object,
        *,
        receipt: Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None]:
        if isinstance(resolution, Mapping):
            if set(resolution) not in ({"outcome"}, {"outcome", "receipt"}):
                raise DeliveryOutboxConflictError("resolution must have outcome and optional receipt fields")
            outcome = resolution["outcome"]
            receipt_value = resolution.get("receipt")
        else:
            outcome = resolution
            receipt_value = receipt
        if not isinstance(outcome, str) or outcome not in _RESOLUTION_STATES:
            raise DeliveryOutboxConflictError("reconciliation resolution is invalid")
        if outcome == "confirmed_delivered":
            if not isinstance(receipt_value, Mapping) or not receipt_value:
                raise DeliveryOutboxConflictError("confirmed delivery reconciliation requires a strict receipt")
            return outcome, _copy_json_mapping(receipt_value, label="reconciliation receipt")
        if receipt_value is not None:
            raise DeliveryOutboxConflictError("only confirmed delivery reconciliation may carry a receipt")
        return outcome, None

    @staticmethod
    def _resolution_request(
        *,
        destination_id: str,
        predecessor_attempt: int,
        predecessor_owner: str,
        action_id: str,
        request_digest: str,
        evidence_ref: str,
        predecessor_fenced_ref: str,
        outcome: str,
        receipt: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "destination_id": destination_id,
            "predecessor_attempt": predecessor_attempt,
            "predecessor_owner": predecessor_owner,
            "action_id": action_id,
            "request_digest": request_digest,
            "evidence_ref": evidence_ref,
            "predecessor_fenced_ref": predecessor_fenced_ref,
            "outcome": outcome,
            "receipt": None if receipt is None else _copy_json_mapping(receipt, label="resolution receipt"),
        }

    @staticmethod
    def _find_resolution(wire: Mapping[str, Any], action_id: str) -> dict[str, Any] | None:
        for destination in wire["destinations"]:
            for attempt in destination["attempts"]:
                resolution = attempt["resolution"]
                if resolution is not None and resolution["action_id"] == action_id:
                    return resolution
        return None

    def resolve_after_reconciliation(
        self,
        outbox_id: str | DeliveryClaim,
        destination_id: str | None = None,
        *,
        expected_revision: int,
        action_id: str,
        request_digest: str,
        evidence_ref: str,
        predecessor_fenced_ref: str,
        resolution: str | Mapping[str, Any],
        receipt: Mapping[str, Any] | None = None,
        predecessor_attempt: int | None = None,
        predecessor_owner: str | None = None,
        attempt: int | None = None,
        owner: str | None = None,
    ) -> DeliveryOutboxRecord:
        """Record a trusted, externally fenced reconciliation result.

        The action identity and canonical request digest are retained inside
        the predecessor attempt.  An exact retry is reproved before the
        caller's revision token is compared, which makes a post-acknowledgement
        retry safe without allowing a conflicting action-id reuse.
        """
        claim: DeliveryClaim | None = outbox_id if isinstance(outbox_id, DeliveryClaim) else None
        if claim is not None:
            outbox_id = claim.outbox_id
            if destination_id is None:
                destination_id = claim.destination_id
            if predecessor_attempt is None and attempt is None:
                predecessor_attempt = claim.attempt
            if predecessor_owner is None and owner is None:
                predecessor_owner = claim.owner
        if not isinstance(outbox_id, str):
            raise TypeError("outbox_id must be a string or DeliveryClaim")
        if not isinstance(destination_id, str):
            raise TypeError("destination_id is required")
        _digest(outbox_id, label="outbox_id")
        _digest(destination_id, label="destination_id")
        expected_revision = _exact_int(expected_revision, label="expected_revision")
        action_id = _text(action_id, label="action_id")
        request_digest = _digest(request_digest, label="request_digest")
        evidence_ref = _text(evidence_ref, label="evidence_ref", maximum=_MAX_EVIDENCE_BYTES)
        predecessor_fenced_ref = _text(
            predecessor_fenced_ref,
            label="predecessor_fenced_ref",
            maximum=_MAX_EVIDENCE_BYTES,
        )
        predecessor_attempt = predecessor_attempt if predecessor_attempt is not None else attempt
        predecessor_owner = predecessor_owner if predecessor_owner is not None else owner
        predecessor_attempt = _exact_int(
            predecessor_attempt,
            label="predecessor_attempt",
            minimum=1,
        )
        predecessor_owner = _text(predecessor_owner, label="predecessor_owner")
        outcome, receipt_value = self._parse_resolution_input(resolution, receipt=receipt)
        request = self._resolution_request(
            destination_id=destination_id,
            predecessor_attempt=predecessor_attempt,
            predecessor_owner=predecessor_owner,
            action_id=action_id,
            request_digest=request_digest,
            evidence_ref=evidence_ref,
            predecessor_fenced_ref=predecessor_fenced_ref,
            outcome=outcome,
            receipt=receipt_value,
        )
        with self._held_lock() as held:
            wire, record, predecessor_proof = self._read_bytes(
                held, outbox_id, durable=True, with_proof=True
            )
            existing = self._find_resolution(wire, action_id)
            if existing is not None:
                existing_request = {
                    "destination_id": next(
                        destination["destination_id"]
                        for destination in wire["destinations"]
                        for attempt_item in destination["attempts"]
                        if attempt_item["resolution"] is existing
                    ),
                    "predecessor_attempt": existing["predecessor_attempt"],
                    "predecessor_owner": existing["predecessor_owner"],
                    "action_id": existing["action_id"],
                    "request_digest": existing["request_digest"],
                    "evidence_ref": existing["evidence_ref"],
                    "predecessor_fenced_ref": existing["predecessor_fenced_ref"],
                    "outcome": existing["outcome"],
                    "receipt": existing["receipt"],
                }
                if _canonical_json(existing_request, error=DeliveryOutboxConflictError) != _canonical_json(request, error=DeliveryOutboxConflictError):
                    if existing["request_digest"] != request_digest:
                        raise DeliveryOutboxConflictError("action_id conflicts with a different canonical request digest")
                    raise DeliveryOutboxConflictError("reconciliation action retry conflicts with its retained request")
                # The durable read above reproves the predecessor before this
                # exact idempotent retry bypasses the stale revision token.
                return record
            for destination in wire["destinations"]:
                for attempt_item in destination["attempts"]:
                    if attempt_item["action_id"] == action_id and attempt_item["request_digest"] != request_digest:
                        raise DeliveryOutboxConflictError("action_id conflicts with a different canonical request digest")
            if record.record_revision != expected_revision:
                raise DeliveryOutboxStaleRevisionError("delivery outbox revision is stale")
            _index, destination = self._find_destination(wire, destination_id)
            current = self._find_attempt(destination, predecessor_attempt)
            if (
                current["owner"] != predecessor_owner
                or current["action_id"] != action_id
                or current["request_digest"] != request_digest
            ):
                raise DeliveryOutboxOwnershipError("reconciliation predecessor is stale")
            state = destination["status"]
            if state == "claimed":
                if outcome != "confirmed_not_sent":
                    raise DeliveryOutboxConflictError(
                        "a claimed destination may only be reconciled as confirmed_not_sent"
                    )
            elif state in {"sending", "uncertain"}:
                pass
            else:
                raise DeliveryOutboxConflictError("reconciliation predecessor is no longer in flight")
            if current["status"] not in {"claimed", "sending", "unknown"}:
                raise DeliveryOutboxConflictError("reconciliation predecessor attempt is already settled")
            retained_resolution = {
                "action_id": action_id,
                "request_digest": request_digest,
                "predecessor_attempt": predecessor_attempt,
                "predecessor_owner": predecessor_owner,
                "evidence_ref": evidence_ref,
                "predecessor_fenced_ref": predecessor_fenced_ref,
                "outcome": outcome,
                "receipt": receipt_value,
            }
            current["status"] = outcome
            current["receipt"] = receipt_value
            current["resolution"] = retained_resolution
            destination["status"] = _destination_state(destination["attempts"])
            self._bump_revision(
                wire,
                destination_id=destination_id,
                attempt=predecessor_attempt,
                operation="reconcile",
                outcome=outcome,
                receipt=receipt_value,
                evidence_ref=evidence_ref,
                predecessor_fenced_ref=predecessor_fenced_ref,
            )
            self._replace_record(
                held,
                outbox_id,
                self._wire_bytes(wire),
                predecessor_proof=predecessor_proof,
            )
            # A local claim, when one exists, is no longer usable after a
            # trusted reconciliation outcome consumes its predecessor.
            for key, issued in list(self._issued_claims.items()):
                if (
                    issued.claim.outbox_id == outbox_id
                    and issued.claim.destination_id == destination_id
                    and issued.claim.attempt == predecessor_attempt
                ):
                    self._issued_claims.pop(key, None)
            _new_wire, resolved = self._read_bytes(held, outbox_id, durable=False)
            return resolved

    # Explicit aliases make the package boundary readable to later runtime
    # adapters without introducing another authority or a sender abstraction.
    resolve = resolve_after_reconciliation
    prepare = prepare_completed
    get_outbox = get
    mark_destination_sending = mark_sending
    settle_destination = settle


# Names used by migration notes and downstream adapters remain aliases to the
# one implementation; there is still only one durable authority.
DeliveryOutboxStore = CronDeliveryOutboxStore
CronDeliveryStore = CronDeliveryOutboxStore
CronDeliveryOutboxError = DeliveryOutboxError
CronDeliveryOutboxPersistenceError = DeliveryOutboxPersistenceError
CronDeliveryOutboxCorruptError = DeliveryOutboxCorruptError
CronDeliveryOutboxConflictError = DeliveryOutboxConflictError
CronDeliveryOutboxStaleRevisionError = DeliveryOutboxStaleRevisionError
CronDeliveryOutboxOwnershipError = DeliveryOutboxOwnershipError
CronDeliveryOutboxReconciliationRequiredError = DeliveryOutboxReconciliationRequiredError
CronDeliveryStoreError = DeliveryOutboxError
DeliveryStoreError = DeliveryOutboxError
DeliveryStorePersistenceError = DeliveryOutboxPersistenceError
DeliveryStoreCorruptError = DeliveryOutboxCorruptError
DeliveryStoreConflictError = DeliveryOutboxConflictError
DeliveryStoreStaleRevisionError = DeliveryOutboxStaleRevisionError
DeliveryStoreOwnershipError = DeliveryOutboxOwnershipError
DeliveryStoreReconciliationRequiredError = DeliveryOutboxReconciliationRequiredError
