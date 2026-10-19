"""Explicit offline operator reconciliation for dormant cron delivery.

This module owns content-addressed evidence, plans and application receipts.
It has no channel, provider, bus, scheduler or configuration-path dependency.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from nanobot.channels.delivery_contract import (
    ChannelTextPartReceipt,
    ChannelTextReceipt,
    validate_receipt_binding,
)
from nanobot.cron.delivery_store import (
    CronDeliveryOutboxStore,
    DeliveryAttempt,
    DeliveryDestination,
    DeliveryOutboxConflictError,
    DeliveryOutboxCorruptError,
    DeliveryOutboxNotFoundError,
    DeliveryOutboxOwnershipError,
    DeliveryOutboxPersistenceError,
    DeliveryOutboxRecord,
    DeliveryOutboxStaleRevisionError,
    DeliverySendBusyError,
)

_HASH_RE = re.compile(r"[0-9a-f]{64}")
_REF_RE = re.compile(r"sha256:([0-9a-f]{64})")
_MAX_ARTIFACT = 256 * 1024
_MAX_ENTRIES = 4096
_MAX_INSPECTION_ENTRIES = 4096
_MAX_JSON_DEPTH = 32
_MAX_JSON_VALUES = 16_384
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_KINDS = ("evidence", "plans", "applications")
_PLAN_DOMAIN = b"nanobot-cron-delivery-reconciliation-plan-v1\0"
_ARTIFACT_DOMAIN = b"nanobot-cron-delivery-reconciliation-artifact-v1\0"


class DeliveryReconciliationError(RuntimeError):
    """Base content-free offline reconciliation error."""


class ReconciliationUnsafeError(DeliveryReconciliationError):
    """Malformed data or an unsafe filesystem/binding was observed."""


class ReconciliationConflictError(DeliveryReconciliationError):
    """Evidence, plan, predecessor or confirmation does not agree."""


@dataclass(frozen=True, slots=True)
class AuthorityBinding:
    path: str
    device: int
    inode: int
    mode: int
    uid: int
    gid: int


@dataclass(frozen=True, slots=True)
class ReconciliationEntry:
    outbox_id: str
    destination_id: str
    channel: str
    target_redacted: bool
    status: str
    attempt: int
    owner: str
    action_id: str
    request_digest: str
    record_revision: int
    permitted_outcomes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReconciliationInspection:
    clean: bool
    entries: tuple[ReconciliationEntry, ...]


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    kind: str
    digest: str
    reference: str
    semantic_id: str | None = None


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReconciliationUnsafeError("reconciliation data is malformed") from exc


def _strict_loads(data: bytes) -> dict[str, Any]:
    if len(data) > _MAX_ARTIFACT:
        raise ReconciliationUnsafeError("reconciliation artifact exceeds its size limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ReconciliationUnsafeError("reconciliation artifact has duplicate fields")
            value[key] = item
        return value

    def nonfinite(_value: str) -> None:
        raise ReconciliationUnsafeError("reconciliation artifact contains a non-finite number")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
    except ReconciliationUnsafeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ReconciliationUnsafeError("reconciliation artifact is malformed") from exc
    if not isinstance(value, dict):
        raise ReconciliationUnsafeError("reconciliation artifact must be one object")
    value = cast(dict[str, Any], value)  # JSON decoder/pairs constructs string-key objects.
    count = 0
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > _MAX_JSON_VALUES or depth > _MAX_JSON_DEPTH:
            raise ReconciliationUnsafeError("reconciliation artifact is too complex")
        if isinstance(item, dict):
            item = cast(dict[object, object], item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            item = cast(list[object], item)
            pending.extend((child, depth + 1) for child in item)
    if _canonical(value) != data:
        raise ReconciliationUnsafeError("reconciliation artifact is not canonical JSON")
    return value


def _text(value: object, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ReconciliationUnsafeError("reconciliation artifact contains invalid text")
    return value


def _digest(value: object) -> str:
    value = _text(value)
    if _HASH_RE.fullmatch(value) is None:
        raise ReconciliationUnsafeError("reconciliation artifact contains an invalid digest")
    return value


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReconciliationUnsafeError("reconciliation artifact contains an invalid integer")
    return value


def _validate_selector(path: Path, *, label: str, may_be_absent: bool) -> Path:
    path = Path(path)
    if not path.is_absolute() or path == Path(path.anchor) or ".." in path.parts:
        raise ReconciliationUnsafeError(f"{label} must be an absolute non-root path")
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current /= part
        try:
            info = os.stat(current, follow_symlinks=False)
        except FileNotFoundError:
            if may_be_absent:
                break
            raise ReconciliationUnsafeError(f"{label} is unavailable") from None
        if stat.S_ISLNK(info.st_mode) or (index < len(path.parts[1:]) - 1 and not stat.S_ISDIR(info.st_mode)):
            raise ReconciliationUnsafeError(f"{label} has an unsafe path binding")
    return path


def validate_authority_selectors(
    outbox_root: Path, artifacts_root: Path
) -> tuple[Path, Path]:
    outbox = _validate_selector(outbox_root, label="outbox root", may_be_absent=True)
    artifacts = _validate_selector(artifacts_root, label="artifacts root", may_be_absent=True)
    outbox_canonical = outbox.resolve(strict=False)
    artifacts_canonical = artifacts.resolve(strict=False)
    if (
        outbox_canonical == artifacts_canonical
        or outbox_canonical.is_relative_to(artifacts_canonical)
        or artifacts_canonical.is_relative_to(outbox_canonical)
    ):
        raise ReconciliationUnsafeError("outbox and artifacts roots must not overlap")
    return outbox, artifacts


def _binding(path: Path) -> AuthorityBinding:
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ReconciliationUnsafeError("authority binding is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ReconciliationUnsafeError("authority binding is not a directory")
    return AuthorityBinding(
        str(path), info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid
    )


def _binding_dict(binding: AuthorityBinding) -> dict[str, object]:
    return asdict(binding)


def _binding_from(value: object) -> AuthorityBinding:
    fields = {"path", "device", "inode", "mode", "uid", "gid"}
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
        raise ReconciliationUnsafeError("authority artifact binding is malformed")
    value = cast(dict[str, object], value)
    return AuthorityBinding(
        _text(value["path"], maximum=4096),
        _integer(value["device"]),
        _integer(value["inode"]),
        _integer(value["mode"]),
        _integer(value["uid"]),
        _integer(value["gid"]),
    )


def _assert_binding(expected: AuthorityBinding, path: Path) -> None:
    if _binding(path) != expected:
        raise ReconciliationConflictError("authority binding changed")


def _private(info: os.stat_result, *, directory: bool) -> None:
    expected = 0o700 if directory else 0o600
    right_type = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if (
        not right_type
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != expected
        or (not directory and info.st_nlink != 1)
    ):
        raise ReconciliationUnsafeError("reconciliation artifact storage is not private")


class _ArtifactStore:
    def __init__(self, root: Path, *, create: bool) -> None:
        if os.name != "posix" or not _O_NOFOLLOW or not _O_DIRECTORY or not _O_NONBLOCK:
            raise ReconciliationUnsafeError("reconciliation artifacts require qualified POSIX support")
        self.root = root
        created_root = False
        if create:
            try:
                os.mkdir(root, 0o700)
                created_root = True
            except FileExistsError:
                pass
            except OSError as exc:
                raise ReconciliationUnsafeError("artifact root could not be created safely") from exc
        self.fd = self._open_dir(root)
        self.identity = (os.fstat(self.fd).st_dev, os.fstat(self.fd).st_ino)
        self.kind_fds: dict[str, int] = {}
        try:
            for kind in _KINDS:
                if created_root:
                    try:
                        os.mkdir(kind, 0o700, dir_fd=self.fd)
                    except FileExistsError:
                        raise ReconciliationUnsafeError(
                            "new artifact root contains a conflicting entry"
                        ) from None
                self.kind_fds[kind] = self._open_dir(kind, dir_fd=self.fd)
                if created_root:
                    os.fsync(self.kind_fds[kind])
            if created_root:
                os.fsync(self.fd)
                parent_fd = self._open_dir(root.parent)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            self._validate_tree()
        except OSError as exc:
            self.close()
            raise ReconciliationUnsafeError(
                "artifact directory could not be established durably"
            ) from exc
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _open_dir(path: str | Path, *, dir_fd: int | None = None) -> int:
        fd: int | None = None
        try:
            fd = os.open(path, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=dir_fd)
            _private(os.fstat(fd), directory=True)
            entry = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
            _private(entry, directory=True)
            if (entry.st_dev, entry.st_ino) != (os.fstat(fd).st_dev, os.fstat(fd).st_ino):
                raise ReconciliationUnsafeError("artifact directory binding changed")
            return fd
        except ReconciliationUnsafeError:
            if fd is not None:
                os.close(fd)
            raise
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            raise ReconciliationUnsafeError("artifact directory is unavailable or unsafe") from exc

    def close(self) -> None:
        for fd in self.kind_fds.values():
            os.close(fd)
        self.kind_fds.clear()
        if hasattr(self, "fd"):
            os.close(self.fd)

    def __enter__(self) -> _ArtifactStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def require_reconciliation_binding(self) -> None:
        """Reprove the existing artifact root and identity-directory binding checks."""
        return self._assert_bound()

    def _assert_bound(self) -> None:
        info = os.fstat(self.fd)
        _private(info, directory=True)
        entry = os.stat(self.root, follow_symlinks=False)
        _private(entry, directory=True)
        if (info.st_dev, info.st_ino) != self.identity or (entry.st_dev, entry.st_ino) != self.identity:
            raise ReconciliationUnsafeError("artifact root binding changed")
        for kind, fd in self.kind_fds.items():
            info = os.fstat(fd)
            entry = os.stat(kind, dir_fd=self.fd, follow_symlinks=False)
            _private(info, directory=True)
            _private(entry, directory=True)
            if (info.st_dev, info.st_ino) != (entry.st_dev, entry.st_ino):
                raise ReconciliationUnsafeError("artifact identity directory binding changed")

    def _validate_tree(self) -> None:
        self._assert_bound()
        root_names = os.listdir(self.fd)
        if len(root_names) > len(_KINDS) or set(root_names) != set(_KINDS):
            raise ReconciliationUnsafeError("artifact root contains unexpected entries")
        total = 0
        for fd in self.kind_fds.values():
            names = os.listdir(fd)
            total += len(names)
            if total > _MAX_ENTRIES:
                raise ReconciliationUnsafeError("artifact store has too many entries")
            for name in names:
                if not name.endswith(".json") or _HASH_RE.fullmatch(name[:-5]) is None:
                    raise ReconciliationUnsafeError("artifact store contains an unexpected entry")
                data = self._read_name(fd, name)
                if hashlib.sha256(_ARTIFACT_DOMAIN + data).hexdigest() != name[:-5]:
                    raise ReconciliationUnsafeError("artifact digest does not match its content")
                _strict_loads(data)

    @staticmethod
    def _read_name(directory_fd: int, name: str) -> bytes:
        fd: int | None = None
        try:
            fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=directory_fd)
            before = os.fstat(fd)
            _private(before, directory=False)
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            _private(entry, directory=False)
            if (before.st_dev, before.st_ino) != (entry.st_dev, entry.st_ino):
                raise ReconciliationUnsafeError("artifact file binding changed")
            if before.st_size > _MAX_ARTIFACT:
                raise ReconciliationUnsafeError("artifact exceeds its size limit")
            chunks: list[bytes] = []
            remaining = _MAX_ARTIFACT + 1
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            after = os.fstat(fd)
            if len(data) > _MAX_ARTIFACT or before != after:
                raise ReconciliationUnsafeError("artifact changed during read")
            return data
        except ReconciliationUnsafeError:
            raise
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENXIO}:
                raise ReconciliationUnsafeError("artifact is not a private regular file") from exc
            raise ReconciliationUnsafeError("artifact could not be read safely") from exc
        finally:
            if fd is not None:
                os.close(fd)

    @classmethod
    def _assert_open_file(
        cls,
        directory_fd: int,
        name: str,
        fd: int,
        *,
        inode: tuple[int, int],
        signature: tuple[int, int, int, int, int, int, int],
        expected: bytes,
    ) -> None:
        descriptor = os.fstat(fd)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _private(descriptor, directory=False)
        _private(entry, directory=False)
        current_signature = (
            descriptor.st_mode,
            descriptor.st_uid,
            descriptor.st_gid,
            descriptor.st_nlink,
            descriptor.st_size,
            descriptor.st_mtime_ns,
            descriptor.st_ctime_ns,
        )
        if (
            (descriptor.st_dev, descriptor.st_ino) != inode
            or (entry.st_dev, entry.st_ino) != inode
            or current_signature != signature
        ):
            raise ReconciliationUnsafeError("artifact file binding changed")
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = _MAX_ARTIFACT + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if b"".join(chunks) != expected:
            raise ReconciliationUnsafeError("immutable artifact content changed")

    @classmethod
    def _fsync_existing_exact(
        cls, directory_fd: int, name: str, expected: bytes
    ) -> None:
        fd: int | None = None
        try:
            fd = os.open(
                name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=directory_fd
            )
            info = os.fstat(fd)
            _private(info, directory=False)
            inode = (info.st_dev, info.st_ino)
            signature = (
                info.st_mode,
                info.st_uid,
                info.st_gid,
                info.st_nlink,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
            cls._assert_open_file(
                directory_fd, name, fd, inode=inode, signature=signature, expected=expected
            )
            os.fsync(fd)
            cls._assert_open_file(
                directory_fd, name, fd, inode=inode, signature=signature, expected=expected
            )
            os.fsync(directory_fd)
            cls._assert_open_file(
                directory_fd, name, fd, inode=inode, signature=signature, expected=expected
            )
        except ReconciliationUnsafeError:
            raise
        except OSError as exc:
            raise ReconciliationUnsafeError("artifact durability reproof failed") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def read(self, kind: str, digest: str) -> dict[str, Any]:
        _digest(digest)
        self._assert_bound()
        data = self._read_name(self.kind_fds[kind], f"{digest}.json")
        if hashlib.sha256(_ARTIFACT_DOMAIN + data).hexdigest() != digest:
            raise ReconciliationUnsafeError("artifact digest does not match its content")
        value = _strict_loads(data)
        self._assert_bound()
        return value

    def publish(self, kind: str, value: Mapping[str, Any]) -> StoredArtifact:
        data = _canonical(value)
        if len(data) > _MAX_ARTIFACT:
            raise ReconciliationUnsafeError("artifact exceeds its size limit")
        digest = hashlib.sha256(_ARTIFACT_DOMAIN + data).hexdigest()
        name = f"{digest}.json"
        directory_fd = self.kind_fds[kind]
        self._assert_bound()
        fd: int | None = None
        created = False
        try:
            fd = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_NONBLOCK,
                0o600,
                dir_fd=directory_fd,
            )
            created = True
            os.fchmod(fd, 0o600)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short artifact write")
                view = view[written:]
            info = os.fstat(fd)
            inode = (info.st_dev, info.st_ino)
            signature = (
                info.st_mode,
                info.st_uid,
                info.st_gid,
                info.st_nlink,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
            self._assert_open_file(
                directory_fd, name, fd, inode=inode, signature=signature, expected=data
            )
            os.fsync(fd)
            self._assert_open_file(
                directory_fd, name, fd, inode=inode, signature=signature, expected=data
            )
            os.fsync(directory_fd)
            self._assert_open_file(
                directory_fd, name, fd, inode=inode, signature=signature, expected=data
            )
        except FileExistsError:
            pass
        except OSError as exc:
            raise ReconciliationUnsafeError("artifact publication failed") from exc
        finally:
            if fd is not None:
                os.close(fd)
        if not created:
            self._fsync_existing_exact(directory_fd, name, data)
        if self._read_name(directory_fd, name) != data:
            raise ReconciliationUnsafeError("immutable artifact conflicts with existing content")
        self._assert_bound()
        return StoredArtifact(kind, digest, f"sha256:{digest}")


def _current_attempt(
    record: DeliveryOutboxRecord, destination_id: str
) -> tuple[DeliveryDestination, DeliveryAttempt]:
    for destination in record.destinations:
        if destination.destination_id == destination_id:
            if not destination.attempts:
                break
            return destination, destination.attempts[-1]
    raise ReconciliationConflictError("reconciliation predecessor is unavailable")


def _permitted(status: str) -> tuple[str, ...]:
    if status == "claimed":
        return ("confirmed_not_sent",)
    if status in {"sending", "uncertain"}:
        return ("confirmed_delivered", "abandon")
    return ()


def inspect_reconciliation(
    outbox_root: Path, *, namespace: str, artifacts_root: Path
) -> ReconciliationInspection:
    _text(namespace)
    outbox_root, _artifacts_root = validate_authority_selectors(outbox_root, artifacts_root)
    try:
        records = CronDeliveryOutboxStore.inspect_existing(outbox_root, namespace=namespace)
    except (ValueError, DeliveryOutboxCorruptError, DeliveryOutboxPersistenceError) as exc:
        raise ReconciliationUnsafeError("delivery outbox inspection failed") from exc
    entries: list[ReconciliationEntry] = []
    for record in records:
        for destination in record.destinations:
            if destination.status not in {"claimed", "sending", "uncertain"}:
                continue
            attempt = destination.attempts[-1]
            entries.append(
                ReconciliationEntry(
                    record.outbox_id,
                    destination.destination_id,
                    destination.channel,
                    True,
                    destination.status,
                    attempt.attempt,
                    attempt.owner,
                    attempt.action_id,
                    attempt.request_digest,
                    record.record_revision,
                    _permitted(destination.status),
                )
            )
            if len(entries) > _MAX_INSPECTION_ENTRIES:
                raise ReconciliationUnsafeError("inspection result exceeds its entry limit")
    return ReconciliationInspection(not entries, tuple(entries))


def _parse_receipt(value: object, *, destination: DeliveryDestination, attempt: DeliveryAttempt, count: int) -> dict[str, Any]:
    fields = {"schema", "channel", "target", "action_id", "request_digest", "parts"}
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields or value["schema"] != "nanobot.channel_text_receipt.v1":
        raise ReconciliationUnsafeError("Telegram receipt evidence is malformed")
    value = cast(dict[str, Any], value)  # Exact raw receipt wire field set.
    if not isinstance(value["parts"], list):
        raise ReconciliationUnsafeError("Telegram receipt parts are malformed")
    parts: list[ChannelTextPartReceipt] = []
    for item in cast(list[object], value["parts"]):
        required = {"index", "message_index", "part_index", "provider_message_id", "provider_chat_id"}
        if not isinstance(item, dict) or set(cast(dict[object, object], item)) not in (required, required | {"provider_timestamp"}):
            raise ReconciliationUnsafeError("Telegram receipt part is malformed")
        # Raw kwargs stay unconverted; the dataclass owns ordered field validation.
        item = cast(dict[str, Any], item)
        try:
            parts.append(ChannelTextPartReceipt(**item))
        except (TypeError, ValueError) as exc:
            raise ReconciliationUnsafeError("Telegram receipt part is malformed") from exc
    try:
        receipt = ChannelTextReceipt(
            channel=value["channel"],
            target=value["target"],
            action_id=value["action_id"],
            request_digest=value["request_digest"],
            parts=tuple(parts),
        )
        validate_receipt_binding(
            receipt,
            channel=destination.channel,
            target=destination.to,
            action_id=attempt.action_id,
            request_digest=attempt.request_digest,
            message_count=count,
        )
    except (TypeError, ValueError) as exc:
        raise ReconciliationConflictError("Telegram receipt does not match the predecessor") from exc
    if receipt.channel != "telegram":
        raise ReconciliationConflictError("only strict Telegram receipt evidence is supported")
    return receipt.to_json()


def _redact_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Retain exact request/provider evidence without retaining the raw target."""
    return {
        "schema": receipt["schema"],
        "channel": receipt["channel"],
        "target_digest": hashlib.sha256(
            b"nanobot-cron-delivery-target-v1\0" + str(receipt["target"]).encode("utf-8")
        ).hexdigest(),
        "action_id": receipt["action_id"],
        "request_digest": receipt["request_digest"],
        "parts": receipt["parts"],
    }


def _restore_receipt(
    value: object, *, destination: DeliveryDestination, attempt: DeliveryAttempt, count: int
) -> dict[str, Any]:
    fields = {"schema", "channel", "target_digest", "action_id", "request_digest", "parts"}
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
        raise ReconciliationUnsafeError("redacted Telegram receipt evidence is malformed")
    value = cast(dict[str, Any], value)  # Exact raw receipt wire field set.
    target_digest = hashlib.sha256(
        b"nanobot-cron-delivery-target-v1\0" + destination.to.encode("utf-8")
    ).hexdigest()
    if value["target_digest"] != target_digest:
        raise ReconciliationConflictError("Telegram receipt target binding does not match")
    strict = {
        "schema": value["schema"],
        "channel": value["channel"],
        "target": destination.to,
        "action_id": value["action_id"],
        "request_digest": value["request_digest"],
        "parts": value["parts"],
    }
    return _parse_receipt(strict, destination=destination, attempt=attempt, count=count)


def _evidence_body(
    *, record: DeliveryOutboxRecord, destination: DeliveryDestination, attempt: DeliveryAttempt,
    outcome: str, evidence_type: str, receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": "nanobot.cron_delivery_reconciliation_evidence.v1",
        "evidence_type": evidence_type,
        "namespace": record.store_namespace,
        "outbox_id": record.outbox_id,
        "destination_id": destination.destination_id,
        "predecessor": {
            "attempt": attempt.attempt,
            "owner": attempt.owner,
            "status": destination.status,
            "action_id": attempt.action_id,
            "request_digest": attempt.request_digest,
            "record_revision": record.record_revision,
        },
        "outcome": outcome,
        "receipt": receipt,
        "attestation": {
            "provider_acknowledgement_meaning": "Telegram acceptance; not proof of user read"
            if receipt is not None else "operator attestation; no provider lookup was performed"
        },
    }


def import_evidence(
    outbox_root: Path, *, namespace: str, artifacts_root: Path, outbox_id: str,
    destination_id: str, outcome: Literal["confirmed_not_sent", "confirmed_delivered", "abandon"],
    receipt: Mapping[str, Any] | None = None,
) -> StoredArtifact:
    _text(namespace)
    outbox_root, artifacts_root = validate_authority_selectors(outbox_root, artifacts_root)
    outbox_binding = _binding(outbox_root)
    records = CronDeliveryOutboxStore.inspect_existing(outbox_root, namespace=namespace)
    _assert_binding(outbox_binding, outbox_root)
    record = next((item for item in records if item.outbox_id == _digest(outbox_id)), None)
    if record is None:
        raise ReconciliationConflictError("outbox does not exist")
    destination, attempt = _current_attempt(record, _digest(destination_id))
    if outcome not in _permitted(destination.status):
        raise ReconciliationConflictError("outcome is not permitted for predecessor state")
    normalized: dict[str, Any] | None = None
    if outcome == "confirmed_delivered":
        if receipt is None:
            raise ReconciliationConflictError("confirmed delivery requires a strict receipt")
        normalized = _redact_receipt(_parse_receipt(
            dict(receipt), destination=destination, attempt=attempt, count=len(record.messages)
        ))
        evidence_type = "telegram_provider_ack"
    elif outcome == "confirmed_not_sent":
        if receipt is not None:
            raise ReconciliationConflictError("not-sent evidence cannot contain a receipt")
        evidence_type = "claimed_never_sending"
    else:
        if receipt is not None:
            raise ReconciliationConflictError("abandon evidence cannot contain a receipt")
        evidence_type = "operator_abandon"
    body = _evidence_body(
        record=record, destination=destination, attempt=attempt, outcome=outcome,
        evidence_type=evidence_type, receipt=normalized,
    )
    with _ArtifactStore(artifacts_root, create=True) as store:
        return store.publish("evidence", body)


def _parse_evidence(value: object) -> dict[str, Any]:
    fields = {
        "schema", "evidence_type", "namespace", "outbox_id", "destination_id",
        "predecessor", "outcome", "receipt", "attestation",
    }
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
        raise ReconciliationUnsafeError("evidence artifact schema is malformed")
    value = cast(dict[str, Any], value)  # Exact raw artifact wire field set.
    if value["schema"] != "nanobot.cron_delivery_reconciliation_evidence.v1":
        raise ReconciliationUnsafeError("evidence artifact schema is unsupported")
    if value["evidence_type"] not in {
        "claimed_never_sending", "telegram_provider_ack", "operator_abandon"
    }:
        raise ReconciliationUnsafeError("evidence type is unsupported")
    predecessor_fields = {"attempt", "owner", "status", "action_id", "request_digest", "record_revision"}
    predecessor = value["predecessor"]
    if not isinstance(predecessor, dict) or set(cast(dict[object, object], predecessor)) != predecessor_fields:
        raise ReconciliationUnsafeError("evidence predecessor is malformed")
    predecessor = cast(dict[str, object], predecessor)
    _text(value["namespace"])
    _digest(value["outbox_id"])
    _digest(value["destination_id"])
    _integer(predecessor["attempt"], minimum=1)
    _text(predecessor["owner"])
    _text(predecessor["status"])
    _digest(predecessor["action_id"])
    _digest(predecessor["request_digest"])
    _integer(predecessor["record_revision"])
    if not isinstance(value["attestation"], dict) or set(cast(dict[object, object], value["attestation"])) != {"provider_acknowledgement_meaning"}:
        raise ReconciliationUnsafeError("evidence attestation is malformed")
    attestation = _text(
        cast(dict[str, object], value["attestation"])["provider_acknowledgement_meaning"], maximum=256
    )
    expected = {
        "claimed_never_sending": ("confirmed_not_sent", False),
        "telegram_provider_ack": ("confirmed_delivered", True),
        "operator_abandon": ("abandon", False),
    }[value["evidence_type"]]
    if value["outcome"] != expected[0] or (value["receipt"] is not None) != expected[1]:
        raise ReconciliationUnsafeError("evidence outcome is inconsistent")
    expected_statuses = {
        "claimed_never_sending": {"claimed"},
        "telegram_provider_ack": {"sending", "uncertain"},
        "operator_abandon": {"sending", "uncertain"},
    }[value["evidence_type"]]
    if predecessor["status"] not in expected_statuses:
        raise ReconciliationUnsafeError("evidence predecessor state is inconsistent")
    expected_attestation = (
        "Telegram acceptance; not proof of user read"
        if value["evidence_type"] == "telegram_provider_ack"
        else "operator attestation; no provider lookup was performed"
    )
    if attestation != expected_attestation:
        raise ReconciliationUnsafeError("evidence attestation is inconsistent")
    if value["receipt"] is not None:
        _validate_redacted_receipt(value["receipt"])
    return value


def _validate_redacted_receipt(value: object) -> dict[str, Any]:
    fields = {"schema", "channel", "target_digest", "action_id", "request_digest", "parts"}
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
        raise ReconciliationUnsafeError("redacted Telegram receipt evidence is malformed")
    value = cast(dict[str, Any], value)  # Exact raw receipt wire field set.
    if value["schema"] != "nanobot.channel_text_receipt.v1" or value["channel"] != "telegram":
        raise ReconciliationUnsafeError("redacted Telegram receipt schema is malformed")
    _digest(value["target_digest"])
    action_id = _digest(value["action_id"])
    request_digest = _digest(value["request_digest"])
    if not isinstance(value["parts"], list):
        raise ReconciliationUnsafeError("redacted Telegram receipt parts are malformed")
    parts: list[ChannelTextPartReceipt] = []
    for item in cast(list[object], value["parts"]):
        required = {"index", "message_index", "part_index", "provider_message_id", "provider_chat_id"}
        if not isinstance(item, dict) or set(cast(dict[object, object], item)) not in (required, required | {"provider_timestamp"}):
            raise ReconciliationUnsafeError("redacted Telegram receipt part is malformed")
        # Preserve the dataclass's validation and the original exception wrapping.
        item = cast(dict[str, Any], item)
        try:
            parts.append(ChannelTextPartReceipt(**item))
        except (TypeError, ValueError) as exc:
            raise ReconciliationUnsafeError("redacted Telegram receipt part is malformed") from exc
    try:
        ChannelTextReceipt("telegram", "redacted", action_id, request_digest, tuple(parts))
    except ValueError as exc:
        raise ReconciliationUnsafeError("redacted Telegram receipt is malformed") from exc
    return value


def create_plan(
    outbox_root: Path, *, namespace: str, artifacts_root: Path, evidence_ref: str
) -> StoredArtifact:
    _text(namespace)
    outbox_root, artifacts_root = validate_authority_selectors(outbox_root, artifacts_root)
    match = _REF_RE.fullmatch(evidence_ref)
    if match is None:
        raise ReconciliationUnsafeError("evidence reference must be sha256:<digest>")
    with _ArtifactStore(artifacts_root, create=False) as store:
        evidence = _parse_evidence(store.read("evidence", match.group(1)))
        outbox_binding = _binding(outbox_root)
        records = CronDeliveryOutboxStore.inspect_existing(outbox_root, namespace=namespace)
        _assert_binding(outbox_binding, outbox_root)
        record = next((item for item in records if item.outbox_id == evidence["outbox_id"]), None)
        if record is None:
            raise ReconciliationConflictError("evidence outbox is unavailable")
        destination, attempt = _current_attempt(record, evidence["destination_id"])
        predecessor = evidence["predecessor"]
        exact = {
            "attempt": attempt.attempt, "owner": attempt.owner, "status": destination.status,
            "action_id": attempt.action_id, "request_digest": attempt.request_digest,
            "record_revision": record.record_revision,
        }
        if evidence["namespace"] != namespace or predecessor != exact:
            raise ReconciliationConflictError("evidence predecessor is stale or foreign")
        if evidence["outcome"] not in _permitted(destination.status):
            raise ReconciliationConflictError("evidence outcome is not safe for predecessor")
        receipt = evidence["receipt"]
        if receipt is not None:
            _restore_receipt(
                receipt, destination=destination, attempt=attempt, count=len(record.messages)
            )
        plan_body = {
            "schema": "nanobot.cron_delivery_reconciliation_plan.v1",
            "namespace": namespace,
            "authorities": {
                "outbox": _binding_dict(outbox_binding),
                "artifacts": _binding_dict(_binding(artifacts_root)),
            },
            "outbox_id": record.outbox_id,
            "destination_id": destination.destination_id,
            "channel": destination.channel,
            "target_redacted": True,
            "predecessor": exact,
            "outcome": evidence["outcome"],
            "evidence_ref": evidence_ref,
            "predecessor_fenced_ref": evidence_ref,
            "receipt": receipt,
        }
        plan_id = hashlib.sha256(_PLAN_DOMAIN + _canonical(plan_body)).hexdigest()
        artifact = {"plan_id": plan_id, **plan_body}
        stored = store.publish("plans", artifact)
        return StoredArtifact(stored.kind, stored.digest, stored.reference, plan_id)


def _parse_plan(value: object) -> dict[str, Any]:
    fields = {
        "plan_id", "schema", "namespace", "authorities", "outbox_id", "destination_id",
        "channel", "target_redacted", "predecessor", "outcome", "evidence_ref",
        "predecessor_fenced_ref", "receipt",
    }
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
        raise ReconciliationUnsafeError("plan artifact schema is malformed")
    value = cast(dict[str, Any], value)  # Exact raw artifact wire field set.
    if value["schema"] != "nanobot.cron_delivery_reconciliation_plan.v1":
        raise ReconciliationUnsafeError("plan artifact schema is unsupported")
    plan_id = _digest(value["plan_id"])
    body = {key: item for key, item in value.items() if key != "plan_id"}
    if hashlib.sha256(_PLAN_DOMAIN + _canonical(body)).hexdigest() != plan_id:
        raise ReconciliationUnsafeError("plan identifier does not match its content")
    if not isinstance(value["authorities"], dict) or set(cast(dict[object, object], value["authorities"])) != {"outbox", "artifacts"}:
        raise ReconciliationUnsafeError("plan authority schema is malformed")
    _binding_from(cast(dict[str, object], value["authorities"])["outbox"])
    _binding_from(cast(dict[str, object], value["authorities"])["artifacts"])
    _text(value["namespace"])
    _digest(value["outbox_id"])
    _digest(value["destination_id"])
    _text(value["channel"])
    if value["target_redacted"] is not True:
        raise ReconciliationUnsafeError("plan target must remain redacted")
    if value["outcome"] not in {"confirmed_not_sent", "confirmed_delivered", "abandon"}:
        raise ReconciliationUnsafeError("plan outcome is unsupported")
    for field in ("evidence_ref", "predecessor_fenced_ref"):
        if not isinstance(value[field], str) or _REF_RE.fullmatch(value[field]) is None:
            raise ReconciliationUnsafeError("plan artifact reference is malformed")
    if value["predecessor_fenced_ref"] != value["evidence_ref"]:
        raise ReconciliationUnsafeError("plan fence evidence reference is inconsistent")
    predecessor = value["predecessor"]
    predecessor_fields = {
        "attempt", "owner", "status", "action_id", "request_digest", "record_revision"
    }
    if not isinstance(predecessor, dict) or set(cast(dict[object, object], predecessor)) != predecessor_fields:
        raise ReconciliationUnsafeError("plan predecessor is malformed")
    predecessor = cast(dict[str, Any], predecessor)  # Exact predecessor wire fields.
    _integer(predecessor["attempt"], minimum=1)
    _text(predecessor["owner"])
    if predecessor["status"] not in {"claimed", "sending", "uncertain"}:
        raise ReconciliationUnsafeError("plan predecessor state is malformed")
    _digest(predecessor["action_id"])
    _digest(predecessor["request_digest"])
    _integer(predecessor["record_revision"])
    permitted = _permitted(predecessor["status"])
    if value["outcome"] not in permitted:
        raise ReconciliationUnsafeError("plan outcome is inconsistent with predecessor")
    if value["outcome"] == "confirmed_delivered":
        receipt = _validate_redacted_receipt(value["receipt"])
        if (
            receipt["action_id"] != predecessor["action_id"]
            or receipt["request_digest"] != predecessor["request_digest"]
        ):
            raise ReconciliationUnsafeError("plan receipt predecessor binding is inconsistent")
    elif value["receipt"] is not None:
        raise ReconciliationUnsafeError("plan receipt is inconsistent with outcome")
    return value


def show_plan(
    outbox_root: Path, *, namespace: str, artifacts_root: Path, plan_id: str
) -> dict[str, Any]:
    _text(namespace)
    outbox_root, artifacts_root = validate_authority_selectors(outbox_root, artifacts_root)
    with _ArtifactStore(artifacts_root, create=False) as store:
        # The artifact address has its own digest; plans are selected by their
        # semantic plan_id, so scan the bounded immutable namespace.
        for name in sorted(os.listdir(store.kind_fds["plans"])):
            candidate = _parse_plan(store.read("plans", name[:-5]))
            if candidate["plan_id"] == _digest(plan_id):
                if candidate["namespace"] != namespace:
                    raise ReconciliationConflictError("plan namespace is foreign")
                _assert_binding(
                    _binding_from(candidate["authorities"]["outbox"]), outbox_root
                )
                _assert_binding(
                    _binding_from(candidate["authorities"]["artifacts"]), artifacts_root
                )
                return {
                    "plan_id": candidate["plan_id"],
                    "outbox_id": candidate["outbox_id"],
                    "destination_id": candidate["destination_id"],
                    "channel": candidate["channel"],
                    "target_redacted": True,
                    "status": candidate["predecessor"]["status"],
                    "attempt": candidate["predecessor"]["attempt"],
                    "record_revision": candidate["predecessor"]["record_revision"],
                    "outcome": candidate["outcome"],
                    "evidence_ref": candidate["evidence_ref"],
                    "receipt_schema": None if candidate["receipt"] is None else candidate["receipt"]["schema"],
                    "receipt_part_count": 0 if candidate["receipt"] is None else len(candidate["receipt"]["parts"]),
                }
    raise ReconciliationConflictError("plan does not exist")


def _load_plan(store: _ArtifactStore, plan_id: str) -> dict[str, Any]:
    plan_id = _digest(plan_id)
    for name in sorted(os.listdir(store.kind_fds["plans"])):
        plan = _parse_plan(store.read("plans", name[:-5]))
        if plan["plan_id"] == plan_id:
            return plan
    raise ReconciliationConflictError("plan does not exist")


def apply_plan(
    outbox_root: Path, *, namespace: str, artifacts_root: Path, plan_id: str,
    confirm_plan: str, confirmation: str,
) -> dict[str, Any]:
    _text(namespace)
    outbox_root, artifacts_root = validate_authority_selectors(outbox_root, artifacts_root)
    if confirm_plan != plan_id:
        raise ReconciliationConflictError("exact plan confirmation is required")
    expected_literal = {
        "confirmed_not_sent": "CONFIRM-NOT-SENT",
        "confirmed_delivered": "CONFIRM-DELIVERED",
        "abandon": "CONFIRM-ABANDON",
    }
    with _ArtifactStore(artifacts_root, create=False) as store:
        plan = _load_plan(store, plan_id)
        if confirmation != expected_literal[plan["outcome"]]:
            raise ReconciliationConflictError("exact outcome confirmation is required")
        if plan["namespace"] != namespace:
            raise ReconciliationConflictError("plan namespace is foreign")
        try:
            _assert_binding(_binding_from(plan["authorities"]["outbox"]), outbox_root)
        except ReconciliationUnsafeError as exc:
            raise ReconciliationConflictError("plan outbox authority is unavailable") from exc
        _assert_binding(_binding_from(plan["authorities"]["artifacts"]), artifacts_root)
        evidence_match = _REF_RE.fullmatch(plan["evidence_ref"])
        assert evidence_match is not None
        evidence = _parse_evidence(store.read("evidence", evidence_match.group(1)))
        if (
            evidence["namespace"] != namespace
            or evidence["outbox_id"] != plan["outbox_id"]
            or evidence["destination_id"] != plan["destination_id"]
            or evidence["predecessor"] != plan["predecessor"]
            or evidence["outcome"] != plan["outcome"]
            or evidence["receipt"] != plan["receipt"]
        ):
            raise ReconciliationConflictError("plan evidence binding conflicts")
        store.require_reconciliation_binding()

        # Construction is safe here because apply requires an existing, fully
        # bound outbox and is itself the explicitly confirmed mutating phase.
        outbox_binding = _binding_from(plan["authorities"]["outbox"])
        try:
            outbox = CronDeliveryOutboxStore.open_existing(
                outbox_root,
                namespace=namespace,
                expected_root_identity=(outbox_binding.device, outbox_binding.inode),
            )
        except (
            ValueError,
            DeliveryOutboxConflictError,
            DeliveryOutboxPersistenceError,
        ) as exc:
            raise ReconciliationConflictError("plan outbox authority is unavailable") from exc
        except DeliveryOutboxCorruptError as exc:
            raise ReconciliationUnsafeError("plan outbox authority is malformed") from exc
        predecessor = plan["predecessor"]
        try:
            current = outbox.get(plan["outbox_id"])
            current_destination, current_attempt = _current_attempt(
                current, plan["destination_id"]
            )
        except (
            DeliveryOutboxNotFoundError,
            DeliveryOutboxConflictError,
            DeliveryOutboxPersistenceError,
            ReconciliationConflictError,
        ) as exc:
            raise ReconciliationConflictError("plan predecessor is unavailable") from exc
        except DeliveryOutboxCorruptError as exc:
            raise ReconciliationUnsafeError("plan predecessor is malformed") from exc
        strict_receipt = None
        if plan["receipt"] is not None:
            strict_receipt = _restore_receipt(
                plan["receipt"],
                destination=current_destination,
                attempt=current_attempt,
                count=len(current.messages),
            )
        try:
            outbox.resolve_after_reconciliation(
                plan["outbox_id"],
                plan["destination_id"],
                expected_revision=predecessor["record_revision"],
                action_id=predecessor["action_id"],
                request_digest=predecessor["request_digest"],
                evidence_ref=plan["evidence_ref"],
                predecessor_fenced_ref=plan["predecessor_fenced_ref"],
                resolution=plan["outcome"],
                receipt=strict_receipt,
                predecessor_attempt=predecessor["attempt"],
                predecessor_owner=predecessor["owner"],
            )
        except DeliverySendBusyError:
            raise
        except DeliveryOutboxStaleRevisionError as exc:
            raise ReconciliationConflictError("plan predecessor revision is stale") from exc
        except (DeliveryOutboxConflictError, DeliveryOutboxOwnershipError, DeliveryOutboxNotFoundError) as exc:
            raise ReconciliationConflictError("plan predecessor conflicts with current state") from exc
        except DeliveryOutboxPersistenceError as exc:
            raise ReconciliationConflictError("plan predecessor became unavailable") from exc
        receipt_body = {
            "schema": "nanobot.cron_delivery_reconciliation_application.v1",
            "plan_id": plan["plan_id"],
            "namespace": namespace,
            "outbox_id": plan["outbox_id"],
            "destination_id": plan["destination_id"],
            "outcome": plan["outcome"],
            "evidence_ref": plan["evidence_ref"],
            "applied_transition_revision": predecessor["record_revision"] + 1,
        }
        application = store.publish("applications", receipt_body)
        return {
            "plan_id": plan["plan_id"],
            "application_ref": application.reference,
            "outbox_id": plan["outbox_id"],
            "destination_id": plan["destination_id"],
            "outcome": plan["outcome"],
            "record_revision": predecessor["record_revision"] + 1,
        }


__all__ = [
    "AuthorityBinding", "DeliveryReconciliationError", "ReconciliationConflictError",
    "ReconciliationEntry", "ReconciliationInspection", "ReconciliationUnsafeError",
    "StoredArtifact", "apply_plan", "create_plan", "import_evidence",
    "inspect_reconciliation", "show_plan", "validate_authority_selectors",
]
