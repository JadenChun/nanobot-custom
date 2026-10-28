"""Durable evidence for processes that can outlive their command roots."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import re
import stat
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, Literal, TypedDict, cast

from filelock import FileLock, Timeout

if TYPE_CHECKING:
    from nanobot.security.desktop_permissions import (
        DesktopPermissionOwner,
        WorkspacePermissionReference,
    )

_MAX_LEDGER_BYTES = 1024 * 1024
_MAX_RESERVATIONS = 4096
_MAX_OUTCOMES = 256
_RECEIPT_PUBLICATION_LOCK_TIMEOUT_SECONDS = 5.0
_LAUNCH_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_LOCAL_LAUNCHES_LOCK = threading.RLock()
_LOCAL_LAUNCHES: dict[str, set[str]] = {}

ReservationState = Literal["starting", "active", "unknown"]
AdmissionOutcome = Literal["quiescent", "unknown_after_boot", "not_started"]


class HostIdentityPayload(TypedDict):
    host_id: str | None
    boot_id: str | None
    pid_namespace: int | None


class NativeOwnerTag(TypedDict):
    kind: Literal["task_once", "chat_full"]
    runtime_id: str
    owner_id: str
    epoch: int
    session_key: str
    store_identity: str
    incarnation: str
    generation: int
    task_input_event_id: str | None


class ReservationRecord(TypedDict):
    launch_id: str
    host_id: str | None
    boot_id: str | None
    pid_namespace: int | None
    supervisor_pid: int | None
    supervisor_starttime: int | None
    state: ReservationState
    created_ns: int
    owner: NativeOwnerTag | None


class AdmissionOutcomeRecord(TypedDict):
    launch_id: str
    result: AdmissionOutcome
    host_id: str | None
    boot_id: str | None
    pid_namespace: int | None
    created_ns: int
    completed_ns: int
    owner: NativeOwnerTag | None


class AdmissionLedgerRecord(TypedDict):
    version: Literal[1, 2]
    reservations: list[ReservationRecord]
    outcomes: list[AdmissionOutcomeRecord]


class QuiescenceReceipt(TypedDict):
    version: Literal[1]
    launch_id: str
    host_id: str
    boot_id: str
    pid_namespace: int
    supervisor_pid: int
    supervisor_starttime: int
    result: Literal["quiescent"]
    completed_ns: int


class NativeAdmissionActiveError(RuntimeError):
    """A previously admitted native process is not proven quiescent."""


class NativeAdmissionUnknownError(NativeAdmissionActiveError):
    """An admitted native process lost its live supervisor proof in this boot."""


@dataclass(frozen=True, slots=True)
class HostIdentity:
    host_id: str | None
    boot_id: str | None
    pid_namespace: int | None

    @property
    def trusted(self) -> bool:
        return bool(self.host_id and self.boot_id and self.pid_namespace)

    def payload(self) -> HostIdentityPayload:
        return {
            "host_id": self.host_id,
            "boot_id": self.boot_id,
            "pid_namespace": self.pid_namespace,
        }


@dataclass(frozen=True, slots=True)
class NativeProcessReservation:
    launch_id: str
    ledger: NativeAdmissionLedger
    owner: NativeOwnerTag | None = None
    permission_owner: DesktopPermissionOwner | None = None
    permission_reference: WorkspacePermissionReference | None = None

    def acknowledge(self, supervisor_pid: int, supervisor_starttime: int) -> None:
        self.ledger.acknowledge(self, supervisor_pid, supervisor_starttime)

    def cancel_before_launch(self) -> None:
        self.ledger.cancel_before_launch(self)

    def cancel_if_unlaunched(self) -> bool:
        return self.ledger.cancel_if_unlaunched(self)

    def mark_unproven(self) -> None:
        self.ledger.mark_unproven(self)

    def admit_launch(self, supervisor_pid: int, supervisor_starttime: int, open_gate: Callable[[], None]) -> None:
        """Acknowledge and open the spawn gate while private authority is fenced."""
        def admit() -> None:
            self.ledger.acknowledge(self, supervisor_pid, supervisor_starttime)
            open_gate()

        permission_owner = self.permission_owner
        permission_reference = self.permission_reference
        if permission_owner is None or permission_reference is None:
            admit()
            return
        if not permission_owner.with_current_reference(permission_reference, admit):
            raise PermissionError("workspace permission changed before native launch")


class NativeAdmissionLedger:
    """Locked, bounded on-disk process-tree barrier for first grant activation."""

    def __init__(self, path: Path) -> None:
        requested = path.expanduser()
        self.path = requested.parent.resolve(strict=False) / requested.name
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.marker_path = self.path.with_name(f".{self.path.name}.initialized")
        self.receipt_dir = self.path.with_name(f".{self.path.stem}-receipts")

    @contextmanager
    def locked(self) -> Generator[None, None, None]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_directory(self.path.parent)
        _reject_symlink_if_present(self.lock_path)
        try:
            with FileLock(str(self.lock_path), timeout=5):
                os.chmod(self.lock_path, 0o600)
                _reject_symlink_if_present(self.lock_path)
                yield
        except Timeout as exc:
            raise OSError("native process admission ledger is busy") from exc

    def begin(self, *, managed: Callable[[], bool]) -> NativeProcessReservation | None:
        with self.locked():
            ledger = self._read()
            reconciled, changed, consumed = self._reconcile(ledger)
            if changed:
                self._write(reconciled)
                self._cleanup_receipts(reconciled)
                self._forget_local(consumed)
            if managed():
                return None
            identity = current_host_identity()
            reservations = reconciled["reservations"]
            if len(reservations) >= _MAX_RESERVATIONS:
                raise ValueError("native process admission ledger is full")
            launch_id = uuid.uuid4().hex
            reservation: ReservationRecord = {
                "launch_id": launch_id,
                **identity.payload(),
                "supervisor_pid": None,
                "supervisor_starttime": None,
                "state": "starting",
                "created_ns": time.time_ns(),
                "owner": None,
            }
            reservations.append(reservation)
            self._write(reconciled)
            with _LOCAL_LAUNCHES_LOCK:
                _LOCAL_LAUNCHES.setdefault(str(self.path), set()).add(launch_id)
            return NativeProcessReservation(launch_id, self)

    def begin_owned(
        self,
        owner: NativeOwnerTag,
        *,
        permission_owner: DesktopPermissionOwner,
        permission_reference: WorkspacePermissionReference,
    ) -> NativeProcessReservation:
        """Reserve a process for one private permission owner, upgrading v1 on write."""
        checked_owner = _validate_owner(owner)
        with self.locked():
            ledger = self._read()
            reconciled, changed, consumed = self._reconcile(ledger)
            if changed:
                self._write(reconciled)
                self._cleanup_receipts(reconciled)
                self._forget_local(consumed)
            if len(reconciled["reservations"]) >= _MAX_RESERVATIONS:
                raise ValueError("native process admission ledger is full")
            identity = current_host_identity()
            launch_id = uuid.uuid4().hex
            reservation: ReservationRecord = {
                "launch_id": launch_id,
                **identity.payload(),
                "supervisor_pid": None,
                "supervisor_starttime": None,
                "state": "starting",
                "created_ns": time.time_ns(),
                "owner": checked_owner,
            }
            upgraded = _upgrade_ledger(reconciled)
            upgraded["reservations"].append(reservation)
            self._write(upgraded)
            with _LOCAL_LAUNCHES_LOCK:
                _LOCAL_LAUNCHES.setdefault(str(self.path), set()).add(launch_id)
            return NativeProcessReservation(
                launch_id,
                self,
                owner=checked_owner,
                permission_owner=permission_owner,
                permission_reference=permission_reference,
            )

    def acknowledge(
        self,
        reservation: NativeProcessReservation,
        supervisor_pid: int,
        supervisor_starttime: int,
    ) -> None:
        with self.locked():
            ledger = self._read()
            item = self._find(ledger, reservation.launch_id)
            identity = current_host_identity()
            if item["state"] != "starting" or not identity.trusted:
                raise ValueError("native process reservation is not acknowledgeable")
            if (
                item["host_id"] != identity.host_id
                or item["boot_id"] != identity.boot_id
                or item["pid_namespace"] != identity.pid_namespace
                or process_start_time(supervisor_pid) != supervisor_starttime
                or process_pid_namespace(supervisor_pid) != identity.pid_namespace
            ):
                raise ValueError("native process supervisor identity could not be verified")
            item["supervisor_pid"] = supervisor_pid
            item["supervisor_starttime"] = supervisor_starttime
            item["state"] = "active"
            self._write(ledger)

    def cancel_before_launch(self, reservation: NativeProcessReservation) -> None:
        with self.locked():
            ledger = self._read()
            item = self._find(ledger, reservation.launch_id)
            if item["state"] != "starting":
                raise ValueError("only an unlaunched native process can be cancelled")
            ledger["reservations"].remove(item)
            self._append_outcome(ledger, item, "not_started")
            self._write(ledger)
            self._forget_local(reservation.launch_id)

    def cancel_if_unlaunched(self, reservation: NativeProcessReservation) -> bool:
        with self.locked():
            ledger = self._read()
            item = next(
                (record for record in ledger["reservations"] if record["launch_id"] == reservation.launch_id),
                None,
            )
            if item is None:
                return False
            if item["state"] != "starting":
                return False
            ledger["reservations"].remove(item)
            self._append_outcome(ledger, item, "not_started")
            self._write(ledger)
            self._forget_local(reservation.launch_id)
            return True

    def mark_unproven(self, reservation: NativeProcessReservation) -> None:
        with self.locked():
            ledger = self._read()
            item = self._find(ledger, reservation.launch_id)
            item["state"] = "unknown"
            self._write(ledger)
            self._forget_local(reservation.launch_id)

    def reconcile(self) -> bool:
        """Consume valid supervisor receipts and return whether activation is blocked."""
        with self.locked():
            ledger = self._read()
            reconciled, changed, _consumed = self._reconcile(ledger)
            if changed:
                self._write(reconciled)
                self._cleanup_receipts(reconciled)
                self._forget_local(_consumed)
            return bool(reconciled["reservations"])

    def owner_counts(self, owner: NativeOwnerTag) -> tuple[int, int]:
        """Return active/unknown reservations for one exact private owner tag."""
        checked = _validate_owner(owner)
        with self.locked():
            ledger = self._read()
            reconciled, changed, consumed = self._reconcile(ledger)
            if changed:
                self._write(reconciled)
                self._cleanup_receipts(reconciled)
                self._forget_local(consumed)
            owned = [
                item for item in reconciled["reservations"]
                if item["owner"] == checked
            ]
            return (
                sum(item["state"] in {"starting", "active"} for item in owned),
                sum(item["state"] == "unknown" for item in owned),
            )

    def source_counts(
        self,
        *,
        session_key: str,
        store_identity: str,
        incarnation: str,
        generation: int,
    ) -> tuple[int, int]:
        """Find durable owned residuals for a source after volatile owners are lost."""
        with self.locked():
            ledger = self._read()
            reconciled, changed, consumed = self._reconcile(ledger)
            if changed:
                self._write(reconciled)
                self._cleanup_receipts(reconciled)
                self._forget_local(consumed)
            owned = [
                item for item in reconciled["reservations"]
                if item["owner"] is not None
                and item["owner"]["session_key"] == session_key
                and item["owner"]["store_identity"] == store_identity
                and item["owner"]["incarnation"] == incarnation
                and item["owner"]["generation"] == generation
            ]
            return (
                sum(item["state"] in {"starting", "active"} for item in owned),
                sum(item["state"] == "unknown" for item in owned),
            )

    def session_owner_counts(
        self,
        *,
        session_key: str,
        kind: Literal["task_once", "chat_full"] | None = None,
    ) -> tuple[int, int]:
        """Reconcile durable owner records for a chat without adopting proof.

        This intentionally spans source generations: a reset or recreated
        session must still account for a process tree owned by its old source.
        ``kind`` lets callers project Once and Full residuals independently.
        """
        if not session_key or len(session_key) > 256:
            raise ValueError("invalid native owner session key")
        with self.locked():
            ledger = self._read()
            reconciled, changed, consumed = self._reconcile(ledger)
            if changed:
                self._write(reconciled)
                self._cleanup_receipts(reconciled)
                self._forget_local(consumed)
            owned = [
                item for item in reconciled["reservations"]
                if item["owner"] is not None
                and item["owner"]["session_key"] == session_key
                and (kind is None or item["owner"]["kind"] == kind)
            ]
            return (
                sum(item["state"] in {"starting", "active"} for item in owned),
                sum(item["state"] == "unknown" for item in owned),
            )

    @contextmanager
    def activation_guard(self, *, managed: Callable[[], bool]) -> Generator[None, None, None]:
        """Hold the same interprocess lock used by launches through activation."""
        with self.locked():
            if managed():
                yield
                return
            ledger = self._read()
            reconciled, changed, _consumed = self._reconcile(ledger)
            if changed:
                self._write(reconciled)
                self._cleanup_receipts(reconciled)
                self._forget_local(_consumed)
            if reconciled["reservations"]:
                if any(item["state"] == "unknown" for item in reconciled["reservations"]):
                    raise NativeAdmissionUnknownError("native_processes_unknown")
                raise NativeAdmissionActiveError("native_processes_active")
            yield

    def _reconcile(
        self,
        ledger: AdmissionLedgerRecord,
    ) -> tuple[AdmissionLedgerRecord, bool, list[str]]:
        changed = False
        identity = current_host_identity()
        keep: list[ReservationRecord] = []
        completed: list[AdmissionOutcomeRecord] = list(ledger["outcomes"])
        consumed: list[str] = []
        for item in ledger["reservations"]:
            receipt = self._read_receipt(item["launch_id"])
            if receipt is not None:
                if not self._receipt_matches(item, receipt):
                    raise ValueError("native process receipt does not match its reservation")
                completed.append(self._outcome(item, "quiescent"))
                changed = True
                consumed.append(item["launch_id"])
                continue

            if _same_execution_identity(item, identity):
                locally_starting = self._is_local_launch(item["launch_id"])
                if (item["state"] == "starting" and not locally_starting) or (
                    item["state"] == "active" and not _supervisor_is_live(item)
                ):
                    item["state"] = "unknown"
                    changed = True
                keep.append(item)
            elif _verified_later_boot(item, identity):
                completed.append(self._outcome(item, "unknown_after_boot"))
                changed = True
            else:
                if item["state"] != "unknown":
                    item["state"] = "unknown"
                    changed = True
                keep.append(item)

        if len(completed) > _MAX_OUTCOMES:
            completed = completed[-_MAX_OUTCOMES:]
            changed = True
        if changed:
            ledger = {
                "version": ledger["version"],
                "reservations": keep,
                "outcomes": completed,
            }
        return ledger, changed, consumed

    def _is_local_launch(self, launch_id: str) -> bool:
        with _LOCAL_LAUNCHES_LOCK:
            return launch_id in _LOCAL_LAUNCHES.get(str(self.path), set())

    def _forget_local(self, launch_ids: list[str] | str) -> None:
        ids = {launch_ids} if isinstance(launch_ids, str) else set(launch_ids)
        with _LOCAL_LAUNCHES_LOCK:
            local = _LOCAL_LAUNCHES.get(str(self.path))
            if local is None:
                return
            local.difference_update(ids)
            if not local:
                _LOCAL_LAUNCHES.pop(str(self.path), None)

    def _read(self) -> AdmissionLedgerRecord:
        raw = _read_regular(self.path, missing_ok=True)
        if raw is None:
            marker = _read_regular(self.marker_path, missing_ok=True)
            if marker is not None and marker != b"nanobot-native-admission-v1\n":
                raise ValueError("native process admission marker is invalid")
            if marker is not None:
                raise ValueError("native process admission ledger is missing")
            return {"version": 1, "reservations": [], "outcomes": []}
        if len(raw) > _MAX_LEDGER_BYTES:
            raise ValueError("native process admission ledger is too large")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("native process admission ledger is corrupt") from exc
        return _validate_ledger(payload)

    def _write(self, ledger: AdmissionLedgerRecord) -> None:
        marker = _read_regular(self.marker_path, missing_ok=True)
        if marker is None:
            _atomic_write(self.marker_path, b"nanobot-native-admission-v1\n")
        elif marker != b"nanobot-native-admission-v1\n":
            raise ValueError("native process admission marker is invalid")
        _reject_symlink_if_present(self.path)
        encoded = (
            json.dumps(_serialized_ledger(ledger), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        if len(encoded) > _MAX_LEDGER_BYTES:
            raise ValueError("native process admission ledger is too large")
        _atomic_write(self.path, encoded)

    def _read_receipt(self, launch_id: str) -> QuiescenceReceipt | None:
        try:
            _require_directory(self.receipt_dir)
        except FileNotFoundError:
            return None
        path = self.receipt_dir / f"{launch_id}.receipt"
        with _receipt_publication_guard(self.receipt_dir):
            try:
                _require_directory(self.receipt_dir)
            except FileNotFoundError:
                return None
            try:
                raw = _read_regular(path, missing_ok=True)
            except ValueError as exc:
                raise ValueError(f"native process receipt is invalid: {exc}") from exc
        if raw is None:
            return None
        if len(raw) > 4096:
            raise ValueError("native process receipt is too large")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("native process receipt is corrupt") from exc
        return _validate_receipt(payload)

    def _cleanup_receipts(self, ledger: AdmissionLedgerRecord) -> None:
        try:
            _require_directory(self.receipt_dir)
        except FileNotFoundError:
            return
        with _receipt_publication_guard(self.receipt_dir):
            try:
                _require_directory(self.receipt_dir)
            except FileNotFoundError:
                return
            active = {item["launch_id"] for item in ledger["reservations"]}
            changed = False
            for path in self.receipt_dir.iterdir():
                info = path.lstat()
                if not _is_private_regular_file(info):
                    raise ValueError("native process receipt directory contains invalid evidence")
                if re.fullmatch(r"\.([0-9a-f]{32})\.receipt\.([0-9a-f]{32})\.tmp", path.name):
                    path.unlink()
                    changed = True
                    continue
                match = re.fullmatch(r"([0-9a-f]{32})\.receipt", path.name)
                if match is None:
                    raise ValueError("native process receipt directory contains an unknown entry")
                if match.group(1) not in active:
                    path.unlink()
                    changed = True
            if changed:
                _fsync_directory(self.receipt_dir)

    @staticmethod
    def _find(ledger: AdmissionLedgerRecord, launch_id: str) -> ReservationRecord:
        item = next(
            (record for record in ledger["reservations"] if record["launch_id"] == launch_id),
            None,
        )
        if item is None:
            raise ValueError("native process reservation is missing")
        return item

    @staticmethod
    def _receipt_matches(item: ReservationRecord, receipt: QuiescenceReceipt) -> bool:
        return all(
            receipt[key] == item[key]
            for key in (
                "launch_id", "host_id", "boot_id", "pid_namespace",
                "supervisor_pid", "supervisor_starttime",
            )
        ) and receipt["result"] == "quiescent"

    @staticmethod
    def _outcome(item: ReservationRecord, result: AdmissionOutcome) -> AdmissionOutcomeRecord:
        return {
            "launch_id": item["launch_id"],
            "result": result,
            "host_id": item["host_id"],
            "boot_id": item["boot_id"],
            "pid_namespace": item["pid_namespace"],
            "created_ns": item["created_ns"],
            "completed_ns": time.time_ns(),
            "owner": item["owner"],
        }

    @classmethod
    def _append_outcome(
        cls,
        ledger: AdmissionLedgerRecord,
        item: ReservationRecord,
        result: AdmissionOutcome,
    ) -> None:
        ledger["outcomes"].append(cls._outcome(item, result))
        del ledger["outcomes"][:-_MAX_OUTCOMES]


def current_host_identity() -> HostIdentity:
    host_id: str | None = None
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            value = path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if value:
            host_id = value[:128]
            break
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        boot_id = None
    try:
        pid_namespace = os.stat("/proc/self/ns/pid").st_ino
    except OSError:
        pid_namespace = None
    return HostIdentity(host_id, boot_id or None, pid_namespace)


def process_start_time(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    closing = raw.rfind(")")
    if closing < 0:
        return None
    fields = raw[closing + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def process_pid_namespace(pid: int) -> int | None:
    try:
        return os.stat(f"/proc/{pid}/ns/pid").st_ino
    except OSError:
        return None


def write_quiescence_receipt(receipt_dir: Path, payload: QuiescenceReceipt) -> None:
    """Write a non-replayable receipt after a supervisor has observed ECHILD."""
    receipt = _validate_receipt(payload)
    receipt_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_directory(receipt_dir)
    encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    target = receipt_dir / f"{receipt['launch_id']}.receipt"
    temporary: Path | None = receipt_dir / f".{receipt['launch_id']}.receipt.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    opened: os.stat_result | None = None
    with _receipt_publication_guard(receipt_dir, wait_for_writer=True):
        _require_directory(receipt_dir)
        try:
            assert temporary is not None
            fd = os.open(temporary, flags, 0o600)
            opened = os.fstat(fd)
            os.fchmod(fd, 0o600)
            if not _is_private_regular_file(os.fstat(fd)):
                raise ValueError("native process receipt staging file is not private")
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            _rename_noreplace(temporary, target)
            temporary = None
            _fsync_directory(receipt_dir)
        except BaseException:
            if fd is not None:
                with suppress(OSError):
                    os.close(fd)
            if temporary is not None and opened is not None:
                try:
                    current = temporary.lstat()
                    if (
                        (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)
                        and _is_private_regular_file(current)
                    ):
                        temporary.unlink()
                        _fsync_directory(receipt_dir)
                except OSError:
                    pass
            raise
        else:
            os.close(fd)


def _json_object(payload: object, error: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(error)
    raw = cast(dict[object, object], payload)
    data: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(error)
        data[key] = value
    return data


def _upgrade_ledger(ledger: AdmissionLedgerRecord) -> AdmissionLedgerRecord:
    """Upgrade only when an explicitly owned reservation is about to be written."""
    return {
        "version": 2,
        "reservations": [
            {**item, "owner": item["owner"]}
            for item in ledger["reservations"]
        ],
        "outcomes": [
            {**item, "owner": item["owner"]}
            for item in ledger["outcomes"]
        ],
    }


def _serialized_ledger(ledger: AdmissionLedgerRecord) -> dict[str, object]:
    if ledger["version"] == 2:
        return {
            "version": 2,
            "reservations": ledger["reservations"],
            "outcomes": ledger["outcomes"],
        }
    return {
        "version": 1,
        "reservations": [
            {key: value for key, value in item.items() if key != "owner"}
            for item in ledger["reservations"]
        ],
        "outcomes": [
            {key: value for key, value in item.items() if key != "owner"}
            for item in ledger["outcomes"]
        ],
    }


def _validate_ledger(payload: object) -> AdmissionLedgerRecord:
    data = _json_object(payload, "native process admission ledger has an invalid shape")
    if set(data) != {"version", "reservations", "outcomes"}:
        raise ValueError("native process admission ledger has an invalid shape")
    version = data["version"]
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("native process admission ledger version is invalid")
    reservations_value = data["reservations"]
    outcomes_value = data["outcomes"]
    if not isinstance(reservations_value, list):
        raise ValueError("native process admission reservations are invalid")
    if not isinstance(outcomes_value, list):
        raise ValueError("native process admission outcomes are invalid")
    reservations = cast(list[object], reservations_value)
    outcomes = cast(list[object], outcomes_value)
    if len(reservations) > _MAX_RESERVATIONS:
        raise ValueError("native process admission reservations are invalid")
    if len(outcomes) > _MAX_OUTCOMES:
        raise ValueError("native process admission outcomes are invalid")
    seen: set[str] = set()
    ledger_version: Literal[1, 2] = cast(Literal[1, 2], version)
    validated = [_validate_reservation(item, ledger_version) for item in reservations]
    for item in validated:
        if item["launch_id"] in seen:
            raise ValueError("native process admission ids are duplicated")
        seen.add(item["launch_id"])
    validated_outcomes = [_validate_outcome(item, ledger_version) for item in outcomes]
    return {
        "version": ledger_version,
        "reservations": validated,
        "outcomes": validated_outcomes,
    }


def _validate_reservation(
    payload: object,
    version: Literal[1, 2],
) -> ReservationRecord:
    keys = {
        "launch_id", "host_id", "boot_id", "pid_namespace", "supervisor_pid",
        "supervisor_starttime", "state", "created_ns",
    }
    if version == 2:
        keys.add("owner")
    data = _json_object(payload, "native process reservation has an invalid shape")
    if set(data) != keys:
        raise ValueError("native process reservation has an invalid shape")
    launch_id = data["launch_id"]
    if not isinstance(launch_id, str) or not _LAUNCH_ID_RE.fullmatch(launch_id):
        raise ValueError("native process launch id is invalid")
    host_id = data["host_id"]
    boot_id = data["boot_id"]
    if host_id is not None and (
        not isinstance(host_id, str) or not host_id or len(host_id) > 128
    ):
        raise ValueError("native process host_id is invalid")
    if boot_id is not None and (
        not isinstance(boot_id, str) or not boot_id or len(boot_id) > 128
    ):
        raise ValueError("native process boot_id is invalid")
    pidns = data["pid_namespace"]
    if pidns is not None and (type(pidns) is not int or pidns <= 0):
        raise ValueError("native process pid namespace is invalid")
    supervisor_pid = data["supervisor_pid"]
    if supervisor_pid is not None and (type(supervisor_pid) is not int or supervisor_pid <= 1):
        raise ValueError("native process supervisor pid is invalid")
    starttime = data["supervisor_starttime"]
    if starttime is not None and (type(starttime) is not int or starttime < 0):
        raise ValueError("native process supervisor start time is invalid")
    state_value = data["state"]
    if state_value == "starting":
        state: ReservationState = "starting"
    elif state_value == "active":
        state = "active"
    elif state_value == "unknown":
        state = "unknown"
    else:
        raise ValueError("native process reservation state is invalid")
    if (supervisor_pid is None) != (starttime is None):
        raise ValueError("native process supervisor identity is incomplete")
    if state == "starting" and supervisor_pid is not None:
        raise ValueError("starting native process cannot have a supervisor identity")
    created_ns = data["created_ns"]
    if type(created_ns) is not int or created_ns < 0:
        raise ValueError("native process creation time is invalid")
    owner = _validate_optional_owner(data.get("owner"))
    return {
        "launch_id": launch_id,
        "host_id": host_id,
        "boot_id": boot_id,
        "pid_namespace": pidns,
        "supervisor_pid": supervisor_pid,
        "supervisor_starttime": starttime,
        "state": state,
        "created_ns": created_ns,
        "owner": owner,
    }


def _validate_outcome(
    payload: object,
    version: Literal[1, 2],
) -> AdmissionOutcomeRecord:
    keys = {
        "launch_id", "result", "host_id", "boot_id", "pid_namespace", "created_ns", "completed_ns",
    }
    if version == 2:
        keys.add("owner")
    data = _json_object(payload, "native process outcome has an invalid shape")
    if set(data) != keys:
        raise ValueError("native process outcome has an invalid shape")
    launch_id = data["launch_id"]
    if not isinstance(launch_id, str) or not _LAUNCH_ID_RE.fullmatch(launch_id):
        raise ValueError("native process outcome id is invalid")
    result_value = data["result"]
    if result_value == "quiescent":
        result: AdmissionOutcome = "quiescent"
    elif result_value == "unknown_after_boot":
        result = "unknown_after_boot"
    elif result_value == "not_started":
        result = "not_started"
    else:
        raise ValueError("native process outcome result is invalid")
    host_id = data["host_id"]
    boot_id = data["boot_id"]
    if host_id is not None and (
        not isinstance(host_id, str) or not host_id or len(host_id) > 128
    ):
        raise ValueError("native process outcome host_id is invalid")
    if boot_id is not None and (
        not isinstance(boot_id, str) or not boot_id or len(boot_id) > 128
    ):
        raise ValueError("native process outcome boot_id is invalid")
    pidns = data["pid_namespace"]
    if pidns is not None and (type(pidns) is not int or pidns < 0):
        raise ValueError("native process outcome pid namespace is invalid")
    created_ns = data["created_ns"]
    completed_ns = data["completed_ns"]
    if type(created_ns) is not int or created_ns < 0:
        raise ValueError("native process outcome created_ns is invalid")
    if type(completed_ns) is not int or completed_ns < 0:
        raise ValueError("native process outcome completed_ns is invalid")
    owner = _validate_optional_owner(data.get("owner"))
    return {
        "launch_id": launch_id,
        "result": result,
        "host_id": host_id,
        "boot_id": boot_id,
        "pid_namespace": pidns,
        "created_ns": created_ns,
        "completed_ns": completed_ns,
        "owner": owner,
    }


def _validate_optional_owner(value: object) -> NativeOwnerTag | None:
    return None if value is None else _validate_owner(value)


def _validate_owner(value: object) -> NativeOwnerTag:
    keys = {
        "kind", "runtime_id", "owner_id", "epoch", "session_key",
        "store_identity", "incarnation", "generation", "task_input_event_id",
    }
    data = _json_object(value, "native process owner tag is invalid")
    if set(data) != keys:
        raise ValueError("native process owner tag has an invalid shape")
    kind = data["kind"]
    if kind not in {"task_once", "chat_full"}:
        raise ValueError("native process owner kind is invalid")
    runtime_id = data["runtime_id"]
    if not isinstance(runtime_id, str) or not _LAUNCH_ID_RE.fullmatch(runtime_id):
        raise ValueError("native process owner runtime id is invalid")
    owner_id = data["owner_id"]
    session_key = data["session_key"]
    store_identity = data["store_identity"]
    incarnation = data["incarnation"]
    if not isinstance(owner_id, str) or not 1 <= len(owner_id) <= 256:
        raise ValueError("native process owner id is invalid")
    if not isinstance(session_key, str) or not 1 <= len(session_key) <= 256:
        raise ValueError("native process owner session key is invalid")
    if not isinstance(store_identity, str) or not 1 <= len(store_identity) <= 128:
        raise ValueError("native process owner store identity is invalid")
    if not isinstance(incarnation, str) or not 1 <= len(incarnation) <= 128:
        raise ValueError("native process owner incarnation is invalid")
    epoch = data["epoch"]
    generation = data["generation"]
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("native process owner epoch is invalid")
    if type(generation) is not int or generation < 0:
        raise ValueError("native process owner generation is invalid")
    task_input_event_id = data["task_input_event_id"]
    if task_input_event_id is not None and (
        not isinstance(task_input_event_id, str)
        or not 1 <= len(task_input_event_id) <= 128
    ):
        raise ValueError("native process owner task event id is invalid")
    if (kind == "task_once") != (task_input_event_id is not None):
        raise ValueError("native process owner task event identity is inconsistent")
    return {
        "kind": cast(Literal["task_once", "chat_full"], kind),
        "runtime_id": runtime_id,
        "owner_id": owner_id,
        "epoch": epoch,
        "session_key": session_key,
        "store_identity": store_identity,
        "incarnation": incarnation,
        "generation": generation,
        "task_input_event_id": task_input_event_id,
    }


def _validate_receipt(payload: object) -> QuiescenceReceipt:
    keys = {
        "version", "launch_id", "host_id", "boot_id", "pid_namespace", "supervisor_pid",
        "supervisor_starttime", "result", "completed_ns",
    }
    data = _json_object(payload, "native process receipt has an invalid shape")
    if set(data) != keys:
        raise ValueError("native process receipt has an invalid shape")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("native process receipt version is invalid")
    launch_id = data["launch_id"]
    if not isinstance(launch_id, str) or not _LAUNCH_ID_RE.fullmatch(launch_id):
        raise ValueError("native process receipt id is invalid")
    host_id = data["host_id"]
    boot_id = data["boot_id"]
    if not isinstance(host_id, str) or not host_id or len(host_id) > 128:
        raise ValueError("native process receipt host_id is invalid")
    if not isinstance(boot_id, str) or not boot_id or len(boot_id) > 128:
        raise ValueError("native process receipt boot_id is invalid")
    pid_namespace = data["pid_namespace"]
    supervisor_pid = data["supervisor_pid"]
    supervisor_starttime = data["supervisor_starttime"]
    completed_ns = data["completed_ns"]
    if type(pid_namespace) is not int or pid_namespace <= 0:
        raise ValueError("native process receipt pid_namespace is invalid")
    if type(supervisor_pid) is not int or supervisor_pid <= 1:
        raise ValueError("native process receipt supervisor_pid is invalid")
    if type(supervisor_starttime) is not int or supervisor_starttime < 0:
        raise ValueError("native process receipt supervisor_starttime is invalid")
    if type(completed_ns) is not int or completed_ns < 0:
        raise ValueError("native process receipt completed_ns is invalid")
    if data["result"] != "quiescent":
        raise ValueError("native process receipt result is invalid")
    return {
        "version": 1,
        "launch_id": launch_id,
        "host_id": host_id,
        "boot_id": boot_id,
        "pid_namespace": pid_namespace,
        "supervisor_pid": supervisor_pid,
        "supervisor_starttime": supervisor_starttime,
        "result": "quiescent",
        "completed_ns": completed_ns,
    }


def _same_execution_identity(item: ReservationRecord, identity: HostIdentity) -> bool:
    return bool(
        identity.trusted
        and item["host_id"] == identity.host_id
        and item["boot_id"] == identity.boot_id
        and item["pid_namespace"] == identity.pid_namespace
    )


def _verified_later_boot(item: ReservationRecord, identity: HostIdentity) -> bool:
    return bool(
        item["host_id"]
        and item["boot_id"]
        and item["pid_namespace"]
        and identity.trusted
        and item["host_id"] == identity.host_id
        and item["boot_id"] != identity.boot_id
        and item["pid_namespace"] == identity.pid_namespace
    )


def _supervisor_is_live(item: ReservationRecord) -> bool:
    pid = item["supervisor_pid"]
    starttime = item["supervisor_starttime"]
    return bool(
        pid is not None
        and starttime is not None
        and process_start_time(pid) == starttime
        and process_pid_namespace(pid) == item["pid_namespace"]
    )


def _read_regular(path: Path, *, missing_ok: bool) -> bytes | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    if not _is_private_regular_file(info):
        raise ValueError("native process evidence must be a singly linked regular file")
    if info.st_size > _MAX_LEDGER_BYTES:
        raise ValueError("native process evidence is too large")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_uid,
            stat.S_IMODE(opened.st_mode) & 0o077,
        ) != (
            info.st_dev,
            info.st_ino,
            1,
            info.st_uid,
            stat.S_IMODE(info.st_mode) & 0o077,
        ):
            raise ValueError("native process evidence changed while opening")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(_MAX_LEDGER_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > _MAX_LEDGER_BYTES:
        raise ValueError("native process evidence is too large")
    return raw


@contextmanager
def _receipt_publication_guard(
    receipt_dir: Path,
    *,
    wait_for_writer: bool = False,
) -> Generator[None, None, None]:
    """Keep ledger readers from observing a partially written receipt."""
    _require_directory(receipt_dir.parent)
    lock_path = receipt_dir.with_name(f"{receipt_dir.name}.lock")
    _reject_symlink_if_present(lock_path)
    try:
        timeout = -1 if wait_for_writer else _RECEIPT_PUBLICATION_LOCK_TIMEOUT_SECONDS
        with FileLock(str(lock_path), timeout=timeout):
            os.chmod(lock_path, 0o600)
            _reject_symlink_if_present(lock_path)
            yield
    except Timeout as exc:
        raise NativeAdmissionActiveError("native_processes_active") from exc


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a complete receipt without replacing prior proof."""
    if sys.platform != "linux":
        raise OSError(errno.ENOTSUP, "atomic native receipt publication is unavailable")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOTSUP, "atomic native receipt publication is unavailable") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def _is_private_regular_file(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        return False
    if os.name != "posix":
        return True
    return info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) & 0o077 == 0


def _atomic_write(path: Path, encoded: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require_directory(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("native process evidence directory is not a real directory")


def _reject_symlink_if_present(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("native process lock is not a regular file")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
