"""Private, pointerless receipt-quarantine evidence below recovery policy.

Inspection is read-only and never acquires or creates a lock. Mutations require
an already-held workspace memory lock; this module never disposes of a receipt.
The committed marker is *structural* evidence, not startup admission proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, NotRequired, Self, TypedDict, cast

from nanobot.agent import memory_recovery_storage as storage

if TYPE_CHECKING:
    from nanobot.agent.memory_recovery_transaction_terminal import TerminalCompleteProof

_INDEX = "receipt-quarantine"
_PENDING = ".consolidation.pending.json"
_COPY = "receipt.bin"
_PHASES = ("prepared", "copy-ready", "committed")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_WORKSPACE_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_RECEIPT = 64 * 1024 * 1024  # Existing memory-target storage bound.
_MAX_RECORD = 4096
_PROOF_KEY = object()
_RECORD_KEYS = frozenset((
    "version", "phase", "predecessor_digest", "workspace", "workspace_stat",
    "workspace_id", "sessions_root", "sessions_root_stat", "transaction_identity",
    "complete_digest", "receipt_digest", "receipt_size", "receipt_stat", "copy_name",
    "copy_stat",
))
_SEALED_KEYS = _RECORD_KEYS | {"semantic_seal_digest"}


class _QuarantineRecord(TypedDict):
    """Validated canonical record; raw JSON never escapes the decoder."""

    version: int
    phase: str
    predecessor_digest: str | None
    workspace: str
    workspace_stat: list[int]
    workspace_id: str
    sessions_root: str
    sessions_root_stat: list[int]
    transaction_identity: str
    complete_digest: str
    receipt_digest: str
    receipt_size: int
    receipt_stat: list[int]
    copy_name: str | None
    copy_stat: list[int] | None
    semantic_seal_digest: NotRequired[str]


class QuarantineError(RuntimeError):
    """Content-free failure; paths and receipt contents never enter exceptions."""

    def __init__(self) -> None:
        super().__init__("receipt quarantine evidence is unavailable or unsafe")


def _fail() -> NoReturn:
    raise QuarantineError() from None


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _counter(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        _fail()
    return value


def _stat_tuple(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory_tuple(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _private_dir(info: os.stat_result, *, memory: bool = False) -> None:
    uid = getattr(os, "geteuid", None)
    if not stat.S_ISDIR(info.st_mode) or (callable(uid) and info.st_uid != uid()):
        _fail()
    mode = stat.S_IMODE(info.st_mode)
    if memory:
        if mode & 0o700 != 0o700 or mode & 0o022 or info.st_mode & 0o7000:
            _fail()
    elif mode != 0o700:
        _fail()


def _private_file(info: os.stat_result) -> None:
    uid = getattr(os, "geteuid", None)
    if (not stat.S_ISREG(info.st_mode) or not callable(uid) or info.st_uid != uid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
        _fail()


def _support() -> None:
    try:
        storage.require_memory_lock_gate_capabilities()
        if not all(f in os.supports_dir_fd for f in (os.open, os.stat, os.mkdir)):
            _fail()
    except storage.MemoryStorageError:
        _fail()


def _absolute(value: object) -> Path:
    if not isinstance(value, (Path, str)):
        _fail()
    try:
        path = Path(os.path.abspath(value))
    except (OSError, TypeError, ValueError):
        _fail()
    if not path.is_absolute() or path == Path(path.anchor) or ".." in path.parts:
        _fail()
    return path


def _open_absolute(path: Path) -> int | None:
    """No-follow absolute walk; a missing component is absence, not a write."""
    fd: int | None = None
    try:
        fd = storage.open_recovery_absolute_directory(os.open, path.anchor)
        for part in path.parts[1:]:
            before = os.stat(part, dir_fd=fd, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                _fail()
            child = storage.open_recovery_relative_directory(os.open, part, fd)
            if _directory_tuple(before) != _directory_tuple(os.fstat(child)):
                os.close(child)
                _fail()
            os.close(fd)
            fd = child
        return fd
    except FileNotFoundError:
        if fd is not None:
            os.close(fd)
        return None
    except (OSError, TypeError, ValueError):
        if fd is not None:
            os.close(fd)
        _fail()
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise


def _entry(fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError):
        _fail()


def _child(fd: int, name: str, *, memory: bool = False) -> int | None:
    before = _entry(fd, name)
    if before is None:
        return None
    _private_dir(before, memory=memory)
    try:
        child = storage.open_recovery_relative_directory(os.open, name, fd)
        after = os.fstat(child)
        if _stat_tuple(before) != _stat_tuple(after):
            os.close(child)
            _fail()
        return child
    except (OSError, TypeError, ValueError):
        _fail()


def _read(fd: int, name: str, limit: int) -> tuple[bytes, os.stat_result] | None:
    before = _entry(fd, name)
    if before is None:
        return None
    _private_file(before)
    leaf: int | None = None
    try:
        leaf = storage.open_recovery_read_leaf(os.open, name, fd)
        if _stat_tuple(os.fstat(leaf)) != _stat_tuple(before):
            _fail()
        if before.st_size > limit:
            _fail()
        data = bytearray()
        while len(data) <= limit:
            chunk = os.read(leaf, min(65536, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if (len(data) > limit or len(data) != before.st_size
                or _stat_tuple(os.fstat(leaf)) != _stat_tuple(before)):
            _fail()
        rebound = _entry(fd, name)
        if rebound is None or _stat_tuple(rebound) != _stat_tuple(before):
            _fail()
        return bytes(data), before
    except (OSError, TypeError, ValueError):
        _fail()
    finally:
        if leaf is not None:
            os.close(leaf)


def _names(fd: int, limit: int) -> set[str]:
    try:
        names: set[str] = set()
        with os.scandir(fd) as entries:
            for entry in entries:
                if entry.name in names or len(names) == limit:
                    _fail()
                names.add(entry.name)
    except (OSError, TypeError, ValueError):
        _fail()
    return names


def _index_names(fd: int) -> Iterator[str]:
    """Stream names without materializing an unbounded directory listing."""
    try:
        with os.scandir(fd) as entries:
            for entry in entries:
                yield entry.name
    except (OSError, TypeError, ValueError):
        _fail()


def _canonical(record: Mapping[str, object]) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n").encode("ascii")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            _fail()
        output[key] = value
    return output


def _bad_constant(_value: str) -> None:
    _fail()


def _stat_values(value: object, length: int) -> list[int]:
    if type(value) is not list:
        _fail()
    # Exact list check establishes the container; each element is checked next.
    items = cast(list[object], value)
    if len(items) != length:
        _fail()
    return [_counter(part) for part in items]


def _seal_digest(record: _QuarantineRecord) -> str:
    seal = record.get("semantic_seal_digest")
    if seal is None:
        _fail()
    return seal


def _parse(raw: bytes, phase: str, predecessor: str | None) -> _QuarantineRecord:
    if len(raw) > _MAX_RECORD:
        _fail()
    try:
        decoded: object = json.loads(raw, object_pairs_hook=_unique, parse_constant=_bad_constant)
    except (ValueError, UnicodeError, TypeError, RecursionError):
        _fail()
    if type(decoded) is not dict:
        _fail()
    # JSON object keys are strings, and _unique rejects duplicate keys.
    value = cast(dict[str, object], decoded)
    if _canonical(value) != raw:
        _fail()
    sealed = phase == "committed" and value.get("version") == 2
    if value.keys() != (_SEALED_KEYS if sealed else _RECORD_KEYS):
        _fail()
    if (type(value["version"]) is not int or value["version"] != (2 if sealed else 1)
            or value["phase"] != phase or value["predecessor_digest"] != predecessor):
        _fail()
    seal = value.get("semantic_seal_digest")
    if sealed and (type(seal) is not str or not _DIGEST.fullmatch(seal)):
        _fail()
    for key in ("transaction_identity", "complete_digest", "receipt_digest"):
        digest = value[key]
        if type(digest) is not str or not _DIGEST.fullmatch(digest):
            _fail()
    workspace_id = value["workspace_id"]
    if type(workspace_id) is not str or not _WORKSPACE_ID.fullmatch(workspace_id):
        _fail()
    for key in ("workspace", "sessions_root"):
        path = value[key]
        if type(path) is not str or len(path) > 4096 or _absolute(path) != Path(path):
            _fail()
    for key in ("workspace_stat", "sessions_root_stat"):
        _stat_values(value[key], 5)
    receipt_size = _counter(value["receipt_size"])
    if receipt_size > _MAX_RECEIPT:
        _fail()
    uid = getattr(os, "geteuid", None)
    if not callable(uid):
        _fail()
    receipt_stat = _stat_values(value["receipt_stat"], 9)
    if (receipt_stat[6] != value["receipt_size"] or not stat.S_ISREG(receipt_stat[2])
            or stat.S_IMODE(receipt_stat[2]) != 0o600 or receipt_stat[5] != 1
            or receipt_stat[3] != uid()):
        _fail()
    if phase == "prepared":
        if value["copy_name"] is not None or value["copy_stat"] is not None:
            _fail()
    else:
        item = _stat_values(value["copy_stat"], 9)
        if value["copy_name"] != _COPY:
            _fail()
        if (item[6] != value["receipt_size"] or not stat.S_ISREG(item[2])
                or stat.S_IMODE(item[2]) != 0o600 or item[5] != 1
                or item[3] != uid()):
            _fail()
    # Exact key set and every field's primitive/container type are validated above.
    return cast(_QuarantineRecord, value)


@dataclass(frozen=True, slots=True)
class QuarantineBinding:
    workspace: Path
    sessions_root: Path
    workspace_id: str
    transaction_identity: str
    complete_digest: str
    receipt_digest: str
    receipt_size: int

    def __post_init__(self) -> None:
        if (type(self.workspace_id) is not str or not _WORKSPACE_ID.fullmatch(self.workspace_id)
                or any(type(d) is not str or not _DIGEST.fullmatch(d) for d in
                       (self.transaction_identity, self.complete_digest, self.receipt_digest))
                or type(self.receipt_size) is not int or not 0 <= self.receipt_size <= _MAX_RECEIPT):
            _fail()
        _absolute(self.workspace)
        _absolute(self.sessions_root)


@dataclass(frozen=True, slots=True)
class QuarantineInspection:
    clean: bool
    incomplete: bool
    committed_identities: tuple[str, ...]
    sealed_identities: tuple[str, ...] = ()


def _walk_scope(workspace_fd: int) -> tuple[int | None, int | None, int | None]:
    memory_fd = _child(workspace_fd, "memory", memory=True)
    if memory_fd is None:
        return None, None, None
    recovery_fd = _child(memory_fd, ".recovery")
    if recovery_fd is None:
        return memory_fd, None, None
    index_fd = _child(recovery_fd, _INDEX)
    return memory_fd, recovery_fd, index_fd


def _assert_inspection_bindings(
    workspace: Path, workspace_fd: int, memory_fd: int | None,
    recovery_fd: int | None, index_fd: int | None,
    observed: tuple[os.stat_result | None, ...], pending: os.stat_result | None,
) -> None:
    reopened = _open_absolute(workspace)
    if reopened is None:
        _fail()
    try:
        root_baseline = observed[0]
        if (root_baseline is None
                or _stat_tuple(os.fstat(reopened)) != _stat_tuple(root_baseline)
                or _stat_tuple(os.fstat(workspace_fd)) != _stat_tuple(root_baseline)):
            _fail()
    finally:
        os.close(reopened)
    parent = workspace_fd
    for position, (name, child) in enumerate(
        (("memory", memory_fd), (".recovery", recovery_fd), (_INDEX, index_fd)), start=1
    ):
        entry = _entry(parent, name)
        if child is None:
            if entry is not None or observed[position] is not None:
                _fail()
            break
        baseline = observed[position]
        if (baseline is None or entry is None or _stat_tuple(entry) != _stat_tuple(baseline)
                or _stat_tuple(os.fstat(child)) != _stat_tuple(baseline)):
            _fail()
        parent = child
    if memory_fd is not None:
        current = _entry(memory_fd, _PENDING)
        if (current is None) != (pending is None):
            _fail()
        if current is not None and pending is not None and _stat_tuple(current) != _stat_tuple(pending):
            _fail()


def _inspect(workspace: Path, expected: QuarantineBinding | None) -> QuarantineInspection:
    fd = _open_absolute(workspace)
    if fd is None:
        rebound = _open_absolute(workspace)
        if rebound is not None:
            os.close(rebound)
            _fail()
        return QuarantineInspection(True, False, ())
    memory_fd = recovery_fd = index_fd = None
    try:
        _private_dir(os.fstat(fd), memory=True)
        memory_fd, recovery_fd, index_fd = _walk_scope(fd)
        observed = tuple(os.fstat(child) if child is not None else None for child in
                         (fd, memory_fd, recovery_fd, index_fd))
        if memory_fd is None:
            _assert_inspection_bindings(workspace, fd, memory_fd, recovery_fd, index_fd,
                                        observed, None)
            return QuarantineInspection(True, False, ())
        pending_read = _read(memory_fd, _PENDING, _MAX_RECEIPT)
        pending = pending_read[1] if pending_read is not None else None
        if index_fd is None:
            _assert_inspection_bindings(workspace, fd, memory_fd, recovery_fd, index_fd,
                                        observed, pending)
            return QuarantineInspection(pending is None, pending is not None, ())
        committed: list[str] = []
        sealed: list[str] = []
        incomplete = pending is not None
        matched_expected = expected is None
        found_any = False
        for name in _index_names(index_fd):
            found_any = True
            if not _DIGEST.fullmatch(name):
                _fail()
            identity_fd = _child(index_fd, name)
            if identity_fd is None:
                _fail()
            try:
                names = _names(identity_fd, 4)
                if not names <= {"prepared.json", "copy-ready.json", "committed.json", _COPY}:
                    _fail()
                previous: str | None = None
                records: list[_QuarantineRecord] = []
                for phase in _PHASES:
                    leaf = f"{phase}.json"
                    if leaf not in names:
                        break
                    result = _read(identity_fd, leaf, _MAX_RECORD)
                    if result is None:
                        _fail()
                    raw, _ = result
                    record = _parse(raw, phase, previous)
                    if record["transaction_identity"] != name:
                        _fail()
                    if records and any(record[key] != records[0][key] for key in (
                        "workspace", "workspace_stat", "workspace_id", "sessions_root",
                        "sessions_root_stat", "transaction_identity", "complete_digest",
                        "receipt_digest", "receipt_size", "receipt_stat",
                    )):
                        _fail()
                    if str(workspace) != record["workspace"] or list(_directory_tuple(os.fstat(fd))) != record["workspace_stat"]:
                        _fail()
                    if expected is not None and name == expected.transaction_identity and any((
                        record["workspace_id"] != expected.workspace_id,
                        record["transaction_identity"] != expected.transaction_identity,
                        record["complete_digest"] != expected.complete_digest,
                        record["receipt_digest"] != expected.receipt_digest,
                        record["receipt_size"] != expected.receipt_size,
                        record["sessions_root"] != str(_absolute(expected.sessions_root)),
                    )):
                        _fail()
                    if expected is not None and name == expected.transaction_identity:
                        matched_expected = True
                    root_fd = _open_absolute(Path(record["sessions_root"]))
                    if root_fd is None:
                        _fail()
                    try:
                        _private_dir(os.fstat(root_fd), memory=True)
                        if list(_directory_tuple(os.fstat(root_fd))) != record["sessions_root_stat"]:
                            _fail()
                    finally:
                        os.close(root_fd)
                    previous = _digest(raw)
                    records.append(record)
                if not records or any(f"{p}.json" in names for p in _PHASES[len(records):]):
                    if _COPY in names and not records:
                        _fail()
                    incomplete = True
                    continue
                copy = _read(identity_fd, _COPY, _MAX_RECEIPT)
                if pending is not None and list(_stat_tuple(pending)) != records[0]["receipt_stat"]:
                    # A different live receipt cannot be mistaken for the prepared one.
                    incomplete = True
                if pending_read is not None and _digest(pending_read[0]) != records[0]["receipt_digest"]:
                    incomplete = True
                if copy is not None:
                    if (_digest(copy[0]) != records[0]["receipt_digest"]
                            or len(copy[0]) != records[0]["receipt_size"]):
                        _fail()
                    for record in records[1:]:
                        if list(_stat_tuple(copy[1])) != record["copy_stat"]:
                            _fail()
                if len(records) > 1 and copy is None:
                    _fail()
                if len(records) == 1 or len(records) == 2:
                    incomplete = True
                elif len(records) == 3:
                    if records[1]["copy_stat"] != records[2]["copy_stat"]:
                        _fail()
                    committed.append(name)
                    if records[2]["version"] == 2:
                        sealed.append(name)
                rebound = _entry(index_fd, name)
                if rebound is None or _stat_tuple(rebound) != _stat_tuple(os.fstat(identity_fd)):
                    _fail()
            finally:
                os.close(identity_fd)
        if not matched_expected:
            _fail()
        _assert_inspection_bindings(workspace, fd, memory_fd, recovery_fd, index_fd,
                                    observed, pending)
        return QuarantineInspection(False, incomplete or not found_any,
                                    tuple(sorted(committed)), tuple(sorted(sealed)))
    finally:
        for child in (index_fd, recovery_fd, memory_fd, fd):
            if child is not None:
                os.close(child)


def inspect_receipt_quarantine(
    workspace: Path | str, *, expected: QuarantineBinding | None = None,
) -> QuarantineInspection:
    """No lock, mkdir, fsync, or other writes. Committed is not admission."""
    path = _absolute(workspace)
    if expected is not None and path != _absolute(expected.workspace):
        _fail()
    try:
        _support()
    except QuarantineError:
        return _unsupported_absence(path)
    try:
        return _inspect(path, expected)
    except (OSError, TypeError, ValueError, storage.MemoryStorageError):
        _fail()


@dataclass(frozen=True, slots=True)
class DurableQuarantineEvidence:
    """Storage durability only; historical semantic admission is a separate check."""

    sealed_identities: tuple[str, ...]
    semantic_seal_digests: tuple[str, ...]


def _sync_checked_file(fd: int, name: str, limit: int) -> None:
    before = _read(fd, name, limit)
    if before is None:
        _fail()
    leaf: int | None = None
    try:
        leaf = storage.open_recovery_read_leaf(os.open, name, fd)
        if _stat_tuple(os.fstat(leaf)) != _stat_tuple(before[1]):
            _fail()
        os.fsync(leaf)
        if _stat_tuple(os.fstat(leaf)) != _stat_tuple(before[1]):
            _fail()
        after = _read(fd, name, limit)
        if after is None or after[0] != before[0] or _stat_tuple(after[1]) != _stat_tuple(before[1]):
            _fail()
    finally:
        if leaf is not None:
            os.close(leaf)


def _sealed_fingerprints(
    gate: storage.WorkspaceMemoryLockGate, index_fd: int | None, *, sync: bool,
) -> tuple[tuple[str, str, tuple[object, ...]], ...]:
    inspection = _inspect(gate.workspace, None)
    if (inspection.incomplete or inspection.committed_identities != inspection.sealed_identities
            or (index_fd is None and not inspection.clean)):
        _fail()
    if index_fd is None:
        return ()
    result: list[tuple[str, str, tuple[object, ...]]] = []
    for name in _index_names(index_fd):
        if not _DIGEST.fullmatch(name):
            _fail()
        identity_fd = _child(index_fd, name)
        if identity_fd is None:
            _fail()
        try:
            baseline = _stat_tuple(os.fstat(identity_fd))
            if _names(identity_fd, 4) != {"prepared.json", "copy-ready.json", "committed.json", _COPY}:
                _fail()
            previous: str | None = None
            fingerprints: list[object] = [baseline]
            records: list[_QuarantineRecord] = []
            for phase in _PHASES:
                leaf = f"{phase}.json"
                value = _read(identity_fd, leaf, _MAX_RECORD)
                if value is None:
                    _fail()
                record = _parse(value[0], phase, previous)
                records.append(record)
                previous = _digest(value[0])
                fingerprints.append((previous, _stat_tuple(value[1])))
                if sync:
                    _sync_checked_file(identity_fd, leaf, _MAX_RECORD)
            copy = _read(identity_fd, _COPY, _MAX_RECEIPT)
            if (copy is None or records[2]["version"] != 2
                    or records[0]["transaction_identity"] != name
                    or any(record[key] != records[0][key] for record in records[1:] for key in (
                        "workspace", "workspace_stat", "workspace_id", "sessions_root",
                        "sessions_root_stat", "transaction_identity", "complete_digest",
                        "receipt_digest", "receipt_size", "receipt_stat",
                    ))
                    or records[1]["copy_stat"] != records[2]["copy_stat"]
                    or list(_stat_tuple(copy[1])) != records[1]["copy_stat"]
                    or list(_stat_tuple(copy[1])) == records[0]["receipt_stat"]
                    or len(copy[0]) != records[0]["receipt_size"]
                    or _digest(copy[0]) != records[0]["receipt_digest"]):
                _fail()
            fingerprints.append((_digest(copy[0]), _stat_tuple(copy[1])))
            if sync:
                _sync_checked_file(identity_fd, _COPY, _MAX_RECEIPT)
                os.fsync(identity_fd)
            rebound = _entry(index_fd, name)
            if (rebound is None or _stat_tuple(rebound) != baseline
                    or _stat_tuple(os.fstat(identity_fd)) != baseline):
                _fail()
            result.append((name, _seal_digest(records[2]), tuple(fingerprints)))
        finally:
            os.close(identity_fd)
    result.sort(key=lambda item: item[0])
    if tuple(item[0] for item in result) != inspection.sealed_identities:
        _fail()
    return tuple(result)


def _open_bound_workspace(workspace: Path, identity: tuple[int, int]) -> int:
    """Own a no-follow directory pin matching an independently checked lock."""
    fd = _open_absolute(workspace)
    if fd is None:
        _fail()
    try:
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) != identity:
            _fail()
        _private_dir(info, memory=True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def establish_durable_quarantine_seals(
    gate: object,
) -> DurableQuarantineEvidence:
    """Under an existing lock, sync all sealed evidence without creating or repairing it."""
    if (not isinstance(gate, storage.WorkspaceMemoryLockGate)
            or not storage.WorkspaceMemoryLockGate.uses_existing_layout_only(gate)):
        _fail()
    workspace_fd: int | None = None
    memory_fd = recovery_fd = index_fd = None
    try:
        gate.require_current()
        workspace_fd = _open_bound_workspace(gate.workspace, gate.workspace_identity)
        gate.require_current()
        _private_dir(os.fstat(workspace_fd), memory=True)
        memory_fd, recovery_fd, index_fd = _walk_scope(workspace_fd)
        observed = tuple(os.fstat(fd) if fd is not None else None for fd in
                         (workspace_fd, memory_fd, recovery_fd, index_fd))
        if memory_fd is not None and _entry(memory_fd, _PENDING) is not None:
            _fail()
        before = _sealed_fingerprints(gate, index_fd, sync=False)
        if before != _sealed_fingerprints(gate, index_fd, sync=True):
            _fail()
        for fd in (index_fd, recovery_fd, memory_fd, workspace_fd):
            if fd is not None:
                os.fsync(fd)
                gate.require_current()
                _assert_inspection_bindings(gate.workspace, workspace_fd, memory_fd,
                                            recovery_fd, index_fd, observed, None)
        if before != _sealed_fingerprints(gate, index_fd, sync=False):
            _fail()
        gate.require_current()
        _assert_inspection_bindings(gate.workspace, workspace_fd, memory_fd,
                                    recovery_fd, index_fd, observed, None)
        return DurableQuarantineEvidence(tuple(item[0] for item in before),
                                         tuple(item[1] for item in before))
    except (OSError, TypeError, ValueError, storage.MemoryStorageError):
        _fail()
    finally:
        for fd in (index_fd, recovery_fd, memory_fd, workspace_fd):
            if fd is not None:
                os.close(fd)


def _unsupported_absence(workspace: Path) -> QuarantineInspection:
    """No writable or descriptor capability: only an absent evidence path is clean."""
    try:
        for path in (workspace, workspace / "memory", workspace / "memory" / ".recovery",
                     workspace / "memory" / ".recovery" / _INDEX):
            current = Path(path.anchor)
            info: os.stat_result | None = None
            for component in path.parts[1:]:
                current /= component
                try:
                    info = os.lstat(current)
                except FileNotFoundError:
                    return QuarantineInspection(True, False, ())
                if not stat.S_ISDIR(info.st_mode):
                    _fail()
            if info is None:
                _fail()
            if os.name != "posix":
                continue
            if path == workspace:
                _private_dir(info, memory=True)
            elif path.name == "memory":
                _private_dir(info, memory=True)
            else:
                _private_dir(info)
            if path.name == "memory" and os.path.lexists(path / _PENDING):
                _fail()
        _fail()  # A present index cannot be inspected without descriptor capabilities.
    except (OSError, TypeError, ValueError):
        _fail()


class DurableAbsenceProof:
    """Opaque same-store proof, issued after a bound parent-directory fsync."""

    __slots__ = ("_store", "_memory_stat", "_used")

    def __init__(self, store: ReceiptQuarantineStore, memory_stat: tuple[int, ...], key: object):
        if key is not _PROOF_KEY:
            _fail()
        self._store = store
        self._memory_stat = memory_stat
        self._used = False

    def is_unused_for(self, store: ReceiptQuarantineStore) -> bool:
        """Compare token ownership/consumption only; do not perform live I/O."""
        return self._store is store and not self._used

    def matches_memory_snapshot(self, observe: Callable[[], tuple[int, ...]]) -> bool:
        """Read the issued identity before the caller's live parent observation."""
        return not (self._memory_stat != observe())

    def consume(self) -> None:
        """Invalidate this token; this never authorizes or publishes a marker."""
        self._used = True


class ReceiptQuarantineStore:
    """One locked preparation/copy/commit operation; never unlinks a receipt."""

    def __init__(self, gate: object, binding: object) -> None:
        if not isinstance(gate, storage.WorkspaceMemoryLockGate) or not isinstance(binding, QuarantineBinding):
            _fail()
        self._gate: storage.WorkspaceMemoryLockGate | None = gate
        self._authority: storage.WorkspaceMemoryAuthority | None = None
        self._operation: storage.WorkspaceMemoryOperation | None = None
        self._binding = binding
        self._owned_workspace_fd: int | None = None
        self._memory_fd: int | None = None
        self._recovery_fd: int | None = None
        self._index_fd: int | None = None
        self._identity_fd: int | None = None
        self._workspace_stat: tuple[int, int, int, int, int] | None = None
        self._sessions_stat: tuple[int, int, int, int, int] | None = None
        self._existing_only = False

    @classmethod
    def from_operation(
        cls, authority: object,
        operation: object, binding: object,
    ) -> Self:
        """Borrow the already-active operation, never acquiring another lock."""
        if (not isinstance(authority, storage.WorkspaceMemoryAuthority)
                or not isinstance(operation, storage.WorkspaceMemoryOperation)
                or not storage.WorkspaceMemoryOperation.is_owned_by(operation, authority)
                or not isinstance(binding, QuarantineBinding)):
            _fail()
        try:
            operation.require_current()
        except storage.MemoryStorageError:
            _fail()
        instance = cls.__new__(cls)
        instance._gate = None
        instance._authority = authority
        instance._operation = operation
        instance._binding = binding
        instance._owned_workspace_fd = None
        instance._memory_fd = None
        instance._recovery_fd = None
        instance._index_fd = None
        instance._identity_fd = None
        instance._workspace_stat = None
        instance._sessions_stat = None
        instance._existing_only = False
        return instance

    @classmethod
    def from_existing_operation(
        cls, authority: storage.WorkspaceMemoryAuthority,
        operation: storage.WorkspaceMemoryOperation, binding: QuarantineBinding,
    ) -> Self:
        """Open an intact copy-ready target without creating or repairing directories."""
        if not storage.WorkspaceMemoryAuthority.uses_existing_layout_only(authority):
            _fail()
        instance = cls.from_operation(authority, operation, binding)
        instance._existing_only = True
        return instance

    def _workspace_identity(self) -> tuple[int, int]:
        if self._gate is not None:
            return self._gate.workspace_identity
        if self._operation is not None:
            return self._operation.workspace_identity
        _fail()

    def _pin_workspace(self) -> None:
        self._require_lock()
        if self._owned_workspace_fd is not None:
            _fail()
        self._owned_workspace_fd = _open_bound_workspace(
            self._workspace_path(), self._workspace_identity(),
        )

    def _workspace_fd(self) -> int:
        self._require_lock()
        fd = self._owned_workspace_fd
        if fd is None:
            _fail()
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) != self._workspace_identity():
            _fail()
        return fd

    def _workspace_path(self) -> Path:
        if self._gate is not None:
            return self._gate.workspace
        if self._authority is not None:
            return self._authority.workspace
        _fail()

    def _require_lock(self) -> None:
        try:
            if self._gate is not None:
                self._gate.require_current()
            elif self._operation is not None and self._authority is not None:
                if not storage.WorkspaceMemoryOperation.matches_authority_observation(
                    self._operation, lambda: self._authority,
                ):
                    _fail()
                self._operation.require_current()
            else:
                _fail()
        except storage.MemoryStorageError:
            _fail()

    def __enter__(self) -> Self:
        try:
            self._require_lock()
            if _absolute(self._binding.workspace) != self._workspace_path():
                _fail()
            self._pin_workspace()
            workspace_fd = self._workspace_fd()
            _private_dir(os.fstat(workspace_fd), memory=True)
            self._workspace_stat = _directory_tuple(os.fstat(workspace_fd))
            sessions_fd = _open_absolute(_absolute(self._binding.sessions_root))
            if sessions_fd is None:
                _fail()
            try:
                _private_dir(os.fstat(sessions_fd), memory=True)
                self._sessions_stat = _directory_tuple(os.fstat(sessions_fd))
            finally:
                os.close(sessions_fd)
            self._memory_fd = _child(workspace_fd, "memory", memory=True)
            if self._memory_fd is None:
                _fail()
            self._recovery_fd = storage.open_recovery_private_directory(
                self._memory_fd, ".recovery", create=not self._existing_only,
                sync_existing=not self._existing_only,
            )
            self._index_fd = storage.open_recovery_private_directory(
                self._recovery_fd, _INDEX, create=not self._existing_only,
                sync_existing=not self._existing_only,
            )
            self._identity_fd = storage.open_recovery_private_directory(
                self._index_fd, self._binding.transaction_identity,
                create=not self._existing_only, sync_existing=not self._existing_only,
            )
            self._current()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        for attr in ("_identity_fd", "_index_fd", "_recovery_fd", "_memory_fd", "_owned_workspace_fd"):
            fd = getattr(self, attr)
            setattr(self, attr, None)
            if fd is not None:
                os.close(fd)

    def _current(self) -> None:
        try:
            workspace_fd = self._workspace_fd()
            if (self._workspace_stat != _directory_tuple(os.fstat(workspace_fd))
                    or _absolute(self._binding.workspace) != self._workspace_path()):
                _fail()
            _private_dir(os.fstat(workspace_fd), memory=True)
            parent = workspace_fd
            for name, fd, memory in (
                ("memory", self._memory_fd, True), (".recovery", self._recovery_fd, False),
                (_INDEX, self._index_fd, False),
                (self._binding.transaction_identity, self._identity_fd, False),
            ):
                if fd is None:
                    _fail()
                before = _entry(parent, name)
                if before is None or _stat_tuple(before) != _stat_tuple(os.fstat(fd)):
                    _fail()
                _private_dir(before, memory=memory)
                parent = fd
            root_fd = _open_absolute(_absolute(self._binding.sessions_root))
            if root_fd is None:
                _fail()
            try:
                if self._sessions_stat != _directory_tuple(os.fstat(root_fd)):
                    _fail()
            finally:
                os.close(root_fd)
        except (OSError, TypeError, ValueError, storage.MemoryStorageError):
            _fail()

    def _record(self, phase: str, predecessor: str | None, receipt_stat: list[int],
                copy_stat: list[int] | None, *, seal_digest: str | None = None) -> bytes:
        binding = self._binding
        record: dict[str, object] = {
            "version": 2 if seal_digest is not None else 1,
            "phase": phase, "predecessor_digest": predecessor,
            "workspace": str(self._workspace_path()), "workspace_stat": list(self._workspace_stat or ()),
            "workspace_id": binding.workspace_id,
            "sessions_root": str(_absolute(binding.sessions_root)),
            "sessions_root_stat": list(self._sessions_stat or ()),
            "transaction_identity": binding.transaction_identity,
            "complete_digest": binding.complete_digest, "receipt_digest": binding.receipt_digest,
            "receipt_size": binding.receipt_size, "receipt_stat": receipt_stat,
            "copy_name": _COPY if copy_stat is not None else None, "copy_stat": copy_stat,
        }
        if seal_digest is not None:
            record["semantic_seal_digest"] = seal_digest
        return _canonical(record)

    def _publish(self, name: str, content: bytes) -> None:
        self._current()
        fd = self._identity_fd
        if fd is None or _entry(fd, name) is not None or len(content) > _MAX_RECORD and name != _COPY:
            _fail()
        leaf: int | None = None
        try:
            leaf = storage.create_recovery_exclusive_leaf(os.open, name, fd)
            if not content and name == _COPY:
                pass
            offset = 0
            while offset < len(content):
                wrote = os.write(leaf, content[offset:])
                if wrote <= 0:
                    _fail()
                offset += wrote
            _private_file(os.fstat(leaf))
            os.fsync(leaf)
            self._current()
            os.fsync(fd)
            self._current()
            result = _read(fd, name, _MAX_RECEIPT if name == _COPY else _MAX_RECORD)
            if result is None or result[0] != content or _stat_tuple(result[1]) != _stat_tuple(os.fstat(leaf)):
                _fail()
        except (OSError, TypeError, ValueError):
            _fail()
        finally:
            if leaf is not None:
                os.close(leaf)

    def _prefix(self, count: int) -> tuple[list[_QuarantineRecord], list[str]]:
        self._current()
        fd = self._identity_fd
        if fd is None:
            _fail()
        names = _names(fd, 4)
        if not names <= {"prepared.json", "copy-ready.json", "committed.json", _COPY}:
            _fail()
        records: list[_QuarantineRecord] = []
        digests: list[str] = []
        predecessor = None
        for phase in _PHASES[:count]:
            result = _read(fd, f"{phase}.json", _MAX_RECORD)
            if result is None:
                _fail()
            raw, _ = result
            record = _parse(raw, phase, predecessor)
            if (record["workspace"] != str(self._workspace_path())
                    or record["workspace_stat"] != list(self._workspace_stat or ())
                    or record["sessions_root_stat"] != list(self._sessions_stat or ())
                    or record["sessions_root"] != str(_absolute(self._binding.sessions_root))
                    or any(record[key] != getattr(self._binding, key) for key in (
                        "workspace_id", "transaction_identity", "complete_digest",
                        "receipt_digest", "receipt_size",
                    ))):
                _fail()
            predecessor = _digest(raw)
            records.append(record)
            digests.append(predecessor)
        return records, digests

    def prepare(self, receipt: bytes) -> None:
        """Exclusively publish receipt identity before any copy is attempted."""
        if type(receipt) is not bytes or len(receipt) != self._binding.receipt_size or _digest(receipt) != self._binding.receipt_digest:
            _fail()
        self._current()
        if self._memory_fd is None or self._identity_fd is None or _names(self._identity_fd, 1):
            _fail()
        pending = _read(self._memory_fd, _PENDING, _MAX_RECEIPT)
        if pending is None or pending[0] != receipt:
            _fail()
        self._publish("prepared.json", self._record("prepared", None, list(_stat_tuple(pending[1])), None))

    def copy_receipt(self) -> None:
        """Copy raw bytes and verify file/parent durability before copy-ready."""
        records, digests = self._prefix(1)
        if self._memory_fd is None or self._identity_fd is None:
            _fail()
        if _names(self._identity_fd, 4) != {"prepared.json"}:
            _fail()
        pending = _read(self._memory_fd, _PENDING, _MAX_RECEIPT)
        if (pending is None or list(_stat_tuple(pending[1])) != records[0]["receipt_stat"]
                or len(pending[0]) != self._binding.receipt_size
                or _digest(pending[0]) != self._binding.receipt_digest):
            _fail()
        self._publish(_COPY, pending[0])
        copied = _read(self._identity_fd, _COPY, _MAX_RECEIPT)
        if copied is None or copied[0] != pending[0]:
            _fail()
        self._publish("copy-ready.json", self._record(
            "copy-ready", digests[0], list(_stat_tuple(pending[1])), list(_stat_tuple(copied[1]))
        ))

    def prove_durable_absence(self) -> DurableAbsenceProof:
        """Issue an opaque proof only after fsync of the bound receipt parent."""
        self._prefix(2)
        self._current()
        fd = self._memory_fd
        if (fd is None or self._identity_fd is None
                or _names(self._identity_fd, 4) != {"prepared.json", "copy-ready.json", _COPY}
                or _entry(fd, _PENDING) is not None):
            _fail()
        try:
            os.fsync(fd)
        except (OSError, TypeError, ValueError):
            _fail()
        self._current()
        if _entry(fd, _PENDING) is not None:
            _fail()
        return DurableAbsenceProof(self, _stat_tuple(os.fstat(fd)), _PROOF_KEY)

    def commit(self, proof: DurableAbsenceProof, terminal_proof: TerminalCompleteProof) -> None:
        """Publish a sealed marker after fresh operation-bound semantic verification."""
        from nanobot.agent.memory_recovery_transaction_terminal import TerminalCompleteProof

        def observe_memory_stat() -> tuple[int, ...]:
            fd = self._memory_fd
            if fd is None:
                _fail()
            return _stat_tuple(os.fstat(fd))

        try:
            if (type(proof) is not DurableAbsenceProof
                    or not DurableAbsenceProof.is_unused_for(proof, self)
                    or self._memory_fd is None
                    or not DurableAbsenceProof.matches_memory_snapshot(proof, observe_memory_stat)
                    or self._authority is None or self._operation is None
                    or type(terminal_proof) is not TerminalCompleteProof
                    or not TerminalCompleteProof.is_sealed_for(
                        terminal_proof, self._authority, self._operation,
                    )):
                _fail()
        except (AttributeError, OSError, TypeError, ValueError):
            _fail()
        records, digests = self._prefix(2)
        if (self._identity_fd is None
                or _names(self._identity_fd, 4) != {"prepared.json", "copy-ready.json", _COPY}
                or _entry(self._memory_fd, _PENDING) is not None):
            _fail()
        copied = _read(self._identity_fd, _COPY, _MAX_RECEIPT)
        if (copied is None or _digest(copied[0]) != self._binding.receipt_digest
                or list(_stat_tuple(copied[1])) != records[1]["copy_stat"]):
            _fail()
        TerminalCompleteProof.require_quarantine_commit_match(
            terminal_proof, self._authority, self._operation, self._binding,
            copy_bytes=copied[0], observe_copy_stat=lambda: _stat_tuple(copied[1]),
            original_stat=records[0]["receipt_stat"], digest_claims=lambda raw: _digest(raw),
        )
        try:
            if _entry(self._memory_fd, _PENDING) is not None:
                _fail()
        except (AttributeError, OSError, TypeError, ValueError):
            _fail()
        self._current()
        try:
            terminal_proof.require_current()
        except Exception:
            _fail()
        DurableAbsenceProof.consume(proof)
        self._publish("committed.json", self._record(
            "committed", digests[1], records[0]["receipt_stat"], records[1]["copy_stat"],
            seal_digest=terminal_proof.semantic_seal_digest,
        ))


class ValidatedQuarantineCopy(ReceiptQuarantineStore):
    """Read-only, operation-scoped lease on one exact quarantine copy."""

    def __init__(
        self, authority: storage.WorkspaceMemoryAuthority,
        operation: storage.WorkspaceMemoryOperation, binding: QuarantineBinding,
        *, require_phase: Literal["copy-ready", "committed"], require_pending_absent: bool,
    ) -> None:
        if (type(require_phase) is not str or require_phase not in ("copy-ready", "committed")
                or type(require_pending_absent) is not bool):
            _fail()
        borrowed = ReceiptQuarantineStore.from_operation(authority, operation, binding)
        self.__dict__.update(borrowed.__dict__)
        self._phase = require_phase
        self._require_pending_absent = require_pending_absent
        self._raw: bytes | None = None
        self._original: tuple[int, ...] | None = None
        self._copied: tuple[int, ...] | None = None
        self._record_bytes: tuple[bytes, ...] | None = None
        self._record_stats: tuple[tuple[int, ...], ...] | None = None
        self._pending_baseline: tuple[int, ...] | None = None
        self._seal_digest: str | None = None
        self._entered = False
        self._used = False

    @classmethod
    def from_operation(
        cls, authority: object,
        operation: object, binding: object,
    ) -> Self:
        # This lease requires an explicit read phase; the parent factory is
        # deliberately unavailable so it cannot create an uninitialized lease.
        _fail()

    def __enter__(self) -> Self:
        if self._used:
            _fail()
        self._used = True
        try:
            self._require_lock()
            if _absolute(self._binding.workspace) != self._workspace_path():
                _fail()
            self._pin_workspace()
            workspace_fd = self._workspace_fd()
            _private_dir(os.fstat(workspace_fd), memory=True)
            self._workspace_stat = _directory_tuple(os.fstat(workspace_fd))
            sessions_fd = _open_absolute(_absolute(self._binding.sessions_root))
            if sessions_fd is None:
                _fail()
            try:
                _private_dir(os.fstat(sessions_fd), memory=True)
                self._sessions_stat = _directory_tuple(os.fstat(sessions_fd))
            finally:
                os.close(sessions_fd)
            self._memory_fd = _child(workspace_fd, "memory", memory=True)
            if self._memory_fd is None:
                _fail()
            self._recovery_fd = _child(self._memory_fd, ".recovery")
            if self._recovery_fd is None:
                _fail()
            self._index_fd = _child(self._recovery_fd, _INDEX)
            if self._index_fd is None:
                _fail()
            self._identity_fd = _child(self._index_fd, self._binding.transaction_identity)
            if self._identity_fd is None:
                _fail()
            self._entered = True
            self._validate(initial=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self._entered = False
        super().__exit__(_type, _value, _traceback)

    def _validate(self, *, initial: bool = False) -> None:
        if not self._entered or self._identity_fd is None or self._memory_fd is None:
            _fail()
        count = 2 if self._phase == "copy-ready" else 3
        records, _ = self._prefix(count)
        names = {"prepared.json", "copy-ready.json", _COPY}
        if count == 3:
            names.add("committed.json")
        if _names(self._identity_fd, 4) != names:
            _fail()
        first = records[0]
        for record in records[1:]:
            if any(record[key] != first[key] for key in (
                "workspace", "workspace_stat", "workspace_id", "sessions_root",
                "sessions_root_stat", "transaction_identity", "complete_digest",
                "receipt_digest", "receipt_size", "receipt_stat",
            )) or record["copy_stat"] != records[1]["copy_stat"]:
                _fail()
        copied = _read(self._identity_fd, _COPY, _MAX_RECEIPT)
        if copied is None:
            _fail()
        raw, copy_info = copied
        if (len(raw) != self._binding.receipt_size or _digest(raw) != self._binding.receipt_digest
                or list(_stat_tuple(copy_info)) != records[1]["copy_stat"]
                or list(_stat_tuple(copy_info)) == first["receipt_stat"]):
            _fail()
        pending = _read(self._memory_fd, _PENDING, _MAX_RECEIPT)
        if self._require_pending_absent and pending is not None:
            _fail()
        if pending is not None and (list(_stat_tuple(pending[1])) != first["receipt_stat"]
                                    or pending[0] != raw):
            _fail()
        record_bytes: list[bytes] = []
        record_stats: list[tuple[int, ...]] = []
        predecessor: str | None = None
        for phase in _PHASES[:count]:
            record_copy = _read(self._identity_fd, f"{phase}.json", _MAX_RECORD)
            if record_copy is None or _parse(record_copy[0], phase, predecessor) != records[len(record_bytes)]:
                _fail()
            record_bytes.append(record_copy[0])
            record_stats.append(_stat_tuple(record_copy[1]))
            predecessor = _digest(record_copy[0])
        self._current()
        rebound_copy = _read(self._identity_fd, _COPY, _MAX_RECEIPT)
        rebound_pending = _read(self._memory_fd, _PENDING, _MAX_RECEIPT)
        if (rebound_copy is None or rebound_copy[0] != raw
                or _stat_tuple(rebound_copy[1]) != _stat_tuple(copy_info)
                or (rebound_pending is None) != (pending is None)
                or (pending is not None and rebound_pending is not None and
                    (rebound_pending[0] != pending[0]
                     or _stat_tuple(rebound_pending[1]) != _stat_tuple(pending[1])))):
            _fail()
        if _names(self._identity_fd, 4) != names:
            _fail()
        for phase, expected_raw, expected_stat in zip(
            _PHASES[:count], record_bytes, record_stats, strict=True,
        ):
            rebound_record = _read(self._identity_fd, f"{phase}.json", _MAX_RECORD)
            if (rebound_record is None or rebound_record[0] != expected_raw
                    or _stat_tuple(rebound_record[1]) != expected_stat):
                _fail()
        self._current()
        final_pending = _entry(self._memory_fd, _PENDING)
        if ((final_pending is None) != (pending is None)
                or (final_pending is not None and pending is not None
                    and _stat_tuple(final_pending) != _stat_tuple(pending[1]))):
            _fail()
        # Both content and inode/stat stay fixed throughout this lease.
        if initial:
            self._raw = raw
            self._original = tuple(first["receipt_stat"])
            self._copied = _stat_tuple(copy_info)
            self._record_bytes = tuple(record_bytes)
            self._record_stats = tuple(record_stats)
            self._pending_baseline = _stat_tuple(pending[1]) if pending is not None else None
            self._seal_digest = (_seal_digest(records[2])
                                 if count == 3 and records[2]["version"] == 2 else None)
        elif (self._raw != raw or self._original != tuple(first["receipt_stat"])
              or self._copied != _stat_tuple(copy_info)
              or self._record_bytes != tuple(record_bytes)
              or self._record_stats != tuple(record_stats)
              or self._pending_baseline != (_stat_tuple(pending[1]) if pending is not None else None)
              or self._seal_digest != (_seal_digest(records[2])
                                        if count == 3 and records[2]["version"] == 2 else None)):
            _fail()

    def prepare(self, receipt: bytes) -> None:
        _fail()

    def copy_receipt(self) -> None:
        _fail()

    def prove_durable_absence(self) -> DurableAbsenceProof:
        _fail()

    def commit(self, proof: DurableAbsenceProof, terminal_proof: TerminalCompleteProof) -> None:
        _fail()

    def __reduce__(self) -> NoReturn:
        raise TypeError("quarantine copy lease cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("quarantine copy lease cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("quarantine copy lease cannot be copied")

    def is_entered_for(
        self, authority: storage.WorkspaceMemoryAuthority,
        operation: storage.WorkspaceMemoryOperation, binding: QuarantineBinding,
    ) -> bool:
        """Compare the actual lease scope; public content reads still revalidate."""
        return (self._entered and self._authority is authority
                and self._operation is operation and not (self._binding != binding))

    def require_current(self) -> None:
        """Reopen exact records, copy, paths and pending state under the live operation."""
        self._validate()

    @property
    def raw_bytes(self) -> bytes:
        self.require_current()
        assert self._raw is not None
        return self._raw

    @property
    def original_stat(self) -> tuple[int, ...]:
        self.require_current()
        assert self._original is not None
        return self._original

    @property
    def copy_stat(self) -> tuple[int, ...]:
        self.require_current()
        assert self._copied is not None
        return self._copied

    @property
    def phase(self) -> Literal["copy-ready", "committed"]:
        self.require_current()
        return self._phase

    @property
    def semantic_seal_digest(self) -> str | None:
        """A structural stored digest, not independent semantic validation."""
        self.require_current()
        return self._seal_digest

    @property
    def binding(self) -> QuarantineBinding:
        self.require_current()
        return self._binding


def open_validated_quarantine_copy(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation, binding: QuarantineBinding,
    *, require_phase: Literal["copy-ready", "committed"], require_pending_absent: bool = False,
) -> ValidatedQuarantineCopy:
    """Open targeted structural evidence; this does not perform startup admission."""
    return ValidatedQuarantineCopy(
        authority, operation, binding, require_phase=require_phase,
        require_pending_absent=require_pending_absent,
    )


def read_existing_copy_ready_binding(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    sessions_root: Path | str,
    transaction_identity: str,
) -> QuarantineBinding:
    """Bootstrap one explicitly named incomplete copy; never semantic authorization."""
    if (type(transaction_identity) is not str or not _DIGEST.fullmatch(transaction_identity)
            or not authority._existing_only):  # pyright: ignore[reportPrivateUsage]
        _fail()
    operation.require_current()
    workspace_fd = authority._workspace_fd  # pyright: ignore[reportPrivateUsage]
    if workspace_fd is None:
        _fail()
    opened: list[int] = []
    try:
        parent = workspace_fd
        for name in ("memory", ".recovery", _INDEX, transaction_identity):
            child = _child(parent, name, memory=name == "memory")
            if child is None:
                _fail()
            opened.append(child)
            parent = child
        identity_fd = opened[-1]
        if _names(identity_fd, 4) != {"prepared.json", "copy-ready.json", _COPY}:
            _fail()
        prepared = _read(identity_fd, "prepared.json", _MAX_RECORD)
        ready = _read(identity_fd, "copy-ready.json", _MAX_RECORD)
        copied = _read(identity_fd, _COPY, _MAX_RECEIPT)
        if prepared is None or ready is None or copied is None:
            _fail()
        first = _parse(prepared[0], "prepared", None)
        second = _parse(ready[0], "copy-ready", _digest(prepared[0]))
        if (first["transaction_identity"] != transaction_identity
                or any(first[key] != second[key] for key in (
                    "workspace", "workspace_stat", "workspace_id", "sessions_root",
                    "sessions_root_stat", "transaction_identity", "complete_digest",
                    "receipt_digest", "receipt_size", "receipt_stat",
                )) or second["receipt_stat"] != first["receipt_stat"]
                or second["copy_stat"] != list(_stat_tuple(copied[1]))
                or len(copied[0]) != first["receipt_size"]
                or _digest(copied[0]) != first["receipt_digest"]
                or _entry(opened[0], _PENDING) is not None
                or first["workspace"] != str(authority.workspace)
                or first["workspace_stat"] != list(_directory_tuple(os.fstat(workspace_fd)))
                or first["sessions_root"] != str(_absolute(sessions_root))):
            _fail()
        sessions_fd = _open_absolute(_absolute(sessions_root))
        if sessions_fd is None:
            _fail()
        try:
            _private_dir(os.fstat(sessions_fd), memory=True)
            if first["sessions_root_stat"] != list(_directory_tuple(os.fstat(sessions_fd))):
                _fail()
        finally:
            os.close(sessions_fd)
        operation.require_current()
        receipt_size = first["receipt_size"]
        if type(receipt_size) is not int:
            _fail()
        return QuarantineBinding(
            authority.workspace, _absolute(sessions_root), str(first["workspace_id"]),
            transaction_identity, str(first["complete_digest"]),
            str(first["receipt_digest"]), receipt_size,
        )
    except (OSError, ValueError, TypeError, storage.MemoryStorageError):
        _fail()
    finally:
        for fd in reversed(opened):
            os.close(fd)


def normalize_recovery_root(value: object) -> Path:
    """Normalize the existing lexical root policy without resolving or admitting it."""
    return _absolute(value)


__all__ = ["DurableAbsenceProof", "DurableQuarantineEvidence", "QuarantineBinding", "QuarantineError",
           "QuarantineInspection", "ReceiptQuarantineStore", "ValidatedQuarantineCopy",
           "establish_durable_quarantine_seals", "inspect_receipt_quarantine",
           "open_validated_quarantine_copy", "read_existing_copy_ready_binding",
           "normalize_recovery_root"]
