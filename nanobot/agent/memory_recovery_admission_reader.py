"""Existing-only, descriptor-relative immutable evidence reads under a borrowed gate.

This is deliberately not a storage authority: it never publishes or repairs a
slot and retains descriptors for only one identity at a time.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, NoReturn

from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery_manifest import MAX_RECORD_BYTES, ManifestRecord, _parse_record

_MAX_BLOB = 64 * 1024 * 1024
_PHASES = {
    "transactions": ("CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED",
                     "OFFSET_APPLIED", "RECEIPT_COMMITTED", "COMPLETE"),
    "conversions": ("CAPTURED", "CONVERSION_PREPARED", "JSONL_APPLIED",
                    "CURSORS_APPLIED", "LEGACY_FINALIZED", "COMPLETE"),
}


def _fail() -> NoReturn:
    quarantine._fail()


class ExistingEvidenceReader:
    """Borrow a held existing-only gate; no second lock or creating constructor."""

    def __init__(self, gate: storage._WorkspaceMemoryLockGate) -> None:
        if (type(gate) is not storage._WorkspaceMemoryLockGate
                or not gate._existing_only):
            _fail()
        gate.require_current()
        self.gate = gate
        workspace_fd = gate._workspace_fd
        if workspace_fd is None:
            _fail()
        self.workspace_fd: int = workspace_fd
        self.workspace_stat = quarantine._stat_tuple(os.fstat(self.workspace_fd))

    def check(self) -> None:
        self.gate.require_current()
        reopened = quarantine._open_absolute(self.gate.workspace)
        if reopened is None:
            _fail()
        try:
            if (quarantine._stat_tuple(os.fstat(reopened)) != self.workspace_stat
                    or quarantine._stat_tuple(os.fstat(self.workspace_fd)) != self.workspace_stat):
                _fail()
        finally:
            os.close(reopened)

    @contextmanager
    def directory(self, parent: int, name: str, *, memory: bool = False) -> Generator[int]:
        self.check()
        child = quarantine._child(parent, name, memory=memory)
        if child is None:
            _fail()
        baseline = quarantine._stat_tuple(os.fstat(child))
        try:
            yield child
        finally:
            try:
                self.check()
                rebound = quarantine._entry(parent, name)
                if (rebound is None or quarantine._stat_tuple(rebound) != baseline
                        or quarantine._stat_tuple(os.fstat(child)) != baseline):
                    _fail()
            finally:
                os.close(child)

    def file(self, parent: int, name: str, limit: int, *, sync: bool = True,
             immutable: bool = False) -> bytes:
        self.check()
        result = self._immutable_file(parent, name, limit, sync=sync) if immutable else quarantine._read(parent, name, limit)
        if result is None:
            _fail()
        raw, before = result
        if sync and not immutable:
            quarantine._sync_checked_file(parent, name, limit)
        after = (self._immutable_file(parent, name, limit, sync=False) if immutable
                 else quarantine._read(parent, name, limit))
        if (after is None or after[0] != raw
                or quarantine._stat_tuple(after[1]) != quarantine._stat_tuple(before)):
            _fail()
        self.check()
        return raw

    def _immutable_file(self, parent: int, name: str, limit: int, *, sync: bool) -> tuple[bytes, os.stat_result]:
        """Check both no-clobber slots, including the optional .install- companion."""
        before = quarantine._entry(parent, name)
        if before is None:
            _fail()
        companion_name = ".install-" + name
        companion = quarantine._entry(parent, companion_name)
        if before.st_nlink not in (1, 2) or before.st_size > limit:
            _fail()
        for info in (before, companion):
            if info is not None and (not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_nlink not in (1, 2) or info.st_size > limit):
                _fail()
        if companion is None and before.st_nlink != 1:
            _fail()
        if companion is not None:
            same = (before.st_dev, before.st_ino) == (companion.st_dev, companion.st_ino)
            if (same and (before.st_nlink != 2 or companion.st_nlink != 2)) or (
                not same and (before.st_nlink != 1 or companion.st_nlink != 1)
            ):
                _fail()
            if same:
                linked: set[str] = set()
                with os.scandir(parent) as entries:
                    for item in entries:
                        info = quarantine._entry(parent, item.name)
                        if info is not None and (info.st_dev, info.st_ino) == (
                            before.st_dev, before.st_ino,
                        ):
                            linked.add(item.name)
                            if len(linked) > 2:
                                _fail()
                if linked != {name, companion_name}:
                    _fail()

        def read_leaf(leaf_name: str, info: os.stat_result) -> bytes:
            fd = os.open(leaf_name, storage._leaf_flags(), dir_fd=parent)
            try:
                if quarantine._stat_tuple(os.fstat(fd)) != quarantine._stat_tuple(info):
                    _fail()
                raw = bytearray()
                while len(raw) <= limit:
                    chunk = os.read(fd, min(65536, limit + 1 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                if len(raw) != info.st_size or len(raw) > limit:
                    _fail()
                if sync:
                    os.fsync(fd)
                rebound = quarantine._entry(parent, leaf_name)
                if (rebound is None or quarantine._stat_tuple(rebound) != quarantine._stat_tuple(info)
                        or quarantine._stat_tuple(os.fstat(fd)) != quarantine._stat_tuple(info)):
                    _fail()
                return bytes(raw)
            finally:
                os.close(fd)

        raw = read_leaf(name, before)
        if companion is not None and read_leaf(companion_name, companion) != raw:
            _fail()
        return raw, before

    def blob(self, recovery_fd: int, digest: str, size: int) -> bytes:
        if (type(digest) is not str or quarantine._DIGEST.fullmatch(digest) is None
                or type(size) is not int or size < 0 or size > _MAX_BLOB):
            _fail()
        with self.directory(recovery_fd, "blobs") as blobs:
            with self.directory(blobs, "sha256") as sha256:
                raw = self.file(sha256, digest, _MAX_BLOB, immutable=True)
                if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
                    _fail()
                return raw

    def chain(self, recovery_fd: int, kind: str, identity: str) -> tuple[ManifestRecord, ...]:
        if kind not in _PHASES or not quarantine._DIGEST.fullmatch(identity):
            _fail()
        with self.directory(recovery_fd, kind) as kind_fd:
            with self.directory(kind_fd, identity) as identity_fd:
                previous: str | None = None
                records: list[ManifestRecord] = []
                slots: list[str] = []
                for index, phase in enumerate(_PHASES[kind]):
                    slot = "genesis.json" if index == 0 else f"next-{previous}.json"
                    raw = self.file(identity_fd, slot, MAX_RECORD_BYTES, immutable=True)
                    record = _parse_record(raw, kind=kind, identity=identity,
                                           sequence=index, predecessor_digest=previous, phase=phase)
                    records.append(record)
                    slots.append(slot)
                    previous = record.digest
                if quarantine._names(identity_fd, 16) != set(slots) | {
                    ".install-" + slot for slot in slots
                    if quarantine._entry(identity_fd, ".install-" + slot) is not None
                }:
                    _fail()
                os.fsync(identity_fd)
                self.check()
                for slot, record in zip(slots, records, strict=True):
                    if hashlib.sha256(self.file(identity_fd, slot, MAX_RECORD_BYTES, immutable=True)).hexdigest() != record.digest:
                        _fail()
                return tuple(records)

    def empty_conversion(self, recovery_fd: int, identity: str) -> None:
        with self.directory(recovery_fd, "conversions") as kind_fd:
            with self.directory(kind_fd, identity) as identity_fd:
                if quarantine._names(identity_fd, 1):
                    _fail()

    def workspace_id(self, expected: str) -> None:
        with self.directory(self.workspace_fd, ".nanobot") as state:
            raw = self.file(state, "workspace-id", 33, sync=False)
        if raw not in (expected.encode("ascii"), expected.encode("ascii") + b"\n"):
            _fail()

    def sessions(self, root: Path, workspace_id: str) -> None:
        fd = quarantine._open_absolute(root)
        if fd is None:
            _fail()
        try:
            quarantine._private_dir(os.fstat(fd), memory=True)
            with self.directory(fd, workspace_id) as namespace:
                marker = self.file(namespace, ".workspace", 4096, sync=False)
                if marker != f"{self.gate.workspace}\n".encode("utf-8"):
                    _fail()
        finally:
            os.close(fd)
