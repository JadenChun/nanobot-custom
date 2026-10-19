"""Immutable, pointerless manifests for memory recovery evidence.

The manifest deliberately stays below the recovery runtime.  It only validates
canonical JSON records and follows the successor slot named by the previous
record's digest through :mod:`memory_recovery_storage`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import NoReturn, TypeAlias, cast

from nanobot.agent import memory_recovery_storage as storage

JsonValue: TypeAlias = None | bool | int | str | dict[str, "JsonValue"] | list["JsonValue"]

MAX_CHAIN_RECORDS = 7
MAX_RECORD_BYTES = 65_536
MAX_DEPTH = 8
MAX_NODES = 1_024
MAX_KEY_BYTES = 128
MAX_STRING_BYTES = 4_096
MIN_INT = -(2**63 - 1)
MAX_INT = 2**63 - 1

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = frozenset(("transactions", "conversions"))
_PHASES = {
    "transactions": (
        "CAPTURED",
        "PREPARED",
        "HISTORY_APPLIED",
        "MEMORY_APPLIED",
        "OFFSET_APPLIED",
        "RECEIPT_COMMITTED",
        "COMPLETE",
    ),
    "conversions": (
        "CAPTURED",
        "CONVERSION_PREPARED",
        "JSONL_APPLIED",
        "CURSORS_APPLIED",
        "LEGACY_FINALIZED",
        "COMPLETE",
    ),
}
_ENVELOPE_FIELDS = frozenset(
    ("version", "kind", "identity", "sequence", "predecessor_digest", "phase", "payload")
)


class MemoryManifestError(RuntimeError):
    """Fixed, content-free errors raised by the manifest boundary."""

    _MESSAGES = {
        "invalid": "memory recovery manifest input is invalid",
        "corrupt": "memory recovery manifest evidence is invalid",
        "conflict": "memory recovery manifest evidence conflicts",
        "stale": "memory recovery manifest predecessor is stale",
        "terminal": "memory recovery manifest is complete",
    }

    def __init__(self, code: str) -> None:
        if code not in self._MESSAGES:
            code = "invalid"
        self.code = code
        super().__init__(self._MESSAGES[code])


def _fail(code: str) -> NoReturn:
    raise MemoryManifestError(code) from None


def _validate_kind(value: object, *, code: str = "invalid") -> str:
    if type(value) is not str or value not in _KINDS:
        _fail(code)
    return value


def _validate_digest(value: object, *, code: str = "invalid") -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _check_string(value: str, limit: int, *, code: str) -> None:
    try:
        length = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        _fail(code)
    if length > limit:
        _fail(code)


class _DuplicateKeyError(ValueError):
    pass


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _validate_payload(payload: object, *, code: str) -> bytes:
    """Validate a JSON payload and return its exact canonical bytes."""

    if type(payload) is not dict:
        _fail(code)

    active: set[int] = set()
    nodes = 0

    def visit(value: object, depth: int) -> None:
        nonlocal nodes
        if depth > MAX_DEPTH:
            _fail(code)
        nodes += 1
        if nodes > MAX_NODES:
            _fail(code)

        if type(value) is dict:
            marker = id(cast(dict[object, object], value))
            if marker in active:
                _fail(code)
            active.add(marker)
            try:
                for key, child in cast(dict[object, object], value).items():
                    if type(key) is not str:
                        _fail(code)
                    _check_string(key, MAX_KEY_BYTES, code=code)
                    visit(child, depth + 1)
            finally:
                active.remove(marker)
        elif type(value) is list:
            marker = id(cast(list[object], value))
            if marker in active:
                _fail(code)
            active.add(marker)
            try:
                for child in cast(list[object], value):
                    visit(child, depth + 1)
            finally:
                active.remove(marker)
        elif type(value) is str:
            _check_string(value, MAX_STRING_BYTES, code=code)
        elif value is None or type(value) is bool:
            return
        elif type(value) is int:
            if value < MIN_INT or value > MAX_INT:
                _fail(code)
        else:
            _fail(code)

    visit(cast(dict[object, object], payload), 1)
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError):
        _fail(code)
    raise AssertionError("unreachable")  # pragma: no cover


def _identity_from_payload(kind: str, payload_bytes: bytes) -> str:
    return hashlib.sha256(
        b"nanobot-memory-manifest-v1\0" + kind.encode("ascii") + b"\0" + payload_bytes
    ).hexdigest()


def derive_manifest_identity(kind: str, captured_payload: object) -> str:
    """Return the domain-separated identity for one captured payload."""

    kind = _validate_kind(kind)
    payload_bytes = _validate_payload(captured_payload, code="invalid")
    return _identity_from_payload(kind, payload_bytes)


def _canonical_record_bytes(
    *,
    kind: str,
    identity: str,
    sequence: int,
    predecessor_digest: str | None,
    phase: str,
    payload: object,
    code: str,
) -> bytes:
    try:
        encoded = json.dumps(
            {
                "version": 1,
                "kind": kind,
                "identity": identity,
                "sequence": sequence,
                "predecessor_digest": predecessor_digest,
                "phase": phase,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError):
        _fail(code)
    if len(encoded) > MAX_RECORD_BYTES:
        _fail(code)
    return encoded


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One validated manifest record with immutable evidence bytes."""

    version: int
    kind: str
    identity: str
    sequence: int
    predecessor_digest: str | None
    phase: str
    raw_bytes: bytes
    digest: str
    _payload_bytes: bytes = field(repr=False, compare=False)

    @property
    def payload(self) -> dict[str, JsonValue]:
        """Return a detached payload copy on every access."""

        try:
            value = json.loads(
                self._payload_bytes.decode("utf-8"),
                object_pairs_hook=_pairs_without_duplicates,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            _fail("corrupt")
        if type(value) is not dict:  # pragma: no cover - construction validates this.
            _fail("corrupt")
        return copy.deepcopy(cast(dict[str, JsonValue], value))


def _parse_record(
    raw_bytes: bytes,
    *,
    kind: str,
    identity: str,
    sequence: int,
    predecessor_digest: str | None,
    phase: str,
) -> ManifestRecord:
    if type(raw_bytes) is not bytes or len(raw_bytes) > MAX_RECORD_BYTES:
        _fail("corrupt")
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
        record = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, UnicodeError, ValueError, RecursionError):
        _fail("corrupt")
    if type(record) is not dict or frozenset(cast(dict[object, object], record)) != _ENVELOPE_FIELDS:
        _fail("corrupt")

    record = cast(dict[str, object], record)
    version = record.get("version")
    record_kind = record.get("kind")
    record_identity = record.get("identity")
    record_sequence = record.get("sequence")
    record_predecessor = record.get("predecessor_digest")
    record_phase = record.get("phase")
    payload = record.get("payload")

    if type(version) is not int or version != 1:
        _fail("corrupt")
    if type(record_kind) is not str:
        _fail("corrupt")
    _check_string(record_kind, MAX_STRING_BYTES, code="corrupt")
    if record_kind != kind:
        _fail("corrupt")
    if type(record_identity) is not str:
        _fail("corrupt")
    _check_string(record_identity, MAX_STRING_BYTES, code="corrupt")
    _validate_digest(record_identity, code="corrupt")
    if record_identity != identity:
        _fail("corrupt")
    if type(record_sequence) is not int or record_sequence != sequence:
        _fail("corrupt")
    if record_sequence < 0 or record_sequence >= MAX_CHAIN_RECORDS:
        _fail("corrupt")
    if record_predecessor is not None:
        if type(record_predecessor) is not str:
            _fail("corrupt")
        _check_string(record_predecessor, MAX_STRING_BYTES, code="corrupt")
        _validate_digest(record_predecessor, code="corrupt")
    if record_predecessor != predecessor_digest:
        _fail("corrupt")
    if type(record_phase) is not str:
        _fail("corrupt")
    _check_string(record_phase, MAX_STRING_BYTES, code="corrupt")
    if record_phase != phase:
        _fail("corrupt")
    payload_bytes = _validate_payload(payload, code="corrupt")
    if sequence == 0 and _identity_from_payload(kind, payload_bytes) != identity:
        _fail("corrupt")
    canonical = _canonical_record_bytes(
        kind=kind,
        identity=identity,
        sequence=sequence,
        predecessor_digest=predecessor_digest,
        phase=phase,
        payload=payload,
        code="corrupt",
    )
    if raw_bytes != canonical:
        _fail("corrupt")
    return ManifestRecord(
        version=1,
        kind=kind,
        identity=identity,
        sequence=sequence,
        predecessor_digest=predecessor_digest,
        phase=phase,
        raw_bytes=raw_bytes,
        digest=hashlib.sha256(raw_bytes).hexdigest(),
        _payload_bytes=payload_bytes,
    )


def _successor_slot(predecessor_digest: str) -> str:
    return f"next-{predecessor_digest}.json"


def _storage_code(error: BaseException, code: str) -> bool:
    return isinstance(error, storage.MemoryStorageError) and getattr(error, "code", None) == code


class MemoryRecoveryManifest:
    """A wrapper that follows one immutable evidence chain."""

    def __init__(self, namespace: storage.ImmutableEvidenceNamespace) -> None:
        if not isinstance(cast(object, namespace), storage.ImmutableEvidenceNamespace):
            _fail("invalid")
        try:
            kind = namespace.kind
            identity = namespace.identity
        except (AttributeError, TypeError, ValueError):
            _fail("invalid")
        self._namespace = namespace
        self._kind = _validate_kind(kind)
        self._identity = _validate_digest(identity)

    def _assert_namespace(self) -> None:
        try:
            kind = self._namespace.kind
            identity = self._namespace.identity
        except (AttributeError, TypeError, ValueError):
            _fail("invalid")
        if type(kind) is not str or kind != self._kind:
            _fail("invalid")
        if type(identity) is not str or identity != self._identity:
            _fail("invalid")

    def _read(self, slot: str) -> bytes | None:
        self._assert_namespace()
        try:
            raw_bytes = self._namespace.read_record(slot)
        except storage.MemoryStorageError as error:
            if _storage_code(error, "conflict"):
                raise MemoryManifestError("conflict") from None
            raise
        self._assert_namespace()
        return raw_bytes

    def _publish(self, slot: str, raw_bytes: bytes) -> None:
        self._assert_namespace()
        try:
            self._namespace.publish_record(slot, raw_bytes)
        except storage.MemoryStorageError as error:
            if _storage_code(error, "conflict"):
                raise MemoryManifestError("conflict") from None
            raise

    def _walk(self, *, allow_missing_slot: str | None = None) -> tuple[tuple[ManifestRecord, ...], str | None]:
        self._assert_namespace()
        phases = _PHASES[self._kind]
        records: list[ManifestRecord] = []
        seen_digests: set[str] = set()
        slot = "genesis.json"
        sequence = 0
        predecessor_digest: str | None = None

        while True:
            self._assert_namespace()
            try:
                raw_bytes = self._namespace.read_record(slot)
            except storage.MemoryStorageError as error:
                if _storage_code(error, "missing"):
                    if allow_missing_slot == slot and sequence < len(phases):
                        return tuple(records), slot
                    raise
                if _storage_code(error, "conflict"):
                    raise MemoryManifestError("conflict") from None
                raise
            self._assert_namespace()

            if raw_bytes is None:
                if allow_missing_slot == slot and sequence < len(phases):
                    return tuple(records), slot
                return tuple(records), None
            if sequence >= len(phases) or len(records) >= MAX_CHAIN_RECORDS:
                _fail("corrupt")

            record = _parse_record(
                raw_bytes,
                kind=self._kind,
                identity=self._identity,
                sequence=sequence,
                predecessor_digest=predecessor_digest,
                phase=phases[sequence],
            )
            if record.digest in seen_digests:
                _fail("corrupt")
            seen_digests.add(record.digest)
            records.append(record)

            if record.phase == "COMPLETE":
                # A terminal record has one deterministic probe.  A pending
                # companion here is storage evidence and must remain visible.
                terminal_successor = _successor_slot(record.digest)
                terminal_probe = self._read(terminal_successor)
                if terminal_probe is not None:
                    _fail("corrupt")
                return tuple(records), None

            sequence += 1
            predecessor_digest = record.digest
            slot = _successor_slot(record.digest)

    def load(self) -> tuple[ManifestRecord, ...]:
        """Load the reachable chain, returning an empty tuple for clean absence."""

        records, _ = self._walk()
        return records

    def create(self, captured_payload: dict[str, JsonValue]) -> ManifestRecord:
        """Publish or repair the canonical genesis record for this namespace."""

        self._assert_namespace()
        payload_bytes = _validate_payload(captured_payload, code="invalid")
        expected_identity = _identity_from_payload(self._kind, payload_bytes)
        if expected_identity != self._identity:
            _fail("invalid")
        raw_bytes = _canonical_record_bytes(
            kind=self._kind,
            identity=self._identity,
            sequence=0,
            predecessor_digest=None,
            phase="CAPTURED",
            payload=captured_payload,
            code="invalid",
        )

        try:
            records = self.load()
        except storage.MemoryStorageError as original:
            if not _storage_code(original, "missing"):
                raise
            # Re-run with only the candidate slot tolerated.  A missing
            # companion elsewhere remains a storage error.
            partial, missing_slot = self._walk(allow_missing_slot="genesis.json")
            if missing_slot != "genesis.json" or partial:
                raise original
            records = partial

        if records and records[0].raw_bytes != raw_bytes:
            raise MemoryManifestError("conflict")
        self._publish("genesis.json", raw_bytes)
        refreshed = self.load()
        if not refreshed:
            _fail("corrupt")
        return refreshed[0]

    def append(
        self,
        *,
        predecessor_digest: str,
        phase: str,
        payload: dict[str, JsonValue],
    ) -> ManifestRecord:
        """Publish or repair one expected successor record."""

        self._assert_namespace()
        predecessor_digest = _validate_digest(predecessor_digest)
        if type(phase) is not str:
            _fail("invalid")
        _check_string(phase, MAX_STRING_BYTES, code="invalid")
        if phase not in _PHASES[self._kind]:
            _fail("invalid")
        _validate_payload(payload, code="invalid")
        candidate_slot = _successor_slot(predecessor_digest)
        records, missing_slot = self._walk(allow_missing_slot=candidate_slot)

        predecessor_index = next(
            (index for index, record in enumerate(records) if record.digest == predecessor_digest),
            None,
        )
        if predecessor_index is None:
            raise MemoryManifestError("stale")
        predecessor = records[predecessor_index]
        if predecessor.phase == "COMPLETE":
            raise MemoryManifestError("terminal")
        next_sequence = predecessor.sequence + 1
        phases = _PHASES[self._kind]
        if next_sequence >= len(phases) or phase != phases[next_sequence]:
            _fail("invalid")
        raw_bytes = _canonical_record_bytes(
            kind=self._kind,
            identity=self._identity,
            sequence=next_sequence,
            predecessor_digest=predecessor_digest,
            phase=phase,
            payload=payload,
            code="invalid",
        )

        if predecessor_index + 1 < len(records):
            existing = records[predecessor_index + 1]
            if existing.raw_bytes != raw_bytes:
                raise MemoryManifestError("conflict")
        elif missing_slot != candidate_slot:
            # The only reachable absence for a nonterminal predecessor should
            # be this exact computed successor.  Anything else is stale/corrupt
            # evidence, depending on which chain validation exposed it.
            _fail("corrupt")

        self._publish(candidate_slot, raw_bytes)
        refreshed = self.load()
        for record in refreshed:
            if record.digest == hashlib.sha256(raw_bytes).hexdigest():
                return record
        _fail("corrupt")


__all__ = [
    "JsonValue",
    "MAX_CHAIN_RECORDS",
    "ManifestRecord",
    "MemoryManifestError",
    "MemoryRecoveryManifest",
    "derive_manifest_identity",
]
