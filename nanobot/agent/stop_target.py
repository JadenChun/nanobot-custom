"""Typed live Stop expectations for original-chat control requests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from nanobot.session.pending_followups import PendingFollowupRecord
from nanobot.session.source_store import SourceState

STOP_TARGET_VERSION = 1
MAX_STOP_TARGET_CHILDREN = 256
MAX_STOP_TARGET_PENDING_RECORDS = 1024
MAX_STOP_TARGET_SOURCE_EVENTS = 100_000
MAX_STOP_TARGET_COUNTER = (1 << 53) - 1

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FIELDS = frozenset({"store_identity", "session_key", "incarnation", "generation"})
_EXPECTATION_FIELDS = frozenset({
    "version",
    "source",
    "control_epoch",
    "active_root_present",
    "active_turn_id",
    "active_root_run_ids",
    "children",
    "pending_work_digest",
})
_CHILD_FIELDS = frozenset({
    "child_id", "source", "child_run_id", "run_sequence", "control_epoch",
})


class StopTargetUnavailableError(ValueError):
    """The complete current Stop target could not be proven safely."""


@dataclass(frozen=True, slots=True)
class StopSourceOrigin:
    store_identity: str
    session_key: str
    incarnation: str
    generation: int

    @classmethod
    def from_source_state(cls, state: SourceState) -> StopSourceOrigin:
        return cls(
            store_identity=state.store_identity,
            session_key=state.session_key,
            incarnation=state.incarnation,
            generation=state.generation,
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "store_identity": self.store_identity,
            "session_key": self.session_key,
            "incarnation": self.incarnation,
            "generation": self.generation,
        }


@dataclass(frozen=True, slots=True)
class StopChildExpectation:
    child_id: str
    source: StopSourceOrigin
    child_run_id: str | None
    run_sequence: int
    control_epoch: int

    def to_dict(self) -> dict[str, object]:
        return {
            "child_id": self.child_id,
            "source": self.source.to_dict(),
            "child_run_id": self.child_run_id,
            "run_sequence": self.run_sequence,
            "control_epoch": self.control_epoch,
        }


@dataclass(frozen=True, slots=True)
class StopExpectation:
    version: int
    source: StopSourceOrigin
    control_epoch: int
    active_root_present: bool
    active_turn_id: str | None
    active_root_run_ids: tuple[str, ...]
    children: tuple[StopChildExpectation, ...]
    pending_work_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source": self.source.to_dict(),
            "control_epoch": self.control_epoch,
            "active_root_present": self.active_root_present,
            "active_turn_id": self.active_turn_id,
            "active_root_run_ids": list(self.active_root_run_ids),
            "children": [child.to_dict() for child in self.children],
            "pending_work_digest": self.pending_work_digest,
        }


def _strict_object(value: object, fields: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    raw = cast(dict[object, object], value)
    duplicates = getattr(raw, "duplicate_keys", ())
    if duplicates:
        raise ValueError("duplicate object fields")
    if any(not isinstance(key, str) for key in raw):
        raise ValueError("object fields do not match the Stop contract")
    checked = cast(dict[str, object], raw)
    if set(checked) != set(fields):
        raise ValueError("object fields do not match the Stop contract")
    return checked


def _strict_text(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty) or len(value) > 512:
        raise ValueError("invalid string field")
    if any(ord(character) < 0x20 for character in value):
        raise ValueError("invalid string field")
    return value


def _counter(value: object) -> int:
    if type(value) is not int or value < 0 or value > MAX_STOP_TARGET_COUNTER:
        raise ValueError("invalid counter")
    return value


def _parse_source(value: object) -> StopSourceOrigin:
    raw = _strict_object(value, _SOURCE_FIELDS)
    store_identity = _strict_text(raw["store_identity"])
    session_key = _strict_text(raw["session_key"])
    incarnation = _strict_text(raw["incarnation"])
    if any(
        _SAFE_ID_RE.fullmatch(item) is None
        for item in (store_identity, session_key, incarnation)
    ):
        raise ValueError("invalid source identity")
    return StopSourceOrigin(
        store_identity=store_identity,
        session_key=session_key,
        incarnation=incarnation,
        generation=_counter(raw["generation"]),
    )


def parse_stop_expectation(value: object) -> StopExpectation:
    """Validate one public Stop expectation at the WebSocket ingress boundary."""
    raw = _strict_object(value, _EXPECTATION_FIELDS)
    version = raw["version"]
    if type(version) is not int or version != STOP_TARGET_VERSION:
        raise ValueError("unsupported Stop expectation version")
    active_root_present = raw["active_root_present"]
    if type(active_root_present) is not bool:
        raise ValueError("invalid active root presence")
    active_turn_value = raw["active_turn_id"]
    active_turn_id = None if active_turn_value is None else _strict_text(active_turn_value)

    run_values = raw["active_root_run_ids"]
    if not isinstance(run_values, list):
        raise ValueError("invalid active root run set")
    runs = cast(list[object], run_values)
    if len(runs) > 64:
        raise ValueError("invalid active root run set")
    run_ids = tuple(_strict_text(item) for item in runs)
    if any(_SAFE_ID_RE.fullmatch(run_id) is None for run_id in run_ids):
        raise ValueError("invalid active root run identity")
    if tuple(sorted(set(run_ids))) != run_ids:
        raise ValueError("active root run identities must be sorted and unique")

    child_values = raw["children"]
    if not isinstance(child_values, list):
        raise ValueError("invalid Stop child set")
    child_rows = cast(list[object], child_values)
    if len(child_rows) > MAX_STOP_TARGET_CHILDREN:
        raise ValueError("invalid Stop child set")
    children: list[StopChildExpectation] = []
    for value_item in child_rows:
        child = _strict_object(value_item, _CHILD_FIELDS)
        child_id = _strict_text(child["child_id"])
        if _SAFE_ID_RE.fullmatch(child_id) is None:
            raise ValueError("invalid child identity")
        run_value = child["child_run_id"]
        child_run_id = None if run_value is None else _strict_text(run_value)
        if child_run_id is not None and _SAFE_ID_RE.fullmatch(child_run_id) is None:
            raise ValueError("invalid child run identity")
        children.append(StopChildExpectation(
            child_id=child_id,
            source=_parse_source(child["source"]),
            child_run_id=child_run_id,
            run_sequence=_counter(child["run_sequence"]),
            control_epoch=_counter(child["control_epoch"]),
        ))
    child_ids = tuple(child.child_id for child in children)
    if tuple(sorted(set(child_ids))) != child_ids:
        raise ValueError("Stop children must be sorted and unique")

    if active_root_present:
        if not run_ids:
            raise ValueError("an active root requires canonical run identity")
    elif active_turn_id is not None or run_ids:
        raise ValueError("an absent root must have no turn or run identity")

    pending_digest = raw["pending_work_digest"]
    if not isinstance(pending_digest, str) or _SHA256_RE.fullmatch(pending_digest) is None:
        raise ValueError("invalid pending work digest")
    return StopExpectation(
        version=version,
        source=_parse_source(raw["source"]),
        control_epoch=_counter(raw["control_epoch"]),
        active_root_present=active_root_present,
        active_turn_id=active_turn_id,
        active_root_run_ids=run_ids,
        children=tuple(children),
        pending_work_digest=pending_digest,
    )


def pending_work_digest(
    sources: Sequence[tuple[StopSourceOrigin, Sequence[PendingFollowupRecord]]],
) -> str:
    """Hash durable pending IDs and typed child identity without message content."""
    records: list[dict[str, object]] = []
    total = 0
    for source, pending in sources:
        total += len(pending)
        if total > MAX_STOP_TARGET_PENDING_RECORDS:
            raise StopTargetUnavailableError("pending input inventory exceeds its safe budget")
        for item in pending:
            child_followup = item.child_followup
            background_result = item.background_result
            records.append({
                "source": source.to_dict(),
                "pending_id": item.identifier,
                "child_followup": (
                    child_followup.to_dict() if child_followup is not None else None
                ),
                "background_result": (
                    {
                        "outbox_id": background_result.outbox_id,
                        "child_id": background_result.child_id,
                        "child_key": background_result.child_key,
                        "child_run_id": background_result.child_run_id,
                        "child_state_sequence": background_result.child_state_sequence,
                        "child_control_epoch": background_result.child_control_epoch,
                    }
                    if background_result is not None else None
                ),
            })
    def source_sort_key(item: dict[str, object]) -> tuple[str, str, str, int, str]:
        source = cast(dict[str, object], item["source"])
        return (
            cast(str, source["store_identity"]),
            cast(str, source["session_key"]),
            cast(str, source["incarnation"]),
            cast(int, source["generation"]),
            cast(str, item["pending_id"]),
        )

    records.sort(key=source_sort_key)
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
