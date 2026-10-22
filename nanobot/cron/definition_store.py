"""Typed definition adapter for the dormant cron action journal.

The action store owns durability, idempotency receipts, and conversion
provenance.  This module owns only the definition-shaped candidate that a
future cron runtime may consume.  It deliberately does not import the cron
service, schedule timers, or run jobs.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Mapping, TypedDict, cast

from nanobot.cron.action_store import (
    ActionReceipt,
    ActionRequest,
    ActionStoreGeneration,
    CronActionStore,
)
from nanobot.cron.types import CONTRACT_FIELDS, validate_contract_fields

_SCHEMA_VERSION = 1
_CANDIDATE_KIND = "cron_definition_candidate_v1"
_CANDIDATE_KIND_V2 = "cron_definition_candidate_v2"
MAX_SCHEDULING_COMMITS = 4096
_ACTION_KIND = "cron_definition_action_v1"
_LEGACY_CANDIDATE_KIND = "legacy_cron_snapshot_candidate"
_LEGACY_ACTION_KIND = "legacy_cron_action_v1"
_HEX_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_ACTION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,511}")

_DEFINITION_FIELDS = {"id", "name", "enabled", "schedule", "payload", "deleteAfterRun"}
_DEFINITION_WIRE_FIELDS = _DEFINITION_FIELDS | {"jobInstanceId", "definitionRevision"}
_SCHEDULE_FIELDS = {"kind", "atMs", "everyMs", "expr", "tz"}
_PAYLOAD_FIELDS = {
    "kind",
    "message",
    "deliver",
    "channel",
    "to",
    "additionalDestinations",
    "channelMeta",
    "sessionKey",
    "executionMode",
    "originChannel",
    "originChatId",
    "originMetadata",
}
_LEGACY_JOB_FIELDS = _DEFINITION_WIRE_FIELDS | {"state", "createdAtMs", "updatedAtMs"}


class GenerationWire(TypedDict):
    journal_generation: int
    snapshot_generation: int
    watermark: int
    sequence: int


class ReceiptWire(TypedDict):
    store_namespace: str
    action_id: str
    request_digest: str
    sequence: int
    journal_generation: int


class ScheduleCommit(TypedDict):
    store_namespace: str
    job_instance_id: str
    definition_revision: str
    schedule: dict[str, Any]
    schedule_digest: str
    schedule_format: int
    commit_clock_ms: int
    commit_kind: Literal["revision", "normalization"]
    mutation_receipt: ReceiptWire | None
    predecessor: GenerationWire
    target: GenerationWire


class DefinitionStoreError(RuntimeError):
    """Base error for the typed cron-definition adapter."""


class DefinitionStoreValidationError(DefinitionStoreError):
    """A definition candidate or typed action is malformed."""


class DefinitionStoreConflictError(DefinitionStoreError):
    """A typed action no longer matches the definition it targets."""


def _canonical_json(value: object, *, label: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError, RecursionError) as exc:
        raise DefinitionStoreValidationError(f"{label} must contain finite JSON data") from exc


def _copy_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DefinitionStoreValidationError(f"{label} must be a JSON object")
    encoded = _canonical_json(dict(cast(Mapping[object, object], value)), label=label)
    try:
        copied = json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as exc:  # pragma: no cover - json.dumps made this defensive
        raise DefinitionStoreValidationError(f"{label} must be JSON data") from exc
    if not isinstance(copied, dict):  # pragma: no cover - mappings always encode as objects
        raise DefinitionStoreValidationError(f"{label} must be a JSON object")
    # JSON decoding constructs string-keyed objects with opaque JSON values.
    return cast(dict[str, Any], copied)


def _require_exact_fields(data: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    unknown = set(data) - fields
    if unknown:
        raise DefinitionStoreValidationError(f"{label} has unknown fields")
    missing = fields - set(data)
    if missing:
        raise DefinitionStoreValidationError(f"{label} is missing required fields")


def _require_string(value: object, *, label: str, nullable: bool = False, empty: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or (not empty and not value):
        raise DefinitionStoreValidationError(f"{label} must be a non-empty string")
    return value


def _require_integer(value: object, *, label: str, nullable: bool = False, minimum: int = 0) -> int | None:
    if nullable and value is None:
        return None
    if type(value) is not int or value < minimum:
        raise DefinitionStoreValidationError(f"{label} must be an exact integer")
    return value


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_DIGEST_RE.fullmatch(value) is None:
        raise DefinitionStoreValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_action_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _ACTION_ID_RE.fullmatch(value) is None:
        raise DefinitionStoreValidationError(f"{label} is invalid")
    return value


def _validate_schedule(value: object) -> dict[str, Any]:
    schedule = _copy_mapping(value, label="cron definition schedule")
    _require_exact_fields(schedule, _SCHEDULE_FIELDS, label="cron definition schedule")
    kind = schedule["kind"]
    if kind not in {"at", "every", "cron"}:
        raise DefinitionStoreValidationError("cron definition schedule has an unknown kind")
    at_ms = _require_integer(schedule["atMs"], label="cron definition schedule atMs", nullable=True)
    every_ms = _require_integer(
        schedule["everyMs"], label="cron definition schedule everyMs", nullable=True, minimum=1
    )
    expr = _require_string(schedule["expr"], label="cron definition schedule expr", nullable=True)
    _require_string(schedule["tz"], label="cron definition schedule tz", nullable=True)
    if kind == "at" and at_ms is None:
        raise DefinitionStoreValidationError("at cron definition schedule is missing atMs")
    if kind == "every" and every_ms is None:
        raise DefinitionStoreValidationError("every cron definition schedule is missing everyMs")
    if kind == "cron" and expr is None:
        raise DefinitionStoreValidationError("cron definition schedule is missing expr")
    return schedule


def _validate_payload(value: object) -> dict[str, Any]:
    payload = _copy_mapping(value, label="cron definition payload")
    _require_exact_fields(
        {name: item for name, item in payload.items() if name not in CONTRACT_FIELDS},
        _PAYLOAD_FIELDS, label="cron definition payload",
    )
    try:
        validate_contract_fields(payload)
    except ValueError as exc:
        raise DefinitionStoreValidationError(str(exc)) from exc
    if payload["kind"] not in {"agent_turn", "system_event"}:
        raise DefinitionStoreValidationError("cron definition payload has an unknown kind")
    _require_string(payload["message"], label="cron definition payload message", empty=True)
    if type(payload["deliver"]) is not bool:
        raise DefinitionStoreValidationError("cron definition payload deliver must be a boolean")
    for name in ("channel", "to", "sessionKey", "originChannel", "originChatId"):
        _require_string(payload[name], label=f"cron definition payload {name}", nullable=True)
    destinations = payload["additionalDestinations"]
    if not isinstance(destinations, list):
        raise DefinitionStoreValidationError("cron definition payload destinations must be a list")
    for destination in cast(list[object], destinations):
        item = _copy_mapping(destination, label="cron definition destination")
        _require_exact_fields(item, {"channel", "to"}, label="cron definition destination")
        _require_string(item["channel"], label="cron definition destination channel")
        _require_string(item["to"], label="cron definition destination to")
    for name in ("channelMeta", "originMetadata"):
        _copy_mapping(payload[name], label=f"cron definition payload {name}")
    if payload["executionMode"] not in {"session_bound", "legacy_fresh"}:
        raise DefinitionStoreValidationError("cron definition payload executionMode is invalid")
    return payload


def _definition_revision(definition: Mapping[str, Any], job_instance_id: str) -> str:
    digest_input = _copy_mapping(definition, label="cron definition")
    digest_input["jobInstanceId"] = job_instance_id
    return hashlib.sha256(_canonical_json(digest_input, label="cron definition").encode("utf-8")).hexdigest()


def _canonical_definition(value: object) -> dict[str, Any]:
    definition = _copy_mapping(value, label="cron definition")
    _require_exact_fields(definition, _DEFINITION_FIELDS, label="cron definition")
    _require_string(definition["id"], label="cron definition id")
    _require_string(definition["name"], label="cron definition name")
    if type(definition["enabled"]) is not bool:
        raise DefinitionStoreValidationError("cron definition enabled must be a boolean")
    definition["schedule"] = _validate_schedule(definition["schedule"])
    definition["payload"] = _validate_payload(definition["payload"])
    if type(definition["deleteAfterRun"]) is not bool:
        raise DefinitionStoreValidationError("cron definition deleteAfterRun must be a boolean")
    return definition


@dataclass(frozen=True)
class CronDefinition:
    """A canonical cron definition without state, history, or timestamps."""

    definition: dict[str, Any]
    job_instance_id: str
    definition_revision: str

    @property
    def job_id(self) -> str:
        return self.definition["id"]

    @classmethod
    def from_definition(cls, definition: Mapping[str, Any], *, job_instance_id: str) -> CronDefinition:
        canonical = _canonical_definition(definition)
        instance = _require_digest(job_instance_id, label="cron definition job_instance_id")
        return cls(canonical, instance, _definition_revision(canonical, instance))

    @classmethod
    def _from_wire(cls, value: object, *, label: str = "cron definition") -> CronDefinition:
        wire = _copy_mapping(value, label=label)
        _require_exact_fields(wire, _DEFINITION_WIRE_FIELDS, label=label)
        instance = _require_digest(wire["jobInstanceId"], label=f"{label} jobInstanceId")
        supplied_revision = _require_digest(wire["definitionRevision"], label=f"{label} definitionRevision")
        definition = _canonical_definition({name: wire[name] for name in _DEFINITION_FIELDS})
        calculated_revision = _definition_revision(definition, instance)
        if supplied_revision != calculated_revision:
            raise DefinitionStoreValidationError(f"{label} definitionRevision does not match its definition")
        return cls(definition, instance, calculated_revision)

    @classmethod
    def decode_wire(cls, value: object, *, label: str = "cron definition") -> CronDefinition:
        """Decode a detached definition without granting store or execution authority."""
        return cls._from_wire(value, label=label)

    def _validated(self) -> CronDefinition:
        validated = CronDefinition.from_definition(self.definition, job_instance_id=self.job_instance_id)
        if self.definition_revision != validated.definition_revision:
            raise DefinitionStoreValidationError("cron definition definitionRevision does not match its definition")
        return validated

    def validated(self) -> CronDefinition:
        """Return the original detached validation result, without store authority."""
        return self._validated()

    def to_wire(self) -> dict[str, Any]:
        validated = self._validated()
        return {
            **_copy_mapping(validated.definition, label="cron definition"),
            "jobInstanceId": validated.job_instance_id,
            "definitionRevision": validated.definition_revision,
        }


@dataclass(frozen=True)
class CronDefinitionAction:
    """One strict, durable request to change a cron definition."""

    operation: str
    job_id: str
    definition: CronDefinition | None = None
    expected_job_instance_id: str | None = None
    expected_definition_revision: str | None = None

    @classmethod
    def create(cls, definition: CronDefinition) -> CronDefinitionAction:
        validated = _validate_definition_object(definition)
        return cls("create", validated.job_id, validated)

    @classmethod
    def update(
        cls,
        definition: CronDefinition,
        *,
        expected_job_instance_id: str,
        expected_definition_revision: str,
    ) -> CronDefinitionAction:
        validated = _validate_definition_object(definition)
        instance = _require_digest(expected_job_instance_id, label="expected_job_instance_id")
        revision = _require_digest(expected_definition_revision, label="expected_definition_revision")
        if validated.job_instance_id != instance:
            raise DefinitionStoreValidationError("updated cron definition must retain its job instance")
        return cls("update", validated.job_id, validated, instance, revision)

    @classmethod
    def delete(
        cls,
        job_id: str,
        *,
        expected_job_instance_id: str,
        expected_definition_revision: str,
    ) -> CronDefinitionAction:
        return cls(
            "delete",
            _require_string(job_id, label="cron definition action job_id") or "",
            None,
            _require_digest(expected_job_instance_id, label="expected_job_instance_id"),
            _require_digest(expected_definition_revision, label="expected_definition_revision"),
        )

    @classmethod
    def _from_wire(cls, value: object) -> CronDefinitionAction:
        request = _copy_mapping(value, label="cron definition action")
        base_fields = {"schema_version", "kind", "operation", "job_id"}
        if request.get("schema_version") != _SCHEMA_VERSION or type(request.get("schema_version")) is not int:
            raise DefinitionStoreValidationError("cron definition action has an invalid schema version")
        if request.get("kind") != _ACTION_KIND:
            raise DefinitionStoreValidationError("cron definition action has an invalid kind")
        operation = request.get("operation")
        if operation == "create":
            _require_exact_fields(request, base_fields | {"definition"}, label="cron definition create action")
            definition = CronDefinition.decode_wire(request["definition"], label="cron definition create action definition")
            if request["job_id"] != definition.job_id:
                raise DefinitionStoreValidationError("cron definition create action job_id does not match its definition")
            return cls.create(definition)
        if operation == "update":
            _require_exact_fields(
                request,
                base_fields | {"definition", "expected_job_instance_id", "expected_definition_revision"},
                label="cron definition update action",
            )
            definition = CronDefinition.decode_wire(request["definition"], label="cron definition update action definition")
            if request["job_id"] != definition.job_id:
                raise DefinitionStoreValidationError("cron definition update action job_id does not match its definition")
            return cls.update(
                definition,
                expected_job_instance_id=request["expected_job_instance_id"],
                expected_definition_revision=request["expected_definition_revision"],
            )
        if operation == "delete":
            _require_exact_fields(
                request,
                base_fields | {"expected_job_instance_id", "expected_definition_revision"},
                label="cron definition delete action",
            )
            return cls.delete(
                request["job_id"],
                expected_job_instance_id=request["expected_job_instance_id"],
                expected_definition_revision=request["expected_definition_revision"],
            )
        raise DefinitionStoreValidationError("cron definition action has an unknown operation")

    @classmethod
    def decode_wire(cls, value: object) -> CronDefinitionAction:
        """Decode a validated action without committing it to an action store."""
        return cls._from_wire(value)

    def to_wire(self) -> dict[str, Any]:
        if self.operation == "create":
            definition = _validate_definition_object(self.definition)
            if self.job_id != definition.job_id:
                raise DefinitionStoreValidationError("cron definition create action job_id does not match its definition")
            return {
                "schema_version": _SCHEMA_VERSION,
                "kind": _ACTION_KIND,
                "operation": "create",
                "job_id": definition.job_id,
                "definition": definition.to_wire(),
            }
        if self.operation == "update":
            definition = _validate_definition_object(self.definition)
            if self.job_id != definition.job_id:
                raise DefinitionStoreValidationError("cron definition update action job_id does not match its definition")
            instance = _require_digest(self.expected_job_instance_id, label="expected_job_instance_id")
            revision = _require_digest(self.expected_definition_revision, label="expected_definition_revision")
            if definition.job_instance_id != instance:
                raise DefinitionStoreValidationError("updated cron definition must retain its job instance")
            return {
                "schema_version": _SCHEMA_VERSION,
                "kind": _ACTION_KIND,
                "operation": "update",
                "job_id": definition.job_id,
                "definition": definition.to_wire(),
                "expected_job_instance_id": instance,
                "expected_definition_revision": revision,
            }
        if self.operation == "delete":
            return {
                "schema_version": _SCHEMA_VERSION,
                "kind": _ACTION_KIND,
                "operation": "delete",
                "job_id": _require_string(self.job_id, label="cron definition action job_id") or "",
                "expected_job_instance_id": _require_digest(
                    self.expected_job_instance_id, label="expected_job_instance_id"
                ),
                "expected_definition_revision": _require_digest(
                    self.expected_definition_revision, label="expected_definition_revision"
                ),
            }
        raise DefinitionStoreValidationError("cron definition action has an unknown operation")


def _validate_definition_object(value: object) -> CronDefinition:
    if not isinstance(value, CronDefinition):
        raise DefinitionStoreValidationError("cron definition action requires a CronDefinition")
    return value.validated()


def _generation_wire(value: object) -> dict[str, int]:
    wire = _copy_mapping(value, label="scheduling generation")
    _require_exact_fields(wire, {"journal_generation", "snapshot_generation", "watermark", "sequence"}, label="scheduling generation")
    for name, item in wire.items():
        _require_integer(item, label=name, minimum=1 if name == "journal_generation" else 0)
    if wire["watermark"] > wire["sequence"]:
        raise DefinitionStoreValidationError("invalid scheduling generation watermark")
    return wire


def _validate_scheduling_commit(value: object) -> ScheduleCommit:
    from nanobot.cron.schedule import SCHEDULE_FORMAT, schedule_digest

    item = _copy_mapping(value, label="scheduling commit")
    _require_exact_fields(item, {"store_namespace", "job_instance_id", "definition_revision", "schedule", "schedule_digest", "schedule_format", "commit_clock_ms", "commit_kind", "mutation_receipt", "predecessor", "target"}, label="scheduling commit")
    _require_string(item["store_namespace"], label="scheduling namespace")
    for name in ("job_instance_id", "definition_revision", "schedule_digest"):
        _require_digest(item[name], label=name)
    item["schedule"] = _validate_schedule(item["schedule"])
    if item["schedule_digest"] != schedule_digest(item["schedule"]):
        raise DefinitionStoreValidationError("scheduling digest mismatch")
    if type(item["schedule_format"]) is not int or item["schedule_format"] != SCHEDULE_FORMAT:
        raise DefinitionStoreValidationError("invalid schedule format")
    _require_integer(item["commit_clock_ms"], label="commit clock")
    if item["commit_kind"] not in {"revision", "normalization"}:
        raise DefinitionStoreValidationError("invalid commit kind")
    prior, target = _generation_wire(item["predecessor"]), _generation_wire(item["target"])
    if target != {"journal_generation": prior["journal_generation"] + 1, "snapshot_generation": prior["snapshot_generation"] + 1, "watermark": prior["sequence"], "sequence": prior["sequence"]}:
        raise DefinitionStoreValidationError("invalid scheduling checkpoint target")
    receipt = item["mutation_receipt"]
    if item["commit_kind"] == "normalization":
        if receipt is not None:
            raise DefinitionStoreValidationError("normalization cannot invent mutation provenance")
    else:
        receipt = _copy_mapping(receipt, label="scheduling receipt")
        _require_exact_fields(receipt, {"store_namespace", "action_id", "request_digest", "sequence", "journal_generation"}, label="scheduling receipt")
        _require_action_id(receipt["action_id"], label="scheduling action id")
        _require_digest(receipt["request_digest"], label="scheduling request digest")
        _require_integer(receipt["sequence"], label="scheduling receipt sequence", minimum=1)
        _require_integer(receipt["journal_generation"], label="scheduling receipt generation", minimum=1)
        if receipt["store_namespace"] != item["store_namespace"] or not prior["watermark"] < receipt["sequence"] <= prior["sequence"] or receipt["journal_generation"] != prior["journal_generation"]:
            raise DefinitionStoreValidationError("scheduling receipt does not belong to checkpoint suffix")
    # Every nested scalar/object and known field was validated above.
    return cast(ScheduleCommit, item)


def parse_persisted_scheduler_commit(value: object) -> ScheduleCommit:
    return _validate_scheduling_commit(value)


@dataclass(frozen=True)
class CronDefinitionCandidate:
    """The durable definition-only state derived from an action-store replay."""

    definitions: tuple[CronDefinition, ...] = ()
    applied_action_ids: tuple[str, ...] = ()
    legacy_conversion_evidence: dict[str, Any] | None = None
    scheduling_commits: tuple[ScheduleCommit, ...] = ()
    scheduling_version: int = 1

    @classmethod
    def _empty(cls) -> CronDefinitionCandidate:
        return cls()

    @classmethod
    def empty(cls) -> CronDefinitionCandidate:
        """Construct through the original bound empty-candidate factory."""
        return cls._empty()

    @classmethod
    def _from_wire(cls, value: object) -> CronDefinitionCandidate:
        candidate = _copy_mapping(value, label="cron definition candidate")
        fields = {"schema_version", "kind", "definitions", "applied_action_ids"}
        if "legacy_conversion_evidence" in candidate:
            fields.add("legacy_conversion_evidence")
        v2 = candidate.get("kind") == _CANDIDATE_KIND_V2
        if v2:
            fields.add("scheduling_commits")
        _require_exact_fields(candidate, fields, label="cron definition candidate")
        if type(candidate["schema_version"]) is not int or candidate["schema_version"] != (2 if v2 else 1):
            raise DefinitionStoreValidationError("cron definition candidate has an invalid schema version")
        if candidate["kind"] not in {_CANDIDATE_KIND, _CANDIDATE_KIND_V2}:
            raise DefinitionStoreValidationError("cron definition candidate has an invalid kind")
        if not isinstance(candidate["definitions"], list):
            raise DefinitionStoreValidationError("cron definition candidate definitions must be a list")
        definitions = tuple(CronDefinition.decode_wire(item, label="cron definition candidate definition") for item in cast(list[object], candidate["definitions"]))
        ids = tuple(definition.job_id for definition in definitions)
        if len(set(ids)) != len(ids):
            raise DefinitionStoreValidationError("cron definition candidate contains duplicate job ids")
        action_ids_value = candidate["applied_action_ids"]
        if not isinstance(action_ids_value, list):
            raise DefinitionStoreValidationError("cron definition candidate applied_action_ids must be a list")
        action_ids = tuple(_require_action_id(item, label="cron definition candidate action id") for item in cast(list[object], action_ids_value))
        if len(set(action_ids)) != len(action_ids):
            raise DefinitionStoreValidationError("cron definition candidate contains duplicate action ids")
        evidence = None
        if "legacy_conversion_evidence" in candidate:
            evidence = _copy_mapping(candidate["legacy_conversion_evidence"], label="legacy conversion evidence")
        commits = candidate.get("scheduling_commits", [])
        if not isinstance(commits, list):
            raise DefinitionStoreValidationError("scheduling commits must be a list")
        return cls._validated_parts(definitions, action_ids, evidence, tuple(cast(list[object], commits)), 2 if v2 else 1)

    @classmethod
    def decode_wire(cls, value: object) -> CronDefinitionCandidate:
        """Decode a validated candidate without acquiring or publishing store state."""
        return cls._from_wire(value)

    def _validated(self) -> CronDefinitionCandidate:
        return self._validated_parts(self.definitions, self.applied_action_ids,
                                     self.legacy_conversion_evidence, self.scheduling_commits,
                                     self.scheduling_version)

    def validated(self) -> CronDefinitionCandidate:
        """Validate the candidate without checkpointing or publishing state."""
        return self._validated()

    @staticmethod
    def _validated_parts(
        definitions: tuple[CronDefinition, ...],
        applied_action_ids: tuple[str, ...],
        legacy_conversion_evidence: dict[str, Any] | None,
        scheduling_commits: tuple[object, ...],
        scheduling_version: int,
    ) -> CronDefinitionCandidate:
        """Validate ordered fields before constructing concrete stored commit records."""
        definitions = tuple(_validate_definition_object(definition) for definition in definitions)
        if len({definition.job_id for definition in definitions}) != len(definitions):
            raise DefinitionStoreValidationError("cron definition candidate contains duplicate job ids")
        action_ids = tuple(_require_action_id(action_id, label="cron definition candidate action id") for action_id in applied_action_ids)
        if len(set(action_ids)) != len(action_ids):
            raise DefinitionStoreValidationError("cron definition candidate contains duplicate action ids")
        evidence = (
            None
            if legacy_conversion_evidence is None
            else _copy_mapping(legacy_conversion_evidence, label="legacy conversion evidence")
        )
        if scheduling_version not in (1, 2) or type(scheduling_version) is not int:
            raise DefinitionStoreValidationError("invalid scheduling version")
        commits = tuple(_validate_scheduling_commit(item) for item in scheduling_commits)
        keys = [(item["job_instance_id"], item["definition_revision"]) for item in commits]
        if len(keys) != len(set(keys)) or len(keys) > MAX_SCHEDULING_COMMITS:
            raise DefinitionStoreValidationError("duplicate or excessive scheduling commits")
        if commits and scheduling_version != 2:
            raise DefinitionStoreValidationError("v1 candidate cannot contain scheduling commits")
        return CronDefinitionCandidate(definitions, action_ids, evidence, commits, scheduling_version)

    def to_wire(self) -> dict[str, Any]:
        validated = self._validated()
        wire: dict[str, Any] = {
            "schema_version": validated.scheduling_version,
            "kind": _CANDIDATE_KIND_V2 if validated.scheduling_version == 2 else _CANDIDATE_KIND,
            "definitions": [definition.to_wire() for definition in validated.definitions],
            "applied_action_ids": list(validated.applied_action_ids),
        }
        if validated.legacy_conversion_evidence is not None:
            wire["legacy_conversion_evidence"] = validated.legacy_conversion_evidence
        if validated.scheduling_version == 2:
            wire["scheduling_commits"] = list(validated.scheduling_commits)
        return wire


@dataclass(frozen=True)
class CronDefinitionActionRequest:
    """One pending typed action paired with its durable receipt."""

    receipt: ActionReceipt
    action: CronDefinitionAction


@dataclass(frozen=True)
class CronDefinitionStoreReplay:
    """A typed action-store replay suitable for a checked checkpoint."""

    store_namespace: str
    generation: ActionStoreGeneration
    candidate: CronDefinitionCandidate
    pending_actions: tuple[CronDefinitionActionRequest, ...]
    receipts: tuple[ActionReceipt, ...]
    _owner: object = field(repr=False, compare=False)

    def matches_owner(self, owner: object) -> bool:
        """Observe exact owner identity; this is not checkpoint admission."""
        return self._owner is owner


# A short alias keeps call sites free to use either the store-oriented or the
# replay-oriented name without exposing the untyped action-store replay.
DefinitionStoreReplay = CronDefinitionStoreReplay


class CronDefinitionStore:
    """Definition-only interpreter over an injected :class:`CronActionStore`."""

    def __init__(self, action_store: CronActionStore, *, commit_clock_ms: Callable[[], int] | None = None):
        if not isinstance(cast(object, action_store), CronActionStore):
            raise TypeError("action_store must be a CronActionStore")
        self._action_store = action_store
        self._owner = object()
        self._commit_clock_ms = commit_clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._prepared: dict[ActionStoreGeneration, tuple[str, dict[str, Any]]] = {}
        self._normalization_prepared: dict[tuple[ActionStoreGeneration, str, str], dict[str, Any]] = {}

    @property
    def action_store(self) -> CronActionStore:
        """Return the injected journal without widening this adapter's scope."""
        return self._action_store

    def append(
        self,
        *,
        action_id: str,
        action: CronDefinitionAction,
        expected_generation: ActionStoreGeneration | None = None,
    ) -> ActionReceipt:
        """Durably queue one typed definition action without applying it."""
        if not isinstance(cast(object, action), CronDefinitionAction):
            raise DefinitionStoreValidationError("action must be a CronDefinitionAction")
        return self._action_store.append(
            action_id=action_id,
            request=action.to_wire(),
            expected_generation=expected_generation,
        )

    def replay(self) -> CronDefinitionStoreReplay:
        """Materialize the current generic journal into a typed candidate."""
        raw = self._action_store.replay()
        candidate = self._candidate_from_raw(raw.candidate, raw.receipts, raw.generation.watermark)
        typed_pending: list[CronDefinitionActionRequest] = []
        definitions = {definition.job_id: definition for definition in candidate.definitions}
        applied = list(candidate.applied_action_ids)
        for request in raw.pending_actions:
            action = self._typed_pending_action(request)
            self._apply_typed_action(definitions, action)
            applied.append(request.receipt.action_id)
            typed_pending.append(CronDefinitionActionRequest(request.receipt, action))
        result = CronDefinitionCandidate(
            tuple(definitions.values()), tuple(applied), candidate.legacy_conversion_evidence,
            candidate.scheduling_commits, candidate.scheduling_version,
        ).validated()
        expected_ids = tuple(receipt.action_id for receipt in raw.receipts)
        if result.applied_action_ids != expected_ids:
            raise DefinitionStoreValidationError("typed candidate actions do not match durable receipt order")
        for entry in result.scheduling_commits:
            if entry["store_namespace"] != raw.store_namespace:
                raise DefinitionStoreValidationError("scheduling commit namespace mismatch")
            target = entry["target"]
            if target["snapshot_generation"] > raw.generation.snapshot_generation or target["journal_generation"] > raw.generation.journal_generation or target["watermark"] > raw.generation.watermark:
                raise DefinitionStoreValidationError("scheduling commit exceeds published checkpoint")
            receipt = entry["mutation_receipt"]
            if receipt is not None and receipt not in [asdict(value) for value in raw.receipts]:
                raise DefinitionStoreValidationError("scheduling receipt missing from retained journal evidence")
        return CronDefinitionStoreReplay(
            raw.store_namespace,
            raw.generation,
            result,
            tuple(typed_pending),
            raw.receipts,
            self._owner,
        )

    def checkpoint(self, replay: CronDefinitionStoreReplay) -> CronDefinitionStoreReplay:
        """CAS-checkpoint this exact typed replay and return the settled replay."""
        if not isinstance(cast(object, replay), CronDefinitionStoreReplay) or not replay.matches_owner(self._owner):
            raise DefinitionStoreConflictError("replay belongs to another cron definition store")
        candidate = replay.candidate.validated()
        receipt_ids = tuple(receipt.action_id for receipt in replay.receipts)
        if candidate.applied_action_ids != receipt_ids:
            raise DefinitionStoreValidationError("checkpoint candidate actions do not match replay receipts")
        signature = _canonical_json({"candidate": candidate.to_wire(), "pending": [{"receipt": asdict(item.receipt), "action": item.action.to_wire()} for item in replay.pending_actions], "receipts": [asdict(item) for item in replay.receipts]}, label="checkpoint replay")
        prepared = self._prepared.get(replay.generation)
        if prepared is not None:
            if prepared[0] != signature:
                raise DefinitionStoreConflictError("prepared checkpoint replay was mutated")
            wire = prepared[1]
        else:
            current = self.replay()
            current_signature = _canonical_json({"candidate": current.candidate.to_wire(), "pending": [{"receipt": asdict(item.receipt), "action": item.action.to_wire()} for item in current.pending_actions], "receipts": [asdict(item) for item in current.receipts]}, label="checkpoint replay")
            if current.generation != replay.generation or current_signature != signature:
                raise DefinitionStoreConflictError("checkpoint replay is stale or forged")
            entries = list(current.candidate.scheduling_commits)
            keys = {(item["job_instance_id"], item["definition_revision"]) for item in entries}
            clock = self._commit_clock_ms()
            _require_integer(clock, label="commit clock")
            for pending in current.pending_actions:
                definition = pending.action.definition
                if definition is None or (definition.job_instance_id, definition.definition_revision) in keys:
                    continue
                entry = self._commit_entry(definition, current.generation, clock, pending.receipt)
                entries.append(entry)
                keys.add((definition.job_instance_id, definition.definition_revision))
            wire = CronDefinitionCandidate(candidate.definitions, candidate.applied_action_ids, candidate.legacy_conversion_evidence, tuple(entries), 2).to_wire()
            self._prepared[replay.generation] = (signature, wire)
        self._action_store.checkpoint(
            expected_generation=replay.generation,
            candidate=wire,
        )
        return self.replay()

    def _candidate_from_raw(
        self,
        raw_candidate: dict[str, Any] | None,
        receipts: tuple[ActionReceipt, ...],
        watermark: int,
    ) -> CronDefinitionCandidate:
        if raw_candidate is None:
            candidate = CronDefinitionCandidate.empty()
        elif raw_candidate.get("kind") in {_CANDIDATE_KIND, _CANDIDATE_KIND_V2}:
            candidate = CronDefinitionCandidate.decode_wire(raw_candidate)
        elif raw_candidate.get("kind") == _LEGACY_CANDIDATE_KIND:
            candidate = self._materialize_legacy_candidate(raw_candidate, receipts, watermark)
        else:
            raise DefinitionStoreValidationError("action store candidate is not a cron definition candidate")
        checkpointed_ids = tuple(receipt.action_id for receipt in receipts if receipt.sequence <= watermark)
        if candidate.applied_action_ids != checkpointed_ids:
            raise DefinitionStoreValidationError("candidate actions do not match its checkpoint watermark")
        return candidate

    def _commit_entry(self, definition: CronDefinition, generation: ActionStoreGeneration, clock: int, receipt: ActionReceipt | None) -> ScheduleCommit:
        from nanobot.cron.schedule import SCHEDULE_FORMAT, schedule_digest

        schedule = definition.to_wire()["schedule"]
        return _validate_scheduling_commit({"store_namespace": self.action_store.namespace, "job_instance_id": definition.job_instance_id, "definition_revision": definition.definition_revision, "schedule": schedule, "schedule_digest": schedule_digest(schedule), "schedule_format": SCHEDULE_FORMAT, "commit_clock_ms": clock, "commit_kind": "revision" if receipt else "normalization", "mutation_receipt": asdict(receipt) if receipt else None, "predecessor": asdict(generation), "target": {"journal_generation": generation.journal_generation + 1, "snapshot_generation": generation.snapshot_generation + 1, "watermark": generation.sequence, "sequence": generation.sequence}})

    def normalize_missing_evidence(self, *, job_id: str, expected_job_instance_id: str,
                                   expected_definition_revision: str, scheduler_store: object,
                                   occurrence_store: object) -> CronDefinitionStoreReplay:
        """Explicit copied-state stopped-writer operation; never a read/startup path."""
        from nanobot.cron.occurrence_store import CronOccurrenceStore
        from nanobot.cron.schedule import next_fire
        from nanobot.cron.scheduler_store import SchedulerStore

        if not isinstance(scheduler_store, SchedulerStore) or not isinstance(occurrence_store, CronOccurrenceStore) or scheduler_store.namespace != self.action_store.namespace or occurrence_store.namespace != self.action_store.namespace:
            raise DefinitionStoreConflictError("normalization requires exact stopped-writer authorities")
        with scheduler_store.stopped_writer() as lease:
            replay = self.replay()
            if replay.pending_actions:
                raise DefinitionStoreConflictError("normalization cannot checkpoint pending mutations")
            definition = next((item for item in replay.candidate.definitions if item.job_id == job_id), None)
            if definition is None or (definition.job_instance_id, definition.definition_revision) != (expected_job_instance_id, expected_definition_revision):
                raise DefinitionStoreConflictError("normalization definition tokens stale")
            if any(item["job_instance_id"] == definition.job_instance_id and item["definition_revision"] == definition.definition_revision for item in replay.candidate.scheduling_commits):
                return replay
            if any(record.job_snapshot.get("jobInstanceId") == definition.job_instance_id and record.identity.job_revision == definition.definition_revision and record.state == "claimed" for record in occurrence_store.inspect_records()):
                raise DefinitionStoreConflictError("normalization relevant occurrence unresolved")
            key = (replay.generation, definition.job_instance_id, definition.definition_revision)
            wire = self._normalization_prepared.get(key)
            if wire is None:
                clock = self._commit_clock_ms()
                _require_integer(clock, label="normalization clock")
                next_fire(definition.definition["schedule"], clock, clock)
                wire = CronDefinitionCandidate(replay.candidate.definitions, replay.candidate.applied_action_ids,
                    replay.candidate.legacy_conversion_evidence, (*replay.candidate.scheduling_commits,
                    self._commit_entry(definition, replay.generation, clock, None)), 2).to_wire()
                self._normalization_prepared[key] = wire
            lease.reprove()
            self.action_store.checkpoint(expected_generation=replay.generation, candidate=wire)
            return self.replay()

    @staticmethod
    def _typed_pending_action(request: ActionRequest) -> CronDefinitionAction:
        if request.request.get("kind") != _ACTION_KIND:
            raise DefinitionStoreValidationError("pending action is not a typed cron definition action")
        return CronDefinitionAction.decode_wire(request.request)

    @staticmethod
    def _apply_typed_action(definitions: dict[str, CronDefinition], action: CronDefinitionAction) -> None:
        if action.operation == "create":
            definition = _validate_definition_object(action.definition)
            if definition.job_id in definitions:
                raise DefinitionStoreConflictError("cron definition already exists")
            definitions[definition.job_id] = definition
            return
        current = definitions.get(action.job_id)
        if current is None:
            raise DefinitionStoreConflictError("cron definition does not exist")
        if current.job_instance_id != action.expected_job_instance_id:
            raise DefinitionStoreConflictError("cron definition job instance does not match")
        if current.definition_revision != action.expected_definition_revision:
            raise DefinitionStoreConflictError("cron definition revision does not match")
        if action.operation == "update":
            definition = _validate_definition_object(action.definition)
            if definition.job_id != action.job_id:
                raise DefinitionStoreValidationError("updated cron definition has a different job id")
            if definition.job_instance_id != current.job_instance_id:
                raise DefinitionStoreConflictError("updated cron definition has a different job instance")
            definitions[action.job_id] = definition
            return
        if action.operation == "delete":
            del definitions[action.job_id]
            return
        raise DefinitionStoreValidationError("cron definition action has an unknown operation")

    def _materialize_legacy_candidate(
        self,
        value: Mapping[str, Any],
        receipts: tuple[ActionReceipt, ...],
        watermark: int,
    ) -> CronDefinitionCandidate:
        candidate = _copy_mapping(value, label="legacy cron snapshot candidate")
        fields = {
            "schema_version",
            "kind",
            "conversion_identity",
            "jobs_source_digest",
            "legacy_conversion_evidence",
            "jobs",
            "operations",
            "runtime_state_policy",
        }
        _require_exact_fields(candidate, fields, label="legacy cron snapshot candidate")
        if type(candidate["schema_version"]) is not int or candidate["schema_version"] != _SCHEMA_VERSION:
            raise DefinitionStoreValidationError("legacy cron snapshot candidate has an invalid schema version")
        if candidate["kind"] != _LEGACY_CANDIDATE_KIND:
            raise DefinitionStoreValidationError("legacy cron snapshot candidate has an invalid kind")
        conversion_identity = _require_digest(candidate["conversion_identity"], label="legacy conversion identity")
        _require_digest(candidate["jobs_source_digest"], label="legacy jobs source digest")
        evidence = _copy_mapping(candidate["legacy_conversion_evidence"], label="legacy conversion evidence")
        if candidate["runtime_state_policy"] != "snapshot_wins":
            raise DefinitionStoreValidationError("legacy cron snapshot candidate has an invalid runtime state policy")
        jobs = candidate["jobs"]
        operations = candidate["operations"]
        if not isinstance(jobs, list) or not isinstance(operations, list):
            raise DefinitionStoreValidationError("legacy cron snapshot candidate jobs and operations must be lists")
        jobs = cast(list[object], jobs)
        operations = cast(list[object], operations)
        definitions: dict[str, CronDefinition] = {}
        for job in jobs:
            definition = self._legacy_definition(job)
            if definition.job_id in definitions:
                raise DefinitionStoreValidationError("legacy cron snapshot candidate contains duplicate job ids")
            definitions[definition.job_id] = definition

        checkpointed = tuple(receipt for receipt in receipts if receipt.sequence <= watermark)
        if len(operations) != len(checkpointed):
            raise DefinitionStoreValidationError("legacy operations do not match checkpointed receipts")
        applied_ids: list[str] = []
        for operation, receipt in zip(operations, checkpointed, strict=True):
            action_id, request = self._legacy_operation(operation, conversion_identity=conversion_identity)
            if action_id != receipt.action_id:
                raise DefinitionStoreValidationError("legacy operation action ids do not match receipt order")
            if _legacy_request_digest(request) != receipt.request_digest:
                raise DefinitionStoreValidationError("legacy operation request digest does not match its receipt")
            self._apply_legacy_operation(definitions, request)
            applied_ids.append(action_id)
        return CronDefinitionCandidate(tuple(definitions.values()), tuple(applied_ids), evidence).validated()

    @staticmethod
    def _legacy_definition(value: object) -> CronDefinition:
        job = _copy_mapping(value, label="legacy cron job")
        _require_exact_fields(job, _LEGACY_JOB_FIELDS, label="legacy cron job")
        # Runtime/history fields are deliberately discarded.  Their complete
        # validation belongs to the stopped-writer C1 conversion boundary;
        # this C2 adapter never moves them into a definition candidate.
        _copy_mapping(job["state"], label="legacy cron job state")
        _require_integer(job["createdAtMs"], label="legacy cron job createdAtMs")
        _require_integer(job["updatedAtMs"], label="legacy cron job updatedAtMs")
        return CronDefinition.decode_wire(
            {name: job[name] for name in _DEFINITION_WIRE_FIELDS}, label="legacy cron job definition"
        )

    @staticmethod
    def _legacy_operation(value: object, *, conversion_identity: str) -> tuple[str, dict[str, Any]]:
        operation = _copy_mapping(value, label="legacy cron operation")
        _require_exact_fields(operation, {"action_id", "request"}, label="legacy cron operation")
        action_id = _require_digest(operation["action_id"], label="legacy cron operation action_id")
        request = _copy_mapping(operation["request"], label="legacy cron operation request")
        if request.get("kind") != _LEGACY_ACTION_KIND:
            raise DefinitionStoreValidationError("legacy cron operation has an invalid request kind")
        source = _copy_mapping(request.get("source"), label="legacy cron operation source")
        _require_exact_fields(
            source, {"conversion_identity", "source_digest", "line_ordinal"}, label="legacy cron operation source"
        )
        if _require_digest(source["conversion_identity"], label="legacy cron operation conversion identity") != conversion_identity:
            raise DefinitionStoreValidationError("legacy cron operation belongs to another conversion")
        _require_digest(source["source_digest"], label="legacy cron operation source digest")
        _require_integer(source["line_ordinal"], label="legacy cron operation line ordinal", minimum=1)
        if request.get("operation") == "upsert":
            fields = {"kind", "operation", "legacy_operation", "job_id", "job", "runtime_state_policy", "source"}
            _require_exact_fields(request, fields, label="legacy cron upsert request")
            if request["legacy_operation"] not in {"add", "update"}:
                raise DefinitionStoreValidationError("legacy cron upsert has an invalid legacy operation")
            if request["runtime_state_policy"] != "snapshot_wins":
                raise DefinitionStoreValidationError("legacy cron upsert has an invalid runtime state policy")
            definition = CronDefinitionStore._legacy_definition(request["job"])
            if request["job_id"] != definition.job_id:
                raise DefinitionStoreValidationError("legacy cron upsert job_id does not match its job")
        elif request.get("operation") == "delete":
            _require_exact_fields(
                request, {"kind", "operation", "legacy_operation", "job_id", "source"}, label="legacy cron delete request"
            )
            if request["legacy_operation"] != "del":
                raise DefinitionStoreValidationError("legacy cron delete has an invalid legacy operation")
            _require_string(request["job_id"], label="legacy cron delete job_id")
        else:
            raise DefinitionStoreValidationError("legacy cron operation has an unknown operation")
        return action_id, request

    @staticmethod
    def _apply_legacy_operation(definitions: dict[str, CronDefinition], request: Mapping[str, Any]) -> None:
        if request["operation"] == "upsert":
            definition = CronDefinitionStore._legacy_definition(request["job"])
            definitions[definition.job_id] = definition
        elif request["operation"] == "delete":
            definitions.pop(request["job_id"], None)
        else:  # _legacy_operation validates this before calling us.
            raise DefinitionStoreValidationError("legacy cron operation has an unknown operation")


def _legacy_request_digest(request: Mapping[str, Any]) -> str:
    """Match the C1 public durable-request digest algorithm exactly."""
    return hashlib.sha256(
        b"nanobot-cron-action-v2\0" + _canonical_json(dict(request), label="legacy operation").encode("utf-8")
    ).hexdigest()
