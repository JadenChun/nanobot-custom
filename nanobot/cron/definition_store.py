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
from dataclasses import dataclass, field
from typing import Any, Mapping

from nanobot.cron.action_store import (
    ActionReceipt,
    ActionRequest,
    ActionStoreGeneration,
    CronActionStore,
)

_SCHEMA_VERSION = 1
_CANDIDATE_KIND = "cron_definition_candidate_v1"
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
    encoded = _canonical_json(dict(value), label=label)
    try:
        copied = json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as exc:  # pragma: no cover - json.dumps made this defensive
        raise DefinitionStoreValidationError(f"{label} must be JSON data") from exc
    if not isinstance(copied, dict):  # pragma: no cover - mappings always encode as objects
        raise DefinitionStoreValidationError(f"{label} must be a JSON object")
    return copied


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
    _require_exact_fields(payload, _PAYLOAD_FIELDS, label="cron definition payload")
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
    for destination in destinations:
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

    def _validated(self) -> CronDefinition:
        validated = CronDefinition.from_definition(self.definition, job_instance_id=self.job_instance_id)
        if self.definition_revision != validated.definition_revision:
            raise DefinitionStoreValidationError("cron definition definitionRevision does not match its definition")
        return validated

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
            definition = CronDefinition._from_wire(request["definition"], label="cron definition create action definition")
            if request["job_id"] != definition.job_id:
                raise DefinitionStoreValidationError("cron definition create action job_id does not match its definition")
            return cls.create(definition)
        if operation == "update":
            _require_exact_fields(
                request,
                base_fields | {"definition", "expected_job_instance_id", "expected_definition_revision"},
                label="cron definition update action",
            )
            definition = CronDefinition._from_wire(request["definition"], label="cron definition update action definition")
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
    return value._validated()


@dataclass(frozen=True)
class CronDefinitionCandidate:
    """The durable definition-only state derived from an action-store replay."""

    definitions: tuple[CronDefinition, ...] = ()
    applied_action_ids: tuple[str, ...] = ()
    legacy_conversion_evidence: dict[str, Any] | None = None

    @classmethod
    def _empty(cls) -> CronDefinitionCandidate:
        return cls()

    @classmethod
    def _from_wire(cls, value: object) -> CronDefinitionCandidate:
        candidate = _copy_mapping(value, label="cron definition candidate")
        fields = {"schema_version", "kind", "definitions", "applied_action_ids"}
        if "legacy_conversion_evidence" in candidate:
            fields.add("legacy_conversion_evidence")
        _require_exact_fields(candidate, fields, label="cron definition candidate")
        if type(candidate["schema_version"]) is not int or candidate["schema_version"] != _SCHEMA_VERSION:
            raise DefinitionStoreValidationError("cron definition candidate has an invalid schema version")
        if candidate["kind"] != _CANDIDATE_KIND:
            raise DefinitionStoreValidationError("cron definition candidate has an invalid kind")
        if not isinstance(candidate["definitions"], list):
            raise DefinitionStoreValidationError("cron definition candidate definitions must be a list")
        definitions = tuple(CronDefinition._from_wire(item, label="cron definition candidate definition") for item in candidate["definitions"])
        ids = tuple(definition.job_id for definition in definitions)
        if len(set(ids)) != len(ids):
            raise DefinitionStoreValidationError("cron definition candidate contains duplicate job ids")
        action_ids_value = candidate["applied_action_ids"]
        if not isinstance(action_ids_value, list):
            raise DefinitionStoreValidationError("cron definition candidate applied_action_ids must be a list")
        action_ids = tuple(_require_action_id(item, label="cron definition candidate action id") for item in action_ids_value)
        if len(set(action_ids)) != len(action_ids):
            raise DefinitionStoreValidationError("cron definition candidate contains duplicate action ids")
        evidence = None
        if "legacy_conversion_evidence" in candidate:
            evidence = _copy_mapping(candidate["legacy_conversion_evidence"], label="legacy conversion evidence")
        return cls(definitions, action_ids, evidence)

    def _validated(self) -> CronDefinitionCandidate:
        definitions = tuple(_validate_definition_object(definition) for definition in self.definitions)
        if len({definition.job_id for definition in definitions}) != len(definitions):
            raise DefinitionStoreValidationError("cron definition candidate contains duplicate job ids")
        action_ids = tuple(_require_action_id(action_id, label="cron definition candidate action id") for action_id in self.applied_action_ids)
        if len(set(action_ids)) != len(action_ids):
            raise DefinitionStoreValidationError("cron definition candidate contains duplicate action ids")
        evidence = (
            None
            if self.legacy_conversion_evidence is None
            else _copy_mapping(self.legacy_conversion_evidence, label="legacy conversion evidence")
        )
        return CronDefinitionCandidate(definitions, action_ids, evidence)

    def to_wire(self) -> dict[str, Any]:
        validated = self._validated()
        wire: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "kind": _CANDIDATE_KIND,
            "definitions": [definition.to_wire() for definition in validated.definitions],
            "applied_action_ids": list(validated.applied_action_ids),
        }
        if validated.legacy_conversion_evidence is not None:
            wire["legacy_conversion_evidence"] = validated.legacy_conversion_evidence
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


# A short alias keeps call sites free to use either the store-oriented or the
# replay-oriented name without exposing the untyped action-store replay.
DefinitionStoreReplay = CronDefinitionStoreReplay


class CronDefinitionStore:
    """Definition-only interpreter over an injected :class:`CronActionStore`."""

    def __init__(self, action_store: CronActionStore):
        if not isinstance(action_store, CronActionStore):
            raise TypeError("action_store must be a CronActionStore")
        self._action_store = action_store
        self._owner = object()

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
        if not isinstance(action, CronDefinitionAction):
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
            tuple(definitions.values()), tuple(applied), candidate.legacy_conversion_evidence
        )._validated()
        expected_ids = tuple(receipt.action_id for receipt in raw.receipts)
        if result.applied_action_ids != expected_ids:
            raise DefinitionStoreValidationError("typed candidate actions do not match durable receipt order")
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
        if not isinstance(replay, CronDefinitionStoreReplay) or replay._owner is not self._owner:
            raise DefinitionStoreConflictError("replay belongs to another cron definition store")
        candidate = replay.candidate._validated()
        receipt_ids = tuple(receipt.action_id for receipt in replay.receipts)
        if candidate.applied_action_ids != receipt_ids:
            raise DefinitionStoreValidationError("checkpoint candidate actions do not match replay receipts")
        self._action_store.checkpoint(
            expected_generation=replay.generation,
            candidate=candidate.to_wire(),
        )
        return self.replay()

    def _candidate_from_raw(
        self,
        raw_candidate: dict[str, Any] | None,
        receipts: tuple[ActionReceipt, ...],
        watermark: int,
    ) -> CronDefinitionCandidate:
        if raw_candidate is None:
            candidate = CronDefinitionCandidate._empty()
        elif raw_candidate.get("kind") == _CANDIDATE_KIND:
            candidate = CronDefinitionCandidate._from_wire(raw_candidate)
        elif raw_candidate.get("kind") == _LEGACY_CANDIDATE_KIND:
            candidate = self._materialize_legacy_candidate(raw_candidate, receipts, watermark)
        else:
            raise DefinitionStoreValidationError("action store candidate is not a cron definition candidate")
        checkpointed_ids = tuple(receipt.action_id for receipt in receipts if receipt.sequence <= watermark)
        if candidate.applied_action_ids != checkpointed_ids:
            raise DefinitionStoreValidationError("candidate actions do not match its checkpoint watermark")
        return candidate

    @staticmethod
    def _typed_pending_action(request: ActionRequest) -> CronDefinitionAction:
        if request.request.get("kind") != _ACTION_KIND:
            raise DefinitionStoreValidationError("pending action is not a typed cron definition action")
        return CronDefinitionAction._from_wire(request.request)

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
        return CronDefinitionCandidate(tuple(definitions.values()), tuple(applied_ids), evidence)._validated()

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
        return CronDefinition._from_wire(
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
