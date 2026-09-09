"""Immutable source-event contracts for the dormant governed session store.

These helpers do not upgrade sessions or write storage. Only a trusted runtime
boundary may allocate identities; provider dictionaries are never identities.
"""

import hashlib
import json
import math
import secrets
from copy import deepcopy
from typing import Any


class SourceValidationError(ValueError):
    """Content-free rejection of invalid persisted source evidence."""


def _validate_json(value: object) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise SourceValidationError("Invalid source representation")


def canonical_events(events: list[dict[str, Any]]) -> bytes:
    """Version 1 canonical representation, including persisted internal fields."""
    try:
        _validate_json(events)
        return json.dumps(
            events, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise SourceValidationError("Invalid source representation") from None


def prefix_digest(events: list[dict[str, Any]], boundary: int) -> str:
    if type(boundary) is not int or not 0 <= boundary <= len(events):
        raise SourceValidationError("Invalid source boundary")
    return hashlib.sha256(canonical_events(events[:boundary])).hexdigest()


def new_source_event(payload: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    """Detach untrusted payload and assign trusted attribution before persistence.

    Execution links are assigned separately by the runtime after allocating an
    assistant-call event. Never accept them from provider output.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise SourceValidationError("Invalid source run identity")
    if not isinstance(payload, dict):
        raise SourceValidationError("Invalid source event representation")
    canonical_events([payload])
    event = deepcopy(payload)
    for key in ("_event_id", "_run_id", "_execution_id", "_execution_ids"):
        event.pop(key, None)
    event.update(_event_id=secrets.token_hex(16), _run_id=run_id)
    canonical_events([event])
    return event


def validate_event_ids(events: list[dict[str, Any]]) -> None:
    if not isinstance(events, list):
        raise SourceValidationError("Invalid source event representation")
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            raise SourceValidationError("Invalid source event representation")
        identity = event.get("_event_id")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise SourceValidationError("Invalid source event identity")
        run_id = event.get("_run_id")
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise SourceValidationError("Invalid source run identity")
        seen.add(identity)
    canonical_events(events)


def merge_source_events(
    committed: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply an ordered retry after the caller has independently checked CAS.

    Committed retry rows must be an ordered suffix, preceding every new row.
    Reject overlapping/interleaved retries rather than silently reorder evidence.
    """
    validate_event_ids(committed)
    validate_event_ids(incoming)
    positions = {event["_event_id"]: i for i, event in enumerate(committed)}
    retry_positions: list[int] = []
    additions: list[dict[str, Any]] = []
    for event in incoming:
        position = positions.get(event["_event_id"])
        if position is None:
            additions.append(event)
        else:
            if additions or canonical_events([event]) != canonical_events([committed[position]]):
                raise SourceValidationError("Conflicting source event retry")
            retry_positions.append(position)
    if retry_positions and retry_positions != list(range(retry_positions[0], len(committed))):
        raise SourceValidationError("Conflicting source event order")
    return deepcopy(committed + additions)


def validate_closed_boundary(events: list[dict[str, Any]], boundary: int) -> None:
    """Require runtime execution identities, independent of provider call IDs.

    Assistant `_execution_ids` aligns with `tool_calls`. Tool `_execution_id`
    references exactly one earlier call; duplicate provider IDs remain legal.
    """
    prefix_digest(events, boundary)
    if validate_execution_links(events[:boundary]):
        raise SourceValidationError("Unresolved source execution boundary")


def validate_execution_links(events: list[dict[str, Any]]) -> frozenset[str]:
    """Validate committed call/result evidence, permitting unresolved tail calls."""
    canonical_events(events)
    pending: dict[str, tuple[str, object]] = {}
    used: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            raise SourceValidationError("Invalid source event representation")
        calls = event.get("tool_calls")
        if "tool_calls" in event:
            identities = event.get("_execution_ids")
            if (
                event.get("role") != "assistant"
                or not isinstance(calls, list)
                or not isinstance(identities, list)
                or len(identities) != len(calls)
            ):
                raise SourceValidationError("Invalid source execution linkage")
            for identity, call in zip(identities, calls):
                if not isinstance(identity, str) or not identity.strip() or identity in used:
                    raise SourceValidationError("Invalid source execution identity")
                if (
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or not call["id"].strip()
                ):
                    raise SourceValidationError("Invalid source provider call identity")
                used.add(identity)
                pending[identity] = (call["id"], event.get("_run_id"))
        if event.get("role") == "tool":
            identity = event.get("_execution_id")
            if not isinstance(identity, str) or identity not in pending:
                raise SourceValidationError("Invalid source execution result")
            provider_id, run_id = pending[identity]
            if event.get("tool_call_id") != provider_id or (
                run_id is not None and event.get("_run_id") is not None and event["_run_id"] != run_id
            ):
                raise SourceValidationError("Invalid source result provenance")
            del pending[identity]
    return frozenset(pending)
