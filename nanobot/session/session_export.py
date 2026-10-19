"""Stopped-writer canonical export with a strict legacy round-trip proof."""

from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, cast

from nanobot.runtime_context import public_history_message
from nanobot.session.artifact import (
    ArtifactDescriptor,
    parse_artifact_descriptor,
    parse_artifact_reference,
)
from nanobot.session.context_checkpoint import validate_context_checkpoint
from nanobot.session.manager import JsonlSessionStore, Session, SessionManager, SessionStorageError
from nanobot.session.persistence_metadata import LEGACY_RECOVERY_KEYS, accepted_metadata_updates
from nanobot.session.session_mode import (
    LockedSessionSnapshot,
    SessionExportTransaction,
    SessionModeError,
)
from nanobot.session.source import SourceValidationError, validate_execution_links
from nanobot.session.source_store import validate_source_session_bytes

_UNSUPPORTED = "compatibility export is unsupported"


class _UnsupportedCompatibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SessionExportResult:
    destination: Path
    compatibility_included: bool


class LegacyExportParser(Protocol):
    """Selected stateless parser, with no session-backend access."""

    def __call__(self, data: bytes, /, *, expected_key: str) -> Session: ...


@dataclass(frozen=True, slots=True)
class CanonicalExportOperations:
    """Typed command handoff, not storage authority or an admission proof."""

    locked_snapshot: Callable[[str], AbstractContextManager[LockedSessionSnapshot]]
    begin_export: Callable[[str, Path, dict[str, bytes]], SessionExportTransaction]
    select_legacy_parser: Callable[[], LegacyExportParser]


@dataclass(frozen=True, slots=True)
class _CompatibilityProof:
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    archive_boundary: int
    tool_pairs: tuple[tuple[int, int, int, str], ...]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _public_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            name: _public_value(item)
            for name, item in cast(dict[object, object], value).items()
            if isinstance(name, str) and not name.startswith("_")
        }
    if isinstance(value, list):
        return [_public_value(item) for item in cast(list[object], value)]
    return deepcopy(value)


def _public_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for event in events:
        if event.get("role") not in {"user", "assistant", "tool"}:
            raise _UnsupportedCompatibilityError()
        public = _public_value(public_history_message(event))
        if not isinstance(public, dict):
            raise _UnsupportedCompatibilityError()
        # _public_value constructs only string-keyed objects and strips private keys.
        messages.append(cast(dict[str, Any], public))
    return messages


def _tool_pairs(messages: list[dict[str, Any]]) -> tuple[tuple[int, int, int, str], ...]:
    pending: dict[str, tuple[int, int]] = {}
    pairs: list[tuple[int, int, int, str]] = []
    for message_index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if calls is not None:
            if message.get("role") != "assistant" or not isinstance(calls, list):
                raise _UnsupportedCompatibilityError()
            for call_index, call in enumerate(cast(list[object], calls)):
                if not isinstance(call, dict):
                    raise _UnsupportedCompatibilityError()
                provider_id = cast(dict[object, object], call).get("id")
                if (
                    not isinstance(provider_id, str)
                    or not provider_id
                    or provider_id in pending
                ):
                    raise _UnsupportedCompatibilityError()
                pending[provider_id] = (message_index, call_index)
        if message.get("role") == "tool":
            provider_id = message.get("tool_call_id")
            if not isinstance(provider_id, str) or provider_id not in pending:
                raise _UnsupportedCompatibilityError()
            assistant_index, call_index = pending.pop(provider_id)
            pairs.append((assistant_index, call_index, message_index, provider_id))
    if pending:
        raise _UnsupportedCompatibilityError()
    return tuple(pairs)


def _artifact_descriptors(
    key: str,
    header: dict[str, Any],
) -> tuple[ArtifactDescriptor, ...]:
    raw_manifest = header["metadata"].get("artifact_manifest")
    if raw_manifest is None:
        return ()
    if not isinstance(raw_manifest, dict):
        raise _UnsupportedCompatibilityError()
    manifest = cast(dict[object, object], raw_manifest)
    raw_descriptors = manifest.get("descriptors")
    if (
        set(manifest) != {"schema_version", "descriptors"}
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or not isinstance(raw_descriptors, dict)
    ):
        raise _UnsupportedCompatibilityError()
    descriptors: list[ArtifactDescriptor] = []
    for artifact_id, raw in cast(dict[object, object], raw_descriptors).items():
        descriptor = parse_artifact_descriptor(raw)
        if (
            artifact_id != descriptor.artifact_id
            or descriptor.session_key != key
            or descriptor.incarnation != header["incarnation"]
            or descriptor.generation != header["generation"]
        ):
            raise _UnsupportedCompatibilityError()
        descriptors.append(descriptor)
    return tuple(descriptors)


def _artifact_references(
    key: str,
    header: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    require_inline: bool,
) -> tuple[str, ...]:
    references: set[str] = set()
    events_by_id = {event.get("_event_id"): event for event in events}
    for event in events:
        receipt = event.get("_artifact_read")
        if receipt is None:
            continue
        if not isinstance(receipt, dict):
            raise _UnsupportedCompatibilityError()
        reference = cast(dict[object, object], receipt).get("reference")
        if not isinstance(reference, str):
            raise _UnsupportedCompatibilityError()
        parse_artifact_reference(reference)
        if require_inline and event.get("content") is None:
            raise _UnsupportedCompatibilityError()
        references.add(reference)

    for descriptor in _artifact_descriptors(key, header):
        references.add(descriptor.reference)
        if not require_inline:
            continue
        event = events_by_id.get(descriptor.event_id)
        if not isinstance(event, dict):
            raise _UnsupportedCompatibilityError()
        content = event.get("content")
        if descriptor.block_index is None:
            text = content
        elif isinstance(content, list) and descriptor.block_index < len(cast(list[object], content)):
            block = cast(list[object], content)[descriptor.block_index]
            block_values = cast(dict[object, object], block) if isinstance(block, dict) else {}
            text = (
                block_values.get("text")
                if block_values.get("type") == "text"
                else None
            )
        else:
            text = None
        if not isinstance(text, str):
            raise _UnsupportedCompatibilityError()
        encoded = text.encode("utf-8")
        if len(encoded) != descriptor.byte_length or _digest(encoded) != descriptor.sha256:
            raise _UnsupportedCompatibilityError()
    return tuple(sorted(references))


def _checkpoint_details(
    header: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[int | None, tuple[str, ...]]:
    candidate = header["metadata"].get("context_checkpoint")
    if candidate is None:
        return None, ()
    if not isinstance(candidate, dict):
        return None, ()
    # The header comes from validate_source_session_bytes, whose recursive JSON
    # validation has already established string keys for every nested object.
    candidate_values = cast(dict[str, Any], candidate)
    try:
        source_revision = candidate_values.get("source_revision")
        parent = candidate_values.get("parent_checkpoint_id")
        if (
            type(source_revision) is not int
            or source_revision < 0
            or source_revision >= header["revision"]
            or (parent is not None and (not isinstance(parent, str) or not parent))
        ):
            return None, ()
        checkpoint = validate_context_checkpoint(
            candidate_values,
            events,
            generation=header["generation"],
            source_revision=source_revision,
            parent_checkpoint_id=parent,
        )
    except SourceValidationError:
        return None, ()
    return checkpoint["schema_version"], tuple(checkpoint["artifact_references"])


def _proof_from_source(
    key: str,
    header: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[_CompatibilityProof, bytes, tuple[str, ...]]:
    metadata = header["metadata"]
    if (
        "execution_lease" in metadata
        or "execution_recovery_required" in metadata
        or any(name in metadata for name in LEGACY_RECOVERY_KEYS)
    ):
        raise _UnsupportedCompatibilityError()
    if validate_execution_links(events):
        raise _UnsupportedCompatibilityError()
    messages = _public_messages(events)
    references = _artifact_references(key, header, events, require_inline=True)
    portable = accepted_metadata_updates(metadata, source=True)
    boundary = header["last_archived"]
    proof = _CompatibilityProof(messages, portable, boundary, _tool_pairs(messages))
    session = Session(
        key=key,
        messages=deepcopy(messages),
        created_at=datetime.fromisoformat(header["created_at"]),
        updated_at=datetime.fromisoformat(header["updated_at"]),
        metadata=deepcopy(portable),
        last_consolidated=boundary,
        revision=header["revision"],
        generation=header["generation"],
        incarnation=header["incarnation"],
    )
    return proof, JsonlSessionStore.render_legacy_bytes(session), references


def _proof_from_legacy(session: Session) -> _CompatibilityProof:
    messages = _public_messages(session.messages)
    return _CompatibilityProof(
        messages,
        deepcopy(session.metadata),
        session.last_archived,
        _tool_pairs(messages),
    )


def _tentative_compatibility(
    key: str,
    header: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[bytes, _CompatibilityProof | None, tuple[str, ...]]:
    try:
        proof, data, references = _proof_from_source(key, header, events)
        return data, proof, references
    except (
        SourceValidationError,
        _UnsupportedCompatibilityError,
        UnicodeError,
        ValueError,
        TypeError,
    ):
        fallback = Session(
            key=key,
            created_at=datetime.fromisoformat(header["created_at"]),
            updated_at=datetime.fromisoformat(header["updated_at"]),
            revision=header["revision"],
            generation=header["generation"],
            incarnation=header["incarnation"],
        )
        return JsonlSessionStore.render_legacy_bytes(fallback), None, ()


def export_canonical_session(
    manager: SessionManager,
    key: str,
    destination: Path,
    *,
    include_compatibility: bool = False,
) -> SessionExportResult:
    """Publish one exact copied-state bundle to a new absolute destination."""
    operations = SessionManager.canonical_export_operations(manager)
    target = Path(destination).expanduser()
    if not target.is_absolute() or target.exists() or target == Path(target.anchor):
        raise SessionModeError("export destination must be absolute and nonexistent")

    with operations.locked_snapshot(key) as locked:
        registry_id, record_revision, mode, incarnation, generation = locked.binding
        if mode != "canonical":
            raise SessionModeError("canonical session export is unavailable")
        try:
            header, events = validate_source_session_bytes(locked.source_bytes, expected_key=key)
        except SourceValidationError:
            raise SessionModeError("canonical session export is unavailable") from None
        if header["incarnation"] != incarnation or header["generation"] != generation:
            raise SessionModeError("canonical session export is unavailable")

        checkpoint_schema, checkpoint_references = _checkpoint_details(header, events)
        compatibility: bytes | None = None
        compatibility_proof: _CompatibilityProof | None = None
        try:
            source_references = _artifact_references(
                key, header, events, require_inline=False
            )
        except (SourceValidationError, _UnsupportedCompatibilityError):
            raise SessionModeError("canonical session export is unavailable") from None
        if include_compatibility:
            compatibility, compatibility_proof, _inline_references = _tentative_compatibility(
                key, header, events
            )

        artifact_references = sorted(set(checkpoint_references) | set(source_references))
        manifest: dict[str, object] = {
            "schema_version": 1,
            "classification": "canonical-lossless",
            "session_key": key,
            "registry_id": registry_id,
            "record_revision": record_revision,
            "mode": mode,
            "incarnation": incarnation,
            "generation": generation,
            "source_schema_version": header["metadata"]["source_schema_version"],
            "checkpoint_schema_version": checkpoint_schema,
            "artifact_references": artifact_references,
            "artifact_blobs_portable": False,
            "source_file": "canonical-source.jsonl",
            "source_digest": _digest(locked.source_bytes),
            "source_size": len(locked.source_bytes),
            "compatibility_file": (
                "compatibility-history.jsonl" if include_compatibility else None
            ),
            "compatibility_classification": "compatibility" if include_compatibility else None,
        }
        files = {
            "canonical-source.jsonl": locked.source_bytes,
            "manifest.json": _canonical_json(manifest),
        }
        if compatibility is not None:
            files["compatibility-history.jsonl"] = compatibility

        transaction = operations.begin_export(key, target, files)
        try:
            for name, expected in files.items():
                if transaction.read_staged(name) != expected:
                    raise _UnsupportedCompatibilityError()
            if include_compatibility:
                if compatibility_proof is None:
                    raise _UnsupportedCompatibilityError()
                parse_legacy_bytes = operations.select_legacy_parser()
                parsed = parse_legacy_bytes(
                    transaction.read_staged("compatibility-history.jsonl"),
                    expected_key=key,
                )
                if _proof_from_legacy(parsed) != compatibility_proof:
                    raise _UnsupportedCompatibilityError()
        except (
            SessionModeError,
            SessionStorageError,
            SourceValidationError,
            _UnsupportedCompatibilityError,
            UnicodeError,
            ValueError,
            TypeError,
        ):
            transaction.abort()
            raise SessionModeError(_UNSUPPORTED) from None

        transaction.mark_published()
        transaction.complete()
    return SessionExportResult(target, include_compatibility)


__all__ = ["SessionExportResult", "export_canonical_session"]
