"""Strict text-artifact descriptors and canonical source binding.

Parsing establishes representation and source integrity only. A descriptor or
opaque reference grants no access: publication needs an execution owner, and
retrieval must independently recheck current scope and source policy.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from nanobot.session.source import (
    SourceValidationError,
    validate_event_ids,
    validate_execution_links,
)
from nanobot.session.source_store import SourceSnapshot

if TYPE_CHECKING:
    from nanobot.agent.tools.read_provenance import FilesystemReadReceipt

_ID = re.compile(r'[0-9a-f]{32}\Z')
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')


class ArtifactValidationError(SourceValidationError):
    """Content-free rejection of an invalid artifact representation or binding."""


def _identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ArtifactValidationError('Invalid artifact binding identity')
    return value


def _opaque_id(value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ArtifactValidationError('Invalid opaque artifact identity')
    return value


def parse_artifact_reference(value: object) -> str:
    """Return only a strict opaque ID; never interpret paths or URLs."""
    if not isinstance(value, str) or not value.startswith('artifact:'):
        raise ArtifactValidationError('Invalid artifact reference')
    return _opaque_id(value[len('artifact:'):])


@dataclass(frozen=True)
class FilesystemArtifactOrigin:
    """Recorded locator for later policy revalidation, never an access grant."""

    kind: Literal['filesystem']
    access_class: Literal['workspace', 'context_repository', 'target_repository']
    repository: str | None
    relative_path: str


def _origin(raw: object) -> FilesystemArtifactOrigin:
    if not isinstance(raw, dict):
        raise ArtifactValidationError('Invalid artifact source locator')
    # Only the container type is claimed here; every value is still untrusted.
    record = cast(dict[object, object], raw)
    if set(record) != {'kind', 'access_class', 'repository', 'relative_path'}:
        raise ArtifactValidationError('Invalid artifact source locator')
    if record['kind'] != 'filesystem':
        raise ArtifactValidationError('Unsupported artifact provenance adapter')
    access = record['access_class']
    if access not in ('workspace', 'context_repository', 'target_repository'):
        raise ArtifactValidationError('Invalid artifact source access class')
    repository = record['repository']
    if access == 'workspace':
        if repository is not None:
            raise ArtifactValidationError('Workspace artifact cannot claim a repository')
    else:
        repository = _identity(repository)
    path = record['relative_path']
    if (not isinstance(path, str) or not path or '\\' in path or ':' in path
            or any(ord(char) < 32 or ord(char) == 127 for char in path)
            or any(part in ('', '.', '..') for part in path.split('/'))):
        raise ArtifactValidationError('Invalid relative artifact source locator')
    return FilesystemArtifactOrigin('filesystem', access, repository, path)


def parse_filesystem_artifact_origin(raw: object) -> FilesystemArtifactOrigin:
    """Validate recorded locator metadata without granting filesystem access."""
    return _origin(raw)


def select_filesystem_artifact_origin_parser() -> Callable[[object], FilesystemArtifactOrigin]:
    """Select the current syntax parser before inspecting an untrusted container."""
    return _origin


@dataclass(frozen=True)
class ArtifactDescriptor:
    schema_version: int
    artifact_id: str
    workspace_id: str
    session_key: str
    incarnation: str
    generation: int
    event_id: str
    execution_id: str
    stream_id: str
    run_id: str
    block_index: int | None
    mime_type: str
    byte_length: int
    sha256: str
    created_at: str
    source_access: FilesystemArtifactOrigin

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def reference(self) -> str:
        return 'artifact:' + _opaque_id(self.artifact_id)

    @property
    def idempotency_key(self) -> tuple[str, int | None]:
        return self.event_id, self.block_index


def parse_artifact_descriptor(raw: object) -> ArtifactDescriptor:
    """Validate every persisted field without opening files or authorizing reads."""
    fields = {'schema_version', 'artifact_id', 'workspace_id', 'session_key', 'incarnation',
              'generation', 'event_id', 'execution_id', 'stream_id', 'run_id', 'block_index',
              'mime_type', 'byte_length', 'sha256', 'created_at', 'source_access'}
    if not isinstance(raw, dict):
        raise ArtifactValidationError('Invalid artifact descriptor schema')
    record = cast(dict[object, object], raw)
    if set(record) != fields:
        raise ArtifactValidationError('Invalid artifact descriptor schema')
    if type(record['schema_version']) is not int or record['schema_version'] != 1:
        raise ArtifactValidationError('Unsupported artifact descriptor version')
    counters: dict[str, int] = {}
    for name in ('generation', 'byte_length'):
        value = record[name]
        if type(value) is not int or value < 0:
            raise ArtifactValidationError('Invalid artifact counter')
        counters[name] = value
    block = record['block_index']
    if block is not None and (type(block) is not int or block < 0):
        raise ArtifactValidationError('Invalid artifact text block selector')
    digest = record['sha256']
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ArtifactValidationError('Invalid artifact content digest')
    if record['mime_type'] != 'text/plain; charset=utf-8':
        raise ArtifactValidationError('Unsupported artifact content type')
    timestamp = record['created_at']
    try:
        if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp).tzinfo is None:
            raise ValueError
    except ValueError:
        raise ArtifactValidationError('Invalid artifact timestamp') from None
    return ArtifactDescriptor(
        1, _opaque_id(record['artifact_id']), _opaque_id(record['workspace_id']),
        _identity(record['session_key']), _identity(record['incarnation']), counters['generation'],
        _identity(record['event_id']), _identity(record['execution_id']), _identity(record['stream_id']),
        _identity(record['run_id']), block, 'text/plain; charset=utf-8', counters['byte_length'], digest,
        timestamp, _origin(record['source_access']),
    )


def bound_artifact_text(descriptor: ArtifactDescriptor, snapshot: SourceSnapshot, *, workspace_id: str) -> bytes:
    """Verify exact canonical text and provenance; this is not read authorization.

    Callers obtain workspace_id from the session store's persisted workspace
    namespace. Neither the descriptor nor a model-supplied string selects it.
    """
    descriptor = parse_artifact_descriptor(descriptor.to_dict())
    if (descriptor.workspace_id != _opaque_id(workspace_id)
            or descriptor.session_key != snapshot.state.session_key
            or descriptor.incarnation != snapshot.state.incarnation
            or descriptor.generation != snapshot.state.generation):
        raise ArtifactValidationError('Artifact belongs to another source namespace')
    validate_event_ids(snapshot.events)
    validate_execution_links(snapshot.events)
    event = next((event for event in snapshot.events if event['_event_id'] == descriptor.event_id), None)
    if (event is None or event.get('role') != 'tool'
            or event.get('_execution_id') != descriptor.execution_id
            or event.get('_stream_id') != descriptor.stream_id
            or event.get('_run_id') != descriptor.run_id):
        raise ArtifactValidationError('Artifact does not match its source execution')
    intent = next((row for row in snapshot.events
                   if row.get('role') == 'assistant' and 'tool_calls' in row
                   and descriptor.execution_id in row['_execution_ids']), None)
    if (intent is None or intent.get('_stream_id') != descriptor.stream_id
            or intent.get('_run_id') != descriptor.run_id
            or intent.get('_parent_stream_id') != event.get('_parent_stream_id')):
        raise ArtifactValidationError('Artifact call/result stream provenance differs')
    content = event.get('content')
    if descriptor.block_index is None:
        text = content
    else:
        if not isinstance(content, list):
            raise ArtifactValidationError('Artifact text block is absent')
        blocks = cast(list[object], content)
        if descriptor.block_index >= len(blocks):
            raise ArtifactValidationError('Artifact text block is absent')
        block = blocks[descriptor.block_index]
        if not isinstance(block, dict):
            raise ArtifactValidationError('Artifact selector does not identify text')
        block_fields = cast(dict[object, object], block)
        if block_fields.get('type') != 'text':
            raise ArtifactValidationError('Artifact selector does not identify text')
        text = block_fields.get('text')
    if not isinstance(text, str):
        raise ArtifactValidationError('Artifact source is not text')
    try:
        encoded = text.encode('utf-8')
    except UnicodeError:
        raise ArtifactValidationError('Artifact text is not valid UTF-8') from None
    if len(encoded) != descriptor.byte_length or hashlib.sha256(encoded).hexdigest() != descriptor.sha256:
        raise ArtifactValidationError('Artifact content does not match source integrity')
    return encoded


def published_artifacts(snapshot: SourceSnapshot, *, workspace_id: str) -> tuple[ArtifactDescriptor, ...]:
    """Validate manifest membership and source integrity, never retrieval access.

    Invalid metadata is retained by source readers for rollback. Artifact users
    must reject it rather than replacing it with an empty, writable manifest.
    """
    _opaque_id(workspace_id)
    metadata = snapshot.header.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ArtifactValidationError('Invalid artifact manifest metadata')
    metadata_fields = cast(dict[object, object], metadata)
    if 'artifact_manifest' not in metadata_fields:
        return ()
    manifest = metadata_fields['artifact_manifest']
    if not isinstance(manifest, dict):
        raise ArtifactValidationError('Invalid artifact manifest schema')
    manifest_fields = cast(dict[object, object], manifest)
    if (set(manifest_fields) != {'schema_version', 'descriptors'}
            or type(manifest_fields['schema_version']) is not int or manifest_fields['schema_version'] != 1):
        raise ArtifactValidationError('Invalid artifact manifest schema')
    raw_descriptors = manifest_fields['descriptors']
    if not isinstance(raw_descriptors, dict):
        raise ArtifactValidationError('Invalid artifact manifest schema')
    descriptors: list[ArtifactDescriptor] = []
    bindings: set[tuple[str, int | None]] = set()
    for artifact_id, raw in cast(dict[object, object], raw_descriptors).items():
        descriptor = parse_artifact_descriptor(raw)
        if artifact_id != descriptor.artifact_id or descriptor.idempotency_key in bindings:
            raise ArtifactValidationError('Conflicting artifact manifest identity')
        bound_artifact_text(descriptor, snapshot, workspace_id=workspace_id)
        bindings.add(descriptor.idempotency_key)
        descriptors.append(descriptor)
    return tuple(descriptors)


def _declared_tool_name(snapshot: SourceSnapshot, execution_id: str) -> str:
    """Find the declared call at this exact trusted execution position."""
    for event in snapshot.events:
        if event.get('role') != 'assistant' or 'tool_calls' not in event:
            continue
        execution_ids = event.get('_execution_ids')
        if not isinstance(execution_ids, list):
            continue
        ids = cast(list[object], execution_ids)
        try:
            index = ids.index(execution_id)
        except ValueError:
            continue
        calls = event.get('tool_calls')
        if not isinstance(calls, list):
            break
        call_items = cast(list[object], calls)
        if index >= len(call_items):
            break
        call = call_items[index]
        function = cast(dict[object, object], call).get('function') if isinstance(call, dict) else None
        name = cast(dict[object, object], function).get('name') if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            return name
        break
    raise ArtifactValidationError('Artifact lacks matching producer evidence')


def attested_artifact_origin(descriptor: ArtifactDescriptor, snapshot: SourceSnapshot, *,
                             workspace_id: str) -> FilesystemReadReceipt:
    """Require producer evidence in addition to source text; still not a read grant.

    The returned receipt's root binding must match current policy during retrieval.
    Legacy inline events without producer receipts cannot be retroactively attested.
    """
    from nanobot.agent.tools.read_provenance import (
        canonical_tool_name_for_producer,
        parse_read_receipt,
    )

    text = bound_artifact_text(descriptor, snapshot, workspace_id=workspace_id).decode('utf-8')
    event = next(row for row in snapshot.events if row['_event_id'] == descriptor.event_id)
    receipt = parse_read_receipt(event.get('_source_origin'))
    declared_name = _declared_tool_name(snapshot, descriptor.execution_id)
    if (event.get('name') != declared_name
            or canonical_tool_name_for_producer(receipt.producer) != declared_name
            or event.get('_dispatched') is not True or event.get('_outcome_status') != 'ok'
            or descriptor.block_index is not None or receipt.source_access != descriptor.source_access
            or not receipt.matches(text)):
        raise ArtifactValidationError('Artifact lacks matching producer evidence')
    return receipt
