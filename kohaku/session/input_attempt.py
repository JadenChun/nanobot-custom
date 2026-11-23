"""Content-free, v1 evidence for the existing canonical input owner.

A lookup is positive-only. It supplies no execution, permission, or replay authority.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from kohaku.security.desktop_permissions import permission_source_token
from kohaku.workspace_access_recovery import parse_workspace_access_source_binding

if TYPE_CHECKING:
    from kohaku.session.source_store import SourceState

INPUT_ATTEMPT_METADATA_KEY = '_webui_input_attempt'
_TURN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')
_HEX = re.compile(r'[0-9a-f]{64}')
_TOKEN = re.compile(r'[0-9a-f]{32}')


class InputAttemptConflictError(ValueError):
    """Unsupported, ambiguous or conflicting evidence; never a replay grant."""


def _object(value: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InputAttemptConflictError('Invalid input attempt')
    raw = cast(dict[object, object], value)
    if set(raw) != keys or getattr(raw, 'duplicate_keys', ()):
        raise InputAttemptConflictError('Invalid input attempt')
    return cast(dict[str, object], raw)  # Exact key equality established string keys.


def turn_identity(value: object) -> str:
    if not isinstance(value, str) or _TURN.fullmatch(value) is None:
        raise InputAttemptConflictError('Invalid input attempt identity')
    return value


def _hex(value: object) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise InputAttemptConflictError('Invalid input attempt digest')
    return value


def fingerprint(text: str) -> str:
    try:
        return hashlib.sha256(b'nanobot.input-attempt.v1\0new\0' + text.encode('utf-8')).hexdigest()
    except UnicodeError:
        raise InputAttemptConflictError('Invalid input attempt text') from None


def principal_tag(identity: str) -> str:
    return hashlib.sha256(b'nanobot.input-principal.v1\0' + identity.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class InputAttemptSource:
    store_identity: str
    session_key: str
    incarnation: str
    generation: int

    @classmethod
    def parse(cls, value: object) -> InputAttemptSource:
        raw = _object(value, {'store_identity', 'session_key', 'incarnation', 'generation'})
        try:
            binding = parse_workspace_access_source_binding({
                key: raw[key] for key in ('store_identity', 'incarnation', 'generation')
            })
        except (TypeError, ValueError):
            raise InputAttemptConflictError('Invalid input attempt source') from None
        key = raw['session_key']
        if (not isinstance(key, str) or not 1 <= len(key) <= 256
                or any(ord(char) < 32 or ord(char) == 127 for char in key)):
            raise InputAttemptConflictError('Invalid input attempt source')
        return cls(binding['store_identity'], key, binding['incarnation'], binding['generation'])

    @classmethod
    def from_state(cls, state: SourceState) -> InputAttemptSource:
        return cls(state.store_identity, state.session_key, state.incarnation, state.generation)

    def to_dict(self) -> dict[str, object]:
        return {'store_identity': self.store_identity, 'session_key': self.session_key,
                'incarnation': self.incarnation, 'generation': self.generation}

    @property
    def token(self) -> str:
        return permission_source_token({'store_identity': self.store_identity,
                                        'incarnation': self.incarnation, 'generation': self.generation})

    def expected_state(self) -> SourceState:
        from kohaku.session.source_store import SourceState
        # Only origin is compared by the existing expected-source admission fence.
        return SourceState(self.store_identity, self.session_key, self.incarnation,
                           0, self.generation, None, '')


@dataclass(frozen=True)
class InputAttempt:
    source: InputAttemptSource
    turn_id: str
    fingerprint: str
    principal: str

    @classmethod
    def parse_query(cls, value: object, *, principal: str, session_key: str) -> InputAttempt:
        raw = _object(value, {'version', 'turn_id', 'source', 'source_token', 'fingerprint'})
        if type(raw['version']) is not int or raw['version'] != 1:
            raise InputAttemptConflictError('Unsupported input attempt')
        source = InputAttemptSource.parse(raw['source'])
        token = raw['source_token']
        if (source.session_key != session_key or not isinstance(token, str)
                or _TOKEN.fullmatch(token) is None or token != source.token):
            raise InputAttemptConflictError('Invalid input attempt source')
        return cls(source, turn_identity(raw['turn_id']), _hex(raw['fingerprint']), _hex(principal))

    @classmethod
    def parse_message(cls, value: object, *, principal: str, session_key: str,
                      turn_id: object, text: str) -> InputAttempt:
        raw = _object(value, {'version', 'source', 'source_token', 'fingerprint'})
        attempt = cls.parse_query({**raw, 'turn_id': turn_id}, principal=principal,
                                  session_key=session_key)
        if attempt.fingerprint != fingerprint(text):
            raise InputAttemptConflictError('Input attempt payload changed')
        return attempt

    @classmethod
    def parse_evidence(cls, value: object) -> InputAttempt:
        raw = _object(value, {'version', 'turn_id', 'source', 'fingerprint', 'principal', 'intent'})
        if type(raw['version']) is not int or raw['version'] != 1 or raw['intent'] != 'new':
            raise InputAttemptConflictError('Unsupported input attempt evidence')
        return cls(InputAttemptSource.parse(raw['source']), turn_identity(raw['turn_id']),
                   _hex(raw['fingerprint']), _hex(raw['principal']))

    def evidence(self) -> dict[str, object]:
        return {'version': 1, 'turn_id': self.turn_id, 'source': self.source.to_dict(),
                'fingerprint': self.fingerprint, 'principal': self.principal, 'intent': 'new'}

    def query(self) -> dict[str, object]:
        return {'version': 1, 'turn_id': self.turn_id, 'source': self.source.to_dict(),
                'source_token': self.source.token, 'fingerprint': self.fingerprint}


@dataclass(frozen=True)
class InputAttemptProof:
    attempt: InputAttempt
    event_id: str
    run_id: str

    def to_dict(self) -> dict[str, object]:
        return {**self.attempt.query(), 'principal': self.attempt.principal,
                'event_id': self.event_id, 'run_id': self.run_id}


@dataclass(frozen=True)
class InputAttemptAdmission:
    disposition: Literal['committed', 'joined']
    proof: InputAttemptProof


def find_input_attempt(events: Sequence[Mapping[str, object]], attempt: InputAttempt) -> InputAttemptProof | None:
    """Scan full raw source; any ambiguous/unsupported evidence fails closed."""
    found: InputAttemptProof | None = None
    for event in events:
        if '_input_attempt' not in event:
            continue
        saved = InputAttempt.parse_evidence(event['_input_attempt'])
        if (saved.source, saved.turn_id) != (attempt.source, attempt.turn_id):
            continue
        content = event.get('content')
        if (saved != attempt or event.get('role') != 'user' or found is not None
                or not isinstance(content, str) or fingerprint(content) != saved.fingerprint):
            raise InputAttemptConflictError('Conflicting input attempt evidence')
        found = InputAttemptProof(saved, turn_identity(event.get('_event_id')),
                                  turn_identity(event.get('_run_id')))
    return found
