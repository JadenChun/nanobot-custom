"""Bounded next-turn preferences kept beside, not inside, settled session history.

Routine settings choices (next working folder, next model preset + effort, next
request permission mode) must neither scale with settled canonical history nor
disturb the admitted turn. They live in one small private record per source
incarnation under ``.session-prefs/``; the canonical source header (the
*active*/confirmed settings, ``SourceState`` and its ``revision``) is never
rewritten by a routine preference write.

A record is bound to the source ``(store_identity, incarnation, generation)``.
Reset, replacement or recreation of the source therefore orphans it; it is then
ignored and replaced, never applied to the new origin.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from kohaku.session.manager import JsonlSessionStore

PREFERENCE_DIRECTORY = ".session-prefs"
NEXT_TURN_PREFERENCE_KEYS = frozenset(
    {"workspace_scope", "model_preset", "reasoning_effort", "permission_mode"}
)
MAX_PREFERENCE_BYTES = 16_384
MAX_HEADER_BYTES = 262_144
_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class NextTurnPreferences:
    """Detached next-turn choices for one source incarnation."""

    store_identity: str
    incarnation: str
    generation: int
    revision: int
    values: Mapping[str, Any]
    updated_at: str

    def get(self, name: str) -> Any:
        return deepcopy(self.values.get(name))

    def payload(self) -> dict[str, Any]:
        return {"revision": self.revision, **{k: deepcopy(v) for k, v in self.values.items()}}


class NextTurnPreferenceError(ValueError):
    """Invalid preference input; never includes caller data."""

    def __init__(self) -> None:
        super().__init__("invalid next-turn preference")


def _identity(backend: JsonlSessionStore) -> str:
    # Same derivation as SourceSessionStore's content-free store identity.
    return hashlib.sha256(str(backend.sessions_dir.resolve()).encode()).hexdigest()


def _directory(backend: JsonlSessionStore) -> Path:
    return backend.sessions_dir / PREFERENCE_DIRECTORY


def preference_path(backend: JsonlSessionStore, key: str) -> Path:
    return _directory(backend) / f"{backend.storage_key(key)}.json"


def read_source_origin(backend: JsonlSessionStore, key: str) -> tuple[str, int] | None:
    """Return ``(incarnation, generation)`` from proof or the header line only (bounded)."""
    from kohaku.session.mode_storage import private_file_version

    path = backend.get_session_path(key)
    proof = backend.saved_source_proof(key)
    if (
        proof is not None
        and not backend.get_runtime_checkpoint_path(key).exists()
    ):
        try:
            current_version = private_file_version(path.lstat())
        except FileNotFoundError:
            return None
        if current_version is not None and current_version == proof.version:
            state = proof.metadata.state
            return state.incarnation, state.generation
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        line = b""
        while b"\n" not in line and len(line) <= MAX_HEADER_BYTES:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            line += chunk
    finally:
        os.close(fd)
    line = line[: line.index(b"\n") + 1] if b"\n" in line else line
    if len(line) > MAX_HEADER_BYTES or not line.endswith(b"\n"):
        return None
    try:
        header = json.loads(line)
    except ValueError:
        return None
    if not isinstance(header, dict):
        return None
    data = cast(dict[str, Any], header)
    metadata = data.get("metadata")
    incarnation = data.get("incarnation")
    generation = data.get("generation")
    if (
        data.get("key") != key
        or not isinstance(metadata, dict)
        or cast(dict[str, Any], metadata).get("source_schema_version") != 1
        or not isinstance(incarnation, str)
        or not incarnation
        or type(generation) is not int
    ):
        return None
    return incarnation, generation


def _read_record(
    backend: JsonlSessionStore, key: str, origin: tuple[str, int],
) -> NextTurnPreferences | None:
    path = preference_path(backend, key)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        raw = os.read(fd, MAX_PREFERENCE_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > MAX_PREFERENCE_BYTES:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    data = cast(dict[str, Any], record)
    values = data.get("values")
    revision = data.get("revision")
    if (
        data.get("schema") != _SCHEMA
        or data.get("key") != key
        or data.get("store_identity") != _identity(backend)
        or data.get("incarnation") != origin[0]
        or data.get("generation") != origin[1]
        or type(revision) is not int
        or revision < 1
        or not isinstance(values, dict)
        or not set(cast(dict[str, Any], values)) <= NEXT_TURN_PREFERENCE_KEYS
        or not isinstance(data.get("updated_at"), str)
    ):
        return None  # Stale origin or damaged record: ignore, never apply.
    return NextTurnPreferences(
        _identity(backend), origin[0], origin[1], revision,
        deepcopy(cast(dict[str, Any], values)), cast(str, data["updated_at"]),
    )


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno != errno.EINVAL:
            raise
    finally:
        os.close(fd)


def _ensure_directory(backend: JsonlSessionStore) -> Path:
    directory = _directory(backend)
    if directory.is_symlink():
        raise NextTurnPreferenceError()
    created = not directory.exists()
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    if created:
        _fsync_directory(backend.sessions_dir)
    return directory


def _write_record(
    backend: JsonlSessionStore, key: str, origin: tuple[str, int], revision: int,
    values: dict[str, Any],
) -> NextTurnPreferences:
    updated_at = datetime.now(timezone.utc).isoformat()
    record = {
        "schema": _SCHEMA, "key": key, "store_identity": _identity(backend),
        "incarnation": origin[0], "generation": origin[1], "revision": revision,
        "values": values, "updated_at": updated_at,
    }
    try:
        data = json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()
    except (TypeError, ValueError):
        raise NextTurnPreferenceError() from None
    if len(data) > MAX_PREFERENCE_BYTES:
        raise NextTurnPreferenceError()
    directory = _ensure_directory(backend)
    path = preference_path(backend, key)
    tmp = directory / f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_directory(directory)
    return NextTurnPreferences(
        _identity(backend), origin[0], origin[1], revision, deepcopy(values), updated_at,
    )


def remove_preferences(backend: JsonlSessionStore, key: str) -> None:
    """Drop the record for a deleted/reset source; absence is success."""
    path = preference_path(backend, key)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def read_preferences(backend: JsonlSessionStore, key: str) -> NextTurnPreferences | None:
    origin = read_source_origin(backend, key)
    if origin is None:
        return None
    return _read_record(backend, key, origin)


def update_preferences(
    backend: JsonlSessionStore,
    key: str,
    updates: Mapping[str, Any],
    *,
    removals: Collection[str] = (),
    expected_revision: int | None = None,
) -> NextTurnPreferences | None:
    """Merge choices into the record; ``None`` when the key has no canonical source.

    ``expected_revision`` makes the write conditional on the record revision
    (``0`` meaning no record), so a stale caller cannot overwrite a newer choice.
    """
    if (
        isinstance(removals, (str, bytes))
        or not set(updates) <= NEXT_TURN_PREFERENCE_KEYS
        or not set(removals) <= NEXT_TURN_PREFERENCE_KEYS
    ):
        raise NextTurnPreferenceError()
    origin = read_source_origin(backend, key)
    if origin is None:
        return None
    current = _read_record(backend, key, origin)
    revision = current.revision if current is not None else 0
    if expected_revision is not None and expected_revision != revision:
        from kohaku.session.manager import SessionWriteConflict

        raise SessionWriteConflict()
    values: dict[str, Any] = dict(current.values) if current is not None else {}
    for name in removals:
        values.pop(name, None)
    values.update(deepcopy(dict(updates)))
    if current is not None and values == dict(current.values):
        return current
    if not values:
        remove_preferences(backend, key)
        return NextTurnPreferences(
            _identity(backend), origin[0], origin[1], revision + 1, {},
            datetime.now(timezone.utc).isoformat(),
        )
    return _write_record(backend, key, origin, revision + 1, values)


def consume_preferences(
    backend: JsonlSessionStore, key: str, consumed: Mapping[str, Any],
) -> NextTurnPreferences | None:
    """Retire only choices equal to those a new-turn admission applied.

    A newer choice made after the admission read is left untouched, so a
    completing or concurrent turn can never erase a later preference.
    """
    origin = read_source_origin(backend, key)
    if origin is None:
        return None
    current = _read_record(backend, key, origin)
    if current is None:
        return None
    values = {k: v for k, v in current.values.items() if k not in consumed or consumed[k] != v}
    if len(values) == len(current.values):
        return current
    if not values:
        remove_preferences(backend, key)
        return None
    return _write_record(backend, key, origin, current.revision + 1, values)
