"""Rebuildable bounded canonical event-offset index for JSONL session files.

Maps canonical ``_event_id`` strings to byte ranges relative to the canonical
file's ``body_offset`` (right after line 0's header). Every index entry is
bound to both the logical ``SourceState`` origin and the physical
``PrivateFileVersion`` (device, inode, size, mtime_ns, ...), so any whole-file
atomic replacement invalidates stale offsets fail-closed.

Reads open the session file once, verify the physical version and bounded
header origin, and issue ``os.pread`` calls for only the requested event IDs
without invoking ``load_source_snapshot`` or decoding the full file body.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kohaku.session.mode_storage import (
    MODE_JSON_LIMIT,
    PrivateFileVersion,
    private_file_version,
)

if TYPE_CHECKING:
    from kohaku.session.manager import JsonlSessionStore
    from kohaku.session.source_store import SourceState

_INDEX_VERSION = 1
_INDEX_DIR_NAME = ".event-index"
_MAX_INDEX_BYTES = 16 * 1024 * 1024  # 16 MiB safety bound


def event_index_dir(backend: JsonlSessionStore) -> Path:
    return backend.sessions_dir / _INDEX_DIR_NAME


def event_index_path(backend: JsonlSessionStore, key: str) -> Path:
    return event_index_dir(backend) / f"{backend.storage_key(key)}.json"


def _origin_dict(state: SourceState) -> dict[str, Any]:
    return {
        "store_identity": state.store_identity,
        "session_key": state.session_key,
        "incarnation": state.incarnation,
        "generation": state.generation,
    }


def _origin_matches(raw: object, expected: SourceState) -> bool:
    if not isinstance(raw, dict):
        return False
    data = cast(dict[str, Any], raw)
    return (
        data.get("store_identity") == expected.store_identity
        and data.get("session_key") == expected.session_key
        and data.get("incarnation") == expected.incarnation
        and type(data.get("generation")) is int
        and data.get("generation") == expected.generation
    )


def _version_tuple_from_json(raw: object) -> PrivateFileVersion | None:
    if not isinstance(raw, list):
        return None
    items = cast(list[object], raw)
    if len(items) != 9 or not all(type(item) is int for item in items):
        return None
    return PrivateFileVersion(*cast(list[int], items))


def build_event_offsets_from_bytes(
    raw: bytes,
    *,
    body_offset: int,
) -> dict[str, tuple[int, int]]:
    """Scan JSONL event lines starting at ``body_offset`` into relative byte ranges."""
    if body_offset < 0 or body_offset > len(raw):
        return {}
    offsets: dict[str, tuple[int, int]] = {}
    pos = body_offset
    total = len(raw)
    while pos < total:
        nl = raw.find(b"\n", pos)
        end = total if nl < 0 else nl
        line = raw[pos:end]
        if line:
            try:
                obj: object = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                obj = None
            if isinstance(obj, dict):
                obj_dict = cast(dict[str, Any], obj)
                event_id = obj_dict.get("_event_id")
                if isinstance(event_id, str) and event_id:
                    offsets[event_id] = (pos - body_offset, end - pos)
        if nl < 0:
            break
        pos = nl + 1
    return offsets


def _scan_fd_event_offsets(
    fd: int,
    *,
    file_size: int,
    body_offset: int,
) -> dict[str, tuple[int, int]]:
    """Stream-scan an open canonical fd from ``body_offset`` without loading full snapshot."""
    if body_offset < 0 or body_offset > file_size:
        return {}
    offsets: dict[str, tuple[int, int]] = {}
    chunk_size = 65536
    pos = body_offset
    buf = b""
    line_start = body_offset

    while pos < file_size:
        chunk = os.pread(fd, min(chunk_size, file_size - pos), pos)
        if not chunk:
            break
        pos += len(chunk)
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = buf[:nl]
            if line:
                try:
                    obj: object = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    obj = None
                if isinstance(obj, dict):
                    obj_dict = cast(dict[str, Any], obj)
                    event_id = obj_dict.get("_event_id")
                    if isinstance(event_id, str) and event_id:
                        offsets[event_id] = (line_start - body_offset, nl)
            line_start += nl + 1
            buf = buf[nl + 1 :]

    if buf:
        try:
            obj_tail: object = json.loads(buf.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            obj_tail = None
        if isinstance(obj_tail, dict):
            obj_dict = cast(dict[str, Any], obj_tail)
            event_id = obj_dict.get("_event_id")
            if isinstance(event_id, str) and event_id:
                offsets[event_id] = (line_start - body_offset, len(buf))
    return offsets


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with open(tmp, "xb") as handle:
            os.chmod(tmp, 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def write_event_index(
    backend: JsonlSessionStore,
    key: str,
    *,
    state: SourceState,
    file_version: PrivateFileVersion,
    body_offset: int,
    events: Mapping[str, tuple[int, int]],
) -> None:
    """Persist the rebuildable event-offset map bound to ``state`` and ``file_version``."""
    path = event_index_path(backend, key)
    payload: dict[str, Any] = {
        "version": _INDEX_VERSION,
        "origin": _origin_dict(state),
        "file_version": list(file_version),
        "body_offset": body_offset,
        "events": {eid: [rel, length] for eid, (rel, length) in events.items()},
    }
    try:
        _atomic_write_json(path, payload)
    except OSError:
        path.unlink(missing_ok=True)


def _read_index_payload(path: Path) -> dict[str, Any] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size <= 0 or st.st_size > _MAX_INDEX_BYTES:
        return None
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    parsed = cast(dict[str, Any], data)
    if parsed.get("version") != _INDEX_VERSION:
        return None
    return parsed


def update_event_index_header(
    backend: JsonlSessionStore,
    key: str,
    *,
    state: SourceState,
    old_version: PrivateFileVersion,
    new_version: PrivateFileVersion,
    new_body_offset: int,
) -> None:
    """Update only ``file_version`` and ``body_offset`` after a header-only atomic replace."""
    path = event_index_path(backend, key)
    data = _read_index_payload(path)
    if data is None:
        return
    if not _origin_matches(data.get("origin"), state):
        path.unlink(missing_ok=True)
        return
    if _version_tuple_from_json(data.get("file_version")) != old_version:
        path.unlink(missing_ok=True)
        return
    data["file_version"] = list(new_version)
    data["body_offset"] = new_body_offset
    try:
        _atomic_write_json(path, data)
    except OSError:
        path.unlink(missing_ok=True)


def delete_event_index(backend: JsonlSessionStore, key: str) -> None:
    try:
        event_index_path(backend, key).unlink(missing_ok=True)
    except OSError:
        pass


def _read_bounded_header_from_fd(fd: int, *, file_size: int) -> tuple[dict[str, Any], int] | None:
    """Read only line 0 from an open canonical fd; return ``(header_dict, body_offset)``."""
    read_len = min(file_size, MODE_JSON_LIMIT + 1)
    if read_len <= 0:
        return None
    head_bytes = os.pread(fd, read_len, 0)
    nl = head_bytes.find(b"\n")
    if nl < 0 or nl > MODE_JSON_LIMIT:
        return None
    try:
        header: object = json.loads(head_bytes[:nl].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    return cast(dict[str, Any], header), nl + 1


def _header_matches_origin(
    header: Mapping[str, Any],
    *,
    store_identity: str,
    expected: SourceState,
) -> bool:
    if header.get("_type") != "metadata":
        return False
    if header.get("key") != expected.session_key:
        return False
    if store_identity != expected.store_identity:
        return False
    if header.get("incarnation") != expected.incarnation:
        return False
    generation = header.get("generation")
    return type(generation) is int and generation == expected.generation


def _parse_event_map(raw: object) -> dict[str, tuple[int, int]] | None:
    if not isinstance(raw, dict):
        return None
    raw_dict = cast(dict[object, object], raw)
    result: dict[str, tuple[int, int]] = {}
    for eid, entry in raw_dict.items():
        if not isinstance(eid, str) or not eid or not isinstance(entry, list):
            return None
        pair = cast(list[object], entry)
        if (
            len(pair) != 2
            or type(pair[0]) is not int
            or type(pair[1]) is not int
            or pair[0] < 0
            or pair[1] <= 0
        ):
            return None
        result[eid] = (pair[0], pair[1])
    return result


def read_canonical_events_by_ids(
    backend: JsonlSessionStore,
    key: str,
    event_ids: Sequence[str],
    *,
    expected_source: SourceState,
) -> dict[str, dict[str, Any]]:
    """Bounded ``os.pread`` lookup of specific canonical events by ``_event_id``.

    Verifies physical ``PrivateFileVersion`` and logical ``SourceState`` origin
    fail-closed. Rebuilds the rebuildable ``.event-index`` sidecar lazily if
    missing or stale, without ever invoking ``load_source_snapshot``.
    """
    unique_ids = list(dict.fromkeys(eid for eid in event_ids if eid))
    if not unique_ids:
        return {}
    if expected_source.session_key != key:
        return {}

    session_path = backend.get_session_path(key)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(session_path, flags)
    except OSError:
        return {}

    try:
        try:
            st = os.fstat(fd)
        except OSError:
            return {}
        actual_version = private_file_version(st)
        if actual_version is None:
            return {}

        header_info = _read_bounded_header_from_fd(fd, file_size=actual_version.size)
        if header_info is None:
            return {}
        header, body_offset = header_info
        store_identity = hashlib.sha256(
            str(backend.sessions_dir.resolve()).encode("utf-8")
        ).hexdigest()
        if not _header_matches_origin(
            header,
            store_identity=store_identity,
            expected=expected_source,
        ):
            return {}

        idx_path = event_index_path(backend, key)
        event_map: dict[str, tuple[int, int]] | None = None
        index_body_offset = body_offset
        data = _read_index_payload(idx_path)
        if (
            data is not None
            and _origin_matches(data.get("origin"), expected_source)
            and _version_tuple_from_json(data.get("file_version")) == actual_version
            and type(data.get("body_offset")) is int
            and data["body_offset"] == body_offset
        ):
            event_map = _parse_event_map(data.get("events"))

        if event_map is None:
            event_map = _scan_fd_event_offsets(
                fd,
                file_size=actual_version.size,
                body_offset=body_offset,
            )
            write_event_index(
                backend,
                key,
                state=expected_source,
                file_version=actual_version,
                body_offset=body_offset,
                events=event_map,
            )

        out: dict[str, dict[str, Any]] = {}
        for eid in unique_ids:
            loc = event_map.get(eid)
            if loc is None:
                continue
            rel_offset, byte_len = loc
            abs_offset = index_body_offset + rel_offset
            if abs_offset < body_offset or abs_offset + byte_len > actual_version.size:
                continue
            try:
                raw_line = os.pread(fd, byte_len, abs_offset)
                if len(raw_line) != byte_len:
                    continue
                obj: object = json.loads(raw_line.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(obj, dict):
                obj_dict = cast(dict[str, Any], obj)
                if obj_dict.get("_event_id") == eid:
                    out[eid] = obj_dict
        return out
    finally:
        os.close(fd)
