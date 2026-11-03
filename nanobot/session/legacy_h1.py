"""Read-only protection for legacy H1 busy-input state.

The H1 coordinator is retired, but a session may still contain accepted or
uncertain work written by an older release. This module deliberately contains no
writer, replay, migration, or delivery path. It only recognises that state so
normal turns and destructive lifecycle operations fail closed.
"""

from __future__ import annotations

from typing import TypedDict, cast

from nanobot.session.persistence_metadata import BUSY_INBOX_METADATA_KEY

_TERMINAL_ENTRY_STATES = frozenset({"completed", "cancelled"})
_ENTRY_STATES = frozenset({
    "pending-routing", "awaiting-clarification", "next-step", "next-turn", "control",
    "claimed", "completed", "cancelled", "uncertain",
})
_RESULT_STATES = frozenset({"sending", "delivered", "definitely-unsent", "uncertain"})
_NOTICE_STATES = frozenset({"pending", "sending", "uncertain"})


class _LegacyEntry(TypedDict):
    state: str
    result: dict[str, object] | None


class _LegacyNotice(TypedDict):
    state: str


class _LegacyState(TypedDict):
    schema_version: int
    owner_id: str
    session_key: str
    temporary_lifetime_id: str | None
    closed: bool
    revision: int
    next_seq: int
    entries: list[_LegacyEntry]
    notices: list[_LegacyNotice]


class LegacyH1StateError(RuntimeError):
    """Legacy H1 metadata is unreadable and needs an explicit recovery decision."""


class LegacyH1RecoveryRequiredError(RuntimeError):
    """Legacy H1 work is not settled and must not be overwritten or replayed."""


def _malformed() -> LegacyH1StateError:
    return LegacyH1StateError(
        "This session contains malformed legacy H1 busy-input metadata. "
        "Inspect it and complete an explicit legacy H1 recovery before changing the session."
    )


def _unsettled() -> LegacyH1RecoveryRequiredError:
    return LegacyH1RecoveryRequiredError(
        "This session contains unsettled legacy H1 busy-input work. "
        "Normal turns and destructive reset/replacement/deletion are blocked until "
        "that state is recovered. Read-only inspection remains available."
    )


def _required_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _malformed()
    return value


def _parse_entries(value: object) -> list[_LegacyEntry]:
    if not isinstance(value, list):
        raise _malformed()
    entries: list[_LegacyEntry] = []
    for raw_entry in cast(list[object], value):
        if not isinstance(raw_entry, dict):
            raise _malformed()
        entry = cast(dict[str, object], raw_entry)
        state = entry.get("state")
        if not isinstance(state, str) or state not in _ENTRY_STATES:
            raise _malformed()
        raw_result = entry.get("result")
        result: dict[str, object] | None = None
        if raw_result is not None:
            if not isinstance(raw_result, dict):
                raise _malformed()
            result = cast(dict[str, object], raw_result)
            disposition = result.get("disposition")
            if not isinstance(disposition, str) or disposition not in _RESULT_STATES:
                raise _malformed()
        entries.append({"state": state, "result": result})
    return entries


def _parse_notices(value: object) -> list[_LegacyNotice]:
    if not isinstance(value, list):
        raise _malformed()
    notices: list[_LegacyNotice] = []
    for raw_notice in cast(list[object], value):
        if not isinstance(raw_notice, dict):
            raise _malformed()
        notice = cast(dict[str, object], raw_notice)
        state = notice.get("state", "pending")
        if not isinstance(state, str) or state not in _NOTICE_STATES:
            raise _malformed()
        notices.append({"state": state})
    return notices


def _parse_legacy_state(raw: object) -> _LegacyState:
    if not isinstance(raw, dict):
        raise _malformed()
    state = cast(dict[str, object], raw)
    if state.get("schema_version") != 1:
        raise _malformed()
    owner_id = _required_string(state.get("owner_id"))
    session_key = _required_string(state.get("session_key"))
    revision = state.get("revision")
    next_seq = state.get("next_seq")
    if type(revision) is not int or type(next_seq) is not int:
        raise _malformed()
    temporary_lifetime = state.get("temporary_lifetime_id")
    if temporary_lifetime is not None and (
        not isinstance(temporary_lifetime, str) or not temporary_lifetime
    ):
        raise _malformed()
    closed = state.get("closed", False)
    if type(closed) is not bool:
        raise _malformed()
    entries = _parse_entries(state.get("entries"))
    notices = _parse_notices(state.get("notices", []))
    emitted_notice_ids = state.get("emitted_notice_ids", [])
    if not isinstance(emitted_notice_ids, list):
        raise _malformed()
    if closed and (temporary_lifetime is None or entries or notices or emitted_notice_ids):
        raise _malformed()
    return {
        "schema_version": 1,
        "owner_id": owner_id,
        "session_key": session_key,
        "temporary_lifetime_id": temporary_lifetime,
        "closed": closed,
        "revision": revision,
        "next_seq": next_seq,
        "entries": entries,
        "notices": notices,
    }


def legacy_h1_is_unsettled(raw: object) -> bool:
    """Return whether parsed legacy state still owns work that needs recovery."""
    state = _parse_legacy_state(raw)
    if state["temporary_lifetime_id"] is not None and not state["closed"]:
        return True
    if state["notices"]:
        return True
    if state["closed"]:
        return bool(state["entries"] or state["notices"])
    for entry in state["entries"]:
        if entry["state"] not in _TERMINAL_ENTRY_STATES:
            return True
        result = entry["result"]
        if result is not None and result["disposition"] in {"sending", "uncertain"}:
            return True
    return False


def require_legacy_h1_disposition(metadata: object) -> None:
    """Refuse normal admission or destructive changes for unsettled legacy H1 state."""
    if not isinstance(metadata, dict):
        raise _malformed()
    raw = cast(dict[str, object], metadata).get(BUSY_INBOX_METADATA_KEY)
    if raw is None:
        return
    if legacy_h1_is_unsettled(raw):
        raise _unsettled()


__all__ = [
    "LegacyH1RecoveryRequiredError",
    "LegacyH1StateError",
    "legacy_h1_is_unsettled",
    "require_legacy_h1_disposition",
]
