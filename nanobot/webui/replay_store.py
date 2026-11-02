"""Durable, bounded replay records for product WebSocket mutations (M12/M38).

The product gateway previously kept mutation results only in memory. A gateway
restart between an effect and its acknowledgement could therefore re-execute a
``replay: submit`` mutation. This store records ``in_progress`` before the
effect and ``completed`` before delivery, so a restarted process returns the
cached result or denies an uncertain replay instead of repeating the effect.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, cast

_MAX_ENTRIES = 256
_TTL_SECONDS = 60 * 60
_IN_PROGRESS_TTL_SECONDS = 24 * 60 * 60
_UNCERTAIN_TTL_SECONDS = 7 * 24 * 60 * 60
_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def result_to_dict(result: Any) -> dict[str, Any]:
    """Serialize one WebUIRequestResult without importing the router."""
    return {
        "status": getattr(result, "status", None),
        "message": getattr(result, "message", None),
        "result": getattr(result, "result", None),
        "workspace_scope": getattr(result, "workspace_scope", None),
        "workspace_permissions": getattr(result, "workspace_permissions", None),
    }


class DurableWebUIReplay:
    """One process-safe JSON replay ledger for a gateway instance."""

    def __init__(
        self,
        path: Path,
        *,
        max_entries: int = _MAX_ENTRIES,
        ttl_s: int = _TTL_SECONDS,
        in_progress_ttl_s: int = _IN_PROGRESS_TTL_SECONDS,
        uncertain_ttl_s: int = _UNCERTAIN_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least one")
        if ttl_s < 1:
            raise ValueError("ttl_s must be positive")
        if in_progress_ttl_s < 1 or uncertain_ttl_s < 1:
            raise ValueError("replay liveness TTLs must be positive")
        self.path = Path(path)
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.in_progress_ttl_s = in_progress_ttl_s
        self.uncertain_ttl_s = uncertain_ttl_s
        self.clock = clock
        self._lock = threading.RLock()
        self._corrupt = False
        self._entries: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError:
            self._corrupt = True
            return {}
        try:
            raw_value: Any = json.loads(text)
        except ValueError:
            self._corrupt = True
            return {}
        if not isinstance(raw_value, dict):
            self._corrupt = True
            return {}
        raw = cast(dict[str, Any], raw_value)
        entries_value = raw.get("entries")
        if not isinstance(entries_value, dict):
            self._corrupt = True
            return {}
        entries = cast(dict[str, Any], entries_value)
        clean: dict[str, dict[str, Any]] = {}
        for key, value in entries.items():
            if isinstance(value, dict):
                clean[key] = value
        return clean

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        payload = json.dumps(
            {"version": _VERSION, "entries": self._entries},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)
        directory = os.open(
            self.path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def lookup(
        self,
        request_id: str,
        action: str,
        digest: str,
        source_binding: dict[str, object] | None,
    ) -> tuple[str, dict[str, Any] | None]:
        with self._lock:
            if self._corrupt:
                return "uncertain", None
            if self._prune_locked():
                self._save()
            entry = self._entries.get(request_id)
            if entry is None:
                return "new", None
            if (
                entry.get("action") != action
                or entry.get("digest") != digest
                or entry.get("source_binding") != _canonical(source_binding)
            ):
                return "mismatch", None
            state = entry.get("state")
            if state == "completed":
                return "completed", entry
            return "uncertain", None

    def begin(
        self,
        request_id: str,
        action: str,
        digest: str,
        source_binding: dict[str, object] | None,
    ) -> bool:
        with self._lock:
            if self._corrupt:
                return False
            if request_id in self._entries:
                return False
            if len(self._entries) >= self.max_entries:
                if self._prune_locked(make_room=True):
                    self._save()
                if len(self._entries) >= self.max_entries:
                    return False
            self._entries[request_id] = {
                "action": action,
                "digest": digest,
                "source_binding": _canonical(source_binding),
                "state": "in_progress",
                "updated_at": self.clock(),
            }
            self._prune_locked()
            self._save()
            return True

    def complete(self, request_id: str, result: Any) -> None:
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            entry["state"] = "completed"
            entry["result"] = result_to_dict(result)
            entry["updated_at"] = self.clock()
            self._prune_locked()
            self._save()

    def uncertain(self, request_id: str) -> None:
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            entry["state"] = "uncertain"
            entry["updated_at"] = self.clock()
            self._prune_locked()
            self._save()

    def _prune_locked(self, *, make_room: bool = False) -> bool:
        """Expire results and age stale crash windows without dropping fresh ones.

        ``in_progress`` entries older than 24 hours are converted to ``uncertain``
        (the effect may have run). ``uncertain`` entries expire after 7 days, which
        defines the documented at-most-once window; after that a new request may
        run again. Fresh ``in_progress`` entries are never evicted at capacity, so
        a crash loop fails closed until the age policy releases them.
        """
        now = self.clock()
        changed = False
        for request_id, entry in list(self._entries.items()):
            state = entry.get("state")
            age = now - float(entry.get("updated_at", now))
            if state == "completed" and age > self.ttl_s:
                self._entries.pop(request_id, None)
                changed = True
            elif state == "in_progress" and age > self.in_progress_ttl_s:
                entry["state"] = "uncertain"
                entry["updated_at"] = now
                changed = True
            elif state == "uncertain" and age > self.uncertain_ttl_s:
                self._entries.pop(request_id, None)
                changed = True
        overflow = len(self._entries) - self.max_entries + (1 if make_room else 0)
        if overflow <= 0:
            return changed
        ordered = sorted(
            self._entries.items(),
            key=lambda item: float(item[1].get("updated_at", 0)),
        )
        completed = [
            request_id for request_id, entry in ordered if entry.get("state") == "completed"
        ]
        for request_id in completed[:overflow]:
            self._entries.pop(request_id, None)
            changed = True
        return changed
