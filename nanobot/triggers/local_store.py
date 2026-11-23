"""Workspace-scoped local trigger store and delivery queue."""

from __future__ import annotations

import errno
import json
import os
import secrets
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

from filelock import FileLock
from loguru import logger

from nanobot.triggers.local_types import LocalTrigger, TriggerDelivery, TriggerRunRecord
from nanobot.utils.dict_keys import get_camel_snake as _get
from nanobot.utils.helpers import truncate_text
from nanobot.utils.run_records import write_run_record as write_automation_run_record

_TRIGGER_ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_MAX_RUN_HISTORY = 20
_MAX_DELIVERY_ATTEMPTS = 10
_RUN_RECORD_TEXT_MAX_CHARS = 4000
_PROCESSING_RECOVERY_ERROR = "delivery was recovered from interrupted processing"
_UNKNOWN_OUTCOME_ERROR = (
    "delivery was interrupted after its turn attempt started; "
    "the outcome is unknown and the owner must resolve it"
)


class TriggerStoreError(RuntimeError):
    """Base class for trigger store errors."""


class TriggerNotFoundError(TriggerStoreError):
    """Raised when a trigger ID does not exist."""


class TriggerDisabledError(TriggerStoreError):
    """Raised when a trigger is disabled."""


class LocalTriggerStore:
    """Persistent local triggers for one workspace."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path)
        self.root = self.workspace_path / "triggers"
        self.store_path = self.root / "triggers.json"
        self.inbox_dir = self.root / "inbox"
        self.processing_dir = self.root / "processing"
        self.unknown_dir = self.root / "unknown"
        self.failed_dir = self.root / "failed"
        self.runs_dir = self.root / "runs"
        self._lock = FileLock(str(self.root / ".lock"))
        self.delivery_active = False  # Process-local capability, owned by the gateway consumer.

    def create(
        self,
        *,
        name: str,
        channel: str,
        chat_id: str,
        session_key: str,
        sender_id: str = "trigger",
        origin_metadata: dict[str, Any] | None = None,
    ) -> LocalTrigger:
        """Create a new session-bound local trigger."""
        clean_name = _clean_name(name)
        channel = channel.strip()
        chat_id = chat_id.strip()
        session_key = session_key.strip()
        if not channel or not chat_id or not session_key:
            raise ValueError("channel, chat_id, and session_key are required")

        now = _now_ms()
        self._ensure_dirs()
        with self._lock:
            triggers = self._load_triggers_unlocked()
            existing_ids = {trigger.id for trigger in triggers}
            trigger_id = _new_trigger_id(existing_ids)
            trigger = LocalTrigger(
                id=trigger_id,
                name=clean_name,
                enabled=True,
                channel=channel,
                chat_id=chat_id,
                session_key=session_key,
                sender_id=sender_id.strip() or "trigger",
                origin_metadata=dict(origin_metadata or {}),
                created_at_ms=now,
                updated_at_ms=now,
            )
            triggers.append(trigger)
            self._save_triggers_unlocked(triggers)
            return trigger

    def list_triggers(self, *, include_disabled: bool = False) -> list[LocalTrigger]:
        """List triggers in this workspace."""
        self._ensure_dirs()
        with self._lock:
            triggers = self._load_triggers_unlocked()
        if not include_disabled:
            triggers = [trigger for trigger in triggers if trigger.enabled]
        return sorted(triggers, key=lambda trigger: (trigger.updated_at_ms, trigger.id), reverse=True)

    def list_for_session(
        self,
        session_key: str,
        *,
        include_disabled: bool = True,
    ) -> list[LocalTrigger]:
        """List triggers bound to one session key."""
        return [
            trigger
            for trigger in self.list_triggers(include_disabled=include_disabled)
            if trigger.session_key == session_key
        ]

    def get(self, trigger_id: str) -> LocalTrigger | None:
        """Return one trigger by ID."""
        self._ensure_dirs()
        with self._lock:
            return self._find_unlocked(self._load_triggers_unlocked(), trigger_id)

    def enable(self, trigger_id: str, *, enabled: bool) -> LocalTrigger | None:
        """Enable or disable a trigger."""
        self._ensure_dirs()
        with self._lock:
            triggers = self._load_triggers_unlocked()
            trigger = self._find_unlocked(triggers, trigger_id)
            if trigger is None:
                return None
            trigger.enabled = enabled
            trigger.updated_at_ms = _now_ms()
            self._save_triggers_unlocked(triggers)
            return trigger

    def update(self, trigger_id: str, *, name: str | None = None) -> LocalTrigger | None:
        """Update mutable trigger fields."""
        self._ensure_dirs()
        with self._lock:
            triggers = self._load_triggers_unlocked()
            trigger = self._find_unlocked(triggers, trigger_id)
            if trigger is None:
                return None
            if name is not None:
                trigger.name = _clean_name(name)
            trigger.updated_at_ms = _now_ms()
            self._save_triggers_unlocked(triggers)
            return trigger

    def delete(self, trigger_id: str) -> bool:
        """Delete a trigger by ID."""
        trigger_id = trigger_id.strip()
        self._ensure_dirs()
        with self._lock:
            triggers = self._load_triggers_unlocked()
            remaining = [trigger for trigger in triggers if trigger.id != trigger_id]
            if len(remaining) == len(triggers):
                return False
            self._save_triggers_unlocked(remaining)
            self._delete_delivery_files_for_trigger_unlocked(trigger_id)
            return True

    def enqueue(self, trigger_id: str, content: str) -> TriggerDelivery:
        """Queue a delivery for the gateway process to consume."""
        trigger_id = trigger_id.strip()
        if not content.strip():
            raise ValueError("trigger message is required")
        self._ensure_dirs()
        with self._lock:
            triggers = self._load_triggers_unlocked()
            trigger = self._find_unlocked(triggers, trigger_id)
            if trigger is None:
                raise TriggerNotFoundError(f"trigger not found: {trigger_id}")
            if not trigger.enabled:
                raise TriggerDisabledError(f"trigger is disabled: {trigger_id}")
            delivery = TriggerDelivery(
                id=f"tdl_{uuid.uuid4().hex[:12]}",
                trigger_id=trigger_id,
                content=content,
                created_at_ms=_now_ms(),
            )
            path = self.inbox_dir / f"{delivery.created_at_ms}-{delivery.id}.json"
            self._atomic_write(path, json.dumps(_delivery_payload(delivery), ensure_ascii=False))
            delivery.path = path
            run_record_path: Path | None = None
            try:
                run_record_path = self.write_delivery_run_record(
                    delivery,
                    trigger=trigger,
                    status="queued",
                )
                trigger.last_message = _run_record_text(content)
                trigger.updated_at_ms = delivery.created_at_ms
                self._save_triggers_unlocked(triggers)
            except BaseException:
                path.unlink(missing_ok=True)
                if run_record_path is not None:
                    run_record_path.unlink(missing_ok=True)
                delivery.path = None
                raise
            return delivery

    def claim_deliveries(self, *, limit: int = 20) -> list[TriggerDelivery]:
        """Move pending deliveries into processing and return them."""
        self._ensure_dirs()
        claimed: list[TriggerDelivery] = []
        with self._lock:
            for path in sorted(self.inbox_dir.glob("*.json"))[: max(0, limit)]:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    delivery = TriggerDelivery.from_dict(
                        data.get("delivery", data),
                        path=self.processing_dir / path.name,
                    )
                except Exception:
                    logger.exception("Trigger: failed to parse delivery {}", path)
                    self._move_bad_delivery_unlocked(path)
                    continue
                os.replace(path, cast(Path, delivery.path))
                claimed.append(delivery)
        return claimed

    def begin_delivery_attempt(self, delivery: TriggerDelivery) -> str:
        """Persist a settlement identity before the turn is submitted.

        The durable identity proves the attempt started. A restart uses it to
        classify the delivery as unknown instead of replaying it.
        """
        if delivery.path is None:
            raise TriggerStoreError("delivery has no processing path")
        self._ensure_dirs()
        settlement_id = f"stl_{uuid.uuid4().hex}"
        with self._lock:
            delivery.attempts += 1
            delivery.last_error = None
            self._atomic_write(
                delivery.path,
                json.dumps(
                    _delivery_payload(delivery, settlement_id=settlement_id),
                    ensure_ascii=False,
                ),
            )
        return settlement_id

    def mark_delivery_unknown(
        self,
        delivery: TriggerDelivery,
        *,
        settlement_id: str | None,
        error: str,
    ) -> bool:
        """Quarantine a claimed delivery whose turn outcome is not proven."""
        if delivery.path is None:
            return False
        self._ensure_dirs()
        with self._lock:
            target = self.unknown_dir / delivery.path.name
            delivery.last_error = error
            if delivery.path.exists():
                os.replace(delivery.path, target)
            delivery.path = target
            if settlement_id:
                self._atomic_write(
                    target,
                    json.dumps(
                        _delivery_payload(delivery, settlement_id=settlement_id),
                        ensure_ascii=False,
                    ),
                )
        self.write_delivery_run_record(
            delivery,
            status="unknown",
            error=error,
            settlement_id=settlement_id,
        )
        return True

    def recover_processing_deliveries(self) -> int:
        """Resolve deliveries left in processing by an interrupted gateway.

        A processing record with a persisted settlement identity may already
        have committed its turn, so it moves to the unknown set and is never
        replayed. Only a claim with no settlement identity proves that no turn
        started, and only that record returns to the inbox once.
        """
        self._ensure_dirs()
        requeued = 0
        unknown: list[tuple[TriggerDelivery, str | None]] = []
        with self._lock:
            for path in sorted(self.processing_dir.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    delivery = TriggerDelivery.from_dict(
                        data.get("delivery", data),
                        path=path,
                    )
                    settlement_id = _payload_settlement_id(data)
                except Exception:
                    logger.exception("Trigger: failed to parse processing delivery {}", path)
                    self._move_bad_delivery_unlocked(path)
                    continue
                if settlement_id:
                    target = self.unknown_dir / path.name
                    os.replace(path, target)
                    delivery.path = target
                    delivery.last_error = _UNKNOWN_OUTCOME_ERROR
                    unknown.append((delivery, settlement_id))
                elif self._requeue_unattempted_unlocked(delivery):
                    requeued += 1
        for delivery, settlement_id in unknown:
            try:
                self.write_delivery_run_record(
                    delivery,
                    status="unknown",
                    error=_UNKNOWN_OUTCOME_ERROR,
                    settlement_id=settlement_id,
                )
            except Exception:
                logger.exception(
                    "Trigger: failed to write unknown outcome for delivery {}",
                    delivery.id,
                )
        return requeued

    def list_unknown_deliveries(self) -> list[dict[str, Any]]:
        """Expose interrupted deliveries that await an explicit owner decision."""
        self._ensure_dirs()
        unresolved: list[dict[str, Any]] = []
        with self._lock:
            for path in sorted(self.unknown_dir.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    logger.exception("Trigger: failed to parse unknown delivery {}", path)
                    continue
                body = _delivery_body(data)
                delivery_id = str(body.get("id") or "")
                if not delivery_id:
                    continue
                record = self._read_run_record(delivery_id)
                settlement_id = _payload_settlement_id(data) or (
                    record.get("settlement_id") if isinstance(record.get("settlement_id"), str) else None
                )
                unresolved.append({
                    "delivery_id": delivery_id,
                    "trigger_id": str(_get(body, "triggerId", "trigger_id", "") or ""),
                    "attempts": int(body.get("attempts") or 0),
                    "settlement_id": settlement_id,
                    "status": "unknown",
                    "error": record.get("error") or _UNKNOWN_OUTCOME_ERROR,
                    "path": str(path),
                })
        return unresolved

    def resolve_unknown_delivery(
        self,
        delivery_id: str,
        *,
        action: Literal["requeue", "fail"] = "requeue",
    ) -> bool:
        """Apply one explicit owner decision to an unknown delivery."""
        if action not in {"requeue", "fail"}:
            raise ValueError(f"unsupported unknown delivery action: {action}")
        self._ensure_dirs()
        found: Path | None = None
        with self._lock:
            for path in sorted(self.unknown_dir.glob("*.json")):
                if self._delivery_file_id(path) == delivery_id:
                    found = path
                    break
            if found is None:
                return False
            try:
                data = json.loads(found.read_text(encoding="utf-8"))
                delivery = TriggerDelivery.from_dict(_delivery_body(data), path=found)
            except Exception:
                logger.exception("Trigger: failed to parse unknown delivery {}", found)
                return False
            if action == "requeue":
                delivery.attempts = 0
                delivery.last_error = None
                target = self.inbox_dir / found.name
                self._atomic_write(
                    target,
                    json.dumps(_delivery_payload(delivery), ensure_ascii=False),
                )
                found.unlink(missing_ok=True)
                delivery.path = target
            else:
                target = self.failed_dir / found.name
                delivery.last_error = "delivery was resolved as failed by the owner"
                os.replace(found, target)
                delivery.path = target
        self.write_delivery_run_record(
            delivery,
            status="requeued" if action == "requeue" else "error",
            error=None if action == "requeue" else delivery.last_error,
        )
        return True

    def complete_delivery(self, delivery: TriggerDelivery) -> None:
        """Delete a claimed delivery after it is handled."""
        if delivery.path is None:
            return
        self._ensure_dirs()
        with self._lock:
            delivery.path.unlink(missing_ok=True)

    def retry_delivery(self, delivery: TriggerDelivery, error: str) -> bool:
        """Retry a claimed delivery unless it exceeded the attempt limit."""
        if delivery.path is None:
            return False
        self._ensure_dirs()
        with self._lock:
            return self._retry_delivery_unlocked(delivery, error)

    def record_delivery(
        self,
        trigger_id: str,
        *,
        status: str,
        error: str | None = None,
        run_at_ms: int | None = None,
    ) -> None:
        """Record the latest delivery status on a trigger."""
        self._ensure_dirs()
        run_at_ms = run_at_ms or _now_ms()
        with self._lock:
            triggers = self._load_triggers_unlocked()
            trigger = self._find_unlocked(triggers, trigger_id)
            if trigger is None:
                return
            trigger.last_run_at_ms = run_at_ms
            trigger.last_status = "ok" if status == "ok" else "error"
            trigger.last_error = None if status == "ok" else (error or "delivery failed")
            trigger.updated_at_ms = _now_ms()
            trigger.run_history.append(
                TriggerRunRecord(
                    run_at_ms=run_at_ms,
                    status=trigger.last_status,
                    error=trigger.last_error,
                )
            )
            trigger.run_history = trigger.run_history[-_MAX_RUN_HISTORY:]
            self._save_triggers_unlocked(triggers)

    def write_run_record(self, run_id: str, record: dict[str, Any]) -> Path:
        """Write an internal audit record for one local trigger delivery."""
        self._ensure_dirs()
        return write_automation_run_record(self.runs_dir, run_id, record)

    def write_delivery_run_record(
        self,
        delivery: TriggerDelivery,
        *,
        status: str,
        trigger: LocalTrigger | None = None,
        error: str | None = None,
        response: str | None = None,
        settlement_id: str | None = None,
    ) -> Path:
        """Write the durable audit record for one local trigger delivery."""
        if trigger is None:
            trigger = self.get(delivery.trigger_id)
        record = _delivery_run_record(delivery, trigger)
        record["status"] = status
        if settlement_id:
            record["settlement_id"] = settlement_id
        if error:
            record["error"] = _run_record_text(error)
        if response is not None:
            record["response"] = _run_record_text(response)
        return self.write_run_record(delivery.id, record)

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _load_triggers_unlocked(self) -> list[LocalTrigger]:
        if not self.store_path.exists():
            return []
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            store_data = cast(dict[str, Any], data)
            raw_triggers = cast(list[Any], store_data.get("triggers", []))
            return [
                LocalTrigger.from_dict(cast(dict[str, Any], raw))
                for raw in raw_triggers
                if isinstance(raw, dict)
            ]
        except Exception as exc:
            backup = self.store_path.with_suffix(
                self.store_path.suffix + f".corrupt-{int(time.time())}"
            )
            with suppress(OSError):
                os.replace(self.store_path, backup)
            raise TriggerStoreError(
                f"trigger store at {self.store_path} could not be loaded and was preserved "
                "as a .corrupt-<ts> backup"
            ) from exc

    def _save_triggers_unlocked(self, triggers: list[LocalTrigger]) -> None:
        payload = {
            "version": 1,
            "triggers": [trigger.to_dict() for trigger in triggers],
        }
        self._atomic_write(self.store_path, json.dumps(payload, indent=2, ensure_ascii=False))

    @staticmethod
    def _find_unlocked(
        triggers: list[LocalTrigger],
        trigger_id: str,
    ) -> LocalTrigger | None:
        return next((trigger for trigger in triggers if trigger.id == trigger_id), None)

    def _move_bad_delivery_unlocked(self, path: Path) -> None:
        target = self.failed_dir / f"{path.name}.bad"
        with suppress(OSError):
            os.replace(path, target)

    def _requeue_unattempted_unlocked(self, delivery: TriggerDelivery) -> bool:
        """Return an unstarted claim to the inbox without counting an attempt."""
        if delivery.path is None:
            return False
        delivery.last_error = _PROCESSING_RECOVERY_ERROR
        target = self.inbox_dir / delivery.path.name
        self._atomic_write(target, json.dumps(_delivery_payload(delivery), ensure_ascii=False))
        delivery.path.unlink(missing_ok=True)
        return True

    def _read_run_record(self, delivery_id: str) -> dict[str, Any]:
        path = self.runs_dir / f"{delivery_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return cast(dict[str, Any], data) if isinstance(data, dict) else {}

    def _retry_delivery_unlocked(self, delivery: TriggerDelivery, error: str) -> bool:
        if delivery.path is None:
            return False
        if delivery.attempts + 1 >= _MAX_DELIVERY_ATTEMPTS:
            delivery.attempts += 1
            delivery.last_error = error
            failed = self.failed_dir / delivery.path.name
            self._atomic_write(failed, json.dumps(_delivery_payload(delivery), ensure_ascii=False))
            delivery.path.unlink(missing_ok=True)
            return False
        delivery.attempts += 1
        delivery.last_error = error
        target = self.inbox_dir / delivery.path.name
        self._atomic_write(target, json.dumps(_delivery_payload(delivery), ensure_ascii=False))
        delivery.path.unlink(missing_ok=True)
        return True

    def _delete_delivery_files_for_trigger_unlocked(self, trigger_id: str) -> None:
        for directory in (
            self.inbox_dir,
            self.processing_dir,
            self.unknown_dir,
            self.failed_dir,
        ):
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                if self._delivery_file_trigger_id(path) != trigger_id:
                    continue
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Trigger: failed to delete delivery file {} for deleted trigger {}: {}",
                        path,
                        trigger_id,
                        exc,
                    )

    @staticmethod
    def _delivery_file_trigger_id(path: Path) -> str | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        value = _get(_delivery_body(data), "triggerId", "trigger_id", None)
        return str(value) if value else None

    @staticmethod
    def _delivery_file_id(path: Path) -> str | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        value = _delivery_body(data).get("id")
        return str(value) if value else None

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            with suppress(PermissionError):
                fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    try:
                        os.fsync(fd)
                    except OSError as exc:
                        if exc.errno != errno.EINVAL:
                            raise
                finally:
                    os.close(fd)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


def _new_trigger_id(existing_ids: set[str]) -> str:
    for _ in range(100):
        suffix = "".join(secrets.choice(_TRIGGER_ID_ALPHABET) for _ in range(8))
        candidate = f"trg_{suffix}"
        if candidate not in existing_ids:
            return candidate
    raise TriggerStoreError("could not allocate a unique trigger id")


def _clean_name(name: str) -> str:
    stripped = " ".join(name.strip().split())
    return (stripped or "Local trigger")[:120]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _delivery_payload(
    delivery: TriggerDelivery,
    *,
    settlement_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": 1,
        "delivery": delivery.to_dict(),
    }
    if settlement_id:
        payload["settlementId"] = settlement_id
    return payload


def _delivery_body(data: object) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    payload = cast(dict[str, Any], data)
    raw = payload.get("delivery", payload)
    return cast(dict[str, Any], raw) if isinstance(raw, dict) else {}


def _payload_settlement_id(data: Any) -> str | None:
    for candidate in (data, _delivery_body(data)):
        value = _get(candidate, "settlementId", "settlement_id", None)
        if isinstance(value, str) and value:
            return value
    return None


def _delivery_run_record(
    delivery: TriggerDelivery,
    trigger: LocalTrigger | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "local_trigger",
        "trigger_id": delivery.trigger_id,
        "delivery_id": delivery.id,
        "content": _run_record_text(delivery.content),
        "created_at_ms": delivery.created_at_ms,
        "attempts": delivery.attempts,
    }
    if delivery.last_error:
        record["last_error"] = _run_record_text(delivery.last_error)
    if trigger is not None:
        record.update(
            {
                "trigger_name": trigger.name,
                "session_key": trigger.session_key,
                "channel": trigger.channel,
                "chat_id": trigger.chat_id,
                "sender_id": trigger.sender_id,
                "origin_metadata": trigger.origin_metadata,
            }
        )
    return record


def _run_record_text(value: str) -> str:
    return truncate_text(value, _RUN_RECORD_TEXT_MAX_CHARS)
