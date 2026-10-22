"""Stopped-source cron import into a new, independently reviewable directory."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict

from nanobot.cron.action_store import CronActionStore
from nanobot.cron.definition_store import CronDefinition, CronDefinitionAction, CronDefinitionStore
from nanobot.cron.gateway import validate_gateway_job
from nanobot.cron.schedule import next_fire
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob


class CronImportReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source: str
    destination: str
    namespace: str
    timezone: str | None
    source_hashes: dict[str, str]
    definitions: dict[str, str]
    legacy_states: dict[str, dict[str, object]]
    pending_retries_require_review: list[str]


def _sources(source: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for name in ("jobs.json", "action.jsonl"):
        path = source / name
        if path.is_symlink():
            raise ValueError("legacy cron source must not be a symlink")
        if path.exists():
            if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("invalid or oversized legacy cron source")
            result[name] = path.read_bytes()
    if not result:
        raise ValueError("no legacy jobs.json or action.jsonl found")
    return result


def _hashes(data: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}


def _identities(store: CronDefinitionStore) -> dict[str, str]:
    return {item.job_id: item.job_instance_id + ":" + item.definition_revision
            for item in store.replay().candidate.definitions}


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        if os.name != "nt":
            os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def import_legacy_cron_copy(source: Path, destination: Path, *,
                            timezone: str | None = None) -> CronImportReport:
    """Copy, strictly convert and qualify; never mutate source or start a timer.

    The caller must use a stopped source or a consistent snapshot. Publication
    is atomic into a previously unused destination; source bytes are checked
    again before publication. Originals and conversion backups remain intact.
    """
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("source and destination must be separate directory trees")
    data = _sources(source)
    hashes = _hashes(data)
    namespace = f"cron:{destination}"
    receipt = destination / "migration-receipt.json"
    if destination.exists():
        if not receipt.is_file() or receipt.is_symlink():
            raise ValueError("destination exists without a completed import receipt")
        report = CronImportReport.model_validate_json(receipt.read_bytes())
        if (report.source_hashes != hashes or report.source != str(source)
                or report.destination != str(destination) or report.namespace != namespace
                or report.timezone != timezone):
            raise ValueError("import inputs differ from the retained receipt")
        service = CronService(destination / "jobs.json")
        if _identities(service.definition_store) != report.definitions:
            raise ValueError("imported definitions changed; do not repeat the import")
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-import-", dir=destination.parent))
    try:
        for name, value in data.items():
            _write(stage / name, value)
        CronActionStore.convert_legacy(stage, namespace=namespace)
        actions = CronActionStore(stage, namespace=namespace)
        candidate = actions.replay().candidate
        if candidate is None:
            raise ValueError("conversion produced no legacy snapshot")
        raw_jobs: object = candidate.get("jobs")
        if not isinstance(raw_jobs, list):
            raise ValueError("conversion produced no checked job inventory")
        states: dict[str, dict[str, object]] = {}
        pending: list[str] = []
        for raw in cast(list[object], raw_jobs):
            if not isinstance(raw, dict):
                raise ValueError("invalid converted job")
            job = CronJob.from_store_dict(cast(dict[str, object], raw))
            # C1 already strictly validated these runtime/history values.
            states[job.id] = job.to_store_dict()["state"]
            if job.state.retry_at_ms is not None or job.state.rate_limit_retries:
                pending.append(job.id)
        store = CronDefinitionStore(actions)
        current = store.replay().candidate.definitions
        for item in current:
            wire = deepcopy(item.definition)
            payload = wire["payload"]
            if payload["executionMode"] != "legacy_fresh":
                if any(payload[key] is not None for key in ("sessionKey", "originChannel", "originChatId")):
                    raise ValueError(f"job {item.job_id} is explicitly session-bound; review its execution contract")
                payload["executionMode"] = "legacy_fresh"
                payload["sessionKey"] = f"cron:{item.job_id}"
                payload["originChannel"] = payload["channel"] or "cli"
                payload["originChatId"] = str(payload["to"] or "direct")
            schedule = wire["schedule"]
            if schedule["kind"] == "cron" and not schedule["tz"]:
                if not timezone:
                    raise ValueError(f"job {item.job_id} needs an explicit import timezone")
                schedule["tz"] = timezone
            # Validate schedule arithmetic and the full enabled gateway contract
            # before publishing anything at the requested destination.
            next_fire(schedule, 0, 0)
            changed = CronDefinition.from_definition(wire, job_instance_id=item.job_instance_id)
            if wire["enabled"]:
                public = CronJob.from_store_dict({**wire, "jobInstanceId": item.job_instance_id})
                validate_gateway_job(public)
            store.append(action_id="import:" + item.job_id, action=CronDefinitionAction.update(changed,
                expected_job_instance_id=item.job_instance_id, expected_definition_revision=item.definition_revision))
            store.checkpoint(store.replay())
        report = CronImportReport(source=str(source), destination=str(destination), namespace=namespace,
            timezone=timezone, source_hashes=hashes, definitions=_identities(store),
            legacy_states=states, pending_retries_require_review=sorted(pending))
        _write(stage / receipt.name, (report.model_dump_json(indent=2) + "\n").encode())
        if _hashes(_sources(source)) != hashes:
            raise ValueError("legacy source changed during import; take a consistent stopped copy")
        os.rename(stage, destination)
        if os.name != "nt":
            parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)
