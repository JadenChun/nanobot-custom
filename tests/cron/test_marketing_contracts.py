"""Preserve main-branch job contracts without activating legacy execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nanobot.cron.action_store import ActionStoreLegacyConversionError, CronActionStore
from nanobot.cron.definition_store import (
    CronDefinition,
    CronDefinitionStore,
    DefinitionStoreValidationError,
)
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob, CronSchedule, CronVerifier
from tests.cron.test_action_store_conversion import NAMESPACE, _action, _job, _legacy_root


def _contract() -> dict:
    return {
        "planningMode": "off",
        "skipVerification": True,
        "alertChannel": "telegram",
        "alertTo": "owner-test",
        "verifiers": [{
            "name": "report evidence",
            "argv": ["python", "verify.py", "--date", "{date}", "--status", "{status}"],
            "cwd": "/test/knowledge",
            "timeout": 45.5,
            "statusFile": "reports/{date}.md",
            "statusRegex": r"status: (\w+)",
        }],
        "postDeliveryCommand": ["python", "ack.py", "{date}", "{ack_status}"],
    }


@pytest.mark.parametrize("snake_case", [False, True])
def test_import_preserves_contract_state_routes_and_idempotency(tmp_path: Path, snake_case: bool):
    job = _job(snake_case=snake_case)
    contract = _contract()
    if snake_case:
        aliases = {"planningMode": "planning_mode", "skipVerification": "skip_verification",
                   "alertChannel": "alert_channel", "alertTo": "alert_to",
                   "postDeliveryCommand": "post_delivery_command"}
        job["payload"].update({aliases.get(k, k): v for k, v in contract.items()})
        verifier = job["payload"]["verifiers"][0]
        verifier["status_file"] = verifier.pop("statusFile")
        verifier["status_regex"] = verifier.pop("statusRegex")
        job["state"].update(retry_at_ms=123456, rate_limit_retries=2)
    else:
        job["payload"].update(contract)
        job["state"].update(retryAtMs=123456, rateLimitRetries=2)
    source = _action("add", job)
    root = _legacy_root(tmp_path, actions=source, jobs=json.dumps({"version": 1, "jobs": [job]}).encode())
    first = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    store = CronActionStore(root, namespace=NAMESPACE)
    candidate = store.replay().candidate
    assert candidate is not None
    imported = candidate["operations"][0]["request"]["job"]
    decoded = CronJob.from_store_dict(imported)
    assert decoded.payload.planning_mode == "off"
    assert decoded.payload.skip_verification is True
    assert decoded.payload.alert_destination().to == "owner-test"
    assert decoded.payload.verifiers[0].argv[-1] == "{status}"
    assert decoded.payload.verifiers[0].status_file == "reports/{date}.md"
    assert decoded.payload.verifiers[0].timeout == 45.5
    assert decoded.payload.post_delivery_command == contract["postDeliveryCommand"]
    assert decoded.payload.channel_meta == {"topic": "reports"}
    assert decoded.payload.session_key == "telegram:42"
    assert decoded.state.retry_at_ms == 123456
    assert decoded.state.rate_limit_retries == 2
    assert decoded.state.run_history[0].occurrence_id == "occurrence-1"
    definitions = CronDefinitionStore(store).replay().candidate.definitions
    assert definitions[0].definition["payload"]["verifiers"][0]["statusFile"] == "reports/{date}.md"
    again = CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert again.conversion_identity == first.conversion_identity
    assert store.replay().candidate == candidate


@pytest.mark.parametrize("field,value", [
    ("skipVerification", "true"), ("planningMode", "auto"),
    ("alertTo", 42), ("postDeliveryCommand", "python ack.py"),
    ("verifiers", [{"name": "bad", "argv": "python verify.py"}]),
    ("verifiers", [{"name": "bad", "argv": ["python"], "timeout": True}]),
    ("verifiers", [{"name": "bad", "argv": ["python"], "timeout": -1}]),
    ("verifiers", [{"name": "bad", "argv": ["python"], "unexpected": True}]),
])
def test_invalid_contract_does_not_retire_legacy_source(tmp_path: Path, field: str, value: object):
    job = _job()
    job["payload"][field] = value
    source = _action("add", job)
    root = _legacy_root(tmp_path, actions=source)
    with pytest.raises(ActionStoreLegacyConversionError):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert (root / "action.jsonl").read_bytes() == source


@pytest.mark.parametrize("field,value", [("retryAtMs", True), ("rateLimitRetries", -1)])
def test_invalid_retry_state_does_not_retire_source(tmp_path: Path, field: str, value: object):
    job = _job()
    job["state"][field] = value
    source = _action("add", job)
    root = _legacy_root(tmp_path, actions=source)
    with pytest.raises(ActionStoreLegacyConversionError):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert (root / "action.jsonl").read_bytes() == source


def test_existing_definition_hash_remains_valid_without_optional_contract_fields():
    job = CronJob.from_store_dict(_job()).to_store_dict()
    definition = {k: job[k] for k in ("id", "name", "enabled", "schedule", "payload", "deleteAfterRun")}
    assert "verifiers" not in definition["payload"]
    instance = "1" * 64
    revision = hashlib.sha256(json.dumps({**definition, "jobInstanceId": instance},
                                        ensure_ascii=False, allow_nan=False,
                                        separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    wire = {**definition, "jobInstanceId": instance, "definitionRevision": revision}
    assert CronDefinition.decode_wire(wire).to_wire() == wire


def test_definition_management_retains_contract_and_detaches_mutable_inputs(tmp_path: Path):
    service = CronService(tmp_path / "cron" / "jobs.json")
    command = ["python", "ack.py", "{ack_status}"]
    verifier = CronVerifier("evidence", ("python", "verify.py"))
    job = service.add_job(
        "report", CronSchedule(kind="every", every_ms=60000), "prepare report",
        planning_mode="off", skip_verification=True, alert_to="test-owner",
        verifiers=[verifier], post_delivery_command=command,
    )
    command.append("not part of the job")
    assert service.get_job(job.id).payload == job.payload
    assert job.payload.alert_destination().channel == "telegram"
    assert job.payload.post_delivery_command == ["python", "ack.py", "{ack_status}"]
    assert job.payload.verifiers == [verifier]


def test_ambiguous_contract_aliases_do_not_retire_source(tmp_path: Path):
    job = _job()
    job["payload"].update(planningMode="on", planning_mode="off")
    source = _action("add", job)
    root = _legacy_root(tmp_path, actions=source)
    with pytest.raises(ActionStoreLegacyConversionError, match="both planningMode and planning_mode"):
        CronActionStore.convert_legacy(root, namespace=NAMESPACE)
    assert (root / "action.jsonl").read_bytes() == source


def test_invalid_optional_contract_is_rejected_in_definition_wire():
    job = CronJob.from_store_dict(_job()).to_store_dict()
    definition = {k: job[k] for k in ("id", "name", "enabled", "schedule", "payload", "deleteAfterRun")}
    definition["payload"]["skipVerification"] = "true"
    with pytest.raises(DefinitionStoreValidationError, match="boolean"):
        CronDefinition.from_definition(definition, job_instance_id="1" * 64)
