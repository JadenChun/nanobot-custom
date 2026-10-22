"""Legacy marketing contracts can be rehearsed in a new store, with originals intact."""

import json
from types import SimpleNamespace

import pytest

from nanobot.config.schema import GatewayCronConfig
from nanobot.cron.gateway import GatewayCron
from nanobot.cron.import_copy import import_legacy_cron_copy
from nanobot.cron.scheduler_store import SchedulerConflictError
from nanobot.cron.service import CronService
from nanobot.session.manager import SessionManager
from tests.cron.stage3_fixtures import FakeChannels, FakeClock
from tests.cron.test_action_store_conversion import _job
from tests.cron.test_marketing_contracts import _contract


def _source(tmp_path, *, bound=False, timezone="Asia/Kuala_Lumpur"):
    source = tmp_path / "original"
    source.mkdir(mode=0o700)
    job = _job()
    job["payload"].update(_contract())
    job["payload"]["additionalDestinations"] = [{"channel": "telegram", "to": "43"}]
    job["payload"]["channelMeta"] = {"message_thread_id": 42}
    if not bound:
        for key in ("sessionKey", "executionMode", "originChannel", "originChatId", "originMetadata"):
            job["payload"].pop(key, None)
    job["schedule"]["tz"] = timezone
    job["state"].update(retryAtMs=123456, rateLimitRetries=2)
    data = json.dumps({"version": 1, "jobs": [job]}).encode()
    path = source / "jobs.json"
    path.write_bytes(data)
    path.chmod(0o600)
    return source, job, data


def test_import_is_idempotent_preserves_history_and_qualifies_new_schedule_evidence(tmp_path):
    source, original, data = _source(tmp_path)
    destination = tmp_path / "candidate" / "cron"
    report = import_legacy_cron_copy(source, destination)
    assert (source / "jobs.json").read_bytes() == data
    assert import_legacy_cron_copy(source, destination) == report
    assert report.pending_retries_require_review == [original["id"]]
    assert report.legacy_states[original["id"]]["rateLimitRetries"] == 2
    assert report.legacy_states[original["id"]]["runHistory"]
    service = CronService(destination / "jobs.json")
    job = service.list_jobs()[0]
    assert job.id == original["id"]
    assert job.payload.execution_mode == "legacy_fresh"
    assert job.payload.session_key == f"cron:{job.id}"
    assert job.payload.verifiers[0].cwd == "/test/knowledge"
    assert job.payload.post_delivery_command == _contract()["postDeliveryCommand"]
    assert job.payload.channel_meta == {"message_thread_id": 42}
    assert job.schedule.tz == "Asia/Kuala_Lumpur"
    assert len(service.definition_store.replay().candidate.scheduling_commits) == 1
    assert service.occurrence_store is None
    # Rollback rehearses the untouched old representation, never the v2 stores.
    assert json.loads((source / "jobs.json").read_bytes())["jobs"][0] == original


def test_explicitly_bound_job_needs_review_and_never_publishes_partial_store(tmp_path):
    source, _original, data = _source(tmp_path, bound=True)
    destination = tmp_path / "candidate" / "cron"
    with pytest.raises(ValueError, match="session-bound"):
        import_legacy_cron_copy(source, destination)
    assert not destination.exists()
    assert (source / "jobs.json").read_bytes() == data


def test_missing_timezone_requires_an_explicit_value_and_repeated_inputs_match(tmp_path):
    source, _original, _data = _source(tmp_path, timezone=None)
    destination = tmp_path / "candidate" / "cron"
    with pytest.raises(ValueError, match="explicit import timezone"):
        import_legacy_cron_copy(source, destination)
    report = import_legacy_cron_copy(source, destination, timezone="Asia/Kuala_Lumpur")
    assert report.timezone == "Asia/Kuala_Lumpur"
    with pytest.raises(ValueError, match="inputs differ"):
        import_legacy_cron_copy(source, destination, timezone="UTC")
    changed = json.loads((source / "jobs.json").read_bytes())
    changed["jobs"][0]["name"] = "changed after snapshot"
    (source / "jobs.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="inputs differ"):
        import_legacy_cron_copy(source, destination, timezone="Asia/Kuala_Lumpur")


def test_source_destination_overlap_is_rejected_without_effect(tmp_path):
    source, _original, data = _source(tmp_path)
    with pytest.raises(ValueError, match="separate directory trees"):
        import_legacy_cron_copy(source, source / "copy")
    assert (source / "jobs.json").read_bytes() == data


async def test_legacy_budget_is_seeded_once_and_uncertain_retry_needs_stopped_review(tmp_path):
    source, original, _data = _source(tmp_path)
    destination = tmp_path / "candidate" / "cron"
    import_legacy_cron_copy(source, destination)
    service = CronService(destination / "jobs.json")
    gateway = GatewayCron(GatewayCronConfig(enabled=True), service, SimpleNamespace(),
        SessionManager(destination.parent, sessions_root=tmp_path / "sessions"),
        FakeChannels(), destination.parent, tmp_path / "instance", clock=FakeClock(1000))
    gateway.runtime.acquire()
    gateway._import_retry_budget()
    quota = gateway.runtime.quota_recovery
    job = service.get_job(original["id"])
    state = quota.state(job.job_instance_id)
    assert state.retry_count == 2 and state.pending.state == "unknown"
    assert state.pending.retry_at_ms == 123456
    with pytest.raises(SchedulerConflictError):
        quota.reconcile_pending(state.pending, scheduler=gateway.runtime.store,
            resolution="abandon", external_evidence="copied old job reviewed")
    gateway.runtime.stop_claiming()
    gateway.runtime.scheduler.release_lease()
    quota.reconcile_pending(state.pending, scheduler=gateway.runtime.store,
        resolution="abandon", external_evidence="copied old job reviewed")
    assert quota.state(job.job_instance_id).pending is None
    assert quota.state(job.job_instance_id).retry_count == 2
    quota.import_legacy(state.pending, instance=job.job_instance_id, retry_count=2)
    assert quota.state(job.job_instance_id).pending is None
    await gateway.aclose()
