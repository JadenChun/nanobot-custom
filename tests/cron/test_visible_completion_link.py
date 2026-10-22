"""A completed cron occurrence can inform its visible chat without mirroring a turn."""

import asyncio
from pathlib import Path

import pytest

from nanobot.cron.action_store import CronActionStore
from nanobot.cron.completion import (
    CronCompletionConflictError,
    CronCompletionIncompleteError,
    CronCompletionProjector,
)
from nanobot.cron.definition_store import CronDefinitionAction, CronDefinitionStore
from nanobot.cron.occurrence_store import CronOccurrenceStore
from nanobot.cron.service import CronService, _job_definition
from nanobot.cron.types import CronDestination, CronSchedule
from nanobot.cron.visible_link import link_completed_text_result
from nanobot.session.manager import Session, SessionManager, SessionPolicy, SessionWriteConflict


def _claim(tmp_path: Path):
    root = tmp_path / "cron"
    namespace = "cron:visible-link-test"
    definitions = CronDefinitionStore(CronActionStore(root, namespace=namespace))
    occurrences = CronOccurrenceStore(root / "occurrences", namespace=namespace)
    service = CronService(
        root / "jobs.json", definition_store=definitions,
        occurrence_store=occurrences,
        completion_projector=CronCompletionProjector(occurrences, root / "runs"),
    )
    job = service.add_job(
        "report", CronSchedule(kind="every", every_ms=60_000), "compute report",
        deliver=True, channel="telegram", to="primary",
        additional_destinations=[CronDestination("discord", "secondary")],
    )
    job.payload.execution_mode = "legacy_fresh"
    job.payload.session_key = f"cron:{job.id}"
    job.payload.origin_channel = "telegram"
    job.payload.origin_chat_id = "primary"
    service._definition_action(CronDefinitionAction.update(
        _job_definition(job, job_instance_id=job.job_instance_id),
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    ))
    job = service.get_job(job.id)
    assert job is not None
    claim = service.claim_scheduled_occurrence(
        job.id, 1_700_000_000_000, owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    return service, claim


def _complete(service: CronService, claim, completion, **kwargs):
    def settle():
        with service.occurrence_store.acquire_execution_fence(claim) as fence:
            service.occurrence_store.begin_execution(claim, fence)
            return service.complete_occurrence(claim, completion, fence=fence, **kwargs)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_sync(settle))
    return settle()


async def _run_sync(callback):
    return callback()


def test_link_is_durable_bounded_idempotent_and_does_not_mirror_transcript(tmp_path: Path):
    service, claim = _claim(tmp_path)
    sessions = SessionManager(tmp_path / "workspace")
    visible = sessions.get_or_create("telegram:primary")
    visible.add_message("user", "while the schedule runs")
    visible.add_message("assistant", "earlier answer")
    visible.add_message("user", "[Scheduled task from cron:old]\nold instruction")
    visible.add_message("assistant", "old duplicated result")
    sessions.save(visible)
    with pytest.raises(CronCompletionIncompleteError):
        link_completed_text_result(service, sessions, claim.occurrence_id)
    assert visible.recent_scheduled_runs() == []

    # The linker receives only the durable occurrence ID, never the model's
    # in-memory result or a claim capable of recomputation.
    _complete(service, claim, {
        "status": "ok", "result": "x" * 300 + " report done",
        "delivery": {"version": 1, "deliver": True,
                     "messages": [{"content": "x" * 300 + " report done", "media": []}]},
    }, run_id=claim.run_id)
    first = link_completed_text_result(service, sessions, claim.occurrence_id)
    second = link_completed_text_result(service, sessions, claim.occurrence_id)
    assert first == second == "telegram:primary"
    reopened = SessionManager(tmp_path / "workspace").get_or_create(first)
    assert [item["content"] for item in reopened.messages] == [
        "while the schedule runs", "earlier answer",
        "[Scheduled task from cron:old]\nold instruction", "old duplicated result",
    ]
    ring = reopened.recent_scheduled_runs()
    assert len(ring) == 1
    assert ring[0]["run_id"] == claim.run_id
    assert ring[0]["occurrence_id"] == claim.occurrence_id
    assert ring[0]["detail_ref"] == {"session_key": f"cron:{claim.identity.job_id}",
                                     "run_id": claim.run_id}
    assert len(ring[0]["result"]) <= 240
    assert "report done" not in ring[0]["result"]  # bounded excerpt, not a full trace
    assert "run=" + claim.run_id in reopened.get_model_history()[0]["content"]
    assert not any(item["content"] == "old duplicated result"
                   for item in reopened.get_model_history())
    assert SessionManager(tmp_path / "workspace").get_or_create("discord:secondary").messages == []


def test_link_preserves_concurrent_chat_writes_and_recovers_after_save_failure(tmp_path: Path,
                                                                                monkeypatch):
    service, claim = _claim(tmp_path)
    sessions = SessionManager(tmp_path / "workspace")
    # Hold a cached empty view while another process-shaped manager commits a
    # visible chat turn. The metadata-only save must merge, not replace it.
    sessions.get_or_create("telegram:primary")
    _complete(service, claim, {
        "status": "ok", "result": "done",
        "delivery": {"version": 1, "deliver": True,
                     "messages": [{"content": "done", "media": []}]},
    }, run_id=claim.run_id)
    other = SessionManager(tmp_path / "workspace")
    active = other.get_or_create("telegram:primary")
    active.add_message("user", "new concurrent input")
    other.save(active)
    original = sessions.upsert_session_metadata
    def fail_once(_key, _updates, **_kwargs):
        raise OSError("session write failed")
    monkeypatch.setattr(sessions, "upsert_session_metadata", fail_once)
    with pytest.raises(OSError, match="session write failed"):
        link_completed_text_result(service, sessions, claim.occurrence_id)
    monkeypatch.setattr(sessions, "upsert_session_metadata", original)
    link_completed_text_result(service, sessions, claim.occurrence_id)
    reloaded = SessionManager(tmp_path / "workspace").get_or_create("telegram:primary")
    assert [m["content"] for m in reloaded.messages] == ["new concurrent input"]
    assert len(reloaded.recent_scheduled_runs()) == 1


def test_failed_or_noncanonical_completion_does_not_link(tmp_path: Path):
    service, claim = _claim(tmp_path)
    sessions = SessionManager(tmp_path / "workspace")
    _complete(service, claim, {"status": "error", "result": "not a successful result"},
                                run_id=claim.run_id)
    with pytest.raises(CronCompletionConflictError, match="text-only"):
        link_completed_text_result(service, sessions, claim.occurrence_id)
    assert sessions.get_or_create("telegram:primary").recent_scheduled_runs() == []


@pytest.mark.parametrize(("extra", "delivery_extra", "message_media"), [
    pytest.param({"error": "contradictory failure"}, {}, [], id="error"),
    pytest.param({
        "sent_messages": [{"content": "already sent", "media": []}],
    }, {}, [], id="sent-messages"),
    pytest.param({"media": ["result.png"]}, {}, [], id="top-level-media"),
    pytest.param({"unknown_evidence": True}, {}, [], id="unknown-top-level-field"),
    pytest.param({}, {"receipt": "unexpected"}, [], id="delivery-extra-field"),
    pytest.param({}, {}, ["result.png"], id="message-media"),
])
def test_link_rejects_noncanonical_text_completion_evidence(
    tmp_path: Path,
    extra: dict,
    delivery_extra: dict,
    message_media: list[str],
):
    service, claim = _claim(tmp_path)
    sessions = SessionManager(tmp_path / "workspace")
    completion = {
        "status": "ok",
        "result": "done",
        "delivery": {
            "version": 1,
            "deliver": True,
            "messages": [{"content": "done", "media": message_media}],
            **delivery_extra,
        },
        **extra,
    }
    _complete(service, claim, completion, run_id=claim.run_id)

    with pytest.raises(CronCompletionConflictError, match="text-only"):
        link_completed_text_result(service, sessions, claim.occurrence_id)

    assert sessions.get_or_create("telegram:primary").recent_scheduled_runs() == []


def test_nonpersistent_visible_session_is_not_acknowledged(tmp_path: Path):
    service, claim = _claim(tmp_path)
    _complete(service, claim, {
        "status": "ok", "result": "done",
        "delivery": {"version": 1, "deliver": True,
                     "messages": [{"content": "done", "media": []}]},
    }, run_id=claim.run_id)
    sessions = SessionManager(tmp_path / "workspace")
    transient = Session(key="telegram:primary", policy=SessionPolicy(persist=False))
    sessions._remember(transient)
    with pytest.raises(CronCompletionConflictError, match="durable chat"):
        link_completed_text_result(service, sessions, claim.occurrence_id)


def test_scheduled_ring_preserves_legacy_metadata_and_bounds_projection():
    session = Session(key="telegram:room")
    for index in range(25):
        session.add_scheduled_run(
            job_id=f"job-{index}", run_id=f"run-{index}",
            instruction="instruction-" + "i" * 400,
            result="result-" + "r" * 400,
            detail_session_key=f"cron:job-{index}",
        )
    assert len(session.recent_scheduled_runs()) == 20
    assert session.recent_scheduled_runs()[0]["run_id"] == "run-5"
    assert len(session.get_model_history()[0]["content"]) <= session.SCHEDULED_PROJECTION_CHARS
    session.add_scheduled_run(
        job_id="job-24", run_id="run-24", instruction="instruction-" + "i" * 400,
        result="result-" + "r" * 400, detail_session_key="cron:job-24",
    )
    assert len(session.recent_scheduled_runs()) == 20
    with pytest.raises(SessionWriteConflict):
        session.add_scheduled_run(
            job_id="job-24", run_id="run-24", instruction="different", result="result",
            detail_session_key="cron:job-24",
        )


@pytest.mark.parametrize("raw", [None, {}, (), "not a ring"])
def test_scheduled_ring_keeps_non_list_metadata_unchanged(raw):
    session = Session(key="telegram:room", metadata={Session.SCHEDULED_RING_KEY: raw})

    assert session.recent_scheduled_runs() == []
    assert session.metadata[Session.SCHEDULED_RING_KEY] is raw


def test_scheduled_ring_preserves_arbitrary_keys_and_shallow_copy_semantics():
    marker = object()
    nested = {"legacy": ["value"]}
    legacy = {"run_id": "legacy", 7: "integer-key", marker: nested}
    entries = [{"run_id": f"old-{index}"} for index in range(24)]
    entries.extend([None, "ignored", legacy])
    session = Session(key="telegram:room", metadata={Session.SCHEDULED_RING_KEY: entries})

    ring = session.recent_scheduled_runs()

    assert len(ring) == 20
    assert ring[0] == {"run_id": "old-5"}
    assert ring[-1] == legacy and ring[-1] is not legacy
    assert ring[-1][7] == "integer-key" and ring[-1][marker] is nested
    assert ring[-1].get("approval_granted", False) is False
    ring[-1][7] = "detached top level"
    assert legacy[7] == "integer-key"
    assert session.metadata[Session.SCHEDULED_RING_KEY] is entries


@pytest.mark.parametrize("reference", [None, False, 0, "", [], ()])
def test_scheduled_projection_preserves_falsey_detail_reference_defaults(reference):
    entry = {"job_id": "legacy-job", "run_id": "legacy-run", "detail_ref": reference}
    session = Session(key="telegram:room", metadata={Session.SCHEDULED_RING_KEY: [entry]})

    assert session.get_model_history() == [{"role": "user", "content": (
        "[Recent Scheduled Runs]\n"
        "Recent scheduled activity is metadata only; retrieve full detail by run reference.\n"
        "Completed means the computation finished; it does not prove channel delivery.\n"
        "- job=legacy-job run=legacy-run status= approval_granted=false ref=/\n"
        "  instruction: \n"
        "  result: "
    )}]
    assert entry["detail_ref"] is reference


@pytest.mark.parametrize("reference", ["bad-reference", 1, ["bad-reference"], object()])
def test_scheduled_projection_retains_truthy_non_mapping_reference_error(reference):
    entry = {"detail_ref": reference}
    session = Session(key="telegram:room", metadata={Session.SCHEDULED_RING_KEY: [entry]})

    with pytest.raises(AttributeError, match="get"):
        session.get_model_history()

    assert entry["detail_ref"] is reference


def test_scheduled_projection_retains_dynamic_detail_reference_lookup():
    calls = []

    class LegacyReference:
        def get(self, key, default):
            calls.append((key, default))
            return {"session_key": "legacy-session", "run_id": "legacy-run"}.get(key, default)

    reference = LegacyReference()
    entry = {"detail_ref": reference}
    session = Session(key="telegram:room", metadata={Session.SCHEDULED_RING_KEY: [entry]})

    assert "ref=legacy-session/legacy-run" in session.get_model_history()[0]["content"]
    assert calls == [("session_key", ""), ("run_id", "")]
    assert entry["detail_ref"] is reference
