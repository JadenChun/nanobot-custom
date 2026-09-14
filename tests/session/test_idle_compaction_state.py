"""Dormant idle-compaction activity, capture, and admission primitives."""

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from nanobot.session.execution_owner import ExecutionOwnershipError, acquire_execution
from nanobot.session.idle_compaction import IdleCaptureToken
from nanobot.session.manager import JsonlSessionStore, Session, SessionWriteConflict
from nanobot.session.source import SourceValidationError, new_source_event, prefix_digest


@pytest.fixture
def backend(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def seed(backend, *, key="test:idle"):
    session = Session(key)
    session.messages = [{"role": "user", "content": "original"}]
    backend.save(session)
    return backend.source.upgrade_source_session(
        session.key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )


def candidate(snapshot, *, identity="idle-checkpoint", references=None):
    references = [] if references is None else references
    boundary = len(snapshot.events)
    return {
        "schema_version": 2 if references else 1,
        "digest_version": 1,
        "summary_format_version": 1,
        "session_generation": snapshot.state.generation,
        "source_revision": snapshot.state.revision,
        "checkpoint_id": identity,
        "parent_checkpoint_id": snapshot.state.checkpoint_id,
        "producing_model": "model",
        "producing_provider": "provider",
        "digest_algorithm": "sha256",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": "Retained facts",
        "artifact_references": references,
        "boundary": boundary,
        "boundary_event_id": snapshot.events[boundary - 1]["_event_id"] if boundary else None,
        "prefix_digest": prefix_digest(snapshot.events, boundary),
        "auxiliary_usage": None,
    }


def admit_user(backend, snapshot):
    return backend.source.record_user_activity(
        snapshot.state.session_key,
        snapshot.state,
        accepted_foreground_user=True,
    )


def capture(backend, snapshot):
    token = backend.source.capture_idle_compaction(snapshot.state.session_key)
    assert isinstance(token, IdleCaptureToken)
    return token


def test_only_accepted_foreground_admission_changes_activity_metadata(backend):
    initial = seed(backend)
    assert backend.source.capture_idle_compaction(initial.state.session_key) is None

    with pytest.raises(SourceValidationError):
        backend.source.record_user_activity(
            initial.state.session_key,
            initial.state,
            accepted_foreground_user=False,
        )
    admitted = admit_user(backend, initial)
    activity = admitted.header["metadata"]
    timestamp = activity["last_user_activity_at"]
    assert activity["user_activity_revision"] == 1

    direct_user = backend.source.append_source_events(admitted.state.session_key, admitted.state, [
        new_source_event({"role": "user", "content": "not an admission"}, run_id="direct"),
    ])
    assert direct_user.header["metadata"]["last_user_activity_at"] == timestamp

    intent = new_source_event({
        "role": "assistant", "content": "", "tool_calls": [{"id": "call", "function": {"name": "x"}}],
    }, run_id="tool")
    intent["_execution_ids"] = ["execution"]
    result = new_source_event({
        "role": "tool", "tool_call_id": "call", "name": "x", "content": "done",
    }, run_id="tool")
    result["_execution_id"] = "execution"
    tool_result = backend.source.append_source_events(
        direct_user.state.session_key, direct_user.state, [intent, result]
    )
    assert tool_result.header["metadata"]["last_user_activity_at"] == timestamp

    checkpoint = backend.source.commit_context_checkpoint(
        tool_result.state.session_key, tool_result.state, candidate(tool_result)
    )
    assert checkpoint.header["metadata"]["last_user_activity_at"] == timestamp
    backend.source.update_metadata(checkpoint.state.session_key, {
        "execution_recovery_required": True,
        "user_activity_revision": 100,
        "last_user_activity_at": "forged",
    })
    recovered = backend.source.load_source_snapshot(checkpoint.state.session_key)
    assert recovered.header["metadata"]["last_user_activity_at"] == timestamp
    assert recovered.header["metadata"]["user_activity_revision"] == 1


def test_generic_metadata_writers_cannot_forge_activity_state(backend):
    snapshot = seed(backend)
    assert backend.update_metadata(snapshot.state.session_key, {
        "user_activity_revision": 9,
        "last_user_activity_at": "2026-09-14T00:00:00+00:00",
    })
    loaded = backend.source.load_source_snapshot(snapshot.state.session_key)
    assert "user_activity_revision" not in loaded.header["metadata"]
    assert "last_user_activity_at" not in loaded.header["metadata"]

    admitted = admit_user(backend, loaded)
    before = dict(admitted.header["metadata"])
    backend.source.update_metadata(admitted.state.session_key, {
        "user_activity_revision": 99,
        "last_user_activity_at": "2026-09-14T00:00:00+00:00",
    })
    current = backend.source.load_source_snapshot(admitted.state.session_key)
    assert {name: current.header["metadata"][name] for name in before if name.startswith("user_") or name.startswith("last_user_")} == {
        name: before[name] for name in before if name.startswith("user_") or name.startswith("last_user_")
    }


def test_missing_or_malformed_activity_metadata_is_idle_ineligible(backend):
    snapshot = seed(backend)
    assert backend.source.capture_idle_compaction(snapshot.state.session_key) is None

    path = backend.get_session_path(snapshot.state.session_key)
    rows = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    header["metadata"]["user_activity_revision"] = 1
    rows[0] = json.dumps(header)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(SourceValidationError, match="activity metadata"):
        backend.source.capture_idle_compaction(snapshot.state.session_key)


def test_idle_capture_is_detached_and_binds_checkpoint_activity_and_durable_state(backend):
    admitted = admit_user(backend, seed(backend))
    token = capture(backend, admitted)
    before = token
    snapshot = backend.source.load_source_snapshot(admitted.state.session_key)
    snapshot.events[0]["content"] = "mutated detached snapshot"
    snapshot.header["metadata"]["last_user_activity_at"] = "mutated detached metadata"
    assert token == before
    assert token.source_revision == admitted.state.revision
    assert token.generation == admitted.state.generation
    assert token.checkpoint_id is None and token.checkpoint_revision is None
    assert token.pending_state.idle_eligible


def test_idle_capture_binds_the_accepted_checkpoint_identity_and_revision(backend):
    initial = seed(backend)
    checkpoint = backend.source.commit_context_checkpoint(
        initial.state.session_key, initial.state, candidate(initial, identity="accepted")
    )
    admitted = admit_user(backend, checkpoint)
    token = capture(backend, admitted)
    assert token.checkpoint_id == "accepted"
    assert token.checkpoint_revision == checkpoint.checkpoint["source_revision"]
    assert token.checkpoint_fingerprint == admitted.state.checkpoint_fingerprint


def test_idle_admission_rejects_source_append_user_activity_reset_and_checkpoint_races(backend):
    admitted = admit_user(backend, seed(backend, key="test:append"))

    token = capture(backend, admitted)
    backend.source.append_source_events(admitted.state.session_key, admitted.state, [
        new_source_event({"role": "assistant", "content": "append"}, run_id="run"),
    ])
    with pytest.raises(SessionWriteConflict):
        backend.source.validate_idle_checkpoint_admission(admitted.state.session_key, token, candidate(admitted))

    admitted = admit_user(backend, seed(backend, key="test:activity"))
    token = capture(backend, admitted)
    latest = admit_user(backend, admitted)
    with pytest.raises(SessionWriteConflict):
        backend.source.validate_idle_checkpoint_admission(latest.state.session_key, token, candidate(admitted))

    admitted = admit_user(backend, seed(backend, key="test:reset"))
    token = capture(backend, admitted)
    reset = backend.source.reset_source_session(admitted.state.session_key, admitted.state, [])
    with pytest.raises(SessionWriteConflict):
        backend.source.validate_idle_checkpoint_admission(reset.state.session_key, token, candidate(admitted))

    admitted = admit_user(backend, seed(backend, key="test:checkpoint"))
    token = capture(backend, admitted)
    checkpoint = backend.source.commit_context_checkpoint(
        admitted.state.session_key, admitted.state, candidate(admitted, identity="other")
    )
    with pytest.raises(SessionWriteConflict):
        backend.source.validate_idle_checkpoint_admission(checkpoint.state.session_key, token, candidate(admitted))


@pytest.mark.parametrize(
    "kind",
    ["pending", "followups", "webui_recovery", "lease", "recovery", "sidecar", "unresolved"],
)
def test_idle_admission_rejects_durable_pending_recovery_and_unresolved_state(backend, kind):
    admitted = admit_user(backend, seed(backend))
    token = capture(backend, admitted)
    path = backend.get_session_path(admitted.state.session_key)
    rows = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    if kind == "pending":
        header["metadata"]["pending_user_turn"] = {"id": "pending"}
        rows[0] = json.dumps(header)
    elif kind == "followups":
        header["metadata"]["pending_user_followups"] = [{"id": "followup"}]
        rows[0] = json.dumps(header)
    elif kind == "webui_recovery":
        header["metadata"]["webui_recovery"] = {"status": "resuming"}
        rows[0] = json.dumps(header)
    elif kind == "lease":
        header["metadata"]["execution_lease"] = {"token": "active"}
        rows[0] = json.dumps(header)
    elif kind == "recovery":
        header["metadata"]["execution_recovery_required"] = True
        rows[0] = json.dumps(header)
    elif kind == "sidecar":
        backend.get_runtime_checkpoint_path(admitted.state.session_key).write_text(
            '{"version": 2}', encoding="utf-8"
        )
    else:
        intent = new_source_event({
            "role": "assistant", "content": "", "tool_calls": [{"id": "call", "function": {"name": "x"}}],
        }, run_id="tool")
        intent["_execution_ids"] = ["unresolved"]
        rows.append(json.dumps(intent))
    if kind != "sidecar":
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(SessionWriteConflict):
        backend.source.validate_idle_checkpoint_admission(admitted.state.session_key, token, candidate(admitted))


def test_idle_admission_rejects_a_forged_or_mismatched_token(backend):
    admitted = admit_user(backend, seed(backend))
    token = capture(backend, admitted)
    forged = replace(token, user_activity_revision=token.user_activity_revision + 1)
    with pytest.raises(SessionWriteConflict):
        backend.source.validate_idle_checkpoint_admission(
            admitted.state.session_key, forged, candidate(admitted)
        )


def test_idle_admission_is_non_mutating_and_dependency_bearing_candidate_fails_closed(backend):
    admitted = admit_user(backend, seed(backend))
    token = capture(backend, admitted)
    path = backend.get_session_path(admitted.state.session_key)
    before = path.read_bytes()
    timestamp = admitted.header["metadata"]["last_user_activity_at"]

    admission = backend.source.validate_idle_checkpoint_admission(
        admitted.state.session_key, token, candidate(admitted)
    )
    assert admission.capture == token
    assert path.read_bytes() == before
    assert backend.source.load_source_snapshot(admitted.state.session_key).header["metadata"]["last_user_activity_at"] == timestamp

    with pytest.raises(SourceValidationError, match="integration-owned dependency publisher"):
        backend.source.validate_idle_checkpoint_admission(
            admitted.state.session_key,
            token,
            candidate(admitted, references=["artifact:" + "a" * 32]),
        )
    assert path.read_bytes() == before


async def test_activity_update_reuses_current_execution_owner(backend):
    initial = seed(backend)
    key = initial.state.session_key
    async with acquire_execution(backend, key, run_id="foreground") as owner:
        before = owner.read_snapshot()
        committed = backend.source.record_user_activity(
            key,
            before.state,
            accepted_foreground_user=True,
            owner=owner,
        )
        assert committed.header["metadata"]["user_activity_revision"] == 1
        owner.validate_dispatch()
    loaded = backend.source.load_source_snapshot(key)
    assert loaded.header["metadata"]["user_activity_revision"] == 1
    assert loaded.state.revision == initial.state.revision + 3


async def test_activity_update_rejects_stale_or_foreign_owner(backend, tmp_path):
    initial = seed(backend)
    key = initial.state.session_key
    async with acquire_execution(backend, key, run_id="foreground") as owner:
        expected = owner.read_snapshot().state
    with pytest.raises(ExecutionOwnershipError):
        backend.source.record_user_activity(
            key, expected, accepted_foreground_user=True, owner=owner,
        )

    other = JsonlSessionStore(tmp_path / "other-workspace", sessions_root=tmp_path / "other-private")
    seed(other, key=key)
    async with acquire_execution(other, key, run_id="foreign") as foreign:
        with pytest.raises(ExecutionOwnershipError):
            backend.source.record_user_activity(
                key,
                initial.state,
                accepted_foreground_user=True,
                owner=foreign,
            )
