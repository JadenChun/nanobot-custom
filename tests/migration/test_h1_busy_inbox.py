"""Durable H1 admission/ordering tests with actual private session authorities."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from nanobot.agent.busy_control import ControlDecision, ControlSnapshot
from nanobot.config.schema import BusyInboxConfig
from nanobot.session.busy_inbox import (
    InboxCapacityError,
    InboxConflictError,
    InboxPayload,
    InboxStateError,
    Origin,
    SessionBusyInbox,
)
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import SessionManager
from nanobot.session.runtime_authority import DurableSessionAuthority


@pytest.fixture(params=["legacy", "canonical"])
async def sessions(tmp_path: Path, request: pytest.FixtureRequest) -> AsyncGenerator[SessionManager, None]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "runtime" / "sessions"
    if request.param == "legacy":
        yield SessionManager(workspace, sessions_root=root)
    else:
        lease = await DurableSessionAuthority(workspace, sessions_root=root).acquire()
        try:
            yield lease.manager
        finally:
            assert lease.close is not None
            await lease.close()


def payload(number: int = 1, *, channel: str = "telegram", content: str = "full user message",
            owner: str = "owner", key: str = "assistant", scope: tuple[str, ...] = ("read",)) -> InboxPayload:
    return InboxPayload(owner_id=owner, session_key=key, content=content, media=("media-ref",),
                        scope=scope, origin=Origin(channel=channel, account_id="account", chat_id="chat",
                                                  sender_id="sender", transport_id=str(number), thread_id="thread"))


def active(run: str = "run", generation: int = 1, queued: int = 0) -> ControlSnapshot:
    return ControlSnapshot(run_id=run, generation=generation, state="running", queued=queued)


def inbox(sessions: SessionManager, **kwargs) -> SessionBusyInbox:
    return SessionBusyInbox(sessions, owner_id="owner", session_key="assistant", **kwargs)


def test_durable_acceptance_keeps_complete_payload_and_origin(sessions: SessionManager) -> None:
    store = inbox(sessions)
    receipt = store.admit(payload(content="完整原始内容\nsecond line"))
    assert receipt.duplicate is False and receipt.entry.state == "pending-routing"
    recovered = inbox(sessions)
    saved = recovered.entries()[0]
    assert saved.payload == payload(content="完整原始内容\nsecond line")
    assert recovered.notices()[0].stage == "accepted"
    snapshot = sessions.read_session_snapshot("assistant")
    assert snapshot is not None and snapshot.messages == []
    assert "完整" not in repr(saved)


def test_duplicate_transport_retry_cannot_create_another_task_or_receipt(sessions: SessionManager) -> None:
    store = inbox(sessions)
    first = store.admit(payload())
    again = inbox(sessions).admit(payload())
    assert again.duplicate is True and again.entry.id == first.entry.id
    assert len(store.entries()) == len(store.notices()) == 1
    with pytest.raises(InboxConflictError):
        store.admit(payload(content="different content for same transport identity"))
    assert len(store.entries()) == 1


@pytest.mark.parametrize("bad", [payload(owner="another-owner"), payload(key="another-session")])
def test_owner_session_mismatch_is_rejected_before_acceptance(sessions: SessionManager, bad: InboxPayload) -> None:
    store = inbox(sessions)
    with pytest.raises(InboxConflictError):
        store.admit(bad)
    assert store.entries() == store.notices() == ()


def test_burst_above_twenty_is_durable_without_overflow_dispatch(sessions: SessionManager) -> None:
    store = inbox(sessions)
    for number in range(45):
        store.admit(payload(number))
    assert len(inbox(sessions).entries()) == 45
    assert [row.seq for row in store.entries()] == list(range(1, 46))
    assert all(row.state == "pending-routing" for row in store.entries())


def test_capacity_rejects_before_new_acceptance_notice(sessions: SessionManager) -> None:
    store = inbox(sessions, limits=BusyInboxConfig(max_pending=1))
    store.admit(payload(1))
    with pytest.raises(InboxCapacityError):
        store.admit(payload(2))
    assert len(store.entries()) == len(store.notices()) == 1


def test_earlier_pending_routing_bars_younger_task_claim(sessions: SessionManager) -> None:
    store = inbox(sessions)
    older = store.admit(payload(1)).entry
    younger = store.admit(payload(2)).entry
    store.route(younger.id, younger.revision, ControlDecision("next-turn", active(), "classified"),
                current_run=active(), active_scope=("read",))
    assert store.claim_next_turn("next", current_run=None) is None
    store.route(older.id, older.revision, ControlDecision("clarify", active(), "ambiguous"),
                current_run=active(), active_scope=("read",))
    claimed = store.claim_next_turn("next", current_run=None)
    assert claimed is not None and claimed.id == younger.id


def test_fifo_and_single_claimed_consumer(sessions: SessionManager) -> None:
    store = inbox(sessions)
    for number in range(3):
        entry = store.admit(payload(number)).entry
        store.route(entry.id, 0, ControlDecision("next-turn", None, "idle"), current_run=None, active_scope=())
    assert store.claim_next_turn("blocked", current_run=active()) is None
    first = store.claim_next_turn("next-1", current_run=None)
    assert first is not None and first.seq == 1
    assert inbox(sessions).claim_next_turn("next-2", current_run=None) is None
    dispatched = store.begin_dispatch(first.id, first.revision)
    store.finish(first.id, dispatched.revision, "completed")
    second = store.claim_next_turn("next-2", current_run=None)
    assert second is not None and second.seq == 2


@pytest.mark.parametrize("intent", ["steer", "stop", "stop-and-replace"])
def test_late_proposal_cannot_affect_a_different_generation(sessions: SessionManager, intent: str) -> None:
    store = inbox(sessions)
    entry = store.admit(payload()).entry
    decision = ControlDecision(intent=intent, observed_run=active("old", 1), reason="classified")
    routed = store.route(entry.id, entry.revision, decision, current_run=active("new", 2), active_scope=("read",))
    if intent == "stop":
        assert routed.state == "control" and routed.intent == "status"
        assert store.notices()[-1].stage == "control"
    else:
        assert routed.state == "next-turn" and routed.intent == "next-turn"
        assert store.notices()[-1].stage == "queued"


def test_run_identity_not_queue_count_controls_generation_fence(sessions: SessionManager) -> None:
    store = inbox(sessions)
    entry = store.admit(payload()).entry
    routed = store.route(entry.id, 0, ControlDecision("steer", active(queued=1), "classified"),
                         current_run=active(queued=20), active_scope=("read",))
    assert routed.state == "next-step"
    assert store.claim_steer(active(queued=30)).id == routed.id


def test_steer_cannot_carry_a_less_authorized_message_into_broader_run(sessions: SessionManager) -> None:
    store = inbox(sessions)
    entry = store.admit(payload(scope=("read",))).entry
    routed = store.route(entry.id, 0, ControlDecision("steer", active(), "classified"),
                         current_run=active(), active_scope=("read", "write"))
    assert routed.state == "next-turn"


def test_stale_revision_and_route_reclassification_are_rejected(sessions: SessionManager) -> None:
    store = inbox(sessions)
    entry = store.admit(payload()).entry
    decision = ControlDecision("next-turn", active(), "classified")
    store.route(entry.id, 0, decision, current_run=active(), active_scope=("read",))
    with pytest.raises(InboxConflictError):
        store.route(entry.id, 0, decision, current_run=active(), active_scope=("read",))


def test_restart_does_not_blindly_replay_started_effects(sessions: SessionManager) -> None:
    store = inbox(sessions)
    entry = store.admit(payload()).entry
    store.route(entry.id, 0, ControlDecision("next-turn", None, "idle"), current_run=None, active_scope=())
    claimed = store.claim_next_turn("turn", current_run=None)
    assert claimed is not None
    store.begin_dispatch(claimed.id, claimed.revision)
    restored = inbox(sessions)
    restored.reconcile_orphans(live_run_ids=frozenset())
    assert restored.entries()[0].state == "uncertain"
    assert restored.claim_next_turn("retry", current_run=None) is None


def test_restart_requeues_a_claim_that_could_not_have_dispatched(sessions: SessionManager) -> None:
    store = inbox(sessions)
    entry = store.admit(payload()).entry
    store.route(entry.id, 0, ControlDecision("next-turn", None, "idle"), current_run=None, active_scope=())
    assert store.claim_next_turn("turn", current_run=None) is not None
    restored = inbox(sessions)
    restored.reconcile_orphans(live_run_ids=frozenset())
    assert restored.entries()[0].state == "next-turn"


def test_uncertain_notice_is_not_automatically_resent(sessions: SessionManager) -> None:
    store = inbox(sessions)
    store.admit(payload())
    notice = store.notices()[0]
    store.claim_notice(notice.id)
    restored = inbox(sessions)
    restored.reconcile_orphans(live_run_ids=frozenset())
    assert restored.notices()[0].state == "uncertain"
    with pytest.raises(InboxConflictError):
        restored.claim_notice(notice.id)


def test_generic_metadata_upserts_cannot_modify_inbox_authority(sessions: SessionManager) -> None:
    store = inbox(sessions)
    store.admit(payload())
    before = store.entries()
    sessions.upsert_session_metadata("assistant", {"busy_inbox": "forged"}, removals=("busy_inbox",))
    assert store.entries() == before


def test_corrupt_state_fails_closed_without_publishing_receipt(sessions: SessionManager, monkeypatch: pytest.MonkeyPatch) -> None:
    store = inbox(sessions)
    monkeypatch.setattr(sessions, "read_session_metadata", lambda _key: {"metadata": {"busy_inbox": {"content": "SECRET"}}})
    with pytest.raises(InboxStateError, match="malformed") as error:
        store.admit(payload())
    assert "SECRET" not in str(error.value)


def test_temporary_recovery_uses_disjoint_journal_not_regular_history(sessions: SessionManager) -> None:
    regular = inbox(sessions)
    regular.admit(payload())
    temp = SessionBusyInbox(sessions, owner_id="owner", session_key="temp-session", temporary_lifetime_id="temp-generation")
    temp.admit(payload(key="temp-session", content="temporary secret"))
    recovered = SessionBusyInbox(sessions, owner_id="owner", session_key="temp-session", temporary_lifetime_id="temp-generation")
    assert recovered.entries()[0].payload.content == "temporary secret"
    assert "temporary secret" not in str(sessions.read_session_snapshot("assistant"))
    assert sessions.read_session_snapshot("temp-session") is None
    with pytest.raises(InboxConflictError):
        temp.erase_temporary()


@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_lifecycle_cannot_discard_accepted_work(sessions: SessionManager, operation: str) -> None:
    store = inbox(sessions)
    store.admit(payload())
    with pytest.raises(InboxConflictError):
        if operation == "reset":
            sessions.reset_session("assistant")
        else:
            sessions.delete_session("assistant")
    assert store.entries()[0].payload == payload()


def test_clarification_reenters_fifo_without_changing_ingress_identity(sessions: SessionManager) -> None:
    store = inbox(sessions)
    first = store.admit(payload(1)).entry
    first = store.route(first.id, first.revision, ControlDecision("clarify", active(), "ambiguous"),
                        current_run=active(), active_scope=("read",))
    second = store.admit(payload(2)).entry
    store.route(second.id, second.revision, ControlDecision("next-turn", None, "idle"),
                current_run=None, active_scope=())
    answer = payload(3, channel="slack", content="This is an independent new task.")
    resumed = store.resume_clarification(first.id, first.revision, answer)
    assert resumed.id == first.id and resumed.seq == first.seq
    assert resumed.clarifications == (answer,) and resumed.origin == first.origin
    duplicate = store.resume_clarification(first.id, first.revision, answer)
    assert duplicate == resumed
    store.route(resumed.id, resumed.revision, ControlDecision("next-turn", None, "idle"),
                current_run=None, active_scope=())
    claimed = store.claim_next_turn("next", current_run=None)
    assert claimed is not None and claimed.id == second.id


def test_only_definite_unsent_notices_are_retryable(sessions: SessionManager) -> None:
    store = inbox(sessions)
    store.admit(payload())
    notice = store.notices()[0]
    store.claim_notice(notice.id)
    store.fail_notice(notice.id, definitely_unsent=True)
    assert store.notices()[0].state == "pending"
    store.claim_notice(notice.id)
    store.fail_notice(notice.id, definitely_unsent=False)
    assert store.notices()[0].state == "uncertain"
    with pytest.raises(InboxConflictError):
        store.claim_notice(notice.id)


def test_temporary_erasure_is_durable_and_cannot_resurrect(sessions: SessionManager) -> None:
    options = dict(owner_id="owner", session_key="temp-session", temporary_lifetime_id="lifetime")
    store = SessionBusyInbox(sessions, **options)
    entry = store.admit(payload(key="temp-session", content="temporary secret")).entry
    store.route(entry.id, entry.revision, ControlDecision("next-turn", None, "idle"),
                current_run=None, active_scope=())
    claimed = store.claim_next_turn("temp-run", current_run=None)
    assert claimed is not None
    dispatched = store.begin_dispatch(claimed.id, claimed.revision)
    store.finish(dispatched.id, dispatched.revision, "completed")
    for notice in store.notices():
        store.claim_notice(notice.id)
        store.acknowledge_notice(notice.id)
    store.erase_temporary()
    assert "temporary secret" not in str(sessions.read_session_metadata(store.journal_key))
    reopened = SessionBusyInbox(sessions, **options)
    with pytest.raises(InboxConflictError, match="closed"):
        reopened.admit(payload(2, key="temp-session"))
    with pytest.raises(InboxConflictError):
        sessions.delete_session(store.journal_key)


def test_failed_storage_commit_does_not_leave_an_acceptance_notice(
    sessions: SessionManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = inbox(sessions)

    def fail(_key: str, _expected: object, _replacement: dict[str, object]) -> None:
        raise OSError("injected storage fault")

    with monkeypatch.context() as fault:
        fault.setattr(sessions, "commit_busy_inbox_metadata", fail)
        with pytest.raises(OSError):
            store.admit(payload())
    assert store.entries() == store.notices() == ()
    assert not store.admit(payload()).duplicate


async def test_canonical_admission_does_not_reacquire_active_execution(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
    try:
        store = inbox(lease.manager)
        store.admit(payload(1))
        backend = lease.manager.source_backend("assistant")
        assert backend is not None
        async with acquire_execution(backend, "assistant", run_id="active-run") as owner:
            async def receive() -> None:
                store.admit(payload(2))

            await asyncio.create_task(receive())
            owner.validate_dispatch()
            assert len(store.entries()) == 2
            assert owner.read_snapshot().events == []
    finally:
        assert lease.close is not None
        await lease.close()


@pytest.mark.parametrize("managed", [False, True])
async def test_restart_reads_durable_state_with_a_new_authority(tmp_path: Path, managed: bool) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    if managed:
        first = await DurableSessionAuthority(workspace, sessions_root=root).acquire()
        saved = inbox(first.manager).admit(payload()).entry
        assert first.close is not None
        await first.close()
        second = await DurableSessionAuthority(workspace, sessions_root=root).acquire()
        try:
            assert inbox(second.manager).entries() == (saved,)
            assert inbox(second.manager).admit(payload()).duplicate
        finally:
            assert second.close is not None
            await second.close()
    else:
        saved = inbox(SessionManager(workspace, sessions_root=root)).admit(payload()).entry
        restored = inbox(SessionManager(workspace, sessions_root=root))
        assert restored.entries() == (saved,) and restored.admit(payload()).duplicate


def test_repeated_undispatched_recovery_does_not_grow_reply_intents(sessions: SessionManager) -> None:
    store = inbox(sessions)
    entry = store.admit(payload()).entry
    store.route(entry.id, entry.revision, ControlDecision("next-turn", None, "idle"),
                current_run=None, active_scope=())
    for number in range(12):
        assert store.claim_next_turn(f"orphan-{number}", current_run=None) is not None
        store.reconcile_orphans(live_run_ids=frozenset())
    assert len(store.notices()) == 2


def test_uncertain_work_fences_later_execution(sessions: SessionManager) -> None:
    store = inbox(sessions)
    for number in range(2):
        entry = store.admit(payload(number)).entry
        store.route(entry.id, entry.revision, ControlDecision("next-turn", None, "idle"),
                    current_run=None, active_scope=())
    claimed = store.claim_next_turn("orphan", current_run=None)
    assert claimed is not None
    store.begin_dispatch(claimed.id, claimed.revision)
    store.reconcile_orphans(live_run_ids=frozenset())
    assert store.claim_next_turn("new", current_run=None) is None
    assert store.entries()[1].state == "next-turn"


def test_recovery_journal_is_not_listed_as_a_regular_chat(sessions: SessionManager) -> None:
    store = inbox(sessions)
    store.admit(payload())
    temp = SessionBusyInbox(sessions, owner_id="owner", session_key="temp-session", temporary_lifetime_id="lifetime")
    temp.admit(payload(2, key="temp-session"))
    assert "assistant" in {item["key"] for item in sessions.list_sessions()}
    assert temp.journal_key not in {item["key"] for item in sessions.list_sessions()}


def test_accepted_payloads_have_capacity_for_mandatory_transitions(sessions: SessionManager) -> None:
    store = inbox(sessions, limits=BusyInboxConfig(max_bytes=18000))
    accepted = []
    for number in range(20):
        try:
            accepted.append(store.admit(payload(number, content="text" * 200)).entry)
        except InboxCapacityError:
            break
    assert accepted
    for entry in accepted:
        store.route(entry.id, entry.revision, ControlDecision("next-turn", None, "idle"),
                    current_run=None, active_scope=())
    for number in range(len(accepted)):
        claimed = store.claim_next_turn(f"run-{number}", current_run=None)
        assert claimed is not None
        dispatched = store.begin_dispatch(claimed.id, claimed.revision)
        store.finish(dispatched.id, dispatched.revision, "completed")
    assert all(entry.state == "completed" for entry in store.entries())


def test_clarification_cannot_expand_active_run_authority(sessions: SessionManager) -> None:
    store = inbox(sessions)
    entry = store.admit(payload(1, scope=("read", "write"))).entry
    entry = store.route(entry.id, entry.revision, ControlDecision("clarify", active(), "ambiguous"),
                        current_run=active(), active_scope=("read", "write"))
    entry = store.resume_clarification(entry.id, entry.revision, payload(2, scope=("read",)))
    routed = store.route(entry.id, entry.revision, ControlDecision("steer", active(), "classified"),
                         current_run=active(), active_scope=("read", "write"))
    assert routed.state == "next-turn" and routed.effective_scope == frozenset({"read"})


def test_idle_stop_never_becomes_a_future_task(sessions: SessionManager) -> None:
    store = inbox(sessions)
    entry = store.admit(payload(content="/stop")).entry
    routed = store.route(entry.id, entry.revision, ControlDecision("stop", None, "explicit-stop"),
                         current_run=None, active_scope=())
    assert routed.state == "control" and routed.intent == "status"
    assert store.claim_next_turn("future-run", current_run=None) is None
