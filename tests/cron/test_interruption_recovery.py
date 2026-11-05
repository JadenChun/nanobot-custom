"""Gateway admission recovers one durably fenced fresh cron source."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import nanobot.cron.interruption_recovery as interruption_recovery
import nanobot.cron.runtime as cron_runtime
from nanobot.agent.turn import TurnResult
from nanobot.config.schema import GatewayCronConfig
from nanobot.cron.gateway import GatewayCron
from nanobot.cron.occurrence import destination_policy
from nanobot.cron.service import CronService
from nanobot.session.execution_owner import ExecutionOwner, acquire_execution
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.session.source import new_source_event
from nanobot.session.source_store import SourceSessionStore
from tests.cron.stage3_fixtures import FakeChannels, definition, definitions


async def _interrupted_gateway(tmp_path, *, leave_stale_lease: bool, destinations=()):
    store, item, clock = definitions(tmp_path, item=definition(destinations=destinations))
    service = CronService(tmp_path / "definitions" / "jobs.json", definition_store=store)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    authority = DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions")
    authority_lease = await authority.acquire()
    sessions = authority_lease.manager
    channels = FakeChannels()
    calls = []

    async def compute(request):
        calls.append(request.run_id)
        return TurnResult(run_id=request.run_id, content="future result")

    gateway = GatewayCron(
        GatewayCronConfig(enabled=True), service,
        SimpleNamespace(execute_silent_cron_turn=compute), sessions, channels,
        workspace, tmp_path / "instance", clock=clock,
    )
    key = f"cron:{item.job_id}"
    sessions.admit_persistent_write(key)
    occurrence = service.occurrence_store
    assert occurrence is not None
    old_fire = 1100
    old_claim = occurrence.claim_scheduled(
        job_id=item.job_id,
        job_revision=item.definition_revision,
        scheduled_fire_time_ms=old_fire,
        job_snapshot=item.to_wire(),
        destination_policy=destination_policy(item),
        owner="interrupted-owner",
    )

    backend = sessions._jsonl_store
    initial = backend.source.load_source_snapshot(key)
    prefix = [
        new_source_event({"role": "user", "content": "completed input"}, run_id="completed-turn"),
        new_source_event({"role": "assistant", "content": "completed result"}, run_id="completed-turn"),
    ]
    tail = [
        new_source_event({"role": "user", "content": "interrupted input"}, run_id=old_claim.run_id),
        new_source_event({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "provider-call-1",
                "type": "function",
                "function": {"name": "uncertain_action", "arguments": "{}"},
            }],
        }, run_id=old_claim.run_id),
    ]
    tail[-1]["_execution_ids"] = ["execution-1"]
    backend.source.append_source_events(key, initial.state, [*prefix, *tail])
    with backend.locked_session_files():
        snapshot = backend.source.load_source_snapshot(key)
        if leave_stale_lease:
            SourceSessionStore.publish_execution_lease(
                backend, snapshot, snapshot.header["metadata"],
                token="stale-owner-token", run_id=old_claim.run_id,
                recovery_required=True,
                select_acquired_at=lambda: datetime.now(timezone.utc).isoformat(),
            )
            snapshot = backend.source.load_source_snapshot(key)
        snapshot.header["metadata"]["execution_recovery_required"] = True
        snapshot.header["revision"] += 1
        backend.source._write(snapshot.header, snapshot.events)
    return gateway, authority_lease, sessions, key, old_claim, prefix, tail, calls, clock


async def _healthy_gateway(tmp_path):
    store, item, clock = definitions(tmp_path, item=definition(destinations=()))
    service = CronService(tmp_path / "definitions" / "jobs.json", definition_store=store)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    authority = DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions")
    authority_lease = await authority.acquire()
    sessions = authority_lease.manager
    channels = FakeChannels()
    calls = []

    async def compute(request):
        calls.append(request.run_id)
        return TurnResult(run_id=request.run_id, content="healthy future result")

    gateway = GatewayCron(
        GatewayCronConfig(enabled=True),
        service,
        SimpleNamespace(execute_silent_cron_turn=compute),
        sessions,
        channels,
        workspace,
        tmp_path / "instance",
        clock=clock,
    )
    key = f"cron:{item.job_id}"
    sessions.admit_persistent_write(key)
    snapshot = sessions._jsonl_store.source.load_source_snapshot(key)
    history = [
        new_source_event({"role": "user", "content": "completed input"}, run_id="completed-turn"),
        new_source_event({"role": "assistant", "content": "completed result"}, run_id="completed-turn"),
    ]
    sessions._jsonl_store.source.append_source_events(key, snapshot.state, history)
    return gateway, authority_lease, sessions, key, calls, clock


@pytest.mark.parametrize("leave_stale_lease", [False, True])
async def test_gateway_acquisition_recovers_then_runs_only_a_distinct_occurrence(
    tmp_path, leave_stale_lease: bool,
) -> None:
    (
        gateway, authority_lease, sessions, key, old_claim, prefix, tail, calls, clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=leave_stale_lease)
    second_gateway = None
    try:
        clock.now = 1100
        await gateway.runtime.acquire_for_gateway()

        recovered = sessions._jsonl_store.source.load_source_snapshot(key)
        assert recovered.events == prefix
        assert "execution_recovery_required" not in recovered.header["metadata"]
        assert "execution_lease" not in recovered.header["metadata"]
        assert gateway.service.occurrence_store.get(old_claim.occurrence_id).state == "claimed"
        assert calls == []
        archive_files = list((sessions.sessions_dir / ".session-mode" / "cron-interruptions").glob("*.json"))
        assert len(archive_files) == 1
        archive = json.loads(archive_files[0].read_text())
        assert archive["outcome"] == "unknown"
        assert archive["interrupted_events"] == tail
        assert archive["source_identity"]["run_id"] == old_claim.run_id

        clock.now = 1200
        await gateway.runtime.scheduler.wake_once()
        assert len(calls) == 1
        assert calls[0] != old_claim.run_id
        assert gateway.service.occurrence_store.get(old_claim.occurrence_id).state == "claimed"
        await gateway.runtime.aclose()

        second_calls = []

        async def second_compute(request):
            second_calls.append(request.run_id)
            return TurnResult(run_id=request.run_id, content="second future result")

        second_service = CronService(
            tmp_path / "definitions" / "jobs.json",
            definition_store=gateway.service.definition_store,
        )
        second_gateway = GatewayCron(
            GatewayCronConfig(enabled=True),
            second_service,
            SimpleNamespace(execute_silent_cron_turn=second_compute),
            sessions,
            FakeChannels(),
            tmp_path / "workspace",
            tmp_path / "instance",
            clock=clock,
        )
        assert await second_gateway.runtime.acquire_for_gateway() is True
        assert second_gateway.status()["recovery_required"] == []
        await second_gateway.runtime.scheduler.wake_once()
        assert second_calls == []
        await second_gateway.runtime.aclose()
    finally:
        if second_gateway is not None and second_gateway.runtime._close_task is None:
            await second_gateway.runtime.aclose()
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()


async def test_healthy_completed_cron_source_is_not_recovery_fenced(tmp_path) -> None:
    gateway, authority_lease, sessions, key, calls, clock = await _healthy_gateway(tmp_path)
    try:
        snapshot = sessions._jsonl_store.source.load_source_snapshot(key)
        assert len(snapshot.events) == 2
        assert "execution_recovery_required" not in snapshot.header["metadata"]
        assert "execution_lease" not in snapshot.header["metadata"]

        assert await gateway.runtime.acquire_for_gateway() is True
        assert gateway.status()["recovery_required"] == []
        clock.now = 1100
        await gateway.runtime.scheduler.wake_once()
        assert len(calls) == 1
        assert gateway.status()["recovery_required"] == []
        await gateway.runtime.aclose()
    finally:
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()


async def test_live_occurrence_owner_fences_fresh_recovery_before_source_owner(
    tmp_path, monkeypatch,
) -> None:
    (
        gateway, authority_lease, sessions, key, old_claim, _prefix, _tail, _calls, _clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False)
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    expected = occurrences.get(old_claim.occurrence_id)
    occurrence_path = occurrences.root / f"{old_claim.occurrence_id}.json"
    occurrence_before = occurrence_path.read_bytes()
    source = sessions.read_existing_canonical_source(key)
    assert source is not None
    source_backend, _snapshot = source
    source_path = source_backend.get_session_path(key)
    source_before = source_path.read_bytes()
    owner_attempts = []

    async def forbidden_owner_acquisition(*args, **kwargs):
        owner_attempts.append((args, kwargs))
        raise AssertionError("source owner acquisition ran before the occurrence lock")

    monkeypatch.setattr(interruption_recovery, "acquire_execution", forbidden_owner_acquisition)
    try:
        with occurrences.hold_claimed_for_recovery(expected):
            fences = await interruption_recovery.recover_interrupted_cron_sources(
                sessions, gateway.service, occurrences
            )
        assert fences == {gateway.service.list_jobs()[0].id: "execution_recovery_required"}
        assert owner_attempts == []
        assert source_path.read_bytes() == source_before
        assert occurrence_path.read_bytes() == occurrence_before
        assert not list(
            (sessions.sessions_dir / ".session-mode" / "cron-interruptions").glob("*.json")
        )
    finally:
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()


async def test_live_source_owner_fences_recovery_without_takeover(tmp_path) -> None:
    (
        gateway, authority_lease, sessions, key, old_claim, _prefix, _tail, _calls, _clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False)
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    source = sessions.read_existing_canonical_source(key)
    assert source is not None
    source_backend, _snapshot = source
    owner_attempts = []
    original_acquire = interruption_recovery.acquire_execution

    async def observed_owner_acquisition(*args, **kwargs):
        owner_attempts.append((args, kwargs))
        return await original_acquire(*args, **kwargs)

    try:
        async with acquire_execution(source_backend, key, run_id=old_claim.run_id):
            fences = await interruption_recovery.recover_interrupted_cron_sources(
                sessions, gateway.service, occurrences
            )
        assert fences == {gateway.service.list_jobs()[0].id: "execution_recovery_required"}
        assert owner_attempts == []
        assert not list(
            (sessions.sessions_dir / ".session-mode" / "cron-interruptions").glob("*.json")
        )
        assert occurrences.get(old_claim.occurrence_id).state == "claimed"
    finally:
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()


@pytest.mark.parametrize("unqualified", ["false_marker", "checkpoint", "foreign_occurrence"])
async def test_unqualified_or_checkpointed_source_stays_fenced(
    tmp_path, unqualified: str,
) -> None:
    (
        gateway, authority_lease, sessions, key, old_claim, _prefix, _tail, calls, _clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False)
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    backend = sessions._jsonl_store
    try:
        if unqualified == "false_marker":
            with backend.locked_session_files():
                snapshot = backend.source.load_source_snapshot(key)
                snapshot.header["metadata"]["execution_recovery_required"] = False
                snapshot.header["revision"] += 1
                backend.source._write(snapshot.header, snapshot.events)
        elif unqualified == "checkpoint":
            with backend.locked_session_files():
                backend.get_runtime_checkpoint_path(key).write_text("{}", encoding="utf-8")
        else:
            expected = occurrences.get(old_claim.occurrence_id)
            altered_snapshot = dict(expected.job_snapshot)
            altered_snapshot["enabled"] = False
            record = occurrences._wire_record(
                occurrence_id=expected.occurrence_id,
                identity=expected.identity,
                state=expected.state,
                owner=expected.owner,
                generation=expected.generation,
                job_snapshot=altered_snapshot,
                destination_policy=expected.destination_policy,
                completion=expected.completion,
                reconciliation=expected.reconciliation,
            )
            (occurrences.root / f"{old_claim.occurrence_id}.json").write_bytes(record)

        fences = await interruption_recovery.recover_interrupted_cron_sources(
            sessions, gateway.service, occurrences
        )
        assert fences == {gateway.service.list_jobs()[0].id: "execution_recovery_required"}
        assert calls == []
        assert occurrences.get(old_claim.occurrence_id).state == "claimed"
        assert not list(
            (sessions.sessions_dir / ".session-mode" / "cron-interruptions").glob("*.json")
        )
    finally:
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()


async def test_unrelated_mode_transition_is_not_reconciled_for_cron_recovery(tmp_path) -> None:
    (
        gateway, authority_lease, sessions, key, old_claim, prefix, _tail, calls, _clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False)
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    resolver = sessions._mode_resolver
    assert resolver is not None
    source = sessions.read_existing_canonical_source(key)
    assert source is not None
    source_backend, _snapshot = source
    try:
        async with acquire_execution(source_backend, key, run_id=old_claim.run_id) as owner:
            await owner.quiesce()
            with pytest.raises(RuntimeError, match="injected session-mode fault"):
                resolver.reset(
                    key,
                    prefix,
                    owner=owner,
                    fault_after="prepared",
                )

        fences = await interruption_recovery.recover_interrupted_cron_sources(
            sessions, gateway.service, occurrences
        )
        assert fences == {gateway.service.list_jobs()[0].id: "execution_recovery_required"}
        assert calls == []
        assert occurrences.get(old_claim.occurrence_id).state == "claimed"
        assert not list(
            (sessions.sessions_dir / ".session-mode" / "cron-interruptions").glob("*.json")
        )
    finally:
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()


@pytest.mark.parametrize(
    ("phase", "abrupt_cleanup"),
    [
        ("prepared", False),
        ("prepared", True),
        ("source_installed", False),
        ("target_installed", False),
        ("mode_record_published", False),
        ("mode_committed", False),
    ],
)
async def test_pending_archive_bound_reset_reconciles_each_crash_window(
    tmp_path, monkeypatch, phase, abrupt_cleanup: bool,
) -> None:
    (
        gateway, authority_lease, sessions, key, old_claim, prefix, tail, calls, _clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False)
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    occurrence_before = (occurrences.root / f"{old_claim.occurrence_id}.json").read_bytes()

    try:
        await _stop_recovery_after_reset_phase(
            gateway,
            sessions,
            occurrences,
            monkeypatch,
            phase=phase,
            abrupt_cleanup=abrupt_cleanup,
        )

        reference = sessions.peek_cron_interruption_reset_occurrence(key)
        assert reference == (old_claim.occurrence_id, old_claim.run_id)
        interrupted = sessions._jsonl_store.source.load_source_snapshot(key)
        assert interrupted.events == [*prefix, *tail] or interrupted.events == prefix
        if abrupt_cleanup:
            lease = interrupted.header["metadata"].get("execution_lease")
            assert isinstance(lease, dict) and lease["run_id"] == old_claim.run_id
            assert lease["recovery_required"] is True
        elif phase == "prepared":
            assert "execution_lease" not in interrupted.header["metadata"]
            assert interrupted.header["metadata"].get("execution_recovery_required") is True

        assert await interruption_recovery.recover_interrupted_cron_sources(
            sessions, gateway.service, occurrences
        ) == {}
        recovered = sessions._jsonl_store.source.load_source_snapshot(key)
        assert recovered.events == prefix
        assert "execution_lease" not in recovered.header["metadata"]
        assert "execution_recovery_required" not in recovered.header["metadata"]
        assert occurrences.get(old_claim.occurrence_id).state == "claimed"
        assert (occurrences.root / f"{old_claim.occurrence_id}.json").read_bytes() == occurrence_before
        assert calls == []
        archives = list(
            (sessions.sessions_dir / ".session-mode" / "cron-interruptions").glob("*.json")
        )
        assert len(archives) == 1
        archive = json.loads(archives[0].read_text())
        assert archive["outcome"] == "unknown"
        assert archive["interrupted_events"] == tail
        assert archive["source_identity"]["run_id"] == old_claim.run_id
        assert sessions.peek_cron_interruption_reset_occurrence(key) is None
    finally:
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()


async def _stop_recovery_after_reset_phase(
    gateway,
    sessions,
    occurrences,
    monkeypatch,
    *,
    phase: str,
    abrupt_cleanup: bool = False,
) -> None:
    resolver = sessions._mode_resolver
    assert resolver is not None

    def stop_at_reset_phase(
        reset_key,
        *,
        owner,
        replacement_events,
        cron_interruption_archive_identity,
        cron_interruption_archive_digest,
    ):
        return resolver.reset(
            reset_key,
            replacement_events,
            owner=owner,
            fault_after=phase,
            archive_identity=cron_interruption_archive_identity,
            archive_digest=cron_interruption_archive_digest,
        )

    with monkeypatch.context() as patch:
        patch.setattr(sessions, "reset_session", stop_at_reset_phase)
        if abrupt_cleanup:

            async def skip_owner_cleanup(_owner):
                return None

            patch.setattr(ExecutionOwner, "aclose", skip_owner_cleanup)
        with pytest.raises(RuntimeError, match="injected session-mode fault"):
            await interruption_recovery.recover_interrupted_cron_sources(
                sessions, gateway.service, occurrences
            )


async def test_live_occurrence_owner_fences_pending_reset_before_source_reconcile(
    tmp_path, monkeypatch,
) -> None:
    (
        gateway, authority_lease, sessions, key, old_claim, _prefix, _tail, _calls, _clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False)
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    resolver = sessions._mode_resolver
    assert resolver is not None
    source = sessions.read_existing_canonical_source(key)
    assert source is not None
    source_backend, _snapshot = source
    source_path = source_backend.get_session_path(key)
    try:
        await _stop_recovery_after_reset_phase(
            gateway,
            sessions,
            occurrences,
            monkeypatch,
            phase="prepared",
        )
        assert sessions.peek_cron_interruption_reset_occurrence(key) == (
            old_claim.occurrence_id,
            old_claim.run_id,
        )
        expected = occurrences.get(old_claim.occurrence_id)
        occurrence_path = occurrences.root / f"{old_claim.occurrence_id}.json"
        occurrence_before = occurrence_path.read_bytes()
        source_before = source_path.read_bytes()
        journal_paths = sorted(
            path
            for path in (sessions.sessions_dir / ".session-mode" / "journals").rglob("*")
            if path.is_file()
        )
        journal_before = {path.relative_to(sessions.sessions_dir).as_posix(): path.read_bytes()
                          for path in journal_paths}
        archive_paths = sorted(
            (sessions.sessions_dir / ".session-mode" / "cron-interruptions").glob("*.json")
        )
        archive_before = {path.name: path.read_bytes() for path in archive_paths}
        owner_attempts = []
        reconcile_attempts = []
        original_reconcile = sessions.recover_cron_interruption_reset

        async def forbidden_owner_acquisition(*args, **kwargs):
            owner_attempts.append((args, kwargs))
            raise AssertionError("source owner acquisition ran before the occurrence lock")

        def observed_reconcile(*args, **kwargs):
            reconcile_attempts.append((args, kwargs))
            return original_reconcile(*args, **kwargs)

        monkeypatch.setattr(interruption_recovery, "acquire_execution", forbidden_owner_acquisition)
        monkeypatch.setattr(sessions, "recover_cron_interruption_reset", observed_reconcile)
        with occurrences.hold_claimed_for_recovery(expected):
            fences = await interruption_recovery.recover_interrupted_cron_sources(
                sessions, gateway.service, occurrences
            )

        assert fences == {gateway.service.list_jobs()[0].id: "execution_recovery_required"}
        assert owner_attempts == []
        assert reconcile_attempts == []
        assert source_path.read_bytes() == source_before
        assert occurrence_path.read_bytes() == occurrence_before
        assert {
            path.relative_to(sessions.sessions_dir).as_posix(): path.read_bytes()
            for path in journal_paths
        } == journal_before
        assert {path.name: path.read_bytes() for path in archive_paths} == archive_before
    finally:
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()


@pytest.mark.parametrize("shutdown", [False, True])
async def test_gateway_keeps_ingress_fenced_until_recovery_finishes(tmp_path, monkeypatch, shutdown):
    (
        gateway, authority_lease, _sessions, _key, _old, _prefix, _tail, _calls, _clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False)
    recovery_started, finish_recovery = asyncio.Event(), asyncio.Event()

    async def wait_for_recovery(_sessions, _service, _occurrences):
        recovery_started.set()
        await finish_recovery.wait()
        return {}

    monkeypatch.setattr(cron_runtime, "recover_interrupted_cron_sources", wait_for_recovery)
    runner = asyncio.create_task(gateway.run())
    try:
        async with asyncio.timeout(3):
            await recovery_started.wait()
        assert gateway.runtime._accepting is False
        assert gateway.runtime._timer_task is None
        assert gateway.status()["state"] == "waiting_for_channels"
        if shutdown:
            gateway.force_shutdown()
            finish_recovery.set()
            await runner
            assert gateway.runtime._accepting is False
            assert gateway.status()["state"] == "stopping"
        else:
            finish_recovery.set()
            async with asyncio.timeout(3):
                while not gateway.runtime._accepting:
                    await asyncio.sleep(0)
            assert gateway.runtime._timer_task is not None
            await gateway.aclose()
            await asyncio.gather(runner, return_exceptions=True)
    finally:
        if not runner.done():
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()


async def test_gateway_cancellation_during_recovery_releases_scheduler_lease(tmp_path, monkeypatch):
    (
        gateway, authority_lease, _sessions, _key, _old, _prefix, _tail, _calls, _clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False)
    recovery_started = asyncio.Event()

    async def wait_forever(_sessions, _service, _occurrences):
        recovery_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(cron_runtime, "recover_interrupted_cron_sources", wait_forever)
    runner = asyncio.create_task(gateway.run())
    try:
        async with asyncio.timeout(3):
            await recovery_started.wait()
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
        assert gateway.runtime._accepting is False
        assert gateway.runtime._startup_task is None
        lease = gateway.runtime.scheduler.store.acquire_lease()
        lease.release()
    finally:
        if not runner.done():
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
        if gateway.runtime._close_task is None:
            gateway.runtime.scheduler.release_lease()
        if authority_lease.close is not None:
            await authority_lease.close()
