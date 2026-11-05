"""Local refusal provenance and evidence windows use copied state and fake effects."""

from __future__ import annotations

import asyncio
import json
import shutil
from copy import deepcopy
from types import SimpleNamespace

import pytest

import nanobot.cron.interruption_recovery as interruption_recovery
from nanobot.config.schema import GatewayCronConfig
from nanobot.cron.definition_store import CronDefinition, CronDefinitionAction
from nanobot.cron.gateway import GatewayCron
from nanobot.cron.job_hooks import CronHookOutcome
from nanobot.cron.service import CronService
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.session.session_mode import SessionModeError, SessionModeRecordError
from tests.cron.stage3_fixtures import FakeChannels, definition
from tests.cron.test_interruption_recovery import _interrupted_gateway


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _namespace_snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


class _FakeEffects:
    def __init__(self, gateway):
        self.runs = {name: [] for name in ("composite", "provider", "tool", "verify", "after_delivery")}
        compute = gateway.runtime.agent.execute_silent_cron_turn

        async def observed_compute(request):
            for name in ("composite", "provider", "tool"):
                self.runs[name].append(request.run_id)
            return await compute(request)

        async def verify(claim, _turn):
            self.runs["verify"].append(claim.run_id)
            return True, "fake_verified"

        async def after_delivery(claim, *_args, **_kwargs):
            self.runs["after_delivery"].append(claim.run_id)
            return CronHookOutcome(None)

        gateway.runtime.agent = SimpleNamespace(execute_silent_cron_turn=observed_compute)
        gateway.runtime.job_hooks = SimpleNamespace(verify=verify, after_delivery=after_delivery)

    def assert_empty(self):
        assert self.runs == {name: [] for name in self.runs}


def _add_healthy_cron(gateway, sessions):
    data = deepcopy(definition(destinations=("HEALTHY",)).definition)
    data["id"] = "healthy"
    data["payload"]["sessionKey"] = "cron:healthy"
    item = CronDefinition.from_definition(data, job_instance_id="b" * 64)
    store = gateway.service.definition_store
    store.append(action_id="healthy-create", action=CronDefinitionAction.create(item))
    store.checkpoint(store.replay())
    sessions.admit_persistent_write("cron:healthy")


async def _close(gateway, authority_lease):
    if gateway.runtime._close_task is None:
        gateway.runtime.scheduler.release_lease()
    if authority_lease.close is not None:
        await authority_lease.close()


@pytest.mark.parametrize("damage", ["parse", "shape", "key_binding"])
async def test_local_mode_record_damage_fences_only_its_cron_job(tmp_path, damage):
    (
        gateway, lease, sessions, key, old, _prefix, _tail, calls, clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False, destinations=("A",))
    _add_healthy_cron(gateway, sessions)
    effects = _FakeEffects(gateway)
    resolver = sessions._mode_resolver
    assert resolver is not None
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    record_path = sessions.sessions_dir / resolver._record_name(key)
    if damage == "parse":
        record_path.write_bytes(b"invalid-json\n")
    elif damage == "shape":
        record_path.write_bytes(b"{}\n")
    elif damage == "key_binding":
        record = json.loads(record_path.read_bytes())
        record["session_key"] = "cron:foreign"
        record_path.write_bytes(_canonical(record))
    namespace_before = _namespace_snapshot(sessions.sessions_dir)
    occurrence_path = occurrences.root / f"{old.occurrence_id}.json"
    occurrence_before = occurrence_path.read_bytes()
    try:
        assert await gateway.runtime.acquire_for_gateway() is True
        assert gateway.status()["recovery_required"] == [
            {"job_id": "job", "reason": "execution_recovery_required"}
        ]
        assert _namespace_snapshot(sessions.sessions_dir) == namespace_before
        assert occurrence_path.read_bytes() == occurrence_before
        effects.assert_empty()
        assert gateway.channels.calls == []
        with pytest.raises(SessionModeError) as refused:
            sessions.resolve_session_mode(key)
        assert type(refused.value).__name__ == "SessionModeRecordError"
        assert refused.value.session_key == key

        for fire in (1100, 1200):
            clock.now = fire
            await gateway.runtime.scheduler.wake_once()
            await gateway.runtime.scheduler.wake_once()
        healthy_runs = {
            f"cron-occurrence-{record.occurrence_id}-{record.generation}"
            for record in occurrences.inspect_records()
            if record.identity.job_id == "healthy"
        }
        assert len(healthy_runs) == 2 and set(calls) == healthy_runs
        assert all(len(runs) == 2 and set(runs) == healthy_runs for runs in effects.runs.values())
        assert gateway.channels.calls == ["HEALTHY", "HEALTHY"]
        assert occurrence_path.read_bytes() == occurrence_before
        assert occurrences.get(old.occurrence_id).state == "claimed"
        skipped = [record for record in occurrences.inspect_records()
                   if record.identity.job_id == "job" and record.occurrence_id != old.occurrence_id]
        assert len(skipped) == 1
        assert skipped[0].completion["reason"] == "execution_recovery_required"
        await gateway.runtime.aclose()
    finally:
        await _close(gateway, lease)


async def _install_valid_foreign_journal(tmp_path, sessions):
    workspace = tmp_path / "foreign-workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "foreign-sessions").acquire()
    try:
        lease.manager.admit_persistent_write("cli:foreign")
        journals = lease.manager.sessions_dir / ".session-mode" / "journals"
        directory = next(path.parent for path in journals.rglob("0001-prepared.json")
                         if json.loads(path.read_bytes())["target_key"] == "cli:foreign")
        shutil.copytree(directory, sessions.sessions_dir / ".session-mode" / "journals" / directory.name)
    finally:
        if lease.close is not None:
            await lease.close()


@pytest.mark.parametrize("damage", [
    "registry", "workspace_witness", "unrelated_journal", "invalid_operation_id",
    "foreign_registry_journal", "inventory", "unrelated_source", "missing_record",
    "missing_records_directory", "storage_read",
    "storage_oserror", "unexpected",
])
async def test_shared_mode_or_storage_failure_remains_a_visible_acquisition_failure(
    tmp_path, monkeypatch, damage,
):
    gateway, lease, sessions, key, old, *_rest = await _interrupted_gateway(
        tmp_path, leave_stale_lease=False, destinations=("A",)
    )
    effects = _FakeEffects(gateway)
    sessions.admit_persistent_write("cli:unrelated")
    resolver = sessions._mode_resolver
    assert resolver is not None
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    expected_error = SessionModeError
    if damage == "registry":
        (sessions.sessions_dir / ".session-mode" / "registry.json").write_bytes(b"{}\n")
    elif damage == "workspace_witness":
        (sessions.workspace / ".nanobot" / "session-mode-binding.json").write_bytes(b"{}\n")
    elif damage == "unrelated_journal":
        path = next(path for path in (sessions.sessions_dir / ".session-mode" / "journals").rglob("0001-prepared.json")
                    if json.loads(path.read_bytes())["target_key"] == "cli:unrelated")
        path.write_bytes(b"{}\n")
    elif damage == "invalid_operation_id":
        (sessions.sessions_dir / ".session-mode" / "journals" / "invalid-operation").mkdir(mode=0o700)
    elif damage == "foreign_registry_journal":
        await _install_valid_foreign_journal(tmp_path, sessions)
    elif damage == "inventory":
        (sessions.sessions_dir / "invalid-source-entry").write_bytes(b"invalid\n")
    elif damage == "unrelated_source":
        sessions._jsonl_store.get_session_path("cli:unrelated").write_bytes(b"invalid-json\n")
    elif damage in {"missing_record", "missing_records_directory"}:
        _add_healthy_cron(gateway, sessions)
        if damage == "missing_record":
            (sessions.sessions_dir / resolver._record_name(key)).unlink()
        else:
            records = sessions.sessions_dir / ".session-mode" / "records"
            records.rename(records.with_name("records-unavailable"))
    elif damage in {"storage_read", "storage_oserror"}:
        original_read = resolver.storage.read
        expected_error = SessionModeError if damage == "storage_read" else PermissionError

        def unavailable(name, **kwargs):
            if name == resolver._record_name(key):
                raise expected_error("injected storage unavailability")
            return original_read(name, **kwargs)

        monkeypatch.setattr(resolver.storage, "read", unavailable)
    else:
        expected_error = RuntimeError

        def unexpected(*_args, **_kwargs):
            raise RuntimeError("injected unexpected failure")

        monkeypatch.setattr(resolver, "_validate_record", unexpected)

    namespace_before = _namespace_snapshot(sessions.sessions_dir)
    occurrence_path = occurrences.root / f"{old.occurrence_id}.json"
    occurrence_before = occurrence_path.read_bytes()
    try:
        with pytest.raises(expected_error) as error:
            await gateway.runtime.acquire_for_gateway()
        if damage in {"missing_record", "missing_records_directory"}:
            assert type(error.value) is SessionModeError
        assert gateway.runtime._accepting is False
        assert gateway.status()["recovery_required"] == []
        assert _namespace_snapshot(sessions.sessions_dir) == namespace_before
        assert occurrence_path.read_bytes() == occurrence_before
        effects.assert_empty()
        assert gateway.channels.calls == []
        scheduler_lease = gateway.runtime.scheduler.store.acquire_lease()
        scheduler_lease.release()
    finally:
        await _close(gateway, lease)


@pytest.mark.parametrize("seam", ["peek", "read", "reread", "reconcile", "inventory"])
async def test_foreign_key_record_refusal_is_not_consumed_by_current_cron_job(
    tmp_path, monkeypatch, seam,
):
    gateway, lease, sessions, _key, old, *_rest = await _interrupted_gateway(
        tmp_path, leave_stale_lease=False, destinations=("A",)
    )
    effects = _FakeEffects(gateway)
    resolver = sessions._mode_resolver
    assert resolver is not None
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    foreign_error = SessionModeRecordError("cli:unrelated")

    def refuse(*_args, **_kwargs):
        raise foreign_error

    if seam == "peek":
        monkeypatch.setattr(sessions, "peek_cron_interruption_reset_occurrence", refuse)
    elif seam == "read":
        monkeypatch.setattr(sessions, "read_existing_canonical_source", refuse)
    elif seam == "reconcile":
        monkeypatch.setattr(sessions, "recover_cron_interruption_reset", refuse)
    elif seam == "inventory":
        monkeypatch.setattr(resolver, "_inventory_for_key", refuse)
    else:
        original_read = sessions.read_existing_canonical_source
        reads = 0

        def reread(*args, **kwargs):
            nonlocal reads
            reads += 1
            return original_read(*args, **kwargs) if reads == 1 else refuse()

        monkeypatch.setattr(sessions, "read_existing_canonical_source", reread)

    namespace_before = _namespace_snapshot(sessions.sessions_dir)
    occurrence_path = occurrences.root / f"{old.occurrence_id}.json"
    occurrence_before = occurrence_path.read_bytes()
    try:
        with pytest.raises(SessionModeRecordError) as error:
            await gateway.runtime.acquire_for_gateway()
        assert error.value is foreign_error
        assert gateway.runtime._accepting is False
        assert gateway.status()["recovery_required"] == []
        assert _namespace_snapshot(sessions.sessions_dir) == namespace_before
        assert occurrence_path.read_bytes() == occurrence_before
        effects.assert_empty()
        assert gateway.channels.calls == []
        scheduler_lease = gateway.runtime.scheduler.store.acquire_lease()
        scheduler_lease.release()
    finally:
        await _close(gateway, lease)


@pytest.mark.parametrize("window", ["before_archive_publication", "after_archive_before_journal"])
@pytest.mark.parametrize("interruption", [False, True])
async def test_archive_publication_windows_retry_without_loss_or_double_reset(
    tmp_path, monkeypatch, window, interruption,
):
    (
        gateway, lease, sessions, key, old, prefix, tail, _calls, _clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False, destinations=("A",))
    effects = _FakeEffects(gateway)
    resolver = sessions._mode_resolver
    assert resolver is not None
    occurrences = gateway.service.occurrence_store
    assert occurrences is not None
    occurrence_path = occurrences.root / f"{old.occurrence_id}.json"
    occurrence_before = occurrence_path.read_bytes()
    initial = sessions._jsonl_store.source.load_source_snapshot(key)
    archive_dir = sessions.sessions_dir / ".session-mode" / "cron-interruptions"
    archived_bytes = None
    expected_error = asyncio.CancelledError if interruption else RuntimeError

    def stop():
        raise expected_error("injected archive publication interruption")

    original_write = resolver.storage.write

    def stop_before_archive(name, data, **kwargs):
        if name.startswith(".session-mode/cron-interruptions/"):
            stop()
        return original_write(name, data, **kwargs)

    def stop_before_journal(*_args, **_kwargs):
        stop()

    try:
        for _attempt in range(2):
            with monkeypatch.context() as patch:
                if window == "before_archive_publication":
                    patch.setattr(resolver.storage, "write", stop_before_archive)
                else:
                    patch.setattr(sessions, "reset_session", stop_before_journal)
                with pytest.raises(expected_error):
                    await gateway.runtime.acquire_for_gateway()
            snapshot = sessions._jsonl_store.source.load_source_snapshot(key)
            assert snapshot.events == [*prefix, *tail]
            assert snapshot.state.generation == initial.state.generation
            assert snapshot.header["metadata"].get("execution_recovery_required") is True
            assert "execution_lease" not in snapshot.header["metadata"]
            assert sessions.peek_cron_interruption_reset_occurrence(key) is None
            archives = list(archive_dir.glob("*.json"))
            assert len(archives) == (0 if window == "before_archive_publication" else 1)
            if archives:
                current_bytes = archives[0].read_bytes()
                if archived_bytes is not None:
                    assert current_bytes == archived_bytes
                archived_bytes = current_bytes
            effects.assert_empty()
            assert gateway.channels.calls == []
            assert occurrence_path.read_bytes() == occurrence_before
            scheduler_lease = gateway.runtime.scheduler.store.acquire_lease()
            scheduler_lease.release()

        assert await gateway.runtime.acquire_for_gateway() is True
        recovered = sessions._jsonl_store.source.load_source_snapshot(key)
        assert recovered.events == prefix
        assert recovered.state.generation == initial.state.generation + 1
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) == 1
        if archived_bytes is not None:
            assert archives[0].read_bytes() == archived_bytes
        archive = json.loads(archives[0].read_bytes())
        assert archive["outcome"] == "unknown" and archive["interrupted_events"] == tail
        for _retry in range(2):
            assert await interruption_recovery.recover_interrupted_cron_sources(
                sessions, gateway.service, occurrences
            ) == {}
            assert sessions._jsonl_store.source.load_source_snapshot(key).state == recovered.state
        effects.assert_empty()
        assert gateway.channels.calls == []
        assert occurrence_path.read_bytes() == occurrence_before
        await gateway.runtime.aclose()
    finally:
        await _close(gateway, lease)


async def test_recovery_and_restart_do_not_admit_old_fake_effects_but_future_runs_once(tmp_path):
    (
        gateway, lease, sessions, _key, old, _prefix, _tail, calls, clock,
    ) = await _interrupted_gateway(tmp_path, leave_stale_lease=False, destinations=("A",))
    effects = _FakeEffects(gateway)
    second = None
    try:
        clock.now = 1100
        assert await gateway.runtime.acquire_for_gateway() is True
        effects.assert_empty()
        assert gateway.channels.calls == []
        clock.now = 1200
        await gateway.runtime.scheduler.wake_once()
        assert len(calls) == 1 and calls[0] != old.run_id
        assert all(runs == calls for runs in effects.runs.values())
        assert gateway.channels.calls == ["A"]
        await gateway.runtime.aclose()

        service = CronService(tmp_path / "definitions" / "jobs.json",
                              definition_store=gateway.service.definition_store)
        second = GatewayCron(GatewayCronConfig(enabled=True), service, gateway.runtime.agent,
                             sessions, FakeChannels(), tmp_path / "workspace",
                             tmp_path / "instance", clock=clock)
        second.runtime.job_hooks = gateway.runtime.job_hooks
        assert await second.runtime.acquire_for_gateway() is True
        await second.runtime.scheduler.wake_once()
        assert len(calls) == 1
        assert all(runs == calls for runs in effects.runs.values())
        assert second.channels.calls == []
        assert service.occurrence_store.get(old.occurrence_id).state == "claimed"
        await second.runtime.aclose()
    finally:
        if second is not None and second.runtime._close_task is None:
            second.runtime.scheduler.release_lease()
        await _close(gateway, lease)
