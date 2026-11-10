import pytest

from nanobot.cron.scheduler_store import SchedulerConflictError
from tests.cron.stage3_fixtures import offline_denial, runtime  # noqa: F401


async def prebegin_crash(tmp_path, monkeypatch):
    graph, item, clock, calls, channels = runtime(tmp_path)
    dispatch = graph.scheduler._dispatch
    clock.now = 1100

    async def crash(*_):
        raise OSError("before begin")

    with monkeypatch.context() as injection:
        injection.setattr(graph.scheduler, "_dispatch", crash)
        with pytest.raises(OSError):
            await graph.scheduler.wake_once()
    graph.stop_claiming()
    graph.scheduler.release_lease()
    assert graph.scheduler._dispatch == dispatch
    return graph, item, clock, calls, channels


def reconcile(graph, record, *, successor="operator-successor", **changes):
    from nanobot.cron.runtime_inspection import reconcile_occurrence

    return reconcile_occurrence(graph.service.occurrence_store.root, namespace="stage3",
        occurrence_id=record.occurrence_id, expected_owner=record.owner,
        expected_generation=record.generation, resolution="confirmed_not_started",
        external_evidence="EXTERNAL-EVIDENCE-VERIFIED", successor_owner=successor,
        scheduler_root=graph.store.root, runtime_identity="copied-root", **changes)


@pytest.mark.parametrize("fresh", [False, True])
async def test_prebegin_reconciliation_resumes_future_without_old_compute(tmp_path, monkeypatch, fresh):
    from nanobot.cron.scheduler import Scheduler
    from nanobot.cron.scheduler_store import SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    before = graph.store.read()
    reconcile(graph, original)
    if fresh:
        graph.store = SchedulerStore(graph.store.root, namespace="stage3", runtime_identity="copied-root")
        graph.scheduler = Scheduler(graph.service.definition_store, graph.service.occurrence_runtime,
            graph.store, clock, graph._dispatch)
    clock.now = 1200
    graph.acquire()
    state = graph.store.read()
    assert state["classifications"][:len(before["classifications"])] == before["classifications"]
    assert len(state["bridges"]) == 1 and state["cursors"][0]["next_fire"] == 1300
    await graph.scheduler.wake_once()
    assert calls == [] and channels.calls == []
    clock.now = 1300
    await graph.scheduler.wake_once()
    assert len(calls) == 1 and channels.calls == ["A", "B"]
    assert {r.identity.scheduled_fire_time_ms for r in graph.service.occurrence_store.inspect_records()} == {1100, 1300}
    await graph.aclose()


async def test_repeated_reconciliation_retains_every_edge_and_original_observations(tmp_path, monkeypatch):

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    before = graph.store.read()
    reconcile(graph, original, successor="second")
    second = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, second, successor="third")
    # Inspection does not refresh the dormant owner's pinned state or publish.
    from nanobot.cron.runtime_inspection import inspect_scheduler

    inspected = inspect_scheduler(graph.store.root, namespace="stage3", runtime_identity="copied-root")
    assert inspected["classifications"] == before["classifications"]
    clock.now = 1500
    graph.acquire()
    state = graph.store.read()
    assert [(b["predecessor_generation"], b["successor_generation"]) for b in state["bridges"]] == [(1, 2), (2, 3)]
    assert state["bridges"][0]["successor_owner"] == state["bridges"][1]["predecessor_owner"] == "second"
    assert state["classifications"][:len(before["classifications"])] == before["classifications"]
    assert state["cursors"][0]["next_fire"] == 1600
    assert calls == [] and channels.calls == []
    await graph.aclose()


async def test_bridge_capacity_rolls_epoch_and_resumes_timers(tmp_path, monkeypatch):
    from copy import deepcopy

    import nanobot.cron.scheduler_store as scheduler_store
    from nanobot.cron.scheduler_store import SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, original, successor="second")
    before = SchedulerStore(
        graph.store.root, namespace="stage3", runtime_identity="copied-root"
    ).read()
    assert len(before["bridges"]) == 1
    cursors = deepcopy(before["cursors"])
    classifications = deepcopy(before["classifications"])
    requests = deepcopy(before["requests"])

    monkeypatch.setattr(scheduler_store, "MAX_BRIDGES", 1)
    current = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, current, successor="third")

    rolled = SchedulerStore(
        graph.store.root, namespace="stage3", runtime_identity="copied-root"
    ).read()
    assert rolled["kind"] == "cron_scheduler_state_v3"
    assert rolled["epoch"] == 1
    assert rolled["predecessor"]["epoch"] == 0
    assert rolled["predecessor"]["bridge_count"] == 1
    assert rolled["predecessor"]["archive"] == "epoch-0.json"
    assert (graph.store.root / ".epochs" / "epoch-0.json").is_file()
    assert rolled["cursors"] == cursors
    assert rolled["classifications"] == classifications
    assert rolled["requests"] == requests
    assert len(rolled["bridges"]) == 1
    assert rolled["generation"] == before["generation"] + 2

    fresh = SchedulerStore(
        graph.store.root, namespace="stage3", runtime_identity="copied-root"
    )
    assert fresh.read()["epoch"] == 1
    graph.acquire()
    resumed = graph.store.read()
    assert resumed["epoch"] == 2
    next_fire = next(
        item["next_fire"] for item in resumed["cursors"] if item["next_fire"] is not None
    )
    clock.now = next_fire
    await graph.scheduler.wake_once()
    assert calls or channels.calls
    await graph.aclose()


async def test_epoch_archive_tamper_fails_closed(tmp_path, monkeypatch):
    from nanobot.cron.occurrence_store import OccurrenceCorruptError
    from nanobot.cron.scheduler_store import SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, original, successor="second")
    import nanobot.cron.scheduler_store as scheduler_store

    monkeypatch.setattr(scheduler_store, "MAX_BRIDGES", 1)
    current = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, current, successor="third")
    archive = graph.store.root / ".epochs" / "epoch-0.json"
    assert archive.is_file()
    archive.write_bytes(archive.read_bytes()[:-1])

    with pytest.raises(OccurrenceCorruptError):
        SchedulerStore(
            graph.store.root, namespace="stage3", runtime_identity="copied-root"
        ).read()
    await graph.aclose()


async def test_epoch_rollover_survives_crash_before_state_publication(
    tmp_path, monkeypatch
):
    from nanobot.cron.scheduler_store import SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, original, successor="second")

    store = SchedulerStore(
        graph.store.root, namespace="stage3", runtime_identity="copied-root"
    )
    original_cas = SchedulerStore.compare_and_swap

    def crash(*args, **kwargs):
        raise RuntimeError("injected after archive")

    with store.stopped_writer() as lease:
        monkeypatch.setattr(SchedulerStore, "compare_and_swap", crash)
        with pytest.raises(RuntimeError, match="injected"):
            store.rollover_epoch(lease)
        archive = store.root / ".epochs" / "epoch-0.json"
        assert archive.is_file()
        before = SchedulerStore(
            store.root, namespace="stage3", runtime_identity="copied-root"
        ).read()
        assert before["kind"] == "cron_scheduler_state_v2"
        assert before["bridges"]

        (archive.parent / ".epoch-0.json.dead.tmp").write_text("residue")
        monkeypatch.setattr(SchedulerStore, "compare_and_swap", original_cas)
        successor = store.rollover_epoch(lease)
        assert successor["epoch"] == 1
        assert successor["predecessor"]["archive"] == "epoch-0.json"
    assert calls == [] and channels.calls == []
    await graph.aclose()


async def test_epoch_missing_archive_fails_closed(tmp_path, monkeypatch):
    import nanobot.cron.scheduler_store as scheduler_store
    from nanobot.cron.occurrence_store import OccurrenceCorruptError
    from nanobot.cron.scheduler_store import SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, original, successor="second")
    monkeypatch.setattr(scheduler_store, "MAX_BRIDGES", 1)
    current = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, current, successor="third")
    archive = graph.store.root / ".epochs" / "epoch-0.json"
    archive.unlink()

    with pytest.raises(OccurrenceCorruptError):
        SchedulerStore(
            graph.store.root, namespace="stage3", runtime_identity="copied-root"
        ).read()
    await graph.aclose()


async def test_epoch_skipped_predecessor_fails_closed(tmp_path, monkeypatch):
    import hashlib
    import json

    import nanobot.cron.scheduler_store as scheduler_store
    from nanobot.cron.occurrence_store import OccurrenceCorruptError
    from nanobot.cron.scheduler_store import SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, original, successor="second")
    monkeypatch.setattr(scheduler_store, "MAX_BRIDGES", 1)
    current = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, current, successor="third")

    path = graph.store.root / "state.json"
    wire = json.loads(path.read_text(encoding="utf-8"))
    state = wire["state"]
    state["predecessor"]["epoch"] = 5
    state["epoch"] = 6
    wire["integrity"] = hashlib.sha256(
        json.dumps(
            state,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(wire), encoding="utf-8")

    with pytest.raises(OccurrenceCorruptError):
        SchedulerStore(
            graph.store.root, namespace="stage3", runtime_identity="copied-root"
        ).read()
    await graph.aclose()


@pytest.mark.parametrize("attack", ["missing-middle", "owner", "skipped", "resolution", "digest", "identity", "bool", "fork", "unknown"])
async def test_reconciliation_chain_attacks_fail_closed(tmp_path, monkeypatch, attack):
    from copy import deepcopy

    from nanobot.cron.occurrence_store import OccurrenceCorruptError
    from nanobot.cron.scheduler import Scheduler
    from nanobot.cron.scheduler_store import SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    for owner in ("second", "third", "fourth"):
        reconcile(graph, graph.service.occurrence_store.inspect_records()[0], successor=owner)
    reopened = SchedulerStore(graph.store.root, namespace="stage3", runtime_identity="copied-root")
    before = reopened.read()
    state = deepcopy(before)
    if attack == "missing-middle":
        del state["bridges"][1]
    elif attack == "owner":
        state["bridges"][1]["predecessor_owner"] = "wrong"
    elif attack == "skipped":
        state["bridges"][-1]["successor_generation"] = 8
    elif attack == "resolution":
        state["bridges"][-1]["resolution"] = "confirmed_compensated"
    elif attack == "digest":
        for bridge in state["bridges"]:
            bridge["immutable_digest"] = "0" * 64
    elif attack == "identity":
        state["bridges"][-1]["identity"]["scheduled_fire_time_ms"] = 1200
    elif attack == "bool":
        state["bridges"][0]["predecessor_generation"] = True
    elif attack == "fork":
        state["bridges"].append({**state["bridges"][0], "publication_generation": state["generation"]})
    else:
        state["bridges"][0]["extra"] = "unknown"
    with pytest.raises((SchedulerConflictError, OccurrenceCorruptError)):
        validated = reopened._validate(state)
        Scheduler._prove_records(validated, graph.service.occurrence_store.inspect_records())
    assert reopened.read() == before and calls == [] and channels.calls == []
    await graph.aclose()


@pytest.mark.parametrize("attack", ["identical-inode", "unexplained-generation", "history-rewrite", "held-owner", "non-bridge-change"])
async def test_acquisition_refresh_rejects_unproved_state_change(tmp_path, monkeypatch, attack):
    import hashlib
    import json
    import os

    from nanobot.cron.occurrence_store import OccurrenceCorruptError
    from nanobot.cron.scheduler_store import SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    if attack == "held-owner":
        graph.acquire()
    elif attack == "history-rewrite":
        reconcile(graph, original, successor="second")
        graph.acquire()  # Pin the first exact bridge, then stop again.
        graph.stop_claiming()
        graph.scheduler.release_lease()
        reconcile(graph, graph.service.occurrence_store.inspect_records()[0], successor="third")
    elif attack == "non-bridge-change":
        reconcile(graph, original)
    path = graph.store.root / "state.json"
    wire = json.loads(path.read_bytes())
    if attack == "unexplained-generation":
        wire["state"]["generation"] += 1
    elif attack == "history-rewrite":
        wire["state"]["bridges"][0]["resolution"] = "confirmed_compensated"
    elif attack == "non-bridge-change":
        wire["state"]["classifications"][0]["kind"] = "observed"
    if attack not in {"identical-inode", "held-owner"}:
        canonical = json.dumps(wire["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        wire["integrity"] = hashlib.sha256(canonical.encode()).hexdigest()
        data = json.dumps(wire, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    else:
        data = path.read_bytes()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(data)
    replacement.chmod(0o600)
    os.replace(replacement, path)
    with pytest.raises(OccurrenceCorruptError):
        graph.acquire()
    # A failed refresh preserves the original pin; a fresh strict parser alone
    # cannot waive the incumbent's provenance requirement.
    if attack == "unexplained-generation":
        assert SchedulerStore(graph.store.root, namespace="stage3", runtime_identity="copied-root").read()["generation"] == wire["state"]["generation"]
    assert calls == [] and channels.calls == []
    await graph.aclose()


@pytest.mark.parametrize("attack", ["bool-replaced-generation", "float-replaced-generation", "frozen-policy", "completed-to-claimed"])
async def test_authoritative_occurrence_binding_and_completed_observation_are_strict(tmp_path, monkeypatch, attack):
    from copy import deepcopy
    from dataclasses import replace

    from nanobot.cron.scheduler import Scheduler
    from nanobot.cron.scheduler_store import SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, original)
    state = SchedulerStore(graph.store.root, namespace="stage3", runtime_identity="copied-root").read()
    current = graph.service.occurrence_store.inspect_records()[0]
    if attack == "frozen-policy":
        changed = replace(current, destination_policy={**current.destination_policy, "to": "wrong"})
    elif attack == "completed-to-claimed":
        state = deepcopy(state)
        state["classifications"][-1]["exclusions"][0].update(owner=current.owner,
            generation=current.generation, state="completed")
        changed = current
    else:
        changed = replace(current, reconciliation={**current.reconciliation,
            "replaced_generation": True if attack == "bool-replaced-generation" else 1.0})
    with pytest.raises(SchedulerConflictError):
        Scheduler._prove_records(state, [changed])
    assert calls == [] and channels.calls == []
    await graph.aclose()


async def test_acquisition_exact_missed_no_effects_then_delayed_due(tmp_path):
    graph, item, clock, calls, channels = runtime(tmp_path)
    graph.stop_claiming()
    graph.scheduler.release_lease()
    clock.now = 1350
    graph.acquire()
    state = graph.store.read()
    assert state["cursors"][0]["next_fire"] == 1400
    assert state["classifications"][-1]["lower"] == 1000
    assert state["classifications"][-1]["upper"] == 1350
    assert calls == [] and channels.calls == []
    clock.now = 1700
    await graph.scheduler.wake_once()
    assert graph.service.occurrence_store.inspect_records()[0].identity.scheduled_fire_time_ms == 1400
    assert graph.store.read()["cursors"][0]["next_fire"] == 1500
    await graph.scheduler.wake_once()
    assert len(calls) == 2 and channels.calls == ["A", "B", "A", "B"]
    clock.now = 1200
    await graph.scheduler.wake_once()
    assert len(calls) == 2
    await graph.aclose()


async def test_claim_before_cursor_crash_is_observed_never_recomputed(tmp_path, monkeypatch):
    graph, item, clock, calls, channels = runtime(tmp_path)
    clock.now = 1100
    real = graph.store.compare_and_swap
    monkeypatch.setattr(graph.store, "compare_and_swap", lambda *_: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        await graph.scheduler.wake_once()
    assert len(graph.service.occurrence_store.inspect_records()) == 1 and calls == []
    graph.stop_claiming()
    graph.scheduler.release_lease()
    monkeypatch.setattr(graph.store, "compare_and_swap", real)
    graph.acquire()
    state = graph.store.read()
    assert state["classifications"][-1]["exclusions"][0]["fire"] == 1100
    assert state["cursors"][0]["next_fire"] == 1200
    assert calls == [] and channels.calls == []
    await graph.aclose()


async def test_cursor_before_begin_crash_remains_claimed(tmp_path, monkeypatch):
    graph, item, clock, calls, channels = runtime(tmp_path)
    clock.now = 1100

    async def crash(*_):
        raise OSError("before begin")

    monkeypatch.setattr(graph.scheduler, "_dispatch", crash)
    with pytest.raises(OSError):
        await graph.scheduler.wake_once()
    assert graph.store.read()["cursors"][0]["next_fire"] == 1200
    assert graph.service.occurrence_store.inspect_records()[0].state == "claimed"
    assert calls == []
    await graph.aclose()


async def test_independent_claim_in_missed_range_conflicts(tmp_path):
    graph, item, clock, calls, channels = runtime(tmp_path)
    graph.stop_claiming()
    graph.scheduler.release_lease()
    clock.now = 1400
    graph.acquire()
    graph.service.claim_scheduled_occurrence(item.job_id, 1200, owner="legacy")
    with pytest.raises(SchedulerConflictError, match="missed"):
        await graph.scheduler.wake_once()
    assert calls == [] and channels.calls == []
    await graph.aclose()


async def test_missing_evidence_does_not_infer_anchor(tmp_path):
    graph, item, clock, calls, channels = runtime(tmp_path)
    graph.stop_claiming()
    graph.scheduler.release_lease()
    definitions = graph.service.definition_store
    raw = definitions.action_store.replay()
    candidate = {key: value for key, value in raw.candidate.items() if key != "scheduling_commits"}
    candidate["kind"], candidate["schema_version"] = "cron_definition_candidate_v1", 1
    definitions.action_store.checkpoint(expected_generation=raw.generation, candidate=candidate)
    with pytest.raises(SchedulerConflictError, match="missing"):
        graph.acquire()
    assert calls == [] and channels.calls == []


@pytest.mark.parametrize("boundary", ["file-fsync", "rename", "directory-fsync"])
async def test_cursor_publication_inplace_attack_grants_no_dispatch(tmp_path, monkeypatch, boundary):
    import os
    import stat

    from nanobot.cron.occurrence_store import OccurrenceCorruptError

    graph, item, clock, calls, channels = runtime(tmp_path)
    clock.now = 1100
    original_fsync, original_rename = os.fsync, os.rename
    published = False
    attacked = False

    def mutate_state(root_fd):
        nonlocal attacked
        fd = os.open("state.json", os.O_RDWR | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            os.pwrite(fd, b"x", 0)
            attacked = True
        finally:
            os.close(fd)

    def rename(source, target, **kwargs):
        nonlocal published
        result = original_rename(source, target, **kwargs)
        if target == "state.json":
            published = True
            if boundary == "rename":
                mutate_state(kwargs["dst_dir_fd"])
        return result

    def fsync(fd):
        nonlocal attacked
        result = original_fsync(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode) and boundary == "file-fsync" and not attacked:
            try:
                data = os.pread(fd, 1_048_576, 0)
            except OSError:
                return result  # Existing P3 claim staging is write-only.
            if b"cron_scheduler_state_v1" in data and b'"kind":"claimed"' in data:
                os.pwrite(fd, b"x", 0)
                attacked = True
        if stat.S_ISDIR(os.fstat(fd).st_mode) and published and boundary == "directory-fsync" and not attacked:
            mutate_state(fd)
        return result

    with monkeypatch.context() as context:
        context.setattr(os, "fsync", fsync)
        context.setattr(os, "rename", rename)
        with pytest.raises(OccurrenceCorruptError):
            await graph.scheduler.wake_once()
    assert attacked and calls == [] and channels.calls == []
    assert graph.service.occurrence_store.inspect_records()[0].state == "claimed"
    await graph.aclose()
