import json

import pytest
from typer.testing import CliRunner

from nanobot.cli.cron_runtime import cron_runtime_app
from nanobot.cron.runtime_inspection import (
    inspect_occurrences,
    inspect_scheduler,
    reconcile_occurrence,
)
from tests.cron.stage3_fixtures import offline_denial, runtime  # noqa: F401
from tests.cron.test_scheduler import prebegin_crash, reconcile


def snapshot(root):
    return [(str(path.relative_to(root)), path.stat().st_ino, path.stat().st_mtime_ns,
             path.read_bytes() if path.is_file() else None)
            for path in [root, *sorted(root.rglob("*"))]]


async def test_existing_only_content_free_inspection_changes_no_objects(tmp_path, monkeypatch):
    import os

    graph, item, clock, calls, channels = runtime(tmp_path)
    claim = graph.service.claim_scheduled_occurrence(item.job_id, 1100, owner="operator-visible")
    before = snapshot(tmp_path)
    with monkeypatch.context() as denied:
        denied.setattr(os, "fsync", lambda *_: (_ for _ in ()).throw(AssertionError("inspection fsync")))
        scheduler = inspect_scheduler(graph.store.root, namespace="stage3", runtime_identity="copied-root")
        occurrences = inspect_occurrences(graph.service.occurrence_store.root, namespace="stage3")
    assert snapshot(tmp_path) == before
    encoded = json.dumps([scheduler, occurrences])
    assert claim.occurrence_id in encoded and "private compute" not in encoded and "private result" not in encoded
    assert calls == [] and channels.calls == []
    await graph.aclose()


def test_missing_root_creates_nothing_and_cli_does_not_discover_config(tmp_path):
    missing = tmp_path / "missing"
    result = CliRunner().invoke(cron_runtime_app, ["scheduler-inspect", "--root", str(missing),
        "--namespace", "stage3", "--runtime-identity", "copied-root"])
    assert result.exit_code == 5 and not missing.exists()
    assert "unavailable" in result.output


async def test_exact_predecessor_reconciliation_no_successor_effect(tmp_path):
    graph, item, clock, calls, channels = runtime(tmp_path)
    claim = graph.service.claim_scheduled_occurrence(item.job_id, 1100, owner="before")
    graph.stop_claiming()
    graph.scheduler.release_lease()
    with pytest.raises(ValueError):
        reconcile_occurrence(graph.service.occurrence_store.root, namespace="stage3", occurrence_id=claim.occurrence_id,
            expected_owner="before", expected_generation=1, resolution="confirmed_not_started",
            external_evidence="free lock", successor_owner="after", scheduler_root=graph.store.root, runtime_identity="copied-root")
    result = reconcile_occurrence(graph.service.occurrence_store.root, namespace="stage3", occurrence_id=claim.occurrence_id,
        expected_owner="before", expected_generation=1, resolution="confirmed_not_started",
        external_evidence="EXTERNAL-EVIDENCE-VERIFIED", successor_owner="after", scheduler_root=graph.store.root, runtime_identity="copied-root")
    assert result["generation"] == 2 and calls == [] and channels.calls == []
    await graph.aclose()


@pytest.mark.parametrize("boundary", ["before-occurrence", "occurrence-lost-ack", "between-authorities", "bridge-lost-ack"])
async def test_operator_crash_exact_retry_retains_authorities_without_extra_reclaim(tmp_path, monkeypatch, boundary):
    from nanobot.cron.occurrence_store import CronOccurrenceStore
    from nanobot.cron.scheduler_store import SchedulerConflictError, SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    transitions = []
    real_reclaim = CronOccurrenceStore.reclaim_after_reconciliation
    real_cas = SchedulerStore.compare_and_swap

    def reclaim(self, *args, **kwargs):
        if boundary == "before-occurrence":
            raise OSError("before occurrence publication")
        result = real_reclaim(self, *args, **kwargs)
        transitions.append(result.generation)
        if boundary == "occurrence-lost-ack":
            raise OSError("occurrence ack lost")
        return result

    def cas(self, expected, successor):
        if boundary == "between-authorities":
            raise OSError("bridge not published")
        result = real_cas(self, expected, successor)
        if boundary == "bridge-lost-ack":
            raise OSError("bridge ack lost")
        return result

    with monkeypatch.context() as injection:
        injection.setattr(CronOccurrenceStore, "reclaim_after_reconciliation", reclaim)
        injection.setattr(SchedulerStore, "compare_and_swap", cas)
        with pytest.raises(OSError):
            reconcile(graph, original)
    record = graph.service.occurrence_store.inspect_records()[0]
    assert record.generation == (1 if boundary == "before-occurrence" else 2)
    clock.now = 1200
    if boundary in {"occurrence-lost-ack", "between-authorities"}:
        with pytest.raises(SchedulerConflictError):
            graph.acquire()
        assert calls == [] and channels.calls == []
    # Monitor the retry itself: already-published successor MUST NOT reclaim.
    retried = []

    def retry_reclaim(self, *args, **kwargs):
        retried.append(1)
        return real_reclaim(self, *args, **kwargs)

    with monkeypatch.context() as injection:
        injection.setattr(CronOccurrenceStore, "reclaim_after_reconciliation", retry_reclaim)
        result = reconcile(graph, original)
        assert reconcile(graph, original) == result  # bridge-ack retry is detached evidence.
    assert len(retried) == (1 if boundary == "before-occurrence" else 0)
    assert result["generation"] == 2
    graph.acquire()
    state = graph.store.read()
    assert len(state["bridges"]) == 1 and state["cursors"][0]["next_fire"] == 1300
    await graph.scheduler.wake_once()
    assert calls == [] and channels.calls == []
    await graph.aclose()


async def test_next_reconciliation_cannot_overwrite_unretained_attestation(tmp_path, monkeypatch):
    from nanobot.cron.scheduler_store import SchedulerConflictError, SchedulerStore

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    with monkeypatch.context() as injection:
        injection.setattr(SchedulerStore, "compare_and_swap", lambda *a: (_ for _ in ()).throw(OSError("between authorities")))
        with pytest.raises(OSError):
            reconcile(graph, original, successor="second")
    second = graph.service.occurrence_store.inspect_records()[0]
    before = snapshot(tmp_path)
    with pytest.raises(SchedulerConflictError, match="prior reconciliation"):
        reconcile(graph, second, successor="third")
    assert snapshot(tmp_path) == before
    reconcile(graph, original, successor="second")
    reconcile(graph, second, successor="third")
    graph.acquire()
    assert len(graph.store.read()["bridges"]) == 2
    assert calls == [] and channels.calls == []
    await graph.aclose()


async def test_operator_active_scheduler_and_live_execution_exclude_mutation(tmp_path, monkeypatch):
    from nanobot.cron.occurrence_store import OccurrenceExecutionBusyError
    from nanobot.cron.scheduler_store import SchedulerConflictError

    graph, item, clock, calls, channels = runtime(tmp_path)
    claim = graph.service.claim_scheduled_occurrence(item.job_id, 1100, owner="before")
    original = graph.service.occurrence_store.get(claim.occurrence_id)
    before = snapshot(tmp_path)
    with pytest.raises(SchedulerConflictError, match="busy"):
        reconcile(graph, original)
    assert snapshot(tmp_path) == before
    graph.stop_claiming()
    graph.scheduler.release_lease()
    fence = graph.service.occurrence_store.acquire_execution_fence(claim)
    before = snapshot(tmp_path)
    try:
        with pytest.raises(OccurrenceExecutionBusyError):
            reconcile(graph, original)
        assert snapshot(tmp_path) == before
    finally:
        graph.service.occurrence_store.release_execution_fence(fence)
    assert calls == [] and channels.calls == []
    await graph.aclose()


@pytest.mark.parametrize("bound", ["records", "bytes"])
async def test_bridge_bounds_reject_before_occurrence_mutation(tmp_path, monkeypatch, bound):
    import nanobot.cron.scheduler_store as module
    from nanobot.cron.occurrence_store import CronOccurrenceStore, OccurrenceCorruptError
    from nanobot.cron.scheduler_store import SchedulerConflictError

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    before = snapshot(tmp_path)
    size = (graph.store.root / "state.json").stat().st_size
    with monkeypatch.context() as injection:
        injection.setattr(module, "MAX_BRIDGES" if bound == "records" else "MAX_STATE_BYTES", 0 if bound == "records" else size + 1)
        injection.setattr(CronOccurrenceStore, "reclaim_after_reconciliation", lambda *a, **k: (_ for _ in ()).throw(AssertionError("mutation before capacity")))
        with pytest.raises((OccurrenceCorruptError, SchedulerConflictError)):
            reconcile(graph, original)
    assert snapshot(tmp_path) == before and calls == [] and channels.calls == []
    await graph.aclose()


@pytest.mark.parametrize("authority", ["scheduler", "occurrence"])
async def test_reconciliation_unsafe_residue_preserves_forensic_bytes(tmp_path, monkeypatch, authority):
    from nanobot.cron.occurrence_store import OccurrenceCorruptError

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    root = graph.store.root if authority == "scheduler" else graph.service.occurrence_store.root
    residue = root / ".state.forensic.tmp" if authority == "scheduler" else root / ("." + original.occurrence_id + ".forensic.tmp")
    residue.write_bytes(b"retained forensic evidence")
    residue.chmod(0o600)
    before = snapshot(tmp_path)
    with pytest.raises(OccurrenceCorruptError):
        reconcile(graph, original)
    assert snapshot(tmp_path) == before and calls == [] and channels.calls == []
    await graph.aclose()


@pytest.mark.parametrize("field", ["owner", "generation", "resolution"])
async def test_exact_reconciliation_retry_rejects_conflicting_request(tmp_path, monkeypatch, field):
    from nanobot.cron.scheduler_store import SchedulerConflictError

    graph, item, clock, calls, channels = await prebegin_crash(tmp_path, monkeypatch)
    original = graph.service.occurrence_store.inspect_records()[0]
    reconcile(graph, original)
    arguments = dict(namespace="stage3", occurrence_id=original.occurrence_id,
        expected_owner=original.owner, expected_generation=1, resolution="confirmed_not_started",
        external_evidence="EXTERNAL-EVIDENCE-VERIFIED", successor_owner="operator-successor",
        scheduler_root=graph.store.root, runtime_identity="copied-root")
    arguments[{"owner": "successor_owner", "generation": "expected_generation", "resolution": "resolution"}[field]] = {"owner": "wrong", "generation": 9, "resolution": "confirmed_compensated"}[field]
    before = snapshot(tmp_path)
    with pytest.raises(SchedulerConflictError):
        reconcile_occurrence(graph.service.occurrence_store.root, **arguments)
    assert snapshot(tmp_path) == before
    assert calls == [] and channels.calls == []
    await graph.aclose()
