"""Existing-only operator preservation, runnable before owner relocation."""

import hashlib
import json
import os
from pathlib import Path

import pytest

import nanobot.cron.occurrence_store as occurrence_module
import nanobot.cron.runtime_inspection as inspection
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceCorruptError,
    OccurrenceNotFoundError,
    OccurrencePersistenceError,
)
from nanobot.cron.scheduler_store import SchedulerStore, _PrivateFiles
from tests.cli.test_cron_runtime_inspection import snapshot


def inspect(kind, root, *, namespace="inspection-test", runtime_identity="original"):
    if kind == "occurrence":
        return inspection.inspect_occurrences(root, namespace=namespace)
    return inspection.inspect_scheduler(root, namespace=namespace, runtime_identity=runtime_identity)


def seed(kind, root):
    if kind == "scheduler":
        return SchedulerStore(root, namespace="inspection-test", runtime_identity="original")
    store = CronOccurrenceStore(root, namespace="inspection-test")
    for fire in (100, 200):
        store.claim_scheduled(job_id="private-job", job_revision=1, scheduled_fire_time_ms=fire,
            job_snapshot={"payload": "private input"}, destination_policy={}, owner="visible")
    return store


class UnreadableBound:
    def __mul__(self, _other):
        raise AssertionError("bound accessed before comparison")


@pytest.mark.parametrize("new_bound,refused", [(0, True), (1, False)])
def test_bound_is_read_after_listdir_at_exact_entry_edge(tmp_path, monkeypatch, new_bound, refused):
    root = tmp_path / "occurrences"
    seed("occurrence", root)
    original = os.listdir
    monkeypatch.setattr(inspection, "MAX_INSPECTION_RECORDS", UnreadableBound())

    def listing(fd):
        names = original(fd)
        assert len(names) == 3
        monkeypatch.setattr(inspection, "MAX_INSPECTION_RECORDS", new_bound)
        return reversed(names)

    monkeypatch.setattr(os, "listdir", listing)
    if refused:
        with pytest.raises(OccurrenceCorruptError, match="inspection record bound exceeded"):
            inspect("occurrence", root)
    else:
        records = inspect("occurrence", root)["records"]
        assert [record["occurrence_id"] for record in records] == sorted(record["occurrence_id"] for record in records)


@pytest.mark.parametrize("failure", ["admission", "open", "enumeration"])
def test_failure_before_comparison_does_not_read_bound(tmp_path, monkeypatch, failure):
    root = tmp_path / "occurrences"
    monkeypatch.setattr(inspection, "MAX_INSPECTION_RECORDS", UnreadableBound())
    if failure == "admission":
        root = Path("relative")
        error = ValueError
    elif failure == "open":
        error = OccurrencePersistenceError
    else:
        seed("occurrence", root)
        def listing(_fd):
            raise OSError("enumeration fault")
        monkeypatch.setattr(os, "listdir", listing)
        error = OSError
    with pytest.raises(error):
        inspect("occurrence", root)


@pytest.mark.parametrize("kind", ["occurrence", "scheduler"])
@pytest.mark.parametrize("root", [Path("relative"), Path("/"), Path("/tmp/../refused")])
def test_qualified_then_lexical_then_namespace_admission(kind, root, monkeypatch):
    events = []
    def qualified():
        events.append("qualified")
    def namespace(_value):
        events.append("namespace")
        raise AssertionError("namespace before lexical refusal")
    monkeypatch.setattr(CronOccurrenceStore, "_require_qualified_platform", staticmethod(qualified))
    monkeypatch.setattr(CronOccurrenceStore, "_validate_namespace", staticmethod(namespace))
    with pytest.raises(ValueError, match="explicit lexical absolute non-root path required"):
        inspect(kind, root, namespace="")
    assert events == ["qualified"]


@pytest.mark.parametrize("kind", ["occurrence", "scheduler"])
def test_platform_refusal_precedes_lexical_and_namespace(kind, monkeypatch):
    fault = OccurrencePersistenceError("unqualified sentinel")
    def qualified():
        raise fault
    monkeypatch.setattr(CronOccurrenceStore, "_require_qualified_platform", staticmethod(qualified))
    with pytest.raises(OccurrencePersistenceError) as caught:
        inspect(kind, Path("relative"), namespace="")
    assert caught.value is fault


@pytest.mark.parametrize("kind", ["occurrence", "scheduler"])
def test_namespace_refusal_before_missing_root(kind, tmp_path):
    with pytest.raises(ValueError, match="namespace must be"):
        inspect(kind, tmp_path / "missing", namespace="")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["occurrence", "scheduler"])
def test_inspection_denies_constructors_and_mutation_preserves_bytes(kind, tmp_path, monkeypatch):
    root = tmp_path / kind
    seed(kind, root)
    before = snapshot(tmp_path)
    original = os.open
    opens = []
    def opened(path, flags, *args, **kwargs):
        assert not flags & (os.O_CREAT | os.O_TRUNC | os.O_EXCL)
        opens.append((path, flags))
        return original(path, flags, *args, **kwargs)
    def denied(*_args, **_kwargs):
        raise AssertionError("inspection mutation or constructor")
    with monkeypatch.context() as context:
        context.setattr(os, "open", opened)
        context.setattr(os, "supports_dir_fd", os.supports_dir_fd | {opened})
        for cls in (CronOccurrenceStore, SchedulerStore, _PrivateFiles):
            context.setattr(cls, "__init__", denied)
        for operation in ("mkdir", "chmod"):
            context.setattr(Path, operation, denied)
        context.setattr(os, "fsync", denied)
        context.setattr(occurrence_module._fcntl, "flock", denied)
        result = inspect(kind, root)
    assert opens and snapshot(tmp_path) == before
    assert "private input" not in str(result)


@pytest.mark.parametrize("identity", ["", "different-copy"])
def test_scheduler_readonly_identity_has_no_writer_constructor_validation(tmp_path, identity):
    root = tmp_path / "scheduler"
    seed("scheduler", root)
    path = root / "state.json"
    wire = json.loads(path.read_bytes())
    wire["state"]["runtime_identity"] = identity
    canonical = json.dumps(wire["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    wire["integrity"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(wire))
    assert inspect("scheduler", root, runtime_identity=identity)["generation"] == 0
    with pytest.raises(OccurrenceCorruptError, match="scheduler authority binding mismatch"):
        inspect("scheduler", root, runtime_identity="wrong")


@pytest.mark.parametrize("kind", ["occurrence", "scheduler"])
def test_missing_lock_is_not_created(kind, tmp_path):
    root = tmp_path / kind
    seed(kind, root)
    (root / ".lock").unlink()
    before = snapshot(tmp_path)
    inspect(kind, root)
    assert snapshot(tmp_path) == before and not (root / ".lock").exists()


def test_missing_state_is_not_created(tmp_path):
    root = tmp_path / "scheduler"
    seed("scheduler", root)
    (root / "state.json").unlink()
    before = snapshot(tmp_path)
    with pytest.raises(OccurrenceNotFoundError, match="scheduler state missing"):
        inspect("scheduler", root)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["occurrence", "scheduler"])
@pytest.mark.parametrize("fault", [False, True])
def test_partial_reads_and_faults_close_every_descriptor(kind, fault, tmp_path, monkeypatch):
    root = tmp_path / kind
    seed(kind, root)
    expected = inspect(kind, root)
    original_open, original_close, original_read = os.open, os.close, os.read
    live = set()
    reads = []
    failure = OSError("injected read fault")
    def opened(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        live.add(fd)
        return fd
    def closed(fd):
        live.remove(fd)
        return original_close(fd)
    def read(fd, size):
        reads.append(fd)
        if fault:
            raise failure
        return original_read(fd, min(size, 7))
    with monkeypatch.context() as context:
        context.setattr(os, "open", opened)
        context.setattr(os, "supports_dir_fd", os.supports_dir_fd | {opened})
        context.setattr(os, "close", closed)
        context.setattr(os, "read", read)
        if fault:
            with pytest.raises((OSError, OccurrencePersistenceError)) as caught:
                inspect(kind, root)
            assert caught.value is failure or caught.value.__cause__ is failure
        else:
            assert inspect(kind, root) == expected
    assert len(reads) > (1 if not fault else 0) and live == set()


@pytest.mark.parametrize("kind", ["occurrence", "scheduler"])
def test_final_root_reproof_closes_chain_on_fault(kind, tmp_path, monkeypatch):
    root = tmp_path / kind
    seed(kind, root)
    # Empty occurrence root reaches its final proof immediately after listdir.
    if kind == "occurrence":
        for path in root.glob("*.json"):
            path.unlink()
    original = occurrence_module._RootBinding.assert_bound
    calls = []
    failure = OccurrenceCorruptError("final root proof fault")
    final_count = 1 if kind == "occurrence" else 3
    def bound(binding):
        calls.append(binding)
        if len(calls) == final_count:
            raise failure
        return original(binding)
    monkeypatch.setattr(occurrence_module._RootBinding, "assert_bound", bound)
    with pytest.raises(OccurrenceCorruptError) as caught:
        inspect(kind, root)
    assert caught.value is failure
    for fd in calls[-1].fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("change,message", [
    ({"external_evidence": "free lock"}, "explicit external evidence"),
    ({"resolution": "unknown"}, "explicit external evidence"),
    ({"expected_generation": True}, "exact predecessor generation"),
    ({"expected_generation": 0}, "exact predecessor generation"),
    ({"expected_owner": ""}, "owner must be"),
    ({"successor_owner": ""}, "owner must be"),
])
def test_invalid_preflight_precedes_all_inspection_and_writers(tmp_path, monkeypatch, change, message):
    def denied(*_args, **_kwargs):
        raise AssertionError("read or writer before preflight")
    monkeypatch.setattr(inspection, "inspect_occurrences", denied)
    monkeypatch.setattr(inspection, "inspect_scheduler", denied)
    monkeypatch.setattr(CronOccurrenceStore, "__init__", denied)
    monkeypatch.setattr(SchedulerStore, "__init__", denied)
    arguments = dict(namespace="inspection-test", occurrence_id="0" * 64,
        expected_owner="before", expected_generation=1, resolution="confirmed_not_started",
        external_evidence="EXTERNAL-EVIDENCE-VERIFIED", successor_owner="after",
        scheduler_root=tmp_path / "scheduler", runtime_identity="copied")
    arguments.update(change)
    with pytest.raises(ValueError, match=message):
        inspection.reconcile_occurrence(tmp_path / "occurrences", **arguments)
    assert list(tmp_path.iterdir()) == []
