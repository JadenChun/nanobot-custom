"""Contract tests for the isolated version-2 cron action store."""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
from pathlib import Path
from queue import Empty

import pytest

import nanobot.cron.action_store as action_store
from nanobot.cron.action_store import (
    ActionStoreConflictError,
    ActionStoreConversionRequiredError,
    ActionStoreCorruptError,
    ActionStoreGeneration,
    ActionStorePersistenceError,
    ActionStoreStaleGenerationError,
    CronActionStore,
)

NAMESPACE = "cron-action-store:tests"


def _store(tmp_path: Path) -> CronActionStore:
    return CronActionStore(tmp_path / "private-actions", namespace=NAMESPACE)


def _append(store: CronActionStore, action_id: str = "action-1", **request: object):
    return store.append(action_id=action_id, request={"operation": "opaque", **request})


def _append_in_process(root: str, ready: object, outcomes: object, action_id: str) -> None:
    store = CronActionStore(Path(root), namespace=NAMESPACE)
    ready.wait()  # type: ignore[union-attr]
    try:
        outcomes.put(("ok", _append(store, action_id, worker=action_id).sequence))  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - makes worker failure legible
        outcomes.put((type(exc).__name__, str(exc)))  # type: ignore[union-attr]


def _append_with_generation_in_process(
    root: str,
    generation: ActionStoreGeneration,
    ready: object,
    go: object,
    outcomes: object,
    action_id: str,
) -> None:
    store = CronActionStore(Path(root), namespace=NAMESPACE)
    ready.put("read")  # type: ignore[union-attr]
    go.wait()  # type: ignore[union-attr]
    try:
        receipt = store.append(
            action_id=action_id,
            request={"operation": "opaque", "worker": action_id},
            expected_generation=generation,
        )
    except ActionStoreStaleGenerationError:
        outcomes.put("stale")  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - makes worker failure legible
        outcomes.put((type(exc).__name__, str(exc)))  # type: ignore[union-attr]
    else:
        outcomes.put(("ok", receipt.sequence))  # type: ignore[union-attr]


def _checkpoint_in_process(root: str, read_ready: object, go: object, outcomes: object, marker: str) -> None:
    store = CronActionStore(Path(root), namespace=NAMESPACE)
    state = store.replay()
    read_ready.put("read")  # type: ignore[union-attr]
    go.wait()  # type: ignore[union-attr]
    try:
        store.checkpoint(expected_generation=state.generation, candidate={"winner": marker})
    except (ActionStoreStaleGenerationError, ActionStoreConflictError):
        outcomes.put("lost")  # type: ignore[union-attr]
    else:
        outcomes.put("won")  # type: ignore[union-attr]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fail_nth_fsync(monkeypatch: pytest.MonkeyPatch, nth: int) -> None:
    original = os.fsync
    calls = 0

    def fail(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == nth:
            raise OSError("injected fsync failure")
        original(fd)

    monkeypatch.setattr(action_store.os, "fsync", fail)


def _replace_owned_entry(path: Path) -> None:
    replacement_source = path.with_name(path.name + ".replaced")
    payload = path.read_bytes()
    os.rename(path, replacement_source)
    path.write_bytes(payload)
    os.chmod(path, 0o600)


def _replace_root(root: Path) -> None:
    moved = root.with_name(root.name + ".replaced")
    os.rename(root, moved)
    root.mkdir(mode=0o700)


def _replace_after_fsync(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    original = os.fsync
    target_inode = (path.stat().st_dev, path.stat().st_ino)
    replaced = False

    def swap(fd: int) -> None:
        nonlocal replaced
        original(fd)
        info = os.fstat(fd)
        if not replaced and stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == target_inode:
            _replace_owned_entry(path)
            replaced = True

    monkeypatch.setattr(action_store.os, "fsync", swap)


def test_append_replay_and_checkpoint_are_generic_and_detached(tmp_path: Path) -> None:
    store = _store(tmp_path)
    one = _append(store, "action-1", value=1)
    two = _append(store, "action-2", value=2)

    before = store.replay()
    assert before.candidate is None
    assert [(item.receipt.action_id, item.request["value"]) for item in before.pending_actions] == [
        ("action-1", 1),
        ("action-2", 2),
    ]
    assert [receipt.sequence for receipt in before.receipts] == [1, 2]
    before.pending_actions[0].request["value"] = "mutated"

    after = store.checkpoint(expected_generation=before.generation, candidate={"opaque": {"kept": True}})
    assert after.candidate == {"opaque": {"kept": True}}
    assert after.pending_actions == ()
    assert after.generation == ActionStoreGeneration(2, 1, 2, 2)
    assert _append(store, "action-1", value=1) == one
    assert _append(store, "action-2", value=2) == two
    assert store.replay().candidate == {"opaque": {"kept": True}}


def test_action_id_exact_request_digest_is_idempotent_and_conflicts_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _append(store, "fixed-id", nested={"a": [1, 2]})
    assert _append(store, "fixed-id", nested={"a": [1, 2]}) == first
    with pytest.raises(ActionStoreConflictError, match="different canonical request"):
        _append(store, "fixed-id", nested={"a": [2, 1]})
    with pytest.raises(ValueError, match="action_id"):
        _append(store, "../../bad")
    with pytest.raises(ActionStoreConflictError, match="finite JSON"):
        _append(store, "not-finite", value=float("nan"))


def test_strict_json_rejects_duplicate_nonfinite_bool_integer_and_unknown_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    journal = store.root / "journal.jsonl"
    header = _read_jsonl(journal)[0]
    journal.write_text(
        json.dumps(header) + "\n" +
        '{"schema_version":2,"schema_version":2,"kind":"action"}\n', encoding="utf-8"
    )
    with pytest.raises(ActionStoreCorruptError, match="strict JSON"):
        store.replay()

    journal.write_text(json.dumps({**header, "base_sequence": True}) + "\n", encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match="integer"):
        store.replay()

    journal.write_text(json.dumps({**header, "unknown": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match="invalid schema"):
        store.replay()

    journal.write_text(json.dumps(header) + "\n" + '{"bad":NaN}\n', encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match="strict JSON"):
        store.replay()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2.0, "schema version.*exact integer"),
        ("schema_version", True, "schema version.*exact integer"),
        ("journal_generation", 1.0, "integer"),
        ("base_sequence", True, "integer"),
    ],
)
def test_journal_header_numeric_fields_require_exact_integers(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    store = _store(tmp_path)
    journal = store.root / "journal.jsonl"
    header = _read_jsonl(journal)[0]
    header[field] = value
    journal.write_text(json.dumps(header) + "\n", encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match=match):
        store.replay()


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2.0), ("journal_generation", True), ("sequence", 1.0)],
)
def test_journal_action_numeric_fields_require_exact_integers(
    tmp_path: Path, field: str, value: object
) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    journal = store.root / "journal.jsonl"
    records = _read_jsonl(journal)
    records[1][field] = value
    journal.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match="integer"):
        store.replay()


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("schema_version",), 2.0, "schema version"),
        (("journal_generation",), True, "integer"),
        (("snapshot_generation",), 1.0, "integer"),
        (("base_sequence",), True, "integer"),
        (("watermark",), 1.0, "integer"),
        (("prior_generation", "journal_generation"), True, "integer"),
        (("prior_generation", "snapshot_generation"), 0.0, "integer"),
        (("prior_generation", "watermark"), True, "integer"),
        (("prior_generation", "sequence"), 1.0, "integer"),
        (("receipts", 0, "sequence"), True, "integer"),
        (("receipts", 0, "journal_generation"), 999, "later than"),
    ],
)
def test_snapshot_numeric_fields_and_receipt_generations_are_exact_and_possible(
    tmp_path: Path, path: tuple[object, ...], value: object, match: str
) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    store.checkpoint(expected_generation=store.replay().generation, candidate={"state": 1})
    snapshot_path = store.root / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    target: object = snapshot
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match=match):
        store.replay()


def test_snapshot_receipt_generation_must_follow_compaction_boundaries(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    store.checkpoint(expected_generation=store.replay().generation, candidate={"state": 1})
    _append(store, "two")
    store.checkpoint(expected_generation=store.replay().generation, candidate={"state": 2})
    snapshot_path = store.root / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    snapshot["receipts"][0]["journal_generation"] = 2
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match="predates"):
        store.replay()

    snapshot["receipts"][0]["journal_generation"] = 1
    snapshot["receipts"][1]["journal_generation"] = 1
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match="compaction boundary"):
        store.replay()

def test_tail_and_middle_corruption_are_fenced(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append(store, "one", n=1)
    _append(store, "two", n=2)
    journal = store.root / "journal.jsonl"
    original = journal.read_bytes()

    journal.write_bytes(original[:-3])
    with pytest.raises(ActionStoreCorruptError, match="newline terminated|strict JSON"):
        store.replay()

    journal.write_bytes(original.replace(b'"action_id":"one"', b'"action_id":"two"'))
    with pytest.raises(ActionStoreCorruptError, match="duplicate action id"):
        store.replay()


def test_namespace_base_and_missing_snapshot_mismatches_are_fenced(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    journal = store.root / "journal.jsonl"
    lines = _read_jsonl(journal)
    lines[0]["store_namespace"] = "other"
    journal.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match="namespace"):
        store.replay()

    other = _store(tmp_path / "second")
    header = _read_jsonl(other.root / "journal.jsonl")[0]
    header["base_sequence"] = 1
    (other.root / "journal.jsonl").write_text(json.dumps(header) + "\n", encoding="utf-8")
    with pytest.raises(ActionStoreCorruptError, match="nonzero base"):
        other.replay()


def test_snapshot_and_journal_compaction_preserve_contiguous_receipts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append(store, "one", n=1)
    _append(store, "two", n=2)
    state = store.replay()
    store.checkpoint(expected_generation=state.generation, candidate={"all": "opaque"})
    _append(store, "three", n=3)
    replay = store.replay()

    assert replay.generation == ActionStoreGeneration(2, 1, 2, 3)
    assert [receipt.sequence for receipt in replay.receipts] == [1, 2, 3]
    assert [item.receipt.sequence for item in replay.pending_actions] == [3]
    header, action = _read_jsonl(store.root / "journal.jsonl")
    assert header["journal_generation"] == 2
    assert header["base_sequence"] == 2
    assert action["sequence"] == 3
    snapshot = json.loads((store.root / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["watermark"] == 2
    assert [item["sequence"] for item in snapshot["receipts"]] == [1, 2]


def test_new_header_fsync_failure_is_not_acknowledged_and_restart_reproves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private-actions"
    root.mkdir(mode=0o700)
    original = os.fsync
    regular_calls = 0

    def fail_header_file(fd: int) -> None:
        nonlocal regular_calls
        if stat.S_ISREG(os.fstat(fd).st_mode):
            regular_calls += 1
            # First regular fsync establishes .lock; the second is the new
            # journal header. Directory fsyncs have a directory descriptor.
            if regular_calls == 2:
                raise OSError("injected new-header fsync failure")
        original(fd)

    monkeypatch.setattr(action_store.os, "fsync", fail_header_file)
    with pytest.raises(ActionStorePersistenceError, match="journal"):
        CronActionStore(root, namespace=NAMESPACE)
    monkeypatch.undo()
    assert CronActionStore(root, namespace=NAMESPACE).replay().receipts == ()


@pytest.mark.parametrize("nth", [1, 2, 3, 4])
def test_append_fsync_boundaries_fail_without_receipt_and_exact_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nth: int
) -> None:
    store = _store(tmp_path)
    _fail_nth_fsync(monkeypatch, nth)
    with pytest.raises(ActionStorePersistenceError):
        _append(store, "retry", n=1)
    monkeypatch.undo()
    receipt = _append(store, "retry", n=1)
    assert receipt.sequence == 1
    assert store.replay().pending_actions[0].receipt == receipt


@pytest.mark.parametrize("nth", [1, 2, 3, 4, 5, 6])
def test_checkpoint_each_fsync_boundary_supports_safe_exact_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nth: int
) -> None:
    store = _store(tmp_path)
    _append(store, "one", n=1)
    state = store.replay()
    _fail_nth_fsync(monkeypatch, nth)
    with pytest.raises(ActionStorePersistenceError):
        store.checkpoint(expected_generation=state.generation, candidate={"phase": "one"})
    monkeypatch.undo()

    recovered = store.checkpoint(expected_generation=state.generation, candidate={"phase": "one"})
    assert recovered.candidate == {"phase": "one"}
    assert recovered.pending_actions == ()
    assert _read_jsonl(store.root / "journal.jsonl") == [
        {
            "schema_version": 2,
            "kind": "journal_header",
            "store_namespace": NAMESPACE,
            "journal_generation": 2,
            "base_sequence": 1,
        }
    ]


def test_snapshot_replace_failure_does_not_advance_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    state = store.replay()
    original = action_store.os.rename

    def fail_snapshot(src: str, dst: str, **kwargs: object) -> None:
        if dst == "snapshot.json":
            raise OSError("injected replace failure")
        original(src, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(action_store.os, "rename", fail_snapshot)
    with pytest.raises(ActionStorePersistenceError):
        store.checkpoint(expected_generation=state.generation, candidate={"x": 1})
    monkeypatch.undo()
    assert store.replay().generation == state.generation


def test_crash_after_durable_snapshot_before_cleanup_is_settled_before_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    _append(store, "one", n=1)
    state = store.replay()
    original = action_store.os.rename

    def fail_journal(src: str, dst: str, **kwargs: object) -> None:
        if dst == "journal.jsonl":
            raise OSError("crash before journal cleanup")
        original(src, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(action_store.os, "rename", fail_journal)
    with pytest.raises(ActionStorePersistenceError):
        store.checkpoint(expected_generation=state.generation, candidate={"checkpoint": 1})
    monkeypatch.undo()

    reopened = CronActionStore(store.root, namespace=NAMESPACE)
    assert reopened.replay().candidate == {"checkpoint": 1}
    receipt = _append(reopened, "two", n=2)
    assert receipt.sequence == 2
    header = _read_jsonl(reopened.root / "journal.jsonl")[0]
    assert header["journal_generation"] == 2 and header["base_sequence"] == 1


@pytest.mark.parametrize("nth", [1, 2, 3, 4, 5, 6])
def test_checkpoint_with_existing_snapshot_retries_each_durability_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nth: int
) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    store.checkpoint(expected_generation=store.replay().generation, candidate={"checkpoint": 1})
    state = store.replay()
    _fail_nth_fsync(monkeypatch, nth)
    with pytest.raises(ActionStorePersistenceError):
        store.checkpoint(expected_generation=state.generation, candidate={"checkpoint": 2})
    monkeypatch.undo()
    settled = store.checkpoint(expected_generation=state.generation, candidate={"checkpoint": 2})
    assert settled.candidate == {"checkpoint": 2}
    assert settled.generation == ActionStoreGeneration(3, 2, 1, 1)


def test_root_replacement_after_append_fsync_is_never_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    original = os.fsync
    replaced = False

    def swap_root_after_fsync(fd: int) -> None:
        nonlocal replaced
        original(fd)
        if not replaced and stat.S_ISREG(os.fstat(fd).st_mode):
            _replace_root(store.root)
            replaced = True

    monkeypatch.setattr(action_store.os, "fsync", swap_root_after_fsync)
    with pytest.raises(ActionStoreCorruptError, match="root.*changed|root was replaced"):
        _append(store, "must-not-ack")


def test_root_replacement_after_checkpoint_fsync_is_never_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    state = store.replay()
    original = os.fsync
    regular_calls = 0

    def swap_root_after_snapshot_fsync(fd: int) -> None:
        nonlocal regular_calls
        original(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode):
            regular_calls += 1
            # Checkpoint first re-proves the journal. The next regular fsync
            # is the new snapshot staging file.
            if regular_calls == 2:
                _replace_root(store.root)

    monkeypatch.setattr(action_store.os, "fsync", swap_root_after_snapshot_fsync)
    with pytest.raises(ActionStoreCorruptError, match="root.*changed|root was replaced"):
        store.checkpoint(expected_generation=state.generation, candidate={"new": "candidate"})


def test_journal_replacement_after_fsync_is_never_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _replace_after_fsync(monkeypatch, store.root / "journal.jsonl")
    with pytest.raises(ActionStoreCorruptError, match="journal.jsonl binding changed"):
        _append(store, "must-not-ack")


def test_snapshot_replacement_during_checkpoint_retry_is_never_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    state = store.replay()
    original_rename = action_store.os.rename

    def fail_journal_cleanup(src: str, dst: str, **kwargs: object) -> None:
        if dst == "journal.jsonl":
            raise OSError("leave durable snapshot for retry")
        original_rename(src, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(action_store.os, "rename", fail_journal_cleanup)
    with pytest.raises(ActionStorePersistenceError):
        store.checkpoint(expected_generation=state.generation, candidate={"checkpoint": 1})
    monkeypatch.undo()

    _replace_after_fsync(monkeypatch, store.root / "snapshot.json")
    with pytest.raises(ActionStoreCorruptError, match="snapshot.json binding changed"):
        store.checkpoint(expected_generation=state.generation, candidate={"checkpoint": 1})


def test_snapshot_rename_substitution_fences_compaction_and_preserves_action_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _append(store, "one", evidence="must-survive")
    state = store.replay()
    journal_path = store.root / "journal.jsonl"
    journal_before = journal_path.read_bytes()
    original_rename = action_store.os.rename
    substituted = False
    intended_snapshot = store.root / "snapshot.json.intended"

    def substitute_snapshot(src: str, dst: str, **kwargs: object) -> None:
        nonlocal substituted
        original_rename(src, dst, **kwargs)  # type: ignore[arg-type]
        if dst != "snapshot.json" or substituted:
            return
        target = store.root / dst
        payload = target.read_bytes()
        original_rename(target, intended_snapshot)
        target.write_bytes(payload)
        os.chmod(target, 0o600)
        substituted = True

    monkeypatch.setattr(action_store.os, "rename", substitute_snapshot)
    with pytest.raises(ActionStoreCorruptError, match="snapshot.json binding changed"):
        store.checkpoint(expected_generation=state.generation, candidate={"candidate": "one"})
    assert substituted
    assert journal_path.read_bytes() == journal_before
    assert len(_read_jsonl(journal_path)) == 2

    monkeypatch.undo()
    (store.root / "snapshot.json").unlink()
    original_rename(intended_snapshot, store.root / "snapshot.json")
    settled = store.checkpoint(expected_generation=state.generation, candidate={"candidate": "one"})
    assert settled.candidate == {"candidate": "one"}
    assert _read_jsonl(journal_path)[0]["base_sequence"] == 1


def test_journal_rename_substitution_fences_ack_and_preserves_action_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _append(store, "one", evidence="must-survive")
    state = store.replay()
    journal_path = store.root / "journal.jsonl"
    journal_before = journal_path.read_bytes()
    original_rename = action_store.os.rename
    substituted = False
    intended_journal = store.root / "journal.jsonl.intended"

    def substitute_journal(src: str, dst: str, **kwargs: object) -> None:
        nonlocal substituted
        original_rename(src, dst, **kwargs)  # type: ignore[arg-type]
        if dst != "journal.jsonl" or substituted:
            return
        target = store.root / dst
        original_rename(target, intended_journal)
        target.write_bytes(journal_before)
        os.chmod(target, 0o600)
        substituted = True

    monkeypatch.setattr(action_store.os, "rename", substitute_journal)
    with pytest.raises(ActionStoreCorruptError, match="journal.jsonl binding changed"):
        store.checkpoint(expected_generation=state.generation, candidate={"candidate": "one"})
    assert substituted
    assert journal_path.read_bytes() == journal_before
    assert len(_read_jsonl(journal_path)) == 2

    monkeypatch.undo()
    settled = store.checkpoint(expected_generation=state.generation, candidate={"candidate": "one"})
    assert settled.candidate == {"candidate": "one"}
    assert _read_jsonl(journal_path)[0]["base_sequence"] == 1


def test_checkpoint_cas_is_stale_and_exact_retry_cannot_change_candidate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    state = store.replay()
    store.checkpoint(expected_generation=state.generation, candidate={"winner": "first"})
    with pytest.raises(ActionStoreConflictError, match="different candidate"):
        store.checkpoint(expected_generation=state.generation, candidate={"winner": "second"})
    _append(store, "two")
    with pytest.raises(ActionStoreStaleGenerationError):
        store.checkpoint(expected_generation=state.generation, candidate={"winner": "first"})


def test_checkpoint_cas_fences_an_append_after_replay(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    state = store.replay()
    _append(store, "two")
    with pytest.raises(ActionStoreStaleGenerationError):
        store.checkpoint(expected_generation=state.generation, candidate={"only": "one"})
    with pytest.raises(ValueError, match="invalid integer"):
        store.checkpoint(
            expected_generation=ActionStoreGeneration(1, 0, 0, True), candidate={"forged": True}
        )


def test_append_generation_cas_serializes_concurrent_new_actions(tmp_path: Path) -> None:
    root = tmp_path / "private-actions"
    store = CronActionStore(root, namespace=NAMESPACE)
    generation = store.replay().generation
    context = multiprocessing.get_context("spawn")
    read_ready = context.Queue()
    go = context.Event()
    outcomes = context.Queue()
    workers = [
        context.Process(
            target=_append_with_generation_in_process,
            args=(str(root), generation, read_ready, go, outcomes, action_id),
        )
        for action_id in ("first", "second")
    ]
    for worker in workers:
        worker.start()
    assert [read_ready.get(timeout=5) for _ in workers] == ["read", "read"]
    go.set()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0

    results = [outcomes.get(timeout=3) for _ in workers]
    assert results.count("stale") == 1
    assert len([result for result in results if isinstance(result, tuple) and result[0] == "ok"]) == 1
    assert len(store.replay().receipts) == 1


def test_append_exact_retry_precedes_stale_generation_rejection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    generation = store.replay().generation
    first = store.append(
        action_id="same-action",
        request={"operation": "opaque", "value": 1},
        expected_generation=generation,
    )

    assert store.append(
        action_id="same-action",
        request={"operation": "opaque", "value": 1},
        expected_generation=generation,
    ) == first
    with pytest.raises(ActionStoreStaleGenerationError, match="append generation is stale"):
        store.append(
            action_id="new-action",
            request={"operation": "opaque", "value": 2},
            expected_generation=generation,
        )


def test_append_and_checkpoint_race_invalidates_the_checkpoint_generation(tmp_path: Path) -> None:
    root = tmp_path / "private-actions"
    store = CronActionStore(root, namespace=NAMESPACE)
    generation = store.replay().generation
    context = multiprocessing.get_context("spawn")
    read_ready = context.Queue()
    go = context.Event()
    outcomes = context.Queue()
    worker = context.Process(
        target=_checkpoint_in_process,
        args=(str(root), read_ready, go, outcomes, "checkpoint"),
    )
    worker.start()
    assert read_ready.get(timeout=5) == "read"
    store.append(
        action_id="new-action",
        request={"operation": "opaque", "value": 1},
        expected_generation=generation,
    )
    go.set()
    worker.join(timeout=15)
    assert worker.exitcode == 0
    assert outcomes.get(timeout=3) == "lost"


def test_two_process_append_and_checkpoint_races_are_serialized(tmp_path: Path) -> None:
    root = tmp_path / "private-actions"
    CronActionStore(root, namespace=NAMESPACE)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    outcomes = context.Queue()
    workers = [
        context.Process(target=_append_in_process, args=(str(root), ready, outcomes, action_id))
        for action_id in ("first", "second")
    ]
    for worker in workers:
        worker.start()
    ready.set()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    results = [outcomes.get(timeout=3) for _ in workers]
    assert sorted(sequence for status, sequence in results if status == "ok") == [1, 2]

    read_ready = context.Queue()
    ready = context.Event()
    outcomes = context.Queue()
    workers = [
        context.Process(target=_checkpoint_in_process, args=(str(root), read_ready, ready, outcomes, marker))
        for marker in ("a", "b")
    ]
    for worker in workers:
        worker.start()
    assert [read_ready.get(timeout=5) for _ in workers] == ["read", "read"]
    ready.set()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    try:
        results = [outcomes.get(timeout=3) for _ in workers]
    except Empty as exc:  # pragma: no cover - worker error becomes legible
        raise AssertionError("checkpoint race worker did not report") from exc
    assert results.count("won") == 1 and results.count("lost") == 1


def test_legacy_conversion_is_explicit_while_corrupt_jobs_snapshot_is_ignored(tmp_path: Path) -> None:
    root = tmp_path / "private-actions"
    root.mkdir(mode=0o700)
    (root / "jobs.json").write_text("not readable JSON", encoding="utf-8")
    store = CronActionStore(root, namespace=NAMESPACE)
    _append(store, "offline-create", value="preserved jobs")
    assert (root / "jobs.json").read_text(encoding="utf-8") == "not readable JSON"

    legacy = tmp_path / "legacy"
    legacy.mkdir(mode=0o700)
    (legacy / "action.jsonl").write_text('{"action":"add"}\n', encoding="utf-8")
    with pytest.raises(ActionStoreConversionRequiredError, match="offline conversion"):
        CronActionStore(legacy, namespace=NAMESPACE)


def test_private_modes_path_symlink_lock_safety_and_windows_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    _append(store, "one")
    state = store.replay()
    store.checkpoint(expected_generation=state.generation, candidate={"x": 1})
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    for name in (".lock", "journal.jsonl", "snapshot.json"):
        assert stat.S_IMODE((store.root / name).stat().st_mode) == 0o600

    with pytest.raises(ValueError, match="absolute"):
        CronActionStore(Path("relative"), namespace=NAMESPACE)
    with pytest.raises(ValueError, match="without '..'"):
        CronActionStore((tmp_path / "safe" / ".." / "unsafe"), namespace=NAMESPACE)

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ActionStoreCorruptError, match="never a link"):
        CronActionStore(root_link, namespace=NAMESPACE)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    (unsafe / ".lock").symlink_to(tmp_path / "target")
    with pytest.raises(ActionStoreCorruptError, match="symlink"):
        CronActionStore(unsafe, namespace=NAMESPACE)

    monkeypatch.setattr(action_store.os, "name", "nt")
    with pytest.raises(ActionStorePersistenceError, match="POSIX"):
        CronActionStore(tmp_path / "windows", namespace=NAMESPACE)


def test_record_and_journal_size_limits_fail_before_an_acknowledged_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    with pytest.raises(ActionStoreConflictError, match="byte limit"):
        _append(store, "too-large", blob="x" * (action_store._MAX_REQUEST_BYTES + 1))
    assert store.replay().receipts == ()

    journal_limit = (store.root / "journal.jsonl").stat().st_size + 10
    monkeypatch.setattr(action_store, "_MAX_JOURNAL_BYTES", journal_limit)
    with pytest.raises(ActionStoreConflictError, match="private journal size"):
        _append(store, "journal-too-large", value="more than ten bytes")
    monkeypatch.undo()

    monkeypatch.setattr(action_store, "_MAX_RECEIPTS", 3)
    for index in range(3):
        _append(store, f"id-{index}", n=index)
    with pytest.raises(ActionStoreConflictError, match="receipt limit"):
        _append(store, "over-limit")
