import json
import os
from pathlib import Path

import pytest

from nanobot.agent.run_store import RunStore
from nanobot.agent.turn import (
    RunRecord,
    RunStatus,
    SessionRunRef,
    TurnSource,
)
from nanobot.session.manager import SessionManager


def _record(run_id: str = "run-1") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        source=TurnSource.DIRECT,
        status=RunStatus.RUNNING,
        session_ref=SessionRunRef(session_key="direct:chat", run_id=run_id),
        metadata={
            "channel": "cli",
            "safe": {"count": 1},
            "prompt": "must not be persisted",
            "tool_args": {"path": "must not be persisted"},
            "api_key": "must not be persisted",
        },
    )


def test_run_record_round_trips_sanitized_schema_atomically(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    record = _record()

    store.save(record)
    path = tmp_path / "runs" / "run-1.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["run_id"] == "run-1"
    assert payload["metadata"] == {"channel": "cli", "safe": {"count": 1}}
    assert "prompt" not in json.dumps(payload)
    assert "api_key" not in json.dumps(payload)
    assert "tool_args" not in json.dumps(payload)

    loaded = store.load("run-1")
    assert loaded is not None
    assert loaded.to_dict() == record.to_dict()


def test_interrupted_atomic_write_keeps_previous_record(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path)
    store.save(_record())
    path = tmp_path / "runs" / "run-1.json"
    original = path.read_text()

    def interrupted_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr("nanobot.agent.run_store.os.replace", interrupted_replace)
    with pytest.raises(KeyboardInterrupt):
        store.save(_record())

    assert path.read_text() == original


def test_update_and_finalize_persist_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.save(_record())

    updated = store.update("run-1", metadata={"safe": True})
    assert updated.status is RunStatus.RUNNING
    finalized = store.finalize("run-1", status=RunStatus.COMPLETED, stop_reason="done")

    assert finalized.status is RunStatus.COMPLETED
    assert finalized.completed_at is not None
    assert store.get("run-1") == finalized


def test_load_trace_selects_only_requested_run_messages(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("direct:chat")
    session.messages.extend(
        [
            {"role": "user", "content": "one", "_run_id": "run-1"},
            {"role": "assistant", "content": "two", "_run_id": "run-2"},
            {"role": "tool", "content": "three", "_run_id": "run-1"},
        ]
    )
    manager.save(session)

    store = RunStore(tmp_path, session_manager=manager)
    store.save(_record())

    trace = store.load_trace("run-1")
    assert [message["content"] for message in trace] == ["one", "three"]
    assert all(message.get("_run_id") == "run-1" for message in trace)


@pytest.mark.parametrize("existing", ["valid", "corrupt", "symlink"])
def test_cancelled_insert_never_replaces_destination(tmp_path, existing):
    store = RunStore(tmp_path)
    path = store.path_for("run-1")
    if existing == "valid":
        store.save(_record())
    elif existing == "corrupt":
        path.write_text("not json")
    else:
        target = tmp_path / "target"
        target.write_text("private target")
        path.symlink_to(target)
    before = path.read_bytes()
    record = _record()
    record.status = RunStatus.CANCELLED
    assert not store.insert_cancelled_terminal(record)
    assert path.read_bytes() == before
    assert path.is_symlink() == (existing == "symlink")
    assert not list(store.runs_dir.glob("*.tmp"))


def test_cancelled_insert_competitor_after_render_wins(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    record = _record()
    record.status = RunStatus.CANCELLED
    link = os.link
    competitor = b"competitor complete bytes\n"

    def competing_link(source, destination):
        assert json.loads(Path(source).read_text())["status"] == "cancelled"
        Path(destination).write_bytes(competitor)
        link(source, destination)

    monkeypatch.setattr("nanobot.agent.run_store.os.link", competing_link)
    assert not store.insert_cancelled_terminal(record)
    assert store.path_for(record.run_id).read_bytes() == competitor
    assert not list(store.runs_dir.glob("*.tmp"))


def test_cancelled_insert_two_instances_one_complete_private_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)

    def insert(marker):
        store = RunStore(tmp_path)
        record = _record()
        record.status = RunStatus.CANCELLED
        record.metadata = {"safe": marker, "prompt": "private"}
        barrier.wait(timeout=5)
        return store.insert_cancelled_terminal(record)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(insert, ["one", "two"]))
    assert sorted(outcomes) == [False, True]
    path = RunStore(tmp_path).path_for("run-1")
    payload = json.loads(path.read_text())
    assert payload["metadata"] in [{"safe": "one"}, {"safe": "two"}]
    assert payload["status"] == "cancelled"
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("fault", ["render", "fsync", "link", "unlink"])
def test_cancelled_insert_faults_clean_staging(tmp_path, monkeypatch, fault):
    store = RunStore(tmp_path)
    record = _record()
    record.status = RunStatus.CANCELLED

    def fail(*args, **kwargs):
        raise OSError("private fault body")

    if fault == "unlink":
        unlink = os.unlink

        def unlink_then_fail(path):
            unlink(path)
            fail()

        monkeypatch.setattr("nanobot.agent.run_store.os.unlink", unlink_then_fail)
    else:
        monkeypatch.setattr("nanobot.agent.run_store." + {
            "render": "json.dump", "fsync": "os.fsync", "link": "os.link",
        }[fault], fail)
    with pytest.raises(OSError):
        store.insert_cancelled_terminal(record)
    assert not list(store.runs_dir.glob("*.tmp"))
    assert store.path_for(record.run_id).exists() == (fault == "unlink")


def test_cancelled_insert_rejects_nonterminal_record(tmp_path):
    store = RunStore(tmp_path)
    with pytest.raises(ValueError):
        store.insert_cancelled_terminal(_record())
    assert not list(store.runs_dir.iterdir())
