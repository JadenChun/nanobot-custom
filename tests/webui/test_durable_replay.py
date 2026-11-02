from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace

from nanobot.webui.replay_store import DurableWebUIReplay


def _result(value: object):
    return SimpleNamespace(
        status=None,
        message=None,
        result=value,
        workspace_scope=None,
        workspace_permissions=None,
    )


def test_durable_replay_restart_completed_and_uncertain(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    store = DurableWebUIReplay(path)
    assert store.lookup("req-1", "sidebar.update", "digest", None) == ("new", None)
    assert store.begin("req-1", "sidebar.update", "digest", None) is True

    restarted = DurableWebUIReplay(path)
    assert restarted.lookup("req-1", "sidebar.update", "digest", None)[0] == "uncertain"
    assert restarted.begin("req-1", "sidebar.update", "digest", None) is False

    restarted.complete("req-1", _result({"ok": True}))
    completed = DurableWebUIReplay(path)
    state, entry = completed.lookup("req-1", "sidebar.update", "digest", None)
    assert state == "completed"
    assert entry is not None
    assert entry["result"]["result"] == {"ok": True}
    assert completed.lookup("req-1", "sidebar.update", "other", None)[0] == "mismatch"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_durable_replay_is_bounded_and_can_mark_uncertain(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    store = DurableWebUIReplay(path, max_entries=2)
    for index in range(3):
        request_id = f"req-{index}"
        assert store.begin(request_id, "sidebar.update", "digest", None)
        store.complete(request_id, _result({"index": index}))
    reloaded = DurableWebUIReplay(path, max_entries=2)
    assert len(reloaded._entries) <= 2

    assert store.begin("req-uncertain", "sidebar.update", "digest", None)
    store.uncertain("req-uncertain")
    assert store.lookup("req-uncertain", "sidebar.update", "digest", None)[0] == "uncertain"


def test_durable_replay_fails_closed_on_corrupt_ledger(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    path.write_text("{not-json")
    store = DurableWebUIReplay(path)
    assert store.lookup("req", "sidebar.update", "digest", None)[0] == "uncertain"
    assert store.begin("req", "sidebar.update", "digest", None) is False


def test_in_progress_replays_are_never_evicted_at_capacity(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    store = DurableWebUIReplay(path, max_entries=2)
    assert store.begin("req-a", "sidebar.update", "digest", None)
    assert store.begin("req-b", "sidebar.update", "digest", None)
    assert store.begin("req-c", "sidebar.update", "digest", None) is False
    restarted = DurableWebUIReplay(path, max_entries=2)
    assert restarted.lookup("req-a", "sidebar.update", "digest", None)[0] == "uncertain"
    assert restarted.lookup("req-b", "sidebar.update", "digest", None)[0] == "uncertain"
    restarted.complete("req-a", _result({"ok": True}))
    assert restarted.begin("req-c", "sidebar.update", "digest", None) is True


def test_replay_liveness_ages_crash_windows_and_releases_capacity(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    now = 1000.0
    store = DurableWebUIReplay(path, max_entries=1, clock=lambda: now)
    assert store.begin("req-old", "sidebar.update", "digest", None)

    now += 25 * 60 * 60
    assert store.begin("req-wait", "sidebar.update", "digest", None) is False
    assert store.lookup("req-old", "sidebar.update", "digest", None)[0] == "uncertain"

    now += 8 * 24 * 60 * 60
    assert store.begin("req-new", "sidebar.update", "digest", None) is True
    assert store.lookup("req-old", "sidebar.update", "digest", None)[0] == "new"


def test_rejected_admission_persists_aged_crash_windows(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    now = 1000.0
    store = DurableWebUIReplay(path, max_entries=1, clock=lambda: now)
    assert store.begin("req-old", "sidebar.update", "digest", None)

    now += 25 * 60 * 60
    assert store.begin("req-wait", "sidebar.update", "digest", None) is False

    restarted = DurableWebUIReplay(path, max_entries=1, clock=lambda: now)
    assert restarted.lookup("req-old", "sidebar.update", "digest", None)[0] == "uncertain"

    now += 8 * 24 * 60 * 60
    assert store.begin("req-new", "sidebar.update", "digest", None) is True
    assert store.lookup("req-old", "sidebar.update", "digest", None)[0] == "new"
