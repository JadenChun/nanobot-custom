"""Tests for the restructured MemoryStore — pure file I/O layer."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Barrier, Event, get_ident

import pytest

import nanobot.utils.gitstore as gitstore_module
from nanobot.agent.memory import _HISTORY_ENTRY_HARD_CAP, MemoryStore
from nanobot.agent.tools.filesystem_effect import FilesystemEffectAttempt
from nanobot.utils.gitstore import GitStore, GitStoreError


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


class TestMemoryStoreBasicIO:
    def test_read_memory_returns_empty_when_missing(self, store):
        assert store.read_memory() == ""

    def test_write_and_read_memory(self, store):
        store.write_memory("hello")
        assert store.read_memory() == "hello"

    def test_read_soul_returns_empty_when_missing(self, store):
        assert store.read_soul() == ""

    def test_write_and_read_soul(self, store):
        store.write_soul("soul content")
        assert store.read_soul() == "soul content"

    def test_read_user_returns_empty_when_missing(self, store):
        assert store.read_user() == ""

    def test_write_and_read_user(self, store):
        store.write_user("user content")
        assert store.read_user() == "user content"

    def test_get_memory_context_returns_empty_when_missing(self, store):
        assert store.get_memory_context() == ""

    def test_get_memory_context_returns_formatted_content(self, store):
        store.write_memory("important fact")
        ctx = store.get_memory_context()
        assert "Long-term Memory" in ctx
        assert "important fact" in ctx


class TestHistoryWithCursor:
    def test_append_history_returns_cursor(self, store):
        cursor = store.append_history("event 1")
        assert cursor == 1
        cursor2 = store.append_history("event 2")
        assert cursor2 == 2

    def test_append_history_includes_cursor_in_file(self, store):
        store.append_history("event 1")
        content = store.read_file(store.history_file)
        data = json.loads(content)
        assert data["cursor"] == 1

    def test_append_history_includes_session_key_when_provided(self, store):
        store.append_history("event 1", session_key="telegram:chat-1")
        content = store.read_file(store.history_file)
        data = json.loads(content)
        assert data["session_key"] == "telegram:chat-1"

    def test_cursor_persists_across_appends(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        cursor = store.append_history("event 3")
        assert cursor == 3

    def test_append_history_strips_thinking_content(self, store):
        """`strip_think` must run before persistence — well-formed thinking
        blocks shouldn't land in history."""
        cursor = store.append_history("<think>reasoning</think>final answer")
        content = store.read_file(store.history_file)
        data = json.loads(content)
        assert data["cursor"] == cursor
        assert data["content"] == "final answer"

    def test_append_history_drops_pure_leak_content(self, store):
        """Regression: entries that strip down to empty (pure template-token
        leak) must NOT fall back to the raw leak. Persisting the raw text
        would re-pollute context via consolidation / replay, undoing the
        protection `strip_think` provides."""
        cursor = store.append_history("<think>nothing user-facing</think>")
        content = store.read_file(store.history_file)
        data = json.loads(content)
        assert data["cursor"] == cursor
        assert data["content"] == ""

    def test_append_history_drops_malformed_leak_prefix(self, store):
        """Channel-marker / malformed opening leaks should not survive."""
        cursor = store.append_history("<channel|>")
        content = store.read_file(store.history_file)
        data = json.loads(content)
        assert data["cursor"] == cursor
        assert data["content"] == ""

    def test_read_unprocessed_history(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        entries = store.read_unprocessed_history(since_cursor=1)
        assert len(entries) == 2
        assert entries[0]["cursor"] == 2

    def test_read_unprocessed_history_returns_all_when_cursor_zero(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_read_unprocessed_skips_entries_without_cursor(self, store):
        """Regression: entries missing the cursor key should be silently skipped."""
        store.history_file.write_text(
            '{"timestamp": "2026-04-01 10:00", "content": "no cursor"}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "valid"}\n'
            '{"cursor": 3, "timestamp": "2026-04-01 10:02", "content": "also valid"}\n',
            encoding="utf-8",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert [e["cursor"] for e in entries] == [2, 3]

    def test_read_unprocessed_skips_malformed_history_payloads(self, store):
        """Externally edited JSONL can keep an int cursor but miss required payload fields."""
        store.history_file.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "valid"}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01"}\n'
            '{"cursor": 3, "content": "missing timestamp"}\n'
            '{"cursor": 4, "timestamp": "2026-04-01 10:03", "content": 123}\n'
            '{"cursor": 5, "timestamp": "2026-04-01 10:04", "content": "bad session", "session_key": 42}\n'
            '{"cursor": 6, "timestamp": "2026-04-01 10:05", "content": "also valid", "session_key": "telegram:chat-1"}\n',
            encoding="utf-8",
        )

        entries = store.read_unprocessed_history(since_cursor=0)

        assert [e["cursor"] for e in entries] == [1, 6]
        assert [e["content"] for e in entries] == ["valid", "also valid"]

    def test_next_cursor_falls_back_when_last_entry_has_no_cursor(self, store):
        """Regression: _next_cursor should not KeyError on entries without cursor."""
        store.history_file.write_text(
            '{"timestamp": "2026-04-01 10:01", "content": "no cursor"}\n',
            encoding="utf-8",
        )
        # Delete .cursor file so _next_cursor falls back to reading JSONL
        store._cursor_file.unlink(missing_ok=True)
        # Last entry has no cursor — should safely return 1, not KeyError
        cursor = store.append_history("new event")
        assert cursor == 1

    def test_append_history_allocates_unique_cursors_under_concurrent_writes(self, store):
        """Regression: concurrent appends must not allocate duplicate cursors."""
        import threading

        writers = 16
        start = threading.Barrier(writers)
        cursors: list[int] = []
        lock = threading.Lock()

        def worker(i):
            start.wait()
            c = store.append_history(f"event {i}")
            with lock:
                cursors.append(c)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(cursors) == writers
        assert len(set(cursors)) == writers, f"duplicate cursors: {sorted(cursors)}"
        assert sorted(cursors) == list(range(1, writers + 1))
        persisted = store.read_unprocessed_history(since_cursor=0)
        assert sorted(e["cursor"] for e in persisted) == list(range(1, writers + 1))

    def test_compact_history_drops_oldest(self, tmp_path):
        store = MemoryStore(tmp_path, max_history_entries=2)
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        store.append_history("event 4")
        store.append_history("event 5")
        store.set_last_dream_cursor(5)
        store.compact_history()
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert entries[0]["cursor"] in {4, 5}

    def test_compact_history_preserves_entries_after_dream_cursor(self, tmp_path):
        store = MemoryStore(tmp_path, max_history_entries=50)
        for index in range(1, 101):
            store.append_history(f"event {index}")
        store.set_last_dream_cursor(20)

        store.compact_history()

        entries = store.read_unprocessed_history(since_cursor=0)
        assert [entry["cursor"] for entry in entries] == list(range(21, 101))

    def test_write_entries_uses_atomic_write(self, tmp_path):
        """_write_entries uses temp file + os.replace for atomicity."""
        store = MemoryStore(tmp_path)
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        entries = store.read_unprocessed_history(since_cursor=0)

        # Monitor temp file existence
        tmp_path_obj = store.history_file.with_suffix(".jsonl.tmp")
        assert not tmp_path_obj.exists()  # Should not exist initially

        # Call _write_entries
        store._write_entries(entries)

        # Temp file should be cleaned up
        assert not tmp_path_obj.exists()
        # Original file should exist
        assert store.history_file.exists()

    def test_write_entries_cleans_up_tmp_on_exception(self, tmp_path, monkeypatch):
        """Exception during _write_entries cleans up the temp file."""
        store = MemoryStore(tmp_path)
        store.append_history("event 1")
        entries = store.read_unprocessed_history(since_cursor=0)

        tmp_path_obj = store.history_file.with_suffix(".jsonl.tmp")

        # Mock os.replace to raise an exception
        def failing_replace(*args, **kwargs):
            raise RuntimeError("Simulated failure")

        monkeypatch.setattr('os.replace', failing_replace)

        with pytest.raises(RuntimeError):
            store._write_entries(entries)

        # Temp file should be cleaned up
        assert not tmp_path_obj.exists()

        # Original file should still exist (because replace failed)
        assert store.history_file.exists()


class TestAppendHistoryHardCap:
    """append_history has a defensive cap that catches new callers who forgot
    to set their own tighter cap. The default is intentionally larger than
    any current caller's per-call cap, so normal operation never trips it."""

    def test_oversized_entry_is_truncated(self, store):
        """An entry above _HISTORY_ENTRY_HARD_CAP is truncated before being persisted."""
        huge = "x" * (_HISTORY_ENTRY_HARD_CAP + 10_000)
        store.append_history(huge)
        entry = store.read_unprocessed_history(since_cursor=0)[0]
        assert len(entry["content"]) <= _HISTORY_ENTRY_HARD_CAP + 50

    def test_oversize_warning_is_emitted_once(self, store, monkeypatch):
        """Repeated oversized writes should warn only on the first occurrence."""
        records: list[str] = []
        monkeypatch.setattr(
            "nanobot.agent.memory.logger.warning",
            lambda message, *args: records.append(message.format(*args)),
        )
        huge = "x" * (_HISTORY_ENTRY_HARD_CAP + 1)
        store.append_history(huge)
        store.append_history(huge)
        store.append_history(huge)

        oversize_warnings = [r for r in records if "exceeds" in r and "chars" in r]
        assert len(oversize_warnings) == 1

    def test_custom_max_chars_overrides_default(self, store):
        """Callers that pass max_chars should get their tighter cap applied."""
        store.append_history("a" * 500, max_chars=100)
        entry = store.read_unprocessed_history(since_cursor=0)[0]
        assert len(entry["content"]) <= 150  # 100 + "\n... (truncated)"

    def test_normal_sized_entries_unaffected(self, store):
        """The hard cap must not alter entries that fit within it."""
        msg = "normal short entry"
        store.append_history(msg)
        entry = store.read_unprocessed_history(since_cursor=0)[0]
        assert entry["content"] == msg


class TestDreamCursor:
    def test_initial_cursor_is_zero(self, store):
        assert store.get_last_dream_cursor() == 0

    def test_returns_zero_when_empty(self, store):
        assert store.get_latest_cursor() == 0

    def test_returns_cursor_of_last_entry(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")

        assert store.get_latest_cursor() == 3

    def test_returns_zero_when_no_entries(self, store):
        store.history_file.write_text("", encoding="utf-8")

        assert store.get_latest_cursor() == 0

    def test_matches_next_cursor_minus_one(self, store):
        store.append_history("event 1")
        store.append_history("event 2")

        assert store.get_latest_cursor() == max(store._next_cursor() - 1, 0)

    def test_set_and_get_cursor(self, store):
        store.set_last_dream_cursor(5)
        assert store.get_last_dream_cursor() == 5

    def test_cursor_persists(self, store):
        store.set_last_dream_cursor(3)
        store2 = MemoryStore(store.workspace)
        assert store2.get_last_dream_cursor() == 3

    def test_git_restore_rolls_back_dream_cursor(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.write_memory("before")
        store.set_last_dream_cursor(1)
        assert store.git.init() is True

        store.write_memory("after")
        store.set_last_dream_cursor(2)
        dream_sha = store.git.auto_commit("dream: update")
        assert dream_sha is not None

        store.write_memory("newer")
        store.set_last_dream_cursor(3)

        restore_sha = store.git.revert(dream_sha)

        assert restore_sha is not None
        assert store.read_memory() == "before"
        assert store.get_last_dream_cursor() == 1


def _versioned_dream_store(tmp_path: Path, name: str = "workspace") -> MemoryStore:
    workspace = tmp_path / name
    workspace.mkdir()
    store = MemoryStore(workspace, sessions_root=tmp_path / f"{name}-sessions")
    store.write_soul("soul before")
    store.write_user("user before")
    store.write_memory("memory before")
    store.append_history("batch input")
    assert store.git.init()
    return store


def test_pending_dream_operation_fences_a_fresh_store(tmp_path: Path) -> None:
    store = _versioned_dream_store(tmp_path)
    store.reserve_dream_run(0, 1)
    fresh = MemoryStore(store.workspace, sessions_root=tmp_path / "workspace-sessions")

    with pytest.raises(GitStoreError, match="unresolved"):
        fresh.build_dream_prompt()
    assert fresh.get_last_dream_cursor() == 0


def test_completed_noop_run_commits_cursor_and_unlocks_batch(tmp_path: Path) -> None:
    store = _versioned_dream_store(tmp_path)
    operation_id = store.reserve_dream_run(0, 1)

    sha = store.finish_dream_run(
        operation_id,
        completed=True,
        commit_message="dream: manual run",
    )

    assert sha is not None
    assert store.get_last_dream_cursor() == 1
    assert store.git.log()[0].sha == sha
    assert store.git.dream_operation_status() is None
    assert store.build_dream_prompt() is None


def test_incomplete_run_commits_partial_files_with_old_cursor(tmp_path: Path) -> None:
    store = _versioned_dream_store(tmp_path)
    operation_id = store.reserve_dream_run(0, 1)
    store.write_soul("partial soul edit")

    sha = store.finish_dream_run(
        operation_id,
        completed=False,
        commit_message="dream: incomplete run",
    )

    assert sha is not None
    assert store.get_last_dream_cursor() == 0
    assert store.git.dream_operation_status() is None
    result = store.build_dream_prompt()
    assert result is not None
    assert result[1] == 1


def test_failed_run_commit_stays_pending_and_fresh_store_reconciles(tmp_path: Path, monkeypatch) -> None:
    store = _versioned_dream_store(tmp_path)
    operation_id = store.reserve_dream_run(0, 1)
    original = GitStore.auto_commit

    def fail_before_commit(self, message):
        if message.startswith("dream:"):
            raise GitStoreError("injected commit failure")
        return original(self, message)

    monkeypatch.setattr(GitStore, "auto_commit", fail_before_commit)
    with pytest.raises(GitStoreError):
        store.finish_dream_run(
            operation_id,
            completed=True,
            commit_message="dream: manual run",
        )
    assert store.get_last_dream_cursor() == 1
    assert store.git.dream_operation_status() is not None

    monkeypatch.setattr(GitStore, "auto_commit", original)
    fresh = MemoryStore(store.workspace, sessions_root=tmp_path / "workspace-sessions")
    assert fresh.build_dream_prompt() is None
    assert fresh.git.log()[0].message == "dream: manual run"
    assert fresh.git.dream_operation_status() is None


def test_commit_acknowledgement_loss_returns_proved_dream_commit(tmp_path: Path, monkeypatch) -> None:
    store = _versioned_dream_store(tmp_path)
    operation_id = store.reserve_dream_run(0, 1)
    original = GitStore.auto_commit

    def commit_then_raise(self, message):
        result = original(self, message)
        if message.startswith("dream:"):
            raise GitStoreError("injected lost acknowledgement")
        return result

    monkeypatch.setattr(GitStore, "auto_commit", commit_then_raise)
    sha = store.finish_dream_run(
        operation_id,
        completed=True,
        commit_message="dream: manual run",
    )

    assert sha is not None
    assert store.git.log()[0].sha == sha
    assert store.git.dream_operation_status() is None


def test_same_workspace_run_reservations_serialize_across_instances(tmp_path: Path) -> None:
    first = _versioned_dream_store(tmp_path)
    second = MemoryStore(first.workspace, sessions_root=tmp_path / "workspace-sessions")
    barrier = Barrier(2)

    def reserve(store: MemoryStore) -> str:
        barrier.wait(timeout=5)
        return store.reserve_dream_run(0, 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reserve, store) for store in (first, second)]
        outcomes: list[str | GitStoreError] = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except GitStoreError as exc:
                outcomes.append(exc)

    assert sum(isinstance(outcome, str) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, GitStoreError) for outcome in outcomes) == 1
    independent = _versioned_dream_store(tmp_path, name="independent")
    assert independent.reserve_dream_run(0, 1)


@pytest.mark.asyncio
async def test_restore_fences_an_already_issued_dream_tool_registry(tmp_path: Path) -> None:
    store = _versioned_dream_store(tmp_path)
    store.write_soul("dream version")
    store.set_last_dream_cursor(1)
    dream_sha = store.git.auto_commit("dream: changed memory")
    assert dream_sha is not None

    operation_id = store.reserve_dream_run(1, 2)
    tools = store.build_dream_tools(operation_id=operation_id)
    restored = store.git.revert(dream_sha, message_prefix="dream:")

    result = await tools.execute(
        "write_file",
        {"path": "skills/stale/SKILL.md", "content": "must not be written"},
    )
    assert "no longer current" in str(result)
    assert not (store.workspace / "skills" / "stale" / "SKILL.md").exists()
    assert restored is not None
    assert store.git.log()[0].message.startswith("revert: undo")


@pytest.mark.asyncio
async def test_issued_dream_write_serializes_with_restore_across_owners(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _versioned_dream_store(tmp_path)
    store.write_soul("dream version")
    store.set_last_dream_cursor(1)
    dream_sha = store.git.auto_commit("dream: changed memory")
    assert dream_sha is not None

    restorer = MemoryStore(store.workspace, sessions_root=tmp_path / "restore-sessions")
    operation_id = store.reserve_dream_run(1, 2)
    tools = store.build_dream_tools(operation_id=operation_id)
    target = store.workspace / "skills" / "stale" / "SKILL.md"
    entered_mutation = Event()
    release_mutation = Event()
    restore_requested_lock = Event()
    restore_thread_id: int | None = None
    original_enter = FilesystemEffectAttempt.enter

    def pause_before_mutation(cls, operation, path, intended):
        if operation == "write_file" and path == target:
            entered_mutation.set()
            if not release_mutation.wait(timeout=5):
                raise TimeoutError("Dream write barrier was not released")
        return original_enter(operation, path, intended)

    original_workspace_lock = gitstore_module.dream_workspace_lock

    def observe_restore_lock(workspace: Path):
        if get_ident() == restore_thread_id:
            restore_requested_lock.set()
        return original_workspace_lock(workspace)

    monkeypatch.setattr(FilesystemEffectAttempt, "enter", classmethod(pause_before_mutation))
    monkeypatch.setattr(gitstore_module, "dream_workspace_lock", observe_restore_lock)

    def issue_write() -> object:
        return asyncio.run(
            tools.execute(
                "write_file",
                {"path": "skills/stale/SKILL.md", "content": "stale Dream write"},
            )
        )

    def restore() -> str | None:
        nonlocal restore_thread_id
        restore_thread_id = get_ident()
        return restorer.git.revert(dream_sha, message_prefix="dream:")

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(issue_write)
        try:
            assert entered_mutation.wait(timeout=5)
            restore_future = pool.submit(restore)
            assert restore_requested_lock.wait(timeout=5)
            assert not restore_future.done()
        finally:
            release_mutation.set()
        write_result = writer.result(timeout=10)
        restored = restore_future.result(timeout=10)

    assert "Successfully wrote" in str(write_result)
    assert restored is not None
    assert not target.exists()
    assert restorer.get_last_dream_cursor() == 0
    assert restorer.git.dream_operation_status() is None
    prompt = restorer.build_dream_prompt()
    assert prompt is not None and prompt[1] == 1
    with pytest.raises(GitStoreError, match="ownership is no longer current"):
        store.finish_dream_run(
            operation_id,
            completed=True,
            commit_message="dream: stale run completion",
        )

    stale_result = await tools.execute(
        "write_file",
        {"path": "skills/late/SKILL.md", "content": "must be refused"},
    )
    assert "no longer current" in str(stale_result)
    assert not (store.workspace / "skills" / "late" / "SKILL.md").exists()
    assert restorer.get_last_dream_cursor() == 0


class TestLegacyHistoryMigration:
    def test_read_unprocessed_history_handles_entries_without_cursor(self, store):
        """JSONL entries with cursor=1 are correctly parsed and returned."""
        store.history_file.write_text(
            '{"cursor": 1, "timestamp": "2026-03-30 14:30", "content": "Old event"}\n',
            encoding="utf-8",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert entries[0]["cursor"] == 1

    def test_migrates_legacy_history_md_preserving_partial_entries(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_content = (
            "[2026-04-01 10:00] User prefers dark mode.\n\n"
            "[2026-04-01 10:05] [RAW] 2 messages\n"
            "[2026-04-01 10:04] USER: hello\n"
            "[2026-04-01 10:04] ASSISTANT: hi\n\n"
            "Legacy chunk without timestamp.\n"
            "Keep whatever content we can recover.\n"
        )
        legacy_file.write_text(legacy_content, encoding="utf-8")

        store = MemoryStore(tmp_path)
        fallback_timestamp = datetime.fromtimestamp(
            (memory_dir / "HISTORY.md.bak").stat().st_mtime,
        ).strftime("%Y-%m-%d %H:%M")

        entries = store.read_unprocessed_history(since_cursor=0)
        assert [entry["cursor"] for entry in entries] == [1, 2, 3]
        assert entries[0]["timestamp"] == "2026-04-01 10:00"
        assert entries[0]["content"] == "User prefers dark mode."
        assert entries[1]["timestamp"] == "2026-04-01 10:05"
        assert entries[1]["content"].startswith("[RAW] 2 messages")
        assert "USER: hello" in entries[1]["content"]
        assert entries[2]["timestamp"] == fallback_timestamp
        assert entries[2]["content"].startswith("Legacy chunk without timestamp.")
        assert store.read_file(store._cursor_file).strip() == "3"
        assert store.read_file(store._dream_cursor_file).strip() == "3"
        assert not legacy_file.exists()
        assert (memory_dir / "HISTORY.md.bak").read_text(encoding="utf-8") == legacy_content

    def test_migrates_consecutive_entries_without_blank_lines(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_content = (
            "[2026-04-01 10:00] First event.\n"
            "[2026-04-01 10:01] Second event.\n"
            "[2026-04-01 10:02] Third event.\n"
        )
        legacy_file.write_text(legacy_content, encoding="utf-8")

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 3
        assert [entry["content"] for entry in entries] == [
            "First event.",
            "Second event.",
            "Third event.",
        ]

    def test_raw_archive_stays_single_entry_while_following_events_split(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_content = (
            "[2026-04-01 10:05] [RAW] 2 messages\n"
            "[2026-04-01 10:04] USER: hello\n"
            "[2026-04-01 10:04] ASSISTANT: hi\n"
            "[2026-04-01 10:06] Normal event after raw block.\n"
        )
        legacy_file.write_text(legacy_content, encoding="utf-8")

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert entries[0]["content"].startswith("[RAW] 2 messages")
        assert "USER: hello" in entries[0]["content"]
        assert entries[1]["content"] == "Normal event after raw block."

    def test_nonstandard_date_headers_still_start_new_entries(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_content = (
            "[2026-03-25–2026-04-02] Multi-day summary.\n[2026-03-26/27] Cross-day summary.\n"
        )
        legacy_file.write_text(legacy_content, encoding="utf-8")

        store = MemoryStore(tmp_path)
        fallback_timestamp = datetime.fromtimestamp(
            (memory_dir / "HISTORY.md.bak").stat().st_mtime,
        ).strftime("%Y-%m-%d %H:%M")

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert entries[0]["timestamp"] == fallback_timestamp
        assert entries[0]["content"] == "[2026-03-25–2026-04-02] Multi-day summary."
        assert entries[1]["timestamp"] == fallback_timestamp
        assert entries[1]["content"] == "[2026-03-26/27] Cross-day summary."

    def test_existing_history_jsonl_skips_legacy_migration(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        history_file = memory_dir / "history.jsonl"
        history_file.write_text(
            '{"cursor": 7, "timestamp": "2026-04-01 12:00", "content": "existing"}\n',
            encoding="utf-8",
        )
        legacy_file = memory_dir / "HISTORY.md"
        legacy_file.write_text("[2026-04-01 10:00] legacy\n\n", encoding="utf-8")

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert entries[0]["cursor"] == 7
        assert entries[0]["content"] == "existing"
        assert legacy_file.exists()
        assert not (memory_dir / "HISTORY.md.bak").exists()

    def test_empty_history_jsonl_still_allows_legacy_migration(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        history_file = memory_dir / "history.jsonl"
        history_file.write_text("", encoding="utf-8")
        legacy_file = memory_dir / "HISTORY.md"
        legacy_file.write_text("[2026-04-01 10:00] legacy\n\n", encoding="utf-8")

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert entries[0]["cursor"] == 1
        assert entries[0]["timestamp"] == "2026-04-01 10:00"
        assert entries[0]["content"] == "legacy"
        assert not legacy_file.exists()
        assert (memory_dir / "HISTORY.md.bak").exists()

    def test_migrates_legacy_history_with_invalid_utf8_bytes(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_file.write_bytes(b"[2026-04-01 10:00] Broken \xff data still needs migration.\n\n")

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert entries[0]["timestamp"] == "2026-04-01 10:00"
        assert "Broken" in entries[0]["content"]
        assert "migration." in entries[0]["content"]


def test_history_skips_non_dict_jsonl_lines(tmp_path: Path) -> None:
    """Null/list/bool history lines must not crash reads or appends."""
    memory = MemoryStore(tmp_path)
    memory.history_file.parent.mkdir(parents=True, exist_ok=True)
    memory.history_file.write_text(
        "\n".join([
            "null",
            "[1, 2]",
            "true",
            json.dumps({
                "cursor": 1,
                "timestamp": "2026-01-01T00:00:00",
                "content": "kept",
                "session_key": "cli:t",
            }),
            "",
        ]),
        encoding="utf-8",
    )
    entries = memory.read_unprocessed_history(since_cursor=0)
    assert entries == [{
        "cursor": 1,
        "timestamp": "2026-01-01T00:00:00",
        "content": "kept",
        "session_key": "cli:t",
    }]
    next_cursor = memory.append_history("next", session_key="cli:t")
    assert next_cursor == 2

def test_raw_archive_handles_none_timestamp_and_missing_role(tmp_path: Path) -> None:
    """raw_archive and _format_messages must safely format messages with None timestamp or missing role.

    Prevents TypeError on NoneType[:16] slicing and KeyError on missing 'role'
    when raw-dumping unconsolidated history entries without timestamps or role fields.
    """
    memory = MemoryStore(tmp_path)
    messages = [
        {"content": "message with none timestamp", "timestamp": None, "role": "user"},
        {"content": "message with int timestamp", "timestamp": 1720000000, "role": "assistant"},
        {"content": "message with missing role", "timestamp": "2026-07-28T12:00:00"},
    ]
    memory.raw_archive(messages, session_key="cli:test")
    raw_history = memory.history_file.read_text(encoding="utf-8")
    assert "[?] USER: message with none timestamp" in raw_history
    assert "[1720000000] ASSISTANT: message with int timestamp" in raw_history
    assert "[2026-07-28T12:00] UNKNOWN: message with missing role" in raw_history
