"""Tests for GitStore — git-backed version control for memory files."""

import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import pytest

import nanobot.utils.gitstore as gitstore_module
from nanobot.agent.memory import MemoryStore
from nanobot.utils.gitstore import CommitInfo, GitStore, GitStoreError

TRACKED = ["SOUL.md", "USER.md", "memory/MEMORY.md"]


@pytest.fixture
def git(tmp_path):
    """Uninitialized GitStore."""
    return GitStore(tmp_path, tracked_files=TRACKED)


@pytest.fixture
def git_ready(git):
    """Initialized GitStore with one initial commit."""
    git.init()
    return git


class TestInit:
    def test_not_initialized_by_default(self, git, tmp_path):
        assert not git.is_initialized()
        assert not (tmp_path / ".git").is_dir()

    def test_init_creates_git_dir(self, git, tmp_path):
        assert git.init()
        assert (tmp_path / ".git").is_dir()

    def test_init_idempotent(self, git_ready):
        assert not git_ready.init()

    def test_init_creates_gitignore(self, git_ready):
        gi = git_ready._workspace / ".gitignore"
        assert gi.exists()
        content = gi.read_text(encoding="utf-8")
        for f in TRACKED:
            assert f"!{f}" in content

    def test_init_touches_tracked_files(self, git_ready):
        for f in TRACKED:
            assert (git_ready._workspace / f).exists()

    def test_init_makes_initial_commit(self, git_ready):
        commits = git_ready.log()
        assert len(commits) == 1
        assert "init" in commits[0].message

    def test_init_failure_is_explicit(self, git):
        with patch("dulwich.porcelain.init", side_effect=OSError("cannot initialize")):
            with pytest.raises(GitStoreError, match="init failed"):
                git.init()


class TestBuildGitignore:
    def test_subdirectory_dirs(self, git):
        content = git._build_gitignore()
        assert "!memory/\n" in content
        for f in TRACKED:
            assert f"!{f}\n" in content
        assert content.startswith("/*\n")

    def test_root_level_files_no_dir_entries(self, tmp_path):
        gs = GitStore(tmp_path, tracked_files=["a.md", "b.md"])
        content = gs._build_gitignore()
        assert "!a.md\n" in content
        assert "!b.md\n" in content
        dir_lines = [
            line
            for line in content.split("\n")
            if line.startswith("!") and line.endswith("/")
        ]
        assert dir_lines == []


class TestAutoCommit:
    def test_returns_none_when_not_initialized(self, git):
        assert git.auto_commit("test") is None

    def test_commits_file_change(self, git_ready):
        (git_ready._workspace / "SOUL.md").write_text("updated", encoding="utf-8")
        sha = git_ready.auto_commit("update soul")
        assert sha is not None
        assert len(sha) == 8

    def test_returns_none_when_no_changes(self, git_ready):
        assert git_ready.auto_commit("no change") is None

    def test_commit_appears_in_log(self, git_ready):
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("v2", encoding="utf-8")
        sha = git_ready.auto_commit("update soul")
        commits = git_ready.log()
        assert len(commits) == 2
        assert commits[0].sha == sha

    def test_commits_same_size_rewrite_with_unchanged_mtime(self, git_ready):
        path = git_ready._workspace / "SOUL.md"
        path.write_text("v1", encoding="utf-8")
        git_ready.auto_commit("v1")
        previous_stat = path.stat()

        path.write_text("v2", encoding="utf-8")
        os.utime(path, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))

        assert git_ready.auto_commit("v2") is not None
        assert [commit.message for commit in git_ready.log()[:2]] == ["v2", "v1"]

    def test_does_not_create_empty_commits(self, git_ready):
        git_ready.auto_commit("nothing 1")
        git_ready.auto_commit("nothing 2")
        assert len(git_ready.log()) == 1  # only init commit

    def test_status_failure_is_explicit(self, git_ready):
        with patch("dulwich.porcelain.status", side_effect=OSError("broken index")):
            with pytest.raises(GitStoreError, match="auto-commit failed"):
                git_ready.auto_commit("update")


class TestLog:
    def test_empty_when_not_initialized(self, git):
        assert git.log() == []

    def test_newest_first(self, git_ready):
        ws = git_ready._workspace
        for i in range(3):
            (ws / "SOUL.md").write_text(f"v{i}", encoding="utf-8")
            git_ready.auto_commit(f"commit {i}")

        commits = git_ready.log()
        assert len(commits) == 4  # init + 3
        assert "commit 2" in commits[0].message
        assert "init" in commits[-1].message

    def test_max_entries(self, git_ready):
        ws = git_ready._workspace
        for i in range(10):
            (ws / "SOUL.md").write_text(f"v{i}", encoding="utf-8")
            git_ready.auto_commit(f"c{i}")
        assert len(git_ready.log(max_entries=3)) == 3

    def test_message_prefix_skips_unrelated_commits_before_counting_limit(self, git_ready):
        ws = git_ready._workspace
        messages = ["dream: older", "backup: first", "dream: latest", "backup: newest"]
        for i, message in enumerate(messages):
            (ws / "SOUL.md").write_text(f"v{i}", encoding="utf-8")
            git_ready.auto_commit(message)

        commits = git_ready.log(max_entries=2, message_prefix="dream:")

        assert [commit.message for commit in commits] == ["dream: latest", "dream: older"]

    def test_commit_info_fields(self, git_ready):
        c = git_ready.log()[0]
        assert isinstance(c, CommitInfo)
        assert len(c.sha) == 8
        assert c.timestamp
        assert c.message


class TestDiffCommits:
    def test_empty_when_not_initialized(self, git):
        assert git.diff_commits("a", "b") == ""

    def test_diff_between_two_commits(self, git_ready):
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("original", encoding="utf-8")
        git_ready.auto_commit("v1")
        (ws / "SOUL.md").write_text("modified", encoding="utf-8")
        git_ready.auto_commit("v2")

        commits = git_ready.log()
        diff = git_ready.diff_commits(commits[1].sha, commits[0].sha)
        assert "modified" in diff

    def test_invalid_sha_returns_empty(self, git_ready):
        assert git_ready.diff_commits("deadbeef", "cafebabe") == ""


class TestShowCommitDiff:
    def test_returns_commit_with_diff(self, git_ready):
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("content", encoding="utf-8")
        sha = git_ready.auto_commit("add content")
        result = git_ready.show_commit_diff(sha)
        assert result is not None
        commit, diff = result
        assert commit.sha == sha
        assert "content" in diff

    def test_first_commit_has_empty_diff(self, git_ready):
        init_sha = git_ready.log()[-1].sha
        result = git_ready.show_commit_diff(init_sha)
        assert result is not None
        _, diff = result
        assert diff == ""

    def test_returns_none_for_unknown(self, git_ready):
        assert git_ready.show_commit_diff("deadbeef") is None

    def test_message_prefix_finds_commit_beyond_unrelated_history_window(self, git_ready):
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("dream content", encoding="utf-8")
        dream_sha = git_ready.auto_commit("dream: latest")
        for i in range(20):
            (ws / "SOUL.md").write_text(f"backup {i}", encoding="utf-8")
            git_ready.auto_commit(f"backup: {i}")

        result = git_ready.show_commit_diff(
            dream_sha,
            max_entries=1,
            message_prefix="dream:",
        )

        assert result is not None
        commit, diff = result
        assert commit.sha == dream_sha
        assert "dream content" in diff


class TestCommitInfoFormat:
    def test_format_with_diff(self):
        from nanobot.utils.gitstore import CommitInfo
        c = CommitInfo(sha="abcd1234", message="test commit\nsecond line", timestamp="2026-04-02 12:00")
        result = c.format(diff="some diff")
        assert "test commit" in result
        assert "`abcd1234`" in result
        assert "some diff" in result

    def test_format_without_diff(self):
        from nanobot.utils.gitstore import CommitInfo
        c = CommitInfo(sha="abcd1234", message="test", timestamp="2026-04-02 12:00")
        result = c.format()
        assert "(no file changes)" in result

    def test_format_empty_message(self):
        from nanobot.utils.gitstore import CommitInfo
        c = CommitInfo(sha="abcd1234", message="", timestamp="2026-04-02 12:00")
        result = c.format()
        assert "(no message)" in result
        assert "`abcd1234`" in result
        assert c.subject() == "(no message)"


class TestRevert:
    def test_returns_none_when_not_initialized(self, git):
        assert git.revert("abc") is None

    def test_undoes_commit_changes(self, git_ready):
        """revert(sha) should undo the given commit by restoring to its parent."""
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("v2 content", encoding="utf-8")
        git_ready.auto_commit("v2")

        commits = git_ready.log()
        # commits[0] = v2 (HEAD), commits[1] = init
        # Revert v2 → restore to init's state (empty SOUL.md)
        new_sha = git_ready.revert(commits[0].sha)
        assert new_sha is not None
        assert (ws / "SOUL.md").read_text(encoding="utf-8") == ""

    def test_root_commit_returns_none(self, git_ready):
        """Cannot revert the root commit (no parent to restore to)."""
        commits = git_ready.log()
        assert len(commits) == 1
        assert git_ready.revert(commits[0].sha) is None

    def test_invalid_sha_returns_none(self, git_ready):
        assert git_ready.revert("deadbeef") is None

    def test_message_prefix_rejects_unrelated_commit_without_changing_files(self, git_ready):
        ws = git_ready._workspace
        (ws / "SOUL.md").write_text("dream v1", encoding="utf-8")
        git_ready.auto_commit("dream: v1")
        (ws / "SOUL.md").write_text("backup state", encoding="utf-8")
        backup_sha = git_ready.auto_commit("backup: workspace")
        (ws / "SOUL.md").write_text("dream v2", encoding="utf-8")
        latest_sha = git_ready.auto_commit("dream: v2")

        assert git_ready.revert(backup_sha, message_prefix="dream:") is None
        assert (ws / "SOUL.md").read_text(encoding="utf-8") == "dream v2"
        assert git_ready.log()[0].sha == latest_sha


def _dream_restore_fixture(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = MemoryStore(workspace, sessions_root=tmp_path / "sessions")
    store.write_soul("soul before")
    store.write_user("user before")
    store.write_memory("memory before")
    store.set_last_dream_cursor(1)
    retained = workspace / "skills" / "nested" / "retained.bin"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"before\xff\x00")
    retained.chmod(0o755)
    deleted = workspace / "skills" / "old" / "gone.bin"
    deleted.parent.mkdir(parents=True)
    deleted.write_bytes(b"baseline deletion target\x00")
    store.git.init()

    store.write_soul("soul after")
    store.write_user("user after")
    store.write_memory("memory after")
    store.set_last_dream_cursor(2)
    retained.write_bytes(b"after\xfe\x00")
    created = workspace / "skills" / "new" / "added.bin"
    created.parent.mkdir(parents=True)
    created.write_bytes(b"new admitted file")
    deleted.unlink()
    deleted.parent.rmdir()
    sha = store.git.auto_commit("dream: bounded restore test")
    assert sha is not None
    return store, sha, retained, deleted, created


def _assert_dream_snapshot_before(store, retained, deleted, created):
    import stat

    assert store.read_soul() == "soul before"
    assert store.read_user() == "user before"
    assert store.read_memory() == "memory before"
    assert store.get_last_dream_cursor() == 1
    assert retained.read_bytes() == b"before\xff\x00"
    assert stat.S_IMODE(retained.stat().st_mode) == 0o755
    assert deleted.read_bytes() == b"baseline deletion target\x00"
    assert not created.exists()


@pytest.mark.parametrize(
    "fault_path",
    [
        "memory/.dream_cursor",
        "memory/MEMORY.md",
        "SOUL.md",
        "USER.md",
        "skills/old/gone.bin",
        "skills/nested/retained.bin",
    ],
)
def test_restore_resumes_after_atomic_file_publication_failure(tmp_path, monkeypatch, fault_path):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    original = gitstore_module.atomic_publish_admitted_file
    faulted = False

    def publish_then_fail(workspace, path, data, mode, *, allowed_current):
        nonlocal faulted
        result = original(workspace, path, data, mode, allowed_current=allowed_current)
        if path == fault_path and not faulted:
            faulted = True
            raise OSError("injected publication interruption")
        return result

    monkeypatch.setattr(gitstore_module, "atomic_publish_admitted_file", publish_then_fail)
    with pytest.raises(GitStoreError):
        store.git.revert(sha, message_prefix="dream:")
    assert faulted

    fresh = MemoryStore(store.workspace, sessions_root=tmp_path / "sessions")
    fresh.reconcile_dream_operation()
    _assert_dream_snapshot_before(fresh, retained, deleted, created)
    assert fresh.git.log()[0].message.startswith("revert: undo")


def test_restore_resumes_after_added_file_deletion_failure(tmp_path, monkeypatch):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    original = gitstore_module.atomic_delete_admitted_file
    faulted = False

    def delete_then_fail(workspace, path, *, allowed_current):
        nonlocal faulted
        original(workspace, path, allowed_current=allowed_current)
        if path == "skills/new/added.bin" and not faulted:
            faulted = True
            raise OSError("injected deletion interruption")

    monkeypatch.setattr(gitstore_module, "atomic_delete_admitted_file", delete_then_fail)
    with pytest.raises(GitStoreError):
        store.git.revert(sha, message_prefix="dream:")
    assert faulted

    monkeypatch.setattr(gitstore_module, "atomic_delete_admitted_file", original)
    restored = store.git.revert(sha, message_prefix="dream:")
    assert restored is not None
    _assert_dream_snapshot_before(store, retained, deleted, created)


def test_restore_reconciles_commit_that_published_before_error(tmp_path, monkeypatch):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    original = GitStore.auto_commit
    observed: list[str | None] = []

    def commit_then_fail(self, message):
        result = original(self, message)
        if message.startswith("revert:"):
            observed.append(result)
            raise GitStoreError("injected lost acknowledgement")
        return result

    monkeypatch.setattr(GitStore, "auto_commit", commit_then_fail)
    restored = store.git.revert(sha, message_prefix="dream:")

    assert restored is not None
    assert observed == [restored]
    assert store.git.log()[0].sha == restored
    _assert_dream_snapshot_before(store, retained, deleted, created)


def test_restore_before_commit_error_stays_pending_and_retries(tmp_path, monkeypatch):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    original = GitStore.auto_commit

    def fail_before_commit(self, message):
        if message.startswith("revert:"):
            raise GitStoreError("injected pre-publication failure")
        return original(self, message)

    monkeypatch.setattr(GitStore, "auto_commit", fail_before_commit)
    with pytest.raises(GitStoreError):
        store.git.revert(sha, message_prefix="dream:")
    assert (store.workspace / ".git" / "nanobot-dream-operation.json").is_file()

    monkeypatch.setattr(GitStore, "auto_commit", original)
    restored = store.git.revert(sha, message_prefix="dream:")
    assert restored is not None
    assert not (store.workspace / ".git" / "nanobot-dream-operation.json").exists()
    _assert_dream_snapshot_before(store, retained, deleted, created)


def test_restore_commit_cleanup_error_is_reconciled_on_retry(tmp_path, monkeypatch):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    original = gitstore_module.clear_operation
    faulted = False

    def clear_then_fail(workspace):
        nonlocal faulted
        if not faulted:
            faulted = True
            raise OSError("injected cleanup failure")
        original(workspace)

    monkeypatch.setattr(gitstore_module, "clear_operation", clear_then_fail)
    with pytest.raises(GitStoreError):
        store.git.revert(sha, message_prefix="dream:")
    assert faulted
    assert store.git.log()[0].message.startswith("revert: undo")
    assert (store.workspace / ".git" / "nanobot-dream-operation.json").is_file()

    monkeypatch.setattr(gitstore_module, "clear_operation", original)
    restored = store.git.revert(sha, message_prefix="dream:")
    assert restored == store.git.log()[0].sha
    assert not (store.workspace / ".git" / "nanobot-dream-operation.json").exists()
    _assert_dream_snapshot_before(store, retained, deleted, created)


def test_malformed_restore_owner_refuses_before_snapshot_mutation(tmp_path):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    before = {
        path: (store.workspace / path).read_bytes()
        for path in ("SOUL.md", "USER.md", "memory/MEMORY.md", "memory/.dream_cursor")
    }
    operation = store.workspace / ".git" / "nanobot-dream-operation.json"
    operation.write_bytes(b'{"version":true,"unknown":"record"}')

    with pytest.raises(GitStoreError):
        store.git.revert(sha, message_prefix="dream:")

    assert {path: (store.workspace / path).read_bytes() for path in before} == before
    assert retained.read_bytes() == b"after\xfe\x00"
    assert not deleted.exists()
    assert created.read_bytes() == b"new admitted file"
    assert operation.read_bytes() == b'{"version":true,"unknown":"record"}'


def test_missing_restore_target_blob_refuses_before_snapshot_mutation(tmp_path):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    from dulwich.repo import Repo

    full_sha = store.git._resolve_sha(sha)
    assert full_sha is not None
    with Repo(str(store.workspace)) as repo:
        target = repo[full_sha]
        parent = repo[target.parents[0]]
        tree = repo[parent.tree]
        soul_blob = tree[b"SOUL.md"][1]
    loose_blob = store.workspace / ".git" / "objects" / soul_blob[:2].decode() / soul_blob[2:].decode()
    assert loose_blob.is_file()
    loose_blob.unlink()

    operation = store.workspace / ".git" / "nanobot-dream-operation.json"
    with pytest.raises(GitStoreError):
        store.git.revert(sha, message_prefix="dream:")

    assert store.soul_file.read_text(encoding="utf-8") == "soul after"
    assert store.user_file.read_text(encoding="utf-8") == "user after"
    assert store.memory_file.read_text(encoding="utf-8") == "memory after"
    assert store.get_last_dream_cursor() == 2
    assert retained.read_bytes() == b"after\xfe\x00"
    assert not deleted.exists()
    assert created.read_bytes() == b"new admitted file"
    assert not operation.exists()


def test_directory_at_restore_file_path_refuses_before_snapshot_mutation(tmp_path):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    obstruction = retained.with_name("retained.bin")
    retained.unlink()
    obstruction.mkdir()
    obstruction_bytes = obstruction / "unrelated.bin"
    obstruction_bytes.write_bytes(b"directory obstruction bytes")
    unrelated = store.workspace / "workspace-note.txt"
    unrelated.write_bytes(b"unrelated workspace bytes")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside bytes")
    fixed_before = {
        path: (store.workspace / path).read_bytes()
        for path in ("SOUL.md", "USER.md", "memory/MEMORY.md", "memory/.dream_cursor")
    }
    operation = store.workspace / ".git" / "nanobot-dream-operation.json"

    with pytest.raises(GitStoreError):
        store.git.revert(sha, message_prefix="dream:")

    assert {path: (store.workspace / path).read_bytes() for path in fixed_before} == fixed_before
    assert obstruction.is_dir()
    assert obstruction_bytes.read_bytes() == b"directory obstruction bytes"
    assert not deleted.exists()
    assert created.read_bytes() == b"new admitted file"
    assert unrelated.read_bytes() == b"unrelated workspace bytes"
    assert sentinel.read_bytes() == b"outside bytes"
    assert not operation.exists()


def test_unsupported_git_skill_symlink_refuses_before_restore(tmp_path):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"outside bytes")
    link = store.workspace / "skills" / "unsafe-link"
    link.symlink_to(sentinel)
    symlink_sha = store.git.auto_commit("dream: unsupported symlink")
    assert symlink_sha is not None
    before_memory = store.memory_file.read_bytes()
    before_soul = store.soul_file.read_bytes()

    with pytest.raises(GitStoreError):
        store.git.revert(symlink_sha, message_prefix="dream:")

    assert sentinel.read_bytes() == b"outside bytes"
    assert store.memory_file.read_bytes() == before_memory
    assert store.soul_file.read_bytes() == before_soul
    assert retained.read_bytes() == b"after\xfe\x00"
    assert not deleted.exists()
    assert created.read_bytes() == b"new admitted file"


def test_restore_revalidates_skill_parent_at_atomic_publication(tmp_path, monkeypatch):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"outside retained sentinel")
    parent = store.workspace / "skills" / "nested"
    moved = tmp_path / "nested-original"
    original = gitstore_module.atomic_publish_admitted_file
    attacked = False

    def replace_parent_then_publish(workspace, path, data, mode, *, allowed_current):
        nonlocal attacked
        if path == "skills/nested/retained.bin" and not attacked:
            attacked = True
            parent.rename(moved)
            parent.symlink_to(outside, target_is_directory=True)
        return original(workspace, path, data, mode, allowed_current=allowed_current)

    monkeypatch.setattr(gitstore_module, "atomic_publish_admitted_file", replace_parent_then_publish)
    with pytest.raises(GitStoreError):
        store.git.revert(sha, message_prefix="dream:")
    assert attacked
    assert sentinel.read_bytes() == b"outside retained sentinel"
    assert (store.workspace / ".git" / "nanobot-dream-operation.json").is_file()

    parent.unlink()
    moved.rename(parent)
    monkeypatch.setattr(gitstore_module, "atomic_publish_admitted_file", original)
    restored = store.git.revert(sha, message_prefix="dream:")
    assert restored is not None
    assert sentinel.read_bytes() == b"outside retained sentinel"
    _assert_dream_snapshot_before(store, retained, deleted, created)


def test_restore_requests_from_distinct_instances_serialize(tmp_path):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    other = MemoryStore(store.workspace, sessions_root=tmp_path / "sessions")
    barrier = Barrier(2)

    def restore(memory):
        barrier.wait(timeout=5)
        return memory.git.revert(sha, message_prefix="dream:")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(restore, memory) for memory in (store, other)]
        outcomes = [future.result(timeout=10) for future in futures]

    assert sum(value is not None for value in outcomes) == 1
    assert sum(value is None for value in outcomes) == 1
    _assert_dream_snapshot_before(store, retained, deleted, created)


def test_restore_hard_exit_reopens_and_resumes_without_cursor_skip(tmp_path):
    store, sha, retained, deleted, created = _dream_restore_fixture(tmp_path)
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path
        from nanobot.config.loader import set_config_path
        import nanobot.utils.gitstore as gitstore
        from nanobot.agent.memory import MemoryStore

        workspace, sessions, target, config = map(Path, sys.argv[1:])
        set_config_path(config)
        publish = gitstore.atomic_publish_admitted_file
        def exit_after_soul(root, path, data, mode, *, allowed_current):
            result = publish(root, path, data, mode, allowed_current=allowed_current)
            if path == "SOUL.md":
                os._exit(73)
            return result
        gitstore.atomic_publish_admitted_file = exit_after_soul
        MemoryStore(workspace, sessions_root=sessions).git.revert(
            target.name, message_prefix="dream:",
        )
        """
    )
    workspace = store.workspace
    config = tmp_path / "instance" / "config.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script, str(workspace), str(tmp_path / "sessions"), sha, str(config)],
        capture_output=True,
        check=False,
        env=env,
        timeout=20,
    )
    assert result.returncode == 73, result.stderr.decode(errors="replace")

    fresh = MemoryStore(workspace, sessions_root=tmp_path / "sessions")
    fresh.reconcile_dream_operation()
    _assert_dream_snapshot_before(fresh, retained, deleted, created)
    assert fresh.build_dream_prompt() is None


class TestMemoryStoreGitProperty:
    def test_git_property_exposes_gitstore(self, tmp_path):
        from nanobot.agent.memory import MemoryStore
        store = MemoryStore(tmp_path)
        assert isinstance(store.git, GitStore)

    def test_git_property_is_same_object(self, tmp_path):
        from nanobot.agent.memory import MemoryStore
        store = MemoryStore(tmp_path)
        assert store.git is store._git
