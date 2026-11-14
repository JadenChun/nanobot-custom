"""Git-backed version control for memory files, using dulwich."""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from loguru import logger

from nanobot.utils.dream_operation import (
    DreamFileChange,
    DreamFileState,
    DreamOperationRecord,
    DreamOperationStorageError,
    atomic_delete_admitted_file,
    atomic_publish_admitted_file,
    clear_operation,
    dream_workspace_lock,
    new_operation_id,
    read_admitted_file,
    read_operation,
    scan_admitted_worktree,
    state_for_bytes,
    validate_admitted_parent,
    workspace_identity,
    write_operation,
)

if TYPE_CHECKING:
    from dulwich.objects import Blob, Commit, ObjectID, Tree
    from dulwich.refs import Ref
    from dulwich.repo import Repo

# Cap on the unified-diff block embedded in Dream commit messages. Memory files
# are tiny in practice, but a pathological rewrite must not blow up the audit
# record. The structured per-file summary is always emitted in full regardless.
_WORKING_TREE_DIFF_MAX_CHARS = 6000
_DREAM_CURSOR_PATH = "memory/.dream_cursor"
_DREAM_FIXED_PATHS = frozenset(("SOUL.md", "USER.md", "memory/MEMORY.md", _DREAM_CURSOR_PATH))


class GitStoreError(RuntimeError):
    """Raised when the memory Git repository cannot complete an operation."""


@dataclass
class CommitInfo:
    sha: str  # Short SHA (8 chars)
    message: str
    timestamp: str  # Formatted datetime

    def subject(self) -> str:
        """First line of the commit message, or a placeholder if empty."""
        lines = self.message.splitlines()
        return lines[0] if lines else "(no message)"

    def format(self, diff: str = "") -> str:
        """Format this commit for display, optionally with a diff."""
        header = f"## {self.subject()}\n`{self.sha}` — {self.timestamp}\n"
        if diff:
            return f"{header}\n```diff\n{diff}\n```"
        return f"{header}\n(no file changes)"


class GitStore:
    """Git-backed version control for memory files."""

    def __init__(self, workspace: Path, tracked_files: list[str]):
        self._workspace = workspace
        self._tracked_files = tracked_files

    def is_initialized(self) -> bool:
        """Check if the git repo has been initialized."""
        return (self._workspace / ".git").is_dir()

    # -- init ------------------------------------------------------------------

    def init(self) -> bool:
        """Initialize a git repo if not already initialized.

        Creates .gitignore and makes an initial commit.
        Returns True if a new repo was created, False if already exists.
        """
        if self.is_initialized():
            return False

        if self._is_inside_git_repo():
            logger.warning(
                "Workspace {} is already inside a git repo; "
                "skipping nested repo initialization",
                self._workspace,
            )
            return False

        try:
            from dulwich import porcelain

            porcelain.init(str(self._workspace))

            # Write .gitignore (merge with existing if present)
            gitignore = self._workspace / ".gitignore"
            dream_entries = self._build_gitignore()
            if gitignore.exists():
                existing = gitignore.read_text(encoding="utf-8")
                existing_lines = set(existing.splitlines())
                new_lines = [
                    line
                    for line in dream_entries.splitlines()
                    if line not in existing_lines
                ]
                if new_lines:
                    merged = existing.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"
                    gitignore.write_text(merged, encoding="utf-8")
            else:
                gitignore.write_text(dream_entries, encoding="utf-8")

            # Ensure tracked files exist (touch them if missing) so the initial
            # commit has something to track.
            for rel in self._tracked_files:
                p = self._workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if rel == "skills":
                    # Directory-scoped tracking for workspace skills.
                    p.mkdir(exist_ok=True)
                elif not p.exists():
                    p.write_text("", encoding="utf-8")

            # Initial commit
            porcelain.add(
                str(self._workspace),
                paths=self._staging_paths(".gitignore", *self._tracked_files),
            )
            porcelain.commit(
                str(self._workspace),
                message=b"init: nanobot memory store",
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            logger.info("Git store initialized at {}", self._workspace)
            return True
        except Exception as exc:
            raise GitStoreError(f"Git store init failed for {self._workspace}") from exc

    # -- daily operations ------------------------------------------------------

    def auto_commit(self, message: str) -> str | None:
        """Stage tracked memory files and commit if there are changes.

        Returns the short commit SHA, or None if nothing to commit.
        """
        if not self.is_initialized():
            return None

        try:
            from dulwich import porcelain

            # Stage first so Dulwich refreshes the content hashes.  A status
            # check can miss rapid same-size rewrites when the filesystem also
            # preserves the file's mtime.
            porcelain.add(str(self._workspace), paths=self._staging_paths(*self._tracked_files))
            st = porcelain.status(str(self._workspace))
            staged = cast(dict[object, list[object]], st.staged)
            if not any(staged.values()):
                return None

            message_value = cast(object, message)
            msg_bytes = (
                message_value.encode("utf-8")
                if isinstance(message_value, str)
                else cast(bytes, message_value)
            )
            sha_bytes = porcelain.commit(
                str(self._workspace),
                message=msg_bytes,
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            if cast(object, sha_bytes) is None:
                return None
            # porcelain.commit returns the id as a 40-char hex string that is
            # already encoded to bytes; .hex() would encode those ASCII bytes
            # again and produce an id no git command can resolve.
            sha = sha_bytes.decode()[:8]
            logger.debug("Git auto-commit: {} ({})", sha, message)
            return sha
        except Exception as exc:
            raise GitStoreError(f"Git auto-commit failed: {message}") from exc

    # -- internal helpers ------------------------------------------------------

    def _staging_paths(self, *paths: str) -> list[str]:
        """Return absolute paths without resolving tracked-file symlinks."""
        return [str((self._workspace / path).absolute()) for path in paths]

    def _resolve_sha(self, short_sha: str) -> bytes | None:
        """Resolve a short SHA prefix to the full SHA bytes."""
        try:
            from dulwich.repo import Repo

            with Repo(str(self._workspace)) as repo:
                try:
                    sha: ObjectID | None = repo.refs[cast("Ref", b"HEAD")]
                except KeyError:
                    return None

                while sha:
                    if sha.decode().startswith(short_sha):
                        return sha
                    commit_obj = repo[sha]
                    if commit_obj.type_name != b"commit":
                        break
                    commit = cast("Commit", commit_obj)
                    sha = commit.parents[0] if commit.parents else None
            return None
        except Exception as exc:
            raise GitStoreError(f"Git SHA resolution failed: {short_sha}") from exc

    def _is_inside_git_repo(self) -> bool:
        """Check if self._workspace is already inside a git repository.

        Walks up from self._workspace to the filesystem root, returning True
        if any parent directory contains a .git entry.

        Git worktrees and submodules can use a ``.git`` file instead of a
        directory, so we must treat either form as "already inside a repo".
        """
        current = self._workspace.resolve()
        while current != current.parent:
            if (current / ".git").exists():
                return True
            current = current.parent
        return False

    def _build_gitignore(self) -> str:
        """Generate .gitignore content from tracked files."""
        dirs: set[str] = set()
        for f in self._tracked_files:
            parent = str(Path(f).parent)
            if parent != ".":
                dirs.add(parent)
        lines = ["/*"]
        for d in sorted(dirs):
            lines.append(f"!{d}/")
        for f in self._tracked_files:
            lines.append(f"!{f}")
        lines.append("!.gitignore")
        return "\n".join(lines) + "\n"

    # -- query -----------------------------------------------------------------

    def log(
        self,
        max_entries: int = 20,
        message_prefix: str | None = None,
    ) -> list[CommitInfo]:
        """Return simplified commit log, optionally filtered by message prefix.

        When filtering, *max_entries* counts matching commits rather than every
        commit traversed in the repository.
        """
        if not self.is_initialized():
            return []

        try:
            from dulwich.repo import Repo

            entries: list[CommitInfo] = []
            with Repo(str(self._workspace)) as repo:
                try:
                    head = repo.refs[cast("Ref", b"HEAD")]
                except KeyError:
                    return []

                sha: ObjectID | None = head
                while sha and len(entries) < max_entries:
                    commit_obj = repo[sha]
                    if commit_obj.type_name != b"commit":
                        break
                    commit = cast("Commit", commit_obj)
                    ts = time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(commit.commit_time),
                    )
                    msg = commit.message.decode("utf-8", errors="replace").strip()
                    if message_prefix is None or msg.startswith(message_prefix):
                        entries.append(CommitInfo(
                            sha=sha.decode()[:8],
                            message=msg,
                            timestamp=ts,
                        ))
                    sha = commit.parents[0] if commit.parents else None

            return entries
        except Exception as exc:
            raise GitStoreError("Git log failed") from exc

    def diff_commits(self, sha1: str, sha2: str) -> str:
        """Show diff between two commits."""
        if not self.is_initialized():
            return ""

        try:
            from dulwich import porcelain

            full1 = self._resolve_sha(sha1)
            full2 = self._resolve_sha(sha2)
            if not full1 or not full2:
                return ""

            out = io.BytesIO()
            porcelain.diff(
                str(self._workspace),
                commit=full1,
                commit2=full2,
                outstream=out,
            )
            return out.getvalue().decode("utf-8", errors="replace")
        except Exception as exc:
            raise GitStoreError(f"Git diff failed for {sha1}..{sha2}") from exc

    def summarize_working_tree(self, paths: list[str]) -> str:
        """Structured summary of working-tree changes vs HEAD for *paths*.

        Pure filesystem/git ground truth — never LLM narrative — suitable as a
        truthful audit record. Returns "" when the repo is not initialized or
        none of *paths* differ from HEAD.

        Format::

            SOUL.md: +3 -1
            memory/MEMORY.md: +12 -8

            2 files changed, 15 insertions(+), 9 deletions(-)

            ```diff
            --- SOUL.md
            +++ SOUL.md
            @@ ...
            - old
            + new
            ```
        """
        if not self.is_initialized():
            return ""

        try:
            import difflib

            from dulwich.repo import Repo
        except ImportError as exc:
            raise GitStoreError("Git working-tree summary dependencies are unavailable") from exc

        summary_lines: list[str] = []
        diff_lines: list[str] = []
        total_added = 0
        total_removed = 0
        changed = 0

        try:
            with Repo(str(self._workspace)) as repo:
                head_tree = self._head_tree(repo)
                expanded: set[str] = set()
                for raw_path in paths:
                    target = self._workspace / raw_path
                    if target.is_dir():
                        expanded.update(
                            item.relative_to(self._workspace).as_posix()
                            for item in target.rglob("*")
                            if item.is_file()
                        )
                        if head_tree is not None:
                            prefix = raw_path.rstrip("/")
                            try:
                                from dulwich.object_store import (
                                    iter_tree_contents,
                                )

                                for entry in iter_tree_contents(
                                    repo.object_store, head_tree.id
                                ):
                                    entry_path = entry.path.decode("utf-8")
                                    if (
                                        entry_path == prefix
                                        or entry_path.startswith(prefix + "/")
                                    ):
                                        expanded.add(entry_path)
                            except Exception:
                                pass
                    else:
                        expanded.add(raw_path)
                for path in sorted(expanded):
                    head_text = (
                        self._read_blob_from_tree(repo, head_tree, path)
                        if head_tree is not None
                        else None
                    )
                    if head_text is None:
                        head_text = ""
                    wt_path = self._workspace / path
                    try:
                        wt_text = (
                            wt_path.read_bytes().decode("utf-8")
                            if wt_path.exists()
                            else ""
                        )
                    except UnicodeDecodeError:
                        # Non-UTF-8 (binary/corrupt) working-tree file: record
                        # the change without a unified diff, which would
                        # otherwise be polluted with replacement characters and
                        # misrepresent the audit record.
                        changed += 1
                        summary_lines.append(f"{path}: binary or non-UTF-8 file changed")
                        continue
                    # Treat CRLF and LF as equivalent without hiding other
                    # newline changes, such as a missing final newline.
                    if head_text.replace("\r\n", "\n") == wt_text.replace("\r\n", "\n"):
                        continue
                    head_lines = head_text.splitlines()
                    wt_lines = wt_text.splitlines()
                    changed += 1
                    hunks = list(difflib.unified_diff(
                        head_lines,
                        wt_lines,
                        fromfile=path,
                        tofile=path,
                        lineterm="",
                    ))
                    added = sum(1 for line in hunks if line.startswith("+") and not line.startswith("+++"))
                    removed = sum(1 for line in hunks if line.startswith("-") and not line.startswith("---"))
                    total_added += added
                    total_removed += removed
                    summary_lines.append(f"{path}: +{added} -{removed}")
                    diff_lines.extend(hunks)
        except Exception as exc:
            raise GitStoreError("Git working-tree summary failed") from exc

        if changed == 0:
            return ""

        diff_text = "\n".join(diff_lines)
        if len(diff_text) > _WORKING_TREE_DIFF_MAX_CHARS:
            diff_text = diff_text[:_WORKING_TREE_DIFF_MAX_CHARS] + "\n...[diff truncated]"

        body = "\n".join(summary_lines)
        body += (
            f"\n{changed} file{'s' if changed != 1 else ''} changed, "
            f"{total_added} insertion{'s' if total_added != 1 else ''}(+), "
            f"{total_removed} deletion{'s' if total_removed != 1 else ''}(-)"
        )
        if diff_lines:
            body += f"\n\n```diff\n{diff_text}\n```"
        return body

    @staticmethod
    def _head_tree(repo: "Repo") -> "Tree | None":
        """Return the tree object at HEAD, or None if there are no commits."""
        try:
            head = repo.refs[cast("Ref", b"HEAD")]
        except KeyError:
            return None
        commit_obj = repo[head]
        if commit_obj.type_name != b"commit":
            return None
        commit = cast("Commit", commit_obj)
        return cast("Tree", repo[commit.tree])

    def show_commit_diff(
        self,
        short_sha: str,
        max_entries: int = 20,
        message_prefix: str | None = None,
    ) -> tuple[CommitInfo, str] | None:
        """Find a commit and return it with its diff vs its actual parent."""
        try:
            from dulwich.repo import Repo

            commits = self.log(max_entries=max_entries, message_prefix=message_prefix)
            for c in commits:
                if c.sha.startswith(short_sha):
                    full_sha = self._resolve_sha(c.sha)
                    if not full_sha:
                        return None
                    with Repo(str(self._workspace)) as repo:
                        commit = cast("Commit", repo[full_sha])
                        parent = commit.parents[0] if commit.parents else None
                    diff = self.diff_commits(parent.decode()[:8], c.sha) if parent else ""
                    return c, diff
            return None
        except Exception as exc:
            raise GitStoreError(f"Git commit display failed for {short_sha}") from exc

    # -- restore ---------------------------------------------------------------

    def revert(self, commit: str, *, message_prefix: str | None = None) -> str | None:
        """Restore the complete admitted snapshot before *commit* with recovery proof."""

        if not self.is_initialized():
            return None
        try:
            with dream_workspace_lock(self._workspace):
                target = self._restore_target(commit, message_prefix)
                if target is None:
                    return None
                full_sha, target_tree_id, target_files, starting_head, start_tree = target
                pending = read_operation(self._workspace)
                if pending is not None:
                    if pending.kind == "restore":
                        if pending.target_commit != full_sha.decode():
                            raise GitStoreError("A different Dream restore is still pending")
                        return self._resume_restore(pending)
                    if pending.phase != "running":
                        raise GitStoreError("Dream finalization is still pending")
                    if pending.starting_head != starting_head.decode():
                        raise GitStoreError("Dream workspace changed while an operation is pending")
                elif self._worktree_matches_target(start_tree, target_files):
                    return None

                record = self._new_restore_record(
                    full_sha,
                    target_tree_id,
                    target_files,
                    starting_head,
                    start_tree,
                )
                write_operation(self._workspace, record)
                return self._resume_restore(record)
        except GitStoreError:
            raise
        except Exception as exc:
            raise GitStoreError(f"Git revert failed for {commit}") from exc

    def _worktree_matches_target(
        self,
        head_tree: "Tree",
        target_files: dict[str, tuple[bytes, DreamFileState]],
    ) -> bool:
        from dulwich.repo import Repo

        with Repo(str(self._workspace)) as repo:
            head_files = self._tree_admitted_files(repo, head_tree)
        working = scan_admitted_worktree(self._workspace)
        paths = set(_DREAM_FIXED_PATHS) | set(head_files) | set(target_files) | set(working)
        if set(working) | set(_DREAM_FIXED_PATHS) != set(target_files) | set(_DREAM_FIXED_PATHS):
            return False
        for path in paths:
            current = working.get(path)
            target = target_files.get(path)
            if (current[1] if current else None) != (target[1] if target else None):
                return False
        return True

    def dream_operation_status(self) -> DreamOperationRecord | None:
        """Read the strictly validated private Dream operation record."""

        if not self.is_initialized():
            return None
        try:
            with dream_workspace_lock(self._workspace):
                return read_operation(self._workspace)
        except Exception as exc:
            raise GitStoreError("Dream operation evidence is unavailable") from exc

    def reconcile_pending_dream_operation(self) -> str | None:
        """Resume a proven restore; run finalization stays owned by MemoryStore."""

        if not self.is_initialized():
            return None
        try:
            with dream_workspace_lock(self._workspace):
                record = read_operation(self._workspace)
                if record is None:
                    return None
                if record.kind != "restore":
                    raise GitStoreError("A Dream run is pending reconciliation")
                return self._resume_restore(record)
        except GitStoreError:
            raise
        except Exception as exc:
            raise GitStoreError("Dream operation could not be reconciled") from exc

    def reserve_dream_run(self, cursor_before: int, cursor_after: int) -> str:
        """Persist one run owner before Dream receives file tools or dispatches."""

        if not self.is_initialized():
            raise GitStoreError("Dream requires initialized memory versioning")
        if type(cursor_before) is not int or type(cursor_after) is not int or cursor_after <= cursor_before:
            raise GitStoreError("Dream cursor batch is invalid")
        try:
            with dream_workspace_lock(self._workspace):
                if read_operation(self._workspace) is not None:
                    raise GitStoreError("Another Dream operation is pending")
                from dulwich.repo import Repo

                with Repo(str(self._workspace)) as repo:
                    head, tree = self._head_and_tree(repo)
                    tree_files = self._tree_admitted_files(repo, tree)
                    working = scan_admitted_worktree(self._workspace)
                    paths = set(_DREAM_FIXED_PATHS) | set(tree_files) | set(working)
                    for path in sorted(paths):
                        validate_admitted_parent(self._workspace, path)
                    current_cursor = self._cursor_from_files(working)
                    if current_cursor != cursor_before:
                        raise GitStoreError("Dream cursor changed while the batch was prepared")
                    changes = tuple(
                        DreamFileChange(
                            path,
                            working[path][1] if path in working else None,
                            working[path][1] if path in working else None,
                        )
                        for path in sorted(paths)
                    )
                    workspace_path, device, inode = workspace_identity(self._workspace)
                    operation_id = new_operation_id()
                    record = DreamOperationRecord(
                        operation_id=operation_id,
                        kind="run",
                        phase="running",
                        workspace_path=workspace_path,
                        workspace_device=device,
                        workspace_inode=inode,
                        starting_head=head.decode(),
                        target_commit=None,
                        target_tree=tree.id.decode(),
                        cursor_before=cursor_before,
                        cursor_after=cursor_after,
                        completion=None,
                        commit_message_sha256=None,
                        changes=changes,
                    )
                    write_operation(self._workspace, record)
                    return operation_id
        except GitStoreError:
            raise
        except Exception as exc:
            raise GitStoreError("Dream run ownership could not be reserved") from exc

    def dream_run_is_active(self, operation_id: str) -> bool:
        """Check one run token while the caller holds the workspace lock."""

        if not self.is_initialized():
            return False
        try:
            record = read_operation(self._workspace)
            return bool(
                record is not None
                and record.operation_id == operation_id
                and record.kind == "run"
                and record.phase == "running"
            )
        except DreamOperationStorageError:
            return False

    def settle_dream_run(
        self,
        operation_id: str,
        *,
        completed: bool,
        message: str,
    ) -> str | None:
        """Record the observed run result, then prove bytes, cursor and commit."""

        if type(completed) is not bool or not message:
            raise GitStoreError("Dream run settlement input is invalid")
        try:
            with dream_workspace_lock(self._workspace):
                record = read_operation(self._workspace)
                if (
                    record is None
                    or record.operation_id != operation_id
                    or record.kind != "run"
                ):
                    raise GitStoreError("Dream run ownership is no longer current")
                message_digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
                if record.phase == "running":
                    record = self._prepare_run_settlement(record, completed, message_digest)
                    write_operation(self._workspace, record)
                elif (
                    record.phase != "settling"
                    or record.completion != ("completed" if completed else "incomplete")
                ):
                    raise GitStoreError("Dream run settlement conflicts with its durable owner")
                return self._continue_run_settlement(record, message)
        except GitStoreError:
            raise
        except Exception as exc:
            raise GitStoreError("Dream run settlement could not be proved") from exc

    def _restore_target(
        self,
        commit: str,
        message_prefix: str | None,
    ) -> tuple[bytes, bytes, dict[str, tuple[bytes, DreamFileState]], bytes, "Tree"] | None:
        from dulwich.repo import Repo

        full_sha = self._resolve_sha(commit)
        if not full_sha:
            logger.warning("Git revert: SHA not found: {}", commit)
            return None
        with Repo(str(self._workspace)) as repo:
            commit_obj = repo[full_sha]
            if commit_obj.type_name != b"commit":
                return None
            typed_commit = cast("Commit", commit_obj)
            commit_message = typed_commit.message.decode("utf-8", errors="replace").strip()
            if message_prefix is not None and not commit_message.startswith(message_prefix):
                logger.warning("Git revert: target {} did not match the required history", commit)
                return None
            if not typed_commit.parents:
                logger.warning("Git revert: cannot restore root commit {}", commit)
                return None
            head, _head_tree = self._head_and_tree(repo)
            parent_obj = cast("Commit", repo[typed_commit.parents[0]])
            target_tree = cast("Tree", repo[parent_obj.tree])
            target_files = self._tree_admitted_files(repo, target_tree)
            # Read the starting tree again from the verified HEAD for the same repository handle.
            head_obj = cast("Commit", repo[head])
            start_tree = cast("Tree", repo[head_obj.tree])
            return full_sha, parent_obj.tree, target_files, head, start_tree

    def _new_restore_record(
        self,
        target_commit: bytes,
        target_tree_id: bytes,
        target_files: dict[str, tuple[bytes, DreamFileState]],
        starting_head: bytes,
        start_tree: "Tree",
    ) -> DreamOperationRecord:
        from dulwich.repo import Repo

        with Repo(str(self._workspace)) as repo:
            current_tree_files = self._tree_admitted_files(repo, start_tree)
        working = scan_admitted_worktree(self._workspace)
        paths = set(_DREAM_FIXED_PATHS) | set(target_files) | set(current_tree_files) | set(working)
        if len(paths) > 10_000:
            raise GitStoreError("Dream restore snapshot exceeds its path limit")
        changes: list[DreamFileChange] = []
        for path in sorted(paths):
            validate_admitted_parent(self._workspace, path)
            read_admitted_file(self._workspace, path)
            current = working[path][1] if path in working else None
            target = target_files[path][1] if path in target_files else None
            changes.append(DreamFileChange(path, current, target))
        cursor_before = self._cursor_from_files(working)
        cursor_after = self._cursor_from_files(target_files)
        workspace_path, device, inode = workspace_identity(self._workspace)
        return DreamOperationRecord(
            operation_id=new_operation_id(),
            kind="restore",
            phase="restoring",
            workspace_path=workspace_path,
            workspace_device=device,
            workspace_inode=inode,
            starting_head=starting_head.decode(),
            target_commit=target_commit.decode(),
            target_tree=target_tree_id.decode(),
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            completion=None,
            commit_message_sha256=None,
            changes=tuple(changes),
        )

    def _resume_restore(self, record: DreamOperationRecord) -> str:
        from dulwich.repo import Repo

        assert record.kind == "restore" and record.target_commit is not None
        with Repo(str(self._workspace)) as repo:
            head, _head_tree = self._head_and_tree(repo)
            if head.decode() != record.starting_head:
                published = self._matching_operation_commit(repo, record, "restore")
                if published is None:
                    raise GitStoreError("Dream restore HEAD changed without matching settlement proof")
                return self._finish_operation_record(record, published)

            target_obj = repo[record.target_commit.encode("ascii")]
            if target_obj.type_name != b"commit":
                raise GitStoreError("Dream restore target proof disappeared")
            target_commit = cast("Commit", target_obj)
            if not target_commit.parents:
                raise GitStoreError("Dream restore target proof has no parent")
            parent = cast("Commit", repo[target_commit.parents[0]])
            if parent.tree.decode() != record.target_tree:
                raise GitStoreError("Dream restore target tree conflicts with its owner")
            target_tree = cast("Tree", repo[parent.tree])
            target_files = self._tree_admitted_files(repo, target_tree)
            self._verify_record_path_set(record, repo, target_files)
            self._verify_record_target_files(record, target_files)
            working = scan_admitted_worktree(self._workspace)
            self._verify_current_states(record, working, allow_before=True)

            for change in record.changes:
                target = target_files.get(change.path)
                allowed = (change.before, change.target)
                if target is None:
                    atomic_delete_admitted_file(
                        self._workspace,
                        change.path,
                        allowed_current=allowed,
                    )
                else:
                    atomic_publish_admitted_file(
                        self._workspace,
                        change.path,
                        target[0],
                        target[1].mode,
                        allowed_current=allowed,
                    )
            readback = scan_admitted_worktree(self._workspace)
            self._verify_current_states(record, readback, allow_before=False)
            if self._cursor_from_files(readback) != record.cursor_after:
                raise GitStoreError("Dream restore cursor readback failed")

        message = f"revert: undo {record.target_commit[:8]}"
        try:
            self.auto_commit(message)
        except GitStoreError:
            # Dulwich may publish the commit and then fail to return its id.
            pass
        with Repo(str(self._workspace)) as repo:
            published = self._matching_operation_commit(repo, record, "restore")
            if published is None:
                raise GitStoreError("Dream restore commit is not fully proved")
        return self._finish_operation_record(record, published)

    def _finish_operation_record(self, record: DreamOperationRecord, commit: bytes) -> str:
        settled = dataclass_replace(record, settlement_sha=commit.decode())
        write_operation(self._workspace, settled)
        from dulwich.repo import Repo

        with Repo(str(self._workspace)) as repo:
            if self._matching_operation_commit(repo, settled, settled.kind) != commit:
                raise GitStoreError("Dream operation commit readback changed")
        clear_operation(self._workspace)
        return commit.decode()[:8]

    def _prepare_run_settlement(
        self,
        record: DreamOperationRecord,
        completed: bool,
        message_digest: str,
    ) -> DreamOperationRecord:
        from dulwich.repo import Repo

        if record.phase != "running" or record.starting_head == "":
            raise GitStoreError("Dream run is not in a settleable phase")
        with Repo(str(self._workspace)) as repo:
            head, start_tree = self._head_and_tree(repo)
            if head.decode() != record.starting_head or start_tree.id.decode() != record.target_tree:
                raise GitStoreError("Dream workspace changed during Dream execution")
            start_files = self._tree_admitted_files(repo, start_tree)
        working = scan_admitted_worktree(self._workspace)
        baseline = {change.path: change.before for change in record.changes}
        paths = set(baseline) | set(start_files) | set(working) | set(_DREAM_FIXED_PATHS)
        expected_cursor = record.cursor_after if completed else record.cursor_before
        current_cursor = self._cursor_from_files(working)
        if current_cursor != record.cursor_before:
            raise GitStoreError("Dream cursor changed before settlement")
        cursor_mode = (
            working[_DREAM_CURSOR_PATH][1].mode
            if _DREAM_CURSOR_PATH in working
            else 0o644
        )
        cursor_state = state_for_bytes(str(expected_cursor).encode("ascii"), cursor_mode)
        changes = tuple(
            DreamFileChange(
                path,
                baseline.get(path),
                cursor_state if path == _DREAM_CURSOR_PATH else (
                    working[path][1] if path in working else None
                ),
            )
            for path in sorted(paths)
        )
        if len(changes) > 10_000:
            raise GitStoreError("Dream run evidence exceeds its path limit")
        return dataclass_replace(
            record,
            phase="settling",
            cursor_after=expected_cursor,
            completion="completed" if completed else "incomplete",
            commit_message_sha256=message_digest,
            changes=changes,
            settlement_sha=None,
        )

    def _continue_run_settlement(self, record: DreamOperationRecord, message: str) -> str | None:
        from dulwich.repo import Repo

        with Repo(str(self._workspace)) as repo:
            head, _tree = self._head_and_tree(repo)
            if head.decode() != record.starting_head:
                published = self._matching_operation_commit(repo, record, "run")
                if published is None:
                    raise GitStoreError("Dream finalization HEAD changed without proof")
                return self._finish_operation_record(record, published)

        if hashlib.sha256(message.encode("utf-8")).hexdigest() != record.commit_message_sha256:
            raise GitStoreError("Dream finalization message does not match its owner")

        working = scan_admitted_worktree(self._workspace)
        self._verify_current_states(record, working, allow_before=False, cursor_before=True)
        cursor_item = working.get(_DREAM_CURSOR_PATH)
        current_cursor_state = cursor_item[1] if cursor_item is not None else None
        expected_cursor_state = next(
            (change.target for change in record.changes if change.path == _DREAM_CURSOR_PATH),
            None,
        )
        if expected_cursor_state is None:
            raise GitStoreError("Dream finalization cursor proof is missing")
        if current_cursor_state != expected_cursor_state:
            cursor_bytes = str(record.cursor_after).encode("ascii")
            atomic_publish_admitted_file(
                self._workspace,
                _DREAM_CURSOR_PATH,
                cursor_bytes,
                expected_cursor_state.mode,
                allowed_current=(current_cursor_state, expected_cursor_state),
            )
        try:
            self.auto_commit(message)
        except GitStoreError:
            # Read back below to distinguish a failed commit from a lost acknowledgement.
            pass
        with Repo(str(self._workspace)) as repo:
            head, _tree = self._head_and_tree(repo)
            if head.decode() == record.starting_head:
                if self._workspace_matches_start_tree(repo, record):
                    clear_operation(self._workspace)
                    return None
                raise GitStoreError("Dream finalization commit was not published")
            published = self._matching_operation_commit(repo, record, "run")
            if published is None:
                raise GitStoreError("Dream finalization commit is not fully proved")
        return self._finish_operation_record(record, published)

    def _workspace_matches_start_tree(self, repo: "Repo", record: DreamOperationRecord) -> bool:
        head, tree = self._head_and_tree(repo)
        if head.decode() != record.starting_head:
            return False
        tree_files = self._tree_admitted_files(repo, tree)
        by_path = {change.path: change.target for change in record.changes}
        for path in set(_DREAM_FIXED_PATHS) | set(tree_files) | set(by_path):
            expected = by_path.get(path)
            actual = tree_files.get(path)
            state = actual[1] if actual is not None else None
            if state != expected:
                return False
        return True

    def _matching_operation_commit(
        self,
        repo: "Repo",
        record: DreamOperationRecord,
        kind: Literal["restore", "run"],
    ) -> bytes | None:
        try:
            head, actual_tree = self._head_and_tree(repo)
            commit = cast("Commit", repo[head])
            if not commit.parents or commit.parents[0].decode() != record.starting_head:
                return None
            if kind == "restore":
                expected_message = f"revert: undo {cast(str, record.target_commit)[:8]}".encode()
                if commit.message.strip() != expected_message:
                    return None
            elif (
                record.commit_message_sha256 is None
                or hashlib.sha256(commit.message).hexdigest() != record.commit_message_sha256
            ):
                return None
            parent = cast("Commit", repo[commit.parents[0]])
            start_tree = cast("Tree", repo[parent.tree])
            if not self._tree_matches_operation(repo, start_tree, actual_tree, record):
                return None
            if record.settlement_sha is not None and head.decode() != record.settlement_sha:
                return None
            return head
        except Exception:
            return None

    def _tree_matches_operation(
        self,
        repo: "Repo",
        start_tree: "Tree",
        actual_tree: "Tree",
        record: DreamOperationRecord,
    ) -> bool:
        start_entries = self._flatten_tree(repo, start_tree)
        actual_entries = self._flatten_tree(repo, actual_tree)
        for path in set(start_entries) | set(actual_entries):
            if self._git_path_is_admitted(path):
                continue
            if start_entries.get(path) != actual_entries.get(path):
                return False
        actual_files = self._tree_admitted_files(repo, actual_tree)
        expected = {change.path: change.target for change in record.changes}
        paths = set(_DREAM_FIXED_PATHS) | set(actual_files) | set(expected)
        for path in paths:
            actual = actual_files.get(path)
            actual_state = actual[1] if actual is not None else None
            if actual_state != expected.get(path):
                return False
        return True

    def _verify_record_path_set(
        self,
        record: DreamOperationRecord,
        repo: "Repo",
        target_files: dict[str, tuple[bytes, DreamFileState]],
    ) -> None:
        target_obj = repo[cast(str, record.target_commit).encode("ascii")]
        target_commit = cast("Commit", target_obj)
        parent = cast("Commit", repo[target_commit.parents[0]])
        start_obj = repo[record.starting_head.encode("ascii")]
        start_commit = cast("Commit", start_obj)
        start_files = self._tree_admitted_files(repo, cast("Tree", repo[start_commit.tree]))
        paths = set(_DREAM_FIXED_PATHS) | set(target_files) | set(start_files)
        known = {change.path for change in record.changes}
        if not paths.issubset(known):
            raise GitStoreError("Dream restore path evidence changed")
        if parent.tree.decode() != record.target_tree:
            raise GitStoreError("Dream restore target tree changed")

    @staticmethod
    def _verify_record_target_files(
        record: DreamOperationRecord,
        target_files: dict[str, tuple[bytes, DreamFileState]],
    ) -> None:
        expected = {change.path: change.target for change in record.changes}
        paths = set(expected) | set(target_files)
        if not set(target_files).issubset(expected):
            raise GitStoreError("Dream restore target path set changed")
        for path in paths:
            target = target_files.get(path)
            state = target[1] if target is not None else None
            if state != expected.get(path):
                raise GitStoreError("Dream restore target bytes changed")

    def _verify_current_states(
        self,
        record: DreamOperationRecord,
        working: dict[str, tuple[bytes, DreamFileState]],
        *,
        allow_before: bool,
        cursor_before: bool = False,
    ) -> None:
        known = {change.path: change for change in record.changes}
        if not set(working).issubset(known):
            raise GitStoreError("Dream admitted path set changed")
        for path, change in known.items():
            actual = working[path][1] if path in working else None
            allowed = (change.target, change.before) if allow_before else (change.target,)
            if cursor_before and path == _DREAM_CURSOR_PATH:
                allowed = (change.target, change.before)
            if actual not in allowed:
                raise GitStoreError("Dream admitted file no longer matches its operation proof")

    def _head_and_tree(self, repo: "Repo") -> tuple[bytes, "Tree"]:
        try:
            head = repo.refs[cast("Ref", b"HEAD")]
        except KeyError as exc:
            raise GitStoreError("Dream memory repository has no HEAD") from exc
        commit_obj = repo[head]
        if commit_obj.type_name != b"commit":
            raise GitStoreError("Dream memory HEAD is not a commit")
        commit = cast("Commit", commit_obj)
        return head, cast("Tree", repo[commit.tree])

    def _tree_admitted_files(
        self,
        repo: "Repo",
        root: "Tree",
    ) -> dict[str, tuple[bytes, DreamFileState]]:
        result: dict[str, tuple[bytes, DreamFileState]] = {}

        def visit(tree: "Tree", prefix: bytes) -> None:
            for name, mode, sha in tree.iteritems():
                raw_path = prefix + b"/" + name if prefix else name
                try:
                    path = raw_path.decode("utf-8")
                except UnicodeDecodeError as exc:
                    if _git_path_bytes_are_admitted(raw_path):
                        raise GitStoreError("Dream tree contains an invalid admitted path") from exc
                    continue
                if path in ("skills", "memory"):
                    if mode != 0o040000:
                        raise GitStoreError("Dream tree has an unsupported admitted parent")
                    visit(cast("Tree", repo[sha]), raw_path)
                    continue
                if path.startswith("skills/"):
                    if mode == 0o040000:
                        visit(cast("Tree", repo[sha]), raw_path)
                        continue
                    self._require_regular_blob(repo, mode, sha)
                    blob = cast("Blob", repo[sha])
                    data = blob.data
                    result[path] = (data, state_for_bytes(data, mode & 0o777))
                    continue
                if path in _DREAM_FIXED_PATHS:
                    self._require_regular_blob(repo, mode, sha)
                    blob = cast("Blob", repo[sha])
                    data = blob.data
                    result[path] = (data, state_for_bytes(data, mode & 0o777))

        visit(root, b"")
        return result

    @staticmethod
    def _require_regular_blob(repo: "Repo", mode: int, sha: bytes) -> None:
        if mode not in (0o100644, 0o100755):
            raise GitStoreError("Dream tree contains an unsupported admitted entry")
        obj = repo[sha]
        if obj.type_name != b"blob":
            raise GitStoreError("Dream tree is missing an admitted file blob")

    @staticmethod
    def _flatten_tree(repo: "Repo", tree: "Tree") -> dict[bytes, tuple[int, bytes]]:
        result: dict[bytes, tuple[int, bytes]] = {}

        def visit(current: "Tree", prefix: bytes) -> None:
            for name, mode, sha in current.iteritems():
                path = prefix + b"/" + name if prefix else name
                if mode == 0o040000:
                    visit(cast("Tree", repo[sha]), path)
                else:
                    result[path] = mode, sha

        visit(tree, b"")
        return result

    @staticmethod
    def _git_path_is_admitted(path: bytes) -> bool:
        return _git_path_bytes_are_admitted(path)

    @staticmethod
    def _cursor_from_files(
        files: dict[str, tuple[bytes, DreamFileState]],
    ) -> int:
        item = files.get(_DREAM_CURSOR_PATH)
        if item is None:
            return 0
        raw = item[0].strip()
        if not raw:
            return 0
        if any(value < 48 or value > 57 for value in raw):
            raise GitStoreError("Dream cursor evidence is invalid")
        try:
            return int(raw.decode("ascii"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise GitStoreError("Dream cursor evidence is invalid") from exc

    @staticmethod
    def _read_blob_from_tree(
        repo: "Repo",
        tree: "Tree",
        filepath: str,
    ) -> str | None:
        """Read a blob's content from a tree object by walking path parts."""
        parts = Path(filepath).parts
        current = tree
        for part in parts:
            try:
                entry = current[part.encode()]
            except KeyError:
                return None
            obj = repo[entry[1]]
            if obj.type_name == b"blob":
                blob = cast("Blob", obj)
                return blob.data.decode("utf-8", errors="replace")
            if obj.type_name == b"tree":
                current = cast("Tree", obj)
            else:
                return None
        return None


def _git_path_bytes_are_admitted(path: bytes) -> bool:
    fixed = {value.encode("ascii") for value in _DREAM_FIXED_PATHS}
    return path in fixed or path == b"skills" or path.startswith(b"skills/")
