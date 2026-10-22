"""Git sync utility for context repositories."""

from __future__ import annotations

import asyncio
import os
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import TypedDict

from filelock import FileLock, Timeout
from loguru import logger

from nanobot.utils.secret_scan import find_secrets_in_blob


def _run_git(repo: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run a git command in the given repo directory."""
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=10")
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        errors="surrogateescape",
        timeout=timeout,
        env=env,
    )


def _git_output(result: subprocess.CompletedProcess[str]) -> str:
    """Return compact stdout/stderr text for logs."""
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository."""
    try:
        result = _run_git(path, "rev-parse", "--is-inside-work-tree")
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def git_repo_root(path: Path) -> Path | None:
    """Resolve an existing working tree without cloning or running a shell."""
    result = _run_git(path, "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve() if result.returncode == 0 else None


def has_unmerged_paths(repo: Path) -> bool:
    """Check if the repo has unresolved merge/rebase conflicts."""
    try:
        result = _run_git(repo, "diff", "--name-only", "--diff-filter=U")
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True


def has_changes(repo: Path) -> bool:
    """Check if the git repo has uncommitted changes."""
    try:
        result = _run_git(repo, "status", "--porcelain", "--untracked-files=all")
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _changed_paths(repo: Path) -> list[str]:
    """Return changed paths relative to repo root, including untracked files."""
    try:
        result = _run_git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if result.returncode != 0:
        return []
    paths: list[str] = []
    entries = iter(result.stdout.split("\0"))
    for entry in entries:
        if len(entry) < 4:
            continue
        paths.append(entry[3:])
        if "R" in entry[:2] or "C" in entry[:2]:
            source = next(entries, "")
            if source:
                paths.append(source)
    return paths


def matches_sync_path(path: str, patterns: list[str]) -> bool:
    clean = path.strip("/")
    for pattern in patterns:
        pat = pattern.strip().strip("/")
        if not pat:
            continue
        if pat.endswith("/**"):
            base = pat[:-3].strip("/")
            if clean == base or clean.startswith(base + "/"):
                return True
        if fnmatch(clean, pat):
            return True
    return False


def _select_changed_paths(
    repo: Path,
    include_paths: list[str] | None,
    exclude_paths: list[str] | None,
) -> list[str]:
    changed = _changed_paths(repo)
    if include_paths is None and not exclude_paths:
        return changed
    selected: list[str] = []
    for path in changed:
        if include_paths is not None and not matches_sync_path(path, include_paths):
            continue
        if exclude_paths and matches_sync_path(path, exclude_paths):
            continue
        selected.append(path)
    return selected


def _upstream_ref(repo: Path) -> str | None:
    result = _run_git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    if result.returncode != 0:
        logger.error("Context repo upstream lookup failed: {}", _git_output(result))
        return None
    ref = result.stdout.strip()
    return ref or None


def _commit_selected_paths(repo: Path, selected_paths: list[str], message: str) -> bool:
    if not selected_paths:
        return True
    staged = _run_git(repo, "diff", "--cached", "--name-only", "-z")
    if staged.returncode != 0 or set(staged.stdout.split("\0")) & set(selected_paths):
        logger.error("Selected knowledge paths already have staged work; refusing to replace it")
        return False

    add = _run_git(repo, "add", "-A", "--", *selected_paths)
    if add.returncode != 0:
        logger.error("Context repo git add failed: {}", _git_output(add))
        return False

    commit = _run_git(repo, "commit", "--only", "-m", message, "--", *selected_paths)
    if commit.returncode != 0:
        output = _git_output(commit)
        if "nothing to commit" in output:
            return True
        logger.error("Context repo git commit failed: {}", output)
        return False
    return True


def sync_with_remote(repo: Path) -> bool:
    """Integrate upstream without stashing, rebasing, or replacing local history."""
    if has_unmerged_paths(repo):
        logger.error("Context repo has unresolved git conflicts; refusing to sync: {}", repo)
        return False

    fetch = _run_git(repo, "fetch", "--prune", timeout=60)
    if fetch.returncode != 0:
        logger.error("Context repo git fetch failed: {}", _git_output(fetch))
        return False

    upstream = _upstream_ref(repo)
    if not upstream:
        return False
    if _run_git(repo, "merge-base", "--is-ancestor", upstream, "HEAD").returncode == 0:
        return True
    fast_forward = _run_git(repo, "merge-base", "--is-ancestor", "HEAD", upstream).returncode == 0
    if not fast_forward and has_changes(repo):
        logger.error("Context repo has local edits and diverging history; refusing automatic merge")
        return False
    merge = _run_git(repo, "merge", "--ff-only" if fast_forward else "--no-edit", upstream, timeout=60)
    if merge.returncode == 0:
        return True
    if has_unmerged_paths(repo):
        abort = _run_git(repo, "merge", "--abort")
        if abort.returncode != 0:
            logger.error("Context repo merge abort failed; manual reconciliation required")
    logger.error("Context repo merge refused; local and upstream history retained")
    return False


class _AheadCommit(TypedDict):
    sha: str
    short: str
    paths: list[str]


def _ahead_commit_summaries(repo: Path, upstream: str) -> list[_AheadCommit]:
    """Inspect exact paths, including merge resolutions and unusual filenames."""
    result = _run_git(repo, "rev-list", f"{upstream}..HEAD", timeout=60)
    if result.returncode != 0:
        raise OSError("could not inspect ahead commits")
    summaries: list[_AheadCommit] = []
    for sha in result.stdout.splitlines():
        changed = _run_git(repo, "diff-tree", "--root", "-m", "--no-commit-id", "--name-only", "-r", "-z", sha)
        if changed.returncode != 0:
            raise OSError("could not inspect ahead commit paths")
        summaries.append({"sha": sha, "short": sha[:7],
                          "paths": list(dict.fromkeys(path for path in changed.stdout.split("\0") if path))})
    return summaries


def _commit_blob(repo: Path, sha: str, path: str) -> str | None:
    """Return the blob text for one path in a given commit, or None if absent."""
    result = _run_git(repo, "show", f"{sha}:{path}", timeout=30)
    if result.returncode != 0:
        absent = _run_git(repo, "ls-tree", "-z", sha, "--", path)
        if absent.returncode == 0 and not absent.stdout:
            return None  # The commit deletes this path.
        raise OSError("could not inspect a committed blob")
    return result.stdout


def _validate_ahead_commits(
    repo: Path,
    *,
    upstream: str,
    include_paths: list[str] | None,
    exclude_paths: list[str] | None,
) -> tuple[bool, list[str]]:
    """Inspect every local-ahead commit. Return (ok, error_lines).

    Each commit must:
        1. Touch only paths that match the include list (or all paths if
           include is None) AND do not match the exclude list.
        2. Have blob content free of probable credentials.

    Error lines describe ONLY the offending path, commit short-hash, and
    redacted fingerprint. Full credential values are never written.
    """
    summaries = _ahead_commit_summaries(repo, upstream)
    errors: list[str] = []
    for summary in summaries:
        commit = summary["short"]
        # 1. Path policy
        for path in summary["paths"]:
            allowed = (
                include_paths is None
                or matches_sync_path(path, include_paths)
            )
            excluded = bool(exclude_paths) and matches_sync_path(path, exclude_paths)
            if not allowed or excluded:
                errors.append(
                    f"commit {commit} contains prohibited path {path}"
                )
                # No need to scan the blob if the path is already disallowed.
                continue
            # 2. Secret content policy
            blob = _commit_blob(repo, summary["sha"], path)
            if blob is None:
                continue
            for hit in find_secrets_in_blob(blob):
                errors.append(
                    "commit {commit} contains probable {kind} credential "
                    "in {path} (fingerprint: {fp}, field: {field})".format(
                        commit=commit,
                        kind=hit["kind"],
                        path=path,
                        fp=hit["fingerprint"],
                        field=hit["field"],
                    )
                )
    return (len(errors) == 0, errors)


def sync_context_repo(
    repo: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    message: str = "nanobot: auto-sync context updates",
) -> bool:
    """Save selected changes and synchronize under a shared Git-directory lock.

    A conflict retains local commits and upstream history for reconciliation.
    Unrelated index/worktree changes are never stashed or committed. Callers
    must opt in only designated knowledge repositories to automatic sync.
    """
    try:
        if not is_git_repo(repo):
            return False
        common = _run_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if common.returncode != 0 or not common.stdout.strip():
            return False
        with FileLock(str(Path(common.stdout.strip()) / "nanobot-sync.lock"), timeout=0):
            return _sync_context_repo_locked(repo, include_paths, exclude_paths, message)
    except Timeout:
        logger.warning("Context repo sync already in progress: {}", repo)
    except (subprocess.TimeoutExpired, OSError):
        logger.error("Context repo sync unavailable or timed out; local changes retained: {}", repo)
    return False


def _sync_context_repo_locked(
    repo: Path, include_paths: list[str] | None, exclude_paths: list[str] | None, message: str,
) -> bool:
    if has_unmerged_paths(repo):
        return False
    for marker in ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD"):
        location = _run_git(repo, "rev-parse", "--git-path", marker)
        path = Path(location.stdout.strip())
        if location.returncode != 0 or (repo / path).exists():
            logger.error("Context repo has an unfinished Git operation; refusing sync")
            return False
    selected = _select_changed_paths(repo, include_paths, exclude_paths)
    if selected and not _commit_selected_paths(repo, selected, message):
        return False
    # Commit before integrating upstream: a conflict can be aborted without
    # losing the saved knowledge or altering unrelated dirty/staged versions.
    for attempt in range(4):
        if not sync_with_remote(repo):
            return False
        upstream = _upstream_ref(repo)
        if not upstream:
            return False
        ok, errors = _validate_ahead_commits(
            repo, upstream=upstream, include_paths=include_paths, exclude_paths=exclude_paths,
        )
        if not ok:
            for line in errors:
                logger.error("Context repo push refused: {}", line)
            return False
        if not _ahead_commit_summaries(repo, upstream):
            return True
        push = _run_git(repo, "push", timeout=60)
        if push.returncode == 0:
            logger.info("Context repo synced successfully: {}", repo)
            return True
        logger.warning("Context repo push attempt {} failed: {}", attempt + 1, _git_output(push))
    logger.error("Context repo push failed after 4 attempts; local commits retained")
    return False


async def async_sync_context_repo(
    repo: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    message: str = "nanobot: auto-sync context updates",
) -> bool:
    """Run sync_context_repo in a thread pool to avoid blocking the event loop."""
    work = asyncio.create_task(asyncio.to_thread(
        sync_context_repo, repo, include_paths=include_paths, exclude_paths=exclude_paths, message=message,
    ))
    cancelled = False
    while not work.done():
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            cancelled = True
    result = work.result()
    if cancelled:
        raise asyncio.CancelledError
    return result
