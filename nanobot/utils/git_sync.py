"""Git sync utility for context repositories."""

from __future__ import annotations

import asyncio
from fnmatch import fnmatch
import subprocess
from pathlib import Path

from loguru import logger


def _run_git(repo: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run a git command in the given repo directory."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=timeout,
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
        result = _run_git(repo, "status", "--porcelain")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if result.returncode != 0:
        return []
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        raw = line[3:].strip()
        if " -> " in raw:
            raw = raw.rsplit(" -> ", 1)[1]
        raw = raw.strip('"')
        if raw:
            if raw.endswith("/") and (repo / raw).is_dir():
                for child in sorted((repo / raw).rglob("*")):
                    if child.is_file():
                        paths.append(child.relative_to(repo).as_posix())
            else:
                paths.append(raw)
    return paths


def _matches(path: str, patterns: list[str]) -> bool:
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
        if include_paths is not None and not _matches(path, include_paths):
            continue
        if exclude_paths and _matches(path, exclude_paths):
            continue
        selected.append(path)
    return selected


def sync_with_remote(repo: Path) -> bool:
    """Fetch and rebase onto the configured upstream before local commits/pushes."""
    if has_unmerged_paths(repo):
        logger.error("Context repo has unresolved git conflicts; refusing to sync: {}", repo)
        return False

    fetch = _run_git(repo, "fetch", "--prune", timeout=60)
    if fetch.returncode != 0:
        logger.error("Context repo git fetch failed: {}", _git_output(fetch))
        return False

    pull = _run_git(repo, "pull", "--rebase", "--autostash", timeout=60)
    if pull.returncode != 0:
        logger.error("Context repo git pull --rebase failed: {}", _git_output(pull))
        return False

    if has_unmerged_paths(repo):
        logger.error("Context repo has unresolved conflicts after pull; refusing to sync: {}", repo)
        return False
    return True


def sync_context_repo(
    repo: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    message: str = "nanobot: auto-sync context updates",
) -> bool:
    """Commit and push any changes in the context repo.

    Returns True if sync succeeded (or nothing to sync), False on failure.
    """
    if not is_git_repo(repo):
        logger.debug("Context path is not a git repo, skipping sync: {}", repo)
        return False

    if not sync_with_remote(repo):
        return False

    if not has_changes(repo):
        logger.debug("No changes in context repo: {}", repo)
        return True

    selected_paths = _select_changed_paths(repo, include_paths, exclude_paths)
    if not selected_paths:
        logger.debug("No selected changes in context repo: {}", repo)
        return True

    try:
        # Stage only selected changes for managed context repos.
        add = _run_git(repo, "add", "-A", "--", *selected_paths)
        if add.returncode != 0:
            logger.error("Context repo git add failed: {}", _git_output(add))
            return False

        # Commit
        commit = _run_git(repo, "commit", "-m", message)
        if commit.returncode != 0:
            output = _git_output(commit)
            if "nothing to commit" in output:
                return True
            logger.error("Context repo git commit failed: {}", output)
            return False

        # Push with retry; re-sync before every push attempt so remote updates are
        # rebased before pushing.
        for attempt in range(4):
            if not sync_with_remote(repo):
                return False
            push = _run_git(repo, "push", timeout=60)
            if push.returncode == 0:
                logger.info("Context repo synced successfully: {}", repo)
                return True
            logger.warning("Context repo push attempt {} failed: {}", attempt + 1, _git_output(push))
            if attempt < 3:
                import time
                time.sleep(2 ** (attempt + 1))

        logger.error("Context repo push failed after 4 attempts")
        return False

    except subprocess.TimeoutExpired:
        logger.error("Context repo sync timed out: {}", repo)
        return False
    except FileNotFoundError:
        logger.error("git not found on PATH, cannot sync context repo")
        return False


async def async_sync_context_repo(
    repo: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    message: str = "nanobot: auto-sync context updates",
) -> bool:
    """Run sync_context_repo in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: sync_context_repo(
            repo,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            message=message,
        ),
    )
