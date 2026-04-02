"""Git sync utility for context repositories."""

from __future__ import annotations

import asyncio
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


def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository."""
    try:
        result = _run_git(path, "rev-parse", "--is-inside-work-tree")
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def has_changes(repo: Path) -> bool:
    """Check if the git repo has uncommitted changes."""
    try:
        result = _run_git(repo, "status", "--porcelain")
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def sync_context_repo(repo: Path) -> bool:
    """Commit and push any changes in the context repo.

    Returns True if sync succeeded (or nothing to sync), False on failure.
    """
    if not is_git_repo(repo):
        logger.debug("Context path is not a git repo, skipping sync: {}", repo)
        return False

    if not has_changes(repo):
        logger.debug("No changes in context repo: {}", repo)
        return True

    try:
        # Pull latest first to avoid conflicts
        pull = _run_git(repo, "pull", "--rebase", "--autostash", timeout=60)
        if pull.returncode != 0:
            logger.warning("Context repo git pull failed: {}", pull.stderr.strip())

        # Stage all changes
        add = _run_git(repo, "add", "-A")
        if add.returncode != 0:
            logger.error("Context repo git add failed: {}", add.stderr.strip())
            return False

        # Commit (pass identity inline so it works even if git user.name/email is not globally configured)
        commit = _run_git(
            repo,
            "-c", "user.name=nanobot",
            "-c", "user.email=nanobot@localhost",
            "commit", "-m", "nanobot: auto-sync context updates",
        )
        if commit.returncode != 0:
            stderr = commit.stderr.strip()
            if "nothing to commit" in (commit.stdout + stderr):
                return True
            logger.error("Context repo git commit failed: {}", stderr)
            return False

        # Push with retry
        for attempt in range(4):
            push = _run_git(repo, "push", timeout=60)
            if push.returncode == 0:
                logger.info("Context repo synced successfully: {}", repo)
                return True
            logger.warning("Context repo push attempt {} failed: {}", attempt + 1, push.stderr.strip())
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


async def async_sync_context_repo(repo: Path) -> bool:
    """Run sync_context_repo in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sync_context_repo, repo)
