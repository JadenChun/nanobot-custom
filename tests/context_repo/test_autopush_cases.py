"""autoPush cases C1-C4 for the knowledge save and sync owner.

Case text, setup, and expected outcomes follow
``a3-evidence/a3-reconciliation-2026-11-14/resume3/req-autopush/CASES.md``.
Remote observations read the remote fixture's refs directly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kohaku.agent.policy import RiskyActionPolicy
from kohaku.agent.tools.filesystem import WriteFileTool
from kohaku.agent.tools.repositories import SaveKnowledgeTool
from kohaku.context_repo.catalog import RepositoryCatalog, RepositoryEntry
from kohaku.providers.base import ToolCallRequest
from kohaku.utils.git_sync import sync_context_repo
from tests.context_repo.test_git_sync_safe_push import (
    _bootstrap_remote_with_initial,
    _git_out,
)

BODY = "Decision: keep the shared knowledge sync."
NOTE = "notes/decision.md"
SYNC_PATHS = ["notes/**"]


def _git_rc(repo: Path, *args: str) -> int:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
    ).returncode


def _knowledge_repo(
    tmp_path: Path, *, auto_push: bool | None,
) -> tuple[Path, Path, RepositoryCatalog, WriteFileTool]:
    """Register a knowledge repository whose remote tracks ``main`` at R0."""
    remote, _seed, local = _bootstrap_remote_with_initial(tmp_path)
    if auto_push is not None:
        (local / "nanobot.context.json").write_text(
            json.dumps({"name": "knowledge", "sync": {"autoPush": auto_push}}),
            encoding="utf-8",
        )
    catalog = RepositoryCatalog(tmp_path / "runtime")
    catalog.update(RepositoryEntry(
        name="knowledge", path=str(local), kind="knowledge",
        auto_sync=True, sync_paths=SYNC_PATHS,
    ))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return remote, local, catalog, WriteFileTool(workspace=workspace)


def _set_auto_push(local: Path, enabled: bool) -> None:
    (local / "nanobot.context.json").write_text(
        json.dumps({"name": "knowledge", "sync": {"autoPush": enabled}}),
        encoding="utf-8",
    )


async def _save(
    catalog: RepositoryCatalog,
    writer: WriteFileTool,
    *,
    path: str = NOTE,
    body: str = BODY,
) -> dict[str, Any]:
    result = await SaveKnowledgeTool(catalog, writer).execute("knowledge", path, body)
    return json.loads(result)


# C1 -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c1_autopush_off_keeps_the_save_and_commit_local(tmp_path: Path) -> None:
    remote, local, catalog, writer = _knowledge_repo(tmp_path, auto_push=False)
    remote_before = _git_out(remote, "rev-parse", "main")

    result = await _save(catalog, writer)

    # 1: the save succeeds and the note holds exactly the saved content.
    assert result["saved"] is True
    assert (local / NOTE).read_text(encoding="utf-8").rstrip("\n") == BODY
    # 2/5: the reported state is pending and never claims synced.
    assert result["sync_state"] == "pending"
    assert result["synced"] is False
    # 3: the remote is untouched.
    assert _git_out(remote, "rev-parse", "main") == remote_before
    # 4: at most one new local commit, it contains the note, and R0 cannot reach it.
    assert _git_out(local, "rev-list", "--count", f"{remote_before}..HEAD") == "1"
    changed = _git_out(local, "show", "--name-only", "--pretty=format:", "HEAD")
    assert NOTE in changed.splitlines()
    assert _git_rc(local, "merge-base", "--is-ancestor", "HEAD", remote_before) != 0


# C2 -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c2_autopush_on_pushes_through_the_owner(tmp_path: Path) -> None:
    remote, local, catalog, writer = _knowledge_repo(tmp_path, auto_push=True)
    remote_before = _git_out(remote, "rev-parse", "main")

    result = await _save(catalog, writer)

    # 1/2: the note exists and the state is synced.
    assert (local / NOTE).read_text(encoding="utf-8").rstrip("\n") == BODY
    assert result["sync_state"] == "synced"
    assert result["synced"] is True
    # 3/4: exactly one commit, and it holds only the selected note path.
    head = _git_out(local, "rev-parse", "HEAD")
    assert _git_out(local, "rev-list", "--count", f"{remote_before}..HEAD") == "1"
    changed = _git_out(local, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "HEAD")
    assert changed.splitlines() == [NOTE]
    # 5: the remote ref is that commit and the note content arrived intact.
    assert _git_out(remote, "rev-parse", "main") == head
    assert _git_out(remote, "show", f"main:{NOTE}").rstrip("\n") == BODY

    # 6: a path outside the configured sync set reports no sync and leaves R0.
    outside = await _save(catalog, writer, path="other/note.md")
    assert outside["saved"] is True
    assert outside["sync_state"] == "not_enabled"
    assert outside["synced"] is None
    assert _git_out(remote, "rev-parse", "main") == head


# C3 -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c3_shell_approval_is_identical_for_autopush_states(tmp_path: Path) -> None:
    # One repository, two configurations, fresh policy state for each run.
    _remote, local, _catalog, _writer = _knowledge_repo(tmp_path, auto_push=False)
    command = "git reset --hard"
    call = ToolCallRequest(
        id="c3", name="exec",
        arguments={"command": command, "working_dir": str(local)},
    )

    decisions = []
    for enabled in (True, False):
        _set_auto_push(local, enabled)
        policy = RiskyActionPolicy(workspace=local)
        decisions.append(await policy.evaluate(messages=[], tool_calls=[call]))

    on, off = decisions
    # 1: both runs require approval.
    assert on.requires_approval is True
    assert off.requires_approval is True
    assert on.stop_reason == off.stop_reason == "approval_required"
    # 3: neither run is exempt or pre-approved.
    assert on.action == off.action == "respond"
    # 2/4: every field is identical once the per-request approval ID is removed.
    def normalised(decision: Any) -> dict[str, Any]:
        metadata = {k: v for k, v in decision.metadata.items() if k != "approval_id"}
        return {
            "action": decision.action, "stop_reason": decision.stop_reason,
            "metadata": metadata, "requires_approval": decision.requires_approval,
        }

    assert normalised(on) == normalised(off)
    assert "autopush" not in json.dumps(normalised(on)).lower()
    assert "exempt" not in json.dumps(normalised(on)).lower()


# C4 -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c4_pending_commit_pushes_after_reenable(tmp_path: Path) -> None:
    remote, local, catalog, writer = _knowledge_repo(tmp_path, auto_push=False)
    first = await _save(catalog, writer)
    assert first["sync_state"] == "pending"
    retained = _git_out(local, "rev-parse", "HEAD")

    # Re-enable autoPush and push the retained commit without re-saving the note.
    _set_auto_push(local, True)
    assert sync_context_repo(local, include_paths=SYNC_PATHS, push=True) is True

    # 1/2: the owner reports a completed sync and the remote matches the local head.
    remote_head = _git_out(remote, "rev-parse", "main")
    assert remote_head == _git_out(local, "rev-parse", "HEAD")
    # 3: the earlier local commit is an ancestor of the pushed commit.
    assert _git_rc(local, "merge-base", "--is-ancestor", retained, remote_head) == 0
    # 4: the note content arrived without loss.
    assert _git_out(remote, "show", f"main:{NOTE}").rstrip("\n") == BODY
    # 5: no duplicate commit introduces the note.
    assert _git_out(local, "rev-list", "--count", "--all", "--", NOTE) == "1"
    assert _git_out(local, "status", "--porcelain", "--", NOTE) == ""
