"""A3-F acceptance cases U1-U8 for H01-F5 (shell/Git approval) and K01-F06.

Case text, inputs, and expected outcomes follow
``a3-evidence/a3-reconciliation-2026-11-14/resume3/req-shell/CASES.md``.
Only the public boundary is asserted: the risky-action policy decision, the
exact-effect approval record, and the prepared command handed to the executor.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.policy import (
    RiskyActionPolicy,
    nested_launch_id,
    nested_launch_path,
)
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.turn import ApprovalDecision, TurnContext
from nanobot.agent.turn_admission import ApprovalAdmission
from nanobot.providers.base import ToolCallRequest
from nanobot.utils import git_sync
from nanobot.utils.git_sync import sync_context_repo

LAUNCH_REFUSAL = "launch_confinement_unproven"
REDUCED_LAUNCH = "codex --yolo 'fix the bug'"
PROTECTED_LAUNCH = "codex exec --sandbox workspace-write 'fix the bug'"
REDUCED_TMUX_LAUNCH = f"tmux send-keys -t agent-1 \"{REDUCED_LAUNCH}\" Enter"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote and a clone with a tracked upstream and one commit."""
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    local = tmp_path / "local"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "clone", str(remote), str(seed))
    for repo in (seed,):
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test")
    (seed / "README.md").write_text("base\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "branch", "-M", "main")
    _git(seed, "push", "-u", "origin", "main")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(tmp_path, "clone", str(remote), str(local))
    _git(local, "config", "user.email", "test@example.com")
    _git(local, "config", "user.name", "Test")
    return remote, local


def _auto_push_manager(root: Path) -> tuple[MagicMock, MagicMock]:
    """A context manager reporting an autoPush knowledge repository."""
    repo = MagicMock()
    repo.path = str(root)
    repo.auto_push_enabled.return_value = True
    manager = MagicMock()
    manager.find_repo_for_path.return_value = repo
    manager.find_target_repo_for_path.return_value = None
    return manager, repo


def _exec(command: str, working_dir: str | None = None) -> ToolCallRequest:
    arguments: dict[str, Any] = {"command": command}
    if working_dir is not None:
        arguments["working_dir"] = working_dir
    return ToolCallRequest(id=f"exec-{abs(hash(command))}", name="exec", arguments=arguments)


async def _decide(policy: RiskyActionPolicy, call: ToolCallRequest) -> Any:
    return await policy.evaluate(messages=[], tool_calls=[call])


async def _approved_policy(
    root: Path, call: ToolCallRequest, **kwargs: Any,
) -> tuple[RiskyActionPolicy, TurnContext]:
    """Grant one exact-effect approval for ``call`` and return the bound policy."""
    effect = RiskyActionPolicy(workspace=root)._effect_fingerprint(call)  # pyright: ignore[reportPrivateUsage]
    admission = ApprovalAdmission()
    pending = await admission.store(
        session_key="cli:shell", channel="cli", chat_id="shell",
        summary="approve the exact effect", effect_id=effect,
    )
    context = TurnContext(run_id="resume", session_key="cli:shell")
    snapshot = await admission.snapshot_decision(
        "cli:shell", ApprovalDecision(approval_id=pending.approval_id, decision="approve"),
    )
    assert await admission.consume(snapshot, context=context) == "affirmative"
    return (
        RiskyActionPolicy(
            workspace=root, effect_authority=context.policy.effect_authority, **kwargs,
        ),
        context,
    )


# U1 -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u1_ordinary_git_read_allowed_without_approval_or_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, local = _repo(tmp_path)
    manager, repo = _auto_push_manager(local)
    synced: list[Path] = []
    monkeypatch.setattr(git_sync, "sync_context_repo", lambda *a, **k: synced.append(a[0]))
    policy = RiskyActionPolicy(workspace=local, context_manager=manager)

    for command in ("git status --porcelain", "git log -1 --oneline", "git diff --stat"):
        decision = await _decide(policy, _exec(command, str(local)))
        assert decision.action == "allow"
        assert decision.requires_approval is False
        assert decision.metadata == {}

    # OBS-2: the read runs and returns its result.
    tool = ExecTool(working_dir=str(local))
    result = await tool.execute("git status --porcelain")
    assert "Error" not in str(result)

    # OBS-3 and "must not": the sync owner is never invoked and autoPush is
    # never consulted by the shell policy.
    assert synced == []
    repo.auto_push_enabled.assert_not_called()
    manager.find_repo_for_path.assert_not_called()


# U2 -----------------------------------------------------------------------


def test_u2_configured_sync_runs_through_owner_with_each_check(
    tmp_path: Path,
) -> None:
    remote, local = _repo(tmp_path)
    (local / "notes").mkdir()
    (local / "notes" / "case.md").write_text("knowledge\n", encoding="utf-8")

    # OBS-1/OBS-3: one configured sync for the configured path commits and pushes.
    assert sync_context_repo(local, include_paths=["notes/**"]) is True
    assert "knowledge" in _git(remote, "show", "main:notes/case.md")

    # OBS-2: each owner check runs and can fail on its own.
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert sync_context_repo(not_a_repo, include_paths=["notes/**"]) is False

    no_upstream = tmp_path / "no-upstream"
    no_upstream.mkdir()
    _git(no_upstream, "init")
    _git(no_upstream, "config", "user.email", "test@example.com")
    _git(no_upstream, "config", "user.name", "Test")
    (no_upstream / "notes").mkdir()
    (no_upstream / "notes" / "a.md").write_text("x\n", encoding="utf-8")
    _git(no_upstream, "add", ".")
    _git(no_upstream, "commit", "-m", "local only")
    assert sync_context_repo(no_upstream, include_paths=["notes/**"]) is False


def test_u2_sync_refuses_prohibited_path_and_secret_without_pushing(
    tmp_path: Path,
) -> None:
    remote, local = _repo(tmp_path)
    head = _git(remote, "rev-parse", "main")

    (local / "secrets").mkdir()
    (local / "secrets" / "token.txt").write_text("not allowed\n", encoding="utf-8")
    _git(local, "add", ".")
    _git(local, "commit", "-m", "prohibited path")
    assert sync_context_repo(local, include_paths=["notes/**"]) is False
    assert _git(remote, "rev-parse", "main") == head

    _git(local, "reset", "--hard", "HEAD~1")
    (local / "notes").mkdir()
    (local / "notes" / "leak.json").write_text(
        '{"access_token": "EAA' + "Z" * 200 + '"}\n', encoding="utf-8",
    )
    _git(local, "add", ".")
    _git(local, "commit", "-m", "leaked token")
    assert sync_context_repo(local, include_paths=["notes/**"]) is False
    assert _git(remote, "rev-parse", "main") == head


@pytest.mark.asyncio
async def test_u2_sync_grant_does_not_authorize_shell(tmp_path: Path) -> None:
    _, local = _repo(tmp_path)
    (local / "notes").mkdir()
    (local / "notes" / "after.md").write_text("after\n", encoding="utf-8")
    assert sync_context_repo(local, include_paths=["notes/**"]) is True

    # OBS-4/OBS-5: no shell approval for the sync, and no grant carries over.
    manager, _ = _auto_push_manager(local)
    policy = RiskyActionPolicy(workspace=local, context_manager=manager)
    decision = await _decide(policy, _exec("git reset --hard", str(local)))
    assert decision.stop_reason == "approval_required"
    assert decision.requires_approval is True


# U3 -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u3_destructive_git_requires_exact_approval(tmp_path: Path) -> None:
    _, local = _repo(tmp_path)
    (local / "second.txt").write_text("second\n", encoding="utf-8")
    _git(local, "add", ".")
    _git(local, "commit", "-m", "second")
    manager, _ = _auto_push_manager(local)
    policy = RiskyActionPolicy(workspace=local, context_manager=manager)
    baseline = _git(local, "rev-parse", "HEAD")

    for command, summary in (
        ("git reset --hard HEAD~1", "reset git history"),
        ("git clean -fd", "remove untracked files"),
    ):
        (local / "untracked.txt").write_text("keep\n", encoding="utf-8")
        decision = await _decide(policy, _exec(command, str(local)))
        assert decision.stop_reason == "approval_required"
        assert decision.requires_approval is True
        assert decision.metadata["summary"] == summary
        assert decision.metadata["effect_id"]
        # OBS-2: the denied run changes nothing.
        assert _git(local, "rev-parse", "HEAD") == baseline
        assert (local / "untracked.txt").exists()

    # OBS-3: an exact approval performs the effect once.
    call = _exec("git reset --hard HEAD~1", str(local))
    parent = _git(local, "rev-parse", "HEAD~1")
    approved, context = await _approved_policy(local, call)
    assert (await _decide(approved, call)).action == "allow"
    assert approved.authorize_dispatch(call, dict(call.arguments), context).action == "allow"
    await ExecTool(working_dir=str(local)).execute("git reset --hard HEAD~1")
    assert _git(local, "rev-parse", "HEAD") == parent


# U4 -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u4_benign_git_segment_does_not_exempt_risky_effect(
    tmp_path: Path,
) -> None:
    _, local = _repo(tmp_path)
    build = local / "build"
    build.mkdir()
    (build / "artifact.bin").write_text("artifact\n", encoding="utf-8")
    manager, _ = _auto_push_manager(local)
    policy = RiskyActionPolicy(workspace=local, context_manager=manager)

    call = _exec("git status && rm -rf build", str(local))
    decision = await _decide(policy, call)
    # OBS-1: the benign Git segment does not make the invocation allowed.
    assert decision.stop_reason == "approval_required"
    # OBS-2: the approved effect is the removal, not the Git read.
    assert decision.metadata["summary"] == "delete files or directories"
    # OBS-3: the denied run leaves the directory intact. (OBS-4 is supporting
    # and is also blocked by the shell tool's independent deny rule.)
    assert (build / "artifact.bin").read_text() == "artifact\n"


# U5 -----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox", ["", "bwrap"], ids=["full", "sandbox"])
async def test_u5_supported_launch_keeps_protections_without_approval(
    tmp_path: Path, sandbox: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for command in (PROTECTED_LAUNCH, f"tmux send-keys -t a1 \"{PROTECTED_LAUNCH}\" Enter"):
        tool = ExecTool(working_dir=str(tmp_path), sandbox=sandbox)
        decisions = await _decide(
            RiskyActionPolicy(workspace=tmp_path), _exec(command, str(tmp_path)),
        )
        # OBS-1/OBS-2: allowed, and no reduced-protection approval is requested.
        assert decisions.action == "allow"
        assert decisions.requires_approval is False
        assert decisions.metadata == {}

        # OBS-3: the executor receives the parent's settings unchanged.
        launch = AsyncMock(return_value="session-ready")
        monkeypatch.setattr(tool, "_execute_session", launch)
        assert await tool.execute(command=command, yield_time_ms=0) == "session-ready"
        launch.assert_awaited_once()
        prepared = launch.call_args.args[0]
        assert "workspace-write" in prepared.command
        assert "--yolo" not in prepared.command
        assert prepared.launch_attestation is None


# U6 / U7 ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u6_reduced_launch_without_confinement_is_refused_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = ExecTool(working_dir=str(tmp_path))
    launch = AsyncMock(return_value="started")
    monkeypatch.setattr(tool, "_execute_session", launch)
    decision = await _decide(RiskyActionPolicy(workspace=tmp_path), _exec(REDUCED_LAUNCH))

    # OBS-1: one typed refusal, identical to U7.
    assert decision.stop_reason == "policy_blocked"
    assert decision.metadata["reason_id"] == LAUNCH_REFUSAL
    # OBS-3: the outcome is not an approval request.
    assert decision.requires_approval is False
    assert decision.stop_reason != "approval_required"
    assert "approval_id" not in decision.metadata
    # OBS-2: no executor launch, no process start, and no attestation.
    launch.assert_not_awaited()
    assert tool.attest_launch_confinement(REDUCED_LAUNCH, str(tmp_path)) is None


@pytest.mark.asyncio
async def test_u7_exact_approval_without_confinement_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _exec(REDUCED_LAUNCH)
    approved, context = await _approved_policy(tmp_path, call)
    tool = ExecTool(working_dir=str(tmp_path))
    launch = AsyncMock(return_value="started")
    monkeypatch.setattr(tool, "_execute_session", launch)

    decision = await _decide(approved, call)
    # OBS-1/OBS-3: same typed refusal; approval is not a substitute.
    assert decision.stop_reason == "policy_blocked"
    assert decision.metadata["reason_id"] == LAUNCH_REFUSAL
    assert decision.requires_approval is False
    # OBS-2: the refusal happens before any process.
    launch.assert_not_awaited()
    refused = approved.authorize_dispatch(call, dict(call.arguments), context)
    assert refused.stop_reason == "policy_blocked"
    assert refused.metadata["reason_id"] == LAUNCH_REFUSAL


# U8 -----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("restrict", [False, True], ids=["full-access", "sandbox"])
async def test_u8_reduced_launch_with_approval_and_attestation_starts_once(
    tmp_path: Path, restrict: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = ExecTool(
        working_dir=str(tmp_path), sandbox="bwrap", restrict_to_workspace=restrict,
    )
    call = _exec(REDUCED_LAUNCH, str(tmp_path))
    approved, context = await _approved_policy(tmp_path, call)

    # Reduced protections require exact approval in both access modes.
    assert (await _decide(approved, call)).action == "allow"
    dispatch = approved.authorize_dispatch(call, dict(call.arguments), context)
    assert dispatch.action == "allow"

    launch = AsyncMock(return_value="started")
    monkeypatch.setattr(tool, "_execute_session", launch)
    assert await tool.execute(command=REDUCED_LAUNCH, yield_time_ms=0) == "started"
    # OBS-1: exactly one launch.
    launch.assert_awaited_once()
    prepared = launch.call_args.args[0]
    # OBS-2: the delivered settings keep the approved reduced protections.
    assert "--yolo" in prepared.command
    assert prepared.command != REDUCED_LAUNCH  # outer sandbox is applied
    # OBS-4: the attestation is bound to the launch that started.
    attestation = prepared.launch_attestation
    assert attestation is not None
    assert attestation.covers(
        nested_launch_id(REDUCED_LAUNCH, str(tmp_path)), "shell",
    )
    assert attestation.sandbox == "bwrap"
    assert attestation.process_owner


@pytest.mark.asyncio
async def test_u8_full_access_without_attestation_still_refuses(tmp_path: Path) -> None:
    # The reduced-protection rule does not disappear at Full access.
    ExecTool(working_dir=str(tmp_path))  # an unconfined executor
    decision = await _decide(
        RiskyActionPolicy(workspace=tmp_path), _exec(REDUCED_LAUNCH, str(tmp_path)),
    )
    assert decision.metadata["reason_id"] == LAUNCH_REFUSAL


@pytest.mark.asyncio
async def test_u8_tmux_reduced_launch_cannot_prove_ownership(tmp_path: Path) -> None:
    # A tmux command types into a process the executor neither confines for
    # that launch nor owns, so R2 requires the refusal even with a sandbox.
    ExecTool(working_dir=str(tmp_path), sandbox="bwrap")
    decision = await _decide(
        RiskyActionPolicy(workspace=tmp_path),
        _exec(REDUCED_TMUX_LAUNCH, str(tmp_path)),
    )
    assert decision.stop_reason == "policy_blocked"
    assert decision.metadata["reason_id"] == LAUNCH_REFUSAL
    assert decision.metadata["launch_path"] == "tmux"
    assert decision.requires_approval is False


@pytest.mark.asyncio
async def test_u8v_attestation_for_another_launch_is_refused(tmp_path: Path) -> None:
    tool = ExecTool(working_dir=str(tmp_path), sandbox="bwrap")
    other = tool.attest_launch_confinement("codex --yolo 'a different task'", str(tmp_path))
    assert other is not None

    policy = RiskyActionPolicy(
        workspace=tmp_path, launch_confinement_attestor=lambda command, cwd: other,
    )
    decision = await _decide(policy, _exec(REDUCED_LAUNCH, str(tmp_path)))
    assert decision.stop_reason == "policy_blocked"
    assert decision.metadata["reason_id"] == LAUNCH_REFUSAL
    assert decision.requires_approval is False


@pytest.mark.asyncio
async def test_u8_executor_refuses_unattested_launch_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executor boundary refuses independently of the policy layer."""
    tool = ExecTool(working_dir=str(tmp_path))
    launch = AsyncMock(return_value="started")
    monkeypatch.setattr(tool, "_execute_session", launch)
    result = await tool.execute(command=REDUCED_LAUNCH)
    assert LAUNCH_REFUSAL in str(result)
    assert getattr(result, "is_error", False) is True
    launch.assert_not_awaited()


# K01-F06 review follow-up: case-correct and config-override bypass forms ------

# A lowercased classifier alternative never matched ``bypassPermissions``, and a
# Codex ``-c``/``--config`` override was not classified. These are the same
# reduced-protection launch as ``--yolo`` and must be refused before launch
# unless exact approval and a covering attestation exist.
BYPASS_LAUNCHES = (
    "claude --permission-mode bypassPermissions 'fix the bug'",
    "claude --permission-mode=bypassPermissions 'fix the bug'",
    "claude --permission-mode=\"bypassPermissions\" 'fix the bug'",
    "codex -c 'sandbox_mode=\"danger-full-access\"' 'fix the bug'",
    "codex --config sandbox_mode=danger-full-access 'fix the bug'",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", BYPASS_LAUNCHES)
async def test_k01f06_bypass_form_refused_before_spawn(
    tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = ExecTool(working_dir=str(tmp_path))  # unconfined executor
    launch, spawn = AsyncMock(return_value="started"), AsyncMock()
    monkeypatch.setattr(tool, "_execute_session", launch)
    monkeypatch.setattr(tool, "_spawn", spawn)

    decision = await _decide(
        RiskyActionPolicy(workspace=tmp_path), _exec(command, str(tmp_path)),
    )
    assert decision.stop_reason == "policy_blocked"
    assert decision.metadata["reason_id"] == LAUNCH_REFUSAL
    assert decision.requires_approval is False

    result = await tool.execute(command=command)
    assert LAUNCH_REFUSAL in str(result)
    launch.assert_not_awaited()
    spawn.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", BYPASS_LAUNCHES)
async def test_k01f06_bypass_form_with_approval_still_refused(
    tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = _exec(command, str(tmp_path))
    approved, context = await _approved_policy(tmp_path, call)
    tool = ExecTool(working_dir=str(tmp_path))
    launch = AsyncMock(return_value="started")
    monkeypatch.setattr(tool, "_execute_session", launch)

    decision = await _decide(approved, call)
    assert decision.stop_reason == "policy_blocked"
    assert decision.metadata["reason_id"] == LAUNCH_REFUSAL
    assert decision.requires_approval is False
    refused = approved.authorize_dispatch(call, dict(call.arguments), context)
    assert refused.metadata["reason_id"] == LAUNCH_REFUSAL
    result = await tool.execute(command=command)
    assert LAUNCH_REFUSAL in str(result)
    launch.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", BYPASS_LAUNCHES)
async def test_k01f06_bypass_form_allowed_with_approval_and_attestation(
    tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = ExecTool(working_dir=str(tmp_path), sandbox="bwrap")
    call = _exec(command, str(tmp_path))
    approved, context = await _approved_policy(tmp_path, call)

    assert (await _decide(approved, call)).action == "allow"
    assert approved.authorize_dispatch(call, dict(call.arguments), context).action == "allow"

    launch = AsyncMock(return_value="started")
    monkeypatch.setattr(tool, "_execute_session", launch)
    assert await tool.execute(command=command, yield_time_ms=0) == "started"
    launch.assert_awaited_once()
    prepared = launch.call_args.args[0]
    attestation = prepared.launch_attestation
    assert attestation is not None
    assert attestation.covers(nested_launch_id(command, str(tmp_path)), "shell")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "tmux send-keys -t a1 \"claude --permission-mode bypassPermissions 'x'\" Enter",
        "tmux send-keys -t a1 \"codex --config sandbox_mode=danger-full-access 'x'\" Enter",
    ),
)
async def test_k01f06_bypass_form_via_tmux_cannot_prove_ownership(
    tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = ExecTool(working_dir=str(tmp_path), sandbox="bwrap")
    launch, spawn = AsyncMock(return_value="started"), AsyncMock()
    monkeypatch.setattr(tool, "_execute_session", launch)
    monkeypatch.setattr(tool, "_spawn", spawn)

    decision = await _decide(
        RiskyActionPolicy(workspace=tmp_path), _exec(command, str(tmp_path)),
    )
    assert decision.stop_reason == "policy_blocked"
    assert decision.metadata["reason_id"] == LAUNCH_REFUSAL
    assert decision.metadata["launch_path"] == "tmux"
    result = await tool.execute(command=command)
    assert LAUNCH_REFUSAL in str(result)
    launch.assert_not_awaited()
    spawn.assert_not_awaited()


@pytest.mark.parametrize(
    "command",
    (
        "codex exec --sandbox workspace-write 'fix the bug'",
        "codex -c model=o3 'fix the bug'",
        "claude --permission-mode acceptEdits 'fix the bug'",
    ),
)
def test_k01f06_protected_forms_stay_ordinary(command: str) -> None:
    assert nested_launch_path(command) is None
