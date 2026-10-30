"""Managed WebUI commands are confined and rechecked at each process dispatch."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.exec_session import ExecSessionManager, ExecSessionTool
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.native_process import native_supervisor_available
from nanobot.agent.tools.registry import ManagedToolDispatchGuard, ToolRegistry
from nanobot.agent.tools.sandbox import managed_bwrap_available, wrap_managed_command
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.turn import ToolOutcome
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.security.desktop_permissions import (
    DesktopPermissionOwner,
    PermissionSourceBinding,
    WorkspacePermissionUnavailableError,
)
from nanobot.security.folder_grants import (
    FolderGrantStore,
    NativeProcessesActiveError,
    NativeProcessesUnknownError,
)
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)


def _managed_tools(tmp_path: Path, *, access: str = "read_write", commands: bool = True):
    home = tmp_path / "agent-home"
    project = tmp_path / "project-b"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(project), "file_access": access, "commands": commands},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
        managed_webui_only=True,
    )
    return home, project, store, policy


def _bind_managed_turn(home: Path, project: Path, *, persistent: bool = True):
    scope = build_workspace_scope(project, "full", source_channel="websocket")
    scope_token = bind_workspace_scope(scope)
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        workspace_scope=scope,
        persistent_session=persistent,
    )
    return scope_token, request_context(request)


def _legacy_exec_tool(tmp_path: Path) -> tuple[Path, FolderGrantStore, ExecTool]:
    home = tmp_path / "agent-home"
    instance = tmp_path / "instance"
    home.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
        managed_webui_only=False,
    )
    return home, store, ExecTool(working_dir=str(home), resource_policy=policy)


def _enable_full(
    owner: DesktopPermissionOwner,
    source_binding: PermissionSourceBinding,
    session_key: str,
):
    return owner.enable_full(
        session_key=session_key,
        source_binding=source_binding,
        persistent=True,
        existing=True,
        local=True,
        idle=True,
        configured_sandbox=False,
        supervisor_available=True,
        stores_healthy=True,
    )


@pytest.mark.asyncio
async def test_exec_preflight_denies_disabled_context_cwd_and_reference_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "agent-home"
    shared = tmp_path / "shared"
    home.mkdir()
    shared.mkdir()
    candidate = shared / "script.py"
    candidate.write_text("pass", encoding="utf-8")
    alias = home / "shared-alias"
    alias.symlink_to(shared, target_is_directory=True)
    (shared / "nanobot.context.json").write_text("{}", encoding="utf-8")
    manager = ContextRepoManager.from_config(
        context_repos=[{"path": str(shared), "allowExec": False}],
    )
    policy = ResourceAccessPolicy(workspace=home, context_manager=manager)
    tool = ExecTool(working_dir=str(home), resource_policy=policy)
    spawns: list[str] = []

    async def record_spawn(*args: object, **kwargs: object) -> None:
        _ = kwargs
        spawns.append(str(args[0]) if args else "")
        raise AssertionError("disabled context repo reached process launch")

    monkeypatch.setattr(tool, "_spawn", record_spawn)

    for command, cwd in (
        ("python3", str(shared)),
        (f"cat {candidate}", str(home)),
        ("cat ../shared/script.py", str(home)),
        (f"cat {alias / 'script.py'}", str(home)),
    ):
        result = await tool.execute(command=command, working_dir=cwd)
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "execution is disabled" in str(result)

    assert spawns == []


@pytest.mark.asyncio
@pytest.mark.parametrize("managed_latched", [False, True], ids=["pre-latch", "post-latch"])
async def test_private_full_authorizes_exec_outside_cwd_before_and_after_managed_latch(
    tmp_path: Path,
    managed_latched: bool,
) -> None:
    if not native_supervisor_available():
        pytest.skip("native process supervision is unavailable")

    home = tmp_path / "agent-home"
    outside = tmp_path / "outside"
    unrelated_grant = tmp_path / "unrelated-grant"
    home.mkdir()
    outside.mkdir()
    unrelated_grant.mkdir()
    marker = outside / "full-access-marker.txt"
    marker.write_text("PRIVATE_FULL_EXEC", encoding="utf-8")
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    if managed_latched:
        store.upsert(
            {"path": str(unrelated_grant), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
        managed_webui_only=True,
    )
    tool = ExecTool(
        working_dir=str(home),
        resource_policy=policy,
        restrict_to_workspace=True,
    )
    owner = DesktopPermissionOwner()
    session_key = "websocket:private-full-exec"
    source_binding: PermissionSourceBinding = {
        "store_identity": "private-full-exec-store",
        "incarnation": "private-full-exec-incarnation",
        "generation": 0,
    }
    reference = _enable_full(owner, source_binding, session_key)
    scope = build_workspace_scope(home, "full", source_channel="websocket")
    token = bind_workspace_scope(
        scope,
        permission_owner=owner,
        full_reference=reference,
    )
    request = RequestContext(
        channel="websocket",
        chat_id="private-full-exec",
        session_key=session_key,
        workspace=home,
        workspace_scope=scope,
        persistent_session=True,
    )
    command = f"cat {shlex.quote(str(marker))}"
    try:
        with request_context(request):
            result = await tool.execute(command=command, working_dir=str(outside))
        assert not isinstance(result, ToolOutcome), repr(result)
        assert "PRIVATE_FULL_EXEC" in result
        assert "Exit code: 0" in result
    finally:
        reset_workspace_scope(token)


@pytest.mark.parametrize(
    "proof_state",
    ["none", "copied", "revoked", "stale_source", "older_than_current"],
)
@pytest.mark.asyncio
async def test_exec_outside_workspace_requires_exact_live_full_reference(
    tmp_path: Path,
    proof_state: str,
) -> None:
    home = tmp_path / "agent-home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
        managed_webui_only=True,
    )
    owner = DesktopPermissionOwner()
    marker = outside / "must-not-be-created.txt"
    session_key = "websocket:full-exec-negative"
    source_binding: PermissionSourceBinding = {
        "store_identity": "full-exec-negative-store",
        "incarnation": "full-exec-negative-incarnation",
        "generation": 0,
    }
    reference = _enable_full(owner, source_binding, session_key)
    bound_reference = reference
    if proof_state == "copied":
        bound_reference = replace(reference)
    elif proof_state == "revoked":
        owner.disable_full(
            session_key=session_key,
            source_binding=source_binding,
            active_turn=False,
            active_processes=0,
            unknown_processes=0,
        )
    elif proof_state == "stale_source":
        owner.retire_source_for_lifecycle(
            session_key=session_key,
            source_binding=source_binding,
            reason="source_changed",
        )
    elif proof_state == "older_than_current":
        owner.disable_full(
            session_key=session_key,
            source_binding=source_binding,
            active_turn=False,
            active_processes=0,
            unknown_processes=0,
        )
        _enable_full(owner, source_binding, session_key)

    scope = build_workspace_scope(home, "full", source_channel="websocket")
    token = bind_workspace_scope(
        scope,
        permission_owner=owner,
        full_reference=None if proof_state == "none" else bound_reference,
    )
    tool = ExecTool(
        working_dir=str(home),
        resource_policy=policy,
        restrict_to_workspace=True,
    )
    try:
        result = await tool.execute(
            command=f"printf DENIED > {shlex.quote(str(marker))}",
            working_dir=str(home),
        )
        assert isinstance(result, ToolOutcome), repr(result)
        assert result.stop_reason == "folder_access_required"
        assert result.policy_metadata["folder_access"]["reason"] == "folder_not_granted"
        assert not marker.exists()
    finally:
        reset_workspace_scope(token)


def test_full_permission_remains_unavailable_with_configured_exec_sandbox(
    tmp_path: Path,
) -> None:
    owner = DesktopPermissionOwner()
    source_binding: PermissionSourceBinding = {
        "store_identity": "sandbox-full-store",
        "incarnation": "sandbox-full-incarnation",
        "generation": 0,
    }
    with pytest.raises(WorkspacePermissionUnavailableError) as rejected:
        owner.enable_full(
            session_key="websocket:sandbox-full",
            source_binding=source_binding,
            persistent=True,
            existing=True,
            local=True,
            idle=True,
            configured_sandbox=True,
            supervisor_available=True,
            stores_healthy=True,
        )
    assert rejected.value.reason == "configured_sandbox"


class _NativeHostTool(Tool):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    @property
    def name(self) -> str:
        return "agent_browser"

    @property
    def description(self) -> str:
        return "A fake unconfined host browser wrapper."

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: object) -> str:
        _ = kwargs
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return "host call completed"


@pytest.mark.asyncio
async def test_captured_legacy_registry_rechecks_host_tool_after_activation(tmp_path: Path) -> None:
    home = tmp_path / "agent-home"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    registry = ToolRegistry()
    tool = _NativeHostTool()
    registry.register(tool)
    registry.set_dispatch_guard(ManagedToolDispatchGuard(store.view))
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        persistent_session=True,
    )

    with request_context(request):
        store.upsert(
            {"path": str(project), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
        denied = await registry.execute("agent_browser", {})

    assert isinstance(denied, ToolResult)
    assert denied.is_error
    assert "managed folder access" in denied
    assert tool.calls == 0


@pytest.mark.asyncio
async def test_opaque_native_host_return_keeps_first_activation_unknown(tmp_path: Path) -> None:
    home = tmp_path / "agent-home"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    registry = ToolRegistry()
    tool = _NativeHostTool()
    registry.register(tool)
    registry.set_dispatch_guard(ManagedToolDispatchGuard(store.view))
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        persistent_session=True,
    )

    with request_context(request):
        running = asyncio.create_task(registry.execute("agent_browser", {}))
        await asyncio.wait_for(tool.started.wait(), timeout=2)
        with pytest.raises(NativeProcessesActiveError):
            store.upsert(
                {"path": str(project), "file_access": "read_only", "commands": False},
                expected_revision=0,
            )
        tool.release.set()
        assert await asyncio.wait_for(running, timeout=2) == "host call completed"

        with pytest.raises(NativeProcessesUnknownError):
            store.upsert(
                {"path": str(project), "file_access": "read_only", "commands": False},
                expected_revision=0,
            )

    assert tool.calls == 1


@pytest.mark.asyncio
async def test_managed_temporary_websocket_denies_unconfined_host_tool(tmp_path: Path) -> None:
    home, project, _store, policy = _managed_tools(tmp_path)
    registry = ToolRegistry()
    tool = _NativeHostTool()
    registry.register(tool)
    registry.set_dispatch_guard(
        ManagedToolDispatchGuard(policy.folder_grants),
    )
    scope_token, request = _bind_managed_turn(home, project, persistent=False)
    try:
        with request:
            result = await registry.execute("agent_browser", {})
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "managed folder access" in result
        assert tool.calls == 0
    finally:
        reset_workspace_scope(scope_token)


@pytest.mark.asyncio
async def test_managed_exec_runs_from_selected_grant_and_obeys_nested_mounts(tmp_path):
    if not managed_bwrap_available():
        pytest.skip("bubblewrap is unavailable on this host")
    home, project, store, policy = _managed_tools(tmp_path)
    (project / "marker.txt").write_text("B_MANAGED_MARKER", encoding="utf-8")
    child_read_only = project / "readonly"
    child_commands_off = project / "commands-off"
    child_read_only.mkdir()
    child_commands_off.mkdir()
    (child_read_only / "marker.txt").write_text("READ_ONLY_CHILD", encoding="utf-8")
    (child_commands_off / "secret.txt").write_text("HIDDEN_COMMANDS_OFF", encoding="utf-8")
    store.upsert(
        {"path": str(child_read_only), "file_access": "read_only", "commands": True},
        expected_revision=1,
    )
    store.upsert(
        {"path": str(child_commands_off), "file_access": "read_write", "commands": False},
        expected_revision=2,
    )
    tool = ExecTool(working_dir=str(home), resource_policy=policy, restrict_to_workspace=True)
    token, request = _bind_managed_turn(home, project)
    try:
        with request:
            read = await tool.execute(command="cat marker.txt")
            assert "B_MANAGED_MARKER" in read

            write = await tool.execute(command="printf changed > new-file.txt")
            assert "Exit code: 0" in write
            assert (project / "new-file.txt").read_text(encoding="utf-8") == "changed"

            ro_write = await tool.execute(command="printf denied > readonly/new-file.txt")
            assert "Exit code: 0" not in ro_write
            assert not (child_read_only / "new-file.txt").exists()

            readable = await tool.execute(command="cat commands-off/secret.txt")
            assert "HIDDEN_COMMANDS_OFF" in readable
            assert "Exit code: 0" in readable
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_managed_exec_hides_configured_context_and_target_repositories(tmp_path):
    if not managed_bwrap_available():
        pytest.skip("bubblewrap is unavailable on this host")

    home = tmp_path / "agent-home"
    parent = tmp_path / "granted-parent"
    context_repo = parent / "context"
    target_repo = parent / "target"
    instance = tmp_path / "instance"
    for directory in (home, context_repo, target_repo):
        directory.mkdir(parents=True)
    (context_repo / "context.canary").write_text("CONTEXT_PRIVATE", encoding="utf-8")
    (target_repo / "target.canary").write_text("TARGET_PRIVATE", encoding="utf-8")
    (context_repo / "nanobot.context.json").write_text(
        json.dumps({
            "name": "managed-context",
            "writable": ["outputs/**"],
            "targetRepos": {
                "managed-target": {
                    "path": str(target_repo),
                    "read": ["**"],
                    "write": ["content/**"],
                    "proposalRequired": ["src/**"],
                },
            },
        }),
        encoding="utf-8",
    )
    context_manager = ContextRepoManager.from_config(context_repos=[str(context_repo)])
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(parent), "file_access": "read_write", "commands": True},
        expected_revision=0,
    )
    store.upsert(
        {"path": str(context_repo), "file_access": "read_write", "commands": True},
        expected_revision=1,
    )
    store.upsert(
        {"path": str(target_repo), "file_access": "read_write", "commands": True},
        expected_revision=2,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=context_manager,
        restrict_to_workspace=True,
        folder_grants=store.view,
        managed_webui_only=True,
    )
    tool = ExecTool(working_dir=str(home), resource_policy=policy, restrict_to_workspace=True)
    scope = build_workspace_scope(parent, "full", source_channel="websocket")
    token = bind_workspace_scope(scope)
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        workspace_scope=scope,
    )
    try:
        with request_context(request):
            listing = await tool.execute(command="find . -name '*.canary' -print")
            repo_cwd = await tool.execute(command="pwd", working_dir=str(context_repo))

        assert "context.canary" not in listing
        assert "target.canary" not in listing
        assert "CONTEXT_PRIVATE" not in listing
        assert "TARGET_PRIVATE" not in listing
        assert "Exit code: 0" in listing
        assert "managed_repository_policy" in repo_cwd
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_native_exec_preserves_command_result_and_reconciles_ordinary_tree(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux subreaper supervisor is only available on Linux")
    home, store, tool = _legacy_exec_tool(tmp_path)
    token, request = _bind_managed_turn(home, home)
    try:
        with request:
            result = await tool.execute(
                command=f"{shlex.quote(sys.executable)} -c 'import sys; print(\"ROOT_STDOUT\"); sys.exit(7)'"
            )
        assert "ROOT_STDOUT" in result
        assert "Exit code: 7" in result

        root = tmp_path / "project"
        root.mkdir()
        for _attempt in range(100):
            try:
                snapshot = store.upsert(
                    {"path": str(root), "file_access": "read_only", "commands": False},
                    expected_revision=0,
                )
                break
            except NativeProcessesActiveError:
                await asyncio.sleep(0.02)
        else:
            pytest.fail("ordinary native command did not produce a quiescence receipt")
        assert snapshot.enabled is True
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_detached_double_fork_blocks_activation_after_root_result_until_echild(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux subreaper supervisor is only available on Linux")
    home, store, tool = _legacy_exec_tool(tmp_path)
    marker = tmp_path / "detached-finished"
    script = f"""import os, time
child = os.fork()
if child:
    os.write(1, b'ROOT_STDOUT\\n')
    os._exit(7)
os.setsid()
grandchild = os.fork()
if grandchild:
    os._exit(0)
for fd in (0, 1, 2):
    os.close(fd)
time.sleep(0.8)
open({str(marker)!r}, 'w').write('finished')
os._exit(0)
"""
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    token, request = _bind_managed_turn(home, home)
    try:
        started = time.monotonic()
        with request:
            result = await tool.execute(command=command)
        elapsed = time.monotonic() - started
        assert "ROOT_STDOUT" in result
        assert "Exit code: 7" in result
        assert elapsed < 0.7, f"command result waited for detached descendant ({elapsed:.3f}s)"
        assert not marker.exists()

        project = tmp_path / "project"
        project.mkdir()
        with pytest.raises(NativeProcessesActiveError):
            store.upsert(
                {"path": str(project), "file_access": "read_only", "commands": False},
                expected_revision=0,
            )
        await asyncio.sleep(0.35)
        assert not marker.exists()
        with pytest.raises(NativeProcessesActiveError):
            store.upsert(
                {"path": str(project), "file_access": "read_only", "commands": False},
                expected_revision=0,
            )

        for _attempt in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.02)
        assert marker.read_text(encoding="utf-8") == "finished"
        for _attempt in range(100):
            try:
                snapshot = store.upsert(
                    {"path": str(project), "file_access": "read_only", "commands": False},
                    expected_revision=0,
                )
                break
            except NativeProcessesActiveError:
                await asyncio.sleep(0.02)
        else:
            pytest.fail("detached native command did not produce a quiescence receipt")
        assert snapshot.enabled is True
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_once_bwrap_descendant_is_supervised_until_owner_receipt(tmp_path: Path) -> None:
    """A real task-only bwrap tree stays owned through its quiescence receipt."""
    if sys.platform != "linux" or not managed_bwrap_available():
        pytest.skip("Linux bubblewrap and native supervision are required")

    home = tmp_path / "agent-home"
    project = tmp_path / "task-project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
        managed_webui_only=True,
    )
    owner = DesktopPermissionOwner()
    source_binding = {
        "store_identity": "once-bwrap-store",
        "incarnation": "once-bwrap-incarnation",
        "generation": 0,
    }
    reference = owner.activate_once(
        session_key="websocket:once-bwrap-chat",
        source_binding=source_binding,
        task_turn_id="once-bwrap-turn",
        task_input_event_id="once-bwrap-event",
        path=project,
        file_access="read_write",
        commands=True,
    )
    descendant_ready_marker = project / "detached-descendant-ready"
    started_marker = project / "task-root-started"
    finished_marker = project / "detached-descendant-finished"
    script = f"""import os, time
open({str(started_marker)!r}, 'w').write('started')
child = os.fork()
if child:
    deadline = time.monotonic() + 2
    while not os.path.exists({str(descendant_ready_marker)!r}):
        if time.monotonic() > deadline:
            os._exit(9)
        time.sleep(0.005)
    os._exit(0)
os.setsid()
grandchild = os.fork()
if grandchild:
    os._exit(0)
open({str(descendant_ready_marker)!r}, 'w').write('ready')
for fd in (0, 1, 2):
    os.close(fd)
time.sleep(0.8)
open({str(finished_marker)!r}, 'w').write('finished')
os._exit(0)
"""
    # The QA interpreter under /tmp is deliberately not exposed to the sandbox.
    # A bare command name avoids treating the runtime binary itself as a
    # workspace path operand in the resource-policy shell parser.
    command = f"python3 -c {shlex.quote(script)}"
    tool = ExecTool(
        working_dir=str(home),
        resource_policy=policy,
        restrict_to_workspace=True,
    )
    scope = build_workspace_scope(project, "restricted", source_channel="websocket")
    token = bind_workspace_scope(
        scope,
        permission_owner=owner,
        task_reference=reference,
    )
    request = RequestContext(
        channel="websocket",
        chat_id="once-bwrap-chat",
        session_key="websocket:once-bwrap-chat",
        workspace=home,
        workspace_scope=scope,
        persistent_session=True,
    )
    try:
        started = time.monotonic()
        with request_context(request):
            result = await tool.execute(command=command)
        elapsed = time.monotonic() - started
        assert not isinstance(result, ToolOutcome), repr(result)
        assert "Exit code: 0" in result
        assert elapsed < 0.7, f"tool waited for detached bwrap descendant ({elapsed:.3f}s)"
        assert started_marker.read_text(encoding="utf-8") == "started"
        assert descendant_ready_marker.read_text(encoding="utf-8") == "ready"
        assert not finished_marker.exists()
        owner_tag = owner.native_owner_tag(reference)
        assert owner_tag is not None
        assert store.view.tagged_native_process_counts(owner_tag) == (1, 0)

        active, unknown = owner.retire_task_reference(
            reference,
            reason="task_cancelled",
            active_turn=False,
        )
        assert (active, unknown) == (1, 0)
        projection = owner.projection(
            session_key="websocket:once-bwrap-chat",
            source_binding=source_binding,
            task_turn_id="once-bwrap-turn",
            task_input_event_id="once-bwrap-event",
        )
        assert projection["task_access"] is not None
        assert projection["task_access"]["state"] == "retiring"
        assert projection["retirement"]["active_processes"] == 1

        for _attempt in range(100):
            projection = owner.projection(
                session_key="websocket:once-bwrap-chat",
                source_binding=source_binding,
                task_turn_id="once-bwrap-turn",
                task_input_event_id="once-bwrap-event",
            )
            if projection["task_access"] is not None and (
                projection["task_access"]["state"] == "expired"
            ):
                break
            await asyncio.sleep(0.02)
        assert projection["task_access"] is not None
        assert projection["task_access"]["state"] == "expired"
        assert projection["retirement"] is None
        assert store.view.tagged_native_process_counts(owner_tag) == (0, 0)
        assert not finished_marker.exists(), "detached bwrap namespace child outlived its root"
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_cancelled_native_session_kills_detached_child_before_activation(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux subreaper supervisor is only available on Linux")
    home, store, legacy_tool = _legacy_exec_tool(tmp_path)
    manager = ExecSessionManager()
    tool = ExecTool(
        working_dir=str(home),
        resource_policy=legacy_tool.resource_policy,
        session_manager=manager,
    )
    session_tool = ExecSessionTool(manager=manager)
    started_marker = tmp_path / "cancel-child-started"
    finished_marker = tmp_path / "cancel-child-finished"
    script = (
        "import os, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    os.setsid()\n"
        "    for fd in (0, 1, 2):\n"
        "        os.close(fd)\n"
        f"    open({str(started_marker)!r}, 'w').write('started')\n"
        "    time.sleep(10)\n"
        f"    open({str(finished_marker)!r}, 'w').write('finished')\n"
        "    os._exit(0)\n"
        "time.sleep(10)\n"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    token, request = _bind_managed_turn(home, home)
    try:
        with request:
            started = await tool.execute(command=command, yield_time_ms=0)
            match = re.search(r"session_id: ([a-f0-9]+)", started)
            assert match is not None, started
            for _attempt in range(100):
                if started_marker.exists():
                    break
                await asyncio.sleep(0.01)
            assert started_marker.exists()
            project = tmp_path / "after-cancel"
            project.mkdir()
            with pytest.raises(NativeProcessesActiveError):
                store.upsert(
                    {"path": str(project), "file_access": "read_only", "commands": False},
                    expected_revision=0,
                )

            terminated = await session_tool.execute(session_id=match.group(1), terminate=True)
            assert "Session terminated." in terminated
            assert not finished_marker.exists()

            for _attempt in range(100):
                try:
                    snapshot = store.upsert(
                        {"path": str(project), "file_access": "read_only", "commands": False},
                        expected_revision=0,
                    )
                    break
                except NativeProcessesActiveError:
                    await asyncio.sleep(0.02)
            else:
                pytest.fail("cancelled native process tree did not produce a quiescence receipt")
            assert snapshot.enabled is True
    finally:
        reset_workspace_scope(token)
        await manager.close_all()


@pytest.mark.asyncio
async def test_lost_native_supervisor_keeps_same_boot_unknown_after_child_exit(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux subreaper supervisor is only available on Linux")
    home, store, tool = _legacy_exec_tool(tmp_path)
    started_marker = tmp_path / "crash-child-started"
    finished_marker = tmp_path / "crash-child-finished"
    script = (
        "import os, time\n"
        "child = os.fork()\n"
        "if child:\n"
        "    os.write(1, b'CRASH_ROOT\\n')\n"
        "    os._exit(7)\n"
        "os.setsid()\n"
        "grandchild = os.fork()\n"
        "if grandchild:\n"
        "    os._exit(0)\n"
        "for fd in (0, 1, 2):\n"
        "    os.close(fd)\n"
        f"open({str(started_marker)!r}, 'w').write('started')\n"
        "time.sleep(0.6)\n"
        f"open({str(finished_marker)!r}, 'w').write('finished')\n"
        "os._exit(0)\n"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    token, request = _bind_managed_turn(home, home)
    try:
        with request:
            result = await tool.execute(command=command)
        assert "CRASH_ROOT" in result
        assert "Exit code: 7" in result
        for _attempt in range(100):
            if started_marker.exists():
                break
            await asyncio.sleep(0.01)
        assert started_marker.exists()

        ledger_path = store.path.with_name(".native-process-admissions.json")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        [reservation] = ledger["reservations"]
        supervisor_pid = reservation["supervisor_pid"]
        assert isinstance(supervisor_pid, int)
        os.kill(supervisor_pid, signal.SIGKILL)

        for _attempt in range(100):
            if finished_marker.exists():
                break
            await asyncio.sleep(0.02)
        assert finished_marker.read_text(encoding="utf-8") == "finished"

        project = tmp_path / "after-supervisor-loss"
        project.mkdir()
        with pytest.raises(NativeProcessesUnknownError, match="native_processes_unknown"):
            store.upsert(
                {"path": str(project), "file_access": "read_only", "commands": False},
                expected_revision=0,
            )
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_managed_exec_configured_bwrap_does_not_wrap_managed_launch_twice(tmp_path):
    if not managed_bwrap_available():
        pytest.skip("bubblewrap is unavailable on this host")
    home, project, _store, policy = _managed_tools(tmp_path)
    tool = ExecTool(
        working_dir=str(home),
        resource_policy=policy,
        restrict_to_workspace=True,
        sandbox="bwrap",
    )
    token, request = _bind_managed_turn(home, project)
    try:
        with request:
            result = await tool.execute(command="pwd")
        assert str(project.resolve()) in result
        assert "missing" not in result
    finally:
        reset_workspace_scope(token)


@pytest.mark.parametrize("excluded", [Path("/"), Path("/usr")])
def test_managed_exec_rejects_unmaskable_repository_root(tmp_path, monkeypatch, excluded):
    home, store, _tool = _legacy_exec_tool(tmp_path)
    monkeypatch.setattr("nanobot.agent.tools.sandbox.managed_bwrap_available", lambda: True)

    with pytest.raises(ValueError, match="configured repository root cannot be masked"):
        wrap_managed_command(
            "true",
            cwd=home,
            agent_home=home,
            grants=store.view,
            excluded_roots=(excluded,),
        )


@pytest.mark.asyncio
async def test_managed_exec_session_retains_logical_project_cwd(tmp_path):
    if not managed_bwrap_available():
        pytest.skip("bubblewrap is unavailable on this host")
    home, project, _store, policy = _managed_tools(tmp_path)
    manager = ExecSessionManager()
    tool = ExecTool(
        working_dir=str(home),
        resource_policy=policy,
        restrict_to_workspace=True,
        session_manager=manager,
    )
    token, request = _bind_managed_turn(home, project)
    try:
        with request:
            started = await tool.execute(
                command="sleep 5",
                yield_time_ms=0,
            )
            sessions = await manager.list(owner_session_key="websocket:fixture-chat")
        match = re.search(r"session_id: ([a-f0-9]+)", started)
        assert match is not None, started
        info = next(item for item in sessions if item.session_id == match.group(1))
        assert Path(info.cwd) == project.resolve()
    finally:
        reset_workspace_scope(token)
        await manager.close_all()


@pytest.mark.asyncio
async def test_temporary_websocket_cannot_use_persistent_external_grants(tmp_path):
    if not managed_bwrap_available():
        pytest.skip("bubblewrap is unavailable on this host")
    home, project, _store, policy = _managed_tools(tmp_path)
    external = project / "temporary.txt"
    external.write_text("PERSISTENT_ONLY", encoding="utf-8")
    read_tool = ReadFileTool(workspace=home, resource_policy=policy)
    shell_tool = ExecTool(working_dir=str(home), resource_policy=policy)
    token, request = _bind_managed_turn(home, project, persistent=False)
    try:
        with request:
            read = await read_tool.execute(path=str(external))
            command = await shell_tool.execute(command="cat temporary.txt")
        assert "PERSISTENT_ONLY" in read
        assert "PERSISTENT_ONLY" not in command
        assert "Exit code: 0" not in command
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_process_input_is_blocked_after_grant_revocation_but_poll_remains_available(tmp_path):
    if not managed_bwrap_available():
        pytest.skip("bubblewrap is unavailable on this host")
    home, project, store, policy = _managed_tools(tmp_path)
    manager = ExecSessionManager()
    shell_tool = ExecTool(
        working_dir=str(home),
        resource_policy=policy,
        session_manager=manager,
    )
    session_tool = ExecSessionTool(manager=manager)
    token, request = _bind_managed_turn(home, project)
    try:
        with request:
            started = await shell_tool.execute(command="cat", yield_time_ms=0)
            match = re.search(r"session_id: ([a-f0-9]+)", started)
            assert match is not None, started
            store.revoke(str(project), expected_revision=1)
            denied_input = await session_tool.execute(session_id=match.group(1), input="MUST_NOT_APPEAR\n")
            poll = await session_tool.execute(session_id=match.group(1), wait_for=None, timeout_ms=0)
        assert "process input is blocked by current workspace access" in denied_input
        assert "MUST_NOT_APPEAR" not in poll
    finally:
        reset_workspace_scope(token)
        await manager.close_all()
