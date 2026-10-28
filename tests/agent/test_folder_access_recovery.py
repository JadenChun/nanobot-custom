"""Tool-turn behavior for typed folder access recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.context import (
    ForkToolDependencies,
    RequestContext,
    ToolContext,
    request_context,
)
from nanobot.agent.tools.execution import KnownToolResult, execute_tool_calls
from nanobot.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.workspace import RequestFolderAccessTool
from nanobot.agent.turn import ToolOutcome
from nanobot.config.schema import ToolsConfig
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.context_repo.catalog import RepositoryCatalog
from nanobot.providers.base import ToolCallRequest
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)


class _NeedsFolderTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "fixture missing-folder operation"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        _ = kwargs
        return ToolOutcome(
            "Folder access is needed to continue this task. Choose a folder or use an allowed project.",
            "folder_access_required",
            {"folder_access": {
                "version": 1,
                "reason": "folder_not_granted",
                "operation": "read",
                "target": {"kind": "unknown", "path": None},
                "purpose": "Read the requested project",
                "allowed_roots": [],
                "roots_truncated": False,
            }},
        )


class _AfterTool(Tool):
    def __init__(self, name: str, calls: list[str]) -> None:
        self._name, self.calls = name, calls

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        _ = kwargs
        self.calls.append(self._name)
        return "executed"


@pytest.mark.asyncio
async def test_need_stops_batch_except_exact_first_party_catalog_list(tmp_path: Path) -> None:
    home = tmp_path / "agent"
    home.mkdir()
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(_NeedsFolderTool())
    catalog = RepositoryCatalog(tmp_path / "catalog", runtime_root=tmp_path / "instance")
    from nanobot.agent.tools.filesystem import ReadFileTool
    from nanobot.agent.tools.repositories import RepositoryCatalogTool

    registry.register(RepositoryCatalogTool(catalog, ReadFileTool(workspace=home)))
    registry.register(_AfterTool("write_file", calls))
    recorded: list[KnownToolResult] = []

    async def record(value: KnownToolResult) -> None:
        recorded.append(value)

    results, events = await execute_tool_calls(
        registry,
        [
            ToolCallRequest(id="need", name="read_file", arguments={}),
            ToolCallRequest(id="list", name="repository_catalog", arguments={"action": "list"}),
            ToolCallRequest(id="write", name="write_file", arguments={}),
        ],
        concurrent=False,
        external_lookup_counts={},
        workspace_violation_counts={},
        hook=AgentHook(),
        context=AgentHookContext(iteration=0, messages=[]),
        fail_on_tool_error=True,
        tool_result_callback=record,
    )

    assert isinstance(results[0], ToolOutcome)
    assert results[0].stop_reason == "folder_access_required"
    assert json.loads(results[1]) == []
    assert calls == []
    assert [item.dispatched for item in recorded] == [True, True, False]
    assert events[-1]["status"] == "skipped"


def test_request_folder_access_tool_is_exclusive_and_schema_is_narrow() -> None:
    tool = RequestFolderAccessTool(None)
    assert tool.name == "request_folder_access"
    assert tool.exclusive is True
    assert tool.concurrency_safe is False
    assert tool.parameters == {
        "type": "object",
        "required": ["purpose"],
        "additionalProperties": False,
        "properties": {
            "purpose": {"type": "string", "minLength": 1, "maxLength": 240},
            "path": {"type": "string", "minLength": 1, "maxLength": 2048},
        },
    }


@pytest.mark.asyncio
async def test_request_folder_access_without_active_owner_fails_closed() -> None:
    tool = RequestFolderAccessTool(None)
    result = await tool.execute(purpose="Locate the project")
    assert isinstance(result, ToolResult) and result.is_error


@pytest.mark.asyncio
async def test_real_file_and_exec_adapters_emit_typed_recovery(tmp_path: Path) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "selected"
    denied = tmp_path / "other"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    denied.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(project), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
        managed_webui_only=True,
    )
    scope_token = bind_workspace_scope(
        build_workspace_scope(project, "restricted", source_channel="websocket")
    )
    request = request_context(RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        workspace_scope=build_workspace_scope(project, "restricted", source_channel="websocket"),
        persistent_session=True,
    ))
    try:
        with request:
            read = await ReadFileTool(workspace=home, resource_policy=policy).execute(
                path=str(denied / "README.md")
            )
            assert isinstance(read, ToolOutcome)
            assert read.stop_reason == "folder_access_required"
            need = read.policy_metadata["folder_access"]
            assert need["reason"] == "folder_not_granted"
            assert need["target"] == {"kind": "file", "path": str(denied / "README.md")}

            command = await ExecTool(working_dir=str(home), resource_policy=policy).execute(
                command="find / -name README.md"
            )
            assert isinstance(command, ToolOutcome)
            assert command.stop_reason == "folder_access_required"
            command_need = command.policy_metadata["folder_access"]
            assert command_need["operation"] == "execute"
            assert command_need["target"] == {"kind": "directory", "path": str(project)}
            assert "find /" not in command.content
    finally:
        reset_workspace_scope(scope_token)


@pytest.mark.parametrize("repo_kind", ["context", "target"])
@pytest.mark.asyncio
async def test_configured_private_runtime_is_terminal_for_file_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_kind: str,
) -> None:
    from nanobot.config.loader import set_config_path

    instance = tmp_path / "instance"
    home = instance / "workspace"
    private_repo = instance / "sessions" / f"configured-{repo_kind}"
    home.mkdir(parents=True)
    private_repo.mkdir(parents=True)
    target_file = private_repo / "runtime-note.txt"
    target_file.write_text("PRIVATE_RUNTIME_SENTINEL", encoding="utf-8")
    config_path = instance / "config.json"
    prior_config_path = set_config_path(config_path)

    if repo_kind == "context":
        (private_repo / "nanobot.context.json").write_text(
            '{"name":"private-context","writable":["**"]}',
            encoding="utf-8",
        )
        manager = ContextRepoManager.from_config(
            context_repos=[{"path": str(private_repo)}],
        )
    else:
        manifest_root = tmp_path / "target-manifest"
        manifest_root.mkdir()
        (manifest_root / "nanobot.context.json").write_text(
            json.dumps({
                "name": "target-config",
                "targetRepos": {
                    "private-target": {
                        "path": str(private_repo),
                        "read": ["**"],
                        "write": ["**"],
                    },
                },
            }),
            encoding="utf-8",
        )
        manager = ContextRepoManager.from_config(
            context_repos=[{"path": str(manifest_root)}],
        )

    unrelated = tmp_path / "unrelated-public-project"
    unrelated.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        sessions_root=instance / "sessions",
        path=instance / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(unrelated), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=manager,
        restrict_to_workspace=True,
        folder_grants=store.view,
        managed_webui_only=True,
    )
    scope = build_workspace_scope(home, "restricted", source_channel="websocket")
    scope_token = bind_workspace_scope(scope)
    request = request_context(RequestContext(
        channel="websocket",
        chat_id="private-runtime-adapter",
        session_key="websocket:private-runtime-adapter",
        workspace=home,
        workspace_scope=scope,
        persistent_session=True,
    ))
    tools = (
        ReadFileTool(
            workspace=home,
            allowed_dir=home,
            restrict_to_workspace=True,
            resource_policy=policy,
        ),
        ListDirTool(
            workspace=home,
            allowed_dir=home,
            restrict_to_workspace=True,
            resource_policy=policy,
        ),
        WriteFileTool(
            workspace=home,
            allowed_dir=home,
            restrict_to_workspace=True,
            resource_policy=policy,
        ),
        EditFileTool(
            workspace=home,
            allowed_dir=home,
            restrict_to_workspace=True,
            resource_policy=policy,
        ),
    )
    try:
        with request:
            read = await tools[0].execute(path=str(target_file))
            listing = await tools[1].execute(path=str(private_repo))
            write = await tools[2].execute(
                path=str(target_file),
                content="changed by write_file",
            )
            edit = await tools[3].execute(
                path=str(target_file),
                old_text="PRIVATE_RUNTIME_SENTINEL",
                new_text="changed by edit_file",
            )

        for result in (read, listing, write, edit):
            assert not isinstance(result, ToolOutcome)
            assert "folder access" not in str(result).lower()
        assert "PRIVATE_RUNTIME_SENTINEL" not in str(read)
        assert "runtime-note.txt" not in str(listing)
        assert "Error" in str(write)
        assert "Error" in str(edit)
        assert target_file.read_text(encoding="utf-8") == "PRIVATE_RUNTIME_SENTINEL"
    finally:
        reset_workspace_scope(scope_token)
        set_config_path(prior_config_path)


@pytest.mark.asyncio
async def test_tool_factory_configured_containment_keeps_typed_boundary_recovery(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    denied = tmp_path / "external-project"
    instance = tmp_path / "instance"
    home.mkdir()
    denied.mkdir()
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
    )
    dependencies = cast(ForkToolDependencies, type("Deps", (), {
        "resource_policy": policy,
        "file_locks": None,
    })())
    context = ToolContext(
        config=ToolsConfig(restrict_to_workspace=True),
        workspace=str(home),
        fork_dependencies=dependencies,
    )
    read_tool = ReadFileTool.create(context)
    list_tool = ListDirTool.create(context)

    read = await read_tool.execute(path=str(denied / "README.md"))
    listing = await list_tool.execute(path=str(denied))

    assert isinstance(read, ToolOutcome)
    assert read.policy_metadata["folder_access"]["target"] == {
        "kind": "file",
        "path": str(denied / "README.md"),
    }
    assert isinstance(listing, ToolOutcome)
    assert listing.policy_metadata["folder_access"]["target"] == {
        "kind": "directory",
        "path": str(denied),
    }
