"""Phase 3 tests: per-chat sandbox_default mode without HostProcessAdmissionGuard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kohaku.agent.foreground import build_foreground_dependencies
from kohaku.agent.loop import AgentLoop
from kohaku.agent.memory import MemoryStore
from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.context import RequestContext, request_context
from kohaku.agent.tools.registry import ToolDispatchGuard, ToolRegistry
from kohaku.agent.turn import DeliveryTarget, TurnCallbacks, TurnRequest
from kohaku.bus.queue import MessageBus
from kohaku.config.schema import ToolsConfig
from kohaku.context_repo import ContextRepoManager, ResourceAccessPolicy
from kohaku.providers.base import GenerationSettings, LLMResponse
from kohaku.security.folder_grants import FolderGrantStore
from kohaku.utils.llm_runtime import LLMRuntime
from kohaku.webui.media_gateway import WebUIMediaGateway


class _DummyHostTool(Tool):
    """Minimal host-process tool to verify no HostProcessAdmissionGuard blocks it."""

    launches_host_process = True

    def __init__(self, tool_name: str = "agent_browser") -> None:
        self._name = tool_name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Dummy host tool."

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: object) -> str:
        _ = kwargs
        self.calls += 1
        return "host-ok"


def _make_agent_loop(
    tmp_path: Path,
    *,
    sandbox_default: bool,
) -> tuple[AgentLoop, MagicMock, Path, Path]:
    home = tmp_path / "agent-home"
    sessions_root = tmp_path / "instance"
    home.mkdir()
    sessions_root.mkdir()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=1024)
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="done", tool_calls=[], usage=None)
    )
    runtime = LLMRuntime.capture(provider, "test-model", context_window_tokens=120000)
    dependencies = build_foreground_dependencies(
        workspace=home,
        runtime=runtime,
        tools_config=ToolsConfig(),
        restrict_to_workspace=True,
        sessions_root=sessions_root,
    )
    dependencies.resource_policy.sandbox_default = sandbox_default
    with (
        patch("kohaku.agent.loop.SubagentManager") as mock_sub_mgr,
        patch("kohaku.agent.loop.Consolidator"),
    ):
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=home,
            model="test-model",
            fork_dependencies=dependencies,
            restrict_to_workspace=True,
        )
    return loop, provider, home, sessions_root


@pytest.mark.asyncio
async def test_brand_new_webui_chat_sandbox_default_with_zero_grants_gets_managed_prompt_and_tools(
    tmp_path: Path,
) -> None:
    loop, _provider, home, _sessions_root = _make_agent_loop(tmp_path, sandbox_default=True)
    grants = loop._fork_dependencies.resource_policy.folder_grants
    assert grants is not None
    assert grants.is_managed is False
    assert loop._fork_dependencies.resource_policy.managed_mode is True

    other_project = tmp_path / "other-project"
    other_project.mkdir()

    captured: dict[str, object] = {}
    original_build_transcript = loop.context.build_transcript

    def spy_build_transcript(*args: object, **kwargs: object):
        captured["workspace"] = kwargs.get("workspace")
        captured["tools"] = kwargs.get("tools")
        return original_build_transcript(*args, **kwargs)

    loop.context.build_transcript = spy_build_transcript  # type: ignore[method-assign]

    result = await loop.execute_turn(
        TurnRequest(
            content="hello",
            session_key="websocket:new-chat",
            sender_id="webui-user",
            metadata={"workspace_scope": {"project_path": str(other_project)}},
            route=DeliveryTarget(channel="websocket", chat_id="new-chat"),
            callbacks=TurnCallbacks(),
        ),
    )

    assert result.content == "done"
    # managed_prompt=True passes self.workspace (agent home) rather than project_path
    assert captured["workspace"] == home
    turn_tools = captured["tools"]
    assert isinstance(turn_tools, ToolRegistry)
    assert turn_tools.has("select_workspace")
    assert turn_tools.has("request_folder_access")
    assert turn_tools.dispatch_error("exec") is None


def test_media_gateway_protects_private_paths_with_sandbox_default_and_zero_grants(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent-home"
    instance = tmp_path / "instance"
    media_root = tmp_path / "media"
    home.mkdir()
    instance.mkdir()
    media_root.mkdir()
    secret_file = instance / "config.json"
    secret_file.write_text('{"secret": true}', encoding="utf-8")

    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    assert store.view.is_managed is False

    sandbox_policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
        sandbox_default=True,
    )
    sandbox_gateway = WebUIMediaGateway(
        workspace_path=home,
        logger=None,
        media_dir=lambda _channel: media_root,
        secret=b"fixture-media-secret",
        resource_policy=sandbox_policy,
    )
    assert sandbox_gateway._authorize_managed_source(secret_file, allow_folder_grants=True) is False

    full_policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=False,
        folder_grants=store.view,
        sandbox_default=False,
    )
    full_gateway = WebUIMediaGateway(
        workspace_path=home,
        logger=None,
        media_dir=lambda _channel: media_root,
        secret=b"fixture-media-secret",
        resource_policy=full_policy,
    )
    assert full_gateway._authorize_managed_source(secret_file, allow_folder_grants=True) is True


@pytest.mark.asyncio
async def test_full_access_chat_keeps_host_tools_callable_without_admission_guard(
    tmp_path: Path,
) -> None:
    loop, _provider, home, _sessions_root = _make_agent_loop(tmp_path, sandbox_default=False)
    host_tool = _DummyHostTool("dummy_host_tool")
    loop.tools.register(host_tool)

    captured_tools: list[ToolRegistry] = []
    original_build_transcript = loop.context.build_transcript

    def spy_build_transcript(*args: object, **kwargs: object):
        tools = kwargs.get("tools")
        if isinstance(tools, ToolRegistry):
            captured_tools.append(tools)
        return original_build_transcript(*args, **kwargs)

    loop.context.build_transcript = spy_build_transcript  # type: ignore[method-assign]

    result = await loop.execute_turn(
        TurnRequest(
            content="open browser",
            session_key="websocket:full-chat",
            sender_id="webui-user",
            route=DeliveryTarget(channel="websocket", chat_id="full-chat"),
            callbacks=TurnCallbacks(),
        ),
    )

    assert result.content == "done"
    assert len(captured_tools) == 1
    turn_tools = captured_tools[0]
    assert turn_tools.has("agent_browser")
    assert turn_tools.has("dummy_host_tool")
    assert turn_tools.dispatch_error("agent_browser") is None
    assert turn_tools.dispatch_error("dummy_host_tool") is None

    req = RequestContext(
        channel="websocket",
        chat_id="full-chat",
        session_key="websocket:full-chat",
        workspace=home,
        persistent_session=True,
    )
    with request_context(req):
        out = await turn_tools.execute("dummy_host_tool", {})
    assert out == "host-ok"
    assert host_tool.calls == 1


@pytest.mark.asyncio
async def test_dream_tool_dispatch_guard_still_enforces_fence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = MemoryStore(workspace)
    store.memory_file.write_text("DREAM_MEMORY_OK", encoding="utf-8")

    class _DreamFence(ToolDispatchGuard):
        def __init__(self, *, block: bool) -> None:
            self.block = block

        def error(self, name: str) -> str | None:
            if self.block and name == "read_file":
                return "Dream operation fenced"
            return None

    issued = store.build_dream_tools()
    fence = _DreamFence(block=False)
    issued.set_dispatch_guard(fence)
    derived = store.filter_dream_tools(
        issued, frozenset({"write_file", "edit_file", "apply_patch"}), workspace,
    )
    assert derived is not None
    assert "DREAM_MEMORY_OK" in await derived.execute("read_file", {"path": str(store.memory_file)})

    fence.block = True
    assert derived.dispatch_error("read_file") == "Dream operation fenced"
    denied = await derived.execute("read_file", {"path": str(store.memory_file)})
    assert "Dream operation fenced" in str(denied)
