"""Real bus ingress coverage for one-shot workspace access continuation."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from nanobot.agent.turn import DeliveryTarget, RunStatus, TurnRequest, TurnResult, TurnSource
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.runtime_composition import RuntimeComposition, RuntimeHandle
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.session.workspace_access_recovery import (
    WorkspaceAccessActionError,
    WorkspaceAccessRecoveryOwner,
)
from nanobot.webui.workspaces import WebUIWorkspaceController


class _RecordingProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(provider_name="workspace-access-ingress-test")
        self.requests: list[list[dict[str, object]]] = []
        self.called = asyncio.Event()
        self.gated_call_number: int | None = None
        self.gate_started = asyncio.Event()
        self.call_release = asyncio.Event()
        self.call_release.set()

    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        _ = tools, model, kwargs
        self.requests.append(messages)
        self.called.set()
        if self.gated_call_number == len(self.requests):
            self.gate_started.set()
            await self.call_release.wait()
        return LLMResponse(content="continued from the selected folder")

    def estimate_prompt_tokens(self, messages, tools=None, model=None):
        _ = tools, model
        return len(messages) + 1, "workspace-access-test"

    def get_default_model(self) -> str:
        return "workspace-access-ingress-test"


class _ProtectedWriteProvider(_RecordingProvider):
    def __init__(self, target: Path) -> None:
        super().__init__()
        self.target = target

    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        _ = tools, model, kwargs
        self.requests.append(messages)
        self.called.set()
        if len(self.requests) == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(
                    id="protected-write",
                    name="write_file",
                    arguments={"path": str(self.target), "content": "must stay blocked"},
                )],
            )
        return LLMResponse(content="The configured file remains protected.")


class _RepeatedRecoveryProvider(_RecordingProvider):
    """Drive two independently blocked project writes through canonical turns."""

    def __init__(self, first_project: Path, second_project: Path) -> None:
        super().__init__()
        self.first_project = first_project
        self.second_project = second_project

    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        _ = tools, model, kwargs
        self.requests.append(messages)
        self.called.set()
        latest_user = next(
            (row.get("content") for row in reversed(messages) if row.get("role") == "user"),
            None,
        )
        if latest_user == "Please start a fresh task now.":
            return LLMResponse(content="Fresh task completed.")
        call_number = len(self.requests)
        if call_number in {1, 2}:
            path = self.first_project / "README.md"
        elif call_number in {3, 4}:
            path = self.second_project / "README.md"
        else:
            return LLMResponse(content="Finished both project writes.")
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id=f"read-project-{call_number}",
                name="write_file",
                arguments={"path": str(path), "content": "first project marker" if call_number in {1, 2} else "second project marker"},
            )],
        )


def _config(tmp_path: Path) -> Config:
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    workspace.mkdir()
    runtime.mkdir()
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "workspace-access-ingress-test"
    config.agents.defaults.provider = "custom"
    config.agents.defaults.session_ttl_minutes = 0
    config.agents.defaults.idle_compaction.enabled = False
    config.agents.defaults.dream.enabled = False
    config.gateway.heartbeat.enabled = False
    config.tools.restrict_to_workspace = True
    config.tools.mcp_servers = {}
    config.artifacts.enabled = False
    config.channels = type(config.channels)()
    config.providers.custom.api_base = "https://provider.invalid/v1"
    config.providers.custom.api_key = "synthetic-test-key"
    config.bind_source_path(runtime / "config.json")
    return config


@pytest.mark.asyncio
async def test_canonical_turn_finalizes_terminal_configured_write_without_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.agents.defaults.context_repos = []
    target = tmp_path / "configured" / "config" / "private.txt"
    target.parent.mkdir(parents=True)
    repo = target.parents[1]
    (repo / "nanobot.context.json").write_text(
        '{"name":"configured","protected":["config/**"]}',
        encoding="utf-8",
    )
    config.agents.defaults.context_repos = [str(repo)]
    provider = _ProtectedWriteProvider(target)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    sessions_root = tmp_path / "sessions"
    grants = FolderGrantStore.for_workspace(
        config.workspace_path,
        sessions_root=sessions_root,
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    grants.upsert(
        {"path": str(unrelated), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    composition = RuntimeComposition.define(
        config,
        sessions_root=sessions_root,
        startup_receipt_admitted=True,
    )
    handle = await composition.acquire()
    key = "websocket:terminal-policy"
    try:
        assert handle.agent._fork_dependencies.resource_policy.folder_grants.is_managed
        result = await handle.agent.execute_turn(TurnRequest(
            content="Edit the configured private file.",
            session_key=key,
            source=TurnSource.GATEWAY,
            route=DeliveryTarget(channel="websocket", chat_id="terminal-policy"),
            metadata={"webui": True, "webui_turn_id": "terminal-policy-turn"},
        ))

        assert result.status is RunStatus.TOOL_ERROR
        assert "protected by context repo policy" in result.content
        assert len(provider.requests) == 1
        assert not target.exists()
        backend = handle.sessions.source_backend(key)
        assert backend is not None
        snapshot = backend.source.load_source_snapshot(key)
        assert "workspace_access_request" not in snapshot.header["metadata"]
        assert any(
            "protected by context repo policy" in str(event.get("content", ""))
            for event in snapshot.events
            if event.get("role") == "tool"
        )
    finally:
        await composition.aclose()


@dataclass(slots=True)
class _PreparedContinuation:
    provider: _RecordingProvider
    initial_result: TurnResult
    config: Config
    project: Path
    composition: RuntimeComposition
    handle: RuntimeHandle
    recovery: WorkspaceAccessRecoveryOwner
    chat_id: str
    key: str
    original_task: str
    access_request_id: str
    ready_revision: int


def _track_continuation_owner_cleanup(
    request: pytest.FixtureRequest,
    bus,
    chat_id: str,
) -> None:
    """Release only queued owners whose messages this test publishes."""
    from nanobot.session.webui_turns import clear_websocket_turn_if_current
    from nanobot.webui.metadata import WEBSOCKET_TURN_OWNER_METADATA_KEY

    owners: set[str] = set()
    publish_inbound = bus.publish_inbound

    async def publish_and_track(message) -> None:
        if message.channel == "websocket" and message.chat_id == chat_id:
            owner = message.metadata.get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
            if isinstance(owner, str):
                owners.add(owner)
        await publish_inbound(message)

    bus.publish_inbound = publish_and_track

    def cleanup() -> None:
        for owner in owners:
            clear_websocket_turn_if_current(chat_id, owner)

    request.addfinalizer(cleanup)


async def _prepare_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> _PreparedContinuation:
    provider = _RecordingProvider()
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    config = _config(tmp_path)
    selected_project = tmp_path / "selected-project"
    selected_project.mkdir()
    (selected_project / "README.md").write_text(
        "current project marker",
        encoding="utf-8",
    )
    composition = RuntimeComposition.define(
        config,
        sessions_root=tmp_path / "sessions",
        startup_receipt_admitted=True,
    )
    handle = await composition.acquire()
    chat_id = "continue-ingress"
    key = f"websocket:{chat_id}"
    original_task = "Please inspect the project inventory and README."
    initial = await handle.agent.execute_turn(TurnRequest(
        content=original_task,
        session_key=key,
        source=TurnSource.GATEWAY,
        route=DeliveryTarget(channel="websocket", chat_id=chat_id),
        metadata={"webui": True, "webui_turn_id": "original-turn"},
    ))
    assert initial.content == "continued from the selected folder"
    backend = handle.sessions.source_backend(key)
    assert backend is not None
    grants = FolderGrantStore.for_workspace(
        config.workspace_path,
        sessions_root=handle.sessions.sessions_root,
    )
    grants.upsert(
        {
            "path": str(selected_project),
            "file_access": "read_only",
            "commands": False,
        },
        expected_revision=grants.snapshot().revision,
    )
    workspaces = WebUIWorkspaceController(
        session_manager=handle.sessions,
        default_workspace=config.workspace_path,
        default_restrict_to_workspace=True,
        grant_store=grants,
    )
    scope = workspaces.scope_for_session_key(key)
    expected_scope = {
        "project_path": str(scope.project_path),
        "access_mode": scope.access_mode,
    }
    workspaces.persist_scope(chat_id, scope)
    source = backend.source.load_source_snapshot(key)
    task_event = next(
        event
        for event in reversed(source.events)
        if event.get("role") == "user" and event.get("content") == original_task
    )
    input_id = task_event["_event_id"]
    binding = {
        "store_identity": source.state.store_identity,
        "incarnation": source.state.incarnation,
        "generation": source.state.generation,
    }
    record = {
        "version": 1,
        "access_request_id": "a" * 32,
        "revision": 1,
        "task": {"turn_id": "original-turn"},
        "blocked_turn_id": "original-turn",
        "tool_call_id": None,
        "created_at_ms": 1,
        "status": "pending",
        "reason": "folder_selection_required",
        "operation": "locate",
        "target": {"kind": "unknown", "path": None},
        "purpose": "Locate the project requested in the original task",
        "selected_path": None,
        "continuation_turn_id": None,
        "source_binding": binding,
        "task_input_event_id": input_id,
        "latest_input_event_id": input_id,
        "blocked_run_id": task_event["_run_id"],
        "expected_scope": expected_scope,
        "resolved_grants_revision": None,
    }
    handle.sessions.commit_workspace_access_request(
        key,
        expected_binding=binding,
        expected_record=None,
        replacement_record=record,
        expected_latest_input_event_id=input_id,
    )
    recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
    _track_continuation_owner_cleanup(request, handle.bus, chat_id)
    ready = recovery.resolve({
        "chat_id": chat_id,
        "access_request_id": record["access_request_id"],
        "expected_revision": 1,
        "path": str(selected_project),
        "expected_grants_revision": grants.snapshot().revision,
    })
    return _PreparedContinuation(
        provider=provider,
        initial_result=initial,
        config=config,
        project=selected_project,
        composition=composition,
        handle=handle,
        recovery=recovery,
        chat_id=chat_id,
        key=key,
        original_task=original_task,
        access_request_id=record["access_request_id"],
        ready_revision=ready["workspace_access_request"]["revision"],
    )


@pytest.mark.parametrize("duration", ["always", "once"])
@pytest.mark.asyncio
async def test_continue_survives_cache_invalidation_and_runs_once_on_real_bus(
    duration: str,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    provider = _RecordingProvider()
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    config = _config(tmp_path)
    selected_project = tmp_path / "selected-project"
    selected_project.mkdir()
    (selected_project / "README.md").write_text(
        "current project marker",
        encoding="utf-8",
    )
    composition = RuntimeComposition.define(
        config,
        sessions_root=tmp_path / "sessions",
        startup_receipt_admitted=True,
    )
    handle = await composition.acquire()
    chat_id = "continue-ingress"
    key = f"websocket:{chat_id}"
    original_task = "Please inspect the project inventory and README."
    run_task: asyncio.Task[None] | None = None
    try:
        initial = await handle.agent.execute_turn(TurnRequest(
            content=original_task,
            session_key=key,
            source=TurnSource.GATEWAY,
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
        ))
        assert initial.content == "continued from the selected folder"
        assert len(provider.requests) == 1
        backend = handle.sessions.source_backend(key)
        assert backend is not None
        grants = FolderGrantStore.for_workspace(
            config.workspace_path,
            sessions_root=handle.sessions.sessions_root,
        )
        if duration == "always":
            grants.upsert(
                {
                    "path": str(selected_project),
                    "file_access": "read_only",
                    "commands": False,
                },
                expected_revision=grants.snapshot().revision,
            )
        workspaces = WebUIWorkspaceController(
            session_manager=handle.sessions,
            default_workspace=config.workspace_path,
            default_restrict_to_workspace=True,
            grant_store=grants,
        )
        scope = workspaces.scope_for_session_key(key)
        expected_scope = {
            "project_path": str(scope.project_path),
            "access_mode": scope.access_mode,
        }
        workspaces.persist_scope(chat_id, scope)
        source = backend.source.load_source_snapshot(key)
        task_event = next(
            event
            for event in reversed(source.events)
            if event.get("role") == "user" and event.get("content") == original_task
        )
        input_id = task_event["_event_id"]
        binding = {
            "store_identity": source.state.store_identity,
            "incarnation": source.state.incarnation,
            "generation": source.state.generation,
        }
        record = {
            "version": 1,
            "access_request_id": "a" * 32,
            "revision": 1,
            "task": {"turn_id": "original-turn"},
            "blocked_turn_id": "original-turn",
            "tool_call_id": None,
            "created_at_ms": 1,
            "status": "pending",
            "reason": "folder_selection_required",
            "operation": "locate",
            "target": {"kind": "unknown", "path": None},
            "purpose": "Locate the project requested in the original task",
            "selected_path": None,
            "continuation_turn_id": None,
            "source_binding": binding,
            "task_input_event_id": input_id,
            "latest_input_event_id": input_id,
            "blocked_run_id": task_event["_run_id"],
            "expected_scope": expected_scope,
            "resolved_grants_revision": None,
        }
        handle.sessions.commit_workspace_access_request(
            key,
            expected_binding=binding,
            expected_record=None,
            replacement_record=record,
            expected_latest_input_event_id=input_id,
        )
        recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
        _track_continuation_owner_cleanup(request, handle.bus, chat_id)
        current_permissions = recovery.workspace_permissions(
            chat_id,
            controls_available=True,
        )
        if duration == "once":
            ready = recovery.allow_once({
                "chat_id": chat_id,
                "access_request_id": record["access_request_id"],
                "expected_revision": 1,
                "path": str(selected_project),
                "file_access": "read_only",
                "commands": False,
                "expected_grants_revision": grants.snapshot().revision,
                "expected_runtime_id": current_permissions["runtime_id"],
                "expected_permission_revision": current_permissions["revision"],
            })
        else:
            ready = recovery.resolve({
                "chat_id": chat_id,
                "access_request_id": record["access_request_id"],
                "expected_revision": 1,
                "path": str(selected_project),
                "expected_grants_revision": grants.snapshot().revision,
            })
        assert ready["workspace_access_request"]["selected_duration"] == duration
        run_task = asyncio.create_task(handle.agent.run())
        provider.called.clear()
        continued = await recovery.continue_request({
            "chat_id": chat_id,
            "access_request_id": record["access_request_id"],
            "expected_revision": ready["workspace_access_request"]["revision"],
        })

        assert continued["workspace_access_request"]["status"] == "continuing"
        if duration == "once":
            acknowledged_request = continued["workspace_access_request"]
            assert acknowledged_request["selected_duration"] == "once"
            assert acknowledged_request["task_access_state"] == "active"
            assert acknowledged_request["selected_access"]["source"] == "task"
            assert acknowledged_request["selected_access"]["read"] is True
            assert not acknowledged_request["can_continue"]
            assert acknowledged_request["unavailable_reason"] == "continuation_unknown"
            assert continued["workspace_scope"]["effective_access"]["source"] == "task"
            assert continued["workspace_scope"]["effective_access"]["read"] is True
        assert handle.sessions.get_cached(key) is None
        await asyncio.wait_for(provider.called.wait(), timeout=3)
        assert len(provider.requests) == 2
        visible_inputs = [
            str(row.get("content"))
            for row in provider.requests[1]
            if row.get("role") == "user"
        ]
        assert visible_inputs.count(original_task) == 1
        assert visible_inputs.count("Continue task") == 1
        assert str(selected_project) in repr(provider.requests[1])

        async def wait_for_terminal_task() -> None:
            while True:
                snapshot = backend.source.load_source_snapshot(key)
                matching_responses = [
                    event for event in snapshot.events
                    if event.get("role") == "assistant"
                    and event.get("content") == "continued from the selected folder"
                ]
                if len(matching_responses) >= 2 and (
                    snapshot.header["metadata"].get("execution_lease") is None
                ):
                    return
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_terminal_task(), timeout=5)

        latest = backend.source.load_source_snapshot(key)
        access_record = latest.header["metadata"]["workspace_access_request"]
        assert access_record["status"] == "continued"
        assert [event["content"] for event in latest.events if event.get("role") == "user"].count(
            "Continue task"
        ) == 1
        once_reference = handle.sessions.desktop_permissions.task_reference(
            session_key=key,
            source_binding=binding,
            task_turn_id="original-turn",
            task_input_event_id=input_id,
        )
        assert once_reference is None
        expired = recovery.workspace_permissions(chat_id, controls_available=True)
        if duration == "once":
            assert expired["task_access"]["state"] == "expired"
            final_grants = grants.snapshot()
            assert final_grants.revision == 0
            assert final_grants.grants == ()
            completed_request = recovery.get(chat_id)
            assert completed_request is not None
            assert completed_request["selected_duration"] == "once"
            assert completed_request["task_access_state"] == "expired"
            assert completed_request["selected_access"] is None
            assert completed_request["unavailable_reason"] == "task_access_expired"
        else:
            assert expired["task_access"] is None
    finally:
        await composition.aclose()
        if run_task is not None and not run_task.done():
            await handle.bus.publish_inbound(InboundMessage(
                channel="system", sender_id="test", chat_id="", content="shutdown wake",
            ))
        if run_task is not None:
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=5)


@pytest.mark.parametrize("supersede_second_continue", [False, True])
@pytest.mark.asyncio
async def test_repeated_folder_recovery_continues_original_task_on_real_bus(
    supersede_second_continue: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    (project_a / "README.md").write_text("first project marker", encoding="utf-8")
    (project_b / "README.md").write_text("second project marker", encoding="utf-8")
    provider = _RepeatedRecoveryProvider(project_a, project_b)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    config = _config(tmp_path)
    composition = RuntimeComposition.define(
        config,
        sessions_root=tmp_path / "sessions",
        startup_receipt_admitted=True,
    )
    handle = await composition.acquire()
    chat_id = "repeat-continue"
    key = f"websocket:{chat_id}"
    original_task = "Write notes to both project README files."
    grants = FolderGrantStore.for_workspace(
        config.workspace_path,
        sessions_root=handle.sessions.sessions_root,
    )
    workspaces = WebUIWorkspaceController(
        session_manager=handle.sessions,
        default_workspace=config.workspace_path,
        default_restrict_to_workspace=True,
        grant_store=grants,
    )
    workspaces.persist_scope(chat_id, workspaces.default_scope())
    recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
    from nanobot.channels.websocket.runtime import WebSocketChannel
    from nanobot.webui.gateway_services import GatewayServices

    run_task: asyncio.Task[None] | None = None
    outbound_task: asyncio.Task[None] | None = None
    disconnect_webui: Callable[[], None] | None = None
    prior_config_path: Path | None = None
    gateway: GatewayServices | None = None
    channel: WebSocketChannel | None = None
    dispatched: list[tuple[str, bool]] = []
    continuation_messages: list[InboundMessage] = []
    delivered_outbound = []
    owned_turn_owners: set[str] = set()
    duplicate_dispatch_completed = asyncio.Event()
    duplicate_request_id: str | None = None
    duplicate_dispatch_count = 0

    async def wait_for_request(previous_id: str | None = None):
        async def poll():
            while True:
                try:
                    current = recovery.get(chat_id)
                except WorkspaceAccessActionError as exc:
                    if exc.status != 404:
                        raise
                    current = None
                backend = handle.sessions.source_backend(key)
                idle = False
                if current is not None and backend is not None:
                    state = backend.source.load_source_snapshot(key)
                    idle = state.header["metadata"].get("execution_lease") is None
                if (
                    current is not None
                    and current["status"] == "pending"
                    and current["access_request_id"] != previous_id
                    and idle
                ):
                    return current
                await asyncio.sleep(0.01)

        return await asyncio.wait_for(poll(), timeout=5)

    async def continue_into(path: Path, request: dict[str, object]):
        grants.upsert(
            {"path": str(path), "file_access": "read_write", "commands": False},
            expected_revision=grants.snapshot().revision,
        )
        runtime_focus = handle.agent._workspace_focus_owner()
        assert runtime_focus is not None
        assert runtime_focus.grants._path == grants.path
        assert runtime_focus.grants.snapshot() == grants.snapshot()
        assert runtime_focus.effective_access(path)["read"] is True
        ready = recovery.resolve({
            "chat_id": chat_id,
            "access_request_id": request["access_request_id"],
            "expected_revision": request["revision"],
            "path": str(path),
            "expected_grants_revision": grants.snapshot().revision,
        })
        assert ready["workspace_access_request"]["status"] == "ready"
        return await recovery.continue_request({
            "chat_id": chat_id,
            "access_request_id": request["access_request_id"],
            "expected_revision": ready["workspace_access_request"]["revision"],
        })

    def read_thread() -> dict[str, object]:
        assert gateway is not None
        from websockets.datastructures import Headers
        from websockets.http11 import Request

        encoded_key = quote(key, safe="")
        request = Request(
            f"/api/sessions/{encoded_key}/webui-thread",
            Headers([("Authorization", "Bearer lifecycle-test-token")]),
        )
        response = gateway.http._handle_webui_thread_get(request, encoded_key)
        assert response.status_code == 200
        body = json.loads(response.body.decode())
        assert isinstance(body, dict)
        return body

    async def wait_for_thread_completion(turn_id: str) -> dict[str, object]:
        async def poll() -> dict[str, object]:
            while True:
                body = read_thread()
                completed_turn_ids = body.get("completed_turn_ids")
                if (
                    isinstance(completed_turn_ids, list)
                    and turn_id in completed_turn_ids
                    and body.get("active_turn_id") is None
                ):
                    return body
                await asyncio.sleep(0.01)

        return await asyncio.wait_for(poll(), timeout=5)

    async def wait_for_lifecycle_idle(turn_id: str) -> None:
        from nanobot.bus.outbound_events import GoalStatusEvent
        from nanobot.webui.metadata import WEBUI_TURN_METADATA_KEY

        async def poll() -> None:
            while not any(
                message.metadata.get(WEBUI_TURN_METADATA_KEY) == turn_id
                and isinstance(message.event, GoalStatusEvent)
                and message.event.status == "idle"
                for message in delivered_outbound
            ):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(poll(), timeout=5)

    async def deliver_outbound() -> None:
        assert channel is not None
        from nanobot.webui.metadata import WEBSOCKET_TURN_OWNER_METADATA_KEY

        while True:
            outbound = await handle.bus.consume_outbound()
            delivered_outbound.append(outbound)
            owner = outbound.metadata.get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
            if isinstance(owner, str):
                owned_turn_owners.add(owner)
            await channel.send(outbound)

    try:
        from dataclasses import replace

        from nanobot.channels.websocket.runtime import WebSocketChannel, WebSocketConfig
        from nanobot.config.loader import get_config_path, set_config_path
        from nanobot.session.webui_turns import WebuiTurnCoordinator
        from nanobot.webui.gateway_services import build_gateway_services
        from nanobot.webui.metadata import (
            WEBSOCKET_TURN_OWNER_METADATA_KEY,
            WEBUI_TURN_METADATA_KEY,
        )

        prior_config_path = get_config_path()
        runtime_data_dir = config.runtime_data_dir
        assert runtime_data_dir is not None
        config_path = runtime_data_dir / "config.json"
        set_config_path(config_path)
        websocket_config = WebSocketConfig(enabled=True, allow_from=["*"])
        gateway = build_gateway_services(
            config=websocket_config,
            bus=handle.bus,
            session_manager=handle.sessions,
            static_dist_path=None,
            workspace_path=config.workspace_path,
            default_restrict_to_workspace=True,
            config_path=config_path,
            runtime_model_name=lambda: None,
            runtime_surface="test",
            runtime_capabilities_overrides=None,
        )
        gateway.http.tokens.api_tokens["lifecycle-test-token"] = time.monotonic() + 300
        channel = WebSocketChannel(websocket_config, handle.bus, gateway=gateway)
        coordinator = WebuiTurnCoordinator(
            bus=handle.bus,
            sessions=handle.sessions,
            schedule_background=lambda coroutine: coroutine.close(),
        )
        disconnect_webui = coordinator.subscribe()
        outbound_task = asyncio.create_task(deliver_outbound())

        from nanobot.session.webui_turns import register_queued_websocket_turn_if_idle
        from nanobot.session.workspace_access_continuation import (
            copy_workspace_access_continuation,
            workspace_access_continuation,
        )

        original_enqueue = handle.agent._enqueue_bus_dispatch

        def capture_dispatch(message, session_key: str):
            nonlocal duplicate_dispatch_count
            reference = workspace_access_continuation(message, session_key=session_key)
            dispatched.append((
                message.content,
                reference is not None,
            ))
            if reference is not None:
                continuation_messages.append(message)
                owner = message.metadata.get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
                if isinstance(owner, str):
                    owned_turn_owners.add(owner)
            task = original_enqueue(message, session_key)
            if reference is not None and reference.access_request_id == duplicate_request_id:
                duplicate_dispatch_count += 1
                if duplicate_dispatch_count == 2:
                    task.add_done_callback(lambda _task: duplicate_dispatch_completed.set())
            return task

        monkeypatch.setattr(handle.agent, "_enqueue_bus_dispatch", capture_dispatch)
        run_task = asyncio.create_task(handle.agent.run())
        original_turn_id = "original-user-turn"
        original_owner = register_queued_websocket_turn_if_idle(chat_id, original_turn_id)
        assert original_owner is not None
        owned_turn_owners.add(original_owner)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket",
            sender_id="webui",
            chat_id=chat_id,
            content=original_task,
            metadata={
                "webui": True,
                "_wants_stream": True,
                WEBUI_TURN_METADATA_KEY: original_turn_id,
                WEBSOCKET_TURN_OWNER_METADATA_KEY: original_owner,
            },
        ))
        try:
            first_request = await wait_for_request()
        except TimeoutError:
            metadata = handle.sessions.read_session_metadata(key)
            backend = handle.sessions.source_backend(key)
            source_header = (
                backend.source.load_source_snapshot(key).header
                if backend is not None else None
            )
            pytest.fail(
                f"initial request timed out; metadata={metadata!r} "
                f"source_header={source_header!r} calls={len(provider.requests)}",
                pytrace=False,
            )
        assert first_request["task"]["excerpt"] == original_task
        initial_snapshot = handle.sessions.source_backend(key).source.load_source_snapshot(key)
        initial_record = initial_snapshot.header["metadata"]["workspace_access_request"]
        original_event = next(
            event for event in initial_snapshot.events
            if event.get("_event_id") == initial_record["task_input_event_id"]
        )
        assert original_event.get("_run_id") == initial_record["blocked_run_id"]
        first_continue = await continue_into(project_a, first_request)
        assert first_continue["workspace_access_request"]["status"] == "continuing"

        try:
            second_request = await wait_for_request(first_request["access_request_id"])
        except TimeoutError:
            pytest.fail(
                f"second request timed out; calls={len(provider.requests)} "
                f"dispatches={dispatched!r} queue={handle.bus.inbound_size} "
                f"state={recovery.get(chat_id)!r}",
                pytrace=False,
            )
        assert second_request["task"] == first_request["task"]
        first_turn_id = first_continue["workspace_access_request"]["continuation_turn_id"]
        assert isinstance(first_turn_id, str)
        first_thread = await wait_for_thread_completion(first_turn_id)
        await wait_for_lifecycle_idle(first_turn_id)
        assert first_turn_id in first_thread["completed_turn_ids"]
        assert first_thread["active_turn_id"] is None
        assert second_request["blocked_turn_id"] == first_continue[
            "workspace_access_request"
        ]["continuation_turn_id"]
        snapshot = handle.sessions.source_backend(key).source.load_source_snapshot(key)
        source_events = snapshot.events
        original_event = next(
            event for event in source_events
            if event.get("role") == "user" and event.get("content") == original_task
        )
        latest_event = next(
            event for event in reversed(source_events) if event.get("role") == "user"
        )
        assert latest_event.get("content") == "Continue task"
        assert original_event.get("_run_id") != latest_event.get("_run_id")
        assert len(provider.requests) == 3

        duplicate_request_id = str(second_request["access_request_id"])
        if supersede_second_continue:
            held_continue: list[InboundMessage] = []
            held_event = asyncio.Event()
            downstream_enqueue = handle.agent._enqueue_bus_dispatch

            def hold_second_continue(message, session_key: str):
                reference = workspace_access_continuation(message, session_key=session_key)
                if reference is not None and reference.access_request_id == duplicate_request_id:
                    held_continue.append(message)
                    held_event.set()
                    return asyncio.create_task(asyncio.sleep(0))
                return downstream_enqueue(message, session_key)

            monkeypatch.setattr(handle.agent, "_enqueue_bus_dispatch", hold_second_continue)
            second_continue = await continue_into(project_b, second_request)
            assert second_continue["workspace_access_request"]["status"] == "continuing"
            await asyncio.wait_for(held_event.wait(), timeout=5)

            fresh_input = "Please start a fresh task now."
            await handle.bus.publish_inbound(InboundMessage(
                channel="websocket",
                sender_id="webui",
                chat_id=chat_id,
                content=fresh_input,
                metadata={"webui": True, "webui_turn_id": "new-user-turn"},
            ))

            async def wait_for_fresh_completion():
                while True:
                    current = handle.sessions.source_backend(key).source.load_source_snapshot(key)
                    record = current.header["metadata"]["workspace_access_request"]
                    if (
                        record["status"] == "invalidated"
                        and any(
                            event.get("role") == "assistant"
                            and event.get("content") == "Fresh task completed."
                            for event in current.events
                        )
                    ):
                        return current
                    await asyncio.sleep(0.01)

            superseded = await asyncio.wait_for(wait_for_fresh_completion(), timeout=5)
            assert len(held_continue) == 1
            stale_message = held_continue[0]
            duplicate = replace(stale_message, metadata=dict(stale_message.metadata))
            copy_workspace_access_continuation(stale_message, duplicate)
            monkeypatch.setattr(handle.agent, "_enqueue_bus_dispatch", downstream_enqueue)
            await handle.bus.publish_inbound(stale_message)
            await handle.bus.publish_inbound(duplicate)
            await asyncio.wait_for(duplicate_dispatch_completed.wait(), timeout=5)

            final_snapshot = handle.sessions.source_backend(key).source.load_source_snapshot(key)
            assert final_snapshot.header["metadata"]["workspace_access_request"]["status"] == (
                "invalidated"
            )
            assert len(provider.requests) == 4
            assert any(
                event.get("role") == "assistant"
                and event.get("content") == "Fresh task completed."
                for event in superseded.events
            )
            assert sum(
                event.get("role") == "user" and event.get("content") == "Continue task"
                for event in final_snapshot.events
            ) == 1
            assert any(
                event.get("role") == "user" and event.get("content") == fresh_input
                for event in final_snapshot.events
            )
            return

        second_continue = await continue_into(project_b, second_request)
        assert second_continue["workspace_access_request"]["status"] == "continuing"

        async def wait_for_completion():
            while len(provider.requests) < 5:
                await asyncio.sleep(0.01)
            while True:
                current = handle.sessions.source_backend(key).source.load_source_snapshot(key)
                if any(
                    event.get("role") == "assistant"
                    and event.get("content") == "Finished both project writes."
                    for event in current.events
                ):
                    return current
                await asyncio.sleep(0.01)

        completed = await asyncio.wait_for(wait_for_completion(), timeout=5)
        second_turn_id = second_continue["workspace_access_request"]["continuation_turn_id"]
        assert isinstance(second_turn_id, str)
        assert second_turn_id != first_turn_id
        final_thread = await wait_for_thread_completion(second_turn_id)
        await wait_for_lifecycle_idle(second_turn_id)
        assert second_turn_id in final_thread["completed_turn_ids"]
        assert final_thread["active_turn_id"] is None

        from nanobot.bus.outbound_events import GoalStatusEvent, TurnEndEvent

        for turn_id in (first_turn_id, second_turn_id):
            lifecycle = [
                message
                for message in delivered_outbound
                if message.metadata.get(WEBUI_TURN_METADATA_KEY) == turn_id
                and isinstance(message.event, GoalStatusEvent | TurnEndEvent)
            ]
            statuses = [
                message.event.status
                for message in lifecycle
                if isinstance(message.event, GoalStatusEvent)
            ]
            owners = {
                message.metadata.get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
                for message in lifecycle
            }
            assert statuses == ["running", "idle"]
            assert sum(isinstance(message.event, TurnEndEvent) for message in lifecycle) == 1
            assert len(owners) == 1
            owner = owners.pop()
            assert isinstance(owner, str)
            assert owner in owned_turn_owners

        final_record = completed.header["metadata"]["workspace_access_request"]
        assert final_record["status"] == "continued"
        assert final_record["task_input_event_id"] == original_event["_event_id"]
        user_inputs = [
            str(message.get("content"))
            for request_messages in provider.requests
            for message in request_messages
            if message.get("role") == "user"
        ]
        assert user_inputs.count(original_task) >= 4
        assert user_inputs.count("Continue task") >= 4
        assert str(project_b) in repr(provider.requests[3])
        assert "second project marker" in repr(provider.requests[4])
        assert sum(
            event.get("role") == "user" and event.get("content") == "Continue task"
            for event in completed.events
        ) == 2

        assert len(continuation_messages) == 2
        duplicate = replace(
            continuation_messages[1],
            metadata=dict(continuation_messages[1].metadata),
        )
        copy_workspace_access_continuation(continuation_messages[1], duplicate)
        await handle.bus.publish_inbound(duplicate)
        await asyncio.wait_for(duplicate_dispatch_completed.wait(), timeout=5)
        assert len(provider.requests) == 5
        replayed_backend = handle.sessions.source_backend(key)
        assert replayed_backend is not None
        replayed = replayed_backend.source.load_source_snapshot(key)
        replayed_record = replayed.header["metadata"]["workspace_access_request"]
        assert replayed_record["status"] == "continued"
        assert sum(
            event.get("role") == "user" and event.get("content") == "Continue task"
            for event in replayed.events
        ) == 2
    finally:
        if disconnect_webui is not None:
            disconnect_webui()
        if outbound_task is not None:
            outbound_task.cancel()
            await asyncio.gather(outbound_task, return_exceptions=True)
        handle.agent.stop()
        if run_task is not None and not run_task.done():
            run_task.cancel()
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=5)
        from nanobot.config.loader import set_config_path
        from nanobot.session.webui_turns import clear_websocket_turn_if_current

        for owner in owned_turn_owners:
            clear_websocket_turn_if_current(chat_id, owner)
        if prior_config_path is not None:
            set_config_path(prior_config_path)
        await composition.aclose()


@pytest.mark.parametrize("mutation", ["reset", "delete", "recreate"])
@pytest.mark.asyncio
async def test_queued_continue_rejects_changed_canonical_source_without_recreation(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    prepared = await _prepare_continuation(monkeypatch, tmp_path, request)
    handle = prepared.handle
    path = handle.sessions._get_session_path(prepared.key)
    admission_checked = asyncio.Event()
    dispatch_done = asyncio.Event()
    run_task: asyncio.Task[None] | None = None
    try:
        result = await prepared.recovery.continue_request({
            "chat_id": prepared.chat_id,
            "access_request_id": prepared.access_request_id,
            "expected_revision": prepared.ready_revision,
        })
        assert result["workspace_access_request"]["status"] == "continuing"
        assert handle.bus.inbound.qsize() == 1

        if mutation in {"delete", "recreate"}:
            assert handle.sessions.delete_session(prepared.key)
        if mutation in {"reset", "recreate"}:
            handle.sessions.reset_session(prepared.key)

        if mutation == "delete":
            assert not path.exists()
            replacement_bytes = None
        else:
            assert path.is_file()
            replacement_bytes = path.read_bytes()

        original_read = handle.sessions.read_existing_canonical_source

        def track_admission(key: str, *, expected_binding=None):
            try:
                return original_read(key, expected_binding=expected_binding)
            finally:
                if expected_binding is not None:
                    admission_checked.set()

        monkeypatch.setattr(handle.sessions, "read_existing_canonical_source", track_admission)
        original_enqueue = handle.agent._enqueue_bus_dispatch

        def track_dispatch(message, key: str):
            task = original_enqueue(message, key)
            task.add_done_callback(lambda _task: dispatch_done.set())
            return task

        monkeypatch.setattr(handle.agent, "_enqueue_bus_dispatch", track_dispatch)
        run_task = asyncio.create_task(handle.agent.run())
        await asyncio.wait_for(admission_checked.wait(), timeout=5)
        await asyncio.wait_for(dispatch_done.wait(), timeout=5)

        assert len(prepared.provider.requests) == 1
        from nanobot.session.webui_turns import (
            websocket_turn_id,
            websocket_turn_wall_started_at,
        )

        assert websocket_turn_id(prepared.chat_id) is None
        assert websocket_turn_wall_started_at(prepared.chat_id) is None
        assert path.exists() is (mutation != "delete")
        if replacement_bytes is not None:
            assert path.read_bytes() == replacement_bytes
            snapshot = handle.sessions._jsonl_store.source.load_source_snapshot(prepared.key)
            assert all(
                event.get("content") != "Continue task"
                for event in snapshot.events
                if event.get("role") == "user"
            )
            assert "execution_lease" not in snapshot.header["metadata"]
    finally:
        prepared.provider.call_release.set()
        handle.agent.stop()
        if run_task is not None and not run_task.done():
            await handle.bus.publish_inbound(InboundMessage(
                channel="system", sender_id="test", chat_id="", content="shutdown wake",
            ))
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=5)
        await prepared.composition.aclose()


@pytest.mark.asyncio
async def test_rejected_queued_continue_releases_only_unadmitted_gateway_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    from nanobot.channels.websocket.runtime import WebSocketChannel, WebSocketConfig
    from nanobot.config.loader import get_config_path, set_config_path
    from nanobot.session import webui_turns
    from nanobot.session.webui_turns import WebuiTurnCoordinator
    from nanobot.webui.gateway_services import build_gateway_services

    prepared = await _prepare_continuation(monkeypatch, tmp_path, request)
    handle = prepared.handle
    prior_config_path = get_config_path()
    config_path = tmp_path / "runtime" / "gateway-config.json"
    set_config_path(config_path)
    config = WebSocketConfig(enabled=True, allow_from=["*"])
    gateway = build_gateway_services(
        config=config,
        bus=handle.bus,
        session_manager=handle.sessions,
        static_dist_path=None,
        workspace_path=prepared.config.workspace_path,
        default_restrict_to_workspace=True,
        config_path=config_path,
        runtime_model_name=lambda: None,
        runtime_surface="test",
        runtime_capabilities_overrides=None,
    )
    gateway.http.tokens.api_tokens["queued-owner-test-token"] = time.monotonic() + 300
    channel = WebSocketChannel(config, handle.bus, gateway=gateway)
    coordinator = WebuiTurnCoordinator(
        bus=handle.bus,
        sessions=handle.sessions,
        schedule_background=lambda coroutine: coroutine.close(),
    )
    disconnect = coordinator.subscribe()
    try:
        # Deliver the initial canonical result through the real coordinator and
        # channel so the thread GET reads its normal persisted transcript.
        assert prepared.initial_result.status is RunStatus.COMPLETED
        await coordinator.handle_turn_end(
            InboundMessage(
                channel="websocket",
                sender_id="runtime",
                chat_id=prepared.chat_id,
                content="",
                metadata={"webui": True, "webui_turn_id": "original-turn"},
                session_key_override=prepared.key,
            ),
            session_key=prepared.key,
            latency_ms=0,
            outcome="completed",
        )
        while handle.bus.outbound_size:
            await channel.send(await handle.bus.consume_outbound())
        accepted = await prepared.recovery.continue_request({
            "chat_id": prepared.chat_id,
            "access_request_id": prepared.access_request_id,
            "expected_revision": prepared.ready_revision,
        })
        turn_id = accepted["workspace_access_request"]["continuation_turn_id"]
        assert isinstance(turn_id, str)
        assert webui_turns.websocket_turn_id(prepared.chat_id) == turn_id

        inbound = await handle.bus.consume_inbound()
        grants = prepared.recovery.workspaces.grant_store
        grants.revoke(str(prepared.project), expected_revision=grants.snapshot().revision)
        provider_calls_before = len(prepared.provider.requests)

        await handle.agent._dispatch(inbound)

        outbound = []
        while handle.bus.outbound_size:
            message = await handle.bus.consume_outbound()
            outbound.append(message)
            await channel.send(message)
        from websockets.datastructures import Headers
        from websockets.http11 import Request

        encoded_key = quote(prepared.key, safe="")
        response = gateway.http._handle_webui_thread_get(
            Request(
                f"/api/sessions/{encoded_key}/webui-thread",
                Headers([("Authorization", "Bearer queued-owner-test-token")]),
            ),
            encoded_key,
        )
        assert response.status_code == 200
        thread = json.loads(response.body.decode())
        backend = handle.sessions.source_backend(prepared.key)
        assert backend is not None
        snapshot = backend.source.load_source_snapshot(prepared.key)
        record = snapshot.header["metadata"]["workspace_access_request"]

        # These assertions run before the test-only fixture owner finalizer.
        assert len(prepared.provider.requests) == provider_calls_before
        assert snapshot.header["metadata"].get("execution_lease") is None
        assert not any(
            event.get("role") == "user" and event.get("content") == "Continue task"
            for event in snapshot.events
        )
        assert record["status"] == "continuing"
        assert webui_turns.websocket_turn_wall_started_at(prepared.chat_id) is None
        assert webui_turns.websocket_turn_id(prepared.chat_id) is None
        assert thread["active_turn_id"] is None
        assert turn_id not in thread["completed_turn_ids"]

        with pytest.raises(WorkspaceAccessActionError) as retry:
            await prepared.recovery.continue_request({
                "chat_id": prepared.chat_id,
                "access_request_id": prepared.access_request_id,
                "expected_revision": record["revision"],
            })
        assert retry.value.message == "access_request_inactive"
    finally:
        disconnect()
        set_config_path(prior_config_path)
        await prepared.composition.aclose()


@pytest.mark.asyncio
async def test_rejected_continuation_does_not_clear_foreign_superseding_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    from dataclasses import replace

    from nanobot.session.webui_turns import (
        publish_turn_run_status,
        websocket_turn_owner_is_registered,
    )
    from nanobot.session.workspace_access_continuation import (
        copy_workspace_access_continuation,
    )
    from nanobot.webui.metadata import (
        WEBSOCKET_TURN_OWNER_METADATA_KEY,
        WEBUI_TURN_METADATA_KEY,
    )

    prepared = await _prepare_continuation(monkeypatch, tmp_path, request)
    handle = prepared.handle
    foreign_owner = "foreign-running-owner"
    try:
        accepted = await prepared.recovery.continue_request({
            "chat_id": prepared.chat_id,
            "access_request_id": prepared.access_request_id,
            "expected_revision": prepared.ready_revision,
        })
        inbound = await handle.bus.consume_inbound()
        original_owner = inbound.metadata[WEBSOCKET_TURN_OWNER_METADATA_KEY]
        turn_id = accepted["workspace_access_request"]["continuation_turn_id"]
        assert isinstance(original_owner, str)
        assert isinstance(turn_id, str)

        foreign_turn_id = "foreign-turn"
        foreign = replace(
            inbound,
            metadata={
                **inbound.metadata,
                WEBSOCKET_TURN_OWNER_METADATA_KEY: foreign_owner,
                WEBUI_TURN_METADATA_KEY: foreign_turn_id,
            },
        )
        copy_workspace_access_continuation(inbound, foreign)
        await publish_turn_run_status(handle.bus, foreign, "running")

        await handle.agent._dispatch(foreign)

        backend = handle.sessions.source_backend(prepared.key)
        assert backend is not None
        snapshot = backend.source.load_source_snapshot(prepared.key)
        assert snapshot.header["metadata"].get("execution_lease") is None
        assert snapshot.header["metadata"]["workspace_access_request"]["status"] == "continuing"
        assert len(prepared.provider.requests) == 1
        assert websocket_turn_owner_is_registered(
            prepared.chat_id,
            original_owner,
            turn_id,
        )
        assert websocket_turn_owner_is_registered(
            prepared.chat_id,
            foreign_owner,
            foreign_turn_id,
        )
    finally:
        from nanobot.session.webui_turns import clear_websocket_turn_if_current

        clear_websocket_turn_if_current(prepared.chat_id, foreign_owner)
        await prepared.composition.aclose()


@pytest.mark.asyncio
async def test_continue_binding_is_rechecked_after_existing_source_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    from nanobot.agent import loop as loop_module
    from nanobot.session.execution_owner import acquire_execution as original_acquire

    prepared = await _prepare_continuation(monkeypatch, tmp_path, request)
    handle = prepared.handle
    dispatch_done = asyncio.Event()
    run_task: asyncio.Task[None] | None = None
    try:
        result = await prepared.recovery.continue_request({
            "chat_id": prepared.chat_id,
            "access_request_id": prepared.access_request_id,
            "expected_revision": prepared.ready_revision,
        })
        assert result["workspace_access_request"]["status"] == "continuing"
        original_bytes = handle.sessions._get_session_path(prepared.key).read_bytes()
        reset_complete = asyncio.Event()

        @asynccontextmanager
        async def reset_before_lock(
            backend,
            key: str,
            *,
            run_id: str,
            expected_source=None,
        ):
            handle.sessions.reset_session(prepared.key)
            reset_complete.set()
            async with original_acquire(
                backend,
                key,
                run_id=run_id,
                expected_source=expected_source,
            ) as owner:
                yield owner

        monkeypatch.setattr(loop_module, "acquire_execution", reset_before_lock)
        original_enqueue = handle.agent._enqueue_bus_dispatch

        def track_dispatch(message, key: str):
            task = original_enqueue(message, key)
            task.add_done_callback(lambda _task: dispatch_done.set())
            return task

        monkeypatch.setattr(handle.agent, "_enqueue_bus_dispatch", track_dispatch)
        run_task = asyncio.create_task(handle.agent.run())
        await asyncio.wait_for(reset_complete.wait(), timeout=5)
        await asyncio.wait_for(dispatch_done.wait(), timeout=5)
        assert len(prepared.provider.requests) == 1
        replacement_path = handle.sessions._get_session_path(prepared.key)
        replacement_bytes = replacement_path.read_bytes()
        assert replacement_bytes != original_bytes
        assert "Continue task" not in replacement_bytes.decode("utf-8")
        snapshot = handle.sessions._jsonl_store.source.load_source_snapshot(prepared.key)
        assert "execution_lease" not in snapshot.header["metadata"]
        assert all(
            event.get("content") != "Continue task"
            for event in snapshot.events
            if event.get("role") == "user"
        )
        from nanobot.session.webui_turns import (
            websocket_turn_id,
            websocket_turn_wall_started_at,
        )

        assert websocket_turn_id(prepared.chat_id) is None
        assert websocket_turn_wall_started_at(prepared.chat_id) is None
    finally:
        handle.agent.stop()
        if run_task is not None and not run_task.done():
            await handle.bus.publish_inbound(InboundMessage(
                channel="system", sender_id="test", chat_id="", content="shutdown wake",
            ))
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=5)
        await prepared.composition.aclose()


@pytest.mark.asyncio
async def test_accepted_new_input_invalidates_queued_continue_without_busy_injection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    from nanobot.session.execution_owner import acquire_execution

    prepared = await _prepare_continuation(monkeypatch, tmp_path, request)
    handle = prepared.handle
    backend = handle.sessions.source_backend(prepared.key)
    assert backend is not None
    enqueued_one = asyncio.Event()
    enqueued_two = asyncio.Event()
    both_dispatched = asyncio.Event()
    dispatch_count = 0
    finished_count = 0
    run_task: asyncio.Task[None] | None = None
    try:
        result = await prepared.recovery.continue_request({
            "chat_id": prepared.chat_id,
            "access_request_id": prepared.access_request_id,
            "expected_revision": prepared.ready_revision,
        })
        assert result["workspace_access_request"]["status"] == "continuing"

        original_enqueue = handle.agent._enqueue_bus_dispatch

        def track_dispatch(message, key: str):
            nonlocal dispatch_count, finished_count
            task = original_enqueue(message, key)
            dispatch_count += 1
            if dispatch_count == 1:
                enqueued_one.set()
            elif dispatch_count == 2:
                enqueued_two.set()

            def finished(_task) -> None:
                nonlocal finished_count
                finished_count += 1
                if finished_count == 2:
                    both_dispatched.set()

            task.add_done_callback(finished)
            return task

        monkeypatch.setattr(handle.agent, "_enqueue_bus_dispatch", track_dispatch)
        async with acquire_execution(backend, prepared.key, run_id="busy-owner"):
            run_task = asyncio.create_task(handle.agent.run())
            await asyncio.wait_for(enqueued_one.wait(), timeout=5)
            assert prepared.key not in handle.agent._pending_queues
            await handle.bus.publish_inbound(InboundMessage(
                channel="websocket",
                sender_id="webui",
                chat_id=prepared.chat_id,
                content="Please inspect the changelog too.",
                metadata={"webui": True},
            ))
            await asyncio.wait_for(enqueued_two.wait(), timeout=5)
            for _ in range(500):
                snapshot = backend.source.load_source_snapshot(prepared.key)
                record = snapshot.header["metadata"]["workspace_access_request"]
                if record["status"] == "invalidated":
                    break
                await asyncio.sleep(0.01)
            assert record["status"] == "invalidated"
            assert prepared.key not in handle.agent._pending_queues

        prepared.provider.called.clear()
        await asyncio.wait_for(prepared.provider.called.wait(), timeout=5)
        await asyncio.wait_for(both_dispatched.wait(), timeout=5)
        assert len(prepared.provider.requests) == 2
        continued_inputs = [
            str(row.get("content"))
            for row in prepared.provider.requests[1]
            if row.get("role") == "user"
        ]
        assert "Please inspect the changelog too." in continued_inputs
        assert "Continue task" not in continued_inputs
        final_snapshot = backend.source.load_source_snapshot(prepared.key)
        assert all(
            event.get("content") != "Continue task"
            for event in final_snapshot.events
            if event.get("role") == "user"
        )
    finally:
        handle.agent.stop()
        if run_task is not None and not run_task.done():
            await handle.bus.publish_inbound(InboundMessage(
                channel="system", sender_id="test", chat_id="", content="shutdown wake",
            ))
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=5)
        await prepared.composition.aclose()


@pytest.mark.asyncio
async def test_duplicate_private_continue_dispatch_runs_provider_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    from dataclasses import replace

    from nanobot.session.workspace_access_continuation import copy_workspace_access_continuation

    prepared = await _prepare_continuation(monkeypatch, tmp_path, request)
    handle = prepared.handle
    original_dispatch_done = asyncio.Event()
    duplicate_dispatch_done = asyncio.Event()
    dispatch_count = 0
    run_task: asyncio.Task[None] | None = None
    try:
        result = await prepared.recovery.continue_request({
            "chat_id": prepared.chat_id,
            "access_request_id": prepared.access_request_id,
            "expected_revision": prepared.ready_revision,
        })
        assert result["workspace_access_request"]["status"] == "continuing"
        original_message = handle.bus.inbound.get_nowait()
        duplicate_message = replace(original_message, metadata=dict(original_message.metadata))
        copy_workspace_access_continuation(original_message, duplicate_message)
        assert duplicate_message.require_existing_session
        prepared.provider.gated_call_number = 2
        prepared.provider.call_release.clear()

        original_enqueue = handle.agent._enqueue_bus_dispatch

        def track_dispatch(message, key: str):
            nonlocal dispatch_count
            task = original_enqueue(message, key)
            dispatch_count += 1
            task.add_done_callback(lambda _task: original_dispatch_done.set())
            return task

        monkeypatch.setattr(handle.agent, "_enqueue_bus_dispatch", track_dispatch)
        await handle.bus.publish_inbound(original_message)
        run_task = asyncio.create_task(handle.agent.run())
        await asyncio.wait_for(prepared.provider.gate_started.wait(), timeout=5)
        assert dispatch_count == 1
        owner = original_message.metadata["_websocket_turn_owner"]
        turn_id = result["workspace_access_request"]["continuation_turn_id"]
        from nanobot.session.webui_turns import (
            websocket_turn_id,
            websocket_turn_owner_is_registered,
            websocket_turn_wall_started_at,
        )

        assert websocket_turn_id(prepared.chat_id) == turn_id
        assert websocket_turn_wall_started_at(prepared.chat_id) is not None
        assert websocket_turn_owner_is_registered(prepared.chat_id, owner, turn_id)

        # This second dispatch carries the same internal continuation and owner
        # while the original canonical execution owns the source.
        duplicate_task = asyncio.create_task(handle.agent._dispatch(duplicate_message))
        duplicate_task.add_done_callback(lambda _task: duplicate_dispatch_done.set())
        assert websocket_turn_id(prepared.chat_id) == turn_id
        assert websocket_turn_wall_started_at(prepared.chat_id) is not None
        assert websocket_turn_owner_is_registered(prepared.chat_id, owner, turn_id)
        assert len(prepared.provider.requests) == 2

        prepared.provider.call_release.set()
        await asyncio.wait_for(original_dispatch_done.wait(), timeout=5)
        await asyncio.wait_for(duplicate_dispatch_done.wait(), timeout=5)
        assert dispatch_count == 1
        assert len(prepared.provider.requests) == 2
        visible_inputs = [
            str(row.get("content"))
            for row in prepared.provider.requests[1]
            if row.get("role") == "user"
        ]
        assert visible_inputs.count("Continue task") == 1
        backend = handle.sessions.source_backend(prepared.key)
        assert backend is not None
        snapshot = backend.source.load_source_snapshot(prepared.key)
        assert [
            event["content"]
            for event in snapshot.events
            if event.get("role") == "user" and event.get("content") == "Continue task"
        ] == ["Continue task"]
        assert snapshot.header["metadata"]["workspace_access_request"]["status"] == "continued"
    finally:
        prepared.provider.call_release.set()
        handle.agent.stop()
        if run_task is not None and not run_task.done():
            await handle.bus.publish_inbound(InboundMessage(
                channel="system", sender_id="test", chat_id="", content="shutdown wake",
            ))
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=5)
        await prepared.composition.aclose()


@pytest.mark.parametrize("terminal_persistence_succeeds", [True, False])
@pytest.mark.asyncio
async def test_duplicate_rejection_preserves_owner_during_terminal_channel_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    terminal_persistence_succeeds: bool,
) -> None:
    from dataclasses import replace

    from nanobot.bus.outbound_events import TurnEndEvent
    from nanobot.channels.websocket.runtime import WebSocketChannel, WebSocketConfig
    from nanobot.config.loader import get_config_path, set_config_path
    from nanobot.session import webui_turns
    from nanobot.session.webui_turns import WebuiTurnCoordinator
    from nanobot.session.workspace_access_continuation import (
        copy_workspace_access_continuation,
    )
    from nanobot.webui.gateway_services import build_gateway_services
    from nanobot.webui.metadata import WEBSOCKET_TURN_OWNER_METADATA_KEY, WEBUI_TURN_METADATA_KEY

    prepared = await _prepare_continuation(monkeypatch, tmp_path, request)
    handle = prepared.handle
    prior_config_path = get_config_path()
    config_path = tmp_path / "runtime" / "gateway-config.json"
    set_config_path(config_path)
    config = WebSocketConfig(enabled=True, allow_from=["*"])
    gateway = build_gateway_services(
        config=config,
        bus=handle.bus,
        session_manager=handle.sessions,
        static_dist_path=None,
        workspace_path=prepared.config.workspace_path,
        default_restrict_to_workspace=True,
        config_path=config_path,
        runtime_model_name=lambda: None,
        runtime_surface="test",
        runtime_capabilities_overrides=None,
    )
    gateway.http.tokens.api_tokens["terminal-owner-test-token"] = time.monotonic() + 300
    channel = WebSocketChannel(config, handle.bus, gateway=gateway)
    coordinator = WebuiTurnCoordinator(
        bus=handle.bus,
        sessions=handle.sessions,
        schedule_background=lambda coroutine: coroutine.close(),
    )
    disconnect = coordinator.subscribe()
    terminal_seen = asyncio.Event()
    release_terminal = asyncio.Event()
    terminal_persisted = asyncio.Event()
    run_task: asyncio.Task[None] | None = None
    output_task: asyncio.Task[None] | None = None
    try:
        accepted = await prepared.recovery.continue_request({
            "chat_id": prepared.chat_id,
            "access_request_id": prepared.access_request_id,
            "expected_revision": prepared.ready_revision,
        })
        turn_id = accepted["workspace_access_request"]["continuation_turn_id"]
        assert isinstance(turn_id, str)
        inbound = await handle.bus.consume_inbound()
        owner = inbound.metadata[WEBSOCKET_TURN_OWNER_METADATA_KEY]
        assert isinstance(owner, str)
        duplicate = replace(inbound, metadata=dict(inbound.metadata))
        copy_workspace_access_continuation(inbound, duplicate)
        assert duplicate.metadata[WEBSOCKET_TURN_OWNER_METADATA_KEY] == owner

        if not terminal_persistence_succeeds:
            original_prepare_and_append = channel._transcripts.prepare_and_append

            def fail_terminal_persistence(
                chat_id: str,
                event: dict[str, Any],
                **kwargs: Any,
            ) -> bool:
                metadata = kwargs.get("metadata")
                if (
                    event.get("event") == "turn_end"
                    and isinstance(metadata, dict)
                    and metadata.get(WEBUI_TURN_METADATA_KEY) == turn_id
                ):
                    return False
                return original_prepare_and_append(chat_id, event, **kwargs)

            monkeypatch.setattr(
                channel._transcripts,
                "prepare_and_append",
                fail_terminal_persistence,
            )

        async def drain_until_terminal() -> None:
            while True:
                outbound = await handle.bus.consume_outbound()
                if (
                    isinstance(outbound.event, TurnEndEvent)
                    and outbound.metadata.get(WEBUI_TURN_METADATA_KEY) == turn_id
                ):
                    terminal_seen.set()
                    await release_terminal.wait()
                    await channel.send(outbound)
                    terminal_persisted.set()
                    return
                await channel.send(outbound)

        output_task = asyncio.create_task(drain_until_terminal())
        await handle.bus.publish_inbound(inbound)
        run_task = asyncio.create_task(handle.agent.run())
        await asyncio.wait_for(terminal_seen.wait(), timeout=5)
        assert webui_turns.websocket_turn_id(prepared.chat_id) == turn_id
        assert webui_turns.websocket_turn_wall_started_at(prepared.chat_id) is not None
        assert webui_turns.websocket_turn_owner_is_registered(
            prepared.chat_id,
            owner,
            turn_id,
        )

        # The duplicate rejects after canonical source admission while the real
        # channel is holding this owner's terminal transcript delivery.
        await asyncio.wait_for(handle.agent._dispatch(duplicate), timeout=5)
        assert webui_turns.websocket_turn_id(prepared.chat_id) == turn_id
        assert webui_turns.websocket_turn_wall_started_at(prepared.chat_id) is not None
        assert webui_turns.websocket_turn_owner_is_registered(
            prepared.chat_id,
            owner,
            turn_id,
        )

        release_terminal.set()
        await asyncio.wait_for(terminal_persisted.wait(), timeout=5)
        if terminal_persistence_succeeds:
            assert webui_turns.websocket_turn_id(prepared.chat_id) is None
            assert webui_turns.websocket_turn_wall_started_at(prepared.chat_id) is None
        else:
            assert webui_turns.websocket_turn_id(prepared.chat_id) == turn_id
            assert webui_turns.websocket_turn_wall_started_at(prepared.chat_id) is not None
            assert webui_turns.websocket_turn_transcript_persistence_failed(
                prepared.chat_id,
                owner,
            )
            await handle.agent._dispatch(duplicate)
            assert webui_turns.websocket_turn_id(prepared.chat_id) == turn_id
            assert webui_turns.websocket_turn_owner_is_registered(
                prepared.chat_id,
                owner,
                turn_id,
            )
    finally:
        release_terminal.set()
        if output_task is not None:
            if not output_task.done():
                output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)
        handle.agent.stop()
        if run_task is not None and not run_task.done():
            await handle.bus.publish_inbound(InboundMessage(
                channel="system", sender_id="test", chat_id="", content="shutdown wake",
            ))
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=5)
        disconnect()
        set_config_path(prior_config_path)
        await prepared.composition.aclose()
