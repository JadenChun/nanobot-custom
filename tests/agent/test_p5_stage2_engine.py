from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.loop import AgentLoop, _TurnExecutionOptions
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnRequest
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.model_selection import SESSION_MODEL_PRESET_METADATA_KEY
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.webui.temporary_chats import WebUITemporaryChats
from nanobot.webui.workspaces import WebUIWorkspaceController


class ScriptedProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        from tests.agent.test_migration_engine import ScriptedProvider as Provider

        self._provider = Provider(responses)

    def __getattr__(self, name: str) -> object:
        return getattr(self._provider, name)


@pytest.fixture
def engine_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("subprocess.Popen", Mock(side_effect=AssertionError("process execution")))
    for target in (
        "socket.socket.connect",
        "socket.socket.connect_ex",
        "socket.create_connection",
        "socket.getaddrinfo",
    ):
        monkeypatch.setattr(target, Mock(side_effect=AssertionError("outbound network access")))
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        AsyncMock(side_effect=AssertionError("process execution")),
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_shell",
        AsyncMock(side_effect=AssertionError("shell execution")),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config()
    config.bind_source_path(tmp_path / "instance" / "config.json")
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "synthetic-model"
    config.agents.defaults.model_preset = None
    config.agents.defaults.max_tool_iterations = 4
    config.tools.agent_browser.enabled = False
    config.tools.agent_device.enabled = False
    config.tools.desktop_use.enabled = False
    config.tools.restrict_to_workspace = True
    return config


async def _managed_loop(config: Config, provider: object):
    lease = await DurableSessionAuthority(
        config.workspace_path,
        sessions_root=config.runtime_data_dir / "sessions",  # type: ignore[operator]
    ).acquire()
    loop = AgentLoop.from_config(
        config,
        tool_registry=ToolRegistry(),
        provider=provider,
        session_manager=lease.manager,
    )
    return lease, loop


async def test_persistent_turn_creates_canonical_before_provider_and_uses_real_tool(
    engine_config: Config,
) -> None:
    (engine_config.workspace_path / "sample.txt").write_text("fixture contents")
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="read-1",
                        name="read_file",
                        arguments={"path": "sample.txt"},
                    )
                ],
            ),
            LLMResponse(content="done"),
        ]
    )
    lease, loop = await _managed_loop(engine_config, provider)
    try:
        result = await loop.execute_turn(
            TurnRequest(content="read", session_key="cli:item")
        )
        assert result.content == "done"
        resolved = lease.manager.resolve_session_mode("cli:item")
        assert resolved.mode == "canonical"
        snapshot = lease.manager._jsonl_store.source.load_source_snapshot("cli:item")
        assert snapshot.to_session().provider_state is None
        assert [event["role"] for event in snapshot.events] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert loop.run_store.load(result.run_id) is not None
    finally:
        await loop.aclose()
        assert lease.close is not None
        await lease.close()


@pytest.mark.parametrize("existing", [False, True])
async def test_ephemeral_turn_is_detached_from_absent_or_canonical_state(
    engine_config: Config,
    existing: bool,
) -> None:
    provider = ScriptedProvider([LLMResponse(content="ephemeral")])
    lease, loop = await _managed_loop(engine_config, provider)
    key = "cli:item"
    try:
        if existing:
            lease.manager.admit_persistent_write(key)
            durable = lease.manager.get_or_create(key)
            source_before = lease.manager._get_session_path(key).read_bytes()
            mode_before = lease.manager.resolve_session_mode(key).binding
        else:
            durable = None
            source_before = None
            mode_before = lease.manager.resolve_session_mode(key).binding
        result = await loop.execute_turn(
            TurnRequest(content="temporary", session_key=key),
            _options=_TurnExecutionOptions(ephemeral=True),
        )
        assert result.content == "ephemeral"
        assert loop.run_store.load(result.run_id) is None
        assert lease.manager.resolve_session_mode(key).binding == mode_before
        if existing:
            assert lease.manager._get_session_path(key).read_bytes() == source_before
            assert lease.manager.get_cached(key) is durable
            assert durable is not None and durable.messages == []
        else:
            assert not lease.manager._get_session_path(key).exists()
            assert lease.manager.get_cached(key) is None
    finally:
        await loop.aclose()
        assert lease.close is not None
        await lease.close()


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/model default", "Switched model preset"),
        ("/new", "New session started."),
    ],
)
async def test_ephemeral_commands_do_not_mutate_existing_canonical_state(
    engine_config: Config,
    command: str,
    expected: str,
) -> None:
    provider = ScriptedProvider([])
    lease, loop = await _managed_loop(engine_config, provider)
    key = "cli:item"
    try:
        lease.manager.admit_persistent_write(key)
        durable = lease.manager.get_or_create(key)
        durable.messages.append({"role": "user", "content": "cached"})
        durable.metadata["cached"] = {"value": True}
        cached_messages = deepcopy(durable.messages)
        cached_metadata = deepcopy(durable.metadata)
        source_before = lease.manager._get_session_path(key).read_bytes()
        binding_before = lease.manager.resolve_session_mode(key).binding
        record_name = lease.manager._mode_resolver._record_name(key)
        record_before = lease.manager._mode_resolver.storage.read(record_name)
        original_reset = lease.manager.reset_session
        lease.manager.reset_session = Mock(wraps=original_reset)  # type: ignore[method-assign]
        if command == "/model default":
            loop.runtime_resolver.resolve_preset = Mock(  # type: ignore[method-assign]
                return_value=loop.runtime_resolver.runtime
            )

        result = await loop.execute_turn(
            TurnRequest(content=command, session_key=key),
            _options=_TurnExecutionOptions(ephemeral=True),
        )

        assert result.content is not None and expected in result.content
        assert loop.run_store.load(result.run_id) is None
        assert provider._provider.calls == []
        assert lease.manager._get_session_path(key).read_bytes() == source_before
        assert lease.manager.resolve_session_mode(key).binding == binding_before
        assert lease.manager._mode_resolver.storage.read(record_name) == record_before
        assert lease.manager._cache.get(key) is durable
        assert durable.messages == cached_messages
        assert durable.metadata == cached_metadata
        lease.manager.reset_session.assert_not_called()  # type: ignore[attr-defined]
    finally:
        await loop.aclose()
        assert lease.close is not None
        await lease.close()


async def test_corrupt_mode_refuses_before_approval_or_provider(engine_config: Config) -> None:
    provider = ScriptedProvider([LLMResponse(content="must not run")])
    lease, loop = await _managed_loop(engine_config, provider)
    key = "cli:item"
    try:
        lease.manager.admit_persistent_write(key)
        record_name = lease.manager._mode_resolver._record_name(key)
        record = lease.manager._mode_resolver.storage.read(record_name)
        lease.manager._mode_resolver.storage.write(record_name, record + b" ")
        loop.approval_admission.consume = AsyncMock(  # type: ignore[method-assign]
            wraps=loop.approval_admission.consume
        )
        result = await loop.execute_turn(TurnRequest(content="run", session_key=key))
        assert result.error is not None
        loop.approval_admission.consume.assert_not_awaited()  # type: ignore[attr-defined]
        assert provider._provider.calls == []
    finally:
        await loop.aclose()
        assert lease.close is not None
        await lease.close()


async def test_ephemeral_corrupt_mode_refuses_without_durable_mutation(
    engine_config: Config,
) -> None:
    provider = ScriptedProvider([LLMResponse(content="must not run")])
    lease, loop = await _managed_loop(engine_config, provider)
    key = "cli:item"
    try:
        lease.manager.admit_persistent_write(key)
        durable = lease.manager.get_or_create(key)
        cached_messages = deepcopy(durable.messages)
        cached_metadata = deepcopy(durable.metadata)
        source_before = lease.manager._get_session_path(key).read_bytes()
        record_name = lease.manager._mode_resolver._record_name(key)
        record = lease.manager._mode_resolver.storage.read(record_name)
        lease.manager._mode_resolver.storage.write(record_name, record + b" ")
        corrupt_record = lease.manager._mode_resolver.storage.read(record_name)
        loop.approval_admission.consume = AsyncMock(  # type: ignore[method-assign]
            wraps=loop.approval_admission.consume
        )

        result = await loop.execute_turn(
            TurnRequest(content="temporary", session_key=key),
            _options=_TurnExecutionOptions(ephemeral=True),
        )

        assert result.error is not None
        assert loop.run_store.load(result.run_id) is None
        loop.approval_admission.consume.assert_not_awaited()  # type: ignore[attr-defined]
        assert provider._provider.calls == []
        assert lease.manager._get_session_path(key).read_bytes() == source_before
        assert lease.manager._mode_resolver.storage.read(record_name) == corrupt_record
        assert lease.manager._cache.get(key) is durable
        assert durable.messages == cached_messages
        assert durable.metadata == cached_metadata
    finally:
        await loop.aclose()
        assert lease.close is not None
        await lease.close()


async def test_source_new_command_delegates_to_manager_reset(engine_config: Config) -> None:
    provider = ScriptedProvider([])
    lease, loop = await _managed_loop(engine_config, provider)
    key = "cli:item"
    try:
        lease.manager.admit_persistent_write(key)
        original = lease.manager.reset_session
        lease.manager.reset_session = Mock(wraps=original)  # type: ignore[method-assign]
        result = await loop.execute_turn(TurnRequest(content="/new", session_key=key))
        assert result.content == "New session started."
        lease.manager.reset_session.assert_called_once()  # type: ignore[attr-defined]
        assert provider._provider.calls == []
        assert lease.manager.resolve_session_mode(key).generation == 1
    finally:
        await loop.aclose()
        assert lease.close is not None
        await lease.close()


def _temporary_chats(manager, bus):
    return WebUITemporaryChats(bus=bus, session_manager=manager,
        workspaces=WebUIWorkspaceController(session_manager=manager,
            default_workspace=manager.workspace, default_restrict_to_workspace=True), logger=Mock())


async def test_live_temporary_engine_turns_share_identity_history_and_exclude_durable_effects(
    engine_config: Config, monkeypatch,
) -> None:
    provider = ScriptedProvider([LLMResponse(content="first answer"), LLMResponse(content="second answer")])
    lease, loop = await _managed_loop(engine_config, provider)
    chats = _temporary_chats(lease.manager, loop.bus)
    try:
        from tests.session.test_p5_stage2_consumers import state

        chat_id = chats.create(object(), trusted_webui=True)
        key = f"websocket:{chat_id}"
        session = chats._active_sessions[chat_id]
        before = state(lease.manager)
        assert lease.manager.get_cached(key) is session
        memory_read = Mock(side_effect=AssertionError("temporary turn read durable memory"))
        monkeypatch.setattr(loop.context.memory, "read_memory", memory_read)
        restored = []
        restore = loop._restore_turn

        async def capture(ctx):
            assert ctx.session is session
            await restore(ctx)
            restored.append(ctx.session)

        monkeypatch.setattr(loop, "_restore_turn", capture)
        hidden_logs = Mock()
        monkeypatch.setattr("nanobot.agent.loop.logger", hidden_logs)
        results = []
        for content in ("first private message", "second private message"):
            inbound = InboundMessage(channel="websocket", sender_id="webui", chat_id=chat_id,
                content=content, require_existing_session=True)
            results.append(await loop.execute_turn(TurnRequest(content=content, session_key=key),
                _options=_TurnExecutionOptions(inbound=inbound)))
        assert [result.content for result in results] == ["first answer", "second answer"]
        assert restored == [session, session] and lease.manager.get_cached(key) is session
        assert [row["content"] for row in session.messages] == [
            "first private message", "first answer", "second private message", "second answer"]
        assert "first private message" in str(provider._provider.calls[1][0])
        assert not {"create_goal", "update_goal", "spawn", "cron"} & {
            tool["function"]["name"] for tool in provider._provider.calls[0][1]}
        memory_read.assert_not_called()
        assert all(loop.run_store.load(result.run_id) is None for result in results)
        assert state(lease.manager) == before
        assert lease.manager.read_session_metadata(key) is None
        assert all(secret not in repr(hidden_logs.info.call_args_list) for secret in (
            "first private message", "second private message", "first answer", "second answer"))
        loop.runtime_resolver.resolve_preset = Mock(return_value=replace(
            loop.runtime_resolver.runtime, model_preset="default"))
        command = await loop.execute_turn(TurnRequest(content="/model default", session_key=key))
        assert command.content is not None and "Switched model preset" in command.content
        assert session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] == "default"
        assert lease.manager.get_cached(key) is session and state(lease.manager) == before
        assert loop.run_store.load(command.run_id) is None
    finally:
        chats.close()
        await loop.aclose()
        await lease.close()


@pytest.mark.parametrize("conflict", ["persistent", "record", "sidecar", "recovery"])
async def test_live_temporary_admission_rechecks_managed_evidence_before_effects(engine_config: Config, conflict):
    provider = ScriptedProvider([LLMResponse(content="must not run")])
    lease, loop = await _managed_loop(engine_config, provider)
    chats = _temporary_chats(lease.manager, loop.bus)
    try:
        from tests.session.test_p5_stage2_consumers import state

        lease.manager.admit_persistent_write("cli:registry-fixture")
        chat_id = chats.create(object(), trusted_webui=True)
        key = f"websocket:{chat_id}"
        live = chats._active_sessions[chat_id]
        resolver = lease.manager._mode_resolver
        if conflict == "record":
            resolver.storage.write(resolver._record_name(key), b"{}")
        elif conflict == "sidecar":
            resolver.storage.write(f"{lease.manager._storage_key(key)}.checkpoint.json", b"{}")
        else:
            lease.manager.admit_persistent_write(key)
        if conflict == "recovery":
            rows = [json.loads(row) for row in lease.manager._get_session_path(key).read_bytes().splitlines()]
            rows[0]["revision"] += 1
            rows[0]["metadata"]["execution_recovery_required"] = True
            resolver.storage.write(f"{lease.manager._storage_key(key)}.jsonl",
                b"".join((json.dumps(row) + "\n").encode() for row in rows))
        before = state(lease.manager)
        loop.approval_admission.consume = AsyncMock(wraps=loop.approval_admission.consume)
        inbound = InboundMessage(channel="websocket", sender_id="webui", chat_id=chat_id,
            content="private", require_existing_session=True)
        result = await loop.execute_turn(TurnRequest(content=inbound.content, session_key=key),
            _options=_TurnExecutionOptions(inbound=inbound))
        assert result.error is not None
        assert key not in result.error and str(lease.manager.sessions_dir) not in result.error
        assert state(lease.manager) == before and loop.run_store.load(result.run_id) is None
        loop.approval_admission.consume.assert_not_awaited()
        assert provider._provider.calls == []
        if conflict in {"record", "sidecar"}:
            assert lease.manager._cache[key] is live and live.messages == []
    finally:
        chats.close()
        await loop.aclose()
        await lease.close()


async def test_runtime_control_discard_cancels_active_temporary_bus_turn(engine_config: Config, monkeypatch):
    provider = ScriptedProvider([])
    lease, loop = await _managed_loop(engine_config, provider)
    chats = _temporary_chats(lease.manager, loop.bus)
    started, cancelled = asyncio.Event(), asyncio.Event()
    run_task = None
    try:
        from tests.session.test_p5_stage2_consumers import state

        owner = object()
        chat_id = chats.create(owner, trusted_webui=True)
        key = f"websocket:{chat_id}"
        before = state(lease.manager)

        async def blocked_provider(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr(provider._provider, "chat", blocked_provider)
        durable_run = Mock(side_effect=AssertionError("temporary run persistence"))
        monkeypatch.setattr(loop.run_store, "save", durable_run)
        run_task = asyncio.create_task(loop.run())
        await loop.bus.publish_inbound(InboundMessage(channel="websocket", sender_id="webui",
            chat_id=chat_id, content="private active turn", require_existing_session=True))
        await asyncio.wait_for(started.wait(), timeout=5)
        assert key in loop._active_tasks
        await chats.discard(owner, chat_id)
        await asyncio.wait_for(cancelled.wait(), timeout=5)
        await loop.aclose()
        assert lease.manager.get_cached(key) is None
        assert state(lease.manager) == before
        durable_run.assert_not_called()
    finally:
        chats.close()
        await loop.aclose()
        if run_task:
            await asyncio.gather(run_task, return_exceptions=True)
        await lease.close()


async def test_explicit_ephemeral_at_live_temporary_key_is_detached(engine_config: Config) -> None:
    provider = ScriptedProvider([LLMResponse(content="detached answer")])
    lease, loop = await _managed_loop(engine_config, provider)
    chats = _temporary_chats(lease.manager, loop.bus)
    try:
        chat_id = chats.create(object(), trusted_webui=True)
        key = f"websocket:{chat_id}"
        session = chats._active_sessions[chat_id]
        session.add_message("user", "live history")
        result = await loop.execute_turn(TurnRequest(content="detached", session_key=key),
            _options=_TurnExecutionOptions(ephemeral=True))
        assert result.content == "detached answer"
        assert lease.manager.get_cached(key) is session
        assert [row["content"] for row in session.messages] == ["live history"]
        assert "live history" not in str(provider._provider.calls[0][0])
        assert loop.run_store.load(result.run_id) is None
    finally:
        chats.close()
        await loop.aclose()
        await lease.close()


async def test_disposal_during_cleanup_fence_rejects_queued_required_admission(engine_config: Config, monkeypatch) -> None:
    provider = ScriptedProvider([LLMResponse(content="must not run")])
    lease, loop = await _managed_loop(engine_config, provider)
    bus = MessageBus()
    chats = _temporary_chats(lease.manager, bus)
    release, waiting, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
    task = None
    cleanup = None
    try:
        from tests.session.test_p5_stage2_consumers import state

        owner = object()
        chat_id = chats.create(owner, trusted_webui=True)
        key = f"websocket:{chat_id}"
        before = state(lease.manager)

        async def pending_cleanup():
            waiting.set()
            await release.wait()

        cleanup = asyncio.create_task(pending_cleanup())
        loop._bus_cleanup_tasks[key] = {cleanup}
        await waiting.wait()
        real_wait = asyncio.wait

        async def observe_wait(tasks, **kwargs):
            if cleanup in tasks:
                queued.set()
            return await real_wait(tasks, **kwargs)

        monkeypatch.setattr(asyncio, "wait", observe_wait)
        inbound = InboundMessage(channel="websocket", sender_id="webui", chat_id=chat_id,
            content="queued private", require_existing_session=True)
        loop.approval_admission.consume = AsyncMock(wraps=loop.approval_admission.consume)
        task = asyncio.create_task(loop.execute_turn(TurnRequest(content=inbound.content, session_key=key),
            _options=_TurnExecutionOptions(inbound=inbound)))
        # Observe entry into the real cleanup wait, with no clock-dependent sleep.
        await asyncio.wait_for(queued.wait(), timeout=5)
        await chats.discard(owner, chat_id)
        control = await bus.consume_inbound()
        assert control.session_key == key
        release.set()
        result = await task
        assert result.error is not None
        assert "required session is not active" in result.error
        loop.approval_admission.consume.assert_not_awaited()
        assert provider._provider.calls == []
        assert loop.run_store.load(result.run_id) is None
        assert state(lease.manager) == before and lease.manager.get_cached(key) is None
    finally:
        release.set()
        if cleanup:
            await cleanup
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        chats.close()
        await loop.aclose()
        await lease.close()
