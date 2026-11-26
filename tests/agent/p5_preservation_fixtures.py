"""Copied-state graph for historical legacy turn preservation, never a turn owner."""

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kohaku.agent.loop import AgentLoop, _AdmittedTurnIO, _TurnExecutionOptions
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import DeliveryTarget, TurnRequest
from kohaku.bus.queue import MessageBus
from kohaku.config.schema import Config
from kohaku.cron.service import CronService
from kohaku.providers.base import GenerationSettings, LLMProvider, LLMResponse
from kohaku.runtime_composition import RuntimeComposition, RuntimeFactories
from kohaku.session.manager import Session, SessionManager
from kohaku.session.runtime_authority import BorrowedSessionAuthority, OfflineSessionOperator
from kohaku.session.webui_turns import WebuiTurnCoordinator


class _OwnedMCP:
    async def connect(self):
        return None

    async def aclose(self):
        return None


@pytest.fixture
async def _make_full_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Explicit positive-fixture admission; OS full denial is the outer effect fence."""
    monkeypatch.setattr("kohaku.migration_guard.require_runtime_ready", lambda: None)
    graphs = []

    async def acquire(root: Path, *, keys=(), borrowed=False):
        assert root == tmp_path
        config = Config()
        config.bind_source_path(root / "instance" / "config.json")
        workspace = root / "workspace"
        workspace.mkdir()
        config.agents.defaults.workspace = str(workspace)
        config.agents.defaults.model = "test-model"
        config.agents.defaults.model_preset = None
        config.agents.defaults.session_ttl_minutes = 0
        config.agents.defaults.idle_compaction.enabled = False
        config.tools.agent_browser.enabled = False
        config.tools.agent_device.enabled = False
        config.tools.desktop_use.enabled = False
        config.tools.restrict_to_workspace = True
        sessions_root = config.runtime_data_dir / "sessions"
        authority = None
        if borrowed:
            authority = BorrowedSessionAuthority(SessionManager(workspace, sessions_root=sessions_root))
        elif keys:
            unmanaged = SessionManager(workspace, sessions_root=sessions_root)
            for key in keys:
                unmanaged.save(Session(key), fsync=True)
            operator = await OfflineSessionOperator(workspace, copied_sessions_root=sessions_root).acquire()
            try:
                operator.manager.bootstrap_existing_sessions(offline_copy=True)
            finally:
                await operator.close()

        provider = MagicMock(spec=LLMProvider)
        provider.get_default_model.return_value = "test-model"
        provider.generation = GenerationSettings(max_tokens=4096)
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="Test title"))
        provider.supports_bound_conversation_state.return_value = False
        provider.can_resume_conversation_state.return_value = False

        def agent(cfg, bus, sessions, tools, mcp, service, delivery):
            return AgentLoop.from_config(
                cfg, bus, tool_registry=tools, provider=provider, session_manager=sessions,
                cron_service=service, turn_delivery_factory=delivery,
            )

        graph = RuntimeComposition.define(
            config, session_authority=authority,
            factories=RuntimeFactories(
                MessageBus, ToolRegistry, lambda *_: _OwnedMCP(),
                lambda report: CronService(report.workspace / "cron" / "jobs.json"), agent,
            ),
        )
        handle = await graph.acquire()
        loop = handle.agent
        unsubscribe = WebuiTurnCoordinator(
            bus=handle.bus, sessions=handle.sessions, schedule_background=loop.schedule_background,
        ).subscribe()
        graphs.append((graph, unsubscribe))
        assert loop.sessions is handle.sessions
        assert loop.tools is handle.tools
        if borrowed:
            assert handle.sessions is authority.manager
            assert (await authority.acquire()).close is None
        else:
            for key in keys:
                mode = handle.sessions.resolve_session_mode(key)
                assert mode.mode == "legacy"
                assert mode.binding is not None
        return loop

    try:
        yield acquire
    finally:
        for graph, unsubscribe in reversed(graphs):
            unsubscribe()
            await graph.aclose()


async def _execute_inbound(loop, msg, *, session_key=None, runtime=None, pending_queue=None):
    """Translate the old inbound fixture to one real, authorized execution."""
    key = session_key or msg.session_key_override or (
        msg.chat_id if msg.channel == "system" else msg.session_key
    )

    @asynccontextmanager
    async def queue_scope(effective_key):
        yield _AdmittedTurnIO(
            pending_queue=pending_queue,
            delivery=loop.turn_delivery_factory.create(msg, effective_key),
        )

    result = await loop.execute_turn(
        TurnRequest(
            content=msg.content, media=tuple(msg.media), sender_id=msg.sender_id,
            metadata=dict(msg.metadata), session_key=key, source=loop._inbound_turn_source(msg),
            route=DeliveryTarget(
                channel=msg.channel, chat_id=msg.chat_id,
                message_id=msg.metadata.get("message_id"),
            ),
        ),
        _options=_TurnExecutionOptions(
            inbound=msg, runtime=runtime, raise_errors=True,
            admission_scope=queue_scope if pending_queue is not None else None,
        ),
    )
    return result.outbound
