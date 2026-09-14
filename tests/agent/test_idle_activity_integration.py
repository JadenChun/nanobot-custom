"""Foreground idle-activity admission stays dormant until explicitly injected."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from agent import test_migration_engine as engine_fixtures
from agent.test_migration_engine import ScriptedProvider
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import RunStatus, TurnSource
from nanobot.bus.events import InboundMessage
from nanobot.providers.base import LLMResponse
from nanobot.session.recovery import RECOVERY_INBOUND_METADATA_KEY

engine_config = engine_fixtures.engine_config


def upgrade(loop: AgentLoop, key: str = "cli:idle"):
    session = loop.sessions.get_or_create(key)
    loop.sessions.save(session)
    backend = loop.sessions._jsonl_store
    snapshot = backend.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    loop.sessions.invalidate(key)
    return backend, snapshot


class _IdleIntegration:
    enabled = True

    def __init__(self, backend, *, failure: BaseException | None = None):
        self.backend = backend
        self.failure = failure
        self.calls: list[str] = []
        self.persisted_revisions: list[int] = []

    async def foreground_admitted(self, session_key: str) -> None:
        self.calls.append(session_key)
        snapshot = self.backend.source.load_source_snapshot(session_key)
        self.persisted_revisions.append(snapshot.header["metadata"]["user_activity_revision"])
        if self.failure is not None:
            raise self.failure


async def test_injected_activity_is_durable_before_callback_and_model(engine_config):
    provider = ScriptedProvider([LLMResponse(content="done")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend, _ = upgrade(loop)
    integration = _IdleIntegration(backend)
    loop._idle_integration = integration
    try:
        result = await loop._execute_direct(
            "hello", session_key="cli:idle", source=TurnSource.DIRECT,
        )
        assert result.status is RunStatus.COMPLETED
        assert integration.calls == ["cli:idle"]
        assert integration.persisted_revisions == [1]
        snapshot = backend.source.load_source_snapshot("cli:idle")
        assert snapshot.header["metadata"]["user_activity_revision"] == 1
        assert len(provider.calls) == 1
    finally:
        await loop.aclose()


async def test_disabled_or_partial_integration_does_not_write_activity(engine_config):
    provider = ScriptedProvider([LLMResponse(content="done"), LLMResponse(content="done")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend, _ = upgrade(loop)
    callback = AsyncMock()
    loop._idle_integration = type("Partial", (), {"enabled": False, "foreground_admitted": callback})()
    try:
        await loop._execute_direct("disabled", session_key="cli:idle")
        assert callback.await_count == 0
        snapshot = backend.source.load_source_snapshot("cli:idle")
        assert "user_activity_revision" not in snapshot.header["metadata"]

        loop._idle_integration = type("Partial", (), {"enabled": True})()
        await loop._execute_direct("partial", session_key="cli:idle")
        assert "user_activity_revision" not in backend.source.load_source_snapshot(
            "cli:idle"
        ).header["metadata"]
    finally:
        await loop.aclose()


@pytest.mark.parametrize("source", [TurnSource.DIRECT, TurnSource.GATEWAY, TurnSource.SDK, TurnSource.API])
async def test_foreground_sources_are_admitted_and_automation_is_excluded(engine_config, source):
    provider = ScriptedProvider([LLMResponse(content="done") for _ in range(5)])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend, _ = upgrade(loop)
    integration = _IdleIntegration(backend)
    loop._idle_integration = integration
    try:
        result = await loop._execute_direct(
            source.value, session_key="cli:idle", source=source,
        )
        assert result.error is None
        assert integration.calls == ["cli:idle"]

        for excluded, sender_id in (
            (TurnSource.CRON, "user"),
            (TurnSource.HEARTBEAT, "user"),
            (TurnSource.SYSTEM_COMPAT, "system"),
            (TurnSource.DIRECT, "subagent"),
        ):
            await loop._execute_direct(
                excluded.value,
                session_key="cli:idle",
                source=excluded,
                sender_id=sender_id,
            )
        assert integration.calls == ["cli:idle"]
        assert backend.source.load_source_snapshot("cli:idle").header["metadata"][
            "user_activity_revision"
        ] == 1
    finally:
        await loop.aclose()


async def test_activity_callback_failure_or_cancellation_prevents_model_work(engine_config):
    provider = ScriptedProvider([LLMResponse(content="must not run")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend, _ = upgrade(loop)
    integration = _IdleIntegration(backend, failure=RuntimeError("callback failed"))
    loop._idle_integration = integration
    try:
        with pytest.raises(RuntimeError, match="callback failed"):
            await loop._execute_direct("blocked", session_key="cli:idle")
        assert not provider.calls
        assert backend.source.load_source_snapshot("cli:idle").header["metadata"][
            "user_activity_revision"
        ] == 1

        integration.failure = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await loop._execute_direct("cancelled", session_key="cli:idle")
        assert len(provider.calls) == 0
    finally:
        await loop.aclose()


async def test_rejected_approvals_do_not_record_foreground_activity(engine_config, monkeypatch):
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend, _ = upgrade(loop)
    integration = _IdleIntegration(backend)
    loop._idle_integration = integration
    try:
        await loop.approval_admission.store(
            session_key="cli:idle", channel="cli", chat_id="direct", summary="risky"
        )
        negative = await loop._execute_direct("no", session_key="cli:idle")
        assert negative.stop_reason == "cancelled"

        await loop.approval_admission.store(
            session_key="cli:idle", channel="cli", chat_id="direct", summary="old"
        )
        consume = loop.approval_admission.consume

        async def replace_before_consume(snapshot):
            await loop.approval_admission.store(
                session_key="cli:idle", channel="cli", chat_id="direct", summary="new"
            )
            return await consume(snapshot)

        monkeypatch.setattr(loop.approval_admission, "consume", replace_before_consume)
        stale = await loop._execute_direct("yes", session_key="cli:idle")
        assert stale.stop_reason == "approval_stale"
        assert integration.calls == []
        assert "user_activity_revision" not in backend.source.load_source_snapshot(
            "cli:idle"
        ).header["metadata"]
        assert provider.calls == []
    finally:
        await loop.aclose()


@pytest.mark.parametrize(
    "message",
    [
        InboundMessage(
            channel="system", chat_id="idle", sender_id="system", content="system",
            input_role="user",
        ),
        InboundMessage(
            channel="websocket", chat_id="idle", sender_id="system:recovery", content="recovered",
            metadata={RECOVERY_INBOUND_METADATA_KEY: "recovery"}, input_role="user",
        ),
    ],
)
async def test_system_origin_and_recovery_do_not_record_activity(engine_config, message):
    provider = ScriptedProvider([LLMResponse(content="done")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = message.session_key
    backend, _ = upgrade(loop, key)
    integration = _IdleIntegration(backend)
    loop._idle_integration = integration
    try:
        await loop._dispatch(message)
        assert integration.calls == []
        assert "user_activity_revision" not in backend.source.load_source_snapshot(
            key
        ).header["metadata"]
    finally:
        await loop.aclose()


async def test_activity_identity_is_removed_when_republished_dispatch_is_cancelled(engine_config):
    loop = AgentLoop.from_config(
        engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([])
    )
    entered = asyncio.Event()
    message = InboundMessage(
        channel="telegram", chat_id="idle", sender_id="user", content="queued",
    )

    async def blocked_execute(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    loop.execute_turn = blocked_execute  # type: ignore[method-assign]
    loop._remember_admitted_activity_message(message)
    task = asyncio.create_task(loop._dispatch(message))
    try:
        await entered.wait()
        assert loop._admitted_activity_messages == {}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert loop._admitted_activity_messages == {}
    finally:
        if not task.done():
            task.cancel()
        await loop.aclose()


async def test_activity_identity_is_not_created_when_pending_enqueue_fails(engine_config):
    loop = AgentLoop.from_config(
        engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([])
    )
    key = "cli:idle"
    pending = asyncio.Queue(maxsize=1)
    pending.put_nowait(InboundMessage(
        channel="cli", chat_id="idle", sender_id="user", content="already queued",
    ))
    loop._pending_queues[key] = pending
    dispatched = asyncio.Event()

    async def observe_dispatch(_message):
        dispatched.set()

    loop._dispatch = observe_dispatch  # type: ignore[method-assign]
    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(InboundMessage(
            channel="cli", chat_id="idle", sender_id="user", content="overflow",
        ))
        await asyncio.wait_for(dispatched.wait(), timeout=2)
        assert loop._admitted_activity_messages == {}
    finally:
        loop.stop()
        await asyncio.wait_for(run_task, timeout=2)


async def test_activity_identity_clears_when_cleanup_cannot_republish(engine_config):
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingProvider(ScriptedProvider):
        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls.append((messages, tools, model))
            if len(self.calls) == 1:
                first_started.set()
                await release_first.wait()
            assert self.responses, "unexpected extra provider request"
            return self.responses.popleft()

    provider = BlockingProvider([LLMResponse(content="first")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend, _ = upgrade(loop, "telegram:idle")
    integration = _IdleIntegration(backend)
    loop._idle_integration = integration
    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(InboundMessage(
            channel="telegram", chat_id="idle", sender_id="user", content="first",
        ))
        await asyncio.wait_for(first_started.wait(), timeout=2)
        await loop.bus.publish_inbound(InboundMessage(
            channel="telegram", chat_id="idle", sender_id="user", content="queued",
        ))
        while len(integration.calls) < 2:
            await asyncio.sleep(0)
        assert len(loop._admitted_activity_messages) == 1

        def fail_republish(_message):
            raise RuntimeError("republish failed")

        loop.bus.publish_inbound_nowait = fail_republish
        release_first.set()
        while "telegram:idle" not in loop._pending_republish:
            await asyncio.sleep(0)
        assert loop._admitted_activity_messages == {}
        assert len(loop._pending_republish["telegram:idle"]) == 1
    finally:
        release_first.set()
        loop.stop()
        await asyncio.wait_for(run_task, timeout=2)
        await loop.aclose()


async def test_activity_identity_clears_on_shutdown(engine_config):
    loop = AgentLoop.from_config(
        engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([])
    )
    message = InboundMessage(
        channel="telegram", chat_id="idle", sender_id="user", content="queued",
    )
    loop._remember_admitted_activity_message(message)
    assert loop._admitted_activity_messages
    await loop.aclose()
    assert loop._admitted_activity_messages == {}
    assert not hasattr(message, "_nanobot_activity_admitted")


async def test_pending_foreground_followup_republishes_without_duplicate_activity(engine_config):
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingProvider(ScriptedProvider):
        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls.append((messages, tools, model))
            if len(self.calls) == 1:
                first_started.set()
                await release_first.wait()
            assert self.responses, "unexpected extra provider request"
            return self.responses.popleft()

    provider = BlockingProvider([LLMResponse(content="first"), LLMResponse(content="second")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend, _ = upgrade(loop, "telegram:idle")
    integration = _IdleIntegration(backend)
    loop._idle_integration = integration
    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(InboundMessage(
            channel="telegram", chat_id="idle", sender_id="user", content="first",
        ))
        await asyncio.wait_for(first_started.wait(), timeout=2)
        await loop.bus.publish_inbound(InboundMessage(
            channel="telegram", chat_id="idle", sender_id="user", content="queued",
        ))
        while len(integration.calls) < 2:
            await asyncio.sleep(0)
        assert integration.calls == ["telegram:idle", "telegram:idle"]
        assert len(loop._admitted_activity_messages) == 1
        release_first.set()
        while len(provider.calls) < 2:
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        assert integration.calls == ["telegram:idle", "telegram:idle"]
        assert integration.persisted_revisions == [1, 2]
        assert len(provider.calls) == 2
        assert loop._admitted_activity_messages == {}
    finally:
        release_first.set()
        loop.stop()
        await asyncio.wait_for(run_task, timeout=2)
        await loop.aclose()


@pytest.mark.parametrize("failure_kind", ["runtime", "cancelled"])
async def test_pending_activity_commit_survives_callback_failure_on_republish(
    engine_config, failure_kind,
):
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingProvider(ScriptedProvider):
        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls.append((messages, tools, model))
            if len(self.calls) == 1:
                first_started.set()
                await release_first.wait()
            assert self.responses, "unexpected extra provider request"
            return self.responses.popleft()

    class FailsOnPendingCallback(_IdleIntegration):
        async def foreground_admitted(self, session_key: str) -> None:
            self.calls.append(session_key)
            snapshot = self.backend.source.load_source_snapshot(session_key)
            self.persisted_revisions.append(
                snapshot.header["metadata"]["user_activity_revision"]
            )
            if len(self.calls) == 2:
                if failure_kind == "runtime":
                    raise RuntimeError("pending callback failed")
                raise asyncio.CancelledError()

    provider = BlockingProvider([
        LLMResponse(content="first"),
        LLMResponse(content="second"),
        LLMResponse(content="third"),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend, _ = upgrade(loop, "telegram:idle")
    integration = FailsOnPendingCallback(backend)
    loop._idle_integration = integration
    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(InboundMessage(
            channel="telegram", chat_id="idle", sender_id="user", content="first",
        ))
        await asyncio.wait_for(first_started.wait(), timeout=2)
        await loop.bus.publish_inbound(InboundMessage(
            channel="telegram", chat_id="idle", sender_id="user", content="queued",
        ))
        while len(integration.calls) < 2:
            await asyncio.sleep(0)
        assert integration.persisted_revisions == [1, 2]
        assert len(loop._admitted_activity_messages) == 1

        release_first.set()
        while len(provider.calls) < 2 or loop._pending_queues:
            await asyncio.sleep(0)
        assert integration.calls == ["telegram:idle", "telegram:idle"]
        assert integration.persisted_revisions == [1, 2]
        assert loop._admitted_activity_messages == {}

        await loop.bus.publish_inbound(InboundMessage(
            channel="telegram", chat_id="idle", sender_id="user", content="unrelated",
        ))
        while len(integration.calls) < 3:
            await asyncio.sleep(0)
        assert integration.calls == ["telegram:idle"] * 3
        assert integration.persisted_revisions == [1, 2, 3]
        while len(provider.calls) < 3:
            await asyncio.sleep(0)
        assert len(provider.calls) == 3
        assert backend.source.load_source_snapshot("telegram:idle").header["metadata"][
            "user_activity_revision"
        ] == 3
        assert loop._admitted_activity_messages == {}
    finally:
        release_first.set()
        loop.stop()
        await asyncio.wait_for(run_task, timeout=2)
        await loop.aclose()
