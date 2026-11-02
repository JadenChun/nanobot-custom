"""Signal driven cron shutdown retains durable source ownership until settlement."""

from __future__ import annotations

import asyncio
import signal
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import (
    DeliveryTarget,
    HistoryMode,
    ScheduledTurnLink,
    TurnRequest,
    TurnSource,
)
from nanobot.cli.gateway_runtime import (
    _close_gateway_runtime,
    _install_gateway_shutdown_handlers,
)
from nanobot.config.schema import Config, GatewayCronConfig
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.definition_store import CronDefinitionStore
from nanobot.cron.gateway import GatewayCron
from nanobot.cron.runtime import CronWorkFailedError
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.manager import Session, SessionManager
from nanobot.session.source import validate_execution_links


class _FakeLoop:
    def __init__(self) -> None:
        self.handlers: dict[int, tuple[Any, tuple[Any, ...]]] = {}
        self.removed: list[int] = []

    def add_signal_handler(self, signum: int, callback: Any, *args: Any) -> None:
        self.handlers[int(signum)] = (callback, args)

    def remove_signal_handler(self, signum: int) -> bool:
        self.removed.append(int(signum))
        self.handlers.pop(int(signum), None)
        return True

    def fire(self, signum: int) -> None:
        callback, args = self.handlers[int(signum)]
        callback(*args)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000

    def wall_time_ms(self) -> int:
        return self.now

    def monotonic_time(self) -> float:
        return 0.0

    async def sleep(self, _seconds: float) -> None:
        await asyncio.sleep(0)


class _ReceiptChannel:
    receipt_transport_ready = True

    async def send_text_with_receipt(
        self,
        target: str,
        messages: tuple[str, ...] | list[str],
        *,
        action_id: str,
        request_digest: str,
    ) -> Any:
        from nanobot.channels.delivery_contract import ChannelTextPartReceipt, ChannelTextReceipt

        del messages
        return ChannelTextReceipt(
            channel="telegram",
            target=target,
            action_id=action_id,
            request_digest=request_digest,
            parts=(ChannelTextPartReceipt(0, 0, 0, 1, 1),),
        )

    def get_channel(self, name: str) -> _ReceiptChannel | None:
        return self if name == "telegram" else None


class _ScriptedProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(provider_name="shutdown-test")
        self.responses = deque([
            LLMResponse(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest("probe-1", "event_gate", {"label": "local"})],
            ),
            LLMResponse(content="completed local turn"),
            LLMResponse(content="next admission succeeded"),
        ])
        self.calls: list[Any] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        self.calls.append((messages, tools, model))
        return self.responses.popleft()

    def get_default_model(self) -> str:
        return "shutdown-test-model"


@tool_parameters({
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
})
class _EventGateTool(Tool):
    def __init__(self, *, resist_cancellation: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.resist_cancellation = resist_cancellation

    @property
    def name(self) -> str:
        return "event_gate"

    @property
    def description(self) -> str:
        return "Wait at a deterministic local test gate."

    async def execute(self, *, label: str) -> str:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            if self.resist_cancellation:
                await self.release.wait()
            raise
        return f"{label} finished"


def _gateway(tmp_path, tool: _EventGateTool) -> tuple[GatewayCron, AgentLoop, _Clock, _ScriptedProvider]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config()
    config.bind_source_path(tmp_path / "instance" / "config.json")
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "shutdown-test-model"
    config.agents.defaults.model_preset = None
    config.agents.defaults.session_ttl_minutes = 0

    clock = _Clock()
    definition_store = CronDefinitionStore(
        CronActionStore(workspace / "cron", namespace="shutdown-test"),
        commit_clock_ms=clock.wall_time_ms,
    )
    service = CronService(
        workspace / "cron" / "jobs.json", definition_store=definition_store,
    )
    sessions = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    provider = _ScriptedProvider()
    tools = ToolRegistry()
    tools.register(tool)
    agent = AgentLoop.from_config(
        config,
        provider=provider,
        tool_registry=tools,
        cron_service=service,
        session_manager=sessions,
    )
    channels = _ReceiptChannel()
    gateway = GatewayCron(
        GatewayCronConfig(
            enabled=True,
            poll_interval_s=0.05,
            readiness_timeout_s=1,
            drain_timeout_s=0.01,
            codex_quota_retry=False,
        ),
        service,
        agent,
        sessions,
        channels,
        workspace,
        tmp_path / "instance",
        clock=clock,
    )
    gateway.service.add_job(
        "local shutdown probe",
        CronSchedule(kind="every", every_ms=100),
        "Run one local tool call",
        deliver=True,
        channel="telegram",
        to="test-chat",
    )
    job = gateway.service.list_jobs()[0]
    key = f"cron:{job.id}"
    seed = Session(key)
    sessions.save(seed)
    sessions._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=seed.revision,
        expected_generation=seed.generation,
        expected_incarnation=seed.incarnation,
    )
    return gateway, agent, clock, provider


async def _wait_for(predicate) -> None:
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


async def _wait_event(event: asyncio.Event) -> None:
    async with asyncio.timeout(1):
        await event.wait()


def _install_signals(gateway: GatewayCron, task: asyncio.Task[Any]):
    loop = _FakeLoop()
    shutdown = asyncio.Event()
    restore = _install_gateway_shutdown_handlers(
        loop,
        shutdown,
        [task],
        lambda _message: None,
        on_first_sigint=gateway.begin_graceful_shutdown,
        on_force=gateway.force_shutdown,
        on_sigterm=gateway.request_bounded_shutdown,
    )
    return loop, shutdown, restore


def _noop() -> None:
    return None


def test_signal_handler_fallback_restores_previous_handlers(monkeypatch) -> None:
    class _FallbackLoop(_FakeLoop):
        def add_signal_handler(self, _signum: int, _callback: Any, *_args: Any) -> None:
            raise NotImplementedError

    previous = {int(signal.SIGINT): object(), int(signal.SIGTERM): object()}
    calls: list[tuple[int, Any]] = []
    monkeypatch.setattr(signal, "getsignal", lambda signum: previous[int(signum)])
    monkeypatch.setattr(signal, "signal", lambda signum, handler: calls.append((int(signum), handler)))

    restore = _install_gateway_shutdown_handlers(
        _FallbackLoop(), asyncio.Event(), [], _noop,
    )
    assert len(calls) == 2 and all(callable(handler) for _signum, handler in calls)

    restore()

    assert calls[-2:] == [
        (int(signal.SIGINT), previous[int(signal.SIGINT)]),
        (int(signal.SIGTERM), previous[int(signal.SIGTERM)]),
    ]


async def _start_due_turn(gateway: GatewayCron, clock: _Clock) -> asyncio.Task[None]:
    task = asyncio.create_task(gateway.run())
    await _wait_for(
        lambda: gateway.runtime._accepting and gateway.runtime._timer_task is not None
    )
    clock.now = 1_100
    return task


async def test_first_sigint_joins_admitted_source_turn_past_normal_drain_budget(tmp_path) -> None:
    tool = _EventGateTool()
    gateway, agent, clock, provider = _gateway(tmp_path, tool)
    task = await _start_due_turn(gateway, clock)
    close: asyncio.Task[None] | None = None
    restore = _noop
    try:
        loop, shutdown, restore = _install_signals(gateway, task)
        await _wait_event(tool.started)
        loop.fire(signal.SIGINT)
        assert shutdown.is_set()
        assert gateway.runtime._accepting is False
        with pytest.raises(Exception, match="scheduler ingress fenced"):
            await gateway.runtime.scheduler.wake_once()

        close = asyncio.create_task(gateway.aclose())
        await asyncio.sleep(0.04)
        assert not close.done()
        assert not tool.cancelled.is_set()
        assert len(provider.calls) == 1

        tool.release.set()
        await asyncio.wait_for(close, timeout=1)
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

        source = agent.sessions.source_backend(f"cron:{gateway.service.list_jobs()[0].id}").source
        snapshot = source.load_source_snapshot(f"cron:{gateway.service.list_jobs()[0].id}")
        assert not validate_execution_links(snapshot.events)
        assert "execution_lease" not in snapshot.header["metadata"]
        assert "execution_recovery_required" not in snapshot.header["metadata"]
        assert gateway.runtime.scheduler._lease is None

        job = gateway.service.list_jobs()[0]
        await agent.execute_silent_cron_turn(TurnRequest(
            content=job.payload.message,
            source=TurnSource.CRON,
            session_key=f"cron:{job.id}",
            route=DeliveryTarget(channel="telegram", chat_id="test-chat"),
            run_id="next-admission",
            history_mode=HistoryMode.FRESH,
            scheduled_link=ScheduledTurnLink(
                job_id=job.id,
                job_name=job.name,
                instruction=job.payload.message,
                visible_session_key="telegram:test-chat",
                occurrence_id="next-local-occurrence",
            ),
        ))
        assert len(provider.calls) == 3
        assert not validate_execution_links(source.load_source_snapshot(f"cron:{job.id}").events)
    finally:
        tool.release.set()
        restore()
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        else:
            await asyncio.gather(gateway.aclose(), return_exceptions=True)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()


async def test_second_sigint_cancels_owned_work_and_retains_recovery_fence(tmp_path) -> None:
    tool = _EventGateTool(resist_cancellation=True)
    gateway, agent, clock, provider = _gateway(tmp_path, tool)
    task = await _start_due_turn(gateway, clock)
    restore = _noop
    close: asyncio.Task[None] | None = None
    job = gateway.service.list_jobs()[0]
    key = f"cron:{job.id}"
    try:
        loop, _shutdown, restore = _install_signals(gateway, task)
        await _wait_event(tool.started)
        loop.fire(signal.SIGINT)
        loop.fire(signal.SIGINT)
        await _wait_event(tool.cancelled)

        close = asyncio.create_task(gateway.aclose())
        from nanobot.runtime_errors import RuntimeOwnedWorkError

        with pytest.raises(RuntimeOwnedWorkError):
            await close
        close = None
        assert gateway.runtime.scheduler._lease is not None
        assert agent._closed is False

        tool.release.set()
        await _wait_for(lambda: not gateway.runtime._tasks)
        await asyncio.wait_for(gateway.aclose(), timeout=1)

        source = agent.sessions.source_backend(key).source
        snapshot = source.load_source_snapshot(key)
        assert snapshot.header["metadata"].get("execution_recovery_required") is True
        calls_before_retry = len(provider.calls)
        with pytest.raises(RuntimeError, match="Source session requires recovery"):
            await agent.execute_silent_cron_turn(TurnRequest(
                content=job.payload.message,
                source=TurnSource.CRON,
                session_key=key,
                route=DeliveryTarget(channel="telegram", chat_id="test-chat"),
                run_id="must-not-dispatch",
                history_mode=HistoryMode.FRESH,
                scheduled_link=ScheduledTurnLink(
                    job_id=job.id,
                    job_name=job.name,
                    instruction=job.payload.message,
                    visible_session_key="telegram:test-chat",
                    occurrence_id="recovery-blocked-occurrence",
                ),
            ))
        assert len(provider.calls) == calls_before_retry == 1
    finally:
        tool.release.set()
        restore()
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        if gateway.runtime._close_task is None or gateway.runtime._close_task.done():
            await asyncio.gather(gateway.aclose(), return_exceptions=True)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()


async def test_sigterm_only_uses_bounded_close_policy(tmp_path) -> None:
    tool = _EventGateTool()
    gateway, agent, clock, _provider = _gateway(tmp_path, tool)
    task = await _start_due_turn(gateway, clock)
    close: asyncio.Task[None] | None = None
    restore = _noop
    try:
        loop, _shutdown, restore = _install_signals(gateway, task)
        await _wait_event(tool.started)
        loop.fire(signal.SIGTERM)
        close = asyncio.create_task(gateway.aclose())
        await _wait_event(tool.cancelled)
        await asyncio.wait_for(close, timeout=1)
        close = None
        assert gateway.runtime.scheduler._lease is None
        assert task.done()
    finally:
        tool.release.set()
        restore()
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        else:
            await asyncio.gather(gateway.aclose(), return_exceptions=True)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()


async def test_sigterm_during_graceful_wait_switches_to_forced_drain(tmp_path) -> None:
    tool = _EventGateTool()
    gateway, agent, clock, _provider = _gateway(tmp_path, tool)
    task = await _start_due_turn(gateway, clock)
    close: asyncio.Task[None] | None = None
    restore = _noop
    try:
        loop, _shutdown, restore = _install_signals(gateway, task)
        await _wait_event(tool.started)
        loop.fire(signal.SIGINT)
        close = asyncio.create_task(gateway.aclose())
        await asyncio.sleep(0.02)
        assert not close.done()

        loop.fire(signal.SIGTERM)
        await _wait_event(tool.cancelled)
        await asyncio.wait_for(close, timeout=1)
        close = None
        assert gateway.runtime.scheduler._lease is None
    finally:
        tool.release.set()
        restore()
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        else:
            await asyncio.gather(gateway.aclose(), return_exceptions=True)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()


async def test_joined_cron_failure_still_finishes_shared_gateway_teardown() -> None:
    events: list[str] = []
    source_error = OSError("synthetic cron source failure")
    cleanup_error = RuntimeError("synthetic shared cleanup failure")
    cron_error = CronWorkFailedError("admitted cron turn joined with failure")

    class Composition:
        async def aclose(self) -> None:
            events.append("composition-close")
            try:
                raise source_error
            except OSError as exc:
                raise cron_error from exc

    class Bus:
        async def drain(self) -> None:
            events.append("bus-drain")
            raise cleanup_error

    handle = SimpleNamespace(
        agent=SimpleNamespace(stop=lambda: events.append("agent-stop")),
        bus=Bus(),
    )
    with pytest.raises(ExceptionGroup) as caught:
        await _close_gateway_runtime(
            handle,
            Composition(),
            (),
            disconnect_webui=lambda: events.append("disconnect-webui"),
            restore=lambda: events.append("restore"),
        )

    assert caught.value.exceptions == (cron_error, cleanup_error)
    assert cron_error.__cause__ is source_error
    assert events == [
        "agent-stop", "disconnect-webui", "composition-close", "bus-drain", "restore",
    ]


async def test_owned_work_failure_retains_shared_gateway_dependencies() -> None:
    from nanobot.runtime_errors import RuntimeOwnedWorkError

    events: list[str] = []
    owned_error = RuntimeOwnedWorkError("cron work still owns shared resources")

    class Composition:
        async def aclose(self) -> None:
            events.append("composition-close")
            raise owned_error

    class Bus:
        async def drain(self) -> None:
            events.append("bus-drain")

    handle = SimpleNamespace(
        agent=SimpleNamespace(stop=lambda: events.append("agent-stop")),
        bus=Bus(),
    )
    with pytest.raises(RuntimeOwnedWorkError) as caught:
        await _close_gateway_runtime(
            handle,
            Composition(),
            (),
            disconnect_webui=lambda: events.append("disconnect-webui"),
            restore=lambda: events.append("restore"),
        )

    assert caught.value is owned_error
    assert events == ["agent-stop", "disconnect-webui", "composition-close", "restore"]


async def test_cron_work_failure_preserves_source_error_after_releasing_lease(tmp_path) -> None:
    tool = _EventGateTool()
    gateway, agent, clock, provider = _gateway(tmp_path, tool)
    source_error = RuntimeError("synthetic provider/source failure")

    async def fail_source(_request: TurnRequest):
        raise source_error

    agent.execute_silent_cron_turn = fail_source
    task = await _start_due_turn(gateway, clock)
    try:
        with pytest.raises(RuntimeError, match="provider/source failure") as run_failure:
            await asyncio.wait_for(task, timeout=1)
        assert run_failure.value is source_error

        with pytest.raises(CronWorkFailedError) as close_failure:
            await gateway.aclose()
        assert close_failure.value.__cause__ is source_error
        assert gateway.runtime.scheduler._lease is None
        assert provider.calls == []
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()
