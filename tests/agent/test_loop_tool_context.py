import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.context import TranscriptInput
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import (
    RequestContext,
    bind_request_context,
    current_request_context,
    reset_request_context,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import RunPolicyContext
from nanobot.agent.turn import TurnContext as RunContext
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.turn_continuation import INTERNAL_CONTINUATION_META
from tests.agent.p6_admitted_fixtures import (
    AgentInputs,
    AsyncAdmittedFactory,
    SyncAdmittedFactory,
)
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.agent.p6_admitted_fixtures import admitted_sync as admitted_sync
from tests.agent.test_migration_engine import ScriptedProvider


class _ContextRecordingTool(Tool):
    name = "cron"
    description = "Record the scoped request context for this unit test."
    parameters = {"type": "object", "properties": {"action": {"type": "string"}},
                  "required": ["action"]}
    concurrency_safe = False

    def __init__(self) -> None:
        self.contexts: list[dict] = []
        self.runtimes: list[object] = []

    async def execute(self, **_kwargs) -> str:
        ctx = current_request_context()
        assert ctx is not None
        self.runtimes.append(ctx.runtime)
        self.contexts.append({
            "channel": ctx.channel,
            "chat_id": ctx.chat_id,
            "metadata": ctx.metadata,
            "session_key": ctx.session_key,
        })
        return "created"


def _build_loop(provider: LLMProvider, registries: list[ToolRegistry] | None = None):
    def build(inputs: AgentInputs) -> AgentLoop:
        if registries is not None:
            registries.append(inputs.tools)
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test-model", session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            session_ttl_minutes=0,
        )
    return build


def test_loop_registers_default_tools_in_injected_registry(
    tmp_path: Path, admitted_sync: SyncAdmittedFactory,
) -> None:
    provider = ScriptedProvider([])
    registries: list[ToolRegistry] = []
    loop = admitted_sync(tmp_path, _build_loop(provider, registries))
    registry = registries[0]

    assert loop.tools is registry
    assert registry.has("read_file")


def _config_for_loop(tmp_path: Path) -> Config:
    return Config.model_validate({"agents": {"defaults": {"workspace": str(tmp_path)}}})


def _provider_for_loop() -> LLMProvider:
    return ScriptedProvider([])


def _run_context(loop: AgentLoop, *, approval_granted: bool = False) -> RunContext:
    return RunContext(policy=RunPolicyContext(
        workspace=loop.workspace,
        context_manager=loop._fork_dependencies.resource_policy.context_manager,
        approval_granted=approval_granted,
    ))


def test_loop_from_config_requires_caller_owned_registry(tmp_path: Path) -> None:
    signature = inspect.signature(AgentLoop.from_config)

    with pytest.raises(TypeError, match="tool_registry"):
        signature.bind(_config_for_loop(tmp_path))


def test_loop_from_config_uses_caller_owned_registry(
    tmp_path: Path, admitted_sync: SyncAdmittedFactory,
) -> None:
    registries: list[ToolRegistry] = []

    def build(inputs: AgentInputs) -> AgentLoop:
        registry = inputs.tools
        registries.append(registry)
        return AgentLoop.from_config(
            inputs.config, bus=inputs.bus, tool_registry=registry,
            provider=_provider_for_loop(), session_manager=inputs.sessions,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
        )

    loop = admitted_sync(tmp_path, build, config=_config_for_loop(tmp_path))
    registry = registries[0]

    assert loop.tools is registry
    assert loop.tools.has("read_file")


@pytest.mark.asyncio
async def test_loop_binds_request_context_for_tool_execution(
    tmp_path: Path, admitted_async: AsyncAdmittedFactory,
) -> None:
    provider = ScriptedProvider([
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="call_1", name="cron", arguments={"action": "add"})],
        ),
        LLMResponse(content="done", tool_calls=[]),
    ])
    loop = await admitted_async(tmp_path, _build_loop(provider))
    cron = _ContextRecordingTool()
    original_cron = loop.tools.get("cron")
    assert original_cron is not None
    loop.tools.replace(cron)

    metadata = {"slack": {"thread_ts": "111.222", "channel_type": "channel"}}
    runtime = loop.llm_runtime()
    try:
        await loop._run_agent_loop(
            TranscriptInput(history=[], current_message=None),
            runtime=runtime,
            run_context=_run_context(loop, approval_granted=True),
            request_context=RequestContext(
                channel="slack",
                chat_id="C123",
                session_key="slack:C123:111.222",
                runtime=runtime,
                metadata=metadata,
            ),
        )
    finally:
        loop.tools.replace(original_cron)

    assert cron.contexts[-1] == {
        "channel": "slack",
        "chat_id": "C123",
        "metadata": metadata,
        "session_key": "slack:C123:111.222",
    }
    assert cron.runtimes[-1] is runtime


def test_request_context_nested_bind_restores_outer_context() -> None:
    outer = RequestContext(channel="slack", chat_id="outer", session_key="slack:outer")
    inner = RequestContext(channel="email", chat_id="inner", session_key="email:inner")

    outer_token = bind_request_context(outer)
    try:
        assert current_request_context() is outer
        inner_token = bind_request_context(inner)
        try:
            assert current_request_context() is inner
        finally:
            reset_request_context(inner_token)
        assert current_request_context() is outer
    finally:
        reset_request_context(outer_token)

    assert current_request_context() is None


@pytest.mark.asyncio
async def test_request_context_bindings_are_isolated_between_concurrent_tasks() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def observe(ctx: RequestContext, *, wait_first: bool) -> RequestContext | None:
        token = bind_request_context(ctx)
        try:
            if wait_first:
                entered.set()
                await release.wait()
            else:
                await entered.wait()
                release.set()
            await asyncio.sleep(0)
            return current_request_context()
        finally:
            reset_request_context(token)

    first = RequestContext(channel="feishu", chat_id="first", session_key="feishu:first")
    second = RequestContext(channel="telegram", chat_id="second", session_key="telegram:second")

    observed = await asyncio.gather(
        observe(first, wait_first=True),
        observe(second, wait_first=False),
    )

    assert observed == [first, second]
    assert current_request_context() is None


@pytest.mark.asyncio
async def test_agent_loop_restores_outer_request_context_after_runner_exception(
    tmp_path: Path,
    admitted_async: AsyncAdmittedFactory,
) -> None:
    provider = ScriptedProvider([])
    loop = await admitted_async(tmp_path, _build_loop(provider))
    outer = RequestContext(channel="test", chat_id="outer", session_key="test:outer")
    runtime = loop.llm_runtime()

    async def fail_run(spec):
        current = current_request_context()
        assert current is not None
        assert spec.runtime is runtime
        assert current.runtime is runtime
        assert current.channel == "slack"
        assert current.chat_id == "C123"
        assert current.session_key == "slack:C123:111.222"
        assert current.original_user_text == "  unchanged user text  "
        raise RuntimeError("runner failed")

    loop.runner.run = AsyncMock(side_effect=fail_run)
    outer_token = bind_request_context(outer)
    try:
        with pytest.raises(RuntimeError, match="runner failed"):
            await loop._run_agent_loop(
                TranscriptInput(history=[], current_message=None),
                runtime=runtime,
                run_context=_run_context(loop),
                request_context=RequestContext(
                    channel="slack",
                    chat_id="C123",
                    session_key="slack:C123:111.222",
                    original_user_text="  unchanged user text  ",
                    runtime=runtime,
                ),
            )
        assert current_request_context() is outer
    finally:
        reset_request_context(outer_token)

    assert current_request_context() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, "  original user text  "),
        ({INTERNAL_CONTINUATION_META: True}, None),
    ],
)
async def test_process_message_captures_original_text_before_restore(
    tmp_path: Path,
    metadata: dict,
    expected: str | None,
    admitted_async: AsyncAdmittedFactory,
) -> None:
    provider = ScriptedProvider([])
    loop = await admitted_async(tmp_path, _build_loop(provider))
    runtime = loop.llm_runtime()
    seen: list[tuple[str | None, object]] = []

    async def stop_after_capture(ctx) -> str:
        seen.append((ctx.original_user_text, ctx.runtime))
        raise RuntimeError("captured before restore")

    loop._restore_turn = stop_after_capture  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="captured before restore"):
        await loop._process_turn(
            InboundMessage(
                channel="slack",
                sender_id="user",
                chat_id="C123",
                content="  original user text  ",
                metadata=metadata,
            ),
            runtime=runtime,
        )

    assert seen == [(expected, runtime)]


def test_tool_context_construction_rejects_pending_startup_receipt(tmp_path: Path) -> None:
    from nanobot.agent.memory_recovery_startup_admission import StartupReceiptAdmissionError
    from nanobot.bus.queue import MessageBus

    workspace = tmp_path / "workspace"
    pending = workspace / "memory" / ".consolidation.pending.json"
    pending.parent.mkdir(parents=True)
    pending.write_text("{broken", encoding="utf-8")
    with pytest.raises(StartupReceiptAdmissionError):
        AgentLoop(
            bus=MessageBus(), provider=ScriptedProvider([]), workspace=workspace,
            sessions_root=tmp_path / "sessions",
        )


async def test_unapproved_context_recording_call_is_not_dispatched(
    tmp_path: Path, admitted_async: AsyncAdmittedFactory,
) -> None:
    provider = ScriptedProvider([LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="unapproved", name="cron", arguments={"action": "add"})],
    )])
    loop = await admitted_async(tmp_path, _build_loop(provider))
    recording = _ContextRecordingTool()
    original_cron = loop.tools.get("cron")
    assert original_cron is not None
    loop.tools.replace(recording)
    runtime = loop.llm_runtime()
    try:
        result = await loop._run_agent_loop(
            TranscriptInput(history=[], current_message=None), runtime=runtime,
            run_context=_run_context(loop),
            request_context=RequestContext(channel="slack", chat_id="C123", runtime=runtime),
        )
    finally:
        loop.tools.replace(original_cron)
    assert result.stop_reason == "approval_required"
    assert recording.contexts == [] and len(provider.calls) == 1
    assert current_request_context() is None
