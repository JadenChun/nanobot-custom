from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import RunStatus, TurnRequest
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.runtime_composition import (
    RuntimeComposition,
    RuntimePreflightError,
    RuntimeState,
    runtime_preflight,
)
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.pending_followups import mint_pending_followup_id, record_pending_followup
from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY
from nanobot.session.recovery import RecoveryCoordinator
from nanobot.session.source_store import SourceSessionStore


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse], *, block_first: bool = False) -> None:
        super().__init__(provider_name="local-runtime-test")
        self.responses = list(responses)
        self.requests: list[tuple[object, ...]] = []
        self.block_first = block_first
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.first_cancelled = asyncio.Event()
        self.followup_seen = asyncio.Event()

    async def chat(
        self, messages: list[dict[str, object]], tools=None, model=None, **kwargs: object,
    ) -> LLMResponse:
        self.requests.append(tuple(messages))
        call_number = len(self.requests)
        if call_number == 1 and self.block_first:
            self.first_started.set()
            try:
                await self.release_first.wait()
            except asyncio.CancelledError:
                self.first_cancelled.set()
                raise
        if call_number > 1 and any(
            "queued follow-up marker" in str(message.get("content") or "")
            for message in messages
        ):
            self.followup_seen.set()
        assert self.responses, "unexpected extra provider request"
        return self.responses.pop(0)

    def estimate_prompt_tokens(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
    ) -> tuple[int, str]:
        estimate = sum(len(str(message.get("content") or "")) for message in messages)
        return max(1, estimate), "scripted"

    def get_default_model(self) -> str:
        return "local-runtime-test"


def _config(tmp_path: Path) -> Config:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "local-runtime-test"
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
    config.bind_source_path(runtime_dir / "config.json")
    return config


def test_default_composition_constructs_real_tools_and_reopens_one_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = _config(tmp_path)
        note = config.workspace_path / "note.txt"
        note.write_text("workspace evidence", encoding="utf-8")
        provider = _ScriptedProvider([
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(
                    id="read-note", name="read_file", arguments={"path": "note.txt"},
                )],
            ),
            LLMResponse(content="read complete", tool_calls=[]),
        ])
        monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)

        first = RuntimeComposition.define(config)
        assert first.state is RuntimeState.NEW
        handle = await first.acquire()
        try:
            read_file = handle.tools.get("read_file")
            assert read_file is not None
            result = await handle.agent.execute_turn(TurnRequest(
                content="Read note.txt", session_key="sdk:local-runtime",
            ))
            assert result.status is RunStatus.COMPLETED
            assert result.content == "read complete"
            assert result.tools_used == ["read_file"]
            assert len(provider.requests) == 2
            accepted = handle.sessions.get_or_create("sdk:local-runtime").messages
            assert any("workspace evidence" in str(row.get("content")) for row in accepted)
            backend = handle.sessions.source_backend("sdk:local-runtime")
            assert backend is not None
            source_path = backend.get_session_path("sdk:local-runtime")
            persisted = source_path.read_bytes()
            assert persisted
        finally:
            await first.aclose()
        assert first.state is RuntimeState.CLOSED

        second = RuntimeComposition.define(config)
        reopened = await second.acquire()
        try:
            assert reopened.sessions is not handle.sessions
            assert reopened.sessions.get_or_create("sdk:local-runtime").messages == accepted
            assert source_path.read_bytes() == persisted
            assert len(provider.requests) == 2
        finally:
            await second.aclose()
        assert second.state is RuntimeState.CLOSED

    asyncio.run(exercise())


@asynccontextmanager
async def _running_composition(config, provider, monkeypatch):
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    composition = RuntimeComposition.define(config)
    handle = await composition.acquire()
    run_task = asyncio.create_task(handle.agent.run())
    try:
        yield handle, run_task
    finally:
        provider.release_first.set()
        await composition.aclose()
        if not run_task.done():
            await handle.bus.publish_inbound(InboundMessage(
                channel="system", sender_id="runtime", chat_id="", content="shutdown wake",
            ))
        await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=5)


async def test_busy_followup_admission_failure_keeps_bus_consumer_alive(monkeypatch, tmp_path):
    config = _config(tmp_path)
    provider = _ScriptedProvider([LLMResponse(content="first response")], block_first=True)
    async with _running_composition(config, provider, monkeypatch) as (handle, run_task):
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="admission-failure", content="first",
        ))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)

        def fail_admission(*args, **kwargs):
            raise OSError("journal unavailable")

        monkeypatch.setattr(SourceSessionStore, "admit_pending_followup", fail_admission)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="admission-failure", content="second",
        ))
        while True:
            reply = await asyncio.wait_for(handle.bus.consume_outbound(), timeout=5)
            if "could not be accepted" in reply.content:
                break
        assert not run_task.done()
        assert handle.agent._pending_queues["websocket:admission-failure"].empty()
        provider.release_first.set()
        while True:
            reply = await asyncio.wait_for(handle.bus.consume_outbound(), timeout=5)
            if reply.content == "first response":
                break
        assert len(provider.requests) == 1 and not run_task.done()


@pytest.mark.parametrize("discard_fails", [False, True])
async def test_stop_discards_saved_followup_before_cancelling_owner(
    monkeypatch, tmp_path, discard_fails,
):
    config = _config(tmp_path)
    provider = _ScriptedProvider([LLMResponse(content="unused")], block_first=True)
    async with _running_composition(config, provider, monkeypatch) as (handle, run_task):
        admitted = asyncio.Event()
        original = handle.agent._admit_pending_activity

        async def observe(key, message, *, followup_id=None):
            outcome = await original(key, message, followup_id=followup_id)
            admitted.set()
            return outcome

        monkeypatch.setattr(handle.agent, "_admit_pending_activity", observe)
        for content in ("first", "pending input to discard"):
            await handle.bus.publish_inbound(InboundMessage(
                channel="websocket", sender_id="webui", chat_id="stop-followup", content=content,
            ))
            if content == "first":
                await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        await asyncio.wait_for(admitted.wait(), timeout=5)
        backend = handle.sessions.source_backend("websocket:stop-followup")
        assert SOURCE_PENDING_FOLLOWUPS_METADATA_KEY in backend.source.load_source_snapshot(
            "websocket:stop-followup",
        ).header["metadata"]
        if discard_fails:
            def fail_discard(*args, **kwargs):
                raise OSError("stop journal unavailable")
            monkeypatch.setattr(SourceSessionStore, "discard_pending_followups", fail_discard)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="stop-followup", content="/stop",
        ))
        while True:
            reply = await asyncio.wait_for(handle.bus.consume_outbound(), timeout=5)
            if reply.content.startswith(("Stopped", "Could not complete the stop request")):
                break
        if discard_fails:
            assert reply.content == "Could not complete the stop request. Please retry."
            assert not provider.first_cancelled.is_set() and not run_task.done()
            assert SOURCE_PENDING_FOLLOWUPS_METADATA_KEY in backend.source.load_source_snapshot(
                "websocket:stop-followup",
            ).header["metadata"]
            return
        assert provider.first_cancelled.is_set() and not run_task.done()
        snapshot = backend.source.load_source_snapshot("websocket:stop-followup")
        assert SOURCE_PENDING_FOLLOWUPS_METADATA_KEY not in snapshot.header["metadata"]
        assert not any(event.get("content") == "pending input to discard" for event in snapshot.events)
        assert handle.bus.inbound.empty() and len(provider.requests) == 1


async def test_recovered_followup_is_consumed_once_by_normal_composed_turn(monkeypatch, tmp_path):
    config = _config(tmp_path)
    provider = _ScriptedProvider([
        LLMResponse(content="initial response"), LLMResponse(content="recovered response"),
    ])
    key = "websocket:recovered-followup"
    async with _running_composition(config, provider, monkeypatch) as (handle, run_task):
        initial = await handle.agent.execute_turn(TurnRequest(content="initial", session_key=key))
        assert initial.status is RunStatus.COMPLETED
        backend = handle.sessions.source_backend(key)
        message = InboundMessage(
            channel="websocket", sender_id="webui", chat_id="recovered-followup",
            content="original recovered input", metadata={"webui": True},
        )
        mint_pending_followup_id(message)
        async with acquire_execution(backend, key, run_id="journal-admission") as owner:
            backend.source.admit_pending_followup(key, owner, record_pending_followup(message))
        await RecoveryCoordinator(handle.sessions, handle.bus).scan()
        while True:
            reply = await asyncio.wait_for(handle.bus.consume_outbound(), timeout=5)
            if reply.content == "recovered response":
                break
        snapshot = backend.source.load_source_snapshot(key)
        assert SOURCE_PENDING_FOLLOWUPS_METADATA_KEY not in snapshot.header["metadata"]
        assert sum(event.get("content") == message.content for event in snapshot.events) == 1
        assert len(provider.requests) == 2 and not run_task.done()


def test_from_config_rejects_malformed_session_evidence_before_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    legacy_sessions = config.workspace_path / "sessions"
    legacy_sessions.mkdir()
    (legacy_sessions / "broken.jsonl").write_text("{broken", encoding="utf-8")
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda _config: pytest.fail("provider construction preceded session preflight"),
    )
    with pytest.raises(RuntimePreflightError):
        AgentLoop.from_config(config, tool_registry=ToolRegistry())
    after = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    }
    assert after == before


def test_from_config_rejects_unsupported_idle_before_runtime_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.agents.defaults.session_ttl_minutes = 15
    before = tuple(tmp_path.iterdir())
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda _config: pytest.fail("provider construction preceded idle preflight"),
    )
    with pytest.raises(RuntimePreflightError, match="idle"):
        AgentLoop.from_config(config, tool_registry=ToolRegistry())
    assert tuple(tmp_path.iterdir()) == before
    with pytest.raises(RuntimePreflightError, match="idle"):
        runtime_preflight(config)


async def test_default_composition_injects_canonical_bus_followup_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    provider = _ScriptedProvider(
        [LLMResponse(content="first response"), LLMResponse(content="follow-up handled")],
        block_first=True,
    )
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)

    composition = RuntimeComposition.define(config)
    handle = await composition.acquire()
    run_task = asyncio.create_task(handle.agent.run())
    queued = asyncio.Event()
    queued_task: asyncio.Task[bool] | None = None
    admit_pending = handle.agent._admit_pending_activity

    async def observe_pending(
        key: str, message: InboundMessage, *, followup_id: str | None = None,
    ):
        accepted = await admit_pending(key, message, followup_id=followup_id)
        if message.content == "queued follow-up marker":
            queued.set()
        return accepted

    monkeypatch.setattr(handle.agent, "_admit_pending_activity", observe_pending)
    try:
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket",
            sender_id="webui",
            chat_id="local-runtime-followup",
            content="start held turn",
            metadata={"webui": True},
        ))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)

        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket",
            sender_id="webui",
            chat_id="local-runtime-followup",
            content="queued follow-up marker",
            metadata={"webui": True},
        ))

        queued_task = asyncio.create_task(queued.wait())
        done, _ = await asyncio.wait(
            {queued_task, run_task},
            timeout=5,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if run_task in done:
            run_task.result()
        assert queued_task in done, "composed bus follow-up was not admitted to the turn queue"

        provider.release_first.set()
        await asyncio.wait_for(provider.followup_seen.wait(), timeout=5)
        assert len(provider.requests) == 2
        assert sum(
            "queued follow-up marker" in str(message.get("content") or "")
            for message in provider.requests[1]
        ) == 1
        while True:
            outbound = await asyncio.wait_for(handle.bus.consume_outbound(), timeout=5)
            if outbound.chat_id == "local-runtime-followup" and outbound.content == "follow-up handled":
                break
    finally:
        provider.release_first.set()
        if queued_task is not None and not queued_task.done():
            queued_task.cancel()
            await asyncio.gather(queued_task, return_exceptions=True)
        await composition.aclose()
        if not run_task.done():
            await handle.bus.publish_inbound(InboundMessage(
                channel="system", sender_id="runtime", chat_id="", content="shutdown wake",
            ))
        await asyncio.gather(run_task, return_exceptions=True)
