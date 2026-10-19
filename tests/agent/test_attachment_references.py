import asyncio
import base64
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import TranscriptInput
from nanobot.agent.loop import AgentLoop, TurnContext, TurnKind, _TurnExecutionOptions
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.turn import (
    DeliveryTarget,
    RunPolicyContext,
    RunStatus,
    TraceMode,
    TurnRequest,
    TurnSource,
)
from nanobot.agent.turn import TurnContext as RunTurnContext
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ChannelsConfig
from nanobot.providers.base import LLMResponse
from nanobot.session.source_projection import source_nodes
from nanobot.utils.document import reference_non_image_attachments
from tests.agent.p6_admitted_fixtures import AgentInputs, AsyncAdmittedFactory
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async


async def _make_loop(
    tmp_path: Path,
    acquire: AsyncAdmittedFactory,
    channels_config: ChannelsConfig | None = None,
) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok"))
    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=provider,
            workspace=inputs.config.workspace_path,
            model="test-model",
            channels_config=channels_config,
            session_manager=inputs.sessions,
            tool_registry=inputs.tools,
            cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools,
            restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    return await acquire(tmp_path, build)


def _turn_context(loop: AgentLoop, msg: InboundMessage) -> TurnContext:
    return TurnContext(
        msg=msg,
        session_key=f"{msg.channel}:{msg.chat_id}",
        turn_id="turn-1",
        runtime=loop.llm_runtime(),
        kind=TurnKind.USER,
        delivery=loop.turn_delivery_factory.create(msg, f"{msg.channel}:{msg.chat_id}"),
    )


async def _execute_inbound(loop: AgentLoop, msg: InboundMessage) -> None:
    await loop.execute_turn(
        TurnRequest(
            content=msg.content,
            session_key=msg.session_key,
            sender_id=msg.sender_id,
            source=loop._inbound_turn_source(msg),
            media=tuple(msg.media),
            metadata=dict(msg.metadata),
            route=DeliveryTarget(channel=msg.channel, chat_id=msg.chat_id),
        ),
        _options=_TurnExecutionOptions(
            inbound=msg,
            raise_errors=True,
            trusted_foreground_activity=loop._trusted_foreground_inbound(msg),
        ),
    )


def _run_context(loop: AgentLoop) -> RunTurnContext:
    return RunTurnContext(policy=RunPolicyContext(
        workspace=loop.workspace,
        context_manager=loop._fork_dependencies.resource_policy.context_manager,
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("extract_document_text", [True, False])
async def test_document_attachment_is_referenced_and_read_on_demand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extract_document_text: bool,
    admitted_async,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    csv_path = media_dir / "report.csv"
    csv_path.write_text("name,value\nnanobot,1", encoding="utf-8")
    monkeypatch.setattr("nanobot.agent.tools.path_utils.get_media_dir", lambda: media_dir)

    loop = await _make_loop(
        tmp_path, admitted_async,
        ChannelsConfig(extract_document_text=extract_document_text),
    )
    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="import this report",
        media=[str(csv_path)],
    )
    ctx = _turn_context(loop, msg)

    await loop._restore_turn(ctx)

    assert ctx.msg.content == f"import this report\n\n[Attachment: {csv_path}]"
    assert "name,value" not in ctx.msg.content
    assert ctx.msg.media == []

    read_tool = ReadFileTool(workspace=workspace, allowed_dir=workspace)
    result = await read_tool.execute(path=str(csv_path))

    assert "1| name,value" in result
    assert "2| nanobot,1" in result


@pytest.mark.asyncio
async def test_document_reference_survives_session_reload(tmp_path: Path, admitted_async) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    doc_path = tmp_path / "report.csv"
    doc_path.write_text("name,value", encoding="utf-8")

    loop = await _make_loop(tmp_path, admitted_async)
    loop._run_agent_loop = AsyncMock(side_effect=RuntimeError("interrupt"))  # type: ignore[method-assign]
    msg = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="persisted-attachment",
        content="review this",
        media=[str(doc_path)],
    )

    with pytest.raises(RuntimeError, match="interrupt"):
        await _execute_inbound(loop, msg)

    session_key = "websocket:persisted-attachment"
    loop.sessions.invalidate(session_key)
    persisted = loop.sessions.get_or_create(session_key)

    assert [message["role"] for message in persisted.messages] == ["user"]
    assert persisted.messages[0]["content"] == (
        f"review this\n\n[Attachment: {doc_path.resolve()}]"
    )
    assert "media" not in persisted.messages[0]


@pytest.mark.asyncio
async def test_pending_document_attachment_keeps_body_out_of_prompt(
    tmp_path: Path,
    admitted_async,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    doc_path = tmp_path / "followup.txt"
    doc_path.write_text("Do not inject this file body", encoding="utf-8")
    captured_messages: list[list[dict]] = []
    call_count = 0

    async def chat_with_retry(*, messages: list[dict], **kwargs: object) -> LLMResponse:
        nonlocal call_count
        call_count += 1
        captured_messages.append([dict(message) for message in messages])
        return LLMResponse(content=f"answer-{call_count}", tool_calls=[], usage=None)

    loop = await _make_loop(tmp_path, admitted_async)
    loop.provider.chat_with_retry = chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])

    pending_queue: asyncio.Queue[InboundMessage] = asyncio.Queue()
    await pending_queue.put(
        InboundMessage(
            channel="cli",
            sender_id="u",
            chat_id="c",
            content="check this",
            media=[str(doc_path)],
        )
    )

    runtime = loop.llm_runtime()
    result = await loop._run_agent_loop(
        TranscriptInput(
            history=[{"role": "user", "content": "hello"}],
            current_message=None,
        ),
        runtime=runtime,
        request_context=RequestContext(channel="cli", chat_id="c", runtime=runtime),
        pending_queue=pending_queue,
        run_context=_run_context(loop),
    )

    assert result.final_content == "answer-2"
    assert result.had_injections is True
    injected_user_content = [
        message["content"]
        for message in captured_messages[-1]
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ][-1]
    assert "check this" in injected_user_content
    assert f"[Attachment: {doc_path}]" in injected_user_content
    assert "Do not inject this file body" not in injected_user_content


def test_attachment_references_still_preserve_images(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
        )
    )
    doc_path = tmp_path / "report.txt"
    doc_path.write_text("manual extraction target", encoding="utf-8")

    content, media = reference_non_image_attachments(
        "review these",
        [str(image_path), str(doc_path)],
    )

    assert media == [str(image_path)]
    assert f"[Attachment: {doc_path}]" in content
    assert "manual extraction target" not in content


def test_attachment_references_canonicalize_existing_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
        )
    )
    doc_path = tmp_path / "report.csv"
    doc_path.write_text("name,value", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    content, media = reference_non_image_attachments(
        "review these",
        [image_path.name, doc_path.name],
    )

    assert media == [str(image_path.resolve())]
    assert f"[Attachment: {doc_path.resolve()}]" in content


def test_attachment_setup_unadmitted_constructor_refuses_before_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.memory_recovery_startup_admission import StartupReceiptAdmissionError
    from nanobot.session.manager import SessionManager

    provider = MagicMock()
    bus = MessageBus()
    workspace = tmp_path / "unadmitted"
    pending = workspace / "memory" / ".consolidation.pending.json"
    pending.parent.mkdir(parents=True)
    pending.write_text("{broken", encoding="utf-8")

    def denied(*_args, **_kwargs):
        pytest.fail("unadmitted attachment setup reached a session or directory effect")

    monkeypatch.setattr(SessionManager, "__init__", denied)
    monkeypatch.setattr(Path, "mkdir", denied)
    before = tuple(tmp_path.iterdir())
    with pytest.raises(StartupReceiptAdmissionError):
        AgentLoop(
            bus=bus, provider=provider, workspace=workspace,
            sessions_root=tmp_path / "sessions",
        )
    assert tuple(tmp_path.iterdir()) == before
    assert not bus._pending
    provider.chat.assert_not_called()
    provider.chat_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_attachment_setup_uses_owned_workspace_and_inert_composition(
    tmp_path: Path,
    admitted_async,
) -> None:
    from nanobot.config.loader import load_config

    channels = ChannelsConfig(extract_document_text=False)
    loop = await _make_loop(tmp_path, admitted_async, channels)
    config = load_config()

    assert loop.channels_config is channels
    assert loop.workspace == loop.context.workspace == loop.sessions.workspace == config.workspace_path
    assert loop.workspace.parent.parent == tmp_path
    assert config._source_path.parent == loop.sessions.sessions_root.parent
    assert loop.sessions.has_managed_authority
    assert loop.turn_delivery_factory.bus is loop.bus
    assert loop.tools_config.restrict_to_workspace is True
    assert loop.auto_compact is None
    assert not loop._background_tasks
    assert not loop.cron_service._running
    loop.provider.chat.assert_not_called()
    loop.provider.chat_with_retry.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["completed", "error", "cancelled"])
@pytest.mark.parametrize("media_kind", ["document", "image", "mixed"])
@pytest.mark.parametrize("content", ["review these", ""])
async def test_canonical_attachment_input_is_durable_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admitted_async,
    terminal: str,
    media_kind: str,
    content: str,
) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    doc_path = loop.workspace / "report.csv"
    doc_body = "private attachment body must not enter history or prompts"
    doc_path.write_text(doc_body, encoding="utf-8")
    image_path = loop.workspace / "chart.png"
    image_path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
    ))
    monkeypatch.chdir(loop.workspace)
    monkeypatch.setattr(
        "nanobot.utils.document.extract_text",
        MagicMock(side_effect=AssertionError("automatic document extraction")),
    )
    monkeypatch.setattr(
        "nanobot.utils.document.open_document_line_source",
        MagicMock(side_effect=AssertionError("automatic document extraction")),
    )
    media = {
        "document": (doc_path.name,),
        "image": (image_path.name,),
        "mixed": (image_path.name, doc_path.name),
    }[media_kind]
    key = "websocket:canonical-attachment"
    request = TurnRequest(
        content=content,
        session_key=key,
        source=TurnSource.SDK,
        route=DeliveryTarget(channel="websocket", chat_id="canonical-attachment"),
        media=media,
        metadata={"message_id": "original-message"},
        trace_mode=TraceMode.SANITIZED,
        run_id="attachment-run",
    )
    original_request = deepcopy(request)
    observed_contexts: list[RunTurnContext] = []
    observed_inputs: list[TranscriptInput] = []
    early_events: list[dict] = []
    run = loop._run_agent_loop
    fault = asyncio.CancelledError("attachment cancellation") if terminal == "cancelled" else RuntimeError("attachment interruption")

    async def observe_input(
        transcript: TranscriptInput, *, run_context: RunTurnContext, **kwargs,
    ):
        writer = run_context.source_writer
        assert writer is not None
        observed_contexts.append(run_context)
        observed_inputs.append(transcript)
        early_events.extend(writer.owner.read_snapshot().events)
        if terminal != "completed":
            raise fault
        return await run(transcript, run_context=run_context, **kwargs)

    monkeypatch.setattr(loop, "_run_agent_loop", observe_input)
    if terminal == "completed":
        result = await loop.execute_turn(request, _options=_TurnExecutionOptions(raise_errors=True))
        assert result.status is RunStatus.COMPLETED and result.content == "ok"
        loop.provider.chat_with_retry.assert_awaited_once()
        assert doc_body not in str(loop.provider.chat_with_retry.call_args)
    else:
        with pytest.raises(type(fault)) as caught:
            await loop.execute_turn(request, _options=_TurnExecutionOptions(raise_errors=True))
        assert caught.value is fault
        loop.provider.chat_with_retry.assert_not_awaited()

    assert request == original_request
    context = observed_contexts[0]
    assert context.request is request
    assert context.sanitized_messages is not None
    assert context.sanitized_messages[0] == {"role": "user", "content": content}
    assert doc_path.name not in str(context.sanitized_messages)
    assert image_path.name not in str(context.sanitized_messages)
    assert [event["role"] for event in early_events] == ["user"]
    expected_content = content
    if media_kind != "image":
        reference = f"[Attachment: {doc_path.resolve()}]"
        expected_content = f"{content}\n\n{reference}" if content else reference
    expected_media = [str(image_path.resolve())] if media_kind != "document" else []
    assert observed_inputs[0].current_message == expected_content
    assert observed_inputs[0].media == (expected_media or None)
    assert early_events[0]["content"] == expected_content
    if expected_media:
        assert early_events[0]["media"] == expected_media
    else:
        assert "media" not in early_events[0]
    assert early_events[0]["_event_id"] == context.source_input_event_id
    assert early_events[0]["_run_id"] == request.run_id
    assert context.source_writer is not None
    assert early_events[0]["_stream_id"] == context.source_writer.stream_id
    assert "_source_origin" not in early_events[0]
    assert "_artifact_read" not in early_events[0]
    assert doc_body not in str(early_events)

    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.messages[0] == early_events[0]
    assert [event["role"] for event in reloaded.messages] == (
        ["user", "assistant"] if terminal == "completed" else ["user"]
    )
    projected = source_nodes(reloaded.messages)
    assert projected[0].event_id == context.source_input_event_id
    if media_kind != "image":
        assert projected[0].message["content"].count(reference) == 1
    assert doc_body not in str(projected)
    record = loop.run_store.load("attachment-run")
    assert record is not None and record.status.value == terminal
    assert record.metadata["source_stream_id"] == early_events[0]["_stream_id"]
    assert not loop._source_activity_admissions


@pytest.mark.asyncio
async def test_canonical_approval_attachment_keeps_expansion_out_of_source_and_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admitted_async,
) -> None:
    loop = await _make_loop(tmp_path, admitted_async)
    doc_path = loop.workspace / "approval.csv"
    doc_path.write_text("private document body", encoding="utf-8")
    key = "telegram:attachment-approval"
    summary = "private expanded approval instructions"
    await loop.approval_admission.store(
        session_key=key, channel="telegram", chat_id="attachment-approval", summary=summary,
    )
    request = TurnRequest(
        content="yes",
        source=TurnSource.GATEWAY,
        route=DeliveryTarget(channel="telegram", chat_id="attachment-approval"),
        media=(str(doc_path),),
        trace_mode=TraceMode.SANITIZED,
    )
    original_request = deepcopy(request)
    observed_inputs: list[TranscriptInput] = []
    run = loop._run_agent_loop

    async def observe_input(transcript: TranscriptInput, **kwargs):
        observed_inputs.append(transcript)
        return await run(transcript, **kwargs)

    monkeypatch.setattr(loop, "_run_agent_loop", observe_input)
    result = await loop.execute_turn(request, _options=_TurnExecutionOptions(raise_errors=True))

    assert result.status is RunStatus.COMPLETED and result.content == "ok"
    assert request == original_request
    assert summary in observed_inputs[0].current_message
    assert f"[Attachment: {doc_path.resolve()}]" in observed_inputs[0].current_message
    assert "private document body" not in str(loop.provider.chat_with_retry.call_args)
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert [event["role"] for event in reloaded.messages] == ["user", "assistant"]
    assert reloaded.messages[0]["content"] == f"yes\n\n[Attachment: {doc_path.resolve()}]"
    assert "media" not in reloaded.messages[0]
    assert summary not in str(reloaded.messages)
    assert result.messages == [
        {"role": "user", "content": "yes"}, {"role": "assistant", "content": "ok"},
    ]
