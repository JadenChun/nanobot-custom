"""The gateway must persist display rows under the source that produced them."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.outbound_events import DelegatedChildEvent, MessageAdmittedEvent
from nanobot.bus.runtime_events import TurnCompleted
from nanobot.channels.manager import ChannelManager
from nanobot.providers.base import LLMResponse
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.session.session_handles import session_handle_for_name
from nanobot.session.session_messages import SESSION_MESSAGE_METADATA_KEY
from nanobot.session.source_store import SourceSessionStore
from nanobot.session.webui_turns import WebuiTurnCoordinator
from nanobot.webui.forking import create_webui_chat_fork
from nanobot.webui.transcript import (
    append_transcript_object,
    hydrate_consolidated_transcript_lines,
    read_transcript_lines,
    replay_transcript_to_ui_messages,
)
from tests.agent.test_input_attempt_reconciliation import channel_for
from tests.agent.test_input_attempt_reconciliation import isolate_projection as isolate_projection
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_engine import engine_config as engine_config
from tests.webui.test_message_attempt_binding import _message_envelope

KEY = "websocket:display-origin"
CHAT_ID = "display-origin"


def _hydrated(sessions, key, state):
    rows = read_transcript_lines(key, expected_source=state)
    return hydrate_consolidated_transcript_lines(
        rows,
        lambda ids: sessions.read_canonical_events_by_ids(key, list(ids), expected_source=state),
    )


@asynccontextmanager
async def _gateway(config, tmp_path, *answers):
    lease = await DurableSessionAuthority(
        config.workspace_path, sessions_root=tmp_path / "sessions",
    ).acquire()
    provider = ScriptedProvider([LLMResponse(content=answer) for answer in answers])
    loop = AgentLoop.from_config(
        config, tool_registry=ToolRegistry(), session_manager=lease.manager, provider=provider,
    )
    channel, conn = channel_for(loop, config)
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus, sessions=loop.sessions, schedule_background=lambda coro: coro.close(),
    )
    try:
        with coordinator.connected():
            yield loop, channel, conn, provider
    finally:
        await channel.stop()
        await loop.aclose()
        await lease.close()


async def _submit(channel, conn, text, *, turn_id="display-turn"):
    await channel._dispatch_envelope(conn, "owner", _message_envelope(
        chat_id=CHAT_ID, content=text, turn_id=turn_id,
    ))


async def _drain(loop, channel):
    while loop.bus.outbound_size:
        await ChannelManager._send_once(channel, await loop.bus.consume_outbound())


async def _dispatch_one(loop, channel):
    assert loop.bus.inbound_size == 1
    await loop._dispatch(await loop.bus.consume_inbound())
    await _drain(loop, channel)


@pytest.mark.parametrize("failed", [False, True])
async def test_first_display_input_depends_on_canonical_append(
    engine_config, tmp_path, monkeypatch, failed,
):
    async with _gateway(engine_config, tmp_path, "First answer") as (loop, channel, conn, provider):
        assert loop.sessions.read_saved_source_metadata(KEY) is None
        await _submit(channel, conn, "First prompt")
        assert loop.bus.inbound_size == 1
        assert not any(row.get("event") == "user" for row in read_transcript_lines(KEY))
        if failed:
            def refuse(*args, **kwargs):
                raise OSError("failed before canonical input commit")

            monkeypatch.setattr(SourceSessionStore, "append_source_events", refuse)
        await _dispatch_one(loop, channel)
        prompts = [row for row in read_transcript_lines(KEY) if row.get("event") == "user"]
        if failed:
            assert not prompts
            assert not provider.calls
        else:
            source = loop.sessions.read_saved_source_metadata(KEY)
            assert source is not None
            qualified = _hydrated(loop.sessions, KEY, source.state)
            assert qualified[0]["text"] == "First prompt"
            assert any(row.get("text") == "First answer" for row in qualified)
            assert qualified[-1]["event"] == "turn_end"
            assert [row["text"] for row in qualified if row.get("event") == "user"] == ["First prompt"]
            canonical = loop.sessions.read_existing_canonical_source(KEY)
            assert canonical is not None
            committed = next(row for row in canonical[1].events if row.get("role") == "user")
            assert prompts[0]["_webui_source_prompt_id"] == committed["_event_id"]
            assert len(provider.calls) == 1


@pytest.mark.parametrize("replacement", ["reset", "delete"])
async def test_delayed_answer_is_not_rebound_to_replacement(engine_config, tmp_path, replacement):
    async with _gateway(engine_config, tmp_path, "Old answer", "New answer") as (loop, channel, conn, _):
        await _submit(channel, conn, "Old prompt")
        await loop._dispatch(await loop.bus.consume_inbound())
        delayed = []
        while loop.bus.outbound_size:
            delayed.append(await loop.bus.consume_outbound())
        assert delayed
        if replacement == "reset":
            assert loop.sessions.reset_session(KEY) is not None
        else:
            assert loop.sessions.delete_session(KEY)
            assert loop.sessions.upsert_session_metadata(KEY, {"webui": True}, create_if_absent=True)

        for output in delayed:
            await ChannelManager._send_once(channel, output)
        assert not {"Old prompt", "Old answer"}.intersection(
            row.get("text") for row in read_transcript_lines(KEY)
        ), "delayed admission or answer appended after replacement"

        await _submit(channel, conn, "New prompt", turn_id="new-display-turn")
        await _dispatch_one(loop, channel)
        current = loop.sessions.read_saved_source_metadata(KEY)
        assert current is not None
        assert any(
            row.get("text") == "New answer"
            for row in _hydrated(loop.sessions, KEY, current.state)
        )


async def test_lifetime_check_does_not_adopt_prior_records_or_unbound_legacy(engine_config, tmp_path):
    async with _gateway(engine_config, tmp_path, "Old answer", "New answer") as (loop, channel, conn, _):
        append_transcript_object(KEY, {"event": "user", "chat_id": CHAT_ID, "text": "Unbound input"})
        await _submit(channel, conn, "Old prompt")
        await _dispatch_one(loop, channel)
        old = loop.sessions.read_saved_source_metadata(KEY)
        assert old is not None
        assert loop.sessions.upsert_session_metadata(KEY, {"title": "Longer title"})
        title_edit = loop.sessions.read_saved_source_metadata(KEY)
        assert title_edit is not None
        assert title_edit.state.revision > old.state.revision
        assert [row["text"] for row in _hydrated(loop.sessions, KEY, title_edit.state)
                if row.get("event") == "user"] == ["Old prompt"]
        assert loop.sessions.delete_session(KEY)
        await _submit(channel, conn, "New prompt", turn_id="new-display-turn")
        await _dispatch_one(loop, channel)
        current = loop.sessions.read_saved_source_metadata(KEY)
        assert current is not None
        assert [row["text"] for row in _hydrated(loop.sessions, KEY, current.state)
                if row.get("event") == "user"] == ["New prompt"]
        assert {
            row.get("text") or row.get("prompt_preview")
            for row in read_transcript_lines(KEY)
        } >= {"Unbound input", "Old prompt"}

    async with _gateway(engine_config, tmp_path) as (loop, _, _, _):
        current = loop.sessions.read_saved_source_metadata(KEY)
        assert current is not None
        assert [row["text"] for row in _hydrated(loop.sessions, KEY, current.state)
                if row.get("event") == "user"] == ["New prompt"]


@pytest.mark.parametrize("write_error", [False, True])
async def test_steer_display_uses_the_committed_injection(engine_config, tmp_path, monkeypatch, write_error):
    async with _gateway(engine_config, tmp_path, "Before steer", "After steer") as (
        loop, channel, conn, provider,
    ):
        provider_entered, provider_release = asyncio.Event(), asyncio.Event()
        steer_queued, turn_completed = asyncio.Event(), asyncio.Event()
        chat = provider.chat
        admit = loop._admit_pending_activity

        async def pause_first_response(*args, **kwargs):
            if not provider.calls:
                provider_entered.set()
                await provider_release.wait()
            return await chat(*args, **kwargs)

        async def observe_admission(*args, **kwargs):
            result = await admit(*args, **kwargs)
            if result.followup_pending is True:
                steer_queued.set()
            return result

        monkeypatch.setattr(provider, "chat", pause_first_response)
        monkeypatch.setattr(loop, "_admit_pending_activity", observe_admission)
        if write_error:
            def refuse_guidance(session_key, row):
                if row.get("event") == "user" and (
                    row.get("text") == "Guidance" or row.get("prompt_preview") == "Guidance"
                ):
                    raise OSError("display write failed")
                append_transcript_object(session_key, row)

            monkeypatch.setattr("nanobot.webui.transcript.append_transcript_object", refuse_guidance)
        loop.bus.subscribe(lambda _: turn_completed.set(), TurnCompleted)
        runner = asyncio.create_task(loop.run())
        try:
            await _submit(channel, conn, "First prompt")
            await asyncio.wait_for(provider_entered.wait(), timeout=5)
            await channel._dispatch_envelope(conn, "owner", {
                **_message_envelope(chat_id=CHAT_ID, content="Guidance", turn_id="guide-turn"),
                "request_intent": "steer", "target_turn_id": "display-turn",
            })
            await asyncio.wait_for(steer_queued.wait(), timeout=5)
            await _drain(loop, channel)
            assert not any(row.get("text") == "Guidance" for row in read_transcript_lines(KEY))

            provider_release.set()
            await asyncio.wait_for(turn_completed.wait(), timeout=5)
            await _drain(loop, channel)
            canonical = loop.sessions.read_existing_canonical_source(KEY)
            assert canonical is not None
            events = canonical[1].events
            origin = canonical[1].state
            rows = _hydrated(loop.sessions, KEY, origin)
            texts = [row["content"] for row in replay_transcript_to_ui_messages(rows)
                     if row.get("content")]
            committed = next(row for row in events if row.get("content") == "Guidance")
            if write_error:
                assert texts == ["First prompt", "Before steer", "After steer"]
                assert rows[-1]["event"] == "turn_end"
                assert rows[-1]["transcript_incomplete"] is True
            else:
                assert texts == ["First prompt", "Before steer", "Guidance", "After steer"]
                projected = next(row for row in rows if row.get("text") == "Guidance")
                assert projected["_webui_source_prompt_id"] == committed["_event_id"]
        finally:
            provider_release.set()
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)


@pytest.mark.parametrize("qualified_parent", [False, True])
async def test_fork_publishes_only_qualified_prefix_under_its_own_origin(
    engine_config, tmp_path, qualified_parent,
):
    async with _gateway(engine_config, tmp_path, "Included answer", "Excluded answer") as (
        loop, channel, conn, _,
    ):
        await _submit(channel, conn, "Included prompt")
        await _dispatch_one(loop, channel)
        parent = loop.sessions.read_saved_source_metadata(KEY)
        assert parent is not None
        append_transcript_object(KEY, {
            "event": "message", "chat_id": CHAT_ID, "text": "Unbound display answer",
        })
        if not qualified_parent:
            # An imported canonical prompt has no display receipt; the read
            # cannot assume that display prompt position equals canonical position.
            source = loop.sessions.read_existing_canonical_source(KEY)
            assert source is not None
            store, snapshot = source
            from nanobot.session.source import new_source_event

            store.source.append_source_events(KEY, snapshot.state, [
                new_source_event({"role": "user", "content": "Canonical-only prompt"}, run_id="imported"),
            ])
        await _submit(channel, conn, "Excluded prompt", turn_id="later")
        await _dispatch_one(loop, channel)
        fork = create_webui_chat_fork(loop.sessions, source_chat_id=CHAT_ID, before_user_index=1)
        assert fork is not None
        target = loop.sessions.read_saved_source_metadata(fork[1])
        assert target is not None
        assert not target.state.same_origin(parent.state)
        rows = read_transcript_lines(fork[1], expected_source=target.state)
        texts = {row["text"] for row in rows if row.get("text")}
        assert texts == {"Included prompt", "Included answer"}
        assert rows[-1]["event"] == "fork_marker"
        assert read_transcript_lines(fork[1], expected_source=parent.state) == []
        assert not any(row.get("text") == "Unbound display answer" for row in read_transcript_lines(fork[1]))


@pytest.mark.parametrize("failure", ["write_error", "interrupted"])
async def test_fork_recovers_committed_answer_when_display_is_incomplete(
    engine_config, tmp_path, monkeypatch, failure,
):
    async with _gateway(engine_config, tmp_path, "Committed answer") as (loop, channel, conn, _):
        def refuse_answer(session_key, row):
            if session_key == KEY and row.get("event") == "stream_end":
                raise OSError("display write failed")
            append_transcript_object(session_key, row)

        await _submit(channel, conn, "First prompt")
        if failure == "write_error":
            monkeypatch.setattr("nanobot.webui.transcript.append_transcript_object", refuse_answer)
            await _dispatch_one(loop, channel)
        else:
            await loop._dispatch(await loop.bus.consume_inbound())
            # Simulate losing the pending display delivery after canonical commit.
            while loop.bus.outbound_size:
                output = await loop.bus.consume_outbound()
                if isinstance(output.event, MessageAdmittedEvent):
                    await ChannelManager._send_once(channel, output)
        source = loop.sessions.read_existing_canonical_source(KEY)
        assert source is not None and any(
            event.get("content") == "Committed answer" for event in source[1].events
        )
        parent_rows = read_transcript_lines(KEY, expected_source=source[1].state)
        assert not any(row.get("text") == "Committed answer" for row in parent_rows)
        if failure == "write_error":
            assert parent_rows[-1]["transcript_incomplete"] is True
        else:
            assert [row["event"] for row in parent_rows] == ["user"]

        fork = create_webui_chat_fork(loop.sessions, source_chat_id=CHAT_ID, before_user_index=1)
        assert fork is not None
        target = loop.sessions.read_saved_source_metadata(fork[1])
        assert target is not None
        fork_rows = read_transcript_lines(fork[1], expected_source=target.state)
        assert [row["text"] for row in fork_rows if row.get("text")] == [
            "First prompt", "Committed answer",
        ]
        assert not any(row.get("transcript_incomplete") for row in fork_rows)


class _StubExecTool(Tool):
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "stub exec"

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {"command": {"type": "string"}}}

    async def execute(self, command: str = "", **kwargs: object) -> str:
        self.calls.append(command)
        return f"ran:{command}"


def _session_message_inbound(content: str, *, message_id: str = "msg-1") -> InboundMessage:
    source = session_handle_for_name("websocket:peer-source", "luma")
    envelope = {
        "message_id": message_id,
        "created_at_ms": 1_700_000_000_000,
        "expect_reply": False,
        "source_handle": source.name,
        "source_session_key": source.session_key,
        "target_session_key": KEY,
    }
    return InboundMessage(
        channel="system",
        sender_id="session",
        chat_id=KEY,
        content=content,
        metadata={SESSION_MESSAGE_METADATA_KEY: envelope},
        session_key_override=KEY,
        input_role="user",
    )


@pytest.mark.parametrize("kind", ["shell", "slash"])
@pytest.mark.parametrize("failed", [False, True])
async def test_managed_shell_and_slash_command_archive_only_after_canonical_commit(
    engine_config, tmp_path, monkeypatch, kind, failed,
):
    async with _gateway(engine_config, tmp_path) as (loop, channel, conn, _):
        exec_tool = _StubExecTool()
        loop.tools.unregister("exec")
        loop.tools.register(exec_tool)
        envelope = (
            {**_message_envelope(chat_id=CHAT_ID, content="!printf ok", turn_id="shell-turn"),
             "user_shell": True}
            if kind == "shell"
            else _message_envelope(chat_id=CHAT_ID, content="/help", turn_id="slash-turn")
        )
        await channel._dispatch_envelope(conn, "owner", envelope)
        assert loop.bus.inbound_size == 1
        assert read_transcript_lines(KEY) == []
        if failed:
            def refuse(*args, **kwargs):
                raise OSError("canonical command append failed")

            monkeypatch.setattr(SourceSessionStore, "append_source_events", refuse)
        await _dispatch_one(loop, channel)
        source = loop.sessions.read_saved_source_metadata(KEY)
        assert source is not None
        qualified = _hydrated(loop.sessions, KEY, source.state)
        all_rows = read_transcript_lines(KEY)
        assert len(qualified) == len(all_rows)
        prompts = [row for row in qualified if row.get("event") == "user"]
        if failed:
            assert not prompts
            assert exec_tool.calls == []
            return
        expected_text = "!printf ok" if kind == "shell" else "/help"
        assert [row["text"] for row in prompts] == [expected_text]
        canonical = loop.sessions.read_existing_canonical_source(KEY)
        assert canonical is not None
        committed_user = next(row for row in canonical[1].events if row.get("role") == "user")
        assert prompts[0]["_webui_source_prompt_id"] == committed_user["_event_id"]
        if kind == "shell":
            assert exec_tool.calls == ["printf ok"]
            assert any(row.get("text") == "ran:printf ok" for row in qualified)
        else:
            assert any("nanobot commands" in str(row.get("text", "")) for row in qualified)
        assert qualified[-1]["event"] == "turn_end"


async def test_managed_new_and_busy_inline_commands_do_not_write_unbound_rows(
    engine_config, tmp_path, monkeypatch,
):
    async with _gateway(engine_config, tmp_path, "Initial reply", "Active reply") as (
        loop, channel, conn, provider,
    ):
        await _submit(channel, conn, "Initial prompt")
        await _dispatch_one(loop, channel)
        old_source = loop.sessions.read_saved_source_metadata(KEY)
        assert old_source is not None
        old_count = len(read_transcript_lines(KEY))
        assert old_count >= 3

        conn.send.reset_mock()
        await _submit(channel, conn, "/new", turn_id="reset-turn")
        await _dispatch_one(loop, channel)
        new_source = loop.sessions.read_saved_source_metadata(KEY)
        assert new_source is not None
        assert not new_source.state.same_origin(old_source.state)
        assert read_transcript_lines(KEY, expected_source=new_source.state) == []
        assert len(read_transcript_lines(KEY)) == old_count
        sent_frames = [json.loads(call.args[0]) for call in conn.send.await_args_list]
        assert any(
            frame.get("event") == "message" and frame.get("text") == "New session started."
            for frame in sent_frames
        )

        # Now run a live turn and send an inline /help while the turn is active.
        provider_entered, provider_release = asyncio.Event(), asyncio.Event()
        turn_completed = asyncio.Event()
        orig_chat = provider.chat

        async def hold_turn(*args, **kwargs):
            provider_entered.set()
            await provider_release.wait()
            return await orig_chat(*args, **kwargs)

        monkeypatch.setattr(provider, "chat", hold_turn)
        loop.bus.subscribe(lambda _: turn_completed.set(), TurnCompleted)
        runner = asyncio.create_task(loop.run())
        try:
            await _submit(channel, conn, "Busy prompt", turn_id="busy-turn")
            await asyncio.wait_for(provider_entered.wait(), timeout=5)
            conn.send.reset_mock()
            await channel._dispatch_envelope(conn, "owner", {
                **_message_envelope(chat_id=CHAT_ID, content="/help", turn_id="inline-help"),
                "request_intent": "steer",
                "target_turn_id": "busy-turn",
            })
            await asyncio.sleep(0.05)
            await _drain(loop, channel)
            sent_frames = [json.loads(call.args[0]) for call in conn.send.await_args_list]
            assert any(
                frame.get("event") == "message" and "nanobot commands" in str(frame.get("text", ""))
                for frame in sent_frames
            ), sent_frames
            provider_release.set()
            await asyncio.wait_for(turn_completed.wait(), timeout=5)
            await _drain(loop, channel)
        finally:
            provider_release.set()
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)

        active_rows = read_transcript_lines(KEY, expected_source=new_source.state)
        all_after = read_transcript_lines(KEY)[old_count:]
        assert len(active_rows) == len(all_after)
        assert not any(row.get("text") == "/help" for row in all_after)
        assert not any("nanobot commands" in str(row.get("text", "")) for row in all_after)


@pytest.mark.parametrize("failed", [False, True])
async def test_cross_session_input_archives_only_after_canonical_commit(
    engine_config, tmp_path, monkeypatch, failed,
):
    async with _gateway(engine_config, tmp_path, "Setup reply", "Cross reply") as (
        loop, channel, conn, _,
    ):
        await _submit(channel, conn, "Setup prompt")
        await _dispatch_one(loop, channel)
        source = loop.sessions.read_saved_source_metadata(KEY)
        assert source is not None

        if failed:
            def refuse(*args, **kwargs):
                raise OSError("canonical cross-session append failed")

            monkeypatch.setattr(SourceSessionStore, "append_source_events", refuse)

        inbound = _session_message_inbound("Please review")
        # Trigger UserInputAccepted before canonical dispatch, matching AgentLoop.run().
        await loop.runtime_event_publisher.user_input_accepted(inbound, KEY)
        await _drain(loop, channel)
        assert not any(row.get("text") == "Please review" for row in read_transcript_lines(KEY))

        await loop._dispatch(inbound)
        await _drain(loop, channel)

        qualified = _hydrated(loop.sessions, KEY, source.state)
        all_rows = read_transcript_lines(KEY)
        assert len(qualified) == len(all_rows)
        if failed:
            assert not any(
                (row.get("text") or row.get("prompt_preview")) == "Please review"
                for row in all_rows
            )
            return

        prompts = [row for row in qualified if row.get("event") == "user"]
        assert [row["text"] for row in prompts] == ["Setup prompt", "Please review"]
        cross_row = prompts[1]
        assert cross_row["session_message"]["message_id"] == "msg-1"
        canonical = loop.sessions.read_existing_canonical_source(KEY)
        assert canonical is not None
        committed = next(
            row for row in canonical[1].events
            if row.get("role") == "user" and row.get("content") == "Please review"
        )
        assert cross_row["_webui_source_prompt_id"] == committed["_event_id"]
        assert any(row.get("text") == "Cross reply" for row in qualified)
        assert qualified[-1]["event"] == "turn_end"


async def test_cross_session_mid_turn_injection_orders_after_stream_with_origin(
    engine_config, tmp_path, monkeypatch,
):
    async with _gateway(engine_config, tmp_path, "Setup reply", "Before inject", "After inject") as (
        loop, channel, conn, provider,
    ):
        await _submit(channel, conn, "Setup prompt")
        await _dispatch_one(loop, channel)
        source = loop.sessions.read_saved_source_metadata(KEY)
        assert source is not None

        entered, release = asyncio.Event(), asyncio.Event()
        completed = asyncio.Event()
        orig_chat = provider.chat
        turn_calls = 0

        async def pause_first(*args, **kwargs):
            nonlocal turn_calls
            turn_calls += 1
            if turn_calls == 1:
                entered.set()
                await release.wait()
            return await orig_chat(*args, **kwargs)

        monkeypatch.setattr(provider, "chat", pause_first)
        loop.bus.subscribe(lambda _: completed.set(), TurnCompleted)
        runner = asyncio.create_task(loop.run())
        try:
            await _submit(channel, conn, "First prompt", turn_id="live-turn")
            await asyncio.wait_for(entered.wait(), timeout=5)
            await loop.bus.publish_inbound(_session_message_inbound("Cross review", message_id="msg-inj"))
            await asyncio.sleep(0.05)
            await _drain(loop, channel)
            assert not any(row.get("text") == "Cross review" for row in read_transcript_lines(KEY))

            release.set()
            await asyncio.wait_for(completed.wait(), timeout=5)
            await _drain(loop, channel)
        finally:
            release.set()
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)

        canonical = loop.sessions.read_existing_canonical_source(KEY)
        assert canonical is not None
        qualified = _hydrated(loop.sessions, KEY, canonical[1].state)
        texts = [
            row["content"] for row in replay_transcript_to_ui_messages(qualified)
            if row.get("content")
        ]
        assert texts == ["Setup prompt", "Setup reply", "First prompt", "Before inject", "Cross review", "After inject"]
        injected_row = next(row for row in qualified if row.get("text") == "Cross review")
        committed = next(row for row in canonical[1].events if row.get("content") == "Cross review")
        assert injected_row["_webui_source_prompt_id"] == committed["_event_id"]
        assert injected_row["session_message"]["message_id"] == "msg-inj"


async def test_delegated_child_activity_carries_root_origin_and_drops_after_replacement(
    engine_config, tmp_path,
):
    async with _gateway(engine_config, tmp_path, "Root answer") as (loop, channel, conn, _):
        await _submit(channel, conn, "Root prompt")
        await _dispatch_one(loop, channel)
        root = loop.sessions.read_saved_source_metadata(KEY)
        assert root is not None

        child_stub = SimpleNamespace(
            parent_key=KEY,
            root_key=KEY,
            session_key="delegated:child-1",
            child_id="child-1",
            profile="delegated",
            state="running",
            outcome=None,
            runtime_generation=1,
            task_label="Check source",
            task_label_truncated=False,
            result_summary=None,
            result_summary_truncated=False,
            attempt_id="run-1",
            current_run_id="crun-1",
            parent_source_state=root.state,
            root_source_state=root.state,
            source_state=None,
            run_sequence=1,
            state_sequence=1,
            control_epoch=1,
            holds_capacity=True,
            live=True,
            pending_approval_id=None,
            blocked_run_id=None,
            background_result=None,
        )
        token = bind_request_context(RequestContext(
            channel="websocket",
            chat_id=CHAT_ID,
            session_key=KEY,
            metadata={"webui_turn_id": "display-turn"},
        ))
        try:
            await loop._publish_delegated_child(child_stub)
        finally:
            reset_request_context(token)
        await _drain(loop, channel)

        qualified = read_transcript_lines(KEY, expected_source=root.state)
        delegated_rows = [row for row in qualified if row.get("event") == "delegated_child"]
        assert len(delegated_rows) == 1
        assert delegated_rows[0]["child_id"] == "child-1"

        # Orphan delegated activity without a root lifetime must not append unbound.
        await ChannelManager._send_once(channel, OutboundMessage(
            channel="websocket",
            chat_id=CHAT_ID,
            content="",
            event=DelegatedChildEvent(
                parent_session_key=KEY,
                child_session_key="delegated:orphan",
                child_id="orphan",
                profile="delegated",
                state="running",
                task_label="Orphan",
            ),
        ))
        assert not any(row.get("child_id") == "orphan" for row in read_transcript_lines(KEY))

        # Delayed delegated activity for the old root lifetime must drop after reset.
        assert loop.sessions.reset_session(KEY) is not None
        replaced = loop.sessions.read_saved_source_metadata(KEY)
        assert replaced is not None
        assert not replaced.state.same_origin(root.state)
        await ChannelManager._send_once(channel, OutboundMessage(
            channel="websocket",
            chat_id=CHAT_ID,
            content="",
            event=DelegatedChildEvent(
                parent_session_key=KEY,
                child_session_key="delegated:child-late",
                child_id="child-late",
                profile="delegated",
                state="completed",
                task_label="Late update",
                root_key=KEY,
                root_store_identity=root.state.store_identity,
                root_incarnation=root.state.incarnation,
                root_generation=root.state.generation,
            ),
        ))
        assert read_transcript_lines(KEY, expected_source=replaced.state) == []
        assert not any(row.get("child_id") == "child-late" for row in read_transcript_lines(KEY))


async def test_managed_fork_preserves_shell_slash_and_cross_session_origin_parity(
    engine_config, tmp_path,
):
    async with _gateway(engine_config, tmp_path, "Normal reply", "Cross reply", "Tail reply") as (
        loop, channel, conn, _,
    ):
        exec_tool = _StubExecTool()
        loop.tools.unregister("exec")
        loop.tools.register(exec_tool)
        await _submit(channel, conn, "Normal prompt", turn_id="t1")
        await _dispatch_one(loop, channel)
        await channel._dispatch_envelope(conn, "owner", {
            **_message_envelope(chat_id=CHAT_ID, content="!echo hi", turn_id="t2"),
            "user_shell": True,
        })
        await _dispatch_one(loop, channel)
        await _submit(channel, conn, "/help", turn_id="t3")
        await _dispatch_one(loop, channel)
        await loop._dispatch(_session_message_inbound("Cross prompt", message_id="msg-fork"))
        await _drain(loop, channel)
        await _submit(channel, conn, "Excluded tail", turn_id="t5")
        await _dispatch_one(loop, channel)

        parent = loop.sessions.read_existing_canonical_source(KEY)
        assert parent is not None
        parent_rows = read_transcript_lines(KEY, expected_source=parent[1].state)
        canonical_user_ids = [
            row["_event_id"] for row in parent[1].events if row.get("role") == "user"
        ]
        display_user_ids = [
            row.get("_webui_source_prompt_id") for row in parent_rows if row.get("event") == "user"
        ]
        assert display_user_ids == canonical_user_ids

        fork = create_webui_chat_fork(loop.sessions, source_chat_id=CHAT_ID, before_user_index=4)
        assert fork is not None
        target = loop.sessions.read_saved_source_metadata(fork[1])
        assert target is not None
        fork_rows = read_transcript_lines(fork[1], expected_source=target.state)
        assert [row["text"] for row in fork_rows if row.get("event") == "user"] == [
            "Normal prompt", "!echo hi", "/help", "Cross prompt",
        ]
        assert any(row.get("text") == "ran:echo hi" for row in fork_rows)
        assert fork_rows[-1]["event"] == "fork_marker"
