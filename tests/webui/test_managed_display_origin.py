"""The gateway must persist display rows under the source that produced them."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.outbound_events import MessageAdmittedEvent
from nanobot.bus.runtime_events import TurnCompleted
from nanobot.channels.manager import ChannelManager
from nanobot.providers.base import LLMResponse
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.session.source_store import SourceSessionStore
from nanobot.session.webui_turns import WebuiTurnCoordinator
from nanobot.webui.forking import create_webui_chat_fork
from nanobot.webui.transcript import (
    append_transcript_object,
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
            qualified = read_transcript_lines(KEY, expected_source=source.state)
            assert qualified[0]["text"] == "First prompt"
            assert any(row.get("text") == "First answer" for row in qualified)
            assert qualified[-1]["event"] == "turn_end"
            assert [row["text"] for row in prompts] == ["First prompt"]
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
            for row in read_transcript_lines(KEY, expected_source=current.state)
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
        assert [row["text"] for row in read_transcript_lines(KEY, expected_source=title_edit.state)
                if row.get("event") == "user"] == ["Old prompt"]
        assert loop.sessions.delete_session(KEY)
        await _submit(channel, conn, "New prompt", turn_id="new-display-turn")
        await _dispatch_one(loop, channel)
        current = loop.sessions.read_saved_source_metadata(KEY)
        assert current is not None
        assert [row["text"] for row in read_transcript_lines(KEY, expected_source=current.state)
                if row.get("event") == "user"] == ["New prompt"]
        assert {row.get("text") for row in read_transcript_lines(KEY)} >= {"Unbound input", "Old prompt"}

    async with _gateway(engine_config, tmp_path) as (loop, _, _, _):
        current = loop.sessions.read_saved_source_metadata(KEY)
        assert current is not None
        assert [row["text"] for row in read_transcript_lines(KEY, expected_source=current.state)
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
                if row.get("event") == "user" and row.get("text") == "Guidance":
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
            rows = read_transcript_lines(KEY, expected_source=origin)
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
            if session_key == KEY and row.get("text") == "Committed answer":
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
