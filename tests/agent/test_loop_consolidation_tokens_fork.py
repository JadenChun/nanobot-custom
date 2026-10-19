"""Fork pressure invariants under P4 governance, not retired background trimming.

Canonical requests trigger at 90% and target 60% of available input capacity.
Legacy foreground compatibility compacts before its main model request;
neither path reinstates the old post-turn half-target memory reducer.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.context_governance import ContextGovernor
from nanobot.agent.tools.execution import KnownToolResult
from nanobot.providers.base import LLMResponse
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.source import SourceValidationError
from nanobot.session.source_projection import RuntimeNode, source_nodes
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript
from nanobot.session.turn_writer import SourceTurnWriter
from tests.agent import p6_admitted_fixtures
from tests.agent.test_loop_consolidation_tokens import _make_loop
from tests.agent.test_source_compaction_publication import store as store
from tests.agent.test_source_pressure_compaction import prepare, pressure_state

admitted_async = p6_admitted_fixtures.admitted_async


@pytest.mark.parametrize("offset", [-1, 0, 1])
async def test_canonical_pressure_threshold_preserves_source_and_current_input(store, offset):
    async with acquire_execution(store, "test:compact", run_id="run") as owner:
        state = pressure_state(SourceTurnWriter(store.source, owner, run_id="run"))
        trigger = int(ContextGovernor.input_budget(state.config) * 0.9)

        def count(messages, tools, model):
            auxiliary = messages[0]["content"].startswith("Summarize historical evidence")
            return (100 if auxiliary or "chunk-" not in str(messages) else trigger + offset,
                    "boundary fixture")

        state.config.provider.estimate_prompt_tokens = count
        before = owner.read_snapshot()
        messages, _ = await prepare(state)
        after = owner.read_snapshot()
        assert after.events == before.events
        assert messages[-1]["content"] == "current instruction"
        assert len(state.config.provider.calls) == (0 if offset < 0 else 1)
        if offset < 0:
            assert after == before and not state.source_compaction.attempts
            assert messages == state.source.transcript.messages()
        else:
            assert after.checkpoint["boundary"] == len(before.events) - 1
            assert after.checkpoint["summary"] == "retained facts"
            assert "chunk-" not in str(messages)
            assert state.source_compaction.attempts[-1].status == "succeeded"


async def test_pressure_keeps_call_and_result_together_and_excludes_new_user(store):
    async with acquire_execution(store, "test:compact", run_id="run") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="run")
        state = pressure_state(writer, chunks=0)
        writer.begin_tool_batch({"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-read", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"notes.txt"}'},
        }]})
        await writer.record_tool_result(KnownToolResult(
            0, "call-read", "read_file", "historical tool evidence",
            "historical tool evidence", {"status": "ok"}, True,
        ))
        writer.append_message({"role": "user", "content": "newest correction"})
        nodes = source_nodes(owner.read_snapshot().events)
        active = SourceTranscript((RuntimeNode("policy", {"role": "system", "content": "policy"}),
                                   *nodes))
        state.source = SourceRequestLedger(active,
            initial_eligible_ids=frozenset(node.event_id for node in nodes[:-1]))
        state.source_compaction.raw_messages[:] = active.messages()

        def count(messages, tools, model):
            auxiliary = messages[0]["content"].startswith("Summarize historical evidence")
            return (100 if auxiliary or "historical tool evidence" not in str(messages)
                    else 10000, "closed group fixture")

        state.config.provider.estimate_prompt_tokens = count
        before = owner.read_snapshot()
        messages, _ = await prepare(state)
        assert len(state.config.provider.calls) == 1
        evidence = json.loads(state.config.provider.calls[0][0][1]["content"])["events"]
        call_index = next(i for i, row in enumerate(evidence) if row.get("tool_calls"))
        assert evidence[call_index]["tool_calls"][0]["id"] == "call-read"
        assert evidence[call_index + 1]["tool_call_id"] == "call-read"
        assert "newest correction" not in str(evidence)
        assert messages[-1]["content"] == "newest correction"
        after = owner.read_snapshot()
        assert after.events == before.events
        assert after.checkpoint["boundary"] == len(before.events) - 1


@pytest.mark.parametrize(("fixed", "expected_calls"), [(4800, 2), (5500, 3)])
async def test_pressure_continues_below_trigger_but_stops_at_sixty_percent(
    store, fixed, expected_calls,
):
    async with acquire_execution(store, "test:compact", run_id="run") as owner:
        state = pressure_state(SourceTurnWriter(store.source, owner, run_id="run"))
        budget = ContextGovernor.input_budget(state.config)

        def count(messages, tools, model):
            auxiliary = messages[0]["content"].startswith("Summarize historical evidence")
            chunks = str(messages).count("chunk-")
            # Auxiliary capacity admits one whole chunk per call.
            tokens = 100 + chunks * 6000 if auxiliary else fixed + chunks * 1500
            if not auxiliary and chunks == 3:
                tokens = 10000
            return tokens, "target fixture"

        state.config.provider.estimate_prompt_tokens = count
        before = owner.read_snapshot()
        await prepare(state)
        metrics = state.source_compaction.attempts[-1]
        assert metrics.calls == expected_calls
        assert budget * 0.5 < metrics.after_tokens <= budget * 0.6
        assert owner.read_snapshot().events == before.events
        payloads = [json.loads(call[0][1]["content"]) for call in state.config.provider.calls]
        chunks = [row["content"] for payload in payloads for row in payload["events"]
                  if row["content"].startswith("chunk-")]
        assert chunks == [f"chunk-{i}" for i in range(expected_calls)]
        assert all("current instruction" not in str(payload["events"]) for payload in payloads)


@pytest.mark.parametrize("failure", ["empty", "cancelled"])
async def test_failed_pressure_does_not_publish_or_retry_in_background(store, failure):
    async with acquire_execution(store, "test:compact", run_id="run") as owner:
        state = pressure_state(SourceTurnWriter(store.source, owner, run_id="run"))
        state.config.provider.response = (
            LLMResponse(content="") if failure == "empty" else asyncio.CancelledError()
        )
        before, projection = owner.read_snapshot(), state.source.transcript
        with pytest.raises(SourceValidationError if failure == "empty" else asyncio.CancelledError):
            await prepare(state)
        assert len(state.config.provider.calls) == 1
        assert owner.read_snapshot() == before
        assert state.source.transcript == projection
        assert not state.source.has_pending_requests
        assert state.source_compaction.attempts[-1].status == (
            "failed" if failure == "empty" else "cancelled"
        )


async def test_foreground_loop_compacts_before_main_call_without_background_archival(
    tmp_path, admitted_async,
):
    loop = await _make_loop(tmp_path, admitted_async, estimated_tokens=100,
                            context_window_tokens=1624, max_tokens=100)
    loop.provider.can_resume_conversation_state.return_value = False
    session = loop.sessions.get_or_create("cli:test")
    original = [{"role": role, "content": f"old-{role}-{turn}"}
                for turn in range(3) for role in ("user", "assistant")]
    session.messages = list(original)
    loop.sessions.save(session)
    # Legacy saves allocate their own event identities. Compare compaction with
    # the actual persisted prefix, not the pre-save input dictionaries.
    before = loop.sessions.read_session_snapshot("cli:test")
    assert before is not None
    assert [{"role": row["role"], "content": row["content"]}
            for row in before.messages] == original
    order = []

    def count(messages, tools, model):
        if "SNIP" in str(messages[-1].get("content")):
            return 300, "fixture"
        return (600 if "old-user-" in str(messages) else 100), "fixture"

    async def chat(**kwargs):
        if "SNIP" in str(kwargs["messages"][-1].get("content")):
            order.append("summary")
            return LLMResponse(content="Retained objective and evidence.")
        order.append("main")
        assert "Retained objective and evidence." in str(kwargs["messages"])
        assert kwargs["messages"][-1]["content"] == "new correction"
        return LLMResponse(content="done")

    loop.provider.estimate_prompt_tokens.side_effect = count
    loop.provider.chat_with_retry = AsyncMock(side_effect=chat)
    loop.consolidator.compact_idle_session = AsyncMock()
    result = await loop.process_direct("new correction", session_key="cli:test")
    assert result.content == "done"
    assert order == ["summary", "main"]
    assert loop.auto_compact is None
    loop.consolidator.compact_idle_session.assert_not_awaited()
    assert not loop._background_tasks
    saved = loop.sessions.get_or_create("cli:test")
    assert saved.messages[:len(before.messages)] == before.messages
    assert saved.metadata["_last_summary"]["text"] == "Retained objective and evidence."
    assert [row["content"] for row in saved.get_history()][-2:] == ["new correction", "done"]
