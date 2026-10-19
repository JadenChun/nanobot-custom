"""Existing turn guards and trace/output views retain their runtime behavior."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import RunPolicyContext, TraceMode, TurnRequest
from nanobot.agent.turn import TurnContext as RunContext
from nanobot.bus.events import InboundMessage
from nanobot.events import NO_EVENTS
from nanobot.session.manager import Session
from nanobot.utils.llm_runtime import LLMRuntime
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_fresh_turn import stage


@pytest.mark.parametrize("entry", ["execute_turn", "execute_silent_cron_turn"])
@pytest.mark.parametrize("payload", [None, False, 3, "task", [], {}])
async def test_invalid_request_is_refused_before_any_lifecycle_or_route_read(entry, payload) -> None:
    loop = object.__new__(AgentLoop)

    def unavailable(*args, **kwargs):
        pytest.fail("invalid request reached lifecycle or route admission")

    loop._lifecycle_is_closing = unavailable
    loop._request_route = unavailable
    with pytest.raises(TypeError, match="TurnRequest"):
        await getattr(loop, entry)(payload)


async def test_missing_foreground_dependencies_are_refused_before_routing() -> None:
    loop = object.__new__(AgentLoop)
    loop._fork_dependencies = None
    loop._request_route = lambda *_: pytest.fail("missing dependencies reached routing")
    with pytest.raises(ValueError, match="foreground dependencies"):
        await loop.execute_turn(TurnRequest(content="task"))


def _binding_inputs(tmp_path: Path, *, definitions=(), fault=None):
    calls = []

    class Roots:
        def read_roots(self):
            calls.append("read_roots")
            if fault == "read_roots":
                raise RuntimeError("read root failure")
            return [str(tmp_path)]

        def write_roots(self):
            calls.append("write_roots")
            if fault == "write_roots":
                raise RuntimeError("write root failure")
            return [str(tmp_path)]

    class Definitions(ToolRegistry):
        def get_definitions(self):
            calls.append("definitions")
            if fault == "definitions":
                raise RuntimeError("definition failure")
            return [] if definitions == () else definitions

    context = RunContext(
        request=TurnRequest(content="task"),
        policy=RunPolicyContext(workspace=tmp_path, context_manager=Roots()),
        lock_owner="main:fixture",
    )
    provider = ScriptedProvider([])
    runtime = LLMRuntime.capture(provider, "fixture", context_window_tokens=8192)
    return {
        "session": Session("cli:fixture"), "runtime": runtime, "run_context": context,
        "workspace": tmp_path, "tools": Definitions(), "ephemeral": False,
    }, calls


@pytest.mark.parametrize("model", [None, False, 3, [], {}, object()])
def test_invalid_model_retains_early_binding_refusal(tmp_path, model) -> None:
    values, calls = _binding_inputs(tmp_path)
    values["runtime"] = replace(values["runtime"], model=model)
    assert AgentLoop._conversation_state_binding(**values) is None
    assert calls == []


@pytest.mark.parametrize("definitions", [None, False, 3, "schema", {}, object()])
def test_nonlist_definitions_are_refused_after_the_original_reads(tmp_path, definitions) -> None:
    values, calls = _binding_inputs(tmp_path, definitions=definitions)
    assert AgentLoop._conversation_state_binding(**values) is None
    assert calls == ["read_roots", "write_roots", "definitions"]


@pytest.mark.parametrize("fault,reads", [
    ("read_roots", ["read_roots"]),
    ("write_roots", ["read_roots", "write_roots"]),
    ("definitions", ["read_roots", "write_roots", "definitions"]),
])
def test_binding_keeps_the_original_catch_and_short_circuit(tmp_path, fault, reads) -> None:
    values, calls = _binding_inputs(tmp_path, fault=fault)
    assert AgentLoop._conversation_state_binding(**values) is None
    assert calls == reads


def test_definition_copy_failure_is_still_a_binding_refusal(tmp_path) -> None:
    calls = []

    class BrokenCopy(list):
        def __deepcopy__(self, memo):
            calls.append("deepcopy")
            raise RuntimeError("copy failure")

    values, reads = _binding_inputs(tmp_path, definitions=BrokenCopy([{"opaque": "value"}]))
    assert AgentLoop._conversation_state_binding(**values) is None
    assert reads == ["read_roots", "write_roots", "definitions"] and calls == ["deepcopy"]


def test_valid_binding_retains_model_and_order_without_mutating_definitions(tmp_path) -> None:
    definitions = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
    values, calls = _binding_inputs(tmp_path, definitions=definitions)
    before = deepcopy(definitions)
    binding = AgentLoop._conversation_state_binding(**values)
    assert binding is not None and binding.runtime_prefix["model"] == "fixture"
    assert binding.tools == definitions and binding.tools is not definitions
    assert definitions == before and calls == ["read_roots", "write_roots", "definitions"]


class _Message:
    def __init__(self, content: object, *, matching: bool, second=None) -> None:
        self.value, self.matching, self.second = content, matching, second
        self.reads = []
        self.content_reads = 0

    @property
    def channel(self):
        self.reads.append("channel")
        return "telegram" if self.matching else "elsewhere"

    @property
    def chat_id(self):
        self.reads.append("chat_id")
        return "42"

    @property
    def content(self):
        self.reads.append("content")
        self.content_reads += 1
        return self.second if self.content_reads > 1 and self.second is not None else self.value


async def _run_output(message):
    loop = object.__new__(AgentLoop)
    context = stage()
    context.delivery.running = AsyncMock()
    context.delivery.record_usage = lambda *_: None
    context.delivery.route = SimpleNamespace(channel="telegram", chat_id="42")
    context.transcript_input = object()
    context.run_context.delivery.sent_messages.append(message)
    loop._run_agent_loop = AsyncMock(return_value=AgentRunResult(final_content="final", messages=[]))
    await loop._run_turn(context)
    return context


@pytest.mark.parametrize("matching", [False, True])
@pytest.mark.parametrize("content", [None, False, 3, [], {}, " final "])
async def test_canonical_content_validation_keeps_route_and_read_order(matching, content) -> None:
    message = _Message(content, matching=matching)
    context = await _run_output(message)
    assert context.suppress_response is (matching and isinstance(content, str))
    expected = ["channel", "chat_id"]
    if matching:
        expected += ["content"] + (["content"] if isinstance(content, str) else [])
    assert message.reads == expected


async def test_second_content_read_keeps_the_original_strip_fault() -> None:
    failure = RuntimeError("strip failure")

    class BrokenStrip(str):
        def strip(self, *args, **kwargs):
            raise failure

    message = _Message("final", matching=True, second=BrokenStrip("final"))
    with pytest.raises(RuntimeError) as raised:
        await _run_output(message)
    assert raised.value is failure
    assert message.reads == ["channel", "chat_id", "content", "content"]


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_trace_collector_retains_malformed_rows_identity_selection_and_finally(source, failure) -> None:
    trigger = {"role": "user", "content": "expanded input", "_event_id": "opaque"}
    equal_but_distinct = dict(trigger)
    rows = [trigger, equal_but_distinct, None, 3, [], {"role": []},
            {"role": "assistant", "content": "public", "reasoning_content": "private"}]
    before = deepcopy(rows)
    run = RunContext(request=TurnRequest(content="request text", trace_mode=TraceMode.SANITIZED))
    if source:
        # Only the trace-view collaborator is doubled; no source authority,
        # admission or publication is represented by this narrow unit fixture.
        run.source_writer = SimpleNamespace(committed_events=rows)
    loop = object.__new__(AgentLoop)

    async def restore(context):
        context.trace_input_message = trigger
        context.trace_detail_messages.extend(rows)

    failed = ValueError("build failure")

    async def build(context):
        if failure:
            raise failed

    loop._restore_turn = restore
    loop._compact_session = AsyncMock()
    loop._dispatch_command = AsyncMock(return_value=False)
    loop._build_turn = build
    loop._run_turn = AsyncMock()
    loop._persist_turn = AsyncMock()
    loop._prepare_outbound = AsyncMock()
    delivery = SimpleNamespace(session_key="telegram:42", streaming=False, events=NO_EVENTS,
                               record_latency=lambda *_: None)
    request = InboundMessage(channel="telegram", sender_id="user", chat_id="42", content="expanded input")
    if failure:
        with pytest.raises(ValueError) as raised:
            await loop._process_turn(request, delivery=delivery, run_context=run)
        assert raised.value is failed
    else:
        await loop._process_turn(request, delivery=delivery, run_context=run)
    assert run.sanitized_messages == [
        {"role": "user", "content": "request text"},
        {"role": "user", "content": "expanded input"},
        {"role": "assistant", "content": "public"},
    ]
    assert rows == before and rows[0] is trigger and rows[1] is equal_but_distinct
