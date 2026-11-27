"""Delegated child card facts: short label, final-reply summary, partial outcome, run totals."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kohaku.agent import child_sessions
from kohaku.agent.child_sessions import ChildAdmission, _parse_child_admission
from kohaku.agent.delegation import ForegroundAgentManager
from kohaku.agent.runner import AgentRunResult
from kohaku.agent.tools.delegation import DelegateTaskTool
from kohaku.agent.tools.explore import ExploreTool
from kohaku.session.source import new_source_event
from tests.agent.test_delegation_child_sessions import _new_parent, _new_tree


async def _admit(tree, parent, assignment="inspect the fixture", *, role="explorer", **kwargs):
    admission = await tree.reserve(
        parent=parent,
        parent_key=parent.key,
        assignment=assignment,
        attempt_id=f"attempt-{role}-{len(tree.children_of(parent.key))}",
        policy_facts={"role": role},
        **kwargs,
    )
    assert isinstance(admission, ChildAdmission)
    return admission


async def _running(tree, parent, *, role="explorer", run_id="run-1"):
    admission = await _admit(tree, parent, role=role)
    assert await tree.activate(admission.child_id) is not None
    started = await tree.register_run(admission.child_id, run_id)
    assert started is not None
    return started


def _append(sessions, tree, child_id, run_id, *messages):
    child = tree.child(child_id)
    backend = sessions.source_backend(child.session_key)
    snapshot = backend.source.load_source_snapshot(child.session_key)
    events = [new_source_event(message, run_id=run_id) for message in messages]
    for event in events:
        # The source constructor strips untrusted links; emulate runtime linkage.
        if calls := event.get("tool_calls"):
            event["_execution_ids"] = [f"exec-{run_id}-{call['id']}" for call in calls]
        elif event["role"] == "tool":
            event["_execution_id"] = f"exec-{run_id}-{event['tool_call_id']}"
    backend.source.append_source_events(
        child.session_key,
        snapshot.state,
        events,
    )
    sessions.invalidate(child.session_key)


def _tool_step(call_id):
    return (
        {
            "role": "assistant",
            "content": "Checking the fixture.",
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "name": "read_file", "content": "data"},
    )


def _report(partial, summary="Traced the flow."):
    return (
        "---EXPLORE---\n"
        + json.dumps({"summary": summary, "findings": ["One route."], "partial": partial})
        + "\n---END---"
    )


# Short label


@pytest.mark.parametrize(
    ("assignment", "expected"),
    [
        ("Inspect the loader. Then report back.", "Inspect the loader"),
        ("  inspect   the source\nfor edge cases  ", "inspect the source for edge cases"),
        ("Read {root}/src/app.py and {root}/README.md.", "Read src/app.py and README.md"),
        ("Audit {root} for dead code. Report.", "Audit repo for dead code"),
        ("Audit {root}-old/a.py now.", "Audit {root}-old/a.py now"),
        ("## Objective\n\nFix the login redirect.\n\n## Evidence\nlogs", "Fix the login redirect"),
        ("Objective:\nPlan the migration\n\nAvailable context:\nnone", "Plan the migration"),
        ("# Fix the cache key\n\nDetails follow.", "Fix the cache key"),
        ("Compare e.g. the two loaders. Done.", "Compare e.g. the two loaders"),
        ("检查加载器。然后汇报。", "检查加载器。"),
    ],
)
async def test_default_label_is_the_first_sentence_with_relative_paths(
    tmp_path, assignment, expected,
):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    root = str(tmp_path / "repo")
    expected = expected.format(root=root)
    assignment = assignment.format(root=root)

    admission = await _admit(tree, parent, assignment, workspace_roots=(root,))

    if len(expected) <= 60:
        assert admission.task_label == expected
        assert admission.task_label_truncated is False
    else:
        assert len(admission.task_label) <= 60
        assert admission.task_label.endswith("…")
    assert admission.assignment == assignment


async def test_long_default_label_is_cut_at_a_word_within_sixty_characters(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    assignment = "Investigate " + "the delegated child projection " * 5

    admission = await _admit(tree, parent, assignment)

    assert len(admission.task_label) <= 60
    assert admission.task_label.endswith("…")
    assert assignment.startswith(admission.task_label[:-1] + " ")
    assert admission.task_label_truncated is True


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("  Trace   the loader ", "Trace the loader"),
        ("   ", "inspect the fixture"),
        ("x" * 80, "x" * 59 + "…"),
    ],
)
async def test_explicit_label_overrides_the_default_and_is_persisted(tmp_path, label, expected):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)

    admission = await _admit(tree, parent, label=label)

    assert admission.task_label == expected
    saved = sessions.read_session_metadata(admission.session_key)["metadata"]["delegation_child"]
    assert saved["task_label"] == expected
    assert saved["assignment"] == "inspect the fixture"


@pytest.mark.parametrize(
    ("tool_class", "method", "arguments"),
    [
        (ExploreTool, "run_explore", {"task": "Read the fixture source."}),
        (
            DelegateTaskTool,
            "run_worker",
            {"contract": "Implement the fixture and run checks.", "write_scope": ["out.txt"]},
        ),
    ],
)
async def test_role_tools_document_and_forward_the_label(tool_class, method, arguments):
    manager = SimpleNamespace(**{method: AsyncMock(return_value={"summary": "ok"})})
    tool = (
        tool_class(manager, max_iterations=3) if tool_class is ExploreTool else tool_class(manager)
    )
    schema = tool.parameters["properties"]["label"]
    assert schema["type"] == "string"
    assert schema["maxLength"] == 60
    assert "short title shown in the UI" in schema["description"].lower().replace("ui", "UI")
    assert "label" not in tool.parameters["required"]
    assert tool.validate_params({**arguments, "label": "Fixture"}) == []
    assert tool.validate_params({**arguments, "label": "x" * 61})

    await tool.execute(**arguments, label="Fixture")
    assert getattr(manager, method).await_args.kwargs["label"] == "Fixture"
    context = SimpleNamespace()
    await tool.execute_with_context(context, **arguments, label="Context fixture")
    assert getattr(manager, method).await_args.kwargs["label"] == "Context fixture"


# Result summary


async def test_summary_is_the_final_reply_with_newlines(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    child = await _running(tree, parent, role="worker")
    final = "# Report\r\n\r\n\r\n\r\n- first  item\n  - nested\n"
    _append(
        sessions, tree, child.child_id, "run-1",
        *_tool_step("call-1"), {"role": "assistant", "content": final},
    )

    # External harnesses report every text segment as one result.
    assert tree.record_transcript(
        child.child_id, assignment="a", result_text="Checking the fixture." + final,
    )

    updated = tree.child(child.child_id)
    assert updated.result_summary == "# Report\n\n- first  item\n  - nested"
    assert updated.result_summary_truncated is False


async def test_summary_keeps_an_error_result_over_an_unrelated_reply(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    child = await _running(tree, parent, role="worker")
    _append(sessions, tree, child.child_id, "run-1", {"role": "assistant", "content": "Early."})

    assert tree.record_transcript(
        child.child_id, assignment="a", result_text="Error: worker reached its limit.",
    )

    assert tree.child(child.child_id).result_summary == "Error: worker reached its limit."


async def test_summary_is_capped_at_one_thousand_characters(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    child = await _running(tree, parent, role="worker")

    assert tree.record_transcript(child.child_id, assignment="a", result_text="line\n" * 400)

    updated = tree.child(child.child_id)
    assert len(updated.result_summary) == 1000
    assert updated.result_summary.startswith("line\nline\n")
    assert updated.result_summary_truncated is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("## Findings\n\n- one\n\n" + _report(True), "## Findings\n\n- one"),
        (_report(False), "Traced the flow.\n\n- One route."),
        ("---EXPLORE---\nnot json\n---END---", "---EXPLORE---\nnot json\n---END---"),
    ],
)
async def test_explore_summary_hides_the_machine_report(tmp_path, text, expected):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    child = await _running(tree, parent)

    assert tree.record_transcript(child.child_id, assignment="a", result_text=text)

    assert tree.child(child.child_id).result_summary == expected


# Partial outcome


def _manager(tree):
    manager = ForegroundAgentManager.__new__(ForegroundAgentManager)
    manager._delegation_tree = tree
    manager._emit_child = AsyncMock()
    return manager


@pytest.mark.parametrize(
    ("role", "stop_reason", "content", "expected"),
    [
        ("explorer", "completed", _report(True), ("completed", "partial")),
        ("explorer", "completed", "narration only", ("completed", "partial")),
        ("explorer", "completed", _report(False), ("completed", "completed")),
        ("worker", "completed", "plain report", ("completed", "completed")),
        ("worker", "repeated_no_progress", "Stopped.", ("completed", "partial")),
        ("explorer", "max_iterations", None, ("failed", "max_iterations")),
        ("explorer", "cancelled", None, ("stopped", "cancelled")),
    ],
)
async def test_partial_results_complete_with_a_partial_outcome(
    tmp_path, role, stop_reason, content, expected,
):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    child = await _running(tree, parent, role=role)
    result = AgentRunResult(final_content=content, messages=[], stop_reason=stop_reason)

    await _manager(tree)._finalize_admission(
        child, result, role=role, assignment="a", source_writer=None, canonical=True,
    )

    updated = tree.child(child.child_id)
    assert (updated.state, updated.outcome) == expected
    assert result.policy_metadata["delegated_child"]["outcome"] == expected[1]


@pytest.mark.parametrize(
    ("content", "outcome"),
    [("A short follow-up answer.", "completed"), (_report(True), "partial")],
)
async def test_followup_run_is_partial_only_when_its_report_says_so(tmp_path, content, outcome):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    child = await _running(tree, parent)
    assert await tree.complete(child.child_id, outcome="partial")
    # A partial child is a healthy settled child: it can claim a follow-up run.
    claimed = await tree.begin_followup_run(
        child.child_id,
        run_id="run-2",
        expected_run_id="run-1",
        expected=tree.capture_child_control_state(child.child_id),
    )
    assert claimed is not None
    assert claimed.outcome is None
    assert await tree.activate(child.child_id)

    await _manager(tree)._finalize_admission(
        claimed,
        AgentRunResult(final_content=content, messages=[], stop_reason="completed"),
        role="explorer", assignment="more", source_writer=None, canonical=True,
    )

    assert tree.child(child.child_id).outcome == outcome


async def test_repeated_attempt_recovers_a_partial_result(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    child = await _running(tree, parent)
    assert tree.record_transcript(child.child_id, assignment="a", result_text="half")
    partial = await tree.complete(child.child_id, outcome="partial")

    result = await _manager(tree)._recover_existing_attempt(
        partial, role="explorer", background=False,
    )

    assert (result.stop_reason, result.final_content) == ("completed", "half")


# Run totals


async def test_finished_run_persists_its_own_duration_and_step_count(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(child_sessions.time, "time", lambda: clock[0])
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    admission = await _admit(tree, parent)
    assert await tree.activate(admission.child_id)
    clock[0] = 1010.0
    live = await tree.register_run(admission.child_id, "run-1")
    assert live.duration_ms is None and live.step_count is None
    assert "duration_ms" not in live.metadata() and "step_count" not in live.metadata()
    _append(sessions, tree, live.child_id, "run-1", *_tool_step("a"), *_tool_step("b"))
    clock[0] = 1072.5

    done = await tree.complete(live.child_id, outcome="completed")

    assert (done.duration_ms, done.step_count) == (62_500, 2)
    saved = sessions.read_session_metadata(done.session_key)["metadata"]["delegation_child"]
    assert (saved["duration_ms"], saved["step_count"]) == (62_500, 2)
    parsed = _parse_child_admission(done.session_key, saved)
    assert (parsed.duration_ms, parsed.step_count) == (62_500, 2)
    # Later bookkeeping moves updated_at, not the run totals.
    clock[0] = 9000.0
    assert tree.record_transcript(done.child_id, assignment="a", result_text="late")
    assert tree.child(done.child_id).duration_ms == 62_500

    # A follow-up run measures only itself.
    claimed = await tree.begin_followup_run(
        done.child_id,
        run_id="run-2",
        expected_run_id="run-1",
        expected=tree.capture_child_control_state(done.child_id),
    )
    assert claimed.duration_ms is None and claimed.step_count is None
    assert await tree.activate(done.child_id)
    _append(sessions, tree, done.child_id, "run-2", *_tool_step("c"))
    clock[0] = 9004.0
    again = await tree.complete(done.child_id, outcome="completed")
    assert (again.duration_ms, again.step_count) == (4_000, 1)


async def test_uncertain_and_legacy_records_have_no_run_totals(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    child = await _running(tree, parent)
    uncertain = await tree.complete(
        child.child_id, outcome="settlement_unconfirmed", state="uncertain",
    )
    assert uncertain.duration_ms is None
    settled = await tree.settle_uncertain(
        child.child_id, state="completed", outcome="recovered_completed",
    )
    assert settled.duration_ms is None and settled.step_count is None

    saved = settled.metadata()
    parsed = _parse_child_admission(settled.session_key, saved)
    assert parsed.duration_ms is None and parsed.step_count is None
    for field, value in (("duration_ms", -1), ("step_count", "3")):
        broken = _parse_child_admission(settled.session_key, {**saved, field: value})
        assert getattr(broken, field) is None


async def test_child_event_wire_transcript_and_hydration_carry_run_totals():
    from dataclasses import replace

    from kohaku.agent.loop import AgentLoop
    from kohaku.agent.tools.context import (
        RequestContext,
        bind_request_context,
        reset_request_context,
    )
    from kohaku.webui.outbound_wire import encode_delegated_child
    from kohaku.webui.transcript import (
        replay_transcript_to_ui_messages,
        validate_retained_transcript_record,
    )
    from kohaku.webui.ws_http import _hydrated_child_payload
    from tests.agent.test_delegated_child_events import _make_admission

    legacy = _make_admission()
    finished = replace(legacy, duration_ms=62_500, step_count=2)
    empty = replace(legacy, duration_ms=0, step_count=0)
    for admission, expected in ((finished, (62_500, 2)), (empty, (0, 0)), (legacy, (None, None))):
        loop = AgentLoop.__new__(AgentLoop)
        loop.bus = SimpleNamespace(publish_outbound=AsyncMock())
        token = bind_request_context(RequestContext(
            channel="websocket", chat_id="chat-a", session_key="websocket:chat-a",
        ))
        try:
            await loop._publish_delegated_child(admission)
        finally:
            reset_request_context(token)
        event = loop.bus.publish_outbound.await_args.args[0].event
        assert (event.duration_ms, event.step_count) == expected
        wire = dict(encode_delegated_child("chat-a", event))
        validate_retained_transcript_record(wire)
        [row] = [
            message["delegatedChild"]
            for message in replay_transcript_to_ui_messages([wire])
            if message.get("delegatedChild")
        ]
        hydrated = _hydrated_child_payload(
            admission.session_key,
            admission.metadata(),
            parent_key=admission.parent_key,
            root_key=admission.root_key,
            parent_source=None,
            root_source=None,
            child_source=None,
            runtime_executable=False,
        )
        for payload in (wire, row, hydrated):
            assert (payload.get("duration_ms"), payload.get("step_count")) == expected
            assert ("duration_ms" in payload) is (expected[0] is not None)
            assert ("step_count" in payload) is (expected[1] is not None)
    for invalid in (-1, True, 1.2, "3", None, 2**53):
        with pytest.raises(ValueError):
            validate_retained_transcript_record({**wire, "step_count": invalid})
        [row] = replay_transcript_to_ui_messages([{**wire, "step_count": invalid}])
        assert "step_count" not in row["delegatedChild"]

    def replay(*updates):
        [message] = replay_transcript_to_ui_messages(updates)
        return message["delegatedChild"]

    with_totals = {**wire, "duration_ms": 62_500, "step_count": 2}
    assert replay(wire, with_totals)["step_count"] == 2
    later = {**wire, "state_sequence": wire["state_sequence"] + 1}
    assert replay(with_totals, later)["step_count"] == 2
    followup = {
        **wire, "run_sequence": wire["run_sequence"] + 1,
        "child_run_id": "child-run-b", "state": "active", "outcome": None,
    }
    assert "step_count" not in replay(with_totals, followup, with_totals)
