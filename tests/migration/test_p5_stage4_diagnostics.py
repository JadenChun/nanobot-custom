"""Pure two-plane privacy and bounded copied-evidence checks."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from nanobot._integration_diagnostics import (
    MAX_BYTES,
    MAX_DEPTH,
    MAX_MESSAGES,
    MAX_TOOL_CALLS,
    PublicEvidence,
    canonical_evidence,
    compare,
    diagnostic_payload,
    legacy_evidence,
    operator_compare,
)
from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import OfflineSessionOperator
from nanobot.session.source_store import SourceSnapshot, SourceState
from tests.migration.p5_stage4_fixtures import (
    Effects,
    config_at,
    file_bytes,
    offline_denial,  # noqa: F401
)


def test_equal_routine_report_has_only_classifications_and_counts():
    evidence = legacy_evidence([{"role": "user", "content": "private sentinel"}], 0)
    report = compare(evidence, evidence)
    assert report == {"component": "shadow", "result": "equal", "left_count": 1, "right_count": 1}
    assert "private sentinel" not in repr(evidence)
    assert diagnostic_payload(report) == report


SENTINEL = "PRIVATE-secret-/path/user:42-artifact:reference-receipt-token"


@pytest.mark.parametrize("field", ["session_id", "path", "digest", "prompt", "receipt", "credential", SENTINEL])
def test_unknown_dynamic_labels_reject_without_echo(field):
    with pytest.raises(ValueError) as error:
        diagnostic_payload({"component": "shadow", "result": "equal", "left_count": 0,
                            "right_count": 0, field: SENTINEL})
    assert str(error.value) == "diagnostic payload rejected" and SENTINEL not in repr(error.value)


@pytest.mark.parametrize("value", [True, False, 1.0, -1, 1000001, None, SENTINEL, float("nan")])
@pytest.mark.parametrize("field", ["left_count", "right_count"])
def test_counts_are_exact_bounded_integers(field, value):
    payload = {"component": "session", "result": "equal", "left_count": 0, "right_count": 0}
    payload[field] = value
    with pytest.raises(ValueError, match="^diagnostic payload rejected$"):
        diagnostic_payload(payload)


@pytest.mark.parametrize("field", ["component", "result"])
def test_label_values_are_finite_enums(field):
    payload = {"component": "shadow", "result": "equal", "left_count": 0, "right_count": 0}
    payload[field] = SENTINEL
    with pytest.raises(ValueError) as error:
        diagnostic_payload(payload)
    assert SENTINEL not in str(error.value)


def tool_history():
    return [{"role": "user", "content": SENTINEL},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tool-1", "type": "function", "function": {"name": "fake", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "tool-1", "name": "fake", "content": "fake evidence"},
            {"role": "assistant", "content": "answer"}]


def test_frozen_boundary_routine_and_private_operator_planes():
    rows = tool_history()
    before = deepcopy(rows)
    left = legacy_evidence(rows, 0)
    right = legacy_evidence(deepcopy(rows), 0)
    rows[-1]["content"] = "changed after freeze"
    report = compare(left, right)
    assert report["result"] == "equal"
    private = operator_compare(left, right)
    wire = json.dumps(before, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    expected = hashlib.sha256(wire + b"\x000").hexdigest()
    assert private.left_digest == private.right_digest == expected
    assert expected not in repr(private) and SENTINEL not in repr(left)
    assert "digest" not in repr(report) and SENTINEL not in repr(private)
    changed = legacy_evidence(rows, 0)
    assert compare(left, changed)["result"] == "public_history"
    assert compare(left, legacy_evidence(before, 1))["result"] == "archive_boundary"
    paired = deepcopy(before)
    paired[1]["tool_calls"][0]["id"] = "new-id"
    paired[2]["tool_call_id"] = "new-id"
    assert compare(left, legacy_evidence(paired, 0))["result"] == "tool_pairing"


@pytest.mark.parametrize("attack", ["unknown", "role", "content", "bool-boundary", "missing-tool",
                                    "unpaired", "duplicate", "nan", "cycle", "depth", "bytes", "messages", "calls"])
def test_malformed_unsupported_and_oversized_trees_are_constant_and_unchanged(attack):
    rows = tool_history()
    boundary = 0
    expected = "invalid_input"
    if attack == "unknown":
        rows[0][SENTINEL] = SENTINEL
        expected = "unsupported"
    elif attack == "role":
        rows[0]["role"] = SENTINEL
        expected = "unsupported"
    elif attack == "content":
        rows[0]["content"] = {"private": SENTINEL}
        expected = "unsupported"
    elif attack == "bool-boundary":
        boundary = True
    elif attack == "missing-tool":
        rows[2]["tool_call_id"] = "unknown"
        expected = "tool_pairing"
    elif attack == "unpaired":
        rows.pop(2)
        expected = "tool_pairing"
    elif attack == "duplicate":
        rows[1]["tool_calls"] *= 2
        expected = "tool_pairing"
    elif attack == "nan":
        rows[0]["content"] = float("nan")
    elif attack == "cycle":
        rows.append(rows)
        expected = "bound_exceeded"
    elif attack == "depth":
        nested = []
        for _ in range(MAX_DEPTH + 1):
            nested = [nested]
        rows[0]["content"] = nested
        expected = "bound_exceeded"
    elif attack == "bytes":
        rows[0]["content"] = "x" * (MAX_BYTES + 1)
        expected = "bound_exceeded"
    elif attack == "messages":
        rows = [{"role": "user", "content": "x"}] * (MAX_MESSAGES + 1)
        expected = "bound_exceeded"
    elif attack == "calls":
        rows = [{"role": "assistant", "content": None, "tool_calls": [
            {"id": str(index), "type": "function", "function": {"name": "fake", "arguments": "{}"}}
            for index in range(MAX_TOOL_CALLS + 1)]}]
        expected = "bound_exceeded"
    # repr supports cycles and retains exact dynamic input without normalizing it.
    before = repr(rows)
    evidence = legacy_evidence(rows, boundary)
    report = compare(evidence, legacy_evidence([], 0))
    assert report["result"] == expected and repr(rows) == before
    assert SENTINEL not in repr(evidence) and SENTINEL not in repr(report)
    assert operator_compare(evidence, evidence).left_digest is None


async def test_actual_conversion_retained_root_public_projection_is_pure(tmp_path, monkeypatch):
    config = config_at(tmp_path)
    root = config.runtime_data_dir / "sessions"
    rows = tool_history()
    manager = SessionManager(config.workspace_path, sessions_root=root)
    manager.save(Session("cli:copied", messages=rows, last_consolidated=0), fsync=True)
    operator = await OfflineSessionOperator(config.workspace_path, copied_sessions_root=root).acquire()
    operator.manager.bootstrap_existing_sessions(offline_copy=True)
    legacy = legacy_evidence(rows, 0)
    operator.manager.convert_legacy_session("cli:copied", offline_copy=True)
    snapshot = operator.manager._jsonl_store.source.load_source_snapshot("cli:copied")
    await operator.close()
    before_tree = file_bytes(tmp_path)
    before_snapshot = deepcopy(snapshot)
    effects = Effects()

    def no_effect(*_args, **_kwargs):
        raise AssertionError("shadow attempted store/effect access")

    monkeypatch.setattr(SessionManager, "__init__", no_effect)
    canonical = canonical_evidence(snapshot)
    assert compare(legacy, canonical)["result"] == "equal"
    # A checkpoint summary is not the equivalence object. Retained full history wins.
    summary = replace(snapshot, checkpoint={"summary": SENTINEL, "boundary": 4,
                                          "projection_scope": "root", "projection_version": 1})
    assert compare(canonical, canonical_evidence(summary))["result"] == "equal"
    snapshot.events[-1]["content"] = "different public answer"
    assert compare(canonical, canonical_evidence(snapshot))["result"] == "public_history"
    snapshot.events[-1]["content"] = before_snapshot.events[-1]["content"]
    assert snapshot == before_snapshot and file_bytes(tmp_path) == before_tree
    assert effects == Effects()


@pytest.mark.parametrize("attack", ["unknown", "role", "role-list", "role-dict", "role-bool", "role-int",
                                    "tool-link", "event-id", "bytes", "depth"])
async def test_mutable_canonical_snapshot_is_revalidated_before_projection(tmp_path, attack):
    config = config_at(tmp_path)
    root = config.runtime_data_dir / "sessions"
    manager = SessionManager(config.workspace_path, sessions_root=root)
    manager.save(Session("cli:copied", messages=tool_history()), fsync=True)
    operator = await OfflineSessionOperator(config.workspace_path, copied_sessions_root=root).acquire()
    operator.manager.bootstrap_existing_sessions(offline_copy=True)
    operator.manager.convert_legacy_session("cli:copied", offline_copy=True)
    snapshot = operator.manager._jsonl_store.source.load_source_snapshot("cli:copied")
    await operator.close()
    expected = "invalid_input"
    if attack == "unknown":
        snapshot.events[0][SENTINEL] = SENTINEL
        expected = "unsupported"
    elif attack.startswith("role"):
        snapshot.events[0]["role"] = {"role": "system", "role-list": [], "role-dict": {},
                                     "role-bool": True, "role-int": 1}[attack]
        expected = "unsupported"
    elif attack == "tool-link":
        snapshot.events[2]["_execution_id"] = "bad"
        expected = "tool_pairing"
    elif attack == "event-id":
        snapshot.events[1]["_event_id"] = snapshot.events[0]["_event_id"]
    elif attack == "bytes":
        snapshot.events[0]["content"] = "x" * (MAX_BYTES + 1)
        expected = "bound_exceeded"
    else:
        nested = []
        for _ in range(MAX_DEPTH + 1):
            nested = [nested]
        snapshot.events[0]["content"] = nested
        expected = "bound_exceeded"
    before = deepcopy(snapshot)
    evidence = canonical_evidence(snapshot)
    assert evidence.result == expected and snapshot == before
    assert SENTINEL not in repr(evidence)


@pytest.mark.parametrize("value", [{"role": []}, {"role": {}}, {"role": True}, {"role": 1},
                                    {"role": None}, {"role": "user", "content": "\ud800"},
                                    {"role": "user", "content": "x", "_runtime_context": {"unknown": SENTINEL}}])
def test_dynamic_wrong_shapes_cannot_leak_raw_errors(value):
    evidence = legacy_evidence([value], 0)
    assert evidence.result in {"unsupported", "invalid_input"}
    assert SENTINEL not in repr(evidence)


def test_provider_tool_ids_may_repeat_after_a_closed_batch():
    # Canonical execution linkage, not globally unique provider IDs, is authority.
    history = tool_history() * 2
    evidence = legacy_evidence(history, 0)
    assert evidence.result == "equal" and compare(evidence, evidence)["left_count"] == 8


def original_snapshot(rows, *, visibility="root", boundary=0):
    events = deepcopy(rows)
    for index, event in enumerate(events):
        event.update(_event_id=f"event-{index}", _run_id="synthetic-run", _stream_id="root")
        if visibility == "child":
            event.update(_parent_stream_id="root", _stream_id="child")
        elif visibility == "command":
            event["_command"] = True
        if "tool_calls" in event:
            event["_execution_ids"] = [f"effect-{call['id']}" for call in event["tool_calls"]]
        if event.get("role") == "tool":
            event["_execution_id"] = f"effect-{event['tool_call_id']}"
    return SourceSnapshot(SourceState("copied", "test:original", "incarnation", 1, 0,
                                     None, "fingerprint"), {"last_archived": boundary}, events, None)


@pytest.mark.parametrize("visibility", ["root", "child", "command"])
@pytest.mark.parametrize("content", [{"unexpected": "shape"}, [{"type": "text", "text": SENTINEL}]])
def test_original_canonical_rich_content_rejects_before_coercion_or_visibility(visibility, content):
    rows = [{"role": "user", "content": content}]
    snapshot = original_snapshot(rows, visibility=visibility)
    before = deepcopy(snapshot)
    canonical = canonical_evidence(snapshot)
    assert canonical.result == legacy_evidence(rows, 0).result == "unsupported"
    literal = legacy_evidence([{"role": "user", "content": json.dumps(content)}], 0)
    assert compare(canonical, literal)["result"] == "unsupported"
    assert operator_compare(canonical, literal).left_digest is None
    assert snapshot == before and SENTINEL not in repr(canonical)


@pytest.mark.parametrize("visibility", ["child", "command"])
@pytest.mark.parametrize("attack", ["calls-shape", "function-shape", "function-extra", "arguments-json",
                                    "arguments-array", "wrong-role", "result-name", "extra-result-field"])
def test_hidden_original_tool_grammar_is_not_discarded_before_validation(visibility, attack):
    rows = tool_history()
    if attack == "calls-shape":
        # No private execution links are invented for a malformed calls value.
        rows = [{"role": "assistant", "content": None, "tool_calls": {}}]
    elif attack == "function-shape":
        rows[1]["tool_calls"][0]["function"] = []
    elif attack == "function-extra":
        rows[1]["tool_calls"][0]["function"][SENTINEL] = SENTINEL
    elif attack == "arguments-json":
        rows[1]["tool_calls"][0]["function"]["arguments"] = SENTINEL
    elif attack == "arguments-array":
        rows[1]["tool_calls"][0]["function"]["arguments"] = "[]"
    elif attack == "wrong-role":
        rows[1]["role"] = "user"
    elif attack == "result-name":
        rows[2]["name"] = []
    else:
        rows[2]["tool_calls"] = []
    snapshot = original_snapshot(rows, visibility=visibility)
    before = deepcopy(snapshot)
    evidence = canonical_evidence(snapshot)
    assert evidence.result in {"tool_pairing", "invalid_input", "unsupported"}
    assert snapshot == before and SENTINEL not in repr(evidence)


def closed_tool_batch(count):
    return [{"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call-{index}", "type": "function", "function": {"name": "fake", "arguments": "{}"}}
        for index in range(count)]}] + [
        {"role": "tool", "tool_call_id": f"call-{index}", "name": "fake", "content": "fake evidence"}
        for index in range(count)]


@pytest.mark.parametrize("visibility", ["root", "child", "command"])
@pytest.mark.parametrize("count", [MAX_TOOL_CALLS, MAX_TOOL_CALLS + 1])
def test_total_original_tool_bound_includes_correctly_linked_hidden_batches(visibility, count):
    snapshot = original_snapshot(closed_tool_batch(count), visibility=visibility)
    before = deepcopy(snapshot)
    evidence = canonical_evidence(snapshot)
    assert evidence.result == ("equal" if count == MAX_TOOL_CALLS else "bound_exceeded")
    if count == MAX_TOOL_CALLS:
        assert evidence.count == (count + 1 if visibility == "root" else 0)
    assert snapshot == before


@pytest.mark.parametrize("visibility", ["child", "command"])
def test_valid_hidden_text_and_closed_tools_preserve_root_only_archive_semantics(visibility):
    root = original_snapshot([{"role": "user", "content": "retained root"}])
    hidden = original_snapshot(tool_history(), visibility=visibility)
    for event in hidden.events:
        event["_event_id"] = "hidden-" + event["_event_id"]
    snapshot = replace(root, events=root.events + hidden.events, header={"last_archived": 5})
    before = deepcopy(snapshot)
    expected = legacy_evidence([{"role": "user", "content": "retained root"}], 1)
    assert compare(canonical_evidence(snapshot), expected)["result"] == "equal"
    assert snapshot == before


@pytest.mark.parametrize("case,expected", [
    ("equal", "equal"), ("left", "unsupported"), ("right", "invalid_input"),
    ("pairing", "tool_pairing"), ("wire", "public_history"),
    ("boundary", "archive_boundary"),
])
def test_owned_comparison_retains_result_priority_and_privacy(case, expected):
    left = legacy_evidence([{"role": "user", "content": SENTINEL}], 0)
    right = left
    if case == "left":
        left = replace(left, result="unsupported")
        right = replace(right, result="invalid_input")
    elif case == "right":
        right = replace(right, result="invalid_input")
    elif case == "pairing":
        right = replace(right, _pairing=b"different", _wire=b"different", _boundary=1)
    elif case == "wire":
        right = replace(right, _wire=b"different", _boundary=1)
    elif case == "boundary":
        right = replace(right, _boundary=1)

    report = left.compare(right)

    assert report == compare(left=left, right=right) == {
        "component": "shadow", "result": expected, "left_count": 1, "right_count": 1,
    }
    assert SENTINEL not in repr(report)
    assert "digest" not in repr(report)


@pytest.mark.parametrize("result", ["equal", "invalid_input"])
def test_owned_operator_digest_retains_exact_private_plane(result):
    rows = [{"role": "user", "content": SENTINEL}]
    evidence = replace(legacy_evidence(rows, 0), result=result)
    wire = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    expected = hashlib.sha256(wire + b"\x000").hexdigest() if result == "equal" else None

    assert evidence.operator_digest() == expected
    comparison = operator_compare(left=evidence, right=evidence)
    assert comparison.left_digest == comparison.right_digest == expected
    assert SENTINEL not in repr(evidence) + repr(comparison)
    if expected is not None:
        assert expected not in repr(comparison)


def test_module_evidence_api_does_not_dispatch_through_subclass_overrides():
    class OverrideEvidence(PublicEvidence):
        def compare(self, right):
            raise AssertionError("module comparison must retain its base implementation")

        def operator_digest(self):
            raise AssertionError("module digest must retain its base implementation")

    evidence = OverrideEvidence("equal", 1, b"synthetic-private-wire")
    assert compare(left=evidence, right=evidence) == {
        "component": "shadow", "result": "equal", "left_count": 1, "right_count": 1,
    }
    expected = hashlib.sha256(b"synthetic-private-wire\x000").hexdigest()
    comparison = operator_compare(left=evidence, right=evidence)
    assert comparison.left_digest == comparison.right_digest == expected
    assert "synthetic-private-wire" not in repr(evidence) + repr(comparison)
