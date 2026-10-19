"""Container views must not change message rendering or source-batch behavior."""

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import overload

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.message_content import content_to_text
from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.trace import sanitize_trace
from nanobot.runtime_context import RUNTIME_CONTEXT_HISTORY_META
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import SourceValidationError
from nanobot.session.turn_writer import SourceTurnWriter


class WireMapping(Mapping[object, object]):
    def __init__(self, values: dict[object, object], events: list[object], fault=None):
        self.values, self.events, self.fault = values, events, fault

    def __getitem__(self, key: object) -> object:
        self.events.append(("get", key))
        if self.fault is not None:
            raise self.fault
        return self.values[key]

    def __iter__(self) -> Iterator[object]:
        self.events.append("mapping-iterate")
        yield from self.values

    def __len__(self) -> int:
        return len(self.values)


class WireSequence(Sequence[object]):
    def __init__(self, values: list[object], events: list[object], fault=None):
        self.values, self.events, self.fault = values, events, fault

    def __iter__(self) -> Iterator[object]:
        self.events.append("sequence-iterate")
        if self.fault is not None:
            raise self.fault
        yield from self.values

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...

    def __getitem__(self, index: int | slice) -> object | Sequence[object]:
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)


class Opaque:
    def __str__(self) -> str:
        raise AssertionError("opaque values must not be stringified")

    def __repr__(self) -> str:
        raise AssertionError("opaque values must not be represented")


@pytest.mark.parametrize("nested", [False, True])
def test_content_mapping_reads_only_existing_text_fields(nested) -> None:
    events: list[object] = []
    block = WireMapping({"type": "text", "text": " visible ", 42: Opaque()}, events)
    content = WireSequence([block], events) if nested else block

    assert content_to_text(content) == "visible"
    assert events == (["sequence-iterate"] if nested else []) + [
        ("get", "type"), ("get", "text")
    ]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("error_class", [RuntimeError, KeyboardInterrupt])
def test_content_mapping_fault_retains_identity_and_read_order(nested, error_class) -> None:
    events: list[object] = []
    fault = error_class("mapping fault")
    block = WireMapping({"type": "text"}, events, fault)
    content = WireSequence([block], events) if nested else block

    with pytest.raises(error_class) as caught:
        content_to_text(content)

    assert caught.value is fault
    assert events == (["sequence-iterate"] if nested else []) + [("get", "type")]


@pytest.mark.parametrize("consumer", [content_to_text, sanitize_trace])
@pytest.mark.parametrize("error_class", [RuntimeError, KeyboardInterrupt])
def test_sequence_iteration_fault_is_not_caught_or_retried(consumer, error_class) -> None:
    events: list[object] = []
    fault = error_class("sequence fault")

    with pytest.raises(error_class) as caught:
        consumer(WireSequence([], events, fault))

    assert caught.value is fault
    assert events == ["sequence-iterate"]


@pytest.mark.parametrize("container", [list, tuple])
def test_heterogeneous_content_remains_bounded_without_raw_payloads(container) -> None:
    content = container([
        {"type": "image_url", "image_url": {"url": "private-image"}, "alt": "photo"},
        {"type": "text", "text": "A" * 64},
        b"private-binary", Opaque(), None, {None: "not textual", 42: Opaque()},
    ])

    text = content_to_text(content)
    assert text == (
        "[image content omitted] photo\n[encoded content omitted]\n[binary content omitted]"
        "\n[structured content omitted]\n[structured content omitted]"
        "\n[structured content omitted]"
    )
    assert "private" not in text and "A" * 64 not in text
    assert content_to_text(content, max_chars=12) == "[image conte..."


@pytest.mark.parametrize("raw", [None, "text", b"bytes", bytearray(b"bytes"), {"role": "user"}])
def test_trace_keeps_existing_outer_container_refusal(raw) -> None:
    assert sanitize_trace(raw) == []


def test_trace_arbitrary_keys_detachment_and_private_projection_are_unchanged() -> None:
    events: list[object] = []
    raw = WireMapping({
        "role": "assistant", 42: "non-string secret",
        "content": [
            {"type": "thinking", "text": "private reasoning"},
            {"type": "text", "text": "visible"},
            {"type": "image_url", "image_url": {"url": "private-image"}},
        ],
        "tool_calls": [{"id": "call", "function": {"name": "read_file", "arguments": "private args"}}],
        "_run_id": "private run",
    }, events)
    trace = sanitize_trace(WireSequence([raw, 7], events))

    assert trace == [{
        "role": "assistant", "content": "visible\n[image content omitted]",
        "tool_calls": [{"id": "call", "name": "read_file"}],
    }]
    assert events == ["sequence-iterate", "mapping-iterate", *(
        ("get", key) for key in raw.values
    )]
    trace[0]["tool_calls"][0]["name"] = "changed"
    assert raw.values["tool_calls"][0]["function"]["name"] == "read_file"
    assert "private" not in repr(sanitize_trace([raw]))


def test_trace_removes_only_trusted_exact_runtime_suffix() -> None:
    marker = {"version": 1, "suffix": "private runtime"}
    raw = [
        {"role": "user", "content": "visible\n\nprivate runtime", RUNTIME_CONTEXT_HISTORY_META: marker},
        {"role": "user", "content": "user wrote private runtime", RUNTIME_CONTEXT_HISTORY_META: marker},
    ]
    before = deepcopy(raw)

    assert sanitize_trace(tuple(raw)) == [
        {"role": "user", "content": "visible"},
        {"role": "user", "content": "user wrote private runtime"},
    ]
    assert raw == before


@pytest.fixture
def wire_store(tmp_path):
    backend = JsonlSessionStore(tmp_path)
    session = Session("test:wire-views")
    backend.save(session)
    backend.source.upgrade_source_session(
        session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation,
    )
    return backend


class WireCalls(list[object]):
    def __init__(self, values: list[object], events: list[object], fault=None, fault_at="iterate"):
        super().__init__(values)
        self.events, self.fault, self.fault_at = events, fault, fault_at

    def __iter__(self) -> Iterator[object]:
        self.events.append("calls-iterate")
        if self.fault is not None and self.fault_at == "iterate":
            raise self.fault
        yield from super().__iter__()

    def __deepcopy__(self, memo) -> list[object]:
        self.events.append("calls-copy")
        if self.fault is not None and self.fault_at == "copy":
            raise self.fault
        return [deepcopy(item, memo) for item in list.__iter__(self)]


def fake_append(writer, monkeypatch, events):
    def append(payload, **metadata):
        events.append(("append", payload["role"]))
        return {**payload, **metadata, "_event_id": "event-" + payload["role"]}

    monkeypatch.setattr(writer, "_append", append)


async def test_source_batch_allocates_then_appends_then_copies_wire_declaration(
    wire_store, monkeypatch
) -> None:
    events: list[object] = []
    call = {"id": "call", "function": {"name": "one"}}
    calls = WireCalls([call], events)
    async with acquire_execution(wire_store, "test:wire-views", run_id="run") as owner:
        writer = SourceTurnWriter(wire_store.source, owner, run_id="run")
        fake_append(writer, monkeypatch, events)

        intent = writer.begin_tool_batch({"role": "assistant", "tool_calls": calls})
        assert events == ["calls-iterate", ("append", "assistant"), "calls-copy"]
        assert len(intent["execution_ids"]) == 1
        call["id"], call["function"]["name"] = "mutated", "mutated"
        value = KnownToolResult(0, "call", "one", "visible", "visible", {"status": "ok"}, True)
        assert await writer.record_tool_result_event(value) == "event-tool"
        assert events == ["calls-iterate", ("append", "assistant"), "calls-copy", ("append", "tool")]
        assert not writer.failed
        with pytest.raises(SourceValidationError, match="Invalid source result binding"):
            await writer.record_tool_result_event(value)
        assert writer.failed


@pytest.mark.parametrize("fault_at", ["iterate", "copy"])
@pytest.mark.parametrize("error_class", [RuntimeError, KeyboardInterrupt])
async def test_source_wire_list_fault_identity_and_existing_failure_extent(
    wire_store, monkeypatch, fault_at, error_class
) -> None:
    events: list[object] = []
    fault = error_class("wire list fault")
    calls = WireCalls([{"id": "call", "function": {"name": "one"}}], events, fault, fault_at)
    async with acquire_execution(wire_store, "test:wire-views", run_id="run") as owner:
        writer = SourceTurnWriter(wire_store.source, owner, run_id="run")
        fake_append(writer, monkeypatch, events)
        with pytest.raises(error_class) as caught:
            writer.begin_tool_batch({"role": "assistant", "tool_calls": calls})

        assert caught.value is fault
        assert not writer.failed  # Allocation/declaration-copy faults are outside _append's catch.
        assert events == (["calls-iterate"] if fault_at == "iterate" else [
            "calls-iterate", ("append", "assistant"), "calls-copy"
        ])
        assert owner.read_snapshot().events == []


@pytest.mark.parametrize("calls", [None, [], (), ({"id": "call"},), {"id": "call"}, "call"])
async def test_source_batch_keeps_list_only_nonempty_refusal(wire_store, monkeypatch, calls) -> None:
    events: list[object] = []
    async with acquire_execution(wire_store, "test:wire-views", run_id="run") as owner:
        writer = SourceTurnWriter(wire_store.source, owner, run_id="run")
        fake_append(writer, monkeypatch, events)
        with pytest.raises(SourceValidationError, match="Invalid source tool batch"):
            writer.begin_tool_batch({"role": "assistant", "tool_calls": calls})
        assert writer.failed
        assert events == [] and owner.read_snapshot().events == []


async def test_dynamic_list_view_does_not_claim_validated_call_elements(wire_store, monkeypatch) -> None:
    events: list[object] = []
    async with acquire_execution(wire_store, "test:wire-views", run_id="run") as owner:
        writer = SourceTurnWriter(wire_store.source, owner, run_id="run")
        fake_append(writer, monkeypatch, events)
        writer.begin_tool_batch({"role": "assistant", "tool_calls": [7]})
        value = KnownToolResult(0, "call", "one", "visible", "visible", {}, True)
        with pytest.raises(AttributeError, match="has no attribute 'get'"):
            await writer.record_tool_result_event(value)
        assert writer.failed and events == [("append", "assistant")]


def test_constructed_context_run_summaries_remain_ordered_and_omit_blank_files(
    tmp_path, monkeypatch
) -> None:
    builder = ContextBuilder(tmp_path)
    folder = tmp_path / "context-repo" / "runs"
    folder.mkdir(parents=True)
    paths = [folder / name for name in ("one.md", "blank.md", "two.md")]
    for path, text in zip(paths, (" first summary ", " \n", " second summary ")):
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(builder.context_manager, "run_summary_files", lambda: paths)

    prompt = builder.build_system_prompt(include_memory=False)

    assert prompt.index("first summary") < prompt.index("second summary")
    assert "context-repo/runs/one.md" in prompt and "context-repo/runs/two.md" in prompt
    assert "blank.md" not in prompt
