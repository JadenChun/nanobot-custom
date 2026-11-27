"""Child-history tool rows must read like the parent activity stream."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from kohaku.agent.tools.execution import KnownToolResult
from kohaku.session.execution_owner import acquire_execution
from kohaku.session.manager import SessionManager
from kohaku.session.turn_writer import SourceTurnWriter
from tests.webui.test_child_thread_route import (
    _PARENT_KEY,
    _auth_headers,
    _make_channel,
    _make_session_manager,
    _seed_canonical_child,
    _seed_child_session,
    _thread_url,
    http_get,
)

_ASSIGNMENT = "Read-only investigation of the fixture.\nSecond line with rules."
_TRACE = re.compile(r"^(?P<name>[a-zA-Z0-9_.-]+)\((?P<args>.*)\)$")
# (tool name, arguments, output). Outputs are deliberately unlike any trace line.
_CLAUDE_SDK_CALLS = (
    ("grep", {"pattern": "refs/heads/main", "path": "/w/.git"}, ".git/packed-refs:5:847dc33e"),
    ("read_file", {"path": "/w/AGENTS.md"}, "1| # AGENTS\n2| Follow these rules"),
    ("find_files", {"pattern": "AGENTS*.md", "path": "/w"}, "AGENTS.md\nAGENTS.override.md"),
)
_NATIVE_CALLS = (
    ("exec", {"command": "rg token src"}, "src/a.ts:1: token\nExit code: 0"),
    ("read_file", {"path": "src/a.ts"}, "1| export const token = 1"),
    ("web_search", {"query": "kohaku"}, "1. Kohaku result"),
)


@pytest.fixture(autouse=True)
def _isolate_runtime_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)


def _update_child_record(
    session_manager: SessionManager, child_key: str, changes: Mapping[str, Any]
) -> None:
    metadata = session_manager.read_session_metadata(child_key)
    assert metadata is not None
    payload = {**metadata["metadata"]["delegation_child"], **changes}
    session_manager.upsert_session_metadata(
        child_key, {"delegation_child": payload}, create_if_absent=False
    )


async def _seed_tool_child(
    session_manager: SessionManager,
    child_key: str,
    calls: list[tuple[str, str, dict[str, Any], str]],
    *,
    batch_size: int = 4,
    status: str = "ok",
    record: Mapping[str, Any] | None = None,
    first: Mapping[str, Any] | None = None,
    last: Mapping[str, Any] | None = None,
) -> None:
    await _seed_canonical_child(
        session_manager,
        child_key=child_key,
        parent_key=_PARENT_KEY,
        messages=[{"role": "user", "content": _ASSIGNMENT, **(first or {})}],
    )
    backend = session_manager.source_backend(child_key)
    assert backend is not None
    async with acquire_execution(backend, child_key, run_id="tool-run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="tool-run")
        for start in range(0, len(calls), batch_size):
            batch = calls[start : start + batch_size]
            writer.begin_tool_batch({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                    for call_id, name, arguments, _output in batch
                ],
            })
            for index, (call_id, name, _arguments, output) in enumerate(batch):
                await writer.record_tool_result_event(KnownToolResult(
                    index, call_id, name, output, output,
                    {"name": name, "status": status}, True,
                ))
        writer.append_message({"role": "assistant", "content": "Report", **(last or {})})
    session_manager.invalidate(child_key)
    if record:
        _update_child_record(session_manager, child_key, record)


async def _fetch(
    session_manager: SessionManager, tmp_path: Path, child_key: str, *queries: str
) -> list[dict[str, Any]]:
    """Fetch pages; ``{before}`` in a query takes the previous page's cursor."""
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    bodies: list[dict[str, Any]] = []
    try:
        for query in queries:
            if bodies:
                query = query.replace("{before}", bodies[-1]["page"]["before_cursor"] or "")
            response = await http_get(
                f"{_thread_url(channel, child_key)}?{query}", headers=_auth_headers(channel)
            )
            assert response.status_code == 200, response.text
            bodies.append(response.json())
    finally:
        await channel.stop()
        await server_task
    return bodies


def _expand(
    shapes: tuple[tuple[str, dict[str, Any], str], ...], count: int, id_format: str
) -> list[tuple[str, str, dict[str, Any], str]]:
    calls: list[tuple[str, str, dict[str, Any], str]] = []
    for index in range(count):
        name, arguments, output = shapes[index % len(shapes)]
        calls.append((id_format.format(index), name, {**arguments, "n": index}, output))
    return calls


def _tool_rows(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row for row in body["messages"] if row["role"] == "tool"]


def _assert_rows_trace_calls(
    rows: list[dict[str, Any]], calls: list[tuple[str, str, dict[str, Any], str]]
) -> None:
    by_id = {call[0]: call for call in calls}
    for row in rows:
        [event] = row["toolEvents"]
        _call_id, name, arguments, output = by_id[event["call_id"]]
        assert row["kind"] == "trace"
        [line] = row["traces"]
        match = _TRACE.match(line)
        assert match is not None, line
        assert match["name"] == name
        assert json.loads(match["args"]) == arguments
        # Output is result detail only, never the line a client can show.
        assert row["content"] == line
        assert event["result"] == output
        assert event["arguments"] == arguments


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("harness", "shapes", "id_format"),
    [
        ("claude_sdk", _CLAUDE_SDK_CALLS, "toolu_{0:04d}|fc_{0:04d}"),
        ("native", _NATIVE_CALLS, "call_{0}"),
    ],
)
async def test_child_history_tool_rows_carry_call_traces(
    tmp_path: Path,
    harness: str,
    shapes: tuple[tuple[str, dict[str, Any], str], ...],
    id_format: str,
) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = f"delegated:traces-{harness}"
    calls = _expand(shapes, 6, id_format)
    await _seed_tool_child(session_manager, child_key, calls, record={"harness": harness})

    [body] = await _fetch(session_manager, tmp_path, child_key, "limit=100")

    rows = _tool_rows(body)
    assert len(rows) == len(calls)
    _assert_rows_trace_calls(rows, calls)


@pytest.mark.asyncio
async def test_reference_shaped_child_replays_every_page_with_traces_and_assignment(
    tmp_path: Path,
) -> None:
    """124 tool calls in a claude_sdk child, read with the overlay's page size."""
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:traces-reference"
    calls = _expand(_CLAUDE_SDK_CALLS, 124, "toolu_{0:04d}|fc_{0:04d}")
    await _seed_tool_child(
        session_manager,
        child_key,
        calls,
        record={
            "harness": "claude_sdk",
            "assignment": _ASSIGNMENT,
            "created_at": 1_795_770_697.0,
            "updated_at": 1_795_771_072.0,
        },
    )

    latest, older = await _fetch(
        session_manager, tmp_path, child_key, "limit=120", "limit=120&before={before}"
    )

    assert latest["page"]["has_more_before"] is True
    assert all(row["role"] != "user" for row in latest["messages"])
    for body in (latest, older):
        assert body["assignment"] == _ASSIGNMENT
        assert body["duration_ms"] == 375_000
        _assert_rows_trace_calls(_tool_rows(body), calls)
    assert len(_tool_rows(latest)) + len(_tool_rows(older)) == 124


@pytest.mark.asyncio
async def test_tool_row_matches_a_call_that_is_on_an_earlier_page(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:traces-split-page"
    calls = _expand(_CLAUDE_SDK_CALLS, 3, "call_{0}")
    await _seed_tool_child(session_manager, child_key, calls)

    # Newest two events: the last tool result and the report. The call is older.
    [body] = await _fetch(session_manager, tmp_path, child_key, "limit=2")

    assert [row["role"] for row in body["messages"]] == ["tool", "assistant"]
    _assert_rows_trace_calls(_tool_rows(body), calls)


@pytest.mark.asyncio
async def test_failed_and_oversized_results_stay_in_result_detail(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:traces-large"
    output = "SECRET-OUTPUT " * 2_000
    calls = [("call_0", "read_file", {"path": "big.txt", "note": "x" * 20_000}, output)]
    await _seed_tool_child(session_manager, child_key, calls, status="error")

    [body] = await _fetch(session_manager, tmp_path, child_key, "limit=10")

    [row] = _tool_rows(body)
    [event] = row["toolEvents"]
    assert event["phase"] == "error"
    assert row["traces"][0].startswith('read_file({"path": "big.txt"')
    assert "SECRET-OUTPUT" not in row["traces"][0] + row["content"]
    # Same bounding as the parent transcript projection.
    assert len(row["traces"][0]) < 5_000
    assert event["result_truncated"] is True
    assert output.startswith(event["result"])


def test_tool_row_without_a_matching_call_never_uses_output_as_its_line() -> None:
    from types import SimpleNamespace

    from kohaku.webui.ws_http import _canonical_child_rows

    rows = _canonical_child_rows(
        SimpleNamespace(),  # type: ignore[arg-type]
        [(0, {
            "role": "tool", "tool_call_id": "orphan", "name": "grep",
            "content": "src/a.ts:1: token", "_event_id": "event-1",
        })],
        scope=SimpleNamespace(),  # type: ignore[arg-type]
        allow_folder_grants=False,
        tool_calls={},
    )

    assert rows[0]["traces"] == ["grep()"]
    assert rows[0]["content"] == "grep()"
    assert rows[0]["toolEvents"][0]["result"] == "src/a.ts:1: token"


@pytest.mark.asyncio
async def test_assignment_falls_back_to_the_first_user_event(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:traces-no-assignment"
    await _seed_tool_child(session_manager, child_key, _expand(_NATIVE_CALLS, 3, "call_{0}"))

    [body] = await _fetch(session_manager, tmp_path, child_key, "limit=1")

    assert body["assignment"] == _ASSIGNMENT


@pytest.mark.asyncio
async def test_duration_spans_first_to_last_child_event_timestamp(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:traces-event-times"
    await _seed_tool_child(
        session_manager,
        child_key,
        _expand(_NATIVE_CALLS, 1, "call_{0}"),
        first={"created_at_ms": 1_000_000},
        last={"created_at_ms": 1_372_000},
        record={"created_at": 1.0, "updated_at": 9_999.0},
    )

    [body] = await _fetch(session_manager, tmp_path, child_key, "limit=1")

    assert body["duration_ms"] == 372_000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record",
    [
        {"state": "active", "created_at": 100.0, "updated_at": 200.0},
        # Delivering a background result moves updated_at past the end of the run.
        {
            "created_at": 100.0,
            "updated_at": 9_000.0,
            "background_result": {"state": "admitted"},
        },
        {"created_at": 200.0, "updated_at": 100.0},
        {},
    ],
    ids=["running", "delivered-later", "clock-skew", "no-times"],
)
async def test_duration_is_omitted_when_it_cannot_be_trusted(
    tmp_path: Path, record: dict[str, Any]
) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:traces-no-duration"
    await _seed_tool_child(
        session_manager, child_key, _expand(_NATIVE_CALLS, 1, "call_{0}"), record=record
    )

    [body] = await _fetch(session_manager, tmp_path, child_key, "limit=10")

    assert "duration_ms" not in body


@pytest.mark.asyncio
async def test_summary_history_includes_the_assignment(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    child_key = "delegated:traces-summary"
    _seed_child_session(
        session_manager,
        child_key=child_key,
        transcript=[
            {"role": "user", "content": "the original assignment"},
            {"role": "assistant", "content": "interim"},
            {"role": "assistant", "content": "result"},
        ],
    )

    [body] = await _fetch(session_manager, tmp_path, child_key, "limit=10")

    assert body["history_kind"] == "summary"
    assert body["assignment"] == "the original assignment"
