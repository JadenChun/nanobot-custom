"""Live tool progress flows from a running tool to the streaming hook."""

from __future__ import annotations

from typing import Any

import pytest

from kohaku.agent.hook import AgentHook, AgentHookContext
from kohaku.agent.progress_hook import AgentProgressHook
from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.execution import execute_tool_calls
from kohaku.agent.tools.progress import current_tool_progress_sink
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.bus.outbound_events import ProgressEvent
from kohaku.events import EventSink
from kohaku.providers.base import ToolCallRequest


class _ProgressReportingTool(Tool):
    name = "slow_media"
    description = "Report progress while it runs."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        sink = current_tool_progress_sink()
        assert sink is not None
        await sink.emit(35, 100, "Starting ComfyUI")
        await sink.emit(62, 100, "ComfyUI is generating prompt...")
        return "video ready"


def _call(call_id: str = "call-media") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name="slow_media", arguments={})


@pytest.mark.asyncio
async def test_running_tool_progress_reaches_the_published_frames() -> None:
    published: list[ProgressEvent] = []

    async def publish(event: Any) -> None:
        if isinstance(event, ProgressEvent):
            published.append(event)

    hook = AgentProgressHook(EventSink(publish), streaming=True)
    registry = ToolRegistry()
    registry.register(_ProgressReportingTool())

    await execute_tool_calls(
        registry,
        [_call()],
        concurrent=False,
        external_lookup_counts={},
        workspace_violation_counts={},
        hook=hook,
        context=AgentHookContext(iteration=0, messages=[]),
    )

    frames = [
        event
        for event in published
        for event in (event.tool_events or [])
        if event.get("phase") == "progress"
    ]
    assert [frame["message"] for frame in frames] == [
        "Starting ComfyUI",
        "ComfyUI is generating prompt...",
    ]
    assert frames[-1]["progress"] == 62
    assert frames[-1]["total"] == 100
    assert all(frame["call_id"] == "call-media" for frame in frames)
    assert all(frame["name"] == "slow_media" for frame in frames)


@pytest.mark.asyncio
async def test_tool_without_a_sink_reports_no_progress() -> None:
    calls: list[Any] = []

    class _PlainTool(Tool):
        name = "plain"
        description = "No progress reporting."
        parameters = {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs: Any) -> str:
            calls.append(current_tool_progress_sink())
            return "done"

    registry = ToolRegistry()
    registry.register(_PlainTool())

    await execute_tool_calls(
        registry,
        [ToolCallRequest(id="call-plain", name="plain", arguments={})],
        concurrent=False,
        external_lookup_counts={},
        workspace_violation_counts={},
        hook=AgentHook(),
        context=AgentHookContext(iteration=0, messages=[]),
    )

    assert calls == [None]
