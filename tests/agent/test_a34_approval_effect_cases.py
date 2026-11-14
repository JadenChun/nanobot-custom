"""A3-4 P1 requirements-only cases C1-C9 at the canonical approval boundary.

Each case keeps the authoring session's scenario, inputs, and expected outcome.
The boundary names are mapped onto the real loop entrypoint (``execute_turn``),
the real delegated-child path, and the typed ``ApprovalDecision`` control.
"""

from pathlib import Path

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.tools.desktop_use import DesktopUseTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import (
    ApprovalDecision,
    DeliveryTarget,
    RunStatus,
    TurnRequest,
)
from nanobot.providers.base import LLMResponse, ToolCallRequest
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

engine_config = engine_fixtures.engine_config
pytest_plugins = ("tests.agent.test_migration_source_engine",)

ROUTE = DeliveryTarget(channel="cli", chat_id="original")
SOURCE_KEY = "cli:source"
STATE_CHANGING_ACTION = "left_click"
READ_ONLY_ACTIONS = ["info", "screenshot", "cursor_position", "wait"]


class CountingDesktopBackend:
    """Inert backend that records the global-input effects it receives."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def info(self) -> dict[str, object]:
        self.calls.append("info")
        return {
            "platform": "test",
            "screen": {"width": 1024, "height": 768},
            "cursor": {"x": 1, "y": 2},
        }

    def screen_size(self) -> tuple[int, int]:
        self.calls.append("screen_size")
        return (1024, 768)

    def cursor_position(self) -> tuple[int, int]:
        self.calls.append("cursor_position")
        return (1, 2)

    def screenshot(self, path: str) -> None:
        self.calls.append("screenshot")
        Path(path).write_bytes(b"fake-png")

    def resize_png(self, path: str, width: int, height: int) -> None:
        self.calls.append("resize_png")

    def mouse_move(self, x: int, y: int) -> None:
        self.calls.append("mouse_move")

    def mouse_click(self, x: int, y: int, button: str, clicks: int) -> None:
        self.calls.append("mouse_click")

    def mouse_drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.calls.append("mouse_drag")

    def scroll(self, direction: str, amount: int) -> None:
        self.calls.append("scroll")

    def type_text(self, text: str, delay_ms: int, delays_ms: list[int] | None = None) -> None:
        self.calls.append("type_text")

    def key_chord(self, chord: str) -> None:
        self.calls.append("key_chord")

    def hold_key(self, chord: str, duration: float) -> None:
        self.calls.append("hold_key")


class CountingDesktopUse(DesktopUseTool):
    def __init__(self, backend: CountingDesktopBackend) -> None:
        super().__init__(
            screenshot_delay=0,
            humanize_typing=False,
            scaling_enabled=False,
            working_dir=None,
        )
        setattr(self, "_backend", backend)  # noqa: B010 - inject the inert backend
        self.executions = 0

    async def execute(self, action: str, **kwargs: object) -> object:
        self.executions += 1
        return await super().execute(action, **kwargs)


def install_desktop(loop: AgentLoop) -> tuple[CountingDesktopUse, CountingDesktopBackend]:
    backend = CountingDesktopBackend()
    tool = CountingDesktopUse(backend)
    try:
        loop.tools.replace(tool)
    except ValueError:
        loop.tools.register(tool)
    return tool, backend


def desktop_call(action: str, identifier: str = "desktop-action") -> ToolCallRequest:
    arguments: dict[str, object] = {"action": action}
    if action in {"left_click", "mouse_move"}:
        arguments["coordinate"] = [10, 20]
    if action == "wait":
        arguments["duration"] = 0.01
    return ToolCallRequest(id=identifier, name="desktop_use", arguments=arguments)


async def decision(loop: AgentLoop, *, decision_value: str = "approve",
                   session_key: str | None = None, **overrides: object):
    pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
    assert pending is not None
    fields: dict[str, object] = {
        "approval_id": pending.approval_id,
        "decision": decision_value,
    }
    fields.update(overrides)
    return await loop.execute_turn(TurnRequest(
        content="approval_decision",
        session_key=session_key,
        route=ROUTE,
        approval_decision=ApprovalDecision(**fields),
    ))


async def blocked_turn(loop: AgentLoop, content: str = "Perform the risky effect."):
    return await loop.execute_turn(TurnRequest(
        content=content, session_key=SOURCE_KEY, route=ROUTE,
    ))


async def test_c1_state_changing_desktop_action_requires_approval(engine_config):
    engine_config.tools.desktop_use.enabled = True
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[desktop_call(STATE_CHANGING_ACTION)]),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, SOURCE_KEY)
    tool, backend = install_desktop(loop)
    try:
        result = await blocked_turn(loop, "Click the button.")
        assert result.stop_reason == "approval_required"
        assert tool.executions == 0
        assert backend.calls == []
        assert len(loop.approval_admission._pending) == 1
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        assert pending is not None
        assert pending.effect_id is not None
    finally:
        await loop.aclose()


async def test_c2_later_identical_desktop_action_needs_fresh_approval(engine_config):
    engine_config.tools.desktop_use.enabled = True
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[desktop_call(STATE_CHANGING_ACTION, "blocked")]),
        LLMResponse(content=None, tool_calls=[desktop_call(STATE_CHANGING_ACTION, "first")]),
        LLMResponse(content=None, tool_calls=[desktop_call(STATE_CHANGING_ACTION, "repeat")]),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, SOURCE_KEY)
    tool, backend = install_desktop(loop)
    try:
        await blocked_turn(loop, "Click the button.")
        approved = await decision(loop)
        assert approved.stop_reason == "approval_required", approved.error
        assert tool.executions == 1
        assert backend.calls.count("mouse_click") == 1
        assert len(loop.approval_admission._pending) == 1
    finally:
        await loop.aclose()


async def test_c3_denied_desktop_action_starts_nothing(engine_config):
    engine_config.tools.desktop_use.enabled = True
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[desktop_call(STATE_CHANGING_ACTION)]),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, SOURCE_KEY)
    tool, backend = install_desktop(loop)
    try:
        await blocked_turn(loop, "Click the button.")
        denied = await decision(loop, decision_value="deny")
        assert denied.status is RunStatus.CANCELLED
        assert denied.stop_reason == "cancelled"
        assert tool.executions == 0
        assert backend.calls == []
        assert loop.approval_admission._pending == {}
    finally:
        await loop.aclose()


@pytest.mark.parametrize("action", READ_ONLY_ACTIONS)
async def test_c4_read_only_desktop_actions_need_no_approval(engine_config, action):
    engine_config.tools.desktop_use.enabled = True
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[desktop_call(action)]),
        LLMResponse(content="Done."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, SOURCE_KEY)
    tool, backend = install_desktop(loop)
    try:
        result = await blocked_turn(loop, f"Run {action}.")
        assert result.status is RunStatus.COMPLETED, (result.stop_reason, result.error)
        assert result.stop_reason != "approval_required"
        assert tool.executions == 1
        assert loop.approval_admission._pending == {}
        if action == "wait":
            # `wait` is a local timer; it has no backend method to call.
            assert backend.calls == []
        else:
            assert backend.calls.count(action) == 1
    finally:
        await loop.aclose()


@pytest.mark.parametrize("action", [
    "mouse_move", "left_click", "right_click", "middle_click", "double_click",
    "left_click_drag", "scroll", "type", "key", "hold_key", "unknown_action",
])
async def test_desktop_input_actions_are_all_gated(engine_config, action):
    policy = RiskyActionPolicy(workspace=engine_config.workspace_path)
    decision = await policy.evaluate(messages=[], tool_calls=[desktop_call(action)])
    assert decision.stop_reason == "approval_required"
    assert decision.requires_approval is True


async def test_desktop_gate_refuses_non_string_action(engine_config):
    policy = RiskyActionPolicy(workspace=engine_config.workspace_path)
    call = ToolCallRequest(id="odd", name="desktop_use", arguments={"action": 7})
    decision = await policy.evaluate(messages=[], tool_calls=[call])
    assert decision.stop_reason == "approval_required"


async def test_c5_child_overwrite_waits_for_approval_under_child_identity(engine_config):
    target = engine_config.workspace_path / "protected.txt"
    original = "".join(f"protected line {index}\n" for index in range(20))
    target.write_text(original)
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="delegate-write",
            name="delegate_task",
            arguments={
                "contract": "Replace protected.txt with the supplied content.",
                "write_scope": ["protected.txt"],
            },
        )]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="write-protected",
            name="write_file",
            arguments={"path": "protected.txt", "content": "unauthorized replacement"},
        )]),
        LLMResponse(content="The parent task is complete."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    parent_backend = upgrade(loop, SOURCE_KEY)
    try:
        result = await blocked_turn(loop, "Delegate the bounded write.")
        assert result.error is None, result.error
        assert target.read_text() == original
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        child, = tree.children_of(SOURCE_KEY)
        assert child.state == "waiting_approval"
        assert child.pending_approval_id is not None
        assert child.current_run_id is not None
        child_run = loop.run_store.load(child.current_run_id)
        assert child_run is not None and child_run.status is RunStatus.APPROVAL_REQUIRED
        child_backend = loop.sessions.source_backend(child.session_key)
        assert child_backend is not None
        child_events = child_backend.source.load_source_snapshot(child.session_key).events
        parent_events = parent_backend.source.load_source_snapshot(SOURCE_KEY).events
        assert any(row.get("tool_call_id") == "write-protected" for row in child_events)
        assert not any(row.get("tool_call_id") == "write-protected" for row in parent_events)
    finally:
        await loop.aclose()


async def test_c6_repeat_child_effect_needs_fresh_approval_and_applies_once(engine_config):
    target = engine_config.workspace_path / "protected.txt"
    original = "".join(f"protected line {index}\n" for index in range(20))
    target.write_text(original)
    repeated = ToolCallRequest(
        id="edit-protected",
        name="edit_file",
        arguments={
            "path": "protected.txt",
            "old_text": "protected line",
            "new_text": "approved line",
            "replace_all": True,
        },
    )
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="delegate-write",
            name="delegate_task",
            arguments={
                "contract": "Replace protected.txt with the supplied content.",
                "write_scope": ["protected.txt"],
            },
        )]),
        LLMResponse(content=None, tool_calls=[repeated]),
        LLMResponse(content=None, tool_calls=[repeated]),
        LLMResponse(content=None, tool_calls=[repeated]),
        LLMResponse(content="The approved replacement is complete."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, SOURCE_KEY)
    try:
        await blocked_turn(loop, "Delegate the bounded replace.")
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        child, = tree.children_of(SOURCE_KEY)
        assert child.state == "waiting_approval"
        approved = await decision(loop)
        assert approved.error is None, approved.error
        after_first = target.read_text()
        assert after_first.count("approved line") == 20
        settled = tree.child(child.child_id)
        assert settled is not None
        assert settled.state == "waiting_approval"
        assert settled.pending_approval_id is not None
        assert target.read_text() == after_first
        assert len(provider.calls) == 4
    finally:
        await loop.aclose()


async def test_c7_denied_file_overwrite_starts_no_tool_call(engine_config):
    target = engine_config.workspace_path / "existing.txt"
    before = "".join(f"original {index}\n" for index in range(20))
    target.write_text(before)
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="write-existing",
            name="write_file",
            arguments={"path": "existing.txt", "content": "denied replacement"},
        )]),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, SOURCE_KEY)
    try:
        await blocked_turn(loop, "Overwrite existing.txt.")
        denied = await decision(loop, decision_value="deny")
        assert denied.status is RunStatus.CANCELLED
        assert denied.stop_reason == "cancelled"
        assert target.read_text() == before
        assert len(provider.calls) == 1
    finally:
        await loop.aclose()


async def test_c8_stale_decision_leaves_pending_for_genuine_decision(engine_config):
    target = engine_config.workspace_path / "existing.txt"
    before = "".join(f"original {index}\n" for index in range(20))
    after = "".join(f"replacement {index}\n" for index in range(20))
    target.write_text(before)
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="blocked",
            name="write_file",
            arguments={"path": "existing.txt", "content": after},
        )]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="first",
            name="write_file",
            arguments={"path": "existing.txt", "content": after},
        )]),
        LLMResponse(content="Done."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, SOURCE_KEY)
    try:
        await blocked_turn(loop, "Overwrite existing.txt.")
        stale = await decision(loop, effect_id="0" * 64)
        assert stale.stop_reason == "approval_stale"
        assert target.read_text() == before
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        assert pending is not None
        genuine = await decision(loop)
        assert genuine.error is None, genuine.error
        assert target.read_text() == after
        assert target.read_text().count("replacement") == 20
    finally:
        await loop.aclose()


async def test_c9_foreign_session_alias_cannot_consume_pending(engine_config):
    target = engine_config.workspace_path / "existing.txt"
    before = "".join(f"original {index}\n" for index in range(20))
    after = "".join(f"replacement {index}\n" for index in range(20))
    target.write_text(before)
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="blocked",
            name="write_file",
            arguments={"path": "existing.txt", "content": after},
        )]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="first",
            name="write_file",
            arguments={"path": "existing.txt", "content": after},
        )]),
        LLMResponse(content="Done."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, SOURCE_KEY)
    try:
        await blocked_turn(loop, "Overwrite existing.txt.")
        foreign = await decision(loop, session_key="cli:other")
        assert foreign.stop_reason == "approval_stale"
        assert target.read_text() == before
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        assert pending is not None
        genuine = await decision(loop)
        assert genuine.error is None, genuine.error
        assert target.read_text() == after
        assert target.read_text().count("replacement") == 20
    finally:
        await loop.aclose()
