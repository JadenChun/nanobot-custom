"""Tests for Phase 1 Tasks 1.1 & 1.2: host-tool rule precedence, generalized auto-review, and PendingTurnApproval arguments."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from kohaku.agent.auto_review import (
    check_auto_review_eligibility,
    review_sandbox_escalation,
    sanitize_tool_arguments,
)
from kohaku.agent.policy import RiskyActionPolicy
from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.registry import ToolDispatchGuard, ToolRegistry
from kohaku.bus.events import InboundMessage
from kohaku.bus.outbound_events import ExternalApprovalEvent
from kohaku.config.schema import ToolPermissionsConfig
from kohaku.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest
from kohaku.utils.llm_runtime import LLMRuntime
from tests.migration.test_local_runtime_activation import (
    _config,
    _running_composition,
    _ScriptedProvider,
)


class _SpyProvider(LLMProvider):
    def __init__(self, response: LLMResponse | None = None) -> None:
        super().__init__(provider_name="test")
        self._response = response or LLMResponse(
            content=json.dumps({"decision": "approve", "reason": "ok"}),
        )
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "model": model})
        return self._response

    def get_default_model(self) -> str:
        return "test-model"


def _make_runtime(provider: LLMProvider) -> LLMRuntime:
    return LLMRuntime(
        provider=provider,
        model="test-model",
        generation=GenerationSettings(),
        context_window_tokens=128_000,
    )


class _DummyHostTool(Tool):
    launches_host_process = True

    def __init__(self, tool_name: str = "agent_browser") -> None:
        self._name = tool_name
        self.executed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "dummy host tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"args": {"type": "array"}}}

    async def execute(self, **kwargs: Any) -> str:
        self.executed = True
        return "ok"


# ---------------------------------------------------------------------------
# Task 1.1: RiskyActionPolicy explicit allow/deny precedence in Sandbox
# ---------------------------------------------------------------------------


def test_explicit_allow_overrides_sandbox_crossing_for_host_tool(tmp_path: Path) -> None:
    policy = RiskyActionPolicy(
        workspace=tmp_path,
        permissions=ToolPermissionsConfig(
            profile="balanced",
            rules={"agent_browser": "allow"},
        ),
        sandbox_active=lambda: True,
    )
    call = ToolCallRequest(id="c1", name="agent_browser", arguments={"args": ["click", "@e1"]})
    assert policy._risky_reason(call) is None


def test_sandbox_crossing_still_asks_without_explicit_allow(tmp_path: Path) -> None:
    policy = RiskyActionPolicy(
        workspace=tmp_path,
        permissions=ToolPermissionsConfig(profile="autonomous"),
        sandbox_active=lambda: True,
    )
    call = ToolCallRequest(id="c1", name="agent_browser", arguments={"args": ["click", "@e1"]})
    reason = policy._risky_reason(call)
    assert isinstance(reason, str)
    assert "Sandbox" in reason


def test_explicit_deny_still_refuses_in_sandbox(tmp_path: Path) -> None:
    policy = RiskyActionPolicy(
        workspace=tmp_path,
        permissions=ToolPermissionsConfig(
            profile="autonomous",
            rules={"agent_browser": "deny"},
        ),
        sandbox_active=lambda: True,
    )
    call = ToolCallRequest(id="c1", name="agent_browser", arguments={"args": ["open", "https://example.com"]})
    decision = policy._risky_reason(call)
    assert not isinstance(decision, str)
    assert decision is not None
    assert decision.action == "respond"
    assert "denied" in (decision.response or "").lower()


def test_full_access_unchanged_without_sandbox_crossing(tmp_path: Path) -> None:
    policy = RiskyActionPolicy(
        workspace=tmp_path,
        permissions=ToolPermissionsConfig(profile="balanced"),
        sandbox_active=lambda: False,
    )
    click_call = ToolCallRequest(id="c1", name="agent_browser", arguments={"args": ["click", "@e1"]})
    open_call = ToolCallRequest(id="c2", name="agent_browser", arguments={"args": ["open", "https://example.com"]})
    assert policy._risky_reason(click_call) == "perform a state-changing browser action"
    assert policy._risky_reason(open_call) is None


# ---------------------------------------------------------------------------
# Task 1.2: Generalized auto_review.py (Stage 1 hard-stops & Stage 2 LLM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("call", [
    ToolCallRequest(id="b1", name="agent_browser", arguments={"args": ["click", "@e12"]}),
    ToolCallRequest(id="b2", name="agent_browser", arguments={"args": ["screenshot", "--output", "media/page.png"]}),
    ToolCallRequest(id="d1", name="desktop_use", arguments={"action": "click", "x": 100, "y": 200}),
    ToolCallRequest(id="w1", name="write_file", arguments={"path": "src/main.py", "content": "print(1)\n"}),
    ToolCallRequest(id="e1", name="edit_file", arguments={"path": "src/main.py", "old_text": "1", "new_text": "2"}),
])
async def test_safe_non_exec_tools_reach_stage2_llm_review(
    tmp_path: Path,
    call: ToolCallRequest,
) -> None:
    provider = _SpyProvider(
        LLMResponse(content='{"decision": "approve", "reason": "Matches user request"}'),
    )
    runtime = _make_runtime(provider)

    result = await review_sandbox_escalation(
        runtime=runtime,
        permissions=ToolPermissionsConfig(sandbox_auto_review=True),
        workspace=tmp_path,
        call=call,
        summary=f"run {call.name}",
        user_prompt="Please perform this action in the workspace",
    )

    assert result.decision == "approve"
    assert result.stage == "llm"
    assert len(provider.calls) == 1
    user_payload = json.loads(provider.calls[0]["messages"][1]["content"])
    assert user_payload["tool_call"]["name"] == call.name
    assert "arguments" in user_payload["tool_call"]


@pytest.mark.parametrize("call", [
    ToolCallRequest(id="u1", name="agent_browser", arguments={"args": ["upload", "@1", "/etc/passwd"]}),
    ToolCallRequest(id="u2", name="agent_browser", arguments={"args": ["eval", "document.cookie"]}),
    ToolCallRequest(id="u3", name="agent_browser", arguments={"args": ["evaluate", "1+1"]}),
    ToolCallRequest(id="u4", name="agent_browser", arguments={"args": ["open", "file:///etc/passwd"]}),
    ToolCallRequest(id="u5", name="agent_browser", arguments={"args": ["open", "javascript:alert(1)"]}),
    ToolCallRequest(id="u6", name="agent_browser", arguments={"args": ["open", "data:text/html,<h1>x</h1>"]}),
    ToolCallRequest(id="u7", name="agent_browser", arguments={"args": ["open", "view-source:file:///etc/hosts"]}),
    ToolCallRequest(id="u8", name="agent_browser", arguments={"args": ["screenshot", "--output", "/tmp/leak.png"]}),
    ToolCallRequest(id="u9", name="agent_browser", arguments={"args": ["screenshot", "--screenshot-dir=/tmp/out"]}),
    ToolCallRequest(id="d1", name="agent_device", arguments={"args": ["eval", "secret()"]}),
    ToolCallRequest(id="d2", name="agent_device", arguments={"args": ["screenshot", "-o", "/tmp/dev.png"]}),
    ToolCallRequest(id="k1", name="exec", arguments={"command": "cat ~/.kohaku/config.json"}),
    ToolCallRequest(id="k2", name="exec", arguments={"command": "jq . ~/.nanobot/config.json"}),
    ToolCallRequest(id="k3", name="write_file", arguments={"path": "~/.kohaku/config.json", "content": "{}"}),
    ToolCallRequest(id="k4", name="edit_file", arguments={"path": "config.json", "old_text": "a", "new_text": "b"}),
    ToolCallRequest(id="k5", name="desktop_use", arguments={"action": "type", "text": "~/.kohaku/config.json"}),
    ToolCallRequest(id="c1", name="write_file", arguments={"path": "~/.ssh/authorized_keys", "content": "ssh-rsa ..."}),
    ToolCallRequest(id="c2", name="edit_file", arguments={"path": "/etc/sudoers", "old_text": "a", "new_text": "b"}),
    ToolCallRequest(id="c3", name="write_file", arguments={"path": "/etc/passwd", "content": "root::0:0::/:/bin/sh"}),
])
async def test_stage1_hard_stops_force_ask_user_without_llm(
    tmp_path: Path,
    call: ToolCallRequest,
) -> None:
    provider = _SpyProvider(
        LLMResponse(content='{"decision": "approve", "reason": "should not run"}'),
    )
    runtime = _make_runtime(provider)

    result = await review_sandbox_escalation(
        runtime=runtime,
        permissions=ToolPermissionsConfig(sandbox_auto_review=True),
        workspace=tmp_path,
        call=call,
        summary="escalation",
        user_prompt="Do whatever is needed",
    )

    assert result.decision == "ask_user"
    assert result.stage == "filter"
    assert provider.calls == []


def test_stage1_explicit_ask_and_deny_rules_for_non_exec_tools(tmp_path: Path) -> None:
    perms = ToolPermissionsConfig(
        sandbox_auto_review=True,
        rules={
            "write_file": {"src/*.py": "ask"},
            "edit_file": {"secret.txt": "deny"},
            "agent_browser": "ask",
        },
    )
    assert check_auto_review_eligibility(
        ToolCallRequest(id="1", name="write_file", arguments={"path": "src/app.py", "content": "x"}),
        permissions=perms,
        workspace=tmp_path,
    ) is not None
    assert check_auto_review_eligibility(
        ToolCallRequest(id="2", name="edit_file", arguments={"path": "secret.txt", "old_text": "a", "new_text": "b"}),
        permissions=perms,
        workspace=tmp_path,
    ) is not None
    assert check_auto_review_eligibility(
        ToolCallRequest(id="3", name="agent_browser", arguments={"args": ["click", "@e1"]}),
        permissions=perms,
        workspace=tmp_path,
    ) is not None


def test_sanitize_tool_arguments_redacts_secrets_and_bounds_size() -> None:
    sanitized = sanitize_tool_arguments({
        "api_key": "sk-secret-123",
        "token": "tok-456",
        "text": "x" * 1000,
    })
    assert sanitized["api_key"] == "<redacted>"
    assert sanitized["token"] == "<redacted>"
    assert isinstance(sanitized["text"], str)
    assert len(sanitized["text"]) < 600


# ---------------------------------------------------------------------------
# Task 1.2: PendingTurnApproval carries real arguments for non-exec tools
# ---------------------------------------------------------------------------


async def test_pending_turn_approval_carries_non_exec_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.tools.permissions.profile = "balanced"
    config.tools.permissions.sandbox_auto_review = False

    provider = _ScriptedProvider([
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="browser-1",
                    name="agent_browser",
                    arguments={"args": ["click", "@e42"]},
                ),
            ],
        ),
        LLMResponse(content="Clicked button."),
    ])

    async with _running_composition(config, provider, monkeypatch) as (handle, _):
        from kohaku.agent.tools.agent_browser import AgentBrowserTool

        executed: list[Any] = []

        async def _fake_browser_execute(self: AgentBrowserTool, **kwargs: Any) -> str:
            executed.append(kwargs)
            return "ok"

        monkeypatch.setattr(AgentBrowserTool, "execute", _fake_browser_execute)

        chat_id = "non-exec-approval-args"
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket",
            sender_id="webui",
            chat_id=chat_id,
            content="Click the submit button",
            metadata={
                "webui": True,
                "webui_turn_id": "turn-browser-args",
                "live_turn_controls": True,
            },
        ))

        async with asyncio.timeout(5):
            while True:
                outbound = await handle.bus.consume_outbound()
                if isinstance(outbound.event, ExternalApprovalEvent) and outbound.event.approval:
                    approval = outbound.event.approval
                    break

        assert approval.tool_name == "agent_browser"
        parsed_args = json.loads(approval.arguments_json)
        assert parsed_args == {"args": ["click", "@e42"]}

        from kohaku.agent.turn import ApprovalDecision

        assert handle.agent.resolve_run_approval(
            f"websocket:{chat_id}",
            ApprovalDecision(
                approval_id=approval.approval_id,
                decision="deny",
                session_key=f"websocket:{chat_id}",
                session_incarnation=approval.session_incarnation,
                run_id=approval.run_id,
                effect_id=approval.tool_call_id,
                generation=approval.runtime_generation,
            ),
        )
        assert executed == []


# ---------------------------------------------------------------------------
# Dream ToolDispatchGuard still denies stale Dream operations
# ---------------------------------------------------------------------------


async def test_dream_tool_dispatch_guard_denies_stale_operations() -> None:
    class _StaleDreamGuard(ToolDispatchGuard):
        def error(self, name: str) -> str | None:
            return "Dream operation expired"

    registry = ToolRegistry()
    dummy = _DummyHostTool("agent_browser")
    registry.register(dummy)
    registry.set_dispatch_guard(_StaleDreamGuard())

    result = await registry.execute("agent_browser", {"args": ["open", "https://example.com"]})
    assert isinstance(result, str)
    assert "Dream operation expired" in result
    assert dummy.executed is False
