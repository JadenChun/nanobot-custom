"""Qualify the optional SDK boundary with a scripted Claude client."""

import json
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from kohaku.harness import claude_sdk
from kohaku.harness.backend import (
    ExternalReasoning,
    ExternalSessionStarted,
    ExternalText,
    ExternalToolPolicy,
    ExternalToolResult,
    ExternalTurnRequest,
    HarnessCapabilityError,
    SessionHarnessBinding,
)
from kohaku.harness.claude_sdk import ClaudeSDKBackend


@dataclass
class Bridge:
    base_url: str = "http://127.0.0.1:58463"
    auth_token: str = "fake-bridge-token"
    closed: bool = False

    async def close(self):
        self.closed = True


class FakeSDKClient:
    def __init__(self, options, *, transport):
        self.options = options
        self.transport = transport
        sdk.clients.append(self)

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def query(self, prompt):
        self.prompt = prompt

    async def receive_response(self):
        async for message in sdk.script(self.options):
            yield message


@pytest.fixture
def sdk_boundary(monkeypatch, tmp_path):
    sdk.clients = []
    sdk.script = None
    real_which = claude_sdk.shutil.which
    monkeypatch.setattr(claude_sdk, "import_module", lambda name: sdk)
    monkeypatch.setattr(claude_sdk, "managed_bwrap_available", lambda: True)
    monkeypatch.setattr(
        claude_sdk.shutil,
        "which",
        lambda name, *args, **kwargs: "/usr/bin/socat" if name == "socat" else real_which(name, *args, **kwargs),
    )
    monkeypatch.setattr(claude_sdk.sys, "platform", "linux")
    bridge = Bridge()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv("UPSTREAM_PROVIDER_KEY", "ambient-other-key")

    async def bridge_factory(runtime, source):
        return bridge

    return ClaudeSDKBackend(bridge_factory, state_dir=tmp_path / "state"), bridge


def request_for(tmp_path, policy=ExternalToolPolicy()):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return ExternalTurnRequest(
        session_key="test:claude", run_id="run",
        binding=SessionHarnessBinding("external", "claude_sdk", None, None, 1),
        workspace=workspace, prompt="Continue.", runtime=SimpleNamespace(model="test-model"),
        hook=None, writer=None, is_authorized=lambda: True, policy=policy,
    )


async def test_missing_sdk_fails_only_at_launch(tmp_path, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(claude_sdk, "import_module", missing)

    async def should_not_start(*args):
        pytest.fail("The model bridge should not start without the SDK")

    backend = ClaudeSDKBackend(should_not_start, state_dir=tmp_path / "state")
    with pytest.raises(HarnessCapabilityError, match="Claude Agent SDK"):
        await backend.start(request_for(tmp_path), resume=None, fork_session=False)


@pytest.mark.parametrize("dependency", ["bwrap", "socat"])
async def test_linux_sandbox_refuses_to_start_without_dependencies(
    tmp_path, monkeypatch, sdk_boundary, dependency,
):
    backend, _ = sdk_boundary
    if dependency == "bwrap":
        monkeypatch.setattr(claude_sdk, "managed_bwrap_available", lambda: False)
    else:
        real_which = claude_sdk.shutil.which
        monkeypatch.setattr(
            claude_sdk.shutil,
            "which",
            lambda name, *args, **kwargs: None if name == "socat" else real_which(name, *args, **kwargs),
        )
    with pytest.raises(HarnessCapabilityError, match="bubblewrap and socat"):
        await backend.start(request_for(tmp_path), resume=None, fork_session=False)
    assert not sdk.clients


async def check_tool(options, name, arguments, call_id="tool-1"):
    [matcher] = options.hooks["PreToolUse"]
    [callback] = matcher.hooks
    result = await callback(
        {"tool_name": name, "tool_input": arguments, "tool_use_id": call_id}, call_id, {},
    )
    return result["hookSpecificOutput"]


async def test_narrow_scope_keeps_bash_sandboxed_and_denies_file_path_escapes(
    tmp_path, sdk_boundary,
):
    backend, bridge = sdk_boundary
    policy = ExternalToolPolicy(mode="full_access", write_scope=(tmp_path / "workspace" / "src",))
    request = request_for(tmp_path, policy)
    src = request.workspace / "src"
    src.mkdir()
    private = tmp_path / ".ssh"
    private.mkdir()
    (src / "linked").symlink_to(private, target_is_directory=True)
    process = await backend.start(request, resume="prior-id", fork_session=True)

    async def script(options):
        assert options.resume == "prior-id" and options.fork_session
        assert options.setting_sources == []
        assert options.strict_mcp_config and options.include_partial_messages
        assert options.hooks["PreToolUse"][0].on_failure == "block"
        env = options.env
        assert env["ANTHROPIC_BASE_URL"] == bridge.base_url
        assert env["ANTHROPIC_API_KEY"] == env["ANTHROPIC_AUTH_TOKEN"] == bridge.auth_token
        assert env["ANTHROPIC_MODEL"] == env["ANTHROPIC_SMALL_FAST_MODEL"] == "test-model"
        assert env["DISABLE_TELEMETRY"] == env["DISABLE_ERROR_REPORTING"] == "1"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert not {"OPENAI_API_KEY", "UPSTREAM_PROVIDER_KEY"} & env.keys()
        sandbox = json.loads(options.settings)["sandbox"]
        assert sandbox["enabled"] and sandbox["autoAllowBashIfSandboxed"]
        assert not sandbox["allowUnsandboxedCommands"]

        async def decision(name, **arguments):
            return (await check_tool(options, name, arguments))["permissionDecision"]

        assert await decision("Read", file_path=str(tmp_path / "outside.txt")) == "deny"
        assert await decision("Read", file_path=str(private / "private.txt")) == "deny"
        assert await decision("Read", file_path="src/linked/private.txt") == "deny"
        assert await decision("Write", file_path="outside-scope.txt") == "deny"
        assert await decision("Edit", file_path="src/linked/private.txt") == "deny"
        assert await decision("NotebookEdit", notebook_path="outside.ipynb") == "deny"
        assert await decision("Read", file_path="outside-scope.txt") == "allow"
        assert await decision("Write", file_path="src/inside.txt") == "allow"
        assert await decision("Bash", command="pytest -q") == "allow"
        assert await decision("Bash", command="pytest -q", dangerouslyDisableSandbox=True) == "deny"
        yield sdk.ResultMessage(session_id="fork-id", is_error=False, terminal_reason=None, result=None)

    async def approve(*args):
        pytest.fail("Safe and prohibited tools do not require approval")

    sdk.script = script
    try:
        events = [event async for event in process.run(request, approve)]
        assert events[0] == ExternalSessionStarted("fork-id")
    finally:
        await process.close()
    assert bridge.closed


async def test_sandbox_escalation_and_live_saved_exclusion(tmp_path, sdk_boundary):
    backend, _ = sdk_boundary
    request = request_for(tmp_path)
    process = await backend.start(request, resume=None, fork_session=False)
    seen = []

    async def script(options):
        sandbox = json.loads(options.settings)["sandbox"]
        assert sandbox["enabled"] and sandbox["autoAllowBashIfSandboxed"]
        context = SimpleNamespace(tool_use_id="tool-1")
        risky = {"command": "git clean -fd"}
        exception = {"command": "pytest -q", "dangerouslyDisableSandbox": True}
        for arguments in (risky, exception):
            result = await check_tool(options, "Bash", arguments)
            assert result["permissionDecision"] == "ask"
            reply = await options.can_use_tool("Bash", arguments, context)
            assert isinstance(reply, sdk.PermissionResultDeny if arguments is risky else sdk.PermissionResultAllow)
        saved = await check_tool(options, "Bash", exception)
        assert saved["permissionDecision"] == "allow"
        yield sdk.ResultMessage(session_id="id", is_error=False, terminal_reason=None, result=None)

    async def approve(call, reason, save):
        seen.append((call, reason, save))
        return "approve_and_save" if save else "deny"

    sdk.script = script
    try:
        events = [event async for event in process.run(request, approve)]
        assert events[0] == ExternalSessionStarted("id")
        assert [entry[-1] for entry in seen] == [False, True]
    finally:
        await process.close()


async def test_read_only_role_has_no_shell_or_edit_tools(tmp_path, sdk_boundary):
    backend, _ = sdk_boundary
    request = request_for(tmp_path, ExternalToolPolicy(write_scope=()))
    process = await backend.start(request, resume=None, fork_session=False)

    async def script(options):
        assert set(options.tools) == {"Read", "Glob", "Grep"}
        assert (await check_tool(options, "Write", {"file_path": "test.txt"}))["permissionDecision"] == "deny"
        assert (await check_tool(options, "Bash", {"command": "pytest"}))["permissionDecision"] == "deny"
        yield sdk.ResultMessage(session_id="read-only", is_error=False, terminal_reason=None, result=None)

    async def approve(*args):
        pytest.fail("A prohibited tool cannot be approved")

    sdk.script = script
    try:
        assert [event async for event in process.run(request, approve)][0] == ExternalSessionStarted("read-only")
    finally:
        await process.close()


async def test_partial_and_complete_messages_report_only_observed_text_and_file_changes(
    tmp_path, sdk_boundary,
):
    backend, _ = sdk_boundary
    request = request_for(tmp_path)
    process = await backend.start(request, resume=None, fork_session=False)
    file = request.workspace / "scratch.txt"

    async def script(options):
        yield sdk.SystemMessage(subtype="init", data={"session_id": "observed"})
        for kind, chunks, complete in (
            ("thinking", ["thinking"], sdk.ThinkingBlock(thinking="thinking")),
            ("text", ["Do", "ne"], sdk.TextBlock(text="Done")),
        ):
            yield sdk.StreamEvent(event={"type": "content_block_start", "index": 0, "content_block": {"type": kind}})
            for chunk in chunks:
                yield sdk.StreamEvent(event={"type": "content_block_delta", "index": 0, "delta": {kind: chunk}})
            yield sdk.AssistantMessage(content=[complete])
            yield sdk.StreamEvent(event={"type": "content_block_stop", "index": 0})

        for call_id in ("change", "unchanged"):
            arguments = {"file_path": str(file), "content": "scratch"}
            yield sdk.AssistantMessage(content=[sdk.ToolUseBlock(id=call_id, name="Write", input=arguments)])
            assert (await check_tool(options, "Write", arguments, call_id))["permissionDecision"] == "allow"
            file.write_text("scratch")
            [post] = options.hooks["PostToolUse"][0].hooks
            await post({"tool_use_id": call_id}, call_id, {})
            yield sdk.UserMessage(content=[sdk.ToolResultBlock(tool_use_id=call_id, content="ok")])
        yield sdk.ResultMessage(session_id="observed", is_error=False, terminal_reason=None, result=None)

    async def approve(*args):
        pytest.fail("Scoped file writes need no approval")

    sdk.script = script
    try:
        events = [event async for event in process.run(request, approve)]
        assert [event for event in events if isinstance(event, ExternalSessionStarted)] == [ExternalSessionStarted("observed")]
        assert [event.text for event in events if isinstance(event, ExternalText)] == ["Do", "ne"]
        assert [event.text for event in events if isinstance(event, ExternalReasoning)] == ["thinking", None]
        changed, unchanged = [event for event in events if isinstance(event, ExternalToolResult)]
        assert changed.file_edit["path"] == "scratch.txt"
        assert changed.file_edit["added"] == 1
        assert unchanged.file_edit is None
    finally:
        await process.close()


sdk = ModuleType("claude_agent_sdk")
for name in (
    "ClaudeAgentOptions", "HookMatcher", "AssistantMessage", "SystemMessage", "StreamEvent",
    "TextBlock", "ThinkingBlock", "ToolUseBlock", "ToolResultBlock", "UserMessage",
    "ResultMessage", "PermissionResultAllow", "PermissionResultDeny",
):
    setattr(sdk, name, type(name, (SimpleNamespace,), {}))
sdk.ClaudeSDKClient = FakeSDKClient
