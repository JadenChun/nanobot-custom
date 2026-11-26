"""Adapt the Claude Agent SDK to Kohaku's external harness contract."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import shutil
import signal
import sys
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

from loguru import logger

from kohaku.agent.permission_rules import (
    extract_saveable_sandbox_command,
    matches_sandbox_excluded_command,
    sandbox_exec_risk_reason,
)
from kohaku.agent.tools.sandbox import managed_bwrap_available
from kohaku.harness.backend import (
    BridgeFactory,
    ExternalEvent,
    ExternalReasoning,
    ExternalSessionStarted,
    ExternalText,
    ExternalToolBatchReady,
    ExternalToolCall,
    ExternalToolIntent,
    ExternalToolResult,
    ExternalTurnFinished,
    ExternalTurnRequest,
    HarnessCapabilityError,
    LoopbackBridgeHandle,
    RequestApproval,
)
from kohaku.utils.file_edit_events import (
    FileEditTracker,
    build_file_edit_end_event,
    display_file_edit_path,
    read_file_snapshot,
)

_READ_TOOLS = ("Read", "Glob", "Grep")
_EDIT_TOOLS = {"Write": "write_file", "Edit": "edit_file", "NotebookEdit": "edit_file"}
_SAFE_ENV = {
    "HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SYSTEMROOT", "WINDIR",
    "PATHEXT", "COMSPEC", "TEMP", "TMP", "TMPDIR",
}


def _object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        data = cast(dict[object, object], value)
        if all(isinstance(key, str) for key in data):
            return cast(dict[str, object], data)
    return {}


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def claude_sdk_available() -> bool:
    """Check whether the optional SDK is installed without loading it."""
    return find_spec("claude_agent_sdk") is not None


def harness_capabilities() -> dict[str, dict[str, bool | str]]:
    available = claude_sdk_available()
    return {"claude_sdk": {
        "available": available,
        **({} if available else {"reason": "sdk_missing"}),
    }}


def child_harness_parameter() -> dict[str, object]:
    guidance = (
        ' Use claude_sdk for codebase investigation, technical planning, implementation, and testing.'
        if claude_sdk_available() else ""
    )
    return {
        "type": "string", "enum": ["native", "claude_sdk"], "default": "native",
        "description": "Execution harness for this role." + guidance,
    }


def _load_sdk() -> ModuleType:
    try:
        return import_module("claude_agent_sdk")
    except ImportError as exc:
        raise HarnessCapabilityError(
            "Claude Code requires the Claude Agent SDK. Reinstall Kohaku or install claude-agent-sdk."
        ) from exc


class ClaudeSDKBackend:
    """Start Claude Code with the current admission and an injected model bridge."""

    def __init__(self, bridge_factory: BridgeFactory, *, state_dir: Path) -> None:
        self._bridge_factory = bridge_factory
        self._state_dir = state_dir.resolve()

    async def start(
        self, request: ExternalTurnRequest, *, resume: str | None, fork_session: bool,
    ) -> _ClaudeProcess:
        sdk = _load_sdk()
        sandboxed = request.policy.sandboxed
        allowed = request.policy.allowed_tools
        if sandboxed and request.policy.write_scope != () and (allowed is None or "Bash" in allowed):
            if sys.platform.startswith("linux"):
                if not await asyncio.to_thread(managed_bwrap_available) or shutil.which("socat") is None:
                    raise HarnessCapabilityError(
                        "Claude Code Sandbox requires a working bubblewrap and socat. Install or enable them."
                    )
            elif sys.platform != "darwin":
                raise HarnessCapabilityError("Claude Code Sandbox requires Linux or macOS.")
        self._state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        bridge = await self._bridge_factory(request.runtime, request.usage_source)
        process: _ClaudeProcess | None = None
        try:
            process = _ClaudeProcess(sdk, bridge, request, self._state_dir, resume, fork_session)
            await process.connect()
        except BaseException:
            if process is None:
                try:
                    await bridge.close()
                except BaseException as cleanup_error:
                    logger.warning(
                        "Claude SDK bridge cleanup failed after connect stopped: {}",
                        type(cleanup_error).__name__,
                    )
            else:
                await process.close()
            raise
        return process


@dataclass(slots=True)
class _Policy:
    workspace: Path
    private: tuple[Path, ...]
    scope: tuple[Path, ...] | None
    sandboxed: bool
    exclusions: tuple[str, ...]
    exact_write_paths: frozenset[Path]
    allowed_tools: frozenset[str] | None

    def path(self, value: object) -> Path | None:
        if not isinstance(value, str) or not value or "\0" in value:
            return None
        try:
            lexical = self.workspace / Path(value).expanduser()
            target = lexical.resolve()
        except (OSError, ValueError, RuntimeError):
            return None
        if ((self.sandboxed or self.scope is not None) and not target.is_relative_to(self.workspace)) or any(
            lexical.is_relative_to(root) or target.is_relative_to(root) for root in self.private
        ):
            return None
        return target

    def check(
        self, name: str, arguments: dict[str, object],
    ) -> tuple[Literal["allow", "deny", "ask"], str, dict[str, object]]:
        if self.allowed_tools is not None and name not in self.allowed_tools:
            return "deny", "This tool is outside the admitted scope.", arguments
        if name in _READ_TOOLS or name in _EDIT_TOOLS:
            search = name in {"Glob", "Grep"}
            path_key = "path" if search else "notebook_path" if name == "NotebookEdit" else "file_path"
            target = self.path(arguments.get(path_key, str(self.workspace) if search else None))
            if target is None or (search and any(root.is_relative_to(target) for root in self.private)):
                return "deny", "The tool path is outside the admitted read scope.", arguments
            if name in _EDIT_TOOLS and self.scope is not None and not any(
                target == root or (root not in self.exact_write_paths and target.is_relative_to(root))
                for root in self.scope
            ):
                return "deny", "The file is outside the admitted write scope.", arguments
            if name == "Glob":
                pattern = Path(_string(arguments.get("pattern")))
                if pattern.is_absolute() or ".." in pattern.parts:
                    return "deny", "The search pattern is outside the admitted read scope.", arguments
            return "allow", "", {**arguments, path_key: str(target)}
        if name != "Bash" or self.scope == ():
            return "deny", "This tool is outside the admitted scope.", arguments
        command = _string(arguments.get("command"))
        if not command:
            return "deny", "The shell command is missing.", arguments
        unsandboxed = arguments.get("dangerouslyDisableSandbox", False)
        if not isinstance(unsandboxed, bool):
            return "deny", "The sandbox override must be a boolean.", arguments
        if unsandboxed and self.scope is not None:
            return "deny", "This assignment requires sandboxed shell commands.", arguments
        if risk := sandbox_exec_risk_reason(command):
            return "ask", f"Allow the command to {risk}?", arguments
        if self.sandboxed and matches_sandbox_excluded_command(command, self.exclusions):
            return "allow", "", {**arguments, "dangerouslyDisableSandbox": True}
        if self.sandboxed and unsandboxed:
            return "ask", "Allow this command to run outside the sandbox?", arguments
        return "allow", "", arguments


@dataclass(slots=True)
class _SDKTurn:
    request: ExternalTurnRequest
    approve: RequestApproval
    trackers: dict[str, FileEditTracker] = field(default_factory=dict)
    edits: dict[str, dict[str, object]] = field(default_factory=dict)
    denied: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _SDKCommand:
    request: ExternalTurnRequest
    approve: RequestApproval
    events: asyncio.Queue[object]


@dataclass(frozen=True, slots=True)
class _SDKFailure:
    error: BaseException


_SDK_END = object()


class _ClaudeProcess:
    def __init__(
        self, sdk: ModuleType, bridge: LoopbackBridgeHandle, request: ExternalTurnRequest,
        state_dir: Path, resume: str | None, fork_session: bool,
    ) -> None:
        self._sdk = sdk
        self._fork_parent_id = resume if fork_session else None
        self._bridge = bridge
        self._active: _SDKTurn | None = None
        self._workspace = request.workspace.resolve()
        scope = request.policy.write_scope
        private = (
            *(Path.home() / name for name in (".kohaku", ".ssh", ".aws")),
            state_dir, *request.private_roots,
        )
        private = tuple(dict.fromkeys((*private, *(root.resolve() for root in private))))
        self._policy = _Policy(
            self._workspace, private,
            None if scope is None else tuple(
                root if root.is_absolute() else (self._workspace / root).resolve()
                for root in scope
            ),
            request.policy.sandboxed,
            request.policy.sandbox_excluded_commands if scope is None else (),
            request.policy.exact_write_paths,
            request.policy.allowed_tools,
        )
        tools: list[str] = list(_READ_TOOLS)
        if scope != ():
            tools.extend((*_EDIT_TOOLS, "Bash"))
        if self._policy.allowed_tools is not None:
            tools = [name for name in tools if name in self._policy.allowed_tools]
        env = {key: value for key, value in os.environ.items() if key.upper() in _SAFE_ENV}
        env.update({
            "ANTHROPIC_BASE_URL": bridge.base_url,
            "ANTHROPIC_AUTH_TOKEN": bridge.auth_token,
            "ANTHROPIC_API_KEY": bridge.auth_token,
            "ANTHROPIC_MODEL": request.runtime.model,
            "ANTHROPIC_SMALL_FAST_MODEL": request.runtime.model,
            "CLAUDE_CONFIG_DIR": str(state_dir),
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
            "CLAUDE_CODE_SDK_READS_SESSION_STATE": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        })
        sandbox = {
            "enabled": self._policy.sandboxed and "Bash" in tools,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": scope is None,
            "network": {
                "allowLocalBinding": request.policy.sandbox_allow_local_binding,
                "allowUnixSockets": list(request.policy.sandbox_allow_unix_sockets) if scope is None else [],
            },
            "filesystem": {
                "denyRead": [str(root) for root in private],
                "denyWrite": [str(root) for root in private],
            },
        }
        # The CLI's exclusion matcher has different semantics from Stage 1B.
        # Apply saved exclusions in PreToolUse after the complete command matches.
        options: Any = sdk.ClaudeAgentOptions(
            cwd=self._workspace, model=request.runtime.model, env=env, resume=resume,
            max_turns=request.max_turns,
            fork_session=fork_session, setting_sources=[], strict_mcp_config=True,
            include_partial_messages=True, tools=tools, verbatim_prompts=True,
            system_prompt={"type": "preset", "preset": "claude_code"},
            settings=json.dumps({"sandbox": sandbox}),
            can_use_tool=self._can_use_tool,
            hooks={
                "PreToolUse": [sdk.HookMatcher(matcher=None, hooks=[self._pre_tool], on_failure="block")],
                "PostToolUse": [sdk.HookMatcher(matcher=None, hooks=[self._post_tool])],
            },
        )
        self._transport = _RestrictedTransport(options, env, self._workspace)
        self._client: Any = sdk.ClaudeSDKClient(options=options, transport=self._transport)
        self._owner_task: asyncio.Task[None] | None = None
        self._owner_ready: asyncio.Future[None] | None = None
        self._commands: asyncio.Queue[_SDKCommand] = asyncio.Queue()
        self._running = False
        self._closing = False

    async def connect(self) -> None:
        if self._owner_task is not None:
            raise RuntimeError("The Claude Code process is already connected.")
        self._owner_ready = asyncio.get_running_loop().create_future()
        self._owner_task = asyncio.create_task(
            self._own_sdk_lifecycle(), context=contextvars.Context(),
        )
        try:
            await asyncio.shield(self._owner_ready)
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        self._closing = True
        owner = self._owner_task
        if owner is not None and not owner.done():
            owner.cancel()
        if owner is not None:
            try:
                await asyncio.shield(owner)
            except asyncio.CancelledError:
                if not owner.cancelled():
                    raise
            except BaseException as cleanup_error:
                logger.warning(
                    "Claude SDK lifecycle owner stopped with an error: {}",
                    type(cleanup_error).__name__,
                )

    async def _own_sdk_lifecycle(self) -> None:
        ready = self._owner_ready
        active_events: asyncio.Queue[object] | None = None
        command: _SDKCommand | None = None
        try:
            await self._client.connect()
            if ready is not None and not ready.done():
                ready.set_result(None)
            while not self._closing:
                command = await self._commands.get()
                active_events = command.events
                try:
                    async for event in self._run_turn(command.request, command.approve):
                        await command.events.put(event)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    await command.events.put(_SDKFailure(exc))
                finally:
                    await command.events.put(_SDK_END)
                    active_events = None
                    command = None
        except BaseException as exc:
            if ready is not None and not ready.done():
                ready.set_exception(exc)
            elif active_events is not None:
                await active_events.put(_SDKFailure(exc))
                await active_events.put(_SDK_END)
        finally:
            try:
                await self._client.disconnect()
            except BaseException as cleanup_error:
                logger.warning(
                    "Claude SDK disconnect failed during lifecycle cleanup: {}",
                    type(cleanup_error).__name__,
                )
            try:
                await self._transport.close()
            except BaseException as cleanup_error:
                logger.warning(
                    "Claude SDK transport failed during lifecycle cleanup: {}",
                    type(cleanup_error).__name__,
                )
            try:
                await self._bridge.close()
            except BaseException as cleanup_error:
                logger.warning(
                    "Claude SDK bridge failed during lifecycle cleanup: {}",
                    type(cleanup_error).__name__,
                )

    async def _pre_tool(
        self, hook_input: object, _tool_use_id: object, _context: object,
    ) -> dict[str, object]:
        input_data = _object(hook_input)
        name = _string(input_data.get("tool_name"))
        call_id = _string(input_data.get("tool_use_id"))
        active = self._active
        if not call_id or active is None or not active.request.is_authorized():
            decision, reason, arguments = "deny", "The turn has no current permission to run.", {}
        else:
            decision, reason, arguments = self._policy.check(name, _object(input_data.get("tool_input")))
            authorize = active.request.authorize_tool
            if decision != "deny" and authorize is not None and not authorize(name, arguments):
                decision, reason = "deny", "The child tool no longer has access."
            if decision == "deny":
                active.denied.add(call_id)
            if decision == "allow" and name in _EDIT_TOOLS:
                key = "notebook_path" if name == "NotebookEdit" else "file_path"
                target = Path(_string(arguments.get(key)))
                active.trackers[call_id] = FileEditTracker(
                    call_id, _EDIT_TOOLS[name], target,
                    display_file_edit_path(target, self._workspace), read_file_snapshot(target),
                )
        output: dict[str, object] = {
            "hookEventName": "PreToolUse", "permissionDecision": decision,
        }
        if reason:
            output["permissionDecisionReason"] = reason
        if decision == "allow":
            output["updatedInput"] = arguments
        return {"hookSpecificOutput": output}

    async def _post_tool(
        self, hook_input: object, _tool_use_id: object, _context: object,
    ) -> dict[str, object]:
        call_id = _string(_object(hook_input).get("tool_use_id"))
        if self._active is not None and (tracker := self._active.trackers.pop(call_id, None)):
            if self._policy.path(str(tracker.path)) == tracker.path:
                after = read_file_snapshot(tracker.path)
                if after.exists != tracker.before.exists or (
                    after.countable and tracker.before.countable and after.text != tracker.before.text
                ):
                    self._active.edits[call_id] = build_file_edit_end_event(tracker)
        return {}

    async def _can_use_tool(self, name: str, raw: object, context: object) -> object:
        active = self._active
        call_id = _string(getattr(context, "tool_use_id", None))
        if not call_id or active is None or not active.request.is_authorized():
            return self._sdk.PermissionResultDeny(message="The turn has no current permission to run.")
        arguments = _object(raw)
        decision, reason, args = self._policy.check(name, arguments)
        authorize = active.request.authorize_tool
        if decision != "deny" and authorize is not None and not authorize(name, args):
            decision, reason = "deny", "The child tool no longer has access."
        if decision == "ask":
            allow_save = (
                self._policy.sandboxed and self._policy.scope is None
                and arguments.get("dangerouslyDisableSandbox") is True
                and sandbox_exec_risk_reason(_string(arguments.get("command"))) is None
            )
            action = await active.approve(ExternalToolCall(call_id, name, arguments), reason, allow_save)
            if action != "deny":
                decision = "allow"
            if action == "approve_and_save" and allow_save:
                saved = extract_saveable_sandbox_command(_string(arguments.get("command")))
                if saved and saved not in self._policy.exclusions:
                    self._policy.exclusions += (saved,)
        if (
            decision == "allow" and self._active is active and active.request.is_authorized()
            and (authorize is None or authorize(name, args))
        ):
            return self._sdk.PermissionResultAllow(updated_input=args)
        active.denied.add(call_id)
        return self._sdk.PermissionResultDeny(message=reason or "The tool request was denied.")

    async def run(
        self, request: ExternalTurnRequest, approve: RequestApproval,
    ) -> AsyncGenerator[ExternalEvent, None]:
        if self._closing or self._owner_task is None or self._owner_task.done():
            raise RuntimeError("The Claude Code process is closed.")
        if self._running:
            raise RuntimeError("The Claude Code process already has an active turn.")
        self._running = True
        events: asyncio.Queue[object] = asyncio.Queue()
        await self._commands.put(_SDKCommand(request, approve, events))
        try:
            while True:
                event = await events.get()
                if event is _SDK_END:
                    return
                if isinstance(event, _SDKFailure):
                    raise event.error
                yield cast(ExternalEvent, event)
        finally:
            self._running = False

    async def _run_turn(
        self, request: ExternalTurnRequest, approve: RequestApproval,
    ) -> AsyncGenerator[ExternalEvent, None]:
        if self._active is not None:
            raise RuntimeError("The Claude Code process already has an active turn.")
        # A saved exclusion is live for this turn. The next admission supplies
        # the current exclusions, including any that the owner has removed.
        self._policy.exclusions = (
            request.policy.sandbox_excluded_commands if self._policy.scope is None else ()
        )
        active = _SDKTurn(request, approve)
        self._active = active
        sdk = self._sdk
        partial: dict[str, deque[list[str]]] = {"text": deque(), "thinking": deque()}
        current_blocks: dict[int, tuple[str, list[str]]] = {}
        session_id = None
        try:
            await self._client.query(request.prompt)
            async for message in self._client.receive_response():
                if getattr(message, "parent_tool_use_id", None) is not None:
                    continue
                observed_id = _string(getattr(message, "session_id", None))
                if isinstance(message, sdk.SystemMessage) and message.subtype == "init":
                    observed_id = _string(_object(message.data).get("session_id"))
                if observed_id and observed_id != session_id:
                    if observed_id == self._fork_parent_id:
                        yield ExternalTurnFinished(error="Claude Code did not create a distinct fork.")
                        return
                    session_id = observed_id
                    yield ExternalSessionStarted(observed_id)
                if isinstance(message, sdk.StreamEvent):
                    raw = _object(message.event)
                    if raw.get("type") == "message_stop":
                        yield ExternalToolBatchReady()
                    index = raw.get("index")
                    if not isinstance(index, int):
                        continue
                    if raw.get("type") == "content_block_start":
                        kind = _string(_object(raw.get("content_block")).get("type"))
                        if kind in partial:
                            parts: list[str] = []
                            current_blocks[index] = kind, parts
                            partial[kind].append(parts)
                    elif index in current_blocks:
                        kind, parts = current_blocks[index]
                        if raw.get("type") == "content_block_delta":
                            delta = _string(_object(raw.get("delta")).get(kind))
                            if delta:
                                parts.append(delta)
                                yield ExternalText(delta) if kind == "text" else ExternalReasoning(delta)
                        elif raw.get("type") == "content_block_stop":
                            del current_blocks[index]
                            if kind == "thinking" and parts:
                                yield ExternalReasoning(None)
                elif isinstance(message, sdk.AssistantMessage):
                    for block in message.content:
                        if isinstance(block, sdk.TextBlock):
                            text = _remaining(_string(block.text), partial["text"])
                            if text:
                                yield ExternalText(text)
                        elif isinstance(block, sdk.ThinkingBlock):
                            thinking = _remaining(_string(block.thinking), partial["thinking"])
                            if thinking:
                                yield ExternalReasoning(thinking)
                                yield ExternalReasoning(None)
                        elif isinstance(block, sdk.ToolUseBlock):
                            yield ExternalToolIntent(ExternalToolCall(
                                _string(block.id), _string(block.name), _object(block.input),
                            ))
                elif isinstance(message, sdk.UserMessage):
                    content: object = message.content
                    if isinstance(content, list):
                        for block in cast(list[object], content):
                            if isinstance(block, sdk.ToolResultBlock):
                                call_id = _string(getattr(block, "tool_use_id", None))
                                yield ExternalToolResult(
                                    call_id, getattr(block, "content", None),
                                    bool(getattr(block, "is_error", False)),
                                    call_id not in active.denied, active.edits.pop(call_id, None),
                                )
                elif isinstance(message, sdk.ResultMessage):
                    yield ExternalTurnFinished(
                        error="Claude Code reported a turn error." if message.is_error else None,
                        cancelled=message.terminal_reason in {"aborted_streaming", "aborted_tools"},
                        final_text=_string(message.result) or None,
                    )
                    return
        finally:
            self._active = None


def _remaining(full: str, partial: deque[list[str]]) -> str:
    streamed = "".join(partial.popleft()) if partial else ""
    return full[len(streamed):] if full.startswith(streamed) else full


class _RestrictedTransport:
    """Use SDK framing and CLI arguments with an explicitly owned subprocess."""

    def __init__(self, options: Any, env: dict[str, str], cwd: Path) -> None:
        self._options = options
        self._env = env
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None

    async def connect(self) -> None:
        # SDK 0.2.188 merges all of os.environ in its transport. Reuse its CLI
        # argument builder, but start the process with only the admitted env.
        sdk_transport = import_module("claude_agent_sdk._internal.transport.subprocess_cli")
        options = import_module("claude_agent_sdk.types")._configure_can_use_tool(self._options)
        builder: Any = sdk_transport.SubprocessCLITransport(prompt="", options=options)
        cli: str = await asyncio.to_thread(builder._find_cli)
        builder._reject_windows_batch_cli(cli)
        builder._cli_path = cli
        command: list[str] = builder._build_command()
        self._process = await asyncio.create_subprocess_exec(
            *command, cwd=self._cwd, env=self._env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=1024 * 1024 * 16,
            start_new_session=os.name == "posix",
        )

    def is_ready(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def write(self, data: str) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("The Claude Code process is closed.")
        process.stdin.write(data.encode())
        await process.stdin.drain()

    async def read_messages(self) -> AsyncIterator[dict[str, object]]:
        process = self._process
        assert process is not None and process.stdout is not None
        async for line in process.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            message = _object(json.loads(stripped))
            if message:
                yield message

    async def end_input(self) -> None:
        if self._process is not None and self._process.stdin is not None:
            self._process.stdin.close()

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        await self.end_input()
        try:
            if os.name == "nt" and process.returncode is None:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill", "/T", "/F", "/PID", str(process.pid),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(killer.wait(), timeout=3)
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            pass
        finally:
            with suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.returncode is None:
                    process.kill()
        await asyncio.wait_for(process.wait(), timeout=3)
        self._process = None
