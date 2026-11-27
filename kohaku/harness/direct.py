"""Gateway admission for direct Claude Code turns."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from kohaku.agent.auto_review import (
    AutoReviewResult,
    build_auto_review_context,
    review_sandbox_escalation,
)
from kohaku.agent.runner import AgentRunResult
from kohaku.bus.outbound_events import ExternalApprovalEvent, SessionUpdatedEvent
from kohaku.events import NO_EVENTS
from kohaku.harness.anthropic_bridge import LoopbackAnthropicBridge
from kohaku.harness.backend import (
    ExternalToolCall,
    ExternalToolPolicy,
    ExternalTurnOutcome,
    ExternalTurnRequest,
    HarnessCapabilityError,
    PendingExternalApproval,
)
from kohaku.providers.base import ToolCallRequest
from kohaku.session.harness_binding import (
    EXTERNAL_ACTIVE_RUN_KEY,
    EXTERNAL_RECOVERY_KEY,
    HARNESS_BINDING_KEY,
    SessionHarnessBinding,
    external_binding,
)
from kohaku.session.source import validate_execution_links

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from kohaku.agent.child_backend import ChildTurnRequest
    from kohaku.agent.hook import AgentHook
    from kohaku.agent.turn import TurnContext
    from kohaku.config.schema import ToolPermissionsConfig
    from kohaku.events import EventSink
    from kohaku.harness.runner import ExternalHarnessRunner
    from kohaku.llm_usage.context import LLMUsageSource
    from kohaku.security.desktop_permissions import DesktopPermissionOwner
    from kohaku.security.workspace_access import WorkspaceScope
    from kohaku.utils.llm_runtime import LLMRuntime


async def start_bridge(
    runtime: LLMRuntime, usage_source: LLMUsageSource | None,
) -> LoopbackAnthropicBridge:
    bridge = LoopbackAnthropicBridge(runtime, usage_source=usage_source)
    await bridge.start()
    return bridge


async def run_direct_turn(
    runner: ExternalHarnessRunner,
    *,
    context: TurnContext,
    binding: SessionHarnessBinding,
    scope: WorkspaceScope,
    prompt: str,
    runtime: LLMRuntime,
    policy: ExternalToolPolicy,
    permissions: DesktopPermissionOwner,
    hook: AgentHook,
    events: EventSink,
    usage_source: LLMUsageSource,
    private_roots: tuple[Path, ...],
    is_closing: Callable[[], bool],
    workspace_is_authorized: Callable[[], bool],
    child: ChildTurnRequest | None = None,
    approval_events: EventSink | None = None,
    on_approval_state: Callable[[bool], Awaitable[None]] | None = None,
    drain_steer: Callable[[Callable[[], Awaitable[None]], bool], Awaitable[list[str]]] | None = None,
    authorize_tool: Callable[[str, dict[str, object]], bool] | None = None,
    tool_permissions: ToolPermissionsConfig | None = None,

) -> AgentRunResult:
    writer = context.source_writer
    if writer is None or context.session_key is None:
        raise HarnessCapabilityError("Claude Code requires a durable session.")
    task = asyncio.current_task()
    assert task is not None
    workspace = scope.project_path.resolve(strict=True)
    if not workspace.is_dir() or binding.locked_workspace not in (None, str(workspace)):
        raise HarnessCapabilityError("The Claude Code working folder is locked or unavailable.")
    full = context.workspace_full_reference if policy.mode == "full_access" else None
    active = True

    def is_authorized() -> bool:
        return (
            active and not task.cancelling() and not is_closing()
            and not (context.explicit_stop_requested and context.explicit_stop_requested())
            and (full is None or permissions.is_current_full_reference(full))
            and workspace_is_authorized()
        )

    approvals = events if approval_events is None else approval_events
    child_id = child.admission.child_id if child is not None else None

    async def approval_requested(approval: PendingExternalApproval) -> None:
        if on_approval_state is not None:
            await on_approval_state(True)
        await approvals.emit(ExternalApprovalEvent(
            approval_id=approval.approval_id, approval=approval, child_id=child_id,
        ))

    async def approval_closed(approval_id: str) -> None:
        await approvals.emit(ExternalApprovalEvent(approval_id=approval_id, child_id=child_id))
        if on_approval_state is not None:
            await on_approval_state(False)

    first_turn = binding.locked_workspace is None
    binding = replace(
        binding, locked_workspace=str(workspace), runtime_generation=binding.runtime_generation + 1,
    )
    writer.store.update_metadata(context.session_key, {
        HARNESS_BINDING_KEY: binding.payload(), EXTERNAL_ACTIVE_RUN_KEY: writer.run_id,
        **({"workspace_scope": scope.metadata()} if first_turn else {}),
    }, removals=(EXTERNAL_RECOVERY_KEY,))
    launched = False

    def revoke_full() -> bool:
        task.get_loop().call_soon_threadsafe(task.cancel)
        return True

    try:
        if child is None:
            await events.emit(SessionUpdatedEvent(scope="metadata", harness_binding=binding.payload()))
        if not is_authorized():
            raise HarnessCapabilityError("The Claude Code turn no longer has working folder access.")
        if full is not None:
            launched = permissions.register_native_stdin(
                full, task, revoke_full, lambda: (int(active), 0),
            )
            if not launched:
                raise HarnessCapabilityError("Full access is no longer available for this chat.")
        trusted_prompt = prompt
        if child is not None:
            prompt = f"{child.instructions}\n\n## Assignment\n{prompt}"

        history_messages = (
            context.source_history.messages() if context.source_history is not None else None
        )

        async def _auto_review(call: ExternalToolCall, reason: str) -> AutoReviewResult:
            review_context = build_auto_review_context(
                history_messages,
                current_user_prompt=trusted_prompt,
                tools_used=context.tools_used,
            )
            return await review_sandbox_escalation(
                runtime=runtime,
                permissions=tool_permissions,
                workspace=workspace,
                call=ToolCallRequest(
                    id=call.id,
                    name=call.name,
                    arguments=dict(call.arguments),
                ),
                summary=reason,
                user_prompt=trusted_prompt,
                review_context=review_context,
            )

        auto_review_fn = (
            _auto_review
            if policy.sandboxed and tool_permissions is not None and tool_permissions.sandbox_auto_review
            else None
        )
        outcome = await runner.run(ExternalTurnRequest(
            context.session_key, writer.run_id, binding, workspace, prompt, runtime, hook, writer,
            is_authorized, policy=policy, usage_source=usage_source,
            on_approval=approval_requested if approvals is not NO_EVENTS else None,
            events=events, private_roots=private_roots,
            max_turns=child.max_iterations if child is not None else None,
            authorize_tool=authorize_tool,
            approval_session_key=child.admission.root_key if child is not None else None,
            prior_external_session_id=child.prior_external_session_id if child is not None else None,
            on_approval_closed=approval_closed, retain_process=False, drain_steer=drain_steer,
            auto_review=auto_review_fn,
        ))
        acknowledge = context.followup_request_acknowledged
        if child is not None and outcome.status == "completed" and acknowledge is not None:
            acknowledge(tuple(context.pending_followup_acknowledgments))
            context.pending_followup_acknowledgments.clear()
        content = outcome.content or outcome.error
        details: dict[str, object] = {}
        if child is not None and child.role == "worker":
            evidence, summary = _worker_evidence(outcome)
            details["implementation"] = evidence
            content = f"{content or ''}\n\n{summary}".strip()
        return AgentRunResult(
            content,
            writer.committed_events,
            stop_reason=(
                "external_effect_uncertain"
                if child is not None and validate_execution_links(writer.owner.read_snapshot().events)
                else "error" if outcome.status == "failed" else outcome.status
            ),
            tools_used=list(outcome.tools_used),
            policy_metadata=details,
            error=outcome.error,
        )
    except HarnessCapabilityError as exc:
        return AgentRunResult(
            str(exc), writer.committed_events, stop_reason="error", error=str(exc),
            failure_error_kind="harness_unavailable",
        )
    finally:
        active = False
        if launched:
            permissions.unregister_native_stdin(full, task)
        # Keep the marker if source recording failed. Startup classifies the
        # interruption from the durable source instead of retrying the SDK.
        snapshot = writer.owner.read_snapshot()
        if not writer.failed and not validate_execution_links(snapshot.events):
            writer.store.update_metadata(context.session_key, {}, removals=(EXTERNAL_ACTIVE_RUN_KEY,))
        saved = external_binding(snapshot.header["metadata"])
        if child is None and saved is not None and saved != binding:
            await events.emit(SessionUpdatedEvent(scope="metadata", harness_binding=saved.payload()))


def _worker_evidence(outcome: ExternalTurnOutcome) -> tuple[dict[str, object], str]:
    files = sorted({path for edit in outcome.file_edits if isinstance(path := edit.get("path"), str)})
    commands = [{"command": command, "outcome": status} for command, status in outcome.commands]
    failures = [row["command"] for row in commands if row["outcome"] == "error"]
    details: dict[str, object] = {
        "changed_files": files, "commands": commands, "failed_commands": failures,
    }
    lines = ["### Observed changed files", *(f"- {path}" for path in files)]
    if not files:
        lines.append("- No file edits reported.")
    lines.append("### Shell commands and outcomes")
    if commands:
        lines.extend(f"- {row['outcome']}: {row['command']}" for row in commands)
    else:
        lines.append("- No shell commands ran.")
    return details, "\n".join(lines)
