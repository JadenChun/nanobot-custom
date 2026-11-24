"""Gateway admission for direct Claude Code turns."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from kohaku.agent.runner import AgentRunResult
from kohaku.bus.outbound_events import ExternalApprovalEvent, SessionUpdatedEvent
from kohaku.harness.anthropic_bridge import LoopbackAnthropicBridge
from kohaku.harness.backend import (
    ExternalToolPolicy,
    ExternalTurnRequest,
    HarnessCapabilityError,
    PendingExternalApproval,
)
from kohaku.session.harness_binding import (
    EXTERNAL_ACTIVE_RUN_KEY,
    EXTERNAL_RECOVERY_KEY,
    HARNESS_BINDING_KEY,
    SessionHarnessBinding,
    external_binding,
)
from kohaku.session.source import validate_execution_links

if TYPE_CHECKING:
    from collections.abc import Callable

    from kohaku.agent.hook import AgentHook
    from kohaku.agent.turn import TurnContext
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
            and (full is None or permissions.is_current_full_reference(full))
            and workspace_is_authorized()
        )

    async def approval_requested(approval: PendingExternalApproval) -> None:
        await events.emit(ExternalApprovalEvent(approval_id=approval.approval_id, approval=approval))

    async def approval_closed(approval_id: str) -> None:
        await events.emit(ExternalApprovalEvent(approval_id=approval_id))

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
        await events.emit(SessionUpdatedEvent(scope="metadata", harness_binding=binding.payload()))
        if full is not None:
            launched = permissions.register_native_stdin(
                full, task, revoke_full, lambda: (int(active), 0),
            )
            if not launched:
                raise HarnessCapabilityError("Full access is no longer available for this chat.")
        outcome = await runner.run(ExternalTurnRequest(
            context.session_key, writer.run_id, binding, workspace, prompt, runtime, hook, writer,
            is_authorized, policy=policy, usage_source=usage_source,
            on_approval=approval_requested, events=events, private_roots=private_roots,
            on_approval_closed=approval_closed, retain_process=False,
        ))
        return AgentRunResult(
            outcome.content or outcome.error,
            writer.committed_events,
            stop_reason="error" if outcome.status == "failed" else outcome.status,
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
        if saved is not None and saved != binding:
            await events.emit(SessionUpdatedEvent(scope="metadata", harness_binding=saved.payload()))
