"""Execute tool calls and turn their outcomes into model observations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.policy import ToolPolicy
from nanobot.agent.tools.base import ToolPreparationError
from nanobot.agent.tools.registry import ToolRegistry, is_tool_error_result
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.providers.base import ToolCallRequest
from nanobot.utils.runtime import (
    repeated_external_lookup_error,
    repeated_workspace_violation_error,
)

if TYPE_CHECKING:
    from nanobot.agent.tools.artifact import ArtifactRetrievalReceipt
    from nanobot.agent.tools.read_provenance import FilesystemReadReceipt
    from nanobot.session.artifact_grants import ArtifactHandoffReceipt


@dataclass(frozen=True, slots=True)
class KnownToolResult:
    """Detached evidence before projection normalization or observer callbacks."""

    ordinal: int
    tool_call_id: str
    name: str
    raw_result: Any
    result: Any
    event: dict[str, str]
    dispatched: bool
    producer_receipt: FilesystemReadReceipt | None = None
    retrieval_receipt: ArtifactRetrievalReceipt | None = None
    handoff_receipt: ArtifactHandoffReceipt | None = None


ToolResultCallback = Callable[[KnownToolResult], Awaitable[None]]


def _copy_recorded_value(value: Any) -> Any:
    """Copy model-visible payloads without copying process-local authorities."""
    if isinstance(value, ToolOutcome):
        return ToolOutcome(
            content=deepcopy(value.content),
            stop_reason=value.stop_reason,
            policy_metadata=deepcopy(value.policy_metadata),
            producer_receipt=value.producer_receipt,
            retrieval_receipt=value.retrieval_receipt,
            handoff_receipt=value.handoff_receipt,
        )
    return deepcopy(value)


async def _record_result(callback: ToolResultCallback | None, ordinal: int,
                         call: ToolCallRequest, raw: Any, result: Any,
                         event: dict[str, str], *, dispatched: bool,
                         producer_receipt: FilesystemReadReceipt | None = None,
                         retrieval_receipt: ArtifactRetrievalReceipt | None = None,
    handoff_receipt: ArtifactHandoffReceipt | None = None) -> None:
    if callback is not None:
        await callback(KnownToolResult(ordinal, call.id, call.name, _copy_recorded_value(raw),
                                       _copy_recorded_value(result), deepcopy(event), dispatched, producer_receipt,
                                       retrieval_receipt, handoff_receipt))

_RETRY_HINT = "\n\n[Analyze the error above and try a different approach.]"
# SSRF is a hard security block at the tool boundary, but the agent turn
# should recover conversationally instead of aborting the runtime.
_SSRF_MARKERS: tuple[str, ...] = (
    "internal/private url detected",
    "private/internal address",
    "private address",
)
_SSRF_BOUNDARY_NOTE = (
    "This is a non-bypassable security boundary. Stop trying to access "
    "private/internal URLs. Do not retry with curl, wget, encoded IPs, "
    "alternate DNS, redirects, proxies, or another tool. Ask the user for "
    "local files, logs, screenshots, or an explicit safe public URL instead. "
    "If the user explicitly trusts this private URL, ask them to whitelist "
    "the exact IP/CIDR via tools.ssrfWhitelist."
)
# Non-SSRF boundary markers returned to the model as recoverable tool errors.
_WORKSPACE_VIOLATION_MARKERS: tuple[str, ...] = (
    "outside the configured workspace",
    "outside allowed directory",
    "working_dir is outside",
    "working_dir could not be resolved",
    "path outside working dir",
    "path traversal detected",
)


def _with_retry_hint(payload: str) -> str:
    """Append the recovery hint exactly once."""
    if payload.endswith(_RETRY_HINT):
        return payload
    return payload + _RETRY_HINT


async def execute_tool_calls(
    tools: ToolRegistry,
    tool_calls: list[ToolCallRequest],
    *,
    concurrent: bool,
    external_lookup_counts: dict[str, int],
    workspace_violation_counts: dict[str, int],
    hook: AgentHook,
    context: AgentHookContext,
    tool_policy: ToolPolicy | None = None,
    messages: list[dict[str, Any]] | None = None,
    turn_context: TurnContext | None = None,
    fail_on_tool_error: bool = False,
    tool_result_callback: ToolResultCallback | None = None,
) -> tuple[list[Any], list[dict[str, str]]]:
    """Execute one model response's tool calls in stable result order."""
    if turn_context is not None and turn_context.execution_owner is not None and tool_result_callback is None:
        raise ValueError("Execution ownership requires durable tool-result recording")
    # Policy inspects the complete response before any side effect can begin.
    terminal: ToolOutcome | None = None
    if tool_policy is not None:
        decision = await tool_policy.evaluate(messages=messages or [], tool_calls=tool_calls)
        if decision.action != "allow":
            terminal = ToolOutcome(decision.response, decision.stop_reason or "policy_blocked",
                                   decision.metadata)
    tool_results: list[tuple[Any, dict[str, str]]] = []
    # Contextual delegates require strict stop propagation across the batch.
    concurrent = (concurrent and turn_context is None and tool_policy is None
                  and not fail_on_tool_error and tool_result_callback is None)

    ordinal = 0
    for batch in _partition_tool_batches(tools, tool_calls, concurrent=concurrent):
        if concurrent and len(batch) > 1 and terminal is None:
            batch_results = await asyncio.gather(*(
                _execute_tool_call(
                    tools,
                    tool_call,
                    external_lookup_counts,
                    workspace_violation_counts,
                    hook,
                    context,
                    turn_context,
                )
                for tool_call in batch
            ))
            terminal = next((value for value, _ in batch_results
                             if isinstance(value, ToolOutcome) and value.stop_reason
                             and value.stop_reason != "tool_error"), None)
            tool_results.extend((value.content if isinstance(value, ToolOutcome)
                                 and value.stop_reason == "tool_error" else value, event)
                                for value, event in batch_results)
        else:
            for tool_call in batch:
                if terminal is not None:
                    if (
                        terminal.stop_reason == "folder_access_required"
                        and fail_on_tool_error
                        and _is_safe_catalog_tail(tools, tool_call)
                    ):
                        result = await _execute_tool_call(
                            tools,
                            tool_call,
                            external_lookup_counts,
                            workspace_violation_counts,
                            hook,
                            context,
                            turn_context,
                            tool_result_callback,
                            ordinal,
                        )
                        ordinal += 1
                        outcome, event = result
                        if isinstance(outcome, ToolOutcome):
                            if outcome.stop_reason:
                                terminal = outcome
                        elif event.get("status") == "error":
                            terminal = ToolOutcome(outcome, "tool_error")
                            result = (terminal, event)
                        tool_results.append(result)
                        continue
                    payload = terminal if not tool_results else "Skipped: tool batch stopped."
                    event = {"name": tool_call.name, "status": "skipped", "detail": "tool batch stopped"}
                    await _record_result(tool_result_callback, ordinal, tool_call, payload,
                                         payload, event, dispatched=False)
                    ordinal += 1
                    tool_results.append((payload, event))
                    continue
                result = await _execute_tool_call(
                    tools,
                    tool_call,
                    external_lookup_counts,
                    workspace_violation_counts,
                    hook,
                    context,
                    turn_context,
                    tool_result_callback,
                    ordinal,
                )
                ordinal += 1
                outcome, event = result
                if isinstance(outcome, ToolOutcome):
                    if outcome.stop_reason and (outcome.stop_reason != "tool_error" or fail_on_tool_error):
                        terminal = outcome
                    else:
                        result = (outcome.content, event)
                elif fail_on_tool_error and event.get("status") == "error":
                    terminal = ToolOutcome(outcome, "tool_error")
                    result = (terminal, event)
                tool_results.append(result)

    results = [result for result, _event in tool_results]
    events = [event for _result, event in tool_results]
    return results, events


def _is_safe_catalog_tail(tools: ToolRegistry, call: ToolCallRequest) -> bool:
    """Allow only a concrete first-party, already-queued filtered catalog read."""
    from nanobot.agent.tools.repositories import RepositoryCatalogTool

    registered = tools.get(call.name)
    if call.name != "repository_catalog" or type(registered) is not RepositoryCatalogTool:
        return False
    prepared = tools.prepare_call(call.name, call.arguments)
    tool, params, error = prepared
    return (
        error is None
        and type(tool) is RepositoryCatalogTool
        and type(params) is dict
        and params in ({"action": "list"}, {"action": "roots"})
    )


async def _execute_tool_call(
    tools: ToolRegistry,
    tool_call: ToolCallRequest,
    external_lookup_counts: dict[str, int],
    workspace_violation_counts: dict[str, int],
    hook: AgentHook,
    context: AgentHookContext,
    turn_context: TurnContext | None = None,
    tool_result_callback: ToolResultCallback | None = None,
    ordinal: int = 0,
) -> tuple[Any, dict[str, str]]:
    lookup_error = repeated_external_lookup_error(
        tool_call.name,
        tool_call.arguments,
        external_lookup_counts,
    )
    if lookup_error:
        event = {
            "name": tool_call.name,
            "status": "error",
            "detail": "repeated external lookup blocked",
        }
        payload = _with_retry_hint(lookup_error)
        await _record_result(tool_result_callback, ordinal, tool_call, lookup_error,
                             payload, event, dispatched=False)
        return payload, event

    prepare_call = cast(
        Callable[[str, Any], object] | None,
        getattr(tools, "prepare_call", None),
    )
    tool, params, prep_error = None, tool_call.arguments, None
    if callable(prepare_call):
        prepared = prepare_call(tool_call.name, tool_call.arguments)
        if isinstance(prepared, tuple):
            prepared_tuple = cast(tuple[object, ...], prepared)
            if len(prepared_tuple) == 3:
                tool, params, prep_error = cast(tuple[Any, Any, str | None], prepared_tuple)
    if prep_error:
        preparation_error_kind = (
            prep_error.kind.value if isinstance(prep_error, ToolPreparationError) else None
        )
        payload = _with_retry_hint(prep_error)
        event = {
            "name": tool_call.name,
            "status": "error",
            "detail": prep_error.split(": ", 1)[-1][:120],
        }
        if preparation_error_kind is not None:
            event["_prepare_error_kind"] = preparation_error_kind
        handled = _classify_violation(
            raw_text=prep_error,
            soft_payload=payload,
            event=event,
            tool_call=tool_call,
            workspace_violation_counts=workspace_violation_counts,
        )
        if handled is not None:
            payload, event = handled
        await _record_result(tool_result_callback, ordinal, tool_call, prep_error,
                             payload, event, dispatched=False)
        return payload, event

    await hook.before_execute_tool(context, tool_call, tool, params)
    with tools.dispatch_admission(tool_call.name) as dispatch_error:
        if dispatch_error is not None:
            payload = _with_retry_hint(dispatch_error)
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": dispatch_error[:120],
            }
            await _record_result(tool_result_callback, ordinal, tool_call, dispatch_error,
                                 payload, event, dispatched=False)
            return payload, event
        if turn_context is not None and turn_context.busy_turn is not None:
            turn_context.busy_turn.check_stop()
        if turn_context is not None and turn_context.execution_owner is not None:
            turn_context.execution_owner.validate_dispatch()
        try:
            if tool is not None:
                result = await tools.execute_prepared(tool, params, turn_context)
            else:
                result = await tools.execute(tool_call.name, params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            payload = _with_retry_hint(f"Error: {type(exc).__name__}: {exc}")
            handled = _classify_violation(
                raw_text=str(exc),
                soft_payload=payload,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=workspace_violation_counts,
            )
            if handled is not None:
                payload, event = handled
            await _record_result(tool_result_callback, ordinal, tool_call,
                                 {"exception_type": type(exc).__name__, "message": str(exc)},
                                 payload, event, dispatched=True)
            await hook.on_execute_tool_error(context, tool_call, tool, params, exc)
            return payload, event

    raw_result = result
    outcome = result if isinstance(result, ToolOutcome) else None
    if outcome is not None:
        result = outcome.content
        if outcome.stop_reason and outcome.stop_reason != "tool_error":
            event = {"name": tool_call.name, "status": "stopped", "detail": outcome.stop_reason}
            await _record_result(tool_result_callback, ordinal, tool_call, raw_result,
                                 outcome, event, dispatched=True)
            await hook.after_execute_tool(context, tool_call, tool, params, result)
            return outcome, event

    if is_tool_error_result(result) or (outcome is not None and outcome.stop_reason == "tool_error"):
        result = str(result)
        payload = _with_retry_hint(result)
        event = {
            "name": tool_call.name,
            "status": "error",
            "detail": result.replace("\n", " ").strip()[:120],
        }
        handled = _classify_violation(
            raw_text=result,
            soft_payload=payload,
            event=event,
            tool_call=tool_call,
            workspace_violation_counts=workspace_violation_counts,
        )
        if handled is not None:
            payload, event = handled
        classified = ToolOutcome(payload, "tool_error", outcome.policy_metadata if outcome else {})
        await _record_result(tool_result_callback, ordinal, tool_call, raw_result,
                             classified, event, dispatched=True)
        await hook.on_execute_tool_error(context, tool_call, tool, params, result)
        return classified, event

    detail = "" if result is None else str(result)
    detail = detail.replace("\n", " ").strip()
    if not detail:
        detail = "(empty)"
    elif len(detail) > 120:
        detail = detail[:120] + "..."
    event = {"name": tool_call.name, "status": "ok", "detail": detail}
    await _record_result(tool_result_callback, ordinal, tool_call, raw_result,
                         result, event, dispatched=True,
                         producer_receipt=outcome.producer_receipt if outcome else None,
                         retrieval_receipt=outcome.retrieval_receipt if outcome else None,
                         handoff_receipt=outcome.handoff_receipt if outcome else None)
    await hook.after_execute_tool(context, tool_call, tool, params, result)
    return result, event


def is_ssrf_violation(text: str) -> bool:
    """Return whether a tool error describes a blocked private-network request."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _SSRF_MARKERS)


def _is_workspace_violation(text: str) -> bool:
    """Return whether text describes any workspace or network boundary rejection."""
    if not text:
        return False
    lowered = text.lower()
    if is_ssrf_violation(lowered):
        return True
    return any(marker in lowered for marker in _WORKSPACE_VIOLATION_MARKERS)


def _classify_violation(
    *,
    raw_text: str,
    soft_payload: str,
    event: dict[str, str],
    tool_call: ToolCallRequest,
    workspace_violation_counts: dict[str, int],
) -> tuple[Any, dict[str, str]] | None:
    if is_ssrf_violation(raw_text):
        logger.warning(
            "Tool {} blocked by SSRF guard; returning non-retryable tool error: {}",
            tool_call.name,
            raw_text.replace("\n", " ").strip()[:200],
        )
        event["detail"] = _event_detail("ssrf_violation: ", raw_text)
        return _ssrf_soft_payload(raw_text), event

    if _is_workspace_violation(raw_text):
        escalation = repeated_workspace_violation_error(
            tool_call.name,
            tool_call.arguments,
            workspace_violation_counts,
        )
        event["detail"] = _event_detail("workspace_violation: ", raw_text)
        if escalation is not None:
            logger.warning(
                "Tool {} hit workspace boundary repeatedly; escalating hint",
                tool_call.name,
            )
            event["detail"] = _event_detail(
                "workspace_violation_escalated: ",
                raw_text,
            )
            return escalation, event
        return soft_payload, event

    return None


def _ssrf_soft_payload(raw_text: str) -> str:
    text = raw_text.strip() or "Error: request blocked by SSRF guard"
    return f"{text}\n\n{_SSRF_BOUNDARY_NOTE}"


def _event_detail(prefix: str, text: str, limit: int = 160) -> str:
    return (prefix + text.replace("\n", " ").strip())[:limit]


def _partition_tool_batches(
    tools: ToolRegistry,
    tool_calls: list[ToolCallRequest],
    *,
    concurrent: bool,
) -> list[list[ToolCallRequest]]:
    if not concurrent:
        return [[tool_call] for tool_call in tool_calls]

    batches: list[list[ToolCallRequest]] = []
    current: list[ToolCallRequest] = []
    for tool_call in tool_calls:
        get_tool = cast(Callable[[str], Any] | None, getattr(tools, "get", None))
        tool = get_tool(tool_call.name) if callable(get_tool) else None
        can_batch = bool(tool and tool.concurrency_safe)
        if can_batch:
            current.append(tool_call)
            continue
        if current:
            batches.append(current)
            current = []
        batches.append([tool_call])
    if current:
        batches.append(current)
    return batches
