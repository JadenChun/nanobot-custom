"""Own live approval requests for native and external turns."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from kohaku.agent.permission_rules import extract_saveable_sandbox_command
from kohaku.agent.turn import ApprovalDecision

if TYPE_CHECKING:
    from kohaku.harness.backend import ExternalApprovalAction, ExternalToolCall, ExternalTurnRequest


class TurnApprovalSaveError(RuntimeError):
    """The owner could not persist the approved exclusion."""


@dataclass(frozen=True, slots=True)
class PendingTurnApproval:
    approval_id: str
    session_key: str
    session_incarnation: str
    run_id: str
    runtime_generation: int
    tool_call_id: str
    tool_name: str
    arguments_json: str
    reason: str
    save_exclusion_command: str | None


@dataclass(slots=True)
class _ApprovalWaiter:
    approval: PendingTurnApproval
    future: asyncio.Future[ExternalApprovalAction]
    is_authorized: Callable[[], bool]
    control_session_key: str
    require_exact_identity: bool


class TurnApprovalRegistry:
    """Own live approvals and settle each server-issued ID once."""

    def __init__(self, save_exclusion: Callable[[str], bool | None] | None = None) -> None:
        self._pending: dict[str, _ApprovalWaiter] = {}
        self._save_exclusion = save_exclusion

    async def request(
        self, request: ExternalTurnRequest, call: ExternalToolCall, reason: str, allow_save: bool
    ) -> ExternalApprovalAction:
        if request.on_approval is None or not request.is_authorized():
            return "deny"
        command = call.arguments.get("command")
        saveable = (
            extract_saveable_sandbox_command(command)
            if allow_save and call.name == "exec" and self._save_exclusion is not None
            and isinstance(command, str)
            else None
        )
        approval = PendingTurnApproval(
            uuid4().hex, request.session_key, request.writer.committed_state.incarnation,
            request.run_id, request.binding.runtime_generation,
            call.id, call.name, json.dumps(call.arguments, sort_keys=True), reason, saveable,
        )
        return await self.wait_for_decision(
            approval,
            is_authorized=request.is_authorized,
            control_session_key=request.approval_session_key or request.session_key,
            on_approval=request.on_approval,
            on_closed=request.on_approval_closed,
        )

    async def wait_for_decision(
        self,
        approval: PendingTurnApproval,
        *,
        is_authorized: Callable[[], bool],
        control_session_key: str,
        on_approval: Callable[[PendingTurnApproval], Awaitable[None]],
        on_closed: Callable[[str], Awaitable[None]] | None,
        require_exact_identity: bool = False,
    ) -> ExternalApprovalAction:
        if not is_authorized() or approval.approval_id in self._pending:
            return "deny"
        future: asyncio.Future[ExternalApprovalAction] = asyncio.get_running_loop().create_future()
        self._pending[approval.approval_id] = _ApprovalWaiter(
            approval, future, is_authorized, control_session_key, require_exact_identity,
        )
        try:
            await on_approval(approval)
            decision = await future
            return decision if is_authorized() else "deny"
        finally:
            self._pending.pop(approval.approval_id, None)
            future.cancel()
            if on_closed is not None:
                await on_closed(approval.approval_id)

    def owns(self, approval_id: str) -> bool:
        return approval_id in self._pending

    def resolve(
        self, decision: ApprovalDecision, *, save_exclusion_command: str | None = None,
        control_session_key: str | None = None,
    ) -> bool:
        waiter = self._pending.get(decision.approval_id)
        if waiter is None or waiter.future.done():
            return False
        approval = waiter.approval
        if (
            (control_session_key or decision.session_key) != waiter.control_session_key
            or decision.session_key != approval.session_key or decision.run_id != approval.run_id
            or decision.generation != approval.runtime_generation
            or ((waiter.require_exact_identity or decision.session_incarnation is not None)
                and decision.session_incarnation != approval.session_incarnation)
            or ((waiter.require_exact_identity or decision.effect_id is not None)
                and decision.effect_id != approval.tool_call_id)
            or not waiter.is_authorized()
        ):
            return False
        if decision.decision == "approve_and_save":
            command = approval.save_exclusion_command
            save = self._save_exclusion
            if command is None or save is None or save_exclusion_command not in (None, command):
                return False
            if save(command) is False:
                raise TurnApprovalSaveError("Kohaku could not save the Sandbox exclusion.")
        elif save_exclusion_command is not None:
            return False
        self._pending.pop(approval.approval_id)
        action: ExternalApprovalAction = (
            "approve_and_save" if decision.decision == "approve_and_save"
            else "approve" if decision.decision == "approve" else "deny"
        )
        waiter.future.set_result(action)
        return True

    def pending_tool_calls(self, run_id: str) -> frozenset[str]:
        return frozenset(
            waiter.approval.tool_call_id for waiter in self._pending.values()
            if waiter.approval.run_id == run_id and not waiter.future.done()
        )

    def cancel_for_run(self, run_id: str) -> None:
        self._cancel(lambda approval: approval.run_id == run_id)

    def cancel_for_session(self, session_key: str) -> None:
        self._cancel(lambda approval: approval.session_key == session_key)

    def _cancel(self, matches: Callable[[PendingTurnApproval], bool]) -> None:
        for key, waiter in tuple(self._pending.items()):
            if matches(waiter.approval):
                del self._pending[key]
                if not waiter.future.done():
                    waiter.future.set_result("deny")
