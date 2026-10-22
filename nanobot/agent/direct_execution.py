"""Typed optional fields at the direct-adapter boundary; no execution policy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from nanobot.agent.hook import AgentHook, AgentTurnHookFactory
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.agent.turn import TraceMode, TurnSource
    from nanobot.utils.llm_runtime import LLMRuntime


class DirectRequestOptions(TypedDict, total=False):
    """Optional request kwargs, excluding the SDK's separately supplied fields."""

    session_key: str
    channel: str
    chat_id: str
    sender_id: str
    media: list[str] | None
    on_progress: Callable[..., Awaitable[None]] | None
    on_stream: Callable[[str], Awaitable[None]] | None
    on_stream_end: Callable[..., Awaitable[None]] | None
    ephemeral: bool
    _run_extra_hooks_for_ephemeral: bool
    hook_factories: list[AgentTurnHookFactory] | None
    tools: ToolRegistry | None
    persist_user_message: bool
    runtime: LLMRuntime | None
    on_runtime_admitted: Callable[[LLMRuntime], Awaitable[None]] | None
    attributes: Mapping[str, object] | None


class DirectExecutionOptions(DirectRequestOptions, total=False):
    """All supported options for canonical direct-adapter results."""

    hooks: list[AgentHook] | None
    source: TurnSource
    trace_mode: TraceMode
