"""Agent exports without initializing the runtime when importing a tool."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kohaku.agent.context import ContextBuilder
    from kohaku.agent.hook import (
        AgentHook,
        AgentHookContext,
        AgentRunHookContext,
        AgentTurnHookContext,
        AgentTurnHookFactory,
        CompositeHook,
    )
    from kohaku.agent.loop import AgentLoop
    from kohaku.agent.memory import MemoryStore
    from kohaku.agent.skills import SkillsLoader
    from kohaku.agent.subagent import SubagentManager


_LAZY_EXPORTS = {
    "ContextBuilder": ".context",
    "AgentHook": ".hook",
    "AgentHookContext": ".hook",
    "AgentRunHookContext": ".hook",
    "AgentTurnHookContext": ".hook",
    "AgentTurnHookFactory": ".hook",
    "CompositeHook": ".hook",
    "AgentLoop": ".loop",
    "MemoryStore": ".memory",
    "SkillsLoader": ".skills",
    "SubagentManager": ".subagent",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "AgentHook",
    "AgentHookContext",
    "AgentRunHookContext",
    "AgentTurnHookContext",
    "AgentTurnHookFactory",
    "AgentLoop",
    "CompositeHook",
    "ContextBuilder",
    "MemoryStore",
    "SkillsLoader",
    "SubagentManager",
]
