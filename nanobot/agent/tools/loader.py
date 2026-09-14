"""Tool discovery and registration via package scanning."""

# pyright: reportIncompatibleVariableOverride=false

from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from nanobot.agent.tools.context import RequestContext, ToolContext

_SKIP_MODULES = frozenset({
    "base", "schema", "registry", "context", "loader", "config",
    "file_state", "sandbox", "mcp", "__init__", "runtime_control",
})


class ToolLoader:
    def __init__(self, package: Any = None, *, test_classes: list[type[Tool]] | None = None):
        if package is None:
            import nanobot.agent.tools as _pkg
            package = _pkg
        self._package = package
        self._test_classes = test_classes
        self._discovered: list[type[Tool]] | None = None
        self._plugins: dict[str, type[Tool]] | None = None

    def discover(self) -> list[type[Tool]]:
        if self._test_classes is not None:
            return list(self._test_classes)
        if self._discovered is not None:
            return self._discovered
        seen: set[int] = set()
        results: list[type[Tool]] = []
        for _importer, module_name, _ispkg in pkgutil.iter_modules(self._package.__path__):
            if module_name.startswith("_") or module_name in _SKIP_MODULES:
                continue
            try:
                module = importlib.import_module(f".{module_name}", self._package.__name__)
            except Exception:
                logger.exception("Failed to import tool module: %s", module_name)
                continue
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Tool)
                    and attr is not Tool
                    and not attr_name.startswith("_")
                    and not getattr(attr, "__abstractmethods__", None)
                    and getattr(attr, "_plugin_discoverable", True)
                    and id(attr) not in seen
                ):
                    seen.add(id(attr))
                    results.append(attr)
        results.sort(key=lambda cls: cls.__name__)
        self._discovered = results
        return results

    def _discover_plugins(self) -> dict[str, type[Tool]]:
        """Discover external tool plugins registered via entry_points."""
        if self._plugins is not None:
            return self._plugins
        plugins: dict[str, type[Tool]] = {}
        try:
            eps = entry_points(group="nanobot.tools")
        except Exception:
            return plugins
        entries = list(eps)
        entry_names: set[str] = set()
        for ep in entries:
            if not isinstance(ep.name, str) or not ep.name:
                raise ValueError("Tool plugin entry point has an invalid name")
            if ep.name in entry_names:
                raise ValueError(f"Duplicate tool plugin entry point: {ep.name}")
            entry_names.add(ep.name)
        entries.sort(key=lambda entry: entry.name)
        for ep in entries:
            try:
                cls = ep.load()
                if (
                    isinstance(cls, type)
                    and issubclass(cls, Tool)
                    and not getattr(cls, "__abstractmethods__", None)
                    and getattr(cls, "_plugin_discoverable", True)
                ):
                    plugins[ep.name] = cls
            except Exception:
                logger.exception("Failed to load tool plugin: %s", ep.name)
        self._plugins = plugins
        return plugins

    def _construct_plugins(self, ctx: ToolContext) -> list[Tool]:
        """Create enabled core-scope entry-point tools behind the compatibility adapter."""
        tools: list[Tool] = []
        for entry_name, cls in sorted(self._discover_plugins().items()):
            scopes = getattr(cls, "_scopes", {"core"})
            if not isinstance(scopes, set) or not all(isinstance(scope, str) for scope in scopes):
                raise ValueError(f"Tool plugin '{entry_name}' has invalid scope metadata")
            if "core" not in scopes or not cls.enabled(ctx):
                continue
            tool = cls.create(ctx)
            if not isinstance(tool, Tool):
                raise TypeError(f"Tool plugin '{entry_name}' did not create a Tool")
            if not isinstance(tool.name, str) or not tool.name:
                raise ValueError(f"Tool plugin '{entry_name}' has an invalid tool name")
            if ToolRegistry._schema_name(tool.to_schema()) != tool.name:
                raise ValueError(f"Tool plugin '{entry_name}' schema name does not match its tool name")
            tools.append(_LegacyErrorPrefixTool(tool))
        return tools

    def load(self, ctx: ToolContext, registry: ToolRegistry, *, scope: str = "core") -> list[str]:
        from nanobot.migration_guard import require_runtime_ready
        require_runtime_ready()
        from nanobot.agent.tools.fork_factories import construct_core_tools

        if scope != "core":
            raise ValueError(f"Unsupported runtime tool scope: {scope}")
        tools = [*construct_core_tools(ctx), *self._construct_plugins(ctx)]
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)) or any(registry.has(name) for name in names):
            raise ValueError("Duplicate tool registration in foreground manifest")
        for tool in tools:
            registry.register(tool)
        return names


class _LegacyErrorPrefixTool(Tool):
    """Compatibility wrapper for external tools using the old error-string contract."""

    _plugin_discoverable = False

    def __init__(self, wrapped: Tool) -> None:
        self._wrapped = wrapped
        self._name = wrapped.name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._wrapped.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._wrapped.parameters

    def runtime_context_provider(self):
        return self._wrapped.runtime_context_provider()

    @property
    def read_only(self) -> bool:
        return self._wrapped.read_only

    @property
    def exclusive(self) -> bool:
        return self._wrapped.exclusive

    @property
    def concurrency_safe(self) -> bool:
        return self._wrapped.concurrency_safe

    @property
    def config_key(self) -> str:
        return getattr(self._wrapped, "config_key", "")

    def set_context(self, ctx: RequestContext) -> None:
        set_context = getattr(self._wrapped, "set_context", None)
        if callable(set_context):
            set_context(ctx)

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._wrapped.cast_params(params)

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        return self._wrapped.validate_params(params)

    def to_schema(self) -> dict[str, Any]:
        schema = self._wrapped.to_schema()
        if ToolRegistry._schema_name(schema) != self._name:
            raise ValueError("Tool plugin schema name does not match its registered name")
        return schema

    async def execute(self, **kwargs: Any) -> Any:
        return self._adapt_legacy_error(await self._wrapped.execute(**kwargs))

    async def execute_with_context(self, context: Any, **kwargs: Any) -> Any:
        return self._adapt_legacy_error(await self._wrapped.execute_with_context(context, **kwargs))

    @staticmethod
    def _adapt_legacy_error(result: Any) -> Any:
        if (
            isinstance(result, str)
            and not isinstance(result, ToolResult)
            and result.startswith("Error:")
        ):
            return ToolResult.error(result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)
