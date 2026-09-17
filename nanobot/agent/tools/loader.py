"""Tool discovery and registration via package scanning."""

# pyright: reportIncompatibleVariableOverride=false

from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.plugin_context import (
    ExternalContextError,
    ExternalContextSnapshot,
    new_external_context,
    snapshot_external_context,
)
from nanobot.agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from nanobot.agent.tools.context import RequestContext, ToolContext


_EXTERNAL_SETUP_STAGES = frozenset({
    "discovery", "configuration", "scope", "enablement", "construction", "name", "schema",
})
_RESERVED_EXTERNAL_TOOL_NAMES = frozenset({"spawn", "spawn_pipeline"})
RESERVED_EXTERNAL_TOOL_NAMES = _RESERVED_EXTERNAL_TOOL_NAMES

_SKIP_MODULES = frozenset({
    "base", "schema", "registry", "context", "loader", "config",
    "file_state", "sandbox", "mcp", "plugin_context", "__init__", "runtime_control",
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
            entries = list(eps)
            named_entries: list[tuple[str, Any]] = []
            entry_names: set[str] = set()
            for ep in entries:
                name = ep.name
                if type(name) is not str or not name or name in _RESERVED_EXTERNAL_TOOL_NAMES:
                    raise ValueError
                if name in entry_names:
                    raise ValueError
                entry_names.add(name)
                named_entries.append((name, ep))
            named_entries.sort(key=lambda item: item[0])
        except Exception:
            self._log_external_setup_failure("discovery")
            raise self._external_setup_error("discovery") from None

        for entry_name, ep in named_entries:
            try:
                cls = ep.load()
                if (
                    isinstance(cls, type)
                    and issubclass(cls, Tool)
                    and not getattr(cls, "__abstractmethods__", None)
                    and getattr(cls, "_plugin_discoverable", True)
                ):
                    plugins[entry_name] = cls
            except Exception:
                self._log_external_setup_failure("discovery")
                raise self._external_setup_error("discovery") from None
        self._plugins = plugins
        return plugins

    @staticmethod
    def _external_setup_error(stage: str) -> ValueError:
        if stage not in _EXTERNAL_SETUP_STAGES:
            stage = "construction"
        return ValueError(f"External tool setup failed at {stage}")

    @staticmethod
    def _log_external_setup_failure(stage: str) -> None:
        if stage not in _EXTERNAL_SETUP_STAGES:
            stage = "construction"
        logger.warning("External tool setup failed at {}", stage)

    def _construct_plugins(self, snapshot: ExternalContextSnapshot, *, scope: str) -> list[Tool]:
        """Create enabled core-scope entry-point tools behind the compatibility adapter."""
        tools: list[Tool] = []
        try:
            discovered = self._discover_plugins()
            entries = sorted(discovered.items())
        except Exception:
            self._log_external_setup_failure("discovery")
            raise self._external_setup_error("discovery") from None
        for _entry_name, cls in entries:
            try:
                scopes_value: object = getattr(cls, "_scopes", {"core"})
                if type(scopes_value) is not set:
                    raise ValueError
                scope_values = cast(set[object], scopes_value)
                if not all(type(value) is str for value in scope_values):
                    raise ValueError
                scopes = cast(set[str], scope_values)
            except Exception:
                self._log_external_setup_failure("scope")
                raise self._external_setup_error("scope") from None
            if scope not in scopes:
                continue
            try:
                plugin_ctx = new_external_context(snapshot)
            except ExternalContextError:
                self._log_external_setup_failure("configuration")
                raise self._external_setup_error("configuration") from None
            except Exception:
                self._log_external_setup_failure("configuration")
                raise self._external_setup_error("configuration") from None
            try:
                enabled = cast(Any, cls).enabled(plugin_ctx)
                if type(enabled) is not bool:
                    raise ValueError
            except Exception:
                self._log_external_setup_failure("enablement")
                raise self._external_setup_error("enablement") from None
            if not enabled:
                continue
            try:
                tool = cast(Any, cls).create(plugin_ctx)
                if not isinstance(tool, Tool):
                    raise TypeError
            except Exception:
                self._log_external_setup_failure("construction")
                raise self._external_setup_error("construction") from None
            try:
                tool_name = tool.name
                if type(tool_name) is not str or not tool_name:
                    raise ValueError
                if tool_name in _RESERVED_EXTERNAL_TOOL_NAMES:
                    raise ValueError
            except Exception:
                self._log_external_setup_failure("name")
                raise self._external_setup_error("name") from None
            try:
                schema_value: Any = tool.to_schema()
                if not isinstance(schema_value, dict):
                    raise ValueError
                schema = cast(dict[str, Any], schema_value)
                if ToolRegistry._schema_name(  # pyright: ignore[reportPrivateUsage]
                    schema
                ) != tool_name:
                    raise ValueError
            except Exception:
                self._log_external_setup_failure("schema")
                raise self._external_setup_error("schema") from None
            tools.append(_LegacyErrorPrefixTool(tool, tool_name))
        return tools

    def load(self, ctx: ToolContext, registry: ToolRegistry, *, scope: str = "core") -> list[str]:
        from nanobot.migration_guard import require_runtime_ready
        require_runtime_ready()
        from nanobot.agent.tools.fork_factories import construct_core_tools

        if scope != "core":
            raise ValueError("Unsupported runtime tool scope")
        try:
            snapshot = snapshot_external_context(ctx)
        except Exception:
            self._log_external_setup_failure("configuration")
            raise self._external_setup_error("configuration") from None
        tools = [*construct_core_tools(ctx), *self._construct_plugins(snapshot, scope=scope)]
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)) or any(registry.has(name) for name in names):
            raise ValueError("Duplicate tool registration in foreground manifest")
        for tool in tools:
            registry.register(tool)
        return names


class _LegacyErrorPrefixTool(Tool):
    """Compatibility wrapper for external tools using the old error-string contract."""

    _plugin_discoverable = False

    def __init__(self, wrapped: Tool, name: str) -> None:
        self._wrapped = wrapped
        self._name = name

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
        try:
            schema_value: Any = self._wrapped.to_schema()
            if not isinstance(schema_value, dict):
                raise ValueError
            schema = cast(dict[str, Any], schema_value)
            valid = ToolRegistry._schema_name(  # pyright: ignore[reportPrivateUsage]
                schema
            ) == self._name
        except Exception:
            schema = None
            valid = False
        if not valid:
            raise ValueError("Tool plugin schema name does not match its registered name")
        assert schema is not None
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
