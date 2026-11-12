"""Tool registry for dynamic tool management."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Collection, Iterable
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Generator, cast
from weakref import ReferenceType, ref

from nanobot.agent.tools.base import (
    Tool,
    ToolDenialKind,
    ToolPreparationError,
    ToolPreparationErrorKind,
    ToolResult,
)
from nanobot.agent.tools.context import ContextAware, current_request_context
from nanobot.agent.tools.filesystem_effect import (
    FilesystemEffectCancelled,
    accepted_filesystem_effects,
    capture_filesystem_effects,
)
from nanobot.agent.tools.producer_adapters import accepted_producer_receipt
from nanobot.security.folder_grants import FolderGrantView

if TYPE_CHECKING:
    from nanobot.agent.tools.mcp import MCPGenerationLease
    from nanobot.runtime_context import RuntimeContextProvider


MANAGED_WEBUI_TOOL_NAMES = frozenset({
    "read_file", "write_file", "edit_file", "list_dir", "exec",
    "exec_session", "list_exec_sessions", "web_search", "web_fetch",
    "message", "cron", "generate_image", "repository_catalog",
    "save_knowledge", "select_workspace", "plan_task", "delegate_task",
    "request_folder_access", "review_work", "explore",
})

_MANAGED_CAPABILITY_DENIAL = (
    "Tool '{name}' is unavailable with managed folder access. This unconfined "
    "host capability is not part of the configured managed tool set. Do not "
    "retry the denied tool or reach the same external capability through another tool."
)


def managed_webui_tool_names(tools: Iterable[Tool]) -> frozenset[str]:
    """Return built-in managed tools plus the user's configured MCP capabilities.

    MCP servers are opt-in configuration, not ambient host work. The MCP
    connection owns the server lifetime; each call is a configured capability
    rather than a new unconfined process launch.
    """
    names = set(MANAGED_WEBUI_TOOL_NAMES)
    from nanobot.agent.tools.mcp import (
        MCPPromptWrapper,
        MCPResourceWrapper,
        MCPToolWrapper,
    )

    managed_mcp_types = {MCPToolWrapper, MCPResourceWrapper, MCPPromptWrapper}
    for tool in tools:
        if type(tool) in managed_mcp_types:
            names.add(tool.name)
    return frozenset(names)


class ManagedToolDispatchGuard:
    """Deny stale unconfined WebUI tools and serialize their legacy admission."""

    def __init__(
        self,
        grants: FolderGrantView | None,
        *,
        allowed_names: Collection[str] | None = None,
    ) -> None:
        self._grants = grants
        self._allowed_names = frozenset(
            MANAGED_WEBUI_TOOL_NAMES if allowed_names is None else allowed_names
        )

    def _managed_context(self) -> bool:
        request = current_request_context()
        grants = self._grants
        return bool(
            request is not None
            and request.channel == "websocket"
            and grants is not None
            and grants.is_managed
        )

    def error(self, name: str) -> str | None:
        if not self._managed_context() or name in self._allowed_names:
            return None
        return _MANAGED_CAPABILITY_DENIAL.format(name=name)

    @contextmanager
    def admit(self, name: str) -> Generator[str | None, None, None]:
        request = current_request_context()
        grants = self._grants
        if (
            request is None
            or request.channel != "websocket"
            or grants is None
            or name in self._allowed_names
        ):
            yield None
            return
        if self._managed_context():
            yield self.error(name) or _MANAGED_CAPABILITY_DENIAL.format(name=name)
            return
        # Reserve before first activation even when the store is not managed
        # yet; the grant change may activate confinement while this call runs.
        reservation = grants.try_begin_native_process()
        if reservation is None:
            yield _MANAGED_CAPABILITY_DENIAL.format(name=name)
            return
        try:
            yield None
        finally:
            grants.end_native_process(reservation)


def is_tool_error_result(result: Any) -> bool:
    return isinstance(result, ToolResult) and result.is_error


@dataclass
class _ToolAdmission:
    """Live guard state that can outlive an unowned registry overlay's issuer."""

    guard: ManagedToolDispatchGuard | None = None
    parent: _ToolAdmission | None = None

    def error(self, name: str) -> str | None:
        if self.guard is not None and (error := self.guard.error(name)) is not None:
            return error
        return self.parent.error(name) if self.parent is not None else None

    @contextmanager
    def admit(self, name: str) -> Generator[str | None, None, None]:
        with ExitStack() as admissions:
            if self.parent is not None:
                error = admissions.enter_context(self.parent.admit(name))
                if error is not None:
                    yield error
                    return
            yield admissions.enter_context(self.guard.admit(name)) if self.guard is not None else None


@dataclass
class _ToolRegistration:
    """One registration occurrence; a later registration cannot revive it."""

    tool: Tool
    admission: _ToolAdmission
    parent: _ToolRegistration | None = None
    revoked: bool = False

    def error(self) -> str | None:
        if self.revoked:
            return ToolPreparationError(
                f"Tool '{self.tool.name}' call refused: its captured registration was revoked.",
                kind=ToolPreparationErrorKind.WITHHELD_TOOL,
            )
        if self.parent is not None:
            error = self.parent.error()
            if error is not None:
                return error
        return self.admission.error(self.tool.name)


class _CapturedParameterSchema(Tool):
    """Use the base caster/validator with detached request parameters only."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    @property
    def name(self) -> str:
        return "captured_parameters"

    @property
    def description(self) -> str:
        return "Internal parameter preparation"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._schema

    async def execute(self, **kwargs: Any) -> Any:
        raise RuntimeError("A captured parameter schema cannot execute a tool")


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._registrations: dict[str, _ToolRegistration] = {}
        self._withheld_names: set[str] = set()
        self._cached_definitions: list[dict[str, Any]] | None = None
        self._admission = _ToolAdmission()
        self._dispatch_parent: ReferenceType[ToolRegistry] | None = None

    def set_dispatch_guard(self, guard: ManagedToolDispatchGuard | None) -> None:
        """Install a live admission check shared by this turn's tool dispatches."""
        self._admission.guard = guard

    def dispatch_error(self, name: str) -> str | None:
        """Return a current managed-capability denial, if one applies."""
        registration = self._registrations.get(name)
        if registration is not None and (error := registration.error()) is not None:
            return error
        return self._admission.error(name)

    def dispatch_guard_error(self, name: str) -> str | None:
        """Read this registry owner's current guard without resolving a tool."""
        if self._admission.guard is None:
            return None
        return self._admission.guard.error(name)

    @contextmanager
    def dispatch_admission(self, name: str) -> Generator[str | None, None, None]:
        """Hold first-activation admission for a native WebUI tool while it runs."""
        if (error := self.dispatch_error(name)) is not None:
            yield error
            return
        with ExitStack() as admissions:
            parent = self._dispatch_parent() if self._dispatch_parent is not None else None
            if parent is not None:
                error = admissions.enter_context(parent.dispatch_admission(name))
                if error is not None:
                    yield error
                    return
            elif self._admission.parent is not None:
                error = admissions.enter_context(self._admission.parent.admit(name))
                if error is not None:
                    yield error
                    return
            guard = self._admission.guard
            error = admissions.enter_context(guard.admit(name)) if guard is not None else None
            yield error

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool registration: {tool.name}")
        self._tools[tool.name] = tool
        self._registrations[tool.name] = _ToolRegistration(tool, self._admission)
        self._withheld_names.discard(tool.name)
        self._cached_definitions = None

    def replace(self, tool: Tool) -> None:
        """Explicitly replace an existing registration during validated reload."""
        if tool.name not in self._tools:
            raise ValueError(f"Cannot replace unregistered tool: {tool.name}")
        self._registrations[tool.name].revoked = True
        self._tools[tool.name] = tool
        self._registrations[tool.name] = _ToolRegistration(tool, self._admission)
        self._withheld_names.discard(tool.name)
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        if self._tools.pop(name, None) is not None:
            self._registrations.pop(name).revoked = True
            self._withheld_names.add(name)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_runtime_context_providers(self) -> list[RuntimeContextProvider]:
        """Return tool-owned providers in stable tool-name order."""
        providers: list[RuntimeContextProvider] = []
        for name in sorted(self._tools):
            provider = self._tools[name].runtime_context_provider()
            if provider is not None:
                providers.append(provider)
        return providers

    @staticmethod
    def _lookup_key(name: str) -> str:
        """Normalize names for suggestions only; never for execution."""
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def _suggest_name(self, name: str) -> str | None:
        key = self._lookup_key(str(name or ""))
        if not key:
            return None
        matches = [
            registered
            for registered in self._tools
            if self._lookup_key(registered) == key
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return self.get(name) is not None

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        """Extract a normalized tool name from either OpenAI or flat schemas."""
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = cast(dict[str, Any], fn).get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions with stable ordering for cache-friendly prompts.

        Built-in tools are sorted first as a stable prefix, then MCP tools are
        sorted and appended. The result is cached until the next
        register/unregister call.
        """
        if self._cached_definitions is None:
            definitions = [deepcopy(tool.to_schema()) for tool in self._tools.values()]
            builtins: list[dict[str, Any]] = []
            mcp_tools: list[dict[str, Any]] = []
            for schema in definitions:
                name = self._schema_name(schema)
                if name.startswith("mcp_"):
                    mcp_tools.append(schema)
                else:
                    builtins.append(schema)

            builtins.sort(key=self._schema_name)
            mcp_tools.sort(key=self._schema_name)
            self._cached_definitions = builtins + mcp_tools

        return deepcopy(self._cached_definitions)

    def capture_request_tools(self) -> CapturedToolBindings:
        """Synchronously own the definitions, registrations and generation resources."""
        return CapturedToolBindings(self, self.tool_names)

    def borrow_filtered(self, names: Collection[str]) -> CapturedToolBindings:
        """Take independent, narrowed ownership before the current owner releases."""
        return CapturedToolBindings(self, names)

    def prepare_call(
        self,
        name: str,
        params: Any,
    ) -> tuple[Tool | None, Any, str | None]:
        """Resolve, cast, and validate one tool call."""
        tool = self.get(name)
        dispatch_error = self.dispatch_error(name)
        if not tool:
            suggestion = self._suggest_name(str(name))
            hint = f" Did you mean '{suggestion}'? Tool names must match exactly." if suggestion else ""
            if name in self._withheld_names and dispatch_error is not None:
                return None, params, ToolPreparationError(
                    dispatch_error,
                    kind=ToolPreparationErrorKind.MANAGED_UNAVAILABLE,
                )
            kind = (
                ToolPreparationErrorKind.WITHHELD_TOOL
                if name in self._withheld_names
                else ToolPreparationErrorKind.UNKNOWN_TOOL
            )
            return None, params, ToolPreparationError(
                f"Error: Tool '{name}' not found.{hint} Available: {', '.join(self.tool_names)}",
                kind=kind,
            )
        if dispatch_error is not None:
            if isinstance(dispatch_error, ToolPreparationError):
                return None, params, dispatch_error
            return None, params, ToolPreparationError(
                dispatch_error,
                kind=ToolPreparationErrorKind.MANAGED_UNAVAILABLE,
            )
        # Compatibility for external tools that still implement the legacy
        # setter protocol. Built-ins read the authoritative ContextVar
        # directly and never copy routing state.
        if isinstance(tool, ContextAware) and (ctx := current_request_context()) is not None:
            tool.set_context(ctx)

        params = self._coerce_params(tool, params, schema=self._parameter_schema(tool))
        if not isinstance(params, dict):
            return tool, params, (
                ToolResult.error(
                    f"Error: Tool '{name}' parameters must be a JSON object, got "
                    f"{type(params).__name__}. Use named parameters like "
                    'tool_name(param1="value1", param2="value2") matching the tool schema.'
                )
            )

        cast_params, errors = self._cast_and_validate(tool, cast(dict[str, Any], params))
        if errors:
            return tool, cast_params, (
                ToolResult.error(f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors))
            )
        return tool, cast_params, None

    def _cast_and_validate(self, tool: Tool, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        cast_params = tool.cast_params(params)
        return cast_params, tool.validate_params(cast_params)

    def _parameter_schema(self, tool: Tool) -> dict[str, Any]:
        return tool.parameters

    @classmethod
    def _coerce_argument_value(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped:
            return {}

        if not stripped.startswith(("{", "[")):
            return value

        try:
            parsed = json.loads(stripped)
        except Exception:
            return value

        return parsed

    @classmethod
    def _coerce_params(cls, tool: Tool, params: Any, *, schema: dict[str, Any] | None = None) -> Any:
        params = cls._coerce_argument_value(params)
        return cls._unwrap_arguments_payload(tool, params, schema=schema)

    @classmethod
    def _unwrap_arguments_payload(cls, tool: Tool, params: Any, *, schema: dict[str, Any] | None = None) -> Any:
        if not isinstance(params, dict):
            return params
        arguments_payload = cast(dict[str, Any], params)
        if set(arguments_payload) != {"arguments"}:
            return arguments_payload
        properties = (tool.parameters if schema is None else schema).get("properties", {})
        if isinstance(properties, dict) and "arguments" in properties:
            return arguments_payload
        return cls._coerce_argument_value(arguments_payload.get("arguments"))

    @staticmethod
    async def execute_prepared(tool: Tool, params: dict[str, Any], context: Any = None) -> Any:
        """Execute a validated call once, retaining terminal fork metadata."""
        from nanobot.agent.turn import ToolOutcome

        with capture_filesystem_effects(tool) as capture:
            try:
                result = (await tool.execute_with_context(context, **params)
                          if context is not None else await tool.execute(**params))
            except asyncio.CancelledError as exc:
                if capture is not None and capture.attempts:
                    raise FilesystemEffectCancelled(str(exc), capture.failed()) from exc
                if isinstance(exc, FilesystemEffectCancelled):
                    # Plugin envelopes and cancellation text cannot create effect evidence.
                    raise asyncio.CancelledError(str(exc)) from exc
                raise
            except Exception as exc:
                if capture is None:
                    raise
                result = ToolOutcome(content=ToolResult.error(f"Error executing {tool.name}: {exc}"))
        if isinstance(result, ToolOutcome):
            outcome = result
        elif isinstance(result, ToolResult) and result.media_artifacts:
            outcome = ToolOutcome(content=result, media_artifacts=result.media_artifacts)
        else:
            outcome = ToolOutcome(content=result)
        effects = None
        if capture is not None and (is_tool_error_result(outcome.content)
                                    or outcome.stop_reason == "tool_error"
                                    or outcome.filesystem_effects):
            effects = capture.failed()
        if not accepted_filesystem_effects(tool, effects):
            effects = None
        outcome = replace(outcome, filesystem_effects=effects)
        if outcome.recovery_kind is not None:
            # Recovery classification is owned by the registry/executor, never
            # by a tool's returned envelope.
            outcome = replace(outcome, recovery_kind=None)
        if outcome.producer_receipt is not None:
            if not accepted_producer_receipt(tool, outcome.content, outcome.producer_receipt):
                # Strip only untrusted producer authority.  Other independently
                # validated terminal, retrieval, and handoff behavior must retain
                # its existing handling below.
                outcome = replace(outcome, producer_receipt=None)

        if outcome.retrieval_receipt is not None:
            from nanobot.agent.tools.artifact import ArtifactRetrievalReceipt, ReadArtifactTool

            receipt = outcome.retrieval_receipt
            if (type(tool) is not ReadArtifactTool or type(receipt) is not ArtifactRetrievalReceipt
                    or not receipt.matches(outcome.content)):
                outcome = replace(outcome, retrieval_receipt=None)
        if outcome.handoff_receipt is not None:
            receipt = outcome.handoff_receipt
            if not ToolRegistry._accepts_handoff_receipt(tool, receipt, context):
                outcome = replace(outcome, handoff_receipt=None)
        # Tool output is opaque text.  In particular, command-like tools can
        # successfully report a payload that starts with ``Error``.  Only the
        # structured result marker (or an explicit ToolOutcome stop state
        # supplied by the tool) establishes terminal semantics.
        if is_tool_error_result(outcome.content):
            return replace(outcome, stop_reason=outcome.stop_reason or "tool_error")
        return outcome

    @staticmethod
    def _accepts_handoff_receipt(tool: Tool, receipt: object, context: Any) -> bool:
        """Retain receipts only from the child selector or concrete foreground wrappers."""
        from nanobot.agent.tools.artifact_handoff import ArtifactHandoffTool

        if type(tool) is ArtifactHandoffTool:
            return tool.accepts_receipt(receipt)

        from nanobot.agent.tools.delegation import (
            DelegateTaskTool,
            PlanTaskTool,
            ReviewWorkTool,
        )
        from nanobot.agent.tools.explore import ExploreTool
        from nanobot.agent.turn import TurnContext
        from nanobot.session.artifact_grants import ArtifactHandoffReceipt

        return (
            type(tool) in {DelegateTaskTool, ExploreTool, PlanTaskTool, ReviewWorkTool}
            and isinstance(context, TurnContext)
            and context.source_writer is not None
            and isinstance(receipt, ArtifactHandoffReceipt)
            and receipt.targets_parent_writer(context.source_writer)
        )

    async def execute(self, name: str, params: Any, context: Any = None, *, execution_context: Any = None) -> Any:
        """Preserve contextual outcomes and upstream preparation/legacy errors."""
        from nanobot.agent.turn import ToolOutcome

        context = context if context is not None else execution_context
        hint = "\n\n[Analyze the error above and try a different approach.]"
        try:
            tool, params, error = self.prepare_call(name, params)
            if error:
                recovery_kind = (
                    ToolDenialKind.MANAGED_UNAVAILABLE
                    if isinstance(error, ToolPreparationError)
                    and error.kind is ToolPreparationErrorKind.MANAGED_UNAVAILABLE
                    else None
                )
                outcome = ToolOutcome(
                    content=ToolResult.error(str(error) + hint),
                    stop_reason="tool_error",
                    recovery_kind=recovery_kind,
                )
            else:
                assert tool is not None
                with self.dispatch_admission(name) as dispatch_error:
                    if dispatch_error is not None:
                        outcome = ToolOutcome(
                            content=ToolResult.error(dispatch_error + hint),
                            stop_reason="tool_error",
                            recovery_kind=ToolDenialKind.MANAGED_UNAVAILABLE,
                        )
                    else:
                        outcome = await self.execute_prepared(tool, params, context)
                        if (outcome.stop_reason == "tool_error" and isinstance(outcome.content, str)
                                and not outcome.filesystem_effects):
                            outcome = replace(outcome, content=ToolResult.error(outcome.content + hint))
        except Exception as exc:
            outcome = ToolOutcome(content=ToolResult.error(f"Error executing {name}: {exc}" + hint),
                                  stop_reason="tool_error")
        return outcome if context is not None or outcome.filesystem_effects else outcome.content

    async def execute_with_context(self, name: str, params: Any, context: Any) -> Any:
        return await self.execute(name, params, context=context)

    def iter_tools(self):
        return iter(self._tools.values())

    def _filtered_dispatch_parent(self) -> ToolRegistry:
        return self

    def filtered(self, names: list[str]) -> ToolRegistry:
        registry = ToolRegistry()
        # A filtered invocation overlay remains mutable for its own artifact
        # reader. Inherit live admission, including native reservations, while
        # explicit borrow_filtered owns any independent resource lifetime.
        parent = self._filtered_dispatch_parent()
        registry._dispatch_parent = ref(parent)
        registry._admission.parent = parent._admission
        included = set(names)
        registry._withheld_names = self._withheld_names | (self._tools.keys() - included)
        for name in names:
            if name in self._tools:
                registry.register(self._tools[name])
                registry._registrations[name].parent = self._registrations[name]
        return registry

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return self.has(name)


class CapturedToolBindings(ToolRegistry):
    """Owned request/borrower view of exact registrations, with live revocation.

    Definitions are copied at capture and on access. Resource acquisition/release
    never awaits, so reload and cancellation cannot split ownership on the loop.
    """

    def __init__(self, source: ToolRegistry, names: Collection[str]) -> None:
        super().__init__()
        if isinstance(source, CapturedToolBindings) and source._released:
            raise RuntimeError("Cannot borrow released tool bindings")
        origin = source._source if isinstance(source, CapturedToolBindings) else source
        assert origin is not None  # A released source was rejected above.
        self._source: ToolRegistry | None = origin
        # Unowned filters retain live admission state, not the issuer registry.
        # Owned requests retain exact available issuers only until release.
        origins = [origin]
        while origins[-1]._dispatch_parent is not None:
            parent = origins[-1]._dispatch_parent()
            if parent is None:
                break
            origins.append(parent)
        self._origin_registries = tuple(origins)
        self._released = False
        included = set(names)
        self._tools = {name: tool for name, tool in source._tools.items() if name in included}
        self._registrations = {name: source._registrations[name] for name in self._tools}
        self._withheld_names = source._withheld_names | (source._tools.keys() - included)
        definitions = (source.get_definitions() if isinstance(source, CapturedToolBindings)
                       else [deepcopy(tool.to_schema()) for tool in self._tools.values()])
        self._definitions = sorted(
            (definition for definition in definitions if self._schema_name(definition) in self._tools),
            key=lambda definition: (self._schema_name(definition).startswith("mcp_"),
                                    self._schema_name(definition)),
        )
        self._schemas: dict[str, dict[str, Any]] = {
            name: deepcopy(tool.parameters) for name, tool in self._tools.items()
        }
        if isinstance(source, CapturedToolBindings):
            self._schemas = {name: deepcopy(source._schemas[name]) for name in self._tools}
        from nanobot.agent.tools.mcp import mcp_generation_lease

        leases = {lease for tool in self._tools.values()
                  if (lease := mcp_generation_lease(tool)) is not None}
        self._leases: tuple[MCPGenerationLease, ...] = tuple(
            lease for lease in leases if lease.acquire_sampled()
        )

    def get_definitions(self) -> list[dict[str, Any]]:
        return deepcopy(self._definitions)

    def dispatch_error(self, name: str) -> str | None:
        if self._released:
            return ToolPreparationError(
                f"Tool '{name}' call refused: its captured bindings were released.",
                kind=ToolPreparationErrorKind.WITHHELD_TOOL,
            )
        error = super().dispatch_error(name)
        if error is not None:
            return error
        assert self._source is not None  # Release is checked before using the source.
        if (error := self._source.dispatch_error(name)) is not None:
            return error
        from nanobot.agent.tools.mcp import mcp_binding_error

        tool = self.get(name)
        if tool is not None and (error := mcp_binding_error(tool)) is not None:
            return ToolPreparationError(error, kind=ToolPreparationErrorKind.WITHHELD_TOOL)
        return None

    @contextmanager
    def dispatch_admission(self, name: str) -> Generator[str | None, None, None]:
        if (error := self.dispatch_error(name)) is not None:
            yield error
            return
        # Preserve the originating view's live admission (including managed
        # first-activation reservations); never resolve its current tool.
        assert self._source is not None  # dispatch_error refused a released capture.
        with self._source.dispatch_admission(name) as error:
            yield error

    def _cast_and_validate(self, tool: Tool, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        schema = self._schemas[tool.name] or {}
        preparation = _CapturedParameterSchema(schema)
        cast_params = (preparation.cast_params(params)
                       if type(tool).cast_params is Tool.cast_params
                       else tool.cast_params(params))
        if type(tool).validate_params is not Tool.validate_params:
            return cast_params, tool.validate_params(cast_params)
        return cast_params, preparation.validate_params(cast_params)

    def _parameter_schema(self, tool: Tool) -> dict[str, Any]:
        return self._schemas[tool.name]

    def _filtered_dispatch_parent(self) -> ToolRegistry:
        if self._source is None:
            raise RuntimeError("Cannot filter released tool bindings")
        return self._source

    def register(self, tool: Tool) -> None:
        raise TypeError("Captured tool bindings cannot be changed")

    def replace(self, tool: Tool) -> None:
        raise TypeError("Captured tool bindings cannot be changed")

    def unregister(self, name: str) -> None:
        raise TypeError("Captured tool bindings cannot be changed")

    def release(self) -> None:
        """Release every sampled hold exactly once, even on a cancelled task."""
        if self._released:
            return
        self._released = True
        for lease in self._leases:
            lease.release_sampled()
        self._source = None
        self._origin_registries = ()
