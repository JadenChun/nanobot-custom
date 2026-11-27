"""Tool registry for dynamic tool management."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Collection, Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Generator, cast
from weakref import ReferenceType, ref

from kohaku.agent.tools.base import (
    Tool,
    ToolDenialKind,
    ToolPreparationError,
    ToolPreparationErrorKind,
    ToolResult,
)
from kohaku.agent.tools.context import ContextAware, current_request_context
from kohaku.agent.tools.filesystem_effect import (
    FilesystemEffectCancelled,
    accepted_filesystem_effects,
    capture_filesystem_effects,
)
from kohaku.agent.tools.producer_adapters import accepted_producer_receipt
from kohaku.security.folder_grants import FolderGrantView

if TYPE_CHECKING:
    from kohaku.agent.tools.mcp import MCPGenerationLease
    from kohaku.runtime_context import RuntimeContextProvider


_HOST_ADMISSION_CHANGED = (
    "Tool '{name}' was not started: workspace access changed to Sandbox while the "
    "call was being admitted. Retry the call so it is checked against the current "
    "permissions."
)


def is_managed_mcp_tool(tool: Tool) -> bool:
    from kohaku.agent.tools.mcp import (
        MCPMetaTool,
        MCPPromptWrapper,
        MCPResourceWrapper,
        MCPToolWrapper,
    )

    return type(tool) in {MCPMetaTool, MCPToolWrapper, MCPResourceWrapper, MCPPromptWrapper}


class ToolDispatchGuard:
    """Live admission hook shared by one registry's dispatches.

    The base guard admits everything. Whether a tool call may run is decided by
    the tool policy and permission mode (``RiskyActionPolicy``), not by a name
    list here. Subclasses implement narrow lifecycle fences, such as Dream's
    operation fence or the host-process admission reservation below.
    """

    def error(self, name: str) -> str | None:
        return None

    @contextmanager
    def admit(self, name: str) -> Generator[str | None, None, None]:
        yield self.error(name)


class HostProcessAdmissionGuard(ToolDispatchGuard):
    """Fence Sandbox activation against host-process tools that are running.

    Tools that declare ``launches_host_process`` start work outside Sandbox
    (for example a browser daemon). Whether such a call may run is decided by
    the policy layer. This guard does not deny by name. It only reserves native
    admission while the grant store is not yet managed, so turning Sandbox on
    cannot race with an unconfined launch that is already in flight.
    """

    def __init__(
        self,
        grants: FolderGrantView | None,
        *,
        host_process: Callable[[str], bool],
    ) -> None:
        self._grants = grants
        self._host_process = host_process

    @contextmanager
    def admit(self, name: str) -> Generator[str | None, None, None]:
        request = current_request_context()
        grants = self._grants
        if (
            request is None
            or request.channel != "websocket"
            or grants is None
            or grants.is_managed
            or not self._host_process(name)
        ):
            yield None
            return
        reservation = grants.try_begin_native_process()
        if reservation is None:
            # Sandbox was activated after the policy assessed this call.
            yield _HOST_ADMISSION_CHANGED.format(name=name)
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

    guard: ToolDispatchGuard | None = None
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

    def set_dispatch_guard(self, guard: ToolDispatchGuard | None) -> None:
        """Install a live admission check shared by this turn's tool dispatches."""
        self._admission.guard = guard

    def dispatch_error(self, name: str) -> str | None:
        """Return a current dispatch-guard denial, if one applies."""
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
    def schema_name(schema: dict[str, Any]) -> str:
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
                name = self.schema_name(schema)
                if name.startswith("mcp_"):
                    mcp_tools.append(schema)
                else:
                    builtins.append(schema)

            builtins.sort(key=self.schema_name)
            mcp_tools.sort(key=self.schema_name)
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
        from kohaku.agent.turn import ToolOutcome

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
            from kohaku.agent.tools.artifact import ArtifactRetrievalReceipt, ReadArtifactTool

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
        from kohaku.agent.tools.artifact_handoff import ArtifactHandoffTool

        if type(tool) is ArtifactHandoffTool:
            return tool.accepts_receipt(receipt)

        from kohaku.agent.tools.delegation import (
            DelegateTaskTool,
            PlanTaskTool,
            ReviewWorkTool,
        )
        from kohaku.agent.tools.explore import ExploreTool
        from kohaku.agent.turn import TurnContext
        from kohaku.session.artifact_grants import ArtifactHandoffReceipt

        return (
            type(tool) in {DelegateTaskTool, ExploreTool, PlanTaskTool, ReviewWorkTool}
            and isinstance(context, TurnContext)
            and context.source_writer is not None
            and isinstance(receipt, ArtifactHandoffReceipt)
            and receipt.targets_parent_writer(context.source_writer)
        )

    async def execute(self, name: str, params: Any, context: Any = None, *, execution_context: Any = None) -> Any:
        """Preserve contextual outcomes and upstream preparation/legacy errors."""
        from kohaku.agent.turn import ToolOutcome

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

    def iter_tools(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def _live_registrations(self) -> dict[str, _ToolRegistration]:
        return self._registrations

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


class PolicyToolView(ToolRegistry):
    """A turn-local overlay that borrows current admitted registrations."""

    def __init__(self, root: ToolRegistry, allows: Callable[[Tool], bool]) -> None:
        super().__init__()
        self._root = root
        self._allows = allows
        self._local: set[str] = set()
        self._removed: set[str] = set()
        self._mcp_schemas: dict[str, dict[str, Any]] = {}
        self._dispatch_parent = ref(root)
        self._admission.parent = root._admission

    def _sync(self) -> None:
        available = self._root._live_registrations()
        inherited: dict[str, _ToolRegistration] = {}
        for name, registration in available.items():
            if name in self._local or name in self._removed or not self._allows(registration.tool):
                continue
            current = self._registrations.get(name)
            if (
                name in self._mcp_schemas
                and (current is None or current.parent is not registration)
                and registration.tool.to_schema() != self._mcp_schemas[name]
            ):
                continue
            inherited[name] = registration
        for name in set(self._tools) - self._local - inherited.keys():
            super().unregister(name)
        for name, registration in inherited.items():
            current = self._registrations.get(name)
            if current is not None and current.parent is registration:
                continue
            if current is not None:
                super().unregister(name)
            super().register(registration.tool)
            self._registrations[name].parent = registration
        self._withheld_names.update(available.keys() - self._tools.keys())
        self._withheld_names.update(self._root._withheld_names - self._tools.keys())

    def register(self, tool: Tool) -> None:
        self._sync()
        super().register(tool)
        self._local.add(tool.name)

    def replace(self, tool: Tool) -> None:
        self._sync()
        super().replace(tool)
        self._local.add(tool.name)

    def unregister(self, name: str) -> None:
        self._sync()
        super().unregister(name)
        self._local.discard(name)
        self._removed.add(name)

    def _live_registrations(self) -> dict[str, _ToolRegistration]:
        self._sync()
        return super()._live_registrations()

    def dispatch_error(self, name: str) -> str | None:
        self._sync()
        return super().dispatch_error(name)

    def get(self, name: str) -> Tool | None:
        self._sync()
        return super().get(name)

    def iter_tools(self) -> Iterator[Tool]:
        self._sync()
        return super().iter_tools()

    def get_definitions(self) -> list[dict[str, Any]]:
        self._sync()
        return super().get_definitions()

    def get_runtime_context_providers(self) -> list[RuntimeContextProvider]:
        self._sync()
        return super().get_runtime_context_providers()

    def capture_request_tools(self) -> CapturedToolBindings:
        self._sync()
        for tool in self._tools.values():
            if tool.name not in self._mcp_schemas and is_managed_mcp_tool(tool):
                self._mcp_schemas[tool.name] = deepcopy(tool.to_schema())
        return super().capture_request_tools()

    @property
    def tool_names(self) -> list[str]:
        self._sync()
        return super().tool_names

    def __len__(self) -> int:
        self._sync()
        return super().__len__()

    def filtered(self, names: list[str]) -> ToolRegistry:
        included = frozenset(names)
        return PolicyToolView(self, lambda tool: tool.name in included)

    def without(self, names: Collection[str]) -> PolicyToolView:
        excluded = frozenset(names)
        return PolicyToolView(self, lambda tool: tool.name not in excluded)


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
        self._registrations = {
            name: registration for name, registration in source._live_registrations().items()
            if name in included
        }
        self._tools = {name: registration.tool for name, registration in self._registrations.items()}
        self._withheld_names = source._withheld_names | (source._tools.keys() - included)
        definitions = (source.get_definitions() if isinstance(source, CapturedToolBindings)
                       else [deepcopy(tool.to_schema()) for tool in self._tools.values()])
        self._definitions = sorted(
            (definition for definition in definitions if self.schema_name(definition) in self._tools),
            key=lambda definition: (self.schema_name(definition).startswith("mcp_"),
                                    self.schema_name(definition)),
        )
        self._schemas: dict[str, dict[str, Any]] = {
            name: deepcopy(tool.parameters) for name, tool in self._tools.items()
        }
        if isinstance(source, CapturedToolBindings):
            self._schemas = {name: deepcopy(source._schemas[name]) for name in self._tools}
        from kohaku.agent.tools.mcp import mcp_generation_leases

        leases = {
            lease
            for tool in self._tools.values()
            for lease in mcp_generation_leases(tool)
        }
        self._leases: tuple[MCPGenerationLease, ...] = tuple(
            lease for lease in leases if lease.acquire_sampled()
        )

    def get_definitions(self) -> list[dict[str, Any]]:
        return deepcopy(self._definitions)

    def refresh_reconnected_tools(self) -> None:
        """Keep sampled schemas and bind equivalent successors after a transport reconnect."""
        source = self._source
        if source is None:
            return
        from kohaku.agent.tools.mcp import MCPMetaTool, MCPToolWrapper, mcp_generation_leases

        current = source._live_registrations()
        leases = set(self._leases)
        for name, old_tool in self._tools.items():
            next_registration = current.get(name)
            if (
                next_registration is None
                or not isinstance(old_tool, (MCPToolWrapper, MCPMetaTool))
                or not old_tool.accepts_reconnected_tool(next_registration.tool)
                or next_registration.error() is not None
            ):
                continue
            next_leases = mcp_generation_leases(next_registration.tool)
            if not next_leases:
                continue
            acquired: list[MCPGenerationLease] = []
            ok = True
            for lease in next_leases:
                if lease in leases:
                    continue
                if not lease.acquire_sampled():
                    ok = False
                    break
                acquired.append(lease)
            if not ok:
                for lease in acquired:
                    lease.release_sampled()
                continue
            if acquired:
                self._leases = (*self._leases, *acquired)
                leases.update(acquired)
            self._tools[name] = next_registration.tool
            self._registrations[name] = next_registration

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
        from kohaku.agent.tools.mcp import mcp_binding_error

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
