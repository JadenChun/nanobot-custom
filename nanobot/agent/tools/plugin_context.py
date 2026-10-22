"""Bounded, data-only construction context for installed tool plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from nanobot.agent.tools.context import ExternalToolContext
from nanobot.config.schema import ToolsConfig
from nanobot.security.workspace_access import WorkspaceSandboxStatus

if TYPE_CHECKING:
    from nanobot.agent.tools.context import ToolContext


CONFIGURATION_STAGE = "configuration"


class ExternalContextError(ValueError):
    """A sanitized external construction-context error."""

    def __init__(self, stage: str = CONFIGURATION_STAGE) -> None:
        if stage != CONFIGURATION_STAGE:
            stage = CONFIGURATION_STAGE
        super().__init__(f"External tool setup failed at {stage}")
        self.stage = stage


@dataclass(frozen=True, slots=True)
class ExternalContextSnapshot:
    """Immutable serialized values used to build one plugin context at a time."""

    config_json: str
    workspace: str
    timezone: str
    workspace_sandbox: tuple[bool, str, str, bool, str, str, str] | None = None


def _checked_sandbox(
    sandbox: WorkspaceSandboxStatus | None,
) -> tuple[bool, str, str, bool, str, str, str] | None:
    if sandbox is None:
        return None
    if type(sandbox) is not WorkspaceSandboxStatus:
        raise ExternalContextError()

    try:
        values: tuple[Any, ...] = (
            sandbox.restrict_to_workspace,
            sandbox.workspace_root,
            sandbox.level,
            sandbox.enforced,
            sandbox.provider,
            sandbox.provider_label,
            sandbox.summary,
        )
    except Exception:
        raise ExternalContextError() from None
    expected = (bool, str, str, bool, str, str, str)
    if any(type(value) is not value_type for value, value_type in zip(values, expected)):
        raise ExternalContextError()
    return cast(tuple[bool, str, str, bool, str, str, str], values)


def snapshot_external_context(ctx: ToolContext) -> ExternalContextSnapshot:
    """Validate and serialize the small set of values plugins may inspect."""
    try:
        config = ctx.config
    except Exception:
        raise ExternalContextError() from None

    if type(config) is not ToolsConfig:
        raise ExternalContextError()

    try:
        workspace = ctx.workspace
        timezone = ctx.timezone
        sandbox = ctx.workspace_sandbox
    except Exception:
        raise ExternalContextError() from None
    if type(workspace) is not str or type(timezone) is not str:
        raise ExternalContextError()
    sandbox_values = _checked_sandbox(sandbox)

    try:
        config_json = ToolsConfig.model_dump_json(
            config,
            by_alias=True,
            warnings="error",
            serialize_as_any=False,
        )
        if type(config_json) is not str:
            raise ValueError
        # Preflight the exact wire representation, then discard this object.
        preflight = ToolsConfig.model_validate_json(config_json, strict=True)
        if type(preflight) is not ToolsConfig:
            raise ValueError
    except Exception:
        raise ExternalContextError() from None

    return ExternalContextSnapshot(
        config_json=config_json,
        workspace=workspace,
        timezone=timezone,
        workspace_sandbox=sandbox_values,
    )


def _snapshot_values(
    snapshot: ExternalContextSnapshot,
) -> tuple[str, str, str, tuple[bool, str, str, bool, str, str, str] | None]:
    if type(snapshot) is not ExternalContextSnapshot:
        raise ExternalContextError()
    config_json = snapshot.config_json
    workspace = snapshot.workspace
    timezone = snapshot.timezone
    sandbox_values = snapshot.workspace_sandbox
    if type(config_json) is not str or type(workspace) is not str or type(timezone) is not str:
        raise ExternalContextError()
    if sandbox_values is not None:
        if type(sandbox_values) is not tuple or len(sandbox_values) != 7:
            raise ExternalContextError()
        expected = (bool, str, str, bool, str, str, str)
        if not all(type(value) is value_type for value, value_type in zip(sandbox_values, expected)):
            raise ExternalContextError()
    return config_json, workspace, timezone, sandbox_values


def new_external_context(snapshot: ExternalContextSnapshot) -> ExternalToolContext:
    """Build a fresh plugin context from the immutable snapshot."""
    config_json, workspace, timezone, sandbox_values = _snapshot_values(snapshot)
    try:
        config = ToolsConfig.model_validate_json(config_json, strict=True)
        if type(config) is not ToolsConfig:
            raise ValueError
    except Exception:
        raise ExternalContextError() from None

    sandbox = None
    if sandbox_values is not None:
        (
            restrict_to_workspace,
            workspace_root,
            level,
            enforced,
            provider,
            provider_label,
            summary,
        ) = sandbox_values
        sandbox = WorkspaceSandboxStatus(
            restrict_to_workspace=restrict_to_workspace,
            workspace_root=workspace_root,
            level=level,
            enforced=enforced,
            provider=provider,
            provider_label=provider_label,
            summary=summary,
        )
    return ExternalToolContext(
        config=config,
        workspace=workspace,
        timezone=timezone,
        workspace_sandbox=sandbox,
    )


__all__ = [
    "CONFIGURATION_STAGE",
    "ExternalContextError",
    "ExternalContextSnapshot",
    "new_external_context",
    "snapshot_external_context",
]
