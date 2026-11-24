"""Persistent identity and public projection of a direct chat harness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

HARNESS_BINDING_KEY = "harness_binding"
EXTERNAL_ACTIVE_RUN_KEY = "_external_active_run"
EXTERNAL_RECOVERY_KEY = "_external_recovery"


class HarnessBindingPayload(TypedDict):
    backend: Literal["native", "external"]
    harness: str | None
    external_session_id: str | None
    locked_workspace: str | None
    runtime_generation: int


@dataclass(frozen=True, slots=True)
class SessionHarnessBinding:
    backend: Literal["native", "external"]
    harness: str | None
    external_session_id: str | None
    locked_workspace: str | None
    runtime_generation: int

    def payload(self) -> HarnessBindingPayload:
        return {
            "backend": self.backend,
            "harness": self.harness,
            "external_session_id": self.external_session_id,
            "locked_workspace": self.locked_workspace,
            "runtime_generation": self.runtime_generation,
        }


def external_binding(metadata: object) -> SessionHarnessBinding | None:
    if not isinstance(metadata, Mapping):
        return None
    raw = cast(Mapping[object, object], metadata).get(HARNESS_BINDING_KEY)
    if not isinstance(raw, dict):
        return None
    binding = cast(dict[object, object], raw)
    if binding.get("backend") != "external":
        return None
    session_id = binding.get("external_session_id")
    workspace = binding.get("locked_workspace")
    generation = binding.get("runtime_generation")
    if (
        binding.get("harness") != "claude_sdk"
        or (session_id is not None and not isinstance(session_id, str))
        or (workspace is not None and not isinstance(workspace, str))
        or type(generation) is not int or generation < 0
    ):
        raise ValueError("The Claude Code session binding is invalid.")
    return SessionHarnessBinding("external", "claude_sdk", session_id, workspace, generation)


def validate_harness_update(
    metadata: Mapping[str, object], updates: Mapping[str, object], removals: tuple[str, ...],
) -> None:
    """Keep the harness identity and its admitted folder fixed across writes."""
    current = external_binding(metadata)
    if current is None:
        return
    updated = external_binding(updates) if HARNESS_BINDING_KEY in updates else current
    if (
        HARNESS_BINDING_KEY in removals or updated is None
        or updated.runtime_generation < current.runtime_generation
        or current.locked_workspace not in (None, updated.locked_workspace)
    ):
        raise ValueError("The Claude Code session binding is immutable.")
    if current.locked_workspace is not None and "workspace_scope" in removals:
        raise ValueError("The Claude Code working folder is locked.")
    if current.locked_workspace is not None and "workspace_scope" in updates:
        raw = updates["workspace_scope"]
        if (
            not isinstance(raw, dict)
            or cast(dict[object, object], raw).get("project_path") != current.locked_workspace
        ):
            raise ValueError("The Claude Code working folder is locked.")
