"""Private bounded filesystem facts; persisted data never grants write authority."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

_MINT = object()
_OPERATIONS = frozenset({"write_file", "edit_file", "apply_patch", "generate_image", "save_knowledge"})


@dataclass(frozen=True, slots=True)
class FilesystemEffectFact:
    operation: str
    target: str
    disposition: Literal["applied", "uncertain"]
    intended_sha256: str
    intended_bytes: int
    reconciliation_required: bool
    _mint: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._mint is not _MINT:
            raise TypeError("filesystem effect facts require their concrete owner")

    def belongs_to(self, operation: str) -> bool:
        return self._mint is _MINT and self.operation == operation

    def to_dict(self, execution_id: str) -> dict[str, object]:
        return {
            "version": 1, "operation": self.operation, "target": self.target,
            "disposition": self.disposition, "intended_sha256": self.intended_sha256,
            "intended_bytes": self.intended_bytes, "reconciliation_required": self.reconciliation_required,
            "execution_id": execution_id,
        }


class FilesystemEffectReconciliationRequiredError(RuntimeError):
    """Abort the invocation after persisting an unresolved filesystem result."""

    def __init__(self) -> None:
        super().__init__("Filesystem effect requires reconciliation before execution can continue")


class FilesystemEffectCancelled(asyncio.CancelledError):
    """Keep cancellation distinct while transporting concrete effect evidence."""

    def __init__(self, message: str, facts: tuple[FilesystemEffectFact, ...]) -> None:
        super().__init__(message)
        self.facts = facts


@dataclass(slots=True)
class FilesystemEffectAttempt:
    """Allocated immediately before the first mutation, including parent mkdir."""

    operation: str
    target: str
    intended_sha256: str
    intended_bytes: int
    applied: bool = False

    @classmethod
    def enter(cls, operation: str, target: Path, intended: bytes) -> FilesystemEffectAttempt:
        capture = _CURRENT_CAPTURE.get()
        if capture is not None:
            operation = capture.operation
        resolved = str(target)
        if operation not in _OPERATIONS or len(resolved) > 32768:
            raise ValueError("Invalid filesystem effect target")
        attempt = cls(operation, resolved, hashlib.sha256(intended).hexdigest(), len(intended))
        if capture is not None:
            if len(capture.attempts) >= 20:
                raise ValueError("Too many filesystem effect targets")
            capture.attempts.append(attempt)
        return attempt

    def failed(self) -> FilesystemEffectFact:
        return FilesystemEffectFact(
            self.operation, self.target, "applied" if self.applied else "uncertain",
            self.intended_sha256, self.intended_bytes, True, _MINT,
        )


def _concrete_operation(tool: object) -> str | None:
    from nanobot.agent.tools.apply_patch import ApplyPatchTool
    from nanobot.agent.tools.base import Tool
    from nanobot.agent.tools.filesystem import EditFileTool, WriteFileTool
    from nanobot.agent.tools.image_generation import ImageGenerationTool
    from nanobot.agent.tools.repositories import SaveKnowledgeTool

    if isinstance(tool, Tool) and type(tool) in {
        WriteFileTool, EditFileTool, ApplyPatchTool, ImageGenerationTool, SaveKnowledgeTool,
    }:
        return tool.name
    return None


@dataclass(slots=True)
class FilesystemEffectCapture:
    """One concrete dispatch's attempts, including nested binary publication."""

    operation: str
    attempts: list[FilesystemEffectAttempt] = field(default_factory=list)

    def failed(self) -> tuple[FilesystemEffectFact, ...]:
        return tuple(attempt.failed() for attempt in self.attempts)


_CURRENT_CAPTURE: ContextVar[FilesystemEffectCapture | None] = ContextVar(
    "filesystem_effect_capture", default=None,
)


@contextmanager
def capture_filesystem_effects(tool: object) -> Generator[FilesystemEffectCapture | None]:
    operation = _concrete_operation(tool)
    capture = FilesystemEffectCapture(operation) if operation is not None else None
    token = _CURRENT_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _CURRENT_CAPTURE.reset(token)


def accepted_filesystem_effects(tool: object, facts: object) -> bool:
    """Only concrete first-party mutation implementations can supply these facts."""
    operation = _concrete_operation(tool)
    if operation is None or not isinstance(facts, tuple):
        return False
    values = cast(tuple[object, ...], facts)
    return len(values) <= 20 and all(
        isinstance(fact, FilesystemEffectFact) and type(fact) is FilesystemEffectFact
        and fact.belongs_to(operation) for fact in values
    )


def validate_persisted_filesystem_effects(value: object, *, execution_id: str, name: object) -> bool:
    """Validate data shape and report unresolved evidence, without minting a fact."""
    if not isinstance(value, list):
        raise ValueError("Invalid filesystem effect evidence")
    values = cast(list[object], value)
    if len(values) > 20:
        raise ValueError("Invalid filesystem effect evidence")
    required = {"version", "operation", "target", "disposition", "intended_sha256",
                "intended_bytes", "reconciliation_required", "execution_id"}
    unresolved = False
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("Invalid filesystem effect evidence")
        fact = cast(dict[str, object], item)
        digest = fact.get("intended_sha256")
        target = fact.get("target")
        if (
            set(fact) != required or type(fact.get("version")) is not int or fact["version"] != 1
            or fact.get("operation") not in _OPERATIONS or fact["operation"] != name
            or not isinstance(target, str) or not target or len(target) > 32768
            or fact.get("disposition") not in {"applied", "uncertain"}
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or type(fact.get("intended_bytes")) is not int or cast(int, fact["intended_bytes"]) < 0
            or type(fact.get("reconciliation_required")) is not bool
            or fact.get("execution_id") != execution_id
        ):
            raise ValueError("Invalid filesystem effect evidence")
        unresolved |= fact["reconciliation_required"] is True
    return unresolved
