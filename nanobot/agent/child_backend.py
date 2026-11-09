"""Narrow execution backend for one delegated child generation.

The native backend routes work through the normal runner. External backends are
not implemented in this cutover and must fail closed through the same admission
response. Resume requires both the bound backend and its runtime generation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from nanobot.agent.child_sessions import ChildAdmission


class ChildBackendError(RuntimeError):
    """A child backend cannot start, resume, or control this generation."""


class ChildBackendUnavailableError(ChildBackendError):
    """The requested backend or generation is unavailable."""


@dataclass(frozen=True, slots=True)
class ChildBackendBinding:
    """Content-free proof for the current child execution generation."""

    backend: str = "native"
    runtime_generation: int = 0
    proof: str = ""

    def __post_init__(self) -> None:
        if self.backend not in {"native", "external"}:
            raise ValueError("child backend binding is invalid")
        if type(self.runtime_generation) is not int or self.runtime_generation < 0:
            raise ValueError("child backend generation is invalid")
        if type(self.proof) is not str:
            raise ValueError("child backend proof is invalid")


@dataclass(slots=True)
class ChildBackendHandle:
    """One live child execution owned by a backend."""

    child_id: str
    runtime_generation: int
    task: asyncio.Task[Any]


RunCall = Callable[[], Coroutine[Any, Any, Any]]


@runtime_checkable
class ChildBackend(Protocol):
    """Start, control, and settle one child generation."""

    async def start(
        self,
        admission: ChildAdmission,
        run: RunCall,
        binding: ChildBackendBinding,
    ) -> ChildBackendHandle: ...

    async def resume(
        self,
        admission: ChildAdmission,
        run: RunCall,
        binding: ChildBackendBinding,
    ) -> ChildBackendHandle: ...

    async def send(self, child_id: str, content: str) -> None: ...

    async def steer(self, child_id: str, content: str) -> None: ...

    async def interrupt(self, child_id: str) -> bool: ...

    async def wait(self, handle: ChildBackendHandle) -> Any: ...

    async def close(self, child_id: str) -> None: ...


class NativeChildBackend:
    """Run one child through the normal Nanobot runner."""

    backend_name = "native"

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def start(
        self,
        admission: ChildAdmission,
        run: RunCall,
        binding: ChildBackendBinding,
    ) -> ChildBackendHandle:
        if binding.backend != self.backend_name:
            raise ChildBackendUnavailableError("child backend binding does not match")
        if binding.runtime_generation != admission.runtime_generation:
            raise ChildBackendUnavailableError("child runtime generation is stale")
        existing = self._tasks.get(admission.child_id)
        if existing is not None and not existing.done():
            raise ChildBackendUnavailableError("child already has a live execution")
        task = asyncio.create_task(run())
        self._tasks[admission.child_id] = task
        return ChildBackendHandle(
            child_id=admission.child_id,
            runtime_generation=admission.runtime_generation,
            task=task,
        )

    async def resume(
        self,
        admission: ChildAdmission,
        run: RunCall,
        binding: ChildBackendBinding,
    ) -> ChildBackendHandle:
        if binding.backend != self.backend_name:
            raise ChildBackendUnavailableError("child backend binding does not match")
        if (
            binding.runtime_generation != admission.runtime_generation
            or not binding.proof.strip()
        ):
            raise ChildBackendUnavailableError("child resume proof is missing or stale")
        # The native runner is process-local. A restarted child must be
        # reconciled as uncertain rather than replayed.
        raise ChildBackendUnavailableError("native child execution cannot resume after restart")

    async def send(self, child_id: str, content: str) -> None:
        raise ChildBackendUnavailableError("native child has no mid-run input")

    async def steer(self, child_id: str, content: str) -> None:
        raise ChildBackendUnavailableError("native child has no mid-run steering")

    async def interrupt(self, child_id: str) -> bool:
        task = self._tasks.pop(child_id, None)
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def wait(self, handle: ChildBackendHandle) -> Any:
        return await handle.task

    async def close(self, child_id: str) -> None:
        await self.interrupt(child_id)


class ChildBackendRegistry:
    """Resolve one backend from an admitted profile; external is unsupported."""

    def __init__(self, native: NativeChildBackend | None = None) -> None:
        self._native = native or NativeChildBackend()

    @property
    def native(self) -> NativeChildBackend:
        return self._native

    def resolve(self, profile: str) -> ChildBackend:
        if profile == "delegated":
            return self._native
        raise ChildBackendUnavailableError(
            f"child backend for profile {profile!r} is not available"
        )


__all__ = [
    "ChildBackend",
    "ChildBackendBinding",
    "ChildBackendError",
    "ChildBackendHandle",
    "ChildBackendRegistry",
    "ChildBackendUnavailableError",
    "NativeChildBackend",
]
