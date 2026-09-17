"""Explicit, dormant I3e ownership for the idle compaction substrate.

This module is deliberately a small lifecycle boundary.  It does not build a
scanner, registry, source store, runtime, provider, or transport.  A caller
that has already constructed the complete I3d/I3b objects may transfer the
scanner and/or registry here and then inject the resulting object into an
``AgentLoop``.
"""

from __future__ import annotations

import asyncio

from nanobot.agent.idle_runtime import IdleRuntimeBindingRegistry
from nanobot.agent.idle_scanner import IdleCompactionScanner


class IdleLifecycleError(RuntimeError):
    """The explicitly injected idle lifecycle is malformed or unavailable."""


class IdleLifecycleIntegration:
    """Own and coordinate explicitly supplied I3d and I3b components.

    ``owns_scanner`` and ``owns_registry`` are intentionally explicit.  A
    borrowed component is never closed by this wrapper, which lets a caller
    share a registry or scanner with another owner without double teardown.
    The registry may not own the same coordinator that the scanner composes;
    one lifecycle owner must be unambiguously responsible for that callback
    scheduler.
    """

    def __init__(
        self,
        *,
        scanner: IdleCompactionScanner,
        registry: IdleRuntimeBindingRegistry,
        owns_scanner: bool = False,
        owns_registry: bool = False,
    ) -> None:
        if type(scanner) is not IdleCompactionScanner:
            raise TypeError("scanner must be an exact IdleCompactionScanner")
        if type(registry) is not IdleRuntimeBindingRegistry:
            raise TypeError("registry must be an exact IdleRuntimeBindingRegistry")
        if type(owns_scanner) is not bool:
            raise TypeError("owns_scanner must be a boolean")
        if type(owns_registry) is not bool:
            raise TypeError("owns_registry must be a boolean")

        scanner_registry = getattr(scanner, "binding_registry", None)
        if scanner_registry is not registry:
            raise ValueError("scanner and registry must be the exact injected pair")
        scanner_source = getattr(scanner, "source_store", None)
        try:
            registry_source = registry.source_store
        except Exception as exc:
            raise ValueError("registry source authority is unavailable") from exc
        if scanner_source is not registry_source:
            raise ValueError("scanner and registry must share the exact source store")

        scanner_coordinator = getattr(scanner, "coordinator", None)
        registry_coordinator = getattr(registry, "_coordinator", None)
        registry_owns_coordinator = bool(getattr(registry, "_owns_coordinator", False))
        if (
            scanner_coordinator is not None
            and registry_coordinator is scanner_coordinator
            and registry_owns_coordinator
        ):
            raise ValueError("registry cannot own the scanner coordinator")

        self._scanner = scanner
        self._registry = registry
        self._owns_scanner = owns_scanner
        self._owns_registry = owns_registry
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def scanner(self) -> IdleCompactionScanner:
        return self._scanner

    @property
    def registry(self) -> IdleRuntimeBindingRegistry:
        return self._registry

    @property
    def owns_scanner(self) -> bool:
        return self._owns_scanner

    @property
    def owns_registry(self) -> bool:
        return self._owns_registry

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def enabled(self) -> bool:
        """Return enabled while both transferred authorities are open."""

        if self._closing or self._closed:
            return False
        try:
            scanner_enabled = self._scanner.enabled
            scanner_closed = self._scanner.closed
            scanner_closing = bool(getattr(self._scanner, "_closing", False))
            registry_closed = self._registry.closed
            registry_closing = bool(getattr(self._registry, "_closing", False))
        except Exception:
            return False
        return (
            scanner_enabled is True
            and scanner_closed is False
            and not scanner_closing
            and registry_closed is False
            and not registry_closing
        )

    async def foreground_admitted(self, session_key: str) -> None:
        """Invalidate scanner work after a durable foreground activity write."""

        if not self.enabled:
            return
        await self._scanner.foreground_admitted(session_key)

    async def scan_once(self) -> int:
        """Run one explicit scan when the injected lifecycle is enabled."""

        if not self.enabled:
            return 0
        return await self._scanner.scan_once()

    async def _close_owned(self) -> None:
        self._closed = True
        errors: list[BaseException] = []
        # The scanner owns the discovery/root side of its composition.  Close
        # it before the registry, while still retaining the registry gate for
        # scanner callbacks.  Every later cleanup step runs after an error.
        if self._owns_scanner:
            try:
                await self._scanner.aclose()
            except BaseException as exc:
                errors.append(exc)
        if self._owns_registry:
            try:
                await self._registry.aclose()
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("failed to close idle lifecycle", errors)

    async def aclose(self) -> None:
        """Close transferred components once, surviving waiter cancellation."""

        # This assignment is intentionally before task creation.  A concurrent
        # foreground notification or manual scan must observe the fence even if
        # the actual teardown coroutine has not been scheduled yet.
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned())
        task = self._close_task
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # The shared teardown remains owned by this lifecycle object;
                # a bounded waiter may return without cancelling that task.
                raise
            except BaseException:
                # Inspect the settled shared task below and preserve its
                # error without scheduling a second teardown.
                break
        if task.cancelled():
            raise asyncio.CancelledError
        error = task.exception()
        if error is not None:
            raise error

    async def close(self) -> None:
        """Compatibility alias for asynchronous lifecycle owners."""

        await self.aclose()


# Short descriptive aliases keep integration call sites readable while the
# fully qualified type remains the authoritative contract.
IdleCompactionLifecycle = IdleLifecycleIntegration
IdleLifecycle = IdleLifecycleIntegration


__all__ = [
    "IdleCompactionLifecycle",
    "IdleLifecycle",
    "IdleLifecycleError",
    "IdleLifecycleIntegration",
]
