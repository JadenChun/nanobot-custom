"""Regression tests for A3-5 AC5.4: the narrow child backend."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from nanobot.agent.child_backend import (
    ChildAdmission,
    ChildBackendBinding,
    ChildBackendRegistry,
    ChildBackendUnavailableError,
    NativeChildBackend,
)


def _admission() -> ChildAdmission:
    return ChildAdmission(
        child_id="child-a",
        session_key="delegated:child-a",
        parent_key="websocket:parent",
        parent_incarnation="inc-1",
        root_key="websocket:parent",
        profile="delegated",
        depth=1,
        attempt_id="turn-1",
        assignment_digest="digest",
        policy_facts={},
        runtime_generation=0,
    )


def _binding(runtime_generation: int = 0, proof: str = "") -> ChildBackendBinding:
    return ChildBackendBinding(
        backend="native",
        runtime_generation=runtime_generation,
        proof=proof,
    )


@pytest.mark.asyncio
async def test_native_backend_starts_waits_and_rejects_second_start() -> None:
    backend = NativeChildBackend()
    admission = _admission()

    async def run() -> str:
        return "done"

    handle = await backend.start(admission, run, _binding())
    assert handle.child_id == "child-a"
    assert handle.runtime_generation == 0
    assert await backend.wait(handle) == "done"

    release = asyncio.Event()

    async def live_run() -> str:
        await release.wait()
        return "never"

    live_handle = await backend.start(admission, live_run, _binding())
    try:
        with pytest.raises(ChildBackendUnavailableError):
            await backend.start(admission, live_run, _binding())
        assert not live_handle.task.done()
    finally:
        release.set()
        await backend.close("child-a")


@pytest.mark.asyncio
async def test_interrupt_cancels_only_the_child_task() -> None:
    backend = NativeChildBackend()
    admission = _admission()

    async def run() -> str:
        await asyncio.Event().wait()
        return "never"

    sibling = replace(
        _admission(),
        child_id="child-b",
        session_key="delegated:child-b",
    )

    handle = await backend.start(admission, run, _binding())
    sibling_handle = await backend.start(sibling, run, _binding())
    try:
        assert await backend.interrupt("child-a") is True
        with pytest.raises(asyncio.CancelledError):
            await handle.task
        assert handle.task.done()
        assert handle.task.cancelled()

        assert await backend.interrupt("child-a") is False
        assert not sibling_handle.task.done()
    finally:
        await backend.close("child-a")
        await backend.close("child-b")


@pytest.mark.asyncio
async def test_resume_requires_generation_and_proof() -> None:
    backend = NativeChildBackend()
    admission = _admission()

    async def run() -> str:
        return "resumed"

    with pytest.raises(ChildBackendUnavailableError):
        await backend.resume(admission, run, _binding(runtime_generation=1, proof="proof-1"))

    with pytest.raises(ChildBackendUnavailableError):
        await backend.resume(admission, run, _binding(runtime_generation=0, proof=""))

    with pytest.raises(ChildBackendUnavailableError) as excinfo:
        await backend.resume(admission, run, _binding(runtime_generation=0, proof="proof-1"))
    assert "restart" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_native_send_and_steer_are_unavailable() -> None:
    backend = NativeChildBackend()

    with pytest.raises(ChildBackendUnavailableError):
        await backend.send("child-a", "hello")

    with pytest.raises(ChildBackendUnavailableError):
        await backend.steer("child-a", "hello")


@pytest.mark.asyncio
async def test_registry_rejects_unknown_profiles() -> None:
    registry = ChildBackendRegistry()

    assert isinstance(registry.resolve("delegated"), NativeChildBackend)

    with pytest.raises(ChildBackendUnavailableError):
        registry.resolve("crawler")

    with pytest.raises(ChildBackendUnavailableError):
        registry.resolve("external")
