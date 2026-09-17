"""Lifecycle ownership and teardown checks for the composed I3f fixture."""

from __future__ import annotations

import asyncio
import gc

import pytest

from agent.idle_acceptance_support import NetworkDenial, make_harness
from nanobot.agent.idle_compaction import IdleCompactionPolicy
from nanobot.agent.idle_lifecycle import IdleLifecycleIntegration
from nanobot.agent.idle_runtime import IdleRuntimeBindingRegistry
from nanobot.agent.idle_scanner import IdleCompactionScanner
from nanobot.providers.base import GenerationSettings
from nanobot.session.manager import JsonlSessionStore
from nanobot.utils.llm_runtime import LLMRuntime


@pytest.fixture(autouse=True)
def deny_external_io(monkeypatch: pytest.MonkeyPatch) -> NetworkDenial:
    denial = NetworkDenial()
    denial.install(monkeypatch)
    return denial


async def test_disabled_pair_is_inert_without_callbacks_or_workers(
    deny_external_io, tmp_path
) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    source = backend.source
    runtime = LLMRuntime(
        object(),
        "disabled-model",
        GenerationSettings(max_tokens=8),
        128,
        snapshot_signature=("disabled",),
    )
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=runtime,
        auxiliary_runtime=runtime,
    )
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        policy=IdleCompactionPolicy(),
    )
    lifecycle = IdleLifecycleIntegration(
        scanner=scanner,
        registry=registry,
        owns_scanner=True,
        owns_registry=True,
    )
    try:
        assert scanner.enabled is False
        assert lifecycle.enabled is False
        assert scanner.coordinator.worker_count == 0
        assert scanner.coordinator.pending_callback_count == 0
        assert await lifecycle.scan_once() == 0
        assert scanner.coordinator.worker_count == 0
        assert scanner.coordinator.pending_callback_count == 0
    finally:
        await lifecycle.aclose()
    assert scanner.closed
    assert registry.closed
    assert deny_external_io.attempts == []


async def test_borrowed_lifecycle_does_not_close_explicitly_owned_pair(
    deny_external_io, tmp_path
) -> None:
    harness = await make_harness(tmp_path)
    borrowed = IdleLifecycleIntegration(scanner=harness.scanner, registry=harness.registry)
    try:
        await borrowed.aclose()
        assert borrowed.closed
        assert harness.scanner.closed is False
        assert harness.registry.closed is False
        assert harness.scanner.enabled is True
        await harness.scanner.aclose()
        await harness.registry.aclose()
        assert harness.scanner.closed
        assert harness.registry.closed
        assert borrowed.enabled is False
    finally:
        await harness.close()
    assert deny_external_io.attempts == []


async def test_lifecycle_close_fences_scan_and_joins_transport_cleanup(
    deny_external_io, tmp_path
) -> None:
    harness = await make_harness(tmp_path, wait_for_release=True)
    close_task: asyncio.Task[None] | None = None
    try:
        assert await harness.lifecycle.scan_once() == 1
        await harness.records.started.wait()
        close_task = asyncio.create_task(harness.lifecycle.aclose())
        await harness.records.cancelled.wait()
        assert harness.lifecycle.enabled is False
        assert await harness.lifecycle.scan_once() == 0
        harness.records.release.set()
        await close_task
        close_task = None
        assert harness.lifecycle.closed
        assert harness.scanner.closed
        assert harness.registry.closed
        assert harness.scanner.discovery is not None
        assert harness.scanner.discovery.closed
        assert harness.scanner.root_authority is not None
        assert harness.scanner.root_authority.closed
        assert harness.scanner.coordinator.pending_callback_count == 0
        assert harness.scanner.coordinator.admission_count == 0
        assert harness.records.active_calls == 0
        gc.collect()
        assert all(request_ref() is None for request_ref in harness.records.request_refs)
    finally:
        harness.records.release.set()
        if close_task is not None:
            await close_task
        await harness.close()
    assert deny_external_io.attempts == []


async def test_concurrent_lifecycle_close_waiters_share_teardown(tmp_path) -> None:
    harness = await make_harness(tmp_path)
    try:
        first = asyncio.create_task(harness.lifecycle.aclose())
        second = asyncio.create_task(harness.lifecycle.aclose())
        await asyncio.gather(first, second)
        assert await harness.lifecycle.scan_once() == 0
        assert harness.lifecycle.closed
        assert harness.scanner.closed
        assert harness.registry.closed
    finally:
        await harness.close()
