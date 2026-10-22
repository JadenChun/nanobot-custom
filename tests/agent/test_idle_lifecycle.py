"""Focused I3e lifecycle ownership and explicit manual admission checks."""

from __future__ import annotations

import asyncio

import pytest

from nanobot.agent.idle_compaction import IdleCompactionPolicy
from nanobot.agent.idle_lifecycle import IdleLifecycleIntegration
from nanobot.agent.idle_runtime import IdleRuntimeBindingRegistry
from nanobot.agent.idle_scanner import IdleCompactionScanner
from nanobot.providers.base import GenerationSettings
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source_store import SourceSessionStore
from nanobot.utils.llm_runtime import LLMRuntime
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider

engine_config = engine_fixtures.engine_config


def _pair(tmp_path, *, enabled: bool = False):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    source = SourceSessionStore(backend)
    provider = ScriptedProvider([])
    runtime = LLMRuntime(
        provider,
        "fixture",
        GenerationSettings(max_tokens=32),
        4096,
        snapshot_signature=("fixture",),
    )
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=runtime,
        auxiliary_runtime=runtime,
    )
    policy = (
        IdleCompactionPolicy(
            idle_ttl_seconds=1,
            queue_capacity=1,
            max_concurrent_jobs=1,
            max_auxiliary_calls=1,
            max_auxiliary_tokens=1,
            max_auxiliary_wall_seconds=1,
            initial_backoff_seconds=1,
            max_backoff_seconds=1,
        )
        if enabled
        else IdleCompactionPolicy()
    )
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        policy=policy,
    )
    return scanner, registry


@pytest.mark.asyncio
async def test_lifecycle_disabled_pair_is_inert_and_manual_scan_is_explicit(tmp_path):
    scanner, registry = _pair(tmp_path)
    lifecycle = IdleLifecycleIntegration(
        scanner=scanner,
        registry=registry,
        owns_scanner=True,
        owns_registry=True,
    )
    assert scanner.enabled is False
    assert lifecycle.enabled is False
    assert await lifecycle.scan_once() == 0
    await lifecycle.aclose()
    await lifecycle.aclose()
    assert scanner.closed is True
    assert scanner.enabled is False
    assert registry.closed is True


@pytest.mark.asyncio
async def test_borrowed_pair_is_not_closed_by_lifecycle(tmp_path):
    scanner, registry = _pair(tmp_path)
    lifecycle = IdleLifecycleIntegration(scanner=scanner, registry=registry)
    await lifecycle.aclose()
    assert scanner.closed is False
    assert registry.closed is False
    await scanner.aclose()
    assert scanner.enabled is False
    assert lifecycle.enabled is False
    await registry.aclose()


@pytest.mark.asyncio
async def test_lifecycle_fences_when_borrowed_registry_starts_closing(tmp_path):
    scanner, registry = _pair(tmp_path)
    lifecycle = IdleLifecycleIntegration(scanner=scanner, registry=registry)
    closing = asyncio.create_task(registry.aclose())
    await asyncio.sleep(0)
    assert lifecycle.enabled is False
    await closing
    await lifecycle.aclose()
    await scanner.aclose()


@pytest.mark.asyncio
async def test_lifecycle_close_waiters_share_one_teardown_and_fence_scans(tmp_path):
    scanner, registry = _pair(tmp_path)
    lifecycle = IdleLifecycleIntegration(
        scanner=scanner,
        registry=registry,
        owns_scanner=True,
        owns_registry=True,
    )
    first = asyncio.create_task(lifecycle.aclose())
    second = asyncio.create_task(lifecycle.aclose())
    await asyncio.sleep(0)
    assert lifecycle.enabled is False
    assert await lifecycle.scan_once() == 0
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert scanner.closed is True
    assert registry.closed is True


def test_lifecycle_rejects_mismatched_source_pair(tmp_path):
    scanner, registry = _pair(tmp_path)
    other_scanner, _ = _pair(tmp_path / "other")
    with pytest.raises(ValueError, match="exact injected pair"):
        IdleLifecycleIntegration(scanner=other_scanner, registry=registry)
    # Keep the first pair's resources deterministic for test teardown callers.
    assert scanner.source_store is not other_scanner.source_store


async def test_loop_run_self_close_and_external_close_share_teardown(engine_config):
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.registry import ToolRegistry

    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
    )
    run_task = asyncio.create_task(loop.run())
    await asyncio.sleep(0)
    close_task = asyncio.create_task(loop.aclose())
    loop.stop()
    results = await asyncio.gather(run_task, close_task, return_exceptions=True)
    assert results[0] is None
    assert results[1] is None
    assert loop._closed is True
    await loop.aclose()


async def test_loop_bounded_waiter_can_leave_cooperative_close_owned(engine_config):
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.registry import ToolRegistry

    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def resistant_cleanup():
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    loop.schedule_background(resistant_cleanup())
    await started.wait()
    waiter = asyncio.create_task(loop.aclose())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(waiter, timeout=0.01)
    assert loop._close_task is not None and not loop._close_task.done()
    release.set()
    await loop.aclose()
    assert loop._closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["active", "background"])
async def test_loop_close_drains_reentrant_task_without_deadlock(engine_config, kind):
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.registry import ToolRegistry

    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
    )
    started = asyncio.Event()
    reentrant_finished = asyncio.Event()

    async def reentrant_task():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await loop.aclose()
            reentrant_finished.set()

    task = asyncio.create_task(reentrant_task())
    if kind == "active":
        loop._track_active_task("cli:reentrant", task)
    else:
        loop._background_tasks.add(task)
        task.add_done_callback(loop._background_tasks.discard)
    close_task = None
    try:
        await started.wait()
        close_task = asyncio.create_task(loop.aclose())
        await asyncio.wait_for(close_task, timeout=1)
        assert task.done()
        assert reentrant_finished.is_set()
        assert not loop._close_draining_tasks
    finally:
        if close_task is not None and not close_task.done():
            close_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await close_task
        if not loop._closed:
            await loop.aclose()


@pytest.mark.asyncio
async def test_loop_close_failure_is_shared_with_later_waiters(engine_config):
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.registry import ToolRegistry

    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
    )

    async def fail_cleanup():
        raise RuntimeError("injected cleanup failure")

    loop._exec_session_manager.close_all = fail_cleanup
    first = asyncio.create_task(loop.aclose())
    second = asyncio.create_task(loop.aclose())
    first_error, second_error = await asyncio.gather(
        first, second, return_exceptions=True
    )
    assert isinstance(first_error, RuntimeError)
    assert isinstance(second_error, RuntimeError)
    assert str(first_error) == str(second_error) == "injected cleanup failure"

    later_error = await asyncio.gather(loop.aclose(), return_exceptions=True)
    assert isinstance(later_error[0], RuntimeError)
    assert str(later_error[0]) == "injected cleanup failure"
