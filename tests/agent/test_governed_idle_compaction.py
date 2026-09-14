"""Tests for the dormant, callback-owned idle-compaction coordinator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from nanobot.agent.idle_compaction import (
    IdleAuxiliaryLimits,
    IdleAuxiliaryUsage,
    IdleCompactionCoordinator,
    IdleCompactionPolicy,
    IdleComputation,
    IdleJobOutcome,
)


@dataclass
class FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def policy(**changes: object) -> IdleCompactionPolicy:
    values: dict[str, object] = {
        "idle_ttl_seconds": 30.0,
        "queue_capacity": 2,
        "max_concurrent_jobs": 1,
        "max_auxiliary_calls": 1,
        "max_auxiliary_tokens": 100,
        "max_auxiliary_wall_seconds": 10.0,
        "initial_backoff_seconds": 2.0,
        "max_backoff_seconds": 8.0,
    }
    values.update(changes)
    return IdleCompactionPolicy(**values)


@pytest.mark.parametrize(
    "disabled_policy",
    (IdleCompactionPolicy(), IdleCompactionPolicy(idle_ttl_seconds=30.0)),
    ids=("ttl-zero", "incomplete-finite-limits"),
)
async def test_disabled_policy_never_scans_or_invokes_callbacks(
    disabled_policy: IdleCompactionPolicy,
):
    calls: list[str] = []

    async def sessions() -> tuple[str, ...]:
        calls.append("sessions")
        return ("session",)

    async def capture(_: str) -> object | None:
        calls.append("capture")
        return object()

    async def compute(_: str, __: object, ___: object) -> IdleComputation[str]:
        calls.append("compute")
        return IdleComputation("candidate", IdleAuxiliaryUsage(calls=1, tokens=1))

    async def validate(_: str, __: object, ___: str) -> bool:
        calls.append("validate")
        return True

    coordinator: IdleCompactionCoordinator[object, str] = IdleCompactionCoordinator(
        policy=disabled_policy,
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    try:
        assert await coordinator.scan_once() == 0
        assert calls == []
        assert coordinator.worker_count == 0
    finally:
        await coordinator.aclose()


async def test_capture_none_surfaces_an_eligibility_blocker_without_compute_or_validate():
    calls: list[str] = []

    async def sessions() -> tuple[str, ...]:
        return ("blocked",)

    async def capture(key: str) -> object | None:
        calls.append(f"capture:{key}")
        return None

    async def compute(_: str, __: object, ___: object) -> IdleComputation[str]:
        calls.append("compute")
        raise AssertionError("ineligible captures must never compute")

    async def validate(_: str, __: object, ___: str) -> bool:
        calls.append("validate")
        raise AssertionError("ineligible captures must never validate")

    coordinator: IdleCompactionCoordinator[object, str] = IdleCompactionCoordinator(
        policy=policy(),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    try:
        assert await coordinator.scan_once() == 0
        await coordinator.drain()
        assert calls == ["capture:blocked"]
        assert coordinator.outcome_for("blocked") is IdleJobOutcome.INELIGIBLE
    finally:
        await coordinator.aclose()


async def test_compute_runs_outside_coordinator_locks_and_passes_opaque_tokens_to_validator():
    capture_token = object()
    observed: list[object] = []
    notification_finished = asyncio.Event()

    async def sessions() -> tuple[str, ...]:
        return ("idle",)

    async def capture(_: str) -> object | None:
        return capture_token

    async def compute(_: str, __: object, ___: object) -> IdleComputation[str]:
        # A foreground admission has to acquire the coordinator's bookkeeping
        # lock while this callback is running.  It would deadlock if compute
        # retained any coordinator lock.
        await asyncio.wait_for(coordinator.foreground_admitted("other"), timeout=0.2)
        notification_finished.set()
        return IdleComputation("candidate", IdleAuxiliaryUsage(calls=1, tokens=1))

    async def validate(_: str, token: object, candidate: str) -> bool:
        observed.extend((token, candidate))
        return True

    coordinator: IdleCompactionCoordinator[object, str] = IdleCompactionCoordinator(
        policy=policy(),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    try:
        assert await coordinator.scan_once() == 1
        await coordinator.drain()
        assert notification_finished.is_set()
        assert observed == [capture_token, "candidate"]
        assert coordinator.outcome_for("idle") is IdleJobOutcome.SUCCEEDED
    finally:
        await coordinator.aclose()


async def test_queue_capacity_global_concurrency_and_per_session_deduplication():
    scan_keys: list[tuple[str, ...]] = [("one",), ("two", "one"), ("three",)]
    started = asyncio.Event()
    release = asyncio.Event()
    captures: list[str] = []
    running = 0
    maximum_running = 0

    async def sessions() -> tuple[str, ...]:
        return scan_keys.pop(0)

    async def capture(key: str) -> str | None:
        captures.append(key)
        return f"token:{key}"

    async def compute(key: str, _: str, __: object) -> IdleComputation[str]:
        nonlocal maximum_running, running
        running += 1
        maximum_running = max(maximum_running, running)
        try:
            if key == "one":
                started.set()
                await release.wait()
            return IdleComputation(key, IdleAuxiliaryUsage(calls=1, tokens=1))
        finally:
            running -= 1

    async def validate(_: str, __: str, ___: str) -> bool:
        return True

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(queue_capacity=1, max_concurrent_jobs=1),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    try:
        assert await coordinator.scan_once() == 1
        await asyncio.wait_for(started.wait(), timeout=0.5)
        assert await coordinator.scan_once() == 1
        assert await coordinator.scan_once() == 0
        assert captures == ["one", "two"]
        assert coordinator.outcome_for("three") is IdleJobOutcome.QUEUE_FULL
        release.set()
        await coordinator.drain()
        assert maximum_running == 1
        assert coordinator.outcome_for("one") is IdleJobOutcome.SUCCEEDED
        assert coordinator.outcome_for("two") is IdleJobOutcome.SUCCEEDED
    finally:
        await coordinator.aclose()


async def test_foreground_admission_cancels_queued_and_running_jobs_and_discards_results():
    scan_keys: list[tuple[str, ...]] = [("running",), ("queued",)]
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    validated: list[str] = []

    async def sessions() -> tuple[str, ...]:
        return scan_keys.pop(0)

    async def capture(key: str) -> str | None:
        return key

    async def compute(key: str, _: str, __: object) -> IdleComputation[str]:
        if key == "running":
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                raise
        return IdleComputation(key, IdleAuxiliaryUsage(calls=1, tokens=1))

    async def validate(key: str, _: str, __: str) -> bool:
        validated.append(key)
        return True

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(queue_capacity=1),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    try:
        assert await coordinator.scan_once() == 1
        await asyncio.wait_for(started.wait(), timeout=0.5)
        assert await coordinator.scan_once() == 1
        await coordinator.foreground_admitted("queued")
        await coordinator.foreground_admitted("running")
        await coordinator.drain()
        assert cancellation_seen.is_set()
        assert validated == []
        assert coordinator.outcome_for("queued") is IdleJobOutcome.CANCELLED
        assert coordinator.outcome_for("running") is IdleJobOutcome.CANCELLED
    finally:
        await coordinator.aclose()


async def test_validator_controls_generation_and_activity_compare_and_swap_discard():
    token = {"generation": 4, "activity": 8}
    validated: list[tuple[object, str]] = []

    async def sessions() -> tuple[str, ...]:
        return ("race",)

    async def capture(_: str) -> dict[str, int] | None:
        return token

    async def compute(_: str, __: dict[str, int], ___: object) -> IdleComputation[str]:
        return IdleComputation("candidate", IdleAuxiliaryUsage(calls=1, tokens=1))

    async def validate(_: str, captured: object, candidate: str) -> bool:
        validated.append((captured, candidate))
        return False

    coordinator: IdleCompactionCoordinator[dict[str, int], str] = IdleCompactionCoordinator(
        policy=policy(),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    try:
        assert await coordinator.scan_once() == 1
        await coordinator.drain()
        assert validated == [(token, "candidate")]
        assert coordinator.outcome_for("race") is IdleJobOutcome.DISCARDED
        assert coordinator.backoff_for("race") is None
    finally:
        await coordinator.aclose()


@pytest.mark.parametrize(
    ("result", "advance"),
    [
        (IdleComputation("candidate", None), 0.0),
        (IdleComputation("candidate", IdleAuxiliaryUsage(calls=2, tokens=1)), 0.0),
        (IdleComputation("candidate", IdleAuxiliaryUsage(calls=1, tokens=101)), 0.0),
        (IdleComputation("candidate", IdleAuxiliaryUsage(calls=1, tokens=1)), 11.0),
    ],
    ids=("unknown-usage", "call-ceiling", "token-ceiling", "wall-ceiling"),
)
async def test_auxiliary_budget_failures_back_off_without_validation(
    result: IdleComputation[str], advance: float,
):
    clock = FakeClock()
    captures = 0
    validations = 0

    async def sessions() -> tuple[str, ...]:
        return ("cost",)

    async def capture(_: str) -> str | None:
        nonlocal captures
        captures += 1
        return "token"

    async def compute(_: str, __: str, ___: object) -> IdleComputation[str]:
        clock.advance(advance)
        return result

    async def validate(_: str, __: str, ___: str) -> bool:
        nonlocal validations
        validations += 1
        return True

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
        clock=clock,
    )
    try:
        assert await coordinator.scan_once() == 1
        await coordinator.drain()
        assert coordinator.outcome_for("cost") is IdleJobOutcome.BUDGET_EXCEEDED
        backoff = coordinator.backoff_for("cost")
        assert backoff is not None and backoff.failures == 1 and backoff.retry_at == clock.now + 2.0
        assert await coordinator.scan_once() == 0
        assert captures == 1
        assert coordinator.outcome_for("cost") is IdleJobOutcome.BACKING_OFF
        assert validations == 0
    finally:
        await coordinator.aclose()


async def test_exponential_backoff_resets_only_after_a_validated_success():
    clock = FakeClock()
    attempts = 0

    async def sessions() -> tuple[str, ...]:
        return ("retry",)

    async def capture(_: str) -> str | None:
        return "token"

    async def compute(_: str, __: str, ___: object) -> IdleComputation[str]:
        nonlocal attempts
        attempts += 1
        tokens = 101 if attempts < 3 else 1
        return IdleComputation("candidate", IdleAuxiliaryUsage(calls=1, tokens=tokens))

    async def validate(_: str, __: str, ___: str) -> bool:
        return True

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
        clock=clock,
    )
    try:
        assert await coordinator.scan_once() == 1
        await coordinator.drain()
        clock.advance(2.0)
        assert await coordinator.scan_once() == 1
        await coordinator.drain()
        backoff = coordinator.backoff_for("retry")
        assert backoff is not None and backoff.failures == 2 and backoff.retry_at == 6.0
        clock.advance(4.0)
        assert await coordinator.scan_once() == 1
        await coordinator.drain()
        assert attempts == 3
        assert coordinator.outcome_for("retry") is IdleJobOutcome.SUCCEEDED
        assert coordinator.backoff_for("retry") is None
    finally:
        await coordinator.aclose()


async def test_aclose_stops_admission_clears_queue_cancels_work_and_awaits_cleanup():
    scan_keys: list[tuple[str, ...]] = [("running",), ("queued",)]
    started = asyncio.Event()
    cancelled = asyncio.Event()
    session_scans = 0

    async def sessions() -> tuple[str, ...]:
        nonlocal session_scans
        session_scans += 1
        return scan_keys.pop(0)

    async def capture(key: str) -> str | None:
        return key

    async def compute(key: str, _: str, __: object) -> IdleComputation[str]:
        if key == "running":
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return IdleComputation(key, IdleAuxiliaryUsage(calls=1, tokens=1))

    async def validate(_: str, __: str, ___: str) -> bool:
        return True

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(queue_capacity=1),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    assert await coordinator.scan_once() == 1
    await asyncio.wait_for(started.wait(), timeout=0.5)
    assert await coordinator.scan_once() == 1
    await asyncio.wait_for(coordinator.aclose(), timeout=0.5)
    assert cancelled.is_set()
    assert coordinator.worker_count == 0
    assert coordinator.queue_depth == 0
    assert await coordinator.scan_once() == 0
    assert session_scans == 2


async def test_compute_receives_immutable_pre_dispatch_auxiliary_limits():
    observed: list[IdleAuxiliaryLimits] = []

    async def sessions() -> tuple[str, ...]:
        return ("limits",)

    async def capture(_: str) -> str | None:
        return "token"

    async def compute(_: str, __: str, limits: IdleAuxiliaryLimits) -> IdleComputation[str]:
        observed.append(limits)
        with pytest.raises(AttributeError):
            limits.max_calls = 999  # type: ignore[misc]
        return IdleComputation("candidate", IdleAuxiliaryUsage(calls=1, tokens=1))

    async def validate(_: str, __: str, ___: str) -> bool:
        return True

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    try:
        assert await coordinator.scan_once() == 1
        await coordinator.drain()
        assert observed == [IdleAuxiliaryLimits(max_calls=1, max_tokens=100, max_wall_seconds=10.0)]
    finally:
        await coordinator.aclose()


class _SubclassedUsage(IdleAuxiliaryUsage):
    pass


@pytest.mark.parametrize(
    "usage",
    (
        _SubclassedUsage(calls=1, tokens=1),
        object.__new__(IdleAuxiliaryUsage),
    ),
    ids=("subclass", "uninitialized-exact-type"),
)
async def test_usage_receipts_require_the_exact_initialized_type(usage: IdleAuxiliaryUsage):
    validations = 0

    async def sessions() -> tuple[str, ...]:
        return ("usage",)

    async def capture(_: str) -> str | None:
        return "token"

    async def compute(_: str, __: str, ___: IdleAuxiliaryLimits) -> IdleComputation[str]:
        return IdleComputation("candidate", usage)

    async def validate(_: str, __: str, ___: str) -> bool:
        nonlocal validations
        validations += 1
        return True

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    try:
        assert await coordinator.scan_once() == 1
        await coordinator.drain()
        assert coordinator.outcome_for("usage") is IdleJobOutcome.BUDGET_EXCEEDED
        assert validations == 0
    finally:
        await coordinator.aclose()


async def test_concurrent_scans_reserve_total_admission_capacity_before_capture():
    next_session = 0
    capture_started: list[str] = []
    release_captures = asyncio.Event()

    async def sessions() -> tuple[str, ...]:
        nonlocal next_session
        next_session += 1
        return (f"session-{next_session}",)

    async def capture(key: str) -> str | None:
        capture_started.append(key)
        await release_captures.wait()
        return key

    async def compute(_: str, __: str, ___: IdleAuxiliaryLimits) -> IdleComputation[str]:
        return IdleComputation("candidate", IdleAuxiliaryUsage(calls=1, tokens=1))

    async def validate(_: str, __: str, ___: str) -> bool:
        return True

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(max_concurrent_jobs=1, queue_capacity=1),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    try:
        scans = [asyncio.create_task(coordinator.scan_once()) for _ in range(3)]
        for _ in range(20):
            if len(capture_started) == 2:
                break
            await asyncio.sleep(0)
        assert len(capture_started) == 2
        release_captures.set()
        assert sum(await asyncio.gather(*scans)) == 2
        await coordinator.drain()
        assert coordinator.outcome_for("session-3") is IdleJobOutcome.QUEUE_FULL
    finally:
        await coordinator.aclose()


async def test_aclose_cancels_and_awaits_blocked_enumeration_despite_caller_cancellation():
    enumeration_started = asyncio.Event()
    enumeration_cancelled = asyncio.Event()
    release_enumeration_cleanup = asyncio.Event()

    async def sessions() -> tuple[str, ...]:
        enumeration_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            enumeration_cancelled.set()
            await release_enumeration_cleanup.wait()
            raise

    async def capture(_: str) -> str | None:
        raise AssertionError("blocked enumeration must not capture")

    async def compute(_: str, __: str, ___: IdleAuxiliaryLimits) -> IdleComputation[str]:
        raise AssertionError("blocked enumeration must not compute")

    async def validate(_: str, __: str, ___: str) -> bool:
        raise AssertionError("blocked enumeration must not validate")

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    scan = asyncio.create_task(coordinator.scan_once())
    await asyncio.wait_for(enumeration_started.wait(), timeout=0.5)
    first_close = asyncio.create_task(coordinator.aclose())
    await asyncio.wait_for(enumeration_cancelled.wait(), timeout=0.5)
    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close
    second_close = asyncio.create_task(coordinator.aclose())
    await asyncio.sleep(0)
    assert not second_close.done()
    release_enumeration_cleanup.set()
    await asyncio.wait_for(second_close, timeout=0.5)
    assert await asyncio.wait_for(scan, timeout=0.5) == 0
    assert coordinator.pending_callback_count == 0
    await asyncio.wait_for(coordinator.drain(), timeout=0.5)


async def test_foreground_then_close_awaits_cooperative_compute_cleanup_without_orphaning():
    compute_started = asyncio.Event()
    compute_cancelled = asyncio.Event()
    release_compute_cleanup = asyncio.Event()
    compute_cleaned = asyncio.Event()

    async def sessions() -> tuple[str, ...]:
        return ("running",)

    async def capture(_: str) -> str | None:
        return "token"

    async def compute(_: str, __: str, ___: IdleAuxiliaryLimits) -> IdleComputation[str]:
        compute_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            compute_cancelled.set()
            await release_compute_cleanup.wait()
            compute_cleaned.set()
            raise

    async def validate(_: str, __: str, ___: str) -> bool:
        raise AssertionError("cancelled compute must never validate")

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    assert await coordinator.scan_once() == 1
    await asyncio.wait_for(compute_started.wait(), timeout=0.5)
    await coordinator.foreground_admitted("running")
    await asyncio.wait_for(compute_cancelled.wait(), timeout=0.5)
    closing = asyncio.create_task(coordinator.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    release_compute_cleanup.set()
    await asyncio.wait_for(closing, timeout=0.5)
    assert compute_cleaned.is_set()
    assert coordinator.pending_callback_count == 0
    assert coordinator.queue_depth == 0
    await asyncio.wait_for(coordinator.drain(), timeout=0.5)


async def test_timeout_cleanup_stays_owned_through_aclose_and_settles_budget_outcome():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def sessions() -> tuple[str, ...]:
        return ("timeout",)

    async def capture(_: str) -> str | None:
        return "token"

    async def compute(_: str, __: str, ___: IdleAuxiliaryLimits) -> IdleComputation[str]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished.set()
            raise

    async def validate(_: str, __: str, ___: str) -> bool:
        raise AssertionError("a timed out compute must never validate")

    coordinator: IdleCompactionCoordinator[str, str] = IdleCompactionCoordinator(
        policy=policy(max_auxiliary_wall_seconds=0.01),
        enumerate_sessions=sessions,
        capture=capture,
        compute=compute,
        validate=validate,
    )
    assert await coordinator.scan_once() == 1
    await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
    closing = asyncio.create_task(coordinator.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    release_cleanup.set()
    await asyncio.wait_for(closing, timeout=0.5)
    assert cleanup_finished.is_set()
    assert coordinator.outcome_for("timeout") is IdleJobOutcome.BUDGET_EXCEEDED
    assert coordinator.admission_count == 0
    assert coordinator.pending_callback_count == 0
    assert coordinator.queue_depth == 0
    await asyncio.wait_for(coordinator.drain(), timeout=0.5)
