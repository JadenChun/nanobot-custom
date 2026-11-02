from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nanobot.remote.protocol import (
    MutationDecision,
    MutationEnvelope,
    stable_operation_digest,
)
from nanobot.remote.replay import OperationReplayCache


def envelope(**overrides) -> MutationEnvelope:
    values = {
        "request_id": "req-0001",
        "installation_id": "install-0001",
        "principal": "device-0001",
        "action": "chat.send",
        "digest": stable_operation_digest("device-0001", "chat.send", {"text": "hello"}),
        "gateway_generation": 2,
        "epoch": 7,
        "mode": "submit",
        "payload": {"text": "hello"},
    }
    values.update(overrides)
    return MutationEnvelope.model_validate(values)


class Counter:
    def __init__(self) -> None:
        self.effects = 0

    async def effect(self, _envelope: MutationEnvelope) -> dict:
        await asyncio.sleep(0)
        self.effects += 1
        return {"effect": self.effects}


@pytest.mark.asyncio
async def test_submit_then_replay_effect_once() -> None:
    cache = OperationReplayCache()
    counter = Counter()
    first = await cache.execute(
        envelope(),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    assert first.decision is MutationDecision.APPLIED
    assert first.effect_count == 1
    second = await cache.execute(
        envelope(mode="replay_only", epoch=1),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    assert second.decision is MutationDecision.REPLAYED
    assert second.effect_count == 1
    submit_again = await cache.execute(
        envelope(),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    assert submit_again.decision is MutationDecision.REPLAYED
    assert counter.effects == 1


@pytest.mark.asyncio
async def test_replay_only_miss_and_stale_submit_are_denied_without_effect() -> None:
    cache = OperationReplayCache()
    counter = Counter()
    unknown = await cache.execute(
        envelope(mode="replay_only"),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    assert unknown.decision is MutationDecision.OUTCOME_UNKNOWN
    assert unknown.effect_count == 0
    assert unknown.http_status == 409

    stale = await cache.execute(
        envelope(epoch=6),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    assert stale.decision is MutationDecision.DENIED
    assert stale.effect_count == 0
    generation = await cache.execute(
        envelope(gateway_generation=1),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    assert generation.decision is MutationDecision.DENIED
    assert counter.effects == 0


@pytest.mark.asyncio
async def test_changed_identity_digest_and_eviction() -> None:
    cache = OperationReplayCache(max_entries=1)
    counter = Counter()
    await cache.execute(
        envelope(),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    for changed in (
        {"principal": "device-0002"},
        {"digest": stable_operation_digest("device-0001", "chat.send", {"text": "changed"})},
        {"action": "chat.stop"},
    ):
        denied = await cache.execute(
            envelope(mode="replay_only", **changed),
            current_generation=2,
            current_epoch=7,
            authorize=lambda: asyncio.sleep(0, result=True),
            effect=counter.effect,
        )
        assert denied.decision is MutationDecision.DENIED

    await cache.execute(
        envelope(request_id="req-0002"),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    evicted = await cache.execute(
        envelope(mode="replay_only"),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    assert evicted.decision is MutationDecision.OUTCOME_UNKNOWN
    assert counter.effects == 2


@pytest.mark.asyncio
async def test_ttl_and_authorization_after_effect() -> None:
    class Clock:
        now = 1000.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    cache = OperationReplayCache(ttl_seconds=10, clock=clock)
    counter = Counter()
    allowed = True

    async def authorize() -> bool:
        return allowed

    # First authorization is checked before effect; revoke only for delivery.
    calls = 0

    async def authorize_revoke_midway() -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    denied = await cache.execute(
        envelope(),
        current_generation=2,
        current_epoch=7,
        authorize=authorize_revoke_midway,
        effect=counter.effect,
    )
    assert denied.decision is MutationDecision.DENIED
    assert counter.effects == 1
    allowed = True
    replay = await cache.execute(
        envelope(mode="replay_only"),
        current_generation=2,
        current_epoch=7,
        authorize=authorize,
        effect=counter.effect,
    )
    assert replay.decision is MutationDecision.REPLAYED
    clock.now += 11
    expired = await cache.execute(
        envelope(mode="replay_only"),
        current_generation=2,
        current_epoch=7,
        authorize=authorize,
        effect=counter.effect,
    )
    assert expired.decision is MutationDecision.OUTCOME_UNKNOWN


@pytest.mark.asyncio
async def test_concurrent_submit_has_one_effect() -> None:
    cache = OperationReplayCache()
    counter = Counter()
    barrier = asyncio.Barrier(2)

    async def run() -> MutationDecision:
        await barrier.wait()
        outcome = await cache.execute(
            envelope(),
            current_generation=2,
            current_epoch=7,
            authorize=lambda: asyncio.sleep(0, result=True),
            effect=counter.effect,
        )
        return outcome.decision

    decisions = await asyncio.gather(run(), run())
    assert sorted(decision.value for decision in decisions) == ["applied", "replayed"]
    assert counter.effects == 1


def test_replay_contract_vectors_are_present() -> None:
    path = Path(__file__).resolve().parents[2] / "mobile" / "contracts" / "replay-v1.json"
    vectors = json.loads(path.read_text(encoding="utf-8"))
    assert vectors["version"] == 1
    assert {case["mode"] for case in vectors["cases"]} == {"submit", "replay_only"}
    assert all("expected_effect_count" in case for case in vectors["cases"])


@pytest.mark.asyncio
async def test_failed_effect_is_uncertain_and_not_retried() -> None:
    cache = OperationReplayCache()
    counter = Counter()

    async def failing_effect(_envelope: MutationEnvelope) -> dict:
        counter.effects += 1
        raise TimeoutError("effect timed out")

    with pytest.raises(TimeoutError):
        await cache.execute(
            envelope(),
            current_generation=2,
            current_epoch=7,
            authorize=lambda: asyncio.sleep(0, result=True),
            effect=failing_effect,
        )
    retry = await cache.execute(
        envelope(),
        current_generation=2,
        current_epoch=7,
        authorize=lambda: asyncio.sleep(0, result=True),
        effect=counter.effect,
    )
    assert retry.decision is MutationDecision.OUTCOME_UNKNOWN
    assert retry.effect_count == 0
    assert counter.effects == 1


@pytest.mark.asyncio
async def test_concurrent_same_key_effect_runs_once_and_releases_lock() -> None:
    cache = OperationReplayCache()
    counter = Counter()
    gate = asyncio.Event()

    async def slow_effect(_envelope: MutationEnvelope) -> dict:
        counter.effects += 1
        await gate.wait()
        return {"effect": counter.effects}

    first = asyncio.create_task(
        cache.execute(
            envelope(),
            current_generation=2,
            current_epoch=7,
            authorize=lambda: asyncio.sleep(0, result=True),
            effect=slow_effect,
        )
    )
    await asyncio.sleep(0)
    second = asyncio.create_task(
        cache.execute(
            envelope(),
            current_generation=2,
            current_epoch=7,
            authorize=lambda: asyncio.sleep(0, result=True),
            effect=counter.effect,
        )
    )
    await asyncio.sleep(0)
    gate.set()
    results = await asyncio.gather(first, second)
    assert counter.effects == 1
    assert [item.decision for item in results] == [
        MutationDecision.APPLIED,
        MutationDecision.REPLAYED,
    ]
    assert cache._key_locks == {}
    assert cache._key_lock_refs == {}
