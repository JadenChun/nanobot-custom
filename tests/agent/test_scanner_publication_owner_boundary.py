"""Retain the offline scanner prerequisite before publication-owner changes."""

import asyncio
import socket
import traceback
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from nanobot.agent.idle_scanner import (
    IdleScannerCapture,
    IdleScannerComputation,
    IdleScannerPublicationConflict,
)
from tests.agent.idle_acceptance_support import NetworkDenial, make_harness


async def test_real_scanner_proposal_validation_performs_no_dns_lookup(tmp_path, monkeypatch):
    denial = NetworkDenial()
    denial.install(monkeypatch)
    denied_getaddrinfo = socket.getaddrinfo
    origins = []

    def observe_denied_lookup(*args, **kwargs):
        origins.append("".join(traceback.format_stack()))
        return denied_getaddrinfo(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", observe_denied_lookup)
    harness = await make_harness(tmp_path)
    try:
        capture = await harness.scanner._capture(harness.key)
        assert capture is not None
        capture.checkpoint.require_attestation()
        before = harness.source.load_source_snapshot(harness.key)
        computed = await harness.scanner._compute(
            harness.key, capture, harness.scanner._policy.auxiliary_limits,
        )
        assert computed.candidate.result.status == "complete"
        assert await harness.scanner._validate(harness.key, capture, computed.candidate)
        assert harness.source.load_source_snapshot(harness.key) == before
    finally:
        await harness.close()
    assert denial.attempts == [], "\n".join(origins)


@pytest.fixture
def offline(monkeypatch):
    denial = NetworkDenial()
    denial.install(monkeypatch)
    yield denial
    assert denial.attempts == []


async def _ready(harness):
    capture = await harness.scanner._capture(harness.key)
    assert type(capture) is IdleScannerCapture
    capture.checkpoint.require_attestation()
    before = harness.source.load_source_snapshot(harness.key)
    computed = await harness.scanner._compute(
        harness.key, capture, harness.scanner._policy.auxiliary_limits,
    )
    computation = computed.candidate
    assert type(computation) is IdleScannerComputation
    assert computation.result.status == "complete"
    assert await harness.scanner._validate(harness.key, capture, computation)
    assert harness.source.load_source_snapshot(harness.key) == before
    return capture, computation


async def test_capture_owns_availability_and_normal_return_observation(tmp_path, offline):
    harness = await make_harness(tmp_path)
    try:
        capture, _ = await _ready(harness)
        assert IdleScannerCapture.require_publication_available(capture) is None
        assert IdleScannerCapture.publication_returned_normally(capture) is False
    finally:
        await harness.close()


async def test_real_publication_completes_once_and_duplicate_is_refused(tmp_path, offline):
    harness = await make_harness(tmp_path)
    try:
        capture, computation = await _ready(harness)
        state = capture._publication
        before = harness.source.load_source_snapshot(harness.key)
        assert (state.consumed, state.completed) == (False, False)
        await harness.scanner._publish(harness.key, capture, computation)
        after = harness.source.load_source_snapshot(harness.key)
        assert after.checkpoint is not None
        assert after.checkpoint["summary"] == "idle summary"
        assert after.events == before.events
        assert (state.consumed, state.completed) == (True, True)
        with pytest.raises(IdleScannerPublicationConflict, match="already consumed"):
            harness.scanner._publish_sync(harness.key, capture, computation, capture.binding)
        assert harness.source.load_source_snapshot(harness.key) == after
        assert harness.records.provider_calls == 1
    finally:
        await harness.close()


@pytest.mark.parametrize("refusal", ["ttl", "blocker", "proposal"])
async def test_preconsumption_refusals_leave_attempt_available(
    tmp_path, monkeypatch, offline, refusal,
):
    harness = await make_harness(tmp_path)
    try:
        capture, computation = await _ready(harness)
        state = capture._publication
        before = harness.source.load_source_snapshot(harness.key)
        if refusal == "ttl":
            monkeypatch.setattr(
                harness.scanner, "_utc_clock",
                lambda: datetime.fromisoformat(harness.activity) + timedelta(seconds=1),
            )
            message = "TTL is no longer current"
        elif refusal == "blocker":
            harness.blocker.block(harness.key)
            message = "runtime blocker state changed"
        else:
            computation = replace(computation, result=replace(computation.result, status="no_progress"))
            message = "proposal is no longer admissible"
        with pytest.raises(IdleScannerPublicationConflict, match=message):
            harness.scanner._publish_sync(harness.key, capture, computation, capture.binding)
        assert (state.consumed, state.completed) == (False, False)
        assert harness.source.load_source_snapshot(harness.key) == before
    finally:
        await harness.close()


@pytest.mark.parametrize("failure", ["policy-unavailable", "policy-read", "proposal-read"])
async def test_late_selections_fail_after_consumption_and_do_not_retry(
    tmp_path, monkeypatch, offline, failure,
):
    harness = await make_harness(tmp_path)
    try:
        capture, computation = await _ready(harness)
        state = capture._publication
        binding = capture.binding
        before = harness.source.load_source_snapshot(harness.key)
        fault = RuntimeError("late selection sentinel")
        if failure.startswith("policy"):
            original = type(binding).__getattribute__

            def policy_read(value, name):
                if value is binding and name == "policy_binding":
                    assert state.consumed and not state.completed
                    if failure == "policy-read":
                        raise fault
                    return None
                return original(value, name)

            monkeypatch.setattr(type(binding), "__getattribute__", policy_read)
        else:
            def proposal_read(value):
                assert value is computation
                assert state.consumed and not state.completed
                raise fault

            monkeypatch.setattr(IdleScannerComputation, "proposal", property(proposal_read))
        if failure == "policy-unavailable":
            with pytest.raises(IdleScannerPublicationConflict, match="policy binding is unavailable"):
                harness.scanner._publish_sync(harness.key, capture, computation, binding)
        else:
            with pytest.raises(RuntimeError) as raised:
                harness.scanner._publish_sync(harness.key, capture, computation, binding)
            assert raised.value is fault
        assert (state.consumed, state.completed) == (True, False)
        with pytest.raises(IdleScannerPublicationConflict, match="already consumed"):
            harness.scanner._publish_sync(harness.key, capture, computation, binding)
        assert harness.source.load_source_snapshot(harness.key) == before
    finally:
        await harness.close()


async def test_real_write_then_raise_is_consumed_but_not_completed(
    tmp_path, monkeypatch, offline,
):
    harness = await make_harness(tmp_path)
    try:
        capture, computation = await _ready(harness)
        state = capture._publication
        original = harness.source.publish_idle_checkpoint
        fault = RuntimeError("real write then raise")
        calls = []

        def write_then_raise(*args, **kwargs):
            assert (state.consumed, state.completed) == (True, False)
            result = original(*args, **kwargs)
            calls.append(result)
            raise fault

        monkeypatch.setattr(harness.source, "publish_idle_checkpoint", write_then_raise)
        with pytest.raises(RuntimeError) as raised:
            await harness.scanner._publish(harness.key, capture, computation)
        assert raised.value is fault
        committed = harness.source.load_source_snapshot(harness.key)
        assert committed.checkpoint is not None
        assert (state.consumed, state.completed) == (True, False)
        with pytest.raises(IdleScannerPublicationConflict, match="already consumed"):
            harness.scanner._publish_sync(harness.key, capture, computation, capture.binding)
        assert harness.source.load_source_snapshot(harness.key) == committed
        assert len(calls) == 1
    finally:
        await harness.close()


async def test_cancellation_before_callback_does_not_consume_or_publish(
    tmp_path, monkeypatch, offline,
):
    harness = await make_harness(tmp_path)
    task = None
    try:
        capture, computation = await _ready(harness)
        state = capture._publication
        before = harness.source.load_source_snapshot(harness.key)
        waiting = asyncio.Event()
        release = asyncio.Event()
        original = harness.registry.publish_if_current

        async def wait_before_callback(*args, **kwargs):
            waiting.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(harness.registry, "publish_if_current", wait_before_callback)
        task = asyncio.create_task(harness.scanner._publish(harness.key, capture, computation))
        await asyncio.wait_for(waiting.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (state.consumed, state.completed) == (False, False)
        assert harness.source.load_source_snapshot(harness.key) == before
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await harness.close()


async def test_real_callback_self_cancellation_retains_completed_publication(
    tmp_path, monkeypatch, offline,
):
    harness = await make_harness(tmp_path)
    try:
        capture, computation = await _ready(harness)
        state = capture._publication
        original = harness.source.publish_idle_checkpoint
        original_registry = harness.registry.publish_if_current

        def write_then_cancel(*args, **kwargs):
            result = original(*args, **kwargs)
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
            return result

        async def deliver_cancellation_after_registry_return(*args, **kwargs):
            result = await original_registry(*args, **kwargs)
            # An uncontended registry can return without suspending. Deliver
            # the callback's pending cancellation at this exact normal-return
            # seam, so the scanner's existing handler is actually exercised.
            await asyncio.sleep(0)
            return result

        monkeypatch.setattr(harness.source, "publish_idle_checkpoint", write_then_cancel)
        monkeypatch.setattr(
            harness.registry, "publish_if_current", deliver_cancellation_after_registry_return,
        )
        task = asyncio.create_task(harness.scanner._publish(harness.key, capture, computation))
        assert await task is None
        assert not task.cancelled()
        assert (state.consumed, state.completed) == (True, True)
        assert harness.source.load_source_snapshot(harness.key).checkpoint is not None
    finally:
        await harness.close()


async def test_original_publication_state_reads_remain_on_their_boundaries(
    tmp_path, monkeypatch, offline,
):
    harness = await make_harness(tmp_path)
    try:
        capture, computation = await _ready(harness)
        state = capture._publication
        original_get = IdleScannerCapture.__getattribute__
        original_publish = harness.source.publish_idle_checkpoint
        original_registry = harness.registry.publish_if_current
        observations = []

        def read(value, name):
            if value is capture and name == "_publication":
                observations.append(("state", state.consumed, state.completed))
            return original_get(value, name)

        def publish(*args, **kwargs):
            observations.append(("i2-enter", state.consumed, state.completed))
            result = original_publish(*args, **kwargs)
            observations.append(("i2-return", state.consumed, state.completed))
            return result

        async def cancel_after_normal_return(*args, **kwargs):
            await original_registry(*args, **kwargs)
            raise asyncio.CancelledError

        monkeypatch.setattr(IdleScannerCapture, "__getattribute__", read)
        monkeypatch.setattr(harness.source, "publish_idle_checkpoint", publish)
        monkeypatch.setattr(harness.registry, "publish_if_current", cancel_after_normal_return)
        assert await harness.scanner._publish(harness.key, capture, computation) is None
        assert observations == [
            ("state", False, False),
            ("state", False, False),
            ("i2-enter", True, False),
            ("i2-return", True, False),
            ("state", True, False),
            ("state", True, True),
        ]
    finally:
        await harness.close()


@pytest.mark.parametrize("ordinal", [1, 2, 3, 4])
async def test_each_original_publication_owner_lookup_preserves_first_fault(
    tmp_path, monkeypatch, offline, ordinal,
):
    harness = await make_harness(tmp_path)
    try:
        capture, computation = await _ready(harness)
        state = capture._publication
        before = harness.source.load_source_snapshot(harness.key)
        original_get = IdleScannerCapture.__getattribute__
        original_registry = harness.registry.publish_if_current
        fault = RuntimeError("publication owner lookup sentinel")
        reads = []

        def read(value, name):
            if value is capture and name == "_publication":
                reads.append((state.consumed, state.completed))
                if len(reads) == ordinal:
                    raise fault
            return original_get(value, name)

        async def cancel_after_normal_return(*args, **kwargs):
            await original_registry(*args, **kwargs)
            raise asyncio.CancelledError

        monkeypatch.setattr(IdleScannerCapture, "__getattribute__", read)
        if ordinal == 4:
            monkeypatch.setattr(harness.registry, "publish_if_current", cancel_after_normal_return)
        with pytest.raises(RuntimeError) as raised:
            await harness.scanner._publish(harness.key, capture, computation)
        assert raised.value is fault
        assert len(reads) == ordinal
        assert state.consumed is (ordinal >= 3)
        assert state.completed is (ordinal == 4)
        after = harness.source.load_source_snapshot(harness.key)
        if ordinal < 3:
            assert after == before
        else:
            assert after.checkpoint is not None
            assert after.events == before.events
    finally:
        await harness.close()


@pytest.mark.parametrize("field_fault", ["availability", "consume", "complete", "observe"])
async def test_publication_field_faults_preserve_original_exception_and_state(
    tmp_path, monkeypatch, offline, field_fault,
):
    harness = await make_harness(tmp_path)
    try:
        capture, computation = await _ready(harness)
        state = capture._publication
        original_get = type(state).__getattribute__
        original_set = type(state).__setattr__
        original_registry = harness.registry.publish_if_current
        fault = RuntimeError("publication field sentinel")

        def read(value, name):
            if value is state and (
                field_fault == "availability" and name == "consumed"
                or field_fault == "observe" and name == "completed"
            ):
                raise fault
            return original_get(value, name)

        def write(value, name, new_value):
            if value is state and (
                field_fault == "consume" and name == "consumed"
                or field_fault == "complete" and name == "completed"
            ):
                raise fault
            return original_set(value, name, new_value)

        async def cancel_after_normal_return(*args, **kwargs):
            await original_registry(*args, **kwargs)
            raise asyncio.CancelledError

        monkeypatch.setattr(type(state), "__getattribute__", read)
        monkeypatch.setattr(type(state), "__setattr__", write)
        if field_fault == "observe":
            monkeypatch.setattr(harness.registry, "publish_if_current", cancel_after_normal_return)
        with pytest.raises(RuntimeError) as raised:
            await harness.scanner._publish(harness.key, capture, computation)
        assert raised.value is fault
        assert original_get(state, "consumed") is (field_fault in {"complete", "observe"})
        assert original_get(state, "completed") is (field_fault == "observe")
        after = harness.source.load_source_snapshot(harness.key)
        assert (after.checkpoint is not None) is (field_fault in {"complete", "observe"})
    finally:
        await harness.close()
