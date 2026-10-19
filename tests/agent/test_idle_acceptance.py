"""Integrated I3f acceptance for the explicitly injected dormant idle path."""

from __future__ import annotations

import asyncio
import copy
import gc
from pathlib import Path

import pytest

from nanobot.agent.idle_compaction import IdleCompactionPolicy, IdleJobOutcome
from nanobot.agent.idle_scanner import IdleCompactionScanner
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import Config
from nanobot.migration_guard import MigrationIncompleteError
from nanobot.session.context_checkpoint import validate_context_checkpoint
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source import prefix_digest, validate_closed_boundary
from tests.agent.idle_acceptance_support import (
    AcceptanceProvider,
    NetworkDenial,
    make_harness,
    source_event,
)


@pytest.fixture(autouse=True)
def deny_external_io(monkeypatch: pytest.MonkeyPatch) -> NetworkDenial:
    denial = NetworkDenial()
    denial.install(monkeypatch)
    return denial


async def _settle(harness) -> None:  # type: ignore[no-untyped-def]
    await harness.scanner.coordinator.drain()
    assert harness.scanner.coordinator.pending_callback_count == 0
    assert harness.scanner.coordinator.admission_count == 0


def _assert_dispatch_measurement_identity(
    harness, *, require_success_usage: bool = False  # type: ignore[no-untyped-def]
) -> tuple[int, int, int, int]:
    records = harness.records
    assert records.measurement_calls == len(records.measurements)
    assert records.measurement_calls == len(records.measurement_records)
    assert records.provider_calls == len(records.attempts)
    assert records.provider_calls == len(records.dispatch_records)
    for observation, measured in zip(records.measurements, records.measurement_records):
        assert measured.observation == observation
        assert measured.receipt.request_digest == observation.digest
        assert measured.expected_usage.input_tokens == measured.receipt.input_tokens
        assert measured.expected_usage.output_tokens == 2
        assert measured.expected_usage.total_tokens == measured.receipt.input_tokens + 2

    dispatched_input = dispatched_output = dispatched_total = 0
    for observation, dispatched in zip(records.attempts, records.dispatch_records):
        assert dispatched.observation == observation
        assert dispatched.receipt.request_digest == observation.digest
        matches = [
            measured
            for measured in records.measurement_records
            if measured.receipt.request_digest == dispatched.receipt.request_digest
        ]
        assert matches
        assert all(measured.receipt == matches[0].receipt for measured in matches)
        assert dispatched.receipt == matches[-1].receipt
        if require_success_usage:
            assert dispatched.attempts == 1
            assert dispatched.usage == matches[-1].expected_usage
        if dispatched.usage is not None:
            dispatched_input += dispatched.usage.input_tokens
            dispatched_output += dispatched.usage.output_tokens
            dispatched_total += dispatched.usage.total_tokens
    return len(records.dispatch_records), dispatched_input, dispatched_output, dispatched_total


def _assert_request_owners_released(harness) -> None:  # type: ignore[no-untyped-def]
    gc.collect()
    assert harness.records.request_refs
    assert all(request_ref() is None for request_ref in harness.records.request_refs)


async def test_dependency_free_multichunk_lifecycle_reload_and_rescan(deny_external_io, tmp_path: Path) -> None:
    harness = await make_harness(tmp_path, multi_chunk=True)
    try:
        before = harness.source.load_source_snapshot(harness.key)
        before_bytes = harness.backend.get_session_path(harness.key).read_bytes()
        before_activity = copy.deepcopy(before.header["metadata"])

        assert await harness.lifecycle.scan_once() == 1
        await _settle(harness)

        after = harness.source.load_source_snapshot(harness.key)
        checkpoint = after.checkpoint
        assert checkpoint is not None
        assert harness.scanner.coordinator.outcome_for(harness.key) is IdleJobOutcome.SUCCEEDED
        assert checkpoint["schema_version"] == 1
        assert checkpoint["parent_checkpoint_id"] is None
        assert checkpoint["boundary"] == len(after.events)
        assert checkpoint["boundary_event_id"] == after.events[-1]["_event_id"]
        assert checkpoint["prefix_digest"] == prefix_digest(after.events, checkpoint["boundary"])
        assert validate_closed_boundary(after.events, checkpoint["boundary"]) is None
        assert validate_context_checkpoint(
            checkpoint,
            after.events,
            generation=after.state.generation,
            source_revision=before.state.revision,
            parent_checkpoint_id=None,
        ) == checkpoint
        assert after.events == before.events
        assert after.header["metadata"]["last_user_activity_at"] == before_activity["last_user_activity_at"]
        assert after.header["metadata"]["user_activity_revision"] == before_activity[
            "user_activity_revision"
        ]
        assert checkpoint["producing_provider"] == "acceptance-provider"
        assert checkpoint["producing_model"] == "acceptance-model"
        calls, input_tokens, output_tokens, total_tokens = _assert_dispatch_measurement_identity(
            harness, require_success_usage=True
        )
        usage = checkpoint["auxiliary_usage"]
        assert usage == {
            "calls": calls,
            "tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        assert harness.records.measurement_calls == 7
        assert (calls, input_tokens, output_tokens, total_tokens) == (2, 200, 4, 204)
        assert harness.records.max_active_calls == 1
        assert not harness.records.forbidden_seen

        committed_bytes = harness.backend.get_session_path(harness.key).read_bytes()
        assert committed_bytes != before_bytes
        checkpoint_id = checkpoint["checkpoint_id"]
        assert await harness.lifecycle.scan_once() == 1
        await _settle(harness)
        rescanned = harness.source.load_source_snapshot(harness.key)
        assert rescanned.checkpoint is not None
        assert rescanned.checkpoint["checkpoint_id"] == checkpoint_id
        assert harness.records.provider_calls == 2
        assert harness.backend.get_session_path(harness.key).read_bytes() == committed_bytes
    finally:
        await harness.close()
    assert deny_external_io.attempts == []


async def test_projection_only_root_artifact_publishes_schema_v2_without_tail_or_child_disclosure(
    deny_external_io, tmp_path: Path
) -> None:
    harness = await make_harness(tmp_path, artifact=True)
    try:
        before = harness.source.load_source_snapshot(harness.key)
        manifest = copy.deepcopy(before.header["metadata"]["artifact_manifest"])
        assert harness.descriptor is not None
        assert not any("_artifact_read" in event for event in before.events)

        assert await harness.lifecycle.scan_once() == 1
        await _settle(harness)
        after = harness.source.load_source_snapshot(harness.key)
        checkpoint = after.checkpoint
        assert checkpoint is not None
        assert checkpoint["schema_version"] == 2
        assert checkpoint["artifact_references"] == [harness.descriptor.reference]
        assert checkpoint["projection_scope"] == "root"
        assert checkpoint["projection_version"] == 1
        assert after.events == before.events
        assert after.header["metadata"]["artifact_manifest"] == manifest
        calls, input_tokens, output_tokens, total_tokens = _assert_dispatch_measurement_identity(
            harness, require_success_usage=True
        )
        assert any(harness.descriptor.reference in row.references for row in harness.records.measurements)
        assert any(harness.descriptor.reference in row.references for row in harness.records.attempts)
        assert all(not row.saw_full_tail for row in harness.records.measurements)
        assert all(not row.saw_full_tail for row in harness.records.attempts)
        assert not harness.records.forbidden_seen
        assert (calls, input_tokens, output_tokens, total_tokens) == (1, input_tokens, 2, input_tokens + 2)
        assert checkpoint["auxiliary_usage"] == {
            "calls": calls,
            "tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        assert harness.records.measurement_calls == 3
        assert harness.records.max_active_calls == 1
    finally:
        await harness.close()
    assert deny_external_io.attempts == []


async def test_retrieval_and_prior_checkpoint_continuity_keep_reference_and_event_identity(
    deny_external_io, tmp_path: Path
) -> None:
    harness = await make_harness(tmp_path, artifact=True, retrieval=True)
    try:
        assert await harness.lifecycle.scan_once() == 1
        await _settle(harness)
        first_calls, first_input, first_output, first_total = _assert_dispatch_measurement_identity(
            harness, require_success_usage=True
        )
        assert harness.records.measurement_calls == 4
        first = harness.source.load_source_snapshot(harness.key)
        first_checkpoint = first.checkpoint
        assert first_checkpoint is not None
        assert first_checkpoint["schema_version"] == 2
        assert first_checkpoint["artifact_references"] == [harness.descriptor.reference]
        assert any("_artifact_read" in event for event in first.events)
        assert first_checkpoint["auxiliary_usage"] == {
            "calls": first_calls,
            "tokens": first_total,
            "input_tokens": first_input,
            "output_tokens": first_output,
            "total_tokens": first_total,
        }

        appended = source_event(harness.key, "later authorized root evidence")
        harness.source.append_source_events(harness.key, first.state, [appended])
        assert await harness.lifecycle.scan_once() == 1
        await _settle(harness)
        second = harness.source.load_source_snapshot(harness.key)
        checkpoint = second.checkpoint
        assert checkpoint is not None
        assert checkpoint["parent_checkpoint_id"] == first_checkpoint["checkpoint_id"]
        assert checkpoint["boundary"] > first_checkpoint["boundary"]
        assert checkpoint["artifact_references"] == [harness.descriptor.reference]
        assert second.events[: len(first.events)] == first.events
        assert second.events[-1] == appended
        calls, input_tokens, output_tokens, total_tokens = _assert_dispatch_measurement_identity(
            harness, require_success_usage=True
        )
        assert (calls, output_tokens) == (2, 4)
        assert checkpoint["auxiliary_usage"] == {
            "calls": calls - first_calls,
            "tokens": total_tokens - first_total,
            "input_tokens": input_tokens - first_input,
            "output_tokens": output_tokens - first_output,
            "total_tokens": total_tokens - first_total,
        }
        assert harness.records.measurement_calls == 6
        assert any(harness.descriptor.reference in row.references for row in harness.records.attempts)
        assert not harness.records.forbidden_seen
    finally:
        await harness.close()
    assert deny_external_io.attempts == []


async def test_foreground_activity_invalidates_inflight_transport_before_publication(
    deny_external_io, tmp_path: Path
) -> None:
    harness = await make_harness(tmp_path, wait_for_release=True)
    try:
        before = harness.source.load_source_snapshot(harness.key)
        assert await harness.lifecycle.scan_once() == 1
        await harness.records.started.wait()
        admitted = harness.source.record_user_activity(
            harness.key,
            harness.source.load_source_snapshot(harness.key).state,
            accepted_foreground_user=True,
        )
        await harness.lifecycle.foreground_admitted(harness.key)
        await harness.records.cancelled.wait()
        harness.records.release.set()
        await _settle(harness)
        after = harness.source.load_source_snapshot(harness.key)
        assert harness.scanner.coordinator.outcome_for(harness.key) is IdleJobOutcome.CANCELLED
        assert after.checkpoint is None
        assert after.events == before.events
        assert after.header["metadata"]["user_activity_revision"] == (
            admitted.header["metadata"]["user_activity_revision"]
        )
        assert harness.records.provider_calls == 1
        assert harness.records.cancellation_seen >= 1
        assert harness.records.active_calls == 0
        _assert_dispatch_measurement_identity(harness)
        assert harness.records.measurement_calls == 3
        _assert_request_owners_released(harness)
    finally:
        harness.records.release.set()
        await harness.close()
    assert deny_external_io.attempts == []


async def test_blocker_rejection_admits_no_capture_or_provider_work(deny_external_io, tmp_path: Path) -> None:
    harness = await make_harness(tmp_path)
    try:
        harness.blocker.block(harness.key)
        before = harness.source.load_source_snapshot(harness.key)
        assert await harness.lifecycle.scan_once() == 0
        await _settle(harness)
        assert harness.scanner.coordinator.outcome_for(harness.key) is IdleJobOutcome.INELIGIBLE
        assert harness.records.provider_calls == 0
        assert harness.records.measurements == []
        assert harness.source.load_source_snapshot(harness.key) == before
    finally:
        await harness.close()
    assert deny_external_io.attempts == []


@pytest.mark.parametrize(
    ("mode", "outcome"),
    [
        ("unknown-usage", IdleJobOutcome.FAILED),
        ("malformed-response", IdleJobOutcome.FAILED),
        ("attempts-2", IdleJobOutcome.FAILED),
        ("error", IdleJobOutcome.FAILED),
    ],
)
async def test_failure_accounting_is_terminal_without_retry_or_orphan(
    deny_external_io, tmp_path: Path, mode: str, outcome: IdleJobOutcome
) -> None:
    harness = await make_harness(tmp_path, response_mode=mode)
    try:
        before_snapshot = harness.source.load_source_snapshot(harness.key)
        before = harness.backend.get_session_path(harness.key).read_bytes()
        assert await harness.lifecycle.scan_once() == 1
        await _settle(harness)
        after = harness.source.load_source_snapshot(harness.key)
        assert harness.scanner.coordinator.outcome_for(harness.key) is outcome
        assert after.checkpoint is None
        assert after.events == before_snapshot.events
        _assert_dispatch_measurement_identity(harness)
        assert harness.records.measurement_calls == 3
        assert harness.records.provider_calls == 1
        assert harness.records.active_calls == 0
        assert harness.backend.get_session_path(harness.key).read_bytes() == before
        assert harness.scanner.coordinator.pending_callback_count == 0
        assert not harness.records.forbidden_seen
        _assert_request_owners_released(harness)
    finally:
        await harness.close()
    assert deny_external_io.attempts == []


async def test_blocker_change_at_prepublication_gate_discards_stale_proposal(
    deny_external_io, tmp_path: Path
) -> None:
    harness = await make_harness(tmp_path)
    harness.records.hold_before_result = True
    try:
        before = harness.source.load_source_snapshot(harness.key)
        assert await harness.lifecycle.scan_once() == 1
        await harness.records.transport_completed.wait()
        harness.blocker.block(harness.key)
        harness.records.release_publication.set()
        await _settle(harness)
        assert harness.scanner.coordinator.outcome_for(harness.key) is IdleJobOutcome.DISCARDED
        assert harness.source.load_source_snapshot(harness.key).checkpoint is None
        assert harness.source.load_source_snapshot(harness.key).events == before.events
        assert harness.records.provider_calls == 1
    finally:
        harness.records.release_publication.set()
        await harness.close()
    assert deny_external_io.attempts == []


async def test_runtime_epoch_replacement_prevents_stale_publish(deny_external_io, tmp_path: Path) -> None:
    harness = await make_harness(tmp_path)
    harness.records.hold_before_result = True
    try:
        assert await harness.lifecycle.scan_once() == 1
        await harness.records.transport_completed.wait()
        harness.registry.replace_config()
        harness.records.release_publication.set()
        await _settle(harness)
        assert harness.scanner.coordinator.outcome_for(harness.key) is IdleJobOutcome.DISCARDED
        assert harness.source.load_source_snapshot(harness.key).checkpoint is None
        assert harness.records.provider_calls == 1
    finally:
        harness.records.release_publication.set()
        await harness.close()
    assert deny_external_io.attempts == []


async def test_lifecycle_close_joins_transport_and_releases_request_owners(
    deny_external_io, tmp_path: Path
) -> None:
    harness = await make_harness(tmp_path, wait_for_release=True)
    try:
        assert await harness.lifecycle.scan_once() == 1
        await harness.records.started.wait()
        close_task = asyncio.create_task(harness.lifecycle.aclose())
        await harness.records.cancelled.wait()
        harness.records.release.set()
        await close_task
        assert harness.lifecycle.closed
        assert harness.scanner.closed
        assert harness.registry.closed
        assert harness.scanner.discovery is not None and harness.scanner.discovery.closed
        assert harness.scanner.coordinator.pending_callback_count == 0
        gc.collect()
        assert all(request_ref() is None for request_ref in harness.records.request_refs)
    finally:
        harness.records.release.set()
        await harness.close()
    assert deny_external_io.attempts == []


async def test_production_guard_and_unpatched_idle_factory_remain_closed(
    deny_external_io, tmp_path: Path
) -> None:
    from nanobot.migration_guard import require_runtime_ready
    from nanobot.runtime_composition import RuntimePreflightError

    with pytest.raises(MigrationIncompleteError):
        require_runtime_ready()
    config = Config()
    config.bind_source_path(tmp_path / "config.json")
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    config.agents.defaults.idle_compaction.enabled = True
    with pytest.raises(RuntimePreflightError, match="idle compaction"):
        AgentLoop.from_config(
            config,
            tool_registry=ToolRegistry(),
            provider=AcceptanceProvider(),
        )

    backend = JsonlSessionStore(tmp_path / "plain-workspace", sessions_root=tmp_path / "plain-sessions")
    scanner = IdleCompactionScanner(
        source_store=backend.source,
        policy=IdleCompactionPolicy(),
    )
    try:
        assert scanner.enabled is False
        assert await scanner.scan_once() == 0
        assert scanner.coordinator.worker_count == 0
    finally:
        await scanner.aclose()
    assert deny_external_io.attempts == []
