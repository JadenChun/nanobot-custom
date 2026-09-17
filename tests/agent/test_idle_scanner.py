"""Focused dormant I3d composition checks."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import shutil
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from filelock import FileLock
from session.test_artifact_reader import setup_read
from session.test_idle_checkpoint import _hold_artifact_store

from agent.test_idle_summary import _adapter, _Attempt, _authority, _limits, _Provider
from agent.test_source_dependency_compaction import _record_retrieval
from nanobot.agent.idle_compaction import IdleCompactionPolicy
from nanobot.agent.idle_runtime import IdleRuntimeBindingRegistry
from nanobot.agent.idle_scanner import (
    IdleBlockerSnapshot,
    IdleCompactionScanner,
    IdleScannerIneligible,
    InMemoryIdleBlockerAuthority,
)
from nanobot.agent.idle_summary import IdleDependencyVerification, IdleSummaryBindingError
from nanobot.providers.base import GenerationSettings
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.idle_checkpoint import IdlePolicyBinding
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source_discovery import SourceDiscoveryLimits
from nanobot.session.source_store import SourceSessionStore
from nanobot.utils.llm_runtime import LLMRuntime


def _enabled_policy(*, max_auxiliary_tokens: int = 1000) -> IdleCompactionPolicy:
    return IdleCompactionPolicy(
        idle_ttl_seconds=30.0,
        queue_capacity=1,
        max_concurrent_jobs=1,
        max_auxiliary_calls=1,
        max_auxiliary_tokens=max_auxiliary_tokens,
        max_auxiliary_wall_seconds=5.0,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
    )


async def _dependency_fixture(
    tmp_path: Path,
    *,
    retrieval: bool = True,
    text: str = "evidence 中文 🌟",
):
    """Build a released source with a published root artifact."""

    async with setup_read(tmp_path, text) as (writer, tool, store, descriptor, _):
        writer.append_message({"role": "user", "content": "idle objective"})
        reader = ArtifactReader(writer, "root_history", store, tool)
        if retrieval:
            await _record_retrieval(writer, reader, descriptor.reference)
        backend = writer.store._backend
        key = writer.owner.key

    source = SourceSessionStore(backend, artifact_store=store)
    policy = IdlePolicyBinding(
        "policy-1",
        tool._workspace,
        None,
        lambda value: tool._workspace / value,
    )
    source.record_user_activity(
        key,
        source.load_source_snapshot(key).state,
        accepted_foreground_user=True,
    )
    provider = _Provider()
    runtime = LLMRuntime(
        provider,
        "fixture",
        GenerationSettings(max_tokens=256),
        4096,
        snapshot_signature=("fixture", "fixture"),
    )
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=runtime,
        auxiliary_runtime=runtime,
        workspace=tool._workspace,
        artifact_namespace="artifact-ns-1",
        policy_binding=policy,
    )
    binding = registry.issue(key)
    activity = source.load_source_snapshot(key).header["metadata"]["last_user_activity_at"]
    return source, store, descriptor, policy, registry, binding, activity


class _ScannerDependencyVerifier:
    def __init__(self, source, store, policy, *, verify_storage: bool = True) -> None:
        self.source = source
        self.store = store
        self.policy = policy
        self.verify_storage = verify_storage
        self.calls: list[tuple[str, ...]] = []

    def verify(self, *, capture, binding, references):  # type: ignore[no-untyped-def]
        self.calls.append(references)
        if self.verify_storage:
            with self.store.transaction(blocking=False) as blobs:
                snapshot = self.source.load_source_snapshot(binding.session_key)
                if capture.dependencies is None:
                    raise AssertionError("dependency capture did not retain artifact authority")
                for reference in references:
                    capture.dependencies._verify(reference, snapshot, blobs, self.policy)
                blobs.assert_reachable()
        return IdleDependencyVerification(
            capture,
            binding,
            references,
            self.policy.fingerprint,
            binding.artifact_namespace,
        )


@pytest.mark.asyncio
async def test_disabled_scanner_is_construction_inert(tmp_path: Path) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    source = backend.source
    before = {task for task in asyncio.all_tasks() if not task.done()}
    scanner = IdleCompactionScanner(
        source_store=source,
        policy=IdleCompactionPolicy(),
    )
    after = {task for task in asyncio.all_tasks() if not task.done()}
    assert before == after
    assert await scanner.scan_once() == 0
    await scanner.aclose()


def test_blocker_authority_is_revisioned_and_missing_state_fails_closed() -> None:
    authority = InMemoryIdleBlockerAuthority()
    with pytest.raises(Exception):
        authority.snapshot("missing")
    first = authority.clear("session")
    second = authority.block("session")
    third = authority.clear("session")
    assert first == IdleBlockerSnapshot(0, True)
    assert second == IdleBlockerSnapshot(1, False)
    assert third == IdleBlockerSnapshot(2, True)


@pytest.mark.asyncio
async def test_scanner_wires_i2_i3c_and_publishes_once(tmp_path: Path) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    key = binding.session_key
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000, max_wall_seconds=5.0),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(key)
    activity = source.load_source_snapshot(key).header["metadata"]["last_user_activity_at"]
    def activity_clock() -> datetime:
        return datetime.fromisoformat(activity) + timedelta(seconds=30)
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=activity_clock,
    )
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        snapshot = source.load_source_snapshot(key)
        assert snapshot.checkpoint is not None
        assert len(attempt.requests) == 1
        assert scanner.coordinator.outcome_for(key).value == "succeeded"
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_overlapping_scans_share_one_discovery_and_provider_spend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000, max_wall_seconds=5.0),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    assert scanner.discovery is not None
    original_scan = scanner.discovery.scan_once
    discovery_started = asyncio.Event()
    release_discovery = asyncio.Event()

    async def delayed_scan():  # type: ignore[no-untyped-def]
        discovery_started.set()
        await release_discovery.wait()
        return await original_scan()

    monkeypatch.setattr(scanner.discovery, "scan_once", delayed_scan)
    try:
        first = asyncio.create_task(scanner.scan_once())
        await discovery_started.wait()
        second = asyncio.create_task(scanner.scan_once())
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        release_discovery.set()
        assert await first == 1
        assert await second == 1
        await scanner.coordinator.drain()
        assert len(attempt.requests) == 1
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_dependency_bearing_scanner_publishes_schema_v2_proof_once(tmp_path: Path) -> None:
    source, store, descriptor, policy, registry, binding, activity = await _dependency_fixture(
        tmp_path
    )
    attempt = _Attempt(["offline summary"])
    verifier = _ScannerDependencyVerifier(source, store, policy)
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=10_000, max_wall_seconds=5.0),
        dependency_verifier=verifier,
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(max_auxiliary_tokens=10_000),
        discovery_limits=SourceDiscoveryLimits(100, 100, 4000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        snapshot = source.load_source_snapshot(binding.session_key)
        assert scanner.coordinator.outcome_for(binding.session_key).value == "succeeded"
        assert snapshot.checkpoint is not None
        assert snapshot.checkpoint["schema_version"] == 2
        assert snapshot.checkpoint["artifact_references"] == [descriptor.reference]
        assert len(attempt.requests) == 1
        assert verifier.calls and verifier.calls[-1] == (descriptor.reference,)
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_root_artifact_projection_scanner_without_retrieval_is_compact_and_authorized(
    tmp_path: Path,
) -> None:
    source, store, descriptor, policy, registry, binding, activity = await _dependency_fixture(
        tmp_path,
        retrieval=False,
        text="preview head\n" + ("a" * 2000) + "\nunique full-tail sentinel",
    )
    before = source.load_source_snapshot(binding.session_key)
    attempt = _Attempt(["offline summary"])
    verifier = _ScannerDependencyVerifier(source, store, policy)
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=10_000, max_wall_seconds=5.0),
        dependency_verifier=verifier,
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(max_auxiliary_tokens=10_000),
        discovery_limits=SourceDiscoveryLimits(100, 100, 4000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        after = source.load_source_snapshot(binding.session_key)
        assert scanner.coordinator.outcome_for(binding.session_key).value == "succeeded"
        assert after.checkpoint is not None
        assert after.checkpoint["schema_version"] == 2
        assert after.checkpoint["artifact_references"] == [descriptor.reference]
        assert verifier.calls and verifier.calls[-1] == (descriptor.reference,)
        assert len(attempt.requests) == 1
        measured_text = str(attempt.measurements)
        assert descriptor.reference in measured_text
        assert "Incomplete preview" in measured_text
        assert "unique full-tail sentinel" not in measured_text
        assert after.events == before.events
        assert not any("_artifact_read" in event for event in after.events)
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_forged_idle_capture_cannot_drop_manifest_before_i3c_measurement(
    tmp_path: Path,
) -> None:
    source, store, _, policy, registry, binding, _ = await _dependency_fixture(
        tmp_path,
        retrieval=False,
        text=("preview head\n" + ("a" * 2000) + "\nSECRET_TAIL"),
    )
    capture = source.capture_idle_checkpoint(
        binding.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None and capture.dependencies is not None
    forged_snapshot = deepcopy(capture.snapshot)
    forged_snapshot.header["metadata"].pop("artifact_manifest")
    forged = replace(capture, snapshot=forged_snapshot, dependencies=None)
    before = source.load_source_snapshot(binding.session_key)
    attempt = _Attempt(["offline summary"])
    verifier = _ScannerDependencyVerifier(source, store, policy)
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=10_000, max_wall_seconds=5.0),
        dependency_verifier=verifier,
    )
    with pytest.raises(IdleSummaryBindingError):
        await adapter.summarize(
            forged, binding, deadline=asyncio.get_running_loop().time() + 2
        )
    assert attempt.measurements == []
    assert attempt.requests == []
    assert verifier.calls == []
    after = source.load_source_snapshot(binding.session_key)
    assert after == before


@pytest.mark.asyncio
@pytest.mark.parametrize("retrieval", [True, False])
async def test_dependency_mutation_rejects_before_i3c_dispatch_without_source_write(
    tmp_path: Path,
    retrieval: bool,
) -> None:
    source, store, _, policy, registry, binding, activity = await _dependency_fixture(
        tmp_path, retrieval=retrieval
    )
    before = source._backend.get_session_path(binding.session_key).read_bytes()
    next(store.root.rglob("*.blob")).write_bytes(b"corrupt")
    attempt = _Attempt(["offline summary"])
    verifier = _ScannerDependencyVerifier(source, store, policy)
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=10_000, max_wall_seconds=5.0),
        dependency_verifier=verifier,
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(max_auxiliary_tokens=10_000),
        discovery_limits=SourceDiscoveryLimits(100, 100, 4000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "failed"
        assert verifier.calls and verifier.calls[-1]
        assert attempt.requests == []
        assert source._backend.get_session_path(binding.session_key).read_bytes() == before
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_root_projection_policy_denial_discloses_no_artifact_or_provider_request(
    tmp_path: Path,
) -> None:
    source, store, _, policy, registry, binding, activity = await _dependency_fixture(
        tmp_path,
        retrieval=False,
        text="preview head\n" + ("a" * 2000) + "\nSECRET_TAIL",
    )

    def deny(_value: str) -> Path:
        raise PermissionError("policy denied")

    denied = IdlePolicyBinding(policy.fingerprint, policy.workspace, None, deny)
    attempt = _Attempt(["offline summary"])
    verifier = _ScannerDependencyVerifier(source, store, denied)
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=10_000, max_wall_seconds=5.0),
        dependency_verifier=verifier,
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(max_auxiliary_tokens=10_000),
        discovery_limits=SourceDiscoveryLimits(100, 100, 4000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "failed"
        assert attempt.requests == []
        assert "SECRET_TAIL" not in str(attempt.measurements)
        assert source.load_source_snapshot(binding.session_key).checkpoint is None
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_root_projection_reference_deduplicates_across_multi_chunk_fold(
    tmp_path: Path,
) -> None:
    source, store, descriptor, policy, registry, binding, _ = await _dependency_fixture(
        tmp_path,
        retrieval=False,
        text="preview head\n" + ("a" * 2000) + "\nSECRET_TAIL",
    )
    capture = source.capture_idle_checkpoint(
        binding.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None

    class SplitAttempt(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            combined = "Tool output stored as" in str(messages) and "idle objective" in str(messages)
            input_tokens = 1_100 if combined else 100
            return replace(
                receipt,
                input_tokens=input_tokens,
                total_reservation_tokens=input_tokens + max_output_tokens + safety_buffer,
            )

    attempt = SplitAttempt(["first", "second"])
    verifier = _ScannerDependencyVerifier(source, store, policy)
    result = await _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_input_tokens=1_000, max_chunks=2, max_total_tokens=10_000),
        dependency_verifier=verifier,
    ).summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert result.status == "complete" and result.proposal is not None
    assert result.proposal.checkpoint["artifact_references"] == [descriptor.reference]
    assert len(result.proposal.proof.chunks) == 2
    assert verifier.calls and all(call == (descriptor.reference,) for call in verifier.calls if call)
    assert "SECRET_TAIL" not in str(attempt.measurements)


@pytest.mark.asyncio
async def test_dependency_contention_discards_before_source_write(tmp_path: Path) -> None:
    source, store, _, policy, registry, binding, activity = await _dependency_fixture(tmp_path)
    before = source._backend.get_session_path(binding.session_key).read_bytes()
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_artifact_store,
        args=(
            str(store.root),
            store.workspace_id,
            store.session_key,
            store.incarnation,
            store.generation,
            entered,
            release,
        ),
    )
    holder.start()
    attempt = _Attempt(["offline summary"])
    verifier = _ScannerDependencyVerifier(source, store, policy, verify_storage=False)
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=10_000, max_wall_seconds=5.0),
        dependency_verifier=verifier,
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(max_auxiliary_tokens=10_000),
        discovery_limits=SourceDiscoveryLimits(100, 100, 4000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "discarded"
        assert verifier.calls and verifier.calls[-1]
        assert len(attempt.requests) == 1
        assert source._backend.get_session_path(binding.session_key).read_bytes() == before
    finally:
        release.set()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
        await scanner.aclose()
    assert holder.exitcode == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_calls", "max_total_tokens", "max_wall_seconds"),
    [(2, 1000, 5.0), (1, 1001, 5.0), (1, 1000, 5.1)],
)
async def test_incompatible_i3c_ceiling_is_rejected_before_measurement(
    tmp_path: Path,
    max_calls: int,
    max_total_tokens: int,
    max_wall_seconds: float,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(
            _limits(),
            max_calls=max_calls,
            max_total_tokens=max_total_tokens,
            max_wall_seconds=max_wall_seconds,
        ),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    before = source.load_source_snapshot(binding.session_key)
    try:
        assert await scanner.scan_once() == 0
        await scanner.coordinator.drain()
        assert attempt.measurements == []
        assert attempt.requests == []
        assert source.load_source_snapshot(binding.session_key) == before
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_adapter_ceiling_mutation_is_rejected_before_measurement(tmp_path: Path) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000, max_wall_seconds=5.0),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        captured = await scanner._capture(binding.session_key)
        assert captured is not None
        adapter._limits = replace(adapter.limits, max_wall_seconds=5.1)
        with pytest.raises(IdleScannerIneligible):
            await scanner._compute(
                binding.session_key,
                captured,
                scanner._policy.auxiliary_limits,
            )
        assert attempt.measurements == []
        assert attempt.requests == []
    finally:
        await scanner.aclose()


def _replace_source_root_with_copy(source_root: Path) -> tuple[Path, dict[str, bytes]]:
    moved = source_root.with_name(f"{source_root.name}-original")
    os.rename(source_root, moved)
    shutil.copytree(moved, source_root)
    contents = {
        str(path.relative_to(source_root)): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    return moved, contents


@pytest.mark.asyncio
async def test_root_replacement_before_publication_rejects_and_preserves_copy(
    tmp_path: Path,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    attempt.wait = True
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        assert await scanner.scan_once() == 1
        await attempt.started.wait()
        _, replacement = _replace_source_root_with_copy(source._backend.sessions_dir)
        attempt.release.set()
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "discarded"
        assert {
            str(path.relative_to(source._backend.sessions_dir)): path.read_bytes()
            for path in source._backend.sessions_dir.rglob("*")
            if path.is_file()
        } == replacement
        assert scanner.discovery is not None and scanner.discovery.closed
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_descriptor_replace_race_commits_only_to_original_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    authority = scanner.root_authority
    assert authority is not None
    root = source._backend.sessions_dir
    original_write = authority.write_text_atomic
    race: dict[str, object] = {}

    def replace_before_write(name: str, text: str, *, mode: int = 0o600) -> None:
        moved, replacement = _replace_source_root_with_copy(root)
        race["moved"] = moved
        race["replacement"] = replacement
        original_write(name, text, mode=mode)

    monkeypatch.setattr(authority, "write_text_atomic", replace_before_write)
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "succeeded"
        replacement = race["replacement"]
        assert isinstance(replacement, dict)
        assert {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        } == replacement
        moved = race["moved"]
        assert isinstance(moved, Path)
        assert moved.joinpath(
            f"{source._backend.storage_key(binding.session_key)}.jsonl"
        ).read_bytes() != replacement[
            f"{source._backend.storage_key(binding.session_key)}.jsonl"
        ]
        assert b'"context_checkpoint"' in moved.joinpath(
            f"{source._backend.storage_key(binding.session_key)}.jsonl"
        ).read_bytes()
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_root_replacement_during_execution_lock_acquisition_cannot_redirect_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    authority = scanner.root_authority
    assert authority is not None
    root = source._backend.sessions_dir
    original_lock = authority.lock_nested_file
    race: dict[str, object] = {}

    def replace_before_lock(directory: str, name: str, *, blocking: bool):
        moved, replacement = _replace_source_root_with_copy(root)
        race["moved"] = moved
        race["replacement"] = replacement
        return original_lock(directory, name, blocking=blocking)

    monkeypatch.setattr(authority, "lock_nested_file", replace_before_lock)
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value in {
            "succeeded",
            "discarded",
        }
        assert "replacement" in race and "moved" in race
        replacement = race["replacement"]
        assert isinstance(replacement, dict)
        assert {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        } == replacement
        moved = race["moved"]
        assert isinstance(moved, Path)
        moved_bytes = moved.joinpath(
            f"{source._backend.storage_key(binding.session_key)}.jsonl"
        ).read_bytes()
        if scanner.coordinator.outcome_for(binding.session_key).value == "succeeded":
            assert b'"context_checkpoint"' in moved_bytes
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_close_joins_i3c_and_closes_root_descriptor(
    tmp_path: Path,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    attempt.wait = True
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    authority = scanner.root_authority
    assert authority is not None
    descriptor = authority._fd
    try:
        assert await scanner.scan_once() == 1
        await attempt.started.wait()
        closing = asyncio.create_task(scanner.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        attempt.release.set()
        await closing
        assert authority.closed
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert scanner.discovery is not None and scanner.discovery.closed
        assert scanner.coordinator.pending_callback_count == 0
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_publication_barrier_rechecks_blocker_and_binding_before_i2_write(
    tmp_path: Path,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    attempt.wait = True
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        assert await scanner.scan_once() == 1
        await attempt.started.wait()
        blockers.block(binding.session_key)
        registry.invalidate(session_key=binding.session_key)
        attempt.release.set()
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "discarded"
        assert source.load_source_snapshot(binding.session_key).checkpoint is None
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_blocker_revision_change_at_publication_barrier_preserves_current_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    publication_waiting = asyncio.Event()
    release_publication = asyncio.Event()
    original_publish = registry.publish_if_current

    async def blocked_publish(*args: object, **kwargs: object) -> object:
        publication_waiting.set()
        await release_publication.wait()
        return await original_publish(*args, **kwargs)

    monkeypatch.setattr(registry, "publish_if_current", blocked_publish)
    before = source.load_source_snapshot(binding.session_key)
    try:
        assert await scanner.scan_once() == 1
        await publication_waiting.wait()
        blockers.block(binding.session_key)
        assert registry.revalidate(binding, binding.session_key)
        release_publication.set()
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "discarded"
        assert registry.revalidate(binding, binding.session_key)
        assert source.load_source_snapshot(binding.session_key) == before
    finally:
        release_publication.set()
        await scanner.aclose()


@pytest.mark.asyncio
async def test_registry_invalidation_at_publication_barrier_preserves_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    publication_waiting = asyncio.Event()
    release_publication = asyncio.Event()
    original_publish = registry.publish_if_current

    async def blocked_publish(*args: object, **kwargs: object) -> object:
        publication_waiting.set()
        await release_publication.wait()
        return await original_publish(*args, **kwargs)

    monkeypatch.setattr(registry, "publish_if_current", blocked_publish)
    before = source.load_source_snapshot(binding.session_key)
    try:
        assert await scanner.scan_once() == 1
        await publication_waiting.wait()
        registry.invalidate(session_key=binding.session_key)
        assert not registry.revalidate(binding, binding.session_key)
        release_publication.set()
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "discarded"
        assert source.load_source_snapshot(binding.session_key) == before
    finally:
        release_publication.set()
        await scanner.aclose()


@pytest.mark.asyncio
async def test_foreground_admission_at_publication_boundary_cancels_without_source_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    publication_waiting = asyncio.Event()
    release_publication = asyncio.Event()
    original_publish = registry.publish_if_current

    async def blocked_publish(*args: object, **kwargs: object) -> object:
        publication_waiting.set()
        await release_publication.wait()
        return await original_publish(*args, **kwargs)

    monkeypatch.setattr(registry, "publish_if_current", blocked_publish)
    before = source.load_source_snapshot(binding.session_key)
    try:
        assert await scanner.scan_once() == 1
        await publication_waiting.wait()
        await scanner.foreground_admitted(binding.session_key)
        release_publication.set()
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value in {
            "cancelled",
            "discarded",
        }
        assert source.load_source_snapshot(binding.session_key) == before
    finally:
        release_publication.set()
        await scanner.aclose()


@pytest.mark.asyncio
async def test_foreground_execution_lock_contention_discards_without_source_write(
    tmp_path: Path,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    lock_path = source._backend.sessions_dir / ".execution-locks" / (
        f"{source._backend.storage_key(binding.session_key)}.lock"
    )
    lock = FileLock(str(lock_path), timeout=0)
    lock.acquire()
    before = source.load_source_snapshot(binding.session_key)
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "discarded"
        assert source.load_source_snapshot(binding.session_key) == before
    finally:
        lock.release()
        await scanner.aclose()


@pytest.mark.asyncio
async def test_synchronous_i2_success_survives_immediate_publication_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    original_publish = source.publish_idle_checkpoint

    def publish_then_cancel(*args: object, **kwargs: object) -> object:
        result = original_publish(*args, **kwargs)
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        return result

    monkeypatch.setattr(source, "publish_idle_checkpoint", publish_then_cancel)
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "succeeded"
        assert source.load_source_snapshot(binding.session_key).checkpoint is not None
        assert len(attempt.requests) == 1
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_real_i2_write_then_raise_consumes_publication_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, registry, binding, _ = _authority(
        tmp_path,
        [{"role": "user", "content": "old objective"}],
    )
    attempt = _Attempt(["offline summary"])
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        replace(_limits(), max_calls=1, max_total_tokens=1000),
    )
    blockers = InMemoryIdleBlockerAuthority()
    blockers.clear(binding.session_key)
    activity = source.load_source_snapshot(binding.session_key).header["metadata"][
        "last_user_activity_at"
    ]
    original_publish = source.publish_idle_checkpoint

    def publish_then_raise(*args: object, **kwargs: object) -> object:
        original_publish(*args, **kwargs)
        raise RuntimeError("write then raise")

    monkeypatch.setattr(source, "publish_idle_checkpoint", publish_then_raise)
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blockers,
        summary_adapter=adapter,
        policy=_enabled_policy(),
        discovery_limits=SourceDiscoveryLimits(100, 100, 2000, 256),
        utc_clock=lambda: datetime.fromisoformat(activity) + timedelta(seconds=30),
    )
    try:
        assert await scanner.scan_once() == 1
        await scanner.coordinator.drain()
        assert scanner.coordinator.outcome_for(binding.session_key).value == "failed"
        assert scanner.coordinator.backoff_for(binding.session_key) is not None
        assert source.load_source_snapshot(binding.session_key).checkpoint is not None
        assert len(attempt.requests) == 1
    finally:
        await scanner.aclose()
