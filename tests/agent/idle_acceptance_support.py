"""Self-contained fixtures for the composed dormant-idle acceptance tests.

This module is intentionally not collected by pytest.  It owns only test data,
an offline one-attempt transport, and the small amount of source/artifact setup
needed to exercise the real I2--I3e composition.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from weakref import ReferenceType, ref

from nanobot.agent.idle_compaction import IdleCompactionPolicy
from nanobot.agent.idle_lifecycle import IdleLifecycleIntegration
from nanobot.agent.idle_runtime import IdleRuntimeBinding, IdleRuntimeBindingRegistry
from nanobot.agent.idle_scanner import (
    IdleCompactionScanner,
    InMemoryIdleBlockerAuthority,
)
from nanobot.agent.idle_summary import (
    IdleDependencyVerification,
    IdleSummaryAdapter,
    IdleSummaryAttemptResponse,
    IdleSummaryLimits,
    IdleSummaryMeasuredUsage,
    IdleSummaryRequest,
    IdleSummaryRequestReceipt,
    IdleSummaryTransportRegistry,
)
from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.session.artifact import parse_artifact_descriptor
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_store import PrivateArtifactStore
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.idle_checkpoint import IdlePolicyBinding
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import new_source_event
from nanobot.session.source_discovery import SourceDiscoveryLimits
from nanobot.session.source_store import SourceSessionStore
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.utils.llm_runtime import LLMRuntime


class AcceptanceNetworkViolationError(AssertionError):
    """Raised when an acceptance path attempts external I/O."""


@dataclass
class NetworkDenial:
    """Python-level denial counters; this is deliberately not an OS sandbox."""

    attempts: list[str] = field(default_factory=list)

    def install(self, monkeypatch: Any) -> None:
        def blocked(label: str) -> Callable[..., Any]:
            def deny(*_: object, **__: object) -> object:
                self.attempts.append(label)
                raise AcceptanceNetworkViolationError(f"forbidden {label}")

            return deny

        for name in ("connect", "connect_ex"):
            monkeypatch.setattr(socket.socket, name, blocked(f"socket.{name}"))
        monkeypatch.setattr(socket, "create_connection", blocked("socket.create_connection"))
        monkeypatch.setattr(socket, "getaddrinfo", blocked("socket.getaddrinfo"))
        for name in ("sendto", "sendmsg"):
            if hasattr(socket.socket, name):
                monkeypatch.setattr(socket.socket, name, blocked(f"socket.{name}"))
        monkeypatch.setattr(subprocess, "Popen", blocked("subprocess.Popen"))
        monkeypatch.setattr(asyncio, "create_subprocess_exec", blocked("asyncio.subprocess_exec"))
        monkeypatch.setattr(asyncio, "create_subprocess_shell", blocked("asyncio.subprocess_shell"))
        monkeypatch.setattr(os, "system", blocked("os.system"))
        for name in (
            "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle",
            "spawnlp", "spawnlpe", "posix_spawn", "posix_spawnp",
        ):
            if hasattr(os, name):
                monkeypatch.setattr(os, name, blocked(f"os.{name}"))


@dataclass
class AcceptanceClock:
    """Explicit UTC and monotonic clocks used as positive test data."""

    utc_now: datetime = field(
        default_factory=lambda: datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
    )
    monotonic_now: float = 100.0

    def utc(self) -> datetime:
        return self.utc_now

    def monotonic(self) -> float:
        return self.monotonic_now

    def advance(self, seconds: float) -> None:
        self.utc_now += timedelta(seconds=seconds)
        self.monotonic_now += seconds


@dataclass(frozen=True)
class AttemptObservation:
    """Content-free observation made while a callback owns a request."""

    digest: str
    roles: tuple[str, ...]
    message_count: int
    references: tuple[str, ...]
    saw_full_tail: bool
    saw_forbidden: bool


@dataclass(frozen=True)
class FinalizedMeasurement:
    """Independent receipt and expected charge for one measured payload."""

    observation: AttemptObservation
    receipt: IdleSummaryRequestReceipt
    expected_usage: IdleSummaryMeasuredUsage


@dataclass(frozen=True)
class DispatchedAttempt:
    """Independent transport record for one dispatched payload."""

    observation: AttemptObservation
    receipt: IdleSummaryRequestReceipt
    usage: IdleSummaryMeasuredUsage | None
    attempts: int


@dataclass
class AcceptanceRecords:
    """Shared counters for independently rebound fake transport instances."""

    measurements: list[AttemptObservation] = field(default_factory=list)
    measurement_records: list[FinalizedMeasurement] = field(default_factory=list)
    attempts: list[AttemptObservation] = field(default_factory=list)
    dispatch_records: list[DispatchedAttempt] = field(default_factory=list)
    request_refs: list[ReferenceType[IdleSummaryRequest]] = field(default_factory=list)
    measurement_calls: int = 0
    provider_calls: int = 0
    active_calls: int = 0
    max_active_calls: int = 0
    cancellation_seen: int = 0
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    verify_started: asyncio.Event = field(default_factory=asyncio.Event)
    verify_release: asyncio.Event = field(default_factory=asyncio.Event)
    suppress_cancellation: bool = False
    wait_for_release: bool = False
    force_split: bool = False
    hold_before_result: bool = False
    transport_completed: asyncio.Event = field(default_factory=asyncio.Event)
    release_publication: asyncio.Event = field(default_factory=asyncio.Event)
    response_mode: str = "success"
    outputs: deque[str] = field(default_factory=lambda: deque(["idle summary"]))
    forbidden_sentinels: tuple[str, ...] = (
        "FORBIDDEN_FOREIGN_ARTIFACT",
        "UNAUTHORIZED_PATH",
        "CHILD_STREAM_SENTINEL",
        "RUNTIME_SIGNATURE_SENTINEL",
    )
    full_tail_sentinel: str = "FULL_ARTIFACT_TAIL_SENTINEL"

    def __post_init__(self) -> None:
        # Dependency verification is synchronous by default; tests clear this
        # event explicitly when they need a preflight barrier.
        self.verify_release.set()

    @property
    def forbidden_seen(self) -> bool:
        return any(observation.saw_forbidden for observation in (*self.measurements, *self.attempts))

    def clear_barriers(self) -> None:
        self.release.set()
        self.verify_release.set()


class AcceptanceProvider(LLMProvider):
    """Provider object used only to form a real immutable ``LLMRuntime``."""

    def __init__(self) -> None:
        super().__init__(provider_name="acceptance-provider")
        self.generation = GenerationSettings(max_tokens=32, temperature=0.0)

    def get_default_model(self) -> str:
        return "acceptance-model"

    async def chat(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("the idle acceptance path must use the injected fake transport")


class AcceptanceMeasurer:
    """Deterministic local measurer; it never retains request dictionaries."""

    def __init__(self, records: AcceptanceRecords, runtime_fingerprint: str) -> None:
        self.provider_name = "acceptance-provider"
        self.transport = "acceptance.one-shot"
        self.model = "acceptance-model"
        self.context_window_tokens = 4096
        self.runtime_fingerprint = runtime_fingerprint
        self.records = records

    @staticmethod
    def _observe(messages: Any, sentinels: tuple[str, ...], tail: str) -> AttemptObservation:
        raw = json.dumps(list(messages), ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        roles: list[str] = []
        references: list[str] = []
        text = raw
        for message in messages:
            role = message.get("role") if isinstance(message, dict) else None
            if isinstance(role, str):
                roles.append(role)
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and "artifact:" in content:
                    references.extend(
                        part for part in content.split() if part.startswith("artifact:")
                    )
        return AttemptObservation(
            digest,
            tuple(roles),
            len(list(messages)),
            tuple(sorted(set(references))),
            tail in text,
            any(sentinel in text for sentinel in sentinels),
        )

    def measure_request(
        self,
        messages: Any,
        *,
        max_output_tokens: int,
        safety_buffer: int,
    ) -> IdleSummaryRequestReceipt:
        observation = self._observe(messages, self.records.forbidden_sentinels, self.records.full_tail_sentinel)
        self.records.measurement_calls += 1
        self.records.measurements.append(observation)
        input_tokens = max(1, len(json.dumps(list(messages), ensure_ascii=False).encode("utf-8")) // 4)
        if self.records.force_split:
            payload_bytes = len(json.dumps(list(messages), ensure_ascii=False).encode("utf-8"))
            input_tokens = 6_000 if payload_bytes > 5_000 else 100
        request_digest = hashlib.sha256(
            json.dumps(list(messages), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        receipt = IdleSummaryRequestReceipt(
            self.provider_name,
            self.transport,
            self.model,
            input_tokens,
            max_output_tokens,
            safety_buffer,
            self.context_window_tokens,
            input_tokens + max_output_tokens + safety_buffer,
            request_digest,
        )
        self.records.measurement_records.append(
            FinalizedMeasurement(
                observation,
                receipt,
                IdleSummaryMeasuredUsage(input_tokens, 2, input_tokens + 2),
            )
        )
        return receipt


class AcceptanceTransport:
    """Qualified raw adapter implementing exactly one fake transport attempt."""

    def __init__(self, records: AcceptanceRecords, runtime_fingerprint: str) -> None:
        self.provider_name = "acceptance-provider"
        self.transport = "acceptance.one-shot"
        self.model = "acceptance-model"
        self.context_window_tokens = 4096
        self.runtime_fingerprint = runtime_fingerprint
        self.records = records

    async def attempt_once(self, request: IdleSummaryRequest) -> IdleSummaryAttemptResponse:
        records = self.records
        observation = AcceptanceMeasurer._observe(
            request.messages, records.forbidden_sentinels, records.full_tail_sentinel
        )
        records.attempts.append(observation)
        receipt = request.receipt
        detached_receipt = IdleSummaryRequestReceipt(
            receipt.provider,
            receipt.transport,
            receipt.model,
            receipt.input_tokens,
            receipt.reserved_output_tokens,
            receipt.safety_buffer,
            receipt.model_window_tokens,
            receipt.total_reservation_tokens,
            receipt.request_digest,
        )
        records.request_refs.append(ref(request))
        records.provider_calls += 1
        records.active_calls += 1
        records.max_active_calls = max(records.max_active_calls, records.active_calls)
        records.started.set()
        dispatch_index = len(records.dispatch_records)
        records.dispatch_records.append(DispatchedAttempt(observation, detached_receipt, None, 1))
        try:
            if records.wait_for_release:
                try:
                    await records.release.wait()
                except asyncio.CancelledError:
                    records.cancellation_seen += 1
                    records.cancelled.set()
                    if records.suppress_cancellation:
                        await records.release.wait()
                    else:
                        raise
            if records.hold_before_result:
                records.transport_completed.set()
                await records.release_publication.wait()
            mode = records.response_mode
            if mode == "error":
                raise RuntimeError("fake transport failure")
            if mode == "unknown-usage":
                return IdleSummaryAttemptResponse("idle summary", None, request.receipt)
            if mode == "malformed-response":
                return IdleSummaryAttemptResponse(
                    "idle summary", object(), request.receipt  # type: ignore[arg-type]
                )
            if mode == "attempts-2":
                usage = IdleSummaryMeasuredUsage(request.receipt.input_tokens, 2, request.receipt.input_tokens + 2)
                records.dispatch_records[dispatch_index] = DispatchedAttempt(
                    observation, detached_receipt, usage, 2
                )
                return IdleSummaryAttemptResponse(
                    "idle summary",
                    usage,
                    request.receipt,
                    attempts=2,
                )
            output = records.outputs.popleft() if records.outputs else "idle summary"
            usage = IdleSummaryMeasuredUsage(request.receipt.input_tokens, 2, request.receipt.input_tokens + 2)
            records.dispatch_records[dispatch_index] = DispatchedAttempt(
                observation, detached_receipt, usage, 1
            )
            return IdleSummaryAttemptResponse(
                output,
                usage,
                request.receipt,
            )
        finally:
            records.active_calls -= 1


class AcceptanceDependencyVerifier:
    """Verifier that checks the sealed root dependency without retaining bytes."""

    def __init__(self, source: SourceSessionStore, store: PrivateArtifactStore, policy: IdlePolicyBinding,
                 records: AcceptanceRecords) -> None:
        self.source = source
        self.store = store
        self.policy = policy
        self.records = records
        self.calls: list[tuple[str, ...]] = []

    async def verify(self, *, capture: Any, binding: IdleRuntimeBinding,
                     references: tuple[str, ...]) -> IdleDependencyVerification:
        self.calls.append(references)
        self.records.verify_started.set()
        if not self.records.verify_release.is_set():
            await self.records.verify_release.wait()
        with self.store.transaction(blocking=False) as blobs:
            snapshot = self.source.load_source_snapshot(binding.session_key)
            if capture.dependencies is None:
                raise AssertionError("I2 did not retain dependency authority")
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


@dataclass
class IdleAcceptanceHarness:
    backend: JsonlSessionStore
    source: SourceSessionStore
    store: PrivateArtifactStore | None
    descriptor: Any | None
    policy: IdlePolicyBinding
    registry: IdleRuntimeBindingRegistry
    blocker: InMemoryIdleBlockerAuthority
    records: AcceptanceRecords
    scanner: IdleCompactionScanner
    lifecycle: IdleLifecycleIntegration
    clock: AcceptanceClock
    key: str
    activity: str

    async def close(self) -> None:
        self.records.clear_barriers()
        await self.lifecycle.aclose()


def _policy() -> IdleCompactionPolicy:
    return IdleCompactionPolicy(
        idle_ttl_seconds=30.0,
        queue_capacity=2,
        max_concurrent_jobs=1,
        max_auxiliary_calls=4,
        max_auxiliary_tokens=20_000,
        max_auxiliary_wall_seconds=5.0,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
    )


def _summary_limits(*, force_split: bool = False) -> IdleSummaryLimits:
    return IdleSummaryLimits(
        max_calls=4,
        max_total_tokens=20_000,
        max_wall_seconds=5.0,
        max_input_tokens=5_000 if force_split else 20_000,
        max_output_tokens=32,
        max_chunks=4,
        safety_buffer=2,
    )


async def append_retrieval(writer: SourceTurnWriter, reader: ArtifactReader, reference: str) -> None:
    from nanobot.agent.tools.artifact import ReadArtifactTool

    outcome = await ToolRegistry.execute_prepared(
        ReadArtifactTool(reader), {"reference": reference}, SimpleNamespace(lock_owner=None)
    )
    if not isinstance(outcome, ToolOutcome):
        raise AssertionError("fixture retrieval did not produce an outcome")
    writer.begin_tool_batch({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "artifact", "type": "function", "function": {
            "name": "read_artifact", "arguments": "{}"
        }}],
    })
    await writer.record_tool_result(KnownToolResult(
        0, "artifact", "read_artifact", outcome, outcome.content,
        {"status": "ok"}, True, retrieval_receipt=outcome.retrieval_receipt,
    ))


async def _build_artifact_source(
    tmp_path: Path,
    *,
    key: str,
    text: str,
    retrieval: bool,
    child_sentinel: bool,
) -> tuple[JsonlSessionStore, SourceSessionStore, PrivateArtifactStore, Any]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "evidence.txt").write_text(text, encoding="utf-8")
    backend = JsonlSessionStore(workspace, sessions_root=tmp_path / "sessions")
    session = Session(key)
    backend.save(session)
    upgraded = backend.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    async with acquire_execution(backend, key, run_id="acceptance-run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="acceptance-run")
        writer.append_message({"role": "user", "content": "idle objective"})
        tool = ReadFileTool(workspace=workspace, allowed_dir=workspace)
        outcome = await ToolRegistry.execute_prepared(
            tool, {"path": "evidence.txt"}, SimpleNamespace(lock_owner=None)
        )
        if not isinstance(outcome, ToolOutcome) or outcome.producer_receipt is None:
            raise AssertionError("fixture read did not produce a source receipt")
        writer.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "read", "type": "function", "function": {
                "name": "read_file", "arguments": "{}"
            }}],
        })
        await writer.record_tool_result(KnownToolResult(
            0, "read", "read_file", outcome, outcome.content,
            {"status": "ok"}, True, outcome.producer_receipt,
        ))
        snapshot = owner.read_snapshot()
        event = snapshot.events[-1]
        raw_content = event["content"].encode("utf-8")
        raw = {
            "schema_version": 1,
            "artifact_id": uuid.uuid4().hex,
            "workspace_id": backend.sessions_dir.name,
            "session_key": key,
            "incarnation": snapshot.state.incarnation,
            "generation": snapshot.state.generation,
            "event_id": event["_event_id"],
            "execution_id": event["_execution_id"],
            "stream_id": writer.stream_id,
            "run_id": "acceptance-run",
            "block_index": None,
            "mime_type": "text/plain; charset=utf-8",
            "byte_length": len(raw_content),
            "sha256": hashlib.sha256(raw_content).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_access": outcome.producer_receipt.to_dict()["source_access"],
        }
        descriptor = parse_artifact_descriptor(raw)
        store = PrivateArtifactStore(
            (tmp_path / "artifacts").resolve(),
            workspace_id=descriptor.workspace_id,
            session_key=key,
            incarnation=descriptor.incarnation,
            generation=descriptor.generation,
        )
        source = SourceSessionStore(backend, artifact_store=store)
        with store.transaction() as blobs:
            blobs.put(descriptor, raw_content)
            source.commit_artifact_descriptor(key, snapshot.state, descriptor, owner=owner, blobs=blobs)
        if child_sentinel:
            child = writer.child_stream()
            child.append_message({"role": "user", "content": "CHILD_STREAM_SENTINEL"})
        if retrieval:
            reader = ArtifactReader(writer, "root_history", store, tool)
            await append_retrieval(writer, reader, descriptor.reference)
    # Confirm the descriptor remains bound to the final source generation.
    if source.load_source_snapshot(key).state.generation != upgraded.state.generation:
        raise AssertionError("fixture changed source generation")
    return backend, source, store, descriptor


async def make_harness(
    tmp_path: Path,
    *,
    artifact: bool = False,
    retrieval: bool = False,
    text: str | None = None,
    response_mode: str = "success",
    wait_for_release: bool = False,
    multi_chunk: bool = False,
    child_sentinel: bool = False,
) -> IdleAcceptanceHarness:
    key = "acceptance:idle"
    clock = AcceptanceClock()
    if artifact:
        text = text or ("preview head\n" + ("a" * 2000) + "\n" + "FULL_ARTIFACT_TAIL_SENTINEL")
        backend, source, store, descriptor = await _build_artifact_source(
            tmp_path, key=key, text=text, retrieval=retrieval, child_sentinel=child_sentinel
        )
    else:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        backend = JsonlSessionStore(workspace, sessions_root=tmp_path / "sessions")
        session = Session(key)
        session.messages = (
            [
                {"role": "user", "content": "idle objective"},
                {"role": "assistant", "content": "completed fixture work"},
            ]
            if not multi_chunk
            else [
                {"role": "user", "content": "idle objective one " + ("x" * 1100)},
                {"role": "assistant", "content": "completed fixture work one " + ("y" * 1100)},
                {"role": "user", "content": "idle objective two " + ("z" * 1100)},
                {"role": "assistant", "content": "completed fixture work two " + ("q" * 1100)},
            ]
        )
        backend.save(session)
        backend.source.upgrade_source_session(
            key,
            expected_revision=session.revision,
            expected_generation=session.generation,
            expected_incarnation=session.incarnation,
        )
        source = backend.source
        store = None
        descriptor = None
    activity_snapshot = source.record_user_activity(
        key, source.load_source_snapshot(key).state, accepted_foreground_user=True
    )
    activity = activity_snapshot.header["metadata"]["last_user_activity_at"]
    policy = IdlePolicyBinding(
        "acceptance-policy",
        tmp_path / "workspace",
        None,
        lambda value: tmp_path / "workspace" / value,
    )
    provider = AcceptanceProvider()
    runtime = LLMRuntime(
        provider,
        "acceptance-model",
        GenerationSettings(max_tokens=32, temperature=0.0),
        4096,
        snapshot_signature=("acceptance", "model"),
    )
    limits = _policy()
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=runtime,
        auxiliary_runtime=runtime,
        limits=limits,
        workspace=tmp_path / "workspace",
        artifact_namespace="acceptance-artifacts",
        policy_binding=policy,
    )
    records = AcceptanceRecords(
        response_mode=response_mode,
        wait_for_release=wait_for_release,
        suppress_cancellation=wait_for_release,
        force_split=multi_chunk,
    )
    dependency_verifier = (
        AcceptanceDependencyVerifier(source, store, policy, records) if store is not None else None
    )

    def bound_factory(current: IdleRuntimeBinding) -> IdleSummaryAdapter:
        raw = AcceptanceTransport(records, "acceptance-runtime")
        measurer = AcceptanceMeasurer(records, "acceptance-runtime")
        transport = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source).issue(current, raw)
        return IdleSummaryAdapter(
            transport,
            _summary_limits(force_split=records.force_split),
            request_measurer=measurer,
            binding_registry=registry,
            dependency_verifier=dependency_verifier,
            clock=clock.monotonic,
        )

    blocker: InMemoryIdleBlockerAuthority = InMemoryIdleBlockerAuthority()
    blocker.clear(key)
    scanner = IdleCompactionScanner(
        source_store=source,
        binding_registry=registry,
        blocker_authority=blocker,
        summary_adapter_factory=bound_factory,
        policy=limits,
        discovery_limits=SourceDiscoveryLimits(100, 100, 4096, 256),
        utc_clock=lambda: datetime.fromisoformat(activity).astimezone(timezone.utc) + timedelta(seconds=60),
        monotonic_clock=clock.monotonic,
    )
    lifecycle = IdleLifecycleIntegration(
        scanner=scanner,
        registry=registry,
        owns_scanner=True,
        owns_registry=True,
    )
    return IdleAcceptanceHarness(
        backend, source, store, descriptor, policy, registry, blocker, records,
        scanner, lifecycle, clock, key, activity,
    )


def source_event(key: str, content: str, *, run_id: str = "acceptance-append") -> dict[str, Any]:
    """Create one trusted test-only source event for a later-version check."""

    return new_source_event({"role": "user", "content": content}, run_id=run_id)
