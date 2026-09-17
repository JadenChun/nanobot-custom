"""Focused tests for the dormant lease-free idle checkpoint substrate."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import pickle
import threading
from contextlib import contextmanager
from copy import copy as shallow_copy
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    parse_artifact_descriptor,
)
from nanobot.session.artifact_store import ArtifactStorageError, PrivateArtifactStore
from nanobot.session.execution_owner import (
    ExecutionBusyError,
    ExecutionOwnershipError,
    acquire_execution,
)
from nanobot.session.idle_checkpoint import (
    IdleCheckpointCapture,
    IdleCheckpointConflictError,
    IdleCheckpointError,
    IdleDependencyAccess,
    IdlePolicyBinding,
    project_idle_root_nodes,
)
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import prefix_digest
from nanobot.session.source_compaction import SourceCompactionCandidate
from nanobot.session.source_projection import CheckpointNode, source_nodes
from nanobot.session.source_request import SourceTranscript
from nanobot.session.source_store import SourceSessionStore
from nanobot.session.turn_writer import SourceTurnWriter


def _seed(backend: JsonlSessionStore, key: str = "test:idle"):
    session = Session(key)
    session.messages = [{"role": "user", "content": "original"}]
    backend.save(session)
    return backend.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )


def _admit(source: SourceSessionStore, snapshot):
    return source.record_user_activity(
        snapshot.state.session_key,
        snapshot.state,
        accepted_foreground_user=True,
    )


def _proposal(snapshot, *, references=(), identity="idle-checkpoint"):
    boundary = len(snapshot.events)
    return {
        "schema_version": 2 if references else 1,
        "digest_version": 1,
        "summary_format_version": 1,
        "session_generation": snapshot.state.generation,
        "source_revision": snapshot.state.revision,
        "checkpoint_id": identity,
        "parent_checkpoint_id": snapshot.state.checkpoint_id,
        "producing_model": "idle-model",
        "producing_provider": "idle-provider",
        "digest_algorithm": "sha256",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": "Retained idle facts",
        "artifact_references": list(references),
        "boundary": boundary,
        "boundary_event_id": snapshot.events[boundary - 1]["_event_id"] if boundary else None,
        "prefix_digest": prefix_digest(snapshot.events, boundary),
        "auxiliary_usage": None,
    }


def _dependency_proof(capture: IdleCheckpointCapture, summary: str, descriptor):
    nodes = source_nodes(capture.snapshot.events)
    from nanobot.session.artifact_projection import render_artifact_projection

    projected = tuple(
        render_artifact_projection(node, descriptor)
        if node.event_id == descriptor.event_id
        else node
        for node in nodes
    )
    candidate = SourceCompactionCandidate.capture(
        capture.snapshot,
        SourceTranscript(projected),
        eligible_ids=frozenset(node.event_id for node in projected),
    )
    evidence = candidate.summary_evidence(candidate.boundaries[-1])
    from nanobot.session.checkpoint_dependencies import ConsumedSummaryChunk, SummaryDependencyProof

    return SummaryDependencyProof(candidate, (ConsumedSummaryChunk(evidence, None, summary),))


def _artifact_policy(workspace, *, fingerprint="policy-1"):
    return IdlePolicyBinding(
        fingerprint,
        workspace,
        None,
        lambda value: workspace / value,
    )


def _hold_artifact_store(root, workspace_id, session_key, incarnation, generation, entered, release):
    store = PrivateArtifactStore(
        root,
        workspace_id=workspace_id,
        session_key=session_key,
        incarnation=incarnation,
        generation=generation,
    )
    with store.transaction():
        entered.set()
        release.wait(timeout=10)


async def _artifact_source(tmp_path, text="evidence 中文 🌟"):
    """Build a published root artifact, then release its foreground owner."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file").write_text(text, encoding="utf-8")
    backend = JsonlSessionStore(workspace)
    session = Session("test:read")
    backend.save(session)
    backend.source.upgrade_source_session(
        session.key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    artifact_root = (tmp_path / "artifacts").resolve()
    descriptor = None
    async with acquire_execution(backend, session.key, run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        tool = ReadFileTool(workspace=workspace, allowed_dir=workspace)
        outcome = await ToolRegistry.execute_prepared(
            tool, {"path": "file"}, SimpleNamespace(lock_owner=None)
        )
        writer.begin_tool_batch(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "read",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        )
        await writer.record_tool_result(
            KnownToolResult(
                0,
                "read",
                "read_file",
                outcome,
                outcome.content,
                {"status": "ok"},
                True,
                outcome.producer_receipt,
            )
        )
        snapshot = owner.read_snapshot()
        event = snapshot.events[-1]
        from session.test_artifact import fixture

        raw, _ = fixture()
        content = event["content"].encode()
        raw.update(
            workspace_id=backend.sessions_dir.name,
            session_key=session.key,
            incarnation=snapshot.state.incarnation,
            generation=snapshot.state.generation,
            event_id=event["_event_id"],
            execution_id=event["_execution_id"],
            stream_id=writer.stream_id,
            byte_length=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            source_access=outcome.producer_receipt.to_dict()["source_access"],
        )
        descriptor = parse_artifact_descriptor(raw)
        store = PrivateArtifactStore(
            artifact_root,
            workspace_id=descriptor.workspace_id,
            session_key=session.key,
            incarnation=descriptor.incarnation,
            generation=descriptor.generation,
        )
        source = SourceSessionStore(backend, artifact_store=store)
        with store.transaction() as blobs:
            blobs.put(descriptor, content)
            source.commit_artifact_descriptor(session.key, snapshot.state, descriptor, owner=owner, blobs=blobs)
    source = SourceSessionStore(backend, artifact_store=store)
    admitted = _admit(source, source.load_source_snapshot(session.key))
    return source, admitted, store, descriptor, _artifact_policy(workspace)


@pytest.mark.asyncio
async def test_dependency_free_idle_checkpoint_writes_once_and_preserves_activity(tmp_path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    source = backend.source
    admitted = _admit(source, _seed(backend))
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
    )
    assert isinstance(capture, IdleCheckpointCapture)
    before_activity = capture.snapshot.header["metadata"]["last_user_activity_at"]
    candidate = _proposal(capture.snapshot)
    committed = source.publish_idle_checkpoint(
        admitted.state.session_key,
        capture,
        candidate,
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
    )
    assert committed.checkpoint == candidate
    assert committed.header["metadata"]["last_user_activity_at"] == before_activity
    assert committed.state.revision == capture.snapshot.state.revision + 1


@pytest.mark.asyncio
async def test_dependency_bearing_idle_checkpoint_verifies_exact_root_dependencies(tmp_path):
    source, admitted, store, descriptor, policy = await _artifact_source(tmp_path)
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None and isinstance(capture.dependencies, IdleDependencyAccess)
    assert capture.dependencies._snapshot is not capture.snapshot
    candidate = _proposal(capture.snapshot, references=[descriptor.reference])
    proof = _dependency_proof(capture, candidate["summary"], descriptor)
    committed = source.publish_idle_checkpoint(
        admitted.state.session_key,
        capture,
        candidate,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
        proof=proof,
    )
    assert committed.checkpoint == candidate
    assert committed.checkpoint["artifact_references"] == [descriptor.reference]
    assert store.root.exists()


@pytest.mark.asyncio
async def test_idle_projection_bridge_uses_captured_root_descriptor_without_retrieval(tmp_path):
    source, admitted, _, descriptor, policy = await _artifact_source(
        tmp_path, "preview head\n" + ("a" * 2000) + "\nunique full-tail sentinel"
    )
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None and isinstance(capture.dependencies, IdleDependencyAccess)
    original = source_nodes(capture.snapshot.events)
    projected = project_idle_root_nodes(capture, original)
    assert projected[:-1] == original[:-1]
    assert projected[-1].event_id == descriptor.event_id
    assert projected[-1].artifact is not None
    assert descriptor.reference in projected[-1].message["content"]
    assert "unique full-tail sentinel" in original[-1].message["content"]
    assert "unique full-tail sentinel" not in projected[-1].message["content"]


@pytest.mark.asyncio
async def test_idle_projection_bridge_seal_rejects_reconstruction_mutation_and_swaps(tmp_path):
    source, admitted, store, descriptor, policy = await _artifact_source(
        tmp_path, "preview head\n" + ("a" * 2000) + "\nunique full-tail sentinel"
    )
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    other = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None and other is not None
    canonical = source_nodes(capture.snapshot.events)
    for forged in (
        replace(capture, snapshot=deepcopy(capture.snapshot)),
        replace(capture, dependencies=None),
        replace(capture, dependencies=other.dependencies),
    ):
        with pytest.raises(IdleCheckpointError):
            project_idle_root_nodes(forged, canonical)
    with pytest.raises(TypeError):
        shallow_copy(capture)

    binding_mutation = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert binding_mutation is not None
    binding_mutation.snapshot.header["metadata"]["artifact_manifest"]["descriptors"][
        descriptor.artifact_id
    ]["event_id"] = "foreign-event"
    with pytest.raises(IdleCheckpointError):
        project_idle_root_nodes(binding_mutation, source_nodes(binding_mutation.snapshot.events))

    capture.snapshot.header["metadata"].pop("artifact_manifest")
    with pytest.raises(IdleCheckpointError):
        project_idle_root_nodes(capture, source_nodes(capture.snapshot.events))

    assert store.root.exists()


@pytest.mark.asyncio
async def test_idle_projection_bridge_is_storage_and_policy_free(tmp_path, monkeypatch):
    source, admitted, _, descriptor, policy = await _artifact_source(
        tmp_path, "preview head\n" + ("a" * 2000) + "\nunique full-tail sentinel"
    )
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None

    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("pure idle projection touched external authority")

    monkeypatch.setattr(source, "_read_unlocked", fail)
    monkeypatch.setattr(PrivateArtifactStore, "transaction", fail)
    projected = project_idle_root_nodes(capture, source_nodes(capture.snapshot.events))
    assert projected[-1].artifact is not None
    assert projected[-1].artifact.artifact_id == descriptor.artifact_id
    assert all(
        not isinstance(node.artifact, (ArtifactDescriptor, IdleDependencyAccess, IdlePolicyBinding))
        for node in projected
    )


@pytest.mark.asyncio
async def test_idle_projection_bridge_fails_closed_without_matching_capability(tmp_path):
    source, admitted, _, _, _ = await _artifact_source(tmp_path)
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
    )
    assert capture is not None and capture.dependencies is None
    with pytest.raises(IdleCheckpointError):
        project_idle_root_nodes(capture, source_nodes(capture.snapshot.events))


@pytest.mark.asyncio
async def test_idle_capture_and_access_are_detached_and_nonserializable(tmp_path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    source = backend.source
    admitted = _admit(source, _seed(backend))
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
    )
    assert capture is not None
    capture.snapshot.events[0]["content"] = "detached"
    assert source.load_source_snapshot(admitted.state.session_key).events[0]["content"] != "detached"
    with pytest.raises(TypeError):
        pickle.dumps(capture)


@pytest.mark.asyncio
async def test_forged_stale_and_foreign_capture_never_writes(tmp_path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    source = backend.source
    admitted = _admit(source, _seed(backend))
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
    )
    assert capture is not None
    candidate = _proposal(capture.snapshot)
    forged = replace(capture, token=replace(capture.token, user_activity_revision=99))
    with pytest.raises(IdleCheckpointConflictError):
        source.publish_idle_checkpoint(
            admitted.state.session_key,
            forged,
            candidate,
            runtime_fingerprint="runtime-1",
            policy_fingerprint="policy-1",
        )
    _admit(source, source.load_source_snapshot(admitted.state.session_key))
    with pytest.raises(IdleCheckpointConflictError):
        source.publish_idle_checkpoint(
            admitted.state.session_key,
            capture,
            candidate,
            runtime_fingerprint="runtime-1",
            policy_fingerprint="policy-1",
        )


@pytest.mark.asyncio
async def test_dependency_mutation_policy_revocation_or_blob_failure_never_writes(tmp_path):
    source, admitted, store, descriptor, policy = await _artifact_source(tmp_path)
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None
    candidate = _proposal(capture.snapshot, references=[descriptor.reference])
    proof = _dependency_proof(capture, candidate["summary"], descriptor)
    before = source._backend.get_session_path(admitted.state.session_key).read_bytes()
    with pytest.raises(IdleCheckpointConflictError):
        source.publish_idle_checkpoint(
            admitted.state.session_key,
            capture,
            candidate,
            runtime_fingerprint="runtime-1",
            policy_fingerprint="policy-revoked",
            policy=_artifact_policy(policy.workspace, fingerprint="policy-revoked"),
            proof=proof,
        )
    next(store.root.rglob("*.blob")).write_bytes(b"corrupt")
    with pytest.raises((ArtifactStorageError, ArtifactValidationError)):
        source.publish_idle_checkpoint(
            admitted.state.session_key,
            capture,
            candidate,
            runtime_fingerprint="runtime-1",
            policy_fingerprint=policy.fingerprint,
            policy=policy,
            proof=proof,
        )
    assert source._backend.get_session_path(admitted.state.session_key).read_bytes() == before


@pytest.mark.asyncio
async def test_foreign_store_forged_proof_and_runtime_change_never_write(tmp_path):
    source, admitted, store, descriptor, policy = await _artifact_source(tmp_path)
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None
    candidate = _proposal(capture.snapshot, references=[descriptor.reference])
    proof = _dependency_proof(capture, candidate["summary"], descriptor)
    before = source._backend.get_session_path(admitted.state.session_key).read_bytes()
    foreign_store = PrivateArtifactStore(
        (tmp_path / "foreign-artifacts").resolve(),
        workspace_id=store.workspace_id,
        session_key=store.session_key,
        incarnation=store.incarnation,
        generation=store.generation,
    )
    foreign_source = SourceSessionStore(source._backend, artifact_store=foreign_store)
    with pytest.raises(IdleCheckpointConflictError):
        foreign_source.publish_idle_checkpoint(
            admitted.state.session_key,
            capture,
            candidate,
            runtime_fingerprint="runtime-1",
            policy_fingerprint=policy.fingerprint,
            policy=policy,
            proof=proof,
        )
    with pytest.raises(IdleCheckpointError):
        source.publish_idle_checkpoint(
            admitted.state.session_key,
            capture,
            candidate,
            runtime_fingerprint="runtime-2",
            policy_fingerprint=policy.fingerprint,
            policy=policy,
            proof=proof,
        )
    forged = replace(proof, candidate=replace(proof.candidate, stream_id="child"))
    with pytest.raises(IdleCheckpointError):
        source.publish_idle_checkpoint(
            admitted.state.session_key,
            capture,
            candidate,
            runtime_fingerprint="runtime-1",
            policy_fingerprint=policy.fingerprint,
            policy=policy,
            proof=forged,
        )
    assert source._backend.get_session_path(admitted.state.session_key).read_bytes() == before


@pytest.mark.asyncio
async def test_publicly_forged_prior_checkpoint_cannot_authorize_idle_proof(tmp_path):
    source, admitted, store, descriptor, policy = await _artifact_source(tmp_path)
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None
    candidate = _proposal(capture.snapshot, references=[descriptor.reference])

    # Public dataclass construction can detach a plausible historical checkpoint
    # without access to the module's capability sentinel or object.__setattr__.
    fake_boundary = 1
    fake_previous = CheckpointNode("forged-prior", "Forged prior summary")
    forged_snapshot = replace(
        capture.snapshot,
        checkpoint={
            "checkpoint_id": fake_previous.checkpoint_id,
            "summary": fake_previous.summary,
            "artifact_references": [],
            "boundary": fake_boundary,
        },
    )
    forged_capture = replace(capture, snapshot=forged_snapshot)
    nodes = source_nodes(forged_snapshot.events)
    from nanobot.session.artifact_projection import render_artifact_projection

    projected = tuple(
        render_artifact_projection(node, descriptor)
        if node.event_id == descriptor.event_id
        else node
        for node in nodes
    )
    proof_candidate = SourceCompactionCandidate.capture(
        forged_snapshot,
        SourceTranscript((fake_previous, *projected[fake_boundary:])),
        eligible_ids=frozenset(node.event_id for node in projected),
        accepted_boundary=fake_boundary,
        accepted_checkpoint=fake_previous,
    )
    assert proof_candidate.boundaries[-1] == candidate["boundary"]
    evidence = proof_candidate.summary_evidence(proof_candidate.boundaries[-1])
    from nanobot.session.checkpoint_dependencies import (
        ConsumedSummaryChunk,
        SummaryDependencyProof,
    )

    proof = SummaryDependencyProof(
        proof_candidate,
        (ConsumedSummaryChunk(evidence, fake_previous.summary, candidate["summary"]),),
    )
    before = source._backend.get_session_path(admitted.state.session_key).read_bytes()
    with pytest.raises(IdleCheckpointConflictError):
        source.publish_idle_checkpoint(
            admitted.state.session_key,
            forged_capture,
            candidate,
            runtime_fingerprint="runtime-1",
            policy_fingerprint=policy.fingerprint,
            policy=policy,
            proof=proof,
        )
    assert source._backend.get_session_path(admitted.state.session_key).read_bytes() == before


@pytest.mark.asyncio
async def test_pending_and_source_mutation_races_are_rejected_without_write(tmp_path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    source = backend.source
    admitted = _admit(source, _seed(backend))
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
    )
    assert capture is not None
    candidate = _proposal(capture.snapshot)
    path = backend.get_session_path(admitted.state.session_key)
    rows = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    header["metadata"]["pending_user_followups"] = [{"id": "pending"}]
    rows[0] = json.dumps(header)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(IdleCheckpointConflictError):
        source.publish_idle_checkpoint(
            admitted.state.session_key,
            capture,
            candidate,
            runtime_fingerprint="runtime-1",
            policy_fingerprint="policy-1",
        )
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_dependency_lock_precedes_execution_lock_and_restart_retains_evidence(
    tmp_path, monkeypatch
):
    source, admitted, store, descriptor, policy = await _artifact_source(tmp_path)
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None
    candidate = _proposal(capture.snapshot, references=[descriptor.reference])
    proof = _dependency_proof(capture, candidate["summary"], descriptor)
    order = []
    original_transaction = store.transaction
    original_execution = __import__("nanobot.session.idle_checkpoint", fromlist=["execution_transaction"]).execution_transaction

    @contextmanager
    def marked_artifacts(*, blocking=True):
        order.append("artifacts")
        with original_transaction(blocking=blocking) as blobs:
            yield blobs

    @contextmanager
    def marked_execution(*args, **kwargs):
        order.append("execution")
        with original_execution(*args, **kwargs):
            yield

    monkeypatch.setattr(store, "transaction", marked_artifacts)
    monkeypatch.setattr(
        "nanobot.session.idle_checkpoint.execution_transaction", marked_execution
    )
    source.publish_idle_checkpoint(
        admitted.state.session_key,
        capture,
        candidate,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
        proof=proof,
    )
    assert order[:2] == ["artifacts", "execution"]
    restarted = SourceSessionStore(source._backend, artifact_store=store)
    loaded = restarted.load_source_snapshot(admitted.state.session_key)
    assert loaded.checkpoint is not None
    assert loaded.checkpoint["artifact_references"] == [descriptor.reference]


@pytest.mark.asyncio
async def test_active_owner_and_recovery_are_nonblocking_and_do_not_write(tmp_path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    source = backend.source
    admitted = _admit(source, _seed(backend))
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
    )
    assert capture is not None
    candidate = _proposal(capture.snapshot)
    async with acquire_execution(backend, admitted.state.session_key, run_id="foreground"):
        with pytest.raises(ExecutionBusyError):
            source.publish_idle_checkpoint(
                admitted.state.session_key,
                capture,
                candidate,
                runtime_fingerprint="runtime-1",
                policy_fingerprint="policy-1",
            )
    backend.update_metadata(admitted.state.session_key, {"execution_recovery_required": True})
    before_recovery_attempt = backend.get_session_path(admitted.state.session_key).read_bytes()
    with pytest.raises((ExecutionOwnershipError, IdleCheckpointConflictError)):
        source.publish_idle_checkpoint(
            admitted.state.session_key,
            capture,
            candidate,
            runtime_fingerprint="runtime-1",
            policy_fingerprint="policy-1",
        )
    assert backend.get_session_path(admitted.state.session_key).read_bytes() == before_recovery_attempt


@pytest.mark.asyncio
async def test_dependency_idle_rejects_artifact_contention_before_lock_release(tmp_path):
    source, admitted, store, descriptor, policy = await _artifact_source(tmp_path)
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None
    candidate = _proposal(capture.snapshot, references=[descriptor.reference])
    proof = _dependency_proof(capture, candidate["summary"], descriptor)
    source_path = source._backend.get_session_path(admitted.state.session_key)
    before = source_path.read_bytes()

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
    attempt_finished = threading.Event()
    outcome: dict[str, BaseException | None] = {"error": None}

    def attempt() -> None:
        try:
            source.publish_idle_checkpoint(
                admitted.state.session_key,
                capture,
                candidate,
                runtime_fingerprint="runtime-1",
                policy_fingerprint=policy.fingerprint,
                policy=policy,
                proof=proof,
            )
        except BaseException as exc:  # pragma: no cover - assertion below reports the type
            outcome["error"] = exc
        finally:
            attempt_finished.set()

    worker: threading.Thread | None = None
    try:
        assert entered.wait(timeout=5)
        worker = threading.Thread(target=attempt)
        worker.start()
        assert attempt_finished.wait(timeout=1), "idle publication waited for artifact lock release"
        assert not release.is_set()
        assert isinstance(outcome["error"], ExecutionBusyError)
        assert source_path.read_bytes() == before
    finally:
        release.set()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
        if worker is not None:
            worker.join(timeout=10)
    assert holder.exitcode == 0
    assert worker is not None and not worker.is_alive()
