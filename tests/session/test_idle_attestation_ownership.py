"""Real issued idle seals retain their validation at each owning boundary."""
from copy import deepcopy
from dataclasses import replace

import pytest

from nanobot.session.artifact_store import PrivateArtifactStore
from nanobot.session.idle_checkpoint import IdleCheckpointError
from nanobot.session.manager import JsonlSessionStore
from tests.session.test_idle_checkpoint import _admit, _artifact_source, _seed


async def issued(tmp_path):
    source, snapshot, store, descriptor, policy = await _artifact_source(tmp_path)
    capture = source.capture_idle_checkpoint(
        snapshot.state.session_key, runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint, policy=policy,
    )
    assert capture is not None and capture.dependencies is not None
    return source, snapshot, store, descriptor, capture


@pytest.mark.parametrize("field,message", [
    ("_capability", "attestation changed"), ("_capture_id", "attestation changed"),
    ("_nonce", "attestation changed"), ("_token", "attestation changed"),
    ("_source", "attestation changed"), ("_runtime_fingerprint", "attestation changed"),
    ("_policy_fingerprint", "attestation changed"), ("_root_authority", "attestation changed"),
    ("_store", "artifact store binding changed"), ("_store_binding", "artifact store binding changed"),
    ("_snapshot_digest", "snapshot changed"), ("_dependency_present", "dependency authority changed"),
    ("_dependency_id", "dependency authority changed"), ("_descriptor_digest", "dependency authority changed"),
])
async def test_attestation_owner_rejects_every_changed_seal_without_writing(tmp_path, monkeypatch, field, message):
    source, before, store, _, capture = await issued(tmp_path)
    seal = capture._attestation
    assert seal.require_capture(capture) is None
    assert capture.require_attestation() is None
    foreign = PrivateArtifactStore(
        tmp_path / "foreign", workspace_id=store.workspace_id, session_key=store.session_key,
        incarnation=store.incarnation, generation=store.generation,
    )
    changed = {
        "_capability": object(), "_capture_id": id(capture) + 1, "_nonce": object(),
        "_token": replace(capture.token), "_source": object(),
        "_runtime_fingerprint": "changed", "_policy_fingerprint": "changed",
        "_root_authority": object(), "_store": foreign,
        "_store_binding": (store.workspace_id, store.session_key, "changed", store.generation),
        "_snapshot_digest": "0" * 64, "_dependency_present": False,
        "_dependency_id": id(capture.dependencies) + 1, "_descriptor_digest": "0" * 64,
    }
    object.__setattr__(seal, field, changed[field])

    def deny(*_args, **_kwargs):
        raise AssertionError("seal validation attempted storage access")

    with monkeypatch.context() as patch:
        patch.setattr(source, "_read_unlocked", deny)
        patch.setattr(PrivateArtifactStore, "transaction", deny)
        with pytest.raises(IdleCheckpointError, match=message) as owned:
            seal.require_capture(capture)
        with pytest.raises(IdleCheckpointError, match=message) as entry:
            capture.require_attestation()
        assert str(owned.value) == str(entry.value)
    assert source.load_source_snapshot(before.state.session_key) == before


@pytest.mark.parametrize("field", [
    "_source", "_capture_nonce", "_token", "_store", "_policy_fingerprint",
    "_snapshot_digest", "_snapshot", "_descriptors",
])
async def test_dependency_owner_rejects_changed_bindings_and_hashes(tmp_path, field):
    source, before, store, _, capture = await issued(tmp_path)
    seal, access = capture._attestation, capture.dependencies
    arguments = dict(
        snapshot_digest=seal._snapshot_digest, descriptor_digest=seal._descriptor_digest,
    )
    assert access.require_sealed_binding(capture, **arguments) is None
    assert capture.require_attestation() is None
    changed_snapshot = deepcopy(access._snapshot)
    changed_snapshot.events[0]["content"] = "changed captured data"
    foreign = PrivateArtifactStore(
        tmp_path / "foreign", workspace_id=store.workspace_id, session_key=store.session_key,
        incarnation=store.incarnation, generation=store.generation,
    )
    changes = {
        "_source": object(), "_capture_nonce": object(), "_token": replace(capture.token),
        "_store": foreign, "_policy_fingerprint": "changed", "_snapshot_digest": "0" * 64,
        "_snapshot": changed_snapshot, "_descriptors": (),
    }
    object.__setattr__(access, field, changes[field])
    with pytest.raises(IdleCheckpointError, match="dependency authority changed") as owned:
        access.require_sealed_binding(capture, **arguments)
    with pytest.raises(IdleCheckpointError, match="dependency authority changed") as entry:
        capture.require_attestation()
    assert str(owned.value) == str(entry.value)
    assert source.load_source_snapshot(before.state.session_key) == before


async def test_configured_store_identity_observation_does_not_admit_changed_metadata(tmp_path):
    source, before, store, _, capture = await issued(tmp_path)
    binding = (store.workspace_id, store.session_key, store.incarnation, store.generation)
    assert source.matches_configured_artifact_store(store) is True
    assert source.matches_configured_artifact_store(None) is False
    assert source.require_idle_artifact_binding(store, binding) is None
    foreign = PrivateArtifactStore(
        tmp_path / "foreign", workspace_id=store.workspace_id, session_key=store.session_key,
        incarnation=store.incarnation, generation=store.generation,
    )
    assert source.matches_configured_artifact_store(foreign) is False
    with pytest.raises(IdleCheckpointError, match="artifact store binding changed"):
        source.require_idle_artifact_binding(foreign, binding)
    previous = store.generation
    try:
        store.generation += 1
        assert source.matches_configured_artifact_store(store) is True
        with pytest.raises(IdleCheckpointError, match="artifact store binding changed"):
            source.require_idle_artifact_binding(store, binding)
        with pytest.raises(IdleCheckpointError, match="artifact store binding changed"):
            capture.require_attestation()
    finally:
        store.generation = previous
    assert capture.require_attestation() is None
    assert source.load_source_snapshot(before.state.session_key) == before


@pytest.mark.parametrize("field", ["_dependency_id", "_descriptor_digest"])
def test_absent_dependencies_require_absent_identity_and_descriptor_seals(tmp_path, field):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    source = backend.source
    before = _admit(source, _seed(backend))
    capture = source.capture_idle_checkpoint(
        before.state.session_key, runtime_fingerprint="runtime-1", policy_fingerprint="policy-1",
    )
    assert capture is not None and capture.dependencies is None
    assert source.matches_configured_artifact_store(None) is True
    assert source.require_idle_artifact_binding(None, None) is None
    assert capture._attestation.require_capture(capture) is None
    object.__setattr__(capture._attestation, field, "changed")
    with pytest.raises(IdleCheckpointError, match="dependency authority changed"):
        capture.require_attestation()
    assert source.load_source_snapshot(before.state.session_key) == before


async def test_owner_validation_preserves_first_error_and_capture_identity(tmp_path, monkeypatch):
    source, before, _, _, capture = await issued(tmp_path)
    seal = capture._attestation
    assert seal.require_capture(capture) is None
    forged = replace(capture)
    with pytest.raises(IdleCheckpointError, match="attestation changed"):
        forged.require_attestation()
    object.__setattr__(seal, "_capability", object())
    object.__setattr__(seal, "_snapshot_digest", "0" * 64)
    object.__setattr__(capture.dependencies, "_snapshot_digest", "0" * 64)

    def deny(*_args, **_kwargs):
        raise AssertionError("later store guard ran before identity rejection")

    with monkeypatch.context() as patch:
        patch.setattr(type(source), "require_idle_artifact_binding", deny)
        with pytest.raises(IdleCheckpointError, match="attestation changed"):
            capture.require_attestation()
    assert source.load_source_snapshot(before.state.session_key) == before
