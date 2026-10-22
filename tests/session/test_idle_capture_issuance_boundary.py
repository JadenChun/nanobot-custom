"""Real idle captures keep source-owned artifact reads and sealed construction."""

import pickle
from copy import copy, deepcopy
from dataclasses import replace

import pytest

from nanobot.session import idle_checkpoint as idle
from nanobot.session.idle_compaction import IdleCaptureToken
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source_store import SourceSessionStore
from tests.session.test_idle_attestation_ownership import issued
from tests.session.test_idle_checkpoint import _admit, _artifact_policy, _seed


def require_boundary():
    assert callable(getattr(SourceSessionStore, "issue_idle_capture_artifacts", None)), (
        "Source-owned idle capture issuance is missing"
    )
    assert callable(getattr(SourceSessionStore, "captured_idle_root_descriptors", None))
    assert callable(getattr(SourceSessionStore, "idle_capture_identity_differs", None))
    assert callable(getattr(idle.IdleCheckpointCapture, "captured_root_descriptors", None))
    assert isinstance(getattr(idle, "IdleCaptureIssuer", None), type)


def plain(tmp_path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    source = backend.source
    before = _admit(source, _seed(backend))
    return backend, source, before


def observe_issuer(monkeypatch):
    seen = []
    operation = SourceSessionStore.issue_idle_capture_artifacts

    def observe(source, issuer, *args, **kwargs):
        seen.append(issuer)
        return operation(source, issuer, *args, **kwargs)

    monkeypatch.setattr(SourceSessionStore, "issue_idle_capture_artifacts", observe)
    return seen


def capture_plain(source, before):
    result = source.capture_idle_checkpoint(
        before.state.session_key, runtime_fingerprint="runtime-1", policy_fingerprint="policy-1",
    )
    assert result is not None
    return result


async def test_real_artifact_capture_is_detached_sealed_and_read_only(tmp_path, monkeypatch):
    require_boundary()
    source, before, _, descriptor, _ = await issued(tmp_path)
    seen = observe_issuer(monkeypatch)
    capture = source.capture_idle_checkpoint(
        before.state.session_key, runtime_fingerprint="runtime-1", policy_fingerprint="policy-1",
        policy=_artifact_policy(tmp_path / "workspace"),
    )
    assert capture is not None and capture.dependencies is not None
    assert len(seen) == 1 and seen[0].require_source(source) is None
    assert capture.require_attestation() is None
    assert capture.snapshot is not before
    assert capture.dependencies.matches_capture(source, capture) is True
    assert capture.captured_root_descriptors() == (descriptor,)
    assert source.captured_idle_root_descriptors(capture) == (descriptor,)
    assert source.idle_capture_identity_differs(capture.token) is False
    assert source.load_source_snapshot(before.state.session_key) == before


@pytest.mark.parametrize("operation", [copy, deepcopy, pickle.dumps])
def test_transient_issuer_cannot_be_copied_serialized_or_expose_storage(tmp_path, monkeypatch, operation):
    require_boundary()
    _, source, before = plain(tmp_path)
    seen = observe_issuer(monkeypatch)
    capture = capture_plain(source, before)
    assert len(seen) == 1
    issuer = seen[0]
    assert "SourceSessionStore" not in repr(issuer)
    assert not any(hasattr(issuer, name) for name in (
        "source", "store", "backend", "fd", "capability", "publish", "write",
    ))
    with pytest.raises(TypeError, match="Idle capture issuer"):
        operation(issuer)
    assert capture.require_attestation() is None
    assert source.load_source_snapshot(before.state.session_key) == before


@pytest.mark.parametrize("kind", ["constructor", "operation"])
def test_unbound_issuer_refusal_precedes_hostile_source_fields(tmp_path, monkeypatch, kind):
    require_boundary()
    _, source, before = plain(tmp_path)
    read = SourceSessionStore.__getattribute__

    def hostile(self, name):
        if self is source and name in {"_artifact_store", "_backend"}:
            raise AssertionError("unbound issuer touched source authority")
        return read(self, name)

    with monkeypatch.context() as patch:
        patch.setattr(SourceSessionStore, "__getattribute__", hostile)
        with pytest.raises(idle.IdleCheckpointError, match="Invalid idle capture issuer"):
            if kind == "constructor":
                idle.IdleCaptureIssuer(object(), source)
            else:
                SourceSessionStore.issue_idle_capture_artifacts(
                    source, object(), before, None, before, object(),
                    runtime_fingerprint="runtime-1", policy_fingerprint="policy-1",
                    policy=None, root_authority=None,
                )
    assert source.load_source_snapshot(before.state.session_key) == before


@pytest.mark.parametrize("phase", range(1, 11))
async def test_each_original_live_store_read_retains_first_fault_and_no_source_write(tmp_path, monkeypatch, phase):
    require_boundary()
    source, before, _, _, _ = await issued(tmp_path)
    read = SourceSessionStore.__getattribute__
    failure = RuntimeError("controlled configured-store read failure")
    observed = []

    def fail(self, name):
        if self is source and name == "_artifact_store":
            observed.append(name)
            if len(observed) == phase:
                raise failure
        return read(self, name)

    with monkeypatch.context() as patch:
        patch.setattr(SourceSessionStore, "__getattribute__", fail)
        with pytest.raises(RuntimeError) as caught:
            source.capture_idle_checkpoint(
                before.state.session_key, runtime_fingerprint="runtime-1", policy_fingerprint="policy-1",
                policy=_artifact_policy(tmp_path / "workspace"),
            )
        assert caught.value is failure
        assert len(observed) == phase
    assert source.load_source_snapshot(before.state.session_key) == before


@pytest.mark.parametrize("shape", ["missing", "empty", "none", "version", "container", "nonempty"])
def test_no_store_descriptor_read_retains_manifest_refusal_without_admitting_a_write(tmp_path, shape):
    require_boundary()
    _, source, before = plain(tmp_path)
    capture = capture_plain(source, before)
    metadata = capture.snapshot.header["metadata"]
    values = {
        "empty": {"schema_version": 1, "descriptors": {}},
        "none": None,
        "version": {"schema_version": 2, "descriptors": {}},
        "container": {"schema_version": 1, "descriptors": []},
        "nonempty": {"schema_version": 1, "descriptors": {"unavailable": {}}},
    }
    if shape == "missing":
        metadata.pop("artifact_manifest", None)
    else:
        metadata["artifact_manifest"] = values[shape]
    if shape in {"missing", "empty"}:
        assert capture.captured_root_descriptors() == ()
        assert idle._captured_root_descriptors(capture) == ()
    else:
        with pytest.raises(idle.ArtifactValidationError):
            capture.captured_root_descriptors()
        with pytest.raises(idle.ArtifactValidationError):
            idle._captured_root_descriptors(capture)
    assert source.load_source_snapshot(before.state.session_key) == before


async def test_descriptor_reader_selection_is_late_and_uses_the_original_captured_source(tmp_path, monkeypatch):
    require_boundary()
    source, before, _, descriptor, capture = await issued(tmp_path)
    reader = idle._root_descriptors
    observed = []

    def read(actual_source, snapshot):
        observed.append((actual_source, snapshot))
        return reader(actual_source, snapshot)

    monkeypatch.setattr(idle, "_root_descriptors", read)
    assert idle._captured_root_descriptors(capture) == (descriptor,)
    assert observed == [(source, capture.snapshot)]
    assert source.load_source_snapshot(before.state.session_key) == before


@pytest.mark.parametrize("different", [False, True])
def test_identity_inequality_keeps_token_then_source_operand_order(tmp_path, monkeypatch, different):
    require_boundary()
    _, source, before = plain(tmp_path)
    capture = capture_plain(source, before)
    token = replace(capture.token, store_identity="different") if different else capture.token
    token_read = IdleCaptureToken.__getattribute__
    source_read = SourceSessionStore.__getattribute__
    observed = []

    def read_token(self, name):
        if self is token and name == "store_identity":
            observed.append("token")
        return token_read(self, name)

    def read_source(self, name):
        if self is source and name == "_store_identity":
            observed.append("source")
        return source_read(self, name)

    with monkeypatch.context() as patch:
        patch.setattr(IdleCaptureToken, "__getattribute__", read_token)
        patch.setattr(SourceSessionStore, "__getattribute__", read_source)
        assert source.idle_capture_identity_differs(token) is different
    assert observed == ["token", "source"]
    assert capture.require_attestation() is None
    assert source.load_source_snapshot(before.state.session_key) == before


def test_issued_constructor_refuses_a_foreign_source_identity(tmp_path, monkeypatch):
    require_boundary()
    _, source, before = plain(tmp_path)
    seen = observe_issuer(monkeypatch)
    capture = capture_plain(source, before)
    foreign = JsonlSessionStore(tmp_path / "foreign", sessions_root=tmp_path / "foreign-sessions").source
    with pytest.raises(idle.IdleCheckpointError, match="Invalid idle capture issuer"):
        seen[0].require_source(foreign)
    assert capture.require_attestation() is None
    assert source.load_source_snapshot(before.state.session_key) == before


@pytest.mark.parametrize("name,phase", [("IdleDependencyAccess", 3), ("_IdleCaptureAttestation", 5)])
async def test_constructor_is_selected_before_its_live_configured_store_operands(tmp_path, monkeypatch, name, phase):
    require_boundary()
    source, before, _, _, _ = await issued(tmp_path)
    constructor = getattr(idle, name)
    source_read = SourceSessionStore.__getattribute__
    calls = []
    reads = []

    def replacement(*args, **kwargs):
        calls.append(name)
        return constructor(*args, **kwargs)

    with monkeypatch.context() as patch:
        def read(self, field):
            if self is source and field == "_artifact_store":
                reads.append(field)
                if len(reads) == phase:
                    patch.setattr(idle, name, replacement)
            return source_read(self, field)

        patch.setattr(SourceSessionStore, "__getattribute__", read)
        if name == "IdleDependencyAccess":
            # The original exact-type guard must still refuse the temporarily
            # replaced class name, even when the original constructor was selected.
            with pytest.raises(idle.IdleCheckpointError, match="Invalid idle dependency capability"):
                source.capture_idle_checkpoint(
                    before.state.session_key, runtime_fingerprint="runtime-1", policy_fingerprint="policy-1",
                    policy=_artifact_policy(tmp_path / "workspace"),
                )
            capture = None
        else:
            capture = source.capture_idle_checkpoint(
                before.state.session_key, runtime_fingerprint="runtime-1", policy_fingerprint="policy-1",
                policy=_artifact_policy(tmp_path / "workspace"),
            )
    assert calls == []
    assert len(reads) == (4 if name == "IdleDependencyAccess" else 10)
    if capture is not None:
        assert capture.require_attestation() is None
    assert source.load_source_snapshot(before.state.session_key) == before


async def test_attestation_setter_is_selected_before_its_live_configured_store_operands(tmp_path, monkeypatch):
    require_boundary()
    source, before, _, _, _ = await issued(tmp_path)
    source_read = SourceSessionStore.__getattribute__
    writes = []
    reads = []

    class ReplacementObject:
        @staticmethod
        def __setattr__(instance, field, value):
            writes.append(field)
            return object.__setattr__(instance, field, value)

    with monkeypatch.context() as patch:
        def read(self, field):
            if self is source and field == "_artifact_store":
                reads.append(field)
                if len(reads) == 5:
                    patch.setattr(idle, "object", ReplacementObject, raising=False)
            return source_read(self, field)

        patch.setattr(SourceSessionStore, "__getattribute__", read)
        capture = source.capture_idle_checkpoint(
            before.state.session_key, runtime_fingerprint="runtime-1", policy_fingerprint="policy-1",
            policy=_artifact_policy(tmp_path / "workspace"),
        )
    assert writes == []
    assert len(reads) == 10
    assert capture is not None and capture.require_attestation() is None
    assert source.load_source_snapshot(before.state.session_key) == before
