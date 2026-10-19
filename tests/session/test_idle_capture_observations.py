"""Issued captures retain live identity checks without leaking sealed authority."""

import pickle
from copy import copy, deepcopy
from dataclasses import replace

import pytest

from nanobot.agent import idle_summary as summary
from nanobot.agent.idle_runtime import IdleRuntimeBinding
from nanobot.session import idle_checkpoint as idle
from nanobot.session.source_store import SourceSessionStore
from tests.agent.test_idle_summary import _adapter, _Attempt, _authority
from tests.session.test_idle_attestation_ownership import issued


def _plain(tmp_path):
    return _authority(tmp_path, [{"role": "user", "content": "remember this"}])


def test_source_observation_is_an_opaque_snapshot_not_live_authority(tmp_path):
    source, _, _, _, capture = _plain(tmp_path)
    before = source.load_source_snapshot(capture.token.session_key)
    observed = capture.observe_source_identity()
    assert observed.matches_source(source) is True
    assert observed.matches_source(object()) is False
    assert "SourceSessionStore" not in repr(observed)
    assert not any(hasattr(observed, name) for name in (
        "source", "source_store", "store", "read", "publish", "require_attestation",
    ))
    object.__setattr__(capture, "_source", object())
    assert observed.matches_source(source) is True
    assert capture.observe_source_identity().matches_source(source) is False
    with pytest.raises(idle.IdleCheckpointError, match="attestation changed"):
        capture.require_attestation()
    assert source.load_source_snapshot(capture.token.session_key) == before


@pytest.mark.parametrize("operation", [copy, deepcopy, pickle.dumps])
def test_source_observation_cannot_be_copied_or_serialized(tmp_path, operation):
    _, _, _, _, capture = _plain(tmp_path)
    observed = capture.observe_source_identity()
    with pytest.raises(TypeError, match="source identity observation"):
        operation(observed)


def test_source_observation_is_not_a_capture_attestation(tmp_path):
    source, _, registry, binding, capture = _plain(tmp_path)
    observed = capture.observe_source_identity()
    forged = replace(capture)
    assert forged.observe_source_identity().matches_source(source) is True
    adapter = _adapter(_Attempt(), source, registry, binding)
    assert adapter.transport.require_for_authority(registry, observed, binding) is None
    assert adapter.transport.require_for_authority(registry, source, binding) is None
    with pytest.raises(summary.IdleSummaryBindingError, match="capture attestation is invalid"):
        summary._validate_capture_binding(
            forged, binding, adapter._measurement_identity, registry,
        )


@pytest.mark.parametrize("fault", [RuntimeError("source read"), KeyboardInterrupt("source read")])
def test_observation_reads_the_live_source_once_and_preserves_its_fault(tmp_path, monkeypatch, fault):
    source, _, _, _, capture = _plain(tmp_path)
    trace = []
    original = idle.IdleCheckpointCapture.__getattribute__

    def read(self, name):
        if self is capture and name == "_source":
            trace.append(name)
        return original(self, name)

    monkeypatch.setattr(idle.IdleCheckpointCapture, "__getattribute__", read)
    assert capture.observe_source_identity().matches_source(source) is True
    assert trace == ["_source"]

    def fail(self, name):
        if self is capture and name == "_source":
            trace.append("fault")
            raise fault
        return original(self, name)

    monkeypatch.setattr(idle.IdleCheckpointCapture, "__getattribute__", fail)
    with pytest.raises(type(fault)) as caught:
        summary._qualified_transport(
            object(), binding_registry=object(), source_store=capture.observe_source_identity(),
        )
    assert caught.value is fault
    assert trace == ["_source", "fault"]


@pytest.mark.parametrize("point,end", [(None, 3), ("binding_registry", 1), ("source_store", 2), ("binding", 3)])
def test_transport_checks_keep_original_order_and_first_error(tmp_path, monkeypatch, point, end):
    source, _, registry, binding, capture = _plain(tmp_path)
    observed = capture.observe_source_identity()
    adapter = _adapter(_Attempt(), source, registry, binding)
    transport = adapter.transport
    trace = []
    original = summary.IdleSummaryQualifiedTransport.__getattribute__

    def read(self, name):
        if self is transport and name in {"binding_registry", "source_store", "binding"}:
            trace.append(name)
            if name == point:
                return object()
        return original(self, name)

    monkeypatch.setattr(summary.IdleSummaryQualifiedTransport, "__getattribute__", read)
    monkeypatch.setattr(summary.IdleSummaryTransportRegistry, "validate_transport", lambda *_: None)
    if point is None:
        assert transport.require_for_authority(registry, observed, binding) is None
    else:
        with pytest.raises(summary.IdleSummaryCapabilityError, match="foreign to this authority"):
            transport.require_for_authority(registry, observed, binding)
    assert trace == ["binding_registry", "source_store", "binding"][:end]


def test_capture_binding_comparison_keeps_binding_before_capture_read(tmp_path, monkeypatch):
    _, _, _, binding, capture = _plain(tmp_path)
    assert capture.matches_binding(binding) is True
    trace = []
    binding_read = IdleRuntimeBinding.__getattribute__
    capture_read = idle.IdleCheckpointCapture.__getattribute__

    def read_binding(self, name):
        if self is binding and name == "source_store":
            trace.append("binding")
            object.__setattr__(capture, "_source", object())
        return binding_read(self, name)

    def read_capture(self, name):
        if self is capture and name == "_source":
            trace.append("capture")
        return capture_read(self, name)

    monkeypatch.setattr(IdleRuntimeBinding, "__getattribute__", read_binding)
    monkeypatch.setattr(idle.IdleCheckpointCapture, "__getattribute__", read_capture)
    assert capture.matches_binding(binding) is False
    assert trace == ["binding", "capture"]


@pytest.mark.parametrize("different_source", [False, True])
async def test_dependency_summary_comparison_keeps_live_source_policy_order(tmp_path, monkeypatch, different_source):
    source, _, _, _, capture = await issued(tmp_path)
    access = capture.dependencies
    trace = []
    assert access is not None
    assert callable(getattr(access, "differs_from_summary_binding"))
    assert callable(getattr(capture, "matches_dependencies"))
    dep_read = idle.IdleDependencyAccess.__getattribute__
    policy_read = idle.IdlePolicyBinding.__getattribute__

    class Binding:
        @property
        def source_store(self):
            trace.append("binding.source_store")
            return object() if different_source else source

    policy = idle.IdlePolicyBinding("policy-1", tmp_path, None, lambda value: tmp_path / value)

    def read_dep(self, name):
        if self is access and name in {"_source", "_policy_fingerprint"}:
            trace.append("dependency." + name)
        return dep_read(self, name)

    def read_policy(self, name):
        if self is policy and name == "fingerprint":
            trace.append("policy.fingerprint")
        return policy_read(self, name)

    monkeypatch.setattr(idle.IdleDependencyAccess, "__getattribute__", read_dep)
    monkeypatch.setattr(idle.IdlePolicyBinding, "__getattribute__", read_policy)
    assert access.differs_from_summary_binding(Binding(), policy) is different_source
    assert trace == ["dependency._source", "binding.source_store"] + ([] if different_source else [
        "dependency._policy_fingerprint", "policy.fingerprint",
    ])


@pytest.mark.parametrize("field", [
    "_source", "_capture_nonce", "_token", "_store", "_workspace_id", "_session_key",
    "_incarnation", "_generation", "_policy_fingerprint", "_snapshot_digest", "_snapshot",
])
async def test_dependency_identity_observations_retain_all_original_binding_guards(tmp_path, field):
    source, before, store, _, capture = await issued(tmp_path)
    access = capture.dependencies
    assert access is not None
    assert access.matches_capture(source, capture) is True
    assert capture.matches_dependencies(access) is True
    changed_snapshot = deepcopy(access._snapshot)
    changed_snapshot.events[0]["content"] = "modified"
    changes = {
        "_source": object(), "_capture_nonce": object(), "_token": replace(capture.token),
        "_store": store.__class__(tmp_path / "foreign", workspace_id=store.workspace_id,
            session_key=store.session_key, incarnation=store.incarnation, generation=store.generation),
        "_workspace_id": "other", "_session_key": "other", "_incarnation": "other",
        "_generation": store.generation + 1, "_policy_fingerprint": "other",
        "_snapshot_digest": "0" * 64, "_snapshot": changed_snapshot,
    }
    object.__setattr__(access, field, changes[field])
    assert access.matches_capture(source, capture) is False
    assert capture.matches_dependencies(access) is False
    assert source.load_source_snapshot(before.state.session_key) == before


async def test_dependency_nonce_and_configured_store_are_still_live_reads(tmp_path, monkeypatch):
    source, _, store, _, capture = await issued(tmp_path)
    access = capture.dependencies
    assert access is not None and access.matches_capture(source, capture) is True
    trace = []
    dep_read = idle.IdleDependencyAccess.__getattribute__
    cap_read = idle.IdleCheckpointCapture.__getattribute__
    source_read = SourceSessionStore.__getattribute__

    def read_dep(self, name):
        if self is access and name in {"_source", "_capture_nonce", "_token", "_store"}:
            trace.append("dep." + name)
        return dep_read(self, name)

    def read_cap(self, name):
        if self is capture and name in {"_nonce", "token"}:
            trace.append("cap." + name)
        return cap_read(self, name)

    def read_source(self, name):
        if self is source and name == "_artifact_store":
            trace.append("source.store")
            return store
        return source_read(self, name)

    monkeypatch.setattr(idle.IdleDependencyAccess, "__getattribute__", read_dep)
    monkeypatch.setattr(idle.IdleCheckpointCapture, "__getattribute__", read_cap)
    monkeypatch.setattr(SourceSessionStore, "__getattribute__", read_source)
    assert access.matches_capture(source, capture) is True
    assert trace[:7] == ["dep._source", "dep._capture_nonce", "cap._nonce", "dep._token",
        "cap.token", "dep._store", "source.store"]


async def test_descriptor_observations_retain_hash_tuple_validation_and_live_lookup(tmp_path, monkeypatch):
    _, _, _, descriptor, capture = await issued(tmp_path)
    access = capture.dependencies
    assert access is not None
    assert access.descriptor_digest() == idle._descriptor_digest(access._descriptors)
    assert access.descriptors_differ((descriptor,)) is False
    assert access.descriptors_differ(()) is True
    trace = []
    original_digest = idle._descriptor_digest
    original_read = idle.IdleDependencyAccess.__getattribute__

    def first(descriptors):
        trace.append("first")
        return original_digest(descriptors)

    def next_digest(descriptors):
        trace.append("next")
        return original_digest(descriptors)

    def read(self, name):
        if self is access and name == "_descriptors":
            trace.append("read")
            monkeypatch.setattr(idle, "_descriptor_digest", next_digest)
        return original_read(self, name)

    monkeypatch.setattr(idle, "_descriptor_digest", first)
    monkeypatch.setattr(idle.IdleDependencyAccess, "__getattribute__", read)
    access.descriptor_digest()
    access.descriptor_digest()
    assert trace == ["read", "first", "read", "next"]
    object.__setattr__(access, "_descriptors", list(access._descriptors))
    assert access.descriptors_differ((descriptor,)) is False
    with pytest.raises(idle.IdleCheckpointError, match="Invalid idle dependency descriptor seal"):
        access.descriptor_digest()


@pytest.mark.parametrize("phase", ["measurement", "attempt", "terminal-accounting"])
async def test_summary_keeps_measurement_snapshot_and_fresh_surrounding_checks(tmp_path, monkeypatch, phase):
    source, _, registry, binding, capture = _plain(tmp_path)
    assert capture.observe_source_identity().matches_source(source) is True
    before = source.load_source_snapshot(capture.token.session_key)
    attempt = _Attempt()
    original_measure = attempt.measure_request
    original_attempt = attempt.attempt_once
    measurements = []
    mutation = object()

    def measure(messages, **kwargs):
        result = original_measure(messages, **kwargs)
        if phase == "measurement":
            object.__setattr__(capture, "_source", mutation)
        return result

    async def run(request):
        result = await original_attempt(request)
        if phase == "attempt":
            object.__setattr__(capture, "_source", mutation)
        return result

    def accounting(value):
        if phase == "terminal-accounting" and value.exact:
            object.__setattr__(capture, "_source", mutation)

    monkeypatch.setattr(attempt, "measure_request", measure)
    monkeypatch.setattr(attempt, "attempt_once", run)
    adapter = _adapter(attempt, source, registry, binding, accounting_sink=accounting)
    original_authority = adapter._require_measurement_authority

    def authority(*args):
        result = original_authority(*args)
        measurements.append("admitted")
        return result

    monkeypatch.setattr(adapter, "_require_measurement_authority", authority)
    if phase == "measurement":
        with pytest.raises(summary.IdleSummaryBindingError, match="qualified transport authority changed"):
            await adapter.summarize(capture, binding, deadline=adapter._now() + 3)
        # Both checks inside _measure use its original source snapshot. The
        # immediately following guard observes the changed capture and rejects.
        assert measurements == ["admitted", "admitted"]
        assert attempt.requests == []
    else:
        result = await adapter.summarize(capture, binding, deadline=adapter._now() + 3)
        assert result.proposal is None and result.status == "stale_binding"
        assert len(attempt.requests) == 1
    assert adapter.last_accounting.exact is (phase != "measurement")
    assert adapter.last_accounting.dispatched_calls == (0 if phase == "measurement" else 1)
    assert source.load_source_snapshot(capture.token.session_key) == before
