"""Additive owner-operation and late-lookup preservation checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nanobot.session import idle_checkpoint as idle
from nanobot.session.source_store import SourceSessionStore


def _forbid(*args, **kwargs):
    raise AssertionError("unexpected override, helper or storage access")


@pytest.mark.parametrize("owner,name", [
    (idle.IdleCheckpointCapture, "require_attestation"),
    (SourceSessionStore, "require_idle_root_authority"),
    (SourceSessionStore, "idle_root_descriptors"),
])
def test_new_owner_operation_exists(owner, name):
    assert callable(getattr(owner, name))


def _sealed(monkeypatch, *, dependent=False):
    source = object.__new__(SourceSessionStore)
    source._artifact_store = None
    snapshot = SimpleNamespace(marker="snapshot")
    token, nonce = object(), object()
    capture = object.__new__(idle.IdleCheckpointCapture)
    dependency = object.__new__(idle.IdleDependencyAccess) if dependent else None
    fields = dict(token=token, snapshot=snapshot, runtime_fingerprint="runtime",
                  policy_fingerprint="policy", _source=source, _nonce=nonce,
                  dependencies=dependency, root_authority=None)
    for name, value in fields.items():
        object.__setattr__(capture, name, value)
    if dependent:
        for name, value in dict(_source=source, _capture_nonce=nonce, _token=token,
                                _store=None, _policy_fingerprint="policy",
                                _snapshot_digest="digest", _snapshot=object(),
                                _descriptors=()).items():
            object.__setattr__(dependency, name, value)
    seal = idle._IdleCaptureAttestation(
        idle._CAPABILITY, id(capture), nonce, token, source, None, None,
        "runtime", "policy", "digest", dependent,
        id(dependency) if dependent else None, "descriptors" if dependent else None, None,
    )
    object.__setattr__(capture, "_attestation", seal)
    monkeypatch.setattr(idle, "_snapshot_digest", lambda value: "digest")
    monkeypatch.setattr(idle, "_descriptor_digest", lambda value: "descriptors")
    return capture, seal, source, dependency


@pytest.mark.parametrize("field,value", [
    ("_capability", object()), ("_capture_id", -1), ("_nonce", object()),
    ("_token", object()), ("_source", object()), ("_runtime_fingerprint", "bad"),
    ("_policy_fingerprint", "bad"), ("_root_authority", object()),
])
def test_attestation_identity_short_circuits_before_source_helper(monkeypatch, field, value):
    capture, seal, _, _ = _sealed(monkeypatch)
    object.__setattr__(seal, field, value)
    monkeypatch.setattr(idle, "_source_store_type", _forbid)
    with pytest.raises(idle.IdleCheckpointError, match="Idle capture attestation changed"):
        idle._validate_capture_attestation(capture)


@pytest.mark.parametrize("dependent", [False, True])
def test_attestation_repeated_reads_and_live_digest_helpers(monkeypatch, dependent):
    capture, _, _, _ = _sealed(monkeypatch, dependent=dependent)
    trace = []
    original = idle.IdleCheckpointCapture.__getattribute__

    def read(self, name):
        trace.append(name)
        if name == "_source" and trace.count(name) == 2:
            monkeypatch.setattr(idle, "_source_store_type", lambda: SourceSessionStore)
        return original(self, name)

    monkeypatch.setattr(idle.IdleCheckpointCapture, "__getattribute__", read)
    monkeypatch.setattr(idle, "_source_store_type", _forbid)

    def digest(value):
        trace.append("digest-first")
        monkeypatch.setattr(idle, "_snapshot_digest", lambda value: trace.append("digest-next") or "digest")
        monkeypatch.setattr(idle, "_descriptor_digest", lambda value: trace.append("descriptors-live") or "descriptors")
        return "digest"

    monkeypatch.setattr(idle, "_snapshot_digest", digest)
    idle._validate_capture_attestation(capture)
    assert trace == [
        "_attestation", "_nonce", "token", "_source", "runtime_fingerprint",
        "policy_fingerprint", "root_authority", "_source", "_source", "snapshot",
        "digest-first", "dependencies",
    ] + (["_source", "_nonce", "token", "_source", "policy_fingerprint",
          "digest-next", "descriptors-live"] if dependent else [])


@pytest.mark.parametrize("field,value,message", [
    ("_store", object(), "artifact store binding"),
    ("_store_binding", (), "artifact store binding"),
    ("_snapshot_digest", "bad", "snapshot changed"),
    ("_dependency_present", True, "dependency authority"),
    ("_dependency_id", 1, "dependency authority"),
    ("_descriptor_digest", "bad", "dependency authority"),
])
def test_attestation_later_refusals(monkeypatch, field, value, message):
    capture, seal, _, _ = _sealed(monkeypatch)
    object.__setattr__(seal, field, value)
    with pytest.raises(idle.IdleCheckpointError, match=message):
        idle._validate_capture_attestation(capture)


@pytest.mark.parametrize("field", ["_source", "_capture_nonce", "_token", "_store",
                                  "_policy_fingerprint", "_snapshot_digest"])
def test_dependency_refusals(monkeypatch, field):
    capture, _, _, dependency = _sealed(monkeypatch, dependent=True)
    object.__setattr__(dependency, field, object())
    with pytest.raises(idle.IdleCheckpointError, match="dependency authority"):
        idle._validate_capture_attestation(capture)


def test_hostile_capture_and_subclass_overrides_are_refused_before_reads():
    class Hostile:
        __getattribute__ = _forbid

    class CaptureSubclass(idle.IdleCheckpointCapture):
        require_attestation = _forbid
        __getattribute__ = _forbid

    for value in (Hostile(), object.__new__(CaptureSubclass)):
        with pytest.raises(idle.IdleCheckpointError, match="Invalid idle checkpoint capture"):
            idle._validate_capture_attestation(value)


class _Probe:
    def __init__(self, trace, label, **values):
        self.__dict__.update(trace=trace, label=label, values=values, hooks={})

    def __getattr__(self, name):
        self.trace.append(f"{self.label}.{name}")
        hook = self.hooks.get(name)
        if hook:
            hook()
        return self.values[name]


def _descriptor_binding():
    trace = []
    store = _Probe(trace, "store", workspace_id="workspace", session_key="key",
                   incarnation="inc", generation=2)
    state = _Probe(trace, "state", session_key="key", incarnation="inc", generation=2,
                   store_identity="identity")
    backend = _Probe(trace, "backend", sessions_dir=SimpleNamespace(name="workspace"))
    source = _Probe(trace, "source", _artifact_store=store, _backend=backend,
                    _store_identity="identity")
    snapshot = SimpleNamespace(state=state, events=[])
    return source, snapshot, store, trace


def test_source_wrappers_bypass_type_helper_and_receiver_overrides(monkeypatch):
    monkeypatch.setattr(idle, "_source_store_type", _forbid)

    class Duck:
        _artifact_store = None
        require_idle_root_authority = _forbid
        idle_root_descriptors = _forbid

    class Subclass(SourceSessionStore):
        require_idle_root_authority = _forbid
        idle_root_descriptors = _forbid

    source = object.__new__(Subclass)
    source._artifact_store = None
    for value in (Duck(), source):
        assert idle._validate_root_authority(value, None) is None
        assert idle._root_descriptors(value, object()) == ()


@pytest.mark.parametrize("operand", range(5))
def test_descriptor_binding_short_circuit_order(monkeypatch, operand):
    source, snapshot, store, trace = _descriptor_binding()
    fields = ["workspace_id", "session_key", "incarnation", "generation"]
    if operand < 4:
        store.values[fields[operand]] = "mismatch"
    else:
        source.values["_store_identity"] = "mismatch"
    monkeypatch.setattr(idle, "published_artifacts", _forbid)
    monkeypatch.setattr(idle, "_source_store_type", _forbid)
    with pytest.raises(idle.ArtifactValidationError, match="binding changed"):
        idle._root_descriptors(source, snapshot)
    expected = ["source._artifact_store", "store.workspace_id", "source._backend",
                "backend.sessions_dir", "store.session_key", "state.session_key",
                "store.incarnation", "state.incarnation", "store.generation",
                "state.generation", "state.store_identity", "source._store_identity"]
    assert trace == expected[:[4, 6, 8, 10, 12][operand]]


def test_descriptors_late_publication_parser_lookup_order_filter_and_detachment(monkeypatch):
    source, snapshot, store, trace = _descriptor_binding()
    snapshot.events = [
        {"_event_id": "a", "_stream_id": "root"},
        {"_event_id": "b", "_stream_id": "root"},
        {"_event_id": "child", "_stream_id": "root", "_parent_stream_id": "parent"},
        {"_event_id": "wrong", "_stream_id": "other"},
    ]

    class Descriptor:
        stream_id = "root"

        def __init__(self, event_id):
            self.event_id = event_id

        def to_dict(self):
            trace.append("to_dict:" + self.event_id)
            monkeypatch.setattr(idle, "parse_artifact_descriptor", second)
            return {"event_id": self.event_id}

    def first(raw):
        trace.append("parse-first:" + raw["event_id"])
        return dict(raw)

    def second(raw):
        trace.append("parse-second:" + raw["event_id"])
        return dict(raw)

    def published(value, *, workspace_id):
        assert value is snapshot and workspace_id == "workspace"
        trace.append("published-live")
        return tuple(Descriptor(name) for name in ("a", "b", "child", "wrong", "absent"))

    monkeypatch.setattr(idle, "published_artifacts", _forbid)
    store.hooks["generation"] = lambda: monkeypatch.setattr(idle, "published_artifacts", published)
    monkeypatch.setattr(idle, "parse_artifact_descriptor", first)
    monkeypatch.setattr(idle, "_source_store_type", _forbid)
    result = idle._root_descriptors(source, snapshot)
    assert result == ({"event_id": "a"}, {"event_id": "b"})
    assert trace[-6:] == ["store.workspace_id", "published-live", "to_dict:a",
                          "parse-first:a", "to_dict:b", "parse-second:b"]
    result[0]["event_id"] = "detached"
    assert snapshot.events[0]["_event_id"] == "a"


@pytest.mark.parametrize("outcome", ["success", "translated", "system"])
def test_authority_method_backend_order_and_dynamic_catch(monkeypatch, outcome):
    trace = []
    backend = object()
    source = _Probe(trace, "source", _backend=backend)
    system = KeyboardInterrupt("system")

    class ChangedError(Exception):
        pass

    class Authority:
        def __getattribute__(self, name):
            trace.append("authority." + name)
            return object.__getattribute__(self, name)

        def require_current(self, value):
            assert value is backend
            trace.append("require-current")
            if outcome == "translated":
                monkeypatch.setattr(idle, "SourceRootAuthorityError", ChangedError)
                raise ChangedError("changed")
            if outcome == "system":
                raise system

    monkeypatch.setattr(idle, "SourceRootAuthority", Authority)
    monkeypatch.setattr(idle, "_source_store_type", _forbid)
    if outcome == "translated":
        with pytest.raises(idle.IdleCheckpointConflict, match="Idle source-root authority changed") as error:
            idle._validate_root_authority(source, Authority())
        assert error.value.__cause__ is None and error.value.__suppress_context__
        assert type(error.value.__context__) is ChangedError
    elif outcome == "system":
        with pytest.raises(KeyboardInterrupt) as error:
            idle._validate_root_authority(source, Authority())
        assert error.value is system
    else:
        assert idle._validate_root_authority(source, Authority()) is None
    assert trace == ["authority.require_current", "source._backend", "require-current"]


def test_authority_none_and_wrong_exact_type_do_not_read_hostile_backend(monkeypatch):
    source = _Probe([], "source")
    source.hooks["_backend"] = _forbid
    monkeypatch.setattr(idle, "_source_store_type", _forbid)
    assert idle._validate_root_authority(source, None) is None
    with pytest.raises(idle.IdleCheckpointError, match="Invalid idle source-root authority"):
        idle._validate_root_authority(source, object())
    assert source.trace == []
