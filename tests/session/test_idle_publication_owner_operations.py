"""Original-entrypoint traces for the capture/publication ownership boundary."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from nanobot.session import idle_checkpoint as idle
from nanobot.session.source_store import SourceSessionStore
from tests.session.test_idle_checkpoint import _admit, _proposal, _seed


def _forbid(*args, **kwargs):
    raise AssertionError("unexpected field, override, helper or write")


@pytest.mark.parametrize("owner,name", [
    (idle.IdleCheckpointCapture, "admit_source_capture"),
    (idle.IdleCheckpointCapture, "issue_from_snapshot"),
    (idle.IdleCheckpointCapture, "prepare_source_publication"),
    (idle.IdleDependencyAccess, "matches_configured_store"),
    (idle.IdleDependencyAccess, "publish_in_artifact_transaction"),
    (idle.IdleDependencyAccess, "verify_captured_references"),
    (SourceSessionStore, "publish_idle_checkpoint_in_execution_transaction"),
    (SourceSessionStore, "commit_idle_checkpoint_unlocked"),
])
def test_new_owner_operation_exists(owner, name):
    assert callable(getattr(owner, name))


class _Probe:
    def __init__(self, trace, label, **values):
        self.__dict__.update(trace=trace, label=label, values=values)

    def __getattr__(self, name):
        self.trace.append(f"{self.label}.{name}")
        return self.values[name]


def _context(trace, name, *, enter=None, cleanup=None):
    @contextmanager
    def held():
        trace.append(name + ":enter")
        if enter is not None:
            raise enter
        try:
            yield
        finally:
            trace.append(name + ":exit")
            if cleanup is not None:
                raise cleanup
    return held()


@pytest.mark.parametrize("entry", ["capture", "publish"])
def test_exact_source_refusal_before_hostile_fields_and_overrides(entry):
    class Hostile:
        __getattribute__ = _forbid

    class Subclass(SourceSessionStore):
        __getattribute__ = _forbid
        capture_idle_checkpoint = _forbid
        publish_idle_checkpoint = _forbid

    for source in (Hostile(), object.__new__(Subclass)):
        with pytest.raises(idle.IdleCheckpointError, match="Invalid idle source store"):
            if entry == "capture":
                idle.capture_idle_checkpoint(source, "key", runtime_fingerprint="r",
                                             policy_fingerprint="p")
            else:
                idle.publish_idle_checkpoint(source, "key", object(), {},
                                             runtime_fingerprint="r", policy_fingerprint="p")


@pytest.mark.parametrize("entry", ["capture", "publish"])
@pytest.mark.parametrize("phase", ["runtime", "policy", "binding", "root"])
def test_admission_refusal_precedes_backend(monkeypatch, entry, phase):
    source = object.__new__(SourceSessionStore)
    trace = []
    original = SourceSessionStore.__getattribute__

    def read(self, name):
        if name == "_backend":
            _forbid()
        return original(self, name)

    monkeypatch.setattr(SourceSessionStore, "__getattribute__", read)
    runtime = "" if phase == "runtime" else "r"
    fingerprint = "" if phase == "policy" else "p"
    policy = SimpleNamespace(fingerprint="bad") if phase == "binding" else None

    def root(*args, **kwargs):
        trace.append("root")
        raise idle.IdleCheckpointError("root refusal")

    monkeypatch.setattr(idle, "_validate_root_authority", root)
    monkeypatch.setattr(idle, "_validate_capture_for_source", root)
    with pytest.raises(idle.IdleCheckpointError):
        if entry == "capture":
            idle.capture_idle_checkpoint(source, "key", runtime_fingerprint=runtime,
                                         policy_fingerprint=fingerprint, policy=policy)
        else:
            idle.publish_idle_checkpoint(source, "key", object(), {}, runtime_fingerprint=runtime,
                                         policy_fingerprint=fingerprint, policy=policy)
    assert trace == (["root"] if phase == "root" else [])


@pytest.mark.parametrize("descriptor", [False, True])
def test_capture_locks_read_snapshot_sidecar_and_late_issuance(monkeypatch, descriptor):
    trace = []
    snapshot = object()
    header, events = {}, []
    backend = SimpleNamespace(locked_session_files=lambda: _context(trace, "files"))
    authority = SimpleNamespace(
        operation=lambda value: _context(trace, "root"),
        lock_file=lambda name, blocking: _context(trace, "descriptor"),
    ) if descriptor else None

    def sidecar(key, **kwargs):
        trace.append("sidecar")
        monkeypatch.setattr(idle, "capture_idle_token",
                            lambda value, **kwargs: trace.append("issue-token") or None)
        return False

    source = _Probe(trace, "source", _backend=backend,
                    _read_unlocked=lambda *args, **kwargs: trace.append("read") or (header, events),
                    _snapshot=lambda *args: trace.append("snapshot") or snapshot,
                    _runtime_checkpoint_present=sidecar)
    monkeypatch.setattr(idle, "_source_store_type", lambda: _Probe)
    monkeypatch.setattr(idle, "_validate_root_authority", lambda *args: trace.append("admit-root"))
    monkeypatch.setattr(idle, "capture_idle_token", _forbid)
    assert idle.capture_idle_checkpoint(source, "key", runtime_fingerprint="r",
                                        policy_fingerprint="p", root_authority=authority) is None
    assert trace == ["admit-root", "source._backend"] + (
        ["root:enter", "descriptor:enter"] if descriptor else ["files:enter"]
    ) + ["source._read_unlocked", "read", "source._snapshot", "snapshot",
         "source._runtime_checkpoint_present", "sidecar", "issue-token"] + (
        ["descriptor:exit", "root:exit"] if descriptor else ["files:exit"]
    )


def test_token_absent_factory_does_not_read_source(monkeypatch):
    monkeypatch.setattr(idle, "capture_idle_token", lambda *args, **kwargs: None)
    assert idle._capture_with_snapshot(_Probe([], "source"), "key", object(),
                                       runtime_fingerprint="r", policy_fingerprint="p", policy=None,
                                       runtime_checkpoint_present=False, root_authority=None) is None


def test_factory_repeated_store_reads_detachment_and_late_helpers(tmp_path, monkeypatch):
    from nanobot.session.manager import JsonlSessionStore

    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    source = backend.source
    snapshot = _admit(source, _seed(backend))
    store = SimpleNamespace(workspace_id="w", session_key=snapshot.state.session_key,
                            incarnation=snapshot.state.incarnation, generation=snapshot.state.generation)
    source._artifact_store = store
    trace = []
    original = SourceSessionStore.__getattribute__

    def read(self, name):
        if self is source and name == "_artifact_store":
            trace.append("store")
        return original(self, name)

    def digest(value):
        trace.append("digest-first")
        monkeypatch.setattr(idle, "_snapshot_digest", lambda value: trace.append("digest-next") or "d")
        return "d"

    def descriptors(*args):
        trace.append("roots")
        monkeypatch.setattr(idle, "_snapshot_digest", digest)
        return ()

    monkeypatch.setattr(SourceSessionStore, "__getattribute__", read)
    monkeypatch.setattr(idle, "_root_descriptors", descriptors)
    monkeypatch.setattr(idle, "_snapshot_digest", _forbid)
    monkeypatch.setattr(idle, "_descriptor_digest", lambda value: trace.append("descriptors") or "e")
    capture = idle._capture_with_snapshot(source, "key", snapshot, runtime_fingerprint="r",
                                          policy_fingerprint="p", policy=SimpleNamespace(fingerprint="p"),
                                          runtime_checkpoint_present=False, root_authority=None)
    assert trace == ["store", "roots", "store", "store", "digest-first", "digest-next",
                     "store", "store", "store", "store", "store", "store", "digest-next",
                     "descriptors"]
    assert capture.snapshot is not snapshot and capture.dependencies._snapshot is not capture.snapshot
    capture.snapshot.events[0]["content"] = "changed"
    assert snapshot.events[0]["content"] == "original"


def _publication(monkeypatch, *, dependent=True, rooted=True):
    trace = []
    header = {"metadata": {"last_user_activity_at": "exact-bytes", "user_activity_revision": 7},
              "revision": 10, "updated_at": "unchanged"}
    events = []
    snapshot = SimpleNamespace(state=SimpleNamespace(generation=2, revision=10, checkpoint_id=None))
    validated = {"boundary": 1, "summary": "facts"}
    references = ("a", "b") if dependent else ()
    proof = SimpleNamespace(candidate=SimpleNamespace(snapshot=snapshot, stream_id=None),
                            chunks=(SimpleNamespace(evidence=SimpleNamespace(boundary=1), summary="facts"),))
    backend = object()
    blobs = SimpleNamespace(assert_reachable=lambda: trace.append("reachable"))
    failures = {}

    @contextmanager
    def transaction(*, blocking):
        assert blocking is False
        trace.append("artifact:enter")
        if "entry" in failures:
            raise failures["entry"]
        try:
            yield blobs
        finally:
            trace.append("artifact:exit")
            if "cleanup" in failures:
                raise failures["cleanup"]

    store = SimpleNamespace(transaction=transaction)
    access = object.__new__(idle.IdleDependencyAccess)
    object.__setattr__(access, "_store", store)
    capture = _Probe(trace, "capture", token=object(), dependencies=access)

    def write(value, rows, **kwargs):
        trace.append("write")
        if "write" in failures:
            raise failures["write"]
        return deepcopy(value)

    source = _Probe(trace, "source", _backend=backend, _artifact_store=store,
                    _read_unlocked=lambda *args, **kwargs: trace.append("read") or (header, events),
                    _snapshot=lambda *args: trace.append("snapshot") or snapshot,
                    _runtime_checkpoint_present=lambda *args, **kwargs: trace.append("sidecar") or False,
                    _write=write)
    authority = SimpleNamespace(operation=lambda value: _context(trace, "root")) if rooted else None

    @contextmanager
    def execution(value, key, *, root_authority):
        assert value is backend and root_authority is authority
        trace.append("execution:enter")
        try:
            if "body" in failures:
                raise failures["body"]
            yield
        finally:
            trace.append("execution:exit")

    def reachable():
        trace.append("reachable")
        monkeypatch.setattr(idle, "execution_transaction", execution)

    blobs.assert_reachable = reachable
    monkeypatch.setattr(idle, "execution_transaction", _forbid if dependent else execution)
    monkeypatch.setattr(idle, "_source_store_type", lambda: _Probe)
    monkeypatch.setattr(idle, "_validate_capture_for_source", lambda *args, **kwargs: trace.append("prepare"))
    monkeypatch.setattr(idle, "_candidate_and_proof", lambda *args: (validated, references, proof))
    monkeypatch.setattr(idle, "validate_idle_capture", lambda *args, **kwargs: trace.append("token"))
    monkeypatch.setattr(idle, "validate_context_checkpoint", lambda *args, **kwargs: trace.append("structural") or validated)
    monkeypatch.setattr(idle, "derive_summary_references", lambda *args: trace.append("proof") or references)

    def verify(self, reference, *args):
        trace.append("verify-first:" + reference)
        monkeypatch.setattr(idle.IdleDependencyAccess, "_verify",
                            lambda self, ref, *args: trace.append("verify-next:" + ref))

    monkeypatch.setattr(idle.IdleDependencyAccess, "_verify", verify)
    original = idle.IdleDependencyAccess.__getattribute__

    def dependency_read(self, name):
        if name == "_store":
            trace.append("access._store")
        return original(self, name)

    monkeypatch.setattr(idle.IdleDependencyAccess, "__getattribute__", dependency_read)
    return SimpleNamespace(source=source, capture=capture, policy=SimpleNamespace(fingerprint="p"),
                           authority=authority, proof=proof, blobs=blobs, validated=validated,
                           references=references, snapshot=snapshot, trace=trace,
                           failures=failures, header=header, events=events)


def _publish(case):
    return idle.publish_idle_checkpoint(case.source, "key", case.capture, {},
                                        runtime_fingerprint="r", policy_fingerprint="p",
                                        policy=case.policy, root_authority=case.authority)


@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("rooted", [False, True])
def test_publication_transaction_trace_and_exact_activity_bytes(monkeypatch, dependent, rooted):
    case = _publication(monkeypatch, dependent=dependent, rooted=rooted)
    result = _publish(case)
    prefix = ["prepare"] + (["capture.dependencies", "source._artifact_store", "access._store"]
                            if dependent else [])
    prefix += ["source._backend", "root:enter"] if rooted else []
    prefix += ["access._store", "artifact:enter", "reachable"] if dependent else []
    assert case.trace == prefix + ["source._backend", "execution:enter", "source._read_unlocked",
                                  "read", "source._snapshot", "snapshot", "capture.token",
                                  "source._runtime_checkpoint_present", "sidecar", "token",
                                  "structural", "proof"] + (
        ["capture.dependencies", "capture.dependencies", "verify-first:a",
         "capture.dependencies", "verify-next:b", "reachable"] if dependent else []
    ) + ["source._write", "write", "execution:exit"] + (
        ["artifact:exit"] if dependent else []
    ) + (["root:exit"] if rooted else [])
    assert result["metadata"] == {"last_user_activity_at": "exact-bytes",
                                  "user_activity_revision": 7, "context_checkpoint": case.validated}
    assert result["revision"] == 11 and result["updated_at"] == "unchanged"
    result["metadata"]["last_user_activity_at"] = "detached"
    assert case.header["metadata"]["last_user_activity_at"] == "exact-bytes"


@pytest.mark.parametrize("missing", ["access", "policy", "store"])
def test_exact_configured_store_gate_short_circuits(monkeypatch, missing):
    case = _publication(monkeypatch)
    if missing == "access":
        case.capture.values["dependencies"] = None
    elif missing == "policy":
        case.policy = None
    else:
        case.source.values["_artifact_store"] = object()
    with pytest.raises(idle.IdleCheckpointError, match="access is unavailable"):
        _publish(case)
    assert case.trace == ["prepare", "capture.dependencies"] + (
        ["source._artifact_store", "access._store"] if missing == "store" else []
    )


@pytest.mark.parametrize("missing", ["proof", "blobs", "dependencies", "policy"])
def test_reference_guard_short_circuits_before_loop_and_write(monkeypatch, missing):
    case = _publication(monkeypatch)
    if missing == "dependencies":
        case.capture.values["dependencies"] = None
    proof = None if missing == "proof" else case.proof
    blobs = None if missing == "blobs" else case.blobs
    policy = None if missing == "policy" else case.policy
    with pytest.raises(idle.IdleCheckpointError, match="access is unavailable"):
        idle._publish_unlocked(case.source, "key", case.capture, case.validated, case.references,
                               proof, policy=policy, blobs=blobs, root_authority=case.authority)
    assert case.trace == ["source._read_unlocked", "read", "source._snapshot", "snapshot",
                          "capture.token", "source._runtime_checkpoint_present", "sidecar",
                          "token", "structural"] + (["proof"] if proof is not None else []) + (
        ["capture.dependencies"] if missing in ("dependencies", "policy") else []
    )


@pytest.mark.parametrize("phase", ["entry", "body", "cleanup"])
def test_busy_translation_preserves_cleanup_extent(monkeypatch, phase):
    case = _publication(monkeypatch)
    failure = idle.ArtifactStorageBusyError("busy")
    case.failures[phase] = failure
    with pytest.raises(idle.ExecutionBusyError) as error:
        _publish(case)
    assert error.value.__context__ is failure and error.value.__suppress_context__
    assert case.trace[-1] == "root:exit"
    assert case.trace.count("write") == (1 if phase == "cleanup" else 0)
    assert case.trace.count("artifact:exit") == (0 if phase == "entry" else 1)


@pytest.mark.parametrize("failure", [RuntimeError("storage"), KeyboardInterrupt("system")])
@pytest.mark.parametrize("phase", ["entry", "body", "cleanup", "write"])
def test_nonbusy_and_baseexception_identity_no_retry(monkeypatch, failure, phase):
    case = _publication(monkeypatch)
    case.failures[phase] = failure
    with pytest.raises(type(failure)) as error:
        _publish(case)
    assert error.value is failure
    assert case.trace[-1] == "root:exit"
    assert case.trace.count("write") == (1 if phase in ("cleanup", "write") else 0)


def test_outer_root_cleanup_busy_is_translated_only_for_dependent_path(monkeypatch):
    for dependent in (True, False):
        case = _publication(monkeypatch, dependent=dependent)
        failure = idle.ArtifactStorageBusyError("root cleanup")
        case.authority.operation = lambda value: _context(case.trace, "root", cleanup=failure)
        expected = idle.ExecutionBusyError if dependent else idle.ArtifactStorageBusyError
        with pytest.raises(expected) as error:
            _publish(case)
        if not dependent:
            assert error.value is failure
        assert case.trace[-1] == "root:exit" and case.trace.count("write") == 1


def test_real_write_then_raise_has_one_attempt_and_detached_durable_output(tmp_path, monkeypatch):
    from nanobot.session.manager import JsonlSessionStore

    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    source = backend.source
    admitted = _admit(source, _seed(backend))
    capture = idle.capture_idle_checkpoint(source, admitted.state.session_key,
                                          runtime_fingerprint="r", policy_fingerprint="p")
    candidate = _proposal(capture.snapshot)
    original = source._write
    writes = []
    failure = RuntimeError("durable write then raise")

    def write(*args, **kwargs):
        writes.append(original(*args, **kwargs))
        raise failure

    monkeypatch.setattr(source, "_write", write)
    with pytest.raises(RuntimeError) as error:
        idle.publish_idle_checkpoint(source, admitted.state.session_key, capture, candidate,
                                     runtime_fingerprint="r", policy_fingerprint="p")
    assert error.value is failure and len(writes) == 1
    current = source.load_source_snapshot(admitted.state.session_key)
    assert current.state.revision == admitted.state.revision + 1
    assert current.checkpoint == candidate
    assert current.header["metadata"]["last_user_activity_at"] == admitted.header["metadata"]["last_user_activity_at"]
    writes[0].events[0]["content"] = "detached"
    assert source.load_source_snapshot(admitted.state.session_key).events[0]["content"] == "original"


def test_compatibility_wildcard_exports_are_exact_and_inert(monkeypatch):
    from nanobot.session.artifact_store import PrivateArtifactStore

    monkeypatch.setattr(idle, "capture_idle_token", _forbid)
    monkeypatch.setattr(SourceSessionStore, "_read_unlocked", _forbid)
    monkeypatch.setattr(SourceSessionStore, "__init__", _forbid)
    monkeypatch.setattr(SourceSessionStore, "_write", _forbid)
    monkeypatch.setattr(PrivateArtifactStore, "transaction", _forbid)
    namespace = {}
    exec("from nanobot.session.idle_checkpoint import *", namespace)
    expected = [
        "IdleCheckpointCapture",
        "IdleCheckpointConflict",
        "IdleCheckpointConflictError",
        "IdleCheckpointError",
        "IdleDependencyAccess",
        "IdlePolicyBinding",
        "capture_idle_checkpoint",
        "project_idle_root_nodes",
        "publish_idle_checkpoint",
        "_capture_with_snapshot",
        "_publish_unlocked",
    ]
    assert idle.__all__ == expected
    assert len(idle.__all__) == len(set(idle.__all__))
    assert set(namespace) - {"__builtins__"} == set(expected)
    assert namespace["_capture_with_snapshot"] is idle._capture_with_snapshot
    assert namespace["_publish_unlocked"] is idle._publish_unlocked
