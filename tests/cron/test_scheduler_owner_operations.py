"""Preservation cases pass before relocation; new-surface cases fail by API absence.

Fault doubles never issue an owner permit. Private state is inspected only to
observe original partial states; supported paths alone exercise successful APIs.
"""

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

import nanobot.cron.action_store as action
import nanobot.cron.definition_store as definitions
import nanobot.cron.occurrence as occurrence
import nanobot.cron.occurrence_store as ledger
import nanobot.cron.scheduler_store as scheduler
from tests.cron.stage3_fixtures import definition


class Fault(BaseException):
    pass


class Registry:
    def __init__(self, events, fail=None):
        self.events, self.fail, self.items = events, fail, []

    def add(self, value):
        self.items.append(value)
        self.events.append("add")
        if self.fail == "add":
            raise FAILURE

    def discard(self, value):
        self.events.append("discard")
        self.items.remove(value)
        if self.fail == "discard":
            raise FAILURE

    def clear(self):
        self.events.append("clear")
        self.items.clear()

    def __iter__(self):
        return iter(self.items)


FAILURE = Fault("phase fault")
REPROOF = Fault("retained reproof fault")


class Binding:
    def __init__(self, events, fail=None):
        self.events, self.fail = events, fail

    @property
    def root_fd(self):
        return 901

    def assert_bound(self):
        self.events.append("bound")
        if self.fail == "bound":
            raise FAILURE

    def close(self):
        self.events.append("binding-close")
        if self.fail == "binding-close":
            raise FAILURE


class TracedStore(scheduler.SchedulerStore):
    def __setattr__(self, name, value):
        if name == "_lease" and hasattr(self, "events"):
            self.events.append("slot-clear" if value is None else "slot-install")
            if self.fail == "slot":
                raise FAILURE
        super().__setattr__(name, value)


def bare_store(events, fail=None):
    store = TracedStore.__new__(TracedStore)
    store._lease = None
    store.events, store.fail = events, fail
    return store


def fake_lease(store, binding, fd=902):
    lease = scheduler.SchedulerLease.__new__(scheduler.SchedulerLease)
    lease._store, lease._binding, lease._fd = store, binding, fd
    lease._active, lease._acquisition_read = True, False
    lease._pid, lease._inode = os.getpid(), (1, 2)
    return lease


@pytest.mark.parametrize("operation", ["reprove", "release"])
@pytest.mark.parametrize("phase", ["inactive", "pid", "slot", "bound", "check"])
def test_preservation_three_operand_guard_and_reproof(operation, phase, monkeypatch):
    events = []
    store = bare_store(events)
    lease = fake_lease(store, Binding(events, "bound" if phase == "bound" else None))
    store._lease = lease
    events.clear()
    if phase == "inactive":
        lease._active = False
    if phase == "pid":
        lease._pid = -1
    if phase == "slot":
        store._lease = None
        events.clear()

    def pid():
        events.append("pid")
        return 100

    lease._pid = -1 if phase == "pid" else 100
    monkeypatch.setattr(os, "getpid", pid)

    def check(*_args):
        events.append("check")
        raise FAILURE

    monkeypatch.setattr(store, "_check_file", check)
    if operation == "release" and phase in {"bound", "check"}:
        # Separately caught reproof fault, then exact original cleanup.
        monkeypatch.setattr(fcntl, "flock", lambda *_: events.append("unlock"))
        monkeypatch.setattr(os, "close", lambda *_: events.append("close"))
        registry = Registry(events)
        registry.items.append(lease)
        monkeypatch.setattr(scheduler, "_LEASES", registry)
    error = scheduler.SchedulerConflictError if phase in {"inactive", "pid", "slot"} else Fault
    with pytest.raises(error) as caught:
        getattr(lease, operation)()
    expected = {"inactive": [], "pid": ["pid"], "slot": ["pid"],
                "bound": ["pid", "bound"], "check": ["pid", "bound", "check"]}[phase]
    if operation == "release" and phase in {"bound", "check"}:
        expected = ["pid", *expected, "unlock", "close", "binding-close", "slot-clear", "discard"]
        assert not lease._active and store._lease is None and registry.items == []
    assert events == expected
    if error is Fault:
        assert caught.value is FAILURE
    else:
        assert str(caught.value) == "scheduler lease is inactive or foreign"


@pytest.mark.parametrize("phase", [None, "unlock", "close", "binding-close", "slot", "discard"])
def test_preservation_release_supersession_and_unconditional_retirement(phase, monkeypatch):
    events = []
    store = bare_store(events, "slot" if phase == "slot" else None)
    # Setup bypasses the deliberately failing assignment seam.
    object.__setattr__(store, "fail", None)
    lease = fake_lease(store, Binding(events, phase))
    store._lease = lease
    registry = Registry(events, phase)
    registry.items.append(lease)
    monkeypatch.setattr(scheduler, "_LEASES", registry)

    def reprove():
        events.append("reprove")
        # Original terminal clear is unconditional, even if reproof changes slot.
        object.__setattr__(store, "_lease", "successor sentinel")
        raise REPROOF

    def unlock(*_args):
        events.append("unlock")
        if phase == "unlock":
            raise FAILURE

    def close(_fd):
        events.append("close")
        if phase == "close":
            raise FAILURE

    monkeypatch.setattr(lease, "reprove", reprove)
    monkeypatch.setattr(fcntl, "flock", unlock)
    monkeypatch.setattr(os, "close", close)
    store.fail = phase
    events.clear()
    with pytest.raises(Fault) as caught:
        lease.release()
    assert caught.value is (REPROOF if phase is None else FAILURE)
    expected = ["reprove", "unlock", "close", "binding-close", "slot-clear", "discard"]
    stop = {"close": 3, "binding-close": 4, "slot": 5}.get(phase, 6)
    assert events == expected[:stop]
    assert lease._active is (phase in {"close", "binding-close"})
    assert store._lease == ("successor sentinel" if stop <= 5 else None)
    assert registry.items == ([lease] if stop <= 5 else [])


ACQUIRE_PHASES = ["open-root", "create", "chmod", "check-1", "fsync-file", "fsync-root",
                  "bound", "check-2", "busy", "flock", "constructor-stat", "add", "slot",
                  "final-reprove", "cleanup-close", "cleanup-binding"]


@pytest.mark.parametrize("phase", [None, *ACQUIRE_PHASES, "existing", "chmod-exists"])
def test_preservation_acquisition_complete_phase_matrix(phase, monkeypatch):
    events = []
    store = bare_store(events)
    binding = Binding(events, "binding-close" if phase == "cleanup-binding" else None)
    registry = Registry(events, "add" if phase == "add" else None)
    monkeypatch.setattr(scheduler, "_LEASES", registry)

    def step(name):
        events.append(name)
        if name == phase:
            raise FAILURE

    class Files(scheduler._PrivateFiles):
        def _open_root_binding(self):
            step("open-root")
            return binding

    store._files = Files.__new__(Files)
    opens = []

    def opened(_name, flags, *_args, **_kwargs):
        if flags & os.O_EXCL:
            step("create")
            if phase == "existing":
                raise FileExistsError("existing")
        else:
            step("existing-open")
        fd = 902 + len(opens)
        opens.append(fd)
        return fd

    def chmod(_fd, _mode):
        step("chmod")
        if phase == "chmod-exists":
            raise FileExistsError("fchmod catch extent")

    stats = []

    def fstat(_fd):
        name = "constructor-stat" if stats else "inode-stat"
        stats.append(name)
        step(name)
        return os.stat_result((0o100600, 2, 1, 1, os.geteuid(), 0, 0, 0, 0, 0))

    checks = []

    def check(*_args):
        checks.append(1)
        step("check-" + str(len(checks)))

    def fsync(fd):
        step("fsync-root" if fd == 901 else "fsync-file")

    def bound():
        step("bound")

    def flock(_fd, flags):
        assert flags == fcntl.LOCK_EX | fcntl.LOCK_NB
        step("flock")
        if phase == "busy":
            raise BlockingIOError("busy sentinel")

    def final(lease):
        step("final-reprove")
        assert registry.items == [lease] and store._lease is lease
        if phase in {"cleanup-close", "cleanup-binding"}:
            raise REPROOF

    def close(_fd):
        step("close")
        if phase == "cleanup-close":
            raise FAILURE

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "fchmod", chmod)
    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(fcntl, "flock", flock)
    monkeypatch.setattr(binding, "assert_bound", bound)
    monkeypatch.setattr(store, "_check_file", check)
    monkeypatch.setattr(scheduler.SchedulerLease, "reprove", final)
    store.fail = "slot" if phase == "slot" else None
    if phase in {None, "existing", "chmod-exists"}:
        result = store.acquire_lease()
        assert result is store._lease and result._active and registry.items == [result]
    else:
        with pytest.raises((Fault, scheduler.SchedulerConflictError)) as caught:
            store.acquire_lease()
        if phase == "busy":
            assert str(caught.value) == "scheduler lease busy"
            assert isinstance(caught.value.__cause__, BlockingIOError)
        else:
            assert caught.value is FAILURE
    ordered = ["open-root", "create", "chmod", "inode-stat", "check-1", "fsync-file",
               "fsync-root", "bound", "check-2", "flock", "constructor-stat", "add",
               "slot-install", "final-reprove"]
    if phase == "existing":
        ordered[2:3] = ["existing-open"]
    if phase == "chmod-exists":
        ordered[3:3] = ["existing-open"]
        assert opens == [902, 903]  # Original leaked first fd is retained, not repaired.
    if phase not in {None, "existing", "chmod-exists"}:
        edge = {"busy": "flock", "slot": "slot-install", "cleanup-close": "final-reprove",
                "cleanup-binding": "final-reprove"}.get(phase, phase)
        ordered = ordered[:ordered.index(edge) + 1]
        if phase != "open-root":
            if opens:
                ordered.append("close")
            if phase != "cleanup-close":
                ordered.append("binding-close")
    assert events == ordered
    installed = phase in {None, "existing", "chmod-exists", "final-reprove", "cleanup-close", "cleanup-binding"}
    assert (store._lease is not None) is installed
    registered = installed or phase in {"add", "slot"}
    assert bool(registry.items) is registered
    assert all(lease._active for lease in registry.items)


@pytest.mark.parametrize("phase", ["enter", "read", "before-read", "after-read", "exit", "foreign"])
def test_preservation_refresh_consumed_before_scope_no_rollback(phase, monkeypatch):
    events = []
    store = bare_store(events)
    issuer = bare_store([]) if phase == "foreign" else store
    lease = fake_lease(issuer, Binding(events))
    count = 0

    def reprove():
        nonlocal count
        count += 1
        events.append("reprove")
        if (phase == "before-read" and count == 2) or (phase == "after-read" and count == 3):
            raise FAILURE

    class Files(scheduler._PrivateFiles):
        @contextmanager
        def _held_lock(self):
            events.append("enter")
            assert lease._acquisition_read
            if phase == "enter":
                raise FAILURE
            yield Binding(events)
            events.append("exit")
            if phase == "exit":
                raise FAILURE

    def read(_held, *, reconciliation_refresh):
        events.append(("read", reconciliation_refresh))
        if phase == "read":
            raise FAILURE
        return {"state": "sentinel"}

    store._files = Files.__new__(Files)
    monkeypatch.setattr(lease, "reprove", reprove)
    monkeypatch.setattr(store, "_read", read)
    with pytest.raises(scheduler.SchedulerConflictError if phase == "foreign" else Fault) as caught:
        store.read_for_acquisition(lease)
    if phase == "foreign":
        assert events == ["reprove"] and not lease._acquisition_read
        assert str(caught.value) == "scheduler acquisition lease foreign"
    else:
        assert lease._acquisition_read and caught.value is FAILURE
        assert events == {"enter": ["reprove", "enter"],
            "before-read": ["reprove", "enter", "reprove"],
            "read": ["reprove", "enter", "reprove", ("read", True)],
            "after-read": ["reprove", "enter", "reprove", ("read", True), "reprove"],
            "exit": ["reprove", "enter", "reprove", ("read", True), "reprove", "exit"]}[phase]


@pytest.mark.parametrize("position", [0, 1, None])
@pytest.mark.parametrize("fault", ["close", "binding-close"])
def test_preservation_fork_tuple_abort_and_final_clear(position, fault, monkeypatch):
    events = []
    registry = Registry(events)
    for index in range(3):
        binding = Binding(events, fault if index == position else None)
        registry.items.append(fake_lease(bare_store([]), binding, 902 + index))
    leases = tuple(registry.items)

    def close(fd):
        events.append(("close", fd))
        if fault == "close" and fd == 902 + (position if position is not None else -1):
            raise FAILURE

    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(fcntl, "flock", lambda *_: pytest.fail("fork must not unlock"))
    monkeypatch.setattr(scheduler, "_LEASES", registry)
    if position is None:
        scheduler._fork_child()
    else:
        with pytest.raises(Fault) as caught:
            scheduler._fork_child()
        assert caught.value is FAILURE
    expected = []
    for index in range(3 if position is None else position + 1):
        expected.append(("close", 902 + index))
        if index != position or fault != "close":
            expected.append("binding-close")
    if position is None:
        expected.append("clear")
    assert events == expected
    assert [lease._active for lease in leases] == [position is not None and index >= position for index in range(3)]
    assert registry.items == ([] if position is None else list(leases))


@pytest.mark.parametrize("bad", [b'\xff', b'{"x":1,"x":2}', b'NaN', b'Infinity', b'[' * 2000])
@pytest.mark.parametrize("label", ["scheduler state", "pinned scheduler state"])
def test_preservation_strict_labels_causes_and_live_action_hooks(bad, label):
    with pytest.raises(action.ActionStoreCorruptError) as caught:
        scheduler._strict_json_loads(bad, label=label)
    assert str(caught.value) == label + " is not strict JSON"
    assert isinstance(caught.value.__cause__, (UnicodeDecodeError, ValueError, RecursionError))


def test_preservation_canonical_finite_unicode_and_exact_numeric_bytes():
    assert scheduler._canonical_json({"z": [True, 1, 1.0], "é": "雪"}) == '{"z":[true,1,1.0],"é":"雪"}'
    for value in (float("nan"), object()):
        with pytest.raises(ledger.OccurrenceConflictError) as caught:
            scheduler._canonical_json(value)
        assert str(caught.value) == "ledger values must be finite JSON data"
        assert isinstance(caught.value.__cause__, (TypeError, ValueError))


def test_preservation_real_state_bytes_cas_and_lost_ack_pins(tmp_path, monkeypatch):
    store = scheduler.SchedulerStore(tmp_path / "scheduler", namespace="owner", runtime_identity="copy")
    state = store.read()
    before = (store.root / "state.json").read_bytes()
    canonical = json.dumps(state, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    assert before == scheduler._canonical_json({"state": state, "integrity": hashlib.sha256(canonical.encode()).hexdigest()}).encode()
    successor = {**state, "generation": 1}
    original = os.fsync
    published = False
    rename = os.rename

    def renamed(*args, **kwargs):
        nonlocal published
        result = rename(*args, **kwargs)
        published = True
        return result

    def fsync(fd):
        if published and os.fstat(fd).st_mode & 0o170000 == 0o040000:
            assert store._state_content == (store.root / "state.json").read_bytes()
            raise FAILURE
        return original(fd)

    with monkeypatch.context() as fault:
        fault.setattr(os, "rename", renamed)
        fault.setattr(os, "fsync", fsync)
        with pytest.raises(Fault) as caught:
            store.compare_and_swap(state, successor)
        assert caught.value is FAILURE
    assert store.read() == successor
    assert store.compare_and_swap(state, successor) == successor
    assert store.read()["kind"] == "cron_scheduler_state_v1"
    lease = store.acquire_lease()
    try:
        assert store.acquire_lease() is lease
        assert store.read_for_acquisition(lease) == successor
        assert store.read_for_acquisition(lease) == successor
    finally:
        lease.release()


GUARDED = [("Store", name) for name in (
    "check_scheduler_lease_file", "install_scheduler_lease", "retire_scheduler_lease",
    "initialize_scheduler_held", "read_scheduler_held", "read_acquisition_scheduler_held",
    "compare_scheduler_held")] + [("Files", name) for name in (
    "initialize_scheduler", "read_scheduler", "read_scheduler_acquisition", "compare_scheduler_state",
    "write_scheduler_staging", "acquire_scheduler_lease")] + [("Lease", "consume_acquisition_refresh"),
    ("Occurrence", "acquire_scheduler_permanent_lock"), ("Occurrence", "release_scheduler_permanent_lock")]


@pytest.mark.parametrize("owner,name", GUARDED)
@pytest.mark.parametrize("token", ["object", "none", "callable", "lease-token", "lookalike"])
def test_new_api_invalid_permit_before_inputs_or_effects(owner, name, token, monkeypatch):
    events = []

    class Bomb:
        def __getattribute__(self, key):
            events.append(key)
            raise AssertionError("supplied input accessed before permit")

    bomb = Bomb()
    bad = {"object": object(), "none": None, "callable": lambda: None,
           "lease-token": scheduler._LEASE_TOKEN, "lookalike": SimpleNamespace(permit=True)}[token]
    cls = {"Store": scheduler.SchedulerStore, "Files": scheduler._PrivateFiles,
           "Lease": scheduler.SchedulerLease, "Occurrence": ledger.CronOccurrenceStore}[owner]
    method = getattr(cls, name)  # Expected original-production red: absent fixed API.
    instance = cls.__new__(cls)
    arguments = {
        "check_scheduler_lease_file": (bomb, 902, (1, 2)), "install_scheduler_lease": (bomb, 902),
        "retire_scheduler_lease": (), "initialize_scheduler_held": (bomb,), "read_scheduler_held": (bomb,),
        "read_acquisition_scheduler_held": (bomb, bomb), "compare_scheduler_held": (bomb, bomb, bomb),
        "initialize_scheduler": (bomb,), "read_scheduler": (bomb,), "read_scheduler_acquisition": (bomb, bomb),
        "compare_scheduler_state": (bomb, bomb, bomb), "write_scheduler_staging": (902, b"private"),
        "acquire_scheduler_lease": (bomb,), "consume_acquisition_refresh": (bomb,),
        "acquire_scheduler_permanent_lock": (902,), "release_scheduler_permanent_lock": (902,),
    }[name]
    kwargs = {"refresh": True} if name in {"read_scheduler_acquisition", "read_acquisition_scheduler_held"} else {}
    def denied(*_args, **_kwargs):
        events.append("syscall")
        raise AssertionError("syscall before permit")
    monkeypatch.setattr(os, "close", denied)
    monkeypatch.setattr(os, "write", denied)
    monkeypatch.setattr(fcntl, "flock", denied)
    with pytest.raises(TypeError, match="^scheduler owner collaboration requires internal permit$"):
        if owner == "Occurrence":
            method(bad, *arguments, **kwargs)
        else:
            method(instance, bad, *arguments, **kwargs)
    assert events == [] and vars(instance) == {}


def test_new_api_public_validator_is_rejection_only():
    validator = getattr(scheduler, "require_scheduler_owner_permit")
    for bad in (object(), None, lambda: None, scheduler._LEASE_TOKEN):
        with pytest.raises(TypeError, match="internal permit"):
            validator(bad)


def test_new_api_fixed_dispatch_bypasses_all_nominal_subclass_overrides(tmp_path, monkeypatch):
    calls = []
    def malicious(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("new dispatch leaked permit")

    class Store(scheduler.SchedulerStore):
        pass

    class Files(scheduler._PrivateFiles):
        pass

    for owner, name in GUARDED:
        cls = {"Store": scheduler.SchedulerStore, "Files": scheduler._PrivateFiles,
               "Lease": scheduler.SchedulerLease, "Occurrence": ledger.CronOccurrenceStore}[owner]
        assert hasattr(cls, name), name
        if owner == "Store":
            setattr(Store, name, malicious)
        elif owner == "Files":
            setattr(Files, name, malicious)
    assert hasattr(scheduler.SchedulerStore, "matches_issued_scheduler_lease")
    Store.matches_issued_scheduler_lease = malicious
    # Existing constructor uses fixed original Files; then exercise subclass Files.
    store = Store(tmp_path / "scheduler", namespace="owner", runtime_identity="copy")
    original = store._files
    files = Files.__new__(Files)
    files.__dict__.update(original.__dict__)
    store._files = files
    state = store.read()
    assert store.compare_and_swap(state, {**state, "generation": 1})["generation"] == 1
    lease = store.acquire_lease()
    class Lease(scheduler.SchedulerLease):
        consume_acquisition_refresh = malicious
        close_inherited_scheduler_descriptors = malicious
    # No copied/forged lease: change only nominal dispatch type of the actual issuance.
    lease.__class__ = Lease
    try:
        lease.reprove()
        assert store.read_for_acquisition(lease)["generation"] == 1
        assert calls == []
    finally:
        lease.release()
    assert calls == [] and store._lease is None


@pytest.mark.parametrize("name,label", [("decode_scheduler_state_json", "scheduler state"),
                                        ("decode_pinned_scheduler_state_json", "pinned scheduler state")])
def test_new_api_fixed_decoders_keep_live_owner_hooks(name, label, monkeypatch):
    decoder = getattr(action, name)
    calls = []
    def strict(data, *, label):
        calls.append((data, label))
        return "sentinel"
    monkeypatch.setattr(action, "_strict_json_loads", strict)
    assert decoder(b"raw") == "sentinel"
    assert scheduler._strict_json_loads(b"raw", label=label) == "sentinel"
    assert calls == [(b"raw", label), (b"raw", label)]


def test_new_api_fixed_parser_namespace_bridge_and_destination_seams(monkeypatch):
    parser = getattr(definitions, "parse_persisted_scheduler_commit")
    namespace = getattr(ledger.CronOccurrenceStore, "validate_scheduler_namespace")
    bridge = getattr(ledger, "scheduler_bridge_identity_conflicts")
    destination = getattr(occurrence, "scheduler_destination_policy_conflicts")
    calls = []
    monkeypatch.setattr(definitions, "_validate_scheduling_commit", lambda value: calls.append(value) or "commit")
    assert parser("input") == "commit" and calls == ["input"]
    monkeypatch.setattr(ledger.CronOccurrenceStore, "_validate_namespace", staticmethod(lambda value: calls.append(value) or "namespace"))
    assert namespace("input") == "namespace"
    identity = ledger.ScheduledOccurrenceIdentity("wrong", "job", "revision", 1)
    monkeypatch.setattr(ledger, "_occurrence_id", lambda value: calls.append(value) or "digest")
    class MappingBomb(dict):
        def __getitem__(self, key):
            raise AssertionError("eager bridge input")
    before = list(calls)
    assert bridge(identity, "owner", MappingBomb()) and calls == before
    assert not bridge(identity, "wrong", {"occurrence_id": "digest"})
    item = definition()
    record = SimpleNamespace(destination_policy={"sentinel": True})
    monkeypatch.setattr(occurrence, "_destination_policy", lambda value: calls.append(value) or {"sentinel": True})
    assert not destination(item, record) and calls[-1] is item


def test_preservation_destination_job_revision_short_circuits(monkeypatch):
    item = definition()
    snapshot = item.to_wire()
    monkeypatch.setattr(occurrence, "_destination_policy", lambda *_: pytest.fail("eager projection"))
    for job, revision in (("foreign", item.definition_revision), (item.job_id, "foreign")):
        record = SimpleNamespace(job_snapshot=deepcopy(snapshot), identity=SimpleNamespace(job_id=job, job_revision=revision))
        with pytest.raises(scheduler.SchedulerConflictError, match="immutable inputs conflict"):
            scheduler.immutable_occurrence_digest(record)


def test_new_api_dynamic_dispatch_negative_control_exposes_captured_permit(monkeypatch):
    """A runtime QA mutation, not a supported issuance or reflection route."""
    getattr(scheduler.SchedulerStore, "initialize_scheduler_held")
    captured = []
    class Store(scheduler.SchedulerStore):
        def initialize_scheduler_held(self, permit, held):
            captured.append(permit)

    class Files(scheduler._PrivateFiles):
        @contextmanager
        def _held_lock(self):
            yield Binding([])

    store = Store.__new__(Store)
    store._files = Files.__new__(Files)
    store._lease = None
    # Mutate only a fixed Files operation in this isolated frozen QA test process.
    def dynamic(files, permit, target):
        scheduler.require_scheduler_owner_permit(permit)
        with files._held_lock() as held:
            target.initialize_scheduler_held(permit, held)
    monkeypatch.setattr(scheduler._PrivateFiles, "initialize_scheduler", dynamic)
    # The existing supported Store constructor call edge supplies the permit.
    monkeypatch.setattr(scheduler, "_PrivateFiles", Files)
    monkeypatch.setattr(Files, "__init__", lambda self, root: setattr(self, "root", root))
    from pathlib import Path
    mutated = Store(Path("/qa-only-unused"), namespace="owner", runtime_identity="copy")
    assert len(captured) == 1
    fabricated = Binding([])
    monkeypatch.setattr(mutated, "_read", lambda held: "fabricated" if held is fabricated else None)
    assert scheduler.SchedulerStore.read_scheduler_held(mutated, captured[0], fabricated) == "fabricated"


@pytest.mark.parametrize("phase", ["install", "retire", "unlock", "check"])
def test_new_api_injected_guard_fault_is_separate_from_original_parity(phase, tmp_path, monkeypatch):
    import inspect

    validator = getattr(scheduler, "require_scheduler_owner_permit")
    store = scheduler.SchedulerStore(tmp_path / "scheduler", namespace="owner", runtime_identity="copy")
    method = {"install": "install_scheduler_lease", "retire": "retire_scheduler_lease",
              "unlock": "release_scheduler_permanent_lock", "check": "check_scheduler_lease_file"}[phase]
    def guard(permit):
        frame = inspect.currentframe()
        if frame.f_back.f_code.co_name == method:
            raise FAILURE
        return validator(permit)
    if phase == "install":
        with monkeypatch.context() as fault:
            fault.setattr(scheduler, "require_scheduler_owner_permit", guard)
            with pytest.raises(Fault) as caught:
                store.acquire_lease()
        assert caught.value is FAILURE and store._lease is None
        retry = store.acquire_lease()
        retry.release()
        return
    lease = store.acquire_lease()
    try:
        with monkeypatch.context() as fault:
            fault.setattr(scheduler, "require_scheduler_owner_permit", guard)
            with pytest.raises(Fault) as caught:
                lease.reprove() if phase == "check" else lease.release()
        assert caught.value is FAILURE
        if phase == "retire":
            assert not lease._active and store._lease is lease and lease in scheduler._LEASES
        elif phase == "unlock":
            assert not lease._active and store._lease is None and lease not in scheduler._LEASES
        else:
            assert lease._active and store._lease is lease
    finally:
        if lease._active:
            lease.release()
        elif phase == "retire":
            store._lease = None
            scheduler._LEASES.discard(lease)
