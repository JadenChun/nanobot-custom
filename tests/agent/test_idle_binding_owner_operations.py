"""Owner relocation probes for dormant binding identity and opaque material."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nanobot.agent import idle_runtime as idle
from tests.agent.test_idle_runtime import _registry

_READS = [
    "binding:_authority", "registry:_authority", "binding:_pid", "registry:_pid",
    "registry:_current", "binding:_session_key", "binding:_source", "material:source",
    "binding:_session_key", "binding:_incarnation", "binding:_revision", "binding:_generation",
    "binding:_store_identity", "binding:_root_identity", "binding:_checkpoint_id",
    "binding:_checkpoint_fingerprint", "material:identity", "binding:_selected_preset",
    "material:selected_preset", "binding:_runtime_epoch", "material:runtime_epoch",
    "binding:_config_epoch", "material:config_epoch", "binding:_source_fingerprint",
    "material:source_fingerprint", "binding:_runtime_fingerprint", "material:runtime_fingerprint",
    "binding:_config_fingerprint", "material:config_fingerprint", "binding:_fingerprint",
    "material:fingerprint",
]


def _case(tmp_path):
    registry = _registry(tmp_path, preset_runtimes={})
    binding = registry.issue("cli:i3b")
    material = registry._build_material(registry._capture_source("cli:i3b"))
    return registry, binding, material


def _trace_reads(monkeypatch, registry, binding, material, trace, position=None, fault=None):
    for target, value, prefix in [
        (idle.IdleRuntimeBindingRegistry, registry, "registry"),
        (idle.IdleRuntimeBinding, binding, "binding"),
        (type(material), material, "material"),
    ]:
        original = target.__getattribute__

        def read(self, name, *, original=original, value=value, prefix=prefix):
            entry = f"{prefix}:{name}"
            if self is value and entry in _READS:
                trace.append(entry)
                if len(trace) == position:
                    raise fault
            return original(self, name)

        monkeypatch.setattr(target, "__getattribute__", read)


def test_existing_matcher_keeps_complete_live_operand_order(tmp_path, monkeypatch):
    registry, binding, material = _case(tmp_path)
    trace = []
    _trace_reads(monkeypatch, registry, binding, material, trace)
    assert registry._binding_matches(binding, material) is True
    assert trace == _READS


@pytest.mark.parametrize("position", range(1, len(_READS) + 1))
@pytest.mark.parametrize("fault", [RuntimeError("ordinary"), KeyboardInterrupt("system")])
def test_existing_matcher_preserves_each_operand_fault(tmp_path, monkeypatch, position, fault):
    registry, binding, material = _case(tmp_path)
    trace = []
    _trace_reads(monkeypatch, registry, binding, material, trace, position, fault)
    with pytest.raises(type(fault)) as caught:
        registry._binding_matches(binding, material)
    assert caught.value is fault
    assert trace == _READS[:position]


@pytest.mark.parametrize("field, replacement, end", [
    ("_authority", object(), 2), ("_pid", -1, 4),
    ("_source", object(), 8), ("_incarnation", "changed", 17),
    ("_selected_preset", "changed", 19), ("_runtime_epoch", -1, 21),
    ("_config_epoch", -1, 23), ("_source_fingerprint", object(), 25),
    ("_runtime_fingerprint", object(), 27), ("_config_fingerprint", object(), 29),
    ("_fingerprint", object(), 31),
])
def test_existing_matcher_retains_short_circuit_boundaries(
    tmp_path, monkeypatch, field, replacement, end,
):
    registry, binding, material = _case(tmp_path)
    object.__setattr__(binding, field, replacement)
    trace = []
    _trace_reads(monkeypatch, registry, binding, material, trace)
    assert registry._binding_matches(binding, material) is False
    assert trace == _READS[:end]


def test_existing_matcher_get_lookup_precedes_binding_key_read(tmp_path):
    registry, binding, material = _case(tmp_path)
    trace = []

    class Current(dict):
        @property
        def get(self):
            trace.append("get_lookup")
            object.__setattr__(binding, "_session_key", "changed")

            def lookup(key):
                trace.append(("get_call", key))
                return None

            return lookup

    registry._current = Current()
    assert registry._binding_matches(binding, material) is False
    assert trace == ["get_lookup", ("get_call", "changed")]


@pytest.mark.parametrize("point", ["authority", "pid"])
@pytest.mark.parametrize("fault", [RuntimeError("ordinary"), KeyboardInterrupt("system")])
def test_existing_identity_check_keeps_owner_fault_order(tmp_path, monkeypatch, point, fault):
    registry, binding, _ = _case(tmp_path)
    trace = []
    original = idle.IdleRuntimeBindingRegistry.__getattribute__

    def read(self, name):
        if self is registry and name in {"_authority", "_pid"}:
            trace.append(name)
            if name == "_" + point:
                raise fault
        return original(self, name)

    monkeypatch.setattr(idle.IdleRuntimeBindingRegistry, "__getattribute__", read)
    with pytest.raises(type(fault)) as caught:
        registry._validate_binding(binding)
    assert caught.value is fault
    assert trace == (["_authority"] if point == "authority" else ["_authority", "_pid"])


def test_existing_factory_lookup_and_cached_callable_precede_owner_arguments(tmp_path, monkeypatch):
    registry, _, material = _case(tmp_path)
    registry._current.clear()
    monkeypatch.setattr(registry, "_capture_source", lambda *args, **kwargs: object())
    monkeypatch.setattr(registry, "_build_material", lambda _: material)
    original = idle.IdleRuntimeBindingRegistry.__getattribute__
    trace = []
    active = False
    sentinel = object()
    authority = object()
    initial_pid = registry._pid

    def forbidden(*args, **kwargs):
        raise AssertionError("private issuer was looked up twice")

    class Issuer:
        def __get__(self, instance, cls):
            nonlocal active
            trace.append("private_lookup")
            registry._authority = authority
            active = True

            def issue(**kwargs):
                trace.append(("issue", tuple(kwargs), kwargs["authority"], kwargs["registry_pid"],
                              kwargs["material"]))
                return sentinel

            return issue

    def read(self, name):
        if self is registry and active and name in {"_authority", "_pid"}:
            trace.append(name)
            if name == "_authority":
                # The outer monkeypatch restores this slot. Do not let its own
                # save-old-value lookup invoke the instrumented descriptor.
                setattr(idle.IdleRuntimeBinding, "_issue", forbidden)
        return original(self, name)

    monkeypatch.setattr(idle.IdleRuntimeBindingRegistry, "__getattribute__", read)
    monkeypatch.setattr(idle.IdleRuntimeBinding, "_issue", Issuer())
    assert registry._issue_locked("cli:i3b") is sentinel
    assert trace[:3] == ["private_lookup", "_authority", "_pid"]
    assert len(trace) == 4
    assert trace[3][0:3] == ("issue", ("authority", "registry_pid", "material"), authority)
    assert trace[3][3] == initial_pid
    assert trace[3][4] is material
    assert registry._current["cli:i3b"] is sentinel


@pytest.mark.parametrize("point", [0, 1, 2, 3])
@pytest.mark.parametrize("fault", [RuntimeError("ordinary"), KeyboardInterrupt("system")])
def test_opaque_owner_composes_exact_private_operands_and_faults(monkeypatch, point, fault):
    fingerprints = [idle.OpaqueFingerprint(bytes([index]) * 32) for index in (1, 2, 3)]
    trace = []
    original = idle.OpaqueFingerprint.__getattribute__
    sentinel = object()

    def read(self, name):
        if name == "_digest":
            index = next(index for index, value in enumerate(fingerprints) if value is self)
            trace.append(index)
            if point == index:
                raise fault
        return original(self, name)

    def digest(value):
        trace.append(value)
        if point == 3:
            raise fault
        return sentinel

    monkeypatch.setattr(idle.OpaqueFingerprint, "__getattribute__", read)
    monkeypatch.setattr(idle, "_opaque_digest", digest)
    with pytest.raises(type(fault)) as caught:
        idle.OpaqueFingerprint.combine_binding(*fingerprints)
    assert caught.value is fault
    assert trace[:min(point + 1, 3)] == list(range(min(point + 1, 3)))
    if point == 3:
        assert trace[-1] == ("binding", bytes([1]) * 32, bytes([2]) * 32, bytes([3]) * 32)


def test_opaque_owner_returns_only_the_existing_digest_result(monkeypatch):
    fingerprints = [idle.OpaqueFingerprint(bytes([index]) * 32) for index in (1, 2, 3)]
    sentinel = object()
    captured = []

    def digest(value):
        captured.append(value)
        return sentinel

    monkeypatch.setattr(idle, "_opaque_digest", digest)
    assert idle.OpaqueFingerprint.combine_binding(*fingerprints) is sentinel
    assert captured == [("binding", bytes([1]) * 32, bytes([2]) * 32, bytes([3]) * 32)]


def test_binding_owner_predicate_reads_lazy_suppliers_and_returns_bool(tmp_path, monkeypatch):
    registry, binding, material = _case(tmp_path)
    trace = []
    _trace_reads(monkeypatch, registry, binding, material, trace)
    assert idle.IdleRuntimeBinding.matches_registry_material(
        binding, authority=lambda: registry._authority, registry_pid=lambda: registry._pid,
        current_lookup=lambda: registry._current.get, material=material,
    ) is True
    assert trace == _READS


@pytest.mark.parametrize("authority_matches, pid_matches", [(False, False), (False, True), (True, False), (True, True)])
def test_identity_owner_retains_exact_refusals_and_short_circuit(
    tmp_path, authority_matches, pid_matches,
):
    registry, binding, _ = _case(tmp_path)
    trace = []

    def authority():
        trace.append("authority")
        return registry._authority if authority_matches else object()

    def pid():
        trace.append("pid")
        return registry._pid if pid_matches else -1

    if not authority_matches or not pid_matches:
        error = idle.IdleRuntimeBindingError if not authority_matches else idle.IdleRuntimeStaleError
        with pytest.raises(error) as caught:
            idle.IdleRuntimeBinding.validate_registry_identity(binding, authority=authority, registry_pid=pid)
        assert str(caught.value) == ("foreign or reconstructed idle runtime binding" if not authority_matches
                                     else "idle runtime binding is stale")
    else:
        assert idle.IdleRuntimeBinding.validate_registry_identity(
            binding, authority=authority, registry_pid=pid,
        ) is None
    assert trace == (["authority", "pid"] if authority_matches else ["authority"])


@pytest.mark.parametrize("point", ["lookup", "arguments", "issue"])
@pytest.mark.parametrize("fault", [RuntimeError("ordinary"), KeyboardInterrupt("system")])
def test_owner_factory_preserves_subclass_private_seam_and_fault_order(point, fault):
    trace = []

    class Issuer:
        def __get__(self, instance, cls):
            trace.append("lookup")
            if point == "lookup":
                raise fault

            def issue(**kwargs):
                trace.append("issue")
                assert tuple(kwargs) == ("authority", "registry_pid", "material")
                raise fault

            return issue

    class Child(idle.IdleRuntimeBinding):
        _issue = Issuer()

    def arguments():
        trace.append("arguments")
        if point == "arguments":
            raise fault
        return object(), 7, object()

    with pytest.raises(type(fault)) as caught:
        Child.from_registry_material(arguments)
    assert caught.value is fault
    order = ["lookup", "arguments", "issue"]
    assert trace == order[:order.index(point) + 1]


def test_owner_factory_does_not_register_reconstructed_binding(tmp_path):
    registry, current, material = _case(tmp_path)
    issued_count = registry.issued_count
    reconstructed = idle.IdleRuntimeBinding.from_registry_material(
        lambda: (registry._authority, registry._pid, material),
    )
    assert reconstructed is not current
    assert registry.revalidate(reconstructed, "cli:i3b") is False
    assert registry.issue("cli:i3b") is current
    assert registry.issued_count == issued_count


def test_owner_operations_do_not_load_dispatch_register_or_write(tmp_path, monkeypatch):
    registry, binding, material = _case(tmp_path)
    before = dict(registry._current), registry._issued_count, registry._invalidation_count

    def forbidden(*args, **kwargs):
        raise AssertionError("pure binding operation attempted an effect")

    monkeypatch.setattr(registry._source, "load_source_snapshot", forbidden)
    assert idle.IdleRuntimeBinding.matches_registry_material(
        binding, authority=lambda: registry._authority, registry_pid=lambda: registry._pid,
        current_lookup=lambda: registry._current.get, material=material,
    ) is True
    assert idle.IdleRuntimeBinding.validate_registry_identity(
        binding, authority=lambda: registry._authority, registry_pid=lambda: registry._pid,
    ) is None
    created = idle.IdleRuntimeBinding.from_registry_material(
        lambda: (registry._authority, registry._pid, material),
    )
    assert created is not binding and created.session_key == binding.session_key
    assert before == (dict(registry._current), registry._issued_count, registry._invalidation_count)


@pytest.mark.parametrize("kind", ["hostile", "subclass", "none", "object"])
def test_existing_exact_type_guard_refuses_before_identity_reads(tmp_path, monkeypatch, kind):
    registry, _, _ = _case(tmp_path)
    before = dict(registry._current), registry._issued_count, registry._invalidation_count
    trace = []

    class Hostile:
        def __getattribute__(self, name):
            trace.append(("invalid_attribute", name))
            raise AssertionError("invalid binding attribute accessed")

        def __repr__(self):
            trace.append("invalid_repr")
            raise AssertionError("invalid binding repr accessed")

    class Child(idle.IdleRuntimeBinding):
        __getattribute__ = Hostile.__getattribute__
        __repr__ = Hostile.__repr__

    invalid = {"hostile": Hostile(), "subclass": object.__new__(Child),
               "none": None, "object": object()}[kind]
    original = idle.IdleRuntimeBindingRegistry.__getattribute__

    def read(self, name):
        if self is registry and name in {"_authority", "_pid"}:
            trace.append(("registry_attribute", name))
            raise AssertionError("registry identity read before exact-type refusal")
        return original(self, name)

    def process_pid():
        trace.append("process_pid")
        raise AssertionError("process PID read before exact-type refusal")

    with monkeypatch.context() as patch:
        patch.setattr(idle.IdleRuntimeBindingRegistry, "__getattribute__", read)
        patch.setattr(idle, "os", SimpleNamespace(getpid=process_pid))
        with pytest.raises(idle.IdleRuntimeBindingError) as caught:
            registry._validate_binding(invalid)
        assert type(caught.value) is idle.IdleRuntimeBindingError
        assert str(caught.value) == "invalid idle runtime binding"
        assert trace == []
    assert before == (dict(registry._current), registry._issued_count, registry._invalidation_count)


@pytest.mark.parametrize("kind", ["hostile", "subclass", "none", "object"])
def test_existing_public_revalidate_refuses_invalid_inputs_without_effects(
    tmp_path, monkeypatch, kind,
):
    registry, _, _ = _case(tmp_path)
    before = dict(registry._current), registry._issued_count, registry._invalidation_count
    gate = registry._gate
    trace = []

    class Hostile:
        def __getattribute__(self, name):
            trace.append(("invalid_attribute", name))
            raise AssertionError("invalid binding attribute accessed")

        def __repr__(self):
            trace.append("invalid_repr")
            raise AssertionError("invalid binding repr accessed")

    class Child(idle.IdleRuntimeBinding):
        __getattribute__ = Hostile.__getattribute__
        __repr__ = Hostile.__repr__

    invalid = {"hostile": Hostile(), "subclass": object.__new__(Child),
               "none": None, "object": object()}[kind]
    original_read = idle.IdleRuntimeBindingRegistry.__getattribute__
    original_context = gate.check_context
    original_pid = idle.os.getpid

    def read(self, name):
        if self is registry and name == "_pid":
            trace.append("registry_pid")
        if self is registry and name in {
            "_authority", "_current", "_issued_count", "_invalidation_count",
        }:
            trace.append(("registry_attribute", name))
            raise AssertionError("invalid binding reached registry authority or accounting")
        return original_read(self, name)

    def check_context():
        trace.append("context")
        return original_context()

    def process_pid():
        trace.append("process_pid")
        return original_pid()

    def forbidden(*args, **kwargs):
        trace.append("source_load")
        raise AssertionError("invalid binding reached source loading")

    with monkeypatch.context() as patch:
        patch.setattr(idle.IdleRuntimeBindingRegistry, "__getattribute__", read)
        patch.setattr(gate, "check_context", check_context)
        patch.setattr(idle, "os", SimpleNamespace(getpid=process_pid))
        patch.setattr(registry._source, "load_source_snapshot", forbidden)
        with pytest.raises(idle.IdleRuntimeBindingError) as caught:
            registry.revalidate(invalid, "cli:i3b")
        assert type(caught.value) is idle.IdleRuntimeBindingError
        assert str(caught.value) == "invalid idle runtime binding"
        assert trace == ["context", "context", "process_pid", "registry_pid"]
    assert gate.sync_active is False
    assert before == (dict(registry._current), registry._issued_count, registry._invalidation_count)
