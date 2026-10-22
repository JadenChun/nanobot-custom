"""A prepared mode reset keeps the original late hook and real parent scope."""

from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.session.execution_owner import (
    ExecutionBusyError,
    acquire_execution,
    execution_lock_guard,
)
from nanobot.session.manager import JsonlSessionStore, SessionWriteConflict
from nanobot.session.runtime_authority import OfflineSessionOperator
from nanobot.session.source import SourceValidationError
from nanobot.session.source_store import SourceResetPlan, SourceSessionStore
from tests.session.test_execution_owner import seed

TIMESTAMP = "2026-10-13T00:00:00+00:00"


@pytest.fixture
def seeded(tmp_path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    return backend, seed(backend)


def command():
    operation = getattr(SourceSessionStore, "install_prepared_mode_reset", None)
    assert callable(operation), "source-owned prepared mode reset operation is missing"
    return operation


def namespace_bytes(backend):
    return {path.relative_to(backend.sessions_dir): path.read_bytes()
            for path in backend.sessions_dir.rglob("*") if path.is_file()}


def prepared(backend, snapshot, events):
    return backend.source.plan_source_reset(snapshot, events, updated_at=TIMESTAMP)


def test_prepared_reset_is_a_concrete_source_domain_operation():
    command()


def test_original_source_hook_and_expected_operand_order_are_retained(seeded, monkeypatch):
    operation = command()
    backend, initial = seeded
    events = []
    plan = prepared(backend, initial, events)
    expected = plan.expected
    before = namespace_bytes(backend)
    trace = []

    class SelectedSource(SourceSessionStore):
        def install_prepared_mode_reset(self, *args, **kwargs):
            pytest.fail("a selected-source override redirected the concrete operation")

        def __getattribute__(self, name):
            if name == "_reset_source_session_unlocked":
                trace.append("hook-lookup")
            return super().__getattribute__(name)

    selected = SelectedSource(backend)

    def source(actual_backend):
        assert actual_backend is backend
        trace.append("source")
        return selected

    def expected_state(actual_plan):
        assert actual_plan is plan
        trace.append("expected")
        return expected

    def hook(actual_source, key, actual_expected, actual_events, *, owner, plan):
        assert actual_source is selected and key == initial.state.session_key
        assert actual_expected is expected and actual_events is events
        assert owner is None and plan.expected is expected
        assert backend.session_files_locked
        trace.append("invoke")
        return initial

    with monkeypatch.context() as patch:
        patch.setattr(JsonlSessionStore, "source", property(source))
        patch.setattr(SourceResetPlan, "expected", property(expected_state), raising=False)
        patch.setattr(SourceSessionStore, "_reset_source_session_unlocked", hook)
        with backend.locked_session_files():
            result = operation(backend, initial.state.session_key, events, owner=None, plan=plan)
    assert result is initial
    assert trace == ["source", "hook-lookup", "expected", "expected", "invoke"]
    assert namespace_bytes(backend) == before


@pytest.mark.parametrize("phase", ["source", "hook-lookup", "expected", "invoke"])
def test_first_boundary_fault_keeps_identity_and_stops_later_operands(
    seeded, monkeypatch, phase,
):
    operation = command()
    backend, initial = seeded
    events = []
    plan = prepared(backend, initial, events)
    expected = plan.expected
    before = namespace_bytes(backend)
    fault = RuntimeError(phase)
    trace = []

    def step(name):
        trace.append(name)
        if name == phase:
            raise fault

    class SelectedSource(SourceSessionStore):
        def __getattribute__(self, name):
            if name == "_reset_source_session_unlocked":
                step("hook-lookup")
            return super().__getattribute__(name)

    selected = SelectedSource(backend)

    def source(_backend):
        step("source")
        return selected

    def expected_state(_plan):
        step("expected")
        return expected

    def hook(_source, _key, _expected, _events, **_keywords):
        step("invoke")
        pytest.fail("faulting reset hook returned")

    with monkeypatch.context() as patch:
        patch.setattr(JsonlSessionStore, "source", property(source))
        patch.setattr(SourceResetPlan, "expected", property(expected_state), raising=False)
        patch.setattr(SourceSessionStore, "_reset_source_session_unlocked", hook)
        with backend.locked_session_files(), pytest.raises(RuntimeError) as caught:
            operation(backend, initial.state.session_key, events, owner=None, plan=plan)
    assert caught.value is fault
    sequence = ["source", "hook-lookup", "expected", "invoke"]
    assert trace == sequence[:sequence.index(phase) + 1]
    assert namespace_bytes(backend) == before


@pytest.mark.parametrize("field", ["bytes", "state", "timestamp", "expected"])
def test_invalid_or_stale_prepared_data_still_refuses_without_writing(seeded, field):
    operation = command()
    backend, initial = seeded
    events = []
    plan = prepared(backend, initial, events)
    if field == "bytes":
        plan = replace(plan, target_bytes=b"invalid source\n")
    elif field == "state":
        plan = replace(plan, target_state=replace(plan.target_state, revision=1000))
    elif field == "timestamp":
        plan = replace(plan, updated_at="invalid timestamp")
    else:
        plan = replace(plan, expected=replace(plan.expected, revision=1000))
    before = namespace_bytes(backend)
    error = SessionWriteConflict if field == "expected" else SourceValidationError
    with backend.locked_session_files(), pytest.raises(error):
        operation(backend, initial.state.session_key, events, owner=None, plan=plan)
    assert namespace_bytes(backend) == before


@pytest.mark.parametrize("phase", ["before", "after"])
def test_original_private_hook_fault_preserves_exact_pre_and_post_write_effects(
    seeded, monkeypatch, phase,
):
    operation = command()
    backend, initial = seeded
    events = []
    plan = prepared(backend, initial, events)
    before = namespace_bytes(backend)
    original = SourceSessionStore._reset_source_session_unlocked
    fault = OSError("original reset hook failed")
    trace = []

    def hook(source, key, expected, replacements, **keywords):
        trace.append("hook")
        if phase == "before":
            raise fault
        original(source, key, expected, replacements, **keywords)
        trace.append("committed")
        raise fault

    with monkeypatch.context() as patch:
        patch.setattr(SourceSessionStore, "_reset_source_session_unlocked", hook)
        with backend.locked_session_files(), pytest.raises(OSError) as caught:
            operation(backend, initial.state.session_key, events, owner=None, plan=plan)
    assert caught.value is fault
    assert trace == (["hook"] if phase == "before" else ["hook", "committed"])
    if phase == "before":
        assert namespace_bytes(backend) == before
    else:
        assert backend.get_session_path(initial.state.session_key).read_bytes() == plan.target_bytes
        assert backend.source.load_source_snapshot(initial.state.session_key).state == plan.target_state


@pytest.mark.parametrize("owned", [False, True])
async def test_real_mode_reset_keeps_scope_plan_cas_and_owner_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owned: bool,
):
    operation = command()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    lease = await OfflineSessionOperator(
        workspace, copied_sessions_root=tmp_path / "sessions"
    ).acquire()
    try:
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        key = "cli:prepared-mode-reset"
        manager.admit_persistent_write(key)
        backend = manager._jsonl_store
        resolver = manager._mode_resolver
        before = backend.source.load_source_snapshot(key)
        previous_mode = manager.resolve_session_mode(key)
        events = []
        trace = []
        plans = []
        original_hook = SourceSessionStore._reset_source_session_unlocked

        class SelectedSource(SourceSessionStore):
            def install_prepared_mode_reset(self, *args, **kwargs):
                pytest.fail("instance override redirected the prepared reset")

        def source(actual_backend):
            return SelectedSource(actual_backend)

        def install(actual_backend, actual_key, replacements, *, plan, owner):
            assert actual_backend is backend and actual_key == key and replacements is events
            assert backend.session_files_locked
            with pytest.raises(ExecutionBusyError), execution_lock_guard(backend, key):
                pytest.fail("prepared reset lost its real execution lock")
            if owner is not None:
                assert owner._active and owner._lock.is_locked
            plans.append(plan)
            trace.append("domain")
            return operation(actual_backend, actual_key, replacements, plan=plan, owner=owner)

        def hook(actual_source, actual_key, expected, replacements, *, plan, owner):
            assert isinstance(actual_source, SelectedSource)
            assert actual_key == key and expected is plan.expected and replacements is events
            assert backend.session_files_locked
            trace.append("hook")
            result = original_hook(
                actual_source, actual_key, expected, replacements, plan=plan, owner=owner,
            )
            assert backend.get_session_path(key).read_bytes() == plan.target_bytes
            if owner is not None:
                assert not owner._active and owner._lock.is_locked
            trace.append("installed")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(JsonlSessionStore, "source", property(source))
            patch.setattr(SourceSessionStore, "install_prepared_mode_reset", staticmethod(install))
            patch.setattr(SourceSessionStore, "_reset_source_session_unlocked", hook)
            if owned:
                async with acquire_execution(backend, key, run_id="prepared-reset") as owner:
                    await owner.quiesce()
                    result = resolver.reset(key, events, owner=owner)
                    assert not owner._active and owner._lock.is_locked
            else:
                result = resolver.reset(key, events, owner=None)
        assert trace == ["domain", "hook", "installed"]
        assert len(plans) == 1 and result.mode == "canonical" and result.state == "active"
        assert result.record_revision == previous_mode.record_revision + 1
        snapshot = backend.source.load_source_snapshot(key)
        assert snapshot.state == plans[0].target_state
        assert snapshot.state.generation == before.state.generation + 1
        assert snapshot.state.incarnation == before.state.incarnation
        assert snapshot.events == [] and "execution_lease" not in snapshot.header["metadata"]
        assert not backend.session_files_locked and not hasattr(resolver._guard_local, "scope")
    finally:
        await lease.close()
