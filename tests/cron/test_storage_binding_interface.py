"""Cron owners reprove actual held storage bindings through typed operations."""

from __future__ import annotations

import inspect
import stat
from contextlib import ExitStack
from pathlib import Path

import pytest

from nanobot.cron.action_store import ActionStoreCorruptError, CronActionStore
from nanobot.cron.callback_store import CronCallbackConflictError, CronCallbackStore
from nanobot.cron.completion import CronCompletionConflictError, CronCompletionProjector
from nanobot.cron.delivery_reconciliation import ReconciliationUnsafeError
from nanobot.cron.occurrence_store import CronOccurrenceStore

_CASES = (
    ("action", "require_held_store_binding", "_assert_binding", "held", "_HeldStore"),
    ("callback", "require_callback_root_binding", "_assert_root_binding", "binding", "_RootBinding"),
    ("projection-root", "require_projection_root_binding", "_assert_path_binding", "binding", "_RootBinding"),
    ("projection-held", "require_held_projection_binding", "_assert_binding", "held", "_HeldProjection"),
    ("reconciliation", "require_reconciliation_binding", "_assert_bound", None, None),
)


@pytest.fixture(params=_CASES, ids=[case[0] for case in _CASES])
def _binding_case(tmp_path: Path, request):
    kind, public, private, argument, annotation = request.param
    root = tmp_path / "storage"
    with ExitStack() as contexts:
        if kind == "action":
            store = CronActionStore(root, namespace="cron:binding-interface")
            binding = contexts.enter_context(store._held_lock())
            error = ActionStoreCorruptError
        elif kind == "callback":
            def observer(_value):
                raise AssertionError("binding validation must not invoke a callback")

            store = CronCallbackStore(root, observer)
            binding = contexts.enter_context(store._root_binding())
            error = CronCallbackConflictError
        elif kind.startswith("projection"):
            occurrences = CronOccurrenceStore(tmp_path / "occurrences", namespace="cron:binding-interface")
            store = CronCompletionProjector(occurrences, root)
            binding = contexts.enter_context(
                store._root_binding() if kind == "projection-root" else store._held_lock(),
            )
            error = CronCompletionConflictError
        else:
            from nanobot.cron import delivery_reconciliation as reconciliation

            store = contexts.enter_context(reconciliation._ArtifactStore(root, create=True))
            binding = None
            error = ReconciliationUnsafeError
        yield store, binding, public, private, argument, annotation, root, error


def _storage_snapshot(root: Path):
    snapshot = []
    for path in [root, *sorted(root.rglob("*"))]:
        info = path.lstat()
        snapshot.append((
            str(path.relative_to(root)), info.st_dev, info.st_ino,
            stat.S_IMODE(info.st_mode), path.read_bytes() if path.is_file() else None,
        ))
    return snapshot


def test_storage_binding_operation_has_explicit_typed_signature(_binding_case) -> None:
    store, binding, public, private, argument, annotation, root, error = _binding_case
    operation = getattr(store, public)
    signature = inspect.signature(operation)
    assert tuple(signature.parameters) == (() if argument is None else (argument,))
    if argument is not None:
        parameter = signature.parameters[argument]
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameter.default is inspect.Parameter.empty
        assert parameter.annotation == annotation
    assert signature.return_annotation == "None"


def test_storage_binding_operation_dynamically_runs_real_guard_without_writes(
    _binding_case, monkeypatch,
) -> None:
    store, binding, public, private, argument, annotation, root, error = _binding_case
    operation = getattr(store, public)
    original = getattr(store, private)
    before = _storage_snapshot(root)
    calls = []

    def first(*actual):
        assert len(actual) == (0 if argument is None else 1)
        if actual:
            assert actual[0] is binding
        calls.append("first")
        return original(*actual)

    def second(*actual):
        assert len(actual) == (0 if argument is None else 1)
        if actual:
            assert actual[0] is binding
        calls.append("second")
        return original(*actual)

    with monkeypatch.context() as patch:
        patch.setattr(store, private, first)
        first_result = operation() if argument is None else operation(binding)
        patch.setattr(store, private, second)
        second_result = operation() if argument is None else operation(**{argument: binding})
    assert first_result is None and second_result is None
    assert calls == ["first", "second"]
    assert _storage_snapshot(root) == before


@pytest.mark.parametrize("fault_kind", [OSError, ValueError, KeyboardInterrupt])
def test_storage_binding_operation_preserves_exact_guard_faults(
    _binding_case, monkeypatch, fault_kind,
) -> None:
    store, binding, public, private, argument, annotation, root, error = _binding_case
    operation = getattr(store, public)
    fault = fault_kind("binding guard probe")
    calls = []

    def fail(*actual):
        calls.append(actual)
        raise fault

    before = _storage_snapshot(root)
    with monkeypatch.context() as patch:
        patch.setattr(store, private, fail)
        with pytest.raises(fault_kind) as raised:
            operation() if argument is None else operation(binding)
    assert raised.value is fault and len(calls) == 1
    assert len(calls[0]) == (0 if argument is None else 1)
    if calls[0]:
        assert calls[0][0] is binding
    assert _storage_snapshot(root) == before


def test_storage_binding_operation_rejects_actual_path_substitution(_binding_case) -> None:
    store, binding, public, private, argument, annotation, root, error = _binding_case
    operation = getattr(store, public)
    before = _storage_snapshot(root)
    retained = root.with_name("retained-storage")
    root.rename(retained)
    root.mkdir(mode=0o700)
    try:
        with pytest.raises(error):
            operation() if argument is None else operation(binding)
    finally:
        root.rmdir()
        retained.rename(root)
    assert (operation() if argument is None else operation(binding)) is None
    assert _storage_snapshot(root) == before
