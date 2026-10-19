"""Additive owner-operation regressions; no live credentials or network."""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from nanobot.providers import oauth_credentials as credentials


@pytest.fixture(autouse=True)
def _require_owner_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    assert callable(getattr(credentials._LoginGeneration, "require_current_intent", None)), (
        "login-intent owner operation is missing"
    )

    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("network is forbidden in login-intent boundary tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


class _FaultError(RuntimeError):
    pass


class _ObservedGeneration(credentials._LoginGeneration):
    def __init__(
        self, owner: object, value: str, trace: list[str],
        *, stage: str | None = None, fault: _FaultError | None = None, missing_value: bool = False,
    ) -> None:
        super().__init__(owner, value)
        self.trace = trace
        self.stage = stage
        self.fault = fault
        self.missing_value = missing_value

    def __getattribute__(self, name: str) -> object:
        if name == "_value_for":
            self.trace.append("lookup")
            if self.stage == "lookup":
                assert self.fault is not None
                raise self.fault
        return super().__getattribute__(name)

    def _value_for(self, owner: object) -> str | None:
        self.trace.append("value")
        if self.stage == "value":
            assert self.fault is not None
            raise self.fault
        if self.missing_value:
            return None
        return super()._value_for(owner)


class _ObservedState:
    def __init__(self, intent: str | None, trace: list[str], fault: _FaultError | None = None) -> None:
        self.intent = intent
        self.trace = trace
        self.fault = fault

    @property
    def intent_generation(self) -> str | None:
        self.trace.append("intent")
        if self.fault is not None:
            raise self.fault
        return self.intent


class _ObservedCoordinator(credentials.OAuthCredentialCoordinator):
    def __getattribute__(self, name: str) -> object:
        if name == "_generation_owner":
            self.trace.append("owner")
            if self.fault is not None:
                raise self.fault
        return super().__getattribute__(name)


def _validation(
    route: str, owner: object, trace: list[str], owner_fault: _FaultError | None = None,
) -> Callable[[object, object], str]:
    if route == "direct":
        def observe_owner() -> object:
            trace.append("owner")
            if owner_fault is not None:
                raise owner_fault
            return owner

        def validate(generation: object, state: object) -> str:
            return credentials._LoginGeneration.require_current_intent(
                generation, observe_owner, state,
            )

        return validate

    # Pure coordinator validation: this object is never used for storage/admission.
    coordinator = object.__new__(_ObservedCoordinator)
    coordinator.trace = trace
    coordinator.fault = owner_fault
    coordinator._generation_owner = owner
    return coordinator._generation_value


@pytest.mark.parametrize("route", ["direct", "coordinator"])
@pytest.mark.parametrize("case", [
    "current", "foreign-owner", "missing-state", "different-intent", "missing-intent", "missing-value",
])
def test_exact_owner_intent_predicate_and_short_circuit_order(route: str, case: str) -> None:
    owner = object()
    trace: list[str] = []
    value = "synthetic-intent"
    generation = _ObservedGeneration(
        object() if case == "foreign-owner" else owner, value, trace,
        missing_value=case == "missing-value",
    )
    state = None if case == "missing-state" else _ObservedState(
        None if case == "missing-intent" else "other-intent" if case == "different-intent" else value,
        trace,
    )
    validate = _validation(route, owner, trace)

    if case == "current":
        assert validate(generation, state) is value
    else:
        with pytest.raises(credentials.CredentialError, match="^login attempt is stale$"):
            validate(generation, state)

    expected = ["lookup", "owner", "value"]
    if case not in {"foreign-owner", "missing-state", "missing-value"}:
        expected.append("intent")
    assert trace == expected


@pytest.mark.parametrize("route", ["direct", "coordinator"])
@pytest.mark.parametrize("stage", ["lookup", "owner", "value", "intent"])
def test_first_lookup_observation_or_validation_exception_is_unchanged(
    route: str, stage: str,
) -> None:
    trace: list[str] = []
    owner = object()
    fault = _FaultError("synthetic first exception")
    generation = _ObservedGeneration(owner, "intent", trace, stage=stage, fault=fault)
    state = _ObservedState("intent", trace, fault if stage == "intent" else None)
    validate = _validation(route, owner, trace, fault if stage == "owner" else None)

    with pytest.raises(_FaultError) as raised:
        validate(generation, state)

    assert raised.value is fault
    order = ["lookup", "owner", "value", "intent"]
    assert trace == order[:order.index(stage) + 1]


@pytest.mark.parametrize("operation", ["commit", "reimport"])
def test_real_credential_mutation_keeps_intent_check_under_existing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    token = credentials.OAuthToken(
        "synthetic-access", "synthetic-refresh", 9_999_999_999_999, "synthetic-account",
    )
    trace: list[str] = []
    active = False

    def importer() -> credentials.OAuthToken:
        assert active
        trace.append("import")
        return token

    coordinator = credentials.OAuthCredentialCoordinator(
        tmp_path / "synthetic.json", credentials.CredentialFlavor.CODEX, importer=importer,
    )
    generation = coordinator.begin_login()
    original_locked = coordinator._locked
    original_validation = credentials._LoginGeneration.require_current_intent

    @contextmanager
    def locked() -> Iterator[int | None]:
        nonlocal active
        with original_locked() as directory_fd:
            active = True
            trace.append("lock-enter")
            try:
                yield directory_fd
            finally:
                active = False
                trace.append("lock-exit")

    def validate(
        current: credentials._LoginGeneration, observe_owner: Callable[[], object],
        state: credentials._StoreState | None,
    ) -> str:
        assert active and current is generation
        assert state is not None and state.intent_generation is not None
        trace.append("intent")
        return original_validation(current, observe_owner, state)

    monkeypatch.setattr(coordinator, "_locked", locked)
    monkeypatch.setattr(credentials._LoginGeneration, "require_current_intent", validate)
    if operation == "commit":
        result = coordinator.commit_login(token, generation)
    else:
        result = coordinator.explicit_reimport(generation)

    assert result == token
    assert not active
    assert trace == ["lock-enter", "intent", *(["import"] if operation == "reimport" else []), "lock-exit"]
    assert list(tmp_path.glob("*.tmp")) == []
