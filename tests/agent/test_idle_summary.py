"""Specification tests for the dormant, storage-free I3c summary adapter."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import socket
import subprocess
import weakref
from copy import copy
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

import nanobot.agent.idle_summary as idle_summary_module
from nanobot.agent.idle_runtime import IdleRuntimeBindingRegistry
from nanobot.agent.idle_summary import (
    IdleDependencyVerification,
    IdleSummaryAdapter,
    IdleSummaryAttemptAccounting,
    IdleSummaryAttemptResponse,
    IdleSummaryBindingError,
    IdleSummaryBudgetError,
    IdleSummaryCapabilityError,
    IdleSummaryDependencyError,
    IdleSummaryLimits,
    IdleSummaryMeasuredUsage,
    IdleSummaryNoProgressError,
    IdleSummaryRequest,
    IdleSummaryRequestReceipt,
    IdleSummaryTransportRegistry,
    _checkpoint_from_capture,
    _verify_dependencies_async,
)
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.session.idle_checkpoint import IdlePolicyBinding
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import new_source_event
from nanobot.utils.llm_runtime import LLMRuntime
from tests.session.test_idle_checkpoint import _artifact_source


class _Provider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(provider_name="fixture")

    def get_default_model(self) -> str:
        return "fixture"

    async def chat(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("I3c must use the injected attempt capability")


class _MalformedUsage:
    input_tokens = True
    output_tokens = float("nan")
    total_tokens = float("inf")
    request_count = 1


class _Attempt:
    provider_name = "fixture"
    transport = "fixture.one-shot"
    model = "fixture"
    context_window_tokens = 4096
    one_attempt = True
    runtime_fingerprint = "runtime-1"

    def __init__(self, outputs: list[object] | None = None, *, input_limit: int | None = None) -> None:
        self.outputs = list(outputs or ["summary"])
        self.input_limit = input_limit
        self.measurements: list[tuple[dict[str, object], ...]] = []
        self.requests = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.wait = False

    def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
        payload = tuple(dict(message) for message in messages)
        self.measurements.append(payload)
        raw = json.dumps(list(payload), ensure_ascii=False, separators=(",", ":")).encode()
        input_tokens = len(raw)
        if self.input_limit is not None:
            input_tokens = self.input_limit if input_tokens <= self.input_limit else input_tokens
        return IdleSummaryRequestReceipt(
            self.provider_name,
            self.transport,
            self.model,
            input_tokens,
            max_output_tokens,
            safety_buffer,
            self.context_window_tokens,
            input_tokens + max_output_tokens + safety_buffer,
            hashlib.sha256(raw).hexdigest(),
        )

    async def attempt_once(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        self.started.set()
        if self.wait:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                # Exercise the adapter's cancellation-safe join path.
                self.cancelled.set()
                await self.release.wait()
        output = self.outputs.pop(0) if self.outputs else "summary"
        if isinstance(output, BaseException):
            raise output
        if output == "second-attempt":
            return IdleSummaryAttemptResponse(
                "summary",
                IdleSummaryMeasuredUsage(request.receipt.input_tokens, 1, request.receipt.input_tokens + 1),
                request.receipt,
                attempts=2,
            )
        if output == "unknown-usage":
            return IdleSummaryAttemptResponse("summary", None, request.receipt)
        if output == "malformed-usage":
            return IdleSummaryAttemptResponse("summary", _MalformedUsage(), request.receipt)  # type: ignore[arg-type]
        if output == "truncated":
            return IdleSummaryAttemptResponse(
                "summary",
                IdleSummaryMeasuredUsage(request.receipt.input_tokens, 1, request.receipt.input_tokens + 1),
                request.receipt,
                finish_reason="length",
            )
        if output == "tool-call":
            return IdleSummaryAttemptResponse(
                "summary",
                IdleSummaryMeasuredUsage(request.receipt.input_tokens, 1, request.receipt.input_tokens + 1),
                request.receipt,
                tool_calls=(object(),),
            )
        return IdleSummaryAttemptResponse(
            str(output),
            IdleSummaryMeasuredUsage(request.receipt.input_tokens, 1, request.receipt.input_tokens + 1),
            request.receipt,
        )


def _limits(**overrides: object) -> IdleSummaryLimits:
    values: dict[str, object] = {
        "max_calls": 4,
        "max_total_tokens": 20_000,
        "max_wall_seconds": 5.0,
        "max_input_tokens": 20_000,
        "max_output_tokens": 256,
        "max_chunks": 4,
        "safety_buffer": 10,
    }
    values.update(overrides)
    return IdleSummaryLimits(**values)  # type: ignore[arg-type]


def _authority(tmp_path: Path, messages: list[dict[str, object]]):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    session = Session("test:idle-summary")
    session.messages = messages
    backend.save(session)
    source = backend.source
    source.upgrade_source_session(
        session.key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    source_snapshot = source.load_source_snapshot(session.key)
    source.record_user_activity(session.key, source_snapshot.state, accepted_foreground_user=True)
    policy = IdlePolicyBinding(
        "policy-1",
        tmp_path / "policy",
        None,
        lambda value: tmp_path / "policy" / value,
    )
    provider = _Provider()
    runtime = LLMRuntime(
        provider,
        "fixture",
        GenerationSettings(max_tokens=256),
        4096,
        snapshot_signature=("fixture", "fixture"),
    )
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=runtime,
        auxiliary_runtime=runtime,
        policy_binding=policy,
    )
    binding = registry.issue(session.key)
    capture = source.capture_idle_checkpoint(
        session.key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
        policy=policy,
    )
    assert capture is not None
    return source, policy, registry, binding, capture


def _adapter(
    attempt: _Attempt,
    source,
    registry: IdleRuntimeBindingRegistry,
    binding,
    limits: IdleSummaryLimits | None = None,
    **kwargs: object,
):
    transport = IdleSummaryTransportRegistry(
        binding_registry=registry,
        source_store=source,
    ).issue(binding, attempt)
    return IdleSummaryAdapter(
        transport,
        limits or _limits(),
        request_measurer=attempt,
        binding_registry=registry,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_subclassed_request_receipt_is_structurally_rejected(tmp_path: Path) -> None:
    class ReceiptSubclass(IdleSummaryRequestReceipt):
        pass

    source, _, registry, binding, capture = _authority(
        tmp_path, [{"role": "user", "content": "x"}]
    )
    attempt = _Attempt()
    original_measure = attempt.measure_request

    def measure(messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
        receipt = original_measure(
            messages,
            max_output_tokens=max_output_tokens,
            safety_buffer=safety_buffer,
        )
        return ReceiptSubclass(
            receipt.provider,
            receipt.transport,
            receipt.model,
            receipt.input_tokens,
            receipt.reserved_output_tokens,
            receipt.safety_buffer,
            receipt.model_window_tokens,
            receipt.total_reservation_tokens,
            receipt.request_digest,
        )

    attempt.measure_request = measure
    with pytest.raises(IdleSummaryBudgetError):
        await _adapter(attempt, source, registry, binding).summarize(
            capture, binding, deadline=asyncio.get_running_loop().time() + 2
        )
    assert attempt.requests == []


def test_raw_transport_cannot_bypass_sealing_or_copy_boundary(tmp_path: Path) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = _Attempt()
    with pytest.raises(IdleSummaryCapabilityError):
        IdleSummaryAdapter(
            attempt,
            _limits(),
            request_measurer=attempt,
            binding_registry=registry,
        )
    qualified = IdleSummaryTransportRegistry(
        binding_registry=registry,
        source_store=source,
    ).issue(binding, attempt)
    with pytest.raises(TypeError):
        copy(qualified)


@pytest.mark.parametrize("owner,name", [
    (idle_summary_module.IdleSummaryQualifiedTransport, name) for name in (
        "require_for_authority", "require_registry_relationship", "require_current_binding",
        "require_live_adapter_identity", "require_adapter_identity", "initialize_registry_authority",
    )
] + [
    (IdleSummaryTransportRegistry, name) for name in (
        "validate_transport", "require_adapter_identity", "initialize_transport_authority",
        "admit_transport_request", "record_transport_reservation", "matches_transport_authority",
        "matches_transport_pid", "matches_transport_source",
    )
] + [(IdleSummaryRequest, "claim_transport_reservation")])
def test_transport_owner_operation_exists(owner: type, name: str) -> None:
    assert callable(getattr(owner, name))


@pytest.mark.parametrize("mismatch", ["registry", "source", "binding", None])
def test_transport_authority_relationship_short_circuits_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str | None,
) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    transport_registry = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source)
    capability = transport_registry.issue(binding, _Attempt())
    validate = Mock()
    monkeypatch.setattr(transport_registry, "_validate_capability", validate)
    if mismatch is not None:
        with pytest.raises(IdleSummaryCapabilityError, match="foreign to this authority"):
            idle_summary_module._qualified_transport(
                capability, binding_registry=object() if mismatch == "registry" else registry,
                source_store=object() if mismatch == "source" else source,
                binding=object() if mismatch == "binding" else binding,
            )
        validate.assert_not_called()
    else:
        assert idle_summary_module._qualified_transport(
            capability, binding_registry=registry, source_store=source, binding=binding,
        ) is capability
        validate.assert_called_once_with(capability)


@pytest.mark.parametrize("failure_type", [None, RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit, BaseException])
def test_transport_raw_accessor_order_and_exception_boundary(failure_type: type | None) -> None:
    accessed: list[str] = []
    failure = failure_type("synthetic accessor fault") if failure_type is not None else None
    values = {name: getattr(_Attempt, name) for name in (
        "provider_name", "transport", "model", "runtime_fingerprint", "context_window_tokens", "attempt_once",
    )}

    class Raw:
        def __getattribute__(self, name: str):
            accessed.append(name)
            if name == "model" and failure is not None:
                raise failure
            return values[name]

    if failure is None:
        raw = Raw()
        assert IdleSummaryTransportRegistry._validate_raw(raw) is raw
        assert accessed == list(values)
    else:
        expected = failure_type if failure_type in (KeyboardInterrupt, SystemExit, GeneratorExit) else IdleSummaryCapabilityError
        with pytest.raises(expected) as caught:
            IdleSummaryTransportRegistry._validate_raw(Raw())
        assert accessed == ["provider_name", "transport", "model"]
        if expected is IdleSummaryCapabilityError:
            assert str(caught.value) == "raw one-attempt adapter is invalid"
            assert caught.value.__cause__ is None
        else:
            assert caught.value is failure


@pytest.mark.parametrize("mismatch", [None, "provider_name", "transport", "model", "context_window_tokens", "runtime_fingerprint"])
def test_transport_live_identity_reads_current_fields_and_first_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str | None,
) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    transport_registry = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source)
    capability = transport_registry.issue(binding, _Attempt())
    order = ["provider_name", "transport", "model", "context_window_tokens", "runtime_fingerprint"]
    accessed: list[str] = []
    for name in order:
        object.__setattr__(capability, "_" + name, -1 if name == "context_window_tokens" else "stale")

    class Raw:
        def __getattribute__(self, name: str):
            accessed.append(name)
            value = getattr(_Attempt, name)
            if name == mismatch:
                return -2 if name == "context_window_tokens" else "different"
            # Mutate the live frozen capability BEFORE its corresponding read.
            object.__setattr__(capability, "_" + name, value)
            return value

    monkeypatch.setattr(transport_registry, "_validate_raw", Mock(return_value=Raw()))
    if mismatch is None:
        transport_registry._validate_capability(capability)
        assert accessed == order
    else:
        with pytest.raises(IdleSummaryCapabilityError, match="identity changed"):
            transport_registry._validate_capability(capability)
        assert accessed == order[:order.index(mismatch) + 1]
    transport_registry._validate_raw.assert_called_once_with(capability._raw)


@pytest.mark.parametrize("fault", [None, "process", "validate", "claim"])
def test_transport_original_dynamic_seams_and_fault_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None,
) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    transport_registry = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source)
    capability = transport_registry.issue(binding, raw)
    request = IdleSummaryRequest(({"role": "user", "content": "x"},), raw.measure_request(
        ({"role": "user", "content": "x"},), max_output_tokens=1, safety_buffer=0,
    ))
    events: list[str] = []
    failure = RuntimeError("synthetic seam fault")
    for name, label in (("_check_process", "process"), ("_validate_capability", "validate"), ("_claim_request", "claim")):
        original = getattr(transport_registry, name)

        def seam(*args, original=original, label=label):
            events.append(label)
            if fault == label:
                raise failure
            return original(*args)

        monkeypatch.setattr(transport_registry, name, seam)
    if fault is None:
        capability._admit_request(request)
        assert events == ["claim", "validate", "process"]
        assert request._reservation.claimed is True
    else:
        with pytest.raises(RuntimeError) as caught:
            capability._admit_request(request)
        assert caught.value is failure
        assert events == ["claim", "validate", "process"][:["claim", "validate", "process"].index(fault) + 1]
        assert request._reservation.claimed is False
    assert raw.requests == []


@pytest.mark.parametrize("failure_type", [None, AttributeError, RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit])
def test_transport_binding_registry_read_stays_within_stale_refusal(
    tmp_path: Path, failure_type: type | None,
) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    failure = failure_type("synthetic binding registry access") if failure_type is not None else None
    activated = [False]

    class HostileRegistry(IdleSummaryTransportRegistry):
        def __getattribute__(self, name: str):
            if name == "_binding_registry" and activated[0] and failure is not None:
                raise failure
            return object.__getattribute__(self, name)

    transport_registry = HostileRegistry(binding_registry=registry, source_store=source)
    capability = transport_registry.issue(binding, raw)
    request = IdleSummaryRequest(({"role": "user", "content": "x"},), raw.measure_request(
        ({"role": "user", "content": "x"},), max_output_tokens=1, safety_buffer=0,
    ))
    if failure_type is None:
        del transport_registry._binding_registry
    activated[0] = True
    system = failure_type in (KeyboardInterrupt, SystemExit, GeneratorExit)
    expected = failure_type if system else IdleSummaryCapabilityError
    with pytest.raises(expected) as caught:
        capability._admit_request(request)
    if system:
        assert caught.value is failure
    else:
        assert str(caught.value) == "qualified transport binding is stale"
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True
        if failure is None:
            assert isinstance(caught.value.__context__, AttributeError)
        else:
            assert caught.value.__context__ is failure
    assert request._reservation.claimed is False
    assert transport_registry._issued_requests == {}
    assert raw.requests == []


@pytest.mark.parametrize("kind", ["missing", "malformed", "copied", "hostile", "system"])
def test_transport_reservation_malformed_and_hostile_owner(
    tmp_path: Path, kind: str,
) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    transport_registry = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source)
    capability = transport_registry.issue(binding, raw)
    request = IdleSummaryRequest(({"role": "user", "content": "x"},), raw.measure_request(
        ({"role": "user", "content": "x"},), max_output_tokens=1, safety_buffer=0,
    ))
    failure = KeyboardInterrupt("synthetic") if kind == "system" else BaseException("synthetic")
    reads: list[str] = []

    class Hostile:
        @property
        def owner(self):
            reads.append("owner")
            if len(reads) == 2:
                raise failure
            return lambda: request

    if kind == "missing":
        request = object.__new__(IdleSummaryRequest)
    elif kind == "copied":
        request = copy(request)
    else:
        object.__setattr__(request, "_reservation", object() if kind == "malformed" else Hostile())
    expected = KeyboardInterrupt if kind == "system" else IdleSummaryCapabilityError
    with pytest.raises(expected) as caught:
        capability._admit_request(request)
    if kind in ("hostile", "system"):
        assert reads == ["owner", "owner"]
    if kind == "system":
        assert caught.value is failure
    elif kind != "copied":
        assert str(caught.value) == "transport reservation request is invalid"
        assert caught.value.__cause__ is None
    assert transport_registry._issued_requests == {} and raw.requests == []


@pytest.mark.parametrize("entry", ["matching", "collision", "dead"])
def test_transport_reservation_registry_entry_precedes_claimed_bit(tmp_path: Path, entry: str) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    transport_registry = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source)
    capability = transport_registry.issue(binding, raw)
    request = IdleSummaryRequest(({"role": "user", "content": "x"},), raw.measure_request(
        ({"role": "user", "content": "x"},), max_output_tokens=1, safety_buffer=0,
    ))
    reservation = request._reservation
    reservation.claimed = True
    other = idle_summary_module._IdleSummaryRequestReservation()
    reference = weakref.ref(reservation if entry == "matching" else other)
    transport_registry._issued_requests[id(reservation)] = reference
    if entry == "dead":
        del other
        gc.collect()
    with pytest.raises(IdleSummaryCapabilityError, match="collided" if entry == "collision" else "more than once"):
        capability._admit_request(request)
    assert transport_registry._issued_requests == ({} if entry == "dead" else {id(reservation): reference})
    assert raw.requests == []


@pytest.mark.parametrize("entry", ["matching", "unrelated", "dead"])
def test_transport_weak_cleanup_uses_replaced_current_table(tmp_path: Path, entry: str) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    transport_registry = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source)
    capability = transport_registry.issue(binding, raw)
    request = IdleSummaryRequest(({"role": "user", "content": "x"},), raw.measure_request(
        ({"role": "user", "content": "x"},), max_output_tokens=1, safety_buffer=0,
    ))
    capability._admit_request(request)
    reservation_id = id(request._reservation)
    original_table = transport_registry._issued_requests
    original_reference = original_table[reservation_id]
    other = idle_summary_module._IdleSummaryRequestReservation()
    reference = original_reference if entry == "matching" else weakref.ref(other)
    transport_registry._issued_requests = {reservation_id: reference}
    if entry == "dead":
        del other
        gc.collect()
    del request
    gc.collect()
    assert original_reference() is None
    assert original_table == {reservation_id: original_reference}
    assert transport_registry._issued_requests == ({reservation_id: reference} if entry == "unrelated" else {})
    assert raw.requests == []


def test_transport_owner_wrong_seals_are_effect_free(tmp_path: Path) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    transport_registry = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source)
    capability = transport_registry.issue(binding, raw)
    request = IdleSummaryRequest(({"role": "user", "content": "x"},), raw.measure_request(
        ({"role": "user", "content": "x"},), max_output_tokens=1, safety_buffer=0,
    ))
    authority = capability._authority
    for operation, args, error in (
        (transport_registry.initialize_transport_authority, (object(), capability), "cannot be reconstructed"),
        (capability.initialize_registry_authority, (object(), object()), "cannot be reconstructed"),
        (request.claim_transport_reservation, (object(), transport_registry), "reservation request is invalid"),
        (transport_registry.record_transport_reservation, (object(), request, request._reservation, request), "reservation request is invalid"),
    ):
        with pytest.raises(IdleSummaryCapabilityError, match=error):
            operation(*args)
        assert capability._authority is authority
        assert request._reservation.claimed is False
        assert transport_registry._issued_requests == {} and raw.requests == []


def test_transport_constructor_reinitialization_frozen_copy_and_aliases(tmp_path: Path) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    transport_registry = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source)
    capability = transport_registry.issue(binding, raw)
    assert IdleSummaryTransportRegistry.qualify is IdleSummaryTransportRegistry.issue
    assert idle_summary_module.QualifiedIdleSummaryAttempt is type(capability)
    assert idle_summary_module.IdleQualifiedSummaryProvider is type(capability)
    with pytest.raises(AttributeError):
        capability._authority = object()
    with pytest.raises(TypeError):
        copy(capability)
    with pytest.raises(IdleSummaryCapabilityError, match="cannot be reconstructed"):
        capability.__init__(object(), transport_registry, binding, raw, capability._pid)
    replacement = IdleSummaryTransportRegistry(binding_registry=registry, source_store=source)
    capability.__init__(idle_summary_module._TRANSPORT_SEAL, replacement, binding, raw, capability._pid)
    assert capability._registry is replacement and capability._authority is replacement._authority
    replacement._validate_capability(capability)


def test_constructor_rejects_measurement_identity_not_matching_binding_runtime(tmp_path: Path) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    measurer = _Attempt()
    raw.context_window_tokens = 8192
    measurer.context_window_tokens = 8192
    transport = IdleSummaryTransportRegistry(
        binding_registry=registry,
        source_store=source,
    ).issue(binding, raw)
    with pytest.raises(IdleSummaryCapabilityError):
        IdleSummaryAdapter(
            transport,
            _limits(),
            request_measurer=measurer,
            binding_registry=registry,
        )


@pytest.mark.asyncio
async def test_transport_reservation_rejects_copies_without_retaining_request_payload(
    tmp_path: Path,
) -> None:
    source, _, registry, binding, _ = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = _Attempt()
    transport = IdleSummaryTransportRegistry(
        binding_registry=registry,
        source_store=source,
    ).issue(binding, attempt)
    messages = ({"role": "user", "content": "secret payload" * 100},)
    receipt = attempt.measure_request(messages, max_output_tokens=256, safety_buffer=10)
    request = IdleSummaryRequest(messages, receipt)
    request_reference = weakref.ref(request)

    copied_before_claim = copy(request)
    with pytest.raises(IdleSummaryCapabilityError):
        transport.attempt_once(copied_before_claim)
    forged = object.__new__(IdleSummaryRequest)
    with pytest.raises(IdleSummaryCapabilityError):
        transport.attempt_once(forged)
    await transport.attempt_once(request)
    for copied in (copy(request), replace(request, messages=request.messages)):
        with pytest.raises(IdleSummaryCapabilityError):
            transport.attempt_once(copied)
    assert len(attempt.requests) == 1

    attempt.requests.clear()
    del copied
    del copied_before_claim
    del request
    gc.collect()
    assert request_reference() is None
    assert transport._registry._issued_requests == {}

    # A new request gets a fresh reservation even if the allocator reuses the
    # collected request's identity.
    replacement = IdleSummaryRequest(messages, receipt)
    await transport.attempt_once(replacement)
    assert len(attempt.requests) == 1


@pytest.mark.parametrize(
    "bad",
    [
        object(),
        type("Unqualified", (), {"provider_name": "x"})(),
    ],
)
def test_construction_rejects_missing_or_unqualified_capability_before_source_access(bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(IdleSummaryCapabilityError):
        IdleSummaryAdapter(bad, _limits())


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_calls", 0),
        ("max_total_tokens", 0),
        ("max_wall_seconds", 0.0),
        ("max_input_tokens", 0),
        ("max_output_tokens", 0),
        ("max_chunks", 0),
        ("safety_buffer", -1),
    ],
)
def test_limits_are_explicit_and_strict(field: str, value: object) -> None:
    values = {
        "max_calls": 1,
        "max_total_tokens": 1,
        "max_wall_seconds": 1.0,
        "max_input_tokens": 1,
        "max_output_tokens": 1,
        "max_chunks": 1,
        "safety_buffer": 0,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        IdleSummaryLimits(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_tokens", True),
        ("output_tokens", float("nan")),
        ("total_tokens", float("inf")),
        ("reasoning_tokens", False),
    ],
)
def test_usage_receipts_reject_boolean_and_nonfinite_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "reasoning_tokens": None,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        IdleSummaryMeasuredUsage(**values)  # type: ignore[arg-type]


def test_request_receipt_rejects_boolean_and_nonfinite_reservations() -> None:
    values: dict[str, object] = {
        "provider": "fixture",
        "transport": "fixture.one-shot",
        "model": "fixture",
        "input_tokens": 1,
        "reserved_output_tokens": 1,
        "safety_buffer": 0,
        "model_window_tokens": 2,
        "total_reservation_tokens": 2,
        "request_digest": "0" * 64,
    }
    for field, value in (("input_tokens", True), ("total_reservation_tokens", float("nan"))):
        invalid = dict(values)
        invalid[field] = value
        with pytest.raises((TypeError, ValueError)):
            IdleSummaryRequestReceipt(**invalid)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_success_is_root_history_only_and_does_not_write(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "old objective"},
            {"role": "assistant", "content": "observed progress"},
            {"role": "user", "content": "new objective"},
        ],
    )
    attempt = _Attempt(["facts and next action"])
    before = source.load_source_snapshot("test:idle-summary")
    result = await _adapter(attempt, source, registry, binding).summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    after = source.load_source_snapshot("test:idle-summary")
    assert result.status == "complete" and result.proposal is not None
    assert result.proposal.checkpoint["boundary"] == len(capture.snapshot.events)
    assert result.proposal.checkpoint["producing_provider"] == "fixture"
    assert result.proposal.checkpoint["producing_model"] == "fixture"
    assert len(result.proposal.proof.chunks) == 1
    assert before == after
    assert len(attempt.requests) == 1
    receipt = result.requests[0]
    assert result.accounting == IdleSummaryAttemptAccounting(
        1,
        receipt.total_reservation_tokens,
        1,
        receipt.input_tokens + 1,
        0,
        True,
    )


@pytest.mark.asyncio
async def test_adapter_is_network_and_process_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = _Attempt(["offline summary"])

    def deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("idle summary attempted outbound network access")

    def deny_process(*args: object, **kwargs: object) -> None:
        raise AssertionError("idle summary attempted process execution")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setattr(subprocess, "Popen", Mock(side_effect=AssertionError("idle summary attempted process execution")))
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=AssertionError("idle summary attempted process execution")),
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        AsyncMock(side_effect=AssertionError("idle summary attempted process execution")),
    )

    result = await _adapter(attempt, source, registry, binding).summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    assert result.status == "complete" and result.proposal is not None


@pytest.mark.asyncio
async def test_live_transport_identity_mutation_after_attempt_has_no_proposal(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])

    class MutatingAttempt(_Attempt):
        async def attempt_once(self, request):  # type: ignore[no-untyped-def]
            response = await super().attempt_once(request)
            self.model = "mutated-model"
            return response

    attempt = MutatingAttempt()
    result = await _adapter(attempt, source, registry, binding).summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    assert result.proposal is None and result.status == "stale_binding"
    assert result.accounting == IdleSummaryAttemptAccounting(
        1, result.requests[0].total_reservation_tokens, 1, result.requests[0].input_tokens + 1, 0, True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "receipt_field", "value"),
    [
        ("provider_name", "provider", "mutated-provider"),
        ("transport", "transport", "mutated.transport"),
        ("model", "model", "mutated-model"),
        ("context_window_tokens", "model_window_tokens", 8192),
        ("runtime_fingerprint", None, "runtime-mutated"),
    ],
)
async def test_measurer_identity_mutation_during_measurement_rejects_before_transport(
    tmp_path: Path, field: str, receipt_field: str | None, value: object
) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()

    class MutatingMeasurer(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            setattr(self, field, value)
            if receipt_field is None:
                return receipt
            return replace(receipt, **{receipt_field: value})

    measurer = MutatingMeasurer()
    transport = IdleSummaryTransportRegistry(
        binding_registry=registry,
        source_store=source,
    ).issue(binding, raw)
    adapter = IdleSummaryAdapter(
        transport,
        _limits(),
        request_measurer=measurer,
        binding_registry=registry,
    )
    with pytest.raises(IdleSummaryBindingError):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert measurer.measurements
    assert raw.requests == []
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)


@pytest.mark.asyncio
async def test_measurer_identity_changed_before_measurement_does_not_receive_evidence(
    tmp_path: Path,
) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    measurer = _Attempt()
    transport = IdleSummaryTransportRegistry(
        binding_registry=registry,
        source_store=source,
    ).issue(binding, raw)
    adapter = IdleSummaryAdapter(
        transport,
        _limits(),
        request_measurer=measurer,
        binding_registry=registry,
    )
    measurer.model = "changed-before-measurement"
    with pytest.raises(IdleSummaryBindingError):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert measurer.measurements == []
    assert raw.requests == []
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)


@pytest.mark.asyncio
async def test_receipt_identity_different_from_frozen_authority_rejects_before_dispatch(
    tmp_path: Path,
) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()

    class MismatchedReceiptMeasurer(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            return replace(receipt, provider="receipt-only-mutation")

    measurer = MismatchedReceiptMeasurer()
    transport = IdleSummaryTransportRegistry(
        binding_registry=registry,
        source_store=source,
    ).issue(binding, raw)
    adapter = IdleSummaryAdapter(
        transport,
        _limits(),
        request_measurer=measurer,
        binding_registry=registry,
    )
    with pytest.raises(IdleSummaryBindingError):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert raw.requests == []
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)


@pytest.mark.asyncio
async def test_stable_separate_measurer_and_raw_capability_succeeds(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    raw = _Attempt()
    measurer = _Attempt()
    transport = IdleSummaryTransportRegistry(
        binding_registry=registry,
        source_store=source,
    ).issue(binding, raw)
    result = await IdleSummaryAdapter(
        transport,
        _limits(),
        request_measurer=measurer,
        binding_registry=registry,
    ).summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert result.status == "complete" and result.proposal is not None
    assert len(raw.requests) == 1
    assert measurer.measurements


@pytest.mark.asyncio
async def test_capture_binding_and_pending_state_fail_before_transport(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = _Attempt()
    adapter = _adapter(attempt, source, registry, binding)
    with pytest.raises(IdleSummaryBindingError):
        await adapter.summarize(
            replace(capture, runtime_fingerprint="different"),
            binding,
            deadline=asyncio.get_running_loop().time() + 2,
        )
    pending = replace(capture.token.pending_state, pending_user_turn_present=True)
    with pytest.raises(IdleSummaryBindingError):
        await adapter.summarize(
            replace(capture, token=replace(capture.token, pending_state=pending)),
            binding,
            deadline=asyncio.get_running_loop().time() + 2,
        )
    assert attempt.measurements == [] and attempt.requests == []
    assert source.load_source_snapshot("test:idle-summary") == capture.snapshot


@pytest.mark.asyncio
async def test_stale_capture_is_rejected_even_with_a_fresh_binding(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path, [{"role": "user", "content": "original"}]
    )
    current = source.load_source_snapshot("test:idle-summary")
    source.append_source_events(
        "test:idle-summary",
        current.state,
        [new_source_event({"role": "assistant", "content": "new source evidence"}, run_id="fresh")],
    )
    fresh_binding = registry.issue("test:idle-summary")
    assert fresh_binding is not binding

    attempt = _Attempt()
    with pytest.raises(IdleSummaryBindingError):
        await _adapter(attempt, source, registry, fresh_binding).summarize(
            capture, fresh_binding, deadline=asyncio.get_running_loop().time() + 2
        )
    assert attempt.measurements == [] and attempt.requests == []


@pytest.mark.asyncio
async def test_no_progress_and_indivisible_budget_are_preflight_failures(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = _Attempt()
    adapter = _adapter(attempt, source, registry, binding, _limits(max_input_tokens=1))
    with pytest.raises(IdleSummaryBudgetError):
        await adapter.summarize(
            capture, binding, deadline=asyncio.get_running_loop().time() + 2
        )
    assert attempt.requests == []

    # The source remains untouched after this preflight rejection.
    assert source.load_source_snapshot("test:idle-summary") == capture.snapshot


@pytest.mark.asyncio
async def test_impossible_later_indivisible_group_fails_before_dispatch(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "small first group"},
            {"role": "assistant", "content": "x" * 5_000},
        ],
    )
    attempt = _Attempt()
    adapter = _adapter(attempt, source, registry, binding, _limits(max_input_tokens=700))
    with pytest.raises(IdleSummaryBudgetError):
        await adapter.summarize(
            capture, binding, deadline=asyncio.get_running_loop().time() + 2
        )
    assert len(attempt.measurements) == 2 and attempt.requests == []
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)


@pytest.mark.asyncio
async def test_later_atomic_group_uses_finalized_measurement_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "small first group"},
            {"role": "assistant", "content": "later group"},
        ],
    )
    order: list[str] = []

    class MeasuredAttempt(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            order.append("measure")
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            input_tokens = 2_000 if '"later group"' in str(messages[-1]["content"]) else 100
            return replace(
                receipt,
                input_tokens=input_tokens,
                total_reservation_tokens=input_tokens + max_output_tokens + safety_buffer,
            )

    class Verifier:
        def verify(self, *, capture, binding, references):  # type: ignore[no-untyped-def]
            order.append("verify")
            return IdleDependencyVerification(
                capture,
                binding,
                references,
                binding.policy_binding.fingerprint,
                binding.artifact_namespace,
            )

    # The source fixture is dependency-free.  Force the same exact-reference
    # path used by dependent evidence so this regression proves authorization
    # precedes every evidence-bearing preflight measurement.
    monkeypatch.setattr(
        idle_summary_module,
        "derive_summary_references",
        lambda proof, snapshot: ("artifact:test",),
    )
    attempt = MeasuredAttempt()
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        _limits(max_input_tokens=700),
        dependency_verifier=Verifier(),
    )
    with pytest.raises(IdleSummaryBudgetError):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert order == ["verify", "measure", "verify", "measure"]
    assert attempt.requests == []
    assert source.load_source_snapshot("test:idle-summary") == capture.snapshot
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)


@pytest.mark.asyncio
async def test_repetitive_large_payload_uses_trusted_token_measurement(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path, [{"role": "user", "content": "x" * 1_000}]
    )

    class TokenMeasuredAttempt(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            return replace(
                receipt,
                input_tokens=100,
                total_reservation_tokens=100 + max_output_tokens + safety_buffer,
            )

    attempt = TokenMeasuredAttempt(["folded summary"])
    before = source.load_source_snapshot("test:idle-summary")
    result = await _adapter(
        attempt,
        source,
        registry,
        binding,
        _limits(max_input_tokens=700),
    ).summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert result.status == "complete" and result.proposal is not None
    assert len(attempt.requests) == 1
    assert len(attempt.measurements) == 2
    assert source.load_source_snapshot("test:idle-summary") == before == capture.snapshot


@pytest.mark.asyncio
async def test_identical_payload_conflicting_measurements_reject_before_dispatch(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])

    class ConflictingAttempt(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            input_tokens = 100 + len(self.measurements) - 1
            return replace(
                receipt,
                input_tokens=input_tokens,
                total_reservation_tokens=input_tokens + max_output_tokens + safety_buffer,
            )

    attempt = ConflictingAttempt()
    adapter = _adapter(attempt, source, registry, binding, _limits(max_input_tokens=700))
    with pytest.raises(IdleSummaryBudgetError):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert len(attempt.measurements) == 2 and attempt.requests == []
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)


@pytest.mark.asyncio
async def test_retained_receipt_mutation_cannot_hide_conflicting_measurement(
    tmp_path: Path,
) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])

    class RetainingAttempt(_Attempt):
        def __init__(self) -> None:
            super().__init__()
            self.retained_receipt = None

        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            if self.retained_receipt is None:
                self.retained_receipt = replace(
                    receipt,
                    input_tokens=100,
                    total_reservation_tokens=100 + max_output_tokens + safety_buffer,
                )
                return self.retained_receipt
            second = replace(
                receipt,
                input_tokens=101,
                total_reservation_tokens=101 + max_output_tokens + safety_buffer,
            )
            object.__setattr__(self.retained_receipt, "input_tokens", second.input_tokens)
            object.__setattr__(
                self.retained_receipt,
                "total_reservation_tokens",
                second.total_reservation_tokens,
            )
            return second

    attempt = RetainingAttempt()
    adapter = _adapter(attempt, source, registry, binding, _limits(max_input_tokens=700))
    with pytest.raises(IdleSummaryBudgetError):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert attempt.requests == []
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)


@pytest.mark.asyncio
async def test_single_measurement_receipt_is_detached_from_measurer(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])

    class RetainingAttempt(_Attempt):
        def __init__(self) -> None:
            super().__init__()
            self.retained_receipt = None

        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            self.retained_receipt = receipt
            return receipt

    attempt = RetainingAttempt()
    adapter = _adapter(attempt, source, registry, binding)
    candidate, previous = _checkpoint_from_capture(capture)
    prepared = adapter._prepare_payload(
        candidate,
        candidate.boundaries[-1],
        candidate.previous_boundary,
        previous.summary if previous is not None else None,
        (),
    )
    measured = adapter._measure(
        prepared,
        asyncio.get_running_loop().time() + 2,
        {},
    )
    assert measured.receipt is not None and attempt.retained_receipt is not measured.receipt
    detached_input_tokens = measured.receipt.input_tokens
    detached_total_tokens = measured.receipt.total_reservation_tokens
    object.__setattr__(attempt.retained_receipt, "input_tokens", detached_input_tokens + 1)
    object.__setattr__(attempt.retained_receipt, "total_reservation_tokens", detached_total_tokens + 1)
    request = IdleSummaryRequest(measured.payload, measured.receipt)
    assert request.receipt.input_tokens == detached_input_tokens
    assert request.receipt.total_reservation_tokens == detached_total_tokens


@pytest.mark.asyncio
async def test_measurer_input_mutation_rejects_before_dispatch(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])

    class MutatingMeasurer(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            messages[-1]["content"] = "mutated finalized request"
            return super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )

    attempt = MutatingMeasurer()
    adapter = _adapter(attempt, source, registry, binding)
    with pytest.raises(IdleSummaryBudgetError):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert attempt.requests == []
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)


@pytest.mark.asyncio
async def test_root_candidate_retains_newest_user_and_waits_for_tool_closure(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "old"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call", "type": "function", "function": {"name": "tool", "arguments": "{}"}}
                ],
            },
            {"role": "user", "content": "child", "_stream_id": "child", "_parent_stream_id": "root"},
            {
                "role": "assistant",
                "content": "child secret",
                "_stream_id": "child",
                "_parent_stream_id": "root",
            },
            {"role": "tool", "content": "tool result", "tool_call_id": "call"},
            {"role": "user", "content": "newest"},
        ],
    )
    candidate, previous = _checkpoint_from_capture(capture)
    assert previous is None
    assert candidate.boundaries == (1, 5, 6)
    assert [node.message.get("content") for node in candidate.transcript.nodes] == [
        "old",
        "",
        "tool result",
        "newest",
    ]
    assert "child secret" not in str(candidate.summary_evidence(5))
    attempt = _Attempt(["closed summary"])
    result = await _adapter(attempt, source, registry, binding).summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    assert result.proposal is not None and result.proposal.checkpoint["boundary"] == 6


@pytest.mark.asyncio
async def test_root_candidate_projects_captured_artifact_without_retrieval(tmp_path: Path) -> None:
    source, admitted, _, descriptor, policy = await _artifact_source(
        tmp_path, "preview head\n" + ("a" * 2000) + "\nunique full-tail sentinel"
    )
    capture = source.capture_idle_checkpoint(
        admitted.state.session_key,
        runtime_fingerprint="runtime-1",
        policy_fingerprint=policy.fingerprint,
        policy=policy,
    )
    assert capture is not None
    candidate, previous = _checkpoint_from_capture(capture)
    assert previous is None
    evidence = candidate.summary_evidence(candidate.boundaries[-1])
    assert tuple(projection.event_id for projection in evidence.projections) == (descriptor.event_id,)
    assert evidence.projections[0].binding.artifact_id == descriptor.artifact_id
    assert "unique full-tail sentinel" not in str(evidence.nodes)


@pytest.mark.asyncio
async def test_checkpoint_only_capture_is_rejected_as_no_progress(tmp_path: Path) -> None:
    source, policy, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    first = await _adapter(_Attempt(["first"]), source, registry, binding).summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    assert first.proposal is not None
    source.publish_idle_checkpoint(
        "test:idle-summary",
        capture,
        first.proposal.checkpoint,
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
        policy=policy,
        proof=first.proposal.proof,
    )
    current = source.load_source_snapshot("test:idle-summary")
    registry = IdleRuntimeBindingRegistry(
        source_store=source,
        foreground_runtime=binding.foreground_runtime,
        auxiliary_runtime=binding.auxiliary_runtime,
        policy_binding=policy,
    )
    binding = registry.issue("test:idle-summary")
    next_capture = source.capture_idle_checkpoint(
        "test:idle-summary",
        runtime_fingerprint="runtime-1",
        policy_fingerprint="policy-1",
        policy=policy,
    )
    assert next_capture is not None and next_capture.snapshot == current
    with pytest.raises(IdleSummaryNoProgressError):
        await _adapter(_Attempt(), source, registry, binding).summarize(
            next_capture, binding, deadline=asyncio.get_running_loop().time() + 2
        )


@pytest.mark.asyncio
async def test_multiple_disjoint_chunks_fold_exact_proof_chain(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        ],
    )
    attempt = _Attempt(["first", "second", "third"])
    # One event-sized payload fits; a combined payload is rejected by the
    # injected finalized measurement, forcing three contiguous calls.
    attempt.input_limit = 700
    result = await _adapter(
        attempt, source, registry, binding, _limits(max_input_tokens=700, max_chunks=3)
    ).summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    assert result.status == "complete" and result.proposal is not None
    assert len(result.proposal.proof.chunks) == 3
    assert [chunk.previous_summary for chunk in result.proposal.proof.chunks] == [None, "first", "second"]
    assert [chunk.summary for chunk in result.proposal.proof.chunks] == ["first", "second", "third"]
    assert [chunk.evidence.after for chunk in result.proposal.proof.chunks] == [0, 1, 2]
    assert [chunk.evidence.boundary for chunk in result.proposal.proof.chunks] == [1, 2, 3]
    assert result.proposal.usage.calls == 3
    assert source.load_source_snapshot("test:idle-summary") == capture.snapshot


@pytest.mark.asyncio
async def test_oversized_furthest_boundary_searches_smaller_closed_boundary(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        ],
    )

    class SplittableAttempt(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            evidence = str(messages[-1]["content"])
            input_tokens = 1_000 if '"three"' in evidence and '"two"' in evidence else 100
            return replace(
                receipt,
                input_tokens=input_tokens,
                total_reservation_tokens=input_tokens + max_output_tokens + safety_buffer,
            )

    attempt = SplittableAttempt(["first", "second"])
    result = await _adapter(
        attempt,
        source,
        registry,
        binding,
        _limits(max_input_tokens=700, max_chunks=2),
    ).summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert result.status == "complete" and result.proposal is not None
    assert [chunk.evidence.boundary for chunk in result.proposal.proof.chunks] == [2, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output", ["unknown-usage", "malformed-usage", "truncated", "tool-call", "second-attempt"]
)
async def test_malformed_or_hidden_retry_result_has_no_proposal(tmp_path: Path, output: str) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    result = await _adapter(_Attempt([output]), source, registry, binding).summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    assert result.proposal is None
    assert result.status in {"invalid_response", "failed"}
    assert result.reserved_usage.calls == 1
    assert result.requests


@pytest.mark.asyncio
async def test_usage_accessor_failure_is_terminally_accounted(tmp_path: Path) -> None:
    class UsageAccessorFailure:
        @property
        def usage(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("usage accessor failed")

    class UsageAccessorAttempt(_Attempt):
        async def attempt_once(self, request):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            self.started.set()
            return UsageAccessorFailure()

    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = UsageAccessorAttempt()
    result = await _adapter(attempt, source, registry, binding).summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    assert result.status == "invalid_response" and result.proposal is None
    assert result.accounting == IdleSummaryAttemptAccounting(1, result.requests[0].total_reservation_tokens, 1, 0, 1, False)


@pytest.mark.asyncio
async def test_content_accessor_failure_is_terminally_accounted_and_system_errors_propagate(
    tmp_path: Path,
) -> None:
    class ContentAccessorFailure(BaseException):
        pass

    class ContentAccessorResponse(IdleSummaryAttemptResponse):
        def __getattribute__(self, name):  # type: ignore[no-untyped-def]
            if name == "content":
                raise ContentAccessorFailure("content accessor failed")
            return super().__getattribute__(name)

    class ContentAccessorAttempt(_Attempt):
        async def attempt_once(self, request):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            self.started.set()
            response = object.__new__(ContentAccessorResponse)
            object.__setattr__(
                response,
                "usage",
                IdleSummaryMeasuredUsage(
                    request.receipt.input_tokens,
                    1,
                    request.receipt.input_tokens + 1,
                ),
            )
            object.__setattr__(response, "request", request.receipt)
            object.__setattr__(response, "finish_reason", "stop")
            object.__setattr__(response, "tool_calls", ())
            object.__setattr__(response, "attempts", 1)
            object.__setattr__(response, "provider_state", None)
            object.__setattr__(response, "content", "summary")
            return response

    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = ContentAccessorAttempt()
    result = await _adapter(attempt, source, registry, binding).summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    assert result.status == "invalid_response" and result.proposal is None
    assert result.accounting == IdleSummaryAttemptAccounting(
        1, result.requests[0].total_reservation_tokens, 1, result.requests[0].input_tokens + 1, 0, True
    )

    class SystemUsageAccessorFailure:
        @property
        def usage(self):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt()

    class SystemAccessorAttempt(_Attempt):
        async def attempt_once(self, request):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            self.started.set()
            return SystemUsageAccessorFailure()

    system_attempt = SystemAccessorAttempt()
    with pytest.raises(KeyboardInterrupt):
        await _adapter(system_attempt, source, registry, binding).summarize(
            capture, binding, deadline=asyncio.get_running_loop().time() + 2
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("system_error", [KeyboardInterrupt, SystemExit, GeneratorExit])
async def test_usage_system_exception_finalizes_unknown_accounting(
    tmp_path: Path, system_error: type[BaseException]
) -> None:
    class UsageSystemFailure:
        @property
        def usage(self):  # type: ignore[no-untyped-def]
            raise system_error()

    class UsageSystemAttempt(_Attempt):
        async def attempt_once(self, request):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            self.started.set()
            return UsageSystemFailure()

    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    snapshots: list[IdleSummaryAttemptAccounting] = []
    attempt = UsageSystemAttempt()
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        accounting_sink=snapshots.append,
    )
    with pytest.raises(system_error):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(
        1, attempt.requests[0].receipt.total_reservation_tokens, 1, 0, 1, False
    )
    assert snapshots and snapshots[-1] == adapter.last_accounting


@pytest.mark.asyncio
@pytest.mark.parametrize("system_error", [KeyboardInterrupt, SystemExit, GeneratorExit])
async def test_content_system_exception_preserves_known_usage_and_finalizes(
    tmp_path: Path, system_error: type[BaseException]
) -> None:
    class ContentSystemResponse(IdleSummaryAttemptResponse):
        def __getattribute__(self, name):  # type: ignore[no-untyped-def]
            if name == "content":
                raise system_error()
            return super().__getattribute__(name)

    class ContentSystemAttempt(_Attempt):
        async def attempt_once(self, request):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            self.started.set()
            response = object.__new__(ContentSystemResponse)
            object.__setattr__(
                response,
                "usage",
                IdleSummaryMeasuredUsage(
                    request.receipt.input_tokens,
                    1,
                    request.receipt.input_tokens + 1,
                ),
            )
            object.__setattr__(response, "request", request.receipt)
            object.__setattr__(response, "finish_reason", "stop")
            object.__setattr__(response, "tool_calls", ())
            object.__setattr__(response, "attempts", 1)
            object.__setattr__(response, "provider_state", None)
            object.__setattr__(response, "content", "summary")
            return response

    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    snapshots: list[IdleSummaryAttemptAccounting] = []
    attempt = ContentSystemAttempt()
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        accounting_sink=snapshots.append,
    )
    with pytest.raises(system_error):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(
        1,
        attempt.requests[0].receipt.total_reservation_tokens,
        1,
        attempt.requests[0].receipt.input_tokens + 1,
        0,
        True,
    )
    assert snapshots and snapshots[-1] == adapter.last_accounting


@pytest.mark.asyncio
async def test_later_unknown_attempt_charges_reservation_and_retains_prior_usage(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        ],
    )
    attempt = _Attempt(["first", "unknown-usage"], input_limit=700)
    result = await _adapter(
        attempt,
        source,
        registry,
        binding,
        _limits(max_input_tokens=700, max_chunks=2),
    ).summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert result.proposal is None and result.status == "invalid_response"
    assert result.measured_usage is not None
    assert result.measured_usage.total_tokens == result.requests[0].input_tokens + 1
    assert result.accounting.dispatched_calls == 2
    assert result.accounting.unknown_dispatched == 1
    assert result.accounting.known_measured_tokens == result.measured_usage.total_tokens
    assert result.accounting.exact is False
    assert result.usage == result.reserved_usage


@pytest.mark.asyncio
async def test_later_dependency_failure_finalizes_known_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        ],
    )

    class SplittableAttempt(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            evidence = str(messages[-1]["content"])
            input_tokens = 1_000 if '"three"' in evidence and '"two"' in evidence else 100
            return replace(
                receipt,
                input_tokens=input_tokens,
                total_reservation_tokens=input_tokens + max_output_tokens + safety_buffer,
            )

    attempt = SplittableAttempt(["first", "second"], input_limit=700)

    class Verifier:
        def verify(self, *, capture, binding, references):  # type: ignore[no-untyped-def]
            if attempt.requests:
                raise RuntimeError("dependency authority changed")
            return IdleDependencyVerification(
                capture,
                binding,
                references,
                binding.policy_binding.fingerprint,
                binding.artifact_namespace,
            )

    monkeypatch.setattr(
        idle_summary_module,
        "derive_summary_references",
        lambda proof, snapshot: ("artifact:test",),
    )
    snapshots: list[IdleSummaryAttemptAccounting] = []
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        _limits(max_input_tokens=700, max_chunks=2),
        dependency_verifier=Verifier(),
        accounting_sink=snapshots.append,
    )
    with pytest.raises(IdleSummaryDependencyError):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(
        1,
        attempt.requests[0].receipt.total_reservation_tokens,
        1,
        attempt.requests[0].receipt.input_tokens + 1,
        0,
        True,
    )
    assert snapshots and snapshots[-1] == adapter.last_accounting


@pytest.mark.asyncio
async def test_later_measurement_failure_finalizes_known_usage(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        ],
    )

    class FailingAfterFirstAttempt(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            if self.requests:
                raise RuntimeError("measurement authority changed")
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            evidence = str(messages[-1]["content"])
            input_tokens = 1_000 if '"three"' in evidence and '"two"' in evidence else 100
            return replace(
                receipt,
                input_tokens=input_tokens,
                total_reservation_tokens=input_tokens + max_output_tokens + safety_buffer,
            )

    attempt = FailingAfterFirstAttempt(["first"], input_limit=700)
    snapshots: list[IdleSummaryAttemptAccounting] = []
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        _limits(max_input_tokens=700, max_chunks=2),
        accounting_sink=snapshots.append,
    )
    with pytest.raises(IdleSummaryBudgetError):
        await adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(
        1,
        attempt.requests[0].receipt.total_reservation_tokens,
        1,
        attempt.requests[0].receipt.input_tokens + 1,
        0,
        True,
    )
    assert snapshots and snapshots[-1] == adapter.last_accounting


@pytest.mark.asyncio
async def test_later_clock_failure_finalizes_known_usage(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ],
    )

    class FailingClock:
        def __init__(self) -> None:
            self.failed = False

        def __call__(self) -> float:
            if self.failed:
                raise RuntimeError("clock failed")
            return asyncio.get_running_loop().time()

    clock = FailingClock()

    class FailingAfterAttempt(_Attempt):
        pass

    snapshots: list[IdleSummaryAttemptAccounting] = []
    attempt = FailingAfterAttempt()

    def observe(snapshot: IdleSummaryAttemptAccounting) -> None:
        snapshots.append(snapshot)
        if snapshot.known_measured_tokens:
            clock.failed = True

    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        clock=clock,
        accounting_sink=observe,
    )
    with pytest.raises(RuntimeError, match="clock failed"):
        await adapter.summarize(capture, binding, deadline=clock() + 2)
    assert adapter.last_accounting == IdleSummaryAttemptAccounting(
        1,
        attempt.requests[0].receipt.total_reservation_tokens,
        1,
        attempt.requests[0].receipt.input_tokens + 1,
        0,
        True,
    )
    assert snapshots and snapshots[-1] == adapter.last_accounting


@pytest.mark.asyncio
async def test_custom_non_system_base_exception_is_typed_and_accounted(tmp_path: Path) -> None:
    class AdapterFailure(BaseException):
        pass

    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    result = await _adapter(
        _Attempt([AdapterFailure("transport failed")]), source, registry, binding
    ).summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert result.status == "failed" and result.proposal is None
    assert result.accounting.dispatched_calls == 1
    assert result.accounting.unknown_dispatched == 1
    assert result.accounting.exact is False


@pytest.mark.asyncio
async def test_dependency_verifier_is_required_and_bound_before_call(tmp_path: Path) -> None:
    source, policy, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = _Attempt()

    # Exercise the narrow verifier contract directly: a boolean or a result
    # bound to another capture cannot authorize disclosure.
    class BadVerifier:
        def verify(self, **kwargs):  # type: ignore[no-untyped-def]
            return True

    with pytest.raises(IdleSummaryDependencyError):
        from nanobot.agent.idle_summary import _verify_dependencies

        _verify_dependencies(
            BadVerifier(),
            capture=capture,
            binding=binding,
            references=("artifact:test",),
        )
    assert policy.fingerprint == binding.policy_binding.fingerprint
    assert source.load_source_snapshot("test:idle-summary") == capture.snapshot
    assert attempt.requests == []


@pytest.mark.asyncio
async def test_dependency_verifier_cancellation_is_joined_before_propagation(tmp_path: Path) -> None:
    source, policy, _, binding, capture = _authority(
        tmp_path, [{"role": "user", "content": "x"}]
    )

    class WaitingVerifier:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def verify(self, **kwargs):  # type: ignore[no-untyped-def]
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
            return IdleDependencyVerification(
                capture,
                binding,
                ("artifact:test",),
                policy.fingerprint,
                binding.artifact_namespace,
            )

    verifier = WaitingVerifier()
    task = asyncio.create_task(
        _verify_dependencies_async(
            verifier,
            capture=capture,
            binding=binding,
            references=("artifact:test",),
            deadline=asyncio.get_running_loop().time() + 10,
            clock=asyncio.get_running_loop().time,
        )
    )
    await verifier.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    verifier.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.done() and verifier.release.is_set()


@pytest.mark.asyncio
async def test_deadline_cancellation_joins_attempt_and_discards_result(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = _Attempt()
    attempt.wait = True
    # Freeze the injected budget clock during synchronous preparation so this
    # test exercises an in-flight timeout, not pre-dispatch machine load.
    adapter = _adapter(
        attempt, source, registry, binding, _limits(max_wall_seconds=0.02), clock=lambda: 0.0,
    )
    task = asyncio.create_task(adapter.summarize(capture, binding, deadline=0.02))
    try:
        await asyncio.wait_for(attempt.cancelled.wait(), timeout=5)
        assert attempt.started.is_set() and not task.done()
    finally:
        attempt.release.set()
        await asyncio.gather(task, return_exceptions=True)
    result = task.result()
    assert result.status == "timeout" and result.proposal is None
    assert len(attempt.requests) == 1
    assert attempt.release.is_set()


@pytest.mark.asyncio
async def test_invalidation_before_transport_admission_has_no_dispatch_cost(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = _Attempt()
    adapter = _adapter(attempt, source, registry, binding)
    asyncio.get_running_loop().call_soon(registry.invalidate, binding)
    result = await adapter.summarize(
        capture, binding, deadline=asyncio.get_running_loop().time() + 2
    )
    assert result.status == "failed" and result.proposal is None
    assert attempt.requests == []
    assert result.accounting.dispatched_calls == 0
    assert result.accounting.unknown_dispatched == 0
    assert result.accounting.exact is False


@pytest.mark.asyncio
async def test_binding_invalidation_after_first_attempt_preserves_known_usage(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ],
    )

    class InvalidatingAttempt(_Attempt):
        async def attempt_once(self, request):  # type: ignore[no-untyped-def]
            response = await super().attempt_once(request)
            registry.invalidate(binding)
            return response

    attempt = InvalidatingAttempt(["first", "second"], input_limit=700)
    result = await _adapter(
        attempt,
        source,
        registry,
        binding,
        _limits(max_input_tokens=700, max_chunks=2),
    ).summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert result.proposal is None and result.status == "stale_binding"
    assert result.measured_usage is not None
    assert result.measured_usage.total_tokens == result.requests[0].input_tokens + 1
    assert result.accounting == IdleSummaryAttemptAccounting(
        1, result.requests[0].total_reservation_tokens, 1, result.measured_usage.total_tokens, 0, True
    )
    assert len(attempt.requests) == 1


@pytest.mark.asyncio
async def test_caller_cancellation_joins_suppressed_attempt(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    attempt = _Attempt()
    attempt.wait = True
    accounting: list[object] = []
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        accounting_sink=accounting.append,
    )
    task = asyncio.create_task(
        adapter.summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 10)
    )
    await attempt.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    attempt.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(attempt.requests) == 1
    request_reference = weakref.ref(attempt.requests[0])
    attempt.requests.clear()
    gc.collect()
    assert request_reference() is None
    assert adapter.last_accounting.dispatched_calls == 1
    assert adapter.last_accounting.unknown_dispatched == 1
    assert adapter.last_accounting.exact is False
    assert accounting and accounting[-1] == adapter.last_accounting


@pytest.mark.asyncio
async def test_synchronous_measurement_overrun_returns_timeout_result(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path, [{"role": "user", "content": "x"}]
    )
    now = [0.0]

    class OverrunningAttempt(_Attempt):
        def measure_request(self, messages, *, max_output_tokens, safety_buffer):  # type: ignore[no-untyped-def]
            receipt = super().measure_request(
                messages,
                max_output_tokens=max_output_tokens,
                safety_buffer=safety_buffer,
            )
            now[0] = 10.0
            return receipt

    attempt = OverrunningAttempt()
    adapter = _adapter(
        attempt,
        source,
        registry,
        binding,
        _limits(max_wall_seconds=5.0),
        clock=lambda: now[0],
    )
    result = await adapter.summarize(capture, binding, deadline=100.0)
    assert result.status == "timeout" and result.proposal is None
    assert result.requests == ()
    assert result.accounting == IdleSummaryAttemptAccounting(0, 0, 0, 0, 0, False)
    assert attempt.measurements and attempt.requests == []


@pytest.mark.asyncio
async def test_accounting_sink_marks_only_terminal_snapshot_exact(tmp_path: Path) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        ],
    )
    attempt = _Attempt(["first", "second"], input_limit=700)
    snapshots: list[IdleSummaryAttemptAccounting] = []
    result = await _adapter(
        attempt,
        source,
        registry,
        binding,
        _limits(max_input_tokens=700, max_chunks=2),
        accounting_sink=snapshots.append,
    ).summarize(capture, binding, deadline=asyncio.get_running_loop().time() + 2)
    assert result.status == "complete" and result.proposal is not None
    assert snapshots and snapshots[-1] == result.accounting
    assert result.accounting.exact is True
    assert all(snapshot.exact is False for snapshot in snapshots[:-1])


@pytest.mark.parametrize("failure_at", [None, "attempts", "provider_state", "request", "finish_reason", "tool_calls", "content"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit])
def test_hostile_response_access_order_precedes_exact_rejection(failure_at, error_type) -> None:
    order = ["attempts", "provider_state", "request", "finish_reason", "tool_calls", "content"]
    accessed = []
    failure = error_type("private accessor detail")

    class HostileResponse:
        def __getattribute__(self, name):
            accessed.append(name)
            if name == failure_at:
                raise failure
            return None

    expected_error = error_type if failure_at is not None and error_type is not RuntimeError else idle_summary_module.IdleSummaryError
    with pytest.raises(expected_error) as caught:
        idle_summary_module._validate_response_content(
            HostileResponse(), request=Mock(), transport=Mock(), limits=_limits()
        )
    expected_order = order if failure_at is None else order[:order.index(failure_at) + 1]
    assert accessed == expected_order
    if expected_error is idle_summary_module.IdleSummaryError:
        assert str(caught.value) == "summary attempt response is malformed"
        assert caught.value.__cause__ is None
    else:
        assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable", [False, True])
async def test_dependency_verifier_accepts_sync_and_awaitable_bound_results(tmp_path: Path, awaitable: bool) -> None:
    _, policy, _, binding, capture = _authority(tmp_path, [{"role": "user", "content": "x"}])
    references = ("artifact:test",)
    calls = []
    verification = IdleDependencyVerification(
        capture, binding, references, policy.fingerprint, binding.artifact_namespace
    )

    class Verifier:
        def verify(self, **kwargs):
            calls.append(kwargs)
            if awaitable:
                async def complete():
                    return verification
                return complete()
            return verification

    await _verify_dependencies_async(
        Verifier(), capture=capture, binding=binding, references=references,
        deadline=asyncio.get_running_loop().time() + 2,
        clock=asyncio.get_running_loop().time,
    )
    assert calls == [{"capture": capture, "binding": binding, "references": references}]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "expiry", "clock", "accounting", "sink", "admit", "invoke"])
async def test_claim_dispatch_phase_order_and_fault_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None,
) -> None:
    source, _, registry, binding, capture = _authority(
        tmp_path, [{"role": "user", "content": "x"}]
    )
    events: list[str] = []
    snapshots: list[IdleSummaryAttemptAccounting] = []
    references: list[weakref.ReferenceType[IdleSummaryRequest]] = []
    now = [1.0]
    claimed = [False]
    raw_calls = [0]
    accounting_failed = [False]

    class SynchronousEntryAttempt(_Attempt):
        def attempt_once(self, request: IdleSummaryRequest):
            raw_calls[0] += 1
            events.append("raw")
            return super().attempt_once(request)

    attempt = SynchronousEntryAttempt()
    transport_registry = IdleSummaryTransportRegistry(
        binding_registry=registry, source_store=source,
    )
    transport = transport_registry.issue(binding, attempt)
    original_claim = IdleSummaryTransportRegistry._claim_request
    original_admit = idle_summary_module.IdleSummaryQualifiedTransport._admit_request
    original_invoke = idle_summary_module.IdleSummaryQualifiedTransport._invoke_admitted
    original_publish = IdleSummaryAdapter._publish_accounting

    def claim(owner, capability, request: IdleSummaryRequest) -> None:
        original_claim(owner, capability, request)
        events.append("claim")
        references.append(weakref.ref(request))
        # Check consumption under the still-current binding, retaining no payload.
        temporary_request = references[-1]()
        assert temporary_request is request
        with pytest.raises(IdleSummaryCapabilityError, match="more than once"):
            original_claim(owner, capability, temporary_request)
        del temporary_request
        claimed[0] = True
        if fault == "expiry":
            now[0] = 11.0

    def clock() -> float:
        if claimed[0]:
            events.append("clock")
            if fault == "clock":
                raise RuntimeError("second deadline fault")
        return now[0]

    def observe(snapshot: IdleSummaryAttemptAccounting) -> None:
        snapshots.append(snapshot)
        if snapshot.dispatched_calls and "accounting" not in events:
            events.append("accounting")
            if fault == "sink":
                raise RuntimeError("dispatch sink fault")

    def publish(owner, ledger, *, terminal: bool = False):
        snapshot = original_publish(owner, ledger, terminal=terminal)
        if fault == "accounting" and snapshot.dispatched_calls and not accounting_failed[0]:
            accounting_failed[0] = True
            raise RuntimeError("on_started fault after dispatch accounting")
        return snapshot

    def admit(owner, request: IdleSummaryRequest) -> None:
        if fault == "admit":
            events.append("admit-fault")
            raise RuntimeError("admission fault")
        original_admit(owner, request)

    def invoke(owner, request: IdleSummaryRequest):
        if fault == "invoke":
            events.append("invoke-fault")
            raise RuntimeError("invocation fault")
        return original_invoke(owner, request)

    monkeypatch.setattr(IdleSummaryTransportRegistry, "_claim_request", claim)
    monkeypatch.setattr(idle_summary_module.IdleSummaryQualifiedTransport, "_admit_request", admit)
    monkeypatch.setattr(idle_summary_module.IdleSummaryQualifiedTransport, "_invoke_admitted", invoke)
    monkeypatch.setattr(IdleSummaryAdapter, "_publish_accounting", publish)
    adapter = IdleSummaryAdapter(
        transport, _limits(max_wall_seconds=20.0), request_measurer=attempt,
        binding_registry=registry, clock=clock, accounting_sink=observe,
    )
    result = await adapter.summarize(capture, binding, deadline=10.0)
    reservation = result.requests[0].total_reservation_tokens
    if fault in (None, "sink"):
        assert events[:4] == ["claim", "clock", "accounting", "raw"]
        assert raw_calls == [1]
        assert result.status == "complete" and result.proposal is not None
        assert result.accounting.exact is True and result.usage_exact is True
    else:
        assert raw_calls == [0] and attempt.requests == []
        assert result.status == ("timeout" if fault == "expiry" else "failed")
        assert result.proposal is None and result.measured_usage is None
        dispatched = int(fault in ("accounting", "invoke"))
        assert result.accounting == IdleSummaryAttemptAccounting(
            1, reservation, dispatched, 0, dispatched, False,
        )
        assert result.usage_exact is False
        assert result.usage == result.reserved_usage
        assert result.reserved_usage.calls == 1 and result.reserved_usage.tokens == reservation
        if fault == "admit":
            assert events == ["admit-fault"] and references == []
        elif fault in ("expiry", "clock"):
            assert events == ["claim", "clock"]
            assert all(snapshot.dispatched_calls == 0 for snapshot in snapshots)
        elif fault == "accounting":
            assert events == ["claim", "clock", "accounting"]
        else:
            assert events == ["claim", "clock", "accounting", "invoke-fault"]
    assert snapshots[-1] == adapter.last_accounting == result.accounting
    attempt.requests.clear()
    # Let wait_for's completed callback/exception frames leave the ready queue.
    settled = asyncio.Event()
    asyncio.get_running_loop().call_soon(settled.set)
    await settled.wait()
    gc.collect()
    assert all(reference() is None for reference in references)
    assert transport_registry._issued_requests == {}
