"""Preserve native initialization without operating a desktop or a live runtime."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest

from nanobot.agent import memory_recovery_conversion_runtime as recovery
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery_manifest import (
    JsonValue,
    ManifestRecord,
    MemoryManifestError,
    MemoryRecoveryManifest,
    derive_manifest_identity,
)
from nanobot.agent.tools import desktop_use as desktop

_ENV_KEY = "NANOBOT_DESKTOP_USE_MAX_INLINE_BYTES"
_DEFAULT_LIMIT = 900 * 1024


def _fresh_desktop_module() -> ModuleType:
    """Execute the real module in an isolated namespace, never instantiate a backend."""
    assert desktop.__file__ is not None
    spec = importlib.util.spec_from_file_location("_desktop_initialization_probe", desktop.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, _DEFAULT_LIMIT),
        ("1", 1),
        ("0", _DEFAULT_LIMIT),
        ("-1", _DEFAULT_LIMIT),
        ("-999999999999999999", _DEFAULT_LIMIT),
        ("900000", 900000),
        ("  12  ", 12),
        ("+7", 7),
        ("1_024", 1024),
        ("１２３", 123),
        ("999999999999999999999999999999", 999999999999999999999999999999),
        ("", _DEFAULT_LIMIT),
        ("bad", _DEFAULT_LIMIT),
        ("1.5", _DEFAULT_LIMIT),
        ("1e3", _DEFAULT_LIMIT),
    ],
)
def test_fresh_limit_initialization(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int,
) -> None:
    if value is None:
        monkeypatch.delenv(_ENV_KEY, raising=False)
    else:
        monkeypatch.setenv(_ENV_KEY, value)
    module = _fresh_desktop_module()
    assert module._DEFAULT_MAX_INLINE_SCREENSHOT_BYTES == _DEFAULT_LIMIT
    assert type(module._MAX_INLINE_SCREENSHOT_BYTES) is int
    assert module._MAX_INLINE_SCREENSHOT_BYTES == expected


@pytest.mark.parametrize("value", [7, 0, -7, None])
def test_initialization_reads_environment_once_with_exact_default(
    monkeypatch: pytest.MonkeyPatch, value: object,
) -> None:
    trace: list[tuple[str, object]] = []

    def read(key: str, default: object = None) -> object:
        trace.append((key, default))
        return value

    monkeypatch.setattr(desktop.os.environ, "get", read)
    module = _fresh_desktop_module()
    assert trace == [(_ENV_KEY, _DEFAULT_LIMIT)]
    assert module._MAX_INLINE_SCREENSHOT_BYTES == (7 if value == 7 else _DEFAULT_LIMIT)


@pytest.mark.parametrize("result", [8, 0, -8])
def test_int_conversion_runs_once_before_fallback(
    monkeypatch: pytest.MonkeyPatch, result: int,
) -> None:
    trace: list[str] = []

    class NativeIntInput:
        def __int__(self) -> int:
            trace.append("int")
            return result

    def read(key: str, default: object = None) -> object:
        assert (key, default) == (_ENV_KEY, _DEFAULT_LIMIT)
        trace.append("get")
        return NativeIntInput()

    monkeypatch.setattr(desktop.os.environ, "get", read)
    module = _fresh_desktop_module()
    assert trace == ["get", "int"]
    assert module._MAX_INLINE_SCREENSHOT_BYTES == (result if result > 0 else _DEFAULT_LIMIT)


@pytest.mark.parametrize("error_type", [TypeError, ValueError, OverflowError, RuntimeError])
def test_initialization_retains_exact_exception_boundary(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception],
) -> None:
    error = error_type("conversion probe")
    trace: list[str] = []

    class NativeIntInput:
        def __int__(self) -> int:
            trace.append("int")
            raise error

    def read(key: str, default: object = None) -> object:
        assert (key, default) == (_ENV_KEY, _DEFAULT_LIMIT)
        trace.append("get")
        return NativeIntInput()

    monkeypatch.setattr(desktop.os.environ, "get", read)
    if error_type in (TypeError, ValueError):
        module = _fresh_desktop_module()
        assert module._MAX_INLINE_SCREENSHOT_BYTES == _DEFAULT_LIMIT
    else:
        with pytest.raises(error_type) as caught:
            _fresh_desktop_module()
        assert caught.value is error
    assert trace == ["get", "int"]


@pytest.mark.parametrize("size", [0, 3, 4, 5])
def test_inline_payload_uses_published_limit_at_exact_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, size: int,
) -> None:
    monkeypatch.setenv(_ENV_KEY, "4")
    module = _fresh_desktop_module()
    tool = module.DesktopUseTool()
    content = b"x" * size
    image = tmp_path / "probe.png"
    image.write_bytes(content)
    block, reason = tool._inline_image(str(image))
    if size == 0:
        assert (block, reason) == (None, "image file empty")
    elif size > 4:
        assert block is None
        assert reason == (
            "image 5 bytes exceeds inline cap 4 bytes (override with "
            "NANOBOT_DESKTOP_USE_MAX_INLINE_BYTES env)"
        )
    else:
        encoded = base64.b64encode(content).decode("ascii")
        assert (block, reason) == (
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            None,
        )
    assert tool._backend is None


@pytest.fixture
def manifest(
    tmp_path: Path,
) -> Generator[tuple[MemoryRecoveryManifest, dict[str, JsonValue]], None, None]:
    (tmp_path / "memory").mkdir()
    payload: dict[str, JsonValue] = {"probe": ["unicode ✓", True, None, 2, {"nested": "value"}]}
    identity = derive_manifest_identity("conversions", payload)
    with storage.open_workspace_memory(tmp_path) as authority:
        chain = MemoryRecoveryManifest(authority.open_evidence_namespace("conversions", identity))
        yield chain, payload


@pytest.mark.parametrize("preexisting", [False, True])
def test_captured_helper_with_genuine_manifest_and_detached_json(
    manifest: tuple[MemoryRecoveryManifest, dict[str, JsonValue]], preexisting: bool,
) -> None:
    chain, payload = manifest
    previous = chain.create(payload) if preexisting else None
    returned = recovery._loaded_captured(chain, payload)
    assert returned.phase == "CAPTURED"
    assert returned.payload == payload
    assert returned.payload is not payload
    assert chain.load() == (returned,)
    if previous is not None:
        assert returned.raw_bytes == previous.raw_bytes
        assert returned.digest == previous.digest
    detached = returned.payload
    detached["probe"] = "changed"
    assert returned.payload == payload
    assert chain.load() == (returned,)


class _ScriptedManifest(MemoryRecoveryManifest):
    """Nominal stage-unit double; it does not issue or qualify storage authority."""

    def __init__(
        self,
        reads: list[tuple[ManifestRecord, ...] | BaseException],
        created: ManifestRecord | BaseException,
    ) -> None:
        self.reads = list(reads)
        self.created = created
        self.trace: list[tuple[str, object]] = []

    def load(self) -> tuple[ManifestRecord, ...]:
        self.trace.append(("load", None))
        result = self.reads.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def create(self, captured_payload: dict[str, JsonValue]) -> ManifestRecord:
        self.trace.append(("create", captured_payload))
        if isinstance(self.created, BaseException):
            raise self.created
        return self.created


@pytest.mark.parametrize("preexisting", [False, True])
def test_helper_preserves_payload_return_identity_and_read_create_read_order(
    manifest: tuple[MemoryRecoveryManifest, dict[str, JsonValue]], preexisting: bool,
) -> None:
    chain, payload = manifest
    record = chain.create(payload)
    initial = (record,) if preexisting else ()
    scripted = _ScriptedManifest([initial, (record,)], record)
    assert recovery._loaded_captured(scripted, payload) is record
    assert scripted.trace == [("load", None), ("create", payload), ("load", None)]
    assert scripted.trace[1][1] is payload
    assert scripted.reads == []


@pytest.mark.parametrize("wrong_phase", [False, True])
def test_initial_chain_refusal_precedes_create(
    manifest: tuple[MemoryRecoveryManifest, dict[str, JsonValue]], wrong_phase: bool,
) -> None:
    chain, payload = manifest
    captured = chain.create(payload)
    prepared = chain.append(
        predecessor_digest=captured.digest, phase="CONVERSION_PREPARED", payload={"step": 1},
    )
    initial = (prepared,) if wrong_phase else (captured, prepared)
    scripted = _ScriptedManifest([initial], captured)
    with pytest.raises(recovery.ConversionPreparationError):
        recovery._loaded_captured(scripted, payload)
    assert scripted.trace == [("load", None)]
    assert scripted.reads == []


@pytest.mark.parametrize("extra_record", [False, True])
def test_reload_mismatch_refuses_after_create(
    manifest: tuple[MemoryRecoveryManifest, dict[str, JsonValue]], extra_record: bool,
) -> None:
    chain, payload = manifest
    record = chain.create(payload)
    final = (record, record) if extra_record else ()
    scripted = _ScriptedManifest([(), final], record)
    with pytest.raises(recovery.ConversionPreparationError):
        recovery._loaded_captured(scripted, payload)
    assert scripted.trace == [("load", None), ("create", payload), ("load", None)]
    assert scripted.trace[1][1] is payload
    assert scripted.reads == []


@pytest.mark.parametrize("phase", ["initial", "create", "reload"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_helper_propagates_exact_failure_without_extra_phases(
    manifest: tuple[MemoryRecoveryManifest, dict[str, JsonValue]],
    phase: str,
    error_type: type[BaseException],
) -> None:
    chain, payload = manifest
    record = chain.create(payload)
    error = error_type("phase probe")
    reads: list[tuple[ManifestRecord, ...] | BaseException] = [(), (record,)]
    created: ManifestRecord | BaseException = record
    if phase == "initial":
        reads = [error]
    elif phase == "create":
        created = error
    else:
        reads[1] = error
    scripted = _ScriptedManifest(reads, created)
    with pytest.raises(error_type) as caught:
        recovery._loaded_captured(scripted, payload)
    assert caught.value is error
    expected = [("load", None)]
    if phase != "initial":
        expected.append(("create", payload))
    if phase == "reload":
        expected.append(("load", None))
    assert scripted.trace == expected


@pytest.mark.parametrize("invalid", [1.5, b"binary", ("tuple",)])
def test_helper_keeps_native_manifest_validation_for_invalid_json(
    manifest: tuple[MemoryRecoveryManifest, dict[str, JsonValue]], invalid: object,
) -> None:
    chain, _payload = manifest
    with pytest.raises(MemoryManifestError) as caught:
        recovery._loaded_captured(chain, {"invalid": invalid})
    assert caught.value.code == "invalid"
    assert chain.load() == ()
