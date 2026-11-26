"""Preserve native initialization without operating a desktop or a live runtime."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from kohaku.agent.tools import desktop_use as desktop

_ENV_KEY = "KOHAKU_DESKTOP_USE_MAX_INLINE_BYTES"
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
            "KOHAKU_DESKTOP_USE_MAX_INLINE_BYTES env)"
        )
    else:
        encoded = base64.b64encode(content).decode("ascii")
        assert (block, reason) == (
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            None,
        )
    assert tool._backend is None
