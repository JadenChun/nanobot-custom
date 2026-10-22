"""CoreGraphics/AppKit regression traces using only inert Python module fakes."""

import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nanobot.agent.tools import desktop_use as desktop


class Bridge:
    def __init__(self) -> None:
        self.quartz = ModuleType("Quartz")
        self.appkit = ModuleType("AppKit")
        self.trace: list[tuple[str, tuple[object, ...]]] = []
        self.results: dict[str, object] = {}
        self.screen: object = SimpleNamespace(
            frame=lambda: SimpleNamespace(size=SimpleNamespace(width=1920.9, height=1080.5))
        )
        constants = (
            "kCGEventFlagMaskAlternate kCGEventFlagMaskCommand kCGEventFlagMaskControl "
            "kCGEventFlagMaskSecondaryFn kCGEventFlagMaskShift kCGEventLeftMouseDown "
            "kCGEventLeftMouseDragged kCGEventLeftMouseUp kCGEventMouseMoved "
            "kCGEventOtherMouseDown kCGEventOtherMouseUp kCGEventRightMouseDown "
            "kCGEventRightMouseUp kCGHIDEventTap kCGImageAlphaPremultipliedLast "
            "kCGMouseButtonCenter kCGMouseButtonLeft kCGMouseButtonRight "
            "kCGMouseEventClickState kCGNullWindowID kCGScrollEventUnitLine "
            "kCGWindowImageDefault kCGWindowListOptionOnScreenOnly"
        ).split()
        for index, name in enumerate(constants):
            setattr(self.quartz, name, 1 << index)
        setattr(self.quartz, "CGRectInfinite", "infinite")
        functions = (
            "CGEventCreate CGEventGetLocation CGWindowListCreateImage "
            "CGImageDestinationCreateWithURL CGImageDestinationAddImage "
            "CGImageDestinationFinalize CGImageSourceCreateWithURL CGImageSourceGetCount "
            "CGImageSourceCreateImageAtIndex CGImageGetColorSpace CGBitmapContextCreate "
            "CGRectMake CGContextDrawImage CGBitmapContextCreateImage CGEventCreateMouseEvent "
            "CGEventPost CGEventSetIntegerValueField CGEventCreateScrollWheelEvent "
            "CGEventCreateKeyboardEvent CGEventKeyboardSetUnicodeString CGEventSetFlags"
        ).split()
        for name in functions:
            setattr(self.quartz, name, self.function(name))
        setattr(self.appkit, "NSScreen", SimpleNamespace(mainScreen=lambda: self.screen))
        setattr(self.appkit, "NSURL", SimpleNamespace(fileURLWithPath_=self.function("url")))
        self.results.update(CGEventGetLocation=SimpleNamespace(x=10.9, y=20.8),
                            CGImageSourceGetCount=1, CGImageDestinationFinalize=True)

    def function(self, name: str):
        def call(*args: object) -> object:
            self.trace.append((name, args))
            if name == "url":
                return args[0]
            if name == "CGImageDestinationFinalize" and self.results.get(name, True):
                destinations = [a for n, a in self.trace if n == "CGImageDestinationCreateWithURL"]
                Path(str(destinations[-1][0])).write_bytes(b"fake PNG")
            return self.results.get(name, (name, args))
        return call

    def calls(self, name: str) -> list[tuple[object, ...]]:
        return [args for event, args in self.trace if event == name]

    def constant(self, name: str) -> int:
        value = getattr(self.quartz, name)
        assert isinstance(value, int)
        return value


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> Bridge:
    fake = Bridge()
    monkeypatch.setitem(sys.modules, "Quartz", fake.quartz)
    monkeypatch.setitem(sys.modules, "AppKit", fake.appkit)
    original_import = builtins.__import__

    def safe_import(name: str, *args: object, **kwargs: object):
        if name in {"Quartz", "AppKit"}:
            assert sys.modules[name] is getattr(fake, "quartz" if name == "Quartz" else "appkit")
            return sys.modules[name]
        assert name.split(".")[0] not in {"objc", "Foundation"}, "real SDK import attempted"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", safe_import)
    monkeypatch.setattr(desktop.time, "sleep", fake.function("sleep"))
    return fake


def test_lazy_backend_and_partial_module(bridge: Bridge, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop.platform, "system", lambda: "Darwin")
    tool = desktop.DesktopUseTool()
    assert tool._backend is None and bridge.trace == []
    backend = tool._get_backend()
    assert isinstance(backend, desktop._MacBackend)
    assert bridge.trace == []
    assert tool._get_backend() is backend
    delattr(bridge.quartz, "CGEventGetLocation")
    partial = desktop._MacBackend()  # No eager member admission/validation.
    with pytest.raises(AttributeError, match="CGEventGetLocation"):
        partial.cursor_position()


@pytest.mark.parametrize("system", ["Linux", "Windows", "Plan9"])
def test_platform_rejection_cached(bridge: Bridge, monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    monkeypatch.setattr(desktop.platform, "system", lambda: system)
    tool = desktop.DesktopUseTool()
    assert tool._get_backend() is None
    error = tool._backend_error
    monkeypatch.setattr(desktop.platform, "system", lambda: "Darwin")
    assert tool._get_backend() is None and tool._backend_error == error
    assert bridge.trace == []


def test_import_error_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "AppKit", None)
    monkeypatch.setitem(sys.modules, "Quartz", None)
    monkeypatch.setattr(desktop.platform, "system", lambda: "Darwin")
    tool = desktop.DesktopUseTool()
    assert tool._get_backend() is None
    assert "requires PyObjC" in str(tool._backend_error)
    assert tool._get_backend() is None


def test_screen_and_cursor(bridge: Bridge) -> None:
    backend = desktop._MacBackend()
    assert backend.info() == {"platform": "darwin", "screen": {"width": 1920, "height": 1080},
                              "cursor": {"x": 10, "y": 20}}
    assert bridge.calls("CGEventCreate") == [(None,)]
    bridge.screen = None
    with pytest.raises(desktop._DesktopError, match="no main screen"):
        backend.screen_size()


@pytest.mark.parametrize("function,result,message", [
    ("CGWindowListCreateImage", None, "Screen Recording"),
    ("CGImageDestinationCreateWithURL", None, "image destination"),
    ("CGImageDestinationFinalize", False, "write screenshot"),
])
def test_screenshot_errors(bridge: Bridge, tmp_path: Path, function: str, result: object,
                           message: str) -> None:
    bridge.results[function] = result
    with pytest.raises(desktop._DesktopError, match=message):
        desktop._MacBackend().screenshot(str(tmp_path / "image.png"))


@pytest.mark.parametrize("function,result,message", [
    ("CGImageSourceCreateWithURL", None, "cannot read PNG"),
    ("CGImageSourceGetCount", 0, "cannot read PNG"),
    ("CGImageSourceCreateImageAtIndex", None, "cannot decode PNG"),
    ("CGBitmapContextCreate", None, "bitmap context"),
    ("CGImageDestinationFinalize", False, "write resized PNG"),
])
def test_resize_errors(bridge: Bridge, tmp_path: Path, function: str, result: object,
                       message: str) -> None:
    bridge.results[function] = result
    with pytest.raises(desktop._DesktopError, match=message):
        desktop._MacBackend().resize_png(str(tmp_path / "image.png"), 100, 200)


def test_screenshot_resize_optional_handles(bridge: Bridge, tmp_path: Path) -> None:
    backend = desktop._MacBackend()
    path = tmp_path / "image.png"
    backend.screenshot(str(path))
    assert path.read_bytes() == b"fake PNG"
    bridge.trace.clear()
    # The original resize path passes an unchecked optional destination through.
    bridge.results["CGImageDestinationCreateWithURL"] = None
    backend.resize_png(str(path), 100, 200)
    assert bridge.calls("CGImageDestinationAddImage")[0][0] is None
    assert bridge.calls("CGImageDestinationFinalize") == [(None,)]
    assert bridge.calls("CGRectMake") == [(0, 0, 100, 200)]
    assert bridge.calls("CGBitmapContextCreate")[0][:5] == (None, 100, 200, 8, 0)


@pytest.mark.parametrize("button", ["left", "right", "middle"])
def test_click_trace(bridge: Bridge, button: str) -> None:
    desktop._MacBackend().mouse_click(30, 40, button, 2)
    names = [name for name, _ in bridge.trace]
    unit = ["CGEventCreateMouseEvent", "CGEventCreateMouseEvent",
            "CGEventSetIntegerValueField", "CGEventSetIntegerValueField", "CGEventPost", "CGEventPost"]
    assert names == unit + ["sleep"] + unit
    assert [args[-1] for args in bridge.calls("CGEventSetIntegerValueField")] == [1, 1, 2, 2]
    assert bridge.calls("sleep") == [(0.05,)]
    assert all(args[2] == (30, 40) for args in bridge.calls("CGEventCreateMouseEvent"))
    with pytest.raises(desktop._DesktopError, match="unknown mouse button"):
        desktop._MacBackend().mouse_click(0, 0, "invalid", 1)


def test_move_and_drag_traces(bridge: Bridge) -> None:
    backend = desktop._MacBackend()
    backend.mouse_move(10, 20)
    assert len(bridge.calls("CGEventPost")) == 1 and not bridge.calls("sleep")
    bridge.trace.clear()
    backend.mouse_move(50, 20)
    assert [a[2] for a in bridge.calls("CGEventCreateMouseEvent")] == [
        (18.0, 20.0), (26.0, 20.0), (34.0, 20.0), (42.0, 20.0), (50.0, 20.0)]
    assert bridge.calls("sleep") == [(0.015,)] * 5
    bridge.trace.clear()
    backend.mouse_drag(0, 0, 100, 200)
    mouse = bridge.calls("CGEventCreateMouseEvent")
    assert len(mouse) == 22
    assert mouse[0][2] == (0, 0) and mouse[-1][2] == (100, 200)
    assert [a[2] for a in mouse[1:-1]] == [(5.0*i, 10.0*i) for i in range(1, 21)]
    assert bridge.calls("sleep") == [(0.05,)] + [(0.01,)] * 20
    assert [n for n, _ in bridge.trace] == ["CGEventCreateMouseEvent", "CGEventPost", "sleep"] * 21 + ["CGEventCreateMouseEvent", "CGEventPost"]


@pytest.mark.parametrize("direction,axes", [("up", (6, 0)), ("down", (-6, 0)),
                                             ("left", (0, -6)), ("right", (0, 6)),
                                             ("unknown", (0, 0))])
def test_scroll_trace(bridge: Bridge, direction: str, axes: tuple[int, int]) -> None:
    desktop._MacBackend().scroll(direction, 2)
    assert bridge.calls("CGEventCreateScrollWheelEvent") == [
        (None, bridge.constant("kCGScrollEventUnitLine"), 2, *axes)]
    assert [n for n, _ in bridge.trace] == ["CGEventCreateScrollWheelEvent", "CGEventPost"]


@pytest.mark.parametrize("delays,expected", [(None, [(0.012,)]*3), ([5], [(0.005,), (0.012,), (0.012,)]),
                                            ([0, 20, -1, 99], [(0.02,)])])
def test_unicode_and_delays(bridge: Bridge, delays: list[int] | None,
                            expected: list[tuple[float]]) -> None:
    desktop._MacBackend().type_text("a😀b", 12, delays)
    assert [a[1:] for a in bridge.calls("CGEventKeyboardSetUnicodeString")] == [
        (1, "a"), (1, "a"), (2, "😀"), (2, "😀"), (1, "b"), (1, "b")]
    assert bridge.calls("sleep") == expected
    assert len(bridge.calls("CGEventPost")) == 6


@pytest.mark.parametrize("hold", [False, True])
def test_chord_order(bridge: Bridge, hold: bool) -> None:
    backend = desktop._MacBackend()
    if hold:
        backend.hold_key("Cmd+Shift+S", 0.25)
    else:
        backend.key_chord("Cmd+Shift+S")
    posted = [a[1] for a in bridge.calls("CGEventPost")]
    assert posted == [("CGEventCreateKeyboardEvent", (None, code, down)) for code, down in
                      [(55, True), (56, True), (1, True), (1, False), (56, False), (55, False)]]
    flags = bridge.constant("kCGEventFlagMaskCommand") | bridge.constant("kCGEventFlagMaskShift")
    assert [a[1] for a in bridge.calls("CGEventSetFlags")] == [flags, flags]
    assert bridge.calls("sleep") == ([(0.25,)] if hold else [])
    bridge.trace.clear()
    if hold:
        backend.hold_key("Cmd+Option", 0.5)
    else:
        backend.key_chord("Cmd+Option")
    assert [a[1:] for a in bridge.calls("CGEventCreateKeyboardEvent")] == [(58, True), (58, False)]
    assert not bridge.calls("CGEventSetFlags")
    assert bridge.calls("sleep") == ([(0.5,)] if hold else [])


@pytest.mark.parametrize("chord,message", [("", "empty key chord"), ("a+b", "multiple non-modifier"),
                                          ("NotAKey", "unknown key")])
def test_invalid_chords(bridge: Bridge, chord: str, message: str) -> None:
    with pytest.raises(desktop._DesktopError, match=message):
        desktop._MacBackend().key_chord(chord)
    assert bridge.trace == []


@pytest.mark.parametrize("value,pair", [([True, False], (1, 0)), ((1.9, -2.9), (1, -2)),
                                       ([3, 4], (3, 4))])
def test_coordinates(value: object, pair: tuple[int, int]) -> None:
    assert desktop.DesktopUseTool._require_coord(value) == pair


@pytest.mark.parametrize("value", [None, "12", [1], [1, 2, 3], [1, "2"], (object(), 2)])
def test_invalid_coordinates(value: object) -> None:
    with pytest.raises(desktop._DesktopError, match="two numbers"):
        desktop.DesktopUseTool._require_coord(value)


async def test_missing_action_unchanged(bridge: Bridge) -> None:
    tool = desktop.DesktopUseTool()
    with pytest.raises(TypeError, match="action"):
        await tool.execute()
    assert bridge.trace == []
