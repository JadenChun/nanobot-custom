"""Pure SDK/JSON container preservation probes; no browser or socket operation."""

import asyncio
import json
import signal
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.crawler.worker import _VIEWPORT_HTML_JS, Crawl4AIWorker, WorkerRequestError

_VIEWPORT = '<section data-crawl4ai-viewport="true">visible</section>'


def _result(**kwargs):
    return SimpleNamespace(cleaned_html="<main>body</main>", url="https://example.com/",
                           status_code=200, screenshot=None, **kwargs)


@pytest.mark.parametrize("method, empty", [
    ("_viewport_html", ""), ("_js_string_result", None),
])
@pytest.mark.parametrize("execution", [None, False, 0, "text", [], (), object()])
def test_nonmapping_execution_is_not_interpreted(method, empty, execution):
    result = _result(js_execution_result=execution)
    assert getattr(Crawl4AIWorker, method)(result) == empty


@pytest.mark.parametrize("method, empty", [
    ("_viewport_html", ""), ("_js_string_result", None),
])
@pytest.mark.parametrize("values", [None, False, 0, "text", {}, (), object()])
def test_nonlist_execution_values_are_not_iterated(method, empty, values):
    result = _result(js_execution_result={"results": values})
    assert getattr(Crawl4AIWorker, method)(result) == empty


def test_mixed_values_retain_reverse_selection_and_bounded_strings():
    result = _result(js_execution_result={
        "results": [_VIEWPORT, "  earlier  ", None, _VIEWPORT + "later", {}, " x " * 5000],
    })
    assert Crawl4AIWorker._viewport_html(result) == _VIEWPORT + "later"
    assert Crawl4AIWorker._js_string_result(result) == (" x " * 5000).strip()[:4096]


@pytest.mark.parametrize("method", ["_viewport_html", "_js_string_result"])
@pytest.mark.parametrize("fault", [RuntimeError("ordinary"), KeyboardInterrupt("system")])
def test_execution_get_fault_is_preserved(method, fault):
    trace = []

    class Execution(dict):
        def get(self, key):
            trace.append(key)
            raise fault

    with pytest.raises(type(fault)) as caught:
        getattr(Crawl4AIWorker, method)(_result(js_execution_result=Execution()))
    assert caught.value is fault
    assert trace == ["results"]


@pytest.mark.parametrize("method", ["_viewport_html", "_js_string_result"])
def test_execution_list_subclass_keeps_reverse_hook(method):
    trace = []

    class Values(list):
        def __reversed__(self):
            trace.append("reverse")
            return iter([_VIEWPORT, " selected "])

    result = _result(js_execution_result={"results": Values()})
    expected = _VIEWPORT if method == "_viewport_html" else "selected"
    assert getattr(Crawl4AIWorker, method)(result) == expected
    assert trace == ["reverse"]


@pytest.mark.parametrize("media", [None, False, 0, "text", [], (), {}])
def test_nonmapping_media_retains_document_payload(media):
    payload = Crawl4AIWorker._html_payload(_result(media=media),
        session_id="id", offset=1, max_chars=5)
    assert payload["html"] == "main>"
    assert payload["html_offset"] == 1 and payload["next_html_offset"] == 6
    assert payload["html_total_chars"] == len("<main>body</main>")
    assert payload["content_scope"] == "rendered_document"
    assert payload["status_code"] == 200
    assert "screenshot_base64" not in payload


@pytest.mark.parametrize("images", [None, False, 0, "text", {}, (), object()])
def test_nonlist_media_images_do_not_generate_image_section(images):
    payload = Crawl4AIWorker._html_payload(_result(media={"images": images}),
        session_id="id", offset=0, max_chars=1000)
    assert payload["html"] == "<main>body</main>"


@pytest.mark.parametrize("image", [None, False, 0, "text", [], (), {},
    {"src": 1, "alt": [], "desc": False}])
def test_nonmapping_or_nontext_images_retain_empty_section(image):
    payload = Crawl4AIWorker._html_payload(_result(media={"images": [image]}),
        session_id="id", offset=0, max_chars=1000)
    assert payload["html"] == (
        '<section data-crawl4ai-media="images" aria-label="Extracted image metadata">'
        '\n</section>\n<main>body</main>'
    )


def test_live_media_reads_and_image_fields_keep_exact_order_and_escaping():
    trace = []

    class Image(dict):
        def get(self, key):
            trace.append(("image", key))
            return super().get(key)

    images = [Image(src=' <&" ', alt=" cat ", desc=" " + "x" * 801)]

    class Media(dict):
        def get(self, key):
            trace.append(("media", key))
            return images

    class Result:
        cleaned_html = "body"
        url = "https://example.com/"
        status_code = 200
        screenshot = None
        js_execution_result = None

        @property
        def media(self):
            trace.append("media_attribute")
            return Media()

    payload = Crawl4AIWorker._html_payload(Result(), session_id="id", offset=0, max_chars=2000)
    assert trace == ["media_attribute", "media_attribute", ("media", "images"),
                     ("media", "images"), ("image", "src"), ("image", "alt"), ("image", "desc")]
    assert 'src="&lt;&amp;&quot;"' in payload["html"]
    assert 'alt="cat"' in payload["html"]
    assert 'data-description="' + "x" * 800 + '"' in payload["html"]
    assert "x" * 801 not in payload["html"]


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("value", [None, False, 0, [], {}, "", "  ", "x" * 501, " .item "])
def test_selector_required_branch_matches_its_existing_contract(required, value):
    if value is None and not required:
        assert Crawl4AIWorker._selector(value, required=required) is None
    elif value == " .item ":
        assert Crawl4AIWorker._selector(value, required=required) == ".item"
    else:
        with pytest.raises(WorkerRequestError) as caught:
            Crawl4AIWorker._selector(value, required=required)
        assert str(caught.value) == "selector must be a non-empty CSS selector under 500 characters"


@pytest.mark.parametrize("raw, dispatched, error", [
    (b'{"op":"probe","nested":{"x":[null,false,3]}}\n', True, None),
    (b'{"0":"value"}\n', True, None),
    (b'[]\n', False, "request must be a JSON object"),
    (b'false\n', False, "request must be a JSON object"),
    (b'null\n', False, "request must be a JSON object"),
    (b'1\n', False, "request must be a JSON object"),
    (b'"text"\n', False, "request must be a JSON object"),
    (b'{invalid}\n', False, "request must be one JSON object"),
    (b'\xff\n', False, "request must be one JSON object"),
    (b'', False, "request is empty or too large"),
    (b'x' * (64 * 1024 + 1), False, "request is empty or too large"),
])
async def test_wire_default_decoder_guard_dispatch_and_cleanup(tmp_path, raw, dispatched, error):
    worker = Crawl4AIWorker(str(tmp_path / "unused.sock"), {"example.com"})
    dispatch = AsyncMock(return_value={"ok": True, "text": "猫"})
    worker.dispatch = dispatch
    reader = SimpleNamespace(readline=AsyncMock(return_value=raw))
    trace = []

    class Writer:
        def write(self, value):
            trace.append(("write", value))

        async def drain(self):
            trace.append("drain")

        def close(self):
            trace.append("close")

        async def wait_closed(self):
            trace.append("wait_closed")

    await worker.handle_connection(reader, Writer())
    reader.readline.assert_awaited_once_with()
    assert trace[1:] == ["drain", "close", "wait_closed"]
    assert trace[0][1].endswith(b"\n")
    response = json.loads(trace[0][1])
    if dispatched:
        dispatch.assert_awaited_once_with(json.loads(raw))
        assert response == {"ok": True, "text": "猫"}
        assert "猫".encode() in trace[0][1]
    else:
        dispatch.assert_not_awaited()
        assert response == {"ok": False, "error": error}


@pytest.mark.parametrize("profile_kind", [None, "directory", "missing", "file"])
def test_startup_probe_only_checks_dependency_and_profile(tmp_path, monkeypatch, profile_kind):
    sdk = ModuleType("crawl4ai")
    launch = MagicMock(side_effect=AssertionError("the dependency probe must not launch a browser"))
    sdk.AsyncWebCrawler = launch
    sdk.BrowserConfig = launch
    monkeypatch.setitem(sys.modules, "crawl4ai", sdk)
    profile = tmp_path / "profile"
    if profile_kind == "directory":
        profile.mkdir()
    elif profile_kind == "file":
        profile.write_text("not a directory", encoding="utf-8")
    worker = Crawl4AIWorker(
        str(tmp_path / "unused.sock"), {"example.com"},
        user_data_dir=None if profile_kind is None else str(profile),
    )

    if profile_kind in ("missing", "file"):
        with pytest.raises(RuntimeError) as caught:
            worker._validate_browser_setup()
        assert str(caught.value) == f"Crawl4AI browser profile was not found: {profile}"
    else:
        assert worker._validate_browser_setup() is None
    launch.assert_not_called()
    assert worker._crawler is None
    assert worker.sessions == {}


def test_startup_probe_missing_dependency_keeps_original_error_and_cause(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "crawl4ai", None)
    worker = Crawl4AIWorker(
        str(tmp_path / "unused.sock"), {"example.com"},
        user_data_dir=str(tmp_path / "also-missing-profile"),
    )

    with pytest.raises(RuntimeError) as caught:
        worker._validate_browser_setup()

    assert str(caught.value) == "crawl4ai is not installed; install nanobot-ai[crawler-worker]"
    assert isinstance(caught.value.__cause__, ImportError)
    assert worker._crawler is None
    assert worker.sessions == {}


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("transport", ["tcp", "unix"])
@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM"])
async def test_serve_signal_paths_preserve_scheduling_and_cleanup(
    tmp_path, monkeypatch, fallback, transport, signal_name,
):
    import nanobot.crawler.worker as worker_module

    worker = Crawl4AIWorker(
        str(tmp_path / "unused.sock"), {"example.com"},
        tcp_host="127.0.0.1" if transport == "tcp" else None,
    )
    trace = []
    installed = {}
    fallback_handlers = {}
    scheduled = []
    returned = []
    real_loop = asyncio.get_running_loop()

    class Loop:
        def add_signal_handler(self, signum, callback):
            trace.append(("register", signum))
            if fallback:
                raise NotImplementedError("synthetic Windows event loop")
            installed[signum] = callback

        def call_soon_threadsafe(self, callback):
            handle = real_loop.call_soon_threadsafe(callback)
            scheduled.append((callback, handle))
            return handle

    def register_fallback(signum, callback):
        trace.append(("fallback", signum))
        fallback_handlers[signum] = callback

    class Server:
        async def __aenter__(self):
            trace.append("enter")
            signum = getattr(signal, signal_name)
            if fallback:
                returned.append(fallback_handlers[signum](signum, None, "ignored extra argument"))
            else:
                installed[signum]()
            await asyncio.sleep(0)
            return self

        async def __aexit__(self, *_args):
            trace.append("exit")

        def close(self):
            trace.append("close")

        async def wait_closed(self):
            trace.append("wait_closed")

    server = Server()
    tcp_bind = AsyncMock(return_value=server)
    unix_bind = AsyncMock(return_value=server)
    chmod = MagicMock()
    worker.start = AsyncMock(side_effect=lambda: trace.append("start"))
    worker.stop = AsyncMock(side_effect=lambda: trace.append("stop"))
    monkeypatch.setattr(worker_module, "asyncio", SimpleNamespace(
        start_server=tcp_bind, start_unix_server=unix_bind, Event=asyncio.Event,
        Handle=asyncio.Handle, get_running_loop=lambda: Loop(),
    ))
    monkeypatch.setattr(worker_module, "signal", SimpleNamespace(
        SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM, signal=register_fallback,
    ))
    monkeypatch.setattr(worker_module, "os", SimpleNamespace(chmod=chmod))

    await asyncio.wait_for(worker.serve(), timeout=1)

    worker.start.assert_awaited_once_with()
    worker.stop.assert_awaited_once_with()
    assert trace[0] == "start"
    assert trace[-5:] == ["enter", "exit", "close", "wait_closed", "stop"]
    assert [signum for kind, signum in trace[1:-5] if kind == "register"] == [
        signal.SIGINT, signal.SIGTERM,
    ]
    if fallback:
        assert list(fallback_handlers) == [signal.SIGINT, signal.SIGTERM]
        assert installed == {}
        assert len(scheduled) == len(returned) == 1
        assert returned[0] is scheduled[0][1]
        assert isinstance(returned[0], asyncio.Handle)
        assert scheduled[0][0].__self__.is_set()
    else:
        assert list(installed) == [signal.SIGINT, signal.SIGTERM]
        assert fallback_handlers == {}
        assert scheduled == returned == []
        assert installed[getattr(signal, signal_name)].__self__.is_set()
    if transport == "tcp":
        tcp_bind.assert_awaited_once_with(worker.handle_connection, host="127.0.0.1", port=worker.tcp_port)
        unix_bind.assert_not_awaited()
        chmod.assert_not_called()
    else:
        unix_bind.assert_awaited_once_with(worker.handle_connection, path=worker.socket_path)
        tcp_bind.assert_not_awaited()
        chmod.assert_called_once_with(worker.socket_path, 0o660)


@pytest.mark.parametrize("selector", [None, "", "article", "  [data-title='猫'] > span  "])
@pytest.mark.parametrize("js_code", [None, "return 'custom';"])
@pytest.mark.parametrize("terminal", ["ok", "sdk_error", "timeout", "cancelled"])
async def test_crawl_selector_options_preserve_sdk_defaults_and_operation_outcome(
    tmp_path, monkeypatch, selector, js_code, terminal,
):
    sdk = ModuleType("crawl4ai")
    bypass = object()
    sdk.CacheMode = SimpleNamespace(BYPASS=bypass)
    constructor = MagicMock(side_effect=lambda **options: SimpleNamespace(
        **({"css_selector": None} | options),
    ))
    sdk.CrawlerRunConfig = constructor
    monkeypatch.setitem(sys.modules, "crawl4ai", sdk)
    result = SimpleNamespace(success=terminal != "sdk_error", error_message="x" * 500)
    operation = AsyncMock(return_value=result)
    cancellation = asyncio.CancelledError("crawl cancelled")
    if terminal == "timeout":
        operation.side_effect = asyncio.TimeoutError("crawl timeout")
    elif terminal == "cancelled":
        operation.side_effect = cancellation
    worker = Crawl4AIWorker(str(tmp_path / "unused.sock"), {"example.com"})
    worker._crawler = SimpleNamespace(arun=operation)
    worker._kill_session = AsyncMock()
    arguments = dict(
        url="https://example.com/article", session_id="session-selector",
        js_only=True, selector=selector, js_code=js_code, wait_ms=250,
        timeout_seconds=7, screenshot=True,
    )

    if terminal == "ok":
        assert await worker._crawl_result(**arguments) is result
    elif terminal == "cancelled":
        with pytest.raises(asyncio.CancelledError) as caught:
            await worker._crawl_result(**arguments)
        assert caught.value is cancellation
    else:
        with pytest.raises(WorkerRequestError) as caught:
            await worker._crawl_result(**arguments)
        assert str(caught.value) == (
            "browser operation timed out" if terminal == "timeout" else "x" * 400
        )

    constructor.assert_called_once()
    options = constructor.call_args.kwargs
    assert ("css_selector" in options) is (selector is not None)
    expected = {
        "cache_mode": bypass,
        "session_id": "session-selector",
        "js_only": True,
        "js_code": [js_code, _VIEWPORT_HTML_JS] if js_code else [_VIEWPORT_HTML_JS],
        "page_timeout": 7000,
        "delay_before_return_html": 0.25,
        "wait_until": "domcontentloaded",
        "wait_for_images": True,
        "screenshot": True,
        "force_viewport_screenshot": True,
        "image_score_threshold": 0,
        "image_description_min_word_threshold": 1,
        "excluded_tags": ["script", "style", "noscript", "svg"],
        "keep_attrs": [
            "id", "class", "role", "aria-label", "href", "src", "alt", "title",
            "poster", "name", "type", "value",
        ],
        "keep_data_attributes": True,
        "remove_overlay_elements": False,
        "verbose": False,
    }
    if selector is not None:
        expected["css_selector"] = selector
    assert options == expected
    config = operation.await_args.kwargs["config"]
    assert config.css_selector == selector
    operation.assert_awaited_once_with(url=arguments["url"], config=config)
    if terminal in ("timeout", "cancelled"):
        worker._kill_session.assert_awaited_once_with("session-selector")
    else:
        worker._kill_session.assert_not_awaited()
    assert worker.sessions == {}
    assert arguments["selector"] == selector
