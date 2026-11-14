"""Regression tests for A3-6 AC6.4: extractor download bounds before buffering."""

from nanobot.agent.tools import web as web_module
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.config.schema import WebSearchConfig


class OversizeStreamResponse:
    status_code = 200
    headers: dict[str, str] = {}
    content_accessed = False

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        yield b"x" * 40
        yield b"x" * 40

    @property
    def content(self):
        OversizeStreamResponse.content_accessed = True
        return b"x" * 80

    @property
    def json(self):
        OversizeStreamResponse.content_accessed = True
        return {"data": {"title": "T", "content": "body"}}


class FakeStream:
    async def __aenter__(self):
        return OversizeStreamResponse()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeClient:
    stream_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, **kwargs):
        FakeClient.stream_calls += 1
        return FakeStream()

    async def get(self, url, **kwargs):
        raise AssertionError("buffered get must not be used")

    async def post(self, url, **kwargs):
        raise AssertionError("buffered post must not be used")


def _install_fake_http(monkeypatch):
    monkeypatch.setattr(web_module, "MAX_FETCH_DOWNLOAD_BYTES", 64)
    monkeypatch.setattr(web_module.httpx, "AsyncClient", FakeClient)
    FakeClient.stream_calls = 0
    OversizeStreamResponse.content_accessed = False


async def test_jina_refuses_oversize_body_before_buffering(monkeypatch):
    _install_fake_http(monkeypatch)
    tool = WebFetchTool()

    result = await tool._fetch_jina("https://example.com/page", max_chars=1000)

    assert result is None
    assert FakeClient.stream_calls == 1
    assert OversizeStreamResponse.content_accessed is False


async def test_tinyfish_fetch_refuses_oversize_body_before_buffering(monkeypatch):
    _install_fake_http(monkeypatch)
    tool = WebFetchTool(
        config=web_module.WebFetchConfig(provider="tinyfish"),
        tinyfish_api_key="key",
    )

    result = await tool._fetch_tinyfish("https://example.com/page", max_chars=1000)

    assert result is None
    assert FakeClient.stream_calls == 1
    assert OversizeStreamResponse.content_accessed is False


async def test_tinyfish_search_falls_back_on_oversize_body(monkeypatch):
    _install_fake_http(monkeypatch)
    tool = WebSearchTool(config=WebSearchConfig(provider="tinyfish", api_key="key"))

    async def fallback(self, query, n):
        return "ddg fallback"

    monkeypatch.setattr(WebSearchTool, "_search_duckduckgo", fallback)

    result = await tool.execute(query="q")

    assert "ddg fallback" in result
    assert OversizeStreamResponse.content_accessed is False
