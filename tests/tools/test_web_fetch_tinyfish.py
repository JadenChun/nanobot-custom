"""Tests for the TinyFish-backed web_fetch path and its fallbacks."""

from __future__ import annotations

import json

import httpx
import pytest

from nanobot.agent.tools import web as web_module
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.web import WebFetchConfig, WebFetchTool
from nanobot.config.schema import Config, ToolsConfig


class _FakeClient:
    """Minimal httpx.AsyncClient replacement that records outbound calls."""

    calls: list[tuple[str, str]] = []
    post_status = 200
    post_payload: dict | None = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, **kwargs):
        _FakeClient.calls.append(("GET", url))
        payload = {"data": {"title": "Jina title", "content": "jina body", "url": url}}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    async def post(self, url, **kwargs):
        _FakeClient.calls.append(("POST", url))
        payload = _FakeClient.post_payload or {
            "results": [
                {
                    "url": url,
                    "final_url": url,
                    "title": "TinyFish title",
                    "format": "markdown",
                    "text": "# TinyFish title\n\nbody",
                }
            ],
            "errors": [],
        }
        return httpx.Response(
            _FakeClient.post_status, json=payload, request=httpx.Request("POST", url)
        )


@pytest.fixture
def fake_fetch(monkeypatch):
    _FakeClient.calls = []
    _FakeClient.post_status = 200
    _FakeClient.post_payload = None
    _FakeClient.jina_payload = None

    async def fake_stream(client, url, *, headers, blocked_hosts):
        class _Response:
            headers = {"content-type": "text/html"}

        return _Response(), None, None, False

    monkeypatch.setattr(web_module, "_stream_with_safe_redirects", fake_stream)
    monkeypatch.setattr(web_module.httpx, "AsyncClient", _FakeClient)
    yield _FakeClient


@pytest.fixture(autouse=True)
def _clear_key(monkeypatch):
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)


@pytest.mark.asyncio
async def test_tinyfish_fetch_used_when_key_present(fake_fetch):
    tool = WebFetchTool(config=WebFetchConfig(provider="auto"), tinyfish_api_key="tf-key")
    result = await tool.execute("https://example.com/article")
    payload = json.loads(result)
    assert payload["extractor"] == "tinyfish"
    assert "TinyFish title" in payload["text"]
    assert payload["untrusted"] is True
    assert ("POST", "https://api.fetch.tinyfish.ai") in fake_fetch.calls


@pytest.mark.asyncio
async def test_tinyfish_fetch_falls_back_to_jina_on_rate_limit(fake_fetch):
    fake_fetch.post_status = 429
    tool = WebFetchTool(config=WebFetchConfig(provider="auto"), tinyfish_api_key="tf-key")
    result = await tool.execute("https://example.com/article")
    payload = json.loads(result)
    assert payload["extractor"] == "jina"
    assert ("GET", "https://r.jina.ai/https://example.com/article") in fake_fetch.calls


@pytest.mark.asyncio
async def test_fetch_without_key_uses_jina(fake_fetch):
    tool = WebFetchTool(config=WebFetchConfig(provider="auto", use_jina_reader=True))
    result = await tool.execute("https://example.com/article")
    payload = json.loads(result)
    assert payload["extractor"] == "jina"
    assert all(method != "POST" for method, _url in fake_fetch.calls)


@pytest.mark.asyncio
async def test_non_tinyfish_search_secret_is_never_sent_to_tinyfish(fake_fetch, monkeypatch):
    tools = ToolsConfig()
    tools.web.search.provider = "brave"
    tools.web.search.api_key = "brave-search-sentinel"
    tools.web.fetch.use_jina_reader = False

    async def readability(self, *_args):
        return "local body"

    monkeypatch.setattr(WebFetchTool, "_fetch_readability", readability)

    tool = WebFetchTool.create(ToolContext(config=tools, workspace="."))

    assert await tool.execute("https://example.com/article") == "local body"
    assert ("POST", "https://api.fetch.tinyfish.ai") not in fake_fetch.calls


@pytest.mark.asyncio
async def test_local_provider_skips_remote_extractors(fake_fetch, monkeypatch):
    async def fake_readability(self, url, extract_mode, max_chars):
        return "local-body"

    monkeypatch.setattr(WebFetchTool, "_fetch_readability", fake_readability)
    tool = WebFetchTool(config=WebFetchConfig(provider="local"), tinyfish_api_key="tf-key")
    result = str(await tool.execute("https://example.com/article"))
    assert result == "local-body"
    assert all(method != "POST" for method, _url in fake_fetch.calls)
    assert all("r.jina.ai" not in url for _method, url in fake_fetch.calls)


@pytest.mark.asyncio
async def test_blocked_hosts_disable_remote_extractors(fake_fetch):
    tool = WebFetchTool(
        config=WebFetchConfig(provider="auto", blocked_hosts=["example.com"]),
        tinyfish_api_key="tf-key",
    )
    result = str(await tool.execute("https://example.com/article"))
    assert fake_fetch.calls == []
    assert "disabled" in result


@pytest.mark.asyncio
async def test_credential_url_is_never_sent_to_remote_extractors(fake_fetch):
    tool = WebFetchTool(config=WebFetchConfig(provider="tinyfish"), tinyfish_api_key="tf-key")
    result = str(await tool.execute("https://example.com/cb?access_token=secret"))
    assert all(method != "POST" for method, _url in fake_fetch.calls)
    assert all("r.jina.ai" not in url for _method, url in fake_fetch.calls)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_fetch_picks_up_key_saved_after_start(fake_fetch):
    state: dict[str, object] = {"key": ""}

    def loader():
        return WebFetchConfig(provider="auto"), state["key"]  # type: ignore[dict-item]

    tool = WebFetchTool(
        config=WebFetchConfig(provider="auto"),
        config_loader=loader,  # type: ignore[arg-type]
    )

    first = json.loads(await tool.execute("https://example.com/article"))
    assert first["extractor"] == "jina"

    state["key"] = "tf-key"
    second = json.loads(await tool.execute("https://example.com/article"))
    assert second["extractor"] == "tinyfish"


@pytest.mark.parametrize("provider", ["brave", "tavily", "jina", "  BRAVE  "])
def test_create_does_not_share_non_tinyfish_search_key(provider):
    tools = ToolsConfig()
    tools.web.search.provider = provider
    tools.web.search.api_key = "search-provider-sentinel"

    tool = WebFetchTool.create(ToolContext(config=tools, workspace="."))

    assert tool._tinyfish_fetch_key() == ""


def test_create_shares_search_key_only_for_normalized_tinyfish_provider():
    tools = ToolsConfig()
    tools.web.search.provider = " TinyFish "
    tools.web.search.api_key = "tinyfish-search-key"

    tool = WebFetchTool.create(ToolContext(config=tools, workspace="."))

    assert tool._tinyfish_fetch_key() == "tinyfish-search-key"


def test_explicit_fetch_key_and_environment_key_still_work(monkeypatch):
    explicit = WebFetchTool(config=WebFetchConfig(provider="tinyfish", api_key="fetch-only-key"))
    assert explicit._tinyfish_fetch_key() == "fetch-only-key"

    monkeypatch.setenv("TINYFISH_API_KEY", "environment-key")
    environmental = WebFetchTool(config=WebFetchConfig(provider="tinyfish"))
    assert environmental._tinyfish_fetch_key() == "environment-key"


def test_live_loader_clears_shared_key_when_search_provider_changes(monkeypatch):
    active = Config()
    active.tools.web.search.provider = "tinyfish"
    active.tools.web.search.api_key = "shared-tinyfish-key"
    active.tools.web.fetch.provider = "tinyfish"
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: active)
    monkeypatch.setattr("nanobot.config.loader.resolve_config_env_vars", lambda config: config)
    tools = ToolsConfig()
    tools.web.search.provider = "tinyfish"
    tools.web.search.api_key = "shared-tinyfish-key"
    tools.web.fetch.provider = "tinyfish"

    tool = WebFetchTool.create(ToolContext(
        config=tools,
        workspace=".",
        provider_snapshot_loader=lambda: None,
    ))
    assert tool._tinyfish_fetch_key() == "shared-tinyfish-key"

    active.tools.web.search.provider = "brave"
    active.tools.web.search.api_key = "search-provider-sentinel"
    tool._refresh_config()

    assert tool.config.provider == "tinyfish"
    assert tool._tinyfish_fetch_key() == ""
