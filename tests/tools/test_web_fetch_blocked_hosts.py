"""Configured web routing must stop before DNS, transport, or remote extraction."""

import json
import socket
from unittest.mock import AsyncMock

import httpx
import pytest

from nanobot.agent.tools import web
from nanobot.config.schema import Config


@pytest.mark.parametrize("url", [
    "https://tiktok.com/cats", "https://www.tiktok.com/cats",
    "https://WWW.TIKTOK.COM./cats", "https://user:secret@tiktok.com/cats",
    "https://tiktok。com/cats", "https://%74iktok.com/cats",
])
async def test_environment_block_precedes_all_external_effects(monkeypatch, url):
    monkeypatch.setenv("NANOBOT_WEB_FETCH_BLOCKED_HOSTS", " facebook.com, TIKTOK.COM. ")

    def forbidden(*_args, **_kwargs):
        pytest.fail("blocked target reached DNS or HTTP construction")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(web.httpx, "AsyncClient", forbidden)
    result = await web.WebFetchTool().execute(url)
    assert result.startswith("Error: direct web_fetch is disabled")
    assert "secret" not in result


def test_blocked_hosts_config_roundtrips():
    config = Config.model_validate({"tools": {"web": {"fetch": {
        "blockedHosts": ["tiktok.com"],
    }}}})
    loaded = Config.model_validate_json(config.model_dump_json(by_alias=True))
    assert loaded.tools.web.fetch.blocked_hosts == ["tiktok.com"]


@pytest.mark.parametrize("target", ["facebook.com", "www.tiktok.com"])
async def test_config_and_environment_blocks_are_additive(monkeypatch, target):
    monkeypatch.setenv("NANOBOT_WEB_FETCH_BLOCKED_HOSTS", "facebook.com")
    config = web.WebFetchConfig.model_validate({"blockedHosts": ["tiktok.com"]})
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("DNS reached"))
    result = await web.WebFetchTool(config=config).execute(f"https://{target}/")
    assert result.startswith("Error: direct web_fetch is disabled")


@pytest.mark.parametrize("host", ["not-tiktok.com", "tiktok.com.example", "example.com"])
async def test_nonmatching_hosts_use_local_fetch_without_remote_extractor(monkeypatch, host):
    monkeypatch.setenv("NANOBOT_WEB_FETCH_BLOCKED_HOSTS", "tiktok.com")
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text="public page", headers={"content-type": "text/plain"})

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443)),
    ])
    monkeypatch.setattr(web, "_fetch_client_kwargs", lambda *_a: {
        "transport": httpx.MockTransport(handler),
    })
    tool = web.WebFetchTool()
    tool._fetch_jina = AsyncMock(side_effect=AssertionError("remote redirects cannot be policed"))
    data = json.loads(await tool.execute(f"https://{host}/"))
    assert data["untrusted"] is True
    assert "public page" in data["text"]
    assert seen == [f"https://{host}/"] * 2
    tool._fetch_jina.assert_not_awaited()


@pytest.mark.parametrize("mode", ["stream", "get"])
@pytest.mark.parametrize("target", ["https://www.tiktok.com/private", "//tiktok.com/private"])
async def test_redirect_is_rejected_before_destination_dns_and_request(monkeypatch, mode, target):
    dns = []
    requested = []

    def resolve(host, *_args, **_kwargs):
        dns.append(host)
        assert host == "example.com"
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))]

    def handler(request):
        requested.append(str(request.url))
        assert request.url.host == "example.com"
        return httpx.Response(302, headers={"location": target})

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if mode == "get":
            response, error = await web._get_with_safe_redirects(
                client, "https://example.com/", blocked_hosts=("tiktok.com",),
            )
        else:
            response, stream, error, _credentials = await web._stream_with_safe_redirects(
                client, "https://example.com/", blocked_hosts=("tiktok.com",),
            )
            assert stream is None
    assert response is None
    assert error is not None and "direct web_fetch is disabled" in error
    assert requested == ["https://example.com/"]
    assert dns == ["example.com"]


@pytest.mark.parametrize("value", ["https://tiktok.com", "*.tiktok.com", "tiktok.com/path"])
async def test_invalid_block_configuration_fails_closed(monkeypatch, value):
    monkeypatch.setenv("NANOBOT_WEB_FETCH_BLOCKED_HOSTS", value)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("DNS reached"))
    result = await web.WebFetchTool().execute("https://example.com/")
    assert result.startswith("Error: Invalid web_fetch blocked-host configuration")


@pytest.mark.parametrize("preflight_fails", [False, True])
async def test_execute_enforces_redirect_policy_in_preflight_and_fallback(monkeypatch, preflight_fails):
    monkeypatch.setenv("NANOBOT_WEB_FETCH_BLOCKED_HOSTS", "tiktok.com")
    requested = []

    def resolve(host, *_args, **_kwargs):
        assert host == "example.com", "blocked redirect must not reach DNS"
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))]

    def handler(request):
        requested.append(str(request.url))
        assert request.url.host == "example.com"
        if preflight_fails and len(requested) == 1:
            raise httpx.ConnectError("synthetic preflight failure", request=request)
        return httpx.Response(302, headers={"location": "https://www.tiktok.com/private"})

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(web, "_fetch_client_kwargs", lambda *_a: {
        "transport": httpx.MockTransport(handler),
    })
    data = json.loads(await web.WebFetchTool().execute("https://example.com/"))
    assert "direct web_fetch is disabled" in data["error"]
    assert requested == ["https://example.com/"] * (2 if preflight_fails else 1)


async def test_direct_remote_extractor_cannot_bypass_configured_policy(monkeypatch):
    monkeypatch.setenv("NANOBOT_WEB_FETCH_BLOCKED_HOSTS", "tiktok.com")
    monkeypatch.setattr(web.httpx, "AsyncClient", lambda **_k: pytest.fail("remote request attempted"))
    assert await web.WebFetchTool()._fetch_jina("https://example.com/", 1000) is None
