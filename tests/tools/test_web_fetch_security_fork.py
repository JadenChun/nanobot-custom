"""Tests for web_fetch SSRF protection and untrusted content marking."""

from __future__ import annotations

import json
import socket
from unittest.mock import patch

import httpx
import pytest

from nanobot.agent.tools.web import WebFetchConfig, WebFetchTool


@pytest.mark.asyncio
async def test_web_fetch_blocks_configured_login_gated_platform(monkeypatch):
    monkeypatch.setenv("NANOBOT_WEB_FETCH_BLOCKED_HOSTS", "facebook.com,tiktok.com")

    result = await WebFetchTool().execute("https://www.tiktok.com/tag/cats")

    assert result.startswith("Error: direct web_fetch is disabled")


def _fake_resolve_private(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0))]


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


@pytest.mark.asyncio
async def test_web_fetch_blocks_private_ip():
    tool = WebFetchTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(url="http://169.254.169.254/computeMetadata/v1/")
    data = json.loads(result)
    assert "error" in data
    assert "private" in data["error"].lower() or "blocked" in data["error"].lower()


@pytest.mark.asyncio
async def test_web_fetch_blocks_localhost():
    tool = WebFetchTool()
    def _resolve_localhost(hostname, port, family=0, type_=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]
    with patch("nanobot.security.network.socket.getaddrinfo", _resolve_localhost):
        result = await tool.execute(url="http://localhost/admin")
    data = json.loads(result)
    assert "error" in data


@pytest.mark.asyncio
async def test_web_fetch_result_contains_untrusted_flag():
    """When fetch succeeds, result JSON must include untrusted=True and the banner."""
    tool = WebFetchTool(config=WebFetchConfig(use_jina_reader=False))

    fake_html = "<html><head><title>Test</title></head><body><p>Hello world</p></body></html>"

    class FakeStreamResponse:
        status_code = 200
        headers = {"content-type": "text/html"}
        url = "https://example.com/page"
        request = httpx.Request("GET", "https://example.com/page")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_bytes(self):
            yield fake_html.encode()

    def _fake_stream(self, method, url, headers=None, **kwargs):
        return FakeStreamResponse()

    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public), \
         patch("httpx.AsyncClient.stream", _fake_stream):
        result = await tool.execute(url="https://example.com/page")

    data = json.loads(result)
    assert data.get("untrusted") is True
    assert "[External content" in data.get("text", "")


@pytest.mark.asyncio
async def test_web_fetch_blocks_private_redirect_before_returning_image(monkeypatch):
    tool = WebFetchTool()
    requested: list[str] = []
    closed: list[bool] = []

    class FakeStreamResponse:
        status_code = 302
        headers = {"location": "http://127.0.0.1/secret.png"}
        url = "https://example.com/image.png"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            closed.append(True)
            return False

        async def aread(self):
            pytest.fail("private redirect image must not be read")

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers=None, *, follow_redirects):
            assert method == "GET" and follow_redirects is False
            assert url == "https://example.com/image.png"
            requested.append(url)
            return FakeStreamResponse()

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", FakeClient)

    def resolve(hostname, *_args, **_kwargs):
        # A literal loopback address cannot resolve to the public fixture address.
        address = "127.0.0.1" if hostname == "127.0.0.1" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (address, 0))]

    with patch("nanobot.security.network.socket.getaddrinfo", resolve):
        result = await tool.execute(url="https://example.com/image.png")

    data = json.loads(result)
    assert "error" in data
    assert "redirect blocked" in data["error"].lower()
    assert requested == ["https://example.com/image.png"]
    assert closed == [True]
