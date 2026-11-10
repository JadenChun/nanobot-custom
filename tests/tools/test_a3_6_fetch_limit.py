"""Regression tests for the web_fetch download bound (A3-6)."""

from __future__ import annotations

import json
import socket

import httpx
import pytest

from nanobot.agent.tools import web as web_module
from nanobot.agent.tools.web import MAX_FETCH_DOWNLOAD_BYTES, WebFetchTool
from nanobot.config.schema import WebFetchConfig


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


def _handler(request: httpx.Request) -> httpx.Response:
    body = b"ok" if request.url.path == "/small" else b"x" * (MAX_FETCH_DOWNLOAD_BYTES + 1)
    return httpx.Response(
        200,
        headers={"content-type": "text/plain"},
        content=body,
        request=request,
    )


@pytest.fixture(autouse=True)
def _fixture_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public)
    monkeypatch.setattr(
        web_module,
        "_pinned_dns_transport",
        lambda: httpx.MockTransport(_handler),
    )


@pytest.mark.asyncio
async def test_fetch_rejects_streamed_body_above_download_limit():
    tool = WebFetchTool(config=WebFetchConfig(use_jina_reader=False))

    result = json.loads(await tool.execute(url="https://example.com/big"))

    assert MAX_FETCH_DOWNLOAD_BYTES == 8 * 1024 * 1024
    assert "error" in result
    assert "8 MiB" in result["error"]


@pytest.mark.asyncio
async def test_fetch_returns_a_normal_response():
    tool = WebFetchTool(config=WebFetchConfig(use_jina_reader=False))

    result = json.loads(await tool.execute(url="https://example.com/small"))

    assert "error" not in result
    assert result["extractor"] == "raw"
    assert result["untrusted"] is True
    assert "ok" in result["text"]
