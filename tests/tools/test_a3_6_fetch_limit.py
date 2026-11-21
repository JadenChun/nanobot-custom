"""Regression tests for the web_fetch download bound (A3-6)."""

from __future__ import annotations

import gzip
import json
import random
import socket

import httpx
import pytest

from nanobot.agent.tools import web as web_module
from nanobot.agent.tools.web import WebFetchTool
from nanobot.config.schema import WebFetchConfig


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


class FixtureStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self):
        for offset in range(0, len(self.body), 65_536):
            yield self.body[offset:offset + 65_536]


def _handler(request: httpx.Request) -> httpx.Response:
    headers = {"content-type": "text/plain"}
    limit = 8 * 1024 * 1024
    if request.url.path == "/small":
        body = b"ok"
    elif request.url.path == "/encoded-too-large":
        body = gzip.compress(random.Random(0).randbytes(limit))
        assert len(body) > limit
        headers["content-encoding"] = "gzip"
    else:
        body = b"x" * (limit + 1)
    return httpx.Response(
        200,
        headers=headers,
        stream=FixtureStream(body),
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
@pytest.mark.parametrize("path", ["/big", "/encoded-too-large"])
async def test_fetch_rejects_streamed_body_above_download_limit(path):
    tool = WebFetchTool(config=WebFetchConfig(use_jina_reader=False))

    result = json.loads(await tool.execute(url=f"https://example.com{path}"))

    assert "error" in result
    assert "8 MiB" in result["error"]
    assert "text" not in result


@pytest.mark.asyncio
async def test_fetch_returns_a_normal_response():
    tool = WebFetchTool(config=WebFetchConfig(use_jina_reader=False))

    result = json.loads(await tool.execute(url="https://example.com/small"))

    assert "error" not in result
    assert result["extractor"] == "raw"
    assert result["untrusted"] is True
    assert "ok" in result["text"]
