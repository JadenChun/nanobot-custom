"""Regression tests for the web_fetch download bound (A3-6)."""

from __future__ import annotations

import socket

import httpx
import pytest

from nanobot.agent.tools.web import MAX_FETCH_DOWNLOAD_BYTES, _get_with_safe_redirects


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


def _constant_body_handler(body: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=body,
            request=request,
        )

    return handler


@pytest.mark.asyncio
async def test_fetch_rejects_body_above_download_limit_before_decoding(monkeypatch):
    monkeypatch.setattr("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public)
    body = b"x" * (MAX_FETCH_DOWNLOAD_BYTES + 1)

    assert MAX_FETCH_DOWNLOAD_BYTES == 8 * 1024 * 1024

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_constant_body_handler(body))
    ) as client:
        response, error = await _get_with_safe_redirects(client, "https://example.com/big")

    assert response is None
    assert error is not None
    assert "8 MiB" in error


@pytest.mark.asyncio
async def test_fetch_accepts_body_below_limit(monkeypatch):
    monkeypatch.setattr("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_constant_body_handler(b"ok"))
    ) as client:
        response, error = await _get_with_safe_redirects(client, "https://example.com/small")

    assert error is None
    assert response is not None
    assert response.content == b"ok"
