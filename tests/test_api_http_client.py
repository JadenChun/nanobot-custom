"""Controls for the socket-free API route fixture, not native HTTP qualification."""

from __future__ import annotations

import socket
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nanobot.agent.turn import TurnResult
from nanobot.api.server import create_app
from tests.api_http_client import InProcessApiClient


@pytest.fixture
def no_http_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*_args, **_kwargs):
        pytest.fail("in-process HTTP client attempted native or network transport")

    for method in ("bind", "listen", "connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, method, denied)
    monkeypatch.setattr(aiohttp.ClientSession, "__init__", denied)
    monkeypatch.setattr(TestClient, "start_server", denied)
    monkeypatch.setattr(TestServer, "start_server", denied)


@pytest.mark.asyncio
async def test_in_process_dispatch_uses_real_auth_router_and_handler(no_http_network) -> None:
    agent = MagicMock()
    agent._execute_direct = AsyncMock(return_value=TurnResult(content="real handler result"))
    client = InProcessApiClient(create_app(agent, model_name="fixture-model", api_key="fixture-secret"))
    await client.start()
    try:
        health = await client.get("/health")
        refused = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "denied"}]},
        )
        wrong = await client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert health.status == 200
        assert await health.json() == {"status": "ok"}
        assert refused.status == wrong.status == 401
        agent._execute_direct.assert_not_awaited()
        response = await client.post(
            "/v1/chat/completions", headers={"authorization": "Bearer fixture-secret"},
            json={"messages": [{"role": "user", "content": "owned request"}]},
        )
        assert response.status == 200
        assert response.content_type == "application/json"
        assert (await response.json())["choices"][0]["message"]["content"] == "real handler result"
        agent._execute_direct.assert_awaited_once()
        assert agent._execute_direct.await_args.args == ("owned request",)
    finally:
        await client.close()
    assert client.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [None, False, 0, {}, [], {"unicode": "你好"}])
async def test_in_process_json_is_serialized_and_read_by_real_request(body, no_http_network) -> None:
    app = web.Application()

    async def echo(request: web.Request) -> web.Response:
        assert request.content_type == "application/json"
        return web.json_response(await request.json())

    app.router.add_post("/echo", echo)
    client = InProcessApiClient(app)
    await client.start()
    try:
        response = await client.post("/echo", json=body)
        assert response.status == 200
        assert await response.json() == body
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_form", [False, True])
async def test_in_process_multipart_uses_real_form_writer_and_reader(explicit_form, no_http_network) -> None:
    app = web.Application()

    async def upload(request: web.Request) -> web.Response:
        assert request.content_type == "multipart/form-data"
        parts = []
        async for part in await request.multipart():
            parts.append([part.name, part.filename, bytes(await part.read()).decode()])
        return web.json_response(parts)

    app.router.add_post("/upload", upload)
    client = InProcessApiClient(app)
    await client.start()
    try:
        if explicit_form:
            form = aiohttp.FormData()
            form.add_field("message", "analyze")
            form.add_field("files", b"first bytes", filename="first.txt", content_type="text/plain")
            form.add_field("files", b"second bytes", filename="second.txt", content_type="text/plain")
            expected = [
                ["message", None, "analyze"], ["files", "first.txt", "first bytes"],
                ["files", "second.txt", "second bytes"],
            ]
        else:
            form = {"message": "analyze", "files": BytesIO(b"first bytes")}
            expected = [["message", None, "analyze"], ["files", "files", "first bytes"]]
        response = await client.post("/upload", data=form)
        assert response.status == 200
        assert await response.json() == expected
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_in_process_router_errors_and_body_limit_are_real(no_http_network) -> None:
    app = web.Application(client_max_size=4)

    async def body(request: web.Request) -> web.Response:
        return web.Response(body=await request.read())

    app.router.add_post("/body", body)
    client = InProcessApiClient(app)
    await client.start()
    try:
        missing = await client.get("/missing")
        wrong_method = await client.get("/body")
        oversized = await client.post("/body", data=b"12345")
        assert missing.status == 404
        assert "Not Found" in await missing.text()
        assert wrong_method.status == 405
        assert wrong_method.headers["Allow"] == "POST"
        assert oversized.status == 413
        assert "Maximum request body size 4 exceeded" in await oversized.text()
        accepted = await client.post("/body", data=b"123")
        assert accepted.status == 200
        assert await accepted.read() == b"123"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_in_process_app_lifecycle_runs_once_and_unwinds_on_owner_loop(no_http_network) -> None:
    app = web.Application()
    events = []

    async def owned_context(_app):
        events.append("acquire")
        try:
            yield
        finally:
            events.append("release")

    async def shutdown(_app):
        events.append("shutdown")

    app.cleanup_ctx.append(owned_context)
    app.on_shutdown.append(shutdown)
    client = InProcessApiClient(app)
    await client.start()
    assert events == ["acquire"]
    await client.close()
    await client.close()
    assert events == ["acquire", "shutdown", "release"]
    assert client.closed
    with pytest.raises(RuntimeError, match="live owning loop"):
        await client.get("/health")


@pytest.mark.asyncio
async def test_in_process_partial_startup_releases_entered_context(no_http_network) -> None:
    app = web.Application()
    events = []
    failure = RuntimeError("synthetic startup failure")

    async def acquired(_app):
        events.append("acquire")
        try:
            yield
        finally:
            events.append("release")

    async def refused(_app):
        raise failure
        yield

    app.cleanup_ctx.extend([acquired, refused])
    client = InProcessApiClient(app)
    with pytest.raises(RuntimeError) as raised:
        await client.start()
    assert raised.value is failure
    assert events == ["acquire", "release"]
    assert client.closed
    await client.close()
    assert events == ["acquire", "release"]


@pytest.mark.asyncio
async def test_in_process_client_refuses_network_paths_and_ambiguous_body(no_http_network) -> None:
    client = InProcessApiClient(web.Application())
    await client.start()
    try:
        with pytest.raises(ValueError, match="local route"):
            await client.get("https://example.invalid/health")
        with pytest.raises(ValueError, match="cannot be combined"):
            await client.post("/body", json={}, data=b"bytes")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_in_process_client_does_not_fabricate_streaming_delivery(no_http_network) -> None:
    app = web.Application()

    async def stream(_request):
        return web.StreamResponse()

    app.router.add_get("/stream", stream)
    client = InProcessApiClient(app)
    await client.start()
    try:
        with pytest.raises(TypeError, match="does not simulate streaming"):
            await client.get("/stream")
    finally:
        await client.close()
