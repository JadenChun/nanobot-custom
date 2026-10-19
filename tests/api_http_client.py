"""Buffered aiohttp route tests without native sockets or a network fallback.

This exercises request serialization, routing, middleware and app lifecycle. It
does not qualify an HTTP listener, wire transport or streaming response delivery.
"""

from __future__ import annotations

import asyncio
import json as json_module
from collections.abc import Mapping

from aiohttp import FormData, web
from aiohttp.base_protocol import BaseProtocol
from aiohttp.payload import JsonPayload, Payload, get_payload
from aiohttp.payload import LookupError as PayloadLookupError
from aiohttp.streams import StreamReader
from aiohttp.test_utils import make_mocked_request

_UNSET = object()


class _BufferedWriter:
    def __init__(self) -> None:
        self.body = bytearray()

    async def write(self, data: bytes) -> None:
        self.body.extend(data)


class InProcessResponse:
    def __init__(self, response: web.Response) -> None:
        self.response = response

    @property
    def status(self) -> int:
        return self.response.status

    @property
    def headers(self) -> Mapping[str, str]:
        return self.response.headers

    @property
    def content_type(self) -> str:
        return self.response.content_type

    async def read(self) -> bytes:
        body = self.response.body
        if body is None:
            return b""
        if not isinstance(body, bytes):
            raise TypeError("in-process client supports buffered bytes responses only")
        return body

    async def text(self) -> str:
        return (await self.read()).decode(self.response.charset or "utf-8")

    async def json(self) -> object:
        return json_module.loads(await self.text())


class InProcessApiClient:
    def __init__(self, app: web.Application) -> None:
        self.app = app
        self.closed = False
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        if self.closed or self._started:
            raise RuntimeError("in-process client cannot be started again")
        self._loop = asyncio.get_running_loop()
        # Match AppRunner's application setup, without constructing its server.
        self.app._set_loop(self._loop)
        self.app.on_startup.freeze()
        try:
            await self.app.startup()
            self.app.freeze()
        except BaseException:
            try:
                await self.app.cleanup()
            finally:
                self.closed = True
            raise
        self._started = True

    async def close(self) -> None:
        if self.closed:
            return
        if self._loop is not None and asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("in-process client must close on its owning loop")
        try:
            if self._started:
                await self.app.shutdown()
        finally:
            try:
                await self.app.cleanup()
            finally:
                self.closed = True

    async def get(self, path: str, **kwargs: object) -> InProcessResponse:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: object) -> InProcessResponse:
        return await self.request("POST", path, **kwargs)

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: object = _UNSET,
        data: object = None,
    ) -> InProcessResponse:
        loop = asyncio.get_running_loop()
        if self.closed or not self._started or loop is not self._loop:
            raise RuntimeError("in-process request requires its live owning loop")
        if not path.startswith("/") or "://" in path:
            raise ValueError("in-process requests require a local route path")
        if json is not _UNSET and data is not None:
            raise ValueError("json and data cannot be combined")

        payload: Payload | None = None
        if json is not _UNSET:
            payload = JsonPayload(json)
        elif data is not None:
            if isinstance(data, FormData):
                payload = data()
            else:
                try:
                    payload = get_payload(data)
                except PayloadLookupError:
                    payload = FormData(data)()

        request_headers = dict(headers or {})
        writer = _BufferedWriter()
        if payload is not None:
            for key, value in payload.headers.items():
                if not any(existing.lower() == key.lower() for existing in request_headers):
                    request_headers[key] = value
            await payload.write(writer)
        request_headers["Content-Length"] = str(len(writer.body))
        # The payload is already buffered: there is no network parser to pause.
        # This watermark does not change Request's separate application body cap.
        reader = StreamReader(
            BaseProtocol(loop), limit=max(2**16, len(writer.body) + 1), loop=loop,
        )
        reader.feed_data(bytes(writer.body))
        reader.feed_eof()
        request = make_mocked_request(
            method, path, headers=request_headers, payload=reader, app=self.app,
            client_max_size=self.app._client_max_size, loop=loop,
        )
        try:
            response = await self.app._handle(request)
        except web.HTTPException as exc:
            response = web.Response(
                status=exc.status, reason=exc.reason, headers=exc.headers, body=exc.body,
            )
        if not isinstance(response, web.Response):
            raise TypeError("in-process client does not simulate streaming responses")
        return InProcessResponse(response)
