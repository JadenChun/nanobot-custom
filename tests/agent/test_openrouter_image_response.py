"""Native JSON response parsing through local HTTPX transport, never a live provider."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from nanobot.agent.tools import image as image_tool
from nanobot.config.schema import ImageConfig

_IMAGE = b"local image bytes"
_URI = "data:image/png;base64," + base64.b64encode(_IMAGE).decode("ascii")
_VALID_ITEM = {"type": "image_url", "image_url": {"url": _URI}}
_NO_IMAGE = {"error": "No image data found in API response"}


class _LocalHTTP:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.requests: list[httpx.Request] = []
        self.clients: list[httpx.AsyncClient] = []
        self.response_body: object = {}
        self.raw_body: bytes | None = None
        self.status = 200
        self.error: Exception | None = None
        native_client = httpx.AsyncClient

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if self.error is not None:
                raise self.error
            if self.raw_body is not None:
                return httpx.Response(self.status, content=self.raw_body, request=request)
            return httpx.Response(self.status, json=self.response_body, request=request)

        def client(*, timeout: float) -> httpx.AsyncClient:
            assert timeout == 120.0
            result = native_client(transport=httpx.MockTransport(respond), timeout=timeout)
            self.clients.append(result)
            return result

        monkeypatch.setattr(image_tool.httpx, "AsyncClient", client)

    def content(self, value: object) -> None:
        self.response_body = {"choices": [{"message": {"content": value}}]}

    def closed_once(self) -> None:
        assert len(self.requests) == len(self.clients) == 1
        assert self.clients[0].is_closed


@pytest.fixture
def local_http(monkeypatch: pytest.MonkeyPatch) -> _LocalHTTP:
    return _LocalHTTP(monkeypatch)


def _tool(tmp_path: Path) -> image_tool.ImageGenerationTool:
    return image_tool.ImageGenerationTool(
        ImageConfig(
            provider="openrouter", api_key="dummy-no-network",
            api_base="https://images.invalid/api/v1", model="local-fixture-model",
        ),
        workspace=tmp_path,
    )


async def _execute(
    tmp_path: Path, local_http: _LocalHTTP, content: object,
) -> tuple[dict[str, object], Path]:
    local_http.content(content)
    result = json.loads(await _tool(tmp_path).execute("local scene", "probe.png", "3:2"))
    local_http.closed_once()
    return result, tmp_path / "generated_images" / "probe.png"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (_URI, _IMAGE),
        ([_VALID_ITEM], _IMAGE),
        ([None, 3, "text", [], {"type": "text"}, _VALID_ITEM], _IMAGE),
        ([_VALID_ITEM, {"type": "image_url", "image_url": {"url": "invalid"}}], _IMAGE),
        ([{"type": "image_url", "image_url": {"url": "prefix;base64,YQ=="}}], b"a"),
        ("data:image/png;base64,", b""),
        ("data:image/png;base64,!!!!", b""),
    ],
)
async def test_valid_native_json_and_existing_base64_defaults(
    tmp_path: Path, local_http: _LocalHTTP, content: object, expected: bytes,
) -> None:
    result, output = await _execute(tmp_path, local_http, content)
    assert result == {"path": str(output), "size_bytes": len(expected)}
    assert output.read_bytes() == expected
    request = local_http.requests[0]
    assert str(request.url) == "https://images.invalid/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer dummy-no-network"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "model": "local-fixture-model",
        "messages": [{"role": "user", "content": "local scene (aspect ratio: 3:2)"}],
        "modalities": ["image"],
    }


@pytest.mark.parametrize("content", [None, False, 3, {}, [], "ordinary text", [{"type": "text"}]])
async def test_non_image_content_retains_no_image_result(
    tmp_path: Path, local_http: _LocalHTTP, content: object,
) -> None:
    result, output = await _execute(tmp_path, local_http, content)
    assert result == _NO_IMAGE
    assert not output.exists()


@pytest.mark.parametrize("value", [None, False, 0, 0.0, "", [], {}])
async def test_first_falsy_url_does_not_fall_through_to_later_image(
    tmp_path: Path, local_http: _LocalHTTP, value: object,
) -> None:
    content = [{"type": "image_url", "image_url": {"url": value}}, _VALID_ITEM]
    result, output = await _execute(tmp_path, local_http, content)
    assert result == _NO_IMAGE
    assert not output.exists()


@pytest.mark.parametrize("item", [{"type": "image_url"}, {"type": "image_url", "image_url": {}}])
async def test_first_missing_url_keeps_existing_empty_default(
    tmp_path: Path, local_http: _LocalHTTP, item: dict[str, object],
) -> None:
    result, output = await _execute(tmp_path, local_http, [item, _VALID_ITEM])
    assert result == _NO_IMAGE
    assert not output.exists()


@pytest.mark.parametrize("value", [None, False, 0, "not an object", [], ["nested"]])
async def test_malformed_nested_image_object_is_explicitly_rejected(
    tmp_path: Path, local_http: _LocalHTTP, value: object,
) -> None:
    result, output = await _execute(
        tmp_path, local_http, [{"type": "image_url", "image_url": value}, _VALID_ITEM],
    )
    assert result == {"error": "Failed to decode image: image_url must be an object"}
    assert not output.exists()


@pytest.mark.parametrize("value", [True, 1, 1.5, ["not text"], {"not": "text"}])
async def test_truthy_non_string_url_is_explicitly_rejected(
    tmp_path: Path, local_http: _LocalHTTP, value: object,
) -> None:
    result, output = await _execute(
        tmp_path, local_http, [{"type": "image_url", "image_url": {"url": value}}, _VALID_ITEM],
    )
    assert result == {"error": "Failed to decode image: image_url.url must be text"}
    assert not output.exists()


@pytest.mark.parametrize("value", ["https://images.invalid/probe.png", "data:image/png,plain"])
async def test_non_base64_text_keeps_existing_format_result(
    tmp_path: Path, local_http: _LocalHTTP, value: str,
) -> None:
    result, output = await _execute(
        tmp_path, local_http, [{"type": "image_url", "image_url": {"url": value}}],
    )
    assert result == {"error": "Unexpected image data format (not base64)"}
    assert not output.exists()


async def test_invalid_base64_keeps_native_decode_error(
    tmp_path: Path, local_http: _LocalHTTP,
) -> None:
    result, output = await _execute(tmp_path, local_http, "data:image/png;base64,abc")
    assert result == {"error": "Failed to decode image: Incorrect padding"}
    assert not output.exists()


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ({}, "Failed to parse response: 'choices'"),
        ({"choices": []}, "Failed to parse response: list index out of range"),
        ({"choices": [{}]}, "Failed to parse response: 'message'"),
        ({"choices": [{"message": {}}]}, "Failed to parse response: 'content'"),
    ],
)
async def test_root_key_and_index_failures_keep_parse_envelope(
    tmp_path: Path, local_http: _LocalHTTP, body: object, error: str,
) -> None:
    local_http.response_body = body
    result = json.loads(await _tool(tmp_path).execute("local scene", "probe.png"))
    assert result == {"error": error}
    assert not (tmp_path / "generated_images").exists()
    local_http.closed_once()


async def test_http_status_failure_precedes_json_decode(
    tmp_path: Path, local_http: _LocalHTTP,
) -> None:
    local_http.status = 403
    local_http.raw_body = b"denied fixture"
    result = json.loads(await _tool(tmp_path).execute("local scene", "probe.png"))
    assert result == {"error": "API error: 403 denied fixture"}
    assert not (tmp_path / "generated_images").exists()
    local_http.closed_once()


async def test_request_failure_keeps_existing_envelope_and_closes_client(
    tmp_path: Path, local_http: _LocalHTTP,
) -> None:
    local_http.error = httpx.ConnectError("local transport failure")
    result = json.loads(await _tool(tmp_path).execute("local scene", "probe.png"))
    assert result == {"error": "Request failed: local transport failure"}
    assert not (tmp_path / "generated_images").exists()
    local_http.closed_once()


async def test_invalid_json_keeps_existing_decode_envelope(
    tmp_path: Path, local_http: _LocalHTTP,
) -> None:
    local_http.raw_body = b"not json"
    result = json.loads(await _tool(tmp_path).execute("local scene", "probe.png"))
    assert result == {"error": "Failed to decode image: Expecting value: line 1 column 1 (char 0)"}
    assert not (tmp_path / "generated_images").exists()
    local_http.closed_once()


async def test_save_failure_is_not_reclassified_as_decode_failure(
    tmp_path: Path, local_http: _LocalHTTP, monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = OSError("local write refusal")

    def refuse(self: image_tool.ImageGenerationTool, path: str) -> Path:
        assert self.workspace == tmp_path and path == "probe.png"
        raise error

    monkeypatch.setattr(image_tool.ImageGenerationTool, "_resolve_output_path", refuse)
    result, output = await _execute(tmp_path, local_http, [_VALID_ITEM])
    assert result == {"error": "Failed to save image: local write refusal"}
    assert not output.exists()


async def test_missing_key_never_constructs_http_client(
    tmp_path: Path, local_http: _LocalHTTP, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    tool = image_tool.ImageGenerationTool(ImageConfig(provider="openrouter"), workspace=tmp_path)
    result = json.loads(await tool.execute("local scene", "probe.png"))
    assert result == {
        "error": "No API key configured. Set OPENROUTER_API_KEY or configure tools.image.api_key.",
    }
    assert local_http.clients == local_http.requests == []
    assert not (tmp_path / "generated_images").exists()
