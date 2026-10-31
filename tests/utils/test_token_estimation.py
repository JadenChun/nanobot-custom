import base64
import json
import struct
import zlib
from collections.abc import Iterator

import pytest
from tiktoken import Encoding, load

from nanobot.utils import helpers
from nanobot.utils.helpers import (
    estimate_message_tokens,
    estimate_prompt_tokens,
    estimate_prompt_tokens_chain,
    truncate_text_to_tokens,
)


class _NoCounterProvider:
    pass


class _BrokenCounterProvider:
    def estimate_prompt_tokens(self, messages, tools=None, model=None):
        raise RuntimeError("counter unavailable")


def _inline_image_data_uri() -> str:
    ihdr_data = struct.pack(">IIBBBBB", 1024, 1024, 8, 2, 0, 0, 0)
    ihdr = (
        struct.pack(">I", len(ihdr_data))
        + b"IHDR"
        + ihdr_data
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    )
    payload = b"\x89PNG\r\n\x1a\n" + ihdr + b"\x00" * 1_000_000
    return "data:image/png;base64," + base64.b64encode(payload).decode()


@pytest.fixture(autouse=True)
def isolated_token_caches() -> Iterator[None]:
    """Neither real nor synthetic encodings may leak between these unit tests."""
    # Tests replace the loader fault seam before fixture teardown.
    clear_encoding = helpers._get_token_encoding.cache_clear
    clear_tools = helpers._TOOLS_TOKEN_CACHE.clear
    clear_encoding()
    clear_tools()
    try:
        yield
    finally:
        clear_encoding()
        clear_tools()


@pytest.fixture
def unit_encoding(monkeypatch: pytest.MonkeyPatch) -> Encoding:
    """Exercise real BPE offline; this vocabulary is not model-specific cl100k_base."""
    def refuse_asset_read(blobpath: str, expected_hash: str | None = None) -> bytes:
        raise AssertionError("unit token counter must not load external encoding assets")

    monkeypatch.setattr(load, "read_file_cached", refuse_asset_read)
    ranks = {bytes([value]): value for value in range(256)}
    character = "你".encode("utf-8")
    ranks[character[:2]] = 256
    ranks[character] = 257
    encoding = Encoding(
        name="nanobot-offline-unit-bpe", pat_str=r"[\s\S]+",
        mergeable_ranks=ranks, special_tokens={},
    )

    def get_encoding(name: str) -> Encoding:
        assert name == "cl100k_base"
        return encoding

    monkeypatch.setattr(helpers.tiktoken, "get_encoding", get_encoding)
    return encoding


def test_estimate_prompt_tokens_chain_falls_back_without_provider_counter(
    unit_encoding: Encoding,
) -> None:
    tokens, source = estimate_prompt_tokens_chain(
        _NoCounterProvider(),
        "test-model",
        [{"role": "user", "content": "hello"}],
    )

    assert tokens > 0
    assert source == "tiktoken"


def test_estimate_prompt_tokens_chain_falls_back_when_provider_counter_fails(
    unit_encoding: Encoding,
) -> None:
    tokens, source = estimate_prompt_tokens_chain(
        _BrokenCounterProvider(),
        "test-model",
        [{"role": "user", "content": "hello"}],
    )

    assert tokens > 0
    assert source == "tiktoken"


def test_estimate_prompt_tokens_uses_conservative_fallback_when_tiktoken_fails(
    monkeypatch,
    unit_encoding: Encoding,
) -> None:
    monkeypatch.setattr(
        helpers,
        "_get_token_encoding",
        lambda: (_ for _ in ()).throw(RuntimeError("encoding unavailable")),
    )

    content = "你" * 1_000
    messages = [{"role": "user", "content": content}]
    tokens = estimate_prompt_tokens(messages)
    chain_tokens, source = estimate_prompt_tokens_chain(
        _NoCounterProvider(),
        "test-model",
        messages,
    )

    actual_tokens = len(helpers.tiktoken.get_encoding("cl100k_base").encode(content)) + 4
    assert tokens == len(content.encode("utf-8")) + 4
    assert tokens >= actual_tokens
    assert chain_tokens == tokens
    assert source == "heuristic"


@pytest.mark.parametrize("text", ["hello", "你" * 1_000, "🙂你", "line\n你"])
def test_unit_encoding_is_real_bpe_without_external_assets(
    unit_encoding: Encoding, text: str,
) -> None:
    encoded = unit_encoding.encode(text)
    assert unit_encoding.decode(encoded) == text
    assert 0 < len(encoded) <= len(text.encode("utf-8"))
    if "你" in text:
        assert len(encoded) < len(text.encode("utf-8"))
    messages = [{"role": "user", "content": text}]
    tokens, source = estimate_prompt_tokens_chain(_NoCounterProvider(), "test-model", messages)
    assert tokens == len(encoded) + 4
    assert source == "tiktoken"


def test_estimate_message_tokens_uses_utf8_byte_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        helpers,
        "_get_token_encoding",
        lambda: (_ for _ in ()).throw(RuntimeError("encoding unavailable")),
    )
    content = "🙂你" * 100

    assert estimate_message_tokens({"role": "user", "content": content}) == (
        len(content.encode("utf-8")) + 4
    )


def test_truncate_text_to_tokens_uses_utf8_byte_budget_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        helpers,
        "_get_token_encoding",
        lambda: (_ for _ in ()).throw(RuntimeError("encoding unavailable")),
    )

    result = truncate_text_to_tokens("🙂你" * 100, 40)

    assert result.endswith("\n... (truncated)")
    assert len(result.encode("utf-8")) <= 40


def test_estimate_prompt_tokens_caches_tools_encoding(monkeypatch) -> None:
    helpers._get_token_encoding.cache_clear()
    helpers._TOOLS_TOKEN_CACHE.clear()

    class FakeEncoding:
        def __init__(self) -> None:
            self.encoded: list[str] = []

        def encode(self, text: str) -> list[int]:
            self.encoded.append(text)
            return list(range(max(1, len(text) // 4)))

    fake_encoding = FakeEncoding()
    get_encoding_calls = 0

    def fake_get_encoding(name: str) -> FakeEncoding:
        nonlocal get_encoding_calls
        assert name == "cl100k_base"
        get_encoding_calls += 1
        return fake_encoding

    monkeypatch.setattr(helpers.tiktoken, "get_encoding", fake_get_encoding)
    tools = [{"type": "function", "function": {"name": "demo", "description": "cached"}}]
    messages = [{"role": "user", "content": "hello"}]

    first = estimate_prompt_tokens(messages, tools)
    second = estimate_prompt_tokens(messages, tools)

    assert first == second
    assert get_encoding_calls == 1
    rendered_tools = "\n" + json.dumps(tools, ensure_ascii=False)
    assert fake_encoding.encoded.count(rendered_tools) == 1


def test_estimate_prompt_tokens_recomputes_when_tool_items_change(monkeypatch) -> None:
    helpers._get_token_encoding.cache_clear()
    helpers._TOOLS_TOKEN_CACHE.clear()

    class FakeEncoding:
        def __init__(self) -> None:
            self.encoded: list[str] = []

        def encode(self, text: str) -> list[int]:
            self.encoded.append(text)
            return list(range(max(1, len(text) // 4)))

    fake_encoding = FakeEncoding()
    monkeypatch.setattr(helpers.tiktoken, "get_encoding", lambda _name: fake_encoding)

    tools = [{"type": "function", "function": {"name": "before"}}]
    messages = [{"role": "user", "content": "hello"}]
    estimate_prompt_tokens(messages, tools)

    tools[0] = {"type": "function", "function": {"name": "after"}}
    estimate_prompt_tokens(messages, tools)

    before_tools = "\n" + json.dumps(
        [{"type": "function", "function": {"name": "before"}}], ensure_ascii=False
    )
    after_tools = "\n" + json.dumps(tools, ensure_ascii=False)
    assert before_tools in fake_encoding.encoded
    assert after_tools in fake_encoding.encoded


def test_estimate_prompt_tokens_counts_images_as_bounded_vision_tokens(
    unit_encoding: Encoding,
) -> None:
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "compare"},
            {"type": "image_url", "image_url": {"url": _inline_image_data_uri()}},
        ],
    }

    tokens, source = estimate_prompt_tokens_chain(
        _NoCounterProvider(), "test-model", [message],
    )

    assert source == "tiktoken"
    assert 1_300 <= tokens < 5_000
    assert estimate_message_tokens(message) < 5_000


def test_estimate_prompt_tokens_does_not_count_image_payload_as_text(
    unit_encoding: Encoding,
) -> None:
    image = _inline_image_data_uri()
    text_only = estimate_prompt_tokens([{"role": "user", "content": image}])
    with_image = estimate_prompt_tokens([{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": image}}],
    }])

    assert with_image < 5_000
    assert with_image < text_only // 100


def test_module_local_setup_does_not_replay_other_test_caches() -> None:
    assert helpers._get_token_encoding.cache_info().currsize == 0
    assert helpers._TOOLS_TOKEN_CACHE == {}
