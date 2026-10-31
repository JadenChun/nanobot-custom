import base64
import struct
import zlib

from nanobot.utils.media_tokens import (
    MAX_AUDIO_TOKENS,
    MAX_IMAGE_TOKENS,
    MAX_VIDEO_TOKENS,
    binary_media_tokens,
    estimate_image_tokens,
    image_dimensions,
    inline_media_tokens,
    media_block_tokens,
    unknown_image_tokens,
)


def png_payload(width: int, height: int, padding: int = 0) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = (
        struct.pack(">I", len(ihdr_data))
        + b"IHDR"
        + ihdr_data
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    )
    return signature + ihdr + b"\x00" * padding


def data_uri(payload: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def test_image_dimensions_are_read_from_container_headers() -> None:
    assert image_dimensions(png_payload(1024, 768)) == (1024, 768)
    gif = b"GIF89a" + struct.pack("<HH", 320, 240) + b"\x00" * 8
    assert image_dimensions(gif) == (320, 240)
    webp = b"RIFF" + struct.pack("<I", 30) + b"WEBP" + b"VP8X"
    webp += b"\x00" * 8 + struct.pack("<I", 1023)[:3] + struct.pack("<I", 767)[:3]
    assert image_dimensions(webp) == (1024, 768)


def test_image_allowance_covers_common_vision_accounting() -> None:
    assert estimate_image_tokens(1024, 1024) >= 1_300
    assert estimate_image_tokens(1024, 1024) < MAX_IMAGE_TOKENS
    assert estimate_image_tokens(1, 1) < 1_000
    assert estimate_image_tokens(10_000, 10_000) == MAX_IMAGE_TOKENS


def test_inline_media_uses_dimensions_and_bounded_unknown_fallbacks() -> None:
    inline = data_uri(png_payload(1024, 1024, padding=1_000_000))
    tokens, kind = inline_media_tokens(inline) or (0, "")
    assert kind == "image"
    assert 1_300 <= tokens < 2_000

    tiny_unknown = data_uri(b"not an image")
    assert inline_media_tokens(tiny_unknown) == (160, "image")
    large_unknown = data_uri(b"x" * (256 * 1024))
    assert inline_media_tokens(large_unknown) == (MAX_IMAGE_TOKENS, "image")
    assert inline_media_tokens(data_uri(b"audio", "audio/wav")) == (
        MAX_AUDIO_TOKENS,
        "audio",
    )
    assert inline_media_tokens(data_uri(b"video", "video/mp4")) == (
        MAX_VIDEO_TOKENS,
        "video",
    )
    assert inline_media_tokens("data:text/plain;base64,aGVsbG8=") is None
    assert inline_media_tokens("https://example.test/image.png") is None


def test_binary_media_uses_same_bounded_image_allowance() -> None:
    assert 1_300 <= binary_media_tokens(png_payload(1024, 1024), "image") < 2_000
    assert binary_media_tokens(b"tiny", "image") == unknown_image_tokens(len(b"tiny"))
    assert binary_media_tokens(b"video", "video") == MAX_VIDEO_TOKENS


def test_media_blocks_cover_chat_responses_and_anthropic_shapes() -> None:
    inline = data_uri(png_payload(1024, 1024))
    assert media_block_tokens({"type": "image_url", "image_url": {"url": inline}})[1] == "image"
    assert media_block_tokens({"type": "input_image", "image_url": inline})[1] == "image"
    encoded = inline.split(",", 1)[1]
    assert media_block_tokens({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
    })[1] == "image"
    assert media_block_tokens({
        "image": {"source": {"bytes": png_payload(1024, 1024)}},
    })[1] == "image"
    assert media_block_tokens({"type": "text", "text": "plain"}) is None
