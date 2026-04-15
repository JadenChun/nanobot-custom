"""Image generation tool using OpenRouter API."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.config.schema import ImageConfig


class ImageGenerationTool(Tool):
    """Generate images using AI via OpenRouter API."""

    name = "generate_image"
    description = "Generate an image using AI. Returns the file path of the saved image."
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Image generation prompt"},
            "output_path": {"type": "string", "description": "Where to save the image file"},
            "aspect_ratio": {
                "type": "string",
                "description": "Aspect ratio hint (e.g. 16:9, 1:1, 9:16)",
                "default": "16:9",
            },
        },
        "required": ["prompt", "output_path"],
    }

    def __init__(self, config: ImageConfig | None = None):
        from nanobot.config.schema import ImageConfig

        self.config = config if config is not None else ImageConfig()

    async def execute(self, prompt: str, output_path: str, aspect_ratio: str = "16:9", **kwargs: Any) -> str:
        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return json.dumps({"error": "No API key configured. Set OPENROUTER_API_KEY or configure tools.image.api_key."})

        model = self.config.model or "bytedance-seed/seedream-4.5"
        api_base = self.config.api_base or "https://openrouter.ai/api/v1"
        url = f"{api_base.rstrip('/')}/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": f"{prompt} (aspect ratio: {aspect_ratio})"}
            ],
            "modalities": ["image"],
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                r.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("Image generation API error: {} {}", e.response.status_code, e.response.text[:500])
            return json.dumps({"error": f"API error: {e.response.status_code} {e.response.text[:200]}"})
        except Exception as e:
            logger.error("Image generation request failed: {}", e)
            return json.dumps({"error": f"Request failed: {e}"})

        try:
            data = r.json()
            content = data["choices"][0]["message"]["content"]

            # Content is a list; find the image_url entry
            image_data_url = None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_data_url = item.get("image_url", {}).get("url", "")
                        break
            elif isinstance(content, str) and content.startswith("data:image/"):
                image_data_url = content

            if not image_data_url:
                logger.error("No image data found in API response")
                return json.dumps({"error": "No image data found in API response"})

            # Parse data URI: data:image/png;base64,<data>
            if ";base64," not in image_data_url:
                return json.dumps({"error": "Unexpected image data format (not base64)"})

            b64_data = image_data_url.split(";base64,", 1)[1]
            image_bytes = base64.b64decode(b64_data)
        except (KeyError, IndexError) as e:
            logger.error("Failed to parse image generation response: {}", e)
            return json.dumps({"error": f"Failed to parse response: {e}"})
        except Exception as e:
            logger.error("Failed to decode image data: {}", e)
            return json.dumps({"error": f"Failed to decode image: {e}"})

        try:
            out = Path(output_path).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(image_bytes)
            logger.info("Image saved to {} ({} bytes)", out, len(image_bytes))
        except Exception as e:
            logger.error("Failed to save image to {}: {}", output_path, e)
            return json.dumps({"error": f"Failed to save image: {e}"})

        return json.dumps({"path": str(out), "size_bytes": len(image_bytes)})
