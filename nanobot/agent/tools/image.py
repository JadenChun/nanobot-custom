"""Image generation tool using OpenRouter API or Codex CLI."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.config.schema import ImageConfig


_CURRENT_USER_IMAGE_REQUEST: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nanobot_current_user_image_request",
    default=None,
)


def set_current_user_image_request(text: str) -> contextvars.Token[str | None]:
    """Make the raw user request available to image generation tool calls."""
    return _CURRENT_USER_IMAGE_REQUEST.set(text)


def reset_current_user_image_request(token: contextvars.Token[str | None]) -> None:
    _CURRENT_USER_IMAGE_REQUEST.reset(token)


class ImageGenerationTool(Tool):
    """Generate images using AI via OpenRouter API or Codex CLI."""

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

    def __init__(self, config: ImageConfig | None = None, workspace: Path | None = None):
        from nanobot.config.schema import ImageConfig

        self.config = config if config is not None else ImageConfig()
        self.workspace = workspace

    async def execute(self, prompt: str, output_path: str, aspect_ratio: str = "16:9", **kwargs: Any) -> str:
        provider = (self.config.provider or "auto").lower()
        if provider == "auto":
            if _codex_command_available(self.config.codex_command):
                provider = "codex_cli"
            elif self.config.api_key or os.environ.get("OPENROUTER_API_KEY"):
                provider = "openrouter"

        if provider in {"codex", "codex_cli", "codex-cli"}:
            return await self._execute_codex_cli(prompt, output_path, aspect_ratio)
        if provider == "openrouter":
            return await self._execute_openrouter(prompt, output_path, aspect_ratio)
        return json.dumps({
            "error": (
                "No image provider is available. Configure tools.image.provider as "
                "'codex_cli' or 'openrouter'."
            )
        })

    async def _execute_openrouter(self, prompt: str, output_path: str, aspect_ratio: str) -> str:
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
            out = self._resolve_output_path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(image_bytes)
            logger.info("Image saved to {} ({} bytes)", out, len(image_bytes))
        except Exception as e:
            logger.error("Failed to save image to {}: {}", output_path, e)
            return json.dumps({"error": f"Failed to save image: {e}"})

        return json.dumps({"path": str(out), "size_bytes": len(image_bytes)})

    async def _execute_codex_cli(self, prompt: str, output_path: str, aspect_ratio: str) -> str:
        command = self.config.codex_command or "codex"
        executable = _resolve_codex_command(command)
        if not executable:
            return json.dumps({
                "error": (
                    f"Codex CLI command not found: {command!r}. Install Codex CLI or set "
                    "tools.image.codex_command to the full executable path."
                )
            })

        output = self._resolve_output_path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        codex_home = _codex_home(self.config.codex_home)
        start_time = time.time()
        generated_before = _known_generated_images(codex_home)
        cli_prompt = _build_codex_prompt(prompt, aspect_ratio)
        model = self.config.codex_model or "gpt-5.4-mini"
        timeout = max(1, int(self.config.codex_timeout or 300))

        stdout = ""
        stderr = ""
        args = [
            executable,
            "exec",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-m",
            model,
            cli_prompt,
        ]
        try:
            logger.info("Running Codex CLI image generation via non-interactive exec mode")
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return json.dumps({"error": f"Codex CLI image generation timed out after {timeout} seconds"})
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            if process.returncode != 0:
                return json.dumps({
                    "error": f"Codex CLI exited with code {process.returncode}",
                    "stderr": _tail(stderr),
                    "stdout": _tail(stdout),
                })
        except Exception as e:
            logger.error("Codex CLI image generation failed: {}", e)
            return json.dumps({"error": f"Codex CLI request failed: {e}"})

        source = _find_generated_image(stdout + "\n" + stderr, codex_home, generated_before, start_time)
        if not source:
            if output.exists() and output.is_file():
                source = output
            else:
                return json.dumps({
                    "error": "Codex CLI finished but no generated image file was found",
                    "stdout": _tail(stdout),
                    "stderr": _tail(stderr),
                })

        try:
            if source.resolve() != output:
                shutil.copy2(source, output)
            size = output.stat().st_size
        except Exception as e:
            logger.error("Failed to copy Codex-generated image from {} to {}: {}", source, output, e)
            return json.dumps({"error": f"Failed to save image: {e}", "source_path": str(source)})

        logger.info("Codex CLI image saved to {} from {}", output, source)
        return json.dumps({
            "path": str(output),
            "source_path": str(source),
            "size_bytes": size,
            "provider": "codex_cli",
            "prompt": cli_prompt,
        })

    def _resolve_output_path(self, output_path: str) -> Path:
        requested = Path(output_path).expanduser()
        if requested.is_absolute():
            resolved = requested.resolve()
        elif self.workspace:
            resolved = (self.workspace / "generated_images" / requested.name).resolve()
        else:
            resolved = requested.resolve()

        if not self.workspace:
            return resolved

        workspace = self.workspace.resolve()
        generated_dir = (workspace / "generated_images").resolve()
        try:
            resolved.relative_to(generated_dir)
            return resolved
        except ValueError:
            pass

        try:
            resolved.relative_to(workspace)
        except ValueError:
            return resolved
        return generated_dir / resolved.name


def _codex_command_available(command: str) -> bool:
    return _resolve_codex_command(command) is not None


def image_generation_available(config: "ImageConfig") -> bool:
    """Return whether the image tool should be exposed to the agent."""
    provider = (config.provider or "auto").lower()
    if provider in {"codex", "codex_cli", "codex-cli"}:
        return True
    if config.api_key or os.environ.get("OPENROUTER_API_KEY"):
        return True
    if provider == "auto":
        return _codex_command_available(config.codex_command or "codex")
    return False


def _resolve_codex_command(command: str) -> str | None:
    command = command or "codex"
    if any(sep in command for sep in ("/", "\\")):
        path = Path(command).expanduser()
        return str(path) if path.exists() else None
    return shutil.which(command)


def _codex_home(configured: str = "") -> Path:
    if configured:
        return Path(configured).expanduser()
    if override := os.environ.get("CODEX_HOME"):
        return Path(override).expanduser()
    return Path.home() / ".codex"


def _build_codex_prompt(prompt: str, aspect_ratio: str) -> str:
    return prompt.strip()


def _known_generated_images(codex_home: Path) -> set[Path]:
    root = codex_home / "generated_images"
    if not root.exists():
        return set()
    return {p.resolve() for p in root.rglob("*") if p.is_file() and _is_image_path(p)}


def _find_generated_image(
    text: str,
    codex_home: Path,
    generated_before: set[Path],
    start_time: float,
) -> Path | None:
    for path in _extract_image_paths(text):
        if path.exists() and path.is_file():
            return path.resolve()

    root = codex_home / "generated_images"
    if not root.exists():
        return None
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or not _is_image_path(path):
            continue
        resolved = path.resolve()
        if resolved in generated_before:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= start_time - 2:
            candidates.append((mtime, resolved))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _extract_image_paths(text: str) -> list[Path]:
    cleaned = _strip_ansi(text)
    patterns = [
        r"[A-Za-z]:\\[^\r\n\"'`<>|]*?\.(?:png|jpg|jpeg|webp)",
        r"/[^\s\"'`<>|]*?\.(?:png|jpg|jpeg|webp)",
    ]
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            value = match.group(0).rstrip(".,;:)]}")
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            paths.append(Path(value).expanduser())
    return paths


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text or "")


def _tail(text: str, limit: int = 2000) -> str:
    text = _strip_ansi(text or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]
