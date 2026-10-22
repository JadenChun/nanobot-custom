"""Legacy Codex CLI image backend with owned invocation and bounded provenance."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from nanobot.agent.tools.process import run_owned_process
from nanobot.providers.image_generation import (
    GeneratedImageResponse,
    ImageGenerationError,
    ImageGenerationProvider,
)
from nanobot.utils.helpers import detect_image_mime

if TYPE_CHECKING:
    from nanobot.config.schema import ImageConfig

_LOCKS: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Path, asyncio.Lock]] = WeakKeyDictionary()


def codex_home(configured: str = "") -> Path:
    return Path(configured or os.environ.get("CODEX_HOME") or Path.home() / '.codex').expanduser().resolve()


def resolve_command(command: str) -> str | None:
    return shutil.which(command or 'codex')


def _snapshot(root: Path) -> dict[Path, tuple[int, int, str, int, int]]:
    files: dict[Path, tuple[int, int, str, int, int]] = {}
    if not root.exists():
        return files
    for path in root.rglob('*'):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'} or not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        stat = resolved.stat()
        if stat.st_size > 32 * 1024 * 1024:
            continue
        files[resolved] = (stat.st_mtime_ns, stat.st_size, hashlib.sha256(resolved.read_bytes()).hexdigest(), stat.st_dev, stat.st_ino)
    return files


class CodexCLIImageGenerationClient(ImageGenerationProvider):
    provider_name = 'codex_cli'

    def __init__(self, config: ImageConfig) -> None:
        super().__init__(api_key=None)
        self.config = config

    async def generate(self, *, prompt: str, model: str, reference_images: list[str] | None = None,
                       aspect_ratio: str | None = None, image_size: str | None = None) -> GeneratedImageResponse:
        if reference_images or image_size:
            raise ImageGenerationError('Codex CLI image edits and explicit image_size are not supported')
        executable = resolve_command(self.config.codex_command)
        if not executable:
            raise ImageGenerationError(f'Codex CLI command not found: {self.config.codex_command}')
        home = codex_home(self.config.codex_home)
        root = (home / 'generated_images').resolve()
        loop_locks = _LOCKS.setdefault(asyncio.get_running_loop(), {})
        lock = loop_locks.setdefault(root, asyncio.Lock())
        async with lock:
            before = _snapshot(root)
            env = dict(os.environ)
            env['CODEX_HOME'] = str(home)
            effective_prompt = prompt.strip()
            result = await run_owned_process(
                [executable, 'exec', '--skip-git-repo-check', '--color', 'never', '-m', model, effective_prompt],
                timeout=max(1, self.config.codex_timeout), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            if result.timed_out:
                raise ImageGenerationError('Codex CLI image generation timed out')
            if result.returncode != 0:
                raise ImageGenerationError(f'Codex CLI exited with code {result.returncode}')
            after = _snapshot(root)
            fresh = {path for path, stamp in after.items() if path not in before or before[path][2] != stamp[2]}
            emitted = result.stdout.decode("utf-8", errors="replace") + "\n" + result.stderr.decode("utf-8", errors="replace")
            candidates: set[Path] = set()
            references = re.findall(r"(?<![\w./\\])(?:[A-Za-z]:[\\/]|/)[^\s\"'`<>|]*?\.(?:png|jpg|jpeg|webp)\b", emitted, re.IGNORECASE)
            for reported in references:
                resolved = Path(reported).resolve()
                if resolved.is_relative_to(root) and resolved in fresh:
                    candidates.add(resolved)
                emitted = emitted.replace(reported, "")
            # Preserve directory scans only when correlated with a child-reported
            # relative path/basename. An output-free scan cannot identify its producer.
            for token in re.findall(r"[A-Za-z0-9_./-]+\.(?:png|jpg|jpeg|webp)\b", emitted, re.IGNORECASE):
                relative = Path(token)
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                matches = {path for path in fresh if path.relative_to(root).as_posix() == token
                           or path.relative_to(root.parent).as_posix() == token
                           or (len(relative.parts) == 1 and path.name == token)}
                candidates.update(matches)
            if len(candidates) != 1:
                raise ImageGenerationError("Codex completed without a unique attributable generated image")
            source = candidates.pop()
            initial = source.stat()
            raw = source.read_bytes()
            final = source.stat()
            digest = hashlib.sha256(raw).hexdigest()
            stamp = (final.st_mtime_ns, final.st_size, digest, final.st_dev, final.st_ino)
            identity_before = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns)
            identity_after = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns)
            if identity_before != identity_after or stamp != after[source]:
                raise ImageGenerationError("Generated image changed during capture")
            mime = detect_image_mime(raw)
            if mime is None:
                raise ImageGenerationError("Codex generated an unsupported image")
            data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
            return GeneratedImageResponse([data_url], '', {
                'source_path': str(source), 'provider': 'codex_cli', 'prompt': effective_prompt,
            })
