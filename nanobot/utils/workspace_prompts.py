"""Shared file handling for workspace-local prompt overrides."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from nanobot.utils.helpers import truncate_text

WORKSPACE_PROMPT_MAX_CHARS = 32_000


def workspace_prompt_file(workspace: Path, name: str) -> Path:
    """Return the conventional path for a named workspace prompt override."""
    return workspace / "prompts" / f"{name}.md"


def _confined_prompt_path(path: Path, workspace: Path | None) -> Path | None:
    """Resolve an override path and require it to stay inside its workspace."""
    if workspace is None:
        return path
    try:
        resolved = path.resolve(strict=True)
        root = workspace.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_relative_to(root) else None


def load_workspace_prompt_override(
    path: Path,
    *,
    workspace: Path | None = None,
    max_chars: int = WORKSPACE_PROMPT_MAX_CHARS,
) -> tuple[str | None, int]:
    """Load and cap a non-empty UTF-8 prompt override.

    An explicitly provided workspace confines the resolved file. An escaping
    symlink is refused before its contents can become instructions.
    """
    candidate = _confined_prompt_path(path, workspace)
    if candidate is None:
        return None, 0
    with suppress(OSError, UnicodeDecodeError):
        text = candidate.read_text(encoding="utf-8").rstrip()
        if text:
            original_chars = len(text)
            return truncate_text(text, max_chars), original_chars
    return None, 0


def has_workspace_prompt_override(
    path: Path,
    *,
    workspace: Path | None = None,
) -> bool:
    """Return whether a path contains a non-empty workspace prompt override."""
    text, _original_chars = load_workspace_prompt_override(
        path,
        workspace=workspace,
    )
    return text is not None


def initialize_workspace_prompt(
    path: Path,
    default_prompt: str,
    *,
    workspace: Path | None = None,
) -> bool:
    """Create a default prompt copy when the target is missing or empty.

    Returns ``False`` without overwriting a non-empty file, non-file path,
    escaping path, or path whose current state cannot be read safely.
    """
    if workspace is not None:
        try:
            root = workspace.expanduser().resolve(strict=False)
            parent = path.parent.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return False
        if not parent.is_relative_to(root):
            return False
        if path.is_symlink():
            try:
                if not path.resolve(strict=True).is_relative_to(root):
                    return False
            except (OSError, RuntimeError, ValueError):
                return False
    try:
        if path.exists() and (
            not path.is_file() or bool(path.read_text(encoding="utf-8").strip())
        ):
            return False
    except (OSError, UnicodeDecodeError):
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_prompt + "\n", encoding="utf-8")
    return True
