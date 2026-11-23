"""Shared path helpers for workspace-scoped tools."""

from pathlib import Path

from nanobot.agent.tools.context import current_request_context
from nanobot.config.paths import get_media_dir
from nanobot.security.workspace_policy import resolve_allowed_path


def current_media_read_root() -> Path:
    """Return the media read root for the current request.

    A request bound to an explicit instance carries that instance's media root.
    Use it instead of the process-global (ambient) media directory so an
    instance artifact stays readable when another config is ambient. Requests
    without a bound root keep the legacy ambient behavior.
    """
    ctx = current_request_context()
    root = ctx.media_root if ctx is not None else None
    return Path(root).expanduser() if root is not None else get_media_dir()


def resolve_workspace_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
    extra_allowed_files: list[Path] | None = None,
    include_media_dir: bool = True,
) -> Path:
    """Resolve path against workspace and enforce allowed directory containment."""
    media_roots = [current_media_read_root()] if include_media_dir else []
    extra_roots = [*media_roots, *(extra_allowed_dirs or [])] if allowed_dir else None
    return resolve_allowed_path(
        path,
        workspace=workspace,
        allowed_root=allowed_dir,
        extra_allowed_roots=extra_roots,
        extra_allowed_files=extra_allowed_files,
    )
