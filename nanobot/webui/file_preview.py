"""Workspace and assistant-shared source previews for the WebUI."""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from posixpath import normpath
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlparse

from nanobot.config.paths import get_media_dir
from nanobot.context_repo.policy import ResourceAccessPolicy
from nanobot.security.workspace_access import WorkspaceScope
from nanobot.security.workspace_policy import WorkspaceBoundaryError, resolve_allowed_path

if TYPE_CHECKING:
    from nanobot.security.desktop_permissions import (
        DesktopPermissionOwner,
        WorkspacePermissionReference,
    )

MAX_FILE_PREVIEW_BYTES = 384 * 1024
_INLINE_PATH_RE = re.compile(r"(?<!`)`([^`\r\n]+)`(?!`)")
_LINK_PATH_RE = re.compile(r"(?<!!)\[[^\]\r\n]*\]\(\s*(?:<([^>\r\n]+)>|([^\s)]+))")


class WebUIFilePreviewError(ValueError):
    """Raised when a file cannot be previewed through the WebUI."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def file_preview_reference(
    raw_path: str | None,
    records: Sequence[Mapping[str, object]],
) -> Path | None:
    """Resolve only a file the assistant shared in this chat's display transcript.

    Relative references use absolute paths observed in successful tool results in
    the same turn. A filename alone never triggers a filesystem search or grants
    access to a directory. Read-policy checks still apply to the returned file.
    """
    requested = _clean_preview_path(raw_path)
    if not requested or len(requested) > 4096:
        return None
    texts: dict[str, list[str]] = {}
    deltas: dict[str, str] = {}
    observed: dict[str, set[Path]] = {}
    legacy_turn = 0
    for record in records:
        event = record.get("event")
        if event in ("user", "user_message"):
            legacy_turn += 1
        turn = record.get("turn_id")
        turn = turn if isinstance(turn, str) and turn else f"legacy:{legacy_turn}"
        text = record.get("text")
        if event == "delta" and isinstance(text, str):
            deltas[turn] = deltas.get(turn, "") + text
        elif event in ("message", "stream_end") and record.get("kind") not in (
            "progress", "tool_hint", "reasoning",
        ) and isinstance(text, str):
            texts.setdefault(turn, []).append(text)
        tool_events = record.get("tool_events")
        if not isinstance(tool_events, list):
            continue
        # Display transcript objects come from JSON, whose object keys are strings.
        for item in cast(list[object], tool_events):
            if not isinstance(item, dict):
                continue
            tool = cast(dict[str, object], item)
            if tool.get("phase") != "end" or tool.get("error"):
                continue
            paths = observed.setdefault(turn, set())
            result = tool.get("result")
            if isinstance(result, str):
                paths.update(_absolute_preview_paths(result))
            arguments = tool.get("arguments")
            if isinstance(arguments, dict):
                args = cast(dict[str, object], arguments)
                for key in ("path", "file_path", "absolute_path"):
                    value = args.get(key)
                    if isinstance(value, str):
                        paths.update(_absolute_preview_paths(value))

    for turn, delta in deltas.items():
        if turn not in texts:
            texts[turn] = [delta]
    suffix = PurePosixPath(normpath(requested.replace("\\", "/"))).parts
    while suffix and suffix[0] in {".", ".."}:
        suffix = suffix[1:]
    matches: set[Path] = set()
    for turn, replies in texts.items():
        references = [path for text in replies for path in _preview_text_references(text)]
        if requested not in references:
            continue
        if Path(requested).expanduser().is_absolute():
            try:
                matches.add(Path(requested).expanduser().resolve(strict=False))
            except (OSError, ValueError, RuntimeError) as exc:
                raise WebUIFilePreviewError(400, "invalid path") from exc
            continue
        candidates = set(observed.get(turn, ()))
        for reference in references:
            candidates.update(_absolute_preview_paths(reference))
        for candidate in candidates:
            parts = PurePosixPath(candidate.as_posix()).parts
            if suffix and parts[-len(suffix):] == suffix:
                matches.add(candidate)
    if len(matches) > 1:
        raise WebUIFilePreviewError(409, "file reference is ambiguous; use its full path")
    return next(iter(matches), None)


def _preview_text_references(text: str) -> list[str]:
    paths = [match.group(1) for match in _INLINE_PATH_RE.finditer(text)]
    paths.extend(match.group(1) or match.group(2) for match in _LINK_PATH_RE.finditer(text))
    return [_clean_preview_path(path) for path in paths]


def _absolute_preview_paths(text: str) -> set[Path]:
    paths: set[Path] = set()
    for value in [*text.splitlines(), *_preview_text_references(text)]:
        value = _clean_preview_path(value.strip().strip("\"'"))
        if value and len(value) <= 4096 and Path(value).expanduser().is_absolute():
            try:
                paths.add(Path(value).expanduser().resolve(strict=False))
            except (OSError, ValueError, RuntimeError):
                continue
    return paths


def file_preview_payload(
    raw_path: str | None,
    *,
    scope: WorkspaceScope,
    max_bytes: int = MAX_FILE_PREVIEW_BYTES,
    resource_policy: ResourceAccessPolicy | None = None,
    managed_access: bool = False,
    allow_folder_grants: bool = False,
    permission_owner: DesktopPermissionOwner | None = None,
    permission_reference: WorkspacePermissionReference | None = None,
    restrict_to_root: Path | None = None,
    image_url: Callable[[Path], dict[str, str] | None] | None = None,
    referenced_path: Path | None = None,
) -> dict[str, Any]:
    """Return a text, image or directory preview for an allowed path."""

    resolved = _resolve_preview_path(
        raw_path,
        scope=scope,
        resource_policy=resource_policy,
        managed_access=managed_access,
        allow_folder_grants=allow_folder_grants,
        permission_owner=permission_owner,
        permission_reference=permission_reference,
        restrict_to_root=restrict_to_root,
        allow_directory=True,
        referenced_path=referenced_path,
    )
    if resolved.is_dir():
        return _directory_preview_payload(resolved, scope)
    mime, _ = mimetypes.guess_type(resolved.name)
    if (mime or "").startswith("image/"):
        media = image_url(resolved) if image_url is not None else None
        if media is not None:
            return {
                "kind": "image",
                "path": str(resolved),
                "display_path": _display_path(resolved, scope.project_path),
                "project_path": str(scope.project_path),
                "language": "",
                "content": "",
                "size": resolved.stat().st_size,
                "truncated": False,
                "mime": mime,
                "media_url": media.get("url", ""),
            }

    try:
        with open(resolved, "rb") as f:
            raw = f.read(max_bytes + 1)
    except OSError as e:
        raise WebUIFilePreviewError(500, "failed to read file") from e

    if b"\0" in raw[:4096]:
        raise WebUIFilePreviewError(415, "binary files cannot be previewed")

    truncated = len(raw) > max_bytes
    preview_bytes = raw[:max_bytes]
    try:
        content = preview_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = preview_bytes.decode("utf-8", errors="replace")

    display_path = _display_path(resolved, scope.project_path)
    return {
        "kind": "text",
        "path": str(resolved),
        "display_path": display_path,
        "project_path": str(scope.project_path),
        "language": _language_for_path(resolved),
        "content": content,
        "size": resolved.stat().st_size,
        "truncated": truncated,
    }


def file_preview_availability_payload(
    raw_path: str | None,
    *,
    scope: WorkspaceScope,
    resource_policy: ResourceAccessPolicy | None = None,
    managed_access: bool = False,
    allow_folder_grants: bool = False,
    permission_owner: DesktopPermissionOwner | None = None,
    permission_reference: WorkspacePermissionReference | None = None,
    restrict_to_root: Path | None = None,
    referenced_path: Path | None = None,
) -> dict[str, bool]:
    """Confirm that a path is a readable text preview candidate without loading it fully."""

    resolved = _resolve_preview_path(
        raw_path,
        scope=scope,
        resource_policy=resource_policy,
        managed_access=managed_access,
        allow_folder_grants=allow_folder_grants,
        permission_owner=permission_owner,
        permission_reference=permission_reference,
        restrict_to_root=restrict_to_root,
        allow_directory=True,
        referenced_path=referenced_path,
    )
    if resolved.is_dir():
        return {"available": True}
    mime, _ = mimetypes.guess_type(resolved.name)
    if (mime or "").startswith("image/"):
        return {"available": True}
    try:
        with open(resolved, "rb") as f:
            prefix = f.read(4096)
    except OSError as e:
        raise WebUIFilePreviewError(500, "failed to read file") from e
    if b"\0" in prefix:
        raise WebUIFilePreviewError(415, "binary files cannot be previewed")
    return {"available": True}


def _resolve_preview_path(
    raw_path: str | None,
    *,
    scope: WorkspaceScope,
    resource_policy: ResourceAccessPolicy | None,
    managed_access: bool,
    allow_folder_grants: bool,
    permission_owner: DesktopPermissionOwner | None,
    permission_reference: WorkspacePermissionReference | None,
    restrict_to_root: Path | None = None,
    allow_directory: bool = False,
    referenced_path: Path | None = None,
) -> Path:
    path = _clean_preview_path(raw_path)
    if not path:
        raise WebUIFilePreviewError(400, "missing path")
    if len(path) > 4096:
        raise WebUIFilePreviewError(400, "path is too long")
    # Only the server-resolved, assistant-shared file extends the preview scope;
    # its parent folder and unrelated files receive no additional access.
    shared_files = [referenced_path] if referenced_path is not None and resource_policy else []
    if referenced_path is not None:
        path = str(referenced_path)

    try:
        allow_granted_outside_root = (
            allow_folder_grants
            and resource_policy is not None
            and restrict_to_root is None
        )
        allowed_root = (
            None
            if allow_granted_outside_root
            else (
                restrict_to_root
                if restrict_to_root is not None
                else (scope.project_path if scope.restrict_to_workspace else None)
            )
        )
        extra_roots = [get_media_dir()] if scope.restrict_to_workspace else None
        resolved = resolve_allowed_path(
            path,
            workspace=scope.project_path,
            allowed_root=allowed_root,
            extra_allowed_roots=extra_roots,
            extra_allowed_files=shared_files,
            strict=True,
        )
    except FileNotFoundError as e:
        raise WebUIFilePreviewError(404, "file not found") from e
    except WorkspaceBoundaryError as e:
        raise WebUIFilePreviewError(403, "file is outside the current workspace") from e
    except OSError as e:
        raise WebUIFilePreviewError(400, "invalid path") from e

    if not resolved.is_file() and not (allow_directory and resolved.is_dir()):
        raise WebUIFilePreviewError(404, "file not found")
    if resource_policy is not None:
        extra_roots = [get_media_dir()] if scope.restrict_to_workspace else []
        try:
            resource_policy.validate_path(
                resolved,
                "read",
                extra_read_roots=extra_roots,
                extra_read_files=shared_files,
                managed=managed_access,
                allow_folder_grants=allow_folder_grants,
                permission_owner=permission_owner,
                permission_reference=permission_reference,
            )
        except PermissionError as exc:
            raise WebUIFilePreviewError(
                403,
                "file is outside the current folder access",
            ) from exc
    return resolved


def _clean_preview_path(raw_path: str | None) -> str:
    if raw_path is None:
        return ""
    value = raw_path.strip()
    if not value:
        return ""
    if value.startswith("file://"):
        parsed = urlparse(value)
        value = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:[\\/]", value):
            value = value[1:]
    else:
        value = unquote(value)
    value = value.split("?", 1)[0].split("#", 1)[0].strip()
    if not re.match(r"^[A-Za-z]:[\\/]", value):
        value = re.sub(r":\d+(?::\d+)?$", "", value)
    return value


MAX_DIRECTORY_PREVIEW_ENTRIES = 500


def _directory_preview_payload(path: Path, scope: WorkspaceScope) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    try:
        children = sorted(
            path.iterdir(),
            key=lambda child: (not child.is_dir(), child.name.lower()),
        )
    except OSError as e:
        raise WebUIFilePreviewError(500, "failed to read directory") from e
    for child in children[:MAX_DIRECTORY_PREVIEW_ENTRIES]:
        try:
            is_dir = child.is_dir()
            size = 0 if is_dir else child.stat().st_size
        except OSError:
            continue
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "kind": "directory" if is_dir else "file",
                "size": size,
            }
        )
    return {
        "kind": "directory",
        "path": str(path),
        "display_path": _display_path(path, scope.project_path),
        "project_path": str(scope.project_path),
        "language": "",
        "content": "",
        "size": 0,
        "truncated": len(children) > len(entries),
        "entries": entries,
    }


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _language_for_path(path: Path) -> str:
    name = path.name.lower()
    ext = path.suffix.lower().lstrip(".")
    if name == "dockerfile":
        return "dockerfile"
    return {
        "cjs": "javascript",
        "css": "css",
        "cts": "typescript",
        "html": "html",
        "js": "javascript",
        "json": "json",
        "jsonl": "json",
        "jsx": "jsx",
        "md": "markdown",
        "mdx": "markdown",
        "mjs": "javascript",
        "mts": "typescript",
        "py": "python",
        "pyi": "python",
        "scss": "scss",
        "sh": "bash",
        "toml": "toml",
        "ts": "typescript",
        "tsx": "tsx",
        "yaml": "yaml",
        "yml": "yaml",
    }.get(ext, ext or "text")
