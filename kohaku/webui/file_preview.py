"""Workspace and assistant-shared source previews for the WebUI."""

from __future__ import annotations

import mimetypes
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from posixpath import normpath
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import unquote, urlparse

from kohaku.config.paths import get_media_dir
from kohaku.context_repo.policy import ResourceAccessPolicy
from kohaku.security.workspace_access import WorkspaceScope
from kohaku.security.workspace_policy import WorkspaceBoundaryError, resolve_allowed_path
from kohaku.webui.media_api import guess_media_mime, is_allowed_media_mime

if TYPE_CHECKING:
    from kohaku.security.desktop_permissions import (
        DesktopPermissionOwner,
        WorkspacePermissionReference,
    )

MAX_FILE_PREVIEW_BYTES = 384 * 1024
_INLINE_PATH_RE = re.compile(r"(?<!`)`([^`\r\n]+)`(?!`)")
_LINK_PATH_RE = re.compile(r"(?<!!)\[[^\]\r\n]*\]\(\s*(?:<([^>\r\n]+)>|([^\s)]+))")
_QUOTED_ABSOLUTE_PATH_RE = re.compile(
    r"""(?:"((?:/|[A-Za-z]:[\\/])[^"\r\n]+)"|'((?:/|[A-Za-z]:[\\/])[^'\r\n]+)')"""
)
_EMBEDDED_ABSOLUTE_PATH_RE = re.compile(
    r"(?:->|=>|\b(?:path|file|out|output|saved(?:\s+to)?|wrote(?:\s+to)?|created)\s*[:=]?)\s*"
    r"((?:/|~/|[A-Za-z]:[\\/])[^\r\n\"'<>]+?)"
    r"(?=\s+\(\d[\d,.\s]*(?:bytes?|kb|mb|gb|b)\)\s*$|\s+(?:/|[A-Za-z]:[\\/])|$)",
    re.IGNORECASE | re.MULTILINE,
)
_TOKEN_ABSOLUTE_PATH_RE = re.compile(
    r"""(?:^|[\s=(>,]|:(?!//))((?:/(?!/)|~/|[A-Za-z]:[\\/])[^\s"'<>|;]+)"""
)
_NON_SINGLE_PATH_LINE_RE = re.compile(
    r"""["']|\s(?:->|=>|&&|\|\||[|;]|--?[A-Za-z])|\s+(?:/|~/|[A-Za-z]:[\\/])"""
)
_OFFICE_DOCUMENT_EXTENSIONS = frozenset({".docx", ".xlsx", ".pptx"})
_TOOL_ARGUMENT_PATH_KEYS = (
    "path",
    "file_path",
    "absolute_path",
    "command",
    "output",
    "out",
    "output_path",
    "dest",
    "destination",
    "target",
)


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
                for key in _TOOL_ARGUMENT_PATH_KEYS:
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
        try:
            requested_path = Path(requested).expanduser()
        except (OSError, RuntimeError, ValueError) as exc:
            raise WebUIFilePreviewError(400, "invalid path") from exc
        if requested_path.is_absolute():
            try:
                matches.add(requested_path.resolve(strict=False))
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
    raw_candidates: list[str] = [
        *[
            line
            for line in text.splitlines()
            if not _NON_SINGLE_PATH_LINE_RE.search(line)
        ],
        *_preview_text_references(text),
    ]
    for match in _QUOTED_ABSOLUTE_PATH_RE.finditer(text):
        raw_candidates.append(match.group(1) or match.group(2))
    for match in _EMBEDDED_ABSOLUTE_PATH_RE.finditer(text):
        raw_candidates.append(match.group(1).rstrip(".,;:)]}"))
    for match in _TOKEN_ABSOLUTE_PATH_RE.finditer(text):
        raw_candidates.append(match.group(1).rstrip(".,;:)]}"))

    for value in raw_candidates:
        value = _clean_preview_path(value.strip().strip("\"'"))
        if not value or len(value) > 4096:
            continue
        try:
            candidate = Path(value).expanduser()
        except (OSError, RuntimeError, ValueError):
            continue
        if not candidate.is_absolute():
            continue
        try:
            paths.add(candidate.resolve(strict=False))
        except (OSError, ValueError, RuntimeError):
            continue
    return paths


def _preview_media_kind(path: Path) -> tuple[str, str] | None:
    mime = guess_media_mime(path.name)
    if mime is None or not is_allowed_media_mime(mime):
        return None
    if mime.startswith("image/"):
        return ("image", mime)
    if mime.startswith("video/"):
        return ("video", mime)
    if mime.startswith("audio/"):
        return ("audio", mime)
    if mime == "application/pdf":
        return ("pdf", mime)
    return None


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
    media_url: Callable[[Path], dict[str, str] | None] | None = None,
    referenced_path: Path | None = None,
) -> dict[str, Any]:
    """Return a text, media, PDF, or directory preview for an allowed path."""

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
    signer = media_url if media_url is not None else image_url
    media_info = _preview_media_kind(resolved)
    if media_info is not None:
        kind, mime = media_info
        if signer is None:
            raise WebUIFilePreviewError(415, "binary files cannot be previewed")
        media = signer(resolved)
        media_href = media.get("url", "") if media is not None else ""
        if not media_href:
            raise WebUIFilePreviewError(500, "failed to prepare media preview")
        return {
            "kind": kind,
            "path": str(resolved),
            "display_path": _display_path(resolved, scope.project_path),
            "project_path": str(scope.project_path),
            "language": "",
            "content": "",
            "size": resolved.stat().st_size,
            "truncated": False,
            "mime": mime,
            "media_url": media_href,
        }
    if resolved.suffix.lower() in _OFFICE_DOCUMENT_EXTENSIONS:
        from kohaku.utils.document import extract_text

        extracted = extract_text(resolved)
        if extracted is None or extracted.startswith("[error:"):
            raise WebUIFilePreviewError(415, "binary files cannot be previewed")
        truncated = len(extracted) > max_bytes
        payload: dict[str, Any] = {
            "kind": "text",
            "path": str(resolved),
            "display_path": _display_path(resolved, scope.project_path),
            "project_path": str(scope.project_path),
            "language": "text",
            "content": extracted[:max_bytes],
            "size": resolved.stat().st_size,
            "truncated": truncated,
        }
        if signer is not None:
            media = signer(resolved)
            media_href = media.get("url", "") if media is not None else ""
            if media_href:
                payload["mime"] = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
                payload["media_url"] = media_href
        return payload

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
    """Confirm that a path is a readable preview candidate without loading it fully."""

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
    if _preview_media_kind(resolved) is not None:
        return {"available": True}
    if resolved.suffix.lower() in _OFFICE_DOCUMENT_EXTENSIONS:
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


LocalFileOpenAction = Literal["reveal", "open", "open_with"]
_DESKTOP_ENTRY_ID_RE = re.compile(r"^[A-Za-z0-9._-]+\.desktop$")
_MAX_OPEN_WITH_APPS = 12


def resolve_file_preview_path(
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
) -> Path:
    """Resolve and authorize a preview target path (file or directory)."""
    return _resolve_preview_path(
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


def _xdg_application_dirs() -> list[Path]:
    dirs: list[Path] = []
    home = Path.home()
    data_home = Path(os.environ.get("XDG_DATA_HOME", "").strip() or (home / ".local/share"))
    dirs.append(data_home / "applications")
    raw_data_dirs = os.environ.get(
        "XDG_DATA_DIRS",
        "/usr/local/share:/usr/share:/var/lib/flatpak/exports/share",
    )
    for item in raw_data_dirs.split(":"):
        cleaned = item.strip()
        if cleaned:
            dirs.append(Path(cleaned) / "applications")
    dirs.append(home / ".local/share/flatpak/exports/share/applications")
    dirs.append(Path("/var/lib/flatpak/exports/share/applications"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in dirs:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _find_desktop_file(desktop_id: str) -> Path | None:
    if not _DESKTOP_ENTRY_ID_RE.fullmatch(desktop_id):
        return None
    for root in _xdg_application_dirs():
        candidate = root / desktop_id
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _desktop_entry_name(desktop_file: Path) -> str:
    try:
        text = desktop_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return desktop_file.stem
    in_entry = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            if in_entry:
                break
            in_entry = line == "[Desktop Entry]"
            continue
        if in_entry and line.startswith("Name="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    return desktop_file.stem


def _mime_type_for_local_open(path: Path) -> str | None:
    guessed = guess_media_mime(path.name) or mimetypes.guess_type(path.name)[0]
    if guessed:
        return guessed
    if sys.platform.startswith("linux") and shutil.which("gio"):
        try:
            proc = subprocess.run(
                ["gio", "info", "-a", "standard::content-type", str(path)],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for line in proc.stdout.splitlines():
            if "standard::content-type:" in line:
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
    return None


def list_local_open_apps(path: Path) -> list[dict[str, Any]]:
    """Return registered desktop applications capable of opening ``path``."""
    if not sys.platform.startswith("linux") or path.is_dir():
        return []
    if not shutil.which("gio"):
        return []
    mime = _mime_type_for_local_open(path)
    if not mime:
        return []
    try:
        proc = subprocess.run(
            ["gio", "mime", mime],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    default_id: str | None = None
    ordered_ids: list[str] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.search(r"([A-Za-z0-9._-]+\.desktop)\b", line)
        if match is None:
            continue
        desktop_id = match.group(1)
        if line.lower().startswith("default application") and default_id is None:
            default_id = desktop_id
        if desktop_id not in ordered_ids:
            ordered_ids.append(desktop_id)

    apps: list[dict[str, Any]] = []
    for desktop_id in ordered_ids:
        desktop_file = _find_desktop_file(desktop_id)
        if desktop_file is None:
            continue
        apps.append({
            "id": desktop_id,
            "name": _desktop_entry_name(desktop_file),
            "default": desktop_id == default_id,
        })
        if len(apps) >= _MAX_OPEN_WITH_APPS:
            break
    return apps


def _spawn_detached(argv: Sequence[str]) -> None:
    try:
        subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise WebUIFilePreviewError(500, "failed to launch desktop application") from exc


def open_local_file_preview_path(
    path: Path,
    *,
    action: LocalFileOpenAction,
    app_id: str | None = None,
) -> dict[str, Any]:
    """Open or reveal an already-authorized local file in the host desktop."""
    if not path.exists():
        raise WebUIFilePreviewError(404, "file not found")

    if sys.platform == "darwin":
        if action == "reveal":
            cmd = ["open", str(path)] if path.is_dir() else ["open", "-R", str(path)]
            _spawn_detached(cmd)
        elif action == "open_with" and app_id:
            _spawn_detached(["open", "-a", app_id, str(path)])
        else:
            _spawn_detached(["open", str(path)])
        return {"ok": True, "action": action, "path": str(path)}

    if sys.platform == "win32":
        if action == "reveal":
            if path.is_dir():
                _spawn_detached(["explorer", str(path)])
            else:
                _spawn_detached(["explorer", f"/select,{path}"])
        elif action == "open_with":
            _spawn_detached(["rundll32.exe", "shell32.dll,OpenAs_RunDLL", str(path)])
        else:
            startfile = getattr(os, "startfile", None)
            if callable(startfile):
                try:
                    startfile(str(path))
                except OSError as exc:
                    raise WebUIFilePreviewError(500, "failed to open file") from exc
            else:
                raise WebUIFilePreviewError(500, "failed to open file")
        return {"ok": True, "action": action, "path": str(path)}

    # Linux / BSD desktop
    if action == "reveal":
        if not path.is_dir() and shutil.which("gdbus"):
            try:
                uri = path.resolve().as_uri()
                escaped_uri = uri.replace("\\", "\\\\").replace("'", "\\'")
                proc = subprocess.run(
                    [
                        "gdbus",
                        "call",
                        "--session",
                        "--dest",
                        "org.freedesktop.FileManager1",
                        "--object-path",
                        "/org/freedesktop/FileManager1",
                        "--method",
                        "org.freedesktop.FileManager1.ShowItems",
                        f"['{escaped_uri}']",
                        "",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    check=False,
                )
                if proc.returncode == 0:
                    return {"ok": True, "action": action, "path": str(path)}
            except (OSError, subprocess.SubprocessError):
                pass
        folder = path if path.is_dir() else path.parent
        opener = shutil.which("xdg-open") or shutil.which("gio")
        if opener and opener.endswith("gio"):
            _spawn_detached([opener, "open", str(folder)])
        elif opener:
            _spawn_detached([opener, str(folder)])
        else:
            raise WebUIFilePreviewError(503, "no desktop file manager opener available")
        return {"ok": True, "action": action, "path": str(path)}

    if action == "open_with" and app_id:
        allowed_ids = {str(entry["id"]) for entry in list_local_open_apps(path)}
        desktop_file = _find_desktop_file(app_id) if app_id in allowed_ids else None
        if desktop_file is None:
            raise WebUIFilePreviewError(400, "invalid application selection")
        gio = shutil.which("gio")
        if not gio:
            raise WebUIFilePreviewError(503, "gio launcher is unavailable")
        _spawn_detached([gio, "launch", str(desktop_file), str(path)])
        return {"ok": True, "action": action, "path": str(path)}

    opener = shutil.which("xdg-open") or shutil.which("gio")
    if opener and opener.endswith("gio"):
        _spawn_detached([opener, "open", str(path)])
    elif opener:
        _spawn_detached([opener, str(path)])
    else:
        raise WebUIFilePreviewError(503, "no desktop opener available")
    return {"ok": True, "action": action, "path": str(path)}
