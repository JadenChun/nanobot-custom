"""Find project instructions in root-to-leaf order within a byte limit."""

from collections.abc import Callable
from pathlib import Path

ProjectFileValidator = Callable[[Path], None]


def discover_project_instructions(
    workspace: Path,
    project_path: Path,
    *,
    max_bytes: int = 32_768,
    validator: ProjectFileValidator | None = None,
    skip_content: Callable[[Path, str], bool] | None = None,
) -> str:
    """Prefer AGENTS.override.md to AGENTS.md at each authorized directory."""
    project = project_path.expanduser().resolve()
    base = workspace.expanduser().resolve()
    root = next(
        (directory for directory in (project, *project.parents) if (directory / ".git").exists()),
        base if project.is_relative_to(base) else project,
    )
    relative = project.relative_to(root)
    parts: list[str] = []
    remaining = max_bytes
    for level in range(len(relative.parts) + 1):
        directory = root.joinpath(*relative.parts[:level])
        for name in ("AGENTS.override.md", "AGENTS.md"):
            candidate = directory / name
            if not candidate.exists() and not candidate.is_symlink():
                continue
            if not candidate.is_file():
                break
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                break
            if not resolved.is_relative_to(directory):
                break
            if validator is not None:
                try:
                    validator(candidate)
                except PermissionError:
                    break
            label = candidate.relative_to(root).as_posix()
            separator = "\n\n" if parts else ""
            header = f"{separator}## {label}\n\n"
            room = remaining - len(header.encode("utf-8"))
            if room <= 0:
                return "".join(parts)
            try:
                with resolved.open("rb") as stream:
                    raw = stream.read(room + 1)
            except OSError:
                break
            if len(raw) > room:
                truncated = raw[:room].decode("utf-8", errors="ignore")
                hint = "\n\n[Project instructions truncated. Use read_file for the full file.]"
                allowance = max(0, room - len(hint.encode("utf-8")))
                text = raw[:allowance].decode("utf-8", errors="ignore").rstrip() + hint
                parts.append(header + (text if room >= len(hint) else truncated))
                return "".join(parts)
            text = raw.decode("utf-8", errors="replace").strip()
            if text and (skip_content is None or not skip_content(candidate, text)):
                section = header + text
                parts.append(section)
                remaining -= len(section.encode("utf-8"))
            break
    return "".join(parts)
