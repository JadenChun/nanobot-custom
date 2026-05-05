"""Discovery and runtime view of managed context repositories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable

from loguru import logger


_MANIFEST_NAME = "nanobot.context.json"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item).strip()]


def _get_any(data: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


def _rel(path: Path, root: Path) -> str | None:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return None


def _match_any(rel_path: str, patterns: Iterable[str]) -> bool:
    rel_path = rel_path.strip("/")
    for raw_pattern in patterns:
        pattern = raw_pattern.strip().strip("/")
        if not pattern:
            continue
        if pattern.endswith("/**"):
            base = pattern[:-3].strip("/")
            if rel_path == base or rel_path.startswith(base + "/"):
                return True
        if fnmatch(rel_path, pattern):
            return True
    return False


def _pattern_root(repo_root: Path, pattern: str) -> Path:
    """Return the stable directory prefix for a glob-like repo-relative pattern."""
    cleaned = pattern.strip().strip("/")
    if not cleaned:
        return repo_root
    pieces: list[str] = []
    for part in cleaned.split("/"):
        if any(ch in part for ch in "*?["):
            break
        pieces.append(part)
    if cleaned.endswith("/**") and pieces:
        return repo_root.joinpath(*pieces)
    if pieces and "." in pieces[-1]:
        pieces = pieces[:-1]
    return repo_root.joinpath(*pieces) if pieces else repo_root


@dataclass(frozen=True)
class ContextRepoRuntimeConfig:
    """A context repo entry from user config, normalized for runtime use."""

    path: Path
    read_only: bool = False
    auto_sync: bool = True
    credential_profile: str | None = None
    manifest_name: str = _MANIFEST_NAME

    @classmethod
    def from_raw(cls, raw: Any) -> "ContextRepoRuntimeConfig | None":
        if raw is None:
            return None
        if isinstance(raw, (str, Path)):
            return cls(path=Path(raw).expanduser().resolve(strict=False))
        if isinstance(raw, dict):
            path = raw.get("path")
            if not path:
                return None
            return cls(
                path=Path(str(path)).expanduser().resolve(strict=False),
                read_only=bool(_get_any(raw, "readOnly", "read_only", default=False)),
                auto_sync=bool(_get_any(raw, "autoSync", "auto_sync", default=True)),
                credential_profile=_get_any(raw, "credentialProfile", "credential_profile"),
                manifest_name=str(_get_any(raw, "manifest", "manifestName", "manifest_name", default=_MANIFEST_NAME)),
            )
        path = getattr(raw, "path", None)
        if not path:
            return None
        return cls(
            path=Path(str(path)).expanduser().resolve(strict=False),
            read_only=bool(getattr(raw, "read_only", getattr(raw, "readOnly", False))),
            auto_sync=bool(getattr(raw, "auto_sync", getattr(raw, "autoSync", True))),
            credential_profile=getattr(raw, "credential_profile", getattr(raw, "credentialProfile", None)),
            manifest_name=str(getattr(raw, "manifest", getattr(raw, "manifest_name", _MANIFEST_NAME))),
        )


@dataclass
class ManagedContextRepo:
    """Runtime metadata and helpers for one context repository."""

    path: Path
    name: str
    managed: bool
    read_only: bool = False
    auto_sync: bool = True
    credential_profile: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None

    @classmethod
    def load(cls, config: ContextRepoRuntimeConfig) -> "ManagedContextRepo":
        manifest_path = config.path / config.manifest_name
        manifest: dict[str, Any] = {}
        managed = manifest_path.is_file()
        if managed:
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    manifest = raw
            except Exception as exc:
                logger.warning("Failed to load context repo manifest {}: {}", manifest_path, exc)
                managed = False
        name = str(manifest.get("name") or config.path.name)
        credential_profile = config.credential_profile or manifest.get("credentialProfile") or manifest.get("credential_profile")
        return cls(
            path=config.path,
            name=name,
            managed=managed,
            read_only=config.read_only,
            auto_sync=config.auto_sync,
            credential_profile=credential_profile,
            manifest=manifest,
            manifest_path=manifest_path if managed else None,
        )

    @property
    def entrypoints(self) -> dict[str, Any]:
        value = self.manifest.get("entrypoints") or {}
        return value if isinstance(value, dict) else {}

    @property
    def sync_config(self) -> dict[str, Any]:
        value = self.manifest.get("sync") or {}
        return value if isinstance(value, dict) else {}

    @property
    def stores(self) -> dict[str, Any]:
        value = self.manifest.get("stores") or {}
        return value if isinstance(value, dict) else {}

    @property
    def modules(self) -> dict[str, Any]:
        value = self.manifest.get("modules") or {}
        return value if isinstance(value, dict) else {}

    @property
    def tools(self) -> dict[str, Any]:
        value = self.manifest.get("tools") or {}
        return value if isinstance(value, dict) else {}

    def rel_path(self, path: Path) -> str | None:
        return _rel(path, self.path)

    def matches(self, rel_path: str, patterns: Iterable[str]) -> bool:
        return _match_any(rel_path, patterns)

    def protected_patterns(self) -> list[str]:
        defaults = [
            ".env",
            ".env.*",
            "config/.env",
            "config/**/*.env",
            "config/**/*oauth*.json",
            "config/**/*secret*.json",
            "config/**/*token*.json",
            "**/*oauth*.json",
            "**/*secret*.json",
            "**/*token*.json",
            ".claude/**",
            ".nanobot/**",
        ]
        return [*defaults, *_as_str_list(self.manifest.get("protected"))]

    def writable_patterns(self) -> list[str]:
        if self.read_only:
            return []
        if not self.managed:
            return ["**"]
        defaults = [
            "runs/**",
            "outputs/**",
            "feedback/**",
            "proposals/**",
            "skills/**",
            "modules/**",
            "stores/**",
        ]
        return [*defaults, *_as_str_list(self.manifest.get("writable"))]

    def proposal_required_patterns(self) -> list[str]:
        return _as_str_list(
            _get_any(self.manifest, "proposalRequired", "proposal_required", default=[])
        )

    def store_paths(self, *, direct_edit_only: bool | None = None) -> list[str]:
        paths: list[str] = []
        for store in self.stores.values():
            if not isinstance(store, dict):
                continue
            direct_edit = bool(_get_any(store, "directEdit", "direct_edit", default=True))
            if direct_edit_only is not None and direct_edit != direct_edit_only:
                continue
            raw_path = store.get("path")
            if raw_path:
                paths.append(str(raw_path))
        return paths

    def is_protected(self, rel_path: str) -> bool:
        return self.matches(rel_path, self.protected_patterns())

    def is_writable(self, rel_path: str) -> bool:
        return self.matches(rel_path, self.writable_patterns())

    def requires_proposal(self, rel_path: str) -> bool:
        return self.matches(rel_path, self.proposal_required_patterns())

    def blocks_direct_store_edit(self, rel_path: str) -> bool:
        return self.matches(rel_path, self.store_paths(direct_edit_only=False))

    def skill_roots(self) -> list[Path]:
        roots: list[Path] = []
        default = self.path / "skills"
        if default.is_dir():
            roots.append(default)
        for pattern in _as_str_list(self.entrypoints.get("skills")):
            if pattern.rstrip("/") == "skills":
                roots.append(self.path / "skills")
                continue
            for skill_file in self.path.glob(pattern):
                if skill_file.name == "SKILL.md" and skill_file.parent.parent not in roots:
                    roots.append(skill_file.parent.parent)
        return _unique_paths(roots)

    def read_roots(self) -> list[Path]:
        return [self.path]

    def write_roots(self) -> list[Path]:
        if self.read_only:
            return []
        return _unique_paths(_pattern_root(self.path, pattern) for pattern in self.writable_patterns())

    def context_files(self) -> list[Path]:
        files: list[Path] = []
        entry_context = self.entrypoints.get("context") or self.manifest.get("context")
        if entry_context:
            files.append(self.path / str(entry_context))
        elif (self.path / "CONTEXT.md").is_file():
            files.append(self.path / "CONTEXT.md")

        if not self.managed:
            for md_file in sorted(self.path.glob("*.md")):
                if md_file.name not in {"README.md", "CONTEXT.md"}:
                    files.append(md_file)
        return _unique_paths(path for path in files if path.is_file())

    def memory_files(self) -> list[Path]:
        memory = self.path / "memory" / "MEMORY.md"
        return [memory] if memory.is_file() else []

    def run_summary_files(self) -> list[Path]:
        candidates = [
            self.path / "runs" / "summaries.md",
            self.path / "runs" / "current.json",
            self.path / "workspace" / "current" / "summary_index.md",
            self.path / "workspace" / "current" / "plan.json",
        ]
        return [path for path in candidates if path.is_file()]

    def sync_include_patterns(self) -> list[str] | None:
        if not self.managed:
            return None
        include = _as_str_list(_get_any(self.sync_config, "include", "paths", default=[]))
        patterns = [*include, *self.writable_patterns()]
        for store in self.stores.values():
            if isinstance(store, dict):
                patterns.extend(_as_str_list(_get_any(store, "syncPaths", "sync_paths", default=[])))
        return sorted(set(patterns))

    def sync_exclude_patterns(self) -> list[str]:
        return sorted(set([
            *self.protected_patterns(),
            *_as_str_list(_get_any(self.sync_config, "neverCommit", "never_commit", default=[])),
        ]))

    def prompt_summary(self) -> str:
        if not self.managed:
            return (
                f"- {self.name}: legacy context repo at {self.path}. "
                "Root markdown, memory, and skills are loaded when present."
            )
        lines = [f"- {self.name}: managed context repo at {self.path}"]
        if self.modules:
            lines.append(f"  Modules: {', '.join(sorted(self.modules))}")
        if self.stores:
            store_lines = []
            for name, store in sorted(self.stores.items()):
                if isinstance(store, dict):
                    direct = _get_any(store, "directEdit", "direct_edit", default=True)
                    via = _get_any(store, "writeVia", "write_via", default="")
                    suffix = f" via {via}" if via else ""
                    store_lines.append(f"{name} (directEdit={str(bool(direct)).lower()}{suffix})")
                else:
                    store_lines.append(name)
            lines.append(f"  Stores: {', '.join(store_lines)}")
        if self.tools:
            lines.append(f"  Tools: {', '.join(sorted(self.tools))}")
        lines.append("  Normal writes to managed paths are autonomous; protected paths and strategic changes require approval/proposals.")
        return "\n".join(lines)


class ContextRepoManager:
    """Runtime index of configured context repositories."""

    def __init__(self, repos: list[ManagedContextRepo] | None = None):
        self.repos = repos or []

    @classmethod
    def from_config(
        cls,
        *,
        context_paths: Iterable[Path | str] | None = None,
        context_repos: Iterable[Any] | None = None,
    ) -> "ContextRepoManager":
        configs: list[ContextRepoRuntimeConfig] = []
        seen: set[Path] = set()
        for raw in context_repos or []:
            config = ContextRepoRuntimeConfig.from_raw(raw)
            if config and config.path not in seen:
                configs.append(config)
                seen.add(config.path)
        for path in context_paths or []:
            config = ContextRepoRuntimeConfig.from_raw(path)
            if config and config.path not in seen:
                configs.append(config)
                seen.add(config.path)
        return cls([ManagedContextRepo.load(config) for config in configs])

    @property
    def paths(self) -> list[Path]:
        return [repo.path for repo in self.repos]

    def skill_roots(self) -> list[Path]:
        roots: list[Path] = []
        for repo in self.repos:
            roots.extend(repo.skill_roots())
        return _unique_paths(roots)

    def read_roots(self) -> list[Path]:
        roots: list[Path] = []
        for repo in self.repos:
            roots.extend(repo.read_roots())
        return _unique_paths(roots)

    def write_roots(self) -> list[Path]:
        roots: list[Path] = []
        for repo in self.repos:
            roots.extend(repo.write_roots())
        return _unique_paths(roots)

    def context_files(self) -> list[Path]:
        files: list[Path] = []
        for repo in self.repos:
            files.extend(repo.context_files())
        return _unique_paths(files)

    def memory_files(self) -> list[Path]:
        files: list[Path] = []
        for repo in self.repos:
            files.extend(repo.memory_files())
        return _unique_paths(files)

    def run_summary_files(self) -> list[Path]:
        files: list[Path] = []
        for repo in self.repos:
            files.extend(repo.run_summary_files())
        return _unique_paths(files)

    def find_repo_for_path(self, path: Path) -> ManagedContextRepo | None:
        resolved = path.resolve(strict=False)
        matches = [repo for repo in self.repos if repo.rel_path(resolved) is not None]
        if not matches:
            return None
        return max(matches, key=lambda repo: len(str(repo.path)))

    def prompt_summary(self) -> str:
        if not self.repos:
            return ""
        return "\n".join(repo.prompt_summary() for repo in self.repos)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result
