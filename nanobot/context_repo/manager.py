"""Discovery and runtime view of managed context repositories."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, TypeGuard, cast

from loguru import logger

_MANIFEST_NAME = "nanobot.context.json"
_MISSING = object()


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Context repo {name} must be a boolean")
    return value


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return cast(list[object], value)
    return [value]


def _is_manifest_mapping(value: object) -> TypeGuard[dict[str, Any]]:
    """JSON object boundary; values remain dynamic until their owning accessor."""
    if not isinstance(value, dict):
        return False
    if not all(isinstance(key, str) for key in cast(dict[object, object], value)):
        raise ValueError("Context repo manifest object keys must be strings")
    return True


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


def _resolve_declared_path(base: Path, raw_path: str) -> Path:
    path = Path(os.path.expandvars(raw_path)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


@dataclass
class ManagedTargetRepo:
    """Runtime metadata for a code/content repo managed by a context repo."""

    name: str
    context_repo_name: str
    context_repo_path: Path
    description: str = ""
    type: str = "codebase"
    path: Path | None = None
    path_env: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    resolved_from: str | None = None

    @classmethod
    def from_manifest(
        cls,
        *,
        name: str,
        context_repo_name: str,
        context_repo_path: Path,
        raw: Any,
    ) -> "ManagedTargetRepo | None":
        if not _is_manifest_mapping(raw):
            return None

        path_env = _get_any(raw, "pathEnv", "path_env")
        path: Path | None = None
        resolved_from: str | None = None
        if declared_path := raw.get("path"):
            path = _resolve_declared_path(context_repo_path, str(declared_path))
            resolved_from = "path"
        elif path_env and os.environ.get(str(path_env)):
            path = _resolve_declared_path(context_repo_path, os.environ[str(path_env)])
            resolved_from = f"env:{path_env}"
        elif default_path := _get_any(raw, "pathDefault", "path_default"):
            candidate = _resolve_declared_path(context_repo_path, str(default_path))
            if candidate.exists():
                path = candidate
                resolved_from = "pathDefault"

        return cls(
            name=name,
            context_repo_name=context_repo_name,
            context_repo_path=context_repo_path,
            description=str(raw.get("description") or ""),
            type=str(raw.get("type") or "codebase"),
            path=path,
            path_env=str(path_env) if path_env else None,
            manifest=raw,
            resolved_from=resolved_from,
        )

    @property
    def resolved(self) -> bool:
        return self.path is not None

    @property
    def commands(self) -> dict[str, Any]:
        value = self.manifest.get("commands")
        return value if _is_manifest_mapping(value) else {}

    @property
    def publish_config(self) -> dict[str, Any]:
        value = self.manifest.get("publish")
        return value if _is_manifest_mapping(value) else {}

    def rel_path(self, path: Path) -> str | None:
        if self.path is None:
            return None
        return _rel(path, self.path)

    def matches(self, rel_path: str, patterns: Iterable[str]) -> bool:
        return _match_any(rel_path, patterns)

    def read_patterns(self) -> list[str]:
        return _as_str_list(_get_any(self.manifest, "read", "readable", default=["**"]))

    def write_patterns(self) -> list[str]:
        return _as_str_list(_get_any(self.manifest, "write", "writable", default=[]))

    def protected_patterns(self) -> list[str]:
        defaults = [
            ".env",
            ".env.*",
            "**/*oauth*",
            "**/*secret*",
            "**/*token*",
        ]
        return [*defaults, *_as_str_list(self.manifest.get("protected"))]

    def proposal_required_patterns(self) -> list[str]:
        return _as_str_list(
            _get_any(self.manifest, "proposalRequired", "proposal_required", default=[])
        )

    def is_readable(self, rel_path: str) -> bool:
        return self.matches(rel_path, self.read_patterns())

    def is_writable(self, rel_path: str) -> bool:
        return self.matches(rel_path, self.write_patterns())

    def is_protected(self, rel_path: str) -> bool:
        return self.matches(rel_path, self.protected_patterns())

    def requires_proposal(self, rel_path: str) -> bool:
        return self.matches(rel_path, self.proposal_required_patterns())

    def read_roots(self) -> list[Path]:
        if self.path is None:
            return []
        return _unique_paths(_pattern_root(self.path, pattern) for pattern in self.read_patterns())

    def write_roots(self) -> list[Path]:
        if self.path is None:
            return []
        return _unique_paths(_pattern_root(self.path, pattern) for pattern in self.write_patterns())


@dataclass(frozen=True)
class ContextRepoRuntimeConfig:
    """A context repo entry from user config, normalized for runtime use."""

    path: Path
    read_only: bool = False
    auto_sync: bool = True
    credential_profile: str | None = None
    manifest_name: str = _MANIFEST_NAME
    target_repo_paths: dict[str, str] = field(default_factory=dict)
    allow_exec: bool = True

    @classmethod
    def from_raw(cls, raw: Any) -> "ContextRepoRuntimeConfig | None":
        if raw is None:
            return None
        if isinstance(raw, (str, Path)):
            return cls(path=Path(raw).expanduser().resolve(strict=False))
        if _is_manifest_mapping(raw):
            allow_exec = _strict_bool(_get_any(raw, "allowExec", "allow_exec", default=True), "allowExec")
            path = raw.get("path")
            if not path:
                return None
            target_paths: Any = _get_any(raw, "targetRepoPaths", "target_repo_paths", default={}) or {}
            return cls(
                path=Path(str(path)).expanduser().resolve(strict=False),
                read_only=bool(_get_any(raw, "readOnly", "read_only", default=False)),
                allow_exec=allow_exec,
                auto_sync=bool(_get_any(raw, "autoSync", "auto_sync", default=True)),
                credential_profile=_get_any(raw, "credentialProfile", "credential_profile"),
                manifest_name=str(_get_any(raw, "manifest", "manifestName", "manifest_name", default=_MANIFEST_NAME)),
                target_repo_paths={
                    str(name): str(path)
                    for name, path in target_paths.items()
                },
            )
        allow_exec = getattr(raw, "allow_exec", _MISSING)
        if allow_exec is _MISSING:
            allow_exec = getattr(raw, "allowExec", True)
        normalized_allow_exec = _strict_bool(allow_exec, "allowExec")
        path = getattr(raw, "path", None)
        if not path:
            return None
        return cls(
            path=Path(str(path)).expanduser().resolve(strict=False),
            read_only=bool(getattr(raw, "read_only", getattr(raw, "readOnly", False))),
            allow_exec=normalized_allow_exec,
            auto_sync=bool(getattr(raw, "auto_sync", getattr(raw, "autoSync", True))),
            credential_profile=getattr(raw, "credential_profile", getattr(raw, "credentialProfile", None)),
            manifest_name=str(getattr(raw, "manifest", getattr(raw, "manifest_name", _MANIFEST_NAME))),
            target_repo_paths={
                str(name): str(path)
                for name, path in (getattr(raw, "target_repo_paths", getattr(raw, "targetRepoPaths", {})) or {}).items()
            },
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
    target_repo_paths: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None
    allow_exec: bool = True

    @classmethod
    def load(cls, config: ContextRepoRuntimeConfig) -> "ManagedContextRepo":
        manifest_path = config.path / config.manifest_name
        manifest: dict[str, Any] = {}
        managed = manifest_path.is_file()
        if config.manifest_name != _MANIFEST_NAME and not managed:
            raise ValueError(f"Configured context repo manifest is missing or not a file: {manifest_path}")
        if managed:
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not _is_manifest_mapping(raw):
                    raise ValueError("Context repo manifest must be an object")
                manifest = raw
            except Exception as exc:
                logger.warning("Failed to load context repo manifest {}: {}", manifest_path, exc)
                raise ValueError(f"Invalid context repo manifest: {manifest_path}") from exc
        name = str(manifest.get("name") or config.path.name)
        credential_profile = config.credential_profile or manifest.get("credentialProfile") or manifest.get("credential_profile")
        return cls(
            path=config.path,
            name=name,
            managed=managed,
            read_only=config.read_only,
            allow_exec=config.allow_exec,
            auto_sync=config.auto_sync,
            credential_profile=credential_profile,
            target_repo_paths=config.target_repo_paths,
            manifest=manifest,
            manifest_path=manifest_path if managed else None,
        )

    @property
    def entrypoints(self) -> dict[str, Any]:
        value = self.manifest.get("entrypoints")
        return value if _is_manifest_mapping(value) else {}

    @property
    def sync_config(self) -> dict[str, Any]:
        value = self.manifest.get("sync")
        return value if _is_manifest_mapping(value) else {}

    @property
    def stores(self) -> dict[str, Any]:
        value = self.manifest.get("stores")
        return value if _is_manifest_mapping(value) else {}

    @property
    def modules(self) -> dict[str, Any]:
        value = self.manifest.get("modules")
        return value if _is_manifest_mapping(value) else {}

    @property
    def user_experience(self) -> dict[str, Any]:
        value = _get_any(self.manifest, "userExperience", "user_experience", default={})
        return value if _is_manifest_mapping(value) else {}

    @property
    def tools(self) -> dict[str, Any]:
        value = self.manifest.get("tools")
        return value if _is_manifest_mapping(value) else {}

    @property
    def target_repos_config(self) -> dict[str, Any]:
        value = _get_any(self.manifest, "targetRepos", "target_repos", default={})
        return value if _is_manifest_mapping(value) else {}

    def target_repos(self) -> list[ManagedTargetRepo]:
        targets: list[ManagedTargetRepo] = []
        for name, raw in self.target_repos_config.items():
            if _is_manifest_mapping(raw) and (override_path := self.target_repo_paths.get(str(name))):
                raw = {**raw, "path": override_path}
            target = ManagedTargetRepo.from_manifest(
                name=str(name),
                context_repo_name=self.name,
                context_repo_path=self.path,
                raw=raw,
            )
            if target:
                targets.append(target)
        return targets

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
            if not _is_manifest_mapping(store):
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
        roots = [self.path]
        for target in self.target_repos():
            roots.extend(target.read_roots())
        return _unique_paths(roots)

    def write_roots(self) -> list[Path]:
        if self.read_only:
            return []
        roots = [_pattern_root(self.path, pattern) for pattern in self.writable_patterns()]
        for target in self.target_repos():
            roots.extend(target.write_roots())
        return _unique_paths(roots)

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
            if _is_manifest_mapping(store):
                patterns.extend(_as_str_list(_get_any(store, "syncPaths", "sync_paths", default=[])))
        return sorted(set(patterns))

    def sync_exclude_patterns(self) -> list[str]:
        return sorted(set([
            *self.protected_patterns(),
            *_as_str_list(_get_any(self.sync_config, "neverCommit", "never_commit", default=[])),
        ]))

    def auto_push_enabled(self) -> bool:
        if not self.auto_sync:
            return False
        return bool(_get_any(self.sync_config, "autoPush", "auto_push", default=True))

    def prompt_summary(self) -> str:
        if self.managed:
            lines = [f"- {self.name}: managed context repo at {self.path}"]
        else:
            lines = [
                f"- {self.name}: legacy context repo at {self.path}. "
                "Root markdown, memory, and skills are loaded when present."
            ]
        if self.read_only:
            lines.append("  File access is read-only; do not modify this repository.")
        if not self.allow_exec:
            lines.append(
                "  Shell commands cannot access this repository, even for reads. "
                "Use read_file or list_dir for permitted file inspection; "
                "do not retry a denied command through another shell or interpreter."
            )
        if not self.managed:
            return "\n".join(lines)
        if self.user_experience:
            mode = str(_get_any(self.user_experience, "audienceMode", "audience_mode", default="")).strip()
            assistant_name = str(_get_any(self.user_experience, "assistantName", "assistant_name", default="assistant")).strip()
            internals = str(_get_any(self.user_experience, "explainInternals", "explain_internals", default="")).strip()
            approvals = str(_get_any(self.user_experience, "approvalStyle", "approval_style", default="")).strip()
            experience_bits: list[str] = []
            if mode:
                experience_bits.append(f"audience={mode}")
            if assistant_name:
                experience_bits.append(f"assistant term={assistant_name}")
            if internals:
                experience_bits.append(f"internals={internals}")
            if approvals:
                experience_bits.append(f"approvals={approvals}")
            if experience_bits:
                lines.append(f"  User experience: {', '.join(experience_bits)}. Use business-facing language unless the user asks for technical detail.")
        if self.modules:
            module_lines: list[str] = []
            for name, module in sorted(self.modules.items()):
                description = ""
                if _is_manifest_mapping(module):
                    description = str(module.get("description") or "").strip()
                module_file = self.path / "modules" / str(name) / "MODULE.md"
                location = f"; contract: {module_file}" if module_file.is_file() else ""
                suffix = f" - {description}" if description else ""
                module_lines.append(f"{name}{suffix}{location}")
            lines.append(f"  Modules: {'; '.join(module_lines)}")
        if self.stores:
            store_lines: list[str] = []
            for name, store in sorted(self.stores.items()):
                if _is_manifest_mapping(store):
                    direct = _get_any(store, "directEdit", "direct_edit", default=True)
                    via = _get_any(store, "writeVia", "write_via", default="")
                    suffix = f" via {via}" if via else ""
                    store_lines.append(f"{name} (directEdit={str(bool(direct)).lower()}{suffix})")
                else:
                    store_lines.append(name)
            lines.append(f"  Stores: {', '.join(store_lines)}")
        if self.tools:
            lines.append(f"  Tools: {', '.join(sorted(self.tools))}")
        targets = self.target_repos()
        if targets:
            target_lines: list[str] = []
            for target in targets:
                if target.resolved:
                    write = ", ".join(target.write_patterns()) or "none"
                    commands = ", ".join(sorted(target.commands)) or "none"
                    target_lines.append(
                        f"{target.name} ({target.type}) at {target.path}; writes: {write}; commands: {commands}"
                    )
                else:
                    hint = f"set {target.path_env}" if target.path_env else "path unresolved"
                    target_lines.append(f"{target.name} ({target.type}) unresolved; {hint}")
            lines.append(f"  Target repos: {'; '.join(target_lines)}")
        if not self.read_only:
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
        configs: dict[Path, ContextRepoRuntimeConfig] = {}
        for raw in context_repos or []:
            config = ContextRepoRuntimeConfig.from_raw(raw)
            if config is None:
                continue
            previous = configs.get(config.path)
            if previous is None:
                configs[config.path] = config
            elif previous.allow_exec and not config.allow_exec:
                # Keep first-entry precedence without discarding an explicit denial.
                configs[config.path] = replace(previous, allow_exec=False)
        for path in context_paths or []:
            config = ContextRepoRuntimeConfig.from_raw(path)
            if config and config.path not in configs:
                configs[config.path] = config
        return cls([ManagedContextRepo.load(config) for config in configs.values()])

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

    def target_repos(self) -> list[ManagedTargetRepo]:
        targets: list[ManagedTargetRepo] = []
        for repo in self.repos:
            targets.extend(repo.target_repos())
        return targets

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

    def find_target_repo_for_path(self, path: Path) -> ManagedTargetRepo | None:
        resolved = path.resolve(strict=False)
        matches = [target for target in self.target_repos() if target.rel_path(resolved) is not None]
        if not matches:
            return None
        return max(matches, key=lambda target: len(str(target.path or "")))

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
