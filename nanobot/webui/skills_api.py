"""Lightweight skill summaries for the WebUI."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from nanobot.agent.plugins import enabled_agent_plugin_skills
from nanobot.agent.skills import BUILTIN_SKILLS_DIR, parse_skill_metadata
from nanobot.config.loader import load_config, save_config
from nanobot.context_repo.manager import ContextRepoManager
from nanobot.context_repo.policy import ResourceAccessPolicy
from nanobot.security.workspace_policy import WorkspaceBoundaryError, require_path_within

_RESTRICTED_CATALOG_NOTICE = (
    "Some configured skill sources could not be checked. "
    "Protected or unreadable skills cannot be listed or used."
)
_UNVERIFIED_SKILL_REASON = (
    "A higher-priority configured skill library could not be checked."
)


class SkillManagementError(Exception):
    """A safe skill-management error for the WebUI."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class SkillInventoryEntry:
    """One readable skill and the source that wins runtime name precedence."""

    name: str
    source: str
    path: Path
    content: str
    metadata: dict[str, object] | None
    verification: str


@dataclass(frozen=True)
class SkillInventory:
    """Policy-filtered management view of the configured skill catalog."""

    entries: tuple[SkillInventoryEntry, ...]
    catalog_complete: bool

    @property
    def catalog_notice(self) -> str | None:
        return None if self.catalog_complete else _RESTRICTED_CATALOG_NOTICE

    def find(self, name: str) -> SkillInventoryEntry | None:
        return next((entry for entry in self.entries if entry.name == name), None)


@dataclass(frozen=True)
class _SkillRoot:
    path: Path | None
    source: str
    explicit: bool = False


def webui_skills_payload(
    workspace_path: Path,
    *,
    disabled_skills: set[str] | None = None,
    context_manager: ContextRepoManager | None = None,
    resource_policy: ResourceAccessPolicy | None = None,
    extra_skill_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Return agent skills without leaking local filesystem paths."""
    inventory = build_skill_inventory(
        workspace_path,
        context_manager=context_manager,
        resource_policy=resource_policy,
        extra_skill_paths=extra_skill_paths,
    )
    entries = sorted(
        inventory.entries,
        key=lambda entry: (entry.source != "workspace", entry.name),
    )
    payload: dict[str, Any] = {
        "skills": [
            _skill_payload(
                entry,
                disabled_skills=disabled_skills,
                resource_policy=resource_policy,
            )
            for entry in entries
        ],
        "catalog_complete": inventory.catalog_complete,
    }
    if inventory.catalog_notice is not None:
        payload["catalog_notice"] = inventory.catalog_notice
    return payload


def webui_skill_detail_payload(
    workspace_path: Path,
    name: str,
    *,
    disabled_skills: set[str] | None = None,
    context_manager: ContextRepoManager | None = None,
    resource_policy: ResourceAccessPolicy | None = None,
    extra_skill_paths: list[Path] | None = None,
) -> dict[str, Any] | None:
    """Return a single skill's safe detail payload."""
    inventory = build_skill_inventory(
        workspace_path,
        context_manager=context_manager,
        resource_policy=resource_policy,
        extra_skill_paths=extra_skill_paths,
    )
    entry = inventory.find(name)
    if entry is None:
        return None
    return {
        **_skill_payload(
            entry,
            disabled_skills=disabled_skills,
            resource_policy=resource_policy,
        ),
        "requirements": _requirements_payload(entry.metadata),
        "install_options": _install_options(_metadata_for_display(entry.metadata)),
        "raw_markdown": entry.content,
    }


def set_webui_skill_enabled(
    workspace_path: Path,
    name: str,
    *,
    enabled: bool,
    disabled_skills: set[str],
    config_path: Path | None = None,
    context_manager: ContextRepoManager | None = None,
    resource_policy: ResourceAccessPolicy | None = None,
    extra_skill_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Persist and apply one skill's enabled state."""
    _require_skill_entry(
        workspace_path,
        name,
        context_manager=context_manager,
        resource_policy=resource_policy,
        extra_skill_paths=extra_skill_paths,
    )
    config = load_config(config_path)
    next_disabled = set(config.agents.defaults.disabled_skills)
    if enabled:
        next_disabled.discard(name)
    else:
        next_disabled.add(name)
    if next_disabled != set(config.agents.defaults.disabled_skills):
        config.agents.defaults.disabled_skills = sorted(next_disabled)
        save_config(config, config_path)
    disabled_skills.clear()
    disabled_skills.update(next_disabled)
    return {"name": name, "enabled": enabled, "deleted": False}


def delete_webui_skill(
    workspace_path: Path,
    name: str,
    *,
    disabled_skills: set[str],
    config_path: Path | None = None,
    context_manager: ContextRepoManager | None = None,
    resource_policy: ResourceAccessPolicy | None = None,
    extra_skill_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Delete one workspace skill and remove its disabled-state entry."""
    workspace = workspace_path.expanduser().resolve()
    if (workspace / "skills").is_symlink():
        raise SkillManagementError("skills directory must stay inside the workspace", status=403)
    entry = _require_skill_entry(
        workspace_path,
        name,
        context_manager=context_manager,
        resource_policy=resource_policy,
        extra_skill_paths=extra_skill_paths,
    )
    if entry.source != "workspace":
        raise SkillManagementError("only workspace skills can be deleted", status=403)
    if not _skill_tree_is_editable(
        resource_policy,
        entry.path.parent.parent,
        entry.path.parent,
    ):
        raise SkillManagementError(
            "skill cannot be deleted with current access",
            status=403,
        )

    try:
        skills_root = require_path_within(
            workspace / "skills",
            workspace,
            message="skills directory must stay inside the workspace",
        )
    except WorkspaceBoundaryError as exc:
        raise SkillManagementError(str(exc), status=403) from exc
    target = skills_root / name
    if target.parent != skills_root:
        raise SkillManagementError("invalid skill name")
    if not target.is_symlink() and not target.is_dir():
        raise SkillManagementError("skill directory was not found", status=404)

    config = load_config(config_path)
    original_disabled = list(config.agents.defaults.disabled_skills)
    next_disabled = set(original_disabled)
    if name in next_disabled:
        next_disabled.remove(name)
    with tempfile.TemporaryDirectory(prefix=".nanobot-delete-", dir=skills_root) as staging:
        staged_target = Path(staging) / name
        target.replace(staged_target)
        try:
            if next_disabled != set(original_disabled):
                config.agents.defaults.disabled_skills = sorted(next_disabled)
                save_config(config, config_path)
        except Exception:
            config.agents.defaults.disabled_skills = original_disabled
            staged_target.replace(target)
            raise
    disabled_skills.clear()
    disabled_skills.update(next_disabled)
    return {"name": name, "enabled": False, "deleted": True}


def _require_skill_entry(
    workspace_path: Path,
    name: str,
    *,
    context_manager: ContextRepoManager | None = None,
    resource_policy: ResourceAccessPolicy | None = None,
    extra_skill_paths: list[Path] | None = None,
) -> SkillInventoryEntry:
    if not name or "/" in name or "\\" in name:
        raise SkillManagementError("invalid skill name")
    entry = build_skill_inventory(
        workspace_path,
        context_manager=context_manager,
        resource_policy=resource_policy,
        extra_skill_paths=extra_skill_paths,
    ).find(name)
    if entry is None:
        raise SkillManagementError("skill not found", status=404)
    return entry


def build_skill_inventory(
    workspace_path: Path,
    *,
    context_manager: ContextRepoManager | None = None,
    resource_policy: ResourceAccessPolicy | None = None,
    extra_skill_paths: list[Path] | None = None,
) -> SkillInventory:
    """Build an authorized catalog without enumerating denied or globbed roots."""
    workspace = workspace_path.expanduser().resolve(strict=False)
    manager = context_manager or ContextRepoManager()
    roots, complete = _configured_skill_roots(manager)
    roots.extend(_SkillRoot(path, "library", explicit=True) for path in extra_skill_paths or [])
    roots.insert(0, _SkillRoot(workspace / "skills", "workspace"))

    entries: list[SkillInventoryEntry] = []
    reserved_names: set[str] = set()
    opaque_precedence = False
    for root in roots:
        if root.path is None:
            complete = False
            opaque_precedence = True
            continue
        prepared_root, root_state = _prepare_skill_root(
            root.path,
            explicit=root.explicit,
            resource_policy=resource_policy,
        )
        if root_state == "missing":
            continue
        if root_state != "ready" or prepared_root is None:
            complete = False
            opaque_precedence = True
            continue

        try:
            candidates = sorted(prepared_root.iterdir(), key=lambda path: path.name)
        except (OSError, RuntimeError, ValueError):
            complete = False
            opaque_precedence = True
            continue
        for candidate in candidates:
            name = candidate.name
            if name in reserved_names:
                continue
            try:
                candidate_mode = candidate.lstat().st_mode
            except OSError:
                reserved_names.add(name)
                complete = False
                continue
            if stat.S_ISLNK(candidate_mode):
                reserved_names.add(name)
                complete = False
                continue
            if not stat.S_ISDIR(candidate_mode):
                continue
            try:
                candidate_dir = candidate.resolve(strict=True)
                if not candidate_dir.is_relative_to(prepared_root):
                    reserved_names.add(name)
                    complete = False
                    continue
                _validate_skill_path(resource_policy, candidate_dir, "list")
            except Exception:
                reserved_names.add(name)
                complete = False
                continue

            skill_file = candidate / "SKILL.md"
            try:
                file_mode = skill_file.lstat().st_mode
            except FileNotFoundError:
                continue
            except OSError:
                reserved_names.add(name)
                complete = False
                continue
            reserved_names.add(name)
            if stat.S_ISLNK(file_mode) or not stat.S_ISREG(file_mode):
                complete = False
                continue
            try:
                canonical_file = skill_file.resolve(strict=True)
                if not canonical_file.is_relative_to(prepared_root):
                    complete = False
                    continue
                _validate_skill_path(resource_policy, canonical_file, "read")
                content = canonical_file.read_text(encoding="utf-8")
            except (OSError, UnicodeError, RuntimeError, ValueError, PermissionError):
                complete = False
                continue

            metadata = parse_skill_metadata(content)
            entries.append(
                SkillInventoryEntry(
                    name=name,
                    source=root.source,
                    path=canonical_file,
                    content=content,
                    metadata=metadata,
                    verification=(
                        "unverified"
                        if opaque_precedence and root.source != "workspace"
                        else "verified"
                    ),
                )
            )

    # Plugin discovery validates package structure and activation before returning
    # paths. Its workspace root is validated above, then each returned document is
    # independently checked by the same candidate reader.
    plugin_root = workspace / "plugins"
    plugin_path, plugin_state = _prepare_skill_root(
        plugin_root,
        explicit=False,
        resource_policy=resource_policy,
    )
    if plugin_state not in {"ready", "missing"}:
        complete = False
        opaque_precedence = True
    elif plugin_state == "ready" and plugin_path is not None:
        if not _preflight_plugin_tree(plugin_path, resource_policy):
            complete = False
            opaque_precedence = True
            plugin_skills = []
        else:
            try:
                plugin_skills = enabled_agent_plugin_skills(workspace)
            except Exception:
                plugin_skills = []
                complete = False
                opaque_precedence = True
        for name, skill_file in plugin_skills:
            if name in reserved_names:
                continue
            reserved_names.add(name)
            try:
                candidate_dir = skill_file.parent.resolve(strict=True)
                canonical_file = skill_file.resolve(strict=True)
                if (
                    not candidate_dir.is_relative_to(plugin_path)
                    or not canonical_file.is_relative_to(plugin_path)
                    or skill_file.is_symlink()
                    or skill_file.parent.is_symlink()
                ):
                    raise ValueError("plugin skill path is outside its root")
                _validate_skill_path(resource_policy, candidate_dir, "list")
                _validate_skill_path(resource_policy, canonical_file, "read")
                content = canonical_file.read_text(encoding="utf-8")
            except Exception:
                complete = False
                continue
            entries.append(
                SkillInventoryEntry(
                    name=name,
                    source="plugin",
                    path=canonical_file,
                    content=content,
                    metadata=parse_skill_metadata(content),
                    verification=("unverified" if opaque_precedence else "verified"),
                )
            )

    builtin_path, builtin_state = _prepare_skill_root(
        BUILTIN_SKILLS_DIR,
        explicit=False,
        resource_policy=None,
    )
    if builtin_state == "ready" and builtin_path is not None:
        try:
            builtin_candidates = sorted(builtin_path.iterdir(), key=lambda path: path.name)
        except (OSError, RuntimeError, ValueError):
            builtin_candidates = []
            complete = False
        for candidate in builtin_candidates:
            name = candidate.name
            if name in reserved_names:
                continue
            try:
                candidate_mode = candidate.lstat().st_mode
                if stat.S_ISLNK(candidate_mode):
                    complete = False
                    continue
                if not stat.S_ISDIR(candidate_mode):
                    continue
                candidate_dir = candidate.resolve(strict=True)
                if not candidate_dir.is_relative_to(builtin_path):
                    complete = False
                    continue
                skill_file = candidate / "SKILL.md"
                file_mode = skill_file.lstat().st_mode
                reserved_names.add(name)
                if stat.S_ISLNK(file_mode) or not stat.S_ISREG(file_mode):
                    complete = False
                    continue
                canonical_file = skill_file.resolve(strict=True)
                if not canonical_file.is_relative_to(builtin_path):
                    complete = False
                    continue
                content = canonical_file.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            except (OSError, UnicodeError, RuntimeError, ValueError):
                complete = False
                continue
            entries.append(
                SkillInventoryEntry(
                    name=name,
                    source="builtin",
                    path=canonical_file,
                    content=content,
                    metadata=parse_skill_metadata(content),
                    verification=("unverified" if opaque_precedence else "verified"),
                )
            )
    elif builtin_state == "error":
        complete = False

    return SkillInventory(tuple(entries), complete)


def _configured_skill_roots(
    context_manager: ContextRepoManager,
) -> tuple[list[_SkillRoot], bool]:
    """Resolve only supported literal skills entrypoints; never glob a repo."""
    roots: list[_SkillRoot] = []
    complete = True
    for repo in context_manager.repos:
        roots.append(_SkillRoot(repo.path / "skills", "library"))
        declared = repo.entrypoints.get("skills")
        patterns: list[object]
        if declared is None:
            patterns = []
        elif isinstance(declared, list):
            patterns = cast(list[object], declared)
        else:
            patterns = [declared]
        for raw_pattern in patterns:
            if not isinstance(raw_pattern, str) or not raw_pattern.strip():
                complete = False
                roots.append(_SkillRoot(None, "library", explicit=True))
                continue
            pattern = raw_pattern.strip().replace("\\", "/").rstrip("/")
            path = Path(pattern)
            if (
                path.is_absolute()
                or ".." in path.parts
                or any(char in pattern for char in "*?[")
            ):
                complete = False
                roots.append(_SkillRoot(None, "library", explicit=True))
                continue
            if pattern == "skills":
                roots.append(_SkillRoot(repo.path / "skills", "library", explicit=True))
            elif path.name == "SKILL.md" and len(path.parts) >= 2:
                roots.append(
                    _SkillRoot(repo.path.joinpath(*path.parts[:-2]), "library", explicit=True)
                )
            else:
                complete = False
                roots.append(_SkillRoot(None, "library", explicit=True))
    unique: list[_SkillRoot] = []
    seen: dict[Path, int] = {}
    for root in roots:
        if root.path is None:
            unique.append(root)
            continue
        canonical = root.path.expanduser().resolve(strict=False)
        if canonical in seen:
            existing = unique[seen[canonical]]
            if root.explicit and not existing.explicit:
                unique[seen[canonical]] = _SkillRoot(existing.path, existing.source, True)
            continue
        seen[canonical] = len(unique)
        unique.append(root)
    return unique, complete


def _prepare_skill_root(
    path: Path,
    *,
    explicit: bool,
    resource_policy: ResourceAccessPolicy | None,
) -> tuple[Path | None, str]:
    """Authorize a root before checking it or enumerating its children."""
    try:
        canonical = path.expanduser().resolve(strict=False)
        _validate_skill_path(resource_policy, canonical, "list")
        mode = path.expanduser().lstat().st_mode
        if stat.S_ISLNK(mode):
            return None, "error"
        if not stat.S_ISDIR(mode):
            return (None, "error") if explicit else (None, "missing")
        resolved = canonical.resolve(strict=True)
        return resolved, "ready"
    except FileNotFoundError:
        return (None, "error") if explicit else (None, "missing")
    except Exception:
        return None, "error"


def _validate_skill_path(
    resource_policy: ResourceAccessPolicy | None,
    path: Path,
    action: str,
) -> None:
    if resource_policy is not None:
        resource_policy.validate_path(
            path,
            action,
            managed=True,
            allow_folder_grants=resource_policy.folder_grants_allowed,
        )


def _preflight_plugin_tree(
    root: Path,
    resource_policy: ResourceAccessPolicy | None,
) -> bool:
    """Authorize every plugin path before plugin helpers enumerate or read it."""
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            _validate_skill_path(resource_policy, directory, "list")
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except Exception:
            return False
        for child in children:
            try:
                mode = child.lstat().st_mode
                if stat.S_ISLNK(mode):
                    return False
                canonical = child.resolve(strict=True)
                if not canonical.is_relative_to(root):
                    return False
                if stat.S_ISDIR(mode):
                    _validate_skill_path(resource_policy, canonical, "list")
                    pending.append(canonical)
                elif stat.S_ISREG(mode):
                    _validate_skill_path(resource_policy, canonical, "read")
                else:
                    return False
            except Exception:
                return False
    return True


def _skill_tree_is_editable(
    resource_policy: ResourceAccessPolicy | None,
    skills_root: Path,
    skill_root: Path,
) -> bool:
    """Check the full recursive delete target without reading file contents."""
    try:
        if stat.S_ISLNK(skills_root.lstat().st_mode) or stat.S_ISLNK(skill_root.lstat().st_mode):
            return False
        canonical_root = skills_root.resolve(strict=True)
        canonical_skill = skill_root.resolve(strict=True)
        if canonical_skill.parent != canonical_root or not canonical_skill.is_relative_to(
            canonical_root
        ):
            return False
        _validate_skill_path(resource_policy, canonical_root, "edit")
        _validate_skill_path(resource_policy, canonical_root, "list")
        pending = [canonical_skill]
        while pending:
            directory = pending.pop()
            _validate_skill_path(resource_policy, directory, "edit")
            _validate_skill_path(resource_policy, directory, "list")
            for child in directory.iterdir():
                mode = child.lstat().st_mode
                if stat.S_ISLNK(mode):
                    return False
                canonical = child.resolve(strict=True)
                if not canonical.is_relative_to(canonical_skill):
                    return False
                if stat.S_ISDIR(mode):
                    pending.append(canonical)
                elif stat.S_ISREG(mode):
                    _validate_skill_path(resource_policy, canonical, "edit")
                else:
                    return False
    except Exception:
        return False
    return True


def _skill_payload(
    entry: SkillInventoryEntry,
    *,
    disabled_skills: set[str] | None = None,
    resource_policy: ResourceAccessPolicy | None = None,
) -> dict[str, Any]:
    name = entry.name
    metadata = _metadata_for_display(entry.metadata)
    available, unavailable_reason = _skill_availability(entry.metadata)
    if entry.verification == "unverified":
        available = False
        unavailable_reason = _UNVERIFIED_SKILL_REASON
    source = entry.source
    deletable = source == "workspace"
    if deletable and resource_policy is not None:
        deletable = _skill_tree_is_editable(
            resource_policy,
            entry.path.parent.parent,
            entry.path.parent,
        )
    return {
        "name": name,
        "description": _description(metadata, name),
        "source": source,
        "enabled": name not in (disabled_skills or set()),
        "deletable": deletable,
        "available": available,
        "unavailable_reason": unavailable_reason,
        "verification": entry.verification,
    }


def _metadata_for_display(metadata: dict[str, object] | None) -> dict[str, Any] | None:
    return cast(dict[str, Any], metadata) if metadata is not None else None


def _skill_availability(metadata: dict[str, object] | None) -> tuple[bool, str]:
    meta = _nanobot_metadata(_metadata_for_display(metadata))
    requires_raw = meta.get("requires")
    requires = cast(dict[str, Any], requires_raw) if isinstance(requires_raw, dict) else {}
    bins, env = _requirements_from_metadata(requires)
    missing = [f"CLI: {name}" for name in bins if not shutil.which(name)]
    missing.extend(f"ENV: {name}" for name in env if not os.environ.get(name))
    return not missing, ", ".join(missing)


def _requirements_payload(metadata: dict[str, object] | None) -> dict[str, list[str]]:
    meta = _nanobot_metadata(_metadata_for_display(metadata))
    requires_raw = meta.get("requires")
    requires = cast(dict[str, Any], requires_raw) if isinstance(requires_raw, dict) else {}
    bins, env = _requirements_from_metadata(requires)

    return {
        "bins": bins,
        "env": env,
        "missing_bins": [value for value in bins if not shutil.which(value)],
        "missing_env": [value for value in env if not os.environ.get(value)],
    }


def _requirements_from_metadata(requires: dict[str, Any]) -> tuple[list[str], list[str]]:
    raw_bins = requires.get("bins")
    raw_env = requires.get("env")
    bins = (
        [value.strip() for value in cast(list[object], raw_bins) if isinstance(value, str) and value.strip()]
        if isinstance(raw_bins, list)
        else []
    )
    env = (
        [value.strip() for value in cast(list[object], raw_env) if isinstance(value, str) and value.strip()]
        if isinstance(raw_env, list)
        else []
    )
    return bins, env


def _description(metadata: dict[str, Any] | None, fallback: str) -> str:
    if metadata is None:
        return fallback
    value = metadata.get("description")
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _nanobot_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    raw = metadata.get("metadata")
    if isinstance(raw, str):
        try:
            raw = cast(object, json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    metadata_payload = cast(dict[str, Any], raw)
    payload = metadata_payload.get("nanobot", metadata_payload.get("openclaw", {}))
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else {}


def _install_options(metadata: dict[str, Any] | None) -> list[dict[str, str]]:
    """Return safe, copyable setup commands declared by a skill."""
    raw_install = _nanobot_metadata(metadata).get("install")
    if not isinstance(raw_install, list):
        return []
    install = cast(list[object], raw_install)

    options: list[dict[str, str]] = []
    for item in install:
        if not isinstance(item, dict):
            continue
        install_item = cast(dict[str, object], item)
        kind = install_item.get("kind")
        if not isinstance(kind, str) or kind not in {"brew", "apt"}:
            continue
        package = (
            install_item.get("formula") if kind == "brew" else install_item.get("package")
        )
        if not isinstance(package, str) or not package.strip():
            continue
        if kind == "brew":
            command = f"brew install {shlex.quote(package.strip())}"
        else:
            command = f"sudo apt-get install -y {shlex.quote(package.strip())}"
        option_id = install_item.get("id")
        label = install_item.get("label")
        options.append(
            {
                "id": option_id if isinstance(option_id, str) else kind,
                "kind": kind,
                "label": label if isinstance(label, str) else f"Install with {kind}",
                "command": command,
            }
        )
    return options
