"""Bounded copied-state inventory. Never admission, recovery, or lease inspection.

Only config bytes are read. Retained evidence is deliberately UNQUALIFIED: public
authority inspectors acquire locks or follow persisted paths. This adapter must not.
"""
from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import cast

CHECKS = (
    "copy_boundary", "config_structure", "config_schema", "sessions", "recovery",
    "artifact_policy", "artifact_state", "channel_plugins", "dependencies",
    "cron_definitions", "cron_runtime", "background_dormancy", "platform",
    "p6_release", "d9_authorization",
)
STATUSES = frozenset({"PASS", "FAIL", "UNQUALIFIED"})
REASONS = frozenset({"CLEAN", "INVALID", "RETAINED", "UNSAFE", "BOUND",
                     "CHANGED", "DISABLED_REQUIRED", "METADATA", "P6_REQUIRED",
                     "AUTHORIZATION_REQUIRED", "UNSUPPORTED"})
MAX_ENTRIES = 4096
MAX_BYTES = 262144
MAX_DEPTH = 16


@dataclass(frozen=True)
class Check:
    check: str
    status: str
    reason: str
    count: int = 0


def reduce_checks(evidence: tuple[Check, ...]) -> dict[str, object]:
    """Pure exact fixed-evidence reducer; READY grants no runtime authority."""
    valid = (isinstance(cast(object, evidence), tuple) and len(evidence) == len(CHECKS)
             and all(isinstance(cast(object, item), Check) and type(item.check) is str
                     and type(item.status) is str and type(item.reason) is str
                     for item in evidence)
             and tuple(item.check for item in evidence) == CHECKS
             and all(item.status in STATUSES and item.reason in REASONS
                     and type(item.count) is int and 0 <= item.count <= MAX_ENTRIES
                     for item in evidence))
    if not valid:
        evidence = tuple(Check(name, "FAIL", "INVALID") for name in CHECKS)
    return {"version": 1, "scope": "COPIED_ROOT_INVENTORY",
            "activation": "NOT_AUTHORIZED",
            "result": "READY" if valid and all(x.status == "PASS" for x in evidence)
            else "NOT_READY",
            "checks": [{"check": x.check, "status": x.status,
                        "reason": x.reason, "count": x.count} for x in evidence]}


class _UnsafeError(Exception):
    pass


def _path(value: str) -> Path:
    if not isinstance(cast(object, value), str) or len(value) > 4096 or "\x00" in value:
        raise _UnsafeError
    path = Path(value)
    if (not path.is_absolute() or path == Path(path.anchor)
            or str(path) != value or any(x in (".", "..") for x in value.split("/"))):
        raise _UnsafeError
    return path


def _signature(info: os.stat_result) -> tuple[int, ...]:
    # atime can change on a read; deliberately not a content/authority mutation.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _ancestor_signature(info: os.stat_result) -> tuple[int, ...]:
    # Shared ancestors are authority bindings, not part of the copied inventory.
    # Sibling activity may change their size/times/link count without changing
    # the pinned directory, ownership, permissions, or no-follow path binding.
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


class _Copy:
    def __init__(self, root: Path):
        self.root = root
        self.fds: list[int] = []
        self.snapshots: list[tuple[int, str, tuple[int, ...]]] = []
        self.ancestor_snapshots: list[tuple[int, str, tuple[int, ...]]] = []
        self.entries = 0
        try:
            self.root_fd = self.open_dir(root)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for fd in reversed(self.fds):
            os.close(fd)

    def open_dir(self, path: Path) -> int:
        # Open every absolute ancestor no-follow, retaining pinned descriptors.
        fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.fds.append(fd)
        current_path = Path(path.anchor)
        for part in path.parts[1:]:
            current_path /= part
            info = os.stat(part, dir_fd=fd, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise _UnsafeError
            in_copy = current_path.is_relative_to(self.root)
            signature = _signature if in_copy else _ancestor_signature
            snapshots = self.snapshots if in_copy else self.ancestor_snapshots
            snapshots.append((fd, part, signature(info)))
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            self.fds.append(fd)
            if signature(os.fstat(fd)) != signature(info):
                raise _UnsafeError
        return fd

    def parent(self, path: Path) -> tuple[int, str] | None:
        parts = path.relative_to(self.root).parts
        fd = self.root_fd
        for part in parts[:-1]:
            try:
                info = os.stat(part, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                self.snapshots.append((fd, part, ()))
                return None
            if not stat.S_ISDIR(info.st_mode):
                raise _UnsafeError
            self.snapshots.append((fd, part, _signature(info)))
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            self.fds.append(child)
            if _signature(os.fstat(child)) != _signature(info):
                raise _UnsafeError
            fd = child
        return fd, parts[-1]

    def config(self, path: Path) -> object:
        parent = self.parent(path)
        if parent is None:
            raise _UnsafeError
        fd, name = parent
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_BYTES:
            raise _UnsafeError
        self.snapshots.append((fd, name, _signature(info)))
        opened = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        self.fds.append(opened)
        if _signature(os.fstat(opened)) != _signature(info):
            raise _UnsafeError
        content = bytearray()
        while len(content) <= MAX_BYTES:
            chunk = os.read(opened, min(8192, MAX_BYTES + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        if len(content) > MAX_BYTES or _signature(os.fstat(opened)) != _signature(info):
            raise _UnsafeError
        return json.loads(content, object_pairs_hook=_pairs)

    def inventory(self, path: Path) -> int:
        parent = self.parent(path)
        if parent is None:
            return 0
        fd, name = parent
        try:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            self.snapshots.append((fd, name, ()))
            return 0
        self.snapshots.append((fd, name, _signature(info)))
        if not stat.S_ISDIR(info.st_mode):
            raise _UnsafeError
        opened = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        self.fds.append(opened)
        if _signature(os.fstat(opened)) != _signature(info):
            raise _UnsafeError
        before = self.entries
        self.walk(opened, 0)
        return self.entries - before

    def require_dir(self, path: Path) -> None:
        parent = self.parent(path)
        if parent is None:
            raise _UnsafeError
        fd, name = parent
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise _UnsafeError
        self.snapshots.append((fd, name, _signature(info)))

    def walk(self, fd: int, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise _UnsafeError
        before = _signature(os.fstat(fd))
        with os.scandir(fd) as entries:
            for entry in entries:
                self.entries += 1
                if self.entries > MAX_ENTRIES:
                    raise _UnsafeError
                info = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
                self.snapshots.append((fd, entry.name, _signature(info)))
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=fd)
                    self.fds.append(child)
                    if _signature(os.fstat(child)) != _signature(info):
                        raise _UnsafeError
                    self.walk(child, depth + 1)
                elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise _UnsafeError
        if _signature(os.fstat(fd)) != before:
            raise _UnsafeError

    def recheck(self) -> None:
        for fd, name, expected in self.ancestor_snapshots:
            try:
                current = _ancestor_signature(os.stat(name, dir_fd=fd, follow_symlinks=False))
            except FileNotFoundError:
                current = ()
            if current != expected:
                raise _UnsafeError
        for fd, name, expected in self.snapshots:
            try:
                current = _signature(os.stat(name, dir_fd=fd, follow_symlinks=False))
            except FileNotFoundError:
                current = ()
            if current != expected:
                raise _UnsafeError


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    aliases: set[str] = set()
    for key, value in items:
        normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
        if key in result or normalized in aliases:
            raise _UnsafeError
        aliases.add(normalized)
        result[normalized] = value
    return result


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _UnsafeError
    # Production callers supply only _Copy.config/_pairs JSON trees or {} defaults.
    # The decoder supplies string keys and _pairs retains them recursively; this
    # container guard is not a generic string-key validator for arbitrary dicts.
    return cast(dict[str, object], value)


def _disabled(value: object) -> bool:
    return _object(value).get("enabled", False) is False


def _channels(value: object) -> tuple[Check, Check]:
    # Descriptors are first-party dependency-free metadata; never runtime/config callbacks.
    from packaging.requirements import Requirement
    from packaging.version import Version

    from nanobot.channels.plugin import load_channel_package
    from nanobot.channels.plugin import parse_channel_target as _target_parts
    from nanobot.channels.registry import packaged_channel_names

    raw = _object(value)
    reserved = packaged_channel_names()
    enabled: set[str] = set()
    for name, settings in raw.items():
        if isinstance(settings, dict):
            flag = cast(dict[object, object], settings).get("enabled", False)
            if type(flag) is not bool:
                raise _UnsafeError
            instances = cast(dict[object, object], settings).get("instances", {})
            if instances:
                # Instance topology is runtime-owned; do not assume activation defaults.
                return Check("channel_plugins", "UNQUALIFIED", "METADATA"), Check(
                    "dependencies", "UNQUALIFIED", "METADATA")
            if flag:
                enabled.add(name)
        elif name not in {"send_progress", "send_tool_hints", "show_reasoning"}:
            raise _UnsafeError
    candidates = metadata.entry_points(group="nanobot.channels")
    selected: dict[str, metadata.EntryPoint] = {}
    for candidate in candidates:
        if candidate.name in enabled:
            if candidate.name in selected or candidate.name in reserved:
                return Check("channel_plugins", "FAIL", "METADATA"), Check(
                    "dependencies", "UNQUALIFIED", "METADATA")
            selected[candidate.name] = candidate
    requirements: list[str] = []
    legacy = False
    for name in enabled:
        if name in reserved:
            plugin = load_channel_package(name)  # direct errors, no discover/logged exception
            if plugin is None:
                raise _UnsafeError
            requirements.extend(plugin.dependencies)
        elif name not in selected:
            return Check("channel_plugins", "FAIL", "METADATA"), Check(
                "dependencies", "UNQUALIFIED", "METADATA")
        else:
            _target_parts(selected[name].value, label="runtime")
            legacy = True  # empty legacy descriptor dependency tuple is not qualification
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            return Check("channel_plugins", "PASS", "METADATA", len(enabled)), Check(
                "dependencies", "FAIL", "METADATA")
        if requirement.url or Version(version) not in requirement.specifier or requirement.extras:
            # Extras' transitive distribution availability requires P6 artifact metadata.
            return Check("channel_plugins", "PASS", "METADATA", len(enabled)), Check(
                "dependencies", "UNQUALIFIED", "METADATA")
    return (Check("channel_plugins", "PASS", "METADATA", len(enabled)),
            Check("dependencies", "UNQUALIFIED" if legacy else "PASS", "METADATA"))


def inspect_copy(*, copy_root: str, config: str, workspace: str, sessions_root: str,
                 runtime_data_root: str, stopped_writer_copy: bool) -> dict[str, object]:
    evidence = {name: Check(name, "UNQUALIFIED", "UNSUPPORTED") for name in CHECKS}
    evidence["config_schema"] = Check("config_schema", "UNQUALIFIED", "P6_REQUIRED")
    evidence["p6_release"] = Check("p6_release", "UNQUALIFIED", "P6_REQUIRED")
    evidence["d9_authorization"] = Check("d9_authorization", "UNQUALIFIED",
                                           "AUTHORIZATION_REQUIRED")
    copied: _Copy | None = None
    try:
        if stopped_writer_copy is not True or type(stopped_writer_copy) is not bool:
            raise _UnsafeError
        root, cfg, work, sessions, runtime = map(
            _path, (copy_root, config, workspace, sessions_root, runtime_data_root))
        for path in (cfg, work, sessions, runtime):
            if not path.is_relative_to(root) or path == root:
                raise _UnsafeError
        if (cfg.parent != runtime or sessions != runtime / "sessions"
                or work.is_relative_to(runtime) or runtime.is_relative_to(work)):
            raise _UnsafeError
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            return reduce_checks(tuple(evidence[name] for name in CHECKS))
        copied = _Copy(root)
        copied.require_dir(work)
        copied.require_dir(runtime)
        raw = _object(copied.config(cfg))
        defaults = _object(_object(raw.get("agents", {})).get("defaults", {}))
        structure = defaults.get("workspace") == str(work)
        ttl_names = ("session_ttl_minutes", "idle_compact_after_minutes")
        ttl_values = [defaults[name] for name in ttl_names if name in defaults]
        ttl = ttl_values[0] if ttl_values else 0
        ttl_valid = len(ttl_values) <= 1 and type(ttl) is int and ttl >= 0
        structure = structure and ttl_valid
        evidence["config_structure"] = Check("config_structure", "PASS" if structure else "FAIL",
                                              "CLEAN" if structure else "INVALID")
        heartbeat = _object(_object(raw.get("gateway", {})).get("heartbeat", {}))
        dormant = (heartbeat.get("enabled") is False and ttl_valid and ttl == 0
                   and _disabled(defaults.get("dream", {}))
                   and _disabled(defaults.get("idle_compaction", {})))
        # No current public switch qualifies a timer/callback/rematerializer/retention authority.
        for name in ("timer", "scheduler", "callbacks", "rematerializer", "retention"):
            if name in raw and raw[name] not in (None, {}):
                dormant = False
        evidence["background_dormancy"] = Check("background_dormancy",
            "PASS" if dormant else "FAIL", "CLEAN" if dormant else "DISABLED_REQUIRED")
        artifacts = _object(raw.get("artifacts", {}))
        flag = artifacts.get("enabled", False)
        blob = artifacts.get("max_blob_bytes", 64 * 1024 * 1024)
        quota = artifacts.get("max_namespace_bytes", 1024 * 1024 * 1024)
        artifact_root = runtime / "context-artifacts"
        if artifacts.get("root") is not None:
            artifact_root = _path(artifacts["root"])
            if not artifact_root.is_relative_to(root) or artifact_root == root:
                raise _UnsafeError
            for private in (work, sessions):
                if artifact_root.is_relative_to(private) or private.is_relative_to(artifact_root):
                    raise _UnsafeError
        policy = type(flag) is bool and type(blob) is int and type(quota) is int
        policy = policy and 0 < blob <= quota
        # Context/read policy is not traversed or guessed; enabled artifacts require P6.
        evidence["artifact_policy"] = Check("artifact_policy",
            "FAIL" if not policy else "UNQUALIFIED" if flag else "PASS",
            "INVALID" if not policy else "P6_REQUIRED" if flag else "CLEAN")
        for name, paths in (
            ("sessions", (sessions, work / "sessions", work / ".nanobot")),
            ("recovery", (work / "memory",)),
            ("artifact_state", tuple(dict.fromkeys((artifact_root, runtime / "context-artifacts",
                                                    runtime / "artifacts")))),
            ("cron_definitions", (work / "cron", runtime / "cron")),
            ("cron_runtime", (runtime / "cron-runtime",)),
        ):
            count = sum(copied.inventory(path) for path in paths)
            evidence[name] = Check(name, "UNQUALIFIED" if count else "PASS",
                                   "RETAINED" if count else "CLEAN", count)
        try:
            plugins, dependencies = _channels(raw.get("channels", {}))
            evidence["channel_plugins"] = plugins
            evidence["dependencies"] = dependencies
        except Exception:
            evidence["channel_plugins"] = Check("channel_plugins", "FAIL", "METADATA")
            evidence["dependencies"] = Check("dependencies", "UNQUALIFIED", "METADATA")
        copied.recheck()
        evidence["platform"] = Check("platform", "PASS", "CLEAN")
        evidence["copy_boundary"] = Check("copy_boundary", "PASS", "CLEAN")
    except (OSError, ValueError, TypeError, RecursionError, _UnsafeError):
        evidence["copy_boundary"] = Check("copy_boundary", "FAIL", "UNSAFE")
    finally:
        if copied is not None:
            copied.close()
    return reduce_checks(tuple(evidence[name] for name in CHECKS))
