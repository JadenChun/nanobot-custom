"""Offline P6 release inspection and disposable copied-state rehearsal.

This helper deliberately does not build, install, start, or mutate a service.
Archive inspection reads release files; the state rehearsal writes only to a
temporary directory and uses the existing session and cron conversion APIs.
The output is content-free apart from archive member names and dependency
specifiers, so it can be retained as migration evidence.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import hashlib
import io
import json
import os
import shlex
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import webbrowser
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


class ReleaseArtifactError(ValueError):
    """Raised when an artifact is unsafe or cannot be inspected."""


class _ExternalEffectAttemptError(RuntimeError):
    """Raised by the rehearsal guard when a prohibited side effect is attempted."""


_REQUIRED_ASSETS = (
    "nanobot/__init__.py",
    "nanobot/templates/IDENTITY.md",
    "nanobot/templates/TOOLS.md",
    "nanobot/skills/README.md",
    "nanobot/skills/desktop-use/SKILL.md",
    "nanobot/skills/mobile-app-testing/SKILL.md",
    "nanobot/skills/mobile-dogfood/SKILL.md",
    "nanobot/skills/webapp-testing/SKILL.md",
    "nanobot/skills/video-understanding/SKILL.md",
    "nanobot/skills/video-understanding/scripts/extract_keyframes.py",
    "nanobot/crawler/worker.py",
    "nanobot/channels/telegram/webui/index.ts",
    "nanobot/channels/websocket/webui/index.ts",
    "nanobot/web/dist/index.html",
)
_FORBIDDEN_COMPONENTS = {
    ".git",
    ".nanobot",
    "__pycache__",
    "build",
    "dist",
    "logs",
    "media",
    "sessions",
    "tests",
}
_FORBIDDEN_FILE_NAMES = {
    ".env",
    ".env.local",
    "config.json",
    "credentials.json",
    "token.json",
}
_FORBIDDEN_INSTALL_MARKERS = (
    "pip install",
    "python -m pip",
    "bun install",
    "npm install",
    "nanobot webui",
)


@dataclass(frozen=True)
class ArchiveReport:
    """Content-free archive inventory suitable for release evidence."""

    path: str
    format: str
    sha256: str
    size: int
    member_count: int
    missing: tuple[str, ...]
    forbidden: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DependencyReport:
    """Declared dependency inputs and the lockfile availability status."""

    project: str
    dependencies: tuple[str, ...]
    optional_dependencies: dict[str, tuple[str, ...]]
    pyproject_sha256: str
    lockfile_path: str
    lockfile_present: bool
    lockfile_ignored: bool

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServiceReport:
    """Read-only service command contract inspection."""

    path: str
    valid: bool
    missing: tuple[str, ...]
    forbidden: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ArchiveEntry:
    """Archive member name and the type recorded by the archive format."""

    name: str
    kind: str


_STATE_INVARIANTS = {
    "session": (
        "conversion_runs_equal",
        "rollback_retained_new_write",
        "canonical_copy_retained",
        "rollback_restored",
        "backup_unchanged",
    ),
    "cron": (
        "conversion_runs_equal",
        "legacy_backup_matches_source",
        "source_jobs_retained",
    ),
}
_STATE_TOP_LEVEL_KEYS = frozenset((*_STATE_INVARIANTS, "external_effects"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".whl") or name.endswith(".zip"):
        return "wheel"
    if name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar"):
        return "sdist"
    raise ReleaseArtifactError(f"unsupported release archive format: {path.name}")


def _archive_stat(path: Path) -> os.stat_result:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReleaseArtifactError(f"cannot stat release archive {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ReleaseArtifactError(f"release archive is not a regular file: {path}")
    return info


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _assert_archive_binding(path: Path, handle_info: os.stat_result) -> None:
    path_info = _archive_stat(path)
    if not _same_file(handle_info, path_info):
        raise ReleaseArtifactError(f"release archive replaced during inspection: {path}")


@contextlib.contextmanager
def _stable_archive(path: Path) -> Iterator[tuple[Path, bytes]]:
    """Read one stable archive handle and verify that its path stays bound.

    The archive parsers consume the in-memory bytes read from this single open
    handle.  Before and after the read, and once more after parsing, the handle
    identity and the path binding are checked so a concurrent rename or write
    cannot pair an inventory from one file with a digest from another.
    """

    resolved = path.expanduser().resolve(strict=True)
    try:
        with resolved.open("rb") as handle:
            handle_info = os.fstat(handle.fileno())
            if not stat.S_ISREG(handle_info.st_mode):
                raise ReleaseArtifactError(f"release archive is not a regular file: {resolved}")
            _assert_archive_binding(resolved, handle_info)
            data = handle.read()
            after_read = os.fstat(handle.fileno())
            if not _same_file(handle_info, after_read):
                raise ReleaseArtifactError(f"release archive changed during inspection: {resolved}")
            _assert_archive_binding(resolved, after_read)
            yield resolved, data
            after_parse = os.fstat(handle.fileno())
            if not _same_file(handle_info, after_parse):
                raise ReleaseArtifactError(f"release archive changed during inspection: {resolved}")
            _assert_archive_binding(resolved, after_parse)
    except ReleaseArtifactError:
        raise
    except OSError as exc:
        raise ReleaseArtifactError(f"cannot read release archive {resolved}: {exc}") from exc


def _safe_member_name(name: str) -> None:
    if not name or "\\" in name:
        raise ReleaseArtifactError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ReleaseArtifactError(f"unsafe archive member path: {name!r}")


def _forbidden_member_reason(name: str) -> str | None:
    path = PurePosixPath(name)
    if any(part in _FORBIDDEN_COMPONENTS for part in path.parts if part != "dist"):
        return "local-state-or-test-member"
    packaged_webui = any(
        path.parts[index : index + 3] == ("nanobot", "web", "dist")
        for index in range(max(0, len(path.parts) - 2))
    )
    if "dist" in path.parts and not packaged_webui:
        return "local-build-member"
    if path.name in _FORBIDDEN_FILE_NAMES or path.name.startswith(".env."):
        return "credential-or-environment-member"
    return None


def _zip_entries(data: bytes, path: Path) -> tuple[_ArchiveEntry, ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries: list[_ArchiveEntry] = []
            for info in archive.infolist():
                mode = info.external_attr >> 16
                dos_directory = info.create_system == 0 and bool(info.external_attr & 0x10)
                file_type = stat.S_IFMT(mode)
                if info.is_dir() or dos_directory or stat.S_ISDIR(mode):
                    kind = "directory"
                elif stat.S_ISLNK(mode):
                    raise ReleaseArtifactError(
                        f"unsupported symlink archive member: {info.filename!r}"
                    )
                elif file_type not in (0, stat.S_IFREG):
                    raise ReleaseArtifactError(
                        f"unsupported non-regular archive member: {info.filename!r}"
                    )
                else:
                    kind = "file"
                entries.append(_ArchiveEntry(info.filename, kind))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseArtifactError(f"cannot read wheel archive {path}: {exc}") from exc
    return tuple(entries)


def _tar_entries(data: bytes, path: Path) -> tuple[_ArchiveEntry, ...]:
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseArtifactError(f"cannot read source archive {path}: {exc}") from exc
    entries: list[_ArchiveEntry] = []
    for member in members:
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise ReleaseArtifactError(f"unsupported non-regular archive member: {member.name!r}")
        entries.append(_ArchiveEntry(member.name, "directory" if member.isdir() else "file"))
    return tuple(entries)


def _zip_members(path: Path) -> tuple[str, ...]:
    """Return ZIP member names while applying stable-path and type checks."""

    with _stable_archive(path) as (resolved, data):
        entries = _zip_entries(data, resolved)
    return tuple(entry.name for entry in entries)


def _tar_members(path: Path) -> tuple[str, ...]:
    """Return tar member names while applying stable-path and type checks."""

    with _stable_archive(path) as (resolved, data):
        entries = _tar_entries(data, resolved)
    return tuple(entry.name for entry in entries)


def _validate_archive_entries(entries: tuple[_ArchiveEntry, ...]) -> None:
    names = tuple(entry.name for entry in entries)
    if len(set(names)) != len(names):
        raise ReleaseArtifactError("archive contains duplicate member names")

    forbidden: list[str] = []
    for entry in entries:
        _safe_member_name(entry.name)
        if reason := _forbidden_member_reason(entry.name):
            forbidden.append(f"{entry.name} ({reason})")
    if forbidden:
        raise ReleaseArtifactError("forbidden archive members: " + ", ".join(forbidden))


def _wheel_dist_info_root(path: Path) -> str:
    """Derive the canonical wheel ``.dist-info`` directory from its filename."""

    try:
        from packaging.utils import InvalidWheelFilename, parse_wheel_filename

        filename = path.name
        if filename.lower().endswith(".zip"):
            filename = filename[:-4] + ".whl"
        distribution, version, _build, _tags = parse_wheel_filename(filename)
    except (ImportError, InvalidWheelFilename, ValueError) as exc:
        raise ReleaseArtifactError(f"invalid wheel filename: {path.name}") from exc
    normalized_distribution = str(distribution).replace("-", "_")
    return f"{normalized_distribution}-{version}.dist-info"


def _sdist_root(path: Path) -> str:
    lower_name = path.name.lower()
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if lower_name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise ReleaseArtifactError(f"unsupported source archive filename: {path.name}")


def _required_asset_status(
    entries: tuple[_ArchiveEntry, ...],
    required: Iterable[str],
) -> tuple[str, ...]:
    by_name = {entry.name: entry for entry in entries}
    return tuple(
        asset
        for asset in required
        if by_name.get(asset) is None or by_name[asset].kind != "file"
    )


def inspect_archive(path: Path) -> ArchiveReport:
    """Inspect a wheel or sdist without extracting or executing it.

    Missing required assets are returned in the report for machine-readable
    release gating. Unsafe paths and forbidden local-state/credential members
    raise immediately because the archive must never be accepted as-is.
    """
    with _stable_archive(path) as (resolved, data):
        archive_format = _archive_format(resolved)
        entries = _zip_entries(data, resolved) if archive_format == "wheel" else _tar_entries(data, resolved)
        _validate_archive_entries(entries)

        if archive_format == "sdist":
            root = _sdist_root(resolved)
            prefix = f"{root}/"
            relative_entries: list[_ArchiveEntry] = []
            saw_root = False
            for entry in entries:
                if entry.name == root or entry.name == prefix:
                    if saw_root:
                        raise ReleaseArtifactError(
                            f"source archive has multiple distribution roots: {root}"
                        )
                    if entry.kind != "directory":
                        raise ReleaseArtifactError(f"source archive root is not a directory: {root}")
                    saw_root = True
                    continue
                if not entry.name.startswith(prefix):
                    raise ReleaseArtifactError(
                        f"source archive member is outside distribution root {root!r}: {entry.name!r}"
                    )
                relative_entries.append(
                    _ArchiveEntry(entry.name[len(prefix) :], entry.kind)
                )
            if not saw_root and not relative_entries:
                raise ReleaseArtifactError(f"source archive has no distribution root: {root}")
            # A single validated root is established by the exact prefix check above.
            required = (*_REQUIRED_ASSETS, "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "hatch_build.py")
            missing = _required_asset_status(tuple(relative_entries), required)
        else:
            metadata_root = _wheel_dist_info_root(resolved)
            required = (*_REQUIRED_ASSETS, f"{metadata_root}/METADATA", f"{metadata_root}/RECORD")
            missing = _required_asset_status(entries, required)

        digest = hashlib.sha256(data).hexdigest()
        return ArchiveReport(
            path=str(resolved),
            format=archive_format,
            sha256=digest,
            size=len(data),
            member_count=len(entries),
            missing=missing,
            forbidden=(),
        )


def inspect_dependency_inputs(project_root: Path) -> DependencyReport:
    """Record declared dependencies and whether the ignored uv lock exists."""
    import tomllib

    root = project_root.expanduser().resolve(strict=True)
    pyproject_path = root / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseArtifactError(f"cannot read dependency manifest: {exc}") from exc
    project = data.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("name"), str):
        raise ReleaseArtifactError("pyproject.toml has no valid [project] name")
    dependencies = project.get("dependencies", [])
    optional = project.get("optional-dependencies", {})
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise ReleaseArtifactError("project.dependencies must be a list of strings")
    if not isinstance(optional, dict):
        raise ReleaseArtifactError("project.optional-dependencies must be a table")
    optional_dependencies: dict[str, tuple[str, ...]] = {}
    for name, values in optional.items():
        if not isinstance(name, str) or not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            raise ReleaseArtifactError("optional dependency entries must be string lists")
        optional_dependencies[name] = tuple(values)
    lockfile = root / "uv.lock"
    ignored = any(
        line.strip() in {"uv.lock", "/uv.lock"}
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    ) if (root / ".gitignore").is_file() else False
    return DependencyReport(
        project=project["name"],
        dependencies=tuple(dependencies),
        optional_dependencies=optional_dependencies,
        pyproject_sha256=_sha256(pyproject_path),
        lockfile_path="uv.lock",
        lockfile_present=lockfile.is_file(),
        lockfile_ignored=ignored,
    )


def _unit_lines(text: str) -> Iterator[str]:
    """Yield logical systemd lines, joining a trailing backslash continuation."""

    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if pending:
            line = pending + line
        if line.endswith("\\") and not line.endswith("\\\\"):
            pending = line[:-1].rstrip() + " "
            continue
        pending = ""
        yield line
    if pending:
        yield pending.rstrip()


def _active_service_directives(text: str) -> tuple[tuple[str, str], ...]:
    """Parse active ``[Service]`` key/value directives only."""

    section: str | None = None
    directives: list[tuple[str, str]] = []
    for line in _unit_lines(text):
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if section != "Service":
            continue
        if "=" not in line:
            directives.append(("", line))
            continue
        key, value = line.split("=", 1)
        directives.append((key.strip(), value.strip()))
    return tuple(directives)


def _command_has_option(tokens: list[str], option: str) -> bool:
    return any(
        token == option
        and index + 1 < len(tokens)
        and bool(tokens[index + 1])
        and not tokens[index + 1].startswith("--")
        for index, token in enumerate(tokens)
    )


def _command_has_gateway(tokens: list[str]) -> bool:
    """Accept only a direct nanobot executable followed by ``gateway``.

    Wrapper commands such as ``env``, ``sh -c``, and arbitrary executables
    with a ``nanobot gateway`` argument are deliberately unsupported here.
    Keeping the accepted shape narrow makes this static check fail closed.
    """

    return len(tokens) >= 2 and Path(tokens[0]).name == "nanobot" and tokens[1] == "gateway"


def inspect_service_unit(path: Path) -> ServiceReport:
    """Check the retained headless service command without invoking systemd."""

    resolved = path.expanduser().resolve(strict=True)
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseArtifactError(f"cannot read service unit: {exc}") from exc

    directives = _active_service_directives(text)
    exec_values: list[str] = []
    environment_files: list[str] = []
    working_directories: list[str] = []
    forbidden: list[str] = []
    malformed: list[str] = []
    for key, value in directives:
        if not key:
            malformed.append("malformed Service directive")
            continue
        if key == "ExecStart":
            if not value:
                exec_values.clear()
            else:
                exec_values.append(value)
        elif key == "EnvironmentFile":
            if not value:
                environment_files.clear()
            else:
                environment_files.append(value)
        elif key == "WorkingDirectory":
            if not value:
                working_directories.clear()
            else:
                working_directories.append(value)
        directive_text = f"{key}={value}"
        forbidden.extend(
            marker for marker in _FORBIDDEN_INSTALL_MARKERS if marker in directive_text
        )

    missing: list[str] = []
    if not exec_values:
        missing.append("ExecStart=")
    elif len(exec_values) != 1:
        forbidden.append("duplicate ExecStart")
    else:
        try:
            tokens = shlex.split(exec_values[0], posix=True)
        except ValueError:
            tokens = []
            missing.append("ExecStart command")
        if tokens:
            if not _command_has_gateway(tokens):
                missing.append("nanobot gateway")
            if not _command_has_option(tokens, "--config"):
                missing.append("--config ")
            if not _command_has_option(tokens, "--log-file"):
                missing.append("--log-file ")
    if not environment_files:
        missing.append("EnvironmentFile=")
    if not working_directories:
        missing.append("WorkingDirectory=")
    forbidden.extend(malformed)
    return ServiceReport(
        path=str(resolved),
        valid=not missing and not forbidden,
        missing=tuple(dict.fromkeys(missing)),
        forbidden=tuple(dict.fromkeys(forbidden)),
    )


def _write_legacy_session(path: Path, key: str, content: str) -> Path:
    from nanobot.session.manager import JsonlSessionStore

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "_type": "metadata",
                "key": key,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "metadata": {},
                "last_consolidated": 0,
            }
        )
        + "\n"
        + json.dumps({"role": "user", "content": content})
        + "\n",
        encoding="utf-8",
    )
    migrated_path = path.parent / f"{JsonlSessionStore.storage_key(key)}.jsonl"
    path.rename(migrated_path)
    return migrated_path


def _legacy_job() -> dict[str, Any]:
    return {
        "id": "p6-daily",
        "name": "P6 rehearsal",
        "enabled": True,
        "schedule": {"kind": "cron", "expr": "0 9 * * *", "tz": "UTC"},
        "payload": {
            "kind": "agent_turn",
            "message": "rehearsal",
            "deliver": False,
            "channel": None,
            "to": None,
            "additionalDestinations": [],
            "channelMeta": {},
            "sessionKey": None,
            "executionMode": "legacy_fresh",
            "originChannel": None,
            "originChatId": None,
            "originMetadata": {},
        },
        "state": {
            "nextRunAtMs": 100,
            "lastRunAtMs": 90,
            "lastStatus": "ok",
            "lastError": None,
            "lastRunStatus": "succeeded",
            "runHistory": [
                {
                    "runAtMs": 90,
                    "status": "ok",
                    "durationMs": 1,
                    "error": None,
                    "runId": "p6-run",
                    "runStatus": "succeeded",
                    "occurrenceId": "p6-occurrence",
                }
            ],
        },
        "createdAtMs": 10,
        "updatedAtMs": 20,
        "deleteAfterRun": False,
    }


def _write_private(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    os.chmod(path, 0o600)


@contextlib.contextmanager
def _rehearsal_effect_guard() -> Iterator[list[str]]:
    """Block and record network, process, browser, and provider/channel seams.

    The session and cron modules are imported before this context is entered;
    provider and channel imports therefore remain outside the rehearsal
    allowlist.  Any later attempt to import those namespaces or invoke a
    patched effect seam is recorded and stopped.
    """

    attempts: list[str] = []

    def blocked(label: str):
        def reject(*_args: Any, **_kwargs: Any) -> None:
            attempts.append(label)
            raise _ExternalEffectAttemptError(label)

        return reject

    with contextlib.ExitStack() as stack:
        replacements = (
            (socket, "socket", "network.socket"),
            (socket, "create_connection", "network.create_connection"),
            (socket, "create_server", "network.create_server"),
            (socket, "socketpair", "network.socketpair"),
            (socket, "getaddrinfo", "network.getaddrinfo"),
            (socket, "gethostbyname", "network.gethostbyname"),
            (socket, "gethostbyname_ex", "network.gethostbyname_ex"),
            (subprocess, "Popen", "process.Popen"),
            (subprocess, "run", "process.run"),
            (subprocess, "call", "process.call"),
            (subprocess, "check_call", "process.check_call"),
            (subprocess, "check_output", "process.check_output"),
            (subprocess, "getoutput", "process.getoutput"),
            (subprocess, "getstatusoutput", "process.getstatusoutput"),
            (os, "system", "system.os.system"),
            (os, "popen", "system.os.popen"),
            (webbrowser, "open", "browser.open"),
            (webbrowser, "open_new", "browser.open_new"),
            (webbrowser, "open_new_tab", "browser.open_new_tab"),
        )
        for module, name, label in replacements:
            if hasattr(module, name):
                stack.callback(setattr, module, name, getattr(module, name))
                setattr(module, name, blocked(label))
        for name in (
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
            "posix_spawn",
            "posix_spawnp",
            "execv",
            "execve",
            "execvp",
            "execvpe",
            "fork",
            "forkpty",
        ):
            if hasattr(os, name):
                stack.callback(setattr, os, name, getattr(os, name))
                setattr(os, name, blocked(f"system.os.{name}"))

        original_import = builtins.__import__

        def guarded_import(
            name: str,
            globals: dict[str, Any] | None = None,
            locals: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            blocked_namespace = (
                name in {"nanobot.providers", "nanobot.channels"}
                or name.startswith(("nanobot.providers.", "nanobot.channels."))
            )
            if level == 0 and blocked_namespace:
                label = f"import:{name}"
                attempts.append(label)
                raise _ExternalEffectAttemptError(label)
            return original_import(name, globals, locals, fromlist, level)

        stack.callback(setattr, builtins, "__import__", original_import)
        builtins.__import__ = guarded_import
        yield attempts


def _run_state_rehearsal_body() -> dict[str, Any]:
    """Run the state-only body after its dependencies have been imported."""

    from nanobot.cron.action_store import CronActionStore
    from nanobot.session.manager import SessionManager

    with tempfile.TemporaryDirectory(prefix="nanobot-p6-rehearsal-") as raw_root:
        root = Path(raw_root).resolve()
        workspace = root / "workspace"
        sessions_root = root / "new-state" / "sessions"
        key = "telegram:p6"
        legacy_path = workspace / "sessions" / "telegram_p6.jsonl"
        legacy_path = _write_legacy_session(legacy_path, key, "before migration")
        backup_path = root / "backup" / legacy_path.name
        backup_path.parent.mkdir(parents=True)
        backup_path.write_bytes(legacy_path.read_bytes())
        backup_digest = _sha256(backup_path)

        manager = SessionManager(workspace=workspace, sessions_root=sessions_root)
        canonical_path = manager._get_session_path(key)
        first_digest = _sha256(canonical_path)
        second_manager = SessionManager(workspace=workspace, sessions_root=sessions_root)
        second_digest = _sha256(second_manager._get_session_path(key))
        session = second_manager.get_or_create(key)
        session.add_message("assistant", "after migration")
        second_manager.save(session, fsync=True)
        rollback = second_manager.restore_sessions_to_workspace()
        rollback_text = (workspace / "sessions" / canonical_path.name).read_text(encoding="utf-8")

        cron_root = root / "cron-state"
        cron_root.mkdir(mode=0o700)
        job = _legacy_job()
        action_bytes = (
            json.dumps({"action": "add", "params": job}, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        jobs_bytes = json.dumps({"version": 1, "jobs": [job]}, separators=(",", ":")).encode(
            "utf-8"
        )
        _write_private(cron_root / "action.jsonl", action_bytes)
        _write_private(cron_root / "jobs.json", jobs_bytes)
        first_conversion = CronActionStore.convert_legacy(cron_root, namespace="p6-rehearsal")
        second_conversion = CronActionStore.convert_legacy(cron_root, namespace="p6-rehearsal")
        action_backup = cron_root / "legacy-backups" / f"{first_conversion.action_backup_digest}.bin"
        jobs_backup = cron_root / "legacy-backups" / f"{first_conversion.jobs_backup_digest}.bin"

        return {
            "session": {
                "conversion_runs_equal": first_digest == second_digest,
                "rollback_retained_new_write": "after migration" in rollback_text,
                "canonical_copy_retained": canonical_path.exists(),
                "rollback_restored": rollback.restored == 1,
                "backup_unchanged": _sha256(backup_path) == backup_digest,
            },
            "cron": {
                "conversion_runs_equal": first_conversion == second_conversion,
                "legacy_backup_matches_source": (
                    action_backup.read_bytes() == action_bytes
                    and jobs_backup.read_bytes() == jobs_bytes
                ),
                "source_jobs_retained": (cron_root / "jobs.json").read_bytes() == jobs_bytes,
            },
        }


def run_state_rehearsal() -> dict[str, Any]:
    """Exercise copied-state conversion and rollback in a temporary root."""
    # The imports intentionally happen before the allowlist is installed:
    # provider and channel namespaces are never permitted as rehearsal imports.
    from nanobot.cron.action_store import CronActionStore as _CronActionStore
    from nanobot.session.manager import SessionManager as _SessionManager

    del _CronActionStore, _SessionManager
    try:
        with _rehearsal_effect_guard() as attempts:
            report = _run_state_rehearsal_body()
    except _ExternalEffectAttemptError:
        report = {"session": {}, "cron": {}}
    report["external_effects"] = tuple(attempts)
    return report


def _validate_state_rehearsal(value: Any, failures: list[str]) -> None:
    if not isinstance(value, dict):
        failures.append("state_rehearsal: expected an object")
        return
    keys = set(value)
    missing_top = sorted(_STATE_TOP_LEVEL_KEYS - keys)
    unexpected_top = sorted(keys - _STATE_TOP_LEVEL_KEYS)
    if missing_top:
        failures.append(f"state_rehearsal: missing invariants {missing_top}")
    if unexpected_top:
        failures.append(f"state_rehearsal: unexpected invariants {unexpected_top}")

    for group, invariant_names in _STATE_INVARIANTS.items():
        group_value = value.get(group)
        if not isinstance(group_value, dict):
            failures.append(f"state_rehearsal.{group}: expected an object")
            continue
        group_keys = set(group_value)
        missing = sorted(set(invariant_names) - group_keys)
        unexpected = sorted(group_keys - set(invariant_names))
        if missing:
            failures.append(f"state_rehearsal.{group}: missing invariants {missing}")
        if unexpected:
            failures.append(f"state_rehearsal.{group}: unexpected invariants {unexpected}")
        for name in invariant_names:
            if name in group_value and group_value[name] is not True:
                failures.append(f"state_rehearsal.{group}.{name}: expected true")

    effects = value.get("external_effects")
    if not isinstance(effects, (list, tuple)) or not all(isinstance(item, str) for item in effects):
        failures.append("state_rehearsal.external_effects: expected a string sequence")
    elif effects:
        failures.append(f"state_rehearsal.external_effects: attempted {list(effects)}")


def _validate_report(report: dict[str, Any]) -> None:
    failures: list[str] = []
    if not isinstance(report, dict):
        raise ReleaseArtifactError("release report must be an object")
    artifacts = report.get("artifacts", [])
    if not isinstance(artifacts, list):
        failures.append("artifacts: expected a list")
    else:
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                failures.append("artifact: expected an object")
                continue
            path = artifact.get("path", "<unknown artifact>")
            missing = artifact.get("missing")
            if not isinstance(missing, (list, tuple)):
                failures.append(f"{path}: missing must be a sequence")
            elif missing:
                failures.append(f"{path}: missing {missing}")
    service = report.get("service")
    if service is not None:
        if not isinstance(service, dict):
            failures.append("service: expected an object")
        elif service.get("valid") is not True:
            failures.append(f"{service.get('path', '<unknown service>')}: service contract failed")
    if "state_rehearsal" in report:
        _validate_state_rehearsal(report["state_rehearsal"], failures)
    if failures:
        raise ReleaseArtifactError("; ".join(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--service-unit", type=Path)
    parser.add_argument("--state-rehearsal", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    if not args.artifact and args.service_unit is None and not args.state_rehearsal:
        parser.error("provide --artifact, --service-unit, or --state-rehearsal")

    try:
        report: dict[str, Any] = {}
        if args.artifact:
            report["artifacts"] = [inspect_archive(path).as_json() for path in args.artifact]
        if args.project_root:
            report["dependencies"] = inspect_dependency_inputs(args.project_root).as_json()
        if args.service_unit is not None:
            report["service"] = inspect_service_unit(args.service_unit).as_json()
        if args.state_rehearsal:
            report["state_rehearsal"] = run_state_rehearsal()
        _validate_report(report)
    except (OSError, ReleaseArtifactError, ValueError) as exc:
        print(f"p6-release-rehearsal: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
