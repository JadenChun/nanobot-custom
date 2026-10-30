"""Sandbox backends for shell command execution.

To add a new backend, implement a function with the signature:
    _wrap_<name>(command: str, workspace: str, cwd: str) -> str
and register it in _BACKENDS below.
"""

import functools
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from nanobot.config.paths import get_media_dir
from nanobot.security.folder_grants import FolderGrantSnapshot, FolderGrantView


def _normalize_bind_paths(
    paths: Iterable[str] | None,
    *,
    workspace: Path | None = None,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths or []:
        value = str(raw).strip()
        if not value:
            continue
        path = Path(os.path.expandvars(value)).expanduser()
        if not path.is_absolute():
            continue
        resolved_path = path.resolve(strict=False)
        if workspace is not None:
            try:
                workspace.relative_to(resolved_path)
            except ValueError:
                pass
            else:
                # A later bind of the workspace or one of its parents could
                # cover the tmpfs that hides the config directory.
                continue
        resolved = str(resolved_path)
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _bwrap(
    command: str,
    workspace: str,
    cwd: str,
    *,
    sandbox_ro_binds: Iterable[str] | None = None,
    sandbox_rw_binds: Iterable[str] | None = None,
) -> str:
    """Wrap command in a bubblewrap sandbox (requires bwrap in container).

    Only the workspace is bind-mounted read-write; its parent dir (which holds
    config.json) is hidden behind a fresh tmpfs.  The media directory is
    bind-mounted read-only so exec commands can read uploaded attachments.
    """
    ws = Path(workspace).resolve()
    media = get_media_dir().resolve()

    try:
        sandbox_cwd = str(ws / Path(cwd).resolve().relative_to(ws))
    except ValueError:
        sandbox_cwd = str(ws)

    required = ["/usr"]
    optional = [
        "/bin",
        "/lib",
        "/lib64",
        "/etc/alternatives",
        "/etc/ssl/certs",
        "/etc/pki/tls/certs",
        "/etc/pki/ca-trust",
        "/etc/crypto-policies",
        "/etc/resolv.conf",
        "/etc/ld.so.cache",
    ]

    args = ["bwrap", "--new-session", "--die-with-parent", "--setenv", "HOME", str(ws)]
    for p in required:
        args += ["--ro-bind", p, p]
    for p in optional:
        args += ["--ro-bind-try", p, p]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--tmpfs", str(ws.parent),        # mask config dir
        "--dir", str(ws),                 # recreate workspace mount point
        "--bind", str(ws), str(ws),
        "--ro-bind-try", str(media), str(media),  # read-only access to media
    ]
    for p in _normalize_bind_paths(sandbox_ro_binds, workspace=ws):
        args += ["--ro-bind-try", p, p]
    for p in _normalize_bind_paths(sandbox_rw_binds, workspace=ws):
        args += ["--bind-try", p, p]
    args += ["--chdir", sandbox_cwd, "--", "sh", "-c", command]
    return shlex.join(args)


_BACKENDS = {"bwrap": _bwrap}


def wrap_command(
    sandbox: str,
    command: str,
    workspace: str,
    cwd: str,
    *,
    sandbox_ro_binds: Iterable[str] | None = None,
    sandbox_rw_binds: Iterable[str] | None = None,
) -> str:
    """Wrap *command* using the named sandbox backend."""
    if backend := _BACKENDS.get(sandbox):
        return backend(
            command,
            workspace,
            cwd,
            sandbox_ro_binds=sandbox_ro_binds,
            sandbox_rw_binds=sandbox_rw_binds,
        )
    raise ValueError(f"Unknown sandbox backend {sandbox!r}. Available: {list(_BACKENDS)}")


@functools.lru_cache(maxsize=1)
def managed_bwrap_available() -> bool:
    """Return true only when bubblewrap can launch a confined process here."""
    if not sys.platform.startswith("linux"):
        return False
    executable = shutil.which("bwrap")
    if executable is None:
        return False
    command = [executable, "--new-session", "--die-with-parent", "--unshare-pid"]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for source in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(source).exists():
            command += ["--ro-bind", source, source]
    command += ["--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin"]
    command += ["--", "sh", "-c", "exit 0"]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=3,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def wrap_managed_command(
    command: str,
    *,
    cwd: str | Path,
    agent_home: str | Path,
    grants: FolderGrantView,
    include_folder_grants: bool = True,
    grant_snapshot: FolderGrantSnapshot | None = None,
    excluded_roots: Iterable[Path] = (),
    task_roots: Iterable[tuple[str, str, bool]] = (),
    shell_program: str | None = None,
    login: bool = False,
) -> str:
    """Read-only host view with approved writable folders and private temp/cache.

    The legacy ``commands`` bit remains in stored grants for compatibility; it
    does not gate Sandbox programs. More-specific file rules override parents.
    Configured managed repositories retain their separate protected-path/store
    policy. They stay hidden from shell processes; use file tools for them.
    """
    if not managed_bwrap_available():
        raise RuntimeError("Sandbox requires a working bubblewrap installation")
    home = Path(agent_home).expanduser().resolve(strict=True)
    working = Path(cwd).expanduser().resolve(strict=True)
    if not home.is_dir() or not working.is_dir():
        raise ValueError("Sandbox roots must be existing directories")
    snapshot = grants.snapshot()
    if grant_snapshot is not None and grant_snapshot != snapshot:
        raise PermissionError("folder access changed while preparing the Sandbox command")
    if grants.is_private_path(working):
        raise PermissionError("Sandbox cwd is inside private runtime storage")

    mounts: list[tuple[Path, bool, int]] = [(home, True, 0)]
    if snapshot.enabled and include_folder_grants:
        for grant in snapshot.grants:
            root = Path(grant.path)
            if root.is_dir() and root.resolve(strict=True) == root:
                if _overlaps_system_control(root):
                    raise ValueError("Sandbox write rules cannot cover /proc, /sys, /dev, or /run")
                mounts.append((root, grant.file_access == "read_write", 1))
    for raw_path, file_access, commands in task_roots:
        root = Path(raw_path).expanduser()
        if (
            not root.is_dir() or root.resolve(strict=True) != root
            or file_access not in {"read_only", "read_write"}
            or type(commands) is not bool or grants.is_private_path(root)
            or _overlaps_system_control(root)
        ):
            raise ValueError("Once permission root changed while preparing the command")
        mounts.append((root, file_access == "read_write", 2))

    args = ["bwrap", "--new-session", "--die-with-parent", "--unshare-pid",
            "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--tmpfs", "/run", "--remount-ro", "/sys"]
    # Resolver configuration may be a symlink into /run, whose sockets are
    # hidden. Preserve only its actual regular file so network DNS still works.
    resolver = Path("/etc/resolv.conf").resolve(strict=False)
    if resolver.is_file() and resolver.is_relative_to(Path("/run")):
        args += _ensure_mountpoint_args(resolver.parent, reset_roots=(Path("/run"),))
        args += ["--ro-bind", str(resolver), str(resolver)]
    # Hide live gateway state, credentials, catalogs and permission authority.
    # Recreate only the agent workspace inside the hidden instance.
    instance = grants.sandbox_instance_root.resolve(strict=False)
    args += _ensure_mountpoint_args(instance, reset_roots=(Path("/tmp"),))
    args += ["--tmpfs", str(instance)]
    for root, writable, _priority in sorted(mounts, key=lambda row: (len(row[0].parts), str(row[0]), row[2])):
        args += _ensure_mountpoint_args(root, reset_roots=(instance, Path("/tmp")))
        args += ["--bind" if writable else "--ro-bind", str(root), str(root)]

    # These roots cannot be restored by a broad writable ancestor or Once grant.
    for root in sorted({path.expanduser().resolve(strict=False) for path in excluded_roots}, key=lambda path: len(path.parts)):
        if root.exists():
            if _overlaps_system_control(root) or any(
                Path(runtime).resolve(strict=False).is_relative_to(root)
                for runtime in ("/usr", "/bin", "/lib", "/lib64", "/etc/resolv.conf")
            ):
                raise ValueError("configured repository root cannot be masked from the managed runtime")
            args += _ensure_mountpoint_args(root, reset_roots=(instance, Path("/tmp")))
            args += ["--tmpfs", str(root)]
    private_roots = (instance, *grants.sandbox_private_roots)
    for private in sorted(set(private_roots), key=lambda path: (len(path.parts), str(path))):
        # The instance was already masked, then the workspace was restored.
        if private == instance or (home.is_relative_to(private) and private != home):
            continue
        if private.is_file():
            args += _ensure_mountpoint_args(private.parent, reset_roots=(instance, Path("/tmp")))
            args += ["--ro-bind", "/dev/null", str(private)]
        elif private.exists():
            args += _ensure_mountpoint_args(private, reset_roots=(instance, Path("/tmp")))
            args += ["--tmpfs", str(private)]
    args += ["--setenv", "TMPDIR", "/tmp",
             "--setenv", "XDG_CACHE_HOME", "/tmp/nanobot-cache",
             "--setenv", "npm_config_cache", "/tmp/nanobot-cache/npm",
             "--setenv", "XDG_DATA_HOME", "/tmp/nanobot-data",
             "--setenv", "XDG_STATE_HOME", "/tmp/nanobot-state",
             "--setenv", "COREPACK_HOME", "/tmp/nanobot-cache/corepack",
             "--setenv", "COREPACK_DEFAULT_TO_LATEST", "0",
             "--setenv", "BUN_INSTALL_CACHE_DIR", "/tmp/nanobot-cache/bun",
             "--setenv", "PIP_CACHE_DIR", "/tmp/nanobot-cache/pip",
             "--setenv", "UV_CACHE_DIR", "/tmp/nanobot-cache/uv",
             "--setenv", "GOCACHE", "/tmp/nanobot-cache/go-build",
             "--setenv", "GOMODCACHE", "/tmp/nanobot-cache/go-mod",
             "--setenv", "GRADLE_USER_HOME", "/tmp/nanobot-cache/gradle",
             "--chdir", str(working)]
    # Moving XDG caches must not hide already installed Corepack runtimes.
    # Seed only public package-manager metadata and version links, inside the
    # Sandbox. New versions can download into private writable cache storage.
    cache_base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    corepack_source = os.environ.get("COREPACK_HOME") or str(cache_base / "node" / "corepack")
    bootstrap = (
        "set -e\n"
        f"nanobot_corepack_source={shlex.quote(corepack_source)}\n"
        'mkdir -p "$COREPACK_HOME/v1"\n'
        'if [ -f "$nanobot_corepack_source/lastKnownGood.json" ]; then\n'
        '  cp "$nanobot_corepack_source/lastKnownGood.json" "$COREPACK_HOME/lastKnownGood.json"\n'
        'fi\n'
        'for nanobot_manager in npm pnpm yarn; do\n'
        '  mkdir -p "$COREPACK_HOME/v1/$nanobot_manager"\n'
        '  nanobot_version_count=0\n'
        '  for nanobot_version in "$nanobot_corepack_source/v1/$nanobot_manager/"*; do\n'
        '    [ -d "$nanobot_version" ] || continue\n'
        '    ln -s "$nanobot_version" "$COREPACK_HOME/v1/$nanobot_manager/${nanobot_version##*/}"\n'
        '    nanobot_version_count=$((nanobot_version_count + 1))\n'
        '    [ "$nanobot_version_count" -lt 128 ] || break\n'
        '  done\n'
        'done\n'
        'exec "$@"'
    )
    args += ["--", "/bin/sh", "-c", bootstrap, "nanobot-sandbox",
             shell_program or shutil.which("bash") or "/bin/sh"]
    if login:
        args.append("-l")
    args += ["-c", command]
    return shlex.join(args)


def _ensure_mountpoint_args(
    path: Path,
    *,
    reset_roots: Iterable[Path] = (),
    backing_roots: Iterable[Path] = (),
) -> list[str]:
    """Create empty destination ancestors after a private parent was masked."""
    matching = [root for root in reset_roots if _is_relative_to(path, root)]
    backed = [root for root in backing_roots if _is_relative_to(path, root)]
    reset_boundary = max(matching, key=lambda root: len(root.parts), default=None)
    backing_boundary = max(backed, key=lambda root: len(root.parts), default=None)
    if reset_boundary is not None and (
        backing_boundary is None
        or len(reset_boundary.parts) >= len(backing_boundary.parts)
    ):
        boundary = reset_boundary
        try:
            relative = path.relative_to(boundary)
        except ValueError:
            relative = Path()
        current = boundary
        args: list[str] = []
        for part in relative.parts:
            current = current / part
            args.extend(("--dir", str(current)))
        return args
    missing: list[Path] = []
    current = path
    while current != Path("/") and not current.exists():
        missing.append(current)
        current = current.parent
    return [part for directory in reversed(missing) for part in ("--dir", str(directory))]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _overlaps_system_control(path: Path) -> bool:
    return any(
        _is_relative_to(path, Path(root)) or _is_relative_to(Path(root), path)
        for root in ("/proc", "/sys", "/dev", "/run")
    )
