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
) -> str:
    """Build a bwrap command with only agent home and command-enabled grants mounted."""
    if not managed_bwrap_available():
        raise RuntimeError("managed bubblewrap is unavailable")
    home = Path(agent_home).expanduser().resolve(strict=True)
    working = Path(cwd).expanduser().resolve(strict=True)
    if not home.is_dir() or not working.is_dir():
        raise ValueError("managed command roots must be existing directories")
    snapshot = grants.snapshot()
    if grant_snapshot is not None and grant_snapshot != snapshot:
        raise PermissionError("folder access changed while preparing the managed command")
    if grant_snapshot is not None:
        snapshot = grant_snapshot
    if grants.is_private_path(working):
        raise PermissionError("managed command cwd is inside private runtime storage")
    checked_task_roots: list[tuple[Path, bool, bool]] = []
    for raw_path, file_access, commands in task_roots:
        root = Path(raw_path).expanduser()
        canonical = root.resolve(strict=True)
        if (
            not canonical.is_dir()
            or canonical != root
            or file_access not in {"read_only", "read_write"}
            or type(commands) is not bool
            or grants.is_private_path(canonical)
        ):
            raise ValueError("Once permission root changed while preparing the command")
        if commands and _overlaps_system_control(canonical):
            raise ValueError("Once command root cannot cover /proc, /sys, /dev, or /run")
        checked_task_roots.append((canonical, commands, file_access == "read_write"))

    grant = _effective_grant_for(working, snapshot, include_folder_grants)
    task_choice = max(
        (row for row in checked_task_roots if _is_relative_to(working, row[0])),
        key=lambda row: len(row[0].parts),
        default=None,
    )
    task_wins = task_choice is not None and (
        grant is None or len(task_choice[0].parts) >= len(Path(grant.path).parts)
    )
    external_grant_commands = (
        task_choice[1] if task_wins and task_choice is not None
        else grant.commands if grant is not None else False
    )
    if not _is_relative_to(working, home):
        if not external_grant_commands:
            raise PermissionError("managed command cwd is not covered by command access")
    elif (task_wins and task_choice is not None and not task_choice[1]) or (
        not task_wins and grant is not None and not grant.commands
    ):
        raise PermissionError("managed command cwd is covered by a command-disabled folder grant")

    resolved_exclusions = {
        path.expanduser().resolve(strict=False) for path in excluded_roots
    }
    runtime_roots = {
        Path(source).resolve(strict=False)
        for source in (
            "/proc", "/dev", "/tmp", "/sys", "/run",
            "/usr", "/bin", "/lib", "/lib64", "/usr/local/bin",
            "/etc/alternatives", "/etc/ssl/certs", "/etc/pki/tls/certs",
            "/etc/resolv.conf", "/etc/ld.so.cache",
        )
    }
    for root in resolved_exclusions:
        if root == Path("/") or any(
            runtime == root or root in runtime.parents for runtime in runtime_roots
        ):
            raise ValueError("configured repository root cannot be masked from the managed runtime")

    args = ["bwrap", "--new-session", "--die-with-parent", "--unshare-pid"]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--tmpfs", "/sys", "--tmpfs", "/run",
    ]
    for source in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(source).exists():
            args += ["--ro-bind", source, source]
    for source in (
        "/etc/alternatives",
        "/etc/ssl/certs",
        "/etc/pki/tls/certs",
        "/etc/resolv.conf",
        "/etc/ld.so.cache",
    ):
        if Path(source).exists():
            args += ["--ro-bind-try", source, source]

    private_roots = grants.sandbox_private_roots
    masked_roots = {
        Path("/home"),
        Path("/root"),
        grants.sandbox_instance_root.resolve(strict=False),
        *(root.resolve(strict=False) for root in private_roots),
    }
    masked_roots = {root for root in masked_roots if not root.is_file()}
    # Mask broad private ancestors first. Exact allowed roots are reconstructed
    # below, then private descendants of agent home are masked again.
    masked_paths: list[Path] = [Path("/tmp")]
    for root in sorted(masked_roots, key=lambda path: (len(path.parts), str(path))):
        if root == Path("/") or root in {Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64")}:
            continue
        args.extend(_ensure_mountpoint_args(root, reset_roots=masked_paths))
        args += ["--tmpfs", str(root)]
        masked_paths.append(root)

    mount_roots: list[tuple[Path, bool, bool, int]] = [(home, True, True, 0)]
    if snapshot.enabled and include_folder_grants:
        for grant in snapshot.grants:
            root = Path(grant.path)
            if not root.is_dir() or root.resolve(strict=True) != root:
                continue
            if grant.commands and _overlaps_system_control(root):
                raise ValueError("command grants cannot cover /proc, /sys, /dev, or /run")
            mount_roots.append((root, grant.commands, grant.file_access == "read_write", 1))
    for root, commands, write_allowed in checked_task_roots:
        mount_roots.append((root, commands, write_allowed, 2))
    mount_roots.sort(key=lambda item: (len(item[0].parts), str(item[0]), item[3]))
    empty_roots: list[Path] = list(masked_paths)
    backing_roots: list[Path] = []
    for root, commands_allowed, write_allowed, _priority in mount_roots:
        args.extend(_ensure_mountpoint_args(
            root,
            reset_roots=empty_roots,
            backing_roots=backing_roots,
        ))
        if root == home:
            args += ["--bind", str(root), str(root)]
        elif not commands_allowed:
            args += ["--tmpfs", str(root)]
        elif write_allowed:
            args += ["--bind", str(root), str(root)]
        else:
            args += ["--ro-bind", str(root), str(root)]
        if commands_allowed:
            backing_roots.append(root)
        else:
            empty_roots.append(root)

    for private in sorted(private_roots, key=lambda path: (len(path.parts), str(path))):
        if private.is_file():
            if private.exists():
                args += ["--ro-bind", "/dev/null", str(private)]
            continue
        try:
            private.relative_to(home)
        except ValueError:
            continue
        args.extend(_ensure_mountpoint_args(
            private,
            reset_roots=empty_roots,
            backing_roots=backing_roots,
        ))
        args += ["--tmpfs", str(private)]
        empty_roots.append(private)

    # A broad command grant can contain configured repositories. Mask those
    # roots after all home/grant mounts so nested grants cannot re-expose them.
    for root in sorted(
        resolved_exclusions,
        key=lambda path: (len(path.parts), str(path)),
    ):
        args.extend(_ensure_mountpoint_args(
            root,
            reset_roots=empty_roots,
            backing_roots=backing_roots,
        ))
        args += ["--tmpfs", str(root)]
        empty_roots.append(root)

    args += [
        "--setenv", "HOME", str(home),
        "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
        "--chdir", str(working),
        "--", "sh", "-c", command,
    ]
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


def _effective_grant_for(
    path: Path,
    snapshot: FolderGrantSnapshot,
    include_folder_grants: bool,
):
    if not include_folder_grants or not snapshot.enabled:
        return None
    matches = [
        grant
        for grant in snapshot.grants
        if Path(grant.path).is_dir()
        and Path(grant.path).resolve(strict=True) == Path(grant.path)
        and _is_relative_to(path, Path(grant.path))
    ]
    return max(matches, key=lambda grant: len(Path(grant.path).parts), default=None)
