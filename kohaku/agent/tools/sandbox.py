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
from typing import Callable, Iterable

from kohaku.agent.tools.executor_capabilities import current_platform, macos_seatbelt_ready
from kohaku.agent.tools.seatbelt_profile import (
    LAUNCH_TMP_PARAM,
    SeatbeltConfinement,
    managed_cache_env,
    managed_profile,
)
from kohaku.config.paths import get_media_dir
from kohaku.security import darwin_proc
from kohaku.security.folder_grants import FolderGrantSnapshot, FolderGrantView

_DARWIN_SYSTEM_CONTROL = ("/dev", "/System", "/private/var/run")
_DARWIN_RUNTIME = ("/usr", "/bin", "/sbin", "/private/etc", "/Library/Developer")


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
    input_files: Iterable[tuple[Path, str]] = (),
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
    for source, target in input_files:
        args += _ensure_mountpoint_args(
            Path(target).parent,
            reset_roots=(Path("/"),),
        )
        args += ["--ro-bind", str(source), target]
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
    input_files: Iterable[tuple[Path, str]] = (),
) -> str:
    """Wrap *command* using the named sandbox backend."""
    if backend := _BACKENDS.get(sandbox):
        return backend(
            command,
            workspace,
            cwd,
            sandbox_ro_binds=sandbox_ro_binds,
            sandbox_rw_binds=sandbox_rw_binds,
            input_files=input_files,
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
    input_files: Iterable[tuple[Path, str]] = (),
    shell_program: str | None = None,
    login: bool = False,
    allow_local_binding: bool = True,
    allow_unix_sockets: Iterable[str] = (),
) -> str:
    """Read-only host view with approved writable folders and private temp/cache.

    The legacy ``commands`` bit remains in stored grants for compatibility; it
    does not gate Sandbox programs. More-specific file rules override parents.
    Configured managed repositories retain their separate protected-path/store
    policy. They stay hidden from shell processes; use file tools for them.
    """
    if not managed_bwrap_available():
        raise RuntimeError("Sandbox requires a working bubblewrap installation")
    home, working, mounts = _managed_mounts(
        cwd=cwd,
        agent_home=agent_home,
        grants=grants,
        include_folder_grants=include_folder_grants,
        grant_snapshot=grant_snapshot,
        task_roots=task_roots,
        overlaps_system_control=_overlaps_system_control,
        system_control_error="Sandbox write rules cannot cover /proc, /sys, /dev, or /run",
    )

    args = ["bwrap", "--new-session", "--die-with-parent", "--unshare-pid",
            "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--tmpfs", "/run", "--remount-ro", "/sys"]
    if not allow_local_binding:
        args.append("--unshare-net")
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
    repository_roots = {path.expanduser().resolve(strict=False) for path in excluded_roots}
    for root in sorted(repository_roots, key=lambda path: len(path.parts)):
        if root.exists():
            if _overlaps_system_control(root) or any(
                Path(runtime).resolve(strict=False).is_relative_to(root)
                for runtime in ("/usr", "/bin", "/lib", "/lib64", "/etc/resolv.conf")
            ):
                raise ValueError("configured repository root cannot be masked from the managed runtime")
            args += _ensure_mountpoint_args(root, reset_roots=(instance, Path("/tmp")))
            args += ["--tmpfs", str(root)]
    for private in _private_masks((instance, *grants.sandbox_private_roots), instance, home):
        if private.is_file():
            args += _ensure_mountpoint_args(private.parent, reset_roots=(instance, Path("/tmp")))
            args += ["--ro-bind", "/dev/null", str(private)]
        elif private.exists():
            args += _ensure_mountpoint_args(private, reset_roots=(instance, Path("/tmp")))
            args += ["--tmpfs", str(private)]
    for socket_path, endpoint in _managed_sockets(
        grants, allow_unix_sockets, (Path("/proc"), Path("/sys"), Path("/dev"), *repository_roots),
    ):
        args += _ensure_mountpoint_args(
            endpoint.parent,
            reset_roots=(instance, Path("/tmp"), Path("/run")),
            backing_roots=(root for root, _, _ in mounts),
        )
        args += ["--ro-bind", socket_path, socket_path]
    # Exact per-invocation read snapshots are restored after private masks so
    # the approved files remain readable at their advertised paths.
    for source, target in input_files:
        destination = _managed_input_target(grants, target)
        args += _ensure_mountpoint_args(
            destination.parent,
            reset_roots=(instance, Path("/tmp")),
        )
        args += ["--ro-bind", str(source), str(destination)]
    for name, value in managed_cache_env("/tmp").items():
        args += ["--setenv", name, value]
    args += ["--chdir", str(working)]
    args += ["--", *_managed_argv(command, shell_program, login)]
    return shlex.join(args)


def _managed_mounts(
    *,
    cwd: str | Path,
    agent_home: str | Path,
    grants: FolderGrantView,
    include_folder_grants: bool,
    grant_snapshot: FolderGrantSnapshot | None,
    task_roots: Iterable[tuple[str, str, bool]],
    overlaps_system_control: Callable[[Path], bool],
    system_control_error: str,
) -> tuple[Path, Path, list[tuple[Path, bool, int]]]:
    """Validate a managed command's folders: (agent home, cwd, mounts).

    Each mount is (folder, writable, priority); both backends restore them
    least specific first over the hidden folders.
    """
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
                if overlaps_system_control(root):
                    raise ValueError(system_control_error)
                mounts.append((root, grant.file_access == "read_write", 1))
    for raw_path, file_access, commands in task_roots:
        root = Path(raw_path).expanduser()
        if (
            not root.is_dir() or root.resolve(strict=True) != root
            or file_access not in {"read_only", "read_write"}
            or type(commands) is not bool or grants.is_private_path(root)
            or overlaps_system_control(root)
        ):
            raise ValueError("Once permission root changed while preparing the command")
        mounts.append((root, file_access == "read_write", 2))
    return home, working, mounts


def _private_masks(private_roots: Iterable[Path], instance: Path, home: Path) -> list[Path]:
    """Private runtime paths that stay hidden even under a writable parent."""
    return [
        private
        for private in sorted(set(private_roots), key=lambda path: (len(path.parts), str(path)))
        # The instance was already masked, then the workspace was restored.
        if not (private == instance or (home.is_relative_to(private) and private != home))
    ]


def _managed_sockets(
    grants: FolderGrantView,
    allow_unix_sockets: Iterable[str],
    forbidden_roots: Iterable[Path],
) -> list[tuple[str, Path]]:
    """Validate the configured Unix sockets and keep those that exist."""
    forbidden = tuple(forbidden_roots)
    sockets: list[tuple[str, Path]] = []
    for socket_path in _normalize_bind_paths(allow_unix_sockets):
        endpoint = Path(socket_path)
        if grants.is_private_path(endpoint) or any(
            endpoint.is_relative_to(root) for root in forbidden
        ):
            raise PermissionError("Sandbox socket is inside a protected path")
        if not endpoint.exists():
            continue
        if not endpoint.is_socket():
            raise ValueError("Sandbox socket path must identify a Unix socket")
        sockets.append((socket_path, endpoint))
    return sockets


def _managed_input_target(grants: FolderGrantView, target: str) -> Path:
    """Check one approved input file's advertised path before restoring it."""
    destination = Path(target)
    if not destination.is_absolute() or destination.is_symlink():
        raise PermissionError("Sandbox input file target changed")
    if grants.is_private_path(destination) and not destination.is_relative_to(
        get_media_dir().resolve(strict=False)
    ):
        raise PermissionError("Sandbox input file target is private")
    return destination


def _managed_argv(command: str, shell_program: str | None, login: bool) -> tuple[str, ...]:
    """The confined process: seed tool runtimes, then run the command's shell."""
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
    shell = shell_program or shutil.which("bash") or "/bin/sh"
    login_args = ("-l",) if login else ()
    return ("/bin/sh", "-c", bootstrap, "nanobot-sandbox", shell, *login_args, "-c", command)


# Succeeds only if the confined process can read its own arguments but not those
# of the gateway outside the sandbox (sysctl KERN_PROCARGS2).
_PROCESS_PRIVACY_CHECK = (
    "import ctypes, os, sys\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "def readable(pid):\n"
    "    mib = (ctypes.c_int * 3)(1, 49, pid)\n"
    "    size = ctypes.c_size_t(1 << 20)\n"
    "    buffer = ctypes.create_string_buffer(size.value)\n"
    "    return libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) == 0\n"
    "sys.exit(0 if readable(os.getpid()) and not readable(int(sys.argv[1])) else 1)\n"
)


@functools.lru_cache(maxsize=1)
def _seatbelt_profile_probe() -> bool:
    """Return true when this host runs the managed profile and it keeps processes private.

    macOS leaves some process inspection open under (deny default), so the rule
    that closes it is checked on the live host rather than assumed.
    """
    rules, params = managed_profile(
        hidden=(), mounts=(), protected=(), input_files=(), unix_sockets=(), allow_network=True,
    )
    command = [darwin_proc.SANDBOX_EXEC, "-p", rules]
    for name, value in (*params, (LAUNCH_TMP_PARAM, "/private/tmp/kohaku-profile-probe")):
        command += ["-D", f"{name}={value}"]
    command += ["--", sys.executable, "-I", "-S", "-c", _PROCESS_PRIVACY_CHECK, str(os.getpid())]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def managed_seatbelt_available() -> bool:
    """Return true only when macOS can confine and track a managed command.

    This needs the operator's switch and the host self-test, because the same
    Seatbelt markers prove the confined process tree.
    """
    return current_platform() == "darwin" and macos_seatbelt_ready() and _seatbelt_profile_probe()


def prepare_managed_seatbelt(
    command: str,
    *,
    cwd: str | Path,
    agent_home: str | Path,
    grants: FolderGrantView,
    include_folder_grants: bool = True,
    grant_snapshot: FolderGrantSnapshot | None = None,
    excluded_roots: Iterable[Path] = (),
    task_roots: Iterable[tuple[str, str, bool]] = (),
    input_files: Iterable[tuple[Path, str]] = (),
    shell_program: str | None = None,
    login: bool = False,
    allow_local_binding: bool = True,
    allow_unix_sockets: Iterable[str] = (),
) -> SeatbeltConfinement:
    """The macOS form of wrap_managed_command: the same folders as Seatbelt rules.

    Differences from bubblewrap: hidden folders are refused rather than replaced
    by empty ones, and input files are read live at their approved path.
    """
    if not managed_seatbelt_available():
        raise RuntimeError("Sandbox requires a working macOS Seatbelt executor")
    home, working, mounts = _managed_mounts(
        cwd=cwd,
        agent_home=agent_home,
        grants=grants,
        include_folder_grants=include_folder_grants,
        grant_snapshot=grant_snapshot,
        task_roots=task_roots,
        overlaps_system_control=_overlaps_darwin_system_control,
        system_control_error="Sandbox write rules cannot cover /dev, /System, or /private/var/run",
    )
    # Seatbelt matches on-disk paths, so every rule must use that spelling.
    canonical = darwin_proc.canonical_path
    home, working = canonical(home), canonical(working)
    if grants.is_private_path(working):
        raise PermissionError("Sandbox cwd is inside private runtime storage")
    instance = canonical(grants.sandbox_instance_root.resolve(strict=False))

    protected: list[Path] = []
    repository_roots = sorted(
        {canonical(path.expanduser().resolve(strict=False)) for path in excluded_roots},
        key=lambda path: (len(path.parts), str(path)),
    )
    for root in repository_roots:
        if root.exists():
            if _overlaps_darwin_system_control(root) or any(
                _is_relative_to(Path(runtime), root) for runtime in _DARWIN_RUNTIME
            ):
                raise ValueError("configured repository root cannot be masked from the managed runtime")
            protected.append(root)
    private = (instance, *(canonical(path) for path in grants.sandbox_private_roots))
    protected += _private_masks(private, instance, home)

    ordered = sorted(mounts, key=lambda row: (len(row[0].parts), str(row[0]), row[2]))
    sockets = [
        canonical(endpoint)
        for _, endpoint in _managed_sockets(
            grants, allow_unix_sockets, (Path("/dev"), Path("/System"), *repository_roots),
        )
    ]
    inputs: list[Path] = []
    for _source, target in input_files:
        inputs.append(canonical(_managed_input_target(grants, target)))
    rules, params = managed_profile(
        hidden=(*_darwin_temp_roots(), instance),
        mounts=tuple((canonical(root), writable) for root, writable, _priority in ordered),
        protected=tuple(protected),
        input_files=tuple(inputs),
        unix_sockets=tuple(sockets),
        allow_network=allow_local_binding,
    )
    return SeatbeltConfinement(
        rules=rules,
        params=params,
        argv=_managed_argv(command, shell_program, login),
        cwd=str(working),
    )


def _darwin_temp_roots() -> tuple[Path, ...]:
    """Shared temporary folders a managed command must not see."""
    roots = [Path("/private/tmp")]
    user_home = darwin_proc.canonical_path(Path.home())
    for raw in (darwin_proc.user_temp_dir(), os.environ.get("TMPDIR")):
        if not raw or not Path(raw).is_absolute():
            continue
        root = darwin_proc.canonical_path(Path(raw))
        # A TMPDIR such as / or the home folder is not a temporary folder.
        if not _is_relative_to(user_home, root) and root not in roots:
            roots.append(root)
    return tuple(roots)


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


def _overlaps_darwin_system_control(path: Path) -> bool:
    resolved = darwin_proc.canonical_path(path)
    return any(
        _is_relative_to(resolved, Path(root)) or _is_relative_to(Path(root), resolved)
        for root in _DARWIN_SYSTEM_CONTROL
    )


def _overlaps_system_control(path: Path) -> bool:
    return any(
        _is_relative_to(path, Path(root)) or _is_relative_to(Path(root), path)
        for root in ("/proc", "/sys", "/dev", "/run")
    )
