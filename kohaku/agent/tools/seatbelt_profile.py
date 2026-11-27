"""Seatbelt rules for managed commands on macOS (A3-2M, WP5).

Pure text generation: nothing here launches a process. Paths reach the profile
only as sandbox-exec parameters, so a path can never be read as profile syntax.
Seatbelt applies the last matching rule, so the order below is the policy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

_PROFILE_DIR = Path(__file__).with_name("seatbelt")

# The supervisor supplies this parameter: the launch's private temporary folder.
LAUNCH_TMP_PARAM = "KOHAKU_LAUNCH_TMP"
# F_MAKECOMPRESSED and F_TRANSFEREXTENTS change files through read-only descriptors.
_DENIED_FCNTLS = "80 110"
# TIOCSTI pushes input into a terminal that outlives the command.
_TIOCSTI = 0x80017472


@dataclass(frozen=True, slots=True)
class SeatbeltConfinement:
    """One managed launch: its rules, their parameters and the command to run."""

    rules: str
    params: tuple[tuple[str, str], ...]
    argv: tuple[str, ...]
    cwd: str


class _Params:
    def __init__(self, prefix: str = "P") -> None:
        self._prefix = prefix
        self._names: dict[str, str] = {}

    def ref(self, path: Path) -> str:
        value = str(path)
        name = self._names.get(value)
        if name is None:
            name = self._names[value] = f"{self._prefix}{len(self._names)}"
        return f'(param "{name}")'

    def items(self) -> tuple[tuple[str, str], ...]:
        return tuple((name, value) for value, name in self._names.items())


def _by_depth(paths: Sequence[Path]) -> list[Path]:
    return sorted(set(paths), key=lambda path: (len(path.parts), str(path)))


def managed_profile(
    *,
    hidden: Sequence[Path],
    mounts: Sequence[tuple[Path, bool]],
    protected: Sequence[Path],
    input_files: Sequence[Path],
    unix_sockets: Sequence[Path],
    allow_network: bool,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Build the rules and parameters for one managed launch.

    hidden folders are unreadable unless a mount at or inside them restores
    part of them. mounts are (folder, writable) pairs ordered least specific
    first. protected paths stay unreadable whatever the mounts say. input_files
    are then restored read-only. Callers pass canonical paths.
    """
    params = _Params()
    rules = [(_PROFILE_DIR / "managed.sbpl").read_text(encoding="utf-8")]
    if allow_network:
        rules.append((_PROFILE_DIR / "network.sbpl").read_text(encoding="utf-8"))
    rules.append("(allow file-read*)")
    # The blanket read above would also reach the user's other terminals, where a
    # command could read typed input or change the mode. Only its own ptys stay open.
    rules.append('(deny file-read* file-ioctl (regex #"^/dev/ttys[0-9]+"))')
    rules.append(
        '(allow file-read* file-ioctl (require-all (regex #"^/dev/ttys[0-9]+") '
        '(extension "com.apple.sandbox.pty")))'
    )

    # Local sockets follow the file rules: usable where the command may write.
    sockets: list[str] = []
    # Least specific first, so a hidden folder inside a mount stays hidden and
    # a mount inside a hidden folder is restored.
    layers: list[tuple[Path, bool | None]] = [(path, None) for path in _by_depth(hidden)]
    layers += mounts
    for path, writable in sorted(layers, key=lambda layer: len(layer[0].parts)):
        ref = params.ref(path)
        if writable is None:
            rules.append(f"(deny file-read* file-write* (subpath {ref}))")
        elif writable:
            rules.append(f"(allow file-read* file-write* (subpath {ref}))")
        else:
            rules.append(f"(allow file-read* (subpath {ref}))")
            rules.append(f"(deny file-write* (subpath {ref}))")
        sockets.append(_socket_rule("allow" if writable else "deny", path, params))
    launch_tmp = f'(param "{LAUNCH_TMP_PARAM}")'
    rules.append(f"(allow file-read* file-write* (subpath {launch_tmp}))")
    sockets.append(f"(allow network-bind (local unix-socket (subpath {launch_tmp})))")
    sockets.append(f"(allow network-outbound (remote unix-socket (subpath {launch_tmp})))")
    for path in _by_depth(unix_sockets):
        sockets.append(f"(allow network-outbound (remote unix-socket (literal {params.ref(path)})))")
    rules.append("(allow system-socket (socket-domain AF_UNIX))")
    rules.extend(sockets)

    # Path resolution needs metadata on the folders leading to each restored path.
    restored = [path for path, _ in mounts] + list(input_files) + list(unix_sockets)
    ancestors = _by_depth([parent for path in restored for parent in path.parents])
    for path in [Path("/private/tmp"), *ancestors]:
        rules.append(f"(allow file-read-metadata (literal {params.ref(path)}))")

    for path in _by_depth(protected):
        rules.append(f"(deny file-read* file-write* (subpath {params.ref(path)}))")
        rules.append(_socket_rule("deny", path, params))
    for path in _by_depth(input_files):
        ref = params.ref(path)
        rules.append(f"(allow file-read* (literal {ref}))")
        rules.append(f"(deny file-write* (literal {ref}))")

    # Renaming a folder would carry the paths inside it out of their rules.
    anchored = [*hidden, *protected, *input_files, *(path for path, _ in mounts)]
    rules.extend(unlink_denies(
        _by_depth([item for path in anchored for item in (path, *path.parents)]), params.ref,
    ))
    rules.append(f"(deny system-fcntl (fcntl-command {_DENIED_FCNTLS}))")
    rules.append(f"(deny file-ioctl (ioctl-command {_TIOCSTI}))")
    rules.append('(deny mach-lookup (xpc-service-name-prefix ""))')
    return "\n".join(rules) + "\n", params.items()


def runtime_denies(paths: Sequence[Path]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Hide the supervisor's own folders; appended after a launch's other rules.

    Uses its own parameter names so it can follow any managed_profile output.
    Callers pass canonical paths.
    """
    params = _Params("R")
    rules: list[str] = []
    for path in _by_depth(paths):
        rules.append(f"(deny file-read* file-write* (subpath {params.ref(path)}))")
        rules.append(_socket_rule("deny", path, params))
    rules.extend(unlink_denies(
        _by_depth([item for path in paths for item in (path, *path.parents)]), params.ref,
    ))
    return "\n".join(rules) + "\n", params.items()


def _socket_rule(action: str, path: Path, params: _Params) -> str:
    ref = params.ref(path)
    return (
        f"({action} network-bind (local unix-socket (subpath {ref})))\n"
        f"({action} network-outbound (remote unix-socket (subpath {ref})))"
    )


def unlink_denies(paths: Sequence[Path], ref: Callable[[Path], str]) -> list[str]:
    """Forbid removing or renaming each of these folders."""
    return [
        f"(deny file-write-unlink (require-all (vnode-type DIRECTORY) (literal {ref(path)})))"
        for path in paths
        if path != Path("/")
    ]


def managed_cache_env(temp_root: str) -> dict[str, str]:
    """Point temporary files and tool caches at a launch's private folder."""
    cache = f"{temp_root}/nanobot-cache"
    return {
        "TMPDIR": temp_root,
        "XDG_CACHE_HOME": cache,
        "npm_config_cache": f"{cache}/npm",
        "XDG_DATA_HOME": f"{temp_root}/nanobot-data",
        "XDG_STATE_HOME": f"{temp_root}/nanobot-state",
        "COREPACK_HOME": f"{cache}/corepack",
        "COREPACK_DEFAULT_TO_LATEST": "0",
        "BUN_INSTALL_CACHE_DIR": f"{cache}/bun",
        "PIP_CACHE_DIR": f"{cache}/pip",
        "UV_CACHE_DIR": f"{cache}/uv",
        "GOCACHE": f"{cache}/go-build",
        "GOMODCACHE": f"{cache}/go-mod",
        "GRADLE_USER_HOME": f"{cache}/gradle",
    }
