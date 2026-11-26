"""macOS supervisor that proves a command tree quiescent through Seatbelt markers.

The relay, gate, status protocol and receipt come from native_process_supervisor.
This module supplies its backend: the command root runs under sandbox-exec with
a per-launch profile, and the launch domain (seatbelt_tracking) kills and proves
the tree. The supervisor itself is not sandboxed.

It writes a receipt only for a proven tree. When a requested kill cannot be
proven within its deadline, it exits without one, so the admission ledger
records the launch as unknown.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from kohaku.agent.tools.native_process_supervisor import (
    NativeLaunchConfig,
    run_supervisor,
    shell_argv,
    spawn_command_root,
)
from kohaku.agent.tools.seatbelt_tracking import (
    NotQuiescent,
    Quiescent,
    host_launch_domain,
)
from kohaku.security.darwin_proc import SANDBOX_EXEC

SANDBOX_MARKER_ENV = "KOHAKU_SANDBOX"
SANDBOX_MARKER_VALUE = "seatbelt"
KILL_DEADLINE_SECONDS = 10.0
_KILL_INTERVAL_SECONDS = 0.02
_FULL_PROFILE = Path(__file__).with_name("seatbelt") / "full.sbpl"
NESTED_SANDBOX_SIGNATURE = b"sandbox_apply: Operation not permitted"
NESTED_SANDBOX_NOTE = (
    b"\n[kohaku] This command tried to apply its own macOS sandbox. A different nested "
    b"Seatbelt sandbox is not supported inside Kohaku commands. Tools can detect this "
    b"through KOHAKU_SANDBOX=seatbelt.\n"
)


class TreeDomain(Protocol):
    def profile_markers(self) -> str: ...

    def poll(self) -> bool: ...

    def kill_step(self, deadline: float) -> Quiescent | NotQuiescent | None: ...


class DomainFactory(Protocol):
    def __call__(
        self,
        launch_id: str,
        *,
        root_exited: Callable[[], bool],
        protected: Iterable[int],
    ) -> TreeDomain: ...


def profile_directory(receipt_dir: Path) -> Path:
    """Private runtime storage for launch profiles, beside the receipts."""
    return receipt_dir.with_name(f"{receipt_dir.name}-profiles")


def write_launch_profile(directory: Path, launch_id: str, rules: str) -> Path:
    """Create one owner-only profile file; never reuse or follow an existing path."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise RuntimeError("Seatbelt profile directory is not private")
    os.chmod(directory, 0o700)
    path = directory / f"{launch_id}.sb"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(rules)
    return path


class SeatbeltBackend:
    """Launch under sandbox-exec; kill and prove the tree through its markers."""

    def __init__(
        self,
        *,
        domain_factory: DomainFactory = host_launch_domain,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._domain_factory = domain_factory
        self._clock = clock
        self._domain: TreeDomain | None = None
        self._profile: Path | None = None
        self._child: subprocess.Popen[bytes] | None = None
        self._deadline: float | None = None
        self._next_step = 0.0
        self._proven = False
        self._stderr_tail = b""
        self._nested_reported = False

    def prepare(self, payload: NativeLaunchConfig) -> None:
        launch_id = payload["launch_id"]
        self._domain = self._domain_factory(
            launch_id, root_exited=self._root_exited, protected=(os.getppid(),),
        )
        rules = _FULL_PROFILE.read_text(encoding="utf-8") + self._domain.profile_markers()
        self._profile = write_launch_profile(
            profile_directory(Path(payload["receipt_dir"])), launch_id, rules,
        )

    def spawn_root(self, payload: NativeLaunchConfig) -> subprocess.Popen[bytes]:
        if self._profile is None:
            raise RuntimeError("Seatbelt launch profile is not prepared")
        # sandbox-exec replaces itself with the shell, so the root pid is stable.
        args = [SANDBOX_EXEC, "-f", str(self._profile), "--", *shell_argv(payload)]
        env = {**payload["env"], SANDBOX_MARKER_ENV: SANDBOX_MARKER_VALUE}
        self._child = spawn_command_root(args, cwd=payload["cwd"], env=env)
        return self._child

    def filter_output(self, name: str, chunk: bytes) -> bytes:
        if name != "stderr" or self._nested_reported:
            return chunk
        window = self._stderr_tail + chunk
        if NESTED_SANDBOX_SIGNATURE in window:
            self._nested_reported = True
            return chunk + NESTED_SANDBOX_NOTE
        self._stderr_tail = window[-(len(NESTED_SANDBOX_SIGNATURE) - 1):]
        return chunk

    def request_kill(self, child: subprocess.Popen[bytes]) -> None:
        # Until the root is reaped its pid, and so its new session's process
        # group, cannot belong to anyone else. Afterwards only members are killed.
        if child.returncode is None:
            with suppress(OSError):
                os.killpg(child.pid, signal.SIGKILL)

    def kill_step(self, child: subprocess.Popen[bytes], *, root_reported: bool) -> None:
        domain = self._domain
        if self._proven or domain is None:
            return
        now = self._clock()
        if self._deadline is None:
            self._deadline = now + KILL_DEADLINE_SECONDS
        if now < self._next_step:
            return
        result = domain.kill_step(self._deadline)
        self._next_step = self._clock() + _KILL_INTERVAL_SECONDS
        if isinstance(result, Quiescent):
            self._proven = True
        elif isinstance(result, NotQuiescent):
            pids = ", ".join(str(pid) for pid in result.pids[:8]) or "none"
            raise RuntimeError(
                f"command tree could not be proven quiescent ({result.reason}; pids: {pids})"
            )

    def tree_quiescent(self, child: subprocess.Popen[bytes]) -> bool:
        # The root has exited, so its profile is already applied.
        self._discard_profile()
        # After a kill request, kill_step owns observation of the domain.
        if not self._proven and self._deadline is None and self._domain is not None:
            self._proven = self._domain.poll()
        return self._proven

    def close(self) -> None:
        self._discard_profile()

    def _root_exited(self) -> bool:
        return self._child is not None and self._child.returncode is not None

    def _discard_profile(self) -> None:
        if self._profile is not None:
            with suppress(OSError):
                self._profile.unlink(missing_ok=True)
                self._profile = None


def main() -> int:
    return run_supervisor(SeatbeltBackend)


if __name__ == "__main__":
    raise SystemExit(main())
