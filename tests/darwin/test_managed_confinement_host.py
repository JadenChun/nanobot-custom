"""What a confined macOS command can actually reach (A3-2M, AC5.4).

Commands run under the real sandbox-exec through the real supervisor backend.
The launch domain is a stand-in, so nothing is signalled and every command
exits by itself. That makes these safe to run on a developer machine, unlike the
darwin_host tests.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

from kohaku.agent.tools import native_process_supervisor_darwin as darwin_supervisor
from kohaku.agent.tools import sandbox
from kohaku.agent.tools.native_process_supervisor import NativeLaunchConfig
from kohaku.security import darwin_proc
from kohaku.security.folder_grants import FolderGrantStore

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="runs the macOS sandbox")

_SECRET = "PRIVATE-STORAGE-MARKER"
# Characters that would change a profile or a -D argument if they were not quoted.
_HOSTILE = 'a b="c\' (d) ;e'


class _UnsignalledDomain:
    """Marks the profile like a real launch domain, and never signals anything."""

    def __init__(self, launch_id: str) -> None:
        self._launch_id = launch_id

    def profile_markers(self) -> str:
        return darwin_proc.marker_rules(self._launch_id)

    def poll(self) -> bool:
        return False

    def kill_step(self, deadline: float) -> None:
        pytest.fail("these tests must not signal processes")


def _domain(
    launch_id: str, *, root_exited: Callable[[], bool], protected: Iterable[int],
) -> _UnsignalledDomain:
    return _UnsignalledDomain(launch_id)


class _Host:
    def __init__(self, root: Path) -> None:
        self.root = root
        instance = root / "instance"
        self.home = instance / "workspace"
        self.project = root / "project"
        self.reference = root / "reference"
        self.outside = root / "outside"
        for folder in (self.home, self.project, self.reference, self.outside):
            folder.mkdir(parents=True)
        self.secret = instance / "secret.txt"
        self.secret.write_text(_SECRET, encoding="utf-8")
        for folder in (self.reference, self.outside):
            (folder / "note.txt").write_text(f"NOTE-{folder.name}", encoding="utf-8")
        self.receipts = self.project / "receipts"
        self.receipts.mkdir(mode=0o700)
        (self.receipts / "proof").write_text(_SECRET, encoding="utf-8")
        self.grants = FolderGrantStore(
            agent_home=self.home, instance_root=instance,
            path=instance / "webui" / "folder-grants.json",
        )
        for revision, (folder, access) in enumerate(
            ((self.project, "read_write"), (self.reference, "read_only")),
        ):
            self.grants.upsert(
                {"path": str(folder), "file_access": access, "commands": True},
                expected_revision=revision,
            )

    def run(self, command: str, **options: Any) -> subprocess.CompletedProcess[str]:
        confinement = sandbox.prepare_managed_seatbelt(
            command, cwd=self.project, agent_home=self.home, grants=self.grants.view,
            shell_program="/bin/bash", **options,
        )
        launch_id = uuid.uuid4().hex
        payload: NativeLaunchConfig = {
            "command": command, "cwd": confinement.cwd,
            "env": {"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            "shell_program": "/bin/bash", "login": False,
            "launch_id": launch_id, "receipt_dir": str(self.receipts),
            "confinement": {
                "rules": confinement.rules, "params": list(confinement.params),
                "argv": list(confinement.argv), "die_with_parent": True,
            },
        }
        backend = darwin_supervisor.SeatbeltBackend(domain_factory=_domain)
        try:
            backend.prepare(payload)
            child = backend.spawn_root(payload)
            stdout, stderr = child.communicate(timeout=60)
        finally:
            backend.close()
            # The command has exited and started nothing that outlives it.
            darwin_supervisor.remove_launch_temp(
                darwin_supervisor.LAUNCH_TMP_ROOT / f"kh-{launch_id}",
            )
        return subprocess.CompletedProcess(
            command, child.returncode,
            stdout.decode(errors="replace"), stderr.decode(errors="replace"),
        )


@pytest.fixture(scope="module")
def _sandbox_usable() -> None:
    probe = subprocess.run(
        [darwin_proc.SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"],
        capture_output=True, timeout=30, check=False,
    )
    if probe.returncode != 0:
        pytest.skip("this test process is already sandboxed and cannot apply another profile")


@pytest.fixture
def host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _sandbox_usable: None,
) -> _Host:
    # The operator switch and host self-test are covered elsewhere.
    monkeypatch.setattr(sandbox, "managed_seatbelt_available", lambda: True)
    return _Host(tmp_path.resolve() / _HOSTILE)


def _q(path: Path | str) -> str:
    return shlex.quote(str(path))


def test_only_granted_folders_can_be_changed(host: _Host) -> None:
    for folder in (host.home, host.project):
        result = host.run(f"echo changed > {_q(folder / 'new.txt')}")
        assert result.returncode == 0, result.stderr
        assert (folder / "new.txt").read_text(encoding="utf-8") == "changed\n"
    for folder in (host.reference, host.outside):
        result = host.run(f"echo changed > {_q(folder / 'new.txt')}")
        assert result.returncode != 0
        assert "Operation not permitted" in result.stderr
        assert not (folder / "new.txt").exists()
    read = host.run(f"cat {_q(host.reference / 'note.txt')}")
    assert read.stdout == "NOTE-reference", read.stderr


def test_ordinary_files_outside_temporary_folders_are_read_only(host: _Host) -> None:
    target = Path.home() / f".kh-test-{uuid.uuid4().hex}"

    read = host.run("head -c 1 /private/etc/hosts >/dev/null")
    try:
        write = host.run(f"echo changed > {_q(target)}")

        assert read.returncode == 0, read.stderr
        assert write.returncode != 0 and not target.exists()
    finally:
        target.unlink(missing_ok=True)


def test_the_command_keeps_its_folder_and_gets_a_private_temporary_one(host: _Host) -> None:
    result = host.run('pwd -P; echo "$TMPDIR"; echo kept > "$TMPDIR/f" && cat "$TMPDIR/f"')

    assert result.returncode == 0, result.stderr
    cwd, temporary, content = result.stdout.splitlines()
    assert cwd == str(darwin_proc.canonical_path(host.project))
    assert temporary.startswith("/private/tmp/kh-") and content == "kept"
    assert not Path(temporary).exists()


def test_shared_temporary_files_are_unreadable(host: _Host) -> None:
    shared = Path(tempfile.mkdtemp(prefix="kh-test-shared.", dir="/private/tmp"))
    try:
        (shared / "f").write_text(_SECRET, encoding="utf-8")

        result = host.run(f"cat {_q(shared / 'f')}")

        assert result.returncode != 0 and _SECRET not in result.stdout
    finally:
        shutil.rmtree(shared)


def test_private_storage_is_unreadable_by_any_spelling(host: _Host) -> None:
    (host.project / "link").symlink_to(host.secret)
    canonical = darwin_proc.canonical_path(host.secret)
    spellings = [
        str(host.secret),
        str(host.project / "link"),
        f"/System/Volumes/Data{canonical}",
        str(host.secret).replace("instance/secret", "INSTANCE/Secret"),
    ]
    for spelling in spellings:
        result = host.run(f"cat {_q(spelling)}")
        assert result.returncode != 0, spelling
        assert _SECRET not in result.stdout, spelling
    linked = host.run(f"ln {_q(host.secret)} hard && cat hard")
    assert linked.returncode != 0 and _SECRET not in linked.stdout


def test_supervisor_files_stay_hidden_inside_a_writable_folder(host: _Host) -> None:
    read = host.run("cat receipts/proof")
    forge = host.run("echo forged > receipts/forged")
    move = host.run("mv receipts moved")

    assert read.returncode != 0 and _SECRET not in read.stdout
    assert forge.returncode != 0 and not (host.receipts / "forged").exists()
    assert move.returncode != 0 and host.receipts.is_dir()


def test_a_granted_folder_cannot_be_renamed_out_of_its_rules(host: _Host) -> None:
    destination = host.home / "moved"

    result = host.run(f"mv {_q(host.reference)} {_q(destination)}")

    assert result.returncode != 0
    assert host.reference.is_dir() and not destination.exists()


@pytest.mark.parametrize("allowed", [True, False])
def test_network_access_follows_the_setting(host: _Host, allowed: bool) -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        result = host.run(f"exec 3<>/dev/tcp/127.0.0.1/{port}", allow_local_binding=allowed)

    assert (result.returncode == 0) is allowed, result.stderr


# Reads another process's arguments and environment with sysctl KERN_PROCARGS2.
_PROCARGS = """
import ctypes, json, sys
libc = ctypes.CDLL(None, use_errno=True)
mib = (ctypes.c_int * 3)(1, 49, int(sys.argv[1]))
buffer = ctypes.create_string_buffer(1 << 20)
size = ctypes.c_size_t(len(buffer))
code = libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0)
data = buffer.raw[: size.value] if code == 0 else b""
print(json.dumps({"code": code, "secret": sys.argv[2].encode() in data}))
"""


def test_another_process_environment_is_private(host: _Host) -> None:
    script = host.project / "procargs.py"
    script.write_text(_PROCARGS, encoding="utf-8")
    # Waits on its input, so closing the pipe ends it.
    waiting = [sys.executable, "-c", "import sys; sys.stdin.read()"]
    with subprocess.Popen(
        waiting, stdin=subprocess.PIPE, env={**os.environ, "KOHAKU_TEST_VALUE": _SECRET},
    ) as target:
        reader = [sys.executable, str(script), str(target.pid), _SECRET]
        control = subprocess.run(reader, capture_output=True, text=True, timeout=60, check=True)
        confined = host.run(shlex.join(reader))
        own = host.run(
            f"KOHAKU_TEST_VALUE={_SECRET} /bin/sleep 3 & "
            f"{shlex.join(reader[:2])} $! {_SECRET}; wait"
        )
        assert target.stdin is not None
        target.stdin.close()

    # Without the control, a broken reader would look like a working sandbox.
    assert json.loads(control.stdout) == {"code": 0, "secret": True}
    assert confined.returncode == 0, confined.stderr
    assert json.loads(confined.stdout) == {"code": -1, "secret": False}
    assert json.loads(own.stdout)["code"] == 0, own.stderr
