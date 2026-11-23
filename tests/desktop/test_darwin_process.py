import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from kohaku.gateway.darwin_desktop_process import (
    DarwinProcessWatch,
    darwin_desktop_available,
    darwin_gateway_matches,
)

pytestmark = pytest.mark.skipif(
    not darwin_desktop_available(),
    reason="requires macOS audit-token signaling and kqueue",
)


def test_mac_stop_rejects_a_stale_pid_generation_and_awaits_the_captured_exit(tmp_path: Path) -> None:
    command = [
        sys.executable, "-I", "-c",
        "import time; print('ready', flush=True); time.sleep(30)", "arg with spaces",
    ]
    with subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE) as child:
        watch = None
        try:
            assert child.stdout is not None
            assert select.select([child.stdout], [], [], 5)[0]
            assert child.stdout.readline() == b"ready\n"
            watch = DarwinProcessWatch.capture(child.pid)
            assert darwin_gateway_matches(child.pid, command, sys.executable)
            assert not darwin_gateway_matches(child.pid, command[:-1], sys.executable)
            alternate = tmp_path / "alternate-python"
            alternate.symlink_to(sys.executable)
            assert alternate.samefile(sys.executable)
            assert not darwin_gateway_matches(
                child.pid, [str(alternate), *command[1:]], str(alternate),
            )
            generation = watch._token[7]  # audit_token_to_pidversion()
            watch._token[7] ^= 1
            with pytest.raises(ProcessLookupError):
                watch.send_signal(signal.SIGTERM)
            assert not watch.wait_for_exit(0)
            assert child.poll() is None
            watch._token[7] = generation
            watch.send_signal(signal.SIGTERM)
            assert watch.wait_for_exit(5)
            assert child.wait(timeout=5) == -signal.SIGTERM
        finally:
            if watch is not None:
                watch.close()
            if child.poll() is None:
                child.terminate()
