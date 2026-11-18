import signal
import subprocess
import sys

import pytest

from nanobot.gateway.darwin_desktop_process import (
    DarwinProcessWatch,
    darwin_desktop_available,
    darwin_gateway_matches,
)

pytestmark = pytest.mark.skipif(
    not darwin_desktop_available(),
    reason="requires macOS audit-token signaling and kqueue",
)


def test_mac_stop_rejects_a_stale_pid_generation_and_awaits_the_captured_exit() -> None:
    command = [sys.executable, "-I", "-c", "import time; time.sleep(30)", "arg with spaces"]
    with subprocess.Popen(command, start_new_session=True) as child:
        watch = DarwinProcessWatch.capture(child.pid)
        try:
            assert darwin_gateway_matches(child.pid, command, sys.executable)
            assert not darwin_gateway_matches(child.pid, command[:-1], sys.executable)
            generation = watch._token[7]  # audit_token_to_pidversion()
            watch._token[7] ^= 1
            with pytest.raises(ProcessLookupError):
                watch.terminate()
            assert not watch.wait_for_exit(0)
            assert child.poll() is None
            watch._token[7] = generation
            watch.terminate()
            assert watch.wait_for_exit(5)
            assert child.wait(timeout=5) == -signal.SIGTERM
        finally:
            watch.close()
            if child.poll() is None:
                child.terminate()
