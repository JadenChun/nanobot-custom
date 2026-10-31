"""Independent remote-access owner process (M07).

Run with ``python -m nanobot.remote.owner --state-dir <dir>``. The owner owns
durable device state and the private IPC socket; it never imports or starts the
Nanobot agent or gateway.
"""

from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path
from typing import Any

from nanobot.remote.auth import TokenService
from nanobot.remote.ipc import OwnerIPCServer
from nanobot.remote.pairing import PairingService
from nanobot.remote.store import RemoteAccessStore


class RemoteAccessOwner:
    def __init__(self, state_dir: Path | str | None = None, socket_path: Path | str | None = None) -> None:
        self.store = RemoteAccessStore(state_dir)
        self.tokens = TokenService(self.store)
        self.pairing = PairingService(self.store, self.tokens)
        self.server = OwnerIPCServer(
            self.store,
            Path(socket_path) if socket_path else None,
            pairing=self.pairing,
            tokens=self.tokens,
        )

    def start(self) -> None:
        self.store.load_or_create()
        self.server.start()

    def stop(self) -> None:
        self.server.stop()

    def status(self) -> dict[str, Any]:
        status = self.store.snapshot()
        status["ipc_running"] = self.server.running
        return status

    @property
    def socket_path(self) -> Path:
        return self.server.socket_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--socket", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    owner = RemoteAccessOwner(args.state_dir, args.socket)
    owner.start()
    if args.ready_file:
        args.ready_file.write_text(str(owner.socket_path) + "\n", encoding="utf-8")
    print(f"REMOTE_OWNER_READY {owner.socket_path}", flush=True)

    stop_event = threading.Event()

    def _stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        while not stop_event.wait(0.5):
            pass
    finally:
        owner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
