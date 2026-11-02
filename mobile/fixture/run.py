"""Start the M02 fixture on loopback and report its bound address."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from app import create_app

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class FixtureServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, ready_file: Path | None, scheme: str) -> None:
        super().__init__(config)
        self.ready_file = ready_file
        self.scheme = scheme

    async def startup(self, sockets=None) -> None:  # type: ignore[override]
        await super().startup(sockets=sockets)
        if self.should_exit or not self.servers:
            return
        sockets = self.servers[0].sockets
        address = sockets[0].getsockname()
        host = "127.0.0.1" if address[0] in {"0.0.0.0", "::"} else address[0]
        port = address[1]
        url = f"{self.scheme}://{host}:{port}"
        print(f"FIXTURE_READY {url}", flush=True)
        if self.ready_file:
            self.ready_file.write_text(
                json.dumps({"url": url, "pid": __import__("os").getpid()}) + "\n",
                encoding="utf-8",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--ready-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in LOOPBACK_HOSTS:
        print("fixture refuses non-loopback bind addresses", file=sys.stderr)
        return 2
    if bool(args.tls_cert) != bool(args.tls_key):
        print("--tls-cert and --tls-key must be supplied together", file=sys.stderr)
        return 2
    app = create_app()
    scheme = "https" if args.tls_cert else "http"
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
        ssl_certfile=str(args.tls_cert) if args.tls_cert else None,
        ssl_keyfile=str(args.tls_key) if args.tls_key else None,
    )
    FixtureServer(config, args.ready_file, scheme).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
