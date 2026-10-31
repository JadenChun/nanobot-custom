"""M02 fixture contract tests. These do not start or import a Nanobot agent."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

FIXTURE_DIR = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FixtureProcess:
    def __init__(self, *, tls: bool = False) -> None:
        self.port = free_port()
        self.tls = tls
        self.proc: subprocess.Popen[str] | None = None
        self.base_url = ""
        self.cert: Path | None = None
        self.key: Path | None = None

    def __enter__(self) -> "FixtureProcess":
        command = [sys.executable, "run.py", "--port", str(self.port)]
        if self.tls:
            import tempfile

            directory = Path(tempfile.mkdtemp(prefix="nanobot-fixture-tls-"))
            self.cert = directory / "cert.pem"
            self.key = directory / "key.pem"
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(self.key),
                    "-out",
                    str(self.cert),
                    "-days",
                    "1",
                    "-subj",
                    "/CN=fixture.local",
                    "-addext",
                    "subjectAltName=DNS:fixture.local,IP:127.0.0.1",
                ],
                check=True,
                capture_output=True,
            )
            command += ["--tls-cert", str(self.cert), "--tls-key", str(self.key)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            command,
            cwd=FIXTURE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert self.proc.stdout is not None
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("FIXTURE_READY "):
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise AssertionError(f"fixture did not become ready: {ready!r} {stderr}")
        self.base_url = ready.split(" ", 1)[1]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                with httpx.Client(verify=not self.tls, timeout=1.0) as client:
                    if client.get(f"{self.base_url}/health").status_code == 200:
                        return self
            except httpx.HTTPError:
                time.sleep(0.05)
        raise AssertionError("fixture health endpoint did not become ready")

    def __exit__(self, *_: object) -> None:
        assert self.proc is not None
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def client(self, **kwargs: object) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, verify=not self.tls, timeout=5.0, **kwargs)


def admin_headers() -> dict[str, str]:
    return {"X-Fixture-Admin": "fixture-admin"}


def bootstrap(client: httpx.Client) -> dict[str, object]:
    response = client.post(
        "/fixture/v1/bootstrap",
        headers={"X-Fixture-Bootstrap": "fixture-bootstrap"},
        json={"installation_id": "fixture-installation"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_and_loopback_lifecycle() -> None:
    with FixtureProcess() as fixture:
        assert fixture.base_url.startswith("http://127.0.0.1:")
        with fixture.client() as client:
            body = client.get("/health").json()
            assert body["fixture"] is True
            assert body["version"] == "fixture-v1"


def test_rejects_non_loopback_bind() -> None:
    result = subprocess.run(
        [sys.executable, "run.py", "--host", "0.0.0.0", "--port", str(free_port())],
        cwd=FIXTURE_DIR,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "refuses non-loopback" in result.stderr


def test_bootstrap_read_refresh_and_revocation() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        first = bootstrap(client)
        access = str(first["access_token"])
        assert client.get("/fixture/v1/read", headers=bearer(access)).status_code == 200
        assert client.get("/fixture/v1/counter").json()["effects"] == 0

        rotated = client.post("/fixture/v1/refresh", json={"refresh_token": first["refresh_token"]})
        assert rotated.status_code == 200
        assert rotated.json()["refresh_token"] != first["refresh_token"]
        assert client.get("/fixture/v1/read", headers=bearer(access)).status_code == 401
        assert client.get("/fixture/v1/read", headers=bearer(str(rotated.json()["access_token"]))).status_code == 200
        assert client.post("/fixture/v1/refresh", json={"refresh_token": first["refresh_token"]}).status_code == 401

        assert client.post("/fixture/v1/admin/mode", headers=admin_headers(), json={"mode": "revoke"}).status_code == 200
        assert client.get("/fixture/v1/read", headers=bearer(str(rotated.json()["access_token"]))).status_code == 403


def test_expired_access_and_refresh_are_distinct() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        pair = bootstrap(client)
        client.post("/fixture/v1/admin/expire", params={"what": "access"}, headers=admin_headers())
        response = client.get("/fixture/v1/read", headers=bearer(str(pair["access_token"])))
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "expired_access"
        client.post("/fixture/v1/admin/expire", params={"what": "refresh"}, headers=admin_headers())
        response = client.post("/fixture/v1/refresh", json={"refresh_token": pair["refresh_token"]})
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "expired_refresh"


def test_mutation_replay_and_lost_response_effect_once() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        pair = bootstrap(client)
        headers = bearer(str(pair["access_token"]))
        payload = {"request_id": "req-1", "value": "alpha"}
        first = client.post("/fixture/v1/mutation", headers=headers, json=payload)
        assert first.status_code == 200
        assert first.json()["effect"] == 1
        replay = client.post("/fixture/v1/mutation", headers=headers, json=payload)
        assert replay.status_code == 200
        assert replay.json()["effect"] == 1
        assert replay.json()["replayed"] is True
        assert client.get("/fixture/v1/counter").json()["effects"] == 1

        client.post(
            "/fixture/v1/admin/mode",
            headers=admin_headers(),
            json={"mode": "mutation_lost_inflight"},
        )
        lost = client.post("/fixture/v1/mutation", headers=headers, json={"request_id": "req-2"})
        assert lost.status_code == 503
        assert client.get("/fixture/v1/counter").json()["effects"] == 2
        joined = client.post("/fixture/v1/mutation", headers=headers, json={"request_id": "req-2"})
        assert joined.status_code == 200
        assert joined.json()["effect"] == 2
        assert joined.json()["replayed"] is True
        assert client.get("/fixture/v1/counter").json()["effects"] == 2


def test_mutation_connection_drop_and_delay() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        pair = bootstrap(client)
        headers = bearer(str(pair["access_token"]))
        client.post("/fixture/v1/admin/mode", headers=admin_headers(), json={"mode": "mutation_lost_response"})
        with pytest.raises((httpx.RemoteProtocolError, httpx.ReadError)):
            client.post("/fixture/v1/mutation", headers=headers, json={"request_id": "drop-1"})
        assert client.get("/fixture/v1/counter").json()["effects"] == 1
        replay = client.post("/fixture/v1/mutation", headers=headers, json={"request_id": "drop-1"})
        assert replay.status_code == 200
        assert replay.json()["replayed"] is True
        assert client.get("/fixture/v1/counter").json()["effects"] == 1

        client.post("/fixture/v1/admin/mode", headers=admin_headers(), json={"mode": "mutation_delayed", "value": 0.2})
        started = time.monotonic()
        delayed = client.post("/fixture/v1/mutation", headers=headers, json={"request_id": "slow-1"})
        assert delayed.status_code == 200
        assert time.monotonic() - started >= 0.15
        assert delayed.json()["effect"] == 2


def test_bootstrap_wrong_installation_is_rejected() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        response = client.post(
            "/fixture/v1/bootstrap",
            headers={"X-Fixture-Bootstrap": "fixture-bootstrap"},
            json={"installation_id": "another-installation"},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "instance_mismatch"


def test_refresh_lost_response_does_not_resurrect_old_token() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        pair = bootstrap(client)
        client.post(
            "/fixture/v1/admin/mode",
            headers=admin_headers(),
            json={"mode": "refresh_lost_response"},
        )
        response = client.post("/fixture/v1/refresh", json={"refresh_token": pair["refresh_token"]})
        assert response.status_code == 503
        retry = client.post("/fixture/v1/refresh", json={"refresh_token": pair["refresh_token"]})
        assert retry.status_code == 401


def test_media_ranges_and_hostile_samples() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        pair = bootstrap(client)
        headers = bearer(str(pair["access_token"]))
        full = client.get("/fixture/v1/media/sample.png", headers=headers)
        assert full.status_code == 200
        assert full.headers["content-type"] == "image/png"
        assert full.headers["accept-ranges"] == "bytes"
        assert full.content.startswith(bytes.fromhex("89504E47"))

        part = client.get("/fixture/v1/media/sample.png", headers={**headers, "Range": "bytes=0-3"})
        assert part.status_code == 206
        assert part.content == full.content[:4]
        assert part.headers["content-range"] == f"bytes 0-3/{len(full.content)}"

        suffix = client.get("/fixture/v1/media/sample.png", headers={**headers, "Range": "bytes=-3"})
        assert suffix.status_code == 206
        assert suffix.content == full.content[-3:]

        bad = client.get("/fixture/v1/media/sample.png", headers={**headers, "Range": "bytes=999-"})
        assert bad.status_code == 416
        assert bad.headers["content-range"] == f"bytes */{len(full.content)}"

        for name in ("sample.mp3", "sample.mp4", "untrusted.svg", "untrusted.html"):
            assert client.get(f"/fixture/v1/media/{name}", headers=headers).status_code == 200
        assert b"__fixtureXss" in client.get("/fixture/v1/media/untrusted.html", headers=headers).content


def test_attachment_limits_and_checksum() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        pair = bootstrap(client)
        headers = bearer(str(pair["access_token"]))
        payload = b"fixture-attachment"
        response = client.post(
            "/fixture/v1/attachment",
            headers=headers,
            json={
                "name": "note.txt",
                "media_type": "text/plain",
                "content_base64": base64.b64encode(payload).decode(),
            },
        )
        assert response.status_code == 200
        assert response.json()["sha256"] == hashlib.sha256(payload).hexdigest()
        oversized = b"x" * (256 * 1024 + 1)
        response = client.post(
            "/fixture/v1/attachment",
            headers=headers,
            json={
                "name": "big.bin",
                "content_base64": base64.b64encode(oversized).decode(),
            },
        )
        assert response.status_code == 413


def test_redirect_is_explicit_and_cross_origin() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        response = client.get(
            "/fixture/v1/redirect",
            params={"to": "https://evil.example/fixture"},
            follow_redirects=False,
        )
        assert response.status_code == 302
        assert response.headers["location"] == "https://evil.example/fixture"


def test_websocket_echo_stream_and_disconnect() -> None:
    with FixtureProcess() as fixture, fixture.client() as client:
        pair = bootstrap(client)
        assert pair["capabilities"]["chat"] is True
        ws_url = fixture.base_url.replace("http://", "ws://") + "/fixture/v1/ws"
        with ws_connect(ws_url) as websocket:
            hello = json.loads(websocket.recv())
            assert hello["type"] == "hello"
            websocket.send(json.dumps({"type": "echo", "value": "ping"}))
            assert json.loads(websocket.recv()) == {"type": "echo", "value": "ping"}
            websocket.send(json.dumps({"type": "stream", "count": 4}))
            frames = [json.loads(websocket.recv()) for _ in range(4)]
            assert [frame["index"] for frame in frames] == [0, 1, 2, 3]
            websocket.send("not-json")
            assert json.loads(websocket.recv())["code"] == "malformed_json"
            websocket.send(json.dumps({"type": "oversize"}))
            assert json.loads(websocket.recv())["type"] == "oversize"
            websocket.send(json.dumps({"type": "binary"}))
            assert isinstance(websocket.recv(), bytes)

        client.post("/fixture/v1/admin/mode", headers=admin_headers(), json={"mode": "ws_disconnect"})
        with ws_connect(ws_url) as websocket:
            assert json.loads(websocket.recv())["type"] == "hello"
            with pytest.raises(ConnectionClosed):
                websocket.recv()


def test_tls_self_signed_is_rejected_by_default() -> None:
    with FixtureProcess(tls=True) as fixture:
        with pytest.raises(httpx.ConnectError):
            with httpx.Client(verify=True, timeout=3.0) as client:
                client.get(f"{fixture.base_url}/health")
        with httpx.Client(verify=False, timeout=3.0) as client:
            assert client.get(f"{fixture.base_url}/health").status_code == 200


def test_fixture_has_no_agent_imports() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in FIXTURE_DIR.glob("*.py"))
    for forbidden in ("from nanobot", "import nanobot", "AgentLoop", "gateway_runtime"):
        assert forbidden not in source
