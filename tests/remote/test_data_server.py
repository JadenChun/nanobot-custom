from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from nanobot.remote.data_server import MobileDataServer
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.protocol import stable_operation_digest


@pytest.mark.asyncio
async def test_data_server_refresh_replay_and_revoke(tmp_path: Path) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    pair = owner.tokens.issue("install-data-1", ("chat.read", "chat.write", "device.self.logout"))
    data = MobileDataServer(owner, port=0)
    port = data.start()
    base = f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {pair.access_token}"}
    try:
        with httpx.Client(base_url=base, timeout=5.0) as client:
            handshake = client.get("/mobile/v1/handshake", headers=headers)
            assert handshake.status_code == 200
            assert handshake.json()["socket_epoch"] == 1
            assert client.get("/mobile/v1/read", headers=headers).json() == {"sessions": []}

            digest = stable_operation_digest("device-data", "chat.send", {"text": "hello"})
            envelope = {
                "request_id": "data-req-1",
                "installation_id": "install-data-1",
                "principal": "device-data",
                "action": "chat.send",
                "digest": digest,
                "gateway_generation": 1,
                "epoch": 1,
                "mode": "submit",
                "payload": {"text": "hello"},
            }
            first = client.post("/mobile/v1/mutation", headers=headers, json={"envelope": envelope})
            assert first.status_code == 200
            assert first.json()["decision"] == "applied"
            replay = client.post(
                "/mobile/v1/mutation",
                headers=headers,
                json={"envelope": {**envelope, "mode": "replay_only"}},
            )
            assert replay.json()["decision"] == "replayed"
            assert replay.json()["effect_count"] == 1
            assert data.effects == 1

            refreshed = client.post(
                "/mobile/v1/refresh",
                headers=headers,
                json={"refresh_token": pair.refresh_token},
            )
            assert refreshed.status_code == 200
            new_access = refreshed.json()["access_token"]
            assert client.get("/mobile/v1/handshake", headers=headers).status_code == 401
            new_headers = {"Authorization": f"Bearer {new_access}"}
            assert client.get("/mobile/v1/handshake", headers=new_headers).status_code == 200

            logout = client.post("/mobile/v1/logout", headers=new_headers, json={})
            assert logout.status_code == 200
            assert client.get("/mobile/v1/handshake", headers=new_headers).status_code == 401
            assert owner.store.get_device("install-data-1").revoked
    finally:
        data.stop()
        owner.stop()
