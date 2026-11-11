from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from nanobot.remote.data_server import MobileDataServer
from nanobot.remote.device_proof import DeviceProofClaims, device_proof_fingerprint
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.protocol import stable_operation_digest
from tests.remote.proof_helpers import encode_proof_base64url, new_device_key, sign_claims


@pytest.mark.asyncio
async def test_data_server_recovery_replay_and_revoke(tmp_path: Path) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    data = MobileDataServer(owner, port=0)
    port = data.start()
    origin = f"https://127.0.0.1:{port}"
    key, public_der = new_device_key()
    pair = owner.tokens.issue_device(
        "install-data-1",
        ("chat.read", "chat.write", "device.self.logout"),
        device_fingerprint=device_proof_fingerprint(public_der),
        device_generation=1,
        origin=origin,
    )
    base = f"http://127.0.0.1:{port}"
    target_headers = {"Host": f"127.0.0.1:{port}", "Origin": origin}
    headers = {**target_headers, "Authorization": f"Bearer {pair.access_token}"}
    assert pair.refresh_token == ""
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

            recovery_headers = {
                **target_headers,
                "X-Nanobot-Installation": "install-data-1",
                "X-Nanobot-Device-Key": encode_proof_base64url(public_der),
            }
            recovery = client.get(
                "/mobile/v1/pair/recover",
                headers=recovery_headers,
            )
            assert recovery.status_code == 200
            claims = DeviceProofClaims.from_payload(recovery.json()["proof_challenge"])
            complete_headers = {
                **recovery_headers,
                "X-Nanobot-Device-Signature": encode_proof_base64url(
                    sign_claims(key, claims)
                ),
                "X-Nanobot-Proof-Claims": encode_proof_base64url(
                    json.dumps(
                        claims.payload(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
            }
            recovered = client.get(
                "/mobile/v1/pair/recover/complete",
                headers=complete_headers,
            )
            assert recovered.status_code == 200
            recovered_payload = recovered.json()
            assert recovered_payload["installation_id"] == "install-data-1"
            assert not recovered_payload.get("refresh_token")
            new_headers = {
                **target_headers,
                "Authorization": f"Bearer {recovered_payload['access_token']}",
            }
            repeated = client.get(
                "/mobile/v1/pair/recover/complete",
                headers=complete_headers,
            )
            assert repeated.status_code == 410
            assert repeated.json()["error"] == "pairing_consumed"
            assert client.get("/mobile/v1/handshake", headers=headers).status_code == 401
            assert client.get("/mobile/v1/handshake", headers=new_headers).status_code == 200

            logout = client.post("/mobile/v1/logout", headers=new_headers, json={})
            assert logout.status_code == 200
            assert client.get("/mobile/v1/handshake", headers=new_headers).status_code == 401
            assert owner.store.get_device("install-data-1").revoked
            refused_recovery = client.get(
                "/mobile/v1/pair/recover",
                headers=recovery_headers,
            )
            assert refused_recovery.status_code == 403
            assert refused_recovery.json()["error"] == "pairing_rejected"
    finally:
        data.stop()
        owner.stop()
