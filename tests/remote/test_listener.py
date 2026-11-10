from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from nanobot.remote.auth import TokenService
from nanobot.remote.client import OwnerRequestError, RemoteAccessClient
from nanobot.remote.device_proof import DeviceProofClaims
from nanobot.remote.http import RemoteHTTPListener
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.pairing import PairingService
from nanobot.remote.staging import TailscaleServeAdapter
from nanobot.remote.store import RemoteAccessStore
from tests.remote.proof_helpers import (
    complete_proof,
    new_device_key,
    sign_claims,
)


def pairing_service(tmp_path: Path) -> tuple[RemoteAccessStore, PairingService]:
    store = RemoteAccessStore(tmp_path / "remote-access")
    store.load_or_create()
    store.set_enabled(True)
    tokens = TokenService(store)
    return store, PairingService(store, tokens)


def test_listener_manual_pairing_code_without_nonce(tmp_path) -> None:
    store, pairing = pairing_service(tmp_path)
    listener = RemoteHTTPListener(pairing, port=0)
    port = listener.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=3.0) as client:
            created = pairing.create_challenge(
                origin="https://fixture.example:8443",
                label="phone",
                chat_consent=True,
                media_consent=True,
                full_control_consent=True,
                owner_only=True,
            )
            wrong_code = f"{(int(created.pairing_code) + 1) % 100_000_000:08d}"
            wrong = client.post(
                "/remote/v1/pair/code",
                json={
                    "origin": created.origin,
                    "pairing_code": wrong_code,
                    "installation_id": "install-code-1",
                    "label": "Pixel 10",
                },
            )
            assert wrong.status_code == 403
            assert wrong.json()["error"] == "pairing_code_invalid"

            _key, public_der = new_device_key()
            import base64

            collected = client.post(
                "/remote/v1/pair/code",
                json={
                    "origin": created.origin,
                    "pairing_code": created.pairing_code,
                    "installation_id": "install-code-1",
                    "label": "Pixel 10",
                    "platform": "Android 16",
                    "model": "Pixel 10",
                    "device_public_key": base64.b64encode(public_der).decode("ascii"),
                },
            )
            assert collected.status_code == 200
            assert collected.json()["state"] == "pending_approval"
            assert collected.json()["platform"] == "Android 16"

            pairing.approve(
                pairing_id=created.pairing_id,
                pairing_code=created.pairing_code,
            )
            complete_proof(pairing, created.pairing_id, _key)
            approved = client.post(
                "/remote/v1/pair/poll",
                json={
                    "pairing_id": created.pairing_id,
                    "nonce": created.nonce,
                    "collector_secret": created.collector_secret,
                },
            )
            assert approved.status_code == 200
            assert approved.json()["status"] == "approved"
            assert approved.json()["refresh_expires_in"] == 0
    finally:
        listener.stop()


def test_listener_pairing_flow_and_bounds(tmp_path) -> None:
    store, pairing = pairing_service(tmp_path)
    listener = RemoteHTTPListener(pairing, port=0)
    port = listener.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=3.0) as client:
            assert client.get("/remote/v1/health").json()["version"] == 1
            created = pairing.create_challenge(
                origin="https://fixture.example:8443",
                label="phone",
                chat_consent=True,
                media_consent=False,
            )
            _key, public_der = new_device_key()
            import base64

            collect = client.post(
                "/remote/v1/pair/collect",
                json={
                    "pairing_id": created.pairing_id,
                    "nonce": created.nonce,
                    "collector_secret": created.collector_secret,
                    "installation_id": "install-0001",
                    "label": "phone",
                    "device_public_key": base64.b64encode(public_der).decode("ascii"),
                },
            )
            assert collect.status_code == 200
            assert collect.json()["comparison_code"] == created.comparison_code
            pending = client.post(
                "/remote/v1/pair/poll",
                json={
                    "pairing_id": created.pairing_id,
                    "nonce": created.nonce,
                    "collector_secret": created.collector_secret,
                },
            )
            assert pending.status_code == 200
            assert pending.json()["installation_id"] == "install-0001"
            pairing.approve(pairing_id=created.pairing_id, comparison_code=created.comparison_code)
            complete_proof(pairing, created.pairing_id, _key)
            approved = client.post(
                "/remote/v1/pair/poll",
                json={
                    "pairing_id": created.pairing_id,
                    "nonce": created.nonce,
                    "collector_secret": created.collector_secret,
                },
            )
            assert approved.status_code == 200
            assert approved.json()["status"] == "approved"
            assert approved.json()["access_token"].startswith("nbm_at_")
            consumed = client.post(
                "/remote/v1/pair/poll",
                json={
                    "pairing_id": created.pairing_id,
                    "nonce": created.nonce,
                    "collector_secret": created.collector_secret,
                },
            )
            assert consumed.status_code == 410
            assert client.get("/mobile/v1/read").status_code == 404
            assert client.post("/remote/v1/pair/collect", content=b"{}").status_code == 400
            assert client.post("/remote/v1/pair/collect", content=b"x" * 70000).status_code == 400
    finally:
        listener.stop()
    with pytest.raises(ValueError):
        RemoteHTTPListener(pairing, host="0.0.0.0")


def test_owner_pairing_admin_ipc_is_local_only(tmp_path) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access", serve_verifier=lambda _origin: True)
    owner.start()
    token = owner.store.admin_token()
    client = RemoteAccessClient(owner.socket_path, admin_token=token)
    listener = RemoteHTTPListener(owner.pairing, port=0)
    port = listener.start()
    try:
        client.set_enabled(True)
        offer = client.create_pairing(
            origin="https://fixture.example:8443",
            label="phone",
            chat_consent=True,
        )
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3.0) as http:
            import base64

            key, public_der = new_device_key()
            collected = http.post(
                "/remote/v1/pair/collect",
                json={
                    "pairing_id": offer["pairing_id"],
                    "nonce": offer["nonce"],
                    "collector_secret": offer["collector_secret"],
                    "installation_id": "install-0001",
                    "label": "phone",
                    "device_public_key": base64.b64encode(public_der).decode("ascii"),
                },
            )
            assert collected.status_code == 200
            approved = client.approve_pairing(
                pairing_id=offer["pairing_id"],
                comparison_code=offer["comparison_code"],
            )
            assert approved["installation_id"] == "install-0001"
            payload = owner.pairing.proof_challenge_claims(offer["pairing_id"])
            assert payload is not None
            claims = DeviceProofClaims.from_payload(payload)
            client.prove_pairing(
                pairing_id=offer["pairing_id"],
                claims=claims.payload(),
                signature=base64.b64encode(sign_claims(key, claims)).decode("ascii"),
            )
            result = http.post(
                "/remote/v1/pair/poll",
                json={
                    "pairing_id": offer["pairing_id"],
                    "nonce": offer["nonce"],
                    "collector_secret": offer["collector_secret"],
                },
            )
            assert result.json()["status"] == "approved"
        gateway = RemoteAccessClient(owner.socket_path)
        with pytest.raises(OwnerRequestError):
            gateway.create_pairing(
                origin="https://fixture.example:8443",
                label="phone",
                chat_consent=True,
            )
    finally:
        listener.stop()
        owner.stop()


class FakeTailscale(TailscaleServeAdapter):
    def __init__(self, status: dict, serve: dict | None = None, funnel: dict | None = None) -> None:
        super().__init__()
        self._status = status
        self._serve = serve or {}
        self._funnel = funnel or {}

    def status_json(self) -> dict:
        return self._status

    def serve_status_json(self) -> dict:
        return self._serve

    def funnel_status_json(self) -> dict:
        return self._funnel


def test_staging_plan_is_read_only_and_strict() -> None:
    status = {
        "BackendState": "Running",
        "Self": {"DNSName": "host.tail.ts.net."},
    }
    adapter = FakeTailscale(status)
    plan = adapter.plan(origin="https://host.tail.ts.net", listener_port=8124)
    assert plan.ready is True
    assert plan.desired_command[-2:] == ["--set-path=/remote/v1", "http://127.0.0.1:8124"]
    assert "Peers" not in json.dumps(plan.prior)
    with pytest.raises(RuntimeError):
        adapter.apply(plan, allow_live=False)

    assert not adapter.plan(origin="https://other.tail.ts.net", listener_port=8124).ready
    assert not adapter.plan(origin="https://host.tail.ts.net", listener_port=80).ready
    assert not FakeTailscale(status, funnel={"allow": True}).plan(
        origin="https://host.tail.ts.net", listener_port=8124
    ).ready
    conflict = FakeTailscale(status, serve={"Web": {"host.tail.ts.net:443": {"Handlers": {"/remote/v1": {}}}}})
    plan = conflict.plan(origin="https://host.tail.ts.net", listener_port=8124)
    assert plan.ready is False
    assert plan.conflicts
    assert "Peers" not in json.dumps(plan.prior)


def test_owner_process_serves_status_when_listener_port_is_configured(tmp_path) -> None:
    owner = RemoteAccessOwner(tmp_path / "owner", listener_port=0)
    owner.start()
    owner.store.set_enabled(True)
    pair = owner.tokens.issue("install-owner", ("chat.read", "chat.write", "owner"), "owner")
    assert owner.listener is not None
    port = owner.listener.bound_port
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3.0) as http:
            response = http.get(
                "/remote/v1/status",
                headers={"Authorization": f"Bearer {pair.access_token}"},
            )
            assert response.status_code == 200
            assert response.json()["installation_id"] == "install-owner"
    finally:
        owner.stop()
    assert owner.listener is None


def test_owner_control_routes_survive_without_gateway(tmp_path) -> None:
    owner = RemoteAccessOwner(tmp_path / "owner")
    owner.start()
    owner.store.set_enabled(True)
    owner_pair = owner.tokens.issue(
        "install-owner", ("chat.read", "chat.write", "owner"), "owner-phone"
    )
    chat_pair = owner.tokens.issue("install-chat", ("chat.read",), "chat-phone")
    victim_pair = owner.tokens.issue("install-victim", ("chat.read",), "victim-phone")
    listener = RemoteHTTPListener(owner.pairing, owner=owner, port=0)
    port = listener.start()
    owner_headers = {"Authorization": f"Bearer {owner_pair.access_token}"}
    chat_headers = {"Authorization": f"Bearer {chat_pair.access_token}"}
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3.0) as http:
            status = http.get("/remote/v1/status", headers=owner_headers)
            assert status.status_code == 200
            assert status.json()["installation_id"] == "install-owner"
            assert "owner" in status.json()["scopes"]
            assert http.get("/remote/v1/status").status_code == 401

            devices = http.get("/remote/v1/devices", headers=owner_headers)
            assert devices.status_code == 200
            assert {
                device["installation_id"] for device in devices.json()["devices"]
            } == {"install-owner", "install-chat", "install-victim"}
            assert http.get(
                "/remote/v1/devices", headers=chat_headers
            ).status_code == 403

            refreshed = http.post(
                "/remote/v1/refresh",
                json={
                    "installation_id": "install-victim",
                    "refresh_token": victim_pair.refresh_token,
                },
            )
            assert refreshed.status_code == 200
            victim_access = refreshed.json()["access_token"]
            assert http.get(
                "/remote/v1/status",
                headers={"Authorization": f"Bearer {victim_access}"},
            ).status_code == 200

            revoked = http.post(
                "/remote/v1/devices/install-victim/revoke",
                headers=owner_headers,
            )
            assert revoked.status_code == 200
            assert revoked.json()["revoked"] is True
            assert http.get(
                "/remote/v1/status",
                headers={"Authorization": f"Bearer {victim_access}"},
            ).status_code == 401
    finally:
        listener.stop()
        owner.stop()
