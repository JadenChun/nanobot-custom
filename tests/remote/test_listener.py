from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from nanobot.remote.auth import TokenService
from nanobot.remote.client import OwnerRequestError, RemoteAccessClient
from nanobot.remote.http import RemoteHTTPListener
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.pairing import PairingService
from nanobot.remote.staging import TailscaleServeAdapter
from nanobot.remote.store import RemoteAccessStore


def pairing_service(tmp_path: Path) -> tuple[RemoteAccessStore, PairingService]:
    store = RemoteAccessStore(tmp_path / "remote-access")
    store.load_or_create()
    store.set_enabled(True)
    tokens = TokenService(store)
    return store, PairingService(store, tokens)


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
            collect = client.post(
                "/remote/v1/pair/collect",
                json={
                    "pairing_id": created.pairing_id,
                    "nonce": created.nonce,
                    "collector_secret": created.collector_secret,
                    "installation_id": "install-0001",
                    "label": "phone",
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
    owner = RemoteAccessOwner(tmp_path / "remote-access")
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
            collected = http.post(
                "/remote/v1/pair/collect",
                json={
                    "pairing_id": offer["pairing_id"],
                    "nonce": offer["nonce"],
                    "collector_secret": offer["collector_secret"],
                    "installation_id": "install-0001",
                    "label": "phone",
                },
            )
            assert collected.status_code == 200
            approved = client.approve_pairing(
                pairing_id=offer["pairing_id"],
                comparison_code=offer["comparison_code"],
            )
            assert approved["installation_id"] == "install-0001"
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
