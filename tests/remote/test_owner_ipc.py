from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nanobot.remote.client import OwnerRequestError, OwnerUnavailableError, RemoteAccessClient
from nanobot.remote.device_proof import DeviceProofClaims
from nanobot.remote.owner import RemoteAccessOwner
from tests.remote.proof_helpers import new_device_key, sign_claims


def start_owner(tmp_path: Path) -> tuple[RemoteAccessOwner, RemoteAccessClient, str]:
    owner = RemoteAccessOwner(tmp_path / "remote-access", serve_verifier=lambda _origin: True)
    owner.start()
    token = owner.store.admin_token()
    client = RemoteAccessClient(owner.socket_path, timeout=2.0, admin_token=token)
    return owner, client, token


def test_owner_ipc_gateway_and_admin_boundaries(tmp_path) -> None:
    owner, client, token = start_owner(tmp_path)
    try:
        assert stat.S_IMODE(os.stat(owner.socket_path).st_mode) == 0o600
        assert client.ping()["version"] == 1
        status = client.status()
        assert status["enabled"] is False
        assert client.set_enabled(True) is True
        assert client.list_devices() == []
        assert client.add_device(
            "install-0001",
            "phone",
            ["chat.read", "chat.write"],
            "refresh-token-1",
        )["installation_id"] == "install-0001"
        assert client.validate("install-0001", "refresh-token-1")["valid"] is True
        assert client.validate("install-0001", "wrong")["valid"] is False
        assert client.revoke("install-0001")["revoked"] is True
        assert client.validate("install-0001", "refresh-token-1")["valid"] is False

        gateway = RemoteAccessClient(owner.socket_path, timeout=2.0)
        with pytest.raises(OwnerRequestError):
            gateway.list_devices()
        with pytest.raises(OwnerRequestError):
            gateway.set_enabled(False)
        bad_admin = RemoteAccessClient(owner.socket_path, timeout=2.0, admin_token="wrong")
        with pytest.raises(OwnerRequestError):
            bad_admin.list_devices()
        assert token
    finally:
        owner.stop()
    with pytest.raises(OwnerUnavailableError):
        client.ping()


def test_owner_ipc_owner_pairing_pending_qr_and_manual_code(tmp_path) -> None:
    owner, client, _token = start_owner(tmp_path)
    try:
        client.set_enabled(True)
        offer = client.create_pairing(
            origin="https://fixture.example:8443",
            label="Pixel 10",
        )
        assert set(offer["scopes"]) >= {
            "owner",
            "chat.read",
            "chat.write",
            "media.read",
            "attachment.write",
        }
        assert offer["transport"] == "tailscale"
        assert len(offer["pairing_code"]) == 8
        assert offer["qr_link"].startswith("nanobot://pair?")
        assert offer["state"] == "offer_created"

        pending = client.pending_pairings()
        assert pending["pending"][0]["state"] == "offer_created"
        assert pending["pending"][0]["pairing_code"] == offer["pairing_code"]

        import base64

        qr_key, qr_public = new_device_key()
        client.collect_pairing(
            pairing_id=offer["pairing_id"],
            nonce=offer["nonce"],
            collector_secret=offer["collector_secret"],
            installation_id="install-qr-1",
            label="Pixel 10",
            platform="Android 16",
            model="Pixel 10",
            device_public_key=base64.b64encode(qr_public).decode("ascii"),
        )
        status = client.pairing_status(offer["pairing_id"])
        assert status["state"] == "pending_approval"
        assert status["device"]["platform"] == "Android 16"
        assert status["device"]["label"] == "Pixel 10"
        assert status["origin"] == "https://fixture.example:8443"
        assert status["transport"] == "tailscale"
        client.approve_pairing(pairing_id=offer["pairing_id"])
        assert client.pairing_status(offer["pairing_id"])["state"] == "approved"
        qr_claims_payload = owner.pairing.proof_challenge_claims(offer["pairing_id"])
        assert qr_claims_payload is not None
        qr_claims = DeviceProofClaims.from_payload(qr_claims_payload)
        client.prove_pairing(
            pairing_id=offer["pairing_id"],
            claims=qr_claims.payload(),
            signature=base64.b64encode(
                sign_claims(qr_key, qr_claims)
            ).decode("ascii"),
        )

        manual = client.create_pairing(
            origin="https://fixture.example:8443",
            label="Manual phone",
        )
        code_key, code_public = new_device_key()
        collected = client.collect_pairing_code(
            origin="https://fixture.example:8443",
            pairing_code=manual["pairing_code"],
            installation_id="install-code-1",
            label="Manual phone",
            platform="Android 16",
            device_public_key=base64.b64encode(code_public).decode("ascii"),
        )
        assert collected["state"] == "pending_approval"
        with pytest.raises(OwnerRequestError) as missing:
            client.approve_pairing(pairing_id=manual["pairing_id"])
        assert missing.value.code == "pairing_code_invalid"
        approved = client.approve_pairing(
            pairing_id=manual["pairing_id"],
            pairing_code=manual["pairing_code"],
        )
        assert approved["state"] == "approved"
        code_claims_payload = owner.pairing.proof_challenge_claims(manual["pairing_id"])
        assert code_claims_payload is not None
        code_claims = DeviceProofClaims.from_payload(code_claims_payload)
        client.prove_pairing(
            pairing_id=manual["pairing_id"],
            claims=code_claims.payload(),
            signature=base64.b64encode(
                sign_claims(code_key, code_claims)
            ).decode("ascii"),
        )
        assert owner.store.get_device("install-code-1").is_owner  # type: ignore[union-attr]

        denied = client.create_pairing(
            origin="https://fixture.example:8443",
            label="Denied phone",
        )
        deny_key, deny_public = new_device_key()
        client.collect_pairing(
            pairing_id=denied["pairing_id"],
            nonce=denied["nonce"],
            collector_secret=denied["collector_secret"],
            installation_id="install-deny-1",
            label="Denied phone",
            device_public_key=base64.b64encode(deny_public).decode("ascii"),
        )
        assert client.deny_pairing(denied["pairing_id"])["state"] == "denied"
        with pytest.raises(OwnerRequestError) as rejected:
            client.approve_pairing(pairing_id=denied["pairing_id"])
        assert rejected.value.code == "pairing_denied"
    finally:
        owner.stop()


def test_owner_restart_recovers_durable_state(tmp_path) -> None:
    owner, client, token = start_owner(tmp_path)
    client.set_enabled(True)
    client.add_device(
        "install-0001",
        "phone",
        ["chat.read", "chat.write", "owner"],
        "refresh-token-1",
    )
    owner.stop()

    restarted = RemoteAccessOwner(tmp_path / "remote-access", serve_verifier=lambda _origin: True)
    restarted.start()
    recovered = RemoteAccessClient(restarted.socket_path, timeout=2.0, admin_token=token)
    try:
        assert recovered.status()["enabled"] is True
        # The v1 bearer record was durable enough to survive restart, and the
        # startup cutover revoked it before serving refresh authority.
        assert recovered.validate("install-0001", "refresh-token-1")["valid"] is False
        devices = recovered.list_devices()
        assert devices[0]["installation_id"] == "install-0001"
        assert devices[0]["status"] == "revoked"
    finally:
        restarted.stop()


def test_subprocess_owner_is_reachable_and_cleans_up(tmp_path) -> None:
    root = tmp_path / "remote-access"
    socket_path = root / "owner.sock"
    ready = tmp_path / "ready.txt"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nanobot.remote.owner",
            "--state-dir",
            str(root),
            "--socket",
            str(socket_path),
            "--ready-file",
            str(ready),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ready.exists():
            if process.poll() is not None:
                raise AssertionError(process.stderr.read())
            time.sleep(0.05)
        assert ready.exists()
        client = RemoteAccessClient(socket_path, timeout=2.0)
        assert client.ping()["version"] == 1
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert process.returncode == 0
    assert not socket_path.exists()


def test_remote_modules_do_not_import_agent_runtime() -> None:
    root = Path(__file__).resolve().parents[2] / "nanobot" / "remote"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    for forbidden in ("import nanobot.agent", "from nanobot.agent", "gateway_runtime", "AgentLoop"):
        assert forbidden not in source
