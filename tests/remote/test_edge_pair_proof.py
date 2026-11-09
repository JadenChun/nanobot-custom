"""Regression tests for the A3-7 edge device-proof pairing route."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.config.schema import RemoteAccessConfig
from nanobot.remote.device_proof import (
    DeviceProofClaims,
    canonical_device_proof_bytes,
)
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.webui.gateway_services import build_gateway_services

ORIGIN = "https://fixture.example:8443"
INSTALLATION_ID = "install-0001"


def keypair():
    key = ec.generate_private_key(ec.SECP256R1())
    public_der = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return key, public_der


@pytest.mark.asyncio
async def test_edge_pair_prove_issues_access_from_device_proof(
    tmp_path: Path,
) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    services = build_gateway_services(
        config=WebSocketConfig(host="127.0.0.1", port=29998, path="/"),
        bus=MessageBus(),
        session_manager=None,
        static_dist_path=None,
        workspace_path=tmp_path,
        default_restrict_to_workspace=False,
        runtime_model_name=None,
        runtime_surface="mobile",
        runtime_capabilities_overrides=None,
        remote_access=RemoteAccessConfig(
            enabled=True,
            state_dir=str(owner.store.root),
            socket_path=str(owner.socket_path),
        ),
    )
    edge = services.endpoint._mobile_edge
    assert edge is not None
    try:
        key, public_der = keypair()
        created = owner.pairing.create_challenge(
            origin=ORIGIN,
            label="phone",
            chat_consent=True,
            media_consent=False,
        )
        owner.pairing.collect(
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
            installation_id=INSTALLATION_ID,
            label="Jaden's phone",
            device_public_key_der=public_der,
        )
        owner.pairing.approve(pairing_id=created.pairing_id)
        payload = owner.pairing.proof_challenge_claims(created.pairing_id)
        assert payload is not None
        claims = DeviceProofClaims.from_payload(payload)
        signature = key.sign(
            canonical_device_proof_bytes(claims),
            ec.ECDSA(hashes.SHA256()),
        )
        proof_headers = Headers([
            ("X-Nanobot-Pairing-Id", created.pairing_id),
            ("X-Nanobot-Device-Signature", base64.b64encode(signature).decode("ascii")),
            ("X-Nanobot-Proof-Claims", base64.b64encode(
                json.dumps(claims.payload(), separators=(",", ":")).encode()
            ).decode("ascii")),
        ])
        request = WsRequest("/mobile/v1/pair/prove", proof_headers)

        proven = await services.endpoint.process_request(
            None,
            request,
            is_allowed=lambda _client: True,
        )
        assert proven is not None and proven.status_code == 200
        result = json.loads(proven.body)
        assert result["installation_id"] == INSTALLATION_ID
        assert (
            owner.tokens.validate_access(result["access_token"]).installation_id
            == INSTALLATION_ID
        )

        replayed = await services.endpoint.process_request(
            None,
            request,
            is_allowed=lambda _client: True,
        )
        assert replayed is not None and replayed.status_code == 410
        assert json.loads(replayed.body)["error"] == "pairing_consumed"
    finally:
        owner.stop()
