from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nanobot.remote.protocol import (
    ACCESS_TTL_SECONDS,
    BRIDGE_VERSION,
    CHALLENGE_TTL_SECONDS,
    MOBILE_PROTOCOL_VERSION,
    REFRESH_IDLE_SECONDS,
    ROUTES,
    SUPPORTED_SCOPES,
    DeviceRecord,
    HandshakeRequest,
    MutationDecision,
    MutationEnvelope,
    OperationKey,
    PairingChallenge,
    admit_mutation,
    default_limits,
    stable_operation_digest,
    validate_installation_id,
    validate_origin,
    validate_scopes,
)


def envelope(**overrides: object) -> MutationEnvelope:
    values: dict[str, object] = {
        "request_id": "req-0001",
        "installation_id": "install-0001",
        "principal": "device-0001",
        "action": "chat.send",
        "digest": stable_operation_digest("device-0001", "chat.send", {"text": "hello"}),
        "gateway_generation": 2,
        "epoch": 7,
        "mode": "submit",
        "payload": {"text": "hello"},
    }
    values.update(overrides)
    return MutationEnvelope.model_validate(values)


def test_contract_file_matches_registry() -> None:
    path = Path(__file__).resolve().parents[2] / "mobile" / "contracts" / "mobile-v1.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["version"] == MOBILE_PROTOCOL_VERSION
    assert contract["bridge_version"] == BRIDGE_VERSION
    assert contract["limits"] == default_limits()
    assert sorted(contract["scopes"]) == sorted(SUPPORTED_SCOPES)
    assert contract["mandatory_capabilities"] == ["chat"]
    expected = [
        {
            "method": route.method,
            "path": route.path,
            "scope": route.scope,
            "replay": route.replay,
            "websocket": route.websocket,
        }
        for route in ROUTES
    ]
    assert contract["routes"] == expected
    assert contract["pairing"]["challenge_ttl_seconds"] == CHALLENGE_TTL_SECONDS
    assert contract["pairing"]["access_ttl_seconds"] == ACCESS_TTL_SECONDS
    assert contract["pairing"]["refresh_idle_seconds"] == REFRESH_IDLE_SECONDS


def test_origin_and_installation_validation() -> None:
    assert validate_origin("https://fixture.example:8443/") == "https://fixture.example:8443"
    for value in (
        "http://fixture.example",
        "https://fixture.example/path",
        "https://fixture.example?x=1",
        "https://user@fixture.example",
        "https://fixture.example#fragment",
        "fixture.example",
    ):
        with pytest.raises(Exception):
            validate_origin(value)
    assert validate_installation_id("install-0001") == "install-0001"
    for value in ("short", "has space in it", "-leading"):
        with pytest.raises(Exception):
            validate_installation_id(value)


def test_handshake_versions_and_mandatory_chat() -> None:
    request = HandshakeRequest.model_validate(
        {
            "mobile_protocol": 1,
            "bridge_version": 1,
            "installation_id": "install-0001",
            "origin": "https://fixture.example:8443",
            "capabilities": {"chat": True, "media": True, "lifecycle": False},
        }
    )
    assert request.capabilities.chat is True
    assert request.capabilities.lifecycle is False
    for overrides in (
        {"mobile_protocol": 2},
        {"bridge_version": 2},
        {"capabilities": {"chat": False}},
    ):
        values = {
            "mobile_protocol": 1,
            "bridge_version": 1,
            "installation_id": "install-0001",
            "origin": "https://fixture.example:8443",
            "capabilities": {"chat": True},
        }
        values.update(overrides)
        with pytest.raises((ValidationError, ValueError)):
            HandshakeRequest.model_validate(values)


def test_mutation_digest_is_stable_and_sensitive() -> None:
    first = stable_operation_digest("device-0001", "chat.send", {"text": "hello"})
    second = stable_operation_digest("device-0001", "chat.send", {"text": "hello"})
    assert first == second
    assert first != stable_operation_digest("device-0001", "chat.send", {"text": "bye"})
    assert first != stable_operation_digest("device-0002", "chat.send", {"text": "hello"})
    assert first != stable_operation_digest("device-0001", "chat.stop", {"text": "hello"})


def test_operation_key_is_immutable_and_typed() -> None:
    key = OperationKey("install-0001", 2, "req-0001")
    assert key.installation_id == "install-0001"
    with pytest.raises(Exception):
        OperationKey("install-0001", 0, "req-0001")
    with pytest.raises(Exception):
        OperationKey("install-0001", 2, "bad request")


def test_admit_mutation_decisions() -> None:
    request = envelope()
    applied = admit_mutation(
        request,
        current_generation=2,
        current_epoch=7,
        record=None,
        record_digest=None,
        record_principal=None,
    )
    assert applied.decision is MutationDecision.APPLIED
    assert applied.effect_count == 1
    assert applied.http_status == 200

    unknown = admit_mutation(
        envelope(mode="replay_only"),
        current_generation=2,
        current_epoch=7,
        record=None,
        record_digest=None,
        record_principal=None,
    )
    assert unknown.decision is MutationDecision.OUTCOME_UNKNOWN
    assert unknown.effect_count == 0
    assert unknown.http_status == 409

    record = OperationKey("install-0001", 2, "req-0001")
    replayed = admit_mutation(
        envelope(mode="replay_only", epoch=1),
        current_generation=2,
        current_epoch=7,
        record=record,
        record_digest=request.digest,
        record_principal=request.principal,
    )
    assert replayed.decision is MutationDecision.REPLAYED
    assert replayed.effect_count == 1

    for overrides in (
        {"gateway_generation": 3},
        {"epoch": 1, "mode": "submit"},
        {"digest": stable_operation_digest("device-0001", "chat.send", {"text": "changed"})},
        {"principal": "device-0002"},
        {"request_id": "req-0002"},
        {"installation_id": "install-0002"},
    ):
        denied = admit_mutation(
            envelope(**overrides),
            current_generation=2,
            current_epoch=7,
            record=record,
            record_digest=request.digest,
            record_principal=request.principal,
        )
        assert denied.decision is MutationDecision.DENIED
        assert denied.http_status == 409
        assert denied.effect_count == 0


def test_pairing_challenge_and_device_scopes() -> None:
    challenge = PairingChallenge(
        pairing_id="pair-0001",
        origin="https://fixture.example:8443",
        installation_id="install-0001",
        nonce="n" * 32,
        label="Jaden's phone",
        expires_at=100,
        chat_consent=True,
        media_consent=False,
    )
    assert challenge.expires_at == 100
    device = DeviceRecord(
        installation_id="install-0001",
        label="Jaden's phone",
        scopes=("chat.read", "chat.write"),
        created_at=1,
        last_seen_at=2,
    )
    assert device.revoked_at is None
    with pytest.raises(Exception):
        DeviceRecord(
            installation_id="install-0001",
            label="phone",
            scopes=("admin.all",),
            created_at=1,
            last_seen_at=2,
        )
    with pytest.raises(Exception):
        validate_scopes(("chat.read", "admin.all"))


def test_ttl_and_limit_defaults() -> None:
    limits = default_limits()
    assert limits["access_ttl_seconds"] <= 600
    assert limits["challenge_ttl_seconds"] == 120
    assert limits["refresh_idle_seconds"] == 30 * 24 * 60 * 60
    assert limits["max_ws_frame_bytes"] == 37_748_736
    assert limits["max_attachment_bytes"] == 6 * 1024 * 1024
    assert limits["max_media_response_bytes"] == 8 * 1024 * 1024


def test_route_registry_is_unique_and_scoped() -> None:
    seen: set[tuple[str, str]] = set()
    for route in ROUTES:
        key = (route.method, route.path)
        assert key not in seen
        seen.add(key)
        assert route.scope in SUPPORTED_SCOPES or route.scope == "local.admin"
        if route.replay != "none":
            assert route.method == "POST"
