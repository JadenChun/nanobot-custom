from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.remote.auth import TokenService
from nanobot.remote.delegation import (
    DelegatedAdminOperation,
    DelegationError,
    OwnerDelegationService,
    current_digest,
)
from nanobot.remote.pairing import PairingRejectedError, PairingService
from nanobot.remote.store import RemoteAccessStore


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def setup(
    tmp_path: Path, clock: Clock | None = None
) -> tuple[RemoteAccessStore, TokenService, PairingService, OwnerDelegationService]:
    store = RemoteAccessStore(tmp_path / "remote-access")
    store.load_or_create()
    store.set_enabled(True)
    clock = clock or Clock()
    tokens = TokenService(store, clock=clock)
    pairing = PairingService(store, tokens, clock=clock)
    delegation = OwnerDelegationService(store, tokens, clock=clock)
    return store, tokens, pairing, delegation


def pair(
    pairing: PairingService,
    *,
    installation_id: str,
    full_control: bool,
    shared_history: bool | None = None,
) -> tuple[str, str]:
    offer = pairing.create_challenge(
        origin="https://fixture.example:8443",
        label="phone",
        chat_consent=True,
        media_consent=False,
        full_control_consent=full_control,
        shared_history_consent=full_control if shared_history is None else shared_history,
    )
    pairing.collect(
        pairing_id=offer.pairing_id,
        nonce=offer.nonce,
        collector_secret=offer.collector_secret,
        installation_id=installation_id,
        label="phone",
    )
    pairing.approve(pairing_id=offer.pairing_id, comparison_code=offer.comparison_code)
    result = pairing.poll(
        pairing_id=offer.pairing_id,
        nonce=offer.nonce,
        collector_secret=offer.collector_secret,
    )
    return result.access_token, result.refresh_token


def authorize(
    delegation: OwnerDelegationService,
    token: str,
    *,
    installation_id: str,
    operation: str,
    arguments: dict | None = None,
    request_id: str = "req-1",
    digest: str | None = None,
):
    arguments = arguments or {}
    return delegation.authorize(
        access_token=token,
        operation=operation,
        request_id=request_id,
        digest=digest or current_digest(operation, request_id, arguments, installation_id),
        arguments=arguments,
    )


def test_owner_pairing_grants_owner_without_widening_legacy(tmp_path) -> None:
    store, _tokens, pairing, _delegation = setup(tmp_path)
    pair(pairing, installation_id="install-legacy", full_control=False)
    pair(pairing, installation_id="install-owner", full_control=True)

    legacy = store.get_device("install-legacy")
    owner = store.get_device("install-owner")
    assert legacy is not None and owner is not None
    assert "owner" not in legacy.scopes
    assert "owner" in owner.scopes
    assert {"chat.read", "chat.write"}.issubset(set(legacy.scopes))


def test_shared_history_consent_requires_full_control(tmp_path) -> None:
    _store, _tokens, pairing, _delegation = setup(tmp_path)
    with pytest.raises(PairingRejectedError):
        pairing.create_challenge(
            origin="https://fixture.example:8443",
            label="phone",
            chat_consent=True,
            media_consent=False,
            full_control_consent=False,
            shared_history_consent=True,
        )


def test_owner_scope_cannot_be_requested_without_consent(tmp_path) -> None:
    _store, _tokens, pairing, _delegation = setup(tmp_path)
    with pytest.raises(PairingRejectedError):
        pairing.create_challenge(
            origin="https://fixture.example:8443",
            label="phone",
            chat_consent=True,
            media_consent=False,
            scopes=("chat.read", "chat.write", "owner"),
        )


def test_owner_authorizes_typed_operation_and_admission_is_single_use(tmp_path) -> None:
    store, _tokens, pairing, delegation = setup(tmp_path)
    token, _refresh = pair(pairing, installation_id="install-owner", full_control=True)

    context, admission, payload = authorize(
        delegation,
        token,
        installation_id="install-owner",
        operation=DelegatedAdminOperation.DEVICE_LIST.value,
    )
    assert context.is_owner
    assert payload == {}
    consumed = delegation.consume(
        handle=admission.handle,
        installation_id="install-owner",
        operation=DelegatedAdminOperation.DEVICE_LIST.value,
        digest=admission.digest,
    )
    assert consumed.installation_id == "install-owner"
    with pytest.raises(DelegationError):
        delegation.consume(
            handle=admission.handle,
            installation_id="install-owner",
            operation=DelegatedAdminOperation.DEVICE_LIST.value,
            digest=admission.digest,
        )
    assert store.get_device("install-owner") is not None


def test_legacy_device_denies_owner_operations_but_can_self_revoke(tmp_path) -> None:
    _store, _tokens, pairing, delegation = setup(tmp_path)
    token, _refresh = pair(pairing, installation_id="install-legacy", full_control=False)

    with pytest.raises(DelegationError) as exc:
        authorize(
            delegation,
            token,
            installation_id="install-legacy",
            operation=DelegatedAdminOperation.DEVICE_LIST.value,
        )
    assert exc.value.code == "owner_required"

    context, _admission, _payload = authorize(
        delegation,
        token,
        installation_id="install-legacy",
        operation=DelegatedAdminOperation.DEVICE_SELF_REVOKE.value,
    )
    assert not context.is_owner


def test_local_only_admin_operations_are_not_delegable(tmp_path) -> None:
    _store, _tokens, pairing, delegation = setup(tmp_path)
    token, _refresh = pair(pairing, installation_id="install-owner", full_control=True)
    for operation in ("enable", "pair.create", "pair.approve", "scope.upgrade"):
        with pytest.raises(DelegationError) as exc:
            authorize(
                delegation,
                token,
                installation_id="install-owner",
                operation=operation,
            )
        assert exc.value.code == "unknown_operation"


def test_caller_cannot_assert_identity_locality_or_scope(tmp_path) -> None:
    _store, _tokens, pairing, delegation = setup(tmp_path)
    token, _refresh = pair(pairing, installation_id="install-owner", full_control=True)
    for forbidden in (
        {"is_local": True},
        {"is_native": True},
        {"installation_id": "install-other"},
        {"scopes": ["owner"]},
        {"admin_token": "x"},
    ):
        with pytest.raises(DelegationError) as exc:
            authorize(
                delegation,
                token,
                installation_id="install-owner",
                operation=DelegatedAdminOperation.DEVICE_LIST.value,
                arguments=forbidden,
            )
        assert exc.value.code == "invalid_arguments"


def test_digest_mismatch_and_other_device_admission_deny(tmp_path) -> None:
    _store, _tokens, pairing, delegation = setup(tmp_path)
    token, _refresh = pair(pairing, installation_id="install-owner", full_control=True)
    with pytest.raises(DelegationError) as exc:
        authorize(
            delegation,
            token,
            installation_id="install-owner",
            operation=DelegatedAdminOperation.DEVICE_LIST.value,
            digest="0" * 64,
        )
    assert exc.value.code == "digest_mismatch"

    _context, admission, _payload = authorize(
        delegation,
        token,
        installation_id="install-owner",
        operation=DelegatedAdminOperation.DEVICE_LIST.value,
    )
    with pytest.raises(DelegationError) as mismatch:
        delegation.consume(
            handle=admission.handle,
            installation_id="install-other",
            operation=DelegatedAdminOperation.DEVICE_LIST.value,
            digest=admission.digest,
        )
    assert mismatch.value.code == "admission_mismatch"


def test_expired_admission_and_revocation_deny_with_zero_effect(tmp_path) -> None:
    clock = Clock()
    store, _tokens, pairing, delegation = setup(tmp_path, clock)
    token, _refresh = pair(pairing, installation_id="install-owner", full_control=True)
    _context, admission, _payload = authorize(
        delegation,
        token,
        installation_id="install-owner",
        operation=DelegatedAdminOperation.DEVICE_LIST.value,
    )
    clock.advance(31)
    with pytest.raises(DelegationError) as exc:
        delegation.consume(
            handle=admission.handle,
            installation_id="install-owner",
            operation=DelegatedAdminOperation.DEVICE_LIST.value,
            digest=admission.digest,
        )
    assert exc.value.code == "admission_missing"

    _context, second, _payload = authorize(
        delegation,
        token,
        installation_id="install-owner",
        operation=DelegatedAdminOperation.DEVICE_LIST.value,
        request_id="req-2",
    )
    store.revoke_device("install-owner")
    with pytest.raises(DelegationError) as revoked:
        delegation.consume(
            handle=second.handle,
            installation_id="install-owner",
            operation=DelegatedAdminOperation.DEVICE_LIST.value,
            digest=second.digest,
        )
    assert revoked.value.code == "device_revoked"


def test_device_rename_and_revoke_are_owner_scoped(tmp_path) -> None:
    store, _tokens, pairing, delegation = setup(tmp_path)
    owner_token, _refresh = pair(pairing, installation_id="install-owner", full_control=True)
    _legacy_token, _refresh2 = pair(pairing, installation_id="install-legacy", full_control=False)

    renamed = delegation.rename_device("install-legacy", "old tablet")
    assert renamed == {"installation_id": "install-legacy", "label": "old tablet"}
    assert store.get_device("install-legacy").label == "old tablet"  # type: ignore[union-attr]

    revoked = delegation.revoke_device("install-legacy")
    assert revoked["revoked"] is True
    assert store.get_device("install-legacy").revoked is True  # type: ignore[union-attr]

    # The owner token remains valid after revoking another device.
    context = delegation.authenticate(owner_token)
    assert context.is_owner


def test_audit_is_bounded_sanitized_and_distinguishes_devices(tmp_path) -> None:
    store, _tokens, pairing, delegation = setup(tmp_path)
    token, _refresh = pair(pairing, installation_id="install-owner", full_control=True)
    _legacy_token, _refresh2 = pair(pairing, installation_id="install-legacy", full_control=False)

    _context, admission, _payload = authorize(
        delegation,
        token,
        installation_id="install-owner",
        operation=DelegatedAdminOperation.DEVICE_LIST.value,
        request_id="req-audit",
    )
    delegation.consume(
        handle=admission.handle,
        installation_id="install-owner",
        operation=DelegatedAdminOperation.DEVICE_LIST.value,
        digest=admission.digest,
    )
    delegation.rename_device("install-legacy", "tablet")
    delegation.revoke_device("install-legacy")
    early = store.snapshot()["audit"]
    assert any(
        entry["operation"] == "device.revoke" and entry["installation_id"] == "install-legacy"
        for entry in early
    )

    for index in range(250):
        store.record_audit(
            installation_id="install-owner",
            operation="device.list",
            decision="authorized",
            request_id=f"req-{index}",
        )
    snapshot = store.snapshot()
    assert len(snapshot["audit"]) == 100
    assert {device["installation_id"] for device in snapshot["devices"]} == {
        "install-owner",
        "install-legacy",
    }
    serialized = str(snapshot)
    assert "refresh" not in serialized.lower()
    assert "token" not in serialized.lower()


def test_product_pairing_path_grants_owner_via_ipc(tmp_path) -> None:
    from nanobot.remote.client import RemoteAccessClient
    from nanobot.remote.owner import RemoteAccessOwner

    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    try:
        client = RemoteAccessClient(owner.socket_path, admin_token=owner.store.admin_token())
        offer = client.create_pairing(
            origin="https://fixture.example:8443",
            label="phone",
            chat_consent=True,
            media_consent=True,
            full_control_consent=True,
            shared_history_consent=True,
        )
        assert "owner" in offer["scopes"]
        pending = client.collect_pairing(
            pairing_id=offer["pairing_id"],
            nonce=offer["nonce"],
            collector_secret=offer["collector_secret"],
            installation_id="install-product-owner",
            label="phone",
        )
        assert pending["comparison_code"] == offer["comparison_code"]
        client.approve_pairing(
            pairing_id=offer["pairing_id"],
            comparison_code=offer["comparison_code"],
        )
        result = client.poll_pairing(
            pairing_id=offer["pairing_id"],
            nonce=offer["nonce"],
            collector_secret=offer["collector_secret"],
        )
        assert result["status"] == "approved"
        assert "owner" in result["scopes"]
        assert owner.store.get_device("install-product-owner").scopes.count("owner") == 1  # type: ignore[union-attr]
    finally:
        owner.stop()


def test_product_pairing_without_consent_stays_legacy(tmp_path) -> None:
    from nanobot.remote.client import RemoteAccessClient
    from nanobot.remote.owner import RemoteAccessOwner

    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    try:
        client = RemoteAccessClient(owner.socket_path, admin_token=owner.store.admin_token())
        offer = client.create_pairing(
            origin="https://fixture.example:8443",
            label="phone",
            chat_consent=True,
            media_consent=False,
        )
        assert "owner" not in offer["scopes"]
        assert "chat.read" in offer["scopes"]
    finally:
        owner.stop()


def test_audit_records_acting_owner_for_device_revoke(tmp_path) -> None:
    store, _tokens, pairing, delegation = setup(tmp_path)
    pair(pairing, installation_id="install-owner", full_control=True)
    pair(pairing, installation_id="install-legacy", full_control=False)
    delegation.revoke_device("install-legacy", actor_installation_id="install-owner")
    entry = store.snapshot()["audit"][-1]
    assert entry["installation_id"] == "install-legacy"
    assert entry["actor_installation_id"] == "install-owner"
    assert entry["operation"] == "device.revoke"


def test_ipc_revoke_audits_acting_device(tmp_path) -> None:
    from nanobot.remote.client import RemoteAccessClient
    from nanobot.remote.owner import RemoteAccessOwner

    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    try:
        paired = owner.tokens.issue("install-target", ("chat.read", "chat.write"))
        client = RemoteAccessClient(owner.socket_path, admin_token=owner.store.admin_token())
        client.revoke("install-target", actor_installation_id="install-owner")
        entry = owner.store.snapshot()["audit"][-1]
        assert entry["installation_id"] == "install-target"
        assert entry["actor_installation_id"] == "install-owner"
        assert paired.scopes
    finally:
        owner.stop()
