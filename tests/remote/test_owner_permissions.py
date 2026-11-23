from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.remote.client import RemoteAccessClient
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.permissions import (
    OwnerPermissionAdapter,
    PermissionAdapterError,
)


def _owner(tmp_path: Path) -> RemoteAccessOwner:
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    return owner


def _pair_owner(owner: RemoteAccessOwner) -> str:
    offer = owner.pairing.create_challenge(
        origin="https://fixture.example:8443",
        label="phone",
        chat_consent=True,
        media_consent=True,
        full_control_consent=True,
        shared_history_consent=True,
    )
    from tests.remote.proof_helpers import approve_and_prove, collect_with_key

    collect_with_key(
        owner.pairing,
        pairing_id=offer.pairing_id,
        nonce=offer.nonce,
        collector_secret=offer.collector_secret,
        installation_id="install-owner",
        label="phone",
    )
    approve_and_prove(
        owner.pairing,
        pairing_id=offer.pairing_id,
        comparison_code=offer.comparison_code,
    )
    result = owner.pairing.poll(
        pairing_id=offer.pairing_id,
        nonce=offer.nonce,
        collector_secret=offer.collector_secret,
    )
    return result.access_token


def test_owner_permission_adapter_consumes_admission(tmp_path) -> None:
    owner = _owner(tmp_path)
    try:
        token = _pair_owner(owner)
        adapter = OwnerPermissionAdapter(
            RemoteAccessClient(owner.socket_path, timeout=2.0)
        )
        context = adapter.authorize(
            access_token=token,
            installation_id="install-owner",
            action="workspace.access.mode.set",
            payload={"chat_id": "chat-1", "mode": "full"},
        )
        assert context.installation_id == "install-owner"
        assert context.operation.value == "full.commit"
        journal = owner.store.snapshot()["audit"]
        assert journal[-1]["operation"] == "full.commit"
        assert journal[-1]["decision"] == "consumed"
    finally:
        owner.stop()


def test_owner_permission_adapter_mints_grant_release(tmp_path) -> None:
    owner = _owner(tmp_path)
    try:
        token = _pair_owner(owner)
        adapter = OwnerPermissionAdapter(
            RemoteAccessClient(owner.socket_path, timeout=2.0)
        )
        context = adapter.authorize(
            access_token=token,
            installation_id="install-owner",
            action="workspace.grant.revoke",
            payload={"expected_revision": 3, "path": "/tmp/project"},
        )
        assert context.installation_id == "install-owner"
        assert context.operation.value == "grant.release"
        journal = owner.store.snapshot()["audit"]
        assert journal[-1]["operation"] == "grant.release"
        assert journal[-1]["decision"] == "consumed"
    finally:
        owner.stop()


def test_owner_permission_adapter_denies_revoked_device(tmp_path) -> None:
    owner = _owner(tmp_path)
    try:
        token = _pair_owner(owner)
        adapter = OwnerPermissionAdapter(
            RemoteAccessClient(owner.socket_path, timeout=2.0)
        )
        owner.tokens.revoke("install-owner")
        with pytest.raises(PermissionAdapterError) as exc:
            adapter.authorize(
                access_token=token,
                installation_id="install-owner",
                action="workspace.access.prepare_full",
                payload={"chat_id": "chat-1"},
            )
        assert exc.value.status == 403
        assert exc.value.code in {"invalid_access", "device_revoked"}
    finally:
        owner.stop()


def test_owner_permission_adapter_rejects_unknown_action(tmp_path) -> None:
    owner = _owner(tmp_path)
    try:
        token = _pair_owner(owner)
        adapter = OwnerPermissionAdapter(
            RemoteAccessClient(owner.socket_path, timeout=2.0)
        )
        with pytest.raises(PermissionAdapterError) as exc:
            adapter.authorize(
                access_token=token,
                installation_id="install-owner",
                action="workspace.unknown",
                payload={},
            )
        assert exc.value.status == 400
    finally:
        owner.stop()
