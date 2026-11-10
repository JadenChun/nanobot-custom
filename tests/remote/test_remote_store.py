from __future__ import annotations

import json
import os
import stat
import threading

import pytest

from nanobot.remote.store import (
    RemoteAccessStore,
    StoreCorruptError,
    StoreError,
    StoreVersionError,
    hash_refresh_token,
)


def test_store_defaults_modes_and_round_trip(tmp_path) -> None:
    store = RemoteAccessStore(tmp_path / "remote-access")
    assert store.load() is None
    state = store.load_or_create()
    assert state.enabled is False
    assert state.installation_id
    assert stat.S_IMODE(os.stat(store.root).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(store.state_path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(store.admin_token_path).st_mode) == 0o600 if store.admin_token_path.exists() else True

    store.set_enabled(True)
    device = store.add_device(
        installation_id="install-0001",
        label="phone",
        scopes=("chat.read", "chat.write"),
        refresh_token="refresh-token-1",
    )
    assert device.refresh_hash == hash_refresh_token("refresh-token-1")
    assert store.validate_refresh("install-0001", "refresh-token-1") is not None
    assert store.validate_refresh("install-0001", "wrong") is None
    store.rotate_refresh("install-0001", "refresh-token-2")
    assert store.validate_refresh("install-0001", "refresh-token-1") is None
    assert store.validate_refresh("install-0001", "refresh-token-2") is not None
    store.revoke_device("install-0001")
    assert store.validate_refresh("install-0001", "refresh-token-2") is None
    snapshot = store.snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["devices"][0]["status"] == "revoked"
    assert "refresh_hash" not in json.dumps(snapshot)


def test_store_rejects_unknown_version_and_corruption_without_overwrite(tmp_path) -> None:
    store = RemoteAccessStore(tmp_path)
    store.load_or_create()
    original = store.state_path.read_text(encoding="utf-8")
    raw = json.loads(original)
    raw["version"] = 99
    store.state_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StoreVersionError):
        store.load()
    store.state_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(StoreCorruptError):
        store.load()
    assert store.state_path.read_text(encoding="utf-8") == "{not-json"


def test_store_concurrent_writers_and_owned_entries(tmp_path) -> None:
    store = RemoteAccessStore(tmp_path)
    store.load_or_create()
    store.set_enabled(True)

    def add(index: int) -> None:
        store.add_device(
            installation_id=f"install-{index:04d}",
            label=f"phone-{index}",
            scopes=("chat.read",),
            refresh_token=f"refresh-{index}",
        )

    threads = [threading.Thread(target=add, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    snapshot = store.snapshot()
    assert len(snapshot["devices"]) == 10
    assert snapshot["revision"] == 12

    state = store.owned_entries(
        [{"kind": "serve", "value": "https://fixture.example"}, {"kind": "socket", "value": "owner.sock"}]
    )
    assert len(state.owned_entries) == 2
    with pytest.raises(StoreError):
        store.add_device(
            installation_id="install-0000",
            label="duplicate",
            scopes=("chat.read",),
            refresh_token="x",
        )


def test_admin_token_is_bounded_and_high_entropy(tmp_path) -> None:
    store = RemoteAccessStore(tmp_path)
    store.load_or_create()
    token = store.admin_token()
    assert len(token) >= 32
    assert store.check_admin_token(token)
    assert not store.check_admin_token("wrong")
    assert not store.check_admin_token(None)


def test_store_rejects_symlinked_state_file(tmp_path) -> None:
    store = RemoteAccessStore(tmp_path)
    store.load_or_create()
    target = tmp_path / "outside.json"
    store.state_path.rename(target)
    store.state_path.symlink_to(target)
    with pytest.raises(StoreCorruptError):
        store.load()


def test_owner_device_has_no_idle_expiry_and_legacy_helper_revokes_v1_bearer(tmp_path) -> None:
    class Clock:
        now = 1000.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    store = RemoteAccessStore(tmp_path, clock=clock)
    store.load_or_create()
    store.set_enabled(True)
    owner = store.add_device(
        installation_id="install-owner",
        label="Pixel",
        scopes=("owner", "chat.read", "chat.write"),
        refresh_token="owner-refresh",
    )
    legacy = store.add_device(
        installation_id="install-legacy",
        label="old phone",
        scopes=("chat.read", "chat.write"),
        refresh_token="legacy-refresh",
    )
    assert owner.refresh_expires_at == 0
    assert legacy.refresh_expires_at > clock.now

    clock.now += 365 * 24 * 60 * 60
    assert store.validate_refresh("install-owner", "owner-refresh") is not None
    assert store.validate_refresh("install-legacy", "legacy-refresh") is None

    assert set(store.revoke_legacy_chat_only_devices()) == {"install-owner", "install-legacy"}
    owner_device = store.get_device("install-owner")
    legacy_device = store.get_device("install-legacy")
    assert owner_device is not None and owner_device.revoked
    assert legacy_device is not None and legacy_device.revoked
    assert store.revoke_legacy_chat_only_devices() == []
