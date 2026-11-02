from __future__ import annotations

import threading
from pathlib import Path

import pytest

from nanobot.remote.auth import AuthRejectedError, TokenService
from nanobot.remote.store import RemoteAccessStore


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def service(tmp_path: Path, clock: Clock | None = None) -> tuple[RemoteAccessStore, TokenService]:
    clock = clock or Clock()
    store = RemoteAccessStore(tmp_path / "remote-access", clock=clock)
    store.load_or_create()
    store.set_enabled(True)
    return store, TokenService(store, clock=clock)


def test_access_refresh_rotation_and_revocation(tmp_path) -> None:
    store, tokens = service(tmp_path)
    pair = tokens.issue("install-0001", ("chat.read", "chat.write"))
    grant = tokens.validate_access(pair.access_token)
    assert grant.installation_id == "install-0001"
    assert grant.scopes == ("chat.read", "chat.write")

    rotated = tokens.refresh("install-0001", pair.refresh_token)
    assert rotated.refresh_token != pair.refresh_token
    with pytest.raises(AuthRejectedError):
        tokens.validate_access(pair.access_token)
    assert tokens.validate_access(rotated.access_token).installation_id == "install-0001"
    with pytest.raises(AuthRejectedError):
        tokens.refresh("install-0001", pair.refresh_token)

    assert tokens.revoke("install-0001") is True
    with pytest.raises(AuthRejectedError):
        tokens.validate_access(rotated.access_token)
    with pytest.raises(AuthRejectedError):
        tokens.refresh("install-0001", rotated.refresh_token)


def test_access_expiry_and_owner_restart(tmp_path) -> None:
    clock = Clock()
    store, tokens = service(tmp_path, clock)
    pair = tokens.issue("install-0001", ("chat.read",))
    clock.advance(601)
    with pytest.raises(AuthRejectedError):
        tokens.validate_access(pair.access_token)

    restarted_tokens = TokenService(store, clock=clock)
    with pytest.raises(AuthRejectedError):
        restarted_tokens.validate_access(pair.access_token)
    refreshed = restarted_tokens.refresh("install-0001", pair.refresh_token)
    assert restarted_tokens.validate_access(refreshed.access_token).scopes == ("chat.read",)


def test_scope_reduction_and_self_logout(tmp_path) -> None:
    store, tokens = service(tmp_path)
    pair = tokens.issue("install-0001", ("chat.read", "chat.write"))
    store.update_scopes("install-0001", ("chat.read",))
    grant = tokens.validate_access(pair.access_token)
    assert grant.scopes == ("chat.read",)
    assert tokens.logout(pair.access_token) is True
    with pytest.raises(AuthRejectedError):
        tokens.validate_access(pair.access_token)
    with pytest.raises(AuthRejectedError):
        tokens.refresh("install-0001", pair.refresh_token)


def test_disabled_store_rejects_issue_and_validate(tmp_path) -> None:
    store = RemoteAccessStore(tmp_path)
    store.load_or_create()
    tokens = TokenService(store)
    with pytest.raises(AuthRejectedError):
        tokens.issue("install-0001", ("chat.read",))
    store.set_enabled(True)
    pair = tokens.issue("install-0002", ("chat.read",))
    store.set_enabled(False)
    with pytest.raises(AuthRejectedError):
        tokens.validate_access(pair.access_token)


def test_refresh_idle_expiry_and_refresh_revoke_race(tmp_path) -> None:
    clock = Clock()
    store, tokens = service(tmp_path, clock)
    pair = tokens.issue("install-0001", ("chat.read",))
    clock.advance(30 * 24 * 60 * 60 + 1)
    with pytest.raises(AuthRejectedError):
        tokens.refresh("install-0001", pair.refresh_token)

    clock = Clock()
    store, tokens = service(tmp_path / "race", clock)
    pair = tokens.issue("install-0001", ("chat.read",))
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def refresh() -> None:
        barrier.wait()
        try:
            tokens.refresh("install-0001", pair.refresh_token)
            outcomes.append("refreshed")
        except AuthRejectedError:
            outcomes.append("rejected")

    def revoke() -> None:
        barrier.wait()
        tokens.revoke("install-0001")
        outcomes.append("revoked")

    workers = [threading.Thread(target=refresh), threading.Thread(target=revoke)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert sorted(outcomes) == ["refreshed", "revoked"] or sorted(outcomes) == ["rejected", "revoked"]
    with pytest.raises(AuthRejectedError):
        tokens.validate_access(pair.access_token)
    assert store.get_device("install-0001").revoked
