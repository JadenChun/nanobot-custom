"""Author case U3 for A3-7 AC7.2: explicit instances keep owner state apart.

Source case: a3-evidence/a3-reconciliation-2026-11-14/req-a37/CASES.md (U3),
authored by the requirements-only session. The instance root is the directory of
the explicit config path; an explicit ``stateDir`` or ``socketPath`` still wins.
"""

from __future__ import annotations

import json
from pathlib import Path

from websockets.http11 import Request as WsRequest

from nanobot.config.loader import save_config
from nanobot.config.schema import Config
from nanobot.remote.composition import owner_socket_path, remote_access_root
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.webui.settings_remote import handle_remote_access
from nanobot.webui.settings_services import WebUISettingsConfig

_PREFIX = "/api/settings/remote-access"


def _request(path: str, payload: dict | None = None) -> WsRequest:
    request = WsRequest(path, {})
    if payload is not None:
        setattr(request, "_nanobot_webui_mutation_payload", payload)
    return request


def _instance(root: Path) -> WebUISettingsConfig:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    save_config(
        Config.model_validate({"remoteAccess": {"enabled": True}}),
        config_path,
    )
    return WebUISettingsConfig(config_path)


def _listing(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir())


def test_u3_explicit_instances_resolve_distinct_owner_paths(tmp_path: Path) -> None:
    root_a = tmp_path / "instance-a"
    root_b = tmp_path / "instance-b"
    settings_a = _instance(root_a)
    settings_b = _instance(root_b)

    config_a = settings_a.load().remote_access
    config_b = settings_b.load().remote_access
    socket_a = owner_socket_path(config_a, instance_root=settings_a.path.parent)
    socket_b = owner_socket_path(config_b, instance_root=settings_b.path.parent)
    state_a = remote_access_root(config_a, instance_root=settings_a.path.parent)
    state_b = remote_access_root(config_b, instance_root=settings_b.path.parent)

    assert socket_a != socket_b
    assert state_a != state_b
    assert socket_a.resolve(strict=False).is_relative_to(root_a.resolve())
    assert socket_b.resolve(strict=False).is_relative_to(root_b.resolve())
    assert not socket_a.resolve(strict=False).is_relative_to(root_b.resolve())
    assert not socket_b.resolve(strict=False).is_relative_to(root_a.resolve())
    assert not socket_a.is_symlink() and not socket_b.is_symlink()


def test_u3_instance_a_cannot_reach_instance_b_owner(tmp_path: Path) -> None:
    root_a = tmp_path / "instance-a"
    root_b = tmp_path / "instance-b"
    settings_a = _instance(root_a)
    settings_b = _instance(root_b)
    state_b = root_b / "remote-access"

    owner = RemoteAccessOwner(
        root_a / "remote-access",
        serve_verifier=lambda _origin: True,
    )
    owner.start()
    try:
        enabled = handle_remote_access(
            settings_a,
            _request(f"{_PREFIX}/enable", {"enabled": True}),
            f"{_PREFIX}/enable",
        )
        assert enabled.status_code == 200
        owner.tokens.issue("device-a", ("chat.read",))

        # Part B: a read through A's owner returns A's data only.
        status_a = handle_remote_access(settings_a, _request(_PREFIX), _PREFIX)
        payload_a = json.loads(status_a.body)
        assert payload_a["owner_available"] is True
        installations = {device["installation_id"] for device in payload_a["devices"]}
        assert "device-a" in installations

        # B resolves its own owner path, which has no process behind it.
        status_b = handle_remote_access(settings_b, _request(_PREFIX), _PREFIX)
        payload_b = json.loads(status_b.body)
        assert payload_b["owner_available"] is False
        assert payload_b["devices"] == []
        assert not (state_b / "owner.sock").exists()
        assert not (state_b / "state.json").exists()

        before = _listing(state_b)
        pairing = handle_remote_access(
            settings_a,
            _request(
                f"{_PREFIX}/pairing/start",
                {
                    "origin": "https://fixture.example:8443",
                    "label": "phone",
                    "fullControlConsent": True,
                },
            ),
            f"{_PREFIX}/pairing/start",
        )
        assert pairing.status_code == 200
        assert _listing(state_b) == before
        assert owner.socket_path.is_relative_to(root_a)
    finally:
        owner.stop()
