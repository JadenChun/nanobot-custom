from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.remote.edge import AccessIdentity
from nanobot.session.manager import SessionManager
from nanobot.webui.gateway_services import build_gateway_services


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instance = tmp_path / "instance"
    home = instance / "workspace"
    home.mkdir(parents=True)
    config_path = instance / "config.json"
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)
    session_manager = SessionManager(home, sessions_root=instance / "sessions")
    config = WebSocketConfig.model_validate({
        "enabled": True,
        "allowFrom": ["*"],
        "host": "127.0.0.1",
        "port": 0,
        "path": "/ws",
        "websocketRequiresToken": True,
    })
    return build_gateway_services(
        config=config,
        bus=MessageBus(),
        session_manager=session_manager,
        static_dist_path=None,
        workspace_path=home,
        default_restrict_to_workspace=True,
        config_path=config_path,
        runtime_model_name=None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )


def test_owner_workspace_projection_enables_management(gateway) -> None:
    legacy = gateway.http.mobile_workspaces_payload(owner=False)
    owner = gateway.http.mobile_workspaces_payload(owner=True)
    assert legacy["controls"]["can_manage_folders"] is False
    assert owner["controls"]["can_manage_folders"] is True
    assert owner["controls"]["can_manage_catalog"] is True
    assert owner["management"] is not None
    assert legacy.get("management") is None


@pytest.mark.asyncio
async def test_owner_feature_reads_follow_standard_handlers(gateway) -> None:
    settings_status, settings = await gateway.http.mobile_feature_payload("/api/settings")
    assert settings_status == 200
    assert "agent" in settings or "providers" in settings

    roots_status, roots = await gateway.http.mobile_feature_payload(
        "/api/workspace-directories/roots"
    )
    assert roots_status == 200
    assert roots["roots"] == []
    assert roots["empty"] is True


@pytest.mark.asyncio
async def test_owner_mutation_payload_denies_local_only_and_planned(gateway) -> None:
    owner = AccessIdentity(
        "install-owner",
        ("chat.read", "chat.write", "owner"),
        1e12,
        1,
    )
    status, payload = await gateway.http.mobile_mutation_payload(
        owner, "settings.remote_access.enable", {"enabled": True}
    )
    assert status == 403
    assert payload["error"] == "operation_not_allowed"

    status, payload = await gateway.http.mobile_mutation_payload(
        owner, "workspace.grant.revoke", {"path": "/tmp/x", "expected_revision": 0}
    )
    assert status == 403
    assert payload["error"] == "operation_not_allowed"

    legacy = AccessIdentity("install-legacy", ("chat.read", "chat.write"), 1e12, 1)
    status, payload = await gateway.http.mobile_mutation_payload(
        legacy, "settings.agent.update", {}
    )
    assert status == 403
    assert payload["error"] == "owner_required"
