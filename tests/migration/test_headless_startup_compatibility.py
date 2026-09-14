"""S1 ChannelManager construction checks for headless compatibility."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.config.loader import load_config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        {"agents": {"defaults": {"workspace": "{workspace}"}}},
        {
            "agents": {"defaults": {"workspace": "{workspace}"}},
            "channels": {"websocket": {"host": "127.0.0.1", "port": 9876}},
        },
    ],
    ids=["legacy-config-without-gui-fields", "partial-websocket-config"],
)
async def test_channel_manager_construction_keeps_websocket_disabled_and_read_only(
    tmp_path,
    source,
):
    """ChannelManager construction does not activate or prepare the WebUI channel."""
    config_path = tmp_path / "config.json"
    raw_source = json.loads(json.dumps(source).replace("{workspace}", str(tmp_path / "workspace")))
    config_path.write_text(json.dumps(raw_source), encoding="utf-8")
    before = config_path.read_bytes()
    config = load_config(config_path)

    with (
        patch("nanobot.channels.manager._default_webui_dist") as webui_dist,
        patch("nanobot.webui.gateway_services.build_gateway_services") as build_gateway,
        patch("nanobot.cli.webui_support.inspect_webui_bundle") as inspect_bundle,
        patch("nanobot.cli.webui_support.ensure_webui_bundle") as ensure_bundle,
        patch("nanobot.cli.webui_support._launch_browser") as launch_browser,
        patch("nanobot.config.loader.save_config") as save_config,
    ):
        manager = ChannelManager(config, MessageBus(), webui_static_dist=True)
        try:
            assert config.channels.websocket["enabled"] is False
            assert "websocket" not in manager.channels
            assert manager.get_status() == {}
            assert config_path.read_bytes() == before
            webui_dist.assert_not_called()
            build_gateway.assert_not_called()
            inspect_bundle.assert_not_called()
            ensure_bundle.assert_not_called()
            launch_browser.assert_not_called()
            save_config.assert_not_called()
        finally:
            await manager.stop_all()
