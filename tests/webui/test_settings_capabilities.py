from __future__ import annotations

from typing import Any

import httpx

from nanobot.config.schema import Config, MCPServerConfig
from nanobot.webui.settings_capabilities import (
    capability_settings_payload,
    update_api_settings,
    update_image_generation_settings,
    update_network_safety_settings,
    update_transcription_settings,
    update_web_search_settings,
)


def _oauth_status(_spec: Any) -> dict[str, Any]:
    return {"configured": False}


def test_capability_domain_updates_representative_settings() -> None:
    config = Config()
    config.providers.openrouter.api_key = "sk-test"

    web_changed, web_restart = update_web_search_settings(
        config,
        {
            "provider": ["duckduckgo"],
            "max_results": ["7"],
            "use_jina_reader": ["false"],
        },
    )
    update_api_settings(
        config,
        {"host": ["127.0.0.2"], "port": ["8900"], "timeout": ["90"]},
    )
    image_changed = update_image_generation_settings(
        config,
        {"enabled": ["true"], "provider": ["openrouter"]},
        oauth_status=_oauth_status,
    )
    transcription_changed = update_transcription_settings(
        config,
        {"provider": ["openrouter"], "model": ["openai/whisper-large-v3"]},
    )
    network_changed, access_mode = update_network_safety_settings(
        config,
        {
            "webui_allow_local_service_access": ["false"],
            "webui_default_access_mode": ["restricted"],
        },
    )
    payload = capability_settings_payload(config, oauth_status=_oauth_status)

    assert (web_changed, web_restart) == (True, True)
    assert image_changed is True
    assert transcription_changed is True
    assert (network_changed, access_mode) == (True, "default")
    assert payload["web_search"]["max_results"] == 7
    assert payload["api"]["host"] == "127.0.0.2"
    assert payload["api"]["port"] == 8900
    assert payload["image_generation"]["enabled"] is True
    assert payload["transcription"]["provider"] == "openrouter"


def test_capability_payload_reports_unconfigured_comfy_mcp() -> None:
    payload = capability_settings_payload(Config(), oauth_status=_oauth_status)
    comfy = payload["image_generation"]["comfy"]

    assert comfy["configured"] is False
    assert comfy["reachable"] is False
    assert comfy["base_url"] is None
    assert comfy["enabled_tools"] == []


def test_capability_payload_reports_configured_comfy_mcp(
    monkeypatch,
) -> None:
    config = Config()
    config.tools.mcp_servers["comfy"] = MCPServerConfig(
        command="/usr/bin/node",
        args=["/opt/comfy/server.js"],
        env={"COMFY_AGENT_COMFY_URL": "http://127.0.0.1:8188"},
        enabled_tools=["generate_image", "edit_image"],
    )

    def fake_get(url: str, **kwargs):
        assert url == "http://127.0.0.1:8188/system_stats"
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    payload = capability_settings_payload(config, oauth_status=_oauth_status)
    comfy = payload["image_generation"]["comfy"]

    assert comfy["configured"] is True
    assert comfy["reachable"] is True
    assert comfy["base_url"] == "http://127.0.0.1:8188"
    assert comfy["enabled_tools"] == ["generate_image", "edit_image"]
