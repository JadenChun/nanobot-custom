from __future__ import annotations

from nanobot.webui.ws_http import _filter_mobile_commands


def test_mobile_command_filter_keeps_chat_commands_only() -> None:
    payload = {
        "commands": [
            {"command": "/model"},
            {"command": "/stop"},
            {"command": "/pairing"},
            {"command": "/restart"},
            {"command": "/dream"},
            {"command": "/settings"},
            {"command": "/help"},
        ]
    }
    assert _filter_mobile_commands(payload) == {
        "commands": [
            {"command": "/model"},
            {"command": "/stop"},
            {"command": "/help"},
        ]
    }


def test_request_settings_version_negotiation_and_source_binding() -> None:
    from nanobot.webui.inbound_commands import request_settings_error

    assert request_settings_error(
        {"request_settings_version": 1},
        current_source_token="tok-current",
        active_turn_id=None,
    ) is None
    assert request_settings_error(
        {"request_settings_version": 2},
        current_source_token="tok-current",
        active_turn_id=None,
    ) == ("update_required", "request_settings_version")
    assert request_settings_error(
        {"request_settings_version": 1, "source_token": "tok-stale"},
        current_source_token="tok-current",
        active_turn_id=None,
    ) == ("source_changed", "source_token")
    assert request_settings_error(
        {"request_settings_version": 1, "source_token": 3},
        current_source_token="tok-current",
        active_turn_id=None,
    ) == ("invalid_request_settings", "source_token")
    assert request_settings_error(
        {"request_settings_version": 1, "full_confirmation_id": 3},
        current_source_token=None,
        active_turn_id=None,
    ) == ("invalid_request_settings", "full_confirmation_id")


def test_request_intent_never_reinterprets_guidance_as_a_new_run() -> None:
    from nanobot.webui.inbound_commands import request_settings_error

    assert request_settings_error(
        {"request_intent": "steer", "target_turn_id": "turn-1"},
        current_source_token=None,
        active_turn_id="turn-1",
    ) == ("not_admitted", "steering_unavailable")
    assert request_settings_error(
        {"request_intent": "steer", "target_turn_id": "turn-old"},
        current_source_token=None,
        active_turn_id="turn-1",
    ) == ("not_admitted", "steering_target_changed")
    assert request_settings_error(
        {"request_intent": "new"},
        current_source_token=None,
        active_turn_id="turn-1",
    ) is None
    assert request_settings_error(
        {"request_intent": "replay"},
        current_source_token=None,
        active_turn_id="turn-1",
    ) == ("invalid_request_settings", "request_intent")
