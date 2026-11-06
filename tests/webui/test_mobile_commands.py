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


def test_request_settings_requires_negotiated_v1_fields() -> None:
    from nanobot.webui.inbound_commands import request_settings_error

    base = {
        "request_settings_version": 1,
        "request_intent": "new",
    }
    assert request_settings_error(base, active_turn_id=None) is None
    assert request_settings_error(
        {**base, "request_settings_version": None},
        active_turn_id=None,
    ) == ("update_required", "request_settings_version")
    assert request_settings_error(
        {**base, "request_settings_version": 2},
        active_turn_id=None,
    ) == ("update_required", "request_settings_version")
    assert request_settings_error(
        {"request_intent": "new"},
        active_turn_id=None,
    ) == ("update_required", "request_settings_version")


def test_request_settings_rejects_message_level_control_fields() -> None:
    from nanobot.webui.inbound_commands import request_settings_error

    base = {
        "request_settings_version": 1,
        "request_intent": "new",
    }
    assert request_settings_error(
        {
            **base,
            "source_token": "tok-current",
        },
        active_turn_id=None,
    ) == ("invalid_request_settings", "source_token")
    assert request_settings_error(
        {
            **base,
            "workspace_scope": {
                "project_path": "/tmp/project",
                "access_mode": "restricted",
            },
        },
        active_turn_id=None,
    ) == ("invalid_request_settings", "workspace_scope")
    for confirmation in ("full-confirmation", 3):
        assert request_settings_error(
            {
                **base,
                "full_confirmation_id": confirmation,
            },
            active_turn_id=None,
        ) == ("invalid_request_settings", "full_confirmation_id")


def test_request_intent_target_validation() -> None:
    from nanobot.webui.inbound_commands import request_settings_error

    def check(
        extra: dict[str, object],
        *,
        active_turn_id: str | None = "turn-1",
    ) -> tuple[str, str] | None:
        return request_settings_error(
            {"request_settings_version": 1, **extra},
            active_turn_id=active_turn_id,
        )

    assert check({}) == ("invalid_request_settings", "request_intent")
    assert check({"request_intent": "continue"}) == (
        "invalid_request_settings",
        "request_intent",
    )
    assert check({"request_intent": "new", "target_turn_id": "turn-1"}) == (
        "invalid_request_settings",
        "target_turn_id",
    )
    assert check({"request_intent": "steer", "target_turn_id": "turn-1"}) is None
    assert check({"request_intent": "steer"}) == (
        "not_admitted",
        "steering_target_changed",
    )
    assert check({"request_intent": "steer", "target_turn_id": "turn-old"}) == (
        "not_admitted",
        "steering_target_changed",
    )
