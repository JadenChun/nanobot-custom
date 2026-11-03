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
        "source_token": "tok-current",
    }
    assert request_settings_error(
        base,
        current_source_token="tok-current",
        active_turn_id=None,
    ) is None
    assert request_settings_error(
        {**base, "request_settings_version": None},
        current_source_token="tok-current",
        active_turn_id=None,
    ) == ("update_required", "request_settings_version")
    assert request_settings_error(
        {**base, "request_settings_version": 2},
        current_source_token="tok-current",
        active_turn_id=None,
    ) == ("update_required", "request_settings_version")
    assert request_settings_error(
        {
            "request_intent": "new",
            "source_token": "tok-current",
        },
        current_source_token="tok-current",
        active_turn_id=None,
    ) == ("update_required", "request_settings_version")


def test_request_settings_source_token_rules() -> None:
    from nanobot.webui.inbound_commands import request_settings_error

    for token in (None, 3):
        assert request_settings_error(
            {
                "request_settings_version": 1,
                "request_intent": "new",
                **({} if token is None else {"source_token": token}),
            },
            current_source_token="tok-current",
            active_turn_id=None,
        ) == ("invalid_request_settings", "source_token")
    assert request_settings_error(
        {
            "request_settings_version": 1,
            "request_intent": "new",
            "source_token": "tok-stale",
        },
        current_source_token="tok-current",
        active_turn_id=None,
    ) == ("source_changed", "source_token")
    assert request_settings_error(
        {
            "request_settings_version": 1,
            "request_intent": "new",
        },
        current_source_token=None,
        active_turn_id=None,
    ) is None
    assert request_settings_error(
        {
            "request_settings_version": 1,
            "request_intent": "new",
            "source_token": "tok-any",
        },
        current_source_token=None,
        active_turn_id=None,
    ) == ("source_changed", "source_token")


def test_request_intent_target_and_confirmation_validation() -> None:
    from nanobot.webui.inbound_commands import request_settings_error

    def check(
        extra: dict[str, object],
        *,
        active_turn_id: str | None = "turn-1",
        current_source_token: str | None = None,
    ) -> tuple[str, str] | None:
        return request_settings_error(
            {"request_settings_version": 1, **extra},
            current_source_token=current_source_token,
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
    assert check({"request_intent": "new", "full_confirmation_id": 3}) == (
        "invalid_request_settings",
        "full_confirmation_id",
    )
    assert check(
        {
            "request_intent": "steer",
            "target_turn_id": "turn-1",
            "source_token": "tok-old",
        },
        current_source_token="tok-current",
    ) == ("source_changed", "source_token")
