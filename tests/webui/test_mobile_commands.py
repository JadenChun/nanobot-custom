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
