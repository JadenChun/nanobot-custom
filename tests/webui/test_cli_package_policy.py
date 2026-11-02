from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from nanobot.webui.settings_contracts import SettingsRequest
from nanobot.webui.settings_system import SystemSettingsHandler


class _Logger:
    def debug(self, *_a: Any, **_k: Any) -> None: ...
    def info(self, *_a: Any, **_k: Any) -> None: ...
    def warning(self, *_a: Any, **_k: Any) -> None: ...
    def exception(self, *_a: Any, **_k: Any) -> None: ...


def _handler(allow_remote: bool) -> SystemSettingsHandler:
    config = SimpleNamespace(tools=SimpleNamespace(webui_allow_remote_package_install=allow_remote))
    settings = SimpleNamespace(
        config=SimpleNamespace(path=None, load=lambda: config),
    )
    return SystemSettingsHandler(settings, _Logger())  # type: ignore[arg-type]


def _request(local_browser: bool) -> SettingsRequest:
    return SettingsRequest(
        query={"name": ["example"]},
        payload={"name": "example"},
        local_browser=local_browser,
    )


def test_remote_owner_cannot_change_cli_packages_by_default() -> None:
    handler = _handler(allow_remote=False)
    for action in ("cli-install", "cli-update", "cli-uninstall"):
        result = asyncio.run(handler.handle(action, _request(False), SimpleNamespace()))
        assert result.error == "remote_package_install_denied"
        assert result.status == 403


def test_local_browser_may_change_cli_packages() -> None:
    handler = _handler(allow_remote=False)
    # Local browser bypasses the policy and reaches the real operation stub.
    operations = SimpleNamespace(
        cli_apps_action=lambda action, query, *, config_path: {"action": action}
    )
    result = asyncio.run(handler.handle("cli-install", _request(True), operations))
    assert result.error is None
    assert result.payload == {"action": "install"}


def test_remote_install_allowed_by_explicit_policy() -> None:
    handler = _handler(allow_remote=True)
    operations = SimpleNamespace(
        cli_apps_action=lambda action, query, *, config_path: {"action": action}
    )
    result = asyncio.run(handler.handle("cli-install", _request(False), operations))
    assert result.error is None
