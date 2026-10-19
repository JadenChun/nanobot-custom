"""Telegram-specific ChannelManager allowFrom startup policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.channels.plugin import load_channel_package
from nanobot.config.schema import Config


class _ConfigBackedChannel(BaseChannel):
    name = "fixture"
    display_name = "Fixture"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, msg: OutboundMessage) -> None:
        pass


def _manager_with_runtimes(
    channels: dict[str, BaseChannel], owners: dict[str, str]
) -> ChannelManager:
    manager = ChannelManager.__new__(ChannelManager)
    manager.channels = channels
    manager._channel_owners = owners
    return manager


async def _telegram_manager(
    monkeypatch: pytest.MonkeyPatch,
    section: dict[str, Any],
) -> ChannelManager:
    plugin = load_channel_package("telegram")
    assert plugin is not None
    monkeypatch.setattr(
        "nanobot.channels.registry.discover_plugins",
        lambda enabled_names=None: {"telegram": plugin},
    )
    monkeypatch.setattr(
        "nanobot.optional_features.check_enabled_channel_dependencies",
        lambda _enabled, _plugins: {},
    )
    config = Config.model_validate({"channels": {"telegram": section}})
    return await ChannelManager(config, MessageBus()).acquire()


def _config_with_telegram(section: dict[str, Any]) -> Config:
    return Config.model_validate({"channels": {"telegram": section}})


@pytest.mark.parametrize("allow_key", [None, "allowFrom", "allow_from"])
async def test_enabled_telegram_empty_effective_allow_from_fails_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    allow_key: str | None,
) -> None:
    section: dict[str, Any] = {"enabled": True, "token": "test-token"}
    if allow_key is not None:
        section[allow_key] = []

    with pytest.raises(SystemExit) as exc_info:
        await _telegram_manager(monkeypatch, section)

    message = str(exc_info.value)
    assert "Telegram" in message
    assert "allowFrom" in message
    assert "sender ID/username" in message
    assert '"*"' in message
    assert "Pairing does not replace sender authorization" in message


async def test_initial_authorization_error_precedes_dependency_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = load_channel_package("telegram")
    assert plugin is not None
    dependency_check_reached = False

    def fail_dependency_check(_enabled: set[str], _plugins: dict[str, Any]) -> dict[str, str]:
        nonlocal dependency_check_reached
        dependency_check_reached = True
        return {"telegram": "missing dependency"}

    monkeypatch.setattr(
        "nanobot.channels.registry.discover_plugins",
        lambda enabled_names=None: {"telegram": plugin},
    )
    monkeypatch.setattr(
        "nanobot.optional_features.check_enabled_channel_dependencies",
        fail_dependency_check,
    )

    with pytest.raises(SystemExit, match="Telegram allowFrom"):
        await ChannelManager(
            _config_with_telegram({"enabled": True, "token": "test-token"}),
            MessageBus(),
        ).acquire()

    assert dependency_check_reached is False


async def test_initial_authorization_error_precedes_runtime_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = load_channel_package("telegram")
    assert plugin is not None
    runtime_load_reached = False

    def fail_runtime_load(_plugin: Any) -> type[BaseChannel]:
        nonlocal runtime_load_reached
        runtime_load_reached = True
        raise RuntimeError("runtime load should not be reached")

    monkeypatch.setattr(
        "nanobot.channels.registry.discover_plugins",
        lambda enabled_names=None: {"telegram": plugin},
    )
    monkeypatch.setattr(
        "nanobot.optional_features.check_enabled_channel_dependencies",
        lambda _enabled, _plugins: {},
    )
    monkeypatch.setattr(type(plugin), "load_channel_class", fail_runtime_load)

    with pytest.raises(SystemExit, match="Telegram allowFrom"):
        await ChannelManager(
            _config_with_telegram({"enabled": True, "token": "test-token"}),
            MessageBus(),
        ).acquire()

    assert runtime_load_reached is False


@pytest.mark.parametrize("allow_from", [["42"], ["some_username"], ["*"]])
async def test_enabled_telegram_nonempty_allow_from_passes_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    allow_from: list[str],
) -> None:
    manager = await _telegram_manager(
        monkeypatch,
        {"enabled": True, "token": "test-token", "allowFrom": allow_from},
    )

    assert manager.channels["telegram"].config.allow_from == allow_from


async def test_disabled_telegram_empty_allow_from_passes_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = await _telegram_manager(
        monkeypatch,
        {"enabled": False, "token": "test-token", "allowFrom": []},
    )

    assert "telegram" not in manager.channels


@pytest.mark.parametrize("allow_key", [None, "allowFrom", "allow_from"])
async def test_hot_enable_rejects_empty_effective_allow_from_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    allow_key: str | None,
) -> None:
    manager = await _telegram_manager(
        monkeypatch,
        {"enabled": False, "token": "test-token", "allowFrom": []},
    )
    original_config = manager.config
    proposed: dict[str, Any] = {"enabled": True, "token": "test-token"}
    if allow_key is not None:
        proposed[allow_key] = []
    monkeypatch.setattr(
        "nanobot.config.loader.load_config",
        lambda: _config_with_telegram(proposed),
    )

    with pytest.raises(SystemExit, match="Telegram allowFrom"):
        await manager.apply_channel_feature_action("enable", "telegram")

    assert manager.config is original_config
    assert manager.channels == {}
    assert manager._channel_owners == {}
    assert manager._channel_runtime_specs == {}


async def test_invalid_hot_update_preserves_valid_running_telegram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = await _telegram_manager(
        monkeypatch,
        {"enabled": True, "token": "test-token", "allowFrom": ["42"]},
    )
    original_config = manager.config
    original_channel = manager.channels["telegram"]
    original_channel._running = True
    original_specs = dict(manager._channel_runtime_specs)
    monkeypatch.setattr(
        "nanobot.config.loader.load_config",
        lambda: _config_with_telegram(
            {"enabled": True, "token": "replacement-token", "allowFrom": []}
        ),
    )

    with pytest.raises(SystemExit, match="Telegram allowFrom"):
        await manager.apply_channel_feature_action("enable", "telegram")

    assert manager.config is original_config
    assert manager.channels == {"telegram": original_channel}
    assert manager._channel_owners == {"telegram": "telegram"}
    assert manager._channel_runtime_specs == original_specs
    assert original_channel.is_running is True
    assert original_channel.config.token == "test-token"
    assert original_channel.config.allow_from == ["42"]


async def test_valid_hot_enable_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = await _telegram_manager(
        monkeypatch,
        {"enabled": False, "token": "test-token", "allowFrom": []},
    )
    monkeypatch.setattr(
        "nanobot.config.loader.load_config",
        lambda: _config_with_telegram(
            {"enabled": True, "token": "test-token", "allowFrom": ["42"]}
        ),
    )

    result = await manager.apply_channel_feature_action("enable", "telegram")

    assert result["ok"] is True
    assert result["requires_restart"] is False
    assert manager.channels["telegram"].config.allow_from == ["42"]


def test_dict_backed_telegram_runtimes_use_owner_and_snake_case() -> None:
    primary = _ConfigBackedChannel(
        {"enabled": True, "allow_from": ["42"]}, MessageBus()
    )
    secondary = _ConfigBackedChannel({"enabled": True, "allow_from": []}, MessageBus())
    manager = _manager_with_runtimes(
        {"telegram": primary, "telegram.secondary": secondary},
        {"telegram": "telegram", "telegram.secondary": "telegram"},
    )

    with pytest.raises(SystemExit, match="Telegram"):
        manager._validate_allow_from()


def test_non_telegram_empty_or_missing_allow_from_policy_is_unchanged() -> None:
    channels = {
        "other.empty": _ConfigBackedChannel(
            {"enabled": True, "allowFrom": []}, MessageBus()
        ),
        "other.missing": _ConfigBackedChannel({"enabled": True}, MessageBus()),
    }
    manager = _manager_with_runtimes(
        channels,
        {"other.empty": "other", "other.missing": "other"},
    )

    manager._validate_allow_from()


def test_disabled_dict_backed_telegram_empty_allow_from_passes() -> None:
    channel = _ConfigBackedChannel({"enabled": False, "allowFrom": []}, MessageBus())
    manager = _manager_with_runtimes({"telegram": channel}, {"telegram": "telegram"})

    manager._validate_allow_from()


def test_all_telegram_locale_help_requires_authorization_before_pairing() -> None:
    locale_dir = (
        Path(__file__).parents[2] / "nanobot/channels/telegram/webui/locales"
    )
    expected_help = (
        'Required: add at least one Telegram sender ID/username or "*". '
        "Pairing is additional and does not replace sender authorization."
    )

    locale_keys: set[str] | None = None
    for path in sorted(locale_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        help_text = data["setup"]["fields"]["allowFrom"]["help"]
        assert help_text == expected_help, path.name
        current_keys = set(data["setup"]["fields"])
        if locale_keys is None:
            locale_keys = current_keys
        else:
            assert current_keys == locale_keys, path.name
