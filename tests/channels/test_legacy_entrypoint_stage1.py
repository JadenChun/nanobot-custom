from __future__ import annotations

from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.channels.registry import discover_legacy_plugins
from nanobot.config.schema import Config


class ExternalChannel(BaseChannel):
    name = "external"
    display_name = "External"

    async def start(self):
        return None

    async def stop(self):
        return None

    async def send(self, msg: OutboundMessage):
        return None


def _entry_point(load):
    return SimpleNamespace(
        name="external",
        value=f"{__name__}:ExternalChannel",
        load=load,
    )


async def test_disabled_external_channel_is_never_enumerated_or_loaded(monkeypatch):
    monkeypatch.setattr(
        "nanobot.channels.registry.discover_plugins", lambda: {},
    )
    monkeypatch.setattr(
        "nanobot.channels.registry.metadata.entry_points",
        lambda **_kwargs: pytest.fail("disabled plugin metadata must not be queried"),
    )

    manager = ChannelManager(
        Config.model_validate({"channels": {"external": {"enabled": False}}}),
        MessageBus(),
    )
    await manager.acquire()

    assert manager.channels == {}


async def test_enabled_external_channel_is_loaded_lazily_after_metadata_preflight(monkeypatch):
    loads: list[str] = []
    entry_point = _entry_point(lambda: loads.append("load") or ExternalChannel)
    monkeypatch.setattr("nanobot.channels.registry.discover_plugins", lambda: {})
    monkeypatch.setattr(
        "nanobot.channels.registry.metadata.entry_points",
        lambda **_kwargs: [entry_point],
    )
    monkeypatch.setattr(
        "nanobot.optional_features.check_enabled_channel_dependencies",
        lambda _names, _plugins: {},
    )

    manager = ChannelManager(
        Config.model_validate({"channels": {"external": {"enabled": True}}}),
        MessageBus(),
    )
    assert loads == []
    await manager.acquire()

    assert loads == ["load"]
    assert isinstance(manager.channels["external"], ExternalChannel)


def test_legacy_entrypoint_duplicate_reserved_and_malformed_metadata_fail(monkeypatch):
    entry_point = _entry_point(lambda: ExternalChannel)
    monkeypatch.setattr(
        "nanobot.channels.registry.metadata.entry_points",
        lambda **_kwargs: [entry_point, entry_point],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        discover_legacy_plugins({"external"}, reserved_names=set())

    with pytest.raises(ValueError, match="shadow"):
        discover_legacy_plugins({"external"}, reserved_names={"external"})

    malformed = SimpleNamespace(name="external", value="not-a-target", load=lambda: None)
    monkeypatch.setattr(
        "nanobot.channels.registry.metadata.entry_points",
        lambda **_kwargs: [malformed],
    )
    with pytest.raises(ValueError, match="module:attribute"):
        discover_legacy_plugins({"external"}, reserved_names=set())


async def test_malformed_packaged_channel_name_stays_reserved(monkeypatch):
    loads: list[str] = []
    entry_point = SimpleNamespace(
        name="broken",
        value=f"{__name__}:ExternalChannel",
        load=lambda: loads.append("load") or ExternalChannel,
    )
    monkeypatch.setattr(
        "nanobot.channels.registry._channel_package_names",
        lambda: ["broken"],
    )
    monkeypatch.setattr(
        "nanobot.channels.registry.load_channel_package",
        lambda _name: (_ for _ in ()).throw(TypeError("malformed built-in manifest")),
    )
    monkeypatch.setattr(
        "nanobot.channels.registry.metadata.entry_points",
        lambda **_kwargs: [entry_point],
    )

    manager = ChannelManager(
            Config.model_validate({"channels": {"broken": {"enabled": True}}}),
            MessageBus(),
        )
    with pytest.raises(ValueError, match="cannot shadow built-in channels: broken"):
        await manager.acquire()

    assert loads == []
