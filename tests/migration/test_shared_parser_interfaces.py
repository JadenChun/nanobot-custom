"""Shared parser wrappers preserve algorithms and live owner/consumer seams."""

from importlib import metadata

import pytest

from kohaku import _migration_preflight as preflight
from kohaku.channels import plugin, registry


@pytest.mark.parametrize("target", ["module:factory", "pkg.module:Factory", "for:class"])
def test_public_channel_parser_retains_target_parts(target):
    assert plugin.parse_channel_target(target, label="connector") == plugin._target_parts(
        target, label="connector",
    )


@pytest.mark.parametrize("target", [
    "", "module", ":factory", "module:", ".module:factory", "pkg..module:factory",
    "pkg-module:factory", "module:fact.ory", "module:factory:other",
])
def test_public_channel_parser_retains_invalid_target_error(target):
    with pytest.raises(ValueError) as original:
        plugin._target_parts(target, label="connector")
    with pytest.raises(ValueError) as public:
        plugin.parse_channel_target(target, label="connector")
    assert type(public.value) is type(original.value)
    assert str(public.value) == str(original.value)


def test_channel_owner_parser_patch_stays_live_after_consumer_imports(monkeypatch):
    calls = []
    original = plugin._target_parts

    def observe(target, *, label):
        calls.append((target, label))
        return original(target, label=label)

    entry = metadata.EntryPoint(
        name="external", value="p6_missing_external_module:factory", group="nanobot.channels",
    )
    monkeypatch.setattr(plugin, "_target_parts", observe)
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_kwargs: [entry])
    monkeypatch.setattr(registry, "packaged_channel_names", lambda: frozenset())

    assert registry._target_parts(entry.value, label="runtime") == (
        "p6_missing_external_module", "factory",
    )
    assert registry.discover_legacy_plugins({"external"}, reserved_names=set())["external"].runtime == entry.value
    channels, dependencies = preflight._channels({"external": {"enabled": True}})
    assert channels.status == "PASS"
    assert dependencies.status == "UNQUALIFIED"
    assert len(calls) == 4
    assert calls == [(entry.value, "runtime")] * 4


def test_channel_owner_exception_identity_reaches_both_consumers(monkeypatch):
    failure = ValueError("real parser fault")

    def fail(target, *, label):
        raise failure

    entry = metadata.EntryPoint(
        name="external", value="p6_missing_external_module:factory", group="nanobot.channels",
    )
    monkeypatch.setattr(plugin, "_target_parts", fail)
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_kwargs: [entry])
    monkeypatch.setattr(registry, "packaged_channel_names", lambda: frozenset())
    with pytest.raises(ValueError) as registry_error:
        registry.discover_legacy_plugins({"external"}, reserved_names=set())
    with pytest.raises(ValueError) as preflight_error:
        preflight._channels({"external": {"enabled": True}})
    assert registry_error.value is failure
    assert preflight_error.value is failure


def test_consumer_local_channel_parser_seam_remains_patchable(monkeypatch):
    failure = ValueError("consumer parser fault")

    def fail(target, *, label):
        raise failure

    entry = metadata.EntryPoint(
        name="external", value="p6_missing_external_module:factory", group="nanobot.channels",
    )
    monkeypatch.setattr(registry, "_target_parts", fail)
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_kwargs: [entry])
    with pytest.raises(ValueError) as raised:
        registry.discover_legacy_plugins({"external"}, reserved_names=set())
    assert raised.value is failure
