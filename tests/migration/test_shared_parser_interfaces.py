"""Shared parser wrappers preserve algorithms and live owner/consumer seams."""

from importlib import metadata

import pytest

from nanobot import _migration_preflight as preflight
from nanobot.agent import policy
from nanobot.agent.tools import filesystem
from nanobot.channels import plugin, registry


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


@pytest.mark.parametrize(("content", "old_text"), [
    ("hello world", "world"), ("hello world", "absent"), ("hello", ""),
    ("aaa", "aa"), ("x x", "x"), ("  one\n    two\n", "one\ntwo"),
    ("  x\n x\n", "x\n"), ("   one\n   one\n", "one"),
])
def test_public_edit_matcher_retains_original_results(content, old_text):
    assert filesystem.find_edit_match(content, old_text) == filesystem._find_match(
        content, old_text,
    )


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


def test_edit_owner_matcher_patch_stays_live_after_policy_import(monkeypatch):
    calls = []

    def match(content, old_text):
        calls.append((content, old_text))
        return "actual matched fragment", 2

    monkeypatch.setattr(filesystem, "_find_match", match)
    assert policy._find_match("content", "old") == ("actual matched fragment", 2)
    assert filesystem.find_edit_match("other", "text") == ("actual matched fragment", 2)
    assert calls == [("content", "old"), ("other", "text")]


def test_edit_owner_exception_identity_reaches_policy(monkeypatch):
    failure = RuntimeError("real matcher fault")

    def fail(content, old_text):
        raise failure

    monkeypatch.setattr(filesystem, "_find_match", fail)
    with pytest.raises(RuntimeError) as raised:
        policy._find_match("content", "old")
    assert raised.value is failure


def test_consumer_local_parser_seams_remain_patchable(monkeypatch, tmp_path):
    target = tmp_path / "code.py"
    target.write_text("old\n", encoding="utf-8")
    matcher_calls = []

    def match(content, old_text):
        matcher_calls.append((content, old_text))
        return "old", 2

    monkeypatch.setattr(policy, "_find_match", match)
    decision = policy.RiskyActionPolicy(workspace=tmp_path)
    assert decision._risky_edit_reason("code.py", "old", "new", False) == "replace repeated content in code.py"
    assert matcher_calls == [("old\n", "old")]
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
