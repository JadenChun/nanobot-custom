"""Preserve existing checks, identity and faults at decoded channel/CLI edges."""

from __future__ import annotations

import json
from collections import UserDict
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import typer

import nanobot.cli.cron_delivery as cron_cli
import nanobot.cron.delivery_reconciliation as reconciliation
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.cli.gateway_runtime import _gateway_readiness_payload
from nanobot.config.loader import _migrate_config
from nanobot.config.schema import Config


@pytest.mark.parametrize("raw", [None, False, 1, "text", [], UserDict()])
@pytest.mark.parametrize("enabled", [False, True])
def test_readiness_retains_non_dict_status_fallback(raw, enabled):
    channels = SimpleNamespace(
        get_status=lambda: raw, enabled_channels=["websocket"] if enabled else [],
    )
    ready, payload = _gateway_readiness_payload(channels)
    assert ready is not enabled
    assert payload == {
        "status": "degraded" if enabled else "ok", "process": "alive",
        "ready": not enabled, "websocket": "unavailable" if enabled else "disabled",
    }


@pytest.mark.parametrize("value", [False, 0, "text", [], UserDict(), object()])
def test_readiness_retains_present_non_dict_websocket_requirement(value):
    class Channels:
        def get_status(self):
            return {"websocket": value}

        @property
        def enabled_channels(self):
            pytest.fail("a present websocket must short-circuit enabled-channel lookup")

    assert _gateway_readiness_payload(Channels()) == (
        False, {"status": "degraded", "process": "alive", "ready": False,
                "websocket": "unavailable"},
    )


@pytest.mark.parametrize("running,ready", [(True, True), (1, False), ("true", False),
                                           (False, False), (None, False)])
@pytest.mark.parametrize("state", [None, "", 0, "starting"])
def test_readiness_preserves_guarded_receivers_two_reads_and_strict_true(running, ready, state):
    events = []

    class Status(dict):
        def get(self, key, default=None):
            events.append((self, key))
            return super().get(key, default)

    websocket = Status(running=running, state=state)
    top = Status(websocket=websocket)
    channels = SimpleNamespace(get_status=lambda: top)
    result, payload = _gateway_readiness_payload(channels)
    assert result is ready
    assert payload["ready"] is ready
    assert payload["status"] == ("ok" if ready else "degraded")
    assert payload["websocket"] == (str(state) if state else "unavailable")
    assert [key for _receiver, key in events] == ["websocket", "running", "state"]
    assert events[0][0] is top
    assert events[1][0] is websocket and events[2][0] is websocket


@pytest.mark.parametrize("key", ["running", "state"])
@pytest.mark.parametrize("error_type", [TypeError, RuntimeError])
def test_readiness_keeps_native_get_failure_identity_and_phase(key, error_type):
    failure = error_type("status lookup fault")
    events = []

    class Status(dict):
        def get(self, name, default=None):
            events.append(name)
            if name == key:
                raise failure
            return True

    channels = SimpleNamespace(get_status=lambda: {"websocket": Status()})
    with pytest.raises(error_type) as caught:
        _gateway_readiness_payload(channels)
    assert caught.value is failure
    assert events == (["running"] if key == "running" else ["running", "state"])


def test_readiness_keeps_state_coercion_failure_after_the_second_read():
    failure = RuntimeError("state conversion fault")
    events = []

    class State:
        def __bool__(self):
            events.append("bool")
            return True

        def __str__(self):
            events.append("str")
            raise failure

    class Status(dict):
        def get(self, key, default=None):
            events.append(key)
            return True if key == "running" else State()

    with pytest.raises(RuntimeError) as caught:
        _gateway_readiness_payload(SimpleNamespace(get_status=lambda: {"websocket": Status()}))
    assert caught.value is failure
    assert events == ["running", "state", "bool", "str"]


@pytest.mark.parametrize("servers", [None, False, 1, "text", [], UserDict()])
def test_config_migration_retains_non_dict_server_refusal_and_input_identity(servers):
    tools = {"mcpServers": servers}
    data = {"tools": tools}
    assert _migrate_config(data) is data
    assert data["tools"] is tools and tools["mcpServers"] is servers


@pytest.mark.parametrize("existing", [None, "toolTimeout", "tool_timeout"])
@pytest.mark.parametrize("server_key", ["mcpServers", "mcp_servers"])
def test_config_timeout_retains_single_lookup_alias_short_circuit_and_exact_mutation(
    existing, server_key,
):
    events = []
    sentinel = object()

    class Desktop(dict):
        def __contains__(self, key):
            events.append(("contains", key))
            return super().__contains__(key)

        def __setitem__(self, key, value):
            events.append(("set", key, value))
            super().__setitem__(key, value)

    class Servers(dict):
        def get(self, key, default=None):
            events.append(("get", key))
            return super().get(key, default)

    desktop = Desktop({existing: sentinel} if existing else {})
    opaque_key = object()
    servers = Servers({"ltx-desktop": desktop, opaque_key: sentinel})
    data = {"tools": {server_key: servers}}
    assert _migrate_config(data) is data
    assert data["tools"][server_key] is servers
    assert dict.__getitem__(servers, "ltx-desktop") is desktop
    assert dict.__getitem__(servers, opaque_key) is sentinel
    if existing is None:
        assert desktop == {"toolTimeout": 120}
        assert events == [("get", "ltx-desktop"), ("contains", "toolTimeout"),
                          ("contains", "tool_timeout"), ("set", "toolTimeout", 120)]
    else:
        assert desktop == {existing: sentinel}
        assert dict.__getitem__(desktop, existing) is sentinel
        assert events == [("get", "ltx-desktop"), ("contains", "toolTimeout"),
                          *([] if existing == "toolTimeout" else [("contains", "tool_timeout")])]


@pytest.mark.parametrize("phase", ["get", "set"])
def test_config_timeout_keeps_native_fault_identity_before_later_migrations(phase):
    failure = RuntimeError("timeout migration fault")
    events = []

    class Desktop(dict):
        def __setitem__(self, key, value):
            events.append("set")
            raise failure

    class Servers(dict):
        def get(self, key, default=None):
            events.append("get")
            if phase == "get":
                raise failure
            return desktop

    desktop = Desktop()
    tools = {"mcpServers": Servers(), "myEnabled": True}
    with pytest.raises(RuntimeError) as caught:
        _migrate_config({"tools": tools})
    assert caught.value is failure
    assert events == (["get"] if phase == "get" else ["get", "set"])
    assert tools["myEnabled"] is True and "my" not in tools


@pytest.mark.parametrize("mapping_type", [dict, UserDict])
@pytest.mark.parametrize("enabled,selected", [(True, True), (1, False), ("true", False),
                                             (False, False), (None, False)])
async def test_legacy_channel_selection_preserves_mapping_receiver_and_strict_true(
    monkeypatch, mapping_type, enabled, selected,
):
    events = []
    failure = LookupError("selection boundary")

    class Section(mapping_type):
        def get(self, key, default=None):
            events.append((self, key))
            return super().get(key, default)

    class Poison(Mapping):
        def __iter__(self):
            pytest.fail("shared channel field was iterated")

        def __len__(self):
            return 0

        def __getitem__(self, key):
            pytest.fail("shared channel field was read")

    config = Config()
    section = Section(enabled=enabled)
    config.channels.model_extra.update({"external": section, "send_progress": Poison()})
    monkeypatch.setattr("nanobot.channels.registry.packaged_channel_names", lambda: [])
    monkeypatch.setattr("nanobot.channels.registry.discover_plugins", lambda: {})
    captured = []

    def capture(names, *, reserved_names):
        captured.append((names, reserved_names))
        raise failure

    monkeypatch.setattr("nanobot.channels.registry.discover_legacy_plugins", capture)
    manager = ChannelManager(config, MessageBus())
    with pytest.raises(LookupError) as caught:
        await manager.acquire()
    assert caught.value is failure
    assert captured == [({"external"} if selected else set(), set())]
    assert len(events) == 1 and events[0][0] is section and events[0][1] == "enabled"
    assert manager.channels == {} and manager._acquiring is False


@pytest.mark.parametrize("section", [None, False, 1, "text", []])
async def test_legacy_channel_selection_does_not_promote_non_mapping_sections(monkeypatch, section):
    config = Config()
    config.channels.model_extra["external"] = section
    monkeypatch.setattr("nanobot.channels.registry.packaged_channel_names", lambda: [])
    monkeypatch.setattr("nanobot.channels.registry.discover_plugins", lambda: {})
    captured = []

    def discover(names, *, reserved_names):
        captured.append((names, reserved_names))
        return {}

    monkeypatch.setattr("nanobot.channels.registry.discover_legacy_plugins", discover)
    monkeypatch.setattr("nanobot.optional_features.check_enabled_channel_dependencies",
                        lambda _names, _plugins: {})
    manager = ChannelManager(config, MessageBus())
    assert await manager.acquire() is manager
    assert captured == [(set(), set())] and manager.channels == {}


async def test_legacy_channel_selection_keeps_mapping_fault_before_discovery(monkeypatch):
    failure = RuntimeError("enabled lookup fault")

    class Section(dict):
        def get(self, key, default=None):
            assert key == "enabled"
            raise failure

    config = Config()
    config.channels.model_extra["external"] = Section()
    monkeypatch.setattr("nanobot.channels.registry.packaged_channel_names", lambda: [])
    monkeypatch.setattr("nanobot.channels.registry.discover_plugins", lambda: {})
    monkeypatch.setattr("nanobot.channels.registry.discover_legacy_plugins",
                        lambda *_args, **_kwargs: pytest.fail("fault reached discovery"))
    manager = ChannelManager(config, MessageBus())
    with pytest.raises(RuntimeError) as caught:
        await manager.acquire()
    assert caught.value is failure
    assert manager.channels == {} and manager._acquiring is False


def _invoke_receipt_import(tmp_path, *, receipt_file=True, outcome="confirmed_delivered"):
    cron_cli.evidence_import_command(
        outbox_root=tmp_path / "outbox", namespace="cron:payload",
        artifacts_root=tmp_path / "artifacts", outbox_id="a" * 64,
        destination_id="b" * 64, outcome=outcome,
        receipt_file=tmp_path / "receipt.json" if receipt_file else None, json_output=True,
    )


@pytest.mark.parametrize("encoded", [None, b'{}', b'{"value":[1,null,{"text":"ok"}]}'])
def test_receipt_import_keeps_decoded_identity_or_none_at_the_owner_boundary(
    tmp_path, monkeypatch, encoded,
):
    decoded, received, emitted = [], [], []
    loads = json.loads

    def decode(value, **kwargs):
        result = loads(value, **kwargs)
        decoded.append(result)
        return result

    def owner(*args, **kwargs):
        received.append((args, kwargs))
        return SimpleNamespace(reference="sha256:" + "c" * 64, kind="evidence")

    monkeypatch.setattr(cron_cli, "_read_private_receipt_file", lambda _path: encoded)
    monkeypatch.setattr(cron_cli.json, "loads", decode)
    monkeypatch.setattr(reconciliation, "import_evidence", owner)
    monkeypatch.setattr(cron_cli, "_emit", lambda value, json_output: emitted.append((value, json_output)))
    _invoke_receipt_import(tmp_path, receipt_file=encoded is not None)
    assert len(received) == 1
    assert received[0][0] == (tmp_path / "outbox",)
    receipt = received[0][1]["receipt"]
    if encoded is None:
        assert receipt is None and decoded == []
    else:
        assert len(decoded) == 1 and receipt is decoded[0]
        assert receipt == loads(encoded)
    assert received[0][1]["namespace"] == "cron:payload"
    assert received[0][1]["outcome"] == "confirmed_delivered"
    assert emitted == [({"artifact_ref": "sha256:" + "c" * 64, "kind": "evidence",
                        "schema": "nanobot.cron_delivery_reconciliation_import.v1"}, True)]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("encoded", [b'[]', b'null', b'"text"', b'1', b'false',
                                     b'{"a":1,"a":2}', b'{"a":{"b":1,"b":2}}',
                                     b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}',
                                     b'\xff', b'{'])
def test_receipt_import_keeps_strict_parser_refusal_before_owner_without_disclosure(
    tmp_path, monkeypatch, capsys, encoded,
):
    monkeypatch.setattr(cron_cli, "_read_private_receipt_file", lambda _path: encoded)
    monkeypatch.setattr(reconciliation, "import_evidence",
                        lambda *_args, **_kwargs: pytest.fail("invalid receipt reached owner"))
    with pytest.raises(typer.Exit) as caught:
        _invoke_receipt_import(tmp_path)
    assert caught.value.exit_code == 5
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: reconciliation authority or artifact is malformed or unsafe.\n"
    assert list(tmp_path.iterdir()) == []


def test_receipt_import_retains_outcome_refusal_after_read_before_owner(tmp_path, monkeypatch, capsys):
    events = []

    def read(_path):
        events.append("read")
        return b'{}'

    monkeypatch.setattr(cron_cli, "_read_private_receipt_file", read)
    monkeypatch.setattr(reconciliation, "import_evidence",
                        lambda *_args, **_kwargs: pytest.fail("invalid outcome reached owner"))
    with pytest.raises(typer.Exit) as caught:
        _invoke_receipt_import(tmp_path, outcome="unsupported")
    assert caught.value.exit_code == 5 and events == ["read"]
    assert capsys.readouterr().err == (
        "Error: reconciliation authority or artifact is malformed or unsafe.\n"
    )
