"""Pure preservation probes for the retained broker/browser wire boundary."""

from __future__ import annotations

import json

import pytest

from nanobot.agent.tools import social_crawl as crawl
from nanobot.agent.tools import trend_vpn as vpn
from nanobot.agent.turn import ToolOutcome


class _Socket:
    def __init__(self, response):
        self.response = response
        self.trace = []
        self.sent = None

    def __enter__(self):
        self.trace.append("enter")
        return self

    def __exit__(self, *_):
        self.trace.append("close")

    def settimeout(self, timeout):
        self.trace.append(("timeout", timeout))

    def connect(self, path):
        self.trace.append(("connect", path))

    def sendall(self, payload):
        self.sent = payload
        self.trace.append("send")

    def recv(self, limit):
        self.trace.append(("recv", limit))
        chunk, self.response = self.response[:limit], self.response[limit:]
        return chunk


def _response(kind, line, monkeypatch):
    if kind == "crawl":
        return crawl.SocialCrawlClient._decode_response(line)
    connection = _Socket(line)
    monkeypatch.setattr(vpn.socket, "socket", lambda *_: connection)
    return vpn.TrendVpnBrokerClient(socket_path="/synthetic/broker.sock").request("fetch")


def _error_type(kind):
    return crawl.SocialCrawlError if kind == "crawl" else vpn.TrendVpnBrokerError


@pytest.mark.parametrize("kind", ["vpn", "crawl"])
def test_decoded_response_preserves_identity_and_unvalidated_extra_values(kind, monkeypatch):
    nested = [1, None, {"arbitrary": False}]
    answer = {"ok": True, "extra": nested}
    monkeypatch.setattr(json, "loads", lambda _: answer)
    result = _response(kind, b'{"ok":true}\n', monkeypatch)
    assert result is answer and result["extra"] is nested


@pytest.mark.parametrize("kind", ["vpn", "crawl"])
@pytest.mark.parametrize("line, suffix", [
    (b"", "no response"),
    (b"\xff", "invalid response"),
    (b"{", "invalid response"),
    (b"[]", "non-object response"),
    (b"null", "non-object response"),
    (b"42", "non-object response"),
    (b'"text"', "non-object response"),
])
def test_decoder_refusals_preserve_exact_error_and_suppressed_cause(kind, line, suffix, monkeypatch):
    with pytest.raises(_error_type(kind)) as raised:
        _response(kind, line, monkeypatch)
    owner = "VPN broker" if kind == "vpn" else "crawl worker"
    verb = "returned" if suffix != "no response" else "returned"
    article = "an " if suffix == "invalid response" else "a " if suffix == "non-object response" else ""
    assert str(raised.value) == f"{owner} {verb} {article}{suffix}"
    if suffix == "invalid response":
        assert isinstance(raised.value.__cause__, (UnicodeDecodeError, json.JSONDecodeError))
    else:
        assert raised.value.__cause__ is None


@pytest.mark.parametrize("kind", ["vpn", "crawl"])
@pytest.mark.parametrize("ok", [None, False, 0, 1, "true", [], {}])
def test_success_requires_true_identity_not_truthiness(kind, ok, monkeypatch):
    line = json.dumps({"ok": ok}).encode()
    with pytest.raises(_error_type(kind)) as raised:
        _response(kind, line, monkeypatch)
    assert str(raised.value) == ("broker operation failed" if kind == "vpn" else "crawl operation failed")


@pytest.mark.parametrize("kind", ["vpn", "crawl"])
def test_missing_success_field_is_not_adopted(kind, monkeypatch):
    with pytest.raises(_error_type(kind)):
        _response(kind, b"{}", monkeypatch)


@pytest.mark.parametrize("detail, expected", [
    ({}, "broker operation failed"),
    ({"message": None}, "None"),
    ({"message": 17}, "17"),
    ({"message": ""}, ""),
    ({"message": "\u00e9" * 500}, "\u00e9" * 400),
    (None, "broker operation failed"),
    ("raw error is not a message record", "broker operation failed"),
    ([{"message": "not a dictionary"}], "broker operation failed"),
])
def test_vpn_nested_error_guard_defaults_and_character_truncation(detail, expected, monkeypatch):
    with pytest.raises(vpn.TrendVpnBrokerError) as raised:
        _response("vpn", json.dumps({"ok": False, "error": detail}).encode(), monkeypatch)
    assert str(raised.value) == expected


@pytest.mark.parametrize("detail", [None, 17, "", "\u00e9" * 500, {"message": "nested"}])
def test_crawl_error_remains_direct_stringification(detail, monkeypatch):
    with pytest.raises(crawl.SocialCrawlError) as raised:
        _response("crawl", json.dumps({"ok": False, "error": detail}).encode(), monkeypatch)
    assert str(raised.value) == str(detail)[:400]


@pytest.mark.parametrize("kind", ["vpn", "crawl"])
@pytest.mark.parametrize("fault", [RuntimeError("ordinary"), KeyboardInterrupt("system")])
def test_untranslated_decoder_fault_identity(kind, fault, monkeypatch):
    def decode(_):
        raise fault

    monkeypatch.setattr(json, "loads", decode)
    with pytest.raises(type(fault)) as raised:
        _response(kind, b"{}", monkeypatch)
    assert raised.value is fault


@pytest.mark.parametrize("kind", ["vpn", "crawl"])
@pytest.mark.parametrize("fault", [RuntimeError("ordinary"), KeyboardInterrupt("system")])
@pytest.mark.parametrize("point", ["ok", "error", "message", "string"])
def test_live_get_and_stringification_fault_order(kind, fault, point, monkeypatch):
    trace = []

    class Message:
        def __str__(self):
            trace.append("string")
            if point == "string":
                raise fault
            return "changed"

    class Detail(dict):
        def get(self, key, default=None):
            trace.append("message")
            if point == "message":
                raise fault
            return super().get(key, default)

        def __str__(self):
            return str(self.get("message"))

    detail = Detail(message=Message())

    class Answer(dict):
        def get(self, key, default=None):
            trace.append(key)
            if point == key:
                raise fault
            if key == "ok":
                self["error"] = detail
            return super().get(key, default)

    answer = Answer(ok=False, error="stale")
    monkeypatch.setattr(json, "loads", lambda _: answer)
    with pytest.raises(type(fault)) as raised:
        _response(kind, b"{}", monkeypatch)
    assert raised.value is fault
    order = ["ok", "error", "message", "string"]
    assert trace == order[:order.index(point) + 1]


@pytest.mark.parametrize("kind", ["vpn", "crawl"])
def test_vpn_first_line_and_crawl_explicit_line_success(kind, monkeypatch):
    line = b'{"ok":true,"payload":{"unknown":[null,1]}}'
    supplied = line + b"\nnot-json" if kind == "vpn" else line
    assert _response(kind, supplied, monkeypatch) == {"ok": True, "payload": {"unknown": [None, 1]}}


@pytest.mark.parametrize("action, expected", [
    (None, "state_changing"), (0, "state_changing"), (False, "state_changing"),
    ([], "state_changing"), ({}, "state_changing"), (object(), "state_changing"),
    ("", "state_changing"), ("unknown", "state_changing"), ("click", "state_changing"),
    ("submit", "state_changing"), ("type", "state_changing"),
    ("  OpEn  ", "read_navigation"), ("\tNAVIGATE\n", "read_navigation"),
])
def test_raw_classifier_preserves_existing_refusal_and_normalization(action, expected):
    assert crawl.classify_crawler_action(action) == expected


def test_classifier_string_subclass_effect_order_and_nonstring_no_lookup():
    trace = []

    class Action(str):
        def strip(self):
            trace.append("strip")
            return self

        def lower(self):
            trace.append("lower")
            return "open"

    class NonString:
        def __getattribute__(self, name):
            if name in {"strip", "lower"}:
                raise AssertionError("nonstring must not be normalized")
            return super().__getattribute__(name)

    assert crawl.classify_crawler_action(Action("OPEN")) == "read_navigation"
    assert trace == ["strip", "lower"]
    assert crawl.classify_crawler_action(NonString()) == "state_changing"
    assert trace == ["strip", "lower"]


def test_sync_click_refusal_precedes_transport(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("click reached transport")

    monkeypatch.setattr(crawl.socket, "socket", forbidden)
    with pytest.raises(crawl.SocialCrawlError, match="click is disabled for all crawler callers"):
        crawl.SocialCrawlClient(socket_path="/synthetic/worker.sock").request(" CLICK ")


@pytest.mark.asyncio
async def test_async_click_refusal_precedes_transport(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("click reached async transport")

    monkeypatch.setattr(crawl.asyncio, "open_unix_connection", forbidden)
    with pytest.raises(crawl.SocialCrawlError, match="click is disabled for all crawler callers"):
        await crawl.SocialCrawlClient(socket_path="/synthetic/worker.sock").request_async(" CLICK ")


@pytest.mark.parametrize("op", [None, 0, False])
def test_sync_nonstring_serializable_op_keeps_original_wire_behavior(op, monkeypatch):
    connection = _Socket(b'{"ok":true}\n')
    monkeypatch.setattr(crawl.socket, "socket", lambda *_: connection)
    assert crawl.SocialCrawlClient(socket_path="/synthetic/worker.sock").request(op) == {"ok": True}
    assert json.loads(connection.sent) == {"op": op}
    assert connection.trace[-1] == "close"


@pytest.mark.asyncio
@pytest.mark.parametrize("op", [None, 0, False])
async def test_async_nonstring_op_keeps_wire_and_cleanup_order(op, monkeypatch):
    trace = []
    sent = []

    class Reader:
        async def readline(self):
            trace.append("read")
            return b'{"ok":true}\n'

    class Writer:
        def write(self, data):
            sent.append(data)
            trace.append("write")

        async def drain(self):
            trace.append("drain")

        def close(self):
            trace.append("close")

        async def wait_closed(self):
            trace.append("wait_closed")

    async def connect(*args, **kwargs):
        trace.append("connect")
        return Reader(), Writer()

    monkeypatch.setattr(crawl.asyncio, "open_unix_connection", connect)
    assert await crawl.SocialCrawlClient(socket_path="/synthetic/worker.sock").request_async(op) == {"ok": True}
    assert json.loads(sent[0]) == {"op": op}
    assert trace == ["connect", "write", "drain", "read", "close", "wait_closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [None, {}, "unknown"])
async def test_contextual_classifier_raw_input_keeps_policy_envelope(action):
    outcome = await crawl.SocialCrawlTool().execute_with_context(None, action=action)
    assert isinstance(outcome, ToolOutcome)
    assert outcome.stop_reason == "policy_blocked"
    assert outcome.policy_metadata == {
        "policy": "delegated_read_only", "action_class": "state_changing", "requires_approval": False,
    }
