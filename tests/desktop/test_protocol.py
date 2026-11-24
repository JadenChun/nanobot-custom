import io
import json

import pytest

from kohaku.desktop.protocol import MAX_FRAME_BYTES, ProtocolError, parse_request, serve


def test_parse_request_rejects_unknown_fields_and_boolean_ids() -> None:
    with pytest.raises(ProtocolError):
        parse_request(b'{"v":1,"id":true,"op":"hello","params":{}}\n')

    with pytest.raises(ProtocolError):
        parse_request(b'{"v":1,"id":1,"op":"hello","params":{},"extra":1}\n')


def test_parse_request_rejects_oversized_or_truncated_frames() -> None:
    with pytest.raises(ProtocolError):
        parse_request(b" " * MAX_FRAME_BYTES + b"\n")

    with pytest.raises(ProtocolError):
        parse_request(b'{"v":1,"id":1,"op":"hello","params":{}}')


def test_protocol_serves_one_strict_request_per_line() -> None:
    incoming = io.BytesIO(b'{"v":1,"id":7,"op":"start","params":{}}\n')
    outgoing = io.BytesIO()

    serve(incoming, outgoing, lambda op, params: {"op": op, "empty": not params})

    assert json.loads(outgoing.getvalue()) == {
        "v": 1,
        "id": 7,
        "ok": True,
        "result": {"op": "start", "empty": True},
    }


def test_prepare_default_config_cannot_request_an_arbitrary_path() -> None:
    frame = b'{"v":1,"id":7,"op":"prepare_default_config","params":{}}\n'
    assert parse_request(frame) == (7, "prepare_default_config", {})
    with pytest.raises(ProtocolError):
        parse_request(frame.replace(b'"params":{}', b'"params":{"config_path":"/tmp/other"}'))


def test_connect_foreground_takes_no_alternate_target() -> None:
    frame = b'{"v":1,"id":7,"op":"connect_foreground","params":{}}\n'
    assert parse_request(frame) == (7, "connect_foreground", {})
    with pytest.raises(ProtocolError):
        parse_request(frame.replace(b'"params":{}', b'"params":{"url":"http://127.0.0.1:9000"}'))

