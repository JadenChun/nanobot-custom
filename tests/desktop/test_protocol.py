import io
import json

import pytest

from nanobot.desktop.protocol import MAX_FRAME_BYTES, ProtocolError, parse_request, serve


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

