"""Strict newline-delimited JSON protocol used by the private desktop host pipe."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import BinaryIO, TypeAlias, cast

MAX_FRAME_BYTES = 65_536
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_OPS = frozenset({"hello", "start", "status", "bootstrap", "stop"})
Params: TypeAlias = dict[str, object]
Dispatch: TypeAlias = Callable[[str, Params], Mapping[str, object]]


class ProtocolError(ValueError):
    """Safe protocol refusal with an optional recoverable request identifier."""

    def __init__(self, code: str = "protocol_error", request_id: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.request_id = request_id


def _object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _params_valid(op: str, value: object) -> bool:
    if not isinstance(value, dict):
        return False
    params = cast(dict[str, object], value)
    if op == "hello":
        return (
            set(params) == {"config_path", "workspace"}
            and isinstance(params.get("config_path"), str)
            and len(cast(str, params["config_path"])) <= 4096
            and (
                params.get("workspace") is None
                or (
                    isinstance(params.get("workspace"), str)
                    and len(cast(str, params["workspace"])) <= 4096
                )
            )
        )
    if op == "start":
        return (
            not params
            or (
                set(params) == {"handle"}
                and isinstance(params.get("handle"), str)
                and 1 <= len(cast(str, params["handle"])) <= 2048
            )
        )
    return (
        set(params) == {"handle"}
        and isinstance(params.get("handle"), str)
        and 1 <= len(cast(str, params["handle"])) <= 2048
    )


def parse_request(frame: bytes) -> tuple[int, str, Params]:
    """Parse exactly one bounded v1 request frame and reject unknown fields."""
    if not frame.endswith(b"\n") or len(frame) > MAX_FRAME_BYTES:
        raise ProtocolError("frame_invalid")
    try:
        decoded: object = json.loads(
            frame[:-1].decode("utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ProtocolError("frame_invalid") from None
    if not isinstance(decoded, dict):
        raise ProtocolError("request_invalid")
    request = cast(dict[str, object], decoded)
    raw_request_id = request.get("id")
    request_id = (
        raw_request_id
        if type(raw_request_id) is int and 0 <= raw_request_id < 2**64
        else None
    )
    if set(request) != {"v", "id", "op", "params"}:
        raise ProtocolError("request_invalid", request_id)
    if type(request.get("v")) is not int or request.get("v") != 1:
        raise ProtocolError("protocol_version_unsupported", request_id)
    if request_id is None:
        raise ProtocolError("request_invalid")
    op = request.get("op")
    if not isinstance(op, str) or op not in _OPS:
        raise ProtocolError("operation_unsupported", request_id)
    params = request.get("params")
    if not _params_valid(op, params):
        raise ProtocolError("request_invalid", request_id)
    return request_id, op, cast(Params, params)


def _error_code(value: object) -> str:
    if isinstance(value, str) and _ERROR_CODE.fullmatch(value):
        return value
    return "internal_error"


def _write(output: BinaryIO, payload: Mapping[str, object]) -> None:
    try:
        frame = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        frame = ""
    encoded = (frame + "\n").encode("utf-8")
    if not frame or len(encoded) > MAX_FRAME_BYTES:
        encoded = (
            '{"v":1,"id":0,"ok":false,"error":{"code":"response_invalid"}}\n'
        ).encode("ascii")
    output.write(encoded)
    output.flush()


def serve(input_stream: BinaryIO, output_stream: BinaryIO, dispatch: Dispatch) -> None:
    """Serve sequential requests; malformed frames get a safe refusal then close."""
    while True:
        frame = input_stream.readline(MAX_FRAME_BYTES + 1)
        if not frame:
            return
        if len(frame) > MAX_FRAME_BYTES or not frame.endswith(b"\n"):
            _write(
                output_stream,
                {"v": 1, "id": 0, "ok": False, "error": {"code": "frame_invalid"}},
            )
            return
        try:
            request_id, op, params = parse_request(frame)
        except ProtocolError as exc:
            _write(
                output_stream,
                {
                    "v": 1,
                    "id": exc.request_id if exc.request_id is not None else 0,
                    "ok": False,
                    "error": {"code": _error_code(exc.code)},
                },
            )
            return
        try:
            result = dispatch(op, params)
            response: dict[str, object] = {
                "v": 1,
                "id": request_id,
                "ok": True,
                "result": dict(result),
            }
        except Exception as exc:
            response = {
                "v": 1,
                "id": request_id,
                "ok": False,
                "error": {"code": _error_code(getattr(exc, "code", None))},
            }
        _write(output_stream, response)
