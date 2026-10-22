"""Private, pure, bounded two-plane diagnostics; never an execution authority.

Inputs are already-loaded detached public evidence. Root history, not a checkpoint
summary, is compared. Immutable bytes freeze the boundary; routine repr/reports
carry no content or digest. Only explicit private operator comparison has digests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from nanobot.runtime_context import public_history_message
from nanobot.session.source import (
    SourceValidationError,
    validate_event_ids,
    validate_execution_links,
)
from nanobot.session.source_projection import project_source
from nanobot.session.source_store import SourceSnapshot

if TYPE_CHECKING:
    from typing import NotRequired

    class _Function(TypedDict):
        name: str
        arguments: str

    class _Call(TypedDict):
        id: str
        type: Literal["function"]
        function: _Function

    class _Message(TypedDict):
        role: str
        content: NotRequired[str | None]
        tool_calls: NotRequired[list[_Call]]
        tool_call_id: NotRequired[str]
        name: NotRequired[str]


Result = Literal["equal", "public_history", "tool_pairing", "archive_boundary",
                 "unsupported", "invalid_input", "bound_exceeded"]
MAX_BYTES = 262144
MAX_MESSAGES = 512
MAX_DEPTH = 16
MAX_TOOL_CALLS = 128
MAX_COUNT = 1000000
_RESULTS = frozenset({"equal", "public_history", "tool_pairing", "archive_boundary",
                      "unsupported", "invalid_input", "bound_exceeded"})
_COMPONENTS = frozenset({"shadow", "admission", "session", "conversion", "scheduler",
                         "occurrence", "outbox", "callback", "channel", "startup", "shutdown"})
_PUBLIC_FIELDS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name",
                            "_runtime_context"})


class DiagnosticPayload(TypedDict):
    component: str
    result: str
    left_count: int
    right_count: int


@dataclass(frozen=True, slots=True)
class PublicEvidence:
    result: Result
    count: int = 0
    _wire: bytes = field(default=b"", repr=False)
    _pairing: bytes = field(default=b"", repr=False)
    _boundary: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if (type(self.result) is not str or self.result not in _RESULTS
                or type(self.count) is not int or not 0 <= self.count <= MAX_MESSAGES
                or type(self._boundary) is not int or not 0 <= self._boundary <= self.count
                or type(self._wire) is not bytes or len(self._wire) > MAX_BYTES
                or type(self._pairing) is not bytes or len(self._pairing) > MAX_BYTES):
            raise ValueError("public evidence rejected")

    def compare(self, right: PublicEvidence) -> DiagnosticPayload:
        """Routine plane: finite classifications/counts only, never digests."""
        result: Result = "equal"
        if self.result != "equal":
            result = self.result
        elif right.result != "equal":
            result = right.result
        elif self._pairing != right._pairing:
            result = "tool_pairing"
        elif self._wire != right._wire:
            result = "public_history"
        elif self._boundary != right._boundary:
            result = "archive_boundary"
        return diagnostic_payload({"component": "shadow", "result": result,
                                   "left_count": self.count, "right_count": right.count})

    def operator_digest(self) -> str | None:
        """Explicit private digest only; never expose frozen payload bytes."""
        if self.result != "equal":
            return None
        return hashlib.sha256(self._wire + b"\0" + str(self._boundary).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class OperatorComparison:
    routine: DiagnosticPayload
    left_digest: str | None = field(repr=False)
    right_digest: str | None = field(repr=False)


class _RejectedError(Exception):
    def __init__(self, result: Result):
        self.result: Result = result
        super().__init__("public evidence rejected")


def diagnostic_payload(value: object) -> DiagnosticPayload:
    """Reject dynamic fields with constant content-free errors, no label echo."""
    if (type(value) is not dict or set(cast(dict[object, object], value)) != {"component", "result", "left_count", "right_count"}
            or type(cast(dict[object, object], value)["component"]) is not str or cast(dict[object, object], value)["component"] not in _COMPONENTS
            or type(cast(dict[object, object], value)["result"]) is not str or cast(dict[object, object], value)["result"] not in _RESULTS
            or any(type(cast(dict[object, object], value)[key]) is not int or not 0 <= cast(int, cast(dict[object, object], value)[key]) <= MAX_COUNT
                   for key in ("left_count", "right_count"))):
        raise ValueError("diagnostic payload rejected")
    # Exact keys, finite strings and bounded exact integers were checked above.
    return DiagnosticPayload(component=cast(DiagnosticPayload, value)["component"], result=cast(DiagnosticPayload, value)["result"],
                             left_count=cast(DiagnosticPayload, value)["left_count"], right_count=cast(DiagnosticPayload, value)["right_count"])


def _bounded(value: object) -> None:
    """Check before copying/serialization, including cycles and finite JSON."""
    size = 0
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal size, nodes
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_BYTES:
            raise _RejectedError("bound_exceeded")
        if type(item) is str:
            if len(item) > MAX_BYTES:
                raise _RejectedError("bound_exceeded")
            try:
                size += len(item.encode("utf-8"))
            except UnicodeError:
                raise _RejectedError("invalid_input") from None
        elif item is None or type(item) in (bool, int):
            if type(item) is int and item.bit_length() > 64:
                raise _RejectedError("bound_exceeded")
            size += 24
        elif type(item) is float:
            if not math.isfinite(item):
                raise _RejectedError("invalid_input")
            size += 24
        elif type(item) is list:
            size += len(cast(list[object], item)) + 2
            for child in cast(list[object], item):
                visit(child, depth + 1)
        elif type(item) is dict:
            size += len(cast(dict[object, object], item)) * 2 + 2
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str:
                    raise _RejectedError("invalid_input")
                visit(key, depth + 1)
                visit(child, depth + 1)
        else:
            raise _RejectedError("invalid_input")
        if size > MAX_BYTES:
            raise _RejectedError("bound_exceeded")

    visit(value, 0)


def _ordinary_message(raw: object) -> _Message:
    """Validate original ordinary grammar before sanitization/projection.

    Pairing is separate: canonical private execution links span supplied streams,
    while frozen root public history uses provider-call pairing. This validator
    checks every supplied row, including rows that projection will later hide.
    """
    if type(raw) is not dict or set(cast(dict[object, object], raw)) - _PUBLIC_FIELDS:
        raise _RejectedError("unsupported")
    # Callers bound the original tree before entry; allowed fields prove string keys.
    role = cast(dict[str, object], raw).get("role")
    if type(role) is not str or role not in {"user", "assistant", "tool"}:
        raise _RejectedError("unsupported")
    if type(cast(dict[str, object], raw).get("content")) not in (str, type(None)):
        raise _RejectedError("unsupported")
    marker = cast(dict[str, object], raw).get("_runtime_context")
    if marker is not None and (type(marker) is not dict or set(cast(dict[object, object], marker)) - {
            "version", "sources", "suffix", "blocks"} or type(cast(dict[object, object], marker).get("version")) is not int
            or cast(dict[object, object], marker)["version"] != 1 or type(cast(dict[object, object], marker).get("sources")) is not list
            or any(type(source) is not str for source in cast(list[object], cast(dict[object, object], marker)["sources"]))
            or ("suffix" in marker and type(cast(dict[object, object], marker)["suffix"]) is not str)
            or ("blocks" in marker and type(cast(dict[object, object], marker)["blocks"]) is not list)):
        raise _RejectedError("unsupported")
    if role != "tool" and {"tool_call_id", "name"} & cast(dict[str, object], raw).keys():
        raise _RejectedError("invalid_input")
    if role != "assistant" and "tool_calls" in raw:
        raise _RejectedError("invalid_input")
    if role == "tool" and (type(cast(dict[str, object], raw).get("tool_call_id")) is not str
                           or ("name" in raw and type(cast(dict[str, object], raw)["name"]) is not str)):
        raise _RejectedError("tool_pairing")
    tools = cast(dict[str, object], raw).get("tool_calls", [])
    if type(tools) is not list:
        raise _RejectedError("invalid_input")
    for tool in cast(list[object], tools):
        if (type(tool) is not dict or set(cast(dict[object, object], tool)) != {"id", "type", "function"}
                or cast(dict[object, object], tool)["type"] != "function" or type(cast(dict[object, object], tool)["id"]) is not str
                or not cast(dict[object, object], tool)["id"] or type(cast(dict[object, object], tool)["function"]) is not dict
                or set(cast(dict[object, object], cast(dict[object, object], tool)["function"])) != {"name", "arguments"}
                or any(type(cast(dict[object, object], cast(dict[object, object], tool)["function"])[key]) is not str for key in ("name", "arguments"))):
            raise _RejectedError("tool_pairing")
        try:
            arguments = json.loads(cast("_Call", tool)["function"]["arguments"])
            _bounded(arguments)
        except (ValueError, RecursionError):
            raise _RejectedError("invalid_input") from None
        if type(arguments) is not dict:
            raise _RejectedError("invalid_input")
    # Exact call/function keys and string fields are proved above. The sanitizer
    # deep-copies, removes the marker and only trims string content: role, optional
    # field presence and validated call grammar survive unchanged.
    row = public_history_message(cast(dict[str, object], raw))
    row.pop("_runtime_context", None)
    return cast("_Message", row)


def _freeze(messages: object, boundary: object) -> PublicEvidence:
    _bounded(messages)
    if type(messages) is not list or type(boundary) is not int:
        raise _RejectedError("invalid_input")
    if len(cast(list[object], messages)) > MAX_MESSAGES:
        raise _RejectedError("bound_exceeded")
    if not 0 <= boundary <= len(cast(list[object], messages)):
        raise _RejectedError("invalid_input")
    normalized: list[_Message] = []
    pairs: list[list[str]] = []
    pending: set[str] = set()
    calls = 0
    for raw in cast(list[object], messages):
        row = _ordinary_message(raw)
        role = row["role"]
        if role == "tool":
            identity = row.get("tool_call_id")
            if (type(identity) is not str or identity not in pending
                    or ("name" in row and type(row["name"]) is not str)):
                raise _RejectedError("tool_pairing")
            pending.remove(identity)
            pairs.append(["result", identity])
        elif pending:
            raise _RejectedError("tool_pairing")
        tools = row.get("tool_calls", [])
        if type(tools) is not list:
            raise _RejectedError("invalid_input")
        for tool in tools:
            calls += 1
            if calls > MAX_TOOL_CALLS:
                raise _RejectedError("bound_exceeded")
            # _ordinary_message validated the exact call shape before this loop.
            if tool["id"] in pending:
                raise _RejectedError("tool_pairing")
            pending.add(tool["id"])
            pairs.append(["call", tool["id"]])
        normalized.append(row)
    if pending:
        raise _RejectedError("tool_pairing")
    wire = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(wire) > MAX_BYTES:
        raise _RejectedError("bound_exceeded")
    return PublicEvidence("equal", len(normalized), wire, json.dumps(pairs).encode(), boundary)


def legacy_evidence(messages: object, archive_boundary: object) -> PublicEvidence:
    """Freeze supplied public history; unsupported multimodal evidence fails closed."""
    try:
        return _freeze(messages, archive_boundary)
    except _RejectedError as error:
        return PublicEvidence(error.result)
    except (TypeError, ValueError, UnicodeError):
        return PublicEvidence("invalid_input")


def canonical_evidence(snapshot: SourceSnapshot) -> PublicEvidence:
    """Project retained root history with checkpoints explicitly disabled."""
    try:
        if type(snapshot) is not SourceSnapshot:
            raise _RejectedError("invalid_input")
        _bounded(snapshot.events)
        if len(snapshot.events) > MAX_MESSAGES:
            raise _RejectedError("bound_exceeded")
        total_calls = 0
        for event in snapshot.events:
            if (type(event) is not dict or type(event.get("role")) is not str
                    or event["role"] not in {"user", "assistant", "tool"}
                    or set(event) - {"role", "content", "tool_calls", "tool_call_id", "name",
                        "media", "timestamp", "_runtime_context", "_event_id", "_run_id",
                        "_stream_id", "_parent_stream_id", "_execution_id", "_execution_ids",
                        "_command", "_source_origin", "_artifact_read", "_hidden_history"}):
                raise _RejectedError("unsupported")
            # Validate uncoerced content/tool grammar on ALL original rows. Hidden
            # streams/commands are excluded from equivalence, not input bounds.
            original = _ordinary_message({key: value for key, value in event.items()
                                          if key in _PUBLIC_FIELDS})
            tools = original.get("tool_calls", [])
            if type(tools) is not list:
                raise _RejectedError("invalid_input")
            total_calls += len(cast(list[object], tools))
            if total_calls > MAX_TOOL_CALLS:
                raise _RejectedError("bound_exceeded")
        validate_event_ids(snapshot.events)
        try:
            validate_execution_links(snapshot.events)
        except SourceValidationError:
            raise _RejectedError("tool_pairing") from None
        projection = project_source(snapshot, use_checkpoint=False)
        boundary = snapshot.header.get("last_archived", snapshot.header.get("last_consolidated", 0))
        if type(boundary) is not int or not 0 <= boundary <= len(snapshot.events):
            raise _RejectedError("invalid_input")
        # Archive offsets count canonical events; root visibility may exclude child
        # streams/commands. Compare the projected public prefix, not raw offsets.
        from dataclasses import replace

        prefix = project_source(replace(snapshot, events=snapshot.events[:boundary]),
                                use_checkpoint=False)
        return _freeze(projection.messages(), len(prefix.nodes))
    except _RejectedError as error:
        return PublicEvidence(error.result)
    except (KeyError, TypeError, ValueError, AttributeError):
        return PublicEvidence("invalid_input")


def compare(left: PublicEvidence, right: PublicEvidence) -> DiagnosticPayload:
    """Routine plane: finite classifications/counts only, never digests."""
    return PublicEvidence.compare(left, right)


def operator_compare(left: PublicEvidence, right: PublicEvidence) -> OperatorComparison:
    """Explicit private operator evidence; no execution, mode or mutation authority."""
    return OperatorComparison(compare(left, right), PublicEvidence.operator_digest(left),
                              PublicEvidence.operator_digest(right))
