"""Shared recovery parsers preserve JSON, digest, lexical-root and binding policies."""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery as codec
from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_transaction_prepare as preparation
from nanobot.session import legacy_receipt_source as legacy
from tests.session.test_legacy_receipt_source import PINNED_CHUNK_DIGEST, PINNED_MESSAGES

_WRAPPERS = (
    (codec, "reject_duplicate_receipt_keys", "_reject_duplicate_keys", "pairs",
     "list[tuple[str, object]]", "dict[str, object]"),
    (codec, "reject_non_json_receipt_number", "_reject_non_json_number", "value", "str", "None"),
    (preparation, "legacy_transaction_chunk_digest", "_legacy_chunk_digest", "messages",
     "list[dict[str, object]]", "str"),
    (quarantine, "normalize_recovery_root", "_absolute", "value", "object", "Path"),
)
_RETURNING = (_WRAPPERS[0], _WRAPPERS[2], _WRAPPERS[3])


@pytest.fixture
def _messages(tmp_path: Path):
    source = tmp_path / "events.json"
    source.write_bytes(json.dumps(PINNED_MESSAGES, ensure_ascii=False).encode("utf-8"))
    messages = json.loads(source.read_bytes())
    assert messages == PINNED_MESSAGES
    return messages


def _input(public, messages, root):
    if public == "reject_duplicate_receipt_keys":
        return [("events", messages), ("name", "receipt")]
    if public == "reject_non_json_receipt_number":
        return "NaN"
    if public == "legacy_transaction_chunk_digest":
        return messages
    return root


@pytest.mark.parametrize("module,public,private,argument,annotation,result", _WRAPPERS)
def test_supported_parsing_signatures_and_exports(
    module, public, private, argument, annotation, result,
) -> None:
    operation = getattr(module, public)
    signature = inspect.signature(operation)
    assert tuple(signature.parameters) == (argument,)
    parameter = signature.parameters[argument]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty and parameter.annotation == annotation
    assert signature.return_annotation == result and public in module.__all__


@pytest.mark.parametrize("module,public,private,argument,annotation,result", _RETURNING)
def test_dynamic_parsing_forwarding_runs_the_real_helper(
    tmp_path, _messages, monkeypatch, module, public, private, argument, annotation, result,
) -> None:
    operation = getattr(module, public)
    original = getattr(module, private)
    value = _input(public, _messages, tmp_path)
    calls, results = [], []

    def first(actual):
        assert actual is value
        calls.append("first")
        produced = original(actual)
        results.append(produced)
        return produced

    def second(actual):
        assert actual is value
        calls.append("second")
        produced = original(actual)
        results.append(produced)
        return produced

    monkeypatch.setattr(module, private, first)
    first_result = operation(value)
    monkeypatch.setattr(module, private, second)
    second_result = operation(**{argument: value})
    assert first_result is results[0] and second_result is results[1]
    assert calls == ["first", "second"]
    if public == "reject_duplicate_receipt_keys":
        assert first_result == {"events": _messages, "name": "receipt"}
    elif public == "legacy_transaction_chunk_digest":
        assert first_result == PINNED_CHUNK_DIGEST
    else:
        assert first_result == tmp_path


@pytest.mark.parametrize("module,public,private,argument,annotation,result", _WRAPPERS)
@pytest.mark.parametrize("kind", [OSError, ValueError, KeyboardInterrupt])
def test_parsing_forwarding_preserves_exact_faults(
    tmp_path, _messages, monkeypatch, module, public, private, argument, annotation, result, kind,
) -> None:
    operation = getattr(module, public)
    value = _input(public, _messages, tmp_path)
    fault = kind("parser probe")
    calls = []

    def fail(actual):
        calls.append(actual)
        raise fault

    monkeypatch.setattr(module, private, fail)
    with pytest.raises(kind) as raised:
        operation(value)
    assert raised.value is fault and len(calls) == 1 and calls[0] is value


def test_legacy_digest_capture_retains_typed_import_identity(_messages) -> None:
    capture = getattr(legacy, "capture_legacy_receipt_chunk_digest")
    operation = capture()
    assert operation is legacy._chunk_digest
    assert preparation._legacy_chunk_digest is operation
    signature = inspect.signature(capture)
    assert not signature.parameters
    assert signature.return_annotation == "_Callable[[list[dict[str, object]]], str]"
    assert not hasattr(legacy, "__all__")
    assert operation(_messages) == PINNED_CHUNK_DIGEST


def test_legacy_digest_capture_preserves_both_binding_times(_messages, monkeypatch) -> None:
    capture = getattr(legacy, "capture_legacy_receipt_chunk_digest")
    alias = capture()
    operation = getattr(preparation, "legacy_transaction_chunk_digest")
    original = legacy._chunk_digest
    calls = []
    fault = RuntimeError("later private replacement")

    def later(messages):
        calls.append(messages)
        raise fault

    monkeypatch.setattr(legacy, "_chunk_digest", later)
    assert alias is original and preparation._legacy_chunk_digest is original
    assert alias(_messages) == operation(_messages) == PINNED_CHUNK_DIGEST
    assert not calls
    # A later consumer import captures the then-current callback, not a stale alias.
    assert capture() is later
    with pytest.raises(RuntimeError) as raised:
        capture()(_messages)
    assert raised.value is fault and len(calls) == 1 and calls[0] is _messages


@pytest.mark.parametrize("raw", ['{"key":1,"key":2}', '{"nested":{"key":1,"key":2}}'])
def test_duplicate_json_callbacks_refuse_real_duplicate_keys(raw) -> None:
    operation = getattr(codec, "reject_duplicate_receipt_keys")
    with pytest.raises(codec._DuplicateKeyError):
        json.loads(raw, object_pairs_hook=operation)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_json_number_callback_refuses_actual_json_constants(value) -> None:
    operation = getattr(codec, "reject_non_json_receipt_number")
    with pytest.raises(ValueError):
        json.loads("[" + value + "]", parse_constant=operation)


@pytest.mark.parametrize("fallback", [False, True])
def test_transaction_digest_retains_canonical_unicode_and_string_fallback(_messages, fallback) -> None:
    operation = getattr(preparation, "legacy_transaction_chunk_digest")
    if fallback:
        _messages[0]["observed_at"] = datetime(2026, 10, 9, tzinfo=timezone.utc)
    encoded = json.dumps(
        _messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    assert operation(_messages) == expected
    if not fallback:
        assert expected == PINNED_CHUNK_DIGEST


@pytest.mark.parametrize("kind", ["path", "string", "relative"])
def test_lexical_root_normalization_never_creates_a_path(tmp_path, monkeypatch, kind) -> None:
    operation = getattr(quarantine, "normalize_recovery_root")
    expected = tmp_path / "not-created"
    if kind == "relative":
        monkeypatch.chdir(tmp_path)
        value = "not-created"
    elif kind == "string":
        value = str(expected)
    else:
        value = expected
    assert operation(value) == expected
    assert not expected.exists()


@pytest.mark.parametrize("value", [None, 0, "/", object()])
def test_lexical_root_normalization_retains_invalid_value_refusals(value) -> None:
    operation = getattr(quarantine, "normalize_recovery_root")
    with pytest.raises(quarantine.QuarantineError) as raised:
        operation(value)
    assert raised.value.__cause__ is None


def test_lexical_root_normalization_does_not_resolve_symlinks(tmp_path, monkeypatch) -> None:
    operation = getattr(quarantine, "normalize_recovery_root")
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    def forbidden(self, *args, **kwargs):
        raise AssertionError("lexical normalization must not resolve a path")

    monkeypatch.setattr(Path, "resolve", forbidden)
    assert operation(link) == link
