"""Strict, read-only validation of the legacy memory receipt boundary."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import nanobot.agent.memory_recovery as memory_recovery
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_recovery import (
    MemoryReceipt,
    MemoryReceiptRecoveryRequired,
    MemoryReceiptValidationError,
    parse_memory_receipt,
)

RECEIPT_NAME = ".consolidation.pending.json"


def _payload(**overrides: object) -> dict[str, object]:
    memory_output = "# Memory\nUser prefers deterministic tests.\n"
    payload: dict[str, object] = {
        "version": 1,
        "chunk_digest": "a" * 64,
        "history_marker_id": "marker-1",
        "history_entry": "[2026-09-18 12:00] recovered summary",
        "memory_output": memory_output,
        "base_memory_digest": "b" * 64,
        "output_memory_digest": hashlib.sha256(memory_output.encode("utf-8")).hexdigest(),
        "reference_only": False,
        "created_at": "2026-09-18T12:00:00+00:00",
        "session_key": "cli:receipt",
        "start_offset": 0,
        "end_offset": 2,
        "session_generation": 4,
    }
    payload.update(overrides)
    return payload


def _write_receipt(
    workspace: Path,
    payload: object,
    *,
    raw: bytes | None = None,
    mode: int = 0o600,
) -> Path:
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    receipt = memory_dir / RECEIPT_NAME
    receipt.write_bytes(raw if raw is not None else json.dumps(payload).encode("utf-8"))
    receipt.chmod(mode)
    return receipt


def test_valid_literal_receipt_blocks_before_migration_and_preserves_bytes(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    history = memory_dir / "HISTORY.md"
    memory = memory_dir / "MEMORY.md"
    history.write_text("[2026-09-18 11:00] existing\n", encoding="utf-8")
    memory.write_text("# Existing\n", encoding="utf-8")
    receipt = _write_receipt(tmp_path, _payload(extra_recovery_metadata={"keep": True}))
    before = {
        path: (path.stat().st_mode, path.stat().st_mtime_ns, path.read_bytes())
        for path in (history, memory, receipt)
    }
    memory_dir_before = memory_dir.stat()

    with pytest.raises(MemoryReceiptRecoveryRequired) as raised:
        MemoryStore(tmp_path)

    assert str(raised.value) == "memory recovery required"
    assert set(raised.value.args) == {"memory recovery required"}
    for path, (mode, mtime_ns, contents) in before.items():
        current = path.stat()
        assert (current.st_mode, current.st_mtime_ns, path.read_bytes()) == (
            mode,
            mtime_ns,
            contents,
        )
    assert memory_dir.stat().st_mtime_ns == memory_dir_before.st_mtime_ns
    assert not (memory_dir / "history.jsonl").exists()
    assert not (memory_dir / ".cursor").exists()
    assert not (memory_dir / ".dream_cursor").exists()
    assert not list(memory_dir.glob("HISTORY.md.bak*"))


def test_parser_returns_detached_typed_value_and_retains_unknown_metadata(tmp_path: Path) -> None:
    raw_payload = _payload(extra_recovery_metadata={"nested": ["preserve"]})
    receipt_path = _write_receipt(tmp_path, raw_payload)
    raw_bytes = receipt_path.read_bytes()

    parsed = parse_memory_receipt(receipt_path)

    assert isinstance(parsed, MemoryReceipt)
    assert parsed is not None
    assert parsed.raw_bytes == raw_bytes
    assert parsed.memory_output == raw_payload["memory_output"]
    assert parsed.unknown_fields == {"extra_recovery_metadata": {"nested": ["preserve"]}}
    parsed_unknown = dict(parsed.unknown_fields)
    parsed_unknown["extra_recovery_metadata"]["nested"].append("detached")
    assert json.loads(raw_bytes)["extra_recovery_metadata"] == {"nested": ["preserve"]}
    assert parsed.raw["extra_recovery_metadata"] == {
        "nested": ["preserve"],
    }


@pytest.mark.parametrize(
    "override",
    [
        {"version": 2},
        {"version": True},
        {"chunk_digest": "not-a-digest"},
        {"history_marker_id": "   "},
        {"history_entry": 3},
        {"memory_output": None},
        {"memory_output": None, "output_memory_digest": None},
        {
            "memory_output": None,
            "output_memory_digest": None,
            "reference_only": False,
        },
        {
            "memory_output": None,
            "output_memory_digest": hashlib.sha256(b"").hexdigest(),
            "reference_only": True,
        },
        {"output_memory_digest": None},
        {
            "memory_output": None,
            "output_memory_digest": None,
            "reference_only": True,
            "memory_update": "stale output",
        },
        {
            "memory_output": None,
            "output_memory_digest": None,
            "reference_only": True,
            "memory_output_digest": "c" * 64,
        },
        {"base_memory_digest": "c" * 63},
        {"output_memory_digest": "c" * 64},
        {"reference_only": 1},
        {"created_at": ""},
        {"start_offset": True},
        {"start_offset": -1},
        {"start_offset": 3, "end_offset": 2},
        {"session_generation": True},
        {"session_key": "   "},
        {"end_offset": 2, "session_key": None},
        {"memory_update": "different"},
        {"memory_base_digest": "c" * 64},
        {"memory_output_digest": "c" * 64},
        {"generation": 5},
    ],
)
def test_invalid_receipts_raise_content_free_validation_error(tmp_path: Path, override: dict[str, object]) -> None:
    receipt_path = _write_receipt(tmp_path, _payload(**override))

    with pytest.raises(MemoryReceiptValidationError) as raised:
        parse_memory_receipt(receipt_path)

    assert str(raised.value) == "memory recovery required"
    assert "digest" not in str(raised.value)
    assert "memory_output" not in str(raised.value)


def test_duplicate_keys_at_any_nesting_level_are_rejected_without_mutation(tmp_path: Path) -> None:
    raw = (
        b'{"version":1,"chunk_digest":"' + b"a" * 64
        + b'","history_marker_id":"marker","history_entry":"entry",'
        b'"memory_output":"output","base_memory_digest":"' + b"b" * 64
        + b'","output_memory_digest":"'
        + hashlib.sha256(b"output").hexdigest().encode()
        + b'","reference_only":false,"created_at":"now",'
        b'"unknown":{"x":1,"x":2}}'
    )
    receipt_path = _write_receipt(tmp_path, {}, raw=raw)
    before = receipt_path.read_bytes()

    with pytest.raises(MemoryReceiptValidationError):
        parse_memory_receipt(receipt_path)

    assert receipt_path.read_bytes() == before


def test_aliases_must_agree_and_generation_alias_is_supported(tmp_path: Path) -> None:
    payload = _payload(
        memory_update=_payload()["memory_output"],
        memory_base_digest=_payload()["base_memory_digest"],
        memory_output_digest=_payload()["output_memory_digest"],
        generation=_payload()["session_generation"],
    )
    receipt_path = _write_receipt(tmp_path, payload)

    parsed = parse_memory_receipt(receipt_path)

    assert parsed is not None
    assert parsed.memory_update == parsed.memory_output
    assert parsed.memory_base_digest == parsed.base_memory_digest
    assert parsed.memory_output_digest == parsed.output_memory_digest
    assert parsed.generation == parsed.session_generation


def test_staging_reference_receipt_preserves_exact_nullable_fields_and_aliases(
    tmp_path: Path,
) -> None:
    base_memory_digest = hashlib.sha256(b"existing memory").hexdigest()
    payload: dict[str, object] = {
        "version": 1,
        "chunk_digest": "a" * 64,
        "history_marker_id": "marker-1",
        "history_entry": "[2026-09-18 12:00] referenced summary",
        "memory_output": None,
        "base_memory_digest": base_memory_digest,
        "output_memory_digest": None,
        "reference_only": True,
        "created_at": "2026-09-18T12:00:00+00:00",
        "session_key": "cli:receipt",
        "start_offset": 0,
        "end_offset": 2,
        "session_generation": 4,
        "memory_update": None,
        "memory_base_digest": base_memory_digest,
        "memory_output_digest": None,
    }
    raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    receipt_path = _write_receipt(tmp_path, payload, raw=raw_bytes)

    parsed = parse_memory_receipt(receipt_path)

    assert parsed is not None
    assert parsed.raw_bytes == raw_bytes
    assert parsed.memory_output is None
    assert parsed.output_memory_digest is None
    assert parsed.memory_update is None
    assert parsed.memory_output_digest is None
    assert parsed.memory_base_digest == base_memory_digest


def test_missing_receipt_is_the_only_preflight_pass_through(tmp_path: Path) -> None:
    assert parse_memory_receipt(tmp_path / "memory" / RECEIPT_NAME) is None
    store = MemoryStore(tmp_path)
    assert store.memory_dir.is_dir()


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_unsafe_receipt_entry_blocks_without_following_or_mutating(tmp_path: Path, kind: str) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    receipt = memory_dir / RECEIPT_NAME
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    if kind == "symlink":
        try:
            receipt.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"symlink unavailable: {exc}")
    elif kind == "directory":
        receipt.mkdir()
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(receipt)
    before = memory_dir.stat().st_mtime_ns

    with pytest.raises(MemoryReceiptRecoveryRequired):
        MemoryStore(tmp_path)

    assert memory_dir.stat().st_mtime_ns == before
    assert not (memory_dir / "history.jsonl").exists()


def test_non_private_receipt_is_rejected_without_chmod(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path, _payload(), mode=0o644)
    before_mode = stat.S_IMODE(receipt.stat().st_mode)

    with pytest.raises(MemoryReceiptRecoveryRequired):
        MemoryStore(tmp_path)

    assert stat.S_IMODE(receipt.stat().st_mode) == before_mode


def test_hardlinked_receipt_is_rejected_as_ambiguous(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path, _payload())
    alias = tmp_path / "receipt-alias.json"
    try:
        os.link(receipt, alias)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    with pytest.raises(MemoryReceiptRecoveryRequired):
        MemoryStore(tmp_path)

    assert receipt.read_bytes() == alias.read_bytes()


def test_nested_unknown_metadata_and_receipt_bytes_are_preserved_by_parser(tmp_path: Path) -> None:
    payload = _payload(recovery={"future": {"step": 1}})
    receipt_path = _write_receipt(tmp_path, payload)
    before = receipt_path.read_bytes()

    parsed = parse_memory_receipt(receipt_path)

    assert parsed is not None
    assert parsed.raw_bytes == before
    assert parsed.unknown_fields["recovery"] == {"future": {"step": 1}}
    assert receipt_path.read_bytes() == before


def test_ancestor_swap_to_symlink_at_open_boundary_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = _write_receipt(tmp_path, _payload())
    memory_dir = receipt_path.parent
    moved_memory = tmp_path / "memory-before-race"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / RECEIPT_NAME).write_bytes(receipt_path.read_bytes())
    triggered = False
    original_open = os.open

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal triggered
        if path == "memory" and dir_fd is not None and not triggered:
            triggered = True
            memory_dir.rename(moved_memory)
            memory_dir.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    try:
        with pytest.raises(MemoryReceiptRecoveryRequired) as raised:
            MemoryStore(tmp_path)
    finally:
        if memory_dir.is_symlink():
            memory_dir.unlink()
        if moved_memory.exists():
            moved_memory.rename(memory_dir)

    assert triggered
    assert str(raised.value) == "memory recovery required"


def test_observed_leaf_disappearance_is_not_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = _write_receipt(tmp_path, _payload())
    original_open = os.open
    triggered = False

    def disappearing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal triggered
        if path == RECEIPT_NAME and dir_fd is not None and not triggered:
            triggered = True
            receipt_path.unlink()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", disappearing_open)
    with pytest.raises(MemoryReceiptRecoveryRequired) as raised:
        MemoryStore(tmp_path)

    assert triggered
    assert str(raised.value) == "memory recovery required"


def test_leaf_swap_to_fifo_is_nonblocking_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = _write_receipt(tmp_path, _payload())
    moved_receipt = tmp_path / "receipt-before-fifo"
    original_open = os.open
    triggered = False

    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO unavailable")

    def fifo_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal triggered
        if path == RECEIPT_NAME and dir_fd is not None and not triggered:
            triggered = True
            receipt_path.rename(moved_receipt)
            os.mkfifo(receipt_path)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", fifo_open)
    try:
        with pytest.raises(MemoryReceiptRecoveryRequired) as raised:
            MemoryStore(tmp_path)
    finally:
        if receipt_path.exists():
            receipt_path.unlink()
        if moved_receipt.exists():
            moved_receipt.rename(receipt_path)

    assert triggered
    assert str(raised.value) == "memory recovery required"


def test_same_size_mutation_during_read_fails_closed_and_preserves_disk_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = _write_receipt(tmp_path, _payload(extra={"sentinel": "x"}))
    original_bytes = receipt_path.read_bytes()
    mutation = b"sentinel"
    mutation_index = original_bytes.index(mutation)
    mutated_bytes = (
        original_bytes[:mutation_index]
        + mutation.upper()
        + original_bytes[mutation_index + len(mutation):]
    )
    original_read = os.read
    mutated = False

    def mutating_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, size)
        if not mutated:
            mutated = True
            receipt_path.write_bytes(mutated_bytes)
        return chunk

    monkeypatch.setattr(os, "read", mutating_read)
    with pytest.raises(MemoryReceiptValidationError) as raised:
        parse_memory_receipt(receipt_path)

    assert mutated
    assert str(raised.value) == "memory recovery required"
    assert receipt_path.read_bytes() == mutated_bytes


def test_missing_receipt_under_workspace_symlink_preserves_migration_behavior(
    tmp_path: Path,
) -> None:
    target = tmp_path / "real-workspace"
    memory_dir = target / "memory"
    memory_dir.mkdir(parents=True)
    legacy = "[2026-09-18 12:00] preserved legacy entry\n"
    (memory_dir / "HISTORY.md").write_text(legacy, encoding="utf-8")
    workspace_alias = tmp_path / "workspace-alias"
    try:
        workspace_alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    store = MemoryStore(workspace_alias)

    assert store.memory_dir == target.resolve() / "memory"
    assert (workspace_alias / "memory").samefile(store.memory_dir)
    assert workspace_alias.is_symlink()
    assert (memory_dir / "history.jsonl").exists()
    assert (memory_dir / "HISTORY.md.bak").read_text(encoding="utf-8") == legacy


def test_missing_nofollow_primitive_fails_closed_for_existing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = _write_receipt(tmp_path, _payload())
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    with pytest.raises(MemoryReceiptRecoveryRequired) as raised:
        MemoryStore(tmp_path)

    assert str(raised.value) == "memory recovery required"
    assert receipt_path.exists()


def test_deep_json_and_detachment_failures_are_content_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = json.dumps(_payload(), separators=(",", ":"))[:-1] + ","
    deeply_nested = b'"extra":' + b"[" * 2_000 + b"0" + b"]" * 2_000 + b"}"
    receipt_path = _write_receipt(tmp_path, {}, raw=prefix.encode("utf-8") + deeply_nested)
    with pytest.raises(MemoryReceiptValidationError) as raised:
        parse_memory_receipt(receipt_path)
    assert str(raised.value) == "memory recovery required"

    valid = _write_receipt(tmp_path / "detached", _payload())
    monkeypatch.setattr(memory_recovery.copy, "deepcopy", lambda _value: (_ for _ in ()).throw(RecursionError()))
    with pytest.raises(MemoryReceiptValidationError) as raised:
        parse_memory_receipt(valid)
    assert str(raised.value) == "memory recovery required"


def test_constructor_rejection_has_no_forbidden_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    history = memory_dir / "HISTORY.md"
    memory = memory_dir / "MEMORY.md"
    history.write_text("legacy", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    receipt = _write_receipt(tmp_path, _payload())
    before = {path: path.read_bytes() for path in (history, memory, receipt)}

    attempts: list[str] = []

    def forbidden(*_args, **_kwargs):
        attempts.append("forbidden mutation")

    monkeypatch.setattr("nanobot.agent.memory.ensure_dir", forbidden)
    for owner, name in (
        (Path, "mkdir"),
        (Path, "write_text"),
        (Path, "write_bytes"),
        (Path, "replace"),
        (Path, "unlink"),
        (Path, "chmod"),
        (os, "mkdir"),
        (os, "makedirs"),
        (os, "replace"),
        (os, "rename"),
        (os, "chmod"),
    ):
        monkeypatch.setattr(owner, name, forbidden)

    with pytest.raises(MemoryReceiptRecoveryRequired):
        MemoryStore(tmp_path)

    assert attempts == []
    assert {path: path.read_bytes() for path in (history, memory, receipt)} == before
    assert not (memory_dir / "history.jsonl").exists()
    assert not (memory_dir / ".cursor").exists()
    assert not (memory_dir / ".dream_cursor").exists()
    assert not list(memory_dir.glob("HISTORY.md.bak*"))


@pytest.mark.parametrize("variant", ["literal", "aliases", "reference", "no-session"])
def test_bytes_decoder_matches_existing_composition_and_path_parser(
    tmp_path: Path, variant: str,
) -> None:
    payload = _payload(metadata={"nested": [{"future": [1, None, True]}]})
    if variant == "aliases":
        payload.update(
            memory_update=payload["memory_output"],
            memory_base_digest=payload["base_memory_digest"],
            memory_output_digest=payload["output_memory_digest"],
            generation=payload.pop("session_generation"),
        )
    elif variant == "reference":
        payload.update(memory_output=None, output_memory_digest=None, reference_only=True)
    elif variant == "no-session":
        for name in ("session_key", "start_offset", "end_offset", "session_generation"):
            payload.pop(name)
    # Whitespace and insertion order deliberately differ from canonical serialization.
    raw = (" \n" + json.dumps(payload, indent=2) + "\n ").encode("utf-8")
    expected = memory_recovery._validate_payload(memory_recovery._decode(raw), raw)
    parsed = memory_recovery.decode_memory_receipt_bytes(raw)
    path = _write_receipt(tmp_path, {}, raw=raw)

    assert parsed == expected == parse_memory_receipt(path)
    assert parsed.raw_bytes == raw
    assert parsed.raw == payload
    assert parsed.unknown_fields == {"metadata": payload["metadata"]}
    assert "decode_memory_receipt_bytes" in memory_recovery.__all__


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff", b"{", b"null", b"[]", b"1", b"true", b'"text"',
        b'{"version":1,"version":1}',
        b'{"unknown":{"duplicate":1,"duplicate":2}}',
        b'{"unknown":NaN}', b'{"unknown":Infinity}', b'{"unknown":-Infinity}',
        b"{}",
    ] + [
        json.dumps(_payload(**override)).encode("utf-8")
        for override in (
            {"version": True}, {"version": 2},
            {"chunk_digest": "A" * 64}, {"base_memory_digest": "b" * 63},
            {"output_memory_digest": "c" * 64},
            {"start_offset": True}, {"start_offset": -1},
            {"end_offset": 1 << 63}, {"session_generation": 1.5},
            {"generation": True}, {"start_offset": 3, "end_offset": 2},
            {"reference_only": 1}, {"history_entry": ""},
            {"session_key": None}, {"memory_update": "conflicting alias"},
        )
    ],
)
def test_bytes_decoder_preserves_malformed_input_errors(raw: bytes) -> None:
    with pytest.raises(MemoryReceiptValidationError) as expected:
        memory_recovery._validate_payload(memory_recovery._decode(raw), raw)
    with pytest.raises(MemoryReceiptValidationError) as actual:
        memory_recovery.decode_memory_receipt_bytes(raw)

    assert type(actual.value) is type(expected.value)
    assert actual.value.args == expected.value.args == ("memory recovery required",)
    assert actual.value.__cause__ is expected.value.__cause__ is None


def test_bytes_decoder_detaches_nested_metadata_and_each_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode = memory_recovery.decode_memory_receipt_bytes
    payload = _payload(metadata={"nested": ["original"]})
    raw = json.dumps(payload).encode("utf-8")
    monkeypatch.setattr(memory_recovery, "_decode", lambda _raw: payload)
    first, second = decode(raw), decode(raw)

    payload["metadata"]["nested"].append("decoder mutation")
    assert first.unknown_fields["metadata"] == {"nested": ["original"]}
    assert first.raw_fields["metadata"] == {"nested": ["original"]}
    # Detached, not deeply immutable: each nested copy can be mutated independently.
    first.unknown_fields["metadata"]["nested"].append("unknown mutation")
    assert first.raw_fields["metadata"] == {"nested": ["original"]}
    first.raw_fields["metadata"]["nested"].append("raw-field mutation")
    assert first.unknown_fields["metadata"] == {"nested": ["original", "unknown mutation"]}
    assert second.unknown_fields["metadata"] == {"nested": ["original"]}
    assert second.raw_fields["metadata"] == {"nested": ["original"]}
    view = first.raw
    view["metadata"]["nested"].append("raw-property mutation")
    assert first.raw["metadata"] == {"nested": ["original", "raw-field mutation"]}
    assert first.raw_bytes == second.raw_bytes == raw


def test_bytes_decoder_performs_no_filesystem_or_owner_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode = memory_recovery.decode_memory_receipt_bytes
    raw = json.dumps(_payload()).encode("utf-8")
    attempts: list[str] = []

    def forbidden(*_args, **_kwargs):
        attempts.append("filesystem or owner read")
        raise AssertionError("bytes decoding must not establish provenance")

    with monkeypatch.context() as patch:
        patch.setattr(memory_recovery, "_read_owned_receipt", forbidden)
        patch.setattr(memory_recovery, "_canonical_receipt_path", forbidden)
        for owner, name in ((os, "open"), (os, "read"), (os, "stat"),
                            (Path, "open"), (Path, "read_bytes")):
            patch.setattr(owner, name, forbidden)
        parsed = decode(raw)
    assert attempts == []
    assert parsed.raw_bytes == raw


def test_bytes_decoder_calls_dynamic_helpers_in_order_with_exact_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode = memory_recovery.decode_memory_receipt_bytes
    raw = b"sentinel unparsed bytes"
    payload = _payload()
    sentinel = object()
    calls: list[str] = []

    def decode_helper(value):
        assert value is raw
        calls.append("decode")
        return payload

    def validate_helper(value, raw_value):
        assert value is payload
        assert raw_value is raw
        calls.append("validate")
        return sentinel

    monkeypatch.setattr(memory_recovery, "_decode", decode_helper)
    monkeypatch.setattr(memory_recovery, "_validate_payload", validate_helper)
    assert decode(raw) is sentinel
    assert calls == ["decode", "validate"]


@pytest.mark.parametrize("stage", ["decode", "validate"])
def test_bytes_decoder_preserves_dynamic_helper_exception_identity_and_cause(
    monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    decode = memory_recovery.decode_memory_receipt_bytes
    raw = b"sentinel unparsed bytes"
    payload = _payload()
    cause = ValueError("sentinel cause")
    failure = RuntimeError("sentinel failure")
    calls: list[str] = []

    def decode_helper(value):
        assert value is raw
        calls.append("decode")
        if stage == "decode":
            raise failure from cause
        return payload

    def validate_helper(value, raw_value):
        assert value is payload and raw_value is raw
        calls.append("validate")
        raise failure from cause

    monkeypatch.setattr(memory_recovery, "_decode", decode_helper)
    monkeypatch.setattr(memory_recovery, "_validate_payload", validate_helper)
    with pytest.raises(RuntimeError) as raised:
        decode(raw)
    assert raised.value is failure
    assert raised.value.__cause__ is cause
    assert calls == (["decode"] if stage == "decode" else ["decode", "validate"])


def test_bytes_decoder_recursion_escapes_but_path_parser_sanitizes(tmp_path: Path) -> None:
    decode = memory_recovery.decode_memory_receipt_bytes
    prefix = json.dumps(_payload(), separators=(",", ":"))[:-1] + ","
    raw = prefix.encode("utf-8") + b'"extra":' + b"[" * 2_000 + b"0" + b"]" * 2_000 + b"}"
    path = _write_receipt(tmp_path, {}, raw=raw)
    with pytest.raises(RecursionError):
        decode(raw)
    with pytest.raises(MemoryReceiptValidationError) as raised:
        parse_memory_receipt(path)
    assert raised.value.args == ("memory recovery required",)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


@pytest.mark.parametrize("error_type", [RecursionError, RuntimeError])
def test_bytes_decoder_detachment_failure_escapes_but_path_parser_sanitizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception],
) -> None:
    decode = memory_recovery.decode_memory_receipt_bytes
    raw = json.dumps(_payload(metadata={"nested": []})).encode("utf-8")
    path = _write_receipt(tmp_path, {}, raw=raw)
    failure = error_type("sentinel detachment failure")
    cause = ValueError("sentinel detachment cause")

    def fail_detachment(_value):
        raise failure from cause

    monkeypatch.setattr(memory_recovery.copy, "deepcopy", fail_detachment)
    with pytest.raises(error_type) as bare:
        decode(raw)
    assert bare.value is failure
    assert bare.value.__cause__ is cause
    with pytest.raises(MemoryReceiptValidationError) as sanitized:
        parse_memory_receipt(path)
    assert sanitized.value.args == ("memory recovery required",)
    assert sanitized.value.__cause__ is None
    assert sanitized.value.__suppress_context__
