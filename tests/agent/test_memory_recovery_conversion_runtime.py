"""Offline preparation uses real workspace locks, manifests, and immutable blobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_conversion_runtime as runtime
from nanobot.agent import memory_recovery_marker_stage as marker_stage
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery_conversion import capture_conversion_payload
from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest

_WORKSPACE_ID = "a" * 32
_MARKER = "receipt-marker"
_ENTRY = "[2026-09-25 12:00] receipt text"
_FRAME = f"<!-- nanobot-consolidation:{_MARKER} -->\n{_ENTRY}\n\n".encode()


def _receipt(memory: Path, *, output: str | None = None) -> bytes:
    raw = json.dumps(
        {
            "version": 1,
            "chunk_digest": hashlib.sha256(b"chunk").hexdigest(),
            "history_marker_id": _MARKER,
            "history_entry": _ENTRY,
            "memory_output": output,
            "base_memory_digest": hashlib.sha256(b"base").hexdigest(),
            "output_memory_digest": None
            if output is None
            else hashlib.sha256(output.encode()).hexdigest(),
            "reference_only": output is None,
            "created_at": "2026-09-25T00:00:00Z",
            "session_key": "session",
            "start_offset": 1,
            "end_offset": 2,
            "session_generation": 1,
        },
        sort_keys=True,
    ).encode()
    path = memory / ".consolidation.pending.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return raw


def _workspace(
    tmp_path: Path,
    *,
    history: bytes = b"# Existing\n\n",
    jsonl: bytes | None = None,
    output: str | None = None,
) -> tuple[Path, bytes]:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "HISTORY.md").write_bytes(history)
    if jsonl is not None:
        (memory / "history.jsonl").write_bytes(jsonl)
    return memory, _receipt(memory, output=output)


def _prepare(tmp_path: Path):
    return runtime.prepare_receipt_tail_conversion(tmp_path, workspace_id=_WORKSPACE_ID)


@pytest.mark.parametrize("output", [None, "private output"])
@pytest.mark.parametrize("jsonl", [None, b""])
def test_convert_stages_tail_and_only_prepares(
    tmp_path: Path, output: str | None, jsonl: bytes | None
) -> None:
    memory, receipt = _workspace(tmp_path, jsonl=jsonl, output=output)
    record = _prepare(tmp_path)
    assert record.phase == "CONVERSION_PREPARED"
    assert record.payload["disposition"] == "convert"
    assert record.payload["row_count"] == 2
    assert (memory / "HISTORY.md").read_bytes() == b"# Existing\n\n" + _FRAME
    assert not (memory / "HISTORY.md.bak").exists()
    assert not (memory / ".cursor").exists()
    assert not (memory / ".dream_cursor").exists()
    assert (memory / ".consolidation.pending.json").read_bytes() == receipt
    assert (memory / "history.jsonl").exists() == (jsonl is not None)
    assert _prepare(tmp_path).digest == record.digest
    assert (memory / "HISTORY.md").read_bytes().count(_FRAME) == 1
    with storage.open_workspace_memory(tmp_path) as authority:
        assert authority.read_blob(record.payload["receipt"]["blob_digest"]) == receipt
        for item in [record.payload["source"], *record.payload["targets"].values()]:
            if item["exists"]:
                assert len(authority.read_blob(item["blob_digest"])) == item["size"]
        for item in record.payload["outputs"].values():
            assert len(authority.read_blob(item["blob_digest"])) == item["size"]


def test_preserve_nonempty_leaves_history_and_has_no_rows(tmp_path: Path) -> None:
    memory, _ = _workspace(tmp_path, jsonl=b'{"cursor":999}\n')
    before = (memory / "HISTORY.md").stat()
    record = _prepare(tmp_path)
    assert record.payload["disposition"] == "preserve_nonempty_jsonl"
    assert record.payload["outputs"] is None
    assert record.payload["backup"] is None
    assert (memory / "HISTORY.md").stat() == before
    assert _prepare(tmp_path).digest == record.digest


def test_marker_present_and_backup_collision(tmp_path: Path) -> None:
    memory, _ = _workspace(tmp_path, history=b"# Existing\n\n" + _FRAME)
    (memory / "HISTORY.md.bak").write_bytes(b"occupied")
    before = (memory / "HISTORY.md").stat()
    record = _prepare(tmp_path)
    assert record.payload["backup"] == {
        "basename": "HISTORY.md.bak.2",
        "candidates": [
            {"basename": "HISTORY.md.bak", "exists": True},
            {"basename": "HISTORY.md.bak.2", "exists": False},
        ],
    }
    assert (memory / "HISTORY.md").stat() == before


def test_replay_after_marker_stage_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory, _ = _workspace(tmp_path)
    original = marker_stage.stage_history_marker

    def crash(*args: object):
        original(*args)
        raise RuntimeError("injected")

    monkeypatch.setattr(marker_stage, "stage_history_marker", crash)
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    monkeypatch.setattr(marker_stage, "stage_history_marker", original)
    assert _prepare(tmp_path).phase == "CONVERSION_PREPARED"
    assert (memory / "HISTORY.md").read_bytes().count(_FRAME) == 1


def test_replay_after_prepared_manifest_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path)
    original = MemoryRecoveryManifest.append

    def crash(self: MemoryRecoveryManifest, **kwargs: object):
        original(self, **kwargs)
        raise RuntimeError("injected")

    monkeypatch.setattr(MemoryRecoveryManifest, "append", crash)
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    monkeypatch.setattr(MemoryRecoveryManifest, "append", original)
    record = _prepare(tmp_path)
    assert record.phase == "CONVERSION_PREPARED"


def test_prepared_replay_rejects_copied_stat_identity(tmp_path: Path) -> None:
    memory, _ = _workspace(tmp_path)
    _prepare(tmp_path)
    history = memory / "HISTORY.md"
    data = history.read_bytes()
    history.unlink()
    history.write_bytes(data)
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)


def test_v1_evidence_quarantines_without_marker_mutation(tmp_path: Path) -> None:
    memory, _ = _workspace(tmp_path)
    # Derive the v1 identity via the same detached receipt capture, without modifying history.
    from nanobot.agent import memory_recovery as receipt_codec
    from nanobot.agent import memory_recovery_transaction as transaction

    with storage.open_workspace_memory(tmp_path) as authority, authority.operation() as operation:
        raw = authority.snapshot_target(".consolidation.pending.json").content
        assert raw is not None
        receipt = receipt_codec._validate_payload(receipt_codec._decode(raw), raw)
        captured = transaction.capture_transaction_payload(
            receipt,
            workspace_id=_WORKSPACE_ID,
            workspace_identity=operation.workspace_identity,
        )
        v1 = capture_conversion_payload(captured.identity)
        MemoryRecoveryManifest(
            authority.open_evidence_namespace("conversions", v1.identity)
        ).create(v1.to_payload())
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    assert (memory / "HISTORY.md").read_bytes() == b"# Existing\n\n"


def test_crash_before_marker_stage_is_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, _ = _workspace(tmp_path)
    original = marker_stage.stage_history_marker

    def crash(*args: object):
        raise RuntimeError("injected")

    monkeypatch.setattr(marker_stage, "stage_history_marker", crash)
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    assert (memory / "HISTORY.md").read_bytes() == b"# Existing\n\n"
    monkeypatch.setattr(marker_stage, "stage_history_marker", original)
    assert _prepare(tmp_path).phase == "CONVERSION_PREPARED"


def test_crash_after_captured_manifest_write_is_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path)
    original = MemoryRecoveryManifest.create
    triggered = False

    def crash(self: MemoryRecoveryManifest, payload: dict[str, object]):
        nonlocal triggered
        result = original(self, payload)
        if not triggered:
            triggered = True
            raise RuntimeError("injected")
        return result

    monkeypatch.setattr(MemoryRecoveryManifest, "create", crash)
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    monkeypatch.setattr(MemoryRecoveryManifest, "create", original)
    assert _prepare(tmp_path).phase == "CONVERSION_PREPARED"


def test_existing_blob_conflict_fails_before_marker(tmp_path: Path) -> None:
    memory, raw = _workspace(tmp_path)
    digest = hashlib.sha256(raw).hexdigest()
    with storage.open_workspace_memory(tmp_path) as authority:
        blob_path = authority.blob_directory / digest
        blob_path.write_bytes(b"conflicting bytes")
        blob_path.chmod(0o600)
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    assert (memory / "HISTORY.md").read_bytes() == b"# Existing\n\n"


def test_replay_rejects_changed_target_bytes(tmp_path: Path) -> None:
    memory, _ = _workspace(tmp_path, jsonl=b"")
    _prepare(tmp_path)
    (memory / "history.jsonl").write_bytes(b"changed")
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)


def test_missing_receipt_fails_without_marker(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "HISTORY.md").write_bytes(b"# Existing\n\n")
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    assert (memory / "HISTORY.md").read_bytes() == b"# Existing\n\n"


def test_later_conversion_phase_quarantines_without_target_mutation(tmp_path: Path) -> None:
    memory, _ = _workspace(tmp_path)
    prepared = _prepare(tmp_path)
    source_before = (memory / "HISTORY.md").read_bytes()
    with storage.open_workspace_memory(tmp_path) as authority:
        manifest = MemoryRecoveryManifest(
            authority.open_evidence_namespace("conversions", prepared.identity)
        )
        manifest.append(predecessor_digest=prepared.digest, phase="JSONL_APPLIED", payload={})
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    assert (memory / "HISTORY.md").read_bytes() == source_before
    assert not (memory / "history.jsonl").exists()


def test_later_transaction_phase_quarantines_before_marker(tmp_path: Path) -> None:
    memory, raw = _workspace(tmp_path)
    from nanobot.agent import memory_recovery as receipt_codec
    from nanobot.agent import memory_recovery_transaction as transaction

    with storage.open_workspace_memory(tmp_path) as authority, authority.operation() as operation:
        receipt = receipt_codec._validate_payload(receipt_codec._decode(raw), raw)
        captured = transaction.capture_transaction_payload(
            receipt,
            workspace_id=_WORKSPACE_ID,
            workspace_identity=operation.workspace_identity,
        )
        manifest = MemoryRecoveryManifest(
            authority.open_evidence_namespace("transactions", captured.identity)
        )
        record = manifest.create(captured.to_payload())
        manifest.append(predecessor_digest=record.digest, phase="PREPARED", payload={})
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    assert (memory / "HISTORY.md").read_bytes() == b"# Existing\n\n"


def test_canonical_row_is_published_as_standalone_blob(tmp_path: Path) -> None:
    _workspace(tmp_path)
    record = _prepare(tmp_path)
    row = record.payload["canonical_row"]
    with storage.open_workspace_memory(tmp_path) as authority:
        content = authority.read_blob(row["blob_digest"])
    assert len(content) == row["size"]
    assert json.loads(content)["cursor"] == row["cursor"]


@pytest.mark.parametrize(
    "claim", ["fallback_timestamp", "tail_frame", "tail_frame_offset", "canonical_row"]
)
def test_replay_rederives_canonical_prepared_claims(tmp_path: Path, claim: str) -> None:
    memory, _ = _workspace(tmp_path)
    record = _prepare(tmp_path)
    original_source = (memory / "HISTORY.md").read_bytes()
    with storage.open_workspace_memory(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("conversions", record.identity)
        slot = namespace.directory / f"next-{record.predecessor_digest}.json"
        envelope = json.loads(slot.read_bytes())
        if claim == "fallback_timestamp":
            envelope["payload"][claim] = "2026-09-24 12:00"
        elif claim == "tail_frame":
            envelope["payload"][claim]["digest"] = "a" * 64
        elif claim == "tail_frame_offset":
            envelope["payload"]["tail_frame"]["offset"] += 1
            envelope["payload"]["tail_frame"]["size"] -= 1
        else:
            envelope["payload"][claim]["blob_digest"] = "b" * 64
        slot.write_bytes(
            json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
    with pytest.raises(runtime.ConversionPreparationError):
        _prepare(tmp_path)
    assert (memory / "HISTORY.md").read_bytes() == original_source
    assert not (memory / "history.jsonl").exists()
