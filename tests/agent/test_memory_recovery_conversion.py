from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime

import pytest

from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery_conversion import (
    CapturedConversionPayload,
    ConversionError,
    _check_record_bounds,
    capture_conversion_payload,
    complete_conversion_payload,
    cursors_applied_conversion_payload,
    decode_conversion_payload,
    jsonl_applied_conversion_payload,
    legacy_finalized_conversion_payload,
    prepare_conversion_payload,
)
from nanobot.agent.memory_recovery_manifest import (
    MAX_RECORD_BYTES,
    MemoryRecoveryManifest,
    derive_manifest_identity,
)

_SHA_EMPTY = hashlib.sha256(b"").hexdigest()
_DEFAULT_STAT_FIELDS = {
    "device": 1,
    "inode": 2,
    "mode": 0o100644,
    "uid": 501,
    "gid": 20,
    "links": 1,
    "size": 0,
    "mtime_ns": 1_700_000_000_000_000_000,
    "ctime_ns": 1_700_000_000_000_000_001,
}


def _file_descriptor(content: bytes, *, exists: bool = True) -> dict[str, object]:
    if not exists:
        return {"exists": False, "size": 0, "digest": None, "stat": None, "blob_digest": None}
    digest = hashlib.sha256(content).hexdigest()
    stat = dict(_DEFAULT_STAT_FIELDS, size=len(content))
    return {
        "exists": True,
        "size": len(content),
        "digest": digest,
        "stat": stat,
        "blob_digest": digest,
    }


def _capture() -> CapturedConversionPayload:
    return capture_conversion_payload("a" * 64)


def _backup(candidates: list[tuple[str, bool]]) -> dict[str, object]:
    return {
        "basename": candidates[-1][0],
        "candidates": [
            {"basename": basename, "exists": exists} for basename, exists in candidates
        ],
    }


def _prepared_inputs(
    *,
    source_bytes: bytes = b"[2025-04-03 11:22] hello\n\nworld",
    history: bytes | None = b"",
    cursor: bytes | None = None,
    dream_cursor: bytes | None = None,
) -> dict[str, object]:
    targets = {
        "history.jsonl": _file_descriptor(history) if history is not None else _file_descriptor(b"", exists=False),
        ".cursor": _file_descriptor(cursor) if cursor is not None else _file_descriptor(b"", exists=False),
        ".dream_cursor": _file_descriptor(dream_cursor) if dream_cursor is not None else _file_descriptor(b"", exists=False),
    }
    return {
        "source_bytes": source_bytes,
        "source": _file_descriptor(source_bytes),
        "targets": targets,
        "target_contents": {
            "history.jsonl": history,
            ".cursor": cursor,
            ".dream_cursor": dream_cursor,
        },
        "backup": _backup([("HISTORY.md.bak", True), ("HISTORY.md.bak.2", False)]),
        "fallback_timestamp": "2025-04-03 11:22",
    }


def _prepare(**overrides: object):
    values = _prepared_inputs()
    values.update(overrides)
    return prepare_conversion_payload(_capture(), disposition="convert", **values)


def test_captured_payload_is_exact_deterministic_and_workspace_local() -> None:
    captured = _capture()
    expected = {
        "schema_version": 1,
        "conversion_policy": "legacy_history_v1",
        "receipt_transaction_identity": "a" * 64,
    }

    assert captured.to_payload() == expected
    assert captured.identity == derive_manifest_identity("conversions", expected)
    assert captured.identity == _capture().identity
    payload = captured.to_payload()
    payload["receipt_transaction_identity"] = "b" * 64
    assert captured.to_payload() == expected


def test_conversion_record_rejects_payload_over_record_byte_limit() -> None:
    captured = _capture()

    with pytest.raises(ConversionError):
        _check_record_bounds(
            {"padding": "x" * MAX_RECORD_BYTES}, "CAPTURED", captured.identity
        )


@pytest.mark.parametrize("bad", ["A" * 64, "0" * 63, "0" * 65, 12, True])
def test_capture_rejects_non_sha256_receipt_identity(bad: object) -> None:
    with pytest.raises(ConversionError):
        capture_conversion_payload(bad)


def test_convert_preview_matches_legacy_jsonl_serialization_and_cursors() -> None:
    source = b"[2025-04-03 11:22] hello\n\nworld\n\n[RAW] quote \" and \xc3\xa9"
    prepared = _prepare(source_bytes=source, source=_file_descriptor(source))
    payload = prepared.to_payload()
    expected_rows = [
        {"cursor": 1, "timestamp": "2025-04-03 11:22", "content": "hello"},
        {"cursor": 2, "timestamp": "2025-04-03 11:22", "content": "world"},
        {"cursor": 3, "timestamp": "2025-04-03 11:22", "content": "[RAW] quote \" and é"},
    ]
    expected_jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in expected_rows).encode()
    expected_cursor = b"3"

    assert payload["row_count"] == 3
    assert payload["final_legacy_cursor"] == 3
    assert payload["fallback_timestamp"] == "2025-04-03 11:22"
    assert payload["outputs"] == {
        "history.jsonl": {
            "exists": True,
            "size": len(expected_jsonl),
            "blob_digest": hashlib.sha256(expected_jsonl).hexdigest(),
        },
        ".cursor": {
            "exists": True,
            "size": len(expected_cursor),
            "blob_digest": hashlib.sha256(expected_cursor).hexdigest(),
        },
        ".dream_cursor": {
            "exists": True,
            "size": len(expected_cursor),
            "blob_digest": hashlib.sha256(expected_cursor).hexdigest(),
        },
    }
    assert payload["backup"]["basename"] == "HISTORY.md.bak.2"
    assert "hello" not in json.dumps(payload)
    assert "é" not in json.dumps(payload)


def test_convert_freezes_first_free_backup_with_collision_evidence() -> None:
    backup = _backup(
        [
            ("HISTORY.md.bak", True),
            ("HISTORY.md.bak.2", True),
            ("HISTORY.md.bak.3", False),
        ]
    )
    payload = _prepare(backup=backup).to_payload()
    assert payload["backup"] == backup


@pytest.mark.parametrize(
    "backup",
    [
        {"basename": "../HISTORY.md.bak", "candidates": [{"basename": "../HISTORY.md.bak", "exists": False}]},
        {"basename": "HISTORY.md.bak.1", "candidates": [{"basename": "HISTORY.md.bak.1", "exists": False}]},
        {"basename": "HISTORY.md.bak.2", "candidates": [{"basename": "HISTORY.md.bak.2", "exists": False}]},
        {"basename": "HISTORY.md.bak.2", "candidates": [{"basename": "HISTORY.md.bak", "exists": False}, {"basename": "HISTORY.md.bak.2", "exists": False}]},
        {"basename": "HISTORY.md.bak.3", "candidates": [{"basename": "HISTORY.md.bak", "exists": True}, {"basename": "HISTORY.md.bak.2", "exists": False}, {"basename": "HISTORY.md.bak.3", "exists": False}]},
    ],
)
def test_convert_rejects_invalid_or_nonfirst_free_backup_evidence(backup: object) -> None:
    with pytest.raises(ConversionError):
        _prepare(backup=backup)


def test_preserve_nonempty_jsonl_has_no_conversion_or_mutation_outputs() -> None:
    captured = _capture()
    target_bytes = b'{"cursor":1,"timestamp":"2025-01-01 00:00","content":"kept"}\n'
    targets = {
        "history.jsonl": _file_descriptor(target_bytes),
        ".cursor": _file_descriptor(b"1"),
        ".dream_cursor": _file_descriptor(b"1"),
    }
    prepared = prepare_conversion_payload(
        captured,
        disposition="preserve_nonempty_jsonl",
        source=_file_descriptor(b"legacy bytes"),
        source_bytes=None,
        targets=targets,
        backup=None,
        fallback_timestamp=None,
    )
    payload = prepared.to_payload()

    assert payload["disposition"] == "preserve_nonempty_jsonl"
    assert payload["backup"] is None
    assert payload["fallback_timestamp"] is None
    assert payload["row_count"] is None
    assert payload["final_legacy_cursor"] is None
    assert payload["outputs"] is None
    assert payload["sidecar_counters"] is None
    assert decode_conversion_payload(
        payload, "CONVERSION_PREPARED", captured=captured
    ) == prepared


def test_preserve_requires_present_legacy_source_for_builder_and_decoder() -> None:
    target_bytes = b"existing jsonl\n"
    targets = {
        "history.jsonl": _file_descriptor(target_bytes),
        ".cursor": _file_descriptor(b"1"),
        ".dream_cursor": _file_descriptor(b"1"),
    }
    with pytest.raises(ConversionError):
        prepare_conversion_payload(
            _capture(),
            disposition="preserve_nonempty_jsonl",
            source=_file_descriptor(b"", exists=False),
            source_bytes=None,
            targets=targets,
            backup=None,
            fallback_timestamp=None,
        )

    captured = _capture()
    prepared = prepare_conversion_payload(
        captured,
        disposition="preserve_nonempty_jsonl",
        source=_file_descriptor(b"legacy source"),
        source_bytes=None,
        targets=targets,
        backup=None,
        fallback_timestamp=None,
    )
    payload = prepared.to_payload()
    payload["source"] = _file_descriptor(b"", exists=False)
    with pytest.raises(ConversionError):
        decode_conversion_payload(payload, "CONVERSION_PREPARED", captured=captured)


@pytest.mark.parametrize("history", [None, b""])
def test_preserve_requires_existing_nonempty_history(history: bytes | None) -> None:
    targets = {
        "history.jsonl": _file_descriptor(b"", exists=history is not None),
        ".cursor": _file_descriptor(b"", exists=False),
        ".dream_cursor": _file_descriptor(b"", exists=False),
    }
    with pytest.raises(ConversionError):
        prepare_conversion_payload(
            _capture(),
            disposition="preserve_nonempty_jsonl",
            source=_file_descriptor(b"legacy bytes"),
            source_bytes=None,
            targets=targets,
            backup=None,
            fallback_timestamp=None,
        )


def test_empty_conversion_preserves_absent_and_empty_old_targets() -> None:
    values = _prepared_inputs(
        source_bytes=b" \r\n\t ",
        history=None,
        cursor=None,
        dream_cursor=b"0",
    )
    values["backup"] = _backup([("HISTORY.md.bak", False)])
    prepared = _prepare(**values)
    payload = prepared.to_payload()
    assert payload["row_count"] == 0
    assert payload["final_legacy_cursor"] == 0
    assert payload["sidecar_counters"] == {".cursor": 0, ".dream_cursor": 0}
    assert payload["outputs"] == {
        "history.jsonl": {"exists": False, "size": 0, "blob_digest": None},
        ".cursor": {"exists": False, "size": 0, "blob_digest": None},
        ".dream_cursor": {
            "exists": True,
            "size": 1,
            "blob_digest": hashlib.sha256(b"0").hexdigest(),
        },
    }
    targets = payload["targets"]
    assert targets["history.jsonl"]["exists"] is False
    assert targets[".dream_cursor"]["exists"] is True
    assert targets[".dream_cursor"]["digest"] == hashlib.sha256(b"0").hexdigest()


def test_convert_rejects_sidecars_that_would_be_lowered_or_are_inconsistent() -> None:
    for cursor, dream_cursor in (
        (b"1", b"4"),
        (b"4", b"1"),
        (b"bad", b"0"),
        (b"1", b""),
        (b"1", b"2"),
    ):
        values = _prepared_inputs(cursor=cursor, dream_cursor=dream_cursor)
        with pytest.raises(ConversionError):
            _prepare(**values)


@pytest.mark.parametrize(
    ("cursor", "dream_cursor"),
    [(b"bad", b"0"), (b"0", b""), (b"1", b"0"), (b"0", b"1")],
)
def test_zero_row_conversion_rejects_malformed_or_nonzero_sidecars(
    cursor: bytes,
    dream_cursor: bytes,
) -> None:
    values = _prepared_inputs(source_bytes=b" \n", cursor=cursor, dream_cursor=dream_cursor)
    values["source"] = _file_descriptor(values["source_bytes"])
    with pytest.raises(ConversionError):
        _prepare(**values)


def test_convert_records_valid_old_sidecar_counters() -> None:
    prepared = _prepare(**_prepared_inputs(cursor=b"1", dream_cursor=b"0"))
    payload = prepared.to_payload()
    assert payload["sidecar_counters"] == {".cursor": 1, ".dream_cursor": 0}


def test_invalid_utf8_is_previewed_with_replacement_and_raw_source_digest() -> None:
    source = b"[2025-04-03 11:22] invalid \xff byte"
    payload = _prepare(source_bytes=source, source=_file_descriptor(source)).to_payload()
    output = payload["outputs"]["history.jsonl"]
    row = {"cursor": 1, "timestamp": "2025-04-03 11:22", "content": "invalid \ufffd byte"}
    expected = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    assert payload["source"]["digest"] == hashlib.sha256(source).hexdigest()
    assert output["blob_digest"] == hashlib.sha256(expected).hexdigest()


def test_source_and_target_descriptors_preserve_full_stat_and_detach_inputs() -> None:
    values = _prepared_inputs(cursor=b"0", dream_cursor=b"0")
    source = values["source"]
    targets = values["targets"]
    prepared = _prepare(**values)
    source["stat"]["inode"] = 0
    targets[".cursor"]["stat"]["mtime_ns"] = 0
    payload = prepared.to_payload()

    assert payload["source"]["stat"]["inode"] == 2
    assert payload["targets"][".cursor"]["stat"]["mtime_ns"] == _DEFAULT_STAT_FIELDS["mtime_ns"]
    payload["targets"][".cursor"]["stat"]["inode"] = 999
    assert prepared.to_payload()["targets"][".cursor"]["stat"]["inode"] == 2


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("size", True),
        ("digest", "A" * 64),
        ("blob_digest", "b" * 63),
        ("stat", None),
    ],
)
def test_invalid_source_descriptors_are_rejected(field: str, replacement: object) -> None:
    values = _prepared_inputs()
    values["source"] = copy.deepcopy(values["source"])
    values["source"][field] = replacement
    with pytest.raises(ConversionError):
        _prepare(**values)


def test_inconsistent_source_bytes_and_file_stat_are_rejected() -> None:
    values = _prepared_inputs()
    values["source"] = _file_descriptor(b"different source")
    with pytest.raises(ConversionError):
        _prepare(**values)


def test_old_empty_is_distinct_from_absent_and_requires_sha_empty() -> None:
    values = _prepared_inputs(history=b"")
    payload = _prepare(**values).to_payload()
    old = payload["targets"]["history.jsonl"]
    assert old == _file_descriptor(b"")
    values["targets"]["history.jsonl"]["digest"] = "0" * 64
    with pytest.raises(ConversionError):
        _prepare(**values)


def test_roundtrip_requires_captured_predecessor_and_rejects_phase_mismatch() -> None:
    captured = _capture()
    prepared = _prepare()
    payload = prepared.to_payload()
    decoded_capture = decode_conversion_payload(captured.to_payload(), "CAPTURED")
    decoded_prepared = decode_conversion_payload(
        payload, "CONVERSION_PREPARED", captured=decoded_capture
    )

    assert decoded_capture == captured
    assert decoded_prepared == prepared
    with pytest.raises(ConversionError):
        decode_conversion_payload(payload, "CONVERSION_PREPARED")
    with pytest.raises(ConversionError):
        decode_conversion_payload(captured.to_payload(), "CAPTURED", captured=captured)
    with pytest.raises(ConversionError):
        decode_conversion_payload(payload, "CAPTURED")


def test_decoder_rejects_semantic_contradictions_and_extra_fields() -> None:
    captured = _capture()
    payload = _prepare().to_payload()
    payload["conversion_identity"] = "b" * 64
    with pytest.raises(ConversionError):
        decode_conversion_payload(payload, "CONVERSION_PREPARED", captured=captured)

    payload = _prepare().to_payload()
    payload["row_count"] = 0
    with pytest.raises(ConversionError):
        decode_conversion_payload(payload, "CONVERSION_PREPARED", captured=captured)

    payload = _prepare().to_payload()
    payload["unknown"] = 1
    with pytest.raises(ConversionError):
        decode_conversion_payload(payload, "CONVERSION_PREPARED", captured=captured)


def test_generated_jsonl_must_fit_legacy_history_byte_limit() -> None:
    source = b"\x00" * (11 * 1024 * 1024)
    with pytest.raises(ConversionError):
        _prepare(source_bytes=source, source=_file_descriptor(source))


def test_payload_roundtrip_accepts_only_bounded_json_and_exact_types() -> None:
    captured = _capture()
    prepared = _prepare()
    assert decode_conversion_payload(
        json.dumps(prepared.to_payload(), ensure_ascii=False).encode(),
        "CONVERSION_PREPARED",
        captured=captured,
    ) == prepared

    payload = prepared.to_payload()
    payload["schema_version"] = True
    with pytest.raises(ConversionError):
        decode_conversion_payload(payload, "CONVERSION_PREPARED", captured=captured)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda descriptor: descriptor.update(exists=False),
        lambda descriptor: descriptor.update(size=2),
        lambda descriptor: descriptor.update(blob_digest=None),
        lambda descriptor: descriptor["stat"].update(size=2),
        lambda descriptor: descriptor["stat"].update(links=0),
        lambda descriptor: descriptor["stat"].update(mtime_ns=True),
        lambda descriptor: descriptor["stat"].update(device=2**63),
    ],
)
def test_invalid_old_target_descriptors_are_rejected(mutate) -> None:
    values = _prepared_inputs(cursor=b"0", dream_cursor=b"0")
    mutate(values["targets"][".cursor"])
    with pytest.raises(ConversionError):
        _prepare(**values)


def test_conversion_decoder_rejects_duplicate_json_fields_and_invalid_phase_type() -> None:
    captured = _capture()
    captured_json = json.dumps(captured.to_payload())
    duplicate = captured_json[:-1] + ',"schema_version":1}'
    with pytest.raises(ConversionError):
        decode_conversion_payload(duplicate.encode(), "CAPTURED")
    with pytest.raises(ConversionError):
        decode_conversion_payload(captured.to_payload(), True)


def test_conversion_decoder_rejects_inconsistent_output_cursors() -> None:
    captured = _capture()
    payload = _prepare().to_payload()
    payload["outputs"][".cursor"]["blob_digest"] = "0" * 64
    with pytest.raises(ConversionError):
        decode_conversion_payload(payload, "CONVERSION_PREPARED", captured=captured)


def test_convert_rejects_nonempty_history_instead_of_claiming_overwrite() -> None:
    history = b"existing jsonl\n"
    values = _prepared_inputs(history=history)
    with pytest.raises(ConversionError):
        _prepare(**values)


def test_fallback_timestamp_must_be_a_fixed_local_minute_value() -> None:
    with pytest.raises(ConversionError):
        _prepare(fallback_timestamp=datetime.now())
    with pytest.raises(ConversionError):
        _prepare(fallback_timestamp="2025-13-03 11:22")


def _prepare_preserve():
    captured = _capture()
    history = b'{"cursor":4,"timestamp":"2025-04-03 11:22","content":"kept"}\n'
    targets = {
        "history.jsonl": _file_descriptor(history),
        ".cursor": _file_descriptor(b"4"),
        ".dream_cursor": _file_descriptor(b"3"),
    }
    prepared = prepare_conversion_payload(
        captured,
        disposition="preserve_nonempty_jsonl",
        source=_file_descriptor(b"legacy archive"),
        source_bytes=None,
        targets=targets,
        backup=None,
        fallback_timestamp=None,
    )
    return captured, prepared


@pytest.mark.parametrize(
    "disposition", ["convert", "convert_zero_rows", "preserve_nonempty_jsonl"]
)
def test_all_conversion_phases_roundtrip_with_real_manifest_digests(
    tmp_path, disposition: str
) -> None:
    if disposition == "convert":
        captured = _capture()
        prepared = _prepare()
    elif disposition == "convert_zero_rows":
        values = _prepared_inputs(
            source_bytes=b" \r\n",
            history=None,
            cursor=None,
            dream_cursor=b"0",
        )
        values["backup"] = _backup([("HISTORY.md.bak", False)])
        captured = _capture()
        prepared = _prepare(**values)
    else:
        captured, prepared = _prepare_preserve()

    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    namespace = authority.open_evidence_namespace("conversions", captured.identity)
    manifest = MemoryRecoveryManifest(namespace)
    try:
        record = manifest.create(captured.to_payload())
        record = manifest.append(
            predecessor_digest=record.digest,
            phase="CONVERSION_PREPARED",
            payload=prepared.to_payload(),
        )
        prepared_record_digest = record.digest

        jsonl = jsonl_applied_conversion_payload(captured, prepared)
        record = manifest.append(
            predecessor_digest=record.digest,
            phase="JSONL_APPLIED",
            payload=jsonl.to_payload(),
        )
        jsonl_record_digest = record.digest
        if disposition == "convert_zero_rows":
            assert jsonl.history_target == prepared.outputs["history.jsonl"]
            assert not jsonl.history_target.exists

        cursors = cursors_applied_conversion_payload(captured, prepared, jsonl)
        record = manifest.append(
            predecessor_digest=record.digest,
            phase="CURSORS_APPLIED",
            payload=cursors.to_payload(),
        )
        cursors_record_digest = record.digest

        finalized = legacy_finalized_conversion_payload(
            captured,
            prepared,
            jsonl,
            cursors,
            prepared_record_digest=prepared_record_digest,
            jsonl_applied_record_digest=jsonl_record_digest,
            cursors_applied_record_digest=cursors_record_digest,
        )
        record = manifest.append(
            predecessor_digest=record.digest,
            phase="LEGACY_FINALIZED",
            payload=finalized.to_payload(),
        )
        finalized_record_digest = record.digest

        complete = complete_conversion_payload(
            captured,
            prepared,
            jsonl,
            cursors,
            finalized,
            finalized_record_digest=finalized_record_digest,
        )
        record = manifest.append(
            predecessor_digest=record.digest,
            phase="COMPLETE",
            payload=complete.to_payload(),
        )

        records = manifest.load()
        assert [item.phase for item in records] == [
            "CAPTURED",
            "CONVERSION_PREPARED",
            "JSONL_APPLIED",
            "CURSORS_APPLIED",
            "LEGACY_FINALIZED",
            "COMPLETE",
        ]
        assert record.sequence == 5
        assert decode_conversion_payload(
            jsonl.to_payload(), "JSONL_APPLIED", captured=captured, prepared=prepared
        ) == jsonl
        assert decode_conversion_payload(
            cursors.to_payload(),
            "CURSORS_APPLIED",
            captured=captured,
            prepared=prepared,
            jsonl_applied=jsonl,
        ) == cursors
        assert decode_conversion_payload(
            finalized.to_payload(),
            "LEGACY_FINALIZED",
            captured=captured,
            prepared=prepared,
            jsonl_applied=jsonl,
            cursors_applied=cursors,
            prepared_record_digest=prepared_record_digest,
            jsonl_applied_record_digest=jsonl_record_digest,
            cursors_applied_record_digest=cursors_record_digest,
        ) == finalized
        assert decode_conversion_payload(
            complete.to_payload(),
            "COMPLETE",
            captured=captured,
            prepared=prepared,
            jsonl_applied=jsonl,
            cursors_applied=cursors,
            legacy_finalized=finalized,
            finalized_record_digest=finalized_record_digest,
        ) == complete
    finally:
        namespace.close()
        authority.close()


def test_conversion_phase_projection_parity_and_disposition_fields() -> None:
    captured, prepared = _prepare_preserve()
    jsonl = jsonl_applied_conversion_payload(captured, prepared)
    cursors = cursors_applied_conversion_payload(captured, prepared, jsonl)
    finalized = legacy_finalized_conversion_payload(
        captured,
        prepared,
        jsonl,
        cursors,
        prepared_record_digest="1" * 64,
        jsonl_applied_record_digest="2" * 64,
        cursors_applied_record_digest="3" * 64,
    )

    expected_fields = (
        {"schema_version", "conversion_identity", "disposition", "history_target"},
        {
            "schema_version",
            "conversion_identity",
            "disposition",
            "history_target",
            "cursor_target",
            "dream_cursor_target",
        },
        {
            "schema_version",
            "conversion_identity",
            "disposition",
            "prepared_record_digest",
            "jsonl_applied_record_digest",
            "cursors_applied_record_digest",
            "targets",
            "source",
            "backup_basename",
            "backup",
        },
    )
    for evidence, fields in zip((jsonl, cursors, finalized), expected_fields, strict=True):
        payload = evidence.to_payload()
        assert set(payload) == fields
        assert payload["schema_version"] == 1
        assert payload["conversion_identity"] == captured.identity
        assert payload["disposition"] == "preserve_nonempty_jsonl"
    assert jsonl.to_payload()["history_target"] == _file_descriptor_to_new(
        prepared.targets["history.jsonl"]
    )
    assert cursors.history_target == jsonl.history_target
    cursors_payload = cursors.to_payload()
    assert cursors_payload["cursor_target"] == _file_descriptor_to_new(
        prepared.targets[".cursor"]
    )
    assert cursors_payload["dream_cursor_target"] == _file_descriptor_to_new(
        prepared.targets[".dream_cursor"]
    )
    assert finalized.to_payload()["source"] == _file_descriptor_to_new(prepared.source)
    assert finalized.backup_basename is None
    assert finalized.backup is None
    detached = finalized.to_payload()
    detached["targets"]["history.jsonl"]["size"] += 1
    assert finalized.to_payload()["targets"]["history.jsonl"]["size"] == prepared.targets[
        "history.jsonl"
    ].size


def _file_descriptor_to_new(old):
    return {"exists": old.exists, "size": old.size, "blob_digest": old.blob_digest}


def test_conversion_phase_decoders_reject_forged_parity_and_predecessors() -> None:
    captured = _capture()
    prepared = _prepare()
    jsonl = jsonl_applied_conversion_payload(captured, prepared)
    cursors = cursors_applied_conversion_payload(captured, prepared, jsonl)
    wrong_target = jsonl.to_payload()
    wrong_target["history_target"]["blob_digest"] = "0" * 64
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            wrong_target, "JSONL_APPLIED", captured=captured, prepared=prepared
        )
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            cursors.to_payload(),
            "CURSORS_APPLIED",
            captured=captured,
            prepared=prepared,
        )

    wrong_history = cursors.to_payload()
    wrong_history["history_target"]["size"] += 1
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            wrong_history,
            "CURSORS_APPLIED",
            captured=captured,
            prepared=prepared,
            jsonl_applied=jsonl,
        )

    forged_captured = capture_conversion_payload("b" * 64)
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            jsonl.to_payload(),
            "JSONL_APPLIED",
            captured=forged_captured,
            prepared=prepared,
        )


def test_conversion_finalization_rejects_bad_record_digests_and_backup_claims() -> None:
    captured = _capture()
    prepared = _prepare()
    jsonl = jsonl_applied_conversion_payload(captured, prepared)
    cursors = cursors_applied_conversion_payload(captured, prepared, jsonl)
    args = {
        "prepared_record_digest": "1" * 64,
        "jsonl_applied_record_digest": "2" * 64,
        "cursors_applied_record_digest": "3" * 64,
    }
    finalized = legacy_finalized_conversion_payload(
        captured, prepared, jsonl, cursors, **args
    )
    payload = finalized.to_payload()
    payload["backup"]["size"] += 1
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            payload,
            "LEGACY_FINALIZED",
            captured=captured,
            prepared=prepared,
            jsonl_applied=jsonl,
            cursors_applied=cursors,
            **args,
        )

    args["prepared_record_digest"] = "A" * 64
    with pytest.raises(ConversionError):
        legacy_finalized_conversion_payload(captured, prepared, jsonl, cursors, **args)


def test_finalized_decoder_requires_real_manifest_predecessor_digests() -> None:
    captured = _capture()
    prepared = _prepare()
    jsonl = jsonl_applied_conversion_payload(captured, prepared)
    cursors = cursors_applied_conversion_payload(captured, prepared, jsonl)
    digests = {
        "prepared_record_digest": "1" * 64,
        "jsonl_applied_record_digest": "2" * 64,
        "cursors_applied_record_digest": "3" * 64,
    }
    finalized = legacy_finalized_conversion_payload(
        captured, prepared, jsonl, cursors, **digests
    )
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            finalized.to_payload(),
            "LEGACY_FINALIZED",
            captured=captured,
            prepared=prepared,
            jsonl_applied=jsonl,
            cursors_applied=cursors,
        )
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            finalized.to_payload(),
            "LEGACY_FINALIZED",
            captured=captured,
            prepared=prepared,
            jsonl_applied=jsonl,
            cursors_applied=cursors,
            prepared_record_digest="4" * 64,
            jsonl_applied_record_digest=digests["jsonl_applied_record_digest"],
            cursors_applied_record_digest=digests["cursors_applied_record_digest"],
        )



def test_conversion_complete_rejects_forged_finalized_digest_and_targets() -> None:
    captured = _capture()
    prepared = _prepare()
    jsonl = jsonl_applied_conversion_payload(captured, prepared)
    cursors = cursors_applied_conversion_payload(captured, prepared, jsonl)
    digests = {
        "prepared_record_digest": "1" * 64,
        "jsonl_applied_record_digest": "2" * 64,
        "cursors_applied_record_digest": "3" * 64,
    }
    finalized = legacy_finalized_conversion_payload(
        captured, prepared, jsonl, cursors, **digests
    )
    complete = complete_conversion_payload(
        captured,
        prepared,
        jsonl,
        cursors,
        finalized,
        finalized_record_digest="4" * 64,
    )
    forged = complete.to_payload()
    forged["source"] = {"exists": True, "size": 0, "blob_digest": "a" * 64}
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            forged,
            "COMPLETE",
            captured=captured,
            prepared=prepared,
            jsonl_applied=jsonl,
            cursors_applied=cursors,
            legacy_finalized=finalized,
            finalized_record_digest="4" * 64,
        )


def test_conversion_phase_payloads_reject_extra_fields_and_duplicate_keys() -> None:
    captured = _capture()
    prepared = _prepare()
    jsonl = jsonl_applied_conversion_payload(captured, prepared)
    payload = jsonl.to_payload()
    payload["extra"] = True
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            payload, "JSONL_APPLIED", captured=captured, prepared=prepared
        )
    duplicated = json.dumps(jsonl.to_payload())[:-1] + ',"disposition":"convert"}'
    with pytest.raises(ConversionError):
        decode_conversion_payload(
            duplicated.encode(), "JSONL_APPLIED", captured=captured, prepared=prepared
        )
