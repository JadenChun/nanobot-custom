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


def _v2_receipt(entry: str = "[2025-04-03 11:22] first\n\n[2025-04-04 12:34] second"):
    from types import MappingProxyType

    from nanobot.agent.memory_recovery import MemoryReceipt
    from nanobot.agent.memory_recovery_manifest import ManifestRecord
    from nanobot.agent.memory_recovery_transaction import capture_transaction_payload

    values = {
        "version": 1,
        "chunk_digest": hashlib.sha256(b"chunk").hexdigest(),
        "history_marker_id": "receipt-marker",
        "history_entry": entry,
        "memory_output": None,
        "base_memory_digest": hashlib.sha256(b"old").hexdigest(),
        "output_memory_digest": None,
        "reference_only": True,
        "created_at": "2026-09-24T00:00:00Z",
        "session_key": "session",
        "start_offset": 0,
        "end_offset": 1,
        "session_generation": 0,
    }
    raw = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    receipt = MemoryReceipt(
        values["version"], values["chunk_digest"], values["history_marker_id"],
        entry, None, values["base_memory_digest"], None, True,
        values["created_at"], values["session_key"], 0, 1, 0,
        raw_fields=MappingProxyType(values), raw_bytes=raw,
    )
    transaction = capture_transaction_payload(receipt, workspace_id="a" * 32, workspace_identity=(1, 2))
    payload = transaction.to_payload()
    identity = transaction.identity
    record_bytes = json.dumps({
        "version": 1, "kind": "transactions", "identity": identity, "sequence": 0,
        "predecessor_digest": None, "phase": "CAPTURED", "payload": payload,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    record = ManifestRecord(1, "transactions", identity, 0, None, "CAPTURED",
                            record_bytes, hashlib.sha256(record_bytes).hexdigest(),
                            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return receipt, transaction, record


def _v2_prepare(entry: str = "[2025-04-03 11:22] first\n\n[2025-04-04 12:34] second"):
    from nanobot.agent.memory_recovery_conversion import capture_receipt_tail_conversion_payload

    receipt, transaction, record = _v2_receipt(entry)
    captured = capture_receipt_tail_conversion_payload(transaction.identity)
    source = b"[2025-01-01 10:00] prefix\n\n<!-- nanobot-consolidation:receipt-marker -->\n" + entry.rstrip().encode() + b"\n\n"
    values = _prepared_inputs(source_bytes=source)
    prepared = prepare_conversion_payload(
        captured, disposition="convert", receipt=receipt,
        transaction_captured=transaction, transaction_captured_record=record, **values,
    )
    return captured, prepared, values, receipt, transaction, record


def test_receipt_tail_v2_is_single_canonical_row_and_distinct_namespace() -> None:
    captured, prepared, values, receipt, transaction, record = _v2_prepare()
    assert captured.identity != capture_conversion_payload(transaction.identity).identity
    assert captured.to_payload() == {
        "schema_version": 2, "conversion_policy": "legacy_history_receipt_tail_v2",
        "receipt_transaction_identity": transaction.identity,
    }
    assert prepared.row_count == 2
    assert prepared.to_payload()["receipt"] == {
        "size": len(receipt.raw_bytes), "blob_digest": hashlib.sha256(receipt.raw_bytes).hexdigest(),
    }
    assert prepared.to_payload()["transaction_captured_record_digest"] == record.digest
    assert prepared.to_payload()["tail_frame"] == {
        "offset": values["source_bytes"].index(b"<!-- nanobot-consolidation:"),
        "size": len(values["source_bytes"]) - values["source_bytes"].index(b"<!-- nanobot-consolidation:"),
        "digest": hashlib.sha256(values["source_bytes"][values["source_bytes"].index(b"<!-- nanobot-consolidation:"):]).hexdigest(),
    }
    frame_body = "<!-- nanobot-consolidation:receipt-marker -->\n" + receipt.history_entry
    expected = json.dumps({"cursor": 2, "timestamp": "2025-04-03 11:22",
                           "content": frame_body}, ensure_ascii=False).encode() + b"\n"
    assert json.loads(expected)["content"] == frame_body
    assert json.loads(expected)["content"].startswith("<!-- nanobot-consolidation:receipt-marker -->\n")
    assert "\n\n[2025-04-04 12:34] second" in json.loads(expected)["content"]
    entry_only = json.dumps({"cursor": 2, "timestamp": "2025-04-03 11:22",
                             "content": receipt.history_entry}, ensure_ascii=False).encode() + b"\n"
    assert hashlib.sha256(expected).digest() != hashlib.sha256(entry_only).digest()
    prefix = json.dumps({"cursor": 1, "timestamp": "2025-01-01 10:00", "content": "prefix"}, ensure_ascii=False).encode() + b"\n"
    assert prepared.outputs["history.jsonl"].blob_digest == hashlib.sha256(prefix + expected).hexdigest()
    assert prepared.to_payload()["canonical_row"] == {
        "cursor": 2, "size": len(expected), "blob_digest": hashlib.sha256(expected).hexdigest(),
    }
    assert decode_conversion_payload(captured.to_payload(), "CAPTURED") == captured
    assert decode_conversion_payload(prepared.to_payload(), "CONVERSION_PREPARED", captured=captured) == prepared


@pytest.mark.parametrize("source", [
    b"<!-- nanobot-consolidation:receipt-marker -->\nentry\n\nmore",
    b"<!-- nanobot-consolidation:receipt-marker -->\nentry\n\n<!-- nanobot-consolidation:receipt-marker -->\nentry\n\n",
    b"<!-- nanobot-consolidation:receipt-marker",
    b"\xff\n<!-- nanobot-consolidation:receipt-marker -->\nentry\n\n",
])
def test_receipt_tail_v2_rejects_ambiguous_or_invalid_source(source: bytes) -> None:
    from nanobot.agent.memory_recovery_conversion import capture_receipt_tail_conversion_payload

    receipt, transaction, record = _v2_receipt("entry")
    values = _prepared_inputs(source_bytes=source)
    with pytest.raises(ConversionError):
        prepare_conversion_payload(capture_receipt_tail_conversion_payload(transaction.identity),
                                   disposition="convert", receipt=receipt,
                                   transaction_captured=transaction,
                                   transaction_captured_record=record, **values)


def test_receipt_tail_v2_phase_round_trip_and_context() -> None:
    captured, prepared, _, _, _, _ = _v2_prepare()
    jsonl = jsonl_applied_conversion_payload(captured, prepared)
    cursors = cursors_applied_conversion_payload(captured, prepared, jsonl)
    digests = dict(prepared_record_digest="1" * 64, jsonl_applied_record_digest="2" * 64,
                   cursors_applied_record_digest="3" * 64)
    finalized = legacy_finalized_conversion_payload(captured, prepared, jsonl, cursors, **digests)
    complete = complete_conversion_payload(captured, prepared, jsonl, cursors, finalized,
                                           finalized_record_digest="4" * 64)
    for phase, value, kwargs in (
        ("JSONL_APPLIED", jsonl, dict(captured=captured, prepared=prepared)),
        ("CURSORS_APPLIED", cursors, dict(captured=captured, prepared=prepared, jsonl_applied=jsonl)),
        ("LEGACY_FINALIZED", finalized, dict(captured=captured, prepared=prepared,
             jsonl_applied=jsonl, cursors_applied=cursors, **digests)),
        ("COMPLETE", complete, dict(captured=captured, prepared=prepared, jsonl_applied=jsonl,
             cursors_applied=cursors, legacy_finalized=finalized, finalized_record_digest="4" * 64)),
    ):
        payload = value.to_payload()
        assert payload["schema_version"] == 2
        assert decode_conversion_payload(payload, phase, **kwargs) == value
        payload["schema_version"] = 1
        with pytest.raises(ConversionError):
            decode_conversion_payload(payload, phase, **kwargs)


def test_receipt_tail_v2_preserve_has_no_invented_evidence() -> None:
    from nanobot.agent.memory_recovery_conversion import capture_receipt_tail_conversion_payload

    captured = capture_receipt_tail_conversion_payload("a" * 64)
    values = _prepared_inputs(history=b'{"cursor":1}\n')
    prepared = prepare_conversion_payload(captured, disposition="preserve_nonempty_jsonl",
                                          source=values["source"], targets=values["targets"],
                                          source_bytes=None)
    assert not set(prepared.to_payload()) & {"receipt", "transaction_captured_record_digest",
                                                 "tail_frame", "canonical_row"}
    assert decode_conversion_payload(prepared.to_payload(), "CONVERSION_PREPARED", captured=captured) == prepared
    forged = prepared.to_payload()
    forged["receipt"] = {"size": 1, "blob_digest": "a" * 64}
    with pytest.raises(ConversionError):
        decode_conversion_payload(forged, "CONVERSION_PREPARED", captured=captured)


def test_receipt_tail_v2_rejects_mismatched_receipt_record_cursor_and_evidence() -> None:
    from dataclasses import replace

    captured, prepared, values, receipt, transaction, record = _v2_prepare()
    attempts = (
        {"receipt": replace(receipt, history_entry="different")},
        {"transaction_captured": replace(transaction, history_marker_digest="0" * 64)},
        {"transaction_captured": replace(transaction, created_at_digest="0" * 64)},
        {"transaction_captured": replace(transaction, chunk_digest="0" * 64)},
        {"transaction_captured_record": replace(record, digest="0" * 64)},
        {"transaction_captured_record": replace(record, _payload_bytes=b'{}')},
    )
    for override in attempts:
        with pytest.raises(ConversionError):
            prepare_conversion_payload(captured, disposition="convert", **values,
                **(dict(receipt=receipt, transaction_captured=transaction,
                       transaction_captured_record=record) | override))
    old_cursor = _prepared_inputs(source_bytes=values["source_bytes"], cursor=b"3")
    with pytest.raises(ConversionError):
        prepare_conversion_payload(captured, disposition="convert", **old_cursor,
            receipt=receipt, transaction_captured=transaction, transaction_captured_record=record)
    for field, key, bad in (("receipt", "size", 0), ("tail_frame", "offset", 1),
                            ("canonical_row", "cursor", 3)):
        forged = prepared.to_payload()
        forged[field][key] = bad
        with pytest.raises(ConversionError):
            decode_conversion_payload(forged, "CONVERSION_PREPARED", captured=captured)


def test_v1_captured_and_prepared_exact_payload_shape_retained() -> None:
    captured = _capture()
    prepared = _prepare()
    assert captured.to_payload() == {"schema_version": 1, "conversion_policy": "legacy_history_v1",
                                     "receipt_transaction_identity": "a" * 64}
    assert set(prepared.to_payload()) == {
        "schema_version", "conversion_policy", "conversion_identity", "disposition", "source",
        "targets", "backup", "fallback_timestamp", "row_count", "final_legacy_cursor",
        "sidecar_counters", "outputs",
    }
    assert prepared.row_count == 2


def test_receipt_tail_v2_rejects_oversized_and_wrong_policy_payloads() -> None:
    from nanobot.agent.memory_recovery_conversion import capture_receipt_tail_conversion_payload

    captured, prepared, _, _, _, _ = _v2_prepare()
    wrong_policy = captured.to_payload()
    wrong_policy["conversion_policy"] = "legacy_history_v1"
    with pytest.raises(ConversionError):
        decode_conversion_payload(wrong_policy, "CAPTURED")
    wrong_prepared = prepared.to_payload()
    wrong_prepared["conversion_policy"] = "legacy_history_v1"
    with pytest.raises(ConversionError):
        decode_conversion_payload(wrong_prepared, "CONVERSION_PREPARED", captured=captured)
    oversized = prepared.to_payload()
    oversized["extra"] = "x" * MAX_RECORD_BYTES
    with pytest.raises(ConversionError):
        decode_conversion_payload(oversized, "CONVERSION_PREPARED", captured=captured)
    assert capture_receipt_tail_conversion_payload("a" * 64).schema_version == 2


def test_receipt_tail_v2_row_retains_marker_and_trims_only_staging_suffix() -> None:
    captured, prepared, values, receipt, _, _ = _v2_prepare("line one\n\n[2025-04-04 12:34] line two  \n  ")
    frame_body = "<!-- nanobot-consolidation:receipt-marker -->\nline one\n\n[2025-04-04 12:34] line two"
    expected = (json.dumps({"cursor": 2, "timestamp": values["fallback_timestamp"],
                            "content": frame_body}, ensure_ascii=False) + "\n").encode()
    assert prepared.to_payload()["canonical_row"] == {
        "cursor": 2, "size": len(expected), "blob_digest": hashlib.sha256(expected).hexdigest(),
    }
    assert prepared.row_count == 2
    assert decode_conversion_payload(prepared.to_payload(), "CONVERSION_PREPARED", captured=captured) == prepared
    assert receipt.history_entry.endswith("  \n  ")


def test_receipt_tail_v2_roundtrip_through_real_manifest_chain(tmp_path) -> None:
    """The transaction record and every conversion predecessor digest are loaded evidence."""
    from dataclasses import replace

    from nanobot.agent.memory_recovery_conversion import capture_receipt_tail_conversion_payload

    receipt, transaction, _ = _v2_receipt()
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    transaction_namespace = authority.open_evidence_namespace("transactions", transaction.identity)
    conversion_namespace = None
    try:
        transaction_manifest = MemoryRecoveryManifest(transaction_namespace)
        transaction_manifest.create(transaction.to_payload())
        loaded_transaction = transaction_manifest.load()[0]
        assert loaded_transaction.phase == "CAPTURED"
        assert loaded_transaction.payload == transaction.to_payload()

        captured = capture_receipt_tail_conversion_payload(loaded_transaction.identity)
        source = (
            b"[2025-01-01 10:00] prefix\n\n"
            + b"<!-- nanobot-consolidation:receipt-marker -->\n"
            + receipt.history_entry.rstrip().encode("utf-8") + b"\n\n"
        )
        values = _prepared_inputs(source_bytes=source)
        with pytest.raises(ConversionError):
            prepare_conversion_payload(
                captured, disposition="convert", receipt=receipt,
                transaction_captured=transaction,
                transaction_captured_record=replace(loaded_transaction, digest="0" * 64),
                **values,
            )
        prepared = prepare_conversion_payload(
            captured, disposition="convert", receipt=receipt,
            transaction_captured=transaction, transaction_captured_record=loaded_transaction,
            **values,
        )
        frame_body = "<!-- nanobot-consolidation:receipt-marker -->\n" + receipt.history_entry
        final_line = (json.dumps({"cursor": 2, "timestamp": values["fallback_timestamp"],
                                  "content": frame_body}, ensure_ascii=False) + "\n").encode()
        assert prepared.canonical_row == {
            "cursor": 2, "size": len(final_line),
            "blob_digest": hashlib.sha256(final_line).hexdigest(),
        }
        assert "\n\n[2025-04-04 12:34] second" in frame_body

        conversion_namespace = authority.open_evidence_namespace("conversions", captured.identity)
        manifest = MemoryRecoveryManifest(conversion_namespace)
        genesis = manifest.create(captured.to_payload())
        prepared_record = manifest.append(
            predecessor_digest=genesis.digest, phase="CONVERSION_PREPARED",
            payload=prepared.to_payload(),
        )
        jsonl = jsonl_applied_conversion_payload(captured, prepared)
        jsonl_record = manifest.append(
            predecessor_digest=prepared_record.digest, phase="JSONL_APPLIED", payload=jsonl.to_payload(),
        )
        cursors = cursors_applied_conversion_payload(captured, prepared, jsonl)
        cursors_record = manifest.append(
            predecessor_digest=jsonl_record.digest, phase="CURSORS_APPLIED",
            payload=cursors.to_payload(),
        )
        finalized = legacy_finalized_conversion_payload(
            captured, prepared, jsonl, cursors,
            prepared_record_digest=prepared_record.digest,
            jsonl_applied_record_digest=jsonl_record.digest,
            cursors_applied_record_digest=cursors_record.digest,
        )
        finalized_record = manifest.append(
            predecessor_digest=cursors_record.digest, phase="LEGACY_FINALIZED",
            payload=finalized.to_payload(),
        )
        complete = complete_conversion_payload(
            captured, prepared, jsonl, cursors, finalized,
            finalized_record_digest=finalized_record.digest,
        )
        manifest.append(
            predecessor_digest=finalized_record.digest, phase="COMPLETE",
            payload=complete.to_payload(),
        )

        records = manifest.load()
        assert [record.phase for record in records] == [
            "CAPTURED", "CONVERSION_PREPARED", "JSONL_APPLIED",
            "CURSORS_APPLIED", "LEGACY_FINALIZED", "COMPLETE",
        ]
        loaded_captured = decode_conversion_payload(records[0].payload, records[0].phase)
        loaded_prepared = decode_conversion_payload(
            records[1].payload, records[1].phase, captured=loaded_captured,
        )
        loaded_jsonl = decode_conversion_payload(
            records[2].payload, records[2].phase,
            captured=loaded_captured, prepared=loaded_prepared,
        )
        loaded_cursors = decode_conversion_payload(
            records[3].payload, records[3].phase,
            captured=loaded_captured, prepared=loaded_prepared, jsonl_applied=loaded_jsonl,
        )
        loaded_finalized = decode_conversion_payload(
            records[4].payload, records[4].phase,
            captured=loaded_captured, prepared=loaded_prepared, jsonl_applied=loaded_jsonl,
            cursors_applied=loaded_cursors,
            prepared_record_digest=records[1].digest,
            jsonl_applied_record_digest=records[2].digest,
            cursors_applied_record_digest=records[3].digest,
        )
        loaded_complete = decode_conversion_payload(
            records[5].payload, records[5].phase,
            captured=loaded_captured, prepared=loaded_prepared, jsonl_applied=loaded_jsonl,
            cursors_applied=loaded_cursors, legacy_finalized=loaded_finalized,
            finalized_record_digest=records[4].digest,
        )
        assert (loaded_captured, loaded_prepared, loaded_jsonl, loaded_cursors,
                loaded_finalized, loaded_complete) == (
                    captured, prepared, jsonl, cursors, finalized, complete,
                )
        assert records[1].payload["transaction_captured_record_digest"] == loaded_transaction.digest
        assert loaded_prepared.canonical_row["blob_digest"] == hashlib.sha256(final_line).hexdigest()
    finally:
        if conversion_namespace is not None:
            conversion_namespace.close()
        transaction_namespace.close()
        authority.close()
