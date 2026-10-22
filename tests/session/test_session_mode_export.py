from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from nanobot.session.manager import Session, SessionManager, SessionStorageError
from nanobot.session.runtime_authority import OfflineSessionOperator
from nanobot.session.session_export import export_canonical_session
from nanobot.session.session_mode import SessionExportTransaction, SessionModeError


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _assert_single_aborted_export(manager: SessionManager) -> None:
    journals = manager._mode_resolver.storage.list(".session-mode/journals")
    export_rows = [
        manager._mode_resolver._journal_records(operation_id)
        for operation_id in journals
        if manager._mode_resolver._journal_records(operation_id)[0]["operation"] == "export"
    ]
    assert len(export_rows) == 1
    assert [row["phase"] for row in export_rows[0]] == ["prepared", "aborted"]


async def _rich_manager(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = Session(
        "cli:export",
        messages=[
            {
                "role": "user",
                "content": "question\n\ntrusted injected context",
                "_runtime_context": {
                    "version": 1,
                    "sources": ["test"],
                    "suffix": "trusted injected context",
                },
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": "inline evidence",
            },
            {"role": "assistant", "content": "answer"},
        ],
        metadata={"title": "portable", "nested": {"value": 3}},
        last_consolidated=2,
    )
    legacy = SessionManager(workspace, sessions_root=root)
    legacy.save(unmanaged, fsync=True)
    lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
    lease.manager.bootstrap_existing_sessions(offline_copy=True)
    lease.manager.convert_legacy_session(unmanaged.key, offline_copy=True)

    path = lease.manager._get_session_path(unmanaged.key)
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[0]["revision"] += 1
    rows[3]["_artifact_read"] = {"reference": "artifact:" + "a" * 32}
    content = rows[3]["content"].encode()
    descriptor = {
        "schema_version": 1,
        "artifact_id": "a" * 32,
        "workspace_id": "b" * 32,
        "session_key": unmanaged.key,
        "incarnation": rows[0]["incarnation"],
        "generation": rows[0]["generation"],
        "event_id": rows[3]["_event_id"],
        "execution_id": rows[3]["_execution_id"],
        "stream_id": "stream",
        "run_id": "run",
        "block_index": None,
        "mime_type": "text/plain; charset=utf-8",
        "byte_length": len(content),
        "sha256": _digest(content),
        "created_at": "2026-10-01T00:00:00+00:00",
        "source_access": {
            "kind": "filesystem",
            "access_class": "workspace",
            "repository": None,
            "relative_path": "evidence.txt",
        },
    }
    rows[0]["metadata"]["artifact_manifest"] = {
        "schema_version": 1,
        "descriptors": {"a" * 32: descriptor},
    }
    path.write_bytes(b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows))
    return lease


def test_lossless_manifest_and_real_legacy_round_trip(tmp_path: Path) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)
        manager = lease.manager
        source_path = manager._get_session_path("cli:export")
        before = source_path.read_bytes()
        record_path = next((manager.sessions_dir / ".session-mode" / "records").iterdir())
        record_before = record_path.read_bytes()

        result = export_canonical_session(
            manager,
            "cli:export",
            (tmp_path / "bundle").absolute(),
            include_compatibility=True,
        )

        assert (result.destination / "canonical-source.jsonl").read_bytes() == before
        manifest = json.loads((result.destination / "manifest.json").read_bytes())
        assert manifest["classification"] == "canonical-lossless"
        assert manifest["session_key"] == "cli:export"
        assert manifest["mode"] == "canonical"
        assert manifest["source_schema_version"] == 1
        assert manifest["source_digest"] == _digest(before)
        assert manifest["source_size"] == len(before)
        assert manifest["compatibility_file"] == "compatibility-history.jsonl"
        assert manifest["compatibility_classification"] == "compatibility"
        assert manifest["artifact_blobs_portable"] is False
        assert manifest["artifact_references"] == ["artifact:" + "a" * 32]

        compatibility = (result.destination / "compatibility-history.jsonl").read_bytes()
        parsed = manager._jsonl_store.parse_legacy_bytes(
            compatibility,
            expected_key="cli:export",
        )
        assert parsed.metadata == {"title": "portable", "nested": {"value": 3}}
        assert parsed.last_archived == 2
        assert parsed.messages[0]["content"] == "question"
        assert parsed.messages[1]["tool_calls"][0]["id"] == "call-1"
        assert parsed.messages[2]["tool_call_id"] == "call-1"
        assert parsed.messages[2]["content"] == "inline evidence"
        assert all(not any(name.startswith("_") for name in row) for row in parsed.messages)
        assert source_path.read_bytes() == before
        assert record_path.read_bytes() == record_before
        assert result.compatibility_included
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_unsupported_source_only_artifact_evidence_aborts_without_mutation(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)
        manager = lease.manager
        source_path = manager._get_session_path("cli:export")
        rows = [json.loads(line) for line in source_path.read_bytes().splitlines()]
        rows[0]["revision"] += 1
        rows[3]["content"] = None
        source_path.write_bytes(
            b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
        )
        before = source_path.read_bytes()
        record_path = next((manager.sessions_dir / ".session-mode" / "records").iterdir())
        record_before = record_path.read_bytes()
        destination = (tmp_path / "unsupported").absolute()

        with pytest.raises(SessionModeError, match="compatibility export is unsupported"):
            export_canonical_session(
                manager,
                "cli:export",
                destination,
                include_compatibility=True,
            )

        assert not destination.exists()
        assert source_path.read_bytes() == before
        assert record_path.read_bytes() == record_before
        _assert_single_aborted_export(manager)
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("metadata_name", "metadata_value"),
    [
        ("execution_lease", {"active": True}),
        ("execution_recovery_required", True),
        ("pending_user_turn", {"pending": True}),
        ("pending_user_followups", ["pending"]),
        ("webui_recovery", {"pending": True}),
    ],
)
def test_execution_and_legacy_recovery_evidence_is_unsupported(
    tmp_path: Path,
    metadata_name: str,
    metadata_value: object,
) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)
        manager = lease.manager
        source_path = manager._get_session_path("cli:export")
        rows = [json.loads(line) for line in source_path.read_bytes().splitlines()]
        rows[0]["revision"] += 1
        rows[0]["metadata"][metadata_name] = metadata_value
        source_path.write_bytes(
            b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
        )
        before = source_path.read_bytes()
        record_path = next((manager.sessions_dir / ".session-mode" / "records").iterdir())
        record_before = record_path.read_bytes()
        destination = (tmp_path / "recovery-unsupported").absolute()

        with pytest.raises(SessionModeError, match="compatibility export is unsupported"):
            export_canonical_session(
                manager,
                "cli:export",
                destination,
                include_compatibility=True,
            )

        assert source_path.read_bytes() == before
        assert record_path.read_bytes() == record_before
        assert not destination.exists()
        _assert_single_aborted_export(manager)
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_unresolved_tool_tail_is_unsupported_without_mutation(tmp_path: Path) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)
        manager = lease.manager
        source_path = manager._get_session_path("cli:export")
        rows = [json.loads(line) for line in source_path.read_bytes().splitlines()]
        rows[0]["revision"] += 1
        rows[0]["metadata"].pop("artifact_manifest")
        del rows[3]
        source_path.write_bytes(
            b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
        )
        before = source_path.read_bytes()
        record_path = next((manager.sessions_dir / ".session-mode" / "records").iterdir())
        record_before = record_path.read_bytes()
        destination = (tmp_path / "unresolved-unsupported").absolute()

        with pytest.raises(SessionModeError, match="compatibility export is unsupported"):
            export_canonical_session(
                manager,
                "cli:export",
                destination,
                include_compatibility=True,
            )

        assert source_path.read_bytes() == before
        assert record_path.read_bytes() == record_before
        assert not destination.exists()
        _assert_single_aborted_export(manager)
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("role", ["system", "developer"])
def test_unsupported_public_role_aborts_without_mutation(tmp_path: Path, role: str) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)
        manager = lease.manager
        source_path = manager._get_session_path("cli:export")
        rows = [json.loads(line) for line in source_path.read_bytes().splitlines()]
        rows[0]["revision"] += 1
        rows[1]["role"] = role
        source_path.write_bytes(
            b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
        )
        before = source_path.read_bytes()
        record_path = next((manager.sessions_dir / ".session-mode" / "records").iterdir())
        record_before = record_path.read_bytes()
        destination = (tmp_path / "role-unsupported").absolute()

        with pytest.raises(SessionModeError, match="compatibility export is unsupported"):
            export_canonical_session(
                manager,
                "cli:export",
                destination,
                include_compatibility=True,
            )

        assert source_path.read_bytes() == before
        assert record_path.read_bytes() == record_before
        assert not destination.exists()
        _assert_single_aborted_export(manager)
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_lossless_manifest_retains_source_only_artifact_reference(tmp_path: Path) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)
        source_path = lease.manager._get_session_path("cli:export")
        rows = [json.loads(line) for line in source_path.read_bytes().splitlines()]
        rows[0]["revision"] += 1
        rows[3]["content"] = None
        source_path.write_bytes(
            b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
        )

        result = export_canonical_session(
            lease.manager,
            "cli:export",
            (tmp_path / "lossless-only").absolute(),
        )
        manifest = json.loads((result.destination / "manifest.json").read_bytes())
        assert manifest["artifact_references"] == ["artifact:" + "a" * 32]
        assert manifest["artifact_blobs_portable"] is False
        assert not (result.destination / "compatibility-history.jsonl").exists()
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


def test_export_uses_only_public_transaction_surface() -> None:
    source = inspect.getsource(export_canonical_session)
    for forbidden in ("_append_journal", "_atomic_write", "_fsync_dir", "os.replace", ".root"):
        assert forbidden not in source
    assert source.count("begin_export(") == 1
    assert "locked_snapshot(" in source
    assert "parse_legacy_bytes(" in source


def test_preexisting_destination_is_not_overwritten(tmp_path: Path) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)
        destination = (tmp_path / "existing").absolute()
        destination.mkdir()
        marker = destination / "marker"
        marker.write_bytes(b"keep")
        with pytest.raises(SessionModeError, match="absolute and nonexistent"):
            export_canonical_session(lease.manager, "cli:export", destination)
        assert marker.read_bytes() == b"keep"
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "phase",
    ["before_publication", "after_install", "after_published", "before_complete"],
)
def test_export_crashes_leave_transaction_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)
        manager = lease.manager
        destination = (tmp_path / "crash-bundle").absolute()
        original_publish = SessionExportTransaction.mark_published
        original_complete = SessionExportTransaction.complete

        def publish(transaction: SessionExportTransaction) -> None:
            if phase == "before_publication":
                raise RuntimeError("injected export crash")
            if phase == "after_install":
                original_publish(transaction, fault_after_install=True)
                return
            original_publish(transaction)
            if phase == "after_published":
                raise RuntimeError("injected export crash")

        def complete(transaction: SessionExportTransaction) -> None:
            if phase == "before_complete":
                raise RuntimeError("injected export crash")
            original_complete(transaction)

        monkeypatch.setattr(SessionExportTransaction, "mark_published", publish)
        monkeypatch.setattr(SessionExportTransaction, "complete", complete)
        with pytest.raises(RuntimeError, match="injected"):
            export_canonical_session(manager, "cli:export", destination)
        assert lease.close is not None
        await lease.close()

        monkeypatch.setattr(SessionExportTransaction, "mark_published", original_publish)
        monkeypatch.setattr(SessionExportTransaction, "complete", original_complete)
        second = await OfflineSessionOperator(
            tmp_path / "workspace",
            copied_sessions_root=tmp_path / "sessions",
        ).acquire()
        assert second.manager.recover_session_mode("cli:export").mode == "canonical"
        assert destination.exists() is (phase != "before_publication")
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


def test_old_five_field_export_binding_is_rejected_by_normal_resolution(tmp_path: Path) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)
        manager = lease.manager
        resolved = manager.resolve_session_mode("cli:export")
        tx = manager._mode_resolver.begin_export(
            "cli:export",
            (tmp_path / "old-binding").absolute(),
            {"canonical-source.jsonl": b"source\n", "manifest.json": b"{}\n"},
        )
        journal_path = (
            manager.sessions_dir
            / ".session-mode"
            / "journals"
            / tx.operation_id
            / "0001-prepared.json"
        )
        row = json.loads(journal_path.read_bytes())
        row["target_binding"] = {
            "registry_id": resolved.registry_id,
            "record_revision": resolved.record_revision,
            "mode": resolved.mode,
            "incarnation": resolved.incarnation,
            "generation": resolved.generation,
        }
        journal_path.write_bytes(_canonical_json(row))
        tx._parent_pin.close()
        assert lease.close is not None
        await lease.close()

        second = await OfflineSessionOperator(
            tmp_path / "workspace",
            copied_sessions_root=tmp_path / "sessions",
        ).acquire()
        with pytest.raises(SessionModeError, match="offline recovery"):
            second.manager.resolve_session_mode("cli:export")
        assert second.close is not None
        await second.close()

    asyncio.run(exercise())


def test_legacy_proof_is_not_a_json_decode_substitute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def exercise() -> None:
        lease = await _rich_manager(tmp_path)

        def reject(*_args: object, **_kwargs: object) -> Session:
            raise SessionStorageError()

        monkeypatch.setattr(lease.manager._jsonl_store, "parse_legacy_bytes", reject)
        destination = (tmp_path / "strict-reader").absolute()
        with pytest.raises(SessionModeError, match="compatibility export is unsupported"):
            export_canonical_session(
                lease.manager,
                "cli:export",
                destination,
                include_compatibility=True,
            )
        assert not destination.exists()
        assert lease.close is not None
        await lease.close()

    asyncio.run(exercise())
