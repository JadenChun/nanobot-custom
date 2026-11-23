"""A38 cap, bound and binding cases (C1.1-C3.3).

Source: a3-evidence/a3-reconciliation-2026-11-14/resume2/req-a38-cap/CASES.md.
Each test keeps the case scenario, inputs and outcomes. The interface mapping
for the contract gaps (G1-G15) is recorded in
a3-evidence/a3-reconciliation-2026-11-14/resume2/a38-caps/AUTHOR-CASES.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import kohaku.session.mode_storage as mode_storage
from kohaku.session import source_store
from kohaku.session.runtime_authority import DurableSessionAuthority
from kohaku.session.session_mode import SessionModeError
from kohaku.session.source import SourceValidationError, new_source_event


def _event(content: str, run_id: str) -> dict[str, Any]:
    return new_source_event({"role": "user", "content": content}, run_id=run_id)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _open(tmp_path: Path, name: str = "sessions") -> Any:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return await DurableSessionAuthority(
        workspace, sessions_root=tmp_path / name
    ).acquire()


def _source_path(manager: Any, key: str) -> Path:
    return manager._jsonl_store.get_session_path(key)


def _session_texts(session: Any) -> list[str]:
    return [
        message["content"]
        for message in session.messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    ]


async def _seed(tmp_path: Path, contents: dict[str, str]) -> None:
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        for key, content in contents.items():
            manager.admit_persistent_write(key)
            manager.reset_session(key, replacement_events=[_event(content, key)])
    finally:
        await lease.close()


async def _seed_oversize_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str, str, str, bytes, int]:
    """Seed one oversize, one normal and one reset target, then lower the cap.

    The case calls for a source of ``cap + 1`` bytes. The cap is the exact
    serialized size of the seeded oversize source minus one, so the source is
    a valid pre-existing entry that is now one byte above the read bound.
    """
    oversize, normal, reset_target = "cli:oversize", "cli:normal", "cli:reset"
    await _seed(
        tmp_path,
        {
            oversize: "oversize source payload",
            normal: "normal source payload",
            reset_target: "reset target payload",
        },
    )
    lease = await _open(tmp_path)
    try:
        raw = _source_path(lease.manager, oversize).read_bytes()
        normal_raw = _source_path(lease.manager, normal).read_bytes()
    finally:
        await lease.close()
    cap = len(raw) - 1
    assert len(raw) == cap + 1
    assert len(normal_raw) <= cap
    monkeypatch.setattr(source_store, "MAX_SOURCE_SERIALIZED_BYTES", cap)
    return oversize, normal, reset_target, "normal source payload", raw, cap


# --------------------------------------------------------------------------- U1


@pytest.mark.asyncio
async def test_c1_1_new_oversize_source_is_refused_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_a, key_b = "cli:cap-a", "cli:cap-b"
    content = "payload at the bound"
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        manager.admit_persistent_write(key_a)
        manager.reset_session(key_a, replacement_events=[_event(content, key_a)])
        size_a = _source_path(manager, key_a).stat().st_size
    finally:
        await lease.close()

    monkeypatch.setattr(source_store, "MAX_SOURCE_SERIALIZED_BYTES", size_a)
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        # A is exactly at the cap and stays a success.
        manager.admit_persistent_write(key_a)
        manager.reset_session(key_a, replacement_events=[_event(content, key_a)])
        a_bytes = _source_path(manager, key_a).read_bytes()
        assert len(a_bytes) == size_a
        assert _event(content, key_a)["content"] in a_bytes.decode("utf-8")
        assert content in _session_texts(manager.get_or_create(key_a))

        # B is one byte above the cap and is refused before publication.
        manager.admit_persistent_write(key_b)
        before_b = _source_path(manager, key_b).read_bytes()
        with pytest.raises((SourceValidationError, SessionModeError)):
            manager.reset_session(key_b, replacement_events=[_event(content + "!", key_b)])
        after_b = _source_path(manager, key_b).read_bytes()
        assert after_b == before_b
        assert content + "!" not in after_b.decode("utf-8")
        assert _session_texts(manager.get_or_create(key_b)) == []
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_c1_2_present_oversize_source_does_not_deny_the_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversize, normal, reset_target, normal_content, raw, cap = await _seed_oversize_source(
        tmp_path, monkeypatch
    )
    # A fresh acquisition proves the namespace inventory still resolves.
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        modes = manager._mode_resolver.list_modes()
        assert {mode.state for mode in modes} == {"active"}
        assert len(modes) == 2  # oversize key omitted, other keys listed

        manager.reset_session(
            reset_target, replacement_events=[_event("after reset", reset_target)]
        )
        assert "after reset" in _session_texts(manager.get_or_create(reset_target))

        assert _session_texts(manager.get_or_create(normal)) == [normal_content]
        normal_destination = tmp_path / "normal-export"
        receipt = manager.export_session_source(normal, normal_destination)
        exported = (normal_destination / "source.jsonl").read_bytes()
        assert exported == _source_path(manager, normal).read_bytes()
        assert _digest(exported) == receipt.source_digest

        # The oversize bytes are unchanged and keep the same length.
        assert _source_path(manager, oversize).read_bytes() == raw
        assert len(raw) == cap + 1
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_c1_3_oversize_source_is_exported_and_quarantined_with_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversize, _normal, _reset_target, _content, raw, _cap = await _seed_oversize_source(
        tmp_path, monkeypatch
    )
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        export_destination = tmp_path / "oversize-export"
        export = manager.export_session_source(oversize, export_destination)
        exported = (export_destination / "source.jsonl").read_bytes()
        assert export.session_key == oversize
        assert export.action == "export"
        assert export.source_size == len(raw)
        assert export.source_digest == _digest(raw)
        assert exported == raw
        assert len(exported) == export.source_size
        manifest = json.loads((export_destination / "manifest.json").read_bytes())
        assert manifest["session_key"] == oversize
        assert manifest["session_key"] == export.session_key
        assert manifest["source_size"] == export.source_size
        assert manifest["source_digest"] == export.source_digest

        quarantine_destination = tmp_path / "oversize-quarantine"
        quarantined = manager.export_session_source(
            oversize, quarantine_destination, quarantine=True
        )
        retained = (quarantine_destination / "source.jsonl").read_bytes()
        assert quarantined.session_key == oversize
        assert quarantined.action == "quarantine"
        assert retained == raw
        assert _digest(retained) == export.source_digest
        # The stored entry is not deleted and is not changed.
        assert _source_path(manager, oversize).read_bytes() == raw
        assert len(_source_path(manager, oversize).read_bytes()) == len(raw)
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_c1_4_new_oversize_source_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = "refused payload"
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        manager.admit_persistent_write("cli:anchor")
        manager.reset_session("cli:anchor", replacement_events=[_event(content, "cli:anchor")])
        size = _source_path(manager, "cli:anchor").stat().st_size
    finally:
        await lease.close()

    monkeypatch.setattr(source_store, "MAX_SOURCE_SERIALIZED_BYTES", size)
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        key_c = "cli:refused"
        manager.admit_persistent_write(key_c)
        before = _source_path(manager, key_c).read_bytes()
        with pytest.raises((SourceValidationError, SessionModeError)):
            manager.reset_session(
                key_c, replacement_events=[_event(content + "!!!", key_c)]
            )
        assert _source_path(manager, key_c).read_bytes() == before
        assert content + "!!!" not in _source_path(manager, key_c).read_text("utf-8")
    finally:
        await lease.close()


# --------------------------------------------------------------------------- U2


async def _seed_journals(tmp_path: Path, keys: tuple[str, ...]) -> dict[str, dict[str, bytes]]:
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        for key in keys:
            manager.admit_persistent_write(key)
            manager.reset_session(key, replacement_events=[_event(key, key)])
        journals = manager.sessions_dir / ".session-mode" / "journals"
        return {
            name: {
                entry.name: entry.read_bytes()
                for entry in sorted((journals / name).iterdir())
            }
            for name in sorted(os.listdir(journals))
        }
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_c2_1_namespace_above_the_bound_is_still_acquirable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = await _seed_journals(tmp_path, ("cli:j0", "cli:j1", "cli:j2"))
    bound = len(captured) - 1
    monkeypatch.setattr(mode_storage, "MODE_ENUM_LIMIT", bound)

    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        journal_root = manager.sessions_dir / ".session-mode"
        live = {entry.name for entry in (journal_root / "journals").iterdir()}
        retained_root = journal_root / "retained-journals"
        retained = (
            {entry.name for entry in retained_root.iterdir()}
            if retained_root.exists()
            else set()
        )
        assert live | retained == set(captured)
        assert not live & retained
        assert len(live) <= bound

        receipt = manager.export_session_journal_archive(tmp_path / "archive-c21")
        archive = json.loads((tmp_path / "archive-c21" / "journal-archive.json").read_bytes())
        ids = [entry["operation_id"] for entry in archive["journals"]]
        assert sorted(ids) == sorted(captured)
        assert len(ids) == len(set(ids)) == len(captured)
        assert receipt.count == len(captured)
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_c2_2_every_retained_journal_stays_readable_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = await _seed_journals(tmp_path, ("cli:j0", "cli:j1", "cli:j2"))
    monkeypatch.setattr(mode_storage, "MODE_ENUM_LIMIT", len(captured) - 1)
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        for operation_id, records in captured.items():
            journal = manager.read_session_journal(operation_id)
            assert journal
            assert journal == [json.loads(raw) for raw in records.values()]
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_c2_3_operator_export_names_the_retained_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = await _seed_journals(tmp_path, ("cli:j0", "cli:j1", "cli:j2"))
    monkeypatch.setattr(mode_storage, "MODE_ENUM_LIMIT", len(captured) - 1)
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        destination = tmp_path / "archive-c23"
        receipt = manager.export_session_journal_archive(destination)
        archive = json.loads((destination / "journal-archive.json").read_bytes())
        exported = {entry["operation_id"]: entry for entry in archive["journals"]}
        assert set(exported) == set(captured)
        assert archive["count"] == len(captured) == receipt.count
        for operation_id, records in captured.items():
            entry = exported[operation_id]
            assert entry["location"] in {"live", "retained"}
            assert entry["records"] == [json.loads(raw) for raw in records.values()]
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_c2_4_at_the_bound_behaviour_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = await _seed_journals(tmp_path, ("cli:k0", "cli:k1", "cli:k2"))
    bound = len(captured)
    monkeypatch.setattr(mode_storage, "MODE_ENUM_LIMIT", bound)

    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        journals = manager.sessions_dir / ".session-mode" / "journals"
        live = {entry.name for entry in journals.iterdir()}
        assert len(live) == bound

        destination = tmp_path / "archive-c24"
        manager.export_session_journal_archive(destination)
        archive = json.loads((destination / "journal-archive.json").read_bytes())
        assert {entry["operation_id"] for entry in archive["journals"]} == set(captured)

        assert {entry.name for entry in journals.iterdir()} == live
        for operation_id, records in captured.items():
            assert manager.read_session_journal(operation_id) == [
                json.loads(raw) for raw in records.values()
            ]
    finally:
        await lease.close()


# --------------------------------------------------------------------------- U3


class _ResolutionCounter:
    """Test-side counter for full-source resolutions."""

    def __init__(self, resolver: Any) -> None:
        self._resolve = resolver.resolve
        self.keys: list[str] = []

    def __call__(self, key: str) -> Any:
        self.keys.append(key)
        return self._resolve(key)

    def count(self, key: str) -> int:
        return self.keys.count(key)


def _install_counter(manager: Any, monkeypatch: pytest.MonkeyPatch) -> _ResolutionCounter:
    resolver = manager._mode_resolver
    assert resolver is not None
    counter = _ResolutionCounter(resolver)
    monkeypatch.setattr(resolver, "resolve", counter)
    return counter


@pytest.mark.asyncio
async def test_c3_1_first_get_resolves_later_gets_use_the_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(tmp_path, {"cli:a": "content of A"})
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        counter = _install_counter(manager, monkeypatch)

        first = manager.get_or_create("cli:a")
        assert _session_texts(first) == ["content of A"]
        assert counter.count("cli:a") == 1

        second = manager.get_or_create("cli:a")
        assert _session_texts(second) == ["content of A"]
        assert counter.count("cli:a") == 1
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_c3_2_unrelated_source_change_does_not_force_re_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(tmp_path, {"cli:a": "content of A", "cli:b": "content of B"})
    lease = await _open(tmp_path)
    try:
        manager = lease.manager
        counter = _install_counter(manager, monkeypatch)
        manager.get_or_create("cli:a")
        manager.get_or_create("cli:b")
        assert counter.count("cli:a") == 1
        assert counter.count("cli:b") == 1

        with _source_path(manager, "cli:b").open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

        assert _session_texts(manager.get_or_create("cli:a")) == ["content of A"]
        assert counter.count("cli:a") == 1
        assert counter.count("cli:b") == 1
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_c3_3_restart_keeps_correctness(tmp_path: Path) -> None:
    await _seed(tmp_path, {"cli:a": "content of A"})
    lease = await _open(tmp_path)
    try:
        assert _session_texts(lease.manager.get_or_create("cli:a")) == ["content of A"]
    finally:
        await lease.close()

    restarted = await _open(tmp_path)
    try:
        assert _session_texts(restarted.manager.get_or_create("cli:a")) == ["content of A"]
    finally:
        await restarted.close()
