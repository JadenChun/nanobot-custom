"""Bounded, source-owned idle discovery tests."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

import nanobot.session.source_discovery as source_discovery_module
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source_discovery import (
    SourceDiscoveryCursor,
    SourceDiscoveryLimits,
    SourceDiscoveryRootChangedError,
)


def _source_session(backend: JsonlSessionStore, key: str) -> None:
    session = Session(key)
    backend.save(session)
    source = backend.source
    source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )


@pytest.mark.asyncio
async def test_discovery_inspects_bounded_headers_and_returns_only_canonical_keys(tmp_path: Path) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    _source_session(backend, "source:one")
    _source_session(backend, "source:two")
    root = backend.sessions_dir
    (root / "random.txt").write_text("not a session\n", encoding="utf-8")
    (root / "invalid.jsonl").write_text("{\"_type\":\"metadata\"}\n", encoding="utf-8")
    (root / "oversized.jsonl").write_bytes(b"{" + b"x" * 700 + b"}\n")
    (root / "not-a-session").mkdir()
    symlink = root / "symlink.jsonl"
    try:
        symlink.symlink_to(backend.get_session_path("source:one"))
    except OSError:
        pytest.skip("symlinks unavailable")

    cursor = SourceDiscoveryCursor(
        backend,
        limits=SourceDiscoveryLimits(
            max_entries_inspected=20,
            max_candidates_returned=1,
            max_header_bytes=500,
            max_decoded_key_bytes=128,
        ),
    )
    first = await cursor.scan_once()
    second = await cursor.scan_once()
    assert len(first.keys) == 1
    assert len(second.keys) == 1
    assert set(first.keys + second.keys) == {"source:one", "source:two"}
    assert first.counters.inspected + second.counters.inspected >= 2
    assert first.counters.rejected + second.counters.rejected >= 1
    assert first.keys == first.candidate_keys == first.session_keys
    assert not hasattr(first, "preview")
    exhausted = await cursor.scan_once()
    assert exhausted.exhausted
    assert cursor.cursor_open is False

    await cursor.aclose()
    assert cursor.closed
    with pytest.raises(Exception):
        await cursor.scan_once()


@pytest.mark.asyncio
async def test_discovery_physical_limit_counts_invalid_entries_and_restarts_after_exhaustion(
    tmp_path: Path,
) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    _source_session(backend, "source:one")
    root = backend.sessions_dir
    (root / "invalid.txt").write_text("x", encoding="utf-8")
    cursor = SourceDiscoveryCursor(
        backend,
        limits=SourceDiscoveryLimits(1, 1, 500, 128),
    )
    slices = []
    for _ in range(20):
        result = await cursor.scan_once()
        slices.append(result)
        if result.exhausted:
            break
    assert slices
    assert all(item.counters.inspected <= 1 for item in slices)
    assert any(item.counters.rejected for item in slices)
    assert any("source:one" in item.keys for item in slices)
    restarted = []
    for _ in range(20):
        result = await cursor.scan_once()
        restarted.append(result)
        if result.exhausted:
            break
    assert restarted[-1].exhausted
    assert any(item.keys == ("source:one",) for item in restarted)
    await cursor.aclose()


@pytest.mark.asyncio
async def test_discovery_root_replacement_invalidates_cursor(tmp_path: Path) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    _source_session(backend, "source:one")
    cursor = SourceDiscoveryCursor(
        backend,
        limits=SourceDiscoveryLimits(100, 100, 500, 128),
    )
    old_root = backend.sessions_dir.with_name("old-source-root")
    os.rename(backend.sessions_dir, old_root)
    backend.sessions_dir.mkdir()
    with pytest.raises(SourceDiscoveryRootChangedError):
        await cursor.scan_once()
    await cursor.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("via_factory", [False, True])
async def test_root_invalidation_releases_owned_authority_after_aclose(
    tmp_path: Path,
    via_factory: bool,
) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    _source_session(backend, "source:owned")
    limits = SourceDiscoveryLimits(100, 100, 500, 128)
    cursor = (
        backend.source.create_discovery_cursor(limits=limits)
        if via_factory
        else SourceDiscoveryCursor(backend, limits=limits)
    )
    authority = cursor.root_authority
    descriptor = authority._fd
    old_root = backend.sessions_dir.with_name("old-source-root")
    os.rename(backend.sessions_dir, old_root)
    backend.sessions_dir.mkdir()
    with pytest.raises(SourceDiscoveryRootChangedError):
        await cursor.scan_once()
    assert cursor.closed and authority.closed
    with pytest.raises(OSError):
        os.fstat(descriptor)
    await cursor.aclose()
    assert authority.closed


def test_borrowed_root_authority_is_not_closed_by_cursor(tmp_path: Path) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    authority = backend.source.create_source_root_authority()
    cursor = SourceDiscoveryCursor(
        backend,
        limits=SourceDiscoveryLimits(100, 100, 500, 128),
        root_authority=authority,
    )
    try:
        cursor.close()
        assert cursor.closed and not authority.closed
        authority.require_current(backend)
    finally:
        authority.close()


def test_failed_cursor_construction_closes_issued_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    issued: dict[str, object] = {}
    original_factory = source_discovery_module.SourceRootAuthority.issue_for_backend

    def issue(source_backend):  # type: ignore[no-untyped-def]
        authority = original_factory(source_backend)
        issued["authority"] = authority
        return authority

    monkeypatch.setattr(source_discovery_module.SourceRootAuthority, "issue_for_backend", issue)

    def fail_root(_: Path):
        raise SourceDiscoveryRootChangedError("root qualification failed")

    monkeypatch.setattr(source_discovery_module, "_directory_identity", fail_root)
    with pytest.raises(SourceDiscoveryRootChangedError):
        SourceDiscoveryCursor(
            backend,
            limits=SourceDiscoveryLimits(100, 100, 500, 128),
        )
    authority = issued["authority"]
    assert authority.closed
    with pytest.raises(OSError):
        os.fstat(authority._fd)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_overlapping_discovery_scans_are_serialized(tmp_path: Path) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    _source_session(backend, "source:one")
    cursor = SourceDiscoveryCursor(
        backend,
        limits=SourceDiscoveryLimits(100, 100, 500, 128),
    )
    results = await asyncio.gather(*(cursor.scan_once() for _ in range(3)))
    assert all(result.keys == ("source:one",) for result in results)
    await cursor.aclose()


@pytest.mark.asyncio
async def test_discovery_reads_one_header_byte_at_a_time_and_stops_at_newline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    _source_session(backend, "source:bounded")
    path = backend.get_session_path("source:bounded")
    raw = path.read_bytes()
    newline = raw.index(b"\n")
    header = raw[: newline + 1]
    sentinel = b"HEADER_SENTINEL_MUST_NOT_BE_READ"
    path.write_bytes(header + sentinel + raw[newline + 1 :])
    observations: list[tuple[int, int, int, bytes]] = []
    original_read = source_discovery_module.os.read

    def tracked_read(fd: int, size: int) -> bytes:
        before = os.lseek(fd, 0, os.SEEK_CUR)
        result = original_read(fd, size)
        after = os.lseek(fd, 0, os.SEEK_CUR)
        observations.append((size, before, after, result))
        return result

    monkeypatch.setattr(source_discovery_module.os, "read", tracked_read)
    cursor = SourceDiscoveryCursor(
        backend,
        limits=SourceDiscoveryLimits(
            max_entries_inspected=100,
            max_candidates_returned=100,
            max_header_bytes=newline,
            max_decoded_key_bytes=128,
        ),
    )
    try:
        result = await cursor.scan_once()
        assert result.keys == ("source:bounded",)
        assert observations
        assert all(size == 1 for size, _, _, _ in observations)
        assert observations[-1][2] == newline + 1
        assert observations[-1][3] == b"\n"
        assert b"".join(result for _, _, _, result in observations) == header
        assert all(after <= newline + 1 for _, _, after, _ in observations)
    finally:
        await cursor.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [True, 1.0])
async def test_discovery_rejects_non_integer_source_schema(tmp_path: Path, schema: object) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    _source_session(backend, "source:typed")
    path = backend.get_session_path("source:typed")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["metadata"]["source_schema_version"] = schema
    path.write_text(
        "\n".join([json.dumps(header), *lines[1:]]) + "\n",
        encoding="utf-8",
    )
    cursor = SourceDiscoveryCursor(
        backend,
        limits=SourceDiscoveryLimits(100, 100, 2000, 128),
    )
    try:
        result = await cursor.scan_once()
        assert "source:typed" not in result.keys
    finally:
        await cursor.aclose()
