"""Real source/root/cursor owner tests; no provider or idle activation."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.idle_runtime import IdleRuntimeBindingError, _source_material
from nanobot.agent.idle_scanner import IdleCompactionScanner
from nanobot.session import source_discovery as discovery_module
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source_discovery import SourceDiscoveryCursor, SourceDiscoveryLimits
from nanobot.session.source_root import SourceRootAuthority, SourceRootAuthorityError
from nanobot.session.source_store import SourceSessionStore, SourceSnapshot

KEY = "cli:idle-source-owner"
LIMITS = SourceDiscoveryLimits(10, 10, 2000, 256)


def _source(tmp_path: Path) -> tuple[JsonlSessionStore, SourceSessionStore, SourceSnapshot]:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    session = Session(KEY)
    backend.save(session)
    source = backend.source
    snapshot = source.upgrade_source_session(
        KEY, expected_revision=session.revision, expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    return backend, source, snapshot


@pytest.mark.parametrize("same_object", [True, False])
def test_source_backend_observation_uses_exact_object_identity(
    tmp_path: Path, same_object: bool,
) -> None:
    backend, source, _ = _source(tmp_path)
    observed = backend if same_object else JsonlSessionStore(
        tmp_path / "workspace", sessions_root=tmp_path / "sessions",
    )
    assert source.matches_backend(observed) is same_object


@pytest.mark.parametrize("relation", ["exact", "shared-backend", "same-path-new-backend"])
def test_discovery_identity_is_not_equivalent_directory_identity(
    tmp_path: Path, relation: str,
) -> None:
    backend, source, _ = _source(tmp_path)
    authority = source.create_source_root_authority()
    cursor = SourceDiscoveryCursor(backend, limits=LIMITS, root_authority=authority)
    try:
        compared = source
        if relation == "shared-backend":
            compared = SourceSessionStore(backend)
        elif relation == "same-path-new-backend":
            other = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
            compared = SourceSessionStore(other)
        assert cursor.matches_source_store(compared) is (relation != "same-path-new-backend")
    finally:
        cursor.close()
        authority.close()


def test_discovery_identity_keeps_discovery_then_source_backend_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, source, _ = _source(tmp_path)
    authority = source.create_source_root_authority()
    cursor = SourceDiscoveryCursor(backend, limits=LIMITS, root_authority=authority)
    source_read = SourceSessionStore.__getattribute__
    cursor_read = SourceDiscoveryCursor.__getattribute__
    trace: list[str] = []

    def read_source(self, name):
        if self is source and name == "_backend":
            trace.append("source")
        return source_read(self, name)

    def read_cursor(self, name):
        if self is cursor and name == "_backend":
            trace.append("discovery")
        return cursor_read(self, name)

    try:
        monkeypatch.setattr(SourceSessionStore, "__getattribute__", read_source)
        monkeypatch.setattr(SourceDiscoveryCursor, "__getattribute__", read_cursor)
        assert cursor.matches_source_store(source) is True
        assert trace == ["discovery", "source"]
    finally:
        cursor.close()
        authority.close()


class _AsymmetricIdentity(str):
    def __eq__(self, _other: object) -> bool:
        return True

    def __ne__(self, _other: object) -> bool:
        return True


@pytest.mark.parametrize("identity", ["exact", "different", "asymmetric-ne"])
def test_store_identity_observation_preserves_original_inequality(
    tmp_path: Path, identity: str,
) -> None:
    _, source, snapshot = _source(tmp_path)
    compared = snapshot.state.store_identity
    if identity == "different":
        compared += "-different"
    elif identity == "asymmetric-ne":
        compared = _AsymmetricIdentity(compared)
    assert source.store_identity_differs(compared) is (identity != "exact")


def test_store_identity_is_live_not_cached_from_an_earlier_observation(tmp_path: Path) -> None:
    _, source, snapshot = _source(tmp_path)
    assert source.store_identity_differs(snapshot.state.store_identity) is False
    source._store_identity = "changed-owner-identity"
    assert source.store_identity_differs(snapshot.state.store_identity) is True


def test_root_metadata_is_the_original_resolved_directory_string(tmp_path: Path) -> None:
    backend, source, _ = _source(tmp_path)
    assert source.resolved_runtime_source_root() == str(backend.sessions_dir.resolve(strict=False))


def test_root_metadata_keeps_live_path_and_non_strict_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, source, _ = _source(tmp_path)
    original = Path.resolve
    before = backend.sessions_dir
    after = tmp_path / "not-created-root"
    trace: list[tuple[Path, bool]] = []

    def resolve(path: Path, strict: bool = False) -> Path:
        trace.append((path, strict))
        return original(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert source.resolved_runtime_source_root() == str(original(before, strict=False))
    backend.sessions_dir = after
    assert source.resolved_runtime_source_root() == str(original(after, strict=False))
    assert trace == [(before, False), (after, False)]


def test_current_root_operation_keeps_live_authority_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, source, _ = _source(tmp_path)
    authority = source.create_source_root_authority()
    original = SourceRootAuthority.require_current
    calls: list[tuple[object, object]] = []

    def require_current(root, owned_backend):
        calls.append((root, owned_backend))
        return original(root, owned_backend)

    try:
        monkeypatch.setattr(SourceRootAuthority, "require_current", require_current)
        assert source.require_current_root_authority(authority) is None
        assert calls == [(authority, backend)]
    finally:
        authority.close()


@pytest.mark.parametrize("kind", ["closed", "foreign"])
def test_current_root_operation_rejects_real_closed_and_foreign_authorities(
    tmp_path: Path, kind: str,
) -> None:
    _, source, _ = _source(tmp_path)
    authority = source.create_source_root_authority()
    if kind == "closed":
        authority.close()
    else:
        authority.close()
        _, other_source, _ = _source(tmp_path / "other")
        authority = other_source.create_source_root_authority()
    try:
        with pytest.raises(SourceRootAuthorityError):
            source.require_current_root_authority(authority)
    finally:
        authority.close()


def test_root_replacement_is_reproved_after_an_earlier_success(tmp_path: Path) -> None:
    backend, source, _ = _source(tmp_path)
    authority = source.create_source_root_authority()
    try:
        assert source.require_current_root_authority(authority) is None
        backend.sessions_dir.rename(tmp_path / "displaced-root")
        backend.sessions_dir.mkdir()
        with pytest.raises(SourceRootAuthorityError):
            source.require_current_root_authority(authority)
    finally:
        authority.close()


def test_discovery_factory_borrows_exact_root_and_does_not_traverse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, source, _ = _source(tmp_path)
    authority = source.create_source_root_authority()

    def forbidden_iterator(_self):
        pytest.fail("cursor construction traversed the source directory")

    monkeypatch.setattr(SourceDiscoveryCursor, "_open_iterator", forbidden_iterator)
    try:
        cursor = source.create_discovery_cursor_with_root(limits=LIMITS, root_authority=authority)
        assert type(cursor) is SourceDiscoveryCursor
        assert cursor.root_authority is authority
        assert cursor.matches_source_store(source) is True
        cursor.close()
        assert cursor.closed is True
        assert authority.require_current(backend) is None
        owned = source.create_discovery_cursor(limits=LIMITS)
        owned_authority = owned.root_authority
        owned.close()
        assert owned_authority.closed is True
        assert authority.require_current(backend) is None
    finally:
        authority.close()


def test_discovery_factory_retains_live_canonical_constructor_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, source, _ = _source(tmp_path)
    authority = source.create_source_root_authority()
    original = discovery_module.SourceDiscoveryCursor
    calls: list[tuple[object, object, object]] = []

    def construct(owned_backend, *, limits, root_authority):
        calls.append((owned_backend, limits, root_authority))
        return original(owned_backend, limits=limits, root_authority=root_authority)

    try:
        monkeypatch.setattr(discovery_module, "SourceDiscoveryCursor", construct)
        cursor = source.create_discovery_cursor_with_root(limits=LIMITS, root_authority=authority)
        assert type(cursor) is original
        assert calls == [(backend, LIMITS, authority)]
        cursor.close()
        assert authority.require_current(backend) is None
    finally:
        authority.close()


def test_discovery_constructor_failure_cannot_close_the_borrowed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, source, _ = _source(tmp_path)
    authority = source.create_source_root_authority()

    def reject(owned_backend, *, limits, root_authority):
        assert owned_backend is backend and limits is LIMITS and root_authority is authority
        raise RuntimeError("constructor fault")

    try:
        monkeypatch.setattr(discovery_module, "SourceDiscoveryCursor", reject)
        with pytest.raises(RuntimeError, match="constructor fault"):
            source.create_discovery_cursor_with_root(limits=LIMITS, root_authority=authority)
        assert authority.require_current(backend) is None
    finally:
        authority.close()


def test_runtime_store_mismatch_is_rejected_before_root_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, snapshot = _source(tmp_path)
    assert source.store_identity_differs(snapshot.state.store_identity) is False
    source._store_identity = "changed-owner-identity"

    def forbidden_resolve(_path, strict=False):
        pytest.fail("mismatched source reached root resolution")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    with pytest.raises(IdleRuntimeBindingError, match="different source store"):
        _source_material(source, snapshot)


def test_runtime_identity_retains_the_actual_source_snapshot_tuple(tmp_path: Path) -> None:
    _, source, snapshot = _source(tmp_path)
    root = source.resolved_runtime_source_root()
    state = snapshot.state
    assert _source_material(source, snapshot) == (
        state.session_key, state.incarnation, state.revision, state.generation,
        state.store_identity, root, state.checkpoint_id, state.checkpoint_fingerprint,
    )


def test_dormant_scanner_rejects_same_path_different_backend(tmp_path: Path) -> None:
    backend, source, _ = _source(tmp_path)
    authority = source.create_source_root_authority()
    cursor = SourceDiscoveryCursor(backend, limits=LIMITS, root_authority=authority)
    other_backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    other_source = SourceSessionStore(other_backend)
    try:
        assert cursor.matches_source_store(other_source) is False
        with pytest.raises(ValueError, match="exact backend"):
            IdleCompactionScanner(source_store=other_source, discovery=cursor)
    finally:
        cursor.close()
        authority.close()


async def test_dormant_scanner_accepts_distinct_wrapper_over_same_backend(tmp_path: Path) -> None:
    backend, source, _ = _source(tmp_path)
    wrapper = SourceSessionStore(backend)
    authority = source.create_source_root_authority()
    cursor = SourceDiscoveryCursor(backend, limits=LIMITS, root_authority=authority)
    try:
        assert cursor.matches_source_store(wrapper) is True
        scanner = IdleCompactionScanner(source_store=wrapper, discovery=cursor)
        assert scanner.enabled is False
        assert await scanner.scan_once() == 0
        await scanner.aclose()
        assert authority.require_current(backend) is None
    finally:
        cursor.close()
        authority.close()


def test_boolean_owner_observations_do_not_read_or_write_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, source, snapshot = _source(tmp_path)
    path = backend.get_session_path(KEY)
    before = path.read_bytes()

    def forbidden_io(*_args, **_kwargs):
        pytest.fail("identity observation reached source I/O")

    monkeypatch.setattr(source, "_read_unlocked", forbidden_io)
    monkeypatch.setattr(source, "_write", forbidden_io)
    assert source.matches_backend(backend) is True
    assert source.store_identity_differs(snapshot.state.store_identity) is False
    assert path.read_bytes() == before
