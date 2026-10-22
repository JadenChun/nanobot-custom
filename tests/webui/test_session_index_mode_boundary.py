"""Sidebar cache writes must not become authoritative session-mode evidence."""

from pathlib import Path

import pytest

from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import DurableSessionAuthority, OfflineSessionOperator
from nanobot.session.session_mode import SessionModeError
from nanobot.webui import session_list_index as index


@pytest.fixture(autouse=True)
def isolated_webui(tmp_path, monkeypatch):
    webui = tmp_path / "webui"
    webui.mkdir()
    monkeypatch.setattr(index, "get_webui_dir", lambda: webui)


async def test_sidebar_cache_and_stale_cache_temp_do_not_block_first_write(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
    try:
        manager = lease.manager
        assert index.list_webui_sessions(manager) == []
        cache = index._index_path(manager.sessions_dir)
        assert cache.is_file()
        cache.with_suffix(".stale.tmp").write_text("incomplete cache")
        manager.admit_persistent_write("websocket:first")
        assert manager.resolve_session_mode("websocket:first").mode == "canonical"
        assert not cache.is_relative_to(manager.sessions_dir)
    finally:
        assert lease.close is not None
        await lease.close()


async def test_sidebar_cache_does_not_block_copied_legacy_bootstrap(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=root)
    unmanaged.save(Session("websocket:legacy"), fsync=True)
    assert index.list_webui_sessions(unmanaged)
    lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
    try:
        lease.manager.bootstrap_existing_sessions(offline_copy=True)
        assert lease.manager.resolve_session_mode("websocket:legacy").mode == "legacy"
    finally:
        assert lease.close is not None
        await lease.close()


def test_cache_is_separate_for_different_session_roots(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = SessionManager(workspace, sessions_root=tmp_path / "first")
    second = SessionManager(workspace, sessions_root=tmp_path / "second")
    assert first.sessions_dir.name == second.sessions_dir.name
    first_cache = index._index_path(first.sessions_dir)
    second_cache = index._index_path(second.sessions_dir)
    assert first_cache != second_cache
    assert first_cache.is_relative_to(tmp_path / "webui")
    assert second_cache.is_relative_to(tmp_path / "webui")


async def test_unknown_session_namespace_entry_still_blocks_admission(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
    try:
        (lease.manager.sessions_dir / ".unknown-index.json").write_text("{}")
        with pytest.raises(SessionModeError, match="offline recovery is required"):
            lease.manager.admit_persistent_write("websocket:first")
    finally:
        assert lease.close is not None
        await lease.close()
