"""Offline consumer contracts using real session authority and no historical graph."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import nanobot.session.webui_turns as titles
import nanobot.webui.forking as forking
from nanobot.providers.base import LLMResponse
from nanobot.sdk.clients import SessionClient
from nanobot.sdk.types import SessionSnapshot
from nanobot.security.workspace_access import build_workspace_scope
from nanobot.session.execution_owner import (
    ExecutionBusyError,
    ExecutionOwnershipError,
    acquire_execution,
)
from nanobot.session.manager import SessionManager
from nanobot.session.runtime_authority import OfflineSessionOperator
from nanobot.webui.workspaces import WebUIWorkspaceController


@asynccontextmanager
async def managers(tmp_path: Path, *, managed: bool = True, legacy: bool = False):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    if not managed or legacy:
        manager = SessionManager(workspace, sessions_root=root)
        if legacy:
            session = manager.get_or_create("websocket:legacy")
            session.add_message("user", "old")
            manager.save(session)
    if managed:
        lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
        manager = lease.manager
        if legacy:
            manager.bootstrap_existing_sessions(offline_copy=True)
    try:
        yield manager
    finally:
        if managed:
            await lease.close()


def client(manager):
    loop = SimpleNamespace(sessions=manager, discard_session_file_state=MagicMock())
    return SessionClient(loop), loop


def state(manager):
    return {p.relative_to(manager.sessions_dir): (p.read_bytes(), p.stat().st_ino,
            None if p.name.endswith(".lock") else p.stat().st_mtime_ns)
            for p in manager.sessions_dir.rglob("*") if p.is_file()}


def provider(content="Useful title", *, failure=False):
    return SimpleNamespace(chat_with_retry=AsyncMock(
        side_effect=RuntimeError("secret-exception") if failure else None,
        return_value=LLMResponse(content=content, finish_reason="secret-finish"),
    ))


def seed(manager, key="shared:chat", *, user="hello", title=None):
    # The same manager-owned frozen reset seam installs copied canonical test events.
    if manager.has_managed_authority:
        manager.admit_persistent_write(key)
        from nanobot.session.source import new_source_event

        manager.reset_session(key, replacement_events=[
            new_source_event({"role": "user", "content": user}, run_id="fixture"),
            new_source_event({"role": "assistant", "content": "answer"}, run_id="fixture"),
        ])
        manager.upsert_session_metadata(key, {"webui": True})
    else:
        session = manager.get_or_create(key)
        session.add_message("user", user)
        session.add_message("assistant", "answer")
        session.metadata["webui"] = True
        manager.save(session)
    if title is not None:
        manager.upsert_session_metadata(key, {"title": title})


@pytest.mark.parametrize("operation", ["ingest", "restore"])
@pytest.mark.parametrize("kind", ["absent", "legacy", "canonical"])
async def test_saved_managed_import_rejects_before_iteration_cache_or_authority(tmp_path, monkeypatch, operation, kind):
    async with managers(tmp_path, legacy=kind == "legacy") as manager:
        key = "websocket:legacy" if kind == "legacy" else "websocket:secret-key"
        if kind == "canonical":
            manager.admit_persistent_write(key)
        sdk, _loop = client(manager)
        before = state(manager)
        cache = dict(manager._cache)
        def forbidden(*_args, **_kwargs):
            raise AssertionError("import resolved or iterated")
        monkeypatch.setattr(manager, "get_or_create", forbidden)
        monkeypatch.setattr(manager._mode_resolver, "resolve", forbidden)
        class Messages:
            def __iter__(self):
                return forbidden()
        with pytest.raises(RuntimeError, match="^managed session import is unsupported$"):
            if operation == "ingest":
                await sdk.ingest(key, Messages())
            else:
                await sdk.restore(SessionSnapshot(key=key, messages=Messages()))
        assert state(manager) == before
        assert dict(manager._cache) == cache


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("operation", ["ingest", "restore"])
@pytest.mark.parametrize("existing", [False, True])
async def test_detached_import_never_reads_or_changes_existing_session(tmp_path, monkeypatch, managed, operation, existing):
    async with managers(tmp_path, managed=managed) as manager:
        if existing:
            seed(manager, "websocket:existing")
        sdk, _loop = client(manager)
        before = state(manager)
        cache = dict(manager._cache)
        def forbidden(*_args, **_kwargs):
            raise AssertionError("detached import used manager")
        for name in ("get_or_create", "save", "get_cached", "read_session_snapshot"):
            monkeypatch.setattr(manager, name, forbidden)
        if managed:
            monkeypatch.setattr(manager._mode_resolver, "resolve", forbidden)
        messages = [{"role": "user", "content": "detached"}]
        result = (await sdk.ingest("websocket:existing", messages, metadata={"title": "copy"}, save=False)
                  if operation == "ingest" else
                  await sdk.restore(SessionSnapshot(key="websocket:existing", messages=messages, metadata={"title": "copy"}), save=False))
        assert [m["content"] for m in result.messages] == ["detached"]
        assert result.metadata == {"title": "copy"}
        assert state(manager) == before
        assert dict(manager._cache) == cache


@pytest.mark.parametrize("operation", ["ingest", "restore"])
async def test_unmanaged_saved_import_compatible(tmp_path, operation):
    async with managers(tmp_path, managed=False) as manager:
        sdk, _loop = client(manager)
        messages = [{"role": "user", "content": "saved"}]
        result = (await sdk.ingest("cli:saved", messages) if operation == "ingest" else
                  await sdk.restore(SessionSnapshot(key="cli:saved", messages=messages)))
        assert result.messages[0]["content"] == "saved"
        assert manager.read_session_snapshot("cli:saved").messages[0]["content"] == "saved"


async def test_unmanaged_upsert_publishes_committed_cache_without_losing_local_history(tmp_path):
    async with managers(tmp_path, managed=False) as manager:
        session = manager.get_or_create("cli:metadata")
        session.metadata.update({"remove": "old", "local": "kept"})
        session.add_message("user", "durable")
        manager.save(session)
        session.add_message("user", "local unsaved")
        assert manager.upsert_session_metadata(
            session.key, {"title": {"value": "committed"}}, removals=("remove",)
        )
        assert manager.get_cached(session.key) is session
        assert [row["content"] for row in session.messages] == ["durable", "local unsaved"]
        assert "remove" not in session.metadata and "remove" not in session._base_metadata
        assert session.metadata["title"] == session._base_metadata["title"] == {"value": "committed"}
        session.metadata["title"]["value"] = "local change"
        assert session._base_metadata["title"] == {"value": "committed"}
        assert manager.read_session_snapshot(session.key).messages[-1]["content"] == "durable"


async def test_unmanaged_upsert_failure_does_not_publish_cache(tmp_path, monkeypatch):
    async with managers(tmp_path, managed=False) as manager:
        seed(manager, "cli:metadata")
        session = manager.get_or_create("cli:metadata")
        metadata, base = dict(session.metadata), dict(session._base_metadata)
        before = state(manager)
        monkeypatch.setattr(manager._jsonl_store, "update_metadata", MagicMock(side_effect=RuntimeError("fixed")))
        with pytest.raises(RuntimeError, match="^fixed$"):
            manager.upsert_session_metadata(session.key, {"title": "uncommitted"}, removals=("webui",))
        assert state(manager) == before
        assert manager.get_cached(session.key) is session
        assert session.metadata == metadata and session._base_metadata == base


@pytest.mark.parametrize("failure", [False, True])
async def test_unmanaged_upsert_create_preserves_fresh_cached_local_session(tmp_path, monkeypatch, failure):
    async with managers(tmp_path, managed=False) as manager:
        session = manager.get_or_create("cli:fresh")
        session.add_message("user", "local unsaved")
        session.metadata["local"] = "not committed"
        original_update = manager._jsonl_store.update_metadata
        if failure:
            monkeypatch.setattr(manager._jsonl_store, "update_metadata", MagicMock(side_effect=RuntimeError("fixed")))
            with pytest.raises(RuntimeError, match="^fixed$"):
                manager.upsert_session_metadata(session.key, {"title": "committed"}, create_if_absent=True)
            assert session.metadata == {"local": "not committed"}
            assert session._base_metadata == {}
        else:
            assert manager.upsert_session_metadata(session.key, {"title": "committed"}, create_if_absent=True)
            assert session.metadata == {"local": "not committed", "title": "committed"}
            assert session._base_metadata == {"title": "committed"}
            manager.save(session)
            assert manager.read_session_snapshot(session.key).messages[-1]["content"] == "local unsaved"
        assert manager.get_cached(session.key) is session
        assert session.messages[-1]["content"] == "local unsaved"
        if failure:
            assert manager.read_session_snapshot(session.key).messages == []
            monkeypatch.setattr(manager._jsonl_store, "update_metadata", original_update)
            assert manager.upsert_session_metadata(session.key, {"title": "retry"}, create_if_absent=True)
            assert manager.get_cached(session.key) is session
            manager.save(session)
            assert manager.read_session_snapshot(session.key).messages[-1]["content"] == "local unsaved"


async def test_unmanaged_upsert_create_publishes_new_committed_session(tmp_path):
    async with managers(tmp_path, managed=False) as manager:
        assert manager.get_cached("cli:created") is None
        assert manager.upsert_session_metadata("cli:created", {"title": "committed"}, create_if_absent=True)
        created = manager.get_cached("cli:created")
        assert created is not None and created.metadata == {"title": "committed"}
        assert created._base_metadata == created.metadata and created.policy.persist


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
async def test_sdk_clear_committed_snapshot_and_file_state_order(tmp_path, managed, existing):
    async with managers(tmp_path, managed=managed) as manager:
        key = "websocket:clear"
        if existing:
            seed(manager, key)
        sdk, loop = client(manager)
        def check_committed(_key):
            assert manager.read_session_snapshot(key).messages == []
        loop.discard_session_file_state.side_effect = check_committed
        result = sdk.clear(key)
        assert result.messages == []
        assert manager.read_session_snapshot(key).messages == []
        loop.discard_session_file_state.assert_called_once_with(key)
        if managed and not existing:
            assert manager.resolve_session_mode(key).generation == 0


async def test_sdk_busy_clear_keeps_file_state_cache_and_bytes(tmp_path):
    async with managers(tmp_path) as manager:
        seed(manager, "websocket:clear")
        cached = manager.get_or_create("websocket:clear")
        sdk, loop = client(manager)
        async with acquire_execution(manager._jsonl_store, "websocket:clear", run_id="held"):
            before = state(manager)
            with pytest.raises(ExecutionBusyError):
                sdk.clear("websocket:clear")
            assert state(manager) == before
            assert manager._cache["websocket:clear"] is cached
            loop.discard_session_file_state.assert_not_called()


async def test_sdk_recovery_clear_keeps_file_state_cache_and_bytes(tmp_path):
    async with managers(tmp_path) as manager:
        seed(manager, "websocket:clear")
        cached = manager.get_or_create("websocket:clear")
        sdk, loop = client(manager)
        rows = [json.loads(row) for row in manager._get_session_path(cached.key).read_bytes().splitlines()]
        rows[0]["revision"] += 1
        rows[0]["metadata"]["execution_recovery_required"] = True
        manager._mode_resolver.storage.write(
            f"{manager._jsonl_store.storage_key(cached.key)}.jsonl",
            b"".join((json.dumps(row) + "\n").encode() for row in rows),
        )
        before = state(manager)
        with pytest.raises(ExecutionOwnershipError) as error:
            sdk.clear(cached.key)
        assert "websocket:clear" not in str(error.value)
        assert str(manager.sessions_dir) not in str(error.value)
        assert state(manager) == before
        assert manager._cache[cached.key] is cached
        loop.discard_session_file_state.assert_not_called()


@pytest.mark.parametrize("operation", ["ingest", "restore"])
@pytest.mark.parametrize("save", [False, True])
async def test_import_validates_explicit_empty_key_first(tmp_path, operation, save):
    async with managers(tmp_path) as manager:
        sdk, _loop = client(manager)
        before = state(manager)
        with pytest.raises(ValueError, match="session key"):
            if operation == "ingest":
                await sdk.ingest("", [], save=save)
            else:
                await sdk.restore(SessionSnapshot(key="valid", messages=[]), session_key="", save=save)
        assert state(manager) == before


@pytest.mark.parametrize("operation", ["ingest", "restore"])
async def test_detached_import_retains_message_validation(tmp_path, operation):
    async with managers(tmp_path) as manager:
        sdk, _loop = client(manager)
        before = state(manager)
        with pytest.raises(ValueError, match="content"):
            if operation == "ingest":
                await sdk.ingest("cli:copy", [{"role": "user"}], save=False)
            else:
                await sdk.restore(SessionSnapshot(key="cli:copy", messages=[{"role": "user"}]), save=False)
        assert state(manager) == before and not manager._cache


@pytest.mark.parametrize("after_turn", [False, True])
async def test_projected_title_only_after_durable_turn_can_create_target(tmp_path, after_turn):
    async with managers(tmp_path) as manager:
        seed(manager)
        before = state(manager)
        if after_turn:
            generated = await titles.maybe_generate_webui_title_after_turn(channel="websocket", chat_id="projected", metadata={"webui": True}, sessions=manager, session_key="shared:chat", provider=provider(), model="fake")
        else:
            generated = await titles.maybe_generate_webui_title(sessions=manager, session_key="shared:chat", target_session_key="websocket:projected", provider=provider(), model="fake")
        assert generated is after_turn
        target = manager.read_session_snapshot("websocket:projected")
        if after_turn:
            assert target.messages == [] and target.metadata["title"] == "Useful title"
            assert manager.resolve_session_mode(target.key).record_revision == 1
        else:
            assert target is None and state(manager) == before
        assert not manager._cache


@pytest.mark.parametrize("case", ["draft", "temporary", "failure", "unusable", "existing", "no_user", "missing_durability"])
async def test_projected_title_negative_cases_noncreating_and_content_free(tmp_path, monkeypatch, case):
    async with managers(tmp_path) as manager:
        routed_key = "shared:chat"
        if case in {"temporary", "missing_durability"}:
            routed_key = "transient:runtime" if case == "temporary" else "shared:chat"
            transient = manager.get_or_create(routed_key)
            transient.metadata["webui"] = True
            transient.add_message("user", "transient user content")
        if case not in {"draft", "temporary", "missing_durability"}:
            seed(manager, user="" if case == "no_user" else "hello")
        if case == "existing":
            manager.upsert_session_metadata("websocket:secret-key", {"title": "Existing"}, create_if_absent=True)
        logs = []
        monkeypatch.setattr(titles, "logger", SimpleNamespace(debug=lambda *args, **kwargs: logs.append((args, kwargs))))
        before = state(manager)
        generated = await titles.maybe_generate_webui_title_after_turn(channel="websocket", chat_id="secret-key", metadata={"webui": True}, sessions=manager, session_key=routed_key, provider=provider("error secret-content" if case == "unusable" else "Useful title", failure=case == "failure"), model="fake")
        assert generated is False
        assert state(manager) == before
        assert all(secret not in repr(logs) for secret in ("secret-key", "secret-exception", "secret-finish", "secret-content"))
        assert all(not kwargs for _args, kwargs in logs)


async def test_canonical_title_cleaning_and_generation_use_metadata_seam(tmp_path):
    async with managers(tmp_path) as manager:
        seed(manager, "websocket:title", title='"Existing title."')
        assert not await titles.maybe_generate_webui_title(sessions=manager, session_key="websocket:title", provider=provider(), model="fake")
        assert manager.read_session_snapshot("websocket:title").metadata["title"] == "Existing title"
        manager.upsert_session_metadata("websocket:title", {"title": "<think>private</think>"})
        assert await titles.maybe_generate_webui_title(sessions=manager, session_key="websocket:title", provider=provider(), model="fake")
        assert manager.read_session_snapshot("websocket:title").metadata["title"] == "Useful title"


async def test_managed_fork_title_and_failure_rollback(tmp_path, monkeypatch):
    async with managers(tmp_path) as manager:
        seed(manager, "websocket:source")
        for name in ("append_fork_marker", "delete_webui_transcript", "write_session_messages_as_transcript"):
            monkeypatch.setattr(forking, name, MagicMock())
        monkeypatch.setattr(forking, "fork_transcript_before_user_index", lambda *_args: True)
        result = forking.create_webui_chat_fork(manager, source_chat_id="source", before_user_index=1, title='"Fork title."')
        assert manager.read_session_snapshot(result[1]).metadata["title"] == "Fork title"
        monkeypatch.setattr(forking.uuid, "uuid4", lambda: "failure")
        monkeypatch.setattr(manager, "upsert_session_metadata", MagicMock(side_effect=RuntimeError("fixed")))
        with pytest.raises(RuntimeError, match="fixed"):
            forking.create_webui_chat_fork(manager, source_chat_id="source", before_user_index=1, title="Failure")
        assert manager.resolve_session_mode("websocket:failure").state == "deleted"


async def test_workspace_draft_first_accepted_message_update_and_temporary(tmp_path):
    async with managers(tmp_path) as manager:
        controller = WebUIWorkspaceController(session_manager=manager, default_workspace=manager.workspace, default_restrict_to_workspace=True)
        scope = build_workspace_scope(manager.workspace, "restricted", source_channel="websocket")
        before = state(manager)
        controller.stage_scope("draft", scope)
        controller.stage_scope("temporary", scope)
        assert state(manager) == before and not manager._cache
        controller.discard_draft_scope("websocket:temporary")
        controller.persist_scope("draft", scope)
        target = manager.read_session_snapshot("websocket:draft")
        assert target.metadata["webui"] is True
        assert target.metadata["workspace_scope"] == scope.metadata()
        assert target.messages == []
        revision = manager.resolve_session_mode(target.key).record_revision
        full = build_workspace_scope(manager.workspace, "full", source_channel="websocket")
        controller.stage_scope("draft", full)
        assert manager.read_session_snapshot(target.key).metadata["workspace_scope"] == full.metadata()
        assert manager.resolve_session_mode(target.key).record_revision == revision
        assert manager.read_session_snapshot("websocket:temporary") is None
