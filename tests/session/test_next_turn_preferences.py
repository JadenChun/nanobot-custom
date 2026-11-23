"""Routine next-turn preference I/O is bounded and independent of settled history."""

import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from kohaku.session.manager import SessionWriteConflict
from kohaku.session.next_turn_preferences import (
    MAX_PREFERENCE_BYTES,
    NextTurnPreferenceError,
    preference_path,
)
from kohaku.session.runtime_authority import DurableSessionAuthority
from kohaku.session.source import new_source_event
from kohaku.session.source_store import SourceSessionStore

KEY = "websocket:prefs"


@asynccontextmanager
async def _managed(root: Path):
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    lease = await DurableSessionAuthority(workspace, sessions_root=root / "sessions").acquire()
    try:
        yield lease.manager
    finally:
        assert lease.close is not None
        await lease.close()


def _seed(manager, prompts: int, key: str = KEY) -> None:
    assert manager.upsert_session_metadata(key, {"webui": True}, create_if_absent=True)
    source = manager._jsonl_store.source
    state = source.load_source_snapshot(key).state
    source.append_source_events(key, state, [
        new_source_event({"role": "user", "content": f"prompt {i}"}, run_id="t")
        for i in range(prompts)
    ])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("prompts", [10, 100, 1000])
async def test_routine_reads_and_writes_do_not_touch_settled_history(
    tmp_path, monkeypatch, prompts,
):
    async with _managed(tmp_path) as manager:
        _seed(manager, prompts)
        path = manager._get_session_path(KEY)
        before = (path.stat().st_ino, path.stat().st_mtime_ns, _digest(path))
        state = manager._jsonl_store.source.load_source_snapshot(KEY).state

        def forbidden(*args, **kwargs):
            pytest.fail("routine preference I/O scanned or rewrote canonical events")

        read_bytes = [0]
        real_read = os.read

        def counted(fd, length):
            data = real_read(fd, length)
            read_bytes[0] += len(data)
            return data

        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, "_read_unlocked", forbidden)
            patch.setattr(SourceSessionStore, "_write", forbidden)
            patch.setattr(SourceSessionStore, "prepare_source_persistence", forbidden)
            patch.setattr(os, "read", counted)
            for index in range(5):
                saved = manager.update_next_turn_preferences(KEY, {
                    "model_preset": f"preset-{index}", "reasoning_effort": "high",
                    "workspace_scope": {"project_path": "/b", "access_mode": "restricted"},
                })
                assert saved is not None and saved.revision == index + 1
            loaded = manager.read_next_turn_preferences(KEY)
        assert loaded is not None and loaded.get("model_preset") == "preset-4"
        # Identical bounded I/O regardless of history length: header line + one record.
        assert read_bytes[0] < 6 * (MAX_PREFERENCE_BYTES + 4096)
        assert (path.stat().st_ino, path.stat().st_mtime_ns, _digest(path)) == before
        assert manager._jsonl_store.source.load_source_snapshot(KEY).state == state


async def test_io_volume_is_identical_for_short_and_long_history(tmp_path, monkeypatch):
    volumes = []
    for prompts, name in ((10, "run_0010"), (300, "run_0300")):
        key = KEY
        async with _managed(tmp_path / name) as manager:
            _seed(manager, prompts, key)
            total = [0]
            real_read = os.read

            def counted(fd, length, total=total, real_read=real_read):
                data = real_read(fd, length)
                total[0] += len(data)
                return data

            def forbidden(*args, **kwargs):
                pytest.fail("routine metadata or preference I/O scanned or rewrote canonical events")

            with monkeypatch.context() as patch:
                patch.setattr(os, "read", counted)
                patch.setattr(SourceSessionStore, "_read_unlocked", forbidden)
                patch.setattr(SourceSessionStore, "_write", forbidden)
                patch.setattr(SourceSessionStore, "prepare_source_persistence", forbidden)
                manager.update_next_turn_preferences(key, {"model_preset": "m"})
                manager.read_next_turn_preferences(key)
                assert manager.upsert_session_metadata(
                    key,
                    {
                        "_nanobot_model_preset": "fast",
                        "_nanobot_reasoning_effort": "high",
                        "workspace_scope": {
                            "project_path": str((tmp_path / name / "workspace").resolve()),
                            "access_mode": "restricted",
                        },
                    },
                )
                assert manager.update_session_metadata(key, {"title": "Updated"})
                assert manager.read_session_metadata(key) is not None
                assert manager.read_saved_source_metadata(key) is not None
            volumes.append(total[0])
    assert volumes[0] == volumes[1]


async def test_header_state_and_history_are_unchanged_and_recovery_survives_restart(tmp_path):
    async with _managed(tmp_path) as manager:
        _seed(manager, 12)
        state = manager._jsonl_store.source.load_source_snapshot(KEY).state
        manager.update_next_turn_preferences(KEY, {"model_preset": "next"})
        assert manager._jsonl_store.source.load_source_snapshot(KEY).state == state
        assert "model_preset" not in manager.read_session_metadata(KEY)["metadata"]
    async with _managed(tmp_path) as manager:
        loaded = manager.read_next_turn_preferences(KEY)
        assert loaded is not None
        assert loaded.get("model_preset") == "next" and loaded.revision == 1
        assert manager._jsonl_store.source.load_source_snapshot(KEY).state == state


async def test_conditional_write_and_consume_never_erase_a_newer_choice(tmp_path):
    async with _managed(tmp_path) as manager:
        _seed(manager, 3)
        first = manager.update_next_turn_preferences(KEY, {"model_preset": "A"})
        assert first is not None
        second = manager.update_next_turn_preferences(
            KEY, {"model_preset": "B"}, expected_revision=first.revision,
        )
        assert second is not None
        with pytest.raises(SessionWriteConflict):
            manager.update_next_turn_preferences(
                KEY, {"model_preset": "stale"}, expected_revision=first.revision,
            )
        # An admission that applied A must not retire the later B.
        remaining = manager.consume_next_turn_preferences(KEY, {"model_preset": "A"})
        assert remaining is not None and remaining.get("model_preset") == "B"
        assert manager.consume_next_turn_preferences(KEY, {"model_preset": "B"}) is None
        assert manager.read_next_turn_preferences(KEY) is None
        assert not preference_path(manager._jsonl_store, KEY).exists()


async def test_invalid_input_and_oversize_are_rejected_without_state(tmp_path):
    async with _managed(tmp_path) as manager:
        _seed(manager, 2)
        with pytest.raises(NextTurnPreferenceError):
            manager.update_next_turn_preferences(KEY, {"title": "not a next-turn key"})
        with pytest.raises(NextTurnPreferenceError):
            manager.update_next_turn_preferences(KEY, {"model_preset": "x" * MAX_PREFERENCE_BYTES})
        assert manager.read_next_turn_preferences(KEY) is None
        assert manager.update_next_turn_preferences("websocket:absent", {"model_preset": "x"}) is None


async def test_reset_delete_and_recreate_orphan_preferences(tmp_path):
    async with _managed(tmp_path) as manager:
        _seed(manager, 4)
        manager.update_next_turn_preferences(KEY, {"model_preset": "old"})
        path = preference_path(manager._jsonl_store, KEY)
        assert path.exists()
        assert manager.delete_session(KEY)
        assert not path.exists()
        assert manager.read_next_turn_preferences(KEY) is None
        # A recreated source is a new incarnation and never inherits stale choices.
        _seed(manager, 1)
        assert manager.read_next_turn_preferences(KEY) is None
        manager.update_next_turn_preferences(KEY, {"model_preset": "new"})
        stale = path.read_bytes()
        assert manager.delete_session(KEY)
        _seed(manager, 1)
        path.write_bytes(stale)  # Simulate a crash that left the old record behind.
        assert manager.read_next_turn_preferences(KEY) is None


async def test_fork_does_not_share_or_mutate_the_source_preferences(tmp_path):
    async with _managed(tmp_path) as manager:
        _seed(manager, 3)
        manager.update_next_turn_preferences(KEY, {"model_preset": "src"})
        other = "websocket:forked"
        forked = manager.fork_session_before_user_index(KEY, other, 2)
        assert forked is not None
        assert len(forked.messages) == 2
        assert manager.read_next_turn_preferences(other) is None
        manager.update_next_turn_preferences(other, {"model_preset": "fork"})
        assert manager.read_next_turn_preferences(KEY).get("model_preset") == "src"
        assert manager.read_next_turn_preferences(other).get("model_preset") == "fork"
        assert manager.reset_session(KEY) is not None
        assert manager.read_next_turn_preferences(KEY) is None


async def test_damaged_record_is_ignored_and_replaced(tmp_path):
    async with _managed(tmp_path) as manager:
        _seed(manager, 2)
        manager.update_next_turn_preferences(KEY, {"model_preset": "ok"})
        path = preference_path(manager._jsonl_store, KEY)
        path.write_text("{not json")
        assert manager.read_next_turn_preferences(KEY) is None
        replaced = manager.update_next_turn_preferences(KEY, {"model_preset": "again"})
        assert replaced is not None and replaced.revision == 1
        assert (path.stat().st_mode & 0o777) == 0o600


async def test_active_turn_isolates_next_turn_folder_and_model_across_steer_stop_recovery_and_once_grants(
    tmp_path,
    monkeypatch,
):
    import asyncio

    from kohaku.agent.loop import AgentLoop
    from kohaku.agent.turn import DeliveryTarget, TurnRequest, TurnSource
    from kohaku.config.schema import Config, ModelPresetConfig
    from kohaku.providers.base import GenerationSettings, LLMProvider, LLMResponse
    from kohaku.providers.factory import ProviderSnapshot
    from kohaku.runtime_composition import RuntimeComposition
    from kohaku.security.folder_grants import FolderGrantStore
    from kohaku.security.workspace_access import WorkspaceScopeError
    from kohaku.session.model_selection import (
        model_preset_from_metadata,
        reasoning_effort_from_metadata,
    )
    from kohaku.session.webui_turns import (
        activate_queued_websocket_turn,
        bind_queued_websocket_turn_scope,
        clear_websocket_turn_if_current,
        register_queued_websocket_turn_if_idle,
    )
    from kohaku.session.workspace_access_recovery import WorkspaceAccessRecoveryOwner
    from kohaku.webui.inbound_commands import steering_target_error
    from kohaku.webui.session_projection import WebUISessionProjection
    from kohaku.webui.workspaces import WebUIWorkspaceController

    class _GatedProvider(LLMProvider):
        def __init__(self, name: str) -> None:
            super().__init__(provider_name="openai")
            self.name = name
            self.generation = GenerationSettings(max_tokens=256, temperature=0.1)
            self.calls: list[tuple[str | None, list[dict[str, object]], str | None]] = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.block = False

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.calls.append((model, list(messages), kwargs.get("reasoning_effort")))
            if self.block:
                self.started.set()
                await self.release.wait()
            return LLMResponse(content=f"reply from {self.name}", finish_reason="stop")

        def get_default_model(self) -> str:
            return self.name

    workspace = tmp_path / "workspace"
    folder_a = tmp_path / "folder-a"
    folder_b = tmp_path / "folder-b"
    unrelated = tmp_path / "unrelated"
    workspace.mkdir()
    folder_a.mkdir()
    folder_b.mkdir()
    unrelated.mkdir()

    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.tools.restrict_to_workspace = True
    presets = {
        "fast": ModelPresetConfig(
            provider="openai", model="openai/gpt-4.1", context_window_tokens=16_000,
        ),
        "deep": ModelPresetConfig(
            provider="openai", model="openai/o3", context_window_tokens=32_000,
        ),
    }
    config.model_presets = presets
    fast_provider = _GatedProvider("openai/gpt-4.1")
    deep_provider = _GatedProvider("openai/o3")
    providers = {"fast": fast_provider, "deep": deep_provider}
    monkeypatch.setattr("kohaku.providers.factory.make_provider", lambda _config: fast_provider)

    def load_preset(name: str) -> ProviderSnapshot:
        preset = presets[name]
        return ProviderSnapshot(
            provider=providers[name],
            model=preset.model,
            context_window_tokens=preset.context_window_tokens,
            signature=(name, preset.model),
        )

    sessions_root = tmp_path / "sessions"
    grants = FolderGrantStore.for_workspace(workspace, sessions_root=sessions_root)
    # Keep managed mode active via an unrelated grant; grant folder_a initially.
    grants.upsert(
        {"path": str(unrelated), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    grants.upsert(
        {"path": str(folder_a), "file_access": "read_write", "commands": False},
        expected_revision=grants.snapshot().revision,
    )

    composition = RuntimeComposition.define(
        config,
        sessions_root=sessions_root,
        startup_receipt_admitted=True,
    )
    handle = await composition.acquire()
    chat_id = "split-chat"
    key = f"websocket:{chat_id}"
    owner_token: str | None = None
    try:
        loop: AgentLoop = handle.agent
        loop.runtime_resolver._model_presets = dict(presets)
        loop.runtime_resolver._preset_snapshot_loader = load_preset
        workspaces = WebUIWorkspaceController(
            session_manager=handle.sessions,
            default_workspace=workspace,
            default_restrict_to_workspace=True,
            grant_store=grants,
        )
        recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
        projection = WebUISessionProjection(
            handle.sessions, stop_target_for_key=loop.stop_target_for_key,
        )

        # Configure initial session for folder_a and model "fast".
        scope_a = workspaces.scope_for_set_request(
            {"workspace_scope": {"project_path": str(folder_a), "access_mode": "restricted"}},
            chat_id=chat_id,
            chat_running=False,
            controls_available=True,
        )
        workspaces.persist_scope(chat_id, scope_a)
        loop.set_session_model_preset(key, "fast")

        # Seed one turn in folder_a that pauses with a pending folder access recovery request.
        first = await loop.execute_turn(TurnRequest(
            content="Initial turn in A",
            session_key=key,
            source=TurnSource.GATEWAY,
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "turn-0"},
        ))
        assert first.content == "reply from openai/gpt-4.1"

        backend = handle.sessions.source_backend(key)
        assert backend is not None
        snap0 = backend.source.load_source_snapshot(key)
        input_id = next(e["_event_id"] for e in snap0.events if e.get("role") == "user")
        run_id = next(e["_run_id"] for e in snap0.events if e.get("role") == "user")
        binding = {
            "store_identity": snap0.state.store_identity,
            "incarnation": snap0.state.incarnation,
            "generation": snap0.state.generation,
        }
        record = {
            "version": 1,
            "access_request_id": "b" * 32,
            "revision": 1,
            "task": {"turn_id": "turn-0"},
            "blocked_turn_id": "turn-0",
            "tool_call_id": None,
            "created_at_ms": 1,
            "status": "pending",
            "reason": "folder_not_granted",
            "operation": "read",
            "target": {"kind": "directory", "path": str(folder_a)},
            "purpose": "Inspect folder A once",
            "selected_path": None,
            "continuation_turn_id": None,
            "source_binding": binding,
            "task_input_event_id": input_id,
            "latest_input_event_id": input_id,
            "blocked_run_id": run_id,
            "expected_scope": scope_a.metadata(),
            "resolved_grants_revision": None,
        }
        handle.sessions.commit_workspace_access_request(
            key,
            expected_binding=binding,
            expected_record=None,
            replacement_record=record,
            expected_latest_input_event_id=input_id,
        )
        # Revoke permanent grant on folder_a and grant Once access to folder_a instead,
        # so folder_a is authorized ONLY by the Once grant.
        grants.revoke(str(folder_a), expected_revision=grants.snapshot().revision)
        perms = recovery.workspace_permissions(chat_id, controls_available=True)
        ready = recovery.allow_once({
            "chat_id": chat_id,
            "access_request_id": record["access_request_id"],
            "expected_revision": 1,
            "path": str(folder_a),
            "file_access": "read_only",
            "commands": False,
            "expected_grants_revision": grants.snapshot().revision,
            "expected_runtime_id": perms["runtime_id"],
            "expected_permission_revision": perms["revision"],
        })
        assert ready["workspace_access_request"]["status"] == "ready"
        once_ref = handle.sessions.desktop_permissions.task_reference(
            session_key=key,
            source_binding=binding,
            task_turn_id="turn-0",
            task_input_event_id=input_id,
        )
        assert once_ref is not None

        # Choosing a next-turn folder while recovery is ready must not invalidate the recovery request.
        scope_unrelated = workspaces._scope_for_mode(unrelated, "restricted")
        workspaces.stage_scope(chat_id, scope_unrelated)
        assert recovery.get(chat_id)["status"] == "ready"
        assert workspaces.scope_for_session_key(key).project_path == folder_a.resolve()
        handle.sessions.consume_next_turn_preferences(
            key, {"workspace_scope": scope_unrelated.metadata()},
        )

        # Start active Turn A in folder_a with model "fast", blocking in the provider.
        fast_provider.block = True
        owner_token = register_queued_websocket_turn_if_idle(chat_id, "turn-a", publish=False)
        assert owner_token is not None
        bind_queued_websocket_turn_scope(chat_id, owner_token, scope_a.metadata())
        activate_queued_websocket_turn(chat_id, owner_token)

        turn_a_task = asyncio.create_task(loop.execute_turn(TurnRequest(
            content="Running turn A",
            session_key=key,
            source=TurnSource.GATEWAY,
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "turn-a"},
        )))
        await asyncio.wait_for(fast_provider.started.wait(), timeout=5)

        snap_a = backend.source.load_source_snapshot(key)
        turn_a_event = next(
            e for e in reversed(snap_a.events)
            if e.get("role") == "user" and e.get("content") == "Running turn A"
        )
        active_record = {
            **record,
            "access_request_id": "c" * 32,
            "revision": 1,
            "task": {"turn_id": "turn-a"},
            "blocked_turn_id": "turn-a",
            "status": "pending",
            "task_input_event_id": turn_a_event["_event_id"],
            "latest_input_event_id": turn_a_event["_event_id"],
            "blocked_run_id": turn_a_event["_run_id"],
        }
        handle.sessions.commit_workspace_access_request(
            key,
            expected_binding=binding,
            expected_record=snap_a.header["metadata"].get("workspace_access_request"),
            replacement_record=active_record,
            expected_latest_input_event_id=turn_a_event["_event_id"],
        )

        # While Turn A is running:
        # 1. Legacy in-place set_workspace_scope without next_turn is still rejected with chat_running.
        with pytest.raises(WorkspaceScopeError, match="chat_running"):
            workspaces.scope_for_set_request(
                {"workspace_scope": {"project_path": str(folder_b), "access_mode": "restricted"}},
                chat_id=chat_id,
                chat_running=True,
                controls_available=True,
            )

        # 2. Record folder_b as next-turn preference (even before folder_b has a permanent grant,
        # or via update_next_turn_preferences / set_next_turn_scope) and model "deep" + effort "high".
        handle.sessions.update_next_turn_preferences(
            key,
            {"workspace_scope": {"project_path": str(folder_b), "access_mode": "restricted"}},
        )
        selected_runtime = loop.set_session_model_preset(
            key, "deep", reasoning_effort="high",
        )
        assert selected_runtime.model == "openai/o3"
        assert selected_runtime.generation.reasoning_effort == "high"

        # 3. Verify Turn A keeps its effective scope (folder_a), model ("fast"), Steer, recovery, and Once grant.
        active_scope = workspaces.scope_for_session_key(key)
        assert active_scope.project_path == folder_a.resolve()
        next_scope = workspaces.next_scope_for_session_key(key)
        assert next_scope is not None and next_scope.project_path == folder_b.resolve()
        active_meta = handle.sessions.read_session_metadata(key)["metadata"]
        assert model_preset_from_metadata(active_meta) == "fast"
        assert reasoning_effort_from_metadata(active_meta) is None

        # Steer for Turn A remains valid against Turn A's scope (folder_a), and does not consume next-turn B.
        steer_envelope = {
            "type": "message",
            "chat_id": chat_id,
            "content": "Steer turn A",
            "webui": True,
            "request_settings_version": 1,
            "request_intent": "steer",
            "target_turn_id": "turn-a",
        }
        steer_scope = workspaces.scope_for_message(
            steer_envelope, chat_id=chat_id, chat_running=True, controls_available=True,
        )
        assert steer_scope.project_path == folder_a.resolve()
        assert steering_target_error(steer_envelope, chat_id=chat_id, scope=steer_scope) is None

        # Once grant and recovery request remain intact while Turn A runs.
        assert handle.sessions.desktop_permissions.is_current_task_reference(once_ref)
        current_recovery = recovery.get(chat_id)
        assert current_recovery is not None and current_recovery["status"] == "pending"

        # Attach fields expose confirmed active vs next-turn preferences cleanly.
        attach = projection.attach_fields(key)
        assert attach["model_preset"] == "fast"
        assert attach["next_model_preset"] == "deep"
        assert attach["next_reasoning_effort"] == "high"
        assert attach["next_workspace_scope"]["project_path"] == str(folder_b)

        # Complete Turn A and verify it does NOT overwrite next-turn preferences for B.
        fast_provider.block = False
        fast_provider.release.set()
        turn_a_result = await asyncio.wait_for(turn_a_task, timeout=5)
        assert turn_a_result.content == "reply from openai/gpt-4.1"
        clear_websocket_turn_if_current(chat_id, owner_token)
        owner_token = None

        prefs_after_a = handle.sessions.read_next_turn_preferences(key)
        assert prefs_after_a is not None
        assert prefs_after_a.get("model_preset") == "deep"
        assert prefs_after_a.get("reasoning_effort") == "high"
        assert prefs_after_a.get("workspace_scope")["project_path"] == str(folder_b)
        assert workspaces.scope_for_session_key(key).project_path == folder_a.resolve()

        # 4. Next new turn admission with an unauthorized/forbidden folder (private runtime storage):
        # Must retire the Once grant and refuse admission without falling back to folder_a!
        forbidden = sessions_root / "private-forbidden"
        forbidden.mkdir(parents=True, exist_ok=True)
        handle.sessions.update_next_turn_preferences(
            key,
            {"workspace_scope": {"project_path": str(forbidden), "access_mode": "restricted"}},
        )
        with pytest.raises(WorkspaceScopeError, match="folder_access_required"):
            workspaces.scope_for_message(
                {"type": "message", "chat_id": chat_id, "content": "Turn B unauthorized"},
                chat_id=chat_id,
                chat_running=False,
                controls_available=True,
            )
        assert not handle.sessions.desktop_permissions.is_current_task_reference(once_ref)
        assert workspaces.scope_for_session_key(key).project_path == folder_a.resolve()

        # Direct execute_turn also refuses the forbidden next-turn folder without fallback to folder_a.
        failed_turn_b = await loop.execute_turn(TurnRequest(
            content="Turn B direct unauthorized",
            session_key=key,
            source=TurnSource.GATEWAY,
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "turn-b-denied"},
        ))
        assert failed_turn_b.stop_reason == "error"
        assert "folder_access_required" in (failed_turn_b.error or "")
        assert len(deep_provider.calls) == 0

        # 5. Grant folder_b permanently and select it; now the next new turn consumes B (folder_b + model "deep" + effort "high").
        grants.upsert(
            {"path": str(folder_b), "file_access": "read_write", "commands": False},
            expected_revision=grants.snapshot().revision,
        )
        handle.sessions.update_next_turn_preferences(
            key,
            {"workspace_scope": {"project_path": str(folder_b), "access_mode": "restricted"}},
        )
        turn_b_result = await loop.execute_turn(TurnRequest(
            content="Turn B authorized",
            session_key=key,
            source=TurnSource.GATEWAY,
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "turn-b-ok"},
        ))
        assert turn_b_result.content == "reply from openai/o3"
        assert len(deep_provider.calls) == 1
        assert deep_provider.calls[0][0] == "openai/o3"
        assert deep_provider.calls[0][2] == "high"
        assert str(folder_b.resolve()) in str(deep_provider.calls[0][1][0].get("content", ""))

        # Next-turn preferences were consumed and promoted into confirmed session metadata.
        assert handle.sessions.read_next_turn_preferences(key) is None
        assert workspaces.scope_for_session_key(key).project_path == folder_b.resolve()
        confirmed_meta = handle.sessions.read_session_metadata(key)["metadata"]
        assert model_preset_from_metadata(confirmed_meta) == "deep"
        assert reasoning_effort_from_metadata(confirmed_meta) == "high"
    finally:
        fast_provider.release.set()
        if owner_token is not None:
            clear_websocket_turn_if_current(chat_id, owner_token)
        await composition.aclose()
