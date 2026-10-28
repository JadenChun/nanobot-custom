"""Typed workspace access recovery decisions and effective capability edges."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.context_repo.policy import FolderAccessNeededError
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.webui import workspaces as workspace_api
from nanobot.workspace_access_recovery import (
    parse_folder_access_need,
    parse_workspace_access_action_payload,
)


def _need(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "version": 1,
        "reason": "folder_not_granted",
        "operation": "read",
        "target": {"kind": "file", "path": "/projects/alpha/README.md"},
        "purpose": "Read the requested project README",
        "allowed_roots": ["/agent", "/projects/known"],
        "roots_truncated": False,
    }
    value.update(overrides)
    return value


def test_folder_access_need_parser_accepts_known_and_unknown_targets() -> None:
    known = parse_folder_access_need(_need())
    assert known["target"] == {"kind": "file", "path": "/projects/alpha/README.md"}
    unknown = parse_folder_access_need(_need(
        reason="commands_not_granted",
        operation="execute",
        target={"kind": "unknown", "path": None},
    ))
    assert unknown["target"] == {"kind": "unknown", "path": None}


@pytest.mark.parametrize("change", [
    {"version": True},
    {"extra": "ignored fields are not authority"},
    {"reason": "private_path"},
    {"operation": "write_everything"},
    {"target": {"kind": "unknown", "path": "/projects/alpha"}},
    {"target": {"kind": "directory", "path": "relative/project"}},
    {"target": {"kind": "file", "path": "/" + "x" * 2048}},
    {"purpose": ""},
    {"allowed_roots": ["relative"]},
])
def test_folder_access_need_parser_rejects_malformed_or_oversized_records(change) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_folder_access_need(_need(**change))


def test_workspace_access_action_payloads_have_exact_typed_edges() -> None:
    identity = {
        "chat_id": "chat-a",
        "access_request_id": "0" * 32,
        "expected_revision": 1,
    }
    assert parse_workspace_access_action_payload(identity, resolve=False) == identity
    resolve = {
        **identity,
        "path": "~/projects/alpha",
        "expected_grants_revision": 0,
    }
    assert parse_workspace_access_action_payload(resolve, resolve=True) == resolve


@pytest.mark.parametrize(("resolve", "change"), [
    (False, {"unexpected": True}),
    (False, {"expected_revision": True}),
    (False, {"chat_id": "invalid/chat"}),
    (False, {"access_request_id": "not-an-id"}),
    (True, {"path": ""}),
    (True, {"path": "/tmp/\x00project"}),
    (True, {"expected_grants_revision": True}),
    (True, {"extra": 1}),
])
def test_workspace_access_action_payloads_reject_malformed_edges(resolve, change) -> None:
    identity = {
        "chat_id": "chat-a",
        "access_request_id": "0" * 32,
        "expected_revision": 1,
    }
    raw = {**identity, **({"path": "/tmp/project", "expected_grants_revision": 0} if resolve else {})}
    raw.update(change)
    with pytest.raises(ValueError, match="invalid workspace access action payload"):
        parse_workspace_access_action_payload(raw, resolve=resolve)


def test_only_missing_managed_authority_uses_the_recoverable_policy_type(tmp_path: Path) -> None:
    home = tmp_path / "agent"
    instance = tmp_path / "instance"
    project = tmp_path / "not-granted"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    store.revoke(str(project), expected_revision=0)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
    )

    with pytest.raises(FolderAccessNeededError) as denied_read:
        policy.validate_path(project / "README.md", "read")
    assert denied_read.value.reason == "folder_not_granted"
    assert denied_read.value.operation == "read"

    store.upsert(
        {"path": str(project), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    with pytest.raises(FolderAccessNeededError) as denied_write:
        policy.validate_path(project / "README.md", "write")
    assert denied_write.value.reason == "write_not_granted"
    with pytest.raises(FolderAccessNeededError) as denied_exec:
        policy.validate_path(project, "exec")
    assert denied_exec.value.reason == "commands_not_granted"

    private = instance / "credentials" / "token.json"
    private.parent.mkdir(parents=True)
    private.write_text("secret", encoding="utf-8")
    with pytest.raises(PermissionError) as terminal:
        policy.validate_path(private, "read")
    assert type(terminal.value) is PermissionError


def test_recovery_target_kind_uses_tool_intent_without_probing_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "agent"
    instance = tmp_path / "instance"
    project = tmp_path / "not-granted"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    store.revoke(str(project), expected_revision=0)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
    )

    original_is_dir = Path.is_dir

    def forbidden_probe(candidate: Path) -> bool:
        if candidate == project:
            raise AssertionError("recovery classification must not probe the denied path")
        return original_is_dir(candidate)

    monkeypatch.setattr(Path, "is_dir", forbidden_probe)
    with pytest.raises(FolderAccessNeededError) as denied:
        policy.validate_path(project, "read", target_kind="directory")
    assert denied.value.target == {"kind": "directory", "path": str(project)}


def test_legacy_private_runtime_path_stays_terminal_without_grant_store(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "agent"
    instance = tmp_path / "instance"
    home.mkdir()
    credential = instance / "credentials" / "token.json"
    credential.parent.mkdir(parents=True)
    credential.write_text("secret", encoding="utf-8")
    monkeypatch.setattr(
        "nanobot.config.paths.get_config_path",
        lambda: instance / "config.json",
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
    )

    with pytest.raises(PermissionError) as denied:
        policy.validate_path(credential, "read", target_kind="file")

    assert type(denied.value) is PermissionError


def test_legacy_full_is_clamped_in_memory_by_configured_restriction(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "agent"
    home.mkdir()
    monkeypatch.setattr(workspace_api, "read_webui_default_access_mode", lambda: "full")

    default = workspace_api.default_scope_for_webui(home, True)
    payload = workspace_api.workspaces_payload(
        default_workspace=home,
        default_restrict_to_workspace=True,
        controls_available=True,
    )

    assert default.access_mode == "restricted"
    assert default.restrict_to_workspace is True
    assert payload["default_access_mode"] == "full"  # stored preference is reported unchanged
    assert payload["default_scope"]["effective_access"]["boundary"] == "folder"
    assert payload["controls"]["can_use_full_access"] is True
    assert payload["controls"]["full_access_unavailable_reason"] is None


@pytest.mark.asyncio
async def test_recovery_owner_reads_request_from_real_canonical_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    from nanobot.bus.queue import MessageBus
    from nanobot.security.folder_grants import FolderGrantStore
    from nanobot.session.execution_owner import acquire_execution
    from nanobot.session.manager import Session, SessionManager, SessionWriteConflict
    from nanobot.session.turn_writer import SourceTurnWriter
    from nanobot.session.workspace_access_continuation import (
        WorkspaceAccessContinuationReference,
        WorkspaceAccessContinuationRejectedError,
        workspace_access_continuation,
    )
    from nanobot.session.workspace_access_recovery import (
        WorkspaceAccessActionError,
        WorkspaceAccessRecoveryOwner,
        admit_workspace_access_continuation,
    )
    from nanobot.webui.workspaces import WebUIWorkspaceController

    home = tmp_path / "agent"
    instance = tmp_path / "instance"
    home.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    session = Session(
        "websocket:chat-a",
        metadata={"webui": True},
    )
    sessions.save(session)
    snapshot = sessions._jsonl_store.source.upgrade_source_session(
        session.key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    sessions.invalidate(session.key)
    blocked_run_id = "c" * 32
    async with acquire_execution(
        sessions._jsonl_store,
        session.key,
        run_id=blocked_run_id,
    ) as task_owner:
        task_writer = SourceTurnWriter(
            sessions._jsonl_store.source,
            task_owner,
            run_id=blocked_run_id,
        )
        task_input = task_writer.append_message({
            "role": "user",
            "content": "Read the project README",
        })
    input_event_id = task_input["_event_id"]
    scope = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=FolderGrantStore(
            agent_home=home,
            instance_root=instance,
            path=instance / "webui" / "folder-grants.json",
        ),
    ).scope_for_session_key(session.key)
    binding = {
        "store_identity": snapshot.state.store_identity,
        "incarnation": snapshot.state.incarnation,
        "generation": snapshot.state.generation,
    }
    record = {
        "version": 1,
        "access_request_id": "a" * 32,
        "revision": 1,
        "task": {"turn_id": "turn-original"},
        "blocked_turn_id": "turn-original",
        "tool_call_id": None,
        "created_at_ms": 1,
        "status": "pending",
        "reason": "folder_selection_required",
        "operation": "locate",
        "target": {"kind": "unknown", "path": None},
        "purpose": "Locate the requested project",
        "selected_path": None,
        "continuation_turn_id": None,
        "source_binding": binding,
        "task_input_event_id": input_event_id,
        "latest_input_event_id": input_event_id,
        "blocked_run_id": blocked_run_id,
        "expected_scope": {
            "project_path": str(scope.project_path),
            "access_mode": scope.access_mode,
        },
        "resolved_grants_revision": None,
    }
    sessions.commit_workspace_access_request(
        session.key,
        expected_binding=binding,
        expected_record=None,
        replacement_record=record,
        expected_latest_input_event_id=input_event_id,
    )

    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    workspaces = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=grants,
    )
    bus = MessageBus()
    owner = WorkspaceAccessRecoveryOwner(sessions, workspaces, bus)
    from nanobot.session.webui_turns import websocket_turn_wall_started_at

    public_request = owner.get("chat-a")

    assert public_request is not None
    assert public_request["status"] == "pending"
    assert public_request["task"]["excerpt"] == "Read the project README"
    assert public_request["target"] == {"kind": "unknown", "path": None}

    identity = {
        "chat_id": "chat-a",
        "access_request_id": public_request["access_request_id"],
        "expected_revision": public_request["revision"],
    }
    ready = owner.resolve({
        **identity,
        "path": str(home),
        "expected_grants_revision": grants.snapshot().revision,
    })
    assert ready["workspace_access_request"]["status"] == "ready"
    unrelated = tmp_path / "additional-readable-folder"
    unrelated.mkdir()
    grants.upsert(
        {"path": str(unrelated), "file_access": "read_only", "commands": False},
        expected_revision=grants.snapshot().revision,
    )
    stale_ready = owner.get("chat-a")
    assert stale_ready is not None
    assert stale_ready["status"] == "ready"
    assert stale_ready["can_continue"] is False
    assert stale_ready["unavailable_reason"] == "access_changed"
    reviewed = owner.resolve({
        **identity,
        "expected_revision": stale_ready["revision"],
        "path": str(home),
        "expected_grants_revision": grants.snapshot().revision,
    })
    assert reviewed["workspace_access_request"]["status"] == "ready"
    assert reviewed["workspace_access_request"]["revision"] == stale_ready["revision"] + 1
    assert reviewed["workspace_access_request"]["selected_path"] == str(home)

    commit = sessions.commit_workspace_access_request

    def reject_cas(*args, **kwargs) -> None:
        _ = args, kwargs
        raise SessionWriteConflict()

    monkeypatch.setattr(sessions, "commit_workspace_access_request", reject_cas)
    with pytest.raises(WorkspaceAccessActionError) as cas_failure:
        await owner.continue_request(identity | {
            "expected_revision": reviewed["workspace_access_request"]["revision"],
        })
    assert cas_failure.value.status == 409
    assert cas_failure.value.message == "access_request_changed"
    assert websocket_turn_wall_started_at("chat-a") is None
    assert bus.inbound.empty()
    monkeypatch.setattr(sessions, "commit_workspace_access_request", commit)

    publish_inbound = bus.publish_inbound

    async def enqueue_then_raise(message) -> None:
        await publish_inbound(message)
        raise RuntimeError("publisher result is ambiguous after enqueue")

    monkeypatch.setattr(bus, "publish_inbound", enqueue_then_raise)
    with pytest.raises(WorkspaceAccessActionError) as ambiguous_publish:
        await owner.continue_request(identity | {
            "expected_revision": reviewed["workspace_access_request"]["revision"],
        })
    assert ambiguous_publish.value.status == 503
    assert ambiguous_publish.value.message == "continuation_unknown"
    assert websocket_turn_wall_started_at("chat-a") is not None
    monkeypatch.setattr(bus, "publish_inbound", publish_inbound)

    assert bus.inbound.qsize() == 1
    continued = owner.get("chat-a")
    assert continued is not None
    assert continued["status"] == "continuing"
    assert continued["unavailable_reason"] == "continuation_unknown"
    inbound = bus.inbound.get_nowait()
    assert inbound.channel == "websocket"
    assert inbound.chat_id == "chat-a"
    assert inbound.session_key == session.key
    assert inbound.content == "Continue task"
    assert inbound.metadata["webui_turn_id"] == continued["continuation_turn_id"]
    from nanobot.session.webui_turns import (
        clear_websocket_turn_if_current,
        websocket_turn_owner_is_registered,
    )
    from nanobot.webui.metadata import WEBSOCKET_TURN_OWNER_METADATA_KEY

    lifecycle_owner = inbound.metadata.get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
    assert isinstance(lifecycle_owner, str)
    assert websocket_turn_owner_is_registered(
        "chat-a",
        lifecycle_owner,
        continued["continuation_turn_id"],
    )

    def release_test_owner() -> None:
        clear_websocket_turn_if_current("chat-a", lifecycle_owner)

    request.addfinalizer(release_test_owner)
    assert "_workspace_access_continuation" not in inbound.metadata
    continuation_reference = workspace_access_continuation(inbound)
    assert type(continuation_reference) is WorkspaceAccessContinuationReference
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(continuation_reference)
    forged = replace(
        inbound,
        metadata={
            **inbound.metadata,
            "_nanobot_workspace_access_continuation": {
                "access_request_id": public_request["access_request_id"],
                "revision": continued["revision"],
            },
        },
    )
    assert workspace_access_continuation(forged) is None
    with pytest.raises(WorkspaceAccessActionError) as replay:
        await owner.continue_request(identity | {
            "expected_revision": reviewed["workspace_access_request"]["revision"],
        })
    assert replay.value.status == 409

    from nanobot.agent.turn import (
        DeliveryTarget,
        HistoryMode,
        TurnRequest,
        TurnSource,
    )
    from nanobot.agent.turn import TurnContext as RunTurnContext

    backend = sessions._jsonl_store
    async with acquire_execution(backend, session.key, run_id="9" * 32) as execution_owner:
        writer = SourceTurnWriter(backend.source, execution_owner, run_id="9" * 32)
        turn_context = RunTurnContext(
            request=TurnRequest(
                content=inbound.content,
                source=TurnSource.GATEWAY,
                history_mode=HistoryMode.SESSION,
                session_key=session.key,
                route=DeliveryTarget(channel="websocket", chat_id="chat-a"),
                metadata=dict(inbound.metadata),
            ),
            run_id="9" * 32,
            session_key=session.key,
            source_writer=writer,
            execution_owner=execution_owner,
        )
        assert admit_workspace_access_continuation(
            sessions,
            turn_context,
            workspaces.focus_owner,
            continuation_reference,
        )
        current = backend.source.load_source_snapshot(session.key)
        assert current.header["metadata"]["workspace_access_request"]["status"] == "continued"
        with pytest.raises(WorkspaceAccessContinuationRejectedError):
            admit_workspace_access_continuation(
                sessions,
                turn_context,
                workspaces.focus_owner,
                continuation_reference,
            )


def test_focus_cycle_and_new_input_permanently_invalidate_recovery_record(
    tmp_path: Path,
) -> None:
    from nanobot.session.manager import Session, SessionManager

    home = tmp_path / "agent"
    alternate = tmp_path / "alternate"
    home.mkdir()
    alternate.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:focus-cycle"
    scope_a = {"project_path": str(home), "access_mode": "restricted"}
    session = Session(key, messages=[{"role": "user", "content": "Read the report"}],
                      metadata={"webui": True, "workspace_scope": scope_a})
    sessions.save(session)
    source = sessions._jsonl_store.source
    snapshot = source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    sessions.invalidate(key)
    input_event_id = snapshot.events[-1]["_event_id"]
    binding = {
        "store_identity": snapshot.state.store_identity,
        "incarnation": snapshot.state.incarnation,
        "generation": snapshot.state.generation,
    }
    record = {
        "version": 1,
        "access_request_id": "e" * 32,
        "revision": 1,
        "task": {"turn_id": "turn-focus"},
        "blocked_turn_id": "turn-focus",
        "tool_call_id": None,
        "created_at_ms": 1,
        "status": "ready",
        "reason": "folder_selection_required",
        "operation": "locate",
        "target": {"kind": "unknown", "path": None},
        "purpose": "Locate the requested project",
        "selected_path": str(home),
        "continuation_turn_id": None,
        "source_binding": binding,
        "task_input_event_id": input_event_id,
        "latest_input_event_id": input_event_id,
        "blocked_run_id": "f" * 32,
        "expected_scope": scope_a,
        "resolved_grants_revision": 0,
    }
    sessions.commit_workspace_access_request(
        key,
        expected_binding=binding,
        expected_record=None,
        replacement_record=record,
        expected_latest_input_event_id=input_event_id,
    )

    for path in (alternate, home):
        sessions.upsert_session_metadata(
            key,
            {"workspace_scope": {"project_path": str(path), "access_mode": "restricted"}},
        )

    latest = source.load_source_snapshot(key)
    assert latest.header["metadata"]["workspace_access_request"]["status"] == "invalidated"
    assert latest.header["metadata"]["workspace_access_request"]["revision"] == 2

    old_record = latest.header["metadata"]["workspace_access_request"]
    replacement = {
        **record,
        "access_request_id": "a" * 32,
        "revision": 1,
        "status": "pending",
    }
    sessions.commit_workspace_access_request(
        key,
        expected_binding=binding,
        expected_record=old_record,
        replacement_record=replacement,
        expected_latest_input_event_id=input_event_id,
    )
    latest = source.load_source_snapshot(key)
    from nanobot.session.source import new_source_event

    accepted_input = new_source_event(
        {"role": "user", "content": "Now read the latest report"},
        run_id="1" * 32,
    )
    after_input = source.append_source_events(key, latest.state, [accepted_input])
    assert after_input.header["metadata"]["workspace_access_request"]["status"] == "invalidated"


@pytest.mark.asyncio
async def test_request_persistence_commits_from_real_owned_gateway_turn(tmp_path: Path) -> None:
    from nanobot.agent.loop import AgentRunResult, TurnKind
    from nanobot.agent.loop import TurnContext as AgentTurnContext
    from nanobot.agent.tools.context import RequestContext
    from nanobot.agent.tools.workspace import FOLDER_ACCESS_SAFE_FINAL
    from nanobot.agent.turn import (
        HistoryMode,
        RunPolicyContext,
        TurnRequest,
        TurnSource,
    )
    from nanobot.agent.turn import TurnContext as RunTurnContext
    from nanobot.agent.turn_delivery import TurnDeliveryFactory
    from nanobot.bus.events import InboundMessage
    from nanobot.bus.queue import MessageBus
    from nanobot.security.workspace_access import build_workspace_scope
    from nanobot.session.execution_owner import acquire_execution
    from nanobot.session.manager import Session, SessionManager
    from nanobot.session.turn_writer import SourceTurnWriter
    from nanobot.session.workspace_access_continuation import workspace_access_continuation
    from nanobot.session.workspace_access_recovery import (
        WorkspaceAccessRecoveryOwner,
        admit_workspace_access_continuation,
        persist_folder_access_request,
    )
    from nanobot.webui.metadata import WEBUI_TURN_METADATA_KEY

    home = tmp_path / "agent"
    home.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    session = Session("websocket:chat-persist", metadata={"webui": True})
    sessions.save(session)
    backend = sessions._jsonl_store
    backend.source.upgrade_source_session(
        session.key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    sessions.invalidate(session.key)
    run_id = "d" * 32
    message = InboundMessage(
        channel="websocket",
        sender_id="webui",
        chat_id="chat-persist",
        content="Read the README in the project I named.",
        metadata={"webui": True, WEBUI_TURN_METADATA_KEY: "turn-current"},
    )
    scope = build_workspace_scope(home, "restricted", source_channel="websocket")
    request_context = RequestContext(
        channel="websocket",
        chat_id="chat-persist",
        session_key=session.key,
        workspace=home,
        workspace_scope=scope,
        persistent_session=True,
    )
    need = {
        "version": 1,
        "reason": "folder_selection_required",
        "operation": "locate",
        "target": {"kind": "unknown", "path": None},
        "purpose": "Locate the named project",
        "allowed_roots": [str(home)],
        "roots_truncated": False,
    }
    async with acquire_execution(backend, session.key, run_id=run_id) as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id=run_id)
        input_event = writer.append_message({"role": "user", "content": message.content})
        run_context = RunTurnContext(
            request=TurnRequest(
                content=message.content,
                source=TurnSource.GATEWAY,
                history_mode=HistoryMode.SESSION,
                session_key=session.key,
                run_id=run_id,
            ),
            run_id=run_id,
            session_key=session.key,
            policy=RunPolicyContext(
                workspace=home,
                context_manager=ContextRepoManager(),
                source=TurnSource.GATEWAY,
            ),
            execution_owner=owner,
            source_writer=writer,
            source_input_event_id=input_event["_event_id"],
        )
        context = AgentTurnContext(
            msg=message,
            session_key=session.key,
            turn_id="turn-current",
            runtime=None,
            kind=TurnKind.USER,
            delivery=TurnDeliveryFactory(MessageBus()).create(message, session.key),
        )
        context.run_result = AgentRunResult(
            final_content=FOLDER_ACCESS_SAFE_FINAL,
            messages=[],
            stop_reason="folder_access_required",
            policy_metadata={"folder_access": need},
        )
        context.run_context = run_context
        context.request_context = request_context
        public_request = persist_folder_access_request(sessions, context)

    assert public_request is not None
    assert public_request["status"] == "pending"
    assert public_request["task"]["turn_id"] == "turn-current"
    assert public_request["task"]["excerpt"] == message.content
    workspaces = workspace_api.WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=FolderGrantStore(
            agent_home=home,
            instance_root=tmp_path / "instance",
            path=tmp_path / "instance" / "webui" / "folder-grants.json",
        ),
    )
    loaded = WorkspaceAccessRecoveryOwner(sessions, workspaces).get("chat-persist")
    assert loaded == public_request

    grants = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    workspaces = workspace_api.WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=grants,
    )
    bus = MessageBus()
    recovery = WorkspaceAccessRecoveryOwner(sessions, workspaces, bus)
    identity = {
        "chat_id": "chat-persist",
        "access_request_id": public_request["access_request_id"],
        "expected_revision": public_request["revision"],
    }
    ready = recovery.resolve({
        **identity,
        "path": str(home),
        "expected_grants_revision": grants.snapshot().revision,
    })
    continued = await recovery.continue_request({
        **identity,
        "expected_revision": ready["workspace_access_request"]["revision"],
    })
    inbound = bus.inbound.get_nowait()
    continuation_reference = workspace_access_continuation(inbound)
    assert continuation_reference is not None
    assert continued["workspace_access_request"]["status"] == "continuing"
    continuation_turn_id = continued["workspace_access_request"]["continuation_turn_id"]
    continuation_run_id = "e" * 32
    async with acquire_execution(
        backend,
        session.key,
        run_id=continuation_run_id,
    ) as continuation_owner:
        continuation_writer = SourceTurnWriter(
            backend.source,
            continuation_owner,
            run_id=continuation_run_id,
        )
        from nanobot.agent.turn import DeliveryTarget

        continuation_context = RunTurnContext(
            request=TurnRequest(
                content=inbound.content,
                source=TurnSource.GATEWAY,
                history_mode=HistoryMode.SESSION,
                session_key=session.key,
                run_id=continuation_run_id,
                route=DeliveryTarget(channel="websocket", chat_id="chat-persist"),
                metadata=dict(inbound.metadata),
            ),
            run_id=continuation_run_id,
            session_key=session.key,
            policy=RunPolicyContext(
                workspace=home,
                context_manager=ContextRepoManager(),
                source=TurnSource.GATEWAY,
            ),
            execution_owner=continuation_owner,
            source_writer=continuation_writer,
        )
        assert admit_workspace_access_continuation(
            sessions,
            continuation_context,
            workspaces.focus_owner,
            continuation_reference,
        )
        continuation_event = continuation_writer.append_message({
            "role": "user",
            "content": inbound.content,
        })
        continuation_context.source_input_event_id = continuation_event["_event_id"]
        continued_message = InboundMessage(
            channel="websocket",
            sender_id="webui",
            chat_id="chat-persist",
            content=inbound.content,
            metadata=dict(inbound.metadata),
        )
        continued_request_context = RequestContext(
            channel="websocket",
            chat_id="chat-persist",
            session_key=session.key,
            workspace=home,
            workspace_scope=workspaces.scope_for_session_key(session.key),
            persistent_session=True,
        )
        continued_agent_context = AgentTurnContext(
            msg=continued_message,
            session_key=session.key,
            turn_id=continuation_turn_id,
            runtime=None,
            kind=TurnKind.USER,
            delivery=TurnDeliveryFactory(MessageBus()).create(
                continued_message,
                session.key,
            ),
        )
        continued_agent_context.run_context = continuation_context
        continued_agent_context.request_context = continued_request_context
        continued_agent_context.run_result = AgentRunResult(
            final_content=FOLDER_ACCESS_SAFE_FINAL,
            messages=[],
            stop_reason="folder_access_required",
            policy_metadata={"folder_access": need},
        )
        repeated_request = persist_folder_access_request(sessions, continued_agent_context)

    assert repeated_request is not None
    assert repeated_request["status"] == "pending"
    assert repeated_request["task"] == public_request["task"]
    assert repeated_request["blocked_turn_id"] == continuation_turn_id
    assert repeated_request["access_request_id"] != public_request["access_request_id"]
    from nanobot.session.webui_turns import clear_websocket_turn_if_current
    from nanobot.webui.metadata import WEBSOCKET_TURN_OWNER_METADATA_KEY

    assert clear_websocket_turn_if_current(
        "chat-persist",
        inbound.metadata[WEBSOCKET_TURN_OWNER_METADATA_KEY],
    )

    selected = tmp_path / "selected-once"
    selected.mkdir()
    permission_owner = sessions.desktop_permissions
    assert permission_owner.revision == 0
    once_result = WorkspaceAccessRecoveryOwner(sessions, workspaces).allow_once({
        "chat_id": "chat-persist",
        "access_request_id": repeated_request["access_request_id"],
        "expected_revision": repeated_request["revision"],
        "path": str(selected),
        "file_access": "read_only",
        "commands": False,
        "expected_grants_revision": workspaces.grant_snapshot().revision,
        "expected_runtime_id": permission_owner.runtime_id,
        "expected_permission_revision": permission_owner.revision,
    })
    assert once_result["once_receipt"] == {
        "version": 1,
        "duration": "once",
        "runtime_id": permission_owner.runtime_id,
        "permission_revision": 1,
        "access_request_id": repeated_request["access_request_id"],
        "request_revision": repeated_request["revision"] + 1,
        "task_turn_id": repeated_request["task"]["turn_id"],
        "submitted_path": str(selected),
        "canonical_path": str(selected),
        "file_access": "read_only",
        "commands": False,
    }
    assert once_result["workspace_access_request"]["selected_duration"] == "once"
    assert once_result["workspace_access_request"]["task_access_state"] == "active"
    assert once_result["workspace_access_request"]["can_continue"] is True
    assert once_result["workspace_permissions"]["task_access"] == {
        "task_turn_id": repeated_request["task"]["turn_id"],
        "state": "active",
        "folders": [{
            "path": str(selected),
            "file_access": "read_only",
            "commands": False,
        }],
    }
