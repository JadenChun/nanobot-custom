"""Real canonical turns request user approval before saving a folder or grant."""
import json
from pathlib import Path
from uuid import uuid4

import pytest

from nanobot.agent.turn import DeliveryTarget, TurnRequest, TurnSource
from nanobot.context_repo.catalog import RepositoryEntry
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.runtime_composition import RuntimeComposition
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.folder_locations import FolderLocationEdit, apply_folder_edit
from nanobot.session.workspace_access_recovery import (
    WorkspaceAccessActionError,
    WorkspaceAccessRecoveryOwner,
)
from nanobot.webui.workspaces import WebUIWorkspaceController
from tests.migration.test_workspace_access_continue_ingress import _config, _RecordingProvider


class _ProposeFolder(_RecordingProvider):
    def __init__(self, path):
        super().__init__()
        self.path = path

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.requests.append(messages)
        if len(self.requests) == 1:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="add-folder", name="repository_catalog", arguments={
                    "action": "remember", "name": "plain-documents", "path": str(self.path),
                    "purpose": "Project reference documents", "file_access": "read_write",
                },
            )])
        return LLMResponse(content="Continue with the approved folder.")


@pytest.mark.asyncio
async def test_folder_proposal_fresh_confirmation_recovers_after_unrelated_grant_change(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    project = tmp_path / "plain-folder"
    unrelated = tmp_path / "unrelated-folder"
    project.mkdir()
    unrelated.mkdir()
    provider = _ProposeFolder(project)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    composition = RuntimeComposition.define(config, sessions_root=tmp_path / "sessions", startup_receipt_admitted=True)
    handle = await composition.acquire()
    try:
        grants = FolderGrantStore.for_workspace(config.workspace_path, sessions_root=handle.sessions.sessions_root)
        workspaces = WebUIWorkspaceController(session_manager=handle.sessions, default_workspace=config.workspace_path, default_restrict_to_workspace=True, grant_store=grants)
        workspaces.persist_scope("folder-proposal", workspaces.scope_for_session_key("websocket:folder-proposal"))
        recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
        result = await handle.agent.execute_turn(TurnRequest(
            content="Add my reference documents to the folder map.", session_key="websocket:folder-proposal",
            source=TurnSource.GATEWAY, route=DeliveryTarget(channel="websocket", chat_id="folder-proposal"),
            metadata={"webui": True, "webui_turn_id": "proposal-turn"},
        ))
        assert result.stop_reason == "folder_access_required"
        card = recovery.get("folder-proposal")
        approval = {
            "chat_id": "folder-proposal",
            "access_request_id": card["access_request_id"],
            "expected_revision": card["revision"],
            "expected_grants_revision": 0,
            "expected_catalog_revision": 0,
            "file_access": "read_only",
            "agent_map": True,
        }
        original = grants.upsert
        def fail(*args, **kwargs):
            raise OSError("simulated grant disk failure")
        monkeypatch.setattr(grants, "upsert", fail)
        with pytest.raises(WorkspaceAccessActionError):
            recovery.add_folder(approval)
        assert workspaces.catalog.get("plain-documents")
        assert grants.effective_grant_for(project) is None
        assert grants.snapshot().revision == 0
        monkeypatch.setattr(grants, "upsert", original)

        grants.upsert(
            {"path": str(unrelated), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
        assert grants.snapshot().revision == 1

        stale = {
            **approval,
            "expected_catalog_revision": workspaces.catalog.snapshot().revision,
        }
        with pytest.raises(WorkspaceAccessActionError, match="grants_changed"):
            recovery.add_folder(stale)
        assert grants.effective_grant_for(project) is None

        fresh = {**stale, "expected_grants_revision": grants.snapshot().revision}
        recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
        ready = recovery.add_folder(fresh)
        assert ready["workspace_access_request"]["status"] == "ready"
        assert ready["workspace_access_request"]["can_continue"]
        assert grants.effective_grant_for(project).file_access == "read_only"
        assert grants.snapshot().revision == 2
        assert workspaces.catalog.get("plain-documents").agent_map
        journal = workspaces.catalog.root / f"folder-edit-{card['access_request_id']}.json"
        journal_record = json.loads(journal.read_text())
        assert journal_record["catalog_done"] is True
        assert journal_record["grants_done"] is True

        with pytest.raises(WorkspaceAccessActionError):
            recovery.add_folder(fresh)
        assert grants.snapshot().revision == 2
    finally:
        await composition.aclose()


@pytest.mark.asyncio
async def test_folder_proposal_rejects_stale_catalog_revision_during_renewal(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    project = tmp_path / "plain-folder"
    unrelated = tmp_path / "unrelated-folder"
    other = tmp_path / "other-folder"
    for directory in (project, unrelated, other):
        directory.mkdir()
    provider = _ProposeFolder(project)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    composition = RuntimeComposition.define(config, sessions_root=tmp_path / "sessions", startup_receipt_admitted=True)
    handle = await composition.acquire()
    try:
        grants = FolderGrantStore.for_workspace(config.workspace_path, sessions_root=handle.sessions.sessions_root)
        workspaces = WebUIWorkspaceController(session_manager=handle.sessions, default_workspace=config.workspace_path, default_restrict_to_workspace=True, grant_store=grants)
        workspaces.persist_scope("folder-proposal", workspaces.scope_for_session_key("websocket:folder-proposal"))
        recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
        await handle.agent.execute_turn(TurnRequest(
            content="Add my reference documents to the folder map.", session_key="websocket:folder-proposal",
            source=TurnSource.GATEWAY, route=DeliveryTarget(channel="websocket", chat_id="folder-proposal"),
            metadata={"webui": True, "webui_turn_id": "proposal-turn"},
        ))
        card = recovery.get("folder-proposal")
        approval = {
            "chat_id": "folder-proposal",
            "access_request_id": card["access_request_id"],
            "expected_revision": card["revision"],
            "expected_grants_revision": 0,
            "expected_catalog_revision": 0,
            "file_access": "read_only",
            "agent_map": True,
        }
        original = grants.upsert
        def fail(*args, **kwargs):
            raise OSError("simulated grant disk failure")
        monkeypatch.setattr(grants, "upsert", fail)
        with pytest.raises(WorkspaceAccessActionError):
            recovery.add_folder(approval)
        monkeypatch.setattr(grants, "upsert", original)

        grants.upsert(
            {"path": str(unrelated), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
        workspaces.catalog.update(RepositoryEntry(name="other", path=str(other)))
        assert grants.snapshot().revision == 1
        assert workspaces.catalog.snapshot().revision == 2

        stale_catalog = {
            **approval,
            "expected_grants_revision": 1,
            "expected_catalog_revision": 1,
        }
        with pytest.raises(WorkspaceAccessActionError, match="catalog_changed"):
            recovery.add_folder(stale_catalog)
        assert grants.snapshot().revision == 1
        assert grants.effective_grant_for(project) is None
        journal = workspaces.catalog.root / f"folder-edit-{card['access_request_id']}.json"
        assert json.loads(journal.read_text()).get("renewed_expected_catalog_revision") is None

        fresh = {**stale_catalog, "expected_catalog_revision": 2}
        ready = recovery.add_folder(fresh)
        assert ready["workspace_access_request"]["status"] == "ready"
        assert grants.effective_grant_for(project).file_access == "read_only"
        journal_record = json.loads(journal.read_text())
        assert journal_record["edit"]["expected_grants_revision"] == 0
        assert journal_record["edit"]["expected_catalog_revision"] == 0
        assert journal_record["renewed_expected_grants_revision"] == 1
        assert journal_record["renewed_expected_catalog_revision"] == 2
    finally:
        await composition.aclose()


@pytest.mark.asyncio
async def test_folder_proposal_rejects_stale_original_and_renewed_catalog_approvals(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    project = tmp_path / "plain-folder"
    unrelated = tmp_path / "unrelated-folder"
    second_unrelated = tmp_path / "second-unrelated-folder"
    for directory in (project, unrelated, second_unrelated):
        directory.mkdir()
    provider = _ProposeFolder(project)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    composition = RuntimeComposition.define(config, sessions_root=tmp_path / "sessions", startup_receipt_admitted=True)
    handle = await composition.acquire()
    try:
        grants = FolderGrantStore.for_workspace(config.workspace_path, sessions_root=handle.sessions.sessions_root)
        workspaces = WebUIWorkspaceController(session_manager=handle.sessions, default_workspace=config.workspace_path, default_restrict_to_workspace=True, grant_store=grants)
        workspaces.persist_scope("folder-proposal", workspaces.scope_for_session_key("websocket:folder-proposal"))
        recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
        await handle.agent.execute_turn(TurnRequest(
            content="Add my reference documents to the folder map.", session_key="websocket:folder-proposal",
            source=TurnSource.GATEWAY, route=DeliveryTarget(channel="websocket", chat_id="folder-proposal"),
            metadata={"webui": True, "webui_turn_id": "proposal-turn"},
        ))
        card = recovery.get("folder-proposal")
        original_approval = {
            "chat_id": "folder-proposal",
            "access_request_id": card["access_request_id"],
            "expected_revision": card["revision"],
            "expected_grants_revision": 0,
            "expected_catalog_revision": 0,
            "file_access": "read_only",
            "agent_map": True,
        }
        original_upsert = grants.upsert

        def fail(*args, **kwargs):
            raise OSError("simulated grant disk failure")

        monkeypatch.setattr(grants, "upsert", fail)
        with pytest.raises(WorkspaceAccessActionError):
            recovery.add_folder(original_approval)
        monkeypatch.setattr(grants, "upsert", original_upsert)
        journal_path = workspaces.catalog.root / f"folder-edit-{card['access_request_id']}.json"
        legacy_journal = json.loads(journal_path.read_text())
        legacy_journal.pop("catalog_receipt_revision", None)
        journal_path.write_text(json.dumps(legacy_journal))
        workspaces.catalog.update(RepositoryEntry(name="unrelated", path=str(unrelated)))
        assert workspaces.catalog.snapshot().revision == 2

        with pytest.raises(WorkspaceAccessActionError, match="catalog_changed"):
            recovery.add_folder(original_approval)
        assert grants.snapshot().revision == 0
        assert grants.effective_grant_for(project) is None

        current_approval = {**original_approval, "expected_catalog_revision": 2}
        monkeypatch.setattr(grants, "upsert", fail)
        with pytest.raises(WorkspaceAccessActionError):
            recovery.add_folder(current_approval)
        monkeypatch.setattr(grants, "upsert", original_upsert)
        journal = json.loads(journal_path.read_text())
        assert journal["renewed_expected_catalog_revision"] == 2
        assert journal["catalog_receipt_revision"] is None
        assert grants.snapshot().revision == 0

        workspaces.catalog.update(RepositoryEntry(name="second-unrelated", path=str(second_unrelated)))
        assert workspaces.catalog.snapshot().revision == 3
        with pytest.raises(WorkspaceAccessActionError, match="catalog_changed"):
            recovery.add_folder(current_approval)
        assert grants.snapshot().revision == 0

        fresh = {**original_approval, "expected_catalog_revision": 3}
        ready = recovery.add_folder(fresh)
        assert ready["workspace_access_request"]["status"] == "ready"
        assert grants.effective_grant_for(project).file_access == "read_only"
        assert grants.snapshot().revision == 1
        journal = json.loads(journal_path.read_text())
        assert journal["edit"]["expected_catalog_revision"] == 0
        assert journal["renewed_expected_catalog_revision"] == 3
    finally:
        await composition.aclose()


@pytest.mark.asyncio
async def test_folder_proposal_conflicts_when_catalog_changes_during_grant_write(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    project = tmp_path / "plain-folder"
    unrelated = tmp_path / "unrelated-folder"
    other = tmp_path / "other-folder"
    for directory in (project, unrelated, other):
        directory.mkdir()
    provider = _ProposeFolder(project)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    composition = RuntimeComposition.define(config, sessions_root=tmp_path / "sessions", startup_receipt_admitted=True)
    handle = await composition.acquire()
    try:
        grants = FolderGrantStore.for_workspace(config.workspace_path, sessions_root=handle.sessions.sessions_root)
        workspaces = WebUIWorkspaceController(session_manager=handle.sessions, default_workspace=config.workspace_path, default_restrict_to_workspace=True, grant_store=grants)
        workspaces.persist_scope("folder-proposal", workspaces.scope_for_session_key("websocket:folder-proposal"))
        recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
        await handle.agent.execute_turn(TurnRequest(
            content="Add my reference documents to the folder map.", session_key="websocket:folder-proposal",
            source=TurnSource.GATEWAY, route=DeliveryTarget(channel="websocket", chat_id="folder-proposal"),
            metadata={"webui": True, "webui_turn_id": "proposal-turn"},
        ))
        card = recovery.get("folder-proposal")
        approval = {
            "chat_id": "folder-proposal",
            "access_request_id": card["access_request_id"],
            "expected_revision": card["revision"],
            "expected_grants_revision": 0,
            "expected_catalog_revision": 0,
            "file_access": "read_only",
            "agent_map": True,
        }
        original = grants.upsert
        def fail(*args, **kwargs):
            raise OSError("simulated grant disk failure")
        monkeypatch.setattr(grants, "upsert", fail)
        with pytest.raises(WorkspaceAccessActionError):
            recovery.add_folder(approval)
        monkeypatch.setattr(grants, "upsert", original)
        grants.upsert(
            {"path": str(unrelated), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
        fresh = {
            **approval,
            "expected_grants_revision": grants.snapshot().revision,
            "expected_catalog_revision": workspaces.catalog.snapshot().revision,
        }

        original_upsert = grants.upsert
        def concurrent_upsert(*args, **kwargs):
            current = workspaces.catalog.get("plain-documents")
            workspaces.catalog.update(current.model_copy(update={"purpose": "Concurrent editor"}))
            return original_upsert(*args, **kwargs)
        monkeypatch.setattr(grants, "upsert", concurrent_upsert)
        with pytest.raises(WorkspaceAccessActionError, match="catalog_changed"):
            recovery.add_folder(fresh)

        assert workspaces.catalog.get("plain-documents").purpose == "Concurrent editor"
        assert grants.effective_grant_for(project).file_access == "read_only"
        assert grants.snapshot().revision == 2
        journal = workspaces.catalog.root / f"folder-edit-{card['access_request_id']}.json"
        journal_record = json.loads(journal.read_text())
        assert journal_record["catalog_done"] is True
        assert journal_record["grants_done"] is True
        assert recovery.get("folder-proposal")["status"] == "pending"

        retry = {
            **fresh,
            "expected_grants_revision": grants.snapshot().revision,
            "expected_catalog_revision": workspaces.catalog.snapshot().revision,
        }
        with pytest.raises(WorkspaceAccessActionError, match="catalog_changed"):
            recovery.add_folder(retry)
        assert workspaces.catalog.get("plain-documents").purpose == "Concurrent editor"
        assert grants.snapshot().revision == 2

        for number in range(129):
            unrelated_edit = FolderLocationEdit(
                operation_id=uuid4().hex,
                expected_grants_revision=grants.snapshot().revision,
                expected_catalog_revision=workspaces.catalog.snapshot().revision,
                path=str(other),
                entry=RepositoryEntry(name="other", path=str(other), purpose=f"Unrelated edit {number}"),
                access="keep",
            )
            apply_folder_edit(grants, workspaces.catalog, unrelated_edit)
        assert journal.exists()
        replay = {
            **fresh,
            "expected_grants_revision": grants.snapshot().revision,
            "expected_catalog_revision": workspaces.catalog.snapshot().revision,
        }
        with pytest.raises(WorkspaceAccessActionError, match="catalog_changed"):
            recovery.add_folder(replay)
        assert workspaces.catalog.get("plain-documents").purpose == "Concurrent editor"
    finally:
        await composition.aclose()


@pytest.mark.asyncio
async def test_folder_proposal_changed_metadata_blocks_renewed_confirmation(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    project = tmp_path / "plain-folder"
    unrelated = tmp_path / "unrelated-folder"
    project.mkdir()
    unrelated.mkdir()
    provider = _ProposeFolder(project)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    composition = RuntimeComposition.define(config, sessions_root=tmp_path / "sessions", startup_receipt_admitted=True)
    handle = await composition.acquire()
    try:
        grants = FolderGrantStore.for_workspace(config.workspace_path, sessions_root=handle.sessions.sessions_root)
        workspaces = WebUIWorkspaceController(session_manager=handle.sessions, default_workspace=config.workspace_path, default_restrict_to_workspace=True, grant_store=grants)
        workspaces.persist_scope("folder-proposal", workspaces.scope_for_session_key("websocket:folder-proposal"))
        recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
        await handle.agent.execute_turn(TurnRequest(
            content="Add my reference documents to the folder map.", session_key="websocket:folder-proposal",
            source=TurnSource.GATEWAY, route=DeliveryTarget(channel="websocket", chat_id="folder-proposal"),
            metadata={"webui": True, "webui_turn_id": "proposal-turn"},
        ))
        card = recovery.get("folder-proposal")
        approval = {
            "chat_id": "folder-proposal",
            "access_request_id": card["access_request_id"],
            "expected_revision": card["revision"],
            "expected_grants_revision": 0,
            "expected_catalog_revision": 0,
            "file_access": "read_only",
            "agent_map": True,
        }
        original = grants.upsert
        def fail(*args, **kwargs):
            raise OSError("simulated grant disk failure")
        monkeypatch.setattr(grants, "upsert", fail)
        with pytest.raises(WorkspaceAccessActionError):
            recovery.add_folder(approval)
        monkeypatch.setattr(grants, "upsert", original)
        grants.upsert(
            {"path": str(unrelated), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
        entry = workspaces.catalog.get("plain-documents")
        workspaces.catalog.update(entry.model_copy(update={"purpose": "Changed after approval"}))
        fresh = {
            **approval,
            "expected_grants_revision": grants.snapshot().revision,
            "expected_catalog_revision": workspaces.catalog.snapshot().revision,
        }
        with pytest.raises(WorkspaceAccessActionError, match="catalog_changed"):
            recovery.add_folder(fresh)
        assert grants.effective_grant_for(project) is None
    finally:
        await composition.aclose()


@pytest.mark.parametrize("action", ["approve", "decline", "stale", "partial"])
@pytest.mark.asyncio
async def test_folder_proposal_requires_current_user_approval(tmp_path: Path, monkeypatch, action):
    config = _config(tmp_path)
    project = tmp_path / "plain-folder"
    project.mkdir()
    provider = _ProposeFolder(project)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: provider)
    composition = RuntimeComposition.define(config, sessions_root=tmp_path / "sessions", startup_receipt_admitted=True)
    handle = await composition.acquire()
    try:
        grants = FolderGrantStore.for_workspace(config.workspace_path, sessions_root=handle.sessions.sessions_root)
        workspaces = WebUIWorkspaceController(session_manager=handle.sessions, default_workspace=config.workspace_path, default_restrict_to_workspace=True, grant_store=grants)
        workspaces.persist_scope("folder-proposal", workspaces.scope_for_session_key("websocket:folder-proposal"))
        recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
        result = await handle.agent.execute_turn(TurnRequest(
            content="Add my reference documents to the folder map.", session_key="websocket:folder-proposal",
            source=TurnSource.GATEWAY, route=DeliveryTarget(channel="websocket", chat_id="folder-proposal"),
            metadata={"webui": True, "webui_turn_id": "proposal-turn"},
        ))
        assert result.stop_reason == "folder_access_required"
        card = recovery.get("folder-proposal")
        assert card["folder_proposal"]["entry"]["path"] == str(project)
        assert not workspaces.catalog.entries()
        assert grants.effective_grant_for(project) is None
        identity = {"chat_id": "folder-proposal", "access_request_id": card["access_request_id"], "expected_revision": card["revision"]}
        if action == "decline":
            recovery.cancel(identity)
            assert not workspaces.catalog.entries()
            assert grants.effective_grant_for(project) is None
            return
        if action == "stale":
            await handle.agent.execute_turn(TurnRequest(
                content="Please start a different task.", session_key="websocket:folder-proposal", source=TurnSource.GATEWAY,
                route=DeliveryTarget(channel="websocket", chat_id="folder-proposal"),
                metadata={"webui": True, "webui_turn_id": "new-task"},
            ))
        approval = {**identity, "expected_grants_revision": 0, "expected_catalog_revision": 0, "file_access": "read_only", "agent_map": True}
        if action == "stale":
            with pytest.raises(WorkspaceAccessActionError):
                recovery.add_folder(approval)
            assert not workspaces.catalog.entries()
            assert grants.effective_grant_for(project) is None
            return
        if action == "partial":
            original = grants.upsert
            def fail(*args, **kwargs):
                raise OSError("simulated grant disk failure")
            monkeypatch.setattr(grants, "upsert", fail)
            with pytest.raises(WorkspaceAccessActionError):
                recovery.add_folder(approval)
            assert workspaces.catalog.get("plain-documents")
            assert grants.effective_grant_for(project) is None
            assert recovery.get("folder-proposal")["status"] == "pending"
            monkeypatch.setattr(grants, "upsert", original)
            recovery = WorkspaceAccessRecoveryOwner(handle.sessions, workspaces, handle.bus)
            approval["expected_catalog_revision"] = workspaces.catalog.snapshot().revision
        ready = recovery.add_folder(approval)
        assert ready["workspace_access_request"]["status"] == "ready"
        assert ready["workspace_access_request"]["can_continue"]
        assert workspaces.catalog.get("plain-documents").agent_map
        assert grants.effective_grant_for(project).file_access == "read_only"
        with pytest.raises(WorkspaceAccessActionError):
            recovery.add_folder(approval)
    finally:
        await composition.aclose()
