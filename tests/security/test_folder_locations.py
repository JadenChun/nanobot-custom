"""Combined folder edits preserve authority and survive partial persistence."""
import json
from pathlib import Path
from uuid import uuid4

import pytest

from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.context_repo.catalog import (
    CatalogRevisionConflictError,
    RepositoryCatalog,
    RepositoryEntry,
)
from nanobot.security.folder_grants import FolderGrantStore, GrantRevisionConflictError
from nanobot.security.folder_locations import FolderLocationEdit, apply_folder_edit


def test_folder_edit_reuses_parent_and_retries_partial_commit(tmp_path: Path, monkeypatch):
    home, documents, project = (tmp_path / name for name in ("home", "documents", "documents/plain-folder"))
    for path in (home, documents, project):
        path.mkdir(parents=True, exist_ok=True)
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    store.upsert({"path": str(documents), "file_access": "read_write", "commands": False}, expected_revision=0)
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=1, expected_catalog_revision=0, path=str(project), entry=RepositoryEntry(name="plain", path=str(project)), access="read_write")
    apply_folder_edit(store, catalog, edit)
    assert store.snapshot().revision == 1
    assert catalog.get("plain").agent_map
    assert store.effective_grant_for(project).path == str(documents)
    override = edit.model_copy(update={"operation_id": uuid4().hex, "expected_catalog_revision": 1, "access": "read_only"})
    original = store.upsert
    def fail(*args, **kwargs):
        raise OSError("simulated permission store write failure")
    monkeypatch.setattr(store, "upsert", fail)
    with pytest.raises(OSError):
        apply_folder_edit(store, catalog, override)
    assert store.effective_grant_for(project).file_access == "read_write"
    monkeypatch.setattr(store, "upsert", original)
    apply_folder_edit(store, catalog, override)
    apply_folder_edit(store, catalog, override)
    assert store.effective_grant_for(project).file_access == "read_only"
    assert store.snapshot().revision == 2
    with pytest.raises(GrantRevisionConflictError):
        apply_folder_edit(store, catalog, edit.model_copy(update={"operation_id": uuid4().hex}))


def test_workspace_folder_reuses_implicit_write_without_a_grant(tmp_path: Path):
    instance = tmp_path / "instance"
    home = instance / "workspace"
    project = home / "project"
    project.mkdir(parents=True)
    store = FolderGrantStore(agent_home=home, instance_root=instance, path=instance / "webui/folder-grants.json")
    catalog = RepositoryCatalog(instance / "catalog")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    store.upsert(
        {"path": str(unrelated), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=1, expected_catalog_revision=0, path=str(project), entry=RepositoryEntry(name="project", path=str(project)), access="read_write")

    apply_folder_edit(store, catalog, edit)

    assert catalog.get("project").agent_map
    assert store.snapshot().revision == 1
    assert store.effective_grant_for(project) is None


def test_workspace_read_only_child_stays_an_explicit_error(tmp_path: Path):
    instance = tmp_path / "instance"
    home = instance / "workspace"
    project = home / "project"
    project.mkdir(parents=True)
    store = FolderGrantStore(agent_home=home, instance_root=instance, path=instance / "webui/folder-grants.json")
    catalog = RepositoryCatalog(instance / "catalog")
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=0, expected_catalog_revision=0, path=str(project), entry=RepositoryEntry(name="project", path=str(project)), access="read_only")

    with pytest.raises(ValueError, match="read-only"):
        apply_folder_edit(store, catalog, edit)

    assert not catalog.entries()
    assert store.snapshot().revision == 0


def test_workspace_edit_still_rejects_private_sibling_storage(tmp_path: Path):
    instance = tmp_path / "instance"
    home = instance / "workspace"
    home.mkdir(parents=True)
    credentials = instance / "credentials"
    credentials.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=instance, path=instance / "webui/folder-grants.json")
    catalog = RepositoryCatalog(instance / "catalog")
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=0, expected_catalog_revision=0, path=str(credentials), entry=RepositoryEntry(name="credentials", path=str(credentials)), access="read_write")

    with pytest.raises(ValueError, match="private runtime storage"):
        apply_folder_edit(store, catalog, edit)

    assert not catalog.entries()
    assert store.snapshot().revision == 0


def test_custom_workspace_read_only_child_is_enforced(tmp_path: Path):
    documents = tmp_path / "documents"
    home = documents / "agent-workspace"
    project = home / "project"
    instance = tmp_path / "instance"
    project.mkdir(parents=True)
    instance.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=instance, path=instance / "webui/folder-grants.json")
    catalog = RepositoryCatalog(instance / "catalog")
    store.upsert(
        {"path": str(documents), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
        sandbox_default=True,
    )
    marker = project / "marker.txt"
    policy.validate_path(marker, "write")
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=1, expected_catalog_revision=0, path=str(project), entry=RepositoryEntry(name="project", path=str(project)), access="read_only")

    apply_folder_edit(store, catalog, edit)

    with pytest.raises(PermissionError):
        policy.validate_path(marker, "write")
    assert store.effective_grant_for(project).file_access == "read_only"
    assert store.snapshot().revision == 2


def test_custom_workspace_read_write_reuses_workspace_over_read_only_parent(tmp_path: Path):
    documents = tmp_path / "documents"
    home = documents / "agent-workspace"
    project = home / "project"
    instance = tmp_path / "instance"
    project.mkdir(parents=True)
    instance.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=instance, path=instance / "webui/folder-grants.json")
    catalog = RepositoryCatalog(instance / "catalog")
    store.upsert(
        {"path": str(documents), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
        sandbox_default=True,
    )
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=1, expected_catalog_revision=0, path=str(project), entry=RepositoryEntry(name="project", path=str(project)), access="read_write")

    apply_folder_edit(store, catalog, edit)

    policy.validate_path(project / "marker.txt", "write")
    assert store.snapshot().revision == 1
    assert not any(row.path == str(project) for row in store.snapshot().grants)


def test_read_only_child_under_writable_parent_blocks_policy_write(tmp_path: Path):
    home, documents, project = (tmp_path / name for name in ("home", "documents", "documents/plain-folder"))
    for path in (home, documents, project):
        path.mkdir(parents=True, exist_ok=True)
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    store.upsert(
        {"path": str(documents), "file_access": "read_write", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
        sandbox_default=True,
    )
    policy.validate_path(project / "marker.txt", "write")
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=1, expected_catalog_revision=0, path=str(project), entry=RepositoryEntry(name="plain", path=str(project)), access="read_only")

    apply_folder_edit(store, catalog, edit)

    with pytest.raises(PermissionError):
        policy.validate_path(project / "marker.txt", "write")


@pytest.mark.parametrize("existing_entry", [False, True])
def test_identical_retry_resumes_own_catalog_commit(tmp_path: Path, monkeypatch, existing_entry: bool):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    expected_catalog = 0
    if existing_entry:
        catalog.update(RepositoryEntry(name="project", path=str(project), purpose="Old purpose"))
        expected_catalog = catalog.snapshot().revision
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=0, expected_catalog_revision=expected_catalog, path=str(project), entry=RepositoryEntry(name="project", path=str(project), purpose="Approved purpose"), access="read_write")
    original = store.upsert
    def fail(*args, **kwargs):
        raise OSError("simulated transient grant persistence failure")
    monkeypatch.setattr(store, "upsert", fail)
    with pytest.raises(OSError):
        apply_folder_edit(store, catalog, edit)
    committed_catalog = catalog.snapshot().revision
    assert committed_catalog == expected_catalog + 1
    journal_path = catalog.root / f"folder-edit-{edit.operation_id}.json"
    assert json.loads(journal_path.read_text())["catalog_receipt_revision"] == committed_catalog
    monkeypatch.setattr(store, "upsert", original)

    apply_folder_edit(store, catalog, edit)

    assert catalog.snapshot().revision == committed_catalog
    assert catalog.get("project").purpose == "Approved purpose"
    assert store.effective_grant_for(project).file_access == "read_write"
    assert store.snapshot().revision == 1


def test_catalog_receipt_keeps_own_cas_revision_when_another_writer_follows(
    tmp_path: Path,
    monkeypatch,
):
    home, project, unrelated = (tmp_path / name for name in ("home", "project", "unrelated"))
    for directory in (home, project, unrelated):
        directory.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    edit = FolderLocationEdit(
        operation_id=uuid4().hex,
        expected_grants_revision=0,
        expected_catalog_revision=0,
        path=str(project),
        entry=RepositoryEntry(name="project", path=str(project), purpose="Approved purpose"),
        access="read_write",
    )
    original_update = catalog.update
    original_upsert = store.upsert

    def update_then_unrelated(entry, *, remove=None, expected_revision=None):
        result = original_update(entry, remove=remove, expected_revision=expected_revision)
        if entry is not None and entry.path == str(project):
            original_update(
                RepositoryEntry(name="unrelated", path=str(unrelated)),
                expected_revision=1,
            )
        return result

    def fail(*args, **kwargs):
        raise OSError("simulated transient grant persistence failure")

    monkeypatch.setattr(catalog, "update", update_then_unrelated)
    monkeypatch.setattr(store, "upsert", fail)
    with pytest.raises(OSError):
        apply_folder_edit(store, catalog, edit)

    assert catalog.snapshot().revision == 2
    journal_path = catalog.root / f"folder-edit-{edit.operation_id}.json"
    assert json.loads(journal_path.read_text())["catalog_receipt_revision"] == 1
    monkeypatch.setattr(catalog, "update", original_update)
    monkeypatch.setattr(store, "upsert", original_upsert)
    with pytest.raises(CatalogRevisionConflictError, match="catalog_changed"):
        apply_folder_edit(store, catalog, edit)
    assert store.snapshot().revision == 0


def test_journal_without_receipt_needs_fresh_tuple_when_metadata_already_matches(
    tmp_path: Path,
    monkeypatch,
):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    edit = FolderLocationEdit(
        operation_id=uuid4().hex,
        expected_grants_revision=0,
        expected_catalog_revision=0,
        path=str(project),
        entry=RepositoryEntry(name="project", path=str(project), purpose="Approved purpose"),
        access="read_write",
    )
    original = store.upsert

    def fail(*args, **kwargs):
        raise OSError("simulated transient grant persistence failure")

    monkeypatch.setattr(store, "upsert", fail)
    with pytest.raises(OSError):
        apply_folder_edit(store, catalog, edit)
    monkeypatch.setattr(store, "upsert", original)

    journal_path = catalog.root / f"folder-edit-{edit.operation_id}.json"
    legacy_journal = json.loads(journal_path.read_text())
    legacy_journal["catalog_done"] = False
    legacy_journal.pop("catalog_receipt_revision", None)
    journal_path.write_text(json.dumps(legacy_journal))
    assert catalog.snapshot().revision == 1

    with pytest.raises(CatalogRevisionConflictError, match="catalog_changed"):
        apply_folder_edit(store, catalog, edit)
    assert store.snapshot().revision == 0

    fresh = edit.model_copy(update={"expected_catalog_revision": 1})
    apply_folder_edit(store, catalog, fresh)
    assert catalog.snapshot().revision == 1
    assert store.snapshot().revision == 1
    journal = json.loads(journal_path.read_text())
    assert journal["catalog_done"] is True
    assert journal["catalog_receipt_revision"] == 1


def test_original_approval_does_not_resume_after_later_catalog_edit(tmp_path: Path, monkeypatch):
    home, project, unrelated = (tmp_path / name for name in ("home", "project", "unrelated"))
    for directory in (home, project, unrelated):
        directory.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    edit = FolderLocationEdit(
        operation_id=uuid4().hex,
        expected_grants_revision=0,
        expected_catalog_revision=0,
        path=str(project),
        entry=RepositoryEntry(name="project", path=str(project), purpose="Approved purpose"),
        access="read_write",
    )
    original = store.upsert

    def fail(*args, **kwargs):
        raise OSError("simulated transient grant persistence failure")

    monkeypatch.setattr(store, "upsert", fail)
    with pytest.raises(OSError):
        apply_folder_edit(store, catalog, edit)
    monkeypatch.setattr(store, "upsert", original)

    journal_path = catalog.root / f"folder-edit-{edit.operation_id}.json"
    assert json.loads(journal_path.read_text())["catalog_receipt_revision"] == 1
    catalog.update(RepositoryEntry(name="unrelated", path=str(unrelated)))
    assert catalog.snapshot().revision == 2

    with pytest.raises(CatalogRevisionConflictError, match="catalog_changed"):
        apply_folder_edit(store, catalog, edit)

    assert store.snapshot().revision == 0
    assert store.effective_grant_for(project) is None
    journal = json.loads(journal_path.read_text())
    assert journal["renewed_expected_catalog_revision"] is None
    assert journal["catalog_receipt_revision"] == 1

    fresh = edit.model_copy(update={"expected_catalog_revision": 2})
    apply_folder_edit(store, catalog, fresh)

    assert store.snapshot().revision == 1
    assert store.effective_grant_for(project).file_access == "read_write"
    journal = json.loads(journal_path.read_text())
    assert journal["edit"]["expected_catalog_revision"] == 0
    assert journal["renewed_expected_catalog_revision"] == 2


def test_failed_combined_result_keeps_journal_and_conflict(tmp_path: Path, monkeypatch):
    home, project, other, granted = (tmp_path / name for name in ("home", "project", "other", "unrelated-grant"))
    for directory in (home, project, other, granted):
        directory.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    entry = RepositoryEntry(name="project", path=str(project), purpose="Approved purpose")
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=0, expected_catalog_revision=0, path=str(project), entry=entry, access="read_write")
    original = store.upsert
    def fail(*args, **kwargs):
        raise OSError("simulated transient grant persistence failure")
    monkeypatch.setattr(store, "upsert", fail)
    with pytest.raises(OSError):
        apply_folder_edit(store, catalog, edit)
    monkeypatch.setattr(store, "upsert", original)
    store.upsert({"path": str(granted), "file_access": "read_only", "commands": False}, expected_revision=0)

    renewed = edit.model_copy(update={"expected_grants_revision": 1, "expected_catalog_revision": 1})
    def upsert_with_concurrent_metadata(*args, **kwargs):
        saved = original(*args, **kwargs)
        catalog.update(entry.model_copy(update={"purpose": "Concurrent editor purpose"}))
        return saved
    monkeypatch.setattr(store, "upsert", upsert_with_concurrent_metadata)
    with pytest.raises(CatalogRevisionConflictError, match="catalog_changed"):
        apply_folder_edit(store, catalog, renewed)

    journal_path = catalog.root / f"folder-edit-{edit.operation_id}.json"
    journal = json.loads(journal_path.read_text())
    assert journal["catalog_done"] is True
    assert journal["grants_done"] is True
    assert journal["verified"] is False
    assert catalog.get("project").purpose == "Concurrent editor purpose"
    granted_access = store.effective_grant_for(project)
    assert granted_access is not None and granted_access.file_access == "read_write"

    for number in range(129):
        unrelated = FolderLocationEdit(
            operation_id=uuid4().hex,
            expected_grants_revision=store.snapshot().revision,
            expected_catalog_revision=catalog.snapshot().revision,
            path=str(other),
            entry=RepositoryEntry(name="other", path=str(other), purpose=f"Unrelated edit {number}"),
            access="keep",
        )
        apply_folder_edit(store, catalog, unrelated)

    assert journal_path.exists()
    replay = edit.model_copy(update={
        "expected_grants_revision": store.snapshot().revision,
        "expected_catalog_revision": catalog.snapshot().revision,
    })
    with pytest.raises(CatalogRevisionConflictError, match="catalog_changed"):
        apply_folder_edit(store, catalog, replay)
    assert catalog.get("project").purpose == "Concurrent editor purpose"


def test_legacy_journal_cleanup_keeps_only_satisfied_results(tmp_path: Path):
    home, satisfied, conflicted, bulk = (tmp_path / name for name in ("home", "satisfied", "conflicted", "bulk"))
    for directory in (home, satisfied, conflicted, bulk):
        directory.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")

    satisfied_edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=0, expected_catalog_revision=0, path=str(satisfied), entry=RepositoryEntry(name="satisfied", path=str(satisfied), purpose="Done"), access="keep")
    apply_folder_edit(store, catalog, satisfied_edit)
    satisfied_journal = catalog.root / f"folder-edit-{satisfied_edit.operation_id}.json"

    conflicted_edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=0, expected_catalog_revision=catalog.snapshot().revision, path=str(conflicted), entry=RepositoryEntry(name="conflicted", path=str(conflicted), purpose="Approved"), access="keep")
    apply_folder_edit(store, catalog, conflicted_edit)
    conflicted_journal = catalog.root / f"folder-edit-{conflicted_edit.operation_id}.json"
    current = catalog.get("conflicted")
    catalog.update(current.model_copy(update={"purpose": "Concurrent editor purpose"}))
    for journal in (satisfied_journal, conflicted_journal):
        payload = json.loads(journal.read_text())
        payload.pop("verified", None)
        journal.write_text(json.dumps(payload))

    for number in range(129):
        edit = FolderLocationEdit(
            operation_id=uuid4().hex,
            expected_grants_revision=store.snapshot().revision,
            expected_catalog_revision=catalog.snapshot().revision,
            path=str(bulk),
            entry=RepositoryEntry(name="bulk", path=str(bulk), purpose=f"Bulk edit {number}"),
            access="keep",
        )
        apply_folder_edit(store, catalog, edit)

    assert not satisfied_journal.exists()
    assert conflicted_journal.exists()


def test_map_only_and_removing_shortcut_never_revoke_parent(tmp_path: Path):
    home, documents, project = (tmp_path / name for name in ("home", "documents", "documents/project"))
    for path in (home, documents, project):
        path.mkdir(parents=True, exist_ok=True)
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    store.upsert({"path": str(documents), "file_access": "read_write", "commands": True}, expected_revision=0)
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=1, expected_catalog_revision=0, path=str(project), entry=RepositoryEntry(name="project", path=str(project)), access="keep")
    apply_folder_edit(store, catalog, edit)
    apply_folder_edit(store, catalog, edit.model_copy(update={"operation_id": uuid4().hex, "expected_catalog_revision": 1, "entry": None, "access": "inherit"}))
    assert not catalog.entries()
    assert store.snapshot().revision == 1
    assert store.effective_grant_for(project).path == str(documents)


def test_map_edit_preserves_legacy_command_bit_and_grant_revision(tmp_path: Path):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    store.upsert({"path": str(project), "file_access": "read_write", "commands": False}, expected_revision=0)
    edit = FolderLocationEdit(operation_id=uuid4().hex, expected_grants_revision=1, expected_catalog_revision=0, path=str(project), entry=RepositoryEntry(name="project", path=str(project), agent_map=False), access="read_write")
    apply_folder_edit(store, catalog, edit)
    assert store.snapshot().revision == 1
    assert not store.effective_grant_for(project).commands
    assert not catalog.get("project").agent_map
