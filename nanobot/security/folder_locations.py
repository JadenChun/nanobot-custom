"""Trusted, retryable user edits joining folder metadata and Sandbox grants."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from nanobot.context_repo.catalog import (
    CatalogRevisionConflictError,
    RepositoryCatalog,
    RepositoryEntry,
)
from nanobot.security.folder_grants import (
    FolderGrant,
    FolderGrantStore,
    GrantRevisionConflictError,
)


class FolderLocationEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    expected_grants_revision: StrictInt = Field(ge=0)
    expected_catalog_revision: StrictInt = Field(ge=0)
    path: str = Field(min_length=1, max_length=2048)
    entry: RepositoryEntry | None
    access: Literal["keep", "inherit", "read_only", "read_write"]


class FolderEditIncompleteError(OSError):
    """Navigation was saved, but the requested access has not been confirmed."""


class _FolderEditJournal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    edit: FolderLocationEdit
    canonical_path: str
    desired_grant: FolderGrant | None
    change_grant: bool
    catalog_done: bool = False
    grants_done: bool = False
    # The original edit stays intact for durable history; these record the
    # latest trusted renewal of an outstanding grant step.
    renewed_expected_grants_revision: StrictInt | None = Field(default=None, ge=0)
    renewed_expected_catalog_revision: StrictInt | None = Field(default=None, ge=0)
    # Exact catalog revision where this operation verified or committed its
    # metadata. Missing values identify journals written before this receipt.
    catalog_receipt_revision: StrictInt | None = Field(default=None, ge=0)
    # None marks a journal written before this field existed; False keeps an
    # unresolved operation's identity even if both persistence steps ran.
    verified: StrictBool | None = None


def _write_journal(path: Path, record: _FolderEditJournal) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(record.model_dump_json().encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _governing_grant(grants: tuple[FolderGrant, ...], canonical: Path) -> FolderGrant | None:
    """Return the most-specific grant covering the target, if any."""
    matching = [row for row in grants if canonical.is_relative_to(Path(row.path))]
    return max(matching, key=lambda row: len(Path(row.path).parts), default=None)


def _effective_file_access(
    store: FolderGrantStore,
    canonical: Path,
    grant: FolderGrant | None,
) -> Literal["read_only", "read_write"]:
    """Mirror the policy's workspace-vs-grant specificity precedence."""
    if grant is not None and (
        not canonical.is_relative_to(store.agent_home)
        or len(Path(grant.path).parts) >= len(store.agent_home.parts)
    ):
        return grant.file_access
    if canonical.is_relative_to(store.agent_home):
        return "read_write"
    return "read_only"


def _journal_result_is_satisfied(
    store: FolderGrantStore,
    catalog: RepositoryCatalog,
    journal: _FolderEditJournal,
) -> bool:
    """Check a legacy journal's intended result before cleanup deletes it."""
    canonical = journal.canonical_path
    try:
        snapshot = catalog.snapshot()
        grants = store.snapshot()
    except (OSError, RuntimeError, ValueError):
        return False
    current = next((row for row in snapshot.repositories if row.path == canonical), None)
    intended = journal.edit.entry
    if intended is not None:
        intended = intended.model_copy(update={"path": canonical})
    if current != intended:
        return False
    path = Path(canonical)
    if journal.edit.access in {"read_only", "read_write"}:
        actual = _effective_file_access(store, path, _governing_grant(grants.grants, path))
        return actual == journal.edit.access
    if journal.edit.access == "inherit":
        return not any(row.path == canonical for row in grants.grants)
    return True


def apply_folder_edit(store: FolderGrantStore, catalog: RepositoryCatalog, edit: FolderLocationEdit) -> None:
    """One UI action, journaled before either durable store changes.

    A failed action can leave metadata saved without its requested grant. Reusing
    the operation id resumes exactly that action; differing inputs or concurrent
    edits conflict. Success requires both stores to have the intended result.
    """
    catalog.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with FileLock(str(catalog.root / "folder-edits.lock"), timeout=5):
        journal_path = catalog.root / f"folder-edit-{edit.operation_id}.json"
        requested = Path(edit.path).expanduser()
        if not requested.is_absolute():
            raise ValueError("folder path must be absolute")
        removing = edit.entry is None and edit.access in {"inherit", "keep"}
        canonical = Path(os.path.abspath(requested)) if removing else requested.resolve(strict=True)
        if not removing and not canonical.is_dir():
            raise ValueError("folder path must be an existing directory")
        if not removing and store.view.is_private_path(canonical):
            raise ValueError("folder is inside private runtime storage")
        entry = edit.entry
        if entry is not None:
            if Path(entry.path).expanduser().resolve(strict=True) != canonical:
                raise ValueError("folder metadata path does not match the selected folder")
            entry = entry.model_copy(update={"path": str(canonical)})
        if journal_path.exists():
            if journal_path.is_symlink() or journal_path.stat().st_size > 32 * 1024:
                raise ValueError("invalid folder edit journal")
            journal = _FolderEditJournal.model_validate_json(journal_path.read_bytes())
            retry_fields = {"expected_grants_revision", "expected_catalog_revision"}
            if (journal.edit.model_dump(exclude=retry_fields) != edit.model_dump(exclude=retry_fields)
                    or journal.canonical_path != str(canonical)):
                raise ValueError("folder_edit_changed")
            if not journal.grants_done:
                # A recorded tuple is valid at its catalog revision or across
                # this operation's exact CAS commit. A new confirmation must
                # match both current revisions and the intended catalog row.
                current_grants = store.snapshot()
                current_catalog = catalog.snapshot()
                if current_grants.revision != edit.expected_grants_revision:
                    raise GrantRevisionConflictError("grants_changed")
                current_entry = next(
                    (row for row in current_catalog.repositories if row.path == str(canonical)),
                    None,
                )
                if journal.catalog_done and current_entry != entry:
                    raise CatalogRevisionConflictError("catalog_changed")
                recorded = {
                    (
                        journal.edit.expected_grants_revision,
                        journal.edit.expected_catalog_revision,
                    ),
                }
                if (journal.renewed_expected_grants_revision is not None
                        and journal.renewed_expected_catalog_revision is not None):
                    recorded.add((
                        journal.renewed_expected_grants_revision,
                        journal.renewed_expected_catalog_revision,
                    ))
                submitted = (
                    edit.expected_grants_revision,
                    edit.expected_catalog_revision,
                )
                if submitted in recorded:
                    resumed_own_commit = (
                        journal.catalog_done
                        and journal.catalog_receipt_revision is not None
                        and edit.expected_catalog_revision < journal.catalog_receipt_revision
                        and current_catalog.revision == journal.catalog_receipt_revision
                    )
                    if (current_catalog.revision != edit.expected_catalog_revision
                            and not resumed_own_commit):
                        raise CatalogRevisionConflictError("catalog_changed")
                else:
                    if current_catalog.revision != edit.expected_catalog_revision:
                        raise CatalogRevisionConflictError("catalog_changed")
                    if current_entry != entry:
                        raise CatalogRevisionConflictError("catalog_changed")
                    journal = journal.model_copy(update={
                        "renewed_expected_grants_revision": edit.expected_grants_revision,
                        "renewed_expected_catalog_revision": edit.expected_catalog_revision,
                    })
                    _write_journal(journal_path, journal)
                edit = journal.edit.model_copy(update={
                    "expected_grants_revision": edit.expected_grants_revision,
                    "expected_catalog_revision": edit.expected_catalog_revision,
                })
            else:
                edit = journal.edit
        else:
            grants = store.snapshot()
            current = catalog.snapshot()
            if grants.revision != edit.expected_grants_revision:
                raise GrantRevisionConflictError("grants_changed")
            if current.revision != edit.expected_catalog_revision:
                raise CatalogRevisionConflictError("catalog_changed")
            old = next((row for row in current.repositories if row.path == str(canonical)), None)
            if old is not None and (entry is None or entry.name != old.name):
                if any(old.name in row.knowledge_repos for row in current.repositories):
                    raise ValueError("repository_in_use")
            # Validate all catalog constraints before either store is mutated.
            planned = [row for row in current.repositories if row.path != str(canonical)]
            if entry is not None:
                planned.append(entry)
            type(current)(revision=current.revision, repositories=planned)
            exact = next((row for row in grants.grants if row.path == str(canonical)), None)
            desired = exact
            if edit.access == "inherit":
                desired = None
            elif edit.access in {"read_only", "read_write"}:
                governing = _governing_grant(grants.grants, canonical)
                workspace_governs = canonical.is_relative_to(store.agent_home) and (
                    governing is None
                    or len(Path(governing.path).parts) < len(store.agent_home.parts)
                )
                # Reuse a governing rule only when it actually supplies the
                # requested access in the policy's precedence. Otherwise the
                # explicit selection becomes a more specific rule.
                if exact is not None and exact.file_access == edit.access:
                    desired = exact
                elif (governing is not None and not workspace_governs
                      and governing.file_access == edit.access):
                    desired = None
                elif workspace_governs and edit.access == "read_write":
                    desired = None
                else:
                    desired = FolderGrant(path=str(canonical), file_access="read_write" if edit.access == "read_write" else "read_only", commands=True)
                    if store.view.overlaps_private_root(canonical):
                        if edit.access == "read_only":
                            raise ValueError(
                                "read-only access cannot be enforced inside the writable agent workspace"
                            )
                        raise ValueError("folder grants cannot overlap private runtime storage")
            journal = _FolderEditJournal(edit=edit, canonical_path=str(canonical), desired_grant=desired,
                                         change_grant=desired != exact, verified=False)
            _write_journal(journal_path, journal)

        current = catalog.snapshot()
        old = next((row for row in current.repositories if row.path == str(canonical)), None)
        if not journal.catalog_done:
            if old != entry:
                if current.revision != edit.expected_catalog_revision:
                    raise CatalogRevisionConflictError("catalog_changed")
                if old is not None and entry is not None and old.name != entry.name:
                    raise ValueError("folder_name_changed: remove the old shortcut first")
                catalog.update(entry, remove=old.name if old is not None else None,
                               expected_revision=current.revision)
                # update() uses a revision CAS and increments exactly once.
                # Keep that known commit revision even if another writer edits
                # the catalog before the journal receipt is flushed.
                journal.catalog_receipt_revision = current.revision + 1
            else:
                if current.revision != edit.expected_catalog_revision:
                    raise CatalogRevisionConflictError("catalog_changed")
                journal.catalog_receipt_revision = current.revision
            journal.catalog_done = True
            _write_journal(journal_path, journal)
        elif old != entry:
            raise CatalogRevisionConflictError("catalog_changed")

        grants = store.snapshot()
        exact = next((row for row in grants.grants if row.path == str(canonical)), None)
        if not journal.grants_done:
            if journal.change_grant and exact != journal.desired_grant:
                if grants.revision != edit.expected_grants_revision:
                    raise GrantRevisionConflictError("grants_changed")
                try:
                    if journal.desired_grant is None:
                        store.revoke(str(canonical), expected_revision=grants.revision)
                    else:
                        store.upsert(journal.desired_grant.model_dump(), expected_revision=grants.revision)
                except (OSError, ValueError) as exc:
                    raise FolderEditIncompleteError("folder_metadata_saved_access_pending") from exc
            journal.grants_done = True
            _write_journal(journal_path, journal)
        elif journal.change_grant and exact != journal.desired_grant:
            raise GrantRevisionConflictError("grants_changed")
        # Catalog writers use their own lock, so confirm the intended metadata
        # after the grant step as well. A concurrent change is reported, never
        # overwritten, and the journal records whether the grant committed.
        final_catalog = catalog.snapshot()
        final_entry = next(
            (row for row in final_catalog.repositories if row.path == str(canonical)),
            None,
        )
        if final_entry != entry:
            raise CatalogRevisionConflictError("catalog_changed")
        if edit.access in {"read_only", "read_write"}:
            grants = store.snapshot()
            actual = _effective_file_access(
                store,
                canonical,
                _governing_grant(grants.grants, canonical),
            )
            if actual != edit.access:
                raise GrantRevisionConflictError("grants_changed")
        journal.verified = True
        _write_journal(journal_path, journal)
        # Keep a bounded retry window; pending operations are never discarded.
        completed = sorted(catalog.root.glob("folder-edit-*.json"), key=lambda path: path.stat().st_mtime)
        for old_path in completed[:-128]:
            if old_path == journal_path or old_path.stat().st_size > 32 * 1024:
                continue
            old_journal = _FolderEditJournal.model_validate_json(old_path.read_bytes())
            if not (old_journal.catalog_done and old_journal.grants_done):
                continue
            if old_journal.verified is True:
                old_path.unlink()
            elif (old_journal.verified is None
                  and _journal_result_is_satisfied(store, catalog, old_journal)):
                # Journals written before the verified flag existed are kept
                # unless their intended combined result is still present.
                old_path.unlink()
