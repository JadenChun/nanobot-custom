"""Copied artifact operations retain exact canonical and policy authority."""

from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from nanobot.session.artifact_factory import ConfiguredArtifactReaderFactory
from nanobot.session.artifact_operator import (
    ArtifactOperatorError,
    CopiedArtifactOperator,
)
from nanobot.session.runtime_authority import OfflineSessionOperator
from tests.session.test_artifact_reader import setup_read


@asynccontextmanager
async def copied_artifact_fixture(tmp_path: Path, *, copy_blob: bool = True):
    live = tmp_path / "live"
    live.mkdir()
    async with setup_read(live, text="retained canonical 中文 🌟") as setup:
        writer, read_file, live_store, descriptor, content = setup
        live_workspace = writer.store._backend.workspace
        live_sessions_root = writer.store._backend.sessions_root
        key = writer.owner.key
        incarnation = descriptor.incarnation
        generation = descriptor.generation

    copied_workspace = tmp_path / "copied-workspace"
    copied_sessions_root = tmp_path / "copied-sessions"
    copied_artifact_root = tmp_path / "copied-artifacts"
    shutil.copytree(live_workspace, copied_workspace)
    shutil.copytree(live_sessions_root, copied_sessions_root)
    if copy_blob:
        shutil.copytree(live_store.root, copied_artifact_root)

    bootstrap = await OfflineSessionOperator(
        copied_workspace,
        copied_sessions_root=copied_sessions_root,
        live_workspace=live_workspace,
        live_sessions_root=live_sessions_root,
    ).acquire()
    bootstrap.manager.bootstrap_existing_sessions(offline_copy=True)
    source_path = bootstrap.manager._jsonl_store.get_session_path(key)
    workspace_id = bootstrap.manager.sessions_dir.name
    record_path = (
        bootstrap.manager.sessions_dir
        / ".session-mode"
        / "records"
        / f"{bootstrap.manager._jsonl_store.storage_key(key)}.json"
    )
    assert bootstrap.close is not None
    await bootstrap.close()

    factory = ConfiguredArtifactReaderFactory(
        root=live_store.root,
        max_blob_bytes=64 * 1024 * 1024,
        max_namespace_bytes=1024 * 1024 * 1024,
    )
    operator = CopiedArtifactOperator(
        copied_workspace,
        copied_sessions_root=copied_sessions_root,
        copied_artifact_root=copied_artifact_root,
        artifact_factory=factory,
        read_file=read_file,
        live_workspace=live_workspace,
        live_sessions_root=live_sessions_root,
    )
    yield {
        "operator": operator,
        "key": key,
        "incarnation": incarnation,
        "generation": generation,
        "descriptor": descriptor,
        "content": content,
        "read_file": read_file,
        "source_path": source_path,
        "record_path": record_path,
        "artifact_root": copied_artifact_root,
        "workspace_id": workspace_id,
        "live_workspace": live_workspace,
        "live_sessions_root": live_sessions_root,
        "factory": factory,
    }


async def test_disabled_inspection_creates_no_storage(tmp_path: Path) -> None:
    workspace = tmp_path / "copy-workspace"
    sessions = tmp_path / "copy-sessions"
    artifacts = tmp_path / "copy-artifacts"
    read_workspace = tmp_path / "authorized-workspace"
    read_workspace.mkdir()
    from nanobot.agent.tools.filesystem import ReadFileTool

    operator = CopiedArtifactOperator(
        workspace,
        copied_sessions_root=sessions,
        copied_artifact_root=artifacts,
        artifact_factory=None,
        read_file=ReadFileTool(workspace=read_workspace, allowed_dir=read_workspace),
    )

    assert (await operator.inspect("cli:absent")).status == "disabled"
    assert not workspace.exists() and not sessions.exists() and not artifacts.exists()
    with pytest.raises(ArtifactOperatorError):
        await operator.rematerialize(
            "cli:absent",
            incarnation="not-authority",
            generation=0,
            reference="artifact:" + "a" * 32,
        )
    assert not workspace.exists() and not sessions.exists() and not artifacts.exists()


async def test_inspection_is_aggregate_content_free_and_never_deletes(tmp_path: Path) -> None:
    async with copied_artifact_fixture(tmp_path) as fixture:
        scope = next(fixture["artifact_root"].rglob("*.blob")).parent
        orphan = scope / ("f" * 32 + ".blob")
        orphan.write_bytes(b"private orphan")
        orphan.chmod(0o600)

        report = await fixture["operator"].inspect(fixture["key"])

        assert report.status == "complete"
        assert (report.reachable_blobs, report.reachable_bytes) == (
            1,
            len(fixture["content"]),
        )
        assert (report.unreachable_blobs, report.unreachable_bytes) == (1, len(b"private orphan"))
        assert (report.missing_blobs, report.missing_bytes) == (0, 0)
        assert fixture["descriptor"].reference not in repr(report)
        assert fixture["key"] not in repr(report)
        assert orphan.read_bytes() == b"private orphan"


async def test_inspection_reports_missing_reference_without_creating_or_deleting_blob(
    tmp_path: Path,
) -> None:
    async with copied_artifact_fixture(tmp_path) as fixture:
        blob = next(fixture["artifact_root"].rglob("*.blob"))
        blob.unlink()

        report = await fixture["operator"].inspect(fixture["key"])

        assert report.status == "incomplete"
        assert report.reachable_blobs == report.unreachable_blobs == 0
        assert (report.missing_blobs, report.missing_bytes) == (1, len(fixture["content"]))
        assert not blob.exists()


async def test_rematerialization_uses_original_policy_and_changes_no_input_bytes(
    tmp_path: Path,
) -> None:
    async with copied_artifact_fixture(tmp_path) as fixture:
        blob = next(fixture["artifact_root"].rglob("*.blob"))
        blob.unlink()
        fixture["read_file"]._workspace.joinpath("file").write_text(
            "current bytes are intentionally different", encoding="utf-8"
        )
        source_before = fixture["source_path"].read_bytes()
        record_before = fixture["record_path"].read_bytes()

        result = await fixture["operator"].rematerialize(
            fixture["key"],
            incarnation=fixture["incarnation"],
            generation=fixture["generation"],
            reference=fixture["descriptor"].reference,
        )

        assert result.status == "installed"
        assert blob.read_bytes() == fixture["content"]
        assert fixture["source_path"].read_bytes() == source_before
        assert fixture["record_path"].read_bytes() == record_before
        assert not list(tmp_path.rglob("nanobot-artifact-scratch-*"))


async def test_wrong_identity_and_changed_read_policy_fail_closed(tmp_path: Path) -> None:
    async with copied_artifact_fixture(tmp_path) as fixture:
        blob = next(fixture["artifact_root"].rglob("*.blob"))
        blob.unlink()
        source_before = fixture["source_path"].read_bytes()
        record_before = fixture["record_path"].read_bytes()

        with pytest.raises(ArtifactOperatorError) as caught:
            await fixture["operator"].rematerialize(
                fixture["key"],
                incarnation="wrong-incarnation",
                generation=fixture["generation"],
                reference=fixture["descriptor"].reference,
            )
        assert fixture["key"] not in str(caught.value)
        assert fixture["descriptor"].reference not in str(caught.value)

        other = tmp_path / "other-authorized-workspace"
        other.mkdir()
        fixture["read_file"]._workspace = other
        fixture["read_file"]._allowed_dir = other
        with pytest.raises(ArtifactOperatorError):
            await fixture["operator"].rematerialize(
                fixture["key"],
                incarnation=fixture["incarnation"],
                generation=fixture["generation"],
                reference=fixture["descriptor"].reference,
            )

        assert not blob.exists()
        assert fixture["source_path"].read_bytes() == source_before
        assert fixture["record_path"].read_bytes() == record_before


async def test_rematerializer_receives_exact_already_held_store_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with copied_artifact_fixture(tmp_path) as fixture:
        blob = next(fixture["artifact_root"].rglob("*.blob"))
        blob.unlink()
        from nanobot.session.artifact_rematerialization import ArtifactRematerializer

        original = ArtifactRematerializer.rematerialize
        observed = False

        def inspect_transaction(self, reference, *, expected, blobs=None):
            nonlocal observed
            observed = blobs is not None and blobs._store is self._reader.store
            return original(self, reference, expected=expected, blobs=blobs)

        monkeypatch.setattr(ArtifactRematerializer, "rematerialize", inspect_transaction)
        await fixture["operator"].rematerialize(
            fixture["key"],
            incarnation=fixture["incarnation"],
            generation=fixture["generation"],
            reference=fixture["descriptor"].reference,
        )
        assert observed


async def test_copied_operator_rejects_live_root_identities(tmp_path: Path) -> None:
    async with copied_artifact_fixture(tmp_path) as fixture:
        with pytest.raises(ArtifactOperatorError):
            CopiedArtifactOperator(
                fixture["live_workspace"],
                copied_sessions_root=fixture["live_sessions_root"],
                copied_artifact_root=tmp_path / "separate-artifacts",
                artifact_factory=fixture["factory"],
                read_file=fixture["read_file"],
                live_workspace=fixture["live_workspace"],
                live_sessions_root=fixture["live_sessions_root"],
            )
        with pytest.raises(ArtifactOperatorError):
            CopiedArtifactOperator(
                tmp_path / "separate-workspace",
                copied_sessions_root=tmp_path / "separate-sessions",
                copied_artifact_root=fixture["factory"].settings.root,
                artifact_factory=fixture["factory"],
                read_file=fixture["read_file"],
            )
