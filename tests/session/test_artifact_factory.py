"""Configured artifact factories admit only isolated private roots."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.context_repo import ContextRepoManager, ManagedContextRepo, ResourceAccessPolicy
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_factory import ConfiguredArtifactReaderFactory
from nanobot.session.artifact_store import ArtifactStorageError
from tests.session.test_artifact_reader import setup_read


def test_factory_captures_settings_without_creating_the_private_root(tmp_path: Path) -> None:
    root = tmp_path / "private"
    factory = ConfiguredArtifactReaderFactory(
        root=root,
        max_blob_bytes=64,
        max_namespace_bytes=128,
    )

    assert factory.settings.root == root
    assert factory.settings.max_blob_bytes == 64
    assert factory.settings.max_namespace_bytes == 128
    assert not root.exists()


@pytest.mark.parametrize("root_kind", ["workspace", "overlapping_file", "source", "git"])
def test_factory_rejects_configured_roots_that_overlap_readable_or_repository_state(
    tmp_path: Path,
    root_kind: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if root_kind == "workspace":
        root = workspace / "artifacts"
        static_roots = [workspace]
    elif root_kind == "overlapping_file":
        root = tmp_path / "private"
        static_roots = [root / "approved-history.jsonl"]
    elif root_kind == "source":
        root = Path(__file__).resolve().parents[2] / ".private-artifacts"
        static_roots = []
    else:
        repository = tmp_path / "repository"
        repository.mkdir()
        (repository / ".git").mkdir()
        root = repository / "artifacts"
        static_roots = []

    with pytest.raises(ArtifactValidationError, match="overlaps"):
        ConfiguredArtifactReaderFactory(
            root=root,
            max_blob_bytes=64,
            max_namespace_bytes=128,
            static_roots=static_roots,
        )
    assert not root.exists()


def test_factory_rejects_symlinked_root_before_private_store_construction(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ArtifactValidationError, match="symlinks"):
        ConfiguredArtifactReaderFactory(
            root=alias / "artifacts",
            max_blob_bytes=64,
            max_namespace_bytes=128,
        )
    assert not (target / "artifacts").exists()


async def test_factory_uses_current_source_namespace_and_returns_fresh_readers(tmp_path: Path) -> None:
    root = tmp_path / "private"
    async with setup_read(tmp_path) as (writer, read_file, _store, _descriptor, _content):
        factory = ConfiguredArtifactReaderFactory(
            root=root,
            max_blob_bytes=64,
            max_namespace_bytes=128,
        )

        first = factory(writer, False, read_file)
        second = factory(writer, False, read_file)

        assert first is not None and second is not None and first is not second
        assert first.store is not second.store
        assert first.store.workspace_id == writer.store._backend.sessions_dir.name
        state = writer.owner.read_snapshot().state
        assert (first.store.session_key, first.store.incarnation, first.store.generation) == (
            state.session_key,
            state.incarnation,
            state.generation,
        )
    assert root.exists()


async def test_factory_fails_closed_for_a_new_exact_file_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "private"
    async with setup_read(tmp_path) as (writer, read_file, _store, _descriptor, _content):
        read_file._extra_read_allowed_files.append(root / "public.txt")
        factory = ConfiguredArtifactReaderFactory(
            root=root,
            max_blob_bytes=64,
            max_namespace_bytes=128,
        )

        assert factory(writer, False, read_file) is None
    assert not root.exists()


async def test_factory_returns_no_reader_when_private_storage_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private"
    async with setup_read(tmp_path) as (writer, read_file, _store, _descriptor, _content):
        factory = ConfiguredArtifactReaderFactory(
            root=root,
            max_blob_bytes=64,
            max_namespace_bytes=128,
        )

        def unavailable(*_args, **_kwargs):
            raise ArtifactStorageError("unsupported")

        monkeypatch.setattr("nanobot.session.artifact_factory.PrivateArtifactStore", unavailable)
        assert factory(writer, False, read_file) is None
    assert not root.exists()


@pytest.mark.parametrize(
    ("marker_kind", "operation"),
    [
        ("directory", "write"),
        ("worktree_file", "write"),
        ("directory", "read"),
        ("worktree_file", "read"),
    ],
)
async def test_factory_revalidates_repository_marker_before_artifact_use(
    tmp_path: Path,
    marker_kind: str,
    operation: str,
) -> None:
    root = tmp_path / "private"
    async with setup_read(tmp_path) as (writer, read_file, _store, descriptor, content):
        factory = ConfiguredArtifactReaderFactory(
            root=root,
            max_blob_bytes=64,
            max_namespace_bytes=128,
        )
        reader = factory(writer, False, read_file)
        assert reader is not None
        if operation == "read":
            with reader.store.transaction() as blobs:
                blobs.put(descriptor, content)
        marker = root / ".git"
        if marker_kind == "directory":
            marker.mkdir()
        else:
            marker.write_text("gitdir: ../worktree", encoding="utf-8")

        with pytest.raises(ArtifactValidationError, match="overlaps"):
            if operation == "write":
                with reader.store.transaction() as blobs:
                    blobs.put(descriptor, content)
            else:
                reader.read(descriptor.reference)
    if operation == "write":
        assert not list(root.rglob("*.blob"))


async def test_factory_revalidates_repository_marker_during_store_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.session.artifact_store import PrivateArtifactStore

    root = tmp_path / "private"
    async with setup_read(tmp_path) as (writer, read_file, _store, _descriptor, _content):
        factory = ConfiguredArtifactReaderFactory(
            root=root,
            max_blob_bytes=64,
            max_namespace_bytes=128,
        )

        def introduce_worktree_marker(*args, **kwargs):
            root.mkdir(mode=0o700)
            (root / ".git").write_text("gitdir: ../worktree", encoding="utf-8")
            return PrivateArtifactStore(*args, **kwargs)

        monkeypatch.setattr(
            "nanobot.session.artifact_factory.PrivateArtifactStore",
            introduce_worktree_marker,
        )
        assert factory(writer, False, read_file) is None
    assert not list(root.rglob("*.blob"))


def test_factory_policy_roots_ignore_unconfigured_targets_but_retain_restricted_targets(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    context_root = tmp_path / "context"
    configured_root = tmp_path / "configured"
    manager = ContextRepoManager([ManagedContextRepo(
        path=context_root, name="context", managed=True,
        manifest={"targetRepos": {
            "unconfigured": {},
            "restricted": {"path": str(configured_root), "read": [], "write": []},
        }},
    )])
    policy = ResourceAccessPolicy(workspace=workspace, context_manager=manager)
    read_file = ReadFileTool(workspace=workspace, resource_policy=policy)

    assert any(target.path is None for target in manager.target_repos())
    assert configured_root not in policy.extra_read_dirs()
    assert configured_root not in policy.extra_write_dirs()
    roots = ConfiguredArtifactReaderFactory.read_policy_roots(read_file)
    assert None not in roots
    assert all(isinstance(path, Path) for path in roots)
    assert workspace in roots and context_root in roots and configured_root in roots

    private_root = tmp_path / "private"
    factory = ConfiguredArtifactReaderFactory(
        root=private_root, max_blob_bytes=64, max_namespace_bytes=128, static_roots=roots,
    )
    assert factory.settings.root == private_root
    assert not private_root.exists()


@pytest.mark.parametrize("overlap", [False, True])
async def test_factory_unconfigured_targets_do_not_mask_configured_root_policy(
    tmp_path: Path,
    overlap: bool,
) -> None:
    context_root = tmp_path / "context"
    configured_root = tmp_path / "configured"
    root = configured_root / "private" if overlap else tmp_path / "private"
    async with setup_read(tmp_path) as (writer, read_file, _store, _descriptor, _content):
        manager = ContextRepoManager([ManagedContextRepo(
            path=context_root, name="context", managed=True,
            manifest={"targetRepos": {
                "unconfigured": {},
                "restricted": {"path": str(configured_root), "read": [], "write": []},
            }},
        )])
        read_file._resource_policy = ResourceAccessPolicy(
            workspace=writer.store._backend.workspace, context_manager=manager,
        )
        factory = ConfiguredArtifactReaderFactory(
            root=root, max_blob_bytes=64, max_namespace_bytes=128,
        )
        reader = factory(writer, False, read_file)
        if overlap:
            assert reader is None
            assert not root.exists()
        else:
            assert reader is not None
            assert reader.store.root == root
