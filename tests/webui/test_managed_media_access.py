"""Managed WebUI media staging rechecks the live file-read authority."""

from __future__ import annotations

from pathlib import Path

from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.webui.media_gateway import WebUIMediaGateway


def _gateway(
    tmp_path: Path,
    *,
    managed: bool,
) -> tuple[WebUIMediaGateway, FolderGrantStore, Path, Path]:
    home = tmp_path / "agent-home"
    project = tmp_path / "project-b"
    instance = tmp_path / "instance"
    media_root = tmp_path / "media"
    home.mkdir()
    project.mkdir()
    media_root.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    if managed:
        store.upsert(
            {"path": str(project), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
    )
    gateway = WebUIMediaGateway(
        workspace_path=home,
        logger=None,
        media_dir=lambda _channel: media_root,
        secret=b"fixture-media-secret",
        resource_policy=policy,
    )
    return gateway, store, project, media_root


def test_replay_markdown_media_remains_readable_after_write_grant_revocation(tmp_path: Path) -> None:
    gateway, store, project, media_root = _gateway(tmp_path, managed=True)
    before = project / "before.png"
    before.write_bytes(b"authorized fixture image")

    visible = gateway.rewrite_local_markdown_images(
        "![before](before.png)",
        workspace_path=project,
        allow_folder_grants=True,
    )

    assert "/api/media/" in visible
    store.revoke(str(project), expected_revision=1)
    after = project / "after.png"
    after.write_bytes(b"revoked fixture image")

    hidden = gateway.rewrite_local_markdown_images(
        "![after](after.png)",
        workspace_path=project,
        allow_folder_grants=True,
    )

    assert "/api/media/" in hidden
    assert len(list(media_root.iterdir())) == 2


def test_managed_media_rejects_private_symlink_and_temporary_external_source(
    tmp_path: Path,
) -> None:
    gateway, store, project, media_root = _gateway(tmp_path, managed=True)
    private_image = store.path
    home_image = store.agent_home / "leak.png"
    home_image.symlink_to(private_image)

    assert gateway.rewrite_local_markdown_images(
        "![private](leak.png)",
        workspace_path=store.agent_home,
        allow_folder_grants=True,
    ) == "![private](leak.png)"

    external_image = project / "temporary.png"
    external_image.write_bytes(b"temporary fixture image")

    assert gateway.sign_or_stage_media_path(external_image, allow_folder_grants=False) is not None
    assert len(list(media_root.iterdir())) == 1


def test_legacy_media_staging_keeps_existing_behavior(tmp_path: Path) -> None:
    gateway, _store, project, media_root = _gateway(tmp_path, managed=False)
    image = project / "legacy.png"
    image.write_bytes(b"legacy fixture image")

    signed = gateway.sign_or_stage_media_path(image)

    assert signed is not None
    assert signed["url"].startswith("/api/media/")
    assert len(list(media_root.iterdir())) == 1
