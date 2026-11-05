from pathlib import Path
from typing import cast

import pytest

from nanobot.context_repo.policy import ResourceAccessPolicy
from nanobot.security.workspace_access import default_workspace_scope
from nanobot.webui.file_preview import WebUIFilePreviewError, file_preview_payload


def test_restricted_preview_allows_media_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    media = tmp_path / "media"
    media.mkdir()
    uploaded = media / "upload.txt"
    uploaded.write_text("uploaded", encoding="utf-8")
    monkeypatch.setattr("nanobot.webui.file_preview.get_media_dir", lambda: media)

    scope = default_workspace_scope(workspace, restrict_to_workspace=True)

    payload = file_preview_payload(str(uploaded), scope=scope)

    assert payload["content"] == "uploaded"
    assert Path(payload["path"]) == uploaded.resolve()


def test_preview_paths_ignore_tilde_traceback_markers(tmp_path: Path) -> None:
    from nanobot.webui.file_preview import _absolute_preview_paths

    target = tmp_path / "Sena.md"
    target.write_text("content", encoding="utf-8")

    paths = _absolute_preview_paths(
        f"Traceback (most recent call last):\n~~~~~~~~~~~~~^^^^^^\n{target}"
    )

    assert target.resolve(strict=False) in paths


def test_restricted_preview_rejects_other_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    monkeypatch.setattr("nanobot.webui.file_preview.get_media_dir", lambda: media)

    scope = default_workspace_scope(workspace, restrict_to_workspace=True)

    with pytest.raises(WebUIFilePreviewError, match="outside the current workspace") as exc_info:
        file_preview_payload(str(outside), scope=scope)

    assert exc_info.value.status == 403


def test_local_granted_preview_can_read_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    granted = tmp_path / "granted"
    granted.mkdir()
    target = granted / "roadmap.md"
    target.write_text("granted content", encoding="utf-8")
    scope = default_workspace_scope(workspace, restrict_to_workspace=True)
    calls: list[tuple[Path, str, dict[str, object]]] = []

    class AllowPolicy:
        def validate_path(
            self,
            path: Path,
            action: str,
            **kwargs: object,
        ) -> None:
            calls.append((path, action, kwargs))

    payload = file_preview_payload(
        str(target),
        scope=scope,
        resource_policy=cast(ResourceAccessPolicy, AllowPolicy()),
        managed_access=True,
        allow_folder_grants=True,
        restrict_to_root=None,
    )

    assert payload["content"] == "granted content"
    assert calls and calls[0][0] == target.resolve()
    assert calls[0][1] == "read"
    assert calls[0][2]["allow_folder_grants"] is True


def test_local_preview_without_folder_grants_still_rejects_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    granted = tmp_path / "granted"
    granted.mkdir()
    target = granted / "roadmap.md"
    target.write_text("granted content", encoding="utf-8")
    scope = default_workspace_scope(workspace, restrict_to_workspace=True)

    class AllowPolicy:
        def validate_path(
            self,
            path: Path,
            action: str,
            **kwargs: object,
        ) -> None:
            _ = path, action, kwargs

    with pytest.raises(WebUIFilePreviewError, match="outside the current workspace"):
        file_preview_payload(
            str(target),
            scope=scope,
            resource_policy=cast(ResourceAccessPolicy, AllowPolicy()),
            managed_access=True,
            allow_folder_grants=False,
            restrict_to_root=None,
        )
