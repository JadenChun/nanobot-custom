from pathlib import Path
from typing import cast

import pytest

from kohaku.context_repo.policy import ResourceAccessPolicy
from kohaku.security.workspace_access import default_workspace_scope
from kohaku.webui.file_preview import (
    WebUIFilePreviewError,
    _absolute_preview_paths,
    file_preview_availability_payload,
    file_preview_payload,
)


def test_restricted_preview_allows_media_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    media = tmp_path / "media"
    media.mkdir()
    uploaded = media / "upload.txt"
    uploaded.write_text("uploaded", encoding="utf-8")
    monkeypatch.setattr("kohaku.webui.file_preview.get_media_dir", lambda: media)

    scope = default_workspace_scope(workspace, restrict_to_workspace=True)

    payload = file_preview_payload(str(uploaded), scope=scope)

    assert payload["content"] == "uploaded"
    assert Path(payload["path"]) == uploaded.resolve()


def test_preview_paths_ignore_tilde_traceback_markers(tmp_path: Path) -> None:
    from kohaku.webui.file_preview import _absolute_preview_paths

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
    monkeypatch.setattr("kohaku.webui.file_preview.get_media_dir", lambda: media)

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


@pytest.mark.parametrize(
    "filename,payload_bytes,expected_kind,expected_mime",
    [
        ("INV-000009.pdf", b"%PDF-1.7\n\x00binary", "pdf", "application/pdf"),
        ("demo.mp4", b"\x00\x00\x00\x18ftypmp42", "video", "video/mp4"),
        ("clip.mkv", b"\x1a\x45\xdf\xa3\x00", "video", "video/x-matroska"),
        ("clip.webm", b"\x1a\x45\xdf\xa3\x00", "video", "video/webm"),
        ("voice.mp3", b"ID3\x03\x00\x00\x00\x00", "audio", "audio/mpeg"),
        ("sound.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio", "audio/wav"),
        ("track.flac", b"fLaC\x00\x00\x00\x22", "audio", "audio/flac"),
        ("memo.m4a", b"\x00\x00\x00\x20ftypM4A ", "audio", "audio/mp4"),
        ("note.ogg", b"OggS\x00\x02", "audio", "audio/ogg"),
        ("shot.png", b"\x89PNG\r\n\x1a\n\x00", "image", "image/png"),
        ("pic.webp", b"RIFF\x00\x00\x00\x00WEBP", "image", "image/webp"),
        ("photo.avif", b"\x00\x00\x00\x1cftypavif", "image", "image/avif"),
    ],
)
def test_media_and_pdf_preview_availability_and_payload(
    tmp_path: Path,
    filename: str,
    payload_bytes: bytes,
    expected_kind: str,
    expected_mime: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / filename
    target.write_bytes(payload_bytes)
    scope = default_workspace_scope(workspace, restrict_to_workspace=True)

    assert file_preview_availability_payload(str(target), scope=scope) == {"available": True}
    payload = file_preview_payload(
        str(target),
        scope=scope,
        media_url=lambda p: {"url": f"/api/media/sig/{p.name}"},
    )
    assert payload["kind"] == expected_kind
    assert payload["mime"] == expected_mime
    assert payload["media_url"] == f"/api/media/sig/{filename}"
    assert payload["content"] == ""


def test_office_docx_preview_extracts_text_and_media_url(tmp_path: Path) -> None:
    import docx

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "quote.docx"
    document = docx.Document()
    document.add_paragraph("Quotation for TTL & CPC")
    document.save(str(target))

    scope = default_workspace_scope(workspace, restrict_to_workspace=True)
    assert file_preview_availability_payload(str(target), scope=scope) == {"available": True}

    payload = file_preview_payload(
        str(target),
        scope=scope,
        media_url=lambda p: {"url": f"/api/media/sig/{p.name}"},
    )
    assert payload["kind"] == "text"
    assert "Quotation for TTL & CPC" in payload["content"]
    assert payload["media_url"] == "/api/media/sig/quote.docx"


def test_absolute_preview_paths_extracts_quoted_and_multiline_stdout_paths(tmp_path: Path) -> None:
    target = tmp_path / "company website" / "docs" / "INV-000009.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF-1.7\n")

    quoted_cmd = f'/usr/bin/render --out "{target}"'
    assert target.resolve(strict=False) in _absolute_preview_paths(quoted_cmd)

    multiline_stdout = (
        "InvoiceShelf up: http://localhost:8724\n"
        f"Rendered INV-000009 -> {target} (878,937 bytes)\n\n"
        "STDERR:\nwarning line"
    )
    assert target.resolve(strict=False) in _absolute_preview_paths(multiline_stdout)

    two_paths = _absolute_preview_paths("Copied /tmp/left/draft.txt to /tmp/right/draft.txt")
    assert Path("/tmp/left/draft.txt").resolve(strict=False) in two_paths
    assert Path("/tmp/right/draft.txt").resolve(strict=False) in two_paths
    assert Path("/tmp/left/draft.txt to /tmp/right/draft.txt").resolve(strict=False) not in two_paths


def test_unknown_binary_still_rejected_with_415(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    blob = workspace / "blob.bin"
    blob.write_bytes(b"\x00binary-data")
    scope = default_workspace_scope(workspace, restrict_to_workspace=True)

    with pytest.raises(WebUIFilePreviewError) as avail_exc:
        file_preview_availability_payload(str(blob), scope=scope)
    assert avail_exc.value.status == 415

    with pytest.raises(WebUIFilePreviewError) as payload_exc:
        file_preview_payload(str(blob), scope=scope)
    assert payload_exc.value.status == 415
