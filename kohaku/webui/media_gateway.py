"""Media gateway services shared by WebUI HTTP routes and WebSocket frames."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Any

from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from kohaku.attachments.ingress import AttachmentIngress
from kohaku.attachments.models import AttachmentRecord
from kohaku.attachments.policy import AttachmentPolicy, AttachmentRejection
from kohaku.attachments.store import AttachmentStore
from kohaku.attachments.uploads import UploadRegistry
from kohaku.config.paths import get_media_dir
from kohaku.context_repo.policy import ResourceAccessPolicy
from kohaku.webui.attachment_ingress import AttachmentIngressResult
from kohaku.webui.ingress_policy import AttachmentIngressLimits
from kohaku.webui.media_api import (
    serve_signed_media,
    sign_existing_media_path,
    sign_or_stage_media_path,
    signed_media_attachments,
    stage_managed_media_path,
)
from kohaku.webui.transcript import rewrite_local_markdown_images


def _default_media_dir(channel: str | None) -> Path:
    return get_media_dir(channel)


class WebUIMediaGateway:
    """Own media URL signing and WebUI markdown/media augmentation."""

    def __init__(
        self,
        *,
        workspace_path: Path,
        logger: Any,
        media_dir: Callable[[str | None], Path] | None = None,
        secret: bytes | None = None,
        attachment_limits: AttachmentIngressLimits | None = None,
        resource_policy: ResourceAccessPolicy | None = None,
    ) -> None:
        self.workspace_path = workspace_path
        self.logger = logger
        self._media_dir: Callable[[str | None], Path] = media_dir or _default_media_dir
        self.secret = secret or secrets.token_bytes(32)
        self.attachment_limits = attachment_limits or AttachmentIngressLimits()
        self._resource_policy = resource_policy
        media_root = self._media_dir(None)
        self.attachment_store = AttachmentStore(
            media_root / "attachments",
            media_root=media_root,
            logger=logger,
        )
        self.attachment_ingress = AttachmentIngress(
            self.attachment_store,
            policy=AttachmentPolicy(limits=self.attachment_limits),
            logger=logger,
        )
        self.upload_registry = UploadRegistry()

    def set_resource_policy(self, policy: ResourceAccessPolicy | None) -> None:
        """Attach the gateway's shared live folder and managed-repository policy."""
        self._resource_policy = policy

    def store_inbound_attachments(self, media: list[Any]) -> AttachmentIngressResult:
        """Validate and persist attachments from an inbound WebUI message."""
        result = self.attachment_ingress.accept_batch(media, source="websocket")
        if result.rejection is not None:
            return [], result.rejection
        records, rejection = self.attachment_ingress.commit_batch(result.staged)
        if rejection is not None:
            return [], rejection
        return [str(self.attachment_store.path_for(record)) for record in records], None

    def store_inbound_records(
        self,
        media: list[Any],
        *,
        source: str = "websocket",
    ) -> tuple[list[AttachmentRecord], AttachmentRejection | None]:
        """Validate and persist attachments, returning their canonical records."""
        result = self.attachment_ingress.accept_batch(media, source=source)
        if result.rejection is not None:
            return [], result.rejection
        records, rejection = self.attachment_ingress.commit_batch(result.staged)
        if rejection is not None:
            return [], rejection
        return list(records), None

    def serve_signed_media(
        self,
        sig: str,
        payload: str,
        *,
        request: WsRequest | None = None,
        max_response_bytes: int | None = None,
    ) -> Response:
        return serve_signed_media(
            sig,
            payload,
            secret=self.secret,
            request=request,
            max_response_bytes=max_response_bytes,
            media_dir=self._media_dir,
        )

    def sign_or_stage_media_path(
        self,
        path: Path,
        *,
        allow_folder_grants: bool = False,
    ) -> dict[str, str] | None:
        return sign_or_stage_media_path(
            path,
            secret=self.secret,
            media_dir=self._media_dir,
            logger=self.logger,
            authorize_source=lambda source: self._authorize_managed_source(
                source,
                allow_folder_grants=allow_folder_grants,
            ),
        )

    def rewrite_local_markdown_images(
        self,
        text: str,
        *,
        workspace_path: Path | None = None,
        allow_folder_grants: bool = False,
        publication: bool = True,
    ) -> str:
        return rewrite_local_markdown_images(
            text,
            workspace_path=workspace_path or self.workspace_path,
            sign_path=lambda path: (self.sign_or_stage_media_path if publication else self.sign_media_read)(
                path,
                allow_folder_grants=allow_folder_grants,
            ),
            allow_external=bool(
                allow_folder_grants
                and self._resource_policy is not None
                and self._resource_policy.managed_mode
            ),
        )

    def sign_media_read(
        self, path: Path, *, allow_folder_grants: bool = False,
    ) -> dict[str, str] | None:
        """Sign retained managed bytes purely; newly staged bytes keep custody."""
        try:
            managed = path.resolve(strict=True).is_relative_to(self._media_dir(None).resolve(strict=True))
        except OSError:
            return None
        if managed:
            return sign_existing_media_path(path, secret=self.secret, media_dir=self._media_dir)
        return self.sign_or_stage_media_path(path, allow_folder_grants=allow_folder_grants)

    def sign_authorized_preview_media(self, path: Path) -> dict[str, str] | None:
        """Stage (if unmanaged) and sign a file already authorized by file-preview policy.

        Callers MUST have already validated ``path`` through ``_resolve_preview_path``
        (workspace boundary, ``resource_policy``, session permissions, or an explicit
        assistant transcript reference). Uses read-only deterministic staging so repeated
        previews never mint custody publication records.
        """
        managed = stage_managed_media_path(
            path,
            media_dir=self._media_dir,
            authorize_source=None,
        )
        if managed is None:
            return None
        return sign_existing_media_path(managed, secret=self.secret, media_dir=self._media_dir)

    def augment_transcript_media(
        self,
        paths: list[str],
        *,
        allow_folder_grants: bool = False,
    ) -> list[dict[str, Any]]:
        return signed_media_attachments(
            paths,
            sign_path=lambda path: self.sign_transcript_media_read(
                path,
                allow_folder_grants=allow_folder_grants,
            ),
        )

    def sign_transcript_media_read(
        self, path: Path, *, allow_folder_grants: bool = False,
    ) -> dict[str, str] | None:
        """Sign media for a transcript read without creating custody records.

        A transcript projection is read-only: the same bytes are signed again on
        every open, so minting a publication record per read accumulated one
        permanent custody entry per media file per load (measured here: over
        1,200 leaked records, and every pending scan costs ~80 ms because it
        parses each record). Stage the bytes once if the source is unmanaged,
        then sign the managed path directly so a transcript read never grows
        custody state.
        """
        managed = stage_managed_media_path(
            path,
            media_dir=self._media_dir,
            authorize_source=lambda source: self._authorize_managed_source(
                source,
                allow_folder_grants=allow_folder_grants,
            ),
        )
        if managed is None:
            return None
        return sign_existing_media_path(managed, secret=self.secret, media_dir=self._media_dir)

    def augment_transcript_user_media(self, paths: list[str]) -> list[dict[str, Any]]:
        return self.augment_transcript_media(paths)

    def _authorize_managed_source(
        self,
        path: Path,
        *,
        allow_folder_grants: bool,
    ) -> bool:
        policy = self._resource_policy
        grants = policy.folder_grants if policy is not None else None
        if policy is None or grants is None or not policy.managed_mode:
            return True
        try:
            policy.validate_path(
                path,
                "read",
                managed=True,
                allow_folder_grants=allow_folder_grants,
            )
        except PermissionError:
            return False
        return True
