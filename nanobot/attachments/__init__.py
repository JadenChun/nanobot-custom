"""Canonical server-owned attachment store, ingress, and context."""

from nanobot.attachments.adoption import AdoptionResult, adopt_media_paths
from nanobot.attachments.ingress import (
    AttachmentIngress,
    extract_data_url_mime,
    ingress_for_limits,
)
from nanobot.attachments.lifecycle import prune_unreferenced, referenced_hashes
from nanobot.attachments.models import (
    ATTACHMENT_SCHEMA_VERSION,
    AttachmentBatchResult,
    AttachmentInput,
    AttachmentRecord,
    StagedAttachment,
)
from nanobot.attachments.policy import (
    DEFAULT_ATTACHMENT_POLICY,
    AttachmentIngressLimits,
    AttachmentPolicy,
    AttachmentRejection,
)
from nanobot.attachments.store import AttachmentStore
from nanobot.attachments.uploads import PendingUpload, UploadRegistry

__all__ = [
    "AdoptionResult",
    "adopt_media_paths",
    "ATTACHMENT_SCHEMA_VERSION",
    "AttachmentBatchResult",
    "AttachmentIngress",
    "AttachmentIngressLimits",
    "AttachmentInput",
    "AttachmentPolicy",
    "AttachmentRecord",
    "AttachmentRejection",
    "AttachmentStore",
    "PendingUpload",
    "UploadRegistry",
    "DEFAULT_ATTACHMENT_POLICY",
    "StagedAttachment",
    "extract_data_url_mime",
    "ingress_for_limits",
    "prune_unreferenced",
    "referenced_hashes",
]
