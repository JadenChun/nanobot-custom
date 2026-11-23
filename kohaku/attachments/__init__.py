"""Canonical server-owned attachment store, ingress, and context."""

from kohaku.attachments.adoption import AdoptionResult, adopt_media_paths
from kohaku.attachments.ingress import (
    AttachmentIngress,
    extract_data_url_mime,
    ingress_for_limits,
)
from kohaku.attachments.lifecycle import (
    prune_unreferenced,
    reference_evidence,
    referenced_hashes,
)
from kohaku.attachments.models import (
    ATTACHMENT_SCHEMA_VERSION,
    AttachmentBatchResult,
    AttachmentInput,
    AttachmentRecord,
    StagedAttachment,
)
from kohaku.attachments.policy import (
    DEFAULT_ATTACHMENT_POLICY,
    AttachmentIngressLimits,
    AttachmentPolicy,
    AttachmentRejection,
)
from kohaku.attachments.store import AttachmentStore
from kohaku.attachments.uploads import PendingUpload, UploadRegistry

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
    "reference_evidence",
    "referenced_hashes",
]
