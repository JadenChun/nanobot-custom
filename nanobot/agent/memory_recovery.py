"""Read-only validation for legacy memory consolidation receipts.

This module deliberately stops at the recovery boundary.  A pending receipt is
evidence that another memory writer may have been interrupted; Package B owns
the decision to apply, retain, or remove it.  The parser therefore validates
and detaches the receipt without changing the workspace.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

PENDING_MEMORY_RECEIPT_FILENAME = ".consolidation.pending.json"
"""The staging receipt filename, relative to ``workspace/memory``."""

MEMORY_RECEIPT_FILENAME = PENDING_MEMORY_RECEIPT_FILENAME
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_COUNTER = (1 << 63) - 1

_REQUIRED_FIELDS = frozenset(
    {
        "version",
        "chunk_digest",
        "history_marker_id",
        "history_entry",
        "memory_output",
        "base_memory_digest",
        "output_memory_digest",
        "reference_only",
        "created_at",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {"session_key", "start_offset", "end_offset", "session_generation"}
)
_ALIASES = frozenset(
    {"memory_update", "memory_base_digest", "memory_output_digest", "generation"}
)
_KNOWN_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS | _ALIASES


class MemoryReceiptRecoveryRequiredError(RuntimeError):
    """A pending memory receipt must be recovered before memory can start.

    The fixed text is intentional.  Receipt fields, user memory, and digest
    values must never be copied into startup or other user-facing errors.
    """

    _MESSAGE = "memory recovery required"

    def __init__(self) -> None:
        super().__init__(self._MESSAGE)


# Short descriptive aliases make the boundary convenient to callers while
# retaining one exception type and one content-free message.
MemoryReceiptRecoveryRequired = MemoryReceiptRecoveryRequiredError
MemoryRecoveryRequired = MemoryReceiptRecoveryRequiredError
MemoryRecoveryRequiredError = MemoryReceiptRecoveryRequiredError
MemoryReceiptError = MemoryReceiptRecoveryRequiredError


class MemoryReceiptValidationError(MemoryReceiptRecoveryRequiredError):
    """The receipt exists but is malformed, unsafe, or insufficiently evidenced."""


@dataclass(frozen=True, slots=True)
class MemoryReceipt:
    """Detached, validated data from one pending legacy receipt.

    ``raw_bytes`` and ``raw_fields`` are retained for the later recovery
    package.  The parser never returns references to a mutable decoder object;
    callers receive detached copies of unknown metadata instead.
    """

    version: int
    chunk_digest: str
    history_marker_id: str
    history_entry: str
    memory_output: str | None
    base_memory_digest: str
    output_memory_digest: str | None
    reference_only: bool
    created_at: str
    session_key: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    session_generation: int | None = None
    unknown_fields: Mapping[str, Any] = field(default_factory=dict)
    raw_fields: Mapping[str, Any] = field(default_factory=dict)
    raw_bytes: bytes = b""

    @property
    def memory_update(self) -> str | None:
        """Compatibility view of the canonical memory output."""

        return self.memory_output

    @property
    def memory_base_digest(self) -> str:
        """Compatibility view of the canonical base digest."""

        return self.base_memory_digest

    @property
    def memory_output_digest(self) -> str | None:
        """Compatibility view of the canonical output digest."""

        return self.output_memory_digest

    @property
    def generation(self) -> int | None:
        """Compatibility view of the canonical session generation."""

        return self.session_generation

    @property
    def raw(self) -> dict[str, Any]:
        """Return a detached copy of all parsed fields, including aliases."""

        return copy.deepcopy(dict(self.raw_fields))


LegacyMemoryReceipt = MemoryReceipt


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_non_json_number(value: str) -> None:
    raise ValueError


def _invalid() -> MemoryReceiptValidationError:
    """Build the only parser error, with no receipt-derived detail."""

    return MemoryReceiptValidationError()


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _is_bounded_counter(value: Any) -> bool:
    return type(value) is int and 0 <= value <= _MAX_COUNTER


def _nonblank_text(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


def _decode(raw_bytes: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_number,
        )
    except (UnicodeDecodeError, _DuplicateKeyError, TypeError, ValueError, json.JSONDecodeError):
        raise _invalid() from None
    if type(decoded) is not dict:
        raise _invalid()
    return decoded


def _check_aliases(payload: dict[str, Any]) -> None:
    pairs = (
        ("memory_output", "memory_update"),
        ("base_memory_digest", "memory_base_digest"),
        ("output_memory_digest", "memory_output_digest"),
        ("session_generation", "generation"),
    )
    for canonical, alias in pairs:
        if alias in payload and canonical in payload:
            if (
                type(payload[alias]) is not type(payload[canonical])
                or payload[alias] != payload[canonical]
            ):
                raise _invalid()


def _validate_payload(payload: dict[str, Any], raw_bytes: bytes) -> MemoryReceipt:
    if not _REQUIRED_FIELDS.issubset(payload):
        raise _invalid()
    _check_aliases(payload)

    if type(payload["version"]) is not int or payload["version"] != 1:
        raise _invalid()
    if not _is_sha256(payload["chunk_digest"]):
        raise _invalid()
    if not _nonblank_text(payload["history_marker_id"]):
        raise _invalid()
    if type(payload["history_entry"]) is not str or not payload["history_entry"].strip():
        raise _invalid()
    if not _is_sha256(payload["base_memory_digest"]):
        raise _invalid()
    if type(payload["reference_only"]) is not bool:
        raise _invalid()
    memory_output = payload["memory_output"]
    output_memory_digest = payload["output_memory_digest"]
    if memory_output is None:
        if not payload["reference_only"] or output_memory_digest is not None:
            raise _invalid()
    else:
        if type(memory_output) is not str or not _is_sha256(output_memory_digest):
            raise _invalid()
        try:
            output_digest = hashlib.sha256(memory_output.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            raise _invalid() from None
        if output_memory_digest != output_digest:
            raise _invalid()
    if not _nonblank_text(payload["created_at"]):
        raise _invalid()

    for name in ("start_offset", "end_offset", "session_generation"):
        if name in payload and not _is_bounded_counter(payload[name]):
            raise _invalid()
    if "generation" in payload and not _is_bounded_counter(payload["generation"]):
        raise _invalid()
    if (
        "start_offset" in payload
        and "end_offset" in payload
        and payload["start_offset"] > payload["end_offset"]
    ):
        raise _invalid()

    session_key = payload.get("session_key")
    if "session_key" in payload and not _nonblank_text(session_key):
        raise _invalid()

    session_generation = payload.get("session_generation", payload.get("generation"))
    if (
        "session_generation" in payload
        and "generation" in payload
        and payload["session_generation"] != payload["generation"]
    ):
        raise _invalid()

    unknown = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key not in _KNOWN_FIELDS
    }
    return MemoryReceipt(
        version=payload["version"],
        chunk_digest=payload["chunk_digest"],
        history_marker_id=payload["history_marker_id"],
        history_entry=payload["history_entry"],
        memory_output=memory_output,
        base_memory_digest=payload["base_memory_digest"],
        output_memory_digest=output_memory_digest,
        reference_only=payload["reference_only"],
        created_at=payload["created_at"],
        session_key=session_key,
        start_offset=payload.get("start_offset"),
        end_offset=payload.get("end_offset"),
        session_generation=session_generation,
        unknown_fields=MappingProxyType(unknown),
        raw_fields=MappingProxyType(copy.deepcopy(payload)),
        raw_bytes=bytes(raw_bytes),
    )


def _canonical_receipt_path(path: Path) -> Path:
    """Resolve only the workspace alias, never the receipt or memory leaf."""

    try:
        workspace = path.parent.parent.resolve(strict=False)
    except (OSError, RuntimeError):
        raise _invalid() from None
    return workspace / path.parent.name / path.name


def _descriptor_flags() -> tuple[int, int] | None:
    """Return directory/leaf no-follow flags when this platform supports them."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        return None
    if not isinstance(directory, int) or directory <= 0:
        return None
    return nofollow, directory


def _close_descriptor(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _same_binding(first: os.stat_result, second: os.stat_result) -> bool:
    """Compare the physical object identity and type of two stat results."""

    return (
        first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and stat.S_IFMT(first.st_mode) == stat.S_IFMT(second.st_mode)
    )


def _same_snapshot(first: os.stat_result, second: os.stat_result) -> bool:
    """Compare one opened file snapshot with a later snapshot."""

    if not _same_binding(first, second) or first.st_size != second.st_size:
        return False
    for name in ("st_mtime_ns", "st_ctime_ns"):
        first_time = getattr(first, name, None)
        second_time = getattr(second, name, None)
        if first_time is not None and second_time is not None and first_time != second_time:
            return False
    return True


def _fallback_missing_or_ambiguous(path: Path) -> bytes | None:
    """Inspect absence when no-follow descriptors are unavailable.

    Existing entries cannot be safely read on this platform, so they fail
    closed. Missing components retain the historical constructor behavior.
    """

    canonical = _canonical_receipt_path(path)
    parts = canonical.parts
    current = Path(parts[0])
    for index, component in enumerate(parts[1:], start=1):
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            return None
        except NotADirectoryError:
            return None
        except OSError:
            raise _invalid() from None
        if index < len(parts) - 1:
            if stat.S_ISLNK(info.st_mode):
                raise _invalid()
            if not stat.S_ISDIR(info.st_mode):
                return None
        else:
            raise _invalid()
    return None


def _validate_receipt_stat(info: os.stat_result, expected_uid: int) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != expected_uid
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o077
        or not stat.S_IMODE(info.st_mode) & stat.S_IRUSR
    ):
        raise _invalid()


def _read_owned_receipt(path: Path) -> bytes | None:
    """Read one receipt through descriptor-relative, no-follow traversal."""

    canonical = _canonical_receipt_path(path)
    capabilities = _descriptor_flags()
    if capabilities is None:
        return _fallback_missing_or_ambiguous(path)
    nofollow, directory = capabilities
    nonblock = getattr(os, "O_NONBLOCK", None)
    parts = canonical.parts
    if len(parts) < 2:
        raise _invalid()

    directory_flags = os.O_RDONLY | directory | nofollow
    if isinstance(getattr(os, "O_CLOEXEC", None), int):
        directory_flags |= os.O_CLOEXEC
    try:
        parent_fd = os.open(parts[0], directory_flags)
    except OSError:
        raise _invalid() from None

    try:
        for component in parts[1:-1]:
            try:
                expected = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None
            except OSError:
                raise _invalid() from None
            if stat.S_ISLNK(expected.st_mode):
                raise _invalid()
            if not stat.S_ISDIR(expected.st_mode):
                return None
            try:
                child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                # The entry was observed before open; disappearance is not
                # absence and must never become a constructor pass-through.
                raise _invalid() from None
            except OSError:
                raise _invalid() from None
            try:
                opened = os.fstat(child_fd)
                if not _same_binding(expected, opened) or not stat.S_ISDIR(opened.st_mode):
                    raise _invalid()
            except BaseException:
                _close_descriptor(child_fd)
                raise
            _close_descriptor(parent_fd)
            parent_fd = child_fd

        leaf = parts[-1]
        try:
            expected = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError:
            raise _invalid() from None
        if stat.S_ISLNK(expected.st_mode):
            raise _invalid()
        try:
            expected_uid = os.getuid()
        except AttributeError:
            raise _invalid() from None
        _validate_receipt_stat(expected, expected_uid)
        if not isinstance(nonblock, int) or nonblock <= 0:
            raise _invalid()

        leaf_flags = os.O_RDONLY | nofollow | nonblock
        descriptor = -1
        try:
            try:
                descriptor = os.open(leaf, leaf_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                raise _invalid() from None
            except OSError:
                raise _invalid() from None
            opened = os.fstat(descriptor)
            _validate_receipt_stat(opened, expected_uid)
            if not _same_binding(expected, opened):
                raise _invalid()
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after_read = os.fstat(descriptor)
            if not _same_snapshot(opened, after_read):
                raise _invalid()
            final = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            _validate_receipt_stat(final, expected_uid)
            if not _same_snapshot(opened, final):
                raise _invalid()
            return b"".join(chunks)
        finally:
            if descriptor >= 0:
                _close_descriptor(descriptor)
    finally:
        _close_descriptor(parent_fd)


def parse_memory_receipt(path: Path) -> MemoryReceipt | None:
    """Parse a pending receipt without changing the file or its directory.

    ``None`` is returned only when the receipt entry is absent.  Any existing
    entry that is unsafe, malformed, or insufficiently evidenced raises the
    fixed-message validation exception.
    """

    try:
        receipt_path = Path(path)
        raw_bytes = _read_owned_receipt(receipt_path)
        if raw_bytes is None:
            return None
        payload = _decode(raw_bytes)
        return _validate_payload(payload, raw_bytes)
    except MemoryReceiptRecoveryRequiredError:
        raise
    except Exception:
        # JSON recursion, detachment failures, and unexpected read/type
        # failures all remain content-free and fail closed.
        raise _invalid() from None


def preflight_memory_receipt(path: Path) -> MemoryReceipt | None:
    """Read and validate a receipt for a constructor preflight."""

    return parse_memory_receipt(path)


def require_memory_receipt_recovery(path: Path) -> None:
    """Raise when a valid or invalid pending receipt blocks startup."""

    try:
        receipt = parse_memory_receipt(path)
    except MemoryReceiptRecoveryRequiredError:
        raise MemoryReceiptRecoveryRequiredError() from None
    if receipt is not None:
        raise MemoryReceiptRecoveryRequiredError()


# Descriptive compatibility spellings for callers that name the staging
# format explicitly.  They all use the same parser and do not add recovery
# behavior.
parse_legacy_memory_receipt = parse_memory_receipt
parse_pending_memory_receipt = parse_memory_receipt
preflight_pending_memory_receipt = preflight_memory_receipt


__all__ = [
    "MEMORY_RECEIPT_FILENAME",
    "MemoryReceipt",
    "MemoryReceiptError",
    "MemoryReceiptRecoveryRequired",
    "MemoryReceiptRecoveryRequiredError",
    "MemoryReceiptValidationError",
    "MemoryRecoveryRequired",
    "MemoryRecoveryRequiredError",
    "PENDING_MEMORY_RECEIPT_FILENAME",
    "LegacyMemoryReceipt",
    "parse_legacy_memory_receipt",
    "parse_memory_receipt",
    "parse_pending_memory_receipt",
    "preflight_memory_receipt",
    "preflight_pending_memory_receipt",
    "require_memory_receipt_recovery",
]
