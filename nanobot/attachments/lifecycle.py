"""Reference-aware retention through the AttachmentStore's live destruction gate."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from nanobot.attachments._inventory import References, rows, walk_files
from nanobot.attachments._media import read_bytes
from nanobot.attachments.store import AttachmentStore


def reference_evidence(session_files: Iterable[Path]) -> tuple[set[str], bool]:
    """Read-only diagnostics; this result is never destruction authority."""
    references = References(Path("/"))
    complete = True
    for path in session_files:
        try:
            references.scan(rows(read_bytes(path)))
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
            complete = False
    return references.hashes, complete


def referenced_hashes(session_files: Iterable[Path]) -> set[str]:
    return reference_evidence(session_files)[0]


@dataclass(frozen=True)
class AttachmentPruneResult:
    removed: list[Path]
    evidence_complete: bool
    reasons: tuple[str, ...]
    retry_required: bool = False


def prune_unreferenced_result(
    store: AttachmentStore,
    sessions_dir: Path,
    *,
    grace_seconds: float = 7 * 24 * 60 * 60,
    dry_run: bool = False,
) -> AttachmentPruneResult:
    """Reclaim only after fresh complete inventory under the store's live fence."""
    removed: list[Path] = []
    failures: list[str] = []
    try:
        with store.custody.maintenance() as (gate, evidence):
            declarations, _raw = store.custody.declarations()
            roots = cast(list[dict[str, object]], declarations["roots"])
            requested = Path(sessions_dir).expanduser().resolve(strict=False)
            if not any(row["kind"] == "sessions" and requested in {
                Path(str(row["path"])), Path(str(row["sessions_root"])),
            } for row in roots):
                return AttachmentPruneResult([], False, ("requested sessions root is not declared",))
            if not evidence.complete:
                return AttachmentPruneResult([], False, evidence.reasons)
            assets = [path for path in walk_files(store.root) if path.suffix != ".json"] if store.root.exists() else []
            for intent in store.custody.records("deletions"):
                path = Path(str(intent["path"]))
                if path.is_relative_to(store.root) and path not in assets:
                    assets.append(path)
            now = time.time()
            for path in assets:
                if path in evidence.paths or path.stem in evidence.hashes or any(pinned == path for _token, pinned in evidence.pending):
                    continue
                try:
                    if path.exists() and now - path.lstat().st_mtime < grace_seconds:
                        continue
                    gate.require(store.custody)
                    if dry_run or store.custody.delete(gate, evidence, path):
                        removed.append(path)
                except OSError as exc:
                    failures.append(f"{path}: {type(exc).__name__}: {exc}")
                    continue
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        return AttachmentPruneResult(removed, False, (f"{type(exc).__name__}: {exc}",), bool(removed))
    return AttachmentPruneResult(removed, True, tuple(failures), bool(failures))


def prune_unreferenced(
    store: AttachmentStore, sessions_dir: Path, *,
    grace_seconds: float = 7 * 24 * 60 * 60, dry_run: bool = False,
) -> list[Path]:
    return prune_unreferenced_result(
        store, sessions_dir, grace_seconds=grace_seconds, dry_run=dry_run,
    ).removed


def migrate_legacy_media(
    sessions_dir: Path, store: AttachmentStore, *, dry_run: bool = False,
) -> None:
    """Compatibility refusal: the first release has no attachment migration."""
    raise RuntimeError("attachments migration is retired; no files were changed")
