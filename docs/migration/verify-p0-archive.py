#!/usr/bin/env python3
"""Verify a reusable P0 archive has exactly the pinned staging source."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

LEGACY_RUNTIME_DIRS = frozenset(
    {
        "runtime-home",
        "runtime-home-deploy",
        "runtime-home-redaction",
        "runtime-home-safe-push-file",
        "runtime-xdg-cache",
        "runtime-xdg-cache-deploy",
        "runtime-xdg-cache-redaction",
        "runtime-xdg-cache-safe-push-file",
        "runtime-xdg-config",
        "runtime-xdg-config-deploy",
        "runtime-xdg-config-redaction",
        "runtime-xdg-config-safe-push-file",
        "runtime-xdg-data",
        "runtime-xdg-data-deploy",
        "runtime-xdg-data-redaction",
        "runtime-xdg-data-safe-push-file",
    }
)
VENV_ROOTS = frozenset({".venv", ".venv313"})
GENERATED_STATE_ROOTS = frozenset(
    {*VENV_ROOTS, ".pytest_cache", "evidence", *LEGACY_RUNTIME_DIRS}
)


def tree_entries(git: str, commit: str) -> list[tuple[str, str, str]]:
    output = subprocess.check_output([git, "ls-tree", "-r", "-z", commit])
    entries: list[tuple[str, str, str]] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split()
        if kind != "blob":
            raise RuntimeError(f"unexpected non-blob tree entry: {raw_path!r}")
        entries.append((mode, object_id, raw_path.decode("utf-8")))
    return entries


def expected_blob(git: str, object_id: str) -> bytes:
    return subprocess.check_output([git, "cat-file", "blob", object_id])


def tracked_directories(tracked_paths: set[str]) -> set[str]:
    directories: set[str] = set()
    for tracked_path in tracked_paths:
        parent = Path(tracked_path).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def unexpected_entries(root: Path, tracked_paths: set[str]) -> list[str]:
    """Reject non-source entries except the documented local generated state."""

    issues: list[str] = []
    source_directories = tracked_directories(tracked_paths)

    def scan(directory: Path, relative_directory: Path) -> None:
        for child in directory.iterdir():
            relative = relative_directory / child.name
            relative_path = relative.as_posix()
            generated_root = relative.parts[0]

            if child.is_symlink():
                if generated_root in VENV_ROOTS:
                    continue
                issues.append(
                    f"unexpected archive symlink outside allowed virtualenv: {relative_path}"
                )
                continue

            if relative_path in tracked_paths:
                continue
            if relative_path in source_directories:
                if child.is_dir():
                    scan(child, relative)
                else:
                    issues.append(f"expected source directory is not a directory: {relative_path}")
                continue

            if generated_root not in GENERATED_STATE_ROOTS:
                issues.append(f"unexpected archive entry: {relative_path}")
                continue
            if len(relative.parts) == 1 and not child.is_dir():
                issues.append(f"generated-state root is not a directory: {relative_path}")
                continue
            if generated_root in VENV_ROOTS:
                continue
            if child.is_dir():
                scan(child, relative)

    scan(root, Path("."))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    root = args.archive.resolve()
    issues: list[str] = []
    manifest = hashlib.sha256()
    try:
        entries = tree_entries(args.git, args.commit)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        args.report.write_text(
            f"commit={args.commit}\narchive={root}\nresult=fail\nissue=git-tree: {error}\n",
            encoding="utf-8",
        )
        return 1

    root_git = (root / ".git").exists() or (root / ".git").is_symlink()
    if root_git:
        issues.append("archive root contains .git")

    tracked_paths = {path for _, _, path in entries}
    for _mode, object_id, relative_path in entries:
        candidate = root / relative_path
        if not candidate.is_file() or candidate.is_symlink():
            issues.append(f"tracked path is not a regular file: {relative_path}")
            continue
        actual = candidate.read_bytes()
        expected = expected_blob(args.git, object_id)
        actual_sha = hashlib.sha256(actual).hexdigest()
        expected_sha = hashlib.sha256(expected).hexdigest()
        manifest.update(relative_path.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(actual_sha.encode("ascii"))
        manifest.update(b"\n")
        if actual != expected:
            issues.append(
                f"tracked content differs: {relative_path} expected_sha256={expected_sha} "
                f"actual_sha256={actual_sha}"
            )

    issues.extend(unexpected_entries(root, tracked_paths))

    lines = [
        f"commit={args.commit}",
        f"archive={root}",
        f"tracked_file_count={len(entries)}",
        f"source_manifest_sha256={manifest.hexdigest()}",
        "root_git=present" if root_git else "root_git=absent",
        "generated_state_allowlist=" + ",".join(sorted(GENERATED_STATE_ROOTS)),
    ]
    if issues:
        lines.append("result=fail")
        lines.extend(f"issue={issue}" for issue in issues)
    else:
        lines.append("result=pass")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
