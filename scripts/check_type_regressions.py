"""Compare full BasedPyright JSON reports without waiving the strict release gate.

This is a diagnostic ratchet, not a test runner or environment attestation. Run both
checks with the same frozen dependencies and retain their source/command manifests.
Line movement is ignored; file, severity, rule, message and multiplicity are not.

An analyzed-file decrease is accepted only with complete before/after in-scope
source inventories and an explicit JSON list of intentionally deleted paths. The
inventories must match the checker counts and the current inventory must match
the actual remaining in-scope sources. Without that evidence the comparison
fails closed.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath

Signature = tuple[str, str, str, str]


_DIGEST = re.compile(r"[0-9a-f]{64}")


def _relative_inventory_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is not a non-empty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or not all(part for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} path is not normalized and relative")
    return value


def load_inventory(path: Path) -> dict[str, str]:
    """Load a normalized relative-path to SHA-256 inventory."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("source inventory is empty or invalid")
    inventory: dict[str, str] = {}
    for raw_path, raw_digest in data.items():
        relative = _relative_inventory_path(raw_path, label="inventory")
        if relative in inventory:
            raise ValueError("source inventory contains a duplicate path")
        if not isinstance(raw_digest, str) or not _DIGEST.fullmatch(raw_digest):
            raise ValueError("source inventory contains an invalid SHA-256 digest")
        inventory[relative] = raw_digest
    return inventory


def load_deletion_manifest(path: Path) -> frozenset[str]:
    """Load an explicit JSON array of intentionally deleted in-scope paths."""
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("deletion manifest is empty or invalid")
    deleted: set[str] = set()
    for raw_path in data:
        relative = _relative_inventory_path(raw_path, label="deletion manifest")
        if relative in deleted:
            raise ValueError("deletion manifest contains a duplicate path")
        deleted.add(relative)
    return frozenset(deleted)


def _excluded(relative: str, patterns: list[str]) -> bool:
    parts = PurePosixPath(relative).parts
    candidates = ["/".join(parts[:index]) for index in range(1, len(parts))]
    return any(
        fnmatch.fnmatch(relative, pattern)
        or any(fnmatch.fnmatch(candidate, pattern) for candidate in candidates)
        for pattern in patterns
    )


def _included(relative: str, patterns: list[str]) -> bool:
    path = PurePosixPath(relative)
    for pattern in patterns:
        if fnmatch.fnmatch(relative, pattern):
            return True
        if not any(character in pattern for character in "*?[]"):
            prefix = pattern.rstrip("/")
            if relative == prefix or relative.startswith(prefix + "/"):
                return True
        if path.match(pattern):
            return True
    return False


def source_inventory(root: Path, policy: dict[str, object]) -> dict[str, str]:
    """Hash the actual Python sources selected by the frozen checker policy."""
    include = policy.get("include", ["."])
    exclude = policy.get("exclude", [])
    if (
        not isinstance(include, list)
        or not include
        or not all(isinstance(value, str) and value for value in include)
        or not isinstance(exclude, list)
        or not all(isinstance(value, str) and value for value in exclude)
    ):
        raise ValueError("checker include/exclude policy is invalid")
    inventory: dict[str, str] = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if not _included(relative, include) or _excluded(relative, exclude):
            continue
        inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not inventory:
        raise ValueError("source inventory is empty")
    return inventory


def _validate_deletion_evidence(
    *,
    old_files: int,
    files: int,
    before: dict[str, str],
    after: dict[str, str],
    deleted: frozenset[str],
    current_root: Path,
    policy: dict[str, object],
) -> None:
    if old_files != len(before):
        raise ValueError("baseline inventory count does not match analyzed file count")
    if files != len(after):
        raise ValueError("current inventory count does not match analyzed file count")
    disappeared = frozenset(before) - frozenset(after)
    if disappeared != deleted:
        missing = sorted(disappeared - deleted)
        stale = sorted(deleted - disappeared)
        raise ValueError(
            "deletion manifest does not exactly match inventory disappearance: "
            f"undeclared={missing[:5]} still-present={stale[:5]}"
        )
    actual = source_inventory(current_root, policy)
    if actual != after:
        changed = sorted(
            path
            for path in set(actual) | set(after)
            if actual.get(path) != after.get(path)
        )
        raise ValueError(
            "current inventory does not match actual remaining in-scope sources: "
            f"first differences={changed[:5]}"
        )


def load_report(path: Path, root: Path) -> tuple[str, int, Counter[Signature]]:
    report = json.loads(path.read_text())
    version = report["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("missing checker version")
    summary = report["summary"]
    files = summary["filesAnalyzed"]
    if type(files) is not int or files <= 0:
        raise ValueError("empty or invalid analyzed scope")
    diagnostics = report["generalDiagnostics"]
    if not isinstance(diagnostics, list):
        raise ValueError("invalid diagnostics")
    counts: Counter[Signature] = Counter()
    severities: Counter[str] = Counter()
    for diagnostic in diagnostics:
        # Pure lexical containment: do not follow links into another checkout.
        file = Path(diagnostic["file"])
        if not file.is_absolute() or ".." in file.parts:
            raise ValueError("diagnostic path is not absolute and normalized")
        relative = file.relative_to(root).as_posix()
        severity = diagnostic["severity"]
        rule = diagnostic.get("rule", "")
        message = diagnostic["message"]
        if severity not in ("error", "warning", "information"):
            raise ValueError("invalid diagnostic severity")
        if not isinstance(rule, str) or not isinstance(message, str):
            raise ValueError("invalid diagnostic signature")
        counts[(relative, severity, rule, message)] += 1
        severities[severity] += 1
    for severity in ("error", "warning", "information"):
        count = summary[severity + "Count"]
        if type(count) is not int or count != severities[severity]:
            raise ValueError("summary does not match diagnostic records")
    return version, files, counts


def compare(
    baseline: Path,
    current: Path,
    baseline_root: Path,
    current_root: Path,
    *,
    before_inventory: Path | None = None,
    after_inventory: Path | None = None,
    deletion_manifest: Path | None = None,
) -> dict:
    for root in (baseline_root, current_root):
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("source roots must be absolute and normalized")
    policies = [
        tomllib.loads((root / "pyproject.toml").read_text())["tool"]["basedpyright"]
        for root in (baseline_root, current_root)
    ]
    if policies[0] != policies[1] or policies[1].get("typeCheckingMode") != "strict":
        raise ValueError("checker policy changed or is not strict")
    old_version, old_files, old = load_report(baseline, baseline_root)
    version, files, new = load_report(current, current_root)
    if version != old_version:
        raise ValueError("checker version changed")
    evidence = (before_inventory, after_inventory, deletion_manifest)
    if any(path is not None for path in evidence) and not all(
        path is not None for path in evidence
    ):
        raise ValueError("deletion evidence requires both inventories and a manifest")
    if files < old_files and not all(path is not None for path in evidence):
        raise ValueError(
            "analyzed file count decreased without complete inventory and deletion evidence"
        )
    if all(path is not None for path in evidence):
        assert before_inventory is not None
        assert after_inventory is not None
        assert deletion_manifest is not None
        _validate_deletion_evidence(
            old_files=old_files,
            files=files,
            before=load_inventory(before_inventory),
            after=load_inventory(after_inventory),
            deleted=load_deletion_manifest(deletion_manifest),
            current_root=current_root,
            policy=policies[1],
        )
    added, removed = new - old, old - new
    return {
        "verdict": "FAIL" if added else "PASS_NO_NEW_DIAGNOSTICS",
        "strict_clean": not any(key[1] == "error" for key in new),
        "baseline_diagnostics": sum(old.values()),
        "current_diagnostics": sum(new.values()),
        "introduced": sum(added.values()),
        "resolved": sum(removed.values()),
        "new_diagnostics": [
            dict(file=key[0], severity=key[1], rule=key[2], message=key[3], count=count)
            for key, count in sorted(added.items())
        ],
        "scope": "diagnostic comparison only; frozen environment and test evidence required separately",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--current-root", required=True, type=Path)
    parser.add_argument(
        "--before-inventory",
        type=Path,
        help="JSON map of baseline in-scope relative paths to SHA-256 hashes",
    )
    parser.add_argument(
        "--after-inventory",
        type=Path,
        help="JSON map of current in-scope relative paths to SHA-256 hashes",
    )
    parser.add_argument(
        "--deletion-manifest",
        type=Path,
        help="JSON list of intentionally deleted in-scope relative paths",
    )
    args = parser.parse_args()
    try:
        result = compare(
            args.baseline,
            args.current,
            args.baseline_root,
            args.current_root,
            before_inventory=args.before_inventory,
            after_inventory=args.after_inventory,
            deletion_manifest=args.deletion_manifest,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"verdict": "INVALID_EVIDENCE", "error": str(error)}))
        return 2
    print(json.dumps(result, indent=2))
    return int(result["introduced"] != 0)


if __name__ == "__main__":
    sys.exit(main())
