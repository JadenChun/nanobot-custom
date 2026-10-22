"""Compare full BasedPyright JSON reports without waiving the strict release gate.

This is a diagnostic ratchet, not a test runner or environment attestation. Run both
checks with the same frozen dependencies and retain their source/command manifests.
Line movement is ignored; file, severity, rule, message and multiplicity are not.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

Signature = tuple[str, str, str, str]


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


def compare(baseline: Path, current: Path, baseline_root: Path, current_root: Path) -> dict:
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
    if files < old_files:
        raise ValueError("analyzed file count decreased; review scope before rebaselining")
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
    args = parser.parse_args()
    try:
        result = compare(args.baseline, args.current, args.baseline_root, args.current_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"verdict": "INVALID_EVIDENCE", "error": str(error)}))
        return 2
    print(json.dumps(result, indent=2))
    return int(result["introduced"] != 0)


if __name__ == "__main__":
    sys.exit(main())
