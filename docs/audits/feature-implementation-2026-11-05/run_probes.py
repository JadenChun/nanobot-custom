"""Run the saved audit probes with project fixtures; current red results are intentional."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    evidence = Path(__file__).resolve().parent
    repo = evidence.parents[2]
    python = repo / ".venv/bin/python"
    bun = shutil.which("bun")
    if not python.is_file() or bun is None:
        raise SystemExit("Use the existing .venv and install WebUI development dependencies first.")

    output = Path(tempfile.mkdtemp(prefix="nfa-", dir="/tmp"))
    cases = [
        (
            "backend",
            evidence / "backend.probe.py",
            repo / "tests/test_feature_audit_20261105_tmp.py",
            repo,
            [
                str(python), "-m", "pytest", "tests/test_feature_audit_20261105_tmp.py",
                "-q", "--disable-warnings", f"--basetemp={output / 'pytest'}",
            ],
        ),
        (
            "frontend",
            evidence / "frontend.probe.tsx",
            repo / "webui/src/tests/feature-audit-20261105.tmp.test.tsx",
            repo / "webui",
            [bun, "run", "test", "src/tests/feature-audit-20261105.tmp.test.tsx"],
        ),
    ]
    # Check both destinations before creating either probe file.
    if any(case[2].exists() for case in cases):
        raise SystemExit("A temporary probe path already exists; refusing to overwrite it.")

    failed = False
    for name, source, target, cwd, command in cases:
        contents = source.read_text(encoding="utf-8")
        with target.open("x", encoding="utf-8") as handle:
            handle.write(contents)
        try:
            log = output / f"{name}.log"
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
            print(f"{name}: exit {result.returncode}; output: {log}")
            failed |= result.returncode != 0
        finally:
            target.unlink()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
