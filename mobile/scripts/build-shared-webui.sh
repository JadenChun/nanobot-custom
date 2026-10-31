#!/usr/bin/env bash
# Package the shared WebUI assets for the mobile shell without touching the
# browser/wheel output or the M01 fixture bundle.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mobile_root="$(cd -- "${script_dir}/.." && pwd)"
repo_root="$(cd -- "${mobile_root}/.." && pwd)"
output="${mobile_root}/www-shared"

cd "${repo_root}/webui"
bun install --frozen-lockfile
bun run build
rm -rf "${output}"
cp -a "${repo_root}/nanobot/web/dist" "${output}"
rm -f "${output}/sw.js" "${output}/sw.js.map"
find "${output}" -name "*.gz" -delete
cp "${mobile_root}/mobile-bootstrap.js" "${output}/mobile-bootstrap.js"
python3 - "${output}/index.html" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
tag = '<script src="./mobile-bootstrap.js"></script>'
if tag not in text:
    marker = '<script type="module"'
    if marker not in text:
        raise SystemExit("shared index has no module script")
    text = text.replace(marker, tag + "\n    " + marker, 1)
    path.write_text(text, encoding="utf-8")
PY
echo "shared mobile web assets: ${output}"
find "${output}" -maxdepth 1 -type f -printf '%f\n' | sort
