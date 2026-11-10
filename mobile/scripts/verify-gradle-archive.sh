#!/usr/bin/env bash
# Download and verify the pinned Gradle distribution before any release build.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mobile-env.sh
source "${script_dir}/mobile-env.sh"

manifest="${MOBILE_ROOT}/toolchain.lock.json"
version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gradle"]["version"])' "$manifest")"
url="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gradle"]["url"])' "$manifest")"
expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gradle"]["sha256"])' "$manifest")"

archive_dir="${MOBILE_TOOLCHAIN}/gradle-archive"
archive="${archive_dir}/gradle-${version}-all.zip"
mkdir -p "$archive_dir"
if [[ ! -f "$archive" ]]; then
  echo "gradle: downloading ${url}"
  curl -fsSL -o "${archive}.part" "$url"
  mv "${archive}.part" "$archive"
fi
actual="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$actual" != "$expected" ]]; then
  echo "gradle archive checksum mismatch: ${actual} != ${expected}" >&2
  exit 1
fi
echo "gradle archive verified: ${actual} (${version})"
