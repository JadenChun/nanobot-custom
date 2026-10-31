#!/usr/bin/env bash
# Install or update the signed release in place without clearing app data.
# A signature mismatch fails instead of silently uninstalling the app.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mobile-env.sh
source "${script_dir}/mobile-env.sh"
mobile_require_toolchain

SERIAL="${ANDROID_SERIAL:-${1:-}}"
if [[ -z "${SERIAL}" ]]; then
  SERIAL="$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')"
fi
if [[ -z "${SERIAL}" ]]; then
  echo "no Android device serial" >&2
  exit 2
fi

APK="${2:-${MOBILE_ROOT}/evidence/20261101-m22-release/nanobot-mobile-release.apk}"
if [[ ! -f "${APK}" ]]; then
  echo "release APK not found: ${APK}" >&2
  exit 2
fi

echo "installing ${APK} in place on ${SERIAL}"
adb -s "${SERIAL}" install -r "${APK}"
echo "update complete; pairing and Tailscale state were preserved (same signing key required)"
