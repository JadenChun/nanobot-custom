#!/usr/bin/env bash
# Build, sign and verify the Android release APK without committing key material.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
# shellcheck source=mobile-env.sh
source "${script_dir}/mobile-env.sh"
mobile_require_toolchain

release_env="${NANOBOT_MOBILE_RELEASE_ENV:-${HOME}/.config/nanobot-mobile/release.env}"
if [[ -f "${release_env}" ]]; then
  if [[ "$(stat -c '%a' "${release_env}")" != "600" ]]; then
    echo "release environment file must be 600: ${release_env}" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${release_env}"
  set +a
fi

keystore="${NANOBOT_MOBILE_KEYSTORE:-${HOME}/.config/nanobot-mobile/release.keystore}"
if [[ ! -f "${keystore}" ]]; then
  echo "release keystore not found: ${keystore}" >&2
  exit 1
fi
if [[ "$(stat -c '%a' "${keystore}")" != "600" ]]; then
  echo "release keystore must be 600: ${keystore}" >&2
  exit 1
fi
if [[ -z "${NANOBOT_MOBILE_KEY_ALIAS:-}" ]]; then
  echo "NANOBOT_MOBILE_KEY_ALIAS is required" >&2
  exit 1
fi
if [[ -z "${NANOBOT_MOBILE_KEYSTORE_PASSWORD:-}" && -z "${NANOBOT_MOBILE_KEYSTORE_PASSWORD_FILE:-}" ]]; then
  echo "set NANOBOT_MOBILE_KEYSTORE_PASSWORD or NANOBOT_MOBILE_KEYSTORE_PASSWORD_FILE" >&2
  exit 1
fi

export NANOBOT_MOBILE_KEYSTORE="${keystore}"
export NANOBOT_MOBILE_WEB_DIR="www-shared"

"${script_dir}/build-native-network.sh"
"${script_dir}/build-shared-webui.sh"

(
  cd "${MOBILE_ROOT}"
  bunx cap sync android
)

(
  cd "${MOBILE_ROOT}/android"
  ./gradlew --no-daemon :app:assembleRelease
)

apk="${MOBILE_ROOT}/android/app/build/outputs/apk/release/app-release.apk"
if [[ ! -f "${apk}" ]]; then
  echo "release APK was not produced: ${apk}" >&2
  exit 1
fi

artifact_dir="${NANOBOT_MOBILE_ARTIFACT_DIR:-${repo_root}/mobile/evidence/$(date +%Y%m%d)-m22-release}"
mkdir -p "${artifact_dir}"
cp "${apk}" "${artifact_dir}/nanobot-mobile-release.apk"

build_tools="${ANDROID_SDK_ROOT}/build-tools/36.0.0"
"${build_tools}/zipalign" -c -v 4 "${apk}" > "${artifact_dir}/zipalign.txt"
"${build_tools}/apksigner" verify --verbose --print-certs "${apk}" > "${artifact_dir}/apksigner.txt"
sha256sum "${apk}" | tee "${artifact_dir}/sha256.txt"
echo "release APK: ${artifact_dir}/nanobot-mobile-release.apk"
echo "verification: ${artifact_dir}/apksigner.txt"
