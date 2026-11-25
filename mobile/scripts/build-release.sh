#!/usr/bin/env bash
# Build, sign and verify the Android release APK without committing key material.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
# shellcheck source=mobile-env.sh
source "${script_dir}/mobile-env.sh"
mobile_require_toolchain

unsigned="${MOBILE_ROOT}/android/app/build/outputs/apk/release/app-release-unsigned.apk"

(
  unset KOHAKU_MOBILE_RELEASE_ENV KOHAKU_MOBILE_KEYSTORE KOHAKU_MOBILE_KEY_ALIAS
  unset KOHAKU_MOBILE_KEYSTORE_PASSWORD KOHAKU_MOBILE_KEYSTORE_PASSWORD_FILE KOHAKU_MOBILE_KEY_PASSWORD
  export KOHAKU_MOBILE_WEB_DIR="www-shared"
  "${script_dir}/verify-gradle-archive.sh"
  "${script_dir}/build-shared-webui.sh"
  cd "${MOBILE_ROOT}"
  bunx cap sync android
  cd "${MOBILE_ROOT}/android"
  rm -f -- "${unsigned}"
  ./gradlew --no-daemon :app:assembleRelease -PkohakuUnsignedRelease=true
)

if [[ ! -s "${unsigned}" ]]; then
  echo "unsigned release APK was not produced: ${unsigned}" >&2
  exit 1
fi

artifact_dir="${KOHAKU_MOBILE_ARTIFACT_DIR:-${repo_root}/mobile/evidence/$(date +%Y%m%d)-m22-release}"
mkdir -p "${artifact_dir}"
build_tools="${ANDROID_SDK_ROOT}/build-tools/36.0.0"
apk="${artifact_dir}/kohaku-mobile-release.apk"
aligned="$(mktemp "${artifact_dir}/.aligned.XXXXXX.apk")"
trap 'rm -f -- "${aligned}"' EXIT
"${build_tools}/zipalign" -f -p 4 "${unsigned}" "${aligned}"

(
  release_env="${KOHAKU_MOBILE_RELEASE_ENV:-${HOME}/.config/kohaku-mobile/release.env}"
  if [[ -f "${release_env}" ]]; then
    if [[ "$(stat -c '%a' "${release_env}")" != "600" ]]; then
      echo "release environment file must have mode 600" >&2
      exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    source "${release_env}"
    set +a
  fi
  keystore="${KOHAKU_MOBILE_KEYSTORE:-${HOME}/.config/kohaku-mobile/release.keystore}"
  if [[ ! -f "${keystore}" || "$(stat -c '%a' "${keystore}")" != "600" ]]; then
    echo "release keystore is missing or its mode differs from 600" >&2
    exit 1
  fi
  if [[ -z "${KOHAKU_MOBILE_KEY_ALIAS:-}" ]]; then
    echo "KOHAKU_MOBILE_KEY_ALIAS is required" >&2
    exit 1
  fi
  if [[ -n "${KOHAKU_MOBILE_KEYSTORE_PASSWORD:-}" ]]; then
    password="env:KOHAKU_MOBILE_KEYSTORE_PASSWORD"
  elif [[ -n "${KOHAKU_MOBILE_KEYSTORE_PASSWORD_FILE:-}" ]]; then
    password="file:${KOHAKU_MOBILE_KEYSTORE_PASSWORD_FILE}"
  else
    echo "set KOHAKU_MOBILE_KEYSTORE_PASSWORD or KOHAKU_MOBILE_KEYSTORE_PASSWORD_FILE" >&2
    exit 1
  fi
  key_password=()
  if [[ -n "${KOHAKU_MOBILE_KEY_PASSWORD:-}" ]]; then
    key_password=(--key-pass env:KOHAKU_MOBILE_KEY_PASSWORD)
  fi
  "${build_tools}/apksigner" sign --ks "${keystore}" --ks-key-alias "${KOHAKU_MOBILE_KEY_ALIAS}" \
    --ks-pass "${password}" "${key_password[@]}" --out "${apk}" "${aligned}"
)

"${build_tools}/zipalign" -c -v 4 "${apk}" > "${artifact_dir}/zipalign.txt"
"${build_tools}/apksigner" verify --verbose --print-certs "${apk}" > "${artifact_dir}/apksigner.txt"
sha256sum "${apk}" | tee "${artifact_dir}/sha256.txt"
echo "release APK: ${artifact_dir}/kohaku-mobile-release.apk"
echo "verification: ${artifact_dir}/apksigner.txt"
