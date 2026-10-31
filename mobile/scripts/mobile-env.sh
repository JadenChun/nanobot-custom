#!/usr/bin/env bash
# Source this file from the mobile/ directory or repository root.
# It keeps every mobile build input inside this checkout's ignored toolchain/.

set -euo pipefail

_mobile_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MOBILE_ROOT="$(cd -- "${_mobile_script_dir}/.." && pwd)"
export MOBILE_TOOLCHAIN="${MOBILE_ROOT}/toolchain"

export JAVA_HOME="${MOBILE_TOOLCHAIN}/jdk21"
export ANDROID_SDK_ROOT="${MOBILE_TOOLCHAIN}/android-sdk"
export ANDROID_HOME="${ANDROID_SDK_ROOT}"
export ANDROID_AVD_HOME="${MOBILE_TOOLCHAIN}/avd"
export ANDROID_USER_HOME="${MOBILE_TOOLCHAIN}/android-user"
export GRADLE_USER_HOME="${MOBILE_TOOLCHAIN}/gradle"
export GOCACHE="${MOBILE_TOOLCHAIN}/go-cache"
export GOMODCACHE="${MOBILE_TOOLCHAIN}/go-mod-cache"
export GOPATH="${MOBILE_TOOLCHAIN}/go-path"

export PATH="${JAVA_HOME}/bin:${MOBILE_TOOLCHAIN}/go/bin:${ANDROID_SDK_ROOT}/platform-tools:${ANDROID_SDK_ROOT}/emulator:${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin:${PATH}"

mkdir -p "${ANDROID_AVD_HOME}" "${ANDROID_USER_HOME}" "${GRADLE_USER_HOME}" \
  "${GOCACHE}" "${GOMODCACHE}" "${GOPATH}"

mobile_require_toolchain() {
  local missing=0
  for entry in \
    "${JAVA_HOME}/bin/javac" \
    "${MOBILE_TOOLCHAIN}/go/bin/go" \
    "${ANDROID_SDK_ROOT}/platform-tools/adb" \
    "${ANDROID_SDK_ROOT}/emulator/emulator"; do
    if [[ ! -x "${entry}" ]]; then
      echo "missing mobile toolchain entry: ${entry}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    echo "run mobile/scripts/bootstrap-toolchain.sh first" >&2
    return 1
  fi
}
