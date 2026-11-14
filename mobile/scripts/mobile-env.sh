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

export PATH="${MOBILE_TOOLCHAIN}/bun/bin:${JAVA_HOME}/bin:${MOBILE_TOOLCHAIN}/go/bin:${MOBILE_TOOLCHAIN}/uv:${ANDROID_SDK_ROOT}/platform-tools:${ANDROID_SDK_ROOT}/emulator:${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin:${PATH}"

mkdir -p "${ANDROID_AVD_HOME}" "${ANDROID_USER_HOME}" "${GRADLE_USER_HOME}" \
  "${GOCACHE}" "${GOMODCACHE}" "${GOPATH}"

MOBILE_TOOLCHAIN_LOCK="${MOBILE_ROOT}/toolchain.lock.json"

_mobile_lock_value() {
  python3 - "${MOBILE_TOOLCHAIN_LOCK}" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    node = json.load(handle)
for part in sys.argv[2].split("."):
    node = node[part]
print(node)
PY
}

_mobile_check_bun() {
  local pinned cached observed
  pinned="$(_mobile_lock_value app.bun)"
  cached=""
  for entry in "${MOBILE_TOOLCHAIN}/bun/bin/bun" "${MOBILE_TOOLCHAIN}/bin/bun"; do
    if [[ -x "${entry}" ]]; then
      cached="${entry}"
      break
    fi
  done

  if [[ -z "${cached}" ]]; then
    if command -v bun >/dev/null 2>&1; then
      observed="$(bun --version 2>/dev/null | head -n1 | tr -d '[:space:]')"
      if [[ "${observed}" == "${pinned}" ]]; then
        echo "bun: ${pinned} (PATH; checkout toolchain has no cached Bun)"
        export MOBILE_BUN="$(command -v bun)"
        return 0
      fi
      echo "bun: version mismatch: expected ${pinned}, found ${observed:-unknown} (PATH)" >&2
    else
      echo "bun: missing; expected ${pinned} in ${MOBILE_TOOLCHAIN}/bun/bin/bun or on PATH" >&2
    fi
    return 1
  fi

  observed="$("${cached}" --version 2>/dev/null | head -n1 | tr -d '[:space:]')"
  if [[ "${observed}" != "${pinned}" ]]; then
    echo "bun: version mismatch: expected ${pinned}, found ${observed:-unknown} (${cached})" >&2
    return 1
  fi
  echo "bun: ${pinned} (${cached})"
  export MOBILE_BUN="${cached}"
}

_mobile_check_jdk() {
  local pinned base javac observed
  pinned="$(_mobile_lock_value jdk.version)"
  base="${pinned%%+*}"
  javac="${JAVA_HOME}/bin/javac"
  if [[ ! -x "${javac}" ]]; then
    echo "jdk: missing ${javac}; expected ${pinned}" >&2
    return 1
  fi
  observed="$("${javac}" -version 2>&1 | head -n1)"
  observed="${observed##* }"
  if [[ "${observed}" != "${base}" && "${observed}" != "${pinned}" ]]; then
    echo "jdk: version mismatch: expected ${pinned}, found ${observed:-unknown} (${javac})" >&2
    return 1
  fi
  echo "jdk: ${pinned} (${javac})"
}

_mobile_check_go() {
  local pinned go_bin line observed
  pinned="$(_mobile_lock_value go.version)"
  go_bin="${MOBILE_TOOLCHAIN}/go/bin/go"
  if [[ ! -x "${go_bin}" ]]; then
    echo "go: missing ${go_bin}; expected ${pinned}" >&2
    return 1
  fi
  line="$("${go_bin}" version 2>&1 | head -n1)"
  observed="${line#go version go}"
  observed="${observed%% *}"
  if [[ "${observed}" != "${pinned}" ]]; then
    echo "go: version mismatch: expected ${pinned}, found ${observed:-unknown} (${go_bin})" >&2
    return 1
  fi
  echo "go: ${pinned} (${go_bin})"
}

_mobile_check_uv() {
  local pinned uv_bin line observed
  pinned="$(_mobile_lock_value uv.version)"
  uv_bin="${MOBILE_TOOLCHAIN}/uv/uv"
  if [[ ! -x "${uv_bin}" ]]; then
    echo "uv: missing ${uv_bin}; expected ${pinned}" >&2
    return 1
  fi
  line="$("${uv_bin}" --version 2>&1 | head -n1)"
  observed="${line#uv }"
  observed="${observed%% *}"
  if [[ "${observed}" != "${pinned}" ]]; then
    echo "uv: version mismatch: expected ${pinned}, found ${observed:-unknown} (${uv_bin})" >&2
    return 1
  fi
  echo "uv: ${pinned} (${uv_bin})"
}

_mobile_check_adb() {
  local pinned adb line
  pinned="$(_mobile_lock_value android_sdk.platform_tools)"
  adb="${ANDROID_SDK_ROOT}/platform-tools/adb"
  if [[ ! -x "${adb}" ]]; then
    echo "adb: missing ${adb}; expected platform-tools ${pinned}" >&2
    return 1
  fi
  line="$("${adb}" version 2>&1 | head -n2 | tr '\n' ' ')"
  if [[ "${line}" != *"${pinned}"* ]]; then
    echo "adb: version mismatch: expected platform-tools ${pinned}, found ${line:-unknown} (${adb})" >&2
    return 1
  fi
  echo "adb: platform-tools ${pinned} (${adb})"
}

_mobile_check_emulator() {
  local pinned emulator line
  pinned="$(_mobile_lock_value android_sdk.emulator)"
  emulator="${ANDROID_SDK_ROOT}/emulator/emulator"
  if [[ ! -x "${emulator}" ]]; then
    echo "emulator: missing ${emulator}; expected ${pinned}" >&2
    return 1
  fi
  line="$("${emulator}" -version 2>&1 | head -n1)"
  if [[ "${line}" != *"${pinned}"* ]]; then
    echo "emulator: version mismatch: expected ${pinned}, found ${line:-unknown} (${emulator})" >&2
    return 1
  fi
  echo "emulator: ${pinned} (${emulator})"
}

_mobile_android_component() {
  local name="$1" path="$2" expected="$3"
  if [[ -e "${path}" ]]; then
    echo "${name}: ${expected} (${path})"
    return 0
  fi
  echo "${name}: unavailable; expected ${expected} (${path})" >&2
  return 1
}

_mobile_check_cmdline_tools() {
  local pinned target properties revision
  pinned="$(_mobile_lock_value android_cmdline_tools.version)"
  target="${ANDROID_SDK_ROOT}/cmdline-tools/latest"
  if [[ ! -x "${target}/bin/sdkmanager" ]]; then
    echo "android cmdline-tools: missing ${target}/bin/sdkmanager; expected ${pinned}" >&2
    return 1
  fi
  properties="${target}/source.properties"
  if [[ -f "${properties}" ]]; then
    revision="$(grep '^Pkg.Revision=' "${properties}" | head -n1 | cut -d= -f2)"
    if [[ "${revision}" != "${pinned}" ]]; then
      echo "android cmdline-tools: version mismatch: expected ${pinned}, found ${revision:-unknown} (${target})" >&2
      return 1
    fi
  fi
  echo "android cmdline-tools: ${pinned} (${target})"
}

_mobile_check_android_components() {
  local failed=0 build_tools compile_sdk ndk cmake system_image
  build_tools="$(_mobile_lock_value android_sdk.build_tools)"
  compile_sdk="$(_mobile_lock_value android_sdk.compile_sdk)"
  ndk="$(_mobile_lock_value android_sdk.ndk)"
  cmake="$(_mobile_lock_value android_sdk.cmake)"
  system_image="$(_mobile_lock_value android_sdk.system_image)"
  system_image="${system_image//;/\/}"

  _mobile_check_cmdline_tools || failed=1
  _mobile_android_component "android build-tools" \
    "${ANDROID_SDK_ROOT}/build-tools/${build_tools}" "${build_tools}" || failed=1
  _mobile_android_component "android platform" \
    "${ANDROID_SDK_ROOT}/platforms/android-${compile_sdk}" "android-${compile_sdk}" || failed=1
  _mobile_android_component "android ndk" \
    "${ANDROID_SDK_ROOT}/ndk/${ndk}" "${ndk}" || failed=1
  _mobile_android_component "android cmake" \
    "${ANDROID_SDK_ROOT}/cmake/${cmake}" "${cmake}" || failed=1
  _mobile_android_component "android system image" \
    "${ANDROID_SDK_ROOT}/${system_image}" "${system_image}" || failed=1
  return "${failed}"
}

_mobile_check_gradle_archive() {
  local pinned expected actual archive
  pinned="$(_mobile_lock_value gradle.version)"
  expected="$(_mobile_lock_value gradle.sha256)"
  archive="$(find "${GRADLE_USER_HOME}/wrapper/dists" -type f \
    -name "gradle-${pinned}-all.zip" -print -quit 2>/dev/null || true)"
  if [[ -z "${archive}" ]]; then
    echo "gradle archive: unavailable; expected ${pinned} (${expected})" >&2
    return 0
  fi
  actual="$(sha256sum "${archive}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "gradle archive: digest mismatch: expected ${expected}, found ${actual} (${archive})" >&2
    return 1
  fi
  echo "gradle archive: ${pinned} (${archive})"
}

mobile_require_toolchain() {
  local failed=0
  if [[ ! -f "${MOBILE_TOOLCHAIN_LOCK}" ]]; then
    echo "missing mobile toolchain lock: ${MOBILE_TOOLCHAIN_LOCK}" >&2
    return 1
  fi

  _mobile_check_bun || failed=1
  _mobile_check_jdk || failed=1
  _mobile_check_go || failed=1
  _mobile_check_uv || failed=1
  _mobile_check_adb || failed=1
  _mobile_check_emulator || failed=1
  _mobile_check_android_components || failed=1
  _mobile_check_gradle_archive || failed=1

  if [[ "${failed}" -ne 0 ]]; then
    echo "run mobile/scripts/bootstrap-toolchain.sh first and reconcile the versions above with ${MOBILE_TOOLCHAIN_LOCK}" >&2
    return 1
  fi
}
