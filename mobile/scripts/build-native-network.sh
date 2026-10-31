#!/usr/bin/env bash
# Build the Go native transport for Android ABIs and retain its ABI evidence.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mobile-env.sh
source "${script_dir}/mobile-env.sh"
mobile_require_toolchain

NDK_VERSION="${NDK_VERSION:-29.0.14206865}"
NDK_BIN="${ANDROID_SDK_ROOT}/ndk/${NDK_VERSION}/toolchains/llvm/prebuilt/linux-x86_64/bin"
MIN_SDK="${MIN_SDK:-24}"
NETWORK_ROOT="${MOBILE_ROOT}/native-network"
JNI_ROOT="${MOBILE_ROOT}/android/app/src/main/jniLibs"

(
  cd "${NETWORK_ROOT}"
  "${MOBILE_TOOLCHAIN}/go/bin/go" test ./...
  "${MOBILE_TOOLCHAIN}/go/bin/go" vet ./...
)

build_abi() {
  local abi="$1" goarch="$2" triple="$3"
  local cc="${NDK_BIN}/${triple}-linux-android${MIN_SDK}-clang"
  if [[ ! -x "${cc}" ]]; then
    echo "NDK compiler missing: ${cc}" >&2
    exit 1
  fi
  mkdir -p "${JNI_ROOT}/${abi}"
  echo "building native-network ${abi} (${triple}-linux-android${MIN_SDK})"
  (
    cd "${NETWORK_ROOT}"
    CGO_ENABLED=1 GOOS=android GOARCH="${goarch}" CC="${cc}" \
      "${MOBILE_TOOLCHAIN}/go/bin/go" build \
      -trimpath -buildmode=c-shared -ldflags='-s -w -buildid=' \
      -o "${JNI_ROOT}/${abi}/libnanobot_network.so" ./cmd/lib
  )
  local readelf="${NDK_BIN}/llvm-readelf"
  "${readelf}" -h "${JNI_ROOT}/${abi}/libnanobot_network.so" | grep -E 'Class:|Machine:'
  "${readelf}" -l "${JNI_ROOT}/${abi}/libnanobot_network.so" | grep 'LOAD' | head -1
  "${readelf}" --dyn-syms "${JNI_ROOT}/${abi}/libnanobot_network.so" | grep ' nanobot_network_' | head -12
}

build_abi x86_64 amd64 x86_64
build_abi arm64-v8a arm64 aarch64

sha256sum "${JNI_ROOT}"/*/libnanobot_network.so
