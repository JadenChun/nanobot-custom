#!/usr/bin/env bash
# Reproducibly install the pinned M01 mobile toolchain inside mobile/toolchain/.
# No system packages, services or user-wide SDK paths are changed.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mobile_root="$(cd -- "${script_dir}/.." && pwd)"
toolchain="${mobile_root}/toolchain"
download_dir="${TMPDIR:-/tmp}/nanobot-mobile-toolchain"
manifest="${mobile_root}/toolchain.lock.json"

sha256_of() { sha256sum "$1" | awk '{print $1}'; }
sha1_of() { sha1sum "$1" | awk '{print $1}'; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command missing: $1" >&2
    exit 1
  }
}

need_cmd curl
need_cmd python3
need_cmd tar
need_cmd unzip
need_cmd sha256sum
need_cmd sha1sum

mkdir -p "$toolchain" "$download_dir"
json() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]][sys.argv[3]])' "$manifest" "$1" "$2"; }

install_jdk() {
  local url sha expected actual
  url="$(json jdk url)"; sha="$(json jdk sha256)"
  if [[ -x "${toolchain}/jdk21/bin/javac" ]]; then
    echo "jdk: already present"
    return
  fi
  echo "jdk: downloading"
  curl -fsSL -o "${download_dir}/jdk21.tar.gz" "$url"
  actual="$(sha256_of "${download_dir}/jdk21.tar.gz")"
  [[ "$actual" == "$sha" ]] || { echo "jdk checksum mismatch: $actual != $sha" >&2; exit 1; }
  mkdir -p "${toolchain}/jdk21"
  tar -xzf "${download_dir}/jdk21.tar.gz" -C "${toolchain}/jdk21" --strip-components=1
  "${toolchain}/jdk21/bin/javac" -version
}

install_go() {
  local url sha actual
  url="$(json go url)"; sha="$(json go sha256)"
  if [[ -x "${toolchain}/go/bin/go" ]]; then
    echo "go: already present"
    return
  fi
  echo "go: downloading"
  curl -fsSL -o "${download_dir}/go.tar.gz" "$url"
  actual="$(sha256_of "${download_dir}/go.tar.gz")"
  [[ "$actual" == "$sha" ]] || { echo "go checksum mismatch: $actual != $sha" >&2; exit 1; }
  mkdir -p "${toolchain}/go"
  tar -xzf "${download_dir}/go.tar.gz" -C "${toolchain}/go" --strip-components=1
  "${toolchain}/go/bin/go" version
}

install_cmdline_tools() {
  local url sha actual
  url="$(json android_cmdline_tools url)"; sha="$(json android_cmdline_tools sha1)"
  local target="${toolchain}/android-sdk/cmdline-tools/latest"
  if [[ -x "${target}/bin/sdkmanager" ]]; then
    echo "android cmdline-tools: already present"
    return
  fi
  echo "android cmdline-tools: downloading"
  curl -fsSL -o "${download_dir}/cmdline-tools.zip" "$url"
  actual="$(sha1_of "${download_dir}/cmdline-tools.zip")"
  [[ "$actual" == "$sha" ]] || { echo "cmdline-tools checksum mismatch: $actual != $sha" >&2; exit 1; }
  rm -rf "${download_dir}/cmdline-tools-extract"
  unzip -q -o "${download_dir}/cmdline-tools.zip" -d "${download_dir}/cmdline-tools-extract"
  mkdir -p "${toolchain}/android-sdk/cmdline-tools"
  rm -rf "${target}"
  mv "${download_dir}/cmdline-tools-extract/cmdline-tools" "${target}"
}

install_android_sdk() {
  local sdk="${toolchain}/android-sdk"
  local image build_tools ndk cmake
  image="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["android_sdk"]["system_image"])' "$manifest")"
  build_tools="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["android_sdk"]["build_tools"])' "$manifest")"
  ndk="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["android_sdk"]["ndk"])' "$manifest")"
  cmake="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["android_sdk"]["cmake"])' "$manifest")"
  export JAVA_HOME="${toolchain}/jdk21"
  export ANDROID_SDK_ROOT="$sdk"
  export ANDROID_HOME="$sdk"
  mkdir -p "$sdk"
  if [[ ! -x "${sdk}/platform-tools/adb" ]]; then
    yes | "${sdk}/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$sdk" --licenses >/dev/null
    yes | "${sdk}/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$sdk" \
      "platform-tools" \
      "platforms;android-36" \
      "build-tools;${build_tools}" \
      "emulator" \
      "${image}" \
      "ndk;${ndk}" \
      "cmake;${cmake}"
  fi
  "${sdk}/platform-tools/adb" version
}

install_jdk
install_go
install_cmdline_tools
install_android_sdk

echo "toolchain ready under ${toolchain}"
