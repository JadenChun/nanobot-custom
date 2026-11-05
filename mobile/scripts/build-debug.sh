#!/usr/bin/env bash
# Build the shared-WebUI debug app with a distinct application ID.
# The debug package (ai.nanobot.mobile.debug) can be installed alongside the
# release package (ai.nanobot.mobile) without touching release data.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mobile-env.sh
source "${script_dir}/mobile-env.sh"
mobile_require_toolchain

export NANOBOT_MOBILE_WEB_DIR="www-shared"

"${script_dir}/build-native-network.sh"
"${script_dir}/build-shared-webui.sh"

(
  cd "${MOBILE_ROOT}"
  bunx cap sync android
)

(
  cd "${MOBILE_ROOT}/android"
  ./gradlew --no-daemon :app:assembleDebug
)

apk="${MOBILE_ROOT}/android/app/build/outputs/apk/debug/app-debug.apk"
test -f "${apk}"
package_name="$("${ANDROID_SDK_ROOT}/build-tools/36.0.0/aapt" dump badging "${apk}" \
  | sed -n 's/^package: name=\([^ ]*\).*/\1/p' | head -1)"
echo "debug APK: ${apk}"
echo "package: ${package_name}"
echo "install: adb install -r ${apk}"
