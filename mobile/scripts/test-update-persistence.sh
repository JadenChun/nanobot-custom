#!/usr/bin/env bash
# Prove that an in-place APK update preserves encrypted custody and network-state files.
# The installed app must be the debuggable test build. Set ALLOW_UNINSTALL=1 to replace
# a release build (this clears existing app data).
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

ADB=(adb -s "${SERIAL}")
APP_APK="$(cd "${MOBILE_ROOT}/android" && pwd)/app/build/outputs/apk/debug/app-debug.apk"
TEST_APK="$(cd "${MOBILE_ROOT}/android" && pwd)/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
SEED_CLASS="ai.nanobot.mobile.network.UpdatePersistenceSeedInstrumentedTest"
VERIFY_CLASS="ai.nanobot.mobile.network.UpdatePersistenceVerifyInstrumentedTest"

if "${ADB[@]}" shell pm list packages | grep -q '^package:ai.nanobot.mobile$'; then
  if ! "${ADB[@]}" shell dumpsys package ai.nanobot.mobile | grep -q 'debuggable=true'; then
    if [[ "${ALLOW_UNINSTALL:-0}" != "1" ]]; then
      echo "release build installed; set ALLOW_UNINSTALL=1 to clear it for this test" >&2
      exit 2
    fi
    "${ADB[@]}" uninstall ai.nanobot.mobile >/dev/null
  fi
fi

"${ADB[@]}" install -r "${APP_APK}" >/dev/null
"${ADB[@]}" install -r "${TEST_APK}" >/dev/null

echo "seeding custody and network state"
"${ADB[@]}" shell am instrument -w -e class "${SEED_CLASS}" \
  ai.nanobot.mobile.test/androidx.test.runner.AndroidJUnitRunner | tee /tmp/nanobot-update-seed.txt
grep -Eq 'OK \(1 test\)|OK \(1 test\)' /tmp/nanobot-update-seed.txt

echo "updating target APK in place"
"${ADB[@]}" install -r "${APP_APK}" >/dev/null

echo "verifying custody and network state"
"${ADB[@]}" shell am instrument -w -e class "${VERIFY_CLASS}" \
  ai.nanobot.mobile.test/androidx.test.runner.AndroidJUnitRunner | tee /tmp/nanobot-update-verify.txt
grep -Eq 'OK \(1 test\)|OK \(1 test\)' /tmp/nanobot-update-verify.txt

echo "PASS: custody and network state survived adb install -r"
