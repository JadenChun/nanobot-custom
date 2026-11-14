#!/usr/bin/env bash
# Prove that an in-place APK update preserves encrypted custody and network-state files.
# This targets the debuggable dev build (ai.nanobot.mobile.debug), so it can run alongside
# a release install without touching release data.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APK_DIR="$(cd -- "${script_dir}/../android" && pwd)/app/build/outputs/apk"
SEED_APK="${NANOBOT_UPDATE_SEED_APK:-}"
CANDIDATE_APK="${NANOBOT_UPDATE_CANDIDATE_APK:-${DEFAULT_APK_DIR}/debug/app-debug.apk}"
TEST_APK="${NANOBOT_UPDATE_TEST_APK:-${DEFAULT_APK_DIR}/androidTest/debug/app-debug-androidTest.apk}"
DEBUG_PACKAGE="ai.nanobot.mobile.debug"
SEED_CLASS="ai.nanobot.mobile.network.UpdatePersistenceSeedInstrumentedTest"
VERIFY_CLASS="ai.nanobot.mobile.network.UpdatePersistenceVerifyInstrumentedTest"
RECEIPT="${NANOBOT_UPDATE_RECEIPT:-/tmp/nanobot-update-receipt.txt}"

if [[ -z "${SEED_APK}" || ! -f "${SEED_APK}" ]]; then
  echo "NANOBOT_UPDATE_SEED_APK must name the baseline APK" >&2
  exit 2
fi
if [[ ! -f "${CANDIDATE_APK}" ]]; then
  echo "candidate APK not found: ${CANDIDATE_APK}" >&2
  exit 2
fi
seed_hash="$(sha256sum "${SEED_APK}" | awk '{print $1}')"
candidate_hash="$(sha256sum "${CANDIDATE_APK}" | awk '{print $1}')"
if [[ "${seed_hash}" == "${candidate_hash}" ]]; then
  echo "seed and candidate APKs are identical; an update must cross distinct artifacts" >&2
  exit 2
fi

if [[ ! -f "${TEST_APK}" ]]; then
  echo "instrumentation APK not found: ${TEST_APK}" >&2
  exit 2
fi

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

if "${ADB[@]}" shell pm list packages | grep -q "^package:${DEBUG_PACKAGE}$"; then
  if ! "${ADB[@]}" shell dumpsys package "${DEBUG_PACKAGE}" | grep -Eq 'debuggable=true|DEBUGGABLE'; then
    echo "unexpected non-debug package at ${DEBUG_PACKAGE}" >&2
    exit 2
  fi
fi

"${ADB[@]}" uninstall "${DEBUG_PACKAGE}.test" >/dev/null 2>&1 || true
"${ADB[@]}" uninstall "${DEBUG_PACKAGE}" >/dev/null 2>&1 || true
"${ADB[@]}" install -r "${SEED_APK}" >/dev/null
"${ADB[@]}" install -r "${TEST_APK}" >/dev/null
seed_version="$("${ADB[@]}" shell dumpsys package "${DEBUG_PACKAGE}" | awk '/versionCode=/{print $1; exit}')"

echo "seeding custody and network state from baseline APK"
"${ADB[@]}" shell am instrument -w -e class "${SEED_CLASS}" \
  "${DEBUG_PACKAGE}.test/androidx.test.runner.AndroidJUnitRunner" | tee /tmp/nanobot-update-seed.txt
grep -Eq 'OK \(1 test\)|OK \(1 test\)' /tmp/nanobot-update-seed.txt

echo "updating to candidate APK in place"
"${ADB[@]}" install -r "${CANDIDATE_APK}" >/dev/null
candidate_version="$("${ADB[@]}" shell dumpsys package "${DEBUG_PACKAGE}" | awk '/versionCode=/{print $1; exit}')"

echo "verifying custody and network state"
"${ADB[@]}" shell am instrument -w -e class "${VERIFY_CLASS}" \
  "${DEBUG_PACKAGE}.test/androidx.test.runner.AndroidJUnitRunner" | tee /tmp/nanobot-update-verify.txt
grep -Eq 'OK \(1 test\)|OK \(1 test\)' /tmp/nanobot-update-verify.txt

{
  echo "seed_apk=${SEED_APK}"
  echo "seed_sha256=${seed_hash}"
  echo "candidate_apk=${CANDIDATE_APK}"
  echo "candidate_sha256=${candidate_hash}"
  echo "seed_version=${seed_version}"
  echo "candidate_version=${candidate_version}"
  echo "state_transition=retained"
  echo "result=PASS"
} | tee "${RECEIPT}"

echo "PASS: custody and network state survived a distinct-APK adb install -r"
