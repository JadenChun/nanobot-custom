#!/usr/bin/env bash
# M01 emulator qualification: reproducible debug build, isolated AVD, native ABI probe.
# This is fixture-only evidence; it does not claim transport or custody behavior.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mobile-env.sh
source "${script_dir}/mobile-env.sh"
mobile_require_toolchain

run_id="$(date +%Y%m%d-%H%M%S)"
evidence_dir="${MOBILE_ROOT}/evidence/${run_id}/M01"
avd_name="${NANOBOT_MOBILE_AVD:-nanobot-m01}"
emulator_port="${NANOBOT_MOBILE_EMULATOR_PORT:-5590}"
emulator_gpu="${NANOBOT_MOBILE_EMULATOR_GPU:-angle_indirect}"
serial="emulator-${emulator_port}"
keep_emulator=0
native_tests=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence-dir) evidence_dir="$2"; shift 2 ;;
    --avd) avd_name="$2"; shift 2 ;;
    --port) emulator_port="$2"; serial="emulator-${emulator_port}"; shift 2 ;;
    --gpu) emulator_gpu="$2"; shift 2 ;;
    --keep-emulator) keep_emulator=1; shift ;;
    --native-tests) native_tests=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${evidence_dir}"
exec > >(tee "${evidence_dir}/qualify.log") 2>&1

emulator_pid=""
cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "${emulator_pid}" ]] && kill -0 "${emulator_pid}" 2>/dev/null; then
    if [[ "${keep_emulator}" -eq 1 ]]; then
      echo "keeping emulator pid ${emulator_pid} (requested)"
    else
      adb -s "${serial}" emu kill >/dev/null 2>&1 || kill "${emulator_pid}" >/dev/null 2>&1 || true
      timeout 30 bash -c "while kill -0 ${emulator_pid} 2>/dev/null; do sleep 1; done" || true
    fi
  fi
  echo "cleanup exit=${exit_code} finished_at=$(date -Is)"
  exit "${exit_code}"
}
trap cleanup EXIT

echo "started_at=$(date -Is)"
echo "base_head=$(git -C "${MOBILE_ROOT}/.." rev-parse HEAD)"
echo "avd=${avd_name} serial=${serial}"

uname -a
"${JAVA_HOME}/bin/java" -version
"${MOBILE_TOOLCHAIN}/go/bin/go" version
adb version
"${ANDROID_SDK_ROOT}/emulator/emulator" -version | head -1
"${ANDROID_SDK_ROOT}/emulator/emulator" -accel-check

(
  cd "${MOBILE_ROOT}"
  bun install --frozen-lockfile
  bun run build
  bunx cap sync android
)

"${script_dir}/build-native-network.sh"

(
  cd "${MOBILE_ROOT}/android"
  ./gradlew --no-daemon :app:assembleDebug
)

apk="${MOBILE_ROOT}/android/app/build/outputs/apk/debug/app-debug.apk"
test -f "${apk}"
cp "${apk}" "${evidence_dir}/app-debug-universal.apk"
unzip -l "${apk}" | grep 'lib/.*/libnanobot_probe\.so' | tee "${evidence_dir}/native-libs.txt"

(
  cd "${MOBILE_ROOT}/android"
  # Separate compile-only proof for the release ABI; not a device execution claim.
  ./gradlew --no-daemon :app:assembleDebug -PmobileAbis=arm64-v8a
)
cp "${apk}" "${evidence_dir}/app-debug-arm64-only.apk"
unzip -l "${evidence_dir}/app-debug-arm64-only.apk" | grep 'lib/.*/libnanobot_probe\.so' | tee "${evidence_dir}/native-libs-arm64.txt"
echo "arm64_v8a_build=COMPILED_ONLY (not executed on this host)" | tee "${evidence_dir}/abi-labels.txt"

(
  cd "${MOBILE_ROOT}/android"
  # Restore the universal APK used by the x86_64 emulator installation.
  ./gradlew --no-daemon :app:assembleDebug
)
cp "${apk}" "${evidence_dir}/app-debug-emulator.apk"

avdmanager="${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin/avdmanager"
if ! "${avdmanager}" list avd | grep -q "Name: ${avd_name}"; then
  echo no | "${avdmanager}" create avd \
    --name "${avd_name}" \
    --package "system-images;android-36;google_apis;x86_64" \
    --device pixel_7 \
    --force
fi
grep 'image.sysdir.1' "${ANDROID_AVD_HOME}/${avd_name}.avd/config.ini" | tee "${evidence_dir}/avd-sysdir.txt"
if ! grep -q '^image.sysdir.1=system-images/' "${ANDROID_AVD_HOME}/${avd_name}.avd/config.ini"; then
  echo "AVD system-image path is not SDK-root relative" >&2
  exit 1
fi

"${ANDROID_SDK_ROOT}/emulator/emulator" \
  -avd "${avd_name}" \
  -port "${emulator_port}" \
  -no-window \
  -no-audio \
  -no-boot-anim \
  -no-snapshot \
  -gpu "${emulator_gpu}" \
  -camera-back none \
  -camera-front none \
  >"${evidence_dir}/emulator.log" 2>&1 &
emulator_pid=$!

if ! timeout 180 adb -s "${serial}" wait-for-device; then
  echo "emulator did not appear on ${serial} within 180s" >&2
  tail -20 "${evidence_dir}/emulator.log" >&2 || true
  exit 1
fi
booted=0
for _ in $(seq 1 180); do
  if ! kill -0 "${emulator_pid}" 2>/dev/null; then
    echo "emulator process exited during boot" >&2
    tail -20 "${evidence_dir}/emulator.log" >&2 || true
    exit 1
  fi
  if [[ "$(adb -s "${serial}" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]]; then
    booted=1
    break
  fi
  sleep 1
done
if [[ "${booted}" -ne 1 ]]; then
  echo "emulator did not boot within 180s" >&2
  exit 1
fi

adb -s "${serial}" shell input keyevent 82 >/dev/null 2>&1 || true
adb -s "${serial}" shell settings put global window_animation_scale 0
adb -s "${serial}" shell settings put global transition_animation_scale 0
adb -s "${serial}" shell settings put global animator_duration_scale 0

adb -s "${serial}" install -r -t "${apk}"
if [[ "${native_tests}" -eq 1 ]]; then
  (
    cd "${MOBILE_ROOT}/android"
    ANDROID_SERIAL="${serial}" ./gradlew --no-daemon :app:connectedDebugAndroidTest
  )
  # The instrumentation runner uninstalls the target app; restore it for the
  # fixture shell/probe phase.
  adb -s "${serial}" install -r -t "${apk}"
fi
adb -s "${serial}" logcat -c
adb -s "${serial}" shell am force-stop ai.nanobot.mobile.debug
adb -s "${serial}" shell am start -W -n ai.nanobot.mobile.debug/ai.nanobot.mobile.MainActivity

probe_ok=0
for _ in $(seq 1 90); do
  if adb -s "${serial}" logcat -d -s NanobotProbe:I Capacitor/Console:I 2>/dev/null \
    | grep -q 'NANOBOT_MOBILE_PROBE_OK'; then
    probe_ok=1
    break
  fi
  sleep 1
done
adb -s "${serial}" logcat -d -s NanobotProbe:I Capacitor/Console:I >"${evidence_dir}/probe-logcat.txt" || true
cat "${evidence_dir}/probe-logcat.txt"

adb -s "${serial}" shell getprop ro.product.cpu.abi >"${evidence_dir}/abi.txt"
adb -s "${serial}" shell getprop ro.build.version.sdk >>"${evidence_dir}/abi.txt"
adb -s "${serial}" shell getprop ro.build.version.release >>"${evidence_dir}/abi.txt"
adb -s "${serial}" exec-out screencap -p >"${evidence_dir}/fixture-shell.png" || true
adb -s "${serial}" shell dumpsys package ai.nanobot.mobile.debug >"${evidence_dir}/package-dump.txt" || true

if [[ "${probe_ok}" -ne 1 ]]; then
  echo "native probe marker missing" >&2
  exit 1
fi

echo "M01_RESULT=PASS native probe executed on ${serial}"
