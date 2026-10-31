# Nanobot mobile client

Android-first mobile client for a Nanobot server. The selected server owns the Python agent,
sessions, memory, tools, files and provider credentials; the phone is a scoped client.

Status: M01–M04 and M06–M13 are implemented and tested locally. M05 embedded tsnet login and
the M09 owned Serve mapping were exercised live from the x86_64 emulator and then cleaned up;
M05 media/deep-link and M15–M22 still require the remaining live backend/device work. M14 live
pairing plus refresh/replay/revoke were exercised against the isolated owner data plane. M15 native
media handles, fetcher/attachment policy and exact target binding are in place; the product event
adapter, picker activity and shared UI remain. See `docs/mobile-client-tasks.md`.

## Isolated toolchain

All build inputs stay inside this checkout and are ignored by Git:

```bash
mobile/scripts/bootstrap-toolchain.sh   # JDK 21, Go 1.27.3, Android SDK 36, NDK, CMake, AVD image
source mobile/scripts/mobile-env.sh     # exports the checkout-local paths
```

Pinned versions, URLs and checksums are recorded in `toolchain.lock.json`. The script does not
modify system packages, services or user-wide SDK paths.

## Fixture build

```bash
cd mobile
bun install --frozen-lockfile
bun run build
bunx cap sync android
cd android && ./gradlew :app:assembleDebug
```

`M01` also ships a fixture-only JNI/CMake probe (`android/app/src/main/cpp/probe.c`) that reports
the executing ABI and page size. It grants no network, file or credential authority.

## Emulator qualification

```bash
mobile/scripts/qualify-emulator.sh --evidence-dir <private-run-directory>
mobile/scripts/qualify-emulator.sh --evidence-dir <private-run-directory> --native-tests
```

The runner builds, creates/reuses an x86_64 API 36 AVD, installs the debug APK, waits for the
native probe marker, captures a screenshot and cleans up the emulator unless `--keep-emulator`
is passed. Evidence stays under the gitignored `mobile/evidence/`.

`--native-tests` also runs the Android instrumentation suite, which loads the JNI/native-network
seam and checks ABI version, invalid-config fail-closed behavior and unknown-handle errors.

## Native transport

`mobile/native-network/` contains the M03 Go `tsnet` transport and typed C ABI. Build the
Android libraries with `mobile/scripts/build-native-network.sh` before Gradle; they are
generated into `android/app/src/main/jniLibs/` and are not tracked. Host ABI smoke test:

```bash
mobile/scripts/test-cabi.sh
```

Emulator `37.2.13` crashes in its packaged SwiftShader renderer on this Fedora 44 host when
headless software rendering is requested (`-gpu swiftshader_indirect`, `-gpu off`, `-gpu guest`).
The runner therefore defaults to `-gpu host`, which uses the host GPU and KVM and booted the
API 36 image in about 17 seconds. Override with `NANOBOT_MOBILE_EMULATOR_GPU` or `--gpu` when a
different host requires software rendering, and record the result with the evidence.
