# Nanobot mobile client

Android-first mobile client for a Nanobot server. The selected server owns the Python agent,
sessions, memory, tools, files and provider credentials; the phone is a scoped client.

Status: M01–M04 and M06–M13 are implemented and tested locally. M05 embedded tsnet login and
the M09 owned Serve mapping were exercised live from the x86_64 emulator and then cleaned up;
M05 media/deep-link and M15–M22 still require the remaining live backend/device work. M14 live
pairing plus refresh/replay/revoke were exercised against the isolated owner data plane. M15 native
media handles, fetcher/attachment policy, exact target binding and the Capacitor attachment picker
are in place. M16 runtime is wired into bootstrap/HTTP/client surface, mobile service-worker registration is
disabled and shared assets package to `mobile/www-shared`. M17 adds mobile safe-area CSS, Back/Escape
handling and connection/recovery notices. The full Python suite has 17,098 passing tests and five
pre-existing baseline failures. M18 headless management and the shared Remote Access settings page are implemented. M19 live owned
user-service install/restart/removal passed and was cleaned up. M20 automated WebUI/emulator/Python
matrix is captured; five pre-existing baseline Python failures remain. M21 arm64 execution, live
M14 data path and hardware Keystore passed on a physical A142/Android 16 phone. M16 paired-target
persistence boots the packaged shared UI through the native transport (handshake/read 200), and the
shared `/mobile/v1/ws` bridge now opens on the physical phone (connected chat shell). Signed media
resolves through native handles in shared attachments/lightbox; pairing deep links are captured and
handed to the plugin. M20 final matrix: WebUI 1363, Python 17,103 with five pre-existing baseline
failures, phone 29 tests (4 live skipped). Remaining manual device matrix and M22 signing remain.
See `docs/mobile-client-tasks.md`.

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
