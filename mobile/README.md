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
handling and connection/recovery notices. The full Python suite is green: 17,111 passed, 22
skipped. M18 headless management and the shared Remote Access settings page are implemented. M19 live owned
user-service install/restart/removal passed and was cleaned up. M20 automated WebUI/emulator/Python
matrix is captured; five pre-existing baseline Python failures remain. M21 arm64 execution, live
M14 data path and hardware Keystore passed on a physical A142/Android 16 phone. M16 paired-target
persistence boots the packaged shared UI through the native transport, and the shared
`/mobile/v1/ws` bridge opens on the physical phone against both the staging fixture and a real
production-composed gateway. The real-gateway run passed handshake/read 200, WebSocket ready/attach,
media full/range/oversize/hostile cases, and live revocation denial. Signed media resolves through
native handles in shared attachments/lightbox. Product pairing now works end to end: Remote
Access settings creates the exact `nanobot://pair` challenge and renders it as a QR/link, the
phone scans the QR or opens the link, the native bridge collects the owner challenge, displays the
comparison code, then polls approval and stores the issued tokens in Keystore custody. The device
remains linked until it is revoked in Remote Access settings; revoke denies new admissions. The
full flow passed on the A142 phone against
the real gateway, including handshake/read/WebSocket after pairing. M20 final matrix is green: WebUI
1,364 passed; Python 17,111 passed, 22 skipped; phone 31 tests (8 live skipped, 23 passed). The five baseline
failures and the order-dependent flake were repaired. M22 signed release is built and verified with
the release keystore: v2 signature, zipalign, install, same-key update, different-key rejection and
uninstall/reinstall all passed. Automated M21 checks on the signed release also pass:
portrait/landscape layout, process death, forced Doze, Back/relaunch, font scale 1.3 and a
battery/thermal snapshot. Device-tool checks found and fixed three real defects: Capacitor numeric
WebSocket IDs were rejected by `getLong`, the mobile frame bound was too small for a 144 KB image,
and the product client had no access-token refresh. WebSocket IDs now use numeric fallbacks, the
frame bound is 512 KiB, and a 401 triggers native refresh rotation and one retry. The mobile
composer now offers Camera and Library sources; camera capture, compression, attachment and send
passed on the A142. Small-image send and the live refresh instrumentation test also passed.
TalkBack was explicitly skipped. Cellular/Wi-Fi handover passed: Wi-Fi was disabled, cellular became the default network, Wi-Fi was re-enabled, and the existing WebSocket accepted a follow-up message. Remaining gate is final human review/acceptance.
See `docs/mobile-client-tasks.md`.


## Release signing

Release signing is configured from the environment, never from tracked files. The default
keystore path is `~/.config/nanobot-mobile/release.keystore` and must be mode `600`.

Create a private `~/.config/nanobot-mobile/release.env` (mode `600`) with:

```bash
NANOBOT_MOBILE_KEYSTORE=$HOME/.config/nanobot-mobile/release.keystore
NANOBOT_MOBILE_KEY_ALIAS=<key-alias>
NANOBOT_MOBILE_KEYSTORE_PASSWORD=<store-password>
NANOBOT_MOBILE_KEY_PASSWORD=<key-password>
```

`NANOBOT_MOBILE_KEYSTORE_PASSWORD_FILE` can point at a single-line password file instead of
`NANOBOT_MOBILE_KEYSTORE_PASSWORD`. The key password defaults to the store password.

Build and verify the signed release:

```bash
mobile/scripts/build-release.sh
```

The script packages `mobile/www-shared` and the native transport, runs `assembleRelease`,
checks `zipalign` and `apksigner`, and writes the APK, certificate report and SHA-256 under
`mobile/evidence/<date>-m22-release/`. Release tasks fail closed when signing material is absent.

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
