# Nanobot mobile client

Android-first mobile client for a Nanobot server. The selected server owns the Python agent,
sessions, memory, tools, files and provider credentials; the phone is a scoped client.

Current status (2026-11-01 owner tranche): owner-scoped full-control pairing consent, typed
owner delegation/admission, the versioned 109-operation bounded registry, owner feature
reads/mutations over the standard API, the bounded server folder browser, and owner
settings/apps/skills/automations navigation are implemented. Legacy chat-only devices keep their
negotiated ceiling; initial remote enable, new-device approval, scope upgrade, native folder
picking and remote Start remain local-only or unsupported. The private mapped origin, real
gateway/emulator M35 matrix, physical phone M21 qualification and signed M22 artifact remain
pending. Historical claims below predate this tranche and are not fresh acceptance.

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
the real gateway, including handshake/read/WebSocket after pairing. A later run against a full
`nanobot gateway` with the local `Qwen3.8-27B-UD-Q6_K_XL` provider completed the same phone pairing
and produced real model replies in both the host mobile-edge check (`GATEWAY_MODEL_OK`, 11.7 s) and
the phone chat (`PHONE_OK`, 0.9 s); evidence is in
`evidence/20261031-full-gateway/`. The phone's embedded Tailscale node needs a one-time interactive
login when its prior node state was removed. Normal in-place app updates preserve both
the encrypted Nanobot custody and the embedded Tailscale state; uninstall/clear-data
requires a fresh pairing. If tsnet asks for a new login while custody is intact, the
app offers a re-auth panel instead of erasing the pairing. Use
`mobile/scripts/install-release-update.sh` (or a Play Store update) for in-place
updates; do not uninstall unless a fresh pairing is intended. The mobile bridge now also carries the
model-preset, workspace-control and persisted-thread reads, so the composer model
picker and the Sandbox mode menu work on the phone. The 2026-11-01 owner-client revision changes the phone from a
restricted chat viewer to the same full Nanobot client as the desktop WebUI, with no
separate admin pairing. Implementation is tracked in M30–M34; the current signed
artifact is still the chat-scoped build until those tasks land. The signed artifact was rebuilt with that bridge and the Nanobot launcher
icon/splash (`936803375e50f9b54ca243eb2ea1609b4b1aa706bc5bd7182ce83e7ff4e3edd7`). M20 final matrix is green: WebUI
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
TalkBack was explicitly skipped. Cellular/Wi-Fi handover passed: Wi-Fi was disabled, cellular became the default network, Wi-Fi was re-enabled, and the existing WebSocket accepted a follow-up message. The QR scanner is locked to portrait; scanning and cancellation return to the normal portrait app. Remaining gate is final human review/acceptance.
See `docs/mobile-client-tasks.md`.

## Turn-complete notifications

The shared WebUI **Task notifications** setting is bridged to Capacitor Local
Notifications. Android 13+ asks for notification permission when the setting is
enabled, and a completed turn shows a system notification while the app is not
focused. Completion alerts are emitted by the shared client rather than an
individual chat view, so leaving the thread or navigating elsewhere in the app
does not lose the notification. Only runs started from this phone notify the
phone.

While a phone-started run is active, the Android client starts a `dataSync`
foreground service with a low-importance ongoing notification and a partial
wake lock. This keeps the process, WebSocket and WebView JavaScript alive after
the app is backgrounded, so the completion event can still produce a local
notification. The service stops on completion, stop, failure, or when the app
is destroyed/killed.

`dataSync` foreground services are limited to six hours per 24-hour period on
Android 15+. The service implements `onTimeout()` and stops itself. Once the
quota is exhausted, Android rejects another `dataSync` foreground service until
the app is brought to the foreground. Swiping the app from Recents, exiting it,
or force-stopping it ends background monitoring; the v1 client still does not
use push delivery.

## Owner pairing (2026-11-03)

New pairing is owner-only. The desktop shows a 300-second QR challenge plus an
8-digit fallback code; the phone can scan the QR, paste the `nanobot://pair`
link, or open **Use an 8-digit code instead** and enter the full HTTPS origin and
code. Both paths require a desktop approval. The phone no longer displays or asks
for a comparison code.

Owner access is full control (files, commands, settings and existing chats), and
the owner refresh token has no idle expiry: it remains valid until revoked in the
desktop device list. Existing chat-only devices are revoked when the owner
service starts and must re-pair as owner devices. Tailscale direct is the only
v1 transport; there is no relay or biometric gate.

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

## Debug and release side by side

Debug builds use a separate application ID and label so they can be installed beside the
release app:

| Build | Application ID | Label | Signing |
|---|---|---|---|
| Release | `ai.nanobot.mobile` | Nanobot Mobile | release keystore |
| Debug | `ai.nanobot.mobile.debug` | Nanobot Dev | debug keystore |

The debug app has its own app data, custody, pairing and embedded Tailscale state, so development
does not overwrite the release app or require re-pairing it.

Build and install the shared-WebUI dev app:

```bash
mobile/scripts/build-debug.sh
adb install -r mobile/android/app/build/outputs/apk/debug/app-debug.apk
```

`build-debug.sh` packages `mobile/www-shared`. The `nanobot://pair` scheme is shared by both
apps, so Android may show an app chooser; choose **Nanobot Dev** when testing the dev build.

Release builds and in-place release updates keep their existing flow:

```bash
mobile/scripts/build-release.sh
mobile/scripts/install-release-update.sh <serial> mobile/evidence/<date>-m22-release/nanobot-mobile-release.apk
```

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
the executing ABI and page size. It grants no network, file or credential authority. The fixture
debug build installs as `ai.nanobot.mobile.debug`, so it can coexist with the release app.

## Emulator qualification

```bash
mobile/scripts/qualify-emulator.sh --evidence-dir <private-run-directory>
mobile/scripts/qualify-emulator.sh --evidence-dir <private-run-directory> --native-tests
```

The runner builds, creates/reuses an x86_64 API 36 AVD, installs the fixture debug APK
(`ai.nanobot.mobile.debug`), waits for the native probe marker, captures a screenshot and cleans
up the emulator unless `--keep-emulator` is passed. Evidence stays under the gitignored
`mobile/evidence/`.

`--native-tests` also runs the Android instrumentation suite, which loads the JNI/native-network
seam and checks ABI version, invalid-config fail-closed behavior and unknown-handle errors.

## Native transport

`mobile/native-network/` contains the M03 Go `tsnet` transport and typed C ABI. Build the
Android libraries with `mobile/scripts/build-native-network.sh` before Gradle; they are
generated into `android/app/src/main/jniLibs/` and are not tracked. Host ABI smoke test:

```bash
mobile/scripts/test-cabi.sh
```

Emulator `37.2.13` segfaults during display setup on this Fedora 44 host in every tested
hardware/software mode except ANGLE indirect (`-gpu auto`, `host`, `swiftshader`,
`swiftshader_indirect`, `off` and `guest` all crashed). The runner therefore defaults to
`-gpu angle_indirect`, which booted the API 36 image in about 19 seconds on the same kernel.
Override with `NANOBOT_MOBILE_EMULATOR_GPU` or `--gpu` when a host has a different known-good
mode, and record the result with the evidence.
