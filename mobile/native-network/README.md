# Native network transport (M03)

One app-owned `tsnet.Server` inside a typed C ABI, wrapped by JNI/Kotlin and a narrow
Capacitor plugin. The phone never receives raw URL, dial, LocalAPI or filesystem authority.

## Layout

- `internal/broker/` — policy, HTTP, WebSocket, event queue and tsnet lifecycle.
- `cmd/lib/` — exported C ABI (`nanobot_network_*`) and handle registry.
- `include/nanobot_network.h` — hand-written ABI contract, version 1.
- `cabi/smoke.c` — host C smoke test for validation and error paths.
- `../android/native-network/src/main/cpp/native_network_jni.c` — JNI registration and C ABI calls.
- `../android/native-network/src/main/java/ai/kohaku/nativenetwork/NativeNetwork.kt` — Kotlin JNI declarations.
- `../android/app/src/main/java/ai/kohaku/mobile/network/` — app-side Capacitor bridge.

## Build and test

```bash
source mobile/scripts/mobile-env.sh
mobile/scripts/test-cabi.sh
cd mobile/android
./gradlew :app:verifyDebugJniContract
./gradlew :app:connectedDebugAndroidTest
```

Gradle builds the Go libraries under `mobile/android/native-network/build/generated/go-network/`.
The default build includes `x86_64` and `arm64-v8a`. To build only Arm64, add
`-PmobileAbis=arm64-v8a` to the Gradle command.

To build the Go libraries separately, run `mobile/scripts/build-native-network.sh` from the
repository root. This script runs Go tests and `go vet`, then writes the libraries to
`mobile/android/native-network/build/standalone/go-network/` by default.

## Security rules

- The C ABI accepts only a validated `https://` origin plus installation/hostname/state-dir;
  the `tsnet` transport is the only production mode.
- `state_key_base64` is supplied only by native Keystore custody. The native transport uses it
  with `EncryptedFileStore` to AES-GCM-wrap the `ipn.StateStore` records; plaintext node state
  is never written by the transport. The C ABI rejects a missing or short key at start.
- HTTP operations are named and resolved against exact method/path patterns and allowlisted
  query keys; arbitrary URLs, localhost, unknown headers and oversized bodies are denied.
- Redirects must remain same-origin and resolve to an allowlisted GET route; cross-origin
  redirects are denied without navigation.
- Responses and WebSocket frames are bounded; queues are bounded and return explicit errors.
- Login URLs must be `https` on `login.tailscale.com` or `controlplane.tailscale.com`; the
  Kotlin plugin opens the native-approved URL through an Android Intent, never a JS URL.
- Go tests use `NewTestCore` with an injected dialer. That test-only path is not reachable
  from the C ABI, JNI or Capacitor configuration.
- Live embedded tailnet login/path proof is **M05**, not claimed here.
