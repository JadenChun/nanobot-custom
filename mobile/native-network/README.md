# Native network transport (M03)

One app-owned `tsnet.Server` inside a typed C ABI, wrapped by JNI/Kotlin and a narrow
Capacitor plugin. The phone never receives raw URL, dial, LocalAPI or filesystem authority.

## Layout

- `internal/broker/` — policy, HTTP, WebSocket, event queue and tsnet lifecycle.
- `cmd/lib/` — exported C ABI (`nanobot_network_*`) and handle registry.
- `include/nanobot_network.h` — hand-written ABI contract, version 1.
- `cabi/smoke.c` — host C host smoke test (validation and error paths).
- `../android/app/src/main/cpp/native_network_jni.c` — JNI shim.
- `../android/app/src/main/java/ai/nanobot/mobile/network/` — Kotlin/Capacitor bridge.

## Build and test

```bash
cd mobile
../scripts/build-native-network.sh   # go test/vet + x86_64 and arm64-v8a c-shared builds
../scripts/test-cabi.sh              # host C ABI link/smoke test
cd android && ./gradlew :app:connectedDebugAndroidTest
```

Android shared libraries are generated under `app/src/main/jniLibs/` and are intentionally
gitignored. The `qualify-emulator.sh --native-tests` runner builds them, boots the AVD, runs
Android instrumentation and then executes the fixture probe.

## Security rules

- The C ABI accepts only a validated `https://` origin plus installation/hostname/state-dir;
  the `tsnet` transport is the only production mode.
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
