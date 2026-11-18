# Nanobot Desktop

The native bundle contains the Tauri setup/status shell, its local UI, the application
icon, license notices, and `backend-compat.json`. It does not install Python,
the WebUI chat, pip packages, or a gateway service. The setup window selects a
config and an installed interpreter; the shell uses that absolute interpreter
to run `-I -m nanobot.desktop` over a private inherited NDJSON pipe.

## macOS

On macOS with Xcode, Rust, Bun 1.4.2, and Tauri CLI 2.10.1:

```sh
(cd desktop/ui && bun install --frozen-lockfile)
cd desktop/src-tauri
cargo tauri build --bundles app -- --locked
```

Install the resulting `Nanobot Desktop.app` into an Applications folder.
The local build is ad hoc signed, not notarized. Select a matching Python
environment: macOS additionally requires the `darwin-identity-stop-v1` backend
capability, so an older Nanobot 0.3.0 wheel is not sufficient. The native window
uses WKWebView to render the shared chat UI; it is not a browser shortcut.

See [macOS usage and validation](MACOS.md) for config requirements, behavior,
and verification limits.

## Fedora

Build on Fedora 44 with the documented GTK3, WebKitGTK 4.1, AppIndicator and
Rust prerequisites, Bun, and Tauri CLI 2.10.1 on `PATH`:

```sh
cd desktop/src-tauri
cargo tauri build --bundles rpm -- --locked
```

The UI build checks that every local script and stylesheet referenced by the
emitted HTML exists in `dist` before Tauri packages it.

The compatibility manifest pins version 0.3.0, which passed the outside-checkout
pipe/lifecycle smoke test, and the accepted v1 protocol. A
version or capability mismatch leaves the setup controls visible and does not
start a gateway.

Build and runtime evidence is retained in
`.assets/validation/fedora-tauri-2026-10-23/native-*` during qualification.

See [VALIDATION.md](VALIDATION.md) for install, usage, compatibility, rollback,
and qualification notes.
