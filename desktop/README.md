# Fedora desktop shell

The RPM contains the Tauri setup/status shell, its local UI, the application
icon, license notices, and `backend-compat.json`. It does not install Python,
the WebUI chat, pip packages, or a gateway service. Select an existing config
and interpreter in the setup window; the shell uses that absolute interpreter
to run `-I -m nanobot.desktop` over a private inherited NDJSON pipe.

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
