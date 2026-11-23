# macOS desktop usage and validation

Kohaku uses AppKit and WKWebView through Tauri. It does not bundle a
second backend or a browser, change the existing CLI installation, or add a
LaunchAgent. On first launch, the setup window suggests `~/.kohaku/config.json`
and a managed Python install (`~/.local/share/kohaku/desktop-current` first,
then `current`). It remembers the resolved release rather than the moving
selector. You can choose different paths before clicking **Connect**.

If the standard config is missing, **Connect** creates a private loopback
config with a generated WebUI auth secret and the usual Kohaku defaults.
It never replaces an existing file, uses the fallback for a missing custom
path, or rewrites a config that fails validation. No gateway starts before
the normal provider and runtime checks pass; a fresh profile still needs a
model provider if one is not configured in the environment.

Both the gateway management and WebSocket listeners must be
loopback-only. WebSocket authentication must be configured. A management bind
of `0.0.0.0` is not eligible, even if WebSocket is already on loopback. The app
does not rewrite either listener setting.

The shell remembers selectors under
`~/Library/Application Support/kohaku-desktop/preferences.json` (or the
explicit `XDG_CONFIG_HOME`). Chat's website data uses a nonpersistent
WKWebView store; backend chat history remains in the selected profile. The
host grants navigation only to the packaged setup page and the verified
gateway origin, denies external popups and navigation, and exposes native
commands only to the local setup page. macOS asks for microphone permission
after Kohaku opens chat for the first time. If you deny access, use
**Privacy & Security** in System Settings to permit access. The qualification
below predates the microphone change.

Closing the window keeps the gateway alive. Reopen from the Dock or use
**Open Chat** from the menu bar icon. Gateway readiness and status changes
do not restore the window or take focus. If the gateway becomes unavailable,
use the Dock or **Open Chat** to see the setup/status controls. Use
**Quit Kohaku** (Cmd+Q) from the application menu or the menu bar icon to stop
the managed gateway and exit. Force Quit and external app termination are
not managed shutdowns; the persistent gateway can remain available for
reconciliation on relaunch. On stop, the adapter revalidates the selected
process and uses a Darwin audit token (which includes a PID generation) to
send SIGTERM. It then waits for that process's kqueue exit event. It does not
fall back to a PID-only signal or SIGKILL. Linux retains its existing pidfd
path.

## Verification (2026-11-18)

- Apple Silicon, macOS 27.0.1, Xcode 27.0, Rust 1.100.0, Tauri CLI 2.10.1.
  Native release bundle built, passed its install check, and passed strict
  ad hoc code-signature verification. Intel, older macOS versions, and
  notarized distribution were not verified.
- Locked Rust tests: 17 passed, including local-only setup ACL and gateway
  navigation policy. Focused Python desktop tests: 14 passed and 7 skipped
  (Linux-only paths). Full-source BasedPyright: zero diagnostics before
  and after. Fedora runtime was not requalified by this run.
- Launch Services opened the bundle. Native setup selected a separate config
  and port-capable backend. WKWebView connected and received a streamed reply
  from a loopback scripted provider. Reopen and a second executable launch
  retained the original desktop host and gateway. Cmd+Q exited both
  captured processes and closed the fixture's two listeners.
- The operator config was left byte-for-byte unchanged and was not used for
  the run. No live provider credentials or jobs were used.

First-run follow-up: the rebuilt app prefilled the standard config and the
managed interpreter in two disposable homes. One-click Connect used the
existing profile unchanged, streamed a reply, and stopped both test listeners
on Cmd+Q. In the empty home it created a mode-0600 loopback config only on
Connect and showed the missing model-provider guidance without starting a
gateway. The updated suite passed 18 locked Rust and 17 Python desktop tests
(7 existing Linux-only skips); full-source type checking remained clean.
