# Fedora desktop usage and validation

Nanobot Desktop 0.3.0 is a thin Fedora shell for an already installed Nanobot
Python backend. Its RPM contains the native shell, setup/status UI, icon,
compatibility manifest, and license notices. It does not install Python, the
WebUI chat, Python packages, or a gateway service. The selected interpreter
must already provide the compatible `nanobot.desktop` module.

## Supported setup

The qualified target is Fedora 44 x86_64 with KDE Plasma on Wayland, GTK 3,
WebKitGTK 4.1, and AppIndicator. The native build used Rust/Cargo 1.99.0,
Tauri 2.12.5 (CLI 2.10.1), and Bun 1.4.2. Runtime RPM dependencies are
`gtk3`, `webkit2gtk4.1`, and `libappindicator-gtk3`.

The bundled compatibility manifest accepts desktop protocol v1, backend
version 0.3.0, and the `managed-gateway-v1` capability. An unsupported or
unverifiable backend stays in setup/status and is not started. Select an
existing config file and an absolute Python interpreter that has Nanobot
installed; optionally set a workspace override. The shell remembers only
those selectors in `~/.config/nanobot-desktop/preferences.json` (or the
equivalent XDG config directory). It keeps WebKit data in a private profile
under `~/.local/share/nanobot-desktop/` (or the equivalent XDG data directory).

## Install and use

Install the RPM with the system package manager, then open **Nanobot Desktop**
from the application menu or run `nanobot-desktop`. On first use, select the
existing config and Python interpreter and connect. The shell validates the
installation before it opens chat. It never installs or updates the selected
Python environment.

Closing the chat window hides or minimizes the retained window after the
recording-cleanup notification completes. **Open Chat** in the tray menu
restores it. The gateway remains alive while the window is hidden. To stop the
selected managed gateway and exit the shell, choose **Quit Nanobot** in the
tray or setup/status controls and wait for the confirmed stop. Uninstalling the
RPM does not stop a gateway; quit it first if it should stop.

To roll back, quit Nanobot Desktop, then install the previous RPM with the
package manager, for example `sudo dnf downgrade ./nanobot-desktop-<previous-version>.rpm`.
If that RPM is unavailable, remove this shell with `sudo dnf remove nanobot-desktop`
and install the desired previous package. The shell package has no service
scriptlets and does not migrate or delete backend configuration or runtime
data.

## Voice input

Microphone capture relies on WebKitGTK granting the page's media request. The
desktop shell does not grant microphone access. On the qualified release, the
actual Voice input control displayed: “Allow microphone access in the address
bar, then retry.” No recording started. That browser-oriented error is the
existing chat fallback; this release does not provide native microphone
capture. Text chat remains available.
The release-window observation is recorded in
`native-microphone-denied.json`.

## Build and validation

Build from `desktop/src-tauri` on Fedora with the dependencies above and Tauri
CLI 2.10.1 available on `PATH`:

```sh
cargo tauri build --bundles rpm -- --locked
```

The UI build checks that each local script and stylesheet referenced by the
emitted HTML exists in the packaged output. Qualification on Fedora 44 covered
the locked Rust tests and Clippy, UI asset build, release RPM payload/dependency
inspection, RPM install/upgrade/downgrade/uninstall/reinstall, and installed
backend launch outside the source checkout. The final packaged-product
cross-session-bus proof is `main-current-product-crossbus.json`; it uses a
synthetic owner and verifies one host/adapter across the second bus, not real
Python gateway ownership. Real gateway process ownership and stop behavior
were checked separately against the installed backend using disposable config
and state.
See the per-run evidence under
`.assets/validation/fedora-tauri-2026-10-23/native-*` and the corresponding
Python/frontend qualification records for exact commands and results.
The no-tray notification-before-minimize primitive was separately exercised
with the isolated native P1 probe and local WebKit fixture; its raw event-order
logs are in `native-p1-probe/run/no-host-hide-confirmed/`. These primitive
probe results are separate from the final packaged-product cross-bus proof and
the installed Python backend qualification.

## Icon and launcher update (RPM 0.3.0-3)

RPM release 3 uses the packaged Nanobot icon for the GTK setup/chat windows and
the tray. Its desktop entry matches GTK's `nanobot-desktop` window class so KDE
can associate the running window with the launcher. The release RPM SHA-256 is
`08d05e7a3c044c50f7c1ab145c4af5a92671ddc54daf77ad176f4a1b8ac5a27f`.

The update passed all 12 native tests, Clippy with warnings denied, and the
optimized release build. After upgrading the installed RPM on Fedora 44, the
running window's X11 `WM_HINTS` icon pixmap and the tray image were both checked
against the packaged 512-pixel Nanobot icon. The staging gateway remained
healthy and configuration/preferences hashes were unchanged.

On this device the obsolete `Nanobot Personal Assistant` browser launcher was
moved out of the applications directory and retained under
`~/.local/share/nanobot-desktop/legacy-launchers/`. After refreshing KDE's
application cache, only **Nanobot Desktop** remained visible. The package does
not alter other users' launchers. Update evidence is retained under
`.assets/validation/fedora-icons-2026-10-24/`.

## KDE Wayland icon association (RPM 0.3.0-4 source change)

RPM 0.3.0-3 sets the GTK window icon and a `StartupWMClass`, but KDE Wayland
associates a window with its launcher by the GTK app id / window class
(`nanobot-desktop`) matched against the desktop-file basename, not
`StartupWMClass`. The installed `Nanobot Desktop.desktop` therefore fell back to
the default WebKit window icon.

RPM 0.3.0-4 ships an additional `NoDisplay` entry named
`nanobot-desktop.desktop` (same `Exec`/`Icon`) via `bundle.linux.rpm.files`, and
uses the lowercase `StartupWMClass=nanobot-desktop` in the visible template. The
alias associates the Wayland window without adding a duplicate menu entry.

The same alias can be installed immediately for 0.3.0-3 at
`~/.local/share/applications/nanobot-desktop.desktop`; refresh with
`update-desktop-database ~/.local/share/applications` and
`kbuildsycoca6 --noincremental`, then restart Nanobot Desktop.

The rebuilt 0.3.0-4 RPM was produced and installed on 2026-10-28. With no
per-user desktop overrides present, the running KDE Wayland window shows the
Nanobot titlebar icon from the packaged launchers alone. RPM SHA-256:
`746aaf72f0190f8f777a826092c86bb8dc58403e716a653c036677133d82d05c`.

## Initial installed GUI evidence (RPM 0.3.0-1)

The initial Fedora 44 RPM was exercised outside the source checkout with the
installed backend. The installed RPM payload SHA-256 was
`81ddc49a2bfa4c5a83fa3ceb8677c3b59fd8a5453a136e37b998ebe45315012b`; its
installed desktop executable SHA-256 was
`6420c6782d676992030d41a1e24cfdee035f9f28f46fae1e8fca9cec080dc9aa`. A held
response continued while the chat was hidden; after
tray restore, its final text, the unsent `DRAFT_RETAINED` composer text, and
`attachment.txt` were still present in the same window and WebKit process. A
separate five-minute hidden interval kept the same host and gateway processes
while the background process and one scheduled occurrence completed. Restoring
the chat took about 0.46 seconds. Ten additional close/restore cycles kept the
same host window and five child PIDs; the host window left the X client list
while hidden and returned once per restore.

Resource observations on that controlled workload were 359.3 MiB desktop PSS
for the first visible 60 seconds, 372.2 MiB at 300 seconds hidden, and 376.2 MiB
after restore. The hidden samples include the fixture work; backend PSS was
110.5 MiB at idle and about 175 MiB after that work. The six desktop processes
remained stable during the measured windows. These are observations from one
local fixture, not a memory target; the WebKit views remain mounted while
hidden. See `main-resource-analysis.md` for the sampling details.

For the close-notification timeout, the owned WebKit renderer was paused for
4.35 seconds. The window stayed visible, the tray menu remained responsive
(DBusMenu `GetLayout` completed in 0.54 ms), and resuming the renderer did not
allow a late callback to hide the window. In a separate renderer-exit check,
the selected WebKit child was killed; the native window and tray stayed
reachable and the selected gateway remained alive. The shell does not expose
the private WebKit callback error value, so that check records native fallback
behavior rather than asserting an internal error code. Detailed retained-chat,
cycle, timeout, and renderer-exit records are in `native-retention-restore.json`,
`native-hide-show-cycles.json`, `native-renderer-timeout.json`, and
`native-renderer-exit.json`. After an intentional renderer kill, the GTK window
remained visible but its chat content was blank. The available recovery is
**Quit Nanobot** from the tray, then relaunch and reconnect; automatic chat
reconstruction after a renderer crash is deferred.

An actual tray **Quit Nanobot** during a held streaming run stopped the selected
gateway once, recorded the durable run as cancelled, and exited the host and
its children. The original gateway process exited about 0.57 seconds after the
tray action. The run is recorded in `main-active-quit-clicked.json` and
`main-active-quit.json`.

The five protocol fault scenarios use a synthetic owner to exercise the Rust
actor's first-response loss, delayed start, and uncertain-stop paths. They are
not evidence of Python ownership. Separate installed-wheel tests prove real
backend process identity, same-handle reattachment, graceful stop, and
no-respawn behavior.
