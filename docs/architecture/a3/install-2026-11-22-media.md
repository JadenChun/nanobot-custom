# Install receipt 2026-11-22 (media streaming)

Release snapshot `7da0c001f9a26d901c77f1d7d02736b9b2a5492b` with the session
`01a2029b` media-load fixes carried as uncommitted work in the shared checkout.
Linux desktop and the connected Android phone.

## Artifacts

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Frozen source | `~/.local/share/nanobot-release/kohaku-20261122-media.l8YzHl/source` | `source.sha256` `5a6937227fd35cbd9b6634fe2686e568b9458400ed70e16dafef0405ab60f0c0` |
| Linux wheel | `~/.local/share/nanobot-release/kohaku-20261122-media.l8YzHl/wheel/nanobot_ai-0.3.0-py3-none-any.whl` | `c7fbc72ae799341a2838c307c7286500c669e7ced269b7a1c9ee9b3d9b172901` |
| Android APK | `~/.local/share/nanobot-release/kohaku-20261122-media.l8YzHl/android/nanobot-mobile-release.apk` | `6e91dbfa7234af47362eaadfb179f43008b0ce3b0fb7e46fdf52d6feb31c754d` |

The Android APK was built with `mobile/scripts/build-release.sh` from the frozen
source and signed with the existing release key. `apksigner` verifies v2 and v3
signatures, and the signer certificate matches the installed app:
`8c30d04957e9724d1668e585890913b0581676c9460789f658c4ed9f84ba0c81`.

## Desktop

- Release dir: `~/.local/share/nanobot-release/kohaku-20261120-001350`
- Interpreter: `.../kohaku-20261120-001350/venv/bin/python`
- Selection: `~/.config/nanobot-desktop/preferences.json` (`interpreter_path`).
  The value does not change because the release directory is reused.
- Previous tree: `.../kohaku-20261120-001350/venv.pre-media-20261122`
- State: `~/.nanobot-staging`, keyed to the config path, so the switch does not
  move sessions, media, or permission state.
- Activation waited for the running chat to finish. The only leased turn saved
  its final stage before the gateway stopped, and no chat is now busy or flagged
  for recovery. Activation log: `/tmp/opencode/kohaku-activate-media.log`.

Verified after restart:

| Check | Result |
| --- | --- |
| Management health | `ready`, `websocket: running` |
| WebUI root | 200, byte-identical to the installed `index.html` (also gzip) |
| WebUI entry bundle | `index-DfZ1MLwc.js`, byte-identical over HTTP |
| Removed mobile control | `Confirm Full on this device` absent from the bundle |
| Installed module hashes | match the frozen source for the 5 changed modules |
| Config and preferences | SHA-256 unchanged across the swap |
| MCP ComfyUI capabilities | 6 registered |

## Android

- Installed with `adb install -r`; `lastUpdateTime` moved to `2026-11-22 18:14:17`.
- Device APK SHA-256 equals the verified file SHA-256.
- Cold launch reached `MainActivity`, the WebView mounted, pair and node state
  reached `ready`, and the WebSocket opened. No fatal exception was logged.
- Device: `00052141U004256`, Nothing Phone (2a).

## Not verified

- No paired media playback or seek was exercised on the phone after the update.
- The live gateway was not asked to serve a signed media URL. The range and
  cap behaviour was checked directly against the installed
  `serve_signed_media`: `206` with `Content-Range` for a range, `413` for a
  whole body above the cap, and an 8 MiB bounded range on a 20 MiB file.

## Rollback

Quit Kohaku, then move `venv` aside and rename `venv.pre-media-20261122` to
`venv`. On Android, install the previous signed APK with `adb install -r`.
