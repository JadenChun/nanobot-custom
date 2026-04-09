---
name: mobile-app-testing
description: Drive local iOS simulators and Android emulators with agent_device. Use when opening installed mobile apps, taking snapshots or screenshots, tapping, typing, scrolling, checking visible UI state, or reproducing app behavior on a local simulator/emulator.
homepage: https://agent-device.dev
metadata: {"nanobot":{"emoji":"📱","requires":{"bins":["npx","xcrun","adb"]}}}
---

# Mobile App Testing

Use this skill when the user asks to inspect, navigate, or test a mobile app on a local iOS simulator or Android emulator.

Current Nanobot guidance for `agent_device` is intentionally scoped to:

- local iOS simulators on macOS
- local Android emulators
- apps that are already installed on the selected simulator or emulator

The upstream `agent-device` project can also support physical devices, remote daemons, install flows, replay, and more. Treat those as out of scope unless the user explicitly asks for a later upgrade.

## Tool Selection

- Use `agent_device` for real mobile UI automation.
- Use `read_file` on screenshot image files saved in the workspace so you can inspect the rendered UI.
- Use `web_fetch` only for docs or public pages, not for app interaction.
- Use `message(media=[...])` when you need to send screenshot artifacts to the user.

## Default Loop

1. Pin the platform and target app before interacting.
2. Start with read-only inspection when possible.
3. Open the app.
4. Capture `snapshot` or `snapshot -i`.
5. Interact with refs from the current snapshot.
6. Re-snapshot after meaningful UI changes.
7. Capture screenshots when the user needs proof or visual QA.
8. Close the session when done.

Do not reuse stale refs after navigation, submits, modal open/close, or screen transitions.

## Artifact Location

Store mobile screenshots and notes under:

```text
artifacts/mobile-tests/<run-id>/
```

Use a short run id like `20260408-settings-check` or `20260408-login-bug`.

Recommended files:

- `initial.png`
- `result.png`
- `notes.md`

## Open-First Flow

When the app id or package is uncertain, prefer discovery before guessing:

```text
agent_device(args=["devices", "--platform", "ios"])
agent_device(args=["apps", "--platform", "ios"])
agent_device(args=["open", "Settings", "--platform", "ios"])
agent_device(args=["snapshot", "-i"])
```

Android example:

```text
agent_device(args=["devices", "--platform", "android"])
agent_device(args=["apps", "--platform", "android"])
agent_device(args=["open", "com.android.settings", "--platform", "android"])
agent_device(args=["snapshot", "-i"])
```

If no target is ready yet, boot it first:

```text
agent_device(args=["boot", "--platform", "ios"])
agent_device(args=["boot", "--platform", "android"])
```

## Read-Only First

Prefer the least invasive command that answers the question:

- Use `snapshot` when the task is to verify what is visible.
- Use `snapshot -i` when you need actionable refs such as `@e2`.
- Use `get text @eN` for exact text from a known target.
- Use `diff snapshot` after a nearby mutation when you need a compact change view.

Examples:

```text
agent_device(args=["snapshot"])
agent_device(args=["snapshot", "-i"])
agent_device(args=["get", "text", "@e4"])
agent_device(args=["diff", "snapshot", "-i"])
```

## Interaction Rules

- Prefer `press`, `fill`, `type`, `scroll`, and `wait` over raw coordinates.
- Use `fill @eN "text"` when replacing field contents.
- Use `press @eN` then `type "text"` when append semantics matter.
- Re-snapshot after mutations instead of assuming refs are stable.
- If the keyboard blocks the next step, dismiss it before navigating away.

Examples:

```text
agent_device(args=["press", "@e2"])
agent_device(args=["fill", "@e5", "user@example.com"])
agent_device(args=["type", " more"])
agent_device(args=["scroll", "down", "0.5"])
agent_device(args=["wait", "1000"])
```

## Screenshots

Capture workspace-local screenshots when you need visual confirmation:

```text
agent_device(args=["screenshot", "artifacts/mobile-tests/20260408-settings-check/initial.png"])
read_file(path="artifacts/mobile-tests/20260408-settings-check/initial.png")

agent_device(args=["screenshot", "artifacts/mobile-tests/20260408-settings-check/result.png"])
read_file(path="artifacts/mobile-tests/20260408-settings-check/result.png")
```

Use screenshots as the visual source of truth when the accessibility snapshot looks stale or incomplete.

## Common Guardrails

- Do not claim success only because a tap command succeeded.
- Do not invent package names, device ids, refs, or screen state.
- If the app is not installed, say so plainly instead of guessing alternate ids.
- If the task needs physical devices, remote daemon setup, or install/reinstall flows, say that the current Nanobot integration is scoped to local emulator control and call out that an upgrade is needed.

## Wrap Up

- Capture the final screenshot if evidence matters.
- Separate functional failures from visual issues.
- Mention platform checked: iOS simulator, Android emulator, or both.
- Close the session when the task is finished.

```text
agent_device(args=["close"])
```
