---
name: mobile-dogfood
description: Explore an already-installed mobile app on a local iOS simulator or Android emulator, find bugs, and capture reproducible evidence with agent_device screenshots and notes.
homepage: https://agent-device.dev
metadata: {"nanobot":{"emoji":"🧭","requires":{"bins":["npx","xcrun","adb"]}}}
---

# Mobile Dogfood

Use this skill when the user wants exploratory QA, a bug hunt, or a broad pass over a mobile app on a local simulator or emulator.

Current scope is intentionally narrow:

- local iOS simulators
- local Android emulators
- app already installed

If the user needs physical devices, remote host control, install flows, or replay suites, stop and say the current Nanobot integration needs an upgrade for that path.

## Output Layout

Store artifacts under:

```text
artifacts/mobile-dogfood/<run-id>/
```

Recommended files:

- `report.md`
- `screenshots/initial.png`
- `screenshots/issue-001-step-1.png`
- `screenshots/issue-001-result.png`

## Workflow

1. Initialize an output folder and note the target app and platform.
2. Open the app and orient with a screenshot plus `snapshot -i`.
3. Move through major screens and core flows.
4. When an issue appears, stop and capture evidence immediately.
5. Write the issue down before continuing.
6. Close the session at the end.

## Orient

Start with visible evidence and navigation anchors:

```text
agent_device(args=["open", "Settings", "--platform", "ios"])
agent_device(args=["screenshot", "artifacts/mobile-dogfood/20260408-settings-ios/screenshots/initial.png"])
agent_device(args=["snapshot", "-i"])
```

Map the top-level navigation before going deep.

## Exploration Rules

- Re-snapshot after every meaningful mutation.
- Use `diff snapshot -i` after nearby UI transitions when it helps confirm changes.
- Prefer refs for exploration and screenshots for evidence.
- Check both functional behavior and visual layout quality.
- Use logs only if the behavior looks suspicious and you need diagnosis.

Useful commands during exploration:

```text
agent_device(args=["snapshot", "-i"])
agent_device(args=["diff", "snapshot", "-i"])
agent_device(args=["screenshot", "artifacts/mobile-dogfood/20260408-settings-ios/screenshots/screen-1.png"])
agent_device(args=["get", "text", "@e4"])
```

## Evidence-First Issue Capture

When you find a bug, stop and gather repro proof before moving on.

For interactive issues:

1. Capture the pre-action screen.
2. Perform the action.
3. Capture the broken result screen.
4. Write short reproduction steps immediately.

Example:

```text
agent_device(args=["screenshot", "artifacts/mobile-dogfood/20260408-settings-ios/screenshots/issue-001-step-1.png"])
agent_device(args=["press", "@e7"])
agent_device(args=["wait", "1000"])
agent_device(args=["screenshot", "artifacts/mobile-dogfood/20260408-settings-ios/screenshots/issue-001-result.png"])
```

For static or on-load issues, a single screenshot can be enough.

## Reporting

For each issue, record:

- short title
- severity: critical, high, medium, or low
- category: visual, functional, ux, content, performance, diagnostics, permissions, accessibility
- screen or route
- concise repro steps
- screenshot evidence path

Keep the report grounded in observed runtime behavior only.

## Severity Guide

- `critical`: blocks core workflow or crashes the app
- `high`: major feature broken with no practical workaround
- `medium`: notable failure or friction with a workaround
- `low`: cosmetic or minor polish issue

## Close Out

- Summarize total findings and highest-risk issues.
- Mention what platform was checked.
- Close the active session.

```text
agent_device(args=["close"])
```
