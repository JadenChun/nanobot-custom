---
name: webapp-testing
description: Test real web apps with a visible browser, capture screenshots or recordings, and check desktop/mobile UI bugs with agent_browser.
homepage: https://agent-browser.dev
metadata: {"nanobot":{"emoji":"🧪","requires":{"bins":["npx"]}}}
---

# Web App Testing

Use this skill when the user asks to test a website or web app, reproduce a browser bug, validate UI, check responsive layout, watch browser actions live, or capture browser evidence.

## Tool Selection

- Use `web_search` to discover public pages, release notes, issue threads, or documentation.
- Use `web_fetch` to read a known URL when you need article or doc content, not browser interaction.
- Use `agent_browser` for real browser actions, authenticated flows, JavaScript-heavy apps, visual QA, responsive testing, and anything the user wants to watch live.
- Use `read_file` on screenshot image files saved in the workspace. `read_file` returns local images as native image blocks, so you can inspect the actual UI.
- Use `message(media=[...])` to send screenshots or recordings to the user when needed.

## Artifact Location

Store browser test outputs under a dedicated workspace folder:

```text
artifacts/browser-tests/<run-id>/
```

Use a short timestamped run id such as `20260408-login-flow` or `20260408-143000-homepage-check`.

Recommended files inside each run directory:

- `desktop-annotated.png`
- `mobile-annotated.png`
- `session-review.webm`
- `trace.json`
- `notes.md`

This keeps browser QA artifacts separate from normal workspace files and makes cleanup easy.

To find the active artifact root from the CLI:

```text
nanobot artifacts path
```

To remove old browser test artifacts later:

```text
nanobot artifacts clean
```

## Priority Order

1. Prefer a visible browser when the user wants to watch progress.
2. Cover both desktop and mobile views for UI checks.
3. Capture screenshots for visual evidence.
4. Treat screen recording as best-effort and lower priority than completing the test visibly.

## Visible Browser Workflow

Start the observability dashboard before testing if the user wants live progress:

```text
agent_browser(args=["dashboard", "start"])
```

Then open the app in a visible browser window:

```text
agent_browser(args=["--headed", "open", "https://example.com"])
agent_browser(args=["wait", "--load", "networkidle"])
```

If the user is local and wants to watch progress, tell them to open `http://localhost:4848` after the dashboard starts.

If the browser binary is missing, install it once and retry:

```text
agent_browser(args=["install"])
```

If the user wants the agent to drive an existing Chrome session they can also inspect directly, prefer `--auto-connect` or `--cdp` over launching a new isolated browser.

## Standard Test Flow

1. Open the target app visibly and wait for it to settle.
2. Exercise the primary user flow on desktop.
3. Capture an annotated screenshot and inspect it with `read_file`.
4. Check browser console, page errors, and relevant network requests.
5. Switch to a mobile device or viewport.
6. Re-run the critical path on mobile.
7. Capture another screenshot and inspect it.
8. Report findings separately for functional failures and visual bugs.

Do not claim a flow passed purely because the clicks succeeded. Check what actually rendered.

## Desktop Checks

Use a desktop viewport such as:

```text
agent_browser(args=["set", "viewport", "1440", "900"])
```

Check for:

- Missing or overlapped buttons, inputs, dialogs, and menus
- Clipped text, broken wrapping, and unreadable spacing
- Off-screen content, broken scroll containers, or unexpected horizontal scroll
- Layout jumps after loading, hydration, or modal open/close
- Broken images, icons, or empty states
- Sticky headers/footers covering content
- Console errors, page errors, and failed XHR/fetch requests relevant to the flow

## Mobile Checks

Prefer device emulation for mobile UI validation:

```text
agent_browser(args=["set", "device", "iPhone 14"])
```

Or use an explicit mobile viewport if needed:

```text
agent_browser(args=["set", "viewport", "390", "844", "3"])
```

Re-run the critical path and check for:

- Horizontal overflow or content cut off at the edges
- Menus, drawers, sheets, and modals that do not fully fit on screen
- Tap targets that are too small or too close together
- Inputs hidden by the virtual keyboard or sticky UI
- Broken responsive reflow, stacked elements overlapping, or missing controls
- Text truncation and CTA buttons falling below the fold unexpectedly

## Screenshot Pattern

Capture screenshots into workspace files so you can inspect them and preserve artifacts:

```text
agent_browser(args=["screenshot", "artifacts/browser-tests/20260408-login-flow/desktop-annotated.png", "--annotate"])
read_file(path="artifacts/browser-tests/20260408-login-flow/desktop-annotated.png")

agent_browser(args=["screenshot", "artifacts/browser-tests/20260408-login-flow/mobile-annotated.png", "--annotate"])
read_file(path="artifacts/browser-tests/20260408-login-flow/mobile-annotated.png")
```

Use annotated screenshots when you need both visual confirmation and a mapping back to actionable elements.

If a run directory does not exist yet, create it first before saving artifacts.

## State, Logs, and Debugging

Useful commands during testing:

```text
agent_browser(args=["snapshot", "-i", "--json"])
agent_browser(args=["console"])
agent_browser(args=["errors"])
agent_browser(args=["network", "requests", "--type", "xhr,fetch"])
agent_browser(args=["inspect"])
agent_browser(args=["trace", "start"])
agent_browser(args=["trace", "stop", "trace.json"])
```

Use `inspect` when you need Chrome DevTools open for the active page. Use traces and network inspection when the UI looks wrong but the root cause may be data, timing, or JavaScript errors.

## Screen Recording

Screen recording is optional and lower priority than visible progress plus screenshots, but it is useful for long or flaky flows.

Best-effort recording flow:

```text
agent_browser(args=["record", "start", "artifacts/browser-tests/20260408-login-flow/session-review.webm"])
...
agent_browser(args=["record", "stop"])
```

- Start recording before the critical flow begins.
- Stop recording after the flow completes or the bug is reproduced.
- If recording is unavailable or fails, continue testing with screenshots and the live dashboard.
- Mention the saved recording path in the final report, or send it via `message(media=[...])` if the user wants the artifact.

## Reporting

- Separate desktop findings from mobile findings.
- Separate functional failures from visual or layout issues.
- Cite the step, page, and artifact file used as evidence.
- If no issue is found, say exactly what flows, viewports, and artifacts were checked.