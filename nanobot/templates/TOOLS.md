# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## cron — Scheduled Reminders

- Please refer to cron skill for usage.

## agent_browser — Browser/Electron Automation

- Wraps `agent-browser` via `npx`
- Pass CLI args as a string array; use `["--help"]` first when unsure
- Timeout and output size are configurable via `tools.agentBrowser.*`
- For visible web-app testing, prefer `["dashboard", "start"]` and headed commands like `["--headed", "open", "https://example.com"]`
- Save screenshots to workspace files, then inspect them with `read_file` — image files are returned as native image blocks
- Session recording is available on recent `agent-browser` versions via `["record", "start", "session.webm"]` and `["record", "stop"]`

## agent_device — Mobile Device Automation

- Wraps `agent-device` via `npx`
- Pass CLI args as a string array; use `["--help"]` first when unsure
- Timeout and output size are configurable via `tools.agentDevice.*`
- Current built-in Nanobot guidance targets local iOS simulators and Android emulators
- Typical loop: `["devices", "--platform", "ios"]`, `["open", "Settings", "--platform", "ios"]`, `["snapshot", "-i"]`, `["press", "@e2"]`, `["screenshot", "artifacts/mobile-tests/run-1/screen.png"]`, `["close"]`
- Save screenshots to workspace files, then inspect them with `read_file` — image files are returned as native image blocks
