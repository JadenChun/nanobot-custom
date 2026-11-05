# Feature audit evidence — 2026-11-05

This directory supports the [findings report](../../feature-implementation-audit-2026-11-05.md).
It contains a source/environment [snapshot](./snapshot.json) and small
exploratory probes. It is not a new CI suite or a release qualification record.

The sections below retain the original read-only audit. Authorized implementation
changes and final verification are recorded in
[repair-results.md](./repair-results.md), with a separate
[repair snapshot](./repair-snapshot.json). The original snapshot is unchanged.
The later [macOS reliability results](./macos-repair-results.md) and
[macOS repair snapshot](./macos-repair-snapshot.json) record the authentication
repair, retained-failure disposition, full backend run and final fixture timing.

Snapshots retain their state at validation time, before commit packaging. Their
uncommitted-state fields describe that capture; the frozen evidence is unchanged.

The audited source was `364259daed2c11700d4064e7ca58743fd037b0b0`. Existing
dependencies were used; neither lockfile was modified. The source digest in the
snapshot covers 1,141 tracked files in `nanobot`, `webui/src`, and `scripts`,
using the algorithm recorded in that file. It was checked again after the audit
documents were written. Docs and probe fixtures are outside that digest.

## Reproduce the feature probes

From the repository root, with the existing Python/WebUI development environment:

```bash
.venv/bin/python docs/audits/feature-implementation-2026-11-05/run_probes.py
```

The runner temporarily places the saved probe sources inside the existing test
trees so they use the project's fixtures and Vitest setup. It refuses to
overwrite existing files, removes its temporary sources afterward, and keeps
output in a short, disposable `/tmp/nfa-*` directory. It does not install
dependencies or edit implementation files. Its exit code is nonzero when a
probe suite fails.

At the audited revision the expected observation is **four backend and six
frontend failed assertions**, all after successful setup:

| Probe | Observed result | Finding |
|---|---|---|
| Default admitted tool composition exposes goal tools | `create_goal` and `update_goal` absent | F-003 |
| Enabled runtime inspection exposes `my` | `MyTool.enabled(context)` true, registry lacks `my` | F-003 |
| All-sessions API includes session 201 | HTTP 200; `total=200`, 200 groups, `next=null` | F-007 |
| Manual run reaches a gateway-configured service | HTTP 409 with the definition-only execution fence | F-001 |
| Old chat page resolves after switching chats | Current groups become `[b-first, a-old-more]` | F-006 |
| Old chat page resolves after refresh | Current groups become `[refreshed-first, stale-more]` | F-006 |
| Old global-library page resolves after refresh | Current groups become `[library-refreshed, library-stale]` | F-006 |
| Skills refresh fails after success | Existing skill disappears | F-008 |
| Initial skills fetch fails | Empty catalog still has `catalog_complete=true` | F-008 |
| Clipboard contains file and plain text | File callback fires; default text insertion is canceled; hook has no text insertion path | F-009 |

These are exploratory checks of advertised/desired behavior, not immutable
implementation specifications. For example, a corrected unavailable-state UI
may intentionally retain HTTP 409, a bounded media library may introduce a new
session continuation contract, and a skills repair should represent transport
errors separately from catalog trust/completeness. The report's acceptance
criteria describe the required user behavior; repair tests should follow the
chosen public contract.

The manual-run probe calls the real handler and `CronService` after gateway
configuration. Its status callback reports enabled, but no `GatewayCron` task or
execution/delivery owner is started. It verifies the unconditional management
fence, not actual scheduled execution. All backend probe data lives in fixtures.
The clipboard check is at hook level; it is not a browser clipboard test.

## Existing frontend validation

Commands ran in `webui` before temporary probes were added:

```bash
bun run test
bun run build
```

Results: **94 test files and 1,444 tests passed** (23.52 seconds). TypeScript and
Vite production build passed; Vite reported a bundle-size warning. The main
JavaScript bundle was about 906.53 kB / 262.40 kB gzip. This is not a measured
startup or interaction-performance defect. Build output is ignored under
`nanobot/web/dist`; no tracked bundle change is included.

## Existing backend validation

Commands ran from the repository root. The available broad selection was:

```bash
.venv/bin/python -m pytest \
  tests/webui tests/cron tests/remote \
  tests/test_openai_api.py tests/test_openai_api_fork.py \
  tests/test_nanobot_facade.py tests/test_nanobot_facade_fork.py \
  tests/test_sdk_streaming.py tests/gateway \
  tests/cli/test_gateway_automation_boundary.py \
  --ignore=tests/cron/test_cron_channel_sender.py \
  -q --disable-warnings --durations=10 --basetemp=/tmp/nfa.eTByFP
```

Result: **2,119 passed, 39 failed, 1 skipped** (79.33 seconds). There was no
exclusion based on observed failures. An earlier attempt included
`tests/cli/test_gateway_scheduled_execution.py` and did not ignore the cron
channel-sender test; collection stopped because the optional `telegram` package
was missing. The available selection was first run with pytest's default
temporary root: **2,101 passed, 24 failed, 33 errors, 1 skipped**. Long macOS
UNIX-socket paths caused the setup errors. The short-root result above supersedes
that run for analysis; it remains failing.

The 39 failing cases occur in remote owner/edge/settings and mobile-linked file
preview tests, one gateway shell smoke test, three native permission tests, and
one directory-roots fixture. Confirmed relevant observations:

- Real owner IPC rejects the same-user client with `peer credentials are
  unavailable on this platform`; `socket.SO_PEERCRED` is absent on Darwin.
- Remote edge requests then return denied/invalid-access/unavailable responses.
- The gateway smoke reaches shell execution but receives `Sandbox requires a
  working bubblewrap installation` on this host.
- Native process-accounting expectations differ (`unknown_processes` and
  unlaunched-reservation cleanup); not every downstream failure was independently
  reduced to a minimal platform probe.
- The directory-roots test constructs a nested path beyond the host path limit.

Provider/tool/data selection:

```bash
.venv/bin/python -m pytest \
  tests/providers tests/context_repo tests/attachments \
  tests/tools/test_mcp_tool.py tests/tools/test_mcp_oauth.py \
  tests/command/test_builtin_dream.py tests/command/test_trigger_command.py \
  tests/triggers tests/tools/test_process_completion.py \
  tests/agent/test_migration_approval_admission.py \
  -q --disable-warnings --durations=10
```

Result: **2,199 passed, 1 failed, 3 skipped** (41.89 seconds). The failure is
`tests/context_repo/test_sandbox_write_policy.py::test_sandbox_file_reads_cannot_reach_gateway_process_environment`:
it expects `Host process state` rejection for Linux `/proc` paths, while this
Darwin host has no `/proc`. This does not demonstrate a Linux host-state bypass.

Runner/session/WebSocket selection:

```bash
.venv/bin/python -m pytest \
  tests/agent/test_runner_core.py \
  tests/agent/test_runner_tool_execution.py \
  tests/agent/test_runner_safety.py tests/agent/test_runner_errors.py \
  tests/agent/test_runner_hooks.py tests/agent/test_runner_progress_deltas.py \
  tests/agent/test_runner_reasoning.py tests/agent/test_loop_runner_integration.py \
  tests/agent/test_context_budget.py tests/agent/test_source_pressure_compaction.py \
  tests/agent/test_task_cancel.py tests/agent/test_task_cancel_fork.py \
  tests/session/test_turn_continuation.py tests/session/test_canonical_followups.py \
  nanobot/channels/websocket/tests \
  -q --disable-warnings --durations=10 --basetemp=/tmp/nfa-core-20261105
```

Result: **639 passed, 3 failed** (23.14 seconds). Failing WebSocket tests:

- `test_handle_file_preview_probe_reports_binary_file_as_unavailable`
- `test_handle_file_preview_allows_paths_outside_workspace_in_full_access`
- `test_remote_owner_full_message_requires_own_device_consent`

They fail with `full_access_unavailable` during permission setup. They are not
reported as successful browser/full-access qualification.

Registration/delegation/plugins/media/web-tool selection:

```bash
.venv/bin/python -m pytest \
  tests/tools/test_tool_loader.py tests/agent/test_fork_tool_factories.py \
  tests/agent/test_tool_loader_scopes.py tests/agent/test_subagent.py \
  tests/agent/test_subagent_lifecycle.py tests/agent/test_orchestrator_harness.py \
  tests/agent/tools/test_subagent_tools.py tests/agent/test_agent_plugins.py \
  tests/tools/test_image_generation_tool.py tests/agent/test_image_generation_hot_reload.py \
  tests/tools/test_web_search_tool.py tests/tools/test_web_fetch_security.py \
  tests/tools/test_web_fetch_security_fork.py tests/tools/test_web_fetch_blocked_hosts.py \
  tests/tools/test_web_fetch_jina_privacy.py tests/tools/test_web_fetch_tinyfish.py \
  tests/tools/test_web_fetch_url_sanitization.py \
  -q --disable-warnings --durations=10 --basetemp=/tmp/nfa-tools-20261105
```

Result: **323 passed** (3.16 seconds).

The selections provide local contract evidence, not unique test totals or live
integration certification. No full Python suite, unchanged strict type-check,
coverage run, or performance profile was performed. No P6 completion is claimed.

## Source traces used for non-probe findings

For F-002, search production source for `run_local_trigger_queue` and
`local_runner`, then inspect the gateway task composition. Only the runner
definition is found; the gateway never installs it. Existing
`test_default_gateway_preserves_pending_automation_without_execution` includes
the pending-trigger case and passes in the broad selection.

For F-004,
`test_foreground_gateway_does_not_advance_dream_or_activate_configured_automation`
exercises all combinations of Dream/Heartbeat flags. The gateway composition
and settings projection establish the difference between configured and
effective state. These are deliberate inactive-mode tests, not failed tests.

For F-005, existing `tests/remote/test_owner_ipc.py` and
`tests/webui/test_remote_access_settings.py` use real local sockets. Their
failures after the short-root rerun are the platform evidence; no credential
check was mocked out to make the tests pass.
