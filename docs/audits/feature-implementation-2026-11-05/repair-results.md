# Feature repairs — batches 1–3

Status: the authorized media, skills/clipboard and runtime integration batches
are implemented and locally validated on `codex/feature-audit-repairs`, based on
`364259daed2c11700d4064e7ca58743fd037b0b0`. Evidence below was captured before
commit packaging. No publication or external delivery was performed.

This report preserves the bounded acceptance of batches 1–3, which retained
preexisting failures on this host and did not run the full Python suite.
The subsequent [macOS repair results](./macos-repair-results.md) address those
failures and record the passing full backend run, final fixture qualification
and independent review. Strict-type debt, release/platform qualification,
architecture-audit completion and migration P6 closure remain open.

## Changes and supported contracts

### 1. Generated media — F-006/F-007

- Session and library pagination share a generation-scoped loader. Closing,
  refreshing, changing chat or changing credentials invalidates pending requests.
  Stale responses cannot update the current groups, cursor, totals or loading
  state. A synchronous guard rejects duplicate rapid page requests; failed pages
  retain their cursor for retry.
- The all-sessions API now continues across session boundaries and within a
  session. It reads at most 200 session indexes/transcripts per request, with
  opaque `[session_key, group_offset]` cursors. An empty batch can still return
  continuation. The bound is on session reads; directory enumeration remains.
- Library responses expose `totalScope: "page"`, `complete`, `sessionsScanned`
  and `next`. The client accumulates returned artifact counts and labels partial
  results as loaded. Single-session API totals retain their existing semantics;
  the gallery labels the visible media count as loaded while more pages remain.
- Normal media signing and containment checks remain authoritative. Pagination
  traverses a live library, not an immutable snapshot of concurrent asset edits.

Evidence: 21 media backend tests, including 0/200/205-session traversal,
within-session continuation, a full empty batch, malformed cursors and duplicate
protection. Thirteen frontend delayed-response cases cover refresh, chat/token
changes, close/reopen, retries and stale finalizers.

### 2. Skills and clipboard — F-008/F-009

- Installed skills distinguish loading, failed transport, empty success and
  restricted/incomplete sources, with an explicit retry action. A failed refresh
  retains a successful catalog only for the same credentials, including its
  completeness metadata.
- Credential changes clear the visible prior catalog and invalidate old fetches.
  Mutation events capture their originating credentials before awaiting results;
  older requests cannot overwrite a newer mutation. Unscoped events can request
  a fresh current-auth fetch but cannot install supplied catalog rows.
- Mixed file/plain-text paste accepts supported attachments and permits the
  textarea's native text insertion once, including selection replacement and
  caret placement. File-only paste remains consumed; the whitelist is unchanged.

Evidence: hook and component tests cover initial/refresh failure, retry,
restricted sources, credential rotation, mutation races and mixed paste through
`userEvent.paste`. These are DOM tests; native browser/OS clipboard combinations
were not qualified.

### 3. Runtime integration — F-001–F-004

**Scheduled automation.** The existing explicitly enabled gateway cron owner
now accepts supported manual requests through `CronService`. Admission claims an
occurrence synchronously and retains the execution task under `CronRuntime` for
shutdown and error handling. The existing authenticated WebSocket mutation path
is used; HTTP mutation and `force=True` fences remain.

The management view joins scheduler next-fire state, latest occurrence outcome,
pending work and durable delivery status. The UI enables Run now only when the
server advertises support and displays an unavailable reason otherwise.
Definition storage remains independent of runtime state.

The supported scheduler contract is fresh turns with Telegram receipt delivery.
Session-bound and WebSocket scheduled execution remain unavailable. Repeated
requests do not create duplicate work: an exact retained active request can be
acknowledged again, while changed authenticated admission inputs, including a
different clock, return a conflict. This does not promise unconditional HTTP
success when replaying a request ID.

**Local triggers.** `gateway.localTriggers` defaults to false. Enabling it and
restarting installs one supervised consumer of the shared workspace store. A
workspace file lease excludes competing recovery consumers. Queued messages
reach normal session coordination, wait behind a busy foreground turn and
recover interrupted claims. Cancellation requeues the current claim and clears
effective availability. The consumer stops before waiting for other runtime
owners during shutdown. Disabled mode preserves pending queue files; enqueue
and `/trigger` report the activation requirement. Delivery remains at-least-once.

**Goal and runtime tools.** Normal tool construction now includes `create_goal`,
`update_goal` and configured `my`. Explicit `/goal` authorization still controls
goal creation/replacement, and `my` remains read-only unless its existing write
option is enabled. Nonpersistent requests neither load nor mutate durable goal
state. Existing temporary-session isolation tests pass with these tools installed.

**Heartbeat/Dream.** Settings expose configured intent separately from effective
availability. Periodic execution remains inactive; commands, startup output and
guides say so. Manual `/dream` remains available. These corrections close the
availability mismatch, not periodic execution as a feature.

Evidence uses normal gateway composition with controlled local provider and
Telegram adapters. Tests exercise actual due/manual/trigger agent turns, status
and receipts, inactive mode, a busy chat, restart recovery, a competing consumer
and cancellation/shutdown. No live Telegram or model service was contacted.

## Verification results

Host: Darwin 27.0.0 arm64; Python 3.13.2, pytest 9.1.1, Bun 1.3.10. Optional
Telegram test dependencies were installed before freezing the repair baseline.
The accepted batches 1–3 evidence uses that unchanged frozen inventory; the
later installer mutation and restoration are recorded in the macOS follow-up.

| Check | Result | Retained evidence |
|---|---|---|
| Media backend | **21 passed** | `media-tests.log` |
| Final affected cron/goal/runner/registration/gateway selection | **1,588 passed** | `backend-final.log`, `backend-final-command.json`, exit 0 |
| Full frontend | **95 files, 1,468 passed** | `frontend-acceptance.log` |
| TypeScript/Vite production build | **Passed**; bundle-size warning remains | `frontend-build-final.log` |
| Frontend ESLint | **Passed**, zero warnings | `frontend-lint.log` |
| Python lint, including changed tests | **Passed** | `ruff-acceptance.log` |
| Full-source BasedPyright | **1,001 errors before and after**, 537 files; both exit 1 | `strict-before.json`, `strict-final-verified.json` |
| Diagnostic comparator | **PASS_NO_NEW_DIAGNOSTICS**; introduced 0, resolved 0 | `types-final-comparison.json` |
| Whitespace/diff check | **Passed** | `diff-check.log` |

The broad backend selection ran before the last test updates and regression
fixes and returned **7,721 passed, 38 failed, 16 skipped** in 531.61 seconds.
Its command is recorded in `backend-acceptance-command.json`, with failures in
`backend-acceptance.log`. The 38 failures
were resolved or attributed as follows:

- Ten tests had loaded earlier missing-tool assertions before normal goal/`my`
  registration assertions were updated. The final affected selection passes.
- Five cron tests exposed a regression from reading unavailable definition
  storage before rejecting inactive/forced execution. Rejection now precedes
  lookup, preserving the existing fence and stored bytes.
- One temporary-turn test exposed durable cache creation by the newly installed
  goal runtime-context provider. Nonpersistent context/tool requests are now
  rejected before session lookup; existing and new isolation tests pass.
- The other **22 failure node IDs reproduce on both the original revision and
  final implementation** under the same dependency environment.

For the comparison, the original revision was exported into `original-revision/`
and the original 38 failing node IDs were run against its original tests:
**16 passed, 22 failed** in 6.83 seconds. The remaining 22 were then rerun on
the final source: **22 failed** in 5.21 seconds, with an identical node-ID set.
They concern unavailable sandbox/native-process proof, native/full-access
expectations, remote peer authentication and its dependent workflows, and a
host path-length fixture. Reproduction establishes that they predate this
batch; it does not independently reduce every downstream assertion to one root
cause. Exact names are in the repair snapshot and raw logs.

One attempted rerun of all original 38 node IDs could not collect a renamed
registration test (`final-failures.log`, exit 4). The final affected selection
and the corrected 22-node comparison above supersede that attempt. The broad
selection was not rerun in full after the final fixes; its counts must not be
presented as final whole-suite acceptance or added to overlapping selections.

Reproduce the final affected selection from the repository root:

```bash
.venv/bin/python -m pytest \
  tests/cron tests/agent/tools/test_long_task.py \
  tests/agent/test_p5_stage2_engine.py \
  tests/cli/test_gateway_local_triggers.py \
  tests/cli/test_gateway_scheduled_execution.py \
  tests/cli/test_gateway_automation_boundary.py \
  tests/agent/tools/test_runtime_control.py \
  tests/agent/tools/test_self_tool_runtime_sync.py \
  tests/agent/test_loop_save_turn.py \
  tests/agent/test_loop_runner_integration.py tests/tools/test_tool_loader.py \
  -q --disable-warnings --durations=10 --basetemp=/tmp/nfr-final-check
```

## Frozen evidence and review

Raw evidence is retained locally under
`/tmp/nanobot-feature-repairs-20261105.3kdy8inl`, also recorded in
`.git/feature-audit-repairs-evidence`. The compact
[repair snapshot](./repair-snapshot.json) records report hashes, results and
the exact preexisting failure set; it does not replace the raw artifacts.

- Immutable baseline: `source-before/`, `source-before.json`,
  `environment-before.json`, `strict-before.json`.
- Final source/test copy and hashes: `source-final/`, `source-final.json`
  (**1,922 files**); checker scope is **537 production Python files**.
- Final environment/command/policy: `environment-final.json`; installed package
  versions exactly match baseline. `pyproject.toml` and checker policy are
  unchanged. Source hashes were rechecked after full checking.
- Final checker report: `strict-final-verified.json`; comparison uses the
  original repository root for both reports because their diagnostic paths
  reference that root. Earlier package comparison reports remain retained.
- Final frontend count-label assertions were validated with the full frontend
  suite/build/lint. An intermediate assertion incorrectly expected a session-wide
  total where the UI counts visible media; it was corrected to verify loaded
  counts in both paginated views. Final source copies include the updated test;
  all 537 checker inputs remain byte-identical. Earlier logs are retained.

Implementation and integrated self-review were performed directly, without an
independent reviewer or delegated agents. Review covered credential/request
races, bounded media continuation, admission and task ownership, inactive-mode
storage preservation, trigger cancellation/recovery and every broad-run failure.
The two discovered regressions were repaired and retested rather than waived.

## Follow-up disposition and remaining work

- F-005 and the original 22 retained failures were addressed by the
  [macOS reliability follow-up](./macos-repair-results.md): **15 passed,
  7 Linux-only skips**. Its default full backend run passed **17,001 tests**
  with **75 skips**; the final test-only portability adjustment has separate
  focused evidence. The earlier counts in this report remain historical.
- Native command confinement, whole-tree process proof and full-access workflow
  qualification on supported platforms; preserve fail-closed behavior.
- Periodic Heartbeat/Dream and unsupported scheduled-session/transports, if those
  capabilities are pursued as separate work.
- Native clipboard/browser/mobile checks, live external channels/providers,
  install/update/restart qualification, profiling and architecture/ownership work.
- Existing strict-type debt and the required unchanged strict/full-suite checks
  before migration P6 can close. The diagnostic baseline was not refreshed.
