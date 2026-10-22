# P6 direct local-runner checkpoint — 2026-10-06

**PASS for the local process-test runner repair. Q1/P6 remain NOT_ACCEPTED.**
Runtime and D9/P7 remain NOT_AUTHORIZED. No deployment, provider/channel operation,
GitHub repository creation, commit, upload or CI dispatch occurred.

## Scope and ownership

The execution chat failed with provider `503 auth_unavailable`; the owner then
requested direct repair. Work moved to a direct loop in the main chat. Candidate:
`/Users/jaden/Documents/github/nanobot-harness-candidate`; the protected original
was not inspected or edited by this repair. All QA code/evidence is under
`/private/tmp/nanobot-p6-completion.rk7wje`.

Owner decisions: development-only, no previous deployed chat history to preserve,
local tests first and GitHub CI later. See the applicability section of the
[completion checklist](p6-completion-checklist.md). No existing production data or
old-deployment reader should be invented as a prerequisite for local work.

## Root causes and corrections

1. The finite Darwin test policy denied `file-read-data /`. The kernel reported
   that denial and a marker-only Python command exited with SIGABRT before any
   application import. Granting that exact directory made the identical command
   succeed. Secure descriptor walks also need the exact disposable-directory
   ancestors. Grants are literals, not recursive home/original access.
2. macOS `/bin/sh` delegates to `/bin/bash`, needed by Git's local-file transport.
   The finite executable list now binds both. System i18n resources and the
   exact `/dev/fd` directory support those trusted subprocesses.
3. Darwin can create an unconnected socket under `deny network*`. Negative probes
   now verify denied bind/connect operations using disposable Unix and loopback
   targets, rather than falsely demanding socket-construction failure. Existing
   file, executable, mapping and network restrictions remain tested.
4. Pytest reopens/duplicates capture descriptors. Active output now lives in
   writable temporary state and is archived into evidence after child cleanup.
5. A process group can briefly return an observation error while being reaped.
   The Git supervisor records and retries it; only confirmed absence passes.
   Persistent errors/residual groups and failed signals still fail cleanup.
6. One retained exec test expected two subagent cleanup calls after two `aclose`
   calls. The accepted lifecycle implementation shares one teardown task. The
   test now checks exactly-once cleanup and identical completed-task reuse,
   retaining its actual process-termination and empty-session assertions.

For that runner repair, the only candidate Python edit was
`tests/tools/test_exec_session_tools.py::test_agent_loop_shutdown_closes_exec_sessions`.
That runner repair changed no production Python, guard, checker settings, test selection or skip.
AST comparison verifies the rest of that test module is unchanged.

## Executed verification

| Check | Result |
| --- | --- |
| Six complete Git modules, collection and execution | 75 collected; 75 passed |
| Complete exec-session module | 41 collected; 41 passed; all 123 setup/call/teardown reports pass |
| Idle lifecycle and Stage 1 shutdown regression modules | 24 passed |
| Full Ruff (`nanobot tests conftest.py`) | PASS |
| Compile every inventoried Python file | 1,115 compiled |
| Full unchanged strict checker | FAIL: 2,574 errors / 463 files; diagnostic multiset unchanged from the retained quarantine-declaration result |
| Git prerequisite suite | Two current-source replays passed, including timeout/cancel/TERM-resistant/leader-exit cleanup |
| Exec prerequisites | Two current-source replays of all 11 cases passed, including real process lifetime, cancellation, fences and cleanup |

Raw authority is `direct-local-runner-acceptance.json`, with related records:

- `git-process-direct-root-r8/evidence/final.json` and `direct-root-r9` counterpart;
- `git-process-direct-tests-r3/evidence/final.json`;
- `exec-process-worker-root-exec-probes-c/evidence/final.json` and `-d` counterpart;
- `exec-process-worker-direct-exec-tests-r2/evidence/final.json`;
- `direct-source-{lint,strict,lifecycle}.json` and raw stdout/stderr;
- `direct-bootstrap-path-results.json`, `direct-bootstrap-root-result.json` and
  other intermediate diagnostic records.

All records retain their tested source identity. Later checkpoint/plan edits are
documentation-only and do not turn this into an accepted immutable release.
The runner may emit benign termcap/shell-selection or read-only pytest-cache
warnings. They are retained, not filtered from evidence.

## Limitations and retained incidents

Failed diagnostics remain intact. One intermediate exec replay correctly stopped
because its shared QA helper changed concurrently; it is not a pass. Its worker
and recorded children were verified absent before its stop sentinel was archived.
The earlier Git observation-error sentinel was likewise archived only after
verified process/group absence. Neither recovery killed unrelated processes.

The historical `admitted-fixture-scope-incident.json` remains part of the audit;
this checkpoint does not erase or redefine that earlier enumeration incident.
Tests qualify trusted local commands, not hostile escaped descendants, supervisor
hard crashes, Windows, production networking or live service behavior.

The immediate next work is the remaining Q1 source/type/test backlog. Do not
repeat Git/Exec environment design: these routes now run their complete modules.
Local artifact/install and representative startup/recovery work follows source
qualification; CI/native coverage remains explicitly deferred by the owner.
As with any source edit, refreshing the QA source manifest is required before a
new candidate-based run; the new checkpoint adds one documentation entry. Keep
the previous manifests/probe records as historical evidence rather than relabeling
them for the next source generation.

## Subsequent direct source repairs — 2026-10-06

The owner requested direct implementation of the remaining source errors. This
is a separate source-repair batch, not a claim that the earlier runner repair
changed production code or that Q1/P6 is complete. The development skill's direct
reproduce/fix/verify loop was used; no new worker graph or external service ran.

### Reproduced runtime defects

- The earlier tool-boundary repair separates positional internal context from
  schema-declared `context` inputs. Its two failing regressions became green.
- Delivery reconciliation leaked an opened directory descriptor if its subsequent
  stat failed. The new fault-injection test reproduced the leak before the fix.
- Callback storage leaked its lock descriptor when unlocking raised. A second
  fault-injection test now verifies closure even when the unlock fails.
- Closing `Nanobot(None, ...)` raised before runtime acquisition. Closing now
  safely releases the optional MCP owner once, including a repeated close.
- Copied-state preflight unnecessarily invalidated a stable copied tree when an
  unrelated sibling changed in a shared ancestor. A deterministic regression
  reproduced the failure. Shared ancestors now retain no-follow identity,
  ownership and mode checks; the copied root and descendants retain full change
  detection. Negative tests cover copied-content, ancestor-inode and permission
  changes. No runtime-admission or migration-activation guard was removed.

### Contract and validation repairs

- Receipt-quarantine decoding now exposes a checked `TypedDict`; exact keys,
  primitive types, canonical bytes, hashes and stat arrays are still validated
  before it is returned. Fifteen malformed-field cases retain the live receipt
  and fail closed. Validated optional seal and snapshot evidence is checked
  before use, rather than masked with `Any` or unchecked casts.
- Partial channel delivery is represented in the sender protocol without adding
  it to the settled-success union. Delegation/exploration wrappers retain
  `ToolOutcome` metadata, conform to tool metadata properties, and keep the
  planner's evidence text separate from internal context.
- Idle activity, cancellation, recovery, descriptor cleanup, checkpoint, SDK
  runtime and provider-settlement paths use explicit control-flow proofs for
  optional or branch-local state. Existing persistence ordering is retained.
- Legacy token/config normalization and run metadata boundaries keep unvalidated
  values as `object` until validated; typed default factories remove unknown
  inferred containers without weakening checker policy.
- The cancellation timeout test previously depended on dispatch happening in
  20 ms and release happening in 30 ms. Under combined-run load it could time out
  before dispatch. It now freezes the injected preparation clock and uses an
  observed cancellation barrier; the actual asynchronous deadline remains
  20 ms, and the test verifies the outer task cannot complete before child cleanup.

### Evidence and current status

All new evidence is under `/private/tmp/nanobot-p6-completion.rk7wje`.
`direct_remaining_checks.py` snapshots all 1,892 admitted source entries with
descriptor-relative no-follow reads, excludes local Claude settings without
reading them, and records exact hashes and commands. Tests run with the existing
offline sandbox and clean environment. Source/test/checker/guard settings were
not deselected, suppressed, or relaxed.

- `remaining-final-core-r2`: **346 passed**, including the preflight/cancellation
  repairs and full Stage 5 preflight module.
- `remaining-final-adapters`: **547 passed**, with one existing Python warning
  about the fork test running in a multi-threaded process.
- `remaining-final-strict-r2`: **2,372 errors / 463 files**, down from 2,573 at
  this batch's start: **201 removed, zero new diagnostic identities**. Comparison
  normalizes snapshot paths/line movement, not rule or message contents.
- `remaining-final-lint-r2`: full Ruff **PASS**.
- `remaining-final-compile-r2`: all **1,115 Python files** compiled.
- `remaining-final-memory`: all 28 memory-recovery modules completed with
  **906 passed, 2 skipped**. Skips are not passes or native-platform qualification.
  `remaining-skip-confirmation` confirms the existing two special-permission-bit
  cases skip because this filesystem does not preserve those bits.

The three final regression groups contain **1,799 passing tests and 2 skips**;
they are distinct groups, not a sum of repeated intermediate runs.
`remaining-direct-acceptance.json` binds the final records, diagnostic delta and
current Python hashes. It records the preflight/test-only differences between
the earlier memory/adapter snapshots and the latest core snapshot explicitly.

The first combined core run (two failures) and every red regression run remain
preserved. The previously passing narrower runs are not substituted for failed
combined runs. Latest core/preflight changes were revalidated in the r2 run;
the memory and adapter snapshots retain their own exact source identities.
The whole repository's test suite has not been rerun or declared green.

Q1/P6 remain **NOT_ACCEPTED**. The residual 2,372 strict errors still include
663 private-API boundary violations, 1,033 unknown argument/variable/member
diagnostics, platform-lock optionality and other contracts. These are not all
runtime crashes, and this batch does not establish which predate the migration.
Local artifact/startup qualification and deferred CI/native coverage remain
separate from source verification. Runtime and D9/P7 remain **NOT_AUTHORIZED**.

## Subsequent contract/control-flow batch — 2026-10-06

The [attribution and repair record](p6-error-attribution.md#subsequent-contractcontrol-flow-repair--2026-10-06)
now supersedes the remaining-error count above: **2,221 / 463 files**, 151 resolved
and zero newly introduced signatures from a fresh 2,372-error baseline. **599 tests
passed**, lint and 1,117-file compilation passed, and fresh GPT-6.1 Sol review passed.
Seven preexisting standalone gateway cleanup failures were reproduced and repaired;
composition-owned production shutdown remains unchanged. The batch-only diff and
source-bound evidence are `contracts-batch.diff` and `contracts-acceptance.json` in
the existing QA directory. Full-suite completion and Q1/P6 acceptance are not claimed.

## Manifest/storyboard/export repair — 2026-10-06

Current strict backlog is **2,051 errors / 463 files**, superseding 2,221 above:
170 resolved, zero new signatures. Final-source local checks passed **261 distinct
tests** (18 policy/storyboard + 243 source/session-mode) and scoped Ruff. Manifest
and storyboard changes received fresh GPT-6.1 Sol review; export changes preserve
existing runtime checks and passed the broader source/session-mode suite.
See `p6-error-attribution.md` and QA `all-errors-export-r2`,
`all-errors-context-video-tests-final`, `all-errors-export-final-tests`, and
`all-errors-final-lint`. Full-suite/P6 completion is still not established;
runtime and D9/P7 remain NOT_AUTHORIZED.

## Canonical source contract repair — 2026-10-06

The current strict backlog is **1,967 / 463 files**, superseding 2,051: **84 resolved,
zero new diagnostics**. Canonical source/checkpoint parsing and request projections
retain runtime rejection and persistence behavior with explicit checked types.
Final local validation: **441 tests passed, no skips**, scoped Ruff and 1,117-file
compilation passed, exact final-source hash checks passed, and fresh GPT-6.1 Sol
review passed. Evidence is under QA `source-contract-final-{strict,tests,lint}`;
the detailed scope and retained failed collection invocation are recorded in
`p6-error-attribution.md`. This is not full-suite/P6 completion or runtime activation.
