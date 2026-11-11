# Feature repair execution — batches 1–3

User-authorized scope: repair media correctness, skills/clipboard behavior, and
the runtime integration described in the feature audit. macOS peer credentials,
native sandbox support, and broad structural cleanup are separate work.

Implementation and integrated self-review use the direct development loop.
No independent reviewer or delegated review was used. Review covers request
concurrency, runtime/public contracts and final test failures. Production
security/admission fences and existing durable state owners remain authoritative.

Status: all three bounded packages are implemented and locally validated. See
[repair-results.md](./repair-results.md) for evidence and remaining qualification
limits; this is not whole-project acceptance.

## Frozen baseline

- Branch: `codex/feature-audit-repairs`, from
  `364259daed2c11700d4064e7ca58743fd037b0b0`.
- Existing audit documents are retained.
- QA root: `/tmp/nanobot-feature-repairs-20261105.3kdy8inl`.
- Source copies, hashes, dependency versions, checker invocation and raw reports
  are retained there. Telegram test dependencies were installed before freezing
  the environment; final package versions and checker policy match that baseline.
- Full-source baseline: BasedPyright 1.40.1, 537 files, **1,001 errors**, no
  warnings. This is existing debt, not a passing strict baseline.
- Each backend package must compare full-source diagnostics with the immutable
  baseline using `scripts/check_type_regressions.py`, with zero introduced
  signatures across all files. Lower counts alone do not pass.

## Package A — media correctness

- Invalidate obsolete initial/page requests on chat, token, active state, and
  refresh changes. Protect data, cursor, totals and loading state; prevent
  duplicate page requests before React renders.
- Preserve a bound on session transcript/index reads per library request.
  Continue beyond 200 sessions through an explicit cursor and truthful
  completeness/count metadata. Existing single-session asset behavior remains.
- Acceptance: delayed-response permutations, retries, rapid pagination,
  close/reopen/token changes, and a real 201+ session endpoint traversal.

## Package B — skills and clipboard polish

- Distinguish catalog loading, transport failure, empty success, and restricted
  source completeness. Retain successful data only for the same authenticated
  scope and provide retry in the installed-skills UI.
- Preserve plain text exactly once when pasted with supported attachments,
  including textarea selection/cursor behavior; retain the attachment whitelist.
- Acceptance: initial/refresh failures and retry; restricted catalog behavior;
  text-only, file-only, mixed, unsupported-file, and selection-replacement paste.

## Package C — normal runtime integration

- Route supported manual automation runs through the installed occurrence and
  execution owner; project actual runtime/delivery status. Expose unsupported
  modes/transports truthfully rather than bypassing their admission fences.
- Install local-trigger delivery through an explicitly enabled, owned gateway
  consumer, including busy-session behavior, restart recovery, and shutdown.
  Preserve the default inactive-mode contract and retained queue entries.
- Wire `my` through its existing config/read-only gate, and sustained-goal tools
  through explicit user-goal authorization and owned session/run behavior.
- Distinguish configured Heartbeat/Dream intent from effective runtime support;
  align their settings and guides with installed behavior.
- Acceptance: use normal composition, controlled local providers/transports,
  explicit enabled/disabled cases, actual due/manual/trigger turns, and
  cancellation/restart tests. No live external delivery is needed or authorized.

## Validation and completion

Run affected tests after each package, the full frontend suite/build at final
integration, relevant gateway/cron/command/runner/SDK suites, full-source type
comparison, lint and diff checks. Preserve and explain existing platform test
failures. Update the findings report with actual repair/qualification status.
Do not claim full project/P6 acceptance from bounded repair results.

Completed: full frontend suite/build/lint, final affected backend selection,
baseline comparison of remaining failures, full-source type comparison, Python
lint and diff checks. The later [macOS follow-up](./macos-repair-results.md)
addresses the retained failures and records full backend acceptance. Existing
strict-type debt remains open.
