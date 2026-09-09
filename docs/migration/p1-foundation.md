# P1 upstream integration foundation

Date: 2026-09-09. This is an intermediate source checkpoint, not the completed migration or a runnable release. Work is isolated in `/Users/jaden/Documents/github/nanobot-migration-integration`; the original staging-based checkout and its untracked sessions are untouched.

## History and scope

The accepted P0 source/documentation checkpoint is `b98d731a`. The normal merge imports pinned upstream `f47b8f23155ddec23cfb80ec001b544d495df89c` and retains staging `4aa562d92828189b3fcefad079773efbab1d715a` ancestry. The [resolution ledger](p1-resolution-ledger.tsv) records all 70 conflict paths, retained contracts, accountable packages and validation specifications. Foundation-only selection does not close downstream behavior findings.

Upstream module, provider, runner and channel-package structure is the foundation. All 57 fork-changed test modules are preserved byte-for-byte relative to the accepted P0 checkpoint, including the P0 logger fixture; [test map](p1-fork-test-map.tsv) records their candidate locations and hashes. Colliding upstream cases remain alongside renamed fork modules. The historical P0 evidence remains unchanged.

## Implemented compatibility

- Legacy token objects normalize to upstream output-token and total-context fields; staging defaults and missing-only legacy capacity precedence survive. Context paths, repository settings, crawler/browser/device/desktop/image config, provider key pools, pacing/timeouts and heartbeat retention remain declared data.
- BOM reads, explicit save/reload and environment overrides pass synthetic checks. Channels retain extra plugin settings; missing/partial WebSocket config stays disabled, Dream remains disabled, idle TTL remains zero, and absent timezone remains UTC.
- Bare CLI displays help; interactive agent defaults classic. Gateway retains `--log-file` and the 20 MB / seven-day / gzip logging policy. HTTP streaming requests still return 400.
- Public CLI, SDK factory, GUI/desktop and gateway-management activation is blocked before setup, state conversion, service control or provider execution while P2-P5 are unfinished. The guard has no configuration/environment bypass. ToolLoader also stops before plugin discovery. Duplicate tool names fail explicitly.
- Crawler/desktop extras, crawler command, Python skill assets and `.tmp/` exclusion are retained alongside upstream build machinery. CI explicitly targets the migration/staging branches. Wheel/frontend/platform qualification is still P6 work.

The guard is a public activation boundary, not a sandbox for arbitrary Python callers. `create_app` remains composable with caller-supplied test doubles. Low-level modules are not certified for runtime use. Foreground delegation still needs its P2 import/factory bridges, and canonical turn wiring remains P5.

## Validation

The clean disposable HOME/XDG environment, isolated Python 3.11 venv and macOS network-denied sandbox are recorded in the [manifest](p1-validation-manifest.json). All **32 foundation tests pass**. A combined preservation sample has **49 passed / 5 failed / no skips**; [per-case outcomes](p1-validation-cases.tsv) preserve every result. Compilation and scoped Ruff pass. The staged merge whitespace check reports inherited whitespace in 8 exact upstream/fork files; each was verified byte-identical to its source rather than changing preserved evidence. Adaptation-only whitespace checks pass. [Dependency freeze](p1-validation-requirements.txt) records this test environment; it is not a reproducible production lock.

The five failures are concrete downstream integration gates, not accepted regressions:

| Failure | Owner |
| --- | --- |
| ToolRegistry lacks fork contextual execution/result compatibility | P2/P5, CORE-01–03 / EDGE-01 |
| Session history retains internal metadata | P4, CORE-04 |
| Session lacks run-message selection | P3/P4, CORE-07 |
| Two managed-write fixtures reject resource_policy constructor argument | P2, CORE-08 / EDGE-03–05 |

No full fork/upstream suite, real Windows launch, live provider/channel, state conversion, wheel build, deployment or production parity acceptance is claimed.

## Review and continuation

A read-only High analyst checked the consequential activation/config/test-preservation boundaries and rechecked the reported corrections. The final 32-test clean run supplies SDK/desktop verification; the analyst did not run tests. Worker capacity remained exhausted, so the user's retry after the proposed direct-work fallback was treated as authorization for direct implementation; there was no implementation worker or fresh Medium review.

Continue with P2 policy/tool compatibility and P3 provider/channel adapters, then P4 durable context and P5 canonical wiring. Remove the guard only after their behavioral gates pass. P6 copied-state rehearsal/build and P7 production cutover remain separate. P0 acceptance is unchanged; the overall migration remains in progress.
