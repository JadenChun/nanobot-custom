# P6 runtime-test checkpoint — 2026-10-13

**P6 remains NOT_ACCEPTED. Runtime/D9/P7 remain unauthorized.**

## Corrected behavior

Ordinary legacy foreground turns now replay a validated persisted summary even
when the legacy `AutoCompact` object is absent. Previously, disabling that
background component also accidentally disabled summary replay: the summary
remained on disk but was missing from the next model request. The fix reuses the
existing metadata validator; it does not enable timers or archival. Source-turn,
fresh-history, ephemeral and Dream exclusions remain intact.

The original real idle-compaction test and a new ordinary-summary case both fail
before the fix and pass afterward. Six added cases cover repeated turns,
valid/malformed metadata, ephemeral requests and Dream isolation without archive
calls. All original assertions in the two affected compaction modules remain.
Timezone and compaction setup now uses acquired graphs. Retired memory scope is
tested for refusal, and signature replay supplies a complete tool exchange while
retaining a trailing-assistant rejection case.

## Verified evidence

- Full unchanged strict: **0 errors, 0 warnings, 464 files**; global diagnostic
  comparison introduces zero errors. Full lint passes.
- Affected five complete modules: **78 passed**.
- Broader six complete modules: **204 passed**, including legacy AutoCompact,
  save-turn, gateway dormancy, composition and startup guards. Counts overlap.
- Earlier three-module CLI checkpoint: **202 passed** under a separately
  qualified local PTY/Unix-socket profile. TCP, unlisted Unix sockets and unlisted
  executable probes remain denied. That run predates this summary-replay fix.

Raw source/environment/result hashes and assertion/scope verification are in
`p6-summary-replay-r1-progress.json` under the private QA root
`/private/tmp/nanobot-p6-completion.rk7wje`. The production change is one parser
import and one metadata-replay branch; removing those reconstructs the previous
loop bytes. Shared fixtures, production guards, dependencies and the original
runner/profile are unchanged.

## Full-suite result and remaining work

The fresh whole default collection selected **15,945 tests**. It continued making
progress but hit the 900-second runner deadline at **71%**. Before termination it
recorded **202 failed/error outcomes across 35 modules**. These are observed
outcomes, not a complete failure total. The runner killed the process group at
deadline, verified it absent, and verified unchanged candidate/snapshot source.
This is incomplete evidence, never a pass; it predates the repair above.

The exact observed backlog is retained in
`p6-local-full-r1-failure-inventory.json`. Continue repairing it against accepted
contracts, then run a separately documented longer whole-suite qualification
with immediate failure-report capture. Do not alter the frozen original runner
or infer success for the unexecuted remainder. No full-suite, whole-source review,
native/release qualification or applicable Q2–Q5 gate is waived by this package.
