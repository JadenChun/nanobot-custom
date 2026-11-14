# A3-0 local qualification report

**Date:** 2026-11-14
**Host:** Linux x86_64, Python 3.14.10, bubblewrap 0.12.0, 12 CPUs.
**Candidate:** commit `8355e614` code, unchanged through `b176af58` except
documentation and evidence. Tree at `8355e614`: recorded in `evidence.json`.
**Baseline comparison:** `6e2c7761` for the frozen strict report and its
pre-repair full suite.

## Source and environment

- The worktree is `/home/jaden/Documents/Utilities/nanobot-a3-reconciliation`
  on branch `codex/a3-audit-reconciliation`.
- The tracked remote audit branch is
  `origin/codex/staging-audit-implementation` at `722bd8ec`.
- The Python environment is the shared frozen environment at
  `/home/jaden/Documents/Utilities/nanobot-custom-worktrees/staging-audit-implementation/.venv`.
  It has 211 installed pins. The dependency list did not change across the
  paired checks; before and after lists differ only in ordering.
- WebUI uses Bun 1.4.2. Node is 24.9.0.
- No Android SDK, adb, emulator, Go toolchain, JDK 21, macOS host, or Windows
  host is available. Physical device and native broker qualification stay
  outside this run.

## Checks

| Check | Result | Receipt |
| --- | --- | --- |
| Full Linux suite at the candidate | 18,357 passed, 22 skipped, 6 subtests passed | `final/full-suite.log` |
| Strict BasedPyright at the candidate | 546 files, 0 errors, 0 warnings, 0 information | `/tmp/a3-final-strict.json` copied to `final/strict.json` |
| Frozen comparator, `6e2c7761` before vs candidate after | `PASS_NO_NEW_DIAGNOSTICS`, 0 introduced, 0 resolved, strict_clean true | `final/strict-comparison.json` |
| Ruff | All checks passed | existing A3-2 receipt |
| WebUI suite at current source | 101 files, 1,550 tests passed | `w4-client/vitest.log` |
| WebUI lint | No diagnostics or warnings | `w4-client/lint.log` |
| WebUI bundle identity | Fresh rebuild is byte-identical to the shipped bundle; manifest differs only by a node_modules prefix | `w4-client/vite-build-verification.log` |
| Desktop and narrow UI | Desktop PASS; narrow 390x760 PASS including scroll 1790 to 1790 and focus to the originating row | `w4-client/CLIENT-REVIEW.md`, `w13-narrow/PLAYWRIGHT-390x760.md` |
| TUI | 238 tests and type check passed earlier; TUI sources are unchanged since `f40c229c` | `implementation-handoff.md` record |
| Android, macOS, Windows, native Go | Not run. No host toolchain or device. | named gap |

The 22 skipped Python cases are platform or optional-channel cases. The run
contains no failure.

## Result

The Linux source and client candidate is qualified for the repository checks
and the real client paths listed above. The strict checker is clean, not only
unchanged. A3-0 is accepted for the Linux scope. The absent platforms and the
native mobile toolchain remain named release gaps.

## Limits

- No historical migration, two-APK upgrade, or released-version reader is
  required by the selected first release.
- Linux evidence does not qualify macOS, Windows, physical Android, or an
  installed release artifact.
- The full suite includes the repository's optional-channel skips. It is not a
  claim that every optional integration was exercised.
