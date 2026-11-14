# A3 reconciliation run summary

**Date:** 2026-11-15 (Linux x86_64 and Android 16 x86_64 emulator)
**Coordinator and workers:** `commandcode/deepseek/deepseek-v4.1-flash`
**Worktree:** `/home/jaden/Documents/Utilities/nanobot-a3-reconciliation`
**Branch:** `codex/a3-audit-reconciliation`, 174 commits ahead of
`origin/codex/staging-audit-implementation` (`722bd8ec`).
**Candidate:** `17114c02` (code at `9c55817b`, plus evidence and documentation).
**Release state:** HOLD. Push, merge, deployment, and formal release
qualification remain with the project owner.

The [November 15 decisions](decisions.md#a3-release-decisions-for-the-four-remaining-findings)
are applied: all four remaining findings were repaired, reviewed and accepted.
The candidate above is the final integrated state. Release remains on hold until
the project owner completes push, merge and release qualification.

## Ledger

| Measure | Value |
| --- | --- |
| Original findings | 136 |
| Accepted | 136 |
| Ready for review | 0 |
| Unverified | 0 |
| Review findings | 9 (all accepted; RR1 and RR3 accepted in the third wave) |
| Evidence rows | 129 |

Recomputed from `findings`. `review_findings` is tracked separately.

## Stage status

| Stage | Status | Basis |
| --- | --- | --- |
| A3-0 | Accepted, Linux scope | Local qualification report; the final candidate passes 18,563 tests, strict 0/546, comparator clean. |
| A3-1 | Accepted, Linux scope | AC1a-AC6b met; H01-F6/F7 structural debt deferred; the read_file instance media root is accepted. |
| A3-2 | Accepted, Linux scope | AC2.1-AC2.5 met; X01-F01 gated; S02-F03 deferred; Windows/macOS unverified. |
| A3-3 | Accepted, Linux scope | AC3.1-AC3.3 pass; five named gaps recorded. |
| A3-4 | Findings complete | 14 accepted, including the applied-outcome and sidebar-identity repairs and the Git sync/shell approval repair. |
| A3-5 | Findings complete | 3 accepted; nested-launch confinement and the bypass spellings are classified and refused before launch. |
| A3-6 | Findings complete | 30 accepted, including the instance-owned OAuth credentials and the transport-attempt observer. |
| A3-7 | Findings complete | 17 accepted, including the signed-app device-key qualification on the emulator. |
| A3-8 | Findings complete | 24 accepted; durability, retention, cron, journal, approval and recovery repairs reviewed. |
| A3-9 | Findings complete | 16 accepted; the signed release APK qualified device-key recovery and revocation over real Serve TLS. |
| A3-10 | Findings complete | 8 accepted; AC10.3 still waits on a release artifact digest chosen by the project owner. |

## Repairs integrated in this run

| Commit | Change |
| --- | --- |
| `b6000cef` | RR2 rollback procedure clarifies the explicit refusal. |
| `a3e04772`, `dc02c538` | Once/task projection asks the live path policy for read, write, exec and managed-repository eligibility. |
| `dce41144` | Remote-access enable requires a JSON boolean; pairing renewal is routed and classified as a mutation. |
| `19e95d33` | `desktop_use` state-changing actions require one exact-effect approval; AR1 deny and AR2 repeat cases added. |
| `4b5c95a2` | Cron, clawhub and webapp-testing guidance matches the runtime. |

The remaining commits record reviews, evidence, and the stage decisions.

## Accepted findings by stage

- **A3-1:** H01-F4, K01-F12, H01-F1, H01-F2, H03-F3, T01-F10, K01-F11, H01-F3, K01-F10.
- **A3-2:** S01-F01, S01-F03, T01-F07, S02-F01, S02-F02, C00-F01, C00-F02, C00-F03, E02-F14, T01-F09, X01-F03.
- **A3-3:** E02-F01.
- **A3-4:** E02-F15, T01-F08, U01-F01, U01-F03, U01-F04, U01-F06, D01-F01, D01-F02, E02-F08, E02-F19, E02-F21.
- **A3-6:** K01-F01, E02-F11, E02-F13, E02-F18, H03-F2, H03-F7, E02-F23, K01-F03, K01-F04, K01-F05, K01-F07, K01-F09, T01-F01, T01-F02, T01-F03, H01-F8, H03-F6.
- **A3-7:** E02-F17, E02-F02, E02-F22, R01-F02, R01-F01, R01-F03, R01-F08, R01-F09, R01-F10, R01-F15, R01-F04, R01-F07.
- **A3-8:** E02-F06, E02-F10, M01-F05, M01-F06.
- **A3-9:** Q01-F02, R01-F06, C00-F07, Q01-F04, R01-F16, R01-F18, Q01-F06, Q01-F07, R01-F19.
- **A3-10:** A01-F02, A01-F07.

## Open findings by stage

- **A3-4** — H01-F5.
- **A3-5** — K01-F06.
- **A3-6** — E02-F16.
- **A3-9** — R01-F05.

## Final verification

| Check | Result |
| --- | --- |
| Python full suite (code `9c55817b`) | 18,563 passed, 22 skipped, 6 subtests |
| Strict BasedPyright | 0 diagnostics / 546 files |
| Go broker `-race` + C ABI | pass, `NANOBOT_CABI_SMOKE_OK version=1` |
| Android JVM suite | 32 passed |
| Signed-app device-key pass | 24/24 host and app checks |

The final full suite ran at `9c55817b` and is recorded in
`resume3/final/full-suite-final.log`; `resume3/final/strict-final.json` is the
matching 0/546 strict report.

## Remaining work and limits

- **Release hold:** push, merge, deployment and formal release qualification
  stay with the project owner. A3-10 AC10.3 needs a release artifact digest.
- **Device coverage:** the signed-app pass is emulator coverage on a Linux host.
  No physical device, hardware Keystore or process-death matrix was exercised.
- **Mobile WS frames:** `handle_ws_frame` has no caller and `build_mobile_edge`
  receives no replay cache or mutation handler, so mobile `mutation`/`logout`
  frames stay `scope_denied`. The November 15 decision leaves legacy bearer
  refresh/logout routes and headless enrollment keys outside this pass.
- **Latent:** `TokenService.refresh` omits `device_generation`
  (`auth.py:238-242`) while `validate_access` compares it (`auth.py:201`);
  unreachable because pairing issues an empty refresh token.
- **Structural debt:** H01-F6 and H01-F7 remain P2 normalization/dispatch work
  with no acceptance criterion, deferred to their owners.
- **Classifier limits:** the nested-launch classifier is a pattern list; a new
  bypass spelling, alias or env form needs a new alternative, and the H01-F5
  `git -C <dir> reset|clean|push` gap is unchanged.
- **A3-3 gaps:** no single mixed fixture, no RUNNING terminal projection,
  simulated unqualified platform, unreproducible source-reader rollback leg.

## Repository integrity

The host restarted during one documentation commit. Six loose Git objects were
left at zero bytes. The corrupt commit `cdc82806` was abandoned, its empty
objects were removed, and the same content was committed as `02254cdd`. The
corrupt reflog entries were removed. `git fsck --connectivity-only` exits 0.
No product work was lost.
