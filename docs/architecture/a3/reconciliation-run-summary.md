# A3 reconciliation run summary

**Date:** 2026-11-14 (Linux x86_64)
**Coordinator and workers:** `commandcode/deepseek/deepseek-v4.1-flash`
**Worktree:** `/home/jaden/Documents/Utilities/nanobot-a3-reconciliation`
**Branch:** `codex/a3-audit-reconciliation`, 152 commits ahead of
`origin/codex/staging-audit-implementation` (`722bd8ec`).
**Candidate:** `d207383f` (Python code at `74abe219`, mobile at `30f8cd4c`, plus evidence and documentation).
**Release state:** HOLD. Push, merge, deployment, and formal release
qualification remain with the project owner.

The [November 15 decisions](decisions.md#a3-release-decisions-for-the-four-remaining-findings)
and [OpenCode continuation](opencode-handoff.md#continue-from-the-reconciliation-branch)
define the next pass. The candidate above identifies the November 14 run.

## Ledger

| Measure | Value |
| --- | --- |
| Original findings | 136 |
| Accepted | 132 |
| Ready for review | 0 |
| Unverified | 4 |
| Review findings | 9 (all accepted; RR1 and RR3 accepted in the third wave) |
| Evidence rows | 119 |

Recomputed from `findings`. `review_findings` is tracked separately.

## Stage status

| Stage | Status | Basis |
| --- | --- | --- |
| A3-0 | Accepted, Linux scope | Local qualification report; the final candidate passes 18,506 tests, strict 0/546, comparator clean. |
| A3-1 | Accepted, Linux scope | AC1a-AC6b met; H01-F6/F7 structural debt deferred; the read_file instance media root is accepted. |
| A3-2 | Accepted, Linux scope | AC2.1-AC2.5 met; X01-F01 gated; S02-F03 deferred; Windows/macOS unverified. |
| A3-3 | Accepted, Linux scope | AC3.1-AC3.3 pass; five named gaps recorded. |
| A3-4 | H01-F5 unverified | 13 accepted. Remove the shell approval exemption for autoPush. |
| A3-5 | K01-F06 unverified | 2 accepted. The guidance repair is in place. Verify admission under the selected launch policy. |
| A3-6 | E02-F16 unverified | 29 accepted. Bind provider OAuth state to the gateway instance. |
| A3-7 | Findings complete | 16 accepted, including the app UI receipt and launch admission repair. |
| A3-8 | Findings complete | 24 accepted; durability, retention, cron, journal, approval and recovery repairs reviewed. |
| A3-9 | R01-F05 unverified | 15 accepted. Qualify current device-key recovery and revocation through the installed emulator app and production gateway. |
| A3-10 | Dispositions complete | 8 accepted; AC10.3 waits on A3-9 and an artifact digest. |

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

## Next ready packages

1. Give one worker shell approval work for H01-F5 and K01-F06.
2. Give one worker provider credential work for E02-F16.
3. Prepare the signed emulator qualification for R01-F05 in parallel.

## Blockers

- Release acceptance needs matching evidence for the four unverified findings.
- AC10.3 needs A3-9 acceptance and the exact release artifact digest.
- Qualify Linux and the Android emulator in this pass. The existing
  `/remote/v1` bearer rotation receipt applies to the historical path.

## Repository integrity

The host restarted during one documentation commit. Six loose Git objects were
left at zero bytes. The corrupt commit `cdc82806` was abandoned, its empty
objects were removed, and the same content was committed as `02254cdd`. The
corrupt reflog entries were removed. `git fsck --connectivity-only` exits 0.
No product work was lost.
