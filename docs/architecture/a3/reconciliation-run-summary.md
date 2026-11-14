# A3 reconciliation run summary

**Date:** 2026-11-14 (Linux x86_64)
**Coordinator and workers:** `commandcode/deepseek/deepseek-v4.1-flash`
**Worktree:** `/home/jaden/Documents/Utilities/nanobot-a3-reconciliation`
**Branch:** `codex/a3-audit-reconciliation`, 83 commits ahead of
`origin/codex/staging-audit-implementation` (`722bd8ec`).
**Candidate:** `7c1aa1a8` (code at `1474d136`, plus evidence and documentation).
**Release state:** HOLD. Push, merge, deployment, and formal release
qualification remain with the project owner.

## Ledger

| Measure | Value |
| --- | --- |
| Original findings | 136 |
| Accepted | 76 |
| Ready for review | 0 |
| Unverified | 60 |
| Review findings | 9 (AR1-AR6 and RR2 accepted; RR1 and RR3 open) |
| Evidence rows | 73 |

Recomputed from `findings`. `review_findings` is tracked separately.

## Stage status

| Stage | Status | Basis |
| --- | --- | --- |
| A3-0 | Accepted, Linux scope | Local qualification report; 18,441 tests, strict 0/546, comparator clean. |
| A3-1 | Accepted, Linux scope | AC1a-AC6b met; H01-F6/F7 structural retained; C5 read_file residual. |
| A3-2 | Accepted, Linux scope | AC2.1-AC2.5 met; X01-F01 gated; S02-F03 deferred; Windows/macOS unverified. |
| A3-3 | Accepted, Linux scope | AC3.1-AC3.3 pass; five named gaps recorded. |
| A3-4 | In progress | 11 accepted; 3 open. |
| A3-5 | Not started | 3 open. |
| A3-6 | In progress | 17 accepted; 13 open. |
| A3-7 | In progress | 12 accepted; 4 open (native only). |
| A3-8 | In progress | 4 accepted; 20 open; durability repairs packaged. |
| A3-9 | In progress | 9 accepted; 7 open. |
| A3-10 | Inventory complete | 2 accepted; 6 dispositioned (2 isolate decisions, 3 retain, 1 retain gate). |

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

- **A3-1:** H01-F6, H01-F7.
- **A3-2:** X01-F01, S02-F03.
- **A3-4:** H01-F5, E02-F20, U01-F05.
- **A3-5:** T01-F04, K01-F06, X01-F02.
- **A3-6:** K01-F02, C00-F06, E02-F16, H03-F4, K01-F08, X01-F04, H03-F1, H03-F5, T01-F12, T01-F11, U01-F02, T01-F05, T01-F06.
- **A3-7:** R01-F11, R01-F12, R01-F13, R01-F14.
- **A3-8:** A01-F09, K01-F13, H02-CAP-SOURCE64, H02-CAP-JOURNAL4096, H02-IO, A01-F04, E02-F07, A01-F01, A01-F03, A01-F05, A01-F08, M01-F01, M01-F02, M01-F04, E02-F03, E02-F05, E02-F09, M01-F03, E02-F04, A01-F06.
- **A3-9:** Q01-F01, R01-F05, Q01-F03, E02-F12, R01-F17, Q01-F05, R01-F20.
- **A3-10:** C00-F04, C00-F05, S01-F02, H02-ART-SOURCE, H02-LEGACY-ISOLATE, H02-IDLE-DORMANT.

## Next ready packages

1. **A3-4:** verify E02-F20 (mutation outcome) and U01-F05 (sidebar persistence)
   against the now-integrated outcome and client paths; H01-F5 needs the K01/D4
   autoPush decision.
2. **A3-5:** verify T01-F04 (remote MCP read-only hint), K01-F06 (tmux `--yolo`
   guidance and nested-agent policy) and X01-F02 (crawler expand). T01-F04 and
   X01-F02 appear satisfied in current source; K01-F06 needs a text/policy fix.
3. **A3-6:** accept the verified-but-unreviewed items (T01-F11, U01-F02,
   K01-F02, K01-F08, X01-F04, H03-F4, T01-F05, T01-F06, H03-F1). E02-F16 needs
   a credential-profile decision. C00-F06 needs the instance media root. H03-F5
   needs attempt-total evidence.
4. **A3-7:** native R01-F11 through R01-F14 need Go/JVM or Android toolchains;
   source-only evidence exists.
5. **A3-8:** the durability group remains: A01-F01/F03/F04/F05/F06/F08/F09,
   H02-CAP-SOURCE64/JOURNAL4096, H02-IO, E02-F07, K01-F13, M01-F01/F02/F04, and
   E02-F03/F04/F05/F09. Slice order is in `w8-a38a/REPAIRS.md`.
6. **A3-9:** installer (`Q01-F03`), toolchain identity (`R01-F17`), staged
   update (`Q01-F05`), release switch (`E02-F12`), and real chat-path
   qualification (`R01-F05`). Q01-F01 and R01-F20 are dispositions.
7. **A3-10:** decide C00-F04/C00-F05; record the four retained dispositions.
8. **RR1 and RR3:** open review findings; RR1 needs complete-reference proof and
   RR3 needs the retired-migration whole-tree proof.

## Blockers

- **Product decisions:** H01-F5 autoPush subset; E02-F16 credential-profile
  policy; C00-F04/C00-F05 onboarding retire-or-repair.
- **Environment:** macOS, Windows, physical Android, and Tailscale qualification
  remain absent. The Linux host now has the Android SDK, adb, an API 36 x86_64
  emulator image, JDK 21, and Go 1.27.3 under the ignored mobile/toolchain.
- **Open review findings:** RR1 reference completeness and RR3 retired migration
  proof.
- **A3-1 residual:** C5 built-in `read_file` does not yet resolve the instance
  media root.
- **A3-3 gaps:** no single mixed fixture, no RUNNING terminal projection,
  simulated unqualified platform, unreproducible source-reader rollback leg.

## Repository integrity

The host restarted during one documentation commit. Six loose Git objects were
left at zero bytes. The corrupt commit `cdc82806` was abandoned, its empty
objects were removed, and the same content was committed as `02254cdd`. The
corrupt reflog entries were removed. `git fsck --connectivity-only` exits 0.
No product work was lost.
