# A3 reconciliation run summary

**Date:** 2026-11-14 (Linux x86_64)
**Coordinator and workers:** `commandcode/deepseek/deepseek-v4.1-flash`
**Worktree:** `/home/jaden/Documents/Utilities/nanobot-a3-reconciliation`
**Branch:** `codex/a3-audit-reconciliation`, 42 commits ahead of
`origin/codex/staging-audit-implementation` (`722bd8ec`).
**Candidate:** `3e367b95` (Python and WebUI code at `7b517880`, parity inventory at `c99e8d80`, plus evidence and documentation).
**Release state:** HOLD. Push, merge, deployment, and formal release
qualification remain with the project owner.

## Ledger

| Measure | Value |
| --- | --- |
| Original findings | 136 |
| Accepted | 69 |
| Ready for review | 3 |
| Unverified | 64 |
| Review findings | 9 (AR1-AR6 and RR2 accepted; RR1 and RR3 open) |
| Evidence rows | 70 |

Recomputed from `findings`. `review_findings` is tracked separately.

## Stage status

| Stage | Status | Basis |
| --- | --- | --- |
| A3-0 | Accepted, Linux scope | Local qualification report; 18,367 tests, strict 0/546, comparator clean. |
| A3-1 | Accepted, Linux scope | AC1a-AC6b met; H01-F6/F7 structural retained; C5 read_file residual. |
| A3-2 | Accepted, Linux scope | AC2.1-AC2.5 met; X01-F01 gated; S02-F03 deferred; Windows/macOS unverified. |
| A3-3 | Accepted, Linux scope | AC3.1-AC3.3 pass; five named gaps recorded. |
| A3-4 | In progress | 8 accepted; 3 ready; 3 open. |
| A3-5 | Not started | 3 open. |
| A3-6 | In progress | 13 accepted; 17 open. |
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
- **A3-4:** E02-F15, T01-F08, U01-F03, U01-F04, U01-F06, E02-F08, E02-F19, E02-F21.
- **A3-6:** H03-F2, H03-F7, E02-F23, K01-F03, K01-F04, K01-F05, K01-F07, K01-F09, T01-F01, T01-F02, T01-F03, H01-F8, H03-F6.
- **A3-7:** E02-F17, E02-F02, E02-F22, R01-F02, R01-F01, R01-F03, R01-F08, R01-F09, R01-F10, R01-F15, R01-F04, R01-F07.
- **A3-8:** E02-F06, E02-F10, M01-F05, M01-F06.
- **A3-9:** Q01-F02, R01-F06, C00-F07, Q01-F04, R01-F16, R01-F18, Q01-F06, Q01-F07, R01-F19.
- **A3-10:** A01-F02, A01-F07.

## Open findings by stage

- **A3-1:** H01-F6, H01-F7.
- **A3-2:** X01-F01, S02-F03.
- **A3-4:** H01-F5, E02-F20, U01-F05.
- **A3-5:** T01-F04, K01-F06, X01-F02.
- **A3-6:** K01-F01, K01-F02, C00-F06, E02-F11, E02-F13, E02-F16, E02-F18, H03-F4, K01-F08, X01-F04, H03-F1, H03-F5, T01-F12, T01-F11, U01-F02, T01-F05, T01-F06.
- **A3-7:** R01-F11, R01-F12, R01-F13, R01-F14.
- **A3-8:** A01-F09, K01-F13, H02-CAP-SOURCE64, H02-CAP-JOURNAL4096, H02-IO, A01-F04, E02-F07, A01-F01, A01-F03, A01-F05, A01-F08, M01-F01, M01-F02, M01-F04, E02-F03, E02-F05, E02-F09, M01-F03, E02-F04, A01-F06.
- **A3-9:** Q01-F01, R01-F05, Q01-F03, E02-F12, R01-F17, Q01-F05, R01-F20.
- **A3-10:** C00-F04, C00-F05, S01-F02, H02-ART-SOURCE, H02-LEGACY-ISOLATE, H02-IDLE-DORMANT.

## Next ready packages

1. **A3-4 P3/P4:** admission, Stop and outcome/projection. Blocked before this
   run; A3-2 and A3-3 are now accepted. P3 needs one backend writer and one TUI
   writer. P4 needs the backend replay owner and the settings/projection client
   owner.
2. **A3-4 P6/P7:** instance-root folder grants and the search/read contract.
   A3-2 and A3-3 are accepted, so both are ready.
3. **A3-4 P8:** autoPush exemption. Needs the K01/D4 product decision.
4. **A3-6a behavioral:** K01-F03 metadata shadowing, K01-F07 alternative
   binaries, E02-F23 no-query projection ceiling. Requirements-only cases are
   in `req-a36a/CASES.md`.
5. **A3-6b/c:** T01-F02 plugin disposer, T01-F03 waiter cancellation, H03-F2
   Codex cancel/expiry, H03-F7 xAI ordering, H03-F6 route capabilities,
   T01-F01 plugin schemas, H03-F5 attempt totals; CLI-apps/MCP/usage instance
   roots, credential profiles (decision needed), repo entrypoint confinement.
6. **A3-7:** listener auth, owner instance scope and authenticated notification
   are accepted. Native media logging and broker bounds need Go/JVM toolchains.
7. **A3-8:** image reference budget, PDF pre-decode bound, session-delete
   completeness, Temporary Chat tombstone retirement. Requirements-only cases
   are in `req-a38/CASES.md`. RR1 and RR3 remain open.
8. **A3-9:** unify the health parser (C00-F07), truthful systemd outcomes
   (R01-F06), installed pipx executable (Q01-F03), toolchain identity
   (R01-F17), staged update (Q01-F05), release switch (E02-F12), and real
   chat-path qualification (R01-F05). The Windows crawler stays out of scope.
9. **A3-10:** two onboarding decisions (C00-F04, C00-F05); the other six are
   retained or isolated by disposition.
10. **A3-5:** all three findings await a packet pass.

## Blockers

- **Product decisions:** H01-F5 autoPush subset; E02-F16 credential-profile
  policy; E02-F13 MCP credential migration.
- **Environment:** no Android SDK, adb, emulator, Go toolchain, JDK 21, macOS,
  or Windows. Physical Android, native broker, and cross-platform confinement
  stay unqualified.
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
