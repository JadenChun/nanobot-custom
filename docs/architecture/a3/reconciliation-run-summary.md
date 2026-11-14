# A3 reconciliation run summary

**Date:** 2026-11-14 (Linux x86_64)
**Coordinator and workers:** `commandcode/deepseek/deepseek-v4.1-flash`
**Worktree:** `/home/jaden/Documents/Utilities/nanobot-a3-reconciliation`
**Branch:** `codex/a3-audit-reconciliation`, 152 commits ahead of
`origin/codex/staging-audit-implementation` (`722bd8ec`).
**Candidate:** `d207383f` (Python code at `74abe219`, mobile at `30f8cd4c`, plus evidence and documentation).
**Release state:** HOLD. Push, merge, deployment, and formal release
qualification remain with the project owner.

## Ledger

| Measure | Value |
| --- | --- |
| Original findings | 136 |
| Accepted | 132 |
| Ready for review | 0 |
| Unverified | 4 |
| Review findings | 9 (all accepted; RR1 and RR3 accepted in the third wave) |
| Evidence rows | 118 |

Recomputed from `findings`. `review_findings` is tracked separately.

## Stage status

| Stage | Status | Basis |
| --- | --- | --- |
| A3-0 | Accepted, Linux scope | Local qualification report; 18,441 tests, strict 0/546, comparator clean. |
| A3-1 | Accepted, Linux scope | AC1a-AC6b met; H01-F6/F7 structural retained; C5 read_file residual. |
| A3-2 | Accepted, Linux scope | AC2.1-AC2.5 met; X01-F01 gated; S02-F03 deferred; Windows/macOS unverified. |
| A3-3 | Accepted, Linux scope | AC3.1-AC3.3 pass; five named gaps recorded. |
| A3-4 | Findings complete except one decision | 13 accepted; H01-F5 autoPush-exemption decision open. |
| A3-5 | Findings complete except one decision | 2 accepted; K01-F06 guidance repaired; nested-bypass policy decision open. |
| A3-6 | Findings complete except one decision | 29 accepted; E02-F16 credential-profile decision open. |
| A3-7 | Findings complete | 16 accepted, including the cleaned app UI receipt and the launch admission repair; physical device remains an environment gap. |
| A3-8 | Findings complete | 24 accepted; durability, retention, cron, journal, approval and recovery repairs reviewed. |
| A3-9 | Findings complete except R01-F05 | 15 accepted; R01-F05 partial for /mobile/v1 POST refresh/logout, remote logout, auth_key, proof-bound origin and physical device. |
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

1. **A3-7/A3-9 mobile recovery:** the cleaned app receipt exposed a launch-time
   admission failure: 6 of 10 requests are rejected as `native request was not
   admitted`, the broker reason is masked, and a read succeeds only after a
   manual refresh. A bounded native repair and a clean re-run are in progress.
2. **A3-8:** all durability findings are accepted. The remaining residuals are
   the unwired cron admin key, off-loop `pre_begin`, the process-local cache
   index, unknown-trigger resolution UI, and the unvalidated sync journal API.
3. **A3-9:** Q01-F03, R01-F17, Q01-F05, E02-F12 and H03-F5 are accepted.
   R01-F05 remains partial for launch recovery, legacy `/mobile/v1` POST
   refresh/logout, remote logout, refresh rotation replay, headless auth key and
   a physical device.
4. **A3-10:** all six dispositions accepted; AC10.1 still needs the six R18
   durability findings (now accepted) recorded in the stage ledger and the C00
   activation decision remains future work.

## Blockers

- **Product decisions:** H01-F5 autoPush subset; E02-F16 credential-profile
  policy; K01-F06 nested-bypass policy (K01-Q5/D2-D4). C00-F04/C00-F05 stay
  isolated by the migration guard; retire-or-repair is a future activation
  decision.
- **A3-9 R01-F05:** `/mobile/v1` has no composed refresh path and a rotated
  bearer grant carries `origin=""` (`auth.py:213`), so it would answer
  `403 origin_mismatch`; `/mobile/v1` POST refresh/logout and remote logout do
  not exist; `auth_key` is not wired into the app transport; no physical device
  ran. Rotation itself is qualified on `/remote/v1` (18/18).
- **Latent defect:** `TokenService.refresh` omits `device_generation`
  (`auth.py:238-242`), issuing 0 while `validate_access` compares the proof
  generation (`auth.py:201`). Unreachable today because pairing issues an empty
  refresh token (`auth.py:148-153`, `auth.py:177-181`, `store.py:421`).
- **Environment:** macOS, Windows and physical Android remain absent. The Linux
  host has the full Android toolchain and a live Tailscale Serve mapping.
- **A3-3 gaps:** no single mixed fixture, no RUNNING terminal projection,
  simulated unqualified platform, unreproducible source-reader rollback leg.

## Repository integrity

The host restarted during one documentation commit. Six loose Git objects were
left at zero bytes. The corrupt commit `cdc82806` was abandoned, its empty
objects were removed, and the same content was committed as `02254cdd`. The
corrupt reflog entries were removed. `git fsck --connectivity-only` exits 0.
No product work was lost.
