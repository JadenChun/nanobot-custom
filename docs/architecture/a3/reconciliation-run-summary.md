# A3 reconciliation run summary

**Date:** 2026-11-15 (Linux x86_64 and Android 16 x86_64 emulator)
**OpenCode run:** `commandcode/deepseek/deepseek-v4.1-flash`
**Follow-up review and qualification:** Codex
**Worktree:** `/home/jaden/Documents/Utilities/nanobot-a3-reconciliation`
**Branch:** `codex/a3-audit-reconciliation` follows
`origin/codex/staging-audit-implementation` (`722bd8ec`).
**Candidate:** `dd8a02ca`, tree `d5cbf40d9e3407a21af4d78f2f4b21af3c885918`.
**Release state:** HOLD. Local A3 qualification passed for Linux and the
Android emulator. Push, merge, and deployment remain with the project owner.

OpenCode closed the four findings from the
[November 15 decisions](decisions.md#a3-release-decisions-for-the-four-remaining-findings).
Codex reproduced and repaired a remaining OAuth isolation defect under E02-F16.

## Ledger

| Measure | Value |
| --- | --- |
| Original findings | 136 |
| Accepted | 136 |
| Ready for review | 0 |
| Unverified | 0 |
| Review findings | 9 (all accepted; RR1 and RR3 accepted in the third wave) |
| Evidence rows | 132 |

Recomputed from `findings`. `review_findings` is tracked separately.

## Stage status

| Stage | Status | Basis |
| --- | --- | --- |
| A3-0 | Accepted, Linux scope | Final source passes 18,569 tests. Strict checking reports 0/546 and the comparator reports zero new signatures. |
| A3-1 | Accepted, Linux scope | AC1a-AC6b met; H01-F6/F7 structural debt deferred; the read_file instance media root is accepted. |
| A3-2 | Accepted, Linux scope | AC2.1-AC2.5 met; X01-F01 gated; S02-F03 deferred; Windows/macOS unverified. |
| A3-3 | Accepted, Linux scope | AC3.1-AC3.3 pass; five named gaps recorded. |
| A3-4 | Accepted, Linux scope | 14 findings accepted. Review receipts cover approval, admission, Stop and applied-result readback. |
| A3-5 | Accepted, Linux scope | 3 findings accepted. Receipts cover native subagent ownership, history and control. The external backend stays gated. |
| A3-6 | Accepted, Linux scope | 30 findings accepted, including the corrected SDK and gateway OAuth ownership. |
| A3-7 | Accepted, emulator scope | 16 findings accepted. The final signed app paired with a device key, recovered after expiry and restart, and returned to pairing after revocation. |
| A3-8 | Accepted, Linux scope | 24 findings accepted. Receipts cover durability, cleanup, crash recovery and asset paging. |
| A3-9 | Accepted, Linux and emulator scope | 16 findings accepted. The release wheel passed installed startup and rollback. The final APK passed device-key pairing, recovery, and revocation over Serve TLS. |
| A3-10 | Accepted, Linux and emulator scope | 8 findings accepted. The stopped-copy rollback preserved post-cutover history and the final test device's revocation. |

## Repairs integrated in this run

| Commit | Change |
| --- | --- |
| `b6000cef` | RR2 rollback procedure clarifies the explicit refusal. |
| `a3e04772`, `dc02c538` | Once/task projection asks the live path policy for read, write, exec and managed-repository eligibility. |
| `dce41144` | Remote-access enable requires a JSON boolean; pairing renewal is routed and classified as a mutation. |
| `19e95d33` | `desktop_use` state-changing actions require one exact-effect approval; AR1 deny and AR2 repeat cases added. |
| `4b5c95a2` | Cron, clawhub and webapp-testing guidance matches the runtime. |
| `9aa150e5`, `487a99d8`, `9c55817b` | Shell approval and nested-agent launch admission. |
| `22caed46`, `dd8a02ca` | Provider OAuth storage, requests and settings use the configured instance. |

## Final verification

| Check | Result |
| --- | --- |
| Python full suite at `dd8a02ca` | 18,569 passed, 22 skipped, 6 subtests |
| Strict BasedPyright | 0 diagnostics / 546 files |
| WebUI activity and inspector | 68 passed |
| TUI | 241 passed and TypeScript check passed |
| Android host checks | Go race suite, C ABI, and 32 JVM tests passed |
| Installed Linux wheel | Version, package contents, dependencies, startup and isolated rollback passed |
| Stopped-copy rollback | Session and cron state rehearsal passed |
| Copied device revocation | Both installed wheel versions read the final test device's revocation and proof without changing the state digest. |
| Final signed APK | Android 16 x86_64 device-key pairing, expiry recovery after a cold restart, and owner revocation passed over Serve TLS. The app remained at pairing after restart. |
| Final gateway over Serve TLS | 24/24 host device-key checks passed |

The `resume4/final/` receipts contain the suite logs, installed checks,
and artifact identities. The wheel contains 546 matching Python files and 590
WebUI files. The final APK shares 534 files byte for byte with that bundle.

| Artifact | SHA-256 |
| --- | --- |
| Linux wheel | `435736784453bf865fb2f14e4d22fbe39a21bee3cb54a6823df65d436444a461` |
| Source archive | `d74fefb49b8cafa0f610bc09fbf55ca11c5c184c226c7165f8a359a66ce37500` |
| Android APK | `07d7e3e1103443d3bc35ffc8249b53a72cfd71bf5a36f21ac06eeeba45a492ff` |

The [final Android readback](implementation-handoff.md#final-android-emulator-readback)
identifies the app and gateway receipts for this candidate.

## Release handoff and limits

- Local qualification covers Linux and the Android emulator. The remote audit
  branch still has the older candidate. Publication and rollout need a release
  decision and the hosted checks for the published revision.
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
