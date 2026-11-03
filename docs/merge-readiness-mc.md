# Merge-readiness report: mobile remote merge corrections

Date: 2026-11-04
Branch: `fix/mobile-remote-merge-blockers`
Status: **LOCAL CANDIDATE READY FOR STAGING REVIEW — NOT MERGED, PUSHED OR DEPLOYED**

## Exact source identity

| Item | SHA / artifact |
|---|---|
| Reviewed feature handoff | `1689a4a06020c778a6e969b12bb19b27a1ac90f1` |
| Repair branch base (handoff + task docs) | `7be433232d28a88106e15a486af3ade2250347f4` |
| Fetched staging integrated | `aacca07cfadaf51fa1d83913f93ba2d797c0aa58` (`origin/staging`) |
| Staging integration merge | `221101b9a5181839802ea250dd977dd39fab0ffe` |
| Corrected source commit | `7696b85a` (final code/test commit; later commits are docs-only) |
| Candidate APK | `/tmp/opencode/mc-repair-candidate/nanobot-mobile-release.apk` |
| Candidate SHA-256 | `c84d1127c7a09432158fc7f3b449afc8ea89857216fe487c8437baf75107a383` |
| Signer | `CN=jaden`, SHA-256 `8c30d04957e9724d1668e585890913b0581676c9460789f658c4ed9f84ba0c81` |

The candidate includes Capacitor local notifications from staging (`assets/capacitor.plugins.json`); zipalign and apksigner verification pass. No APK was installed.

## Correction status

| Task | Status | Accepted evidence / remaining |
|---|---|---|
| MC01 | PASS focused | v1 version/intent/source/confirmation validation in message and owner control paths; source token carried in projection and rechecked at execution admission. Legacy local controls remain compatible. |
| MC02 | PASS focused | Submitted temporary Sandbox wins over later Full policy; channel-boundary test covers restricted/Full envelopes; discard still retires transient authority. |
| MC03 | PASS focused | Atomic busy reservation before media/scope awaits; concurrent new submissions yield one accepted turn and one `not_admitted/busy_new_request`; steering requires exact active turn and captured scope. |
| MC04 | PASS focused | Remote owner Full admission uses verified owner identity; v1 control requires bound confirmation; second device gets its own consent; UI exposes explicit device confirmation. Actual owner consent is rechecked at loop admission, and revoke clears in-memory consent. |
| MC05 | PASS focused | Request-derived Full proofs are aggregated for native stdin closure/accounting; unrelated owners unaffected; multiple handles per proof covered. Real native-process closure remains platform qualification. |
| MC06 | PASS focused | Native refresh uses `POST /remote/v1/refresh`; owner status/devices/revoke use exact owner operations with custody bearer auth and one 401 refresh retry. Instrumented owner journey added but not executed. |
| MC07 | PASS focused | Reconnect/refresh bounded to eight attempts, explicit retry notice, no auto-resubmit; foreground refresh on closed socket. Installed phone restart/expiry journey remains qualification. |
| MC08 | PASS local | Staging integrated cleanly; final full suite, strict checker, WebUI build/lint, native/shared assets, Android release and Go broker checks pass. |

## Final checks

| Check | Result |
|---|---|
| Full Python suite | **17,569 passed, 22 skipped, 6 subtests, exit 0** (`/tmp/opencode/mc-final3-full-python.log`) |
| Strict basedpyright | **0 errors / 528 files**; comparator `introduced: 0` (`/tmp/opencode/mc-repair/final6-basedpyright.json`) |
| WebUI | **1,430 passed**, build exit 0, lint exit 0 |
| Desktop parity | PASS, 127 rows / 9 categories |
| Go broker | PASS |
| Android release | BUILD SUCCESSFUL; signer/zipalign/local-notifications verified |
| Native/shared assets | rebuilt from this source |

## Remaining qualification (not code-completion claims)

- MC06 installed phone → owner status/list/revoke while gateway absent, plus the new instrumented owner route test.
- MC07 installed-app gateway restart and token-expiry recovery, including draft/scroll preservation.
- MC05 real native command stdin closure on the applicable platform.
- MC04 physical two-installation owner confirmation journey.
- M38 independent release review, signed-release sign-off, radio/Tailscale qualification.

No push, merge into staging, PR, phone installation, credential/Tailscale change or live-service mutation was performed.
