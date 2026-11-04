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
| Corrected source commit | `9588d899` (final code/test commit; later commits are docs-only) |
| Candidate APK | `/tmp/opencode/mc-repair-candidate/nanobot-mobile-release.apk` |
| Candidate SHA-256 | `8e72b5f817d888f85211970cebfdd92b1a3f250bd0d6585908770c951b7b5064` |
| Signer | `CN=jaden`, SHA-256 `8c30d04957e9724d1668e585890913b0581676c9460789f658c4ed9f84ba0c81` |

The candidate includes Capacitor local notifications from staging (`assets/capacitor.plugins.json`); zipalign and apksigner verification pass. This APK supersedes the pre-review `4d62c84b…` candidate and was not installed because the phone was not attached at rebuild time.

## Correction status

| Task | Status | Accepted evidence / remaining |
|---|---|---|
| MC01 | PASS focused | v1 version/intent/source/confirmation validation in message and owner control paths; source token carried in projection and rechecked at execution admission. Post-review: source rechecked after hydration and immediately before transcript/dispatch effects, closing the reset-during-hydration race. Legacy local controls remain compatible. |
| MC02 | PASS focused | Submitted temporary Sandbox wins over later Full policy; channel-boundary test covers restricted/Full envelopes; discard still retires transient authority. |
| MC03 | PASS focused | Atomic busy reservation before media/scope awaits; concurrent new submissions yield one accepted turn and one `not_admitted/busy_new_request`; steering requires exact active turn and captured scope. Post-review: target/scope rechecked after hydration and before effects; loop consumes steering metadata at bus and pending-injection admission; rejected queued prompts stay in the composer queue. |
| MC04 | PASS focused | Remote owner Full admission uses verified owner identity; v1 control requires bound confirmation; second device gets its own consent; UI exposes explicit device confirmation. Actual owner consent is rechecked at loop admission, and revoke clears in-memory consent. |
| MC05 | PASS focused | Request-derived Full proofs are aggregated for native stdin closure/accounting; unrelated owners unaffected; multiple handles per proof covered. Real native-process closure remains platform qualification. |
| MC06 | PASS focused | Native refresh uses `POST /remote/v1/refresh`; owner status/devices/revoke use exact owner operations with custody bearer auth and one 401 refresh retry. Instrumented owner journey added but not executed. |
| MC07 | PASS focused | Reconnect/refresh bounded to eight attempts, explicit retry notice, no auto-resubmit; foreground refresh on closed socket. Installed phone restart/expiry journey remains qualification. |
| MC08 | PASS local | Staging integrated cleanly; final full suite, strict checker, WebUI build/lint, native/shared assets, Android release and Go broker checks pass. |

## Final checks

| Check | Result |
|---|---|
| Full Python suite | **17,575 passed, 22 skipped, 6 subtests, exit 0** (`/tmp/opencode/mc-full-pytest-final.log`) |
| Strict basedpyright | **0 errors / 528 files**; comparator `PASS_NO_NEW_DIAGNOSTICS`, `introduced: 0` (`/tmp/opencode/mc-type/{before,after-final,comparator-final}.json`) |
| WebUI | **1,432 passed**, build exit 0, lint exit 0 (`/tmp/opencode/mc-webui-tests.log`, `/tmp/opencode/mc-webui-lint.log`) |
| Desktop parity | PASS, 127 rows / 9 categories |
| Go broker | PASS |
| Android release | BUILD SUCCESSFUL; signer/zipalign/local-notifications verified |
| Native/shared assets | rebuilt from this source |

## Remaining qualification (not code-completion claims)

- M38 independent release review and signed-release sign-off.
- Applicable radio/Tailscale qualification for the target deployment.
- The literal two-phone UI confirmation was not repeated; the authenticated
  A/B installation journey covers the public protocol. The rebuilt candidate
  was not installed because the phone was not attached.

No push, merge into staging, PR, credential/Tailscale change or live-service
mutation was performed.


## Post-review delta

The first bounded review identified and the branch corrected the source-token
parser loss, remote-consent admission recheck, v1 confirmation correlation, and
second-device UI confirmation. The final revoke path prefers the live gateway
mutation so its in-process Full consents are cleared, then falls back to the
independent owner when the gateway is unavailable.

The second bounded review (`CORRECT — hold staging merge`) identified three
remaining admission gaps. Code/test commit `9588d899` now:

- rechecks the source token after hydration and again immediately before
  transcript/dispatch effects;
- rechecks the steering target and captured scope at the same points and has
  the loop consume `webui_request_intent`/`webui_target_turn_id` at bus and
  pending-injection admission;
- restores a rejected queued prompt snapshot (text, images, quote, mentions,
  workspace scope) to a held waiting state, retried only by explicit user
  action, and removes the duplicate failed optimistic bubble.

The new hydration cases fail on `c0b0bf8d`; the corrected branch passes. Full
Python, WebUI, strict, parity, broker and Android release gates are recorded
above. The rebuilt signed candidate is ready for the independent M38 review.
