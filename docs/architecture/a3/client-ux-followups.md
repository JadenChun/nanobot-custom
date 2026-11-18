# Client UX follow-ups — 2026-11-16

## Git checkpoint — 2026-11-18

These three fixes are committed on `codex/staging-audit-implementation`,
separate from staging:

| Commit | Scope |
| --- | --- |
| `02a1cc77e` | CUI-7: recover exact admission for negotiated plain-text sends in existing durable chats, while the original client survives. No automatic resend or general Retry. |
| `0d7f753b2` | CUI-4: include branded icons in Fedora and Android notification payloads. No change to permissions, delivery, or activation. |
| `e7e110df0` | CUI-5: reveal the latest eligible history reply word by word, not by wiping the viewport. |

The CUI-5 integration run initially found 20 thread-shell assertions that
looked for one contiguous text node. The browser still rendered the whole reply;
Streamdown splits it across word spans. The tests now compare exact normalized
paragraph text, preserving both presence and absence assertions. The production
reveal is unchanged by this test correction. In an isolated macOS Chrome session
with the candidate gateway and a local scripted provider, the full reasoning
disclosure, negotiated send receipt, history reveal, formatted text, and reduced
motion worked.

Final verification passed **1,754 WebUI tests / 110 files**, the production
TypeScript check and WebUI build, ESLint, Ruff and diff checks. The intermediate
input-recovery snapshot passed **661 Python / 324 WebUI tests**; the subsequent
icon snapshot passed **13 Python / 4 WebUI tests**. Frozen full-source Python
checking was clean at baseline and candidate (**547 → 548 files, zero new
diagnostics**). The broader, non-build TypeScript config retained its baseline
test diagnostics (140 before and after; none new). No app was installed or
deployed. Earlier local/uncommitted notes below refer to pre-publication snapshots.

**Android remote image loading remains a device verification item.** The branch
already has the November 4 native streaming change (`11c86ccf2`) and the later
cache-pressure correction (`0b9421792`). The current candidate passed 228 WebUI
and 230 backend media-area checks and the Go media broker checks; that is not
proof for the reported older APK. On one phone, use a matching current APK and
gateway, reopen the affected chat, and load a new uncached image. Record which
image kind fails if the symptom persists. No image code changed in this batch.

Actual OS notification delivery, icon rendering and click/navigation, and
general Retry or recovery after app restart remain open. See the
[handoff checkpoint](implementation-handoff.md#git-checkpoint--2026-11-18).

## Current checkpoint — 2026-11-18 (CUI-5)

**Verified in the local WebUI.** Opening a conversation no longer wipes the
transcript upward. Streamdown's existing word fade applies to the latest short,
completed assistant reply; older replies, long replies and live streaming keep
their usual rendering. Reduced motion disables the fade. Bottom restoration
and composer docking stay in the viewport; no custom animation controller or
Markdown parser was added.

One new regression and the existing viewport/Markdown checks passed (102 tests),
along with TypeScript, affected ESLint and diff checks. A local macOS browser
preview showed the opacity-only fade, intact formatted content and the transcript
at the bottom. This change is local and uncommitted on
`codex/staging-audit-implementation`; it has not been built or installed. The
CUI-4 delivery and CUI-7 recovery gaps below remain separate.

## Current checkpoint — 2026-11-18 (CUI-4-L1)

**PASS_SOURCE_NOTIFICATION_ICON_PAYLOAD** — source qualification for native icon
identity and payload resources only. This supersedes earlier “not started” CUI-4
statements below only for that narrow scope. It does not close the original
intermittent delivery report, notification permissions/background behavior,
activation or chat navigation, or OS rendering and delivery.

The bounded correction keeps the existing completion notification owner,
phone/client eligibility, authenticated same-origin desktop mutation, focus and
preference checks, hidden Fedora-shell suppression and best-effort failure
behavior. Fedora passes the existing bundled
`brand/nanobot_icon_192.png` path as a distinct `notify-send --icon` argument
when the file exists; a missing static bundle or icon keeps the existing
notification behavior. Android's actual `mobile-bootstrap.js` bridge adds the
small and large drawable names to its existing immediate notification. It
retains `isExactNotification=false`, the current permission and failure
behavior, and does not add a schedule/alarm. The foreground keepalive uses the
same small branded resource. The 24dp white-alpha fox mask uses an even-odd
silhouette with transparent eye/nose cutouts; the 192×192 Android bitmap is
byte-identical to the existing WebUI brand asset.

Independent root qualification passed the authenticated notification and
listener/auth backend cases (**13 cases**), the complete WebUI suite
(**1,753 tests / 109 files**), TypeScript `--noEmit`, ESLint, Ruff and diff
checks. Four actual client→hook→bootstrap cases passed with a stub native
notification plugin; they verify the existing owner/focus/eligibility path and
icon-bearing schedule payload, not OS delivery. The unchanged full checker
reported **548 files / 0 errors / 0 warnings / 0 information**, and the global
type-regression comparator found **0 new diagnostic signatures**. Dependencies
and the Python 3.13.12 / 216-distribution environment were unchanged. The
vector preview and canonical PNG were visually inspected; the preview is not
an Android resource-compiler or OS-rendering result.

Worker focused evidence (including intended RED/GREEN and the separate fixture
setup error) and root receipts are in
`/Users/jaden/.codex/qa/nanobot-cui4-20261118-du14uo_z/`. The worker's focused
WebUI run used the preinstalled Bun 1.3.10; root's full run used cached Bun
1.4.2 offline. Root also replayed original HEAD sources in disposable probes:
the original bootstrap had the one expected missing-icon failure plus three
passing controls, and the original desktop handler had the one expected missing
`--icon` failure plus seven controls. An initial external integration probe's
four `disconnect` cleanup errors were corrected in QA by using the existing
`close` method; they are retained as setup errors, not product failures.
Exact root command/exit receipts are `root-backend.json`,
`root-notification-chain-qualified.json`, `root-full-webui.json`, `root-tsc.json`,
`root-eslint.json`, `root-ruff.json`, `root-diff-check.json`,
`root-after-basedpyright.json`, and `root-type-regressions.json`; the paired
`.log` files retain their output.

The seven implementation/test/resource paths and these two progress records
are local, uncommitted changes on `codex/staging-audit-implementation` at
`a89acead0d4287d1b35c5ce73a6bcfbad6759eb3`. Root's before/after qualification
integrity records show the implementation paths, branch, HEAD, installed target
and dependency environment unchanged during independent verification. No build,
install, service restart, device action, commit, push, merge or deployment
occurred. The full Python repository suite, P6 completion, broader A3 and release
acceptance are not claimed.

**Still open:** actual permission/background delivery and click/navigation
verification on the next single remote device, Fedora OS rendering/activation,
and browser/PWA behavior on macOS. No native macOS notification host or push
transport was added. CUI-5 remains untouched. CUI-7 general Retry, recreated
client recovery and installed/native qualification remain open; its prior
bounded qualification and limits are retained below. Do not treat this icon
source result as overall CUI-4 reliability or platform delivery acceptance.

## Current checkpoint — 2026-11-18

**PASS_SOURCE_LIVE_INPUT_RECONCILIATION** — a bounded CUI-7 source qualification,
not completion of the original mobile/remote report. This entry supersedes the
open canonical-binding/readback statements in earlier dated checkpoints only for
the negotiated scope below. Other historical qualifications remain unchanged.

The implementation and these records are **local, uncommitted changes** on
`codex/staging-audit-implementation`, based on
`a89acead0d4287d1b35c5ce73a6bcfbad6759eb3`. A fresh origin fetch confirmed that
commit still matches `origin/codex/staging-audit-implementation`. Git identity is
**Jaden <changherng.jc@gmail.com>**; this package creates no new commit or push.
Staging remains separate at `f1a6bb85588ba044fc2ca50e3e9f41eb2f13b6f9`.
There was no merge, PR, build/install, service restart or deployment. The installed
release remains the unchanged October 20 release.

### Supported behavior and minimum contract

Newly negotiated **ordinary plain-text `request_intent=new` inputs in an existing
durable canonical chat**, while the original `NanobotClient` pending owner
survives, can recover a lost admission reply without resending. Direct sends and
queued sends consume the same qualified canonical fact. Positive input admission
does not claim model/tool completion or restart unknown effects.

The confirmed architectural gap was missing durable browser-attempt provenance:
repeating a valid unbound v1 frame after completion produced two canonical user
events and two scripted model calls. The gateway payload cache and source
event-ID CAS protect different invariants. The correction adds evidence to the
existing canonical event, not another execution ledger or prompt inbox:

- `attached.input_attempt_source` advertises version 1, the full source origin
  `(store_identity, session_key, incarnation, generation)`, its existing
  content-free source token and a trusted principal tag, only after a successful
  noncreating canonical read. Temporary, missing and legacy sources do not
  advertise this capability.
- Before first transport send, the pending owner freezes that provenance, its
  exact `turn_id` and submitted-text fingerprint. Async WebCrypto preparation is
  owner-fenced; failure cannot silently send an unstamped negotiated message.
  A later capability never upgrades an older sent/UNKNOWN attempt.
- Optional `message.input_attempt` carries version, original source/token and
  fingerprint, separately from unchanged request-settings v1 permission controls.
  The fingerprint is lowercase SHA-256 over UTF-8
  `nanobot.input-attempt.v1\0new\0` followed by the exact submitted text, with no
  trimming or Unicode normalization; lone surrogates are rejected. The principal
  is server-derived SHA-256 of `nanobot.input-principal.v1\0` followed by `local`
  or `mobile:<installation_id>`, never a caller label or bearer token.
- The trusted writer atomically adds reserved, versioned `_input_attempt`
  evidence to the canonical user event. It binds turn, original origin,
  principal, fingerprint and intent; existing `_event_id`/`_run_id` supply the
  references. Source schema remains version 1, without backfill or additional
  prompt/media copies. An exact duplicate joins the original admission under the
  canonical owner before fresh RunStore/lifecycle/model/tool work. A changed
  principal or payload conflicts; different turn IDs with identical text remain
  distinct legitimate inputs. The existing file/CAS transaction also rejects
  duplicate attempt claims within one incoming batch.
- Existing `attach` accepts at most one optional `input_attempt_query` per chat.
  Its corresponding `attached.input_attempt_result` is either UNKNOWN or admitted
  with exact original proof. Only one valid matching raw canonical user event
  proves admission. The read acquires no execution lease, creates/upgrades no
  session, writes no source/metadata and publishes no input. Mobile permission,
  principal and subscription are rechecked before delivery. Revision/compaction
  changes retain evidence; reset, fork, incarnation/generation or store changes
  cannot transfer it. Missing, malformed, ambiguous or unavailable proof stays
  UNKNOWN, never retry authority.
- The existing post-append `message_admitted` receipt optionally carries
  `input_attempt_proof`. The client checks original source, token, turn,
  principal, fingerprint and current owner/socket before settling its existing
  queue/direct disposition. A small **local-only** optional `onChat` argument
  carries exact-turn `unresolved`/`admitted` to the stream hook, paired through the
  existing bounded event buffer. It is never trusted from the wire. Activity and
  history remain independent: unqualified running/output cannot accept a
  negotiated input; qualified readback can repair its row/waiter after completion
  without reviving a run, changing a newer row/error or discarding a later draft.

Reserved evidence is stripped from untrusted imports/new payloads and public
history, and excluded from model, RunStore and ordinary notification/log
projections. Explicit authorized admission proof is the only new wire carrier.
No global result framework, new RPC, scope expansion or reconnect owner was added.

### Validation, review and limits

Independent final verification passed **1,749 WebUI tests / 108 files**,
TypeScript `--noEmit`, ESLint and diff checks. Independent affected backend
qualification passed **661 tests / 20 files**, with four retained temporary-chat
AsyncMock warnings; Ruff passed. Full unchanged-policy strict checking passed
**548 files / 0 errors / 0 warnings**, versus baseline 547/0; the global comparator
found **0 introduced diagnostic signatures**. Python 3.13.12 and all 216 installed
distributions remained unchanged. The later TS-only correction preserved every
Python/backend/checker hash, binding those results to the final candidate rather
than refreshing a baseline.

Evidence includes deterministic before/after-append and lost-receipt barriers,
atomic-write faults, router/loop recreation, a second-process lock/read/join,
raw checkpoint retention, authorization/read-purity spies, source/principal
replacement, exact queue consumption and actual hook/client recovery before and
after completion. The real authenticated backend attach generated a wire bridge
consumed by the actual frontend client/hook qualification; no external model,
listener, native app or device was used. This is not actual process-crash testing.

A fresh integrated review found that the first candidate protected the queue but
missed the direct hook consumer. One bounded correction added the local delivery
fact above; the real-hook probe moved from **4 intended failures / 3 controls** to
**7 passing cases**, and 40 direct-consumer regressions pass in the full suite.
The same reviewer's narrow recheck found no remaining defect in that scope.
An additional unkeyed-activity diagnostic only retired a redundant dismissible
global notice; the unresolved waiter, UNKNOWN state and inline row/error remained.
It was not treated as admission proof or patched into a new banner-lifetime rule.

Portable regressions are in `tests/agent/test_input_attempt_reconciliation.py`,
`tests/session/test_input_attempt_evidence.py`,
`tests/webui/test_input_attempt_readback.py`,
`webui/src/tests/nanobot-client-input-attempt.test.ts` and
`webui/src/tests/direct-input-admission.test.tsx`. The route-binding fixture now
includes required `request_intent` and asserts actual first publication.
To reproduce the cross-language bridge, set `NANOBOT_INPUT_ATTEMPT_BRIDGE` to a
disposable JSON path for the production-path backend test and then the frontend
run; portable tests remain runnable without that machine-local artifact.
Raw local receipts, frozen contracts/manifests, RED/GREEN probes, reviews and
root verdict are under
`/Users/jaden/.codex/qa/nanobot-cui7-20261118-supzqn6x/`. They are not installation
or deployment artifacts; historical unbound RED evidence is preserved, not
retroactively relabeled fixed.

**Still open:** general Retry/resend and authoritative negative proof; app/client
recreation recovery and any direct prompt/media retention decision; initial-chat
bootstrap, rich input/Steer/control recovery; actual crash/backup rollback and
mixed-version writers; installed macOS and physical single-remote-device
qualification. Older unbound UNKNOWN rows remain held. CUI-4 notifications/icons
and CUI-5 progressive word reveal are still unstarted independent issues. Work on
one issue/owner at a time; do not infer full Python-repository, P6, broader A3,
release or staging-merge acceptance from this source-only qualification.

## Current checkpoint — 2026-11-17

This checkpoint accompanies all completed client fixes on
`codex/staging-audit-implementation`: CUI-6 reasoning disclosure, CUI-2 context
presentation and request-bound accounting, merged-text continuation across
compaction, CUI-3 folder Browse lifetime safety, related locale corrections,
regressions and progress records. They are included together in the client-fix
checkpoint commit, using author and committer **Jaden <changherng.jc@gmail.com>**.
The base before this checkpoint was
`4f598372299f43041beee6fbe2d049b52b678aba`.

The accompanying Git sync targets only
`origin/codex/staging-audit-implementation`; remote publication must be confirmed
against that branch rather than inferred from historical checkpoint text. Staging
remains separate at `f1a6bb85588ba044fc2ca50e3e9f41eb2f13b6f9`.
No staging merge, PR, build/install, operator restart or release is part of this
checkpoint. Earlier local/uncommitted and no-push statements below describe their
pre-publication source-qualification snapshots, not additional pending fixes.

Fresh commit-snapshot verification passed **1,671 WebUI tests / 106 files**,
**151 selected Python tests** (including the six backend scope guards), TypeScript
`--noEmit`, ESLint, Ruff and diff whitespace checks. Frozen before/after full
strict checking covers **547 files / 0 errors / 0 warnings**, with **0 introduced
diagnostic signatures** and unchanged dependencies. This is not the full Python
repository suite or installed/native-device, P6, broader A3 or release acceptance.
The installed nanobot release remains unchanged; no external model was used.

CUI-6 retains its bounded source-preview qualification. **CUI-2 normal context
accounting is now source-qualified**, beyond the earlier unavailable-state
presentation safeguard; this is not blanket all-compaction client or installed
runtime acceptance. **Live rewritten-cumulative text duplication is separately
source-qualified** for the reproduced compaction/length-recovery mechanism;
earlier OPEN records below are historical, not installed-runtime acceptance.
CUI-3 has **PASS_SOURCE_FOLDER_BROWSE_LIFETIME** for the reproduced local
stale-choice mechanism; the broader intermittent installed/native report remains
unqualified. Active-response and unresolved-input guards still intentionally lock
folder mutation, and the existing backend `chat_running` refusal remains
authoritative. No next-turn staging was introduced. CUI-4 notifications/icons,
CUI-5 word reveal, and general CUI-7 safe Retry, canonical same-attempt
deduplication/readback and authoritative restart recovery remain open and separate.
The user deferred package/build/install work;
no bundle, build, install or release operation was performed. Full Python-repository
suite, broader A3, P6 and native/physical-device acceptance are not claimed.

**Historical Git sync checkpoint, 2026-11-16:** the restored-queue anti-replay
fix is committed and published to `origin/codex/staging-audit-implementation` at
`185c3729bc0f28f3225837aa6d75142a9e82ffd6`, including the earlier owner-seam base
`b1d7eb24873b02bb5f52036cabf2bcc80957a873`. Both commits use author/committer
`Jaden <changherng.jc@gmail.com>`. The 2026-11-16 progress record accompanied
that code checkpoint. Local/uncommitted and no-push statements in dated Nov16
pre-sync qualification entries describe those snapshots only; current state is
recorded above. See
[the current handoff checkpoint](implementation-handoff.md#current-checkpoint--2026-11-17).
Staging remains separate at `f1a6bb85588ba044fc2ca50e3e9f41eb2f13b6f9`.
No staging merge, PR, service change or deployment occurred. Broader issues remain open.

This record separates qualified fixes from open user reports. Work them
**one at a time**, with separate reproduction, acceptance evidence and progress.
Use one write owner for shared client files; do not dispatch overlapping UI changes.
Keep the frozen A3 source/approval/admission contracts intact.

The preceding [macOS Stop corrections](macos-qualification-2026-11-16.md) are
qualified separately. They restore project controls after that specific stopped
run; they do not close the remaining client reports below.

| Order / ID | Report and desired outcome | Status |
| --- | --- | --- |
| 1 / CUI-1 | Queue presentation is awkward; Steer sends but leaves the composer text behind. Make queued guidance clear in a separate panel above the input card; remove submitted text without losing rejected/offline/unadmitted drafts or duplicating input. | **Qualified locally**; [record below](#cui-1-queue-and-steer-qualification) |
| 2 / CUI-2 | Context-window data becomes missing or inconsistent after compaction. Restore normal usage updates across compaction; retain truthful model/window information and never invent measurements. | **Normal accounting source-qualified**, including fresh reported/estimated live and reload measurements; not full client/installed acceptance; [latest record](#cui-2-normal-context-accounting--2026-11-17) |
| Separate / live rewrite | Pre-existing live rewritten-cumulative answer text repeats earlier segments, while canonical replay contains them once. | **Qualified locally** for compaction prematurely closing a merged answer stream; separate from the accounting correction and CUI-3; [latest record](#live-cumulative-text-continuation--2026-11-17) |
| 3 / CUI-3 | Project/folder selection is inconsistently disabled in an existing/running session. Distinguish intentional active-turn binding from stale busy state; changing a draft/next-turn selection must not mutate the running turn's workspace authority. | **PASS_SOURCE_FOLDER_BROWSE_LIFETIME** for the reproduced local stale-choice mechanism; intentional active/unresolved-input lock remains and the broader installed/native report is unqualified; [record below](#cui-3-folder-browse-lifetime--2026-11-17) |
| 4 / CUI-4 | Desktop and mobile notifications are unreliable and lack details such as the app icon. Verify permission/subscription/lifecycle, payload, icon and navigation independently on each surface. | **PASS_SOURCE_NOTIFICATION_ICON_PAYLOAD** for native icon identity/resource payload only; overall delivery, permission/background lifecycle, activation/navigation and OS qualification remain open; [latest record](#current-checkpoint--2026-11-18-cui-4-l1) |
| 5 / CUI-5 | Replace the bottom-to-top viewport motion with a progressive word reveal. Preserve existing Markdown, streaming and scroll behavior. | **Verified locally** for the latest short completed reply on history entry; [checkpoint](#current-checkpoint--2026-11-18-cui-5) |
| 6 / CUI-6 | Reasoning is limited to a one-line preview with no usable way to inspect the full provider-supplied reasoning output. Provide accessible expand/collapse and full readable content where the model actually supplies it; do not fabricate unavailable reasoning. | **Bounded UI qualified locally**; [record below](#cui-6-reasoning-disclosure--2026-11-17) |
| 7 / CUI-7 | Mobile reports failed send even though restart reveals the message was sent; remote connection instability appears to disrupt features. Distinguish transport uncertainty from authoritative refusal, reconcile the same attempt, and provide safe explicit retry without duplicate admissions. | **PASS_SOURCE_LIVE_INPUT_RECONCILIATION** for new negotiated plain-text sends in existing durable chats with a surviving client owner; prior queue anti-replay qualification retained. General Retry, recreated-client recovery and installed/native qualification remain open; [latest record](#current-checkpoint--2026-11-18) |

## CUI-3 folder Browse lifetime — 2026-11-17

**PASS_SOURCE_FOLDER_BROWSE_LIFETIME** — bounded source-only qualification for
the reproduced stale-choice mechanism. This does not close the broader original
installed/native report.

Ordinary idle folder changes in a regular chat remain available. An active
response or unresolved input intentionally locks folder mutation; the existing
server `chat_running` refusal remains authoritative. No active-turn change,
draft/permission redesign or next-turn folder staging was added. The ordinary
idle → running → `turn_end` availability transition already worked.

The confirmed local defect was that Browse checked eligibility before awaiting
the folder choice, then retained the old callback authority if chat owner,
semantic path/access-mode focus or eligibility changed before resolution. A late
choice could therefore mutate a different or active chat; server refusal was a
downstream guard, not the cause.

The picker now owns one pending Browse identity and permanently revokes it on a
committed owner/focus/eligibility change or unmount. It checks the operation
before any late success/error effect and fences finalization so an obsolete
request cannot clear a newer request's busy state. A valid result uses the latest
committed apply callback. The existing chat identity is forwarded through two
narrow picker props. Equivalent same-owner metadata and freshly bound native
picker methods keep the operation valid. No new session authority, async
cancellation protocol or next-turn staging was added.

Seven added regression cases cover the real ThreadShell lifecycle and fresh
choice, owner and focus changes followed by return, equivalent native refresh,
stale rejection/finalization while a replacement is pending, and unmount. In the
valid original decisive/control RED, one control passed and the intended stale-
application assertion failed; the validity-boundary RED had one pass and two
intended failures. The corresponding final sets pass **2/2** and **3/3**. The
setup-invalid artifacts are excluded. See
`folder-diagnosis-valid-red.{json,log}` and
`folder-validity-analysis-valid-red.{json,log}`. The closest composer/shell
selection passes **274 tests / 2 files**. The full WebUI suite passes **1,671
tests / 106 files**. TypeScript
`--noEmit` and ESLint pass. Six selected existing backend scope guards pass;
they were validly reused against unchanged Python hashes. Full-source strict
before/after checking reports **547 files / 0 errors / 0 warnings**, and the
global diagnostic comparator reports **0 introduced signatures**. Counts overlap
across focused and full selections and must not be summed. No full Python
repository suite is claimed.

The frozen pre-docs source manifest checks **2,786 paths** with no unexpected
changes; all **1,517 manifest Python sources** are unchanged. Python **3.13.12**,
**216 distributions** and sampled October 20 installed link/files remain
unchanged. The installed release remains October 20.

From `webui/`, the primary source checks were:

~~~sh
npm exec --offline --yes --package=bun@1.4.2 -- bun run test src/tests/thread-composer.test.tsx src/tests/thread-shell.test.tsx
npm exec --offline --yes --package=bun@1.4.2 -- bun run test
npm exec --offline --yes --package=bun@1.4.2 -- bun run node_modules/typescript/bin/tsc -p tsconfig.build.json --noEmit
npm exec --offline --yes --package=bun@1.4.2 -- bun run lint
~~~

The unchanged strict check was `.venv/bin/python -m basedpyright --outputjson`;
the frozen global comparator was:

~~~sh
.venv/bin/python scripts/check_type_regressions.py \
  /private/tmp/nb-cui3-folder-20261117-e02zb96v/baseline-basedpyright.json \
  /private/tmp/nb-cui3-folder-20261117-e02zb96v/after-basedpyright.json \
  --baseline-root /Users/jaden/.codex/worktrees/audit-architecture-records/nanobot \
  --current-root /Users/jaden/.codex/worktrees/audit-architecture-records/nanobot
~~~

Coordinator receipts also retain the exact six-test backend command. The
`git diff --check` PASS is **pre-docs** evidence from `pre-docs-integrity.json`;
it does not validate these documentation edits.

Evidence root: `/private/tmp/nb-cui3-folder-20261117-e02zb96v/`. See
`precheck.json`, `pre-docs-integrity.json`, `integrated-review.md`,
`focused-final.{json,log}`, `diagnosis-final.{json,log}`,
`validity-final.{json,log}`, `frontend-final.{json,log}`,
`typescript-final.{json,log}`, `eslint-final.{json,log}`,
`backend-scope-guards.{json,log}`, `after-strict-command.json`,
`after-basedpyright.{json,stderr,exit}` and
`type-comparator-final.{json,log}`,
`folder-diagnosis-valid-red.{json,log}` and
`folder-validity-analysis-valid-red.{json,log}`. Final documentation integrity
and the graph verdict are recorded separately in `accepted-integrity.json` and
`acceptance.json`; pre-docs source checks alone are not final documentation
acceptance.

The broader intermittent installed/native behavior remains unreproduced and
unqualified. No actual native dialog, physical-device qualification, real gateway,
build/install, service restart or external/paid model call occurred. General CUI-7
safe Retry/canonical deduplication/readback/restart recovery, CUI-4
notifications/icons and CUI-5 reveal remain separate/open. Broader A3, P6,
full-release and installed/native closure are not claimed.

## Live cumulative-text continuation — 2026-11-17

**PASS_SOURCE_LIVE_REWRITE**, scoped to the reproduced compaction/length-recovery
mechanism; not blanket client, native/device, installed-runtime or release acceptance.
The debugging workflow confirmed a **local implementation defect**, not a missing
architectural state. `resuming=true, merge_next=true` preserves one answer and the
server's cumulative text buffer. The live hook then incorrectly closed that answer
on a compaction notice. Subsequent deltas opened new bubbles, and cumulative
`stream_end.text` replaced only the newest bubble, repeating earlier segments.
Canonical replay already retained its answer buffer correctly.

The production correction changes only the compaction branch's existing flush
call: flush pending text **without closing the answer stream**. Existing stream-end,
tool, reasoning and turn boundaries still own closure. No new state, text-prefix
heuristic, deduplication layer, wire/schema, source/persistence, provider, accounting
or animation change was added. The two closest hook regressions protect one stable
answer through two compactions with changing signed URLs, and two genuinely closed
non-merged segments remaining separate.

Before correction, the hook regression produced **3 answer rows instead of 1**;
the closed-segment control passed. Captured actual-source frames in real
`ThreadShell` produced two live failures (**3/2/1** marker counts), with both
canonical reload controls passing (**1/1/1**). After correction the same frames
are **1/1/1 live and reload** for reported and estimated usage. Visible answer text
matches across both views, and the rendered image uses the exact latest signed URL.
All **16** captured-frame UI checks pass: six modern root/tool/length scenarios
live/reload plus two rewritten scenarios live/reload. Context remains **42%** for
root/tool and **17%** for length recovery; the accounting fix is preserved.

Validation: full WebUI **1,664 tests / 106 files**, TypeScript `--noEmit`, ESLint,
Ruff and diff whitespace checks pass. Closest Python producer/replay selection:
**70 passed**, not the full repository suite. Frozen full strict before/after:
**547 files / 0 errors / 0 warnings**; comparator **0 introduced diagnostics**
globally. Python **3.13.12**, **216** distributions and dependency hashes remain
unchanged. Source hashes confirm that only the two scoped frontend files and these
progress records changed; all preceding uncommitted work is preserved. The sampled
October 20 installed link/files are unchanged. No commit, push, merge, build/install,
operator service restart or
external/paid model call occurred. Installed/native/physical-device, P6, A3 and
release acceptance remain unverified/open.

Evidence: `/private/tmp/nb-live-rewrite-20261117-2u27146w/` contains `scope.md`,
`diagnosis.md`, RED/GREEN command receipts, `actual-frame-final-value.{json,log}`,
`ui-semantic-comparison.json`, full frontend/types/lint/producer/replay receipts,
frozen strict reports/comparator and source/environment/install hashes. Probe
setup and asynchronous-render observation failures are retained, not counted as
passes; the final probe awaits the exact rendered value without product workarounds
or weaker assertions. The prior OPEN finding below records the accounting-stage
checkpoint; this later correction supersedes it only for the reproduced mechanism.
Next independent work is CUI-3 folder-selection availability; general CUI-7,
notifications/icons and reveal animation remain open.

## CUI-6 reasoning disclosure — 2026-11-17

**Bounded source-preview PASS.** The shared reasoning row now keeps its compact
preview by default and provides a keyboard-operable expand/collapse control for
the complete supplied text. The full body uses the existing lazy Markdown renderer,
updates while streaming, and is unmounted while closed. The open/closed choice is
per mounted row and survives appends and completion. Empty or whitespace-only
reasoning has no disclosure. Existing thinking/check markers, streaming sheen and
outer grouped-activity fold remain intact.

The shared leaf is exercised in standalone MessageBubble and in
AgentActivityCluster; the standalone answer remains separate. No generation,
provider, transport, persistence, activity-grouping or tool-content path changed.

The parent verified the actual components and source styles at **1280 × 720** and
**390 × 844** using supplied fixture text only. The opened sample showed its heading,
36 paragraphs, code block and end marker; appending preserved the open state.
Closing before completion stayed closed and removed the full-body descendants.
The grouped outer fold kept its existing behavior, and Enter opened one reasoning
row while Space closed it without opening its sibling. The answer stayed separate.
At the narrow viewport, the page width remained **390 px**, with the end and
appended markers reachable by scrolling and no horizontal page overflow. The
viewport was reset and the temporary preview server was stopped. This is source
preview evidence only; no gateway, provider, model, installed package, native app
or physical-device behavior was exercised.

### CUI-6 validation and limits

The pre-edit regression was RED: the focused run reported three new disclosure
failures with the five existing files passing 145 tests. The final focused command
passed **150 tests / 6 files**:

~~~sh
npm exec --yes --package=bun@1.4.2 -- bun run test \
  src/tests/reasoning-row.test.tsx src/tests/reasoning-preview.test.ts \
  src/tests/thinking-reasoning-shell.test.tsx src/tests/message-bubble.test.tsx \
  src/tests/agent-activity-cluster.test.tsx src/tests/thread-messages.test.tsx
~~~

Parent verification also passed the full WebUI suite (**1,631 tests / 106 files**),
TypeScript --noEmit, ESLint, and git diff --check. Screenshots and browser
observations are retained in
/private/tmp/nb-cui6-reasoning-20261117-JwbUuymJ/.

The nine non-English common.json recovery-notice additions were a separate
root-owned preflight repair (i18n parity **26/26**), not CUI-6; HEAD remains
4f598372299f43041beee6fbe2d049b52b678aba with local, uncommitted changes, while
the Python 3.13.12 / 216-distribution environment and October 20 installed hashes
remain unchanged.

At the 2026-11-17 CUI-6 checkpoint, CUI-2 context accounting was next. It has
since received the bounded presentation qualification below. The user's subsequent
clarification keeps CUI-2 open before CUI-3; CUI-4 notifications/icons and CUI-5 word
reveal remain open.
General CUI-7 safe Retry, canonical same-attempt deduplication/readback and
authoritative recovery after restart remain open, as do native/physical-device
qualification and broader A3 acceptance. These bounded UI results do not qualify
a release.

## CUI-2 normal context accounting — 2026-11-17

*Historical accounting qualification: the then-OPEN live-text finding below is
superseded, only for the reproduced mechanism, by the separate
[live continuation correction](#live-cumulative-text-continuation--2026-11-17).
The accounting verdict and its other limits remain unchanged.*

**Source-accounting PASS (`PASS_AMENDED_SOURCE_ACCOUNTING`); not full client,
installed-runtime or release acceptance.** The proven modern-source length-recovery
case already measured a fresh request after compaction, but cumulative answer text
remained on an earlier row. Replay stamped completion usage onto a compaction notice,
and row-position inference discarded the otherwise valid measurement. Text location
cannot establish when a request was prepared or measured.

The minimal correction keeps one immutable current-context snapshot and one
overwritten final-prepared-request receipt, not a ledger:

- The receipt captures the invocation-local invalidation epoch and exact consumed
  route's window. Successful context replacement, including silent replacement,
  invalidates the snapshot; started/failed/cancelled phases without replacement do
  not. A native success after dispatch cannot promote that older response's usage
  into the new epoch, even if usage is recorded later. A genuinely later reported
  or existing estimated main-request measurement restores context normally.
- **Omitted = no update; present explicit-null = actual invalidation and a hard
  barrier.** Explicit-null cannot fall back to aggregate spend or older snapshots.
  Measured zero remains valid; an unmeasured request without replacement preserves
  the prior observation and its own denominator, not a different route's window.
- The existing terminal hook, scoped delivery/publisher slot and typed completion
  carry the snapshot through wire, live projection and replay. Exact turn ownership
  and completion sequence, not carrier position, qualify it across succeeded
  compactions; same-turn notice/user metadata can carry it without inventing an
  answer. No canonical-text segmentation, source schema/checkpoint persistence,
  provider dispatch policy, billing or round aggregation changed.

Coordinator-rebuilt source probes cover root/tool/length compaction with both
reported and estimated usage: root/tool restore **5,000/12,000 = 42%**, length
**2,000/12,000 = 17%**, live and reload. Length's cumulative spend remains
**12,000**; auxiliary summary **50** is never current context. Six modern cases
and two real signed-Markdown/media rewritten cases pass source chronology and
canonical verification; actual `ThreadShell` live/reload checks pass **12 + 4**.

### Separate OPEN followup — live rewritten-cumulative text

For identical frames, actual pre-edit and candidate live views both contain
`first_segment` **3x**, `second_segment` **2x**, final **1x** (**3/2/1**); canonical
reload is **1/1/1**. The visible canonical answer text nodes are byte-identical
before/after, including signed media URLs. This accounting patch **neither
introduces nor fixes** that live duplication. AC2 is explicitly amended to
**no new text regression plus exact-once canonical replay**, not blanket live
exactly-once or complete CUI-2 all-compaction-client acceptance. Keep the live-text
defect **OPEN** for independent next scoped work; do not fold it into CUI-3.

### Accounting validation, evidence and limits

Coordinator final selected Python: **1,918/1,918 passed in 137.07s**, exit0; this
is the expanded affected selection, not the full Python repository suite. The
initial expanded run failed: four delta-caused incomplete delivery-test-double
failures were corrected by one faithful no-hook fixture field with every original
assertion retained. Long macOS AF_UNIX temporary paths and tight API timing under
heavy concurrent checks were resolved by short temporary paths and a sequential
rerun of the same selection, with no product workaround, test skip or threshold
weakening. The original failure logs remain evidence, not a pass.

Full WebUI **1,662 tests / 106 files**, TypeScript `--noEmit`, ESLint, Ruff and
diff whitespace checks pass. Reusing the full frontend result after the fixture-only
correction is justified by hashes: all original 26 candidate paths, including all
frontend source/tests, stayed byte-identical; the 27th path is only the Python
delivery double. Repeated full strict passes **547 files / 0 errors / 0 warnings**;
the frozen-baseline comparator introduces **0** diagnostic signatures globally.
Python **3.13.12**, all **216** distributions and dependency hashes are unchanged.

Evidence root: `/private/tmp/nb-cui2-accounting-20261117-aebB1h/`. See
`code-qualification.json`, `review/correction-recheck.md`,
`root-python-corrected-final.{json,log}`, `root-python-suite-files.txt`,
`root-frontend-final.{json,log}`, `root-corrected-basedpyright.json`,
`root-corrected-type-comparator.{json,log}`, `root-source-probe/`,
`root-rewrite-baseline-comparison.json` and `review/evidence.json`.
Compact reproduction commands follow; saved receipts retain the exact invocations
and complete selected-path list (run sequentially, not concurrently):

~~~sh
QA=/private/tmp/nb-cui2-accounting-20261117-aebB1h
# Repository root; compact equivalent of the recorded selected Python command.
xargs .venv/bin/python -m pytest -q -o cache_dir="$QA/root-final-short-cache" \
  --basetemp=/tmp/c2fCJH1XW < "$QA/root-python-suite-files.txt"
.venv/bin/python -m basedpyright --outputjson
.venv/bin/python scripts/check_type_regressions.py "$QA/baseline-basedpyright.json" \
  "$QA/root-corrected-basedpyright.json" --baseline-root "$PWD" --current-root "$PWD"
uv run --no-sync ruff check nanobot/
uv run --no-sync ruff check tests/agent/test_migration_fresh_turn.py
git diff --check
# webui/
npm exec --offline --yes --package=bun@1.4.2 -- bun run test
npm exec --offline --yes --package=bun@1.4.2 -- bun run node_modules/typescript/bin/tsc -p tsconfig.build.json --noEmit
npm exec --offline --yes --package=bun@1.4.2 -- bun run lint
~~~

Legacy absent proof after successful compaction remains unknown; unknown fallback
capacity cannot borrow a configured/primary denominator. Success without a later
measurement remains unavailable. These are truthful limits, not fabricated normal
usage. Source tests included **disposable private loopback gateway/remote-owner
fixtures**; no matching disposable processes remain. No operator service was
restarted. The October 20 installed link and three sampled hashes are unchanged;
no external model, installed-runtime, native/physical-device, P6, A3 or release
acceptance is claimed. Build/install remains deferred.

The following configured-capacity section is the **historical presentation-only
checkpoint, retained unchanged**; its pending-accounting statements describe that
earlier checkpoint, not the latest source-accounting verdict above.

## CUI-2 configured capacity and unavailable usage — 2026-11-17

**User clarification: this does not complete CUI-2.** Normal context usage should
continue after compaction. Keeping configuration visible and labelling an unknown
count is only a presentation safeguard, not the requested accounting correction.
The existing pipeline clears the stale request measurement when source projection
changes; compaction completion carries only its ID and phase, not a replacement
measurement. Internal rebuilt-request measurements/estimates exist, but are not
published by that event. Ordinary later measured replies restore usage in the
controlled tests. Persistent absence after such a reply has not been reproduced
against the user's actual run. The next work is to trace the full compaction and
subsequent request/live/replay path, establish where the replacement should come
from, and restore truthful normal updates without reusing stale counts. Do not
move to CUI-3 or treat an unavailable label as final acceptance.

**Bounded presentation qualification PASS; no immediate post-compaction metric or
effective-fallback provenance is claimed.** `ThreadShell` derives a separate
informational descriptor from the exact selected settings preset (local choice,
session choice, then configured default). It exposes only a nonempty configured
model and a finite, positive integer configured window. `ThreadComposer` passes
that descriptor to the existing context popover. The selected model/window remain
visible after successful compaction even when the last request measurement is
correctly invalidated. The popover states **Usage unavailable** with no zero,
percentage or meter in that state, keeps recent rounds independent, and labels any retained request ratio
**Last known request** against that snapshot's own window. The no-round mobile
context control remains present.

The successful-compaction invalidation guard and message ordering are unchanged.
Started, failed and cancelled compactions retain the previous measured snapshot;
a newer ordinary response restores its own measurement. A selected preset never
borrows the global agent window if its row is absent or invalid. Configured values
are not proof of effective runtime/fallback capacity. No active-runtime hint,
event, hook state, persistence, ledger, estimate, wire change or Python/provider
change was added. A genuinely post-compaction numerator can remain unknown until
a newer request is measured. Richer immediate post-compaction metrics or exact
effective-fallback provenance would require a separate scope if requested.

### CUI-2 validation and limits

The approved requirement regressions were RED before the UI change. The worker's
focused command passed **380 tests / 6 files** with preinstalled Bun **1.3.10**:

~~~sh
cd webui
bun run test src/tests/thread-shell.test.tsx src/tests/thread-composer.test.tsx \
  src/tests/agent-activity-cluster.test.tsx src/tests/message-bubble.test.tsx \
  src/tests/reasoning-row.test.tsx src/tests/i18n.test.tsx
~~~

Parent verification passed the full WebUI suite (**1,644 tests / 106 files**) with
cached/offline Bun **1.4.2**, TypeScript `--noEmit`, lint and `git diff --check`.
Exact parent commands (the first three from `webui/`; all used the existing offline
package cache and did not change dependencies):

~~~sh
npm exec --offline --yes --package=bun@1.4.2 -- bun run test
npm exec --offline --yes --package=bun@1.4.2 -- bun run node_modules/typescript/bin/tsc -p tsconfig.build.json --noEmit
npm exec --offline --yes --package=bun@1.4.2 -- bun run lint
git diff --check
~~~

Frozen strict reports before and after both contain **0 errors / 0 warnings across
547 files**; `check_type_regressions.py` returned `PASS_NO_NEW_DIAGNOSTICS` with
zero introduced diagnostics under the same Python **3.13.12 / 216-distribution**
environment. This client-only work changed no Python source; the full Python test
suite was not run or claimed. `source-preservation.json` verifies all prior locale
values and **1,515 Python paths** remain unchanged. The October 20 installed link
and its three sampled hashes are unchanged; no build/install/release operation
was performed.

The strict check used `uv run --no-sync basedpyright --outputjson` for both frozen
reports; the global comparator command passed:

~~~sh
.venv/bin/python scripts/check_type_regressions.py "$QA/baseline-basedpyright.json" \
  "$QA/after-basedpyright.json" --baseline-root "$REPO" --current-root "$REPO"
~~~

In that recorded command, `$QA` resolved to the CUI-2 evidence directory and `$REPO`
to this checkout.

The parent previewed actual `ThreadComposer`, `ComposerUsagePopover`,
`ContextCompactionNotice` and app styles at **1280 × 720** and **390 × 844** using
supplied fixture state, not `ThreadShell`. Return opened the no-round control;
the configured model/window (**65.5K**) and unavailable usage were visible without
a meter or percentage. The existing **14.7K / 65.5K (22%)** and supplied newer
**5.2K / 65.5K (8%)** snapshots were labelled Last known request, rounds remained
available, and the narrow page had no horizontal overflow. Removing config,
measurement and rounds restored the previous hidden state. ThreadShell live/replay
coverage is from its owning tests, not the browser preview. The viewport was reset
and preview PID **76381** was stopped. No gateway, provider, model, native app or
physical device was exercised.

Evidence is retained in `/private/tmp/nb-cui2-context-20261117-_2jn61sv/`, including
`analysis/report.md`, worker RED/focused logs, `root-full-webui.log`, typecheck/lint
receipts, `root-type-regressions.log`, `browser-observations.json`,
`source-preservation.json`, `installed-after.json` and `preview-stopped.json`.
This qualification does not close CUI-3 through CUI-5, general CUI-7, broader A3,
P6, release, native or physical-device acceptance.

## CUI-7 bounded owner-seam and restored-queue qualification — 2026-11-16

**Bounded package PASS.** The two owner-seam defects and restored-row anti-replay
behavior are validated for the current WebUI client within single-device scope using
the browser fixture described below. This
supersedes the historical restored-row gap and pending-acceptance statements below;
it does not close general safe Retry, authoritative same-device restart recovery or
native/physical-device qualification. The accepted owner-seam code is in base
`b1d7eb24873b02bb5f52036cabf2bcc80957a873`; the restored-queue implementation is
published in `185c3729bc0f28f3225837aa6d75142a9e82ffd6` on
`origin/codex/staging-audit-implementation`. No staging merge, PR or deployment occurred.

### Behavior and minimal architecture

- `SessionTurnStarted` remains lifecycle information. One typed
  `UserInputCommitted` fact now produces the existing `message_admitted` receipt
  only after this invocation's successful user append. Normal source/legacy turns
  and applicable command/shell paths use their existing append owners. False or
  failed append emits no receipt. `/new` and other no-input paths invent none;
  temporary-chat acknowledgement retains its connection-owned lifetime, not a
  disk/restart guarantee. Active Steer's post-append `InjectionCommitted` is unchanged.
- Authorized root history with explicit idle activity fields can display committed
  replies and retire optimistic Working independently of UNKNOWN admission. Missing,
  active or incomplete activity metadata remains conservative. Existing source,
  lineage, request, newer-run, durable-tail and final UI-revision fences remain.
- UNKNOWN stays in the existing client attempt owner. History does not resolve its
  admission promise or replay its input. Its original queue row stays held. Focused
  tests preserve later rows and drafts for the exercised receipt paths. A matching
  late receipt removes only that row without reviving old ownership or automatically
  draining the next item.
- The existing queue row now carries a versioned submission state and a
  content-free original binding. Remounts, chat switches and a recreated client
  keep submitted/ambiguous rows held with mutation controls disabled. The owner
  join is read-only; only an exact receipt or the already accepted live running
  proof can resolve that original binding. Unknown/missing-owner outcomes do not
  become refusals or retries. Legacy, malformed and unsupported provenance stays
  held; genuinely unsent legacy rows may therefore be stranded.
- Queue receipt consumption matches the original queue key, prompt id and turn id.
  Persistence errors, failed readback and cap exclusion block dispatch before
  upload/send, and the existing 20-row/4,000-character limits remain in force.
- The existing automatic-send guard consumes the idle edge even when blocked.
  Independent explicit new input remains a different attempt, not Retry. Folder,
  recovery and persistent-chat access-mode mutations stay disabled while UNKNOWN;
  supported known-active permission/revocation controls remain usable.
- Accepted idle notifies existing activity subscribers even if the running clock
  frame was lost, so the sidebar spinner converges with the main conversation.
  The external queue panel is preserved. No second ledger, reconnect owner, source
  schema, prompt inbox or general recovery framework was introduced.

### Earlier owner-seam qualification (historical evidence)

Independent final checks passed **18,492 Python tests / 68 skips / 6 subtests**, plus
the separately run **129 serial CLI tests**; **1,606 WebUI tests / 102 files**;
TypeScript/Vite build, ESLint and Python Ruff. Full strict before/after reports have
**547 files / zero diagnostics**, with zero globally introduced signatures. The
Python source bytes remained unchanged after the accepted checker run, and the
Python 3.13.12 / 216-distribution environment and Bun 1.4.2 were preserved.

Fresh integrated review found two missed consumers (access-mode disabling and the
sidebar idle notification), plus incomplete frontend/backend test doubles. One
bundled client correction and a test-only backend correction resolved them; the
same reviewer narrowly rechecked them and recommended PASS. A full WebUI run
overlapping the full Python run had 59 failures including timeouts/cascades. The
identical unchanged command passed after Python finished. Both results are retained;
the precise contention/flakiness cause is not established or waived by this record.

The final wheel, SHA-256
`641f5be5bc869a17bc4c046d02d725eb90d43a1fc417aeaa29fb3a767bc51d86`,
matched all 547 packaged Python files and 590 WebUI assets in both the checkout and
disposable installation. Installed macOS browser checks used only the scripted
`custom/qualification-offline` provider, not an external/paid model:

1. An automatically drained queued new turn lost its receipt/output and completed
   before actual replacement attach. Its reply appeared once without reload while
   its row remained Awaiting confirmation, the later queued row/draft survived,
   folder/access controls remained blocked, and neither main nor sidebar said busy.
2. Relaying the actual, byte-identical delayed original owner receipt changed the
   queue from two rows to one. No next-item send, old-run revival or draft loss
   occurred. Wire/provider counts stayed at two: one parent plus one queued turn.
3. Reattaching while another turn remained active preserved Working and its draft;
   later completion converged to one reply and idle, with one message/model call.

Desktop and 390 × 844 captures verified the held queue outside the input card.
This is responsive-browser evidence, not physical Android/native-app qualification.
All five disposable gateway/provider/proxy harnesses exited; the viewport was reset.
No operator gateway, service, device, deployment or Git publication was changed.
Raw local evidence is in `/private/tmp/nb-cui7-seams-20261116-xy1unsld/`:
`acceptance.json`, `integrated-review-recheck.md`, qualified manifests/diff, raw
`root-*` check receipts, `browser-final-observations.json`, and the installed wire,
source, API and screenshot records. Temporary evidence is not portable to another
device; this document records the essential behavior, outcomes and limits.

### Restored-queue anti-replay validation — final bounded evidence

This package adds held-attempt provenance to the existing queue rows and joins the
original outcome through the existing `NanobotClient` owner. The binding is
content-free (`queueKey`, `promptId`, `turnId`); it is not a wire authority token or
a second ledger. Submitted/unknown rows stay read-only through remount, chat switch
and recreated-client restoration. An exact refusal remains manual-only, an exact
admission consumes only its matching row, and generic errors, missing owner records,
history/idle or stale activity do not authorize replay. Running-based admission
requires numeric `started_at` and passes the canonical completion fence first. The
exact `message_admitted` receipt retains its accepted late-repair exception without
reviving old activity.

The two bounded correction rounds completed on 2026-11-16. Round one corrected the
canonical-fenced running proof and the shell test client's typed owner join/consume
double, filled all ten locale resources, and added focused storage and owner
regressions. Round two removed only one unused type import. Final parent
verification reported:

- Full WebUI: **1,624 tests / 105 files passed**.
- External restored-queue baselines: **4 tests / 2 files passed**.
- TypeScript/Vite build passed with the existing chunk-size advisory; final lint
  passed. Narrow integrated review found the original defects resolved and no
  remaining production defect.
- Owner/storage focused tests: **16/16 passed** after the final import-only edit.
  The preceding focused composer/adjacent set passed **151/151**, shell passed
  **106/106**, and locale parity passed **26/26**.

The full WebUI, external baseline and build ran on the source frozen immediately
before the unused-import removal; production hashes were unchanged. Parent reran
the affected owner/storage tests and final lint afterward. Raw evidence is in
`/private/tmp/nb-cui7-restored-20261116-f4f8t2qi/`: `root-full-webui-final.log`,
`root-external-final.log`, `root-build-final.log`, `root-lint-qualified.log`,
`root-focused-qualified.log`, and the correction-round logs. The Python manifest
inputs contain **1,515 files with no changes**; the **216-distribution Python
3.13.12 environment** is unchanged. Python tests, strict checking, wheel packaging
and native app qualification were not rerun for this client-only package.

The browser fixture for the single-device scope uses the actual `ThreadShell`,
`useNanobotStream`, `NanobotClient` and app styles with fake socket/fetch transport.
At default **604 × 852** and narrow **390 × 844**, the observed
scenarios kept an original submitted row held across remount; delivered an exact
receipt while A was offscreen, preserving B's current draft and consuming only A's
original on return; preserved a new draft on a duplicate/stale receipt; and kept a
recreated-client UNKNOWN row held with mutation controls disabled and no automatic
next-row drain. The narrow layout had no horizontal overflow. Four screenshots are
`queue-held-remount.jpg`, `queue-exact-receipt-default.jpg`,
`queue-restart-held-narrow.jpg`, and `queue-restart-held-default.jpg` in the same
evidence directory. `preview-stopped.json` confirms preview PID **63691** was
stopped; the viewport was reset. No gateway, provider or model was used, and no
paid model, operator service, device or deployment was involved.

Coverage is deliberately bounded. Exact active-Steer receipt/remount behavior and
late running proof after canonical idle with later-row/draft preservation are
covered in separate deterministic tests, not a combined matrix. Browser evidence
qualifies only current-view receipt draft preservation; it does not claim rich
draft persistence across view remount/recreation. No multi-device, cross-tab or
simultaneous-client matrix was requested or run. The bounded review is complete.

### Still open — bounded qualification is not all of CUI-7

General safe Retry and authoritative recovery of an attempt after same-device
restart remain open. A recreated client has no trusted original in-memory owner;
the persisted submitted row therefore stays UNKNOWN and held rather than retrying.
Canonical deduplication/readback through the existing source owners and a trusted
way to distinguish a lookup miss from an accepted attempt have not been added.
Legacy rows with absent or malformed provenance cannot prove they were never sent;
they remain held and some genuinely unsent legacy rows may be stranded.

General Retry still requires trusted committed-attempt binding, atomic same-attempt
deduplication/join, authorized non-executing readback through existing source/thread
owners, and content-free outstanding-attempt/held metadata preservation. Lookup miss,
source replacement or unavailability remains UNKNOWN, never permission to replay.
No new prompt/media persistence decision was made. Physical Android/native-client
and macOS Full/process-tree qualification also remain open. This bounded package is
not complete CUI-7, A3, P6, release or merge acceptance. At this 2026-11-16
checkpoint CUI-6 reasoning access was the next independent UI item, followed by
CUI-2–5; the CUI-6 qualification is recorded above and no later work is claimed
by this historical section.

## Flat queue and Steer presentation follow-up — 2026-11-16

**PASS for presentation/copy only.** The screenshot-based follow-up replaces the
floating/shadowed queue with a solid, subtly bordered strip touching the input
card. It remains outside the card and inset 12 px on each side. There is no panel
shadow, backdrop blur or visible heading/count. Previews use one-line ellipsis;
the complete text remains in the accessible DOM/title and existing Edit action.
The English action now reads **Steer**. Existing translation keys, callbacks,
admission handling, pending locks, queue state and shared animations are unchanged.
This supersedes the earlier gap, heading/count and multiline presentation below,
not its qualified delivery behavior or the remaining CUI-7 safety limits.

Both thread/hero structural regressions failed before the change. The focused
composer/actual-client/uncertainty checks then passed **144 tests**; final full
WebUI checks passed **1,606 tests / 102 files**, TypeScript/Vite build and ESLint.
Build retained the existing large-chunk advisory. Python/strict checks and an
installed wheel were not rerun for this frontend presentation-only change.

Actual component/global-style previews passed desktop, 390 × 844 and light-hero
checks. Computed panel shadow/backdrop filter were both `none`; no horizontal
overflow or row overlap occurred. Pending Steer kept its confirmation status and
disabled mutation controls while preserving a separately typed draft. Seven rows
remained bounded to a 216 px scrolling list and the last row was reachable.
This is responsive component evidence, not backend/native/physical-phone acceptance.
The temporary viewport was reset and disposable preview server stopped.

Local evidence: `/private/tmp/nb-queue-steer-flat-20261116-zoldy528/` — incremental
diff, red/final-suite/build/lint logs, browser observations and `queue-flat-*.jpg`.
All work remains local/uncommitted; no commit, push, staging merge or deployment.

## Queue placement follow-up — 2026-11-16

**Historical presentation:** the flat queue follow-up above supersedes its spacing,
heading/count and preview styling; placement and behavioral protections remain.

**PASS for layout only.** Following the user's ChatGPT/Codex screenshot, the existing
queue is now a separate sibling panel above the message-input card, not inside its
background/container. It is inset 12 px on each side with an 8 px gap, at both thread
and hero widths. No new component state, queue protocol or recovery mechanism was
added. Existing submission handlers, admission callbacks, pending locks, row actions,
scroll bounds and draft handling are unchanged.

Two structural regressions failed before the move; **129 composer/actual-client
queue tests** then passed, as did TypeScript/Vite build and lint. A real browser
rendered the current component/app styles at desktop and 390 × 844, plus the light
hero variant: no horizontal overflow or row overlap; six rows stayed in a 216 px
scrollable list, its last row was reachable, and a pending row did not replace the
independent later draft. This was an isolated UI fixture, not an installed app,
backend-delivery or physical-phone qualification. The full suite was not rerun for
this presentation-only follow-up. The CUI-7 review hold below remains unchanged.

Local evidence: `/private/tmp/nb-queue-external-20261116-39gmt472/` —
`acceptance.json`, before/source diff, red/green/build/lint logs, browser observations
and `queue-desktop.jpg` / `queue-narrow.jpg`. The preview server was stopped and the
temporary viewport reset. Changes remain local/uncommitted; no deployment occurred.

### Additional intake priority — 2026-11-16

The new mobile false-failure report (CUI-7) is investigated before CUI-2 because a
misclassified delivery outcome makes retrying unsafe. Check the shared client,
native mobile socket bridge, reconnect/auth/lifecycle handling and owner admission
reconciliation together before changing retry behavior. A lost acknowledgement is
not proof of rejection, and a restart/history refresh is not permission to send
another attempt. Keep the frozen A3 once-only/source-binding requirements intact.
Start with discriminating failure cases rather than a broad transport rewrite.

CUI-6 reasoning expansion remains an independent UI issue. CUI-2 through CUI-5
remain open; neither new report is covered by the earlier CUI-1 qualification.

## CUI-7 cumulative review checkpoint

**Historical checkpoint:** the two findings below are corrected by the latest
[bounded owner-seam and restored-queue qualification](#cui-7-bounded-owner-seam-and-restored-queue-qualification--2026-11-16).
The retained evidence and broader durable/device limits are not erased.

**HELD / requires correction, 2026-11-16.** The user requested a cumulative
simplicity review before further patches. CUI-7 owner-seam work is frozen, local and
uncommitted on `codex/staging-audit-implementation`, based on
`de2e0fec517203d0f2c45fa83ecf7c591fa2ab4b`. No staging merge, publication,
deployment or complete CUI-7/A3 acceptance was performed.

### Work retained and verified scope

- Native recovery/renewal no longer silently reopens an adapter behind the shared
  client. The shared client remains the only reconnect owner. Native startup events
  are bounded and ordered; retired-generation cleanup precedes replacement opens.
- A sent message that loses confirmation is **Awaiting confirmation**, not Failed.
  Partial assistant activity is retained, and transport loss does not resend input.
  Route/display echoes and unkeyed legacy lifecycle events cannot resolve UNKNOWN.
  Its explanation is tap/keyboard accessible through a local coarse-pointer Popover;
  this is regression-test evidence, not physical-phone qualification.
- The earlier Mac cancellation and active-Steer fixes are preserved. The cumulative
  review found no reason to remove those root-cause corrections or add a new
  framework. Composer snapshot/click/presentation guards serve distinct purposes;
  test volume alone is not architectural complexity.

The independent review covered the cumulative HEAD-to-local diff: **43 unique
paths**, including 10 production files, 10 locale files, 20 test files and three
existing/new progress records. All 2,779 snapshot files matched the checkout.
That frozen review snapshot independently passed **536 focused tests / 9 files**,
**1,579 full WebUI tests / 102 files**, TypeScript/Vite build and ESLint. The Python
and accepted queue inputs were unchanged from their earlier frozen qualification;
this client-only stage did not rerun the full Python suite or change the checkout's
216-distribution dependency environment. Passing suites do not negate the real
reconnect failure below. The final corrected source was not installed/qualified
as a new wheel or on a phone.

### Two concrete findings, not speculative redesign

1. **UNKNOWN admission currently vetoes valid history and activity projection.**
   In an installed macOS browser with a scripted loopback provider, exactly one
   message and one primary model call produced a canonical user input and assistant
   reply. The reply committed before the client actually reattached. The thread
   response was idle and contained both rows, while the browser said Connected but
   continued showing Awaiting confirmation, Working and a disabled project picker,
   with no reply. `NanobotClient.canReconcileCanonicalCompletion` refuses UNKNOWN;
   `ThreadShell` uses that refusal to reject the entire refreshed transcript and
   never reaches its completion projection. This regression was introduced by the
   CUI-7 safety change and remains after uncertainty correction1. It is not evidence
   that native reconnect failed: the replacement attach was positively recorded.
2. **Fresh-turn admission receipt precedes canonical input commit.**
   `SessionTurnStarted` currently produces `message_admitted` before the user append.
   The producer ordering predates these local changes, but forwarding the receipt
   now exposes it to queued prompts starting a new turn: a row may be removed before
   its mandatory input append succeeds. Active Steer is different: its source append
   precedes `InjectionCommitted`, so keep that correct path. Lifecycle start alone
   must not claim committed input.

### Smallest coherent correction direction

Keep three facts distinct in the **existing owners**: socket availability, exact
attempt disposition, and authorized history/current-activity projection. Permit
appropriately fenced display hydration without clearing UNKNOWN or authorizing
resend. Preserve source/lineage, request/run-generation and UI-revision checks,
uncertain attempts and later drafts. Do not simply call `finishRunLocally`, clear
busy on every disconnect, trust a null Stop target or trigger automatic queue drain.
Fresh-turn admission must originate after successful canonical input append, not
from `SessionTurnStarted`; retain the correctly ordered Steer producer.

The next bounded design/test contract must prove completion both before and after
reconnect, stale/newer-run/source-replacement fences, one input/no replay, preserved
rich later drafts, and **failed append => no receipt/no queued-row removal**.
No admission/projection correction was dispatched after this review. Settle this owner seam
before resuming feature work; do not layer additional local bypasses around it.

General message Retry is **not implemented**. Before enabling it, reconcile the
original attempt and deduplicate through the existing canonical source/read owners.
The earlier analysis identified missing durable attempt binding/readback and
restart-persistent pending metadata (including queue held/attempt metadata). Those
are remaining guarantees, not permission to add a second execution ledger, persist
new prompt copies or build another reconnect framework. A lookup miss or changed
source remains unknown, not permission to replay. Persisting direct prompt/media
for one-tap recovery after app death would be a separate privacy/product decision;
it is not required for the current conservative no-retry behavior.

### Evidence and remaining scope

Raw local evidence: `/private/tmp/nb-cui7-20261116-4kzzm322/` —
`cumulative-review-verdict.json`, cumulative manifest/diff/inventory,
`r1-*-review.json` command receipts, correction red/green evidence, and
`browser-recovery/{quick-release-result,api-thread-after-real-reconnect,observations}.json`.
The diagnostic installed R0 wheel was
`acb417989d2910f7420e78d468f51eb253397ef708c7ba56bd2374a41fdd7eb0`;
it is not a qualification artifact for correction1. The fixture used only
`custom/qualification-offline`, never an external/paid model. Temporary browser
viewport overrides were reset and all disposable gateway/provider/proxy processes
exited cleanly; operator services and credentials were untouched.

No authorized Android device was connected on this host. Physical remote-mobile
and native installed-client acceptance remain open, as do durable same-attempt
Retry and macOS Full/process-tree capability. This reconnect regression can disable
the project picker, but does not establish the cause of every CUI-3 report.
CUI-6 reasoning and CUI-2 through CUI-5 are untouched. Resume only after this review's
coherent correction contract is settled; preserve the qualified active-Steer work.

## Per-issue completion gate

1. Reproduce the reported behavior independently; distinguish confirmed cause from
   a hypothesis and record the relevant source/commands/environment.
2. Settle observable behavior and any material product/authority decision before
   implementing. Use a clear, bounded Luna-Max worker contract where appropriate;
   escalate analysis only for a demonstrated ambiguity or risk.
3. Add the closest deterministic regression, implement minimally, and run affected
   tests plus relevant build/lint/types. Do not weaken admission, recovery or safety
   assertions to make UI checks pass.
4. Visually verify UI changes at desktop and narrow/mobile layouts where relevant.
   Unit mocks alone do not prove OS notification or physical-device delivery.
5. Record exact results and remaining limits, then close the issue before moving to
   the next one. Preserve unrelated work and keep implementation separate from staging.

## CUI-1 queue and Steer qualification

**Result: PASS for this bounded client correction, 2026-11-16.** Changes remain
local and uncommitted on `codex/staging-audit-implementation`; no push, staging
merge, operator-gateway replacement or full A3 acceptance was performed.

### Confirmed causes and correction

- Adding guidance already clears the textarea into its queued snapshot. The
  lingering duplicate was the **queue row**, not an uncleared input field.
  `NanobotClient` recorded `message_admitted` but returned without notifying chat
  listeners, so the hook could not resolve the row's admission promise. The
  backend already emitted the correct injection receipt; no backend change was
  needed. Non-system owner receipts now reach the existing chat subscriber after
  canonical-turn bookkeeping.
- A row remains visibly **Awaiting confirmation** until its owner receipt arrives.
  A synchronous guard prevents repeated Guide/Steer dispatch; pending edit,
  delete and drag controls cannot mutate that snapshot. Admission removes only
  the matching row, not a separately typed later draft.
- Unknown outcomes remain held, including local failure or an unresolved/negative
  admission promise. Only the exact authoritative `not_admitted` rejection enables
  explicit recovery; an idle transition does not retry it automatically. Every
  unresolved attempt retains its correlation, including more than five pending
  sends; the bounded legacy-history cache no longer evicts an unresolved row.
- The queue has a heading/count, natural non-shrinking multiline rows and a bounded
  scrolling list. This fixes both the original single-row clipping and multi-row
  overlap on narrow screens. Shared animation styles were not changed; CUI-5 is
  still a separate task.

### Acceptance evidence

Implementation used a bounded **Luna Max** worker with read-only **Sol Max** analysis
and fresh integrated review. Initial socket-delivery and duplicate-click tests
failed before the fix; the six-pending rejection regression also failed before its
correction. One correction round resolved the responsive overlap and correlation
limit findings. The fresh narrow recheck then passed.

Independently run on the final frozen source:

| Check | Result |
| --- | --- |
| Socket/client, composer, stream hook, shell and actual-client integration tests | **440 passed / 5 files** |
| Full WebUI tests | **1,541 passed / 100 files** |
| WebUI TypeScript/Vite build and ESLint | **Passed** |
| Fresh-required wheel build, source/bundle parity and isolated install | **Passed** |
| Installed gateway with actual browser on macOS, desktop and 390 × 844 viewport | **Passed** |

The installed browser exercised six multiline rows, pending status, scrolling to
the final row, and a separate later draft. Narrow rows measured 53.9 px around
49.9 px previews; the pending row expanded to 67.9 px. The list stayed bounded at
216 px with scrollable content, without row overlap. An actual owner receipt with
`starts_turn: false` and the original active root removed only the admitted row
(**6 → 5**). The later draft was unchanged; exactly one steer was sent and its
text appeared once in the next provider request. Temporary viewport overrides
were reset and all disposable provider/gateway/proxy processes exited cleanly.

The test model was `custom/qualification-offline`, a scripted loopback provider;
no external or paid model was used. The checkout's **216-distribution** dependency
environment and all pre-CUI Python, TUI, mobile and accepted Mac corrections were
unchanged. This client-only package did not rerun the preceding full Python suite.

Final wheel SHA-256:
`b2b30b5c8a71ae6cffb1902cc29138937d003e7072820ff8e341ec39ef5e4887`.
Local evidence on this Mac:
`/private/tmp/nb-cui1-20261116-_z8n5v2r/` — `acceptance.json`, final source/dependency
manifests, red/green logs, independent `*-final.json` command receipts,
`browser-final-observations.json`, recorded owner frames and `queue-final-*.jpg`.
The earlier recorder allowlist omitted owner receipts; its corrected second
baseline positively captured them. The omission was not a backend failure.

### Limits and next work

Reduced-motion compatibility was checked through unchanged CSS/source equivalence,
not an actual OS preference toggle. The real browser checked a later text draft;
existing isolated attachment/quote/session-mention snapshot tests remain in place,
but a rich later draft was not exercised end-to-end. Unknown-outcome retry is not
newly supported. Native macOS Full/process-tree qualification remains a separate
capability gap in the Mac record.

The original next item was CUI-2, independently reproducing context-window
accounting after compaction. The additional intake above prioritizes CUI-7 mobile
delivery safety first, then CUI-6 reasoning access. CUI-2 through CUI-5 have not
been started or closed by this queue fix.

## CUI-7 restored queue anti-replay — final qualification

**Bounded implementation PASS.** The parent completed full WebUI, build, lint,
external baseline and focused owner/storage checks, plus narrow integrated review.
The authoritative counts and browser scope are recorded in the
[qualification above](#restored-queue-anti-replay-validation--final-bounded-evidence).
This does not close general safe Retry, same-device restart recovery, physical-device
qualification or A3.

### Current behavior

- Persisted rows carry versioned, content-free submission provenance. New rows start
  `ready`; `submitted` rows retain the original `{ queueKey, promptId, turnId }`
  binding; exact refusal becomes manual-only `not_admitted`; missing, malformed or
  unsupported provenance is quarantined as `unknown`. Unknown and submitted content
  remains read-only and cannot be replayed from storage.
- The composer synchronously stores the exact binding before calling the existing
  send owner. A failed, truncated, over-cap or unserializable write leaves the row
  unchanged and blocks `onSend`, upload and message dispatch. The existing 20-row and
  4,000-character limits are preserved; a full or already-over-cap queue is not
  trimmed to make a protected attempt fit.
- The existing `NanobotClient.pendingMessageSends` entry retains a typed queued
  outcome until the corresponding row is durably removed or updated. An exact
  `message_admitted` (or the existing exact running proof) removes only that row.
  Only a correlated `detail=not_admitted` with a reason unlocks manual recovery.
  Generic errors, route acknowledgements, idle/history and a client recreated by
  restarting the same app/device do not resolve the hold.
- A mounted composer can join and consume the existing client observation after a
  remount or while its queue was offscreen. A same-device recreated client has no
  trusted owner record and therefore leaves the restored row held. Temporary queues
  remain connection-owned and are not written to localStorage.

### Historical focused evidence and current limits

The initial RED is retained at
`/private/tmp/nb-cui7-restored-20261116-f4f8t2qi/pre-edit-red.log`. The focused
green snapshot (6 files / 156 tests) is recorded separately at
`/private/tmp/nb-cui7-restored-20261116-f4f8t2qi/focused-green-final.log` after the
restored actual-client A→B→A path, exact proof after remount, proof while offscreen,
exact refusal while offscreen, legacy/malformed quarantine, storage quota and
serialization denial, and 21-row capacity preservation regressions were added.
These are an earlier focused snapshot; the final full-suite, external baseline,
build, lint and 16 owner/storage results are summarized in the current qualification
above. Earlier correction-round failures and focused runs are preserved in the same
QA directory as historical evidence.

There is no durable same-attempt deduplication or lookup after client recreation or
process restart. Those rows remain held indefinitely until the original client owner
can provide a matching result; this package intentionally adds no persisted payload
store, backend authority, protocol field, retry framework or ledger. General Retry,
native/mobile device qualification and safe recovery after restart remain open. This
bounded qualification targets single-device use; no multiple-device,
simultaneous-client or cross-tab coordination matrix is included.
