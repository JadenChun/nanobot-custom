# Client UX follow-ups — 2026-11-16

**Git checkpoint, 2026-11-16:** the completed fixes and records are now committed
locally with author/committer `Jaden <changherng.jc@gmail.com>`; historical
local/uncommitted statements below refer to their original qualification time.
See [the current handoff checkpoint](implementation-handoff.md#git-checkpoint-completed-fixes-committed--2026-11-16).
No push, staging merge or deployment occurred. Open issues remain open.

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
| 2 / CUI-2 | Context-window data becomes missing or inconsistent after compaction. Retain truthful current-model/window and post-compaction context accounting; never invent missing measurements. | Not started; after CUI-7 and CUI-6 |
| 3 / CUI-3 | Project/folder selection is inconsistently disabled in an existing/running session. Distinguish intentional active-turn binding from stale busy state; changing a draft/next-turn selection must not mutate the running turn's workspace authority. | Not started; specific post-Stop stale busy state fixed separately |
| 4 / CUI-4 | Desktop and mobile notifications are unreliable and lack details such as the app icon. Verify permission/subscription/lifecycle, payload, icon and navigation independently on each surface. | Not started; actual device/OS delivery needs suitable platform evidence |
| 5 / CUI-5 | Replace the bottom-to-top viewport motion with an Apple-like progressive word reveal. Preserve markdown/code/math, streaming, selection, accessibility/reduced motion and long-history performance. | Not started |
| 6 / CUI-6 | Reasoning is limited to a one-line preview with no usable way to inspect the full provider-supplied reasoning output. Provide accessible expand/collapse and full readable content where the model actually supplies it; do not fabricate unavailable reasoning. | Not started; separate UI investigation after CUI-7 |
| 7 / CUI-7 | Mobile reports failed send even though restart reveals the message was sent; remote connection instability appears to disrupt features. Distinguish transport uncertainty from authoritative refusal, reconcile the same attempt, and provide safe explicit retry without duplicate admissions. | **Bounded owner-seam correction qualified on macOS; CUI-7 remains open** for restored attempts, safe Retry and physical-device qualification; [latest record below](#cui-7-owner-seam-qualification--2026-11-16) |

## CUI-7 owner-seam qualification — 2026-11-16

**PASS for the two concrete owner-seam defects and their existing UI consumers.**
This supersedes those findings in the historical cumulative review hold below,
not the remaining CUI-7 recovery work. All changes remain local/uncommitted on
`codex/staging-audit-implementation` at
`de2e0fec517203d0f2c45fa83ecf7c591fa2ab4b`; staging was not changed.

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
  admission promise or replay its input. Its original queue row stays held; later
  rows and drafts survive. A matching late receipt removes only that row without
  reviving old ownership or automatically draining the next item.
- The existing automatic-send guard consumes the idle edge even when blocked.
  Independent explicit new input remains a different attempt, not Retry. Folder,
  recovery and persistent-chat access-mode mutations stay disabled while UNKNOWN;
  supported known-active permission/revocation controls remain usable.
- Accepted idle notifies existing activity subscribers even if the running clock
  frame was lost, so the sidebar spinner converges with the main conversation.
  The external queue panel is preserved. No second ledger, reconnect owner, source
  schema, prompt inbox or general recovery framework was introduced.

### Final verification

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

### Still open — do not call CUI-7 complete

Switching away/remounting a queue, or reloading/restarting while awaiting proof,
clears existing queue projection/waiter references. Restored stored rows currently
lack held/attempt metadata, so manual resend of that original row can become
possible. The live UNKNOWN map blocks automatic drain while retained, but does not
qualify restored-row safety or app-death recovery. This pre-existing gap was not
solved by adding another store in this tranche.

General Retry still requires trusted committed-attempt binding, atomic same-attempt
deduplication/join, authorized non-executing readback through existing source/thread
owners, and content-free outstanding-attempt/held metadata preservation. Lookup miss,
source replacement or unavailability remains UNKNOWN, never permission to replay.
No new prompt/media persistence decision was made. Physical Android/native-client
and macOS Full/process-tree qualification also remain open. This is not CUI-7, A3,
P6, release or merge acceptance. CUI-6 reasoning access and CUI-2–5 remain independent
unstarted UI work; no implementation of them is claimed here.

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
[owner-seam qualification](#cui-7-owner-seam-qualification--2026-11-16). The retained
evidence and broader durable/device limits are not erased.

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
