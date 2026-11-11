# A3 review and merge gates

**Documentation package: structural validation PASS. Architecture/implementation merge: HOLD.**
Two independent reviews judge the small-core, edge-owned direction viable, but require
corrections before the candidate can meet its own authority, canonical-history and
durability contracts. This is not an all-architecture PASS. The nine findings below
apply to `0518de5f5013b77cbc20d2c5b456e973f33c68f5`; they remain separate from the
frozen 136-finding partition. Several unsafe paths predate A3 or reflect partial repair,
so these findings are not all classified as new regressions.

## Evidence inspected and its limits

| Evidence | State | Recorded result and limit |
| --- | --- | --- |
| Candidate source bodies | `SOURCE_OBSERVED` | Named owners, formats and source excerpts are linked in [design](design.md) and below. Presence of a mechanism does not attest all callers. |
| [Hosted run 42972763133](https://github.com/JadenChun/nanobot-custom/actions/runs/42972763133) at `0518de5f` | `HOSTED_LOG_INSPECTED` | Overall **cancelled**. Linux Python 3.11/3.14 + coverage, WebUI bundle/WebUI, TUI Linux/Windows, Docker and dedicated Windows process compatibility succeeded. Full Windows Python 3.14 was cancelled. |
| Same run, Linux Python 3.14 logs | `HOSTED_LOG_INSPECTED` | Ruff: `All checks passed!`; strict comparator: `PASS_NO_NEW_DIAGNOSTICS`; 17,540 passed / 14 skipped / 22 warnings in 532.32 s, plus 129 serial CLI passed. Comparator success alone is not zero absolute strict errors. |
| Same run, Linux Python 3.11 logs | `HOSTED_LOG_INSPECTED` | 17,540 passed / 14 skipped / 7 warnings in 445.32 s, plus 129 serial CLI passed. These are matching-head suite results, not per-finding closure. |
| [PR #27](https://github.com/JadenChun/nanobot-custom/pull/27) stage and qualification claims | `REPORTED` | Ten stages `READY_FOR_REVIEW`, clean strict coverage of 541 files, installed rollback and Android 38 instrumentation are reported. Exact zero-error report, installed historical-reader receipts and Android artifacts were not independently inspected for this record. |
| Physical Android waiver in PR | `REPORTED` | Physical qualification is waived in the PR. The human provenance and intended scope of that waiver are unknown here; emulator evidence does not establish AC7.4. |
| Independent authority review, isolated macOS host / Python 3.13.2 | `REPRODUCED` and `SOURCE_OBSERVED` | 57 focused tests passed and one owner-offer test failed. Initial failure hit the Unix socket path limit; a short temporary base reached unavailable peer credentials on macOS. Boundary probes establish AR1–AR4/AR6; AR5 is a source-confirmed native race, not an Android runtime exercise. |
| Independent recovery review, isolated macOS host / Python 3.13.2 | `REPRODUCED` | 121 focused tests passed, one Linux-supervisor-only test skipped, in 27.05 s. Temporary probes establish RR1–RR3. This is not native Linux/Windows qualification or a historical-executable rehearsal. |
| Coordinator repeat probes | `REPRODUCED` | Independently repeated AR1 policy-only allowance reuse and RR1 malformed-JSON completeness. No shell command was executed by the AR1 probe. |

The pinned [CI workflow](https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/.github/workflows/ci.yml)
still includes the full Windows matrix without `continue-on-error`. A cancelled
configured matrix job cannot be relabelled all-green. Later fixes require fresh source/artifact
pins and affected requalification; results from the CI-exercise migration branch do not
qualify the later candidate fixes.

## Independent authority findings

### AR1: an approved effect can be reused

**High severity, high confidence; conditional on a risky effect being approved.**
[`RiskyActionPolicy.evaluate`][risk-evaluate] checks a retained set of approved fingerprints
without consuming a particular occurrence. A fixture first received action `respond`
with approval-required metadata; after supplying that approval's effect IDs, the same
call duplicated in a batch and repeated in a later iteration both returned `allow`.
The coordinator independently reproduced the policy-only result. No command ran in
that repeat probe.

**Consequence:** one human consent can authorize more executions than the selected
effect. **Narrow correction:** consume an allowance atomically at actual dispatch for
one exact occurrence, including effective cwd/target and current authority; do not
discard the useful typed approval ID and pending-map design. **Owner/stage:** canonical
approval/policy/tool dispatch, A3-4 (with A3-1 target facts). **Acceptance:** approve one
effect, race duplicate calls and later iterations, and observe exactly one effect;
denied, stale, expired, revoked and changed-target controls start none. Optional echoed
identity fields being absent alone are not an established bypass.

### AR2: worker children bypass risky-action consent

**High severity, high confidence; conditional on a write-capable delegated worker.**
The candidate [`ForegroundAgentManager._run_role`][child-runner] installs a read-only policy
for read-only roles and no corresponding risky-action policy for a worker.
`_worker_tools` gives scoped write/edit capability; `run_worker` reaches that runner.
An isolated 300-line file replacement required approval under the normal risky policy,
but the worker with that exact file in `write_scope` overwrote it and completed without
approval.

**Consequence:** location scope is mistaken for consent to a risky effect.
**Narrow correction:** attach the existing risky-action policy at child dispatch and
bind its approval to the child's identity and exact effect while retaining narrowed
tools, scopes and file locks. **Owner/stage:** delegation and canonical approval,
A3-5/A3-4. **Acceptance:** the same risky replacement prompts through the real child
path, no file changes before approval, one approved occurrence applies once, and parent
revocation or stale child generation prevents it.

### AR3: child metadata is disconnected from canonical execution history

**High severity, high confidence; conditional on delegated execution.**
The [`child runner construction`][child-source] takes a child stream from the parent's
source writer. Background construction drops execution/source/artifact authority
without acquiring a replacement child owner. Finalization then records synthetic
assignment/final messages through [`record_exchange`][child-exchange]. In the isolated
worker write probe, child history contained only user/assistant rows, with no tool
intents or results; cancellation can skip that exchange entirely.

**Consequence:** a durable child identity does not supply the promised independently
owned source/effect ledger, undermining restart, cancellation and audit proof.
**Narrow correction:** acquire the existing canonical child execution and source owner
before child work; retain it through required recording and settlement. Parent activity
is a projection of the child reference, not a second history authority. **Owner/stage:**
session/execution/source owners plus delegation, A3-3/A3-5. **Acceptance:** real child
tool intent, result and terminal/uncertain state are durable under the child incarnation
on success, failure, cancellation and restart; no parent tool stream substitutes for
them, and no unrecorded effect is acknowledged.

### AR4: MCP reload can change the binding after definitions were captured

**High severity, high confidence; conditional on reload overlapping a model request.**
The runner captures tool definitions before dispatch, then later resolves a call from
the live registry. [`MCPProvider.reload`][mcp-reload] replaces wrappers; executing-call
lease accounting starts too late. A real `AgentRunner` plus `MCPProvider.reload` fixture
sent `fixture-old` definitions to the model and executed `fixture-new`, then completed.

**Consequence:** model-visible schema/identity and actual execution can differ.
**Narrow correction:** choose safe-boundary reload deferral or a request-owned original
binding/lease from definition capture through settlement; keep revocation live.
**Owner/stage:** runner request/registry and MCP connection owner, A3-6/D5.
**Acceptance:** overlap definition capture, replacement and dispatch: the call uses its
original authorized binding or is refused after revocation, never the replacement by
name. Retire the old resource once after its last request/call user settles; failed
replacement preserves a usable old connection.

### AR5: pairing Cancel does not cancel native pending authority

**High severity, high source confidence; conditional on Cancel racing in-flight native
poll/proof. Android runtime reproduction remains unverified.** The
[`bootstrap Cancel handler`][mobile-cancel] clears JS state/timers, while
[`NativeNetworkPlugin` polling][mobile-poll] can prove and save credentials before the
late JS promise result is ignored. Native custody can therefore survive restart after
the user cancels the visible attempt.

**Consequence:** UI cancellation need not revoke the native pending generation.
**Narrow correction:** persist and cancel a native attempt generation, compare it before
collect/sign/save/publish, and revoke a credential issued concurrently with cancellation.
**Owner/stage:** native custody/bootstrap plus remote pairing, A3-7.
**Acceptance:** physical or appropriately scoped native instrumentation races Cancel
with approval, proof and token delivery, restarts the app and owner, and observes no
usable cancelled authority; a deliberately new attempt can pair.

### AR6: production WebSocket attachment dispatch omits scope enforcement

**High severity, high confidence; conditional on a reduced-scope mobile credential.**
The production [`WebUICommandRouter` attachment path][mobile-attachment] revalidates
identity and commits attachment ingress before checking `attachment.write`; its
WebSocket runtime path bypasses the separately scoped edge handler. A real router with
`AccessIdentity(scopes=('chat.read',))` and fixture ingress committed once and emitted
`attachment_stored`.

**Consequence:** an authenticated read-only mobile identity can reach a forbidden
production mutation. **Narrow correction:** enforce the production command-to-scope
map after identity revalidation and before every effect; deny unmapped commands.
**Owner/stage:** shared production command router/mobile edge, A3-7.
**Acceptance:** reduced-scope credentials fail through the actual WebSocket runtime
without any ingress commit, writer or mutation; authorized scopes work, and production
route-inventory parity covers each command. Existing HTTP scope checks remain valuable.

## Independent recovery findings

### RR1: incomplete reference evidence can permit deletion

**P1, high confidence; conditional on explicit destructive maintenance.**
[`reference_evidence`][reference-evidence] skips malformed/non-object JSON without
marking evidence incomplete; [`prune_unreferenced`][attachment-prune] treats a missing
sessions root as an empty complete inventory. The generated-media
[`quota deletion path`][media-quota] bypasses its incomplete-evidence age gate.
Temporary probes returned completeness `True` for truncated canonical JSON, removed a
blob with a missing sessions root, and deleted a 10-byte generated asset with
`evidence_complete=False, max_bytes=1`. The coordinator independently repeated the
malformed-input result. These are inherited unsafe paths with a partial A3 repair.

**Consequence:** missing reference proof can become permission to delete retained data.
**Narrow correction:** require a complete root inventory and syntactically valid
canonical evidence before **every** deletion path, including quota; preserve all bytes
when incomplete. **Owner/stage:** attachment reference inventory/prune and generated
media maintenance, A3-8/R12. **Acceptance:** missing root, unreadable file, truncated
JSON, non-object record and unresolved reference each preserve blobs and sidecars under
both age and quota pressure. Existing OSError-only and quota-disabled cases are not
sufficient coverage.

### RR2: process completion can target a recreated session

**P1, high confidence; conditional on queued completion and session delete/recreate.**
[`ProcessRecord`][process-record] stores a key without incarnation, and
[`ProcessCompletionRuntime._continue`][process-continue] checks key existence before
admission. An isolated started/finished process with `OLD OUTPUT` was followed by
synchronous deletion/recreation of its session; after the queued task settled, the new
incarnation received one old continuation. Key-only binding is inherited; A3's async
persistence introduces additional scheduling boundaries around it.

**Consequence:** old work can be injected into a new conversation sharing an address.
**Narrow correction:** bind namespace/incarnation at capture and revalidate at canonical
admission, including queued/restart paths. Keep old unbound records inspect-only; state
the `/new` invalidation choice explicitly. `ProcessRecord` currently forbids extra
fields, so evolve its version/reader/export contract deliberately. **Owner/stage:**
completion producer and canonical session admission, A3-8/A3-3.
**Acceptance:** deletion/recreation and reset cannot receive an old continuation;
current matching identities continue once, unknown effects stay non-replayed, and old
records remain readable/exportable without gaining authority.

### RR3: attachment migration dry run writes artifacts

**P2, high confidence; conditional on migration preview with valid legacy media.**
[`_migrate_event`][attachment-migrate-event] commits ingress before
[`migrate_legacy_media`][attachment-migrate] reaches its `dry_run` transcript-write gate;
the CLI previews before confirmation. A temporary legacy user media event created a
canonical PNG and JSON sidecar under `dry_run=True` while leaving the transcript
unchanged. This is an inherited unresolved finding.

**Consequence:** preview/declining confirmation can still change persistent storage.
**Narrow correction:** build a read-only plan or use disposable scratch state and defer
canonical publication until commit. **Owner/stage:** attachment migration/CLI preview,
A3-8/R12. **Acceptance:** full filesystem snapshots before/after dry run and declined
confirmation match for source, blob, sidecar, staging and backup paths. A committed
migration remains atomic/readable; checking transcript bytes alone is insufficient.

## Qualification and decision gates

| Gate | Missing proof or choice | Required acceptance condition |
| --- | --- | --- |
| G1: platform matrix | Full Windows cancelled; no macOS native/source-writer/owner IPC qualification; Linux-specific paths remain | Name supported client/executor/mode combinations, qualify actual hosts and preserve refusal for unavailable proof. Do not equate generic process success with managed confinement. |
| G2: physical mobile and cutover | Physical waiver's human provenance unknown; approved-but-unissued challenge persistence and retained/revoked v1 installation re-pair path unqualified | Record an authorized waiver's exact scope or provide AC7.4 evidence. Prove restart and old-installation re-pair through real owner/edge/native custody; no v1 revival. |
| G3: historical executable and artifacts | Existing rollback tests use current readers; PR installed-prior-reader and Android claims uninspected | Inspect source/dependency/artifact/previous-executable digests and exact upgrade/rollback receipts, including post-cutover history, scheduler once-only and device revocation. |
| G4: consequential choices | Historical D1–D7 pending; implemented values do not establish human acceptance | Add durable acceptance or explicit deferral for supported scope, Full semantics, children/budgets, P-256/cutover, request lifetime, outcomes, retention and compatibility changes. |
| G5: complete traceability | 136 original IDs default unverified; E01 remains limited-depth | Attach criterion-specific proof or an explicit qualified disposition for each relevant finding and retain all remaining IDs/gaps. A full passing suite does not close the partition. |
| G6: repair branch | Six separately published fixes overlap but are not ancestor proof of inclusion | Record a semantic disposition per commit; port only a proven missing invariant through the current owner and rerun its closest check. |

The source review retained the independent hard deadline and shielded termination,
tree-quiescence distinction, durable intent/unknown restart, writer lease/source CAS/
incarnation, proof archives, checksummed scheduler export and fail-closed unrepresentable
rollback. It also retained typed pending approvals and server file targets, narrowed
read-only policy/scopes/locks, MCP readiness and executing-call leases/reconnect refusal,
strict P-256/SPKI/DER/canonical claims and current HTTP scope checks. Corrections should
complete these boundaries rather than replace them with another framework.

Native-child mid-run send/steer is explicitly unavailable in the candidate backend.
This and the mobile restart/re-pair source gaps are qualification/contracts to resolve,
not additional demonstrated runtime defects. The `7fbb00c2a` inode-reuse theory was not
confirmed: changed validated content plus extension proof is a reasonable source
mechanism, with no new finding inferred here.

## Repair-branch reconciliation

The six commits are now durable on the separately published repair branch ending at
[`d3380eb69d3c8416a9ca72522001234ce30a8eeb`](https://github.com/JadenChun/nanobot-custom/tree/d3380eb69d3c8416a9ca72522001234ce30a8eeb).
They require semantic reconciliation before code merge, rather than wholesale
cherry-picking. Absence from ancestry does not mean a correction is absent semantically.

| Repair commit | Current disposition | Evidence required before code merge |
| --- | --- | --- |
| [`2d9da620f`](https://github.com/JadenChun/nanobot-custom/commit/2d9da620fbe1eb89eaed0179087f4e9c28d3c850) | REVIEW_REQUIRED: overlap unverified | Compare its protected invariant and closest regression with current owners; record already covered, port narrowly, or superseded with proof. |
| [`184f22683`](https://github.com/JadenChun/nanobot-custom/commit/184f226839a64c91c536b90bd953026ca50f6712) | REVIEW_REQUIRED: overlap unverified | Same semantic comparison, including public/persisted compatibility and affected callers. |
| [`5d95a1069`](https://github.com/JadenChun/nanobot-custom/commit/5d95a10699829fbf4b45a2b9e04420da9c1efbb9) | REVIEW_REQUIRED: overlap unverified | Same semantic comparison; retain unrelated changes and current ownership. |
| [`d5576f1d2`](https://github.com/JadenChun/nanobot-custom/commit/d5576f1d266df4b70461453f87c95127e32d1129) | MISSING_PLATFORM_PATH_OBSERVED: reconcile Darwin owner peer authentication if macOS owner is supported | Candidate `_check_peer` uses only `SO_PEERCRED`; this repair adds Darwin `LOCAL_PEERCRED`. Port through the current instance/owner boundary and exercise actual peer authentication, or explicitly exclude that executor/owner scope. |
| [`b45519af2`](https://github.com/JadenChun/nanobot-custom/commit/b45519af243ce6bdbd634e025aac633d9b215472) | TEST_RECONCILIATION_REQUIRED: portability and optional-installer isolation | Compare test/environment isolation with the candidate, then run the affected platform checks if a change is needed. This is test-only work. |
| [`d3380eb69`](https://github.com/JadenChun/nanobot-custom/commit/d3380eb69d3c8416a9ca72522001234ce30a8eeb) | RETAIN_HISTORICAL_PROVENANCE: audit documentation only | Preserve the durable repair narrative and reconcile its claims with the current record; there is no runtime fix to port or product test to rerun from this commit. |

No disposition authorizes a live deployment. Preserve the separate local staging work
and review the candidate against the remote staging base named in the [version record](README.md).

## Review commands and evidence discipline

Independent reviews used the already available Python 3.13.2 environment, the frozen
implementation export, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, and isolated
temporary state. The documentation worker ran no product tests. For reproducibility the
review commands below use `$REVIEW_PYTHON` for that interpreter and `$REVIEW_TMP` for a
short isolated temporary directory; these are placeholders, not additional executions.

```sh
PYTHONDONTWRITEBYTECODE=1 "$REVIEW_PYTHON" -B -m pytest -q -p no:cacheprovider \
  tests/agent/test_migration_approval_admission.py tests/agent/test_mcp_binding_refusal.py \
  tests/agent/test_mcp_generation_retention.py tests/agent/test_delegation_child_sessions.py \
  tests/agent/test_delegation_inline_stop.py tests/agent/test_delegation_recursive_background.py \
  tests/remote/test_device_proof.py tests/remote/test_device_recovery.py \
  tests/remote/test_pairing_persistence.py tests/remote/test_a3_7_bearer_cutover.py \
  --basetemp="$REVIEW_TMP/authority"

PYTHONDONTWRITEBYTECODE=1 "$REVIEW_PYTHON" -m pytest -p no:cacheprovider -q \
  tests/tools/test_process_completion.py tests/tools/test_exec_session_tools.py \
  tests/tools/test_exec_session_deadline_tree.py tests/session/test_runtime_session_metadata.py \
  tests/session/test_mode_journal_rollover.py tests/session/test_legacy_mode_journal_compatibility.py \
  tests/cron/test_scheduler.py tests/cron/test_scheduler_store.py \
  tests/attachments/test_attachment_lifecycle.py --basetemp="$REVIEW_TMP/recovery"
```

The authority probe used existing MCP fixtures during the fixture model's first call:
capture old definitions, change configuration, await reload and return the same tool
name. Worker and reduced-scope probes used the real runner/router with fixture external
effects. Recovery probes used copied or newly created stores and compared identities
or filesystem bytes. AR5 remains source analysis rather than physical execution.

The documentation uses a deterministic before/after structural check because it changes
no executable behavior. Before editing, the archive and canonical entry were absent and
the frozen export matched all source Git blobs; no destination conflicts existed.
The post-edit checks below passed with Python standard-library parsing/hashing and Git
object/diff inspection. No application, type-check, build or dependency command ran as
part of this documentation work.

| Documentation check | Result at this record |
| --- | --- |
| Archive import | 138/138 paths match the frozen export and `a14aef` Git blobs, byte length and SHA-256; no missing or changed archive path. |
| JSON parsing | All 54 documents parse: 52 archival JSON files plus the two new ledgers. |
| Traceability | 136/136 unique original IDs, exactly one I05 package and primary stage each; 22 packages, A3-0 plus ten stages; all finding acceptance remains `UNVERIFIED`. Original qualifications and E01/overlay exclusions match I05. |
| Pinned sources | 57 bounded file/symbol excerpts across 32 source files match `0518de5f` Git blobs; curated GitHub source paths resolve in the object database. |
| Navigation and local anchors | Zero curated failures. Historical links checked separately: zero unresolved local links after excluding fenced/inline code; archival bytes remain untouched. This does not attest remote website availability. |
| Scope | 147 changed paths: 138 archive files, seven canonical files and two navigation files; allowlist passes against the pinned staging base. |
| Curated whitespace | The seven canonical files and two navigation files pass whitespace checks. The initial `git diff --check` exited 0 for tracked unstaged changes; it omitted the archive files while they were untracked. |
| Archive whitespace exception | The full staged import check exited 2 with the five inherited diagnostics below. After the archive commit, `git diff 29612c3df01dc3dce3501fe349bcf555cd6c58b6 HEAD --check -- docs/audits/` reproduces them. All archive blobs still match the frozen source, so immutable history is preserved with an explicit whitespace exception. |

| Preserved archive path | Line | Inherited diagnostic |
| --- | ---: | --- |
| [chatgpt-web-handoff.md](../../audits/module-review-2026-11-06/chatgpt-web-handoff.md) | 144 | New blank line at EOF |
| [handoffs/A01.md](../../audits/module-review-2026-11-06/handoffs/A01.md) | 713 | New blank line at EOF |
| [handoffs/D01.md](../../audits/module-review-2026-11-06/handoffs/D01.md) | 222 | New blank line at EOF |
| [handoffs/T01.md](../../audits/module-review-2026-11-06/handoffs/T01.md) | 751 | Trailing whitespace |
| [handoffs/U01.md](../../audits/module-review-2026-11-06/handoffs/U01.md) | 819 | New blank line at EOF |

The [archive manifest](audit-manifest.json) retains per-file digests, and
[evidence.json](evidence.json) retains assignments, source excerpt bounds and validation
counts. Structural PASS includes the named immutable-archive whitespace exception;
it does not claim a clean full-package whitespace check. These checks establish a
reviewable documentation package; they cannot clear
any implementation merge gate above.

<!-- Pinned source references -->

[attachment-migrate]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/attachments/lifecycle.py#L213-L245
[attachment-migrate-event]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/attachments/lifecycle.py#L133-L176
[attachment-prune]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/attachments/lifecycle.py#L78-L121
[child-exchange]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/agent/child_sessions.py#L462-L505
[child-runner]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/agent/delegation.py#L755-L782
[child-source]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/agent/delegation.py#L709-L749
[mcp-reload]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/agent/tools/mcp.py#L1984-L2024
[media-quota]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/webui/media_maintenance.py#L132-L155
[mobile-attachment]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/webui/inbound_commands.py#L556-L621
[mobile-cancel]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/mobile/mobile-bootstrap.js#L712-L720
[mobile-poll]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/mobile/android/app/src/main/java/ai/nanobot/mobile/network/NativeNetworkPlugin.kt#L495-L550
[process-continue]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/process_completion.py#L278-L320
[process-record]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/process_completion.py#L32-L50
[reference-evidence]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/attachments/lifecycle.py#L45-L70
[risk-evaluate]: https://github.com/JadenChun/nanobot-custom/blob/0518de5f5013b77cbc20d2c5b456e973f33c68f5/nanobot/agent/policy.py#L218-L281
