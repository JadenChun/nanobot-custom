# P5 Stage 3 guarded offline implementation acceptance checkpoint

Status as of 2026-10-01: **ROOT PASS for guarded offline Stage 3 implementation**.
Root accepted the actual joined diff after code review, required validation gates,
scope/hash audit and independent narrow PASS. This supersedes former Stage 3
not-started claims without changing historical checkpoints. The
[implementation contract](p5-stage3-implementation-contract.md), accepted D6/D7 in
the [integration/activation plan](p5-integration-activation-plan.md), and the
[P3 composition](p3-cron-composition-contract.md),
[occurrence ledger](p3-cron-occurrence-ledger-contract.md) and
[delivery/outbox](p3-cron-delivery-outbox-contract.md) contracts remain authoritative.

Workspace: `/Users/jaden/Documents/github/nanobot-harness-candidate` only. The
protected original was not accessed or edited; unrelated dirty changes were
preserved. No commit, installation/dependency change, deployment, activation, live
config/state/credential operation or real provider/channel/network effect occurred.
The production guard and disabled background gates remain unchanged.

## Delivered behavior

Atomic C2 v2 revision-commit scheduling evidence is bound to the existing C1
checkpoint/receipt/generation authority, without changing definition digest inputs.
Immutable first-commit interval anchors survive no-op, lost-ack retry and A→B→A.
V1 remains readable/manageable; missing scheduling evidence does not become
execution authority. Normalization is explicit stopped-writer work, not a read or
startup repair. Pure interval/at/cron rules use exact scheduled UTC instants and
explicit IANA timezones, including DST gap/fold and non-hour transitions.

The dormant shared graph reuses the exact foreground agent, SessionManager and
channel identities. One continuously held permanent POSIX lease owns timer
admission. Acquisition records elapsed unclaimed fires as exact bounded missed
ranges, retains claimed/completed exclusions and resumes strictly after one clock
snapshot without computation or automatic catch-up. Continuous delayed wakes admit
the exact scheduled due fire once. Lease/definition reproof fences immediately
before begin. Shutdown stops ingress first, owns cancellation-resistant effect
drain and cleanup, and retains explicit failures/uncertain state rather than
releasing resources while resistant effects remain.

Manual execution requires sealed explicit authorization; system requests require a
separately constructed named controller, with no generic system privilege or
Dream/Heartbeat factory. Their versioned source/sequence identities are disjoint
from unchanged timer v1 identities. Request allocation binds exact definition
tokens and admitted clock; restart allocations do not grant execution. Accepted
claimed-text composition preserves one canonical computation, bounded text,
durable completion-before-outbox/send, exact Telegram receipts, frozen fanout and
nonreplay. Callbacks are injection-only, independent of settled delivery.

Explicit-root inspection is existing-only, bounded and content-free, with no
fsync/publication/repair or live-path discovery. Operator reconciliation retains
the unchanged internally execution-fenced occurrence CAS. Strict scheduler v2
append-only observation bridges preserve original exclusions across owner/generation
transitions; they grant no computation, claim, completion or delivery authority.
Durable successor reproof precedes bridge publication and acknowledgement follows
both authorities. Exact lost-ack retry repairs only a missing exact bridge, never
reclaims twice. Missing continuity blocks startup until explicit repair. Acquisition
refresh accepts only a fully proved reconciliation-only extension of the pinned
state; ordinary/continuously-held ownership stays pinned. Unsafe forensic residue
and binding/content attacks fail closed and are preserved, never auto-repaired.

## Review, correction and root acceptance

The original implementation and independent reviewer used native **GPT-6.1 Sol**,
never GPT-5.6. Initial independent review identified two material defects:

1. Scheduler exclusions pinned an old owner/generation after valid explicit
   occurrence reconciliation, so resumed startup rejected the claim.
2. A newly authorized changed-clock manual/system request could borrow the earlier
   retained request clock and sequence after a pre-claim allocation failure.

Root approved the bounded strict-v2 observation-bridge, operator lost-ack ordering,
lease-epoch refresh and exact-clock-binding design in the contract. **One correction
round** resolved these findings, with no execution/reconciliation state-machine
change, invented grant, automatic recovery or activation expansion. Five primary
red reproductions preceded source changes. The worker's final affected four-module
selection reported **73 passed in 3.38s**; subsequent independent narrow recheck
reported **73 passed in 3.04s**, recommendation **PASS**. Root reviewed the actual
diff/code and raw evidence and reran the required gates below before issuing PASS.

Graph-capacity qualification: exhaustion of the original lifetime graph required
explicitly **USER-authorized review-chat coordination** with one fresh native
**GPT-6.1 Sol High** correction worker, followed by independent narrow recheck.
No model substitution occurred. Ownership transfer was exclusive; the original
implementation owner remained frozen during correction, and no agents overlapped
implementation writes.

## Root validation and reproducibility

Every selection below ran independently with copied/offline fixtures, pinned
Python `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13` and
`NANOBOT_DISABLE_NETWORK=1`. Existing fixtures also deny network and unauthorized
process effects; the environment variable alone is not the denial evidence. The
local POSIX lease worker-process test is the narrow permitted process exception.
**Counts overlap and must never be summed.**

| Root selection | Exact outcome |
| --- | --- |
| Eight new Stage 3 files, contract Section 10 first selection | 93 passed in 6.21s |
| P3 occurrence/composition/canonical/outbox/receipt/reconciliation selection | 266 passed in 10.78s |
| C1/C2/C3 action/conversion/definition/service/schema primitives | 221 passed in 8.24s |
| Prior Stage 1/2 composition/shutdown/adapter/startup selection | 87 passed in 19.45s |
| Unchanged Stage 2 engine/crash/temporary-session selection | 158 passed in 9.58s |
| Ruff, including all original changed/new test paths | Exit 0 |
| Exact source compileall | Exit 0 |
| `git diff --check` | Exit 0 |
| Independent narrow reviewer, four affected modules | 73 passed in 3.04s |

The first four pytest commands and standard lint/compile/diff commands are
reproducible from [contract Section 10](p5-stage3-implementation-contract.md#10-required-test-files-and-exact-validation).
The final root lint/compile also includes `nanobot/cli/cron_runtime.py`. Exact
commands, exits and raw output, including the additional unchanged Stage 2 command,
are retained in `/tmp/nanobot-stage3-correction1.0XTXzo/root-final-gates.txt`.
That additional selection is:

```bash
NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q \
  tests/agent/test_p5_stage2_engine.py \
  tests/migration/test_p5_stage2_crash_matrix.py \
  tests/session/test_p5_stage2_temporary_sessions.py --tb=short
```

## AC1–AC12 accepted evidence

| Criterion | Concrete delivered/test evidence |
| --- | --- |
| AC1 — commit evidence | `test_schedule_commit.py` and C1/C2/C3 gate cover atomic checkpoint/receipt/generation binding, retained anchors/history, exact lost-ack retry, forged/stale replay rejection and explicit-only normalization. |
| AC2 — schedules | `test_schedule.py` covers interval/at boundaries, explicit timezone, increasing UTC fires, DST gap/fold/non-hour transitions and wall-clock reversal. |
| AC3 — startup missed | `test_scheduler.py` covers exact acquisition snapshots, missed ranges/exclusions, claim-before-cursor and cursor-before-begin crash states, same/fresh instance resume and first strictly future fire with zero old computation/send. |
| AC4 — continuous due | Exact delayed wake and successor cursor tests claim scheduled timestamps once, not synthetic now or completion times. |
| AC5 — lease/binding | `test_scheduler_store.py` covers permanent-file contention, the local two-process/fork exception, root/file/residue attacks and bounds; correction refresh tests reject identical replacement, history rewrite, unexplained generation and held-owner changes. |
| AC6 — trigger authorization | `test_runtime_adapters.py` and request-store tests cover no unauthorized allocation, disjoint equal-clock manual/system sequences, unchanged timer v1 wire/hash, strict E2 trigger parsing, changed-clock/new-capability rejection and original pre-effect capability retry without duplicate effects. |
| AC7 — effects/nonreplay | Stage 3 composition and P3 gate prove exact canonical computation/Telegram receipt grants, frozen fanout, unsupported/approval/policy fences, no replay of started/uncertain state and callback failure/cancellation independence. |
| AC8 — lifecycle | `test_p5_stage3_shutdown.py`, composition and prior-stage gates cover stopped ingress before drain/channel close/lease release, pre-begin lease loss, repeated cancellation, resistant effects, transactional acquisition and explicit cleanup errors. |
| AC9 — operator | `test_cron_runtime_inspection.py` proves existing-only unchanged-object inspection, exact stopped-writer reconciliation, busy exclusions, durable two-authority lost-ack windows, explicit repair-before-next-reconcile, pre-mutation capacity checks and preserved forensic residue. |
| AC10 — dormancy/shared graph | `test_p5_stage3_composition.py` and prior-stage gates cover definition-only default with no optional timer/lease/callback/Dream/Heartbeat construction, exact shared foreground/session/channel identities and no legacy callback production registration. |
| AC11 — regression/scope | Root P3/primitives/prior/unchanged Stage 2 selections and hash audit preserve accepted ABI/state machines/frozen boundaries; unchanged mixed historical diagnostic adds zero failed nodes. |
| AC12 — evidence | Exact root gates, initial red evidence, independent GPT-6.1 Sol narrow PASS, reproducible contract commands and qualified unavailable type/native tools are recorded here; no full-suite green claim. |

## Historical diagnostics and qualifications

The unchanged finite mixed diagnostic reported **16 failed, 61 passed in 1.12s**,
exactly the same sixteen baseline nodes, with **zero additional failures**. It is
not a required-green selection or a full-suite green claim. Historical fixtures
were not weakened, removed or rewritten. Exact classification:

- Legacy `jobs.json` publication: the two
  `test_migration_cron_fields.py::test_service_serializer_preserves_identifiers_and_destinations`
  variants `[session_bound]` and `[legacy_fresh]`.
- Removed runtime-state listing ABI: `test_list_shows_last_run_state`,
  `test_list_shows_error_message` and `test_list_shows_next_run` in both
  `test_cron_tool_list.py` and `test_cron_tool_list_fork.py`.
- Rejected legacy route-clearing policy: `test_cron_tool_list.py::test_add_job_binds_current_session_key`
  and `::test_add_job_captures_owner_and_origin_without_legacy_delivery_fields`.
- Missing accepted mutation tokens: `test_list_excludes_disabled_jobs` in both
  tool-list modules; the old untokened enable/disable call cannot bypass C3.
- Removed context-setter ABI in `test_cron_tool_list_fork.py`:
  `test_add_cron_job_defaults_to_tool_timezone`,
  `test_add_cron_job_accepts_additional_destinations`,
  `test_add_cron_job_rejects_invalid_additional_destination` and
  `test_add_at_job_uses_default_timezone_for_naive_datetime`.

Earlier full-cron collection failure and interrupted historical timer diagnostics
remain baseline history in contract Section 10; they were not rerun as acceptance
gates. BasedPyright remains unavailable and **unqualified**; native Windows and
release/install qualification remain **unqualified**. The requests dependency
warning is unchanged. No tool/dependency was installed. Lost-ack fixtures inject at
durable publication boundaries; they do **not** claim abrupt process-death coverage
at every syscall. Unsafe staging/forensic residue remains failclosed, not repaired.

## Scope/hash audit and retained evidence

Root's original snapshot audit found only **12 originally authorized pre-existing
baseline paths changed plus 16 new Stage 3 paths**. Correction touched exactly its
eight authorized paths. Frozen core/guards/session/export and E2/E3 state machines
were preserved. Documentation acceptance publication is a separate four-path
documentation-only ownership transfer; it does not reopen implementation.

Original baseline snapshots/raw evidence are retained at
`/tmp/nanobot-stage3.dZZOBF`. Correction snapshots, pre/post SHA256 manifests,
five-node red reproduction and focused results are retained at
`/tmp/nanobot-stage3-correction1.0XTXzo`. The exact raw correction artifact is
`/tmp/nanobot-stage3-correction1.0XTXzo/correction-raw.diff`, SHA256:

```text
4101a6569e3819db8e7d029901e20e77723a09fb28b4fb752fb01fe2bc90c3bc
```

## Next work and still-open gates

**Stage 4 is next, requires separate scope/authorization, and is NOT STARTED.**
Stage 3 PASS is **not** P5 closure, D9 acceptance, guard removal, cron timer,
Dream/Heartbeat/callback/idle/retention/rematerializer activation, live normalization
or conversion, deployment/live traffic, P6/P7/H1/P8 approval, release installation
or native Windows qualification. No later gate has been accepted by implication.
Rollback remains disabling/removing the injected scheduler facade while retaining
the definition-only default and all durable claim/cursor/outbox evidence; never
delete uncertain state or reset retained anchors/classifications to simulate rollback.
