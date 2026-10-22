# P5 Stage 4 bounded offline implementation contract

Status as of 2026-10-01: guarded offline Stage 4 ROOT PASS; see the
[acceptance checkpoint](p5-stage4-acceptance-checkpoint.md). This retained
implementation contract does not authorize activation or P5 closure.

The authoritative bounded contract and independent validation manifest are retained at
`/tmp/nanobot-stage4.WV8YSi/worker-contract.md` and `manifest.json`. Stage 4 and §4.5
of [the integration plan](p5-integration-activation-plan.md) govern. Accepted Stage
1–3 and P3/P4 contracts remain unchanged. Root owns acceptance and fresh review.

One sole worker owns exactly two narrow error-import corrections, new lightweight
`runtime_errors.py`, private pure `_integration_diagnostics.py`, and four new
Stage 4 fixture/test files. No other production behavior or existing tests change.
All state is temporary/copied; provider/tool/channel effects are fake. Network and
process effects are denied by the manifest OS profile and local fixtures.

AC1–AC8 cover joined four-surface/two-mode real runner identity, deterministic
same-key serialization and cross-key progress, stopped-writer session and whole
cron runtime restart/nonreplay, startup/dormancy/lifecycle, bounded privacy/shadow,
unchanged contract regressions and exact evidence/scope audit.

Diagnostics compare normalized root public history and explicit archive boundary,
not checkpoint summaries or semantic summary quality. Canonical projection uses
`project_source(..., use_checkpoint=False)`. Routine reports contain finite enums
and bounded counts only. Explicit private operator comparison may return SHA256
digests of frozen public evidence; it grants no mode or execution authority.
No logger/exporter, backend, CLI, config flag, discovery, persistence or hook is added.

The shared error class preserves old `nanobot.cron.runtime` import/pickle lookup
and identity; its defining `__module__` becomes `nanobot.runtime_errors`. Cleanup
ordering, failure groups, resource fencing and retry remain unchanged.

TDD records diagnostics/import red before behavior changes, then focused green;
root independently runs each finite manifest gate and audits frozen hashes.
No PASS/checkpoint publication occurs here. DOCX node
`tests/test_api_attachment.py::test_docx_upload_passes_media_path` is environment
blocked/P6, not run/pass. BasedPyright, installed/native/Windows qualifications
remain open. No installs, commits, live operations, guard change, deployment,
activation, P5 closure or Stage 5 is authorized.

## Accepted bounded design and privacy contract

`RuntimeOwnedWorkError` has one dependency-free definition and an identical
reexport at `nanobot.cron.runtime`. Only three optional imports in composition
cleanup are replaced with the lightweight import. Cleanup ordering, exception
groups, FAILED/retry and resistant-effect fencing stay exact; no gate-dependent
classification or state/schema change is permitted. `CronWorkFailedError` stays put.

The private diagnostics module performs no I/O and has no logger, exporter,
backend, configuration, production observer/hook, runner or persisted format.
`diagnostic_payload` accepts exactly component/result/left_count/right_count.
Components are fixed shadow/admission/session/conversion/scheduler/occurrence/
outbox/callback/channel/startup/shutdown enums. Results are fixed equal/
public_history/tool_pairing/archive_boundary/unsupported/invalid_input/
bound_exceeded classifications. Counts are exact integers 0–1,000,000; bool and
float are rejected. Unknown fields/labels fail with constant content-free errors.
No IDs, paths, content, exceptions, references, receipts, credentials or digests
enter the routine plane.

`legacy_evidence` freezes supplied detached public message trees;
`canonical_evidence` validates supplied detached SourceSnapshot events and uses
existing sanitization and `project_source(use_checkpoint=False)`. Input bounds are
262,144 bytes, 512 messages/events, nesting depth 16 and 128 tool calls; finite
ordinary JSON and paired function calls/results are required. Unsupported rich
content/unknown fields are rejected rather than truncated or treated as equal.
Canonical private linkage is validated, then only retained root public history
is frozen. Archive comparison counts the projected root prefix, not hidden child
events. Checkpoint text, provider state and arbitrary metadata are not semantic
equivalence objects. The API does not claim summary quality or broader metadata
equivalence. Frozen normalized evidence stores private immutable bytes, excluded
from repr. Explicit `operator_compare` returns exact SHA256 over those bytes plus
the archive offset, with digest fields excluded from repr. Its routine member is
still classification/count only. Digests never become routine labels/logs.

Shadow never discovers paths, loads/mutates stores, claims/converts/exports,
rematerializes, invokes a model/tool/channel/callback, or schedules. Portable
operator evidence grants neither authoritative-mode selection nor execution.
This preserves §4.5's routine/private operator distinction, not a new digest-label
policy. User-authorized transcript/debug/compatibility-file logging is outside
this diagnostic seam's scope.

## AC1–AC8 and concrete evidence required

| AC | Required joined evidence |
| --- | --- |
| AC1 | Actual RuntimeComposition/DurableSessionAuthority/AgentLoop/AgentRunner; direct, actual API handler, SDK facade and authorized real Telegram fake update share one handle and one execute_turn/runner per ordinary request. Both registered modes coexist. Provider calls are counted separately from runner/tool effects. |
| AC2 | Event barriers and the real per-key lock prove same-key serialization and cross-key progress for legacy/legacy, canonical/canonical and both mixed orders; no timing sleep is proof, all waits bounded. |
| AC3 | Actual fake-tool turn, close/join, stopped-writer conversion pre/postcommit and export publication faults; fresh operator recovery then fresh composition next request. Independent tool count remains one; source stays canonical and exports never select authority. No foreground crash retry or fabricated approvals/receipts. |
| AC4 | Whole cron composition close/rebuild at prebegin, started, uncertain and settled boundaries with fake clock/provider/Telegram receipts. Independent counters and exact old occurrence/outbox/callback bytes/views prove no old computation/send/callback replay. New explicit work is a distinct identity. Stage 3 continuity/state machines remain unchanged. |
| AC5 | Sentinel-backed mode-witness/journal/sidecar, P4 recovery/quarantine, TTL/idle, Telegram/plugin and missing scheduler-continuity startup denial before effects, with rollback/no owned tasks. Default acquire/turn/close has no optional graph/imports/background provider/directories. Lightweight cleanup covers clean, ordinary failure and resource-fenced FAILED/retry; resistant real runner retains resources until join and explicit retry. |
| AC6 | Fixed label/count validation, adversarial sentinels, finite/size/depth/message/tool limits, true mismatch classes, real conversion equality, boundary freeze and unchanged input/files. Routine repr/errors/report have no content/digest. Operator digests are explicit private evidence only. |
| AC7 | Each current-contract manifest selection independently green; no added historical cron failure node. Preserve ordinary/new/temporary sessions, all frozen guards/core/runner/source/checkpoint/artifact/recovery/export/cron/receipt rules and unrelated dirty edits. Existing tests/conftest are not edited. |
| AC8 | Captured deterministic diagnostics/import red before changes; focused green; root independent gates and fresh GPT-6.1 integrated review; exact Ruff/compile/diff/scope/hash/untracked raw diff. At most two root correction rounds; no optional expansion. |

Test evidence belongs to the new `test_p5_stage4_integration.py`,
`test_p5_stage4_recovery.py`, `test_p5_stage4_diagnostics.py` and shared
`p5_stage4_fixtures.py`. Red proves the previously absent diagnostics seam and
the cleanup optional-import defect. Original behavior is also independently
reproduced by
`tests/agent/test_migration_sdk.py::test_from_config_real_runner_keeps_default_trace_private_and_sanitizes_streamed_opt_in`.
Faults are injected at durable publication/actual computation boundaries;
these fixtures do not claim abrupt process death at every syscall.

## Exact finite validation manifest

Every selection below runs independently; overlapping counts must not be summed.
Use `NANOBOT_DISABLE_NETWORK=1 /usr/bin/sandbox-exec -p '<profile>'
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q
<arguments> --tb=short`. Profiles are literal:

```text
full:
(version 1) (allow default) (deny network*) (deny process-fork) (deny process-exec) (allow process-exec (literal "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13") (literal "/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python"))
local-process:
(version 1) (allow default) (deny network*) (deny process-exec) (allow process-exec (literal "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13") (literal "/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python"))
lease:
(version 1) (allow default) (deny network*)
```

Full profile module selections:

```text
focus:
tests/migration/test_p5_stage4_integration.py tests/migration/test_p5_stage4_recovery.py tests/migration/test_p5_stage4_diagnostics.py
prior:
tests/migration/test_p5_stage1_composition.py tests/migration/test_p5_stage1_shutdown.py tests/migration/test_p5_stage1_adapter_matrix.py tests/migration/test_p5_stage2_composition.py tests/cli/test_startup_admission_agent_gateway.py tests/cli/test_startup_admission_onboard_webui_api.py
modes:
tests/session/test_session_mode_registry.py tests/session/test_session_mode_conversion.py tests/session/test_session_mode_export.py tests/session/test_session_mode_operations.py tests/migration/test_p5_stage2_crash_matrix.py tests/agent/test_p5_stage2_engine.py
consumers:
tests/session/test_p5_stage2_temporary_sessions.py tests/session/test_p5_stage2_consumers.py tests/webui/test_forking.py tests/utils/test_webui_workspaces.py
adapters:
tests/channels/test_telegram_headless_composition.py tests/channels/test_telegram_migration_differential.py tests/agent/test_migration_api_engine.py tests/agent/test_migration_sdk.py
cron:
tests/cron/test_schedule.py tests/cron/test_schedule_commit.py tests/cron/test_scheduler.py tests/cron/test_runtime_adapters.py tests/migration/test_p5_stage3_composition.py tests/migration/test_p5_stage3_shutdown.py tests/cli/test_cron_runtime_inspection.py
substrate:
tests/agent/test_memory_recovery_startup_admission.py tests/agent/test_memory_recovery_constructor_admission.py tests/session/test_source_projection.py tests/session/test_artifact_operator.py tests/session/test_artifact_rematerialization.py tests/session/test_artifact_factory.py tests/session/test_artifact_store.py tests/providers/test_conversation_state.py
p3:
tests/cron/test_occurrence_store.py tests/cron/test_cron_package_d.py tests/cron/test_cron_package_e.py tests/cron/test_canonical_execution.py tests/cron/test_canonical_result_adapter.py tests/cron/test_cron_composition.py tests/cron/test_cron_callback_store.py tests/cron/test_cron_channel_sender.py tests/cron/test_cron_delivery_runtime.py tests/cron/test_cron_delivery_store.py tests/cron/test_delivery_reconciliation.py tests/cron/test_visible_completion_link.py tests/cli/test_cron_delivery_reconciliation.py tests/agent/test_cron_silent_turn.py
primitive:
tests/cron/test_action_store.py tests/cron/test_action_store_conversion.py tests/cron/test_definition_store.py tests/cron/test_cron_definition_service.py tests/cron/test_cron_tool_schema_contract.py
historical (diagnostic, not required-green):
tests/cron/test_migration_cron_fields.py tests/cron/test_cron_tool_list.py tests/cron/test_cron_tool_list_fork.py
```

For substrate/p3/primitive add one `--deselect='<node>'` per exact node below
to the full-profile command, then independently run only those same nodes under
local-process profile. No new test receives a process exception:

```text
substrate:
tests/session/test_artifact_store.py::test_concurrent_distinct_blobs_cannot_exceed_namespace_quota
tests/session/test_artifact_store.py::test_two_processes_reuse_one_complete_immutable_blob
tests/session/test_artifact_store.py::test_two_processes_install_missing_once_and_reuse
tests/session/test_artifact_store.py::test_actual_crash_between_link_and_unlink_recovers_only_verified_staging_link
tests/session/test_artifact_store.py::test_install_missing_recovers_actual_crash_after_link
tests/session/test_artifact_store.py::test_install_missing_recovers_actual_crash_before_link[8]
tests/session/test_artifact_store.py::test_install_missing_recovers_actual_crash_before_link[64]
p3:
tests/cron/test_occurrence_store.py::test_only_one_process_wins_the_exclusive_claim
tests/cron/test_occurrence_store.py::test_fork_inherited_claim_cannot_complete
tests/cron/test_occurrence_store.py::test_fork_child_drops_inherited_fence_without_unlocking_or_retaining_parent_lock
tests/cron/test_cron_delivery_store.py::test_two_process_reservation_has_one_winner
tests/cron/test_cron_delivery_store.py::test_fork_inherited_claim_has_no_authority
tests/cron/test_cron_delivery_store.py::test_fork_child_drops_inherited_send_fence_without_unlocking_parent
primitive:
tests/cron/test_action_store.py::test_append_generation_cas_serializes_concurrent_new_actions
tests/cron/test_action_store.py::test_append_and_checkpoint_race_invalidates_the_checkpoint_generation
tests/cron/test_action_store.py::test_two_process_append_and_checkpoint_races_are_serialized
```

The independent lease-profile command runs exactly
`tests/cron/test_scheduler_store.py`; its existing lease fixtures require local
processes. Root verifies OS denial separately; environment variables alone are
not isolation evidence. Local fixture network/process denial remains unchanged.

Run scoped Ruff with `/Users/jaden/.pyenv/versions/3.13.1/bin/ruff check` over
the eight owned Python paths, never `ruff format`; Python 3.13 compileall over
those same paths; `git diff --check`; baseline/final SHA256 scope comparison and
correction-only raw diff including new untracked paths. Root runs the broad
finite regressions; worker does not duplicate completed root gates.

## Sole write set, frozen boundaries and handoff

Exactly these nine paths are authorized:

```text
nanobot/runtime_errors.py
nanobot/_integration_diagnostics.py
nanobot/runtime_composition.py (only lightweight error import correction)
nanobot/cron/runtime.py (only same-class import/reexport)
tests/migration/p5_stage4_fixtures.py
tests/migration/test_p5_stage4_integration.py
tests/migration/test_p5_stage4_recovery.py
tests/migration/test_p5_stage4_diagnostics.py
docs/migration/p5-stage4-implementation-contract.md
```

All other paths and unrelated dirty changes are frozen. No existing test/global
conftest edit, source/state-machine/schema change, guard edit, historical checkpoint
or plan mutation, installation, commit, release/deployment/activation or live
config/state/credential/network/provider/channel operation is authorized. Any
additional owning-stage defect requires return to root before a forbidden edit.
The protected original workspace must not be accessed.

DOCX exact attachment node remains environment-blocked/P6 packaging, not run/pass;
no synthetic dependency or installation. BasedPyright is absent; Windows/native/
installed qualification remains open. The requests warning is unchanged. The
finite historical cron diagnostic must add zero nodes to its exact sixteen
accepted historical failures, never enter green counts. Separate loopback WebSocket
diagnostics blocked by OS network denial remain environmental evidence, not
permission for a new network exception or fixture weakening.

Completion is READY/SOURCEFREEZE with exact files, AC mapping, red/green/raw
outputs, hashes, untracked raw diff and qualifications, followed by STOP. Root
alone accepts after fresh GPT-6.1 review. No acceptance checkpoint, P5 closure,
Stage 5, D9, live conversion, retention/deletion, scheduler/Dream/Heartbeat/
callback/idle/rematerialization activation, deployment or Windows claim follows
from this implementation contract.
