# Nanobot remaining work

Implementation update: **2026-10-23**, branch `codex/complete-remaining-work`.
The accepted prototype features are implemented behind their explicit gateway
settings. Production remains unchanged. See
[configuration, behavior and rehearsal guide](assistant-prototype-integration.md)
for the current implementation; the starting assessment below is retained as
historical scope and acceptance criteria.

| IDs | Current implementation status |
| --- | --- |
| A1–A4/A7 | Existing-repository catalog, compact discovery, on-demand reading, automatic selected-knowledge tool/instructions and opted-in safe Git sync implemented; project switch and managed-policy checks pass |
| A5 | Durable external-process completion queues a continuation to the same assistant; cancellation, duplicates, scope and restart cases covered |
| A6 | Selected persistent Telegram owner wired after authentication and before polling; status/stop/steering/queued work use the existing inbox. Temporary/shared cross-channel admission remains outside scope |
| A8 | Domain workflow, provider setup and optional crawler/VPN registration extracted to extension modules; compatibility methods and capability policies retained |
| M1 | Contract fields, external checks, topic receipts, owner alerts, bookkeeping and idempotent copied-state import implemented |
| M2 | Explicit normal gateway timer, canonical execution, real status, transport readiness, shutdown and restart handling implemented; disabled SDK/API/default gateway paths remain inert |
| M3 | Bounded Codex primary-window recovery, durable intent/budget and successful-delivery reset implemented; uncertain/tool-effect retries refused |
| M4 | Local full-suite, strict checking, build/install and copied-state qualification passed. Deployed identity/drift and actual client workflow/cutover rehearsal remain production release checks |

The local namespace-inventory edits and prior research were preserved. Manifest,
external skill discovery, legacy Copilot import and scheduler atime fixes are
included. Upstream parity, a Rust rewrite and a new semantic task-context system
remain unnecessary for this direction.

## Starting assessment — retained from 2026-10-22


Date: 2026-10-22. Candidate: `chore/upstream-migration@79934a5630ca419e02e124693efa938d9d3407f0`,
plus the pre-existing local namespace-inventory repair. This consolidates the
source review and subsequent product discussion. It is a remaining-work
assessment; this update does not implement or activate the listed features.

The comparison reference for our newer main behavior is
[`main@0976742`](https://github.com/JadenChun/nanobot-custom/commit/09767426d64ab4c346c94bf04be59684aa5d13ba).
Earlier production-audit notes are useful inputs. This review did not reconnect
to the production server; confirm any drift before a release rather than
repeating the completed repository research.

## Direction and existing foundations

Reuse nanobot for one personal assistant that can work across existing
repositories in one conversation. A small catalog helps discovery; the agent
chooses what to read through ordinary tools. A separate semantic task-context
manager, required context-repo schema, duplicate checkouts, Rust rewrite, and
latest-upstream parity are not requirements.

Useful knowledge must be saved automatically. Automatic commit/push applies
only to designated knowledge repositories; code repositories retain their own
Git workflows. Runtime sessions, private artifacts and credentials remain
separate from repository knowledge. Marketing, VPN and crawler behavior belongs
in optional tools/skills/scripts/plugins. Background work means external work
followed by continuation of the same assistant, without another model worker.

The agent loop, file/shell tools, project instructions, skills, sessions, runtime
memory, provider resilience, workspace scopes and context-budget machinery
already exist. Our fork additionally supplies a configured repository index,
repository policies and Git-sync utilities. Their existence is not proof of the
complete automatic catalog/memory/synchronization workflow.

## First: compatibility required before upgrading the marketing agent

| ID | Remaining work and current evidence | Completion criterion |
| --- | --- | --- |
| M1 | **Preserve and import the actual marketing job contracts.** Candidate `CronPayload` lacks newer main's external verifiers, owner-alert routes, per-job planning/verification overrides and post-delivery commands. Strict legacy conversion rejects unknown fields. Existing stores/conversion code are not proof that the deployed jobs import successfully | Copied job/state fixtures import without losing timing, timezone, identity, destinations, thread metadata, checks, alerts or bookkeeping. Repeated import is safe. Keep business scripts and validators in configured adapters rather than marketing-specific loop logic |
| M2 | **Enable qualified scheduled execution in normal gateway composition.** The gateway still says `Cron: definitions only; runtime inactive`. `CronRuntime` requires explicit offline injection, and acquiring it does not start its timer | A due job runs through real gateway startup and the canonical runner. Ownership, status, shutdown and restart work; disabled and direct SDK/API paths remain timer-free. Do not simply remove the injection guard |
| M3 | **Restore whole-job quota recovery and verify provider retry parity.** Five-hour Codex rescheduling and its persisted state are missing. Candidate transient provider retry support already exists in a different implementation | Preserve bounded primary-window rescheduling, reset-plus-buffer timing, weekly exhaustion handling, retry-budget persistence and successful-delivery reset. Compare main's precise transient/exhausted-retry behavior without multiplying whole-job retries or replaying effects |
| M4 | **Qualify installation, state/configuration migration and release.** Parsing the old configuration is insufficient. New serialization changes the legacy token-budget shape; enabled channels need their dependencies; sessions and memory need copied-state rehearsal | Use separate copied configuration/state; verify enabled-channel installation, historical data reopening, checked migration and rollback. Rehearse accepted workflows with isolated delivery. Cutover has one active scheduler and does not repeat already completed work or delivered results |

Reuse the existing production audit, context-repo revision information and job
inventory when building the fixtures. Fill missing evidence and check drift;
do not restart the research from scratch. Exact live compatibility remains a
release check, not an inference from config parsing or foreground test counts.

Detailed contract and validation:
[marketing scheduling compatibility plan](marketing-agent-scheduling-plan.md).
Repository-sync or provider compatibility is also a release requirement wherever
the existing client's actual configuration depends on it; a general-purpose new
catalog is not a prerequisite merely because it shares the repository subsystem.
Source: [payload/state types](../nanobot/cron/types.py),
[strict conversion](../nanobot/cron/action_store.py),
[gateway startup](../nanobot/cli/gateway_runtime.py),
[runtime composition](../nanobot/runtime_composition.py), and
[scheduler runtime](../nanobot/cron/runtime.py).

## Personal-assistant integration left to complete

| ID | Remaining work and current evidence | Completion criterion |
| --- | --- | --- |
| A1 | **Make repository discovery agent-maintainable.** `ContextRepoManager` is a configured path index; the discussed generic catalog is not yet an agent-maintained workflow | The assistant can discover, remember, inspect and update references to existing repositories, their purposes and knowledge-repo associations. No cloning or mandatory repository manifest. Catalog representation remains an implementation choice; an ordinary data file may suffice |
| A2 | **Use context on demand and repair external skill discovery.** The agent already reads files/skills, but configured context entrypoints, supplemental memory and run summaries are loaded together. `library` skills are discovered but omitted from the summary's source groups | Keep discovery information small; relevant details can be read as needed. External skills appear with correct locations and precedence. Preserve required identity/policy guidance and the existing client's context adapter |
| A3 | **Connect automatic knowledge saving to the correct destination.** Runtime memory storage and arbitrary file writes exist; automatic project/client knowledge write-back is not established | Useful facts and agreed decisions reach the designated knowledge repo and can be recovered later, including after compaction. Preserve existing store/script/proposal policies. Do not copy private runtime transcripts wholesale into knowledge repositories |
| A4 | **Connect reliable, knowledge-only Git synchronization.** Helpers exist but have no package runtime caller. The current helper includes rebase/reset-based conflict recovery and uses the shared Git index when committing | Only opted-in knowledge repos sync automatically. Retain local knowledge on failure and report actual sync state. Avoid including unrelated staged work, overwriting another writer, or applying reset-based recovery to general code repos. Serialize conflicting Git operations and demonstrate conflict/retry behavior before reusing the helper |
| A5 | **Complete process completion → assistant continuation.** Exec sessions can return while a process runs and support output/status/cancellation; their manager does not publish a completion into an assistant turn | Completion wakes an idle owner or queues behind active work through existing canonical ownership. Retain job/output identity, original instructions or references, project path, session and reply destination. Cover failure, cancellation, duplicate completion and restart; do not assume a process survives runtime shutdown |
| A6 | **Finish the selected busy-input/channel integration.** H1 has local control/inbox evidence, but normal Telegram startup does not call `register_telegram_busy_inbox`; whole-H1 acceptance remains outstanding | Demonstrate the intended status/stop/steering/queued-work behavior through actual chosen-channel composition. Preserve authorization, origins, output routing and existing WebUI recovery. Treat unimplemented temporary/cross-channel cases explicitly; offline seam tests alone do not qualify live operation |
| A7 | **Verify cross-project behavior using existing tools before adding architecture.** Scope-aware tools exist, while some foreground managers are constructed around a fixed workspace | Exercise A → B → general conversation → A in one chat; checks do not create files in the previous repo; knowledge is retrieved and saved correctly. A background completion for A retains A's directory while the conversation is on B. First assess instructions, explicit paths and current scopes; add runtime changes only for demonstrated gaps |
| A8 | **Remove domain coupling through optional extension boundaries.** VPN registrations and a crawler-specific role/configuration currently occur in core factory/composition paths | Marketing/VPN/crawler capabilities remain available to the client through optional tools, skills, scripts or plugins. The generic harness does not need to understand their domain workflow. Preserve enforceable capability boundaries while moving registrations; avoid a wholesale extension-system rewrite |

Source: [context construction](../nanobot/agent/context.py),
[configured repository index and manifest defaults](../nanobot/context_repo/manager.py),
[skills summary](../nanobot/agent/skills.py),
[workspace scope](../nanobot/security/workspace_access.py),
[foreground composition](../nanobot/agent/foreground.py),
[core tool factory](../nanobot/agent/tools/fork_factories.py),
[Git utility](../nanobot/utils/git_sync.py),
[exec sessions](../nanobot/agent/tools/exec_session.py),
[local-trigger coordination](../nanobot/triggers/local_turns.py), and
[H1 contract](migration/h1-implementation-contract.md).

## Other findings to retain

- **Manifest failure must not widen writes.** A manifest read/parse failure
  currently marks the repo unmanaged; unmanaged writable patterns become `**`
  when not explicitly read-only. Distinguish a deliberately plain repository
  from a failed managed policy. Fix and cover that boundary before relying on
  it for automatic knowledge writes.
- **Filesystem policy and arbitrary shell access are different guarantees.**
  Existing shell checks do not provide process-level isolation or equivalent
  enforcement of every manifest write rule. Keep read/write roots and explicit
  directories consistent with the selected execution mode; do not claim shell
  filtering supplies containment. Full host access remains a deployment choice.
- **Copilot compatibility is conditional.** Its legacy discovery path does not
  include the old default `~/.nanobot/copilot_auth.json`. Resolve and test that
  import if upgrading an instance that uses it. Do not make an unused provider
  a prerequisite for the Codex-based marketing deployment.
- **Preserve the local namespace-inventory work.** The pre-existing edits in
  `nanobot/session/mode_storage.py` and
  `tests/session/test_live_namespace_inventory.py` belong in current validation;
  do not discard them or treat the committed HEAD as the entire checkout.
- **Measure efficiency after the functional paths work.** Compare context
  tokens, repeated reads, model calls and end-to-end latency on representative
  workflows. Separate model/tool waiting from harness overhead before proposing
  language or runtime changes. No performance improvement is established here.

## What is not a missing feature to rebuild

- Main already had API-key pools/pacing; the candidate reorganizes that behavior.
- Candidate Codex transient-error handling exists. Exact parity needs checking;
  a missing old constant does not mean all retry support is absent.
- Main's `spawn_pipeline` manager refuses execution. It is not a working
  production feature that must be restored for parity.
- Imported background subagent code is excluded from registration. The agreed
  background-process/continuation requirement does not require restoring it.
- Context repositories already have native integration; they were not only a
  skill. The work is simplifying discovery and connecting saving/synchronizing.
- Dream, idle compaction, heartbeat and other autonomous callbacks remain
  separately dormant. Do not enable all of them merely to implement automatic
  knowledge saving or process completion. Preserve a specific old behavior only
  where an accepted workflow requires it.

## Validation and recommended order

The 2026-10-19 closeout records 3,637 passing tests across its final batches and
zero strict diagnostics for its tested source/environment. The later reviews
record 57 context/capability checks, 33 provider checks, and 167 busy/process
checks. Those results have different scopes and are not a fresh full-suite pass
for this checkout or evidence that dormant production integrations work.

This consolidation freshly ran
`.venv/bin/python -m pytest -q -p no:cacheprovider tests/session/test_live_namespace_inventory.py`:
**1 passed in 0.50s**. It checks the pre-existing local namespace repair's new
session/use/reopen case, not that repair's complete checker/regression evidence.
The command, output and identical before/after hashes for 541 runtime/selected-
test files are retained in `/tmp/nanobot-remaining-work-1_31pucj/`.

Before release, run unchanged strict checking to zero, the required local full
test suite, affected integration checks and a clean package/enabled-channel
installation. For Python repairs, retain frozen before/after full-source
reports under the same dependency environment and zero new signatures as
required by [AGENTS.md](../AGENTS.md) and the
[P6 attribution policy](migration/p6-error-attribution.md). Do not silently
replace the diagnostic baseline or reopen closed stages solely from stale notes.

The first production implementation slice should preserve job payload/state
compatibility and retry contracts (M1/M3), followed by qualified scheduling
composition (M2) and release rehearsal (M4). Repository/skill policy fixes and a
small catalog → on-demand reading → knowledge-save/sync workflow (A1–A4/A7)
can be developed on the prototype without changing production. Background
completion and busy-channel integration (A5/A6) are separate bounded work.
Keep extension cleanup (A8) compatible with the client's existing capabilities.

Related evidence: [customization inventory](customization-feature-inventory.md),
[accepted direction](harness-prototype-direction.md),
[foreground closeout](migration/local-runtime-closeout.md), and
[historical migration review](migration-branch-review.md). This document and
the corrected inventory supersede conflicting feature-status or port estimates
in the older discussion notes.
