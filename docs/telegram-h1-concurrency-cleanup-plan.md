# Telegram H1 and global concurrency cleanup

Prepared: **2026-11-03–04**. Status: **implementation complete; retained as the cleanup contract**.
Decision record: [SuperAgent research and future session direction](superagent-research-and-session-direction.md).
Code baseline: `a90191a41b025884e76ce029b98a5d5349171f2e`.

## Outcome

Remove the Telegram-specific H1 busy-input coordinator and the optional
all-model inbound-turn semaphore. Keep the current desktop topics, ordinary
Telegram channel, canonical agent loop, and session ownership. Independent
sessions may progress concurrently; the provider or inference server owns its
backend capacity. A single session still has one canonical execution owner.

This prepares the existing architecture for future agent-managed sessions. It
does **not** implement a parent/router agent, worker-session scheduler, session
dashboard, cross-channel shared conversation, or new local inference scheduler.
The future main/router UI must show working sessions, status, and requests for
attention at a glance and retain direct access to each session; its design
remains deferred with the other open SuperAgent questions.

## Prepared workspace

| Item | Value |
| --- | --- |
| Branch | `codex/telegram-h1-concurrency-cleanup` |
| Isolated worktree | `/home/jaden/.codex/worktrees/agent-session-cleanup/nanobot-custom` |
| Original checkout | `/home/jaden/Documents/Utilities/nanobot-custom` |
| Python environment | Worktree-local `.venv`; independent of the running desktop environment |
| Worker entry point | [Coding handoff](../.agent/cleanup-handoff.md) |
| New-case authoring input | [Public requirements contract](../.agent/cleanup-test-requirements.md) |
| Local evidence | `.assets/validation/cleanup-2026-11-03/` in this worktree; ignored by Git |

The original checkout has unrelated uncommitted changes, including
`nanobot/webui/ws_http.py`. They were not copied into the code baseline and must
not be reset, stashed, reformatted, or incorporated opportunistically. Research
documents and their linked mobile/session-recovery plans were copied as context
snapshots; those separate plans are not implementation scope. Their original
hashes and provenance are in `context-snapshot.json` in the evidence directory.
Local sibling-repository links were rebased to pinned source URLs for this
isolated worktree; this does not change those separate plans' scope.

No live gateway, Telegram account, model server, configuration, session storage,
or GPU allocation is part of this preparation or the offline implementation.

**Draft distribution, 2026-11-04:** copy the research and preparation documents
to local `staging` through a documentation-only commit. This does not merge the
cleanup branch's product tree into `staging`, rebase its frozen code baseline,
or activate a runtime change. `staging` already has newer unrelated product
commits; preserve them when integrating a reviewed cleanup delta later. Keep
implementation in the isolated worktree until its own acceptance is complete.

Preparation checks passed in the final frozen environment: Python 3.14.8,
BasedPyright 1.40.2, Ruff 0.17.1, dependency consistency, and imports resolving to
this worktree. All declared project extras and discovered channel dependencies
are installed. `uv` is unavailable, so preparation used the independent venv
and pip; the exact 422-distribution inventory and command evidence are retained.
Full unchanged strict checking: **528 files, 0 errors, 0 warnings, exit 0**.
The eight focused baseline smoke tests pass. These checks establish the starting
point, not acceptance of cleanup code. Preliminary reports from before installing
missing optional dependencies remain separately labelled `pre-extras-*`.

## Required behaviour

1. **Ordinary channel operation:** Telegram authentication/pairing, normal
   messages, media/albums, replies, routing, streaming, and existing stop handling
   continue through the usual bus/turn path. There is no separate H1 routing
   model, H1 steering path, durable H1 enqueue, or H1 startup/shutdown task.
   This does not promise conversational replies to busy input or agent-managed
   follow-up routing; those belong to the future session design.
2. **Independent sessions:** changing `NANOBOT_MAX_CONCURRENT_REQUESTS` no longer
   changes admission or makes unrelated sessions share a turn slot. Remove its
   parsing and semaphore, including obsolete private-field test scaffolding.
   Do not substitute another global gate. Preserve provider/key pacing, scoped
   tool fan-out limits, delegated-work budgets, and cancellation settlement.
3. **Session integrity:** bus, direct, and desktop entry points retain canonical
   same-session ordering, execution/source ownership, checkpoint integrity,
   permission boundaries, and reply destinations. Existing `unifiedSession`
   semantics remain separate; do not force all conversations into one key.
4. **Other asynchronous work:** process completion still resumes its original
   assistant with its original project, session, and route. Preserve generic
   pending follow-ups, cron/local-trigger execution, recovery, and shutdown.
5. **Retired configuration:** an absent legacy `gateway.telegramBusy` block or
   a valid disabled block remains loadable, including the supported snake-case
   alias. Omitted `enabled` means disabled. An enabled or malformed retired
   block must produce an actionable config-load error before gateway work starts;
   silently ignoring an enabled feature is unacceptable. Remove the feature
   from normal schema output/examples; keep only the narrow compatibility
   validation needed for this contract. Saving unrelated settings must not
   resurrect H1. Do not print credentials or the whole legacy block in errors.
6. **Previously accepted input:** removing H1 must not erase, replay, relabel as
   completed, or silently abandon stored accepted/uncertain work. Keep a small
   read-only legacy-state parser/guard where needed. An affected session with
   unsettled work, pending notices/receipts, uncertain effects, an unfinished
   temporary lifetime, or unrecognisable H1 metadata must refuse normal turn
   admission and destructive reset/replacement/deletion with a clear legacy-H1
   recovery explanation. Read-only inspection and unrelated sessions remain
   available. Fully settled legacy metadata may remain inert. Preserve existing
   private recovery namespaces and their visibility restrictions.

The legacy guard is deliberately narrower than a replacement inbox. Do not add
new H1 writers, a replay engine, an automatic migration/export system, or a new
session manager. Reuse the established persistence and admission boundaries.
Actual pending data needs an explicit disposition before any later operational
cutover; there is no authorization here to edit a user's session files.

## Work sequence and ownership

Use one implementation writer at a time for the shared loop/runner/session
boundary. Other work may be in progress in sibling worktrees; do not revert it.
Start from this prepared branch and read repository guidance before coding.

### W0 — establish deletion-aware validation evidence

Freeze the environment and retain a full unchanged strict-check report plus
source hashes **before** source edits. The current comparator
[`scripts/check_type_regressions.py`](../scripts/check_type_regressions.py)
rejects every decrease in `filesAnalyzed`, even when a module is intentionally
deleted. H1 removal will encounter that restriction.

Make a bounded preparatory change to that script and its focused contract
coverage: allow an explicit deletion manifest backed by complete before/after
in-scope source inventories. Acceptance requires the inventory counts to match
the raw checker counts, the removed paths to match the declared deletions, and
the current inventory to match actual remaining in-scope sources. Preserve
default refusal when the extra evidence is absent or inconsistent. Checker
version/policy checks and comparison of **every** diagnostic signature and
multiplicity remain mandatory. Do not suppress new diagnostics in retained or
new files. The current policy includes `nanobot` and excludes `**/tests`;
record any mismatch rather than guessing an allowed count reduction.

Keep environment/source attestation alongside the reports as before. Do not
edit checker JSON, refresh the baseline after deleting modules, weaken strict
policy, or retain empty modules solely to keep the count unchanged. This small
validation change is a prerequisite, not a general checker redesign. New
behaviour cases must be independently authored under the public contract.

### W1 — remove the global turn cap

Primary ownership: `nanobot/agent/loop.py`, the obsolete entry in
`nanobot/agent/tools/self.py`, affected admission/cancellation tests, and the
active runtime-environment guide in `docs/configuration.md`.

Remove environment parsing and the whole-turn semaphore at canonical admission.
Keep same-session locks and all source-owner checks. Adjust tests that currently
manufacture a private concurrency gate; preserve their observable cancellation,
waiting-turn cleanup, and compaction outcomes using remaining public behaviour.
Document that the retired variable has no effect, including positive and
non-numeric values. Do not change provider implementations to compensate.

Relevant existing coverage: `tests/agent/test_loop_concurrency.py`,
`test_migration_canonical_admission.py`, `test_migration_bus_admission.py`,
`test_p6_dream_compact_boundaries.py`, and `tests/agent/tools/test_self_tool.py`.

### W2 — retire H1 and preserve legacy-data protection

Trace actual imports/callers before each deletion. The expected ownership is:

| Surface | Change |
| --- | --- |
| `nanobot/channels/telegram/gateway_busy.py`, `busy_inbox.py` | Remove H1 composition, control-model dispatch, and channel-specific inbox |
| `nanobot/channels/telegram/runtime.py`, `nanobot/cli/gateway_runtime.py` | Remove H1 registration, polling interception, task startup/shutdown; retain ordinary Telegram and gateway operation |
| `nanobot/agent/busy_control.py`, `busy_dispatch.py`, `busy_steering.py` | Remove H1-only services once callers are removed |
| `nanobot/agent/loop.py`, `runner.py`, `turn.py`, `tools/execution.py` | Remove H1 turn fields/branches/steering; preserve generic source injections, tool dispatch validation, cancellation, and continuations |
| `nanobot/config/schema.py`, relevant config-loading boundary | Retire H1 settings and enforce disabled/enabled compatibility above |
| `nanobot/session/busy_inbox.py`, `persistence_metadata.py`, `manager.py`, `source_store.py`, actual mode-storage callers | Remove unused H1 mutation plumbing; retain only necessary legacy parsing, metadata reservation, recovery-key isolation, and destructive-operation/admission protection |

Do not delete broadly by matching the word `busy`: for example,
`nanobot/gateway/runtime.py` uses `gateway_busy` for unrelated desktop lifecycle
errors. Do not remove the generic SessionManager or register currently dormant
session-message/background-worker tools.

Update H1-only tests under `tests/migration/test_h1_*.py` and
`test_gateway_telegram_busy.py` according to retired behaviour. Preserve useful
ordinary-channel and legacy-data cases rather than deleting whole files by
name. `tests/cli/test_gateway_scheduled_execution.py` combines H1 registration
with scheduled execution; retain the scheduling outcome independently of H1.

Update active instructions in `docs/assistant-prototype-integration.md`,
`docs/configuration.md`, and current status notes. Mark historical H1 design and
qualification records as historical/superseded, preserving their evidence.
Research findings describe the inspected baseline and must not be rewritten as
if the old implementation never existed.

## Validation and acceptance

New or behaviour-changing cases require a fresh **requirements-only** author
session under repository instructions. Give it only the public contract linked
above, a temporary output directory outside the repository, and artifact format;
no parent history, repository access, source, existing tests, logs, diff, or this
implementation plan. The implementation agent may wire the accepted cases into
existing runners, preserving their inputs and expected outcomes.

Run the smallest owning suites after each package. Existing regression anchors
include:

- Canonical direct/bus admission, same-session ordering, waiting cancellation,
  compaction boundaries, and self-tool access restrictions.
- `nanobot/channels/telegram/tests/`, relevant gateway/config tests, and the
  retained ordinary-channel portions of H1 integration coverage.
- `tests/agent/test_process_completion_scope.py`,
  `tests/tools/test_process_completion.py`, and gateway scheduled execution.
- `tests/providers/test_pooled_provider.py::test_concurrent_requests_keep_context_and_credentials_local`
  as a focused provider-isolation check; no live provider requests.
- Independently authored removal/compatibility/legacy-state cases and the
  comparator's focused tests for W0.

Use `.venv/bin/python -m pytest ...` and `.venv/bin/ruff check` for affected
Python scopes; never use `ruff format`. Finish with the full unchanged
`.venv/bin/python -m basedpyright --outputjson`, frozen before/after dependency
and source manifests, and the repository comparator with declared deletion
evidence. Zero introduced signatures across **all** files is mandatory; a lower
total is insufficient. Do not omit files from checking to obtain a pass.

Record strict exit status separately from the diagnostic-ratchet verdict. Any
remaining baseline debt is a stated limitation, never a strict PASS. Closing
P6 or making a production-release claim still requires the full unchanged strict
check to exit zero and the repository's required full-suite/local release gates;
this cleanup does not waive or complete those stages. A WebUI build is unnecessary
unless UI source or its configuration actually changes.

Completion requires: no active H1 model/inbox path or global turn gate; all six
behaviour contracts satisfied; focused checks passing; no new full-source type
diagnostics; updated active docs; and a reviewed diff confined to this scope.
Report deleted modules, any retained legacy guard, exact checks, and limitations.

## Delivery boundary

Prepare reviewable commits on the cleanup branch. Do not merge sibling work,
push, restart the desktop/gateway, enable a feature, migrate live state, send
Telegram messages, or call a real model as part of this task. Before an eventual
deployment, use a copied-state rehearsal to identify any enabled legacy config
or unsettled H1 state and retain a backup/rollback path. That operational cutover
is separate from this implementation handoff.
