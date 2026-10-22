# Harness prototype direction

**Status:** direction accepted in discussion on 2026-10-22; implementation details
remain open. This records agreed outcomes and a source-based gap assessment,
not a new task-context architecture or an implementation specification. This
update changes documentation only.

The consolidated [remaining-work assessment](nanobot-remaining-work.md) combines
this direction with the previous production-compatibility findings.

## Agreed outcomes

- One personal assistant can handle different projects in the same conversation;
  the user should not need to create a new session to switch subjects.
- Use a small repository catalog for discovery. Let the agent choose and read
  relevant instructions, notes and source files through ordinary tools instead
  of mechanically injecting a complete task/project context package.
- A separate semantic task-context module is not an established requirement.
  Reuse existing directory scopes, tool execution, history and context-budget
  machinery; add structure only if concrete failures justify it.
- Work in existing repository locations. Do not require duplicate checkouts or
  a nanobot-specific manifest or directory layout for ordinary repositories.
- Automatically save useful facts and agreed decisions to designated knowledge
  repositories and keep those repositories committed and pushed. This policy
  does not apply automatically to application/code repositories, which retain
  their own Git workflows.
- Keep runtime session state and private artifacts separate from repository
  knowledge. Keep VPN, crawler and marketing workflows in tools, skills, scripts
  or plugins rather than business-specific agent-loop logic.
- Background work means a process/job followed by continuation of the same
  assistant. The runtime retains concrete job identity, directory and ownership;
  no additional model worker or semantic task manager is implied.

## How much is already present

The original nanobot already provides the basic agent loop, filesystem/shell
tools, sessions, skills and memory foundations. The current fork also supplies
context-repository metadata and policy, and the migration includes newer
context-budget and persistence machinery. These are starting points to reuse,
not features to rebuild from scratch.

| Need | Current candidate evidence | Remaining distinction |
| --- | --- | --- |
| Work in existing folders | File tools support resolved paths; shell calls accept a working directory; per-turn workspace scope exists | Verify consistent directory behavior across the enabled tools and roles; repository access does not require a new Git engine |
| Discover project guidance and skills | Project `AGENTS.md`, the assistant's profile files, skill summaries and on-demand skill reading exist | External `library` skill discovery has a known summary-visibility gap |
| Discover repositories | `ContextRepoManager` indexes configured paths and exposes repository summaries | This is a configured index, not yet the agent-maintained generic catalog discussed here; representation and management remain open |
| Build context on demand | The agent can inspect files and tools normally | Configured context entrypoints, supplemental memory and run summaries are currently loaded together; that is broader than catalog-only discovery |
| Keep long conversations bounded | Session history, context governance, provider compaction integration and runtime memory stores exist | Presence is not activation of every compaction or memory feature. Dream/idle activation remains separate, and runtime memory is not automatic project knowledge write-back |
| Save and publish knowledge | File writing, shell Git access and selective Git-sync utilities exist | No normal package runtime caller invokes the sync helper; reliable knowledge routing/save/sync is not established by those utilities alone |

Source map: [context construction](../nanobot/agent/context.py),
[configured repository index](../nanobot/context_repo/manager.py),
[workspace scope](../nanobot/security/workspace_access.py),
[runtime memory](../nanobot/agent/memory.py),
[Git synchronization](../nanobot/utils/git_sync.py), and
[foreground qualification and dormant features](migration/local-runtime-closeout.md).

This assessment favors refining and connecting existing capabilities. It does
not establish that the complete one-conversation, automatic-memory workflow is
already available, and it does not choose a catalog schema, a new service or a
task-state machine.

## Boundaries and remaining discussion

Keep the harness general enough to work with ordinary Git repositories in their
existing locations. The agent should be able to work on repository A, then B,
without copying either into a nanobot-owned workspace or requiring either to
adopt a context-repository layout.

| Responsibility | Boundary |
| --- | --- |
| Agent turns, sessions, context budgets, runtime memory, permissions, job ownership and continuation | Generic harness behavior |
| VPN, crawler, marketing integrations and business workflows | Optional tools, skills, scripts or plugins; their domain logic should not shape the core agent loop |
| Git operations and how to edit a repository | Existing filesystem/shell/Git tools plus instructions or skills; a new Git execution engine is unnecessary for basic use |
| Choosing a project and retaining its location | Generic project/session context, using existing absolute paths; optional aliases can improve convenience |
| Knowledge stored in a repository | Ordinary files usable as context or memory sources; optional discovery/retrieval must not require a marketing-specific manifest or directory layout |

The current candidate still has explicit VPN registrations, a crawler-specific
foreground role, and manifest-aware context-repository policy. These are existing
coupling points to review, not requirements for the future harness. Preserve the
production marketing behavior while moving toward optional integration seams;
this direction does not authorize removing its dependencies in place.

## Option: work with existing repositories through ordinary tools

For basic navigation, reading, editing and Git commands, the inspected original
nanobot already has filesystem tools and shell execution with a working-directory
argument. With access to the relevant paths and existing Git authentication,
the agent can operate on the checkouts already organized on this device.
Appropriate access to those paths is sufficient; unrestricted machine access
is a deployment choice rather than a requirement for repository support.

Using that behavior without a repository-management service is one option. A
small project selection feature is another option to consistently bind relative
paths, prompt context and later continuations to the intended checkout. The
candidate already has [per-turn workspace scope](../nanobot/security/workspace_access.py),
but this alone does not establish that all foreground roles and tools support
switching projects consistently. A shell `cd` also does not change the default
root of separately constructed filesystem tools or later shell invocations.

Keep runtime sessions, receipts, private artifacts and credentials separate from
the selected project checkout. Do not implement project switching by relocating
runtime state into every repository. Existing `nanobot.context.json` support
can remain a compatibility adapter for the marketing deployment; a manifest v2,
special package service or required repository schema is not the default next
milestone. Git commit/push policy can live in instructions or an optional tool;
machine access alone does not establish automatic, reliable synchronization.

## Background work means process completion and assistant continuation

The clarified requirement is one assistant launching a process or scheduling
work, returning control to the conversation, and continuing when a completion or
due-time event arrives. No second LLM session is required while waiting.

The current candidate can run a process through exec sessions, but its process
manager has no automatic completion-to-assistant-turn connection. The generic
integration should retain job identity, task instructions or evidence references,
original project path, owning session and reply destination, then wake or queue
a continuation through the canonical runner. A completion for project A must
retain A's context even if the user has moved the conversation to B.

The harness owns process status, cancellation, completion delivery and recovery.
Tools and skills own the actual marketing, crawling, VPN or Git steps. This
pattern is separate from background subagent delegation and does not require
reversing H1's single-assistant choice. See the corrected
[background-work review](customization-feature-inventory.md#background-work-and-conversational-responsiveness).

## Migration priority and evidence

Keep the existing marketing service running while the candidate is qualified.
The [scheduling compatibility plan](marketing-agent-scheduling-plan.md) and
[customization inventory](customization-feature-inventory.md) describe the
current blockers. Configuration parsing alone does not establish a drop-in
replacement: normal gateway scheduling is inactive, and required marketing job
contracts, quota rescheduling and automatic repository-sync integration have
gaps. Workspace and background integration should be bounded follow-ups rather
than assumptions that these compatibility requirements already pass.

## Earlier proposal (historical; not an accepted plan)

The following records an earlier discussion. Its drop-in conclusion and two-small-
ports estimate are superseded by the customization inventory and scheduling
compatibility evidence. Its package/manifest services and repository milestone
order are unaccepted proposals. Historical test or production-audit claims below
are not new verification by this discussion update.

## 1. Where the project actually stands

Three things are now established rather than assumed:

1. **Production is `main@0976742`** (marketing agent, 4 cron jobs, Codex provider),
   running clean and in sync with GitHub since 2026-10-22.
2. **The candidate is a separate history**, not a fork: no merge base, 1,827
   files that exist only there. It carries the harness work (busy-input ledger,
   session modes, context governance, durable receipts) and passed its own
   local validation (3,637 tests, strict checker at zero).
3. **The candidate can consume the production config unchanged.** Verified by
   loading a redacted copy of the live config through the candidate loader:
   model, provider, `maxTokens` object, `contextRepos`, Telegram authorization,
   `fastCommands`, `taskUpdateMode`, `exec` settings and the heartbeat section
   all parse correctly, and a `model_dump` → `model_validate` round-trip
   preserves them.

Point 3 is the important one. It means the candidate is not a research artifact
that needs a new operating model; it is a drop-in for the instance we already run.

One hazard found while testing: the round-trip **rewrites** the legacy
`maxTokens: {input, output}` into `maxTokens: 20000` + `contextWindowTokens:
200000`. The production branch expects the object form. So the candidate must
run against a *copy* of the config, and adopting it is a config-format migration,
not a no-op.

## 2. Decision: the candidate is the base, `main` becomes reference

The alternative — starting the prototype from `main` and porting the harness work
onto it — would mean re-implementing the busy-input ledger, session modes and
context governance on top of code that has none of their assumptions. The
candidate already has them, plus a validated config adapter.

What the candidate genuinely lacks, with cost estimates from the production audit:

| Gap | Real impact | Cost to close |
| --- | --- | --- |
| Cron 5-hour rate-limit reschedule | Four live jobs; never fired yet | Small — one function plus store fields |
| Heartbeat service | Runs every 30 min; no tasks today | Small — port service, or delete the config key |
| `spawn_pipeline` / workflow state | Not used by any job (0 log hits) | None — retirement stands |
| Data migration (sessions, memory receipts) | Unknown until tried on real data | Medium — needs a copy of production state |

So the base decision costs two small ports and one real migration test. The
reverse decision costs months. **Use the candidate; keep `main` as the behavioral
reference and the production runtime.**

## 3. Reframing: the context repo is a capability package, not memory

The production audit changes this argument. The user-facing idea was "make the
context repo part of the harness memory". The data says a context repo is
something else:

| Surface | Size in production | What it really is |
| --- | --- | --- |
| `nanobot.context.json` + modules + skills + tools | ~180 KB of contracts | **Capability package** — defines what the agent can do |
| `stores/*.json` + `outputs/**` | ~13 MB and growing | **Knowledge** — what the agent has learned and produced |
| `memory/MEMORY.md` | 408 bytes, unused | Vestigial |
| `runs/summaries.md` | 89 bytes | Vestigial |

`MEMORY.md` is not where the knowledge lives. The durable knowledge is in the
pipeline stores and in 283 output artifacts — and today none of it is retrieved;
it is either dumped whole into the prompt (the 15 KB `REPO_CONTEXT.md`) or found
by reading files on demand.

So the design splits into two services that are currently one config key:

**A. Package service** — what to load and how it may be used.
- registry with pinned commit, enable/disable, health check (`doctor`)
- manifest v2: `version`, `provides`, policy globs, sync rules, compatibility
- validation at startup: missing entrypoints, dangling store schemas, unresolvable
  target paths, protected/writable overlaps

**B. Knowledge service** — what to recall and how to cite it.
- index over `outputs/**`, `stores/*.json`, `feedback/*.md`
- retrieval on demand instead of injection: rank, summarise, cite the source path
- budgets from the existing context window, not a new knob
- provenance so `read_provenance` can answer "where did this come from"

The agent-controlled part belongs to A (install, inspect, repair a package) and
to B (ask a question of its own history). Neither requires giving the model raw
write access to the package definition: proposals plus human approval match the
`proposalRequired` policy already in the manifest.

## 4. Sequence

**Milestone 1 — one job runs on the candidate (2–3 days).**
Copy production workspace + config to a sandbox directory, run one real job
(daily content idea) through the candidate with the real provider, and compare
artifacts to what production produced that morning. Success means: same tools
resolve, context package loads, output lands in `outputs/`, sync commits it.
This is the cheapest possible proof that the harness is a drop-in, and it
exercises sessions, memory receipts, tool dispatch and context repos together.

**Milestone 2 — close the two behavioral gaps.**
Port the cron reschedule; decide heartbeat (port or delete the key). Gate: the
four production jobs are recognized and scheduled by the candidate's cron store
without rewriting it.

**Milestone 3 — package service.**
Registry + manifest v2 + `doctor`, keeping literal `path` entries working so
existing deployments do not change.

**Milestone 4 — knowledge service.**
Index and retrieval with citations, measured against the current
"dump `REPO_CONTEXT.md`" baseline.

Each milestone ships behind a config flag on the prototype; production keeps
running `main` until the whole flow is boring.

## 5. Open questions

1. Does Milestone 1 run against a **copy of the real client workspace**, or a
   synthetic package? The copy is more convincing but brings client data into the
   prototype environment.
2. Does the prototype keep the same **channel** (Telegram) or start on WebUI/CLI
   so no client can reach it while it is experimental?
3. Is the **config format change** (`maxTokens` object → scalar + context window)
   acceptable as part of adoption, or should the candidate keep writing the
   legacy shape?
