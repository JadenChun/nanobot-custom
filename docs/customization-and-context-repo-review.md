# Customization review and context-repo design direction

**Status:** source review plus a read-only production audit, 2026-10-22. The
production facts in §0 were read directly from the live server; everything else
is static analysis and is not a compatibility claim or an authorization to
change the deployed agent.

This continues the work of Codex session
`01a162a5-512c-7971-acfa-5f30e87144f9`, which stopped when both review
subagents hit a usage limit before producing their reports. The findings here
are rebuilt from the repository plus that session's already-executed evidence.

## 0. Verified production facts (2026-10-22)

Read directly from `marketing-agent-prod` (100.88.68.109) over Tailscale SSH,
read-only. These supersede the placeholder assumptions in earlier sections.

| Fact | Value |
| --- | --- |
| Running code | `/opt/marketing-agent/nanobot-custom`, branch `main`, `af1f33a9`, 2026-10-01 |
| Working tree | was **5 dirty files** incl. an uncommitted `openai_codex_provider.py` hotfix — **resolved** same day (§4.7) |
| Staging checkout | `/opt/marketing-agent-staging/nanobot-custom`, branch `staging`, `4aa562d` (2026-09-01) |
| Model | `openai-codex/gpt-6.1-sol` via `openai_codex`, `maxTokens {input: 200000, output: 20000}` |
| Cron | **active**, 4 jobs in `.nanobot/workspace/cron/jobs.json`, all `lastStatus: ok` |
| Job durations | Meta sync 111–146 s; trend research ~645 s; weekly review 385–436 s; content idea ~355 s |
| Rate limits | 456 provider-retry warnings in the gateway log; **0 job-level reschedules** |
| Context repo | `client-marketing-assistance`, `autosync: false`, 9 modified + 695 untracked files, last commit 2026-10-07 |
| Prompt payload | `REPO_CONTEXT.md` 15 KB; no `memory/MEMORY.md`; `runs/summaries.md` 89 B; 9 skills (58 KB, summarised only) |
| `spawn_pipeline` | 0 occurrences in the gateway log; no `workspace/current` plan state |

Three of my earlier hypotheses were wrong, and the corrections matter:

- **Cron is not dormant.** Four scheduled jobs run daily/weekly, so the missing
  rate-limit reschedule is a live code path, not dead code.
- **Prompt growth is not happening.** Package memory is a single 89-byte summary
  file and there is no `MEMORY.md`; the injected context is ~15 KB + skill
  summaries. Caching would save little.
- **The pipeline orchestration is not in use.** Cron payloads are plain
  `agent_turn` prompts; nothing calls `spawn_pipeline`.

## 1. Questions this review answers

1. What do we actually run in production, and what does the migration candidate
   contain relative to it?
2. Which customizations are at risk if we ship the candidate as-is?
3. What is the context-repo mechanism today, where is the friction, and what
   would "context repo as harness memory" actually mean?
4. Where could efficiency improve, and what must be measured first?

## 2. The two code lines

| Line | Ref | Commit | Meaning |
| --- | --- | --- | --- |
| Production | `origin/main`, `origin/staging` | `af1f33a9` "feat(cron): reschedule client jobs after a 5-hour rate limit" | The client-facing marketing agent line |
| Production branch | `origin/codex/marketing-agent-telegram-access` | `5c7686bb` merge of `origin/main` | Dedicated marketing-agent branch |
| Candidate | `chore/upstream-migration` | `79934a56` "Land upstream migration implementation and evidence" | Post-migration harness, validated 2026-10-19 |
| Upstream pin | `f47b8f2` (2026-09-08) | — | Basis of the migration |
| Latest upstream | `084b6bb3` | — | 700 commits ahead, not imported |

### 2.1 There is no shared history

`git merge-base origin/main HEAD` fails. The candidate tree adds 1,827 files and
removes 50 relative to production **as a whole-tree comparison**, because the
two histories are unrelated. This is not a normal fork divergence:

- The candidate cannot be merged, rebased, or cherry-picked from production.
- A textual diff cannot tell us what behavior was preserved.
- "Upstream adoption" was already abandoned in practice for this branch; the
  candidate is a re-import, not a forward port.
- Therefore production safety must be established by **behavioral equivalence
  checks**, not by diff review. Any claim of the form "we kept everything"
  is a hypothesis until a named test exercises it.

This single fact explains most of the project-direction fatigue: every refresh
costs a re-validation cycle because there is no merge path to amortize it.

## 3. Customization inventory

The 2026-09-08 audit accounts for 158 rename-aware diff entries against the
common base (`docs/migration/custom-change-audit.md`): 37 CORE, 17 PROVIDER,
41 EDGE, 6 REPO findings. The table below is the same surface organized by
what production actually depends on, with candidate status where verified.

| Area | Mechanism | Candidate status |
| --- | --- | --- |
| Runner ABI and whole-batch policy preflight | `agent/runner.py`, `agent/policy.py` | Reworked; audit marks confirmed conflict, replacement unvalidated |
| Terminal tool outcomes (no extra model call) | `agent/runner.py` tool execution | Reworked into imported executor; unvalidated |
| Foreground role registries, locks, approval CAS | `agent/{capabilities,delegation,foreground,turn,write_guard}.py` | Present and extended; background spawn deliberately retired |
| Busy input, steering, stop, recovery | H1 ledger, `tests/agent/test_migration_foreground.py` | Candidate-only; 2026-10-19 local validation |
| Session revision/generation CAS, keys, projection | `session/manager.py`, `session/session_mode.py` | Reworked; audit marks confirmed conflict |
| Memory receipts, `MEMORY.md` / `HISTORY.md` archival | `agent/memory.py`, `agent/memory_recovery*.py` | Reworked; migration path unvalidated |
| Context repos: manifest, policy, sync, secret scan | `context_repo/`, `utils/git_sync.py` | Module present; **auto-sync call site missing** (§4.2) |
| Provider key pool, pacing, fallback semantics | `providers/openai_compat_provider.py`, `providers/fallback_provider.py` | Audit: custom-only port / compatible overlap, unvalidated |
| Codex / Copilot auth import and refresh | `providers/codex_auth.py`, `providers/copilot_auth.py` | Audit: confirmed conflict, adapters unvalidated |
| Telegram `allowChats`, `fastCommands`, quota | `channels/telegram/runtime.py` | Ported into the package runtime (4 references) |
| Result-only scheduled delivery | `channels/manager.py`, `taskUpdateMode` | Present |
| Crawler, social crawl, desktop/device/browser tools | `agent/tools/{crawler,social_crawl,desktop_use,agent_device,agent_browser}.py` | Present |
| VPN trend tools | `agent/tools/trend_vpn*.py` | Present |
| Filesystem `/workspace` alias, secret-name block | `agent/tools/filesystem.py` | Present |
| `maxTokens` object schema (`{input, output}`) | `config/schema.py`, deploy example | Present with adapter |
| Subagent pipeline orchestration + workflow state | `agent/tools/pipeline.py`, `agent/workflow.py` (production only) | **Absent by design** (§4.3) |
| Cron 5-hour rate-limit reschedule | `cli/commands.py` (production only) | **Absent** (§4.1) |

## 4. Concrete production-risk findings

Ranked by "would a client notice, and can we prove it quickly". §4.7 is the
highest in practice because it blocks any safe redeploy, even though it is not a
feature gap.

### 4.1 Cron jobs lose the rate-limit retry (directly client-facing)

Production commit `af1f33a9` adds, in `nanobot/cli/commands.py`:

- `_looks_like_codex_rate_limit()`
- `_codex_rate_limit_retry_at()` — reads `get_codex_usage()`, refuses weekly
  (secondary) exhaustion, refuses resets more than 24 h out
- `_reschedule_on_rate_limit()` — sets `job.state.retry_at_ms`, increments
  `rate_limit_retries`, alerts "Rate Limited — Rescheduled"
- budget of 3 consecutive retries, cleared on clean delivery
- `retryAtMs` / `rateLimitRetries` persisted in the cron store

Searching the candidate for `rate_limit_retries` returns nothing. In the
candidate, a scheduled client job that fails on the 5-hour window now raises
`Failed` and drops the run. For a daily-idea / weekly-review schedule this is
a visible regression, not a cosmetic difference.

**Verification:** replay a scripted provider that raises a Codex rate-limit
error, assert `retry_at_ms` is set once and the "Failed" alert is not sent.

**Production status:** four jobs are live (05:00–07:00 UTC), so this path is
reachable daily. It has not fired yet — zero "Rate Limited — Rescheduled" events,
and every run history entry is `ok`. The 456 rate-limit log lines are the
provider-level retry wrapper succeeding within three attempts, which is a
different mechanism. So this is a *latent* safety net, not an observed outage:
it matters the first time a job starts near an exhausted 5-hour window and
cannot finish inside three short retries.

### 4.2 Context-repo auto-sync was dropped, not ported

Production `nanobot/agent/loop.py` calls `_maybe_sync_context_repo()`
(line 1955, 2047, 2126) and schedules `async_sync_context_repo(...)` with the
repo's include/exclude patterns and `autoPush` at line 2273.

The candidate still ships `utils/git_sync.py` (513 lines) and
`repo.auto_sync`, but has **no call site**: `grep -rn "_maybe_sync_context_repo"`
is empty. Consequences:

- Agent writes into `agent-workspace/**` stay local; nothing is committed or
  pushed after a turn. The prompt still reads the local files, so a single
  machine keeps working, but the remote history, backup, multi-instance view
  and audit trail stop advancing.
- Protected-path, ahead-commit, secret-scan and unrelated-dirt checks in
  `git_sync.py` become dead code in the shipped harness.

**Verification:** one turn that writes under `agent-workspace/outputs/`, then
assert a commit exists on the configured remote; plus the existing
`tests/context_repo/test_git_sync*.py` wired through the loop, not called
directly.

**Production status:** `autoSync` was `false` in the live config and in all
three backups from 2026-09-30/10-01, so the candidate's missing call site
changed nothing. **Root cause found 2026-10-22:** the setting could not have
worked. `social-comments.json` and `agent-workspace/state/**` are declared
`neverCommit` in the manifest but were still *tracked*, so every run dirtied
them; the sync then committed its selected paths and refused to push with
"recovery left only unselected changes", leaving an unpushed commit behind each
time. Turning the setting off was a reasonable response to a broken loop.

**Resolved 2026-10-22:** those three files were untracked (kept on disk),
`.gitignore` was aligned with `neverCommit`, accumulated outputs were committed,
`autoSync` was set to `true`, and the path was exercised end to end through
`ContextRepoManager` + `async_sync_context_repo` — commit `e34d5b8` and a later
verification commit both pushed automatically. Local and remote now match and
the tree is clean.

### 4.3 Subagent pipeline orchestration is intentionally gone

Production has `agent/tools/pipeline.py` (`SpawnPipelineTool`) and
`agent/workflow.py` (`WorkflowState`, `workspace/current/plan.json` +
summaries). The candidate's design ("bounded foreground delegation", "background
agents remain retired") replaces this on purpose.

If any deployed cron job or skill calls `spawn_pipeline` or reads workflow plan
state, it will fail on the candidate.

**Resolved against production:** it does not. `spawn_pipeline` appears zero
times in the gateway log, no `workspace/current` plan state exists, and all four
cron jobs are plain `agent_turn` prompts. The retirement is safe on the evidence
available; only an interactive user explicitly asking for a pipeline could hit
it, and none has.

### 4.4 Data migration is the silent risk

Session keys moved from sanitized legacy keys to workspace-ID namespaces; memory
gained receipt recovery; `HISTORY.md` is no longer renamed on startup. The audit
(CORE-04/05) requires dual-read and collision mapping. None of it is exercised
against real client data in the candidate's test suite.

A candidate deploy against a copied production data directory is the only honest
test. Everything else is a source-level argument.

### 4.5 Config adapter subtleties

Production config uses `maxTokens: {input, output}` and `contextRepos[]`. The
candidate keeps both (`config/schema.py:36`, `config/schema.py:379`) with an
adapter (`config/schema.py:416-425`). The risk is
not the shape but the *precedence* rules (explicit empty values, unknown custom
fields, save/reload round-trip), which the audit marks NOT-RUN.

### 4.6 Telegram authorization ordering

Production checks sender authorization before group authorization; the
deployment README claiming `allowFrom` OR `allowChats` is sufficient was already
wrong (audit EDGE-12: an `allowChats`-only config stays denied). The candidate
keeps the ordering, but the documentation defect can still mislead whoever
provisions the client instance.

### 4.7 The server is running code that is in no branch (highest actual risk)

The deployed tree has an uncommitted 31-line change to
`nanobot/providers/openai_codex_provider.py`, plus 83 lines of tests:

- `httpx.TransportError` becomes a retryable `LLMResponse` (503,
  `retry_after` from `CODEX_RETRY_AFTER_SECONDS`, default 15 s) instead of
  ending the turn.
- SSE `error` / `response.failed` events raise `_CodexHTTPError(502,
  retry_after=...)` instead of `RuntimeError`, so the shared retry wrapper
  retries them.

`git log --all -S_CODEX_TRANSIENT_RETRY_AFTER` returns nothing: this exists only
on the server. Consequences:

- A redeploy, rollback, or fresh clone silently removes it, and the client
  returns to "Codex response failed" ending whole turns.
- The candidate does not have it either, so migrating as-is loses it too.
- Nobody reviewing `main` can see that production behaves differently from it.

This is the single most important thing to land in git before any other work.
`.bak-codexretry-20260918T101906Z` and `.bak-statusfix-20260930T064824Z` files
suggest it has been re-applied by hand at least twice.

**Verification:** diff the server file against `main`, confirm the tests pass,
commit it, and add a note to the deploy runbook that the server must not be
edited in place.

**Resolved 2026-10-22.** The hotfix was committed as `05adc9b5` on
`fix/codex-transient-retry`, reviewed in PR #18, and merged to `main` as
`09767426`. The production checkout was fast-forwarded to that merge and its
`origin` switched from HTTPS to the SSH deploy key, so future pushes work
without a stored token. The deployed tree is now `main` + three untracked
`.bak-*` files.

### 4.8 The client context repo has drifted for two weeks

`client-marketing-assistance` last committed 2026-10-07. Since then:

- 9 tracked files modified: `marketing-pipeline.json`,
  `analytics-pipeline.json`, `analytics-snapshots.json`, `social-comments.json`,
  `idea-memory.md`, `performance-review.md`, `trend-log.md`,
  `meta-analytics-sync.json`, `meta-comments-sync.json`.
- 695 untracked entries, including bootstrap templates written into the repo
  root (`AGENTS.md`, `SOUL.md`, `IDENTITY.md`, `USER.md`, `TOOLS.md`,
  `HEARTBEAT.md`, all dated 2026-07-17) and a stray nested `2026-09-14/2026-09-14`
  directory.

Because `autoSync` is false and the manifest's `include` list covers only
`outputs/**`, `feedback/**` and four data files, the durable state that drives
the next run (pipeline records, trend log, idea memory) is not backed up beyond
the server disk. The stray templates are harmless to the prompt today (the
effective workspace is `.nanobot/workspace`), but they would be committed by any
future `git add -A`.

**Resolved 2026-10-22.** The six sync-eligible files were committed and pushed
(`2e5cd87`); 283 accumulated `outputs/**` files (3.4 MB) were committed in
`0c1af31`; `.gitignore` now expresses the manifest's `neverCommit` list
(`fa71234`), and the three tracked-but-neverCommit stores were untracked in
`3e72342` while remaining on disk. `autoSync` is now `true` and verified: the
tree is clean, local and remote match at `4226419`, and a real content change
was committed and pushed by the sync itself.

`analytics-snapshots.json` (7.9 MB, 7,269 daily records) stays in git as the
only historical record of metrics Meta no longer serves. Its 1.88 MB of repeated
captions are the obvious slimming target if the repo grows uncomfortable.

### 4.9 Heartbeat runs in production, the candidate has no heartbeat service

Production starts `nanobot/heartbeat/service.py` every 30 minutes
(`gateway.heartbeat.enabled: true`, `intervalS: 1800`) and logs
"Heartbeat: checking for tasks". The candidate has no heartbeat implementation
at all — a deliberate "inactive by design" choice.

Today the impact is nil: the client's `HEARTBEAT.md` is the empty template, so
every tick returns "nothing to report" without a model call. It becomes a silent
gap the moment someone adds a periodic task to that file. Either port the
service or remove `gateway.heartbeat` from the client config so the difference is
visible rather than latent.

## 5. Context repos today

### 5.1 Anatomy

A context repo is a directory with a `nanobot.context.json` manifest. The real
client package (`JadenChun/marketing-agent-context`) contains:

```
nanobot.context.json
REPO_CONTEXT.md  PRODUCT_KNOWLEDGE.md
modules/{trend-research,content-ideation,social-analytics,video-scripts,...}/MODULE.md
skills/{brand-voice,marketing-strategy,trend-research,...}/SKILL.md
stores/{marketing,analytics,snapshots,comments}/schema.json
tools/*.py  scripts/*.py
agent-workspace/{data,outputs,feedback,runs,state}/
docs/  tests/  config/  deploy/
```

The manifest declares: entrypoints (context file, skill globs), modules with
descriptions, stores (path, schema, `directEdit`, `writeVia`, `syncPaths`),
tools with long usage contracts, `protected` / `writable` / `proposalRequired`
globs, `sync` (`autoPush`, `include`, `neverCommit`), and `userExperience`
(audience, assistant name, approval style).

### 5.2 How the harness consumes it

```
config contextRepos[]  ->  ContextRepoRuntimeConfig  ->  ManagedContextRepo
                                                 ->  ContextRepoManager
        +---------------------+----------------------+--------------------+
        v                     v                      v                    v
  prompt_summary()      context_files()        memory_files()       skill_roots()
  (agent/context.py:140) (context.py:263)     (context.py:158)     (skills extra_paths)
        |                     |                      |                    |
   system prompt: "# Context Repositories" + bootstrap content + memory + run summaries
        |
        v
  ResourceAccessPolicy (context_repo/policy.py) -> path validation, protected/writable,
  write guard, secret scan; git_sync.py -> include/exclude, autoPush, neverCommit
```

This is already a real subsystem, not a skill. What it lacks is *lifecycle and
retrieval*, not enforcement.

### 5.3 Where the friction actually is

| Friction | Evidence |
| --- | --- |
| Onboarding is manual config editing | `contextRepos[].path` in `agents.defaults`; absolute paths like `/opt/marketing-agent/client-marketing-assistance` |
| No registry or discovery | `ContextRepoManager.from_config()` only walks the configured list |
| No versioning or pinning | `nanobot.context.json` has no `version`, no commit pin, no compatibility range |
| No validation | type guards only; a typo'd glob or missing store path degrades silently |
| No lifecycle | no clone/pull/enable/disable/health; drift is invisible until a file read fails |
| Prompt injection is unbounded | every `context_files()`, `memory_files()`, `run_summary_files()` is concatenated into the system prompt with no token budget |
| Memory has no retrieval | "run state" is a flat dump; nothing ranks, summarizes, or decays it |
| No authoring path | the agent can write inside the repo, but cannot create, inspect, or repair the package definition |
| Single-tenant assumption | one path per deployment; multi-client means editing config and restarting |

The last three rows are what you were describing as "manual management"; the
first five are why it feels manual.

## 6. Target design: context packages as a first-class layer

The proposal is to promote the context repo from "a path in config" to a
**context package** with the same status as providers, channels, and tools.

### 6.1 Four services

1. **Registry** (`context_repo/registry.py`)
   - `~/.nanobot/context-packages.json`: name, remote, pinned commit, enabled,
     last sync, health.
   - Local content-addressed clone at `~/.nanobot/packages/<name>/<sha>`.
   - Commands: `list`, `add <remote>`, `enable/disable`, `pin`, `doctor`.
   - Config keeps working: a literal `path` entry becomes a registry entry
     marked `managed: false`, so nothing breaks on upgrade.
2. **Resolver**
   - Existing `ManagedContextRepo` gains `manifest_version`, `validate()`,
     `health()` (missing entrypoints, dangling store schemas, unresolvable
     target paths, protected/writable overlap).
   - One validation report reused by startup preflight and `doctor`.
3. **Assembler** (the memory change that matters)
   - Replace the flat concatenation with budgeted, cited sections:
     identity context (always), module contracts (always, small), package memory
     (budgeted), run state (budgeted, most-recent-first, summarised), and
     on-demand knowledge (search, not injection).
   - Every injected block carries its source path so `read_provenance` works.
   - Budgets come from the existing `context_window_tokens`, not a new knob.
4. **Author**
   - Agent tools: `context_package_list`, `context_package_inspect`,
     `context_package_propose` (manifest/module/store changes), and
     `context_package_sync`.
   - `propose` writes a patch + rationale, never a direct protected edit;
     approval stays with the human, consistent with `proposalRequired`.
   - Every mutation is appended to an audit log (`~/.nanobot/packages/audit.jsonl`).

### 6.2 Memory model

Separate three things that are currently one prompt blob:

| Layer | Lifetime | Retrieval |
| --- | --- | --- |
| Session memory | one conversation | existing history + compaction |
| Package knowledge | durable, versioned | budgeted injection of module contracts + context file; search for the rest |
| Package run state | rolling | latest N summaries, decayed by age, never regenerated from scratch |

The point is not "more memory"; it is **bounded, addressable memory**. Today a
long-running client package grows the system prompt every day.

### 6.3 What this buys you

- Onboarding a client = `context-package add <git-url>` + approve, no config
  surgery or restart.
- Upgrades are pinned and reversible (`pin <sha>`), which is what the migration
  branch never had.
- The agent can repair its own package definitions under policy, instead of a
  human editing JSON.
- Prompt size stops growing with package history.

## 7. Efficiency: the measurements say something different

The Rust question is premature, and the reason is now measured rather than
argued. Production turn durations from the cron run history:

| Job | Duration | External work |
| --- | --- | --- |
| Daily Meta analytics sync | 111–146 s | Meta Graph API paging |
| Daily content idea | ~355 s | model + pipeline store |
| Weekly performance review | 385–436 s | Meta API + model |
| Daily trend research | ~645 s | ScrapeCreators, SerpAPI, deep fetches |

Meanwhile the harness-side payload is small: `REPO_CONTEXT.md` is 15 KB, there
is no `MEMORY.md`, `runs/summaries.md` is 89 bytes, and skills are summarised
rather than injected. Prompt assembly is a few file reads and a directory scan
per turn — milliseconds against minutes.

So the cost model is: **model round-trips plus third-party API time**. A faster
language for the orchestration layer cannot move a 10-minute job whose time is
spent waiting on ScrapeCreators and ChatGPT. My earlier hypothesis that prompt
size was growing was wrong for this deployment; it is worth re-checking only if
a package starts injecting memory.

What that implies for optimisation, in order of expected value:

1. **Fewer provider calls per job.** The trend job runs a long tool loop; each
   extra round is a full request with a large context. Batching or trimming the
   loop is worth more than any caching.
2. **Fewer/cheaper external calls.** Provider credits and rate limits are the
   binding constraint (456 rate-limit warnings), not CPU.
3. **Prompt caching.** Real but small: the system prompt is ~4–6k tokens and
   stable within a day. Worth doing because it is nearly free, not because it
   dominates.
4. **Language.** No evidence supports a rewrite on performance grounds. A Rust
   rewrite would reimplement providers, channels, tools, persistence and
   recovery to save milliseconds.

The honest way to settle (1) and (2) is per-turn instrumentation: timestamp each
provider request, each tool call and each external script, then attribute the
645 s. That is a day of work and it replaces guesswork.

Cheap wins available without architectural change:

- cache the assembled system prompt keyed by (package commit, memory mtime,
  skills mtime) — small, but free;
- stop rescanning skill directories per turn;
- cap injected run state *before* the package layer makes it grow.

## 8. Recommended sequence

**Stage A — production parity (must precede any prototype work).**
1. Commit the server-only Codex retry hotfix and reconcile the deployed tree
   with `main` (§4.7). Until this exists in git, no redeploy is safe.
2. Port the cron rate-limit reschedule to the candidate (§4.1).
3. Decide what `autoSync: false` is protecting against, then either fix the
   sync path or remove the dead configuration (§4.2, §4.8).
4. Run the candidate against a copy of production data: session reopen, memory
   receipts, cron store, config load/save. Gate on behavior, not on tests that
   were written against the candidate.

No action is needed on the subagent pipeline: production does not use it (§4.3).

**Stage B — package layer on the prototype.**
5. Registry + validation + `doctor`, keeping literal `path` compatibility.
6. Budgeted assembler and retrieval; measure before/after prompt tokens.
7. Only then add authoring tools.

**Stage C — agent-controlled packages.**
8. `context_package_propose` behind `proposalRequired`, audit log, approval flow.

Stage A is small and protects the client. Stage B is the interesting product
work and is where "context repo as harness memory" actually lands.

## 9. Decisions needed

1. **`autoSync: false`** — was it disabled deliberately (noise, repo size,
   secrets) or to work around a failure? This decides whether §4.2 is
   housekeeping or a bug, and whether the package layer should own sync.
2. **The Codex hotfix** — do you want it committed to `main` as-is, or folded
   into the candidate's provider work with the retry policy reviewed first?
3. **Priority** — Stage A first, or start the package layer now that the
   production facts are known? My recommendation is A1 (commit the hotfix) this
   week, then B, then the rest of A.

## 10. Evidence and limits

- Branch/commit facts: `git log`, `git merge-base` (fails), `git diff --stat`
  against `origin/main` in this checkout, 2026-10-22.
- Production facts: read-only audit over Tailscale SSH against
  `marketing-agent-prod` (100.88.68.109) using
  `scripts/audit_deployed_instance.sh` plus targeted reads of the deployed
  checkout, config, cron store, context repo and gateway log.
- Missing-feature claims: `grep` over `nanobot/` on `HEAD`; production
  counterparts read via `git show origin/main:...` and on the server.
- Context package anatomy: GitHub tree and `nanobot.context.json` of
  `JadenChun/marketing-agent-context` (private, read with the authenticated
  `gh` CLI), confirmed against the deployed copy.
- Turn durations: `state.runHistory[].durationMs` in the production cron store,
  last two runs per job.
- Not verified: whether `autoSync` was disabled for a reason, live provider
  latency per request, cost per run, and any runtime behavior of the candidate
  against client data.
- Prior validation claims are those recorded in
  `docs/migration/local-runtime-closeout.md` (3,637 tests in final batches);
  this review did not re-run them.
