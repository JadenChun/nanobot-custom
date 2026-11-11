# Feature implementation audit — 2026-11-05

The first implementation pass found **nine actionable feature gaps**: six P1
workflow/integration issues and three P2 usability/completeness issues. The most
consistent problem is a mismatch between an advertised feature and the
capability installed in the normal runtime. Several underlying components are
deliberately fenced during migration; removing those fences is not a repair.

This report records the initial audit and subsequent bounded repairs. Important
workflows and live platform integrations still need qualification; there is no
clean-code or release PASS. The initial inspection was read-only; authorized
repairs are recorded below. Original observations, line references, and
validation results remain historical evidence for the baseline revision.

This report precedes the module/ownership work in the
[code-quality audit charter](./code-quality-audit-2026-11-04.md).

## Repair status — batches 1–3 and macOS follow-up

The authorized media, skills/clipboard, and runtime integration batches are
implemented on `codex/feature-audit-repairs`. See the
[repair results and evidence](./audits/feature-implementation-2026-11-05/repair-results.md)
for exact validation, supported contracts, and remaining limits.
The subsequent [macOS reliability repair](./audits/feature-implementation-2026-11-05/macos-repair-results.md)
addresses F-005 and the retained backend failures, with independent review.

| Finding | Current disposition |
|---|---|
| F-001 | Repaired for the installed fresh-turn/Telegram scheduler: owned manual admission, effective UI capability, next/latest outcome and delivery status. Session-bound and WebSocket scheduled execution remain unavailable. |
| F-002 | Repaired with explicit `gateway.localTriggers` opt-in, one workspace consumer lease, queued delivery, busy-chat coordination, restart recovery and shutdown. Default inactive mode retains messages. |
| F-003 | Repaired in normal tool construction. `my` retains its config/read-only gates; goal tools retain explicit `/goal` authorization and reject nonpersistent requests. |
| F-004 | Availability claims repaired. Settings, commands and guides report periodic Heartbeat/Dream as inactive; manual `/dream` remains available. Periodic execution was not implemented. |
| F-005 | Repaired on macOS using kernel peer credentials; real IPC, pairing, revocation, restart and dependent workflows pass. Linux credential decoding is tested with controlled responses; real Linux qualification remains open. |
| F-006 | Repaired with generation-scoped requests and synchronous duplicate-page protection. |
| F-007 | Repaired with bounded cross-session continuation and explicit page/completeness metadata. |
| F-008 | Repaired with loading/error/retry states, retained same-auth data, and credential-scoped mutation events. |
| F-009 | Repaired with native mixed-paste text insertion and attachment handling; DOM tests cover selection/caret behavior. Native browser/OS clipboard qualification remains open. |

Batches 1–3 passed **1,588 affected backend tests** and **1,468 frontend tests**;
frontend build and lint pass. The macOS follow-up passed the default full
backend suite: **17,001 passed, 75 skipped, 42 warnings, 6 subtests passed**.
The original 22 retained failures now yield **15 passed and 7 Linux-only
skips**. A final peer-test portability adjustment followed the full run and
passed focused checks; all runtime inputs, other sources, assets and packages
match the passing run. The full suite was not repeated after that fixture-only
adjustment. See the follow-up for exact timing, skip and environment evidence.

Full-source BasedPyright comparison introduces **zero diagnostic signatures**,
with the same **1,001 existing errors**. These results do not establish a clean
codebase, live platform qualification or migration P6 completion. The delivery
branch is `codex/feature-audit-repairs`; evidence was captured before commit
packaging.

## Baseline and method

| Item | Recorded value |
|---|---|
| Source revision | `364259daed2c11700d4064e7ca58743fd037b0b0` on `staging` |
| Working tree at start | Clean |
| Host | Darwin 27.0.0, arm64 |
| Runtimes | Python 3.13.2, pytest 9.1.1, Bun 1.3.10 |
| Dependencies | Existing environment; no dependencies installed or lockfiles changed |
| Method | Compare public docs/config/UI with registration, gateway composition, state projections, and failure handling; run existing tests and targeted behavioral probes |
| Evidence | [Commands, results, source snapshot, and reproducible probes](./audits/feature-implementation-2026-11-05/README.md) |

The source snapshot covers tracked files in `nanobot`, `webui/src`, and
`scripts`. It identifies what was tested; it does not claim every file was
reviewed. Probe failures below are new exploratory expectations, separate from
failures in the existing test suite.

## Findings register

Priority describes user impact: P0 is a critical safety/data-loss/recovery issue,
P1 breaks or misrepresents a substantial workflow, and P2 is a bounded usability
or completeness issue. No P0 was established by this pass.

| ID | Priority | Feature | Confirmed gap |
|---|---|---|---|
| F-001 | P1 | Scheduled automations | WebUI offers manual execution through a permanently fenced API and reads empty compatibility runtime state |
| F-002 | P1 | Local triggers | Creation and durable enqueue work, but the normal gateway installs no queue consumer |
| F-003 | P1 | Sustained goals and runtime inspection | `/goal` and default-enabled `my` are advertised while their tools are absent from the foreground manifest |
| F-004 | P1 | Heartbeat and periodic Dream | Docs/settings imply active background behavior that the gateway deliberately does not install |
| F-005 | P1 | Remote access on macOS | Owner IPC starts but denies every request because only Linux peer credentials are implemented |
| F-006 | P1 | Generated media pagination | Stale pagination responses overwrite or append to a different session or refreshed library |
| F-007 | P2 | All-sessions media library | Only the first 200 session keys are scanned; the response hides the truncation |
| F-008 | P2 | Skills catalog | Fetch failures erase a successful catalog or look like a complete empty catalog, without an error/retry state |
| F-009 | P2 | Composer clipboard | A paste containing both a supported file and plain text cancels text insertion without preserving the text |

All findings have disposition **Repair**: restore an honest feature contract
with the smallest change. None justifies a subsystem rewrite.

## Feature coverage and implementation assessment at baseline

“Exercised” means source inspection and/or local tests for the named contracts,
not full production acceptance. Live credentials, external deliveries, native
applications, and real model services were not used.

| Feature family | Implementation owners | Assessment from this pass |
|---|---|---|
| Agent turns, tool results, streaming, stop, context pressure | `agent/loop.py`, `agent/runner.py`, context/session owners | Representative runner, tool-error, cancellation, and compaction contracts exercised; no new finding in those sampled contracts |
| Foreground planning/delegation/review | `agent/tools/fork_factories.py`, `agent/subagent.py` | Local orchestration and lifecycle tests pass; retain explicit capability construction and lifecycle ownership |
| Provider/model selection, retries, request budgets | `providers/`, runtime model resolver | Provider contract tests exercised; live providers, OAuth refresh, and rate limits remain unqualified |
| Filesystem, shell, web tools | `agent/tools/`, `security/`, resource policy | Local web/tool tests pass; native shell/full-access behavior needs platform qualification; see validation limits |
| Sessions, topics, history, temporary chats | `session/`, `webui/`, WebSocket adapter | Representative persistence, API, temporary-session, and UI contracts exercised; complete recovery/ownership audit remains open |
| Consolidation and manual Dream | `agent/memory.py`, memory recovery modules, `/dream` | Manual command and context-pressure tests exercised; periodic activation is a separate incomplete contract, F-004 |
| Schedules and manual automation actions | `cron/gateway.py`, `cron/service.py`, automation UI | Opt-in fresh-turn scheduler exists; general linked-session/WebUI claims exceed it, F-001 |
| Local script triggers | `triggers/`, CLI, `/trigger` | Storage and enqueue exist; delivery composition incomplete, F-002 |
| Sustained goals and self-inspection | `long_task.py`, `self.py`, command router, tool manifest | Implementations and isolated tests exist; normal registration is missing, F-003 |
| Remote owner, pairing, mobile edge | `remote/`, WebUI remote adapters | Real local IPC tests expose macOS failure, F-005; mobile remains a developing, unqualified surface |
| Image generation and media galleries | provider/tool adapters, generated-asset projections and hooks | Local generation/publication contracts exercised; pagination defects F-006/F-007; no live generation claim |
| Audio transcription and attachments | transcription providers, attachment services, composer | Local provider/WebUI/attachment tests exercised; mixed clipboard gap F-009; no microphone/device qualification |
| Skills, Agent Plugins, MCP | skills loader, `extensions/`, MCP adapters, Apps/Skills UI | Local loading/API/OAuth/plugin contracts exercised; catalog outage handling incomplete, F-008; live third-party tools unqualified |
| Knowledge repositories and sync | `context_repo/`, repository/save tools | Local Git/repository and explicit-save contracts exercised; keep scoped save/sync behavior; one `/proc` test fails on this host |
| Python SDK and OpenAI-compatible API | `nanobot.py`, `api/server.py` | Local request/streaming/fork contracts exercised; deployment/load behavior remains open |
| External chat platforms | `channels/` | WebSocket tested locally; Telegram scheduling collection blocked by missing optional dependency; other live channels not audited end to end |
| Installation, updates, native desktop/browser/device tools | CLI installers, desktop/mobile shells, native tools | Not qualified by this pass; dedicated install/restart/upgrade and native workflow checks still required |

The useful implementation qualities to preserve are explicit runtime ownership,
fail-closed admission, scoped media/workspace access, bounded provider request
handling, and tests for cleanup/cancellation. The gaps below largely occur at
the connections between those components and the user-facing feature.

## F-001 — Scheduled automation UI is ahead of runtime integration

**Contract and trigger.** [Automations](./automations.md) and
[WebUI Automations](./webui.md#automations) describe jobs running in their linked
topic, with next/last-run information and a “Run now” action. Open an enabled,
linked user job and request a manual run.

**Observed.** `AutomationActionGroup` computes `canRun` from the job's enabled,
linked, pending, and protected flags, without checking runtime support
(`webui/src/components/settings/system/AutomationsSettings.tsx:575`). The handler
at `nanobot/webui/ws_http.py:2057` consults
`CronService.admission_diagnostic()`, which unconditionally returns the
definition-only fence (`nanobot/cron/service.py:632`). `run_job()` also always
returns false (`:1139`). A real handler/service probe returned HTTP 409 after
`configure_gateway_execution()` was called and its status callback reported
enabled. That probe did not start a scheduler; the unconditional implementation
establishes that an actual active scheduler cannot change this handler's answer.

The status path has a related gap: `_public_job()` deliberately projects empty
`CronJobState` and zero timestamps (`service.py:257`), and
`nanobot/webui/session_automations.py:158` serializes those fields for the UI.
Runtime/delivery authority records are not joined into that view.

**Boundary.** Scheduling is not wholly absent. `GatewayCron` is an explicit
opt-in execution owner. Its validator requires fresh-turn identity and supports
Telegram receipt destinations (`nanobot/cron/gateway.py:29`); it does not provide
the general linked-topic behavior described in the docs. Its live Telegram path
was not qualified in this environment.

**Minimal repair.** Expose effective automation capabilities and unsupported
reasons; gate manual actions and correct linked-session claims. Then join status
from the existing runtime/occurrence/delivery owners. Implement manual execution
through those owners if it remains supported; do not bypass the service fence.

**Acceptance.** Test UI actions against a real management payload in inactive
and active modes. A supported manual request creates one admitted occurrence and
receipt; unsupported requests are explained before the user tries them. A
completed scheduled run appears with truthful last/next-run and delivery state.

## F-002 — Local triggers stop at the durable queue

**Contract and trigger.** Create `/trigger CI follow-up`, enqueue with
`nanobot trigger <id> "Review this result"`, and leave the normal gateway running.
The docs promise eventual delivery and restart recovery.

**Observed.** `nanobot/command/builtin.py:1017` creates a trigger and advertises
the CLI command. `nanobot/cli/commands.py:506` enqueues and prints “Queued”.
`run_local_trigger_queue()` exists (`nanobot/triggers/local_runner.py:20`), but a
full production-source search finds no caller. The task composition in
`nanobot/cli/gateway_runtime.py:650` starts no trigger consumer, including when
scheduled cron execution is enabled. Existing gateway boundary tests explicitly
assert that pending triggers are preserved without execution.

**Impact.** A user can successfully configure and enqueue an automation that
never reaches the agent through the normal gateway. Queue durability does not
establish delivery readiness.

**Minimal repair.** Report that trigger delivery is unavailable at creation and
enqueue time while retaining queued data. Restore delivery only through an
explicitly owned, admitted consumer with defined startup/shutdown behavior.

**Acceptance.** A normal gateway integration test must deliver one queued
message to the bound session, wait for a busy session, recover an interrupted
claim on restart, and remain inert when the capability is unavailable/disabled.
Keep the existing inactive-mode test.

## F-003 — Advertised agent capabilities are missing from registration

**Contract and trigger.** The
[long-running guide](./guides/long-running-ai-agent.md) presents `/goal` as a
working sustained-task example. [My Tool](./my-tool.md) says read-only runtime
inspection is enabled by default.

**Observed.** `cmd_goal()` sets authorization scopes and metadata, then sends the
request into the ordinary turn (`nanobot/command/builtin.py:942`).
`CreateGoalTool`/`UpdateGoalTool` exist in `agent/tools/long_task.py`. `MyTool`
exists in `agent/tools/self.py`; its default configuration is enabled, and its
availability predicate succeeds with the real loader fixture.

`ToolLoader.load()` uses the explicit foreground factories plus external
plugins (`nanobot/agent/tools/loader.py:203`). The foreground manifest
(`fork_factories.py:32`) includes none of `create_goal`, `update_goal`, or `my`.
Real owned-composition/registry probes confirmed all three are absent. Existing
goal integration tests explicitly install legacy goal tools themselves; their
passing result therefore does not prove the default `/goal` workflow.

**Impact.** A normal response to `/goal` does not establish a sustained goal.
Configuring `tools.my` does not expose the documented tool. Source presence and
isolated tool tests overstate feature readiness if registration is ignored.

**Minimal repair.** Give these features an explicit supported/retired decision.
While unavailable, remove or annotate the exposed command/config/docs. If
retained, add the required owned capabilities to the manifest with their current
authorization and session boundaries intact. Do not re-enable generic discovery
to recover unrelated legacy tools.

**Acceptance.** Use the default gateway/SDK composition, with no manual tool
registration. A supported `/goal` must create, continue, finish, and cancel a
goal; an unsupported one must return a clear availability response. Verify `my`
is present only when enabled and that its default mode rejects mutation.

## F-004 — Background memory/checking claims exceed effective capabilities

**Contract and trigger.** The automation guide says heartbeat is enabled by
default; the memory guide says Dream runs periodically by default. Enable the
config flags or edit `HEARTBEAT.md`, then start the gateway.

**Observed.** Both defaults are false in `nanobot/config/schema.py` (Dream
`:189`, Heartbeat `:720`). Heartbeat explicitly retains `enabled: true` as
compatibility input without constructing a scheduler. Gateway startup always
prints Dream and Heartbeat inactive (`nanobot/cli/gateway_runtime.py:611`).
Gateway boundary tests exercise both enabled flags and confirm no activation.
Meanwhile, `nanobot/webui/settings_system.py:114` exposes the configured heartbeat
flag and a Dream schedule without effective runtime availability.

**Impact.** Following the guides does not create the promised periodic work;
configured intent and an active service are indistinguishable in settings.
Manual `/dream` and ordinary context consolidation are separate capabilities
and are not shown broken by this finding.

**Minimal repair.** Correct defaults and current support in guides; report
configured and effective state separately. Keep activation fences until the
background work has an admitted owner. Reuse the existing schedule/occurrence
authority if periodic work is restored.

**Acceptance.** With each flag false/true, assert the effective status shown by
settings matches actual installed tasks. Disabled/unsupported states perform no
provider work. Any restored periodic service requires an actual due-run,
restart, and cancellation integration test.

## F-005 — Remote owner IPC has no macOS peer-authentication implementation

**Contract and trigger.** Start the owner and use the exposed remote
settings/pairing workflow on macOS.

**Observed.** `nanobot/remote/ipc.py:166` authenticates local peers only with
`socket.SO_PEERCRED`. On the audited Darwin host that constant is absent, so
every connection is denied with “peer credentials are unavailable on this
platform”. The owner socket can still start. Real owner/settings tests reproduce
`OwnerRequestError` and HTTP 503; downstream edge tests then fail with denied or
invalid-access responses. Short socket paths were used to separate this defect
from the macOS UNIX-socket path limit in the first test run.

**Impact.** Successful owner startup does not yield usable remote management or
authentication on this host. This is a current platform limitation in a
developing feature, not proof of a previously accepted release regression.

**Minimal repair.** Implement and test a macOS peer-identity mechanism, or report
remote-owner support as unavailable before setup/start reports readiness. Retain
same-user authentication and fail-closed behavior.

**Acceptance.** On Linux and macOS, exercise real IPC, settings, pairing, token
validation, revoke, owner loss, and restart. Demonstrate wrong-user rejection.
An unsupported platform must give an actionable capability response rather than
appear ready and fail every request.

**Repair result.** The [macOS follow-up](./audits/feature-implementation-2026-11-05/macos-repair-results.md)
implements Darwin kernel peer credentials before request parsing and preserves
same-user, socket privacy, admin and pairing checks. Real Darwin owner and
dependent workflows pass, including denial before dispatch. Linux decoding and
denial paths have controlled-response coverage; a real Linux kernel run remains
separate qualification. The observations above describe the original baseline.

## F-006 — Media pagination accepts responses from obsolete views

**Trigger.** Start “Load more”, then switch chats or refresh before the page
request resolves. Resolve the old page after the new first page.

**Observed.** Initial requests have cancellation guards, but pagination callbacks
do not (`webui/src/hooks/useSessionGeneratedAssets.ts:67`,
`useGeneratedAssetLibrary.ts:54`). Three deferred-response probes reproduced:

- Chat B becomes `[b-first, a-old-more]` after leaving chat A.
- A refreshed chat becomes `[refreshed-first, stale-more]`.
- A refreshed global library becomes `[library-refreshed, library-stale]`.

The obsolete response also writes total/cursor/loading state. This establishes
incorrect presentation of the current user's assets; it does not establish a
server-side authorization leak.

**Minimal repair.** Invalidate page requests when session, token, active/open
state, or refresh generation changes. Only the current generation may mutate
groups, cursor, total, or loading flags. Set the in-flight guard synchronously
so repeated clicks cannot request the same page before React renders.

**Acceptance.** Cover all three reproduced orderings, close/reopen and token
changes, failed pages with retry, and duplicate rapid load-more calls. Confirm a
late failure/finally cannot clear a newer request's loading/error state.

## F-007 — The all-sessions library silently loses sessions after 200

**Trigger.** Persist 201 WebUI sessions, each with one valid generated image,
then request `/api/generated-assets?limit=500`.

**Observed.** The real endpoint returned `total: 200`, 200 groups, and `next:
null`. `load_all_generated_asset_groups()` slices sorted session keys before
aggregation (`nanobot/webui/generated_asset_index.py:120`). Asset pagination
happens after this truncation in `ws_http.py:1788`; it cannot reach omitted
sessions. Selection is lexicographic, not based on recent use.

**Minimal repair.** Keep bounded work while exposing continuation across
sessions, or explicitly label the response/library as partial with a way to
reach omitted results. Do not replace the bound with an unbounded scan.

**Acceptance.** Exercise 0, 200, and 201+ sessions, including a recent asset in
a session beyond the first slice. Continuation must reach every included
session without duplicates, and total/completeness labels must be truthful.

## F-008 — Skills fetch failures are presented as an empty healthy catalog

**Trigger.** Reject the initial skills request, or reject a refresh after a
successful catalog load.

**Observed.** `useSkillsCatalog()` initializes `catalog_complete: true` and
empties `skills` in its catch path without an error or retry state
(`webui/src/hooks/useSkills.ts:7`). Probes observed a complete empty catalog after
HTTP 503 and disappearance of a previously loaded skill on failed refresh.
`SkillsCatalogSettings.tsx:134` warns only when catalog completeness is false.
The composer also consumes this catalog.

**Minimal repair.** Keep the last successful result for the same scope, expose
loading/transport failure separately, and provide retry. Preserve the existing
incomplete/protected-source semantics; transport failure must not silently
reclassify a skill's trust or availability.

**Acceptance.** Distinguish initial loading, genuinely empty success, initial
failure, stale data after refresh failure, retry success, and restricted sources.
Scope/token changes must not retain another scope's private catalog.

## F-009 — A file paste can discard accompanying text

**Trigger.** A clipboard event contains a supported PNG file and nonempty
`text/plain` content, such as a caption/instruction.

**Observed.** The actual hook calls the file callback and `preventDefault()`
without reading or inserting the text
(`webui/src/hooks/useClipboardAndDrop.ts:67`). The composer passes that handler
directly to its textarea. The hook probe confirms cancellation and no text
preservation path; a real browser/OS clipboard matrix has not been exercised.

**Minimal repair.** Preserve plain text exactly once, with normal cursor and
selection behavior, while accepting supported files. Native insertion or an
explicit insertion callback can satisfy the contract.

**Acceptance.** Browser-level checks for text-only, file-only, mixed text/file,
unsupported-file-plus-text, multi-file, and selected-text replacement pastes.
Do not duplicate text or attachments.

## Initial audit validation result

| Existing checks | Result |
|---|---|
| Full frontend suite | 94 files; **1,444 passed** |
| Frontend production build | TypeScript and Vite passed; main bundle size warning remains a measurement lead, not a proven performance defect |
| Available WebUI/cron/remote/API/SDK/gateway selection, short temporary root | **2,119 passed, 39 failed, 1 skipped** |
| Providers/repositories/attachments/MCP/manual Dream/triggers/process selection | **2,199 passed, 1 failed, 3 skipped** |
| Runner/context/cancellation/session-continuation/WebSocket selection | **639 passed, 3 failed** |
| Tool registration/delegation/plugins/image/web-tool selection | **323 passed** |
| New exploratory probes | **4 backend and 6 frontend assertions failed**, reproducing the recorded gaps |

These selections overlap in coverage; do not add their totals as unique tests.
The initial backend selection could not collect two Telegram-dependent modules
because `telegram` is absent. The available selection first produced 33 setup
errors from long macOS UNIX-socket paths; rerunning with a short root removed
those setup errors and yielded the 39 failures reported above.

The 39 include remote peer-authentication failures and their downstream effects,
a gateway shell smoke test requiring unavailable bubblewrap, three native
permission/process-accounting expectations, and one path-length fixture failure.
The three additional WebSocket failures report `full_access_unavailable` during
permission setup. The repository selection's one failure expects Linux `/proc`
host-state rejection on Darwin. These are not a green result, and the exact
platform attribution of every downstream assertion is not claimed complete.

The initial read-only pass obtained no full Python suite, unchanged strict
type-check PASS, coverage measurement, performance profile or live
channel/native/mobile qualification. Subsequent repair validation is recorded
above. Migration P6 and the architecture-audit phases remain open.

## Original repair order and completion criteria

1. **Correct availability claims first (F-001–F-004).** Return effective
   capabilities from their existing owners, gate commands/actions, and align
   docs/config. Preserve definitions, queue entries, and admission fences. These
   are small corrections; background activation is separate work.
2. **Repair local UI behavior (F-006–F-009).** Address request invalidation,
   library completeness, catalog error/retry states, and clipboard preservation
   independently, with behavioral regressions and browser checks where needed.
3. **Complete retained workflows one at a time (F-001–F-005).** Each needs a
   default-composition acceptance test, one execution/state owner, restart and
   cancellation behavior, and a clear supported-platform/transport contract.
4. **Continue the feature audit in live environments.** Qualify install/update,
   native command permissions, desktop/mobile, external chat deliveries,
   providers/OAuth, and third-party tools. Then use actual failing workflows to
   prioritize the charter's ownership, complexity, and performance analysis.

Each repair should be independently reversible. UI/docs corrections can be
reverted without data migration; any newly activated background capability must
support disabling it without discarding retained definitions or queue entries.
New execution authority requires its own rollback/recovery plan.

A finding closes only when the exposed behavior is truthful, the selected
supported workflow passes through normal composition, and the relevant failure
and recovery cases pass. A direct component test or a lower diagnostic count is
insufficient.
