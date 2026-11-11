# Persistence, completion and resource lifetime — REF3/REF5

Keep durable-before-effect source writes, shared writer failure, cancellation-resistant
ownership, unknown-effect recovery and qualified cron receipts. The comparison identifies
conditional candidates around yielded hard deadlines and retained GitStore write scope,
plus smaller persistence/retention opportunities. It establishes no measured latency,
crash behavior, restartable OS process or universal exactly-once delivery guarantee.

The [frozen pins](README.md#source-pins) and [exact targeted/search-only lists](provenance.json)
define the boundary. Normal Codex local writes **await a recorder flush**; the queue alone
is not the relevant caller contract. All tests were source reads, never executions.

## P1 — Preserve durable acknowledgement before effects

| Field | Comparison |
| --- | --- |
| Required outcome | Source intent and process-continuation identity must be durably accepted before effects; responsiveness changes must preserve ordering, CAS, error and shutdown acknowledgement. |
| Nanobot implementation/configuration | [SourceTurnWriter](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/session/turn_writer.py#L95) synchronously persists owner/event/run/stream/shared snapshot (95–132), shares child writes (87–89), poisons on append failure, and commits tool-batch intent before effects (221–228). Source store validates/serializes the complete source (1191–1211) and CAS (1299–1311). [Session persistence](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/session/manager.py#L1144) uses temp/flush/file-fsync/replace/directory-fsync (1144–1195, platform exceptions 1172–1181). Process completion uses synchronous SQLite `synchronous=FULL` (`process_completion.py:69–107`); exec_session.py:400 creates the journal before launch. |
| Reference mechanism | DSH [JSONL storage](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/session/session-persistence-jsonl/src/storage.ts#L187) has explicit durable append/flush (187–211), cloned routed-event buffering (274–315,549–560), and append/fsync/rollback in index.ts:1408–1440. Codex [record_items](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/rollout/src/recorder.rs#L1079) alone admits to a queue, but the normal [local live writer](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/thread-store/src/local/live_writer.rs#L459) immediately awaits `recorder.flush` (459–466); projection follows that barrier (440–450). Concrete recorder writes use `write_all` plus flush (2156–2161), with no file/directory fsync visible there. |
| Writer/identity/lifetime/authority/error boundary | Source CAS, mandatory turn writer and process journal have distinct durable obligations. Queue admission, userspace flush, SQLite FULL and file/directory fsync are not interchangeable acknowledgements. |
| Actual difference/disposition | **Preserve; bounded responsiveness candidate C8.** Synchronous persistence on async callers creates a source-visible blocking risk, with no measured current latency. References illustrate an ordered writer/barrier, not a drop-in durability replacement. |
| Conditions/impact/confidence/next step | High confidence in call/ordering boundaries. Start with the process-completion journal under A01-03; an ordered async owner must retain awaited durable-before-launch acknowledgement, CAS/order/error/shutdown. Broader source persistence is a separate scope, not part of the same repair by default. |

## P2 — Preserve ownership when effects resist teardown

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Do not release exclusive state ownership while cancellation-resistant work may still act. Platform lease support remains a pending platform contract. |
| Nanobot implementation/configuration | [Runtime authority](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/session/runtime_authority.py#L69) acquires a writer lease and closes resolver before lease (69–111). `mode_storage.py:385–454` uses a pinned private nonblocking workspace lease and fails closed without `fcntl`. Execution owner lock order/shielded child drain (108–159) closes execution before lock release (190–208). RuntimeComposition stops teardown on `RuntimeOwnedWorkError` (336–346); cron runtime retains resources/scheduler lease if drain fails (386–406,431–479). |
| Reference mechanism | DSH [lifetime kernel lease](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/session/session-persistence-jsonl/src/lease.ts#L70) has no expiry (1–27,70–133) and a separate Windows mechanism; storage.ts:223–259 releases its claim even if draining fails and reports an error. Jobs-local index.ts:654–711 waits for compliant producers, while throwing cancellation force-fails and warns of a possible orphan. Codex [recorder shutdown](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/rollout/src/recorder.rs#L2022) leaves a failed-shutdown writer alive (2022–2029), with writer-lock ownership at 149–164. ProcessManager requests termination (1802–1817); the remote process path in unified_exec/process.rs:244–254 spawns asynchronous termination. |
| Writer/identity/lifetime/authority/error boundary | Lease, resolver, execution task, scheduler and process lifetime are different owners. Teardown success must not be inferred from a termination request. |
| Actual difference/disposition | **Preserve.** No reference provides a universally stronger close barrier across these scopes. |
| Conditions/impact/confidence/next step | High source confidence in named close paths; operational drain/lease behavior unobserved. Keep resistant-work retention in any async-writer or platform repair and audit alternate close callers separately. |

## P3 — Give yielded hard deadlines an independent owner

| Field | Comparison |
| --- | --- |
| Required outcome | An advertised hard timeout must end the process independently of whether the caller polls or asks for completion notification. |
| Nanobot implementation/configuration | [Shell contract](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/shell.py#L169) advertises a hard timeout (169–172); the session path is selected at 556–563 and receives timeout at 646–656. [ExecSessionManager](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/exec_session.py#L176) stores a deadline (176–177), checked by poll (260–274) or completion watcher (458–463). That watcher is created only with an optional completion sink (396–400,447–450). Spawn paths (`exec_session.py:627–644`, shell.py:943–1042) and native payload (native_process.py:287–296) carry no timeout. Cleanup uses idle age (613–625); stop/shutdown are separate (`loop.py:1487,3079`). One-shot non-yield shell uses `wait_for` (578–584) and enforces its timeout. |
| Reference mechanism | DSH [job wait](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/jobs/jobs-local/src/index.ts#L496) timeout returns a projection (496–527); kill is separate (484–494). Codex [process manager](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/unified_exec/process_manager.rs#L651) kills for a completion-only deadline (651–675); interactive yield has another contract. These are not identical timeout promises. |
| Writer/identity/lifetime/authority/error boundary | Deadline ownership must follow the live process/tree rather than the notification sink, caller polling or next cleanup operation. |
| Actual difference/disposition | **Static conditional defect candidate C13.** A yielded session with finite timeout, `notify_on_exit=False`, and no later poll/stop/shutdown has no independent deadline timer in the inspected path. This contradicts nanobot's advertised hard deadline, not reference feature parity. |
| Conditions/impact/confidence/next step | High static confidence; runtime behavior unverified. After the first S02 matrix, give the process lifetime one deadline owner independent of notifications, with cancellation-safe tree termination and existing close/authority retention. Do not change DSH-like wait semantics or add budgets by imitation. |

## P4 — Preserve unknown-effect recovery and original scope

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Completion continuation must retain the original owner/scope and refuse blind replay of uncertain work; delivery acknowledgement remains separate. Restart identity reconciliation needs a later bounded audit. |
| Nanobot implementation/configuration | Continuations default off (`schema.py:762`) and gateway explicitly installs them (`gateway_runtime.py:586–596,668–669`); `notify_on_exit`, persistent owner and captured scope are required. [ProcessRecord](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/process_completion.py#L31) retains session/route/instruction reference/project/mode (31–49,113–127). Startup marks waiting/dispatching unknown, schedules only ready records (157–166), and `_continue` persists dispatching before canonical submit; interruption becomes unknown (179–207). `done` follows automation response (`automation_turns.py:115–133`), not a channel receipt. Loop notifies before delivery.complete (2823–2838); turn_delivery.py:338–381 queues the bus. |
| Reference mechanism | DSH [jobs-local](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/jobs/jobs-local/src/index.ts#L1) is process-local. Tool-jobs index.ts:354–410 wakes the current owner and drops during teardown/no-owner; pending notices are 260–282. Codex [exit watcher](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/unified_exec/async_watcher.rs#L327) emits a local end event (327–379) from a process-local registry. None proves restartable OS processes or external receipts. |
| Writer/identity/lifetime/authority/error boundary | Journal completion, canonical automation response, queued channel delivery and receipt are distinct. Records lack incarnation/generation; `_continue` checks session existence (183–185), but post/new downstream admission requires further reconciliation. |
| Actual difference/disposition | **Preserve; open identity question.** No replay defect or stale-authority bypass is established by that field absence alone. |
| Conditions/impact/confidence/next step | High confidence in current journal states; restart/recreated-session semantics remain unknown. A01/H01/H02 must trace downstream canonical admission before any schema or recovery change. |

## P5 — Define the completion output view before changing buffers

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Does a completion report promise all retained process output, only unread output, or a bounded summary? The governing product contract is pending. |
| Nanobot implementation/configuration | [Exec output buffers](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/exec_session.py#L87) bound stdout/stderr head/tail (87–144); stream reading is 204–217. Poll drains shared buffers (292–300); the watcher combines initial output and final poll (458–470), while foreground write/poll can consume intermediate/terminal output (514–522). Process journal validates owner (process_completion.py:147–155). |
| Reference mechanism | DSH [OutputRing](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/jobs/jobs-local/src/ring.ts#L38) uses absolute non-consuming offsets/loss (38–111); jobs-local index.ts:282–305 awaits the final pump drain. Codex process.rs:43–55 separates pending output from transcript under lock; its async_watcher.rs:327–379 reads transcript. |
| Writer/identity/lifetime/authority/error boundary | A foreground poll cursor and completion transcript are different consumers. Canonical tool results may already retain bytes removed from the shared buffer. |
| Actual difference/disposition | **Conditional improvement C9/product discussion.** Intermediate polls may omit bytes from the completion report, but conversation-wide data loss is not established. |
| Conditions/impact/confidence/next step | High confidence in consumption mechanics; desired completion view unknown. Decide that view, then consider a bounded separate transcript/cursor under one process-output owner. Preserve truncation and resource limits. |

## P6 — Separate lock scope from retained-record reclamation

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Keep safe launch refusal and failed-close ownership while allowing explicit handling of settled data. Resource budgets, privacy/retention and unknown-record reconciliation remain pending. |
| Nanobot implementation/configuration | Exec sessions default to eight plus idle timeout (`exec_session.py:346–355`). [Start](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/exec_session.py#L385) holds the manager lock across cleanup/spawn (385–429); cleanup awaits kill under lock (613–625). `close_all` already snapshots/kills outside and reinserts failures (551–581). [Process journal admission](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/process_completion.py#L96) caps **all** records at 4096, including terminal/unknown. No prune/delete/export/reconcile API was found in the inspected nanobot/scripts search scope. Refusal occurs before launch. |
| Reference mechanism | DSH jobs-local index.ts:33–60 separates owner active/settled limits, terminal-only removal (340–343) and owner disposal (654–686). Codex [process storage](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/unified_exec/process_manager.rs#L1155) avoids output waits under the store lock (1155–1198), installs watchers/prunes outside it (1247–1305), and uses a soft cap protecting recent/busy entries (1729–1799). This is not a persistence retention policy. |
| Writer/identity/lifetime/authority/error boundary | Manager admission/slots and persisted completion records have different lifetimes. Unknown effects must not be silently reclaimed as ordinary settled records; external work may remain. |
| Actual difference/disposition | **Bounded candidates C10/C11.** Source suggests avoidable lock blocking and eventual safe refusal without a public reclamation route, not measured contention or an unsafe cap. |
| Conditions/impact/confidence/next step | High confidence in named lock/cap paths. Split owner-scoped settled inspection/export/reclaim from lock narrowing with reservations. Retained commands/instructions/output have privacy consequences. Do not copy reference counts or accept new budgets here. |

## P7 — Preserve cron compute, projection, outbox and receipt boundaries

| Field | Comparison |
| --- | --- |
| Required outcome | Recover accepted computation without rerunning the agent, and distinguish partial/uncertain channel effects from a qualified receipt. |
| Nanobot implementation/configuration | Cron gateway is opt-in (`cron/gateway.py:49–73,108–135`); runtime recovers before timers (169–204). [Composition](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/cron/composition.py#L134) begins the exact occurrence (134–137), then canonical completion→visible link→outbox dispatch (193–229). canonical_result.py:24–73 recovers rather than reruns. Delivery runtime durably records sending before channel work (947–965), with partial/exception/cancellation uncertainty (785–820,882–945). channel_sender.py:24–90 binds channel/target/action/digest/message count. |
| Reference mechanism | Inspected DSH/Codex process notices have no equivalent external receipt guarantee. They cannot certify or replace this nanobot-specific delivery contract. |
| Writer/identity/lifetime/authority/error boundary | [Telegram receipt path](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/channels/telegram/runtime.py#L1258) prevalidates a bundle (1258–1281), validates acknowledgements (1160–1190), retains partial prefix (1314–1323) and binds receipt (1327–1333). Its process-local lock (1222–1224) serializes receipt-path bundles per target; it cannot serialize ordinary channel sends or sends from another process. Fallback (1292–1304) is only for confirmed HTML parse errors. |
| Actual difference/disposition | **Preserve.** No universal exactly-once claim. Process completion does not inherit cron receipt semantics, and other channels need their own qualified sender contracts. |
| Conditions/impact/confidence/next step | High confidence for the inspected qualified Telegram path; other adapters remain queued. A01/E01 reconciliation must preserve compute-versus-delivery uncertainty. |

## P8 — Audit retained GitStore caller write ownership

| Field | Comparison |
| --- | --- |
| Required outcome | A memory auto-commit must not capture unrelated staged work or rewrite selected staging without an explicit contract. Retained/public caller compatibility must be traced before changing or deleting it. |
| Nanobot implementation/configuration | [GitStore](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/utils/gitstore.py#L125) stages designated paths, then checks the entire index and calls unrestricted `porcelain.commit` (125–166;139–156); existing `.git` directories are accepted (54–67). Memory selects SOUL/USER/MEMORY/cursor (`memory.py:95–129`); retained manual Dream auto-commits in `finally` (`command/builtin.py:547–550`). Canonical `/dream` is currently rejected (`loop.py:4361–4371`). Separate knowledge Git sync already has selected commits/reject-selected-prestage (git_sync.py:135–148) and common-directory locking (280–293). |
| Supplemental dependency source | `pyproject.toml:50` permits Dulwich `>=0.22,<1`. At the publicly pinned **0.25.2** commit, [porcelain.commit](https://github.com/jelmer/dulwich/blob/287097f0514783d35e2a427bb4451ab3f804aee8/dulwich/porcelain/__init__.py#L1128) calls `worktree.commit` without a tree (1128–1140); [WorkTree.commit](https://github.com/jelmer/dulwich/blob/287097f0514783d35e2a427bb4451ab3f804aee8/dulwich/worktree.py#L467) commits the whole index (467–494). Coordinator verified these blobs match the analyst's read bytes. This is portable allowed-version evidence, not deployed-version attestation. |
| Reference applicability/boundary | DSH/Codex writer leases illustrate ownership only; there is no claimed direct Git mechanism equivalence. A nanobot lock alone cannot coordinate unrelated external Git writers. |
| Actual difference/disposition | **Strong version-specific static conditional defect candidate C7.** On a reachable retained/public caller with existing unrelated staging, that staging can trigger or enter a memory commit even when memory is unchanged; selected staging may be refreshed from the worktree. The normal canonical manual route is dormant. |
| Conditions/impact/confidence/next step | Trace retained/public callers first. Then consider conservative refusal **before mutation**, or selected-tree/index isolation with owned-caller serialization, in a separate K01/H02 contract. No compatibility deletion or global Git abstraction is authorized. `tests/utils/test_gitstore.py` was read, not run. |

## P9 — Preserve useful display repair and identify its provenance

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Does a displayed completed tail promise canonical terminal proof, or may it represent inferred presentation repair? The user-visible guarantee is pending. |
| Nanobot implementation/configuration | [Transcript repair](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/webui/transcript.py#L1486) matches user/assistant history and synthesizes `turn_end` with completed outcome (1486–1535); 2821–2852 returns response-only repair plus `completed_turn_ids`. This is not canonical terminal proof or receipt. |
| Reference mechanism | DSH jobs-local index.ts:603–636 commits terminal state before waiters/notice. Codex [local projection](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/thread-store/src/local/live_writer.rs#L440) is explicitly rebuildable after rollout flush (440–450). Different projections have different authority. |
| Writer/identity/lifetime/authority/error boundary | Response history, inferred display completion, canonical terminal event and delivery receipt are different writers/proofs. |
| Actual difference/disposition | **Preserve presentation repair; product discussion D7/C12.** Decide inferred-provenance/terminal UI promises before changing public schemas or behavior. |
| Conditions/impact/confidence/next step | High confidence in synthesis mechanics; intended UI contract unknown. Reconcile with A8, which concerns mandatory-writer failure and is a different failure boundary. Do not infer that every synthesized completion is incorrect. |

See the [deduplicated roadmap](README.md#improvement-roadmap) and [later source contracts](audit-backlog.md).
Deterministic document validation establishes integrity, not these mechanisms' runtime stability.
