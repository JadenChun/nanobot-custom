# H1 busy-input inbox: proposed engineering contract

Status: **design proposal, not an implementation or enablement authorization** (2026-09-28). The accepted user experience and dependency order are in [the migration plan](../nanobot-upstream-migration-plan.md#6a-h1-conversational-control-of-a-busy-assistant). This document narrows the scheduling and recovery design for review. The bounded P4 receipt-quarantine safety slice passed, but the candidate remains disabled by `MigrationIncompleteError`; H1 may not be activated before stable P5 interfaces and the P7 single-consumer canary.

## Stage discipline

A **stage is a functional capability**, with its implementation, integration and validation in scope. It can span several working rounds and include multiple internal work packages; a test fixture, review, or commit is a task within a stage, not a stage by itself. Agree on the observable outcome, exclusions and executable exit criteria before starting, keep a progress checkpoint across rounds, and close with a PASS/CORRECT/STOP verdict before changing functional scope. Do not silently expand the stage to fix unrelated legacy failures, refresh moving upstream or activate the candidate. If a prerequisite or material product choice is unresolved, pause that boundary with evidence rather than accumulating speculative code. The H1a–H1d boundaries below are implementation packages, **not four tiny stages**. The immediate migration critical path remains the unfinished P4/P5 gates; H1 implementation is not the next automatic step.

## Why the present loop is insufficient

The candidate serializes same-key execution with a session lock, but `AgentLoop.run()` admits busy input into a 20-element injection queue. When full, it dispatches another task; turn cleanup republishes remaining input onto the inbound bus, while automation uses a separate deferred list. Serialization is not an ordering contract: the fallback and republication can overtake earlier work. `record_pending_followup()` journals WebUI follow-ups, not authenticated cross-channel ingress, and the source-writer path disables live injection. The bus itself is an unbounded in-memory transport, not an acceptance ledger. These are specific gaps to repair, not reasons to replace the candidate's canonical turn, source, authorization, and delivery contracts wholesale.

Later upstream nanobot has a per-session FIFO worker and drains its queue at safe model boundaries. It is a useful scheduling pattern, but its in-memory queue and separate automation deferral do not establish H1's durable admission, conversational routing, or per-message authority. DeepSeek's separate inbox lanes and Codex's user-visible Queue/Steer choice are patterns for **distinct intent**, not proof that a channel can infer that intent or that two different lanes have one global FIFO execution order. Keep the pinned migration base unchanged; compare these implementations selectively.

## Proposed ownership and ordering

One **owner-session admission coordinator** is the only writer of the H1 inbox and the only dispatcher of its runnable work. It is bound to an authenticated assistant owner, not to the currently speaking channel. Channel adapters validate sender/chat and supply an immutable origin/reply route and stable transport ID; they do not make the steer/queue decision. The existing direct SDK/API turn contracts retain their explicit session/history behavior and may not be silently converted into IM control messages. Any same-owner execution must still respect the canonical session lock and source owner.

1. In one admission transaction, deduplicate `(channel, account/bot, chat/thread, transport-message ID)` within the authenticated owner scope, assign a monotonically increasing `ingress_seq`, save the complete original content/media references, route, effective trust/scope, and `pending-routing` state. Return the existing receipt state for a duplicate. If a transport lacks a stable ID, its retry/dedupe strategy is a channel-specific gate, not a synthetic exactly-once claim.
2. Only after durable admission may the channel send an **accepted** receipt. A limited, separate control-model worker receives the direct user's text and a minimal factual run snapshot; it proposes a route without tools, history-writing, cancellation, or task authority. A single coordinator validates the proposal against the current owner/run generation and commits one revisioned route before an informed reply is queued. Timeout/error conservatively commits `next-turn`; low-confidence ambiguity commits `awaiting-clarification` and asks a question. An explicit authenticated `/stop` skips model interpretation but uses the same generation-fenced owner transition.
3. `next-step` is eligible only for the named active run and is claimed at a safe **pre-model** boundary. Bind the ingress ID to the source user event before that provider request, pass the **full** message once, and retain the active run's tool/workspace authority; a broader permission request becomes a separate turn/approval. Never inject while a provider call or side-effecting tool is in flight. If the run finishes or its generation changes first, atomically reroute the unclaimed message to `next-turn` and correct its channel confirmation. The classifier's late reply cannot resurrect or stop an unrelated run.
4. `next-turn` is a distinct task lane, never an injected row in the active run. One worker claims eligible messages in `ingress_seq` order after the active turn reaches its bounded terminal/slice boundary. While an older message is still `pending-routing`, younger task claims wait for the bounded routing outcome. `awaiting-clarification` is explicitly suspended, not a permanent head-of-line lock: a later clarification retains the original ingress ID for audit/dedupe but acquires a new runnable eligibility order. A true steer may intentionally overtake a queued task because it targets the *current* run; FIFO is guaranteed **within each runnable lane**, not across incompatible intents. The confirmation must say whether it will affect this run or a later turn.
5. Automation occurrences keep their separate occurrence/delivery IDs and never enter the personal-message classifier. They share the same owner execution worker and lock after the current slice, without another republish path. Eligible human next-turn work has precedence at the slice boundary; automation remains visibly pending and requires an explicit starvation bound, not an unbounded low-priority list. Its completion future is settled only from the occurrence's actual terminal/delivery state, not from admitting or deferring it.

The in-memory queue becomes only a bounded wake-up/cache for the durable ledger. Do not dispatch a competing same-key task when it is full, silently drop accepted input, or republish leftovers to change their order. Define disk/backlog quotas before enablement: at capacity, either admit durably with a documented bound or **reject before acceptance** and send a truthful, retryable overload response. A Telegram update handler may return promptly after the durable write, without waiting for inference. If storage cannot commit, it must not say “accepted”; transport acknowledgment and any failed overload reply need their own retry/visibility treatment.

## Revisioned transitions and crash boundary

```text
authenticated transport input
    → accepted / pending-routing
    → next-step | next-turn | control | awaiting-clarification
    → claimed (run/turn ID + source binding, where applicable)
    → committed | cancelled | uncertain-and-fenced
```

The writer validates expected revision, owner, origin, current run ID/generation, and per-message capability on every transition. A classifier response can be committed out of order, but cannot bypass an earlier unresolved routing barrier for task claims. A clarifying answer references the original ingress ID; reclassification never creates a second task. Receipt, informed routing reply, clarification, and completion notice are idempotent outbound intents keyed by ingress ID and reply stage. Channel send failures remain pending delivery and do not roll back accepted task input or trigger task replay.

For `next-step`, source-event persistence and inbox claim require a common atomic owner or a recoverable binding protocol. Record a dispatch/claim marker before external work; commit the bound user row and retire the inbox entry without duplicating history. If the process dies between those writes, reconcile by ingress/source-event ID before deciding whether the row is present. Requeue only when no provider/tool effect could have started; uncertain external effects remain fenced for explicit recovery, never blind replay. Existing WebUI follow-up journals and recovery checkpoints must be imported/reconciled, not discarded. H1 storage must respect the P4 exact-root quarantine and must not generalize its descriptor-bound receipt writer into an ordinary JSONL helper.

`stop` fences only the observed active run. If it races completion, report the completed status; `stop-and-replace` records the new task as `next-turn` and waits for the stop/recovery boundary, including uncertain tool outcomes. `status` reads actual run/queue/receipt events without inserting a user row into the active task. The route-only control call never uses or mutates the task's provider conversation state.

## Delivery and privacy proposal

The active run's full result remains on its original delivery route. A cross-channel steer receives an admission/routing receipt and a completion notice on its own route, not a duplicate full transcript; a later queued turn's result goes to the route that submitted it. Each output requires the current owner authorization and the immutable originating route, including thread/reply metadata. Do not infer authority merely from a shared session key. Persist no control-model chain of thought, full prompt, provider handle, or sensitive absolute path in operational logs; channel content follows the selected private session/source retention policy. Temporary or nonlogging sessions require a specific inbox retention/erasure design before support.

## Work packages and evidence gates

| Boundary | Work, dependencies, and deterministic evidence |
| --- | --- |
| H1a: contract and route prototype | After P5 interface freeze: freeze ambiguous/quoted/forwarded control examples, origin-ID support, authority intersection, IM reply objectives and model budget. Simulate bounded concurrent inference, out-of-order results, timeout and neutral receipt with no task model calls. Shadow only; no automatic route commits in production. |
| H1b: admission ledger | After approved persistence contract: single writer, revision/ingress ordering, bounded backlog, dedupe, reply outbox and restart reconciliation. Crash tests at accept, route, claim, source append, and delivery; bursts above 20 accepted messages never create another same-owner runner. |
| H1c: canonical execution | After H1b and stable source binding: safe-step steer versus independent turn, generation-fenced stop/status, automation fairness/completion, cancellation and uncertain-effect recovery. Test exact ordered model inputs, one source row per ingress ID, per-message capability, mid-tool blocking and source-writer mode. |
| H1d: channel integration | Telegram first, then each enabled IM adapter. Test authenticated and denied senders, duplicate webhook, failed receipt/confirmation, cross-channel steering and exact thread/destination delivery. No migration or release promotion implied. |

The single-consumer canary must compare held-out routing accuracy, receipt/confirmation latency, extra model tokens/cost, accepted-versus-visible-pending messages, duplicate side effects, and task completion against the current loop. Rollback exports the ledger and delivery state; it does not drop accepted messages or turn off admission while unclaimed work exists.

## Material decisions before implementation dispatch

These are recommendations, **not silently accepted defaults**:

- Select the exact P5 source/session binding and migration path for existing WebUI follow-ups, including temporary/nonlogging sessions; decide whether the inbox can share its atomic owner or needs a reviewed two-store recovery protocol.
- Approve the scoped priority rule above (steers overtake next-turn work; earlier `pending-routing` briefly bars younger task claims; clarification suspends and later re-enters) and a bounded, starvation-safe automation policy.
- Set measured IM neutral-receipt and informed-reply targets, classifier provider/isolation, concurrent-call/rate/token/cost limits, timeout, backlog/disk quotas, and overload behavior. A neutral receipt plus safe next-turn fallback is required when inference is unavailable.
- Approve the cross-channel delivery rule and precise user-visible meanings of “accepted,” “queued,” “steered,” “stopped,” and “completed,” including failed channel acknowledgments. Confirm the supported transport-ID/dedupe scope for each enabled channel.

Until those choices and P5/P7 gates are satisfied, H1 is a testable design, not a claim that the current candidate already behaves like Codex or DeepSeek.
