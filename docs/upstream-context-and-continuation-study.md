# Context management and sustained execution: upstream study

Date: 2026-09-08. Local baseline: staging `4aa562d9`, now on
`codex/upstream-migration`. Upstream baseline: `f47b8f23155ddec23cfb80ec001b544d495df89c`
(main, newer than released v0.3.0).

This is a source study and adoption proposal, not an implemented or benchmarked change.

## Context management

Our `nanobot/agent/context_budget.py` manages a shared prompt budget. Its reduction
sequence compacts old tool results, drops old turns, then applies deterministic
request-local reduction. Despite older docstrings, enforcement deliberately does
not make recursive LLM memory-compression calls. Shared detached reductions and
provider-boundary guards already bound requests; the runner rechecks after hooks.
Dropping turns can still remove decisions and task state from model context.

Upstream's `nanobot/agent/context_governance.py` separates the source transcript
from model-facing messages. `ContextCompactionState` tracks the history already
accepted by the provider and the new messages after that boundary. On pressure,
`_compact_request_history` summarizes the accepted history and preserves the new
messages. It emits a `SessionSummaryCheckpoint` containing both the summary and
the raw transcript boundary it covers.

The important mechanisms are:

- Build requests from copies rather than rewriting the source history in place.
- Preserve the new message delta when replacing older history with a summary.
- Persist an explicit summary boundary, so reloads know what the summary covers.
- Include tool definitions and output reserve when calculating request capacity.
- Use matching provider usage or resumed-state measurements when available;
  otherwise estimate the actual request. Old usage is not automatically valid
  after messages or tool definitions change.
- Reset provider conversation state after replacing history. An append-only
  Responses continuation cannot safely append to a rewritten prefix.
- Check the rebuilt request again. Summary failure or an oversized rebuilt
  request must produce a controlled failure, not silently send an invalid payload.
- Normalize tool messages and repair missing results before provider submission.
- Support disk offloading for large tool results. This preserves a retrieval path
  rather than relying only on lossy snippets.

`nanobot/agent/autocompact.py` also supports idle-session summarization when a
positive TTL is configured. This is optional; it adds model work and should be
evaluated after pressure-triggered compaction is correct.

## Sustained execution

Upstream separates three mechanisms that should remain distinct in our design:

1. **Within-run continuation.** The runner consults a continuation callback at
   an eligible final-response boundary. The loop supplies the active goal and
   asks the agent to continue or mark the goal complete.
2. **Across-turn continuation.** `session/turn_continuation.py` queues another
   internal turn when the iteration limit is reached and an active goal is
   eligible. The current code limits this path to 12 continuation rounds. It
   suppresses the intermediate final response, retains the visible start time,
   and does not save the internal instruction as a new user message. A pending
   queue is required; continuation is not available through every entry point.
3. **Restart recovery.** `session/recovery.py` restores checkpointed history and
   marks pending tool calls as interrupted. It does not simply replay them.
   The coordinator requires confirmation after restart and validates recovery
   admission so newer user input can supersede stale recovery work.

Goal state is stored in session metadata and supplied as runtime context. Ordinary
chats do not automatically become sustained tasks. The 12-round limit applies to
one continuation path; it is not a universal cost or elapsed-time ceiling.
Upstream also relaxes the normal LLM wall timeout for sustained goals, so adopting
that behavior requires an independent decision about stall and resource limits.

None of these mechanisms requires background subagents. Our foreground roles,
read-only profiles, scoped worker writes, and crawler can remain intact.

## Proposed adoption order

1. **Durable summary checkpoints:** add model-facing transcript construction and
   a summary boundary while retaining raw evidence; keep the existing final
   request-budget check. Prefer a checkpoint over dropping task history.
2. **Provider-state coordination:** invalidate continuation state after history
   replacement, and retain new tool results exactly once. Integrate this with
   step 1 before enabling checkpoint compaction for stateful providers.
3. **Explicit sustained goals:** connect goal state to our canonical
   `TurnRequest`/`TurnResult` path and run records. Model a logical goal separately
   from its execution slices; a slice ending is not the goal completing.
4. **Bounded continuation:** schedule eligible slices without fake user messages
   or repeated final notifications. Preserve stop, approval-required,
   policy-blocked, and cancellation states as hard boundaries. Define goal-wide
   limits separately from per-turn delegation budgets so resets do not erase
   the total work budget.
5. **Recovery:** persist execution checkpoints and require explicit admission
   after restart. Unknown outcomes of side-effecting tools must be reconciled,
   not automatically retried.
6. **Optional idle compaction:** evaluate token cost, latency and retained facts
   on realistic long sessions before enabling it by default.

## Acceptance evidence for a future implementation

- A long task retains its objective, constraints, completed work and next steps
  after multiple compactions and after reloading the session.
- New tool results survive compaction exactly once; tool call/result pairs remain
  valid, including cancellation during a tool batch.
- Replacing history clears stale provider continuation state.
- Summary failures and oversized single messages fail predictably while leaving
  the source transcript available.
- Normal chat never enters automatic sustained continuation.
- An active goal crosses an iteration boundary without duplicate user messages,
  duplicate deliveries or an intermediate completion claim.
- Stop, approval and policy blocks prevent the next slice from executing.
- Goal-wide budgets remain effective across per-turn budget resets.
- Restart recovery does not replay completed or uncertain external mutations;
  newer user input supersedes stale continuation requests.

Compare retained task facts, successful completion, token usage, latency and
duplicate side effects against our current harness. More machinery alone is not
evidence of better task performance.

## Source references

All upstream paths above refer to the pinned commit:
https://github.com/HKUDS/nanobot/tree/f47b8f23155ddec23cfb80ec001b544d495df89c/nanobot

Key files: `agent/context_governance.py`, `agent/autocompact.py`, `agent/runner.py`,
`agent/loop.py`, `session/summary.py`, `session/goal_state.py`,
`session/turn_continuation.py`, and `session/recovery.py`.
