# P6 fork compaction test migration — 2026-10-13

Bounded test migration accepted. **P6 remains NOT_ACCEPTED**;
runtime/D9/P7 remain unauthorized. No production implementation, shared fixture,
guard, environment, dependency, or frozen runner/profile changed.

All six old cases in `tests/agent/test_loop_consolidation_tokens_fork.py` failed
at the removed `AgentLoop(max_tokens=...)` argument. Beyond that argument they
depended on removed `memory_consolidator`/`budget_manager` objects and the old
background half-target reducer. Restoring those mechanisms would conflict with
the accepted P4 canonical governor and P5 inactive background-compaction graph.

The replacement preserves the underlying safety concerns under the accepted
contract, rather than pretending the old algorithm's assertions still apply:

| Historical concern | Current qualification |
| --- | --- |
| Below/above pressure threshold | Three canonical-source cases immediately below, at, and above the 90% trigger. |
| Legal history boundary | A real recorded call/result batch is summarized together; the newest user correction stays in the suffix. |
| Repeated reduction until target | Two cases require successive disjoint chunks below the trigger, stop at the 60% target, and deliberately remain above the retired 50% target. |
| Background ordering | An acquired legacy foreground loop proves summary-before-main ordering and no legacy AutoCompact/background archival. |
| Failure isolation (added) | Empty summary and cancellation each retain the prior checkpoint/projection and stop after one auxiliary call. |

Canonical cases use actual owned source storage and verify unchanged source
events. The legacy case compares the persisted pre-turn prefix, including event
identities allocated by its initial save. Deterministic provider responses are
used; this is not live model-quality evidence. Existing pressure tests in the
broader selection also cover the four-call cap, repeated compactions, invalid
metadata races, and irreducible input.

Evidence in `/private/tmp/nanobot-p6-completion.rk7wje`:

- `p6-compaction-contract-r1-before`: six reproduced failures.
- `p6-compaction-contract-r3-owning`: nine passed, complete owning module.
- `p6-compaction-contract-r3-broader`: 163 passed across seven complete modules
  covering legacy replay, canonical pressure/publication, runner/context
  governance, consolidation, and gateway automation dormancy.
- Lint PASS; unchanged full strict before/after: zero errors/warnings, 464 files.
- Comparator: zero new diagnostic signatures.
- `p6-compaction-contract-r3-progress.json`: source inventories, exact scope,
  result hashes, and old-to-current coverage mapping.

All accepted validators terminated cleanly, without timeout or forced cleanup.
Intermediate test corrections used the actual tool-batch recording API and
persisted legacy identities; no production defect was established in this group.

The previous whole-suite run remains incomplete at its 900-second timeout.
Neither this checkpoint nor the [temporary-session checkpoint](p6-temporary-policy-checkpoint.md)
establishes a current whole-suite failure count. The remaining observed backlog,
separately qualified longer full-suite run, and later P6 gates remain open.
