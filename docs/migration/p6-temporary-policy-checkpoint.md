# P6 temporary-session test migration — 2026-10-13

Bounded repair accepted; **P6 remains NOT_ACCEPTED**. Runtime/D9/P7 remain
unauthorized. This checkpoint follows the summary-replay repair and changes no
production code, shared fixture, guard, environment, or dependency.

The four historical failures in `tests/agent/test_loop_session_policy.py` used
direct guarded construction. After actual composition acquisition, the tests
also needed explicit retained transient activation and the canonical turn/bus
adapters replacing the removed `_process_message` method.

The corrected module verifies private history over two turns, durable-memory
exclusion, disabled-tool filtering, separation from the unified session, stale
session rejection without disk fallback, and cancellation of an active turn on
discard. The discard case invokes the real consumer and real shutdown, joins the
consumer in `finally`, and leaves graph cleanup/authority reacquisition checks
to the unchanged acquired-runtime fixture. External effects remain fenced.

The P4 artifact-offloading contract records `read_session` outside the default
core factory. The old unconditional tool-membership assertion is therefore
covered in two explicit cases: default absence and an installed real read-only
`ReadSessionTool`. All other original `assert` statements are retained.

Local evidence in `/private/tmp/nanobot-p6-completion.rk7wje`:

- `p6-temporary-policy-r1-before`: four reproduced constructor-guard failures.
- `p6-temporary-policy-r2-owning`: five passed, complete owning module.
- `p6-temporary-policy-r2-broader`: 69 passed across five complete modules:
  temporary-session manager/channel lifecycle, session-reader tools, Stage 1
  shutdown, Stage 2 composition, and agent/gateway startup admission.
- `p6-temporary-policy-r2-lint`: PASS.
- Full unchanged strict before and after: zero errors/warnings, 464 files.
- `p6-temporary-policy-r2-compare`: zero introduced diagnostics.
- `p6-temporary-policy-r2-progress.json`: bounded acceptance, source hashes,
  exact one-test-file scope, assertion preservation, and terminal process evidence.

Every accepted run finished without timeout or forced process-group cleanup.
The earlier full suite timed out at 71% and is still not passing evidence; see
[the retained full-suite limitation](p6-summary-replay-checkpoint.md). These
focused passes do not establish a new whole-suite remaining-failure count.
Next work remains the observed failure backlog, followed by a separately
qualified longer full-suite run and the remaining P6 acceptance gates.
