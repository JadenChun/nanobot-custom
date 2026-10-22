# P2 adapter checkpoint

Implementation remains isolated in `nanobot-migration-integration`, based on staging plus pinned upstream merge `0b5739aa`. This is partial P2 implementation, not migration acceptance or a runnable release. The public activation guard remains enabled.

Implemented contextual tool dispatch, complete-batch policy checks, terminal/error outcome propagation, skipped observations and cancellation propagation through upstream AgentRunner. Foreground delegation uses immutable LLMRuntime settings and owned upstream MCP connections. Restored managed filesystem policy, write scopes/locks, protected names, bounded search and `/workspace` alias resolution through upstream containment. Browser/device read-only properties and serial execution are compatible. MCP annotation classification and external skill precedence are restored. Session read projection and run-history helpers are additive; durable storage/CAS migration remains P4.

Clean-home, network-denied validation passed 245 tests, with 38 deselected. Cases and raw-evidence hash are recorded in the adjacent validation files. Seven new actual-runner tests cover policy preflight, terminal propagation, cancellation, three fail-fast error representations and crawler runtime capture. Earlier broader runs hit two plugin-loader tests blocked by the activation guard. Historical MCP connector fixtures still call the old AsyncExitStack API; those need differential fixtures against the new owned-connection API. Historical files and P0/P1 evidence remain unchanged.

A reused read-only High analyst reviewed dispatcher/runner/delegation boundaries. Its two findings—lost tool-error terminal metadata and inflated crawler total context capacity—were corrected and narrowly rechecked. This is reduced-isolation review, not fresh full P2 acceptance. Implementation is direct under the user's revised authorization after native worker capacity failures.

Remaining P2 work includes canonical factories and tool exposure, context-repository prompt integration, remaining process/web/message/image adapters and their negative tests. P3 provider/channel ports, P4 durable context, P5 canonical turn integration and P6 rehearsal are still required. No live state, deployment or original-checkout promotion has occurred.

## Context assets follow-up

Restored `TOOLS.md` and `IDENTITY.md` from the agent workspace while retaining upstream project-scoped `AGENTS.md`. ContextBuilder accepts the existing context manager/ordered roots, includes repository entrypoints and policy summaries, and preserves supplemental memory/run summaries. External skills retain workspace-first, ordered-library-before-builtin precedence and disabled filtering. The upstream test requiring TOOLS.md omission was intentionally updated to the accepted fork contract; historical fork test copies remain unchanged.

Validation: `pytest tests/agent/test_migration_context_assets.py tests/agent/test_context_builder.py -q` under network-denied sandbox: **56 passed**. Scoped Ruff and diff whitespace checks pass. This follow-up does not wire ContextBuilder dependencies into AgentLoop; that remains canonical integration work.
