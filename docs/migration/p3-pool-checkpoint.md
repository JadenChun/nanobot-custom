# P3 credential pool and streaming checkpoint

The factory now composes immutable OpenAI-compatible credential leaves with shared admission, preference, and cooldown coordination. Actual attempts are observed once. Pool exhaustion preserves structured errors, stops another base pool pass, and reports one terminal notification with actual attempt count. Outer fallback defers notifications until its chain finishes.

Both native continuation fields have detached owner tags; reconstruction rejects old instance handles and retains portable messages for rebuilding. This follows the migration requirement not to persist provider handles. Durable history is still P4 work.

All externally delivered content, reasoning, and tool-call deltas prohibit replay in base retries, single leaves, pools, fallback (standard/persistent/direct/context), and the compat Responses-to-Chat transport switch. The callback API still accepts on_stream_recover, but delivered timeout errors no longer invoke segmented replay. Two upstream retry tests were intentionally updated to the retained fork contract.

Validation: 165 checks passed in a clean HOME/XDG environment with network denied; invocation, case list and hashes are adjacent. A subsequently added concurrent request-isolation regression passed with all 71 pool tests. Scoped Ruff passed. Reused read-only provider analyst reviewed the corrections and found no remaining concrete no-replay hole; its requested entry-point coverage was added and passed. All provider calls were fixtures.

This is bounded P3 progress, not full migration acceptance. Legacy fallback policy, other P3 authentication/API/channel dispositions, P4 durable context, P5 canonical runtime wiring, and P6 rehearsal remain. Public runtime activation remains guarded.
