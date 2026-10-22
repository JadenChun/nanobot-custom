# P3 legacy fallback checkpoint

Legacy `agents.defaults.fallback` now selects a dedicated adapter; upstream `fallback_models` retains its separate policy. Validation rejects simultaneous nonempty lists before provider construction, and signatures distinguish policy plus all destination transport settings. Shared preset resolution also feeds snapshot capacity.

The adapter caches candidates, tries configured ready order followed by cooling expiry, and applies the exact staging quota/rate classifiers and five-minute cooldown. Any nonquota candidate stops traversal. Persistent mode retains candidate retries but cannot replay the whole chain. Direct calls remain primary-only; explicit generation overrides are preserved, and configured generation reaches every cached candidate at construction and reload. Empty legacy chains are rejected.

Continuation and compaction state are detached and tagged per actual candidate. Resume checks use the destination model, not the primary override. Capacity is selected after candidate selection; session/events are retained. Delivered content/reasoning/tool deltas stop failover. Existing upstream retry notification capture is reused, with transition clearing independent of configured candidate index.

Validation: 188 provider checks passed with clean HOME/XDG settings and network denied. Exact command/environment, artifact hashes, and case list are adjacent. Scoped Ruff passed. The reused read-only provider analyst reviewed the adapter and narrowly rechecked corrections to constructor generation, classifier parity, and empty-chain handling without remaining findings in that scope. No real provider calls were made.

This is bounded P3 acceptance, not full migration completion. Remaining work includes authentication/API/channel dispositions, durable context and session migration (P4), canonical runtime integration (P5), and build/copied-state rehearsal (P6). The runtime guard remains enabled. A standalone import-order fixture also exposed upstream Config forward-reference initialization when providers load before schema; full-runtime import-order validation must cover that during P5.
