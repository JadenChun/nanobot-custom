# P3 authentication/API work in progress

Reviewed bounded corrections: HTTP API delegates session serialization exclusively to process_direct; request timeout includes admission waiting; empty output returns the existing fallback without rerunning the turn; stream:true remains 400. Codex requests no longer retry certificate failures with verification disabled. Legacy expiry parsing treats expires/expires_at numeric values as absolute seconds or milliseconds; only explicit expires_in is relative.

Validation: 68 combined API/Codex tests passed using synthetic tokens and local aiohttp fixtures; the earlier 41 Codex transport/auth tests also passed with network denied. Scoped Ruff passed. Reused read-only provider analyst found no scoped regression; requested admission-timeout/lock-reuse and explicit refresh-expiry fixtures were added and passed. These tests do not establish complete production auth wiring.

The production auth gate is still open. Current Codex production calls bypass the legacy importer; Copilot needs one-way import into canonical storage. Atomic writes, one refresh owner across chat/catalog/image/usage/login/logout, canonical-path compatibility, and logout/import suppression remain required. An async user question asks whether to keep valid Nanobot credentials first, import only when absent, save refreshed credentials only to Nanobot, and require explicit reimport after logout. Do not infer an answer from elapsed time. Official-store writeback and account/logout policy depend on the answer.

Additional reviewed risks: non-object credential JSON, error payload sanitization, conflicting accounts, failed two-store persistence, and independent official-client writers. Use synthetic stores only. Full migration remains guarded and incomplete.
