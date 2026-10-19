# P3 authentication/API work in progress

This is a historical work log. Earlier open/pending statements are preserved
as review evidence and are superseded by the current closure status below.

Reviewed bounded corrections: HTTP API delegates session serialization exclusively to process_direct; request timeout includes admission waiting; empty output returns the existing fallback without rerunning the turn; stream:true remains 400. Codex requests no longer retry certificate failures with verification disabled. Legacy expiry parsing treats expires/expires_at numeric values as absolute seconds or milliseconds; only explicit expires_in is relative.

Validation: 68 combined API/Codex tests passed using synthetic tokens and local aiohttp fixtures; the earlier 41 Codex transport/auth tests also passed with network denied. Scoped Ruff passed. Reused read-only provider analyst found no scoped regression; requested admission-timeout/lock-reuse and explicit refresh-expiry fixtures were added and passed. These tests do not establish complete production auth wiring.

Historical pre-closure status (superseded): the production auth gate had not closed. At that time Codex production calls bypassed the legacy importer; Copilot needed one-way import into canonical storage. Atomic writes, one refresh owner across chat/catalog/image/usage/login/logout, canonical-path compatibility, and logout/import suppression remained required. An async user question asked whether to keep valid Nanobot credentials first, import only when absent, save refreshed credentials only to Nanobot, and require explicit reimport after logout. At that checkpoint, official-store writeback and account/logout policy depended on the answer.

Additional reviewed risks: non-object credential JSON, error payload sanitization, conflicting accounts, failed two-store persistence, and independent official-client writers. Use synthetic stores only. Full migration remains guarded and incomplete.

Current P3 closure (2026-09-29): the authentication and API adapter contracts
are accepted. Canonical Nanobot credentials win; official/legacy import occurs
only when canonical credentials are absent and no logout tombstone exists;
writes remain canonical; logout tombstones suppress automatic reimport.
Conflicting accounts require an explicit generation-checked switch or
reimport, never a merge. POSIX and native Windows persistence are implemented,
although native Windows execution, ACL, and `FileRenameInfoEx` qualification
still require P6 CI. See [the P3 closure checkpoint](p3-closure-checkpoint.md)
for evidence and retained limits. This closes the P3 policy question only: the
global migration guard remains active and no runtime activation, live
credential use, release qualification, or deployment is authorized.
