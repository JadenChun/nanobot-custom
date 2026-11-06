# Source-comparison architecture review — 2026-11-06

**Recommendation: retain nanobot's small loop/runner and canonical source ownership, then
correct narrow admission, authority and lifetime boundaries.** The pinned source supports
incremental improvement. It does not justify adopting another harness, adding persistent
task sessions, widening platforms or retiring compatibility code without discussion.

This checkpoint records **33 targeted comparison rows across REF1–REF6** in four reports.
**Source-comparison checkpoint accepted:** the independent correction recheck and coordinator
content acceptance passed for this targeted comparison. Consequential choices remain **PENDING**
in [decisions.md](decisions.md); no runtime or full semantic-audit acceptance is inferred.
No application code, tests, runtime instructions, schemas, dependencies or configuration changed.

| Report | Reviewed mechanism slice |
| --- | --- |
| [Admission and requests](admission-requests.md) — A1–A8 | REF1/REF2: canonical admission, steering, registration/request lifetime, source tickets, checkpoints, effect ordering and recovery |
| [Persistence and lifecycle](persistence-lifecycle.md) — P1–P9 | REF3/REF5: durable acknowledgement, leases/shutdown, process deadlines/completion/output/retention, cron receipts, retained GitStore and display provenance |
| [Authority and Sandbox](authority-sandbox.md) — S1–S9 | REF4: server-owned proofs, Once/Full, backend/refusal paths, native ownership, grants, Full guarantees, attempt outcomes/status and remote owner authentication |
| [Delegation and context](delegation-context.md) — G1–G7 | REF6: bounded roles versus addressable sessions, admitted project/write scope, instruction/skill admission, cancellation, capacity and intentional legacy fences |

## Preserved strengths and design direction

Nanobot already separates coordination from model/tool iteration and places capabilities at
explicit tool/channel/skill/MCP edges. The source comparison supports preserving:

- Canonical source/execution admission with distinct lease, session/incarnation/generation,
  request epoch and call identities; mandatory recording precedes acknowledgement and effects.
- Portable source checkpoints, bounded recovery, accurate completed/skipped tool recording
  and conservative serialization where policy or mandatory writers require it.
- Server-owned scope, nonserialized authority references, durable grant publication,
  launch/stdin rechecks, private masks and refusal without required confinement/supervision.
- Cancellation-resistant ownership, unknown-effect recovery and separate computation,
  display, queued delivery and qualified receipt boundaries.
- Awaited foreground roles, narrow non-elevatable tool registries, explicit artifact grants,
  immutable request runtime and distinct role budgets.

Keep one owner per invariant. Use a small correction at the actual boundary, rather than
adding another loop, scheduler, persistence owner or general framework. These are recommendations
grounded in the reviewed mechanisms, not newly accepted product architecture.

## Improvement roadmap

Order below is audit/correction sequencing, **not a severity grade or accepted repair**.
High static confidence means the cited conditional source chain is strong; frequency,
deployment exposure and runtime effects were not measured. The first later audit remains
Sandbox, even though independent conditional candidates are already concrete.

| Order / candidate | Condition, implication and confidence | Smallest next action / dependencies |
| --- | --- | --- |
| First audit — **C1 capability/status truth** | [S3/S8](authority-sandbox.md): current managed shell/supervisor/owner IPC have distinct platform limits; environment-declared enforcement differs from local availability. High source confidence, deployment truth unknown; no primary-UI false claim established. | AUDIT-S02-01 reachable caller/capability/error matrix, then narrow docs/status corrections preserving legacy fields. D1 governs new platforms. |
| Next conditional boundary — **C2 mandatory follow-up failure** | [A8](admission-requests.md): transient before-write follow-up failure can poison the source writer, be swallowed as optional injection and allow continuation/final completion on the old ledger. High static confidence; retained input/republish protect against the claimed loss/replay hypotheses. | H01-01 caller reconciliation; propagate mandatory-writer failure while preserving optional/legacy callbacks. Preserve acknowledgement/compaction/effect fences. |
| Next conditional boundary — **C13 yielded hard deadline** | [P3](persistence-lifecycle.md): finite yielded timeout without notification sink or later poll/stop/shutdown lacks an independent timer. High static confidence against the advertised hard timeout; no runtime verification. | S02-02 after the first matrix: one process-lifetime deadline owner, independent of notifications, preserving tree cleanup/authority retention. |
| Next conditional boundary — **C14 foreground skill admission** | [G4](delegation-context.md): custom disabled/read-denied skills can re-enter foreground prompt metadata/body because construction/updates omit main disabled state/read policy. High source confidence; no observed disclosure. | One S01/K01/T01 owner for construction/live-update graph; pass admitted state/policy, preserving bundled exemption and non-elevatable roles. |
| Next conditional boundary — **C3 tool registration lifetime** | [A3](admission-requests.md): active MCP reload/recovery can replace a shared registry binding after schema capture, or close a wrapper retained by a filtered registry. High static possibility; frequency unknown, no authority bypass demonstrated. | T01-01 plus D5: defer replacement at an established boundary or reject changed registration identity while keeping revocation live. |
| Next retained-caller boundary — **C7 GitStore staging scope** | [P8](persistence-lifecycle.md): reachable retained/public auto-commit with unrelated staging can include it under a permitted Dulwich 0.25.2 implementation. Strong version-specific static evidence; canonical manual Dream is fenced, installed version/consumers unknown. | K01-01 caller inventory, then refusal before mutation or selected-tree/index isolation. D4 gates retirement, not audit. |
| Bounded usability — **C15 project/write-scope root** | [G2](delegation-context.md): an authorized selected project outside manager home resolves files against a different root than relative worker scope, causing denial. High static confidence; no escape. | Use one admitted project basis for prompt/scope/tools, preserving grants/symlinks; agree stable-or-reject focus mutation with REF4. |
| Contract discussion — **C4 Full guarantee/instructions** | [S6](authority-sandbox.md): Full shell has normal OS effects while context can imply retained private protection. High confidence in the mismatch of boundaries; intended guarantee pending, no exploit shown. | D2 before text/enforcement changes: describe accepted normal-account semantics accurately or separately specify retained native isolation. |
| Bounded outcome contract — **C5 attempt facts** | [S7](authority-sandbox.md): formatting/heuristics combine unavailable, runner and child outcomes. High source confidence, field frequency unknown. | S02/H01 internal attempt distinction preserving raw output/legacy fields; D6; no automatic widening or error for every nonzero exit. |
| Bounded recovery clarity — **C6 recovery disposition** | [A7](admission-requests.md): bounded correction exists but exclusions are scattered/text-based. High confidence; maintenance benefit unmeasured. | A small internal helper/type under the same attempt-to-recovery owner as C5; preserve public ToolResult and existing limits. |
| Persistence efficiency — **C8 ordered async journal** | [P1](persistence-lifecycle.md): synchronous durable SQLite on async callers can block. High call-chain confidence, no measured current latency. Normal Codex local writes await flush; they are not queue-only. | A01-03 process journal first, retaining durable-before-launch acknowledgement/order/error/shutdown; broad source persistence separate. |
| Completion contract — **C9 output view** | [P5](persistence-lifecycle.md): foreground polls consume shared buffers that a final completion report also reads; canonical tool history may retain those bytes. No conversation-wide loss established. | D7 promised view first, then a bounded transcript/cursor if required; one output owner. |
| Resource policy — **C10 settled-record reclamation** | [P6](persistence-lifecycle.md): all-record cap can safely refuse later launches without an inspected public reclaim path. High source confidence; desired retention/privacy policy pending. | Separate settled inspection/export/reclaim from unknown effects; D7; do not copy reference caps. |
| Local efficiency — **C11 manager lock scope** | [P6](persistence-lifecycle.md): start/cleanup await process work under the manager lock; close_all already demonstrates outside-lock kill with failed-owner reinsertion. Source-based contention risk only. | Narrow with explicit reservations under the single process-lifetime owner; keep cap/failed-close obligations. Separate from C10. |
| Presentation contract — **C12 inferred completion** | [P9](persistence-lifecycle.md): response history can synthesize display completed state without canonical terminal/receipt proof. High mechanism confidence; UI promise pending. | Preserve useful repair; D7 decides provenance/terminal presentation contract before public schema changes. Distinct from C2. |
| Instruction contract — **C16 governing subset** | [G3](delegation-context.md): roles get assignment/catalog/skills but not main project/profile bodies. A governing constraint omitted from assignment may be absent; missing every profile is not automatically defective. | Define admitted minimum/precedence with T01/K01, keeping isolated transcript/artifact grants; do not clone all memory by default. |

The [backlog](audit-backlog.md) turns these into bounded source-only packages, records all
eighteen stream dispositions, and identifies shared write/invariant dependencies. A later
code fix needs its own exact ownership, accepted contract, tests and branch. Disjoint report
reads can proceed independently; shared source writers, error contracts, process lifetime
or public schemas need one owner or an accepted prerequisite before parallel repairs.

## Source pins

| Source | Frozen identity / role |
| --- | --- |
| Nanobot | [27adced0aacd55d61791bb88a7240d7ad7bd5b75](https://github.com/JadenChun/nanobot-custom/tree/27adced0aacd55d61791bb88a7240d7ad7bd5b75), tree `d49aca97c9a34a2d17eae53d672fa591652457f1`; application baseline |
| DSH | [fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159](https://github.com/deepseek-ai/deepseek-harness/tree/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159), tree `6cccb3ee8c1501c9d623258a660f7413761dd0f0`; engineering reference, developer-preview snapshot |
| Public Codex | [6ff1e1ca3b30fc8b914720681fcf5fe62e95f856](https://github.com/openai/codex/tree/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856), tree `03f3b46689faf274f126dc78508ed238564da8ba`; published `0.166.0` / `rust-v0.166.0`, not private hosted/desktop implementation |
| Dulwich supplemental source | [287097f0514783d35e2a427bb4451ab3f804aee8](https://github.com/jelmer/dulwich/tree/287097f0514783d35e2a427bb4451ab3f804aee8), peeled `0.25.2`; two verified implementation blobs support allowed-version Git index semantics, not installed-version attestation |

DSH/Codex are engineering references, not identical products, feature-parity targets or
operational reliability certifications. Hermes supplies secondary product-direction awareness
only; no Hermes codebase review, mechanism comparison, source pin or reliability claim is
included. The separate historical SuperAgent prototype is not new implementation evidence.

## Scope, provenance and continuation

The [discovery checkpoint](../codebase-discovery/completion-2026-11-06.md) remains complete:
**2,455 paths; 919 source targets = 224 current + 695 identical-hash carried; zero source
gaps**. Those counts describe module/relevant-body coverage and are unchanged. This package
uses connected targeted body reads, actual active caller/configuration paths and explicitly
separate search-only paths; it does not exhaust all branches, providers, adapters, clients,
build/support files or native enforcement. The earlier map's REF6-mapped status is historical;
G1–G7 now provide the bounded comparison, without implying a full project semantic audit.

[provenance.json](provenance.json) retains pins, exact per-package reads/search scope and
input integrity. [validation.json](validation.json) records the exact standard-Git and
Python standard-library document/blob checks. The compensating pre/post check is structural:
before, this review package was absent and protected bytes were hashed; after, required
reports/links/anchors/metadata exist and protected bytes must match. A conventional product
red/green cycle is inapplicable because this package changes only source-review documentation.

No repository modules/scripts, tests, builds, type/lint checkers, dependency operations,
applications, services, provider calls, process exercises or benchmarks ran. Source/link/hash
checks establish structural integrity, not semantic correctness, runtime stability, measured
efficiency, deployed dependency/plugin/configuration truth, OS enforcement, crash behavior
or actual delivery. Historical runtime results remain historical.

Continue from this accepted checkpoint, the [decision record](decisions.md) and
[AUDIT-S02-01](audit-backlog.md#audit-s02-01--first-later-audit), after explicit
acceptance/deferral of relevant choices. Preserve the frozen discovery reports,
JSON/TSV manifests and historical harness Decision payload. Any staging advance needs a named
delta with affected caller reinspection; never silently refresh a pin, expand coverage counts
or infer acceptance from documentation publication.
