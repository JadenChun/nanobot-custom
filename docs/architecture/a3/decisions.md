# D1–D7: choices and acceptance provenance

The [original decision record](../../audits/architecture-review-2026-11-06/decisions.md)
and [I05 decision map](../../audits/module-review-2026-11-06/integrated/I05-final-reconciliation.md)
left all seven consequential choices **PENDING**. The later packets specify an intended
implementation sequence, and the pinned candidate implements choices within it.
Neither implementation, an analyst recommendation, a passing check nor publication of
these documents proves human acceptance. No missing approval is invented here.

This record can be completed without settling those choices now. Before the affected
code merge, record a durable decision with the selected scope, rationale, authority,
date, compatibility consequences and exact qualifying evidence, or explicitly defer that
behavior and keep it unavailable. [Review gates](review.md) also require corrections
where the candidate violates its own packet invariants.

| Decision | Historical status | Candidate observation | Acceptance state in this record |
| --- | --- | --- | --- |
| D1 | PENDING platforms/executors | Portable process paths coexist with Linux-specific managed confinement, native proof and POSIX writer admission. | Supported matrix incomplete; full Windows cancelled; macOS executor/owner proof remains unqualified. |
| D2 | PENDING Full/private semantics | Server-owned target/proof checks and typed approvals; normal OS effects remain distinct from private-path tools. | Full contract and credential/profile authority require a durable decision; AR1/AR2/AR6 block current assurances. |
| D3 | PENDING managed sessions | Durable native `delegated` children, runtime metadata v1, finite limits 2/2/4. | Human choice provenance absent; AR2/AR3 block the promised ownership/consent contract. |
| D4 | PENDING compatibility/activation | Device-key v2/bearer cutover; compatibility retirement stage and retained readers. | Removal and cutover dispositions unaccepted; current-reader checks alone do not prove old-executable compatibility. |
| D5 | PENDING request/owner lifetime | MCP generation leases, immutable runtime facts, session/process ownership. | AR4 demonstrates request binding is incomplete; RR2 demonstrates a producer without incarnation binding. |
| D6 | PENDING outcomes/recovery | Owner-local effect outcomes, explicit stream terminal, prepared fallback and login generations. | Preserve compatible fields; exact outcome and rollback acceptance still require receipts. |
| D7 | PENDING output/retention/resources | Bounded output, archive rollover, scheduler proof and parser limits. | Policy values and waivers lack human provenance; RR1/RR3 block deletion and dry-run assurances. |

## D1: platform and executor scope

Historical [D1](../../audits/architecture-review-2026-11-06/decisions.md#d1-supported-executor-and-client-platforms)
offered retaining current limits, adding selected native capability, or using a Linux
executor for other clients. The candidate contains Windows Job/POSIX process handling,
but its Linux native supervisor, Bubblewrap confinement, POSIX writer lease and
`SO_PEERCRED` IPC are separate boundaries. A supported browser, TUI or Android client
does not establish the executor's capabilities.

Inspected hosted jobs at `0518de5f` passed Linux Python 3.11/3.14, dedicated Windows
process checks and several client/build checks. The full Windows job was cancelled.
The independent macOS review could not pass owner IPC's peer-credential check. That is
not a Linux runtime failure and does not qualify macOS. The separately published Darwin
repair requires [reconciliation](review.md#repair-branch-reconciliation).

**Gate:** name every advertised client/executor/mode combination, its confinement,
writer, process-tree and owner-auth guarantees, and its exact artifact/host evidence.
Retain refusal for unavailable guarantees. A3-2, A3-3, A3-7 and A3-9 own the respective
proofs. A smaller supported matrix remains a viable choice.

## D2: authority and private boundaries

Historical [D2](../../audits/architecture-review-2026-11-06/decisions.md#d2-full-authority-and-private-boundaries)
distinguished normal-account Full execution from an optional stronger native private
policy; durable restoration was a separate choice. The candidate's [approval and target
mechanisms](design.md#exact-approval-and-live-authority) are valuable, but do not turn
shell command text checks into process confinement. A folder/write scope answers where
a child may act; it does not answer whether a particular risky effect has consent.

**Gate:** state Full's actual OS/private-path guarantee and any restart restoration
contract; explicitly identify credential-profile and instance boundaries. Correct AR1
effect reuse, AR2 child consent and AR6 production scope checks before claiming the
corresponding authority invariants. A3-1/A3-4/A3-5/A3-7 own these boundaries. An opaque
approval ID selecting server-owned facts is a valid smaller interface; missing optional
echo fields alone are not an established vulnerability.

## D3: durable delegation

Historical [D3](../../audits/architecture-review-2026-11-06/decisions.md#d3-managed-task-sessions)
recommended bounded roles pending requirements. The candidate instead implements real
child identities, a native backend, dedicated history projections, background continuation and
nested admission. `DelegationTreePolicy` defaults to depth 2, two live children per
parent and four live descendants; these are source-observed values, not accepted product
defaults. AR3 shows that these projections lack canonical execution completeness.
Runtime metadata version 1 is likewise an observed persisted contract.

**Gate:** record why durable/addressable children are required rather than awaited
roles, the chosen profiles and finite budgets, governing instruction subset, authority
intersection and cancellation/restart semantics. Correct AR2/AR3 using the existing
canonical child execution/source owner. The native backend explicitly refuses mid-run
send/steer and restart resume; a client control must surface that limit truthfully.
A3-3/A3-4/A3-5 own acceptance. Future external backends require a separate contract.

## D4: compatibility, activation and cutover

Historical [D4](../../audits/architecture-review-2026-11-06/decisions.md#d4-compatibility-and-retirement)
required tracing registration, callers/exports and persisted/recovery state before
retirement. The candidate chooses P-256 protocol 2 and v1 bearer revocation/re-pairing;
the A3-7 packet originally called the signing choice proposed and required review before
issuance. A code choice does not supply that review's human provenance.

**Gate:** approve algorithm/encoding/canonical bytes, consumption and recovery behavior,
v1 revocation, re-pairing and rollback; resolve the native Cancel race AR5. Qualify the
approved-but-unissued restart and retained v1 installation re-pair paths named in the
[review](review.md#qualification-and-decision-gates). For each A3-10 candidate choose
retain, forward, isolate offline or remove with caller and stored-state evidence.
Keeping a narrow historical reader or a gated feature remains a viable smaller option.
Search-only absence of in-tree callers is insufficient.

## D5: request and owner lifetime

Historical [D5](../../audits/architecture-review-2026-11-06/decisions.md#d5-request-tool-registration-lifetime)
offered deferring reload at a safe boundary or binding each request to its original
registration/connection while keeping revocation live. The candidate stages ready MCP
replacements and retains executing calls. AR4 shows that retaining only at execute-time
does not cover a request that already captured old tool definitions. RR2 shows why
process completion must bind the target incarnation through final turn admission.

**Gate:** choose the smallest rule that covers definition capture through settlement,
including reload failure, revocation, cancellation and filtered registries. Use existing
owners; do not add a global generation service. A3-6 owns MCP/request identity;
A3-3/A3-8 own queued and restored producer identity and reader compatibility.

## D6: attempt and effect outcomes

Historical [D6](../../audits/architecture-review-2026-11-06/decisions.md#d6-attempt-outcomes-and-recovery)
preferred a small internal disposition/helper over an unneeded public taxonomy. The
candidate distinguishes mandatory recording, provider terminal, fallback route,
credential generation and WebUI effect outcomes. These facts have different producers.
Ordinary nonzero child exit does not automatically authorize a retry or wider authority.

**Gate:** retain the raw/compatible output fields and prove each owner's pre-effect,
applied, partial and uncertain cases with lost-reply/readback tests. Inspect exact prior
executable and export receipts before accepting rollback. A3-1/A3-4/A3-6/A3-8/A3-9 own
their parts; one global passing suite is not that acceptance proof.

## D7: completion, retention and resource policy

Historical [D7](../../audits/architecture-review-2026-11-06/decisions.md#d7-completion-and-resource-policy)
left unread-tail versus bounded-transcript/summary output, inferred display completion,
retention and capacity policy open. The candidate adds independent deadlines, bounded
output and durable archive/scheduler machinery. These improve ownership without deciding
how long private evidence should remain or which output view is promised.

**Gate:** state output/provenance promises, numeric limits, archive/reclaim policy and
whether audio duration is an enforced server limit; document any authorized qualification
waiver with its source. Correct RR1 and RR3 before destructive maintenance or preview
claims. A3-2/A3-6/A3-8 own limits and data proof; A3-9 owns platform/artifact evidence.
Safe refusal at a known capacity remains an alternative to more complex rollover.
No speed or efficiency improvement is claimed without measurement.
