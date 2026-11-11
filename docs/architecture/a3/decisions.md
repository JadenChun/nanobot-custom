# D1–D7: choices and acceptance provenance

For the current planning discussion, use the
[accepted planning direction](plan-review.md#d2d7-direction-from-the-references).
**D1 is now user-selected: Linux and macOS executors, with Android remote.** Linux/Android
verification is confirmed by the user; macOS is the remaining verification target.
D2–D7 use the [reference-based direction](reference-design.md), accepted by the user on
2026-11-11: "yes the recommendation direction is ok". This also accepts retaining the
existing volatile Full-consent restart behavior. The user later explicitly selected all
four first-release defaults by choosing step 1; this is separate from the earlier direction
acceptance and does not establish implementation conformance. The user also selected a
[first release without legacy migration](v1-product-contract.md#selected-first-release-without-legacy-migration).
The [detailed plan is frozen](plan-review.md#frozen-baseline-and-subsequent-work) for later implementation review.

The [original decision record](../../audits/architecture-review-2026-11-06/decisions.md)
and [I05 decision map](../../audits/module-review-2026-11-06/integrated/I05-final-reconciliation.md)
left all seven consequential choices **PENDING** at that historical checkpoint. The later
packets specify an intended implementation sequence, and the pinned candidate implements
choices within it.
The user's current direction acceptance is recorded above and in the planning overlay
of [the evidence ledger](evidence.json). Historical source observations and test results
retain their original scope; they do not establish implementation acceptance.

The selected internal subagent lifetime/budgets, initial device proof and output/retention
policy are recorded with reasons in the
[four-default selection](v1-product-contract.md#final-selections-for-plan-freeze).
The frozen baseline records these choices. Before an implementation merge, attach matching qualifying evidence.
[Review gates](review.md) retain the historical candidate findings for that later phase.

The user's later subagent clarification narrows the interaction scope to control through
the original chat/main agent, supported by the Codex/OpenCode source follow-up. A separate
public child composer, management panel and steer/close/resume API are not required.
The user approved this direction and required aligned client UI behavior. The
[UI contract](v1-product-contract.md#client-ui-behavior-for-subagents) selects original-chat
activity/results, optional read-only detail/return, existing approvals and subtree Stop,
with the same behavior across desktop/WebUI, TUI and Android remote. Runtime defaults
were subsequently selected through the separate four-default decision.

The user also required [future coding-harness and SuperAgent extensibility](future-harness-extension.md).
SuperAgent is coordination between normal sessions, not another subagent layer. Its
future routing, task lifecycle and direct-session UI must leave ordinary subagent
behavior intact. The extension record preserves this boundary without selecting a
future task schema, enabling a coordinator or changing the selected current defaults.

| Decision | Historical status | Candidate observation | Acceptance state in this record |
| --- | --- | --- | --- |
| D1 | Historically PENDING; user-selected 2026-11-11 | Historical candidate observations retain their original source/host scope. | Linux/macOS executors and Android remote selected. Linux/Android previously verified per user; macOS verification remains. Windows is outside the current required matrix. |
| D2 | Historically PENDING Full/private semantics | Server-owned target/proof checks and typed approvals; normal OS effects remain distinct from private-path tools. | Direction accepted: separate consent/profile/enforcement, truthful Full scope and volatile Full consent. Later implementation review retains AR1/AR2/AR6. |
| D3 | Historically PENDING managed sessions | Durable native `delegated` children, runtime metadata v1, finite limits 2/2/4. | Canonical subagent-session direction and main-chat interaction accepted. Awaited default, explicit background work and finite 2/2/4 plus four gateway-wide child runs selected. Later implementation review retains AR2/AR3. |
| D4 | Historically PENDING compatibility/activation | Device-key v2/bearer cutover; compatibility retirement stage and retained readers. | First-release baseline and initial Android Keystore P-256 proof/encoding/challenge contract selected. Version formats and use one live owner; no legacy migration/re-pairing or historical-release reader. Schema/vectors and ordinary recovery need later evidence. |
| D5 | Historically PENDING request/owner lifetime | MCP generation leases, immutable runtime facts, session/process ownership. | Request-owned bindings from definition capture through settlement, with live revocation, accepted as direction. Later implementation review retains AR4/RR2. |
| D6 | Historically PENDING outcomes/recovery | Owner-local effect outcomes, explicit stream terminal, prepared fallback and login generations. | Owner-local recovery, compatible raw outputs and explicit uncertainty accepted as direction. Exact implementation and rollback receipts remain later work. |
| D7 | Historically PENDING output/retention/resources | Bounded output, archive rollover, scheduler proof and parser limits. | Retain finite output/admission limits; no automatic age expiry for regular history, explicit deletion and preservation of unresolved evidence selected. Later implementation review retains RR1/RR3. |

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

**Current planning decision:** Linux and macOS execution plus Android remote, selected
by the user on 2026-11-11. The user confirms Linux/Android verification already exists;
macOS is the remaining verification target. This records a user-confirmed baseline,
not a new matching-head runtime receipt. Windows qualification is outside this required
scope; existing compatibility is not removed. Historical cancelled-Windows evidence
above remains historical and does not create a current platform requirement.

**Later qualification:** name each selected client/executor/mode combination, its confinement,
writer, process-tree and owner-auth guarantees, and its exact artifact/host evidence.
Retain refusal for unavailable guarantees. A3-2, A3-3, A3-7 and A3-9 own the respective
proofs. Focus new verification on macOS and reuse earlier Linux/Android evidence only
where its inputs and boundary still match. These checks follow plan settlement.

## D2: authority and private boundaries

**Accepted direction:** consent, permission profiles and concrete OS enforcement are
separate. Full must truthfully describe OS-account authority and any separately enforced
private-path exclusions. Full consent remains volatile: reopening persisted policy or
history does not restore an old runtime grant.

Historical [D2](../../audits/architecture-review-2026-11-06/decisions.md#d2-full-authority-and-private-boundaries)
distinguished normal-account Full execution from an optional stronger native private
policy; durable restoration was a separate choice. The candidate's [approval and target
mechanisms](design.md#exact-approval-and-live-authority) are valuable, but do not turn
shell command text checks into process confinement. A folder/write scope answers where
a child may act; it does not answer whether a particular risky effect has consent.

**Later implementation gate:** demonstrate Full's actual OS/private-path guarantee and
the accepted non-restoration contract; identify credential-profile and instance boundaries.
Correct AR1
effect reuse, AR2 child consent and AR6 production scope checks before claiming the
corresponding authority invariants. A3-1/A3-4/A3-5/A3-7 own these boundaries. An opaque
approval ID selecting server-owned facts is a valid smaller interface; missing optional
echo fields alone are not an established vulnerability.

## D3: durable delegation

**Accepted direction and current interaction scope:** canonical subagent sessions use the
normal runner/history path, separate active-run identity, narrowed live authority and
bounded admission. The main agent controls them through the original chat. Optional
inspection does not require a separate child chat or management interface. The user
selected awaited default, explicit background work, no automatic restart/replay and
finite configurable 2/2/4 tree limits plus four gateway-wide child runs.

Historical [D3](../../audits/architecture-review-2026-11-06/decisions.md#d3-managed-task-sessions)
recommended bounded roles pending requirements. The candidate instead implements real
child identities, a native backend, dedicated history projections, background continuation and
nested admission. `DelegationTreePolicy` defaults to depth 2, two live children per
parent and four live descendants. Those observations did not establish product acceptance;
the user subsequently selected the stated defaults through the separate product decision.
AR3 shows that these projections lack canonical execution completeness.
Runtime metadata version 1 is likewise an observed persisted contract.

**Selected contract:** native parent-mediated lifecycle, finite budgets and cancellation/
restart rules are recorded in the product contract. Governing instructions and the live
authority intersection remain required. **Later implementation gate:** correct AR2/AR3 using the existing
canonical child execution/source owner. The pinned native backend's mid-run send/steer and
restart-resume refusals remain historical capability observations, not a reason to require
a public control interface. Internal follow-up behavior must be truthful and explicit.
A3-3/A3-4/A3-5 own acceptance, with the selected client UI scenarios and A3-7 Android
alignment. Future external backends require a separate contract.

## D4: compatibility, activation and cutover

**Accepted direction and current scope:** use one current live owner and explicit format
versions. The user selected a first release without legacy migration. Legacy cutover,
forced re-pairing, historical-reader compatibility and two-version upgrade qualification
are outside scope. Initial device proof, restart recovery and revocation remain required.
Future released-format changes can add explicit migration/rollback contracts.

Historical [D4](../../audits/architecture-review-2026-11-06/decisions.md#d4-compatibility-and-retirement)
required tracing registration, callers/exports and persisted/recovery state before
retirement. The candidate chooses P-256 protocol 2 and v1 bearer revocation/re-pairing;
the A3-7 packet originally called the signing choice proposed and required review before
issuance. A code choice does not supply that review's human provenance.

**Selected contract:** initial Keystore P-256/ECDSA-SHA256 proof, SPKI/DER/base64url,
typed canonical claims, one-use challenges and recovery/revocation are recorded in the
product contract. Complete the fixed schema and shared byte vectors before reviewing that
implementation boundary. Later review retains the native Cancel race AR5
and approved-but-unissued restart behavior. No old-installation re-pair path is required.
For pre-release cleanup, trace current callers and stored-state owners; remove obsolete
unreleased paths without creating a second live owner. Historical cutover criteria above
remain audit history and do not override the user's first-release decision.

## D5: request and owner lifetime

**Accepted direction:** retain exact request-owned definitions, bindings and resource
lifetime from capture through settlement, while keeping revocation live. Ready
replacements serve new requests. Never substitute a replacement merely by matching name.

Historical [D5](../../audits/architecture-review-2026-11-06/decisions.md#d5-request-tool-registration-lifetime)
offered deferring reload at a safe boundary or binding each request to its original
registration/connection while keeping revocation live. The candidate stages ready MCP
replacements and retains executing calls. AR4 shows that retaining only at execute-time
does not cover a request that already captured old tool definitions. RR2 shows why
process completion must bind the target incarnation through final turn admission.

**Later implementation gate:** demonstrate the accepted definition-capture-to-settlement rule,
including reload failure, revocation, cancellation and filtered registries. Use existing
owners; do not add a global generation service. A3-6 owns MCP/request identity;
A3-3/A3-8 own queued and restored producer identity and reader compatibility.

## D6: attempt and effect outcomes

**Accepted direction:** preserve compatible raw results with small owner-local recovery
dispositions; distinguish never-started/applied/partial/uncertain effects from acknowledgement.
Reconcile uncertainty before replay and keep cancellation, denial and mandatory-write
failure distinct from ordinary recoverable feedback.

Historical [D6](../../audits/architecture-review-2026-11-06/decisions.md#d6-attempt-outcomes-and-recovery)
preferred a small internal disposition/helper over an unneeded public taxonomy. The
candidate distinguishes mandatory recording, provider terminal, fallback route,
credential generation and WebUI effect outcomes. These facts have different producers.
Ordinary nonzero child exit does not automatically authorize a retry or wider authority.

**Gate:** retain the raw/compatible output fields and prove each owner's pre-effect,
applied, partial and uncertain cases with lost-reply/readback tests. Inspect compatible
state/export or explicit-refusal receipts before accepting a pre-release rollback; no
historical-release executable is required. A3-1/A3-4/A3-6/A3-8/A3-9 own
their parts; one global passing suite is not that acceptance proof.

## D7: completion, retention and resource policy

**Accepted direction:** separate bounded live polling output, retained completion
transcripts, execution capacity and durable evidence retention. Expose loss/provenance,
preserve unresolved evidence and retain existing finite output/admission limits. The user
selected the child budgets and retention of regular history until explicit deletion;
there is no automatic age expiry for regular history in this release.

Historical [D7](../../audits/architecture-review-2026-11-06/decisions.md#d7-completion-and-resource-policy)
left unread-tail versus bounded-transcript/summary output, inferred display completion,
retention and capacity policy open. The candidate adds independent deadlines, bounded
output and durable archive/scheduler machinery. These improve ownership without deciding
how long private evidence should remain or which output view is promised.

**Later implementation gate:** verify the selected output/provenance, finite-limit and
retention/reclaim contracts, including whether audio duration is an enforced server limit; document any authorized qualification
waiver with its source. Correct RR1 and RR3 before destructive maintenance or preview
claims. A3-2/A3-6/A3-8 own limits and data proof; A3-9 owns platform/artifact evidence.
Safe refusal at a known capacity remains an alternative to more complex rollover.
No speed or efficiency improvement is claimed without measurement.
