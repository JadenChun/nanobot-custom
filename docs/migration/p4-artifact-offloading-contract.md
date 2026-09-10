# P4 tool-output offloading implementation contract

Status: pending implementation. This contract refines the accepted
[context design](../context-management-migration-design.md); it does not authorize
runtime activation or change migration completion scope. Source pressure compaction
is implemented separately; artifact-backed evidence is not implemented yet.

## Required resulting behavior

Keep every canonical text tool result inline and unchanged through the supported
rollback window. Replace only detached request text with actionable opaque
`artifact:<id>` references. Initial thresholds are 16,000 characters per text block
and at most 1,200 preview characters; bytes are measured separately for integrity
and storage. Preserve other blocks. Images, private reasoning and arbitrary
objects are outside text offloading.

Artifacts require a private store and a dedicated scoped `read_artifact` tool.
Do not reuse workspace result buckets or media storage: those expose paths and
lack source-bound authorization and publication. Never grant general filesystem
read access to the artifact root. An opaque ID is a lookup key, not a capability.

## Implementation packages and dependencies

1. **Private storage and source metadata.** Implement immutable text blobs and
   strict descriptors, then reserved source-header artifact metadata with owned
   source-state CAS. Reuse the session store's persisted workspace namespace ID;
   its path-derived `SourceState.store_identity` is a storage binding, not a second
   portable workspace identity. Bind descriptors to workspace, session incarnation
   and generation, event, execution, stream, run, block selector, MIME type, UTF-8
   byte length, digest, timestamp and trusted source-access label. Use random IDs
   and `(event_id, block_selector)` idempotency with digest verification.
2. **Trusted provenance and retrieval.** Capture provenance at the tool execution
   boundary, not from result text or a shared accumulated touched-path set. Start
   with filesystem reads while implementing explicit adapters for other producing
   tools. Register bounded artifact reads through normal capabilities and dispatch.
   Unknown-origin results remain inline until their adapter exists; this is a
   staged implementation boundary, not completion of general tool offloading.
3. **Projection and summary integration.** Offload after canonical result recording
   and before pressure preparation. Preserve SourceNode/event/execution identity,
   replace only authorized text content, invalidate the projection epoch and
   request usage, synchronize the live raw list, and retain retry-only notes
   outside the active projection. Do not emit references when the effective role
   lacks artifact retrieval. Summary inputs may use verified artifact projections;
   extend checkpoint reference validation deliberately rather than bypassing the
   current empty-only artifact-reference contract.
4. **Integrated qualification.** Verify actual-engine retrieval, pressure summary,
   restart, rollback, delegation grants, revocation and crash behavior. Keep the
   migration guard until these and the remaining migration requirements pass.

## Storage and publication invariants

The configured private state location must be outside product/context repositories
and media. Namespace by persisted workspace identity, incarnation and generation.
Document the default path, override rules and backup inclusion before enablement.
Use private permissions, strict identifier syntax, trusted directory descriptors
and no-follow checks for every component. A resolve-then-open sequence is
insufficient against replacement races. Document platform support and fail closed
where equivalent safe operations are unavailable; retaining inline source remains
available subject to the final request budget.

Write immutable bytes using exclusive creation, fsync, atomic installation and
directory fsync. Under artifact-store lock followed by source session transaction,
publish the descriptor with captured source state and execution ownership. Only
then render a reference. No lock spans provider work. CAS failure leaves an orphan
and retains inline projection; do not silently refresh CAS or append another event.
Existing bindings/blobs must match identity, digest and byte length exactly.

Reserve artifact metadata against ordinary metadata updates. Reset invalidates
active-generation authorization immediately. Automatic deletion/eviction remains
disabled in the first package. Never delete reachable evidence under disk pressure.
No canonical event or prefix digest changes because a result is offloaded.

## Retrieval and propagation invariants

Use bounded byte or line ranges with one documented UTF-8 boundary policy. Reject
bool/noninteger/negative offsets, enforce a hard output cap, and validate integrity
before returning content. Resolve only published descriptors in the current
admitted workspace/session/incarnation/generation and authorized stream/grants.
Recheck current source policy, including repository removal/revocation. Persist a
trusted source locator and access class; removed repositories must not fall back
to unrestricted ordinary paths.

Children receive their own artifacts; parent/sibling references require explicit
invocation grants. Parents receive child artifacts only through a trusted result
handoff. Grant propagation and general-tool provenance are implementation gates.
Retrieval results carry trusted origin metadata and cannot recursively offload;
model-controlled strings do not establish this exemption. Retrieval still obeys
the finalized request budget.

Runtime computes summary dependency sets from verified source bindings and prior
checkpoint references. Model summaries cannot grant access or author descriptors.
Checkpoint publication validates dependencies against published, currently
permitted artifacts. Retaining inline source alone does not make reference-only
summary input safe without this dependency validation.

## Failure and acceptance evidence

Failures retain inline source. Continue only if the full resulting request fits;
otherwise fail without main transport or redispatching the producing tool. Missing
or corrupt artifacts yield bounded typed errors; rematerialization may use matching
retained canonical text under current authorization, never rerun side effects.
Restart reconstructs only published verified references. Rollback disables reference
projection and reads retained inline source.

Required tests cover identical provider call IDs with different executions; mixed
blocks; unchanged source digests; idempotency conflicts; forged references; cross-
session/generation/child/sibling denial; revoked repositories; traversal and symlink
replacement races; crash points before/after descriptor publication; missing/corrupt
blob recovery; bounded retrieval and recursion exemption; pressure summaries using
references; explicit grants; restart and rollback. Telemetry contains IDs, sizes,
status and latency, never result text, previews, source paths or prompts.

## Descriptor implementation evidence

`nanobot/session/artifact.py` now implements strict version-1 text descriptors,
filesystem-origin locator shapes, opaque reference parsing, and canonical source
binding. It verifies the exact workspace/session/incarnation/generation, event,
execution, run and stream, actual assistant call/result stream provenance, scalar
or text-block selection, UTF-8 byte length and SHA-256. Repeated provider call IDs
do not collapse distinct execution/event bindings. Parsed locators are recorded
policy inputs, not trusted grants; this module performs no file access, descriptor
publication, policy authorization or reference projection.

The codec rejects extra/missing fields, unsupported MIME/version/provenance classes,
noncanonical relative locators, invalid opaque IDs, malformed timestamps/counters
and mismatched source content. Independent review corrected intent lookup to ignore
non-call rows carrying unrelated `_execution_ids`; regressions cover both forged
lists and null fields. The review recheck independently passed 60 artifact tests.
All 360 session tests pass with network denied; scoped Ruff and diff checks pass.

Immutable blob storage, owned source-header publication, provenance adapters,
retrieval policy/grants, governor/checkpoint references and integrated offloading
acceptance remain unimplemented. This evidence does not authorize activation.
