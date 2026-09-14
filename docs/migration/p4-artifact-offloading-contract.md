# P4 tool-output offloading implementation contract

Status: partially implemented; model-facing offloading and dependency-aware source
summaries are implemented, while runtime activation remains pending. This contract refines the accepted
[context design](../context-management-migration-design.md); it does not authorize
runtime activation or change migration completion scope. Source pressure compaction,
scoped retrieval, and artifact-backed request and summary projections are implemented.
General producer adapters and the remaining operational integration stay pending.

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

## Private blob implementation evidence

`session/artifact_store.py` implements a dormant private byte store with
runtime-selected workspace/session/incarnation/generation binding. It retains the
persisted workspace UUID and hashes session/incarnation strings only as safe
filename components. Its caller must still select and validate the configured
private root against workspace/context-repository locations; no runtime factory
or model tool exposes this constructor yet.

All path traversal uses directory-relative no-follow operations from the filesystem
anchor. Private directories/files require current-user ownership and restrictive
permissions; unexpected types and hard links are rejected. Root identity and
namespace/lock reachability are rechecked. Successful installation uses exclusive
temporary creation, complete writes, file fsync, no-overwrite link installation,
temporary unlink and directory fsync. Existing identical bytes are fully verified
and synced before reuse. Required parent entries are synced even when adopted
from another creator. No descriptor or model reference is published by this API.

Store transactions retain a persistent advisory lock, ordered before any future
source-session transaction. Review isolated a macOS concurrent O_CREAT/O_NOFOLLOW
lock-open race; lookup and exclusive creation are now separate bounded paths.
A crash between blob link and temporary unlink is recoverable only when exactly
one runtime-named staging entry matches the inode and complete descriptor
length/digest/UTF-8 integrity. Arbitrary or corrupt multiply linked files remain
errors. Replacement during installation can leave unreachable private bytes but
cannot be reported as reachable publication success or write through an outside
symlink. The future publisher must call `assert_reachable()` immediately before
its source CAS while retaining the store transaction.

Reads verify the entire immutable blob, including empty output ranges, and return
at most 16,384 raw bytes. Ranges may split UTF-8 code points; the eventual text tool
must implement its documented boundary policy. The dormant store defaults to a
64 MiB per-blob cap; application configuration, aggregate disk quotas, backup
inclusion and cross-platform enablement still require integration. No eviction or
model-visible retrieval is implemented. POSIX no-follow/directory-relative support
is required; other platforms fail closed until an equivalent implementation is
qualified. These platform limits do not enable or disable existing runtime paths.

All 34 blob tests pass, including two spawned concurrent writers, a real process
exit immediately after link installation, adopted-directory durability, symlink
replacement, corruption, strict ranges, root/lock replacement and unsupported
filesystem capability rejection. Independent review accepted the correction bundle
and independently ran those 34 tests. All 394 session tests pass with network
denied; scoped Ruff and diff checks pass. Source-header publication, trusted
provenance and read authorization, effective tool registration, governor/reference
integration and whole-migration acceptance remain unfinished.

## Producer receipt implementation evidence

Ordinary contextual built-in `read_file` text results now carry an immutable receipt
separate from returned content. The receipt covers the exact UTF-8 representation
including decoding, line numbers and pagination footer. A no-follow descriptor
chain qualifies the final text read and verifies object identity and change stamps
through rendering. This claim excludes the earlier legacy format sniff and early
image/empty/dedup paths; those retain existing behavior and receive no receipt.
Unsupported platforms/extra-root namespaces retain inline results without
attestation. PDF/Office/image adapters and general tools remain outstanding.

Locators mirror managed target/context precedence, distinguish target names by
owning context, and include a hash binding to the actual project/repository root.
That binding prevents later retrieval from silently rebasing a relative locator
onto another project. Retrieval still needs to validate it against current policy;
a valid receipt alone is not an access grant.

The registry accepts receipts only from the concrete built-in producer and checks
returned-text integrity. Execution carries them separately to the mandatory source
writer, which reserves `_source_origin`, verifies the persisted text and attaches
provenance to its own allocated event/execution/run/stream envelope. Legacy upgrade
strips this reserved field rather than retroactively authenticating old metadata.
Detached model projections omit it. `attested_artifact_origin` checks receipt/text
and descriptor-origin equality; dormant descriptor publication still does not
require this validator or grant reads. The next runtime publisher/retriever must
require it before emitting or resolving model-facing references.

Tests cover real-engine canonical receipt persistence and repeated provider call
IDs; exact pagination/CRLF/decoding; same-instance concurrent reads; spoofed producer
and altered result rejection; protected paths; managed-root precedence; namespace
replacement; unsupported results/platforms; forged writer fields; and legacy
upgrade. Read-only review found and verified the legacy-upgrade correction.

Validation for producer integration: 556 combined session/filesystem/source-engine
tests and 10 runner tool-execution tests passed with network denied. Two existing
guarded plugin fixtures were deselected; these are not waived for overall migration
acceptance. Scoped Ruff and diff checks passed.

## Invocation-bound retrieval implementation evidence

`ArtifactReader` and explicitly constructed `ReadArtifactTool` now read only
published, producer-attested text under a live dispatch owner, configured private
blob transaction and source transaction. Historical root access selects root
source events in the same session generation; fresh/child readers select their
own stream. Root access does not expose child-private evidence. Explicit
cross-stream grants remain unimplemented.

Every read reconstructs the recorded policy namespace, requires the same root
binding and current managed repository identity, and passes the candidate through
the invocation's built-in file-read policy. Removed/rebound repositories, changed
root classification, protected paths and switched workspaces cannot fall back to
ordinary workspace access. Historical original-file edits do not invalidate stored
evidence. The private blob root is checked against current project, backend,
static/read-only roots, media and exact read-file allowlists on construction and
retrieval. Blob integrity and namespace reachability are verified before return.

Byte ranges accept exact nonnegative integer offsets and 4–16384-byte limits.
Leading UTF-8 continuation bytes are skipped and incomplete trailing characters
remain for the next page. The tool further caps its complete escaped JSON envelope
at 16384 UTF-8 bytes and returns the actual next offset/eof. Missing/corrupt blobs
produce a bounded unavailable error; no source file reread or tool redispatch occurs.
Rematerialization remains a future integration.

A separate validated retrieval receipt travels through the trusted built-in
registry boundary and canonical callback into reserved `_artifact_read` metadata.
Generic events and legacy upgrades strip that marker; malformed receipts leave
inline content intact without a marker. This provides the offloader's future
recursion exclusion, but the offloader is not wired yet.

The runner's optional artifact-reader factory defaults to absent. When supplied,
it constructs an invocation-specific effective registry before budgeting, removes
inherited reader instances, and requires explicit admitted retrieval permission
plus the concrete file-read capability. Root session tool disables propagate into
this admission; child admission inherits that permission and its own capability.
Source restoration now retains admitted/cached SessionPolicy instead of replacing
it with defaults when loading canonical source. No default config, automatic
reference emission or production activation is introduced.

Actual-engine tests qualify root restart retrieval, fresh-history denial and
explicit tool disabling using a descriptor published from an actual earlier
qualified read. Focused tests cover stream isolation, UTF-8/escaped pagination,
policy revocation/rebinding, missing/corrupt bytes, unregistered owners, reset,
namespace replacement, forged markers and current private-root exclusion. Final
combined validation passed 603 tests with network denied; two existing guarded
plugin fixtures remain deselected and are not waived for overall acceptance.
Scoped Ruff and diff checks passed.

## Main-request offloading implementation

With the optional admitted artifact-reader factory present, the source governor
now offloads attested scalar text results longer than 16000 characters before
pressure compaction. It publishes immutable bytes/descriptors using captured CAS,
then renders a deterministic opaque reference, sizes and at most 1200 preview
characters. The canonical result and source IDs remain unchanged. Unsupported
formats/producers and retrieval results remain inline.

`SourceNode.artifact` is an internal typed binding, excluded from messages sent to
the provider. Literal `artifact:` strings never establish this binding. Every
prepare revalidates bound descriptors, producer evidence, current policy and blob
integrity. Existing descriptor IDs/timestamps are reused. A multi-descriptor
operation advances its expected source state only through its own successful
commits; it never reloads/rebases after a CAS conflict. Partially published metadata
can remain without changing canonical source or granting access.

Ledger replacement checks the captured epoch/transcript/raw-list and absence of
outstanding request tickets, updates the ledger/raw list synchronously and clears
usage/provider state. Request-only retry notes remain outside the active ledger.
If the actual request has no retrieval tool definition, authorized bound evidence
returns inline. Storage/CAS failure also falls back inline only after source
access is validated. Policy revocation and owner loss abort; inline restoration
cannot bypass those failures. The complete request still passes the final budget
guard. Restart reconstructs references from canonical source and verified manifests
without publication churn.

### Summary evidence boundary

Legacy evidence remains canonical and never treats references, previews, strings,
or unbound request expansions as source authority. Dependency-aware pressure
summaries may consume a typed projected source node only after reconstructing its
canonical source node and requiring the exact published descriptor, source binding,
attested producer receipt, typed binding, and deterministic reference rendering.
Retrieval results remain canonical inline. Generic checkpoint publication remains
empty-reference-only; version-2 dependency loading and invocation admission are
described below.


## Checkpoint dependency admission and recovery (Package A)

The source loader accepts structurally valid schema-v2 checkpoints with sorted,
unique opaque artifact references, without opening blobs or granting access.
Schema-v1 checkpoints remain empty-reference-only. Generic checkpoint publication
rejects nonempty dependencies until Package B supplies a trusted publication path.

The loop assembles full visible source history, preserving the authorized current
user expansion. The runner registers its invocation reader before selecting a root
checkpoint and before exposing messages to run hooks. Fresh and child invocations
never select root checkpoints. An invocation without an admitted reader skips a
dependent checkpoint and retains full source.

The ledger retains a detached recovery transcript and mirrors subsequent source
and runtime appends into it. Projection changes never overwrite that baseline or
advance eligibility. Each request validates checkpoint dependencies and active
artifact bindings against current manifests, source provenance, scope and policy.
Missing/corrupt bytes or removal of the actual retrieval schema permit expansion
only after independent source authorization. Missing manifests, revoked access,
scope denial and ownership loss abort instead of widening access through inline
fallback.

Recovery checks ledger epoch, exact active transcript/raw-list equality and absence
of outstanding tickets. It synchronously replaces the detached projection and raw
list, clears request/usage/provider caches, resets the accepted boundary, and retains
request-only retry instructions once. Physical checkpoint metadata stays unchanged
for strict CAS; rejected checkpoint fingerprints remain invocation-local. Subsequent
compaction captures explicit accepted state and cannot rediscover a rejected disk
checkpoint. The final request budget still applies to expanded source.

Verification covers dependency admission, missing/corrupt bytes, no-tools recovery,
revocation, manifest loss, stream denial, exact recovery ordering/expansion,
unacknowledged eligibility and pending-ticket protection. Independent review passed
with 116 focused tests. The network-denied broader regression suite passed
657 tests with two existing guarded plugin fixtures deselected; those exclusions
remain outstanding for overall acceptance. Scoped Ruff and diff checks passed. The
production migration guard remains enabled.

## Dependency-aware source pressure compaction (Package B)

Pressure compaction now consumes `SummaryEvidence` from canonical, visible source
intervals and records a contiguous `SummaryDependencyProof` only after each
auxiliary provider call completes. Each selected fold builds one exact auxiliary
message list; the governor estimates that object and
`Consolidator.summarize_prepared_source_chunk` transports the same object without
rebinding tools, conversation state, normalization, or retries. The compatibility
wrapper still builds and preflights dependency-free callers' payloads.

At Package B acceptance, references came only from prior accepted checkpoint
references and canonical, successfully dispatched `read_artifact` receipts in
consumed evidence. The proof reconstructed and sorted that complete set, so later
summaries retained prior references even if model text did not mention them. Typed
projected source nodes remained ineligible until Package C added the separately
verified path below. Strings, prompts, tool arguments, previews, hidden child
records, malformed receipts, unselected chunks, and failed calls still do not
establish dependencies.

Immediately before every auxiliary transport the governor validates dispatch
ownership and the exact captured ledger projection. For a fold with dependencies,
it additionally opens the configured private blob transaction, then the execution
and source transaction, and rechecks canonical proof data, current source scope,
active reader admission, retrieval schema, manifest membership, source policy,
durable bytes, and namespace reachability before releasing locks for transport.
Dependency-free folds retain reader-free behavior. If the effective main request
lacks the exact `read_artifact` schema, dependent compaction is skipped with the
canonical source intact and the final main-request budget guard decides whether the
request fits. Revocation, manifest, scope, ownership, receipt, namespace, or blob
failure aborts before the next auxiliary transport; it never falls back to inline
publication.

Root publication with references uses schema v2 and the configured
`SourceSessionStore` bound to the reader's backend and private store. It revalidates
the full proof while holding blob then execution/source transactions, commits with
the captured CAS, and updates the ledger, raw payload, provider state, and admission
fingerprint only from the returned committed snapshot. Empty-reference checkpoints
remain the generic schema-v1 path. A missing proof cannot downgrade covered
retrieval evidence or a prior dependent checkpoint to schema v1; source conflicts
or write failures leave the existing projection unchanged without retrying or
regenerating a summary.

Isolated fresh and child streams may create local dependency-bearing checkpoints
after the same proof and current-authorization validation. They never mutate durable
root checkpoint metadata. Their source comparison is exact for the admitted stream
while retaining the existing allowance for unrelated root or sibling appends.

Focused verification covers exact prepared payload identity, reader/schema and
no-transport failure paths, schema-v2 root commits, multi-fold revocation before a
second provider call, final CAS failure, local proof scope, reference carry-forward
and deduplication, direct runner admission, and a real migration-engine sequence:
attested read, actual retrieval receipt, pressure compaction, schema-v2 checkpoint,
and a new loop invocation that admits it before transport. The focused command
covering checkpoint, source-compaction, summary, projection, and integration tests
passed 104 tests. The required network-denied session and agent regression command
passed 685 tests with the two existing guarded plugin fixtures deselected.

## Dependency-aware projected source summaries (Package C)

The artifact projector now exposes the deterministic binding and rendering used by
both main-request preparation and summary validation. A projected summary node is
accepted only when its canonical source event, descriptor manifest membership,
descriptor/source binding, attested producer receipt, typed binding, and rendered
message all match exactly. Missing manifests, altered bindings or previews,
unsupported producer evidence, recursive retrieval results, hidden children, and
unbound expansions do not establish projected dependencies.

`SummaryEvidence` records only the projected bindings in each consumed closed
interval. Proof reconstruction carries references from those consumed bindings,
canonical retrieval receipts, and any prior accepted checkpoint; it never infers a
reference from model text, an artifact string, a preview, or an unconsumed suffix.
Preflight and publication retain Package B's current-reader, policy, durable-byte,
lock-order, and captured-CAS validation. Auxiliary summary work never publishes
descriptors or blobs.

A projected dependency requires the active `read_artifact` schema and reader. If a
previously projected node must be restored inline because that schema is absent or
the artifact has become unavailable, pressure compaction is skipped for that
request and the final request-budget guard evaluates canonical source. A proofless
publication cannot downgrade projected evidence to schema v1. Root and local
dependency checkpoints remain schema v2 only when the full proof validates.

Focused verification covers deterministic projected evidence, forged bindings and
renders, missing manifests, recursive and unsupported evidence, canonical legacy
evidence, unbound expansions, proof-reference derivation, unconsumed suffixes,
schema/reader loss, corrupt/revoked/CAS failures, no auxiliary artifact writes,
and the actual-engine large-read projection through restart admission and
retrieval. General producer adapters, cross-stream grants and handoff, artifact
rematerialization, migration-guard removal, and runtime activation remain
outstanding.

The combined focused checkpoint, projection, compaction, and engine suite passes
166 tests. The broader network-denied session and governance suite passes 658 tests
with three migration-guard-only fixtures explicitly deselected while the production
guard remains closed. Scoped Ruff and diff checks pass.

## Invocation-only cross-stream grants and foreground handoff (P4-E)

`ArtifactGrantSet` is a process-local, nonserializable authority bound to one
admitted reader, private store, execution owner, and invocation stream. A parent
may issue at most 16 unique opaque references only to a direct child stream. The
issuer verifies native descriptor membership, current policy, durable bytes, and
the blob-then-execution/source lock order before minting a grant. Issuance changes
neither source events nor reserved source metadata. Received grants are exact
descriptor bindings; a child cannot regrant borrowed parent evidence.

`ArtifactReader` continues to admit native root or own-stream evidence first.
For every non-native read it requires an active matching grant, then repeats the
same descriptor, source-policy, manifest, generation, namespace, and blob checks.
Writer failure, closed grants, changed direct-stream relation, source reset,
revocation, missing publication, or blob corruption all deny retrieval.

Child runs receive an internal `handoff_artifact` selector only when an admitted
reader is registered for their direct child stream. It selects only child-produced
artifacts and creates a sealed receipt after a durable child tool-result event and
a successful child assistant terminal event. A concrete foreground delegation,
planning, review, or exploration wrapper may forward that receipt only to its
direct parent stream. The parent records its outer tool result before adopting the
receipt; adoption revalidates the child evidence and mints an upward grant. The
parent gets a runtime-only notice containing the available opaque references, so
source events and metadata retain no grant or handoff authority.

Focused coverage verifies direct parent-to-child reads, sibling and transitive
denial, bounded and unique inputs, source-write invariants, receipt
nonserializability, failed and unrecorded handoffs, borrowed-artifact rejection,
one-shot upward adoption, concrete-wrapper forwarding, durable outer-result
ordering, and runtime-only parent notices. General producer adapters, artifact
rematerialization, migration-guard removal, and runtime activation remain
pending.

## Private artifact-store configuration (P4-F)

Artifact storage is now an explicit, disabled-by-default top-level setting. A
legacy configuration that omits `artifacts` remains lossless and does not probe,
create, or register any artifact capability. This applies equally to CLI,
headless gateway, and GUI startup paths: `read_artifact` is never a global tool.
An enabled loop adds it only to an admitted source invocation with the concrete
`read_file` capability, and a session-level tool disable removes that invocation
capability.

```json
{
  "artifacts": {
    "enabled": true,
    "root": "/srv/nanobot/state/context-artifacts",
    "maxBlobBytes": 67108864,
    "maxNamespaceBytes": 1073741824
  }
}
```

`enabled` is a strict boolean. `root`, when present, must be an expanded,
absolute, non-root directory. If it is omitted, the loop captures
`<the loaded config directory>/context-artifacts` once during construction; a
later config-path/global-path change cannot relocate that loop's artifact root.
The standard limits are 64 MiB per blob and 1 GiB per source namespace, and the
namespace limit must be at least the blob limit.

Before an enabled factory constructs a private store, it rejects roots that
overlap the workspace, session backend, active project, managed context or target
repositories, built-in product checkout, media root, readable/writable roots, or
an exact read-file allowlist. It also rejects symlinked components and any Git or
worktree ancestor. An unsafe enabled configuration rejects loop startup in CLI,
headless gateway, and GUI paths; it is never relocated to another directory. The
factory freezes that approved-location policy and rechecks it before construction
and every artifact read, verification, write, or publication. A later `.git`
directory or worktree file therefore fails the affected artifact operation closed
without weakening the store's directory-relative no-follow operations.

After a safe configuration has started, an operational capacity or disk failure
retains the validated inline request representation through the existing offload
fallback when it fits the final request budget. It does not silently select a new
artifact root. Unsupported filesystem platforms similarly leave the capability
absent.

Namespace capacity is checked while the existing store transaction lock is held.
Trusted immutable blobs and retained staging entries count conservatively; an
identical verified blob does not consume capacity twice. Capacity pressure never
evicts or deletes bytes. `maxBlobBytes` admits only a new blob installation, while
descriptor validation, reads, durable verification, checkpoint admission, and
identical verified reuse remain valid for already published immutable evidence.
Lowering either quota can prevent new writes while existing evidence remains
readable, even if existing namespace usage is above the new limit.

Artifact bytes are private runtime state. Operational backups must include the
configured artifact root together with its bound session state, under the same
restricted access policy. A session `export()` is a transcript/export boundary;
it does not copy private blobs or turn artifact references into portable content.
Until an explicit materialization/export feature is qualified, restore the
artifact root with its sessions rather than treating ordinary session export as a
complete artifact backup. The production migration guard remains closed pending
the remaining migration acceptance gates.
