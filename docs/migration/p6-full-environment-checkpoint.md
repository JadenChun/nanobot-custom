# P6 full host dependency closure — strict source gate failed

2026-10-02: **host all-extras/channel installation PASS; strict typing FAIL;
full P6 NOT_ACCEPTED**. Root independently verified input integrity and repeated
dependency consistency. This diagnostic checkpoint is not source-repair acceptance,
a production profile or activation authority. P5/P6-A and earlier host qualification
remain accepted at their historical scope. D9 stays NOT_AUTHORIZED; P7/H1/P8
are not delivered. No runtime source or guard changed.

## Actual checks and provenance

All ten extras and23 requirement strings from17 packaged channel manifests resolve
into210 public PyPI wheels on macOS26.6.2 arm64/Python3.13.2. Acquisition was
separate, wheel-only and private-config-free; no sdists/build hooks, browser
installation or application activation. A fresh empty environment installed offline
with `--no-index --require-hashes` under OS network denial. `pip check` passed
originally and on an independent root repetition.

Root verified all wheel hashes/sizes/METADATA/names/versions/pins, installed versions
(210 frozen distributions plus bootstrap pip), requirements hash,1,865 frozen
source files and1,868 pre-publication candidate baseline files. Requirements SHA256:
`7cdcb70efb5e97a46919fa7893b734bd7ef6682884090aaca1146e18c49cfc46`.
The accepted P6-A artifact is unchanged. Only this checkpoint, its machine evidence
and three status documents change after the baseline check.

[Machine evidence](p6-full-environment-evidence.json) binds22 durable raw records/
helpers by path/hash/size under private bundle
`/Users/jaden/Documents/github/nanobot-harness-candidate/.assets/releases/p6-full-environment.xcmuciq3`.
It retains exact commands/environments/OS profiles,210 wheel METADATAs/pins,
original/root pip checks, complete diagnostics and source/upstream inventories.
Records/helpers and wheels are retained0400. The initial helper's wrong archive-root
assumption is preserved separately; the resume helper verified actual `source/source/`
bytes before typing. Preparation failure is not a green initial run or product fix.

## Strict source verdict

BasedPyright1.40.1/pyright1.1.414 ran with unchanged strict Python3.11 policy:

| Frozen source | Files analyzed | Errors | Missing imports | Exit |
| --- | ---: | ---: | ---: | ---: |
| Candidate | 463 | 5,970 | 0 | 1 |
| Pinned upstream `f47b8f23155ddec23cfb80ec001b544d495df89c` | 326 | 0 | 0 | 0 |

Upstream came only from candidate Git objects. Root verified its inventory and equal
checker policy/argv/environment/OS profile. Missing optional SDKs do not explain the
remaining failure. This does not prove5,970 independent runtime bugs or attribute
every error to one stage. The partial environment's7,420 errors/56 missing imports
remain historical, not retroactively green.

## Bounded root-first repair scope — not implemented

A fresh read-only GPT-6.1 Sol High assessment identified three packages:

1. **Session declarations and journal shapes:** `session_mode.py` has822 direct
   diagnostics;638 arise at29 repeated journal keyword-expansion sites. Type existing
   constructed dictionaries with a matching `TypedDict`, and preserve lock-decorator
   signatures with `ParamSpec`/`Concatenate`. Keep serialized values, lock order and
   transaction flow unchanged. Validate unchanged registry/conversion/operations/
   export and checked-source ownership tests, including crash/no-overwrite assertions.
2. **Recovery evidence typing:** `memory_recovery*.py` has1,494 direct diagnostics.
   Throw-only helpers incorrectly annotated `-> None` defeat narrowing; truthful
   `Never`/`NoReturn` is one root correction. Other parser/storage/context-manager
   contracts need concrete checked types. Preserve phases, byte identity, leases,
   atomic durability and corruption refusal; run unchanged recovery/receipt/cursor/
   legacy-codec tests.
3. **Retained provider/desktop boundaries:** legacy fallback/pool/key-pool/desktop
   files have571 direct diagnostics. Repair parent-compatible provider signatures
   and typed callbacks/leaves; declare only used PyObjC APIs at their owning edge
   and validate with mocks. Preserve fallback/state/budget/stream assertions;
   no fallback-policy rewrite or real desktop operation.

One concrete public-call mismatch needs regression tests: legacy fallback `chat`/
`chat_stream` reject positional messages allowed by their parent; the pool also
narrows base positional parameters. Conversely, the export descriptor's `locals()`
guard and nonempty pool construction explain some apparent unbound warnings;
diagnostics alone do not authorize new recovery policies. No repair/test ran here.

These packages own2,887 direct diagnostics; eliminating all leaves3,083 elsewhere
before measuring cascades. This is substantive source qualification, not a tiny fix.
Keep the unchanged whole strict gate and compatibility assertions; no strictness
reduction, blanket suppression/casts or persistence redesign. Actual repairs require
a new source freeze/artifact qualification.

## Remaining release inputs and authority

Zero-error typing would still not close P6. Native Python3.11/3.14, Windows/Linux
and actual production-target evidence is absent. The owner must identify the
production platform/dependency/channel/plugin/crawler profile, approved old immutable
rollback release/environment, and truthful stopped-state/reference-config/backup/
artifact-root set. Actual loader source policy, old-reader/restore/export and
positive isolated service rehearsal remain unqualified. Synthetic fixtures cannot
replace those identities; resolver success does not prove browsers or channels.

No live config/state/credential read, conversion, provider/channel/listener or
background activation, guard removal, deployment or publication occurred.
Full P6 remains **NOT_ACCEPTED**, not completed by this host checkpoint.

## Subsequent source repair wave

[Bounded root-first repairs](p6-type-root-repair-checkpoint.md) are now accepted
with root verification and fresh independent review. Source typing declarations
and provider positional parity changed; no persistence/locking/recovery policy or
guard changed. Root409 session/recovery and207 provider cases pass in isolation.
The unchanged full strict gate still FAILS with4815 errors, net1155 fewer and
zero newly introduced signatures outside owned files. The earlier5970 result and
"not implemented" assessment above remain historical. Full P6 remains NOT_ACCEPTED;
a new corrected-source artifact and all remaining release gates are still required.
