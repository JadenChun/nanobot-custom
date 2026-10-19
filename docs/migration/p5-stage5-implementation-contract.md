# P5 Stage 5 bounded implementation contract

2026-10-02: **ROOT TECHNICAL PASS for bounded preflight/report/rollback/P6-P7
handoff; full Stage5 closure NOT_ACCEPTED, P5 NOT_DONE; activation NOT_AUTHORIZED**.
This is an implementation/handoff package, not activation approval. Authority is
the [integration plan](p5-integration-activation-plan.md), accepted D1–D8 and
[Stages 1](p5-stage1-checkpoint.md), [2](p5-stage2-acceptance-checkpoint.md),
[3](p5-stage3-acceptance-checkpoint.md), [4](p5-stage4-acceptance-checkpoint.md).
D9 remains unaccepted. Root accepted the bounded package after independent
evidence and fresh GPT-6.1 Sol review; implementation has no self-PASS authority.

## Exact package and boundaries

The sole writer owns `_migration_preflight.py`, `cli/migration_preflight.py`,
the two Stage5 preflight test files, this contract, the
[permanent validation manifest](p5-stage5-validation-manifest.json),
[rollback/export runbook](p5-rollback-export-runbook.md),
[P6/P7 handoff](p6-p7-handoff-manifest.md) and
[closure checkpoint](p5-closure-checkpoint.md).
Root explicitly authorized documentation-only publication of the integration plan
top status/links and upstream migration plan dated append. All other files, existing
tests/conftest, guards, state machines, custom dirty/untracked work and prior
checkpoints remain frozen. No installs, commits, live data/credentials, deployment,
conversion, activation or native/Windows qualification. Max two correction rounds;
both are now used: bundled pre-freeze boundaries and final deterministic TTL/
operator-reference-config/closure-evidence correction. No optional expansion.

## Delivered report contract

Standalone `python -m nanobot.cli.migration_preflight` accepts only explicitly
supplied `--copy-root --config --workspace --sessions-root --runtime-data-root`
and `--stopped-writer-copy`. No normal CLI import, home/default/env configuration,
environment expansion, installer, runtime startup or admission is invoked.
Use absolute lexical non-root paths beneath a strict explicit copy boundary.
Resolve macOS temporary-directory aliases before supplying paths; the command
does not resolve symlinks for you. Workspace/runtime must exist as disjoint safe
directories; config parent equals runtime, sessions equals runtime/sessions.
The operator assertion is not proof that an old writer stopped.

Pinned no-follow directory descriptors, regular single-link config, bounded
config reads, no special files/symlinks/hardlinks in inventoried state, and
snapshot rechecks reject unsafe or changed evidence. Limits: 262144 config bytes,
4096 aggregate scanned entries, depth16. Only configuration content is read;
session/recovery/cron/artifact content and embedded paths are never opened.
Configured context roots are not traversed. No lock, constructor, repair,
mkdir/chmod/fsync, publication, provider/channel runtime, subprocess or network.
Filesystem reads may update atime; content, authority, inode and mtime are not
mutated. This is not an atomic hostile-writer filesystem snapshot; the stopped
copy and rechecks are required, not a claim that locks exclude old binaries.

The fixed fifteen checks cover copy boundary, structural config, full schema,
session evidence, recovery, artifact policy/state, channel metadata/dependencies,
cron definitions/runtime, background dormancy, platform, P6 release, D9.
Only fixed enums and exact bounded entry counts appear; no paths, IDs, values,
digests, credentials, content, raw errors or caught exception logs.
Missing/duplicate/unknown/ill-typed reducer evidence is NOT_READY. Synthetic
complete evidence can test reducer READY, but never grants runtime authority.
CLI exits0 only READY,1 NOT_READY,2 invalid command syntax; errors are parseable
deterministic sanitized JSON. No flag supplies or waives P6/D9 evidence.

Root approved a safer narrow realization: Config eagerly imports tool modules
while rebuilding its schema. This report uses stdlib structural checks only and
does not build an alternative full schema. `config_structure=PASS` means explicit
copied paths/types/aliases satisfy this adapter, not complete Config validation.
`config_schema=UNQUALIFIED/P6_REQUIRED` always remains separate; P6 owns full
init-only Config validation with no environment/default fallback, binding the
source to the explicit copied config parent. Explicit false Heartbeat is required,
including when schema defaults currently resolve false. Omission/true/coerced
booleans and nonzero TTL fail; no configuration is rewritten. Runtime timer,
Dream, Heartbeat, idle, AutoCompact, callbacks, rematerialization and retention
remain disabled and Stage4 negative construction evidence is still required.

Clean absent/empty owning inventories can PASS. Any retained session namespace,
workspace sessions or `.nanobot` binding/witness evidence is UNQUALIFIED, never
a claimed authoritative-mode count. Any memory state is conservatively retained;
receipt/quarantine inspectors that can follow persisted sessions_root are not used.
Workspace/cron, runtime/cron and the entire runtime/cron-runtime parent are checked;
copy-changed identity never hides old ledger/cursor/lease state. Retained state is
UNQUALIFIED, not a declaration of lease free/active. The actual default artifact
root runtime/context-artifacts plus legacy runtime/artifacts are inventoried;
explicit artifact roots must remain in the copy, not overlap workspace/session
authority. Enabled artifacts/policy not fully safely qualifiable are UNQUALIFIED;
no artifact store or configured context path is opened.

Channel packaged names are reserved before manifest success. Dependency-free
first-party descriptors load directly, never discovery's exception logging,
runtime, connector or validation callbacks. Selected entry points are metadata
only; duplicate, missing, malformed and reserved shadow entries fail. Legacy
empty dependency metadata, extras/URL or unresolved topology are UNQUALIFIED,
never falsely available. Installed distribution version checks are availability
only; actual channel config/import/extras qualification is P6. Unsupported
platforms stay UNQUALIFIED. Current clean report is therefore necessarily
NOT_READY with activation NOT_AUTHORIZED (full config, P6, D9 remain open).

## Acceptance and evidence

AC1 fixed reducer/report/CLI; AC2 explicit containment/no-follow/bounds/snapshot/
env isolation/no authority mutations; AC3 named inventories/dormancy/plugin and
dependency metadata; AC4 complete rollback/handoff/guard proposal and classified
historical nodes; AC5 red-green and independent exact regression manifest/scope;
AC6 fresh review/root acceptance and honest nine-statement closure.

Focused red under exact full OS denial yielded two missing-module collection
errors before implementation. Initial focused green23, expanded39, bundled
boundary correction45 tests passed; final focused selection46 passed. Scoped Ruff,
in-memory compilation of all four Python files and diff checks pass. Root independently
ran the fifteen unchanged-tree finite regression gates; all required selections
green, historical16fail61pass unchanged exact nodes/zero additions. Root final new
focus57passed0.49s and scope/review PASS; worker never duplicates broad gates.
Temporary supplementary evidence lives at `/tmp/nanobot-stage5.9h2qn6`;
raw evidence is not a substitute for the permanent commands/classifications.
The manifest preserves exact split full-denial/local-process nodes and separate
lease profile, cold SDK and Stage4 focus; unsplit substrate/P3/primitive commands
are not rerun as required-green gates. No count from overlapping selections is
summed. Historical cron16 and Stage2 old108+22 full-network-denied nodes are
classified individually, not deleted or turned green by fixture weakening.
Classification is not preserved-behavior coverage:116 compatibility nodes require
the [finite positive closure-evidence mapping](p5-closure-checkpoint.md#next-finite-closure-evidence-gate-not-implemented-by-stage5)
before root can close §7. Uncovered assertions belong to the exact prior owning
stage, not automatically P6.23 environment/P6 nodes and8 historical conflicts
remain distinct. Technical Stage5 acceptance is separate from full P5 DONE.
The bounded assertion-level closure-evidence audit is next before P6, not a
generic refactor or automatic transfer of preservation requirements.
The final TTL correction's11 regressions first produced5failures/6passes;
deterministic ordered extraction now rejects duplicate aliases, booleans,
ill-typed and negative TTL in both structure and dormancy. Worker final focused
selection57passed0.48s under exact full OS denial; scoped Ruff/in-memory compile/
diff checks pass. Root hashseed1/2 probes both reject structure/dormancy; fresh
GPT-6.1 Sol Medium review CORRECT then narrow PASS resolved all3 findings.
Root scoped Ruff/in-memory compile/diff/27 new-doc links PASS; implementation
scope1849 baseline files byte-identical,9 additions,0 existing changes/removals.
Exact independent outcomes and qualifications are permanently recorded in the
closure checkpoint/manifest; no further correction budget remains.
Operator rehearsal uses exact copied-root arguments and approved reference-config
loader/path/environment qualification as detailed in the rollback runbook; the
copied inspector config cannot bypass the existing equality/samefile exclusion.

Exact DOCX attachment node remains NOT_RUN/environment-blocked/P6; python-docx
is absent. BasedPyright absent/unqualified; native Windows, installed/native and
immutable release qualification remain P6. No dependencies installed. Guard is
byte-identical. Root technical acceptance does not close P5 or authorize D9.
