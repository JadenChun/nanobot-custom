# Source-discovery provenance and verification

This publication preserves module/relevant-body discovery at one frozen source pin. Its
[validation record](provenance/publication-validation.json) reports document/source checks,
not application tests. The source comparison, architecture discussion and bounded detailed
audit remain pending in the [continuation contract](continuation-plan.md).

## Pins, denominator and evidence depth

Current nanobot source is `27adced0aacd55d61791bb88a7240d7ad7bd5b75`, tree
`d49aca97c9a34a2d17eae53d672fa591652457f1`, repository
[nanobot-custom source](https://github.com/JadenChun/nanobot-custom/tree/27adced0aacd55d61791bb88a7240d7ad7bd5b75).
The denominator is exactly its 2,455 tracked paths; discovery documents added afterward
are excluded. Every baseline path has one primary workstream/family/role assignment.
P01 is cross-cutting and adds no primary assignment. The eighteen-stream view consists
of seventeen primary streams plus P01; nine families group those assignments.

Prior discovery was `4e2bf4d8d0b34f133c01591b67826ca956fed384`, tree
`d5cc262acda5fb8051ffac9996ee9ab189c361cd`, with 2,467 paths. The histories diverge at
`b94b12527b014785c1454a04ef3b8138ee6070a5`: seven previous-only and eleven current-origin-only
commits. [baseline.json](baseline.json) records every divergence commit and both trees;
[prior-baseline.json](provenance/prior-baseline.json) retains the original baseline.
Historical planning at b94b1252 used 2,451 paths. Neither historical count is current.
Missing prior local fixes are source differences, not automatic regression grades.

The [ledger](path-inventory.tsv) retains path, workstream, family, role, bytes, current
SHA256, previous SHA256, change, inherited/current report membership and depth. Membership
columns contain semicolon-separated relative coverage-JSON paths. Empty previous hash
means no prior path. Exact source roles are `implementation-or-tooling-source` (878)
and `runtime-instruction-source` (41). Other roles are accounted separately: 997 tests,
54 assets/locks, 290 configuration/support and 195 generic documents.

Three depth rules apply in order:

1. `current-relevant-sections-inspected`: current refresh's exact inspected-path union
   (93 paths/85 source targets), plus completion's exact assigned/inspected union (139
   source targets). The completion assignments are disjoint and equal the frozen refresh
   source-gap list. Incidental completion `extra_context_paths` earn no additional credit.
2. `prior-relevant-sections-carried-forward`: at least one exact historical inspected-list
   membership, identical previous/current SHA256 and no current depth. There are 808 such
   paths, including 695 source targets. Prior hashes remain portable in the ledger.
3. `mapped-only`: assignment/navigation without retained relevant-body evidence. There
   are 1,415 such support paths and zero source targets. Mapping support is not deep review.

Thus current depth is 232 paths/224 source targets, carried depth 808/695, and mapped depth
1,415/0. All 919 source targets have discovery evidence; no exhaustive semantic audit,
line-by-line review or runtime/stability qualification is asserted. Large-file relevant-body
limits and original condensed-report uncertainty are retained.

## Reports, supersession and portability

The [report index](reports/README.md) contains nine historical family reports, three current
refresh reports and three completion reports, each paired with exact coverage JSON.
Historical Markdown is an edited readable edition of coordinator-condensed notes, preserving
factual claims, uncertainty, anchors and original scope. It is not a reconstructed full
transcript or a new read. Completion analyst prose is retained with navigation/source links.
[report-manifest.json](provenance/report-manifest.json) records original input and published
hashes. Coverage JSON files are copied byte-for-byte; their exact lists and original retention
metadata remain authoritative. Original command strings/assigned checkout labels are
historical provenance, not required current input locations.

Read current map → refresh/completion → relevant report. Current refresh claims govern
changed paths; historical claims apply to their old pin. Inherited unchanged paths can link
to current-source permalinks because those bytes are identical, explicitly retaining the
original inspected pin. A changed historical path's old line reference is historical text;
its current behavior has a refresh responsibility row. Prior commits are local-only relative
to current staging, so essential current verification does not require fetching them from a
remote. Exact current bytes, previous hashes and report membership establish inheritance.

All 47 changed current implementation paths have current refresh evidence and distinct
responsibility rows in the [differential checkpoint](refresh-2026-11-06.md#all-47-changed-implementation-responsibilities).
There are 95 changed current paths (four additions included) and 16 deleted prior paths.
[changed-paths.tsv](provenance/changed-paths.tsv) retains the exact differential record;
[coverage-summary.json](coverage-summary.json) retains deleted paths outside the denominator.
[refresh-coverage-summary.json](provenance/refresh-coverage-summary.json) preserves the
original 85 current/695 carried/139 mapped-only source checkpoint. Final depth is never
inferred by changing that historical checkpoint in place.

The [portable reference index](provenance/source-reference-map.json) preserves pinned
DSH/Codex path/hash/URL metadata and existing bounded excerpts, with retained-copy dependencies
removed. It performs no new comparison or review. REF6 remains mapped rather than fully
reviewed. The reference pins are unchanged by this publication.

Historical optional local evidence was retained under a dated nanobot-harness audit-artifact
root, with project-plan, parallel-plan, architecture-reference, execution-hold, goal-alignment,
source-audit-method, codebase-discovery and refresh/completion subdirectories. Temporary
reader worktrees and source clones were execution conveniences. No current discovery input
requires those locations. Checkpoint machinery, copied product sources, reproducible patches,
runtime data, probe logs and executable verification scripts are intentionally outside this
Git publication.

## Independent standard-Git and standard-library checks

From the repository root, standard Git reads can identify the baseline without importing
or executing repository modules:

```bash
git rev-parse 27adced0aacd55d61791bb88a7240d7ad7bd5b75^{tree}
git ls-tree -rz --name-only 27adced0aacd55d61791bb88a7240d7ad7bd5b75
```

An independent verifier can parse `path-inventory.tsv` with `csv.DictReader`, compare its
unique path set with the NUL-delimited Git tree, and hash exact source blobs with
`hashlib.sha256`. For example, this reads one baseline blob without importing the package:

```python
import hashlib
import subprocess

pin = "27adced0aacd55d61791bb88a7240d7ad7bd5b75"
path = "nanobot/agent/runner.py"
blob = subprocess.run(
    ["git", "show", f"{pin}:{path}"], check=True, capture_output=True
).stdout
print(hashlib.sha256(blob).hexdigest())
```

For a full deterministic check:

1. Compare all 2,455 paths and hashes/sizes with baseline blobs (or working bytes after
   confirming that checkout matches the pin). Confirm one primary assignment per path,
   zero duplicate/unassigned rows, nine families and seventeen primary streams plus P01.
2. Parse all fifteen exact coverage files. Recompute current refresh union, disjoint
   completion assigned/inspected union, and every row's exact current/inherited membership.
   Require a historical inspected membership and equal previous/current hash for every
   inherited row. Do not count completion context extras.
3. Recompute depth/role/family/stream counts and compare with coverage summary; require
   224 current/695 carried/0 mapped-only source, 47/47 changed implementation paths, and
   completion counts C00 43/H01 14/H02 35/K01 45/R01 1/D01 1. Confirm 139 distinct primary
   responsibility rows, 137 valid primary line anchors and two zero-byte package markers.
4. Check report input/publication hashes against the manifest and original frozen input
   hashes when available. Exact coverage JSON hash equality attests list retention;
   readable Markdown editing is separately disclosed and needs prose review.
5. Resolve relative Markdown links against their containing documents. For current
   GitHub source permalinks, require the exact pin, valid baseline path and a line in
   `1..len(blob.splitlines())`. Empty markers get path links without line anchors.
   Do not treat absent historical remote objects or external network availability as
   current discovery gaps; current anchor checking uses local pinned source bytes.
6. Compare the historical harness Decision heading and all following bytes with the frozen
   input. The payload after literal `## Decision` (including following newlines) has SHA256
   `654f4c367fbf7bb0ddc698122ecd0326e8fb288da94171d554493123fa3c181c`; heading plus payload
   has SHA256 `75a8b177693f48d8c2cd97c58600f7c473e4a3208b1ca722c9fd619f497ce96e`.
   Only its preceding intro may change. Confirm that all writes are owned audit
   Markdown/JSON/TSV documents and no product/support baseline bytes changed.

The [validation JSON](provenance/publication-validation.json) records these structural
results and the writer's no-runtime-execution declaration. Hashes prove byte identity,
not semantic completeness or stability. Coordinator independently reviews the actual
reports, source anchors, status language and scope before accepting publication.

## Continuation and rollback

The [project audit plan](../project-audit-plan.md) owns decisions and later stages;
[continuation-plan.md](continuation-plan.md) supplies package ownership/dependencies,
acceptance criteria and deterministic validation. Sandbox remains first in the later audit.
No product rollout or data migration occurs. Correct or restore owned documents if a
publication check fails, preserving historical evidence and unrelated work. An explicit
source advance requires a new frozen delta and affected-caller reads; never silently
refresh hashes or reinterpret old claims as new inspection.
