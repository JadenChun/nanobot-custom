# P6 IN_PROGRESS — P6-A ACCEPTED

2026-10-02: **P6 IN_PROGRESS — P6-A ACCEPTED / root PASS / fresh review PASS**.
AC1–AC5 are accepted for this bounded immutable source/artifact stage, with no
actionable review findings. Root final publication recheck is not yet recorded.
This is a bounded artifact stage,
not full P6 acceptance. P5 remains DONE/ACCEPTED; full P6 **NOT_ACCEPTED**,
runtime **NOT_READY/NOT_AUTHORIZED**, D9 **NOT_AUTHORIZED**; P7/H1/P8 are not
delivered. No guard/background/default/activation policy changed.

## Source and retained exact bytes

The authoritative source is the entire dirty candidate, not HEAD alone:
HEAD `0a323e591ac48ae4da1472db8d47f74d7c7157d0`, branch
`codex/harness-migration-candidate`. The complete dirty status and per-file
SHA-256/size/mode inventory are in [permanent evidence](p6-artifact-evidence.json).
The fresh private bundle is
`/Users/jaden/Documents/github/nanobot-harness-candidate/.assets/releases/p6-a.Xq8IyRdH`.
All 1865 tracked/nonignored source files include local additions, docs, deploy
scripts, templates, custom skills, frontend and shared client-event inputs.
Tracked ignored `.DS_Store` is nonruntime source metadata, not a package member;
it explains the difference from the root's 1864-file `rg` baseline. Runtime,
secret, cache, dependency, release-output and `.git` trees are not source inputs.
Symlinks/special files are rejected before content reads; no protected original
workspace, live config/state, credentials, or old rehearsal input was consumed.

Corrected source archive SHA-256:
`fa582770bac7d9dd328558082f874ccefdf6eb1d5a069d51a8ea8194b85c4378` (7293993 bytes).
Source manifest SHA-256:
`2a7a25356f9f8ef6de52eaa40ce4e10ac2f04f773f5cc1c2d60b45b1ddadcdd2`.
Source/snapshot and before/after candidate/build-copy bytes matched. The source
archive, inventory, ignored-lock provenance and both release archives are retained
read-only (0400) beneath the private bundle; no overwrite or deletion occurred.
Publication docs are evidence added after the artifact freeze, not silently
represented as bytes in that snapshot.

## Actual build and narrow red-green correction

The first actual sdist failed the existing archive inspector: unanchored README
and LICENSE include patterns collected 910 npm dependency members, including two
forbidden nested `dist` README paths. Root independently reproduced this red.
Its exact artifacts, full member hashes and failure evidence remain read-only in
`.assets/releases/p6-a.xcJk5gqt`; the failed sdist is never qualified green.

The bounded round1 amendment changed only the seven existing
`[tool.hatch.build.targets.sdist].include` strings to root-anchored equivalents
and added one closest structural regression. Existing assertions, inclusion roots,
dependencies, lockfiles, runtime sources and the original Hatch hook were retained.
The new regression failed before the fix; the actual corrected archives supply
the behavioral green. No artificial `.git`, dependency removal, packaging bypass,
synthetic frontend or weakened inspector manufactured the result.

A fresh disposable copy and private build environment used Python 3.13.2,
build 1.3.0, Hatchling 1.27.0, Node 22.14.0 and npm 10.9.2. Acquisition was a
separate network-enabled phase: pinned build tools only in the fresh environment;
`npm ci --ignore-scripts` used the unchanged package lock, blank explicit user/global
configuration and a private cache. Existing environments/frontend trees and
HOME/XDG/CODEX_HOME were not changed. Exact resolved build-tool versions, manifest/
lock hashes, literal commands, environments and raw results are in the evidence.

Actual `npm run build` (TypeScript + Vite) and `python -m build --no-isolation
--wheel --sdist` ran under `(version 1) (allow default) (deny network*)`.
Both original Hatch hooks executed and reused the freshly produced real frontend
through their normal freshness path; no skip flag or hook edit was used. All 677
npm dependencies remained present throughout the build. Existing stale Browserslist
and large-chunk warnings were retained without dependency updates.

## Positive artifact and test evidence

| Exact retained archive under the bundle's `dist/` | Bytes | SHA-256 |
| --- | ---: | --- |
| `nanobot_ai-0.3.0-py3-none-any.whl` | 5464430 | `3c72e14e4358f33368904c9bad138c4c301889a70b7acaf3d12665df17fb26ff` |
| `nanobot_ai-0.3.0.tar.gz` | 4922981 | `7ce14291f9133a8a38320aae1238dc5c2233fd90121f417bdec5089f0c2213fe` |

The unchanged `inspect_archive()` reports zero missing/forbidden members for both
archives: wheel 1280 members, sdist 1281. Explicit dependency/cache/runtime/secret
member checks are also zero. Complete member hashes, wheel METADATA/entrypoints/
RECORD and RECORD verification are retained in hash-bound `artifact-inspection.json`.
All 587 actual frontend assets equal the real build in both archives; Python,
custom skills/scripts, templates, channels and guard bytes equal the frozen source.
The sdist retains the actual frontend assets, unchanged hook, project manifest,
README, license and third-party notices. Existing policy does not include the root
frontend build-source tree in the sdist; no new source-build policy is claimed.

Guard SHA-256 remains
`5a653ef73328b4840d31a7737fc3d0a79fbf631bf648fd3ebf9793cbee012d74`.
Service inspection is read-only and valid; no service/install/update script ran.

The corrected archive/build/deploy selection passed **32 tests**, with six old
state/CLI-state rehearsal cases explicitly deselected, under the frozen full
OS network/process-denial profile. The exact package-version node separately
passed under the frozen exact-interpreter local-process profile, still denying
all network. Root independently repeated 32+1. The initial root 31-pass/1-fail
result records subprocess denial for that node, not a product defect or waiver;
baseline 31+1 and corrected 32+1 are distinct records, never summed. Scoped Ruff,
in-memory compile and diff checks passed. Raw commands/profiles/results are embedded.

## Acceptance boundary and next gates

AC1–AC5 are accepted by the root's bounded PASS and fresh independent review PASS,
not worker self-acceptance. Exact review/root verdict texts and pre-final root
scope/publication raw records, hashes and private 0400 retained paths are embedded
under `final_acceptance_evidence` in the permanent JSON. Root will independently
recheck this final publication; that recheck has not yet occurred or been claimed.
Ignored `uv.lock` is separately archived
with explicit provenance only, not an accepted production lock or dependency closure.
Source imports/tests do not qualify an installed release.

Still unqualified: clean installed-wheel imports/CLI/core/channel extras, resolved
production wheelhouse/closure, strict typing, exact DOCX node, current copied-state/
config/service-start/rollback old-reader rehearsal, native Windows/Linux/target
Python and live platform inventory. Old state rehearsal is **UNQUALIFIED/NOT_RUN**.
Full P6 release verdict, runtime/D9 authorization and P7 operator/canary decisions
remain separate gates. All optional/background work stays disabled.
