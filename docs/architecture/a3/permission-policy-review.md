# Permission policy and macOS shell review

2026-11-19 · `codex/staging-audit-implementation`, based on
`7b85a818be921e85d3be8905182efe8bb3ade296`.
This is a follow-up to the [accepted Sandbox direction](decisions.md#sandbox-experience)
and [A3-2 process ownership](../../audits/implementation-packets/A3-2-runtime-process-ownership.md).
The edit and approval corrections are verified in the working tree.
Native macOS Sandbox commands remain gated. This does not amend the frozen
plan or claim release acceptance.

## Working contract

A regular chat in **Sandbox** should be able to read, edit, clean up, and run
normal builds and tests in its authorized workspace without asking for each
routine action. **Working in** selects the project; it doesn't grant access.
The existing **Read and write** and **Once** grants authorize their own scopes.
A narrower read-only grant and the private/managed-repository restrictions still
apply. An approval for a risky action does not widen the folder grant or turn
missing confinement into Full access.

| Decision | Owner and observable result |
| --- | --- |
| Can this target be accessed? | The existing workspace/resource policy resolves the effective target and live grant. A missing folder capability uses the existing folder-access flow, not a risky-action prompt. |
| Is this a valid, admissible operation? | The concrete tool owns validation, hard refusals, and backend availability. An invalid, forbidden, or unsupported operation explains what stopped it without creating an approval that cannot work. |
| Does this permitted operation need consent? | The risky-action policy requests one decision for the specific effect. The current exact-effect grant stays single-use and is checked at dispatch. Preserve the high-impact gates, including destructive Git actions and remote effects. |
| Can this command run and be stopped safely? | The executor applies actual OS confinement and owns the launched command tree through settlement. Status shown to the agent and client must reflect what that executor can enforce. |

## Findings and correction boundaries

### 1. Structured edits: assess the change the editor will make

**Confirmed local defect, with a preparation mismatch.** The
[risk rule](../../../nanobot/agent/policy.py) treated replacing one matched
line as replacing all of the matched block. In a writable 100-line fixture
this produced a risky-action prompt and left the file untouched. Sending the
same one-line change through `write_file` succeeded. Read-only and out-of-grant
targets returned a separate folder-access result.

The approval preview uses `find_edit_match`. The
[executor](../../../nanobot/agent/tools/filesystem.py) used a different
matcher, supports `occurrence` and `line_hint`, and adjusts the selected text
before writing. A targeted repeat and an ambiguous repeat both asked for
approval in the probe. The matchers also disagreed on a formatting-normalized
input. Changing only the percentage denominator would leave that mismatch.

**Correction:** use the editor's actual candidate replacement for risk
classification, with matching, selection, and formatting owned by the editor
instead of reimplemented in policy. Judge impact against the resolved file,
using the same standard as an overwrite for material changes; retain the bulk
replacement and high-impact guards. Let the editor report missing or ambiguous
matches as validation feedback; a specified occurrence is not a bulk edit.
At mutation, ensure the applied replacement is still the evaluated one, or
reassess/refuse if the target or relevant file state changed. Keep the existing
path resolver, mutation lock, dispatch rechecks, and effect recording.

### 2. Shell refusals and deletion: only ask when consent can help

**Confirmed ordering defect.** The batch policy asks for approval for
`rm -f scratch.txt`, but [the shell executor](../../../nanobot/agent/tools/shell.py)
has an unconditional deny for that form. The typed approval/resume probe then
hit the deny and started no command. A simple single-file deletion did not hit
the approval gate, but managed macOS execution could not start it. These are
different outcomes.

**Correction:** use the executor's existing deterministic refusal decision
before offering effect approval, using the same normalized call (including
supported aliases) that would dispatch. Preflight must not launch the action,
stage files, or consume a grant. Keep the live guard at dispatch. Don't copy
the deny regexes into another layer or use the shell's working directory as
proof of the affected paths. Keep hard denials, including the existing
recursive-removal denials, distinct from allowed operations that need consent.

Prove ordinary, explicit single-file cleanup through a qualified Sandbox
executor. Do not add a second general delete path to compensate for the missing
shell backend. If file-only cleanup is needed independently, define one
structured operation through the existing filesystem owner: one explicit
regular-file target, current write grant and target recheck, no directory,
glob, symlink, or protected path. Record a deletion as absence, not a
zero-byte write; establish its cancellation and recovery semantics before
advertising it.

### 3. macOS: qualify filesystem enforcement and process ownership separately

**Confirmed platform capability gap.** A harmless ordinary command succeeded
using the interpreter configured for the installed Kohaku. The same runtime
refused the managed Sandbox command before launch: it only has managed
Bubblewrap. Full/Once command paths that require owned native supervision
also lack that capability on macOS; ordinary POSIX execution isn't that proof.
Making a policy less strict cannot supply either missing capability.

[DSH's macOS runner](https://github.com/deepseek-ai/deepseek-harness/blob/5badb15009ae1756c3afe0ae0cef1faafc290ccc/packages/sandbox/sandbox-local/README.md#how-runners-are-selected)
is a useful filesystem reference. Its
[Bash confinement contract](https://github.com/deepseek-ai/deepseek-harness/blob/5badb15009ae1756c3afe0ae0cef1faafc290ccc/packages/shell/bash-sandbox/README.md#known-limitations-and-deferred-work)
does not claim network or uniform process-visibility guarantees. On this Mac a
disposable native Seatbelt probe allowed a workspace write, blocked an adjacent
write and a write under a nested read-only root, and denied a read of a
synthetic private file. This proves that filesystem subset only. The host
manual also marks `sandbox-exec` deprecated.

**Correction direction:** prefer a native macOS executor that passes the
existing A3-2 contract, not just a wrapper that succeeds with a simple command.
Keep the existing Linux backend. Have the Mac adapter use the same live grant
inputs and the existing executor/admission ownership boundary; derive command
readiness for the runtime and UI from the same verified backend capability.
No new permission mode, legacy migration, or automatic switch to Full access.

**Gate before enabling:** qualify both the scope of filesystem effects (nested
read-only grants, private/runtime exclusions, and narrowly scoped build
scratch/cache) and ownership of descendants through Stop, timeout, scope
revocation, and gateway recovery. A root exit or a process-group signal alone
is not proof that all descendants stopped. Check launch/permission races and
keep unsettled work owned and visibly uncertain. When a grant narrows, define
when existing processes using the old scope are settled; fencing new commands
or stdin alone is not evidence of that. Qualify native Sandbox first; don't
advertise Full or Once command support without evidence for those modes.
Seatbelt is a candidate for confinement, not a decision to rely on a deprecated
binary as the complete production solution. If a supported native path cannot
prove the contract, keep the capability unavailable and review an explicitly
isolated executor separately. A Linux executor does not qualify local Mac builds.

## Implementation and acceptance order

1. **Edit impact:** prove a one-line edit and a selected repeat apply in the
   writable project without approval; an ambiguous edit mutates nothing;
   substantial/bulk replacements still wait; read-only, out-of-scope, and
   changed targets stay denied. Exercise the same preparation with a
   formatting-normalized match.
2. **Shell decision:** prove a hard-denied or unsupported command never creates
   pending approval or spawns. Keep one typed, exact decision for an admissible
   risky command; denial or a stale grant starts nothing.
3. **macOS executor:** prove an installed local chat can run a small build/test
   and clean up a fixture in its workspace, refuse disallowed filesystem
   effects, and settle the entire owned process tree under cancellation and
   revocation. Missing or failed backend must refuse before launch.

Check each batch against its owning boundary, with before/after type evidence
per [the migration validation rules](../../migration/p6-error-attribution.md)
when product code changes. Keep existing Linux/Android evidence in its
original scope. Only rerun affected Linux checks for shared behavior and use
one Android remote smoke if the client contract changes.

## Evidence at the reviewed revision

- Darwin arm64, macOS 27.0.1 (26A434). Probes used disposable folders; no live
  permission settings, processes, or app installation were changed.
- Focused pytest: **104 passed, 15 skipped**. The skips require Linux Bubblewrap
  and/or its native supervisor. Covered the parser seam, effective write
  target, target retargeting, exact approvals, shell denials, managed exec,
  and workspace focus.
- Local receipts, source hashes, and runnable scratch probes:
  `/Users/jaden/.codex/qa/kohaku-permission-review-20261119-WGdy5h/`.
  `pytest.log` retains the skip reasons; `manifest.txt` identifies the source
  and environment. The probes establish current behavior and Seatbelt
  feasibility, not a passing correction or a qualified macOS backend.

## Implementation and verification checkpoint

- **Edits:** The editor now prepares one candidate replacement using its real
  match selection and formatting rules. The policy scores the actual change
  against the whole file instead of the size of the matched snippet.
  Ambiguous or invalid edits return tool feedback without mutating the file.
  Bulk and material changes still require exact approval. Dispatch binds the
  file target and before/after byte digests; the editor rechecks the grant,
  target, and candidate after acquiring the mutation lock. The old
  policy-only matcher has been removed.
- **Commands:** The policy uses the captured tool registrations and normal
  argument preparation for the built-in executor. A missing/withheld tool,
  a deterministic hard refusal, or an unavailable managed backend does not
  create an approval request. The same executor preflight runs before an
  exact approval is consumed. The working directory is part of the effect ID
  and is checked again by the executor as it prepares the live launch.
  Input staging, live grants, confinement and process admission still belong
  to the executor; approval never grants those capabilities. A failure after
  dispatch remains a spent attempt rather than permission to retry.
- **Compatibility:** No permission mode, grant schema, client protocol,
  or process backend changed. A pending approval that no longer matches the
  prepared effect requires fresh consent; it is not migrated or widened.

| Check | Result and scope |
| --- | --- |
| Red regressions | The original edit/approval probes reproduced nine failures. Two additional cases reproduced a prompt for a missing registration and a launch reaching the spawn boundary with a different working directory after approval. All pass with the correction. |
| macOS affected suites | **560 passed, 23 platform-dependent skips**, plus **195 passed** in filesystem, child/fork, registry and durable effect tests. |
| Linux focused replay | **38 passed, 9 Bubblewrap-dependent skips**. The disposable container ran the candidate from both the test loader and the supervisor's isolated interpreter. Native supervisor and deadline cases ran after provisioning the container's own machine identity. Its default namespace policy did not allow live managed Bubblewrap, so this is not a new Bubblewrap qualification. |
| Strict typing and lint | **0 → 0 diagnostics across 549 files**, frozen dependency inventory matched, global comparator passed. Ruff and diff whitespace checks passed. |

Validation scripts, source hashes, and logs are in
`/Users/jaden/.codex/qa/kohaku-permission-correction-20261119-aSzNOq/`.
This is affected-suite evidence, not a full-suite, installed Kohaku, or
Android rerun. No client code changed.

## Native macOS qualification — still gated

The filesystem fixture was repeated successfully on this Mac: Seatbelt
allowed a workspace write and denied the adjacent write, nested read-only
write, and synthetic private-file read. A separate lifecycle probe used the
existing POSIX owner with a disposable root and a short-lived child:

| After root exit | Existing POSIX result | Observation after cleanup |
| --- | --- | --- |
| Child in the root process group | Not quiescent | The group cleanup stopped the child. |
| Child in a separate session | Reported quiescent | The child was still alive. The probe closed its control pipe and verified that it exited. |

This rules out **Seatbelt plus the current POSIX process-group owner** as a
managed backend. The host's `launchd.plist(5)` documentation only promises
same-group cleanup, and `sandbox-exec(1)` is deprecated. A successful wrapper
or migration to [Apple App Sandbox](https://developer.apple.com/documentation/security/app-sandbox)
alone does not prove whole-tree settlement.

The next work is to qualify a supported native execution and ownership
mechanism against A3-2: confinement for builds/tests and explicit single-file
cleanup, plus whole-tree settlement on root exit, Stop, timeout, permission
retirement, and gateway recovery. Only expose readiness to the runtime and UI
after that proof. Keep the current refusal in the meantime; don't fall back
to native unsandboxed execution or advertise Full/Once as a workaround.
