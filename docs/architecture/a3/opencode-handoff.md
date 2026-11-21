# OpenCode handoff for the remaining A3 audit

## Continue from the reconciliation branch

Paste this prompt into OpenCode. The original 120-finding handoff is recorded
below for reference.

```text
Continue the last four Nanobot A3 findings in the existing local checkout at
/home/jaden/Documents/Utilities/nanobot-a3-reconciliation.
Read docs/architecture/a3/opencode-handoff.md. Follow "Continue from the
reconciliation branch" and the November 15 decisions linked there.
Use only DeepSeek V4.1 Flash, model commandcode/deepseek/deepseek-v4.1-flash,
for the coordinator and every subagent. Run independent work in parallel
with one writer per boundary. Start from the existing reconciliation branch.
Preserve accepted work. Work on Linux and use the Android emulator.
Integrate and verify locally. Report exact evidence and any remaining blockers.
```

### Starting point

- Use `codex/a3-audit-reconciliation` in the checkout above. The reviewed
  product base is `310d2056699196109f539e8bf98916a61f996b12`, tree
  `224a2d030caa8f875b6c6fa7adb0df117feee696`. It is 157 commits ahead of
  `origin/codex/staging-audit-implementation` at `722bd8ec`.
- Run `git status --short --branch` and `git rev-parse HEAD 'HEAD^{tree}'`.
  Review changes since the base and use the latest valid descendant. Preserve
  local work. Keep the local reconciliation history if you fetch remote refs.
- Read `AGENTS.md`, the [November 15 decisions](decisions.md#a3-release-decisions-for-the-four-remaining-findings),
  the [run summary](reconciliation-run-summary.md), and the newest relevant
  entries in the [implementation handoff](implementation-handoff.md). Read the
  [ledger](evidence.json) and the original reports for the four assigned IDs.
- Recompute status from the ledger. It has 132 accepted and four unverified
  original findings. All nine separate review findings are accepted.
- Reuse relevant receipts under
  `/home/jaden/Documents/Utilities/nanobot-custom-worktrees/a3-evidence/a3-reconciliation-2026-11-14/`.
  The Linux baseline at `3f763a41` passed 18,506 tests, skipped 22, and passed
  six subtests. Strict BasedPyright reported zero diagnostics across 546 files.
  For Android, start with `resume2/a37-admission-review/REVIEW.md` and
  `resume2/a39-r01f05-review/REVIEW.md`. The `30f8cd4c` launch repair and earlier
  installed-app navigation already have receipts. The proof-bound recovery and
  revocation checks still need matching installed-app evidence.

### Work packages

| Finding | Write owner and entry points | Focused checks |
| --- | --- | --- |
| H01-F5 and K01-F06 | One shell-policy worker. Own command risk and launch admission in `nanobot/agent/policy.py` and the existing shell and tmux launch boundary. Read the knowledge sync owner to confirm its scope. Start from the existing K01-F06 guidance repair at `f126a952`. | Check a normal Git read, configured knowledge sync, exact approval on risky effects with `autoPush` enabled, ordinary nested launch, and refusal when reduced protections lack required proof. Start with `tests/context_repo/test_git_sync_safe_push.py` and `tests/tools/test_migration_exec_policy.py`. Use disposable targets. |
| E02-F16 | One OAuth worker. Own provider credential storage and its gateway, settings, SDK, and CLI bindings. Reuse `OAuthCredentialCoordinator` and the resolved instance root. Check the existing xAI instance binding. | Check that login, refresh, status, logout, and cache state stay within the selected instance. Start with `tests/providers/test_oauth_credential_coordinator.py`, `tests/providers/test_oauth_sdk_contracts.py`, and `tests/webui/test_ws_http_oauth.py`. Use two disposable instance roots and synthetic credentials. |
| R01-F05 | One mobile worker. Own the Android custody and production mobile routes needed for device-key recovery, and the emulator for this pass. Reuse the existing owner revoke operation. Reserve any shared gateway file with the coordinator before editing it. | Start with `tests/remote/test_device_proof.py`, `test_device_recovery.py`, `test_edge_pair_proof.py`, and `test_pairing_persistence.py` in the same directory. Run the signed app through Serve TLS for the acceptance criteria in the decision. Record the production routing and signed artifact identity. |

Start the three packages together if the runtime has capacity. Give each writer
a separate branch and worktree at the selected base. Assign distinct ports,
writable runtime roots, and test directories. Use a pinned shared dependency
environment read-only. For each worker, record the allowed paths before edits.
Serialize any shared gateway, registry, or protocol change under one owner.
The coordinator owns integration, the ledger, and final artifact receipts.

Select `commandcode/deepseek/deepseek-v4.1-flash` explicitly for the coordinator,
workers, and any requirements-only test author. Verify the effective model in
each session. If a subagent route cannot guarantee that model, use a fresh
OpenCode session with `run --model` and its assigned working directory. Report
a model blocker for any package that cannot use the selected model. Continue
the other packages when possible. Workers return to the coordinator.

Use existing focused evidence first. When a new case is necessary, follow the
requirements-only session contract in `AGENTS.md`. For each Python repair,
retain frozen before and after full-source BasedPyright reports with the same
dependencies. Run the comparator and verify zero new signatures across all
files. Run the affected checks again where integration changes their inputs.
For P6 or release closure, run the required full checks and tie the receipts
to the integrated candidate. Preserve the release hold until its gates pass.

Each worker returns the finding ID, base and commit, exact changed paths,
commands and exits, receipt path, and remaining gaps. The coordinator checks
the integrated diff and exact evidence before updating the ledger. Keep the
commits local. Use disposable state and test credentials. Leave push, merge,
and deployment to the project owner.

## Historical prompt from the 120-finding checkpoint

```text
Continue the remaining A3 implementation audit for JadenChun/nanobot-custom.
Run git fetch --prune origin from the repository root.
Run the next command to read the handoff.
git show origin/codex/staging-audit-implementation:docs/architecture/a3/opencode-handoff.md
Follow the current product contract and use the evidence ledger as the queue.
The snapshot has 136 original findings. Of these, 16 are accepted and 120 are
unverified. Check the current ledger before you select a finding.

Use a separate worktree based on origin/codex/staging-audit-implementation.
Work on Linux. Use an Android emulator for any Android verification.
Refresh the inventory. Start with K01-F12 in A3-1 if it is still unverified.
Review current code and existing evidence before you make changes.
Fix demonstrated defects in small batches. Run the checks for each batch.
Record the exact evidence, unresolved items, and the next batch.
Continue through the queue in dependency order. Leave merge and deployment
for the project owner. Report blockers with the condition needed to resume.
```

## Source and scope

- Repository: `JadenChun/nanobot-custom`.
- Audit branch: `codex/staging-audit-implementation`. Review is in
  [PR #27](https://github.com/JadenChun/nanobot-custom/pull/27).
- Audit evidence snapshot: `c0dfcc79caaffb4b5effd0d3422cee27fa83ed84`.
  This is a reference commit. Start from the current remote audit branch.
- Staging at the snapshot: `f1a6bb85588ba044fc2ca50e3e9f41eb2f13b6f9`.
  Verify that current staging is an ancestor. Report any new staging changes
  that affect your batch.
- For this pass, the user selected Linux and the Android emulator.
  Historical macOS and Windows results remain useful as records.
- The first release uses the current wire and state contract. It does not
  require historical-release migration or a historical two-APK upgrade.
  Preserve current data, revocation, and interrupted-effect evidence.

From the repository root, use a fresh branch and worktree.
Choose another unused path if this worktree already exists.

```bash
git fetch --prune origin &&
git merge-base --is-ancestor origin/staging origin/codex/staging-audit-implementation &&
git worktree add -b codex/a3-audit-reconciliation ../nanobot-a3-reconciliation \
  origin/codex/staging-audit-implementation &&
cd ../nanobot-a3-reconciliation &&
git status --short --branch &&
git rev-parse HEAD 'HEAD^{tree}'
```

## Read in this order

1. [AGENTS.md](../../../AGENTS.md) and [.agent/design.md](../../../.agent/design.md)
   for the repository rules.
2. [Current handoff](implementation-handoff.md) and [review](review.md) for
   the latest evidence. Historical checkpoints follow the current notes.
3. [Product contract](v1-product-contract.md) for the accepted behavior.
   The original chat owns subagent instructions, approvals, and Stop.
   Inspect subagent activity through its read-only view.
4. [Implementation plan](implementation-plan.md), the relevant
   [stage packet](../../audits/implementation-packets/README.md), and the
   finding's original report. The ledger's `source_reports` maps ID prefixes
   to their reports. Keep those reports and the original IDs intact.
5. [Evidence ledger](evidence.json) for the live queue. Use `findings` for the
   136 original entries. Use `review_findings` for additional AR and RR findings.

## Evidence available at the snapshot

| Check | Result |
| --- | --- |
| Linux at `484a1476` | 18,297 tests passed. The run skipped 22 tests and passed six subtests. Strict BasedPyright reported zero diagnostics across 546 files. |
| Shared clients | All 1,550 WebUI tests and seven installed browser scenarios passed. The TUI suite passed at its earlier qualified checkpoint. |
| Subagent UI at `7d96468c` | All 1,550 WebUI tests, lint, and eight installed browser scenarios passed. An Android emulator debug build passed read-only detail and both Back paths over loopback TLS. |
| Hosted at `c0dfcc79` | Linux Python 3.11 and 3.14, WebUI, TUI, and Docker passed. The broad Windows Python job failed. [Hosted run](https://github.com/JadenChun/nanobot-custom/actions/runs/43504885741) |
| Android | Emulator tests covered exact cancellation, restart, lost initial and renewal responses, and manual-code resumption. The bridge delivered 36 MiB followed by 11 bytes on the same socket. The race and frame fixtures used loopback TLS. The signed APK separately installed and launched. |
| Release limits | The signed APK delivery race over Tailscale remains unverified. Formal A3 and P6 release acceptance remain open. |

On this machine, read the receipts in
`/home/jaden/Documents/Utilities/nanobot-custom-worktrees/a3-evidence/linux-android-2026-11-13/`
when they apply. Use the handoff and linked CI if that directory is unavailable.
Re-run a check when its relevant source, dependency, or artifact changed.

## Work queue

The counts below describe the snapshot. Read current IDs from the ledger.
A3-0 establishes the baseline and has no original findings of its own.

| Stage | Unverified | Next work |
| --- | ---: | --- |
| A3-1 | 10 | Review authority foundations. Start with archive provenance under K01-F12. |
| A3-2 | 12 | Review process ownership, timeouts, and cleanup. |
| A3-3 | 1 | Verify durable session identity and its continuation. |
| A3-4 | 14 | Verify client admission, approvals, and Stop. |
| A3-5 | 3 | Verify delegated execution and parent-mediated control. |
| A3-6 | 29 | Verify capability readiness, request ownership, and replacement. |
| A3-7 | 7 | Verify the remaining mobile boundaries on Linux and Android. |
| A3-8 | 24 | Verify recovery, retained evidence, and data integrity. |
| A3-9 | 12 | Verify the first-release artifacts and rollback. |
| A3-10 | 8 | Review compatibility retirement after release qualification. |
| **Total** | **120** | |

```bash
python - <<'PY'
from collections import defaultdict
import json
from pathlib import Path

ledger = json.loads(Path("docs/architecture/a3/evidence.json").read_text())
remaining = defaultdict(list)
for row in ledger["findings"]:
    if row["acceptance"] != "ACCEPTED":
        remaining[row["primary_acceptance_stage"]].append(
            f'{row["id"]} ({row["package"]})'
        )
for stage in ledger["stages"]:
    rows = remaining[stage["id"]]
    if rows:
        print(f'{stage["id"]} {len(rows)}: {", ".join(rows)}')
print(f"Remaining: {sum(map(len, remaining.values()))}")
PY
```

For the first batch, read the [A3-1 archive criterion](../../audits/implementation-packets/A3-1-authority-foundations.md#2-archive-provenance--r05)
and [K01-F12](../../audits/module-review-2026-11-06/reports/K01.md).
Check that retained archive text enters the next model request once as historical data.
Check the current compaction and saved-history entry points. Use synthetic text.
Record the disposition before you move to another owner.

## How to complete each batch

1. Read the packet's acceptance criteria and the original findings. Select a
   bounded set that shares one owner. Follow stage dependencies.
2. Trace the real entry point and supported configuration. Check existing
   tests and prior evidence. Record whether the current behavior satisfies
   each criterion, needs repair, or cannot be verified in this environment.
3. For a demonstrated defect, make the smallest correction. Use existing
   focused tests or an isolated fixture with ordinary synthetic data.
   Use source inspection and defensive checks for security findings.
   Follow the repository's requirements-only procedure for new behavior tests.
4. For Python source repairs, apply the
   [P6 validation rules](../../migration/p6-error-attribution.md). Freeze the
   before and after checker reports under the same dependency inventory.
   Use `scripts/check_type_regressions.py` to check every file for new diagnostics.
   Run the affected tests. Run full strict and the required full local suite
   before you claim P6 completion.
5. Record finding IDs, exact source and environment, entry point, observed
   result, test commands and exits, artifact paths, and limits. Add evidence
   against each finding. Mark it accepted only when a named reviewer accepts
   matching evidence for the whole criterion. Preserve the historical entries.
6. Leave one owner for each overlapping code path. Use small, reviewable
   commits on the reconciliation branch. Use disposable data and local
   fixtures. Leave the live gateway and user data outside the test scope.

Return each batch with the commit, finding IDs and dispositions, checks, any
remaining issue, and the next batch. Keep blocked items in the queue and record
what evidence or access would unblock them.
