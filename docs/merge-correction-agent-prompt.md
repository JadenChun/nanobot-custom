# Coding-agent prompt: close merge blockers

Continue the existing implementation; do not restart the original feature plan. Implement the
bounded corrections in [merge-correction-tasks.md](merge-correction-tasks.md), MC01 through MC08,
step by step. This authorizes correction work and focused local commits, not deployment or merge.

## Starting point

- Feature checkout: `/home/jaden/Documents/Utilities/nanobot-custom-worktrees/mobile-remote-context-lifecycle`
- Feature branch: `feat/mobile-remote-context-lifecycle`.
- Reviewed code baseline: `1689a4a06020c778a6e969b12bb19b27a1ac90f1`; later task-preparation commits
  may contain documentation only. Inspect current status/history before touching source.
- Last inspected staging: `aacca07cfadaf51fa1d83913f93ba2d797c0aa58`; fetch again for integration.
- Read the correction tasks, [mobile plan](mobile-remote-request-plan.md),
  [context plan](session-recovery-architecture.md) and [progress log](implementation-progress.md).
  Follow `/home/jaden/.agents/AGENTS.md`, repository `AGENTS.md`, `.agent` constraints and the
  applicable `orchestrate-development` skill. Keep the original handoff's safeguards.

The feature worktree was reported as the installed staging gateway's source. Do not edit live
runtime source in place, restart that service or alter its config/data. Prepare a separate repair
worktree/branch descending from the current feature handoff, for example
`fix/mobile-remote-merge-blockers` in a sibling worktree. Never reset/reuse an existing dirty
branch or copy unrelated edits from the original `nanobot-custom` checkout. Keep original source
and live sessions intact while validating the isolated repair candidate.

## Work loop

1. Confirm source ownership/imports, existing environment and evidence. Reuse a valid prepared
   environment where safe; do not blindly repeat the original new-worktree setup. An isolated
   repair environment must import that repair source, not another editable checkout. Freeze
   comparable source/dependency/checker baselines before Python changes, per the P6 repair rules.
2. Append MC01–MC08 status to the progress log. For each task extract requirements and the exact
   public inputs/outputs/error semantics. Check existing coverage first. Where needed, use a
   fresh requirements-only author with no history, repository/path/access, implementation plan,
   review/code, existing tests, diffs or logs. Do not give the author either handoff document.
   Preserve its artifact/session ID and authored outcomes during mechanical integration.
3. Implement MC01, then MC02, MC03, MC04 and MC05; these share authority/admission seams and
   need sequential ownership. Implement MC06 before MC07. Honor the task write boundaries;
   workers are not alone and must preserve others' changes. No automatic extra planner or broad
   review is required per task. Use the skill's available roles and isolation rules if delegating.
4. Validate real user journeys through authenticated ingress and execution. In particular:
   temporary Sandbox must stay Sandbox; new Sandbox work during Full work must not steer that
   Full turn; confirmed owner phones must submit actual Full messages; independently confirmed
   peers must work; revocation must retire owned native handles; owner control must work without
   the gateway; in-session restart/expiry recovery must settle without a cold restart or replay.
5. Inspect actual diffs, run the smallest affected checks plus mandatory type/lint gates, update
   progress and make a focused local commit. Continue autonomously to the next task. Reuse green
   evidence while inputs are unchanged; do not rerun the 17,556-test suite after every patch.
   Do not change expected outcomes to match current code or add source-layout/private-call tests.
6. Finish MC08 against freshly fetched staging, preserving all existing staging features. Run
   remaining required integrated checks once at the final candidate, then obtain a bounded review
   of the original findings and changed delta. Report the remaining M38/physical/signed-candidate
   qualification honestly; unavailable external resources do not stop independent code work.

## Preserve the accepted architecture

Use existing source/checkpoint, admission, permission, replay, native retirement and mobile
recovery owners. No parallel permission database, busy-inbox activation, new routing framework,
unrelated provider refactor, dependencies, `ruff format` or general test infrastructure.
Do not remove request binding to fix retirement, remove localhost guards globally to admit
phones, trust client owner flags, promote Sandbox from preferences, or auto-replay uncertain work.
Keep temporary disposal, current source/device checks and separate Stop/revoke semantics.

If an agreed outcome conflicts with repository reality, document the concrete discrepancy and
missing public contract. Resolve routine choices without new approval questions; do not silently
drop requirements or expand scope. Preserve user-declined physical-device/display operations.

Do not push, open/merge a PR, install over the daily phone app, change Tailscale/credentials,
mutate live services or switch shared model workloads. Prepare artifacts and exact rollback
inputs for any later concrete installation request. Retain applicable existing approval gates.

Finish with task-by-task PASS/remaining status, corrected/staging commit IDs, exact files and
checks/exits, independent case provenance, candidate artifact hashes, merge compatibility and
remaining qualification. Leave a clean local repair branch ready for the next staging review;
do not call the work complete while a required merge correction remains unsatisfied.
