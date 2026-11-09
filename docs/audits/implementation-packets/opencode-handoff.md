# OpenCode handoff for A3

Use DeepSeek V4.1 Flash for implementation. The local OpenCode configuration declares `commandcode/deepseek/deepseek-v4.1-flash`.
Check that the OpenCode session can select that exact ID.
Use a fresh session for each approved slice. Keep one implementation worktree and one write owner.
[Readiness review and slice order](opencode-readiness.md)

## Prepare the session

Run these checks in the shell that will launch OpenCode.
Confirm that `opencode models commandcode` lists the exact model ID above.
The audit shell did not have `opencode` or `uv` on its PATH.
Use a shell with the required tools before dispatch.

```sh
opencode --version
opencode models commandcode
uv --version
bun --version
npm --version
```

Use a clean implementation worktree at the approved staging revision.
Keep the audit packet directory readable to OpenCode. Supply the selected packet as context if the worker cannot read that path.
For a new worktree, use the commands below after verifying that the path and branch are unused.
If the fetched revision differs from the packet's pin, review the source delta before creating the worktree.

```sh
a3_repo=/home/jaden/Documents/Utilities/nanobot-custom
a3_checkout=/home/jaden/Documents/Utilities/nanobot-custom-worktrees/a3-implementation
a3_pin=29612c3df01dc3dce3501fe349bcf555cd6c58b6
git -C "$a3_repo" fetch origin &&
  test "$(git -C "$a3_repo" rev-parse origin/staging)" = "$a3_pin" &&
  git -C "$a3_repo" worktree add -b codex/a3-implementation "$a3_checkout" "$a3_pin" &&
  (cd "$a3_checkout" && opencode --model commandcode/deepseek/deepseek-v4.1-flash)
```

[OpenCode CLI reference](https://github.com/anomalyco/opencode/blob/dev/packages/web/src/content/docs/cli.mdx)

## Paste this prompt first

```text
Run A3-0 qualification for Nanobot in the current isolated implementation worktree.
This assignment ends with the baseline evidence. Keep product files and existing work unchanged.

Read AGENTS.md and this packet:
/home/jaden/Documents/Utilities/nanobot-custom-worktrees/staging-audit/docs/audits/implementation-packets/A3-0-qualification-baseline.md

Expected source: 29612c3df01dc3dce3501fe349bcf555cd6c58b6
Evidence root: /home/jaden/Documents/Utilities/nanobot-custom-worktrees/a3-evidence/A3-0

1. Check git rev-parse --show-toplevel, HEAD, origin/staging, and git status --short.
   Follow the packet's fetch and source checks. Report drift or a dirty checkout before running the baseline.
2. Verify the required Python, uv, Bun, and npm tools. Record actual versions.
   Use the isolated checkout's environment. Keep production config, credentials, and state outside the run.
3. Follow the packet in order. Set A3_BASELINE_PYTHON and A3_EVIDENCE_DIR.
   Make a fresh evidence directory for this run under the evidence root. Keep existing reports.
   Save the exact command, exit code, and output for each check.
4. Capture the full BasedPyright JSON, Python JUnit reports, dependency evidence, and client check results.
   Run independent checks even if one check reports baseline failures.
   Label unavailable tools or platforms, skipped checks, and CI evidence separately.
   Classify failures by actual signature. Leave product fixes to later slices.
5. Write A3-0-baseline.md in the run directory. Link the raw evidence.
   Check git status --short again. Record source, start and end status, and generated artifacts.
   Include command and dependency identities, result by check, and remaining evidence gaps.

Return READY_FOR_REVIEW with the report path and missing qualification evidence.
Return BLOCKED_ENVIRONMENT if a required tool prevents the run. Report the exact failure and any evidence captured.
Wait for coordinator acceptance before A3-1. The CI run alone does not qualify the local baseline.
```

## Paste this prompt for each approved coding slice

Fill every bracketed field before sending. Use the stage packet for the write boundary and the readiness review for the slice order.
Attach only the accepted contract, relevant evidence, and the selected packet section.

```text
Implement one approved A3 slice in Nanobot.

Slice: [stage, packet section, and exact criterion or subcase]
Implementation worktree: [absolute path]
Expected start: [HEAD and expected status]
Accepted baseline and dependency evidence: [absolute report paths]
Packet and accepted contract: [absolute paths or supplied text]
Allowed production files and symbols: [exact list]
Allowed test files and approved independent cases: [exact list or none]
Expected behavior: [observable result and required preserved behavior]
Required checks: [exact commands and expected result or accepted baseline failure]
Rollback boundary: [what to revert and any evidence or state to preserve]

1. Read AGENTS.md and the packet section. Verify the worktree and accepted inputs.
   If source, authority, protocol, or persisted data conflicts with the contract, report the conflict before editing.
2. Trace the normal entry point and its current owner. Make the smallest complete change inside the write set.
   Preserve existing callers, error behavior, and unrelated edits. Keep later slices out of the diff.
3. Verify the result at the public boundary. Run the named focused checks.
   If a necessary new or changed test case lacks independent authorship, return NEEDS_TEST_CASE.
   Give the coordinator the requirement and public interface needed for the requirements-only author.
4. For a strict-type repair, use frozen before and after full-source reports in the same dependency environment.
   Run scripts/check_type_regressions.py with the correct source roots. Require zero introduced signatures.
   Keep before and after source hashes and command evidence. Preserve the accepted baseline.
5. Review the diff and check the write set. Leave the changes in the worktree for review.

Return READY_FOR_REVIEW, NEEDS_CONTRACT, NEEDS_TEST_CASE, BLOCKED_ENVIRONMENT, or FAILED.
Report HEAD, changed paths, criterion results, exact commands and outcomes, evidence paths, and any remaining issue.
Stop at the end of this slice. The coordinator accepts it and sends the next assignment.
```

## First coding slice after A3-0

Use A3-1 §1a and `AC1a` to fill the template.
The normal entry points are SDK construction and runtime-control image reload.
Use the exact §1a write set. Keep the MCP media writer and network policy outside this slice.
Record the accepted A3-0 report path and actual implementation revision before sending.
[A3-1 packet](A3-1-authority-foundations.md)

Use these existing checks first. The root capture, artifact write, and reload checks provide separate evidence.

```sh
uv run --no-sync python -m pytest \
  tests/agent/test_configured_artifact_store.py::test_enabled_loops_capture_independent_config_data_roots \
  tests/tools/test_image_generation_tool.py::test_generate_image_tool_stores_artifact_and_source_images \
  tests/tools/test_migration_image_adapter.py::test_reload_without_previous_tool_keeps_sandbox_path_restriction
```

Verify the full `AC1a` result with a fixture provider.
The generated image and sidecar must remain in the explicit instance after a reload.
Use a deterministic boundary check if it covers the combined outcome.
If a necessary test case is still missing, supply a requirements-only case before dispatch.
