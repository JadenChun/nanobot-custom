# P0 source coverage accounting

Date: 2026-09-08. **The 158-entry file accounting passed. The immutable isolated staging baseline recorded two failures; the 876-case overlay that closes them is P0-CANDIDATE-ACCEPTED.**

The [inventory](p0-compatibility-inventory.tsv) is generated from the exact rename-aware diff between common base `63d646f731a505731e97909d114ce257c6de36bc` and staging `4aa562d92828189b3fcefad079773efbab1d715a`. Upstream comparisons use `f47b8f23155ddec23cfb80ec001b544d495df89c`. Each diff entry has one unique `FILE-nnn` row. A row can cite multiple behavior findings; those are references, not duplicate file rows.

| Reconciliation | Expected | Observed |
| --- | ---: | ---: |
| Total entries | 158 | 158 |
| Added / modified / deleted | 84 / 72 / 2 | 84 / 72 / 2 |
| Renames | 0 | 0 |
| Test files / all other files | 57 / 101 | 57 / 101 |
| CORE / PROVIDER / EDGE / REPO non-test files | 37 / 17 / 41 / 6 | 37 / 17 / 41 / 6 |
| Unique file IDs / unique status-path pairs | 158 / 158 | 158 / 158 |
| Missing / duplicate / extraneous diff entries | 0 / 0 / 0 | 0 / 0 / 0 |
| Report finding anchors | 50 | 50 |
| `BASELINE-PASS` changed-test rows | 55 | 55 |
| `BASELINE-ORDER-DEPENDENT` changed-test rows | 1 | 1 |
| `BASELINE-FAIL` changed-test rows | 1 | 1 |
| `NOT-RUN` non-test rows | 101 | 101 |

Production/support mappings use 10 CORE, 14 PROVIDER, 22 EDGE and 4 REPO finding groups. EDGE-12–14 retain the channel audit's CHAN labels, EDGE-15–16 its CRON labels, EDGE-17–18 its CLI labels, and EDGE-19–22 its DEPLOY labels. Shared files cite additional findings.

All 57 changed test files ran from the pinned staging archive. Of their 800 test cases, 798 passed; the remaining two are recorded below. Four unmodified validation targets cited by the audit also ran and contributed 68 passes: `tests/agent/test_consolidate_offset.py`, `tests/agent/test_loop_save_turn.py`, `tests/agent/test_session_manager_history.py`, and `tests/providers/test_azure_openai_provider.py`. The combined 61-file set collected 868 cases and finished with 866 passes, 2 failures, no skips and no errors. [Baseline evidence](p0-baseline.md) records the clean environment, JUnit outputs and rerunner.

`tests/migration/test_fork_baseline.py` appears in the documentation's 62-path validation-spec extraction but is absent from pinned staging. It remains a future synthetic upstream-adapter validation specification, not an executed or silently-passed staging case. Its eventual assertions are affected-package acceptance gates once the migration contract is concrete.

Both deleted files, `nanobot/agent/subagent.py` and `nanobot/agent/tools/spawn.py`, remain explicit entries documenting the absence of background delegation. Added rows use `-` for the missing old path; deleted rows use `-` for the missing new path. A future rename must record both paths and update the structural validator rather than treating it as an unexplained deletion.

No historical diff entry is omitted. `.DS_Store` is explicitly nonruntime metadata, `.claude/settings.local.json` is developer-local configuration with no Nanobot runtime permission role, and the Azure provider delta is newline-only. These are accounted dispositions, not cleanup instructions. Old implementation plans remain historical evidence. New audit files and existing untracked `sessions/` are outside the pinned source diff; sessions were not inspected.

## Structural verification performed

A Python standard-library check read the TSV and report, then invoked only read-only `git diff --find-renames --name-status`. It asserted exact multiset equality with `(diff_status, effective_path)`, 158 unique IDs/paths, the explicit P0 status distribution below, and that every finding ID resolves to one of the report's 50 anchors. This check neither imports Nanobot nor exercises application behavior.

Reproduce the core exact-set and status check from the repository root:

```sh
python3 - <<'PY'
import collections, csv, subprocess
base = '63d646f731a505731e97909d114ce257c6de36bc'
staging = '4aa562d92828189b3fcefad079773efbab1d715a'
expected = [tuple(line.split('\t')) for line in subprocess.check_output(
    ['git', 'diff', '--find-renames', '--name-status', base, staging], text=True
).splitlines()]
with open('docs/migration/p0-compatibility-inventory.tsv') as handle:
    rows = list(csv.DictReader(handle, delimiter='\t'))
actual = [(r['diff_status'], r['new_path'] if r['new_path'] != '-' else r['old_path']) for r in rows]
assert len(rows) == len(set(actual)) == len({r['file_id'] for r in rows}) == 158
assert collections.Counter(expected) == collections.Counter(actual)
test_rows = [r for r in rows if r['new_path'].startswith('tests/')]
non_test_rows = [r for r in rows if r not in test_rows]
assert len(test_rows) == 57
assert collections.Counter(r['validation_status'] for r in test_rows) == {
    'BASELINE-PASS': 55,
    'BASELINE-ORDER-DEPENDENT': 1,
    'BASELINE-FAIL': 1,
}
assert all(r['validation_status'] == 'NOT-RUN' for r in non_test_rows)
print('158 entries reconciled; 55 changed-test files passed, 2 remain P0 gaps')
PY
```

The parent also performed a static AST shape check on pinned schema/runner source: custom `MaxTokensConfig` becomes upstream scalar `int`, custom context paths/repositories and provider key-pool/rate/timeout fields are absent, and `AgentRunner(provider)` becomes `AgentRunner()`. This confirms interface shapes, not successful runtime adaptation.

## Baseline-failure classification

| Inventory row | Result | Evidence and owner gate |
| --- | --- | --- |
| `FILE-132` `tests/context_repo/test_git_sync_safe_push.py` | `BASELINE-ORDER-DEPENDENT` | The mapped staging run failed `test_l_secret_error_messages_are_redacted` after earlier CLI tests permanently disabled the `nanobot` Loguru namespace. It passed alone and the entire file passed 10/10 in a fresh process. The candidate's CLI fixture closes this historical gap: the 65-test CLI-then-safe-push sequence and the 876-case candidate pass. The immutable historical row remains unchanged. |
| `FILE-137` `tests/deploy/test_deployment_bundle.py` | `BASELINE-FAIL` | Six staging assertions passed, but the launcher assertion failed because `deploy/scripts/start_local_marketing_agent.ps1` and `setup_local_crawl4ai_profile.ps1` are absent from the pinned staging tree. The accepted candidate supplies bounded scripts and passes the existing deployment assertions plus eight portable PowerShell cases. The immutable historical row remains unchanged. |

The initial no-network-at-all run had 16 failures: 14 environment artifacts from local-fixture denial, missing optional Pillow and an incomplete `PATH`, plus the same two logger-isolation and deployment-fixture baseline gaps recorded above. Its corrected loopback-only run is the result recorded above. The environment artifacts and two gaps remain in the immutable staging record; the accepted candidate closes the latter two at the P0 candidate boundary.

## Limits and remaining checkpoint

The source auditors examined every assigned non-test file at behavior-group level and mapped changed test intent. The inventory reconciles files plus identified symbols/contracts; **it is not an exhaustive symbol/branch coverage proof**. The large OpenAI-compatible provider's parser branches, installed OAuth-kit storage behavior, GUI settings mutation paths, full crash matrix and platform-specific native behavior remain named validation gaps. A static compatible-overlap classification does not establish semantic equivalence.

No live credentials, user configuration, sessions, providers, channels, crawler, VPN, MCP endpoint, deployment command, or external delivery was exercised. The temporary venv and its 150-package [exact version freeze](p0-baseline-requirements.txt) are test setup evidence only. Independent CORE/PROVIDER/EDGE source audits and parent checks informed the report. The future migration fixture is not a current blanket P0 blocker, but each affected replacement must qualify it when its adapter contract is accepted.

The two named baseline gaps are resolved at the accepted P0 candidate boundary. Each affected replacement still needs its own differential tests to pass. The integration owner may begin P1 under the migration plan, but source accounting and baseline passes are not ready-to-merge or release evidence.

## Candidate fix supplement

The immutable accounting above remains the staging record. The [candidate fix supplement](p0-fixes.md)
maps an overlay that resolves both defects to the existing P6 release/rehearsal findings and records
876 passing cases. Its [result](p0-candidate-results.tsv), durable [per-case results](p0-candidate-cases.tsv)
and [manifest](p0-candidate-manifest.json) are P0-CANDIDATE-ACCEPTED evidence; they do not change
the 158 rows or their historical validation statuses.
