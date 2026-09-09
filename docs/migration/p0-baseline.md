# P0 source baseline

Audit date: 2026-09-08. The isolated staging baseline was executed. **Historical staging result:** the mapped suite has one pinned-source deployment-fixture failure and one order-dependent test-isolation failure. The 876-case overlay that closes them is P0-CANDIDATE-ACCEPTED. This evidence does not authorize an upstream merge, deployment, or data conversion.

The historical result below is immutable. A later candidate overlay resolves both defects with 876
passing cases and is accepted at the P0 candidate boundary; see the [candidate fix
supplement](p0-fixes.md).

| Reference | Pinned commit |
| --- | --- |
| Common base | `63d646f731a505731e97909d114ce257c6de36bc` |
| Staging fork | `4aa562d92828189b3fcefad079773efbab1d715a` |
| Official upstream target | `f47b8f23155ddec23cfb80ec001b544d495df89c` |

## Isolated source and environment

The test source was a `git archive` of staging at `/tmp/nanobot-p0-staging-4aa562d.InGOY2`, never the mutable worktree. The historical ordered 262-file SHA-256 manifest was `744da896a181a356bf5269d2eac15bbb4b66e3e1f96df0ff25f25ccbc222f3ca`. The runner now rechecks every tracked blob against `4aa562d92828189b3fcefad079773efbab1d715a` before environment reuse or tests, then scans every archive entry. The only accepted untracked roots are the reused `.venv`, ignored legacy `.venv313`, `evidence`, inert `.pytest_cache`, and these prior runtime-state directories: `runtime-home`, `runtime-home-deploy`, `runtime-home-redaction`, `runtime-home-safe-push-file`, and the corresponding `runtime-xdg-{cache,config,data}` directories with no suffix, `-deploy`, `-redaction`, or `-safe-push-file`. Every other file, directory or symlink is rejected; symlinks are accepted only inside the two virtual-environment roots. The verifier writes its explicit path/NUL/content-SHA manifest to each evidence directory. That explicit manifest is `8ca0b4d42fb58586179b77c012a643b1cb091753aaf81a7039d9888d3a075ba1` for the verified archive; it is a different serialization of the same 262 files. The archive has no `.git` worktree and therefore cannot contain the repository's untracked documents or `sessions/` data.

The runner was zsh on macOS 26.6.2 arm64 with CPython 3.11.15, Git 2.53.0, uv 0.10.9, pytest 9.1.1 and pytest-asyncio 1.4.0. A disposable venv installed staging with the declared `dev`, `api`, and `crawler-worker` extras. Material resolved versions were aiohttp 3.14.3, anthropic 0.125.0, crawl4ai 0.9.3, httpx 0.28.1, mcp 1.30.0, openai 3.8.0, Pillow 12.3.0, pydantic 2.13.5, pydantic-settings 2.15.0, python-telegram-bot 22.8, slack-sdk 3.44.1 and typer 0.27.2. Package installation could access the configured index; no test process could.

Each test subprocess received only a disposable `HOME` and XDG config/cache/data directories below the archive. `PYTHONNOUSERSITE=1` prevented user-site imports, no provider/channel credential variables were passed, and `Path.home()` was asserted to equal the disposable home before application imports. `sandbox-exec` denied all network access except loopback fixtures. A socket probe bound loopback successfully and raised `PermissionError` when connecting to `1.1.1.1:443`. The reviewed tests use mocks or local bare-Git/loopback fixtures for their network/process paths; no live provider, channel, crawler, VPN, MCP, credential, production config, session, or deployment service was invoked.

The executable [baseline runner](run-p0-baseline.sh) reproduces the archive, frozen dependency set, clean environment, network guard and JUnit run from the repository root. It creates a unique evidence directory before setup; saves source verification, setup output, phase status, the declared-freeze hash, actual interpreter/tool versions and installed-package freeze; and keeps pytest output separate. Reused `P0_ARCHIVE_DIR` values are accepted only after the same pinned-source verification and require the existing `.venv/bin/python`. Every run receives a new HOME and XDG config/cache/data tree below its evidence directory, an explicit venv-first system PATH, `PYTHONNOUSERSITE=1`, and the loopback-only sandbox. It takes `P0_PYTHON` and `P0_UV` only as executable-path overrides; test and probe subprocesses do not inherit caller configuration or credential variables. The exact [150-package freeze](p0-baseline-requirements.txt) has SHA-256 `78cfe2c0f7a2dbd56fdd66664a6441f9a15046ea1550dc46d2212b60b2a16549`. Use it as follows:

```sh
P0_PYTHON=/opt/homebrew/bin/python3.11 sh docs/migration/run-p0-baseline.sh

# Reuse an existing archive and venv for a critical subset without reinstalling.
printf '%s\n' \
  tests/agent/test_capability_policy.py \
  tests/test_openai_api.py > /tmp/nanobot-p0-critical-tests.txt
P0_ARCHIVE_DIR=/tmp/nanobot-p0-staging-4aa562d.InGOY2 \
P0_TEST_PATHS_FILE=/tmp/nanobot-p0-critical-tests.txt \
sh docs/migration/run-p0-baseline.sh
```

The runner prints a unique `evidence/run-<UTC timestamp>-random>/` directory with selected paths, source verification, setup output, phase/error status, environment probes, versions, package freeze, JUnit and pytest log even when pytest exits nonzero; subset runs cannot overwrite a prior result. A failed source check stops before setup; a failed setup keeps `setup.log` and stops before pytest. The actual archive retains `mapped-collect-clean.log`, `mapped-run-loopback.log`, `mapped-junit-loopback.xml`, `safe-push-file-junit.xml`, `redaction-alone-junit.xml`, and `deploy-alone-junit.xml` for the independent reviewer. The temporary path is disposable; the runner, declared freeze, source verification and saved [sanitized per-test result TSV](p0-mapped-results.tsv) are the durable evidence.

The freeze records exact resolved versions for this local execution, but it is not a hash-authenticated artifact lock: it does not attest wheels, indexes, platform builds or the reused virtual environment. The source verifier authenticates the pinned source only; reuse still trusts the local archive custodian and the contents of its explicitly allowed `.venv`/`.venv313` state. Reproduce only on a compatible macOS/Python/sandbox environment, inspect each run's saved package freeze and versions, and do not treat this P0 evidence as cross-platform qualification.

## Mapped execution result

The documentation specifications name 62 test paths. Sixty-one resolve in the pinned staging tree: all 57 changed test files plus four unmodified validation targets, `tests/agent/test_consolidate_offset.py`, `tests/agent/test_loop_save_turn.py`, `tests/agent/test_session_manager_history.py`, and `tests/providers/test_azure_openai_provider.py`. `tests/migration/test_fork_baseline.py` is a future synthetic migration specification: it is not in the staged tree and was not manufactured or marked passed by this baseline.

`pytest --collect-only` collected 868 tests from the 61 staged files with no collection errors. The corrected loopback-only run completed in 29.52 seconds with **866 passed, 2 failed, 0 skipped, and 0 errors**. The 57 changed test files account for 800 cases; the four unmodified validation targets account for 68 passing cases. [Sanitized per-test outcome and duration evidence](p0-mapped-results.tsv) has 868 data rows and SHA-256 `adc2b3e2473fc5085e9d0d293642a2b17f33002a12c33b050d606e490f52cffd`; the ephemeral JUnit preserves the original case XML for review.

As a separate reviewer check, the parent reran five representative mapped files from the archive's existing venv under the same disposable-home and loopback-only guard: `test_capability_policy.py`, `test_memory_budget_hardening.py`, `test_codex_auth.py`, `test_copilot_auth.py`, and `test_openai_api.py`. That subset passed 51/51 with 33 aiohttp warnings in 1.25 seconds. It is corroborating evidence, not a duplicate addition to the 868-case count or checkpoint acceptance.

| Result | Count | Interpretation |
| --- | ---: | --- |
| Passed | 866 | Current staging behavior under the isolated baseline only; this is not replacement equivalence. |
| Failed: order-dependent logger assertion | 1 | `tests/context_repo/test_git_sync_safe_push.py::test_l_secret_error_messages_are_redacted` had an empty capture after prior mapped CLI tests left `logger.disable("nanobot")` active. It passes alone and its full file passes 10/10 in fresh processes. The failure is a pinned test-isolation/order defect that must not be hidden by P1. |
| Failed: missing deployment fixtures | 1 | `tests/deploy/test_deployment_bundle.py::test_local_marketing_launcher_uses_dedicated_authenticated_profile` requires `deploy/scripts/start_local_marketing_agent.ps1` and `setup_local_crawl4ai_profile.ps1`; neither exists in `git ls-tree -r 4aa562d... -- deploy/scripts`. The isolated test still fails, so this is a pinned staging source/fixture gap. |
| Skipped / collection errors | 0 / 0 | The declared optional dependencies needed by these tests were installed. |

The initial all-network-denied run produced 852 passed and 16 failures. Fourteen were environment artifacts: eleven aiohttp API fixture failures and one loopback socket test from denying local TCP, one screenshot test without Pillow, and one shell-output test without the venv's literal `python` on PATH. The other two were the same logger-isolation and deployment-fixture baseline issues recorded above. The corrected loopback-only run above is authoritative for staging; its two historical failures are closed by the accepted candidate overlay.

The redaction test's fresh-process result preserves its value as a staging behavior check, but it does not erase its full-mapped-suite failure. The accepted candidate closes the redaction and deployment gaps without altering the historical results. The missing future `tests/migration/test_fork_baseline.py` and any fixtures remain assigned validation work for the affected upstream-adapter packages; no synthetic test was added because the available specification describes future migration equivalence rather than an additional deterministic staging-only contract.

## Checkpoint boundary

The exact 158-entry source accounting remains valid and the inventory now records the executed changed-test rows. The 101 non-test rows remain `NOT-RUN`; their mapped test evidence may establish staging behavior but does not prove a preserved/upstream replacement. All compatible-overlap and conflict dispositions remain unverified until their downstream differential and integration tests pass.

The P0 candidate boundary is accepted: the overlay resolves the deployment fixture failure and logger-state test-isolation defect, and the independent reviewer accepted the coverage and test evidence. P1 may begin code replacement under the migration plan. Future synthetic upstream-adapter fixtures remain affected-package acceptance gates; they do not become blanket P0 assertions until their staging-compatible contract is defined.

See [custom-change audit](custom-change-audit.md), [file inventory](p0-compatibility-inventory.tsv), [coverage accounting](p0-coverage.md), and [migration roadmap](../nanobot-upstream-migration-plan.md).
