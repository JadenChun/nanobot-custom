# A3-0 implementation packet — Current-origin and qualification baseline

**Status.** The source check is complete. The local qualification report is pending.
This packet defines the evidence that A3-1 needs before product changes start.

## Source and prior evidence

| Item | Observed result |
| --- | --- |
| Remote `staging` | `29612c3df01dc3dce3501fe349bcf555cd6c58b6`, checked on 2026-11-09 |
| Frozen audit source | `9562e57f8333d7f14b6aaeae837660723a9b2650` |
| Delta | The bundled skill-path fix changes `nanobot/agent/skills.py` and `tests/agent/test_skills_loader.py`. K01-STG-F01 records the remaining provenance issue. |
| Local source check | The `staging` worktree was clean and matched the remote revision. |

[K01 staging addendum](../module-review-2026-11-06/reports/K01.md) · [A3 planning chat](https://chatgpt.com/c/6aef25bd-7db8-83ec-bc98-f7fbf21da9d0)

GitHub Actions recorded these results at the exact staging revision.
Use them as remote evidence and compare them with the local run.

| Check | CI result |
| --- | --- |
| Linux Python 3.11 parallel compatibility tests | Failed. 31 failed, 17,310 passed, and 14 skipped. The serial CLI step was skipped. |
| Linux Python 3.14 Ruff and strict type check | Ruff passed. BasedPyright reported 39 errors. The test and coverage steps were skipped. |
| Windows Python 3.14 | Compatibility job cancelled. Dedicated Windows process tests passed. |
| Docker | Passed. |
| WebUI and TUI | Skipped. |

[Test Suite run](https://github.com/JadenChun/nanobot-custom/actions/runs/42235518494)

## 1. Freeze the worktree and environment

1. Fetch `origin` when baseline work starts. Use a clean worktree at the exact `origin/staging` revision.
2. Record the source SHA, clean status, and diff from the frozen audit source.
3. If staging has advanced, map the new changes to the affected audit owners. Refresh any dependent packet whose assumptions changed.
4. Select the Python interpreter before installing dependencies. Record its version and the host OS.
5. Keep test state and generated output in the baseline workspace. Use an evidence directory outside the source checkout.

Capture the exact commands, exit codes, and log paths for each check.
Retain the resolved Python lockfile, its hash, and the installed package versions.
Channel dependencies also come from separate manifests, so the lockfile alone is insufficient.
Record Bun versions, frontend lockfile hashes, and applicable native toolchain versions.
Record whether the WebUI bundle exists when Python tests run. Match that state when comparing later results.

## 2. Capture the Python baseline

Follow the dependency setup in `.github/workflows/ci.yml`.
Use Python 3.14 for the CI-equivalent strict profile.
If the implementation environment differs, record the difference and preserve the same profile for paired checks.
Set `A3_BASELINE_PYTHON` to the selected interpreter path.
Set `A3_EVIDENCE_DIR` to the existing evidence directory before running these commands.

```sh
uv sync --python "$A3_BASELINE_PYTHON" --all-extras --dev
uv run --no-sync python --version
uv run --no-sync python -m scripts.install_channel_dependencies --all-channels
uv pip check
uv pip list --format json > "$A3_EVIDENCE_DIR/python-packages.json"
uv run --no-sync ruff check nanobot tests conftest.py
uv run --no-sync basedpyright --outputjson > "$A3_EVIDENCE_DIR/basedpyright.json"
uv run --no-sync python -m pytest -n auto --dist loadfile \
  --ignore=tests/cli/test_commands.py --durations=25 --durations-min=1.0 \
  --junitxml="$A3_EVIDENCE_DIR/python-parallel.xml"
uv run --no-sync python -m pytest tests/cli/test_commands.py \
  --durations=25 --durations-min=1.0 --junitxml="$A3_EVIDENCE_DIR/python-cli.xml"
```

Run each check independently after dependency setup succeeds.
Preserve the test IDs and failure messages, including serial CLI failures.
Use the workflow's split process-test command and ignore arguments when the host is Windows.
Keep the full checker JSON and source hashes for the later signature comparison.
[Strict-type evidence rules](../../migration/p6-error-attribution.md)

## 3. Capture client and deployment evidence

Run the WebUI baseline in `webui/`.
Use the recorded Bun version when comparing later results.

```sh
npm ci --ignore-scripts --dry-run
bun install --frozen-lockfile
bun run lint
bun run test:coverage
bun run build
```

Run the TUI baseline in `tui/`.
Run the host terminal smoke test that applies to this platform.

```sh
bun install --frozen-lockfile
bun run check
bun run test
# Run one terminal smoke test for the host:
python3 scripts/pty_smoke.py
# Windows uses scripts/conpty_smoke.py with the workflow's pywinpty setup.
bun run build
```

Run the current mobile inventory check with its self-test.
Run the C ABI smoke and emulator checks if the isolated host has the required toolchain.
Label the C ABI result as host evidence and the emulator result by the composition it tests.
Record unavailable checks and any historical physical Android evidence separately.
Fresh physical-device qualification belongs to A3-7.

```sh
python3 mobile/scripts/check-desktop-parity.py --self-test
mobile/scripts/test-cabi.sh
mobile/scripts/qualify-emulator.sh --evidence-dir "$A3_EVIDENCE_DIR/mobile" --native-tests
```

Keep the known async-route limit of the parity checker in the report.
Use Docker CI from the same source SHA, or run the Docker job in an isolated test environment.
Label Docker evidence as local or remote.
[Release and toolchain findings](../module-review-2026-11-06/reports/R01.md)

## 4. Record the result for handoff

Produce one baseline report with the source SHA, dependency identities, per-check results, and evidence links.
Record each check as passed, failed, skipped, unavailable, or historical.
For each observed failure, keep the test ID or diagnostic signature and the environment.
Classify the failure as reproduced baseline, environmental, unexplained, or blocking.
Treat a failure as blocking when it prevents verification of an affected acceptance criterion.
Leave product fixes to their assigned packets.

Record the target scope from A3. The runtime targets are Linux, macOS, and Windows.
The client targets are browser, desktop, and Android. Tailscale is the supported mobile transport.
The refactor covers native execution, recursive and parallel delegation, and the external harness interface.
Codex and Claude backends, Superagent, public relay, and multi-tenant administration remain later work.
Keep the target scope separate from the platforms and features that this run qualifies.

The coordinator accepts A3-0 when the report identifies the current source, records the required local checks, and accounts for platform gaps.
The report must retain enough evidence to compare later failures in the same environment.
The baseline can include known failures. An unresolved attribution gap blocks the affected A3-1 work.
Keep the accepted report and its raw evidence fixed for later comparisons.
[A3-1 handoff](A3-1-authority-foundations.md)
