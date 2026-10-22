# P0 candidate fix supplement

Date: 2026-09-08. This supplement records a candidate overlay over the immutable staging
baseline. It does not change the historical [868-case result](p0-mapped-results.tsv), its
`adc2b3e2473fc5085e9d0d293642a2b17f33002a12c33b050d606e490f52cffd`
SHA-256, the 158-entry inventory, or the baseline runner.

## Candidate result

The corrected `integrated-accepted` candidate ran against staging
`4aa562d92828189b3fcefad079773efbab1d715a` with a six-file source overlay. It collected 876
cases across 62 files and completed with **876 passed, 33 warnings, 0 failures, 0 skips, 0
errors, exit 0** in 30.93 seconds. The candidate contains all 868 historical node IDs, each
passing, plus eight new PowerShell launcher cases. The [candidate results](p0-candidate-results.tsv)
and [candidate manifest](p0-candidate-manifest.json) preserve the result summary, invocation,
JUnit, log and overlay hashes. The durable [per-case candidate results](p0-candidate-cases.tsv)
preserve the 876 JUnit outcomes and durations after the temporary evidence directory is removed;
their SHA-256 is `17800ea80c448bf99082a4e739581884f3e44c05aeeedb693c9c4c562b339bc2`.

The two historical P0 defects are resolved in this candidate:

| Historical gate | Candidate change | Candidate evidence |
| --- | --- | --- |
| `FILE-132` order-dependent redaction capture | A CLI-test autouse fixture snapshots and restores Loguru's private activation state after each test. This test-only `_core` use preserves production logging behavior but depends on Loguru's internal state shape. | The 65-test CLI-then-safe-push red-to-green sequence passes; the historical safe-push test also passes after the full CLI file in the combined 876-case candidate run. |
| `FILE-137` missing deployment fixtures | Added bounded Windows launcher/profile scripts, a shared script helper, mock PowerShell coverage, and Windows guidance. | All seven existing deployment-bundle assertions and eight new script cases pass in the candidate run. |

The source overlay is mapped separately to existing P6 release/rehearsal findings in
[p0-p6-fix-map.tsv](p0-p6-fix-map.tsv). It supplements the historical inventory; it does not
reclassify any historical `BASELINE-*` or `NOT-RUN` row.

## Candidate boundary and limits

The candidate process used disposable HOME/XDG directories, venv-first `PATH`, no user site,
loopback-only network access, and a portable PowerShell 7 runtime on macOS. The PowerShell tests
use mocked Windows commands, a harmless local child process, and a loopback JSON-lines fixture.
They do not qualify a real Windows Edge installation, CIM process inspection, browser login,
provider, crawler, service, deployment, credential, or user configuration.

`DOTNET_SYSTEM_NET_DISABLEIPV6=1` was set only in the candidate test environment because the
macOS sandbox blocks the default .NET mapped-IPv6 connection path. It is not a launcher setting
or a product requirement. The candidate includes no upstream merge, adapter, migration,
build-artifact, deployment, conversion, or live-release validation.

## Review and coordination status

**P0 checkpoint status: P0-CANDIDATE-ACCEPTED.** Independent scope and coverage review and the
final narrow recheck found no remaining findings. `FILE-132` and `FILE-137` are closed at the P0
candidate boundary, so P1 may begin under the migration plan's remaining package gates.

Thread capacity required sequential reuse of the exact-role implementation worker and source
analyst while corrections were made. That scheduling constraint did not replace the completed
independent integrated review. Review findings drove the two launcher corrections recorded in the
candidate sources; the final narrow recheck accepted this supplement with no remaining findings.
