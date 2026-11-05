# macOS authentication and backend reliability repairs

F-005 now uses macOS kernel peer credentials, and the 22 failures retained from
the previous repair batch are classified and addressed. The exact retained
selection now reports **15 passed, 7 skipped**. The seven skips require the
existing Linux process supervisor; they are not qualified on macOS.

The default full backend suite passed: **17,001 passed, 75 skipped, 42 warnings,
6 subtests passed**, exit 0. Independent review and root acceptance are **PASS**
for the bounded repair contract. The delivery branch is
`codex/feature-audit-repairs`; evidence was captured before commit packaging.

The full run preceded one final peer-test portability adjustment. That final
test version passed the supplemental checks below; the full suite was not
repeated afterward. All production inputs, other sources, built assets and
dependencies remain identical to the passing full-run candidate. Strict
checking still reports **1,001 existing errors**; migration P6 remains open.

## Authentication repair — F-005

`OwnerIPCServer` reads Linux `SO_PEERCRED` or Darwin `LOCAL_PEERCRED`. The Darwin
path checks the version-0 `xucred` size, version and group-count bounds before
extracting the UID. Both paths reject malformed credentials and compare the
kernel UID with the existing selected-user policy. Missing APIs, unsupported
platforms and failed credential reads remain closed to requests.

Peer authentication still precedes reading/parsing the request. Socket mode
`0600`, request roles, admin tokens and pairing consent retain their existing
owners and checks. No new process-supervision or sandbox backend is introduced.

ABI evidence comes from the installed MacOSX27.0 SDK's `sys/un.h` and
`sys/ucred.h`: `SOL_LOCAL=0`, `LOCAL_PEERCRED=1`, version 0 and a 76-byte `xucred`
with 16 group slots. A real local socket and the real owner subprocess were
exercised on this Darwin host. Linux credential decoding and denial behavior
were exercised with controlled kernel responses; a real Linux run remains
separate platform qualification.

The 19 added peer tests cover same-user/wrong-user credentials, malformed
structures, missing APIs, unsupported platforms, failed system calls before
request consumption, and actual IPC denial before dispatch. The real negative
case changes the selected expected UID after a successful kernel-authenticated
ping; it does not run a second OS account.

Only `nanobot/remote/ipc.py` changes production behavior in this follow-up.
The other 14 changed paths contain tests. Earlier repair batches are preserved,
and this batch changes no frontend sources or dependency/lockfiles.

## Disposition of all 22 retained failures

| Cases | Root cause | Repair and current result |
|---|---|---|
| 5 remote settings/linked-file cases | Linux-only IPC peer check denied every macOS connection | Production peer check repaired; all five pass |
| 7 native process I/O cases | Tests launched a Linux-only subreaper/receipt backend on Darwin | Explicit capability marks on the six owning test functions (one parametrized); seven skips on Darwin |
| 1 opaque native-tool state case | A policy-state fixture depended on Linux host identity discovery | Test-local stable host identity; passes without launching a process |
| 3 permission API/state cases | Unlaunched reservations assumed trustworthy Linux host identity; one Full-mode fixture also assumed supervisor availability | Test-local identity/capability fixtures; all three pass |
| 2 Full-access preview/device-consent cases | Authorization tests assumed an available native backend | Capability fixture limited to these tests; both pass and consent denial assertions remain |
| 1 binary preview case | An image extension was used to represent an unsupported binary file, although images are supported | Generic NUL-containing `.bin` fixture; unavailable assertion passes |
| 2 real command/transport cases | Success was required despite unavailable bubblewrap confinement | Assert success on capable hosts and explicit unavailability otherwise; both pass on Darwin |
| 1 directory-response bound case | Deep fixture exceeded Darwin `PATH_MAX` before exercising the response bound | Short real root with a test-local response limit; overflow assertion passes |

The platform-independent stdin-fence concurrency test and the
no-supervisor/no-fallback denial test still run. Production admission proof,
permission ownership, reconciliation and sandbox guards are unchanged. The
gateway smoke test checks the actual assistant response in both transport and
persisted history, including explicit confinement unavailability on this host.

## Additional full-suite findings and installer isolation

The first full run returned **16,997 passed, 17 failed, 60 skipped**, with
40 warnings and 6 passing subtests. It is retained as failed evidence. The
additional failures were addressed at their owning tests:

| Cases | Root cause | Final disposition on Darwin |
|---|---|---|
| 1 repository sandbox case | Real `/proc` process-environment paths require Linux | Explicit Linux-only skip |
| 8 desktop lifecycle cases | Real PID birth identity and pidfd lifecycle require Linux | Targeted Linux-only marks; two new Darwin/Windows cases verify explicit unavailability without launching a process or creating a lease/state file |
| 1 folder-grant reservation case | A policy test assumed host identity discovery | Test-local stable identity; passes without launching a process |
| 6 native receipt cases | Real host/process identity and receipt publication require Linux | Targeted Linux-only marks; independent denial/unknown-identity cases remain active |
| 1 temporary-session mode case | A mode-selection test assumed native supervision availability | Per-test capability fixture; passes without launching a process |

The same 17 node IDs fail in the frozen baseline and current replays, but the
baseline export lacks ignored WebUI build assets: its eight desktop cases fail
earlier with `webui_assets_unavailable`. Matching node IDs do not prove matching
desktop causes. Current traces and the unchanged explicit Linux gate in
`nanobot/gateway/runtime.py` establish that platform classification.

The first full run also exposed a test that invoked the real Feishu dependency
installer when its optional SDK was absent. It added `lark-oapi`,
`pycryptodome` and `requests-toolbelt`, and changed `websockets` from 16.1.1 to
15.0.1. The test now supplies its installed-dependency assumption explicitly and
forbids installer execution. The exact original **183 package versions** were
restored before replay, strict checking and final full-suite acceptance.
No dependency or lockfile change is part of the repair.

Across this batch, **22 cases gained targeted Linux capability marks**: seven
native I/O, one `/proc`, eight desktop and six native receipt cases. The full
run's 75 skips also include existing optional SDK, Windows, PowerShell and other
host requirements. Skipped native behavior is not qualified by this result.

## Verification

Host: Darwin 27.0.0 arm64 (macOS 27.0.1), Python 3.13.2, pytest 9.1.1,
BasedPyright 1.40.1. The accepted run preserves the frozen dependency inventory
and checker policy; the transient installer mutation above is retained and
excluded from acceptance.

| Check | Result | Evidence |
|---|---|---|
| Original retained failures plus owner IPC before this batch | 26 failed, 1 passed | `red.log`, `red-command.json` |
| Peer regressions before implementation | 13 failed, 6 passed | `peer-red.log`, `peer-red-command.json` |
| Full remote modules and dependent WebUI modules | **151 passed** | `remote.log`, `remote-command.json`, exit 0 |
| Affected native/policy/permission/WebSocket/gateway modules | **291 passed, 22 skipped**, 4 warnings | `affected.log`, `affected-command.json`, exit 0 |
| Exact original 22 failure node IDs | **15 passed, 7 skipped** | `retained.log`, `retained-command.json`, exit 0 |
| Six additional correction modules | **182 passed, 16 skipped**, 10 warnings | `round1-validation.json`, `round1-owning-modules.log`, exit 0 |
| Explicit unsupported desktop behavior | **2 passed** | `round1-desktop-unsupported-platform.log`, exit 0 |
| Full default backend suite without exclusions | **17,001 passed, 75 skipped, 42 warnings, 6 subtests passed**, 907.55 seconds | `full-backend-final.log`, `full-backend-final-command.json`, exit 0 |
| Final peer fixture on this host | **19 passed**; independently repeated by root | `round2-peer-normal.log`, `peer-root-acceptance.log`, exits 0 |
| Final peer fixture with `os.getuid` removed before collection | **18 passed, 1 skipped**; only real kernel IPC skips | `round2-peer-no-getuid.log`, exit 0 |
| Full-source strict checking | **1,001 errors before and after**, 537 files; both exit 1 | `strict-before.json`, `strict-final.json` |
| Global diagnostic comparison | **PASS_NO_NEW_DIAGNOSTICS**, introduced 0, resolved 0 | `types-final-comparison.json`, exit 0 |
| Python lint and whitespace/diff check | Passed | `ruff-final.log`, `round2-ruff.log`, `diff-final.log`, reviewer final diff check |
| Independent authentication/test-fidelity review and root acceptance | **PASS**, AC1–AC7; graph closed | `review-terminal.txt`, `root-final-verdict.json` |

Selections overlap; their counts must not be added as unique coverage. The
earlier peer-test scaffold error and an overlong temporary-path attempt are
retained as `peer-scaffold-failure.*` and `peer-red-long-path.*`; the final red
run uses a short `/tmp` root and fails for the intended credential defects.

After the passing full run, the peer-test module stopped using `os.getuid()`
at import time. Simulated cases now use a fixed UID and a per-test API stub;
only actual kernel IPC requires the host API. This prevents collection failure
on platforms without `getuid`. The normal and missing-API runs above verify
the final fixture. This simulation is not native Windows qualification.
The adjustment is outside the configured checker scope; all **537 production
checker inputs** remain byte-identical to the final strict report and passing
full run. No full-suite rerun after this test-only adjustment is claimed.

Reproduce full backend validation from the repository root:

```bash
.venv/bin/python -m pytest -q -rfs --disable-warnings --durations=25 \
  --basetemp=/tmp/nanobot-macos-full
```

## Evidence and remaining qualification

Raw local evidence: `/tmp/nanobot-macos-repairs-20261105.s00w7c6h`, recorded in
`.git/macos-repairs-evidence`. The immutable baseline is `source-before/`,
`source-before.json`, `environment-before.json` and `strict-before.json`.
The baseline contains **1,922 files**. `source-accepted/` and
`source-accepted.json` retain the **1,923-file** passing full-run candidate;
`source-complete/` and `source-complete.json` retain the final peer fixture.
Their sole source difference is `tests/remote/test_peer_credentials.py`, recorded
in `peer-portability.diff`. Earlier `source-final/` artifacts remain historical.

`environment-complete.json` binds the final source manifest and identical
packages, Python, checker and `pyproject.toml`. The comparator uses the original
repository path for both reports because their diagnostic paths reference that
root. `complete-batch.diff` isolates the 15 changed source/test paths from prior
repair batches. The compact [repair snapshot](./macos-repair-snapshot.json)
records the changed paths, failure classifications and evidence hashes.

Root and reviewer independently verified all **1,923 source hashes**, **537
production inputs**, **590 built assets** and **183 package versions**.
`full-backend-final-verification.json`, `complete-source-verification.json`,
`root-final-precheck.json` and `review-terminal-verification.json` record those
bindings and the full-run timing limitation. Two bounded test-fixture correction
rounds completed; the final verdict is `PASS` / `CLOSED_PASS`.

Strict checking is not clean, and this bounded acceptance does not close P6 or
the architecture audit. Real Linux kernel/lifecycle/confinement, native
Windows/mobile/browser interaction and external service delivery remain
qualification work. Earlier frontend evidence (**1,468 passed**, build/lint
passed) remains separate. Original audit and repair observations are preserved
as historical evidence.
