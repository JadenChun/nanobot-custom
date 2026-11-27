# A3-2M macOS Seatbelt qualification — 2026-11-27

## Status

**AC8.1 and AC8.2 pass.** The workload matrix found limits that were put to the user before the
executor could default to on. **Update 2026-11-27: the owner chose on by default, as on Linux, and
the `tools.exec.macosSeatbelt` switch was removed.** The limits below remain documented limits. This record covers the
macOS executor in [packet A3-2M](../../audits/implementation-packets/A3-2M-macos-seatbelt-executor.md)
on branch `codex/macos-seatbelt-executor`. It is not full A3 or merge acceptance.

## Environment

- Evidence host: macOS 27.2.1 (26B131), Apple Silicon, checkout Python 3.13.2.
- CI host: GitHub `macos-latest`, Python 3.14.
- Tools on the evidence host: git 2.53.0, gh 2.104.0, Node 22.22.1, npm 10.9.4, pnpm 10.34.5,
  Bun 1.3.10, Yarn 1.22.22, uv 0.10.9, Homebrew 7.0.13, Go 1.25.12, Docker CLI 29.9.2,
  Xcode 27.0 with Swift 6.4.

## Acceptance

| Criterion | Result | Evidence |
| --- | --- | --- |
| AC8.1 zero false positives | PASS | Evidence host: 20 fresh random launch ids over about 680 live processes matched 0, with the real self-test and profile probe passing. `test_whole_table_scan_with_fresh_ids_matches_nothing` runs in the local suite and in CI. |
| AC8.2 installed wheel | PASS | CI run 45941759662 at `569619849`, job 140317489927: the wheel was installed in a clean environment and tested from a folder without the source tree, with real signals: 239 passed, 7 skipped. The same job's source-tree run gave the same counts. |
| AC8.3 default on | NOT MET, by choice | Waits for the decisions below. |

Local installed-wheel check (no signalling tests): 215 passed, 31 skipped. The wheel holds
`full.sbpl`, `managed.sbpl`, `network.sbpl`, `LICENSE` and `THIRD_PARTY_NOTICES.md`. The CI
wheel is built with `KOHAKU_SKIP_WEBUI_BUILD=1`, so it does not check the bundled WebUI.

The macOS tests cover Stop (`test_stop_kills_escaped_descendants_and_publishes_a_receipt`),
timeout (`test_deadline_settles_descendant_without_poll`), session close
(`test_stopping_a_managed_session_ends_its_detached_descendant`) and task-access retirement
(`test_once_sandbox_descendant_is_supervised_until_owner_receipt`). The skipped cases include four
Linux-subreaper scenarios in `test_managed_exec.py`, which have no macOS run.

## Workload matrix

### Method

Each case ran three ways with the same command: unconfined (control), Full access and Sandbox
with network on. Full access and Sandbox used the product's `SeatbeltBackend` and, for Sandbox,
`prepare_managed_seatbelt` with one read-write project folder. The launch domain was replaced by
a stand-in that cannot signal, so **this matrix does not exercise supervision or killing**; CI
does that. After each root exit, a read-only membership scan looked for leftover processes: none
in any case.

The harness differs from a real gateway in these ways:

- Package managers got empty user configuration, so no registry credentials were sent.
- Homebrew auto-update and analytics were off; `GOTOOLCHAIN=local`.
- Control and Full access redirected tool caches to a temporary folder, to leave the user's
  caches alone.
- Project folders were under `/private/tmp`, which Sandbox hides apart from granted folders.
  "Agent home" cases ran with the agent home, inside hidden instance storage, as the working
  folder.

The harness script is not committed.

### Results

| Workload | Full access | Sandbox | Notes |
| --- | --- | --- | --- |
| git: init, commit, log (also agent home) | pass | pass | |
| git over HTTPS (`ls-remote`) | pass | pass | Refused with network off, as intended. |
| gh API call with an invalid token | pass | pass | HTTP 401 proves TLS and network. Real sign-in not tested. |
| curl over HTTPS | pass | pass | Refused with network off. |
| npm install (also agent home) | pass | pass | |
| pnpm add, bun add | pass | pass | |
| yarn, corepack pnpm | pass | pass | Version commands only. |
| uv venv and install (also agent home) | pass | pass | |
| python venv and pip install | pass | pass | |
| pytest with xdist and multiprocessing (with and without config; also agent home) | pass | pass | |
| brew version, list, info | pass | pass | Read-only. Sandbox about 3.5 s against 0.5 s. |
| C compile and run | pass | pass | Sandbox about 2 s against 0.1 s; `xcrun` cannot write its cache. |
| Go run | pass | pass | |
| `xcodebuild -version`, `xcrun --find` | pass | pass | Sandbox about 4.5 s against 0.1 s, with cache errors. |
| `swift build` (default) | **fail** | **fail** | SwiftPM applies its own sandbox (G2). |
| `swift build --disable-sandbox` | pass | **fail** | Sandbox: `couldNotFindTmpDir` for the per-user `/var/folders` temp folder; link fails after about 65 s. |
| `xcodebuild` package build (default) | **fail** | **fail** | Full access: nested sandbox during package resolution. |
| `xcodebuild -IDEPackageSupportDisableManifestSandbox=YES` | pass | **fail** | Sandbox: cache writes and Xcode services are refused (exit 66). |
| Node and Python local servers | pass | pass | A client outside the sandbox got a response. With network off, listening is refused. |
| PTY programs (`pty.fork`, `script`) | pass | pass | |
| `mktemp` with a template in `TMPDIR` | pass | pass | |
| `mktemp` without a template | pass | **fail** | Per-user temp folder is not writable. |
| setuid program (`ps`) | **refused** | **refused** | Seatbelt rule (E13). |
| a second, different `sandbox-exec` | **refused** | **refused** | G2, exit 71. |
| `KOHAKU_SANDBOX` marker | set | set | |
| Docker CLI version | pass | pass | The daemon was not running, so daemon access is **not tested**. |

Hidden folders refuse listing on macOS, where Linux shows them empty. An earlier pytest run forced
to walk the hidden parents of its project (`-c /dev/null`, no root configuration) failed in Sandbox
with `PermissionError` on `/private/tmp`. The normal layouts above passed.

### Keychain (G3)

The probe read the login Keychain's lock settings, listed keychains and searched for an item that
does not exist. It read no secret and changed nothing.

| Profile | Keychain service reachable |
| --- | --- |
| Unconfined, Full access | yes |
| Sandbox, network on | **yes** |
| Sandbox, network off | no |
| Sandbox, network on, with the `com.apple.SecurityServer` rule removed (harness only) | no |

With that rule removed, the git, gh, curl, npm, pnpm, bun, uv, pip, brew and Go cases still
passed. The Swift and Xcode cases that already fail in Sandbox were not changed by it.

**Not established:** whether a Sandbox command can read a stored secret. Testing that needs a
Keychain item created for the purpose, with the user's permission. Until then, assume a Sandbox
command with network access can use the Keychain as far as each item's access list allows. The
shipped profile is unchanged.

## Not run

git over SSH and signed-in `gh` (they use the user's credentials), the macOS Keychain git
credential helper, the Docker daemon, `brew install`, long-lived development servers, Intel
Macs, and a side-by-side comparison with Codex. Children started through launchd or XPC cannot be
seen by the leftover scan.

## Review fixes

Review of the branch found three defects. They were fixed after the runs recorded above, so those
results and CI run 45941759662 do not cover the fixes.

1. **Other terminals in Sandbox.** The blanket file read came after the terminal rule, so a
   command could open another terminal of the same user, read typed input and change its mode.
   The profile now denies reads and ioctls on `/dev/ttys*` after the blanket read, and allows them
   again only for ptys the command opened itself. A test runs this under the real `sandbox-exec`.
2. **Kill during a pending output write.** If one select batch held a kill (or a control EOF) and
   an output write, the supervisor raised `KeyError` and exited with 125 without a receipt. On macOS
   this left escaped descendants and the launch temporary folder behind. The relay now skips events
   whose registration changed earlier in the same batch. The test replaces signals and the receipt,
   so it does not exercise real cleanup.
3. **Cached host check failures.** One failed self-test or profile probe disabled the macOS
   executor until restart. Success is still cached for good. A failure is now retried after five
   minutes, on the next call, and concurrent callers share one attempt.

## Decisions that were needed before default on (now accepted as limits)

1. **setuid programs in Full access.** `sudo`, `ps` and `top` are refused. Accept this, or keep
   the switch off.
2. **Keychain in Sandbox.** The G3 recommendation was to deny the Keychain by default with an
   explicit allowance. The shipped Codex-style rule set reaches it when network is on.
   Recommended: remove `com.apple.SecurityServer` by default and add a configuration allowance.
   The cost is that Sandbox commands lose Keychain-backed sign-ins.
3. **Swift and Xcode builds.** Accept that package builds need Full access plus a flag that turns
   off the tool's own sandbox. Making them work in Sandbox would need access to the shared
   per-user temp folder, which is not recommended.
