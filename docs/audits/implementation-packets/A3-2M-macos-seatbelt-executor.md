# A3-2M — macOS Seatbelt executor (Sandbox and Full access)

**Status.** WP0-WP3 are implemented on branch `codex/macos-seatbelt-executor`; see §11. All gates are decided (D-M6, D-M7). WP3's real-signal criterion (AC3.2) has not run yet and should pass on a disposable macOS host before WP4 relies on killing. Linux behaviour is unchanged and macOS still fails closed.
**Platform.** macOS 27.2.1 (Apple Silicon) is the evidence host. Linux behaviour stays the same.
**Parent packets.** [A3-2 runtime and processes](A3-2-runtime-process-ownership.md) §2 asks for the executor-mode matrix and per-OS proofs before a new backend. This packet supplies them for macOS.
**Design references.** [A3 design capability table](../../architecture/a3/design.md) · [D1 platform scope](../../architecture/a3/decisions.md) · [macOS qualification 2026-11-16](../../architecture/a3/macos-qualification-2026-11-16.md) · OpenAI Codex Seatbelt implementation (`codex-rs/sandboxing/src/seatbelt.rs`, `seatbelt_base_policy.sbpl`, `codex-rs/core/src/spawn.rs`, read on 2026-11-26)

## 1. Problem

A macOS gateway has no way to run commands:

| Mode | Gate today | macOS result |
| --- | --- | --- |
| Sandbox (managed) | `managed_bwrap_available()` — `kohaku/agent/tools/sandbox.py:143`, Linux + bubblewrap only | Shell unavailable |
| Once/task commands | Bubblewrap **and** the native supervisor — `kohaku/session/workspace_focus.py:213-227` | `supervisor_unavailable` |
| Full access | `native_supervisor_available()` — `kohaku/agent/tools/native_process.py:263`, `sys.platform == "linux"`. Proof comes from `prctl(PR_SET_CHILD_SUBREAPER)` plus `ECHILD` (`native_process_supervisor.py`) | Disabled with `supervisor_unavailable` |

The refusals are intentional and correct. The macOS executor was never built. This packet builds it without weakening the invariant that **unrestricted or confined authority retires only after the command's whole process tree is proven quiescent**. When proof is missing, the system still refuses or reports `unknown_processes`.

## 2. Accepted decisions

- **D-M1 (user, 2026-11-26).** macOS uses Seatbelt (`/usr/bin/sandbox-exec`) for Sandbox confinement **and** for process-tree ownership in Full access. We do not adopt Codex's best-effort process-group cleanup for Full access.
- **D-M2.** Linux behaviour does not change. The shared seam must give identical admission decisions and reason codes on Linux, proven by a parity test (WP1).
- **D-M3.** Codex is the reference for confinement profile structure, not for process lifetime. On macOS, Codex uses process groups plus `kill_on_drop`. Its parent-death signal is `#[cfg(target_os = "linux")]`. Its Full mode runs with no tracking. Its `pid_tracker.rs` (kqueue `NOTE_FORK` + `proc_listchildpids`) only supports the debug sandbox's denial logging.
- **D-M4.** The reason codes on the wire stay the same: `supervisor_unavailable` and `sandbox_unavailable`. Only additive values are allowed, such as the `commands.backend` value `"seatbelt"`.
- **D-M6 (user, 2026-11-26).** G5 is decided: Apple Silicon only. Intel Macs stay refused until an x86_64 host passes the self-test.
- **D-M7 (user, 2026-11-26).** The recommended option is accepted for the remaining gates: G1(a) accept the private `sandbox_check` and the deprecated `sandbox-exec` behind a fail-closed self-test; G2(a) refuse nested-Seatbelt tools with a clear error and export a `KOHAKU_SANDBOX=seatbelt` marker so tools can detect it; G3(b) Codex-style Mach-service allowlist only when network is allowed, with Keychain access tested before any claim is made (WP8); G4(a) `input_files` are read live at the approved path; G6(a) CI allows exactly one named `macos` job.
- **D-M5.** Ship behind a config switch that defaults to off until WP8 qualification passes, then turn it on by default. With the switch off, macOS behaves as it does today: fail closed.

## 3. Gates (all decided on 2026-11-26)

The table keeps the options as they were put to the user. The outcomes are in D-M6 (G5) and D-M7 (G1-G4, G6); where an outcome and a recommendation differ in detail, the D-M entry wins. In particular, Keychain access under G3 is unverified until WP8 tests it.

| Gate | Question | Options | Recommendation | Blocks |
| --- | --- | --- | --- | --- |
| **G1** | Do you accept that proof depends on Apple's private `sandbox_check` and the deprecated `sandbox-exec`? | (a) Accept, with a startup self-test and fail-closed behaviour. (b) Do not ship Full access on macOS. | (a). Both still exist on 27.2.1. Chrome and Codex depend on `sandbox-exec`/Seatbelt. A self-test failure disables the feature, so the result is safe either way. | WP2-WP6 |
| **G2** | What happens to tools that apply their own Seatbelt sandbox? Nested `sandbox_apply` fails (E3); examples are `sandbox-exec`, some `swift build`/Xcode paths, other agent CLIs. | (a) Refuse with a clear error. (b) Offer an approved untracked escape that forces `unknown_processes` retirement. | (a) now. Revisit (b) after the WP8 workload matrix. | WP4, WP5 |
| **G3** | Can Sandbox mode reach Keychain and system Mach services? `gh`/`git` credential helpers on macOS use the Keychain, but Linux bubblewrap reads credential *files*. | (a) Deny by default with an explicit config allowance. (b) Allow Codex's network Mach-service set only. (c) Allow the Keychain. | (b) when network is allowed, plus (a) for the Keychain. | WP5 |
| **G4** | What do `input_files` mean on macOS? Seatbelt cannot remap paths, so it can't show a read-only snapshot at the requested path the way bubblewrap `--ro-bind` does. | (a) Read the live file at the approved path; `_revalidate_input_files` already checks before launch. (b) Refuse `input_files` on macOS. | (a), recording the weaker snapshot consistency in the matrix. | WP5 |
| **G5** | Do Intel Macs count as qualified? | (a) Supported only if the x86_64 self-test passes on a real Intel host. (b) Apple Silicon only. | (b) until an Intel host is qualified. The self-test still refuses unknown ABIs. | WP2 |
| **G6** | `CI policy`: `tests/migration/test_p6_release_rehearsal.py` requires **every** CI job to run on `ubuntu-latest` (a release gate). A macOS job cannot land without changing that test. | (a) Allow exactly one named `macos` job in that test. (b) Keep CI Linux-only and qualify macOS manually on a disposable host. | (a), with an allowlist of that one job id so the gate still rejects any other non-Ubuntu job. | WP0 CI job, AC8.1 in CI |

## 4. Evidence from this host (2026-11-26, macOS 27.2.1 arm64, repo `.venv` Python 3.13.2)

| ID | Observation |
| --- | --- |
| E1 | `/usr/bin/sandbox-exec` is present. `kern.bootsessionuuid` and `IOPlatformUUID` are available. The SDK `sys/event.h` says `NOTE_TRACK` and `NOTE_CHILD` are "no longer supported as of 10.5". |
| E2 | Confinement profile: writes inside the allowed subpath succeeded and writes outside were denied. A later `allow` overrides an earlier `deny` (later rule wins). `deny file-read* (subpath …)` blocked both direct and symlink reads. `deny network-outbound (remote ip)` made `curl` fail (exit 7). |
| E3 | Inside an allow-default Seatbelt profile, `sandbox-exec` failed with `sandbox_apply: Operation not permitted` (exit 71). A member can't replace or remove its sandbox. |
| E4 | A `setsid` + double-fork descendant was reparented to launchd (ppid 1) and **stayed sandboxed**. |
| E5 | `sandbox_check(pid, NULL, 0) == 1` identifies sandboxed processes. `sandbox_check(pid, "mach-lookup", 2 /*global-name*/, name)` reports the target's profile decision for a name. The **two-marker** predicate (allowed `member` name **and** denied `fence` name) matched exactly the launch's processes, including the escaped ones, out of 675. Control markers matched nothing. A native scan took about 1 ms. |
| E6 | **A single deny-marker gives false positives.** App-Sandboxed apps and system agents also deny unknown Mach names. During probing, a kill loop on that predicate killed user-session processes: Safari, Docker Desktop and its VM, WebKit content, and launchd agents. This incident drives the safety invariants in §6. |
| E7 | A plain `ctypes` call to variadic `sandbox_check` returns wrong results on arm64. Apple's arm64 ABI passes variadic arguments on the stack. Padding x3–x7 so the filter argument lands on the stack returns correct results. |
| E8 | Path-filter checks return "denied" for paths that don't exist, which makes them unsuitable as markers. Mach global-name markers need no filesystem artifact. |
| E9 | `sandbox_check` on a **nonexistent** pid, or an unreaped exited child, reports sandboxed with every name denied. Both are therefore non-members. |
| E10 | `PROC_PIDTBSDINFO` returned nothing for 213 of 784 processes (other users). `PROC_PIDT_SHORTBSDINFO` answered for all 784 and agreed on pid, ppid and uid where both worked. It does not answer for a zombie. |
| E11 | A scan of 786 processes took about 7 ms in Python. Polling every 20 ms would keep a large share of a core busy while a long-lived tree runs. |

## 5. Executor capability matrix (target state)

| Guarantee | Linux (unchanged) | macOS (this packet) |
| --- | --- | --- |
| Sandbox confinement | Bubblewrap: read-only `/`, grant binds, tmpfs masks, private `/tmp` | Seatbelt deny-default profile: global read, deny-read private roots, write only to grants, the agent home and a per-launch temp dir |
| Sandbox tree ownership | Bubblewrap `--unshare-pid` / `--die-with-parent` + process group | Seatbelt launch domain (§6) through the Darwin supervisor |
| Once/task commands | Bubblewrap + subreaper supervisor | Seatbelt confinement + launch domain |
| Full access ownership | Subreaper supervisor + `ECHILD` receipt | Allow-default Seatbelt profile with markers + launch-domain receipt |
| Network policy | `--unshare-net` when local binding is disallowed | `deny network*` except allowlisted Unix sockets; G3 decides the Mach services |
| Process-tree receipt | `renameat2(RENAME_NOREPLACE)`, `/proc` identity | `renamex_np(RENAME_EXCL)`; `gethostuuid`/`kern.bootsessionuuid`; `pid_namespace = 1` (one host namespace); start time from `proc_pidinfo(PROC_PIDTBSDINFO)` |
| Owner IPC peer check | `SO_PEERCRED` | `LOCAL_PEERCRED` is already in `kohaku/remote/ipc.py:174` (no work) |
| Known gaps vs Linux | — | Nested Seatbelt is refused (G2). `input_files` are read live (G4). Children started through launchd/XPC (`open`, Apple Events) are outside the tree, as with D-Bus activation on Linux. |

## 6. Mechanism: the Seatbelt launch domain

Each launch gets a random 128-bit `launch_id`. The generated profile includes:

```scheme
(allow mach-lookup (global-name "dev.kohaku.launch.<id>.member"))
(deny  mach-lookup (global-name "dev.kohaku.launch.<id>.fence"))
```

Write the profile to a 0600 file in private runtime storage and pass it with `-f`. The marker names then never appear in `argv`. Only path values go through `-D` parameters, which avoids SBPL string injection.

**Membership** `M(p)`: `sandbox_check(p, NULL, 0) == 1` **and** the member name is allowed **and** the fence name is denied.

**Proof argument.** These are assumptions, and WP8 tests them:
1. A Seatbelt profile is inherited across fork/exec, and a process cannot remove or replace it (E3, E4).
2. No process outside the tree can become a member without the private profile file.
3. `proc_listallpids` returns every live pid at the moment of the call.

From 1–3: if a full scan taken after the root exits finds no members, no member exists later. Any later member would have to descend from a member that was alive at the scan. To cover assumption 3, require **two consecutive empty scans**, a short interval apart.

Two conditions make the argument hold (added in WP3):
- **A scan is a fixed point.** Checking is not atomic with listing, so a process that forks and exits in between could be missed. A scan counts as empty only when a fresh listing holds no pid that hasn't been checked. A table that never settles is not empty.
- **The root must have exited.** The root is not a member while `sandbox-exec` is still applying the profile, so empty scans before its exit prove nothing.

An unknown `sandbox_check` answer blocks the proof. It is never treated as "not a member" here, and never signalled.

**Safety invariants for signalling.** These are mandatory because of E6:
- **K1.** Signal a pid only if `M(p)` is re-checked immediately before `kill`, its uid equals our effective uid, and it is not pid 0/1, the gateway, the supervisor, or any of their ancestors.
- **K2.** Read the start time before the membership check and again just before `kill`. A mismatch means the pid was reused: skip it.
- **K3.** Bounded loop: scan, kill members, repeat until two scans are empty. If the deadline passes, or a member can't be signalled (a setuid child, `EPERM`), the result is **not quiescent**. The ledger then records it as unproven, and the UI shows the existing `unknown_processes` retirement state.
- **K4.** All kill logic runs behind injectable `ProcTable`/`Signaller` interfaces. Unit tests use fakes. Real-signal tests run only on disposable hosts (WP0).
- **K5.** Never signal a process that started before the supervisor, because it cannot be a descendant. This holds even if the membership predicate is wrong. A backwards wall-clock step can only make the result not quiescent.

## 7. Work packages (dependency order)

Write sets don't overlap. Every package keeps the strict-type repair gate from `AGENTS.md` (§8).

### WP0 — Test safety harness and macOS CI (gate-independent)
- **Objective.** Make it safe and repeatable to test macOS process behaviour.
- **Owns.** `.github/workflows/ci.yml` (a new `macos-latest` job), `tests/conftest.py` or `tests/darwin/conftest.py` (a `darwin_host` marker, skipped unless `sys.platform == "darwin"` **and** `KOHAKU_DARWIN_DESTRUCTIVE_TESTS=1`), and `pyproject.toml` marker registration.
- **AC0.1.** Real-signal tests run only in the CI macOS job or with the explicit opt-in variable. They are skipped on developer machines by default.
- **AC0.2.** A non-destructive smoke test, safe anywhere on macOS: scan all live processes with a fresh random launch id and assert **zero** members. This guards against regressions of E6.

### WP1 — Executor capability seam + Linux parity (gate-independent; lands first)
- **Objective.** Replace the scattered Linux booleans with one typed capability, without changing Linux behaviour.
- **Owns.** A new `kohaku/agent/tools/executor_capabilities.py`, plus call-site edits in `kohaku/webui/workspaces.py` (~256-280, 876-1371), `kohaku/security/desktop_permissions.py` (~534-548 parameter source only), `kohaku/session/workspace_focus.py` (~157-245, ~877), `kohaku/session/workspace_access_recovery.py` (~718, ~1057), `kohaku/agent/tools/shell.py` (`_sandbox_backend`, `_process_ownership_identity`, `_native_command_refusal`, `_spawn` selection only), and `kohaku/harness/claude_sdk.py:124` (keeps its Linux requirement).
- **Shape.**
  ```python
  @dataclass(frozen=True)
  class ExecutorCapabilities:
      platform: Literal["linux", "darwin", "other"]
      confinement: Literal["bwrap", "seatbelt"] | None
      tree_ownership: Literal["linux_subreaper", "darwin_seatbelt"] | None
  ```
  The module also provides cached `executor_capabilities()`, `managed_confinement_available()` and `tree_ownership_available()`. `native_supervisor_available()` and `managed_bwrap_available()` stay as compatibility wrappers, because existing tests monkeypatch them (for example `kohaku/channels/websocket/tests/test_websocket_channel.py:6296`).
- **AC1.1 (parity).** With the platform forced to Linux, a table-driven test passes for all combinations of bwrap available/unavailable × supervisor available/unavailable × mode {managed, task, Full, legacy native}: admission results, `unavailable_reason`, `commands.reason` and `commands.backend` match the values computed by the pre-change code. Freeze them as expected values in the test.
- **AC1.2.** On Darwin with the feature switch off, the capability is `confinement=None, tree_ownership=None`. Every macOS projection is byte-identical to today's (`supervisor_unavailable`).
- **AC1.3.** The existing suites pass unchanged: `tests/security/test_native_admission.py`, `tests/security/test_workspace_focus_full_projection.py`, `tests/webui/test_permission_v2_api.py`, `tests/agent/test_managed_exec.py`, `tests/tools/test_exec_native_reservation.py`, and `kohaku/channels/websocket/tests/test_websocket_channel.py`.

### WP2 — Darwin platform primitives (after G1, G5)
- **Owns.** A new `kohaku/security/darwin_proc.py`, plus Darwin branches in `kohaku/security/native_admission.py`: `current_host_identity`, `process_start_time`, `process_pid_namespace`, `_rename_noreplace`.
- **Contents.**
  - `proc_listallpids`, and `proc_pidinfo(PROC_PIDTBSDINFO)` for uid, ppid, status and start time.
  - A `sandbox_check` binding with **ABI-specific calling**: arm64 pads x3–x7 (E7). Any other ABI is refused until G5 qualifies it.
  - `gethostuuid`, `kern.bootsessionuuid`, and `renamex_np(RENAME_EXCL)`. Confirm the constant against the SDK `stdio.h`.
  - `seatbelt_self_test()`. It launches one tiny marked process, then asserts: it is a member, a foreign-marker probe matches nothing, an unsandboxed process is not a member, and a deny-default probe process without our member name is not a member. Any failure makes `tree_ownership=None`, so the feature fails closed.
- **AC2.1.** Receipts written on Darwin pass the **unchanged** `_validate_receipt`, with `pid_namespace == 1`.
- **AC2.2.** `HostIdentity.trusted` is true on Darwin. `_verified_later_boot` is true after a simulated boot-session change.
- **AC2.3.** The self-test passes on the evidence host. Forcing a wrong ABI makes it fail closed.
- **AC2.4.** Linux branches stay byte-identical. `git diff` shows only additive `sys.platform == "darwin"` branches.

### WP3 — Launch-domain tracking engine (after WP2)
- **Owns.** A new `kohaku/agent/tools/seatbelt_tracking.py`.
- **API.** `LaunchDomain(launch_id)` provides `profile_markers()`, `members()`, `wait_quiescent(deadline)`, and `kill_until_quiescent(deadline) -> Quiescent | NotQuiescent(reason)`. It enforces K1–K4.
- **AC3.1 (fake table).** Unit tests confirm these are never signalled: a foreign deny-default sandbox, a different launch's members, unsandboxed processes, members with another uid (these produce `NotQuiescent` instead), and a pid whose start time changed.
- **AC3.2 (`darwin_host`).** All of these reach `Quiescent` with two empty scans: escaped descendants (`setsid` double-fork, `setpgid`), a fork storm, and a member that ignores `SIGTERM`.
- **AC3.3.** `wait_quiescent` without killing returns `True` only after the tree has exited by itself.

### WP4 — Darwin supervisor for Full access (after WP3, G2)
- **Owns.** A new `kohaku/agent/tools/native_process_supervisor_darwin.py`; a Darwin branch in `spawn_native_process` in `kohaku/agent/tools/native_process.py` that picks the supervisor module (same pipes, gate, status protocol and `NativeProcessHandle`); and a profile template `kohaku/agent/tools/seatbelt/full.sbpl` (allow-default + markers) registered as wheel package data.
- **Behaviour.**
  - The supervisor itself is unsandboxed. It starts `/usr/bin/sandbox-exec -f <profile> -- <shell> -c <cmd>` in a new session (absolute path only, as Codex does) and relays I/O exactly as the Linux supervisor does.
  - After the root exits, it calls `kill_until_quiescent` on request, or `wait_quiescent` otherwise. It writes the receipt only on `Quiescent`.
  - If a command tries nested Seatbelt (G2a), it gets a clear refusal: a stderr note from a pre-exec probe, or detection of exit 71 + `sandbox_apply`.
- **AC4.1.** Enabling Full access on macOS succeeds when the capability is present. The projection shows `backend: "native"` and `reason: null`.
- **AC4.2 (`darwin_host`).** A command that leaves a `setsid` daemon behind: Stop or timeout kills it, a receipt is written, and Full access retires cleanly. The existing tests `tests/agent/test_native_process_io.py` and `tests/tools/test_exec_session_deadline_tree.py` pass on macOS.
- **AC4.3.** An unkillable member leads to `mark_unproven` and retirement `unknown_processes`. Full access stays unavailable until reconciled, the same semantics as Linux.
- **AC4.4.** Gateway restart with an unreceipted launch: the ledger reports unknown, as on Linux. An *optional* follow-up is to re-scan by launch id to prove quiescence after a restart; that is stronger than Linux, so it's out of scope here.

### WP5 — Seatbelt Sandbox confinement (after WP4, G3, G4)
- **Owns.** In `kohaku/agent/tools/sandbox.py`, a new `_wrap_managed_seatbelt`, a `managed_seatbelt_available()` probe, and a Darwin `_overlaps_system_control` set (`/dev`, `/System`, `/private/var/run`). Also a new profile template `kohaku/agent/tools/seatbelt/managed.sbpl`, adapted from Codex's `seatbelt_base_policy.sbpl`, with attribution and a licence check. In `shell.py`, `_prepare_command` gets a Darwin branch for `apply_managed_sandbox`, and managed launches route through the Darwin supervisor so they get launch-domain ownership.
- **Profile (mirrors `wrap_managed_command`).**
  - `(deny default)`, process exec/fork, and `signal`/`process-info` with `(target same-sandbox)`.
  - `(allow file-read*)`, then `deny file-read* file-write*` for `grants.sandbox_instance_root`, `grants.sandbox_private_roots`, managed-repository roots, other entries in `/private/tmp`, and the user's `$TMPDIR`.
  - `allow file-write*` for the agent home, read-write grants, read-write Once roots, and a per-launch private temp dir. The cache variables (`TMPDIR`, `XDG_CACHE_HOME`, …) point inside that dir.
  - `/dev/null` and PTY rules taken from Codex. Network follows G3 and `allow_local_binding`. Unix-socket allowlist entries become `network-outbound (path …)`.
  - The launch-domain markers.
  - Paths are passed with `-D` parameters, ordered most-specific-last. Later rules override earlier ones (E2).
- **AC5.1 (golden).** Profile-generation tests cover parameter escaping, ordering (private masks win over a broad writable parent), and refusal for system-control overlap.
- **AC5.2 (`darwin_host`).**
  - Writes outside the grants are denied; reading a private root (including through a symlink) is denied.
  - A Sandbox command can write in the agent home and a read-write grant, but not in a read-only grant.
  - With network disabled, `curl` fails.
  - Escaped descendants are killed on Stop.
- **AC5.3.** `tests/agent/test_managed_exec.py` scenarios run on macOS with the Seatbelt backend. Linux runs of the same file stay unchanged.

### WP6 — Projection, Once/task commands and recovery wiring (after WP5)
- **Owns.** `kohaku/session/workspace_focus.py` (`command_ready` uses the capability, backend `"seatbelt"`), `kohaku/webui/workspaces.py` (Full eligibility), `kohaku/session/workspace_access_recovery.py`, and `kohaku/agent/tools/exec_session.py` (interactive sessions use `NativeProcessHandle` on Darwin).
- **AC6.1.** On macOS, Once/task commands report `available: true, backend: "seatbelt"` when granted.
- **AC6.2.** Interactive `exec_session` works under both modes. Session close and deadlines kill escaped descendants (`darwin_host`).
- **AC6.3.** The Linux parity test from WP1 still passes.

### WP7 — WebUI (after WP6)
- **Owns.** `webui/src/lib/types.ts` (add `"seatbelt"` to the backend union at ~751), `webui/src/components/thread/WorkspaceControls.tsx` (no logic change expected), and locale copy only if new strings are needed.
- **AC7.1.** `cd webui && bun run test` and `bun run build` pass. A test confirms the menu offers Full access when `can_enable_full` is true on a macOS projection.

### WP8 — Qualification and documentation (last)
- **Owns.** `docs/architecture/a3/design.md` (capability table rows), a new `docs/architecture/a3/macos-seatbelt-qualification-<date>.md`, and this packet's status.
- **Workload matrix.** Run each workload under both profiles and record pass/fail: `git`, `gh`, `npm`/`bun`/`pnpm` install, `uv`/`pip`, `brew` (read-only), `docker` CLI, `xcodebuild`, `swift build`, `python -m pytest`, long-running dev servers, and PTY programs. Failures caused by nested Seatbelt are classified under G2.
- **AC8.1.** WP0's zero-false-positive smoke test passes on the evidence host and in CI.
- **AC8.2.** An installed-wheel check on macOS: Sandbox and Full commands run; Stop, timeout and retirement behave as specified; profiles are included as package data.
- **AC8.3.** The feature switch defaults to on only after AC8.1–8.2 pass. Rollback means switching it off, which returns to today's refusal.

## 8. Repository gates (every package)

- Focused `pytest` for the package, then the full local suite before closing the stage. On Linux and macOS: `.venv/bin/python -m pytest -q -n 4 --dist loadfile`.
- `ruff check kohaku/ tests/`.
- Strict types: `uv run --no-sync basedpyright`, following `docs/migration/p6-error-attribution.md`. Take frozen before/after full-source reports in the same dependency environment, compare with `scripts/check_type_regressions.py`, and allow zero new signatures in any file. Keep source hashes and command evidence. Never refresh the baseline to absorb new errors. Darwin-only `ctypes` code must type-check on Linux.
- WebUI: `cd webui && bun run test && bun run build`.

## 9. Non-goals

- Windows (outside D1 scope).
- Changing any Linux guarantee or default.
- The Claude SDK harness sandbox on macOS (`kohaku/harness/claude_sdk.py` keeps its Linux requirement; possible follow-up).
- A legacy configured `sandbox: "seatbelt"` backend for unmanaged mode (`_BACKENDS`).
- Endpoint Security, which needs an Apple entitlement.

## 10. Residual risks

- `sandbox_check` is a private API. A macOS update could change it. The self-test makes this fail closed, but users would lose macOS commands until it is fixed.
- `sandbox-exec` is deprecated, with the same mitigation.
- The completeness of a `proc_listallpids` snapshot is observed, not documented. Two consecutive empty scans reduce, but don't remove, that assumption.
- Pid reuse between the final check and `kill` is narrowed by K2, not eliminated. Linux has a similar small window.
- Children started through launchd/XPC are outside the tracked tree. This is the same class of gap as Linux D-Bus activation.
- Nested-Seatbelt tools (G2) are unavailable in both modes on macOS.


## 11. Implementation log

### 2026-11-26 — WP1 and the gate-independent part of WP0 (committed on the work branch)

**WP1 — capability seam: done.**
- New `kohaku/agent/tools/executor_capabilities.py`: frozen `ExecutorCapabilities(platform, confinement, tree_ownership)`, `executor_capabilities()`, `managed_confinement_available()`, `tree_ownership_available()`. On Darwin and any other platform both guarantees are `None`, so macOS fails closed exactly as today (AC1.2).
- `native_process.native_supervisor_available()` now delegates to `tree_ownership_available()`. Its predicate is unchanged (`linux` and `posix`).
- Parity tests: `tests/agent/test_executor_capability_parity.py` (26 cases) freezes the Full, Once and Sandbox projections, the native-command refusal table and the ownership identity. They were written and passed against the **pre-change** code first. A deliberate break of the Linux branch fails 5 cases (mutation check). `tests/agent/test_executor_capabilities.py` covers the seam itself.

**Deviations from `7, decided while implementing:**
1. **Call sites were not edited.** Existing suites patch `sandbox.managed_bwrap_available`, `shell.managed_bwrap_available`, `native_process.native_supervisor_available`, `shell.native_supervisor_available` and `claude_sdk.managed_bwrap_available`. Rerouting call sites to new functions would bypass those patches. The old names stay the entry points; the platform decision lives behind them. Call-site changes move to the packages where Darwin first needs them (WP5/WP6: variable backend name, `seatbelt`).
2. `managed_bwrap_available` stays the bubblewrap probe. `confinement()` calls it at call time and only on Linux.
3. `executor_capabilities()` is **not** cached. The expensive probe is already `lru_cache`d, and a second cache would pin stale values in tests that patch the platform.
4. The `darwin_host` marker is registered in the root `conftest.py` hook, not `pyproject.toml`. One hook both registers and gates it for every test directory.
5. **AC0.2 (zero-member smoke scan) moves to WP2.** It needs the `sandbox_check` binding, which WP2 creates. It is read-only and sends no signals.
6. The feature switch (D-M5) lands with WP2's self-test, the first thing that can turn a Darwin capability on. Until then Darwin is `None` unconditionally.

**WP0 — safety harness: done except the CI job.**
- `tests/darwin/gate.py` plus the root `conftest.py` hook skip `darwin_host` tests unless the platform is macOS **and** `KOHAKU_DARWIN_DESTRUCTIVE_TESTS=1` exactly. `tests/darwin/test_gate.py` covers the table and has one marked sentinel.
- **The macOS CI job is landed under G6(a).** `tests/migration/test_p6_release_rehearsal.py` now allows exactly one job named `macos` on `macos-latest`; every other job must stay on `ubuntu-latest`. The job (`KOHAKU_DARWIN_DESTRUCTIVE_TESTS=1`, `uv sync --extra dev`, then `pytest -q tests/darwin tests/agent/test_executor_capabilities.py tests/agent/test_executor_capability_parity.py`) has not yet run on a hosted runner.

**Evidence (macOS 27.2.1 arm64, repo `.venv` Python 3.13.2):**
- Strict types, full-source `basedpyright 1.40.1` before (HEAD export) and after, same interpreter, `scripts/check_type_regressions.py`: **`PASS_NO_NEW_DIAGNOSTICS`, 962 → 962, introduced 0, resolved 0**. Source hashes differ only in `executor_capabilities.py` (new) and `native_process.py`. `strict_clean` is still false because of the existing backlog.
- Ruff on the touched files passes.
- Full suite, `-n 4 --dist loadfile --ignore=tests/cli/test_commands.py --basetemp=/private/tmp/kt/…`: **15 failed, 15,684 passed, 83 skipped**. The same 15 tests fail on an untouched HEAD export, so none are new. They are outside this change (`test_claude_child_gateway`, `test_migration_cron_admission`, `test_tool_descriptions` and others). The default pytest temp path on this host additionally causes 77 `AF_UNIX path too long` errors in the remote and WebUI suites; that is environmental and also reproduces on HEAD.
- Not run: `tests/cli/test_commands.py` (CI runs it serially). So the stage-completion rule in `8 is **not** met yet. The full strict checker still exits non-zero from the existing backlog, which is expected to stay until P6 closes.

**Next.** WP2 needs G1 (private `sandbox_check` and deprecated `sandbox-exec`) and G5 (Apple Silicon only). WP5 needs G3 and G4, and WP4 and WP5 need G2.

### 2026-11-26 — WP2 Darwin platform primitives (committed on the work branch)

**Done.**
- New `kohaku/security/darwin_proc.py`: process listing and identity, the arm64 `sandbox_check` binding, the two-marker membership predicate and read-only scan, host and boot identity, `renamex_np(RENAME_EXCL)`, and `seatbelt_self_test()`.
- Additive Darwin branches in `native_admission.py` (`current_host_identity`, `process_start_time`, `process_pid_namespace`, `_rename_noreplace`). The Linux code paths are unchanged.
- `new_launch_id()` produces 32 lowercase hex characters, the format the ledger's `_LAUNCH_ID_RE` already requires. Neither side needed to change.
- AC0.2, moved here from WP0, is covered by the zero-false-positive whole-table scans in `tests/security/test_darwin_proc.py`. That file is now part of the `macos` CI job.

**Deviations.**
1. Nothing consumes the self-test yet, so `tree_ownership` is still `None` on Darwin. The feature switch (D-M5) and the self-test gating land in WP4, with the first supervisor they can turn on.
2. The G2 marker `KOHAKU_SANDBOX=seatbelt` is decided but not yet exported; WP4 exports it.

**Evidence (same host and interpreter as above).**
- Strict types, full-source `basedpyright 1.40.1` against the frozen HEAD export: default platform mode **962 → 962, introduced 0**. Same-platform pairs also introduce 0: `--pythonplatform Darwin` 1152 → 1152 and `--pythonplatform Linux` 1172 → 1172. A first run found 5 new diagnostics from a `ctypes._CFuncPtr` annotation; the return type is now `Callable[..., int]`.
- Focused suites (Darwin primitives, native admission, managed exec, native process I/O, exec reservation, Permission V2, capability seam, gate): 126 passed, 24 skipped.
- WebSocket channel and boundary, release rehearsal and CI path tests: 462 passed, 1 failed (`test_persisted_webui_session_prefix_has_one_production_owner`, one of the known HEAD failures).
- Full suite, `-n 4 --dist loadfile --ignore=tests/cli/test_commands.py` with a short `--basetemp`: **15,707 passed, 15 failed, 83 skipped**. The failing ids are exactly the 15 known on HEAD.
- `tests/cli/test_commands.py`, run serially: 129 passed. This closes the gap noted in the previous entry.
- The §8 full-suite gate still carries the 15 pre-existing failures, so it is not green.
- Not run: the `macos` job on a hosted runner, any Intel host, and any real-signal (`darwin_host`) test on this machine.

**Next.** WP3, the launch-domain tracking engine, tested against fake process tables and signallers. Its real-signal tests run only on a disposable host.

### 2026-11-26 — WP3 launch-domain tracking engine (committed on the work branch)

**Done.**
- New `kohaku/agent/tools/seatbelt_tracking.py`: `LaunchDomain` with `scan`, `poll`, `kill_pass`, `wait_quiescent` and `kill_until_quiescent`; injectable `ProcTable` and `Signaller`; `host_launch_domain()` for the supervisor. Nothing calls it yet.
- `darwin_proc` gained `membership()` (member, not a member, or unknown) and `parent_pid()`.

**Additions beyond the packet** (see §4 E9-E11 and §6): fixed-point scans; root-exit gate; unknown membership blocks the proof; K5; former ancestors stay protected after reparenting; `poll()` backs off to 0.5 s while the tree is alive, so it can be called from the supervisor loop.

**Acceptance.**
- AC3.1 (fake table): passes.
- AC3.3 (waiting without signalling): passes with fakes, and on this host with a real `setsid` double-fork descendant that exits by itself.
- AC3.2 (`darwin_host`, real kills): **written, not run.** `tests/darwin/test_launch_domain_host.py` covers escaped descendants, a `SIGTERM`-ignoring member and a bounded fork storm. Each process also has its own alarm, and a second guard refuses to signal anything older than the test.

**Evidence.**
- Mutation check in a throwaway copy: removing each of the K1 recheck, uid, protected-pid and remembered-ancestor rules, K2, unknown blocking, K5, the fixed-point scan, the root-exit gate or the second empty scan makes at least one test fail. The unmutated control passes.
- Strict types: 0 introduced in default (962 → 962), Darwin (1152 → 1152) and Linux (1172 → 1172) modes.
- Full suite (same command as WP2): **15,745 passed, 15 failed, 85 skipped**; the failures are exactly the 15 known on HEAD. `tests/cli/test_commands.py`: 129 passed.
- Ruff over `kohaku/` and `tests/`: 10 findings, identical to HEAD.

**Next.** Run the `macos` job (or `tests/darwin` with the opt-in variable on a disposable host) to close AC3.2, then WP4.
