# Authority, confinement and platform boundaries — REF4

Preserve server-owned scope, private nonserialized authority proofs, durable grant
publication, launch/stdin rechecks and refusal when required confinement is unavailable.
The next detailed audit should build a reachable capability/caller matrix before changing
platform support or retry behavior. Full's intended guarantees require explicit discussion.
No exploit, operational confinement result or cause of the reported sandbox issue is established.
Host/client OS alone does not establish executor OS or incident cause.

All rows use the [frozen source pins](README.md#source-pins). They cover targeted bodies,
not a full native backend audit. Exact body reads and search-only scope are retained in
[provenance.json](provenance.json); reference mechanisms are engineering examples.

## S1 — Preserve server-owned authority on every attempt

| Field | Comparison |
| --- | --- |
| Required outcome | User-visible labels and untrusted message fields cannot grant file or process authority. Each admitted attempt uses the current server-owned source/policy. |
| Nanobot implementation/configuration | `restrict_to_workspace` defaults false (`schema.py:895`); exec defaults include enabled, 60-second timeout and empty legacy sandbox (`shell.py:127`). [Foreground construction](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/foreground.py#L34) builds grants/policy/locks; `delegation.py:176` passes managed WebUI policy. [Inbound commands](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/webui/inbound_commands.py#L98) reject authority fields; scope comes from server workspace/session state (`workspaces.py:1557`, inbound_commands.py:1023). Loop rechecks source/store/session/incarnation/generation/request (3335). |
| Reference mechanism | DSH [policy resolution](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox-policy/src/index.ts#L134) prioritizes explicit override, session projection and deployment defaults (read-only default), used by `tool-bash/src/index.ts:213,478`. Codex [ToolOrchestrator](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/orchestrator.rs#L134) builds an attempt from turn environment (134,327), then `tools/sandboxing.rs:403,495` delegates. |
| Writer/identity/lifetime/authority/error boundary | Nanobot's private proof and active source validation differ from standing/turn snapshots. A missing or stale proof refuses authority; saved mode labels do not recreate it. |
| Actual difference/disposition | **Preserve.** Defaults belong to distinct legacy/managed paths; a global flag is not the active capability contract. |
| Conditions/impact/confidence/next step | High source confidence for named callers. Later S01/S02 audit must trace each reachable tool/entrypoint combination rather than infer policy from config names. |

## S2 — Preserve Once/Full identity and lifetime distinctions

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Authority must bind the intended source/task/input and expire predictably. Whether Full may be restored durably is a separate pending product/security choice. |
| Nanobot implementation/configuration | [DesktopPermissionOwner](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/security/desktop_permissions.py#L123) uses frozen identity/unserializable references and lock/volatile epoch (229). Once persists before publication (259), then checks exact object/source/task/input (388). Full requires an existing local idle/healthy session, no legacy sandbox, supervisor availability and no residual work (458); request proofs are used at 545/620. Restart clears volatile authority (1279); loop retires Once at 3746. Recovery binds source/runtime/revision/canonical directory (`workspace_access_recovery.py:1005`). |
| Reference mechanism | DSH [session mode](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox-policy/src/session-mode.ts#L53) is durable standing mode, while `sandbox/src/escalation.ts:172` permits one-call escalation. Codex [approval/retry](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/orchestrator.rs#L167) separates initial approval from retry (484); Guardian approval does not automatically widen authority. |
| Writer/identity/lifetime/authority/error boundary | Nanobot Once is task-wide, DSH allowed-once is call-wide. Nanobot Full is volatile; a durable reference mode is not equivalent authority. |
| Actual difference/disposition | **Preserve; terminology clarification.** These are justified product differences until a new requirement is accepted. No automatic restore from saved labels is recommended. |
| Conditions/impact/confidence/next step | High source confidence. Clarify effective lifetime in docs/status; [D1/D2](decisions.md#d1-supported-executor-and-client-platforms) govern any broader platform or Full change. |

## S3 — Separate platform capability from desired platform support

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Required confinement must refuse when unavailable. Which executor platforms must provide managed shell, native supervision and local owner authentication? The accepted platform contract is pending. |
| Nanobot implementation/configuration | [managed_bwrap_available](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/sandbox.py#L143) requires Linux, Bubblewrap and a successful bounded cached probe; wrapping starts at 193 and shell refuses unavailable managed confinement (shell.py:758). [Native supervision](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/native_process.py#L227) requires Linux POSIX; Once/Full launch requires supervisor/admission (shell.py:882). Effective access reports `unsupported_platform`/`sandbox_unavailable` (`workspace_focus.py:802`); Full prerequisites are checked in workspaces.py:791,1319. Legacy shell.py:799 is a distinct unsupported-host behavior, not a universally reachable managed fallback. |
| Reference mechanism | DSH [backend selection](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox-local/src/index.ts#L160) supports Linux Bubblewrap→Landlock, macOS Seatbelt and partial Windows ACL (506). Multiple candidates are functionally probed; a sole candidate may be selected without a probe and refuse during execution. `profiles.ts:16` defines profiles; the Windows ACL runner rejects nonempty `additionalWritableRoots` only in `workspace-write` mode (index.ts:369–371). Codex [SandboxManager](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/sandboxing/src/manager.rs#L50) selects concrete backends (313,437); `linux_run_main.rs:263,380` uses Bubblewrap/seccomp without automatic Landlock fallback. `ensure_native_sandbox` (`tools/sandboxing.rs:236`) applies specifically to `EscalatedSandboxWithRestrictions`, not universal native ownership. |
| Writer/identity/lifetime/authority/error boundary | Executor-host backend availability differs from browser/phone/client OS. Confinement, whole-process supervision and authenticated owner IPC are three contracts. |
| Actual difference/disposition | **Product discussion plus documentation precision C1.** Missing macOS/Windows managed paths are not defects without a supported-product requirement. `.agent/security.md` describes optional Bubblewrap/native fallback incompletely for current managed/supervised paths. |
| Conditions/impact/confidence/next step | High confidence in selection/refusal source, no OS enforcement certification. Preserve refusal and publish an accurate capability matrix first; decide supported platforms under [D1](decisions.md#d1-supported-executor-and-client-platforms) before adding backends. |

## S4 — Preserve native launch/stdin ownership separately from confinement

| Field | Comparison |
| --- | --- |
| Required outcome | Approved source/task authority must still be valid at actual native launch and interactive input; descendants need a process owner independently of filesystem wrapping. |
| Nanobot implementation/configuration | [FolderGrantStore reservation](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/security/folder_grants.py#L272) binds the exact reference; native_admission.py:146 rechecks the launch gate. [Native process launch](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/native_process.py#L231) requires readiness/stdin/durable supervisor identity/admission before GO. Reserved shell launches use this supervisor and refuse unavailable ownership (shell.py:943); exec_session.py:219,474 validates stdin proof/owner lock/session owner. |
| Reference mechanism | DSH/Codex confinement wrappers do not establish equivalent process-tree receipt semantics. Codex [remote execution handoff](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/sandboxing.rs#L495) passes native command/context rather than a local controller wrapper. |
| Writer/identity/lifetime/authority/error boundary | Reservation, source proof, supervisor receipt, process session and stdin have separate lifetimes. A successful wrapper selection is not a supervisor ownership receipt. |
| Actual difference/disposition | **Preserve. Rejected hypothesis:** a supported prepare→reserve→spawn retirement race was not established. The foreground path is synchronous without an await (`shell.py:527–567`); RuntimeComposition uses one event loop/thread (426–454), and authority mutation handlers are synchronous (`ws_http.py:1967,2387–2397,2461–2475`). The `to_thread` call at 2525 handles remote authentication, not owner mutation. |
| Conditions/impact/confidence/next step | High confidence in the reviewed contract; no supported cross-thread retirement path found. Re-evaluate only if a future await/thread mutation changes it. Do not publish this rejected race as a vulnerability or major improvement candidate. |

## S5 — Preserve grant, filesystem and shell correspondence

| Field | Comparison |
| --- | --- |
| Required outcome | Canonical paths, private-state exclusions and most-specific grants must mean the same thing at tool checks and shell construction. |
| Nanobot implementation/configuration | [ResourceAccessPolicy](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/context_repo/policy.py#L177) checks resolved/private paths; normal managed reads/list/exec follow OS access, while writes use most-specific roots/task tie (278,322). Folder grants canonicalize existing directories, exclude private paths, bind revision fingerprints and publish only after persistence (folder_grants.py:553,576). [Managed mounts/masks](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/sandbox.py#L205) use depth priority (238); filesystem.py:240,340 validates resolved destination/credentials. Workspace focus's relative-path selection (455) is not authority. |
| Reference mechanism | DSH [sandbox roots](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox/src/roots.ts#L31) handles physical roots/symlinks (57), `fs-sandbox/src/index.ts:156` re-resolves mutation targets, and Windows local index.ts:406 distinguishes standing ACEs from revocable temporary state. Codex [Seatbelt protected ancestors](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/sandboxing/src/seatbelt.rs#L878) constructs a separate native boundary. |
| Writer/identity/lifetime/authority/error boundary | Grant-store revision/publication, per-task authority, path resolution and mounted process view are distinct. Native differences prevent a blanket TOCTOU guarantee from source comparison. |
| Actual difference/disposition | **Preserve; detailed matrix queued.** No cross-platform policy-equivalence or complete symlink-race certification is claimed. |
| Conditions/impact/confidence/next step | High confidence in named policy paths. S01/S02 should enumerate reachable tool/wrapper combinations and stale/revoked grant readers before proposing simplification. |

## S6 — Decide Full's guarantee and align instructions with it

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Is Full ordinary OS-account authority, or must it retain private-path/network isolation for arbitrary subprocess effects? The latter is a consequential security requirement and remains pending. |
| Nanobot implementation/configuration | Path policy checks private/context restrictions before Full bypass (`policy.py:177,247`). [Full shell](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/shell.py#L702) sets managed false and omits managed masks. Text-path/URL checks (`shell.py:1258`, policy.py:489) cannot enforce arbitrary computed subprocess effects. Yet [context instructions](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/context.py#L159) (159,296), including in Full, say private authority/context policies remain protected. `WorkspaceControls.tsx:904` explicitly describes normal OS-account access. |
| Reference mechanism | DSH [sandbox contract](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox/src/index.ts#L24) scopes file effects and excludes network/process visibility; `bash-sandbox/src/index.ts:89` directly bypasses for Full. Codex [denied-read handling](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/sandboxing.rs#L251) can survive escalation; Linux network/seccomp is separate. Neither reference settles nanobot's intended Full contract. |
| Writer/identity/lifetime/authority/error boundary | Private source proofs authorize launch, but do not themselves confine computed filesystem/network effects once a native shell has normal account access. Text URL checks are not kernel network isolation. |
| Actual difference/disposition | **Instruction precision candidate C4 and product discussion D2.** Potential overstatement is source-established; no exploit is demonstrated. Do not claim a Full confinement guarantee from path-tool guards. |
| Conditions/impact/confidence/next step | Trigger is computed filesystem/network access under Full. High confidence in the boundary, intended guarantee unresolved. Smallest route is accurately describing normal OS semantics if accepted; retaining private-path isolation needs explicit security acceptance and a separate implementation contract. |

## S7 — Preserve raw outcomes and distinguish failed attempts

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Distinguish unavailable backend, runner refusal, file denial, ordinary child failure, cancellation and timeout without widening authority. Agree the internal attempt-outcome contract before code changes. |
| Nanobot implementation/configuration | [Shell preparation/admission](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/shell.py#L527) precedes launch. Nonzero managed output containing read-only-filesystem text receives sandbox guidance (594); generic launch/runtime exceptions use 625, timeout formatting 632 and does not carry concrete backend facts. No automatic unsandboxed retry was observed. |
| Reference mechanism | DSH [bash sandbox facts](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/shell/bash-sandbox/src/index.ts#L59) (59,98,158) attach process/enforcement/runner facts; infrastructure failure outranks file denial. `sandbox/src/diagnostics.ts:33,65` and tool-bash index.ts:165,454 preserve cwd/signature/exit/output. Codex [SandboxDenied handling](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/orchestrator.rs#L394) (394,484) retries only under reviewed authority, preserving raw output; unified_exec/launch.rs:17 separates launch failure. |
| Writer/identity/lifetime/authority/error boundary | Attempt authority and observed process outcome belong together. A nonzero child exit is not automatically a `ToolResult.error` or permission denial. |
| Actual difference/disposition | **Bounded improvement candidate C5.** Add the smallest internal attempt distinction preserving raw output and legacy fields, after the contract is accepted; do not infer denial from every nonzero exit and do not auto-widen. |
| Conditions/impact/confidence/next step | High confidence in current formatting/heuristics, unknown field frequency. Pair S02 outcome review with A7 recovery classification, under one shared invariant owner rather than two competing taxonomies. |

## S8 — Separate declared environment status from local availability

| Field | Comparison |
| --- | --- |
| Required outcome | Status must distinguish declared external enforcement, policy intent and concrete local backend availability without erasing compatibility fields. |
| Nanobot implementation/configuration | [workspace_access status](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/security/workspace_access.py#L193) can mark `enforced=True/system` based on an environment marker (474); actual effective access is alongside it (`workspace_focus.py:320`, workspaces.py:481). Managed availability is computed in workspace_focus.py:802. [Main UI](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/webui/src/components/thread/WorkspaceControls.tsx#L222) prefers effective access with legacy fallback; its menu name (689) is not itself an availability assertion. |
| Reference mechanism | DSH reports actual wrap/process facts (S7); Codex carries the selected attempt backend (S3/S7). Neither makes an external environment declaration self-verifying. |
| Writer/identity/lifetime/authority/error boundary | A deployment marker may truthfully describe an external envelope, but it was not verified. Local executable availability is a different writer/provenance. |
| Actual difference/disposition | **Bounded status improvement C1.** No confirmed false claim in the primary UI. Preserve legacy fields while clarifying status provenance; do not flip a global default. |
| Conditions/impact/confidence/next step | High confidence in the source distinction; deployment truth unknown. Include API/UI fallback readers in the first S02 matrix before a separate documentation/status repair. |

## S9 — Preserve fail-closed remote owner authentication

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Only an authenticated local owner may mutate source/task/revision authority. Desired non-Linux owner IPC support is pending with the platform contract. |
| Nanobot implementation/configuration | [IPC peer credentials](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/remote/ipc.py#L139) are checked before parsing/dispatch, require same UID and reject missing `SO_PEERCRED`. owner.py:32 starts the server; client.py:315 bounds the error path. permissions.py:85 validates operation/payload/install binding; the controller owns actual source/task/revision mutation. ws_http.py:2419 authenticates source; its `to_thread` at 2525 is remote auth only. |
| Reference applicability | DSH/Codex platform confinement wrappers are not equivalent local owner-authentication implementations and supply no justification for weakening this check. |
| Writer/identity/lifetime/authority/error boundary | Local peer credentials, install identity, session source identity and gateway execution are separate authorities. Missing macOS support can cause unavailability, not demonstrated privilege escalation. |
| Actual difference/disposition | **Preserve plus D1 platform discussion.** Add an equivalent mechanism only after the desired platform requirement is settled; no weaker fallback. |
| Conditions/impact/confidence/next step | High source confidence; actual deployment/OS behavior unobserved. R01 and S02 must share the owner-auth contract while keeping execution confinement separate. |

The first later audit is [AUDIT-S02-01](audit-backlog.md#audit-s02-01--first-later-audit).
Its source-only result can establish code-path defects and limits; it cannot certify OS
confinement, native process ownership in operation or parity with either reference.
