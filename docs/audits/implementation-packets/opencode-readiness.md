# A3 readiness review for OpenCode

**Reviewed source.** `origin/staging` at `29612c3df01dc3dce3501fe349bcf555cd6c58b6`.
The local staging checkout was clean and matched origin.
The bundled skill-path fix and its test remain the only delta from the frozen audit source.
[Stage index](README.md)

**Start with A3-0.** It is an evidence task and can start after the OpenCode shell preflight.
The audit worktree contains the baseline packet. The accepted baseline report is still pending.
A3-1 has bounded implementation slices. A3-2 through A3-10 also need the inputs below before their affected slices can start.

## Readiness by stage

| Stage | First usable handoff | Evidence or decision still needed |
| --- | --- | --- |
| [A3-0](A3-0-qualification-baseline.md) | Run the qualification commands in an isolated checkout. | Confirm the OpenCode model route and toolchain. Capture the local reports. The CI run is prior evidence. |
| [A3-1](A3-1-authority-foundations.md) | Start with §1a and `AC1a`. Use the SDK construction and image reload paths. | Accept A3-0. Add independent cases if the current tests cannot prove bound-root generation and reload together. |
| [A3-2](A3-2-runtime-process-ownership.md) | Start with the native reservation path in §1 after A3-1. | Set the executor-mode matrix before a new backend. `native_supervisor_available` currently selects Linux only. Windows native launch is separate evidence from owned-tree settlement or managed confinement. |
| [A3-3](A3-3-durable-session-runtime.md) | Use the current session manager and canonical writer. | Approve the persisted metadata and older-reader contract. The current mode lease requires `flock`. Record which supported hosts can prove equivalent writer ownership. |
| [A3-4](A3-4-client-control-contracts.md) | Start from the exact approval effect and the accepted process and session contracts. | Approve the actor and wire contract for approval, Stop, and durable admission. The current WebUI event follows route dispatch. It does not prove a session commit. |
| [A3-5](A3-5-delegation-cutover.md) | Start with child admission under the A3-3 session owner. | Accept A3-4. Freeze parent and attempt identities, finite depth and live-child limits, role policy, restart rules, and client event shape. |
| [A3-6](A3-6-capability-readiness.md) | Start with one registry or provider owner per slice. | Accept A3-5. Confirm admitted binding lifetime and revocation before MCP changes. Choose credential-profile ownership and numeric resource bounds before changing them. |
| [A3-7](A3-7-mobile-pairing.md) | Start with the accepted v2 proof and device-state contract. | Accept A3-5. Approve the proposed signature contract, Python and Android verification vectors, v1 revocation, and rollback. Arrange physical Android qualification. |
| [A3-8](A3-8-durability-retention.md) | Start with the source admission size guard. Keep separate work for each store. | Accept A3-6 and A3-7. Select each index or rollover format, retention rule, and reader rollback before writing that format. Decide whether audio duration is a server limit. |
| [A3-9](A3-9-release-qualification.md) | Start with installer or build provenance on an isolated target. | Accept A3-8. Supply the exact artifacts, platform targets, distinct APKs, signing route, and rollback fixture. Current Python CI covers Linux and Windows. macOS runtime evidence must come from a separate run. |
| [A3-10](A3-10-compatibility-retirement.md) | Inventory and approve one retirement at a time. | Accept A3-9. Check public callers, retained records, and restore readers before each removal. Requalify the resulting artifact. |

### Source checks that changed the handoff

- The router emits `message_accepted` in `nanobot/webui/inbound_commands.py` after calling `webui_dispatch_message`. That path can return without publication. A successful publication reaches the in-memory bus first. A3-4 now requires owner proof before dequeue. Use the session owner for reconnect evidence and leave unresolved attempts fenced. [WebSocket dispatch](https://github.com/JadenChun/nanobot-custom/blob/29612c3df01dc3dce3501fe349bcf555cd6c58b6/nanobot/channels/websocket/runtime.py#L465-L500) · [Bus queue](https://github.com/JadenChun/nanobot-custom/blob/29612c3df01dc3dce3501fe349bcf555cd6c58b6/nanobot/bus/queue.py#L37-L43)
- `SessionClient.ingest` updates the live session while it validates messages. The ingest slice needs a valid-prefix-plus-invalid-message case. A3-3 separately gates persisted runtime metadata on the current lease and reader contract. [SDK ingest](https://github.com/JadenChun/nanobot-custom/blob/29612c3df01dc3dce3501fe349bcf555cd6c58b6/nanobot/sdk/clients.py#L44-L77)
- Current mobile access rotates bearer credentials. P-256 is a proposed change, so the A3-7 worker must receive the approved signed bytes and test vectors. The current Android update script also reinstalls the same APK. A3-9 requires a distinct baseline and candidate. [Current access owner](https://github.com/JadenChun/nanobot-custom/blob/29612c3df01dc3dce3501fe349bcf555cd6c58b6/nanobot/remote/auth.py#L118) · [Update script](https://github.com/JadenChun/nanobot-custom/blob/29612c3df01dc3dce3501fe349bcf555cd6c58b6/mobile/scripts/test-update-persistence.sh)

## Dispatch one slice at a time

The entries below divide each stage by owner. Select one entry and give the worker only its packet section, criteria, and accepted inputs.
For a compound criterion, identify the exact subcase in the handoff. Keep the stage criterion open until all its subcases pass.
Freeze the exact files and commands for that slice after its prerequisite code is available.

| Stage | Ordered work for separate OpenCode sessions |
| --- | --- |
| A3-0 | **Baseline** — capture the reports from §1–4 and return their paths. Preserve the clean source and the selected environment. |
| A3-1 | **Roots** `AC1a` → **file targets** `AC1b` → **archive provenance** `AC2` → **required writes** `AC3a` → **late continuation** `AC3b` → **fallback attempt** `AC4` → **guidance** `AC5` → **aliases** `AC6a` → **gap validation** `AC6b`. |
| A3-2 | **Reservation and epoch accounting** `AC2.1` → **yielded deadline** `AC2.2` → **bounded output and tree settlement** `AC2.3` → **one host backend at a time** `AC2.4` → **composition close** and **one adapter at a time** `AC2.5`. |
| A3-3 | **Metadata and reader compatibility** `AC3.1` → **restart reconciliation and lease** `AC3.2` → **detached SDK ingest** `AC3.3`. |
| A3-4 | **Approval owner and clients** `AC4.1` → **owner receipt, then queue and reconnect** `AC4.2` → **Stop settlement** `AC4.3` → **mutation receipts, then one client or read-path fix at a time** `AC4.4`. Keep approval cutover and its client changes together at release. |
| A3-5 | **Durable child admission** `AC5.1` → **narrowed role policy** `AC5.2` → **subtree stop and restart** `AC5.3` → **backend contract and child projection** `AC5.4`. Cut over creation, controls, and UI as one integration result. |
| A3-6 | **MCP readiness and schema** `AC6.1` → **Responses terminal**, **attempt and route facts**, then **credential and instance owners** `AC6.2` → **skill origin**, **prompt and knowledge policy**, then **feature guidance** `AC6.3` → **download and result limits**, **VPN lease**, then **workbench projection** `AC6.4`. |
| A3-7 | **Server proof and revocation**, then **Android custody** `AC7.1` → **owner scoping and route admission** `AC7.2` → **native settlement and limits**, then **shared pairing UI** `AC7.3` → **physical cutover** `AC7.4`. Keep v2 activation and v1 revocation in the same cutover. |
| A3-8 | **Source bound**, **mode proof**, then **transcript rotation and identity** `AC8.1` → **continuation persistence**, **scheduler rollover**, then **outbox settlement** `AC8.2` → **prune and deletion**, **temporary cleanup**, then **Dream custody** `AC8.3` → **index paging**, then **one input budget owner at a time** `AC8.4`. |
| A3-9 | **Installed binary**, **bundle postcondition**, then **toolchain and signing** `AC9.1` → **staged switch and rollback**, then **readiness and redaction** `AC9.2` → **historical verifier and CI**, **parity**, then **distinct APK upgrade** `AC9.3` → **exact release matrix** `AC9.4`. |
| A3-10 | **Inventory and disposition** `AC10.1` → **one approved removal at a time** `AC10.2` → **affected qualification and rollback** `AC10.3`. |

## What to attach for a contract decision

| Affected slice | Acceptance input for the coding agent |
| --- | --- |
| A3-2 backend and A3-3 lease | Approved OS and mode matrix, launch and settlement results, failure behavior, and one host check for each claimed mode. Keep required unsupported modes open. |
| A3-3 metadata | Exact key and version, owning writer, older-reader result, stale-incarnation rule, and migration or rollback fixture. |
| A3-4 control and receipts | Accepted event and status fields, owner commit point, lookup authorization, retained attempt lifetime, and legacy and temporary-session behavior. |
| A3-5 child lifecycle | Accepted ancestry and attempt identity, profile and limit values, child event shape, restart states, and rollback reader. |
| A3-6 readiness and limits | Accepted binding and revocation rules, supported schema policy, credential-profile layout and migration, and limits for the affected resource. |
| A3-7 device proof | Approved signed bytes, encoding and verifier, positive and negative cross-language vectors, challenge expiry and consumption, v1 revoke and rollback rules, and device qualification path. |
| A3-8 retained proof and deletion | Chosen format for each writer, old and new fixture, publication and restore proof, retention policy, and failure behavior for missing references. |
| A3-9 release and A3-10 retirement | Candidate and rollback identity, approved platform evidence, isolated deployment target, and proof for each removed caller or retained data reader. |

The coordinator resolves these inputs before dispatching the affected slice.
The worker returns a scoped diff and evidence. The coordinator accepts the slice and decides when to send the next one.
[OpenCode prompts and first handoff](opencode-handoff.md)
