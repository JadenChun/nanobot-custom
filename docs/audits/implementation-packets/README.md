# A3 implementation handoff

**Planning status.** All eleven stage packets are available for review.
The local A3-0 qualification report is pending. Later packets use that report and the accepted results of their dependencies before dispatch.
[Readiness review and dispatch order](opencode-readiness.md) · [OpenCode handoff prompts](opencode-handoff.md)

**Source basis.** `origin/staging` at `29612c3df01dc3dce3501fe349bcf555cd6c58b6`.
The integrated source audit uses `9562e57f8333d7f14b6aaeae837660723a9b2650`.
The only source and test delta is the bundled skill-path fix. A3-6 owns the remaining provenance issue.
[Integrated findings](../module-review-2026-11-06/integrated/I05-final-reconciliation.md) · [A3 roadmap](https://chatgpt.com/c/6aef25bd-7db8-83ec-bc98-f7fbf21da9d0)

## Stage order

| Packet | Needs accepted implementation evidence from | Handoff result |
| --- | --- | --- |
| [A3-0 — Baseline](A3-0-qualification-baseline.md) | Current origin | Frozen source and qualification evidence. |
| [A3-1 — Authority foundations](A3-1-authority-foundations.md) | A3-0 | Instance and target facts, provenance, outcomes, attempts, and early validation. |
| [A3-2 — Runtime and processes](A3-2-runtime-process-ownership.md) | A3-1 | Owned exec, confinement matrix, and composition close contract. |
| [A3-3 — Durable sessions](A3-3-durable-session-runtime.md) | A3-1 | Stable session identity, one history writer, and restart disposition. |
| [A3-4 — Client and control contracts](A3-4-client-control-contracts.md) | A3-2 and A3-3 | Typed approval, admission, Stop, replay, and client reconciliation. |
| [A3-5 — Delegation cutover](A3-5-delegation-cutover.md) | A3-4 | Durable child sessions, narrowed authority, and client navigation. |
| [A3-6 — Capability readiness](A3-6-capability-readiness.md) | A3-5 | Ready tool generations, provider and skill provenance, and bounded workbench projection. |
| [A3-7 — Mobile pairing](A3-7-mobile-pairing.md) | A3-5 | Device-key access, production scope checks, and native custody. |
| [A3-8 — Durability and retention](A3-8-durability-retention.md) | A3-6 and A3-7 | Retained proof, bounded reads, safe cleanup, and durable scheduling. |
| [A3-9 — Release qualification](A3-9-release-qualification.md) | A3-8 | Exact artifacts, verified upgrade, and tested deployment rollback. |
| [A3-10 — Compatibility retirement](A3-10-compatibility-retirement.md) | A3-9 | Evidence-based removal and a requalified final candidate. |

Work through the packets in order. The table shows which outputs are prerequisites.
Use one write owner when packets touch the same files.
The audit worktree supplies the documents. Use a separate implementation worktree based on the accepted source.

## Dispatch and acceptance

1. Verify current source and the accepted baseline. Recheck any owner that changed after the planning source or the preceding stage.
2. Give the coding agent one slice, its write boundary, and the relevant accepted evidence.
3. Resolve the packet's named choices before an affected change starts.
4. Run the closest public-boundary checks for each acceptance criterion.
5. If a new or changed case is required, use the requirements-only author procedure in `AGENTS.md`.
6. Preserve the repository's strict-type repair gate. Compare full-source reports under the same dependency environment.
7. Stop when the coordinator verifies the criterion results, compatibility, and rollback evidence for that stage.

Retain exact commands, exit codes, source and dependency identities, and focused evidence.
Report any baseline or platform gap by name. Reuse earlier passing checks while their inputs still match.
For strict-type comparisons, require zero new diagnostic signatures across all files.
Use the required full checks before closing P6 or qualifying a release.
[Strict-type evidence rules](../../migration/p6-error-attribution.md)

Run Python targets with `uv run --no-sync python -m pytest` from the implementation checkout.
Run the listed WebUI targets with `bun run test` from `webui/`.
Run the listed TUI targets with `bun test` from `tui/`.
Run the native broker tests with `go test ./internal/broker` from `mobile/native-network/`.
Keep platform, emulator, and physical-device evidence separate.

The stage report records the result, source revision, changed paths, acceptance evidence, commands, and remaining finding IDs.
Record which product or platform claims the evidence qualifies.

## Choices to record before the affected work

| Packet | Required decision or evidence |
| --- | --- |
| A3-2 and A3-3 | Record which executor modes and writer leases each supported OS can prove. A request for unavailable confinement or authority fails closed. |
| A3-5 | Record profile policies and finite tree limits. Keep future Codex and Claude backends outside this cutover. |
| A3-6 | Record provider credential-profile ownership and compatibility. Keep inactive features gated unless their activation has an accepted contract. |
| A3-7 | Review the device proof format, verifier, persisted-state upgrade, and rollback before enabling v2 issuance. |
| A3-8 | Approve any new retention or archive policy. Record whether audio duration is an enforced server limit. |
| A3-9 and A3-10 | Verify the supported target matrix and rollback. Approve each removal that affects a public or persisted contract. |

## Audit repair ownership

This map places the 22 I05 repair packages. Use the source findings for exact evidence and qualifications.
A packet may address only a named part of a repair package. Record each unresolved part in its stage report.

| Repair | Planned stage and scope |
| --- | --- |
| R01 | A3-1 instance and file facts, A3-2 native scope and CLI execution, A3-4 folder grants, A3-5 child policy, A3-6 prompt, knowledge, and instance stores, A3-7 owner state. |
| R02 | A3-4 exact approval, A3-5 delegated read-only authority, A3-8 automation activation gate. |
| R03 | A3-7 listener, scope, and local mutation admission, A3-9 optional Windows crawler caller proof. |
| R04 | A3-6 provider and catalog provenance, A3-7 mobile media log custody, A3-9 release diagnostics. |
| R05 | A3-1 historical prompt provenance, A3-8 Dream custody and restore. |
| R06 | A3-1 source and terminal outcomes, A3-3 SDK ingest, A3-4 mutation and replay results. |
| R07 | A3-8 source, journal, transcript, and scheduler capacity and retained proof. |
| R08 | A3-1 prepared fallback cursor, A3-6 stream terminal, attempts, and OAuth generations. |
| R09 | A3-2 process and composition lifetime, A3-6 VPN lease custody, A3-9 systemd outcome. |
| R10 | A3-8 process continuation, scheduler, outbox, and trigger settlement. |
| R11 | A3-7 Android key custody, pairing, and native completion. |
| R12 | A3-8 attachment and session cleanup, transcript identity, and Temporary Chat retention. |
| R13 | A3-6 web fetch and thread result bounds, A3-7 mobile cache, A3-8 parsers and asset index. |
| R14 | A3-4 queues and client state, A3-6 workbench and paging fences. |
| R15 | A3-1 core guidance, A3-6 skill and tool readiness, A3-10 dormant exposure review. |
| R16 | A3-4 input and read contracts, A3-7 remote routes, A3-9 readiness. |
| R17 | A3-6 schema, connection lifetime, and route generation. |
| R18 | A3-10 supported compatibility and retirement. |
| R19 | A3-8 cron administration scope, A3-9 production qualification. |
| R20 | A3-9 executable, bundle, toolchain, and signing identity. |
| R21 | A3-9 deployment, historical verifier, CI, and upgrade qualification. |
| R22 | A3-1 alias validation and gap progress. |

The E01 limited-depth channel exception remains part of the audit scope. Qualify any channel control that a packet changes through its real entry point.
