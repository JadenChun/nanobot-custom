# I05 — Final whole-project finding reconciliation and repair ownership

**Stage:** 2026-11-07. This is the final **source-based integrated reconciliation** of the frozen audit, not a product repair, runtime certification or D1–D7 decision. I05 reconciles all **136 original source findings**, the previously published P01 simplification packages, I01–I04 integration maps, and the separate current-staging **K01-STG-F01** overlay.

**Result:** integrated source review **complete** under the audit's own status definition: cross-module reconciliation, material gap/disposition accounting and checkpoint evidence are now recorded. **No D1–D7 value is accepted**, no implementation is authorized, no original finding/question count is rewritten, and no runtime/build/test/deploy claim is created.

Frozen source remains `9562e57f8333d7f14b6aaeae837660723a9b2650` / tree `bbe2f237efe0cd7b935946b533f66ed0097a3c93`. Current staging remains separately observed at `29612c3df01dc3dce3501fe349bcf555cd6c58b6`, whose only source/test delta was the already-reviewed K01 skill-summary change. E01 remains an explicit **95-primary limited-depth exception** and is not silently upgraded to deep PASS.

## Final deduplication conclusion

### Whole-finding collapse: **none justified**

I05 found **zero pairs of distinct original finding IDs whose entire factual claim, trigger, consequence and owner are equivalent enough to delete one source finding**. The original **136 finding entries remain 136**.

This is intentional, not a failure to simplify. The audit repeatedly found superficially similar symptoms at different failure windows:

- pre-effect mandatory source-write failure versus post-effect response failure;
- stale client completion versus backend mutation replay;
- waiter-not-yet-registered versus admitted completion dropped under backpressure;
- root process exit versus descendant-tree quiescence;
- dependency lock provenance versus post-build artifact proof;
- credential first-custody versus credential rotation;
- tool registration absence versus instructions that advertise unavailable tools.

Collapsing these would make repair and failure-injection tests less precise.

### One proven **partial factual overlap**, retained as two findings

**T01-F10** proves the default core manifest omitted `apply_patch`. **K01-F11** independently proves shipped guidance advertises `apply_patch`, `grep` and `find_files` while none are in the default core registry. The **`apply_patch`-missing registration fact is shared**, so implementation should have **one manifest owner** for that fact. But K01-F11 remains broader: fixing only `apply_patch` does not resolve the `grep/find_files` guidance mismatch. Therefore neither original ID is removed.

The I01–I04 repeated IDs—**S02-F02**, **K01-F03**, **A01-F07**—are not duplicate findings at all; they are the **same original IDs referenced from two integration perspectives**. K01-STG-F01 is a separate staging-only overlay, not a frozen duplicate.

## 22 canonical repair packages — every original ID assigned exactly once

These packages are **implementation coordination units**, not rewritten issue counts. A package can contain multiple distinct findings because one owner can repair/test them together. The machine-readable [I05 evidence ledger](I05-final-reconciliation-evidence.json) proves the 22 lists form an exact partition of all 136 original IDs.

| Package | Canonical owner / source of truth | Findings | Final disposition |
| --- | --- | --- | --- |
| **R01 — Effective resource, workspace and instance authority** | `ResourceAccessPolicy` / selected workspace facts + explicit gateway-instance roots | S01-F01, S01-F03, H01-F4, K01-F01, K01-F02, T01-F07, X01-F01, C00-F06, E02-F11, E02-F13, E02-F15, E02-F16, E02-F17, E02-F18 | One effective target/root fact passed to adapters; keep filesystem policy, OS sandbox and remote auth separate. |
| **R02 — Approval and delegated mutation trust** | risky-action policy + approval admission provenance | H01-F5, T01-F04, T01-F08, K01-F06, A01-F09, X01-F02 | User approval/delegated read-only labels may not be inferred from command words, remote hints or automation text. |
| **R03 — Listener, mobile command and local side-effect authentication** | gateway listener auth + remote WS action/scope map + crawler IPC auth | E02-F02, E02-F22, R01-F02, Q01-F01 | Fail closed at each actual transport boundary; UI/route presence does not grant authority. |
| **R04 — External credential/provenance/privacy boundary** | provider credential owner + log/provenance owner | H03-F4, K01-F08, Q01-F02, R01-F11, X01-F04 | Remove ambient/secrets-in-logs/provenance ambiguity; no single global “trust” flag. |
| **R05 — Instruction authority and auditable self-mutation** | compaction/system-prompt checkpoint owner + Dream transaction owner | K01-F12, K01-F13 | Raw user/tool evidence never gains system authority; Dream must audit/restore every path it mutates. |
| **R06 — Canonical turn/source/effect outcome** | TurnWriter/Runner + mutation executor/replay outcome contract | H01-F1, H01-F2, H01-F6, H01-F7, E02-F01, E02-F20 | Required-write failure, applied effect, uncertain effect and final terminal projection remain distinct. |
| **R07 — Source/history capacity and rotation** | canonical source/mode/scheduler/transcript stores | H02-CAP-SOURCE64, H02-CAP-JOURNAL4096, H02-IO, A01-F04, E02-F07 | Pre-admission bounds/epoch or index proof without silent proof deletion; performance claims require measurement. |
| **R08 — Provider terminal, attempt and OAuth generation** | provider adapter + route attempt + login-generation owners | H03-F1, H03-F2, H03-F3, H03-F5, H03-F7 | Explicit stream terminal, physical attempt identity and generation-gated credential publication. |
| **R09 — Process/composition/child lifetime** | supervised process owner + RuntimeComposition/SDK close owners | S02-F01, S02-F02, S02-F03, C00-F01, C00-F02, C00-F03, E02-F14, R01-F06, T01-F09, T01-F12, X01-F03 | One owned lifetime pattern, but preserve prelaunch/running/unknown and OS-specific process semantics. |
| **R10 — Scheduler/process continuation and trigger durability** | ProcessCompletionRuntime + GatewayCron + local-trigger delivery | A01-F01, A01-F03, A01-F05, A01-F08 | Ordered async persistence may replace event-loop blocking only if durable-before-effect and no-blind-replay proofs survive. |
| **R11 — Mobile credential custody and native completion** | remote auth/pairing + Android/native broker completion owner | R01-F01, R01-F03, R01-F08, R01-F09, R01-F10, R01-F12, R01-F13, R01-F15 | Custody, cancellation, waiter registration, queue capacity and JNI size/concurrency are distinct states under one lifecycle. |
| **R12 — Media/session deletion and destructive maintenance** | attachment store/prune + canonical session/transcript cleanup | M01-F01, M01-F02, M01-F04, E02-F03, E02-F05, E02-F06, E02-F09, E02-F10 | Destructive cleanup must fail closed on incomplete reference proof and report partial cleanup truthfully. |
| **R13 — Resource budgets and finite index visibility** | parser/provider/cache/web-fetch owners + generated-asset index | R01-F14, M01-F03, M01-F05, M01-F06, E02-F23, T01-F11, E02-F04 | Bound bytes/work before allocation/decoding; pagination/truncation is different from memory ceilings. |
| **R14 — Client queue, generation and projection integrity** | React per-surface state + TUI per-session state | U01-F01, U01-F02, U01-F03, U01-F04, U01-F05, U01-F06, D01-F01, D01-F02 | Transactional local dequeue and request-generation fences; no generic frontend state framework. |
| **R15 — Capability manifest, feature activation and skill/runtime parity** | ToolLoader/core manifest + feature runtime readiness + skill renderer | T01-F10, K01-F03, K01-F04, K01-F05, K01-F07, K01-F09, K01-F11, A01-F02, A01-F07 | Advertise only registered/ready features or consciously activate complete implementations. **K01-STG-F01** attaches here as a separate current-origin path-provenance overlay. |
| **R16 — Route/input/local readiness contracts** | WebUI settings router + mobile production edge + local health parser | E02-F08, E02-F19, E02-F21, R01-F04, R01-F07, C00-F07 | Generate/verify route facts from production dispatch; strict input types and readiness shape remain owner-local. |
| **R17 — Tool schema, plugin/MCP readiness and runtime generation** | ToolRegistry/ToolLoader + MCP connection owner + ModelRuntimeResolver | T01-F01, T01-F02, T01-F03, T01-F05, T01-F06, H01-F8, H03-F6 | Distinguish declared/registered/ready/live-generation capabilities; explicit timeout/rollback/revocation. |
| **R18 — Compatibility, gated paths and dormant machinery** | explicit compatibility-retirement ledger | C00-F04, C00-F05, S01-F02, H02-ART-SOURCE, H02-LEGACY-ISOLATE, H02-IDLE-DORMANT | Gated/dead/optional code is not a default incident; remove only after external/storage compatibility proof or activate with full qualification. |
| **R19 — Qualification fidelity and administration scope** | production test composition + product admin-scope contract | R01-F05, A01-F06 | Test the real production composition; define whether cron admin is single-owner or multi-tenant before changing scope. |
| **R20 — Release artifact provenance and signing/toolchain trust** | installer/build hook/mobile release pipeline | Q01-F03, Q01-F04, E02-F12, R01-F16, R01-F17, R01-F18 | Bind installed executable, lock/bundle, toolchain/archive and signing phase to immutable evidence. |
| **R21 — Deployment, CI and real upgrade qualification** | deploy switch/rollback + CI/migration/upgrade gates | Q01-F05, Q01-F06, Q01-F07, R01-F19, R01-F20 | Stage before switching, pin verifier inputs, execute CI for executable helpers, and test distinct baseline→candidate artifacts. |
| **R22 — Early input normalization and progress guards** | constructor/script argument owner | H01-F3, K01-F10 | Validate aliases/numeric progress invariants before normalization/loops; focused local fixes only. |

## Priority reconciliation — preserve original priority, add reachability gates

I05 does **not** convert the streams' heterogeneous P1/P2/high/medium/conditional labels into a fake numeric score. Instead, remediation order uses two axes:

1. **consequence class already established by the source report**, and
2. **whether the affected boundary is actually supported/enabled in the intended deployment**.

### Lane A — supported-boundary authority/security blockers

If the relevant feature/configuration is supported, resolve or explicitly disable the boundary before relying on it:

- selected-project/path/approval authority: H01-F4, K01-F01, T01-F07, X01-F01;
- delegated/browser/device mutation authority: T01-F04, T01-F08, K01-F06, X01-F02;
- network/remote auth: E02-F02, R01-F02, Q01-F01;
- multi-instance authority roots: E02-F15/F16/F17 when multiple explicit gateway instances are supported;
- instruction/provider authority: K01-F08, K01-F12, K01-F13;
- **local-trigger activation blockers:** A01-F08 and A01-F09 stay blockers **only if** local trigger execution is activated. Current default gateway does not run that consumer.

Where product scope excludes a conditional boundary—e.g. authenticated Windows crawler profile, multi-tenant cron administration, multi-instance gateways, optional desktop/crawler/CLI-app tools—the safe near-term disposition is **fail closed / hide / keep disabled**, not silently downgrade the source finding.

### Lane B — active correctness, durability and destructive-data risks

Prioritize mechanisms that can lie about success/completion, lose ownership or destroy retained data:

- H01-F1 mandatory source write swallowing;
- H03-F1 provider stream without terminal;
- S02-F01 hard deadline lacking independent owner and S02-F02 prelaunch reservation;
- E02-F20 applied mutation represented as completed failure;
- A01-F04 finite scheduler history and A01-F01/F03 blocking durability ownership;
- M01-F04 destructive prune on incomplete references;
- R01-F09/F10/F13 native completion/correlation/size pressure;
- credential custody R01-F01/F03 and provider login generations H03-F2/F7 where those login flows are used.

These do not all share implementation. They share the requirement that **success, ownership and durable effect state be truthful**.

### Lane C — release/qualification blockers for production distribution

Before calling a release/build/update path reproducible or migration-qualified, address R20/R21:

- exact pipx installed executable;
- frozen frontend resolution + bundle proof;
- cached toolchain and Gradle archive identity;
- signing secret phase isolation;
- staged deployment/rollback;
- immutable historical verifier;
- executable migration CI;
- async parity source coverage;
- two-distinct-APK upgrade test.

These can outrank ordinary P2 UI cleanup whenever a mobile/gateway release is being prepared.

### Lane D — bounded resource and client correctness

Then address per-owner resource and UI issues: R13, R14, R16, R17, R22 and the lower-risk portions of R07/R09. They matter, but should not preempt a live authority or destructive-data blocker merely because the fix is locally convenient.

### Lane E — structural simplification / optional retirement

R18 and the unmeasured H02-IO/S02-F03 structural costs remain **last unless measurements or maintenance burden justify them**. H02 source-backed artifacts, legacy recovery isolation and idle compaction are design alternatives/retirement candidates, not three urgent production defects. S01-F02 is private dead-code cleanup with unknown external subclass/monkeypatch compatibility.

## Reachability and activation matrix

| Boundary | Current source disposition | I05 treatment |
| --- | --- | --- |
| Default local-trigger execution | **Inactive** in foreground gateway | A01-F08/F09 are activation blockers; do not enable consumer before recovery + approval provenance. |
| Idle compaction runtime | Explicitly refuses start / no production constructor found | H02-IDLE-DORMANT remains D4 choice, not runtime incident. |
| Legacy onboarding wizard C00-F04/F05 | Current migration guard always denies main entry | Fix before reactivation/direct support; not a current default corruption/leak claim. |
| Multi-instance explicit gateway roots | Single-instance CLI usually aligns globals | E02-F15/F16/F17 and related P2 roots are conditional if multi-instance is supported. |
| Cron list/remove across chats | Global by instance | A01-F06 only becomes authority defect for mutually untrusted users/chats; can be intentional for single-owner agent. |
| Windows authenticated crawler profile | Optional deployment | Q01-F01 P1 conditional if same-machine untrusted principals are in threat model; Linux private Unix socket is different. |
| CLI apps / crawler / desktop tools | Optional/enabled by config | X01/T01 conditional P1s are blockers when those tools are enabled under restricted/delegated promises. |
| Mobile direct HTTP contract vs WS | Production WS path exists for some functions | R01-F04 is contract drift, not proof that all mobile attachment behavior fails. |
| Staging built-in skill relative path | Only current staging K01 delta | K01-STG-F01 remains conditional P2 overlay in R15, outside frozen 136. |

## Cross-package seams: share facts, never merge enforcement planes

The final architecture should stay **small-core / edge-owned** as required by `.agent/design.md`. The following shared facts are useful; global frameworks are not:

1. **Effective target fact** — absolute prepared path/project/instance identity may be shared between display, approval and execution. Filesystem policy, OS sandbox, remote scopes and approval remain separate enforcement planes.
2. **Effect outcome fact** — `not_applied / applied / partial / uncertain / ack_outstanding` may be shared across replay/projection. Source store, cron outbox, provider stream and mobile broker keep separate persistence.
3. **Generation fact** — UI selection, OAuth login, MCP readiness, model capability and tool registry can all use monotonic generation/identity fences, but no one global generation service is needed.
4. **Capability fact** — distinguish `declared / configured / registered / ready / authorized`; authorization never follows merely from registration.
5. **Process ownership fact** — reuse supervised child/process-group semantics, but retain OS-specific Windows Job/POSIX group handling and prelaunch-vs-running distinctions.
6. **Release provenance tuple** — source/dependencies/bundle/toolchain/artifact/migration/rollback identities can be emitted by existing build/deploy adapters, not a new release platform.

## D1–D7 dependency map — questions only, no choices

I05 reduces the later discussion to seven explicit product decisions while preserving the original **105 module-level questions** as evidence:

| Decision | Backlog materially controlled |
| --- | --- |
| **D1 — Supported executor/client platforms** | Windows process/crawler support, Android/native qualification, local sandbox/process guarantees, release target matrix. |
| **D2 — Authority/private boundaries** | R01–R05, remote/mobile scopes, multi-instance credential/root separation, logs/media capabilities, external CLI/browser/device trust. |
| **D3 — Managed sessions/delegation** | managed workspace projection, delegated read-only authority, task/source identity and per-session capability presentation. |
| **D4 — Compatibility/retirement/feature promises** | R15, R18, inactive goal/trigger/idle features, public setters/wrappers, legacy onboarding/recovery and optional tools. |
| **D5 — Registration and owner lifetime** | R07–R11, R17, tool/MCP request generation, process/close lifetime, history/data ownership. |
| **D6 — Attempt/recovery/release outcomes** | R06, R08, R10–R12, R20–R21, replay/uncertain effects, credential custody and staged rollback. |
| **D7 — Completion/resource/retention policy** | R06–R14, queue completion, byte/history limits, deletion completeness, process quiescence and public completion semantics. |

No value in this table is a recommendation disguised as a decision. The later user discussion may choose different supported modes, which can legitimately change which conditional findings remain implementation blockers versus explicitly unsupported paths.

## Final material gaps and audit limits

- **E01:** 95 primary channel/adapter files remain the user-authorized limited-depth exception. Twenty-six retained E01 findings remain separate limited-depth evidence and are **not included in the frozen 136 deep-source finding denominator**. I05 does not claim channel-level architecture completeness.
- **Runtime proof:** no audit stage executed application code, tests, builds, installers, browsers, mobile/native clients, services, deployments, fault injection or benchmarks. Static confidence and source reachability qualifications remain exactly that.
- **External compatibility:** public/private Python APIs, CLI wrappers, old storage, installed plugins/MCP servers and external subclasses may have consumers outside the repository. Retirement packages require explicit compatibility evidence.
- **Current-origin freshness:** current staging differs from frozen only in the separately audited K01 skill-summary source/test. Future staging changes would require targeted refresh before implementation.
- **H01-F8:** remains an open registration/connection-lifetime contract whose exact D5 behavior is a product decision; I05 assigns an owner but does not pretend the semantics are already selected.
- **No measured performance claims:** H02-IO, S02-F03 and similar structural findings remain unmeasured; simplification is justified only where correctness/maintenance evidence is independent of performance.

## Final integrated-review status

The source-review sequence is now complete:

- **824/824** deep-required frozen primary bodies coordinator-checked across sixteen source streams;
- **95** E01 primary bodies intentionally limited-depth;
- **136/136** original source findings mapped and reconciled;
- **105** original module discussion questions retained;
- **0** whole original finding entries removed as duplicates;
- **1** partial factual overlap documented (T01-F10 subset of K01-F11);
- **22** canonical repair packages, not new findings;
- **1** separate staging-only conditional P2 overlay, K01-STG-F01;
- **0** product fixes, tests, builds or deployments executed;
- **0** D1–D7 decisions accepted.

Under `status.json`'s definition—“cross-module reconciliation, material gap/disposition accounting and checkpoint evidence”—the **integrated source review is complete after I05**. The next stage is **D1–D7 discussion only**. Implementation planning should follow those decisions rather than silently assuming platform, authority, compatibility, lifecycle or resource contracts.
