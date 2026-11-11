# I04 — Release provenance, operational qualification, resource bounds and residual integration

**Stage:** 2026-11-07; fourth bounded integrated source checkpoint. **PASS for I04 source reconciliation only**, same assistant, no independent peer/runtime/build/deployment qualification. I04 assigns the final **26 previously unmapped original findings** to integrated owners. With I01–I04 together, **all 136 original published finding IDs now have at least one integrated substage reference**, but **this is not final deduplication, priority acceptance, repair authorization or integrated-review completion**. Original **105 discussion questions**, the **95-file E01 limited-depth exception**, and **K01-STG-F01** staging-only overlay remain unchanged.

Frozen source: `9562e57f8333d7f14b6aaeae837660723a9b2650`, tree `bbe2f237efe0cd7b935946b533f66ed0097a3c93`. Current staging `29612c3df01dc3dce3501fe349bcf555cd6c58b6` changes only K01 skill-summary source/test and is handled separately. This stage read the exact remaining finding entries from H01/H03/S01/T01/A01/K01/E02/R01/X01/Q01 coverage and selected pinned source ranges. It does **not** add primary-body credit to the frozen 824/824 denominator.

## Seven residual groups — all 26 remaining original IDs

| Group | Exact original IDs | Integrated disposition |
| --- | --- | --- |
| **I04-A — Installed/build artifact identity and release trust** | Q01-F03, Q01-F04, E02-F12, R01-F16, R01-F17, R01-F18 | Bind installer/build/signing output to exact installed executable, locked dependencies, verified bundle postcondition, toolchain identity, Gradle archive digest and phase-scoped signing authority. |
| **I04-B — Deployment switch, historical/CI and upgrade qualification** | Q01-F05, Q01-F06, Q01-F07, R01-F19, R01-F20 | Stage before switch, preserve rollback identity, pin historical verifier inputs, run CI for executable migration code, and qualify real async dispatch plus distinct baseline/candidate upgrade artifacts. |
| **I04-C — External process/network/output and global-index bounds** | T01-F09, T01-F11, T01-F12, X01-F03, E02-F04 | Preserve process-tree ownership and bound bytes/lifetimes **before** capture/download; cancellation must not orphan acquired VPN session state; global asset library must expose truncation/continuation. |
| **I04-D — Constructor/script input normalization** | H01-F3, K01-F10 | Validate canonical aliases/inputs before mutation/normalization and reject non-progressing numeric parameters before loops. |
| **I04-E — Approval target/provenance and narrow exemptions** | H01-F4, H01-F5, A01-F09 | Approval must inspect the actual selected execution target and only trusted user ingress may settle risky approval; AutoPush exemptions must be exact synchronization operations, not “contains git”. |
| **I04-F — Runtime capability/credential generations** | H01-F8, H03-F4, H03-F6 | Define tool-registry/connection request snapshot semantics, avoid persistent process-global provider credential mutation, and recapture immutable capability facts when direct model/window identity changes. |
| **I04-G — Compatibility and knowledge ownership cleanup** | S01-F02, K01-F02 | Remove/retire private unreachable legacy focus paths only after external compatibility check; bind knowledge writes to active-project association or expose deliberate global-write capability. |

**Count check:** 6 + 5 + 5 + 2 + 3 + 3 + 2 = **26 unique original IDs**. None appeared in I01/I02/I03. That exact non-overlap is mechanically checked in [I04 evidence](I04-release-residuals-evidence.json). Therefore I01–I04 now cover **136 unique original IDs**, but an ID having an integrated home is not equivalent to saying findings are duplicates, fixed, equally urgent, or accepted into one implementation package.

## I04-A — Release identity is a tuple, not “command exited zero”

### Q01-F03 — pipx verifies a temporary environment, not the installed executable

Frozen `scripts/install.sh:76–101,145–149,289–321` and PowerShell equivalent use `pipx install --force <target>`, then invoke Nanobot through `pipx run --spec <target> nanobot`. That execution path can resolve/build a **separate temporary pipx environment**, so version/setup verification need not exercise the persistent executable just installed. The finding is specific to the pipx fallback; Q01 explicitly does **not** generalize it to the uv path. Smallest correction: resolve and invoke the installed pipx app binary/environment directly for `--version`, onboarding and post-install readiness.

### Q01-F04 + E02-F12 — dependency lock and bundle postcondition are separate gates

Frozen `nanobot/webui/build.py:187–218` runs `[bun|npm, "install"]` without `--frozen-lockfile`/npm `ci`, while CI uses `bun install --frozen-lockfile` and `npm ci --dry-run`. That is Q01-F04's release-dependency reproducibility gap. **Do not merge it with E02-F12.**

E02-F12 is a *post-build evidence* gap. `build_webui_bundle` runs install and build, returns a fresh inspection object, but Hatch `hatch_build.py:40–103` discards that return value and only checks whether `nanobot/web/dist/index.html` exists. A pre-existing stale bundle can therefore survive a zero-exit no-op/misconfigured build and satisfy the existence check. The build source itself detects source-vs-index mtime before build; the packaging hook must prove the **post-build expected bundle/freshness/digest**, not merely that an old index file exists.

Smallest release contract: use the repository's locked frontend install path and record source commit + lock digest + built bundle digest; after build, assert the expected output was produced for that source. Preserve editable/local skip behavior as an explicit non-release mode rather than weakening release build identity.

### R01-F17/F18 — cached tool existence and version-pinned URL are not byte qualification

Frozen `mobile/scripts/bootstrap-toolchain.sh:30–132` verifies checksums only **when downloading** JDK/Go/uv/cmdline tools; if checkout-local executables already exist, it prints “already present” and returns without validating version/digest. Android components are installed only when `adb` is absent, so one old `adb` can suppress verification/install of the rest. `mobile/scripts/mobile-env.sh:24–46` checks executability, not versions. R01 source review additionally found the recorded Bun version was not installed/version-checked.

R01-F18 is separate: `mobile/android/gradle/wrapper/gradle-wrapper.properties` pins `gradle-8.14.3-all.zip` over HTTPS but has **no `distributionSha256Sum`**, so the Gradle archive itself is not repository-byte-pinned. Toolchain validation and wrapper archive integrity are different gates. Revalidate cached tools against a manifest or recreate them; add Gradle's official distribution digest; invoke a checkout-owned validated Bun. No assertion is made that a current host already has a malicious or wrong tool.

### R01-F16 — withhold release signing credentials until the signing phase

Frozen `mobile/scripts/build-release.sh:1–55` loads a mode-0600 release environment with `set -a`, then exports keystore context **before** `build-native-network.sh`, `build-shared-webui.sh`, `bunx cap sync android` and Gradle. Therefore compromised/untrusted build-time code in those earlier phases can read variables intended only for release signing. This does **not** claim current build dependencies are compromised or that a secret was exfiltrated.

Build unsigned/native/shared assets first; then invoke the final Gradle/signing process with a minimal environment containing key alias/password file/keystore path. Preserve keystore mode checks and final `apksigner`/SHA verification. Signing isolation is not solved by merely pinning Gradle or Bun.

**Reference comparison only:** pinned DSH release code binds source commit, clean tree, artifact digest, target and concrete runtime evidence before publishing checksums. Its pre-merge helper also uses a frozen lockfile. These are useful mechanisms, not Nanobot feature/parity requirements. Pinned Codex documentation uses locked Cargo installs, but it is secondary documentation evidence, not a source-implementation standard.

## I04-B — A real release gate must qualify the exact code, checker and upgrade transition

### Q01-F05 — deploy source advances before deployment is proven

Frozen `deploy/scripts/update_nanobot_gateway.sh:25–45` records old HEAD, fetches/switches/pulls the working checkout to the new branch revision, **then** calls the deployment script. The deploy script mutates the existing environment/config/service and restarts systemd. A dependency/config/restart failure can leave source, environment and service state inconsistent with no atomic previous-release switch/rollback receipt. No outage is claimed.

Smallest correction: prepare a versioned/staged release directory/venv from an immutable commit, run build/config/schema/service readiness checks there, atomically switch the service pointer only after success, retain the previous release and a rollback receipt. Updating a developer checkout in place is not an adequate production deployment transaction.

### Q01-F06 — historical baseline application pin is insufficient if the verifier moves

Frozen `docs/migration/run-p0-baseline.sh:7–20` pins old application commits but reads `p0-baseline-requirements.txt` and `verify-p0-archive.py` from the **current repository worktree**. It records the dependency freeze hash only after selecting/installing from that mutable file. A later helper/freeze change can alter the meaning of a “same old P0” run. This is a **historical comparability** concern, not evidence an existing P0 result is false. Pin helper + requirements digests, or extract them from an immutable audit/toolchain artifact associated with the baseline.

### Q01-F07 — executable migration code under docs bypasses the main CI path filter

Frozen `.github/workflows/ci.yml:1–30` ignores `docs/**` for push/PR. But executable `docs/migration/run-p0-baseline.sh` and `verify-p0-archive.py` live there. A PR touching only those executables can skip this workflow. Branch protection/other workflows were not proven absent, so the finding remains qualified. Move executable migration helpers under `scripts/` or narrow the ignore pattern to prose/assets and explicitly include executable migration files.

### R01-F19 — parity checker ignores async dispatch functions

Frozen `mobile/scripts/check-desktop-parity.py:118–143` searches only `ast.FunctionDef` named `dispatch`/`_dispatch_misc_routes`; production dispatchers in the audited source are `ast.AsyncFunctionDef`. Consequently advertised coverage can silently skip command/API rows. Fix parser coverage for both sync/async definitions and include one known async command + misc route self-test so the checker fails if it sees neither.

This is **qualification logic**, not another production authorization failure. It cannot close R01-F04 by itself until it examines the actual production dispatch shape.

### R01-F20 — same-APK reinstall proves persistence, not migration

Frozen `mobile/scripts/test-update-persistence.sh:18–51` installs `APP_APK`, seeds state, then “updates” by running `adb install -r` on the **same path** again. No baseline/candidate hash distinction exists. That proves reinstall persistence of one artifact, not schema/state migration across releases. Require separate baseline and candidate APK inputs, verify their hashes differ, seed under baseline, install candidate, then verify storage/credential/network migrations.

**Do not merge F19 and F20:** one tests checker source coverage, the other tests application update transition. Nor should R01-F05's substitute data-plane fixture (already I03) be counted here again; it remains a cross-reference that real production composition must be qualified.

## I04-C — Display caps are not resource caps; root exit is not descendant quiescence

### T01-F09 — generic owned-process cleanup stops reasoning when root is already exited

Frozen `nanobot/agent/tools/process.py:37–108` creates a process group/session, but `_signal_process_group` immediately returns if `process.returncode is not None`. `terminate_owned_process` then only waits/reaps the direct child. A launcher can exit while a descendant in the owned group remains; cleanup can therefore return without proving process-tree quiescence. This is **not** a claim all subprocess callers strand children or that the existing process helper has no value. Use the already established Windows Job owner where possible and on POSIX signal/check the known process group independently of root exit, with bounded escalation and quiescence proof.

### X01-F03 — CLI app output is bounded only after full capture

Frozen `nanobot/apps/cli/service.py:_run_argv:1022–1038` uses `subprocess.run(capture_output=True, text=True)`; later display truncation does not bound in-memory stdout/stderr accumulated during up to the configured run timeout. No OOM/peak memory was measured. Move CLI Apps through a bounded streaming reader under the same real descendant owner; preserve exit code and a bounded retained prefix/tail. This combines implementation seams with T01-F09 but **not defect identity**: one is process-tree lifetime, the other pre-truncation memory capture.

### T01-F11 — web_fetch model-output truncation occurs after unbounded body read

Frozen `nanobot/agent/tools/web.py:1328–1365` opens a safe streamed response but, for images, calls `await r.aread()` with no separate response-byte ceiling before producing bounded model content. The wider source finding applies to retrieved response bodies generally. SSRF/redirect safety remains a positive control; the defect is **resource admission**, not URL authorization. Enforce Content-Length where trustworthy and stream/count actual bytes with a hard independent ceiling; keep model-visible maxChars as a later projection cap.

### T01-F12 — thread-backed VPN setup can outlive cancellation and lose the returned lease

`TrendVpnBrokerClient.request_async` is `await asyncio.to_thread(self.request,...)`. `TrendVpnTool.execute` can be cancelled while the blocking thread continues a privileged `start_session`; if it later succeeds, the cancelled caller never receives the session handle needed for cleanup. Existing Unix socket request/response byte/time limits do not provide cancellation ownership. Use native async broker I/O or request IDs/lease generations so cancellation can query/reap a late session result. Do not blindly issue a second `start_session`.

### E02-F04 — global generated-assets library silently stops at first 200 sessions

Frozen `nanobot/webui/generated_asset_index.py:104–130` derives persisted WebUI session keys, then `load_all_generated_asset_groups(..., max_sessions=200)` iterates only `webui_session_keys(webui_dir)[:max_sessions]` with no continuation/truncation field. This is a **visibility/capacity contract**, not an unbounded-memory issue. Add page/cursor or persistent global index, and explicitly return truncation/continuation; do not silently scan everything on each request or discard older sessions.

**Cross-reference only:** I02 already owns R01 media cache, M01 reference image/PDF decode, and E02 full transcript projection budgets. They are not re-counted in I04.

## I04-D — Normalize/validate once before state mutation or non-progressing loops

### H01-F3 — media adoption mutates aliases before alias agreement validation

Frozen `nanobot/agent/turn.py:241–331` chooses canonical `content/prompt/message`, then calls `_adopt_media()`, which rewrites content/prompt and removes attachment markers **before** checking `prompt` versus original `message`. A valid alias pair can be transformed into an artificial mismatch; content/prompt-only construction behaves differently. This is a constructor ordering/integrity bug, not an attachment-retention finding. Validate supplied aliases against a stable pre-normalization canonical value, then adopt media exactly once and assign all compatibility aliases from the normalized result.

### K01-F10 — zero/negative gap parameter makes the frame-fill loop non-progressing

Frozen `extract_keyframes.py:147–160` loops while `current_time - prev_time > max_gap_seconds` and increments `prev_time += max_gap_seconds`. `_parse_args:569–581` accepts any float for `--max-gap-seconds`; `main:601,621` passes it directly. Zero never advances; negative moves backward, so later frame cap does not save the loop. Reject `max_gap_seconds <= 0` (and other clearly invalid spacing/count parameters as appropriate) before probing/processing. No script was executed in this audit.

## I04-E — Approval must be tied to the actual effect target and trusted actor

### H01-F4 — risky-file approval inspects agent-home-relative file while execution targets selected project

At frozen `nanobot/agent/loop.py:3518–3532`, `RunPolicyContext.workspace` is initialized from `deps.resource_policy.workspace` (agent home). `RiskyActionPolicy` resolves write/edit paths through that workspace; the selected project used by actual filesystem execution can be different (already cross-verified in I01). Thus a relative project edit can be approved based on a nonexistent/different same-name file under agent home, missing the intended large/bulk-change approval. It is **not a sandbox escape**. Approval inspection must consume the same prepared absolute target/source proof that the file tool will execute, after current workspace/grant selection.

### H01-F5 — AutoPush risk exemption is broader than synchronization

Frozen `nanobot/agent/policy.py:278–325` returns “not risky” for any command containing a Git token when cwd resolves inside an AutoPush repo, except a narrow branch-delete detector. Shell deny patterns still block some destructive compounds, and managed Sandbox/Once restrict configured repos, so the original report correctly rejects a blanket exploit claim. Narrow the approval exemption to an explicit allowlist/structured operation for intended status/add/commit/push synchronization; commands such as reset/clean must pass normal risky-exec classification. This does not require replacing shell policy.

### A01-F09 — inactive trigger automation can settle approval if later activated

The default gateway does **not** run the local-trigger consumer (I03 A01-F07), so A01-F09 is an **activation blocker**, not a default exploit. Frozen `AgentLoop` calls `approval_admission.snapshot(key, request.content)` before deriving `trusted_foreground_activity`; the special guard only prevents `silent_output` cron from consuming a pending approval. A local-trigger request bound to the same session can therefore contain text such as “yes/go ahead” that matches approval admission despite not being trusted foreground user ingress.

Before activating local triggers, make approval consume require trusted user/foreground provenance and exact pending-request identity. Cron/trigger/recovery/background automation may report approval needed but may not mint affirmative/negative user authority. This complements I01 authority work without re-counting the same finding.

## I04-F — Runtime identities need explicit generation ownership

### H01-F8 — tool definition snapshot vs live registry/reconnection is an unresolved contract, not a proven blanket bug

Runner snapshots `spec.tools.get_definitions()` before provider request, while live registry paths can later resolve current names and MCP settings can reload/replace connections. H01 deliberately classified this as an **open request/connection registration contract**, not proof every call dispatches through stale tools. I04 preserves that qualification.

Final D5/D7 design must state whether a turn uses:
1. an immutable tool+connection generation from prompt through dispatch, rejecting revoked handles after generation change, or
2. live resolution at execution with an explicit “tool became unavailable/changed” outcome and no hidden capability expansion.

Filtered role/artifact registries and dynamic MCP reload must follow the same declared contract. Do not clone every registry per token or block legitimate revocation merely for consistency.

### H03-F4 — Bedrock API key mutates process-global environment

Frozen `nanobot/providers/bedrock_provider.py:45–76` writes `AWS_BEARER_TOKEN_BEDROCK` into `os.environ` when an explicit `api_key` is supplied, then leaves it there while creating a boto3 session/client. This creates ambient process credential state outside the provider instance and can affect later SDK consumers/configs. No deployed cross-account leak is proven. Prefer explicit SDK credential injection where available; otherwise deliberately document one-process credential ownership and restore/guard mutation with synchronization rather than silently leaving a provider constructor side effect.

### H03-F6 — direct model/context setters keep stale immutable capability facts

Frozen `nanobot/agent/model_runtime.py:145–188` builds immutable runtimes/capabilities; `select_model` and `select_context_window` replace identity fields directly without reconstructing capability metadata/snapshot generation. Source review narrowed reachability because the normal preset/config path refreshes snapshots and active-session tooling blocks some direct setters. Either recapture capability facts whenever these public setters are supported, or retire/narrow the direct setters after caller compatibility review. Do not infer stale capabilities across normal preset selection.

## I04-G — Compatibility removal and global knowledge writes need explicit product contracts

### S01-F02 — private legacy WorkspaceFocus branches are unreachable in current owner

Frozen `WorkspaceFocusOwner.managed` always returns `True`; therefore current in-tree owner construction cannot take `effective_access`'s unmanaged legacy branch, `validate_scope`'s false branch, or private `_legacy_access/_allowed/_source` compatibility helpers. This is **low-priority simplification**, not a security defect. The current managed focus path and public contracts remain required. Remove private dead branches only after checking external subclass/monkeypatch/package compatibility; do not alter ResourceAccessPolicy or Full/Sandbox semantics as part of cleanup.

### K01-F02 — knowledge tool says “current project's knowledge_repos” but accepts any visible named knowledge repo

Frozen `SaveKnowledgeTool.execute:220–287` obtains `self.catalog.get(repository)`, requires `kind=="knowledge"`, validates path/secret/root/write authority and optional sync—but does **not** verify that this repository is associated with the active selected project. The write is still restricted to the registered knowledge repo; this is **ownership/integrity**, not filesystem escape. Bind allowed repository names to current project association, or expose/document a separate explicit instance/global knowledge-write capability. Preserve the existing secret scan, root containment and writer policy.

## Release/qualification simplification direction — minimal shared facts, not a new build platform

The combined I04 evidence supports a compact **release provenance tuple**:
- source commit/tree + dirty state,
- Python/package dependency identity,
- frontend lock + built bundle digest,
- native/mobile toolchain version+digest,
- target/platform + produced artifact digest/signature,
- migration/checker version,
- deployment active release + previous rollback release.

Each build/deploy/test owner should emit/consume only the fields it owns. Do **not** invent one universal package manager or replace service/systemd/Android/CI tooling. Similarly resource owners should share byte/time/deadline vocabulary without centralizing parsers, HTTP clients, process managers or UI indexes.

**Order of work candidate, not accepted plan:** (1) fix deterministic qualification false positives—pipx installed executable, frozen frontend build/postcondition, Gradle digest, async parity checker, two-artifact upgrade test; (2) phase-scope signing secrets and validate cached toolchain; (3) stage/rollback deploy; (4) close bounded resource/process edges; (5) input/approval/runtime generation and low-risk compatibility cleanup. Actual ordering must still consider original priorities and deployed modes during the final reconciliation stage.

## Final accounting after I04 — mapping complete, integration still open

The [I04 evidence ledger](I04-release-residuals-evidence.json) mechanically reads all 16 original deep-source coverage ledgers and I01/I02/I03 evidence maps. Before I04 there were exactly **110 unique original IDs mapped and 26 unmapped**. The 26 IDs in I04 are exactly that residual set, no duplicates with earlier integrated groups. Therefore:

- **136 / 136 original published finding IDs now have at least one I01–I04 integrated group**
- **136 original findings remain the count**; I04 promotes zero new findings
- **105 original discussion questions remain**; no new D1–D7 question is added
- **824 / 824 deep-required source bodies remain the frozen source-review count**
- **95 E01 primary bodies remain limited-depth**, not upgraded
- **K01-STG-F01 remains one separate staging-only conditional P2**, outside the frozen 136
- **integrated_review_complete remains 0**

Why not close integrated review now? Mapping is only the prerequisite. The next stage must perform **whole-project reconciliation across the 136 mapped IDs**: identify actual duplicates versus related-but-separate findings, produce one source-of-truth repair/package ownership table, retain all conditional/default-reachability qualifiers, reconcile severity/priority order without inflation, and show which final D1–D7 questions materially control the backlog. That stage must not silently choose D1–D7 for the user.

No repository tests, builds, installers, scripts, apps, services, browser/mobile/native runs, package downloads, deployment commands, network probes or product source changes were performed in I04. Static source and existing accepted report evidence only.
