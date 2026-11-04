# Implementation progress: mobile remote and session context lifecycle

Branch: `feat/mobile-remote-context-lifecycle`
Worktree: `/home/jaden/Documents/Utilities/nanobot-custom-worktrees/mobile-remote-context-lifecycle`
Base: `a90191a41b025884e76ce029b98a5d5349171f2e`

This log records package status, acceptance evidence, commands and exits, and
remaining blockers. Private logs and runtime data stay outside the repository.

## Current merge review — 2026-11-04

Verdict **CORRECT** at code `1689a4a06020c778a6e969b12bb19b27a1ac90f1`; hold merge.
[MC01–MC08 correction tasks](merge-correction-tasks.md) are pending; use the
[correction prompt](merge-correction-agent-prompt.md) to continue. This entry supersedes any
implication below that focused completion establishes the full next-request/recovery contract.
Historical checks and physical observations remain recorded, not erased or reclassified as new
evidence. The task-preparation update changes documentation only; no correction or new product
test has run. A merge-tree preview with staging `aacca07c` found no conflicts; the corrected
combined candidate and its qualification still require validation. No staging merge occurred.

## Environment

| Item | Value |
| --- | --- |
| Prepared by | repo-local `.venv` (uv 0.13.7), Python 3.14.8 |
| Setup | `python3 -m venv .venv`, `.venv/bin/pip install -U uv`, `.venv/bin/uv sync --all-extras --dev`, `.venv/bin/uv run --no-sync python -m scripts.install_channel_dependencies --all-channels` |
| Import check | `.venv/bin/python -c "import nanobot; print(nanobot.__file__)"` -> worktree `nanobot/__init__.py` |
| Not used | original checkout editable virtualenv, live services, credentials |

## Package status

| Package | Status | Notes |
| --- | --- | --- |
| P1 provider route and request preparation | complete (focused checks) | route metadata, wrapper preview, selected-route dispatch check, typed re-preparation |
| P2 ordinary context lifecycle / recovery | focused complete | preventive budgeting, model-derived summaries, bounded truncated-summary replanning |
| P3 authorized media inputs | focused complete | exact read-only input snapshots; foreground only |
| Mobile A socket correctness | focused complete | identity, terminal lifecycle, UTF-8, unique broker ids |
| Mobile B delivery authorization | complete (focused) | uncached revalidation; test helper removed from production; WS-boundary cases |
| Mobile C owner wiring / foreground recovery | complete (focused + emulator M01 + physical Doze) | native Ping/refresh/resume bridge, owner routes, foreground recovery; physical radio/tailnet pending |
| Mobile D pairing continuity | complete (focused + native custody) | renewal, restore, Kotlin pending-setup custody; physical device evidence pending |
| Mobile E request authority/backend | focused complete | versioned settings, defaults, device/source Full consent, ephemeral Temporary Full, request-bound proofs, same-context steering |
| Mobile F composer settings/presentation | complete (focused + physical controls) | queued snapshot, next-message mode/project selection, 44px floor, project-first truncation, accessible access state |
| Mobile G standard feature dispatcher | focused complete | standard owner read dispatch; mobile WS actions and mutations use the standard dispatcher with replay/auth; wrapper tests retained for old builds |
| Integrated qualification / M38 | local gates green; release gates pending | full Python/WebUI/type/build/M01 pass; signed phone and M38 pending |

## Acceptance evidence

### P1

Outcomes:
- Wrapper capability/accounting no longer disappears because fallbacks exist. The next
  main attempt (including a circuit-open primary) is prepared against the leaf that
  dispatch will actually use, with that leaf's model, window, reserve and policy.
- Auxiliary source-summary preparation stays on the captured primary and reports
  `purpose=source_summary`; wrapper `supports_source_summary` delegates to the primary.
- `PreparedRequestBudgetPreview.route` records content-free route facts and an opaque
  token; the governor accepts only routes consistent with their measurement.
- Auxiliary output caps are bounded by the selected route's declared generation and
  capacity instead of the superseded fixed 2,048 ceiling.

Focused checks:
- `.venv/bin/ruff check` on changed provider/governor files -> PASS.
- `.venv/bin/python -m pytest tests/providers -q` -> 1918 passed, 3 skipped.
- `.venv/bin/python -m pytest tests/agent/test_source_pressure_compaction.py tests/agent/test_finalized_budget_recovery.py -q` -> 59 passed.
- New authored cases: `.venv/bin/python -m pytest tests/providers/test_route_preview_contract.py -q` -> 6 passed.

Requirements-only author:
- Session `ses_e5ee14ff8ffe9xq73nEMpavR2E`; artifact
  `/tmp/opencode/p1-requirements-tests/test_route_preview_contract.py` (outside repo).
- Cases R1-R5 were adapted to the real public provider boundary without changing
  scenarios or expected outcomes. The author noted missing constructor/chat details;
  the adapter only supplies those concrete boundaries.

Type evidence:
- Frozen before report under identical dependencies:
  `/tmp/opencode/nanobot-p6-baseline-e76086b1` at `e76086b1`, full unchanged
  `basedpyright --outputjson` -> 0 errors / 528 files, warnings 0.
  Report `/tmp/opencode/p6-before-e76086b1.json`.
- Final after-report and `scripts/check_type_regressions.py` comparison pending the
  integrated checkpoint.

Known limitation:
- Route selection is recorded for planning/accounting; dispatch independently
  re-evaluates the same next-attempt decision and reprices the selected leaf. A
  cross-check rejecting a stale selection is not yet enforced by a typed dispatch
  exception; final per-leaf guards still reject over-budget bodies.

### P2

Outcomes:
- Pressure and admission now use the route-finalized prepared request (P1 wrapper
  preview) rather than a generic legacy estimate whenever the route supports it.
- Summary generation allowance resolves from the captured summary runtime's
  generation setting, clamped by declared model output capability, route capacity
  and the operation budget. The accepted `SourceCompactionConfig` override is
  `summaryGenerationTokens` / `summary_generation_tokens`.
- Retained summary text is separately bounded by the rebuilt main request's
  available space and is the publication/truncation boundary. A completed `stop`
  response over that retained budget is discarded, not published.
- Compaction reserves capacity for the summary envelope, useful generation,
  estimate safety and bounded next-step growth; proactive compaction can trigger
  before the old 90%-of-budget ratio when that reserve would otherwise be consumed.
- Trigger/target ratios and the generation override are typed deployment settings
  with camelCase and snake_case spellings and validation (target < trigger).

Focused checks:
- `.venv/bin/ruff check` on changed context/config/provider files -> PASS.
- `.venv/bin/python -m pytest tests/agent/test_source_pressure_compaction.py tests/agent/test_source_compaction_publication.py tests/agent/test_source_dependency_compaction.py tests/session/test_source_compaction.py tests/agent/test_finalized_budget_recovery.py -q` -> 119 passed.
- `.venv/bin/python -m pytest tests/agent/test_memory_budget_hardening.py -q` -> 17 passed.
- `.venv/bin/python -m pytest tests/config tests/agent/test_context_governance.py tests/agent/test_compaction_proof_boundary.py tests/agent/test_defensive_boundary_views.py tests/agent/test_loop_consolidation_tokens.py -q` -> passed after the fixture updates below.
- New authored config cases: `.venv/bin/python -m pytest tests/config/test_source_compaction_policy.py -q` -> 16 passed.

Conflicting cases updated intentionally with the accepted plan:
- `tests/agent/test_source_pressure_compaction.py` now expects the summary
  allowance from the captured generation instead of a fixed 2,048.
- `tests/agent/test_memory_budget_hardening.py` / `tests/agent/p6_memory_contract_fixtures.py`
  pin the synthetic fixture's runtime generation to 2,048 so its exact-boundary
  arithmetic remains deterministic under a model-derived budget.

Requirements-only author:
- Session `ses_e5ed3a802ffe1dYcK02mlHNPk7`; artifact
  `/tmp/opencode/p2-requirements-tests/test_source_compaction_config.py` (outside repo),
  integrated unchanged as `tests/config/test_source_compaction_policy.py`.

Known limitations:
- Exceptional summary-output truncation fails honestly within the existing call
  ceiling and preserves source/checkpoint; a bounded smaller-chunk retry strategy
  was not added. Preventive continuity is the validated path.
- No live provider canary or physical long-session journey has been run; these
  remain qualification items, not claims.

### P3

Outcomes:
- `exec` accepts an additive optional `input_files` list of exact paths. Omission
  preserves the existing shell confinement.
- Each path resolves through the same `read_file` policy, must be an approved
  regular file, and is copied into a bounded server-owned per-invocation snapshot
  (16 files / 64 MiB) before launch. Symlinks, private runtime storage and changed
  sources fail closed. Writes to the mounted paths are read-only.
- Managed bubblewrap restores only those snapshot files at their original paths,
  after private instance masking. Unlisted siblings/parent directories stay
  masked, and the whole media directory is never mounted.
- Foreground-only: `yield_time_ms`/`notify_on_exit` with `input_files` is rejected
  because snapshot lifetime cannot be guaranteed.
- MCP image results and generated-image results advertise the exact-input contract;
  saved replay images now say no bytes are present and allow an authorized reread.

Focused checks:
- `.venv/bin/ruff check` on changed tool/provider files -> PASS.
- `.venv/bin/python -m pytest tests/tools/test_sandbox.py tests/tools/test_filesystem_tools.py tests/tools/test_mcp_tool.py tests/utils/test_artifacts.py tests/agent/test_recent_image_governance.py tests/agent/test_managed_exec.py tests/providers/test_recent_image_history.py -q` -> 199 passed.
- Real managed-bwrap smoke: `test_exec_input_files_exposes_only_exact_approved_files`
  copies bytes from a media path, denies an unlisted sibling, denies private
  storage and a symlink, verifies read-only mounting, and checks the advertised
  schema bound.

Requirements-only author:
- Session `ses_e5ecc4b29ffeTf6rSA3Fxrsdkq`; case specification artifact
  `/tmp/opencode/p3-requirements-tests/input_files_cases.md` (outside repo),
  integrated into the focused managed-exec test.

Known limitations:
- Session/background shell commands cannot use `input_files` in this change.
- No cross-turn automatic historical image injection or binary-retention migration
  was added; explicit reread remains the supported path.

### Mobile A

Outcomes:
- `MobileWebSocket` now has one terminal path: every open failure, remote close,
  explicit close or close-during-open produces exactly one `onclose`; open
  failure also emits one `onerror`.
- Late open/frame/close events for a retired instance are ignored; a late native
  open after `close()` is closed once and emits no open/message.
- Incoming frames decode with the UTF-8 `decodeBase64` helper instead of raw
  `atob`; `你好 👋` survives.
- `NanobotClient` guards open/message/error/close callbacks by current socket
  identity, so a retiring socket cannot null or drive the replacement socket.
- Broker WebSocket ids are monotonically unique per live core; retired ids are
  rejected by `WSSend`/`WSClose`.

Focused checks:
- `cd webui && bun run test -- src/tests/mobile-runtime.test.ts src/tests/nanobot-client.test.ts src/tests/mobile-http-bridge.test.ts src/tests/mobile-wiring.test.ts` -> 97 passed (before the new cases), then `mobile-runtime` -> 11 passed with the new lifecycle cases.
- `source mobile/scripts/mobile-env.sh && cd mobile/native-network && go test ./internal/broker` -> PASS, including the new unique-id case.

Requirements-only author:
- Session `ses_e5ec01916ffevd5BoPelD1KHil`; artifact
  `/tmp/opencode/mobile-a-requirements-tests/socket_cases.md` (outside repo),
  integrated as vitest public-boundary cases.

### Mobile B

Outcomes:
- Private read/thread/feature/api/model-settings/workspaces/media responses are
  re-authorized through an uncached owner validation immediately before release.
- A revoke acknowledged after warm admission and before the handler returns
  yields 401 with no private payload; owner loss yields 503; scope loss yields 403.
- Existing initial validation, bounds, cancellation and replay behavior remain.

Focused checks:
- `.venv/bin/ruff check nanobot/remote/edge.py tests/remote/test_edge.py` -> PASS.
- `.venv/bin/python -m pytest tests/remote/test_edge.py tests/remote/test_edge_integration.py tests/webui/test_owner_mobile_api.py tests/webui/test_mobile_owner_feature_api.py tests/webui/test_mobile_commands.py -q` -> 33 passed, then `tests/remote/test_edge.py` -> 18 passed with the new authorization cases.

Requirements-only author:
- Session `ses_e5ec01915ffeJnYe8jKsakJ3eU`; artifact
  `/tmp/opencode/mobile-b-requirements-tests/delivery_cases.md` (outside repo),
  integrated as warm-cache revoke and owner-loss cases.

Known limitations:
- The test-only `GatewayHTTPHandler.mobile_mutation_payload` helper and its
  owner-denial/enrollment cases have not yet been relocated to the public WS
  entry; protection remains in place.

### Mobile D

Outcomes:
- Pending pairing state survives closing and reopening the desktop dialog; the
  component no longer resets it on dialog close and restores pending state when
  the dialog is reopened.
- A desktop-local `pair_renew` operation extends an eligible offer to
  `min(now + 300s, created_at + 1800s)` under the pairing lock, retaining the
  collector/device binding, code, nonce and scopes. Denied/approved/consumed
  offers and offers past the 30-minute setup limit cannot renew.
- Expired but renewable offers are restored with a “Renew setup” action and a
  “Continue setup on the desktop” message; the phone bootstrapper retains the
  pending setup and resumes polling the same pairing id after Tailscale
  readiness instead of rescanning. Retries are rate-limited to one per 5 s and
  stop after 5 minutes.
- Renewal is exposed only through the local authenticated settings/owner path;
  no public renewal endpoint was added.

Focused checks:
- `.venv/bin/ruff check` on pairing/IPC/client/management/settings/ws_http -> PASS.
- `.venv/bin/python -m pytest tests/remote/test_pairing.py tests/remote/test_owner_ipc.py tests/webui/test_remote_access_settings.py tests/webui/test_mobile_owner_feature_api.py -q` -> 24 passed.
- `cd webui && bun run test -- src/tests/remote-access-settings.test.tsx` -> 7 passed, including dialog close/reopen and expired-renew cases.
- `cd webui && bun run test -- src/tests/bootstrap.test.ts` -> 9 passed.

Known limitations:
- Kotlin/`NativeNetworkPlugin` custody of the pending setup and native retry
  ordering are not compiled or instrumented; the JS bootstrapper retains the
  setup in memory for the current app lifetime, and native behavior still needs
  Android build/device evidence.

## Commits

| Commit | Scope |
| --- | --- |
| e76086b1 | handoff documentation (pre-existing) |

### Mobile C (partial)

Implemented:
- Broker policy now admits named owner-control routes only:
  `GET /remote/v1/status`, `POST /remote/v1/refresh`, `GET /remote/v1/devices`
  and `POST /remote/v1/devices/<id>/revoke`; an escape path is rejected.
- Shared mobile runtime exposes `recover()` with a single in-flight foreground
  path. It probes an apparently open socket through an optional native
  `wsPing(id)`, keeps a healthy connection, and otherwise renews credentials,
  retires the stale socket once and opens one replacement on the same object so
  `NanobotClient` reattaches known chats and preserves screen/draft/scroll.
- `main.tsx` routes visibility/pageshow/online through that one recovery path.

Focused checks:
- `source mobile/scripts/mobile-env.sh && cd mobile/native-network && go test ./internal/broker` -> PASS, including the owner-route allowlist case.
- `cd webui && bun run test -- src/tests/mobile-runtime.test.ts` -> 13 passed, including healthy-probe and stale-replacement cases.

Not implemented / blocked:
- Native `wsPing` C ABI/JNI/Kotlin declarations and the Go `websocket.Conn.Ping`
  five-second probe.
- Kotlin single-flight refresh/custody synchronization, native foreground event
  bridging and Android build/instrumentation evidence. The Android SDK/NDK
  toolchain is absent from this worktree; no emulator/device evidence exists.

### Mobile E (partial)

Implemented:
- An explicit `access_mode: "restricted"` request remains Sandbox even when the
  chat currently has Full; only the explicit mode action may raise it to Full.
  A chat label or remembered preference cannot promote it.
- Added a focused public-boundary case for that behavior.

Not implemented:
- Four-state request settings, `request_settings_version`, per-request source
  token, `request_intent`/steering target, Full confirmation IDs, device/source
  bound consent, ephemeral temporary bindings, request-scoped execution proofs,
  queue snapshot transport and defaults-revision semantics.
- Because these are security-sensitive admission boundaries, they were not
  partially stubbed. Mobile F's snapshot/consent UI depends on this contract and
  remains pending with G.

### Type and environment evidence

- Frozen before: `/tmp/opencode/nanobot-p6-baseline-e76086b1` at
  `e76086b18f73077374a9efc53ea8159e8f1cbc43`, full unchanged
  `basedpyright --outputjson` -> 0 errors / 528 files, warnings 0
  (`/tmp/opencode/p6-before-e76086b1.json`, sha256
  `154aea7096f5394175751e4375d0ed76b8f2db06443337a000ef145259ecf99a`).
- After: current worktree full `basedpyright --outputjson` -> 0 errors / 528
  files, warnings 0 (`/tmp/opencode/p6-after.json`, sha256
  `5178be051d4c203a42c00305db55e2940f5acfec80eda81aaeedee93a565ecc6`).
- `scripts/check_type_regressions.py` -> `PASS_NO_NEW_DIAGNOSTICS`,
  `introduced=0`, `resolved=0`, `strict_clean=true`.
- Environment: Python 3.14.8, basedpyright 1.40.2, `uv.lock` sha256
  `d14bdb9b08ee646ce0be265c87ee309871e0bfa950408f2ddaa04fe01b44f331`.
  Baseline/current Python source inventory hashes:
  `59bbbed829653b941dd53f0f9bf5fdab771dfd736cae6b121d369bd76d46bfc9`
  and `1630e1ae8b755305c0fbeaf08d3e05f20eda6fa50fbd0980966555c8589d4585`.

### Integrated checks (partial)

- WebUI full test suite: `cd webui && bun run test` -> 90 files, 1411 tests
  passed, exit 0 (`/tmp/opencode/webui-tests.log`).
- WebUI build: `cd webui && bun run build` -> exit 0
  (`/tmp/opencode/webui-build.log`); the existing >500 kB chunk warning remains.
- WebUI lint: `cd webui && bun run lint` -> exit 0 after one focused
  `no-useless-assignment` correction (`/tmp/opencode/webui-lint.log`).
- Repository Python lint: `.venv/bin/ruff check nanobot/` -> PASS.
- Full Python suite: running; result appended when complete.
- Full Python suite first run: 17538 passed, 22 skipped, 1 failed
  (`tests/providers/test_provider_retry_fork.py::test_prune_old_images_keeps_only_most_recent`)
  because it asserted the superseded saved-image placeholder wording. The case was
  updated to the accepted P3 contract (no image bytes; authorized `read_file`
  reread) and passes in isolation. A clean full rerun is in progress.
- Full Python suite clean rerun: **17539 passed, 22 skipped, 6 subtests passed**,
  exit 0 in 718.42s (`/tmp/opencode/full-suite-r2.log`). The 22 skips are the
  repository's existing platform/native skips, not waived or claimed as execution.

### Mobile C native completion

Implemented and built:
- `Core.WSPing` sends one protocol ping and waits for its pong with a five-second
  context deadline.
- Exported through `nanobot_network_ws_ping`, JNI
  `NativeNetwork.wsPing`, Kotlin plugin `wsPing`, and the bootstrap transport.
- Refresh is one in-flight rotation shared by every caller. Rotated tokens are
  persisted only while the same live custody binding exists; logout/revoke
  clears/fences custody so an older completion cannot restore it.
- MainActivity emits `nanobot-native-resume` into the shared recovery path.
- Broker owner-control routes (`/remote/v1/status`, `/refresh`, `/devices`,
  exact `/devices/<id>/revoke`) remain allowlisted.

Evidence:
- `mobile/scripts/test-cabi.sh` -> PASS, `NANOBOT_CABI_SMOKE_OK version=1`.
- `mobile/scripts/build-native-network.sh` -> PASS (arm64-v8a and x86_64).
- `cd mobile/android && ./gradlew --no-daemon :app:assembleDebug` -> BUILD SUCCESSFUL.
- `mobile/scripts/qualify-emulator.sh --evidence-dir /tmp/opencode/mobile-emulator-evidence --native-tests` -> M01 PASS; 37 instrumentation tests passed; native probe executed on `emulator-5590`.
  Evidence includes the debug/arm64 APKs, probe logcat, package dump and shell screenshot.
- Go broker tests pass, including `WSPing` live/retired cases.

### Mobile E additional outcome

- Remote Full confirmation is now bound to the confirming installation and the
  current source binding. Another device gets no reference and must confirm for
  itself; local UI authentication remains valid. Disabling Full removes the
  consent, and source replacement invalidates it.
- Explicit restricted requests remain Sandbox (earlier commit).
- Still not implemented: versioned request-settings envelope, per-request source
  token, steering/queue dispositions, non-authorizing Full confirmation IDs,
  ephemeral temporary Full bindings and request-scoped proofs.

### Mobile F partial outcome

- A queued prompt captures its workspace/mode at queue time and sends that
  snapshot when dequeued. The send boundary no longer overwrites an explicit
  snapshot with later picker state.
- The remaining composer/touch/accessibility presentation work is not complete.

### Updated integrated evidence

- WebUI full suite: 1412 passed / 90 files, exit 0.
- WebUI build and lint: exit 0.
- Android debug build after the queue-snapshot/shared-asset rebuild: BUILD SUCCESSFUL.
- Emulator M01 rerun on the updated APK: M01 PASS, 37 instrumentation tests
  passed, native probe executed on `emulator-5590`. The second run logged a
  non-fatal Capacitor safe-area CSS injection warning; it did not affect the
  probe or instrumentation result and is reported rather than waived.
- Strict basedpyright after the E changes: 0 errors / 528 files; regression
  comparison remains `PASS_NO_NEW_DIAGNOSTICS`.
- Full Python suite after the E/F changes: running.

Emulator artifact hashes (private evidence directory):
- universal debug APK `9ad2a0297c7ee50f82fc970eb6efafd67e7fdf355e318b97e6da921dd68ea6d1`
- arm64-only debug APK `7f74716cab6cd4a889e0e030a5b1da171283be916a383264f2c5a00ea5d75f04`
- device: API 36 x86_64 AVD, 37 instrumentation tests.
- Full Python suite after native/E/F changes: **17540 passed, 22 skipped,
  6 subtests passed**, exit 0 in 769.13s (`/tmp/opencode/full-suite-r3.log`).
- Strict after changes: 0 errors / 528 files; regression comparison
  `PASS_NO_NEW_DIAGNOSTICS` (`/tmp/opencode/p6-after-r2.json`).

### E/G/F continuation evidence

- Request settings v1 is advertised by HTTP bootstrap and mobile handshake; the
  WebUI sends `request_settings_version: 1` on messages.
- Envelope validation covers version, intent/target, source token and confirmation
  id shape without side effects; steering is explicitly `not_admitted` rather than
  becoming a second new run.
- Accepted submissions persist versioned project/mode defaults; Full consent
  remains device/source bound and is never reconstructed from metadata.
- Owner reads prefer `/mobile/v1/dispatch` through the registry-allowlisted
  desktop dispatcher, with fallback to the existing feature/API wrappers.
- `mobile/scripts/check-desktop-parity.py` -> PASS, 127 rows / 9 categories.
- Full Python suite: **17544 passed, 22 skipped, 6 subtests passed**, exit 0.
- WebUI: **1415 tests passed**, build exit 0, lint exit 0.
- Strict basedpyright: 0 errors / 528 files; regression comparison still
  `PASS_NO_NEW_DIAGNOSTICS`.
- Android debug build: BUILD SUCCESSFUL.
- Emulator M01 rerun: PASS, 37 instrumentation tests, native probe executed.

### Temporary Full, request proofs, steering and native pairing custody

- Temporary chats own a private in-memory source binding (`temporary:<chat_id>`,
  fresh incarnation, generation fence). The same permission owner can enable
  Full for it without creating a canonical session, source file or durable grant.
- Discarding the temporary chat retires the ephemeral Full and its device
  consent; another temporary chat's binding cannot resolve the reference.
- Full references can be bound to one `request_turn_id` and are rejected at
  admission for another turn. Persistent ingress derives the same bound proof.
- Same-context steering with a matching active target is admitted into the
  existing pending queue; changed context returns explicit `not_admitted`
  (`steering_context_changed`) so it can be queued as a new request.
- Android custody now retains the eligible pairing id/origin/challenge across
  process death; approved polling clears it and the phone bootstrapper restores
  the active challenge instead of rescanning.

Focused checks:
- `tests/webui/test_temporary_full.py` -> 4 passed (binding isolation, request
  binding, ephemeral loop execution, discard retirement).
- `tests/webui tests/webui/test_permission_v2_api.py tests/security/test_permission_v2_authority.py tests/session/test_p5_stage2_temporary_sessions.py` -> 896 passed.
- Android `:app:assembleDebug` -> BUILD SUCCESSFUL.
- Emulator M01 -> PASS, 37 instrumentation tests.
- Full Python suite, WebUI suite/build/lint and strict comparator run after these
  changes; totals recorded below.

### Final local verification after Temporary Full and custody

- Full Python suite: **17549 passed, 22 skipped, 6 subtests passed**, exit 0
  in 664.19s (`/tmp/opencode/full-suite-r8.log`).
- WebUI: **1415 tests passed**, build exit 0, lint exit 0.
- Strict basedpyright: 0 errors / 528 files; comparator introduced 0.
- Go broker tests: PASS.
- Android debug build: BUILD SUCCESSFUL.
- Emulator M01 (updated APK): PASS, 37 instrumentation tests, native probe
  executed on `emulator-5590`.
- Parity contract check: PASS, 127 rows / 9 categories.

Remaining are external or non-code qualification items: physical-phone
Doze/radio/Tailscale/signed-update journey, signed candidate evidence, and the
M38 independent integrated release review. No installation, merge or push was
performed.

### P1/P2/B/G code-gap closure

- P1: `ProviderRouteChanged` is raised when dispatch selects a leaf different
  from the captured preview; the runner drops the stale selection once and lets
  the selected leaf's final guard re-verify. Added a stale-quote regression.
- P2: a truncated summary is discarded and the same evidence cursor retries the
  next smaller closed chunk within the existing deadline and four-call ceiling.
- B: removed `GatewayHTTPHandler.mobile_mutation_payload` from production;
  admission cases now use a test adapter plus authenticated WS-boundary owner
  denial/success cases.
- G: authenticated mobile WS mutation entry is verified to route into the
  standard dispatcher with operation allowlisting and replay/auth semantics.
- Focused: 132 passed across route/context/mutation selectors.
- Strict: 0/528; comparator introduced 0.

### Code-gap closure verification

- P1 typed dispatch: stale selected-route regression passes; runner clears a
  stale selection once and relies on the selected leaf's final guard.
- P2 bounded retry: truncated summary retries smaller closed chunks; when no
  smaller chunk fits, the original `summary_truncated` reason is preserved.
- B: production `mobile_mutation_payload` removed; authenticated WS-boundary
  owner-denial and standard-dispatcher success cases pass.
- G: standard dispatcher cutover verified at the mobile WS entry with replay/auth.
- Focused selectors: 132 passed, then the affected context/memory selectors
  61 passed after the truncation-reason correction.
- Strict basedpyright: 0/528, comparator introduced 0.
- Final full Python suite after P1/P2/B/G closure: **17552 passed, 22 skipped,
  6 subtests passed**, exit 0 in 721.77s (`/tmp/opencode/full-suite-r10.log`).
- Final strict: 0 errors / 528 files; comparator introduced 0
  (`/tmp/opencode/p6-after-r10.json`).

### Physical phone qualification (in progress)

Device: Nothing A142, Android 16 / API 36, arm64-v8a, Wi-Fi.
Installed candidate: same release signer `CN=jaden`, `adb install -r` preserved
`firstInstallTime`; rollback APK captured at
`/tmp/opencode/phone-readonly/installed-base.apk`.

Observed:
- Cold launch after update reached the reconnect screen; Retry restored the
  existing pairing and chat without rescanning or re-pairing.
- Composer access mode is now visibly labeled (Sandbox/Full) on mobile; the
  screenshot is `/tmp/opencode/phone-evidence/chat-visible-access-3.png`.
- A typed draft survived Home/background and an airplane-mode offline/online
  cycle; airplane mode was restored off.
- A 10+ minute Doze/background draft-retention run is in progress.

WebUI after the visible-label change: 1415 tests passed, build/lint exit 0.

### Physical composer/AC10 evidence (Nothing A142, Android 16)

- Candidate `r5`: `sha256=528f874a20a8491b89cd9223447fca12c5cc09f6527532d86a176838e24d5afb`,
  signer `CN=jaden` / SHA-256
  `8c30d04957e9724d1668e585890913b0581676c9460789f658c4ed9f84ba0c81`,
  `zipalign -c -v 4` and `apksigner verify` pass. The pulled installed APK has
  the same SHA-256 and its bundle hash matches `mobile/www-shared`.
  `adb install -r` preserved `firstInstallTime` (2026-11-02 22:04:34).
- Composer controls measured from the live accessibility tree at the normal
  phone width (density 420): attach, project, voice and send are **118x118 px =
  45.0x45.0 dp**; the Sandbox access control is **176x118 px = 67.0x45.0 dp**.
  All meet the >=44 px requirement.
- Accessible names exposed on the phone: `Attach files`, `Choose project
  folder`, `Sandbox`, `Voice input`, `Send message`.
- The previous 32 px narrow-container shrink rule was removed; the project
  control now has an unconditional 44 px floor and yields its text before the
  permission label.
- A 12-minute Doze/background run retained the composer draft, resumed at
  `Connecting…`, then restored the chat without a reconnect screen. Airplane
  mode remained off and device idle returned to ACTIVE.
- Narrow-width physical measurement was not performed: changing display
  overrides was declined by the user and the overrides were restored
  (`wm size` / `wm density` reset). The 44 px floor is enforced in CSS but only
  the normal-width physical bounds are recorded.

### Next-request mode while a turn is running

- Project and access-mode controls stay usable while streaming. The running turn
  keeps the workspace scope captured at its admission; the selection becomes
  the mode/project carried by the next submitted message.
- Selecting Full during a running restricted turn enables Full for the next
  request without expiring the active turn's task references or escalating its
  captured scope. The mode is persisted for refresh/recovery, and the active
  restricted work continues under its original scope.
- The deferred Full grant can be rolled back while the turn is still running
  and no Full request has been admitted. Once a Full request is admitted, the
  rollback follows the normal `chat_running`/retirement rules.
- Message envelopes are treated as submitted per-request snapshots: a queued
  next message may carry a different project/mode while the current turn runs.
  `set_workspace_scope` remains the idle-only chat mutation.
- Focused coverage: `test_full_selected_during_a_running_turn_applies_to_the_next_request`,
  `test_deferred_full_can_be_rolled_back_before_the_next_request`,
  `test_deferred_full_preserves_an_active_task_reference`, the updated
  `test_webui_message_scope_change_while_running_uses_submitted_snapshot`, and
  the composer test `keeps access and project controls usable while streaming`.
- Focused selector: 488 passed. WebUI: 1419 passed, build/lint exit 0.

### Final code-gate results after next-request mode

- Full Python suite: **17555 passed, 22 skipped, 6 subtests passed**, exit 0
  in 777.66s (`/tmp/opencode/full-suite-r11.log`).
- Strict basedpyright: **0 errors / 528 files**; comparator introduced 0.
- WebUI: **1419 passed**, build exit 0, lint exit 0.
- Installed candidate `r6`:
  `sha256=709914e331ee96662274494f3bb16b5da4b290ffb8b7752c653d48deea66dac8`;
  pulled installed APK matches, signer `CN=jaden`, `zipalign` verifies, and the
  bundled JS hash matches `mobile/www-shared`.
- The phone relaunched into the updated composer after the update and showed
  the project and Sandbox controls once the gateway connection re-established.
- Physical exercise of the next-request Full path still needs the staging
  gateway restarted from this branch; the currently running gateway was started
  from the older checkout before these commits.

### Mobile owner Full projection and physical next-request test

- The mobile thread route now carries the authenticated owner flag into the
  permission projection. Owner devices see `can_enable_full=true` and can use
  the existing owner-authorized `workspace.access.mode.set` path; legacy
  devices keep `controls_unavailable` and cannot promote themselves.
- Physical phone test on the Nothing A142 against a staging gateway started
  from this branch:
  - While a turn was running (`Stop response` visible, `Working for 4s`), the
    access control opened with an enabled `Full access` item.
  - Confirming Full switched the composer label to **Full** while the same turn
    continued, with **For the next message** visible.
  - The session metadata was updated to `workspace_scope.access_mode=full`
    while the active turn continued; the running turn's captured scope was not
    changed.
- Restarting the staging gateway invalidated the phone's existing access token.
  The open WebSocket closed with `1008 access expired`; the in-session composer
  stayed at `Sending…` instead of surfacing the reconnect screen. A cold app
  restart then **Retry** restored the phone without re-pairing. This is recorded
  as a gateway-restart recovery finding; ordinary same-gateway token rotation
  still needs its own physical run.
- Final gates after the owner-projection fix:
  - Full Python suite: **17556 passed, 22 skipped, 6 subtests passed**, exit 0.
  - Strict basedpyright: **0 / 528**, comparator introduced 0.
  - WebUI: 1419 passed, build/lint exit 0.
  - Installed app: candidate `r6` hash/signature verified; staging gateway now
    runs from this worktree.

### Merge corrections (isolated repair branch `fix/mobile-remote-merge-blockers`)

Baseline: `7be433232d28a88106e15a486af3ade2250347f4` (feature handoff plus task docs).
Checker baseline: 0 errors / 528 files (`/tmp/opencode/mc-repair/baseline-basedpyright.json`).

#### MC01 — PASS (focused)

- Embedded WebUI/versioned messages now require `request_settings_version: 1`
  once any v1 field is present; missing/unknown version returns
  `update_required/request_settings_version` before media, transcript or default
  effects.
- v1 intent/target/source/confirmation fields are validated; steering no longer
  skips source validation. Bare legacy WebUI envelopes and non-WebUI messages
  remain compatible; system-command turns remain internal legacy controls.
- Permission-control payloads accept v1 fields and reject stale/unknown
  version/source before any permission effect.
- A content-free source token is exposed in the permission projection, sent by
  the WebUI client, frozen in inbound metadata, and rechecked at source
  execution admission.
- Post-review correction: the router now rechecks the source after transcript
  hydration and again immediately before transcript/dispatch effects. A reset
  during the hydration await returns `source_changed/source_token` with no
  acknowledged user row and no inbound message.
- Requirements-only author: `ses_e5d84f1bdffeYPaeqSMb7tTW1p`
  (`/tmp/opencode/mc01-requirements-author/request-admission-cases.md`).
- Focused: 259 Python tests, 444 WebUI tests, WebUI build/lint; strict 0/528,
  comparator introduced 0.

#### MC02 — PASS (focused)

- Temporary message admission now uses the submitted workspace scope, not the
  live temporary policy scope. Explicit Sandbox remains Sandbox even after
  temporary Full was enabled; explicit Full still resolves through the same
  temporary owner/proof path.
- A differing temporary project is rejected before execution. Temporary
  messages remain connection-owned and use the transient source; canonical
  source-token metadata is not attached to them.
- Added a real channel-boundary test that enables temporary Full, then submits
  one restricted and one Full envelope and observes the effective scope.
- Focused: 55 Python tests; strict 0/528, comparator introduced 0.

#### MC03 — PASS (focused)

- The active WebUI turn now stores its captured workspace scope alongside its
  turn identity.
- A v1 `request_intent: "new"` message arriving while a turn is active is
  rejected `not_admitted/busy_new_request` before media, transcript, or default
  effects. The same submission can be retried after the turn ends.
- Steering validates the exact active turn and the captured active scope, not
  the mutable chat preference. Wrong target or changed project/mode is
  `not_admitted`; source validation still applies.
- Post-review correction: target and scope are rechecked after hydration and
  immediately before transcript/dispatch effects. The loop also consumes
  `webui_request_intent` and `webui_target_turn_id` at bus and pending-injection
  admission, so guidance whose target ended is rejected with
  `not_admitted/steering_target_changed` instead of being injected into a
  successor turn.
- Added public channel-boundary coverage: busy Sandbox B is not admitted while
  Full A runs; matching steer is admitted; changed-context steer is rejected
  without a second published message.
- Focused: 231 Python tests, 281 WebUI tests; strict 0/528, comparator no new
  diagnostics.

#### MC04 — PASS (focused)

- Remote authenticated owner identities now count as permission-control
  availability for Full message admission; untrusted/legacy connections still
  require localhost controls.
- Full messages still resolve the existing installation/source-bound consent
  proof. A second owner installation remains denied until it explicitly
  confirms; confirming while Full is already active records that device's own
  consent.
- Added a two-installation controller proof and a remote owner WS-boundary
  journey: A executes Full, B is denied, B confirms, B then submits a Full
  message carrying a real request proof.
- Focused: 242 Python tests; strict 0/528, comparator no new diagnostics.

#### MC05 — PASS (focused)

- Full retirement now aggregates every request-derived proof for the same
  session/source/epoch. Native stdin handles and process-count callbacks
  registered against request-bound references are closed and discarded with
  the base authority; unrelated sources remain untouched.
- Request-specific authorization and epoch fences are unchanged. Conflicting
  active/unknown counts still block retirement visibly after closure attempts.
- Added a two-request/multiple-handle proof plus an unrelated-source negative;
  native-handle close callbacks are observed, not just reference invalidation.
- Focused: 72 Python tests; strict 0/528, comparator no new diagnostics.

#### MC06 — PASS (focused; native instrumentation pending)

- Android refresh now uses the independent owner route `POST /remote/v1/refresh`
  with `installation_id`/`refresh_token` in the bounded body, not the
  gateway-dependent `/mobile/v1/refresh` GET.
- Native plugin exposes owner status/devices/revoke through the same exact
  broker operations. Shared WebUI remote-access reads and revoke use the owner
  transport when the paired phone is an owner; gateway mutation remains the
  fallback/unsupported Start path.
- Added `/remote/v1/refresh` to the protocol route contract and mobile contract.
- Checks: Kotlin compile, 92 WebUI tests, WebUI build/lint, broker Go tests,
  strict 0/528. The owner-refresh/status/device instrumented journey is added
  but not executed here (no live service shutdown authorized).

#### MC07 — PASS (focused; installed recovery qualification pending)

- Foreground visibility on a closed native socket now refreshes the access
  token without opening a duplicate socket; the owning client performs the
  single reconnect.
- Reconnect reauthentication is bounded to three failures. It then surfaces the
  existing mobile connection notice in `error` state with an explicit Retry
  action instead of staying indefinitely at Sending/Connecting.
- Manual Retry refreshes the URL/token, resets reconnect counters and starts
  one current socket. Draft and pairing state are untouched; no uncertain
  message/mutation is auto-resubmitted. Confirmed device revocation remains the
  native refresh path's distinct logout signal.
- Added client tests for bounded failure and manual retry, plus the notice's
  actionable button. Focused: 185 WebUI tests, build/lint.

#### MC08 — PASS (local candidate)

- Fetched staging `aacca07cfadaf51fa1d83913f93ba2d797c0aa58` and integrated it
  with merge `221101b9a5181839802ea250dd977dd39fab0ffe` without rewriting
  shared history. Staging notifications, image lightbox/scroll and settings
  changes remain present.
- Final gates after the admission-race review: **17,575 passed** Python
  (22 skipped, 6 subtests), strict **0/528**, WebUI **1,432 passed** plus
  build/lint, parity PASS, Go broker PASS, Android release PASS.
- Candidate: `/tmp/opencode/mc-repair-candidate/nanobot-mobile-release.apk`,
  SHA-256 `8e72b5f817d888f85211970cebfdd92b1a3f250bd0d6585908770c951b7b5064`,
  same `CN=jaden` signer, zipalign verified, local-notifications plugin present.
  Rebuilt after the post-review fixes; the prior `4d62c84b…` APK is superseded.
- Bounded review found and the branch corrected: the busy reservation race,
  missing owner auth on status/devices/revoke, source-token parser loss, v1
  control confirmation correlation, second-device UI consent, unbounded
  reconnect, and the post-review source/steering admission races plus queued
  prompt loss. Remaining items are the independent release review and signed
  release sign-off.
- Merge-readiness details: `docs/merge-readiness-mc.md`. No push/merge/deploy.

### MC08 physical candidate evidence (2026-11-04)

- Installed `/tmp/opencode/mc-repair-candidate/nanobot-mobile-release.apk`
  over `ai.nanobot.mobile` with `adb install -r`; `firstInstallTime`
  `2026-11-02 22:04:34` preserved. Pulled installed APK SHA-256
  `4d62c84b308db6f09d5106a2317a275b1c1f83d6696dccd679f6c5bfea3811e6`,
  matching the candidate; signer `CN=jaden` unchanged. Rollback APK:
  `/tmp/opencode/mc-phone-evidence/rollback-installed.apk`
  (`dfc7ee90…`).
- Staging gateway and owner now run from the repair source. The independent
  owner listener is on `127.0.0.1:8900`; Tailscale serve has
  `/remote/v1 → 127.0.0.1:8900` in addition to `/mobile/v1 → 8767`.
- MC06 physical: with the gateway stopped, the phone's Remote access settings
  still rendered owner status and the device list through the owner route,
  with the connection-error notice visible. Revoke buttons were present; an
  actual phone revoke was not executed (would remove a live device).
- MC07 physical: after a gateway restart, the app showed **Connection error**
  with **Retry** instead of staying at Sending; the draft
  `restart draft check` remained. Manual Retry reconnected, the draft remained,
  and sending it produced exactly one gateway turn
  (`Processing message … restart draft check`). The app returned to a clean
  connected composer after the final gateway restart.
- Evidence: `/tmp/opencode/mc-phone-evidence/` (`install.log`,
  `package-before/after.txt`, `after-restart.xml`, `after-manual-retry2.xml`,
  `remote-offline.xml`, `installed-final.apk`, gateway/owner logs).

Physical/instrumented coverage now includes MC04 two-installation owner
confirmation and revoke, MC06 owner status/list/revoke, MC07 restart/draft
recovery, MC05 real native closure, and the M01 emulator rerun. Remaining:
M38/signed-release review and applicable radio/Tailscale qualification.

### MC04/MC06 authenticated A/B and emulator evidence

- Created two disposable owner installations through the live owner IPC:
  `mc-test-a` and `mc-test-b`, both with owner/chat/media scopes.
- A created a real chat through `/mobile/v1/ws`, prepared Full with the v1
  source token and confirmation ID, and enabled Full successfully.
- B, on a separate authenticated mobile WS, submitted Full:
  - before confirmation: `workspace_scope_rejected/full_access_unavailable`;
  - after its own prepare/confirm: `message_accepted` and a real Full turn;
  - after A revoked `mc-test-b` through the gateway revoke mutation, B's next
    frame closed with `1008 access expired` and was not admitted.
- A then revoked `mc-test-a`; both disposable installations are gone.
- The same gateway revoke path is what the phone Remote access UI calls when
  the gateway is reachable, so its in-process Full consent callback runs.
- Emulator M01 rerun: `M01_RESULT=PASS`, 38 instrumentation tests.
- Evidence: `/tmp/opencode/mc-phone-evidence/ab-journey.log`,
  `a-token.json`, `b-token.json`, `/tmp/opencode/mc-emulator-b.log`.
- Remaining qualification: M38/signed-release review. The literal two-phone UI
  confirmation was not repeated; the authenticated A/B installation journey
  covers the public protocol.

### MC05 real native closure evidence

- Ran `/tmp/opencode/mc05_native_closure.py` on the Linux host with the real
  `ExecTool` native supervisor, a request-derived Full reference
  (`request_turn_id=turn-native`), and a live `cat` process.
- Before revocation the owner held the derived native registration; after
  `disable_full`, the derived stdin/registration was retired and the base Full
  proof was removed.
- Result: `MC05_NATIVE_PASS base_handles=1 derived_retired=true session=ef7a086ade9c`
  (`/tmp/opencode/mc-phone-evidence/mc05-native.log`).
- This is real process closure, not only synthetic callback aggregation.

### Post-review admission-race corrections (2026-11-04)

The bounded review (`CORRECT — hold staging merge`) found three remaining
blockers; code/test correction commit `45c97b18`; final wire-coverage test commit `9588d899`.

- MC01 source reset: the router rechecks the source token after hydration and
  again immediately before transcript/dispatch effects. The independently
  authored hydration race now returns `source_changed/source_token` with zero
  user transcript rows and zero inbound messages; the old `c0b0bf8d` code
  emitted `message_accepted`, one row and one inbound message.
- MC03 steering target: target and captured scope are rechecked after hydration
  and immediately before effects. A second check runs after metadata/mention
  preparation. The loop now reads `webui_request_intent` and
  `webui_target_turn_id` at bus admission and pending injection, emits
  `not_admitted/steering_target_changed`, and never injects guidance into the
  successor turn.
- MC03 queued prompt retention: the WebUI composer keeps the full queued
  snapshot (text, images, quote, mentions, workspace scope) when a correlated
  `busy_new_request` or `steering_target_changed` rejection arrives, holds it
  out of automatic idle flushes, and retries only on an explicit Guide/click.
  The failed optimistic row is removed instead of leaving a duplicate bubble.
- Requirements-only author: session `ses_e5b4f2adbffepCXrhG5pQ63D2c`;
  scenarios and skeletons under `/tmp/opencode/mc-admission-cases/`. Cases were
  mechanically adapted to the real test harness without changing expected
  outcomes; no repository source, existing tests or parent history were given
  to the author.
- New boundary tests:
  - `test_webui_message_source_reset_during_hydration_is_rejected_before_effects`
  - `test_webui_steer_rejects_target_replaced_during_hydration_before_effects`
  - `test_webui_steering_target_change_is_rejected_before_bus_dispatch`
  - `test_webui_steering_matching_target_still_dispatches`
  - `test_stale_webui_steering_is_rejected_at_pending_injection`
  - `test_send_admission_rejected_emits_correlated_error_frame`
  - `restores the full queued snapshot and waits for an explicit retry`
- Defect detection: the two hydration tests fail on `c0b0bf8d`
  (`/tmp/opencode/mc-admission-cases/before-fail.log`); the composer test fails
  without the snapshot/requeue path.
- Evidence:
  - Full Python: **17,575 passed, 22 skipped, 6 subtests**, exit 0
    (`/tmp/opencode/mc-full-pytest-final.log`).
  - WebUI: **1,432 passed**, build/lint exit 0
    (`/tmp/opencode/mc-webui-tests.log`,
    `/tmp/opencode/mc-webui-lint.log`).
  - Strict basedpyright: before `c0b0bf8d` 0/528, after 0/528; comparator
    `PASS_NO_NEW_DIAGNOSTICS`, introduced 0
    (`/tmp/opencode/mc-type/{before,after-final,comparator-final}.json`).
- Candidate rebuilt after the correction:
  `/tmp/opencode/mc-repair-candidate/nanobot-mobile-release.apk`,
  SHA-256 `8e72b5f817d888f85211970cebfdd92b1a3f250bd0d6585908770c951b7b5064`,
  `CN=jaden`, zipalign/local-notifications verified. The phone was not attached
  (`adb devices` empty), so the superseded `4d62c84b…` installation was not
  replaced.
